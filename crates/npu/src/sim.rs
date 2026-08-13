// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hardware-free fake-quant simulation: reproduce the INT8 graph's arithmetic in
//! fp32 inside brain's own engine, so INT8 accuracy can be measured (head-tensor
//! fidelity and mAP) on `--device cpu` with NO NPU and NO OpenVINO.
//!
//! The simulation mirrors the exported INT8 graph exactly:
//!   * weights: each conv weight is replaced by `dequant(quant(fold_bn(w)))`
//!     (per-output-channel), and BN is set to an identity-with-bias so the engine
//!     computes `Conv(fq_weight) + bias` — the same as the ONNX `Conv`;
//!   * activations: a [`FakeQuantTap`] applies `dequant(quant(x))` (per-tensor) to
//!     each conv input, the Q/DQ pair the graph inserts.

use std::collections::HashMap;

use yolov8::net::ActTap;
use yolov8::Yolo;

use crate::fold::{fold_bn, quantize_weight_per_channel, BN_EPS};
use crate::quant::Quant;

/// Per-tensor symmetric activation fake-quant (quant→dequant in place).
pub struct FakeQuantTap {
    scales: HashMap<String, f32>,
}

impl FakeQuantTap {
    pub fn new(q: &Quant) -> FakeQuantTap {
        FakeQuantTap { scales: q.act_scales.clone() }
    }
}

impl ActTap for FakeQuantTap {
    fn tap(&self, name: &str, x: &mut [f32]) {
        if let Some(&s) = self.scales.get(name) {
            for v in x.iter_mut() {
                let q = (*v / s).round().clamp(-127.0, 127.0);
                *v = q * s;
            }
        }
    }
}

/// Rewrite every conv in `model` so the engine computes `Conv(fq_folded_weight) +
/// bias + SiLU`, matching the exported INT8 graph. BN is set to identity-with-bias
/// (`gamma=1, beta=0, run_var=1−eps, run_mean=−bias` ⇒ eval scale 1, eval bias
/// `bias`). Mutates the model in place (caller uses a throwaway model).
pub fn install_fake_quant_weights(model: &Yolo) {
    let plist = model.cfg.full_param_list();
    for (name, _) in plist {
        let pfx = match name.strip_suffix(".conv.weight") {
            Some(p) => p.to_string(),
            None => continue,
        };
        let w = model.read_weight(&format!("{pfx}.conv.weight"));
        let gamma = model.read_weight(&format!("{pfx}.bn.gamma"));
        let beta = model.read_weight(&format!("{pfx}.bn.beta"));
        let rm = model.read_weight(&format!("{pfx}.bn.run_mean"));
        let rv = model.read_weight(&format!("{pfx}.bn.run_var"));
        let cout = gamma.len();
        let (wp, bias) = fold_bn(&w, &gamma, &beta, &rm, &rv, cout);
        let per = wp.len() / cout;
        let (wq, sc) = quantize_weight_per_channel(&wp, cout, per);
        let mut fqw = vec![0.0f32; wp.len()];
        for o in 0..cout {
            for i in 0..per {
                fqw[o * per + i] = wq[o * per + i] as f32 * sc[o];
            }
        }
        model.write_weight(&format!("{pfx}.conv.weight"), &fqw);
        model.write_weight(&format!("{pfx}.bn.gamma"), &vec![1.0f32; cout]);
        model.write_weight(&format!("{pfx}.bn.beta"), &vec![0.0f32; cout]);
        model.write_weight(&format!("{pfx}.bn.run_var"), &vec![1.0 - BN_EPS; cout]);
        let neg_bias: Vec<f32> = bias.iter().map(|b| -b).collect();
        model.write_weight(&format!("{pfx}.bn.run_mean"), &neg_bias);
    }
    // The BN-eval collapse is recomputed lazily on the first eval forward; the
    // model has not run forward yet here, so the rewritten params take effect.
}

/// fp32 reference raw logits `(cls [A,nc], box [A,4*reg_max])` for one CHW input.
pub fn reference_logits(weights_path: &str, chw: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let model = Yolo::load(weights_path, 1);
    model.set_eval(true);
    model.set_image(chw);
    model.forward_net_pub();
    model.raw_logits()
}

/// INT8-simulated raw logits for one CHW input (fake-quant weights + activations).
pub fn simulate_logits(weights_path: &str, chw: &[f32], quant: &Quant) -> (Vec<f32>, Vec<f32>) {
    let model = Yolo::load(weights_path, 1);
    install_fake_quant_weights(&model);
    model.set_eval(true);
    model.set_image(chw);
    let tap = FakeQuantTap::new(quant);
    model.forward_net_tapped(&tap);
    model.raw_logits()
}

/// Cosine similarity between two equal-length vectors (1.0 = identical
/// direction) — the shared `model::hostmath::cosine`, re-exported for this
/// module's existing callers. NOTE one behavioural fix vs the old local
/// copy: a zero vector now scores 0.0 (no direction), not a vacuous 1.0.
pub use model::hostmath::cosine;

/// Mean-absolute fp32-vs-INT8-sim mAP@0.5 over a brain detection dataset. Returns
/// `(map_fp32, map_int8_sim)`. Reuses the shared decode tail + `eval::detection`.
pub fn simulate_map(weights_path: &str, dataset_dir: &str, quant: &Quant, conf: f32, iou: f32) -> (f32, f32) {
    use yolov8::boxmath::{letterbox_rgb, xywhn_to_xyxy};

    let ds = data::gen_detect::load_dataset(std::path::Path::new(dataset_dir)).expect("load dataset");
    let cfg = checkpoint_cfg(weights_path);
    let nc = cfg.nc;
    let input = cfg.input;
    let stride = ds.image_stride();

    // Reference (fp32) and INT8-sim models, built once.
    let ref_model = Yolo::load(weights_path, 1);
    ref_model.set_eval(true);
    let sim_model = Yolo::load(weights_path, 1);
    install_fake_quant_weights(&sim_model);
    sim_model.set_eval(true);
    let tap = FakeQuantTap::new(quant);

    let mut preds_fp32 = Vec::new();
    let mut preds_int8 = Vec::new();
    let mut gts = Vec::new();
    for i in 0..ds.n {
        let chw = &ds.images[i * stride..(i + 1) * stride];
        let hwc = imaging::pixels::chw_to_hwc(chw, 3, ds.h as usize, ds.w as usize);
        let (lbchw, lb) = letterbox_rgb(&hwc, ds.w, ds.h, input, 114.0 / 255.0);

        let f = decode_one(&ref_model, &lbchw, &lb, ds.w, ds.h, &cfg, conf, iou, None);
        let q = decode_one(&sim_model, &lbchw, &lb, ds.w, ds.h, &cfg, conf, iou, Some(&tap));
        // mAP matching is global, so shift each image's boxes into a disjoint x
        // strip — this keeps matching strictly within-image without per-image
        // bookkeeping. (Same trick the yolo eval CLI uses.)
        let off = i as f32 * (ds.w as f32 + 16.0);
        push_dets(&mut preds_fp32, off, &f);
        push_dets(&mut preds_int8, off, &q);
        for b in &ds.boxes[i] {
            let mut bx = xywhn_to_xyxy(b.cx, b.cy, b.w, b.h, ds.w as f32);
            bx[0] += off;
            bx[2] += off;
            gts.push(eval::detection::GtBox { class: b.class, bbox: bx });
        }
    }
    let m_fp32 = eval::detection::map50(&preds_fp32, &gts, nc);
    let m_int8 = eval::detection::map50(&preds_int8, &gts, nc);
    (m_fp32, m_int8)
}

fn push_dets(out: &mut Vec<yolov8::nms::Detection>, x_off: f32, dets: &[yolov8::nms::Detection]) {
    for d in dets {
        out.push([d[0] + x_off, d[1], d[2] + x_off, d[3], d[4], d[5]]);
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_one(
    model: &Yolo,
    chw: &[f32],
    lb: &yolov8::boxmath::Letterbox,
    w0: u32,
    h0: u32,
    cfg: &yolov8::YoloConfig,
    conf: f32,
    iou: f32,
    tap: Option<&FakeQuantTap>,
) -> Vec<yolov8::nms::Detection> {
    model.set_image(chw);
    match tap {
        Some(t) => model.forward_net_tapped(t),
        None => model.forward_net_pub(),
    }
    let (cls, boxl) = model.raw_logits();
    let a = model.num_anchors() as usize;
    let dist = yolov8::infer::dfl_decode_cpu(&boxl, a, cfg.reg_max as usize);
    let anchors = model.anchor_geometry();
    let mut out = yolov8::infer::decode_detections(
        &cls, &dist, &anchors, 1, a, cfg.nc as usize, &[*lb], &[(w0, h0)], conf, iou, 300,
    );
    out.pop().unwrap_or_default()
}

fn checkpoint_cfg(weights_path: &str) -> yolov8::YoloConfig {
    let c = checkpoint::load(weights_path);
    yolov8::YoloConfig::from_json(&c.header["config"])
}
