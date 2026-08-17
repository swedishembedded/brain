// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side decode of the NPU's raw head tensors → detections. Reuses the
//! shared yolo post-processing (`repack_heads_to_flat`, `dfl_decode_cpu`,
//! `decode_detections`) so the NPU path produces byte-identical boxes to the
//! engine path. No GPU / no `Yolo` instance needed.

use yolov8::assign::Anchor;
use yolov8::boxmath::Letterbox;
use yolov8::head::repack_heads_to_flat;
use yolov8::infer::{decode_detections, dfl_decode_cpu};
use yolov8::nms::Detection;
use yolov8::YoloConfig;

use crate::openvino::{HeadOutputs, NpuConfig, NpuError, NpuSession};

/// One-call detection on the NPU straight from a `.safetensors` checkpoint: export an
/// fp32 ONNX in memory, compile it for the device, run, and decode. This backs
/// `brain yolo detect --device npu` (the convenience fp32 path). For INT8, export
/// + quantize first, then `brain npu run` on the quantized ONNX.
#[allow(clippy::too_many_arguments)]
pub fn detect_weights_on_npu(
    weights_path: &str,
    hwc: &[f32],
    w0: u32,
    h0: u32,
    conf: f32,
    iou: f32,
    npu_cfg: &NpuConfig,
    input: Option<u32>,
) -> Result<Vec<Detection>, NpuError> {
    let mut cfg = crate::export::config_of(weights_path);
    if let Some(s) = input {
        cfg.input = s;
    }
    let bytes = crate::export::build_fp32_bytes(weights_path, input, onnx::DEFAULT_OPSET);
    let mut session = NpuSession::load_bytes(&bytes, npu_cfg)?;
    detect_image(&mut session, hwc, w0, h0, &cfg, conf, iou)
}

/// End-to-end: letterbox an HWC-RGB image, run it on the NPU session, and decode
/// to detections. The one-call path used by `brain npu run` and
/// `brain yolo detect --device npu`.
pub fn detect_image(
    session: &mut NpuSession,
    hwc: &[f32],
    w0: u32,
    h0: u32,
    cfg: &YoloConfig,
    conf: f32,
    iou: f32,
) -> Result<Vec<Detection>, NpuError> {
    let (chw, lb) = yolov8::boxmath::letterbox_rgb(hwc, w0, h0, cfg.input, 114.0 / 255.0);
    let shape = [1usize, 3, cfg.input as usize, cfg.input as usize];
    let heads = session.run(&chw, shape)?;
    Ok(decode_npu_outputs(&heads, cfg, &lb, w0, h0, conf, iou))
}

/// Decode the 6 raw head tensors (`head.{s}.cls` / `head.{s}.reg`, NCHW) produced
/// by OpenVINO into detections in original-image pixel coords.
pub fn decode_npu_outputs(
    heads: &HeadOutputs,
    cfg: &YoloConfig,
    lb: &Letterbox,
    w0: u32,
    h0: u32,
    conf: f32,
    iou: f32,
) -> Vec<Detection> {
    let nc = cfg.nc as usize;
    let reg_max = cfg.reg_max as usize;
    let four_rm = 4 * reg_max;

    // Per-scale cls/reg tensors in scale order 0,1,2 (strides 8/16/32).
    let cls: Vec<(Vec<f32>, u32, u32)> = (0..3).map(|s| find_output(heads, &format!("head.{s}.cls"))).collect();
    let reg: Vec<(Vec<f32>, u32, u32)> = (0..3).map(|s| find_output(heads, &format!("head.{s}.reg"))).collect();

    let a: usize = cls.iter().map(|(_, h, w)| (h * w) as usize).sum();
    let cls_refs: Vec<(&[f32], u32, u32)> = cls.iter().map(|(d, h, w)| (d.as_slice(), *h, *w)).collect();
    let reg_refs: Vec<(&[f32], u32, u32)> = reg.iter().map(|(d, h, w)| (d.as_slice(), *h, *w)).collect();
    let cls_flat = repack_heads_to_flat(&cls_refs, 1, nc, a);
    let box_flat = repack_heads_to_flat(&reg_refs, 1, four_rm, a);
    let dist = dfl_decode_cpu(&box_flat, a, reg_max);

    // Anchor geometry, scale-major then row-major — matching the repack order.
    let mut anchors = Vec::with_capacity(a);
    for (s, (_, h, w)) in cls.iter().enumerate() {
        let stride = cfg.strides[s] as f32;
        for yy in 0..*h {
            for xx in 0..*w {
                let (ax, ay) = (xx as f32 + 0.5, yy as f32 + 0.5);
                anchors.push(Anchor { cx: ax * stride, cy: ay * stride, ax, ay, stride });
            }
        }
    }

    decode_detections(&cls_flat, &dist, &anchors, 1, a, nc, &[*lb], &[(w0, h0)], conf, iou, 300)
        .pop()
        .unwrap_or_default()
}

/// Look up a named NCHW output and return `(data, H, W)`.
fn find_output(heads: &HeadOutputs, name: &str) -> (Vec<f32>, u32, u32) {
    let (_, shape, data) = heads
        .tensors
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("NPU output `{name}` not found (have: {:?})", heads.tensors.iter().map(|(n, _, _)| n).collect::<Vec<_>>()));
    assert!(shape.len() >= 4, "output `{name}` is not NCHW: shape {shape:?}");
    (data.clone(), shape[2] as u32, shape[3] as u32)
}
