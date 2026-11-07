// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The official NVFP4 release of Ripple-65B on real tensors: layers 0 (sliding) and 4 (global) of
//! `ricursiveAI/Ripple-65B-Preview-NVFP4`, with its embedding, final norm and `lm_head`, fetched by HTTP range
//! reads (`~/resources/tools/hf_fetch_tensors.py`, 3.4 GB) beside the bf16 partial checkpoint they were
//! quantized from.
//!
//! Swedish Embedded AB implements solutions for serving released quantized language models exactly as published,
//! with their loss against the full-precision model measured on the real tensors, for its clients. If your team
//! needs expertise in NVFP4 and ModelOpt checkpoints then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Three questions:
//!
//! 1. **Do the files decode as the format defines?** An independent decode of the stored bytes (E2M1 code x E4M3
//!    block scale x f32 `weight_scale_2`, the ModelOpt NVFP4 semantics) against what the device holds.
//! 2. **What does NVFP4 cost?** Layers 0 and 4 run on the NVFP4 files and on the bf16 files they came from over
//!    115 tokens of text: the hidden stream's relative error per layer, and the KL divergence and top-1
//!    agreement of the next-token distribution at the end of the two-layer stack.
//! 3. **What does the static input scale say?** The release is W4A4 (activations quantized to FP4 with a stored
//!    per-tensor scale) and brain holds the activations in float. The range each scale was calibrated for is
//!    compared with the activations the MLP actually saw.
//!
//! Skips by name when the files are absent (`testdata/ricursive/ripple-65b-partial/`).

use std::os::unix::fs::FileExt;
use std::path::Path;

use checkpoint::packed::{BoundedWorkspace, MemoryBudget};
use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use model::ops::{upload_from_source, Ops};
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use ricursive::{open_safetensors, resolve_tiers, RicursiveConfig, RicursiveEngine, RicursiveModel, ServeConfig};

use crate::taps::scaled_error;

const SOURCE_LAYERS: [usize; 2] = [0, 4];
const BF16: &str = "ricursive/ripple-65b-partial/layers-0-4.safetensors";
const NVFP4: &str = "ricursive/ripple-65b-partial/layers-0-4-nvfp4.safetensors";
const CONFIG: &str = "ricursive/ripple-65b-partial/config.json";

/// A fixed English-and-code paragraph tokenized with the release's own vocabulary (greedy longest match over the
/// byte-level BPE vocabulary: valid tokens of a realistic stream, not the canonical merge order).
const TEXT: [u32; 115] = [
    791, 34932, 3321, 13912, 1193, 279, 1455, 3293, 7039, 315, 1855, 8668, 11, 1418, 279, 3728, 13931, 1373, 1475, 21224, 4037, 13, 32541, 4954, 279, 14661, 315, 279, 5510, 45904, 10215, 311, 3116, 9660, 75212, 279, 5044, 1578, 11, 323, 279, 52662,
    1493, 315, 1855, 2565, 374, 62215, 555, 1202, 5569, 13, 763, 34889, 11, 264, 734, 430, 4780, 264, 5832, 2011, 3790, 279, 1493, 1162, 21650, 25, 5279, 4820, 1161, 25, 612, 496, 8, 1492, 5832, 35937, 843, 11, 935, 29, 314, 274, 4736, 28738, 84, 843,
    10703, 2235, 9450, 23236, 68, 91, 384, 2446, 3991, 2189, 335, 362, 25569, 449, 264, 907, 33247, 63526, 279, 1176, 6324, 1988, 1139, 1475, 907, 22343, 13,
];

struct Files {
    bf16: String,
    nvfp4: String,
    cfg: RicursiveConfig,
}

fn files() -> Option<Files> {
    let (bf16, nvfp4, config) = (brain_testutil::testdata(BF16), brain_testutil::testdata(NVFP4), brain_testutil::testdata(CONFIG));
    for path in [&bf16, &nvfp4, &config] {
        if !Path::new(path).exists() {
            brain_testutil::skip(&format!("{path} absent: place the partial Ripple-65B checkpoints (bf16 and NVFP4: embed, norm, lm_head, layers 0 and 4) under testdata/ricursive/ripple-65b-partial/"));
            return None;
        }
    }
    let cfg = RicursiveConfig::from_file(&config).expect("config").truncated(&SOURCE_LAYERS).expect("truncated config");
    Some(Files { bf16, nvfp4, cfg })
}

/// A safetensors file's tensor table, read directly (no brain code): `(name -> (dtype, shape, absolute offset, length))`.
struct Raw {
    file: std::fs::File,
    table: serde_json::Map<String, serde_json::Value>,
    base: u64,
}

impl Raw {
    fn open(path: &str) -> Raw {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut len = [0u8; 8];
        file.read_exact_at(&mut len, 0).unwrap();
        let n = u64::from_le_bytes(len);
        let mut header = vec![0u8; n as usize];
        file.read_exact_at(&mut header, 8).unwrap();
        let table = serde_json::from_slice::<serde_json::Value>(&header).unwrap().as_object().unwrap().clone();
        Raw { file, table, base: 8 + n }
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.table[name]["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
    }

    /// `len` bytes of tensor `name` from byte `from` of its data.
    fn bytes(&self, name: &str, from: usize, len: usize) -> Vec<u8> {
        let offsets = self.table[name]["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap();
        assert!(from + len <= (offsets[1].as_u64().unwrap() - start) as usize, "{name}: read past the tensor");
        let mut out = vec![0u8; len];
        self.file.read_exact_at(&mut out, self.base + start + from as u64).unwrap();
        out
    }

    fn scalar(&self, name: &str) -> f32 {
        f32::from_le_bytes(self.bytes(name, 0, 4).try_into().unwrap())
    }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// OCP E4M3 (finite, bias 7): sign, 4 exponent bits, 3 mantissa bits.
fn e4m3(byte: u8) -> f32 {
    let (e, m) = (i32::from(byte >> 3 & 15), f32::from(byte & 7));
    let v = if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if byte & 0x80 != 0 { -v } else { v }
}

/// Row `r` of a ModelOpt NVFP4 tensor `module` (`weight` U8 `[n, k/2]` with the even element in the low nibble,
/// `weight_scale` E4M3 `[n, k/16]`, `weight_scale_2` f32): `E2M1(code) x scale_2 x E4M3(block scale)`.
fn reference_row(raw: &Raw, module: &str, r: usize) -> Vec<f32> {
    let k = raw.shape(&format!("{module}.weight"))[1] * 2;
    let codes = raw.bytes(&format!("{module}.weight"), r * k / 2, k / 2);
    let scales = raw.bytes(&format!("{module}.weight_scale"), r * k / 16, k / 16);
    let global = raw.scalar(&format!("{module}.weight_scale_2"));
    (0..k)
        .map(|c| {
            let nibble = if c % 2 == 0 { codes[c / 2] & 15 } else { codes[c / 2] >> 4 };
            let v = E2M1[(nibble & 7) as usize] * if nibble & 8 != 0 { -1.0 } else { 1.0 };
            v * (e4m3(scales[c / 16]) * global)
        })
        .collect()
}

fn ops(cfg: &RicursiveConfig) -> Ops {
    Ops::new(gpu_core::Gpu::open(None, &ricursive::pipelines(cfg, &[]))).expect("ops")
}

/// Spec: the device holds the MLP exactly as the ModelOpt format defines, checked row by row (the first, the last
/// and nine spread between) against a decode that shares no code with brain's.
#[test]
fn the_real_nvfp4_tensors_decode_as_the_modelopt_format_defines() {
    let Some(f) = files() else { return };
    let raw = Raw::open(&brain_testutil::testdata(NVFP4));
    let src = open_safetensors(&f.nvfp4, &f.cfg, &SOURCE_LAYERS).expect("open the release");
    let ops = ops(&f.cfg);
    let staging = BoundedWorkspace::new(MemoryBudget::new(64 << 20, 0));
    let mut worst = 0.0f32;
    for (canonical, file_layer) in [(0usize, 0usize), (1, 4)] {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let module = format!("model.layers.{file_layer}.mlp.{proj}");
            let (n, k) = (raw.shape(&format!("{module}.weight"))[0], raw.shape(&format!("{module}.weight"))[1] * 2);
            let name = format!("model.layers.{canonical}.mlp.{proj}.weight");
            assert_eq!(model::ops::stored_tier(&src, &name), Some(Dtype::F4E2M1), "{name} is stored as NVFP4");
            let w = upload_from_source(&ops, &src, &name, n, k, Dtype::F4E2M1, &staging).expect("tensor");
            assert!(matches!(w, model::ops::Weight::Micro { kind: model::ops::MicroScaled::Nvfp4, .. }));
            let decoded = ops.read_host(&w);
            for r in [0, 1, n / 7, n / 3, n / 2, n / 2 + 1, 2 * n / 3, 5 * n / 6, n - 2, n - 1] {
                let want = reference_row(&raw, &module, r);
                let got = &decoded[r * k..(r + 1) * k];
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                let err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max) / scale.max(f32::MIN_POSITIVE);
                assert!(err <= 1e-6, "{module} row {r}: device decode differs from the format's by {err:.3e} of the row's range");
                worst = worst.max(err);
            }
        }
    }
    eprintln!("NVFP4 tensors: device decode vs the independent ModelOpt decode over 60 rows: worst error {worst:.3e} of the row range");
}

fn rms_norm(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let h = gain.len();
    x.chunks(h)
        .flat_map(|row| {
            let r = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / h as f32 + eps).sqrt();
            row.iter().zip(gain).map(|(v, g)| v * r * g).collect::<Vec<_>>()
        })
        .collect()
}

fn softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = z.iter().map(|&v| f64::from(v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn argmax(z: &[f32]) -> usize {
    z.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den).sqrt()
}

fn forward(f: &Files, path: &str) -> ricursive::Forward {
    let src = open_safetensors(path, &f.cfg, &SOURCE_LAYERS).expect("open");
    let tiers = resolve_tiers(&f.cfg, &src, None);
    let started = std::time::Instant::now();
    let out = RicursiveModel::new(f.cfg.clone(), &src, None).with_tiers(tiers).forward(&TEXT).expect("forward");
    eprintln!("forward over {} tokens in {:.1}s ({path})", TEXT.len(), started.elapsed().as_secs_f64());
    out
}

/// Spec: NVFP4 weights in the MLP change the model by 4-bit rounding and nothing else. The measured loss against
/// the bf16 checkpoint they were quantized from is recorded in the roadmap; the bounds are those measurements with margin.
/// Also checks the static input scales against the activations the MLP saw (question 3).
#[test]
fn the_real_nvfp4_layers_stay_close_to_the_bf16_layers_and_the_input_scales_cover_the_activations() {
    let Some(f) = files() else { return };
    let (bf, nv) = (forward(&f, &f.bf16), forward(&f, &f.nvfp4));
    let vocab = f.cfg.vocab_size;

    let rel: Vec<f64> = nv.layer_outputs.iter().zip(&bf.layer_outputs).map(|(a, b)| rel_l2(a, b)).collect();
    let (mut kl, mut same, mut kl_last) = (0.0f64, 0usize, 0.0f64);
    let t = TEXT.len();
    for pos in 0..t {
        let (p, q) = (softmax(&bf.logits[pos * vocab..(pos + 1) * vocab]), softmax(&nv.logits[pos * vocab..(pos + 1) * vocab]));
        let d: f64 = p.iter().zip(&q).map(|(p, q)| if *p > 0.0 { p * (p / q.max(1e-300)).ln() } else { 0.0 }).sum();
        kl += d;
        if pos == t - 1 {
            kl_last = d;
        }
        same += usize::from(argmax(&bf.logits[pos * vocab..(pos + 1) * vocab]) == argmax(&nv.logits[pos * vocab..(pos + 1) * vocab]));
    }
    let (kl, top1) = (kl / t as f64, same as f64 / t as f64);
    let worst = scaled_error("logits nvfp4 vs bf16", &nv.logits, &bf.logits);
    eprintln!(
        "NVFP4 vs bf16, layers 0 and 4 over {t} tokens: hidden relative L2 error per layer {:.3e} / {:.3e}; logits mean KL {kl:.3e} (last position {kl_last:.3e}), top-1 agreement {:.1}% ({same}/{t}); worst scaled logits error {worst:.3e}",
        rel[0],
        rel[1],
        top1 * 100.0
    );
    assert!(rel.iter().all(|r| *r < 0.1) && kl < 0.1 && top1 >= 0.8, "NVFP4 diverged from bf16: rel {rel:?}, KL {kl:.3e}, top-1 {top1}");

    // Question 3: the range each MLP input scale was calibrated for (`input_scale x 6 x 448` is the amax a
    // per-tensor FP4 activation quantizer covers) against the largest activation the gate/up input took here.
    let raw = Raw::open(&f.nvfp4);
    let src = open_safetensors(&f.bf16, &f.cfg, &SOURCE_LAYERS).expect("open");
    let read = |name: &str| {
        let mut v = Vec::new();
        src.with_tensor(name, &mut |d| v = d.to_vec());
        v
    };
    let eps = f.cfg.rms_norm_eps;
    let mut x_in = bf.embeddings.clone();
    for (layer, file_layer) in SOURCE_LAYERS.iter().copied().enumerate() {
        let post_attn = rms_norm(&bf.attn_outputs[layer], &read(&format!("model.layers.{layer}.post_attention_layernorm.weight")), eps);
        let x1: Vec<f32> = x_in.iter().zip(&post_attn).map(|(a, b)| a + b).collect();
        let m = rms_norm(&x1, &read(&format!("model.layers.{layer}.pre_feedforward_layernorm.weight")), eps);
        let amax = m.iter().fold(0f32, |a, v| a.max(v.abs()));
        let rms = (m.iter().map(|v| v * v).sum::<f32>() / m.len() as f32).sqrt();
        let (gate, up) = (raw.scalar(&format!("model.layers.{file_layer}.mlp.gate_proj.input_scale")), raw.scalar(&format!("model.layers.{file_layer}.mlp.up_proj.input_scale")));
        assert_eq!(gate, up, "gate and up read the same activation and share its scale");
        let covered = gate * 6.0 * 448.0;
        eprintln!("layer {file_layer} MLP input: observed amax {amax:.3} (rms {rms:.3}) over {t} tokens; input_scale {gate:.3e} covers amax {covered:.3} ({:.0}% of the calibrated range used)", 100.0 * amax / covered);
        assert!(gate.is_finite() && gate > 0.0);
        x_in = bf.layer_outputs[layer].clone();
    }
}

/// Spec: the serving engine on the NVFP4 files decodes what the prefill model computes from them (the layer-level
/// forward is the engine's reference), through the paged global layer and the ring, with the MLP resident as
/// NVFP4 and the attention and `lm_head` as stored (bf16).
#[test]
fn the_serving_engine_holds_the_nvfp4_release_as_stored_and_decodes_as_the_prefill_model() {
    let Some(f) = files() else { return };
    let src = open_safetensors(&f.nvfp4, &f.cfg, &SOURCE_LAYERS).expect("open");
    let tiers = resolve_tiers(&f.cfg, &src, None);
    let ids = &TEXT[..13];
    let want = RicursiveModel::new(f.cfg.clone(), &src, None).with_tiers(tiers).forward(ids).expect("forward").logits;
    let vocab = f.cfg.vocab_size;

    let mut engine = RicursiveEngine::from_source(f.cfg.clone(), Box::new(open_safetensors(&f.nvfp4, &f.cfg, &SOURCE_LAYERS).unwrap()), ServeConfig::new(16, 4, 32, 2, 0, 4), None).expect("engine");
    let report = engine.weight_report().clone();
    eprintln!("resident weights: {}", report.describe());
    assert_eq!(report.count(Dtype::F4E2M1), 6, "three MLP linears in each of the two layers");
    assert_eq!(report.count(Dtype::BF16), 2 * 5 + 1, "five attention linears a layer and lm_head");
    assert_eq!(report.count(Dtype::F32), 0, "nothing is widened to f32");

    let (prompt, rest) = ids.split_at(9);
    let mut table = BlockTable::new();
    let hidden = loop {
        if let model::serve::PrefillProgress::Done(h) = engine.prefill_chunk(&mut table, prompt, 4) {
            break h;
        }
    };
    let mut worst = scaled_error("prefill, position 8", &engine.logits(&hidden), &want[8 * vocab..9 * vocab]);
    for (i, &token) in rest.iter().enumerate() {
        let pos = 9 + i;
        let logits = engine.decode_logits(&mut [&mut table], &[token]);
        worst = worst.max(scaled_error(&format!("decode, position {pos}"), &logits[0], &want[pos * vocab..(pos + 1) * vocab]));
    }
    eprintln!("NVFP4 engine vs the prefill model on the same files: worst scaled error {worst:.3e}");
    assert!(worst <= 5e-5, "{worst:.3e}");
}
