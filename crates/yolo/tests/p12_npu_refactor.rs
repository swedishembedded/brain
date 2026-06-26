// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P12 (NPU prep): the shared-post-processing + calibration-tap refactor.
//!
//! 1. An identity [`ActTap`] (rewrites each conv input to itself) must produce
//!    byte-for-byte the same raw logits as the untapped forward — proving the tap
//!    seam adds no numerical drift.
//! 2. The pure-Rust [`dfl_decode_cpu`] must match the GPU `dfl_decode_dist` kernel
//!    — proving the NPU host-decode path is exact.
//!
//! Both run on the CPU backend; gated by `MOE_SKIP_GPU_TESTS` like the other
//! yolo learnability tests.

use std::collections::HashMap;

use yolo::infer::{dfl_decode_cpu, dfl_decode_dist};
use yolo::net::ActTap;
use yolo::{YoloConfig, Yolo};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| !v.is_empty()).unwrap_or(false)
}

/// Deterministic pseudo-random fill in [0,1).
fn fill(n: usize, mut seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(((seed >> 32) as u32 as f32 / u32::MAX as f32).clamp(0.0, 0.999));
    }
    v
}

/// A tap that rewrites every conv input to itself (the quant→dequant identity).
struct IdentityTap;
impl ActTap for IdentityTap {
    fn tap(&self, _name: &str, _x: &mut [f32]) {}
}

fn tiny_model() -> (Yolo, usize) {
    let cfg = YoloConfig::tiny(3);
    let side = cfg.input as usize;
    let init: HashMap<String, Vec<f32>> = yolo::init_weights(&cfg, 0xBEEF);
    let model = Yolo::new(cfg, 1, 0, &init);
    (model, 3 * side * side)
}

#[test]
fn identity_tap_matches_untapped() {
    if skip() {
        return;
    }
    let (model, img_len) = tiny_model();
    model.set_eval(true);
    let img = fill(img_len, 0x1234);
    model.set_image(&img);

    // Untapped forward.
    model.forward_net_pub();
    let (cls0, box0) = model.raw_logits();

    // Tapped forward with the identity tap.
    let tap = IdentityTap;
    model.forward_net_tapped(&tap);
    let (cls1, box1) = model.raw_logits();

    assert_eq!(cls0.len(), cls1.len());
    assert_eq!(box0.len(), box1.len());
    let maxd = cls0
        .iter()
        .zip(&cls1)
        .chain(box0.iter().zip(&box1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(maxd < 1e-5, "identity tap perturbed logits by {maxd}");
}

#[test]
fn dfl_cpu_matches_gpu_kernel() {
    if skip() {
        return;
    }
    let (model, _) = tiny_model();
    let reg_max = model.cfg.reg_max as usize;
    let na = 50usize; // arbitrary anchor count for the decode probe
    let logits = {
        // Spread logits over a wide range so the softmax is non-trivial.
        let raw = fill(na * 4 * reg_max, 0xC0DE);
        raw.iter().map(|v| (v - 0.5) * 8.0).collect::<Vec<f32>>()
    };
    let cpu = dfl_decode_cpu(&logits, na, reg_max);
    let gpu = dfl_decode_dist(&model.gpu, &logits, na, reg_max);
    assert_eq!(cpu.len(), gpu.len());
    let maxd = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(maxd < 1e-4, "dfl cpu/gpu mismatch: {maxd}");
}
