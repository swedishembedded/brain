// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cross-attention injection seam (`model::attninject::CrossAttnInject`).
//!
//! Two properties, and the first is the one that would rot silently:
//!
//! 1. **A zero-contribution adapter is bit-identical to no adapter.** If merely
//!    installing the seam perturbed the graph, every number in `tests/parity.rs`
//!    would have been quietly invalidated by adding it.
//! 2. **An installed adapter reaches EVERY site, once, in graph order.** A seam
//!    that is wired but never dispatched looks exactly like a correct no-op —
//!    `.agents/rules/lessons.md` #1 and #8 in one.
//!
//! Weight-free at `UNetConfig::tiny`, so it always runs.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::attninject::CrossAttnInject;
use unet::config::UNetConfig;
use unet::init::init_weights;
use unet::model::{Unet, KERNELS};

/// The backbone's kernels PLUS what the stub adapter dispatches. An adapter runs
/// on the backbone's device, so the caller builds it from the union — the same
/// tail-append `facenet::caps::SERVING_PIPELINES` uses, which keeps one kernel
/// index space and leaves every existing index valid.
const N: usize = KERNELS.len() + 1;
static INJECT_KERNELS: [(&str, &str); N] = union_set();
const fn union_set() -> [(&'static str, &'static str); N] {
    let mut k = [("", ""); N];
    let mut i = 0;
    while i < KERNELS.len() {
        k[i] = KERNELS[i];
        i += 1;
    }
    k[N - 1] = ("axpy", kernels::AXPY);
    k
}

/// Adds `bump` to every element of every cross-attention context, and records
/// which sites it was asked for.
struct StubInject {
    n: usize,
    bump: f32,
    seen: std::sync::Mutex<Vec<usize>>,
    axpy: usize,
    ones: std::sync::Mutex<Vec<DeviceBuffer>>,
}

impl StubInject {
    fn new(n: usize, bump: f32) -> StubInject {
        let axpy = INJECT_KERNELS.iter().position(|(k, _)| *k == "axpy").expect("axpy in the union");
        StubInject { n, bump, seen: std::sync::Mutex::new(Vec::new()), axpy, ones: std::sync::Mutex::new(Vec::new()) }
    }
}

impl CrossAttnInject for StubInject {
    fn kernels(&self) -> &'static [(&'static str, &'static str)] {
        &[("axpy", kernels::AXPY)]
    }
    fn sites(&self) -> usize {
        self.n
    }
    fn inject(&self, steps: &mut Vec<Step>, gpu: &Gpu, k: usize, _q: &DeviceBuffer, ctx: &DeviceBuffer, t: u32, c: u32) {
        self.seen.lock().unwrap().push(k);
        let ones = gpu.storage((t * c) as u64);
        gpu.write_f32(&ones, &vec![1.0f32; (t * c) as usize]);
        // ctx += bump * ones
        steps.push(gpu.step(self.axpy, &[ctx, &ones], &[t * c, self.bump.to_bits()], t * c));
        // The buffer must outlive the recorded step; the adapter owns its scratch.
        self.ones.lock().unwrap().push(ones);
    }
}

fn cfg_w() -> (UNetConfig, unet::import::Tensors) {
    let cfg = UNetConfig::tiny();
    let w = init_weights(&cfg, 7);
    (cfg, w)
}

fn inputs(m: &Unet) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = m.config();
    let sample: Vec<f32> = (0..(c.in_channels * 16 * 16) as usize).map(|i| ((i as f32) * 0.017).sin()).collect();
    let enc: Vec<f32> = (0..(9 * c.cross_attention_dim) as usize).map(|i| ((i as f32) * 0.031).cos()).collect();
    let pooled: Vec<f32> = (0..c.pooled_dim() as usize).map(|i| ((i as f32) * 0.11).sin()).collect();
    (sample, enc, pooled, vec![64.0, 64.0, 0.0, 0.0, 64.0, 64.0])
}

#[test]
fn a_zero_contribution_adapter_is_bit_identical_to_none() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (cfg, w) = cfg_w();
    let gpu = gpu_core::testgpu::dev(&INJECT_KERNELS);

    let plain = Unet::new(gpu.share(), cfg.clone(), &w, 16, 16, 9, false);
    let (s, e, p, t) = inputs(&plain);
    let base = plain.run(&s, 601.0, &e, &p, &t);

    let n = plain.cross_attention_sites();
    let stub = StubInject::new(n, 0.0);
    let inj = Unet::new_injected(gpu.share(), cfg, &w, 16, 16, 9, false, false, &stub);
    let got = inj.run(&s, 601.0, &e, &p, &t);

    assert_eq!(base.len(), got.len());
    let diff = base.iter().zip(&got).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(diff, 0, "a zero-contribution adapter changed {diff} of {} outputs", base.len());
}

#[test]
fn an_installed_adapter_reaches_every_site_once_in_order() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (cfg, w) = cfg_w();
    let gpu = gpu_core::testgpu::dev(&INJECT_KERNELS);

    let plain = Unet::new(gpu.share(), cfg.clone(), &w, 16, 16, 9, false);
    let (s, e, p, t) = inputs(&plain);
    let base = plain.run(&s, 601.0, &e, &p, &t);
    let n = plain.cross_attention_sites();
    assert!(n > 1, "the tiny config should record several cross-attention sites, got {n}");

    let stub = StubInject::new(n, 0.5);
    let inj = Unet::new_injected(gpu.share(), cfg, &w, 16, 16, 9, false, false, &stub);
    let got = inj.run(&s, 601.0, &e, &p, &t);

    let seen = stub.seen.lock().unwrap().clone();
    assert_eq!(seen, (0..n).collect::<Vec<_>>(), "sites must arrive once each, in graph order");

    let moved = base.iter().zip(&got).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert!(moved > 0, "the adapter contributed nothing to the output");
    assert!(got.iter().all(|v| v.is_finite()), "injected output has non-finite values");
}

/// A count mismatch must fail at construction with two numbers, not mid-forward
/// with a shape.
#[test]
#[should_panic(expected = "cross-attention sites")]
fn a_wrong_site_count_is_rejected_at_construction() {
    let (cfg, w) = cfg_w();
    let gpu = gpu_core::testgpu::dev(&INJECT_KERNELS);
    let stub = StubInject::new(1, 0.0);
    let _ = Unet::new_injected(gpu, cfg, &w, 16, 16, 9, false, false, &stub);
}
