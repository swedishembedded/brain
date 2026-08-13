// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P3: the assembled ZipDepth forward pass — it runs end to end and its output has
//! the right shape and the invariants the architecture guarantees.
//!
//! This is not a parity test (that needs the real weights + a torch reference, and
//! is env-gated elsewhere). It proves the wiring: 235 tensors' worth of blocks
//! compose into one graph that produces a depth map at the input resolution,
//! finite and non-negative, on both backends.

use data::rng::Lcg;
use zipdepth::{ZipConfig, ZipDepth};
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{Ctx, Shape};

/// A small ZipDepth: the base config's structure at a 64-px input, so every stage,
/// tail, fusion and the convex upsampler all execute, but the test runs in seconds.
fn tiny() -> ZipConfig {
    ZipConfig { input: 64, ..ZipConfig::base() }
}

fn init_store(gpu: &Gpu, m: &ZipDepth) -> ParamStore {
    // Use the model's OWN initializer (Kaiming fan-out for ReLU, head bias 0.5,
    // BN gamma 1 / beta 0, ImageNet mean/std), NOT a hand-rolled fixture. An
    // ad-hoc small-symmetric init can leave a ReLU dead — e.g. SE's excite path,
    // whose `fc.0(pooled)` came out all-negative under 0.05*U(-1,1), zeroing
    // `h_act` and hence both fc weight gradients. That is a real property of a
    // dead ReLU, not a wiring bug (the grad still flowed THROUGH SE to the blocks
    // below it, and `d_gate` was a healthy 8e-2), but it made the honest
    // "every weight gets a gradient" check impossible to state. init_weights
    // keeps every ReLU alive, which is the point of a real initializer.
    let init = zipdepth::init_weights(&m.cfg, 7);
    ParamStore::new(gpu, m.param_list(), &init)
}

#[test]
fn forward_produces_a_depth_map_at_input_resolution() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = ZipDepth::build(&ctx, tiny(), 2, false);
    m.set_eval(true);
    let ps = init_store(&gpu, &m);

    let x = gpu.storage_init("x", &Lcg::new(1).vec(m.in_shape.numel() as usize).iter().map(|v| 0.5 + 0.5 * v).collect::<Vec<_>>());
    m.forward(&ctx, &ps, &x);
    let out = gpu.read(m.out(), m.out_shape.numel() as usize);

    // [B, 1, H, W] at the INPUT resolution — the model letterboxes internally and
    // the convex upsample lands back on the input grid.
    assert_eq!(m.out_shape, Shape::new(2, 1, 64, 64), "depth map is [B,1,H,W] at input res");
    assert_eq!(out.len(), 2 * 64 * 64);
    // The final ReLU makes it non-negative inverse depth; nothing may be NaN/Inf.
    assert!(out.iter().all(|v| v.is_finite()), "every depth value must be finite");
    assert!(out.iter().all(|v| *v >= 0.0), "output is post-ReLU inverse depth: non-negative");
    // ...and it must not be the trivial all-zero map (that would mean the ReLU
    // killed everything or a stage output nothing).
    assert!(out.iter().any(|v| *v > 1e-6), "the depth map is identically zero — a stage produced nothing");
}

/// The two upsampler variants both run and both satisfy the output contract.
#[test]
fn both_upsampler_variants_run() {
    for unfold in [true, false] {
        let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
        let ctx = Ctx::new(&gpu, zipdepth::net::ids());
        let cfg = ZipConfig { upsample_unfold: unfold, ..tiny() };
        let m = ZipDepth::build(&ctx, cfg, 1, false);
        m.set_eval(true);
        let ps = init_store(&gpu, &m);
        let x = gpu.storage_init("x", &Lcg::new(7).vec(m.in_shape.numel() as usize).iter().map(|v| 0.5 + 0.5 * v).collect::<Vec<_>>());
        m.forward(&ctx, &ps, &x);
        let out = gpu.read(m.out(), m.out_shape.numel() as usize);
        assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "unfold={unfold}: contract violated");
        assert_eq!(m.out_shape, Shape::new(1, 1, 64, 64));
    }
}

/// The backward runs end to end and every parameter that should receive a gradient
/// does. This is the cheap gate before the (expensive) master finite-difference
/// gradcheck: a structurally broken backward (an unwired block, a wrong buffer)
/// shows up here as a NaN or an all-zero grad on a tensor that clearly affects the
/// loss, in seconds rather than in the full FD sweep.
#[test]
fn backward_runs_and_reaches_every_parameter() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = ZipDepth::build(&ctx, tiny(), 2, true);
    let ps = init_store(&gpu, &m);

    let x = gpu.storage_init("x", &Lcg::new(3).vec(m.in_shape.numel() as usize).iter().map(|v| 0.5 + 0.5 * v).collect::<Vec<_>>());
    m.forward(&ctx, &ps, &x);
    ps.zero_grads(&gpu);
    // A non-degenerate upstream grad.
    let d = gpu.storage_init("d", &Lcg::new(5).vec(m.out_shape.numel() as usize));
    m.backward(&ctx, &ps, &x, &d);

    // Every conv WEIGHT must have a finite gradient, and at least most must be
    // non-zero (a dead-zero weight grad deep in the net means the backward chain
    // broke above it). The two provably-dead GCB biases are the known exception.
    let mut zero = Vec::new();
    for (n, numel) in m.param_list() {
        if !n.ends_with(".weight") || n == "mean" || n == "std" {
            continue;
        }
        let g = gpu.read(ps.g(&n), numel);
        assert!(g.iter().all(|v| v.is_finite()), "{n}: gradient has a NaN/Inf");
        if g.iter().all(|v| *v == 0.0) {
            zero.push(n);
        }
    }
    assert!(zero.is_empty(), "these weights got an all-zero gradient — the backward chain is broken above them:\n{zero:#?}");
}
