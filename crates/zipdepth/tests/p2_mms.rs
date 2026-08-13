// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: `MinimalMultiScale` — `x + BN(dw_d1(x) + dw_d2(x))`.
//!
//! The block that forced `vision::BatchNorm` to exist: its BN spans the SUM of two
//! convs, which a unit whose BN is welded to one conv cannot express.
use data::rng::Lcg;
use std::collections::HashMap;
use zipdepth::blocks::MinimalMultiScale;
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{Ctx, Shape};

fn fixture(shape: Shape, seed: u64) -> (Gpu, ParamStore, Vec<f32>) {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let probe = MinimalMultiScale::new(&ctx, "m", shape, true);
    let params = probe.param_list();
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in params.iter().enumerate() {
        let v: Vec<f32> = if n.ends_with("running_mean") { vec![0.0; *numel] }
            else if n.ends_with("running_var") { vec![1.0; *numel] }
            else if n == "m.bn.weight" { Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 1.0 + 0.2 * v).collect() }
            else if n == "m.bn.bias" { Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 0.1 * v).collect() }
            else { Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 0.3 * v).collect() };
        init.insert(n.clone(), v);
    }
    let ps = ParamStore::new(&gpu, params, &init);
    (gpu, ps, Lcg::new(seed ^ 0xD).vec(shape.numel() as usize))
}

/// The layout the reference actually has: TWO depthwise weights `[C,1,3,3]` and
/// exactly ONE bn. The branches must contribute no BN tensors of their own — if
/// they did, BatchNorm would run twice and the extra tensors would not exist in
/// the checkpoint.
#[test]
fn mms_has_two_depthwise_weights_and_exactly_one_bn() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = MinimalMultiScale::new(&ctx, "encoder.stage2.2", Shape::new(1, 96, 48, 48), true);
    let p = m.param_list();
    assert_eq!(p, vec![
        ("encoder.stage2.2.branch1.weight".to_string(), 96 * 3 * 3),
        ("encoder.stage2.2.branch2.weight".to_string(), 96 * 3 * 3),
        ("encoder.stage2.2.bn.weight".to_string(), 96),
        ("encoder.stage2.2.bn.bias".to_string(), 96),
        ("encoder.stage2.2.bn.running_mean".to_string(), 96),
        ("encoder.stage2.2.bn.running_var".to_string(), 96),
    ], "layout must be 2 depthwise weights + ONE bn");
    assert_eq!(p.iter().filter(|(n, _)| n.contains("running_var")).count(), 1, "exactly one BatchNorm");
}

/// Both branches must be shape-preserving despite different dilations — that is
/// the point of the block (two receptive fields, one output shape).
#[test]
fn mms_is_shape_preserving() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    for s in [Shape::new(1, 96, 48, 48), Shape::new(2, 8, 7, 5)] {
        let m = MinimalMultiScale::new(&ctx, "m", s, true);
        assert_eq!(m.shape, s);
    }
}

#[test]
fn mms_backward_matches_finite_differences() {
    let shape = Shape::new(2, 8, 6, 6);
    let (gpu, ps, x) = fixture(shape, 17);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = MinimalMultiScale::new(&ctx, "m", shape, true);
    let tot = shape.numel() as usize;
    let r = Lcg::new(41).vec(tot);

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let d_in = gpu.storage(tot as u64);
    m.backward(&ctx, &ps, &xb, &d_out, &d_in);

    let loss = |gpu: &Gpu, ps: &ParamStore| -> f32 {
        let ctx = Ctx::new(gpu, zipdepth::net::ids());
        let m = MinimalMultiScale::new(&ctx, "m", shape, true);
        let xb = gpu.storage_init("x", &x);
        m.forward(&ctx, ps, &xb);
        gpu.read(m.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    // Both branch weights AND the shared BN's gamma — the BN grad is the one that
    // would be silently wrong if the standalone unit's packing diverged.
    for wname in ["m.branch1.weight", "m.branch2.weight", "m.bn.weight"] {
        let g = gpu.read(ps.g(wname), ps.numel(wname));
        let n = g.len();
        let dir: Vec<f32> = Lcg::new(3).vec(n).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
        let analytic: f32 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let w0 = gpu.read(ps.w(wname), n);
        let eps = 5e-4f32;
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w + eps * d).collect();
        gpu.write(ps.w(wname), bytemuck::cast_slice(&wp));
        let lp = loss(&gpu, &ps);
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w - eps * d).collect();
        gpu.write(ps.w(wname), bytemuck::cast_slice(&wm));
        let lm = loss(&gpu, &ps);
        gpu.write(ps.w(wname), bytemuck::cast_slice(&w0));
        let numeric = (lp - lm) / (2.0 * eps);
        let abs = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        assert!(abs <= 4e-3 + 8e-2 * denom, "{wname}: analytic {analytic}, fd {numeric}");
    }
}
