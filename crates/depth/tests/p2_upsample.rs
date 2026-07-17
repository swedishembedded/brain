// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: `FastConvexUpsample`, both variants.
//!
//! The decoder's tail, and the only block whose two variants are DIFFERENT
//! ARCHITECTURES rather than two implementations — the two released checkpoints
//! differ by exactly this.
use std::collections::HashMap;

use depth::blocks::{FastConvexUpsample, UpsampleKind};
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{Ctx, Shape};

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}
fn rv(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

fn store(gpu: &Gpu, params: Vec<(String, usize)>, seed: u64) -> ParamStore {
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in params.iter().enumerate() {
        let v: Vec<f32> = if n.ends_with("running_mean") {
            vec![0.0; *numel]
        } else if n.ends_with("running_var") {
            vec![1.0; *numel]
        } else if n.ends_with(".1.weight") || n.ends_with(".4.weight") {
            rv(seed ^ i as u64, *numel).iter().map(|v| 1.0 + 0.2 * v).collect()
        } else {
            rv(seed ^ i as u64, *numel).iter().map(|v| 0.4 * v).collect()
        };
        init.insert(n.clone(), v);
    }
    ParamStore::new(gpu, params, &init)
}

/// The unfold path's layout: `mask_pred.{0,1,3}`. `.3` predicts `9*S*S` channels —
/// nine neighbours x every sub-pixel — and is BIASED; `.0` is bias-free with BN.
#[test]
fn unfold_layout_is_mask_pred() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let m = FastConvexUpsample::new(
        &ctx,
        "dec.up",
        UpsampleKind::Unfold,
        Shape::new(1, 32, 96, 96),
        Shape::new(1, 1, 96, 96),
        4,
        1.0,
        true,
    );
    // hidden = max(32/4, 8) = 8; out = 9*4*4 = 144.
    assert_eq!(
        m.param_list(),
        vec![
            ("dec.up.mask_pred.0.weight".to_string(), 8 * 32 * 3 * 3),
            ("dec.up.mask_pred.1.weight".to_string(), 8),
            ("dec.up.mask_pred.1.bias".to_string(), 8),
            ("dec.up.mask_pred.1.running_mean".to_string(), 8),
            ("dec.up.mask_pred.1.running_var".to_string(), 8),
            ("dec.up.mask_pred.3.weight".to_string(), 144 * 8),
            ("dec.up.mask_pred.3.bias".to_string(), 144),
        ]
    );
    assert_eq!(m.out_shape, Shape::new(1, 1, 384, 384), "S=4 from the half-res grid");
}

/// The NPU path is a DIFFERENT architecture, not a different implementation:
/// `where_conv.{0,1,3,4,6}`, all bias-free, with a depthwise 5x5 in the middle.
/// `where_hidden = max(in/2, 8)` — note /2, where the unfold path uses /4.
#[test]
fn blend_layout_is_where_conv() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let m = FastConvexUpsample::new(
        &ctx,
        "dec.up",
        UpsampleKind::Blend,
        Shape::new(1, 32, 96, 96),
        Shape::new(1, 1, 96, 96),
        4,
        1.0,
        true,
    );
    // where_hidden = max(32/2, 8) = 16.
    assert_eq!(
        m.param_list(),
        vec![
            ("dec.up.where_conv.0.weight".to_string(), 16 * 32),
            ("dec.up.where_conv.1.weight".to_string(), 16),
            ("dec.up.where_conv.1.bias".to_string(), 16),
            ("dec.up.where_conv.1.running_mean".to_string(), 16),
            ("dec.up.where_conv.1.running_var".to_string(), 16),
            // depthwise 5x5 -> [16, 1, 5, 5]
            ("dec.up.where_conv.3.weight".to_string(), 16 * 5 * 5),
            ("dec.up.where_conv.4.weight".to_string(), 16),
            ("dec.up.where_conv.4.bias".to_string(), 16),
            ("dec.up.where_conv.4.running_mean".to_string(), 16),
            ("dec.up.where_conv.4.running_var".to_string(), 16),
            ("dec.up.where_conv.6.weight".to_string(), 16),
        ],
        "no biases anywhere on this path"
    );
}

/// The two variants share NO parameter names — which is why the released
/// checkpoints have 278 vs 283 tensors and why picking the wrong one fails a
/// strict load rather than degrading quietly.
#[test]
fn the_two_variants_share_no_parameters() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let (feat, d) = (Shape::new(1, 32, 8, 8), Shape::new(1, 1, 8, 8));
    let u = FastConvexUpsample::new(&ctx, "u", UpsampleKind::Unfold, feat, d, 4, 1.0, true);
    let b = FastConvexUpsample::new(&ctx, "u", UpsampleKind::Blend, feat, d, 4, 1.0, true);
    let un: Vec<_> = u.param_list().into_iter().map(|(n, _)| n).collect();
    let bn: Vec<_> = b.param_list().into_iter().map(|(n, _)| n).collect();
    assert!(un.iter().all(|n| !bn.contains(n)), "the paths must not alias each other's tensors");
}

/// The defining property of the convex path: the mask is softmax'd over the nine
/// neighbours, so every output is a CONVEX combination of the 3x3 input
/// neighbourhood — weights >= 0 summing to 1. The output therefore cannot
/// overshoot the local input range, whatever the mask predicts.
///
/// Pinned with a constant depth map: every neighbour is the same value, so any
/// convex combination of them is that value exactly. A mask that did not sum to 1
/// (softmax over the wrong axis, say) would scale the output away from it.
#[test]
fn convex_upsample_of_a_constant_depth_is_that_constant() {
    let (feat, d) = (Shape::new(2, 16, 5, 4), Shape::new(2, 1, 5, 4));
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let probe = FastConvexUpsample::new(&ctx, "u", UpsampleKind::Unfold, feat, d, 2, 1.0, true);
    let ps = store(&gpu, probe.param_list(), 97);
    let m = FastConvexUpsample::new(&ctx, "u", UpsampleKind::Unfold, feat, d, 2, 1.0, true);

    let fb = gpu.storage_init("f", &rv(0xE1, feat.numel() as usize));
    // A positive constant, so the final ReLU is the identity here.
    let db = gpu.storage_init("d", &vec![2.5f32; d.numel() as usize]);
    m.forward(&ctx, &ps, &fb, &db);
    let out = gpu.read(m.out(), m.out_shape.numel() as usize);
    for (i, v) in out.iter().enumerate() {
        assert!((v - 2.5).abs() < 1e-5, "out[{i}] = {v}: a convex combination of 2.5s must be 2.5");
    }
}

/// The blend path's counterpart: `a*nn + (1-a)*bi` of a constant is that constant
/// for ANY alpha, since nearest and bilinear both reproduce a constant.
#[test]
fn blend_upsample_of_a_constant_depth_is_that_constant() {
    let (feat, d) = (Shape::new(2, 16, 5, 4), Shape::new(2, 1, 5, 4));
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let probe = FastConvexUpsample::new(&ctx, "u", UpsampleKind::Blend, feat, d, 2, 1.0, true);
    let ps = store(&gpu, probe.param_list(), 101);
    let m = FastConvexUpsample::new(&ctx, "u", UpsampleKind::Blend, feat, d, 2, 1.0, true);

    let fb = gpu.storage_init("f", &rv(0xE2, feat.numel() as usize));
    let db = gpu.storage_init("d", &vec![1.75f32; d.numel() as usize]);
    m.forward(&ctx, &ps, &fb, &db);
    let out = gpu.read(m.out(), m.out_shape.numel() as usize);
    for (i, v) in out.iter().enumerate() {
        assert!((v - 1.75).abs() < 1e-5, "out[{i}] = {v}");
    }
}

fn fd_both(kind: UpsampleKind, seed: u64, weights: &[&str]) {
    let (feat, d) = (Shape::new(2, 16, 4, 3), Shape::new(2, 1, 4, 3));
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let probe = FastConvexUpsample::new(&ctx, "u", kind, feat, d, 2, 1.0, true);
    let ps = store(&gpu, probe.param_list(), seed);
    let m = FastConvexUpsample::new(&ctx, "u", kind, feat, d, 2, 1.0, true);
    let no = m.out_shape.numel() as usize;
    let fv = rv(0xF1 ^ seed, feat.numel() as usize);
    // Strictly positive: keeps the output off the final ReLU's kink.
    let dv: Vec<f32> = rv(0xF2 ^ seed, d.numel() as usize).iter().map(|v| 2.0 + 0.3 * v).collect();
    // The loss weights are CENTERED, and that is not cosmetic. `out` here is ~2.0
    // everywhere (a depth map, not a zero-mean feature map), so an uncentered `r`
    // makes the loss ~= 2.0 * sum(r) — measured at -87.74, whose f32 ULP is 7.6e-6.
    // The signal being differentiated is ~1e-2, so at eps=5e-4 the whole central
    // difference `lp - lm` is FIVE ULPs and every FD value comes out an exact
    // multiple of the ULP: pure quantization, converging to the analytic only once
    // eps reaches 1e-2. Centering `r` removes the constant that carries no gradient
    // and restores ~4 orders of FD headroom.
    let raw = rv(0xF3 ^ seed, no);
    let mean = raw.iter().sum::<f32>() / raw.len() as f32;
    let r: Vec<f32> = raw.iter().map(|v| v - mean).collect();

    let fb = gpu.storage_init("f", &fv);
    let db = gpu.storage_init("d", &dv);
    m.forward(&ctx, &ps, &fb, &db);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let (dfe, dde) = (gpu.storage(feat.numel() as u64), gpu.storage(d.numel() as u64));
    m.backward(&ctx, &ps, &fb, &db, &d_out, &dfe, &dde);

    let loss = |gpu: &Gpu, ps: &ParamStore| -> f32 {
        let ctx = Ctx::new(gpu, depth::net::ids());
        let m = FastConvexUpsample::new(&ctx, "u", kind, feat, d, 2, 1.0, true);
        let fb = gpu.storage_init("f", &fv);
        let db = gpu.storage_init("d", &dv);
        m.forward(&ctx, ps, &fb, &db);
        gpu.read(m.out(), no).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    for w in weights {
        let g = gpu.read(ps.g(w), ps.numel(w));
        let n = g.len();
        let dir: Vec<f32> = rv(3, n).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
        let analytic: f32 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let w0 = gpu.read(ps.w(w), n);
        let eps = 5e-4f32;
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(a, b)| a + eps * b).collect();
        gpu.write(ps.w(w), bytemuck::cast_slice(&wp));
        let lp = loss(&gpu, &ps);
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(a, b)| a - eps * b).collect();
        gpu.write(ps.w(w), bytemuck::cast_slice(&wm));
        let lm = loss(&gpu, &ps);
        gpu.write(ps.w(w), bytemuck::cast_slice(&w0));
        let numeric = (lp - lm) / (2.0 * eps);
        let abs = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        assert!(abs <= 4e-3 + 8e-2 * denom, "{w}: analytic {analytic}, fd {numeric}");
    }

    // `depth`'s own gradient — elementwise, since it reaches every output pixel.
    let gd = gpu.read(&dde, d.numel() as usize);
    let loss_d = |dvv: &[f32]| -> f32 {
        let ctx = Ctx::new(&gpu, depth::net::ids());
        let m = FastConvexUpsample::new(&ctx, "u", kind, feat, d, 2, 1.0, true);
        let fb = gpu.storage_init("f", &fv);
        let db = gpu.storage_init("d", dvv);
        m.forward(&ctx, &ps, &fb, &db);
        gpu.read(m.out(), no).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let eps = 1e-3f32;
    for i in 0..dv.len() {
        let (mut a, mut b) = (dv.clone(), dv.clone());
        a[i] += eps;
        b[i] -= eps;
        let numeric = (loss_d(&a) - loss_d(&b)) / (2.0 * eps);
        let tol = 4e-3 + 8e-2 * gd[i].abs().max(numeric.abs()).max(1e-3);
        assert!((gd[i] - numeric).abs() <= tol, "d_depth[{i}]: analytic {} vs fd {numeric}", gd[i]);
    }
}

#[test]
fn unfold_backward_matches_finite_differences() {
    fd_both(UpsampleKind::Unfold, 103, &["u.mask_pred.0.weight", "u.mask_pred.3.weight", "u.mask_pred.3.bias"]);
}

#[test]
fn blend_backward_matches_finite_differences() {
    fd_both(
        UpsampleKind::Blend,
        107,
        &["u.where_conv.0.weight", "u.where_conv.3.weight", "u.where_conv.6.weight", "u.where_conv.1.weight"],
    );
}
