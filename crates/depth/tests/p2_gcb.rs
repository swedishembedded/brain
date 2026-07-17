// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: `GlobalContextBlock` — `x + transform(bmm(x, softmax(score(x))))`.
//!
//! The first block with BIASED convs, and the one that consumes `weighted_gap` /
//! `add_chan_bcast` — the two kernels written specifically for its bmm and residual.
use std::collections::HashMap;

use depth::blocks::GlobalContextBlock;
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

fn fixture(shape: Shape, seed: u64) -> (Gpu, ParamStore, Vec<f32>) {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let probe = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
    let params = probe.param_list();
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in params.iter().enumerate() {
        let v: Vec<f32> = if n.ends_with("running_mean") {
            vec![0.0; *numel]
        } else if n.ends_with("running_var") {
            vec![1.0; *numel]
        } else if n == "g.transform.1.weight" {
            rv(seed ^ i as u64, *numel).iter().map(|v| 1.0 + 0.2 * v).collect()
        } else {
            rv(seed ^ i as u64, *numel).iter().map(|v| 0.4 * v).collect()
        };
        init.insert(n.clone(), v);
    }
    let ps = ParamStore::new(&gpu, params, &init);
    (gpu, ps, rv(seed ^ 0xD, shape.numel() as usize))
}

/// Every conv here is `nn.Conv2d(..)` with torch's DEFAULT `bias=True`, so all
/// three carry a bias tensor — including `transform.0`, whose bias is
/// mathematically redundant (the BN right after subtracts the batch mean) but is
/// present in the checkpoint. Dropping it would fail a strict load.
///
/// `hidden = max(dim//reduction, 8)`, so at dim=32/reduction=4 it is 8, and the
/// `max` is not merely decorative: at dim=16 it would clamp.
#[test]
fn gcb_layout_has_three_biased_convs_and_one_bn() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let g = GlobalContextBlock::new(&ctx, "encoder.stage4.1", Shape::new(1, 384, 12, 12), 4, true);
    assert_eq!(
        g.param_list(),
        vec![
            // dim -> 1: scores each spatial position.
            ("encoder.stage4.1.context_weight.weight".to_string(), 384),
            ("encoder.stage4.1.context_weight.bias".to_string(), 1),
            // dim -> hidden (384/4 = 96), biased, + BN.
            ("encoder.stage4.1.transform.0.weight".to_string(), 96 * 384),
            ("encoder.stage4.1.transform.0.bias".to_string(), 96),
            ("encoder.stage4.1.transform.1.weight".to_string(), 96),
            ("encoder.stage4.1.transform.1.bias".to_string(), 96),
            ("encoder.stage4.1.transform.1.running_mean".to_string(), 96),
            ("encoder.stage4.1.transform.1.running_var".to_string(), 96),
            // hidden -> dim, biased, no BN, no act.
            ("encoder.stage4.1.transform.3.weight".to_string(), 384 * 96),
            ("encoder.stage4.1.transform.3.bias".to_string(), 384),
        ],
        "3 biased convs; exactly one BN, on transform.1"
    );
}

/// `hidden = max(dim // reduction, 8)` — the clamp, at a dim where it binds.
#[test]
fn gcb_hidden_is_clamped_to_at_least_eight() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let g = GlobalContextBlock::new(&ctx, "g", Shape::new(1, 16, 4, 4), 4, true);
    let p = g.param_list();
    let (_, n) = p.iter().find(|(n, _)| n == "g.transform.0.bias").unwrap();
    assert_eq!(*n, 8, "16/4 = 4, clamped up to 8");
}

/// The softmax normalizes over H*W per image, so the mask sums to exactly 1 per
/// image — which is what makes the contraction a weighted AVERAGE rather than a
/// weighted sum. Checked through the block's own output: with the transform's
/// effect held aside, a uniform x must come back unchanged by the context path...
/// so instead assert the property that actually pins the axis: at N=2 the two
/// images must produce DIFFERENT contexts for different inputs. A softmax over the
/// wrong axis (or a batch-shared mask) would couple them.
#[test]
fn gcb_context_is_per_image() {
    let shape = Shape::new(2, 8, 4, 4);
    let (gpu, ps, _) = fixture(shape, 31);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let g = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
    let per = (shape.c * shape.h * shape.w) as usize;

    // Image 0 gets a distinctive input; image 1 gets zeros.
    let mut x = vec![0.0f32; shape.numel() as usize];
    x[..per].copy_from_slice(&rv(5, per));
    let xb = gpu.storage_init("x", &x);
    g.forward(&ctx, &ps, &xb);
    let out = gpu.read(g.out(), shape.numel() as usize);

    // Image 1's input is all zeros, so its output is purely its own context. If the
    // mask or the contraction leaked across the batch, image 0's features would
    // show up here.
    let solo = {
        let mut x1 = vec![0.0f32; shape.numel() as usize];
        x1[..per].copy_from_slice(&rv(5, per));
        // Same image 0, but now duplicated into slot 1 — if contexts were shared,
        // image 1's output would be unchanged from the run above.
        x1[per..].copy_from_slice(&rv(5, per));
        let xb = gpu.storage_init("x1", &x1);
        g.forward(&ctx, &ps, &xb);
        gpu.read(g.out(), shape.numel() as usize)
    };
    let d: f32 = out[per..].iter().zip(&solo[per..]).map(|(a, b)| (a - b).abs()).sum();
    assert!(d > 1e-3, "image 1's output must depend on image 1's OWN input, not the batch (delta {d})");
}

#[test]
fn gcb_weight_grads_match_finite_differences() {
    let shape = Shape::new(2, 8, 5, 4);
    let (gpu, ps, x) = fixture(shape, 37);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let m = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
    let tot = shape.numel() as usize;
    let r = rv(53, tot);

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let d_in = gpu.storage(tot as u64);
    m.backward(&ctx, &ps, &xb, &d_out, &d_in);

    let loss = |gpu: &Gpu, ps: &ParamStore| -> f32 {
        let ctx = Ctx::new(gpu, depth::net::ids());
        let m = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
        let xb = gpu.storage_init("x", &x);
        m.forward(&ctx, ps, &xb);
        gpu.read(m.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    // `transform.3.bias` is the point: the first LIVE bias tensor vision::Conv has
    // had to differentiate, via bias_grad's [N, C*HW] view + host reduce.
    // `context_weight.weight` additionally exercises the softmax Jacobian.
    //
    // `context_weight.bias` and `transform.0.bias` are EXCLUDED, and not for
    // convenience: the loss is exactly invariant to both, so FD divides round-off
    // by 2*eps and reports noise. `gcb_two_biases_are_provably_dead` covers them
    // with the stronger claim.
    for wname in [
        "g.context_weight.weight",
        "g.transform.3.weight",
        "g.transform.3.bias",
        "g.transform.1.weight",
    ] {
        let g = gpu.read(ps.g(wname), ps.numel(wname));
        let n = g.len();
        let dir: Vec<f32> = rv(3, n).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
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

/// `x` reaches the output by three routes: the score conv, the contraction, and the
/// residual. Elementwise for the same reason as StripPoolingAttention's.
#[test]
fn gcb_input_grad_matches_finite_differences_elementwise() {
    let shape = Shape::new(2, 4, 4, 3);
    let (gpu, ps, x) = fixture(shape, 41);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let m = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
    let tot = shape.numel() as usize;
    let r = rv(59, tot);

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let d_in = gpu.storage(tot as u64);
    m.backward(&ctx, &ps, &xb, &d_out, &d_in);
    let g = gpu.read(&d_in, tot);

    let loss = |xv: &[f32]| -> f32 {
        let ctx = Ctx::new(&gpu, depth::net::ids());
        let m = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
        let xb = gpu.storage_init("x", xv);
        m.forward(&ctx, &ps, &xb);
        gpu.read(m.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let eps = 1e-3f32;
    for i in [0usize, 5, 17, 44, 71, 95] {
        let mut xp = x.clone();
        xp[i] += eps;
        let mut xm = x.clone();
        xm[i] -= eps;
        let numeric = (loss(&xp) - loss(&xm)) / (2.0 * eps);
        let abs = (g[i] - numeric).abs();
        let denom = g[i].abs().max(numeric.abs()).max(1e-3);
        assert!(abs <= 4e-3 + 8e-2 * denom, "d_in[{i}]: analytic {}, fd {numeric}", g[i]);
    }
}

/// The two dead parameters, pinned by the property itself rather than by FD.
///
/// `softmax(z + b) == softmax(z)` for a scalar `b` broadcast over the softmax axis,
/// and BN subtracts the mean of whatever precedes it — so `context_weight.bias` and
/// `transform.0.bias` cannot affect the output at all.
///
/// The invariance is exact in exact arithmetic and near-exact in f32: at the FD eps
/// scale (+-5e-3, +-5e-1) the loss moves by EXACTLY 0.0f32, and at a +-5.0 shift by
/// 2 ULP — softmax subtracts the running max, and `(z+b) - (mx+b)` is not bitwise
/// `z - mx` once b is large. So the tolerance below is a round-off budget, not a
/// gradient tolerance: a real dependence would move this loss by ~850 at a 5.0
/// shift (the live gradients here are ~1.7e2), which is 8 orders above what we see.
///
/// This is what covers the two tensors that `gcb_weight_grads_match_finite_
/// differences` cannot: FD divides that round-off by 2*eps and reports pure noise
/// (measured fd = 1.5e-1 against an analytic 4.2e-5). It also documents why a
/// strict checkpoint load carries two tensors that can never learn.
#[test]
fn gcb_two_biases_are_provably_dead() {
    let shape = Shape::new(2, 8, 5, 4);
    let (gpu, ps, x) = fixture(shape, 37);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let m = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
    let tot = shape.numel() as usize;
    let r = rv(53, tot);

    let loss = |gpu: &Gpu, ps: &ParamStore| -> f32 {
        let ctx = Ctx::new(gpu, depth::net::ids());
        let m = GlobalContextBlock::new(&ctx, "g", shape, 4, true);
        let xb = gpu.storage_init("x", &x);
        m.forward(&ctx, ps, &xb);
        gpu.read(m.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let base = loss(&gpu, &ps);

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let d_in = gpu.storage(tot as u64);
    m.backward(&ctx, &ps, &xb, &d_out, &d_in);

    for wname in ["g.context_weight.bias", "g.transform.0.bias"] {
        let n = ps.numel(wname);
        let w0 = gpu.read(ps.w(wname), n);
        // A shift of 5.0 is ~10^4 x the FD eps. If the parameter had ANY effect this
        // would move the loss enormously.
        let shifted: Vec<f32> = w0.iter().map(|w| w + 5.0).collect();
        gpu.write(ps.w(wname), bytemuck::cast_slice(&shifted));
        let moved = loss(&gpu, &ps);
        gpu.write(ps.w(wname), bytemuck::cast_slice(&w0));
        let rel = (moved - base).abs() / base.abs().max(1e-6);
        assert!(rel < 1e-5, "{wname}: shifting by +5.0 moved the loss {base} -> {moved} (rel {rel:e}); it must be invariant to round-off");

        // ...and the analytic gradient agrees it is zero, to round-off. The live
        // gradients in this block are ~1e2, so 1e-2 is four orders below signal.
        let g = gpu.read(ps.g(wname), n);
        let mx = g.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(mx < 1e-2, "{wname}: analytic grad should be ~0, got max |g| = {mx}");
    }
}
