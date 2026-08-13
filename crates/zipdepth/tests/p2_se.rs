// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: `ChannelAttention` (squeeze-and-excitation) — stage3's gate.
use data::rng::Lcg;
use std::collections::HashMap;
use zipdepth::blocks::ChannelAttention;
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{Ctx, Shape};

fn fixture(shape: Shape, seed: u64) -> (Gpu, ParamStore, Vec<f32>) {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let probe = ChannelAttention::new(&ctx, "se", shape);
    let params = probe.param_list();
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in params.iter().enumerate() {
        init.insert(n.clone(), Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 0.4 * v).collect());
    }
    let ps = ParamStore::new(&gpu, params, &init);
    (gpu, ps, Lcg::new(seed ^ 0xF).vec(shape.numel() as usize))
}

/// `hidden = max(dim/8, 4)`, and both convs are BIAS-FREE with no BatchNorm.
#[test]
fn se_param_layout_matches_the_reference() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let se = ChannelAttention::new(&ctx, "encoder.stage3.6", Shape::new(1, 192, 24, 24));
    let p = se.param_list();
    // dim 192 -> hidden 24
    assert_eq!(p, vec![
        ("encoder.stage3.6.fc.0.weight".to_string(), 24 * 192),
        ("encoder.stage3.6.fc.2.weight".to_string(), 192 * 24),
    ]);
    // The reduction floor: max(dim/8, 4).
    let tiny = ChannelAttention::new(&ctx, "t", Shape::new(1, 16, 4, 4));
    assert_eq!(tiny.param_list()[0].1, 4 * 16, "hidden must floor at 4, not 2");
}

/// THE per-image trap. The gate is [N,C,1,1], so each image must be scaled by its
/// OWN gate. A `scale_chan(c=C, inner=H*W)` would apply image 0's gate to every
/// image — plausible output, wrong model. Asserted by making the two images
/// different and checking each against its own pooled statistics.
#[test]
fn se_gate_is_per_image_not_shared_across_the_batch() {
    let shape = Shape::new(2, 8, 4, 4);
    let (gpu, ps, _) = fixture(shape, 3);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let se = ChannelAttention::new(&ctx, "se", shape);

    // Image 0 all +1, image 1 all -1 -> their pooled means differ in sign, so
    // their gates must differ, so their outputs must not be scaled identically.
    let mut x = vec![1.0f32; shape.numel() as usize];
    let half = (shape.numel() / 2) as usize;
    for v in x[half..].iter_mut() { *v = -1.0; }
    let xb = gpu.storage_init("x", &x);
    se.forward(&ctx, &ps, &xb);
    let out = gpu.read(se.out(), shape.numel() as usize);

    // out = x * gate; with x = +/-1 the gate is recoverable as |out|.
    let g0: Vec<f32> = (0..shape.c as usize).map(|c| out[c * 16].abs()).collect();
    let g1: Vec<f32> = (0..shape.c as usize).map(|c| out[half + c * 16].abs()).collect();
    assert!(
        g0.iter().zip(&g1).any(|(a, b)| (a - b).abs() > 1e-4),
        "image 0 and image 1 got the SAME gate ({g0:?} vs {g1:?}) — scale_chan is \
         indexing by channel only, so the batch shares image 0's gate"
    );
}

#[test]
fn se_backward_matches_finite_differences() {
    for shape in [Shape::new(2, 8, 5, 5), Shape::new(1, 32, 4, 4)] {
        let (gpu, ps, x) = fixture(shape, 9);
        let ctx = Ctx::new(&gpu, zipdepth::net::ids());
        let se = ChannelAttention::new(&ctx, "se", shape);
        let tot = shape.numel() as usize;
        let r = Lcg::new(55).vec(tot);

        let xb = gpu.storage_init("x", &x);
        se.forward(&ctx, &ps, &xb);
        ps.zero_grads(&gpu);
        let d_out = gpu.storage_init("dout", &r);
        let d_in = gpu.storage(tot as u64);
        se.backward(&ctx, &ps, &xb, &d_out, &d_in);

        let loss = |gpu: &Gpu, ps: &ParamStore| -> f32 {
            let ctx = Ctx::new(gpu, zipdepth::net::ids());
            let se = ChannelAttention::new(&ctx, "se", shape);
            let xb = gpu.storage_init("x", &x);
            se.forward(&ctx, ps, &xb);
            gpu.read(se.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
        };

        for wname in ["se.fc.0.weight", "se.fc.2.weight"] {
            let g = gpu.read(ps.g(wname), ps.numel(wname));
            let n = g.len();
            let dir: Vec<f32> = Lcg::new(7).vec(n).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
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
            assert!(abs <= 4e-3 + 8e-2 * denom, "{wname} @ {shape:?}: analytic {analytic}, fd {numeric}");
        }

        // ...and d_in, which sums TWO paths (the pool and the multiply). A missing
        // path here is the classic SE bug and the weight grads would not catch it.
        let dg = gpu.read(&d_in, tot);
        let dir: Vec<f32> = Lcg::new(8).vec(tot).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
        let analytic: f32 = dg.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let eps = 5e-4f32;
        let lp = {
            let xp: Vec<f32> = x.iter().zip(&dir).map(|(v, d)| v + eps * d).collect();
            let ctx = Ctx::new(&gpu, zipdepth::net::ids());
            let se = ChannelAttention::new(&ctx, "se", shape);
            let xb = gpu.storage_init("x", &xp);
            se.forward(&ctx, &ps, &xb);
            gpu.read(se.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum::<f32>()
        };
        let lm = {
            let xm: Vec<f32> = x.iter().zip(&dir).map(|(v, d)| v - eps * d).collect();
            let ctx = Ctx::new(&gpu, zipdepth::net::ids());
            let se = ChannelAttention::new(&ctx, "se", shape);
            let xb = gpu.storage_init("x", &xm);
            se.forward(&ctx, &ps, &xb);
            gpu.read(se.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum::<f32>()
        };
        let numeric = (lp - lm) / (2.0 * eps);
        let abs = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        assert!(abs <= 4e-3 + 8e-2 * denom, "d_in @ {shape:?}: analytic {analytic}, fd {numeric} — one of the two x-paths is missing");
    }
}
