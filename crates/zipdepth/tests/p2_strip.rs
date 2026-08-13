// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: `StripPoolingAttention` — `x * sigmoid(BN(dw1x1(mean_W(x) + mean_H(x))))`.
//!
//! The first block where the INPUT gradient is the interesting one: `x` reaches the
//! output by three routes (both strips and the final multiply), so a dropped route
//! is invisible in the weight grads and shows up only in `d_in`.
use data::rng::Lcg;
use std::collections::HashMap;

use zipdepth::blocks::StripPoolingAttention;
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{Ctx, Shape};

fn fixture(shape: Shape, seed: u64) -> (Gpu, ParamStore, Vec<f32>) {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let probe = StripPoolingAttention::new(&ctx, "s", shape, true);
    let params = probe.param_list();
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (n, numel)) in params.iter().enumerate() {
        let v: Vec<f32> = if n.ends_with("running_mean") {
            vec![0.0; *numel]
        } else if n.ends_with("running_var") {
            vec![1.0; *numel]
        } else if n.ends_with("gate_conv.1.weight") {
            Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 1.0 + 0.2 * v).collect()
        } else if n.ends_with("gate_conv.1.bias") {
            Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 0.1 * v).collect()
        } else {
            Lcg::new(seed ^ i as u64).vec(*numel).iter().map(|v| 0.5 * v).collect()
        };
        init.insert(n.clone(), v);
    }
    let ps = ParamStore::new(&gpu, params, &init);
    (gpu, ps, Lcg::new(seed ^ 0xD).vec(shape.numel() as usize))
}

/// `nn.Sequential(Conv2d(dim,dim,1,groups=dim,bias=False), BatchNorm2d, Sigmoid)`.
/// Depthwise means the weight is `[dim, 1, 1, 1]` — ONE scalar per channel, not
/// `dim*dim`. Getting `groups` wrong here would inflate the tensor 96x and fail the
/// checkpoint's strict load, which is exactly what this pins.
#[test]
fn strip_gate_is_a_biasless_depthwise_1x1_plus_bn() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let s = StripPoolingAttention::new(&ctx, "encoder.stage3.1", Shape::new(1, 192, 24, 24), true);
    assert_eq!(
        s.param_list(),
        vec![
            ("encoder.stage3.1.gate_conv.0.weight".to_string(), 192),
            ("encoder.stage3.1.gate_conv.1.weight".to_string(), 192),
            ("encoder.stage3.1.gate_conv.1.bias".to_string(), 192),
            ("encoder.stage3.1.gate_conv.1.running_mean".to_string(), 192),
            ("encoder.stage3.1.gate_conv.1.running_var".to_string(), 192),
        ],
        "depthwise 1x1 -> [C,1,1,1]; bias=False -> no `.0.bias`"
    );
}

/// The gate is elementwise with `x`, so the block is shape-preserving at any shape
/// — including the non-square ones the strips make it easy to transpose.
#[test]
fn strip_is_shape_preserving() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    for sh in [Shape::new(1, 192, 24, 24), Shape::new(2, 8, 7, 5)] {
        let s = StripPoolingAttention::new(&ctx, "s", sh, true);
        assert_eq!(s.shape, sh);
    }
}

#[test]
fn strip_weight_grads_match_finite_differences() {
    let shape = Shape::new(2, 8, 6, 5);
    let (gpu, ps, x) = fixture(shape, 23);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = StripPoolingAttention::new(&ctx, "s", shape, true);
    let tot = shape.numel() as usize;
    let r = Lcg::new(43).vec(tot);

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let d_in = gpu.storage(tot as u64);
    m.backward(&ctx, &ps, &xb, &d_out, &d_in);

    let loss = |gpu: &Gpu, ps: &ParamStore| -> f32 {
        let ctx = Ctx::new(gpu, zipdepth::net::ids());
        let m = StripPoolingAttention::new(&ctx, "s", shape, true);
        let xb = gpu.storage_init("x", &x);
        m.forward(&ctx, ps, &xb);
        gpu.read(m.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    for wname in ["s.gate_conv.0.weight", "s.gate_conv.1.weight"] {
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

/// THE test for this block. `x` feeds the h-strip, the w-strip and the multiply;
/// dropping any one route still leaves every weight gradient correct, because each
/// weight sits downstream of the join. Only `d_in` sees the difference.
///
/// Perturbing a single element (not a direction) is deliberate: a whole-tensor
/// direction sums 480 elements, and a dropped strip route contributes a term that
/// is ~1/H of the multiply's — large enough to matter, small enough for the sum to
/// mask it.
#[test]
fn strip_input_grad_matches_finite_differences_elementwise() {
    let shape = Shape::new(2, 4, 5, 3);
    let (gpu, ps, x) = fixture(shape, 29);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = StripPoolingAttention::new(&ctx, "s", shape, true);
    let tot = shape.numel() as usize;
    let r = Lcg::new(47).vec(tot);

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d_out = gpu.storage_init("dout", &r);
    let d_in = gpu.storage(tot as u64);
    m.backward(&ctx, &ps, &xb, &d_out, &d_in);
    let g = gpu.read(&d_in, tot);

    let loss = |xv: &[f32]| -> f32 {
        let ctx = Ctx::new(&gpu, zipdepth::net::ids());
        let m = StripPoolingAttention::new(&ctx, "s", shape, true);
        let xb = gpu.storage_init("x", xv);
        m.forward(&ctx, &ps, &xb);
        gpu.read(m.out(), tot).iter().zip(&r).map(|(a, b)| a * b).sum()
    };
    let eps = 1e-3f32;
    for i in [0usize, 7, 31, 59, 91, 113] {
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
