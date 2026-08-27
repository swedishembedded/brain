// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: the shared `vision::Conv`, driven through ZipDepth's spec — grouped,
//! dilated, depthwise, ReLU.
//!
//! This is the test that proves P1's whole point: ONE conv block serves both
//! models. yolo exercises it dense + SiLU + fused; ZipDepth exercises it
//! grouped/dilated + ReLU + unfused, and both must be correct. yolo's side is
//! already pinned bitwise (`yolo/tests/p1_forward_pin.rs`); this covers the paths
//! only ZipDepth reaches.
//!
//! Lives in `crates/zipdepth`, not `crates/vision`, because the block's kernels are
//! resolved from the OWNING MODEL's `PIPELINES` — and `conv2d_gd`/`leaky_relu`
//! are registered by depth, not by yolo. Testing it here is testing it as it is
//! actually used.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use data::rng::Lcg;
use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::blocks::{Act, Conv, ConvSpec};
use vision::{Ctx, Shape};

/// Build a `Conv` from a spec with deterministic weights, and return everything
/// needed to drive it.
struct Fix {
    gpu: Gpu,
    ps: ParamStore,
    x: Vec<f32>,
    in_shape: Shape,
}

impl Fix {
    fn new(in_shape: Shape, spec: ConvSpec, seed: u64) -> (Fix, ConvSpec) {
        let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
        let cin_g = in_shape.c / spec.groups;
        let params: Vec<(String, usize)> = vec![
            ("c.conv.weight".into(), (spec.cout * cin_g * spec.k * spec.k) as usize),
            ("c.bn.gamma".into(), spec.cout as usize),
            ("c.bn.beta".into(), spec.cout as usize),
            ("c.bn.run_mean".into(), spec.cout as usize),
            ("c.bn.run_var".into(), spec.cout as usize),
        ];
        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        init.insert("c.conv.weight".into(), Lcg::new(seed).vec(params[0].1));
        init.insert("c.bn.gamma".into(), Lcg::new(seed ^ 1).vec(spec.cout as usize).iter().map(|v| 1.0 + 0.1 * v).collect());
        init.insert("c.bn.beta".into(), Lcg::new(seed ^ 2).vec(spec.cout as usize).iter().map(|v| 0.05 * v).collect());
        init.insert("c.bn.run_mean".into(), vec![0.0; spec.cout as usize]);
        init.insert("c.bn.run_var".into(), vec![1.0; spec.cout as usize]);
        let ps = ParamStore::new(&gpu, params, &init);
        let x = Lcg::new(seed ^ 3).vec(in_shape.numel() as usize);
        (Fix { gpu, ps, x, in_shape }, spec)
    }
}

/// Scalar loss `<r, conv(x)>` for a fixed random `r`, so FD has something to
/// differentiate.
fn loss(fix: &Fix, spec: ConvSpec, train: bool, r: &[f32]) -> f32 {
    let ctx = Ctx::new(&fix.gpu, zipdepth::net::ids());
    let c = Conv::with_spec(&ctx, "c", fix.in_shape, spec, train);
    let xb = fix.gpu.storage_init("x", &fix.x);
    c.forward(&ctx, &fix.ps, &xb);
    let out = fix.gpu.read(c.out(), c.out_shape.numel() as usize);
    out.iter().zip(r).map(|(a, b)| a * b).sum()
}

/// Shapes must follow the dilated formula, not the dense one.
#[test]
fn grouped_dilated_shapes_are_right() {
    let s = Shape::new(1, 96, 48, 48);
    // ZipDepth's MinimalMultiScale: depthwise 3x3 dilation 2, pad 2 -> same size.
    let dil = ConvSpec::depthwise(96, 3, 1, 2, Act::Relu).with_dilation(2);
    assert_eq!(dil.out_shape(s), Shape::new(1, 96, 48, 48));
    // ...and dilation 1 pad 1 likewise.
    let d1 = ConvSpec::depthwise(96, 3, 1, 1, Act::Relu);
    assert_eq!(d1.out_shape(s), Shape::new(1, 96, 48, 48));
    // A grouped 1x1 projection.
    let g = ConvSpec::relu(192, 1, 1, 0).with_groups(4);
    assert_eq!(g.out_shape(Shape::new(1, 384, 12, 12)), Shape::new(1, 192, 12, 12));
}

/// `is_dense` decides which conv kernel is dispatched, and getting it wrong is
/// silent: the dense CPU fast path ignores `groups` and returns plausible numbers.
#[test]
fn is_dense_gates_the_fast_path_correctly() {
    assert!(ConvSpec::silu(16, 3, 1, 1).is_dense());
    assert!(!ConvSpec::relu(16, 3, 1, 1).with_groups(4).is_dense());
    assert!(!ConvSpec::relu(16, 3, 1, 1).with_dilation(2).is_dense());
    assert!(!ConvSpec::depthwise(16, 3, 1, 1, Act::Relu).is_dense());
}

/// The real gate: analytic weight gradients vs central differences, in TRAIN
/// mode, for each spec shape ZipDepth actually uses.
#[test]
fn grouped_relu_conv_backward_matches_finite_differences() {
    let cases: Vec<(&str, Shape, ConvSpec)> = vec![
        ("dense relu 3x3", Shape::new(2, 8, 6, 6), ConvSpec::relu(8, 3, 1, 1)),
        ("grouped 1x1 g=4", Shape::new(2, 8, 5, 5), ConvSpec::relu(8, 1, 1, 0).with_groups(4)),
        ("depthwise 3x3", Shape::new(2, 6, 5, 5), ConvSpec::depthwise(6, 3, 1, 1, Act::Relu)),
        ("depthwise 3x3 dil=2", Shape::new(2, 6, 7, 7), ConvSpec::depthwise(6, 3, 1, 2, Act::Relu).with_dilation(2)),
        ("strided relu 3x3 s=2", Shape::new(2, 4, 8, 8), ConvSpec::relu(8, 3, 2, 1)),
        ("no-act conv+bn", Shape::new(2, 4, 5, 5), ConvSpec::relu(4, 3, 1, 1).with_act(Act::None)),
    ];
    for (tag, in_shape, spec) in cases {
        let (fix, spec) = Fix::new(in_shape, spec, 11);
        let out_n = spec.out_shape(in_shape).numel() as usize;
        let r = Lcg::new(99).vec(out_n);

        // Analytic dW.
        let ctx = Ctx::new(&fix.gpu, zipdepth::net::ids());
        let c = Conv::with_spec(&ctx, "c", in_shape, spec, true);
        let xb = fix.gpu.storage_init("x", &fix.x);
        c.forward(&ctx, &fix.ps, &xb);
        fix.ps.zero_grads(&fix.gpu);
        let d_out = fix.gpu.storage_init("dout", &r);
        let d_in = fix.gpu.storage(in_shape.numel() as u64);
        c.backward(&ctx, &fix.ps, &xb, &d_out, &d_in);
        let g = fix.gpu.read(fix.ps.g("c.conv.weight"), fix.ps.numel("c.conv.weight"));

        // FD over a random ±1 direction (the same directional trick gradcheck
        // uses: it averages per-element round-off into one well-conditioned
        // number instead of probing 1e3 elements individually).
        let n = g.len();
        let dir: Vec<f32> = Lcg::new(123).vec(n).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
        let analytic: f32 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let w0 = fix.gpu.read(fix.ps.w("c.conv.weight"), n);
        let eps = 5e-4f32;
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w + eps * d).collect();
        fix.gpu.write(fix.ps.w("c.conv.weight"), bytemuck::cast_slice(&wp));
        let lp = loss(&fix, spec, true, &r);
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w - eps * d).collect();
        fix.gpu.write(fix.ps.w("c.conv.weight"), bytemuck::cast_slice(&wm));
        let lm = loss(&fix, spec, true, &r);
        fix.gpu.write(fix.ps.w("c.conv.weight"), bytemuck::cast_slice(&w0));

        let numeric = (lp - lm) / (2.0 * eps);
        let abs = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        assert!(
            abs <= 4e-3 + 8e-2 * denom,
            "{tag}: analytic {analytic}, fd {numeric} (abs {abs}, rel {})",
            abs / denom
        );
    }
}

/// ReLU must actually clamp: a unit whose output is all-positive is not testing
/// the activation at all, so assert the forward really has zeros where the
/// pre-activation was negative.
#[test]
fn relu_units_actually_clamp() {
    let in_shape = Shape::new(2, 8, 6, 6);
    let (fix, spec) = Fix::new(in_shape, ConvSpec::relu(8, 3, 1, 1), 5);
    let ctx = Ctx::new(&fix.gpu, zipdepth::net::ids());
    let c = Conv::with_spec(&ctx, "c", in_shape, spec, true);
    let xb = fix.gpu.storage_init("x", &fix.x);
    c.forward(&ctx, &fix.ps, &xb);
    let out = fix.gpu.read(c.out(), c.out_shape.numel() as usize);
    assert!(out.iter().all(|v| *v >= 0.0), "ReLU output must be non-negative");
    let zeros = out.iter().filter(|v| **v == 0.0).count();
    assert!(zeros > 0, "no output was clamped — the fixture does not exercise ReLU");
    assert!(zeros < out.len(), "everything was clamped — the fixture is degenerate");
}

/// Train and eval BN differ (batch stats vs running estimates), and BOTH must run
/// for a grouped ReLU unit — eval takes the unfused conv -> bn_eval -> act path,
/// which nothing dispatched before ZipDepth existed.
#[test]
fn grouped_relu_runs_on_both_the_train_and_eval_paths() {
    let in_shape = Shape::new(2, 8, 6, 6);
    let spec = ConvSpec::relu(8, 1, 1, 0).with_groups(4);
    let (fix, spec) = Fix::new(in_shape, spec, 13);
    let out_n = spec.out_shape(in_shape).numel() as usize;
    let r = vec![1.0f32; out_n];

    let lt = loss(&fix, spec, true, &r);
    let le = loss(&fix, spec, false, &r);
    assert!(lt.is_finite() && le.is_finite(), "train {lt}, eval {le}");
    // run_mean=0 / run_var=1 with a non-trivial batch => the two paths normalize
    // by different statistics and must not coincide.
    assert!((lt - le).abs() > 1e-6, "train and eval BN produced the same result ({lt} vs {le}) — one path is not running");
}
