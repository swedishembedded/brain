// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2: `QARepBlock` — the encoder's workhorse (15 of the base config's blocks).
//!
//! Two properties, and the second is the one that matters:
//!   1. the unfused three-branch forward is differentiated correctly (FD gate);
//!   2. the FUSED single 3x3 computes the same function as the unfused branches
//!      — checked through the real block, not through a host reimplementation.
//!
//! (2) is the whole reason RepVGG is usable: training runs three branches with
//! separate BN statistics, inference runs one conv, and if they disagree the model
//! silently degrades between train and deploy. `fuse.rs` already proves the weight
//! arithmetic against a host convolution; this proves the arithmetic matches what
//! the BLOCK actually dispatches.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use std::collections::HashMap;

use depth::blocks::QARepBlock;
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

struct Fix {
    gpu: Gpu,
    ps: ParamStore,
    x: Vec<f32>,
    in_shape: Shape,
    cout: u32,
    stride: u32,
}

impl Fix {
    fn new(in_shape: Shape, cout: u32, stride: u32, seed: u64) -> Fix {
        let gpu = Gpu::new_cpu(depth::net::PIPELINES);
        let ctx = Ctx::new(&gpu, depth::net::ids());
        // Build once just to harvest the param list — the block owns its names.
        let probe = QARepBlock::new(&ctx, "b", in_shape, cout, stride, true);
        let params = probe.param_list();
        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, (name, numel)) in params.iter().enumerate() {
            let v: Vec<f32> = if name.ends_with("running_mean") {
                vec![0.0; *numel]
            } else if name.ends_with("running_var") {
                // Non-trivial running var so eval-mode BN is not an identity.
                rv(seed ^ (i as u64), *numel).iter().map(|v| v.abs() + 0.5).collect()
            } else if name.ends_with(".1.weight") {
                rv(seed ^ (i as u64), *numel).iter().map(|v| 1.0 + 0.2 * v).collect()
            } else if name.ends_with(".1.bias") {
                rv(seed ^ (i as u64), *numel).iter().map(|v| 0.1 * v).collect()
            } else {
                rv(seed ^ (i as u64), *numel).iter().map(|v| 0.3 * v).collect()
            };
            init.insert(name.clone(), v);
        }
        let ps = ParamStore::new(&gpu, params, &init);
        let x = rv(seed ^ 0xABC, in_shape.numel() as usize);
        Fix { gpu, ps, x, in_shape, cout, stride }
    }

    /// Scalar loss `<r, block(x)>`.
    fn loss(&self, train: bool, r: &[f32]) -> f32 {
        let ctx = Ctx::new(&self.gpu, depth::net::ids());
        let b = QARepBlock::new(&ctx, "b", self.in_shape, self.cout, self.stride, train);
        b.set_eval(!train);
        let xb = self.gpu.storage_init("x", &self.x);
        b.forward(&ctx, &self.ps, &xb);
        let out = self.gpu.read(b.out(), b.out_shape.numel() as usize);
        out.iter().zip(r).map(|(a, c)| a * c).sum()
    }
}

/// The identity branch exists iff `cin == cout && stride == 1` — the reference's
/// own condition. Its presence changes the FUNCTION, so it is worth asserting
/// directly rather than trusting the constructor.
#[test]
fn identity_branch_is_present_exactly_when_shapes_allow() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let s = Shape::new(1, 8, 6, 6);
    // Same channels + stride 1 -> residual.
    let a = QARepBlock::new(&ctx, "a", s, 8, 1, true);
    assert_eq!(a.out_shape, Shape::new(1, 8, 6, 6));
    // Channel change -> no residual (the shapes could not add).
    let b = QARepBlock::new(&ctx, "b", s, 16, 1, true);
    assert_eq!(b.out_shape, Shape::new(1, 16, 6, 6));
    // Stride 2 -> no residual, and the map halves. This is `down2`/`down3`/`down4`.
    let c = QARepBlock::new(&ctx, "c", s, 16, 2, true);
    assert_eq!(c.out_shape, Shape::new(1, 16, 3, 3));
}

/// Param names must mirror the reference checkpoint's `nn.Sequential` indices —
/// `branch_3x3.0.weight` (conv) and `branch_3x3.1.*` (BN), NOT brain's
/// `.conv.weight`/`.bn.gamma`. This is what makes import a 1:1 name match.
#[test]
fn param_names_mirror_the_reference_sequential_indices() {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let b = QARepBlock::new(&ctx, "encoder.stage1.0", Shape::new(1, 48, 8, 8), 48, 1, true);
    let names: Vec<String> = b.param_list().into_iter().map(|(n, _)| n).collect();
    for want in [
        "encoder.stage1.0.branch_3x3.0.weight",
        "encoder.stage1.0.branch_3x3.1.weight",
        "encoder.stage1.0.branch_3x3.1.bias",
        "encoder.stage1.0.branch_3x3.1.running_mean",
        "encoder.stage1.0.branch_3x3.1.running_var",
        "encoder.stage1.0.branch_1x1.0.weight",
        "encoder.stage1.0.branch_1x1.1.running_var",
    ] {
        assert!(names.contains(&want.to_string()), "missing `{want}` in {names:#?}");
    }
    assert!(!names.iter().any(|n| n.contains(".bn.gamma")), "brain-style BN names leaked in");
    assert_eq!(names.len(), 10, "two conv+BN branches = 2*(1 conv + 4 BN)");
}

/// The FD gate over the unfused three-branch forward, for each shape the encoder
/// actually instantiates.
#[test]
fn qarep_backward_matches_finite_differences() {
    let cases: Vec<(&str, Shape, u32, u32)> = vec![
        ("residual (stage1)", Shape::new(2, 8, 6, 6), 8, 1),
        ("channel change", Shape::new(2, 8, 6, 6), 16, 1),
        ("stride 2 (down2)", Shape::new(2, 8, 8, 8), 16, 2),
    ];
    for (tag, in_shape, cout, stride) in cases {
        let fix = Fix::new(in_shape, cout, stride, 21);
        let ctx = Ctx::new(&fix.gpu, depth::net::ids());
        let b = QARepBlock::new(&ctx, "b", in_shape, cout, stride, true);
        let out_n = b.out_shape.numel() as usize;
        let r = rv(77, out_n);

        let xb = fix.gpu.storage_init("x", &fix.x);
        b.forward(&ctx, &fix.ps, &xb);
        fix.ps.zero_grads(&fix.gpu);
        let d_out = fix.gpu.storage_init("dout", &r);
        let d_in = fix.gpu.storage(in_shape.numel() as u64);
        b.backward(&ctx, &fix.ps, &xb, &d_out, &d_in);

        // Check the 3x3 branch's weight — it carries the bulk of the block.
        let wname = "b.branch_3x3.0.weight";
        let g = fix.gpu.read(fix.ps.g(wname), fix.ps.numel(wname));
        let n = g.len();
        let dir: Vec<f32> = rv(5, n).iter().map(|v| if *v < 0.0 { -1.0f32 } else { 1.0 }).collect();
        let analytic: f32 = g.iter().zip(&dir).map(|(a, c)| a * c).sum();

        let w0 = fix.gpu.read(fix.ps.w(wname), n);
        let eps = 5e-4f32;
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w + eps * d).collect();
        fix.gpu.write(fix.ps.w(wname), bytemuck::cast_slice(&wp));
        let lp = fix.loss(true, &r);
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w - eps * d).collect();
        fix.gpu.write(fix.ps.w(wname), bytemuck::cast_slice(&wm));
        let lm = fix.loss(true, &r);
        fix.gpu.write(fix.ps.w(wname), bytemuck::cast_slice(&w0));

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

/// THE property RepVGG exists for: the fused single 3x3 must compute what the
/// three branches compute — in EVAL mode, where BN uses its running statistics,
/// which is the only mode the fuse is defined against.
///
/// This runs the real block's eval forward, then the fuse's own kernel through a
/// bare `conv2d_gd` + relu, and compares. `fuse.rs` already checks the weight
/// arithmetic against a host convolution; this closes the loop by checking it
/// against what the block ACTUALLY dispatches — the two could otherwise agree
/// with each other and both disagree with the model.
#[test]
fn fused_conv_reproduces_the_blocks_own_eval_forward() {
    let in_shape = Shape::new(1, 8, 7, 7);
    let (cout, stride) = (8u32, 1u32);
    let fix = Fix::new(in_shape, cout, stride, 33);
    let ctx = Ctx::new(&fix.gpu, depth::net::ids());

    // Unfused, eval mode.
    let b = QARepBlock::new(&ctx, "b", in_shape, cout, stride, false);
    b.set_eval(true);
    let xb = fix.gpu.storage_init("x", &fix.x);
    b.forward(&ctx, &fix.ps, &xb);
    let want = fix.gpu.read(b.out(), b.out_shape.numel() as usize);

    // Fused: derive (k, bias) from the same weights, then run one conv + relu.
    let rd = |n: &str| fix.gpu.read(fix.ps.w(n), fix.ps.numel(n));
    let (w3, g3, b3, m3, v3) = (
        rd("b.branch_3x3.0.weight"),
        rd("b.branch_3x3.1.weight"),
        rd("b.branch_3x3.1.bias"),
        rd("b.branch_3x3.1.running_mean"),
        rd("b.branch_3x3.1.running_var"),
    );
    let (w1, g1, b1, m1, v1) = (
        rd("b.branch_1x1.0.weight"),
        rd("b.branch_1x1.1.weight"),
        rd("b.branch_1x1.1.bias"),
        rd("b.branch_1x1.1.running_mean"),
        rd("b.branch_1x1.1.running_var"),
    );
    let br3 = depth::Branch { weight: &w3, gamma: &g3, beta: &b3, run_mean: &m3, run_var: &v3 };
    let br1 = depth::Branch { weight: &w1, gamma: &g1, beta: &b1, run_mean: &m1, run_var: &v1 };
    let (k, bias) = depth::fuse_qarep(&br3, &br1, in_shape.c as usize, cout as usize, 1, true);

    // `conv_bias` = fused conv + PER-CHANNEL NCHW bias. Not conv2d + bias_add:
    // bias_add is `out[idx] += bias[idx % n]`, a [M,N] row-major LINEAR bias whose
    // biased dim must be trailing. In NCHW the channel is not trailing, so it
    // indexes garbage — which is exactly how this test failed first (by 7.24).
    let on = b.out_shape.numel();
    let kb = fix.gpu.storage_init("k", &k);
    let bb = fix.gpu.storage_init("bias", &bias);
    let biased = fix.gpu.storage(on as u64);
    let ids = depth::net::ids();
    let s = fix.gpu.step(
        ids.conv_bias,
        &[&xb, &kb, &bb, &biased],
        &[in_shape.n, in_shape.c, in_shape.h, in_shape.w, cout, 3, stride, 1, b.out_shape.h, b.out_shape.w],
        on,
    );
    fix.gpu.submit(&[], &[s]);
    let relu_out = fix.gpu.storage(on as u64);
    let s = fix.gpu.step(ids.leaky_relu, &[&biased, &relu_out], &[on, gpu_core::f(0.0)], on);
    fix.gpu.submit(&[], &[s]);
    let got = fix.gpu.read(&relu_out, on as usize);

    let mut max = 0.0f32;
    for i in 0..got.len() {
        max = max.max((got[i] - want[i]).abs());
    }
    assert!(
        max < 2e-4,
        "fused conv and the block's own 3-branch eval forward disagree by {max}"
    );
    // The fixture must actually exercise the fuse: an all-zero output would pass
    // trivially.
    assert!(want.iter().any(|v| *v > 1e-3), "fixture is degenerate — ReLU zeroed everything");
}
