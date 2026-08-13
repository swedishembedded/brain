// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fused conv->BN(eval)->act path, as ZipDepth actually uses it.
//!
//! Since the act selector landed in `conv_act*` (0 identity, 1 relu, 2 silu,
//! 3 sigmoid), a ReLU model fuses its dense+BN units exactly like yolo's SiLU
//! ones: ONE `conv_act_reg` dispatch instead of conv2d + bn_eval + leaky_relu —
//! three full-tensor passes collapsed into one, and ~8x less input traffic on
//! the GPU. These tests pin, against depth's own `PIPELINES`:
//!
//!   1. that the fused path is actually TAKEN for ZipDepth's unit shapes (a
//!      silent fall-back to unfused is a perf regression no output comparison
//!      catches), and that grouped units correctly do NOT take it;
//!   2. that fused output == unfused output for every act ZipDepth uses
//!      (Relu, None, Sigmoid) — the unfused reference runs on a registry with
//!      the fused kernels REMOVED, which is exactly yesterday's engine;
//!   3. same equivalence on the real GPU (wgpu), gated by `MOE_SKIP_GPU_TESTS`,
//!      which is what validates the WGSL act-selector branches end to end.
//!
//! Lives in `crates/depth` (not `crates/vision`) because kernels resolve from
//! the OWNING model's `PIPELINES` — same reasoning as `p2_conv_spec.rs`.
//!
//! Run with `BRAIN_DEVICE=cpu` (test 3 needs a real GPU and skips without one).

use data::rng::Lcg;
use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::blocks::{Act, Conv, ConvSpec};
use vision::{ConvKernelIds, Ctx, Shape};

/// Cross-implementation agreement, measured **against the tensor's own scale**:
/// `max|a-b| / max|b|`.
///
/// NOT a per-element `|a-b| / (|b| + 1e-4)`. These outputs span four decades
/// (|out| runs from ~1e-4 to ~9), and every element carries the same *absolute*
/// round-off — the one produced by summing the largest terms in the
/// accumulation, ~2.9e-6 here. A per-element ratio with a 1e-4 floor therefore
/// reports 2e-4 for an element that merely happened to land near zero (a
/// legitimate 2.1e-7 difference on a -9.1e-4 value) while allowing a 9.3e-4
/// difference on an element of magnitude 9.3 — it is loose exactly where the
/// error lives and tight exactly where it does not. That is why it failed the
/// moment the test RNG stopped being one-sided and outputs started landing near
/// zero; the arithmetic never changed.
///
/// Measured worst case on a P40 (wgpu) vs the CPU JIT, over every spec below:
/// 3.4e-7. The bounds are set an order above that.
fn scale_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / scale
}

/// depth's PIPELINES with the fused eval kernels removed — the unfused
/// reference engine (what every ZipDepth conv ran before the act selector).
fn pipelines_without_fused() -> Vec<(&'static str, &'static str)> {
    zipdepth::net::PIPELINES
        .iter()
        .copied()
        .filter(|(n, _)| !n.starts_with("conv_act"))
        .collect()
}

/// Deterministic params for one `Conv` named `c`, with running stats that make
/// BN-eval a real transform (non-zero mean, non-unit var), so a fused/unfused
/// mismatch in the affine collapse cannot hide behind identity stats.
fn params_for(spec: &ConvSpec, cin: u32, seed: u64) -> (Vec<(String, usize)>, HashMap<String, Vec<f32>>) {
    let cin_g = cin / spec.groups;
    let wlen = (spec.cout * cin_g * spec.k * spec.k) as usize;
    let c = spec.cout as usize;
    let params: Vec<(String, usize)> = vec![
        ("c.conv.weight".into(), wlen),
        ("c.bn.gamma".into(), c),
        ("c.bn.beta".into(), c),
        ("c.bn.run_mean".into(), c),
        ("c.bn.run_var".into(), c),
    ];
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    init.insert("c.conv.weight".into(), Lcg::new(seed).vec(wlen));
    init.insert("c.bn.gamma".into(), Lcg::new(seed ^ 1).vec(c).iter().map(|v| 1.0 + 0.2 * v).collect());
    init.insert("c.bn.beta".into(), Lcg::new(seed ^ 2).vec(c).iter().map(|v| 0.1 * v).collect());
    init.insert("c.bn.run_mean".into(), Lcg::new(seed ^ 4).vec(c).iter().map(|v| 0.3 * v).collect());
    init.insert("c.bn.run_var".into(), Lcg::new(seed ^ 5).vec(c).iter().map(|v| 1.0 + 0.5 * v.abs()).collect());
    (params, init)
}

/// Eval-mode forward of one `Conv` on the given engine; returns the output.
fn eval_forward(gpu: &Gpu, ids: &ConvKernelIds, in_shape: Shape, spec: ConvSpec, seed: u64) -> Vec<f32> {
    let (params, init) = params_for(&spec, in_shape.c, seed);
    let ps = ParamStore::new(gpu, params, &init);
    let ctx = Ctx::new(gpu, ids);
    let c = Conv::with_spec(&ctx, "c", in_shape, spec, false);
    c.set_eval(true);
    let x = gpu.storage_init("x", &Lcg::new(seed ^ 3).vec(in_shape.numel() as usize));
    c.forward(&ctx, &ps, &x);
    gpu.read(c.out(), c.out_shape.numel() as usize)
}

/// ZipDepth-shaped dense specs, one per act the model uses. (SiLU is yolo's,
/// pinned bitwise by `yolo/tests/p1_forward_pin.rs`.)
fn dense_specs() -> Vec<ConvSpec> {
    vec![
        ConvSpec::relu(16, 3, 2, 1),                       // stem-like ReLU 3x3 s2
        ConvSpec { act: Act::None, ..ConvSpec::relu(12, 3, 1, 1) }, // QARep branch: BN, no act
        ConvSpec { act: Act::Sigmoid, ..ConvSpec::relu(8, 1, 1, 0) }, // gate-producing 1x1
    ]
}

/// 1. The fused path is TAKEN for dense+BN units of every act — and NOT taken
///    for grouped ones (the fused kernels are dense; binding a grouped unit
///    would silently ignore `groups`).
#[test]
fn dense_bn_units_take_the_fused_path_grouped_do_not() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let in_shape = Shape::new(1, 8, 10, 10);
    for spec in dense_specs() {
        let c = Conv::with_spec(&ctx, "c", in_shape, spec, false);
        assert!(
            c.can_fuse(&ctx),
            "dense+BN {:?} unit must take the fused conv_act_reg path",
            c.spec.act
        );
    }
    // Depthwise 3x3 (groups = cin): must stay unfused.
    let dw = ConvSpec { groups: 8, ..ConvSpec::relu(8, 3, 1, 1) };
    let c = Conv::with_spec(&ctx, "c", in_shape, dw, false);
    assert!(!c.can_fuse(&ctx), "grouped units must NOT take the dense fused path");
}

/// 2. Fused == unfused on the CPU backend, for every act ZipDepth uses. The
///    reference engine strips `conv_act*` from the registry, which forces the
///    conv -> bn_eval -> act path — the two differ only in arithmetic order
///    (collapsed scale/bias vs per-element normalize), so agreement is to
///    round-off, not to a loose model tolerance.
#[test]
fn fused_eval_matches_unfused_reference() {
    let fused_gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let plain = pipelines_without_fused();
    let plain_gpu = Gpu::new_cpu(&plain);
    let plain_ids = ConvKernelIds::resolve(&plain);
    let in_shape = Shape::new(2, 8, 12, 12); // n=2: catches a per-image indexing slip
    for (i, spec) in dense_specs().into_iter().enumerate() {
        let seed = 11 + i as u64;
        let f = eval_forward(&fused_gpu, zipdepth::net::ids(), in_shape, spec, seed);
        let u = eval_forward(&plain_gpu, &plain_ids, in_shape, spec, seed);
        assert_eq!(f.len(), u.len());
        let max_rel = scale_rel(&f, &u);
        assert!(
            max_rel < 5e-6,
            "fused vs unfused rel err {max_rel} for act {:?}",
            spec.act
        );
    }
}

/// 3. Same equivalence with the fused side on the REAL GPU (wgpu) — this is
///    what executes the WGSL act-selector branches (the CPU backend intercepts
///    `conv_act*` with its native fast path, so only a GPU run proves the WGSL).
#[test]
fn fused_eval_gpu_matches_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("skipping fused GPU parity (MOE_SKIP_GPU_TESTS)");
        return;
    }
    let gpu = Gpu::new_wgpu(zipdepth::net::PIPELINES);
    let cpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let in_shape = Shape::new(1, 8, 12, 12);
    for (i, spec) in dense_specs().into_iter().enumerate() {
        let seed = 29 + i as u64;
        let g = eval_forward(&gpu, zipdepth::net::ids(), in_shape, spec, seed);
        let c = eval_forward(&cpu, zipdepth::net::ids(), in_shape, spec, seed);
        let max_rel = scale_rel(&g, &c);
        assert!(max_rel < 5e-6, "GPU vs CPU fused rel err {max_rel} for act {:?}", spec.act);
    }
}

/// 5. The QARep RepVGG collapse at eval: the whole block — two conv+BN
///    branches (+ identity) + ReLU — runs as ONE fused dispatch whose output
///    matches the unfused block to round-off. The reference engine strips
///    `conv_act*`, which forces the branch-by-branch path (`eval_fused` is
///    false there, which doubles as the negative path check). All three block
///    geometries: residual (identity), downsample (stride 2), channel change.
#[test]
fn qarep_fused_eval_matches_unfused() {
    use zipdepth::blocks::QARepBlock;

    let fused_gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let plain = pipelines_without_fused();
    let plain_gpu = Gpu::new_cpu(&plain);
    let plain_ids = ConvKernelIds::resolve(&plain);

    let block_forward = |gpu: &Gpu, ids: &ConvKernelIds, in_shape: Shape, cout: u32, stride: u32, seed: u64, expect_fused: bool| -> Vec<f32> {
        let ctx = Ctx::new(gpu, ids);
        let blk = QARepBlock::new(&ctx, "q", in_shape, cout, stride, false);
        blk.set_eval(true);
        assert_eq!(
            blk.eval_fused(&ctx),
            expect_fused,
            "fused-path selection (expect_fused={expect_fused})"
        );
        let mut s = Lcg::new(seed);
        let mut mapped = |n: usize, f: &dyn Fn(f32) -> f32| -> Vec<f32> {
            (0..n).map(|_| f(s.signed())).collect()
        };
        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, numel) in blk.param_list() {
            let v = if name.ends_with(".1.weight") {
                mapped(numel, &|r| 1.0 + 0.2 * r) // BN gamma
            } else if name.ends_with(".1.bias") {
                mapped(numel, &|r| 0.1 * r) // BN beta
            } else if name.ends_with("running_mean") {
                mapped(numel, &|r| 0.3 * r)
            } else if name.ends_with("running_var") {
                mapped(numel, &|r| 1.0 + 0.5 * r.abs())
            } else {
                mapped(numel, &|r| 0.5 * r) // conv weights
            };
            init.insert(name, v);
        }
        let ps = ParamStore::new(gpu, blk.param_list(), &init);
        let x = gpu.storage_init("x", &mapped(in_shape.numel() as usize, &|r| r));
        blk.forward(&ctx, &ps, &x);
        gpu.read(blk.out(), blk.out_shape.numel() as usize)
    };

    for (i, (in_shape, cout, stride)) in [
        (Shape::new(2, 12, 10, 14), 12, 1), // residual: identity branch live
        (Shape::new(2, 12, 10, 14), 20, 1), // channel change: no identity
        (Shape::new(2, 12, 10, 14), 24, 2), // downsample
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 101 + i as u64;
        let f = block_forward(&fused_gpu, zipdepth::net::ids(), in_shape, cout, stride, seed, true);
        let u = block_forward(&plain_gpu, &plain_ids, in_shape, cout, stride, seed, false);
        assert_eq!(f.len(), u.len());
        let max_rel = scale_rel(&f, &u);
        assert!(
            max_rel < 5e-6,
            "QARep fused vs unfused rel err {max_rel} (cout {cout}, stride {stride})"
        );
    }
}

/// 4. The GROUPED register-tiled forward (`conv2d_gd_reg`), GPU vs CPU, over
///    ZipDepth's grouped shapes: grouped 1x1 (the fusion projections — octets
///    within one group, the hot case), depthwise 3x3, and depthwise DILATED 3x3
///    (the MinimalMultiScale branch). The CPU side runs the per-group GEMM /
///    depthwise fast path; the GPU side runs the WGSL tile — two independent
///    implementations of the same op agreeing to round-off. cout_g = 12 in the
///    grouped-1x1 case makes octets straddle-prone: the group-aligned masking
///    (`nc = min(8, cout_g - oc*8)`) is exactly what this pins.
#[test]
fn grouped_reg_conv_gpu_matches_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("skipping grouped GPU parity (MOE_SKIP_GPU_TESTS)");
        return;
    }
    let gpu = Gpu::new_wgpu(zipdepth::net::PIPELINES);
    let cpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let in_shape = Shape::new(2, 24, 10, 14);
    let specs = [
        // grouped 1x1, groups=2, cout=24 -> cout_g=12 (octet tail masked)
        ConvSpec { groups: 2, act: Act::None, norm: vision::blocks::Norm::None, ..ConvSpec::relu(24, 1, 1, 0) },
        // depthwise 3x3
        ConvSpec { groups: 24, act: Act::None, norm: vision::blocks::Norm::None, ..ConvSpec::relu(24, 3, 1, 1) },
        // depthwise 3x3 dilation 2 (pad 2 keeps the size)
        ConvSpec {
            groups: 24,
            dilation: 2,
            act: Act::None,
            norm: vision::blocks::Norm::None,
            ..ConvSpec::relu(24, 3, 1, 2)
        },
    ];
    for (i, spec) in specs.into_iter().enumerate() {
        let seed = 71 + i as u64;
        let g = eval_forward(&gpu, zipdepth::net::ids(), in_shape, spec, seed);
        let c = eval_forward(&cpu, zipdepth::net::ids(), in_shape, spec, seed);
        let max_rel = scale_rel(&g, &c);
        assert!(
            max_rel < 5e-6,
            "GPU vs CPU grouped-reg rel err {max_rel} for groups {} dilation {}",
            spec.groups,
            spec.dilation
        );
    }
}
