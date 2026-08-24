// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The GEMM-lowered 1D convolutions (`audio::conv::conv1d_bias_fwd` /
//! `convtr1d_bias_fwd`) against a HOST oracle, across the shape space the
//! lowering has to cover.
//!
//! Why a host oracle and not the direct kernel: kernel-vs-kernel agreement
//! cannot tell you which one is wrong, and this repo has already been burnt by
//! an A/B whose *harness* was the liar rather than either kernel. The
//! oracle here is `conv::conv1d_ref` / `conv::convtr1d_ref`, which are the
//! written-out definitions of `conv1d.wgsl` / `convtr1d.wgsl` and share no code
//! with either the direct kernels or the lowering.
//!
//! The GEMM REASSOCIATES the `Cin*K` reduction (register accumulators instead
//! of one serial f32 chain), so this is a tolerance, never `assert_eq!`.
//!
//! Every case also asserts WHICH path ran. A lowering test that silently fell
//! back to the direct kernel would pass while testing nothing - and the
//! fallbacks here are real (narrow `Cout`, grouping, no workgroup reductions),
//! so "it agreed" is only meaningful beside "and it was the lowered form".

use audio::conv::{conv1d_bias_fwd, conv1d_ref, convtr1d_bias_fwd, convtr1d_ref, Conv1d, ConvGemmKernels, ConvKernels, ConvScratch};

use data::rng::Lcg;

const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("im2col1d_at", kernels::IM2COL1D_AT),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("col2im1d_bias", kernels::COL2IM1D_BIAS),
];

fn kern(fwd: usize) -> ConvGemmKernels {
    ConvGemmKernels {
        direct: ConvKernels { fwd, dx: 0, dw: 0 },
        bias: 2,
        im2col: 3,
        matmul: 4,
        matmul_nn: 5,
        matmul_tn: 6,
        nlc_bias: 7,
        col2im: 8,
    }
}

/// Which lowering a recorded step list actually is, read from the dispatched
/// kernel indices rather than inferred from the step COUNT - the transposed
/// lowering is two dispatches, exactly as many as the direct path, so a count
/// cannot tell them apart. `Step::meta()` carries the caller-space kernel
/// index, which is unambiguous.
fn path(steps: &[gpu_core::Step]) -> &'static str {
    let used: Vec<usize> = steps.iter().map(|s| s.meta().expect("step recorded through the facade").kernel).collect();
    match () {
        _ if used.contains(&3) => "im2col+reg3",
        _ if used.contains(&5) => "matmul_dx_reg",
        _ if used.contains(&6) => "matmul_dw_reg_splitk",
        _ if used.contains(&0) || used.contains(&1) => "direct",
        _ => panic!("no recognisable conv kernel in {used:?}"),
    }
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(&a, &b)| f64::from(a - b).powi(2)).sum();
    let den: f64 = want.iter().map(|&b| f64::from(b).powi(2)).sum::<f64>().max(1e-30);
    (num / den).sqrt()
}

/// Bias-folded oracle: the reference conv plus `bias[co]` per output channel.
fn with_bias(c: &Conv1d, mut y: Vec<f32>, bias: &[f32]) -> Vec<f32> {
    for (i, v) in y.iter_mut().enumerate() {
        let co = (i as u32 / c.lo) % c.cout;
        *v += bias[co as usize];
    }
    y
}

struct Case {
    label: &'static str,
    c: Conv1d,
    /// The path expected on a device WITH workgroup reductions. Without them
    /// every case must fall back to `direct`.
    want: &'static str,
}

fn conv_case(label: &'static str, n: u32, cin: u32, l: u32, cout: u32, k: u32, stride: u32, pad: u32, dilation: u32, groups: u32, want: &'static str) -> Case {
    let lo = Conv1d::out_len(l, k, stride, pad, pad, dilation);
    Case { label, c: Conv1d { n, cin, l, cout, k, stride, pad, dilation, groups, lo }, want }
}

#[test]
fn lowered_conv1d_matches_the_host_oracle() {
    let g = gpu_core::testgpu::dev(PIPELINES);
    let coop = g.caps().workgroup_reductions;
    let cases = [
        // The vocoder's own two shapes, small: a k=7 dilated pad-3 residual
        // conv and a k=1 projection (the `matmul_dx_reg` NN path).
        conv_case("k7 pad3 dil1", 1, 64, 200, 64, 7, 1, 3, 1, 1, "im2col+reg3"),
        conv_case("k7 pad9 dil3", 1, 64, 200, 64, 7, 1, 9, 3, 1, "im2col+reg3"),
        conv_case("k7 pad27 dil9", 1, 64, 200, 64, 7, 1, 27, 9, 1, "im2col+reg3"),
        conv_case("k1 proj", 1, 64, 200, 96, 1, 1, 0, 1, 1, "matmul_dx_reg"),
        // Batch > 1 exercises the per-row bound sub-ranges on BOTH paths.
        conv_case("k7 batch2", 2, 64, 192, 64, 7, 1, 3, 1, 1, "im2col+reg3"),
        conv_case("k1 batch2", 2, 64, 192, 64, 1, 1, 0, 1, 1, "matmul_dx_reg"),
        // Strided, unpadded, asymmetric length: `Lo` is not a multiple of the
        // GEMM tile and the last chunk is ragged.
        conv_case("k3 stride2 pad0", 1, 64, 301, 64, 3, 2, 0, 1, 1, "im2col+reg3"),
        conv_case("k5 stride3 pad2", 1, 96, 257, 64, 5, 3, 2, 1, 1, "im2col+reg3"),
        // Cin*K straddles the GEMM's own 128-wide contraction tile.
        conv_case("wide contraction", 1, 128, 300, 128, 3, 1, 1, 1, 1, "im2col+reg3"),
        // Below `GEMM_CONV1D_MIN_COUT`: must FALL BACK, and still be right.
        conv_case("narrow cout", 1, 64, 200, 8, 7, 1, 3, 1, 1, "direct"),
        // Grouped: the lowering cannot express it, so it must fall back.
        conv_case("grouped", 1, 64, 200, 64, 3, 1, 1, 1, 4, "direct"),
        // k=1 with a pad: NOT the NN fast path (it shifts), so it goes through
        // im2col like any other kernel width.
        conv_case("k1 padded", 1, 64, 200, 64, 1, 1, 1, 1, 1, "im2col+reg3"),
    ];

    let mut rng = Lcg::new(0x51D1);
    for case in cases {
        let c = case.c;
        let x = rng.vec_scaled((c.n * c.cin * c.l) as usize, 1.0);
        let w = rng.vec_scaled(c.weight_numel(), 0.5);
        let bias = rng.vec_scaled(c.cout as usize, 0.25);
        let want = with_bias(&c, conv1d_ref(&c, &x, &w), &bias);

        let xb = g.storage_init("x", &x);
        let wb = g.storage_init("w", &w);
        let bb = g.storage_init("b", &bias);
        let yb = g.storage(u64::from(c.n) * u64::from(c.cout) * u64::from(c.lo));
        let mut scratch = ConvScratch::new();
        let steps = conv1d_bias_fwd(&g, &kern(0), &c, &xb, &wb, &bb, &yb, &mut scratch);
        let took = path(&steps);
        let want_path = if coop { case.want } else { "direct" };
        assert_eq!(took, want_path, "conv1d[{}]: took the wrong path", case.label);
        g.submit(&[], &steps);
        g.poll_wait();
        let got = g.read(&yb, want.len());

        let rel = rel_err(&got, &want);
        println!("conv1d[{}] path={took} rel_l2={rel:.3e}", case.label);
        assert!(rel < 1e-5, "conv1d[{}]: rel_l2 {rel:.3e} vs the host oracle", case.label);
    }
}

#[test]
fn lowered_convtr1d_matches_the_host_oracle() {
    let g = gpu_core::testgpu::dev(PIPELINES);
    let coop = g.caps().workgroup_reductions;
    // The vocoder's own family: K = 2*stride, pad = ceil(stride/2), plus a
    // couple of shapes outside it.
    let tr = |label: &'static str, n: u32, cin: u32, l: u32, cout: u32, k: u32, stride: u32, pad: u32, dilation: u32, want: &'static str| {
        let lo = Conv1d::out_len_transposed(l, k, stride, pad, 0, dilation);
        Case { label, c: Conv1d { n, cin, l, cout, k, stride, pad, dilation, groups: 1, lo }, want }
    };
    let cases = [
        tr("stride2 k4", 1, 128, 100, 64, 4, 2, 1, 1, "matmul_dw_reg_splitk"),
        tr("stride4 k8", 1, 128, 100, 64, 8, 4, 2, 1, "matmul_dw_reg_splitk"),
        tr("stride8 k16", 1, 96, 61, 64, 16, 8, 4, 1, "matmul_dw_reg_splitk"),
        tr("stride1 k3 pad1", 1, 64, 100, 64, 3, 1, 1, 1, "matmul_dw_reg_splitk"),
        tr("dilated", 1, 64, 100, 64, 4, 2, 1, 2, "matmul_dw_reg_splitk"),
        tr("batch2", 2, 64, 96, 64, 4, 2, 1, 1, "matmul_dw_reg_splitk"),
        tr("odd length", 1, 96, 97, 96, 4, 2, 0, 1, "matmul_dw_reg_splitk"),
        // Below `GEMM_CONVTR1D_MIN_COUT`, which is 4 rather than the plain
        // conv's 16 - the transposed pair's crossover is genuinely elsewhere.
        tr("narrow cout", 1, 64, 100, 2, 4, 2, 1, 1, "direct"),
        // ... and one just above it, so the boundary is pinned from both
        // sides rather than only from the fallback side.
        tr("at the threshold", 1, 64, 100, 4, 4, 2, 1, 1, "matmul_dw_reg_splitk"),
    ];

    let mut rng = Lcg::new(0x51D2);
    for case in cases {
        let c = case.c;
        let x = rng.vec_scaled((c.n * c.cin * c.l) as usize, 1.0);
        let w = rng.vec_scaled(c.weight_numel_transposed(), 0.5);
        let bias = rng.vec_scaled(c.cout as usize, 0.25);
        let want = with_bias(&c, convtr1d_ref(&c, &x, &w), &bias);

        let xb = g.storage_init("x", &x);
        let wb = g.storage_init("w", &w);
        let bb = g.storage_init("b", &bias);
        let yb = g.storage(u64::from(c.n) * u64::from(c.cout) * u64::from(c.lo));
        let mut scratch = ConvScratch::new();
        let steps = convtr1d_bias_fwd(&g, &kern(1), &c, &xb, &wb, &bb, &yb, &mut scratch);
        let took = path(&steps);
        let want_path = if coop { case.want } else { "direct" };
        assert_eq!(took, want_path, "convtr1d[{}]: took the wrong path", case.label);
        g.submit(&[], &steps);
        g.poll_wait();
        let got = g.read(&yb, want.len());

        let rel = rel_err(&got, &want);
        println!("convtr1d[{}] path={took} rel_l2={rel:.3e}", case.label);
        assert!(rel < 1e-5, "convtr1d[{}]: rel_l2 {rel:.3e} vs the host oracle", case.label);
    }
}

/// The chunked path, with the scratch budget squeezed so `Lo` needs SEVERAL
/// GEMM chunks.
///
/// Every other case here fits one chunk, and a single-chunk run cannot see a
/// chunking bug at all: `im2col1d_at` ignoring its `pos0` window origin - the
/// exact defect the windowing exists to get right - passed the entire
/// shape-coverage test above unnoticed, and is caught here.
#[test]
fn the_chunked_conv1d_matches_the_host_oracle_across_chunk_boundaries() {
    let g = gpu_core::testgpu::dev(PIPELINES);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip("no workgroup reductions: the lowering is not selected on this device");
        return;
    }
    let mut rng = Lcg::new(0x51D4);
    // cink = 64*3 = 192 floats/row; a 1 MiB budget is 262144 floats = 1365
    // rows, snapped to 1280 - so L = 3000 is three chunks, the last ragged.
    let l = 3000u32;
    let c = Conv1d { n: 2, cin: 64, l, cout: 64, k: 3, stride: 1, pad: 1, dilation: 1, groups: 1, lo: Conv1d::out_len(l, 3, 1, 1, 1, 1) };
    let x = rng.vec_scaled((c.n * c.cin * c.l) as usize, 1.0);
    let w = rng.vec_scaled(c.weight_numel(), 0.5);
    let bias = rng.vec_scaled(c.cout as usize, 0.25);
    let want = with_bias(&c, conv1d_ref(&c, &x, &w), &bias);

    let xb = g.storage_init("x", &x);
    let wb = g.storage_init("w", &w);
    let bb = g.storage_init("b", &bias);
    let yb = g.storage(u64::from(c.n) * u64::from(c.cout) * u64::from(c.lo));
    let mut scratch = ConvScratch::with_budget_mib(1);
    let steps = conv1d_bias_fwd(&g, &kern(0), &c, &xb, &wb, &bb, &yb, &mut scratch);
    assert_eq!(path(&steps), "im2col+reg3");
    // 2 batch rows x (3 chunks x 2 dispatches + 1 epilogue) = 14. Asserting
    // the count is what proves the budget actually produced several chunks -
    // without it a wider budget would silently make this a one-chunk test
    // again, i.e. a duplicate of the case above.
    assert_eq!(steps.len(), 14, "expected 3 chunks per batch row, got {} steps", steps.len());
    g.submit(&[], &steps);
    g.poll_wait();
    let got = g.read(&yb, want.len());
    let rel = rel_err(&got, &want);
    println!("chunked conv1d rel_l2={rel:.3e}");
    assert!(rel < 1e-5, "chunked conv1d: rel_l2 {rel:.3e} vs the host oracle");

    // CHUNK-COUNT INVARIANCE, bit-exact. The chunking is supposed to change
    // nothing at all - each output position's dot product is computed from the
    // same taps in the same order whatever window it falls in - so the answer
    // here is exactly 0.0 difference, not a tolerance. That is a far sharper
    // instrument than the oracle comparison above: a boundary bug that
    // perturbs a few hundred of 350k samples can hide inside a 1e-5 relative
    // L2 and cannot hide from `assert_eq!`.
    let one = g.storage(u64::from(c.n) * u64::from(c.cout) * u64::from(c.lo));
    let mut whole = ConvScratch::with_budget_mib(4096);
    let steps = conv1d_bias_fwd(&g, &kern(0), &c, &xb, &wb, &bb, &one, &mut whole);
    // 2 batch rows x (1 chunk x 2 dispatches + 1 epilogue).
    assert_eq!(steps.len(), 6, "expected ONE chunk per batch row at a 4 GiB budget, got {} steps", steps.len());
    g.submit(&[], &steps);
    g.poll_wait();
    assert_eq!(g.read(&one, want.len()), got, "the chunked result differs from the single-chunk one");
}

/// One `ConvScratch` shared by a CHAIN of convs, which is how every caller
/// uses it (the whole decoder is one recorded pass). This is the property that
/// makes the reuse safe: the scratch grows to the largest need and each conv's
/// epilogue reads it before the next conv's GEMM overwrites it.
///
/// It also pins the thing that forced `matmul_dw_reg_splitk` over
/// `matmul_dw_reg`: with an ACCUMULATING GEMM the second transposed conv in a
/// chain would fold the first one's `col` into its own and this would go red.
#[test]
fn a_shared_scratch_survives_a_chain_of_convs() {
    let g = gpu_core::testgpu::dev(PIPELINES);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip("no workgroup reductions: the lowering is not selected on this device");
        return;
    }
    let mut rng = Lcg::new(0x51D3);
    let mut scratch = ConvScratch::new();
    let mut steps = Vec::new();
    let mut checks = Vec::new();

    // Deliberately DECREASING sizes after an increase, so the scratch is grown
    // once and then reused at a smaller need - the case where stale bytes from
    // the bigger predecessor are still sitting in the buffer.
    let shapes = [(64u32, 128u32, 64u32), (128, 256, 128), (64, 96, 64), (96, 200, 96)];
    for (i, &(cin, l, cout)) in shapes.iter().enumerate() {
        let lo = Conv1d::out_len(l, 3, 1, 1, 1, 1);
        let c = Conv1d { n: 1, cin, l, cout, k: 3, stride: 1, pad: 1, dilation: 1, groups: 1, lo };
        let x = rng.vec_scaled((cin * l) as usize, 1.0);
        let w = rng.vec_scaled(c.weight_numel(), 0.5);
        let bias = rng.vec_scaled(cout as usize, 0.25);
        let want = with_bias(&c, conv1d_ref(&c, &x, &w), &bias);
        let xb = g.storage_init("x", &x);
        let wb = g.storage_init("w", &w);
        let bb = g.storage_init("b", &bias);
        let yb = g.storage(u64::from(cout) * u64::from(lo));
        steps.extend(conv1d_bias_fwd(&g, &kern(0), &c, &xb, &wb, &bb, &yb, &mut scratch));

        // ... and a transposed conv between each pair, sharing the same slot.
        let tlo = Conv1d::out_len_transposed(l, 4, 2, 1, 0, 1);
        let tc = Conv1d { n: 1, cin, l, cout, k: 4, stride: 2, pad: 1, dilation: 1, groups: 1, lo: tlo };
        let tw = rng.vec_scaled(tc.weight_numel_transposed(), 0.5);
        let twant = with_bias(&tc, convtr1d_ref(&tc, &x, &tw), &bias);
        let twb = g.storage_init("tw", &tw);
        let tyb = g.storage(u64::from(cout) * u64::from(tlo));
        steps.extend(convtr1d_bias_fwd(&g, &kern(1), &tc, &xb, &twb, &bb, &tyb, &mut scratch));

        checks.push((format!("conv1d#{i}"), yb, want));
        checks.push((format!("convtr1d#{i}"), tyb, twant));
    }
    g.submit(&[], &steps);
    g.poll_wait();
    for (label, buf, want) in checks {
        let got = g.read(&buf, want.len());
        let rel = rel_err(&got, &want);
        println!("chain[{label}] rel_l2={rel:.3e}");
        assert!(rel < 1e-5, "chain[{label}]: rel_l2 {rel:.3e} - a shared scratch leaked between convs");
    }
}
