// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `quant_epilogue` kernel (`kernels_cuda`,
//! `cu/quant_epilogue.cu`): the producer of an int8 linear's input activation
//! and its per-row quantisation in ONE launch.
//!
//! Swedish Embedded AB implements bit-exact fused kernels for quantised LLM
//! decode. If your team needs expertise in collapsing a chain of small GPU
//! kernels without changing a single output bit, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! The reference is the WGSL chain the kernel replaces, dispatched the way a
//! decode token dispatches it, on the same device from the same inputs:
//!
//! - mode 0: `max_abs_rows`, `quant_pack`;
//! - mode 1: `silu_mul`, `max_abs_rows`, `quant_pack`;
//! - mode 2: `sigmoid`, `mul`, `max_abs_rows`, `quant_pack`.
//!
//! Nothing in the kernel is order-sensitive (the max is exact, every other
//! step is elementwise), so the output is BYTE-identical, not merely close.

use gpu_core::{Dispatch, Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("silu_mul", kernels::SILU_MUL),
    ("sigmoid", kernels::SIGMOID),
    ("mul", kernels::MUL),
    ("max_abs_rows", kernels::MAX_ABS_ROWS),
    ("quant_pack", kernels::QUANT_PACK),
];
const SILU_MUL: usize = 0;
const SIGMOID: usize = 1;
const MUL: usize = 2;
const MAX_ABS: usize = 3;
const QUANT: usize = 4;

/// Whether this device is offered the native kernels: a CUDA device, with
/// native kernels not withheld (`gpu_core::set_native_kernels`, the A/B
/// switch that pins the WGSL tier).
/// Under it these gates skip, and the "offered only where it can run" test
/// asserts the withholding.
fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n)
        .map(|i| {
            let u = (r.next_u32() % 20001) as f32 / 10000.0 - 1.0;
            let mag = 10f32.powf((r.next_u32() % 5) as f32 - 2.0);
            if i % 89 == 0 { 0.0 } else { u * mag * spread }
        })
        .collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

struct Out {
    y: Vec<u32>,
    xq: Vec<u32>,
    sx: Vec<u32>,
}

fn run(gpu: &Gpu, rows: u32, k: u32, mode: u32, a: &[f32], b: &[f32]) -> (Out, Out) {
    let n = (rows * k) as usize;
    let words = (rows * k / 4) as usize;
    let ab = gpu.storage_init("a", a);
    let bb = gpu.storage_init("b", b);

    // Reference chain.
    let y_r = gpu.storage(n as u64);
    let groups = (rows * k / 32) as usize;
    let sx_r = gpu.storage(groups as u64);
    let xq_r = gpu.storage(words as u64);
    let mut steps = Vec::new();
    let produced = match mode {
        0 => &ab,
        1 => {
            steps.push(gpu.dispatch(SILU_MUL, &[&ab, &bb, &y_r], &[rows * k], Dispatch::Threads(rows * k)));
            &y_r
        }
        _ => {
            let gate = gpu.storage(n as u64);
            steps.push(gpu.dispatch(SIGMOID, &[&bb, &gate], &[rows * k], Dispatch::Threads(rows * k)));
            steps.push(gpu.dispatch(MUL, &[&ab, &gate, &y_r], &[rows * k], Dispatch::Threads(rows * k)));
            // Keep `gate` alive until the chain has been submitted.
            gpu.submit(&[], &steps);
            steps = Vec::new();
            drop(gate);
            &y_r
        }
    };
    steps.push(gpu.dispatch(MAX_ABS, &[produced, &sx_r], &[rows, k], Dispatch::Workgroups(rows)));
    steps.push(gpu.dispatch(QUANT, &[produced, &sx_r, &xq_r], &[rows, k], Dispatch::Threads(rows * k / 4)));
    gpu.submit(&[], &steps);

    // Native: one launch.
    let y_n = gpu.storage(n as u64);
    let sx_n = gpu.storage(groups as u64);
    let xq_n = gpu.storage(words as u64);
    let step = gpu
        .fused_step(Fused::QuantEpilogue, &[&ab, &bb, &y_n, &xq_n, &sx_n], &[k, rows, mode, 0])
        .expect("the fused kernel was declined on a CUDA device");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();

    let read = |buf: &backend_api::DeviceBuffer, len: usize| bits(gpu.read(buf, len));
    // Mode 0 produces no `y`: the activation is `a` itself.
    let y = |buf: &backend_api::DeviceBuffer| if mode == 0 { vec![] } else { read(buf, n) };
    (
        Out { y: y(&y_n), xq: read(&xq_n, words), sx: read(&sx_n, groups) },
        Out { y: y(&y_r), xq: read(&xq_r, words), sx: read(&sx_r, groups) },
    )
}

fn check(gpu: &Gpu, rows: u32, k: u32, mode: u32, a: &[f32], b: &[f32], what: &str) {
    let (n, r) = run(gpu, rows, k, mode, a, b);
    let ctx = format!("rows={rows} k={k} mode={mode} ({what})");
    assert_eq!(n.y, r.y, "produced activation differs: {ctx}");
    assert_eq!(n.sx, r.sx, "group scales differ: {ctx}");
    assert_eq!(n.xq, r.xq, "packed int8 activation differs: {ctx}");
}

#[test]
fn the_fused_kernel_is_offered_only_where_it_can_run() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert!(!gpu.has_fused(Fused::QuantEpilogue), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert!(gpu.has_fused(Fused::QuantEpilogue));
    assert!(Fused::QuantEpilogue.serves(&[17408, 1, 1, 0]));
    assert!(!Fused::QuantEpilogue.serves(&[17392, 1, 1, 0]), "k not a multiple of 32 is not whole scale groups");
    assert!(!Fused::QuantEpilogue.serves(&[1024 * 19, 1, 1, 0]), "wider than the registers hold");
    assert!(!Fused::QuantEpilogue.serves(&[64, 1, 3, 0]), "unknown mode");
    assert!(!Fused::QuantEpilogue.serves(&[64, 0, 1, 0]), "no rows");
}

#[test]
fn byte_identical_to_the_wgsl_chain_in_every_mode() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    // 17408 and 6144 are the real decode widths (MLP hidden, GDN/attention
    // value width); the rest are tails shorter and longer than a warp's
    // stride and partial last slices.
    for (rows, k) in [(1u32, 17408u32), (1, 6144), (1, 32), (1, 96), (1, 1024), (2, 4128), (3, 256), (8, 6144), (1, 18432), (2, 2080)] {
        let n = (rows * k) as usize;
        let seed = u64::from(rows) * 977 + u64::from(k);
        let a = values(n, seed, 4.0);
        let b = values(n, seed + 1, 4.0);
        for mode in 0..=2 {
            check(&gpu, rows, k, mode, &a, &b, "random");
        }
    }
}

#[test]
fn rows_that_break_the_quantiser_s_assumptions_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let (rows, k) = (3u32, 6144u32);
    let n = (rows * k) as usize;
    let zero = vec![0.0f32; n];
    let b = values(n, 5, 3.0);
    for mode in 0..=2 {
        check(&gpu, rows, k, mode, &zero, &zero, "all zero hits the 1e-8 scale floor");
    }
    let mut outlier = values(n, 7, 1.0);
    for r in 0..rows as usize {
        outlier[r * k as usize + 41 * (r + 1)] = 5.0e3;
    }
    for mode in 0..=2 {
        check(&gpu, rows, k, mode, &outlier, &b, "one outlier per row");
    }
    // Large magnitudes drive silu/sigmoid into their saturated and
    // exp-overflow regimes, where the formula (not the maths) is the contract.
    let big: Vec<f32> = values(n, 9, 1.0).iter().map(|v| v * 1e3).collect();
    for mode in 1..=2 {
        check(&gpu, rows, k, mode, &big, &big, "saturated activations");
    }
}

#[test]
fn random_shapes_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    for _ in 0..40 {
        let rows = 1 + r.next_u32() % 8;
        let k = 32 * (1 + r.next_u32() % 576);
        let n = (rows * k) as usize;
        let a = values(n, u64::from(r.next_u32()), 3.0);
        let b = values(n, u64::from(r.next_u32()), 3.0);
        check(&gpu, rows, k, r.next_u32() % 3, &a, &b, "seeded");
    }
}

#[test]
fn the_no_pack_flag_produces_the_same_activation_and_leaves_the_packed_rows_alone() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    for (rows, k, mode) in [(1u32, 17408u32, 1u32), (3, 6144, 2), (8, 4128, 1)] {
        let n = (rows * k) as usize;
        let a = values(n, 31 + u64::from(k), 4.0);
        let b = values(n, 32 + u64::from(k), 4.0);
        let (ab, bb) = (gpu.storage_init("a", &a), gpu.storage_init("b", &b));
        let (y_p, y_u) = (gpu.storage(n as u64), gpu.storage(n as u64));
        let (xq, sx) = (gpu.storage_init("xq", &[f32::from_bits(0xdead_beef); 1]), gpu.storage_init("sx", &[f32::from_bits(0xdead_beef); 1]));
        let (xq_p, sx_p) = (gpu.storage((rows * k / 4) as u64), gpu.storage((rows * k / 32) as u64));
        let packed = gpu.fused_step(Fused::QuantEpilogue, &[&ab, &bb, &y_p, &xq_p, &sx_p], &[k, rows, mode, 0]).expect("offered");
        let unpacked = gpu.fused_step(Fused::QuantEpilogue, &[&ab, &bb, &y_u, &xq, &sx], &[k, rows, mode, 1]).expect("offered");
        gpu.submit(&[], &[packed, unpacked]);
        gpu.poll_wait();
        assert_eq!(bits(gpu.read(&y_u, n)), bits(gpu.read(&y_p, n)), "the product differs: rows={rows} k={k} mode={mode}");
        assert_eq!(bits(gpu.read(&xq, 1)), vec![0xdead_beef], "the packed rows were written");
        assert_eq!(bits(gpu.read(&sx, 1)), vec![0xdead_beef], "the scales were written");
    }
}
