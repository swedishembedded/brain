// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `add_rms_quant` kernel (`kernels_cuda`,
//! `cu/add_rms_quant.cu`): the activation front end of an int8 linear -
//! residual add, RMSNorm, per-row scale, int8 pack - in ONE launch.
//!
//! Swedish Embedded AB implements bit-exact fused kernels for quantised LLM
//! decode. If your team needs expertise in collapsing a chain of small GPU
//! kernels without changing a single output bit, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! The claim is the one every substitution in this tree makes: the output is
//! BYTE-identical to what the chain it replaces produces. The reference is
//! that chain itself - the WGSL `add2`, `rmsnorm_rows`, `max_abs_rows` and
//! `quant_pack` kernels, dispatched exactly as a decode token dispatches them -
//! on the same device from the same inputs. The kernel keeps `rmsnorm_rows`'
//! 64-lane accumulation order, which is the only order-sensitive step, so
//! identity is achievable and a tolerance would only hide a defect.
//!
//! Shapes cover what a 64-thread-per-row layout can get wrong: `d` that is
//! not a multiple of the 64-lane stride (ragged tail), the widest row served (5120),
//! a row narrower than the lane count, several rows per launch, with and
//! without the residual add, and rows whose values break the quantiser's
//! assumptions (all zero, one huge outlier, subnormal-small).

use gpu_core::{Dispatch, Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("add2", kernels::ADD2),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("max_abs_rows", kernels::MAX_ABS_ROWS),
    ("quant_pack", kernels::QUANT_PACK),
];
const ADD2: usize = 0;
const RMS: usize = 1;
const MAX_ABS: usize = 2;
const QUANT: usize = 3;

const EPS: f32 = 1e-6;

/// Whether this device is offered the native kernels: a CUDA device, with
/// native kernels not withheld (`gpu_core::set_native_kernels`, the A/B
/// switch that pins the WGSL tier).
/// Under it these gates skip, and the "offered only where it can run" test
/// asserts the withholding.
fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

/// Deterministic values with a wide dynamic range, including exact zeros.
fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n)
        .map(|i| {
            let u = (r.next_u32() % 20001) as f32 / 10000.0 - 1.0;
            let mag = 10f32.powf((r.next_u32() % 5) as f32 - 2.0);
            if i % 97 == 0 { 0.0 } else { u * mag * spread }
        })
        .collect()
}

struct Out {
    sum: Vec<u32>,
    xn: Vec<u32>,
    xq: Vec<u32>,
    sx: Vec<u32>,
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Run both implementations on `a`, `b` (ignored without `add`) and gain `w`.
fn run(gpu: &Gpu, rows: u32, d: u32, add: bool, a: &[f32], b: &[f32], w: &[f32]) -> (Out, Out) {
    let n = (rows * d) as usize;
    let words = (rows * d / 4) as usize;
    let mk = |data: &[f32]| gpu.storage_init("in", data);
    let (ab, bb, wb) = (mk(a), mk(b), mk(w));

    // Reference: add2 -> rmsnorm_rows -> max_abs_rows -> quant_pack.
    let sum_r = gpu.storage(n as u64);
    let xn_r = gpu.storage(n as u64);
    let sx_r = gpu.storage(rows as u64);
    let xq_r = gpu.storage(words as u64);
    let mut steps = Vec::new();
    let src = if add {
        steps.push(gpu.dispatch(ADD2, &[&ab, &bb, &sum_r], &[rows * d], Dispatch::Threads(rows * d)));
        &sum_r
    } else {
        &ab
    };
    steps.push(gpu.dispatch(RMS, &[src, &wb, &xn_r], &[d, rows, EPS.to_bits()], Dispatch::Workgroups(rows)));
    steps.push(gpu.dispatch(MAX_ABS, &[&xn_r, &sx_r], &[rows, d], Dispatch::Workgroups(rows)));
    steps.push(gpu.dispatch(QUANT, &[&xn_r, &sx_r, &xq_r], &[rows, d], Dispatch::Threads(rows * d / 4)));
    gpu.submit(&[], &steps);

    // Native: one launch.
    let sum_n = gpu.storage(n as u64);
    let xn_n = gpu.storage(n as u64);
    let sx_n = gpu.storage(rows as u64);
    let xq_n = gpu.storage(words as u64);
    let step = gpu
        .fused_step(Fused::AddRmsQuant, &[&ab, &bb, &wb, &sum_n, &xn_n, &xq_n, &sx_n], &[d, rows, EPS.to_bits(), add as u32])
        .expect("the fused kernel was declined on a CUDA device");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();

    let read = |buf: &backend_api::DeviceBuffer, len: usize| bits(gpu.read(buf, len));
    let reference = Out { sum: if add { read(&sum_r, n) } else { vec![] }, xn: read(&xn_r, n), xq: read(&xq_r, words), sx: read(&sx_r, rows as usize) };
    let native = Out { sum: if add { read(&sum_n, n) } else { vec![] }, xn: read(&xn_n, n), xq: read(&xq_n, words), sx: read(&sx_n, rows as usize) };
    (native, reference)
}

fn check(gpu: &Gpu, rows: u32, d: u32, add: bool, a: &[f32], b: &[f32], w: &[f32], what: &str) {
    let (n, r) = run(gpu, rows, d, add, a, b, w);
    let ctx = format!("rows={rows} d={d} add={add} ({what})");
    assert_eq!(n.sum, r.sum, "residual sum differs: {ctx}");
    assert_eq!(n.xn, r.xn, "normalised activation differs: {ctx}");
    assert_eq!(n.sx, r.sx, "per-row scale differs: {ctx}");
    assert_eq!(n.xq, r.xq, "packed int8 activation differs: {ctx}");
}

#[test]
fn the_fused_kernel_is_offered_only_where_it_can_run() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert!(!gpu.has_fused(Fused::AddRmsQuant), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert!(gpu.has_fused(Fused::AddRmsQuant));
    assert!(Fused::AddRmsQuant.serves(&[5120, 1, 1e-6f32.to_bits(), 1]));
    assert!(!Fused::AddRmsQuant.serves(&[5122, 1, 1e-6f32.to_bits(), 1]), "d not a multiple of 4 cannot be packed");
    assert!(!Fused::AddRmsQuant.serves(&[64 * 81, 1, 1e-6f32.to_bits(), 1]), "wider than the registers hold");
    assert!(!Fused::AddRmsQuant.serves(&[5120, 0, 1e-6f32.to_bits(), 1]), "no rows");
}

#[test]
fn byte_identical_to_the_wgsl_chain_over_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    for (rows, d) in [(1u32, 5120u32), (1, 4), (1, 60), (1, 68), (2, 1000), (3, 4096), (8, 5120), (1, 4224), (2, 2048), (1, 17 * 64)] {
        let n = (rows * d) as usize;
        let seed = u64::from(rows) * 131 + u64::from(d);
        let a = values(n, seed, 1.0);
        let b = values(n, seed + 1, 0.5);
        let w = values(d as usize, seed + 2, 1.0);
        for add in [false, true] {
            check(&gpu, rows, d, add, &a, &b, &w, "random");
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
    let (rows, d) = (3u32, 5120u32);
    let n = (rows * d) as usize;
    let dd = d as usize;
    let w = values(dd, 9, 1.0);
    let zero = vec![0.0f32; n];
    check(&gpu, rows, d, true, &zero, &zero, &w, "all-zero rows hit the 1e-8 scale floor");
    // One huge outlier per row crushes everything else to 0 or +-1.
    let mut outlier = values(n, 11, 1.0);
    for r in 0..rows as usize {
        outlier[r * dd + 37 * (r + 1)] = 3.0e4;
    }
    check(&gpu, rows, d, false, &outlier, &zero, &w, "one outlier per row");
    let tiny: Vec<f32> = values(n, 13, 1.0).iter().map(|v| v * 1e-20).collect();
    check(&gpu, rows, d, true, &tiny, &tiny, &w, "values near the scale floor");
    let big: Vec<f32> = values(n, 15, 1.0).iter().map(|v| v * 1e12).collect();
    check(&gpu, rows, d, true, &big, &big, &w, "values whose squares overflow toward the f32 range");
}

/// Seeded random shapes: the grid above is not the only place the lane tails
/// and the shared-memory byte exchange have been looked at.
#[test]
fn random_shapes_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let mut r = data::rng::Lcg::new(0x00a1_1ce5);
    for _ in 0..40 {
        let rows = 1 + r.next_u32() % 8;
        let d = 4 * (1 + r.next_u32() % 1280);
        let n = (rows * d) as usize;
        let a = values(n, u64::from(r.next_u32()), 1.0);
        let b = values(n, u64::from(r.next_u32()), 2.0);
        let w = values(d as usize, u64::from(r.next_u32()), 1.0);
        check(&gpu, rows, d, r.next_u32() % 2 == 0, &a, &b, &w, "seeded");
    }
}
