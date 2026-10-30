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
    let groups = (rows * d / 32) as usize;
    let sx_r = gpu.storage(groups as u64);
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
    let sx_n = gpu.storage(groups as u64);
    let xq_n = gpu.storage(words as u64);
    let step = gpu
        .fused_step(Fused::AddRmsQuant, &[&ab, &bb, &wb, &sum_n, &xn_n, &xq_n, &sx_n], &[d, rows, EPS.to_bits(), add as u32])
        .expect("the fused kernel was declined on a CUDA device");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();

    let read = |buf: &backend_api::DeviceBuffer, len: usize| bits(gpu.read(buf, len));
    let reference = Out { sum: if add { read(&sum_r, n) } else { vec![] }, xn: read(&xn_r, n), xq: read(&xq_r, words), sx: read(&sx_r, groups) };
    let native = Out { sum: if add { read(&sum_n, n) } else { vec![] }, xn: read(&xn_n, n), xq: read(&xq_n, words), sx: read(&sx_n, groups) };
    (native, reference)
}

fn check(gpu: &Gpu, rows: u32, d: u32, add: bool, a: &[f32], b: &[f32], w: &[f32], what: &str) {
    let (n, r) = run(gpu, rows, d, add, a, b, w);
    let ctx = format!("rows={rows} d={d} add={add} ({what})");
    assert_eq!(n.sum, r.sum, "residual sum differs: {ctx}");
    assert_eq!(n.xn, r.xn, "normalised activation differs: {ctx}");
    assert_eq!(n.sx, r.sx, "group scales differ: {ctx}");
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
    assert!(!Fused::AddRmsQuant.serves(&[5104, 1, 1e-6f32.to_bits(), 1]), "d not a multiple of 32 is not whole scale groups");
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
    for (rows, d) in [(1u32, 5120u32), (1, 32), (1, 96), (1, 160), (2, 992), (3, 4096), (8, 5120), (1, 4224), (2, 2048), (1, 17 * 64)] {
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
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    for _ in 0..40 {
        let rows = 1 + r.next_u32() % 8;
        let d = 32 * (1 + r.next_u32() % 160);
        let n = (rows * d) as usize;
        let a = values(n, u64::from(r.next_u32()), 1.0);
        let b = values(n, u64::from(r.next_u32()), 2.0);
        let w = values(d as usize, u64::from(r.next_u32()), 1.0);
        check(&gpu, rows, d, r.next_u32().is_multiple_of(2), &a, &b, &w, "seeded");
    }
}

/// `v` rounded to bf16 the way `cast_f32_bf16` does: nearest even, NaN to the quiet NaN.
fn bf16_bits(v: f32) -> u32 {
    let b = v.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16
}

#[test]
fn the_bf16_flag_writes_the_rows_cast_f32_bf16_would_make_and_leaves_the_packed_rows_alone() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    for (rows, d, add) in [(1u32, 5120u32, true), (4, 5120, false), (2, 4128, true)] {
        let n = (rows * d) as usize;
        let a = values(n, 41 + u64::from(d), 3.0);
        let b = values(n, 42 + u64::from(d), 3.0);
        let w = values(d as usize, 43, 2.0);
        let (ab, bb, wb) = (gpu.storage_init("a", &a), gpu.storage_init("b", &b), gpu.storage_init("w", &w));
        let (sum, xn, bf16) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64 / 2));
        let sx = gpu.storage_init("sx", &[f32::from_bits(0xdead_beef); 1]);
        let step = gpu
            .fused_step(Fused::AddRmsQuant, &[&ab, &bb, &wb, &sum, &xn, &bf16, &sx], &[d, rows, EPS.to_bits(), add as u32 | 2 | 4])
            .expect("offered");
        gpu.submit(&[], &[step]);
        gpu.poll_wait();
        let xn_host = gpu.read(&xn, n);
        let want: Vec<u32> = xn_host.chunks(2).map(|p| bf16_bits(p[0]) | bf16_bits(p[1]) << 16).collect();
        assert_eq!(bits(gpu.read(&bf16, n / 2)), want, "the bf16 rows differ: rows={rows} d={d} add={add}");
        assert_eq!(bits(gpu.read(&sx, 1)), vec![0xdead_beef], "the scales were written");
    }
}

/// The E4M3 variant (`add_rms_quant_e4m3`) is a wide-row layout of its own (a warp a scale block, the sum of squares by shuffles),
/// so its normalised rows equal the int8 kernel's up to the order of that sum - a few ulps of the row's mean square - and the
/// residual sum bit for bit; what it claims exactly is that its E4M3 bytes and scales are what `quant_e4m3_128` makes of the
/// `xn` it stores, and that with bit 4 it leaves `xn` unwritten.
#[test]
fn the_e4m3_variant_writes_what_the_quantiser_makes_of_its_stored_rows() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (rows, d) in [(1u32, 128u32), (3, 640), (130, 5120), (7, 2560), (5, 1152)] {
        for add in [true, false] {
            let n = (rows * d) as usize;
            let (a, b, w) = (values(n, 31 + u64::from(d), 3.0), values(n, 77, 1.0), values(d as usize, 5, 1.0));
            let mk = |data: &[f32]| gpu.storage_init("in", data);
            let (ab, bb, wb) = (mk(&a), mk(&b), mk(&w));
            // The int8 kernel without the packing: the chain's `sum` and `xn`.
            let (sum_r, xn_r) = (gpu.storage(n as u64), gpu.storage(n as u64));
            let (one_q, one_s) = (gpu.storage(1), gpu.storage(1));
            let step = gpu.fused_step(Fused::AddRmsQuant, &[&ab, &bb, &wb, &sum_r, &xn_r, &one_q, &one_s], &[d, rows, EPS.to_bits(), add as u32 | 2]).expect("offered");
            gpu.submit(&[], &[step]);
            for keep in [true, false] {
                let (sum, xn) = (gpu.storage(n as u64), gpu.storage_init("untouched", &vec![1.5; n]));
                let (q, s) = (gpu.storage(u64::from(rows * d / 4)), gpu.storage(u64::from(rows * d / 128)));
                let step = gpu.fused_step(Fused::AddRmsQuantE4m3, &[&ab, &bb, &wb, &sum, &xn, &q, &s], &[d, rows, EPS.to_bits(), add as u32 | if keep { 0 } else { 16 }]).expect("offered");
                gpu.submit(&[], &[step]);
                gpu.poll_wait();
                let ctx = format!("rows={rows} d={d} add={add} keep_xn={keep}");
                if add {
                    assert_eq!(bits(gpu.read(&sum, n)), bits(gpu.read(&sum_r, n)), "residual sum differs: {ctx}");
                }
                if !keep {
                    assert!(gpu.read(&xn, n).iter().all(|v| *v == 1.5), "xn was written: {ctx}");
                    continue;
                }
                // The rows agree with the chain's to the order of the sum of squares.
                let (got, want) = (gpu.read(&xn, n), gpu.read(&xn_r, n));
                let worst = got.iter().zip(&want).map(|(g, r)| (g - r).abs() / r.abs().max(1e-30)).fold(0.0f32, f32::max);
                assert!(worst < 1e-5, "normalised rows differ by {worst:.2e} relative: {ctx}");
                // The bytes and scales are the quantiser's on the rows this kernel stored.
                let mut steps = Vec::new();
                let x = gpu_core::provider::library::Operand { buf: &xn, range: (0, n as u64) };
                let (q_ref, s_ref) = gpu_core::provider::library::quantize_e4m3_rows(&gpu, &mut steps, &x, rows, d).expect("offered");
                gpu.submit(&[], &steps);
                gpu.poll_wait();
                assert_eq!(bits(gpu.read(&q, (rows * d / 4) as usize)), bits(gpu.read(&q_ref, (rows * d / 4) as usize)), "E4M3 bytes differ: {ctx}");
                assert_eq!(bits(gpu.read(&s, (rows * d / 128) as usize)), bits(gpu.read(&s_ref, (rows * d / 128) as usize)), "E4M3 scales differ: {ctx}");
            }
        }
    }
}

/// The front end of an E4M3 product at a mixed round's rows, as the two launches it replaces and as one (`#[ignore]`: a measurement).
///
/// ```text
/// make test/one P=brain-gpu-core T=add_rms_quant_native ARGS="--ignored --nocapture add_rms_front_rate"
/// ```
#[test]
#[ignore]
fn add_rms_front_rate() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    let d = 5120u32;
    for rows in [64u32, 256, 768, 2048] {
        let n = (rows * d) as usize;
        let (a, b, w) = (values(n, 1, 3.0), values(n, 2, 1.0), values(d as usize, 5, 1.0));
        let mk = |data: &[f32]| gpu.storage_init("in", data);
        let (ab, bb, wb) = (mk(&a), mk(&b), mk(&w));
        let (sum, xn) = (gpu.storage(n as u64), gpu.storage(n as u64));
        let (one_q, one_s) = (gpu.storage(1), gpu.storage(1));
        let (q, s) = (gpu.storage(u64::from(rows * d / 4)), gpu.storage(u64::from(rows * d / 128)));
        gpu.set_kernel_timing(true);
        for (label, store_xn) in [("pair (add_rms_quant + quant_e4m3_128)", true), ("one launch, xn stored", true), ("one launch, xn not stored", false)] {
            let mut steps = Vec::new();
            for _ in 0..20 {
                if label.starts_with("pair") {
                    steps.push(gpu.fused_step(Fused::AddRmsQuant, &[&ab, &bb, &wb, &sum, &xn, &one_q, &one_s], &[d, rows, EPS.to_bits(), 1 | 2]).expect("offered"));
                    let x = gpu_core::provider::library::Operand { buf: &xn, range: (0, n as u64) };
                    gpu_core::provider::library::quantize_e4m3_rows(&gpu, &mut steps, &x, rows, d).expect("offered");
                } else {
                    steps.push(gpu.fused_step(Fused::AddRmsQuantE4m3, &[&ab, &bb, &wb, &sum, &xn, &q, &s], &[d, rows, EPS.to_bits(), 1 | if store_xn { 0 } else { 16 }]).expect("offered"));
                }
            }
            gpu.reset_kernel_times();
            gpu.submit(&[], &steps);
            gpu.poll_wait();
            let total: f64 = gpu.kernel_times().unwrap_or_default().iter().map(|r| r.1).sum::<f64>() / 20.0 * 1e3;
            println!("rows {rows:5} {label:40} {total:8.1} us");
        }
        gpu.set_kernel_timing(false);
    }
}
