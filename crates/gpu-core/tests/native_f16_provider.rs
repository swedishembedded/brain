// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.7 - `NativeF16Provider` on REAL hardware: the measured gate, forward-
//! pass parity against the WGSL fp32 reference, and the documented
//! subnormal flush-to-zero finding, all against a live device. Skips cleanly
//! (with a printed reason) wherever `Gpu::supports_native_f16` is false,
//! matching `crates/backend-wgpu/tests/native_f16.rs`'s own discipline -
//! never assuming either outcome.

use backend_api::select::{Dtype, Op, OpShape};
use backend_api::DType;
use gpu_core::provider::native_f16::NativeF16Provider;
use gpu_core::provider::parity::{assert_provider_parity, ParityCase, Tolerance};
use gpu_core::provider::{LowerCtx, Operand, OpRequest, OperatorProvider, Pass, Role};
use gpu_core::Gpu;

fn kernels() -> [(&'static str, &'static str); 2] {
    [("matmul", kernels::MATMUL), gpu_core::provider::native_f16::kernel()]
}

/// A live wgpu device with both the WGSL reference `matmul` kernel and
/// `NativeF16Provider`'s own `matmul_reg3_f16n` registered. `None` when this
/// adapter never asked for `enable f16;` support - compiling it anyway would
/// be a hard device-fault panic, not a graceful skip (see
/// `backend_api::Backend::supports_native_f16`'s own doc comment).
fn device() -> Option<Gpu> {
    let gpu = Gpu::new_wgpu(&kernels());
    if !gpu.supports_native_f16() {
        return None;
    }
    Some(gpu)
}

fn bind(_v: backend_api::select::KernelVariant) -> (usize, &'static str) {
    panic!("NativeF16Provider never calls OpRequest::bind -- it resolves its own fixed kernel")
}

/// The real measurement: `NativeF16Provider::probe`'s own fp32-vs-f16
/// GEMM-shaped GFLOP/s comparison on this sandbox's real adapter. Prints the
/// honest number either way - "either outcome is fine" is this milestone's
/// own explicit instruction, so nothing here asserts which way the ratio
/// falls.
#[test]
fn native_f16_matmul_measured_speedup_reported_honestly() {
    let Some(gpu) = device() else {
        brain_testutil::skip_unavailable(
            "native_f16_matmul_measured_speedup_reported_honestly: this adapter never asked for \
             wgpu::Features::SHADER_F16",
        );
        return;
    };
    let provider = NativeF16Provider::probe(&gpu);
    match provider.measured_speedup() {
        Some(s) => eprintln!(
            "NativeF16Provider::probe on this real adapter: speedup_vs_f32 = {s:.3}x \
             (FAST_TIER_MIN_SPEEDUP = {}x) -- provider {}",
            backend_api::arch::FAST_TIER_MIN_SPEEDUP,
            if s as f64 >= backend_api::arch::FAST_TIER_MIN_SPEEDUP { "WOULD be selected" } else { "would NOT be selected" }
        ),
        None => eprintln!(
            "NativeF16Provider::probe on this real adapter: no measurement (probe aborted or \
             timed out) -- provider would NOT be selected, fail-closed"
        ),
    }
}

fn numeric_cases() -> Vec<ParityCase> {
    // Same four shapes `provider::parity::MATMUL_CASES` uses (decode-shaped,
    // the 128-tile crossover, a large multi-tile shape, a non-tile-multiple
    // shape - each seed also drives a non-zero row offset via
    // `assert_matmul_parity`'s own `xr0` derivation), at a NUMERIC tolerance
    // instead of bit-identical: a real f16-narrowed multiply reassociates
    // rounding, so bit-identity is the wrong bar here (see
    // `provider::parity::Tolerance`'s own doc comment).
    //
    // `atol = 1e-2`: measured on this real adapter, the 300x260x128 case's
    // own near-zero output elements (an fp32 sum that happens to land close
    // to zero from sign cancellation across ~128 random-signed terms) show
    // the largest absolute deviation from the fp32 reference - up to ~3.1e-3
    // on a ~6.4e-3-magnitude element - because a near-zero OUTPUT is
    // inherently the highest-relative-error regime for ANY reduced-precision
    // reassociation (the per-element check's `rtol` term alone cannot cover
    // it at a value this close to zero). `1e-2` covers that measured worst
    // case with margin while staying two orders of magnitude below the
    // typical (~1-10 magnitude) output this shape produces - a real,
    // measured bound, not a guess.
    let tol = Tolerance::Numeric { atol: 1e-2, rtol: 3e-2, cosine_min: 0.999 };
    vec![
        ParityCase { op: Op::MatMul, shape: OpShape { m: 1, n: 64, k: 64, dtype: Dtype::F32 }, pass: Pass::Forward, seed: 1, tol },
        ParityCase {
            op: Op::MatMul,
            shape: OpShape { m: 128, n: 128, k: 128, dtype: Dtype::F32 },
            pass: Pass::Forward,
            seed: 2,
            tol,
        },
        ParityCase {
            op: Op::MatMul,
            shape: OpShape { m: 300, n: 260, k: 128, dtype: Dtype::F32 },
            pass: Pass::Forward,
            seed: 3,
            tol,
        },
        ParityCase { op: Op::MatMul, shape: OpShape { m: 37, n: 53, k: 17, dtype: Dtype::F32 }, pass: Pass::Forward, seed: 4, tol },
    ]
}

/// Forward-pass parity against the WGSL fp32 reference, at a numeric
/// tolerance, across decode/crossover/large/ragged shapes - reuses
/// `provider::parity::assert_provider_parity` UNCHANGED (it never calls
/// `OpRequest::bind` on this provider's side, so the shared harness's own
/// `KernelVariant::Reference`-only `bind` closure is safe for both sides).
/// Skips (both the missing-feature AND the measured-not-fast cases) rather
/// than asserting a specific measurement outcome - see this milestone's own
/// "either outcome is fine, report honestly" instruction.
#[test]
fn native_f16_matmul_forward_parity_against_wgsl_reference() {
    let Some(gpu) = device() else {
        brain_testutil::skip_unavailable(
            "native_f16_matmul_forward_parity_against_wgsl_reference: this adapter never asked \
             for wgpu::Features::SHADER_F16",
        );
        return;
    };
    let provider = NativeF16Provider::probe(&gpu);
    if provider.measured_speedup().map(f64::from).is_none_or(|s| s < backend_api::arch::FAST_TIER_MIN_SPEEDUP) {
        brain_testutil::skip_unavailable(&format!(
            "native_f16_matmul_forward_parity_against_wgsl_reference: measured speedup {:?} does \
             not clear FAST_TIER_MIN_SPEEDUP on this adapter -- this provider would not be \
             selected here, so there is nothing to gate parity on",
            provider.measured_speedup()
        ));
        return;
    }
    for case in numeric_cases() {
        assert_provider_parity(&gpu, &provider, &case);
    }
}

/// f16's smallest NORMAL magnitude (`2^-14`) - below this a value is
/// subnormal. Same constant `crates/backend-wgpu/tests/native_f16.rs` uses.
const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;

/// The documented hardware fact (`kernel-performance.md`'s B11/M8.7 entries:
/// this Intel Arc iGPU's native-f16 ALU flushes a subnormal RESULT to zero),
/// pinned directly against this provider's own GEMM kernel rather than
/// silently accepting either outcome the way a generic correctness test
/// might. `a * b` lands well under `F16_MIN_NORMAL` by construction (checked
/// below), and the SAME `f16(a) * f16(b)` primitive
/// `native_f16_poc::ELEMENTWISE_FMA` already measured this behaviour on runs
/// this GEMM kernel's own inner loop.
#[test]
fn native_f16_matmul_subnormal_product_matches_documented_flush_to_zero() {
    let Some(gpu) = device() else {
        brain_testutil::skip_unavailable(
            "native_f16_matmul_subnormal_product_matches_documented_flush_to_zero: this adapter \
             never asked for wgpu::Features::SHADER_F16",
        );
        return;
    };
    let provider = NativeF16Provider::probe(&gpu);
    if provider.measured_speedup().map(f64::from).is_none_or(|s| s < backend_api::arch::FAST_TIER_MIN_SPEEDUP) {
        brain_testutil::skip_unavailable(
            "native_f16_matmul_subnormal_product_matches_documented_flush_to_zero: measured \
             speedup does not clear FAST_TIER_MIN_SPEEDUP on this adapter",
        );
        return;
    }

    let (a, b) = (0.006_f32, 0.005_f32);
    let ah = half::f16::from_f32(a);
    let bh = half::f16::from_f32(b);
    let want = (ah * bh).to_f32();
    assert!(
        want.abs() > 0.0 && want.abs() < F16_MIN_NORMAL,
        "test input must actually target the f16 subnormal regime, got {want:e}"
    );

    let x = gpu.storage_init("subnormal_x", &[a]);
    let w = gpu.storage_init("subnormal_w", &[b]);
    let y = gpu.storage(1);
    let operands = [
        Operand { role: Role::Act, buf: &x, range: (0, 1), dtype: DType::F32 },
        Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &y, range: (0, 1), dtype: DType::F32 },
    ];
    let req = OpRequest {
        op: Op::MatMul,
        shape: OpShape { m: 1, n: 1, k: 1, dtype: Dtype::F32 },
        pass: Pass::Forward,
        operands: &operands,
        attrs: &[1, 1, 1],
        bind: &bind,
    };
    let mut steps = Vec::new();
    let caps = gpu.caps();
    {
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        provider.lower(&mut ctx, &req).expect("NativeF16Provider must lower this shape");
    }
    gpu.submit(&[], &steps);
    let got = gpu.read(&y, 1)[0];

    eprintln!(
        "native f16 matmul subnormal case: a={a} b={b} want(host half::f16)={want:e} \
         got(device)={got:e}"
    );
    assert_eq!(
        got, 0.0,
        "this Intel Arc iGPU (MTL) is documented (kernel-performance.md's B11/M8.7 entries) to \
         flush subnormal f16 results to zero -- if this now preserves the subnormal value \
         instead, the documented hardware behaviour changed and this assertion (a specific, real \
         outcome, not a permissive 'either') must be updated to match"
    );
}
