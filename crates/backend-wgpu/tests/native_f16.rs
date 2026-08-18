// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! B11 - native f16 COMPUTE: correctness + the roofline gate, on real
//! hardware where available.
//!
//! This is a DIFFERENT tier from B4/B5's storage-tier `#w=bf16`/`#w=f16`
//! decode (which needs no device feature at all, since it stays fp32
//! arithmetic the whole time): `enable f16;` + real `f16`-typed registers,
//! gated on `wgpu::Features::SHADER_F16`. Everything here skips cleanly
//! (with a printed reason) when that feature genuinely isn't available,
//! rather than assuming either outcome.

use backend_wgpu::WgpuBackend;
use kernels::template::{native_f16_poc, native_f16_variant};

/// A harmless kernel used purely to stand up a device and check what it
/// supports before attempting to compile anything that needs `enable f16;`
/// (compiling an f16 kernel on a device that lacks the feature is a wgpu
/// validation failure, not a graceful `Result` - so capability must be
/// checked BEFORE that kernel is ever handed to the device, never after).
fn probe_device() -> WgpuBackend {
    WgpuBackend::new(&[("axpy", kernels::AXPY)])
}

/// Verifies the CPU JIT's actual behaviour on a native-f16 kernel - and the
/// real answer, found by actually compiling AND RUNNING it (not assumed from
/// reading `wgsl_cpu::Ty`'s variant list), is more dangerous than a clean
/// rejection: `wgsl_cpu::Jit::new` SUCCEEDS on this source. naga's
/// `ScalarKind::Float` does not carry bit width, so `Ty::from_scalar` maps
/// f16 (width 2) to the exact same `Ty::F32` arm as f32 (width 4), so the
/// CPU JIT silently executes every `f16` operation as plain fp32. Proven
/// with the overflow case from the correctness test below: `60000.0 * 1.0 +
/// 6000.0`, which a real f16 ALU must saturate to `+inf` (past f16's 65504
/// max), comes back from the CPU JIT as the un-saturated fp32 sum
/// `66000.0`, a silently WRONG answer rather than a compile error. This is why
/// `caps.numeric.f16` staying unconditionally `false` on `backend-cpu`
/// (checked next, `numeric_f16_never_entangles_across_backends`) is
/// structural insurance, not a redundant belt-and-suspenders: nothing about
/// this toolchain will refuse a native-f16 dispatch reaching the CPU JIT by
/// mistake, so the capability gate is the ONLY thing preventing a silent
/// wrong-answer bug, not a compiler error this crate could rely on instead.
/// No GPU needed for this one.
#[test]
fn native_f16_kernel_silently_diverges_on_the_cpu_jit_rather_than_being_rejected() {
    let (name, src) = native_f16_variant("elementwise_fma_f16", native_f16_poc::ELEMENTWISE_FMA);
    let jit = wgsl_cpu::Jit::new(&[(name, src)]).unwrap_or_else(|e| {
        panic!(
            "expected the CPU JIT to (wrongly) ACCEPT this source (naga's ScalarKind has no \
             width, so wsgl-cpu's Ty::from_scalar cannot distinguish f16 from f32) -- if it now \
             rejects it instead, wgsl-cpu grew real f16 awareness and this test (and the doc \
             comment on `native_f16_variant` explaining why the CPU backend must never reach this \
             tier) needs updating to match. Actual error: {e}"
        )
    });

    // Run it for real: overflow past f16's range must come back UN-saturated
    // (the fp32 sum, not +inf) - the concrete, numeric proof that this is a
    // silent semantic divergence, not merely "compiles but happens to still
    // be correct because f16 is a strict subset of f32's range".
    let backend = backend_cpu::CpuBackend::new(&[(name, src)]);
    let _ = jit; // `Jit::new` above is the structural check; `CpuBackend::new` re-derives its own `Jit` internally for the dispatch below.
    let a = backend.storage_init("a", &[60000.0]);
    let b = backend.storage_init("b", &[1.0]);
    let c = backend.storage_init("c", &[6000.0]);
    let out = backend.storage(1);
    let step = backend.step(0, &[&a, &b, &c, &out], &[1], 1);
    backend.submit(&[], &[step]);
    backend.poll_wait();
    let got = backend.read(&out, 1)[0];
    eprintln!(
        "CPU JIT ran the native-f16-labeled overflow case as plain fp32: got {got} (a real f16 \
         ALU would saturate to +inf here)"
    );
    assert_eq!(
        got, 66000.0,
        "expected the un-saturated fp32 sum (confirming the CPU JIT treats this as fp32, not \
         f16) -- got {got} instead, which would mean the divergence this test documents no \
         longer reproduces"
    );
}

/// The structural guarantee the divergence above makes load-bearing:
/// `backend-cpu`'s `NumericSupport.f16` is a hard-coded `false` (via
/// `..NumericSupport::BASELINE`, never touched by anything this backend
/// measures) and stays that way regardless of what THIS backend's own real
/// hardware measurement says - the two backends' capability structs must
/// never entangle. `backend-wgpu`'s own `numeric.f16` similarly never reads
/// `backend-cpu`'s caps; each backend's `query_caps` is self-contained. Both
/// halves checked directly against the real, live `caps()` each backend
/// reports, not re-derived/assumed.
#[test]
fn numeric_f16_never_entangles_across_backends() {
    use backend_api::Backend;
    let cpu = backend_cpu::CpuBackend::new(&[("axpy", kernels::AXPY)]);
    assert!(
        !cpu.caps().numeric.f16,
        "backend-cpu's NumericSupport.f16 must be unconditionally false -- B11 built the wgpu \
         roofline measurement specifically so it would NEVER be read by, or flip, the CPU \
         backend's own capability struct"
    );
    // A live wgpu device's own caps() is likewise untouched by this test's
    // f16 roofline measurement elsewhere in this file (`query_caps` builds
    // `numeric.f16` as a fixed `false` today, deliberately not wired to the
    // measurement -- see that function's own doc comment) -- checked here so
    // a future change that DOES wire the two together is forced to keep this
    // assertion true rather than silently starting to entangle them.
    let gpu = probe_device();
    assert!(!Backend::caps(&gpu).numeric.f16, "backend-wgpu's own numeric.f16 must stay the safe default too");
}

/// Pure decision-logic gate (B11's TDD item 2, the synthetic half): no GPU
/// needed. Confirms the threshold is real (a >1.0x ratio alone is not
/// enough) and that malformed/aborted measurements fail closed.
#[test]
fn f16_worth_enabling_gate_logic() {
    // Comfortably faster: enables.
    assert!(WgpuBackend::f16_worth_enabling(/* f32 */ 2.0, /* f16 */ 1.0));
    // Exactly at the threshold: enables (>=, not >).
    let f32_secs = 1.2;
    let f16_secs = 1.0;
    assert!(WgpuBackend::f16_worth_enabling(f32_secs, f16_secs));
    // Just under the threshold: refuses.
    assert!(!WgpuBackend::f16_worth_enabling(1.19, 1.0));
    // Equal (no speedup at all): refuses.
    assert!(!WgpuBackend::f16_worth_enabling(1.0, 1.0));
    // f16 SLOWER than f32 (the Pascal 1/64-rate trap, or a decode-bound
    // device): refuses, decisively.
    assert!(!WgpuBackend::f16_worth_enabling(1.0, 5.0));
    // Aborted/nonsensical measurements fail closed, never enable by accident.
    assert!(!WgpuBackend::f16_worth_enabling(f64::NAN, 1.0));
    assert!(!WgpuBackend::f16_worth_enabling(1.0, f64::NAN));
    assert!(!WgpuBackend::f16_worth_enabling(0.0, 1.0));
    assert!(!WgpuBackend::f16_worth_enabling(1.0, 0.0));
    assert!(!WgpuBackend::f16_worth_enabling(-1.0, 1.0));
}

/// Host reference: `half::f16`'s own (independently implemented) arithmetic,
/// not a hand-rolled bit trick -- deliberately a SECOND implementation from
/// the shader's, per this program's own "independent reference" discipline
/// (B5's ledger).
fn host_f16_fma(a: f32, b: f32, c: f32) -> f32 {
    let ah = half::f16::from_f32(a);
    let bh = half::f16::from_f32(b);
    let ch = half::f16::from_f32(c);
    (ah * bh + ch).to_f32()
}

/// f16's smallest NORMAL magnitude (`2^-14`) -- below this a value is
/// subnormal, where a real ALU may legally flush to zero (this is exactly
/// the class of hardware behaviour B5's own ledger found in the DECODE
/// path; this test checks whether it also applies to native COMPUTE, and
/// reports the answer honestly either way rather than assuming it).
const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;

/// Correctness: real hardware, real `f16` arithmetic, compared against an
/// independent host reference, at representative magnitudes -- near-zero
/// (subnormal), near f16's max (including a deliberate OVERFLOW), and
/// typical mid-range values, matching B5's own edge-case discipline for the
/// storage-tier decode, now applied to native compute instead.
#[test]
fn native_f16_elementwise_fma_matches_f32_reference_on_real_gpu() {
    let probe = probe_device();
    if !probe.supports_shader_f16() {
        brain_testutil::skip_unavailable("native_f16_elementwise_fma_matches_f32_reference_on_real_gpu: \
             this adapter does not report wgpu::Features::SHADER_F16");
        return;
    }
    eprintln!("running on a real wgpu device with SHADER_F16 available");

    let (name, src) = native_f16_variant("elementwise_fma_f16", native_f16_poc::ELEMENTWISE_FMA);
    let gpu = probe.new_like_device(&[(name, src)]);

    // (a, b, c, label) -- one case per magnitude regime.
    let cases: &[(f32, f32, f32, &str)] = &[
        (6.0e-6, 1.0, 0.0, "subnormal magnitude"),
        (-6.0e-6, 1.0, 0.0, "negative subnormal magnitude"),
        (60000.0, 1.0, 6000.0, "overflow past f16 max (65504)"),
        (1.5, -2.25, 0.75, "typical mid-range, exactly representable"),
        (0.0, 123.0, 0.0, "zero"),
        (100.25, 4.0, -1.5, "another typical mid-range value"),
    ];
    let n = cases.len() as u32;
    let a: Vec<f32> = cases.iter().map(|c| c.0).collect();
    let b: Vec<f32> = cases.iter().map(|c| c.1).collect();
    let c: Vec<f32> = cases.iter().map(|c| c.2).collect();

    let ab = gpu.storage_init("a", &a);
    let bb = gpu.storage_init("b", &b);
    let cb = gpu.storage_init("c", &c);
    let outb = gpu.storage(n as u64);
    let step = gpu.step(0, &[&ab, &bb, &cb, &outb], &[n], n);
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let got = gpu.read(&outb, n as usize);

    for (i, &(av, bv, cv, label)) in cases.iter().enumerate() {
        let want = host_f16_fma(av, bv, cv);
        let g = got[i];
        assert!(g.is_finite() || g.is_infinite(), "case {i} ({label}): got NaN, want {want}");
        if want.is_infinite() {
            assert!(
                g.is_infinite() && g.is_sign_positive() == want.is_sign_positive(),
                "case {i} ({label}): overflow must saturate to a same-signed infinity, got {g} want {want}"
            );
            eprintln!("case {i:>2} ({label:<38}): got {g:>12} want {want:>12}  [overflow, exact]");
            continue;
        }
        let want_abs = want.abs();
        if want_abs > 0.0 && want_abs < F16_MIN_NORMAL {
            // Subnormal target: accept either the (correctly rounded) real
            // value or a hardware flush-to-zero -- both are legitimate ALU
            // behaviours; which one this adapter actually does is reported,
            // not assumed.
            let flushed = g == 0.0;
            let close = (g - want).abs() <= want_abs * 0.25 + 1e-7;
            assert!(
                flushed || close,
                "case {i} ({label}): got {g}, want ~{want} (subnormal -- neither an exact match nor a flush-to-zero)"
            );
            eprintln!(
                "case {i:>2} ({label:<38}): got {g:>12e} want {want:>12e}  [{}]",
                if flushed { "flushed to zero" } else { "preserved" }
            );
            continue;
        }
        // Normal-range target: a real fp32-vs-hardware-fused-FMA tolerance,
        // matching B5's own precedent of a generous-but-bounded relative
        // margin rather than expecting bit-exact agreement (the shader may
        // compute a genuinely FUSED multiply-add -- one rounding -- while
        // the host reference above rounds twice).
        let tol = want_abs * (2f32).powi(-9) + 1e-4;
        assert!(
            (g - want).abs() <= tol,
            "case {i} ({label}): got {g}, want {want} (tol {tol})"
        );
        eprintln!("case {i:>2} ({label:<38}): got {g:>12.6} want {want:>12.6}  [ok, |err|={:.2e}]", (g - want).abs());
    }
}

/// The wall-clock cost of one submit, best-of-`reps`, `poll_wait`-bracketed
/// (the bracketing is load-bearing -- `submit` only enqueues on this
/// backend, so an unbracketed loop times host-side recording instead of the
/// device, exactly the trap `gpu_core::roof`'s own doc comment warns about).
fn best_of(gpu: &WgpuBackend, step: backend_wgpu::WgpuStep, reps: usize) -> f64 {
    gpu.submit(&[], std::slice::from_ref(&step));
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        gpu.submit(&[], std::slice::from_ref(&step));
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// Calibrates `iters` for kernel `kind` until the timed region clears
/// `MIN_SECS`, mirroring `gpu_core::roof::measure_compute`'s own discipline
/// (self-calibrating trip count so launch/drain overhead cannot dominate),
/// reimplemented locally here rather than reused: `backend-wgpu` cannot
/// depend on `gpu-core` (the dependency runs the other way -- `gpu-core`
/// depends on `backend-wgpu`), so this is the honest "measure inline in a
/// test" scope-down the phase brief itself allows when the existing
/// roofline infrastructure sits on the wrong side of a crate boundary for a
/// backend-local capability decision.
fn calibrated_seconds_per_iter(gpu: &WgpuBackend, kind: usize, inp: &wgpu::Buffer, out: &wgpu::Buffer, n: u32) -> (f64, u32) {
    const MIN_SECS: f64 = 0.05;
    const MAX_ITERS: u32 = 1 << 20;
    let (c_bits, d_bits) = (0.5f32.to_bits(), 0.5f32.to_bits());
    let mut iters: u32 = 256;
    loop {
        let step = gpu.step(kind, &[inp, out], &[n, iters, c_bits, d_bits], n);
        let secs = best_of(gpu, step, 3);
        if secs >= MIN_SECS || iters >= MAX_ITERS {
            return (secs / iters as f64, iters);
        }
        let want = (iters as f64 * MIN_SECS / secs.max(1e-9)).ceil();
        iters = (want as u32).max(iters.saturating_mul(2)).min(MAX_ITERS);
    }
}

/// The actual roofline measurement (B11's TDD item 2, the real half): times
/// the native-f16 FMA-chain kernel against the byte-for-byte-identical-shape
/// fp32 `kernels::ROOF_FMA` probe, on THIS sandbox's real adapter, and
/// reports -- honestly, whichever way it comes out -- whether native f16
/// clears `f16_worth_enabling`'s threshold here. This does not flip
/// `NumericSupport.f16` itself (see `query_caps`'s own doc comment for why
/// that stays a deliberate non-hot-path decision); it is the evidence a
/// human/ledger entry reads to answer "is native f16 compute worth it on
/// this hardware".
#[test]
fn native_f16_roof_fma_throughput_vs_f32_one_shot() {
    let probe = probe_device();
    if !probe.supports_shader_f16() {
        brain_testutil::skip_unavailable("native_f16_roof_fma_throughput_vs_f32_one_shot: \
             this adapter does not report wgpu::Features::SHADER_F16");
        return;
    }
    let (f16_name, f16_src) = native_f16_variant("roof_fma_f16", native_f16_poc::ROOF_FMA);
    let gpu = probe.new_like_device(&[("roof_fma_f32", kernels::ROOF_FMA), (f16_name, f16_src)]);

    const THREADS: u32 = 1 << 18;
    let inp = gpu.storage(THREADS as u64);
    let out = gpu.storage(THREADS as u64);
    gpu.write(&inp, &vec![1.0f32.to_bits(); THREADS as usize]);

    let (f32_secs_per_iter, f32_iters) = calibrated_seconds_per_iter(&gpu, 0, &inp, &out, THREADS);
    let (f16_secs_per_iter, f16_iters) = calibrated_seconds_per_iter(&gpu, 1, &inp, &out, THREADS);

    assert!(f32_secs_per_iter.is_finite() && f32_secs_per_iter > 0.0, "f32 probe produced a nonsensical time");
    assert!(f16_secs_per_iter.is_finite() && f16_secs_per_iter > 0.0, "f16 probe produced a nonsensical time");

    let ratio = f32_secs_per_iter / f16_secs_per_iter;
    let would_enable = WgpuBackend::f16_worth_enabling(f32_secs_per_iter, f16_secs_per_iter);
    eprintln!(
        "B11 roofline: fp32 {:.4} ns/iter ({} iters), native f16 {:.4} ns/iter ({} iters), \
         ratio (fp32/f16) = {:.4}x, threshold = {}x, numeric.f16 would be: {}",
        f32_secs_per_iter * 1e9,
        f32_iters,
        f16_secs_per_iter * 1e9,
        f16_iters,
        ratio,
        WgpuBackend::F16_COMPUTE_MIN_SPEEDUP,
        would_enable,
    );
    // No assertion on WHICH way the ratio falls -- "measure, never assume"
    // means this test's job is to produce and print the real number, not to
    // pin an outcome this program explicitly said must not be assumed in
    // advance.
}
