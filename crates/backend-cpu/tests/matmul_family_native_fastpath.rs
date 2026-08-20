// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native-CPU-fast-path correctness for the matmul family that the Cranelift
//! JIT (`brain-wgsl-cpu`) cannot compile - every kernel whose WGSL uses more
//! than one top-level `workgroupBarrier()` (a tiled GEMM's staged
//! load/compute/store), which `wgsl_cpu::Jit::new` skips with a warning
//! ("not JIT-compiled ... must use a native fast path or the GPU") rather
//! than a hard error, leaving that kernel's compiled slot `None`.
//! `wgsl_cpu::Jit::run` panics if anything ever dispatches one of those
//! `None` slots, so on the CPU backend these kernels are ONLY safe to
//! register if `backend-cpu`'s own `dispatch()` routes them to a hand-written
//! Rust fast path (`fast_ops`/`fast_conv`) by kernel identity, bypassing the
//! JIT entirely - see `crates/backend-cpu/src/lib.rs`'s `FastIdx`.
//!
//! This was investigated (not merely assumed) after a real training run
//! (`qwen35::stream_train_step`, forced onto `Gpu::new_cpu` because its
//! resident fp32 `lm_head` exceeds this box's Vulkan `max_buffer_size`)
//! printed exactly this warning for `matmul_i8_dyn`, `matmul_reg2`,
//! `matmul_reg3#w=bf16` and `matmul_reg3#w=f16`, and the run did not panic -
//! which only proves those specific dtype-templated names were never
//! DISPATCHED in that run (confirmed separately: `qwen35::model::pipelines`
//! registers the bf16/f16 storage tiers only because `Ops::REQUIRED_KERNELS`
//! demands the full façade set, and that crate never builds a
//! `Weight::BF16`/`Weight::F16` weight to dispatch them against). It says
//! nothing about whether the NATIVE PATH the plain (non-templated)
//! `matmul_reg2`/`matmul_reg3` names route to is actually correct, or
//! whether the same "JIT refuses, must use native or GPU" situation exists
//! for other kernels nobody happened to check. This file settles both:
//! dispatches every JIT-uncompilable matmul-family kernel through the real
//! `backend_cpu::CpuBackend` (via `gpu_core::Gpu::new_cpu`, the exact
//! construction path a training run uses) and compares the result against a
//! from-first-principles scalar reference, with real numeric tolerance.

use data::rng::Lcg;
use gpu_core::Gpu;

const TOL: f32 = 1e-4;

/// `out[M,N] = A[M,K] @ B[N,K]ᵀ` - the contract every kernel in the forward
/// matmul family shares (`matmul.wgsl`'s own header).
fn matmul_abt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ni * k + ki];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// `dX[M,K] = dY[M,N] @ W[N,K]` (no accumulate - `acc=false`).
fn matmul_dx_ref(dy: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut dx = vec![0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut acc = 0f32;
            for ni in 0..n {
                acc += dy[mi * n + ni] * w[ni * k + ki];
            }
            dx[mi * k + ki] = acc;
        }
    }
    dx
}

/// `dW[N,K] = sum_m dY[M,N]ᵀ @ X[M,K]`.
fn matmul_dw_ref(dy: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut dw = vec![0f32; n * k];
    for ni in 0..n {
        for ki in 0..k {
            let mut acc = 0f32;
            for mi in 0..m {
                acc += dy[mi * n + ni] * x[mi * k + ki];
            }
            dw[ni * k + ki] = acc;
        }
    }
    dw
}

fn worst_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).fold(0f32, |acc, (a, b)| acc.max((a - b).abs()))
}

/// Every kernel this test exercises must have >1 top-level `workgroupBarrier()`
/// (i.e. genuinely be one the CPU JIT cannot compile) - otherwise this file
/// would be silently testing the JIT path instead of the native fast path it
/// claims to.
fn assert_jit_uncompilable(name: &'static str, src: &'static str) {
    let barriers = src.matches("workgroupBarrier").count();
    assert!(
        barriers > 1,
        "{name}: has {barriers} top-level workgroupBarrier() (expected >1) - this test's premise \
         (\"the CPU JIT refuses this kernel, so a native fast path is the only way it can run here\") \
         does not hold for it; if the kernel changed, move it out of this file"
    );
    let jit = wgsl_cpu::Jit::new(&[(name, src)]).unwrap_or_else(|e| {
        panic!("{name}: Jit::new returned a hard error ({e}) instead of the soft multi-barrier skip")
    });
    let idx = jit.index_of(name).expect("registered");
    // The compiled slot must be exactly `None` (soft skip) - proven by calling
    // over an EMPTY invocation range (`Jit::run`'s entry block loads each
    // binding's base pointer before the loop body runs, so real backing
    // memory is required, but nothing is read/written when start==end==0) and
    // asserting the documented panic fires. Same technique
    // `wgsl-cpu/tests/compile_all.rs::kernel_is_compiled` uses.
    let bufs: [*mut u8; 8] = [std::ptr::null_mut(); 8];
    let uniform = [0u32; 16];
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        jit.run(idx, 0, 0, 1, 1, uniform.as_ptr(), bufs.as_ptr());
    }));
    std::panic::set_hook(hook);
    assert!(
        result.is_err(),
        "{name}: Jit::run did not panic on an uncompiled slot - the JIT compiled it after all, \
         so backend-cpu's native fast path is no longer the only way to run it (harmless, but this \
         test's premise changed)"
    );
}

/// The forward matmul family: `matmul` (JIT-compilable, 0 barriers - the
/// reference every native routing choice is checked against) plus every
/// tiled variant the JIT refuses (`matmul_tiled` 2 barriers, `matmul_reg` 2,
/// `matmul_reg2` 3, `matmul_reg3` 3 - the exact kernel the incident's warning
/// named). `backend-cpu::dispatch` special-cases all five BY KERNEL IDENTITY
/// to the same AVX2 `fast_ops::matmul_abt` ("the one-graph rule": a model may
/// register whichever variant suits its shapes without forking its CPU
/// path) - this proves that routing is real and correct, not merely
/// non-panicking.
#[test]
fn forward_matmul_family_native_fastpath_matches_scalar_reference() {
    assert_jit_uncompilable("matmul_tiled", kernels::MATMUL_TILED);
    assert_jit_uncompilable("matmul_reg", kernels::MATMUL_REG);
    assert_jit_uncompilable("matmul_reg2", kernels::MATMUL_REG2);
    assert_jit_uncompilable("matmul_reg3", kernels::MATMUL_REG3);

    let ks: &[(&str, &str)] = &[
        ("matmul", kernels::MATMUL),
        ("matmul_tiled", kernels::MATMUL_TILED),
        ("matmul_reg", kernels::MATMUL_REG),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("matmul_reg3", kernels::MATMUL_REG3),
    ];
    let gpu = Gpu::new_cpu(ks);
    let mut seed = Lcg::new(0x5EED);

    // Shapes deliberately NOT multiples of the 128x128 tile (so the guarded
    // boundary loads/stores are exercised, not just a perfectly-filled tile),
    // plus one shape that IS large enough to span several tiles.
    for (m, k, n) in [(1usize, 3usize, 1usize), (5, 7, 9), (37, 20, 41), (200, 33, 260)] {
        let a: Vec<f32> = (0..m * k).map(|_| seed.scaled(0.5)).collect();
        let b: Vec<f32> = (0..n * k).map(|_| seed.scaled(0.5)).collect();
        let want = matmul_abt(&a, &b, m, k, n);

        for name in ["matmul", "matmul_tiled", "matmul_reg", "matmul_reg2", "matmul_reg3"] {
            let ab = gpu.storage_init("a", &a);
            let bb = gpu.storage_init("b", &b);
            let ob = gpu.storage((m * n) as u64);
            let kind = gpu.kernel_index(name).expect("registered above");
            let steps = vec![gpu.step(kind, &[&ab, &bb, &ob], &[m as u32, k as u32, n as u32], (m * n) as u32)];
            gpu.submit(&[], &steps);
            let got = gpu.read(&ob, m * n);
            let w = worst_abs(&got, &want);
            assert!(w < TOL, "{name} m={m} k={k} n={n}: worst|Δ|={w} >= {TOL}");
        }
    }
}

/// The backward matmul family: `matmul_dx`/`matmul_dw` (JIT-compilable, 0
/// barriers) and their tiled `_reg` siblings (`matmul_dx_reg`/`matmul_dw_reg`,
/// 3 barriers each - JIT-uncompilable). Unlike the forward family, no
/// existing test in this workspace dispatches these tiled kernels' NAMES on
/// the CPU backend and checks the output: `gpu-core/tests/bench_backward.rs`
/// only ever builds `Gpu::new_wgpu` for them. This closes that gap - the
/// "check whether this situation applies to any OTHER kernel" the same
/// incident raised for the forward family.
#[test]
fn backward_matmul_family_native_fastpath_matches_scalar_reference() {
    assert_jit_uncompilable("matmul_dx_reg", kernels::MATMUL_DX_REG);
    assert_jit_uncompilable("matmul_dw_reg", kernels::MATMUL_DW_REG);

    let ks: &[(&str, &str)] = &[
        ("matmul_dx", kernels::MATMUL_DX),
        ("matmul_dx_reg", kernels::MATMUL_DX_REG),
        ("matmul_dw", kernels::MATMUL_DW),
        ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ];
    let gpu = Gpu::new_cpu(ks);
    let mut seed = Lcg::new(0xBEEF);

    for (m, k, n) in [(3usize, 5usize, 2usize), (37, 20, 41), (200, 33, 260)] {
        let dy: Vec<f32> = (0..m * n).map(|_| seed.scaled(0.5)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| seed.scaled(0.5)).collect();
        let x: Vec<f32> = (0..m * k).map(|_| seed.scaled(0.5)).collect();
        let want_dx = matmul_dx_ref(&dy, &w, m, k, n);
        let want_dw = matmul_dw_ref(&dy, &x, m, k, n);

        let dyb = gpu.storage_init("dy", &dy);
        let wb = gpu.storage_init("w", &w);
        let xb = gpu.storage_init("x", &x);

        for name in ["matmul_dx", "matmul_dx_reg"] {
            let dxb = gpu.storage((m * k) as u64);
            let kind = gpu.kernel_index(name).expect("registered above");
            // acc=0 (overwrite, not accumulate).
            let steps =
                vec![gpu.step(kind, &[&dyb, &wb, &dxb], &[m as u32, k as u32, n as u32, 0], (m * k) as u32)];
            gpu.submit(&[], &steps);
            let got = gpu.read(&dxb, m * k);
            let w_ = worst_abs(&got, &want_dx);
            assert!(w_ < TOL, "{name} m={m} k={k} n={n}: worst|Δ|={w_} >= {TOL}");
        }

        for name in ["matmul_dw", "matmul_dw_reg"] {
            let dwb = gpu.storage((n * k) as u64);
            let kind = gpu.kernel_index(name).expect("registered above");
            let steps = vec![gpu.step(kind, &[&dyb, &xb, &dwb], &[m as u32, k as u32, n as u32], (n * k) as u32)];
            gpu.submit(&[], &steps);
            let got = gpu.read(&dwb, n * k);
            let w_ = worst_abs(&got, &want_dw);
            assert!(w_ < TOL, "{name} m={m} k={k} n={n}: worst|Δ|={w_} >= {TOL}");
        }
    }
}

/// `matmul_i8_dyn` is the ONE matmul-family member with NO native CPU fast
/// path at all (absent from `backend-cpu`'s `FastIdx`) - confirmed by
/// `kernels/matmul_i8_dyn.wgsl`'s own header (`@cpu no`) and independently
/// here. Dispatching it on the CPU backend would panic. This is safe only
/// because it is structurally *unreachable* there: the real,
/// capability-gated selector (`gpu_core::select::candidates`) requires
/// `caps.numeric.int8_dot` for `KernelVariant::PackedInt8`
/// (`matmul_i8_dyn`'s physical kernel), and `backend-cpu`'s own `caps()`
/// reports `int8_dot: false` ("no VNNI fast path yet") - so `candidates`
/// never returns `PackedInt8` for CPU caps, at ANY shape. That is an
/// architectural guarantee, not a lucky absence of a call site: this test
/// proves it directly against the real CPU backend's own `caps()`, not a
/// synthetic stand-in.
#[test]
fn matmul_i8_dyn_has_no_cpu_native_fastpath_and_is_unreachable_by_the_selector() {
    assert_jit_uncompilable("matmul_i8_dyn", kernels::MATMUL_I8_DYN);

    use backend_api::Backend;
    let cpu = backend_cpu::CpuBackend::new(&[("matmul_i8_dyn", kernels::MATMUL_I8_DYN)]);
    let caps = cpu.caps();
    assert!(!caps.numeric.int8_dot, "backend-cpu now reports int8_dot=true; matmul_i8_dyn may be reachable");

    use gpu_core::select::{candidates, Dtype, Op, OpShape};
    for m in [1u32, 8, 32, 33, 128, 4096] {
        let shape = OpShape { m, n: 512, k: 512, dtype: Dtype::I8 };
        let cands = candidates(Op::MatMul, shape, &caps);
        assert!(
            !cands.contains(&gpu_core::select::KernelVariant::PackedInt8),
            "m={m}: candidates() returned PackedInt8 (-> matmul_i8_dyn) for CPU caps, which has no \
             native fast path for it - this would panic on real dispatch"
        );
    }
}
