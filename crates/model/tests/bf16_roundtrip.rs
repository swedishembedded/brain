// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Weight::BF16` dual-backend roundtrip (B4) - the core deliverable of the
//! kernel templater phase.
//!
//! Packs a small f32 weight matrix via `model::half::pack_bf16` (through
//! `Weight::upload(.., Dtype::BF16)`, its one construction path), runs
//! `Ops::matmul` against it, and checks the result against an f32 host
//! reference - on BOTH the CPU JIT backend and a real wgpu GPU backend
//! (`Gpu::new_cpu`/`Gpu::new_wgpu`, `MOE_SKIP_GPU_TESTS` gates the GPU half
//! exactly like this tree's other dual-backend tests, e.g.
//! `crates/depth/tests/p3_fused_eval.rs`'s `fused_eval_gpu_matches_cpu`).
//! This is what proves the templated `#w=bf16` kernel sources
//! (`kernels::template::dtype_variant`) run correctly on both, with zero
//! device-feature requirement - the whole point of a storage-tier decode
//! expressed in plain integer/bitcast WGSL rather than a native bf16 type.
//!
//! **Shape sweep exercises all three templatized kernels**, by choosing
//! `(m, n)` that route through each of `select::candidates`'s `Op::MatMul`
//! variants (`backend_api::select`'s `DECODE_REGIME_MAX_ROWS=32`/
//! `GEMM_TILE_MIN_ROWS=8`/`GEMM_TILE_MIN_COLS=128` crossover constants):
//!
//! * `m=8,  n=128` (`m <= 32`) → `WorkgroupPerOutput` → `matmul_gemv#w=bf16`
//! * `m=64, n=128` (`m > 32`, `n >= 128`) → `RegisterTiled` → `matmul_reg3#w=bf16`
//! * `m=64, n=64`  (`m > 32`, `n < 128`) → `Reference` → `matmul#w=bf16`
//!
//! **Tolerance - the ULP-level math, explicit.** Only the WEIGHT narrows to
//! bf16 (`Ops::act` never quantizes for an `F32`/`BF16` weight - see
//! `model::ops`'s module doc); activations stay full f32. bf16 has 7 explicit
//! mantissa bits, round-to-nearest - [`model::half::f32_to_bf16`] - so each
//! stored weight's relative error is at most half a step of the 8th bit,
//! `2^-8` of its own magnitude. For `out[m,n] = sum_k x[m,k] * w[n,k]`, each
//! term's absolute error is therefore bounded by `|x[m,k] * w[n,k]| * 2^-8`
//! (only `w` rounds; `x` is exact), so the SUM's absolute error is bounded by
//! `2^-8 * sum_k |x[m,k] * w[n,k]|` - computed PER OUTPUT ELEMENT below (not
//! one flat epsilon, since the bound scales with each element's own
//! magnitude), plus a small absolute floor for elements whose true value is
//! near zero (where relative-to-itself bounds are meaningless).

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{Ops, Weight};

/// Every kernel `Ops::new` requires, plus the three bf16-storage variants
/// this test exercises - mirrors `model::ops::tests::kernel_list` (a private
/// test-only helper in the crate under test, so it can't be shared directly;
/// duplicated here the same way `ops_facade_parity.rs`'s own `KERNELS`
/// already duplicates that list for the `F32`/`I8`/`Q4` tiers).
fn kernel_list() -> Vec<(&'static str, &'static str)> {
    let bf16_matmul = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
    let bf16_gemv =
        kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
    let bf16_reg3 =
        kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
    vec![
        ("matmul", kernels::MATMUL),
        ("matmul_gemv", kernels::MATMUL_GEMV),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
        ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
        ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
        ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
        ("max_abs_row", kernels::MAX_ABS_ROW),
        ("quant_pack", kernels::QUANT_PACK),
        bf16_matmul,
        bf16_gemv,
        bf16_reg3,
    ]
}

fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

/// Per-output-element tolerance - see this file's module doc comment for the
/// derivation. `f64` accumulation so the tolerance's own arithmetic is not
/// itself the source of round-off being measured against.
fn bf16_tol(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut tol = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut abs_sum = 0f64;
            for i in 0..k {
                abs_sum += (x[r * k + i] as f64 * w[j * k + i] as f64).abs();
            }
            tol[r * n + j] = (abs_sum * 2f64.powi(-8)) as f32 + 1e-5;
        }
    }
    tol
}

/// Build a `Weight::BF16`, dispatch `Ops::matmul`, and check against the f32
/// host reference within [`bf16_tol`]. Consumes `gpu` (`Ops::new` takes
/// ownership) so the caller passes a freshly-built one per shape/backend.
fn check_bf16_matmul(gpu: Gpu, m: usize, n: usize, k: usize, seed: u64, label: &str) {
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);
    let x = g.storage_init("x", &x_h);
    let weight = Weight::upload(&ops, &w_h, n, k, Dtype::BF16);
    assert_eq!(
        weight.dtype(),
        Dtype::BF16,
        "{label} m={m} n={n} k={k}: this device must report bf16_storage for Weight::upload to \
         land on the BF16 tier (see backend-wgpu/backend-cpu's NumericSupport construction)"
    );

    let mut steps = Vec::new();
    let act = ops.act(&mut steps, &x, 0, m as u32, k as u32);
    let out = g.storage((m * n) as u64);
    ops.matmul(&mut steps, &weight, &act, &out, 0);
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    let want = host_matmul(&x_h, &w_h, m, k, n);
    let tol = bf16_tol(&x_h, &w_h, m, k, n);
    assert_eq!(got.len(), want.len());
    let mut worst: f32 = 0.0;
    for i in 0..got.len() {
        let err = (got[i] - want[i]).abs();
        worst = worst.max(err / tol[i].max(1e-12));
        assert!(
            err <= tol[i],
            "{label} m={m} n={n} k={k}: elem {i} got {} want {} (err {err}, tol {})",
            got[i],
            want[i],
            tol[i]
        );
    }
    eprintln!("{label} m={m} n={n} k={k}: worst err/tol ratio {worst:.4}");
}

/// `(m, n, k, tag)` - see this file's module doc comment for which
/// `KernelVariant`/kernel each shape routes through.
const SHAPES: &[(usize, usize, usize, &str)] =
    &[(8, 128, 128, "workgroup_per_output/matmul_gemv#w=bf16"), (64, 128, 128, "register_tiled/matmul_reg3#w=bf16"), (64, 64, 128, "reference/matmul#w=bf16")];

/// **A real finding, checked rather than just noted in prose.** The CPU JIT
/// (`crates/wgsl-cpu`) cannot compile a multi-barrier work-group kernel
/// (`matmul_reg3#w=bf16` has three, same as `matmul_reg2`/`matmul_i8_dyn`
/// before it - confirmed by the `wgsl-cpu: kernel "matmul_reg3#w=bf16" not
/// JIT-compiled` stderr line this test suite prints), and `backend-cpu`'s own
/// `DeviceCaps.workgroup_reductions` is `false` for exactly that reason (see
/// its own doc comment: "The split-at-barrier JIT mis-executes the
/// workgroup-cooperative reduction kernels"). `RegisterTiled` REQUIRES
/// `workgroup_reductions` (`backend_api::select::KernelVariant::requires`),
/// so `select::candidates` filters it out on CPU caps and falls back to
/// `Reference` - `matmul_reg3#w=bf16` is therefore NEVER actually dispatched
/// on the CPU backend; the `register_tiled` shape below silently runs through
/// `matmul#w=bf16` there instead. This is the SAME pre-existing behaviour
/// `F32`'s own `matmul_reg2`/`matmul_i8_dyn` already have on CPU (B2's own
/// ledger: "on CPU all of them route to the AVX2 gemm... GPU-only") - not a
/// gap this phase introduced, but worth asserting explicitly rather than
/// silently relying on it, since it is exactly the kind of thing that looks
/// like a bug report waiting to happen.
#[test]
fn register_tiled_bf16_is_not_reachable_on_the_cpu_backend() {
    use gpu_core::select::{self, KernelSelector, Op, OpShape};
    let gpu = Gpu::new_cpu(&kernel_list());
    let caps = gpu.caps();
    assert!(!caps.workgroup_reductions, "backend-cpu's caps must still report workgroup_reductions=false");
    let shape = OpShape { m: 64, n: 128, k: 128, dtype: Dtype::BF16 };
    let variant = select::DefaultSelector.select(Op::MatMul, shape, &caps);
    assert_eq!(
        variant,
        select::KernelVariant::Reference,
        "CPU caps must fall back to Reference for a shape that would pick RegisterTiled on a real GPU"
    );
}

/// The `register_tiled` shape's TAG names the kernel a real GPU dispatches
/// for it; on CPU it silently runs through `Reference` instead (see
/// [`register_tiled_bf16_is_not_reachable_on_the_cpu_backend`]) - still a
/// valid, useful check (CPU output must still match the f32 host reference),
/// just not proof that `matmul_reg3#w=bf16` itself executed correctly. Only
/// the GPU run (`bf16_matmul_matches_f32_reference_on_gpu`) proves that.
#[test]
fn bf16_matmul_matches_f32_reference_on_cpu() {
    for &(m, n, k, tag) in SHAPES {
        let list = kernel_list();
        let gpu = Gpu::new_cpu(&list);
        check_bf16_matmul(gpu, m, n, k, 0xB16_0000 ^ (m as u64) << 16 ^ n as u64, &format!("cpu/{tag}"));
    }
}

/// Real GPU execution - gated by `MOE_SKIP_GPU_TESTS` like every other
/// dual-backend test in this tree. Prints which branch actually ran (real
/// wgpu execution vs a clean skip) so a report can state which happened
/// rather than assume.
#[test]
fn bf16_matmul_matches_f32_reference_on_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("bf16_matmul_matches_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("bf16_matmul_matches_f32_reference_on_gpu: running on a real wgpu device");
    for &(m, n, k, tag) in SHAPES {
        let list = kernel_list();
        let gpu = Gpu::new_wgpu(&list);
        check_bf16_matmul(gpu, m, n, k, 0xB16_0000 ^ (m as u64) << 16 ^ n as u64, &format!("gpu/{tag}"));
    }
}
