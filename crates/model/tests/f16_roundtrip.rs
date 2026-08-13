// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Weight::F16` dual-backend roundtrip (B5) - the harder storage-tier sibling
//! of B4's bf16 roundtrip (`crates/model/tests/bf16_roundtrip.rs`, this
//! file's structural template).
//!
//! bf16 narrowing is *exact* (one shift, no rounding on the exponent, no
//! denormal/inf/NaN special-casing, since bf16 shares f32's full 8-bit
//! exponent range). Real f16 (binary16) is harder: a 5-bit exponent means
//! narrowing/widening needs actual re-biasing, and f16's exponent range is
//! much narrower than f32's, so an ordinary weight matrix can genuinely
//! overflow (saturate to `+-inf`) or underflow (flush to subnormal/zero) on
//! the way down to f16 - unlike bf16, where that never happens. This test
//! deliberately injects a subnormal-magnitude weight row and a
//! near-f16-ceiling weight row (see [`inject_f16_edge_case_rows`]) so the
//! sweep would actually catch a broken decode at those extremes, not just
//! typical mid-range values.
//!
//! Packs a small f32 weight matrix via `model::half::pack_f16` (through
//! `Weight::upload(.., Dtype::F16)`, its one construction path), runs
//! `Ops::matmul` against it, and checks the result against an f32 host
//! reference - on BOTH the CPU JIT backend and a real wgpu GPU backend
//! (`Gpu::new_cpu`/`Gpu::new_wgpu`, `MOE_SKIP_GPU_TESTS` gates the GPU half
//! exactly like `bf16_roundtrip.rs` and this tree's other dual-backend
//! tests). This is what proves the templated `#w=f16` kernel sources
//! (`kernels::template::dtype_variant`'s magic-multiply decode) run
//! correctly on both, with zero device-feature requirement.
//!
//! **Shape sweep exercises all three templatized kernels**, exactly the same
//! `(m, n)` crossover points `bf16_roundtrip.rs` uses (`backend_api::select`'s
//! `DECODE_REGIME_MAX_ROWS=32`/`GEMM_TILE_MIN_ROWS=8`/`GEMM_TILE_MIN_COLS=128`):
//!
//! * `m=8,  n=128` (`m <= 32`) → `WorkgroupPerOutput` → `matmul_gemv#w=f16`
//! * `m=64, n=128` (`m > 32`, `n >= 128`) → `RegisterTiled` → `matmul_reg3#w=f16`
//! * `m=64, n=64`  (`m > 32`, `n < 128`) → `Reference` → `matmul#w=f16`
//!
//! **Tolerance - the ULP-level math, explicit.** Only the WEIGHT narrows to
//! f16 (`Ops::act` never quantizes for an `F32`/`F16` weight - see
//! `model::ops`'s module doc); activations stay full f32. f16 has 10 explicit
//! mantissa bits, round-to-nearest ([`model::half::f32_to_f16`], which
//! delegates to the `half` crate) - so each stored weight's relative error is
//! at most half a step of the 11th bit, `2^-11` of its own magnitude (vs
//! bf16's `2^-8` - f16 keeps 3 more mantissa bits, at the cost of a much
//! narrower exponent range, which is exactly what the edge-case rows below
//! exercise). For `out[m,n] = sum_k x[m,k] * w[n,k]`, each term's absolute
//! error is bounded by `|x[m,k] * w[n,k]| * 2^-11` (only `w` rounds; `x` is
//! exact), so the SUM's absolute error is bounded by `2^-11 * sum_k
//! |x[m,k] * w[n,k]|` - computed PER OUTPUT ELEMENT below, plus a small
//! absolute floor for elements whose true value is near zero.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{Ops, Weight};

/// Every kernel `Ops::new` requires, plus the three bf16-storage and three
/// f16-storage variants (`REQUIRED_KERNELS` now lists both tiers) - mirrors
/// `model::ops::tests::kernel_list` (a private test-only helper in the crate
/// under test, so it can't be shared directly; duplicated here the same way
/// `bf16_roundtrip.rs`/`ops_facade_parity.rs` already duplicate this list).
fn kernel_list() -> Vec<(&'static str, &'static str)> {
    let bf16_matmul = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
    let bf16_gemv =
        kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
    let bf16_reg3 =
        kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
    let f16_matmul = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::F16).unwrap();
    let f16_gemv = kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::F16).unwrap();
    let f16_reg3 = kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::F16).unwrap();
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
        f16_matmul,
        f16_gemv,
        f16_reg3,
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

/// Overwrites weight row 0 with subnormal-magnitude values and row 1 with
/// large-magnitude values near f16's ceiling (65504.0) - the phase brief's
/// own edge-case requirement. Without this, an all-mid-range-random weight
/// matrix (`Lcg::vec_scaled(.., 1.0)`, uniform in `[-1, 1)`) would never
/// touch f16's subnormal or near-overflow encoding at all, and a broken
/// magic-multiply decode at those extremes would slip past this test
/// unnoticed - exactly the gap this phase's brief called out explicitly.
fn inject_f16_edge_case_rows(w: &mut [f32], n: usize, k: usize) {
    assert!(n >= 2, "inject_f16_edge_case_rows needs at least 2 output rows (n={n})");
    // Row 0: subnormal magnitude. f16 subnormals live in (0, 2^-14); every
    // value here is a small multiple of 2^-20, well inside that range and
    // alternating sign so the row is not degenerate.
    for (i, v) in w[0..k].iter_mut().enumerate() {
        let mag = (1.0 + (i % 37) as f32) * 2f32.powi(-20);
        *v = if i % 2 == 0 { mag } else { -mag };
    }
    // Row 1: large magnitude, close to f16's ceiling (65504.0) - close enough
    // that a broken exponent-field/saturation path in the DECODE would
    // visibly corrupt this row's output. `model::half::f32_to_f16`'s own
    // overflow-to-infinity behaviour is covered separately by
    // `crates/model/src/half.rs`'s unit tests; this row exercises the DEVICE
    // decode of a large-but-in-range value, not the host encoder's overflow
    // branch.
    for (i, v) in w[k..2 * k].iter_mut().enumerate() {
        let mag = 60000.0 - (i % 500) as f32;
        *v = if i % 2 == 0 { mag } else { -mag };
    }
}

/// Per-output-element tolerance - see this file's module doc comment for the
/// derivation. `f64` accumulation so the tolerance's own arithmetic is not
/// itself the source of round-off being measured against.
fn f16_tol(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut tol = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut abs_sum = 0f64;
            for i in 0..k {
                abs_sum += (x[r * k + i] as f64 * w[j * k + i] as f64).abs();
            }
            tol[r * n + j] = (abs_sum * 2f64.powi(-11)) as f32 + 1e-4;
        }
    }
    tol
}

/// Build a `Weight::F16`, dispatch `Ops::matmul`, and check against the f32
/// host reference within [`f16_tol`]. Consumes `gpu` (`Ops::new` takes
/// ownership) so the caller passes a freshly-built one per shape/backend.
/// `w_h` already carries [`inject_f16_edge_case_rows`]'s overrides - both the
/// device dispatch and the host reference/tolerance use the SAME weight
/// matrix.
fn check_f16_matmul(gpu: Gpu, x_h: &[f32], w_h: &[f32], m: usize, n: usize, k: usize, label: &str) {
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let x = g.storage_init("x", x_h);
    let weight = Weight::upload(&ops, w_h, n, k, Dtype::F16);
    assert_eq!(
        weight.dtype(),
        Dtype::F16,
        "{label} m={m} n={n} k={k}: this device must report f16_storage for Weight::upload to land \
         on the F16 tier (see backend-wgpu/backend-cpu's NumericSupport construction)"
    );

    let mut steps = Vec::new();
    let act = ops.act(&mut steps, &x, 0, m as u32, k as u32);
    let out = g.storage((m * n) as u64);
    ops.matmul(&mut steps, &weight, &act, &out, 0);
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    let want = host_matmul(x_h, w_h, m, k, n);
    let tol = f16_tol(x_h, w_h, m, k, n);
    assert_eq!(got.len(), want.len());
    let mut worst: f32 = 0.0;
    for i in 0..got.len() {
        assert!(got[i].is_finite(), "{label} m={m} n={n} k={k}: elem {i} is non-finite: {}", got[i]);
        let err = (got[i] - want[i]).abs();
        worst = worst.max(err / tol[i].max(1e-9));
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
    &[(8, 128, 128, "workgroup_per_output/matmul_gemv#w=f16"), (64, 128, 128, "register_tiled/matmul_reg3#w=f16"), (64, 64, 128, "reference/matmul#w=f16")];

/// Every shape's `(x, w)` pair, `w` carrying [`inject_f16_edge_case_rows`]'s
/// subnormal/near-ceiling override on rows 0/1 - generated once per shape so
/// the CPU and GPU runs see byte-identical inputs.
fn shape_inputs(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let mut w_h = rng.vec_scaled(n * k, 1.0);
    inject_f16_edge_case_rows(&mut w_h, n, k);
    (x_h, w_h)
}

/// **A real finding, checked rather than just noted in prose (same shape as
/// B4's own CPU-reachability finding for bf16).** `matmul_reg3#w=f16` has the
/// same 3-`workgroupBarrier()` structure as `matmul_reg3#w=bf16` - the CPU JIT
/// cannot compile a multi-barrier workgroup kernel, and `backend-cpu`'s own
/// `DeviceCaps.workgroup_reductions` is `false` for exactly that reason.
/// `RegisterTiled` REQUIRES `workgroup_reductions`
/// (`backend_api::select::KernelVariant::requires`), so `select::candidates`
/// filters it out on CPU caps and falls back to `Reference` -
/// `matmul_reg3#w=f16` is therefore NEVER actually dispatched on the CPU
/// backend, same as its bf16 sibling.
#[test]
fn register_tiled_f16_is_not_reachable_on_the_cpu_backend() {
    use gpu_core::select::{self, KernelSelector, Op, OpShape};
    let gpu = Gpu::new_cpu(&kernel_list());
    let caps = gpu.caps();
    assert!(!caps.workgroup_reductions, "backend-cpu's caps must still report workgroup_reductions=false");
    let shape = OpShape { m: 64, n: 128, k: 128, dtype: Dtype::F16 };
    let variant = select::DefaultSelector.select(Op::MatMul, shape, &caps);
    assert_eq!(
        variant,
        select::KernelVariant::Reference,
        "CPU caps must fall back to Reference for a shape that would pick RegisterTiled on a real GPU"
    );
}

/// The `register_tiled` shape's TAG names the kernel a real GPU dispatches
/// for it; on CPU it silently runs through `Reference` instead (see
/// [`register_tiled_f16_is_not_reachable_on_the_cpu_backend`]) - still a
/// valid, useful check (CPU output must still match the f32 host reference,
/// including at the injected subnormal/near-ceiling rows), just not proof
/// that `matmul_reg3#w=f16` itself executed correctly. Only the GPU run
/// (`f16_matmul_matches_f32_reference_on_gpu`) proves that.
#[test]
fn f16_matmul_matches_f32_reference_on_cpu() {
    for &(m, n, k, tag) in SHAPES {
        let (x_h, w_h) = shape_inputs(m, n, k, 0xF16_0000 ^ (m as u64) << 16 ^ n as u64);
        let list = kernel_list();
        let gpu = Gpu::new_cpu(&list);
        check_f16_matmul(gpu, &x_h, &w_h, m, n, k, &format!("cpu/{tag}"));
    }
}

/// Real GPU execution - gated by `MOE_SKIP_GPU_TESTS` like every other
/// dual-backend test in this tree. Prints which branch actually ran (real
/// wgpu execution vs a clean skip) so a report can state which happened
/// rather than assume.
#[test]
fn f16_matmul_matches_f32_reference_on_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("f16_matmul_matches_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("f16_matmul_matches_f32_reference_on_gpu: running on a real wgpu device");
    for &(m, n, k, tag) in SHAPES {
        let (x_h, w_h) = shape_inputs(m, n, k, 0xF16_0000 ^ (m as u64) << 16 ^ n as u64);
        let list = kernel_list();
        let gpu = Gpu::new_wgpu(&list);
        check_f16_matmul(gpu, &x_h, &w_h, m, n, k, &format!("gpu/{tag}"));
    }
}
