// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.3's own no-regression gate, stronger than
//! `ops_facade_parity.rs`'s output-bit-identity check: for a representative
//! shape/tier sweep, the exact [`backend_api::StepMeta`] (kernel index,
//! params, thread count) `Ops::matmul` records THROUGH `ProviderRegistry::
//! dispatch` (the post-M8.3 path) must match [`Ops::matmul_kernel`]'s own
//! report (a pre-existing, UNCHANGED-by-this-milestone diagnostic method,
//! see its doc comment, that resolves the exact same `self.selector.select`
//! plus kernel-name-bind decision `Ops::matmul` itself now delegates to the
//! provider) and the thread-count formula that decision implies.
//!
//! **Why the oracle is `matmul_kernel`, not `model::dispatch::
//! {mm_rows_off,mm8_rows_off,mm4_rows_off}`.** Those helpers (`ops_facade_
//! parity.rs`'s own oracle) resolve a kernel via `model::block::gemm_variant`,
//! a SEPARATE, independently-tuned selection heuristic (`candidates(..)
//! .next()` against a synthetic "fast tier" caps) that is not guaranteed to
//! pick the same `KernelVariant` `Ops::matmul`'s real `DefaultSelector`
//! policy does at every shape (confirmed while building this test: they
//! disagree at `m=64,n=64,k=128` on this box's real device, yet
//! `ops_facade_parity.rs` still passes there, because a register-tiled and a
//! reference F32 GEMM compute the identical result whichever one runs). That
//! divergence predates M8.3 and is not this milestone's to fix; comparing
//! against it here would either force this test to reproduce that pre-
//! existing bug or wrongly flag it as a NEW one. `matmul_kernel` is the
//! correct oracle because it consults the exact same selector/bind path a
//! caller of `Ops::matmul` itself goes through, untouched by this move.

use gpu_core::select::Dtype;
use gpu_core::{Gpu, Step};
use model::ops::{Ops, Weight};

fn kernel_list() -> &'static [(&'static str, &'static str)] {
    model::ops::kernel_list()
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// The thread-count formula `gpu_core::provider::wgsl::WgslProvider::
/// threads` implements (itself `model::ops::Ops::threads`, moved verbatim at
/// M8.3) - restated here, independently, as a frozen expected-value fixture
/// keyed by kernel NAME (which `Self::bind`'s table makes a 1:1 stand-in for
/// `KernelVariant` at a fixed dtype/group). Any drift between the real
/// dispatch and this fixture is a real thread-count regression, not a
/// re-derivation of the same code under test.
fn expected_threads(kernel_name: &str, m: u32, n: u32) -> u32 {
    let tile = || m.div_ceil(128) * n.div_ceil(128) * 256;
    match kernel_name {
        "matmul" | "matmul#w=bf16" | "matmul#w=f16" => m * n,
        "matmul_gemv" | "matmul_gemv#w=bf16" | "matmul_gemv#w=f16" | "matmul_i8_gemv" | "matmul_i8_gemv#WPG=4"
        | "matmul_q4_gemv" | "matmul_kq_gemv#CODE_BITS=4" | "matmul_kq_gemv#CODE_BITS=8" => n * 64,
        "matmul_reg2" | "matmul_reg3#w=bf16" | "matmul_reg3#w=f16" | "matmul_i8_dyn" | "matmul_i8_dyn#QPG=1"
        | "matmul_q4_dyn_reg" | "matmul_kq_dyn#CODE_BITS=4" | "matmul_kq_dyn#CODE_BITS=8" => tile(),
        other => panic!("expected_threads: this test's frozen name table has no entry for {other:?}"),
    }
}

/// Asserts `got`'s [`backend_api::StepMeta`] matches what dispatching `w` at
/// `(m, n)` through [`Ops::matmul_kernel`]'s own reported kernel implies.
fn assert_matches_matmul_kernel(g: &Gpu, ops: &Ops, w: &Weight, m: u32, n: u32, expected_params: &[u32], got: &Step, ctx: &str) {
    let name = ops.matmul_kernel(w, m);
    let meta = got.meta().expect("Ops::matmul must record StepMeta");
    assert_eq!(meta.kernel, idx(g, name), "{ctx}: kernel index differs from Ops::matmul_kernel's own report ({name:?})");
    assert_eq!(meta.threads, expected_threads(name, m, n), "{ctx}: thread count differs from the frozen formula for {name:?}");
    assert_eq!(meta.params, Some(expected_params.to_vec()), "{ctx}: uniform params differ");
}

fn check_step_identity(m: usize, n: usize, k: usize, dt: Dtype) {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let x_h = vec![0.25f32; m * k];
    let w_h = vec![0.5f32; n * k];
    let x = g.storage_init("x", &x_h);
    let weight = Weight::upload(&ops, &w_h, n, k, dt);
    if weight.dtype() != dt {
        // This device doesn't support the tier (no int8_dot) - nothing to
        // check, exactly like `ops_facade_parity.rs`'s own tiers skip this
        // way (see its `assert_eq!(weight.dtype(), ..)` calls).
        return;
    }

    let mut steps = Vec::new();
    let act = ops.act(&mut steps, &x, 0, m as u32, k as u32);
    let out = g.storage((m * n) as u64);
    ops.matmul(&mut steps, &weight, &act, &out, 0);
    // `Ops::act`'s own quant prepass dispatches (unused by an F32/BF16/F16
    // weight, always built anyway) land in `steps` too - the matmul
    // dispatch itself is always the LAST step.
    let got = steps.last().expect("Ops::matmul must push exactly one Step");

    let param_k = match dt {
        Dtype::I8 => (k as u32) / (Dtype::I8.per_word()),
        _ => k as u32,
    };
    let params = [m as u32, param_k, n as u32];
    assert_matches_matmul_kernel(g, &ops, &weight, m as u32, n as u32, &params, got, &format!("{dt:?} m={m} n={n} k={k}"));
}

/// `m ∈ {1, 8, 64, 512}` - the same decode/GEMV-crossover/register-tiled
/// sweep `ops_facade_parity.rs`'s own output-bit-identity test uses, across
/// every dtype `Ops::matmul` dispatches through the provider seam
/// (`F32`/`BF16`/`F16`/`I8`/`Q4`).
#[test]
fn matmul_dispatches_the_step_matmul_kernel_reports() {
    let (n, k) = (64usize, 128usize);
    for &m in &[1usize, 8, 64, 512] {
        for dt in [Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::Q4] {
            check_step_identity(m, n, k, dt);
        }
    }
}

/// A non-zero row offset (`Ops::act`'s `xr0`) - the offset arithmetic that
/// decides `StepMeta::params`' `m` field and the operand ranges must still
/// resolve to the SAME kernel/thread-count `matmul_kernel` reports; the
/// offset itself is not observable in `StepMeta` (it lives in the bound
/// buffer's sub-range, which `StepMeta` does not carry - `ops_facade_
/// parity.rs`'s own offset test is the READ-BACK-level gate for that half).
#[test]
fn matmul_row_offset_still_matches_matmul_kernel() {
    let (n, k) = (64usize, 128usize);
    let (xr0, m) = (64u32, 8usize);
    let total_rows = xr0 as usize + m;

    for dt in [Dtype::I8, Dtype::Q4] {
        let gpu = gpu_core::testgpu::dev(kernel_list());
        let ops = Ops::new(gpu).expect("Ops::new");
        let g = ops.gpu();

        let x_h = vec![0.25f32; total_rows * k];
        let w_h = vec![0.5f32; n * k];
        let x_full = g.storage_init("x_full", &x_h);
        let weight = Weight::upload(&ops, &w_h, n, k, dt);
        if weight.dtype() != dt {
            continue;
        }

        let mut steps = Vec::new();
        let act = ops.act(&mut steps, &x_full, xr0, m as u32, k as u32);
        let out = g.storage((m * n) as u64);
        ops.matmul(&mut steps, &weight, &act, &out, 0);
        let got = steps.last().expect("Ops::matmul must push exactly one Step");

        let param_k = match dt {
            Dtype::I8 => (k as u32) / (Dtype::I8.per_word()),
            _ => k as u32,
        };
        let params = [m as u32, param_k, n as u32];
        assert_matches_matmul_kernel(g, &ops, &weight, m as u32, n as u32, &params, got, &format!("{dt:?} row offset xr0={xr0} m={m}"));
    }
}
