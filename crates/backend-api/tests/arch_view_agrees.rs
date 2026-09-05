// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.1's backward-compatibility gate: [`ArchDesc::numeric_view`] must
//! reproduce EXACTLY what each real backend's pre-M8.1 `query_caps`/`caps`
//! set, for every [`NumericSupport`] field this milestone does not
//! deliberately change.
//!
//! Each case below builds the SAME [`ArchDesc`] the corresponding backend's
//! new (post-M8.1) `query_caps` builds for a real, plausible device-query
//! outcome (`backend-wgpu`'s `query_caps`, `backend-cpu`'s `caps`,
//! `backend-vulkan`'s `query_caps` - read directly from source before
//! writing this table, not guessed), then asserts `numeric_view()` against
//! the literal the OLD (pre-M8.1) code would have produced for that same
//! query outcome - transcribed by hand from each backend's real formula:
//!
//! - `backend-wgpu` (unconditional): `NumericSupport { int8_dot: true,
//!   bf16_storage: true, f16_storage: true, ..BASELINE }` - every wgpu
//!   target, no branching on device state at all.
//! - `backend-cpu` (unconditional): `NumericSupport { int8_dot: false,
//!   f16_storage: true, bf16_storage: true, ..BASELINE }`.
//! - `backend-vulkan`: `NumericSupport { int8_dot: ctx.prec.dp4a,
//!   coop_matrix: ctx.caps.feature_supported && !ctx.caps.shapes.is_empty(),
//!   ..BASELINE }` - `f16`/`bf16`/`f16_storage`/`bf16_storage` stayed `false`
//!   UNCONDITIONALLY (nothing in the old code ever touched them, regardless
//!   of `ctx.prec.f16`).
//!
//! `int8_dot`/`coop_matrix` are this milestone's two deliberate fixes (see
//! `crate::arch`'s module doc): every case below still asserts them because,
//! on every query outcome this table actually covers, the new formula
//! reproduces the SAME bit the old one did (the fix only changes behaviour
//! on a device the old formula got wrong - vulkan I8 with a naga polyfill
//! but no DP4A hardware, which the old code reported `false` for despite the
//! kernel executing; case `vulkan_no_dp4a_still_executes_i8` below pins that
//! exact, intentional divergence separately, not folded into the "agrees"
//! table).

use backend_api::arch::{ArchDesc, MatShape, MatrixEngine, MatrixKind, TierLevel, TierSupport};
use backend_api::{DType, NumericSupport};

fn storage(level: TierLevel) -> TierSupport {
    TierSupport { level, ..Default::default() }
}

fn wgpu_arch() -> ArchDesc {
    let mut a = ArchDesc::default();
    for dt in [DType::I8, DType::Q4, DType::Q4K, DType::Q8K] {
        a.set_tier(dt, storage(TierLevel::Emulated));
    }
    a.set_tier(DType::F16, storage(TierLevel::Storage));
    a.set_tier(DType::BF16, storage(TierLevel::Storage));
    a
}

fn cpu_arch() -> ArchDesc {
    let mut a = ArchDesc::default();
    a.set_tier(DType::F16, storage(TierLevel::Storage));
    a.set_tier(DType::BF16, storage(TierLevel::Storage));
    a
}

fn vulkan_arch(dp4a: bool, f16: bool, matrix_shapes: usize) -> ArchDesc {
    let mut a = ArchDesc::default();
    a.set_tier(DType::I8, storage(if dp4a { TierLevel::Native } else { TierLevel::Emulated }));
    a.set_tier(DType::F16, storage(if f16 { TierLevel::Native } else { TierLevel::Absent }));
    if matrix_shapes > 0 {
        let shapes = (0..matrix_shapes)
            .map(|_| MatShape { m: 16, n: 16, k: 16, a: DType::F16, b: DType::F16, accum: DType::F32, scope_width: None })
            .collect();
        a.matrix = Some(MatrixEngine { kind: MatrixKind::CoopMatrix, shapes });
    }
    a
}

fn old_wgpu() -> NumericSupport {
    NumericSupport { int8_dot: true, bf16_storage: true, f16_storage: true, ..NumericSupport::BASELINE }
}

fn old_cpu() -> NumericSupport {
    NumericSupport { int8_dot: false, f16_storage: true, bf16_storage: true, ..NumericSupport::BASELINE }
}

fn old_vulkan(dp4a: bool, coop_matrix: bool) -> NumericSupport {
    NumericSupport { int8_dot: dp4a, coop_matrix, ..NumericSupport::BASELINE }
}

/// The table: `(case name, ArchDesc built the M8.1 way, NumericSupport the
/// pre-M8.1 code would have produced for the same query outcome)`.
fn cases() -> Vec<(&'static str, ArchDesc, NumericSupport)> {
    vec![
        ("wgpu: the only real query outcome this backend has", wgpu_arch(), old_wgpu()),
        ("cpu: the only real query outcome this backend has", cpu_arch(), old_cpu()),
        // NOTE: a `dp4a: false` case is deliberately NOT in this table - that
        // is exactly the one query outcome M8.1 changes the derived view
        // for (the packed-int8 kernels execute via naga's polyfill even
        // without DP4A hardware, so `numeric_view().int8_dot` is now `true`
        // where the old formula said `false`). See
        // `vulkan_no_dp4a_still_executes_i8_unlike_the_old_formula` below,
        // which pins that exact, intentional divergence on its own. Every
        // remaining row therefore fixes `dp4a: true` and sweeps `f16` and
        // the coop-matrix shape count instead - the two axes that stayed
        // faithful to the old formula.
        ("vulkan: dp4a, no f16, no coop matrix (P40-shaped)", vulkan_arch(true, false, 0), old_vulkan(true, false)),
        ("vulkan: dp4a, f16, no coop matrix", vulkan_arch(true, true, 0), old_vulkan(true, false)),
        ("vulkan: dp4a, f16, one coop-matrix shape (Turing-shaped)", vulkan_arch(true, true, 1), old_vulkan(true, true)),
        ("vulkan: dp4a, no f16, several coop-matrix shapes", vulkan_arch(true, false, 3), old_vulkan(true, true)),
    ]
}

#[test]
fn numeric_view_agrees_with_every_backends_pre_m8_1_output() {
    for (name, arch, expected) in cases() {
        let actual = arch.numeric_view();
        assert_eq!(actual, expected, "case {name:?}: numeric_view() diverged from the pre-M8.1 formula");
    }
}

/// Pinned separately from the table above: this is the ONE query outcome
/// where the new formula and the old one legitimately disagree, and it is
/// the entire point of M8.1 - the old code reported `int8_dot: false` for a
/// Vulkan device with no DP4A hardware even though the packed-int8 kernels
/// still execute there (naga's polyfill), because the old bool conflated
/// "executes" with "native hardware". The new formula reports `true`
/// (`executes(I8)` at `Emulated`), matching what `NumericSupport::int8_dot`'s
/// OWN doc comment always said this field meant ("execute[s]... regardless"
/// of whether the driver has hardware DP4A).
#[test]
fn vulkan_no_dp4a_still_executes_i8_unlike_the_old_formula() {
    let arch = vulkan_arch(false, false, 0);
    assert_eq!(arch.tier(DType::I8).level, TierLevel::Emulated);
    assert!(arch.numeric_view().int8_dot, "the new formula must report the kernel as executing");
    assert!(!old_vulkan(false, false).int8_dot, "the old formula reported false here -- this is the fix");
}
