// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cross-provider parity - proves a candidate [`super::OperatorProvider`]
//! produces the SAME result the WGSL reference provider does, on the same
//! device and the same seeded inputs. The oracle is always WGSL, never the
//! provider under test - matching `AGENTS.md`'s own framing of the
//! reference as "every other one is gated against [it] for correctness".
//!
//! This milestone (M8.3) builds the machinery and proves it against the
//! reference provider itself (the only provider that exists this wave) - 
//! trivially bit-identical, since both sides run the identical code path,
//! but that is exactly what proves the harness's OWN plumbing (buffer
//! seeding, dispatch, readback, comparison) is sound before a real second
//! provider ever registers a case here. A wave-2 provider gets this whole
//! table for free by naming which [`select::Op`]s it claims via
//! [`cases_for`] and calling [`assert_provider_parity`] once per case in its
//! own test.

use std::sync::Arc;

use backend_api::select;
use backend_api::DType;

use super::wgsl::WgslProvider;
use super::{LowerCtx, Operand, OpRequest, OperatorProvider, Pass, Role};
use crate::Gpu;

/// How closely two providers' outputs must agree.
#[derive(Clone, Copy, Debug)]
pub enum Tolerance {
    /// Exact - the only honest bar for two implementations of the SAME
    /// associative-order arithmetic (e.g. two dispatch geometries of the
    /// identical reference formula).
    BitIdentical,
    /// `|a-b| <= atol + rtol*|a|` per element, plus a cosine-similarity
    /// floor across the whole output - for a genuinely different arithmetic
    /// path (a real second provider, wave 2) where reassociated summation
    /// order makes bit-identity the wrong bar.
    Numeric { atol: f32, rtol: f32, cosine_min: f64 },
}

/// One parity fixture: an op, the shape that selects its implementation,
/// which pass, a seed for reproducible random inputs, and the tolerance two
/// providers' outputs must agree within.
#[derive(Clone, Copy, Debug)]
pub struct ParityCase {
    pub op: select::Op,
    pub shape: select::OpShape,
    pub pass: Pass,
    pub seed: u64,
    pub tol: Tolerance,
}

/// splitmix64 - a tiny deterministic PRNG so these fixtures need no external
/// `rand` dependency and reproduce identically across OS/arch (integer-only,
/// no platform RNG).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` values in `[-1.0, 1.0)`, deterministic in `seed`.
fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
    (0..n)
        .map(|_| {
            let bits = splitmix64(&mut state);
            ((bits >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Runs `case` through both the reference WGSL provider and `p`, on `gpu`,
/// over the same seeded inputs, and asserts they agree within `case.tol`.
pub fn assert_provider_parity(gpu: &Gpu, p: &dyn OperatorProvider, case: &ParityCase) {
    match case.op {
        select::Op::MatMul => assert_matmul_parity(gpu, p, case),
        other => panic!("parity::assert_provider_parity: no case builder wired for {other:?} yet"),
    }
}

fn assert_matmul_parity(gpu: &Gpu, p: &dyn OperatorProvider, case: &ParityCase) {
    let (m, n, k) = (case.shape.m, case.shape.n, case.shape.k);
    // A deterministic non-zero row offset on roughly a third of the fixed
    // case table (see `MATMUL_CASES`) - the "non-zero row offset" regime the
    // brief's case list names, without widening `ParityCase` itself. `64`
    // rows (not 16): a storage-buffer binding offset must be a multiple of
    // the device's `min_storage_buffer_offset_alignment` (256 bytes / 64
    // f32 words), and `64 * k * 4` is a multiple of 256 for EVERY `k`
    // (`64*4=256` divides it regardless of `k`'s own value) - the same
    // 64-row/256B rule `quant_rows_steps`'s own alignment contract already
    // documents, generalised here to F32 offsets too.
    let xr0 = ((case.seed % 3) as u32) * 64;
    let total_rows = xr0 + m;

    let x = gpu.storage_init("parity_x", &seeded_f32(case.seed, (total_rows * k) as usize));
    let w = gpu.storage_init("parity_w", &seeded_f32(case.seed ^ 1, (n * k) as usize));
    let y_ref = gpu.storage((m * n) as u64);
    let y_got = gpu.storage((m * n) as u64);

    let reference_selector: Arc<dyn select::KernelSelector> =
        Arc::new(select::CachedSelector::new(select::AlwaysReference));
    let reference = WgslProvider::new(reference_selector);

    // This harness's own `Op::MatMul` case table is F32-only (see
    // `MATMUL_CASES`), so the reference selector it drives (`AlwaysReference`)
    // only ever asks for `KernelVariant::Reference` - the "matmul" kernel a
    // parity-testing caller must already have registered on `gpu`.
    let bind = |v: select::KernelVariant| -> (usize, &'static str) {
        match v {
            select::KernelVariant::Reference => {
                (gpu.kernel_index("matmul").expect("parity harness: 'matmul' must be registered on gpu"), "matmul")
            }
            other => panic!(
                "parity::assert_matmul_parity: this harness's F32 case table only ever selects \
                 KernelVariant::Reference -- got {other:?}, which means either the selector under \
                 test is not F32-scoped or a non-F32 case slipped into MATMUL_CASES"
            ),
        }
    };

    let xo = (xr0 as u64 * k as u64, m as u64 * k as u64);
    let attrs = [m, k, n];
    let caps = gpu.caps();

    let operands_ref = [
        Operand { role: Role::Act, buf: &x, range: xo, dtype: DType::F32 },
        Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &y_ref, range: (0, (m * n) as u64), dtype: DType::F32 },
    ];
    let req_ref =
        OpRequest { op: case.op, shape: case.shape, pass: case.pass, operands: &operands_ref, attrs: &attrs, bind: &bind };
    let mut steps_ref = Vec::new();
    {
        let mut ctx = LowerCtx { gpu, caps: &caps, steps: &mut steps_ref, capture: false };
        reference.lower(&mut ctx, &req_ref).expect("reference provider must lower every ParityCase in MATMUL_CASES");
    }
    gpu.submit(&[], &steps_ref);

    let operands_got = [
        Operand { role: Role::Act, buf: &x, range: xo, dtype: DType::F32 },
        Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &y_got, range: (0, (m * n) as u64), dtype: DType::F32 },
    ];
    let req_got =
        OpRequest { op: case.op, shape: case.shape, pass: case.pass, operands: &operands_got, attrs: &attrs, bind: &bind };
    let mut steps_got = Vec::new();
    {
        let mut ctx = LowerCtx { gpu, caps: &caps, steps: &mut steps_got, capture: false };
        p.lower(&mut ctx, &req_got).expect("provider under test must lower this ParityCase (or decline it before this call)");
    }
    gpu.submit(&[], &steps_got);

    let expect = gpu.read(&y_ref, (m * n) as usize);
    let got = gpu.read(&y_got, (m * n) as usize);
    match case.tol {
        Tolerance::BitIdentical => {
            assert_eq!(
                expect, got,
                "parity mismatch (BitIdentical) for {:?} at shape {:?}, xr0={xr0}",
                case.op, case.shape
            );
        }
        Tolerance::Numeric { atol, rtol, cosine_min } => {
            for (i, (a, b)) in expect.iter().zip(got.iter()).enumerate() {
                let d = (a - b).abs();
                assert!(d <= atol + rtol * a.abs(), "parity mismatch at element {i}: expect {a}, got {b}, diff {d}");
            }
            let dot: f64 = expect.iter().zip(got.iter()).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            let na: f64 = expect.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
            let nb: f64 = got.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
            let cosine = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 1.0 };
            assert!(cosine >= cosine_min, "parity cosine similarity {cosine} below floor {cosine_min}");
        }
    }
}

/// The fixed case table for `op` - decode-shaped (small `m`), the GEMM tile
/// crossover, a large multi-tile shape, a non-tile-multiple shape, and (via
/// `assert_matmul_parity`'s own `xr0` derivation) a non-zero row offset.
/// Empty for any `Op` this harness has no case builder for yet.
pub fn cases_for(op: select::Op) -> &'static [ParityCase] {
    match op {
        select::Op::MatMul => &MATMUL_CASES,
        _ => &[],
    }
}

static MATMUL_CASES: [ParityCase; 4] = [
    // Decode-shaped: one row. seed%3==1 -> xr0=16 (a non-zero row offset).
    ParityCase {
        op: select::Op::MatMul,
        shape: select::OpShape { m: 1, n: 64, k: 64, dtype: select::Dtype::F32 },
        pass: Pass::Forward,
        seed: 1,
        tol: Tolerance::BitIdentical,
    },
    // The 128-row/col GEMM tile crossover. seed%3==2 -> xr0=32.
    ParityCase {
        op: select::Op::MatMul,
        shape: select::OpShape { m: 128, n: 128, k: 128, dtype: select::Dtype::F32 },
        pass: Pass::Forward,
        seed: 2,
        tol: Tolerance::BitIdentical,
    },
    // A large tiled shape, several tiles in each dimension. seed%3==0 -> xr0=0.
    ParityCase {
        op: select::Op::MatMul,
        shape: select::OpShape { m: 300, n: 260, k: 128, dtype: select::Dtype::F32 },
        pass: Pass::Forward,
        seed: 3,
        tol: Tolerance::BitIdentical,
    },
    // A non-tile-multiple shape (neither m nor n divides 128 evenly).
    // seed%3==1 -> xr0=16.
    ParityCase {
        op: select::Op::MatMul,
        shape: select::OpShape { m: 37, n: 53, k: 17, dtype: select::Dtype::F32 },
        pass: Pass::Forward,
        seed: 4,
        tol: Tolerance::BitIdentical,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgpu;

    static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL)];

    /// The harness proves itself: every registered `Op::MatMul` case, run
    /// through the reference provider on BOTH sides, is bit-identical to
    /// itself - the plumbing (seeding, dispatch, readback, compare) is sound
    /// before any real second provider ever calls this.
    #[test]
    fn reference_provider_is_self_parity_clean_on_every_matmul_case() {
        let gpu = testgpu::dev(KERNELS);
        let selector: Arc<dyn select::KernelSelector> = Arc::new(select::CachedSelector::new(select::AlwaysReference));
        let reference = WgslProvider::new(selector);
        for case in cases_for(select::Op::MatMul) {
            assert_provider_parity(&gpu, &reference, case);
        }
    }

    #[test]
    fn cases_for_is_empty_for_an_unimplemented_op() {
        assert!(cases_for(select::Op::RmsNorm).is_empty());
    }
}
