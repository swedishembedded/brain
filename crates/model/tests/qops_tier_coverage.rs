// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fast inference over quantized checkpoints for its clients. If your team needs
// expertise in decode and prefill GEMMs for every weight storage format then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: `Ops::matmul_rows` is ONE entry for every weight storage tier. For every `Weight` variant and format
//! (dense f32/bf16/f16, int8, W4A8, NF4, FP8 E4M3/E5M2, K-quants Q4_K/Q5_K/Q6_K, NVFP4, MXFP4, MXFP8 E4M3/E5M2) the
//! product of `m` activation rows (decode batch `m = 1` and `8`, a ragged prefill `m = 33`, a full tile `m = 256`)
//! equals the f64 product of the values the stored bytes decode to, whichever kernel the device chose for that row
//! count, and a row window that does not start at 0 selects exactly those weight rows.
//!
//! The oracle is exact decoding of the stored bytes (independent host code, `tier_kit`) followed by f64 arithmetic;
//! what remains is the arithmetic of the device path (f32 accumulation, bf16/TF32 tensor cores, int8 activation
//! rounding for the integer tiers), so the tolerance is relative to the root of the sum of squared terms.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::ops::{training_kernel_list, Ops, Weight};

mod tier_kit;
use tier_kit::build_case;

const TIERS: [&str; 15] =
    ["f32", "bf16", "f16", "i8", "q4", "nf4", "fp8_e4m3", "fp8_e5m2", "q4k", "q5k", "q6k", "nvfp4", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2"];
const ROWS: [usize; 4] = [1, 8, 33, 256];
/// Ragged output width (not a multiple of any kernel's row round); `K` holds a K-quant super-block and an NVFP4 64-block.
const N: usize = 37;
const K: usize = 256;
/// bf16 tensor-core products and int8 activation rounding stay well below this fraction of the term magnitude.
const REL_TOL: f64 = 3e-2;

/// `y[mi, c] = sum_i x[mi, i] * wd[first + c, i]` in f64, with the root of the sum of squared terms beside each.
fn host_product(x: &[f32], m: usize, wd: &[f32], first: usize, count: usize) -> Vec<(f64, f64)> {
    let mut y = vec![(0f64, 0f64); m * count];
    for mi in 0..m {
        for c in 0..count {
            let terms: Vec<f64> = (0..K).map(|i| x[mi * K + i] as f64 * wd[(first + c) * K + i] as f64).collect();
            y[mi * count + c] = (terms.iter().sum(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
        }
    }
    y
}

fn check(ops: &Ops, w: &Weight, wd: &[f32], label: &str, m: usize, (first, count): (usize, usize), rng: &mut Lcg) {
    let g = ops.gpu();
    let x = rng.vec_scaled(m * K, 1.0);
    let xb = g.storage_init("x", &x);
    let y = g.storage((m * count) as u64);
    let mut steps = Vec::new();
    // The K-quant prepass is the superset activation: the integer tiers read its int8 rows, the float tiers its f32 rows.
    let act = ops.act_kq(&mut steps, &xb, 0, m as u32, K as u32);
    ops.matmul_rows(&mut steps, w, (first as u32, count as u32), &act, &y, 0);
    g.submit(&[], &steps);
    let got = g.read(&y, m * count);
    for (i, (a, (b, scale))) in got.iter().zip(host_product(&x, m, wd, first, count)).enumerate() {
        assert!(
            (*a as f64 - b).abs() <= REL_TOL * scale + 1e-6,
            "{label} m={m} rows {first}+{count}: element {i} is {a}, host product {b} (term scale {scale})"
        );
    }
}

#[test]
fn every_weight_tier_multiplies_any_row_count_through_matmul_rows() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if !ops.gpu().caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the tiled and GEMV kernels need workgroup barriers (not the CPU JIT)");
    }
    let mut rng = Lcg::new(data::rng::random_seed());
    let mut failures = Vec::new();
    for label in TIERS {
        let case = build_case(&ops, label, N, K, &mut rng);
        for m in ROWS {
            let windows: &[(usize, usize)] = if m == 33 { &[(0, N), (3, N - 5)] } else { &[(0, N)] };
            for &window in windows {
                // Collect every failing (tier, m) so one run reports the whole table.
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&ops, &case.weight, &case.decoded, label, m, window, &mut rng)));
                if let Err(e) = res {
                    failures.push(e.downcast_ref::<String>().cloned().unwrap_or_else(|| format!("{label} m={m}: panicked")));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} tier/row-count cases disagree with the oracle:\n{}", failures.len(), failures.join("\n"));
}
