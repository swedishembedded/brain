// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `moe_linear_gated_gemv.wgsl` against `moe_linear_gated.wgsl`, the kernel it
//! is the skinny-M tier of.
//!
//! `moe::expert_fwd_tiered` swaps the two by device capability and row count,
//! so they must compute the same function - but NOT bit-identically: the GEMV
//! splits the K reduction across 64 lanes and folds their partials, which
//! reassociates a sum. That is precisely why it is a visible call-site seam
//! rather than a `gpu_core::upgrade` row (whose own bar demands bit-identity),
//! and why adopting it owes this gate.
//!
//! Three things are checked, and the third is the one a tolerance alone would
//! miss:
//!
//! 1. **Routed rows agree** with the element-per-thread kernel, to the fp32
//!    reduction-order floor - measured on an absolute-plus-relative bound
//!    rather than a relative one, because both kernels sum the same `k`
//!    O(1) products in different orders: the DIFFERENCE is bounded by the
//!    terms, not by the sum, and a sum that nearly cancels has an enormous
//!    relative error at an entirely ordinary absolute one.
//! 2. **Non-routed rows are EXACTLY zero**, bit for bit. `moe_linear_gated`'s
//!    contract is that a row whose gate weight for this expert is `<= 0`
//!    writes `0` and is never reduced; `scale_add` then reads that slot
//!    unconditionally. A GEMV that merely wrote something small there would
//!    pass a tolerance and quietly poison the accumulator with a value scaled
//!    by a zero gate - which is zero, until the day the combine changes.
//! 3. **Every legal row count**, 1 through the decode regime's own ceiling of
//!    32, because the kernel indexes `partial[m * 64 + t]` out of a 2048-float
//!    array and `m = 32` is the exact edge of it.
//!
//! Runs on whatever `BRAIN_DEVICE` selects. On `backend-cpu` the two kernels
//! are compared through its JIT (both are `@cpu yes`), which is not a
//! placement this seam ever selects but is a free extra check that the WGSL
//! itself is right.

use data::rng::Lcg;
use gpu_core::Gpu;

const PIPES: &[(&str, &str)] = &[
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("moe_linear_gated_gemv", kernels::MOE_LINEAR_GATED_GEMV),
];

/// `backend_api::select::DECODE_REGIME_MAX_ROWS` - the ceiling the GEMV's
/// `partial` array is sized for and the one `block::gemv_tier` selects under.
const MAX_ROWS: u32 = 32;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// A dense `[m, n_experts]` gate in `router_gate.wgsl`'s own shape: exactly
/// `top_k` positive entries per row, the rest exactly zero.
fn gate_rows(rng: &mut Lcg, m: u32, e: u32, top_k: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (m * e) as usize];
    for r in 0..m {
        // A deterministic, row-dependent stride so different rows route to
        // different experts - a gate where every row picks the same experts
        // would leave the non-routed path exercised on whole columns only.
        for j in 0..top_k {
            let col = ((r * 7 + j * 3 + 1) % e) as usize;
            out[(r * e) as usize + col] = 0.25 + rng.unit() * 0.5;
        }
    }
    out
}

#[test]
fn the_gemv_tier_matches_the_element_per_thread_kernel() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (slow, fast) = (idx(&g, "moe_linear_gated"), idx(&g, "moe_linear_gated_gemv"));
    let mut rng = Lcg::new(20260908);

    // k deliberately NOT a multiple of the 64-wide k-stride (the GEMV's inner
    // loop is `k = t; k += 64`, so a k that divides evenly would hide a
    // ragged-tail bug), and n both above and below 64 so a column grid that
    // is not a whole number of workgroups is covered.
    let (e, top_k) = (8u32, 2u32);
    let mut worst = 0.0f32;
    let mut checked = 0usize;
    for &(k, n) in &[(70u32, 96u32), (128, 40), (259, 129)] {
        for &m in &[1u32, 2, 7, MAX_ROWS] {
            let x = g.storage_init("x", &rng.vec_scaled((m * k) as usize, 1.0));
            let w = g.storage_init("w", &rng.vec_scaled((n * k) as usize, 0.5));
            let host_gate = gate_rows(&mut rng, m, e, top_k);
            let gate = g.storage_init("gate", &host_gate);

            for e_idx in 0..e {
                let want_buf = g.storage((m * n) as u64);
                let got_buf = g.storage((m * n) as u64);
                g.submit(
                    &[],
                    &[
                        g.step(slow, &[&x, &w, &gate, &want_buf], &[m, k, n, e, e_idx], m * n),
                        g.step(fast, &[&x, &w, &gate, &got_buf], &[m, k, n, e, e_idx], n * 64),
                    ],
                );
                let want = g.read(&want_buf, (m * n) as usize);
                let got = g.read(&got_buf, (m * n) as usize);

                for r in 0..m as usize {
                    let routed = host_gate[r * e as usize + e_idx as usize] > 0.0;
                    for c in 0..n as usize {
                        let (a, b) = (want[r * n as usize + c], got[r * n as usize + c]);
                        if routed {
                            // Combined absolute + relative, not relative alone.
                            // Both kernels sum `k` products of O(1) operands in
                            // different orders, so the DIFFERENCE is bounded by
                            // the terms' magnitude, not by the sum's - and a
                            // sum that nearly cancels (measured: -0.0089 from
                            // 70 terms) has a huge relative error at a
                            // perfectly ordinary absolute one. A relative-only
                            // bound would be a test that fails on well-behaved
                            // inputs and says nothing about badly-behaved ones.
                            let abs = (a - b).abs();
                            worst = worst.max(abs);
                            assert!(
                                abs <= 1e-5 + 1e-5 * a.abs().max(b.abs()),
                                "k={k} n={n} m={m} e_idx={e_idx} row {r} col {c}: slow {a} vs gemv {b} (abs {abs:e})"
                            );
                        } else {
                            assert_eq!(
                                b.to_bits(),
                                0f32.to_bits(),
                                "k={k} n={n} m={m} e_idx={e_idx} row {r} col {c}: a non-routed row must be EXACTLY zero, got {b}"
                            );
                            assert_eq!(a.to_bits(), 0f32.to_bits(), "the element-per-thread kernel's own contract");
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    println!("moe_linear_gated_gemv: {checked} elements compared, worst routed |slow - gemv| {worst:e}");
    assert!(checked > 0);
}
