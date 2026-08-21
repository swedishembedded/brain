// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `backend_cpu::host_gemm::blocked_linear` must be **bit-identical** to the
//! naive row-parallel loop it replaces - not close, identical - because the
//! callers it is dropped into (LTX-2.5's adaLN-single table, its patchify and
//! output projections) are gated by parity tests that assert exact agreement
//! with a reference forward.
//!
//! So the comparison here is on BIT PATTERNS (`to_bits`), not on a tolerance:
//! a difference of one ulp in one element out of 130 million is a failure,
//! and `assert_eq!` on f32 values would additionally report two NaNs as
//! unequal, which is the wrong verdict for a byte-identical result.
//!
//! The reference is `host_gemm::naive_linear` - the exact loop that shipped
//! before this change, kept in the module rather than transcribed here, so
//! the gate cannot drift from what callers actually used to run.

use backend_cpu::host_gemm::{blocked_linear, blocked_linear_tiled, default_m_tile, naive_linear, MR};

/// Deterministic operands with a wide exponent range, so cancellation and
/// rounding actually happen: a matrix of small similar numbers would agree
/// under almost any reassociation and prove nothing.
fn operands(rows: usize, in_dim: usize, out_dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 40) as f32 / 16_777_216.0 - 0.5;
        // Spread across ~6 decades so partial sums differ in magnitude and
        // the order of addition is observable.
        u * 10f32.powi(((s >> 20) % 7) as i32 - 3)
    };
    let x = (0..rows * in_dim).map(|_| next()).collect();
    let w = (0..out_dim * in_dim).map(|_| next()).collect();
    let b = (0..out_dim).map(|_| next()).collect();
    (x, w, b)
}

fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    if differing != 0 {
        let (i, (x, y)) = a.iter().zip(b).enumerate().find(|(_, (x, y))| x.to_bits() != y.to_bits()).unwrap();
        panic!("{what}: {differing}/{} elements differ in bit pattern; first at {i}: {x:e} ({:08x}) vs {y:e} ({:08x})", a.len(), x.to_bits(), y.to_bits());
    }
}

/// The core gate. Shapes chosen to cover every structural case the blocking
/// introduces: tiles that do not divide `rows`, row counts shorter than one
/// register block, row counts that leave a tail shorter than one register
/// block, a single row (the `coeff=1` adaLN gate tables), a single output
/// column, and `in_dim` that is not a multiple of anything.
#[test]
fn blocked_linear_is_bit_identical_to_the_naive_loop_it_replaces() {
    let shapes = [
        (1, 7, 5),        // one row: no register block ever fires
        (3, 16, 4),       // fewer rows than MR
        (MR, 32, 9),      // exactly one register block
        (MR + 1, 13, 6),  // one block plus a one-row tail
        (17, 4096, 11),   // the real in_dim, an awkward row count
        (64, 64, 128),    // square-ish, several blocks
        (128, 256, 37),   // out_dim not a multiple of anything
        (35, 4096, 288),  // in_dim/out_dim near the real ratio
    ];
    for (rows, in_dim, out_dim) in shapes {
        let (x, w, b) = operands(rows, in_dim, out_dim, 0x51D2_ACE0 ^ rows as u64);
        let reference = naive_linear(&x, rows, in_dim, &w, Some(&b), out_dim);
        let what = format!("[{rows},{in_dim}]x[{out_dim},{in_dim}]T");

        assert_bit_identical(&blocked_linear(&x, rows, in_dim, &w, Some(&b), out_dim), &reference, &format!("{what} default tile"));

        // Every tile size must give the same answer, including ones that do
        // not divide `rows` and ones smaller than the register block - the
        // tile is a scheduling choice and must never be a numerical one.
        for m_tile in [1, 2, 3, MR - 1, MR, MR + 1, 13, 32, 64, rows.max(1), rows + 5] {
            assert_bit_identical(&blocked_linear_tiled(&x, rows, in_dim, &w, Some(&b), out_dim, m_tile), &reference, &format!("{what} tile {m_tile}"));
        }

        // Bias-free, the shape the FFN linears use.
        let no_bias_ref = naive_linear(&x, rows, in_dim, &w, None, out_dim);
        assert_bit_identical(&blocked_linear(&x, rows, in_dim, &w, None, out_dim), &no_bias_ref, &format!("{what} bias-free"));
    }
}

/// Mutation check: the gate above must actually be able to FAIL, on the
/// exact bug class blocking a GEMM invites - a REASSOCIATED reduction.
///
/// This is what a naive vectorization does: split `k` across lanes, sum each
/// lane, then combine. It is mathematically the same sum and numerically a
/// different one, and it is the mistake that would silently break every
/// parity test downstream. Here it is written out deliberately (two
/// accumulators, even and odd `k`, combined at the end) and the comparison
/// must report a difference - otherwise "bit-identical" is a claim about a
/// test that cannot distinguish anything.
#[test]
fn the_bit_identity_gate_catches_a_reassociated_reduction() {
    let (rows, in_dim, out_dim) = (35, 512, 64);
    let (x, w, b) = operands(rows, in_dim, out_dim, 0x5EED_0001);
    let reference = naive_linear(&x, rows, in_dim, &w, Some(&b), out_dim);

    let mut split = vec![0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let (xr, wr) = (&x[r * in_dim..][..in_dim], &w[o * in_dim..][..in_dim]);
            let (mut e, mut od) = (b[o], 0.0f32);
            for k in 0..in_dim {
                if k % 2 == 0 {
                    e += xr[k] * wr[k];
                } else {
                    od += xr[k] * wr[k];
                }
            }
            split[r * out_dim + o] = e + od;
        }
    }
    let differing = reference.iter().zip(&split).filter(|(a, c)| a.to_bits() != c.to_bits()).count();
    assert!(differing > rows * out_dim / 2, "a split-accumulator reduction must differ in bit pattern from the sequential one on most elements, got {differing}/{}", rows * out_dim);

    // And the blocked kernel, which does NOT reassociate, must be on the
    // reference's side of that line.
    assert_bit_identical(&blocked_linear(&x, rows, in_dim, &w, Some(&b), out_dim), &reference, "blocked vs sequential");
}

/// A degenerate `in_dim = 0` must still produce the bias, matching the naive
/// loop (whose empty inner `zip` leaves `acc` at the bias).
#[test]
fn an_empty_reduction_still_produces_the_bias() {
    let b = vec![1.5f32, -2.0, 3.25];
    let reference = naive_linear(&[], 4, 0, &[], Some(&b), 3);
    assert_bit_identical(&blocked_linear(&[], 4, 0, &[], Some(&b), 3), &reference, "in_dim=0");
    assert_eq!(reference, [1.5, -2.0, 3.25].repeat(4));
}

// --------------------------------------------------------------- the sweep

/// The tile sweep that chose [`default_m_tile`]'s clamp, at LTX-2.5's real
/// adaLN-single shape. `#[ignore]`d: it allocates ~700 MB and runs for
/// minutes, and its output is a table for a human, not an assertion.
///
/// ```text
/// cargo test --release -p brain-backend-cpu --test host_gemm -- --ignored --nocapture tile_sweep
/// ```
#[test]
#[ignore = "multi-minute benchmark, allocates ~700 MB; run explicitly"]
fn tile_sweep_at_the_real_adaln_shape() {
    // `[3520, 4096] x [36864, 4096]T` - the real 720p latent token count
    // (T=3520) against the 22B checkpoint's 9-row adaLN-single table.
    let (rows, in_dim, out_dim) = (3520, 4096, 36864);
    eprintln!("building operands for [{rows},{in_dim}]x[{out_dim},{in_dim}]T ({:.0} MB of weights)...", (out_dim * in_dim * 4) as f64 / 1e6);
    let (x, w, b) = operands(rows, in_dim, out_dim, 0xADA1_0000);
    let flop = 2.0 * rows as f64 * in_dim as f64 * out_dim as f64;

    let t = std::time::Instant::now();
    let reference = naive_linear(&x, rows, in_dim, &w, Some(&b), out_dim);
    let naive_s = t.elapsed().as_secs_f64();
    eprintln!("| tile | secs | GFLOP/s | vs naive |");
    eprintln!("|---|---:|---:|---:|");
    eprintln!("| naive (row-parallel) | {naive_s:.2} | {:.1} | 1.00x |", flop / naive_s / 1e9);

    for m_tile in [MR, 16, 32, 64, 128, 256] {
        let t = std::time::Instant::now();
        let got = blocked_linear_tiled(&x, rows, in_dim, &w, Some(&b), out_dim, m_tile);
        let s = t.elapsed().as_secs_f64();
        assert_bit_identical(&got, &reference, &format!("tile {m_tile} at the real shape"));
        eprintln!("| {m_tile} | {s:.2} | {:.1} | {:.2}x |", flop / s / 1e9, naive_s / s);
    }
    eprintln!("default_m_tile() = {} (the row count does not enter it - see its doc)", default_m_tile());
}
