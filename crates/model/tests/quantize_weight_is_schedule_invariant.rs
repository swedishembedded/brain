// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight quantization must be **bit-identical regardless of how its rows are
//! scheduled** - the correctness contract behind running `model::int8::
//! quantize_weight` / `model::int4::quantize_weight_q4` row-parallel.
//!
//! Both quantizers are per-OUTPUT-ROW: row `r`'s scale is `max|w[r,:]|/q_max`
//! and row `r`'s packed words read only row `r`. Nothing crosses a row
//! boundary, so fanning the row loop across threads cannot change a value -
//! it can only change how long it takes. This file pins that as an assertion
//! instead of a comment, against an oracle written straight from the doc
//! comments' own formulas rather than by calling the implementation.
//!
//! The failure this catches is the obvious way to get a parallel quantizer
//! wrong: hoisting the `max|.|` fold out of the row loop into ONE scale for
//! the whole tensor (a reduction that IS associative, so it parallelises
//! "naturally" - and silently turns per-channel quantization into per-tensor,
//! which lesson #2's cosine-only ladder would not see either, because a
//! uniformly rescaled matrix keeps its direction).
//!
//! Swedish Embedded AB implements quantized inference tiers for its clients.
//! If your team needs expertise in int8/int4 weight packing and the numerical
//! gates that keep a quantized model honest, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// `n*k` deterministic, sign-varying, wide-dynamic-range values. Deliberately
/// NOT uniform per row: rows must end up with genuinely DIFFERENT `amax`, or a
/// per-tensor/per-row mix-up would produce the same answer either way (lesson
/// #4 - a degenerate fixture hides the whole bug class the test exists for).
fn fixture(n: usize, k: usize) -> Vec<f32> {
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut out = vec![0f32; n * k];
    for i in 0..n * k {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        // Row-dependent gain, so every row's own scale differs from its
        // neighbours'.
        out[i] = u * (1.0 + (i / k) as f32 * 0.37);
    }
    out
}

/// `quantize_weight`'s documented formula, transcribed - one row at a time,
/// no shared state, no threads.
fn oracle_int8(w: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let kg = k / 4;
    let mut sw = vec![0f32; n];
    let mut packed = vec![0u32; n * kg];
    for r in 0..n {
        let row = &w[r * k..r * k + k];
        let s = row.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 127.0;
        sw[r] = s;
        let inv = 1.0 / s;
        for g in 0..kg {
            let mut word = 0u32;
            for b in 0..4 {
                let q = (row[g * 4 + b] * inv).round().clamp(-127.0, 127.0) as i32;
                word |= ((q as u8) as u32) << (8 * b);
            }
            packed[r * kg + g] = word;
        }
    }
    (packed, sw)
}

/// `quantize_weight_q4`'s documented formula, transcribed the same way.
fn oracle_int4(w: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let kg = k / 8;
    let mut sw = vec![0f32; n];
    let mut packed = vec![0u32; n * kg];
    for r in 0..n {
        let row = &w[r * k..r * k + k];
        let s = row.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 7.0;
        sw[r] = s;
        let inv = 1.0 / s;
        for g in 0..kg {
            let mut word = 0u32;
            for b in 0..8 {
                let q = (row[g * 8 + b] * inv).round().clamp(-7.0, 7.0) as i32;
                word |= ((q as u8) as u32 & 0xF) << (4 * b);
            }
            packed[r * kg + g] = word;
        }
    }
    (packed, sw)
}

/// Shapes worth covering: one row (no parallelism available at all), a row
/// count under the core count, one that is not a multiple of any plausible
/// chunking, and one wide enough that the row loop genuinely splits.
const SHAPES: [(usize, usize); 5] = [(1, 64), (3, 8), (17, 24), (64, 4096), (129, 128)];

#[test]
fn int8_quantize_matches_a_serial_per_row_oracle_bit_for_bit() {
    for (n, k) in SHAPES {
        let w = fixture(n, k);
        let (packed, sw) = model::int8::quantize_weight(&w, n, k);
        let (want_packed, want_sw) = oracle_int8(&w, n, k);
        assert_eq!(packed, want_packed, "int8 packed words differ at [{n},{k}]");
        assert_eq!(sw, want_sw, "int8 per-row scales differ at [{n},{k}]");
        // Not merely "close": every row's scale must be its OWN, so a
        // per-tensor collapse is visible as an equality, not a tolerance.
        assert!(sw.iter().any(|&s| s != sw[0]) || n == 1, "fixture must give rows distinct scales at [{n},{k}]");
    }
}

#[test]
fn int4_quantize_matches_a_serial_per_row_oracle_bit_for_bit() {
    for (n, k) in SHAPES {
        let w = fixture(n, k);
        let (packed, sw) = model::int4::quantize_weight_q4(&w, n, k);
        let (want_packed, want_sw) = oracle_int4(&w, n, k);
        assert_eq!(packed, want_packed, "int4 packed words differ at [{n},{k}]");
        assert_eq!(sw, want_sw, "int4 per-row scales differ at [{n},{k}]");
    }
}

/// Repeated calls on the same input must agree exactly. A row-parallel
/// implementation that let two rows race on a shared accumulator would show up
/// here as run-to-run drift rather than as a wrong-but-stable answer.
#[test]
fn quantize_is_deterministic_across_repeated_runs() {
    let (n, k) = (129, 512);
    let w = fixture(n, k);
    let first = model::int8::quantize_weight(&w, n, k);
    for _ in 0..8 {
        assert_eq!(model::int8::quantize_weight(&w, n, k), first, "int8 quantization must not vary run to run");
    }
    let first4 = model::int4::quantize_weight_q4(&w, n, k);
    for _ in 0..8 {
        assert_eq!(model::int4::quantize_weight_q4(&w, n, k), first4, "int4 quantization must not vary run to run");
    }
}
