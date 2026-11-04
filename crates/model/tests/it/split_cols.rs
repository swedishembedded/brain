// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for `hostmath::split_cols`: splitting a fused weight's columns in
//! parallel must land every element exactly where the serial loop landed it.
//!
//! Swedish Embedded AB implements host-side checkpoint preparation and
//! quantized model-build paths for its clients. If your team needs expertise
//! in on-device model loading then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! This is pure data movement, so the property is EQUALITY OF BITS against
//! the serial `extend_from_slice` loop it replaced - a tolerance would be
//! meaningless here and would hide the failure mode that actually matters: a
//! row/stride mix-up that transposes or shifts the split, which leaves every
//! VALUE present and only their POSITIONS wrong.

use model::hostmath::split_cols;

/// The serial form `flux1::model` and `flux2::model` both used before
/// `split_cols` existed. Deliberately a verbatim transcription, not a
/// refactor: an oracle that shares structure with the code under test cannot
/// catch a shared mistake.
fn serial(w: &[f32], rows: usize, a: usize, b: usize) -> (Vec<f32>, Vec<f32>) {
    let mut left = Vec::with_capacity(rows * a);
    let mut right = Vec::with_capacity(rows * b);
    for r in 0..rows {
        left.extend_from_slice(&w[r * (a + b)..r * (a + b) + a]);
        right.extend_from_slice(&w[r * (a + b) + a..(r + 1) * (a + b)]);
    }
    (left, right)
}

fn filler(n: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
fn split_cols_is_bit_identical_to_the_serial_loop() {
    // A ladder of shapes, not one: `a == b` and `rows == a` are exactly the
    // coincidences that let a stride/row transposition pass unnoticed, so the
    // ladder deliberately includes asymmetric, non-square and non-multiple
    // cases alongside FLUX's own (d, d, mlp) proportions.
    for &(rows, a, b) in &[(1usize, 1usize, 1usize), (4, 4, 4), (3, 5, 7), (8, 2, 6), (7, 1, 13), (16, 16, 48), (5, 12, 3)] {
        let w = filler(rows * (a + b));
        let (ga, gb) = split_cols(&w, rows, a, b);
        let (wa, wb) = serial(&w, rows, a, b);
        assert_eq!(ga, wa, "left block, rows={rows} a={a} b={b}");
        assert_eq!(gb, wb, "right block, rows={rows} a={a} b={b}");
    }
}

#[test]
fn split_cols_is_bit_identical_at_a_size_that_actually_threads() {
    // The ladder above is small enough that rayon may run it on one thread,
    // which would leave the parallel split untested as a PARALLEL split.
    // FLUX.2 klein-9b's own proportions, scaled down but still many rows.
    let (rows, a, b) = (512usize, 256usize, 768usize);
    let w = filler(rows * (a + b));
    let (ga, gb) = split_cols(&w, rows, a, b);
    let (wa, wb) = serial(&w, rows, a, b);
    assert_eq!(ga.len(), rows * a);
    assert_eq!(gb.len(), rows * b);
    assert_eq!(ga, wa);
    assert_eq!(gb, wb);
}

#[test]
#[should_panic(expected = "split_cols")]
fn a_length_that_does_not_match_the_split_is_rejected_not_truncated() {
    let w = filler(10 * 8);
    let _ = split_cols(&w, 10, 3, 4); // 10*7 != 80
}
