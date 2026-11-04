// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::vit::rel_pos_gather` against HAND-COMPUTED tables.
//!
//! This is the one part of SAM's decomposed relative-position bias that is pure
//! host arithmetic, and it is where a half-pixel-vs-align-corners mistake or an
//! off-by-one in the shifted gather would hide: both produce a plausible bias
//! that trains, and the gradient checker downstream would still pass because it
//! differentiates whatever map this function chose. So the reference values
//! below are worked out by hand from the reference formulas, not captured from
//! a run.
//!
//! No device is touched.

use model::vit::rel_pos_gather;

/// `w0` and `w1` are the two taps of a linear interpolation, so they must sum
/// to exactly 1 for EVERY pair - including the clamped ends, where both taps
/// name the same table row and the sum is what makes the result `T[idx0]`.
fn assert_partition_of_unity(q: u32, k: u32, l: u32) {
    let g = rel_pos_gather(q, k, l);
    for i in 0..g.len() {
        assert!((g.w0[i] + g.w1[i] - 1.0).abs() < 1e-6, "q{q} k{k} l{l} entry {i}: w0+w1 = {}", g.w0[i] + g.w1[i]);
        assert!(g.idx0[i] < l && g.idx1[i] < l, "q{q} k{k} l{l} entry {i}: index out of table");
    }
}

/// `table_len == 2*max(q,k)-1`: no resample at all, so the gather is exactly
/// the shifted index and the second tap is inert.
///
/// q = k = 3 -> `max_rel_dist = 5`, `rel(i,j) = i - j + 2`:
/// ```text
///   i=0: 2 1 0
///   i=1: 3 2 1
///   i=2: 4 3 2
/// ```
#[test]
fn identity_table_is_the_bare_shifted_gather() {
    let g = rel_pos_gather(3, 3, 5);
    assert_eq!(g.idx0, vec![2, 1, 0, 3, 2, 1, 4, 3, 2]);
    assert_eq!(g.idx1, g.idx0);
    assert!(g.w0.iter().all(|&w| w == 1.0));
    assert!(g.w1.iter().all(|&w| w == 0.0));
}

/// Downsample 5 -> 3 (q = k = 2, `max_rel_dist = 3`), half-pixel rule
/// `src = (d + 0.5)*(5/3) - 0.5`:
/// ```text
///   d=0: 0.3333  -> rows 0,1 w1=1/3
///   d=1: 2.0     -> rows 2,3 w1=0
///   d=2: 3.6667  -> rows 3,4 w1=2/3
/// ```
/// with `rel(i,j) = i - j + 1`, i.e. `[[1,0],[2,1]]`.
#[test]
fn downsampled_table_uses_the_half_pixel_rule() {
    let g = rel_pos_gather(2, 2, 5);
    assert_eq!(g.idx0, vec![2, 0, 3, 2]);
    assert_eq!(g.idx1, vec![3, 1, 4, 3]);
    let want_w1 = [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 0.0];
    for (i, w) in want_w1.iter().enumerate() {
        assert!((g.w1[i] - w).abs() < 1e-6, "entry {i}: w1 {} != {w}", g.w1[i]);
    }
    assert_partition_of_unity(2, 2, 5);
}

/// Upsample 3 -> 5 (q = k = 3, `max_rel_dist = 5`),
/// `src = (d + 0.5)*0.6 - 0.5`:
/// ```text
///   d=0: -0.2 -> CLAMPED to 0   -> rows 0,1 w1=0
///   d=1:  0.4                   -> rows 0,1 w1=0.4
///   d=2:  1.0                   -> rows 1,2 w1=0
///   d=3:  1.6                   -> rows 1,2 w1=0.6
///   d=4:  2.2 -> last row, so idx1 == idx0 == 2 and the two taps
///                still sum to T[2]
/// ```
/// The clamp at both ends is the align_corners=False signature: an
/// align_corners=True implementation would map d=0 to 0.0 and d=4 to exactly
/// 2.0 with no clamping and no `w1` at the top end.
#[test]
fn upsampled_table_clamps_both_ends() {
    let g = rel_pos_gather(3, 3, 3);
    // Per resampled row d = 0..5, read straight off the block comment above.
    let i0 = [0u32, 0, 1, 1, 2];
    let i1 = [1u32, 1, 2, 2, 2];
    let w1 = [0.0f32, 0.4, 0.0, 0.6, 0.2];
    // rel(i,j) = i - j + 2 -> the same 3x3 index block as the identity case.
    let d = [2usize, 1, 0, 3, 2, 1, 4, 3, 2];
    for (e, &dd) in d.iter().enumerate() {
        assert_eq!(g.idx0[e], i0[dd], "entry {e} (d={dd}) idx0");
        assert_eq!(g.idx1[e], i1[dd], "entry {e} (d={dd}) idx1");
        assert!((g.w1[e] - w1[dd]).abs() < 1e-6, "entry {e} (d={dd}) w1 {} != {}", g.w1[e], w1[dd]);
    }
    assert_partition_of_unity(3, 3, 3);
}

/// The general `q != k` branch of the shifted gather, which SAM itself never
/// takes (its windows are square in the sense `q_size == k_size` on each axis)
/// but the formula supports: `q = 2`, `k = 4` gives
/// `max(k/q,1) = 2`, `max(q/k,1) = 1`, so `rel(i,j) = 2i - j + 3`:
/// ```text
///   i=0: 3 2 1 0
///   i=1: 5 4 3 2
/// ```
/// against a `2*4-1 = 7`-row table (identity, so `idx0` IS `rel`).
#[test]
fn asymmetric_extents_scale_both_coordinate_axes() {
    let g = rel_pos_gather(2, 4, 7);
    assert_eq!(g.idx0, vec![3, 2, 1, 0, 5, 4, 3, 2]);
    assert_eq!(g.idx1, g.idx0);
    // ... and the mirror case, which scales the OTHER axis.
    let g = rel_pos_gather(4, 2, 7);
    // max(k/q,1) = 1, max(q/k,1) = 2 -> rel(i,j) = i - 2j + 2
    assert_eq!(g.idx0, vec![2, 0, 3, 1, 4, 2, 5, 3]);
}

/// Every shape the DeepSeek-OCR SAM tower and the gradcheck fixture use, plus
/// the real ViT-B extents, must stay inside their tables and keep the two taps
/// a partition of unity.
#[test]
fn every_fixture_and_real_shape_is_in_range() {
    for &(q, k, l) in &[
        (4u32, 4u32, 5u32),   // fixture: windowed height, upsample
        (2, 2, 3),            // fixture: windowed width, identity
        (5, 5, 13),           // fixture: global height, downsample
        (3, 3, 7),            // fixture: global width, downsample
        (14, 14, 27),         // SAM ViT-B window, identity
        (64, 64, 127),        // SAM ViT-B global grid, identity
        (14, 14, 127),        // a global table reused at window resolution
    ] {
        assert_partition_of_unity(q, k, l);
    }
}
