// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for the direct Q8_0 -> packed-int8 path: it must produce exactly
//! the bytes the fp32 round trip produces, not merely close ones.
//!
//! Swedish Embedded AB implements quantized checkpoint import and low-memory
//! model loading for its clients. If your team needs expertise in on-device
//! model loading then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The claim being gated is EQUALITY OF BITS, and the reason it is available
//! at all is arithmetic rather than luck: `deq_q8_0` yields exactly
//! `(q as i8 as f32) * d`, needing at most 18 significand bits against fp32's
//! 24, so decoding a block reproduces the round trip's f32 input exactly, and
//! `group_scales`/`pack_row` are then literally the same functions
//! `quantize_weight` calls - over the SAME 32-element blocks Q8_0 itself uses,
//! now that `model::int8::GROUP` is 32.
//!
//! So these tests use `assert_eq!` on the packed `u32` words and on the `f32`
//! scales. If one goes red, the premise above is wrong somewhere and the fix
//! is to find out where - NOT to soften the assertion into a cosine/rel_l2
//! pair, which would let a genuinely wrong requantization through while
//! reporting a healthy-looking number.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use checkpoint::quantize::{convert, Policy, Tier};
use flux2::weights::DitWeights;

/// Deterministic filler with both signs and a magnitude spread along the row,
/// so neighbouring 32-element groups genuinely differ and a dropped or
/// mis-indexed scale shows.
fn filler(seed: u64, n: usize, row: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 32) as u32 as f32) / (u32::MAX as f32);
            (u - 0.5) * 2.0 * (1.0 + (i / row.max(1)) as f32 * 0.05)
        })
        .collect()
}

/// Write a Q8_0 GGUF holding one `[rows, cols]` tensor and map it back.
fn q8_fixture(name: &str, rows: usize, cols: usize, seed: u64, file: &str) -> (MmapGguf, String) {
    let mut src: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    src.insert(name.to_string(), (vec![rows, cols], filler(seed, rows * cols, cols)));
    let path = std::env::temp_dir().join(file).to_string_lossy().into_owned();
    let report = convert(&src, Tier::Q8_0, &Policy::new().min_elems(32), &[], &path, &mut |_, _| {}).unwrap();
    assert_eq!(report.quantized(), 1, "fixture must actually be stored as Q8_0");
    let g = MmapGguf::open(&path).unwrap();
    assert_eq!(g.dtype(name), Some("Q8_0"));
    (g, path)
}

/// The fp32 round trip the direct path replaces: decode the whole tensor,
/// slice the rectangle out, quantize it.
fn round_trip(g: &MmapGguf, name: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let full = g.tensor(name).unwrap().unwrap();
    let mut rect = Vec::with_capacity(n_out * k);
    for i in 0..n_out {
        let e0 = (r0 + i) * stride + c0;
        rect.extend_from_slice(&full[e0..e0 + k]);
    }
    model::int8::quantize_weight(&rect, n_out, k)
}

#[test]
fn gguf_int8_is_bit_identical_to_the_fp32_round_trip() {
    // 4096 columns is what klein-9b's `hidden` actually is; 160 rows keeps the
    // fixture small while still threading. Both bounds are block-aligned.
    let (rows, cols) = (160usize, 4096usize);
    let (g, path) = q8_fixture("w", rows, cols, 11, "flux2-direct-int8-rows.gguf");
    let src = DitWeights::gguf(&g);

    // Whole tensor, and the three row thirds a fused qkv is consumed as -
    // r0 != 0 is the case that a naive implementation gets wrong by ignoring
    // the row offset when locating blocks.
    for &(r0, n_out) in &[(0usize, rows), (0, rows / 4), (rows / 4, rows / 4), (rows / 2, rows / 2)] {
        let got = src.try_i8_rect("w", cols, r0, n_out, 0, cols).expect("Q8_0 rect must take the direct path");
        let want = round_trip(&g, "w", cols, r0, n_out, 0, cols);
        assert_eq!(got.0, want.0, "packed words, rows [{r0}, {})", r0 + n_out);
        assert_eq!(got.1, want.1, "group scales, rows [{r0}, {})", r0 + n_out);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_column_block_is_bit_identical_too() {
    // `linear2` is consumed as two COLUMN blocks of one stored matrix, which
    // is the case where the source elements of one output row are not
    // contiguous in the file.
    let (rows, stride) = (64usize, 4096usize + 512);
    let (g, path) = q8_fixture("w", rows, stride, 23, "flux2-direct-int8-cols.gguf");
    let src = DitWeights::gguf(&g);
    for &(c0, k) in &[(0usize, 4096usize), (4096, 512)] {
        let got = src.try_i8_rect("w", stride, 0, rows, c0, k).expect("block-aligned column range must take the direct path");
        let want = round_trip(&g, "w", stride, 0, rows, c0, k);
        assert_eq!(got.0, want.0, "packed words, cols [{c0}, {})", c0 + k);
        assert_eq!(got.1, want.1, "group scales, cols [{c0}, {})", c0 + k);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_unaligned_rectangle_declines_instead_of_returning_wrong_bytes() {
    // 20 is not a multiple of the 32-element block, so the direct path cannot
    // serve it exactly. Declining (None) sends the caller to the fp32 route;
    // returning approximate bytes here is the failure this asserts against.
    let (g, path) = q8_fixture("w", 32, 640, 5, "flux2-direct-int8-unaligned.gguf");
    let src = DitWeights::gguf(&g);
    assert!(src.try_i8_rect("w", 640, 0, 32, 0, 20).is_none(), "k=20 is not block-aligned");
    assert!(src.try_i8_rect("w", 640, 0, 32, 20, 64).is_none(), "c0=20 is not block-aligned");
    assert!(src.try_i8_rect("w", 650, 0, 32, 0, 650).is_none(), "stride=650 is not block-aligned");
    // ... and an aligned one still works, so the three above fail for the
    // stated reason rather than because the fixture is broken.
    assert!(src.try_i8_rect("w", 640, 0, 32, 0, 640).is_some());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_fp32_map_source_always_declines_the_direct_path() {
    let mut ts: flux2::Tensors = HashMap::new();
    ts.insert("w".to_string(), (vec![32, 64], filler(3, 32 * 64, 64)));
    let src = DitWeights::Map(&ts);
    assert!(src.try_i8_rect("w", 64, 0, 32, 0, 64).is_none());
    // but it still serves fp32
    src.with_f32("w", |d| assert_eq!(d.len(), 32 * 64));
}
