// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for `gguf::kquant::try_kq_rect`: for each of the six GGUF block
//! formats it targets, the relaid-out `(wq, wsm, wd)` must reconstruct to
//! EXACTLY the same bytes `checkpoint::gguf::MmapGguf::tensor` (the oracle
//! every other read path decodes through) produces - not merely close ones.
//!
//! Swedish Embedded AB implements quantized checkpoint import and device
//! layout tooling for its clients. If your team needs expertise in loading
//! GGUF K-quant checkpoints without a fp32 detour then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! This works because `try_kq_rect` performs no arithmetic on weight values:
//! it only re-derives `ds`/`dm` with the identical expressions the oracle's
//! private `deq_*` functions already use (M14: now via a packed `(sc,m)` u8
//! pair times a shared f16 `(d,dmin)` pair rather than one flat f32 product,
//! see `gguf::kquant`'s own module doc comment), and moves codes. So these
//! tests use `assert_eq!` on the reconstructed `f32` values - if one goes
//! red, the fix is to find the actual bit that disagrees, not to soften the
//! assertion.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use checkpoint::quantize::{convert, Policy, Tier};
use gguf::kquant::{try_kq_rect, unpack_row_codes, unpack_row_scales, KqLayout};

/// Deterministic filler with both signs and a magnitude spread, so
/// neighbouring groups genuinely differ and a dropped or mis-indexed scale
/// shows up as a wrong reconstructed value, not a coincidentally-right one.
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

/// Write a `tier`-quantized GGUF holding one `[rows, cols]` tensor named
/// `"w"` and map it back.
fn fixture(tier: Tier, rows: usize, cols: usize, seed: u64, file: &str) -> (MmapGguf, String) {
    let mut src: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    src.insert("w".to_string(), (vec![rows, cols], filler(seed, rows * cols, cols)));
    let path = std::env::temp_dir().join(file).to_string_lossy().into_owned();
    let report = convert(&src, tier, &Policy::new(), &[], &path, &mut |_, _| {}).unwrap();
    assert_eq!(report.quantized(), 1, "fixture must actually be stored as {}", tier.name());
    let g = MmapGguf::open(&path).unwrap();
    assert_eq!(g.dtype("w"), Some(tier.name()));
    (g, path)
}

/// Reconstruct every element of a relaid-out rectangle back to f32, using
/// exactly the algebra `try_kq_rect`'s own doc promises: `ds*code - dm`
/// (affine) or `ds*code` (symmetric, `dm` is always `0.0`), with `(ds, dm)`
/// unpacked from `(wsm, wd)` via `unpack_row_scales`.
fn reconstruct(wq: &[u32], wsm: &[u32], wd: &[u32], layout: &KqLayout) -> Vec<f32> {
    let wpr = layout.words_per_row();
    let wsm_wpr = layout.wsm_words_per_row();
    let wd_wpr = layout.wd_words_per_row();
    let mut out = Vec::with_capacity(layout.n * layout.k);
    for r in 0..layout.n {
        let words = &wq[r * wpr..(r + 1) * wpr];
        let codes = unpack_row_codes(words, layout);
        assert_eq!(codes.len(), layout.k);
        let (ds, dm) = unpack_row_scales(&wsm[r * wsm_wpr..(r + 1) * wsm_wpr], &wd[r * wd_wpr..(r + 1) * wd_wpr], layout);
        for (l, &code) in codes.iter().enumerate() {
            let g = l / layout.group;
            out.push(if layout.affine { ds[g] * code as f32 - dm[g] } else { ds[g] * code as f32 });
        }
    }
    out
}

/// The rectangle's oracle values: `MmapGguf::tensor`'s full decode, sliced.
fn oracle_rect(g: &MmapGguf, name: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize) -> Vec<f32> {
    let full = g.tensor(name).unwrap().unwrap();
    let mut rect = Vec::with_capacity(n_out * k);
    for i in 0..n_out {
        let e0 = (r0 + i) * stride + c0;
        rect.extend_from_slice(&full[e0..e0 + k]);
    }
    rect
}

/// One (tier, rows, cols) case exercised through both the whole-tensor
/// rectangle and a genuine sub-rectangle (`r0 != 0` AND `c0 != 0` at once -
/// block 0 alone cannot catch a mis-indexed row or column offset).
fn round_trips(tier: Tier, rows: usize, cols: usize, block: usize, seed: u64, file: &str) {
    let (g, path) = fixture(tier, rows, cols, seed, file);

    // Whole tensor.
    let (wq, wsm, wd, layout) = try_kq_rect(&g, "w", cols, 0, rows, 0, cols).unwrap_or_else(|| panic!("{}: whole-tensor rect must be servable", tier.name()));
    assert_eq!(layout.n, rows);
    assert_eq!(layout.k, cols);
    let got = reconstruct(&wq, &wsm, &wd, &layout);
    let want = oracle_rect(&g, "w", cols, 0, rows, 0, cols);
    assert_eq!(got, want, "{}: whole-tensor reconstruction", tier.name());

    // A genuine sub-rectangle: r0 != 0 AND c0 != 0 simultaneously.
    let r0 = 1;
    let n_out = rows - 2;
    let c0 = block;
    let k = cols - 2 * block;
    let (wq, wsm, wd, layout) = try_kq_rect(&g, "w", cols, r0, n_out, c0, k).unwrap_or_else(|| panic!("{}: sub-rectangle must be servable", tier.name()));
    let got = reconstruct(&wq, &wsm, &wd, &layout);
    let want = oracle_rect(&g, "w", cols, r0, n_out, c0, k);
    assert_eq!(got, want, "{}: sub-rectangle [{r0},{}) x [{c0},{}) reconstruction", tier.name(), r0 + n_out, c0 + k);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn q4_k_round_trips_bit_exact() {
    round_trips(Tier::Q4K, 8, 768, 256, 1, "kquant-q4k.gguf");
}

#[test]
fn q5_k_round_trips_bit_exact() {
    round_trips(Tier::Q5K, 8, 768, 256, 2, "kquant-q5k.gguf");
}

#[test]
fn q6_k_round_trips_bit_exact() {
    round_trips(Tier::Q6K, 8, 768, 256, 3, "kquant-q6k.gguf");
}

#[test]
fn q5_0_round_trips_bit_exact() {
    round_trips(Tier::Q5_0, 8, 128, 32, 4, "kquant-q5-0.gguf");
}

#[test]
fn q4_0_round_trips_bit_exact() {
    round_trips(Tier::Q4_0, 8, 128, 32, 5, "kquant-q4-0.gguf");
}

#[test]
fn q8_0_round_trips_bit_exact() {
    round_trips(Tier::Q8_0, 8, 128, 32, 6, "kquant-q8-0.gguf");
}

#[test]
fn k_not_a_multiple_of_256_declines_for_a_k_quant_type() {
    let (g, path) = fixture(Tier::Q4K, 4, 768, 7, "kquant-refuse-k.gguf");
    // 512 is 256-aligned (would be fine); 500 is not - a K-quant super-block
    // cannot be split, so this must decline rather than guess.
    assert!(try_kq_rect(&g, "w", 768, 0, 4, 0, 500).is_none(), "k=500 is not a multiple of 256");
    assert!(try_kq_rect(&g, "w", 768, 0, 4, 0, 512).is_some(), "k=512 is 256-aligned and must still work");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn unaligned_stride_or_c0_declines_for_a_legacy_type() {
    let (g, path) = fixture(Tier::Q8_0, 4, 128, 8, "kquant-refuse-legacy.gguf");
    assert!(try_kq_rect(&g, "w", 100, 0, 4, 0, 100).is_none(), "stride=100 is not a multiple of 32");
    assert!(try_kq_rect(&g, "w", 128, 0, 4, 20, 32).is_none(), "c0=20 is not a multiple of 32");
    assert!(try_kq_rect(&g, "w", 128, 0, 4, 0, 20).is_none(), "k=20 is not a multiple of 32");
    assert!(try_kq_rect(&g, "w", 128, 0, 4, 0, 128).is_some(), "an aligned rectangle must still work");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_unsupported_type_declines() {
    // A tensor kept as plain F32 (too small to quantize under this policy's
    // block-alignment rule) has a `GgmlType` `raw_blocks` happily reports,
    // but one `kquant::try_kq_rect` does not target.
    let mut src: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    src.insert("w".to_string(), (vec![4, 17], filler(9, 4 * 17, 17)));
    let path = std::env::temp_dir().join("kquant-refuse-unsupported.gguf").to_string_lossy().into_owned();
    // Row length 17 is not a multiple of any tier's block size, so `convert`
    // keeps it as F32 regardless of the requested tier.
    let report = convert(&src, Tier::Q8_0, &Policy::new(), &[], &path, &mut |_, _| {}).unwrap();
    assert_eq!(report.kept(), 1, "row=17 must not be block-aligned to any tier");
    let g = MmapGguf::open(&path).unwrap();
    assert_eq!(g.dtype("w"), Some("F32"));
    assert!(try_kq_rect(&g, "w", 17, 0, 4, 0, 17).is_none(), "F32 is not one of the six K-quant/legacy types");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_source_with_no_raw_blocks_declines() {
    let mut ts: HashMap<String, Vec<f32>> = HashMap::new();
    ts.insert("w".to_string(), filler(10, 4 * 128, 128));
    // The plain-`Vec<f32>` `TensorSource` impl inherits the trait's default
    // `raw_blocks`, which always declines - there is no quantized block
    // format to relay out of an already-f32 in-memory map.
    assert!(try_kq_rect(&ts, "w", 128, 0, 4, 0, 128).is_none());
}

/// (M14 gating (b)) The real device bytes-per-parameter for each of the six
/// types, recomputed from the ACTUAL layout rather than guessed: `wq` costs
/// `bits/8` bytes/param (no padding - `32/bits` codes/word exactly), `wsm`
/// costs `2/group` bytes/param (one packed `(sc,m)` byte pair per group,
/// two groups/word, `groups_per_row = k/group`, `wsm_words_per_row =
/// groups_per_row/2` at the even group counts every case below uses), `wd`
/// costs `4/spb` bytes/param (one packed `(d,dmin)` f16 pair per
/// super-block, `wd_words_per_row = k/spb`). Per this codebase's own lesson
/// about byte-cost formulas going stale silently: if this test ever goes
/// red, the fix is to recompute the ceiling from the layout, never to widen
/// the assertion's band.
#[test]
fn device_bytes_per_parameter_matches_the_recomputed_layout() {
    // (tier, rows, cols, seed, file, bits, group, spb)
    let cases: &[(Tier, u32, usize, usize, &str)] = &[
        (Tier::Q4K, 4, 32, 256, "kquant-bpp-q4k.gguf"),
        (Tier::Q5K, 8, 32, 256, "kquant-bpp-q5k.gguf"),
        (Tier::Q6K, 8, 16, 256, "kquant-bpp-q6k.gguf"),
        (Tier::Q5_0, 8, 32, 32, "kquant-bpp-q5-0.gguf"),
        (Tier::Q4_0, 4, 32, 32, "kquant-bpp-q4-0.gguf"),
        (Tier::Q8_0, 8, 32, 32, "kquant-bpp-q8-0.gguf"),
    ];
    let (rows, cols) = (4usize, 1024usize); // 1024 = 4 K-quant super-blocks, 32 legacy blocks
    for &(tier, bits, group, spb, file) in cases {
        let (g, path) = fixture(tier, rows, cols, 42, file);
        let (wq, wsm, wd, layout) = try_kq_rect(&g, "w", cols, 0, rows, 0, cols).unwrap_or_else(|| panic!("{}: rect must be servable", tier.name()));
        assert_eq!(layout.bits, bits, "{}: bits", tier.name());
        assert_eq!(layout.group, group, "{}: group", tier.name());
        assert_eq!(layout.spb, spb, "{}: spb", tier.name());

        let total_bytes = (wq.len() + wsm.len() + wd.len()) * 4;
        let n_params = rows * cols;
        let got_bpp = total_bytes as f64 / n_params as f64;
        let want_bpp = bits as f64 / 8.0 + 2.0 / group as f64 + 4.0 / spb as f64;
        assert!(
            (got_bpp - want_bpp).abs() < 1e-9,
            "{}: device bytes/param = {got_bpp} (wq={} + wsm={} + wd={} bytes over {n_params} params), \
             recomputed-from-layout formula (bits/8 + 2/group + 4/spb) says {want_bpp}",
            tier.name(),
            wq.len() * 4,
            wsm.len() * 4,
            wd.len() * 4
        );
        let _ = std::fs::remove_file(&path);
    }
}
