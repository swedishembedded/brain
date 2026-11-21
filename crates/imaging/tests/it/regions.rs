// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the pad-and-tile preprocessor (`imaging::regions`): its pixel rows are the reference processor's
//! `pixel_values` bit for bit, for RGB, RGBA (every alpha class, odd sizes, 1x1) and grayscale sources, and an
//! input outside the supported range is refused by name rather than clipped or wrapped.
//!
//! The goldens come from `tools/goldens/inkling2_vision_dump.py preproc`: PNG files plus the reference processor's
//! output for each, read back from the file by PIL as a caller would. The PNGs are the input of the Rust side too,
//! so the decoder is part of what is compared (lossless formats only: a JPEG decoder is allowed to differ).

use imaging::{Checkerboard, Decoded, RegionTiling, Rgb8};

/// The reference processor's constants: 40-pixel regions, an 8-pixel 204/153 checkerboard, a pad of -1 on the
/// 0-255 scale, mean and std 0.5 of the rescaled value.
const TILING: RegionTiling = RegionTiling {
    region: 40,
    checkerboard: Checkerboard { cell: 8, light: 204, dark: 153 },
    pad_fill: -1.0,
    rescale_factor: 0.00392156862745098,
    mean: [0.5; 3],
    std: [0.5; 3],
    max_side: 4096,
};

const CASES: &[&str] = &["rgb_odd", "rgba_mixed", "rgb_1x1", "rgba_1x1", "rgb_exact", "gray_odd", "rgba_opaque"];

#[test]
fn region_pixels_equal_the_reference_processor_bit_for_bit() {
    let dir = brain_testutil::testdata_path("golden/inkling2/vision/preproc");
    let golden_path = dir.join("golden.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip(&format!("{}: run tools/goldens/inkling2_vision_dump.py preproc", golden_path.display()));
        return;
    }
    let golden = brain_testutil::parity::load(&golden_path);
    for case in CASES {
        let bytes = std::fs::read(dir.join(format!("{case}.png"))).unwrap_or_else(|e| panic!("{case}: {e}"));
        let got = TILING.preprocess_bytes(&bytes).unwrap_or_else(|e| panic!("{case}: {e}"));
        let want = &golden[&format!("{case}.pixels")];
        let grid = &golden[&format!("{case}.grid")].data;
        assert_eq!(got.grid_thw().map(|v| v as f32).to_vec(), *grid, "{case}: grid");
        assert_eq!(got.pixels.len(), want.data.len(), "{case}: pixel count");
        let first_diff = got.pixels.iter().zip(&want.data).position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            first_diff.is_none(),
            "{case}: pixel {:?} differs: {:?} vs reference {:?}",
            first_diff,
            first_diff.map(|i| got.pixels[i]),
            first_diff.map(|i| want.data[i])
        );
    }
}

#[test]
fn input_outside_the_supported_range_is_refused_by_name() {
    let rgb = |w: u32, h: u32| Decoded::Rgb(Rgb8 { w, h, px: vec![7; w as usize * h as usize * 3] });
    // The limit is inclusive: a 4096-pixel side is the last supported one.
    let widest = TILING.preprocess(&rgb(4096, 1)).expect("4096 is within the limit");
    assert_eq!((widest.rows, widest.cols), (1, 103));
    for (w, h) in [(4097, 1), (1, 4097)] {
        let e = TILING.preprocess(&rgb(w, h)).unwrap_err();
        assert!(e.contains(&format!("{w}x{h}")) && e.contains("4096"), "{e}");
    }
    let e = TILING.preprocess(&Decoded::Rgb(Rgb8 { w: 0, h: 5, px: Vec::new() })).unwrap_err();
    assert!(e.contains("empty"), "{e}");
    let e = TILING.preprocess_bytes(b"not an image").unwrap_err();
    assert!(e.contains("unsupported image"), "{e}");
    let broken = RegionTiling { region: 0, ..TILING };
    assert!(broken.preprocess(&rgb(1, 1)).is_err());
}
