// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified depth-estimation training for its clients. If your team needs expertise in
// training monocular depth models on your own camera and sensor data, you can procure our services by sending an
// email to info@swedishembedded.com.

//! P4: the paired RGB / depth directory loader (`zipdepth::dataset`), on a tiny dataset the test writes itself.
//!
//! A directory holds `manifest.json`, and per sample `<stem>.png` (or `.jpg`) beside `<stem>.depth.png`, a 16-bit
//! grayscale PNG of depth counts. Pinned here: the conversion to inverse depth and its validity mask, the crop and
//! resize onto the model input, the shuffled epochs, and that every malformed input is refused with a message naming
//! the file (and the field, for the manifest) instead of being coerced or dropped.

use std::path::Path;

use imaging::{save_gray16_png, save_png, Gray16, Rgb8};
use zipdepth::dataset::DepthDataset;

const SCALE: f64 = 0.5;

fn manifest(dir: &Path, body: &str) {
    std::fs::write(dir.join("manifest.json"), body).unwrap();
}

/// RGB whose red channel encodes the column and green the row, so a test can tell where a pixel came from.
fn write_rgb(dir: &Path, stem: &str, w: u32, h: u32) {
    let px: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| [x as u8 * 10, y as u8 * 10, 7])).collect();
    save_png(dir.join(format!("{stem}.png")), &Rgb8::new(w, h, px).unwrap()).unwrap();
}

/// Depth count `1 + x` (so a column identifies itself), 0 (invalid) at the top-left pixel.
fn write_depth(dir: &Path, stem: &str, w: u32, h: u32) {
    let px: Vec<u16> = (0..h).flat_map(|y| (0..w).map(move |x| if (x, y) == (0, 0) { 0 } else { 1 + x as u16 })).collect();
    save_gray16_png(dir.join(format!("{stem}.depth.png")), &Gray16 { w, h, px }).unwrap();
}

fn valid_set(dir: &Path, stems: &[&str], w: u32, h: u32) {
    manifest(dir, &format!(r#"{{"depth_per_count": {SCALE}, "invalid_count": 0}}"#));
    for s in stems {
        write_rgb(dir, s, w, h);
        write_depth(dir, s, w, h);
    }
}

/// Open the dataset and draw every sample at 8x8: the first refusal, which is what a trainer would hit.
fn first_error(dir: &Path) -> String {
    let ds = match DepthDataset::open(dir, 1) {
        Ok(ds) => ds,
        Err(e) => return e,
    };
    (0..ds.len()).find_map(|i| ds.sample(i, 8, 8).err()).expect("the dataset was expected to be refused")
}

#[test]
fn pairs_load_in_name_order_as_inverse_depth_with_a_validity_mask() {
    let dir = tempfile::tempdir().unwrap();
    valid_set(dir.path(), &["c", "a", "b"], 8, 8);
    let ds = DepthDataset::open(dir.path(), 1).unwrap();
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.stems(), ["a", "b", "c"]);
    let s = ds.sample(0, 8, 8).unwrap();
    assert_eq!((s.rgb.len(), s.inv.len(), s.mask.len()), (3 * 64, 64, 64));
    // Pixel (x=3, y=2): depth count 4 -> depth 2.0 -> inverse depth 0.5; red 30/255, green 20/255.
    let i = 2 * 8 + 3;
    assert_eq!(s.inv[i], (1.0 / (4.0 * SCALE)) as f32);
    assert_eq!(s.mask[i], 1.0);
    assert_eq!(s.rgb[i], 30.0 / 255.0);
    assert_eq!(s.rgb[64 + i], 20.0 / 255.0);
    // The invalid count is masked out and carries no value, not a huge inverse depth.
    assert_eq!((s.mask[0], s.inv[0]), (0.0, 0.0));
    assert_eq!(s.mask.iter().filter(|&&m| m == 1.0).count(), 63);
}

#[test]
fn a_wide_image_is_centre_cropped_to_the_model_aspect_before_resizing() {
    let dir = tempfile::tempdir().unwrap();
    valid_set(dir.path(), &["a"], 16, 8);
    let s = DepthDataset::open(dir.path(), 1).unwrap().sample(0, 8, 8).unwrap();
    // 16x8 onto a square: columns 4..12 survive, one source pixel per target pixel, so target column j is source
    // column 4 + j and its depth count is 5 + j. Rows are untouched.
    for j in 0..8 {
        assert_eq!(s.inv[3 * 8 + j], (1.0 / ((5 + j) as f64 * SCALE)) as f32, "column {j}");
        assert_eq!(s.rgb[3 * 8 + j], ((4 + j) * 10) as f32 / 255.0, "red of column {j}");
    }
}

#[test]
fn resizing_keeps_inverse_depth_and_mask_on_source_values() {
    // Nearest, not bilinear: a bilinear tap across a hole would invent a depth that no sensor measured.
    let dir = tempfile::tempdir().unwrap();
    valid_set(dir.path(), &["a"], 16, 16);
    let s = DepthDataset::open(dir.path(), 1).unwrap().sample(0, 8, 8).unwrap();
    let source: Vec<f32> = (1..=16).map(|c| (1.0 / (c as f64 * SCALE)) as f32).collect();
    for (&v, &m) in s.inv.iter().zip(&s.mask) {
        assert!(m == 0.0 && v == 0.0 || m == 1.0 && source.contains(&v), "{v} (mask {m}) is not a source value");
    }
}

/// What a malformed input is called, how it is made in a dataset dir, and what its refusal must name.
type Case = (&'static str, fn(&Path), &'static [&'static str]);

#[test]
fn every_malformed_input_is_refused_naming_the_file() {
    let cases: [Case; 12] = [
        ("an unknown manifest field", |d| manifest(d, r#"{"depth_per_count": 0.5, "invalid_count": 0, "scale": 2}"#), &["manifest.json", "scale"]),
        ("a missing manifest field", |d| manifest(d, r#"{"depth_per_count": 0.5}"#), &["manifest.json", "invalid_count"]),
        ("a non-positive depth unit", |d| manifest(d, r#"{"depth_per_count": 0, "invalid_count": 0}"#), &["manifest.json", "depth_per_count"]),
        ("a missing manifest", |d| std::fs::remove_file(d.join("manifest.json")).unwrap(), &["manifest.json"]),
        ("an RGB image without depth", |d| std::fs::remove_file(d.join("b.depth.png")).unwrap(), &["b.png", "b.depth.png"]),
        ("a depth map without an RGB image", |d| std::fs::remove_file(d.join("b.png")).unwrap(), &["b.depth.png", "b.png"]),
        ("an unrecognised file", |d| std::fs::write(d.join("notes.txt"), "x").unwrap(), &["notes.txt"]),
        ("an 8-bit depth map", |d| save_png(d.join("b.depth.png"), &Rgb8::new(8, 8, vec![9; 192]).unwrap()).unwrap(), &["b.depth.png", "16-bit"]),
        ("a depth map of another size", |d| write_depth(d, "b", 8, 4), &["b.png", "b.depth.png", "8x4"]),
        ("a truncated depth PNG", |d| std::fs::write(d.join("b.depth.png"), &std::fs::read(d.join("b.depth.png")).unwrap()[..40]).unwrap(), &["b.depth.png"]),
        ("a zero count that is not the invalid marker", |d| manifest(d, r#"{"depth_per_count": 0.5, "invalid_count": 65535}"#), &["a.depth.png", "count 0"]),
        ("an image with nothing valid", |d| {
            save_gray16_png(d.join("b.depth.png"), &Gray16 { w: 8, h: 8, px: vec![0; 64] }).unwrap();
        }, &["b.depth.png", "valid pixel"]),
    ];
    for (what, corrupt, must_name) in cases {
        let dir = tempfile::tempdir().unwrap();
        valid_set(dir.path(), &["a", "b"], 8, 8);
        corrupt(dir.path());
        let err = first_error(dir.path());
        for needle in must_name {
            assert!(err.contains(needle), "{what}: the refusal must name `{needle}`, got: {err}");
        }
    }
}

#[test]
fn an_empty_directory_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    manifest(dir.path(), r#"{"depth_per_count": 0.5, "invalid_count": 0}"#);
    assert!(first_error(dir.path()).contains("no samples"));
}

#[test]
fn epochs_are_seeded_permutations_that_visit_every_sample_once() {
    let dir = tempfile::tempdir().unwrap();
    valid_set(dir.path(), &["a", "b", "c", "d", "e"], 8, 8);
    let draw = |seed: u64, k: usize| DepthDataset::open(dir.path(), seed).unwrap().draw(k);
    for epoch in draw(3, 15).chunks(5) {
        let mut sorted = epoch.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3, 4], "an epoch must visit each sample exactly once");
    }
    assert_eq!(draw(3, 15), draw(3, 15), "the same seed must give the same order");
    let firsts: std::collections::HashSet<Vec<usize>> = (0..8).map(|seed| draw(seed, 5)).collect();
    assert!(firsts.len() > 1, "the order must depend on the seed");
}
