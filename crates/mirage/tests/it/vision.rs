// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vision parity ladder against FP32 goldens dumped from an independent
//! numpy reference of the same graph: every stage must keep cosine similarity
//! at or above 0.9999 with the reference tap, and proposals must match.

use std::collections::HashMap;
use std::sync::OnceLock;

use mirage::vision::dense::Proposal;
use mirage::vision::masks::{self, coefficient_cell, round_box};
use mirage::vision::ops::valid_cells;
use mirage::vision::preprocess::{letterbox, letterbox_size, Resized};
use mirage::vision::{Analysis, ProposalOptions, VisionModel};

use crate::common::released_dir;

/// Minimum cosine similarity per stage.
const MIN_COSINE: f64 = 0.9999;

pub struct Fixture {
    pub model: VisionModel,
    pub golden: HashMap<String, Vec<f32>>,
    pub image: Resized,
    pub analysis: Analysis,
}

/// The shared fixture: weights, goldens and one forward pass, or `None` (after
/// a loud skip) when any of them is absent.
pub fn fixture(test: &str) -> Option<&'static Fixture> {
    static CELL: OnceLock<Option<Fixture>> = OnceLock::new();
    let f = CELL.get_or_init(|| {
        let dir = released_dir()?;
        let path = brain_testutil::testdata("golden/mirage/goldens/cats.safetensors");
        if !std::path::Path::new(&path).is_file() {
            return None;
        }
        let ck = mirage::import::load_dir(&dir, &mirage::import::ImportOptions::default()).expect("released checkpoint imports");
        let model = VisionModel::new(&ck);
        let golden = checkpoint::st::load_safetensors(&path).expect("golden loads").tensors;
        let rgb: Vec<u8> = golden["input_rgb"].iter().map(|&v| v as u8).collect();
        let image = letterbox(&rgb, 480, 640).expect("golden image letterboxes");
        // The reference keeps every cell as a peak candidate, padding included.
        let analysis = model.analyze(&image, ProposalOptions { valid_only: false, ..ProposalOptions::default() });
        Some(Fixture { model, golden, image, analysis })
    });
    if f.is_none() {
        brain_testutil::skip(&format!("{test}: the released checkpoint or testdata/golden/mirage/goldens/cats.safetensors is absent (make testdata/fetch)"));
    }
    f.as_ref()
}

/// Cosine similarity and largest absolute difference.
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f32) {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut a, mut b, mut worst) = (0f64, 0f64, 0f64, 0f32);
    for (&x, &y) in got.iter().zip(want) {
        dot += f64::from(x) * f64::from(y);
        a += f64::from(x) * f64::from(x);
        b += f64::from(y) * f64::from(y);
        worst = worst.max((x - y).abs());
    }
    (dot / (a.sqrt() * b.sqrt()).max(1e-30), worst)
}

pub fn assert_stage(name: &str, got: &[f32], want: &[f32]) {
    let (cos, max_abs) = compare(got, want);
    eprintln!("parity {name}: cosine {cos:.8}, max_abs {max_abs:.3e}");
    assert!(cos >= MIN_COSINE, "{name}: cosine {cos} below {MIN_COSINE} (max_abs {max_abs})");
}

#[test]
fn the_letterbox_reproduces_the_reference_resize() {
    let Some(f) = fixture("letterbox") else { return };
    assert_eq!(letterbox_size(480, 640), (384, 512));
    assert_eq!(letterbox_size(1000, 1000), (512, 512));
    assert_eq!(letterbox_size(100, 4000), (32, 512), "a thin image keeps the 32-pixel floor");
    let want = &f.golden["resized_rgb"];
    let differing = f.image.rgb.iter().zip(want).filter(|(a, b)| f32::from(**a) != **b).count();
    eprintln!("letterbox: {differing} of {} samples differ", want.len());
    assert_eq!(differing, 0);
    assert_eq!(f.golden["valid_hw"], vec![384.0, 512.0]);
}

#[test]
fn the_stem_stages_match() {
    let Some(f) = fixture("stem") else { return };
    for ((map, got), name) in f.analysis.encoder.stages.iter().zip(["stage_256", "stage_128", "stage_64", "stage_32", "stage_16"]) {
        assert_eq!(map.h * map.w * map.c, f.golden[name].len(), "{name} geometry");
        assert_stage(name, got, &f.golden[name]);
    }
}

#[test]
fn the_body_top_and_encoder_outputs_match() {
    let Some(f) = fixture("encoder") else { return };
    assert_stage("body", &f.analysis.encoder.body, &f.golden["body"]);
    assert_stage("top", &f.analysis.encoder.top, &f.golden["top"]);
    assert_stage("encoder", &f.analysis.encoder.encoder, &f.golden["encoder"]);
}

#[test]
fn the_dense_head_matches() {
    let Some(f) = fixture("dense") else { return };
    assert_stage("dense64", &f.analysis.dense.d64, &f.golden["dense64"]);
    assert_stage("head_in", &f.analysis.dense.head_in, &f.golden["head_in"]);
    assert_stage("head", &f.analysis.dense.head, &f.golden["head"]);
}

fn assert_proposals(name: &str, got: &[Proposal], want: &[f32]) {
    assert_eq!(got.len(), want.len() / 7, "{name}: proposal count");
    for (i, (p, w)) in got.iter().zip(want.as_chunks::<7>().0.iter()).enumerate() {
        assert_eq!((p.y, p.x), (w[1] as usize, w[2] as usize), "{name}[{i}]: peak cell");
        assert!((p.score - w[0]).abs() < 1e-3, "{name}[{i}]: score {} vs {}", p.score, w[0]);
        for (a, b) in p.bbox.iter().zip(&w[3..7]) {
            assert!((a - b).abs() < 1e-3, "{name}[{i}]: box {:?} vs {:?}", p.bbox, &w[3..7]);
        }
    }
}

#[test]
fn proposals_and_their_order_match() {
    let Some(f) = fixture("proposals") else { return };
    assert_proposals("objects", &f.analysis.objects, &f.golden["peaks_obj"]);
    assert_proposals("text", &f.analysis.text, &f.golden["peaks_txt"]);
}

#[test]
fn restricting_peaks_to_the_image_drops_padding_candidates() {
    let Some(f) = fixture("valid-only proposals") else { return };
    let all = &f.analysis;
    let strict = f.model.analyze(&f.image, ProposalOptions::default());
    assert_eq!(strict.objects, all.objects, "objects live inside the image");
    let valid = valid_cells(f.image.h, f.image.w, 64);
    assert!(strict.text.iter().all(|p| valid[p.y * 64 + p.x]));
    assert!(strict.text.len() < all.text.len());
}

#[test]
fn mask_prototypes_and_coefficients_match() {
    let Some(f) = fixture("masks") else { return };
    let a = &f.analysis;
    let (protos, coefs) = f.model.dense.mask_maps(&a.dense.head_in, &a.encoder, f.image.h, f.image.w);
    assert_stage("protos", &protos, &f.golden["protos"]);
    let want = &f.golden["coefs_obj"];
    let mut got = Vec::new();
    for p in &a.objects {
        let (cy, cx) = coefficient_cell(round_box(p.bbox));
        got.extend_from_slice(&coefs[(cy * masks::COEF_SIDE + cx) * 32..][..32]);
    }
    assert_stage("coefs_obj", &got, want);
}

#[test]
fn decoder_memory_matches() {
    let Some(f) = fixture("memory") else { return };
    let memory = f.model.decoder_memory(&f.analysis);
    assert_stage("memory", &memory, &f.golden["memory"]);
}

#[test]
fn the_reader_matches_on_both_windows() {
    let Some(f) = fixture("reader") else { return };
    for i in 0..2 {
        let window: Vec<u8> = f.golden[&format!("ocr_window_{i}")].iter().map(|&v| v as u8).collect();
        let logits = f.model.reader.forward(&window);
        assert_stage(&format!("ocr_logits_{i}"), &logits, &f.golden[&format!("ocr_logits_{i}")]);
    }
}
