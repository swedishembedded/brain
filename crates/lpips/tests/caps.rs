// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS behind the capability interface (`brain do brain/lpips distance`).
//!
//! The weights come from the model store and the official example pairs from
//! `testdata/lpips/` (both put there by `tools/goldens/lpips_dump_reference.py`);
//! the tests that need them skip, by name, without them.

use std::sync::Arc;

use capability::blob::image_blob;
use capability::{Invocation, Media, Registry};
use lpips::caps::{manifest, LpipsProvider, MODEL};

/// The tolerance `reference.rs` holds the device to against the official
/// package's distances.
fn within_reference_tolerance(got: f64, want: f64) -> bool {
    (got - want).abs() <= 2e-4 * want.abs() + 1e-6
}

fn registry() -> Option<Registry> {
    match lpips::spec::resolve() {
        Ok((trunk, heads)) => {
            let mut reg = Registry::new();
            reg.register(Arc::new(LpipsProvider::new(trunk, heads)));
            Some(reg)
        }
        Err(e) => {
            brain_testutil::skip(&format!("LPIPS weights not in the model store ({e})"));
            None
        }
    }
}

fn distance(reg: &Registry, a: &(Vec<f32>, u32, u32), b: &(Vec<f32>, u32, u32)) -> Result<serde_json::Value, String> {
    let inv = Invocation::new().blob("a", image_blob(&a.0, a.1, a.2, 3)).blob("b", image_blob(&b.0, b.1, b.2, 3));
    reg.run(MODEL, "distance", inv, &mut |_| {}).map(|o| o.outputs)
}

/// A deterministic textured RGB image.
fn texture(w: u32, h: u32, phase: f32) -> (Vec<f32>, u32, u32) {
    let px = (0..w * h).flat_map(|p| {
        let (x, y) = ((p % w) as f32, (p / w) as f32);
        [
            0.5 + 0.5 * (0.31 * x + phase).sin(),
            0.5 + 0.5 * (0.23 * y - phase).cos(),
            0.5 + 0.5 * (0.17 * (x + y) + phase).sin(),
        ]
    });
    (px.collect(), w, h)
}

#[test]
fn the_manifest_takes_two_images_and_no_weights_location() {
    let m = manifest();
    assert_eq!(m.model, MODEL);
    let a = m.actions.iter().find(|a| a.name == "distance").expect("a `distance` action");
    assert!(!a.streaming);
    for name in ["a", "b"] {
        assert!(a.inputs.iter().any(|b| b.name == name && b.media == Media::Image && b.required), "input {name}");
    }
    assert!(a.params.is_empty(), "weights are resolved from the model store, never named by a caller");
}

#[test]
fn identical_images_are_at_distance_zero() {
    let Some(reg) = registry() else { return };
    let img = texture(64, 64, 0.0);
    let out = distance(&reg, &img, &img).expect("distance");
    assert_eq!(out["distance"].as_f64(), Some(0.0), "{out}");
}

#[test]
fn different_images_are_at_a_positive_distance_with_one_share_per_tap() {
    let Some(reg) = registry() else { return };
    let out = distance(&reg, &texture(64, 64, 0.0), &texture(64, 64, 2.0)).expect("distance");
    let total = out["distance"].as_f64().expect("scalar distance");
    assert!(total > 0.0, "{out}");
    let layers: Vec<f64> = out["layers"].as_array().expect("per-tap shares").iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(layers.len(), 5);
    assert!(within_reference_tolerance(layers.iter().sum(), total), "{layers:?} vs {total}");
}

#[test]
fn the_official_example_distances_are_reproduced() {
    let Some(reg) = registry() else { return };
    let dir = brain_testutil::testdata_path("lpips");
    let Ok(manifest) = std::fs::read(dir.join("manifest.json")) else {
        brain_testutil::skip("testdata/lpips is not generated; run tools/goldens/lpips_dump_reference.py");
        return;
    };
    let manifest: serde_json::Value = serde_json::from_slice(&manifest).expect("manifest json");
    let load = |name: &str| {
        let img = imaging::load(dir.join(name)).expect("example image");
        (img.to_hwc_unit(), img.w, img.h)
    };
    let reference = load("ex_ref.png");
    for other in ["ex_p0.png", "ex_p1.png"] {
        let official = manifest["pairs"][format!("ex_ref.png|{other}")]["lpips"].as_f64().expect("official distance");
        let got = distance(&reg, &reference, &load(other)).expect("distance")["distance"].as_f64().unwrap();
        assert!(within_reference_tolerance(got, official), "{other}: {got} vs official {official}");
    }
}

#[test]
fn unusable_inputs_are_errors_not_scores() {
    let Some(reg) = registry() else { return };
    let (a, b) = (texture(64, 64, 0.0), texture(48, 64, 0.0));
    let err = distance(&reg, &a, &b).unwrap_err();
    assert!(err.contains("same size"), "{err}");
    let tiny = texture(16, 16, 0.0);
    assert!(distance(&reg, &tiny, &tiny).is_err(), "smaller than AlexNet can pool");
    let only_a = Invocation::new().blob("a", image_blob(&a.0, a.1, a.2, 3));
    assert!(reg.run(MODEL, "distance", only_a, &mut |_| {}).is_err(), "a missing input");
}

#[test]
fn absent_weights_are_a_clean_error() {
    let mut reg = Registry::new();
    reg.register(Arc::new(LpipsProvider::new("/nonexistent/trunk.pth", "/nonexistent/heads.safetensors")));
    let img = texture(64, 64, 0.0);
    let err = distance(&reg, &img, &img).unwrap_err();
    assert!(err.contains("/nonexistent"), "{err}");
}
