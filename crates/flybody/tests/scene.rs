// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The flight model is GENERATED, so the generation is what needs gating.
//!
//! A textual rewrite of someone else's XML has one dangerous failure mode: a
//! pattern that stops matching because the upstream file was reformatted. The
//! rewrite then silently does less than it claims, and what comes out is a fly
//! that flaps and does not fly while every other reading looks healthy. These
//! assert that every substitution is required and that a miss is loud.

use std::path::Path;

fn base() -> Option<std::path::PathBuf> {
    match std::env::var_os("BRAIN_FLYBODY_FRUITFLY_XML").filter(|v| !v.is_empty()) {
        Some(p) => Some(std::path::PathBuf::from(p)),
        None => {
            brain_testutil::skip("BRAIN_FLYBODY_FRUITFLY_XML unset (the model is not in this repo)");
            None
        }
    }
}

#[test]
fn the_flight_model_differs_from_the_walking_one_in_the_four_ways_that_matter() {
    let Some(src) = base() else { return };
    let dir = tempfile::tempdir().unwrap();
    let cfg = flybody::Flight::default();
    let out = flybody::flight_model(&src, dir.path(), cfg).expect("the flight model builds");
    let text = std::fs::read_to_string(&out).unwrap();
    let original = std::fs::read_to_string(&src).unwrap();

    // The one that decides whether flight is possible at all: without an
    // explicit ellipsoid fluid model the wings are approximated by their
    // inertia box and a flapping plate makes almost no lift.
    assert!(text.contains("fluidshape=\"ellipsoid\""), "the wing aerodynamic model was not set");
    assert!(!original.contains("fluidshape"), "the published model already sets a fluid shape; this rewrite is stale");

    assert!(text.contains(&format!("timestep=\"{}\"", cfg.timestep)), "the timestep was not changed");
    assert!(text.contains(&format!("gainprm=\"{}\"", cfg.wing_gain)), "the wing actuator gain was not changed");
    assert!(text.contains(&format!("damping=\"{}\"", cfg.wing_damping)), "the wing hinge damping was not changed");

    // The assets live next to the ORIGINAL, and the copy does not, so the
    // search paths have to be absolute or 85 meshes fail to resolve.
    let assets = src.parent().unwrap().canonicalize().unwrap();
    assert!(text.contains(&format!("meshdir=\"{}\"", assets.display())), "the mesh search path was not made absolute");
}

#[test]
fn a_model_this_rewrite_does_not_recognise_is_an_error_naming_the_pattern() {
    let Some(src) = base() else { return };
    let dir = tempfile::tempdir().unwrap();

    // THE CONTROL: the unmodified model must succeed first, or this test also
    // passes against a rewrite that rejects everything.
    assert!(flybody::flight_model(&src, dir.path(), flybody::Flight::default()).is_ok());

    // Now reformat one attribute the way a future release plausibly might.
    let mutated = dir.path().join("fruitfly.xml");
    let text = std::fs::read_to_string(&src).unwrap().replace("timestep=\"0.0001\"", "timestep=\"1e-4\"");
    std::fs::write(&mutated, text).unwrap();
    let err = flybody::flight_model(&mutated, dir.path(), flybody::Flight::default())
        .expect_err("a model with a reformatted timestep must be refused, not silently half-rewritten");
    assert!(err.contains("timestep"), "the error must name the pattern that missed: {err}");
}

#[test]
fn the_wing_muscle_vocabulary_splits_power_from_steering() {
    use flybody::{wing_muscle, WingAction};

    // Power muscles are matched by prefix because the published names carry
    // the fibre range: `DLM_a,_b` and `DLM_c-f` are rows of one muscle.
    for name in ["MN-WTct-DLM_a,_b", "MN-WTct-DLM_c-f", "MN-WTct-DVM_1a-c", "MN-WTct-DVM_3a,_b"] {
        assert_eq!(wing_muscle(name), Some(WingAction::Power), "{name} is a power muscle");
    }
    // Steering muscles are not, and the basalare/axillary split has to carry
    // OPPOSITE polarity or an antagonist pair becomes an agonist pair.
    assert_eq!(wing_muscle("MN-UTct-b1"), Some(WingAction::Amplitude(1.0)));
    assert_eq!(wing_muscle("MN-multi-i1"), Some(WingAction::Amplitude(-1.0)));
    assert_eq!(wing_muscle("MN-UTct-hg3"), Some(WingAction::AngleOfAttack(1.0)));
    // And something that is neither is not quietly filed as one.
    assert_eq!(wing_muscle("MN-LegNpT2-Ti_flexor"), None);
    assert_eq!(wing_muscle("MN-LegNpT2-TTM"), None, "the tergotrochanter is a jump muscle, not a wing muscle");
}

/// The wingbeat's shape, which is what makes it a wingbeat rather than a wobble.
#[test]
fn the_check_that_the_flight_scene_wraps_the_model() {
    let Some(src) = base() else { return };
    let dir = tempfile::tempdir().unwrap();
    let scene = flybody::flight_scene(&src, dir.path(), flybody::Flight::default()).unwrap();
    let text = std::fs::read_to_string(&scene).unwrap();
    assert!(text.contains("<include file="), "the scene must include the generated body");
    assert!(!text.contains("type=\"plane\""), "a hovering fly wants no floor for its first downstroke to hit");
    assert!(Path::new(&dir.path().join("fruitfly-flight.xml")).is_file());
}
