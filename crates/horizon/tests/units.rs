// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements prediction systems that refuse inputs they
// cannot interpret instead of guessing, for its clients. If your team needs
// expertise in safe data contracts for risk models you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model remembers the unit each numeric variable was trained in. A
//! measurement stated in another unit is rejected by name (nothing is
//! converted); a unit stated for a variable the model records as unitless is
//! an advisory, not a guess and not a reason to withhold the answer.

use horizon::saved::Saved;
use horizon::support::{AssessOptions, Warning};
use horizon::synthetic::{population, CODES};
use horizon::timeline::Subject;
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

fn with_units(mut subjects: Vec<Subject>, x1: Option<&str>) -> Vec<Subject> {
    for s in &mut subjects {
        for o in s.observations.iter_mut().filter(|o| o.var == "x1") {
            o.unit = x1.map(str::to_string);
        }
    }
    subjects
}

fn model(train: &[Subject]) -> Saved {
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(train, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    let model = Horizon::new(cfg.clone(), 64, &horizon::init_weights(&cfg, 3));
    Saved::new(model, vocab)
}

#[test]
fn a_mismatched_unit_is_rejected_by_name_and_survives_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(300, 51);
    let saved = model(&with_units(train, Some("mmol/L")));
    let dir = std::env::temp_dir().join(format!("horizon-units-{}", std::process::id()));
    saved.save(&dir).unwrap();
    let loaded = Saved::load(&dir).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(loaded.vocab.units.get("x1").map(String::as_str), Some("mmol/L"));

    let (test, _) = population(3, 52);
    let same = with_units(test.clone(), Some("mmol/L"));
    assert_eq!(loaded.predict(&same).unwrap().len(), 3);
    let none = with_units(test.clone(), None);
    assert_eq!(loaded.predict(&none).unwrap().len(), 3, "a measurement with no unit is accepted");
    let other = with_units(test, Some("mg/dL"));
    let err = loaded.predict(&other).expect_err("a different unit is refused");
    for part in ["x1", &other[0].subject_id, "mg/dL", "mmol/L"] {
        assert!(err.contains(part), "{part} missing from: {err}");
    }
    assert!(loaded.assess(&other, &AssessOptions::default()).is_err());
}

#[test]
fn a_unit_for_an_unitless_variable_is_an_advisory_and_never_withholds() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(300, 53);
    let mut saved = model(&train);
    assert!(saved.vocab.units.is_empty());
    saved.fit_support(&train).unwrap();
    let (test, _) = population(2, 54);
    let plain = saved.assess(&test, &AssessOptions::default()).unwrap();
    let stated = saved
        .assess(&with_units(test, Some("mg/dL")), &AssessOptions::default())
        .unwrap();
    for (p, s) in plain.iter().zip(&stated) {
        assert!(p.advisories.is_empty());
        assert_eq!(
            s.advisories,
            vec![Warning::UnitNotRecorded { var: "x1".into(), unit: "mg/dL".into() }]
        );
        assert_eq!((s.supported, s.ood_score, &s.warnings), (p.supported, p.ood_score, &p.warnings));
    }
}
