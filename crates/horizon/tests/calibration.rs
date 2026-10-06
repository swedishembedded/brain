// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements risk models whose probabilities hold up
// against outcomes observed years later, for its clients. If your team needs
// expertise in calibrating survival predictions you can procure our services
// by sending an email to info@swedishembedded.com.

//! A calibration is a file beside the weights: it survives save and load,
//! is served beside the raw risk, belongs to exactly the weights it was
//! fitted for, and is absent - not zero - where the validation data had too
//! few events.

use capability::{Blob, Invocation, Media, Provider};
use horizon::caps::HorizonProvider;
use horizon::saved::Saved;
use horizon::synthetic::{population, CODES};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use serde_json::{json, Value};

const HORIZONS: [f64; 3] = [2.0, 5.0, 10.0];

fn model(seed: u64, vocab: Vocab) -> Saved {
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    let model = Horizon::new(cfg.clone(), 64, &horizon::init_weights(&cfg, seed));
    Saved::new(model, vocab)
}

fn vocab(subjects: &[horizon::timeline::Subject]) -> Vocab {
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    Vocab::fit(subjects, &codes, &codes[..2], &FitOptions::default()).unwrap()
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("horizon-cal-{name}-{}", std::process::id()))
}

#[test]
fn a_calibration_survives_save_and_load_and_is_served() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (validation, _) = population(1500, 21);
    let mut saved = model(3, vocab(&validation));
    saved.calibrate(&validation, &HORIZONS, 30).unwrap();
    let cal = saved.calibration.clone().unwrap();
    assert_eq!(cal.validation_subjects, 1500);

    // Some (code, horizon) pairs have the events for a calibrator and some
    // do not: each is exactly one of the two.
    let mut calibrated = 0;
    for code in CODES {
        for h in HORIZONS {
            match (cal.entry(code, h), cal.uncalibrated().iter().find(|g| g.code == code && g.horizon == h)) {
                (Some(e), None) => {
                    calibrated += 1;
                    assert!(e.events >= 30, "{code}@{h}: {} events", e.events);
                }
                (None, Some(g)) => assert!(g.events < 30 || g.event_free < 30, "{code}@{h}: {g:?}"),
                other => panic!("{code}@{h}: neither or both: {other:?}"),
            }
        }
    }
    assert!(calibrated > 0 && calibrated < CODES.len() * HORIZONS.len(), "{calibrated} calibrated: the test needs both kinds");

    let dir = scratch("roundtrip");
    saved.save(&dir).unwrap();
    assert!(dir.join("calibration.json").is_file());
    let loaded = Saved::load(&dir).unwrap();
    let back = loaded.calibration.clone().expect("the calibration loads");
    assert_eq!(back.weights_sha256, cal.weights_sha256);
    let curves = loaded.predict(&validation[..40]).unwrap();
    for (k, code) in CODES.iter().enumerate() {
        for h in HORIZONS {
            for c in &curves {
                assert_eq!(cal.apply(code, h, c.cif(k, h)), back.apply(code, h, c.cif(k, h)), "{code}@{h}");
            }
            assert_eq!(cal.entry(code, h).is_some(), back.entry(code, h).is_some());
        }
    }

    // Served beside the raw risk; null where not calibrated.
    let jsonl: String = validation[..5]
        .iter()
        .map(|s| serde_json::to_string(s).unwrap() + "\n")
        .collect();
    let inv = Invocation::new()
        .set("weights", json!(dir.to_string_lossy()))
        .set("times", json!("2, 5, 10, 7.5"))
        .blob("subjects", Blob::new(Media::Text, jsonl.into_bytes()));
    let out = HorizonProvider::new().action("predict").unwrap().run(&inv, &mut |_| {}).unwrap();
    let text = std::str::from_utf8(&out.blobs["predictions"].bytes).unwrap();
    for (line, c) in text.lines().zip(&curves) {
        let line: Value = serde_json::from_str(line).unwrap();
        for (k, code) in CODES.iter().enumerate() {
            for (i, h) in [2.0, 5.0, 10.0, 7.5].into_iter().enumerate() {
                let served = &line["cif_calibrated"][code][i];
                match cal.apply(code, h, c.cif(k, h)) {
                    Some(x) => {
                        assert!((served.as_f64().unwrap() - x.risk).abs() < 1e-12);
                        assert!((line["cif_interval"][code][i][1].as_f64().unwrap() - x.upper).abs() < 1e-12);
                    }
                    None => assert!(served.is_null() && line["cif_interval"][code][i].is_null(), "{code}@{h}: {served}"),
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_calibration_of_other_weights_is_refused_and_an_old_directory_loads_uncalibrated() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (validation, _) = population(1500, 22);
    let v = vocab(&validation);
    let mut a = model(3, v.clone());
    a.calibrate(&validation, &[5.0], 30).unwrap();
    let (dir_a, dir_b) = (scratch("a"), scratch("b"));
    a.save(&dir_a).unwrap();
    // Same configuration, different weights, A's calibration.
    let b = model(4, v);
    b.save(&dir_b).unwrap();
    std::fs::copy(dir_a.join("calibration.json"), dir_b.join("calibration.json")).unwrap();
    let err = Saved::load(&dir_b).err().expect("refused");
    assert!(err.contains("different model"), "{err}");
    // A model whose weights changed cannot be saved with the old calibration.
    let mut moved = model(4, vocab(&validation));
    moved.calibration = a.calibration.clone();
    assert!(moved.save(&scratch("c")).unwrap_err().contains("fitted for weights"));
    std::fs::remove_dir_all(scratch("c")).ok();

    // Without the file the directory is a valid, uncalibrated model.
    std::fs::remove_file(dir_b.join("calibration.json")).unwrap();
    let old = Saved::load(&dir_b).unwrap();
    assert!(old.calibration.is_none());
    let inv = Invocation::new()
        .set("weights", json!(dir_b.to_string_lossy()))
        .blob("subjects", Blob::new(Media::Text, (serde_json::to_string(&validation[0]).unwrap() + "\n").into_bytes()));
    let out = HorizonProvider::new().action("predict").unwrap().run(&inv, &mut |_| {}).unwrap();
    let line: Value = serde_json::from_slice(&out.blobs["predictions"].bytes[..out.blobs["predictions"].bytes.len() - 1]).unwrap();
    assert!(line.get("cif_calibrated").is_none(), "an uncalibrated model serves no calibrated risk");
    // Saving an uncalibrated model over a calibrated directory removes the stale file.
    std::fs::copy(dir_a.join("calibration.json"), dir_a.join("stale.json")).unwrap();
    b.save(&dir_a).unwrap();
    assert!(!dir_a.join("calibration.json").exists());
    std::fs::remove_dir_all(&dir_a).ok();
    std::fs::remove_dir_all(&dir_b).ok();
}

#[test]
fn too_few_events_calibrate_nothing_and_a_bad_request_is_refused() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (validation, _) = population(300, 23);
    let mut saved = model(3, vocab(&validation));
    saved.calibrate(&validation, &HORIZONS, 10_000).unwrap();
    let cal = saved.calibration.as_ref().unwrap();
    assert!(cal.entries().is_empty());
    assert_eq!(cal.uncalibrated().len(), CODES.len() * HORIZONS.len());
    assert!(cal.apply("death:a", 5.0, 0.1).is_none());
    assert!(saved.calibrate(&validation, &[11.0], 30).unwrap_err().contains("outside the model's range"));
    assert!(saved.calibrate(&validation, &[0.0], 30).is_err());
    assert!(saved.calibrate(&[], &[5.0], 30).is_err());
    assert!(saved.calibrate(&validation, &[], 30).is_err());
}
