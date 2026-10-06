// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `predict` capability returns, for a saved model, exactly what the
//! model predicts in-process, at the times asked for, and refuses what it
//! cannot answer (a time past the last knot, no subjects, no model).

use capability::{Blob, Invocation, Media, Provider};
use horizon::caps::HorizonProvider;
use horizon::saved::Saved;
use horizon::synthetic::{population, CODES};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use serde_json::{json, Value};

#[test]
fn predict_serves_what_the_saved_model_predicts() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (subjects, _) = population(300, 4);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&subjects, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    let model = Horizon::new(cfg.clone(), 64, &horizon::init_weights(&cfg, 3));
    let dir = std::env::temp_dir().join(format!("horizon-caps-{}", std::process::id()));
    Saved::new(model, vocab).save(&dir).unwrap();
    let saved = Saved::load(&dir).unwrap();

    let some = &subjects[..5];
    let jsonl: String = some
        .iter()
        .map(|s| serde_json::to_string(s).unwrap() + "\n")
        .collect();
    let provider = HorizonProvider::new();
    let action = provider.action("predict").unwrap();
    let inv = Invocation::new()
        .set("weights", json!(dir.to_string_lossy()))
        .set("times", json!("2, 7.5"))
        .blob("subjects", Blob::new(Media::Text, jsonl.into_bytes()));
    let out = action.run(&inv, &mut |_| {}).unwrap();
    let text = String::from_utf8(out.blobs["predictions"].bytes.clone()).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), some.len());
    let direct = saved.predict(some).unwrap();
    for ((line, s), c) in lines.iter().zip(some).zip(&direct) {
        assert_eq!(line["subject_id"], json!(s.subject_id));
        for (i, t) in [2.0, 7.5].into_iter().enumerate() {
            let survival = line["survival"][i].as_f64().unwrap();
            assert!(
                (survival - c.survival(t)).abs() < 1e-12,
                "{survival} vs {}",
                c.survival(t)
            );
            for (k, code) in saved.vocab.codes.iter().enumerate() {
                let cif = line["cif"][code][i].as_f64().unwrap();
                assert!((cif - c.cif(k, t)).abs() < 1e-12);
            }
        }
    }

    let past_the_knots = inv.clone().set("times", json!("11"));
    let err = action.run(&past_the_knots, &mut |_| {}).unwrap_err();
    assert!(err.contains("outside the model's range"), "{err}");
    let no_subjects = Invocation::new().set("weights", json!(dir.to_string_lossy()));
    assert!(action
        .run(&no_subjects, &mut |_| {})
        .unwrap_err()
        .contains("'subjects'"));
    let no_model = inv.set("weights", json!(dir.join("absent").to_string_lossy()));
    assert!(action
        .run(&no_model, &mut |_| {})
        .unwrap_err()
        .contains("no saved timeline model"));
    std::fs::remove_dir_all(&dir).ok();
}
