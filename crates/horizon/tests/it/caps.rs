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

/// A saved model whose `x1` was trained in `unit`, in a scratch directory.
fn saved_in(name: &str, unit: &str) -> (std::path::PathBuf, Saved) {
    let (mut subjects, _) = population(300, 5);
    for s in &mut subjects {
        s.observations.iter_mut().filter(|o| o.var == "x1").for_each(|o| o.unit = Some(unit.into()));
    }
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
    let dir = std::env::temp_dir().join(format!("horizon-caps-{name}-{}", std::process::id()));
    Saved::new(model, vocab).save(&dir).unwrap();
    (dir.clone(), Saved::load(&dir).unwrap())
}

const HISTORY: &str = r#"{"id": "p1", "as_of": 55.0, "calendar": 2000.5, "events": [
    {"time": 55.0, "code": "x1", "value": 0.4, "unit": "mmol/L"}, {"time": 55.0, "code": "x2", "value": 0.2},
    {"time": 55.0, "code": "noise", "value": 0.1}, {"time": 55.0, "code": "age", "value": 55.0},
    {"time": 55.0, "code": "group", "value": "a"}, {"time": 45.0, "code": "dx"}]}"#;

/// `predict` takes a patient history instead of subjects and answers with
/// the structured forecast: the same one `horizon::forecast` gives, as JSON
/// lines and as the `forecasts` output (what `--json` prints). A request that
/// gives both inputs, or a unit the model was not trained on, fails alone
/// while a request beside it in the same batch is answered.
#[test]
fn predict_takes_a_patient_history_and_answers_with_a_forecast() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (dir, saved) = saved_in("history", "mmol/L");
    let provider = HorizonProvider::new();
    let action = provider.action("predict").unwrap();
    let inv = |history: &str| {
        Invocation::new()
            .set("weights", json!(dir.to_string_lossy()))
            .set("times", json!("5, 10"))
            .set("max_ood_score", json!(1e9))
            .blob("history", Blob::new(Media::Text, history.as_bytes().to_vec()))
    };
    let out = action.run(&inv(HISTORY), &mut |_| {}).unwrap();
    assert_eq!(out.outputs["subjects"], json!(1));
    assert_eq!(out.outputs["abstained"], json!(0));
    let line = String::from_utf8(out.blobs["predictions"].bytes.clone()).unwrap();
    let served: Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(out.outputs["forecasts"][0], served, "the JSON output is the blob's forecast");

    let history = horizon::history::PatientHistory::parse_all(HISTORY).unwrap();
    let request = horizon::forecast::ForecastRequest::new([5.0, 10.0]).max_ood_score(1e9);
    let direct = horizon::forecast::forecast(&saved, &history, &request).unwrap();
    assert_eq!(served, serde_json::to_value(&direct[0]).unwrap(), "one code path");
    assert_eq!(served["subject_id"], "p1");
    assert_eq!(served["risk"], "available");
    assert_eq!(served["horizons"][1]["horizon"], 10.0);
    assert!(served["disclaimer"].as_str().unwrap().contains("not a treatment recommendation"));

    // Subjects and history together are ambiguous; neither is an error.
    let both = inv(HISTORY).blob("subjects", Blob::new(Media::Text, b"{}".to_vec()));
    assert!(action.run(&both, &mut |_| {}).unwrap_err().contains("not both"));
    let neither = Invocation::new().set("weights", json!(dir.to_string_lossy()));
    assert!(action.run(&neither, &mut |_| {}).unwrap_err().contains("'history'"));

    // In one batch, a request with a unit mismatch fails alone.
    let bad = HISTORY.replace("mmol/L", "mg/dL");
    let results = horizon::caps::predict_batch(&saved, &[inv(HISTORY), inv(&bad), inv(HISTORY)]);
    assert!(results[0].is_ok() && results[2].is_ok());
    let err = results[1].as_ref().unwrap_err();
    assert!(err.contains("mg/dL") && err.contains("mmol/L") && err.contains("p1"), "{err}");
    let (a, b) = (results[0].as_ref().unwrap(), results[2].as_ref().unwrap());
    assert_eq!(a.outputs["forecasts"], b.outputs["forecasts"], "neighbours do not matter");

    // A horizon past the last knot is refused, not extrapolated.
    let past = inv(HISTORY).set("times", json!("11"));
    assert!(action.run(&past, &mut |_| {}).unwrap_err().contains("outside the model's range"));
    std::fs::remove_dir_all(&dir).ok();
}
