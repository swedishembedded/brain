// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements risk models that say how sure they are, for
// its clients. If your team needs expertise in uncertainty for time-to-event
// predictions you can procure our services by sending an email to
// info@swedishembedded.com.

//! An ensemble is one trained object: its members differ by seed (and, for a
//! bootstrap, by the groups they were trained on), it is saved and loaded as
//! one directory whose manifest binds every member's weights by digest, and
//! `predict` answers with the mean and the spread of its members without the
//! caller asking for an ensemble.

use capability::{Blob, Invocation, Media, Outcome, Provider};
use horizon::caps::{predict, HorizonProvider};
use horizon::ensemble::{Ensemble, Kind, Loaded, Members, MANIFEST_FILE};
use horizon::fit::TrainSpec;
use horizon::synthetic::{population, CODES};
use horizon::timeline::Subject;
use horizon::HorizonConfig;
use serde_json::{json, Value};

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("horizon-ens-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn spec() -> TrainSpec {
    let mut cfg = HorizonConfig::default_for(1, CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    TrainSpec::new(CODES, ["death:a", "death:b"])
        .shape(cfg)
        .batch(32)
        .steps(20)
        .eval_interval(10)
        .seed(7)
}

/// Households of two, so a bootstrap has groups to resample.
fn households(subjects: &mut [Subject]) {
    for (i, s) in subjects.iter_mut().enumerate() {
        s.group_id = Some(format!("h{}", i / 2));
    }
}

fn train(kind: Kind, train: &[Subject], held: &[Subject]) -> Ensemble {
    let mut seen = Vec::new();
    let (ensemble, reports) =
        Ensemble::train(train, held, &spec(), 3, kind, &mut |m, _| seen.push(m), &|| false).unwrap();
    assert_eq!(reports.len(), 3);
    assert!(seen.contains(&0) && seen.contains(&2), "progress names the member");
    ensemble
}

fn jsonl(subjects: &[Subject]) -> Blob {
    let text: String = subjects.iter().map(|s| serde_json::to_string(s).unwrap() + "\n").collect();
    Blob::new(Media::Text, text.into_bytes())
}

fn lines(outcome: &Outcome) -> Vec<Value> {
    std::str::from_utf8(&outcome.blobs["predictions"].bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn request(subjects: &[Subject]) -> Invocation {
    Invocation::new().set("times", json!("2,5")).blob("subjects", jsonl(subjects))
}

#[test]
fn a_seeded_ensemble_saves_loads_and_predicts_identically() {
    if skip_gpu() {
        return;
    }
    let (train_set, _) = population(150, 1);
    let (held, _) = population(40, 2);
    let (test, _) = population(8, 3);
    let ensemble = train(Kind::Seeded, &train_set, &held);
    let manifest = ensemble.manifest();
    assert_eq!(manifest.members.iter().map(|m| m.seed).collect::<Vec<_>>(), [7, 8, 9]);
    assert!(manifest.members.iter().all(|m| m.bootstrap_draws.is_none()));
    let digests: std::collections::BTreeSet<_> = manifest.members.iter().map(|m| &m.weights_sha256).collect();
    assert_eq!(digests.len(), 3, "different seeds, different weights");

    let dir = scratch("seeded");
    let path = dir.join("ensemble");
    ensemble.save(&path, false).unwrap();
    assert!(ensemble.save(&path, false).unwrap_err().contains("already exists"));
    let loaded = Ensemble::load(&path).unwrap();
    assert_eq!(loaded.manifest(), ensemble.manifest());
    let (before, after) = (predict(&ensemble, &request(&test)).unwrap(), predict(&loaded, &request(&test)).unwrap());
    assert_eq!(lines(&before), lines(&after), "a saved and loaded ensemble predicts identically");

    // The mean is the mean of the members, and the spread is reported.
    let singles: Vec<Vec<Value>> = ensemble
        .members()
        .iter()
        .map(|m| lines(&predict(m, &request(&test)).unwrap()))
        .collect();
    for (i, line) in lines(&before).iter().enumerate() {
        assert_eq!(line["ensemble"]["members"], json!(3));
        for code in CODES {
            for t in 0..2 {
                let each: Vec<f64> = singles.iter().map(|s| s[i]["cif"][code][t].as_f64().unwrap()).collect();
                let mean = each.iter().sum::<f64>() / 3.0;
                assert!((line["cif"][code][t].as_f64().unwrap() - mean).abs() < 1e-12);
                let lo = each.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = each.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(line["ensemble"]["cif_min"][code][t].as_f64().unwrap(), lo);
                assert_eq!(line["ensemble"]["cif_max"][code][t].as_f64().unwrap(), hi);
            }
        }
    }

    // The served predict loads the directory without being told it is one.
    let served = HorizonProvider::new()
        .action("predict")
        .unwrap()
        .run(&request(&test).set("weights", json!(path.to_string_lossy())), &mut |_| {})
        .unwrap();
    assert_eq!(lines(&served), lines(&before));
    assert!(matches!(Loaded::load(&path).unwrap(), Loaded::Ensemble(_)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_member_that_is_not_the_one_trained_is_refused() {
    if skip_gpu() {
        return;
    }
    let (train_set, _) = population(100, 1);
    let (held, _) = population(30, 2);
    let ensemble = train(Kind::Seeded, &train_set, &held);
    let dir = scratch("digest");
    let path = dir.join("ensemble");
    ensemble.save(&path, false).unwrap();
    // Swap member 1's weights for member 0's: a valid model, the wrong one.
    std::fs::copy(path.join("members/0/model.safetensors"), path.join("members/1/model.safetensors")).unwrap();
    let err = Ensemble::load(&path).err().expect("refused");
    assert!(err.contains("member 1") && err.contains("manifest records"), "{err}");

    // A manifest cannot point outside the ensemble.
    ensemble.save(&path, true).unwrap();
    let text = std::fs::read_to_string(path.join(MANIFEST_FILE)).unwrap();
    std::fs::write(path.join(MANIFEST_FILE), text.replace("members/1", "../elsewhere")).unwrap();
    assert!(Ensemble::load(&path).err().unwrap().contains("expected"));

    // Only an ensemble directory is replaced.
    let other = dir.join("other");
    std::fs::create_dir_all(&other).unwrap();
    assert!(ensemble.save(&other, true).unwrap_err().contains("not an ensemble"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_bootstrap_ensemble_records_the_groups_each_member_drew() {
    if skip_gpu() {
        return;
    }
    let (mut train_set, _) = population(120, 1);
    households(&mut train_set);
    let (held, _) = population(30, 2);
    let ensemble = train(Kind::Bootstrap, &train_set, &held);
    let manifest = ensemble.manifest();
    assert_eq!(manifest.groups, Some(60));
    let draws: Vec<&Vec<u32>> = manifest.members.iter().map(|m| m.bootstrap_draws.as_ref().unwrap()).collect();
    for d in &draws {
        assert_eq!(d.len(), 60, "as many draws as groups");
        assert!(d.iter().all(|&g| g < 60));
        let distinct: std::collections::BTreeSet<_> = d.iter().collect();
        assert!(distinct.len() < 60, "with replacement: some group is drawn twice or never");
    }
    assert_ne!(draws[0], draws[1]);
    let dir = scratch("bootstrap");
    ensemble.save(&dir.join("e"), false).unwrap();
    assert_eq!(Ensemble::load(&dir.join("e")).unwrap().manifest(), manifest);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_ensemble_needs_two_members() {
    let (train_set, _) = population(20, 1);
    let err = Ensemble::train(&train_set, &train_set, &spec(), 1, Kind::Seeded, &mut |_, _| {}, &|| false)
        .err()
        .unwrap();
    assert!(err.to_string().contains("at least 2"));
}

const HISTORY: &str = r#"{"id": "p1", "as_of": 55.0, "calendar": 2000.5, "events": [
    {"time": 55.0, "code": "x1", "value": 0.4}, {"time": 55.0, "code": "x2", "value": 0.2},
    {"time": 55.0, "code": "noise", "value": 0.1}, {"time": 55.0, "code": "age", "value": 55.0},
    {"time": 55.0, "code": "group", "value": "a"}, {"time": 45.0, "code": "dx"}]}"#;

/// A patient history sent to an ensemble comes back as ONE forecast: the mean
/// of the members, the range they span at every horizon and each member's
/// identity.
#[test]
fn a_history_sent_to_an_ensemble_gets_the_mean_forecast_and_the_member_range() {
    if skip_gpu() {
        return;
    }
    let (train_set, _) = population(150, 1);
    let (held, _) = population(40, 2);
    let ensemble = train(Kind::Seeded, &train_set, &held);
    let inv = Invocation::new()
        .set("times", json!("2,5"))
        .blob("history", Blob::new(Media::Text, HISTORY.as_bytes().to_vec()));
    let forecasts = predict(&ensemble, &inv).unwrap().outputs["forecasts"].clone();
    let f = &forecasts[0];
    assert_eq!(f["ensemble"]["members"].as_array().unwrap().len(), 3);
    for code in CODES {
        let risk = &f["horizons"][1]["risks"][code];
        let [lo, hi] = [0, 1].map(|i| risk["member_range"][i].as_f64().unwrap());
        let mean = risk["raw"].as_f64().unwrap();
        assert!(lo <= mean && mean <= hi, "{code}: {lo} <= {mean} <= {hi}");
        let singles: Vec<f64> = ensemble
            .members()
            .iter()
            .map(|m| predict(m, &inv).unwrap().outputs["forecasts"][0]["horizons"][1]["risks"][code]["raw"].as_f64().unwrap())
            .collect();
        assert!((mean - singles.iter().sum::<f64>() / 3.0).abs() < 1e-12);
    }
}
