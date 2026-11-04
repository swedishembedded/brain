// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements prediction systems that know when not to
// answer, for its clients. If your team needs expertise in out-of-
// distribution detection and abstention for risk models you can procure our
// services by sending an email to info@swedishembedded.com.

//! A saved model remembers what it was trained on and says so when a subject
//! is outside it: unknown codes and variables, impossible values, an entry
//! far outside the trained clock, a history unlike any in training. The
//! served `predict` answers `risk: unavailable` for such a subject instead of
//! a probability; a model saved before this existed reports support unknown.

use capability::{Blob, Invocation, Media, Provider};
use horizon::caps::HorizonProvider;
use horizon::saved::Saved;
use horizon::support::{AssessOptions, Warning};
use horizon::synthetic::{population, CODES};
use horizon::timeline::{Event, Observation, Subject, Value};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use serde_json::{json, Value as Json};

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
    let mut saved = Saved::new(model, vocab);
    saved.fit_support(train).unwrap();
    saved
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("horizon-support-{name}-{}", std::process::id()))
}

fn obs(s: &mut Subject, var: &str, value: Value) {
    s.observations.push(Observation { t: s.entry, var: var.into(), value, unit: None });
}

fn kinds(saved: &Saved, s: &Subject) -> Vec<Warning> {
    saved.assess(std::slice::from_ref(s), &AssessOptions::default()).unwrap().remove(0).warnings
}

#[test]
fn what_is_outside_the_training_support_is_flagged_and_what_is_inside_is_not() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(1500, 41);
    let saved = model(&train);
    let (fresh, _) = population(1500, 42);
    let opts = AssessOptions::default();
    let assessed = saved.assess(&fresh, &opts).unwrap();
    let supported = assessed.iter().filter(|a| a.supported == Some(true)).count();
    eprintln!("in-distribution: {supported} of {} supported", fresh.len());
    assert!(supported as f64 >= 0.97 * fresh.len() as f64, "{supported} of {}", fresh.len());
    assert!(assessed.iter().all(|a| a.ood_score.is_some()));
    let inside = &fresh[0];
    assert!(kinds(&saved, inside).is_empty());

    let mut unknown_code = inside.clone();
    unknown_code.events.push(Event { t: unknown_code.entry - 1.0, code: "dx:never-seen".into() });
    assert!(kinds(&saved, &unknown_code).contains(&Warning::UnknownEventCode { code: "dx:never-seen".into() }));

    let mut unknown_var = inside.clone();
    obs(&mut unknown_var, "lactate", Value::Number(1.0));
    assert!(kinds(&saved, &unknown_var).contains(&Warning::UnknownVariable { var: "lactate".into() }));

    let mut unknown_level = inside.clone();
    obs(&mut unknown_level, "group", Value::Category("zz".into()));
    assert!(kinds(&saved, &unknown_level)
        .contains(&Warning::UnknownCategory { var: "group".into(), level: "zz".into() }));

    let mut impossible = inside.clone();
    obs(&mut impossible, "x1", Value::Number(1.0e6));
    assert!(matches!(kinds(&saved, &impossible)[..], [Warning::ValueOutOfRange { value, .. }] if value == 1.0e6));
    // The detection-limit form is judged by its limit.
    let mut limit = inside.clone();
    obs(&mut limit, "x1", Value::Above { above: 1.0e6 });
    assert!(!kinds(&saved, &limit).is_empty());

    for entry in [5.0, 130.0] {
        // The same record on a clock displaced to `entry`.
        let mut far = inside.clone();
        let delta = entry - far.entry;
        far.entry = entry;
        far.observations.iter_mut().for_each(|o| o.t += delta);
        far.events.iter_mut().for_each(|e| e.t += delta);
        far.at_risk.iter_mut().for_each(|w| {
            w.from += delta;
            w.to += delta;
        });
        far.validate().unwrap();
        assert!(
            kinds(&saved, &far).iter().any(|w| matches!(w, Warning::EntryOutOfRange { value, .. } if *value == entry)),
            "entry {entry}"
        );
    }
    let mut calendar = inside.clone();
    calendar.calendar_at_entry = 1850.0;
    assert!(kinds(&saved, &calendar).iter().any(|w| matches!(w, Warning::CalendarOutOfRange { .. })));

    let mut bare = inside.clone();
    bare.observations.clear();
    bare.events.clear();
    assert!(kinds(&saved, &bare).iter().any(|w| matches!(w, Warning::HistoryLength { measure, .. } if measure == "tokens")));
    let mut long = inside.clone();
    for i in 0..60 {
        long.observations.push(Observation { t: long.entry - 1.0 - i as f64, var: "x1".into(), value: Value::Number(0.0), unit: None });
    }
    assert!(kinds(&saved, &long).iter().any(|w| matches!(w, Warning::HistoryLength { .. })));

    // The score is continuous and ranks the worse subject higher.
    let scores: Vec<f64> = [inside, &impossible, &unknown_code]
        .iter()
        .map(|s| saved.assess(std::slice::from_ref(*s), &opts).unwrap()[0].ood_score.unwrap())
        .collect();
    assert!(scores[0] <= 1.0 && scores[1] > 1.0 && scores[2] >= horizon::support::UNKNOWN_SCORE, "{scores:?}");
    // A looser margin tolerates what a tight one flags.
    let loose = AssessOptions { margin: 10.0, ..opts };
    assert_eq!(saved.assess(&[impossible.clone()], &loose).unwrap()[0].supported, Some(false), "1e6 is far beyond even 10 widths");
    let mut modest = inside.clone();
    obs(&mut modest, "x1", Value::Number(6.0));
    assert_eq!(saved.assess(&[modest.clone()], &opts).unwrap()[0].supported, Some(false));
    assert_eq!(saved.assess(&[modest], &loose).unwrap()[0].supported, Some(true));
}

fn request(subjects: &[Subject], max_ood_score: Option<f64>, dir: &std::path::Path) -> Invocation {
    let jsonl: String = subjects.iter().map(|s| serde_json::to_string(s).unwrap() + "\n").collect();
    let mut inv = Invocation::new()
        .set("weights", json!(dir.to_string_lossy()))
        .set("times", json!("2,5"))
        .blob("subjects", Blob::new(Media::Text, jsonl.into_bytes()));
    if let Some(m) = max_ood_score {
        inv = inv.set("max_ood_score", json!(m));
    }
    inv
}

fn lines(out: &capability::Outcome) -> Vec<Json> {
    std::str::from_utf8(&out.blobs["predictions"].bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn an_unsupported_subject_gets_no_probability_and_the_support_travels_with_the_model() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(1500, 43);
    let saved = model(&train);
    let dir = scratch("serve");
    saved.save(&dir).unwrap();
    assert!(dir.join("support.json").is_file());

    let inside = train[0].clone();
    let mut unknown = train[1].clone();
    unknown.events.push(Event { t: unknown.entry - 1.0, code: "dx:never-seen".into() });
    let subjects = [inside.clone(), unknown.clone()];
    let action = HorizonProvider::new().action("predict").unwrap();

    // Default: the unsupported subject gets `risk: unavailable`, not numbers.
    let out = action.run(&request(&subjects, None, &dir), &mut |_| {}).unwrap();
    let served = lines(&out);
    assert_eq!(out.outputs["abstained"], json!(1));
    assert!(served[0].get("cif").is_some() && served[0]["support"]["supported"] == json!(true));
    assert_eq!(served[1]["risk"], json!("unavailable"));
    assert_eq!(served[1]["reason"], json!("insufficient support"));
    assert!(served[1].get("cif").is_none() && served[1].get("survival").is_none());
    assert_eq!(served[1]["support"]["warnings"][0]["kind"], json!("unknown_event_code"));
    assert_eq!(served[1]["support"]["warnings"][0]["code"], json!("dx:never-seen"));

    // The operator can raise the threshold: then the same subject is answered,
    // still carrying its warning, and the supported subject is unchanged.
    let out2 = action.run(&request(&subjects, Some(1e9), &dir), &mut |_| {}).unwrap();
    let tolerant = lines(&out2);
    assert_eq!(out2.outputs["abstained"], json!(0));
    assert!(tolerant[1].get("cif").is_some());
    assert_eq!(tolerant[1]["support"]["supported"], json!(false));
    assert_eq!(tolerant[0]["cif"], served[0]["cif"], "abstention never changes a supported answer");
    assert!(action.run(&request(&subjects, Some(0.0), &dir), &mut |_| {}).is_err(), "a threshold must be positive");

    // The support survives the reload: same verdicts.
    let loaded = Saved::load(&dir).unwrap();
    let a = loaded.assess(&subjects, &AssessOptions::default()).unwrap();
    let b = saved.assess(&subjects, &AssessOptions::default()).unwrap();
    assert_eq!(a[1].warnings, b[1].warnings);
    assert!((a[0].ood_score.unwrap() - b[0].ood_score.unwrap()).abs() < 1e-3);

    // A model directory without support.json reports support UNKNOWN: neither
    // supported nor unsupported, and nothing is withheld.
    std::fs::remove_file(dir.join("support.json")).unwrap();
    let old = Saved::load(&dir).unwrap();
    let verdict = old.assess(&subjects, &AssessOptions::default()).unwrap();
    assert!(verdict.iter().all(|a| a.supported.is_none() && a.ood_score.is_none() && a.warnings.is_empty()));
    // (A new provider: the first one keeps the model it loaded for the directory.)
    let out3 = HorizonProvider::new()
        .action("predict")
        .unwrap()
        .run(&request(&subjects, None, &dir), &mut |_| {})
        .unwrap();
    let unknown_support = lines(&out3);
    assert_eq!(out3.outputs["abstained"], json!(0));
    assert!(unknown_support.iter().all(|l| l["support"]["supported"].is_null() && l.get("cif").is_some()));

    // Support belongs to its weights: beside other weights it is refused.
    saved.save(&dir).unwrap();
    let other = scratch("other");
    let mut moved = model(&train);
    moved.model = Horizon::new(moved.model.cfg.clone(), 64, &horizon::init_weights(&moved.model.cfg, 8));
    moved.support = None;
    moved.save(&other).unwrap();
    std::fs::copy(dir.join("support.json"), other.join("support.json")).unwrap();
    assert!(Saved::load(&other).err().unwrap().contains("different model"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&other).ok();
}
