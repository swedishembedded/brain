// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements stateless record-to-risk inference whose
// answers are reproducible and auditable, for its clients. If your team needs
// expertise in safe longitudinal-record interfaces for risk models you can
// procure our services by sending an email to info@swedishembedded.com.

//! The patient-history format end to end: the same history gives the same
//! forecast, whatever order it is listed in and however often a record is
//! repeated; a new checkup changes it; the future never reaches the model; a
//! unit the model was not trained on is refused; and a forecast says what it
//! was computed from and withholds numbers outside the training support.

use horizon::evaluation::{evaluate, EvaluationSpec};
use horizon::forecast::{forecast, ForecastRequest, RiskForecast};
use horizon::history::{HistoryWarning, PatientHistory};
use horizon::saved::Saved;
use horizon::synthetic::{population, CODES};
use horizon::timeline::Subject;
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

fn model(train: &[Subject], seed: u64) -> Saved {
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(train, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 12;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    let model = Horizon::new(cfg.clone(), 64, &horizon::init_weights(&cfg, seed));
    Saved::new(model, vocab)
}

fn history(extra: &str) -> PatientHistory {
    let text = format!(
        r#"{{"id": "p1", "as_of": 55.0, "calendar": 2000.5, "events": [
            {{"time": 55.0, "code": "x1", "value": 0.4}},
            {{"time": 55.0, "code": "x2", "value": {{"below": -1.0}}}},
            {{"time": 55.0, "code": "noise", "value": 0.1}},
            {{"time": 55.0, "code": "age", "value": 55.0}},
            {{"time": 55.0, "code": "group", "value": "a"}},
            {{"time": 45.0, "code": "dx"}}{extra}]}}"#
    );
    PatientHistory::parse_all(&text).unwrap().remove(0)
}

fn one(saved: &Saved, h: &PatientHistory) -> RiskForecast {
    forecast(saved, std::slice::from_ref(h), &ForecastRequest::new([5.0]).max_ood_score(1e9))
        .unwrap()
        .remove(0)
}

fn numbers(f: &RiskForecast) -> (String, String) {
    (
        serde_json::to_string(&f.curves).unwrap(),
        serde_json::to_string(&f.horizons).unwrap(),
    )
}

#[test]
fn a_new_checkup_changes_the_forecast_and_a_repeat_or_a_reordering_does_not() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(300, 61);
    let saved = model(&train, 3);
    let base = one(&saved, &history(""));
    let checkup = r#", {"time": 50.0, "code": "x1", "value": 2.5}"#;
    let appended = one(&saved, &history(checkup));
    assert_ne!(numbers(&base), numbers(&appended), "an older checkup is evidence");
    assert_eq!(appended.coverage.observations, base.coverage.observations + 1);
    assert_eq!(appended.coverage.oldest_observation.as_ref().unwrap().time, 50.0);

    // The same record again: bit for bit the same numbers.
    let repeated = one(&saved, &history(&format!("{checkup}{checkup}")));
    assert_eq!(numbers(&repeated), numbers(&appended));
    assert!(repeated
        .input_warnings
        .iter()
        .any(|w| matches!(w, HistoryWarning::DuplicateIgnored { code, .. } if code == "x1")));
    assert_eq!(repeated.coverage, appended.coverage);

    // Listed in another order: the same forecast in every field.
    let shuffled = PatientHistory::parse_all(
        r#"{"id": "p1", "as_of": 55.0, "calendar": 2000.5, "events": [
            {"time": 45.0, "code": "dx"}, {"time": 50.0, "code": "x1", "value": 2.5},
            {"time": 55.0, "code": "group", "value": "a"}, {"time": 55.0, "code": "age", "value": 55.0},
            {"time": 55.0, "code": "noise", "value": 0.1}, {"time": 55.0, "code": "x1", "value": 0.4},
            {"time": 55.0, "code": "x2", "value": {"below": -1.0}}]}"#,
    )
    .unwrap()
    .remove(0);
    assert_eq!(one(&saved, &shuffled), appended);
}

#[test]
fn the_future_never_reaches_the_model() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(300, 62);
    let saved = model(&train, 4);
    let base = one(&saved, &history(""));
    let future = one(
        &saved,
        &history(r#", {"time": 56.0, "code": "x1", "value": 9.0}, {"time": 60.0, "code": "death:a"}, {"time": 55.0, "code": "dx:today"}"#),
    );
    assert_eq!(numbers(&future), numbers(&base));
    assert_eq!(future.coverage, base.coverage);
    let kinds: Vec<_> = future.input_warnings.iter().collect();
    assert_eq!(kinds.len(), 3, "{kinds:?}");
    assert!(matches!(kinds[0], HistoryWarning::FutureIgnored { code, .. } if code == "x1"));
    assert!(matches!(kinds[2], HistoryWarning::EventAtAsOf { code, .. } if code == "dx:today"));
}

#[test]
fn units_are_checked_against_the_model_and_a_malformed_one_never_gets_that_far() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (mut train, _) = population(300, 63);
    for s in &mut train {
        s.observations.iter_mut().filter(|o| o.var == "x1").for_each(|o| o.unit = Some("mmol/L".into()));
    }
    let saved = model(&train, 5);
    let with_unit = |u: &str| history(&format!(r#", {{"time": 50.0, "code": "x1", "value": 2.5, "unit": "{u}"}}"#));
    one(&saved, &with_unit("mmol/L"));
    let err = forecast(&saved, &[with_unit("mg/dL")], &ForecastRequest::new([5.0])).unwrap_err();
    for part in ["p1", "x1", "mg/dL", "mmol/L"] {
        assert!(err.contains(part), "{part} missing: {err}");
    }
    let err = PatientHistory::parse_all(
        r#"{"as_of": 55, "calendar": 2000, "events": [{"time": 50, "code": "x1", "value": 1, "unit": " mmol/L"}]}"#,
    )
    .unwrap_err();
    assert!(err.contains("events[0]") && err.contains("unit"), "{err}");
}

#[test]
fn a_forecast_reports_its_coverage_horizons_and_identity_and_refuses_what_it_cannot_say() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(1500, 64);
    let mut saved = model(&train, 6);
    let last = saved.horizon();
    let f = forecast(
        &saved,
        &[PatientHistory::parse_all(
            r#"{"as_of": 55.0, "calendar": 2000.5, "events": [{"time": 52.0, "code": "x1", "value": 0.4}]}"#,
        )
        .unwrap()
        .remove(0)],
        &ForecastRequest::new([5.0, 10.0]).max_ood_score(1e9),
    )
    .unwrap()
    .remove(0);
    assert!(f.is_available());
    assert_eq!(f.coverage.variables, vec!["x1"]);
    assert_eq!(f.coverage.missing_variables, vec!["age", "group", "noise", "x2"]);
    assert_eq!(f.coverage.newest_observation.as_ref().unwrap().ago, 3.0);
    assert_eq!(f.model.weights_sha256, saved.weights_digest().unwrap());
    assert_eq!(f.model.config_sha256.len(), 64);
    let curves = f.curves.as_ref().unwrap();
    assert_eq!(curves.times, vec![0.0, 2.0, 5.0, 10.0]);
    assert_eq!(curves.cif.len(), CODES.len());
    assert!(curves.cif["onset"].windows(2).all(|w| w[0] <= w[1]), "cumulative incidence never falls");
    assert_eq!(f.horizons.len(), 2);
    assert!(f.horizons[0].risks["onset"].calibrated.is_none(), "uncalibrated: absent, not zero");
    let json = serde_json::to_value(&f).unwrap();
    assert!(json["horizons"][0]["risks"]["onset"].get("calibrated").is_none());
    assert!(json["disclaimer"].as_str().unwrap().contains("not a diagnosis"));

    let past = ForecastRequest::new([last + 1.0]);
    assert!(forecast(&saved, &[history("")], &past).unwrap_err().contains("does not extrapolate"));

    // Calibrated at 5 only: the interval and calibrated risk exist there and nowhere else.
    let (validation, _) = population(2500, 65);
    saved.calibrate(&validation, &[5.0], 30).unwrap();
    let f = one_with(&saved, &[5.0, 10.0]);
    let risk = |h: usize| &f.horizons[h].risks["onset"];
    let (cal, [lo, hi]) = (risk(0).calibrated.unwrap(), risk(0).interval.unwrap());
    assert!(lo <= cal && cal <= hi, "{lo} {cal} {hi}");
    assert!(risk(1).calibrated.is_none() && risk(1).interval.is_none());
}

fn one_with(saved: &Saved, horizons: &[f64]) -> RiskForecast {
    forecast(saved, &[history("")], &ForecastRequest::new(horizons.iter().copied()).max_ood_score(1e9))
        .unwrap()
        .remove(0)
}

#[test]
fn a_history_outside_the_training_support_gets_no_numbers() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(1500, 66);
    let mut saved = model(&train, 7);
    saved.fit_support(&train).unwrap();
    let odd = PatientHistory::parse_all(
        r#"{"id": "odd", "as_of": 55.0, "calendar": 2000.5, "events": [{"time": 55.0, "code": "x1", "value": 400.0}]}"#,
    )
    .unwrap();
    let f = forecast(&saved, &odd, &ForecastRequest::new([5.0])).unwrap().remove(0);
    assert!(!f.is_available());
    assert_eq!(f.reason.as_deref(), Some("insufficient support"));
    assert!(f.curves.is_none() && f.horizons.is_empty());
    let json = serde_json::to_value(&f).unwrap();
    assert_eq!(json["risk"], "unavailable");
    assert!(json.get("curves").is_none());
    assert_eq!(json["support"]["supported"], false);
    // Raising the threshold on purpose gives the numbers back.
    let f = forecast(&saved, &odd, &ForecastRequest::new([5.0]).max_ood_score(1e9)).unwrap().remove(0);
    assert!(f.is_available());
}

#[test]
fn an_ensemble_averages_its_members_and_reports_their_range() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(300, 67);
    let (a, b) = (model(&train, 8), model(&train, 9));
    let h = history("");
    let (fa, fb) = (one(&a, &h), one(&b, &h));
    let both = RiskForecast::ensemble(&[fa.clone(), fb.clone()]).unwrap();
    let (ra, rb) = (fa.horizons[0].risks["onset"].raw, fb.horizons[0].risks["onset"].raw);
    let r = &both.horizons[0].risks["onset"];
    assert!((r.raw - 0.5 * (ra + rb)).abs() < 1e-12);
    assert_eq!(r.member_range, Some([ra.min(rb), ra.max(rb)]));
    assert_eq!(both.ensemble.as_ref().unwrap().members.len(), 2);
    assert_ne!(fa.model, fb.model);
    assert!(RiskForecast::ensemble(&[]).is_err());
}

#[test]
fn evaluation_reports_what_it_can_and_leaves_out_what_has_too_few_events() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(300, 68);
    let saved = model(&train, 10);
    let (mut test, _) = population(1500, 69);
    for (i, s) in test.iter_mut().enumerate() {
        s.group_id = Some(format!("g{}", i / 3));
    }
    let eval = evaluate(&saved, &test, &EvaluationSpec::new([5.0, 10.0]).min_events(30)).unwrap();
    assert_eq!(eval.subjects, 1500);
    assert!(eval.event_nll.is_finite());
    let m = eval.at("death:a", 10.0).expect("enough events at 10");
    assert!(m.events >= 30);
    let (c, auc, b, ibs) = (m.uno_c.unwrap(), m.auc.unwrap(), m.brier.unwrap(), m.integrated_brier.unwrap());
    assert!((0.0..=1.0).contains(&c) && (0.0..=1.0).contains(&auc) && (0.0..1.0).contains(&b) && (0.0..1.0).contains(&ibs));
    assert!(m.calibration.slope.is_some() && m.calibration.ece.is_some());
    let ci = m.intervals.unwrap().uno_c.unwrap();
    assert!(ci.lo <= ci.estimate && ci.estimate <= ci.hi, "{ci:?}");
    assert_eq!(ci.estimate, c, "the interval is centred on the full-sample statistic");

    let strict = evaluate(&saved, &test, &EvaluationSpec::new([5.0]).min_events(100_000)).unwrap();
    assert!(strict.results.is_empty());
    assert_eq!(strict.absent.len(), CODES.len());
    assert!(strict.absent.iter().all(|a| a.min_events == 100_000 && a.horizon == 5.0));
    assert!(evaluate(&saved, &test, &EvaluationSpec::new([saved.horizon() + 1.0])).is_err());
    for s in &mut test {
        s.group_id = None;
    }
    let ungrouped = evaluate(&saved, &test, &EvaluationSpec::new([10.0])).unwrap();
    assert!(ungrouped.at("death:a", 10.0).unwrap().intervals.is_none(), "no groups, no cluster bootstrap");
}
