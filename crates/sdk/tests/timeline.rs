// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::TimelineModel` end to end: train on a synthetic population, save,
//! load, and predict the same curves from the loaded model; predictions rank
//! the subjects the truth says are riskier above the others.
#![cfg(feature = "timeline")]

use brain::timeline::{synthetic, TimelineModel, TimelineSpec};

#[test]
fn train_save_load_predict() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = synthetic::population(6000, 1);
    let (held_out, truth) = synthetic::population(1000, 2);
    let codes = ["death:a", "death:b", "onset"];
    let spec = TimelineSpec::new(codes, ["death:a", "death:b"])
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(400)
        .batch(128)
        .lr(3e-3);
    let (model, report) = TimelineModel::train(&train, &held_out, &spec).unwrap();
    assert!(report.held_out_event_nll.is_finite());
    assert_eq!(report.truncated_tokens, 0);
    let dir = std::env::temp_dir().join(format!("brain-timeline-sdk-{}", std::process::id()));
    model.save(&dir).unwrap();
    let loaded = TimelineModel::load(&dir).unwrap();
    let (a, b) = (
        model.predict(&held_out[..50]).unwrap(),
        loaded.predict(&held_out[..50]).unwrap(),
    );
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(
            x.cif("death:a", 10.0),
            y.cif("death:a", 10.0),
            "the loaded model predicts the same curves"
        );
    }
    assert_eq!(a[0].cif("no-such-code", 1.0), None);
    assert!(a
        .iter()
        .all(|p| (0.0..=1.0).contains(&p.cif("onset", 10.0).unwrap())));
    // The subjects the truth puts in the riskiest fifth get more predicted risk.
    let all = model.predict(&held_out).unwrap();
    let mut order: Vec<usize> = (0..held_out.len()).collect();
    order.sort_by(|&i, &j| {
        truth[i]
            .cif(0, 10.0)
            .partial_cmp(&truth[j].cif(0, 10.0))
            .unwrap()
    });
    let fifth = held_out.len() / 5;
    let mean = |ix: &[usize]| {
        ix.iter()
            .map(|&i| all[i].cif("death:a", 10.0).unwrap())
            .sum::<f64>()
            / ix.len() as f64
    };
    let (low, high) = (mean(&order[..fifth]), mean(&order[order.len() - fifth..]));
    assert!(
        high > 2.0 * low,
        "riskiest fifth {high:.4} vs safest fifth {low:.4}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Both visit backbones save and load: the loaded model predicts the same
/// curves as the trained one.
#[test]
fn visit_backbones_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use brain::timeline::synthetic::drifting::{self, Gaps};
    use brain::timeline::Backbone;
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 2.0), visits: (1, 4) };
    let (train, _) = drifting::population(2000, 1, &gaps, 10.0);
    let (held_out, _) = drifting::population(300, 2, &gaps, 10.0);
    for backbone in [Backbone::State, Backbone::Attention] {
        let spec = TimelineSpec::new([drifting::CODE], [drifting::CODE])
            .knots(vec![0.0, 2.0, 5.0, 10.0])
            .max_tokens(8)
            .steps(60)
            .batch(64)
            .visits(4)
            .backbone(backbone);
        let (model, _) = TimelineModel::train(&train, &held_out, &spec).unwrap();
        let dir = std::env::temp_dir().join(format!("brain-timeline-visits-{backbone:?}-{}", std::process::id()));
        model.save(&dir).unwrap();
        let loaded = TimelineModel::load(&dir).unwrap();
        assert_eq!(loaded.config().visits, 4);
        assert_eq!(loaded.config().backbone, backbone);
        let (a, b) = (model.predict(&held_out[..40]).unwrap(), loaded.predict(&held_out[..40]).unwrap());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.cif(drifting::CODE, 5.0), y.cif(drifting::CODE, 5.0), "{backbone:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Each mixer of the stack backbone trains, saves and loads: the loaded model
/// predicts exactly the curves the trained one does.
#[test]
fn stack_mixers_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use brain::timeline::synthetic::drifting::{self, Gaps};
    use brain::timeline::Mixer;
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 2.0), visits: (1, 4) };
    let (train, _) = drifting::population(2000, 1, &gaps, 10.0);
    let (held_out, _) = drifting::population(300, 2, &gaps, 10.0);
    for (mixer, blocks) in [(Mixer::Attention, 2), (Mixer::GatedDeltaNet, 2), (Mixer::Hybrid, 4)] {
        let spec = TimelineSpec::new([drifting::CODE], [drifting::CODE])
            .knots(vec![0.0, 2.0, 5.0, 10.0])
            .max_tokens(8)
            .steps(60)
            .batch(64)
            .visits(4)
            .mixer(mixer, blocks);
        let (model, _) = TimelineModel::train(&train, &held_out, &spec).unwrap();
        let dir = std::env::temp_dir().join(format!("brain-timeline-stack-{mixer:?}-{}", std::process::id()));
        model.save(&dir).unwrap();
        let loaded = TimelineModel::load(&dir).unwrap();
        assert_eq!(loaded.config().backbone, model.config().backbone, "{mixer:?}");
        let (a, b) = (model.predict(&held_out[..40]).unwrap(), loaded.predict(&held_out[..40]).unwrap());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.cif(drifting::CODE, 5.0), y.cif(drifting::CODE, 5.0), "{mixer:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The next-event group trains beside the outcomes, saves with the model and
/// reads back identically; its first-event probabilities add up to the
/// probability that any event has happened, and the held-out event NLL
/// reported is the outcome codes' alone.
#[test]
fn next_events_train_save_load_and_predict() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = synthetic::population(6000, 1);
    let (held_out, _) = synthetic::population(1000, 2);
    let codes = ["death:a", "death:b", "onset"];
    let spec = TimelineSpec::new(codes, ["death:a", "death:b"])
        .next_events(codes, 0.5)
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(400)
        .batch(128)
        .lr(3e-3);
    let (model, report) = TimelineModel::train(&train, &held_out, &spec).unwrap();
    assert!(report.held_out_event_nll.is_finite());
    assert_eq!(model.next_event_codes(), codes);
    assert_eq!(model.config().n_codes, 6, "three outcomes and three next events");
    let dir = std::env::temp_dir().join(format!("brain-timeline-next-{}", std::process::id()));
    model.save(&dir).unwrap();
    let loaded = TimelineModel::load(&dir).unwrap();
    let (a, b) = (
        model.predict_next_events(&held_out[..40]).unwrap(),
        loaded.predict_next_events(&held_out[..40]).unwrap(),
    );
    for (x, y) in a.iter().zip(&b) {
        for code in codes {
            assert_eq!(x.first(code, 8.0), y.first(code, 8.0), "{code} survives a reload");
        }
        let sum: f64 = codes.iter().map(|c| x.first(c, 8.0).unwrap()).sum();
        assert!((sum - x.any(8.0)).abs() < 1e-6, "first events sum to any event: {sum} vs {}", x.any(8.0));
        assert_eq!(x.first("no-such-code", 1.0), None);
    }
    // The outcome curves are still served, and a model without the group
    // returns no next-event forecasts.
    assert!(model.predict(&held_out[..5]).unwrap()[0].cif("onset", 8.0).is_some());
    let plain = TimelineSpec::new(codes, ["death:a", "death:b"]).steps(10).batch(128);
    let (plain_model, _) = TimelineModel::train(&train, &held_out, &plain).unwrap();
    assert!(plain_model.predict_next_events(&held_out[..5]).unwrap().is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

/// Calibrating a deliberately overconfident model (a wide model trained on
/// few subjects: its first-onset risk is too extreme) on validation subjects
/// moves its risks toward the truth on a test population neither saw: the
/// recalibration slope and the observed-over-expected ratio both go toward
/// one. The calibration survives save and load, is refused beside other
/// weights, and is absent - never zero - where it was not fitted.
#[test]
fn calibration_repairs_an_overconfident_model_and_persists() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use brain::survival::calibration::at_horizon;
    use brain::survival::estimate::censoring;
    use brain::timeline::{observed, CalibrationSpec};
    let (train, _) = synthetic::population(600, 31);
    let (stop, _) = synthetic::population(300, 32);
    let (validation, _) = synthetic::population(8000, 33);
    let (test, _) = synthetic::population(8000, 34);
    let codes = ["death:a", "death:b", "onset"];
    let spec = TimelineSpec::new(codes, ["death:a", "death:b"])
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(1500)
        .patience(1000)
        .batch(128)
        .lr(3e-3);
    let (mut model, _) = TimelineModel::train(&train, &stop, &spec).unwrap();
    assert!(model.calibration().is_none());
    let preds = model.predict(&test).unwrap();
    assert!(preds[0].calibrated_cif("onset", 5.0).is_none(), "no calibration, no calibrated risk");
    model.calibrate(&validation, &CalibrationSpec::new([5.0, 10.0])).unwrap();
    let preds = model.predict(&test).unwrap();

    // First onset competes with both deaths.
    let (t, list) = (5.0, ["onset", "death:a", "death:b"]);
    let obs = observed(&test, &list);
    let g = censoring(&obs);
    let raw: Vec<f64> = preds.iter().map(|p| p.cif("onset", t).unwrap()).collect();
    let cal: Vec<f64> = preds.iter().map(|p| p.calibrated_cif("onset", t).unwrap()).collect();
    let before = at_horizon(&raw, &obs, 0, t, &g, 10);
    let after = at_horizon(&cal, &obs, 0, t, &g, 10);
    eprintln!(
        "onset by {t}: slope {:.3} -> {:.3}, O/E {:.3} -> {:.3}, ECE {:.4} -> {:.4}",
        before.slope, after.slope, before.oe_ratio, after.oe_ratio, before.ece(), after.ece()
    );
    assert!(before.slope < 0.7, "the model must start overconfident: {before:?}");
    assert!(
        (after.slope - 1.0).abs() < 0.5 * (before.slope - 1.0).abs() && (after.slope - 1.0).abs() < 0.2,
        "slope {:.3} -> {:.3}",
        before.slope,
        after.slope
    );
    assert!(
        (after.oe_ratio - 1.0).abs() < (before.oe_ratio - 1.0).abs() && (after.oe_ratio - 1.0).abs() < 0.1,
        "O/E {:.3} -> {:.3}",
        before.oe_ratio,
        after.oe_ratio
    );
    assert!(after.ece() < before.ece(), "ECE {:.4} -> {:.4}", before.ece(), after.ece());
    for p in &preds {
        let (lo, hi) = p.cif_interval("onset", t).unwrap();
        let risk = p.calibrated_cif("onset", t).unwrap();
        assert!(lo <= risk && risk <= hi, "{lo} <= {risk} <= {hi}");
        assert!(p.calibrated_cif("onset", 7.0).is_none(), "a horizon that was not calibrated has no calibrated risk");
        assert!(p.calibrated_cif("no-such-code", t).is_none());
    }

    // It survives save and load, and is refused beside other weights.
    let dir = std::env::temp_dir().join(format!("brain-timeline-cal-{}", std::process::id()));
    model.save(&dir).unwrap();
    let loaded = TimelineModel::load(&dir).unwrap();
    assert_eq!(loaded.calibration().unwrap().entries().len(), 6);
    let again = loaded.predict(&test[..200]).unwrap();
    for (a, b) in preds.iter().zip(&again) {
        for code in codes {
            let (x, y) = (a.calibrated_cif(code, t).unwrap(), b.calibrated_cif(code, t).unwrap());
            assert!((x - y).abs() < 5e-3, "{code}: {x} vs {y}");
        }
    }
    let other = std::env::temp_dir().join(format!("brain-timeline-cal-other-{}", std::process::id()));
    let (unrelated, _) = TimelineModel::train(&train, &stop, &spec.clone().steps(5).seed(9)).unwrap();
    unrelated.save(&other).unwrap();
    std::fs::copy(dir.join("calibration.json"), other.join("calibration.json")).unwrap();
    assert!(TimelineModel::load(&other).unwrap_err().to_string().contains("different model"));

    // Too few events: nothing is calibrated and nothing is exposed.
    let report = model.calibrate(&validation, &CalibrationSpec::new([5.0]).min_events(100_000)).unwrap();
    assert!(report.entries().is_empty() && report.uncalibrated().len() == 3);
    assert!(model.predict(&test[..3]).unwrap().iter().all(|p| p.calibrated_cif("onset", 5.0).is_none()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&other).ok();
}

/// A subject as a patient history: every observation and past event on the
/// model's own clock, as the update format states them.
fn history_of(s: &brain::timeline::Subject) -> brain::timeline::PatientHistory {
    use brain::timeline::Value;
    let value = |v: &Value| match v {
        Value::Number(x) => serde_json::json!(x),
        Value::Below { below } => serde_json::json!({ "below": below }),
        Value::Above { above } => serde_json::json!({ "above": above }),
        Value::Category(c) => serde_json::json!(c),
    };
    let mut records: Vec<serde_json::Value> = s
        .known_observations()
        .map(|o| serde_json::json!({ "time": o.t, "code": o.var, "value": value(&o.value) }))
        .collect();
    records.extend(s.known_events().map(|e| serde_json::json!({ "time": e.t, "code": e.code })));
    let doc = serde_json::json!({
        "id": s.subject_id, "as_of": s.entry, "calendar": s.calendar_at_entry, "events": records,
    });
    brain::timeline::PatientHistory::parse_all(&doc.to_string()).unwrap().remove(0)
}

/// A trained model forecasts a patient history exactly as it predicts the
/// equivalent subject, reports calibrated risks only at calibrated horizons,
/// refuses a horizon beyond its knots, and its evaluation agrees with the
/// survival arithmetic computed by hand and with the truth of the synthetic
/// population.
#[test]
fn forecast_history_and_evaluate_a_trained_model() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use brain::survival::estimate::censoring;
    use brain::survival::{auc, concordance};
    use brain::timeline::{observed, CalibrationSpec, EvaluationSpec, ForecastRequest};
    let (train, _) = synthetic::population(6000, 71);
    let (stop, _) = synthetic::population(1000, 72);
    let (validation, _) = synthetic::population(4000, 73);
    let (mut test, truth) = synthetic::population(6000, 74);
    let codes = ["death:a", "death:b", "onset"];
    let spec = TimelineSpec::new(codes, ["death:a", "death:b"])
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(400)
        .batch(128)
        .lr(3e-3);
    let (mut model, _) = TimelineModel::train(&train, &stop, &spec).unwrap();
    model.calibrate(&validation, &CalibrationSpec::new([5.0, 10.0])).unwrap();

    // The history route equals the subject route, bit for bit.
    let request = ForecastRequest::new([5.0, 7.0, 10.0]);
    let sample = &test[..30];
    let histories: Vec<_> = sample.iter().map(history_of).collect();
    let forecasts = model.forecast_histories(&histories, &request).unwrap();
    let predictions = model.predict(sample).unwrap();
    let mut abstained = 0;
    for (f, p) in forecasts.iter().zip(&predictions) {
        if !f.is_available() {
            abstained += 1;
            assert!(f.curves.is_none() && f.horizons.is_empty());
            continue;
        }
        for h in &f.horizons {
            for code in codes {
                let r = &h.risks[code];
                assert_eq!(r.raw, p.cif(code, h.horizon).unwrap(), "{code} at {}", h.horizon);
                assert_eq!(r.calibrated, p.calibrated_cif(code, h.horizon), "{code} at {}", h.horizon);
                assert_eq!(r.interval, p.cif_interval(code, h.horizon).map(|(lo, hi)| [lo, hi]));
                if h.horizon == 7.0 {
                    assert!(r.calibrated.is_none() && r.interval.is_none(), "7 was not calibrated: absent");
                } else {
                    assert!(r.calibrated.is_some() && r.interval.is_some());
                }
            }
        }
        let curves = f.curves.as_ref().unwrap();
        assert_eq!(curves.times, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]);
        assert_eq!(curves.cif["onset"][5], p.cif("onset", 10.0).unwrap());
    }
    eprintln!("abstained {abstained} of {}", sample.len());
    assert!(abstained <= 3, "an in-distribution sample is almost never withheld");
    assert!(model.forecast_history(&histories[0], &ForecastRequest::new([15.5])).is_err(), "beyond the last knot");
    assert_eq!(forecasts[0].model.weights_sha256.len(), 64);

    // Evaluation agrees with the arithmetic done by hand.
    for (i, s) in test.iter_mut().enumerate() {
        s.group_id = Some(format!("household-{}", i / 2));
    }
    let eval = model.evaluate(&test, &EvaluationSpec::new([5.0, 10.0])).unwrap();
    let m = eval.at("death:a", 10.0).expect("death:a by 10 has events");
    let obs = observed(&test, &["death:a", "death:b"]);
    let g = censoring(&obs);
    let preds = model.predict(&test).unwrap();
    let risk: Vec<f64> = preds.iter().map(|p| p.cif("death:a", 10.0).unwrap()).collect();
    assert_eq!(m.uno_c, concordance::uno(&risk, &obs, 0, 10.0, &g));
    assert_eq!(m.auc, auc::at(&risk, &obs, 0, 10.0, &g));
    let true_risk: Vec<f64> = truth.iter().map(|t| t.cif(0, 10.0)).collect();
    let true_auc = auc::at(&true_risk, &obs, 0, 10.0, &g).unwrap();
    eprintln!(
        "death:a by 10: events {}, C {:.3}, AUC {:.3} (truth {:.3}), Brier {:.4}, IBS {:.4}, slope {:.3}, O/E {:.3}, ECE {:.4}, NLL {:.4}",
        m.events, m.uno_c.unwrap(), m.auc.unwrap(), true_auc, m.brier.unwrap(), m.integrated_brier.unwrap(),
        m.calibration.slope.unwrap(), m.calibration.observed_over_expected.unwrap(), m.calibration.ece.unwrap(), eval.event_nll
    );
    assert!((m.auc.unwrap() - true_auc).abs() < 0.03, "AUC {:.3} vs truth {true_auc:.3}", m.auc.unwrap());
    let ci = m.intervals.expect("households are groups").auc.unwrap();
    assert!(ci.lo <= m.auc.unwrap() && m.auc.unwrap() <= ci.hi && ci.hi - ci.lo > 0.0, "{ci:?}");
    assert!(eval.event_nll.is_finite() && !eval.results.is_empty());
}

/// Average ranks (ties share their mean rank).
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
            j += 1;
        }
        let mean = 0.5 * (i + j) as f64 + 1.0;
        order[i..=j].iter().for_each(|&k| r[k] = mean);
        i = j + 1;
    }
    r
}

/// Spearman's rank correlation.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let (va, vb): (f64, f64) = (
        ra.iter().map(|x| (x - ma).powi(2)).sum(),
        rb.iter().map(|y| (y - mb).powi(2)).sum(),
    );
    cov / (va * vb).sqrt()
}

/// On a test population shifted further and further from the training one
/// (older entry ages, higher `x1`), the model's error against the TRUE risk
/// grows, and so does its out-of-distribution score: the score ranks subjects
/// by how wrong the model is about them, and the subjects it flags
/// as unsupported are worse predicted than those it supports, while the
/// population it was trained on is hardly flagged at all.
#[test]
fn the_ood_score_rises_with_the_error_under_covariate_shift() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use brain::timeline::synthetic::{population_shifted, Shift};
    let (train, _) = synthetic::population(6000, 1);
    let (stop, _) = synthetic::population(1000, 2);
    let codes = ["death:a", "death:b", "onset"];
    let spec = TimelineSpec::new(codes, ["death:a", "death:b"])
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(400)
        .batch(128)
        .lr(3e-3);
    let (model, _) = TimelineModel::train(&train, &stop, &spec).unwrap();
    assert!(model.support().is_some(), "training records the support");

    let t = 5.0;
    let (mut score, mut error, mut level) = (vec![], vec![], vec![]);
    for k in 0..4 {
        let shift = Shift { age: 5.0 * k as f64, x1: k as f64 };
        let (subjects, truth) = population_shifted(1000, 50 + k as u64, shift);
        let assessed = model.assess(&subjects).unwrap();
        let preds = model.predict(&subjects).unwrap();
        for ((a, p), tr) in assessed.iter().zip(&preds).zip(&truth) {
            score.push(a.ood_score.unwrap());
            let err: f64 = codes
                .iter()
                .enumerate()
                .map(|(i, c)| (p.cif(c, t).unwrap() - tr.cif(i, t)).abs())
                .sum::<f64>()
                / 3.0;
            error.push(err);
            level.push(k);
        }
    }
    let mean = |v: &[f64], k: usize| {
        let xs: Vec<f64> = v.iter().zip(&level).filter(|(_, &l)| l == k).map(|(x, _)| *x).collect();
        xs.iter().sum::<f64>() / xs.len() as f64
    };
    let within = |k: usize| {
        let pick = |v: &[f64]| -> Vec<f64> {
            v.iter().zip(&level).filter(|(_, &l)| l == k).map(|(x, _)| *x).collect()
        };
        spearman(&pick(&score), &pick(&error))
    };
    for k in 0..4 {
        eprintln!(
            "shift {k}: mean score {:.3}, mean error {:.4}, Spearman within the level {:.3}",
            mean(&score, k),
            mean(&error, k),
            within(k)
        );
    }
    let rho = spearman(&score, &error);
    let (mut flagged, mut kept) = (vec![], vec![]);
    for (s, e) in score.iter().zip(&error) {
        if *s > 1.0 { flagged.push(*e) } else { kept.push(*e) }
    }
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "Spearman(ood_score, |error|) over {} subjects: {rho:.3}; flagged unsupported {} (mean error {:.4}), supported {} (mean error {:.4})",
        score.len(), flagged.len(), avg(&flagged), kept.len(), avg(&kept)
    );
    let in_distribution = score.iter().zip(&level).filter(|(_, &l)| l == 0);
    let wrongly_flagged = in_distribution.clone().filter(|(s, _)| **s > 1.0).count();
    eprintln!("unshifted subjects flagged: {wrongly_flagged} of {}", in_distribution.count());
    assert!(wrongly_flagged < 50, "{wrongly_flagged} of 1000 unshifted subjects flagged");
    assert!(rho > 0.5, "rank correlation of the score with the error: {rho:.3}");
    assert!(
        (0..3).all(|k| mean(&score, k) < mean(&score, k + 1) && mean(&error, k) < mean(&error, k + 1)),
        "both rise with the shift"
    );
    assert!(!flagged.is_empty() && !kept.is_empty());
    assert!(
        avg(&flagged) > 2.0 * avg(&kept),
        "unsupported subjects are predicted worse: {:.4} vs {:.4}",
        avg(&flagged),
        avg(&kept)
    );

    // Abstention: the unsupported subject gets no probability, the supported
    // one the same prediction as `predict`; an operator can accept more.
    use brain::timeline::{Abstain, Event};
    let (fresh, _) = synthetic::population(20, 99);
    let mut odd = fresh[0].clone();
    odd.events.push(Event { t: odd.entry - 1.0, code: "dx:never-seen".into() });
    let batch = [fresh[1].clone(), odd.clone()];
    let guarded = model.predict_or_abstain(&batch, &Abstain::default()).unwrap();
    let plain = model.predict(&batch).unwrap();
    assert_eq!(guarded[0].as_ref().unwrap().cif("onset", t), plain[0].cif("onset", t));
    let refusal = guarded[1].as_ref().unwrap_err();
    assert_eq!(refusal.to_string(), "risk unavailable: insufficient support");
    assert_eq!(refusal.assessment.supported, Some(false));
    assert!(refusal.assessment.warnings.iter().any(|w| matches!(w, brain::timeline::Warning::UnknownEventCode { .. })));
    let tolerant = model.predict_or_abstain(&batch, &Abstain::above(10.0)).unwrap();
    assert!(tolerant[1].is_ok(), "a threshold past the unknown-input score lets it through");

    // The support is saved with the model and read back.
    let dir = std::env::temp_dir().join(format!("brain-timeline-support-{}", std::process::id()));
    model.save(&dir).unwrap();
    let loaded = TimelineModel::load(&dir).unwrap();
    let (a, b) = (model.assess(&batch).unwrap(), loaded.assess(&batch).unwrap());
    assert_eq!(a[1].warnings, b[1].warnings);
    assert!((a[0].ood_score.unwrap() - b[0].ood_score.unwrap()).abs() < 1e-3);
    std::fs::remove_file(dir.join("support.json")).unwrap();
    let old = TimelineModel::load(&dir).unwrap();
    assert!(old.support().is_none());
    assert!(old.assess(&batch).unwrap().iter().all(|a| a.supported.is_none()), "support unknown, not unsupported");
    assert!(old.predict_or_abstain(&batch, &Abstain::default()).unwrap().iter().all(|r| r.is_ok()));
    std::fs::remove_dir_all(&dir).ok();
}
