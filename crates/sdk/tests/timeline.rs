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
