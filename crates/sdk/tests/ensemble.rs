// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements risk models that say how sure they are, for
// its clients. If your team needs expertise in uncertainty for time-to-event
// predictions you can procure our services by sending an email to
// info@swedishembedded.com.

//! `brain::TimelineEnsemble`: a seeded and a bootstrap ensemble as trained
//! objects. The tests hold the robust properties (a saved ensemble predicts
//! what it did, the member spread rises with the error under covariate
//! shift); `ensemble_comparison` is the measurement of single model against
//! both kinds, run on demand:
//!
//! ```text
//! BRAIN_BACKEND=cuda cargo test --release -p brain --features timeline \
//!     --test ensemble -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "timeline")]

use std::time::Instant;

use brain::survival::calibration::at_horizon;
use brain::survival::estimate::censoring;
use brain::timeline::synthetic::{population, population_shifted, Shift, Truth};
use brain::timeline::{observed, EnsembleKind, Prediction, Subject, TimelineEnsemble, TimelineModel, TimelineSpec};

const CODES: [&str; 3] = ["death:a", "death:b", "onset"];
const ABSORBING: [&str; 2] = ["death:a", "death:b"];
const HORIZON: f64 = 5.0;

fn spec() -> TimelineSpec {
    TimelineSpec::new(CODES, ABSORBING)
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(400)
        .batch(128)
        .lr(3e-3)
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

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Per subject: the mean over codes of |mean risk - true risk| and of the
/// members' spread, both at [`HORIZON`].
fn error_and_spread(predictions: &[Prediction], truth: &[Truth]) -> (Vec<f64>, Vec<f64>) {
    predictions
        .iter()
        .zip(truth)
        .map(|(p, t)| {
            let err = CODES.iter().enumerate().map(|(k, c)| (p.cif(c, HORIZON).unwrap() - t.cif(k, HORIZON)).abs());
            let spread = CODES.iter().map(|c| p.cif_spread(c, HORIZON).unwrap_or(0.0));
            (mean(&err.collect::<Vec<_>>()), mean(&spread.collect::<Vec<_>>()))
        })
        .unzip()
}

fn shifted(k: usize, n: usize) -> (Vec<Subject>, Vec<Truth>) {
    population_shifted(n, 50 + k as u64, Shift { age: 5.0 * k as f64, x1: k as f64 })
}

#[test]
fn a_saved_ensemble_predicts_what_it_did_and_its_spread_rises_with_the_shift() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population(6000, 1);
    let (stop, _) = population(1000, 2);
    let (ensemble, reports) = TimelineEnsemble::train(&train, &stop, &spec(), 4, EnsembleKind::Seeded).unwrap();
    assert_eq!(reports.len(), 4);
    let dir = std::env::temp_dir().join(format!("brain-ensemble-{}", std::process::id()));
    ensemble.save(&dir).unwrap();
    let loaded = TimelineEnsemble::load(&dir).unwrap();

    let mut level_spread = Vec::new();
    let (mut all_error, mut all_spread) = (Vec::new(), Vec::new());
    for k in 0..4 {
        let (subjects, truth) = shifted(k, 1000);
        let predictions = ensemble.predict(&subjects).unwrap();
        let again = loaded.predict(&subjects).unwrap();
        for (a, b) in predictions.iter().zip(&again) {
            for c in CODES {
                assert_eq!(a.cif(c, HORIZON), b.cif(c, HORIZON), "a loaded ensemble predicts what it did");
                assert_eq!(a.member_cifs(c, HORIZON).unwrap().len(), 4);
            }
        }
        let (error, spread) = error_and_spread(&predictions, &truth);
        eprintln!("shift {k}: mean error {:.4}, mean spread {:.4}, Spearman {:.3}", mean(&error), mean(&spread), spearman(&spread, &error));
        level_spread.push(mean(&spread));
        all_error.extend(error);
        all_spread.extend(spread);
    }
    assert!(level_spread.windows(2).all(|w| w[0] < w[1]), "the spread rises with the shift: {level_spread:?}");
    let rho = spearman(&all_spread, &all_error);
    assert!(rho > 0.0, "the spread correlates positively with the error: {rho:.3}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Calibration of the mean risk of `code` at [`HORIZON`]: slope, observed over
/// expected, expected calibration error.
fn calibration_of(predictions: &[Prediction], subjects: &[Subject], code: &str) -> (f64, f64, f64) {
    let mut list = vec![code];
    list.extend(ABSORBING.iter().filter(|c| **c != code));
    let obs = observed(subjects, &list);
    let g = censoring(&obs);
    let risk: Vec<f64> = predictions.iter().map(|p| p.cif(code, HORIZON).unwrap()).collect();
    let c = at_horizon(&risk, &obs, 0, HORIZON, &g, 10);
    (c.slope, c.oe_ratio, c.ece())
}

/// Single model against a seeded and a bootstrap ensemble of five, on the
/// synthetic populations: calibration of the mean risk on the training
/// population, the Spearman correlation between the member spread and the
/// error to the true risk per shift level, and what training cost. The single
/// model is the first member of the seeded ensemble (the same seed).
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn ensemble_comparison() {
    const MEMBERS: usize = 5;
    for data_seed in [1u64, 11] {
        let (train, _) = population(6000, data_seed);
        let (stop, _) = population(1000, data_seed + 1);
        let (test, _) = population(6000, data_seed + 2);
        println!("\n== data seed {data_seed}: {} training, {} early-stopping, {} calibration-test subjects", train.len(), stop.len(), test.len());

        let started = Instant::now();
        let (single, single_report) = TimelineModel::train(&train, &stop, &spec().seed(data_seed)).unwrap();
        let single_time = started.elapsed().as_secs_f64();
        let kinds = [(EnsembleKind::Seeded, "seeded"), (EnsembleKind::Bootstrap, "bootstrap")];
        let mut trained = Vec::new();
        for (kind, name) in kinds {
            let started = Instant::now();
            let (ensemble, reports) =
                TimelineEnsemble::train(&train, &stop, &spec().seed(data_seed), MEMBERS, kind).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let steps: u32 = reports.iter().map(|r| r.steps).sum();
            println!(
                "{name:>9}: training {seconds:.1}s = {:.2}x single ({single_time:.1}s); {steps} steps vs {} single",
                seconds / single_time,
                single_report.steps
            );
            trained.push((name, ensemble));
        }

        println!("calibration of the mean risk at {HORIZON} (slope, observed/expected, ECE):");
        let single_predictions = single.predict(&test).unwrap();
        for code in CODES {
            let (s, oe, ece) = calibration_of(&single_predictions, &test, code);
            println!("  {code:>8} single    : slope {s:.3}  O/E {oe:.3}  ECE {ece:.4}");
            for (name, ensemble) in &trained {
                let (s, oe, ece) = calibration_of(&ensemble.predict(&test).unwrap(), &test, code);
                println!("  {code:>8} {name:<9} : slope {s:.3}  O/E {oe:.3}  ECE {ece:.4}");
            }
        }

        println!("spread vs error to the true risk (Spearman; mean error of the mean risk) per shift level:");
        for k in 0..4 {
            let (subjects, truth) = shifted(k, 2000);
            let (single_error, _) = error_and_spread(&single.predict(&subjects).unwrap(), &truth);
            let mut line = format!("  shift {k}: single error {:.4}", mean(&single_error));
            for (name, ensemble) in &trained {
                let (error, spread) = error_and_spread(&ensemble.predict(&subjects).unwrap(), &truth);
                line += &format!(" | {name}: error {:.4} spread {:.4} rho {:.3}", mean(&error), mean(&spread), spearman(&spread, &error));
            }
            println!("{line}");
        }
    }
}
