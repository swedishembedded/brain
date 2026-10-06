// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements risk models whose probabilities hold up
// against outcomes observed years later, for its clients. If your team needs
// expertise in measuring and repairing the calibration of survival
// predictions you can procure our services by sending an email to
// info@swedishembedded.com.

//! A measurement, not a gate: how well each way of calibrating a cumulative
//! incidence recovers the TRUTH, on a synthetic population whose true risk is
//! known per subject. Both tests are `#[ignore]`d; run them with
//! `--ignored --nocapture` and read the tab-separated rows on stdout.
//!
//! - `calibrators_on_a_trained_model` trains a small model per seed (it needs
//!   the device: run it under the shared device lock), calibrates it on
//!   validation populations of growing size and scores raw and calibrated
//!   risk on a large independent test population against the censored
//!   outcomes (the lens a release gate uses) and against the true risk.
//! - `calibrators_on_a_known_distortion` needs no model: the "model" is the
//!   true risk, undistorted and with a known monotone distortion, so the
//!   right answer is known exactly (calibration must leave the first alone
//!   and undo the second).
//!
//! A validation size is measured on every disjoint slice of a pool of
//! `POOL` independent subjects, so the spread over replications (the noise of
//! the calibration set) is part of the measurement.
//!
//! Environment: `CAL_SEEDS` (comma list, default `1,2,3`), `CAL_SIZES`
//! (default `840,1500,3000,6000,12000`), `CAL_TEST` (test subjects, default
//! 40000).

use horizon::calibration::observed;
use horizon::fit::{train, Hooks, TrainSpec};
use horizon::synthetic::{population, Truth, ABSORBING, CODES};
use horizon::timeline::Subject;
use survival::brier::brier;
use survival::calibration::at_horizon;
use survival::recalibration::{Recalibration, Slope};
use survival::estimate::{censoring, Step};
use survival::venn_abers::{merged, VennAbers};
use survival::Obs;

/// Independent subjects the validation sets are cut from.
const POOL: usize = 24_000;
const HORIZONS: [f64; 3] = [1.0, 5.0, 10.0];
const METHODS: [&str; 9] = [
    "raw", "venn_abers", "va_midpoint", "iso_step", "iso_linear", "logistic", "intercept", "evidence2", "evidence3",
];

fn env_list<T: std::str::FromStr>(name: &str, default: &str) -> Vec<T> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|x| x.trim().parse().unwrap_or_else(|_| panic!("{name}: bad value {x}")))
        .collect()
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    (p / (1.0 - p)).ln()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The competing list of code `k`, as `Calibration::fit` builds it.
fn competing(k: usize) -> Vec<&'static str> {
    let mut list = vec![CODES[k]];
    list.extend((0..CODES.len()).filter(|&j| j != k && ABSORBING[j]).map(|j| CODES[j]));
    list
}

/// Weighted pool-adjacent-violators fit of events over weights per point.
fn isotonic(weight: &[f64], events: &[f64]) -> Vec<f64> {
    let mut stack: Vec<(f64, f64, usize)> = Vec::new();
    for i in 0..weight.len() {
        let mut cur = (weight[i], events[i], 1usize);
        while let Some(&(w, e, n)) = stack.last() {
            if e / w < cur.1 / cur.0 {
                break;
            }
            stack.pop();
            cur = (w + cur.0, e + cur.1, n + cur.2);
        }
        stack.push(cur);
    }
    stack.into_iter().flat_map(|(w, e, n)| std::iter::repeat_n(e / w, n)).collect()
}

/// Every calibrator fitted on one calibration set for one (code, horizon).
struct Fitted {
    va: VennAbers,
    mean_weight: f64,
    scores: Vec<f64>,
    fit: Vec<f64>,
    free: Option<Recalibration>,
    fixed: Option<Recalibration>,
    evidence2: Option<Recalibration>,
    evidence3: Option<Recalibration>,
}

impl Fitted {
    fn new(cal_scores: &[f64], obs: &[Obs], t: f64, g: &Step) -> Fitted {
        let va = VennAbers::at_horizon(cal_scores, obs, 0, t, g);
        let s = va.state();
        let mean_weight = va.mean_weight().expect("labelled calibration subjects");
        let fit = |slope| Recalibration::at_horizon(cal_scores, obs, 0, t, g, slope);
        Fitted {
            fit: isotonic(&s.weight, &s.events),
            scores: s.scores,
            va,
            mean_weight,
            free: fit(Slope::Free),
            fixed: fit(Slope::Fixed),
            evidence2: fit(Slope::Evidence(2.0)),
            evidence3: fit(Slope::Evidence(3.0)),
        }
    }

    fn apply(&self, method: &str, f: f64) -> f64 {
        match method {
            "raw" => f,
            "venn_abers" => merged(self.va.interval(f, self.mean_weight)),
            "va_midpoint" => {
                let (p0, p1) = self.va.interval(f, self.mean_weight);
                0.5 * (p0 + p1)
            }
            "iso_step" => {
                let i = self.scores.partition_point(|&s| s <= f);
                self.fit[i.saturating_sub(1)]
            }
            "iso_linear" => {
                let i = self.scores.partition_point(|&s| s <= f);
                if i == 0 {
                    self.fit[0]
                } else if i == self.scores.len() {
                    self.fit[i - 1]
                } else {
                    let (x0, x1) = (self.scores[i - 1], self.scores[i]);
                    let w = (f - x0) / (x1 - x0);
                    self.fit[i - 1] * (1.0 - w) + self.fit[i] * w
                }
            }
            "logistic" => self.free.map_or(f, |r| r.apply(f)),
            "intercept" => self.fixed.map_or(f, |r| r.apply(f)),
            "evidence2" => self.evidence2.map_or(f, |r| r.apply(f)),
            "evidence3" => self.evidence3.map_or(f, |r| r.apply(f)),
            _ => f,
        }
    }
}

/// Apply `method` to every test score, split over threads (Venn-Abers refits
/// an isotonic regression per score).
fn apply_all(fitted: &Fitted, method: &str, scores: &[f64]) -> Vec<f64> {
    if method == "raw" {
        return scores.to_vec();
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
    let chunk = scores.len().div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        let parts: Vec<_> = scores
            .chunks(chunk)
            .map(|c| sc.spawn(move || c.iter().map(|&f| fitted.apply(method, f)).collect::<Vec<_>>()))
            .collect();
        parts.into_iter().flat_map(|p| p.join().unwrap()).collect()
    })
}

/// One test population: the censored outcomes of each code and the truth.
struct Test {
    obs: Vec<Vec<Obs>>,
    g: Vec<Step>,
    truth: Vec<Vec<Vec<f64>>>, // [code][horizon][subject]
}

impl Test {
    fn new(subjects: &[Subject], truths: &[Truth]) -> Test {
        let obs: Vec<Vec<Obs>> = (0..CODES.len()).map(|k| observed(subjects, &competing(k))).collect();
        let g = obs.iter().map(|o| censoring(o)).collect();
        let truth = (0..CODES.len())
            .map(|k| HORIZONS.iter().map(|&t| truths.iter().map(|tr| tr.cif(k, t)).collect()).collect())
            .collect();
        Test { obs, g, truth }
    }
}

/// One row per method for (code, horizon): the lens a release gate reads and
/// the distance to the truth.
fn report(tag: &str, size: usize, k: usize, hi: usize, events: usize, cal: &Fitted, test: &Test, test_scores: &[f64]) {
    let t = HORIZONS[hi];
    let truth = &test.truth[k][hi];
    let mean_truth = truth.iter().sum::<f64>() / truth.len() as f64;
    for method in METHODS.iter() {
        let pred = apply_all(cal, method, test_scores);
        let h = at_horizon(&pred, &test.obs[k], 0, t, &test.g[k], 10);
        let n = pred.len() as f64;
        let mae = pred.iter().zip(truth).map(|(p, q)| (p - q).abs()).sum::<f64>() / n;
        let rmse = (pred.iter().zip(truth).map(|(p, q)| (p - q).powi(2)).sum::<f64>() / n).sqrt();
        let bias = pred.iter().sum::<f64>() / n - mean_truth;
        let bs = brier(&pred, &test.obs[k], 0, t, &test.g[k]).unwrap_or(f64::NAN);
        println!(
            "ROW\t{tag}\t{size}\t{}\t{t}\t{events}\t{method}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.5}\t{:.5}\t{:+.5}\t{:.5}",
            CODES[k], h.slope, h.intercept, h.oe_ratio, h.ece(), mae, rmse, bias, bs
        );
    }
}

fn events_by(obs: &[Obs], t: f64) -> usize {
    obs.iter().filter(|o| o.time <= t && o.cause == Some(0)).count()
}

#[test]
#[ignore = "a measurement: trains models on the device; run under the shared device lock"]
fn calibrators_on_a_trained_model() {
    let seeds: Vec<u64> = env_list("CAL_SEEDS", "1,2,3");
    let sizes: Vec<usize> = env_list("CAL_SIZES", "840,1500,3000,6000,12000");
    let n_test: usize = env_list("CAL_TEST", "40000")[0];
    println!("ROW\ttag\tsize\tcode\thorizon\tevents\tmethod\tslope\tintercept\toe\tece\tmae_truth\trmse_truth\tbias_truth\tbrier");
    for seed in seeds {
        let (fit_set, _) = population(8400, 100 + seed);
        let (stop_set, _) = population(1700, 200 + seed);
        let (pool, _) = population(POOL, 300 + seed);
        let (test_subjects, test_truths) = population(n_test, 400 + seed);
        let spec = TrainSpec::new(CODES, CODES[..2].iter().copied())
            .knots(vec![0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
            .max_tokens(16)
            .batch(256)
            .steps(1500)
            .seed(seed);
        let (saved, rep) = train(&fit_set, &stop_set, &spec, &mut Hooks::default()).unwrap();
        println!("# seed {seed}: {} steps, held-out NLL {}", rep.steps, rep.held_out_event_nll);
        let test = Test::new(&test_subjects, &test_truths);
        let test_curves = saved.predict(&test_subjects).unwrap();
        let pool_curves = saved.predict(&pool).unwrap();
        let stop_curves = saved.predict(&stop_set[..1500]).unwrap();
        // The calibration sets: nested prefixes of an independent pool, and
        // the early-stopping subjects themselves (what brain forbids).
        let mut sets: Vec<(String, usize, &[Subject], &[horizon::survival::Curves])> = Vec::new();
        for &n in &sizes {
            for rep in 0..POOL / n {
                let r = rep * n..(rep + 1) * n;
                sets.push((format!("indep\tseed{seed}.{rep}"), n, &pool[r.clone()], &pool_curves[r]));
            }
        }
        sets.push((format!("overlap_with_stopping\tseed{seed}.0"), 1500, &stop_set[..1500], &stop_curves[..]));
        for (tag, size, subjects, curves) in sets {
            for k in 0..CODES.len() {
                let obs = observed(subjects, &competing(k));
                let g = censoring(&obs);
                for (hi, &t) in HORIZONS.iter().enumerate() {
                    let cal_scores: Vec<f64> = curves.iter().map(|c| c.cif(k, t)).collect();
                    let test_scores: Vec<f64> = test_curves.iter().map(|c| c.cif(k, t)).collect();
                    let cal = Fitted::new(&cal_scores, &obs, t, &g);
                    report(&tag, size, k, hi, events_by(&obs, t), &cal, &test, &test_scores);
                }
            }
        }
    }
}

/// A distortion of the true risk, in log-odds: `a + b * logit(p)`.
fn distorted(p: f64, a: f64, b: f64) -> f64 {
    sigmoid(a + b * logit(p))
}

#[test]
#[ignore = "a measurement; CPU only"]
fn calibrators_on_a_known_distortion() {
    let seeds: Vec<u64> = env_list("CAL_SEEDS", "1,2,3");
    let sizes: Vec<usize> = env_list("CAL_SIZES", "840,1500,3000,6000,12000");
    let n_test: usize = env_list("CAL_TEST", "40000")[0];
    println!("ROW\ttag\tsize\tcode\thorizon\tevents\tmethod\tslope\tintercept\toe\tece\tmae_truth\trmse_truth\tbias_truth\tbrier");
    // (name, intercept, slope) of the distortion of the truth.
    let cases = [("truth", 0.0, 1.0), ("overconfident", 0.0, 1.6), ("shifted_flat", -0.6, 0.6)];
    for seed in seeds {
        let (pool, pool_truths) = population(POOL, 300 + seed);
        let (test_subjects, test_truths) = population(n_test, 400 + seed);
        let test = Test::new(&test_subjects, &test_truths);
        for (name, a, b) in cases {
            for k in 0..CODES.len() {
                for (hi, &t) in HORIZONS.iter().enumerate() {
                    let test_scores: Vec<f64> = test.truth[k][hi].iter().map(|&p| distorted(p, a, b)).collect();
                    let pool_scores: Vec<f64> = pool_truths.iter().map(|tr| distorted(tr.cif(k, t), a, b)).collect();
                    for &size in &sizes {
                        for rep in 0..POOL / size {
                            let r = rep * size..(rep + 1) * size;
                            let obs = observed(&pool[r.clone()], &competing(k));
                            let g = censoring(&obs);
                            let cal = Fitted::new(&pool_scores[r], &obs, t, &g);
                            report(&format!("{name}\tseed{seed}.{rep}"), size, k, hi, events_by(&obs, t), &cal, &test, &test_scores);
                        }
                    }
                }
            }
        }
    }
}
