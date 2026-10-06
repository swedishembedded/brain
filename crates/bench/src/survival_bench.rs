// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Four synthetic **survival** benchmarks with known ground truth, as
//! first-class [`Benchmark`]s on the `survival` capability axis.
//!
//! Swedish Embedded AB implements validation of survival and longitudinal
//! models against data whose true risks are known, for its clients. If your
//! team needs expertise in checking that a risk model recovers what the data
//! actually contains, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! | benchmark | what it isolates | baselines the score is taken over |
//! |---|---|---|
//! | `survival_single` | covariates -> one absorbing event under censoring | the covariate-blind Kaplan-Meier |
//! | `survival_competing` | two causes with different covariates and time shapes | the covariate-blind Aalen-Johansen |
//! | `survival_irregular` | the visit TIMES carry the information, the values are noise | the covariate-blind estimate and a visit-blind model |
//! | `survival_longitudinal` | a hidden state inferred from noisy partial measurements, an action, forecasts | the covariate-blind estimate and a baseline-covariates-only model |
//!
//! The generators live in `horizon::synthetic` ([`single`], [`competing`],
//! [`irregular`], [`longitudinal`]) and each returns the truth a score needs.
//! Each benchmark trains a SMALL `horizon` model (a few thousand subjects, a
//! few thousand steps, `d_model` 24-32) on the device `BRAIN_BACKEND` selects.
//!
//! Like [`crate::forecast_bench`] the objective is **not** a causal next-token
//! decoder, so these benchmarks IGNORE the supplied `lm`: the score is how
//! much of the available structure the horizon model recovers, not a property
//! of a decoder architecture. They are therefore
//! [`informational`](Benchmark::informational): reported, never gating the
//! architecture suite. The pass/fail checks are `tests/survival_bench.rs`.
//!
//! ## Headline score
//! A 0..1 **skill**: the share of the achievable integrated Brier improvement
//! the trained model captures, over the strongest ablation baseline,
//!
//! ```text
//! skill = clamp((IBS_base - IBS_model) / (IBS_base - IBS_oracle), 0, 1)
//! ```
//!
//! with `IBS_base` the lowest integrated Brier score among the benchmark's
//! baselines (table above), `IBS_oracle` that of the generator's true (or, for
//! the benchmarks whose inputs are noisy, best-possible) cumulative incidence,
//! all on the same held-out subjects, with the censoring distribution
//! estimated on the training subjects. 0 means no better than the strongest
//! baseline, 1 means as good as the oracle.
//!
//! Extra fields: `ibs`, `ibs_base`, `ibs_oracle`; Harrell's and Uno's
//! concordance and the time-dependent AUC at the horizon (`harrell`, `uno`,
//! `auc`, with the oracle's concordance as `harrell_oracle`); calibration
//! observed-over-expected at the horizon (`oe`); the rank correlation of the
//! predicted risk with the oracle risk (`spearman_oracle`); and, where the
//! benchmark is built on a comparison, the held-out event NLLs it compares.

use std::io;
use std::path::Path;

use horizon::encode::{encode, forecast_query, Encoded};
use horizon::survival::Curves;
use horizon::synthetic::{competing, irregular, longitudinal, single};
use horizon::timeline::Subject;
use horizon::train::{event_nll, predict_forecasts, predict_log_hazards, TimelineObjective};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use serde::{de::DeserializeOwned, Serialize};
use survival::estimate::{aalen_johansen, censoring, Step};
use survival::Obs;

use crate::{Benchmark, DecoderLm, Metrics};

// ---------------------------------------------------------------- data on disk

const TRAIN: &str = "train.jsonl";
const HELD_OUT: &str = "held_out.jsonl";
const TEST: &str = "test.jsonl";
const TRUTH: &str = "truth.json";

/// Subjects for fitting, for early stopping, and for scoring (with its truth).
struct Dataset<T> {
    train: Vec<Subject>,
    held_out: Vec<Subject>,
    test: Vec<Subject>,
    truth: Vec<T>,
}

fn write_jsonl(path: &Path, subjects: &[Subject]) -> io::Result<()> {
    let mut text = String::new();
    for s in subjects {
        text.push_str(&serde_json::to_string(s).map_err(io::Error::other)?);
        text.push('\n');
    }
    std::fs::write(path, text)
}

fn read_jsonl(path: &Path) -> io::Result<Vec<Subject>> {
    std::fs::read_to_string(path)?
        .lines()
        .map(|l| Subject::from_json_line(l).map_err(io::Error::other))
        .collect()
}

/// Write the `timeline-v1` splits and the test subjects' truth under `dir`.
/// The three splits come from independent seeds derived from `seed`.
fn write_dataset<T: Serialize>(
    dir: &Path,
    sizes: (usize, usize, usize),
    seed: u64,
    population: impl Fn(usize, u64) -> (Vec<Subject>, Vec<T>),
) -> io::Result<()> {
    let split = |n: usize, k: u64| population(n, seed ^ (k << 40));
    std::fs::create_dir_all(dir)?;
    write_jsonl(&dir.join(TRAIN), &split(sizes.0, 1).0)?;
    write_jsonl(&dir.join(HELD_OUT), &split(sizes.1, 2).0)?;
    let (test, truth) = split(sizes.2, 3);
    write_jsonl(&dir.join(TEST), &test)?;
    std::fs::write(dir.join(TRUTH), serde_json::to_vec(&truth).map_err(io::Error::other)?)
}

fn read_dataset<T: DeserializeOwned>(dir: &Path) -> io::Result<Dataset<T>> {
    Ok(Dataset {
        train: read_jsonl(&dir.join(TRAIN))?,
        held_out: read_jsonl(&dir.join(HELD_OUT))?,
        test: read_jsonl(&dir.join(TEST))?,
        truth: serde_json::from_slice(&std::fs::read(dir.join(TRUTH))?).map_err(io::Error::other)?,
    })
}

impl<T> Dataset<T> {
    /// The same dataset with every split passed through `f` (an ablation as a
    /// pure data transform); outcomes and truth are untouched.
    fn map(&self, f: impl Fn(&[Subject]) -> Vec<Subject>) -> Dataset<T>
    where
        T: Clone,
    {
        Dataset {
            train: f(&self.train),
            held_out: f(&self.held_out),
            test: f(&self.test),
            truth: self.truth.clone(),
        }
    }
}

// ----------------------------------------------------------------- the model

/// What differs between the benchmarks' small horizon models.
#[derive(Clone)]
struct ModelSpec {
    codes: Vec<String>,
    absorbing: Vec<bool>,
    /// Hazard piece boundaries, in time since entry.
    knots: Vec<f32>,
    max_tokens: u32,
    d_model: u32,
    /// Forecast queries per subject (0: no forecast head).
    forecasts: u32,
    /// Visits carried by the continuous-time state (0: one set).
    visits: u32,
    steps: u32,
}

/// A trained model with the vocabulary and configuration it was trained under.
struct Fitted {
    model: Horizon,
    vocab: Vocab,
    cfg: HorizonConfig,
    absorbing: Vec<bool>,
}

impl Fitted {
    /// Train on `train` with early stopping on the held-out event NLL (the
    /// procedure a real run uses), keeping the best evaluation's weights.
    fn fit(spec: &ModelSpec, train: &[Subject], held_out: &[Subject], seed: u64) -> io::Result<Fitted> {
        let absorbing_codes: Vec<String> = spec
            .codes
            .iter()
            .zip(&spec.absorbing)
            .filter(|(_, a)| **a)
            .map(|(c, _)| c.clone())
            .collect();
        let vocab = Vocab::fit(train, &spec.codes, &absorbing_codes, &FitOptions::default())
            .map_err(io::Error::other)?;
        let mut cfg = HorizonConfig::default_for(vocab.len(), spec.codes.len() as u32);
        cfg.max_tokens = spec.max_tokens;
        cfg.d_model = spec.d_model;
        cfg.n_layers = 1;
        cfg.n_heads = 4;
        cfg.d_ff = 2 * spec.d_model;
        cfg.rank = 8;
        cfg.knots = spec.knots.clone();
        cfg.forecasts = spec.forecasts;
        cfg.forecast_weight = if spec.forecasts > 0 { 1.0 } else { 0.0 };
        cfg.visits = spec.visits;
        let enc = |s: &[Subject]| -> Vec<Encoded> { s.iter().map(|x| encode(x, &vocab, &cfg)).collect() };
        let (enc_train, enc_held) = (enc(train), enc(held_out));
        let batch = 256;
        let model = Horizon::new(cfg.clone(), batch, &horizon::init_weights(&cfg, seed));
        let opts = ::model::FitOpts {
            steps: spec.steps,
            batch_size: batch,
            lr: 3e-3,
            min_lr: 3e-4,
            warmup: 50,
            decay_iters: spec.steps,
            weight_decay: 0.1,
            eval_interval: 100,
            patience: 5,
            checkpoint_secs: 0,
            seed,
            ..Default::default()
        };
        let objective = TimelineObjective::new(&enc_train, Some(&enc_held), 0.3);
        let (_, model) = ::model::fit_controlled(model, objective, &opts, None, ::model::FitControl::default())?;
        Ok(Fitted { model, vocab, cfg, absorbing: spec.absorbing.clone() })
    }

    fn encode(&self, subjects: &[Subject]) -> Vec<Encoded> {
        subjects.iter().map(|s| encode(s, &self.vocab, &self.cfg)).collect()
    }

    /// Each subject's cause-specific curves.
    fn curves(&self, subjects: &[Subject]) -> Vec<Curves> {
        predict_log_hazards(&self.model, &self.encode(subjects))
            .iter()
            .map(|lh| Curves::new(lh, &self.cfg.knots, &self.absorbing))
            .collect()
    }

    /// The weighted mean event NLL the training loop early-stops on.
    fn nll(&self, subjects: &[Subject]) -> f32 {
        event_nll(&self.model, &self.encode(subjects))
    }
}

// ------------------------------------------------------------------- scoring

/// Cumulative incidence tabulated as `[cause][grid point][subject]`: the
/// model, the oracle and the baselines are each computed once per grid point
/// (an oracle may be an expensive integral) and then only compared.
struct Table(Vec<Vec<Vec<f64>>>);

impl Table {
    fn new(grid: &Grid, causes: usize, n: usize, cif: impl Fn(usize, usize, f64) -> f64) -> Table {
        Table(
            (0..causes)
                .map(|c| grid.times.iter().map(|&t| (0..n).map(|i| cif(i, c, t)).collect()).collect())
                .collect(),
        )
    }

    /// The predictions at the horizon (the grid's last point).
    fn at_tau(&self, cause: usize) -> &[f64] {
        self.0[cause].last().expect("a non-empty grid")
    }
}

/// The observed outcomes of `subjects` in time since entry, cause indexed by
/// the position of its code in `codes`.
fn observations(subjects: &[Subject], codes: &[&str]) -> Vec<Obs> {
    horizon::synthetic::sim::outcomes(subjects, codes)
        .into_iter()
        .map(|(time, cause)| Obs { time, cause, weight: 1.0 })
        .collect()
}

/// Where and how a benchmark scores: the horizon `tau` (the grid's last point)
/// and the grid the Brier score is integrated over.
#[derive(Clone)]
struct Grid {
    tau: f64,
    times: Vec<f64>,
}

impl Grid {
    fn new(tau: f64, points: usize) -> Grid {
        Grid { tau, times: (1..=points).map(|k| tau * k as f64 / points as f64).collect() }
    }
}

/// One predictor's scores, averaged over the causes.
#[derive(Default, Clone, Copy)]
struct Scores {
    ibs: f64,
    harrell: f64,
    uno: f64,
    auc: f64,
    oe: f64,
}

/// Score a tabulated predictor on the test outcomes `obs`, with the
/// censoring distribution `g` of the training subjects.
fn score(table: &Table, obs: &[Obs], g: &Step, grid: &Grid) -> Scores {
    let causes = table.0.len();
    let mut total = Scores::default();
    for cause in 0..causes {
        let risk = table.at_tau(cause);
        let ibs = survival::brier::integrated_brier(&grid.times, |k| table.0[cause][k].clone(), obs, cause, g);
        total.ibs += ibs.unwrap_or(f64::NAN);
        total.harrell += survival::concordance::harrell(risk, obs, cause).unwrap_or(f64::NAN);
        total.uno += survival::concordance::uno(risk, obs, cause, grid.tau, g).unwrap_or(f64::NAN);
        total.auc += survival::auc::at(risk, obs, cause, grid.tau, g).unwrap_or(f64::NAN);
        total.oe += survival::calibration::at_horizon(risk, obs, cause, grid.tau, g, 5).oe_ratio;
    }
    let k = causes as f64;
    Scores { ibs: total.ibs / k, harrell: total.harrell / k, uno: total.uno / k, auc: total.auc / k, oe: total.oe / k }
}

/// Spearman rank correlation (average ranks for ties).
fn spearman(x: &[f64], y: &[f64]) -> f64 {
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
            for &k in &order[i..=j] {
                r[k] = 0.5 * (i + j) as f64;
            }
            i = j + 1;
        }
        r
    }
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let var = |r: &[f64], m: f64| r.iter().map(|a| (a - m).powi(2)).sum::<f64>();
    cov / (var(&rx, mx) * var(&ry, my)).sqrt()
}

/// `clamp((base - model) / (base - oracle), 0, 1)`; 0 when the oracle gains
/// nothing over the baseline (no skill is available to measure).
fn skill(base: f64, model: f64, oracle: f64) -> f32 {
    let room = base - oracle;
    if room > 1e-12 && model.is_finite() {
        ((base - model) / room).clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// The mean over causes and subjects of `f(prediction, oracle)` at the horizon.
fn mean_at_tau(a: &Table, b: &Table, f: impl Fn(f64, f64) -> f64) -> f64 {
    let causes = a.0.len();
    let n = a.at_tau(0).len();
    (0..causes)
        .map(|c| a.at_tau(c).iter().zip(b.at_tau(c)).map(|(x, y)| f(*x, *y)).sum::<f64>())
        .sum::<f64>()
        / (causes * n) as f64
}

/// The scored test set: what every survival benchmark computes once it has a
/// model's curves and an oracle - the model, the oracle and the covariate-
/// blind baseline on the same held-out outcomes. Further baselines (a model
/// trained on ablated data) are added by the caller.
struct Evaluation<'a> {
    test: &'a [Subject],
    train: &'a [Subject],
    codes: &'static [&'static str],
    grid: Grid,
}

impl<'a> Evaluation<'a> {
    /// The same scoring on other subjects (an ablated copy of the data).
    fn on<'b>(&self, test: &'b [Subject], train: &'b [Subject]) -> Evaluation<'b> {
        Evaluation { test, train, codes: self.codes, grid: self.grid.clone() }
    }

    fn tabulate(&self, cif: impl Fn(usize, usize, f64) -> f64) -> Table {
        Table::new(&self.grid, self.codes.len(), self.test.len(), cif)
    }

    /// Tabulate each subject's model curves.
    fn model_table(&self, curves: &[Curves]) -> Table {
        self.tabulate(|i, c, t| curves[i].cif(c, t))
    }

    /// Score `model` against `oracle`; `extra` are baselines' integrated Brier
    /// scores (the covariate-blind one is always included).
    fn compare(&self, model: &Table, oracle: &Table, extra: &[(&'static str, f64)]) -> Metrics {
        let obs = observations(self.test, self.codes);
        let train_obs = observations(self.train, self.codes);
        let g = censoring(&train_obs);
        let marginal: Vec<Step> = (0..self.codes.len()).map(|c| aalen_johansen(&train_obs, c)).collect();
        let null = self.tabulate(|_, c, t| marginal[c].at(t));
        let (m, o, n) = (
            score(model, &obs, &g, &self.grid),
            score(oracle, &obs, &g, &self.grid),
            score(&null, &obs, &g, &self.grid),
        );
        let mut baselines = vec![("null", n.ibs)];
        baselines.extend_from_slice(extra);
        let base = baselines.iter().map(|b| b.1).fold(f64::INFINITY, f64::min);
        let rho = (0..self.codes.len())
            .map(|c| spearman(model.at_tau(c), oracle.at_tau(c)))
            .sum::<f64>()
            / self.codes.len() as f64;
        let mut metrics = Metrics::new(skill(base, m.ibs, o.ibs))
            .with("ibs", m.ibs as f32)
            .with("ibs_base", base as f32)
            .with("ibs_oracle", o.ibs as f32)
            .with("harrell", m.harrell as f32)
            .with("harrell_oracle", o.harrell as f32)
            .with("uno", m.uno as f32)
            .with("auc", m.auc as f32)
            .with("auc_oracle", o.auc as f32)
            .with("oe", m.oe as f32)
            .with("spearman_oracle", rho as f32)
            .with("cif_mae", mean_at_tau(model, oracle, |a, b| (a - b).abs()) as f32)
            .with("cif_mae_null", mean_at_tau(&null, oracle, |a, b| (a - b).abs()) as f32)
            .with("cif_bias", mean_at_tau(model, oracle, |a, b| a - b) as f32)
            .with("cif_mean", mean_at_tau(oracle, oracle, |_, b| b) as f32);
        for (name, ibs) in &baselines {
            metrics = metrics.with(&format!("ibs_{name}"), *ibs as f32);
        }
        metrics
    }

    /// The integrated Brier score of a tabulated predictor on the test set.
    fn ibs(&self, table: &Table) -> f64 {
        let obs = observations(self.test, self.codes);
        let g = censoring(&observations(self.train, self.codes));
        score(table, &obs, &g, &self.grid).ibs
    }
}

/// Split sizes of the smoke variants: enough subjects for every estimator to
/// be defined, far too few for the scores to mean anything.
const SMOKE_SIZES: (usize, usize, usize) = (500, 150, 300);

const REPORT: [&str; 6] = ["ibs", "ibs_oracle", "harrell", "harrell_oracle", "spearman_oracle", "oe"];

// --------------------------------------------------------------- the benchmarks

/// `survival_single`: covariates -> one absorbing event under uniform censoring.
///
/// Headline: skill over the covariate-blind Kaplan-Meier (see the module
/// docs); the oracle is the true closed-form cumulative incidence.
pub struct SurvivalSingle {
    /// Training / early-stopping / scoring subjects.
    pub sizes: (usize, usize, usize),
    /// Training steps (early stopping may end sooner).
    pub steps: u32,
}

impl SurvivalSingle {
    /// A slashed variant for the smoke registry and capability sweeps: the
    /// scores are not meaningful as model quality.
    pub fn smoke(steps: u32) -> Self {
        SurvivalSingle { sizes: SMOKE_SIZES, steps }
    }
}

impl Default for SurvivalSingle {
    fn default() -> Self {
        SurvivalSingle { sizes: (8_000, 2_000, 3_000), steps: 1_500 }
    }
}

impl Benchmark for SurvivalSingle {
    fn name(&self) -> &str {
        "survival_single"
    }

    fn description(&self) -> &str {
        "single absorbing event, log-linear hazard, uniform censoring [survival]"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> io::Result<()> {
        write_dataset(dir, self.sizes, seed, single::population)
    }

    /// Non-LM objective: `lm` is ignored (see the module docs).
    fn evaluate_with(&self, _lm: &dyn DecoderLm, dir: &Path, seed: u64) -> io::Result<Metrics> {
        let data: Dataset<single::Truth> = read_dataset(dir)?;
        let spec = ModelSpec {
            codes: vec![single::CODE.into()],
            absorbing: vec![true],
            knots: vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
            max_tokens: 8,
            d_model: 24,
            forecasts: 0,
            visits: 0,
            steps: self.steps,
        };
        let fitted = Fitted::fit(&spec, &data.train, &data.held_out, seed)?;
        let eval = Evaluation { test: &data.test, train: &data.train, codes: &[single::CODE], grid: Grid::new(4.0, 8) };
        let model = eval.model_table(&fitted.curves(&data.test));
        let oracle = eval.tabulate(|i, _, t| data.truth[i].cif(t));
        Ok(eval.compare(&model, &oracle, &[]).with("nll", fitted.nll(&data.test)))
    }

    fn threshold(&self) -> f32 {
        // Set from seeded runs on the CUDA backend (seeds 1 to 4, default
        // budget): the headline skill measured 0.916 to 0.937. The bar sits well
        // under that; it is a reference line for an informational
        // benchmark, not a claim about the model.
        0.8
    }

    fn report_fields(&self) -> Vec<&str> {
        REPORT.to_vec()
    }

    fn informational(&self) -> bool {
        true
    }
}

/// `survival_competing`: two absorbing causes with different covariates and
/// time shapes.
///
/// Headline: skill over the covariate-blind Aalen-Johansen estimate; the
/// oracle is the analytic cumulative incidence `int h_k S`. `cif_mae` and
/// `cif_bias` are the model's distance to it at the horizon.
pub struct SurvivalCompeting {
    /// Training / early-stopping / scoring subjects.
    pub sizes: (usize, usize, usize),
    /// Training steps (early stopping may end sooner).
    pub steps: u32,
}

impl SurvivalCompeting {
    /// A slashed variant for the smoke registry and capability sweeps: the
    /// scores are not meaningful as model quality.
    pub fn smoke(steps: u32) -> Self {
        SurvivalCompeting { sizes: SMOKE_SIZES, steps }
    }
}

impl Default for SurvivalCompeting {
    fn default() -> Self {
        SurvivalCompeting { sizes: (10_000, 2_500, 3_000), steps: 2_000 }
    }
}

impl Benchmark for SurvivalCompeting {
    fn name(&self) -> &str {
        "survival_competing"
    }

    fn description(&self) -> &str {
        "two competing absorbing causes with different covariates and time shapes [survival]"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> io::Result<()> {
        write_dataset(dir, self.sizes, seed, competing::population)
    }

    /// Non-LM objective: `lm` is ignored (see the module docs).
    fn evaluate_with(&self, _lm: &dyn DecoderLm, dir: &Path, seed: u64) -> io::Result<Metrics> {
        let data: Dataset<competing::Truth> = read_dataset(dir)?;
        let spec = ModelSpec {
            codes: competing::CODES.iter().map(|c| c.to_string()).collect(),
            absorbing: competing::ABSORBING.to_vec(),
            knots: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
            max_tokens: 8,
            d_model: 24,
            forecasts: 0,
            visits: 0,
            steps: self.steps,
        };
        let fitted = Fitted::fit(&spec, &data.train, &data.held_out, seed)?;
        let eval = Evaluation { test: &data.test, train: &data.train, codes: &competing::CODES, grid: Grid::new(5.0, 10) };
        let model = eval.model_table(&fitted.curves(&data.test));
        let oracle = eval.tabulate(|i, k, t| data.truth[i].cif(k, t));
        Ok(eval.compare(&model, &oracle, &[]).with("nll", fitted.nll(&data.test)))
    }

    fn threshold(&self) -> f32 {
        // Set from seeded runs on the CUDA backend (seeds 1 to 4, default
        // budget): the headline skill measured 0.852 to 0.966. The bar sits well
        // under that; it is a reference line for an informational
        // benchmark, not a claim about the model.
        0.7
    }

    fn report_fields(&self) -> Vec<&str> {
        REPORT.to_vec()
    }

    fn informational(&self) -> bool {
        true
    }
}

/// `survival_irregular`: observation times carry the information; the values
/// are noise.
///
/// Two models are trained: one on the data, one on the same data with every
/// subject's visits replaced by another subject's
/// ([`irregular::swap_visits`], the visit-blind ablation). Headline: skill
/// over the stronger of the covariate-blind estimate and the visit-blind
/// model; the oracle is the posterior given the visit count
/// ([`irregular::Truth::oracle_cif`]). `nll` / `nll_blind` are the two models'
/// held-out event NLL on their own inputs, `nll_gain` their difference.
pub struct SurvivalIrregular {
    /// Training / early-stopping / scoring subjects.
    pub sizes: (usize, usize, usize),
    /// Training steps (early stopping may end sooner).
    pub steps: u32,
    /// Visits carried by the continuous-time state (0: one set).
    pub visits: u32,
}

impl SurvivalIrregular {
    /// A slashed variant for the smoke registry and capability sweeps: the
    /// scores are not meaningful as model quality.
    pub fn smoke(steps: u32) -> Self {
        SurvivalIrregular { sizes: SMOKE_SIZES, steps, visits: 0 }
    }
}

impl Default for SurvivalIrregular {
    fn default() -> Self {
        SurvivalIrregular { sizes: (10_000, 2_500, 3_000), steps: 2_000, visits: 0 }
    }
}

impl Benchmark for SurvivalIrregular {
    fn name(&self) -> &str {
        "survival_irregular"
    }

    fn description(&self) -> &str {
        "informative visit times, uninformative values; ablation: visit-blind model [survival]"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> io::Result<()> {
        write_dataset(dir, self.sizes, seed, irregular::population)
    }

    /// Non-LM objective: `lm` is ignored (see the module docs).
    fn evaluate_with(&self, _lm: &dyn DecoderLm, dir: &Path, seed: u64) -> io::Result<Metrics> {
        let data: Dataset<irregular::Truth> = read_dataset(dir)?;
        let spec = ModelSpec {
            codes: vec![irregular::CODE.into()],
            absorbing: vec![true],
            knots: vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
            max_tokens: 40,
            d_model: 32,
            forecasts: 0,
            visits: self.visits,
            steps: self.steps,
        };
        let eval = Evaluation { test: &data.test, train: &data.train, codes: &[irregular::CODE], grid: Grid::new(4.0, 8) };
        let oracle = eval.tabulate(|i, _, t| data.truth[i].oracle_cif(t));

        let blind_data = data.map(|s| irregular::swap_visits(s, seed));
        let blind = Fitted::fit(&spec, &blind_data.train, &blind_data.held_out, seed)?;
        let blind_eval = eval.on(&blind_data.test, &blind_data.train);
        let blind_ibs = eval.ibs(&blind_eval.model_table(&blind.curves(&blind_data.test)));
        let nll_blind = blind.nll(&blind_data.test);
        drop(blind);

        let aware = Fitted::fit(&spec, &data.train, &data.held_out, seed)?;
        let model = eval.model_table(&aware.curves(&data.test));
        let nll = aware.nll(&data.test);
        Ok(eval
            .compare(&model, &oracle, &[("blind", blind_ibs)])
            .with("nll", nll)
            .with("nll_blind", nll_blind)
            .with("nll_gain", nll_blind - nll))
    }

    fn threshold(&self) -> f32 {
        // Set from seeded runs on the CUDA backend (seeds 1 to 4, default
        // budget): the headline skill measured 0.947 to 0.961. The bar sits well
        // under that; it is a reference line for an informational
        // benchmark, not a claim about the model.
        0.8
    }

    fn report_fields(&self) -> Vec<&str> {
        let mut f = REPORT.to_vec();
        f.push("nll_gain");
        f
    }

    fn informational(&self) -> bool {
        true
    }
}

/// `survival_longitudinal`: a hidden state inferred from noisy partial
/// measurements, an action, forecasts and an event.
///
/// Two models are trained: one on everything (with the forecast head), one on
/// the baseline covariate only ([`longitudinal::baseline_only`]). Headline:
/// skill over the stronger of the covariate-blind estimate and the
/// baseline-covariate model; the oracle is the conjugate posterior of the
/// state ([`longitudinal::Truth::oracle_cif`]). `nll` / `nll_baseline` are
/// the held-out event NLLs, `nll_gain` their difference. Forecasts: the
/// model's median forecast of `m1` and `m2` 1.5 and 3 time units after entry
/// against the measurements actually taken (`forecast_mae`), against the
/// population mean at the same horizon (`forecast_mae_mean`) and the noise
/// floor of knowing the true state (`forecast_mae_truth`); `forecast_skill`
/// is `1 - forecast_mae / forecast_mae_mean`.
pub struct SurvivalLongitudinal {
    /// Training / early-stopping / scoring subjects.
    pub sizes: (usize, usize, usize),
    /// Training steps (early stopping may end sooner).
    pub steps: u32,
}

impl SurvivalLongitudinal {
    /// A slashed variant for the smoke registry and capability sweeps: the
    /// scores are not meaningful as model quality.
    pub fn smoke(steps: u32) -> Self {
        SurvivalLongitudinal { sizes: SMOKE_SIZES, steps }
    }
}

impl Default for SurvivalLongitudinal {
    fn default() -> Self {
        SurvivalLongitudinal { sizes: (10_000, 2_500, 3_000), steps: 2_000 }
    }
}

/// The measured value of `var` at `ahead` after entry, if the subject has one.
fn follow_up(s: &Subject, var: &str, ahead: f64) -> Option<f64> {
    s.observations.iter().find_map(|o| match o.value {
        horizon::timeline::Value::Number(y) if o.var == var && (o.t - s.entry - ahead).abs() < 1e-9 => Some(y),
        _ => None,
    })
}

impl SurvivalLongitudinal {
    /// Forecast accuracy on the follow-up measurements the test subjects
    /// actually have: `(model, population mean, true state)` mean absolute
    /// errors.
    fn forecast_errors(
        fitted: &Fitted,
        data: &Dataset<longitudinal::Truth>,
    ) -> io::Result<(f64, f64, f64)> {
        let enc = fitted.encode(&data.test);
        let asks: Vec<(usize, &str, f64)> = longitudinal::FOLLOW_UP_VISITS
            .iter()
            .flat_map(|&ahead| (0..2).map(move |j| (j, longitudinal::CHANNELS[j], ahead)))
            .collect();
        let queries: Vec<Vec<_>> = enc
            .iter()
            .map(|_| {
                asks.iter()
                    .map(|(_, var, ahead)| forecast_query(&fitted.vocab, &fitted.cfg, var, *ahead).expect("a measured variable"))
                    .collect()
            })
            .collect();
        let pred = predict_forecasts(&fitted.model, &enc, &queries).map_err(io::Error::other)?;
        let (mut model, mut mean, mut truth, mut count) = (0.0, 0.0, 0.0, 0usize);
        for (q, (j, var, ahead)) in asks.iter().enumerate() {
            let train: Vec<f64> = data.train.iter().filter_map(|s| follow_up(s, var, *ahead)).collect();
            let pop = train.iter().sum::<f64>() / train.len() as f64;
            for (i, s) in data.test.iter().enumerate() {
                let Some(y) = follow_up(s, var, *ahead) else { continue };
                let (mu, sigma) = (pred[i][q].0 as f64, pred[i][q].1 as f64);
                let median = fitted.vocab.forecast_quantile(var, mu, sigma, 0.5).expect("a fitted variable");
                model += (median - y).abs();
                mean += (pop - y).abs();
                truth += (data.truth[i].channel_mean(*j, *ahead) - y).abs();
                count += 1;
            }
        }
        let n = count as f64;
        Ok((model / n, mean / n, truth / n))
    }
}

impl Benchmark for SurvivalLongitudinal {
    fn name(&self) -> &str {
        "survival_longitudinal"
    }

    fn description(&self) -> &str {
        "hidden state from noisy partial measurements, action, forecasts and an event [survival]"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> io::Result<()> {
        write_dataset(dir, self.sizes, seed, longitudinal::population)
    }

    /// Non-LM objective: `lm` is ignored (see the module docs).
    fn evaluate_with(&self, _lm: &dyn DecoderLm, dir: &Path, seed: u64) -> io::Result<Metrics> {
        let data: Dataset<longitudinal::Truth> = read_dataset(dir)?;
        let spec = ModelSpec {
            codes: vec![longitudinal::CODE.into()],
            absorbing: vec![true],
            knots: vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
            max_tokens: 16,
            d_model: 32,
            forecasts: 4,
            visits: 0,
            steps: self.steps,
        };
        let eval = Evaluation { test: &data.test, train: &data.train, codes: &[longitudinal::CODE], grid: Grid::new(4.0, 8) };
        let oracle = eval.tabulate(|i, _, t| data.truth[i].oracle_cif(t));

        let baseline_data = data.map(longitudinal::baseline_only);
        let baseline_spec = ModelSpec { forecasts: 0, ..spec.clone() };
        let baseline = Fitted::fit(&baseline_spec, &baseline_data.train, &baseline_data.held_out, seed)?;
        let baseline_eval = eval.on(&baseline_data.test, &baseline_data.train);
        let baseline_ibs = eval.ibs(&baseline_eval.model_table(&baseline.curves(&baseline_data.test)));
        let nll_baseline = baseline.nll(&baseline_data.test);
        drop(baseline);

        let full = Fitted::fit(&spec, &data.train, &data.held_out, seed)?;
        let model = eval.model_table(&full.curves(&data.test));
        let nll = full.nll(&data.test);
        let (forecast, forecast_mean, forecast_truth) = Self::forecast_errors(&full, &data)?;
        Ok(eval
            .compare(&model, &oracle, &[("baseline", baseline_ibs)])
            .with("nll", nll)
            .with("nll_baseline", nll_baseline)
            .with("nll_gain", nll_baseline - nll)
            .with("forecast_mae", forecast as f32)
            .with("forecast_mae_mean", forecast_mean as f32)
            .with("forecast_mae_truth", forecast_truth as f32)
            .with("forecast_skill", (1.0 - forecast / forecast_mean) as f32))
    }

    fn threshold(&self) -> f32 {
        // Set from seeded runs on the CUDA backend (seeds 1 to 4, default
        // budget): the headline skill measured 0.930 to 0.976. The bar sits well
        // under that; it is a reference line for an informational
        // benchmark, not a claim about the model.
        0.8
    }

    fn report_fields(&self) -> Vec<&str> {
        let mut f = REPORT.to_vec();
        f.extend(["nll_gain", "forecast_skill"]);
        f
    }

    fn informational(&self) -> bool {
        true
    }
}

/// The four survival benchmarks at their calibrated budgets.
pub fn survival_benchmarks() -> Vec<Box<dyn Benchmark>> {
    vec![
        Box::new(SurvivalSingle::default()),
        Box::new(SurvivalCompeting::default()),
        Box::new(SurvivalIrregular::default()),
        Box::new(SurvivalLongitudinal::default()),
    ]
}

/// The four survival benchmarks slashed to `steps` training steps on a few
/// hundred subjects, for the smoke registry.
pub fn survival_benchmarks_smoke(steps: u32) -> Vec<Box<dyn Benchmark>> {
    ["survival_single", "survival_competing", "survival_irregular", "survival_longitudinal"]
        .iter()
        .filter_map(|n| build(n, steps))
        .collect()
}

/// One slashed survival benchmark by name (used by capscale to build the
/// survival probe).
pub fn build(name: &str, steps: u32) -> Option<Box<dyn Benchmark>> {
    Some(match name {
        "survival_single" => Box::new(SurvivalSingle::smoke(steps)),
        "survival_competing" => Box::new(SurvivalCompeting::smoke(steps)),
        "survival_irregular" => Box::new(SurvivalIrregular::smoke(steps)),
        "survival_longitudinal" => Box::new(SurvivalLongitudinal::smoke(steps)),
        _ => return None,
    })
}
