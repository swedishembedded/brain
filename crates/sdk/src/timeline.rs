// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::TimelineModel` - risk over time from irregular records.
//!
//! A subject's history (measurements with values, events, each at a
//! real-valued time; [`Subject`], the `timeline-v1` format) goes in; for any
//! horizon inside the model's knots, the probability of each outcome code by
//! then comes out ([`Prediction`]), with competing absorbing outcomes and
//! delayed entry handled exactly. The model is `crates/horizon`; the
//! evaluation arithmetic a caller judges it with is [`crate::survival`].
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! use brain::timeline::{read_jsonl, TimelineModel, TimelineSpec};
//! let train = read_jsonl("train.jsonl")?;
//! let held_out = read_jsonl("held_out.jsonl")?;
//! let spec = TimelineSpec::new(["death:heart", "death:other"], ["death:heart", "death:other"]);
//! let (model, report) = TimelineModel::train(&train, &held_out, &spec)?;
//! let risk = model.predict(&held_out)?[0].cif("death:heart", 10.0);
//! model.save("model")?;
//! # let _ = (risk, report); Ok(()) }
//! ```

use std::path::Path;

use horizon::encode::{encode, forecast_query, Encoded};
use horizon::saved::{parse_jsonl, Saved};
use horizon::survival::Curves;
use horizon::train::{
    event_nll, predict_forecasts, predict_states, TimelineObjective,
};
use horizon::vocab::{FitOptions, Vocab};
use horizon::Horizon;

/// A synthetic population with KNOWN hazards, to check a pipeline against the
/// truth before trusting it on real data.
pub use horizon::synthetic;
pub use horizon::timeline::{AtRisk, Event, Observation, Subject, Value};
pub use horizon::HorizonConfig as TimelineConfig;

use crate::{Error, Result};

/// Subjects per batch when training and predicting, unless the spec says otherwise.
pub const DEFAULT_BATCH: u32 = 256;
/// Optimiser steps at most (early stopping usually ends sooner).
pub const DEFAULT_STEPS: u32 = 4000;
/// Peak learning rate.
pub const DEFAULT_LR: f32 = 1e-3;
/// Held-out evaluations without improvement before training stops.
pub const DEFAULT_PATIENCE: u32 = 8;
/// Steps between held-out evaluations.
pub const DEFAULT_EVAL_INTERVAL: u32 = 100;
/// Share of each subject's numeric values hidden for the value objective.
pub const DEFAULT_MASK_RATE: f64 = 0.3;


/// Read a `timeline-v1` file: one subject per line, every line validated.
pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<Subject>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    parse_jsonl(&text).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

/// Each subject's observed outcome among `codes`, for the metrics in
/// [`crate::survival`]: the time from entry to its first event of any of the
/// codes inside that code's observation window (its cause is the code's
/// index in `codes`), else censored at the end of the window of `codes[0]`.
/// Codes listed together compete: evaluate a cause of death with the other
/// causes listed after it, a first diagnosis with the causes of death after
/// it.
pub fn observed(subjects: &[Subject], codes: &[&str]) -> Vec<survival::Obs> {
    subjects
        .iter()
        .map(|s| {
            let mut first: Option<(f64, usize)> = None;
            for e in s.events.iter().filter(|e| e.t > s.entry) {
                let Some(k) = codes.iter().position(|c| *c == e.code) else {
                    continue;
                };
                let Some(w) = s.window(&e.code) else { continue };
                // Inside the window, with the same rounding slack the encoder allows.
                if e.t <= w.to + 1e-9 * w.to.abs().max(1.0) && first.is_none_or(|(t, _)| e.t < t) {
                    first = Some((e.t, k));
                }
            }
            let weight = s.weight;
            match first {
                Some((t, k)) => survival::Obs {
                    time: t - s.entry,
                    cause: Some(k),
                    weight,
                },
                None => {
                    let end = codes
                        .first()
                        .and_then(|c| s.window(c))
                        .map_or(s.entry, |w| w.to);
                    survival::Obs {
                        time: end - s.entry,
                        cause: None,
                        weight,
                    }
                }
            }
        })
        .collect()
}

/// What to train: the outcome codes, the model's shape and the optimiser's
/// schedule. Only the codes are required; everything else has a default.
#[derive(Clone, Debug)]
pub struct TimelineSpec {
    codes: Vec<String>,
    absorbing: Vec<String>,
    shape: Option<TimelineConfig>,
    additive: bool,
    knots: Option<Vec<f32>>,
    max_tokens: Option<u32>,
    batch: u32,
    steps: u32,
    lr: f32,
    patience: u32,
    eval_interval: u32,
    mask_rate: f64,
    seed: u64,
    vocab: FitOptions,
    forecasts: Option<(u32, f32)>,
    visits: Option<u32>,
}

impl TimelineSpec {
    /// Predict `codes`; `absorbing` (a subset) end follow-up for all of them.
    pub fn new<C: Into<String>, A: Into<String>>(
        codes: impl IntoIterator<Item = C>,
        absorbing: impl IntoIterator<Item = A>,
    ) -> TimelineSpec {
        TimelineSpec {
            codes: codes.into_iter().map(Into::into).collect(),
            absorbing: absorbing.into_iter().map(Into::into).collect(),
            shape: None,
            additive: false,
            knots: None,
            max_tokens: None,
            batch: DEFAULT_BATCH,
            steps: DEFAULT_STEPS,
            lr: DEFAULT_LR,
            patience: DEFAULT_PATIENCE,
            eval_interval: DEFAULT_EVAL_INTERVAL,
            mask_rate: DEFAULT_MASK_RATE,
            seed: 1,
            vocab: FitOptions::default(),
            forecasts: None,
            visits: None,
        }
    }
    /// Use exactly this model shape (its `vocab` and `n_codes` are replaced
    /// by the fitted vocabulary's and the spec's codes).
    pub fn shape(mut self, cfg: TimelineConfig) -> Self {
        self.shape = Some(cfg);
        self
    }
    /// Train the additive proportional-hazards baseline instead of the set encoder.
    pub fn additive(mut self, on: bool) -> Self {
        self.additive = on;
        self
    }
    /// Hazard piece boundaries over time since prediction, starting at 0.
    pub fn knots(mut self, knots: Vec<f32>) -> Self {
        self.knots = Some(knots);
        self
    }
    /// Tokens per subject, including the summary token.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
    /// Subjects per batch.
    pub fn batch(mut self, b: u32) -> Self {
        self.batch = b;
        self
    }
    /// Optimiser steps at most.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }
    /// Peak learning rate.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }
    /// Early-stopping patience, in held-out evaluations.
    pub fn patience(mut self, p: u32) -> Self {
        self.patience = p;
        self
    }
    /// Steps between held-out evaluations.
    pub fn eval_interval(mut self, n: u32) -> Self {
        self.eval_interval = n;
        self
    }
    /// Share of numeric values hidden for the value objective (0 turns it off).
    pub fn mask_rate(mut self, r: f64) -> Self {
        self.mask_rate = r;
        self
    }
    /// Seed of the initial weights, the batches and the masks.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Train a forecast head on up to `per_subject` future measurements of
    /// each subject (observations after its entry), weighted `weight`
    /// against the event objective; [`TimelineModel::forecast`] needs it.
    pub fn forecasts(mut self, per_subject: u32, weight: f32) -> Self {
        self.forecasts = Some((per_subject, weight));
        self
    }
    /// Carry a continuous-time state across the most recent `visits` visits
    /// (distinct observation times) instead of reading the whole history as
    /// one set: between visits the state reverts towards the population over
    /// the elapsed time, so gaps longer than any in training extrapolate.
    pub fn visits(mut self, visits: u32) -> Self {
        self.visits = Some(visits);
        self
    }
    /// Quantile knots per numeric variable, and the subjects a categorical
    /// level needs to get its own token.
    pub fn vocabulary(mut self, knots: usize, min_count: usize) -> Self {
        self.vocab = FitOptions { knots, min_count };
        self
    }

    fn config(&self, vocab: &Vocab) -> TimelineConfig {
        let mut cfg = self
            .shape
            .clone()
            .unwrap_or_else(|| TimelineConfig::default_for(vocab.len(), self.codes.len() as u32));
        cfg.vocab = vocab.len();
        cfg.n_codes = self.codes.len() as u32;
        cfg.additive = self.additive || cfg.additive;
        if let Some(k) = &self.knots {
            cfg.knots = k.clone();
        }
        if let Some(n) = self.max_tokens {
            cfg.max_tokens = n;
        }
        if let Some((n, w)) = self.forecasts {
            cfg.forecasts = n;
            cfg.forecast_weight = w;
        }
        if let Some(v) = self.visits {
            cfg.visits = v;
        }
        cfg
    }
}

/// What training did.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineReport {
    /// Optimiser steps run (early stopping may end before the spec's limit).
    pub steps: u32,
    /// Training loss before the first step.
    pub initial_loss: f32,
    /// Training loss at the last step.
    pub final_loss: Option<f32>,
    /// Weighted event NLL on the held-out subjects, of the model kept.
    pub held_out_event_nll: f32,
    /// Trainable parameters.
    pub parameters: usize,
    /// Known tokens left out because a subject had more than `max_tokens - 1`.
    pub truncated_tokens: usize,
}

/// A trained timeline model with its vocabulary.
pub struct TimelineModel {
    saved: Saved,
}

impl std::fmt::Debug for TimelineModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimelineModel")
            .field("config", &self.saved.model.cfg)
            .field("codes", &self.saved.vocab.codes)
            .finish()
    }
}

impl TimelineModel {
    /// Fit the vocabulary on `train`, then train with early stopping on the
    /// event NLL of `held_out` (which must not overlap `train`), keeping the
    /// best model.
    pub fn train(
        train: &[Subject],
        held_out: &[Subject],
        spec: &TimelineSpec,
    ) -> Result<(TimelineModel, TimelineReport)> {
        if train.is_empty() || held_out.is_empty() {
            return Err(Error::MissingArgument(
                "training and held-out subjects are both required".into(),
            ));
        }
        let vocab =
            Vocab::fit(train, &spec.codes, &spec.absorbing, &spec.vocab).map_err(Error::Backend)?;
        let cfg = spec.config(&vocab);
        cfg.validate().map_err(Error::Backend)?;
        let enc_train: Vec<Encoded> = train.iter().map(|s| encode(s, &vocab, &cfg)).collect();
        let enc_held: Vec<Encoded> = held_out.iter().map(|s| encode(s, &vocab, &cfg)).collect();
        let truncated_tokens = enc_train.iter().chain(&enc_held).map(|e| e.truncated).sum();
        let model = Horizon::new(
            cfg.clone(),
            spec.batch,
            &horizon::init_weights(&cfg, spec.seed),
        );
        let opts = model::FitOpts {
            steps: spec.steps,
            batch_size: spec.batch,
            block_size: cfg.max_tokens,
            lr: spec.lr,
            min_lr: spec.lr / 10.0,
            warmup: (spec.steps / 40).max(1),
            decay_iters: spec.steps,
            weight_decay: 0.1,
            eval_interval: spec.eval_interval,
            patience: spec.patience,
            checkpoint_secs: 0,
            seed: spec.seed,
            ..Default::default()
        };
        let objective = TimelineObjective::new(&enc_train, Some(&enc_held), spec.mask_rate);
        let (fit, model) =
            model::fit_controlled(model, objective, &opts, None, model::FitControl::default())?;
        let held_out_event_nll = event_nll(&model, &enc_held);
        let parameters = cfg.param_list().iter().map(|(_, n)| n).sum();
        let report = TimelineReport {
            steps: fit.steps_completed,
            initial_loss: fit.initial_loss,
            final_loss: fit.final_loss,
            held_out_event_nll,
            parameters,
            truncated_tokens,
        };
        Ok((TimelineModel { saved: Saved { model, vocab } }, report))
    }

    /// Load a model [`TimelineModel::save`] wrote.
    pub fn load(dir: impl AsRef<Path>) -> Result<TimelineModel> {
        Ok(TimelineModel {
            saved: Saved::load(dir.as_ref()).map_err(Error::Backend)?,
        })
    }

    /// Write the weights (with the configuration in their header) and the
    /// vocabulary into `dir`.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        self.saved.save(dir.as_ref()).map_err(Error::Backend)
    }

    /// The outcome codes, in the model's order.
    pub fn codes(&self) -> &[String] {
        &self.saved.vocab.codes
    }

    /// The model's configuration.
    pub fn config(&self) -> &TimelineConfig {
        &self.saved.model.cfg
    }

    fn encode(&self, subjects: &[Subject]) -> Result<Vec<Encoded>> {
        self.saved.encode(subjects).map_err(Error::Backend)
    }

    /// One prediction per subject, in order.
    pub fn predict(&self, subjects: &[Subject]) -> Result<Vec<Prediction>> {
        let last_knot = self.saved.horizon();
        Ok(self
            .saved
            .predict(subjects)
            .map_err(Error::Backend)?
            .into_iter()
            .map(|curves| Prediction {
                members: vec![curves],
                codes: self.saved.vocab.codes.clone(),
                last_knot,
            })
            .collect())
    }

    /// The learned summary state of each subject (its representation, for
    /// probing or clustering).
    pub fn states(&self, subjects: &[Subject]) -> Result<Vec<Vec<f32>>> {
        Ok(predict_states(&self.saved.model, &self.encode(subjects)?))
    }

    /// Quantiles (`levels`, each in `(0, 1)`) of numeric `var`, in its own
    /// unit, `ahead` after each subject's entry; one list per subject. Needs
    /// a model trained with [`TimelineSpec::forecasts`].
    pub fn forecast(
        &self,
        subjects: &[Subject],
        var: &str,
        ahead: f64,
        levels: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        if self.saved.model.cfg.forecasts == 0 {
            return Err(Error::Backend(
                "this model has no forecast head: train it with TimelineSpec::forecasts".into(),
            ));
        }
        if let Some(l) = levels.iter().find(|l| !(**l > 0.0 && **l < 1.0)) {
            return Err(Error::Backend(format!(
                "quantile level {l} is outside (0, 1)"
            )));
        }
        let query = forecast_query(&self.saved.vocab, &self.saved.model.cfg, var, ahead).ok_or_else(|| {
            Error::Backend(format!("{var} is not a numeric variable of this model"))
        })?;
        let enc = self.encode(subjects)?;
        let queries = vec![vec![query]; enc.len()];
        let pred = predict_forecasts(&self.saved.model, &enc, &queries).map_err(Error::Backend)?;
        Ok(pred
            .iter()
            .map(|p| {
                let (mu, sigma) = (p[0].0 as f64, p[0].1 as f64);
                levels
                    .iter()
                    .map(|&q| {
                        self.saved.vocab
                            .forecast_quantile(var, mu, sigma, q)
                            .unwrap_or(f64::NAN)
                    })
                    .collect()
            })
            .collect())
    }

    /// The weighted mean event NLL over `subjects` (lower is better; the
    /// quantity training early-stops on).
    pub fn event_nll(&self, subjects: &[Subject]) -> Result<f32> {
        Ok(event_nll(&self.saved.model, &self.encode(subjects)?))
    }
}

/// One subject's predicted outcome curves: one model's, or the equal-weight
/// mixture of several models' ([`Prediction::ensemble`]).
#[derive(Clone, Debug)]
pub struct Prediction {
    members: Vec<Curves>,
    codes: Vec<String>,
    last_knot: f64,
}

impl Prediction {
    /// The ensemble of `parts` (one subject's predictions from models trained
    /// apart, say with different seeds): every probability is the mean of the
    /// parts'. `None` for no parts, or parts over different outcome codes or
    /// horizons.
    pub fn ensemble(parts: &[Prediction]) -> Option<Prediction> {
        let first = parts.first()?;
        if parts.iter().any(|p| p.codes != first.codes || p.last_knot != first.last_knot) {
            return None;
        }
        Some(Prediction {
            members: parts.iter().flat_map(|p| p.members.iter().cloned()).collect(),
            codes: first.codes.clone(),
            last_knot: first.last_knot,
        })
    }
    /// Probability that `code` happens within `t` (in the dataset's unit) of
    /// the prediction time; `None` for an unknown code. Held constant past
    /// the last knot ([`Prediction::horizon`]): the model says nothing later.
    pub fn cif(&self, code: &str, t: f64) -> Option<f64> {
        self.member_cifs(code, t).map(|v| v.iter().sum::<f64>() / v.len() as f64)
    }
    /// Each member's probability that `code` happens within `t`: their spread
    /// is the disagreement between models trained apart.
    pub fn member_cifs(&self, code: &str, t: f64) -> Option<Vec<f64>> {
        let k = self.codes.iter().position(|c| c == code)?;
        Some(self.members.iter().map(|m| m.cif(k, t)).collect())
    }
    /// Probability of no absorbing outcome within `t`.
    pub fn survival(&self, t: f64) -> f64 {
        self.members.iter().map(|m| m.survival(t)).sum::<f64>() / self.members.len() as f64
    }
    /// The time by which survival falls to `q`, if within the knots.
    pub fn survival_quantile(&self, q: f64) -> Option<f64> {
        if let [only] = self.members.as_slice() {
            return only.survival_quantile(q);
        }
        // The mixture's survival is continuous and decreasing: bisect.
        if self.survival(self.last_knot) > q {
            return None;
        }
        let (mut lo, mut hi) = (0.0, self.last_knot);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.survival(mid) > q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(hi)
    }
    /// The longest horizon the model predicts to.
    pub fn horizon(&self) -> f64 {
        self.last_knot
    }
    /// The members' curves (hazard per piece and code): one for a single
    /// model.
    pub fn members(&self) -> &[Curves] {
        &self.members
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_applies_only_what_it_was_given() {
        let s = TimelineSpec::new(["a", "b"], ["a"])
            .knots(vec![0.0, 1.0, 2.0])
            .steps(10);
        let subjects = horizon::synthetic::population(50, 1).0;
        let vocab = Vocab::fit(&subjects, &s.codes, &s.absorbing, &s.vocab).unwrap();
        let cfg = s.config(&vocab);
        assert_eq!(cfg.knots, vec![0.0, 1.0, 2.0]);
        assert_eq!(cfg.n_codes, 2);
        assert_eq!(
            cfg.d_model,
            TimelineConfig::default_for(1, 1).d_model,
            "unset fields keep the default"
        );
        assert!(!cfg.additive);
        assert!(s.clone().additive(true).config(&vocab).additive);
    }

    #[test]
    fn observed_outcomes_compete_and_censor_at_the_window() {
        let line = |events: &str| {
            Subject::from_json_line(&format!(
                r#"{{"subject_id":"a","weight":2,"source":"s","entry":50,"calendar_at_entry":2000,"events":[{events}],"at_risk":[{{"code":"*","from":50,"to":60}}]}}"#
            ))
            .unwrap()
        };
        let s = [
            line(r#"{"t":45,"code":"x"},{"t":53,"code":"y"},{"t":55,"code":"x"}"#),
            line(""),
            line(r#"{"t":61,"code":"x"}"#),
        ];
        let o = observed(&s, &["x", "y"]);
        assert_eq!(
            o[0],
            survival::Obs {
                time: 3.0,
                cause: Some(1),
                weight: 2.0
            },
            "y came first; history before entry ignored"
        );
        assert_eq!(
            o[1],
            survival::Obs {
                time: 10.0,
                cause: None,
                weight: 2.0
            }
        );
        assert_eq!(
            o[2].cause, None,
            "an event after the window is not observed"
        );
    }

    #[test]
    fn reading_a_file_reports_the_bad_line() {
        let dir = std::env::temp_dir().join(format!("brain-timeline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.jsonl");
        std::fs::write(&path, "{\"subject_id\":\"a\",\"source\":\"s\",\"entry\":1,\"calendar_at_entry\":2000}\n\n{\"subject_id\":\"b\"}\n").unwrap();
        let err = read_jsonl(&path).unwrap_err().to_string();
        assert!(err.contains("line 3"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_ensemble_averages_its_members() {
        let knots = [0.0f32, 1.0, 5.0];
        let one = |rate: f32| Prediction {
            members: vec![Curves::new(&[rate.ln(), rate.ln()], &knots, &[true])],
            codes: vec!["death".into()],
            last_knot: 5.0,
        };
        let (a, b) = (one(0.1), one(0.3));
        let same = Prediction::ensemble(std::slice::from_ref(&a)).unwrap();
        assert_eq!(same.cif("death", 2.0), a.cif("death", 2.0));
        let both = Prediction::ensemble(&[a.clone(), b.clone()]).unwrap();
        let mean = 0.5 * ((1.0 - (-0.2f64).exp()) + (1.0 - (-0.6f64).exp()));
        assert!((both.cif("death", 2.0).unwrap() - mean).abs() < 1e-6);
        assert_eq!(both.member_cifs("death", 2.0).unwrap().len(), 2);
        let median = both.survival_quantile(0.5).unwrap();
        assert!((both.survival(median) - 0.5).abs() < 1e-9, "{median}");
        let other = Prediction { last_knot: 4.0, ..one(0.2) };
        assert!(Prediction::ensemble(&[a, other]).is_none(), "different horizons do not mix");
        assert!(Prediction::ensemble(&[]).is_none());
    }
}
