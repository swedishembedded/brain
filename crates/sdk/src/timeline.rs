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
use std::sync::Arc;

use horizon::encode::{forecast_query, Encoded};
use horizon::ensemble::{Ensemble, Members};
use horizon::saved::{parse_jsonl, Saved, Scored};
use horizon::survival::Curves;
use horizon::train::{event_nll, event_nll_each, predict_forecasts, predict_states};

/// A synthetic population with KNOWN hazards, to check a pipeline against the
/// truth before trusting it on real data.
pub use horizon::synthetic;
pub use horizon::calibration::{observed, Calibration, Gap, MIN_EVENTS};
pub use horizon::ensemble::{Kind as EnsembleKind, Manifest as EnsembleManifest, MemberRecord};
pub use horizon::fit::{
    TrainSpec as TimelineSpec, DEFAULT_BATCH, DEFAULT_EVAL_INTERVAL, DEFAULT_LR, DEFAULT_MASK_RATE,
    DEFAULT_PATIENCE, DEFAULT_STEPS,
};
pub use horizon::evaluation::{
    Absent, Bootstrap, CalibrationMetrics, Evaluation, EvaluationSpec, HorizonMetrics, Interval, Intervals,
};
pub use horizon::forecast::{
    AsOf, CodeRisk, Coverage, CurveSet, EnsembleSummary, ForecastRequest, HorizonRisks, RiskForecast,
    RiskStatus, DISCLAIMER,
};
pub use horizon::history::{HistoryWarning, PatientHistory};
pub use horizon::saved::ModelIdentity;
pub use horizon::support::{AssessOptions, Assessment, Support, Warning, DEFAULT_MAX_OOD_SCORE};
pub use horizon::timeline::{AtRisk, Event, Observation, Subject, Value};
pub use horizon::HorizonConfig as TimelineConfig;
pub use horizon::{Backbone, Mixer, StackConfig};

use crate::{Error, Result};

/// Read a `timeline-v1` file: one subject per line, every line validated.
pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<Subject>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    parse_jsonl(&text).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

/// How to calibrate a trained model ([`TimelineModel::calibrate`]): the
/// horizons whose risks are to be calibrated, and the event minimum a horizon
/// needs on the validation subjects to be.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationSpec {
    horizons: Vec<f64>,
    min_events: usize,
}

impl CalibrationSpec {
    /// Calibrate the risk by each of `horizons` (after entry, in the data's
    /// unit, each within the model's knots).
    pub fn new(horizons: impl IntoIterator<Item = f64>) -> CalibrationSpec {
        CalibrationSpec {
            horizons: horizons.into_iter().collect(),
            min_events: MIN_EVENTS,
        }
    }
    /// Calibrate a (code, horizon) only if the validation subjects hold at
    /// least this many events of the code by then, and as many still
    /// event-free ([`MIN_EVENTS`] unless set). A lower minimum fits a
    /// calibrator to fewer events; the others are reported, not guessed.
    pub fn min_events(mut self, n: usize) -> Self {
        self.min_events = n;
        self
    }
}

/// When a prediction is withheld for lack of support
/// ([`TimelineModel::predict_or_abstain`]): a subject whose
/// [`Assessment::ood_score`] is above `max_ood_score` gets no probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Abstain {
    max_ood_score: f64,
    options: AssessOptions,
}

impl Default for Abstain {
    /// Withhold at the edge of the training support (any warning): the same
    /// default the served capability uses.
    fn default() -> Self {
        Abstain {
            max_ood_score: DEFAULT_MAX_OOD_SCORE,
            options: AssessOptions::default(),
        }
    }
}

impl Abstain {
    /// Withhold when the out-of-distribution score is above `max_ood_score`
    /// (`1.0` is the edge of the support; unknown variables, categories and
    /// event codes score 10, so only a threshold of 10 or more lets them
    /// through).
    pub fn above(max_ood_score: f64) -> Abstain {
        Abstain {
            max_ood_score,
            ..Abstain::default()
        }
    }
    /// How strictly the support is judged (the margins beyond the trained
    /// ranges) before the score is compared with the threshold.
    pub fn options(mut self, options: AssessOptions) -> Self {
        self.options = options;
        self
    }
}

/// Risk withheld: the subject is outside what the model was trained on, so
/// no probability is given where a made-up one would look confident.
#[derive(Clone, Debug, PartialEq)]
pub struct Unavailable {
    /// Why: the score and the warnings.
    pub assessment: Assessment,
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("risk unavailable: insufficient support")
    }
}

impl std::error::Error for Unavailable {}

/// A subject's prediction, or the reason there is none.
pub type Risk = std::result::Result<Prediction, Unavailable>;

/// What training did.
pub type TimelineReport = horizon::fit::Report;

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
        let (saved, report) = horizon::fit::train(train, held_out, spec, &mut horizon::fit::Hooks::default())
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok((TimelineModel { saved }, report))
    }

    /// Load a model [`TimelineModel::save`] wrote.
    pub fn load(dir: impl AsRef<Path>) -> Result<TimelineModel> {
        Ok(TimelineModel {
            saved: Saved::load(dir.as_ref()).map_err(Error::Backend)?,
        })
    }

    /// Write the weights (with the configuration in their header), the
    /// vocabulary, the training support and, if the model was calibrated, the
    /// calibration into `dir`.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        self.saved.save(dir.as_ref()).map_err(Error::Backend)
    }

    /// Fit the model's calibration on `validation` subjects: for every outcome
    /// code and every horizon of `spec`, a Venn-Abers calibrator under
    /// inverse-probability-of-censoring weights, the censoring distribution
    /// estimated on `validation` itself. Use subjects the model was neither
    /// trained nor early-stopped on, and never the test set the result is
    /// judged on. [`Prediction::calibrated_cif`] then answers at the
    /// calibrated horizons; a (code, horizon) with too few validation events
    /// is left uncalibrated and listed in the returned [`Calibration`], where
    /// the prediction has no calibrated risk (never a zero). The calibration
    /// is bound to these exact weights, saved with them, and replaces any
    /// earlier one.
    pub fn calibrate(
        &mut self,
        validation: &[Subject],
        spec: &CalibrationSpec,
    ) -> Result<&Calibration> {
        self.saved
            .calibrate(validation, &spec.horizons, spec.min_events)
            .map_err(Error::Backend)?;
        self.calibration()
            .ok_or_else(|| Error::Backend("calibration was not stored".into()))
    }

    /// What the model was trained on, recorded at training; `None` for a
    /// model saved before it was kept (its support is unknown).
    pub fn support(&self) -> Option<&Support> {
        self.saved.support.as_ref()
    }

    /// Each subject against what the model was trained on, with the default
    /// margins: `supported`, a continuous `ood_score` (1 at the edge) and
    /// typed warnings (unknown codes, values and clocks outside the trained
    /// range, unusual history lengths, a state far from the training
    /// states). `supported` is `None` for a model with no recorded support:
    /// unknown, neither supported nor unsupported.
    pub fn assess(&self, subjects: &[Subject]) -> Result<Vec<Assessment>> {
        self.assess_with(subjects, &AssessOptions::default())
    }

    /// [`TimelineModel::assess`] with explicit margins.
    pub fn assess_with(&self, subjects: &[Subject], options: &AssessOptions) -> Result<Vec<Assessment>> {
        self.saved.assess(subjects, options).map_err(Error::Backend)
    }

    /// The model's calibration, if it has one.
    pub fn calibration(&self) -> Option<&Calibration> {
        self.saved.calibration.as_deref()
    }

    /// The outcome codes, in the model's order.
    pub fn codes(&self) -> &[String] {
        &self.saved.vocab.codes
    }

    /// The next-event group's codes (empty without
    /// [`TimelineSpec::next_events`]).
    pub fn next_event_codes(&self) -> &[String] {
        &self.saved.vocab.next_events
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
        predictions(&self.saved, subjects)
    }

    /// One prediction per subject, in order, or `Err(Unavailable)` for a
    /// subject outside the training support (see [`Abstain`]): no probability
    /// instead of a confident one the model cannot stand behind. A model with
    /// no recorded support never withholds (its support is unknown) and says
    /// so in [`TimelineModel::assess`].
    pub fn predict_or_abstain(&self, subjects: &[Subject], policy: &Abstain) -> Result<Vec<Risk>> {
        let last_knot = self.saved.horizon();
        Ok(self
            .saved
            .score(subjects, &policy.options)
            .map_err(Error::Backend)?
            .into_iter()
            .map(|Scored { curves, assessment }| {
                if assessment.abstains(policy.max_ood_score) {
                    Err(Unavailable { assessment })
                } else {
                    Ok(Prediction {
                        members: vec![curves],
                        calibrations: vec![self.saved.calibration.clone()],
                        codes: self.saved.vocab.codes.clone(),
                        last_knot,
                    })
                }
            })
            .collect())
    }

    /// One next-event forecast per subject, in order: which code of the
    /// next-event group happens first and when. Empty for a model trained
    /// without [`TimelineSpec::next_events`].
    pub fn predict_next_events(&self, subjects: &[Subject]) -> Result<Vec<NextEvent>> {
        let last_knot = self.saved.horizon();
        Ok(self
            .saved
            .predict_next_events(subjects)
            .map_err(Error::Backend)?
            .into_iter()
            .map(|curves| NextEvent {
                curves,
                codes: self.saved.vocab.next_events.clone(),
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

    /// The structured forecast for one patient history (the stateless
    /// "append a checkup and re-predict" call): the whole history goes in and
    /// nothing is kept, so the forecast depends only on the history and the
    /// model. It reports the model's identity, what the history covered and
    /// missed, the risk curves over the knots, the risks at the request's
    /// horizons (calibrated, with the Venn-Abers interval, only where the model
    /// was calibrated for exactly that horizon: absent, never zero) and the
    /// data-quality assessment, or `risk: unavailable` with no numbers when the
    /// history is outside what the model was trained on.
    ///
    /// A forecast does not diagnose and recommends no treatment: its risks are
    /// associations in the training population, not effects of any action.
    /// A horizon past the last knot, or a unit other than the one the model
    /// was trained on, is an error.
    pub fn forecast_history(&self, history: &PatientHistory, request: &ForecastRequest) -> Result<RiskForecast> {
        let mut all = self.forecast_histories(std::slice::from_ref(history), request)?;
        all.pop().ok_or_else(|| Error::Backend("no forecast was produced".into()))
    }

    /// [`TimelineModel::forecast_history`] for several histories, in order,
    /// through one forward pass. One history the model cannot read fails the
    /// call, naming it.
    pub fn forecast_histories(&self, histories: &[PatientHistory], request: &ForecastRequest) -> Result<Vec<RiskForecast>> {
        horizon::forecast::forecast(&self.saved, histories, request).map_err(Error::Backend)
    }

    /// Judge the model on held-out `subjects` (never ones it was trained,
    /// early-stopped or calibrated on): per outcome code and horizon of
    /// `spec`, Uno's concordance, the time-dependent AUC, the IPCW Brier score
    /// and its integral, the calibration slope, intercept, observed over
    /// expected and error, and, when the subjects carry groups, cluster-
    /// bootstrap intervals; plus the held-out event NLL. A (code, horizon)
    /// with fewer than [`EvaluationSpec::min_events`] events is left out and
    /// listed, never reported from noise.
    pub fn evaluate(&self, subjects: &[Subject], spec: &EvaluationSpec) -> Result<Evaluation> {
        horizon::evaluation::evaluate(&self.saved, subjects, spec).map_err(Error::Backend)
    }

    /// Every subject's own event NLL of the outcome codes (per unit of its
    /// weight), in order, from one pass: what [`TimelineModel::event_nll`]
    /// averages, for paired comparisons between models subject by subject.
    pub fn event_nll_each(&self, subjects: &[Subject]) -> Result<Vec<f32>> {
        Ok(event_nll_each(&self.saved.model, &self.encode(subjects)?))
    }

    /// The weighted mean event NLL over `subjects` (lower is better; the
    /// quantity training early-stops on).
    pub fn event_nll(&self, subjects: &[Subject]) -> Result<f32> {
        Ok(event_nll(&self.saved.model, &self.encode(subjects)?))
    }
}

/// One prediction per subject from one model.
fn predictions(saved: &Saved, subjects: &[Subject]) -> Result<Vec<Prediction>> {
    let last_knot = saved.horizon();
    Ok(saved
        .predict(subjects)
        .map_err(Error::Backend)?
        .into_iter()
        .map(|curves| Prediction {
            members: vec![curves],
            calibrations: vec![saved.calibration.clone()],
            codes: saved.vocab.codes.clone(),
            last_knot,
        })
        .collect())
}

/// Several models trained apart ([`EnsembleKind::Seeded`]: another seed each;
/// [`EnsembleKind::Bootstrap`]: subjects resampled with replacement by group):
/// the prediction is their mean and the disagreement between them is the
/// uncertainty about it ([`Prediction::member_cifs`], [`Prediction::cif_spread`],
/// and `member_range` and the knots' `cif_min`/`cif_max` of a forecast). Saved as
/// one directory, `ensemble.json` with the member seeds, bootstrap draws and
/// weights digests, and `members/`.
///
/// There is no Monte-Carlo dropout: horizon has no dropout in its architecture.
pub struct TimelineEnsemble {
    inner: Ensemble,
}

impl std::fmt::Debug for TimelineEnsemble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimelineEnsemble")
            .field("kind", &self.inner.manifest().kind)
            .field("members", &self.inner.members().len())
            .finish()
    }
}

impl TimelineEnsemble {
    /// Train `members` (at least two) models on `train`, each early-stopped on
    /// `held_out`; member `i` uses the spec's seed plus `i`. One report per
    /// member.
    pub fn train(
        train: &[Subject],
        held_out: &[Subject],
        spec: &TimelineSpec,
        members: usize,
        kind: EnsembleKind,
    ) -> Result<(TimelineEnsemble, Vec<TimelineReport>)> {
        if train.is_empty() || held_out.is_empty() {
            return Err(Error::MissingArgument(
                "training and held-out subjects are both required".into(),
            ));
        }
        let (inner, reports) =
            Ensemble::train(train, held_out, spec, members, kind, &mut |_, _| {}, &|| false)
                .map_err(|e| Error::Backend(e.to_string()))?;
        Ok((TimelineEnsemble { inner }, reports))
    }

    /// Load an ensemble [`TimelineEnsemble::save`] wrote, verifying every
    /// member's weights against the recorded digest.
    pub fn load(dir: impl AsRef<Path>) -> Result<TimelineEnsemble> {
        Ok(TimelineEnsemble { inner: Ensemble::load(dir.as_ref()).map_err(Error::Backend)? })
    }

    /// Write the ensemble into `dir`, atomically. An existing ensemble
    /// directory is replaced; any other existing path is refused.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        self.inner.save(dir.as_ref(), true).map_err(Error::Backend)
    }

    /// How the members differ, their seeds, bootstrap draws and digests.
    pub fn manifest(&self) -> &EnsembleManifest {
        self.inner.manifest()
    }

    /// The outcome codes, in the members' shared order.
    pub fn codes(&self) -> &[String] {
        &self.inner.members()[0].vocab.codes
    }

    /// One prediction per subject, in order: the members' mean, with every
    /// member's curves kept.
    pub fn predict(&self, subjects: &[Subject]) -> Result<Vec<Prediction>> {
        let per_member: Vec<Vec<Prediction>> = self
            .inner
            .members()
            .iter()
            .map(|m| predictions(m, subjects))
            .collect::<Result<_>>()?;
        (0..subjects.len())
            .map(|i| {
                let parts: Vec<Prediction> = per_member.iter().map(|p| p[i].clone()).collect();
                Prediction::ensemble(&parts).ok_or_else(|| Error::Backend("ensemble members disagree on codes or horizon".into()))
            })
            .collect()
    }

    /// The structured forecast for one history: the members' mean, the member
    /// range per horizon and the curves' extremes (see
    /// [`TimelineModel::forecast_history`]).
    pub fn forecast_history(&self, history: &PatientHistory, request: &ForecastRequest) -> Result<RiskForecast> {
        let mut all = self.forecast_histories(std::slice::from_ref(history), request)?;
        all.pop().ok_or_else(|| Error::Backend("no forecast was produced".into()))
    }

    /// [`TimelineEnsemble::forecast_histories`] for several histories, in order.
    pub fn forecast_histories(&self, histories: &[PatientHistory], request: &ForecastRequest) -> Result<Vec<RiskForecast>> {
        self.inner.forecast(histories, request).map_err(Error::Backend)
    }
}

/// One subject's forecast of which event of the next-event group comes first.
#[derive(Clone, Debug)]
pub struct NextEvent {
    curves: Curves,
    codes: Vec<String>,
    last_knot: f64,
}

impl NextEvent {
    /// Probability that `code` is the first event of the group and happens
    /// within `t` of the prediction time; `None` for a code outside the group.
    /// Held constant past [`NextEvent::horizon`].
    pub fn first(&self, code: &str, t: f64) -> Option<f64> {
        let k = self.codes.iter().position(|c| c == code)?;
        Some(self.curves.cif(k, t))
    }
    /// Probability that any event of the group has happened within `t`.
    pub fn any(&self, t: f64) -> f64 {
        1.0 - self.curves.survival(t)
    }
    /// The longest horizon the model predicts to.
    pub fn horizon(&self) -> f64 {
        self.last_knot
    }
}

/// One subject's predicted outcome curves: one model's, or the equal-weight
/// mixture of several models' ([`Prediction::ensemble`]). The curves are the
/// model's raw risk; a calibrated model also gives the calibrated risk at its
/// calibrated horizons ([`Prediction::calibrated_cif`]).
#[derive(Clone, Debug)]
pub struct Prediction {
    members: Vec<Curves>,
    /// Each member's calibration (`None` for an uncalibrated model).
    calibrations: Vec<Option<Arc<Calibration>>>,
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
            calibrations: parts.iter().flat_map(|p| p.calibrations.iter().cloned()).collect(),
            codes: first.codes.clone(),
            last_knot: first.last_knot,
        })
    }
    /// The model's raw probability that `code` happens within `t` (in the
    /// dataset's unit) of the prediction time; `None` for an unknown code.
    /// Held constant past the last knot ([`Prediction::horizon`]): the model
    /// says nothing later. Raw is not calibrated: see
    /// [`Prediction::calibrated_cif`].
    pub fn cif(&self, code: &str, t: f64) -> Option<f64> {
        self.member_cifs(code, t).map(|v| v.iter().sum::<f64>() / v.len() as f64)
    }
    /// The calibrated probability that `code` happens within exactly `t`:
    /// the Venn-Abers risk of the model's raw one, at a horizon the model was
    /// calibrated for. `None` for an unknown code, an uncalibrated model, and
    /// a horizon that was not calibrated (including one whose validation data
    /// had too few events): absent, never a number made up. For an ensemble,
    /// the mean of its members' calibrated risks, `None` unless every member
    /// has one.
    pub fn calibrated_cif(&self, code: &str, t: f64) -> Option<f64> {
        let n = self.members.len() as f64;
        Some(self.calibrated_each(code, t)?.iter().map(|c| c.risk).sum::<f64>() / n)
    }
    /// The Venn-Abers interval `(p0, p1)` behind [`Prediction::calibrated_cif`]:
    /// one end is calibrated whatever the model, and the width says how little
    /// calibration data stands behind the score. `None` wherever
    /// [`Prediction::calibrated_cif`] is.
    pub fn cif_interval(&self, code: &str, t: f64) -> Option<(f64, f64)> {
        let each = self.calibrated_each(code, t)?;
        let n = each.len() as f64;
        Some((
            each.iter().map(|c| c.lower).sum::<f64>() / n,
            each.iter().map(|c| c.upper).sum::<f64>() / n,
        ))
    }
    fn calibrated_each(&self, code: &str, t: f64) -> Option<Vec<horizon::calibration::Calibrated>> {
        let k = self.codes.iter().position(|c| c == code)?;
        self.members
            .iter()
            .zip(&self.calibrations)
            .map(|(m, cal)| cal.as_ref()?.apply(code, t, m.cif(k, t)))
            .collect()
    }
    /// Each member's probability that `code` happens within `t`: their spread
    /// is the disagreement between models trained apart.
    pub fn member_cifs(&self, code: &str, t: f64) -> Option<Vec<f64>> {
        let k = self.codes.iter().position(|c| c == code)?;
        Some(self.members.iter().map(|m| m.cif(k, t)).collect())
    }
    /// The sample standard deviation across the members of the probability
    /// that `code` happens within `t`: how much the models trained apart
    /// disagree. `None` for an unknown code and for a single model.
    pub fn cif_spread(&self, code: &str, t: f64) -> Option<f64> {
        let v = self.member_cifs(code, t)?;
        if v.len() < 2 {
            return None;
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        Some((v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt())
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
            calibrations: vec![None],
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
