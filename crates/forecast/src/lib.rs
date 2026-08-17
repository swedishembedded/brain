// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-agnostic forecasting domain types and the [`ForecastModel`] seam.
//!
//! This crate is the spine of brain's forecasting API. It defines *what a
//! forecast request and response are* — independent of any particular model —
//! so a financial tool written against it works unchanged when a new model is
//! registered. It holds **no model code**: the models (Chronos-2, Kronos,
//! FinCast, and the statistical baselines) implement [`ForecastModel`] in their
//! own crates.
//!
//! - [`csv::parse_ohlcv`] - an untrusted OHLCV CSV, validated structurally and
//!   semantically at the boundary, into a [`Panel`] (see [`csv`]).
//! - [`chart::render_png`] - history, forecast and held-out actual on one pair
//!   of axes, via the `gnuplot` CLI (see [`chart`]).
//! - [`Panel`] / [`Item`] / [`Variate`] — the input (see [`panel`]).
//! - [`Forecast`] / [`TargetForecast`] / [`Block`] — the output, a distribution
//!   over the horizon (see [`forecast`]).
//! - [`Representation`] — quantiles | samples | distribution | point | classes.
//! - [`convert::ensure_representations`] — the honesty layer that derives
//!   requested representations from the native one, flagging derived data and
//!   erroring on impossible conversions.
//! - [`Capabilities`] — what a model can do, for capability negotiation.
//! - [`ForecastSpec`] — the knobs of one request.
//! - [`ForecastError`] — a structured, wire-mappable error.

pub mod backtest;
pub mod chart;
pub mod convert;
pub mod csv;
pub mod forecast;
pub mod metrics;
pub mod panel;
pub mod train_data;

pub use backtest::{BacktestReport, BacktestRow, BacktestSpec};
pub use csv::{parse_ohlcv, Bar, OhlcvSeries, Split, Stamp};
pub use forecast::{Block, Forecast, TargetForecast};
pub use panel::{Item, Kind, Panel, Role, Variate};

/// The five ways a forecast distribution can be represented. A model emits one
/// natively; [`convert::ensure_representations`] derives the rest where sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    /// Predicted values at given quantile levels, `[horizon, n_levels]`.
    Quantiles,
    /// Sampled trajectories, `[n_samples, horizon]`.
    Samples,
    /// Parametric distribution parameters, `[horizon, n_params]`.
    Distribution,
    /// A single point path, `[horizon]`.
    Point,
    /// Per-step class probabilities, `[horizon, n_classes]` (direction models).
    Classes,
}

impl Representation {
    pub fn as_str(self) -> &'static str {
        match self {
            Representation::Quantiles => "quantiles",
            Representation::Samples => "samples",
            Representation::Distribution => "distribution",
            Representation::Point => "point",
            Representation::Classes => "classes",
        }
    }
    pub fn parse(s: &str) -> Option<Representation> {
        Some(match s {
            "quantiles" => Representation::Quantiles,
            "samples" => Representation::Samples,
            "distribution" => Representation::Distribution,
            "point" => Representation::Point,
            "classes" => Representation::Classes,
            _ => return None,
        })
    }
}

/// What covariate support a model advertises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovariateSupport {
    /// No covariates; only target series are used.
    None,
    /// Only calendar/time covariates (e.g. Kronos's minute/hour/weekday).
    CalendarOnly,
    /// Arbitrary past and known-future covariates (e.g. Chronos-2).
    Full,
}

impl CovariateSupport {
    pub fn as_str(self) -> &'static str {
        match self {
            CovariateSupport::None => "none",
            CovariateSupport::CalendarOnly => "calendar_only",
            CovariateSupport::Full => "full",
        }
    }
}

/// A model's self-description, returned by capability negotiation so apps
/// discover constraints rather than hard-coding them.
#[derive(Clone, Debug, PartialEq)]
pub struct Capabilities {
    /// Registered name (`"chronos2"`).
    pub name: String,
    /// Longest input context, in native units (see the model's docs on whether
    /// that is timesteps or subtokens).
    pub max_context: usize,
    /// Longest horizon, or `None` if unbounded (autoregressive rollout).
    pub max_horizon: Option<usize>,
    /// The representation this model emits natively.
    pub native_representation: Representation,
    /// Covariate handling.
    pub covariates: CovariateSupport,
    /// Whether known-future covariates are consumed.
    pub supports_known_future: bool,
    /// Whether the model jointly forecasts multiple variates.
    pub multivariate: bool,
    /// Whether output quantile levels are user-selectable (`true`) or fixed.
    pub arbitrary_quantile_levels: bool,
    /// Whether the model is stochastic (sampling-based).
    pub stochastic: bool,
    /// Variate names the model requires (e.g. Kronos needs OHLCV). Empty if the
    /// model accepts any target set.
    pub requires_variates: Vec<String>,
}

impl Capabilities {
    /// Serialize for a `capabilities_result` event.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "max_context": self.max_context,
            "max_horizon": self.max_horizon,
            "native_representation": self.native_representation.as_str(),
            "covariates": self.covariates.as_str(),
            "supports_known_future": self.supports_known_future,
            "multivariate": self.multivariate,
            "arbitrary_quantile_levels": self.arbitrary_quantile_levels,
            "stochastic": self.stochastic,
            "requires_variates": self.requires_variates,
        })
    }
}

/// The knobs of one forecast request: horizon, which representations to return,
/// the quantile grid, sample count, and the seed for reproducibility.
#[derive(Clone, Debug, PartialEq)]
pub struct ForecastSpec {
    /// Number of steps to forecast.
    pub horizon: usize,
    /// Representations the caller wants back (in addition to the native one).
    pub representations: Vec<Representation>,
    /// Quantile levels for quantile output.
    pub quantile_levels: Vec<f32>,
    /// Number of trajectories for sample output.
    pub num_samples: usize,
    /// Seed for any stochastic path — makes a request reproducible.
    pub seed: u64,
}

impl Default for ForecastSpec {
    fn default() -> Self {
        ForecastSpec {
            horizon: 1,
            representations: vec![Representation::Quantiles],
            quantile_levels: vec![0.1, 0.5, 0.9],
            num_samples: 0,
            seed: 0,
        }
    }
}

/// A structured, wire-mappable forecasting error. `code` is a stable
/// machine-readable slug (never a free-form string a client must parse), and
/// `retryable` tells a client whether the same request could succeed later.
#[derive(Clone, Debug, PartialEq)]
pub struct ForecastError {
    /// Stable slug: `unsupported_capability`, `context_too_long`,
    /// `unknown_model`, `missing_variate`, `bad_request`, `internal`.
    pub code: String,
    /// Human-readable detail.
    pub message: String,
    /// Whether retrying the identical request could plausibly succeed.
    pub retryable: bool,
    /// Optional structured detail for the client (e.g. `{max_context, got}`).
    pub detail: serde_json::Value,
}

impl ForecastError {
    fn of(code: &str, message: impl Into<String>, retryable: bool) -> ForecastError {
        ForecastError {
            code: code.to_string(),
            message: message.into(),
            retryable,
            detail: serde_json::Value::Null,
        }
    }
    /// The model cannot satisfy a requested capability (e.g. quantiles from a
    /// point model). Not retryable.
    pub fn unsupported(message: impl Into<String>) -> ForecastError {
        Self::of("unsupported_capability", message, false)
    }
    /// The context exceeds the model's `max_context`. Not retryable as-is.
    pub fn context_too_long(max: usize, got: usize) -> ForecastError {
        let mut e = Self::of(
            "context_too_long",
            format!("context {got} exceeds max_context {max}"),
            false,
        );
        e.detail = serde_json::json!({ "max_context": max, "got": got });
        e
    }
    /// No model registered under the requested name.
    pub fn unknown_model(name: impl Into<String>) -> ForecastError {
        Self::of("unknown_model", format!("no forecasting model named {}", name.into()), false)
    }
    /// A required variate is absent from the panel.
    pub fn missing_variate(name: impl Into<String>) -> ForecastError {
        Self::of("missing_variate", format!("required variate {} absent", name.into()), false)
    }
    /// The request is malformed (bad shape, NaN, empty panel).
    pub fn bad_request(message: impl Into<String>) -> ForecastError {
        Self::of("bad_request", message, false)
    }
    /// An internal invariant was violated — a bug. Not retryable.
    pub fn internal(message: impl Into<String>) -> ForecastError {
        Self::of("internal", message, false)
    }

    /// Serialize for an `error` event.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        })
    }

    /// Reconstruct from a JSON object produced by [`to_json`](ForecastError::to_json).
    pub fn from_json(v: &serde_json::Value) -> ForecastError {
        ForecastError {
            code: v["code"].as_str().unwrap_or("internal").to_string(),
            message: v["message"].as_str().unwrap_or_default().to_string(),
            retryable: v["retryable"].as_bool().unwrap_or(false),
            detail: v.get("detail").cloned().unwrap_or(serde_json::Value::Null),
        }
    }
}

/// A forecasting model behind an object-safe seam — the fourth peer of
/// `InferModel` / `DetectModel` / `SynthModel` in the runtime. Unlike those,
/// this one is float-in / distribution-out.
///
/// `Send` so a model replica can be owned by a worker thread in the server. It
/// is deliberately *not* `Sync` — brain's models carry per-instance scratch
/// state — so throughput comes from N replicas, one per worker, not shared
/// access.
pub trait ForecastModel: Send {
    /// This model's self-description for capability negotiation.
    fn capabilities(&self) -> Capabilities;

    /// Forecast `panel` over `spec.horizon`, returning the native representation
    /// plus any derivable ones the caller requested. Validates the request
    /// against [`capabilities`](ForecastModel::capabilities) and errors rather
    /// than silently degrading.
    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError>;

    /// Validate a panel + spec against this model's capabilities. The default
    /// checks context length, required variates, and covariate/known-future
    /// support; a model may override to add its own checks. Called by the server
    /// before `forecast`.
    fn validate(&self, panel: &Panel, spec: &ForecastSpec) -> Result<(), ForecastError> {
        let caps = self.capabilities();
        if panel.items.is_empty() {
            return Err(ForecastError::bad_request("empty panel"));
        }
        let ctx = panel.max_context_len();
        if ctx > caps.max_context {
            return Err(ForecastError::context_too_long(caps.max_context, ctx));
        }
        if let Some(maxh) = caps.max_horizon {
            if spec.horizon > maxh {
                return Err(ForecastError::unsupported(format!(
                    "horizon {} exceeds max_horizon {}",
                    spec.horizon, maxh
                )));
            }
        }
        for req in &caps.requires_variates {
            let ok = panel.items.iter().all(|it| it.variate(req).is_some());
            if !ok {
                return Err(ForecastError::missing_variate(req));
            }
        }
        if caps.covariates == CovariateSupport::None && panel.has_covariates() {
            // A no-covariate model must not silently ignore covariates the user
            // supplied under the impression they matter.
            return Err(ForecastError::unsupported(format!(
                "{} does not consume covariates",
                caps.name
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_wire_tags_roundtrip() {
        for r in [
            Representation::Quantiles,
            Representation::Samples,
            Representation::Distribution,
            Representation::Point,
            Representation::Classes,
        ] {
            assert_eq!(Representation::parse(r.as_str()), Some(r));
        }
    }

    #[test]
    fn error_codes_are_stable_slugs() {
        assert_eq!(ForecastError::context_too_long(512, 900).code, "context_too_long");
        assert_eq!(ForecastError::unknown_model("x").code, "unknown_model");
        assert_eq!(ForecastError::missing_variate("close").code, "missing_variate");
        // detail is machine-readable, not just a string
        let e = ForecastError::context_too_long(512, 900);
        assert_eq!(e.detail["max_context"], 512);
        assert_eq!(e.detail["got"], 900);
    }

    // A trivial point-forecast model to exercise the default validate().
    struct FakePoint;
    impl ForecastModel for FakePoint {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                name: "fake_point".into(),
                max_context: 64,
                max_horizon: Some(10),
                native_representation: Representation::Point,
                covariates: CovariateSupport::None,
                supports_known_future: false,
                multivariate: false,
                arbitrary_quantile_levels: false,
                stochastic: false,
                requires_variates: vec![],
            }
        }
        fn forecast(
            &self,
            panel: &Panel,
            spec: &ForecastSpec,
        ) -> Result<Forecast, ForecastError> {
            self.validate(panel, spec)?;
            let mut fc = Forecast::new("fake_point", Representation::Point, spec.horizon, &panel.freq);
            for it in &panel.items {
                for tgt in it.targets() {
                    let last = *tgt.data.last().unwrap_or(&0.0);
                    let mut tf = TargetForecast::new(&it.item_id, &tgt.name);
                    tf.mean = Some(Block::native(vec![spec.horizon], vec![last; spec.horizon]));
                    fc.targets.push(tf);
                }
            }
            Ok(fc)
        }
    }

    #[test]
    fn default_validate_rejects_overlong_context() {
        let m = FakePoint;
        let panel = Panel::single("1d", "X", vec![Variate::target("close", vec![1.0; 100])]);
        let err = m.validate(&panel, &ForecastSpec::default()).unwrap_err();
        assert_eq!(err.code, "context_too_long");
    }

    #[test]
    fn default_validate_rejects_covariates_for_a_no_covariate_model() {
        let m = FakePoint;
        let panel = Panel::single(
            "1d",
            "X",
            vec![
                Variate::target("close", vec![1.0; 10]),
                Variate {
                    name: "vix".into(),
                    role: Role::PastCovariate,
                    kind: Kind::Continuous,
                    data: vec![1.0; 10],
                    future: None,
                    observed: None,
                    cardinality: None,
                },
            ],
        );
        let err = m.validate(&panel, &ForecastSpec::default()).unwrap_err();
        assert_eq!(err.code, "unsupported_capability");
    }

    #[test]
    fn fake_point_model_forecasts_last_value() {
        let m = FakePoint;
        let panel = Panel::single("1d", "X", vec![Variate::target("close", vec![1.0, 2.0, 3.0])]);
        let spec = ForecastSpec { horizon: 4, ..Default::default() };
        let fc = m.forecast(&panel, &spec).unwrap();
        assert_eq!(fc.native_representation, Representation::Point);
        assert_eq!(fc.targets.len(), 1);
        let mean = fc.targets[0].mean.as_ref().unwrap();
        assert_eq!(mean.data, vec![3.0, 3.0, 3.0, 3.0]);
    }
}
