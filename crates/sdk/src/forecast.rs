// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`ForecastPipeline`]: brain's time-series forecasting surface, over two
//! architectures behind one public type - kronos (BSQ-tokenizer candlestick
//! model, two roles: tokenizer + decoder) and timesfm3 (Google's TimesFM-3,
//! one role: weights). Both resolve through `crates/loader`'s resolver - the
//! SAME resolver `brain kronos`/`brain timesfm3` and `ImagePipeline` use -
//! so a checkpoint that resolves there resolves here too, with no
//! environment variable and no CLI process in the loop.
//!
//! chronos2 and fincast are a real, tracked gap, not silently unsupported:
//! neither has a model-store resolver `ArchSpec` or a `default_ref`
//! registered anywhere in this workspace yet (both load from a plain
//! `BRAIN_CHRONOS2`/`BRAIN_FINCAST` path today), so there is nothing for
//! this pipeline's resolver to resolve them against.
//!
//! Once resolved, every architecture behind this pipeline is uniform: all
//! four forecasting models (including the two not wired in here yet)
//! already share ONE object-safe trait, `forecast::ForecastModel` - unlike
//! [`crate::ImagePipeline`]'s two structurally different backends, there is
//! no per-architecture match arm anywhere past construction.
//!
//! ```no_run
//! let pipe = brain::ForecastPipeline::from_pretrained("NeoQuasar/Kronos-base")?;
//! let series: Vec<f32> = vec![100.0, 101.2, 99.8, 102.5];
//! let forecast = pipe.forecast(&series, 24)?;
//! println!("{} steps ahead, {} target(s)", forecast.horizon, forecast.targets.len());
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use ::forecast::ForecastModel;

use crate::{Device, Error, Result};

/// `brain`'s time-series forecasting pipeline. See this module's doc for
/// which architectures resolve today.
pub struct ForecastPipeline {
    model: Box<dyn ForecastModel>,
}

/// Hand-written, not derived: `Box<dyn ForecastModel>` carries no `Debug`
/// bound (the trait doesn't require one - a model may hold live GPU
/// handles, the same reason `ImagePipeline` isn't derived either). A short
/// summary is still worth printing.
impl std::fmt::Debug for ForecastPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForecastPipeline").field("model", &self.capabilities().name).finish()
    }
}

impl ForecastPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<ForecastPipeline> {
        ForecastPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> ForecastPipelineBuilder {
        ForecastPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default() }
    }

    /// This model's capability-negotiation self-description: context/horizon
    /// limits, the representation it emits natively, covariate support.
    pub fn capabilities(&self) -> ::forecast::Capabilities {
        self.model.capabilities()
    }

    /// Forecast one target series `horizon` steps ahead, at every other
    /// default (`[0.1, 0.5, 0.9]` quantile levels - see
    /// [`::forecast::ForecastSpec::default`]). See
    /// [`ForecastPipeline::forecast_with`] for a multi-item, multi-variate
    /// panel, or to request samples/a distribution instead of quantiles.
    pub fn forecast(&self, series: &[f32], horizon: usize) -> Result<::forecast::Forecast> {
        let panel = ::forecast::Panel::single("", "series", vec![::forecast::Variate::target("target", series.to_vec())]);
        let spec = ::forecast::ForecastSpec { horizon, ..::forecast::ForecastSpec::default() };
        self.forecast_with(&panel, &spec)
    }

    /// The full-control entry point [`ForecastPipeline::forecast`] delegates
    /// to: a multi-item, multi-variate [`::forecast::Panel`] and every
    /// [`::forecast::ForecastSpec`] knob (representations, quantile levels,
    /// sample count, seed). Validated against [`ForecastPipeline::capabilities`]
    /// before running - an unsupported request (context too long, a missing
    /// required variate, an unsupported representation) is a named
    /// [`Error::Forecast`], never a silently degraded result.
    pub fn forecast_with(&self, panel: &::forecast::Panel, spec: &::forecast::ForecastSpec) -> Result<::forecast::Forecast> {
        self.model.validate(panel, spec).map_err(Error::from)?;
        self.model.forecast(panel, spec).map_err(Error::from)
    }
}

/// Which architecture [`ForecastPipelineBuilder::load`] resolved to - unlike
/// [`crate::ImagePipeline`]'s `Backend`, this exists only to carry the
/// resolved [`capability::Assembly`] through construction; every method past
/// [`ForecastPipelineBuilder::load`] is uniform across both (see this
/// module's doc).
enum ResolvedArch {
    Kronos(capability::Assembly),
    Timesfm3(capability::Assembly),
}

/// Resolve `overrides` against BOTH known forecasting architectures with a
/// real resolver - kronos first (an arbitrary tie-break, same reason
/// [`crate::pipeline::resolve_arch`] tries flux2 first: it was the first one
/// wired in), then timesfm3 only when kronos did not resolve. See this
/// module's doc for why chronos2/fincast are not tried here at all.
fn resolve_arch(overrides: &BTreeMap<String, String>, model_id: &str, download_policy: loader::DownloadPolicy) -> Result<ResolvedArch> {
    let resolved =
        crate::resolve_policy::resolve_two_with_policy("kronos", &kronos::spec::KronosSpec, "timesfm3", &timesfm3::spec::Timesfm3Spec, model_id, overrides, download_policy)?;
    Ok(match resolved {
        crate::resolve_policy::Resolved2::A(a) => ResolvedArch::Kronos(a),
        crate::resolve_policy::Resolved2::B(a) => ResolvedArch::Timesfm3(a),
    })
}

/// Builds a [`ForecastPipeline`]. `.device(...)` is the only knob this
/// milestone exposes; every other decision follows each architecture's own
/// defaults.
pub struct ForecastPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
}

impl ForecastPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How [`ForecastPipelineBuilder::load`] may use the network to resolve
    /// `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`] -- see
    /// that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`ForecastPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. [`resolve_arch`] (via [`crate::resolve_policy::resolve_two_with_policy`])
    ///    tries the resolver against each known forecasting architecture's
    ///    roles FIRST; only on `Missing` does `model_id` get parsed and,
    ///    under [`ForecastPipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched - then resolution
    ///    is retried once. kronos/timesfm3 both have working recipes, so
    ///    unlike `crate::tts::TtsPipelineBuilder::load`'s own reorder this
    ///    closes no live bug, only unifies onto one resolve strategy.
    /// 3. The resolved architecture's own `Forecaster::load` builds the
    ///    model: kronos from its two resolved roles (`tokenizer`, `decoder`),
    ///    timesfm3 from its one (`weights`).
    ///
    /// Every failure path returns a typed [`Error`] - never a panic on a
    /// caller-reachable input.
    pub fn load(self) -> Result<ForecastPipeline> {
        let ForecastPipelineBuilder { model_id, device, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        match resolve_arch(&overrides, &model_id, download_policy)? {
            ResolvedArch::Kronos(assembly) => {
                let tok = assembly.roles.get("tokenizer").ok_or_else(|| Error::Backend(format!("kronos: resolved assembly {:?} has no tokenizer role", assembly.id)))?;
                let dec = assembly.roles.get("decoder").ok_or_else(|| Error::Backend(format!("kronos: resolved assembly {:?} has no decoder role", assembly.id)))?;
                let model = kronos::KronosForecaster::load(&tok.to_string_lossy(), &dec.to_string_lossy()).map_err(Error::Backend)?;
                Ok(ForecastPipeline { model: Box::new(model) })
            }
            ResolvedArch::Timesfm3(assembly) => {
                let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("timesfm3: resolved assembly {:?} has no weights role", assembly.id)))?;
                let model = timesfm3::Timesfm3Forecaster::load(&weights.to_string_lossy()).map_err(Error::Backend)?;
                Ok(ForecastPipeline { model: Box::new(model) })
            }
        }
    }
}
