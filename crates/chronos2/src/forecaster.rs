// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`forecast::ForecastModel`] adapter — makes a loaded [`Chronos2`] drivable
//! through the whole forecasting API (CLI, server, comparison harness).
//!
//! Chronos-2 emits **21 fixed quantile levels** natively. The adapter runs the
//! forward once, then serves the caller's requested levels by interpolating
//! across those 21 (they are monotone in level); requested levels that coincide
//! with a native level are exact. The native representation is `Quantiles`, so
//! the honesty layer derives samples / point / etc. from there on request.
//!
//! Phase 1 is univariate: it forecasts the first target series and advertises no
//! covariate support (`CovariateSupport::None`). Multivariate + covariates land
//! with the group-attention path in Phase 2.

use crate::config::QUANTILES;
use crate::Chronos2;
use forecast::{
    convert, Block, Capabilities, CovariateSupport, Forecast, ForecastError, ForecastModel,
    ForecastSpec, Panel, Role, Representation, TargetForecast,
};

/// A [`Chronos2`] behind the object-safe [`ForecastModel`] seam.
pub struct Chronos2Forecaster {
    model: Chronos2,
    version: String,
}

impl Chronos2Forecaster {
    pub fn new(model: Chronos2) -> Chronos2Forecaster {
        Chronos2Forecaster { model, version: "amazon/chronos-2".into() }
    }

    /// Load from a brain `.weights` container.
    pub fn load(path: &str) -> Result<Chronos2Forecaster, String> {
        Ok(Chronos2Forecaster::new(Chronos2::load(path)?))
    }

    /// Interpolate the requested `levels` from the native 21-quantile matrix
    /// `native` (`[21, horizon]`, quantile-major). Returns `[horizon, n_levels]`
    /// (step-major) — the layout the API's quantile block expects.
    fn interp_levels(native: &[f32], horizon: usize, levels: &[f32]) -> Vec<f32> {
        let nq = QUANTILES.len();
        let mut out = vec![0.0f32; horizon * levels.len()];
        for t in 0..horizon {
            for (j, &lv) in levels.iter().enumerate() {
                out[t * levels.len() + j] = interp_one(native, horizon, nq, t, lv);
            }
        }
        out
    }
}

/// One interpolated quantile at step `t`, level `lv`, from the native grid.
fn interp_one(native: &[f32], horizon: usize, nq: usize, t: usize, lv: f32) -> f32 {
    let at = |k: usize| native[k * horizon + t]; // native is [nq, horizon]
    if lv <= QUANTILES[0] {
        return at(0);
    }
    if lv >= QUANTILES[nq - 1] {
        return at(nq - 1);
    }
    for k in 1..nq {
        if lv <= QUANTILES[k] {
            let span = QUANTILES[k] - QUANTILES[k - 1];
            let frac = if span > 0.0 { (lv - QUANTILES[k - 1]) / span } else { 0.0 };
            return at(k - 1) + (at(k) - at(k - 1)) * frac;
        }
    }
    at(nq - 1)
}

impl ForecastModel for Chronos2Forecaster {
    fn capabilities(&self) -> Capabilities {
        let cfg = self.model.config();
        Capabilities {
            name: "chronos2".into(),
            max_context: cfg.context_length,
            max_horizon: Some(cfg.max_output_patches * cfg.output_patch_size),
            native_representation: Representation::Quantiles,
            // Past + known-future covariates enter via the group-attention path;
            // known-future variates additionally supply their horizon values.
            covariates: CovariateSupport::Full,
            supports_known_future: true,
            multivariate: false,
            arbitrary_quantile_levels: true, // served by interpolating the 21 native
            stochastic: false,
            requires_variates: vec![],
        }
    }

    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        let cfg = self.model.config();
        let mut fc = Forecast::new("chronos2", Representation::Quantiles, spec.horizon, &panel.freq);
        fc.model_version = self.version.clone();

        let levels = if spec.quantile_levels.is_empty() {
            vec![0.1, 0.5, 0.9]
        } else {
            spec.quantile_levels.clone()
        };

        for item in &panel.items {
            // Covariates (past + known-future) inform every target via group
            // attention. A known-future covariate additionally carries its future
            // path, which enters the future patches; a past covariate's future is
            // unknown (masked). Validated for length below.
            let covs: Vec<&forecast::Variate> = item
                .variates
                .iter()
                .filter(|v| matches!(v.role, Role::PastCovariate | Role::KnownFuture))
                .collect();
            for tgt in item.targets() {
                // native [21, horizon], quantile-major. With covariates, run the
                // multivariate group-attention path (target + covariates in one
                // group); otherwise the univariate path.
                let native = if covs.is_empty() {
                    self.model.forecast_quantiles(&tgt.data, spec.horizon)
                } else {
                    let mut series: Vec<&[f32]> = Vec::with_capacity(covs.len() + 1);
                    let mut futures: Vec<Option<&[f32]>> = Vec::with_capacity(covs.len() + 1);
                    series.push(&tgt.data);
                    futures.push(None); // the target's future is what we predict
                    for c in &covs {
                        series.push(&c.data);
                        match c.role {
                            Role::KnownFuture => {
                                let fv = c.future.as_deref().ok_or_else(|| {
                                    ForecastError::bad_request(
                                        "chronos2: known_future covariate is missing its future path",
                                    )
                                })?;
                                if fv.len() != spec.horizon {
                                    return Err(ForecastError::bad_request(
                                        "chronos2: known_future length must equal the horizon",
                                    ));
                                }
                                futures.push(Some(fv));
                            }
                            _ => futures.push(None),
                        }
                    }
                    if series.iter().any(|s| s.len() != tgt.data.len()) {
                        return Err(ForecastError::bad_request(
                            "chronos2: target and covariates must share context length",
                        ));
                    }
                    self.model.forecast_quantiles_mv_kf(&series, &futures, spec.horizon)
                };
                if native.len() != QUANTILES.len() * spec.horizon {
                    return Err(ForecastError::internal("chronos2: unexpected forecast shape"));
                }
                let q = Chronos2Forecaster::interp_levels(&native, spec.horizon, &levels);

                let mut tf = TargetForecast::new(&item.item_id, &tgt.name);
                tf.levels = levels.clone();
                tf.quantiles = Some(Block::native(vec![spec.horizon, levels.len()], q));
                // derive any other requested representations (samples/point/…)
                convert::ensure_representations(
                    &mut tf,
                    Representation::Quantiles,
                    &spec.representations,
                    &levels,
                    spec.num_samples,
                    spec.seed,
                )?;
                fc.targets.push(tf);
            }
        }
        // guard the context length explicitly (validate already did, but the
        // model would silently truncate otherwise).
        if panel.max_context_len() > cfg.context_length {
            return Err(ForecastError::context_too_long(cfg.context_length, panel.max_context_len()));
        }
        Ok(fc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Chronos2Config;
    use forecast::Variate;
    use std::collections::HashMap;

    fn zero_model() -> Chronos2 {
        let cfg = Chronos2Config::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        Chronos2::from_weights_on(gpu_core::testgpu::dev(crate::model::PIPELINES), cfg, &weights).unwrap()
    }

    #[test]
    fn capabilities_advertise_chronos2() {
        let f = Chronos2Forecaster::new(zero_model());
        let c = f.capabilities();
        assert_eq!(c.name, "chronos2");
        assert_eq!(c.native_representation, Representation::Quantiles);
        // the tiny config used in tests: context 64, horizon = 8*4 = 32.
        assert_eq!(c.max_context, 64);
        assert_eq!(c.max_horizon, Some(32));
        assert!(c.arbitrary_quantile_levels);
    }

    #[test]
    fn default_config_advertises_published_limits() {
        // build the capabilities from a default-config model's cfg without
        // allocating 120M zero weights: check the mapping directly.
        let cfg = Chronos2Config::default();
        assert_eq!(cfg.context_length, 8192);
        assert_eq!(cfg.max_output_patches * cfg.output_patch_size, 1024);
    }

    #[test]
    fn interp_levels_picks_native_and_interpolates() {
        // native [21, horizon=1]: value at level index k is k (so quantile k = k).
        let horizon = 1;
        let native: Vec<f32> = (0..QUANTILES.len()).map(|k| k as f32).collect();
        // exact native level 0.5 is index 10 -> value 10
        let q = Chronos2Forecaster::interp_levels(&native, horizon, &[0.5]);
        assert!((q[0] - 10.0).abs() < 1e-4, "{q:?}");
        // a level between 0.5 (idx10) and 0.55 (idx11) interpolates between 10 and 11
        let mid = (0.5 + 0.55) / 2.0;
        let q2 = Chronos2Forecaster::interp_levels(&native, horizon, &[mid]);
        assert!((q2[0] - 10.5).abs() < 1e-3, "{q2:?}");
    }

    #[test]
    fn forecast_through_the_seam_returns_quantiles() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let f = Chronos2Forecaster::new(zero_model());
        let ctx: Vec<f32> = (0..30).map(|i| 5.0 + i as f32 * 0.2).collect();
        let mean = ctx.iter().sum::<f32>() / ctx.len() as f32;
        let panel = Panel::single("1d", "X", vec![Variate::target("y", ctx)]);
        let spec = ForecastSpec {
            horizon: 4,
            representations: vec![Representation::Quantiles, Representation::Point],
            quantile_levels: vec![0.1, 0.5, 0.9],
            num_samples: 0,
            seed: 0,
        };
        let out = f.forecast(&panel, &spec).unwrap();
        assert_eq!(out.model, "chronos2");
        let tf = &out.targets[0];
        let q = tf.quantiles.as_ref().unwrap();
        assert_eq!(q.shape, vec![4, 3]);
        // zero-weight model -> every quantile is the series mean
        assert!(q.data.iter().all(|&v| (v - mean).abs() < 1e-2), "quantiles should be the mean");
        // point representation derived
        assert!(tf.mean.is_some());
    }
}
