// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`forecast::ForecastModel`] adapter — makes a loaded [`Fincast`] drivable
//! through the whole forecasting API (CLI, server, comparison harness).
//!
//! FinCast emits **9 fixed quantile levels** natively (`[0.1..0.9]`) plus a mean.
//! The adapter runs the forward once, then serves the caller's requested levels
//! by interpolating across those 9 (monotone in level); requested levels that
//! coincide with a native level are exact. Native representation is `Quantiles`;
//! the honesty layer derives samples / point / etc. from there on request.
//!
//! FinCast is univariate (context + a frequency bucket), so it advertises no
//! covariate support. Routing is deterministic top-2, so `stochastic = false`.

use crate::config::QUANTILES;
use crate::Fincast;
use forecast::{
    convert, Block, Capabilities, CovariateSupport, Forecast, ForecastError, ForecastModel, ForecastSpec,
    Panel, Representation, TargetForecast,
};

/// A [`Fincast`] behind the object-safe [`ForecastModel`] seam.
pub struct FincastForecaster {
    model: Fincast,
    version: String,
}

/// Map a pandas-style frequency string to FinCast's bucket (0 high / 1 med /
/// 2 low), per the reference `ffm_base.freq_map`.
fn freq_bucket(freq: &str) -> usize {
    let f = freq.to_uppercase();
    let f = f.trim_start_matches(|c: char| c.is_ascii_digit());
    if f.starts_with("MS") {
        1
    } else if f.starts_with(['H', 'T', 'D', 'B', 'U', 'S']) || f.starts_with("MIN") {
        0
    } else if f.starts_with(['W', 'M']) {
        1
    } else if f.starts_with(['Y', 'Q', 'A']) {
        2
    } else {
        0
    }
}

impl FincastForecaster {
    pub fn new(model: Fincast) -> FincastForecaster {
        FincastForecaster { model, version: "Vincent05R/FinCast".into() }
    }

    /// Load from a brain `.weights` container.
    pub fn load(path: &str) -> Result<FincastForecaster, String> {
        Ok(FincastForecaster::new(Fincast::load(path)?))
    }

    /// Interpolate the requested `levels` from the native 9-quantile matrix
    /// `native` (`[9, horizon]`, quantile-major). Returns `[horizon, n_levels]`.
    fn interp_levels(native: &[f32], horizon: usize, levels: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; horizon * levels.len()];
        for t in 0..horizon {
            for (j, &lv) in levels.iter().enumerate() {
                out[t * levels.len() + j] = interp_one(native, horizon, t, lv);
            }
        }
        out
    }
}

/// One interpolated quantile at step `t`, level `lv`, from the native 9-grid.
fn interp_one(native: &[f32], horizon: usize, t: usize, lv: f32) -> f32 {
    let nq = QUANTILES.len();
    let at = |k: usize| native[k * horizon + t];
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

impl ForecastModel for FincastForecaster {
    fn capabilities(&self) -> Capabilities {
        let cfg = self.model.config();
        Capabilities {
            name: "fincast".into(),
            max_context: cfg.context_len,
            // AR-decode extends beyond one patch; advertise a generous cap.
            max_horizon: Some(cfg.horizon_len * 8),
            native_representation: Representation::Quantiles,
            covariates: CovariateSupport::None,
            supports_known_future: false,
            multivariate: false,
            arbitrary_quantile_levels: true,
            stochastic: false, // deterministic top-2 MoE
            requires_variates: vec![],
        }
    }

    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        let mut fc = Forecast::new("fincast", Representation::Quantiles, spec.horizon, &panel.freq);
        fc.model_version = self.version.clone();
        let freq = freq_bucket(&panel.freq);

        let levels = if spec.quantile_levels.is_empty() {
            vec![0.1, 0.5, 0.9]
        } else {
            spec.quantile_levels.clone()
        };

        for item in &panel.items {
            for tgt in item.targets() {
                let native = self.model.forecast_quantiles(&tgt.data, freq, spec.horizon); // [9, horizon]
                if native.len() != QUANTILES.len() * spec.horizon {
                    return Err(ForecastError::internal("fincast: unexpected forecast shape"));
                }
                let q = FincastForecaster::interp_levels(&native, spec.horizon, &levels);

                let mut tf = TargetForecast::new(&item.item_id, &tgt.name);
                tf.levels = levels.clone();
                tf.quantiles = Some(Block::native(vec![spec.horizon, levels.len()], q));
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
        Ok(fc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FincastConfig;
    use std::collections::HashMap;

    fn zero_model() -> Fincast {
        let cfg = FincastConfig::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        Fincast::from_weights(cfg, &weights).unwrap()
    }

    #[test]
    fn freq_bucket_maps_common_frequencies() {
        assert_eq!(freq_bucket("1d"), 0);
        assert_eq!(freq_bucket("1h"), 0);
        assert_eq!(freq_bucket("1min"), 0);
        assert_eq!(freq_bucket("1W"), 1);
        assert_eq!(freq_bucket("1M"), 1);
        assert_eq!(freq_bucket("1Q"), 2);
        assert_eq!(freq_bucket("1Y"), 2);
    }

    #[test]
    fn capabilities_advertise_fincast() {
        let f = FincastForecaster::new(zero_model());
        let c = f.capabilities();
        assert_eq!(c.name, "fincast");
        assert_eq!(c.native_representation, Representation::Quantiles);
        assert_eq!(c.max_context, 32); // tiny context_len
        assert!(!c.stochastic);
        assert!(c.arbitrary_quantile_levels);
        assert!(matches!(c.covariates, CovariateSupport::None));
    }

    #[test]
    fn interp_picks_native_and_interpolates() {
        // native [9, horizon=1]: value at level index k is k.
        let native: Vec<f32> = (0..QUANTILES.len()).map(|k| k as f32).collect();
        let q = FincastForecaster::interp_levels(&native, 1, &[0.5]); // idx 4 -> 4.0
        assert!((q[0] - 4.0).abs() < 1e-4, "{q:?}");
        let mid = (0.5 + 0.6) / 2.0; // between idx 4 and 5
        let q2 = FincastForecaster::interp_levels(&native, 1, &[mid]);
        assert!((q2[0] - 4.5).abs() < 1e-3, "{q2:?}");
    }

    #[test]
    fn forecast_through_the_seam_returns_quantiles() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        use forecast::Variate;
        let f = FincastForecaster::new(zero_model());
        let ctx: Vec<f32> = (0..32).map(|i| 5.0 + i as f32 * 0.2).collect();
        let panel = Panel::single("1d", "X", vec![Variate::target("y", ctx)]);
        let spec = ForecastSpec {
            horizon: 4,
            representations: vec![Representation::Quantiles, Representation::Point],
            quantile_levels: vec![0.1, 0.5, 0.9],
            num_samples: 0,
            seed: 0,
        };
        let out = f.forecast(&panel, &spec).unwrap();
        assert_eq!(out.model, "fincast");
        let tf = &out.targets[0];
        let q = tf.quantiles.as_ref().unwrap();
        assert_eq!(q.shape, vec![4, 3]);
        assert!(q.data.iter().all(|v| v.is_finite()));
        assert!(tf.mean.is_some());
    }
}
