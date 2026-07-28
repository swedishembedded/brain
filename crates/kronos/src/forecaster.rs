// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`forecast::ForecastModel`] adapter — makes a loaded [`KronosModel`] drivable
//! through the whole forecasting API (CLI, server, comparison harness), beside
//! Chronos-2 and the statistical baselines.
//!
//! Kronos is finance-native: it consumes an OHLCV(+amount) bar panel and emits
//! **samples** (its AR rollout is stochastic). The adapter draws
//! `spec.num_samples` future-bar trajectories (a floor of a few so a
//! distribution can be derived even when the caller only asked for a point), and
//! returns the target column's paths as the native `Samples` block; the honesty
//! layer derives quantiles / mean / intervals from there on request.
//!
//! Inputs are read **by name** (`open,high,low,close,volume`), independent of
//! their `Role`, so a caller can mark `close` the `Target` and the others
//! `PastCovariate`. If the tokenizer wants a sixth feature (`amount`/turnover),
//! it is synthesised as `volume × close` — Yahoo & most feeds don't ship it.
//!
//! Calendar: Kronos consumes per-bar time features (minute/hour/weekday/day/month,
//! the reference `calc_time_stamps`). Since trading bars skip weekends/holidays,
//! a bar index can't be mapped to a date from `start`+`freq` alone, so the caller
//! supplies the calendar explicitly as five variates named `minute`, `hour`,
//! `weekday`, `day`, `month` (context values in `data`, the horizon values in
//! `future` — a known-future covariate). When present the adapter feeds the real
//! stamp; when absent it runs **calendar-agnostic** (zero stamps), so an
//! OHLCV-only panel still forecasts (a touch more extreme than the reference).

use crate::generate::{GenOpts, KronosModel};
use crate::import;
use forecast::{
    Block, Capabilities, CovariateSupport, Forecast, ForecastError, ForecastModel, ForecastSpec,
    Panel, Representation, TargetForecast,
};

/// Fixed input column order the adapter builds bars in.
const OHLCV: [&str; 5] = ["open", "high", "low", "close", "volume"];

/// Calendar variate names, in the reference's stamp order
/// (`decoder::CAL`): minute, hour, weekday, day, month.
const CAL_NAMES: [&str; 5] = ["minute", "hour", "weekday", "day", "month"];

/// Sample-trajectory floor: even a point/quantile-only request draws at least
/// this many paths so a distribution is available to derive from.
const SAMPLE_FLOOR: usize = 16;

/// A [`KronosModel`] behind the object-safe [`ForecastModel`] seam.
pub struct KronosForecaster {
    model: KronosModel,
    version: String,
}

impl KronosForecaster {
    pub fn new(model: KronosModel) -> KronosForecaster {
        KronosForecaster { model, version: "NeoQuasar/Kronos-small".into() }
    }

    /// Load from the two HF checkpoint dirs (tokenizer + decoder).
    pub fn load(tokenizer_dir: &str, decoder_dir: &str) -> Result<KronosForecaster, String> {
        Ok(KronosForecaster::new(import::load_model(tokenizer_dir, decoder_dir)?))
    }

    /// Assemble a `[T, feat]` row-major bar matrix from the item's OHLCV
    /// variates, synthesising `amount = volume × close` if the model wants 6
    /// features. Errors if a required column is missing or ragged.
    fn build_bars(&self, item: &forecast::Item) -> Result<(Vec<f32>, usize), ForecastError> {
        let feat = self.model.feat();
        let cols: Vec<&[f32]> = OHLCV
            .iter()
            .map(|n| item.variate(n).map(|v| v.data.as_slice()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ForecastError::missing_variate("one of open/high/low/close/volume"))?;
        let t = cols[0].len();
        if t == 0 || cols.iter().any(|c| c.len() != t) {
            return Err(ForecastError::bad_request("kronos: OHLCV columns must be non-empty and equal length"));
        }
        let mut bars = vec![0.0f32; t * feat];
        for r in 0..t {
            for (c, col) in cols.iter().enumerate() {
                bars[r * feat + c] = col[r];
            }
            if feat == 6 {
                // amount / turnover = volume × mean(open,high,low,close), matching
                // the reference KronosPredictor's synthesis when amount is absent.
                let mean_ohlc = (cols[0][r] + cols[1][r] + cols[2][r] + cols[3][r]) * 0.25;
                bars[r * feat + 5] = cols[4][r] * mean_ohlc;
            }
        }
        if feat != 5 && feat != 6 {
            return Err(ForecastError::internal(format!("kronos: unexpected d_in {feat}")));
        }
        Ok((bars, feat))
    }

    /// Build the `(ctx_stamp, fut_stamp)` calendar index streams from the item's
    /// calendar variates (`minute`/`hour`/`weekday`/`day`/`month`), matching the
    /// reference `calc_time_stamps`. The future stamp comes from each variate's
    /// `future` path (a known-future covariate). If the full calendar is absent,
    /// returns all-zero stamps — the calendar-agnostic default (backward
    /// compatible: a caller that supplies only OHLCV still forecasts).
    fn calendar_stamps(&self, item: &forecast::Item, ctx_len: usize, horizon: usize) -> (Vec<u32>, Vec<u32>) {
        let cal: Option<Vec<&forecast::Variate>> = CAL_NAMES.iter().map(|n| item.variate(n)).collect();
        let Some(cal) = cal else {
            return (vec![0u32; ctx_len * 5], vec![0u32; horizon * 5]);
        };
        let mut ctx_stamp = vec![0u32; ctx_len * 5];
        let mut fut_stamp = vec![0u32; horizon * 5];
        for (c, v) in cal.iter().enumerate() {
            for b in 0..ctx_len.min(v.data.len()) {
                ctx_stamp[b * 5 + c] = v.data[b].max(0.0) as u32;
            }
            if let Some(fut) = &v.future {
                for h in 0..horizon.min(fut.len()) {
                    fut_stamp[h * 5 + c] = fut[h].max(0.0) as u32;
                }
            }
        }
        (ctx_stamp, fut_stamp)
    }
}

impl ForecastModel for KronosForecaster {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: "kronos".into(),
            max_context: self.model.max_context(), // bars, not subtokens
            max_horizon: None,                      // autoregressive rollout
            native_representation: Representation::Samples,
            covariates: CovariateSupport::CalendarOnly,
            supports_known_future: false,
            multivariate: false, // one target column emitted per request
            arbitrary_quantile_levels: true, // any quantile derivable from samples
            stochastic: true,
            requires_variates: OHLCV.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        if spec.horizon == 0 {
            return Err(ForecastError::bad_request("kronos: horizon must be >= 1"));
        }
        // Honor an explicit sample count; only fall back to the floor when the
        // caller asked for none (they still need a distribution to derive from).
        // `KRONOS_ARGMAX=1` selects the deterministic modal rollout (argmax over
        // the token distribution): a single stable draw, matching the reference
        // RankIC evaluation's point path — no sampling noise, and ~N× cheaper.
        let argmax = std::env::var("KRONOS_ARGMAX").map(|v| v != "0").unwrap_or(false);
        let n_samples = if argmax {
            1
        } else if spec.num_samples == 0 {
            SAMPLE_FLOOR
        } else {
            spec.num_samples
        };
        let mut fc = Forecast::new("kronos", Representation::Samples, spec.horizon, &panel.freq);
        fc.model_version = self.version.clone();

        for item in &panel.items {
            let (bars, feat) = self.build_bars(item)?;
            // Real calendar if the caller supplied minute/hour/weekday/day/month
            // variates; otherwise zeros (calendar-agnostic).
            let (ctx_stamp, fut_stamp) = self.calendar_stamps(item, bars.len() / feat, spec.horizon);

            // Draw n_samples stochastic AR trajectories of the full bar; keep each
            // target column's path. targets default to `close` if the caller
            // marked none (Kronos always has a close).
            let mut target_names: Vec<String> =
                item.targets().map(|t| t.name.clone()).filter(|n| OHLCV.contains(&n.as_str())).collect();
            if target_names.is_empty() {
                target_names.push("close".into());
            }
            let cols: Vec<usize> = target_names
                .iter()
                .map(|n| OHLCV.iter().position(|c| c == n).unwrap_or(3))
                .collect();

            // samples[k][col_i] = trajectory of length horizon for target col_i
            let mut per_target: Vec<Vec<f32>> = vec![Vec::with_capacity(n_samples * spec.horizon); cols.len()];
            for k in 0..n_samples {
                let opts = GenOpts {
                    temperature: 1.0,
                    top_k: 0,
                    // nucleus truncation matching the reference KronosPredictor
                    // default; top_p=1.0 (no truncation) samples the full tails and
                    // makes the rollout wildly over-dispersed on real data.
                    top_p: 0.9,
                    argmax,
                    seed: spec.seed.wrapping_add(k as u64),
                };
                // KV-cached rollout: exact-parity with the un-cached path but
                // prefills the context once and advances one token at a time.
                let path = self.model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, spec.horizon, &opts);
                // path is [horizon, feat]
                for (ci, &col) in cols.iter().enumerate() {
                    for h in 0..spec.horizon {
                        per_target[ci].push(path[h * feat + col]);
                    }
                }
            }

            for (ci, name) in target_names.iter().enumerate() {
                let mut tf = TargetForecast::new(&item.item_id, name);
                tf.samples = Some(Block::native(vec![n_samples, spec.horizon], per_target[ci].clone()));
                forecast::convert::ensure_representations(
                    &mut tf,
                    Representation::Samples,
                    &spec.representations,
                    &if spec.quantile_levels.is_empty() { vec![0.1, 0.5, 0.9] } else { spec.quantile_levels.clone() },
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
    use crate::config::{KronosConfig, KronosTokenizerConfig};
    use forecast::{Role, Variate};
    use std::collections::HashMap;

    fn zero_model() -> KronosModel {
        let tc = KronosTokenizerConfig::tiny();
        let dc = KronosConfig::tiny();
        let tw: HashMap<String, Vec<f32>> =
            tc.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        let dw: HashMap<String, Vec<f32>> =
            dc.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        KronosModel::from_weights_on(gpu_core::testgpu::dev(crate::nn::PIPELINES), tc, &tw, dc, &dw).unwrap()
    }

    fn ohlcv_panel(t: usize) -> Panel {
        let mk = |name: &str, role: Role, base: f32| Variate {
            name: name.into(),
            role,
            kind: forecast::Kind::Continuous,
            data: (0..t).map(|i| base + (i as f32 * 0.1).sin()).collect(),
            future: None,
            observed: None,
            cardinality: None,
        };
        Panel::single(
            "1d",
            "MDB",
            vec![
                mk("open", Role::PastCovariate, 100.0),
                mk("high", Role::PastCovariate, 101.0),
                mk("low", Role::PastCovariate, 99.0),
                mk("close", Role::Target, 100.0),
                mk("volume", Role::PastCovariate, 1000.0),
            ],
        )
    }

    #[test]
    fn capabilities_advertise_kronos() {
        let f = KronosForecaster::new(zero_model());
        let c = f.capabilities();
        assert_eq!(c.name, "kronos");
        assert_eq!(c.native_representation, Representation::Samples);
        assert!(c.stochastic);
        assert_eq!(c.requires_variates, vec!["open", "high", "low", "close", "volume"]);
        assert_eq!(c.covariates, CovariateSupport::CalendarOnly);
    }

    #[test]
    fn calendar_stamps_interleave_the_five_features() {
        let f = KronosForecaster::new(zero_model());
        let t = 4usize;
        let horizon = 2usize;
        let cal = |name: &str, data: Vec<f32>, future: Vec<f32>| Variate {
            name: name.into(),
            role: Role::KnownFuture,
            kind: forecast::Kind::Categorical,
            data,
            future: Some(future),
            observed: None,
            cardinality: None,
        };
        // minute,hour,weekday,day,month — reference stamp order.
        let item = forecast::Item::new(
            "X",
            vec![
                cal("minute", vec![0.0; t], vec![0.0; horizon]),
                cal("hour", vec![0.0; t], vec![0.0; horizon]),
                cal("weekday", vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]),
                cal("day", vec![10.0, 11.0, 12.0, 13.0], vec![14.0, 17.0]),
                cal("month", vec![6.0, 6.0, 6.0, 6.0], vec![7.0, 7.0]),
            ],
        );
        let (ctx, fut) = f.calendar_stamps(&item, t, horizon);
        assert_eq!(ctx.len(), t * 5);
        assert_eq!(fut.len(), horizon * 5);
        // weekday is index 2, day index 3, month index 4.
        assert_eq!(ctx[0 * 5 + 2], 1); // weekday bar 0
        assert_eq!(ctx[3 * 5 + 2], 4); // weekday bar 3
        assert_eq!(ctx[2 * 5 + 3], 12); // day bar 2
        assert_eq!(ctx[0 * 5 + 4], 6); // month bar 0
        assert_eq!(fut[1 * 5 + 3], 17); // day future step 1
        assert_eq!(fut[0 * 5 + 4], 7); // month future step 0
    }

    #[test]
    fn no_calendar_gives_zero_stamps() {
        // an OHLCV-only item (no calendar variates) => all-zero stamps.
        let f = KronosForecaster::new(zero_model());
        let panel = ohlcv_panel(20);
        let (ctx, fut) = f.calendar_stamps(&panel.items[0], 20, 5);
        assert!(ctx.iter().all(|&v| v == 0) && fut.iter().all(|&v| v == 0));
    }

    #[test]
    fn missing_ohlcv_column_errors() {
        let f = KronosForecaster::new(zero_model());
        let mut panel = ohlcv_panel(20);
        panel.items[0].variates.retain(|v| v.name != "volume");
        let spec = ForecastSpec { horizon: 4, ..Default::default() };
        let err = f.forecast(&panel, &spec).unwrap_err();
        assert_eq!(err.code, "missing_variate");
    }

    #[test]
    fn forecast_through_the_seam_returns_samples() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let f = KronosForecaster::new(zero_model());
        let panel = ohlcv_panel(20);
        let spec = ForecastSpec {
            horizon: 5,
            representations: vec![Representation::Samples, Representation::Quantiles, Representation::Point],
            quantile_levels: vec![0.1, 0.5, 0.9],
            num_samples: 8,
            seed: 42,
        };
        let out = f.forecast(&panel, &spec).unwrap();
        assert_eq!(out.model, "kronos");
        assert_eq!(out.native_representation, Representation::Samples);
        let tf = &out.targets[0];
        assert_eq!(tf.name, "close");
        let s = tf.samples.as_ref().unwrap();
        assert_eq!(s.shape, vec![8, 5]); // [n_samples, horizon]
        assert!(!s.derived);
        // derived quantiles + point present and finite
        assert!(tf.quantiles.is_some());
        assert!(tf.mean.is_some());
        assert!(s.data.iter().all(|v| v.is_finite()));
    }
}
