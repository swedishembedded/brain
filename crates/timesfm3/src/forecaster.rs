// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`forecast::ForecastModel`] adapter - makes a loaded [`Timesfm3`] drivable
//! through the whole forecasting API (CLI, server, comparison harness).
//!
//! Unlike `chronos2`/`fincast` (one target series per call, covariates folded
//! in via group attention), TimesFM-3 is NATIVELY multivariate: every
//! `Role::Target` variate in an `Item` is forecast in the SAME decode() call,
//! attending to every other target and covariate through its variate
//! attention sublayer - this is the model's headline capability, and this
//! adapter's whole job is mapping `Panel`'s generic `Role` vocabulary onto
//! `DecodeShape`'s `(target, past_only, past_future)` split, which is
//! genuinely a 1:1 correspondence (`Role::Target` -> target,
//! `Role::PastCovariate` -> past_only, `Role::KnownFuture` -> past_future).
//!
//! Native representation is 9 fixed quantiles (like `fincast`, unlike
//! `chronos2`'s interpolatable 21) - requested levels are served by
//! interpolating across those 9, same pattern `fincast::forecaster` uses.
//!
//! Forecaster-level postprocessing implemented here: quantile sorting
//! (monotonicity is not guaranteed per-quantile-head output) and a positivity
//! clamp when every input value was non-negative. NOT implemented: symmetric
//! averaging (doubles compute per request; the reference's own default is
//! evaluator-only, not the plain forecaster path), z-normalization, and
//! 32-variate chunking for panels with more targets than the model's
//! `max_variates`.

use crate::preprocess::{self, DecodeShape};
use crate::Timesfm3;
use forecast::{Block, Capabilities, CovariateSupport, Forecast, ForecastError, ForecastModel, ForecastSpec, Panel, Representation, Role, TargetForecast};

pub struct Timesfm3Forecaster {
    model: Timesfm3,
    version: String,
}

impl Timesfm3Forecaster {
    pub fn new(model: Timesfm3) -> Timesfm3Forecaster {
        Timesfm3Forecaster { model, version: "google/timesfm-3.0-pytorch".into() }
    }

    pub fn load(path: &str) -> Result<Timesfm3Forecaster, String> {
        Ok(Timesfm3Forecaster::new(Timesfm3::load(path)?))
    }

    /// Interpolate the requested `levels` from the native quantile matrix
    /// `native` (`[horizon, native_levels.len()]`, step-major - `postprocess`'s
    /// own output layout), against THIS model's actual quantile levels
    /// (`native_levels` - never assumed to be the crate-level
    /// [`crate::config::QUANTILES`] constant, which is only the real
    /// checkpoint's own 9; a differently
    /// configured model, e.g. [`crate::Timesfm3Config::tiny`], has fewer).
    /// Returns `[horizon, n_levels]` step-major.
    fn interp_levels(native: &[f32], native_levels: &[f32], horizon: usize, levels: &[f32]) -> Vec<f32> {
        let nq = native_levels.len();
        let mut out = vec![0.0f32; horizon * levels.len()];
        for t in 0..horizon {
            for (j, &lv) in levels.iter().enumerate() {
                out[t * levels.len() + j] = interp_one(native, native_levels, t, nq, lv);
            }
        }
        out
    }
}

fn interp_one(native: &[f32], native_levels: &[f32], t: usize, nq: usize, lv: f32) -> f32 {
    let at = |k: usize| native[t * nq + k];
    if lv <= native_levels[0] {
        return at(0);
    }
    if lv >= native_levels[nq - 1] {
        return at(nq - 1);
    }
    for k in 1..nq {
        if lv <= native_levels[k] {
            let span = native_levels[k] - native_levels[k - 1];
            let frac = if span > 0.0 { (lv - native_levels[k - 1]) / span } else { 0.0 };
            return at(k - 1) + (at(k) - at(k - 1)) * frac;
        }
    }
    at(nq - 1)
}

/// Sort every (batch*variate, step) row's quantile values into non-decreasing
/// order - the output head has no monotonicity constraint built in, so a
/// lower quantile can come out numerically above a higher one; the reference
/// forecaster corrects this the same way (`sort_quantiles`, applied before
/// any other postprocessing). `out` is `[bv, horizon, nq]` flattened - every
/// row across the WHOLE buffer is sorted, not just the first `horizon` rows
/// (a bug that would silently skip every variate past the first).
fn sort_quantiles_inplace(out: &mut [f32], nq: usize) {
    for row in out.chunks_exact_mut(nq) {
        row.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
}

impl ForecastModel for Timesfm3Forecaster {
    fn capabilities(&self) -> Capabilities {
        let cfg = self.model.config();
        Capabilities {
            name: "timesfm3".into(),
            max_context: cfg.max_context,
            max_horizon: None, // stitching covers any horizon; no fixed cap like a single-patch head
            native_representation: Representation::Quantiles,
            covariates: CovariateSupport::Full,
            supports_known_future: true,
            multivariate: true,
            arbitrary_quantile_levels: true, // served by interpolating the 9 native
            stochastic: false,
            requires_variates: vec![],
        }
    }

    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        let cfg = self.model.config();
        let mut fc = Forecast::new("timesfm3", Representation::Quantiles, spec.horizon, &panel.freq);
        fc.model_version = self.version.clone();

        let levels = if spec.quantile_levels.is_empty() { vec![0.1, 0.5, 0.9] } else { spec.quantile_levels.clone() };

        for item in &panel.items {
            let targets: Vec<&forecast::Variate> = item.variates.iter().filter(|v| matches!(v.role, Role::Target)).collect();
            let past_only: Vec<&forecast::Variate> = item.variates.iter().filter(|v| matches!(v.role, Role::PastCovariate)).collect();
            let known_future: Vec<&forecast::Variate> = item.variates.iter().filter(|v| matches!(v.role, Role::KnownFuture)).collect();
            if targets.is_empty() {
                continue;
            }
            let context = targets[0].data.len();
            if context % cfg.input_patch_len != 0 {
                return Err(ForecastError::bad_request(format!(
                    "timesfm3: context length {context} is not a multiple of input_patch_len {} (left-padding to a patch boundary is not implemented yet)",
                    cfg.input_patch_len
                )));
            }
            if targets.iter().chain(&past_only).any(|v| v.data.len() != context) {
                return Err(ForecastError::bad_request("timesfm3: every target/past-covariate must share the target's context length"));
            }

            let mut target_data = Vec::with_capacity(targets.len() * context);
            for t in &targets {
                target_data.extend_from_slice(&t.data);
            }
            let mut past_only_data = Vec::with_capacity(past_only.len() * context);
            for c in &past_only {
                past_only_data.extend_from_slice(&c.data);
            }
            let mut past_future_data = Vec::with_capacity(known_future.len() * (context + spec.horizon));
            for c in &known_future {
                let future = c.future.as_deref().ok_or_else(|| ForecastError::bad_request("timesfm3: known_future covariate is missing its future path"))?;
                if future.len() != spec.horizon {
                    return Err(ForecastError::bad_request("timesfm3: known_future length must equal the horizon"));
                }
                past_future_data.extend_from_slice(&c.data);
                past_future_data.extend_from_slice(future);
            }

            let shape = DecodeShape { batch: 1, num_target: targets.len(), num_past_only: past_only.len(), num_past_future: known_future.len(), context, horizon: spec.horizon };
            let built = preprocess::build_input(cfg, shape, &target_data, &past_only_data, &past_future_data);
            let n = built.num_context_patches + built.num_horizon_patches;
            let raw_logits = self.model.core_forward(&built.resblock_input, &built.patch_mask, shape.batch, shape.num_variates(), n);
            let mut out = preprocess::postprocess(cfg, shape, &built, &raw_logits);
            sort_quantiles_inplace(&mut out, cfg.num_quantiles);

            for (ti, t) in targets.iter().enumerate() {
                let native = &out[ti * spec.horizon * cfg.num_quantiles..(ti + 1) * spec.horizon * cfg.num_quantiles];
                let nonneg = t.data.iter().all(|&x| x >= 0.0);
                let mut native = native.to_vec();
                if nonneg {
                    for v in &mut native {
                        *v = v.max(0.0);
                    }
                }
                let mut q = Self::interp_levels(&native, &cfg.quantile_levels, spec.horizon, &levels);
                if nonneg {
                    for v in &mut q {
                        *v = v.max(0.0);
                    }
                }
                let mut tf = TargetForecast::new(item.item_id.clone(), t.name.clone());
                tf.quantiles = Some(Block::native(vec![spec.horizon, levels.len()], q));
                tf.levels = levels.clone();
                forecast::convert::ensure_representations(&mut tf, Representation::Quantiles, &spec.representations, &levels, spec.num_samples, spec.seed)?;
                fc.targets.push(tf);
            }
        }
        Ok(fc)
    }
}
