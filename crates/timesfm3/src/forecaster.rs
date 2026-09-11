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

    pub fn config(&self) -> &crate::Timesfm3Config {
        self.model.config()
    }

    /// A variate's step `i` is observed if `Variate::observed[i] != 0.0` when
    /// that mask is present, else if the raw value itself is finite -
    /// `observed` wins when both could apply, matching a caller that sets
    /// both consistently and letting one that only ever sets one of them work
    /// unsurprisingly either way.
    fn is_observed(v: &forecast::Variate, i: usize) -> bool {
        match &v.observed {
            Some(o) => o[i] != 0.0,
            None => v.data[i].is_finite(),
        }
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
        let patch = cfg.input_patch_len;

        // Pass 1: validate every item and compute its shape, before any
        // device call - an item found invalid partway through the OLD
        // per-item loop still failed the whole request (no partial result is
        // ever returned), so validating everything up front changes nothing
        // observable, only when the error is raised.
        let mut plans: Vec<ItemPlan> = Vec::new();
        for item in &panel.items {
            let targets: Vec<&forecast::Variate> = item.variates.iter().filter(|v| matches!(v.role, Role::Target)).collect();
            let past_only: Vec<&forecast::Variate> = item.variates.iter().filter(|v| matches!(v.role, Role::PastCovariate)).collect();
            let known_future: Vec<&forecast::Variate> = item.variates.iter().filter(|v| matches!(v.role, Role::KnownFuture)).collect();
            if targets.is_empty() {
                continue;
            }
            let valid = targets[0].data.len();
            if targets.iter().chain(&past_only).any(|v| v.data.len() != valid) {
                return Err(ForecastError::bad_request("timesfm3: every target/past-covariate must share the target's context length"));
            }
            for t in &targets {
                if (0..valid).all(|i| !Self::is_observed(t, i)) {
                    return Err(ForecastError::bad_request(format!("timesfm3: target '{}' has no observed steps", t.name)));
                }
            }
            for c in &known_future {
                let future = c.future.as_deref().ok_or_else(|| ForecastError::bad_request("timesfm3: known_future covariate is missing its future path"))?;
                if future.len() != spec.horizon {
                    return Err(ForecastError::bad_request("timesfm3: known_future length must equal the horizon"));
                }
            }
            // Left-padding and a per-step gap are the same mechanism as far as
            // `preprocess::build_input` is concerned: both become `f32::NAN`,
            // which it masks and excludes from RevIN/detrend statistics (see
            // its own doc). `context` is the padded length `build_input`
            // requires (a multiple of `input_patch_len`); items whose raw
            // history rounds up to the SAME padded context are batched
            // together below even when their raw lengths differ.
            let context = valid.div_ceil(patch).max(1) * patch;
            plans.push(ItemPlan { item, targets, past_only, known_future, valid, context });
        }

        // Pass 2: group by padded context (the only per-item dimension
        // `core_forward` needs uniform beyond horizon, which is spec-wide),
        // batch one `core_forward` call per group, and pad variate counts up
        // to the group's max with wholly-NaN rows. `build_input` already
        // derives an all-masked (leading-cumprod) attention mask for a row
        // that is NaN start to finish, so a padding row is excluded as a KEY
        // in both attention sublayers with no dedicated masking logic here -
        // see `preprocess::build_input`'s own doc.
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for (i, p) in plans.iter().enumerate() {
            groups.entry(p.context).or_default().push(i);
        }

        let mut per_item_targets: Vec<Vec<TargetForecast>> = (0..plans.len()).map(|_| Vec::new()).collect();

        for (&context, idxs) in &groups {
            let max_t = idxs.iter().map(|&i| plans[i].targets.len()).max().unwrap();
            let max_p = idxs.iter().map(|&i| plans[i].past_only.len()).max().unwrap();
            let max_f = idxs.iter().map(|&i| plans[i].known_future.len()).max().unwrap();
            let b = idxs.len();
            let v = max_t + max_p + max_f;
            // Padding relies on the softmax reduction combining the SAME real
            // columns in the SAME order regardless of how many masked pad
            // columns follow them - true within one workgroup pass, not
            // guaranteed across a multi-tile reduction.
            debug_assert!(v <= 64, "variate padding assumes the padded row width fits one softmax workgroup (v={v})");

            let real_row = |var: &forecast::Variate, context: usize| -> Vec<f32> {
                let valid = var.data.len();
                let pad = context - valid;
                let mut out = Vec::with_capacity(context);
                out.resize(pad, f32::NAN);
                out.extend((0..valid).map(|i| if Self::is_observed(var, i) { var.data[i] } else { f32::NAN }));
                out
            };
            let pad_row = |n: usize| -> Vec<f32> { vec![f32::NAN; n] };

            let mut target_data = Vec::with_capacity(b * max_t * context);
            let mut past_only_data = Vec::with_capacity(b * max_p * context);
            let mut past_future_data = Vec::with_capacity(b * max_f * (context + spec.horizon));
            for &i in idxs {
                let p = &plans[i];
                for ti in 0..max_t {
                    target_data.extend(if ti < p.targets.len() { real_row(p.targets[ti], context) } else { pad_row(context) });
                }
                for pi in 0..max_p {
                    past_only_data.extend(if pi < p.past_only.len() { real_row(p.past_only[pi], context) } else { pad_row(context) });
                }
                for fi in 0..max_f {
                    if fi < p.known_future.len() {
                        let c = p.known_future[fi];
                        past_future_data.extend(real_row(c, context));
                        past_future_data.extend_from_slice(c.future.as_deref().expect("validated in pass 1"));
                    } else {
                        past_future_data.extend(pad_row(context + spec.horizon));
                    }
                }
            }

            let shape = DecodeShape { batch: b, num_target: max_t, num_past_only: max_p, num_past_future: max_f, context, horizon: spec.horizon };
            let built = preprocess::build_input(cfg, shape, &target_data, &past_only_data, &past_future_data);
            let n = built.num_context_patches + built.num_horizon_patches;
            let raw_logits = self.model.core_forward(&built.resblock_input, &built.patch_mask, shape.batch, shape.num_variates(), n);
            let mut out = preprocess::postprocess(cfg, shape, &built, &raw_logits);
            sort_quantiles_inplace(&mut out, cfg.num_quantiles);

            for (bi, &i) in idxs.iter().enumerate() {
                let p = &plans[i];
                for (ti, t) in p.targets.iter().enumerate() {
                    let base = (bi * v + ti) * spec.horizon * cfg.num_quantiles;
                    let native = &out[base..base + spec.horizon * cfg.num_quantiles];
                    // Unobserved steps must not defeat the clamp: `t.data[i]`
                    // may be a stale or NaN placeholder there, so only
                    // observed steps are asked to justify it.
                    let nonneg = (0..p.valid).all(|k| !Self::is_observed(t, k) || t.data[k] >= 0.0);
                    let mut native = native.to_vec();
                    if nonneg {
                        for x in &mut native {
                            *x = x.max(0.0);
                        }
                    }
                    let mut q = Self::interp_levels(&native, &cfg.quantile_levels, spec.horizon, &levels);
                    if nonneg {
                        for x in &mut q {
                            *x = x.max(0.0);
                        }
                    }
                    let mut tf = TargetForecast::new(p.item.item_id.clone(), t.name.clone());
                    tf.quantiles = Some(Block::native(vec![spec.horizon, levels.len()], q));
                    tf.levels = levels.clone();
                    forecast::convert::ensure_representations(&mut tf, Representation::Quantiles, &spec.representations, &levels, spec.num_samples, spec.seed)?;
                    per_item_targets[i].push(tf);
                }
            }
        }

        for targets in per_item_targets {
            fc.targets.extend(targets);
        }
        Ok(fc)
    }
}

/// One `Panel` item's role split and padded shape, computed once in pass 1
/// and consumed by every group it lands in during pass 2.
struct ItemPlan<'a> {
    item: &'a forecast::Item,
    targets: Vec<&'a forecast::Variate>,
    past_only: Vec<&'a forecast::Variate>,
    known_future: Vec<&'a forecast::Variate>,
    valid: usize,
    context: usize,
}
