// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gated fine-tuning of the TimesFM-3 core over a universe of series.
//!
//! Swedish Embedded AB implements fine-tuning and promotion gates for
//! forecasting models in production, for teams who need a model update to be
//! a decision backed by held-out evidence rather than a deployment. If your
//! team needs expertise in time-series fine-tuning, temporal validation
//! splits or model promotion policy, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! Structurally this is `kronos::finetune`: enumerate windows, split them
//! temporally with an embargo, train on the past, and return a checkpoint
//! ONLY alongside a verdict about held-out future data.
//!
//! # The objective lives here, not in the trainable graph
//!
//! [`crate::train::Timesfm3Train::backward`] takes `d_logits` from its
//! caller, because TimesFM-3's output-side map (RevIN-reverse, clamp,
//! stitching, trend re-add) holds no learnable parameters. This module is
//! that caller: it runs the REAL [`preprocess::postprocess`] to get a
//! forecast in original units, scores it with the pinball loss the model is
//! trained under, and maps the gradient back through the same affine map.
//!
//! [`objective`] is the whole of that bridge, and
//! `tests::the_objective_gradient_matches_finite_differences_of_postprocess`
//! finite-difference-checks it against `postprocess` itself. That matters
//! more than it looks: the affine map's coefficients are data-dependent
//! (per-patch RevIN stats, a per-variate trend, a clamp that saturates), and
//! a hand-derived Jacobian over a pipeline of four host-side stages is
//! exactly the kind of thing that is 95% right and silently wrong on the
//! clamp.
//!
//! # One forecast patch, no autoregressive feedback
//!
//! A training example is restricted to `horizon <= stitch_extract_len()`. At
//! that horizon `postprocess` emits a single forecast patch, read from source
//! patch `num_context_patches - 1` - a CONTEXT patch, so the CPM refinement
//! (which only revises the HORIZON patches' RevIN stats) does not enter the
//! Jacobian at all, and the map really is affine with parameter-independent
//! coefficients. Past that horizon the forecast is stitched from several
//! patches whose stats are themselves functions of the logits, which is a
//! second backward and not a rescale. `crate::train`'s module doc records
//! the same boundary from the trainer's side.

use std::collections::HashMap;

use forecast::metrics::{mean_pinball, mean_pinball_grad};
use forecast::train_data::{self, Series, Window, WindowRef};

use crate::config::Timesfm3Config;
use crate::preprocess::{self, BuiltInput, DecodeShape};
use crate::train::{LoraCfg, Timesfm3Train, TRAIN_PIPELINES};

/// The close price - column 3 of an OHLCV bar. TimesFM-3 is fine-tuned here
/// as a UNIVARIATE forecaster over it, rather than over all five columns as
/// a 5-variate panel: the promotion gate scores a forecast of the close, so
/// training on the quantity the gate scores is what makes the gate's verdict
/// about the model that will be served.
const CLOSE: usize = 3;

/// Fine-tune hyper-parameters. Field-for-field `kronos::train::FinetuneOpts`,
/// so one CLI verb configures either model from one set of flags.
#[derive(Clone, Debug)]
pub struct FinetuneOpts {
    pub epochs: u32,
    pub lr: f32,
    pub wd: f32,
    pub clip: f32,
    pub lora: Option<LoraCfg>,
    pub batch: u32,
    pub progress: bool,
}

impl Default for FinetuneOpts {
    fn default() -> FinetuneOpts {
        FinetuneOpts { epochs: 8, lr: 4e-5, wd: 0.1, clip: 3.0, lora: None, batch: 1, progress: false }
    }
}

/// The gate decision: whether the fine-tune beat the base on held-out data,
/// and by how much. Field-for-field `kronos::train::FinetuneReport`, so the
/// CLI prints one gate line whichever model ran.
#[derive(Clone, Debug)]
pub struct FinetuneReport {
    pub promoted: bool,
    pub base_val: f32,
    pub ft_val: f32,
    pub steps: u32,
}

/// One training example: everything `core_forward` consumes for one window,
/// plus everything [`objective`] needs to score its output.
pub struct Example {
    /// `[n, resblock_in_dim]` for a single univariate series.
    pub built: BuiltInput,
    pub shape: DecodeShape,
    /// The realised future in ORIGINAL units, `[horizon]`.
    pub actual: Vec<f32>,
}

/// Round a requested context up to a whole number of input patches.
/// [`preprocess::build_input`] requires that, and the shortfall is filled
/// with `NAN` at the FRONT - the same left-padding `Timesfm3Forecaster` uses,
/// where a non-finite value means "not observed" rather than "zero".
fn padded_context(cfg: &Timesfm3Config, context: usize) -> usize {
    context.div_ceil(cfg.input_patch_len).max(1) * cfg.input_patch_len
}

/// The number of patch rows one example occupies, `n` in `core_forward`'s
/// `[b*v*n, ...]`. Constant across a run, which is what lets one trainer be
/// built once and fed batch after batch through `set_input`.
pub fn patch_rows(cfg: &Timesfm3Config, context: usize) -> usize {
    let padded = padded_context(cfg, context);
    // Mirrors `build_input`'s own arithmetic; `horizon <= stitch_extract_len`
    // (checked by `supported_horizon`) makes `num_forecast_patches` exactly 1.
    padded / cfg.input_patch_len + cfg.rolls()
}

/// `Err` naming the limit if `horizon` is past what one forecast patch
/// covers. See the module doc for why that boundary is real.
pub fn supported_horizon(cfg: &Timesfm3Config, horizon: usize) -> Result<(), String> {
    let max = cfg.stitch_extract_len();
    if horizon == 0 || horizon > max {
        return Err(format!(
            "horizon {horizon}: this fine-tune trains one forecast patch, so the horizon must be 1..={max} (stitch_extract_len). A longer horizon is produced by stitching several patches whose own RevIN statistics depend on the logits, which is a second backward rather than a rescale"
        ));
    }
    Ok(())
}

/// Turn one extracted window into a training example, or `None` if its close
/// prices are not usable (a non-finite bar anywhere in the future makes the
/// pinball target undefined).
pub fn window_example(cfg: &Timesfm3Config, w: &Window, context: usize, horizon: usize) -> Option<Example> {
    let ctx: Vec<f32> = (0..context).map(|i| w.ctx[i * 5 + CLOSE]).collect();
    let actual: Vec<f32> = (0..horizon).map(|i| w.fut[i * 5 + CLOSE]).collect();
    if !actual.iter().all(|x| x.is_finite()) || !ctx.iter().any(|x| x.is_finite()) {
        return None;
    }
    let padded = padded_context(cfg, context);
    let mut target = vec![f32::NAN; padded];
    target[padded - context..].copy_from_slice(&ctx);

    let shape = DecodeShape { batch: 1, num_target: 1, num_past_only: 0, num_past_future: 0, context: padded, horizon };
    let built = preprocess::build_input(cfg, shape, &target, &[], &[]);
    Some(Example { built, shape, actual })
}

/// The loss in ORIGINAL units and its gradient w.r.t. the RAW output-head
/// logits, for a batch of `b` single-variate examples sharing one
/// `BuiltInput`.
///
/// The value comes from the real [`preprocess::postprocess`], so the number
/// being minimised is the number the served model would be scored on. The
/// gradient is the chain rule through that same map:
///
/// ```text
/// pred[h][q] = clamp(raw[row, h*Q + q] * sigma + mu, +-value_clip) + trend(h)
/// ```
///
/// so `d(loss)/d(raw) = d(loss)/d(pred) * sigma`, and zero wherever the clamp
/// saturated (the map is locally constant there). `row` is the single source
/// patch the forecast is read from; every other row of `d_logits` is zero,
/// because no other row reaches the loss.
pub fn objective(cfg: &Timesfm3Config, shape: DecodeShape, built: &BuiltInput, raw_logits: &[f32], actual: &[f32]) -> (f32, Vec<f32>) {
    let (b, horizon, nq) = (shape.batch, shape.horizon, cfg.num_quantiles);
    let v = shape.num_variates();
    let n = built.num_context_patches + built.num_horizon_patches;
    let pred = preprocess::postprocess(cfg, shape, built, raw_logits);

    let mut d = vec![0f32; raw_logits.len()];
    let mut total = 0f64;
    for bi in 0..b {
        // Univariate: variate 0 is the target, and it is the only one.
        let idx = bi * v * n + (built.num_context_patches - 1);
        let sigma = built.running_sigma[idx];
        let p0 = bi * v * horizon * nq;
        let y = &actual[bi * horizon..(bi + 1) * horizon];
        let q = &pred[p0..p0 + horizon * nq];
        total += mean_pinball(q, &cfg.quantile_levels, y) as f64;

        let dq = mean_pinball_grad(q, &cfg.quantile_levels, y);
        let base = idx * cfg.output_patch_len * nq;
        for h in 0..horizon {
            for j in 0..nq {
                // The clamp is the one non-linearity in the chain. Recover
                // whether it bit from the PRE-clamp value rather than from
                // the post-clamp one: a forecast that legitimately lands
                // exactly on the clip bound is not saturated.
                let i = base + h * nq + j;
                let mu = built.running_mu[idx];
                let raw = preprocess::revin(raw_logits[i], mu, sigma, true);
                if raw.abs() >= cfg.value_clip {
                    continue;
                }
                d[i] = dq[h * nq + j] * sigma;
            }
        }
    }
    ((total / b as f64) as f32, d)
}

/// Mean loss of `weights` over (a subsample of) every window in `series` -
/// the number both sides of the promotion gate are measured with.
pub fn eval_universe_loss(cfg: &Timesfm3Config, weights: &HashMap<String, Vec<f32>>, series: &[Series], context: usize, horizon: usize) -> f32 {
    let windows = train_data::enumerate_windows(series, context, horizon);
    if windows.is_empty() {
        return f32::NAN;
    }
    // Subsampled to a few hundred windows, matching `kronos::finetune`: this
    // runs three times per fine-tune (base, tuned, and once per held-out
    // universe) and a universe of a few hundred names enumerates tens of
    // thousands of windows.
    let step = (windows.len() / 300).max(1);
    let refs: Vec<WindowRef> = windows.into_iter().step_by(step).collect();
    let examples = build_examples(cfg, series, &refs, context, horizon);
    mean_loss(cfg, weights, None, &examples, context, horizon, 1)
}

fn build_examples(cfg: &Timesfm3Config, series: &[Series], refs: &[WindowRef], context: usize, horizon: usize) -> Vec<Example> {
    refs.iter()
        .filter_map(|&wr| window_example(cfg, &train_data::extract(series, wr, context, horizon), context, horizon))
        .collect()
}

/// Concatenate `group`'s per-example inputs into one `[b*v*n, ...]` batch,
/// and return the `BuiltInput` describing it. Every example has the same
/// shape by construction ([`patch_rows`]), so this is a splice, not a pad.
fn batch_of(cfg: &Timesfm3Config, group: &[&Example], horizon: usize) -> (Vec<f32>, Vec<bool>, BuiltInput, DecodeShape, Vec<f32>) {
    let b = group.len();
    let width = cfg.resblock_in_dim();
    let mut resblock_input = Vec::with_capacity(b * group[0].built.resblock_input.len());
    let mut patch_mask = Vec::with_capacity(b * group[0].built.patch_mask.len());
    let (mut n_, mut mu, mut sigma, mut trends) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut actual = Vec::with_capacity(b * horizon);
    for e in group {
        resblock_input.extend_from_slice(&e.built.resblock_input);
        patch_mask.extend_from_slice(&e.built.patch_mask);
        n_.extend_from_slice(&e.built.running_n);
        mu.extend_from_slice(&e.built.running_mu);
        sigma.extend_from_slice(&e.built.running_sigma);
        trends.extend_from_slice(&e.built.trends);
        actual.extend_from_slice(&e.actual);
    }
    debug_assert_eq!(resblock_input.len(), b * group[0].built.patch_mask.len() * width);
    let built = BuiltInput {
        resblock_input: resblock_input.clone(),
        patch_mask: patch_mask.clone(),
        running_n: n_,
        running_mu: mu,
        running_sigma: sigma,
        trends,
        num_context_patches: group[0].built.num_context_patches,
        num_horizon_patches: group[0].built.num_horizon_patches,
    };
    let shape = DecodeShape { batch: b, num_target: 1, num_past_only: 0, num_past_future: 0, context: group[0].shape.context, horizon };
    (resblock_input, patch_mask, built, shape, actual)
}

/// The mean objective of one weight set over `examples`, at batch size `b`.
/// A trailing partial batch is dropped: the graph is built for a fixed `b`.
fn mean_loss(cfg: &Timesfm3Config, weights: &HashMap<String, Vec<f32>>, lora: Option<&LoraCfg>, examples: &[Example], context: usize, horizon: usize, b: usize) -> f32 {
    if examples.len() < b {
        return f32::NAN;
    }
    let n = patch_rows(cfg, context);
    let rows = b * n;
    let zeros = vec![0f32; rows * cfg.resblock_in_dim()];
    let dev = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let m = match lora {
        None => Timesfm3Train::new_on(dev, cfg.clone(), &zeros, &vec![false; rows], b, 1, n, weights),
        Some(lc) => Timesfm3Train::new_lora_on(dev, cfg.clone(), lc.clone(), &zeros, &vec![false; rows], b, 1, n, weights),
    };
    let (mut total, mut count) = (0f64, 0u32);
    for group in examples.chunks_exact(b) {
        let refs: Vec<&Example> = group.iter().collect();
        let (x, mask, built, shape, actual) = batch_of(cfg, &refs, horizon);
        m.set_input(&x, &mask);
        m.forward();
        m.poll_wait();
        let (l, _) = objective(cfg, shape, &built, &m.read_logits(), &actual);
        total += l as f64;
        count += 1;
    }
    if count == 0 {
        f32::NAN
    } else {
        (total / count as f64) as f32
    }
}

/// Fine-tune over a universe of series and return the gate's verdict plus the
/// resulting weights, in the checkpoint's own tensor names with any LoRA
/// delta already folded in.
///
/// The weights are returned whether or not the gate promoted, so the caller
/// decides what to do with a losing run; `report.promoted` is the only thing
/// that should decide whether they are WRITTEN.
pub fn finetune_universe(cfg: &Timesfm3Config, base: &HashMap<String, Vec<f32>>, series: &[Series], context: usize, horizon: usize, split: train_data::SplitConfig, opts: &FinetuneOpts) -> (FinetuneReport, Option<HashMap<String, Vec<f32>>>) {
    let windows = train_data::enumerate_windows(series, context, horizon);
    let sp = train_data::temporal_split(series, &windows, horizon, split);
    let train = build_examples(cfg, series, &sp.train, context, horizon);
    let val = build_examples(cfg, series, &sp.val, context, horizon);

    let b = (opts.batch.max(1) as usize).min(train.len().max(1));
    let n = patch_rows(cfg, context);
    let rows = b * n;
    let zeros = vec![0f32; rows * cfg.resblock_in_dim()];
    let mask0 = vec![false; rows];

    let base_val = mean_loss(cfg, base, None, &val, context, horizon, b);

    let dev = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let m = match &opts.lora {
        None => Timesfm3Train::new_on(dev, cfg.clone(), &zeros, &mask0, b, 1, n, base),
        Some(lc) => Timesfm3Train::new_lora_on(dev, cfg.clone(), lc.clone(), &zeros, &mask0, b, 1, n, base),
    };

    let mut steps = 0u32;
    for epoch in 0..opts.epochs {
        for group in train.chunks_exact(b) {
            let refs: Vec<&Example> = group.iter().collect();
            let (x, mask, built, shape, actual) = batch_of(cfg, &refs, horizon);
            m.set_input(&x, &mask);
            m.forward();
            m.poll_wait();
            let (_, d_logits) = objective(cfg, shape, &built, &m.read_logits(), &actual);
            m.zero_grads();
            m.backward(&d_logits);
            steps += 1;
            m.adamw_step(steps, opts.lr, opts.wd, Some(opts.clip));
            m.poll_wait();
        }
        if opts.progress {
            eprintln!("timesfm3 finetune: epoch {}/{} ({steps} steps)", epoch + 1, opts.epochs);
        }
    }

    let weights = m.to_reference_weights();
    let ft_val = if steps == 0 { f32::NAN } else { mean_loss(cfg, &weights, None, &val, context, horizon, b) };
    // Lower held-out pinball loss wins, and a non-finite number never does -
    // `kronos::train::finetune`'s rule, so one CLI gate line means the same
    // thing whichever model produced it.
    let promoted = ft_val.is_finite() && base_val.is_finite() && ft_val < base_val;
    (FinetuneReport { promoted, base_val, ft_val, steps }, Some(weights))
}

/// The licence the TimesFM-3 3.0 pretrained weights carry. Non-commercial
/// and non-production, and neither the checkpoint nor ANY derivative of it
/// may be redistributed.
pub const WEIGHTS_LICENSE: &str = "timesfm-non-commercial-license-v1.0";

/// The upstream those weights came from, recorded on every artifact derived
/// from them.
pub const UPSTREAM: &str = "google/timesfm-3.0-pytorch";

/// The one-line warning a run that touches these weights prints at start.
pub const LICENSE_NOTICE: &str = "timesfm3: the TimesFM-3 3.0 weights are timesfm-non-commercial-license-v1.0 (non-commercial, non-production); a fine-tuned checkpoint is a DERIVATIVE and may not be redistributed";

/// Write a fine-tuned checkpoint: the reference's own tensor names and
/// shapes, plus a [`checkpoint::st::ModelCard`] carrying the upstream licence
/// and the base it derives from.
///
/// The licence is copied onto the artifact rather than inferred later by
/// whoever picks the file up. A fine-tune, a folded LoRA and a requantisation
/// are all derivatives, and a derivative of non-redistributable weights is
/// non-redistributable; the only way that survives a `cp` is for the terms to
/// live in the file. `checkpoint::license::redistributable` is the other half
/// of the pair, and it is what every publishing path checks.
pub fn save_weights(cfg: &Timesfm3Config, weights: &HashMap<String, Vec<f32>>, path: &str) {
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let data = weights.get(&name).unwrap_or_else(|| panic!("missing tensor {name}")).clone();
            (name, shape.iter().map(|&x| x as u64).collect(), data)
        })
        .collect();
    let mut card = checkpoint::st::ModelCard::new("timesfm3-ft", "timesfm3");
    card.architecture = Some("timesfm3".into());
    card.license = Some(WEIGHTS_LICENSE.into());
    // `variant_of` IS this repo's spelling of "derived from"; it is what
    // `qwen3`'s adapter cards already use for an adapter's base.
    card.variant_of = Some(UPSTREAM.into());
    card.context_length = Some(cfg.max_context as u64);
    card.param_count = Some(cfg.param_count() as u64);
    checkpoint::save_carded(path, cfg.to_json(), &tensors, &card);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// A deterministic series with a trend and a seasonal component, so the
    /// detrending path is exercised rather than skipped.
    ///
    /// 200 bars, not 70: `temporal_split` cuts the GLOBAL calendar at
    /// `train_frac` and `train_frac + val_frac` and clears an embargo band on
    /// both sides of each cut, so the validation slice is only non-empty once
    /// `val_frac * n` exceeds `2 * embargo + horizon`.
    fn series(n: usize, seed: f32) -> Series {
        let ohlcv: Vec<[f32; 5]> = (0..n)
            .map(|i| {
                let t = i as f32;
                let c = 100.0 + 0.05 * t + 3.0 * (t * 0.3 + seed).sin();
                [c, c + 0.5, c - 0.5, c, 1000.0]
            })
            .collect();
        // Distinct, increasing dates: `temporal_split` works on the calendar,
            // so repeated dates would collapse the timeline it cuts.
            let dates = (0..n).map(|i| (2020 + (i / 336) as i32, 1 + (i / 28) as u32 % 12, 1 + (i % 28) as u32)).collect();
        Series { ticker: format!("T{seed}"), dates, ohlcv }
    }

    /// **The bridge gate.** `objective` hand-derives the Jacobian of a
    /// four-stage host-side map (CPM stats, RevIN-reverse, a saturating
    /// clamp, stitching, trend re-add) composed with the pinball loss. Every
    /// coefficient in it is data-dependent, so nothing about it is checkable
    /// by inspection.
    ///
    /// Central differences against `postprocess` itself, per ENTRY of the raw
    /// logits, over the whole forecast row plus entries OUTSIDE it (which
    /// must be exactly zero: no other patch row reaches the loss, and a
    /// gradient that leaked into one would be training against a quantity
    /// the forecast never reads).
    ///
    /// Runs on the host - no device, no backward pass. The thing under test
    /// is the objective, and the trainable graph has its own gate.
    #[test]
    fn the_objective_gradient_matches_finite_differences_of_postprocess() {
        let cfg = Timesfm3Config::tiny();
        let (context, horizon) = (8usize, 4usize);
        supported_horizon(&cfg, horizon).expect("tiny covers this horizon");
        let s = vec![series(64, 0.0)];
        let w = train_data::extract(&s, WindowRef { series_idx: 0, origin: 40 }, context, horizon);
        let ex = window_example(&cfg, &w, context, horizon).expect("a clean window is usable");

        let n = ex.built.num_context_patches + ex.built.num_horizon_patches;
        assert_eq!(n, patch_rows(&cfg, context), "patch_rows must agree with build_input");

        // Deterministic pseudo-logits at a scale where the clamp does not bite.
        let len = n * cfg.head_out_dim();
        let raw: Vec<f32> = (0..len).map(|i| (((i * 37 + 11) % 41) as f32 - 20.0) * 0.05).collect();

        let (l0, d) = objective(&cfg, ex.shape, &ex.built, &raw, &ex.actual);
        assert!(l0.is_finite() && l0 > 0.0, "degenerate fixture: loss {l0}");
        assert_eq!(d.len(), len);

        let row = ex.built.num_context_patches - 1;
        let live = row * cfg.head_out_dim();
        assert!(d[live..live + horizon * cfg.num_quantiles].iter().any(|g| g.abs() > 1e-9), "the forecast row has no gradient at all");
        for (i, g) in d.iter().enumerate() {
            let in_row = (live..live + horizon * cfg.num_quantiles).contains(&i);
            assert!(in_row || *g == 0.0, "entry {i} is outside the forecast row and must have exactly zero gradient, got {g}");
        }

        let h = 1e-2f32;
        for i in live..live + horizon * cfg.num_quantiles {
            let mut plus = raw.clone();
            plus[i] += h;
            let mut minus = raw.clone();
            minus[i] -= h;
            let (lp, _) = objective(&cfg, ex.shape, &ex.built, &plus, &ex.actual);
            let (lm, _) = objective(&cfg, ex.shape, &ex.built, &minus, &ex.actual);
            let numeric = (lp - lm) / (2.0 * h);
            let tol = 4e-3 + 8e-2 * d[i].abs().max(numeric.abs());
            assert!((d[i] - numeric).abs() <= tol, "entry {i}: analytic {} numeric {numeric} (tol {tol})", d[i]);
        }
    }

    /// The horizon boundary is a refusal with a reason, not a panic deep in
    /// the stitching arithmetic.
    #[test]
    fn a_horizon_past_one_forecast_patch_is_refused_by_name() {
        let cfg = Timesfm3Config::tiny();
        assert!(supported_horizon(&cfg, 1).is_ok());
        assert!(supported_horizon(&cfg, cfg.stitch_extract_len()).is_ok());
        let e = supported_horizon(&cfg, cfg.stitch_extract_len() + 1).expect_err("must refuse");
        assert!(e.contains("stitch_extract_len"), "{e}");
        assert!(supported_horizon(&cfg, 0).is_err(), "a zero horizon has nothing to score");
    }

    /// End to end: the gate runs, trains a real number of steps, and reports
    /// two finite held-out numbers. Small enough to be a unit test, real
    /// enough that every stage (window enumeration, the embargoed split,
    /// batching, `set_input`, the objective, the backward, AdamW, the fold)
    /// runs at least once.
    #[test]
    fn the_gate_trains_and_reports_two_finite_held_out_numbers() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let (context, horizon) = (8usize, 4usize);
        let s: Vec<Series> = (0..3).map(|i| series(200, i as f32)).collect();
        let base = crate::train::init_weights(&cfg, 7);
        let opts = FinetuneOpts { epochs: 1, lr: 1e-3, wd: 0.0, clip: 1.0, lora: None, batch: 2, progress: false };
        let split = train_data::SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo: horizon };
        let (rep, w) = finetune_universe(&cfg, &base, &s, context, horizon, split, &opts);

        assert!(rep.steps > 0, "the fixture produced no training windows");
        assert!(rep.base_val.is_finite(), "base_val must be a real number, got {}", rep.base_val);
        assert!(rep.ft_val.is_finite(), "ft_val must be a real number, got {}", rep.ft_val);
        let w = w.expect("weights are always returned");
        assert_eq!(w.len(), cfg.param_list().len(), "a returned checkpoint carries the reference tensors and nothing else");
        for (name, _) in cfg.param_list() {
            assert!(w.contains_key(&name), "{name} missing from the returned checkpoint");
        }
    }

    /// Every artifact a fine-tune writes carries the upstream licence and the
    /// base it derives from, and is consequently refused by the publishing
    /// path. Both halves in one test on one real file: recording the licence
    /// without the refusal is a note nobody reads, and the refusal without
    /// the recording never fires.
    #[test]
    fn a_written_checkpoint_carries_the_licence_and_is_refused_by_the_publish_gate() {
        let cfg = Timesfm3Config::tiny();
        let w = crate::train::init_weights(&cfg, 7);
        let dir = std::env::temp_dir().join(format!("timesfm3-ft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ft.safetensors");
        let p = path.to_string_lossy().to_string();
        save_weights(&cfg, &w, &p);

        let card = checkpoint::st::read_card(&p).expect("readable").expect("a card was written");
        assert_eq!(card.license.as_deref(), Some(WEIGHTS_LICENSE));
        assert_eq!(card.variant_of.as_deref(), Some(UPSTREAM), "the artifact records the base it derives from");
        assert_eq!(card.family, "timesfm3");

        let e = checkpoint::license::redistributable(card.license.as_deref()).expect_err("a derivative of NC weights must not be publishable");
        assert!(e.contains(WEIGHTS_LICENSE), "{e}");

        // The tensors survived the card: a checkpoint that records its
        // licence but loses its weights is not a checkpoint.
        let back = checkpoint::st::load_safetensors(&p).expect("reloadable");
        assert_eq!(back.tensors.len(), cfg.param_list().len());
        assert_eq!(Timesfm3Config::from_json(&back.config()).expect("config round-trips").param_count(), cfg.param_count());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A LoRA run has to reach the same place through the adapter path, and
    /// must leave the base tensors it froze untouched in the checkpoint it
    /// returns except for the folded delta.
    #[test]
    fn a_lora_run_trains_and_folds_into_a_full_checkpoint() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let (context, horizon) = (8usize, 4usize);
        let s: Vec<Series> = (0..3).map(|i| series(200, i as f32)).collect();
        let base = crate::train::init_weights(&cfg, 7);
        let opts = FinetuneOpts { epochs: 1, lr: 1e-2, wd: 0.0, clip: 1.0, lora: Some(LoraCfg::attn(2, 4.0)), batch: 2, progress: false };
        let split = train_data::SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo: horizon };
        let (rep, w) = finetune_universe(&cfg, &base, &s, context, horizon, split, &opts);
        assert!(rep.steps > 0);
        let w = w.expect("weights are always returned");
        assert_eq!(w.len(), cfg.param_list().len(), "the fold leaves no adapter tensors behind");

        // A tensor no adapter targets is bit-identical to the base; one that
        // is targeted moved. Without both halves this would pass for a fold
        // that did nothing and for a fold that overwrote everything.
        let untouched = "transformer_stack.layers.0.pre_ff_ln.weight";
        assert_eq!(w[untouched], base[untouched], "{untouched} is not a LoRA target and must be unchanged");
        let targeted = "transformer_stack.layers.0.seq_attn.query_proj.weight";
        assert_ne!(w[targeted], base[targeted], "{targeted} is a LoRA target and the delta must be folded in");
    }
}
