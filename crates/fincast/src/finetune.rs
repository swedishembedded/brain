// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gated fine-tuning of FinCast over a universe of series.
//!
//! Swedish Embedded AB implements fine-tuning and promotion gates for
//! forecasting models in production, for teams who need a model update to be
//! a decision backed by held-out evidence rather than a deployment. If your
//! team needs expertise in time-series fine-tuning, temporal validation
//! splits or model promotion policy, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! The shape is `timesfm3::finetune`: enumerate windows, split them in time
//! with an embargo, train on the past, and return the tuned weights together
//! with a verdict ([`forecast::FinetuneReport`]) about held-out future data.
//!
//! # The trained graph and its loss
//!
//! The trainable twin is the host [`FincastTrain`]: input embedding (with the
//! frequency row), every decoder block (attention and top-2 MoE) and the
//! horizon head, all trained. The loop around it is the shared
//! [`forecast::host_train::finetune_host`]; this module supplies what is
//! FinCast's own: how a window becomes patch features and a target
//! ([`window_example`]) and the loss, the model's probabilistic-quantile loss
//! (mean pinball over its 9 quantiles plus the squared error of the mean head)
//! against the target standardized by the context's own statistics, which is
//! the space the model is trained in.
//!
//! The router picks the top two experts by a hard sort, so the loss is only
//! piecewise differentiable in the gate; the gradient is the one of the
//! selected experts, as the reference's is.
//!
//! # Scope
//!
//! - Univariate, over the close (column 3 of an OHLCV bar).
//! - One forecast patch with a full context: the context is exactly
//!   `context_len` (the host trainer attends over unpadded patches) and the
//!   horizon is exactly `horizon_len` ([`supported_shape`]). A longer forecast
//!   is produced by feeding the mean back, which this does not train through.
//! - Full fine-tune only. There is no adapter in the host trainer
//!   ([`NoAdapter`] makes `opts.lora` unrepresentable).
//! - Host (CPU) compute: slow at the 50-layer scale, exact, and gradcheck-gated.

use std::collections::HashMap;

use forecast::host_train::{self, HostTrainable};
use forecast::train_data::{self, Series, Window, WindowRef};
use model::saved::SavedPolicy;

use crate::config::FincastConfig;
use crate::forecaster::freq_bucket;
use crate::preprocess;
use crate::train::FincastTrain;

/// The close price - column 3 of an OHLCV bar.
const CLOSE: usize = 3;

/// No adapter exists for the host trainer; the type has no values, so
/// `FinetuneOpts::lora` can only be `None`.
#[derive(Clone, Debug)]
pub enum NoAdapter {}

/// Fine-tuning hyper-parameters: the shared forecaster contract
/// ([`forecast::finetune`]), with no adapter.
pub type FinetuneOpts = forecast::finetune::FinetuneOpts<NoAdapter, Option<SavedPolicy>>;

/// The gate decision of one fine-tune ([`forecast::finetune`]).
pub type FinetuneReport = forecast::finetune::FinetuneReport;

/// One training example: the embedding input and the standardized target.
pub struct Example {
    /// `[context_len / patch_len, 2 * patch_len]` (values, padding flags).
    features: Vec<f32>,
    /// The frequency bucket of the series (0 high / 1 med / 2 low).
    freq: usize,
    /// The realised future, standardized, `[horizon_len]`.
    target: Vec<f32>,
}

/// `Err` naming the limit unless the window is exactly one full context and
/// one forecast patch.
pub fn supported_shape(cfg: &FincastConfig, context: usize, horizon: usize) -> Result<(), String> {
    if context != cfg.context_len {
        return Err(format!("context {context}: this fine-tune trains full unpadded contexts, so the context must be exactly context_len ({})", cfg.context_len));
    }
    if horizon != cfg.horizon_len {
        return Err(format!("horizon {horizon}: this fine-tune trains one forecast patch, so the horizon must be exactly horizon_len ({})", cfg.horizon_len));
    }
    Ok(())
}

/// Turn one extracted window into a training example, or `None` if its close
/// prices are not usable (a non-finite bar anywhere makes the standardization
/// or the target undefined).
pub fn window_example(cfg: &FincastConfig, w: &Window, freq: usize, context: usize, horizon: usize) -> Option<Example> {
    let ctx: Vec<f32> = (0..context).map(|i| w.ctx[i * 5 + CLOSE]).collect();
    let actual: Vec<f32> = (0..horizon).map(|i| w.fut[i * 5 + CLOSE]).collect();
    if !ctx.iter().chain(&actual).all(|x| x.is_finite()) {
        return None;
    }
    let pp = preprocess::preprocess(cfg, &ctx);
    let target = actual.iter().map(|&v| preprocess::normalize(v, pp.loc_scale)).collect();
    Some(Example { features: pp.features, freq, target })
}

impl HostTrainable for FincastTrain {
    type Example = Example;

    fn params(&self) -> &HashMap<String, Vec<f32>> {
        &self.w
    }

    fn params_mut(&mut self) -> &mut HashMap<String, Vec<f32>> {
        &mut self.w
    }

    fn loss(&self, ex: &Example) -> f32 {
        let (emb, _) = self.embed_forward(&ex.features, ex.freq);
        self.full_forward(&emb, &ex.target).0
    }

    fn loss_and_grad(&self, ex: &Example, grads: &mut HashMap<String, Vec<f32>>) -> f32 {
        let (emb, embed_cache) = self.embed_forward(&ex.features, ex.freq);
        let (loss, cache) = self.full_forward(&emb, &ex.target);
        let d_emb = self.full_backward(&cache, &ex.target, grads);
        self.embed_backward(&embed_cache, &d_emb, grads);
        loss
    }
}

fn build_examples(cfg: &FincastConfig, series: &[Series], refs: &[WindowRef], freq: usize, context: usize, horizon: usize) -> Vec<Example> {
    refs.iter()
        .filter_map(|&wr| window_example(cfg, &train_data::extract(series, wr, context, horizon), freq, context, horizon))
        .collect()
}

/// Mean loss of `weights` over (a subsample of) every window in `series` - the
/// number both sides of the promotion gate are measured with. `freq` is the
/// series' sampling frequency (`"1d"`).
pub fn eval_universe_loss(cfg: &FincastConfig, weights: &HashMap<String, Vec<f32>>, series: &[Series], freq: &str, context: usize, horizon: usize) -> f32 {
    let windows = train_data::enumerate_windows(series, context, horizon);
    if windows.is_empty() || supported_shape(cfg, context, horizon).is_err() {
        return f32::NAN;
    }
    // Subsampled to a few hundred windows, as `timesfm3::finetune` does.
    let step = (windows.len() / 300).max(1);
    let refs: Vec<WindowRef> = windows.into_iter().step_by(step).collect();
    let examples = build_examples(cfg, series, &refs, freq_bucket(freq), context, horizon);
    host_train::mean_loss(&FincastTrain::new(cfg.clone(), weights.clone()), &examples)
}

/// Fine-tune over a universe of series sampled at `freq` and return the gate's
/// verdict plus the tuned weights in the checkpoint's own tensor names.
///
/// The weights are returned whether or not the gate promoted, so the caller
/// decides what to do with a losing run; `report.promoted` is the only thing
/// that should decide whether they are WRITTEN.
#[allow(clippy::too_many_arguments)]
pub fn finetune_universe(cfg: &FincastConfig, base: &HashMap<String, Vec<f32>>, series: &[Series], freq: &str, context: usize, horizon: usize, split: train_data::SplitConfig, opts: &FinetuneOpts, observer: forecast::finetune::StepObserver) -> Result<forecast::FinetuneOutcome, String> {
    supported_shape(cfg, context, horizon)?;
    let windows = train_data::enumerate_windows(series, context, horizon);
    let sp = train_data::temporal_split(series, &windows, horizon, split);
    let bucket = freq_bucket(freq);
    let train = build_examples(cfg, series, &sp.train, bucket, context, horizon);
    let val = build_examples(cfg, series, &sp.val, bucket, context, horizon);

    let mut model = FincastTrain::new(cfg.clone(), base.clone());
    if let Some(policy) = opts.saved {
        model.set_saved_policy(policy)?;
    }
    host_train::finetune_host(&mut model, &train, &val, opts, observer)
}

/// The licence the FinCast pretrained weights are published under. The
/// authors' added statement that the model is for research and educational
/// purposes only is flagged in the crate note, not encoded here.
pub const WEIGHTS_LICENSE: &str = "apache-2.0";

/// The upstream the FinCast pretrained weights came from, recorded on every
/// artifact derived from them.
pub const UPSTREAM: &str = "Vincent05R/FinCast";

/// Read a base checkpoint's config and weights, host-side, from a brain
/// container (see [`crate::import`]).
pub fn load_base(path: &str) -> Result<(FincastConfig, HashMap<String, Vec<f32>>), String> {
    checkpoint::load_host_weights(path, |c| {
        let cfg = FincastConfig::from_json(c)?;
        let expected = cfg.param_list().into_iter().map(|(name, shape)| (name, shape.iter().product())).collect();
        Ok((cfg, expected))
    })
}

/// Write a fine-tuned checkpoint in the container [`crate::model::Fincast::load`]
/// reads, with a [`checkpoint::st::ModelCard`] recording the licence and the
/// base it derives from.
pub fn save_weights(cfg: &FincastConfig, weights: &HashMap<String, Vec<f32>>, path: &str) {
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let data = weights.get(&name).unwrap_or_else(|| panic!("missing tensor {name}")).clone();
            (name, vec![shape.iter().product::<usize>() as u64], data)
        })
        .collect();
    let mut card = checkpoint::st::ModelCard::new("fincast-ft", "fincast");
    card.architecture = Some("fincast".into());
    card.license = Some(WEIGHTS_LICENSE.into());
    card.variant_of = Some(UPSTREAM.into());
    card.context_length = Some(cfg.context_len as u64);
    card.param_count = Some(cfg.param_count() as u64);
    checkpoint::save_carded(path, cfg.to_json(), &tensors, &card);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::init_weights;

    /// A deterministic series with a trend and a seasonal component, long
    /// enough that `temporal_split` leaves a non-empty validation slice.
    fn series(n: usize, seed: f32) -> Series {
        let ohlcv: Vec<[f32; 5]> = (0..n)
            .map(|i| {
                let t = i as f32;
                let c = 100.0 + 0.05 * t + 3.0 * (t * 0.3 + seed).sin();
                [c, c + 0.5, c - 0.5, c, 1000.0]
            })
            .collect();
        let dates = (0..n).map(|i| (2020 + (i / 336) as i32, 1 + (i / 28) as u32 % 12, 1 + (i % 28) as u32)).collect();
        Series { ticker: format!("T{seed}"), dates, ohlcv }
    }

    #[test]
    fn a_window_that_is_not_one_full_context_and_one_patch_is_refused_by_name() {
        let cfg = FincastConfig::tiny(); // context 32, horizon 4
        assert!(supported_shape(&cfg, 32, 4).is_ok());
        assert!(supported_shape(&cfg, 16, 4).unwrap_err().contains("context_len"));
        assert!(supported_shape(&cfg, 32, 8).unwrap_err().contains("horizon_len"));
    }

    #[test]
    fn a_window_becomes_standardized_features_and_an_unusable_one_is_dropped() {
        let cfg = FincastConfig::tiny();
        let s = vec![series(80, 0.0)];
        let w = train_data::extract(&s, WindowRef { series_idx: 0, origin: 40 }, 32, 4);
        let ex = window_example(&cfg, &w, 0, 32, 4).expect("a clean window is usable");
        assert_eq!(ex.features.len(), (32 / cfg.patch_len) * cfg.patch_feat_dim());
        assert_eq!(ex.target.len(), 4);
        assert!(ex.target.iter().all(|t| t.is_finite()));

        let mut bad = w.clone();
        bad.ctx[5 * 5 + CLOSE] = f32::NAN;
        assert!(window_example(&cfg, &bad, 0, 32, 4).is_none());
    }

    /// End to end on the host: the gate trains, reports two finite held-out
    /// numbers, and on a learnable series the tuned model beats the base.
    #[test]
    fn the_gate_trains_and_the_tuned_weights_beat_the_base_on_held_out_windows() {
        let cfg = FincastConfig::tiny();
        let (context, horizon) = (32usize, 4usize);
        let s: Vec<Series> = (0..3).map(|i| series(200, i as f32)).collect();
        let base = init_weights(&cfg, 7);
        let opts = FinetuneOpts { epochs: 2, lr: 3e-3, wd: 0.0, clip: 1.0, lora: None, batch: 4, saved: None };
        let split = train_data::SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo: horizon };
        let (rep, w) = finetune_universe(&cfg, &base, &s, "1d", context, horizon, split, &opts, &mut |_| true).unwrap();

        assert!(rep.steps > 0, "the fixture produced no training windows");
        assert!(rep.base_val.is_finite() && rep.ft_val.is_finite(), "{rep:?}");
        assert!(rep.promoted && rep.ft_val < rep.base_val, "training on a learnable series must beat random weights: {rep:?}");
        let w = w.expect("weights are always returned");
        assert_eq!(w.len(), cfg.param_list().len());
        assert_ne!(w["horizon_ff_layer.output_layer.bias"], base["horizon_ff_layer.output_layer.bias"], "the weights moved");
        assert!(eval_universe_loss(&cfg, &w, &s, "1d", context, horizon).is_finite());
    }

    #[test]
    fn a_written_checkpoint_records_its_provenance_and_reloads() {
        let cfg = FincastConfig::tiny();
        let w = init_weights(&cfg, 7);
        let dir = std::env::temp_dir().join(format!("fincast-ft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("ft.safetensors").to_string_lossy().to_string();
        save_weights(&cfg, &w, &p);

        let card = checkpoint::st::read_card(&p).expect("readable").expect("a card was written");
        assert_eq!(card.variant_of.as_deref(), Some(UPSTREAM));
        assert_eq!(card.license.as_deref(), Some(WEIGHTS_LICENSE));
        let (back_cfg, back) = load_base(&p).expect("load_base reads what save_weights wrote");
        assert_eq!(back_cfg, cfg);
        assert_eq!(back, w);
        assert!(load_base(&dir.join("missing").to_string_lossy()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
