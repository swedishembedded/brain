// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gated fine-tuning of Chronos-2 over a universe of series.
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
//! The trainable twin is the host [`Chronos2Train`]: input embedding, every
//! encoder block, the final norm and the quantile head, all trained, with the
//! scaler (data-derived) outside. The loop around it is the shared
//! [`forecast::host_train::finetune_host`] (AdamW, global-norm clip, batch
//! mean, `promoted` gate); this module supplies what is Chronos-2's own: how a
//! window becomes tokens and a target ([`window_example`]) and the loss.
//!
//! The loss is the mean pinball loss over the model's 21 quantile levels
//! against the target standardized by the same `instance_norm` the forecast is
//! mapped back through, which is the space the model is trained in. A window's
//! loss therefore does not depend on the price level of its series, and both
//! sides of the gate average windows on equal footing.
//!
//! # Scope
//!
//! - Univariate, over the close (column 3 of an OHLCV bar), as the gate scores
//!   a forecast of the close.
//! - The horizon is a whole number of output patches
//!   ([`supported_horizon`]): a partial last patch has no target for its tail.
//! - Full fine-tune only. There is no adapter in the host trainer
//!   ([`NoAdapter`] makes `opts.lora` unrepresentable), so the setting the
//!   other forecasters offer is refused at compile time rather than ignored.
//! - Host (CPU) compute: slow at the 120M scale, exact, and gradcheck-gated.

use std::collections::HashMap;

use forecast::host_train::{self, HostTrainable};
use forecast::train_data::{self, Series, Window, WindowRef};
use model::saved::SavedPolicy;

use crate::config::{Chronos2Config, QUANTILES};
use crate::preprocess;
use crate::train::Chronos2Train;

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
    /// `[n_ctx + n_out, 3 * patch]` patch features ([`preprocess::layout`]).
    features: Vec<f32>,
    /// Additive attention mask over the `n_ctx + 1 + n_out` tokens.
    mask: Vec<f32>,
    n_ctx: usize,
    n_out: usize,
    /// The realised future, standardized, `[n_out * output_patch_size]`.
    target: Vec<f32>,
}

/// `Err` naming the limit if `horizon` is not a whole number of output patches
/// within `max_output_patches`.
pub fn supported_horizon(cfg: &Chronos2Config, horizon: usize) -> Result<(), String> {
    let patch = cfg.output_patch_size;
    let max = cfg.max_output_patches * patch;
    if horizon == 0 || !horizon.is_multiple_of(patch) || horizon > max {
        return Err(format!(
            "horizon {horizon}: this fine-tune scores whole output patches, so the horizon must be a multiple of output_patch_size ({patch}) in {patch}..={max}"
        ));
    }
    Ok(())
}

/// Turn one extracted window into a training example, or `None` if its close
/// prices are not usable (a non-finite bar in the future makes the target
/// undefined; a context with no finite value has nothing to standardize).
pub fn window_example(cfg: &Chronos2Config, w: &Window, context: usize, horizon: usize) -> Option<Example> {
    let ctx: Vec<f32> = (0..context).map(|i| w.ctx[i * 5 + CLOSE]).collect();
    let actual: Vec<f32> = (0..horizon).map(|i| w.fut[i * 5 + CLOSE]).collect();
    if !actual.iter().all(|x| x.is_finite()) || !ctx.iter().any(|x| x.is_finite()) {
        return None;
    }
    let lay = preprocess::layout(cfg, &ctx, horizon);
    let target = preprocess::instance_norm_apply(&actual, lay.ls, cfg.use_arcsinh);
    Some(Example { features: lay.features, mask: lay.mask, n_ctx: lay.n_ctx, n_out: lay.n_out, target })
}

impl HostTrainable for Chronos2Train {
    type Example = Example;

    fn params(&self) -> &HashMap<String, Vec<f32>> {
        &self.w
    }

    fn params_mut(&mut self) -> &mut HashMap<String, Vec<f32>> {
        &mut self.w
    }

    fn loss(&self, ex: &Example) -> f32 {
        let (emb, _) = self.embed_forward(&ex.features, ex.n_ctx, ex.n_out);
        self.full_forward(&emb, &ex.mask, ex.n_out, &ex.target, &QUANTILES).0
    }

    fn loss_and_grad(&self, ex: &Example, grads: &mut HashMap<String, Vec<f32>>) -> f32 {
        let (emb, embed_cache) = self.embed_forward(&ex.features, ex.n_ctx, ex.n_out);
        let (loss, cache) = self.full_forward(&emb, &ex.mask, ex.n_out, &ex.target, &QUANTILES);
        let d_emb = self.full_backward(&cache, &ex.target, &QUANTILES, grads);
        self.embed_backward(&embed_cache, &d_emb, grads);
        loss
    }
}

fn build_examples(cfg: &Chronos2Config, series: &[Series], refs: &[WindowRef], context: usize, horizon: usize) -> Vec<Example> {
    refs.iter()
        .filter_map(|&wr| window_example(cfg, &train_data::extract(series, wr, context, horizon), context, horizon))
        .collect()
}

fn check_shape(cfg: &Chronos2Config, context: usize, horizon: usize) -> Result<(), String> {
    supported_horizon(cfg, horizon)?;
    if context == 0 || context > cfg.context_length {
        return Err(format!("context {context}: must be 1..={} (context_length)", cfg.context_length));
    }
    Ok(())
}

/// Mean loss of `weights` over (a subsample of) every window in `series` - the
/// number both sides of the promotion gate are measured with.
pub fn eval_universe_loss(cfg: &Chronos2Config, weights: &HashMap<String, Vec<f32>>, series: &[Series], context: usize, horizon: usize) -> f32 {
    let windows = train_data::enumerate_windows(series, context, horizon);
    if windows.is_empty() || check_shape(cfg, context, horizon).is_err() {
        return f32::NAN;
    }
    // Subsampled to a few hundred windows, as `timesfm3::finetune` does.
    let step = (windows.len() / 300).max(1);
    let refs: Vec<WindowRef> = windows.into_iter().step_by(step).collect();
    let examples = build_examples(cfg, series, &refs, context, horizon);
    host_train::mean_loss(&Chronos2Train::new(cfg.clone(), weights.clone()), &examples)
}

/// Fine-tune over a universe of series and return the gate's verdict plus the
/// tuned weights in the checkpoint's own tensor names.
///
/// The weights are returned whether or not the gate promoted, so the caller
/// decides what to do with a losing run; `report.promoted` is the only thing
/// that should decide whether they are WRITTEN.
pub fn finetune_universe(cfg: &Chronos2Config, base: &HashMap<String, Vec<f32>>, series: &[Series], context: usize, horizon: usize, split: train_data::SplitConfig, opts: &FinetuneOpts, observer: forecast::finetune::StepObserver) -> Result<forecast::FinetuneOutcome, String> {
    check_shape(cfg, context, horizon)?;
    let windows = train_data::enumerate_windows(series, context, horizon);
    let sp = train_data::temporal_split(series, &windows, horizon, split);
    let train = build_examples(cfg, series, &sp.train, context, horizon);
    let val = build_examples(cfg, series, &sp.val, context, horizon);

    let mut model = Chronos2Train::new(cfg.clone(), base.clone());
    if let Some(policy) = opts.saved {
        model.set_saved_policy(policy)?;
    }
    host_train::finetune_host(&mut model, &train, &val, opts, observer)
}

/// The licence of the Chronos-2 pretrained weights.
pub const WEIGHTS_LICENSE: &str = "apache-2.0";

/// The upstream those weights came from, recorded on every artifact derived
/// from them.
pub const UPSTREAM: &str = "amazon/chronos-2";

/// Read a base checkpoint's config and weights, host-side: a brain container
/// (see [`crate::import`]) or a reference Hugging Face directory.
pub fn load_base(path: &str) -> Result<(Chronos2Config, HashMap<String, Vec<f32>>), String> {
    if std::path::Path::new(path).is_dir() {
        let cfg = crate::import::config_from_dir(path)?;
        let weights = crate::import::load_hf(&cfg, path)?;
        return Ok((cfg, weights));
    }
    checkpoint::load_host_weights(path, |c| {
        let cfg = Chronos2Config::from_hf(c)?;
        let expected = cfg.param_list().into_iter().map(|(name, shape)| (name, shape.iter().product())).collect();
        Ok((cfg, expected))
    })
}

/// Write a fine-tuned checkpoint in the container [`crate::model::Chronos2::load`]
/// reads, with a [`checkpoint::st::ModelCard`] recording the licence and the
/// base it derives from.
pub fn save_weights(cfg: &Chronos2Config, weights: &HashMap<String, Vec<f32>>, path: &str) {
    // The container stores element counts, not param_list's real shape (the
    // convention of `crate::import::import`).
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let data = weights.get(&name).unwrap_or_else(|| panic!("missing tensor {name}")).clone();
            (name, vec![shape.iter().product::<usize>() as u64], data)
        })
        .collect();
    let mut card = checkpoint::st::ModelCard::new("chronos2-ft", "chronos2");
    card.architecture = Some("chronos2".into());
    card.license = Some(WEIGHTS_LICENSE.into());
    card.variant_of = Some(UPSTREAM.into());
    card.context_length = Some(cfg.context_length as u64);
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
    fn a_horizon_that_is_not_whole_output_patches_is_refused_by_name() {
        let cfg = Chronos2Config::tiny(); // output patch 4, 8 patches
        assert!(supported_horizon(&cfg, 4).is_ok() && supported_horizon(&cfg, 32).is_ok());
        for bad in [0, 3, 6, 36] {
            let e = supported_horizon(&cfg, bad).expect_err("must refuse");
            assert!(e.contains("output_patch_size"), "{e}");
        }
    }

    #[test]
    fn a_window_becomes_standardized_tokens_and_an_unusable_one_is_dropped() {
        let cfg = Chronos2Config::tiny();
        let s = vec![series(80, 0.0)];
        let w = train_data::extract(&s, WindowRef { series_idx: 0, origin: 40 }, 16, 8);
        let ex = window_example(&cfg, &w, 16, 8).expect("a clean window is usable");
        assert_eq!(ex.target.len(), ex.n_out * cfg.output_patch_size);
        assert_eq!(ex.features.len(), (ex.n_ctx + ex.n_out) * cfg.patch_feat_dim());
        assert!(ex.target.iter().all(|t| t.is_finite()));

        let mut bad = w.clone();
        bad.fut[3 * 5 + CLOSE] = f32::NAN;
        assert!(window_example(&cfg, &bad, 16, 8).is_none(), "a non-finite future bar has no target");
    }

    /// End to end on the host: the gate trains, reports two finite held-out
    /// numbers, and on a learnable series the tuned model beats the base.
    #[test]
    fn the_gate_trains_and_the_tuned_weights_beat_the_base_on_held_out_windows() {
        let cfg = Chronos2Config::tiny();
        let (context, horizon) = (16usize, 8usize);
        let s: Vec<Series> = (0..3).map(|i| series(200, i as f32)).collect();
        let base = init_weights(&cfg, 7);
        let opts = FinetuneOpts { epochs: 2, lr: 3e-3, wd: 0.0, clip: 1.0, lora: None, batch: 4, saved: None };
        let split = train_data::SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo: horizon };
        let (rep, w) = finetune_universe(&cfg, &base, &s, context, horizon, split, &opts, &mut |_| true).unwrap();

        assert!(rep.steps > 0, "the fixture produced no training windows");
        assert!(rep.base_val.is_finite() && rep.ft_val.is_finite(), "{rep:?}");
        assert!(rep.promoted && rep.ft_val < rep.base_val, "training on a learnable series must beat random weights: {rep:?}");
        let w = w.expect("weights are always returned");
        assert_eq!(w.len(), cfg.param_list().len());
        assert_ne!(w["encoder.final_layer_norm.weight"], base["encoder.final_layer_norm.weight"], "the weights moved");
        // the reported held-out number is the one eval_universe_loss measures
        assert!(eval_universe_loss(&cfg, &w, &s, context, horizon).is_finite());
    }

    /// A written checkpoint carries the licence and the base it derives from,
    /// reloads with every tensor, and (Apache-2.0) is publishable.
    #[test]
    fn a_written_checkpoint_carries_its_provenance_and_reloads() {
        let cfg = Chronos2Config::tiny();
        let w = init_weights(&cfg, 7);
        let dir = std::env::temp_dir().join(format!("chronos2-ft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("ft.safetensors").to_string_lossy().to_string();
        save_weights(&cfg, &w, &p);

        let card = checkpoint::st::read_card(&p).expect("readable").expect("a card was written");
        assert_eq!(card.license.as_deref(), Some(WEIGHTS_LICENSE));
        assert_eq!(card.variant_of.as_deref(), Some(UPSTREAM));
        assert!(checkpoint::license::redistributable(card.license.as_deref()).is_ok());

        let (back_cfg, back) = load_base(&p).expect("load_base reads what save_weights wrote");
        assert_eq!(back_cfg, cfg);
        assert_eq!(back, w);
        assert!(load_base(&dir.join("missing").to_string_lossy()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
