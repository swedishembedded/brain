// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic continuous/reward-driven training over any [`model::Model`] that
//! opts into [`model::Model::enable_weighted_loss`].
//!
//! [`fit_weighted`] is `model::train::fit` lifted to weighted batches: same
//! control flow (cosine-with-warmup LR, grad accumulation, periodic eval,
//! resumable checkpointing - all reused from `model::train` directly, not
//! re-implemented), the one difference being that every batch carries a
//! per-position reward/advantage weight (`model::Batch::LmWeighted`) instead
//! of implicit uniform weight. No architecture-specific code lives here -
//! `qwen3` is simply the first `M: Model` this gets instantiated with (see
//! `crates/rl/tests/qwen3_fit_weighted.rs`); any other `Model` that
//! implements `enable_weighted_loss` (today only `qwen3`) can use this
//! unchanged.
//!
//! ## Weight file format
//!
//! A dataset directory's optional `train.weight.bin`/`val.weight.bin` (raw
//! `f32`, [`data::binio::read_f32_bin`]/[`data::binio::write_f32_bin`] - no
//! new file format) carries one weight per TOKEN in the corresponding
//! `train`/`val` split, parallel to `train.u32.bin`. Absent means every
//! position implicitly weights `1.0` ([`data::loader::TokenDataset::
//! get_batch_weighted`]'s own default), so a dataset directory produced for
//! ordinary [`model::train::fit`] also works here unchanged.
//!
//! ## What is NOT here yet
//!
//! Turning real ATIF trajectories (`atif::Trajectory`) into a weighted
//! dataset directory in exactly that format is self-improve roadmap **P5**
//! - see the [`atif`] module.

pub mod atif;
pub mod continual;
#[cfg(feature = "qwen3")]
pub mod continuous;
pub mod curriculum;
pub mod env;
pub mod gate;
pub mod improve;
pub mod objective;

use std::path::Path;

use data::binio;
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use model::{Batch, FitOpts, Model, Objective, IGNORE};

/// i32 targets from the loader (`-1` = ignore) reinterpreted as the model's
/// `u32` IGNORE sentinel. Mirrors the one-line private helper of the same
/// name in `model::train` - not worth widening that crate's public surface
/// for a single reused line.
fn targets_to_u32(y: &[i32]) -> Vec<u32> {
    y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect()
}

/// Load `dir`'s dataset the same way [`model::train::load_dataset_with_itos`]
/// does, then attach `train.weight.bin`/`val.weight.bin` (per-token `f32`,
/// [`data::binio::read_f32_bin`]) when present. Neither file existing is not
/// an error - see this module's doc comment on the default-1.0 semantics.
/// Also threads through the dataset's char-tokenizer `itos` (when present) so
/// [`fit_weighted`] can carry it into its own checkpoint, same as
/// [`model::train::fit`] does.
#[allow(clippy::type_complexity)]
fn load_weighted(dir: &Path, opts: &FitOpts) -> std::io::Result<(TokenDataset, TokenDataset, BatchConfig, u32, Option<Vec<char>>)> {
    let (train, val, batch_cfg, vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let attach = |ds: TokenDataset, weight_path: &Path, expected_len: usize| -> std::io::Result<TokenDataset> {
        match binio::read_f32_bin(weight_path) {
            Ok(w) if w.len() == expected_len => Ok(ds.with_weights(w)),
            Ok(w) => Err(std::io::Error::other(format!(
                "{}: {} weights but the token split has a different length ({expected_len}) -- weight file must be parallel to the token file, one f32 per token",
                weight_path.display(),
                w.len()
            ))),
            Err(_) => Ok(ds), // no weight file: every position implicitly weights 1.0
        }
    };
    let train_len = train.len();
    let val_len = val.len();
    let train = attach(train, &dir.join("train.weight.bin"), train_len)?;
    let val = attach(val, &dir.join("val.weight.bin"), val_len)?;
    Ok((train, val, batch_cfg, vocab, itos))
}

/// Weighted/reward-driven training - [`fit_weighted`]'s [`Objective`]. One
/// micro-step draws a `Batch::LmWeighted` batch (per-token weight from the
/// dataset's optional weight file, default `1.0`) from the train split,
/// forwards, and backwards; eval samples the held-out val split the same way,
/// forward-only. This is exactly the step body `fit_weighted` used to run
/// inline, unchanged, now behind the [`Objective`] seam.
struct WeightedLm {
    train: TokenDataset,
    val: TokenDataset,
    batch_cfg: BatchConfig,
    itos: Option<Vec<char>>,
}

impl<M: Model> Objective<M> for WeightedLm {
    fn regime(&self) -> &'static str {
        "weighted_lm"
    }

    fn prepare(&mut self, model: &mut M) {
        model.enable_weighted_loss();
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let (x, y, w) = self.train.get_batch_weighted(&self.batch_cfg, rng);
        let targets = targets_to_u32(&y);
        model.set_batch(Batch::LmWeighted { tokens: &x, targets: &targets, weights: &w });
        let loss = model.forward();
        model.backward();
        loss
    }

    fn eval(&mut self, model: &M, rng: &mut Rng, batches: u32) -> Option<f32> {
        let mut total = 0.0;
        for _ in 0..batches.max(1) {
            let (x, y, w) = self.val.get_batch_weighted(&self.batch_cfg, rng);
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::LmWeighted { tokens: &x, targets: &targets, weights: &w });
            total += model.forward();
        }
        Some(total / batches.max(1) as f32)
    }

    fn itos(&self) -> Option<&[char]> {
        self.itos.as_deref()
    }
}

/// Train any weighted-loss-capable [`Model`] on the weighted dataset in
/// `dir`, writing the final checkpoint to `out`. Same resume/eval/checkpoint
/// semantics as [`model::train::fit`] (reused via [`model::build_or_resume`] +
/// [`model::fit_with`], not duplicated) - the only difference is every batch
/// is [`Batch::LmWeighted`] instead of [`Batch::Lm`], via the [`WeightedLm`]
/// objective, which also opts the model into weighted-loss mode via
/// [`Model::enable_weighted_loss`] right after construction. Returns
/// `(initial_loss, final_loss)` - both are the WEIGHTED loss (see
/// [`Model::forward`]'s contract on a weighted-loss-enabled model).
///
/// Always calls [`Model::save_with_itos`] (never `save`) with the dataset's
/// `itos` when it has one - fixing a real bug this used to have: it called
/// `save` where `model::train::fit` calls `save_with_itos`, so every
/// weighted checkpoint silently lost its char vocab.
///
/// Panics (via [`Model::enable_weighted_loss`]'s default) if `M` has not
/// implemented weighted-loss support - a clear, immediate failure rather
/// than silently training unweighted.
pub fn fit_weighted<M: Model>(dir: &Path, cfg: M::Config, opts: &FitOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    let (train, val, batch_cfg, vocab, itos) = load_weighted(dir, opts)?;
    let model = model::build_or_resume::<M>(cfg, opts, out, vocab);
    let obj = WeightedLm { train, val, batch_cfg, itos };
    model::fit_with(model, obj, opts, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::binio::Meta;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-rl-load-weighted-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn load_weighted_attaches_the_weight_file_when_present() {
        let dir = tmp("attach");
        let tokens: Vec<u32> = (0..100).map(|i| i % 8).collect();
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(8)).unwrap();
        // Deliberately non-uniform so a bug that silently defaults to 1.0
        // everywhere (e.g. never reading the file) would be caught.
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        binio::write_f32_bin(&dir.join("train.weight.bin"), &weights).unwrap();

        let opts = FitOpts { block_size: 8, batch_size: 4, ..Default::default() };
        let (train, _val, batch_cfg, _vocab, _itos) = load_weighted(&dir, &opts).expect("load_weighted");
        let mut rng = Rng::new(3);
        let (x, _y, w) = train.get_batch_weighted(&batch_cfg, &mut rng);
        // The weight at each gathered position must equal 0.5 * (token id at
        // that position's absolute offset + 1) per how the test data was
        // constructed - i.e. NOT all 1.0.
        assert!(w.iter().any(|&wi| wi != 1.0), "expected the attached (non-uniform) weight file's values, not the no-file default");
        assert_eq!(x.len(), w.len());
    }

    #[test]
    fn load_weighted_defaults_to_uniform_1_when_no_weight_file_exists() {
        let dir = tmp("no-file");
        let tokens: Vec<u32> = (0..100).map(|i| i % 8).collect();
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(8)).unwrap();

        let opts = FitOpts { block_size: 8, batch_size: 4, ..Default::default() };
        let (train, _val, batch_cfg, _vocab, _itos) = load_weighted(&dir, &opts).expect("load_weighted");
        let mut rng = Rng::new(3);
        let (_x, _y, w) = train.get_batch_weighted(&batch_cfg, &mut rng);
        assert!(w.iter().all(|&wi| wi == 1.0), "an ordinary model::train::fit dataset dir (no weight file) must train exactly as unweighted");
    }

    #[test]
    fn load_weighted_rejects_a_weight_file_whose_length_does_not_match_the_token_split() {
        let dir = tmp("mismatch");
        let tokens: Vec<u32> = (0..100).map(|i| i % 8).collect();
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(8)).unwrap();
        binio::write_f32_bin(&dir.join("train.weight.bin"), &[1.0, 2.0, 3.0]).unwrap();

        let opts = FitOpts { block_size: 8, batch_size: 4, ..Default::default() };
        let Err(err) = load_weighted(&dir, &opts) else { panic!("expected a length-mismatch error") };
        assert!(err.to_string().contains("weights"), "{err}");
    }
}
