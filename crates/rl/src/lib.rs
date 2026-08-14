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
pub mod continuous;

use std::path::Path;

use data::binio;
use data::loader::TokenDataset;
use data::rng::Rng;
use model::{cosine_lr, Batch, FitOpts, Model, ModelConfig, IGNORE};

/// i32 targets from the loader (`-1` = ignore) reinterpreted as the model's
/// `u32` IGNORE sentinel. Mirrors the one-line private helper of the same
/// name in `model::train` - not worth widening that crate's public surface
/// for a single reused line.
fn targets_to_u32(y: &[i32]) -> Vec<u32> {
    y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect()
}

/// Load `dir`'s dataset the same way [`model::train::load_dataset`] does,
/// then attach `train.weight.bin`/`val.weight.bin` (per-token `f32`,
/// [`data::binio::read_f32_bin`]) when present. Neither file existing is not
/// an error - see this module's doc comment on the default-1.0 semantics.
fn load_weighted(dir: &Path, opts: &FitOpts) -> std::io::Result<(TokenDataset, TokenDataset, data::loader::BatchConfig, u32)> {
    let (train, val, batch_cfg, vocab) = model::load_dataset(dir, opts)?;
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
    Ok((train, val, batch_cfg, vocab))
}

/// Train any weighted-loss-capable [`Model`] on the weighted dataset in
/// `dir`, writing the final checkpoint to `out`. Same resume/eval/checkpoint
/// semantics as [`model::train::fit`] (reused, not duplicated) - the only
/// difference is every batch is [`Batch::LmWeighted`] instead of
/// [`Batch::Lm`], and the model is switched into weighted-loss mode via
/// [`Model::enable_weighted_loss`] right after construction. Returns
/// `(initial_loss, final_loss)` - both are the WEIGHTED loss (see
/// [`Model::forward`]'s contract on a weighted-loss-enabled model).
///
/// Panics (via [`Model::enable_weighted_loss`]'s default) if `M` has not
/// implemented weighted-loss support - a clear, immediate failure rather
/// than silently training unweighted.
pub fn fit_weighted<M: Model>(dir: &Path, cfg: M::Config, opts: &FitOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    let (train, val, batch_cfg, vocab) = load_weighted(dir, opts)?;

    let resume = out.map(|p| p.exists()).unwrap_or(false);
    let (cfg, init) = if resume {
        let p = out.unwrap();
        println!("resuming from existing checkpoint {}", p.display());
        let c = checkpoint::load(p.to_str().expect("utf-8 path"));
        let rcfg = M::Config::from_json(&c.header["config"]);
        assert_eq!(
            rcfg.block_size(),
            opts.block_size,
            "checkpoint block_size {} != --block {} - resume with the same --block",
            rcfg.block_size(),
            opts.block_size
        );
        assert_eq!(rcfg.vocab(), vocab, "checkpoint vocab != dataset vocab - wrong dataset for this checkpoint");
        (rcfg, c.by_role(""))
    } else {
        let cfg = cfg.finalize_for_dataset(vocab, opts.block_size);
        let init = M::init_weights(&cfg, opts.seed);
        (cfg, init)
    };
    let mut model = M::new(cfg, opts.batch_size, opts.block_size, &init);
    model.enable_weighted_loss();
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);

    let sample_loss = |model: &M, ds: &TokenDataset, rng: &mut Rng, batches: u32| -> f32 {
        let mut total = 0.0;
        for _ in 0..batches.max(1) {
            let (x, y, w) = ds.get_batch_weighted(&batch_cfg, rng);
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::LmWeighted { tokens: &x, targets: &targets, weights: &w });
            total += model.forward();
        }
        total / batches.max(1) as f32
    };

    let initial = sample_loss(&model, &train, &mut rng.clone(), 5);
    let mut last_train = initial;
    let mut last_save = std::time::Instant::now();

    for step in 0..opts.steps {
        let lr = cosine_lr(step, opts);
        model.zero_grads();
        let mut step_loss = 0.0;
        for _ in 0..opts.grad_accum.max(1) {
            let (x, y, w) = train.get_batch_weighted(&batch_cfg, &mut rng);
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::LmWeighted { tokens: &x, targets: &targets, weights: &w });
            step_loss += model.forward();
            model.backward();
        }
        let scale = 1.0 / opts.grad_accum.max(1) as f32;
        let clip = (opts.grad_clip > 0.0).then_some(opts.grad_clip);
        model.adamw_step(step + 1, lr, opts.weight_decay, clip, scale);
        model.poll_wait();
        last_train = step_loss / opts.grad_accum.max(1) as f32;

        if opts.eval_interval > 0 && (step + 1) % opts.eval_interval == 0 {
            let eval_loss = sample_loss(&model, &val, &mut rng.clone(), opts.eval_batches);
            println!("step {:>6}  lr {:.2e}  train {:.4}  eval {:.4}", step + 1, lr, last_train, eval_loss);
        }

        if let Some(p) = out {
            if opts.checkpoint_secs > 0 && last_save.elapsed().as_secs() >= opts.checkpoint_secs {
                let ts = std::time::Instant::now();
                model.save(p.to_str().expect("utf-8 path"));
                println!(
                    "step {:>6}  saved checkpoint -> {} ({:.1} s)",
                    step + 1,
                    p.display(),
                    ts.elapsed().as_secs_f64()
                );
                last_save = std::time::Instant::now();
            }
        }
    }

    if let Some(p) = out {
        let ts = std::time::Instant::now();
        model.save(p.to_str().expect("utf-8 path"));
        println!("saved checkpoint -> {} ({:.1} s)", p.display(), ts.elapsed().as_secs_f64());
    }
    Ok((initial, last_train))
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
        let (train, _val, batch_cfg, _vocab) = load_weighted(&dir, &opts).expect("load_weighted");
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
        let (train, _val, batch_cfg, _vocab) = load_weighted(&dir, &opts).expect("load_weighted");
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
