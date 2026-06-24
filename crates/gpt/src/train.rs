// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPT training loop: ties [`crate::model::Gpt`] to the `data` crate
//! (`TokenDataset` + masking/alignment), with AdamW, cosine-with-warmup LR,
//! gradient accumulation, periodic eval, and checkpointing — a port of
//! nanogpt's trainer for brain's WGSL engine.

use std::path::Path;

use data::binio::{self, Meta};
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;

use crate::init::init_weights;
use crate::model::{Gpt, GptConfig, IGNORE};

/// Training hyperparameters (CLI-facing).
#[derive(Clone, Debug)]
pub struct TrainOpts {
    pub steps: u32,
    pub batch_size: u32,
    pub block_size: u32,
    pub lr: f32,
    pub min_lr: f32,
    pub warmup: u32,
    pub decay_iters: u32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub grad_accum: u32,
    pub eval_interval: u32,
    pub eval_batches: u32,
    pub seed: u64,
    /// Mask loss up to & including this char (e.g. `'='` for calculator).
    pub mask_before: Option<char>,
    pub mask_per_line: bool,
    pub align_to_lines: bool,
}

impl Default for TrainOpts {
    fn default() -> Self {
        TrainOpts {
            steps: 2000,
            batch_size: 32,
            block_size: 64,
            lr: 3e-4,
            min_lr: 3e-5,
            warmup: 100,
            decay_iters: 2000,
            weight_decay: 0.1,
            grad_clip: 1.0,
            grad_accum: 1,
            eval_interval: 250,
            eval_batches: 20,
            seed: 1337,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: false,
        }
    }
}

/// Cosine LR schedule with linear warmup (nanogpt's `get_lr`).
pub fn cosine_lr(it: u32, opts: &TrainOpts) -> f32 {
    if it < opts.warmup {
        return opts.lr * (it + 1) as f32 / opts.warmup.max(1) as f32;
    }
    if it >= opts.decay_iters {
        return opts.min_lr;
    }
    let ratio = (it - opts.warmup) as f32 / (opts.decay_iters - opts.warmup).max(1) as f32;
    let coeff = 0.5 * (1.0 + (std::f32::consts::PI * ratio).cos());
    opts.min_lr + coeff * (opts.lr - opts.min_lr)
}

/// A loaded char/BPE dataset: train/val token splits + optional vocab metadata.
struct Loaded {
    train: TokenDataset,
    val: TokenDataset,
    vocab: u32,
    batch_cfg: BatchConfig,
    /// Char-tokenizer vocab (when the dataset has `meta.json`), embedded into the
    /// checkpoint so inference (`gpt gen`) needs no dataset reference.
    itos: Option<Vec<char>>,
}

fn load(dir: &Path, opts: &TrainOpts) -> std::io::Result<Loaded> {
    let train_tok = binio::read_u16_bin(&dir.join("train.bin"))?;
    let val_tok = binio::read_u16_bin(&dir.join("val.bin"))?;

    // Vocab + mask/newline ids come from meta.json (char datasets). BPE has no
    // meta; vocab is GPT-2's 50257 and masking/alignment are unsupported there.
    let (vocab, mask_id, newline_id, itos) = match std::fs::read_to_string(dir.join("meta.json")) {
        Ok(s) => {
            let meta = Meta::from_json(&s).map_err(std::io::Error::other)?;
            let stoi = meta.stoi();
            let mask_id = opts.mask_before.and_then(|c| stoi.get(&c).copied());
            let newline_id = stoi.get(&'\n').copied();
            (meta.vocab_size as u32, mask_id, newline_id, Some(meta.itos))
        }
        Err(_) => {
            let maxid = train_tok.iter().chain(val_tok.iter()).copied().max().unwrap_or(0);
            (maxid as u32 + 1, None, None, None)
        }
    };

    let batch_cfg = BatchConfig {
        batch_size: opts.batch_size as usize,
        block_size: opts.block_size as usize,
        mask_before_token: mask_id,
        mask_per_line: opts.mask_per_line,
        align_to_lines: opts.align_to_lines,
        newline_token: newline_id,
    };
    Ok(Loaded {
        train: TokenDataset::new(train_tok, &batch_cfg),
        val: TokenDataset::new(val_tok, &batch_cfg),
        vocab,
        batch_cfg,
        itos,
    })
}

/// i32 targets from the loader (`-1` = ignore) reinterpreted as the model's
/// `u32` IGNORE sentinel.
fn targets_to_u32(y: &[i32]) -> Vec<u32> {
    y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect()
}

/// Train a GPT on the dataset in `dir`, writing the final checkpoint to `out`.
/// `cfg` carries the architecture; its `vocab`/`block_size` are overridden from
/// the dataset and `opts`. Returns `(initial_train_loss, final_train_loss)`.
pub fn train(dir: &Path, mut cfg: GptConfig, opts: &TrainOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    let loaded = load(dir, opts)?;

    // Resume from the existing checkpoint if `out` already exists, so repeated
    // `train` runs continue rather than restart from scratch. The checkpoint's
    // architecture wins (and must match the dataset/--block in use). Otherwise
    // start from a fresh random init. (Weights resume; AdamW moments restart.)
    let resume = out.map(|p| p.exists()).unwrap_or(false);
    let (cfg, init) = if resume {
        let p = out.unwrap();
        println!("resuming from existing checkpoint {}", p.display());
        let c = checkpoint::load(p.to_str().expect("utf-8 path"));
        let rcfg = GptConfig::from_json(&c.header["config"]);
        assert_eq!(
            rcfg.block_size, opts.block_size,
            "checkpoint block_size {} != --block {} — resume with the same --block",
            rcfg.block_size, opts.block_size
        );
        assert_eq!(rcfg.vocab, loaded.vocab, "checkpoint vocab != dataset vocab — wrong dataset for this checkpoint");
        (rcfg, c.by_role(""))
    } else {
        cfg.vocab = loaded.vocab;
        cfg.block_size = opts.block_size;
        let cfg = cfg.with_ff_default();
        let init = init_weights(&cfg, opts.seed);
        (cfg, init)
    };
    let model = Gpt::new(cfg, opts.batch_size, opts.block_size, &init);
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);

    let sample_loss = |model: &Gpt, ds: &TokenDataset, rng: &mut Rng, batches: u32| -> f32 {
        let mut total = 0.0;
        for _ in 0..batches.max(1) {
            let (x, y) = ds.get_batch(&loaded.batch_cfg, rng);
            model.set_batch(&x, &targets_to_u32(&y));
            total += model.forward();
        }
        total / batches.max(1) as f32
    };

    let initial = sample_loss(&model, &loaded.train, &mut rng.clone(), 5);
    let mut last_train = initial;

    for step in 0..opts.steps {
        let lr = cosine_lr(step, opts);
        model.zero_grads();
        let mut step_loss = 0.0;
        for _ in 0..opts.grad_accum.max(1) {
            let (x, y) = loaded.train.get_batch(&loaded.batch_cfg, &mut rng);
            model.set_batch(&x, &targets_to_u32(&y));
            step_loss += model.forward();
            model.backward();
        }
        // average grads over accumulation steps
        let scale = 1.0 / opts.grad_accum.max(1) as f32;
        let clip = (opts.grad_clip > 0.0).then_some(opts.grad_clip);
        model.adamw_step(step + 1, lr, opts.weight_decay, clip, scale);
        model.poll_wait();
        last_train = step_loss / opts.grad_accum.max(1) as f32;

        if opts.eval_interval > 0 && (step + 1) % opts.eval_interval == 0 {
            let eval_loss = sample_loss(&model, &loaded.val, &mut rng.clone(), opts.eval_batches);
            // Checkpoint at every eval point so long runs are resumable and a
            // crash loses at most `eval_interval` steps. The write is atomic
            // (checkpoint::save renames a temp over the target).
            let saved = match out {
                Some(p) => {
                    model.save_with_itos(p.to_str().expect("utf-8 path"), loaded.itos.as_deref());
                    format!("  saved -> {}", p.display())
                }
                None => String::new(),
            };
            println!("step {:>6}  lr {:.2e}  train {:.4}  eval {:.4}{saved}", step + 1, lr, last_train, eval_loss);
        }
    }

    if let Some(p) = out {
        model.save_with_itos(p.to_str().expect("utf-8 path"), loaded.itos.as_deref());
        println!("saved checkpoint -> {}", p.display());
    }
    Ok((initial, last_train))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_lr_warmup_peak_and_floor() {
        let o = TrainOpts {
            lr: 1.0,
            min_lr: 0.1,
            warmup: 10,
            decay_iters: 100,
            ..Default::default()
        };
        assert!(cosine_lr(0, &o) < cosine_lr(5, &o)); // ramping up
        assert!((cosine_lr(9, &o) - 1.0).abs() < 0.11); // near peak at end of warmup
        assert!((cosine_lr(200, &o) - 0.1).abs() < 1e-6); // floor after decay
    }

    #[test]
    fn trains_calculator_and_reduces_loss() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        // Prepare a tiny calculator dataset.
        let dir = std::env::temp_dir().join(format!("brain_gpt_train_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        data::prepare::prepare(data::prepare::Dataset::Calculator, &dir, 4000, 1).unwrap();

        let cfg = GptConfig { vocab: 0, block_size: 32, n_layers: 2, d_model: 64, n_heads: 4, d_ff: 256 };
        let opts = TrainOpts {
            steps: 200,
            batch_size: 16,
            block_size: 32,
            lr: 3e-3,
            warmup: 20,
            decay_iters: 200,
            eval_interval: 0,
            mask_before: Some('='),
            mask_per_line: true,
            ..Default::default()
        };
        let (initial, final_loss) = train(&dir, cfg, &opts, None).unwrap();
        assert!(final_loss.is_finite() && initial.is_finite());
        assert!(final_loss < initial * 0.9, "loss did not drop: {initial} -> {final_loss}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
