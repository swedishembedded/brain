// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One generic training/eval/sample loop over any [`Model`](crate::Model)
//! (ADR §3). [`fit`] is `gpt::train::train` lifted to `M: Model` — same control
//! flow (cosine-with-warmup LR, grad accumulation with averaging, periodic eval,
//! resumable atomic checkpointing); [`generate`] is `gpt::sample::generate`
//! lifted to any token-head model.

use data::rng::Rng;

use crate::{Model, ModelConfig};
#[cfg(not(target_arch = "wasm32"))]
use crate::Batch;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use data::binio::{self, Meta};
#[cfg(not(target_arch = "wasm32"))]
use data::loader::{BatchConfig, TokenDataset};

/// Cross-entropy ignore index (masked target positions). The data loader emits
/// `-1` as `i32`; reinterpreted as `u32` that is exactly this value. Mirrors
/// `gpt::model::IGNORE` so the masked-CE path is identical across models.
pub const IGNORE: u32 = 0xFFFF_FFFF;

/// Training-loop options (the CLI-facing hyperparameters), independent of any
/// particular architecture. This is `gpt::train::TrainOpts` lifted to the model
/// crate.
#[derive(Clone, Debug)]
pub struct FitOpts {
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

impl Default for FitOpts {
    fn default() -> Self {
        FitOpts {
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

/// Cosine LR schedule with linear warmup (nanogpt's `get_lr`). Moved verbatim
/// from `gpt::train::cosine_lr`.
pub fn cosine_lr(it: u32, opts: &FitOpts) -> f32 {
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
#[cfg(not(target_arch = "wasm32"))]
struct Loaded {
    train: TokenDataset,
    val: TokenDataset,
    vocab: u32,
    batch_cfg: BatchConfig,
    /// Char-tokenizer vocab (when the dataset has `meta.json`), embedded into the
    /// checkpoint so inference needs no dataset reference.
    itos: Option<Vec<char>>,
}

#[cfg(not(target_arch = "wasm32"))]
fn load(dir: &Path, opts: &FitOpts) -> std::io::Result<Loaded> {
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
#[cfg(not(target_arch = "wasm32"))]
fn targets_to_u32(y: &[i32]) -> Vec<u32> {
    y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect()
}

/// Train any [`Model`] on the token dataset in `dir`, writing the final
/// checkpoint to `out`. `cfg` carries the architecture; its `vocab`/`block_size`
/// are overridden from the dataset and `opts`. Returns `(initial_loss,
/// final_loss)`.
///
/// This is `gpt::train::train` lifted to `M: Model` — same control flow, same
/// resume/eval/checkpoint semantics, no GPT-specific code.
///
/// Native-only: it reads token `.bin` datasets and writes checkpoints, neither
/// of which exists on the wasm32 inference build.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit<M: Model>(
    dir: &Path,
    cfg: M::Config,
    opts: &FitOpts,
    out: Option<&Path>,
) -> std::io::Result<(f32, f32)> {
    let loaded = load(dir, opts)?;

    // Resume from the existing checkpoint if `out` already exists, so repeated
    // runs continue rather than restart from scratch. The checkpoint's
    // architecture wins (and must match the dataset/--block in use). Otherwise
    // start from a fresh random init. (Weights resume; AdamW moments restart.)
    let resume = out.map(|p| p.exists()).unwrap_or(false);
    let (cfg, init) = if resume {
        let p = out.unwrap();
        println!("resuming from existing checkpoint {}", p.display());
        let c = checkpoint::load(p.to_str().expect("utf-8 path"));
        let rcfg = M::Config::from_json(&c.header["config"]);
        assert_eq!(
            rcfg.block_size(),
            opts.block_size,
            "checkpoint block_size {} != --block {} — resume with the same --block",
            rcfg.block_size(),
            opts.block_size
        );
        assert_eq!(rcfg.vocab(), loaded.vocab, "checkpoint vocab != dataset vocab — wrong dataset for this checkpoint");
        let init = c.by_role("");
        (rcfg, init)
    } else {
        let cfg = cfg.finalize_for_dataset(loaded.vocab, opts.block_size);
        let init = M::init_weights(&cfg, opts.seed);
        (cfg, init)
    };
    let model = M::new(cfg, opts.batch_size, opts.block_size, &init);
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);

    let sample_loss = |model: &M, ds: &TokenDataset, rng: &mut Rng, batches: u32| -> f32 {
        let mut total = 0.0;
        for _ in 0..batches.max(1) {
            let (x, y) = ds.get_batch(&loaded.batch_cfg, rng);
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
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
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
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

/// Generate `max_new` tokens continuing `prompt` for any token-head [`Model`].
/// Context is cropped to the model's block size. `temperature <= 0` selects
/// greedy argmax; `top_k = 0` disables top-k filtering. Lifted from
/// `gpt::sample::generate`; depends only on [`Model::logits_all`].
pub fn generate<M: Model>(
    m: &M,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut Rng,
) -> Vec<u32> {
    let block = m.config().block_size() as usize;
    let vocab = m.config().vocab() as usize;
    let mut ctx: Vec<u32> = prompt.to_vec();
    let mut out = Vec::with_capacity(max_new);

    for _ in 0..max_new {
        let window: Vec<u32> = if ctx.len() > block {
            ctx[ctx.len() - block..].to_vec()
        } else {
            ctx.clone()
        };
        let logits = m.logits_all(&window).expect("token head");
        // last position's vocab logits
        let last = &logits[logits.len() - vocab..];
        let next = sample_logits(last, temperature, top_k, rng);
        ctx.push(next);
        out.push(next);
    }
    out
}

fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits) as u32;
    }
    // temperature scale
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // top-k: keep only the k largest logits, rest -> -inf
    if top_k > 0 && top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let threshold = scaled[idx[top_k - 1]];
        for v in scaled.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // softmax (numerically stable) then inverse-CDF sample
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in scaled.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_lr_warmup_peak_and_floor() {
        let o = FitOpts { lr: 1.0, min_lr: 0.1, warmup: 10, decay_iters: 100, ..Default::default() };
        assert!(cosine_lr(0, &o) < cosine_lr(5, &o)); // ramping up
        assert!((cosine_lr(9, &o) - 1.0).abs() < 0.11); // near peak at end of warmup
        assert!((cosine_lr(200, &o) - 0.1).abs() < 1e-6); // floor after decay
    }

    #[test]
    fn sample_logits_greedy_picks_argmax() {
        let mut rng = Rng::new(0);
        let logits = [0.1, 5.0, 0.2, -1.0];
        assert_eq!(sample_logits(&logits, 0.0, 0, &mut rng), 1);
    }
}
