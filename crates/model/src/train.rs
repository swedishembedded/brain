// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One generic training/eval/sample loop over any [`Model`](crate::Model)
//! (ADR §3). [`fit`] is `gpt2::train::train` lifted to `M: Model` - same control
//! flow (cosine-with-warmup LR, grad accumulation with averaging, periodic eval,
//! resumable atomic checkpointing); [`generate`] is `gpt2::sample::generate`
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
/// `gpt2::model::IGNORE` so the masked-CE path is identical across models.
pub const IGNORE: u32 = 0xFFFF_FFFF;

/// Training-loop options (the CLI-facing hyperparameters), independent of any
/// particular architecture. This is `gpt2::train::TrainOpts` lifted to the model
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
    /// Wall-clock checkpoint cadence: once this many seconds have elapsed since
    /// the last save, the NEXT completed step writes a checkpoint (then the timer
    /// restarts). Decoupled from `eval_interval` so a slow big-model step never
    /// pays a 2.4 GB write every eval. `0` disables periodic saves (only the
    /// final one runs). Default 600 (10 min).
    pub checkpoint_secs: u64,
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
            checkpoint_secs: 600,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: false,
        }
    }
}

/// Cosine LR schedule with linear warmup (nanogpt's `get_lr`). Moved verbatim
/// from `gpt2::train::cosine_lr`.
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

/// Public masked-dataset loader for callers running their own training loop
/// (e.g. the full-vs-LoRA finetune comparison). Returns `(train, val, batch_cfg,
/// vocab)` — the token-level `train.mask.bin` (chat/tool-call) is honoured.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_dataset(
    dir: &Path,
    opts: &FitOpts,
) -> std::io::Result<(TokenDataset, TokenDataset, data::loader::BatchConfig, u32)> {
    let l = load(dir, opts)?;
    Ok((l.train, l.val, l.batch_cfg, l.vocab))
}

#[cfg(not(target_arch = "wasm32"))]
fn load(dir: &Path, opts: &FitOpts) -> std::io::Result<Loaded> {
    // Width-detecting read: `train.u32.bin` (large-vocab, e.g. Qwen) wins over
    // `train.bin` (u16, char/GPT-2), both surfaced as `u32`.
    let train_tok = binio::read_tokens_u32(&dir.join("train"))?;
    let val_tok = binio::read_tokens_u32(&dir.join("val"))?;

    // Vocab + mask/newline ids come from meta.json. Char datasets carry the full
    // id->char table (`itos`); large-vocab datasets carry only `vocab_size`. BPE
    // datasets without meta infer vocab from the max observed id.
    let (vocab, mask_id, newline_id, itos) = match std::fs::read_to_string(dir.join("meta.json")) {
        Ok(s) => {
            let meta = Meta::from_json(&s).map_err(std::io::Error::other)?;
            let stoi = meta.stoi();
            let mask_id = opts.mask_before.and_then(|c| stoi.get(&c).copied());
            let newline_id = stoi.get(&'\n').copied();
            let itos = if meta.itos.is_empty() { None } else { Some(meta.itos) };
            (meta.vocab_size as u32, mask_id, newline_id, itos)
        }
        Err(_) => {
            let maxid = train_tok.iter().chain(val_tok.iter()).copied().max().unwrap_or(0);
            (maxid + 1, None, None, None)
        }
    };

    // Chat / tool-call fine-tuning: a `train.mask.bin` (u8, from `data::chat`)
    // supervises only the assistant span at the TOKEN level, aligning windows to
    // the `<|endoftext|>` example separator. When present it takes precedence
    // over the char-boundary `mask_before_token`.
    let train_mask = binio::read_mask_bin(&dir.join("train.mask.bin")).ok();
    let val_mask = binio::read_mask_bin(&dir.join("val.mask.bin")).ok();
    let has_token_mask = train_mask.is_some();

    let batch_cfg = BatchConfig {
        batch_size: opts.batch_size as usize,
        block_size: opts.block_size as usize,
        mask_before_token: if has_token_mask { None } else { mask_id },
        mask_per_line: opts.mask_per_line,
        // A token-masked chat dataset aligns windows to the `<|endoftext|>` example
        // separator so each window starts at an example.
        align_to_lines: opts.align_to_lines || has_token_mask,
        newline_token: if has_token_mask { Some(data::chat::ENDOFTEXT) } else { newline_id },
    };
    // A split shorter than `block_size` has no valid sampling window at all
    // (`TokenDataset::sample_start`'s `data.len() - block_size - 1` requires
    // `data.len() > block_size`) -- validated HERE, at the point this data
    // enters the training loop, rather than left to surface as a
    // `data.len() - block_size` integer underflow deep in a later batch draw
    // (AGENTS.md: validate everything crossing into brain from outside, at
    // the point of entry). An EMPTY split (0 samples, e.g. no
    // `validation.jsonl`) is a deliberate, supported "skip this split"
    // signal elsewhere in the codebase and is not an error here.
    let too_short = |label: &str, tok: &[u32]| -> std::io::Result<()> {
        if !tok.is_empty() && tok.len() <= batch_cfg.block_size {
            return Err(std::io::Error::other(format!(
                "{label} split has {} token(s), too few for block_size {} (need more than block_size); \
                 reduce --block or add more {label} data",
                tok.len(),
                batch_cfg.block_size
            )));
        }
        Ok(())
    };
    too_short("train", &train_tok)?;
    too_short("validation", &val_tok)?;

    let mk = |tok: Vec<u32>, mask: Option<Vec<bool>>| match mask {
        Some(m) if m.len() == tok.len() => TokenDataset::new_with_mask(tok, m, &batch_cfg),
        _ => TokenDataset::new(tok, &batch_cfg),
    };
    Ok(Loaded {
        train: mk(train_tok, train_mask),
        val: mk(val_tok, val_mask),
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
/// This is `gpt2::train::train` lifted to `M: Model` - same control flow, same
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
    let mut last_save = std::time::Instant::now();

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
            println!("step {:>6}  lr {:.2e}  train {:.4}  eval {:.4}", step + 1, lr, last_train, eval_loss);
        }

        // Wall-clock checkpointing: once the timer has expired, the NEXT completed
        // step saves (atomic temp-rename), reports the save duration, and restarts
        // the timer. A slow big-model step thus never pays a per-eval 2.4 GB write.
        if let Some(p) = out {
            if opts.checkpoint_secs > 0 && last_save.elapsed().as_secs() >= opts.checkpoint_secs {
                let ts = std::time::Instant::now();
                model.save_with_itos(p.to_str().expect("utf-8 path"), loaded.itos.as_deref());
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
        model.save_with_itos(p.to_str().expect("utf-8 path"), loaded.itos.as_deref());
        println!("saved checkpoint -> {} ({:.1} s)", p.display(), ts.elapsed().as_secs_f64());
    }
    Ok((initial, last_train))
}

/// Generate `max_new` tokens continuing `prompt` for any token-head [`Model`].
/// Context is cropped to the model's block size. `temperature <= 0` selects
/// greedy argmax; `top_k = 0` disables top-k filtering. Lifted from
/// `gpt2::sample::generate`; depends only on [`Model::logits_all`].
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

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-model-train-load-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The exact defect a real `brain qwen finetune --lora` run hit: a
    /// dataset shorter than `block_size` used to load without error and then
    /// panic much later, deep in `TokenDataset::sample_start`, with a
    /// `data.len() - block_size - 1` usize underflow surfacing as a bizarre
    /// "index out of bounds" on a near-u64::MAX index -- nothing about that
    /// message points back at "your dataset is too small." `load_dataset`
    /// must instead reject it right here, by name, at the point this data
    /// enters the training loop.
    #[test]
    fn load_dataset_rejects_a_train_split_shorter_than_block_size_instead_of_panicking_later() {
        let dir = tmp("train-too-short");
        let tokens: Vec<u32> = (0..10).collect(); // 10 tokens, block_size will be 64
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(32)).unwrap();

        let opts = FitOpts { block_size: 64, ..Default::default() };
        let Err(err) = load_dataset(&dir, &opts) else { panic!("expected an error for a too-short train split") };
        let msg = err.to_string();
        assert!(msg.contains("train"), "{msg}");
        assert!(msg.contains("10"), "{msg}");
        assert!(msg.contains("64"), "{msg}");
    }

    #[test]
    fn load_dataset_accepts_an_empty_validation_split_as_the_deliberate_skip_eval_signal() {
        let dir = tmp("val-empty-ok");
        let tokens: Vec<u32> = (0..100).collect();
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(32)).unwrap();

        let opts = FitOpts { block_size: 16, ..Default::default() };
        load_dataset(&dir, &opts).expect("an empty validation split must not be treated as too-short");
    }

    #[test]
    fn load_dataset_rejects_a_nonempty_validation_split_shorter_than_block_size() {
        let dir = tmp("val-too-short");
        let train: Vec<u32> = (0..100).collect();
        let val: Vec<u32> = (0..5).collect();
        binio::write_u32_bin(&dir.join("train.u32.bin"), &train).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &val).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(32)).unwrap();

        let opts = FitOpts { block_size: 16, ..Default::default() };
        let Err(err) = load_dataset(&dir, &opts) else { panic!("expected an error for a too-short validation split") };
        assert!(err.to_string().contains("validation"), "{err}");
    }
}
