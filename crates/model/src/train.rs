// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One generic training/eval/sample loop over any [`Model`](crate::Model)
//! (ADR §3). [`fit`] is `gpt2::train::train` lifted to `M: Model` - same control
//! flow (cosine-with-warmup LR, grad accumulation with averaging, periodic eval,
//! resumable atomic checkpointing); [`generate`] is `gpt2::sample::generate`
//! lifted to any token-head model.

use data::rng::Rng;

use crate::objective::Objective;
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
    /// final one runs). Default 600 seconds.
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

/// A learning rate as a function of the step number: linear warmup to `peak`,
/// `hold` steps at `peak`, then cosine decay to `floor` by `decay_iters`,
/// holding `floor` after that. nanogpt's `get_lr` (which is the `hold == 0`
/// case) - this tree already computed that curve inside [`cosine_lr`] and,
/// separately spelled, inside `lfm2::train`, so it is named here as its own
/// value a trainer that does not go through [`fit`]'s `FitOpts` can hold.
///
/// **Why a run decays at all.** Adam at a CONSTANT rate does not converge to a
/// minimum on a stochastic objective; it converges to a ball around one whose
/// radius is set by the step size times the gradient noise. The visible
/// signature is a fast initial descent and then a flat line at a loss ABOVE
/// what the same trajectory reaches once the rate comes down.
///
/// **Why `hold` exists, rather than cosine from step 0.** Hägele et al.
/// (arXiv:2405.18392) measure constant-rate training followed by a short
/// COOLDOWN against a full cosine and find the two scale "predictably and
/// reliably" alike, with the cooldown's benefit saturating at roughly 20% of
/// the run. That shape has a property a full cosine does not: the rate for
/// most of the run does not depend on the step budget, so a run that is later
/// extended has not already annealed itself on the strength of a total it was
/// going to beat. Cosine from step 0 bakes the budget into every step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LrSchedule {
    /// The rate the warmup ramps up to, is held at, and the decay starts from.
    pub peak: f32,
    /// The rate held from `decay_iters` on. Never 0 by default: a rate that
    /// reaches exactly zero stops training some steps before the run ends.
    pub floor: f32,
    /// Steps of linear ramp `peak/warmup .. peak`. `0` starts at `peak`.
    pub warmup: u32,
    /// Steps held AT `peak` after the warmup and before the cooldown starts.
    /// `0` is the plain warmup-then-cosine curve.
    pub hold: u32,
    /// The step by which the decay has reached `floor` - normally the run's
    /// total step count, so the LAST trained step is the slowest one.
    pub decay_iters: u32,
}

impl LrSchedule {
    /// The rate `lr` at every step - no warmup, no decay.
    pub fn constant(lr: f32) -> LrSchedule {
        LrSchedule { peak: lr, floor: lr, warmup: 0, hold: 0, decay_iters: 0 }
    }

    /// The step the cooldown starts at.
    pub fn decay_start(&self) -> u32 {
        self.warmup + self.hold
    }

    /// The rate at `step` (0-based, and GLOBAL: a resumed run passes the step
    /// it is actually on, so it continues the curve instead of warming up a
    /// second time).
    pub fn at(&self, step: u32) -> f32 {
        if step < self.warmup {
            return self.peak * (step + 1) as f32 / self.warmup.max(1) as f32;
        }
        let start = self.decay_start();
        if step < start {
            return self.peak;
        }
        if step >= self.decay_iters {
            return self.floor;
        }
        let ratio = (step - start) as f32 / (self.decay_iters - start).max(1) as f32;
        let coeff = 0.5 * (1.0 + (std::f32::consts::PI * ratio).cos());
        self.floor + coeff * (self.peak - self.floor)
    }
}

impl From<&FitOpts> for LrSchedule {
    fn from(o: &FitOpts) -> LrSchedule {
        LrSchedule { peak: o.lr, floor: o.min_lr, warmup: o.warmup, hold: 0, decay_iters: o.decay_iters }
    }
}

/// Cosine LR schedule with linear warmup (nanogpt's `get_lr`) - [`LrSchedule`]
/// built from a [`FitOpts`], kept as a free function for the callers (and the
/// `gpt2::train::cosine_lr` re-export) that already spell it this way.
pub fn cosine_lr(it: u32, opts: &FitOpts) -> f32 {
    LrSchedule::from(opts).at(it)
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
    let (train, val, batch_cfg, vocab, _itos) = load_dataset_with_itos(dir, opts)?;
    Ok((train, val, batch_cfg, vocab))
}

/// Like [`load_dataset`] but additionally returns the char-tokenizer vocab
/// (`itos`) when the dataset carries one - for callers (`crates/rl`'s
/// `fit_weighted`) that must carry it into their own checkpoints, the way
/// [`fit`] already does via [`Model::save_with_itos`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::type_complexity)]
pub fn load_dataset_with_itos(
    dir: &Path,
    opts: &FitOpts,
) -> std::io::Result<(TokenDataset, TokenDataset, data::loader::BatchConfig, u32, Option<Vec<char>>)> {
    let l = load(dir, opts)?;
    Ok((l.train, l.val, l.batch_cfg, l.vocab, l.itos))
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

/// Build a fresh model from `cfg` (finalized against `dataset_vocab`/
/// `opts.block_size`), or resume from `out`'s existing checkpoint if one is
/// already there - the resume-vs-fresh-init block shared by [`fit`] and
/// `crates/rl`'s `fit_weighted`. On resume the checkpoint's own architecture
/// wins (and must match `opts.block_size`/`dataset_vocab` - weights resume,
/// AdamW moments restart); otherwise a fresh random init runs from `opts.seed`.
#[cfg(not(target_arch = "wasm32"))]
pub fn build_or_resume<M: Model>(cfg: M::Config, opts: &FitOpts, out: Option<&Path>, dataset_vocab: u32) -> M {
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
        assert_eq!(rcfg.vocab(), dataset_vocab, "checkpoint vocab != dataset vocab - wrong dataset for this checkpoint");
        let init = c.by_role("");
        (rcfg, init)
    } else {
        let cfg = cfg.finalize_for_dataset(dataset_vocab, opts.block_size);
        let init = M::init_weights(&cfg, opts.seed);
        (cfg, init)
    };
    M::new(cfg, opts.batch_size, opts.block_size, &init)
}

/// Ordinary causal-LM SFT - [`fit`]'s [`Objective`]. One micro-step draws an
/// unweighted `Batch::Lm` batch from the train split, forwards, and
/// backwards; eval samples the held-out val split the same way, forward-only.
/// This is exactly the step body `fit` used to run inline, unchanged, now
/// behind the [`Objective`] seam so it is not the fourth copy-paste of it.
#[cfg(not(target_arch = "wasm32"))]
struct CausalLm {
    train: TokenDataset,
    val: TokenDataset,
    batch_cfg: BatchConfig,
    itos: Option<Vec<char>>,
}

/// Build the ordinary causal-LM [`Objective`] any [`Model`] can plug into
/// [`fit_with`] - the same objective [`fit`] itself uses, exposed so callers
/// with their own resume/offload/LoRA setup around model construction (e.g.
/// `qwen3::finetune`, `qwen3tts::sft`, `qwen35::finetune`) can still run the
/// one shared loop instead of copy-pasting it a fourth time. Takes exactly
/// the four fields [`load_dataset_with_itos`] returns.
#[cfg(not(target_arch = "wasm32"))]
pub fn causal_lm<M: Model>(
    train: TokenDataset,
    val: TokenDataset,
    batch_cfg: BatchConfig,
    itos: Option<Vec<char>>,
) -> impl Objective<M> {
    CausalLm { train, val, batch_cfg, itos }
}

#[cfg(not(target_arch = "wasm32"))]
impl<M: Model> Objective<M> for CausalLm {
    fn regime(&self) -> &'static str {
        "causal_lm"
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let (x, y) = self.train.get_batch(&self.batch_cfg, rng);
        let targets = targets_to_u32(&y);
        model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
        let loss = model.forward();
        model.backward();
        loss
    }

    fn eval(&mut self, model: &M, rng: &mut Rng, batches: u32) -> Option<f32> {
        let mut total = 0.0;
        for _ in 0..batches.max(1) {
            let (x, y) = self.val.get_batch(&self.batch_cfg, rng);
            let targets = targets_to_u32(&y);
            model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
            total += model.forward();
        }
        Some(total / batches.max(1) as f32)
    }

    fn itos(&self) -> Option<&[char]> {
        self.itos.as_deref()
    }
}

/// The one training/eval/checkpoint loop, generic over any [`Objective`].
/// Owns everything that must NOT vary per objective: cosine-with-warmup LR,
/// grad accumulation and its averaging scale, global-norm clipping, AdamW,
/// wall-clock checkpointing, eval cadence, and the final save. Always calls
/// [`Model::save_with_itos`] (asking `obj` for its [`Objective::itos`]) -
/// never [`Model::save`] - so no objective can silently drop the char vocab
/// the way `rl::fit_weighted` used to.
///
/// The initial (pre-training) loss estimate runs 5 [`Objective::micro_step`]s
/// on a throwaway clone of the rng stream, exactly mirroring the original
/// inline loops' 5-batch train-split sample: it uses the same batches
/// `micro_step`'s own accumulation loop draws (train split, whatever
/// per-position weighting the objective applies), just discarded (via the
/// following [`Model::zero_grads`]) before the first real optimizer step.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_with<M: Model, O: Objective<M>>(mut model: M, mut obj: O, opts: &FitOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    obj.prepare(&mut model);
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);

    let initial = {
        let mut sample_rng = rng.clone();
        let mut total = 0.0;
        for _ in 0..5 {
            total += obj.micro_step(&model, &mut sample_rng);
        }
        total / 5.0
    };
    let mut last_train = initial;
    let mut last_save = std::time::Instant::now();

    for step in 0..opts.steps {
        let lr = cosine_lr(step, opts);
        model.zero_grads();
        let mut step_loss = 0.0;
        for _ in 0..opts.grad_accum.max(1) {
            step_loss += obj.micro_step(&model, &mut rng);
        }
        // average grads over accumulation steps
        let scale = 1.0 / opts.grad_accum.max(1) as f32;
        let clip = (opts.grad_clip > 0.0).then_some(opts.grad_clip);
        model.adamw_step(step + 1, lr, opts.weight_decay, clip, scale);
        model.poll_wait();
        last_train = step_loss / opts.grad_accum.max(1) as f32;

        if opts.eval_interval > 0 && (step + 1) % opts.eval_interval == 0 {
            if let Some(eval_loss) = obj.eval(&model, &mut rng.clone(), opts.eval_batches) {
                println!("step {:>6}  lr {:.2e}  train {:.4}  eval {:.4}", step + 1, lr, last_train, eval_loss);
            }
            for (name, value) in obj.metrics() {
                println!("  {name}: {value:.4}");
            }
        }

        // Wall-clock checkpointing: once the timer has expired, the NEXT completed
        // step saves (atomic temp-rename), reports the save duration, and restarts
        // the timer. A slow big-model step thus never pays a per-eval 2.4 GB write.
        if let Some(p) = out {
            if opts.checkpoint_secs > 0 && last_save.elapsed().as_secs() >= opts.checkpoint_secs {
                let ts = std::time::Instant::now();
                model.save_with_itos(p.to_str().expect("utf-8 path"), obj.itos());
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
        model.save_with_itos(p.to_str().expect("utf-8 path"), obj.itos());
        println!("saved checkpoint -> {} ({:.1} s)", p.display(), ts.elapsed().as_secs_f64());
    }
    Ok((initial, last_train))
}

/// Train any [`Model`] on the token dataset in `dir`, writing the final
/// checkpoint to `out`. `cfg` carries the architecture; its `vocab`/`block_size`
/// are overridden from the dataset and `opts`. Returns `(initial_loss,
/// final_loss)`.
///
/// This is `gpt2::train::train` lifted to `M: Model` - same control flow, same
/// resume/eval/checkpoint semantics, no GPT-specific code. Delegates to
/// [`build_or_resume`] + [`fit_with`] over the [`CausalLm`] objective; the
/// loop body itself lives once, in `fit_with`.
///
/// Native-only: it reads token `.bin` datasets and writes checkpoints, neither
/// of which exists on the wasm32 inference build.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit<M: Model>(dir: &Path, cfg: M::Config, opts: &FitOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    let loaded = load(dir, opts)?;
    let model = build_or_resume::<M>(cfg, opts, out, loaded.vocab);
    let obj = CausalLm { train: loaded.train, val: loaded.val, batch_cfg: loaded.batch_cfg, itos: loaded.itos };
    fit_with(model, obj, opts, out)
}

/// [`fit`], starting from WEIGHTS THE CALLER SUPPLIES.
///
/// [`fit`] has exactly two starting points: an existing checkpoint at `out`,
/// or fresh random initialisation. That is right for a training job that
/// owns its output path and either starts or resumes. It is wrong for a
/// caller fine-tuning a PRETRAINED model into a new file, which is neither:
/// such a caller gets random initialisation and silently trains from
/// scratch, with nothing in the logs to say so but a missing "resuming from"
/// line.
///
/// `init` is the starting point, and the architecture comes from `cfg`, so a
/// fine-tune can be re-shaped (a shorter training window than the base was
/// exported with, say) in a way the resume path deliberately refuses.
/// Tensors absent from `init` fall back to the architecture's own fresh
/// initialisation - which is what a LoRA overlay needs, since the base
/// checkpoint has no `.lora_a`/`.lora_b` in it.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_from<M: Model>(
    dir: &Path,
    cfg: M::Config,
    opts: &FitOpts,
    out: Option<&Path>,
    init: &std::collections::HashMap<String, Vec<f32>>,
) -> std::io::Result<(f32, f32)> {
    let loaded = load(dir, opts)?;
    let cfg = cfg.finalize_for_dataset(loaded.vocab, opts.block_size);
    let mut weights = M::init_weights(&cfg, opts.seed);
    for (name, values) in init {
        // Only what the architecture actually has, and only at the shape it
        // has it: a tensor the config sized differently is the caller
        // handing over a different architecture, and overwriting silently
        // would train a model nobody described.
        match weights.get(name) {
            Some(slot) if slot.len() == values.len() => {
                weights.insert(name.clone(), values.clone());
            }
            Some(slot) => {
                return Err(std::io::Error::other(format!(
                    "fit_from: {name} is {} values in the supplied weights and {} in this config",
                    values.len(),
                    slot.len()
                )))
            }
            None => {}
        }
    }
    let model = M::new(cfg, opts.batch_size, opts.block_size, &weights);
    let obj = CausalLm { train: loaded.train, val: loaded.val, batch_cfg: loaded.batch_cfg, itos: loaded.itos };
    fit_with(model, obj, opts, out)
}

/// Generate `max_new` tokens continuing `prompt` for any token-head [`Model`].
/// Context is cropped to the model's block size. `temperature <= 0` selects
/// greedy argmax; `top_k = 0` disables top-k filtering. A thin wrapper over
/// [`crate::rollout::ModelRollout::sample_n`] (`n = 1`, no EOS) - the
/// always-correct, architecture-agnostic rollout this function's own
/// pre-P10 implementation was lifted into, byte-identical for a fixed
/// `(seed, temperature, top_k)` (see `qwen3`'s
/// `generate_output_is_byte_identical_across_the_rollout_refactor`).
pub fn generate<M: Model>(
    m: &M,
    prompt: &[u32],
    max_new: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut Rng,
) -> Vec<u32> {
    use crate::rollout::{ModelRollout, Rollout, RolloutParams};
    use crate::serve::SampleParams;

    let params = RolloutParams { max_new, sample: SampleParams { temp: temperature, top_k, top_p: 1.0 }, eos: None };
    let mut rollout = ModelRollout::new(m);
    rollout.sample_n(prompt, 1, &params, rng).pop().expect("sample_n(1) returns exactly one completion").tokens
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

    /// [`LrSchedule`] is the schedule [`cosine_lr`] always computed, named as
    /// its own value so a trainer that is not `fit` can hold one. The two must
    /// stay the SAME numbers, bit for bit, at every step - that equality is
    /// what makes this a hoist rather than a second schedule.
    #[test]
    fn the_named_schedule_is_the_one_cosine_lr_computes() {
        let o = FitOpts { lr: 3e-4, min_lr: 3e-5, warmup: 17, decay_iters: 250, ..Default::default() };
        let s = LrSchedule::from(&o);
        for it in 0..400 {
            assert_eq!(s.at(it), cosine_lr(it, &o), "step {it}");
        }
    }

    /// The three properties a caller picks this schedule FOR, stated as
    /// numbers rather than left to the shape of the curve: the ramp starts
    /// below peak and reaches it, the decay is monotone down, and it lands
    /// exactly on the floor - the last being the one that matters for a
    /// short run, where "cosine" that never actually arrives is just a
    /// slightly smaller constant rate.
    #[test]
    fn warmup_ramps_decay_is_monotone_and_the_floor_is_reached() {
        let s = LrSchedule { peak: 1e-4, floor: 1e-5, warmup: 10, hold: 0, decay_iters: 200 };
        assert!(s.at(0) < s.peak, "a warmup must start below peak");
        assert!((s.at(9) - s.peak).abs() < 1e-6 * s.peak, "the ramp must reach peak at the end of warmup");
        for it in 10..200 {
            assert!(s.at(it) >= s.at(it + 1), "decay must be monotone at step {it}");
        }
        assert!((s.at(199) - s.floor).abs() < 2e-7, "the last trained step must be at the floor");
        assert_eq!(s.at(500), s.floor, "past the horizon the floor holds");

        // A run that asked for no schedule gets exactly the rate it asked for,
        // at every step - the previous behaviour, expressible.
        let c = LrSchedule::constant(1e-4);
        assert!((0..500).all(|it| c.at(it) == 1e-4));
    }

    /// `hold` is the constant-then-cooldown shape (Hägele et al.
    /// arXiv:2405.18392): the rate for every step before the cooldown is the
    /// peak and does NOT depend on the step budget, which is the property that
    /// makes a run safe to extend. Only the tail is a function of the total.
    #[test]
    fn a_held_peak_is_budget_independent_until_the_cooldown_starts() {
        let short = LrSchedule { peak: 1e-4, floor: 1e-5, warmup: 0, hold: 160, decay_iters: 200 };
        let long = LrSchedule { peak: 1e-4, floor: 1e-5, warmup: 0, hold: 1600, decay_iters: 2000 };
        assert_eq!(short.decay_start(), 160);
        for step in 0..160 {
            assert_eq!(short.at(step), 1e-4, "the held phase is the peak rate at step {step}");
            assert_eq!(long.at(step), short.at(step), "the held phase does not know the budget");
        }
        // The cosine's own first sample is the peak, so the rate leaves it on
        // the step after the hold ends - not on it.
        assert!((short.at(160) - 1e-4).abs() < 1e-6 * 1e-4);
        assert!(short.at(161) < short.at(160), "the cooldown starts where the hold ends");
        for step in 160..199 {
            assert!(short.at(step) >= short.at(step + 1), "the cooldown must be monotone at {step}");
        }
        assert!(short.at(199) < short.floor + 0.01 * (short.peak - short.floor));
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
