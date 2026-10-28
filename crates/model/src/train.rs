// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One generic training/eval/sample loop over any [`Model`](crate::Model)
//! (ADR §3). [`fit`] is `gpt2::train::train` lifted to `M: Model` - same control
//! flow (cosine-with-warmup LR, grad accumulation with averaging, periodic eval,
//! resumable atomic checkpointing); [`generate`] is `gpt2::sample::generate`
//! lifted to any token-head model.

use data::rng::Rng;

use crate::objective::{Objective, StepVerdict};
use crate::{Model, ModelConfig};
#[cfg(not(target_arch = "wasm32"))]
use crate::Batch;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
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
    /// Evaluate on the held-out split every this many steps (`0`: never). A
    /// run that selects on held-out loss ([`Self::patience`]) also evaluates
    /// at its last step, so the budget's end is a candidate too.
    pub eval_interval: u32,
    /// Batches drawn from the held-out split per evaluation. `0` scores every
    /// example of an example-indexed split exactly once, in order: the same
    /// loss for the same parameters, which a drawn sample is not (a token
    /// stream without example boundaries draws one batch).
    pub eval_batches: u32,
    /// Seed of the initial weights and the batch order. [`Default`] draws a
    /// random one; the run's record carries the value it used.
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
    /// Stop when the HELD-OUT loss has not improved for this many
    /// consecutive evaluations, and keep the checkpoint that achieved the
    /// best one.
    ///
    /// `0` never stops early and saves the LAST model, which is what every
    /// caller got before this existed. Any other value changes what `out`
    /// holds: the point of stopping early is to keep the model from before
    /// it started memorising, and saving the last one afterwards would
    /// throw away the thing the stop was for.
    ///
    /// Needs `eval_interval > 0` and a validation split; without either
    /// there is no held-out loss to watch and this is inert. A patience the
    /// run cannot reach (`u32::MAX`) keeps the best evaluation's parameters
    /// and still runs every step.
    pub patience: u32,
    /// Steps a run that runs out of [`Self::patience`] spends bringing its
    /// rate down to [`Self::min_lr`] before it stops, evaluating as it goes
    /// and still keeping the best evaluation. A schedule planned over
    /// [`Self::decay_iters`] steps never reaches its end when the run stops
    /// at a plateau, and the adapter it keeps would be the one trained at a
    /// high rate; the cooldown is the anneal such a run does not otherwise
    /// get. The most remaining steps of the run's budget are used; `0`
    /// stops where the patience ran out.
    pub cooldown_steps: u32,
    /// AdamW's β1, β2 and ε for every step of the run.
    pub adam: crate::Adam,
    /// Arithmetic of the frozen-weight GEMMs on a device with tensor cores (a model that runs them
    /// through [`crate::ops::Ops`] honours it). The default, `Auto`, takes tensor cores where the
    /// device has them (bf16 for bf16 bases, TF32 for fp32 bases); `Fp32Reference` pins the exact fp32 kernels.
    pub gemm_precision: gpu_core::provider::library::GemmPrecision,
    /// Most floats of trainable parameters a run that selects on held-out loss keeps in host
    /// memory as its best-so-far copy. A larger set is written to disk instead (see
    /// [`FitControl::state`] and the `out` path of [`fit_controlled`]) and is never dropped.
    pub best_hold_floats: usize,
    /// What a FULL fine-tune stores each trainable parameter as (value, gradient, optimizer state, master) and
    /// whether the memory planner may narrow its optimizer state to fit. fp32 everywhere by default; a trainer that
    /// does not run a choice refuses it by name ([`crate::TrainPrecision::resolve`]). Part of the identity a saved
    /// training state is resumed under when it departs from the default.
    pub precision: crate::TrainPrecision,
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
            seed: data::rng::random_seed(),
            checkpoint_secs: 600,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: false,
            // Off: unchanged behaviour for every caller that has not asked
            // to stop early.
            patience: 0,
            cooldown_steps: 0,
            adam: crate::Adam::default(),
            gemm_precision: gpu_core::provider::library::GemmPrecision::Auto,
            best_hold_floats: BEST_IN_MEMORY_FLOATS,
            precision: crate::TrainPrecision::default(),
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
/// **Why `hold` exists, rather than cosine from step 0.** The constant-then-cooldown study
/// (arXiv:2405.18392) measures constant-rate training followed by a short
/// COOLDOWN against a full cosine and find the two scale "predictably and
/// reliably" alike, with the cooldown's benefit saturating at roughly 20% of  perf-number: published figure, arXiv:2405.18392
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

/// Where each example of `split` starts: its `<split>.ex.bin` index when the
/// dataset has one, checked against the stream it indexes. A dataset written
/// before examples were indexed closed each one with
/// [`data::chat::LEGACY_EXAMPLE_SEPARATOR`] in band; its examples are found
/// there when the stream holds that id and the vocabulary covers it. `None`:
/// a token stream without example boundaries, sampled in windows.
#[cfg(not(target_arch = "wasm32"))]
fn example_starts(dir: &Path, split: &str, tok: &[u32], vocab: u32) -> std::io::Result<Option<Vec<usize>>> {
    let index = dir.join(format!("{split}.ex.bin"));
    if index.exists() {
        let starts: Vec<usize> = binio::read_u64_bin(&index)?.into_iter().map(|s| s as usize).collect();
        let ascending = starts.first().is_none_or(|&s| s == 0) && starts.windows(2).all(|w| w[0] < w[1]) && starts.last().is_none_or(|&s| s < tok.len());
        if !ascending || (starts.is_empty() && !tok.is_empty()) {
            return Err(std::io::Error::other(format!("{}: not an index of {} tokens (starts must ascend from 0 within the stream)", index.display(), tok.len())));
        }
        return Ok(Some(starts));
    }
    let sep = data::chat::LEGACY_EXAMPLE_SEPARATOR;
    Ok((sep < vocab && tok.contains(&sep)).then(|| TokenDataset::separator_starts(tok, sep)))
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
    // supervises only the assistant span at the TOKEN level, and
    // `train.ex.bin` indexes where each example starts. When present the mask
    // takes precedence over the char-boundary `mask_before_token`.
    let train_mask = binio::read_mask_bin(&dir.join("train.mask.bin")).ok();
    let val_mask = binio::read_mask_bin(&dir.join("val.mask.bin")).ok();
    let has_token_mask = train_mask.is_some();
    let train_starts = example_starts(dir, "train", &train_tok, vocab)?;
    let val_starts = example_starts(dir, "val", &val_tok, vocab)?;

    let batch_cfg = BatchConfig {
        batch_size: opts.batch_size as usize,
        block_size: opts.block_size as usize,
        mask_before_token: if has_token_mask { None } else { mask_id },
        mask_per_line: opts.mask_per_line,
        // A token-masked chat dataset is sampled one example per row instead
        // (see below), which supersedes aligning a long window to an example
        // start: the window used to START at an example and then run on
        // through two dozen more.
        align_to_lines: opts.align_to_lines && !has_token_mask,
        newline_token: if has_token_mask { None } else { newline_id },
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

    // A chat/tool-call dataset carries a token mask AND an example index, so
    // its examples can be told apart and each given a row of its own. Packing them into a shared window instead lets one
    // example attend to its neighbours' answers - which, for short
    // instruction-tuning examples, is most of the row - so the model can
    // drive both training and held-out loss down by copying rather than
    // learning, and neither number then says anything about how it will do on
    // a single question at serving time.
    let mk = |label: &str, tok: Vec<u32>, mask: Option<Vec<bool>>, starts: Option<Vec<usize>>| -> std::io::Result<TokenDataset> {
        let (tok, mask) = match (mask, starts) {
            // One example per row needs no window at all, so it is the one
            // path that does not care how long the split is. Padding is id 0:
            // any id the vocabulary holds, since a pad position is never a
            // target and nothing before it attends to it.
            (Some(m), Some(starts)) if m.len() == tok.len() && has_token_mask && !tok.is_empty() => {
                match TokenDataset::new_examples(tok, m, &starts, 0, &batch_cfg) {
                    Ok(d) => return Ok(d),
                    // Examples too long to be rows of their own: this is
                    // document-scale data, where a window IS the unit of
                    // training and packing is not the problem it is for short
                    // instruction-tuning examples. Fall back, but say so -
                    // silently windowing a dataset that should have been one
                    // example per row is the defect this branch exists to
                    // make visible.
                    Err(e) => {
                        eprintln!("{label}: {e}; sampling windows across examples instead");
                        let (tok, m) = e.returned();
                        (tok, Some(m))
                    }
                }
            }
            (other, _) => (tok, other),
        };
        // Every remaining path draws windows, which needs a split longer than
        // one of them.
        too_short(label, &tok)?;
        Ok(match mask {
            Some(m) if m.len() == tok.len() => TokenDataset::new_with_mask(tok, m, &batch_cfg),
            _ => TokenDataset::new(tok, &batch_cfg),
        })
    };
    Ok(Loaded {
        train: mk("train", train_tok, train_mask, train_starts)?,
        val: mk("validation", val_tok, val_mask, val_starts)?,
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
        eprintln!("resuming from existing checkpoint {}", p.display());
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
    build_for::<M>(cfg, opts, &init).unwrap_or_else(|e| panic!("{e}"))
}

/// Build `M` sized for `opts`: with its layers streaming from the start when the run asks for f32 layers through a
/// window (`weights=f32`, [`Model::new_resident`]), so a model larger than the device is built for training at all;
/// a plain build otherwise, the run's other precision applied by the fit loop.
pub fn build_for<M: Model>(cfg: M::Config, opts: &FitOpts, init: &std::collections::HashMap<String, Vec<f32>>) -> std::io::Result<M> {
    match opts.precision.resident_config(Some(opts.seed)).map_err(std::io::Error::other)? {
        Some(resident) if resident.value == crate::resident_ft::ResidentValue::F32 => {
            M::new_resident(cfg, opts.batch_size, opts.block_size, init, &resident).map_err(std::io::Error::other)
        }
        _ => Ok(M::new(cfg, opts.batch_size, opts.block_size, init)),
    }
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
    /// The order the training examples are drawn in, once the run has
    /// begun: epochs of every example once. `None` before [`Objective::begin`]
    /// and for a plain token stream, which is sampled in windows.
    order: Option<EpochOrder>,
    /// This rank's part of the deterministic training stream, `(rank, world)`:
    /// record `n` belongs to rank `n % world` and the rank drops the rest.
    /// `None` trains on the whole stream.
    shard: Option<(usize, usize)>,
    /// How many records of the whole stream have been consumed, used and
    /// dropped alike. Every rank advances it in the same steps, so record `n`
    /// is the same record on every rank. A plain token stream has no record
    /// boundaries, so there one drawn batch is one record.
    drawn_records: u64,
    /// What the micro-steps since the last [`Objective::step_work`] measured.
    work: Option<StepWork>,
}

/// One drawn training batch: its rows, their targets, and how many of the rows'
/// tokens are real record tokens. A row is laid out in a `block_size` row and
/// holds its example's tokens followed by padding, which is not useful.
#[cfg(not(target_arch = "wasm32"))]
struct Drawn {
    x: Vec<u32>,
    y: Vec<i32>,
    useful_tokens: u64,
}

/// The token half of a micro-step's [`StepWork`]: what the drawn batch holds.
#[cfg(not(target_arch = "wasm32"))]
fn work_of(drawn: &Drawn, targets: &[u32]) -> StepWork {
    StepWork { useful_tokens: drawn.useful_tokens, supervised_positions: targets.iter().filter(|&&t| t != IGNORE).count() as u64, ..StepWork::default() }
}

/// Which example each draw takes: the examples shuffled once per epoch, every
/// one drawn once before any is drawn again, so a pass over the data is a
/// pass and not a sample of it with some examples missed and others repeated.
/// The order of an epoch is a function of the seed and the epoch alone, and a
/// draw's place in it of the count of draws before it, so a run resumed at
/// draw `n` continues the order a straight run follows.
#[cfg(not(target_arch = "wasm32"))]
struct EpochOrder {
    seed: u64,
    examples: usize,
    drawn: u64,
    /// The epoch `permutation` is the order of.
    epoch: u64,
    permutation: Vec<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
/// The order `examples` are trained in during `epoch`: a fresh shuffle per
/// epoch, seeded by the run's seed and the epoch number, so a resumed run
/// continues the order a straight run would have followed.
pub fn epoch_order(seed: u64, examples: usize, epoch: u64) -> Vec<usize> {
    EpochOrder::shuffled(seed, examples, epoch)
}

impl EpochOrder {
    fn new(seed: u64, examples: usize, drawn: u64) -> Self {
        assert!(examples > 0, "an order over no examples");
        let epoch = drawn / examples as u64;
        EpochOrder { seed, examples, drawn, epoch, permutation: Self::shuffled(seed, examples, epoch) }
    }

    /// unbiased in-place over the examples, seeded by the run's seed and the epoch.
    fn shuffled(seed: u64, examples: usize, epoch: u64) -> Vec<usize> {
        let mut rng = Rng::new(seed ^ epoch.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut order: Vec<usize> = (0..examples).collect();
        for i in (1..examples).rev() {
            order.swap(i, rng.gen_range_inclusive(0, i as i64) as usize);
        }
        order
    }

    fn next(&mut self) -> usize {
        let epoch = self.drawn / self.examples as u64;
        if epoch != self.epoch {
            self.epoch = epoch;
            self.permutation = Self::shuffled(self.seed, self.examples, epoch);
        }
        let pick = self.permutation[(self.drawn % self.examples as u64) as usize];
        self.drawn += 1;
        pick
    }
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
    CausalLm::new(train, val, batch_cfg, itos)
}

#[cfg(not(target_arch = "wasm32"))]
impl<M: Model> Objective<M> for CausalLm {
    fn regime(&self) -> &'static str {
        "causal_lm"
    }

    fn begin(&mut self, seed: u64, draws_done: u64) {
        self.order = self.train.example_count().filter(|&n| n > 0).map(|n| EpochOrder::new(seed, n, draws_done));
    }

    /// Shard the training-record stream. The stream itself is fixed by the
    /// run's seed: [`EpochOrder`] is a pure function of the seed and the draw
    /// count, and a token stream advances from the run's seeded generator, so
    /// record `n` is the same record on every rank. Rank `r` trains on record
    /// `n` exactly when `n % world == r`, which splits the stream into `world`
    /// disjoint shards whose interleaving is the unsharded stream. Every rank
    /// still consumes every record (and drops its peers'), so their generators
    /// and epoch orders stay on the same stream.
    ///
    /// On a dataset with example boundaries (the prepared JSONL a fine-tune
    /// trains on) a record is one example. A plain token stream has no record
    /// boundaries, so there one drawn batch is one record.
    ///
    /// Held-out records are deliberately NOT sharded: [`Self::eval`] scores the
    /// whole validation split, and a distributed run has rank 0 score it once
    /// and broadcast the result, so every replica selects on the same held-out
    /// loss.
    fn shard(&mut self, rank: usize, world: usize) -> std::io::Result<()> {
        if world == 0 || rank >= world {
            return Err(std::io::Error::other(format!("rank {rank} is not one of a world of {world}")));
        }
        self.shard = Some((rank, world));
        Ok(())
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let clock = std::time::Instant::now();
        let drawn = self.draw(rng);
        let targets = targets_to_u32(&drawn.y);
        model.set_batch(Batch::Lm { tokens: &drawn.x, targets: &targets });
        let mut work = work_of(&drawn, &targets);
        work.prepare_secs = clock.elapsed().as_secs_f64();
        let pass = std::time::Instant::now();
        let loss = model.forward();
        work.forward_secs = pass.elapsed().as_secs_f64();
        work.computed_tokens = model.computed_tokens().unwrap_or(drawn.x.len() as u64);
        let pass = std::time::Instant::now();
        model.backward();
        work.backward_secs = pass.elapsed().as_secs_f64();
        self.work = Some(work);
        loss
    }

    /// The `k` batches are drawn up front, in the order one at a time would
    /// draw them, so a model that overlaps them trains on the same data. A
    /// model that takes a loss denominator gets the step's mean count of
    /// supervised positions, so the step's gradient is the mean over its
    /// supervised tokens - each token weighs the same whichever record it is
    /// in - and the loss it reports is that per-token mean, the quantity a
    /// held-out evaluation measures.
    fn micro_steps(&mut self, model: &M, rng: &mut Rng, k: u32) -> f32 {
        // Data preparation is the draws and the uploads; the forward and
        // backward passes are measured around each call.
        let clock = std::time::Instant::now();
        let drawn: Vec<Drawn> = (0..k).map(|_| self.draw(rng)).collect();
        let targets: Vec<Vec<u32>> = drawn.iter().map(|d| targets_to_u32(&d.y)).collect();
        let mut work = StepWork { useful_tokens: drawn.iter().map(|d| d.useful_tokens).sum(), supervised_positions: targets.iter().flatten().filter(|&&t| t != IGNORE).count() as u64, ..StepWork::default() };
        if k > 1 {
            let batches: Vec<Batch> = drawn.iter().zip(&targets).map(|(d, y)| Batch::Lm { tokens: &d.x, targets: y }).collect();
            let drew = clock.elapsed().as_secs_f64();
            let pass = std::time::Instant::now();
            if let Some(total) = model.accumulate_overlapped(&batches) {
                // An overlapped pass runs the micro-steps' forwards and
                // backwards together, so its whole time is the backward's.
                work.prepare_secs = drew;
                work.backward_secs = pass.elapsed().as_secs_f64();
                work.computed_tokens = drawn.iter().map(|d| d.x.len() as u64).sum();
                self.work = Some(work);
                return total;
            }
        }
        work.prepare_secs = clock.elapsed().as_secs_f64();
        let mean_supervised = work.supervised_positions as f32 / k.max(1) as f32;
        let mut total = 0.0;
        for (drawn, targets) in drawn.iter().zip(&targets) {
            let pass = std::time::Instant::now();
            model.set_batch(Batch::Lm { tokens: &drawn.x, targets });
            if mean_supervised > 0.0 {
                if let Some(objectives) = model.token_objectives() {
                    objectives.set_loss_denominator(mean_supervised);
                }
            }
            work.prepare_secs += pass.elapsed().as_secs_f64();
            let pass = std::time::Instant::now();
            total += model.forward();
            work.forward_secs += pass.elapsed().as_secs_f64();
            work.computed_tokens += model.computed_tokens().unwrap_or(drawn.x.len() as u64);
            let pass = std::time::Instant::now();
            model.backward();
            work.backward_secs += pass.elapsed().as_secs_f64();
        }
        self.work = Some(work);
        total
    }

    fn step_work(&mut self) -> Option<StepWork> {
        self.work.take()
    }

    fn loss_probe(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let (x, y) = self.train.get_batch(&self.batch_cfg, rng);
        let targets = targets_to_u32(&y);
        model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
        model.forward()
    }

    /// Held-out records are intentionally not sharded: the whole validation
    /// split is scored as one held-out loss. A distributed run has rank 0 do
    /// the scoring and shares the value, so early stopping decides the same
    /// thing on every replica.
    fn eval(&mut self, model: &M, rng: &mut Rng, batches: u32) -> Option<f32> {
        if batches == 0 {
            if let Some(count) = self.val.example_count() {
                return self.eval_every_example(model, count);
            }
        }
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

#[cfg(not(target_arch = "wasm32"))]
impl CausalLm {
    fn new(train: TokenDataset, val: TokenDataset, batch_cfg: BatchConfig, itos: Option<Vec<char>>) -> CausalLm {
        CausalLm { train, val, batch_cfg, itos, order: None, shard: None, drawn_records: 0, work: None }
    }

    /// This rank's next training batch: `batch_size` records of the stream that
    /// belong to its shard, or the next whole batch when the run is not
    /// sharded.
    fn draw(&mut self, rng: &mut Rng) -> Drawn {
        let Some((rank, world)) = self.shard else {
            return self.draw_next(rng);
        };
        if let Some(order) = self.order.as_mut() {
            // Record granularity: collect `batch_size` records whose stream
            // index belongs to this rank. Every rank consumes the same
            // `batch_size * world` records, so their orders advance together.
            let mut picks: Vec<Option<usize>> = Vec::with_capacity(self.batch_cfg.batch_size);
            while picks.len() < self.batch_cfg.batch_size {
                let index = self.drawn_records;
                self.drawn_records += 1;
                let pick = order.next();
                if index % world as u64 == rank as u64 {
                    picks.push(Some(pick));
                }
            }
            return self.draw_examples(&picks).expect("a stream with an epoch order has example rows");
        }
        // A plain token stream has no record boundaries: its drawn batch is the
        // record, so the rank keeps one batch out of every `world`.
        loop {
            let index = self.drawn_records;
            self.drawn_records += 1;
            let batch = self.draw_next(rng);
            if index % world as u64 == rank as u64 {
                return batch;
            }
        }
    }

    /// The next batch of the whole (unsharded) stream: the next examples of
    /// the epoch order, else (before the run has begun, or for a stream of
    /// tokens) a random draw.
    fn draw_next(&mut self, rng: &mut Rng) -> Drawn {
        if let Some(order) = self.order.as_mut() {
            let picks: Vec<Option<usize>> = (0..self.batch_cfg.batch_size).map(|_| Some(order.next())).collect();
            if let Some(drawn) = self.draw_examples(&picks) {
                return drawn;
            }
        }
        let (x, y) = self.train.get_batch(&self.batch_cfg, rng);
        // A window drawn from a token stream is all real tokens: the stream has
        // no record boundary to pad at.
        Drawn { useful_tokens: x.len() as u64, x, y }
    }

    /// The example batch `picks` names, with the number of its rows' own tokens.
    /// `None` for a plain token stream, which has no examples.
    fn draw_examples(&self, picks: &[Option<usize>]) -> Option<Drawn> {
        let (x, y) = self.train.example_batch(&self.batch_cfg, picks)?;
        let useful_tokens = self.train.example_lengths(picks)?.iter().sum::<usize>() as u64;
        Some(Drawn { x, y, useful_tokens })
    }

    /// The mean per-position loss over every supervised position of the
    /// validation split's `count` examples, each a row once, in order:
    /// [`Model::forward`] returns a batch's mean over its supervised
    /// positions, so each batch is weighted back by them. `None` when no
    /// position is supervised.
    fn eval_every_example<M: Model>(&self, model: &M, count: usize) -> Option<f32> {
        let rows = self.batch_cfg.batch_size.max(1);
        let (mut sum, mut positions) = (0.0f64, 0usize);
        let mut first = 0;
        while first < count {
            let (x, y) = self.val.example_rows(&self.batch_cfg, first)?;
            let supervised = y.iter().filter(|&&t| t != data::loader::IGNORE).count();
            if supervised > 0 {
                let targets = targets_to_u32(&y);
                model.set_batch(Batch::Lm { tokens: &x, targets: &targets });
                sum += f64::from(model.forward()) * supervised as f64;
                positions += supervised;
            }
            first += rows;
        }
        (positions > 0).then(|| (sum / positions as f64) as f32)
    }
}

/// How much of the trainable set the best-checkpoint keeper will hold in host
/// memory by default before writing it to disk instead.
///
/// A full finetune's trainable set IS the model, and a second host-side copy
/// of it is not free. A LoRA run's is the adapter alone - about 5M floats
/// where the checkpoint it would otherwise rewrite is 600M - so the case that
/// pays for this fits with room to spare. A 27B model's rank-16 adapter is not
/// that case (117M floats): past the budget the best copy goes to disk
/// (see `spill_path`), never away.
const BEST_IN_MEMORY_FLOATS: usize = 64 << 20;

/// The trainable parameters, if they fit `budget` floats.
///
/// `None` means "too large to hold": the caller falls back to writing the
/// checkpoint, which is what this replaces. Reading stops as soon as the
/// budget is passed, so an oversized model is not fully copied just to
/// discover it does not fit.
#[cfg(not(target_arch = "wasm32"))]
fn hold_trainable<M: Model>(model: &M, budget: usize) -> Option<HashMap<String, Vec<f32>>> {
    let mut held = HashMap::new();
    let mut floats = 0usize;
    for name in model.param_names() {
        let w = model.read_weight(&name);
        floats += w.len();
        if floats > budget {
            return None;
        }
        held.insert(name, w);
    }
    Some(held)
}

/// The file the best parameters of a run are spilled to when they are too large to hold in
/// memory and the run has no checkpoint path: beside its training state, so it survives the
/// process exactly as the state does.
#[cfg(not(target_arch = "wasm32"))]
fn spill_path(state: &Path) -> std::path::PathBuf {
    state.with_extension("best")
}

/// Write the trainable parameters to `path`, atomically (a crash leaves the previous best).
#[cfg(not(target_arch = "wasm32"))]
fn write_spill<M: Model>(model: &M, path: &Path) -> std::io::Result<()> {
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = model
        .param_names()
        .into_iter()
        .map(|name| {
            let w = model.read_weight(&name);
            (name, vec![w.len() as u64], w)
        })
        .collect();
    let tmp = path.with_extension("best.tmp");
    let tmp_str = tmp.to_str().ok_or_else(|| std::io::Error::other(format!("{}: not a UTF-8 path", tmp.display())))?;
    checkpoint::st::save_safetensors(tmp_str, &tensors, &serde_json::json!({"format": "brain-best-parameters-v1"}), None)?;
    std::fs::rename(&tmp, path)
}

/// The parameters [`write_spill`] wrote.
#[cfg(not(target_arch = "wasm32"))]
fn read_spill(path: &Path) -> std::io::Result<HashMap<String, Vec<f32>>> {
    let path_str = path.to_str().ok_or_else(|| std::io::Error::other(format!("{}: not a UTF-8 path", path.display())))?;
    Ok(checkpoint::st::load_safetensors(path_str)?.tensors)
}

/// What to do after one held-out evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Watch {
    /// The best held-out loss so far. Keep this checkpoint.
    Improved,
    /// Worse than the best, but not for long enough to stop.
    Wait,
    /// It has not improved for `patience` evaluations. Stop, and keep the
    /// checkpoint from the best one.
    Stop,
}

/// Watches the held-out loss and decides when training has stopped learning
/// the task and started learning the rows.
///
/// The signal is the DIVERGENCE: training loss falling while held-out loss
/// rises. Watching the training loss alone cannot see it - that curve looks
/// healthy all the way into memorisation, and on a small dataset it reaches
/// ~0 while the model gets worse at everything it was not shown.
///
/// Separate from [`fit_with`] so the policy can be tested against a sequence
/// of losses rather than against a GPU.
#[derive(Clone, Debug)]
pub struct EarlyStop {
    patience: u32,
    best: f32,
    since_improved: u32,
}

impl EarlyStop {
    /// `patience` evaluations without improvement before stopping. `0`
    /// disarms it: nothing is watched and nothing stops.
    pub fn new(patience: u32) -> EarlyStop {
        EarlyStop { patience, best: f32::INFINITY, since_improved: 0 }
    }

    /// [`Self::new`] as it stood after `since_improved` evaluations past a
    /// best of `best`: a watch continued from a saved state.
    pub fn resumed(patience: u32, best: f32, since_improved: u32) -> EarlyStop {
        EarlyStop { best, since_improved, ..EarlyStop::new(patience) }
    }

    pub fn armed(&self) -> bool {
        self.patience > 0
    }

    /// The best held-out loss seen. `f32::INFINITY` before the first one.
    pub fn best(&self) -> f32 {
        self.best
    }

    /// Evaluations since the best one.
    pub fn since_improved(&self) -> u32 {
        self.since_improved
    }

    /// Record one held-out loss and say what to do about it.
    ///
    /// A NaN is treated as no improvement rather than as a new best: a
    /// diverged step must not be allowed to claim the checkpoint.
    pub fn observe(&mut self, eval: f32) -> Watch {
        if !self.armed() {
            return Watch::Wait;
        }
        if eval < self.best {
            self.best = eval;
            self.since_improved = 0;
            return Watch::Improved;
        }
        self.since_improved += 1;
        if self.since_improved >= self.patience {
            Watch::Stop
        } else {
            Watch::Wait
        }
    }
}

/// One periodic held-out evaluation of a [`fit_controlled`] run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluation {
    /// Steps completed when it was taken.
    pub step: u32,
    /// The mean training loss of the steps since the previous evaluation
    /// (since the start, for the first): the training side of the curve,
    /// measured over the same stretch as the held-out side.
    pub train_loss: f32,
    /// The held-out loss [`Objective::eval`] measured.
    pub eval_loss: f32,
}

/// The evaluation whose parameters a run that selects on held-out loss
/// kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeptBest {
    pub step: u32,
    pub eval_loss: f32,
}

/// What a run that selects on held-out loss carries between evaluations:
/// the watch, the best parameters, and the curve so far. Saved with the
/// exact-resume state, so a run continued from one resumes its selection
/// where it stood.
struct Selection {
    watch: EarlyStop,
    /// The best parameters, held rather than written (see `hold_trainable`).
    held: Option<HashMap<String, Vec<f32>>>,
    /// The best parameters are in the spill file beside the training state (see `spill_path`):
    /// a set too large to hold, and with no checkpoint path to write it to.
    spilled: bool,
    best_step: Option<u32>,
    evaluations: Vec<Evaluation>,
    /// The training losses since the last evaluation.
    interval_sum: f32,
    interval_count: u32,
    /// The cooldown the run is in, once its patience has run out.
    cooldown: Option<Cooldown>,
}

/// A cooldown in progress: the rate falls linearly from where the run stood
/// to its floor over `steps`, the first of which is `first_step`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cooldown {
    first_step: u32,
    steps: u32,
    from_lr: f32,
}

impl Cooldown {
    /// The rate at `step` (0-based): the last step of the cooldown is at the floor.
    fn lr(&self, step: u32, floor: f32) -> f32 {
        let done = (step + 1 - self.first_step).min(self.steps) as f32 / self.steps as f32;
        self.from_lr + (floor - self.from_lr) * done
    }

    /// Whether `step` is the last of the cooldown.
    fn ends_at(&self, step: u32) -> bool {
        step + 1 >= self.first_step + self.steps
    }
}

impl Selection {
    fn new(opts: &FitOpts) -> Selection {
        Selection { watch: EarlyStop::new(opts.patience), held: None, spilled: false, best_step: None, evaluations: Vec::new(), interval_sum: 0.0, interval_count: 0, cooldown: None }
    }

    /// Whether the run keeps its best evaluation's parameters.
    fn selecting(&self) -> bool {
        self.watch.armed()
    }

    fn step_trained(&mut self, loss: f32) {
        self.interval_sum += loss;
        self.interval_count += 1;
    }

    /// Close the interval since the last evaluation with its mean training loss.
    fn take_interval(&mut self) -> f32 {
        let mean = if self.interval_count == 0 { f32::NAN } else { self.interval_sum / self.interval_count as f32 };
        self.interval_sum = 0.0;
        self.interval_count = 0;
        mean
    }

    /// The selection as the state file records it; floats as their bits.
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "best_bits": self.watch.best().to_bits(),
            "since_improved": self.watch.since_improved(),
            "best_step": self.best_step,
            "held": self.held.is_some(),
            "spilled": self.spilled,
            "interval_sum_bits": self.interval_sum.to_bits(),
            "interval_count": self.interval_count,
            "cooldown": self.cooldown.map(|c| serde_json::json!([c.first_step, c.steps, c.from_lr.to_bits()])),
            "evaluations": self.evaluations.iter().map(|e| serde_json::json!([e.step, e.train_loss.to_bits(), e.eval_loss.to_bits()])).collect::<Vec<_>>(),
        })
    }

    /// The selection a state file recorded, its held parameters read from
    /// `tensors` by their `best:` names. `None` when the record is not one
    /// [`Self::to_json`] wrote.
    fn from_json(opts: &FitOpts, value: &serde_json::Value, tensors: &HashMap<String, Vec<f32>>) -> Option<Selection> {
        let bits = |v: &serde_json::Value| v.as_u64().and_then(|b| u32::try_from(b).ok()).map(f32::from_bits);
        let count = |v: &serde_json::Value| v.as_u64().and_then(|b| u32::try_from(b).ok());
        let evaluations = value["evaluations"]
            .as_array()?
            .iter()
            .map(|e| Some(Evaluation { step: count(&e[0])?, train_loss: bits(&e[1])?, eval_loss: bits(&e[2])? }))
            .collect::<Option<Vec<_>>>()?;
        let held = value["held"].as_bool()?.then(|| {
            tensors.iter().filter_map(|(name, w)| name.strip_prefix("best:").map(|n| (n.to_string(), w.clone()))).collect::<HashMap<_, _>>()
        });
        Some(Selection {
            watch: EarlyStop::resumed(opts.patience, bits(&value["best_bits"])?, count(&value["since_improved"])?),
            held,
            spilled: value["spilled"].as_bool().unwrap_or(false),
            best_step: match &value["best_step"] {
                serde_json::Value::Null => None,
                step => Some(count(step)?),
            },
            evaluations,
            interval_sum: bits(&value["interval_sum_bits"])?,
            interval_count: count(&value["interval_count"])?,
            cooldown: match &value["cooldown"] {
                serde_json::Value::Null => None,
                c => Some(Cooldown { first_step: count(&c[0])?, steps: count(&c[1])?, from_lr: bits(&c[2])? }),
            },
        })
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
///
/// [`fit_controlled`] with no caller in the loop: every step runs, and the
/// run starts from whatever `model` holds. Returns `(initial_loss,
/// final_loss)`.
pub fn fit_with<M: Model, O: Objective<M>>(model: M, obj: O, opts: &FitOpts, out: Option<&Path>) -> std::io::Result<(f32, f32)> {
    let (report, _) = fit_controlled(model, obj, opts, out, FitControl::default())?;
    Ok((report.initial_loss, report.final_loss.unwrap_or(report.initial_loss)))
}

/// What one completed optimizer step did, measured: the seconds each phase of
/// the step took and the tokens it accounted for.
///
/// The four phases partition the step's wall time - the objective measures the
/// batches it prepares and the forward and backward passes it runs, and
/// [`fit_controlled`] measures the optimizer, gradients reduction included. A
/// phase an objective does not measure is `0`; the periodic evaluation, when a
/// step is due one, is not part of any phase. The loss estimate a run takes
/// before its first step is not a step and is not counted either.
///
/// `useful_tokens` against `computed_tokens` is what makes padding
/// measurable: a record padded up to a longer row's length is paid for in the
/// second and not in the first, and [`Self::padded_tokens`] is the
/// difference.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepWork {
    /// Seconds drawing the step's batches from the dataset and uploading them.
    pub prepare_secs: f64,
    /// Seconds in the step's forward passes.
    pub forward_secs: f64,
    /// Seconds in the step's backward passes.
    pub backward_secs: f64,
    /// Seconds in the optimizer: zeroing gradients, reducing them across
    /// ranks, AdamW, and waiting for the device to finish the step.
    pub optimizer_secs: f64,
    /// Tokens of the records this step trained on, row padding excluded.
    pub useful_tokens: u64,
    /// Tokens the model actually computed: the run length of every row the
    /// step's forward saw ([`Model::computed_tokens`]), so row padding and
    /// the rounding to whole GDN chunks appear here.
    pub computed_tokens: u64,
    /// Positions carrying a supervised target (the loss denominator).
    pub supervised_positions: u64,
    /// The device memory after the step, with the peak of live bytes over the step: read from the model's device
    /// ([`Model::optimizer_parts`]) where its backend accounts for its allocations.
    pub device_memory: Option<gpu_core::MemoryUsage>,
}

impl StepWork {
    /// Tokens computed that carried no record token: the padded work the step
    /// paid for.
    pub fn padded_tokens(&self) -> u64 {
        self.computed_tokens.saturating_sub(self.useful_tokens)
    }

    /// The step's measured wall time: its four phases.
    pub fn step_secs(&self) -> f64 {
        self.prepare_secs + self.forward_secs + self.backward_secs + self.optimizer_secs
    }

    /// Record tokens per second of step time.
    pub fn useful_tokens_per_s(&self) -> f64 {
        rate(self.useful_tokens, self.step_secs())
    }

    /// Tokens computed per second of step time.
    pub fn computed_tokens_per_s(&self) -> f64 {
        rate(self.computed_tokens, self.step_secs())
    }
}

fn rate(tokens: u64, secs: f64) -> f64 {
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}

/// One completed optimizer step, as [`fit_controlled`] hands it to
/// [`FitControl::on_step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    /// Steps completed so far in the whole run, this one included (1-based,
    /// and global: a resumed run continues the count).
    pub step: u32,
    /// The run's total, [`FitOpts::steps`].
    pub steps: u32,
    /// This step's training loss, averaged over its accumulation micro-steps.
    pub loss: f32,
    /// The learning rate this step ran at.
    pub lr: f32,
    /// The global gradient norm the step's clip measured, when the model reports it
    /// ([`Model::last_grad_norm`]).
    pub grad_norm: Option<f32>,
    /// What the step measured: its phases' seconds, its token accounting and its device memory.
    pub work: StepWork,
}

/// What a caller controls about a [`fit_controlled`] run beyond [`FitOpts`].
#[derive(Default)]
pub struct FitControl<'a> {
    /// Called after every optimizer step. Returning `false` stops the run at
    /// that step boundary - after writing the resume state when [`Self::state`]
    /// is set, so the run can be continued exactly.
    pub on_step: Option<&'a mut dyn FnMut(&StepReport) -> bool>,
    /// Exact resume: the file this run's training state is written to, and
    /// read back from when it already exists. The state is the optimised
    /// weights, their AdamW moments, the completed step count (which fixes the
    /// learning-rate schedule's position and AdamW's bias correction) and the
    /// batch generator's position - everything a step depends on besides the
    /// frozen weights and the data, so a run continued from it computes the
    /// same updates an uninterrupted run would. The model must expose its
    /// optimizer state ([`Model::optimized_params`]).
    pub state: Option<&'a Path>,
    /// Also write [`Self::state`] after every this-many steps (`0`: only when
    /// [`Self::on_step`] stops the run), so a crash loses at most this many.
    pub state_every: u32,
    /// What else must be the same for a saved state to be continued: the
    /// caller's description of its data and starting point (digests, ids).
    /// Compared as JSON after a round trip through the state file, so a
    /// float belongs in it as its bits.
    /// A state whose identity or [`FitOpts`] differ is refused, never
    /// continued into a different run.
    pub identity: serde_json::Value,
    /// Write the adapter every held-out evaluation measured, so the one a
    /// run exports is not the only one a caller can choose from. Needs
    /// [`FitOpts::eval_interval`]; a model with no adapter parameters is
    /// refused.
    pub snapshots: Option<AdapterSnapshots<'a>>,
    /// A distributed run: every step's gradients are averaged over this
    /// group, and each rank trains on its own shard of the training-record
    /// stream ([`Objective::shard`]), derived from the group's rank and world
    /// size. `None` is a single-process run.
    ///
    /// This is a request, never a hint: an objective that cannot shard its
    /// stream fails with its own error, and a model that cannot write reduced
    /// gradients fails with [`Model::write_grad`]'s. There is no silent
    /// fallback to one process.
    pub grad_sync: Option<GradSync<'a>>,
}

/// Where and as what [`FitControl::snapshots`] are written: one adapter file
/// per evaluation, `step-<N>.safetensors` in [`Self::dir`], N the number of
/// steps trained when it was taken.
pub struct AdapterSnapshots<'a> {
    /// The directory the files go to; it exists.
    pub dir: &'a Path,
    /// The card id each file is written under.
    pub card_id: &'a str,
    /// The base checkpoint each adapter is an adapter of.
    pub base_id: &'a str,
    /// The architecture family tag of the card (`"qwen"`).
    pub family: &'a str,
    /// The adapter's configuration (rank, alpha, projections, per-target
    /// overrides, hyper-parameters): whoever builds the model fills it in.
    pub lora: crate::adapter::device::LoraCfg,
}

impl AdapterSnapshots<'_> {
    /// The file the adapter after `step` steps is written to.
    pub fn path(&self, step: u32) -> std::path::PathBuf {
        self.dir.join(format!("step-{step}.safetensors"))
    }
}

/// How a [`fit_controlled`] run went.
#[derive(Clone, Debug, PartialEq)]
pub struct FitReport {
    /// The loss estimate taken before this call's first step.
    pub initial_loss: f32,
    /// The last step's training loss; `None` when this call ran no step.
    pub final_loss: Option<f32>,
    /// Steps completed in the whole run, including any before a resume.
    pub steps_completed: u32,
    /// The step count a saved state resumed this call at, if it did.
    pub resumed_at: Option<u32>,
    /// True when [`FitControl::on_step`] stopped the run before
    /// [`FitOpts::steps`].
    pub interrupted: bool,
    /// Every held-out evaluation of the whole run, in order, those before a
    /// resume included.
    pub evaluations: Vec<Evaluation>,
    /// The evaluation whose parameters the run kept - what the model handed
    /// back holds, and `out` when given - when it selected on held-out loss
    /// ([`FitOpts::patience`]) and evaluated at least once; `None` when they
    /// are the last step's.
    pub kept_best: Option<KeptBest>,
    /// True when the held-out loss had not improved for
    /// [`FitOpts::patience`] evaluations and the run stopped there.
    pub stopped_early: bool,
}

/// The format tag of an exact-resume state file.
const TRAIN_STATE_FORMAT: &str = "brain-train-state/1";

/// Everything in [`FitOpts`] a step's arithmetic or batch depends on - what a
/// saved state must agree with to be continued. Rates are compared by their
/// bits: a decimal round trip through JSON is not guaranteed to give the
/// same `f32` back, and "the same rate" here means the same bits.
fn schedule_fingerprint(opts: &FitOpts) -> serde_json::Value {
    let mut fingerprint = serde_json::json!({
        "steps": opts.steps,
        "batch_size": opts.batch_size,
        "block_size": opts.block_size,
        "lr_bits": opts.lr.to_bits(),
        "min_lr_bits": opts.min_lr.to_bits(),
        "warmup": opts.warmup,
        "decay_iters": opts.decay_iters,
        "weight_decay_bits": opts.weight_decay.to_bits(),
        "grad_clip_bits": opts.grad_clip.to_bits(),
        "grad_accum": opts.grad_accum,
        "seed": opts.seed,
        "mask_before": opts.mask_before.map(String::from),
        "mask_per_line": opts.mask_per_line,
        "align_to_lines": opts.align_to_lines,
        "eval_interval": opts.eval_interval,
        "eval_batches": opts.eval_batches,
        "patience": opts.patience,
        "cooldown_steps": opts.cooldown_steps,
    });
    // A state saved under another precision cannot be continued; the key is absent for the default so states
    // written before the field existed still resume.
    if !opts.precision.is_default() {
        fingerprint["precision"] = serde_json::Value::String(format!("{:?}", opts.precision));
    }
    fingerprint
}

/// Where a model's [`crate::OffDeviceState`] is committed for the resume file at `path`.
fn off_device_dir(path: &Path) -> std::path::PathBuf {
    path.with_extension("offdevice")
}

/// Write the exact-resume state of a run that has completed `step` steps,
/// atomically: a crash mid-write leaves the previous state, never half of
/// this one. A run selecting on held-out loss saves its selection with it:
/// the watch, the curve and the best parameters so far.
fn save_train_state<M: Model>(model: &M, path: &Path, step: u32, applied: u32, rng: &Rng, opts: &FitOpts, identity: &serde_json::Value, selection: Option<&Selection>) -> std::io::Result<()> {
    let names = model.optimized_params().ok_or_else(|| std::io::Error::other("exact resume: this model does not expose its optimizer state"))?;
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::with_capacity(names.len() * 4);
    for name in &names {
        let (m, v) = model.read_moments(name).ok_or_else(|| std::io::Error::other(format!("exact resume: {name} has no optimizer moments to save")))?;
        let w = model.read_weight(name);
        tensors.push((format!("w:{name}"), vec![w.len() as u64], w));
        tensors.push((format!("m:{name}"), vec![m.len() as u64], m));
        tensors.push((format!("v:{name}"), vec![v.len() as u64], v));
    }
    if let Some(held) = selection.and_then(|s| s.held.as_ref()) {
        for (name, w) in held {
            tensors.push((format!("best:{name}"), vec![w.len() as u64], w.clone()));
        }
    }
    let header = serde_json::json!({
        "format": TRAIN_STATE_FORMAT,
        "step": step,
        // Steps that reached AdamW; differs from `step` once any was skipped.
        "applied": applied,
        // A string: the generator's state is a full u64, and a JSON reader
        // that goes through f64 would round it.
        "rng": rng.state().to_string(),
        "opts": schedule_fingerprint(opts),
        "identity": identity,
        "selection": selection.map(Selection::to_json),
    });
    // The off-device generation first: the resume file below names `applied`, so a crash between the two leaves
    // the previous resume file and a generation it does not use, never a file naming a state that is missing.
    if let Some(off) = model.off_device_state() {
        off.commit(&off_device_dir(path), applied)?;
    } else if let Some(host) = model.optimizer_parts().filter(|p| !p.store.offload.is_empty()) {
        // The moments of an offloaded store live in its host optimizer: committed beside the resume file the same way.
        host.optim.host_commit(host.gpu, host.store, &optim::checkpoint::CheckpointDir::new(off_device_dir(path))?, applied)?;
    }
    let tmp = path.with_extension("tmp");
    let tmp_str = tmp.to_str().ok_or_else(|| std::io::Error::other(format!("{}: not a UTF-8 path", tmp.display())))?;
    checkpoint::st::save_safetensors(tmp_str, &tensors, &header, None)?;
    std::fs::rename(&tmp, path)
}

/// The seed the run that wrote the exact-resume state at `path` trained with,
/// `None` when no state is there. A run resumed without a seed of its own
/// takes this one, because the state only resumes under the options it was
/// saved with.
pub fn saved_seed(path: &Path) -> std::io::Result<Option<u64>> {
    if !path.exists() {
        return Ok(None);
    }
    let invalid = |why: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {why}", path.display()));
    let path_str = path.to_str().ok_or_else(|| invalid("not a UTF-8 path"))?;
    let st = checkpoint::st::load_safetensors(path_str)?;
    st.config().get("opts").and_then(|o| o.get("seed")).and_then(|s| s.as_u64()).map(Some).ok_or_else(|| invalid("no training seed in the saved options"))
}

/// Restore the state [`save_train_state`] wrote into `model`, returning the
/// completed step count, the AdamW applied-step count, the batch generator to continue with, and the
/// selection the run had made so far when it was selecting.
fn load_train_state<M: Model>(model: &M, path: &Path, opts: &FitOpts, identity: &serde_json::Value) -> std::io::Result<(u32, u32, Rng, Option<Selection>)> {
    let invalid = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {why}", path.display()));
    let path_str = path.to_str().ok_or_else(|| invalid("not a UTF-8 path".to_string()))?;
    let st = checkpoint::st::load_safetensors(path_str)?;
    let header = st.config();
    if header.get("format").and_then(|f| f.as_str()) != Some(TRAIN_STATE_FORMAT) {
        return Err(invalid(format!("not a {TRAIN_STATE_FORMAT} training state")));
    }
    if header.get("opts") != Some(&schedule_fingerprint(opts)) {
        return Err(invalid("saved by a run with different training options; resuming it would not continue that run".to_string()));
    }
    if header.get("identity") != Some(identity) {
        return Err(invalid("saved by a run over different data or a different starting point; resuming it would not continue that run".to_string()));
    }
    let step = header.get("step").and_then(|s| s.as_u64()).and_then(|s| u32::try_from(s).ok()).ok_or_else(|| invalid("no step count".to_string()))?;
    if step > opts.steps {
        return Err(invalid(format!("{step} steps completed, past this run's {}", opts.steps)));
    }
    // A state written before skipped steps were counted applied every step.
    let applied = match header.get("applied") {
        None => step,
        Some(a) => a.as_u64().and_then(|a| u32::try_from(a).ok()).filter(|&a| a <= step).ok_or_else(|| invalid("applied-step count is not within the completed steps".to_string()))?,
    };
    let rng = header.get("rng").and_then(|r| r.as_str()).and_then(|r| r.parse::<u64>().ok()).map(Rng::from_state).ok_or_else(|| invalid("no batch generator state".to_string()))?;
    let names = model.optimized_params().ok_or_else(|| std::io::Error::other("exact resume: this model does not expose its optimizer state"))?;
    let saved = st.tensors.keys().filter(|k| k.starts_with("w:")).count();
    if saved != names.len() {
        return Err(invalid(format!("holds {saved} optimised parameters, this model has {}", names.len())));
    }
    for name in &names {
        let get = |kind: &str| st.tensors.get(&format!("{kind}:{name}")).ok_or_else(|| invalid(format!("no {kind} for {name}")));
        let (w, m, v) = (get("w")?, get("m")?, get("v")?);
        if w.len() != model.read_weight(name).len() {
            return Err(invalid(format!("{name} holds {} values, this model's has a different size", w.len())));
        }
        model.write_weight(name, w);
        model.write_moments(name, m, v).map_err(std::io::Error::other)?;
    }
    if let Some(off) = model.off_device_state() {
        off.restore(&off_device_dir(path), applied)?;
    } else if let Some(host) = model.optimizer_parts().filter(|p| !p.store.offload.is_empty()) {
        let dir = off_device_dir(path);
        if !host.optim.host_restore_at(host.gpu, host.store, &optim::checkpoint::CheckpointDir::new(&dir)?, applied)? {
            return Err(invalid(format!("{}: no host optimizer state was committed at update {applied}", dir.display())));
        }
    }
    model.poll_wait();
    let selection = match (&header["selection"], opts.patience > 0) {
        (serde_json::Value::Null, false) => None,
        (serde_json::Value::Null, true) => return Err(invalid("saved by a run that did not select on held-out loss".to_string())),
        (saved, _) => Some(Selection::from_json(opts, saved, &st.tensors).ok_or_else(|| invalid("its selection record cannot be read".to_string()))?),
    };
    Ok((step, applied, rng, selection))
}

/// [`fit_with`], with the caller in the loop: a per-step report that can
/// stop the run, and exact resume through a saved training state (see
/// [`FitControl`]). Hands the trained model back, so a caller that wants
/// only part of it (a LoRA adapter) need not write and re-read a whole
/// checkpoint.
///
/// **Selecting on held-out loss.** With a [`FitOpts::patience`], every evaluation ([`FitOpts::eval_interval`],
/// and the last step) is a candidate: the parameters of the best one are
/// held in host memory and restored into the model before it is handed back
/// (and written to `out` when given), so what a caller exports is the
/// checkpoint the best held-out loss was measured on, not the last one
/// trained. A trainable set too large to hold ([`FitOpts::best_hold_floats`]) is written to `out` at each
/// improvement instead, as this always did, or - with no `out` - spilled beside [`FitControl::state`]
/// and read back at the end; with neither the run is refused at the first improvement, since
/// returning the last step as the selection would silently defeat early stopping. The selection is
/// part of the exact-resume state.
///
/// **Data parallelism.** A [`FitControl::grad_sync`] makes the run one rank of
/// a distributed job: each step's gradients are averaged over the group before
/// the optimizer runs, and the objective trains on its own shard of the
/// training-record stream ([`Objective::shard`]). Only rank 0 writes `out` and
/// the evaluation snapshots. A group whose objective cannot shard, or whose
/// model cannot write reduced gradients, is an error; it never falls back to a
/// single process.
pub fn fit_controlled<M: Model, O: Objective<M>>(model: M, obj: O, opts: &FitOpts, out: Option<&Path>, control: FitControl<'_>) -> std::io::Result<(FitReport, M)> {
    fit_controlled_sync(model, obj, opts, out, control, None)
}

/// Where a run's gradients are reduced: the group, and this rank in it.
#[derive(Clone, Copy)]
pub struct GradSync<'a> {
    /// The ranks to reduce with.
    pub collective: &'a dyn crate::Collective,
    /// This process's rank.
    pub rank: usize,
    /// The job and plan a checkpoint commit names, when the caller knows them.
    /// Without it a distributed run still trains, but publishes no commit and
    /// cannot verify what it resumes from.
    pub checkpoint: Option<CheckpointIdentity<'a>>,
}

/// What a checkpoint commit is filed under: the job's durable identity and the
/// exact plan every rank ran. A resume only continues a commit that names both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointIdentity<'a> {
    /// The job's identity, as the launcher named it.
    pub job: &'a str,
    /// The plan identity every rank joined under.
    pub plan: &'a str,
}

/// The held-out evaluation a step is due, agreed across `grad_sync`'s group.
///
/// Without a group this is the objective's own evaluation, unchanged. With
/// one, rank 0 scores the held-out batches and the scalar is broadcast to
/// every rank, because the replicas hold identical weights: a second rank
/// evaluating the same split would repeat the work for the same number.
///
/// The broadcast is called on every rank whenever the step is due, including
/// when the objective reports no evaluation: a rank 0 that skipped it while
/// its peers waited for the result would deadlock the group. The root's
/// "no evaluation" is therefore itself the broadcast value.
fn agreed_eval<M: Model, O: Objective<M>>(obj: &mut O, model: &M, rng: &Rng, batches: u32, grad_sync: Option<&GradSync<'_>>) -> std::io::Result<Option<f32>> {
    let Some(sync) = grad_sync else {
        return Ok(obj.eval(model, &mut rng.clone(), batches));
    };
    // Only the root computes; everyone else sends a placeholder the root's
    // value replaces. `broadcast` ignores it, so its contents do not matter.
    let local = if sync.rank == 0 {
        match obj.eval(model, &mut rng.clone(), batches) {
            Some(loss) => crate::collective::Payload::f32(vec![1.0, loss]),
            None => crate::collective::Payload::f32(vec![0.0, 0.0]),
        }
    } else {
        crate::collective::Payload::f32(Vec::new())
    };
    let agreed = pollster::block_on(sync.collective.broadcast(sync.rank, local, 0)).map_err(std::io::Error::other)?;
    if agreed.data.len() != 2 {
        return Err(std::io::Error::other(format!("the evaluation broadcast from rank 0 carried {} values, expected 2", agreed.data.len())));
    }
    Ok((agreed.data[0] != 0.0).then_some(agreed.data[1]))
}

/// [`fit_controlled`] with every optimizer step's gradients averaged across
/// `grad_sync`'s ranks first, which is what a data-parallel run needs. A group
/// already named by [`FitControl::grad_sync`] is the same request; naming a
/// different one is refused, not overridden.
///
/// Exact resume is refused with a sync: the state is one rank's, and a world
/// size is not part of its identity yet.
pub fn fit_controlled_sync<M: Model, O: Objective<M>>(model: M, obj: O, opts: &FitOpts, out: Option<&Path>, control: FitControl<'_>, grad_sync: Option<GradSync<'_>>) -> std::io::Result<(FitReport, M)> {
    fit_controlled_sharded(model, obj, opts, out, control, grad_sync, None)
}

/// [`fit_controlled_sync`] with each rank training on its own shard of the
/// batch stream ([`Objective::shard`]), so the gradients being averaged come
/// from different batches rather than the same one computed `world` times.
///
/// `shard` is `(rank, world)` and must describe the same rank and group as
/// `grad_sync` when both are given: a rank training on one shard while
/// contributing to another position's group would combine batches that do not
/// belong together. Without an explicit `shard`, a `grad_sync` (given here or
/// named by [`FitControl::grad_sync`]) shards the training stream by its own
/// rank and world size.
///
/// Only rank 0 writes: the checkpoint (`out`) and the evaluation snapshots. The
/// replicas are bit-identical, so one writer is enough, and several ranks
/// writing one path at once would corrupt it.
///
/// For the same reason rank 0 alone scores a due held-out evaluation: the
/// replicas hold identical weights, so a second rank's pass over the same
/// held-out split adds nothing to the number and costs a full evaluation on
/// every replica. The scalar is broadcast to the rest of the group, so every
/// rank drives its selection and early stopping from the same value.
#[cfg(target_arch = "wasm32")]
fn publish_commit(_: CheckpointIdentity<'_>, _: &Path, _: u32, _: usize) -> std::io::Result<()> {
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn verify_resume(_: CheckpointIdentity<'_>, _: &Path, _: usize) -> std::io::Result<()> {
    Ok(())
}

/// Publish the checkpoint `out` as a committed generation for `identity`, so a
/// restart can verify what it resumes from and another system can find it.
#[cfg(not(target_arch = "wasm32"))]
fn publish_commit(identity: CheckpointIdentity<'_>, out: &Path, step: u32, world: usize) -> std::io::Result<()> {
    let absolute = std::fs::canonicalize(out).map_err(|e| std::io::Error::other(format!("locating {} for its commit: {e}", out.display())))?;
    let dir = absolute.parent().ok_or_else(|| std::io::Error::other(format!("{} has no directory to commit into", out.display())))?;
    let commit = crate::job::CheckpointCommit::build(identity.job, identity.plan, step, world, dir, &[("weights", &absolute)])
        .map_err(|e| std::io::Error::other(format!("committing {}: {e}", out.display())))?;
    commit.write(dir).map_err(|e| std::io::Error::other(format!("committing {}: {e}", out.display())))?;
    Ok(())
}

/// Refuse to resume unless the newest committed generation beside `out` agrees
/// with this job, plan and world and every file still verifies.
#[cfg(not(target_arch = "wasm32"))]
fn verify_resume(identity: CheckpointIdentity<'_>, out: &Path, world: usize) -> std::io::Result<()> {
    let dir = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let Some(commit) = crate::job::CheckpointCommit::latest(dir).map_err(|e| std::io::Error::other(format!("reading the commits in {}: {e}", dir.display())))? else {
        return Err(std::io::Error::other(format!("{} holds a training state but no committed checkpoint; refusing to resume from an unverified state", dir.display())));
    };
    commit.verify(dir).map_err(|e| std::io::Error::other(format!("the checkpoint to resume is not valid: {e}")))?;
    if !commit.compatible_with(identity.job, identity.plan, world) {
        return Err(std::io::Error::other("the newest committed checkpoint belongs to a different job, plan or world; refusing to resume"));
    }
    Ok(())
}

pub fn fit_controlled_sharded<M: Model, O: Objective<M>>(
    mut model: M,
    mut obj: O,
    opts: &FitOpts,
    out: Option<&Path>,
    mut control: FitControl<'_>,
    grad_sync: Option<GradSync<'_>>,
    shard: Option<(usize, usize)>,
) -> std::io::Result<(FitReport, M)> {
    // The control's group is a request every caller of the shared loop honors,
    // so a trainer that reaches `fit_controlled` cannot silently run the job on
    // one process. Naming two different groups is contradictory, not an
    // override.
    let two_groups = matches!(
        (&grad_sync, &control.grad_sync),
        (Some(explicit), Some(from_control))
            if (explicit.rank, explicit.collective.world_size()) != (from_control.rank, from_control.collective.world_size())
    );
    if two_groups {
        return Err(std::io::Error::other("the gradient group passed to the run disagrees with the one its FitControl names; a run has one group"));
    }
    let grad_sync = grad_sync.or(control.grad_sync);
    let shard = shard.or_else(|| grad_sync.as_ref().map(|sync| (sync.rank, sync.collective.world_size())));
    if let Some(sync) = &grad_sync {
        if model.param_names().is_empty() {
            return Err(std::io::Error::other("the distributed run reduces gradients, but this model exposes no parameters to reduce"));
        }
        if sync.collective.world_size() == 0 {
            return Err(std::io::Error::other("the distributed run's gradient group has no ranks"));
        }
    }
    // One writer: every rank holds the same replica, so a second writer would
    // only race the first.
    let writer = grad_sync.as_ref().is_none_or(|sync| sync.rank == 0);
    let out = out.filter(|_| writer);
    // A writer publishes what it saved, so a later attempt can verify exactly
    // what it resumes from and another system can find the artifact.
    let commit = |path: &Path, step: u32| -> std::io::Result<()> {
        match grad_sync.as_ref().and_then(|sync| sync.checkpoint.map(|identity| (identity, sync.collective.world_size()))) {
            Some((identity, world)) => publish_commit(identity, path, step, world),
            None => Ok(()),
        }
    };
    if let (Some(sync), Some(_)) = (&grad_sync, control.state) {
        let world = sync.collective.world_size();
        // A resumed group must be the same group. The caller's identity is what
        // the state loader compares, so the world size has to be in it, not
        // inferred from whatever the resume happens to run with.
        if control.identity.get("world_size").and_then(serde_json::Value::as_u64) != Some(world as u64) {
            return Err(std::io::Error::other(format!(
                "a distributed run that keeps exact resume state must name its world size ({world}) in the run identity"
            )));
        }
    }
    if let Some((rank, world)) = shard {
        if world == 0 || rank >= world {
            return Err(std::io::Error::other(format!("shard rank {rank} is not one of a world of {world}")));
        }
        if let Some(sync) = &grad_sync {
            let group = sync.collective.world_size();
            if (sync.rank, group) != (rank, world) {
                return Err(std::io::Error::other(format!(
                    "the batch shard {rank}/{world} disagrees with the gradient group's rank {}/{group}; every rank must train on the shard that matches its position in the group",
                    sync.rank
                )));
            }
        }
        obj.shard(rank, world)?;
    }
    obj.prepare(&mut model)?;
    // The run's optimizer-state precision reaches every model here, before any step or resume.
    crate::trainprec::apply_train_precision(&mut model, &opts.precision, std::any::type_name::<M>()).map_err(std::io::Error::other)?;
    model.pin_rounding_seed(opts.seed);
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);
    let mut first_step = 0u32;
    // AdamW's step index `t` is the number of updates its moments have
    // absorbed; a step the objective skipped absorbed none, so it must not
    // advance the bias corrections.
    let mut applied = 0u32;
    let mut resumed_at = None;
    let mut selection = Selection::new(opts);
    if let Some(state) = control.state.filter(|p| p.exists()) {
        if let (Some(sync), Some(path)) = (grad_sync.as_ref(), out) {
            if let Some(identity) = sync.checkpoint {
                verify_resume(identity, path, sync.collective.world_size())?;
            }
        }
        let (step, saved_applied, saved_rng, saved) = load_train_state(&model, state, opts, &control.identity)?;
        eprintln!("resuming at step {step} from {}", state.display());
        first_step = step;
        applied = saved_applied;
        rng = saved_rng;
        resumed_at = Some(step);
        if let Some(saved) = saved {
            selection = saved;
        }
        if let Some(spill) = control.state.map(spill_path).filter(|spill| selection.spilled && !spill.exists()) {
            return Err(std::io::Error::other(format!("{}: the training state names a spilled best-parameters file that is gone; resuming would lose the best evaluation", spill.display())));
        }
    }

    obj.begin(opts.seed, u64::from(first_step) * u64::from(opts.grad_accum.max(1)) * opts.batch_size.max(1) as u64);

    let initial = {
        let probe_clock = std::time::Instant::now();
        let mut sample_rng = rng.clone();
        let mut total = 0.0;
        for _ in 0..5 {
            total += obj.loss_probe(&model, &mut sample_rng);
        }
        gpu_core::profile::stage_time("fit: initial loss probe (5 forwards)", probe_clock);
        total / 5.0
    };
    let mut last_train: Option<f32> = None;
    let mut completed = first_step;
    let mut interrupted = false;
    let mut last_save = std::time::Instant::now();
    // The best parameters were written to `out` as they were found (a
    // trainable set too large to hold), so the final save must not overwrite them.
    let mut best_written = false;
    let mut stopped_early = false;

    for step in first_step..opts.steps {
        let lr = selection.cooldown.map_or_else(|| cosine_lr(step, opts), |c| c.lr(step, opts.min_lr));
        let step_clock = std::time::Instant::now();
        if let Some(parts) = model.optimizer_parts() {
            parts.gpu.reset_memory_peak();
        }
        model.zero_grads();
        // Under BRAIN_PROFILE the first step prints the device time of each kernel kind it ran.
        let step_loss = match model.optimizer_parts().filter(|_| step == first_step) {
            Some(parts) => gpu_core::profile::device_table(parts.gpu, "fit: first step, forward and backward", || obj.micro_steps(&model, &mut rng, opts.grad_accum.max(1))),
            None => obj.micro_steps(&model, &mut rng, opts.grad_accum.max(1)),
        };
        gpu_core::profile::stage_time("fit: step micro-steps (forward+backward)", step_clock);
        let update_clock = std::time::Instant::now();
        match obj.step_verdict() {
            StepVerdict::Apply => {
                // average grads over accumulation steps
                let scale = 1.0 / opts.grad_accum.max(1) as f32;
                let clip = (opts.grad_clip > 0.0).then_some(opts.grad_clip);
                applied += 1;
                if let Some(sync) = &grad_sync {
                    crate::distributed::reduce_grads(&model, sync.collective, sync.rank).map_err(std::io::Error::other)?;
                }
                model.adamw_step(applied, lr, opts.weight_decay, opts.adam, clip, scale);
            }
            StepVerdict::Skip(why) => eprintln!("step {:>6}: no optimiser step taken: {why}", step + 1),
            StepVerdict::Abort(e) => return Err(e),
        }
        model.poll_wait();
        gpu_core::profile::stage_time("fit: step optimizer update", update_clock);
        gpu_core::profile::stage_time("fit: step total", step_clock);
        let grad_norm = model.last_grad_norm();
        let loss = step_loss / opts.grad_accum.max(1) as f32;
        // The optimizer's share of the step: everything the objective did not
        // measure, which is zeroing gradients, reducing them, AdamW and the
        // device wait above. The evaluation a step is due is not part of it.
        let mut work = obj.step_work().unwrap_or_default();
        work.optimizer_secs = (step_clock.elapsed().as_secs_f64() - work.prepare_secs - work.forward_secs - work.backward_secs).max(0.0);
        work.device_memory = model.optimizer_parts().and_then(|parts| parts.gpu.memory_usage());
        if let (Some(m), true) = (work.device_memory, gpu_core::profile::enabled()) {
            eprintln!("fit: device memory after step {}: live {} B, reserved {} B, peak live {} B", step + 1, m.live, m.reserved, m.peak_live);
        }
        if let Some(norm) = grad_norm {
            eprintln!("step {:>6}  loss {loss:.4}  grad_norm {norm:.4}  lr {lr:.2e}", step + 1);
        }
        last_train = Some(loss);
        completed = step + 1;
        selection.step_trained(loss);

        let last = step + 1 == opts.steps;
        let cooled = selection.cooldown.is_some_and(|c| c.ends_at(step));
        let due = opts.eval_interval > 0 && ((step + 1) % opts.eval_interval == 0 || ((last || cooled) && selection.selecting()));
        if due {
            if let Some(eval_loss) = agreed_eval(&mut obj, &model, &rng, opts.eval_batches, grad_sync.as_ref())? {
                let train_loss = selection.take_interval();
                selection.evaluations.push(Evaluation { step: step + 1, train_loss, eval_loss });
                eprintln!("step {:>6}  lr {:.2e}  train {:.4}  eval {:.4}", step + 1, lr, loss, eval_loss);
                if let Some(snapshots) = control.snapshots.as_ref().filter(|_| writer) {
                    let path = snapshots.path(step + 1);
                    crate::lora::device_adapter::save_lora_with_lineage(
                        path.to_str().expect("utf-8 path"),
                        &model,
                        &snapshots.lora,
                        snapshots.card_id,
                        snapshots.base_id,
                        snapshots.family,
                        None,
                        None,
                    )?;
                }
                match selection.watch.observe(eval_loss) {
                    Watch::Improved => {
                        selection.best_step = Some(step + 1);
                        // Hold the best parameters if they fit, and write them
                        // once after the loop. See `hold_trainable`: rewriting
                        // the whole checkpoint at every improvement cost a real
                        // run 19.8 GB of writes to preserve a 20 MB adapter. A
                        // trainable set too large to hold is written here, as
                        // this always did.
                        selection.held = None;
                        selection.spilled = false;
                        match hold_trainable(&model, opts.best_hold_floats) {
                            Some(held) => selection.held = Some(held),
                            None => match (out, control.state) {
                                (Some(p), _) => {
                                    model.save_with_itos(p.to_str().expect("utf-8 path"), obj.itos());
                                    best_written = true;
                                }
                                (None, Some(state)) => {
                                    write_spill(&model, &spill_path(state))?;
                                    selection.spilled = true;
                                }
                                // Nowhere to keep the best parameters: returning the
                                // last step as the selection would defeat the early
                                // stopping the run was asked for.
                                (None, None) => {
                                    return Err(std::io::Error::other(format!(
                                        "the best parameters of step {} exceed the in-memory budget of {} floats and the run has neither a checkpoint path nor a training state to write them to",
                                        step + 1,
                                        opts.best_hold_floats
                                    )))
                                }
                            },
                        }
                    }
                    Watch::Wait => {}
                    Watch::Stop if selection.cooldown.is_none() && opts.cooldown_steps > 0 && step + 1 < opts.steps => {
                        let steps = opts.cooldown_steps.min(opts.steps - (step + 1));
                        eprintln!(
                            "step {:>6}: held-out loss has not improved for {} evaluations (best {:.4}); cooling the rate to its floor over {} steps",
                            step + 1,
                            opts.patience,
                            selection.watch.best(),
                            steps
                        );
                        selection.cooldown = Some(Cooldown { first_step: step + 1, steps, from_lr: lr });
                    }
                    Watch::Stop => {
                        // Patience that ran out again during the cooldown is
                        // not a second stop; the cooldown's end is.
                        if selection.cooldown.is_none() {
                            eprintln!(
                                "stopping at step {}: held-out loss has not improved for {} evaluations (best {:.4}), \
                                 while the training loss went on falling - past this point the model is learning the \
                                 training rows rather than the task",
                                step + 1,
                                opts.patience,
                                selection.watch.best()
                            );
                            stopped_early = true;
                            break;
                        }
                    }
                }
            }
            for (name, value) in obj.metrics() {
                eprintln!("  {name}: {value:.4}");
            }
        }
        if cooled {
            eprintln!("stopping at step {}: the cooldown is done; the best evaluation's parameters are kept", step + 1);
            stopped_early = true;
            break;
        }

        // Wall-clock checkpointing: once the timer has expired, the NEXT completed
        // step saves (atomic temp-rename), reports the save duration, and restarts
        // the timer. A slow big-model step thus never pays a per-eval 2.4 GB write.
        if let Some(p) = out {
            if opts.checkpoint_secs > 0 && last_save.elapsed().as_secs() >= opts.checkpoint_secs {
                let ts = std::time::Instant::now();
                model.save_with_itos(p.to_str().expect("utf-8 path"), obj.itos());
                eprintln!(
                    "step {:>6}  saved checkpoint -> {} ({:.1} s)",
                    step + 1,
                    p.display(),
                    ts.elapsed().as_secs_f64()
                );
                last_save = std::time::Instant::now();
                commit(p, completed)?;
            }
        }

        let keep_going = match control.on_step.as_mut() {
            Some(on_step) => on_step(&StepReport { step: completed, steps: opts.steps, loss, lr, grad_norm, work }),
            None => true,
        };
        if let Some(state) = control.state {
            let periodic = control.state_every > 0 && completed.is_multiple_of(control.state_every);
            if !keep_going || periodic {
                save_train_state(&model, state, completed, applied, &rng, opts, &control.identity, selection.selecting().then_some(&selection))?;
            }
        }
        if !keep_going {
            interrupted = completed < opts.steps;
            break;
        }
    }

    // Held parameters are restored into the model exactly once, here, and
    // written once when there is somewhere to write them: what is handed
    // back and what lands on disk is the checkpoint the best held-out loss
    // was measured on, not the last one trained. An interrupted run is not
    // finished: it is handed back as it stood, its selection in its state.
    let mut kept_best = best_written;
    let spill = control.state.map(spill_path).filter(|_| selection.spilled);
    let restored = match (selection.held.take(), &spill) {
        (Some(held), _) => Some(held),
        (None, Some(path)) if !interrupted => Some(read_spill(path)?),
        (None, _) => None,
    };
    if let Some(held) = restored.filter(|_| !interrupted) {
        for (name, w) in &held {
            model.write_weight(name, w);
        }
        model.poll_wait();
        if let Some(p) = out {
            let ts = std::time::Instant::now();
            model.save_with_itos(p.to_str().expect("utf-8 path"), obj.itos());
            eprintln!("restored the best parameters -> {} ({:.1} s)", p.display(), ts.elapsed().as_secs_f64());
            commit(p, completed)?;
        }
        kept_best = true;
        if let Some(path) = &spill {
            std::fs::remove_file(path)?;
        }
    }

    // The final save would overwrite the best checkpoint with the last one,
    // which is exactly the model early stopping exists to discard.
    if let Some(p) = out.filter(|_| !kept_best) {
        let ts = std::time::Instant::now();
        model.save_with_itos(p.to_str().expect("utf-8 path"), obj.itos());
        eprintln!("saved checkpoint -> {} ({:.1} s)", p.display(), ts.elapsed().as_secs_f64());
        commit(p, completed)?;
    } else if kept_best {
        eprintln!(
            "kept the checkpoint with the best held-out loss ({:.4}){}",
            selection.watch.best(),
            if stopped_early { ", stopped early" } else { "" }
        );
    }
    let kept_best = kept_best.then(|| selection.best_step.map(|step| KeptBest { step, eval_loss: selection.watch.best() })).flatten();
    let report = FitReport { initial_loss: initial, final_loss: last_train, steps_completed: completed, resumed_at, interrupted, evaluations: selection.evaluations, kept_best, stopped_early };
    Ok((report, model))
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
    let obj = CausalLm::new(loaded.train, loaded.val, loaded.batch_cfg, loaded.itos);
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
    // The MODEL's vocabulary wins, not the dataset's. `fit` takes the
    // dataset's because it is sizing a model that does not exist yet; here
    // the model exists and is being fine-tuned, and a pretrained embedding
    // reshaped to fit a tokenizer's count is a different model. Real case:
    // a Qwen3 checkpoint's embedding is padded to 151936 rows while its
    // tokenizer reports 151669, and taking the dataset's number silently
    // rebuilt the embedding 273408 rows short.
    let vocab = cfg.vocab();
    if loaded.vocab > vocab {
        return Err(std::io::Error::other(format!(
            "fit_from: the dataset uses {} token ids and this model has {vocab} - the ids past the end have no embedding row",
            loaded.vocab
        )));
    }
    let cfg = cfg.finalize_for_dataset(vocab, opts.block_size);
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
    let model = build_for::<M>(cfg, opts, &weights)?;
    let obj = CausalLm::new(loaded.train, loaded.val, loaded.batch_cfg, loaded.itos);
    fit_with(model, obj, opts, out)
}

/// Train `M` on one rank's shard of the dataset in `dir`, averaging every
/// step's gradients over `grad_sync`'s group - the library entry point a
/// distributed launcher drives (`gpt2::train::train_distributed` is the GPT
/// binding of it).
///
/// The model and the dataset are built exactly as [`fit`] builds them, so
/// every rank starts from the same weights (an existing checkpoint at `out`
/// is resumed, else the run's seed initialises them), and each rank trains on
/// its own interleaved shard of the batch stream ([`Objective::shard`]). Only
/// rank 0 writes `out`: the replicas are bit-identical, so one writer is
/// enough, and several ranks writing one path at once would corrupt it. The
/// other ranks still read `out` to resume.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_distributed<M: Model>(
    dir: &Path,
    cfg: M::Config,
    opts: &FitOpts,
    out: Option<&Path>,
    grad_sync: GradSync<'_>,
) -> std::io::Result<(FitReport, M)> {
    let (train, val, batch_cfg, vocab, itos) = load_dataset_with_itos(dir, opts)?;
    let model = build_or_resume::<M>(cfg, opts, out, vocab);
    let obj = CausalLm::new(train, val, batch_cfg, itos);
    let shard = (grad_sync.rank, grad_sync.collective.world_size());
    let write = if grad_sync.rank == 0 { out } else { None };
    fit_controlled_sharded(model, obj, opts, write, FitControl::default(), Some(grad_sync), Some(shard))
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

    let params = RolloutParams { max_new, sample: SampleParams { temp: temperature, top_k, top_p: 1.0, ..SampleParams::greedy() }, eos: None };
    let mut rollout = ModelRollout::new(m);
    rollout.sample_n(prompt, 1, &params, rng).pop().expect("sample_n(1) returns exactly one completion").tokens
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::training::TokenObjectives;

    /// A caller of the training loop (the CLI, the SDK, a program that prints
    /// a JSON report) owns stdout; progress is a diagnostic and goes to
    /// stderr. A `println!` in the loop made a report unparseable.
    #[test]
    fn the_training_loop_writes_progress_to_stderr_never_stdout() {
        let source = include_str!("train.rs");
        let production = source.split("#[cfg(test)]").next().unwrap();
        let offenders: Vec<usize> = production.lines().enumerate().filter(|(_, l)| l.contains(" println!(") || l.trim_start().starts_with("println!(")).map(|(n, _)| n + 1).collect();
        assert!(offenders.is_empty(), "println! on lines {offenders:?}: use eprintln!");
    }

    /// The shape this exists to catch: training loss falling all the way
    /// while held-out loss bottoms out and turns back up.
    ///
    /// The stop must land on the TURN, not on the end of the run, and the
    /// best it reports must be the bottom of the curve rather than the last
    /// thing it saw.
    #[test]
    fn it_stops_where_the_held_out_loss_turned_and_keeps_that_checkpoint() {
        // A real overfitting curve: down to 0.42 at the fourth eval, then up.
        let evals = [3.10f32, 1.40, 0.80, 0.42, 0.55, 0.71, 0.93, 1.20];
        let mut w = EarlyStop::new(2);
        let mut stopped_at = None;
        for (i, e) in evals.iter().enumerate() {
            match w.observe(*e) {
                Watch::Stop => {
                    stopped_at = Some(i);
                    break;
                }
                Watch::Improved | Watch::Wait => {}
            }
        }
        assert_eq!(stopped_at, Some(5), "two evaluations after the turn at index 3, not at the end of the run");
        assert!((w.best() - 0.42).abs() < 1e-6, "the best is the bottom of the curve, got {}", w.best());
    }

    /// A curve that is still improving must never be stopped, however long
    /// it runs. A stopper that fires on a healthy run is worse than none.
    #[test]
    fn a_run_that_keeps_improving_is_never_stopped() {
        let mut w = EarlyStop::new(2);
        for i in 0..50 {
            let eval = 3.0 - i as f32 * 0.05;
            assert_eq!(w.observe(eval), Watch::Improved, "step {i}");
        }
    }

    /// Patience is a budget, not a hair trigger: a single worse evaluation
    /// inside a run that then improves again is noise.
    #[test]
    fn one_bad_evaluation_inside_an_improving_run_does_not_stop_it() {
        let mut w = EarlyStop::new(3);
        assert_eq!(w.observe(1.00), Watch::Improved);
        assert_eq!(w.observe(1.05), Watch::Wait);
        assert_eq!(w.observe(0.90), Watch::Improved, "improving again resets the budget");
        assert_eq!(w.observe(0.95), Watch::Wait);
        assert_eq!(w.observe(0.96), Watch::Wait);
        assert_eq!(w.observe(0.97), Watch::Stop);
    }

    /// Disarmed is disarmed: the default must behave exactly as it did
    /// before this existed.
    #[test]
    fn patience_zero_never_stops_and_never_claims_a_best() {
        let mut w = EarlyStop::new(0);
        assert!(!w.armed());
        for e in [5.0f32, 0.1, 9.9, 0.01] {
            assert_eq!(w.observe(e), Watch::Wait);
        }
        assert_eq!(w.best(), f32::INFINITY, "a disarmed watcher has no opinion about any checkpoint");
    }

    /// A diverged step must not be allowed to claim the checkpoint.
    #[test]
    fn a_nan_evaluation_is_not_an_improvement() {
        let mut w = EarlyStop::new(2);
        assert_eq!(w.observe(0.5), Watch::Improved);
        assert_eq!(w.observe(f32::NAN), Watch::Wait);
        assert!((w.best() - 0.5).abs() < 1e-6, "NaN must not become the best, got {}", w.best());
    }

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

    /// `hold` is the constant-then-cooldown shape (arXiv:2405.18392): the rate for every step before the cooldown is the
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

    /// A model whose only parameter records which step produced it, and
    /// which remembers every save it was asked to make. Its forward is the
    /// mean of the batch's supervised targets, so an evaluation's arithmetic
    /// can be checked; its moments are the step count, so a resume can be.
    struct Recorder {
        w: std::cell::RefCell<Vec<f32>>,
        moments: std::cell::RefCell<(Vec<f32>, Vec<f32>)>,
        targets: std::cell::RefCell<Vec<u32>>,
        saves: std::rc::Rc<std::cell::RefCell<Vec<Vec<f32>>>>,
        /// The AdamW hyperparameters of every step, in order.
        adams: std::rc::Rc<std::cell::RefCell<Vec<optim::Adam>>>,
        /// The learning rate of every step, in order.
        lrs: std::rc::Rc<std::cell::RefCell<Vec<f32>>>,
        /// The AdamW step index `t` of every applied step, in order.
        indices: std::rc::Rc<std::cell::RefCell<Vec<u32>>>,
        /// The loss denominator named for each batch, in order.
        denominators: std::rc::Rc<std::cell::RefCell<Vec<f32>>>,
        /// The first supervised target of each batch run forward, in order:
        /// which example it was, when each example's targets are its own.
        drawn: std::rc::Rc<std::cell::RefCell<Vec<u32>>>,
    }

    #[derive(Clone)]
    struct RecorderCfg;
    impl ModelConfig for RecorderCfg {
        fn param_list(&self) -> Vec<(String, usize)> {
            vec![("w".into(), 1)]
        }
        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn from_json(_v: &serde_json::Value) -> Self {
            RecorderCfg
        }
        fn vocab(&self) -> u32 {
            0
        }
        fn block_size(&self) -> u32 {
            0
        }
        fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
            self
        }
    }

    impl Model for Recorder {
        type Config = RecorderCfg;
        fn new(_cfg: RecorderCfg, _b: u32, _t: u32, _init: &HashMap<String, Vec<f32>>) -> Self {
            Recorder { w: std::cell::RefCell::new(vec![0.0]), moments: std::cell::RefCell::new((vec![0.0], vec![0.0])), targets: Default::default(), saves: Default::default(), adams: Default::default(), denominators: Default::default(), drawn: Default::default(), lrs: Default::default(), indices: Default::default() }
        }
        fn init_weights(_cfg: &RecorderCfg, _seed: u64) -> HashMap<String, Vec<f32>> {
            HashMap::new()
        }
        fn config(&self) -> &RecorderCfg {
            &RecorderCfg
        }
        fn set_batch(&self, b: Batch) {
            if let Batch::Lm { targets, .. } = b {
                *self.targets.borrow_mut() = targets.to_vec();
            }
        }
        fn token_objectives(&self) -> Option<&dyn TokenObjectives> {
            Some(self)
        }
        fn forward(&self) -> f32 {
            let targets = self.targets.borrow();
            self.drawn.borrow_mut().extend(targets.iter().find(|&&t| t != IGNORE));
            let supervised: Vec<f32> = targets.iter().filter(|&&t| t != IGNORE).map(|&t| t as f32).collect();
            if supervised.is_empty() {
                0.0
            } else {
                supervised.iter().sum::<f32>() / supervised.len() as f32
            }
        }
        fn backward(&self) {}
        fn zero_grads(&self) {}
        /// The step number IS the weight, so a saved checkpoint says which
        /// step it came from.
        fn adamw_step(&self, t: u32, lr: f32, _wd: f32, adam: optim::Adam, _clip: Option<f32>, _extra: f32) {
            self.lrs.borrow_mut().push(lr);
            self.indices.borrow_mut().push(t);
            *self.w.borrow_mut() = vec![t as f32];
            *self.moments.borrow_mut() = (vec![t as f32], vec![t as f32]);
            self.adams.borrow_mut().push(adam);
        }
        fn poll_wait(&self) {}
        fn optimized_params(&self) -> Option<Vec<String>> {
            Some(vec!["w".into()])
        }
        fn read_moments(&self, _name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
            Some(self.moments.borrow().clone())
        }
        fn write_moments(&self, _name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
            *self.moments.borrow_mut() = (m.to_vec(), v.to_vec());
            Ok(())
        }
        fn param_names(&self) -> Vec<String> {
            vec!["w".into()]
        }
        fn read_weight(&self, _name: &str) -> Vec<f32> {
            self.w.borrow().clone()
        }
        fn write_weight(&self, _name: &str, data: &[f32]) {
            *self.w.borrow_mut() = data.to_vec();
        }
        fn read_grad(&self, _name: &str) -> Vec<f32> {
            vec![0.0]
        }
        fn write_grad(&self, _name: &str, _data: &[f32]) -> Result<(), String> {
            Ok(())
        }
        fn logits_all(&self, _t: &[u32]) -> Option<Vec<f32>> {
            None
        }
        fn save(&self, _path: &str) {
            self.saves.borrow_mut().push(self.w.borrow().clone());
        }
        fn config_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    /// Only the loss denominator is observable on [`Recorder`]; the rest of
    /// the capability has nothing to configure on a model with no logits.
    impl TokenObjectives for Recorder {
        fn enable_weighted_loss(&mut self) {}
        fn batch_token_logprobs(&self) -> Vec<f32> {
            Vec::new()
        }
        fn set_loss_weights(&self, _weights: &[f32]) {}
        fn set_loss_denominator(&self, tokens: f32) {
            self.denominators.borrow_mut().push(tokens);
        }
        fn set_logit_scale(&mut self, _scale: f32) {}
    }

    /// An objective that replays a held-out curve: down to its bottom, then up.
    struct Curve {
        evals: Vec<f32>,
        next: std::cell::Cell<usize>,
    }
    impl Objective<Recorder> for Curve {
        fn regime(&self) -> &'static str {
            "curve"
        }
        fn micro_step(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            0.0
        }
        fn eval(&mut self, _m: &Recorder, _r: &mut Rng, _b: u32) -> Option<f32> {
            let i = self.next.get();
            self.next.set(i + 1);
            self.evals.get(i).copied()
        }
    }

    /// An objective that counts its loss probes and its training steps.
    struct Spy {
        probes: std::rc::Rc<std::cell::Cell<u32>>,
        steps: std::rc::Rc<std::cell::Cell<u32>>,
        /// Whether it can read a loss without differentiating it.
        cheap_probe: bool,
    }
    impl Objective<Recorder> for Spy {
        fn regime(&self) -> &'static str {
            "spy"
        }
        fn micro_step(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            self.steps.set(self.steps.get() + 1);
            0.0
        }
        fn loss_probe(&mut self, m: &Recorder, r: &mut Rng) -> f32 {
            if !self.cheap_probe {
                return self.micro_step(m, r);
            }
            self.probes.set(self.probes.get() + 1);
            0.0
        }
    }

    /// An objective that rules on each optimiser step from a script.
    struct Verdicts(std::collections::VecDeque<StepVerdict>);
    impl Objective<Recorder> for Verdicts {
        fn regime(&self) -> &'static str {
            "verdicts"
        }
        /// It draws no data, so there is no stream to split: a group is
        /// accepted and the objective simply computes the same step on every
        /// rank. This is what lets the reduction hook be tested on its own.
        fn shard(&mut self, _rank: usize, _world: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn micro_step(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            0.0
        }
        fn step_verdict(&mut self) -> StepVerdict {
            self.0.pop_front().unwrap_or(StepVerdict::Apply)
        }
    }

    /// An objective that draws from no stream and does not override
    /// [`Objective::shard`]: the shape of every rollout-driven regime.
    struct Whole;
    impl Objective<Recorder> for Whole {
        fn regime(&self) -> &'static str {
            "whole"
        }
        fn micro_step(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            0.0
        }
    }

    /// An objective can refuse a step whose gradients hold nothing to learn
    /// from or must not reach the weights: no AdamW step is taken, the run
    /// goes on, and a later step it accepts is taken as usual.
    #[test]
    fn a_step_the_objective_skips_is_never_applied() {
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let lrs = std::rc::Rc::clone(&model.lrs);
        let script = [StepVerdict::Apply, StepVerdict::Skip("empty".into()), StepVerdict::Skip("empty".into()), StepVerdict::Apply];
        let opts = FitOpts { steps: 4, eval_interval: 0, ..Default::default() };
        let (report, _) = fit_controlled(model, Verdicts(script.into()), &opts, None, FitControl::default()).expect("fit");
        assert_eq!(lrs.borrow().len(), 2, "two of the four steps were applied");
        assert_eq!(report.steps_completed, 4, "a skipped step still advances the schedule");
    }

    /// AdamW's bias correction depends on how many updates the moments have
    /// absorbed, not on how many steps were attempted: a skipped step must not
    /// advance the index, and a resume must continue the applied count.
    #[test]
    fn adamw_index_counts_applied_steps_only() {
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let indices = std::rc::Rc::clone(&model.indices);
        let script = [StepVerdict::Apply, StepVerdict::Skip("empty".into()), StepVerdict::Apply, StepVerdict::Apply];
        let opts = FitOpts { steps: 4, eval_interval: 0, ..Default::default() };
        fit_controlled(model, Verdicts(script.into()), &opts, None, FitControl::default()).expect("fit");
        assert_eq!(*indices.borrow(), vec![1, 2, 3]);
    }

    /// A distributed step reduces the gradients across the group before the
    /// model's own optimiser runs; over one rank that reduction is the
    /// identity, so the run is unchanged.
    #[test]
    fn a_one_rank_gradient_sync_leaves_the_step_unchanged() {
        let opts = FitOpts { steps: 3, eval_interval: 0, ..Default::default() };
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let indices = std::rc::Rc::clone(&model.indices);
        let coll = crate::HostCollective::new(1);
        let sync = GradSync { collective: &*coll, rank: 0, checkpoint: None };
        fit_controlled_sync(model, Verdicts(Default::default()), &opts, None, FitControl::default(), Some(sync)).expect("fit");
        assert_eq!(*indices.borrow(), vec![1, 2, 3]);
    }

    /// The ranks of a world draw disjoint shards of one deterministic record
    /// stream, and interleaved the shards are exactly the stream an unsharded
    /// run draws: rank `r` trains on record `n` when `n % world == r`, so
    /// averaging the ranks' gradients is data parallelism rather than the same
    /// record computed twice.
    #[test]
    fn the_ranks_draw_disjoint_shards_that_union_to_the_unsharded_stream() {
        const EXAMPLES: usize = 8;
        const WORLD: usize = 2;
        let drawn = |obj: &mut CausalLm, batches: u32| {
            Objective::<Recorder>::begin(obj, 11, 0); // seed: a fixed order makes two ranks' streams comparable
            let model = Recorder::new(RecorderCfg, 1, 4, &HashMap::new());
            let mut rng = Rng::new(1); // seed: the shard is a property of the stream, not of the generator
            Objective::<Recorder>::micro_steps(obj, &model, &mut rng, batches);
            let seen: Vec<u32> = model.drawn.borrow().clone();
            seen
        };
        let whole = drawn(&mut examples_objective(&[4; EXAMPLES], 4), EXAMPLES as u32);
        assert_eq!(whole.len(), EXAMPLES);

        let mut shards = Vec::new();
        for rank in 0..WORLD {
            let mut obj = examples_objective(&[4; EXAMPLES], 4);
            Objective::<Recorder>::shard(&mut obj, rank, WORLD).expect("an example stream shards");
            let seen = drawn(&mut obj, (EXAMPLES / WORLD) as u32);
            let expected: Vec<u32> = whole.iter().skip(rank).step_by(WORLD).copied().collect();
            assert_eq!(seen, expected, "rank {rank} must take every {WORLD}th record from record {rank}");
            shards.push(seen);
        }
        let mut interleaved = Vec::with_capacity(EXAMPLES);
        for i in 0..EXAMPLES / WORLD {
            for shard in &shards {
                interleaved.push(shard[i]);
            }
        }
        assert_eq!(interleaved, whole, "the union of the shards must be the whole stream");
        let mut every: Vec<u32> = shards.iter().flatten().copied().collect();
        every.sort_unstable();
        every.dedup();
        assert_eq!(every.len(), EXAMPLES, "the shards must not overlap");
    }

    /// A shard of one is the whole stream: a run through the sharded path
    /// draws exactly the batches, in exactly the order, of an unsharded run.
    #[test]
    fn a_shard_of_one_trains_exactly_like_an_unsharded_run() {
        let opts = FitOpts { steps: 4, eval_interval: 0, ..Default::default() };
        let plain = Recorder::new(RecorderCfg, 1, 4, &HashMap::new());
        let plain_drawn = std::rc::Rc::clone(&plain.drawn);
        fit_controlled(plain, examples_objective(&[4; 6], 4), &opts, None, FitControl::default()).expect("fit");

        let sharded = Recorder::new(RecorderCfg, 1, 4, &HashMap::new());
        let sharded_drawn = std::rc::Rc::clone(&sharded.drawn);
        let coll = crate::HostCollective::new(1);
        let sync = GradSync { collective: &*coll, rank: 0, checkpoint: None };
        fit_controlled_sharded(sharded, examples_objective(&[4; 6], 4), &opts, None, FitControl::default(), Some(sync), Some((0, 1))).expect("fit");
        assert_eq!(*sharded_drawn.borrow(), *plain_drawn.borrow(), "a one-rank shard must draw the whole stream");
    }

    /// A row holds several records: rank `r` of `world` still trains on the
    /// records whose stream index leaves remainder `r`, not on whole rows. The
    /// union of the ranks' records is the unsharded stream, and each rank's
    /// sequence is that stream every `world`th record.
    #[test]
    fn records_shard_by_residue_even_when_a_row_holds_several() {
        const EXAMPLES: usize = 8;
        const WORLD: usize = 2;
        const BATCH: usize = 2;
        const BLOCK: usize = 4;
        // Each example's supervised targets are its own id (10 + i), so a row's
        // first supervised target names the record it holds.
        let rows = |model: &Recorder| -> Vec<u32> {
            let targets = model.targets.borrow();
            (0..BATCH)
                .map(|b| targets[b * BLOCK..(b + 1) * BLOCK].iter().copied().find(|&t| t != IGNORE).expect("a supervised target names the record"))
                .collect()
        };
        // Every run replays one generator seed so sharded and whole runs are comparable.
        let rng_seed = data::rng::random_seed();
        let run = |obj: &mut CausalLm, steps: u32| {
            Objective::<Recorder>::begin(obj, 21, 0);
            let model = Recorder::new(RecorderCfg, BATCH as u32, BLOCK as u32, &HashMap::new());
            let mut rng = Rng::new(rng_seed);
            let mut seen = Vec::new();
            for _ in 0..steps {
                Objective::<Recorder>::micro_step(obj, &model, &mut rng);
                seen.extend(rows(&model));
            }
            seen
        };
        let whole = run(&mut examples_objective_batched(&[BLOCK; EXAMPLES], BLOCK, BATCH), (EXAMPLES / BATCH) as u32);
        assert_eq!(whole.len(), EXAMPLES);

        let mut union = Vec::new();
        for rank in 0..WORLD {
            let mut obj = examples_objective_batched(&[BLOCK; EXAMPLES], BLOCK, BATCH);
            Objective::<Recorder>::shard(&mut obj, rank, WORLD).expect("an example stream shards by record");
            // Every step consumes BATCH * WORLD records and keeps BATCH.
            let seen = run(&mut obj, (EXAMPLES / (BATCH * WORLD)) as u32);
            let expected: Vec<u32> = whole.iter().skip(rank).step_by(WORLD).copied().collect();
            assert_eq!(seen, expected, "rank {rank} must train on records {rank}, {}, ...", rank + WORLD);
            union.extend(seen);
        }
        let mut sorted = union.clone();
        sorted.sort_unstable();
        let mut all = whole.clone();
        all.sort_unstable();
        assert_eq!(sorted, all, "the shards are disjoint and their union is the whole stream");

        // A world of one is the unsharded run, at record granularity too.
        let mut solo = examples_objective_batched(&[BLOCK; EXAMPLES], BLOCK, BATCH);
        Objective::<Recorder>::shard(&mut solo, 0, 1).expect("a one-rank shard");
        assert_eq!(run(&mut solo, (EXAMPLES / BATCH) as u32), whole, "a one-rank shard draws the whole stream in order");
    }

    /// A `FitControl` naming a gradient group is a distributed request the
    /// shared loop honors: it shards the record stream by the group's rank and
    /// reduces each step, instead of silently running one process. A one-rank
    /// group is the identity, so the run draws exactly the unsharded stream.
    #[test]
    fn a_fit_control_with_a_group_shards_the_record_stream() {
        let opts = FitOpts { steps: 2, eval_interval: 0, ..Default::default() };
        let plain = Recorder::new(RecorderCfg, 2, 4, &HashMap::new());
        let plain_drawn = std::rc::Rc::clone(&plain.drawn);
        fit_controlled(plain, examples_objective_batched(&[4; 6], 4, 2), &opts, None, FitControl::default()).expect("fit");

        let coll = crate::HostCollective::new(1);
        let control = FitControl { grad_sync: Some(GradSync { collective: &*coll, rank: 0, checkpoint: None }), ..Default::default() };
        let sharded = Recorder::new(RecorderCfg, 2, 4, &HashMap::new());
        let sharded_drawn = std::rc::Rc::clone(&sharded.drawn);
        fit_controlled(sharded, examples_objective_batched(&[4; 6], 4, 2), &opts, None, control).expect("fit");
        assert!(!plain_drawn.borrow().is_empty());
        assert_eq!(*sharded_drawn.borrow(), *plain_drawn.borrow(), "a one-rank group must draw exactly the unsharded stream");
    }

    /// An objective whose held-out curve is scripted, counting its
    /// evaluations so a test can see which rank actually ran them.
    struct EvalCurve {
        evals: Vec<f32>,
        next: usize,
        evals_run: Arc<AtomicUsize>,
    }
    impl Objective<Recorder> for EvalCurve {
        fn regime(&self) -> &'static str {
            "eval-curve"
        }
        /// There is no batch stream to split here; the test is about the
        /// evaluation, so every rank may be given the same shard.
        fn shard(&mut self, _rank: usize, _world: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn micro_step(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            0.0
        }
        fn eval(&mut self, _m: &Recorder, _r: &mut Rng, _b: u32) -> Option<f32> {
            self.evals_run.fetch_add(1, Ordering::SeqCst);
            let i = self.next;
            self.next += 1;
            self.evals.get(i).copied()
        }
    }

    /// Every step of a short run is a due evaluation, so the curve below
    /// bottoms out and the patience runs out inside the budget.
    fn eval_options() -> FitOpts {
        FitOpts { steps: 4, eval_interval: 1, eval_batches: 1, patience: 2, ..Default::default() }
    }

    /// One [`EvalCurve`] run per rank over a real two-rank [`crate::HostCollective`],
    /// returning each rank's report and how many evaluations it ran itself.
    fn two_rank_run(evals: Vec<f32>) -> (Vec<FitReport>, Vec<usize>) {
        let coll = crate::HostCollective::new(2);
        let counts: Vec<Arc<AtomicUsize>> = (0..2).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        let reports = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|rank| {
                    let coll = Arc::clone(&coll);
                    let count = Arc::clone(&counts[rank]);
                    let evals = evals.clone();
                    scope.spawn(move || {
                        let opts = eval_options();
                        let obj = EvalCurve { evals, next: 0, evals_run: count };
                        let sync = GradSync { collective: &*coll, rank, checkpoint: None };
                        fit_controlled_sharded(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), obj, &opts, None, FitControl::default(), Some(sync), Some((rank, 2)))
                            .expect("fit")
                            .0
                    })
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().expect("a rank's thread")).collect::<Vec<FitReport>>()
        });
        let counts = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
        (reports, counts)
    }

    /// Replicas hold identical weights, so a held-out evaluation repeated on
    /// every rank measures the same number twice. Rank 0 scores it and
    /// broadcasts the scalar; every rank then selects, stops and reports the
    /// same thing.
    #[test]
    fn only_rank_zero_evaluates_and_every_rank_selects_the_same() {
        let (reports, evals_run) = two_rank_run(vec![1.0, 0.8, 0.9, 1.0]);
        assert_eq!(evals_run, vec![4, 0], "rank 0 runs every due evaluation; a replica must not repeat it");
        assert_eq!(reports[0], reports[1], "the broadcast must leave every rank with the same selection and score");
        assert!(reports[0].stopped_early && reports[0].steps_completed == 4, "{:?}", reports[0]);
        assert_eq!(reports[1].kept_best, Some(KeptBest { step: 2, eval_loss: 0.8 }));
    }

    /// A due step the objective has no evaluation for must still be agreed:
    /// the root broadcasts "none" rather than skipping the collective its
    /// peers wait on.
    #[test]
    fn every_rank_agrees_when_the_objective_reports_no_evaluation() {
        let (reports, evals_run) = two_rank_run(Vec::new());
        assert_eq!(evals_run, vec![4, 0], "the root still asks its objective at every due step");
        assert_eq!(reports[0], reports[1]);
        assert!(reports[0].evaluations.is_empty(), "no evaluation was reported, so none is recorded");
        assert_eq!(reports[0].steps_completed, 4);
    }

    /// Without a group the loop evaluates on the one process at every due
    /// step, exactly as before the distributed path existed.
    #[test]
    fn a_single_process_run_still_evaluates_every_due_step() {
        let count = Arc::new(AtomicUsize::new(0));
        let obj = EvalCurve { evals: vec![1.0, 0.8, 0.9, 1.0], next: 0, evals_run: Arc::clone(&count) };
        let (report, _) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), obj, &eval_options(), None, FitControl::default()).expect("fit");
        assert_eq!(count.load(Ordering::SeqCst), 4, "a single process keeps evaluating every due step");
        let (distributed, _) = two_rank_run(vec![1.0, 0.8, 0.9, 1.0]);
        assert_eq!(report, distributed[0], "the single-process run must be unchanged by the distributed path");
    }

    /// A requested group cannot be reduced to a single-process run: an
    /// objective that does not implement `Objective::shard` fails with its own
    /// reason instead of training every rank on the whole stream.
    #[test]
    fn a_group_over_an_objective_that_cannot_shard_is_refused_by_name() {
        let opts = FitOpts { steps: 1, eval_interval: 0, ..Default::default() };
        let coll = crate::HostCollective::new(1);
        let control = FitControl { grad_sync: Some(GradSync { collective: &*coll, rank: 0, checkpoint: None }), ..Default::default() };
        let err = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Whole, &opts, None, control)
            .err()
            .expect("an objective with no shard must refuse");
        assert!(err.to_string().contains("cannot split its batch stream"), "{err}");
    }

    /// The applied count is part of the exact-resume state: a run stopped
    /// after a skipped step continues the index where it left off.
    #[test]
    fn a_resume_continues_the_applied_step_index() {
        let dir = tmp("applied-index");
        let state = dir.join("train.state");
        let identity = serde_json::json!({"run": 1});
        let opts = FitOpts { steps: 4, eval_interval: 0, ..Default::default() };
        let mut stop_at_two = |s: &StepReport| s.step < 2;
        let control = FitControl { on_step: Some(&mut stop_at_two), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
        let script = [StepVerdict::Skip("empty".into()), StepVerdict::Apply];
        fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Verdicts(script.into()), &opts, None, control).expect("fit");

        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let indices = std::rc::Rc::clone(&model.indices);
        let control = FitControl { on_step: None, state: Some(&state), state_every: 0, identity, snapshots: None, grad_sync: None };
        fit_controlled(model, Verdicts(Default::default()), &opts, None, control).expect("fit");
        assert_eq!(*indices.borrow(), vec![2, 3]);
    }

    /// An abort stops the run with the objective's own error.
    #[test]
    fn an_aborting_objective_stops_the_run_with_its_error() {
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let lrs = std::rc::Rc::clone(&model.lrs);
        let script = [StepVerdict::Apply, StepVerdict::Abort(std::io::Error::other("diverged"))];
        let opts = FitOpts { steps: 4, eval_interval: 0, ..Default::default() };
        let err = fit_controlled(model, Verdicts(script.into()), &opts, None, FitControl::default()).err().expect("the run aborts");
        assert!(err.to_string().contains("diverged"), "{err}");
        assert_eq!(lrs.borrow().len(), 1, "no step after the abort");
    }

    /// The estimate of the loss a run starts from is five losses, and five
    /// backward passes just to read them cost a 7B fine-tune ten of its
    /// twelve minutes. An objective that can read a loss without
    /// differentiating it is asked to; one that cannot is stepped as before.
    #[test]
    fn the_initial_loss_is_estimated_without_backward_passes_where_the_objective_can() {
        for (cheap_probe, want_probes, want_steps) in [(true, 5, 2), (false, 0, 7)] {
            let (probes, steps) = (std::rc::Rc::new(std::cell::Cell::new(0)), std::rc::Rc::new(std::cell::Cell::new(0)));
            let spy = Spy { probes: probes.clone(), steps: steps.clone(), cheap_probe };
            let opts = FitOpts { steps: 2, eval_interval: 0, ..Default::default() };
            fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), spy, &opts, None, FitControl::default()).expect("fit");
            assert_eq!((probes.get(), steps.get()), (want_probes, want_steps), "cheap probe: {cheap_probe}");
        }
    }

    /// Keeping the best checkpoint used to mean rewriting the whole model at
    /// every improvement: a real LoRA run wrote 19.8 GB of checkpoints to
    /// preserve a 20 MB adapter. The parameters are held instead and written
    /// once - and what lands on disk must still be the best step's, not the
    /// last step's.
    #[test]
    fn the_best_parameters_are_written_once_and_are_the_best_ones() {
        let dir = tmp("best-written-once");
        // Eval every step: improves through step 3, then worsens until the
        // patience of 2 runs out.
        let evals = vec![1.0f32, 0.8, 0.6, 0.4, 0.9, 1.1];
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        // Shared, so the test sees what the training loop saved.
        let saves = std::rc::Rc::clone(&model.saves);
        let opts = FitOpts { steps: 6, eval_interval: 1, eval_batches: 1, patience: 2, ..Default::default() };
        let obj = Curve { evals, next: std::cell::Cell::new(0) };
        fit_with(model, obj, &opts, Some(&dir.join("ckpt"))).expect("fit");

        let saves = saves.borrow();
        assert_eq!(saves.len(), 1, "one write, not one per improvement: {saves:?}");
        // Step 4's eval (index 3) was the best, and `adamw_step` is called
        // with t = step + 1, so the best parameters are 4.0.
        assert_eq!(saves[0], vec![4.0], "the checkpoint must hold the best step's parameters");
    }

    /// A caller that wants the model back rather than a checkpoint (a LoRA
    /// run exporting its adapter) must get the best evaluation's parameters
    /// too, and be told which evaluation they are; the curve carries every
    /// evaluation up to the stop.
    #[test]
    fn the_model_handed_back_holds_the_best_evaluation_and_the_report_names_it() {
        let evals = vec![1.0f32, 0.8, 0.6, 0.4, 0.9, 1.1, 1.2, 1.3];
        let opts = FitOpts { steps: 8, eval_interval: 1, eval_batches: 1, patience: 2, ..Default::default() };
        let obj = Curve { evals, next: std::cell::Cell::new(0) };
        let (report, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), obj, &opts, None, FitControl::default()).expect("fit");
        assert_eq!(model.read_weight("w"), vec![4.0], "the parameters of the best evaluation, not the last step");
        assert_eq!(report.kept_best, Some(KeptBest { step: 4, eval_loss: 0.4 }));
        assert!(report.stopped_early && report.steps_completed == 6 && !report.interrupted, "{report:?}");
        assert_eq!(report.evaluations.iter().map(|e| (e.step, e.eval_loss)).collect::<Vec<_>>(), [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4), (5, 0.9), (6, 1.1)]);

        // A patience the run never reaches spends the whole budget and
        // still hands back the best; the last step is evaluated as a
        // candidate even off the cadence.
        let evals = vec![1.0f32, 0.8, 0.6, 0.4, 0.9, 0.3];
        let opts = FitOpts { steps: 11, eval_interval: 2, eval_batches: 1, patience: u32::MAX, ..Default::default() };
        let (report, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals, next: std::cell::Cell::new(0) }, &opts, None, FitControl::default()).expect("fit");
        assert_eq!(report.evaluations.iter().map(|e| e.step).collect::<Vec<_>>(), [2, 4, 6, 8, 10, 11]);
        assert_eq!((report.kept_best, model.read_weight("w"), report.stopped_early), (Some(KeptBest { step: 11, eval_loss: 0.3 }), vec![11.0], false));

        // Not selecting, evaluations are recorded and nothing is kept.
        let opts = FitOpts { steps: 4, eval_interval: 2, eval_batches: 1, ..Default::default() };
        let (report, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: vec![1.0, 2.0], next: std::cell::Cell::new(0) }, &opts, None, FitControl::default()).expect("fit");
        assert_eq!((report.kept_best, model.read_weight("w"), report.evaluations.len()), (None, vec![4.0], 2));
    }

    /// A best-so-far copy too large for host memory is spilled to disk beside the training
    /// state, never dropped: the 27B LoRA adapter (117M floats) is past the default budget, and
    /// a run that "kept" its last, diverged step instead of its best silently defeated early
    /// stopping.
    #[test]
    fn a_best_copy_too_large_to_hold_is_spilled_beside_the_state_and_restored() {
        let dir = tmp("best-spilled");
        let evals = vec![1.0f32, 0.8, 0.6, 0.4, 0.9, 1.1, 1.2, 1.3];
        let opts = FitOpts { steps: 8, eval_interval: 1, eval_batches: 1, patience: 2, best_hold_floats: 0, ..Default::default() };
        let state = dir.join("train.state");
        let control = FitControl { state: Some(&state), identity: serde_json::json!({"run": 1}), ..Default::default() };
        let (report, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: evals.clone(), next: std::cell::Cell::new(0) }, &opts, None, control).expect("fit");
        assert_eq!(model.read_weight("w"), vec![4.0], "the best evaluation's parameters, not the last step's");
        assert_eq!(report.kept_best, Some(KeptBest { step: 4, eval_loss: 0.4 }));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "the spill is removed once the best is restored");

        // Stopped after the best evaluation, the spill outlives the process and the resumed run restores it.
        let mut stop_at_five = |s: &StepReport| s.step < 5;
        let control = FitControl { on_step: Some(&mut stop_at_five), state: Some(&state), identity: serde_json::json!({"run": 1}), ..Default::default() };
        let (stopped, _) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: evals.clone(), next: std::cell::Cell::new(0) }, &opts, None, control).expect("fit");
        assert!(stopped.interrupted && stopped.kept_best.is_none(), "{stopped:?}");
        let control = FitControl { state: Some(&state), identity: serde_json::json!({"run": 1}), ..Default::default() };
        let (resumed, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: evals[5..].to_vec(), next: std::cell::Cell::new(0) }, &opts, None, control).expect("fit");
        assert_eq!((resumed.kept_best, model.read_weight("w")), (Some(KeptBest { step: 4, eval_loss: 0.4 }), vec![4.0]));
    }

    /// With nowhere to put a best copy that does not fit, the run fails at the first improvement
    /// instead of returning the last step as if it were the selection.
    #[test]
    fn a_best_copy_with_nowhere_to_go_fails_the_run_loudly() {
        let opts = FitOpts { steps: 4, eval_interval: 1, eval_batches: 1, patience: 2, best_hold_floats: 0, ..Default::default() };
        let obj = Curve { evals: vec![1.0, 0.8, 0.6, 0.4], next: std::cell::Cell::new(0) };
        let err = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), obj, &opts, None, FitControl::default()).err().expect("the run is refused");
        assert!(err.to_string().contains("best parameters"), "{err}");
    }

    /// An objective whose training loss is the step count, and whose
    /// held-out loss is constant.
    struct Counting {
        steps: std::cell::Cell<u32>,
    }
    impl Objective<Recorder> for Counting {
        fn regime(&self) -> &'static str {
            "counting"
        }
        fn micro_step(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            self.steps.set(self.steps.get() + 1);
            self.steps.get() as f32
        }
        fn loss_probe(&mut self, _m: &Recorder, _r: &mut Rng) -> f32 {
            0.0
        }
        fn eval(&mut self, _m: &Recorder, _r: &mut Rng, _b: u32) -> Option<f32> {
            Some(0.5)
        }
    }

    /// The training side of the curve is the mean over the steps since the
    /// previous evaluation - the same stretch the held-out side was measured
    /// after - not the one step the evaluation happened to land on.
    #[test]
    fn an_evaluation_carries_the_mean_training_loss_of_its_interval() {
        let opts = FitOpts { steps: 5, eval_interval: 2, eval_batches: 1, patience: u32::MAX, ..Default::default() };
        let (report, _) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Counting { steps: std::cell::Cell::new(0) }, &opts, None, FitControl::default()).expect("fit");
        assert_eq!(report.evaluations, [Evaluation { step: 2, train_loss: 1.5, eval_loss: 0.5 }, Evaluation { step: 4, train_loss: 3.5, eval_loss: 0.5 }, Evaluation { step: 5, train_loss: 5.0, eval_loss: 0.5 }]);
    }

    /// A run that selects on held-out loss and is stopped mid-way resumes
    /// with its selection - the best parameters, the watch and the curve -
    /// and ends exactly where the uninterrupted run did.
    #[test]
    fn a_selecting_run_resumes_with_its_selection() {
        let dir = tmp("selecting-resume");
        let evals = vec![1.0f32, 0.8, 0.6, 0.4, 0.9, 1.1, 1.2, 1.3];
        let opts = FitOpts { steps: 8, eval_interval: 1, eval_batches: 1, patience: 2, ..Default::default() };
        let (straight, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: evals.clone(), next: std::cell::Cell::new(0) }, &opts, None, FitControl::default()).expect("fit");
        assert_eq!(model.read_weight("w"), vec![4.0]);

        let state = dir.join("train.state");
        let mut stop_at_three = |s: &StepReport| s.step < 3;
        let control = FitControl { on_step: Some(&mut stop_at_three), state: Some(&state), state_every: 0, identity: serde_json::json!({"run": 1}), snapshots: None, grad_sync: None };
        let (stopped, _) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: evals.clone(), next: std::cell::Cell::new(0) }, &opts, None, control).expect("fit");
        assert!(stopped.interrupted && stopped.steps_completed == 3 && stopped.kept_best.is_none(), "{stopped:?}");
        assert_eq!(stopped.evaluations.len(), 3);

        // The curve continues where the first run left off: its first
        // evaluation is the fourth of the whole run.
        let control = FitControl { on_step: None, state: Some(&state), state_every: 0, identity: serde_json::json!({"run": 1}), snapshots: None, grad_sync: None };
        let (resumed, model) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: evals[3..].to_vec(), next: std::cell::Cell::new(0) }, &opts, None, control).expect("fit");
        assert_eq!(resumed.resumed_at, Some(3));
        assert_eq!(model.read_weight("w"), vec![4.0], "the best parameters of the whole run");
        assert_eq!((resumed.kept_best, resumed.stopped_early, resumed.steps_completed), (straight.kept_best, straight.stopped_early, straight.steps_completed));
        assert_eq!(resumed.evaluations, straight.evaluations, "the curve is the whole run's");

        // A state from a run that did not select is not continued by one that does.
        let plain = FitOpts { patience: 0, ..opts.clone() };
        let control = FitControl { on_step: None, state: Some(&state), state_every: 0, identity: serde_json::json!({"run": 1}), snapshots: None, grad_sync: None };
        let Err(err) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals: vec![], next: std::cell::Cell::new(0) }, &plain, None, control) else {
            panic!("a state saved with another selection must not be continued")
        };
        assert!(err.to_string().contains("different training options"), "{err}");
    }

    /// Scoring the whole validation split, one example per row in order,
    /// gives the per-position mean over every supervised position - each
    /// batch weighted back by its positions - and the same number every time.
    #[test]
    fn evaluating_every_example_is_the_per_position_mean_and_deterministic() {
        let dir = tmp("eval-every-example");
        let mut split = data::chat::EncodedSplit::default();
        for e in 0..3u32 {
            split.push(&[e * 10, e * 10 + 1, e * 10 + 2, e * 10 + 3], &[false, false, true, true]);
        }
        data::chat::write_split(&dir, "train", &split).unwrap();
        data::chat::write_split(&dir, "val", &split).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(64)).unwrap();
        let opts = FitOpts { block_size: 6, batch_size: 2, eval_batches: 0, ..Default::default() };
        let (train, val, cfg, _vocab, itos) = load_dataset_with_itos(&dir, &opts).expect("load");
        let mut obj = causal_lm::<Recorder>(train, val, cfg, itos);
        let model = Recorder::new(RecorderCfg, 2, 6, &HashMap::new());
        // Supervised targets 2, 3, 12, 13, 22, 23: a mean of 12.5, which a
        // mean of the two batch means (7.5 and 22.5) would not give.
        let first = obj.eval(&model, &mut Rng::new(data::rng::random_seed()), 0).expect("scored");
        assert!((first - 12.5).abs() < 1e-5, "{first}");
        assert_eq!(obj.eval(&model, &mut Rng::new(data::rng::random_seed()), 0), Some(first), "the same split scores the same whatever the generator");
    }

    /// Held-out scoring over any model: the supervised-position mean of the
    /// split, with its record and position counts, the base score left out
    /// when a run continues an adapter, and the tuned score filled in later.
    #[test]
    fn a_held_out_set_is_scored_before_and_after_by_any_model() {
        let dir = tmp("score-held-out");
        let mut split = data::chat::EncodedSplit::default();
        for e in 0..3u32 {
            split.push(&[e * 10, e * 10 + 1, e * 10 + 2, e * 10 + 3], &[false, false, true, true]);
        }
        data::chat::write_split(&dir, "train", &split).unwrap();
        data::chat::write_split(&dir, "val", &split).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(64)).unwrap();
        let opts = FitOpts { block_size: 6, batch_size: 2, eval_batches: 0, ..Default::default() };
        let model = Recorder::new(RecorderCfg, 2, 6, &HashMap::new());
        let sets = [dir.as_path()];
        let mut scored = crate::score_base(&model, &sets, &opts, false).expect("score");
        assert_eq!((scored[0].records, scored[0].positions), (3, 6));
        assert!((scored[0].base_loss.expect("scored") - 12.5).abs() < 1e-5);
        assert_eq!(scored[0].tuned_loss, None);
        crate::score_tuned(&model, &sets, &opts, &mut scored).expect("score");
        assert_eq!(scored[0].tuned_loss, scored[0].base_loss);
        let continued = crate::score_base(&model, &sets, &opts, true).expect("score");
        assert_eq!(continued[0].base_loss, None, "a continued adapter's start is not the base");
    }

    /// Every step runs with the run's AdamW hyperparameters, not constants.
    #[test]
    fn every_step_uses_the_runs_adam_hyperparameters() {
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let adams = std::rc::Rc::clone(&model.adams);
        let adam = optim::Adam { beta1: 0.8, beta2: 0.95, eps: 1e-6 };
        let opts = FitOpts { steps: 3, eval_interval: 0, adam, ..Default::default() };
        let obj = Curve { evals: vec![], next: std::cell::Cell::new(0) };
        fit_with(model, obj, &opts, None).expect("fit");
        assert_eq!(*adams.borrow(), vec![adam; 3]);
    }

    /// A run that stops on its plateau does not stop at a high rate: it
    /// spends its cooldown bringing the rate down to the floor, keeps
    /// selecting on what it evaluates on the way, and stops there.
    #[test]
    fn a_run_that_hits_its_patience_anneals_over_its_cooldown_before_stopping() {
        // Best at the second evaluation (step 4); patience 2 runs out at the
        // fourth (step 8); the cooldown is steps 9..=12.
        let evals = vec![0.5f32, 0.4, 0.45, 0.46, 0.47, 0.48];
        let opts = FitOpts { steps: 100, lr: 1.0, min_lr: 0.1, warmup: 0, decay_iters: 100, eval_interval: 2, eval_batches: 1, patience: 2, cooldown_steps: 4, ..Default::default() };
        let model = Recorder::new(RecorderCfg, 1, 1, &HashMap::new());
        let lrs = std::rc::Rc::clone(&model.lrs);
        let (report, model) = fit_controlled(model, Curve { evals, next: std::cell::Cell::new(0) }, &opts, None, FitControl::default()).expect("fit");
        assert!(report.stopped_early);
        assert_eq!(report.steps_completed, 12, "the cooldown, then the stop");
        assert_eq!(report.kept_best, Some(KeptBest { step: 4, eval_loss: 0.4 }));
        assert_eq!(model.read_weight("w"), vec![4.0]);
        assert_eq!(report.evaluations.last().map(|e| e.step), Some(12), "the end of the cooldown is evaluated");
        let lrs = lrs.borrow();
        let cooling = &lrs[8..];
        assert!(cooling.windows(2).all(|w| w[1] < w[0]), "{cooling:?}");
        assert!((cooling[3] - 0.1).abs() < 1e-6, "the cooldown ends at the floor: {cooling:?}");
        assert!(cooling[0] < lrs[7], "and starts from where the run stood: {lrs:?}");

        // Without a cooldown the run stops where its patience ran out.
        let opts = FitOpts { cooldown_steps: 0, ..opts };
        let evals = vec![0.5f32, 0.4, 0.45, 0.46, 0.47, 0.48];
        let (plain, _) = fit_controlled(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Curve { evals, next: std::cell::Cell::new(0) }, &opts, None, FitControl::default()).expect("fit");
        assert_eq!(plain.steps_completed, 8);
    }

    /// A causal-LM objective over `lengths.len()` examples, example `i` being
    /// `lengths[i]` tokens all `10 + i`, so a batch says which example it is.
    fn examples_objective(lengths: &[usize], block: usize) -> CausalLm {
        examples_objective_batched(lengths, block, 1)
    }

    /// [`examples_objective`] with a chosen row count, for the record-level
    /// shard cases: one record is one example regardless of the row count.
    fn examples_objective_batched(lengths: &[usize], block: usize, batch_size: usize) -> CausalLm {
        let cfg = BatchConfig { batch_size, block_size: block, ..Default::default() };
        let (mut data, mut starts) = (Vec::new(), Vec::new());
        for (i, &len) in lengths.iter().enumerate() {
            starts.push(data.len());
            data.extend(std::iter::repeat_n(10 + i as u32, len));
        }
        let dataset = || TokenDataset::new_examples(data.clone(), vec![true; data.len()], &starts, 0, &cfg).unwrap();
        CausalLm::new(dataset(), dataset(), cfg.clone(), None)
    }

    /// A pass over the data takes every example once before any is taken
    /// again, and a run resumed part way through an epoch carries on the order
    /// a straight run follows.
    #[test]
    fn the_examples_are_drawn_in_epochs_without_replacement_and_a_resume_continues_the_order() {
        let rng_seed = data::rng::random_seed();
        let draws = |first: u64, count: u32| {
            let mut obj = examples_objective(&[4; 5], 4);
            Objective::<Recorder>::begin(&mut obj, 7, first);
            let model = Recorder::new(RecorderCfg, 1, 4, &HashMap::new());
            let mut rng = Rng::new(rng_seed);
            Objective::<Recorder>::micro_steps(&mut obj, &model, &mut rng, count);
            let seen = model.drawn.borrow().clone();
            seen
        };
        let straight = draws(0, 15);
        for epoch in straight.chunks(5) {
            let mut sorted = epoch.to_vec();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![10, 11, 12, 13, 14], "an epoch holds each example once: {straight:?}");
        }
        assert_ne!(straight[..5], straight[5..10], "each epoch is shuffled afresh");
        assert_eq!(draws(8, 7), straight[8..], "a resume at draw 8 follows the straight run");
    }

    /// The `k` batches of an optimiser step share one denominator, the mean
    /// count of their supervised positions, so a long record weighs what it
    /// holds and not as much as a short one.
    #[test]
    fn the_batches_of_a_step_are_divided_by_the_steps_mean_supervised_count() {
        // Examples of 3 and 7 tokens supervise 2 and 6 positions.
        let mut obj = examples_objective(&[3, 7], 8);
        Objective::<Recorder>::begin(&mut obj, 1, 0);
        let model = Recorder::new(RecorderCfg, 1, 8, &HashMap::new());
        let mut rng = Rng::new(data::rng::random_seed());
        Objective::<Recorder>::micro_steps(&mut obj, &model, &mut rng, 2);
        assert_eq!(*model.denominators.borrow(), vec![4.0, 4.0]);
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

    /// A chat/tool-call dataset (token mask + `<|endoftext|>` separators) must
    /// load one example per row. Packed into a shared window instead, short
    /// instruction-tuning examples see each other's answers: with 42-token
    /// examples and a 1024-token row, two dozen question/answer pairs share a
    /// row and half of them find their own answer already written out earlier
    /// in it. The model can then drive train AND held-out loss to near zero by
    /// copying, and both numbers stop predicting anything about answering one
    /// question with nothing else in context.
    #[test]
    fn a_masked_chat_dataset_loads_one_example_per_row() {
        let dir = tmp("chat-one-per-row");
        // Four 4-token examples - two prompt tokens, two supervised - in a
        // 64-id vocabulary, indexed out of band.
        let mut split = data::chat::EncodedSplit::default();
        for e in 0..4u32 {
            split.push(&[e * 10, e * 10 + 1, e * 10 + 2, e * 10 + 3], &[false, false, true, true]);
        }
        data::chat::write_split(&dir, "train", &split).unwrap();
        data::chat::write_split(&dir, "val", &split).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(64)).unwrap();

        // A row twice the length of an example: the old stream-window sampler
        // would have filled the rest of it with the following examples.
        let opts = FitOpts { block_size: 10, batch_size: 8, ..Default::default() };
        let (train, _val, cfg, _vocab) = load_dataset(&dir, &opts).expect("load");
        assert_eq!(train.example_count(), Some(4), "example boundaries must be recovered");

        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        for _ in 0..20 {
            let (x, y) = train.get_batch(&cfg, &mut rng);
            for b in 0..cfg.batch_size {
                let row = &x[b * 10..(b + 1) * 10];
                let base = row[0];
                assert_eq!(&row[..4], &[base, base + 1, base + 2, base + 3], "row {row:?} is not one example");
                // The example's last token predicts nothing, and past it the
                // row is padding, which is never a target.
                for t in 3..10 {
                    assert_eq!(y[b * 10 + t], data::loader::IGNORE, "row {row:?} supervises position {t}");
                }
            }
        }
    }

    /// A chat dataset written before examples were indexed closed each one
    /// with Qwen's `<|endoftext|>`; it still loads one example per row.
    #[test]
    fn a_legacy_separator_dataset_still_loads_one_example_per_row() {
        let dir = tmp("chat-legacy-separator");
        let sep = data::chat::LEGACY_EXAMPLE_SEPARATOR;
        let tokens: Vec<u32> = (0..3u32).flat_map(|e| [e * 10, e * 10 + 1, e * 10 + 2, sep]).collect();
        let mask: Vec<bool> = tokens.iter().map(|&t| t != sep).collect();
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
        binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
        binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta::vocab_only(151936)).unwrap();
        let opts = FitOpts { block_size: 8, batch_size: 2, ..Default::default() };
        let (train, ..) = load_dataset(&dir, &opts).expect("load");
        assert_eq!(train.example_count(), Some(3));
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

    #[test]
    fn a_published_commit_is_what_a_resume_verifies() {
        let dir = tmp("commit");
        let weights = dir.join("weights.bin");
        std::fs::write(&weights, b"weights").unwrap();
        let identity = CheckpointIdentity { job: "job", plan: "plan" };
        publish_commit(identity, &weights, 3, 2).unwrap();
        verify_resume(identity, &weights, 2).unwrap();

        // Another plan, another world, or altered bytes are all refused.
        assert!(verify_resume(CheckpointIdentity { job: "job", plan: "other" }, &weights, 2).is_err());
        assert!(verify_resume(identity, &weights, 4).is_err());
        std::fs::write(&weights, b"altered").unwrap();
        assert!(verify_resume(identity, &weights, 2).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_distributed_resume_must_name_its_world_in_the_identity() {
        let dir = tmp("world-identity");
        let state = dir.join("train.state");
        let coll = crate::HostCollective::new(1);
        let sync = GradSync { collective: &*coll, rank: 0, checkpoint: None };
        let opts = FitOpts { steps: 1, eval_interval: 0, ..Default::default() };
        let control = FitControl { state: Some(&state), identity: serde_json::json!({ "run": 1 }), ..Default::default() };
        let err = match fit_controlled_sharded(Recorder::new(RecorderCfg, 1, 1, &HashMap::new()), Verdicts(Default::default()), &opts, None, control, Some(sync), None) {
            Ok(_) => panic!("a distributed resume without a world size must be refused"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("world size"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
