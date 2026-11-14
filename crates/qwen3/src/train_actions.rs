// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two full-parameter trainers of `brain/qwen3`: `train` (a small decoder
//! from scratch) and `finetune` (a pulled checkpoint continued into a new
//! one), both over a tokenized dataset directory (`brain data` prepares one).
//!
//! Each reports every step and stops between steps when its invocation is
//! cancelled, writing the checkpoint with the steps that ran. A checkpoint
//! either writes carries the `qwen` card, so the model store resolves it for
//! `generate`, `eval` and `finetune` like any other.
//!
//! Swedish Embedded AB implements training pipelines for language models on
//! its clients' own hardware. If your team needs expertise in pretraining and
//! fine-tuning decoders on a single machine then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::Path;

use capability::{ActionResult, ActionSpec, Invocation, Outcome, ParamSpec, ParamType, Progress};
use serde_json::json;

use crate::config::QwenConfig;
use crate::model::Qwen;
use crate::served::{count, path_param, Resolved};

/// What one trainer runs with when the caller names nothing.
struct Defaults {
    steps: u32,
    batch: u32,
    block: u32,
    lr: f32,
}

const TRAIN: Defaults = Defaults { steps: 2000, batch: 8, block: 256, lr: 3e-4 };
const FINETUNE: Defaults = Defaults { steps: 2000, batch: 8, block: 256, lr: 1e-5 };

/// Seconds between the checkpoints a run keeps while it trains.
const CHECKPOINT_SECS: u32 = 600;
/// Most sequences one step may average over.
const MAX_BATCH: f64 = 1_024.0;
/// Largest model dimension (width, feed-forward, head size, layer count) a
/// from-scratch run may ask for.
const MAX_DIM: f64 = 65_536.0;

fn shared_params(spec: ActionSpec, d: &Defaults) -> ActionSpec {
    spec.training()
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "the tokenized dataset directory (`brain data` prepares one)").host_resolved().required())
        .param(ParamSpec::new("save", ParamType::Str, "where to write the checkpoint (.safetensors)").host_resolved().required())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(d.steps)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("batch", ParamType::Int, "sequences averaged into each step").default(json!(d.batch)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("block", ParamType::Int, "tokens per sequence").default(json!(d.block)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate (cosine schedule down to a tenth of it)").default(json!(d.lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("mask", ParamType::Str, "train only on what follows this character on each line (the answer after '=' of an LHS=RHS dataset)"))
        .param(ParamSpec::new("align", ParamType::Bool, "start every sequence at a line boundary").default(json!(false)))
        .param(ParamSpec::new("checkpoint_secs", ParamType::Int, "seconds between the checkpoints kept while training; 0 writes only the last").default(json!(CHECKPOINT_SECS)).min(0.0).max(86_400.0))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the initialization and the batches (omit for random; the outcome echoes the one used)").min(0.0))
        .param(ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward (default: the model's own)",
        ))
}

pub(crate) fn train_spec() -> ActionSpec {
    let shape = |name: &str, help: &str, default: u32| ParamSpec::new(name, ParamType::Int, help).default(json!(default)).min(1.0).max(MAX_DIM);
    shared_params(ActionSpec::new("train", "train a small Qwen3 decoder from scratch on a tokenized dataset (train.bin, val.bin, meta.json)").cancellable().from_scratch(), &TRAIN)
        .param(shape("layers", "transformer layers", 6))
        .param(shape("d_model", "model width", 384))
        .param(shape("heads", "attention heads", 6))
        .param(shape("kv_heads", "key/value heads (grouped-query attention)", 2))
        .param(shape("head_dim", "width of one attention head", 64))
        .param(shape("d_ff", "feed-forward width", 1024))
}

pub(crate) fn finetune_spec() -> ActionSpec {
    shared_params(ActionSpec::new("finetune", "continue training the resolved checkpoint (a brain checkpoint, a GGUF or a Hugging Face release) on a tokenized dataset, every weight, into a new checkpoint").cancellable(), &FINETUNE)
        .param(ParamSpec::new("weights", ParamType::Str, "the base checkpoint (.safetensors, .gguf or a Hugging Face release); overrides the model store's own pick when set").host_resolved())
        .param(ParamSpec::new(
            "train_precision",
            ParamType::Str,
            "how the run stores what it trains, as comma-separated key=value: state=f32|bf16|int8[:B]|log8[:B], master=none|bf16|f32, grad=f32|bf16, sr=on|off, narrow=on|off, weights=bf16|fp8[:B]|int8|nf4, window=N, clip=two-pass|running (default: all fp32)",
        ))
}

/// The run's optimizer settings: the warmup is 50 steps unless the run is
/// shorter than twice that, since a warmup longer than the run never reaches
/// the peak rate.
fn fit_opts(inv: &Invocation, d: &Defaults, seed: u64, precision: model::TrainPrecision, action: &str) -> Result<model::FitOpts, String> {
    let steps = count(inv, "steps", d.steps).max(1);
    let lr = inv.get_f64("lr").map_or(d.lr, |v| v as f32);
    let mask = match inv.get_str("mask").filter(|m| !m.is_empty()) {
        Some(m) => {
            let mut chars = m.chars();
            Some(chars.next().filter(|_| chars.next().is_none()).ok_or_else(|| format!("qwen {action}: mask {m:?} is not one character"))?)
        }
        None => None,
    };
    Ok(model::FitOpts {
        steps,
        batch_size: count(inv, "batch", d.batch).max(1),
        block_size: count(inv, "block", d.block).max(1),
        lr,
        warmup: 50.min((steps / 2).max(1)),
        decay_iters: steps,
        min_lr: lr * 0.1,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: (steps / 10).max(1),
        eval_batches: 20,
        checkpoint_secs: u64::from(count(inv, "checkpoint_secs", CHECKPOINT_SECS)),
        mask_before: mask,
        mask_per_line: mask.is_some(),
        align_to_lines: inv.get_bool("align").unwrap_or(false),
        seed,
        precision,
        ..Default::default()
    })
}

fn activations(inv: &Invocation, action: &str) -> Result<Option<model::saved::SavedPolicy>, String> {
    inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a).map_err(|e| format!("qwen {action}: {e}"))).transpose()
}

/// The step observer of a training action: reports each step and says to
/// continue until the invocation is cancelled.
fn watch<'a>(inv: &'a Invocation, progress: &'a mut dyn FnMut(Progress)) -> impl FnMut(&model::StepReport) -> bool + 'a {
    move |s| {
        progress(Progress::step(s.step, s.steps, format!("loss {:.4}", s.loss)));
        !inv.cancel.is_cancelled()
    }
}

/// A stopped run answers the contract's cancelled error; how far it got (and
/// that `save` holds those steps) travels as its last progress event.
fn outcome(report: &model::FitReport, seed: u64, save: &Path, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    if report.interrupted {
        progress(Progress::step(report.steps_completed, report.steps_completed, format!("cancelled after step {} (the checkpoint at {} holds the steps that ran)", report.steps_completed, save.display())));
        return Err(capability::failure::CANCELLED.to_string());
    }
    Ok(Outcome::new()
        .set("initial_loss", json!(report.initial_loss))
        .set("final_loss", json!(report.final_loss))
        .set("steps", json!(report.steps_completed))
        .set("seed", json!(seed))
        .set("saved", json!(save.display().to_string())))
}

pub(crate) fn train(_: &Resolved, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let dir = path_param(inv, "data", "train")?;
    let save = path_param(inv, "save", "train")?;
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let precision = model::TrainPrecision { activations: activations(inv, "train")?, ..Default::default() };
    let opts = fit_opts(inv, &TRAIN, seed, precision, "train")?;
    let dim = |name: &str, default: u32| count(inv, name, default).max(1);
    let cfg = QwenConfig {
        vocab: 0, // filled from the dataset
        block_size: opts.block_size,
        n_layers: dim("layers", 6),
        d_model: dim("d_model", 384),
        n_heads: dim("heads", 6),
        n_kv_heads: dim("kv_heads", 2),
        head_dim: dim("head_dim", 64),
        d_ff: dim("d_ff", 1024),
        rope_theta: 1.0e6,
        rms_eps: 1e-6,
        max_position_embeddings: opts.block_size,
        tie_embeddings: true,
        qk_norm: true,
        attn_bias: false,
        lora: None,
        rope_scaling: None,
    };
    if !cfg.n_heads.is_multiple_of(cfg.n_kv_heads) {
        return Err(format!("qwen train: heads ({}) must be a multiple of kv_heads ({})", cfg.n_heads, cfg.n_kv_heads));
    }
    let report = model::train::fit_watched::<Qwen>(&dir, cfg, &opts, Some(&save), &mut watch(inv, progress)).map_err(|e| format!("qwen train: {e}"))?;
    outcome(&report, seed, &save, progress)
}

pub(crate) fn finetune(resolved: &Resolved, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let weights = resolved.weights(inv, "finetune")?;
    let dir = path_param(inv, "data", "finetune")?;
    let save = path_param(inv, "save", "finetune")?;
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let mut precision = match inv.get_str("train_precision").filter(|p| !p.is_empty()) {
        Some(spec) => model::TrainPrecision::parse(&spec).map_err(|e| format!("qwen finetune: train_precision {spec:?}: {e}"))?,
        None => model::TrainPrecision::default(),
    };
    match (activations(inv, "finetune")?, precision.activations) {
        (Some(named), Some(inside)) if named != inside => {
            return Err(format!("qwen finetune: activations {named:?} conflicts with activations={inside:?} in train_precision; name the choice once"));
        }
        (Some(named), _) => precision.activations = Some(named),
        (None, _) => {}
    }
    let opts = fit_opts(inv, &FINETUNE, seed, precision, "finetune")?;
    // The trained model starts from the base as it is on disk, and continues
    // `save` when that already exists.
    let resume = save.exists();
    let report = crate::finetune::finetune_from_watched(&weights, &dir, &opts, &crate::finetune::Mode::FullOffload, &save.to_string_lossy(), resume, &mut watch(inv, progress)).map_err(|e| format!("qwen finetune: {e}"))?;
    outcome(&report, seed, &save, progress).map(|o| o.set("resumed", json!(resume)))
}
