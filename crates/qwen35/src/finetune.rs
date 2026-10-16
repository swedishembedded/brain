// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B fine-tuning: full (with optimizer offload) and LoRA, over
//! brain's masked token datasets. A self-contained training loop so both
//! modes seed correctly from a base checkpoint - full merges the checkpoint
//! weights as-is; a fresh LoRA start merges them and adds freshly-initialised
//! zero-delta adapters - which `model::fit`'s resume path (checkpoint-config-wins)
//! cannot do. Mirrors `qwen3::finetune` exactly (a genuinely new surface for
//! this family - qwen35moe has none), including [`finetune_from`]'s `resume`
//! switch: it loads architecture and weights from the checkpoint being
//! continued, not the base, so a later cycle continues the SAME adapter.

use std::collections::HashMap;
use std::path::Path;

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::FitOpts;

use crate::base_source::{lora_base, BaseDtype};
use crate::config::{lora_targets, LoraCfg, Qwen35Config};
use crate::model::{pipelines, Qwen35, TrainOpts};

/// Which fine-tuning scheme.
#[derive(Clone, Debug)]
pub enum Mode {
    /// Every weight trainable; AdamW moments offloaded to system RAM (Role::Offload).
    FullOffload,
    /// Low-rank adapters on the 12 targetable projections (`crate::config::
    /// lora_targets`); base frozen.
    Lora { rank: u32, alpha: f32 },
}

/// Fine-tune `base` on the masked dataset in `dir`, writing `out`. Returns
/// `(initial_loss, final_loss)`. A fresh (non-resuming) start - see
/// [`finetune_from`].
pub fn finetune(base: &str, dir: &Path, opts: &FitOpts, mode: &Mode, out: &str) -> std::io::Result<(f32, f32)> {
    finetune_from(base, dir, opts, mode, out, false)
}

/// [`finetune`], with an explicit `resume` switch: when `resume` is true AND
/// `out` already exists, architecture and weights are loaded from `out`
/// itself (not `base`), and training continues the adapter (or full model)
/// already there instead of overlaying a fresh zero-delta LoRA init on top of
/// the base every cycle - mirrors `qwen3::finetune::finetune_from` exactly.
/// `resume` with `out` missing (the very first cycle) falls back to the
/// fresh-start path unchanged.
pub fn finetune_from(base: &str, dir: &Path, opts: &FitOpts, mode: &Mode, out: &str, resume: bool) -> std::io::Result<(f32, f32)> {
    // BRAIN_OFFLOAD_ADAM is a process-global switch (the same convention
    // `model::parallel`/`model::shard` use) -- save/restore the caller's prior
    // value rather than clobbering it, so a nested or later call in the same
    // process doesn't silently inherit this call's mode. This has to be set
    // before `Qwen35::new_train_on` runs regardless of resume-vs-fresh: it
    // governs Role::Offload vs Role::Trainable for a `FullOffload` build.
    let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
    match mode {
        Mode::FullOffload => std::env::set_var("BRAIN_OFFLOAD_ADAM", "1"),
        Mode::Lora { .. } => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }

    let (cfg, init) = if resume && Path::new(out).exists() {
        // Resume: architecture + weights come from the checkpoint being
        // continued, not `base` - the adapter's (or full model's)
        // accumulated state must survive. Skipping the fresh-init overlay
        // below is exactly what lets `lora_b` keep the delta it learned in
        // earlier cycles instead of being reset back to its zero-delta init.
        let c = checkpoint::load(out);
        let cfg = Qwen35Config::from_json(&c.header["config"]);
        if let Mode::Lora { rank, alpha } = mode {
            let lora = cfg
                .lora
                .as_ref()
                .unwrap_or_else(|| panic!("resume checkpoint {out} has no LoRA config, but mode is Lora {{ rank: {rank}, alpha: {alpha} }}"));
            assert_eq!(lora.rank, *rank, "resume checkpoint {out} LoRA rank {} does not match requested rank {rank}", lora.rank);
            assert_eq!(lora.alpha, *alpha, "resume checkpoint {out} LoRA alpha {} does not match requested alpha {alpha}", lora.alpha);
        }
        (cfg, c.by_role(""))
    } else {
        // Fresh start: base architecture + weights from the checkpoint.
        let c = checkpoint::load(base);
        let mut cfg = Qwen35Config::from_json(&c.header["config"]);
        let base_w = c.by_role("");
        if let Mode::Lora { rank, alpha } = mode {
            cfg.lora = Some(LoraCfg { rank: *rank, alpha: *alpha, targets: lora_targets() });
        }
        // Fresh init for the (possibly LoRA-extended) param set, then overwrite
        // the base params with the checkpoint's -- adapters stay at their
        // zero-delta init.
        let mut init: HashMap<String, Vec<f32>> = crate::init::init_weights(&cfg, opts.seed);
        for (k, v) in base_w {
            init.insert(k, v);
        }
        (cfg, init)
    };

    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, &init);
    match prev_off {
        Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
        None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen35>(train, val, bcfg, itos);
    model::fit_with(m, obj, opts, Some(Path::new(out)))
}

/// Micro-step size, in tokens, above which a streamed-base run switches to the
/// long-sequence settings (measured to fit a 27B step: knowledge 233).
const LONG_SEQ_TOKENS: u32 = 2048;
/// Tokens per micro-step a batch wider than one may hold.
const LONG_STEP_TOKENS: u32 = 8192;
/// Gated-DeltaNet segment length of the long-sequence settings.
const LONG_GDN_SEGMENT: u32 = 128;
/// MLP token-chunk size of the long-sequence settings.
const LONG_MLP_ROWS: u32 = 512;

/// LoRA fine-tune on a frozen base that is STREAMED from `base` (a
/// HuggingFace checkpoint directory, FP8 release included, or a brain-native
/// file) and held on the device at `dtype`, so the base never exists on the
/// host as a whole. `out` is an ADAPTER file (`crate::lora::save_adapter`),
/// never a copy of the base; with `resume` and an existing `out` the run
/// continues that adapter's factors instead of starting from zero delta.
/// `recompute` rebuilds each layer's activations in the backward instead of
/// keeping them ([`TrainOpts::recompute`]).
///
/// Only LoRA trains on a streamed base: a full-parameter run needs the base
/// as fp32 optimiser state on the device, which is what the resident
/// [`finetune`] path is for.
#[allow(clippy::too_many_arguments)]
pub fn finetune_streamed(base: &Path, dir: &Path, opts: &FitOpts, mode: &Mode, out: &str, resume: bool, dtype: BaseDtype, recompute: bool) -> std::io::Result<(f32, f32)> {
    let unsupported = |why: String| std::io::Error::new(std::io::ErrorKind::Unsupported, why);
    let Mode::Lora { rank, alpha } = mode else {
        return Err(unsupported(format!("a {} base is frozen: only LoRA trains on it", dtype.name())));
    };
    // One micro-step of more than a few thousand tokens only fits the card with
    // the long-sequence settings (segmented Gated-DeltaNet layers, a bf16
    // residual stream, an MLP run in token chunks), which need a batch of one:
    // refuse a wide batch up front and name the way to get the same effective
    // batch (`--batch 1 --grad-accum G`) instead of failing deep in the run.
    let tokens = opts.batch_size as u64 * opts.block_size as u64;
    if recompute && opts.batch_size > 1 && tokens > LONG_STEP_TOKENS as u64 {
        return Err(unsupported(format!(
            "batch {} x seq-len {} is {tokens} tokens per micro-step; a {} base trains up to {LONG_STEP_TOKENS} tokens per micro-step in a batch wider than one. Use --batch 1 with --grad-accum {} for the same effective batch",
            opts.batch_size,
            opts.block_size,
            dtype.name(),
            opts.batch_size * opts.grad_accum.max(1),
        )));
    }
    let (cfg, src) = lora_base(base, *rank, *alpha, opts.seed, resume.then(|| Path::new(out)).filter(|p| p.exists())).map_err(unsupported)?;
    let tier = match dtype {
        BaseDtype::F32 => Dtype::F32,
        BaseDtype::Bf16 => Dtype::BF16,
        BaseDtype::Nf4 => Dtype::NF4,
    };
    let long = recompute && opts.batch_size == 1 && opts.block_size >= LONG_SEQ_TOKENS;
    let train = TrainOpts {
        base: TierPolicy::uniform(tier),
        recompute,
        gdn_segment_tokens: long.then_some(LONG_GDN_SEGMENT),
        bf16_residuals: long && !cfg.mtp,
        mlp_rows: long.then_some(LONG_MLP_ROWS),
        ..TrainOpts::default()
    };
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, &src, &train);
    let (train_set, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen35>(train_set, val, bcfg, itos);
    let (report, m) = model::fit_controlled(m, obj, opts, None, model::FitControl::default())?;
    let base_id = base.file_name().map_or_else(|| "base".to_string(), |n| n.to_string_lossy().into_owned());
    crate::lora::save_adapter(out, &m, "brain/qwen35-lora", &base_id, None)?;
    Ok((report.initial_loss, report.final_loss.unwrap_or(report.initial_loss)))
}
