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

use gpu_core::Gpu;
use model::FitOpts;

use crate::base_source::{lora_base, BaseDtype};
use crate::config::{lora_targets, LoraCfg, Qwen35Config};
use crate::model::{pipelines, Qwen35};

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

/// LoRA fine-tune on a frozen base that is STREAMED from `base` (a
/// HuggingFace checkpoint directory, FP8 release included, or a brain-native
/// file) and held on the device at `dtype`, so the base never exists on the
/// host as a whole. `out` is an ADAPTER file (`crate::lora::save_adapter`),
/// never a copy of the base; with `resume` and an existing `out` the run
/// continues that adapter's factors instead of starting from zero delta. The base-opening half is complete
/// ([`crate::base_source::lora_base`]: streamed base tensors plus fresh
/// zero-delta adapters, one `TensorSource`); the trainer that consumes it
/// takes its tier and recompute switch from `crate::model`, which this entry
/// point calls once that constructor exists. Until then it opens and
/// validates the base, then reports what is missing instead of training the
/// wrong thing.
pub fn finetune_streamed(base: &Path, _dir: &Path, opts: &FitOpts, mode: &Mode, out: &str, resume: bool, dtype: BaseDtype, _recompute: bool) -> std::io::Result<(f32, f32)> {
    let unsupported = |why: String| std::io::Error::new(std::io::ErrorKind::Unsupported, why);
    let Mode::Lora { rank, alpha } = mode else {
        return Err(unsupported(format!("a {} base is frozen: only LoRA trains on it", dtype.name())));
    };
    let (cfg, _src) = lora_base(base, *rank, *alpha, opts.seed, resume.then(|| Path::new(out)).filter(|p| p.exists())).map_err(unsupported)?;
    Err(unsupported(format!(
        "the {} base opened ({} layers, d_model {}), but this build's trainer has no {} base tier yet",
        base.display(),
        cfg.n_layers,
        cfg.d_model,
        dtype.name()
    )))
}
