// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen fine-tuning: full (with optimizer offload) and LoRA, over brain's masked
//! token datasets (`data::chat` / tool-call). A self-contained training loop so
//! both modes seed correctly from a base checkpoint - full merges the checkpoint
//! weights as-is; a fresh LoRA start merges them and adds freshly-initialised
//! zero-delta adapters - which `model::fit`'s resume path (checkpoint-config-wins)
//! cannot do. [`finetune_from`]'s `resume` switch is what lets a later cycle
//! continue the SAME adapter (or full model) instead of overlaying a fresh
//! zero-delta init on top of the base every time: it loads architecture and
//! weights from the checkpoint being continued, not from the base.

use std::collections::HashMap;
use std::path::Path;

use model::FitOpts;

use crate::config::{LoraCfg, QwenConfig};
use crate::model::Qwen;

/// Which fine-tuning scheme.
#[derive(Clone, Debug)]
pub enum Mode {
    /// Every weight trainable; AdamW moments offloaded to system RAM (Role::Offload).
    FullOffload,
    /// Low-rank adapters on the attention+MLP projections; base frozen.
    Lora { rank: u32, alpha: f32 },
}

/// Fine-tune `base` on the masked dataset in `dir`, writing `out`. Returns
/// `(initial_loss, final_loss)`. A fresh (non-resuming) start - see
/// [`finetune_from`].
pub fn finetune(
    base: &str,
    dir: &Path,
    opts: &FitOpts,
    mode: &Mode,
    out: &str,
) -> std::io::Result<(f32, f32)> {
    finetune_from(base, dir, opts, mode, out, false)
}

/// [`finetune`], with an explicit `resume` switch: when `resume` is true AND
/// `out` already exists, architecture and weights are loaded from `out`
/// itself (not `base`), and training continues the adapter (or full model)
/// already there instead of overlaying a fresh zero-delta LoRA init on top of
/// the base every cycle - the defect that made a LoRA adapter unable to be
/// incrementally continued across cycles. `resume` with `out` missing (the
/// very first cycle) falls back to the fresh-start path unchanged.
pub fn finetune_from(
    base: &str,
    dir: &Path,
    opts: &FitOpts,
    mode: &Mode,
    out: &str,
    resume: bool,
) -> std::io::Result<(f32, f32)> {
    // BRAIN_OFFLOAD_ADAM is a process-global switch (the same convention
    // `model::parallel`/`model::shard` use) -- save/restore the caller's prior
    // value rather than clobbering it, so a nested or later call in the same
    // process doesn't silently inherit this call's mode. This has to be set
    // before `Qwen::new` runs regardless of resume-vs-fresh: it governs
    // Role::Offload vs Role::Trainable for a `FullOffload` build, which
    // `new_impl` reads at construction time no matter which checkpoint
    // supplied the weights.
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
        let cfg = QwenConfig::from_json(&c.header["config"]);
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
        let mut cfg = QwenConfig::from_json(&c.header["config"]);
        let base_w = c.by_role("");
        if let Mode::Lora { rank, alpha } = mode {
            cfg.lora = Some(LoraCfg {
                rank: *rank,
                alpha: *alpha,
                targets: ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect(),
            });
        }
        // Fresh init for the (possibly LoRA-extended) param set, then overwrite
        // the base params with the checkpoint's - adapters stay at their
        // zero-delta init.
        let mut init: HashMap<String, Vec<f32>> = crate::init_weights(&cfg, opts.seed);
        for (k, v) in base_w {
            init.insert(k, v);
        }
        (cfg, init)
    };

    let m = Qwen::new(cfg, opts.batch_size, opts.block_size, &init);
    match prev_off {
        Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
        None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen>(train, val, bcfg, itos);
    model::fit_with(m, obj, opts, Some(Path::new(out)))
}
