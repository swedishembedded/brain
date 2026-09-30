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
    /// Low-rank adapters on the `targets` projections (see
    /// [`parse_lora_targets`]); base frozen.
    Lora { rank: u32, alpha: f32, targets: Vec<String> },
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
        let cfg = QwenConfig::from_json_checked(&c.header["config"]).map_err(std::io::Error::other)?;
        if let Mode::Lora { rank, alpha, targets } = mode {
            let lora = cfg
                .lora
                .as_ref()
                .unwrap_or_else(|| panic!("resume checkpoint {out} has no LoRA config, but mode is Lora {{ rank: {rank}, alpha: {alpha} }}"));
            assert_eq!(lora.rank, *rank, "resume checkpoint {out} LoRA rank {} does not match requested rank {rank}", lora.rank);
            assert_eq!(lora.alpha, *alpha, "resume checkpoint {out} LoRA alpha {} does not match requested alpha {alpha}", lora.alpha);
            assert_eq!(&lora.targets, targets, "resume checkpoint {out} adapts {:?}, not the requested {targets:?}", lora.targets);
        }
        (cfg, c.by_role(""))
    } else if let Mode::Lora { rank, alpha, targets } = mode {
        let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
        lora_start(base, *rank, *alpha, opts.seed, &LoraStart::FreshOn(&targets))?
    } else {
        // Fresh full fine-tune: base architecture + weights from the checkpoint.
        let c = checkpoint::load(base);
        let cfg = QwenConfig::from_json_checked(&c.header["config"]).map_err(std::io::Error::other)?;
        let mut init: HashMap<String, Vec<f32>> = crate::init_weights(&cfg, opts.seed);
        init.extend(c.by_role(""));
        (cfg, init)
    };

    let m = build_for_training(cfg, opts, &init);
    match prev_off {
        Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
        None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }
    let m = m?;
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen>(train, val, bcfg, itos);
    model::fit_with(m, obj, opts, Some(Path::new(out)))
}

/// The trainable model at `opts`' batch and context, placed through the
/// footprint check: a context the devices cannot hold is refused by name
/// before anything is allocated, rather than by an arbitrary length cap.
fn build_for_training(cfg: QwenConfig, opts: &FitOpts, init: &HashMap<String, Vec<f32>>) -> std::io::Result<Qwen> {
    let shard = crate::model::Shard::whole(cfg.n_layers as usize);
    let dt = gpu_core::select::Dtype::F32;
    crate::footprint::place_and_build(&cfg.clone(), &shard, dt, opts.batch_size, opts.block_size, true, false, "qwen3 finetune", || {
        Qwen::new(cfg, opts.batch_size, opts.block_size, init)
    })
    .map_err(std::io::Error::other)
}

/// Every projection a qwen3 LoRA adapter can cover, which is also what a
/// fresh adapter covers unless told otherwise: the attention and MLP
/// projections.
pub const LORA_TARGETS: [&str; 7] = ["wq", "wk", "wv", "wo", "gate", "up", "down"];

/// A LoRA fine-tune's schedule and optimiser beyond its length, rate,
/// batch, context and seed. The defaults are brain's LoRA recipe: weight
/// decay 0.1, clip 1.0, warmup over the first 5% of steps, cosine decay to a
/// tenth of the rate, torch's AdamW betas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoraHyper {
    pub weight_decay: f32,
    /// Global gradient-norm clip; 0 disables it.
    pub grad_clip: f32,
    /// Warmup steps; `None` for 5% of the run (at least one).
    pub warmup: Option<u32>,
    /// The rate the cosine decays to; `None` for a tenth of the peak.
    pub min_lr: Option<f32>,
    pub adam: model::Adam,
}

impl Default for LoraHyper {
    fn default() -> LoraHyper {
        LoraHyper { weight_decay: 0.1, grad_clip: 1.0, warmup: None, min_lr: None, adam: model::Adam::default() }
    }
}

impl LoraHyper {
    /// The run's [`FitOpts`]; evaluation, early stopping and checkpoint
    /// cadence are off, for the caller to set.
    pub fn fit_opts(&self, steps: u32, batch: u32, block: u32, lr: f32, seed: u64) -> FitOpts {
        FitOpts {
            steps,
            batch_size: batch,
            block_size: block,
            lr,
            min_lr: self.min_lr.unwrap_or(lr * 0.1),
            warmup: self.warmup.unwrap_or((steps / 20).max(1)),
            decay_iters: steps,
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            grad_accum: 1,
            eval_interval: 0,
            checkpoint_secs: 0,
            seed,
            adam: self.adam,
            ..FitOpts::default()
        }
    }
}

/// [`LORA_TARGETS`] as a `Mode::Lora` target list.
pub fn default_lora_targets() -> Vec<String> {
    LORA_TARGETS.iter().map(|t| t.to_string()).collect()
}

/// A comma-separated target list (`"wq,wv"`) as the projections it names.
/// A name that is not one of [`LORA_TARGETS`], a name given twice, or an
/// empty list is refused: an adapter silently missing a projection the
/// caller asked for trains something else than was asked.
pub fn parse_lora_targets(spec: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for t in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if !LORA_TARGETS.contains(&t) {
            return Err(format!("LoRA target {t:?} is not a projection; choose from {}", LORA_TARGETS.join(",")));
        }
        if out.iter().any(|o| o == t) {
            return Err(format!("LoRA target {t:?} is named twice"));
        }
        out.push(t.to_string());
    }
    if out.is_empty() {
        return Err(format!("no LoRA targets given; choose from {}", LORA_TARGETS.join(",")));
    }
    Ok(out)
}

/// Where a LoRA fine-tune's adapter starts.
#[derive(Clone, Copy, Debug)]
pub enum LoraStart<'a> {
    /// Zero-delta adapters over [`LORA_TARGETS`]: the model starts as the base.
    Fresh,
    /// Zero-delta adapters over these projections only.
    FreshOn(&'a [&'a str]),
    /// The adapter file at this path (`crate::lora::save_adapter`'s output),
    /// unfolded, so training continues ITS low-rank factors - the model
    /// starts as base plus that adapter.
    Continue(&'a str),
}

/// The configuration and initial weights of a LoRA fine-tune of `base`:
/// the base's own architecture and weights, a `rank`/`alpha` adapter over
/// [`LORA_TARGETS`], the chosen targets, or the continued adapter's own, and the
/// adapter factors either freshly initialised from `seed` (zero delta) or
/// read from the adapter being continued.
///
/// Continuing an adapter at a different rank or alpha than it was trained
/// at is refused: its factors only mean what they mean at their own shape
/// and scale.
pub fn lora_start(base: &str, rank: u32, alpha: f32, seed: u64, start: &LoraStart<'_>) -> std::io::Result<(QwenConfig, HashMap<String, Vec<f32>>)> {
    let invalid = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidData, why);
    let (targets, adapter) = match start {
        LoraStart::Fresh => (LORA_TARGETS.iter().map(|s| s.to_string()).collect::<Vec<_>>(), None),
        LoraStart::FreshOn(targets) => (targets.iter().map(|s| s.to_string()).collect(), None),
        LoraStart::Continue(path) => {
            let st = checkpoint::st::load_safetensors(path)?;
            let card = st.card().and_then(|c| c.adapter).ok_or_else(|| invalid(format!("{path}: not an adapter file (no adapter card)")))?;
            if card.kind != "lora" {
                return Err(invalid(format!("{path}: a {:?} adapter, only LoRA can be continued", card.kind)));
            }
            if card.rank != Some(rank) || card.alpha.is_some_and(|a| a != alpha) {
                return Err(invalid(format!("{path}: trained at rank {:?} alpha {:?}, asked to continue at rank {rank} alpha {alpha}", card.rank, card.alpha)));
            }
            let targets = card.targets.ok_or_else(|| invalid(format!("{path}: the adapter card names no targets")))?;
            (targets, Some(st.tensors))
        }
    };
    let c = checkpoint::load(base);
    let mut cfg = QwenConfig::from_json_checked(&c.header["config"]).map_err(std::io::Error::other)?;
    cfg.lora = Some(LoraCfg { rank, alpha, targets });
    // Fresh init for the LoRA-extended param set, then the base's own
    // weights over it; a fresh adapter stays at its zero-delta init.
    let mut init: HashMap<String, Vec<f32>> = crate::init_weights(&cfg, seed);
    init.extend(c.by_role(""));
    if let Some(tensors) = adapter {
        for (name, values) in tensors {
            match init.get(&name) {
                Some(slot) if slot.len() == values.len() => {
                    init.insert(name, values);
                }
                Some(slot) => return Err(invalid(format!("adapter tensor {name} holds {} values, this base's has {}", values.len(), slot.len()))),
                None => return Err(invalid(format!("adapter tensor {name} has no counterpart in this base's LoRA parameters"))),
            }
        }
    }
    Ok((cfg, init))
}

/// A LoRA fine-tune of `base` on the masked token dataset in `dir`, with the
/// caller in the loop (progress, stopping, exact resume - see
/// `model::FitControl`), returning the trained model rather than writing a
/// whole checkpoint: the caller saves what it wants of it (its adapter).
pub fn finetune_lora_controlled(
    base: &str,
    dir: &Path,
    opts: &FitOpts,
    rank: u32,
    alpha: f32,
    start: &LoraStart<'_>,
    control: model::FitControl<'_>,
) -> std::io::Result<(model::FitReport, Qwen)> {
    let (cfg, init) = lora_start(base, rank, alpha, opts.seed, start)?;
    // A LoRA build never offloads its moments; the process-wide switch is
    // cleared for the construction and restored after, as `finetune_from`
    // does.
    let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
    std::env::remove_var("BRAIN_OFFLOAD_ADAM");
    let m = build_for_training(cfg, opts, &init);
    if let Some(v) = prev_off {
        std::env::set_var("BRAIN_OFFLOAD_ADAM", v);
    }
    let m = m?;
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen>(train, val, bcfg, itos);
    model::fit_controlled(m, obj, opts, None, control)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults are the recipe every LoRA entry point used to spell out
    /// for itself; every knob a caller sets reaches the run.
    #[test]
    fn a_lora_schedule_defaults_to_the_recipe_and_takes_every_override() {
        let o = LoraHyper::default().fit_opts(400, 4, 256, 1e-4, 9);
        assert_eq!((o.steps, o.batch_size, o.block_size, o.seed, o.decay_iters), (400, 4, 256, 9, 400));
        assert_eq!((o.lr, o.min_lr, o.warmup, o.weight_decay, o.grad_clip), (1e-4, 1e-5, 20, 0.1, 1.0));
        assert_eq!(o.adam, model::Adam::default());
        assert_eq!(LoraHyper::default().fit_opts(10, 1, 8, 1e-4, 0).warmup, 1, "at least one warmup step");

        let adam = model::Adam { beta1: 0.8, beta2: 0.95, eps: 1e-6 };
        let h = LoraHyper { weight_decay: 0.0, grad_clip: 0.0, warmup: Some(0), min_lr: Some(0.0), adam };
        let o = h.fit_opts(400, 4, 256, 1e-4, 9);
        assert_eq!((o.min_lr, o.warmup, o.weight_decay, o.grad_clip, o.adam), (0.0, 0, 0.0, 0.0, adam));
    }

    #[test]
    fn lora_targets_are_the_projections_named_and_nothing_else() {
        assert_eq!(parse_lora_targets("wq, wv,down").unwrap(), ["wq", "wv", "down"]);
        let e = parse_lora_targets("wq,embed").unwrap_err();
        assert!(e.contains("embed") && e.contains("wq"), "names the stranger and what is allowed: {e}");
        assert!(parse_lora_targets("wq,wq").unwrap_err().contains("twice"));
        assert!(parse_lora_targets(" , ").is_err(), "an adapter on nothing is not a fine-tune");
    }

    #[test]
    fn a_fresh_adapter_covers_exactly_the_chosen_targets() {
        let cfg = QwenConfig::tiny();
        let init = crate::init_weights(&cfg, 3);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(n, len)| (n.clone(), vec![len as u64], init[&n].clone())).collect();
        let dir = std::env::temp_dir().join(format!("qwen3-lora-targets-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.safetensors");
        checkpoint::save(base.to_str().unwrap(), cfg.to_json(), &tensors);

        let (cfg, init) = lora_start(base.to_str().unwrap(), 2, 4.0, 1, &LoraStart::FreshOn(&["wq", "up"])).unwrap();
        assert_eq!(cfg.lora.as_ref().unwrap().targets, ["wq", "up"]);
        let adapted: std::collections::BTreeSet<&str> = init.keys().filter_map(|k| k.strip_suffix(".lora_a")).filter_map(|k| k.rsplit('.').nth(1)).collect();
        assert_eq!(adapted.into_iter().collect::<Vec<_>>(), ["up", "wq"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
