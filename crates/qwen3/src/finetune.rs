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

use checkpoint::TensorSource;
use gpu_core::devices::{Home, Need};
use gpu_core::select::Dtype;
use model::{FitOpts, Pipeline, PipelineModel, Shard, Shardable};

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

    let (cfg, init): (QwenConfig, Box<dyn TensorSource>) = if resume && Path::new(out).exists() {
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
        (cfg, Box::new(c.by_role("")))
    } else if let Mode::Lora { rank, alpha, targets } = mode {
        let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
        let (cfg, init) = lora_start(base, *rank, *alpha, opts.seed, &LoraStart::FreshOn(&targets))?;
        (cfg, Box::new(init))
    } else {
        // Fresh full fine-tune: the base's architecture and weights, read as
        // they are on disk (a brain checkpoint, a GGUF or a `transformers`
        // directory) and streamed into the training build.
        crate::open_checkpoint(base).map_err(std::io::Error::other)?
    };

    let m = build_for_training(cfg, opts, &*init, Dtype::F32);
    match prev_off {
        Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
        None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }
    let m = m?;
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen>(train, val, bcfg, itos);
    model::fit_with(m, obj, opts, Some(Path::new(out)))
}

/// The trainable model at `opts`' batch and context, its frozen base
/// linears at `dt` (anything but fp32 needs a LoRA configuration), placed
/// through the footprint check: a context the devices cannot hold is refused
/// by name before anything is allocated, rather than by an arbitrary length
/// cap.
pub fn build_for_training(cfg: QwenConfig, opts: &FitOpts, init: &dyn TensorSource, dt: Dtype) -> std::io::Result<Qwen> {
    let shard = crate::model::Shard::whole(cfg.n_layers as usize);
    crate::footprint::place_and_build(&cfg.clone(), &shard.clone(), dt, opts.batch_size, opts.block_size, true, false, "qwen3 finetune", || {
        if dt == Dtype::F32 {
            Qwen::new_shard(cfg, opts.batch_size, opts.block_size, init, true, shard)
        } else {
            Qwen::new_lora_dt(cfg, opts.batch_size, opts.block_size, init, dt)
        }
    })
    .map_err(std::io::Error::other)
}

/// How a LoRA fine-tune is laid out over `cards` cards, by what they can
/// hold and never by a flag: the whole model as one shard when `place`
/// finds a card for it, else the fewest pipeline stages `place` can home
/// (each stage declared with its own footprint), else a refusal naming the
/// bytes. `place` answers a card for each part or refuses the plan - the
/// machine's placer. The returned shards carry the card each was homed on.
pub fn plan_lora_layout(
    cfg: &QwenConfig,
    b: u32,
    t: u32,
    dt: Dtype,
    cards: usize,
    place: impl Fn(&[Need]) -> Result<Vec<Home>, String>,
) -> Result<Vec<Shard>, String> {
    let cost = <Qwen as Shardable>::shard_cost(cfg, b, t);
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    let mut refused = Vec::new();
    for stages in 1..=cards.max(1) {
        let mut shards = model::plan_balanced(&cost, &(0..stages).collect::<Vec<_>>());
        let needs: Vec<Need> = shards
            .iter()
            .enumerate()
            .map(|(i, sh)| {
                let name = if stages == 1 { "qwen3 finetune".to_string() } else { format!("qwen3 finetune stage {i}") };
                Need::sized(name, crate::footprint::estimate_vram_bytes(cfg, sh, dt, b, t, true, false), 0)
            })
            .collect();
        match place(&needs) {
            Ok(homes) => {
                for (sh, home) in shards.iter_mut().zip(homes) {
                    match home {
                        Home::Gpu(card) => sh.gpu_index = card as usize,
                        // The ambient device: the CPU backend the caller selected.
                        Home::Cpu => sh.gpu_index = Shard::ANY_GPU,
                    }
                }
                return Ok(shards);
            }
            Err(why) => {
                let largest = needs.iter().map(|n| n.vram).max().unwrap_or(0);
                refused.push(format!("{stages} stage(s) of up to {:.1} GiB: {why}", gib(largest)));
            }
        }
    }
    Err(format!("qwen3 finetune does not fit {cards} card(s): {}", refused.join("; ")))
}

/// A trained LoRA model: on one card, or split across several as a pipeline.
pub enum Trained {
    Single(Qwen),
    Pipeline(PipelineModel<Qwen>),
}

impl Trained {
    /// Write the trained adapter (never the frozen base) to `path`, as
    /// [`crate::lora::save_adapter_with_lineage`].
    pub fn save_adapter_with_lineage(
        &self,
        path: &str,
        card_id: &str,
        base_id: &str,
        dataset_id: Option<&str>,
        training: Option<checkpoint::st::TrainingProvenance>,
    ) -> std::io::Result<()> {
        match self {
            Trained::Single(m) => crate::lora::save_adapter_with_lineage(path, m, card_id, base_id, dataset_id, training),
            Trained::Pipeline(m) => crate::lora::save_adapter_with_lineage(path, m, card_id, base_id, dataset_id, training),
        }
    }

    /// [`Self::save_adapter_with_lineage`] with no training provenance.
    pub fn save_adapter(&self, path: &str, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
        self.save_adapter_with_lineage(path, card_id, base_id, dataset_id, None)
    }
}

/// The trainable model laid out over the machine's cards ([`plan_lora_layout`]):
/// built as it would be on one card, or as a pipeline of stages each on its
/// own. A pinned device (an explicit `--device`) is one card, as it always
/// was. The layout is printed, so an automatic split is never a silent one.
fn build_trainer(cfg: QwenConfig, opts: &FitOpts, init: &dyn TensorSource, dt: Dtype) -> std::io::Result<Trained> {
    if gpu_core::devices::current_gpu().is_some() {
        return build_for_training(cfg, opts, init, dt).map(Trained::Single);
    }
    let place = |needs: &[Need]| gpu_core::devices::place(needs).map(|homes| homes.parts().iter().map(|(_, home)| *home).collect());
    let shards = plan_lora_layout(&cfg, opts.batch_size, opts.block_size, dt, gpu_core::devices::gpus().len(), place).map_err(std::io::Error::other)?;
    if let [whole] = shards.as_slice() {
        if whole.gpu_index == Shard::ANY_GPU {
            return build_for_training(cfg, opts, init, dt).map(Trained::Single);
        }
        let built = gpu_core::devices::with_gpu(whole.gpu_index as u32, || build_for_training(cfg, opts, init, dt));
        return built.map_err(std::io::Error::other)?.map(Trained::Single);
    }
    let layout: Vec<String> = shards.iter().map(|sh| format!("gpu{} layers {}..{}", sh.gpu_index, sh.start, sh.end)).collect();
    println!("qwen3 finetune: pipeline of {} stages: {}", shards.len(), layout.join(", "));
    let pipe = Pipeline::<Qwen>::with_shards_dt(cfg.clone(), opts.batch_size, opts.block_size, init, shards, dt);
    Ok(Trained::Pipeline(PipelineModel::new(pipe, cfg)))
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

/// The initial weights of a LoRA fine-tune: the adapter factors held in
/// memory over the base checkpoint, which stays wherever it is (mapped,
/// decoded one tensor at a time as the model is built).
pub struct LoraInit<'a> {
    adapters: HashMap<String, Vec<f32>>,
    base: Box<dyn TensorSource + 'a>,
}

impl<'a> LoraInit<'a> {
    /// Zero-delta adapters for `cfg` (whose `lora` is set), drawn from
    /// `seed`, over `base`.
    pub fn fresh(cfg: &QwenConfig, base: Box<dyn TensorSource + 'a>, seed: u64) -> LoraInit<'a> {
        LoraInit { adapters: crate::init::init_adapter_weights(cfg, seed), base }
    }

    /// The adapter tensors' names.
    pub fn adapter_names(&self) -> impl Iterator<Item = &str> {
        self.adapters.keys().map(String::as_str)
    }

    fn owner(&self, name: &str) -> &dyn TensorSource {
        if self.adapters.contains_key(name) {
            &self.adapters
        } else {
            &*self.base
        }
    }
}

impl TensorSource for LoraInit<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.owner(name).with_tensor(name, f)
    }
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.owner(name).raw_words(name)
    }
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        self.owner(name).with_tensor_chunks(name, max_elems, f)
    }
    fn with_tensor_u32_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[u32])) -> bool {
        self.owner(name).with_tensor_u32_chunks(name, max_elems, f)
    }
    fn numel(&self, name: &str) -> Option<usize> {
        self.owner(name).numel(name)
    }
    fn raw_blocks(&self, name: &str) -> Option<(checkpoint::gguf::BlockLayout, std::borrow::Cow<'_, [u8]>)> {
        self.owner(name).raw_blocks(name)
    }
    fn advise_drop(&self, name: &str) {
        self.owner(name).advise_drop(name)
    }
}

/// The configuration and initial weights of a LoRA fine-tune of `base`:
/// the base's own architecture and weights (a brain checkpoint, a GGUF or a
/// `transformers` directory, read as it is on disk), a `rank`/`alpha` adapter
/// over [`LORA_TARGETS`], the chosen targets, or the continued adapter's own,
/// and the adapter factors either freshly initialised from `seed` (zero
/// delta) or read from the adapter being continued.
///
/// Continuing an adapter at a different rank or alpha than it was trained
/// at is refused: its factors only mean what they mean at their own shape
/// and scale.
pub fn lora_start(base: &str, rank: u32, alpha: f32, seed: u64, start: &LoraStart<'_>) -> std::io::Result<(QwenConfig, LoraInit<'static>)> {
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
    let (mut cfg, base_src) = crate::open_checkpoint(base).map_err(std::io::Error::other)?;
    cfg.lora = Some(LoraCfg { rank, alpha, targets });
    // A fresh adapter starts at its zero-delta init; the base's own weights
    // are never copied, they stay behind `base_src`.
    let mut init = LoraInit::fresh(&cfg, base_src, seed);
    if let Some(tensors) = adapter {
        for (name, values) in tensors {
            match init.adapters.get(&name) {
                Some(slot) if slot.len() == values.len() => {
                    init.adapters.insert(name, values);
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
/// The frozen base linears are held at `base_dtype` ([`Dtype::BF16`]: half
/// the bytes of fp32, the adapters staying fp32). The model is placed on one
/// card when one has room for it, otherwise as a pipeline across the fewest
/// cards that do ([`plan_lora_layout`]).
pub fn finetune_lora_controlled(
    base: &str,
    dir: &Path,
    opts: &FitOpts,
    rank: u32,
    alpha: f32,
    start: &LoraStart<'_>,
    control: model::FitControl<'_>,
    base_dtype: Dtype,
) -> std::io::Result<(model::FitReport, Trained)> {
    let (cfg, init) = lora_start(base, rank, alpha, opts.seed, start)?;
    // A LoRA build never offloads its moments; the process-wide switch is
    // cleared for the construction and restored after, as `finetune_from`
    // does.
    let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
    std::env::remove_var("BRAIN_OFFLOAD_ADAM");
    let m = build_trainer(cfg, opts, &init, base_dtype);
    if let Some(v) = prev_off {
        std::env::set_var("BRAIN_OFFLOAD_ADAM", v);
    }
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    match m? {
        Trained::Single(m) => {
            let (report, m) = model::fit_controlled(m, model::causal_lm::<Qwen>(train, val, bcfg, itos), opts, None, control)?;
            Ok((report, Trained::Single(m)))
        }
        Trained::Pipeline(m) => {
            let (report, m) = model::fit_controlled(m, model::causal_lm::<PipelineModel<Qwen>>(train, val, bcfg, itos), opts, None, control)?;
            Ok((report, Trained::Pipeline(m)))
        }
    }
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
        let adapted: std::collections::BTreeSet<&str> = init.adapter_names().filter_map(|k| k.strip_suffix(".lora_a")).filter_map(|k| k.rsplit('.').nth(1)).collect();
        assert_eq!(adapted.into_iter().collect::<Vec<_>>(), ["up", "wq"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
