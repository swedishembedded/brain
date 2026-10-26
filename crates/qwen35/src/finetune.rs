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

use std::path::Path;

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::saved::{SavedFormat, SavedPolicy};
use model::adapter::device::AdapterTuning;
use model::FitOpts;

use crate::base_source::{is_frozen_only, lora_base_tuned, open_base, BaseCheckpoint};
use crate::config::Qwen35Config;
use crate::model::{pipelines, Qwen35, TrainOpts};

/// Which fine-tuning scheme.
#[derive(Clone, Debug)]
pub enum Mode {
    /// Every weight trainable. (This build keeps the AdamW moments on the device; only the
    /// Qwen3 trainer has a host-resident optimiser.)
    FullOffload,
    /// Low-rank adapters on the 12 targetable projections (`crate::config::
    /// lora_targets`); base frozen.
    Lora { rank: u32, alpha: f32, tuning: AdapterTuning },
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
    // Every start streams its tensors from the file through the bounded reader: the host never holds the model.
    let (cfg, init): (Qwen35Config, Box<dyn checkpoint::TensorSource>) = if resume && Path::new(out).exists() {
        // Resume: architecture + weights come from the checkpoint being
        // continued, not `base` - the adapter's (or full model's)
        // accumulated state must survive. Skipping the fresh-init overlay
        // below is exactly what lets `lora_b` keep the delta it learned in
        // earlier cycles instead of being reset back to its zero-delta init.
        let BaseCheckpoint { cfg, src } = open_base(Path::new(out)).map_err(std::io::Error::other)?;
        if let Mode::Lora { rank, alpha, tuning } = mode {
            let lora = cfg
                .lora
                .as_ref()
                .unwrap_or_else(|| panic!("resume checkpoint {out} has no LoRA config, but mode is Lora {{ rank: {rank}, alpha: {alpha} }}"));
            assert_eq!(lora.rank, *rank, "resume checkpoint {out} LoRA rank {} does not match requested rank {rank}", lora.rank);
            assert_eq!(lora.alpha, *alpha, "resume checkpoint {out} LoRA alpha {} does not match requested alpha {alpha}", lora.alpha);
            assert_eq!(lora.tuning(), *tuning, "resume checkpoint {out} was trained with {:?}, not the requested {tuning:?}", lora.tuning());
        }
        (cfg, src)
    } else {
        // Fresh start: the base's architecture and weights; a LoRA start overlays zero-delta adapters on them.
        let opened = open_base(Path::new(base)).map_err(std::io::Error::other)?;
        match mode {
            Mode::FullOffload => (opened.cfg, opened.src),
            Mode::Lora { rank, alpha, tuning } => {
                let (cfg, overlay) = crate::base_source::lora_overlay_tuned(opened, *rank, *alpha, *tuning, opts.seed, None).map_err(std::io::Error::other)?;
                (cfg, Box::new(overlay))
            }
        }
    };

    let m = match mode {
        Mode::FullOffload => build_full(cfg, opts, &*init)?,
        Mode::Lora { .. } => Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, &*init, &TrainOpts::default()),
    };
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen35>(train, val, bcfg, itos);
    let (report, _) = model::fit_controlled(m, obj, opts, Some(Path::new(out)), step_progress_control(&mut step_progress()))?;
    Ok((report.initial_loss, report.final_loss.unwrap_or(report.initial_loss)))
}

fn unsupported_precision(why: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, why)
}

/// The full fine-tune of `cfg` as `opts.precision` asks: the plain trainer with its fp32 weights and device AdamW state
/// (a choice no kernel runs is an error naming the parameter, before anything loads), or, when the precision asks for
/// resident low-precision linears, the resident trainer ([`build_resident`]).
fn build_full(cfg: Qwen35Config, opts: &FitOpts, src: &dyn checkpoint::TensorSource) -> std::io::Result<Qwen35> {
    <Qwen35Trainer as model::finetune::ChatTrainer>::TRAIN_PRECISION.check(&opts.precision).map_err(unsupported_precision)?;
    if let Some(resident) = opts.precision.resident_config(Some(opts.seed)).map_err(std::io::Error::other)? {
        return build_resident(cfg, opts, src, &resident);
    }
    let params = crate::model::shard_param_list(&cfg, &model::Shard::whole(cfg.n_layers as usize));
    opts.precision.resolve(&params, opts.precision.wants_host_moments()).map_err(std::io::Error::other)?;
    let train = TrainOpts { precision: opts.precision.clone(), gemm_precision: opts.gemm_precision, ..TrainOpts::default() };
    Ok(Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, src, &train))
}

/// Planner estimate of the device bytes a full fine-tune's activations take: the layer-boundary residuals, the
/// working set of the layers whose activations are alive at once (every layer when `recompute` is off, one when on)
/// and the chunked head's logits. An upper-bound estimate for placement, not an allocation.
pub fn activation_bytes(cfg: &Qwen35Config, b: u32, t: u32, recompute: bool) -> u64 {
    let n = u64::from(b) * u64::from(t);
    let (d, ff) = (u64::from(cfg.d_model), u64::from(cfg.intermediate_size));
    let gdn = u64::from(cfg.linear_conv_dim()) + 3 * u64::from(cfg.linear_value_dim());
    let gqa = u64::from(cfg.q_proj_dim()) + 3 * u64::from(cfg.q_dim()) + 2 * u64::from(cfg.kv_dim());
    let layer_words = n * (3 * d + 3 * ff + gdn.max(gqa));
    let alive = if recompute { 2 } else { u64::from(cfg.n_layers) };
    let head_words = (n * u64::from(cfg.vocab)).min(1 << 28);
    4 * ((u64::from(cfg.n_layers) + 1) * n * d + alive * layer_words + head_words)
}

/// The resident low-precision full fine-tune of `cfg` at `opts`' batch and context, placed through the planner: the
/// linears (every layer, or the window's layers), the bf16 vocabulary tables and their gradients on the device, the
/// master copy and optimizer state in host memory, the fp32 norms and the activations on the device. A window of layers
/// recomputes each layer's activations in the backward. A configuration the machine cannot hold is refused here, with
/// the planner's arithmetic, before anything is allocated.
pub fn build_resident(cfg: Qwen35Config, opts: &FitOpts, src: &dyn checkpoint::TensorSource, resident: &model::resident_ft::ResidentFtConfig) -> std::io::Result<Qwen35> {
    use model::memplan::{Capacity, Category, Demand};
    let shard = model::Shard::whole(cfg.n_layers as usize);
    let resident_numel: u64 = crate::model::resident_specs(&cfg, &shard).iter().map(|s| u64::from(s.n) * u64::from(s.k)).sum();
    let all_numel: u64 = crate::model::shard_param_list(&cfg, &shard).iter().map(|(_, n)| *n as u64).sum();
    let recompute = resident.window.is_some();
    // fp32 remainder (the norms): value, gradient and the two AdamW moments on the device.
    let remainder = Demand::fixed(Category::Weights, "fp32 norms (with gradients and moments)", (all_numel - resident_numel) * 4 * 4);
    let activations = Demand::fixed(Category::Activations, "activations and scratch", activation_bytes(&cfg, opts.batch_size, opts.block_size, recompute));
    let plan = model::resident_ft::ResidentPlan { cfg: resident.clone(), capacity: Capacity::measured(), others: vec![remainder, activations] };
    let saved = if recompute { SavedPolicy::Recompute } else { SavedPolicy::Keep };
    let train = TrainOpts { saved, gemm_precision: opts.gemm_precision, precision: opts.precision.clone(), resident: Some(plan), ..TrainOpts::default() };
    Qwen35::new_resident(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, src, &train).map_err(std::io::Error::other)
}

/// An `on_step` callback printing one line per optimizer step to stderr: the
/// step, the loss, the learning rate, and where the step's seconds went
/// ([`model::StepWork`] - its four phases) with the tokens it was charged for
/// against the tokens its records hold.
fn step_progress() -> impl FnMut(&model::StepReport) -> bool {
    move |r| {
        let w = r.work;
        eprintln!(
            "qwen35 finetune: step {}/{} loss {:.4} lr {:.2e} {:.1}s (prepare {:.2} forward {:.1} backward {:.1} optimizer {:.1}; {} useful / {} computed tok){}",
            r.step,
            r.steps,
            r.loss,
            r.lr,
            w.step_secs(),
            w.prepare_secs,
            w.forward_secs,
            w.backward_secs,
            w.optimizer_secs,
            w.useful_tokens,
            w.computed_tokens,
            w.device_memory.map_or(String::new(), |m| format!("; device peak {:.2} GiB, reserved {:.2} GiB", m.peak_live as f64 / (1u64 << 30) as f64, m.reserved as f64 / (1u64 << 30) as f64))
        );
        true
    }
}

fn step_progress_control<'a>(cb: &'a mut dyn FnMut(&model::StepReport) -> bool) -> model::FitControl<'a> {
    model::FitControl { on_step: Some(cb), ..model::FitControl::default() }
}

/// Micro-step size, in tokens, above which a streamed-base run switches to the
/// long-sequence settings (measured to fit a 27B step: knowledge 233).
const LONG_SEQ_TOKENS: u32 = 2048;
/// Tokens per micro-step a batch wider than one may hold.
const LONG_STEP_TOKENS: u32 = 8192;
/// Gated-DeltaNet segment length and MLP token-chunk size of the long-sequence
/// settings for a base held at `dtype`.
///
/// Each segment and each MLP chunk is a fixed number of kernel launches per
/// layer, so the largest that the card's memory allows is the cheapest
/// (measured: segment 2048 instead of 128 took the launches of a step from
/// 29,888 to 11,078, knowledge 233/234). What the card allows depends on how
/// much of it the frozen base takes: the 4-bit base (about 16 GB for the 27B)
/// leaves room for 2048-token segments and 2048-row MLP chunks at 8K tokens
/// (about 4 GiB more than the 128/512 settings). The bf16 base (about 54 GB)
/// takes the same 2048/2048: the 64-layer 8K step peaked at 67.1 GiB with 2048/2048
/// and at 65.8 GiB with 512/1024, while the larger settings cut the launches of a
/// step from 214,927 to 74,191 (knowledge 311). Both are whole multiples
/// of the 64-token chunk and of the convolution history the segments carry.
fn long_settings(dtype: Dtype) -> (u32, u32) {
    if is_frozen_only(dtype) {
        (2048, 2048)
    } else {
        (128, 512)
    }
}

/// Explicit values for the long-sequence settings, each replacing what a
/// streamed run would choose by itself (the long-sequence defaults for rows of
/// [`LONG_SEQ_TOKENS`] or more at batch 1, nothing otherwise). `None` keeps the
/// automatic choice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LongSeqOverrides {
    /// Gated-DeltaNet segment length in tokens; `Some(0)` keeps whole-sequence layers.
    pub gdn_segment: Option<u32>,
    /// Hold the layer-boundary residual stream in bf16.
    pub bf16_residuals: Option<bool>,
    /// MLP token-chunk size; `Some(0)` runs the MLP whole.
    pub mlp_rows: Option<u32>,
}

/// How a streamed-base run holds and trains its frozen base.
#[derive(Clone, Copy, Debug)]
pub struct StreamedRun {
    /// Device storage tier of the frozen base.
    pub dtype: Dtype,
    /// Rebuild each layer's activations in the backward instead of keeping them.
    pub recompute: bool,
    /// Explicit long-sequence settings.
    pub long: LongSeqOverrides,
}

/// What the adapter a run writes calls itself. Left empty, the card id is
/// derived from the base release (`<vendor>/<repo>:local:lora:latest`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterNaming {
    /// The whole card id, an adapter reference:
    /// `<vendor>/<repo>[-QUANT]:<owner>:<name>:<tag>`.
    pub id: Option<String>,
    /// Only the adapter's name: the id is `<base release>:local:<name>:latest`.
    pub name: Option<String>,
}

/// The card id and recorded base id of the adapter a run on `base` writes.
///
/// The base is recorded as the RELEASE id (`<vendor>/<repo>`, the last two
/// components of a store path), not the bare directory name, because that is
/// what `brain serve` binds an adapter to a quantisation of the release by. The
/// card id must be an adapter reference, or the server cannot list the adapter
/// as a model: an id that is not one, and a name on a base that has no release
/// id to derive from, are refused before any training time is spent. With no
/// naming at all and a base that has no release id (a bare file), the legacy id
/// `brain/qwen35-lora` is kept.
pub fn adapter_card(base: &Path, naming: &AdapterNaming) -> Result<(String, String), String> {
    let file = base.file_name().map_or_else(|| "base".to_string(), |n| n.to_string_lossy().into_owned());
    let release = base
        .parent()
        .and_then(|p| p.file_name())
        .map(|vendor| format!("{}/{file}", vendor.to_string_lossy()))
        .filter(|id| brain_modelref::ModelRef::parse(id).is_ok_and(|r| r.adapter().is_none()));
    let base_id = release.clone().unwrap_or(file);
    let is_adapter_ref = |id: &str| brain_modelref::ModelRef::parse(id).is_ok_and(|r| r.adapter().is_some());
    let id = match (&naming.id, &naming.name) {
        (Some(_), Some(_)) => return Err("--adapter-id and --name name the same thing; give one".to_string()),
        (Some(id), None) => {
            if !is_adapter_ref(id) {
                return Err(format!("adapter id {id:?} is not an adapter reference: <vendor>/<repo>[-QUANT]:<owner>:<name>:<tag>"));
            }
            id.clone()
        }
        (None, name) => match &release {
            Some(release) => {
                let id = format!("{release}:local:{}:latest", name.as_deref().unwrap_or("lora"));
                if !is_adapter_ref(&id) {
                    return Err(format!("adapter name {:?} does not make an adapter reference ({id}): use letters, digits, '-', '_' and '.'", name.as_deref().unwrap_or("lora")));
                }
                id
            }
            None if name.is_none() => "brain/qwen35-lora".to_string(),
            None => return Err(format!("base {base:?} has no <vendor>/<repo> release id to derive the adapter's id from: name it with --adapter-id <vendor>/<repo>:<owner>:<name>:<tag>")),
        },
    };
    Ok((id, base_id))
}

/// The trainer settings a streamed run builds with: the automatic
/// long-sequence ones unless `run.long` names a value. Refuses settings the
/// trainer cannot honour instead of building something else.
fn streamed_train_opts(run: &StreamedRun, batch: u32, block: u32, mtp: bool) -> Result<TrainOpts, String> {
    let auto = run.recompute && batch == 1 && block >= LONG_SEQ_TOKENS;
    let nonzero = |v: u32| (v > 0).then_some(v);
    let gdn_segment_tokens = run.long.gdn_segment.map_or(auto.then_some(long_settings(run.dtype).0), nonzero);
    let mlp_rows = run.long.mlp_rows.map_or(auto.then_some(long_settings(run.dtype).1), nonzero);
    let bf16_residuals = run.long.bf16_residuals.unwrap_or(auto && !mtp);
    if gdn_segment_tokens.is_some() && batch != 1 {
        return Err(format!("a Gated-DeltaNet segment needs a batch of one (batch is {batch}); use --batch 1 with --grad-accum"));
    }
    if bf16_residuals && mtp {
        return Err("a bf16 residual stream is not available on an MTP build".into());
    }
    if (bf16_residuals || mlp_rows.is_some()) && !run.recompute {
        return Err("a bf16 residual stream and chunked MLP rows need --recompute on".into());
    }
    let saved = match (run.recompute, bf16_residuals) {
        (false, _) => SavedPolicy::Keep,
        (true, false) => SavedPolicy::Recompute,
        (true, true) => SavedPolicy::Quantized(SavedFormat::Bf16),
    };
    Ok(TrainOpts { base: TierPolicy::uniform(run.dtype), saved, gdn_segment_tokens, mlp_rows, ..TrainOpts::default() })
}

/// LoRA fine-tune on a frozen base that is STREAMED from `base` (a
/// HuggingFace checkpoint directory, FP8 release included, or a brain-native
/// file) and held on the device at `run.dtype`, so the base never exists on the
/// host as a whole. `out` is an ADAPTER file (`crate::lora::save_adapter`),
/// never a copy of the base; with `resume` and an existing `out` the run
/// continues that adapter's factors instead of starting from zero delta.
/// `run.recompute` rebuilds each layer's activations in the backward instead
/// of keeping them ([`TrainOpts::recompute`]).
///
/// Only LoRA trains on a streamed base: a full-parameter run needs the base
/// as fp32 optimiser state on the device, which is what the resident
/// [`finetune`] path is for.
pub fn finetune_streamed(base: &Path, dir: &Path, opts: &FitOpts, mode: &Mode, out: &str, resume: bool, run: &StreamedRun) -> std::io::Result<(f32, f32)> {
    finetune_streamed_named(base, dir, opts, mode, out, resume, run, &AdapterNaming::default())
}

/// [`finetune_streamed`] writing the adapter's card under `naming`
/// ([`adapter_card`]); a naming that cannot make a card is refused before the
/// base is loaded.
#[allow(clippy::too_many_arguments)]
pub fn finetune_streamed_named(base: &Path, dir: &Path, opts: &FitOpts, mode: &Mode, out: &str, resume: bool, run: &StreamedRun, naming: &AdapterNaming) -> std::io::Result<(f32, f32)> {
    let (card_id, base_id) = adapter_card(base, naming).map_err(|why| std::io::Error::new(std::io::ErrorKind::InvalidInput, why))?;
    let Mode::Lora { rank, alpha, tuning } = mode else {
        return Err(unsupported(format!("a {} base is frozen: only LoRA trains on it", model::policy::format_name(run.dtype))));
    };
    let resume_from = resume.then(|| Path::new(out)).filter(|p| p.exists());
    let mut progress = step_progress();
    let (report, m) = finetune_lora_tuned(base, dir, opts, *rank, *alpha, *tuning, resume_from, run, step_progress_control(&mut progress))?;
    let provenance = checkpoint::st::TrainingProvenance::on_base("sft_lora", opts.seed, gpu_core::backend_name(), brain_modelstore::fetch::base_digest_sidecar(base)?);
    crate::lora::save_adapter_with_lineage(out, &m, &card_id, &base_id, None, Some(provenance))?;
    Ok((report.initial_loss, report.final_loss.unwrap_or(report.initial_loss)))
}

fn unsupported(why: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, why)
}

/// LoRA fine-tune on the streamed frozen base at `base` (see
/// [`finetune_streamed`]), under the caller's `control` (per-step callback and
/// cancellation, exact-resume state, evaluation snapshots), returning the
/// trained model rather than writing an adapter: the caller saves what it
/// wants of it with [`crate::lora::save_adapter_with_lineage`]. `resume_from`
/// continues an adapter file's factors (same rank and alpha).
#[allow(clippy::too_many_arguments)]
pub fn finetune_lora_controlled(
    base: &Path,
    dir: &Path,
    opts: &FitOpts,
    rank: u32,
    alpha: f32,
    resume_from: Option<&Path>,
    run: &StreamedRun,
    control: model::FitControl<'_>,
) -> std::io::Result<(model::FitReport, Qwen35)> {
    finetune_lora_tuned(base, dir, opts, rank, alpha, AdapterTuning::default(), resume_from, run, control)
}

/// [`finetune_lora_controlled`] with the run's hyper-parameters.
#[allow(clippy::too_many_arguments)]
pub fn finetune_lora_tuned(
    base: &Path,
    dir: &Path,
    opts: &FitOpts,
    rank: u32,
    alpha: f32,
    tuning: AdapterTuning,
    resume_from: Option<&Path>,
    run: &StreamedRun,
    control: model::FitControl<'_>,
) -> std::io::Result<(model::FitReport, Qwen35)> {
    finetune_lora_scored_tuned(base, dir, opts, rank, alpha, tuning, resume_from, run, control, &[]).map(|(report, trained, _)| (report, trained))
}

/// The LoRA model of the base at `base`, its frozen linears held at
/// `run.dtype` and streamed from disk, sized for `opts`' rows.
fn build_lora_model(base: &Path, opts: &FitOpts, rank: u32, alpha: f32, tuning: AdapterTuning, resume_from: Option<&Path>, run: &StreamedRun) -> std::io::Result<Qwen35> {
    // The streamed base is never held as fp32 on the device, whatever its format.
    let (dtype, recompute) = (run.dtype, run.recompute);
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
            model::policy::format_name(dtype),
            opts.batch_size * opts.grad_accum.max(1),
        )));
    }
    let (cfg, src) = lora_base_tuned(base, rank, alpha, tuning, opts.seed, resume_from).map_err(unsupported)?;
    let train = streamed_train_opts(run, opts.batch_size, opts.block_size, cfg.mtp).map_err(unsupported)?;
    let t_load = std::time::Instant::now();
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, &src, &train);
    eprintln!(
        "qwen35 finetune: base loaded as {} in {:.1}s (gdn segment {:?}, mlp rows {:?}, saved activations {})",
        model::policy::format_name(dtype),
        t_load.elapsed().as_secs_f64(),
        train.gdn_segment_tokens,
        train.mlp_rows,
        train.saved.label()
    );
    Ok(m)
}

/// Qwen3.5 as a [`model::finetune::ChatTrainer`]: the generic full-parameter,
/// LoRA and preference fine-tuners run on it. The base is held at fp32 (the
/// tier these trainers expose; `ChatFineTune` picks narrower ones itself).
pub struct Qwen35Trainer;

impl model::finetune::ChatTrainer for Qwen35Trainer {
    type Trained = Qwen35;
    const LORA_BASES: model::policy::TrainBases = model::policy::TrainBases { model: "qwen35", bases: crate::base_source::LORA_BASES };
    const TRAIN_PRECISION: model::TrainPrecisionSupport = model::TrainPrecisionSupport { model: "qwen35", state: model::StateSupport::Any, resident: true, layer_streamed: true };

    fn checkpoint_path(weights: &Path, model_dir: &Path) -> std::path::PathBuf {
        qwen3::store_checkpoint_path(weights, model_dir)
    }

    fn vocab(open: &Path) -> Result<usize, String> {
        Ok(open_base(open)?.cfg.vocab as usize)
    }

    /// Every weight trains in fp32; the moments follow `state`.
    fn train_full(base: &Path, opts: &FitOpts, state: model::finetune::OptimizerState, fit: &mut impl model::finetune::ModelFit) -> std::io::Result<(model::FitReport, Qwen35)> {
        let BaseCheckpoint { cfg, src } = open_base(base).map_err(std::io::Error::other)?;
        // The device AdamW runs the fused kernels only: refuse by name what it does not before anything loads.
        let params = crate::model::shard_param_list(&cfg, &model::Shard::whole(cfg.n_layers as usize));
        opts.precision.resolve(&params, opts.precision.wants_host_moments()).map_err(std::io::Error::other)?;
        let train = TrainOpts { precision: opts.precision.clone(), ..TrainOpts::default() };
        let m = state.scope(|| Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, &*src, &train));
        fit.fit(m)
    }

    /// The linears resident in `resident`'s encoding with a host-side master, every layer on the device or a window of
    /// them streamed through it; the embedding, head and norms train in fp32 on the device.
    fn train_full_resident(base: &Path, opts: &FitOpts, resident: &model::resident_ft::ResidentFtConfig, fit: &mut impl model::finetune::ModelFit) -> std::io::Result<(model::FitReport, Qwen35)> {
        let BaseCheckpoint { cfg, src } = open_base(base).map_err(std::io::Error::other)?;
        let m = build_resident(cfg, opts, &*src, resident)?;
        fit.fit(m)
    }

    fn train_lora(base: &Path, lora: &model::finetune::LoraRequest<'_>, opts: &FitOpts, fit: &mut impl model::finetune::ModelFit) -> std::io::Result<(model::FitReport, Qwen35)> {
        let run = StreamedRun { dtype: lora.checked::<Self>()?, recompute: false, long: LongSeqOverrides::default() };
        let m = model::finetune::OptimizerState::Device.scope(|| build_lora_model(base, opts, lora.rank, lora.alpha, AdapterTuning::default(), lora.continue_from, &run))?;
        fit.fit(m)
    }

    fn save_adapter(trained: &Qwen35, path: &Path, lineage: &model::finetune::AdapterLineage<'_>) -> std::io::Result<()> {
        let path = path.to_str().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "adapter path is not UTF-8"))?;
        crate::lora::save_adapter_with_lineage(path, trained, lineage.card_id, lineage.base_id, lineage.dataset_id, lineage.provenance.clone())
    }

    /// The checkpoint directory holds the brain-format weights as
    /// `model.safetensors` (which [`open_base`] reads back) and the tokenizer
    /// files; there is no Hugging Face writer for this architecture, and an
    /// adapter is not folded in.
    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>) -> Result<(), String> {
        model::finetune::export_brain_checkpoint("qwen35", trained, tokenizer_dir, out, adapter)
    }
}

/// [`finetune_lora_controlled`] that also scores the prepared datasets in
/// `held_out` (each a directory [`data::chat::prepare_chat_samples`] wrote, its
/// validation split being the records to score) before the first step and
/// after the last, on the model already resident for training - a second copy
/// of the base does not fit beside it. The scoring row is the training row:
/// `opts.block_size` must hold the longest held-out example.
#[allow(clippy::too_many_arguments)]
pub fn finetune_lora_scored(
    base: &Path,
    dir: &Path,
    opts: &FitOpts,
    rank: u32,
    alpha: f32,
    resume_from: Option<&Path>,
    run: &StreamedRun,
    control: model::FitControl<'_>,
    held_out: &[&Path],
) -> std::io::Result<(model::FitReport, Qwen35, Vec<model::ScoredSet>)> {
    finetune_lora_scored_tuned(base, dir, opts, rank, alpha, AdapterTuning::default(), resume_from, run, control, held_out)
}

/// [`finetune_lora_scored`] with the run's hyper-parameters.
#[allow(clippy::too_many_arguments)]
pub fn finetune_lora_scored_tuned(
    base: &Path,
    dir: &Path,
    opts: &FitOpts,
    rank: u32,
    alpha: f32,
    tuning: AdapterTuning,
    resume_from: Option<&Path>,
    run: &StreamedRun,
    control: model::FitControl<'_>,
    held_out: &[&Path],
) -> std::io::Result<(model::FitReport, Qwen35, Vec<model::ScoredSet>)> {
    let m = build_lora_model(base, opts, rank, alpha, tuning, resume_from, run)?;
    let (train_set, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let mut scored = model::score_base(&m, held_out, opts, resume_from.is_some())?;
    let obj = model::causal_lm::<Qwen35>(train_set, val, bcfg, itos);
    let (report, trained) = model::fit_controlled(m, obj, opts, None, control)?;
    if !report.interrupted {
        model::score_tuned(&trained, held_out, opts, &mut scored)?;
    }
    Ok((report, trained, scored))
}

/// The `(rank, alpha)` a saved adapter file was trained with, and the tuning its
/// scale has to be reproduced under.
///
/// `rank` and the card's scale come from the file; `tuning` supplies what the
/// card does not carry - the adapter method and whether the scale is rsLoRA's
/// (`alpha/sqrt(rank)`) rather than LoRA's (`alpha/rank`). A `kind` that
/// disagrees with the card, or a file with no adapter card, is refused: the
/// scale a scoring pass applies must be the one the adapter trained at, and a
/// wrong `rank_stabilized` is not detectable from the file afterwards.
pub fn adapter_shape(path: &Path, tuning: AdapterTuning) -> Result<(u32, f32), String> {
    let name = path.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))?;
    let card = checkpoint::st::read_card(name).map_err(|e| format!("{name}: {e}"))?;
    let adapter = card.and_then(|c| c.adapter).ok_or_else(|| format!("{name}: not an adapter file (no adapter card)"))?;
    if adapter.kind != tuning.kind.name() {
        return Err(format!("{name}: a {:?} adapter, asked to score as {:?}", adapter.kind, tuning.kind.name()));
    }
    let rank = adapter.rank.ok_or_else(|| format!("{name}: the adapter card has no rank"))?;
    let card_alpha = adapter.alpha.unwrap_or(rank as f32);
    // The card holds the SCALE the adapter trained at; under rsLoRA the training
    // alpha that reproduces it is that scale times sqrt(rank) (see the SDK's own
    // `lora_shape`, which reads a continued adapter the same way).
    let alpha = if tuning.rank_stabilized { card_alpha / (rank as f32).sqrt() } else { card_alpha };
    Ok((rank, alpha))
}

/// Teacher-forced per-position readouts of `records` from the training forward:
/// the frozen base at `run.dtype` with `adapter`'s factors loaded (or a fresh
/// zero-delta adapter, which leaves the base's logits unchanged exactly) and NO
/// optimizer step.
///
/// The model is [`build_lora_model`]'s - the same build a fine-tune trains - so
/// the scores are the trainer's own, and the records are laid out and walked by
/// [`crate::score::score_records`], the batching a training step uses. `opts`
/// sizes the scoring row (`block_size` must hold the longest record) exactly as
/// it sizes a training row.
pub fn score_records_streamed(
    base: &Path,
    adapter: Option<&Path>,
    tuning: AdapterTuning,
    records: &[crate::score::ScoredRecord],
    opts: &FitOpts,
    run: &StreamedRun,
) -> std::io::Result<crate::score::TokenScores> {
    let (rank, alpha, resume_from) = match adapter {
        Some(path) => {
            let (rank, alpha) = adapter_shape(path, tuning).map_err(std::io::Error::other)?;
            (rank, alpha, Some(path))
        }
        // A fresh adapter initialises `lora_b` to zero, so the base's logits are
        // unchanged: the base arm needs no second build path, and its numbers are
        // the tier's own.
        None => (1, 1.0, None),
    };
    let mut m = build_lora_model(base, opts, rank, alpha, tuning, resume_from, run)?;
    m.enable_token_argmax().map_err(std::io::Error::other)?;
    crate::score::score_records(&m, records, opts.batch_size, opts.block_size).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(recompute: bool, long: LongSeqOverrides) -> StreamedRun {
        StreamedRun { dtype: Dtype::NF4, recompute, long }
    }

    fn bf16_residuals(o: &TrainOpts) -> bool {
        o.saved.boundary_format() == SavedFormat::Bf16
    }

    #[test]
    fn long_rows_at_batch_one_default_to_the_long_sequence_settings() {
        let o = streamed_train_opts(&run(true, LongSeqOverrides::default()), 1, 2048, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, bf16_residuals(&o)), (Some(2048), Some(2048), true));
        let bf16 = StreamedRun { dtype: Dtype::BF16, recompute: true, long: LongSeqOverrides::default() };
        let o = streamed_train_opts(&bf16, 1, 8192, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows), (Some(2048), Some(2048)), "the bf16 base fits the same settings (67.1 GiB peak at 64 layers, 8K)");
        let short = streamed_train_opts(&run(true, LongSeqOverrides::default()), 1, 2047, false).unwrap();
        assert_eq!((short.gdn_segment_tokens, short.mlp_rows, bf16_residuals(&short)), (None, None, false));
        let mtp = streamed_train_opts(&run(true, LongSeqOverrides::default()), 1, 4096, true).unwrap();
        assert!(!bf16_residuals(&mtp), "an MTP build keeps the fp32 stream");
    }

    #[test]
    fn explicit_settings_override_the_defaults_in_both_directions() {
        let big = LongSeqOverrides { gdn_segment: Some(256), bf16_residuals: Some(false), mlp_rows: Some(1024) };
        let o = streamed_train_opts(&run(true, big), 1, 8192, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, bf16_residuals(&o)), (Some(256), Some(1024), false));
        let off = LongSeqOverrides { gdn_segment: Some(0), bf16_residuals: Some(false), mlp_rows: Some(0) };
        let o = streamed_train_opts(&run(true, off), 1, 8192, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, bf16_residuals(&o)), (None, None, false));
        let on = LongSeqOverrides { gdn_segment: Some(64), bf16_residuals: Some(true), mlp_rows: Some(128) };
        let o = streamed_train_opts(&run(true, on), 1, 256, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, bf16_residuals(&o)), (Some(64), Some(128), true));
    }

    #[test]
    fn settings_the_trainer_cannot_honour_are_refused() {
        let seg = LongSeqOverrides { gdn_segment: Some(64), ..Default::default() };
        assert!(streamed_train_opts(&run(true, seg), 2, 128, false).unwrap_err().contains("batch of one"));
        let res = LongSeqOverrides { bf16_residuals: Some(true), ..Default::default() };
        assert!(streamed_train_opts(&run(true, res), 1, 128, true).unwrap_err().contains("MTP"));
        assert!(streamed_train_opts(&run(false, res), 1, 128, false).unwrap_err().contains("--recompute on"));
    }

    /// A fresh adapter must be servable as written: its card id is an adapter
    /// reference and its base is the release the adapter was trained on, so
    /// `brain serve` binds it to the quantisation of that release with no hand
    /// edits.
    #[test]
    fn a_fresh_adapters_card_is_servable_without_edits() {
        let base = Path::new("/models/Qwen/Qwen3.8-27B-FP8");
        let (id, base_id) = adapter_card(base, &AdapterNaming::default()).unwrap();
        assert_eq!(base_id, "Qwen/Qwen3.8-27B-FP8", "the release id, not the directory name");
        assert_eq!(id, "Qwen/Qwen3.8-27B-FP8:local:lora:latest", "derived from the base release when nothing names the adapter");
        let named = adapter_card(base, &AdapterNaming { name: Some("facts".into()), ..Default::default() }).unwrap();
        assert_eq!(named.0, "Qwen/Qwen3.8-27B-FP8:local:facts:latest");
        let explicit = adapter_card(base, &AdapterNaming { id: Some("unsloth/Qwen3.8-27B-GGUF-Q8_0:lab:facts:v2".into()), ..Default::default() }).unwrap();
        assert_eq!((explicit.0.as_str(), explicit.1.as_str()), ("unsloth/Qwen3.8-27B-GGUF-Q8_0:lab:facts:v2", "Qwen/Qwen3.8-27B-FP8"));
        for id in [&id, &named.0, &explicit.0] {
            assert!(brain_modelref::ModelRef::parse(id).unwrap().adapter().is_some(), "{id} is an adapter reference");
        }
    }

    #[test]
    fn an_id_that_is_not_an_adapter_reference_is_refused_before_training() {
        let base = Path::new("/models/Qwen/Qwen3.8-27B-FP8");
        for bad in ["facts", "Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B-FP8:lab:../x:v1"] {
            let why = adapter_card(base, &AdapterNaming { id: Some(bad.into()), ..Default::default() }).unwrap_err();
            assert!(why.contains(bad) && why.contains("<vendor>/<repo>"), "{why}");
        }
        let both = AdapterNaming { id: Some("a/b:o:n:t".into()), name: Some("x".into()) };
        assert!(adapter_card(base, &both).unwrap_err().contains("--adapter-id and --name"));
    }

    /// A base that is a bare file has no release id to derive from: an adapter
    /// then keeps the legacy id unless the caller names it.
    #[test]
    fn a_base_without_a_release_id_needs_an_explicit_name_to_be_servable() {
        let base = Path::new("base.safetensors");
        assert_eq!(adapter_card(base, &AdapterNaming::default()).unwrap(), ("brain/qwen35-lora".to_string(), "base.safetensors".to_string()));
        assert!(adapter_card(base, &AdapterNaming { name: Some("facts".into()), ..Default::default() }).unwrap_err().contains("--adapter-id"));
        assert!(adapter_card(base, &AdapterNaming { id: Some("local/base:lab:facts:v1".into()), ..Default::default() }).is_ok());
    }

    /// The long-sequence settings shrink as the base takes more of the card and
    /// stay whole multiples of the 64-token GDN chunk.
    #[test]
    fn long_settings_follow_the_base_footprint() {
        let (nf4, bf16, f32) = (long_settings(Dtype::NF4), long_settings(Dtype::BF16), long_settings(Dtype::F32));
        assert!(nf4.0 >= bf16.0 && bf16.0 >= f32.0, "segments: {nf4:?} {bf16:?} {f32:?}");
        assert!(nf4.1 >= bf16.1 && bf16.1 >= f32.1, "MLP rows: {nf4:?} {bf16:?} {f32:?}");
        for (seg, rows) in [nf4, bf16, f32] {
            assert_eq!(seg % 64, 0);
            assert!(rows > 0);
        }
    }
}
