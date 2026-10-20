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
    /// Every weight trainable. (This build keeps the AdamW moments on the device; only the
    /// Qwen3 trainer has a host-resident optimiser.)
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
    let (train, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let obj = model::causal_lm::<Qwen35>(train, val, bcfg, itos);
    let (report, _) = model::fit_controlled(m, obj, opts, Some(Path::new(out)), step_progress_control(&mut step_progress()))?;
    Ok((report.initial_loss, report.final_loss.unwrap_or(report.initial_loss)))
}

/// An `on_step` callback printing one line per optimizer step to stderr: the
/// step, the loss, the learning rate and the seconds the step took.
fn step_progress() -> impl FnMut(&model::StepReport) -> bool {
    let mut last = std::time::Instant::now();
    move |r| {
        let now = std::time::Instant::now();
        eprintln!("qwen35 finetune: step {}/{} loss {:.4} lr {:.2e} {:.1}s", r.step, r.steps, r.loss, r.lr, now.duration_since(last).as_secs_f64());
        last = now;
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
fn long_settings(dtype: BaseDtype) -> (u32, u32) {
    match dtype {
        BaseDtype::Nf4 => (2048, 2048),
        BaseDtype::Bf16 => (2048, 2048),
        BaseDtype::F32 => (128, 512),
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
    pub dtype: BaseDtype,
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
    let tier = match run.dtype {
        BaseDtype::F32 => Dtype::F32,
        BaseDtype::Bf16 => Dtype::BF16,
        BaseDtype::Nf4 => Dtype::NF4,
    };
    Ok(TrainOpts { base: TierPolicy::uniform(tier), recompute: run.recompute, gdn_segment_tokens, bf16_residuals, mlp_rows, ..TrainOpts::default() })
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
    let Mode::Lora { rank, alpha } = mode else {
        return Err(unsupported(format!("a {} base is frozen: only LoRA trains on it", run.dtype.name())));
    };
    let resume_from = resume.then(|| Path::new(out)).filter(|p| p.exists());
    let mut progress = step_progress();
    let (report, m) = finetune_lora_controlled(base, dir, opts, *rank, *alpha, resume_from, run, step_progress_control(&mut progress))?;
    let provenance = checkpoint::st::TrainingProvenance::on_base("sft_lora", opts.seed, gpu_core::backend_name(), brain_modelstore::fetch::base_digest(base)?);
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
    finetune_lora_scored(base, dir, opts, rank, alpha, resume_from, run, control, &[]).map(|(report, trained, _)| (report, trained))
}

/// A held-out record set scored in the trainer's own process: the loss of the
/// untouched base (`None` when the run continues an adapter, whose starting
/// point is not the base) and of the model the run hands back, over exactly
/// the positions training supervises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredSet {
    pub base_loss: Option<f32>,
    pub tuned_loss: Option<f32>,
    /// Supervised positions scored, and the records they come from.
    pub positions: usize,
    pub records: usize,
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
) -> std::io::Result<(model::FitReport, Qwen35, Vec<ScoredSet>)> {
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
            dtype.name(),
            opts.batch_size * opts.grad_accum.max(1),
        )));
    }
    let (cfg, src) = lora_base(base, rank, alpha, opts.seed, resume_from).map_err(unsupported)?;
    let train = streamed_train_opts(run, opts.batch_size, opts.block_size, cfg.mtp).map_err(unsupported)?;
    let t_load = std::time::Instant::now();
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, opts.batch_size, opts.block_size, &src, &train);
    eprintln!(
        "qwen35 finetune: base loaded as {} in {:.1}s (gdn segment {:?}, mlp rows {:?}, bf16 residuals {})",
        dtype.name(),
        t_load.elapsed().as_secs_f64(),
        train.gdn_segment_tokens,
        train.mlp_rows,
        train.bf16_residuals
    );
    let (train_set, val, bcfg, _vocab, itos) = model::load_dataset_with_itos(dir, opts)?;
    let mut scored: Vec<ScoredSet> = Vec::with_capacity(held_out.len());
    for set in held_out {
        let (_, records, set_cfg, _, _) = model::load_dataset_with_itos(set, opts)?;
        let (count, positions) = supervised_positions(&records, &set_cfg);
        let base_loss = if resume_from.is_none() { score_set(&m, set, opts)? } else { None };
        scored.push(ScoredSet { base_loss, tuned_loss: None, positions, records: count });
    }
    let obj = model::causal_lm::<Qwen35>(train_set, val, bcfg, itos);
    let (report, trained) = model::fit_controlled(m, obj, opts, None, control)?;
    if !report.interrupted {
        for (set, slot) in held_out.iter().zip(scored.iter_mut()) {
            slot.tuned_loss = score_set(&trained, set, opts)?;
        }
    }
    Ok((report, trained, scored))
}

/// The mean cross-entropy of `m` over every supervised position of `records`.
fn score_set(m: &Qwen35, set: &Path, opts: &FitOpts) -> std::io::Result<Option<f32>> {
    use model::Objective;
    let (train, records, cfg, _, itos) = model::load_dataset_with_itos(set, opts)?;
    let mut obj = model::causal_lm::<Qwen35>(train, records, cfg, itos);
    Ok(obj.eval(m, &mut data::rng::Rng::new(data::rng::random_seed()), 0))
}

/// `(records, supervised positions)` of a prepared split, counted the way the
/// evaluation walks it.
fn supervised_positions(records: &data::loader::TokenDataset, cfg: &data::loader::BatchConfig) -> (usize, usize) {
    let count = records.example_count().unwrap_or(0);
    let rows = cfg.batch_size.max(1);
    let mut positions = 0;
    let mut first = 0;
    while first < count {
        let Some((_, y)) = records.example_rows(cfg, first) else { break };
        positions += y.iter().filter(|&&t| t != data::loader::IGNORE).count();
        first += rows;
    }
    (count, positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(recompute: bool, long: LongSeqOverrides) -> StreamedRun {
        StreamedRun { dtype: BaseDtype::Nf4, recompute, long }
    }

    #[test]
    fn long_rows_at_batch_one_default_to_the_long_sequence_settings() {
        let o = streamed_train_opts(&run(true, LongSeqOverrides::default()), 1, 2048, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, o.bf16_residuals), (Some(2048), Some(2048), true));
        let bf16 = StreamedRun { dtype: BaseDtype::Bf16, recompute: true, long: LongSeqOverrides::default() };
        let o = streamed_train_opts(&bf16, 1, 8192, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows), (Some(2048), Some(2048)), "the bf16 base fits the same settings (67.1 GiB peak at 64 layers, 8K)");
        let short = streamed_train_opts(&run(true, LongSeqOverrides::default()), 1, 2047, false).unwrap();
        assert_eq!((short.gdn_segment_tokens, short.mlp_rows, short.bf16_residuals), (None, None, false));
        let mtp = streamed_train_opts(&run(true, LongSeqOverrides::default()), 1, 4096, true).unwrap();
        assert!(!mtp.bf16_residuals, "an MTP build keeps the fp32 stream");
    }

    #[test]
    fn explicit_settings_override_the_defaults_in_both_directions() {
        let big = LongSeqOverrides { gdn_segment: Some(256), bf16_residuals: Some(false), mlp_rows: Some(1024) };
        let o = streamed_train_opts(&run(true, big), 1, 8192, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, o.bf16_residuals), (Some(256), Some(1024), false));
        let off = LongSeqOverrides { gdn_segment: Some(0), bf16_residuals: Some(false), mlp_rows: Some(0) };
        let o = streamed_train_opts(&run(true, off), 1, 8192, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, o.bf16_residuals), (None, None, false));
        let on = LongSeqOverrides { gdn_segment: Some(64), bf16_residuals: Some(true), mlp_rows: Some(128) };
        let o = streamed_train_opts(&run(true, on), 1, 256, false).unwrap();
        assert_eq!((o.gdn_segment_tokens, o.mlp_rows, o.bf16_residuals), (Some(64), Some(128), true));
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
        let (nf4, bf16, f32) = (long_settings(BaseDtype::Nf4), long_settings(BaseDtype::Bf16), long_settings(BaseDtype::F32));
        assert!(nf4.0 >= bf16.0 && bf16.0 >= f32.0, "segments: {nf4:?} {bf16:?} {f32:?}");
        assert!(nf4.1 >= bf16.1 && bf16.1 >= f32.1, "MLP rows: {nf4:?} {bf16:?} {f32:?}");
        for (seg, rows) in [nf4, bf16, f32] {
            assert_eq!(seg % 64, 0);
            assert!(rows > 0);
        }
    }
}
