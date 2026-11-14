// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `finetune` action of `brain/qwen35`: full-parameter or LoRA fine-tuning
//! of Qwen3.8-27B on a masked token dataset, with every knob the trainers
//! have.
//!
//! The base is a brain checkpoint or a Hugging Face release directory (the FP8
//! release included). A brain checkpoint at fp32 trains in the resident
//! trainer, full or LoRA, and the run writes a checkpoint; a directory, or a
//! base held at bf16, nf4 or fp8, is streamed tensor by tensor onto the device
//! and trains a LoRA adapter only, which is written as an adapter (never a
//! copy of the base). The dataset is a tokenized folder (`brain data`
//! prepares one) or arrives as `dataset`/`validation` JSONL blobs in
//! `data::chat`'s schema, which the run tokenizes with the base's own
//! tokenizer and chat template. An adapter that names itself (`adapter_id` or
//! `adapter_name`) is filed in the model store, where a later run of the same
//! name continues it; an unnamed adapter is written to `save`, or returned as
//! the `adapter` blob.
//!
//! Every optimizer step is reported, the run stops between steps when its
//! invocation is cancelled (what ran is still written), and the seed it drew
//! is echoed so a run can be repeated.
//!
//! Swedish Embedded AB implements memory-bounded fine-tuning of very large
//! language models on a single GPU for its clients. If your team needs
//! expertise in adapting a 27B model to your own data on hardware you own then
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use gpu_core::select::Dtype;
use model::adapter::device::AdapterTuning;
use model::saved::SavedPolicy;
use model::FitOpts;
use serde_json::json;

use crate::base_source::{is_frozen_only, open_base, TRAIN_BASES};
use crate::finetune::{adapter_card, AdapterNaming, LongSeqOverrides, Mode, StreamedFinetune, StreamedRun};

/// LoRA rank a `lora` run uses when `rank` is omitted: 8, the rank this model
/// family's LoRA work is documented against. `alpha` then defaults to twice
/// the rank, the convention every other adapter-training action follows.
const DEFAULT_LORA_RANK: u32 = 8;
/// Largest LoRA alpha a run may ask for.
const MAX_ALPHA: f64 = 4_096.0;
/// Most sequences one micro-step may average over.
const MAX_BATCH: f64 = 1_024.0;
/// Largest gradient-norm clip a run may ask for.
const MAX_CLIP: f64 = 1_000_000.0;
/// Longest wait between the checkpoints a run keeps, seconds (one day).
const MAX_CHECKPOINT_SECS: f64 = 86_400.0;
const MODES: [&str; 2] = ["full", "lora"];
const ADAPTER_KINDS: [&str; 4] = ["lora", "ia3", "lokr", "dora"];

pub(crate) fn finetune_spec() -> ActionSpec {
    let d = FitOpts::default();
    ActionSpec::new(
        "finetune",
        "fine-tune Qwen3.8-27B on a masked token dataset: every weight, or a LoRA/IA3/LoKr/DoRA adapter on a base held at fp32, bf16, nf4 or fp8 (a brain checkpoint or a Hugging Face release in, a checkpoint or an adapter out)",
    )
    .training()
    .streaming()
    .cancellable()
    .param(
        ParamSpec::new("weights", ParamType::Str, "path to the base: a brain checkpoint (.safetensors) or a Hugging Face release directory (FP8 included); overrides the model store's own pick when set")
            .host_resolved(),
    )
    .param(
        ParamSpec::new(
            "tokenizer",
            ParamType::Str,
            "path to the base's tokenizer.json, read when the training set arrives as a `dataset` input (its directory also supplies the chat template); omit to use the model store's own pick, or the tokenizer.json beside `weights`",
        )
        .host_resolved(),
    )
    .param(
        ParamSpec::new("data", ParamType::Str, "path to the tokenized dataset folder (train.u32.bin, val.u32.bin, meta.json, optionally train.mask.bin); give it or the `dataset` input")
            .host_resolved(),
    )
    .param(
        ParamSpec::new(
            "save",
            ParamType::Str,
            "where to write the result: the checkpoint of a full or fp32 run (.safetensors; required), or the adapter of a streamed LoRA run (omit to return it as the `adapter` output, or to file it under `adapter_id`/`adapter_name`)",
        )
        .host_resolved(),
    )
    .param(ParamSpec::new("mode", ParamType::Enum(MODES.map(String::from).to_vec()), "full: every weight trains (needs an fp32 base); lora: a low-rank adapter trains on a frozen base").default(json!("full")))
    .param(ParamSpec::new("rank", ParamType::Int, "adapter rank (lora mode)").default(json!(DEFAULT_LORA_RANK)).min(1.0).max(capability::bounds::LORA_RANK))
    .param(ParamSpec::new("alpha", ParamType::Float, "LoRA alpha (the delta is scaled by alpha/rank); omit for 2 x rank").min(0.0).max(MAX_ALPHA))
    .param(ParamSpec::new("adapter_kind", ParamType::Enum(ADAPTER_KINDS.map(String::from).to_vec()), "the adapter method: lora, ia3 (learned output-channel scales), lokr (Kronecker-product delta) or dora (needs an fp32 base; default lora)"))
    .param(ParamSpec::new("lora_variant", ParamType::Str, "comma list of plain, rs (rank-stabilized scaling: alpha/sqrt(rank)) and fa (a frozen A); a continued adapter must be continued with the same"))
    .param(ParamSpec::new("lora_plus_ratio", ParamType::Float, "learning-rate ratio of B to A (LoRA+); omit for 1; a continued adapter must be continued with the same").min(0.0).max(1_000.0))
    .param(qwen3::lora_params::base_format_param(&TRAIN_BASES, Dtype::F32))
    .param(ParamSpec::new("recompute", ParamType::Bool, "rebuild each layer's activations in the backward instead of keeping them (default on for a bf16, nf4 or fp8 base; a streamed base only)"))
    .param(ParamSpec::new(
        "activations",
        ParamType::Enum(SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
        "what the backward keeps of the forward: every layer's activations, or each layer's input rebuilt in the backward (same gradients, lower memory). With lora it is `recompute` by another name; with full it is the activations= key of `train_precision`; naming both with different values is refused (default: each trainer's own)",
    ))
    .param(ParamSpec::new(
        "train_precision",
        ParamType::Str,
        "how a full run stores what it trains, as comma-separated key=value: state=f32|bf16|int8[:B] (device AdamW moments), master=none|bf16|f32, grad=f32|bf16, sr=on|off, narrow=on|off, weights=bf16|fp8[:B]|int8|nf4 with window=N (layers kept on the device, the rest streamed), clip=two-pass|running; a choice the model does not run is refused by name",
    ))
    .param(ParamSpec::new("gdn_segment", ParamType::Int, "Gated-DeltaNet segment length in tokens of a streamed base (0 keeps whole-sequence layers); rows of 2048 tokens or more at batch 1 get 2048 by themselves").min(0.0).max(capability::bounds::CONTEXT_TOKENS))
    .param(ParamSpec::new("bf16_residuals", ParamType::Bool, "hold the layer-boundary residual stream in bf16 on a streamed base (rows of 2048 tokens or more at batch 1 turn it on by themselves)"))
    .param(ParamSpec::new("mlp_rows", ParamType::Int, "MLP token-chunk size of a streamed base (0 runs the MLP whole); rows of 2048 tokens or more at batch 1 get 2048 by themselves").min(0.0).max(capability::bounds::CONTEXT_TOKENS))
    .param(ParamSpec::new(
        "adapter_id",
        ParamType::Str,
        "the card id of the adapter a streamed LoRA run writes, an adapter reference <vendor>/<repo>[-QUANT]:<owner>:<name>:<tag> (served under this id by `brain serve`); the adapter is kept in the model store",
    ))
    .param(ParamSpec::new(
        "adapter_name",
        ParamType::Str,
        "the name of the adapter a streamed LoRA run writes: its id is <base release>:local:<name>:latest, and it is kept in the model store",
    ))
    .param(ParamSpec::new("fresh", ParamType::Bool, "start a named adapter over from the base, replacing the stored one (otherwise a stored adapter of that name is continued)").default(json!(false)))
    .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(d.steps)).min(1.0).max(capability::bounds::TRAIN_STEPS))
    .param(ParamSpec::new("batch", ParamType::Int, "sequences averaged into each micro-step").default(json!(d.batch_size)).min(1.0).max(MAX_BATCH))
    .param(ParamSpec::new("block", ParamType::Int, "training sequence length, tokens; omit for the default of a tokenized folder, or the longest example of a `dataset`").min(1.0).max(capability::bounds::CONTEXT_TOKENS))
    .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate (cosine schedule down to a tenth of it over the run)").default(json!(d.lr)).min(0.0).max(capability::bounds::UNIT))
    .param(ParamSpec::new("warmup", ParamType::Int, "linear warmup steps; omit for the default, shortened to half the run when the run is shorter than it").min(0.0).max(capability::bounds::TRAIN_COUNT))
    .param(ParamSpec::new("weight_decay", ParamType::Float, "AdamW decoupled weight decay").default(json!(d.weight_decay)).min(0.0).max(capability::bounds::UNIT))
    .param(ParamSpec::new("grad_clip", ParamType::Float, "global gradient-norm clip; 0 disables it").default(json!(d.grad_clip)).min(0.0).max(MAX_CLIP))
    .param(ParamSpec::new("grad_accum", ParamType::Int, "micro-batches accumulated into each optimizer step").default(json!(d.grad_accum)).min(1.0).max(MAX_BATCH))
    .param(ParamSpec::new("eval_interval", ParamType::Int, "evaluate on the held-out split every this many steps; 0 never").default(json!(d.eval_interval)).min(0.0).max(capability::bounds::TRAIN_COUNT))
    .param(ParamSpec::new("eval_batches", ParamType::Int, "batches per evaluation").default(json!(d.eval_batches)).min(1.0).max(capability::bounds::TRAIN_COUNT))
    .param(ParamSpec::new("checkpoint_secs", ParamType::Int, "seconds between the checkpoints kept while training; 0 writes only the last").default(json!(d.checkpoint_secs)).min(0.0).max(MAX_CHECKPOINT_SECS))
    .param(ParamSpec::new("mask", ParamType::Str, "train only on what follows this character on each line (the answer after '=' of an LHS=RHS dataset); a token-masked dataset needs none"))
    .param(ParamSpec::new("align", ParamType::Bool, "start every sequence at a line boundary").default(json!(false)))
    .param(ParamSpec::new("seed", ParamType::Int, "seeds the initialization and the batches (omit for random; the outcome echoes the one used)").min(0.0))
    .param(ParamSpec::new("dataset_id", ParamType::Str, "provenance id of the `dataset` input, recorded in the adapter's model card"))
    .input(BlobSpec::new("dataset", Media::Bytes, "the training set: data::chat 'generic-messages-v2' JSONL, one packed sample per line (instead of `data`)"))
    .input(BlobSpec::new("validation", Media::Bytes, "optional held-out set for `dataset`, same JSONL schema; enables the periodic evaluation"))
    .output(BlobSpec::new("adapter", Media::Bytes, "the trained adapter (a streamed LoRA run with no `save`, `adapter_id` or `adapter_name`)"))
}

/// Everything about a `finetune` invocation that does not need the machine:
/// which trainer, how its base is held, what its optimizer does. Separated
/// from the run so the parameter grammar - which defaults are derived, which
/// are the caller's, which combinations are refused - is testable without a
/// GPU, a checkpoint or a corpus.
#[derive(Debug)]
pub(crate) struct FinetuneRequest {
    pub mode: Mode,
    pub base_dtype: Dtype,
    /// Recompute activations in the backward. Only a frozen-base tier has a
    /// recomputing trainer; the fp32 resident one keeps every activation.
    pub recompute: bool,
    /// Explicit long-sequence settings of a streamed (bf16/nf4/fp8) base.
    pub long: LongSeqOverrides,
    /// What the written adapter calls itself.
    pub naming: AdapterNaming,
    pub fresh: bool,
    /// The explicit training sequence length; `None` leaves `opts.block_size`
    /// at the default, or fits the longest example of a `dataset`.
    pub block: Option<u32>,
    pub opts: FitOpts,
}

/// A numeric param named in the invocation, as a `u32`.
fn whole(inv: &Invocation, name: &str) -> Option<u32> {
    inv.get_i64(name).map(|v| v.clamp(0, i64::from(u32::MAX)) as u32)
}

fn named(inv: &Invocation, name: &str) -> Option<String> {
    inv.get_str(name).filter(|v| !v.is_empty())
}

impl FinetuneRequest {
    pub(crate) fn from_invocation(inv: &Invocation) -> Result<FinetuneRequest, String> {
        let fail = |why: String| format!("qwen35 finetune: {why}");
        let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
        let rank = whole(inv, "rank").unwrap_or(DEFAULT_LORA_RANK).max(1);
        let alpha = inv.get_f64("alpha").map_or(rank as f32 * 2.0, |a| a as f32);
        let tuning = AdapterTuning::parse(
            named(inv, "lora_variant").as_deref(),
            inv.get_f64("lora_plus_ratio").map(|r| r.to_string()).as_deref(),
            named(inv, "adapter_kind").as_deref(),
        )
        .map_err(&fail)?;
        let base_dtype = qwen3::lora_params::base_format(inv, &TRAIN_BASES, Dtype::F32).map_err(&fail)?;
        let long = LongSeqOverrides { gdn_segment: whole(inv, "gdn_segment"), bf16_residuals: inv.get_bool("bf16_residuals"), mlp_rows: whole(inv, "mlp_rows") };
        let naming = AdapterNaming { id: named(inv, "adapter_id"), name: named(inv, "adapter_name") };
        if naming.id.is_some() && naming.name.is_some() {
            return Err(fail("adapter_id and adapter_name name the same thing; give one".into()));
        }
        let activations = inv.get_str("activations").map(|a| SavedPolicy::parse_activations(&a)).transpose().map_err(|e| fail(format!("activations: {e}")))?;
        let mut precision = match named(inv, "train_precision") {
            Some(spec) => model::TrainPrecision::parse(&spec).map_err(|e| fail(format!("train_precision {spec:?}: {e}")))?,
            None => model::TrainPrecision::default(),
        };

        let d = FitOpts::default();
        let steps = whole(inv, "steps").unwrap_or(d.steps).max(1);
        let lr = inv.get_f64("lr").map_or(d.lr, |v| v as f32);
        let mask = match named(inv, "mask") {
            Some(m) => {
                let mut chars = m.chars();
                Some(chars.next().filter(|_| chars.next().is_none()).ok_or_else(|| fail(format!("mask {m:?} is not one character")))?)
            }
            None => None,
        };
        let mut opts = FitOpts {
            steps,
            batch_size: whole(inv, "batch").unwrap_or(d.batch_size).max(1),
            lr,
            // Derived from THIS run's length and rate, not left at the struct
            // default: `decay_iters: 2000` against a 50-step run would spend
            // the whole run in the first 2.5% of the cosine schedule, so the
            // rate would never anneal.
            decay_iters: steps,
            min_lr: lr * 0.1,
            // The default warmup, unless the run is shorter than it: one
            // longer than the run never reaches the peak rate.
            warmup: whole(inv, "warmup").unwrap_or_else(|| d.warmup.min((steps / 2).max(1))),
            weight_decay: inv.get_f64("weight_decay").map_or(d.weight_decay, |v| v as f32),
            grad_clip: inv.get_f64("grad_clip").map_or(d.grad_clip, |v| v as f32),
            grad_accum: whole(inv, "grad_accum").unwrap_or(d.grad_accum).max(1),
            eval_interval: whole(inv, "eval_interval").unwrap_or(d.eval_interval),
            eval_batches: whole(inv, "eval_batches").unwrap_or(d.eval_batches).max(1),
            checkpoint_secs: u64::from(whole(inv, "checkpoint_secs").unwrap_or(d.checkpoint_secs as u32)),
            seed,
            mask_before: mask,
            mask_per_line: mask.is_some(),
            align_to_lines: inv.get_bool("align").unwrap_or(false),
            patience: 0,
            ..d
        };
        let block = whole(inv, "block").map(|b| b.max(1));
        if let Some(block) = block {
            opts.block_size = block;
        }

        // `full` is the default because a plain finetune of the sibling models
        // is full-parameter unless an adapter is asked for.
        if tuning.kind == model::adapter::DeviceKind::Dora && is_frozen_only(base_dtype) {
            return Err(fail(format!("adapter_kind dora reads the base weight, so it needs base_format f32 (a {} base is held narrow)", model::policy::format_name(base_dtype))));
        }
        let mode = match inv.get_str("mode").as_deref().unwrap_or("full") {
            "full" => Mode::FullOffload,
            "lora" => Mode::Lora { rank, alpha, tuning },
            other => return Err(fail(format!("mode {other:?}: expected \"lora\" or \"full\""))),
        };
        if is_frozen_only(base_dtype) && !matches!(mode, Mode::Lora { .. }) {
            return Err(fail(format!(
                "base_format {} holds the base frozen, so only mode lora can train on it; full-parameter fine-tuning needs base_format f32",
                model::policy::format_name(base_dtype)
            )));
        }
        // `activations` is the full run's `activations=` precision key, and the LoRA run's `recompute`.
        let mut recompute_flag = inv.get_bool("recompute");
        if let Some(policy) = activations {
            match mode {
                Mode::FullOffload => match precision.activations {
                    Some(inside) if inside != policy => {
                        return Err(fail(format!("activations {policy:?} conflicts with activations={inside:?} in train_precision; name the choice once")));
                    }
                    _ => precision.activations = Some(policy),
                },
                Mode::Lora { .. } => {
                    let wanted = policy == SavedPolicy::Recompute;
                    if let Some(named) = recompute_flag.filter(|named| *named != wanted) {
                        return Err(fail(format!("activations {policy:?} conflicts with recompute {named}; name the choice once")));
                    }
                    recompute_flag = Some(wanted);
                }
            }
        }
        if !precision.is_default() && !matches!(mode, Mode::FullOffload) {
            return Err(fail("train_precision describes how a full fine-tune stores what it trains: use it with mode full".into()));
        }
        opts.precision = precision;
        if recompute_flag == Some(true) && !is_frozen_only(base_dtype) {
            let name = if activations.is_some() { "activations recompute" } else { "recompute true" };
            return Err(fail(format!("{name}: the fp32 resident trainer keeps every layer's activations; recompute is available with a narrow base_format (bf16, nf4 or fp8)")));
        }
        let recompute = recompute_flag.unwrap_or(is_frozen_only(base_dtype));
        if long != LongSeqOverrides::default() && !is_frozen_only(base_dtype) {
            return Err(fail("gdn_segment, bf16_residuals and mlp_rows apply to the streamed trainer: use a narrow base_format (bf16, nf4 or fp8)".into()));
        }
        Ok(FinetuneRequest { mode, base_dtype, recompute, long, naming, fresh: inv.get_bool("fresh").unwrap_or(false), block, opts })
    }

    /// Whether the base trains through the streamed trainer (a directory, or
    /// a base held narrow) rather than the resident one.
    fn streamed(&self, base: &Path) -> bool {
        self.base_dtype != Dtype::F32 || base.is_dir()
    }
}

/// The path the model store keeps the adapter `card_id` at, under `root`.
fn filed_adapter_path(card_id: &str, root: &Path) -> Result<PathBuf, String> {
    let reference = brain_modelref::ModelRef::parse(card_id).map_err(|e| format!("adapter id {card_id:?}: {e}"))?;
    brain_modelstore::Store::new(root.to_path_buf()).adapter_weights_path(&reference).ok_or_else(|| format!("{card_id} is not an adapter reference"))
}

/// A tokenized dataset directory and what preparing it learned.
struct Dataset {
    dir: PathBuf,
    /// Longest example, tokens, when the run prepared the set from blobs.
    longest_example: Option<usize>,
    train_samples: Option<usize>,
    val_samples: Option<usize>,
}

/// The training set of `inv`: the `data` folder as it is, or the `dataset`
/// (and `validation`) blobs tokenized under `scratch` with the tokenizer and
/// chat template beside `tokenizer`. Blob content is untrusted caller input:
/// it goes through the one strict `data::chat` parse.
fn dataset(inv: &Invocation, weights: &Path, default_tokenizer: Option<&str>, scratch: &Path) -> Result<Dataset, String> {
    let fail = |why: String| format!("qwen35 finetune: {why}");
    let folder = named(inv, "data");
    let blobs = inv.get_blob("dataset").is_some();
    match (folder, blobs) {
        (Some(_), true) => Err(fail("give the training set as `data` or as the `dataset` input, not both".into())),
        (None, false) => Err(fail("missing the training set: name a `data` folder or give the `dataset` input".into())),
        (Some(dir), false) => Ok(Dataset { dir: PathBuf::from(dir), longest_example: None, train_samples: None, val_samples: None }),
        (None, true) => {
            let tok_path = match named(inv, "tokenizer").or_else(|| default_tokenizer.map(String::from)) {
                Some(p) => PathBuf::from(p),
                None => (if weights.is_dir() { weights } else { weights.parent().unwrap_or(Path::new(".")) }).join("tokenizer.json"),
            };
            let tok = QwenBpe::from_file(&tok_path.to_string_lossy()).map_err(|e| fail(format!("tokenizer {}: {e}", tok_path.display())))?;
            let tmpl = ChatTemplate::from_model_dir(tok_path.parent().unwrap_or(Path::new("."))).map_err(|e| fail(e.to_string()))?;
            let train = qwen3::caps::read_jsonl_blob(inv, "dataset", scratch, "qwen35 finetune")?.unwrap_or_default();
            if train.is_empty() {
                return Err(fail("'dataset' contains no samples".into()));
            }
            let val = qwen3::caps::read_jsonl_blob(inv, "validation", scratch, "qwen35 finetune")?.unwrap_or_default();
            // The MODEL's vocab (not the tokenizer's): it must match the base's embedding rows.
            let vocab = open_base(weights).map_err(&fail)?.cfg.vocab as usize;
            let dir = scratch.join("data");
            let render = data::chat::RenderOpts { keep_reasoning: false };
            let prepared = data::chat::prepare_chat_samples(&train, &val, &tok, &tmpl, render, vocab, &dir).map_err(|e| fail(format!("preparing the training data: {e}")))?;
            Ok(Dataset { dir, longest_example: Some(prepared.longest_example), train_samples: Some(train.len()), val_samples: Some(val.len()) })
        }
    }
}

/// The step observer: reports each step and says to continue until the
/// invocation is cancelled.
fn watch<'a>(inv: &'a Invocation, progress: &'a mut dyn FnMut(Progress)) -> impl FnMut(&model::StepReport) -> bool + 'a {
    move |s| {
        progress(Progress::step(s.step, s.steps, format!("loss {:.4} lr {:.2e} {:.1}s", s.loss, s.lr, s.work.step_secs())));
        !inv.cancel.is_cancelled()
    }
}

/// Where a run's result goes.
struct Target {
    /// The file written: the checkpoint, or the adapter.
    path: PathBuf,
    /// Whether it is returned as the `adapter` blob (the run owns a scratch file).
    as_blob: bool,
    /// Whether the model store keeps it, so a later run continues it.
    filed: bool,
}

fn target(request: &FinetuneRequest, inv: &Invocation, streamed_lora: bool, card_id: &str, scratch: &Path) -> Result<Target, String> {
    let fail = |why: String| format!("qwen35 finetune: {why}");
    let save = named(inv, "save").map(PathBuf::from);
    let named_adapter = request.naming.id.is_some() || request.naming.name.is_some();
    if named_adapter && !streamed_lora {
        return Err(fail("adapter_id and adapter_name name the adapter a streamed LoRA run writes (a bf16, nf4 or fp8 base, or a Hugging Face directory); this run writes a checkpoint to `save`".into()));
    }
    match (save, named_adapter) {
        (Some(_), true) => Err(fail("a named adapter is kept in the model store; give `save` or a name, not both".into())),
        (Some(path), false) => Ok(Target { path, as_blob: false, filed: false }),
        (None, true) => {
            let root = brain_modelstore::default_root().ok_or_else(|| fail("no models directory resolved (name one with --brain-data-dir)".into()))?;
            Ok(Target { path: filed_adapter_path(card_id, &root).map_err(&fail)?, as_blob: false, filed: true })
        }
        (None, false) if streamed_lora => Ok(Target { path: scratch.join("adapter.safetensors"), as_blob: true, filed: false }),
        (None, false) => Err(fail("missing required param 'save': a full or fp32 run writes a checkpoint".into())),
    }
}

fn resolved_weights(inv: &Invocation, default_weights: Option<&str>) -> Result<PathBuf, String> {
    let weights = named(inv, "weights")
        .or_else(|| default_weights.map(String::from))
        .ok_or("qwen35 finetune: no base checkpoint (pass 'weights', or pull one into the models directory)")?;
    if !Path::new(&weights).exists() {
        return Err(format!("qwen35 finetune: weights not found at '{weights}'"));
    }
    Ok(PathBuf::from(weights))
}

/// Run `finetune`: `default_weights` and `default_tokenizer` are the model
/// store's own picks, used when the invocation names none.
pub(crate) fn finetune(default_weights: Option<&str>, default_tokenizer: Option<&str>, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let request = FinetuneRequest::from_invocation(inv)?;
    let weights = resolved_weights(inv, default_weights)?;
    let scratch = capability::scratch::scratch_dir("qwen35-finetune").map_err(|e| format!("qwen35 finetune: a scratch directory: {e}"))?;
    let out = train_in(&scratch, &request, &weights, default_tokenizer, inv, progress);
    // A failed removal must not mask the training result the caller asked for.
    let _ = std::fs::remove_dir_all(&scratch);
    out
}

fn train_in(scratch: &Path, request: &FinetuneRequest, weights: &Path, default_tokenizer: Option<&str>, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let fail = |why: String| format!("qwen35 finetune: {why}");
    let streamed = request.streamed(weights);
    let lora = matches!(request.mode, Mode::Lora { .. });
    if streamed && !lora {
        return Err(fail("a Hugging Face directory trains only as a LoRA adapter (mode lora); full-parameter fine-tuning needs a brain checkpoint (.safetensors) at base_format f32".into()));
    }
    // The card of a streamed adapter, refused here when it cannot be made, before any training time is spent.
    let (card_id, base_id) = if streamed { adapter_card(weights, &request.naming).map_err(&fail)? } else { (String::new(), String::new()) };
    let target = target(request, inv, streamed, &card_id, scratch)?;
    let data = dataset(inv, weights, default_tokenizer, scratch)?;

    let mut opts = FitOpts { eval_interval: if data.val_samples == Some(0) { 0 } else { request.opts.eval_interval }, ..request.opts.clone() };
    opts.block_size = match (request.block, data.longest_example) {
        // A row holds one example, so it must hold the longest: a shorter block would train on a cut-off answer.
        (Some(block), Some(longest)) if (block as usize) < longest => {
            return Err(fail(format!("block {block} is shorter than the longest example ({longest} tokens); it would train on a cut-off answer")));
        }
        (Some(block), _) => block,
        (None, Some(longest)) => longest as u32,
        (None, None) => opts.block_size,
    };
    let save_str = target.path.to_string_lossy().into_owned();
    let mode_desc = match &request.mode {
        Mode::FullOffload => "full".to_string(),
        Mode::Lora { rank, alpha, .. } => format!("lora rank {rank} alpha {alpha}"),
    };
    progress(Progress::step(0, opts.steps, format!("training {} steps ({mode_desc}, base {}{})", opts.steps, model::policy::format_name(request.base_dtype), if request.recompute { ", recompute" } else { "" })));

    let mut on_step = watch(inv, progress);
    let mut continued = false;
    let report = if streamed {
        // A named adapter that is already stored is trained further from its own factors; `fresh` starts over.
        let resume = target.filed && target.path.is_file() && !request.fresh;
        let run = StreamedRun::uniform(request.base_dtype, request.recompute, request.long);
        continued = resume;
        StreamedFinetune { base: weights, dir: &data.dir, opts: &opts, mode: &request.mode, out: &save_str, resume, run: &run, card_id: &card_id, base_id: &base_id }.run_watched(&mut on_step)
    } else {
        crate::finetune::finetune_from_watched(&weights.to_string_lossy(), &data.dir, &opts, &request.mode, &save_str, false, &mut on_step)
    }
    .map_err(|e| fail(e.to_string()))?;
    if report.interrupted {
        return Err(format!("qwen35 finetune: cancelled after step {} (what ran is written at {})", report.steps_completed, target.path.display()));
    }

    let mut outcome = Outcome::new()
        .set("seed", json!(opts.seed))
        .set("steps", json!(report.steps_completed))
        .set("initial_loss", json!(report.initial_loss))
        .set("final_loss", json!(report.final_loss))
        .set("block", json!(opts.block_size))
        .set("saved", json!((!target.as_blob).then(|| target.path.display().to_string())))
        .set("train_samples", json!(data.train_samples))
        .set("val_samples", json!(data.val_samples));
    if streamed {
        outcome = outcome.set("adapter", json!(card_id)).set("base", json!(base_id)).set("continued", json!(continued));
    }
    if target.as_blob {
        let bytes = std::fs::read(&target.path).map_err(|e| fail(format!("read the trained adapter: {e}")))?;
        outcome = outcome.blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"id": card_id})));
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ActionSpec {
        finetune_spec()
    }

    /// The request a caller's params describe, defaults filled as the action runner fills them.
    fn request(params: &[(&str, serde_json::Value)]) -> Result<FinetuneRequest, String> {
        let inv = params.iter().fold(Invocation::new(), |inv, (name, value)| inv.set(name, value.clone()));
        FinetuneRequest::from_invocation(&spec().validate(inv)?)
    }

    /// Without `seed` every finetune draws its own: two requests from the same
    /// params are two independent runs, not one repeated default.
    #[test]
    fn an_unseeded_finetune_draws_a_fresh_seed_each_time() {
        let first = request(&[]).unwrap().opts.seed;
        let second = request(&[]).unwrap().opts.seed;
        assert_ne!(first, second);
        assert_eq!(request(&[("seed", json!(5))]).unwrap().opts.seed, 5);
    }

    #[test]
    fn the_adapter_is_named_by_an_id_or_by_a_name_but_not_both() {
        let ft = request(&[("adapter_id", json!("Qwen/Qwen3.8-27B-FP8:lab:facts:v1"))]).unwrap();
        assert_eq!(ft.naming.id.as_deref(), Some("Qwen/Qwen3.8-27B-FP8:lab:facts:v1"));
        let ft = request(&[("adapter_name", json!("facts"))]).unwrap();
        assert_eq!(ft.naming.name.as_deref(), Some("facts"));
        let err = request(&[("adapter_name", json!("facts")), ("adapter_id", json!("a/b:o:n:t"))]).unwrap_err();
        assert!(err.contains("adapter_id") && err.contains("adapter_name"), "{err}");
    }

    /// A plain finetune is a FULL one, and every `FitOpts` field the caller did
    /// not name keeps the shared `FitOpts::default()` value rather than a
    /// qwen35-private one.
    #[test]
    fn a_bare_finetune_is_a_full_finetune_on_the_shared_fitopts_defaults() {
        let d = FitOpts::default();
        let ft = request(&[("seed", json!(d.seed))]).unwrap();
        assert!(matches!(ft.mode, Mode::FullOffload), "{:?}", ft.mode);
        assert_eq!(ft.opts.steps, d.steps);
        assert_eq!(ft.opts.batch_size, d.batch_size);
        assert_eq!(ft.opts.block_size, d.block_size);
        assert_eq!(ft.opts.lr, d.lr);
        assert_eq!(ft.opts.grad_accum, d.grad_accum);
        assert_eq!(ft.opts.warmup, d.warmup);
        assert_eq!(ft.opts.weight_decay, d.weight_decay);
        assert_eq!(ft.opts.grad_clip, d.grad_clip);
        assert_eq!(ft.opts.checkpoint_secs, d.checkpoint_secs);
        assert_eq!(ft.opts.seed, d.seed);
        assert_eq!(ft.opts.mask_before, None);
        assert_eq!((ft.block, ft.base_dtype, ft.recompute), (None, Dtype::F32, false));
    }

    /// `train_precision` is the one model-agnostic value for how a full run stores what it trains; it parses into
    /// the typed value the generic fit loop reads, names the key it cannot read, and is a full-fine-tune option.
    #[test]
    fn a_train_precision_is_one_typed_value_for_a_full_finetune_only() {
        let ft = request(&[("train_precision", json!("state=int8,weights=fp8,window=2,clip=running"))]).unwrap();
        let r = ft.opts.precision.resident.expect("resident request");
        assert_eq!((r.window, r.clip_pass), (Some(2), model::resident_ft::ClipPass::Running));
        assert!(request(&[]).unwrap().opts.precision.is_default());
        let e = request(&[("train_precision", json!("state=fp4"))]).unwrap_err();
        assert!(e.contains("train_precision") && e.contains("state"), "{e}");
        let e = request(&[("mode", json!("lora")), ("train_precision", json!("state=bf16"))]).unwrap_err();
        assert!(e.contains("mode full"), "{e}");
    }

    /// The cosine schedule is defined against the run asked for: a short
    /// finetune left at `FitOpts::default()`'s `decay_iters` would never leave
    /// the flat top of the schedule.
    #[test]
    fn the_lr_schedule_follows_the_requested_run_length_not_the_struct_default() {
        let ft = request(&[("steps", json!(50)), ("lr", json!(1e-3))]).unwrap();
        assert_eq!(ft.opts.steps, 50);
        assert_eq!(ft.opts.decay_iters, 50);
        assert_eq!(ft.opts.lr, 1e-3);
        assert!((ft.opts.min_lr - 1e-4).abs() < 1e-9, "min_lr {}", ft.opts.min_lr);
        // The default warmup does not outlast a short run either.
        assert_eq!(ft.opts.warmup, 25);
        ft.opts.validate().expect("the derived schedule is consistent");
    }

    /// `lora` mode must be usable on its own: a caller who has not chosen a
    /// rank gets the family's documented one, and `alpha = 2 * rank`.
    #[test]
    fn lora_mode_defaults_its_rank_and_alpha() {
        let ft = request(&[("mode", json!("lora"))]).unwrap();
        let Mode::Lora { rank, alpha, .. } = ft.mode else { panic!("mode lora must select a LoRA mode") };
        assert_eq!((rank, alpha), (DEFAULT_LORA_RANK, DEFAULT_LORA_RANK as f32 * 2.0));

        let ft = request(&[("mode", json!("lora")), ("rank", json!(4))]).unwrap();
        let Mode::Lora { rank, alpha, .. } = ft.mode else { panic!("expected LoRA") };
        assert_eq!((rank, alpha), (4, 8.0), "alpha must follow an explicit rank, not stay at the default rank's");

        let ft = request(&[("mode", json!("lora")), ("rank", json!(4)), ("alpha", json!(1.5))]).unwrap();
        let Mode::Lora { rank, alpha, .. } = ft.mode else { panic!("expected LoRA") };
        assert_eq!((rank, alpha), (4, 1.5));
    }

    /// Every param the caller passes reaches `FitOpts`: a silently ignored
    /// `grad_accum` or `weight_decay` is a run that trained something other
    /// than what was asked for.
    #[test]
    fn every_training_param_reaches_fitopts() {
        let ft = request(&[
            ("batch", json!(2)),
            ("block", json!(16)),
            ("grad_accum", json!(4)),
            ("warmup", json!(7)),
            ("weight_decay", json!(0.05)),
            ("grad_clip", json!(0.5)),
            ("checkpoint_secs", json!(0)),
            ("seed", json!(99)),
            ("mask", json!("=")),
            ("align", json!(true)),
        ])
        .unwrap();
        assert_eq!(ft.opts.batch_size, 2);
        assert_eq!(ft.opts.block_size, 16);
        assert_eq!(ft.opts.grad_accum, 4);
        assert_eq!(ft.opts.warmup, 7);
        assert_eq!(ft.opts.weight_decay, 0.05);
        assert_eq!(ft.opts.grad_clip, 0.5);
        assert_eq!(ft.opts.checkpoint_secs, 0);
        assert_eq!(ft.opts.seed, 99);
        assert_eq!(ft.opts.mask_before, Some('='));
        assert!(ft.opts.mask_per_line);
        assert!(ft.opts.align_to_lines);
        assert!(request(&[("mask", json!("=="))]).unwrap_err().contains("mask"));
    }

    /// rsLoRA, LoRA+ and LoRA-FA reach the trainer through the LoRA mode, and a value that cannot train is refused.
    #[test]
    fn the_adapter_variant_params_are_options_of_the_lora_mode() {
        let with = |extra: &[(&str, serde_json::Value)]| request(&[&[("mode", json!("lora"))][..], extra].concat());
        let plain = with(&[]).unwrap();
        assert!(matches!(plain.mode, Mode::Lora { tuning, .. } if tuning == AdapterTuning::default()));
        let tuned = with(&[("lora_variant", json!("rs,fa")), ("lora_plus_ratio", json!(16))]).unwrap();
        let want = AdapterTuning { rank_stabilized: true, lr_ratio: 16.0, freeze_a: true, ..Default::default() };
        assert!(matches!(tuned.mode, Mode::Lora { tuning, .. } if tuning == want), "{:?}", tuned.mode);
        assert!(with(&[("lora_plus_ratio", json!(0))]).unwrap_err().contains("LoRA+"));
        assert!(with(&[("lora_variant", json!("dora"))]).unwrap_err().contains("lora-variant"));
        let lokr = with(&[("adapter_kind", json!("lokr"))]).unwrap();
        assert!(matches!(lokr.mode, Mode::Lora { tuning, .. } if tuning.kind == model::adapter::DeviceKind::Lokr));
    }

    /// The three things a run cannot do without are reported by name, and an unknown mode lists the valid ones.
    #[test]
    fn the_required_inputs_are_reported_by_name() {
        let dir = std::env::temp_dir().join(format!("qwen35-finetune-required-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("base.safetensors");
        std::fs::write(&weights, b"never read").unwrap();
        let run = |inv: Invocation| finetune(None, None, &spec().validate(inv).unwrap(), &mut |_| {}).unwrap_err();

        assert!(run(Invocation::new().set("data", json!("d")).set("save", json!("o"))).contains("weights"));
        let with_base = || Invocation::new().set("weights", json!(weights.to_str().unwrap()));
        assert!(run(with_base().set("save", json!("o"))).contains("`data`"));
        assert!(run(with_base().set("data", json!("d"))).contains("save"));
        let err = spec().validate(Invocation::new().set("mode", json!("qlora"))).unwrap_err();
        assert!(err.contains("mode") && err.contains("qlora"), "an unknown mode is refused by name: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The base precision params, and what a frozen-base tier defaults to.
    #[test]
    fn the_base_precision_params_select_the_tier() {
        let ft = request(&[("mode", json!("lora")), ("base_format", json!("nf4")), ("rank", json!(16)), ("alpha", json!(8)), ("block", json!(512))]).unwrap();
        assert_eq!(ft.base_dtype, Dtype::NF4);
        assert!(matches!(ft.mode, Mode::Lora { rank: 16, alpha, .. } if alpha == 8.0), "{:?}", ft.mode);
        assert_eq!(ft.opts.block_size, 512, "block is the training sequence length");
        assert!(ft.recompute, "a frozen-base tier recomputes by default");

        let off = request(&[("mode", json!("lora")), ("base_format", json!("bf16")), ("recompute", json!(false))]).unwrap();
        assert_eq!((off.base_dtype, off.recompute), (Dtype::BF16, false));

        let plain = request(&[]).unwrap();
        assert_eq!((plain.base_dtype, plain.recompute), (Dtype::F32, false), "the default keeps the resident fp32 trainer");
    }

    /// `activations` is the full run's `activations=` precision key and the LoRA run's `recompute`; naming both
    /// spellings with different values is refused by name.
    #[test]
    fn activations_route_to_the_precision_key_or_the_recompute_switch() {
        let full = |extra: &[(&str, serde_json::Value)]| request(extra);
        assert_eq!(full(&[("activations", json!("recompute"))]).unwrap().opts.precision.activations, Some(SavedPolicy::Recompute));
        assert_eq!(full(&[]).unwrap().opts.precision.activations, None, "absent leaves the trainer's own default");
        let e = full(&[("activations", json!("keep")), ("train_precision", json!("activations=recompute"))]).unwrap_err();
        assert!(e.contains("activations") && e.contains("train_precision"), "{e}");
        assert!(spec().validate(Invocation::new().set("activations", json!("bogus"))).is_err());

        let lora = |extra: &[(&str, serde_json::Value)]| request(&[&[("mode", json!("lora")), ("base_format", json!("bf16"))][..], extra].concat());
        assert!(lora(&[("activations", json!("recompute"))]).unwrap().recompute);
        assert!(!lora(&[("activations", json!("keep"))]).unwrap().recompute);
        let e = lora(&[("activations", json!("keep")), ("recompute", json!(true))]).unwrap_err();
        assert!(e.contains("activations") && e.contains("recompute"), "{e}");
        let e = request(&[("mode", json!("lora")), ("activations", json!("recompute"))]).unwrap_err();
        assert!(e.contains("activations recompute") && e.contains("fp32"), "{e}");
    }

    #[test]
    fn unsupported_tier_combinations_are_refused_with_the_reason() {
        let err = request(&[("base_format", json!("nf4"))]).unwrap_err();
        assert!(err.contains("frozen") && err.contains("mode lora") && err.contains("f32"), "{err}");
        assert!(request(&[("base_format", json!("bf16")), ("mode", json!("full"))]).unwrap_err().contains("frozen"));
        assert!(spec().validate(Invocation::new().set("base_format", json!("int8"))).is_err(), "int8 is not a base format");
        assert!(request(&[("recompute", json!(true))]).unwrap_err().contains("narrow base_format"));
        assert!(request(&[("mode", json!("lora")), ("adapter_kind", json!("dora")), ("base_format", json!("nf4"))]).unwrap_err().contains("dora"));
        assert!(spec().validate(Invocation::new().set("block", json!("long"))).is_err(), "a block is a number");
    }

    #[test]
    fn the_long_sequence_and_eval_params_reach_the_trainer() {
        let nf4 = |extra: &[(&str, serde_json::Value)]| request(&[&[("mode", json!("lora")), ("base_format", json!("nf4"))][..], extra].concat());
        let plain = nf4(&[]).unwrap();
        assert_eq!((plain.long, plain.opts.eval_interval), (LongSeqOverrides::default(), FitOpts::default().eval_interval));
        let ft = nf4(&[("gdn_segment", json!(256)), ("bf16_residuals", json!(false)), ("mlp_rows", json!(0)), ("eval_interval", json!(5)), ("eval_batches", json!(3))]).unwrap();
        assert_eq!(ft.long, LongSeqOverrides { gdn_segment: Some(256), bf16_residuals: Some(false), mlp_rows: Some(0) });
        assert_eq!((ft.opts.eval_interval, ft.opts.eval_batches), (5, 3));
        assert!(spec().validate(Invocation::new().set("mlp_rows", json!("many"))).is_err());
        let resident = request(&[("gdn_segment", json!(128))]).unwrap_err();
        assert!(resident.contains("narrow base_format"), "{resident}");
    }

    /// A named adapter lands where the model store keeps adapters, and only a name that makes an adapter reference does.
    #[test]
    fn a_named_adapter_is_filed_where_the_model_store_keeps_adapters() {
        let root = Path::new("/models");
        let path = filed_adapter_path("Qwen/Qwen3.8-27B-FP8:lab:facts:v1", root).unwrap();
        assert!(path.starts_with(root) && path.to_string_lossy().contains("facts"), "{}", path.display());
        assert!(filed_adapter_path("Qwen/Qwen3.8-27B-FP8", root).is_err(), "a base release is not an adapter");
    }

    /// A tiny base checkpoint, its tokenizer and a token dataset in a fresh directory:
    /// `(dir, base, data)`. Any weights train; the tests check the plumbing, not a loss value.
    fn tiny_fixture(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        use data::binio::{self, Meta};
        let dir = std::env::temp_dir().join(format!("qwen35-finetune-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        let cfg = crate::config::Qwen35Config::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let base = dir.join("base.safetensors");
        crate::model::Qwen35::new_on(gpu_core::Gpu::new(crate::model::pipelines()), cfg.clone(), 1, cfg.block_size, &init).save(base.to_str().unwrap());
        let tokens: Vec<u32> = (0..400).map(|i| [2, 5, 8, 15][i % 4]).collect();
        binio::write_u32_bin(&data.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&data.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(data.join("meta.json"), Meta::vocab_only(cfg.vocab as usize)).unwrap();
        crate::sample_actions::write_letter_tokenizer(&dir);
        (dir, base, data)
    }

    fn tiny_run(base: &Path, data: &Path) -> Invocation {
        let block = crate::config::Qwen35Config::tiny().block_size;
        Invocation::new().set("weights", json!(base.to_str().unwrap())).set("data", json!(data.to_str().unwrap())).set("steps", json!(3)).set("batch", json!(1)).set("block", json!(block)).set("seed", json!(5))
    }

    fn provider() -> capability::Registry {
        let mut reg = capability::Registry::new();
        reg.register(std::sync::Arc::new(crate::caps::Qwen35Provider::new()));
        reg
    }

    /// End to end through the action, tiny synthetic checkpoint and corpus: the fp32 LoRA path trains for every
    /// adapter method, every step is reported, the loss is finite and the checkpoint is written.
    #[test]
    fn a_tiny_f32_lora_finetune_runs_through_the_action_for_every_adapter_method() {
        let (dir, base, data) = tiny_fixture("f32");
        let out = dir.join("tuned.safetensors");
        for kind in ["lora", "ia3", "lokr", "dora"] {
            let _ = std::fs::remove_file(&out);
            let inv = tiny_run(&base, &data).set("save", json!(out.to_str().unwrap())).set("mode", json!("lora")).set("rank", json!(2)).set("adapter_kind", json!(kind)).set("checkpoint_secs", json!(0));
            let mut steps = 0;
            let done = provider().run(crate::caps::MODEL, "finetune", inv, &mut |p| steps += u32::from(p.step > 0 && p.message.starts_with("loss"))).unwrap();
            let (initial, last) = (done.outputs["initial_loss"].as_f64().unwrap(), done.outputs["final_loss"].as_f64().unwrap());
            assert!(initial.is_finite() && last.is_finite(), "{kind}: {initial} -> {last}");
            assert_eq!((steps, done.outputs["seed"].as_u64()), (3, Some(5)), "{kind}: a step is reported per optimizer step, and the seed is echoed");
            assert!(out.exists(), "{kind}: the trained checkpoint must be written to `save`");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_tiny_full_finetune_writes_a_checkpoint_the_model_loads() {
        let (dir, base, data) = tiny_fixture("full");
        let out = dir.join("tuned.safetensors");
        let done = provider().run(crate::caps::MODEL, "finetune", tiny_run(&base, &data).set("save", json!(out.to_str().unwrap())), &mut |_| {}).unwrap();
        assert_eq!(done.outputs["steps"].as_u64(), Some(3));
        let tuned = Invocation::new().set("weights", json!(out.to_str().unwrap())).set("prompt", json!("1 2 3")).set("max_new", json!(2));
        assert_eq!(provider().run(crate::caps::MODEL, "generate", tuned, &mut |_| {}).unwrap().outputs["tokens"].as_u64(), Some(2));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The streamed (bf16) LoRA run returns its adapter as a blob; the same adapter then overlays the base for
    /// `sample_adapter` (adapter on, off, and both) and folds into the weights for `generate`.
    #[test]
    fn a_tiny_bf16_lora_finetune_returns_an_adapter_that_generate_and_sample_adapter_use() {
        let (dir, base, data) = tiny_fixture("bf16");
        let reg = provider();
        let inv = tiny_run(&base, &data).set("mode", json!("lora")).set("base_format", json!("bf16")).set("rank", json!(2));
        let done = reg.run(crate::caps::MODEL, "finetune", inv, &mut |_| {}).unwrap();
        assert!(done.outputs["initial_loss"].as_f64().unwrap().is_finite() && done.outputs["final_loss"].as_f64().unwrap().is_finite());
        assert!(done.outputs["adapter"].as_str().unwrap().ends_with(":local:lora:latest"), "an unnamed adapter is <base release>:local:lora:latest: {}", done.outputs["adapter"]);
        let adapter = dir.join("adapter.safetensors");
        std::fs::write(&adapter, &done.blobs["adapter"].bytes).unwrap();

        let generate = |with: Option<&Path>| {
            let mut inv = Invocation::new().set("weights", json!(base.to_str().unwrap())).set("prompt", json!("1 2 3")).set("max_new", json!(3));
            if let Some(a) = with {
                inv = inv.set("adapter", json!(a.to_str().unwrap()));
            }
            reg.run(crate::caps::MODEL, "generate", inv, &mut |_| {})
        };
        assert_eq!(generate(Some(&adapter)).unwrap().outputs["tokens"].as_u64(), Some(3));
        assert_eq!(generate(None).unwrap().outputs["tokens"].as_u64(), Some(3), "the resident serves the adapter-free request too");
        assert!(generate(Some(&dir.join("missing.safetensors"))).unwrap_err().contains("adapter not found"));

        let sample = |mode: &str| {
            let inv = Invocation::new()
                .set("weights", json!(base.to_str().unwrap()))
                .set("adapter", json!(adapter.to_str().unwrap()))
                .set("prompt", json!("abc"))
                .set("max_new", json!(2))
                .set("base_format", json!("bf16"))
                .set("mode", json!(mode));
            reg.run(crate::caps::MODEL, "sample_adapter", inv, &mut |_| {})
        };
        let both = sample("compare").unwrap();
        let answer = &both.outputs["answers"][0];
        assert!(answer.get("adapter").is_some() && answer.get("base").is_some(), "{answer}");
        assert!(both.outputs["text"].as_str().unwrap().contains("[base] abc"));
        let off = sample("base").unwrap();
        assert_eq!(off.outputs["answers"][0]["base"], answer["base"], "the overlay switched off is the base exactly");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A cancelled run stops between steps and says how far it got.
    #[test]
    fn a_cancelled_finetune_stops_between_steps() {
        let (dir, base, data) = tiny_fixture("cancel");
        let mut inv = tiny_run(&base, &data).set("save", json!(dir.join("o.safetensors").to_str().unwrap())).set("steps", json!(50));
        inv.cancel = capability::CancelToken::armed();
        let cancel = inv.cancel.clone();
        let err = provider()
            .run(crate::caps::MODEL, "finetune", inv, &mut |p| {
                if p.step >= 2 {
                    cancel.cancel();
                }
            })
            .unwrap_err();
        assert!(err.to_string().contains("cancelled after step"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The training set may arrive as JSONL blobs: tokenized with the base's tokenizer and chat template, then
    /// trained on with the block fitted to the longest example.
    #[test]
    fn a_dataset_blob_is_tokenized_and_trained_on() {
        let (dir, base, _) = tiny_fixture("blob");
        std::fs::write(dir.join("tokenizer_config.json"), json!({"chat_template": "{% for m in messages %}{{ m.content }}\n{% endfor %}"}).to_string()).unwrap();
        let row = |q: &str, a: &str| json!({"messages": [{"role": "user", "content": q, "train": false}, {"role": "assistant", "content": a, "train": true}]}).to_string();
        let jsonl = [row("abc", "def"), row("ghi", "jkl"), row("mno", "pqr")].join("\n");
        let inv = Invocation::new()
            .set("weights", json!(base.to_str().unwrap()))
            .set("save", json!(dir.join("o.safetensors").to_str().unwrap()))
            .set("steps", json!(2))
            .set("batch", json!(1))
            .set("seed", json!(3))
            .blob("dataset", Blob::new(Media::Bytes, jsonl.into_bytes()));
        let done = provider().run(crate::caps::MODEL, "finetune", inv, &mut |_| {}).unwrap();
        assert_eq!(done.outputs["train_samples"].as_u64(), Some(3));
        assert!(done.outputs["block"].as_u64().unwrap() <= u64::from(crate::config::Qwen35Config::tiny().block_size));
        let both = Invocation::new().set("weights", json!(base.to_str().unwrap())).set("save", json!("o")).set("data", json!("d")).blob("dataset", Blob::new(Media::Bytes, b"{}".to_vec()));
        assert!(provider().run(crate::caps::MODEL, "finetune", both, &mut |_| {}).unwrap_err().to_string().contains("not both"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
