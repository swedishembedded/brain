// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that pretrain language
// models on raw text beside the models being served, for its clients. If your
// team needs expertise in pretraining small models on shared accelerators,
// you can procure our services by sending an email to info@swedishembedded.com.

//! `pretrain`: raw-text causal-LM training of every weight of a base, as a job of the service
//! ([`brain::Pretrain`]). The corpus and an optional held-out corpus come by digest or blob (JSONL `{"text": ...}`
//! documents); each document is encoded with the base's tokenizer and followed by a separator token, every position
//! is supervised, and the held-out corpus is scored every `eval_every` steps (an `eval_loss` on that step's frame).
//! The checkpoint leaves like every checkpoint job's.

use std::time::Instant;

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::json;

use crate::consolidate::{check_install_name, checkpoint_outcome};
use crate::job::{self, trainer_failure, utf8, JobEnv, Workdir};
use crate::request::{check_attached, DataRefs};

/// The action's name.
pub const ACTION: &str = "pretrain";
/// The longest separator token named, in bytes.
pub const MAX_SEPARATOR_BYTES: usize = 128;

/// What `pretrain` is.
pub fn spec() -> ActionSpec {
    let f = |name: &str, help: &str, min: f64, max: f64| ParamSpec::new(name, ParamType::Float, help).min(min).max(max);
    let count = |name: &str, help: &str, default: u32, min: f64, max: f64| ParamSpec::new(name, ParamType::Int, help).default(json!(default)).min(min).max(max).step(1.0);
    ActionSpec::new(ACTION, "pretrain every weight of a base on raw text (JSONL {\"text\"} documents, each followed by a separator token, every position supervised), scoring a held-out corpus as it goes: the checkpoint leaves by manifest digest in the artifact store (and optionally into the model store)")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("base", ParamType::Str, "the base model, `<vendor>/<repo>` in the server's model store (an init_model checkpoint, or any model of an architecture with a full-parameter trainer)").required())
        .param(ParamSpec::new("data", ParamType::Str, "JSON document naming the corpora: {\"train\": SRC, \"held_out\"?: SRC} where SRC is {\"digest\": \"<blake3 hex>\"} or {\"blob\": \"dataset\" | \"held_out\"}").required())
        .param(count("steps", "optimiser steps", 100, 1.0, capability::bounds::TRAIN_STEPS))
        .param(count("batch", "rows per micro-batch", 8, 1.0, 4096.0))
        .param(count("block", "tokens per row, at most the model's positions", 256, 2.0, capability::bounds::CONTEXT_TOKENS))
        .param(count("grad_accum", "micro-batches averaged into one optimiser step", 1, 1.0, 4096.0))
        .param(f("lr", "peak learning rate (default 1e-3)", 0.0, 1.0))
        .param(f("min_lr", "the rate the cosine decays to (default a tenth of lr)", 0.0, 1.0))
        .param(ParamSpec::new("warmup", ParamType::Int, "linear warmup steps (default a twentieth of the run)").min(0.0).max(capability::bounds::TRAIN_COUNT).step(1.0))
        .param(f("weight_decay", "AdamW weight decay (default 0.1)", 0.0, 1.0))
        .param(f("grad_clip", "global gradient-norm clip (default 1; 0 turns it off)", 0.0, 1e6))
        .param(count("eval_every", "steps between evaluations of the held-out corpus (0: none; needs held_out)", 0, 0.0, capability::bounds::TRAIN_COUNT))
        .param(count("eval_batches", "batches of held-out windows one evaluation averages", 8, 1.0, 4096.0))
        .param(ParamSpec::new("separator", ParamType::Str, "the special token every document is followed by (default `<|endoftext|>`; the vocabulary must hold it)"))
        .param(ParamSpec::new("strict_updates", ParamType::Bool, "abort at the first optimiser step whose loss or gradient is not finite (off: such a step is skipped)").default(json!(false)))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the window draws (random when omitted; the report carries the one used)").min(0.0).max(capability::bounds::EXACT_INTEGER))
        .param(ParamSpec::new("arch", ParamType::Str, "the base's architecture when its checkpoint does not tell it"))
        .param(ParamSpec::new("install_as", ParamType::Str, "also install the checkpoint in the server's model store as `local/<this name>` (letters, digits, `.`, `_`, `-`; it must not exist yet)"))
        .input(BlobSpec::new("dataset", Media::Bytes, "the training corpus (JSONL)"))
        .input(BlobSpec::new("held_out", Media::Bytes, "the held-out corpus (JSONL)"))
        .output(BlobSpec::new("report", Media::Text, "the report (JSON) when the server has no artifact store to publish a checkpoint to"))
}

/// The scalar settings, exactly the params the spec declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub base: String,
    pub data: String,
    pub steps: u32,
    pub batch: u32,
    pub block: u32,
    pub grad_accum: u32,
    pub lr: Option<f32>,
    pub min_lr: Option<f32>,
    pub warmup: Option<u32>,
    pub weight_decay: Option<f32>,
    pub grad_clip: Option<f32>,
    pub eval_every: u32,
    pub eval_batches: u32,
    pub separator: Option<String>,
    pub strict_updates: bool,
    pub seed: Option<u64>,
    pub arch: Option<String>,
    pub install_as: Option<String>,
}

/// A request that passed the spec and every check that needs no file and no device.
#[derive(Debug)]
pub struct Request {
    pub params: Params,
    pub data: DataRefs,
}

impl Request {
    /// Reads the validated invocation `inv`: a `data` document naming more than a training and a held-out corpus is
    /// refused by name, as is an evaluation with nothing to evaluate.
    pub fn from_invocation(inv: &Invocation) -> Result<Request, String> {
        let params: Params = serde_json::from_value(inv.params.clone()).map_err(|e| format!("{ACTION}: {e}"))?;
        let data = DataRefs::parse(&params.data)?;
        let unsupported: Vec<&str> = [
            (data.held_out_text.is_some(), "held_out_text"),
            (data.monitor.is_some(), "monitor"),
            (data.start_adapter.is_some(), "start_adapter"),
            (!data.replay.is_empty(), "replay"),
            (!data.weighted_replay.is_empty(), "weighted_replay"),
        ]
        .into_iter()
        .filter_map(|(named, name)| named.then_some(name))
        .collect();
        if !unsupported.is_empty() {
            return Err(format!("{ACTION}: {} are not corpora of pretraining (it takes train and held_out)", unsupported.join(", ")));
        }
        if params.eval_every > 0 && data.held_out.is_none() {
            return Err(format!("{ACTION}: eval_every needs a held_out corpus"));
        }
        if let Some(sep) = &params.separator {
            if sep.is_empty() || sep.len() > MAX_SEPARATOR_BYTES {
                return Err(format!("{ACTION}: separator must be 1 to {MAX_SEPARATOR_BYTES} bytes"));
            }
        }
        if let (Some(lr), Some(min)) = (params.lr, params.min_lr) {
            if min > lr {
                return Err(format!("{ACTION}: min_lr {min} is above lr {lr}"));
            }
        }
        if let Some(name) = &params.install_as {
            check_install_name(name)?;
        }
        check_attached(&data, inv)?;
        Ok(Request { params, data })
    }

    /// The SDK run over the staged corpora, every setting applied.
    pub fn pretrain(&self, base_dir: &std::path::Path, staged: &job::Staged, out_dir: &std::path::Path) -> Result<brain::Pretrain, String> {
        let p = &self.params;
        let mut run = brain::Pretrain::from_pretrained(utf8(base_dir)?)
            .placed(true)
            .corpus(&staged.train)
            .out_dir(out_dir)
            .steps(p.steps)
            .batch(p.batch)
            .block(p.block)
            .grad_accum(p.grad_accum)
            .eval_every(p.eval_every)
            .eval_batches(p.eval_batches)
            .strict_updates(p.strict_updates);
        if let Some(path) = &staged.held_out {
            run = run.held_out(path);
        }
        if let Some(arch) = &p.arch {
            run = run.arch(arch.as_str());
        }
        if let Some(v) = p.lr {
            run = run.lr(v);
        }
        if let Some(v) = p.min_lr {
            run = run.min_lr(v);
        }
        if let Some(v) = p.warmup {
            run = run.warmup(v);
        }
        if let Some(v) = p.weight_decay {
            run = run.weight_decay(v);
        }
        if let Some(v) = p.grad_clip {
            run = run.grad_clip(v);
        }
        if let Some(v) = &p.separator {
            run = run.separator(v.as_str());
        }
        if let Some(v) = p.seed {
            run = run.seed(v);
        }
        Ok(run)
    }
}

/// The card bytes a run of `request` on the base in `base_dir` reserves: the full-parameter footprint at the tokens
/// of one micro-batch.
pub fn footprint(request: &Request, base_dir: &std::path::Path) -> Result<brain::TrainFootprint, String> {
    let p = &request.params;
    let mut ft = brain::ChatFullFineTune::from_pretrained(utf8(base_dir)?);
    if let Some(arch) = &p.arch {
        ft = ft.arch(arch.as_str());
    }
    ft.footprint(p.block.saturating_mul(p.batch)).map_err(|e| format!("{ACTION}: {e}"))
}

/// Runs the `pretrain` job `inv` on the device this thread is scoped to.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let request = Request::from_invocation(inv)?;
    let p = &request.params;
    let base_dir = job::resolve_base(env.models_dir.as_deref(), &p.base)?;
    let work = Workdir::create(env)?;
    progress(Progress::event(0, p.steps, json!({"type": "stage", "stage": "staging the corpora"})));
    let staged = job::stage(&request.data, &inv.blobs, env.store.as_ref(), &work)?;
    inv.cancel.check()?;
    let out_dir = work.path().join("run");
    let run = request.pretrain(&base_dir, &staged, &out_dir)?;
    progress(Progress::event(0, p.steps, json!({"type": "stage", "stage": "encoding the corpora, loading the base and training"})));
    let began = Instant::now();
    let outcome = run.run_with(&inv.cancel, |step| progress(Progress::event(step.step, step.steps, job::step_frame(step, began.elapsed().as_secs_f64())))).map_err(trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err(capability::failure::CANCELLED.to_string());
    }
    let checkpoint = outcome.checkpoint.clone().ok_or("pretrain: the run completed with no checkpoint")?;
    let report = report(&outcome);
    work.ensure_within_quota()?;
    let mut result = checkpoint_outcome(env, &checkpoint, p.install_as.as_deref(), report)?;
    if env.store.is_none() {
        let text = result.outputs.to_string();
        result = result.blob("report", Blob::new(Media::Text, text.into_bytes()));
    }
    Ok(result)
}

/// The report of a completed run.
pub(crate) fn report(o: &brain::PretrainOutcome) -> serde_json::Value {
    json!({
        "status": "completed",
        "objective": "pretrain",
        "weights_sha256": o.weights_digest,
        "base_digest": o.base_digest,
        "seed": o.seed,
        "steps": o.steps,
        "steps_completed": o.steps_completed,
        "resumed_at": o.resumed_at,
        "initial_loss": o.initial_loss,
        "final_loss": o.final_loss,
        "evaluations": o.evaluations.iter().map(|(step, train, eval)| json!({"step": step, "train_loss": train, "eval_loss": eval})).collect::<Vec<_>>(),
        "documents": o.documents,
        "tokens": o.tokens,
        "held_out_documents": o.held_out_documents,
        "held_out_tokens": o.held_out_tokens,
        "separator": o.separator,
        "separator_id": o.separator_id,
        "batch": o.batch,
        "block": o.block,
        "lr": o.lr,
    })
}
