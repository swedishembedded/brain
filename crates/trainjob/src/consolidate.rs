// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that consolidate what a
// model learned into its own weights without handing a checkpoint around by
// path, for its clients. If your team needs expertise in full-parameter
// fine-tuning and adapter merging as a service on shared accelerators, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The two jobs that produce a checkpoint instead of an adapter: `full_train`
//! (every weight of a chat model trains, [`brain::ChatFullFineTune`]) and
//! `merge` (a LoRA adapter folded into the weights it targets,
//! [`brain::merge_adapter`]).
//!
//! A checkpoint is a directory, and an artifact of the store is one file, so a
//! checkpoint leaves as a **manifest**: each of its files is published to the
//! artifact store by blake3 digest, and the manifest (a small JSON document
//! listing `name`, `digest` and `bytes` of each) is published too; its digest,
//! `checkpoint_digest`, names the checkpoint. Optionally the checkpoint is also
//! installed in the server's model store as `local/<install_as>`, where a later
//! job names it as its `base` and the server can serve it. Nothing here takes a
//! path from a caller: the names are a model id, digests, the names of the
//! call's blobs, and one plain model-store name.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::{self, bring, store_failure, trainer_failure, utf8, JobEnv, Workdir};
use crate::request::{check_attached, DataRefs, Source};
use crate::store::{ArtifactStore, Digest};

/// The full-parameter action's name.
pub const FULL_ACTION: &str = "full_train";
/// The adapter-merge action's name.
pub const MERGE_ACTION: &str = "merge";
/// The longest model-store name `install_as` may take.
pub const MAX_INSTALL_NAME_LEN: usize = 100;
/// The most files a checkpoint directory may hold and still be published: a model is a handful of files, an ensemble
/// of timeline models a few per member.
pub const MAX_CHECKPOINT_FILES: usize = 1024;
/// How many directories deep a checkpoint's files may sit (a model, or an ensemble's members beside it).
pub const MAX_CHECKPOINT_DEPTH: usize = 3;
/// The format named in a checkpoint manifest.
pub const MANIFEST_FORMAT: &str = "brain-checkpoint-manifest-v1";

static INSTALLS: AtomicU64 = AtomicU64::new(0);

fn install_param() -> ParamSpec {
    ParamSpec::new("install_as", ParamType::Str, "also install the checkpoint in the server's model store as `local/<this name>` (letters, digits, `.`, `_`, `-`; it must not exist yet), where a later job names it as its base")
}

/// What `full_train` is.
pub fn full_spec() -> ActionSpec {
    let optional_f = |name: &str, help: &str, min: f64, max: f64| ParamSpec::new(name, ParamType::Float, help).min(min).max(max);
    let blob = |name: &str, help: &str| BlobSpec::new(name, Media::Bytes, help);
    let mut spec = ActionSpec::new(FULL_ACTION, "fine-tune every weight of a chat model on a chat dataset: the checkpoint leaves by manifest digest in the artifact store (and optionally into the model store)")
        .training()
        .streaming()
        .param(ParamSpec::new("base", ParamType::Str, "the base model, `<vendor>/<repo>` in the server's model store").required())
        .param(ParamSpec::new("data", ParamType::Str, "JSON document naming the datasets: {\"train\": SRC, \"replay\"?: [SRC], \"weighted_replay\"?: [{SRC fields, \"share\"}]} where SRC is {\"digest\": \"<blake3 hex>\"} or {\"blob\": \"<input name>\"}").required())
        .param(ParamSpec::new("steps", ParamType::Int, "optimiser steps").default(json!(100)).min(1.0).max(1_000_000.0).step(1.0))
        .param(optional_f("lr", "peak learning rate (default 1e-5)", 0.0, 1.0))
        .param(optional_f("weight_decay", "AdamW weight decay (default 0)", 0.0, 1.0))
        .param(ParamSpec::new("grad_accum", ParamType::Int, "records averaged into one optimiser step").default(json!(1)).min(1.0).max(4096.0).step(1.0))
        .param(ParamSpec::new("strict_updates", ParamType::Bool, "abort at the first optimiser step whose loss or gradient is not finite (off: such a step is skipped)").default(json!(false)))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the batch order (random when omitted; the report carries the one used)").min(0.0))
        .param(ParamSpec::new("max_block", ParamType::Int, "the longest training row in tokens: the memory reserved for the run is sized to it, and a longer record is refused by name").default(json!(8192)).min(64.0).max(1_048_576.0).step(1.0))
        .param(ParamSpec::new("thinking", ParamType::Bool, "train the model to reason before it answers (off: trained for no-think mode)").default(json!(true)))
        .param(ParamSpec::new("arch", ParamType::Str, "the base's architecture when its directory carries no config.json to tell it by"))
        .param(optional_f("replay_share", "the share of draws that come from the replay sets in all, in (0, 1)", 0.0, 1.0))
        .param(ParamSpec::new("train_precision", ParamType::Str, "what each trainable parameter, its gradient and its optimiser state are stored as, `key=value,...` (e.g. `state=int8,sr=on`); the default is fp32 everywhere. An unknown key or value is refused before anything loads"))
        .param(install_param())
        .input(blob("dataset", "the training records (`generic-messages-v2` JSONL)"));
    for name in crate::request::REPLAY_BLOBS {
        spec = spec.input(blob(name, "a replay dataset"));
    }
    spec.output(BlobSpec::new("report", Media::Text, "the report (JSON) when the server has no artifact store to publish a checkpoint to"))
}

/// What `merge` is.
pub fn merge_spec() -> ActionSpec {
    ActionSpec::new(MERGE_ACTION, "fold a LoRA adapter into the weights of its base: the merged checkpoint leaves by manifest digest in the artifact store (and optionally into the model store)")
        .training()
        .param(ParamSpec::new("base", ParamType::Str, "the base model the adapter was trained on, `<vendor>/<repo>` in the server's model store").required())
        .param(ParamSpec::new("adapter_digest", ParamType::Str, "the adapter, as a blake3 digest of the server's artifact store (or attach it as the input blob `adapter`; exactly one of the two)"))
        .param(install_param())
        .input(BlobSpec::new("adapter", Media::Bytes, "the adapter (safetensors) to fold in, instead of `adapter_digest`"))
}

/// The scalar settings of `full_train`, exactly the params its spec declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullParams {
    pub base: String,
    pub data: String,
    pub steps: u32,
    pub lr: Option<f32>,
    pub weight_decay: Option<f32>,
    pub grad_accum: u32,
    pub strict_updates: bool,
    pub seed: Option<u64>,
    pub max_block: u32,
    pub thinking: bool,
    pub arch: Option<String>,
    pub replay_share: Option<f32>,
    pub train_precision: Option<String>,
    pub install_as: Option<String>,
}

/// A `full_train` request that passed the spec and every check that needs no file and no device.
#[derive(Debug)]
pub struct FullRequest {
    pub params: FullParams,
    pub data: DataRefs,
}

impl FullRequest {
    /// Reads the validated invocation `inv`: a dataset document that names what a full fine-tune has no use for
    /// (held-out sets, a monitor, a starting adapter) is refused by name.
    pub fn from_invocation(inv: &Invocation) -> Result<FullRequest, String> {
        let params: FullParams = serde_json::from_value(inv.params.clone()).map_err(|e| format!("full_train: {e}"))?;
        let data = DataRefs::parse(&params.data)?;
        let unsupported: Vec<&str> = [
            (data.held_out.is_some(), "held_out"),
            (data.held_out_text.is_some(), "held_out_text"),
            (data.monitor.is_some(), "monitor"),
            (data.start_adapter.is_some(), "start_adapter"),
        ]
        .into_iter()
        .filter_map(|(named, name)| named.then_some(name))
        .collect();
        if !unsupported.is_empty() {
            return Err(format!("full_train: {} are not datasets of a full fine-tune (it takes train, replay and weighted_replay)", unsupported.join(", ")));
        }
        if let Some(name) = &params.install_as {
            check_install_name(name)?;
        }
        if let Some(spec) = &params.train_precision {
            brain::TrainPrecision::parse(spec).map_err(|e| format!("full_train: train_precision: {e}"))?;
        }
        check_attached(&data, inv)?;
        Ok(FullRequest { params, data })
    }
}

/// The scalar settings of `merge`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeParams {
    pub base: String,
    pub adapter_digest: Option<String>,
    pub install_as: Option<String>,
}

/// A `merge` request that passed the spec and every check that needs no file and no device.
#[derive(Debug)]
pub struct MergeRequest {
    pub params: MergeParams,
    pub adapter: Source,
}

impl MergeRequest {
    /// Reads the validated invocation `inv`: the adapter is the input blob `adapter` or `adapter_digest`, exactly one.
    pub fn from_invocation(inv: &Invocation) -> Result<MergeRequest, String> {
        let params: MergeParams = serde_json::from_value(inv.params.clone()).map_err(|e| format!("merge: {e}"))?;
        let attached = inv.blobs.contains_key("adapter");
        let adapter = match (&params.adapter_digest, attached) {
            (Some(digest), false) => Source::Digest(Digest::parse(digest).map_err(|e| format!("merge: adapter_digest: {e}"))?),
            (None, true) => Source::Blob("adapter".to_string()),
            _ => return Err("merge: name the adapter exactly once: as the input blob `adapter` or as `adapter_digest`".to_string()),
        };
        if let Some(extra) = inv.blobs.keys().find(|k| k.as_str() != "adapter") {
            return Err(format!("merge: the input blob \"{extra}\" is not an input of this action"));
        }
        if let Some(name) = &params.install_as {
            check_install_name(name)?;
        }
        Ok(MergeRequest { params, adapter })
    }
}

/// `install_as` is one plain model-store name under the reserved `local` vendor: it parses as a model reference
/// there (the one parser that refuses separators and dot segments) and is no adapter.
pub(crate) fn check_install_name(name: &str) -> Result<(), String> {
    let plain = !name.is_empty() && name.len() <= MAX_INSTALL_NAME_LEN && !name.starts_with('.') && name.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    let parsed = brain_modelref::ModelRef::parse(&format!("local/{name}")).ok().filter(|r| r.adapter().is_none() && r.vendor() == "local" && r.repo() == name);
    if plain && parsed.is_some() {
        Ok(())
    } else {
        Err(format!("install_as: \"{name}\" is not a model-store name (1 to {MAX_INSTALL_NAME_LEN} letters, digits, `.`, `_`, `-`, not starting with a dot)"))
    }
}

/// The files of the checkpoint directory `dir`, by path relative to it (`/`-separated): regular files, at most
/// [`MAX_CHECKPOINT_DEPTH`] directories deep, at most [`MAX_CHECKPOINT_FILES`] of them. Anything else (a link, a device)
/// refuses the checkpoint.
fn checkpoint_files(dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    fn walk(root: &Path, at: &Path, depth: usize, files: &mut Vec<(String, PathBuf)>) -> Result<(), String> {
        for entry in std::fs::read_dir(at).map_err(|e| format!("train: reading the checkpoint: {e}"))? {
            let entry = entry.map_err(|e| format!("train: reading the checkpoint: {e}"))?;
            let meta = std::fs::symlink_metadata(entry.path()).map_err(|e| format!("train: reading the checkpoint: {e}"))?;
            let relative = entry.path().strip_prefix(root).map_or_else(|_| entry.file_name().to_string_lossy().into_owned(), |r| r.to_string_lossy().replace('\\', "/"));
            if meta.is_file() {
                files.push((relative, entry.path()));
            } else if meta.is_dir() && depth < MAX_CHECKPOINT_DEPTH {
                walk(root, &entry.path(), depth + 1, files)?;
            } else {
                return Err(format!("train: the checkpoint holds \"{relative}\", which is not a plain file or a directory within {MAX_CHECKPOINT_DEPTH} levels"));
            }
            if files.len() > MAX_CHECKPOINT_FILES {
                return Err(format!("train: the checkpoint holds more than {MAX_CHECKPOINT_FILES} files"));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, 0, &mut files)?;
    files.sort();
    Ok(files)
}

/// Publishes every file of `dir` and then the manifest naming them; the manifest's digest names the checkpoint.
fn publish_checkpoint(store: &ArtifactStore, dir: &Path) -> Result<(Digest, Value), String> {
    let mut listed = Vec::new();
    for (name, path) in checkpoint_files(dir)? {
        let (digest, bytes) = store.publish(&path).map_err(|e| store_failure(&e))?;
        listed.push(json!({"name": name, "digest": digest.hex(), "bytes": bytes}));
    }
    let manifest = json!({"format": MANIFEST_FORMAT, "files": listed});
    let digest = store.publish_bytes(manifest.to_string().as_bytes()).map_err(|e| store_failure(&e))?;
    Ok((digest, manifest))
}

/// Installs the checkpoint `dir` as `local/<name>` in the model store `models_dir`. The model store never gains a
/// half-written model and never loses one: the files go to a hidden directory beside the target, which is renamed
/// into place, and a name that is taken is refused.
fn install(models_dir: &Path, name: &str, dir: &Path) -> Result<String, String> {
    let vendor_dir = models_dir.join("local");
    let target = vendor_dir.join(name);
    std::fs::create_dir_all(&vendor_dir).map_err(|e| format!("train: cannot make the model store's local directory: {e}"))?;
    if std::fs::symlink_metadata(&target).is_ok() {
        return Err(format!("train: install_as: local/{name} is already in the model store"));
    }
    let partial = vendor_dir.join(format!(".installing-{}-{}", std::process::id(), INSTALLS.fetch_add(1, Ordering::Relaxed)));
    let result = (|| {
        std::fs::create_dir(&partial).map_err(|e| format!("train: cannot stage the installation: {e}"))?;
        for (file, path) in checkpoint_files(dir)? {
            let to = partial.join(&file);
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("train: installing \"{file}\": {e}"))?;
            }
            std::fs::hard_link(&path, &to).or_else(|_| std::fs::copy(&path, &to).map(drop)).map_err(|e| format!("train: installing \"{file}\": {e}"))?;
        }
        std::fs::rename(&partial, &target).map_err(|e| format!("train: install_as: local/{name} could not be put in place: {e}"))
    })();
    if result.is_err() {
        std::fs::remove_dir_all(&partial).ok();
    }
    result.map(|()| format!("local/{name}"))
}

/// What a finished checkpoint job returns: its report, with the checkpoint's manifest digest and where it was installed.
pub(crate) fn checkpoint_outcome(env: &JobEnv, dir: &Path, install_as: Option<&str>, mut report: Value) -> ActionResult {
    let mut result = Outcome::new();
    if let Some(store) = &env.store {
        let (digest, manifest) = publish_checkpoint(store, dir)?;
        report["checkpoint_digest"] = json!(digest.hex());
        report["files"] = manifest["files"].clone();
    }
    if let Some(name) = install_as {
        let models_dir = env.models_dir.as_deref().ok_or("train: this server has no model store to install into")?;
        report["installed_as"] = json!(install(models_dir, name, dir)?);
    }
    result.outputs = report;
    Ok(result)
}

/// Runs the `full_train` job `inv` on the device this thread is scoped to.
pub fn run_full(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let request = FullRequest::from_invocation(inv)?;
    let p = &request.params;
    let base_dir = job::resolve_base(env.models_dir.as_deref(), &p.base)?;
    let work = Workdir::create(&env.work_root)?;
    progress(Progress::event(0, p.steps, json!({"type": "stage", "stage": "staging the datasets"})));
    let staged = job::stage(&request.data, &inv.blobs, env.store.as_ref(), work.path())?;
    if inv.cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let out_dir = work.path().join("run");
    let mut ft = brain::ChatFullFineTune::from_pretrained(utf8(&base_dir)?)
        .placed(true)
        .dataset(&staged.train)
        .out_dir(&out_dir)
        .steps(p.steps)
        .grad_accum(p.grad_accum)
        .strict_updates(p.strict_updates)
        .thinking(p.thinking)
        .max_block(p.max_block);
    if let Some(arch) = &p.arch {
        ft = ft.arch(arch.as_str());
    }
    if let Some(v) = p.lr {
        ft = ft.lr(v);
    }
    if let Some(v) = p.weight_decay {
        ft = ft.weight_decay(v);
    }
    if let Some(v) = p.seed {
        ft = ft.seed(v);
    }
    if let Some(v) = p.replay_share {
        ft = ft.replay_share(v);
    }
    if let Some(spec) = &p.train_precision {
        ft = ft.train_precision(brain::TrainPrecision::parse(spec).map_err(|e| format!("full_train: train_precision: {e}"))?);
    }
    for path in &staged.replay {
        ft = ft.replay(path);
    }
    for (path, share) in &staged.weighted_replay {
        ft = ft.replay_at(path, *share);
    }
    progress(Progress::event(0, p.steps, json!({"type": "stage", "stage": "loading the base and training"})));
    let began = Instant::now();
    let outcome = ft.run_with(&inv.cancel, |step| progress(Progress::event(step.step, step.steps, job::step_frame(step, began.elapsed().as_secs_f64())))).map_err(trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".into());
    }
    let checkpoint = outcome.checkpoint.clone().ok_or("train: the run completed with no checkpoint")?;
    let report = json!({
        "status": "completed",
        "objective": "full_sft",
        "weights_sha256": outcome.weights_digest,
        "base_digest": outcome.base_digest,
        "records": outcome.train_records,
        "replay_records": outcome.replay_records,
        "block": outcome.block,
        "seed": outcome.seed,
        "steps": outcome.steps,
        "steps_completed": outcome.steps_completed,
        "lr": outcome.lr,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
    });
    let mut result = checkpoint_outcome(env, &checkpoint, p.install_as.as_deref(), report)?;
    // A server with no artifact store has nowhere to publish a checkpoint; it can only install one.
    if env.store.is_none() {
        let text = result.outputs.to_string();
        result = result.blob("report", Blob::new(Media::Text, text.into_bytes()));
    }
    Ok(result)
}

/// Runs the `merge` job `inv`. Folding is host work: it takes no card.
pub fn run_merge(env: &JobEnv, inv: &Invocation) -> ActionResult {
    let request = MergeRequest::from_invocation(inv)?;
    let p = &request.params;
    let base_dir = job::resolve_base(env.models_dir.as_deref(), &p.base)?;
    let work = Workdir::create(&env.work_root)?;
    let adapter = work.path().join("adapter.safetensors");
    bring(&request.adapter, &inv.blobs, env.store.as_ref(), &adapter)?;
    if inv.cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let out_dir = work.path().join("merged");
    let merged = brain::merge_adapter(utf8(&base_dir)?, &adapter, &out_dir).map_err(trainer_failure)?;
    let report = json!({"status": "completed", "objective": "merge", "weights_sha256": merged.weights_digest});
    checkpoint_outcome(env, &merged.dir, p.install_as.as_deref(), report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validated(spec: &ActionSpec, params: Value, blobs: &[&str]) -> Result<Invocation, String> {
        let mut inv = Invocation::new();
        inv.params = params;
        for name in blobs {
            inv.blobs.insert((*name).to_string(), Blob::new(Media::Bytes, b"{}".to_vec()));
        }
        spec.validate_served(inv)
    }

    fn digest(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    #[test]
    fn a_full_fine_tune_takes_train_and_replay_and_refuses_the_datasets_it_has_no_use_for() {
        let data = json!({"train": {"digest": digest(1)}, "replay": [{"digest": digest(2)}]}).to_string();
        let ok = validated(&full_spec(), json!({"base": "a/b", "data": data, "install_as": "tuned-1"}), &[]).and_then(|i| FullRequest::from_invocation(&i)).unwrap();
        assert_eq!((ok.params.steps, ok.params.max_block, ok.params.install_as.as_deref()), (100, 8192, Some("tuned-1")));
        for field in ["held_out", "held_out_text", "monitor", "start_adapter"] {
            let doc = json!({"train": {"digest": digest(1)}, field: {"digest": digest(2)}}).to_string();
            let e = validated(&full_spec(), json!({"base": "a/b", "data": doc}), &[]).and_then(|i| FullRequest::from_invocation(&i)).unwrap_err();
            assert!(e.contains(field) && e.contains("full fine-tune"), "{field}: {e}");
        }
        for unknown in ["path", "out", "save", "weights", "objective", "rank"] {
            let e = validated(&full_spec(), json!({"base": "a/b", "data": data, unknown: "x"}), &[]).unwrap_err();
            assert!(e.contains("unknown param"), "{unknown}: {e}");
        }
    }

    #[test]
    fn a_merge_names_its_adapter_exactly_once_and_takes_no_other_input() {
        let spec = merge_spec();
        let by_digest = validated(&spec, json!({"base": "a/b", "adapter_digest": digest(1)}), &[]).and_then(|i| MergeRequest::from_invocation(&i)).unwrap();
        assert_eq!(by_digest.adapter, Source::Digest(Digest::parse(&digest(1)).unwrap()));
        let by_blob = validated(&spec, json!({"base": "a/b"}), &["adapter"]).and_then(|i| MergeRequest::from_invocation(&i)).unwrap();
        assert_eq!(by_blob.adapter, Source::Blob("adapter".into()));
        for (params, blobs) in [(json!({"base": "a/b"}), vec![]), (json!({"base": "a/b", "adapter_digest": digest(1)}), vec!["adapter"]), (json!({"base": "a/b", "adapter_digest": "../../etc/passwd"}), vec![])] {
            assert!(validated(&spec, params.clone(), &blobs).and_then(|i| MergeRequest::from_invocation(&i)).is_err(), "{params}");
        }
    }

    #[test]
    fn an_install_name_is_one_plain_model_store_name() {
        for good in ["tuned", "tuned-1", "a.b_c", &"a".repeat(MAX_INSTALL_NAME_LEN)] {
            assert!(check_install_name(good).is_ok(), "{good}");
        }
        for bad in ["", ".", "..", ".hidden", "a/b", "../x", "a b", "a:b:c:d", "/etc", "x\n", &"a".repeat(MAX_INSTALL_NAME_LEN + 1)] {
            assert!(check_install_name(bad).is_err(), "{bad:?}");
        }
    }
}
