// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that run a fine-tune beside
// the models being served on the same cards without taking the server down, for
// its clients. If your team needs expertise in scheduling training and serving
// on shared accelerators, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One training job, from a validated invocation to its outcome: the base is
//! resolved in the model store, the datasets are staged into a private
//! directory (copied from the call's blobs, or fetched and verified from the
//! artifact store), [`brain::ChatFineTune`] runs, and the adapter leaves by
//! digest in the artifact store and, when asked, as an output blob.
//!
//! Nothing here chooses a path for a caller: every file the trainer opens is in
//! the job's own directory, and the only names a caller supplies are a model
//! id the model store resolves, digests, and the names of the call's blobs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use capability::{ActionResult, Blob, Invocation, Media, Outcome, Progress};
use serde_json::{json, Value};

use crate::request::{DataRefs, Objective, Request, Source};
use crate::store::{ArtifactStore, Digest, StoreError};

/// The most bytes of one dataset or starting adapter a job takes.
pub const MAX_INPUT_BYTES: u64 = 8 << 30;

/// What a job needs from the host it runs on: where the model store and the artifact store are, and
/// where it may keep its own files.
#[derive(Clone, Debug)]
pub struct JobEnv {
    pub store: Option<ArtifactStore>,
    pub models_dir: Option<PathBuf>,
    /// Where each job makes its private directory.
    pub work_root: PathBuf,
}

/// The directory the jobs of this user make their private directories in: `brain-train-<uid>` under the
/// temporary directory (`TMPDIR`). Windows gives each user a temporary directory of their own, so there it is
/// plain `brain-train`.
pub fn default_work_root() -> PathBuf {
    #[cfg(unix)]
    // SAFETY: `geteuid` has no preconditions.
    let name = format!("brain-train-{}", unsafe { libc::geteuid() });
    #[cfg(not(unix))]
    let name = "brain-train".to_string();
    std::env::temp_dir().join(name)
}

/// A job's private directory, removed when the job ends however it ends.
pub struct Workdir(PathBuf);

static JOBS: AtomicU64 = AtomicU64::new(0);

impl Workdir {
    pub fn create(root: &Path) -> Result<Workdir, String> {
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.recursive(true).create(root).map_err(|e| format!("train: cannot make the work directory {}: {e}", root.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // The root may sit in a directory other users write to: what is there must be this process's own
            // plain directory, not a link or a directory someone else made and can write to.
            let meta = std::fs::symlink_metadata(root).map_err(|e| format!("train: cannot inspect the work directory {}: {e}", root.display()))?;
            // SAFETY: `geteuid` has no preconditions.
            let me = unsafe { libc::geteuid() };
            if !meta.is_dir() || meta.uid() != me || meta.mode() & 0o022 != 0 {
                return Err(format!("train: the work directory {} must be a directory of this user that nobody else can write to", root.display()));
            }
        }
        let dir = root.join(format!("job-{}-{}", std::process::id(), JOBS.fetch_add(1, Ordering::Relaxed)));
        builder.recursive(false).create(&dir).map_err(|e| format!("train: cannot make the job directory {}: {e}", dir.display()))?;
        Ok(Workdir(dir))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Workdir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The datasets of a job as files in its directory.
#[derive(Debug, Default)]
pub struct Staged {
    pub train: PathBuf,
    pub held_out: Option<PathBuf>,
    pub held_out_text: Option<PathBuf>,
    pub monitor: Option<PathBuf>,
    pub start_adapter: Option<PathBuf>,
    pub replay: Vec<PathBuf>,
    pub weighted_replay: Vec<(PathBuf, f32)>,
}

fn store_failure(e: &StoreError) -> String {
    // The operator's log keeps the cause; the caller is told what it may know.
    if let StoreError::Io(_) = e {
        eprintln!("brain train: {e}");
    }
    format!("train: {}", e.client_message())
}

/// Puts every dataset of `data` into `dir` as a file of its own: input blobs are copied, digests are
/// fetched from `store` and verified as they are copied.
pub fn stage(data: &DataRefs, blobs: &BTreeMap<String, Blob>, store: Option<&ArtifactStore>, dir: &Path) -> Result<Staged, String> {
    let mut count = 0usize;
    let mut bring = |source: &Source| -> Result<PathBuf, String> {
        count += 1;
        let dest = dir.join(format!("input-{count}"));
        match source {
            Source::Blob(name) => {
                let blob = blobs.get(name).ok_or_else(|| format!("train: the input blob \"{name}\" is not attached"))?;
                if blob.bytes.len() as u64 > MAX_INPUT_BYTES {
                    return Err(format!("train: the input blob \"{name}\" is larger than {MAX_INPUT_BYTES} bytes"));
                }
                std::fs::write(&dest, &blob.bytes).map_err(|e| format!("train: staging \"{name}\": {e}"))?;
            }
            Source::Digest(digest) => {
                let store = store.ok_or_else(|| "train: this server has no artifact store (--artifact-store), so a dataset cannot be named by digest".to_string())?;
                store.fetch(digest, &dest, MAX_INPUT_BYTES).map_err(|e| store_failure(&e))?;
            }
        }
        Ok(dest)
    };
    Ok(Staged {
        train: bring(&data.train)?,
        held_out: data.held_out.as_ref().map(&mut bring).transpose()?,
        held_out_text: data.held_out_text.as_ref().map(&mut bring).transpose()?,
        monitor: data.monitor.as_ref().map(&mut bring).transpose()?,
        start_adapter: data.start_adapter.as_ref().map(&mut bring).transpose()?,
        replay: data.replay.iter().map(&mut bring).collect::<Result<_, _>>()?,
        weighted_replay: data.weighted_replay.iter().map(|(s, share)| bring(s).map(|p| (p, *share))).collect::<Result<_, _>>()?,
    })
}

/// The directory of `base` in the model store, or why it is not there. `base` must be a plain
/// `<vendor>/<repo>` reference: it is parsed by the one parser that refuses separators, dot segments and
/// adapter suffixes, and resolved against the store, never against the filesystem as given.
pub fn resolve_base(models_dir: Option<&Path>, base: &str) -> Result<PathBuf, String> {
    let reference = brain_modelref::ModelRef::parse(base).map_err(|e| format!("train: base \"{base}\": {e}"))?;
    if reference.adapter().is_some() {
        return Err(format!("train: base \"{base}\" names an adapter; a base is a plain <vendor>/<repo>"));
    }
    let root = models_dir.ok_or("train: this server has no model store to resolve the base in")?;
    brain_modelstore::Store::new(root).local(&reference).map(|local| local.dir).ok_or_else(|| format!("train: base \"{base}\" is not in the model store"))
}

/// The footprints already worked out, by what they depend on, so the door check and the placement ask
/// the checkpoint's headers once.
#[derive(Default)]
pub struct FootprintCache(Mutex<BTreeMap<FootprintKey, Result<brain::TrainFootprint, String>>>);

/// The most footprints kept at once (a server trains a handful of bases).
const MAX_FOOTPRINTS: usize = 64;

/// What a footprint depends on: the base, the row length, the architecture and format named, the card.
type FootprintKey = (PathBuf, u32, Option<String>, Option<String>, Option<u64>);

impl FootprintCache {
    pub fn of(&self, base_dir: &Path, request: &Request, card_bytes: Option<u64>) -> Result<brain::TrainFootprint, String> {
        let key = (base_dir.to_path_buf(), request.params.max_block, request.params.arch.clone(), request.params.base_format.clone(), card_bytes);
        if let Some(known) = self.0.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return known.clone();
        }
        let computed = fine_tune_for_footprint(base_dir, request).and_then(|ft| ft.footprint(request.params.max_block, card_bytes).map_err(|e| format!("train: {e}")));
        // Only a footprint that was worked out is kept, and only so many: the
        // key carries strings a caller chose, and a refusal is cheap to redo,
        // so neither a stream of invented architectures nor of failures can
        // grow the table without end.
        if computed.is_ok() {
            let mut known = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if known.len() >= MAX_FOOTPRINTS {
                known.clear();
            }
            known.insert(key, computed.clone());
        }
        computed
    }
}

fn utf8(path: &Path) -> Result<&str, String> {
    path.to_str().ok_or_else(|| format!("train: {} is not valid UTF-8", path.display()))
}

fn fine_tune_for_footprint(base_dir: &Path, request: &Request) -> Result<brain::ChatFineTune, String> {
    let mut ft = brain::ChatFineTune::from_pretrained(utf8(base_dir)?);
    if let Some(arch) = &request.params.arch {
        ft = ft.arch(arch.as_str());
    }
    if let Some(format) = &request.params.base_format {
        ft = ft.base_format(format.as_str());
    }
    Ok(ft)
}

fn policy_gradient_config(params: &crate::request::Params) -> Result<brain::PolicyGradientConfig, String> {
    let reference = |name: &str| match name {
        "base" => Ok(brain::ReferencePolicy::Base),
        "start" => Ok(brain::ReferencePolicy::StartOfRun),
        other => Err(format!("train: {other:?} is not a reference policy (base or start)")),
    };
    let mut config = brain::PolicyGradientConfig::default();
    if let Some(v) = params.clip_eps {
        config.clip_eps = v;
    }
    config.clip_eps_high = params.clip_eps_high.or(config.clip_eps_high);
    if let Some(v) = &params.importance {
        config.importance = v.parse().map_err(|e| format!("train: importance: {e}"))?;
    }
    config.max_step_kl = params.max_step_kl.or(config.max_step_kl);
    config.max_step_grad_norm = params.max_step_grad_norm.or(config.max_step_grad_norm);
    if let Some(v) = params.kl_beta {
        config.kl_beta = v;
    }
    if let Some(v) = &params.reference {
        config.reference = reference(v)?;
    }
    if let Some(v) = &params.behaviour {
        config.behaviour = v.parse().map_err(|e| format!("train: {e}"))?;
    }
    if let Some(v) = params.loss_agg.as_deref() {
        config.loss_agg = match v {
            "seq-mean" => brain::LossAgg::SeqMean,
            "token-sum" => brain::LossAgg::TokenSum,
            other => return Err(format!("train: {other:?} is not a loss aggregation (seq-mean or token-sum)")),
        };
    }
    if let Some(v) = params.temperature {
        config.temperature = v;
    }
    if let Some(v) = &params.teacher {
        config.teacher = reference(v)?;
    }
    config.advantage_bound = params.advantage_bound.or(config.advantage_bound);
    config.sequence_advantage_cap = params.sequence_advantage_cap.or(config.sequence_advantage_cap);
    Ok(config)
}

/// The trainer for `request` over the files `staged`, writing into `out_dir`, on the device this thread is scoped to.
pub fn build_fine_tune(request: &Request, base_dir: &Path, staged: &Staged, out_dir: &Path) -> Result<brain::ChatFineTune, String> {
    let p = &request.params;
    let mut ft = brain::ChatFineTune::from_pretrained(utf8(base_dir)?)
        .placed(true)
        .dataset(&staged.train)
        .out_dir(out_dir)
        .steps(p.steps)
        .grad_accum(p.grad_accum)
        .strict_updates(p.strict_updates)
        .thinking(p.thinking)
        .pack_answers(p.pack_answers)
        .max_block(p.max_block)
        .eval_every(p.eval_every)
        .patience(p.patience)
        // A monitored run exports its best evaluation whether or not its patience ever runs out.
        .keep_best(p.eval_every > 0);
    if p.objective == Objective::PolicyGradient {
        ft = ft.policy_gradient(policy_gradient_config(p)?);
    }
    if let Some(arch) = &p.arch {
        ft = ft.arch(arch.as_str());
    }
    if let Some(id) = &p.adapter_id {
        ft = ft.adapter_id(id.as_str());
    }
    if let Some(v) = p.rank {
        ft = ft.rank(v);
    }
    if let Some(v) = p.alpha {
        ft = ft.alpha(v);
    }
    if let Some(v) = p.seed {
        ft = ft.seed(v);
    }
    if let Some(v) = p.weight_decay {
        ft = ft.weight_decay(v);
    }
    if let Some(v) = p.lr {
        ft = ft.lr(v);
    }
    if let Some(v) = &p.base_format {
        ft = ft.base_format(v.as_str());
    }
    if let Some(v) = p.replay_share {
        ft = ft.replay_share(v);
    }
    if let Some(path) = &staged.held_out {
        ft = ft.held_out(path);
    }
    if let Some(path) = &staged.held_out_text {
        ft = ft.held_out_text(path);
    }
    if let Some(path) = &staged.monitor {
        ft = ft.monitor(path);
    }
    if let Some(path) = &staged.start_adapter {
        ft = ft.continue_from(path);
    }
    for path in &staged.replay {
        ft = ft.replay(path);
    }
    for (path, share) in &staged.weighted_replay {
        ft = ft.replay_at(path, *share);
    }
    if p.keep_evaluations {
        ft = ft.keep_evaluations(out_dir.join("evaluations"));
    }
    Ok(ft)
}

fn finite(v: f32) -> Value {
    json!(v)
}

/// The frame a step becomes on the wire: the step's loss and rate, and under `pg` what the policy-gradient
/// objective measured in it.
pub fn step_frame(progress: &brain::FineTuneProgress, seconds: f64) -> Value {
    let mut frame = json!({
        "type": "step",
        "step": progress.step,
        "steps": progress.steps,
        "loss": finite(progress.loss),
        "lr": finite(progress.lr),
        "grad_norm": progress.grad_norm.map(finite),
        "seconds": seconds,
    });
    let pg: serde_json::Map<String, Value> = progress.metrics.iter().filter_map(|(name, value)| name.strip_prefix("pg_").map(|n| (n.to_string(), finite(*value)))).collect();
    if !pg.is_empty() {
        frame["pg"] = Value::Object(pg);
    }
    frame
}

fn score_json(score: Option<brain::HeldOutScore>) -> Value {
    match score {
        Some(s) => json!({"loss": s.loss, "token_accuracy": s.token_accuracy, "positions": s.positions, "records": s.records, "skipped": s.skipped}),
        None => Value::Null,
    }
}

/// The report of a finished run: what the trainer measured, and where the adapter is. It rides one frame (a
/// D-Bus datagram, which is lost when it does not fit the socket), so it holds totals and short lists, every
/// one bounded by the request (see [`crate::request::MAX_EVALUATIONS`]); the per-step series are the frames
/// the run streamed and the `record` blob.
fn report(outcome: &brain::ChatFineTuneOutcome, adapter: Option<Digest>, record: Option<Digest>, evaluations: &[(u32, Digest)]) -> Value {
    // The per-step series is in the frames and in the record blob; the report keeps the run's totals.
    let policy_gradient = outcome.policy_gradient.as_ref().map(|pg| {
        let mut stats = serde_json::to_value(&pg.stats).unwrap_or(Value::Null);
        if let Some(fields) = stats.as_object_mut() {
            fields.remove("steps");
            fields.insert("steps_taken".into(), json!(pg.stats.steps.len()));
        }
        stats
    });
    json!({
        "status": "completed",
        "adapter_digest": adapter.map(|d| d.hex()),
        "record_digest": record.map(|d| d.hex()),
        "adapter_sha256": outcome.adapter_digest,
        "base_digest": outcome.base_digest,
        "trained_from": outcome.trained_from,
        "records": outcome.train_records,
        "replay_records": outcome.replay_records,
        "monitor_records": outcome.monitor_records,
        "block": outcome.block,
        "seed": outcome.seed,
        "steps": outcome.steps,
        "steps_completed": outcome.steps_completed,
        "rank": outcome.rank,
        "alpha": outcome.alpha,
        "lr": outcome.lr,
        "weight_decay": outcome.weight_decay,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
        "base_tier": outcome.base_tier,
        "base_tier_reason": outcome.base_tier_reason,
        "base_score": score_json(outcome.base_score),
        "tuned_score": score_json(outcome.tuned_score),
        "base_text_score": score_json(outcome.base_text_score),
        "tuned_text_score": score_json(outcome.tuned_text_score),
        "eval_every": outcome.eval_every,
        "patience": outcome.patience,
        "curve": outcome.curve.iter().map(|p| json!({"step": p.step, "train_loss": p.train_loss, "monitor_loss": p.monitor_loss})).collect::<Vec<_>>(),
        "selected_step": outcome.selected_step,
        "selection": outcome.selection,
        "stopped_early": outcome.stopped_early,
        "evaluations": evaluations.iter().map(|(step, digest)| json!({"step": step, "digest": digest.hex()})).collect::<Vec<_>>(),
        "policy_gradient": policy_gradient,
    })
}

/// Runs the job `inv` on the device this thread is scoped to, reporting each step as a frame.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let request = Request::from_invocation(inv)?;
    let base_dir = resolve_base(env.models_dir.as_deref(), &request.params.base)?;
    let steps = request.params.steps;
    let work = Workdir::create(&env.work_root)?;
    progress(Progress::event(0, steps, json!({"type": "stage", "stage": "staging the datasets"})));
    let staged = stage(&request.data, &inv.blobs, env.store.as_ref(), work.path())?;
    if inv.cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let out_dir = work.path().join("run");
    let fine_tune = build_fine_tune(&request, &base_dir, &staged, &out_dir)?;
    progress(Progress::event(0, steps, json!({"type": "stage", "stage": "loading the base and training"})));
    let began = Instant::now();
    let outcome = fine_tune
        .run_with(&inv.cancel, |step| progress(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64()))))
        .map_err(|e| format!("train: {e}"))?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".into());
    }
    let adapter_path = outcome.adapter.clone().ok_or("train: the run completed with no adapter")?;
    let mut result = Outcome::new();
    let mut digest = None;
    if let Some(store) = &env.store {
        let (published, _) = store.publish(&adapter_path).map_err(|e| store_failure(&e))?;
        digest = Some(published);
    }
    // The training record holds every step's statistics, which is bulk: it goes to the store beside the adapter.
    let mut record_digest = None;
    if let (Some(store), Some(record)) = (&env.store, &outcome.record) {
        record_digest = Some(store.publish(record).map_err(|e| store_failure(&e))?.0);
    }
    let mut evaluations = Vec::new();
    if let (Some(store), true) = (&env.store, request.params.keep_evaluations) {
        for point in &outcome.curve {
            let path = out_dir.join("evaluations").join(format!("step-{}.safetensors", point.step));
            let (published, _) = store.publish(&path).map_err(|e| store_failure(&e))?;
            evaluations.push((point.step, published));
        }
    }
    result.outputs = report(&outcome, digest, record_digest, &evaluations);
    if request.params.return_adapter {
        let bytes = std::fs::read(&adapter_path).map_err(|e| format!("train: reading the adapter: {e}"))?;
        result = result.blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"sha256": outcome.adapter_digest, "blake3": digest.map(|d| d.hex())})));
    }
    // Taken by blob when the caller takes everything by blob (or the server has no store to publish it to).
    if let (Some(record), true) = (&outcome.record, request.params.return_adapter || record_digest.is_none()) {
        result = result.blob("record", Blob::new(Media::Text, std::fs::read(record).map_err(|e| format!("train: reading the training record: {e}"))?));
    }
    Ok(result)
}
