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

use crate::quota::{Quotas, Staging};
use crate::request::{DataRefs, Objective, Request, Source, ACTION};
use crate::store::{ArtifactStore, StoreError};

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
    /// What a job may keep in that directory.
    pub quotas: Quotas,
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

/// A job's private directory, removed when the job ends however it ends (success, failure, cancellation or
/// panic), with the budget of bytes the job may keep in it.
pub struct Workdir {
    path: PathBuf,
    staging: Staging,
}

static JOBS: AtomicU64 = AtomicU64::new(0);

impl Workdir {
    pub fn create(env: &JobEnv) -> Result<Workdir, String> {
        let root = env.work_root.as_path();
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
        Ok(Workdir { path: dir, staging: Staging::new(env.quotas.max_job_staging_bytes) })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The budget the datasets staged into this directory are charged to.
    pub fn staging(&self) -> &Staging {
        &self.staging
    }

    /// Refuses a job whose directory (what the trainer wrote as well as what was staged) has outgrown its budget,
    /// before anything of it is published.
    pub fn ensure_within_quota(&self) -> Result<(), String> {
        if tree_bytes(&self.path) > self.staging.limit() {
            return Err(self.staging.over());
        }
        Ok(())
    }
}

impl Drop for Workdir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.path).ok();
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

pub(crate) fn store_failure(e: &StoreError) -> String {
    // The operator's log keeps the cause; the caller is told what it may know.
    if let StoreError::Io(_) = e {
        eprintln!("brain train: {e}");
    }
    format!("train: {}", e.client_message())
}

/// Puts the input `source` at the new file `dest`: an input blob is copied, a digest is fetched from `store` and
/// verified as it is copied. What lands there is charged to `staging`; an input that does not fit is refused
/// before anything of it is kept.
pub fn bring(source: &Source, blobs: &BTreeMap<String, Blob>, store: Option<&ArtifactStore>, staging: &Staging, dest: &Path) -> Result<(), String> {
    match source {
        Source::Blob(name) => {
            let blob = blobs.get(name).ok_or_else(|| format!("train: the input blob \"{name}\" is not attached"))?;
            if blob.bytes.len() as u64 > MAX_INPUT_BYTES {
                return Err(format!("train: the input blob \"{name}\" is larger than {MAX_INPUT_BYTES} bytes"));
            }
            staging.charge(blob.bytes.len() as u64)?;
            std::fs::write(dest, &blob.bytes).map_err(|e| format!("train: staging \"{name}\": {e}"))
        }
        Source::Digest(digest) => {
            let store = store.ok_or_else(|| "train: this server has no artifact store (--artifact-store), so a dataset cannot be named by digest".to_string())?;
            let limit = MAX_INPUT_BYTES.min(staging.remaining());
            let size = store.fetch(digest, dest, limit).map_err(|e| match e {
                StoreError::TooLarge { .. } if limit < MAX_INPUT_BYTES => staging.over(),
                other => store_failure(&other),
            })?;
            staging.charge(size)
        }
    }
}

/// Puts every dataset of `data` into `dir` as a file of its own: input blobs are copied, digests are
/// fetched from `store` and verified as they are copied.
pub fn stage(data: &DataRefs, blobs: &BTreeMap<String, Blob>, store: Option<&ArtifactStore>, work: &Workdir) -> Result<Staged, String> {
    let mut count = 0usize;
    let mut stage_one = |source: &Source| -> Result<PathBuf, String> {
        count += 1;
        let dest = work.path().join(format!("input-{count}"));
        bring(source, blobs, store, work.staging(), &dest)?;
        Ok(dest)
    };
    Ok(Staged {
        train: stage_one(&data.train)?,
        held_out: data.held_out.as_ref().map(&mut stage_one).transpose()?,
        held_out_text: data.held_out_text.as_ref().map(&mut stage_one).transpose()?,
        monitor: data.monitor.as_ref().map(&mut stage_one).transpose()?,
        start_adapter: data.start_adapter.as_ref().map(&mut stage_one).transpose()?,
        replay: data.replay.iter().map(&mut stage_one).collect::<Result<_, _>>()?,
        weighted_replay: data.weighted_replay.iter().map(|(s, share)| stage_one(s).map(|p| (p, *share))).collect::<Result<_, _>>()?,
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

/// What a footprint depends on: the base, the training rows one step holds, the architecture and format named, the
/// adapter's rank, the card.
type FootprintKey = (PathBuf, u32, Option<String>, Option<String>, Option<u32>, Option<u64>);

impl FootprintCache {
    pub fn of(&self, base_dir: &Path, request: &Request, card_bytes: Option<u64>) -> Result<brain::TrainFootprint, String> {
        // The rows one step holds: a preference step holds both candidates of a pair.
        let rows = request.params.max_block.saturating_mul(request.params.objective.rows_per_step());
        let key = (base_dir.to_path_buf(), rows, request.params.arch.clone(), request.params.base_format.clone(), request.params.rank, card_bytes);
        if let Some(known) = self.0.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return known.clone();
        }
        let computed = fine_tune_for_footprint(base_dir, request).and_then(|ft| ft.footprint(rows, card_bytes).map_err(|e| format!("train: {e}")));
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

/// The bytes of the files under `dir` (one level of subdirectories deep): what reading a checkpoint costs.
pub(crate) fn tree_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .flatten()
        .filter_map(|entry| entry.metadata().ok().map(|m| if m.is_dir() { tree_bytes(&entry.path()) } else { m.len() }))
        .sum()
}

pub(crate) fn utf8(path: &Path) -> Result<&str, String> {
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
    if let Some(rank) = request.params.rank {
        ft = ft.rank(rank);
    }
    Ok(ft)
}

/// The algorithm a request names, if any.
pub(crate) fn algorithm(params: &crate::request::Params) -> Result<Option<brain::PolicyGradientAlgorithm>, String> {
    params.algorithm.as_deref().map(brain::PolicyGradientAlgorithm::parse).transpose().map_err(|e| format!("train: {e}"))
}

/// The configuration a policy-gradient request trains with: its algorithm's (the trainer's default without one), with
/// every setting the request names over it.
pub(crate) fn policy_gradient_config(params: &crate::request::Params) -> Result<brain::PolicyGradientConfig, String> {
    let mut config = algorithm(params)?.map_or_else(brain::PolicyGradientConfig::default, |a| a.config(params.max_block));
    if let Some(v) = params.clip_eps {
        config.clip_eps = v;
    }
    config.clip_eps_high = params.clip_eps_high.or(config.clip_eps_high);
    config.dual_clip = params.dual_clip.or(config.dual_clip);
    if let Some(v) = &params.importance {
        config.importance = v.parse().map_err(|e| format!("train: importance: {e}"))?;
    }
    config.max_step_kl = params.max_step_kl.or(config.max_step_kl);
    config.max_step_grad_norm = params.max_step_grad_norm.or(config.max_step_grad_norm);
    if let Some(v) = params.kl_beta {
        config.kl_beta = v;
    }
    if let Some(v) = &params.reference {
        config.reference = v.parse().map_err(|e| format!("train: reference: {e}"))?;
    }
    if let Some(v) = &params.behaviour {
        config.behaviour = v.parse().map_err(|e| format!("train: {e}"))?;
    }
    config.loss_agg = match (params.loss_agg.as_deref(), params.loss_agg_length) {
        // The constant is its own setting on the wire; omitted, it is the longest row the run trains.
        (Some("constant-length"), length) => brain::LossAgg::ConstantLength(length.unwrap_or(params.max_block)),
        // An algorithm whose aggregation is a constant length (dr-grpo) takes the constant alone.
        (None, Some(n)) if matches!(config.loss_agg, brain::LossAgg::ConstantLength(_)) => brain::LossAgg::ConstantLength(n),
        (_, Some(_)) => return Err("train: loss_agg_length is the constant of loss_agg constant-length; name that aggregation".to_string()),
        (Some(v), None) => v.parse().map_err(|e| format!("train: loss_agg: {e}"))?,
        (None, None) => config.loss_agg,
    };
    if let Some(v) = params.temperature {
        config.temperature = v;
    }
    if let Some(v) = &params.teacher {
        config.teacher = v.parse().map_err(|e| format!("train: teacher: {e}"))?;
    }
    config.advantage_bound = params.advantage_bound.or(config.advantage_bound);
    config.sequence_advantage_cap = params.sequence_advantage_cap.or(config.sequence_advantage_cap);
    if let Some(v) = &params.advantage_norm {
        config.advantage_norm = brain::AdvantageNorm::parse(v).map_err(|e| format!("train: {e}"))?;
    }
    if let Some(v) = &params.advantage_floor {
        config.advantage_floor = brain::AdvantageFloor::parse(v).map_err(|e| format!("train: {e}"))?;
    }
    if let Some(v) = &params.truncated {
        config.truncated = brain::TruncatedRollouts::parse(v).map_err(|e| format!("train: {e}"))?;
    }
    Ok(config)
}

/// Where a run writes: the trainer's out directory in the job's own directory, and what must outlive the job, the
/// training state of a resumable run and the adapters of its evaluations (which a resumed run publishes beside the
/// ones it writes itself). Without a `resume_key` the evaluations live in the job's directory too.
pub struct RunFiles {
    pub out_dir: PathBuf,
    /// A resumable run's training state, the seed it trains with (kept in its record) and how often it saves.
    pub resumable: Option<(PathBuf, u64, u32)>,
    pub evaluations: PathBuf,
}

impl RunFiles {
    fn of(run: Option<&crate::runs::Run>, request: &Request, out_dir: PathBuf) -> RunFiles {
        match run {
            Some(run) => RunFiles { resumable: Some((run.state_path(), run.seed(), request.params.checkpoint_every.unwrap_or(crate::runs::DEFAULT_CHECKPOINT_EVERY))), evaluations: run.evaluations_dir(), out_dir },
            None => RunFiles { resumable: None, evaluations: out_dir.join("evaluations"), out_dir },
        }
    }
}

/// The trainer for `request` over the files `staged`, writing where `files` says, on the device this thread is scoped to.
pub fn build_fine_tune(request: &Request, base_dir: &Path, staged: &Staged, files: &RunFiles) -> Result<brain::ChatFineTune, String> {
    let p = &request.params;
    let mut ft = brain::ChatFineTune::from_pretrained(utf8(base_dir)?)
        .placed(true)
        .dataset(&staged.train)
        .out_dir(&files.out_dir)
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
    match files.resumable {
        Some((ref state, seed, every)) => ft = ft.seed(seed).resume_state(state).checkpoint_every(every),
        None => {
            if let Some(v) = p.seed {
                ft = ft.seed(v);
            }
        }
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
        ft = ft.keep_evaluations(&files.evaluations);
    }
    Ok(ft)
}

/// The pair loss a preference request names: `dpo` when it names none; `gamma` is SimPO's alone.
pub(crate) fn pair_loss(params: &crate::request::Params) -> Result<brain::PairLoss, String> {
    let name = params.loss.as_deref().unwrap_or("dpo");
    if params.gamma.is_some() && name != "simpo" {
        return Err(format!("train: gamma is simpo's target margin; the loss is {name}"));
    }
    brain::PairLoss::parse(name, params.gamma.unwrap_or(brain::DEFAULT_SIMPO_GAMMA)).map_err(|e| format!("train: {e}"))
}

/// The preference trainer for `request` over the files `staged`, writing where `files` says, on the device this thread is
/// scoped to.
pub fn build_preference(request: &Request, base_dir: &Path, staged: &Staged, files: &RunFiles) -> Result<brain::PreferenceFineTune, String> {
    let p = &request.params;
    let mut ft = brain::PreferenceFineTune::from_pretrained(utf8(base_dir)?)
        .placed(true)
        .dataset(&staged.train)
        .out_dir(&files.out_dir)
        .steps(p.steps)
        .grad_accum(p.grad_accum)
        .strict_updates(p.strict_updates)
        .max_block(p.max_block);
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
    match files.resumable {
        Some((ref state, seed, every)) => ft = ft.seed(seed).resume_state(state).checkpoint_every(every),
        None => {
            if let Some(v) = p.seed {
                ft = ft.seed(v);
            }
        }
    }
    if let Some(v) = p.lr {
        ft = ft.lr(v);
    }
    if let Some(v) = &p.base_format {
        ft = ft.base_format(v.as_str());
    }
    ft = ft.loss(pair_loss(p)?);
    if let Some(v) = p.beta {
        ft = ft.beta(v);
    }
    if let Some(v) = p.nll_weight {
        ft = ft.nll_weight(v);
    }
    if let Some(v) = p.keep_reasoning {
        ft = ft.keep_reasoning(v);
    }
    if let Some(path) = &staged.held_out {
        ft = ft.held_out(path);
    }
    if let Some(path) = &staged.start_adapter {
        ft = ft.continue_from(path);
    }
    Ok(ft)
}

/// The feedback (KTO) trainer for `request` over the files `staged`, writing where `files` says, on the device this
/// thread is scoped to.
pub fn build_feedback(request: &Request, base_dir: &Path, staged: &Staged, files: &RunFiles) -> Result<brain::FeedbackFineTune, String> {
    let p = &request.params;
    let mut ft = brain::FeedbackFineTune::from_pretrained(utf8(base_dir)?).placed(true).dataset(&staged.train).out_dir(&files.out_dir).steps(p.steps).grad_accum(p.grad_accum).strict_updates(p.strict_updates).max_block(p.max_block);
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
    match files.resumable {
        Some((ref state, seed, every)) => ft = ft.seed(seed).resume_state(state).checkpoint_every(every),
        None => {
            if let Some(v) = p.seed {
                ft = ft.seed(v);
            }
        }
    }
    if let Some(v) = p.lr {
        ft = ft.lr(v);
    }
    if let Some(v) = &p.base_format {
        ft = ft.base_format(v.as_str());
    }
    if let Some(v) = p.beta {
        ft = ft.beta(v);
    }
    if let Some(v) = p.desirable_weight {
        ft = ft.desirable_weight(v);
    }
    if let Some(v) = p.undesirable_weight {
        ft = ft.undesirable_weight(v);
    }
    if let Some(v) = p.keep_reasoning {
        ft = ft.keep_reasoning(v);
    }
    if let Some(path) = &staged.start_adapter {
        ft = ft.continue_from(path);
    }
    Ok(ft)
}

/// The distillation for `request` over the files `staged` (and the teacher's adapter, staged when it names one),
/// writing where `files` says, on the device this thread is scoped to.
pub fn build_distill(request: &Request, base_dir: &Path, staged: &Staged, teacher_adapter: Option<&Path>, files: &RunFiles) -> Result<brain::DistillFineTune, String> {
    let p = &request.params;
    let teacher = match (crate::request::DistillTeacherRef::parse(p.teacher.as_deref())?, teacher_adapter) {
        (crate::request::DistillTeacherRef::Base, _) => brain::DistillTeacher::Base,
        (crate::request::DistillTeacherRef::Start, _) => brain::DistillTeacher::Start,
        (crate::request::DistillTeacherRef::Adapter(_), Some(path)) => brain::DistillTeacher::Adapter(path.to_path_buf()),
        (crate::request::DistillTeacherRef::Adapter(_), None) => return Err("train: the teacher's adapter was not staged".to_string()),
    };
    let mut ft = brain::DistillFineTune::from_pretrained(utf8(base_dir)?).placed(true).dataset(&staged.train).teacher(teacher).out_dir(&files.out_dir).steps(p.steps).grad_accum(p.grad_accum).strict_updates(p.strict_updates).thinking(p.thinking).max_block(p.max_block);
    if let Some(k) = p.top_k {
        ft = ft.top_k(k as usize);
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
    match files.resumable {
        Some((ref state, seed, every)) => ft = ft.seed(seed).resume_state(state).checkpoint_every(every),
        None => {
            if let Some(v) = p.seed {
                ft = ft.seed(v);
            }
        }
    }
    if let Some(v) = p.lr {
        ft = ft.lr(v);
    }
    if let Some(v) = &p.base_format {
        ft = ft.base_format(v.as_str());
    }
    if let Some(path) = &staged.start_adapter {
        ft = ft.continue_from(path);
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
    if let Some(eval) = progress.eval_loss {
        frame["eval_loss"] = finite(eval);
    }
    let pg: serde_json::Map<String, Value> = progress.metrics.iter().filter_map(|(name, value)| name.strip_prefix("pg_").map(|n| (n.to_string(), finite(*value)))).collect();
    if !pg.is_empty() {
        frame["pg"] = Value::Object(pg);
    }
    frame
}

/// What a finished run produced, before anything is published: the files the trainer wrote and the objective's own
/// summary of the run (which rides one frame, a D-Bus datagram that is lost when it does not fit the socket, so it holds
/// totals and short lists, every one bounded by the request: see [`crate::request::MAX_EVALUATIONS`]; the per-step
/// series are the frames the run streamed and the `record` blob).
struct Trained {
    adapter: PathBuf,
    record: Option<PathBuf>,
    /// `sha256:<hex>` of the adapter file, as the adapter's card records it.
    adapter_sha256: String,
    /// The adapter of each evaluation the run kept, by step.
    evaluations: Vec<(u32, PathBuf)>,
    /// The step this job continued a resumable run's saved state at.
    resumed_at: Option<u32>,
    summary: Value,
}

fn preference_score_json(score: Option<brain::PreferenceScore>) -> Value {
    match score {
        Some(s) => json!({"accuracy": s.accuracy, "mean_margin": s.mean_margin, "pairs": s.pairs, "skipped": s.skipped}),
        None => Value::Null,
    }
}

/// A preference run's totals. `records` counts pairs; the scores are the tuned adapter against the run's own reference
/// (the base, or the adapter continued) on the training and the held-out pairs.
fn preference_summary(outcome: &brain::PreferenceFineTuneOutcome) -> Value {
    json!({
        "adapter_sha256": outcome.adapter_digest,
        "base_digest": outcome.base_digest,
        "trained_from": outcome.trained_from,
        "records": outcome.train_pairs,
        "block": outcome.block,
        "seed": outcome.seed,
        "steps": outcome.steps,
        "steps_completed": outcome.steps_completed,
        "resumed_at": outcome.resumed_at,
        "rank": outcome.rank,
        "alpha": outcome.alpha,
        "beta": outcome.beta,
        "loss": outcome.loss.name(),
        "gamma": match outcome.loss { brain::PairLoss::SimPo { gamma } => Some(gamma), _ => None },
        "nll_weight": outcome.nll_weight,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
        "train_score": preference_score_json(outcome.train_score),
        "held_out_score": preference_score_json(outcome.held_out_score),
        "held_out_repeats": outcome.held_out_repeats,
    })
}

/// A feedback run's totals: the examples by judgement and, after the run, the mean log-ratio of each against the
/// starting policy.
fn feedback_summary(outcome: &brain::FeedbackFineTuneOutcome) -> Value {
    json!({
        "adapter_sha256": outcome.adapter_digest,
        "base_digest": outcome.base_digest,
        "trained_from": outcome.trained_from,
        "records": outcome.desirable_examples + outcome.undesirable_examples,
        "desirable_examples": outcome.desirable_examples,
        "undesirable_examples": outcome.undesirable_examples,
        "block": outcome.block,
        "seed": outcome.seed,
        "steps": outcome.steps,
        "steps_completed": outcome.steps_completed,
        "resumed_at": outcome.resumed_at,
        "rank": outcome.rank,
        "alpha": outcome.alpha,
        "beta": outcome.beta,
        "desirable_weight": outcome.desirable_weight,
        "undesirable_weight": outcome.undesirable_weight,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
        "desirable_logratio": outcome.desirable_logratio,
        "undesirable_logratio": outcome.undesirable_logratio,
    })
}

/// A distillation's totals: the teacher, the records it scored, and the student's cross-entropy to it before and after.
fn distill_summary(outcome: &brain::DistillFineTuneOutcome) -> Value {
    json!({
        "adapter_sha256": outcome.adapter_digest,
        "base_digest": outcome.base_digest,
        "trained_from": outcome.trained_from,
        "teacher": outcome.teacher,
        "teacher_adapter_sha256": outcome.teacher_adapter,
        "top_k": outcome.top_k,
        "records": outcome.examples,
        "block": outcome.block,
        "seed": outcome.seed,
        "steps": outcome.steps,
        "steps_completed": outcome.steps_completed,
        "resumed_at": outcome.resumed_at,
        "rank": outcome.rank,
        "alpha": outcome.alpha,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
    })
}

/// Maps the SDK's error to the job's: a full card is a condition the caller can act on, with nothing wrong in the request.
pub(crate) fn trainer_failure(e: brain::Error) -> String {
    use capability::failure::{tag, FailureClass};
    match e {
        brain::Error::DeviceOutOfMemory { .. } | brain::Error::JobFailed(_) => tag(FailureClass::Unavailable, format!("train: {e}")),
        // The trainer's refusals of what the request asked for name the setting: the caller reads them.
        brain::Error::InvalidRequest(_) | brain::Error::MissingArgument(_) => tag(FailureClass::InvalidRequest, format!("train: {e}")),
        brain::Error::Unsupported(_) => tag(FailureClass::UnsupportedCapability, format!("train: {e}")),
        e => format!("train: {e}"),
    }
}

fn train_chat(request: &Request, base_dir: &Path, staged: &Staged, files: &RunFiles, cancel: &capability::CancelToken, began: Instant, progress: &mut dyn FnMut(Progress)) -> Result<Trained, String> {
    let fine_tune = build_fine_tune(request, base_dir, staged, files)?;
    let outcome = fine_tune.run_with(cancel, |step| progress(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64())))).map_err(trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".into());
    }
    let adapter = outcome.adapter.clone().ok_or("train: the run completed with no adapter")?;
    let evaluations = if request.params.keep_evaluations { outcome.curve.iter().map(|p| (p.step, files.evaluations.join(format!("step-{}.safetensors", p.step)))).collect() } else { Vec::new() };
    Ok(Trained { adapter, record: outcome.record.clone(), adapter_sha256: outcome.adapter_digest.clone().unwrap_or_default(), evaluations, resumed_at: outcome.resumed_at, summary: outcome.report() })
}

fn train_preference(request: &Request, base_dir: &Path, staged: &Staged, files: &RunFiles, cancel: &capability::CancelToken, began: Instant, progress: &mut dyn FnMut(Progress)) -> Result<Trained, String> {
    let fine_tune = build_preference(request, base_dir, staged, files)?;
    let outcome = fine_tune.run_with(cancel, |step| progress(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64())))).map_err(trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".into());
    }
    let adapter = outcome.adapter.clone().ok_or("train: the run completed with no adapter")?;
    Ok(Trained { adapter, record: outcome.record.clone(), adapter_sha256: outcome.adapter_digest.clone().unwrap_or_default(), evaluations: Vec::new(), resumed_at: outcome.resumed_at, summary: preference_summary(&outcome) })
}

fn train_feedback(request: &Request, base_dir: &Path, staged: &Staged, files: &RunFiles, cancel: &capability::CancelToken, began: Instant, progress: &mut dyn FnMut(Progress)) -> Result<Trained, String> {
    let fine_tune = build_feedback(request, base_dir, staged, files)?;
    let outcome = fine_tune.run_with(cancel, |step| progress(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64())))).map_err(trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".into());
    }
    let adapter = outcome.adapter.clone().ok_or("train: the run completed with no adapter")?;
    Ok(Trained { adapter, record: outcome.record.clone(), adapter_sha256: outcome.adapter_digest.clone().unwrap_or_default(), evaluations: Vec::new(), resumed_at: outcome.resumed_at, summary: feedback_summary(&outcome) })
}

#[allow(clippy::too_many_arguments)]
fn train_distill(request: &Request, base_dir: &Path, staged: &Staged, teacher_adapter: Option<&Path>, files: &RunFiles, cancel: &capability::CancelToken, began: Instant, progress: &mut dyn FnMut(Progress)) -> Result<Trained, String> {
    let fine_tune = build_distill(request, base_dir, staged, teacher_adapter, files)?;
    let outcome = fine_tune.run_with(cancel, |step| progress(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64())))).map_err(trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".into());
    }
    let adapter = outcome.adapter.clone().ok_or("train: the run completed with no adapter")?;
    Ok(Trained { adapter, record: outcome.record.clone(), adapter_sha256: outcome.adapter_digest.clone().unwrap_or_default(), evaluations: Vec::new(), resumed_at: outcome.resumed_at, summary: distill_summary(&outcome) })
}

/// Runs the job `inv` on the device this thread is scoped to, reporting each step as a frame. With a `resume_key`
/// the run is resumable ([`crate::runs`]): its training state and evaluation adapters live in the run's directory of
/// the artifact store, and a completed run answers with the result it recorded.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let request = Request::from_invocation(inv)?;
    let p = &request.params;
    let run = match crate::consolidate::open_run(env, ACTION, p.resume_key.as_deref(), inv, p.seed)? {
        Some(crate::runs::Opened::Completed(mut outputs)) => {
            outputs["resume"] = crate::runs::completed_report(p.resume_key.as_deref().unwrap_or_default());
            return recorded(env, &request, outputs);
        }
        Some(crate::runs::Opened::Run(run)) => Some(run),
        None => None,
    };
    let base_dir = resolve_base(env.models_dir.as_deref(), &p.base)?;
    let steps = p.steps;
    let work = Workdir::create(env)?;
    progress(Progress::event(0, steps, json!({"type": "stage", "stage": "staging the datasets"})));
    let staged = stage(&request.data, &inv.blobs, env.store.as_ref(), &work)?;
    if inv.cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let files = RunFiles::of(run.as_ref(), &request, work.path().join("run"));
    progress(Progress::event(0, steps, json!({"type": "stage", "stage": "loading the base and training"})));
    let began = Instant::now();
    let trained = match p.objective {
        Objective::Preference => train_preference(&request, &base_dir, &staged, &files, &inv.cancel, began, progress)?,
        Objective::Feedback => train_feedback(&request, &base_dir, &staged, &files, &inv.cancel, began, progress)?,
        Objective::Distill => {
            // The teacher's adapter, when the request names one by digest, is fetched into the job's directory like a dataset.
            let teacher = match crate::request::DistillTeacherRef::parse(p.teacher.as_deref())? {
                crate::request::DistillTeacherRef::Adapter(digest) => {
                    let path = work.path().join("teacher.safetensors");
                    bring(&Source::Digest(digest), &inv.blobs, env.store.as_ref(), work.staging(), &path)?;
                    Some(path)
                }
                _ => None,
            };
            train_distill(&request, &base_dir, &staged, teacher.as_deref(), &files, &inv.cancel, began, progress)?
        }
        Objective::Sft | Objective::PolicyGradient => train_chat(&request, &base_dir, &staged, &files, &inv.cancel, began, progress)?,
    };
    work.ensure_within_quota()?;
    let mut result = Outcome::new();
    let mut digest = None;
    if let Some(store) = &env.store {
        let (published, _) = store.publish(&trained.adapter).map_err(|e| store_failure(&e))?;
        digest = Some(published);
    }
    // The training record holds every step's statistics, which is bulk: it goes to the store beside the adapter.
    let mut record_digest = None;
    if let (Some(store), Some(record)) = (&env.store, &trained.record) {
        record_digest = Some(store.publish(record).map_err(|e| store_failure(&e))?.0);
    }
    let mut evaluations = Vec::new();
    if let Some(store) = &env.store {
        for (step, path) in &trained.evaluations {
            let (published, _) = store.publish(path).map_err(|e| store_failure(&e))?;
            evaluations.push(json!({"step": step, "digest": published.hex()}));
        }
    }
    let mut outputs = trained.summary;
    if let Some(fields) = outputs.as_object_mut() {
        fields.insert("status".into(), json!("completed"));
        fields.insert("objective".into(), json!(p.objective.name()));
        if p.objective == Objective::PolicyGradient {
            fields.insert("algorithm".into(), json!(p.algorithm));
        }
        fields.insert("adapter_digest".into(), json!(digest.map(|d| d.hex())));
        fields.insert("record_digest".into(), json!(record_digest.map(|d| d.hex())));
        fields.insert("evaluations".into(), json!(evaluations));
    }
    if let Some(run) = run {
        outputs["resume"] = run.report(trained.resumed_at);
        // Everything the result names is published: the run is complete, and a resubmission answers with this.
        run.complete(&outputs)?;
    }
    result.outputs = outputs;
    if p.return_adapter {
        let bytes = std::fs::read(&trained.adapter).map_err(|e| format!("train: reading the adapter: {e}"))?;
        result = result.blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"sha256": trained.adapter_sha256, "blake3": digest.map(|d| d.hex())})));
    }
    // Taken by blob when the caller takes everything by blob (or the server has no store to publish it to).
    if let (Some(record), true) = (&trained.record, p.return_adapter || record_digest.is_none()) {
        result = result.blob("record", Blob::new(Media::Text, std::fs::read(record).map_err(|e| format!("train: reading the training record: {e}"))?));
    }
    Ok(result)
}

/// The answer to a resubmitted run that completed: its recorded result, with the adapter and the training record
/// fetched back from the store (verified against their digests) when the request asks for them as blobs.
fn recorded(env: &JobEnv, request: &Request, outputs: Value) -> ActionResult {
    let mut result = Outcome::new();
    if request.params.return_adapter {
        let store = env.store.as_ref().ok_or("train: a completed run is answered from the artifact store, and this server has none")?;
        let work = Workdir::create(env)?;
        let fetch = |field: &str| -> Result<Vec<u8>, String> {
            let digest = outputs[field].as_str().map(crate::store::Digest::parse).transpose().map_err(|e| format!("train: the recorded {field}: {e}"))?.ok_or_else(|| format!("train: the recorded result names no {field}"))?;
            let path = work.path().join(field);
            store.fetch(&digest, &path, MAX_INPUT_BYTES).map_err(|e| store_failure(&e))?;
            std::fs::read(&path).map_err(|e| format!("train: reading the recorded {field}: {e}"))
        };
        let adapter = fetch("adapter_digest")?;
        result = result.blob("adapter", Blob::new(Media::Bytes, adapter).with_meta(json!({"sha256": outputs["adapter_sha256"], "blake3": outputs["adapter_digest"]})));
        result = result.blob("record", Blob::new(Media::Text, fetch("record_digest")?));
    }
    result.outputs = outputs;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Media;

    fn env_with(root: &Path, max_job_staging_bytes: u64, store: Option<ArtifactStore>) -> JobEnv {
        JobEnv { store, models_dir: None, work_root: root.to_path_buf(), quotas: Quotas { max_job_staging_bytes, ..Quotas::default() } }
    }

    fn blobs(sizes: &[(&str, usize)]) -> BTreeMap<String, Blob> {
        sizes.iter().map(|(name, n)| ((*name).to_string(), Blob::new(Media::Bytes, vec![7u8; *n]))).collect()
    }

    #[test]
    fn staged_inputs_are_charged_and_the_one_that_does_not_fit_is_refused_unwritten() {
        let root = tempfile::tempdir().unwrap();
        let work = Workdir::create(&env_with(&root.path().join("w"), 10, None)).unwrap();
        let inputs = blobs(&[("a", 6), ("b", 6)]);
        let (a, b) = (work.path().join("a"), work.path().join("b"));
        bring(&Source::Blob("a".into()), &inputs, None, work.staging(), &a).unwrap();
        let error = bring(&Source::Blob("b".into()), &inputs, None, work.staging(), &b).unwrap_err();
        assert!(error.contains("staging space"), "{error}");
        assert!(a.exists() && !b.exists(), "the refused input left nothing behind");
    }

    #[test]
    fn a_digest_larger_than_the_staging_room_is_refused_and_leaves_no_copy() {
        let root = tempfile::tempdir().unwrap();
        let artifacts = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(artifacts.path()).unwrap();
        let digest = store.publish_bytes(&[3u8; 64]).unwrap();
        let work = Workdir::create(&env_with(&root.path().join("w"), 32, Some(store.clone()))).unwrap();
        let dest = work.path().join("input");
        let error = bring(&Source::Digest(digest), &BTreeMap::new(), Some(&store), work.staging(), &dest).unwrap_err();
        assert!(error.contains("staging space"), "{error}");
        assert!(!dest.exists());
    }

    #[test]
    fn what_a_trainer_wrote_past_the_budget_is_caught_before_publishing_and_removed_with_the_directory() {
        let root = tempfile::tempdir().unwrap();
        let work = Workdir::create(&env_with(&root.path().join("w"), 100, None)).unwrap();
        std::fs::create_dir(work.path().join("run")).unwrap();
        std::fs::write(work.path().join("run").join("checkpoint"), [0u8; 100]).unwrap();
        work.ensure_within_quota().unwrap();
        std::fs::write(work.path().join("run").join("extra"), [0u8; 1]).unwrap();
        assert!(work.ensure_within_quota().unwrap_err().contains("staging space"));
        let dir = work.path().to_path_buf();
        drop(work);
        assert!(!dir.exists(), "the partial staging is removed when the job ends");
    }
}

#[cfg(test)]
mod failure_tests {
    use super::trainer_failure;
    use capability::failure::{FailureClass, ReplyError};

    #[test]
    fn a_refused_request_stays_readable_and_an_internal_failure_stays_hidden() {
        let refused = trainer_failure(brain::Error::InvalidRequest("ChatFineTune: selecting on the monitoring loss needs eval_every > 0".into()));
        let reply = ReplyError::of(&refused);
        assert_eq!(reply, ReplyError::Classified(FailureClass::InvalidRequest, "train: ChatFineTune: selecting on the monitoring loss needs eval_every > 0".into()));
        assert!(!reply.hides_detail());
        let missing = ReplyError::of(&trainer_failure(brain::Error::MissingArgument("a data-parallel rank needs the seed".into())));
        assert!(matches!(missing, ReplyError::Classified(FailureClass::InvalidRequest, _)));
        let internal = ReplyError::of(&trainer_failure(brain::Error::Backend("training: /models/x: read failed".into())));
        assert!(internal.hides_detail(), "{internal:?}");
    }
}

