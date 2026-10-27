// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From outcome-labelled decision examples to a checkpoint a server can load:
//! the training run behind `brain decide train`.
//!
//! [`train`] reads and validates a JSONL file (`decisions::examples`), trains a
//! decision head, `decide` or Laya, on its `train` split through
//! [`DecisionPipeline::train_cases`], and writes a checkpoint directory whose
//! `decision.manifest.json` records where the weights came from: the base
//! model's identity, the data's digest, the split counts, the hyperparameters,
//! the seed and the commit of the code that trained it.
//!
//! **A training run never claims validation.** The manifest it writes says
//! `unvalidated`, so the `auto` selector will not pick it. Only
//! `brain decide promote`, which re-measures the weights on held-out data,
//! can change that.
//!
//! The encoder is frozen by default: the head is the part the data is for, a
//! step is roughly a forward pass, and an untouched encoder keeps the run's
//! lineage honest (the base identity names exactly the encoder in the output).
//!
//! The same seed on the same data and base gives the same weights.
//!
//! **A Laya base is referenced, not copied.** Its trunk is always frozen, so
//! the output holds the trained head and a reference to the base by store id
//! and digest (`decisions::laya_head`) instead of 1.6 GB of unchanged weights.
//! Training the trunk (`freeze_encoder = false`) is refused for Laya: it could
//! only be saved as a full copy, and it costs a gradient and two optimizer
//! moments for every one of 395M weights.
//!
//! [`promote_model`] and [`rollback_model`] close the loop: promotion
//! re-measures the weights on the test split under a policy and, only if every
//! gate passes, records the validation the `auto` selector requires.
//!
//! [`evaluate_model`] is the other half: it loads a checkpoint (or a stock
//! model), verifies its declared hashes against the files, and measures it on
//! one split of the same kind of file with `decisions::evaluation`.
//!
//! Swedish Embedded AB implements training pipelines that turn recorded
//! outcomes into governed model checkpoints, with lineage on every artifact.
//! If your team needs expertise in decision-model training and provenance, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use decisions::engine::{is_laya_dir, Backend};
use decisions::laya_head::{self, BaseRef};
use decisions::evaluation::{evaluate, CostSpec, EvalRequest, Report, Scorer};
use decisions::examples::{Dataset, Split, Summary};
use decisions::manifest::{verify_identity, weights_identity, DecisionManifest, Lineage, Validation};
use decisions::promotion::{self, Pointer, Policy, Promotion, Verdict};
use decisions::store::{decide_dir_carries_a_head, identity_files, resolve_spec};
use serde_json::{json, Value};

use crate::decision::{ChoiceCase, DecisionPipeline, DEFAULT_BATCH};
use crate::{Error, Result};

/// Optimizer updates a run takes when the caller names none.
pub const DEFAULT_STEPS: usize = 600;

/// Everything a training run is told. Explicit throughout: nothing is read
/// from the environment.
#[derive(Clone, Debug)]
pub struct TrainRun {
    /// The JSONL file of examples.
    pub data: PathBuf,
    /// The base model: a model-store id, or a checkpoint directory.
    pub base: String,
    /// The checkpoint directory to create. It must not exist yet.
    pub out: PathBuf,
    /// Train the head alone. `false` fine-tunes the encoder too (`decide`
    /// bases only; a Laya base refuses it).
    pub freeze_encoder: bool,
    pub steps: usize,
    /// Examples per optimizer update.
    pub batch: usize,
    /// Seeds the head's initialization and batch order. [`TrainRun::new`]
    /// draws a fresh one; the checkpoint's lineage records the one used, and
    /// passing it back reproduces the weights.
    pub seed: u64,
    /// The commit of the code doing the training, recorded in the lineage.
    pub brain_commit: String,
    /// What the backward keeps of the forward. `None` leaves the backend's own default (keep every layer's
    /// intermediates); a choice the backend cannot run fails the run before the first step.
    pub saved: Option<model::saved::SavedPolicy>,
}

impl TrainRun {
    pub fn new(data: impl Into<PathBuf>, base: impl Into<String>, out: impl Into<PathBuf>) -> TrainRun {
        TrainRun {
            data: data.into(),
            base: base.into(),
            out: out.into(),
            freeze_encoder: true,
            steps: DEFAULT_STEPS,
            batch: DEFAULT_BATCH,
            seed: crate::random_seed(),
            brain_commit: "unknown".to_string(),
            saved: None,
        }
    }
}

/// What a training run produced.
#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub out: PathBuf,
    /// Mean training loss over the last tenth of the run.
    pub final_loss: f32,
    pub manifest: DecisionManifest,
    pub summary: Summary,
}

/// Resolve a model reference to the id it is addressed by and its directory.
pub fn resolve_model(spec: &str) -> (String, PathBuf) {
    resolve_spec(brain_modelstore::default_root().as_deref(), spec)
}

/// The training cases of a dataset's `train` split.
fn cases_of(dataset: &Dataset) -> Vec<ChoiceCase> {
    dataset
        .split(Split::Train)
        .map(|e| ChoiceCase { state: e.state.clone(), instructions: e.instruction.clone(), options: e.option_texts(), label: e.label })
        .collect()
}

/// Train a decision head on a dataset's `train` split and write the checkpoint.
///
/// `log` receives `(step, loss)` for every optimizer update.
pub fn train(run: &TrainRun, log: &mut dyn FnMut(usize, f32)) -> Result<TrainOutcome> {
    if run.steps == 0 {
        return Err(Error::MissingArgument("--steps must be at least 1".into()));
    }
    if run.out.exists() {
        return Err(Error::MissingArgument(format!("{} already exists; a training run writes a new directory", run.out.display())));
    }
    let dataset = Dataset::read_file(&run.data).map_err(|e| Error::MissingArgument(format!("{}: {e}", run.data.display())))?;
    let cases = cases_of(&dataset);
    if cases.is_empty() {
        return Err(Error::MissingArgument(format!("{} has no `train` examples", run.data.display())));
    }
    let (base_id, base_dir) = resolve_model(&run.base);
    if !base_dir.is_dir() {
        return Err(Error::ModelNotFound(format!("the base {base_id} is not installed; pull it with `brain pull`, or name a checkpoint directory")));
    }
    let laya_base = is_laya_dir(&base_dir);
    if laya_base {
        if laya_head::is_head_only(&base_dir) {
            return Err(Error::MissingArgument(format!("the base {base_id} is itself a trained head; name the stock Laya checkpoint it was trained from")));
        }
        if !run.freeze_encoder {
            return Err(Error::MissingArgument(
                "--finetune-encoder is not supported for a Laya base: the 395M-parameter trunk is trained with a gradient and two optimizer moments per weight, and a trained trunk could only be saved as a full copy of the weights; train the head with the trunk frozen".into(),
            ));
        }
    }
    let base_sha256 = brain_modelstore::fetch::base_digest(&base_dir).map_err(|e| Error::Backend(format!("hashing the base {base_id}: {e}")))?;
    let base_str = base_dir.to_str().ok_or_else(|| Error::MissingArgument("the base path is not UTF-8".into()))?;

    let mut pipeline = DecisionPipeline::builder(base_str).seed(run.seed).load()?;
    pipeline.set_encoder_frozen(run.freeze_encoder)?;
    if let Some(policy) = run.saved {
        pipeline.set_saved_policy(policy)?;
    }
    let final_loss = pipeline.train_cases(&cases, run.steps, run.batch, run.seed, log)?;

    // Write beside the target and rename, so the output directory either
    // exists complete or does not exist.
    let staging = run.out.with_file_name(format!(".{}.partial", run.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(Error::Io)?;
    }
    let record = RunRecord { run, dataset: &dataset, base: BaseModel { id: base_id, dir: base_dir, sha256: base_sha256 }, final_loss };
    let staged = if laya_base { write_laya_checkpoint(&pipeline, &staging, &record) } else { write_decide_checkpoint(&pipeline, &staging, &record) };
    let manifest = match staged {
        Ok(m) => m,
        Err(e) => {
            std::fs::remove_dir_all(&staging).ok();
            return Err(e);
        }
    };
    std::fs::rename(&staging, &run.out).map_err(Error::Io)?;
    Ok(TrainOutcome { out: run.out.clone(), final_loss, manifest, summary: dataset.summary() })
}

/// The base a run trained on top of.
struct BaseModel {
    id: String,
    dir: PathBuf,
    /// `modelstore::fetch::base_digest` of `dir`.
    sha256: String,
}

/// Everything a finished run knows about itself, for the manifest.
struct RunRecord<'a> {
    run: &'a TrainRun,
    dataset: &'a Dataset,
    base: BaseModel,
    final_loss: f32,
}

impl RunRecord<'_> {
    /// The manifest of a run's output: its lineage, `unvalidated`, and the
    /// hashes of the files it wrote (`weights_sha256` is the caller's, since
    /// what the weights ARE differs by layout).
    fn manifest(&self, backend: Backend, weights_sha256: String, tokenizer_sha256: String, recipe: &str) -> DecisionManifest {
        let summary = self.dataset.summary();
        let split_counts = Split::ALL.into_iter().map(|s| (s.slug().to_string(), summary.by_split.get(&s).map_or(0, |x| x.count))).collect();
        let mut manifest = DecisionManifest::undeclared(backend);
        manifest.weights_sha256 = Some(weights_sha256);
        manifest.tokenizer_sha256 = Some(tokenizer_sha256);
        manifest.question_types = vec![decisions::wire::QuestionType::Choice];
        manifest.lineage = Lineage {
            base: Some(self.base.id.clone()),
            trained_on: Some(format!("{} examples ({})", summary.total, self.dataset.digest.as_deref().unwrap_or("unhashed"))),
            recipe: Some(recipe.to_string()),
            base_sha256: Some(self.base.sha256.clone()),
            data_sha256: self.dataset.digest.clone(),
            split_counts: Some(split_counts),
            hyperparameters: Some(json!({
                "steps": self.run.steps,
                "batch": self.run.batch,
                "freeze_encoder": self.run.freeze_encoder,
                "final_loss": self.final_loss,
            })),
            seed: Some(self.run.seed),
            brain_commit: Some(self.run.brain_commit.clone()),
        };
        manifest.validation = Validation::Unvalidated;
        manifest
    }
}

fn file_digest_in(dir: &Path, file: &str) -> Result<String> {
    brain_modelstore::fetch::file_digest(&dir.join(file)).map_err(|e| Error::Backend(format!("hashing {file}: {e}")))
}

fn write_decide_checkpoint(pipeline: &DecisionPipeline, dir: &Path, record: &RunRecord) -> Result<DecisionManifest> {
    let dir_str = dir.to_str().ok_or_else(|| Error::MissingArgument("the output path is not UTF-8".into()))?;
    pipeline.save_model(dir_str)?;
    let (weights, tokenizer) = identity_files(Backend::Decide);
    let manifest = record.manifest(Backend::Decide, file_digest_in(dir, weights)?, file_digest_in(dir, tokenizer)?, "head training on outcome-labelled choice decisions");
    manifest.write(dir).map_err(Error::Backend)?;
    Ok(manifest)
}

/// Write a trained Laya head as a directory in the head-only layout
/// (`decisions::laya_head`): the head, a reference to the stock backbone by
/// store id and digest, the base's small configuration files, and the
/// manifest. The 395M-parameter trunk was frozen, so it is not copied.
fn write_laya_checkpoint(pipeline: &DecisionPipeline, dir: &Path, record: &RunRecord) -> Result<DecisionManifest> {
    std::fs::create_dir_all(dir).map_err(Error::Io)?;
    let head = dir.join(laya_head::HEAD_FILE);
    pipeline.save_head(head.to_str().ok_or_else(|| Error::MissingArgument("the output path is not UTF-8".into()))?)?;
    let base = BaseRef::with_digest(&record.base.id, &record.base.dir, record.base.sha256.clone()).map_err(Error::Backend)?;
    laya_head::write_layout(dir, &record.base.dir, &base).map_err(Error::Backend)?;
    let (_, tokenizer) = identity_files(Backend::Laya);
    let manifest = record.manifest(
        Backend::Laya,
        weights_identity(dir, Backend::Laya).map_err(Error::Backend)?,
        file_digest_in(dir, tokenizer)?,
        "Laya decision-head training on outcome-labelled choice decisions; the backbone is frozen and referenced, not copied",
    );
    manifest.write(dir).map_err(Error::Backend)?;
    Ok(manifest)
}

impl Scorer for DecisionPipeline {
    fn probabilities(&mut self, state: &str, instruction: &str, options: &[String]) -> std::result::Result<Vec<f32>, String> {
        let options: Vec<&str> = options.iter().map(String::as_str).collect();
        let choice = self.choose(state, instruction, &options).map_err(|e| e.to_string())?;
        Ok(choice.probabilities.into_iter().map(|(_, p)| p).collect())
    }
}

/// A decision model loaded from a checkpoint directory, with what it was
/// loaded under.
pub struct LoadedModel {
    pub id: String,
    pub dir: PathBuf,
    pub manifest: DecisionManifest,
    /// `sha256:<hex>` of the weights file, computed now.
    pub weights_sha256: String,
    pub pipeline: DecisionPipeline,
}

/// Load the decision model `spec` names (a store id or a directory) for
/// measuring or answering.
///
/// A checkpoint whose manifest declares a hash its file does not match is
/// refused: a recorded validation must stay attached to the bytes it was
/// measured on. A bare encoder, which has no trained head, is refused too.
pub fn load_model(spec: &str) -> Result<LoadedModel> {
    let (id, dir) = resolve_model(spec);
    if !dir.is_dir() {
        return Err(Error::ModelNotFound(format!("the model {id} is not installed; pull it with `brain pull`, or name a checkpoint directory")));
    }
    let backend = Backend::sniff(&dir).ok_or_else(|| Error::MissingArgument(format!("{id} is not a decision checkpoint")))?;
    if backend == Backend::Decide && !decide_dir_carries_a_head(&dir) {
        return Err(Error::MissingArgument(format!("{id} is a bare encoder with no trained head; train one with `brain decide train`")));
    }
    let manifest = DecisionManifest::read(&dir, backend).map_err(Error::Backend)?;
    verify_identity(&manifest, &dir, backend).map_err(Error::Backend)?;
    let weights_sha256 = weights_identity(&dir, backend).map_err(Error::Backend)?;
    let dir_str = dir.to_str().ok_or_else(|| Error::MissingArgument("the model path is not UTF-8".into()))?;
    let pipeline = DecisionPipeline::builder(dir_str).load()?;
    Ok(LoadedModel { id, dir, manifest, weights_sha256, pipeline })
}

/// What an evaluation is told.
#[derive(Clone, Debug)]
pub struct EvalRun {
    /// The model: a store id or a checkpoint directory.
    pub model: String,
    pub data: PathBuf,
    pub split: Split,
    /// A cost file, for cost-weighted regret.
    pub costs: Option<PathBuf>,
    /// Seeds the option permutation of the order-stability pass.
    /// [`EvalRun::new`] draws a fresh one; the report records the one used.
    pub order_seed: u64,
}

impl EvalRun {
    pub fn new(model: impl Into<String>, data: impl Into<PathBuf>, split: Split) -> EvalRun {
        EvalRun { model: model.into(), data: data.into(), split, costs: None, order_seed: crate::random_seed() }
    }
}

/// A measurement together with what was measured and on what.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub model_id: String,
    pub weights_sha256: String,
    pub data_sha256: Option<String>,
    pub report: Report,
}

impl Evaluation {
    /// The record promotion stores next to the manifest.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": 1,
            "model": {"id": self.model_id, "weights_sha256": self.weights_sha256},
            "data": {"sha256": self.data_sha256, "split": self.report.split.slug()},
            "report": self.report.to_json(),
        })
    }
}

/// Evaluate a loaded pipeline on one split of a validated dataset.
pub fn evaluate_pipeline(pipeline: &mut DecisionPipeline, dataset: &Dataset, request: &EvalRequest) -> Result<Report> {
    evaluate(pipeline, dataset, request).map_err(Error::Backend)
}

/// Load the model, read the data and measure the model on the chosen split.
pub fn evaluate_model(run: &EvalRun) -> Result<Evaluation> {
    let costs = run.costs.as_deref().map(CostSpec::read).transpose().map_err(Error::MissingArgument)?;
    let dataset = Dataset::read_file(&run.data).map_err(|e| Error::MissingArgument(format!("{}: {e}", run.data.display())))?;
    let mut model = load_model(&run.model)?;
    let report = evaluate_pipeline(&mut model.pipeline, &dataset, &EvalRequest { split: run.split, order_seed: run.order_seed, costs })?;
    Ok(Evaluation { model_id: model.id, weights_sha256: model.weights_sha256, data_sha256: dataset.digest, report })
}

/// What a promotion is told.
#[derive(Clone, Debug)]
pub struct PromoteRun {
    /// The checkpoint directory (or store id) to promote.
    pub model: String,
    pub data: PathBuf,
    pub policy: PathBuf,
}

/// The result of asking for a promotion.
#[derive(Clone, Debug)]
pub struct PromoteOutcome {
    /// The fresh measurement the gates were judged on.
    pub evaluation: Evaluation,
    pub verdict: Verdict,
    /// The pointer after a successful promotion; `None` when a gate failed, in
    /// which case nothing was changed.
    pub pointer: Option<Pointer>,
}

/// Re-evaluate a checkpoint on the test split under a policy, and promote it
/// only if every gate passes.
///
/// The numbers in the checkpoint's manifest are never consulted: the weights
/// on disk are measured now, and the validation that is written names the
/// digest of exactly those weights.
pub fn promote_model(run: &PromoteRun) -> Result<PromoteOutcome> {
    let policy = Policy::read(&run.policy).map_err(Error::MissingArgument)?;
    let dataset = Dataset::read_file(&run.data).map_err(|e| Error::MissingArgument(format!("{}: {e}", run.data.display())))?;
    let mut model = load_model(&run.model)?;
    let report = evaluate_pipeline(&mut model.pipeline, &dataset, &policy.eval_request())?;
    let evaluation = Evaluation { model_id: model.id.clone(), weights_sha256: model.weights_sha256.clone(), data_sha256: dataset.digest, report };
    let verdict = policy.judge(&evaluation.report);
    if !verdict.passed() {
        return Ok(PromoteOutcome { evaluation, verdict, pointer: None });
    }
    let mut record = evaluation.to_json();
    record["policy"] = policy.to_json();
    record["verdict"] = verdict.to_json();
    let pointer = promotion::commit(&Promotion { dir: &model.dir, policy: &policy, weights_sha256: &evaluation.weights_sha256, record: &record }).map_err(Error::Backend)?;
    Ok(PromoteOutcome { evaluation, verdict, pointer: Some(pointer) })
}

/// Make the last-known-good checkpoint of `model`'s profile the validated one
/// again. `model` must be the profile's current checkpoint.
pub fn rollback_model(model: &str) -> Result<Pointer> {
    let (id, dir) = resolve_model(model);
    if !dir.is_dir() {
        return Err(Error::ModelNotFound(format!("the model {id} is not installed")));
    }
    promotion::rollback(&dir).map_err(Error::Backend)
}
