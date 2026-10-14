// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From outcome-labelled decision examples to a checkpoint a server can load:
//! the training run behind `brain decide train`.
//!
//! [`train`] reads and validates a JSONL file (`decisions::examples`), trains a
//! `decide` model's head on its `train` split through
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
//! Swedish Embedded AB implements training pipelines that turn recorded
//! outcomes into governed model checkpoints, with lineage on every artifact.
//! If your team needs expertise in decision-model training and provenance, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use decisions::engine::{is_laya_dir, Backend};
use decisions::examples::{Dataset, Split, Summary};
use decisions::manifest::{DecisionManifest, Lineage, Validation};
use decisions::store::{identity_files, resolve_spec};
use serde_json::json;

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
    /// Train the head alone. `false` fine-tunes the encoder too.
    pub freeze_encoder: bool,
    pub steps: usize,
    /// Examples per optimizer update.
    pub batch: usize,
    pub seed: u64,
    /// The commit of the code doing the training, recorded in the lineage.
    pub brain_commit: String,
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
            seed: 0,
            brain_commit: "unknown".to_string(),
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
    if is_laya_dir(&base_dir) {
        return Err(Error::MissingArgument(
            "the base is a Laya checkpoint; writing a servable Laya checkpoint from a trained head is not supported, train from a `decide` base".into(),
        ));
    }
    let base_sha256 = brain_modelstore::fetch::base_digest(&base_dir).map_err(|e| Error::Backend(format!("hashing the base {base_id}: {e}")))?;
    let base_str = base_dir.to_str().ok_or_else(|| Error::MissingArgument("the base path is not UTF-8".into()))?;

    let mut pipeline = DecisionPipeline::builder(base_str).seed(run.seed).load()?;
    pipeline.set_encoder_frozen(run.freeze_encoder)?;
    let final_loss = pipeline.train_cases(&cases, run.steps, run.batch, run.seed, log)?;

    // Write beside the target and rename, so the output directory either
    // exists complete or does not exist.
    let staging = run.out.with_file_name(format!(".{}.partial", run.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(Error::Io)?;
    }
    let staged = write_checkpoint(&pipeline, &staging, run, &dataset, &base_id, base_sha256, final_loss);
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

fn write_checkpoint(
    pipeline: &DecisionPipeline,
    dir: &Path,
    run: &TrainRun,
    dataset: &Dataset,
    base_id: &str,
    base_sha256: String,
    final_loss: f32,
) -> Result<DecisionManifest> {
    let dir_str = dir.to_str().ok_or_else(|| Error::MissingArgument("the output path is not UTF-8".into()))?;
    pipeline.save_model(dir_str)?;
    let (weights, tokenizer) = identity_files(Backend::Decide);
    let digest = |file: &str| brain_modelstore::fetch::file_digest(&dir.join(file)).map_err(|e| Error::Backend(format!("hashing {file}: {e}")));

    let summary = dataset.summary();
    let split_counts = Split::ALL.into_iter().map(|s| (s.slug().to_string(), summary.by_split.get(&s).map_or(0, |x| x.count))).collect();
    let mut manifest = DecisionManifest::undeclared(Backend::Decide);
    manifest.weights_sha256 = Some(digest(weights)?);
    manifest.tokenizer_sha256 = Some(digest(tokenizer)?);
    manifest.question_types = vec![decisions::wire::QuestionType::Choice];
    manifest.lineage = Lineage {
        base: Some(base_id.to_string()),
        trained_on: Some(format!("{} examples ({})", summary.total, dataset.digest.as_deref().unwrap_or("unhashed"))),
        recipe: Some("head training on outcome-labelled choice decisions".to_string()),
        base_sha256: Some(base_sha256),
        data_sha256: dataset.digest.clone(),
        split_counts: Some(split_counts),
        hyperparameters: Some(json!({
            "steps": run.steps,
            "batch": run.batch,
            "freeze_encoder": run.freeze_encoder,
            "final_loss": final_loss,
        })),
        seed: Some(run.seed),
        brain_commit: Some(run.brain_commit.clone()),
    };
    manifest.validation = Validation::Unvalidated;
    manifest.write(dir).map_err(Error::Backend)?;
    Ok(manifest)
}
