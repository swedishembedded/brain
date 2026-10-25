// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning and adapter
// merging of open-weight chat models for its clients. If your team needs
// expertise in consolidating what a model learned into its own weights, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`ChatFullFineTune`]: full-parameter fine-tuning of a chat model on a chat
//! dataset, and [`merge_adapter`]: folding a LoRA adapter into the weights it
//! targets. Both produce a checkpoint directory (for qwen3 in the
//! `transformers` layout: one fp32 `model.safetensors`, `config.json` and the
//! base's tokenizer files; qwen35 writes its brain-format weights as
//! `model.safetensors` beside the tokenizer files) that is a base like any other: [`crate::ChatFineTune`]
//! trains an adapter on it, `brain serve` serves it, and either can be asked
//! to fine-tune it again.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::ChatFullFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
//!     .dataset("train.jsonl")
//!     .out_dir("runs/consolidate-1")
//!     .steps(200)
//!     .lr(1e-5)
//!     .run()?;
//! if let Some(checkpoint) = outcome.checkpoint {
//!     println!("trained base: {}", checkpoint.display());
//! }
//! # Ok(()) }
//! ```
//!
//! **What trains and where it lives.** Every weight trains, in fp32 (the one
//! precision the trainable build runs in), with the AdamW moments and the
//! master weights held in system RAM and only the weights and their gradients
//! on the card: the architecture's `model::finetune::ChatTrainer` (for qwen3
//! the same build `brain qwen3 finetune` runs) over `model::fit_controlled`,
//! the one training loop. That is two copies of the parameters on one card (a 0.6B model needs
//! about 4.4 GiB for them), which is the limit of the method: an 8B model
//! needs 61 GiB for the pair and does not fit two 24 GiB cards. The dataset,
//! the replay files and their shares are [`crate::ChatFineTune`]'s, validated
//! against the base's own template before a device is claimed.
//!
//! **Cancellation.** A run whose [`crate::CancelToken`] fires stops at the
//! next step boundary and writes no checkpoint: the optimizer state a resume
//! needs is three times the model, so a cancelled full fine-tune starts over.

use std::path::{Path, PathBuf};

use crate::finetune::{base_digest, block_for, digest, fit_opts, mix_rows, resolve_seed, tokenizer_and_template, Asked, Mix};
use model::training::Capability;

use crate::{CancelToken, Device, Error, FineTuneProgress, FineTuneStatus, Result};

/// The training format the run writes before it is exported, and removes.
const TUNED_FILE: &str = "tuned.brain.safetensors";
/// The packed dataset, inside the out directory.
const PREPARED_DIR: &str = "prepared";
/// The checkpoint directory, inside the out directory.
const CHECKPOINT_DIR: &str = "checkpoint";
/// The weights file of a checkpoint directory.
const WEIGHTS_FILE: &str = "model.safetensors";
/// The training record, inside the out directory beside the checkpoint.
const RECORD_FILE: &str = "training.json";

/// AdamW weight decay of a full fine-tune unless named: none, so the trained
/// weights are pulled towards the data and nowhere else.
const FULL_WEIGHT_DECAY: f32 = 0.0;

/// A full-parameter fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ChatFullFineTune {
    base: String,
    arch: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    replay: Vec<PathBuf>,
    replay_share: Option<f32>,
    weighted_replay: Vec<(PathBuf, f32)>,
    grad_accum: u32,
    out_dir: Option<PathBuf>,
    weight_decay: f32,
    steps: u32,
    lr: f32,
    seed: Option<u64>,
    max_block: Option<u32>,
    device: Device,
    thinking: bool,
    precision: model::TrainPrecision,
}

impl ChatFullFineTune {
    /// A full fine-tune of `base`: a checkpoint file, a checkpoint directory
    /// (a previous full fine-tune's or merge's, or a `transformers` one), or
    /// a `vendor/repo` reference in the model store. The tokenizer and chat
    /// template are read from the checkpoint's directory.
    pub fn from_pretrained(base: impl Into<String>) -> ChatFullFineTune {
        ChatFullFineTune {
            base: base.into(),
            arch: "qwen3".to_string(),
            models_dir: None,
            dataset: None,
            replay: Vec::new(),
            replay_share: None,
            weighted_replay: Vec::new(),
            grad_accum: 1,
            out_dir: None,
            weight_decay: FULL_WEIGHT_DECAY,
            steps: 100,
            lr: 1e-5,
            seed: None,
            max_block: None,
            device: Device::default(),
            thinking: true,
            precision: model::TrainPrecision::default(),
        }
    }

    /// What the run stores each trainable parameter as: the format of the AdamW state (f32, bf16, int8 or logarithmic
    /// 8-bit blocks, held in host memory here), whether the planner may narrow it, and so on - one typed value, parsed
    /// once from `state=int8,sr=on,...` by [`model::TrainPrecision::parse`]. The default is fp32 everywhere. A choice
    /// the architecture does not run is refused by parameter name before anything is loaded.
    pub fn train_precision(mut self, precision: model::TrainPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// The architecture of the base (a `brain_arch` id; default `qwen3`).
    /// Looked up in the same registry as [`crate::study::DocumentStudy`]; an
    /// architecture without a chat fine-tuner is refused before anything is read.
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = id.into();
        self
    }

    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The chat dataset to train on (required).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// Mix every record of another chat dataset into training; see
    /// [`crate::ChatFineTune::replay`].
    pub fn replay(mut self, path: impl Into<PathBuf>) -> Self {
        self.replay.push(path.into());
        self
    }

    /// The share of the draws the [`Self::replay`] files take in all; see
    /// [`crate::ChatFineTune::replay_share`].
    pub fn replay_share(mut self, share: f32) -> Self {
        self.replay_share = Some(share);
        self
    }

    /// Mix another chat dataset into training at `share` of the draws of its
    /// own; see [`crate::ChatFineTune::replay_at`].
    pub fn replay_at(mut self, path: impl Into<PathBuf>, share: f32) -> Self {
        self.weighted_replay.push((path.into(), share));
        self
    }

    /// Micro-batches summed into each optimizer step (default 1).
    pub fn grad_accum(mut self, micro_batches: u32) -> Self {
        self.grad_accum = micro_batches;
        self
    }

    /// Where the checkpoint directory and the packed dataset go (required).
    /// One directory per run.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// AdamW's decoupled weight decay (default 0).
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Optimizer steps (default 100). The first twentieth warms the rate up,
    /// and it decays to a tenth over all of them.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// The peak learning rate (default 1e-5: every weight moves, so the rate
    /// is two orders below a LoRA's).
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// The seed of the batch order (default: a random draw).
    /// [`ChatFullFineTuneOutcome::seed`] and the training record carry the
    /// seed used, and passing it reproduces the run.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The longest training row in tokens; a record past it is refused
    /// (default: exactly the longest record, at least a small minimum row).
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// The device the run is placed on (default: the automatic placement).
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether the model is trained to reason before it answers (default
    /// true); `false` trains it for no-think mode, as
    /// [`crate::ChatFineTune::thinking`] does.
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// Run to completion.
    pub fn run(&self) -> Result<ChatFullFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step to `on_progress` and stopping at
    /// the next step boundary once `cancel` fires; a stopped run writes no
    /// checkpoint.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<ChatFullFineTuneOutcome> {
        let chat = chat_entry(&self.arch)?;
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ChatFullFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ChatFullFineTune: no out directory; call .out_dir(path)".to_string()))?;
        if self.grad_accum == 0 {
            return Err(Error::Backend("ChatFullFineTune: grad_accum must be at least 1".to_string()));
        }
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let open = (chat.checkpoint_path)(&weights, &model_dir);

        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: false };
        let train_samples = asked.read(dataset)?;
        let dataset_records = train_samples.len();
        let mut replay_samples = Vec::new();
        for path in &self.replay {
            replay_samples.extend(asked.read(path)?);
        }
        let mut weighted_samples = Vec::with_capacity(self.weighted_replay.len());
        for (path, share) in &self.weighted_replay {
            weighted_samples.push((asked.read(path)?, *share));
        }
        let Mix { rows, replay_records, .. } = mix_rows(train_samples, replay_samples, self.replay_share, weighted_samples)?;

        let vocab = (chat.vocab)(&open).map_err(Error::Backend)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        // No evaluation runs during a full fine-tune; the validation split is
        // the rows themselves, unread.
        let prepared = data::chat::prepare_chat_samples(&rows, &rows, &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        let block = block_for(prepared.longest_example, self.max_block)?;
        let set = crate::device::resolve(&self.device)?;
        plan_before_load(&open, block, &set)?;

        let seed = resolve_seed(self.seed, None)?;
        let opts = model::FitOpts { grad_accum: self.grad_accum, weight_decay: self.weight_decay, precision: self.precision.clone(), ..fit_opts(self.steps, block, self.lr, seed) };
        let base_digest = base_digest(&open)?;
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress { step: s.step, steps: s.steps, loss: s.loss, lr: s.lr });
            !cancel.is_cancelled()
        };
        let control = model::FitControl { on_step: Some(&mut on_step), ..Default::default() };
        let tuned = out_dir.join(TUNED_FILE);
        let report = (chat.full)(&open, &prepared_dir, &opts, &tuned, control).map_err(|e| Error::Backend(format!("training: {e}")))?;

        let mut outcome = ChatFullFineTuneOutcome {
            status: FineTuneStatus::Completed,
            checkpoint: None,
            weights: None,
            weights_digest: None,
            base_digest: Some(base_digest),
            seed,
            record: None,
            steps: self.steps,
            steps_completed: report.steps_completed,
            initial_loss: Some(report.initial_loss),
            final_loss: report.final_loss,
            train_records: dataset_records,
            replay_records,
            block,
            lr: self.lr,
        };
        if report.interrupted {
            std::fs::remove_file(&tuned).ok();
            outcome.status = FineTuneStatus::Cancelled;
            return Ok(outcome);
        }

        let checkpoint = out_dir.join(CHECKPOINT_DIR);
        export_trained(&chat, &tuned, &model_dir, &checkpoint, None)?;
        std::fs::remove_file(&tuned).ok();
        let weights = checkpoint.join(WEIGHTS_FILE);
        outcome.weights_digest = Some(digest(&weights)?);
        outcome.weights = Some(weights);
        outcome.checkpoint = Some(checkpoint);

        let record_path = out_dir.join(RECORD_FILE);
        std::fs::write(&record_path, serde_json::to_vec_pretty(&outcome.record_json(&base_id)).map_err(|e| Error::Backend(e.to_string()))?)
            .map_err(|e| Error::Backend(format!("{}: {e}", record_path.display())))?;
        outcome.record = Some(record_path);
        Ok(outcome)
    }
}

/// What one full fine-tune did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatFullFineTuneOutcome {
    /// How the run ended.
    pub status: FineTuneStatus,
    /// The checkpoint directory; `None` unless completed.
    pub checkpoint: Option<PathBuf>,
    /// The directory's `model.safetensors`; `None` unless completed.
    pub weights: Option<PathBuf>,
    /// `sha256:<hex>` of the weights file.
    pub weights_digest: Option<String>,
    /// `sha256:<hex>` naming the base the run started from.
    pub base_digest: Option<String>,
    /// The JSON training record written beside the checkpoint; `None` unless
    /// completed.
    pub record: Option<PathBuf>,
    /// The seed of the batch order: the one asked for or the one drawn.
    /// Passing it to [`ChatFullFineTune::seed`] reproduces the run.
    pub seed: u64,
    /// Optimizer steps asked for.
    pub steps: u32,
    /// Steps completed.
    pub steps_completed: u32,
    /// The training loss estimated before the first step.
    pub initial_loss: Option<f32>,
    /// The last step's training loss.
    pub final_loss: Option<f32>,
    /// Records of the dataset.
    pub train_records: usize,
    /// Records of the replay and weighted files.
    pub replay_records: usize,
    /// The training row length, sized to the longest record.
    pub block: u32,
    /// The peak learning rate of the run.
    pub lr: f32,
}

impl ChatFullFineTuneOutcome {
    fn record_json(&self, base_id: &str) -> serde_json::Value {
        serde_json::json!({
            "base": base_id,
            "base_digest": self.base_digest,
            "weights_digest": self.weights_digest,
            "seed": self.seed,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "train_records": self.train_records,
            "replay_records": self.replay_records,
            "block": self.block,
            "lr": self.lr,
        })
    }
}

/// The generic fine-tuners' entry points of architecture `arch`, or the
/// refusal naming it and the capability it lacks.
/// Refuse a full fine-tune that cannot fit the selected cards BEFORE the base is loaded, with the planner's
/// per-category arithmetic. Every tensor trains in fp32; the AdamW moments and the master weights live in host
/// memory ([`model::training::OptimizerState::HostOffload`]), so the cards hold the values, the gradients and one
/// step's activations. The activation term is calibrated on the 27B (`crate::finetune::activation_bytes`) and
/// included when the checkpoint's config names its depth and width. No selected card (a CPU run) is not checked.
fn plan_before_load(open: &Path, block: u32, set: &gpu_core::devices::ComputeSet) -> Result<()> {
    let cards: Vec<u64> = gpu_core::devices::gpus().iter().filter(|d| set.gpus.contains(&d.index)).map(|d| d.identity.vram_bytes).collect();
    if cards.is_empty() {
        return Ok(());
    }
    plan_on_cards(open, block, &cards)
}

/// [`plan_before_load`]'s arithmetic for cards of `cards` bytes. `open` is a checkpoint file or a checkpoint
/// directory (a `transformers` one, single-file or sharded, whose depth and width are in its `config.json`).
fn plan_on_cards(open: &Path, block: u32, cards: &[u64]) -> Result<()> {
    use model::memplan::{plan, training_demands, Authorized, Capacity, Category, Demand, TrainingFootprint};
    let reader = checkpoint::weightio::WeightReader::open_any(open.to_str().ok_or_else(|| Error::Backend(format!("{}: not valid UTF-8", open.display())))?).map_err(|e| Error::Backend(format!("{}: {e}", open.display())))?;
    let trainable: Vec<(String, usize)> = reader.names().map(|n| (n.to_string(), reader.shape(n).map_or(0, |s| s.iter().product::<u64>() as usize))).collect();
    let config = if open.is_dir() {
        let path = open.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&text).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?
    } else {
        reader.config()
    };
    let dim = |keys: &[&str]| keys.iter().find_map(|k| config.get(*k).and_then(|v| v.as_u64()));
    let activation_bytes = match (dim(&["num_hidden_layers", "n_layers"]), dim(&["hidden_size", "d_model"])) {
        (Some(layers), Some(width)) => crate::finetune::activation_bytes(block, layers, width),
        _ => 0,
    };
    let numel: u64 = trainable.iter().map(|(_, n)| *n as u64).sum();
    let footprint = TrainingFootprint {
        frozen_weight_bytes: 0,
        trainable,
        precision: model::memplan::ParamPrecision::FP32,
        moment_slots: 2,
        int8_block: 256,
        optimizer_state_on_host: true,
        activation_bytes,
        recompute_activation_bytes: None,
        saved_boundaries: None,
    };
    let mut demands = training_demands(&footprint).map_err(|e| Error::Backend(e.to_string()))?;
    demands.push(Demand { category: Category::Weights, variants: vec![model::memplan::Variant::new(model::memplan::Lever::Base, "master weights on the host", 0, 4 * numel)] });
    // The trainer runs on the selected cards together; host memory is not bounded by the planner here.
    let capacity = Capacity { device_bytes: cards.iter().sum(), host_bytes: u64::MAX };
    plan(&demands, capacity, &Authorized::none()).map(drop).map_err(|e| Error::Backend(format!("ChatFullFineTune: refusing to load {}: {e}", open.display())))
}

fn chat_entry(arch: &str) -> Result<crate::chat_trainer::ChatEntry> {
    let entry = crate::study::arch(arch)?;
    entry.require(Capability::ChatFineTune)?;
    entry.chat.ok_or_else(|| Error::Unsupported(model::training::Unsupported::new(entry.id, Capability::ChatFineTune)))
}

/// Writes the brain checkpoint `trained` to `checkpoint` as a checkpoint
/// directory in the architecture's own format, with the tokenizer files of
/// `tokenizer_dir` and `adapter` folded in when one is given.
fn export_trained(chat: &crate::chat_trainer::ChatEntry, trained: &Path, tokenizer_dir: &Path, checkpoint: &Path, adapter: Option<&Path>) -> Result<()> {
    (chat.export)(trained, tokenizer_dir, checkpoint, adapter).map_err(Error::Backend)
}

/// Folds the LoRA `adapter` into the weights of `base` it targets
/// (`W + alpha/r * B A`) and writes the result to `out_dir` as a checkpoint
/// directory: the base with the adapter, in one set of weights, that needs no
/// adapter to be served. `base` is whatever [`ChatFullFineTune::from_pretrained`]
/// takes; the tokenizer files are copied from the base's directory. Returns
/// `out_dir`, with the weights file and its digest.
pub fn merge_adapter(base: &str, adapter: &Path, out_dir: &Path) -> Result<MergedCheckpoint> {
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    // Folding is a qwen3 export: the architecture that has one.
    let chat = chat_entry("qwen3")?;
    let open = (chat.checkpoint_path)(&weights, &model_dir);
    export_trained(&chat, &open, &model_dir, out_dir, Some(adapter))?;
    let weights = out_dir.join(WEIGHTS_FILE);
    Ok(MergedCheckpoint { weights_digest: digest(&weights)?, dir: out_dir.to_path_buf(), weights })
}

/// The checkpoint directory [`merge_adapter`] wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct MergedCheckpoint {
    /// The directory.
    pub dir: PathBuf,
    /// Its `model.safetensors`.
    pub weights: PathBuf,
    /// `sha256:<hex>` of the weights file.
    pub weights_digest: String,
}

#[cfg(test)]
mod tests {
    use super::plan_on_cards;

    /// A `transformers` checkpoint directory: weights with no embedded config, and the config beside them.
    fn hf_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-sdk-full-plan-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tensors = vec![("model.embed_tokens.weight".to_string(), vec![64, 32], vec![0.0f32; 64 * 32]), ("lm_head.weight".to_string(), vec![64, 32], vec![0.0f32; 64 * 32])];
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"num_hidden_layers": 2, "hidden_size": 32}"#).unwrap();
        dir
    }

    #[test]
    fn a_checkpoint_directory_is_planned_and_refused_only_when_it_does_not_fit() {
        let dir = hf_dir("fit");
        plan_on_cards(&dir, 64, &[1 << 40]).expect("a checkpoint directory fits a large card");
        let refused = plan_on_cards(&dir, 64, &[1024]).expect_err("nothing fits a 1 KiB card");
        assert!(refused.to_string().contains("refusing to load"), "{refused}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
