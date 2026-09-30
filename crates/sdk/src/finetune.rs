// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements LoRA fine-tuning of chat models on a
// product's own conversations for its clients. If your team needs expertise
// in supervised fine-tuning and held-out evaluation of language models, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`ChatFineTune`]: LoRA fine-tuning of a Qwen3 chat model on a chat
//! dataset, and [`score_chat`]: the held-out measurement that says whether it
//! helped.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::ChatFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
//!     .dataset("train.jsonl")
//!     .held_out("held_out.jsonl")
//!     .out_dir("runs/support-1")
//!     .steps(200)
//!     .run()?;
//! if let (Some(base), Some(tuned)) = (outcome.base_score, outcome.tuned_score) {
//!     println!("held-out loss {:?} -> {:?}", base.loss, tuned.loss);
//! }
//! # Ok(()) }
//! ```
//!
//! The dataset is `generic-messages-v2` JSONL (see
//! [`crate::validate_chat_dataset`]), checked by
//! [`crate::validate_chat_dataset_for`] against the base's own tokenizer and
//! chat template before a device is claimed, and packed by
//! `data::chat::prepare_chat_samples`. Training is
//! `qwen3::finetune::finetune_lora_controlled` over `model::fit_controlled`,
//! the one training loop; scoring is `qwen3::eval::score_chat`. Nothing here
//! is a second implementation of any of them.
//!
//! **Exact resume.** A run whose [`CancelToken`] fires stops at the next
//! step boundary and leaves its training state in the out directory:
//! weights, AdamW moments, step and batch position. Running the same
//! fine-tune again continues it, and ends at the same adapter an
//! uninterrupted run would have produced. A state from a run with different
//! data, options or starting point is refused, never continued.

use std::path::{Path, PathBuf};

use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;

use crate::{CancelToken, Device, Error, Result};

/// Where one run keeps its files, inside [`ChatFineTune::out_dir`].
const PREPARED_DIR: &str = "prepared";
const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";

/// Training rows are at least this long, so a slightly longer next dataset
/// does not change the row length.
const MIN_BLOCK: u32 = 64;

/// A LoRA fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ChatFineTune {
    base: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    replay: Vec<PathBuf>,
    continue_from: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    adapter_id: Option<String>,
    rank: Option<u32>,
    alpha: Option<f32>,
    steps: u32,
    lr: f32,
    seed: u64,
    max_block: Option<u32>,
    checkpoint_every: u32,
    cycle: u64,
    device: Device,
}

impl ChatFineTune {
    /// A fine-tune of `base`: a checkpoint file, a `<models>/<vendor>/<repo>`
    /// directory, or a `vendor/repo` reference in the model store. The
    /// tokenizer (`tokenizer.json`) and chat template are read from the
    /// checkpoint's directory, as training and serving both read them.
    pub fn from_pretrained(base: impl Into<String>) -> ChatFineTune {
        ChatFineTune {
            base: base.into(),
            models_dir: None,
            dataset: None,
            held_out: None,
            replay: Vec::new(),
            continue_from: None,
            out_dir: None,
            adapter_id: None,
            rank: None,
            alpha: None,
            steps: 100,
            lr: 3e-4,
            seed: 1337,
            max_block: None,
            checkpoint_every: 0,
            cycle: 0,
            device: Device::default(),
        }
    }

    /// The models directory a `vendor/repo` base resolves in (default:
    /// `BRAIN_MODELS_DIR`, then the standard location).
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The chat dataset to train on (required).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// Records to score base and tuned on, never trained on. Without it the
    /// outcome's scores are `None`.
    pub fn held_out(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out = Some(path.into());
        self
    }

    /// Mix every record of another chat dataset into training - earlier
    /// experience replayed so the new data does not overwrite it. May be
    /// called more than once; each file is validated like the dataset. The
    /// mix is exactly the union of the files: to replay a fraction, pass a
    /// file holding that fraction.
    pub fn replay(mut self, path: impl Into<PathBuf>) -> Self {
        self.replay.push(path.into());
        self
    }

    /// Continue training this adapter (a file [`ChatFineTune`] or brain's
    /// LoRA trainer wrote) instead of starting a fresh one. Its rank and
    /// alpha are the run's; its digest is recorded as the new adapter's
    /// parent (`trained_from`).
    pub fn continue_from(mut self, adapter: impl Into<PathBuf>) -> Self {
        self.continue_from = Some(adapter.into());
        self
    }

    /// Where the adapter, its training record, the packed dataset and any
    /// resume state go (required). One directory per run.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The id written on the adapter's card (default `<base id>:local:chat:latest`).
    pub fn adapter_id(mut self, id: impl Into<String>) -> Self {
        self.adapter_id = Some(id.into());
        self
    }

    /// LoRA rank (default 8). When continuing an adapter the default is its
    /// own rank, and a different one is refused.
    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = Some(rank);
        self
    }

    /// LoRA alpha; the update is scaled by `alpha / rank` (default `2 * rank`,
    /// or a continued adapter's own alpha - a different one is refused).
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Optimizer steps (default 100). The first fifth warms the rate up, and
    /// it decays over all of them.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Peak learning rate (default 3e-4); it decays to a tenth of this.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Seeds the fresh adapter's initialisation and the batch order
    /// (default 1337).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The longest training row allowed. Rows are sized to the longest
    /// record; a record longer than this is refused by name rather than
    /// trained on truncated.
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// Also save the resume state every this-many steps (default 0: only when
    /// cancelled), so a crash loses at most this many steps.
    pub fn checkpoint_every(mut self, steps: u32) -> Self {
        self.checkpoint_every = steps;
        self
    }

    /// The caller's loop iteration this run belongs to, recorded in the
    /// adapter's training provenance (default 0).
    pub fn cycle(mut self, cycle: u64) -> Self {
        self.cycle = cycle;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Run to completion.
    pub fn run(&self) -> Result<ChatFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step to `on_progress` and stopping at
    /// the next step boundary once `cancel` fires. A cancelled run exports no
    /// adapter; it leaves its resume state
    /// ([`ChatFineTuneOutcome::resume_state`]) and running the same fine-tune
    /// again continues it.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<ChatFineTuneOutcome> {
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no out directory; call .out_dir(path)".to_string()))?;
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let weights_str = utf8(&weights)?;

        // Every file is checked against the base's own template before a
        // device is claimed.
        let train_samples = read_checked(dataset, &model_dir)?;
        let mut replay_samples = Vec::new();
        for path in &self.replay {
            replay_samples.extend(read_checked(path, &model_dir)?);
        }
        let held_out = self.held_out.as_deref().map(|path| read_checked(path, &model_dir)).transpose()?;
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;

        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha)?;

        let mut training = train_samples.clone();
        training.extend(replay_samples.iter().cloned());
        // The packed validation split is only read for an eval the run does
        // not ask for; it is the held-out set when there is one.
        let val = held_out.as_deref().unwrap_or(&training[..]);
        let cfg = qwen3::QwenConfig::from_json(&checkpoint::weightio::WeightReader::open(weights_str).map_err(|e| Error::Backend(format!("{weights_str}: {e}")))?.config());
        let prepared_dir = out_dir.join(PREPARED_DIR);
        let prepared = data::chat::prepare_chat_samples(&training, val, &tok, &tmpl, cfg.vocab as usize, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        let block = block_for(prepared.longest_example, self.max_block)?;

        crate::device::resolve(&self.device)?;
        let base_score = held_out.as_ref().map(|records| score_records(weights_str, None, &tok, &tmpl, records, block));

        let opts = fit_opts(self.steps, block, self.lr, self.seed);
        let base_digest = digest(&weights)?;
        let identity = serde_json::json!({
            "base": base_digest,
            "dataset": digest(dataset)?,
            "replay": self.replay.iter().map(|p| digest(p)).collect::<Result<Vec<_>>>()?,
            "parent": parent,
            "rank": rank,
            "alpha_bits": alpha.to_bits(),
        });
        let state_path = out_dir.join(STATE_FILE);
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress { step: s.step, steps: s.steps, loss: s.loss, lr: s.lr });
            !cancel.is_cancelled()
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity };
        let start = match &self.continue_from {
            Some(path) => qwen3::finetune::LoraStart::Continue(utf8(path)?),
            None => qwen3::finetune::LoraStart::Fresh,
        };
        let (report, trained) = qwen3::finetune::finetune_lora_controlled(weights_str, &prepared_dir, &opts, rank, alpha, &start, control).map_err(|e| Error::Backend(format!("training: {e}")))?;

        let mut outcome = ChatFineTuneOutcome {
            status: FineTuneStatus::Completed,
            adapter: None,
            adapter_digest: None,
            record: None,
            resume_state: None,
            steps: self.steps,
            steps_completed: report.steps_completed,
            resumed_at: report.resumed_at,
            initial_loss: Some(report.initial_loss),
            final_loss: report.final_loss,
            train_records: train_samples.len(),
            replay_records: replay_samples.len(),
            block,
            rank,
            alpha,
            trained_from: parent.clone(),
            base_score,
            tuned_score: None,
        };
        if report.interrupted {
            outcome.status = FineTuneStatus::Cancelled;
            outcome.resume_state = Some(state_path);
            return Ok(outcome);
        }

        let adapter_path = out_dir.join(ADAPTER_FILE);
        let adapter_id = self.adapter_id.clone().unwrap_or_else(|| format!("{base_id}:local:chat:latest"));
        let provenance = checkpoint::st::TrainingProvenance {
            code_revision: format!("brain {}", env!("CARGO_PKG_VERSION")),
            regime: "sft_lora".to_string(),
            seed: self.seed,
            hyperparams: serde_json::json!({
                "rank": rank,
                "alpha": alpha,
                "steps": self.steps,
                "lr": self.lr,
                "block": block,
                "train_records": outcome.train_records,
                "replay_records": outcome.replay_records,
            }),
            environment: gpu_core::backend_name().to_string(),
            gate: None,
            trained_from: parent,
            base_digest: Some(base_digest),
            cycle: self.cycle,
        };
        let dataset_id = digest(dataset)?;
        qwen3::lora::save_adapter_with_lineage(utf8(&adapter_path)?, &trained, &adapter_id, &base_id, Some(&dataset_id), Some(provenance))
            .map_err(|e| Error::Backend(format!("{}: saving the adapter: {e}", adapter_path.display())))?;
        drop(trained);
        outcome.adapter_digest = Some(digest(&adapter_path)?);
        let adapter_str = utf8(&adapter_path)?;
        outcome.tuned_score = held_out.as_ref().map(|records| score_records(weights_str, Some(adapter_str), &tok, &tmpl, records, block));
        outcome.adapter = Some(adapter_path);

        let record_path = out_dir.join(RECORD_FILE);
        std::fs::write(&record_path, serde_json::to_vec_pretty(&outcome.record_json(&base_id, &dataset_id)).map_err(|e| Error::Backend(e.to_string()))?)
            .map_err(|e| Error::Backend(format!("{}: {e}", record_path.display())))?;
        outcome.record = Some(record_path);
        // The run is exported; its state would only resume into the same
        // adapter again.
        std::fs::remove_file(&state_path).ok();
        Ok(outcome)
    }
}

/// One completed optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FineTuneProgress {
    /// Steps completed in the whole run (a resumed run continues the count).
    pub step: u32,
    pub steps: u32,
    /// This step's training loss.
    pub loss: f32,
    pub lr: f32,
}

/// How a fine-tune ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FineTuneStatus {
    /// Every step ran; the adapter and its record are written.
    Completed,
    /// The cancel token stopped it; the resume state is written instead.
    Cancelled,
}

/// What one fine-tune did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatFineTuneOutcome {
    pub status: FineTuneStatus,
    /// The adapter file; `None` unless completed.
    pub adapter: Option<PathBuf>,
    /// `sha256:<hex>` of the adapter file - the digest a [`crate::ChatPipeline`]
    /// serving it reports in its identity.
    pub adapter_digest: Option<String>,
    /// The JSON training record written beside the adapter.
    pub record: Option<PathBuf>,
    /// The resume state a cancelled run left; running again continues it.
    pub resume_state: Option<PathBuf>,
    pub steps: u32,
    /// Steps completed in the whole run, including any before a resume.
    pub steps_completed: u32,
    /// The step this call resumed at, when it continued a saved state.
    pub resumed_at: Option<u32>,
    /// The training loss estimated before this call's first step.
    pub initial_loss: Option<f32>,
    /// The last step's training loss; `None` when this call ran no step.
    pub final_loss: Option<f32>,
    pub train_records: usize,
    pub replay_records: usize,
    /// The training row length, sized to the longest record.
    pub block: u32,
    pub rank: u32,
    pub alpha: f32,
    /// The digest of the adapter this run continued, if it continued one.
    pub trained_from: Option<String>,
    /// The base on the held-out records; `None` without [`ChatFineTune::held_out`].
    pub base_score: Option<HeldOutScore>,
    /// Base plus the new adapter on the same records; `None` unless
    /// completed with a held-out set.
    pub tuned_score: Option<HeldOutScore>,
}

impl ChatFineTuneOutcome {
    fn record_json(&self, base_id: &str, dataset_id: &str) -> serde_json::Value {
        let score = |s: &Option<HeldOutScore>| {
            s.as_ref().map(|s| serde_json::json!({ "loss": s.loss, "token_accuracy": s.token_accuracy, "positions": s.positions, "records": s.records, "skipped": s.skipped }))
        };
        serde_json::json!({
            "base": base_id,
            "dataset": dataset_id,
            "adapter_digest": self.adapter_digest,
            "trained_from": self.trained_from,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "train_records": self.train_records,
            "replay_records": self.replay_records,
            "block": self.block,
            "rank": self.rank,
            "alpha": self.alpha,
            "base_score": score(&self.base_score),
            "tuned_score": score(&self.tuned_score),
        })
    }
}

/// Teacher-forced cross-entropy of a model on held-out chat records, over
/// exactly the positions training supervises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldOutScore {
    /// Mean per-token cross-entropy over the supervised positions (lower is
    /// better); `None` when no position was scored.
    pub loss: Option<f32>,
    /// Fraction of supervised positions whose greedy next token was right;
    /// `None` when no position was scored.
    pub token_accuracy: Option<f64>,
    /// Supervised positions scored.
    pub positions: usize,
    /// Records scored.
    pub records: usize,
    /// Records skipped (longer than the scoring row, or not encodable).
    pub skipped: usize,
}

/// Score `base` - optionally with `adapter` folded in, exactly as serving it
/// would - on the held-out chat records in `held_out`. `base` is resolved
/// like [`ChatFineTune::from_pretrained`]'s; the tokenizer and template come
/// from its directory. The scoring row fits the longest record, so none is
/// skipped for length.
pub fn score_chat(base: &str, adapter: Option<&Path>, held_out: &Path) -> Result<HeldOutScore> {
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let weights_str = utf8(&weights)?;
    checkpoint::weightio::WeightReader::open(weights_str).map_err(|e| Error::Backend(format!("{weights_str}: {e}")))?;
    let records = read_checked(held_out, &model_dir)?;
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let mut longest = 0usize;
    for (i, r) in records.iter().enumerate() {
        let (ids, _) = r.encode(&tok, &tmpl).map_err(|e| Error::Backend(format!("{}: record {}: {e}", held_out.display(), i + 1)))?;
        longest = longest.max(ids.len());
    }
    let adapter = match adapter {
        Some(path) => {
            checkpoint::st::read_card(utf8(path)?).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
            Some(utf8(path)?)
        }
        None => None,
    };
    Ok(score_records(weights_str, adapter, &tok, &tmpl, &records, block_for(longest, None)?))
}

fn score_records(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, records: &[ChatSample], block: u32) -> HeldOutScore {
    let s = qwen3::eval::score_chat(weights, adapter, tok, tmpl, records, block);
    HeldOutScore { loss: (s.positions > 0).then_some(s.loss), token_accuracy: s.token_accuracy, positions: s.positions, records: s.samples, skipped: s.skipped }
}

/// The LoRA rank and alpha of a run, and the digest of the adapter it
/// continues: a fresh adapter is rank 8 (or `rank`) at alpha `2 * rank`; a
/// continued one is its own rank - a different requested rank is refused -
/// and its own alpha. `alpha` overrides either.
pub(crate) fn lora_shape(continue_from: Option<&Path>, rank: Option<u32>, alpha: Option<f32>) -> Result<(u32, f32, Option<String>)> {
    let (rank, default_alpha, parent) = match continue_from {
        None => {
            let rank = rank.unwrap_or(8);
            (rank, 2.0 * rank as f32, None)
        }
        Some(path) => {
            let card = checkpoint::st::read_card(utf8(path)?).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
            let adapter = card.and_then(|c| c.adapter).ok_or_else(|| Error::Backend(format!("{}: not an adapter file (no adapter card)", path.display())))?;
            let own = adapter.rank.ok_or_else(|| Error::Backend(format!("{}: the adapter card has no rank", path.display())))?;
            if let Some(asked) = rank.filter(|&asked| asked != own) {
                return Err(Error::Backend(format!("{}: a rank-{own} adapter cannot be continued at rank {asked}", path.display())));
            }
            (own, adapter.alpha.unwrap_or(own as f32), Some(digest(path)?))
        }
    };
    Ok((rank, alpha.unwrap_or(default_alpha), parent))
}

/// Validate `path` against the base's tokenizer and template, then read it.
fn read_checked(path: &Path, model_dir: &Path) -> Result<Vec<ChatSample>> {
    let summary = crate::validate_chat_dataset_for(path, model_dir).map_err(Error::Backend)?;
    if summary.trained_messages == 0 {
        return Err(Error::Backend(format!("{}: holds no supervised turns (`\"train\": true`)", path.display())));
    }
    ChatSample::from_jsonl(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

pub(crate) fn tokenizer_and_template(model_dir: &Path) -> Result<(QwenBpe, ChatTemplate)> {
    let tmpl = ChatTemplate::from_model_dir(model_dir).map_err(|e| Error::Backend(format!("{}: {e}", model_dir.display())))?;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = QwenBpe::from_file(utf8(&tok_path)?).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;
    Ok((tok, tmpl))
}

/// The smallest power-of-two row (at least [`MIN_BLOCK`]) that holds the
/// longest record, capped at `max`: a record longer than the cap is refused,
/// since training on a row that cannot hold it would supervise nothing of it.
pub(crate) fn block_for(longest: usize, max: Option<u32>) -> Result<u32> {
    if let Some(max) = max {
        if longest > max as usize {
            return Err(Error::Backend(format!("the longest record is {longest} tokens, past max_block {max}; raise max_block or shorten the record")));
        }
    }
    let mut block = MIN_BLOCK;
    while (block as usize) < longest {
        block *= 2;
    }
    Ok(max.map_or(block, |m| block.min(m)))
}

pub(crate) fn fit_opts(steps: u32, block: u32, lr: f32, seed: u64) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: block,
        lr,
        min_lr: lr / 10.0,
        warmup: steps / 5,
        decay_iters: steps,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        // Held-out evidence is `score_chat`'s, measured on the served form;
        // the loop's own eval and early stop are off.
        eval_interval: 0,
        eval_batches: 0,
        seed,
        // The resume state is the checkpoint; no whole-model writes.
        checkpoint_secs: 0,
        mask_before: None,
        mask_per_line: false,
        align_to_lines: false,
        patience: 0,
        adam: Default::default(),
    }
}

pub(crate) fn digest(path: &Path) -> Result<String> {
    brain_modelstore::fetch::file_digest(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

pub(crate) fn utf8(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| Error::Backend(format!("{}: not a UTF-8 path", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row holds the longest record, grows in powers of two from 64, and
    /// never passes the caller's cap - a record past it is refused by size.
    #[test]
    fn the_training_row_fits_the_longest_record_within_the_cap() {
        assert_eq!(block_for(10, None).unwrap(), 64);
        assert_eq!(block_for(65, None).unwrap(), 128);
        assert_eq!(block_for(100, Some(100)).unwrap(), 100);
        assert_eq!(block_for(70, Some(512)).unwrap(), 128);
        assert!(block_for(600, Some(512)).unwrap_err().to_string().contains("max_block"));
    }
}
