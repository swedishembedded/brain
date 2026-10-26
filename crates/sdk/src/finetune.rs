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
//! **The mix.** Training draws rows uniformly with replacement from the
//! dataset and every replay file. [`ChatFineTune::replay`] files join the
//! dataset as a plain union, or at [`ChatFineTune::replay_share`] of the
//! draws together; [`ChatFineTune::replay_at`] gives one file a share of its
//! own. A share is kept by listing each group of records as many times as
//! its share needs ([`mix_repeats`]), so a replay set many times the
//! dataset, or many times smaller, is drawn as often as asked.
//!
//! **Monitoring and selection.** With [`ChatFineTune::eval_every`] the run
//! scores a monitoring set ([`ChatFineTune::monitor`], else the held-out set)
//! as it trains and records the curve ([`ChatFineTuneOutcome::curve`]); with
//! [`ChatFineTune::keep_best`] or a [`ChatFineTune::patience`] it exports the
//! adapter of the evaluation with the lowest monitoring loss instead of the
//! last step's, and a patience stops it once that loss has not improved for
//! that many evaluations. Which step was exported, and why, is on the
//! outcome, in the training record and on the adapter's card.
//!
//! **Exact resume.** A run whose [`CancelToken`] fires stops at the next
//! step boundary and leaves its training state in the out directory:
//! weights, AdamW moments, step and batch position, and the selection so
//! far. Running the same fine-tune again continues it, and ends at the same
//! adapter an uninterrupted run would have produced. A state from a run with
//! different data, options or starting point is refused, never continued.

/// One position's teacher-forced readout, re-exported from the qwen35 model
/// crate so both scoring sides of a train-serve agreement check report the same
/// type.
pub use qwen35::score::TokenScore;

use std::path::{Path, PathBuf};

use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;

use model::training::Capability;

use crate::{CancelToken, Device, Error, Result};

/// Where one run keeps its files, inside [`ChatFineTune::out_dir`].
const PREPARED_DIR: &str = "prepared";
const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";

/// Training rows are at least this long, so a tiny dataset does not train on
/// a degenerate row.
const MIN_BLOCK: u32 = 64;

/// The learning rate a scoring pass's [`model::FitOpts`] carries. Scoring takes
/// no step, so the value is never applied; it is the fine-tune default's
/// neighbourhood so nothing downstream sees an unusual setting.
const SCORE_LR: f32 = 1e-4;

/// A LoRA fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ChatFineTune {
    base: String,
    arch: Option<String>,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    held_out_text: Option<PathBuf>,
    monitor: Option<PathBuf>,
    replay: Vec<PathBuf>,
    replay_share: Option<f32>,
    weighted_replay: Vec<(PathBuf, f32)>,
    grad_accum: u32,
    continue_from: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    adapter_id: Option<String>,
    rank: Option<u32>,
    alpha: Option<f32>,
    tuning: model::adapter::device::AdapterTuning,
    weight_decay: f32,
    cooldown_steps: Option<u32>,
    steps: u32,
    lr: f32,
    seed: Option<u64>,
    max_block: Option<u32>,
    checkpoint_every: u32,
    eval_every: u32,
    patience: u32,
    keep_best: bool,
    evaluations_dir: Option<PathBuf>,
    cycle: u64,
    device: Device,
    bf16_base: bool,
    int8_base: bool,
    nf4_base: bool,
    f32_base: bool,
    base_format: Option<String>,
    gemm_precision: crate::GemmPrecision,
    thinking: bool,
    pack_answers: bool,
}

/// The architecture a [`ChatFineTune`] trains, with the tier its frozen base
/// is held at. The one place that names a trainer: `qwen3` and `qwen35` each
/// have their own checkpoint reader and LoRA trainer over the shared
/// `model::fit_controlled` loop.
enum ChatFamily {
    /// Int8 over bf16 over fp32.
    Qwen3(qwen3::Dtype),
    /// NF4 over bf16 over fp32 (the 27B trains on one card only at the first two).
    Qwen35(qwen3::Dtype),
}

/// A trained LoRA, whichever trainer produced it.
enum TrainedLora {
    Qwen3(qwen3::finetune::Trained),
    Qwen35(Box<qwen35::model::Qwen35>),
}

impl TrainedLora {
    fn save(&self, path: &str, card_id: &str, base_id: &str, dataset_id: Option<&str>, provenance: checkpoint::st::TrainingProvenance) -> std::io::Result<()> {
        match self {
            TrainedLora::Qwen3(t) => t.save_adapter_with_lineage(path, card_id, base_id, dataset_id, Some(provenance)),
            TrainedLora::Qwen35(m) => qwen35::lora::save_adapter_with_lineage(path, m, card_id, base_id, dataset_id, Some(provenance)),
        }
    }
}

impl ChatFamily {
    /// The family of `arch_id` with its base held as the narrowest of the flags (nf4, then int8, then
    /// bf16, else fp32), or as `format` when one is given; a format the family's trainer does not
    /// hold is refused naming the family and its supported set.
    fn of(arch_id: &str, bf16: bool, int8: bool, nf4: bool, format: Option<&str>) -> Result<ChatFamily> {
        let bases = match arch_id {
            "qwen3" => qwen3::finetune::TRAIN_BASES,
            "qwen35" => qwen35::base_source::TRAIN_BASES,
            other => return Err(Error::Unsupported(model::training::Unsupported::new(other, Capability::ChatLoraFineTune))),
        };
        let dtype = match format {
            Some(_) if bf16 || int8 || nf4 => return Err(Error::Backend("ChatFineTune: base_format and the bf16_base, int8_base and nf4_base switches name the same setting; give one".to_string())),
            Some(spec) => bases.parse(spec).map_err(Error::Backend)?,
            None => {
                let narrowest = if nf4 {
                    qwen3::Dtype::NF4
                } else if int8 {
                    qwen3::Dtype::I8
                } else if bf16 {
                    qwen3::Dtype::BF16
                } else {
                    qwen3::Dtype::F32
                };
                bases.check(narrowest).map_err(|why| Error::Backend(format!("ChatFineTune: {why}")))?
            }
        };
        Ok(if arch_id == "qwen3" { ChatFamily::Qwen3(dtype) } else { ChatFamily::Qwen35(dtype) })
    }

    /// The tier name recorded on an outcome.
    fn tier_name(&self) -> &'static str {
        match self {
            ChatFamily::Qwen3(dtype) | ChatFamily::Qwen35(dtype) => model::policy::format_name(*dtype),
        }
    }

    /// The same family at tier `tier`.
    fn with_tier(&self, tier: BaseTierChoice) -> ChatFamily {
        match self {
            ChatFamily::Qwen3(_) => ChatFamily::Qwen3(match tier {
                BaseTierChoice::Bf16 => qwen3::Dtype::BF16,
                BaseTierChoice::Quantised => qwen3::Dtype::I8,
                BaseTierChoice::F32 => qwen3::Dtype::F32,
            }),
            ChatFamily::Qwen35(_) => ChatFamily::Qwen35(match tier {
                BaseTierChoice::Bf16 => qwen3::Dtype::BF16,
                BaseTierChoice::Quantised => qwen3::Dtype::NF4,
                BaseTierChoice::F32 => qwen3::Dtype::F32,
            }),
        }
    }

    /// Device bytes of `params` parameters held at this family's quantised base: group-wise int8 for `qwen3`
    /// (one byte plus an f32 scale per group), NF4 for `qwen35` (half a byte plus an f32 scale per 32 weights).
    fn quantised_base_bytes(&self, params: u64) -> u64 {
        match self {
            ChatFamily::Qwen3(_) => params + 4 * params.div_ceil(model::int8::GROUP as u64),
            ChatFamily::Qwen35(_) => params / 2 + 4 * params.div_ceil(32),
        }
    }

    /// `(parameters, layers, model width)` of the checkpoint at `open`, what the tier choice sizes a base by.
    fn base_shape(&self, open: &Path) -> Result<(u64, u64, u64)> {
        match self {
            ChatFamily::Qwen3(_) => {
                let cfg = qwen3::checkpoint_config(utf8(open)?).map_err(Error::Backend)?;
                Ok((cfg.param_list().iter().map(|(_, n)| *n as u64).sum(), cfg.n_layers as u64, cfg.d_model as u64))
            }
            ChatFamily::Qwen35(_) => {
                let cfg = qwen35::base_source::open_base(open).map_err(Error::Backend)?.cfg;
                Ok((cfg.param_list().iter().map(|(_, n)| *n as u64).sum(), cfg.n_layers as u64, cfg.d_model as u64))
            }
        }
    }

    /// The vocabulary the checkpoint at `open` was trained with.
    fn vocab(&self, open: &Path) -> Result<usize> {
        match self {
            ChatFamily::Qwen3(_) => Ok(qwen3::checkpoint_config(utf8(open)?).map_err(Error::Backend)?.vocab as usize),
            ChatFamily::Qwen35(_) => Ok(qwen35::base_source::open_base(open).map_err(Error::Backend)?.cfg.vocab as usize),
        }
    }

    /// The tier held-out scoring runs at; `None` where the family cannot score.
    fn scoring_tier(&self) -> Option<qwen3::Dtype> {
        match self {
            ChatFamily::Qwen3(tier) => Some(*tier),
            ChatFamily::Qwen35(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn train(
        &self,
        open: &Path,
        prepared_dir: &Path,
        opts: &model::FitOpts,
        rank: u32,
        alpha: f32,
        tuning: model::adapter::device::AdapterTuning,
        continue_from: Option<&Path>,
        control: model::FitControl<'_>,
        held_out: &[&Path],
    ) -> std::io::Result<(model::FitReport, TrainedLora, Vec<model::ScoredSet>)> {
        match self {
            ChatFamily::Qwen3(tier) => {
                let start = match continue_from {
                    Some(path) => qwen3::finetune::LoraStart::Continue(path.to_str().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "adapter path is not UTF-8"))?),
                    None => qwen3::finetune::LoraStart::Fresh,
                };
                let weights = open.to_str().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "checkpoint path is not UTF-8"))?;
                let (report, trained) = qwen3::finetune::finetune_lora_tuned(weights, prepared_dir, opts, rank, alpha, tuning, &start, control, *tier)?;
                Ok((report, TrainedLora::Qwen3(trained), Vec::new()))
            }
            ChatFamily::Qwen35(dtype) => {
                let run = qwen35::finetune::StreamedRun { dtype: *dtype, recompute: qwen35::base_source::is_frozen_only(*dtype), long: qwen35::finetune::LongSeqOverrides::default() };
                let (report, trained, scored) = qwen35::finetune::finetune_lora_scored_tuned(open, prepared_dir, opts, rank, alpha, tuning, continue_from, &run, control, held_out)?;
                Ok((report, TrainedLora::Qwen35(Box::new(trained)), scored))
            }
        }
    }
}

/// The tier the automatic choice picks, before it is named per family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BaseTierChoice {
    /// The device's native 16-bit floating-point tensor-core tier.
    Bf16,
    /// A quantised base (int8 for `qwen3`, NF4 for `qwen35`).
    Quantised,
    /// The exact fp32 reference tier.
    F32,
}

/// Activation bytes of one `block`-token row of a `layers`-deep, `width`-wide decoder ([`model::memplan::calibrated_activation_bytes`]).
pub(crate) fn activation_bytes(block: u32, layers: u64, width: u64) -> u64 {
    model::memplan::calibrated_activation_bytes(block, layers, width)
}

/// The frozen base's tier when the caller named none: **the accelerator's native floating-point tensor-core
/// tier by default, a quantised base only when that does not fit.**
///
/// * No selected card with native 16-bit tensor cores (compute capability 8.0 and up: a CPU, an older or
///   non-CUDA card): fp32, the exact reference, as before.
/// * Otherwise bf16 (the native tier brain trains in; there is no fp8 training path yet - see knowledge 336)
///   when the bf16 base plus the activations of a `block`-token row fit: on the largest single card for
///   `qwen35` (its trainer is one card), on the cards together for `qwen3` (it pipelines).
/// * Otherwise the quantised base (NF4 for `qwen35`, int8 for `qwen3`), which is also what a card too small
///   for the bf16 base needs; if even that cannot fit the caller is told.
///
/// `cards` are the selected cards' total memory in bytes, `native` whether any has the native tier.
pub(crate) fn choose_base_tier(single_card: bool, native: bool, params: u64, layers: u64, width: u64, block: u32, quantised_bytes: u64, cards: &[u64]) -> std::result::Result<(BaseTierChoice, String), String> {
    use model::memplan::{plan, Authorized, Capacity, Category, Demand, Lever, Variant};
    if !native {
        return Ok((BaseTierChoice::F32, "no selected device has native bf16 tensor cores: the fp32 reference tier".to_string()));
    }
    let activations = activation_bytes(block, layers, width);
    let room = if single_card { cards.iter().copied().max().unwrap_or(0) } else { cards.iter().sum() };
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    if room == 0 {
        // A device whose memory cannot be queried: the native tier, as before.
        return Ok((BaseTierChoice::Bf16, format!("native bf16 tensor cores; the device's memory is unknown, so the bf16 base and a {block}-token row (about {:.0} GiB) are not checked", gib(2 * params + activations))));
    }
    // The planner decides: the bf16 base is the stored form; converting it to the quantised base is the one lever
    // this caller authorizes, and it is only pulled when the bf16 base and the activations do not fit.
    let weights = Demand::with_variants(Category::Weights, vec![Variant::new(Lever::Base, "bf16 base", 2 * params, 0), Variant::new(Lever::ConvertWeights, "quantised base", quantised_bytes, 0)]).map_err(|e| e.to_string())?;
    let demands = [weights, Demand::fixed(Category::Activations, format!("{block}-token row"), activations)];
    let report = plan(&demands, Capacity { device_bytes: room, host_bytes: 0 }, &Authorized::only(&[Lever::ConvertWeights])).map_err(|e| format!("refusing to load: {e}"))?;
    let need = report.device_bytes;
    if report.decisions.is_empty() {
        Ok((BaseTierChoice::Bf16, format!("native bf16 tensor cores; the bf16 base and a {block}-token row need about {:.0} GiB of {:.0} GiB", gib(need), gib(room))))
    } else {
        Ok((BaseTierChoice::Quantised, format!("the bf16 base and a {block}-token row need about {:.0} GiB, more than the {:.0} GiB available: quantised base ({:.0} GiB with the row)", gib(2 * params + activations), gib(room), gib(need))))
    }
}

impl ChatFineTune {
    /// A fine-tune of `base`: a checkpoint file, a `<models>/<vendor>/<repo>`
    /// directory, or a `vendor/repo` reference in the model store. The
    /// tokenizer (`tokenizer.json`) and chat template are read from the
    /// checkpoint's directory, as training and serving both read them.
    pub fn from_pretrained(base: impl Into<String>) -> ChatFineTune {
        ChatFineTune {
            base: base.into(),
            arch: None,
            models_dir: None,
            dataset: None,
            held_out: None,
            held_out_text: None,
            monitor: None,
            replay: Vec::new(),
            replay_share: None,
            weighted_replay: Vec::new(),
            grad_accum: 1,
            continue_from: None,
            out_dir: None,
            adapter_id: None,
            rank: None,
            alpha: None,
            tuning: Default::default(),
            weight_decay: LORA_WEIGHT_DECAY,
            cooldown_steps: None,
            steps: 100,
            lr: 3e-4,
            seed: None,
            max_block: None,
            checkpoint_every: 0,
            eval_every: 0,
            patience: 0,
            keep_best: false,
            evaluations_dir: None,
            cycle: 0,
            device: Device::default(),
            bf16_base: false,
            int8_base: false,
            nf4_base: false,
            f32_base: false,
            base_format: None,
            gemm_precision: crate::GemmPrecision::Auto,
            thinking: true,
            pack_answers: false,
        }
    }

    /// The architecture of the base (a `brain_arch` id). Unnamed, it is the one
    /// the base's `config.json` names (a Hugging Face checkpoint of the 27B is
    /// `qwen35`), else `qwen3`. Looked up in the same registry as
    /// [`crate::study::DocumentStudy`]; an architecture without a chat
    /// fine-tuner is refused before anything is read.
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = Some(id.into());
        self
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

    /// A second set of records to score base and tuned on, never trained on:
    /// held-out text of another kind than [`Self::held_out`] (a writer's own
    /// passages beside the dialogues it is asked as), reported apart in the
    /// outcome so a loss over the one is not drowned in the other.
    pub fn held_out_text(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out_text = Some(path.into());
        self
    }

    /// Records scored during training, every [`Self::eval_every`] steps,
    /// never trained on: the curve, and what the best adapter is selected on.
    /// Without it the held-out set is monitored - and a selection made on it
    /// biases the held-out score it is then measured by, so a caller that
    /// decides anything on that score gives the run a monitoring set of its own.
    pub fn monitor(mut self, path: impl Into<PathBuf>) -> Self {
        self.monitor = Some(path.into());
        self
    }

    /// Score the monitoring set every this many steps, and at the last step
    /// of a run that selects on it (default 0: never). Each evaluation is
    /// one deterministic pass over every monitoring record.
    pub fn eval_every(mut self, steps: u32) -> Self {
        self.eval_every = steps;
        self
    }

    /// Stop once the monitoring loss has not improved for this many
    /// evaluations, and export the adapter of the best one (default 0:
    /// never stop). Needs [`Self::eval_every`].
    pub fn patience(mut self, evaluations: u32) -> Self {
        self.patience = evaluations;
        self
    }

    /// Export the adapter of the evaluation with the lowest monitoring loss
    /// rather than the last step's, running every step (default off; a
    /// [`Self::patience`] implies it). Needs [`Self::eval_every`].
    pub fn keep_best(mut self, on: bool) -> Self {
        self.keep_best = on;
        self
    }

    /// Write the adapter of every evaluation to `dir` as `step-<N>.safetensors`
    /// (N the steps trained), so the one exported - the best by monitoring
    /// loss, or the last - is not the only one a caller can choose from by
    /// some other measure. The directory is created. Needs
    /// [`Self::eval_every`].
    pub fn keep_evaluations(mut self, dir: impl Into<PathBuf>) -> Self {
        self.evaluations_dir = Some(dir.into());
        self
    }

    /// Mix every record of another chat dataset into training - earlier
    /// experience replayed so the new data does not overwrite it. May be
    /// called more than once; each file is validated like the dataset. The
    /// mix is the union of the files unless [`Self::replay_share`] is set.
    pub fn replay(mut self, path: impl Into<PathBuf>) -> Self {
        self.replay.push(path.into());
        self
    }

    /// The share of training draws that come from the [`Self::replay`] files
    /// in all (0 < share < 1), however large they are next to the dataset.
    /// Examples are drawn uniformly with replacement, so a replay set many
    /// times the dataset's size would otherwise take almost every step, and
    /// one much smaller would hardly be drawn; the dataset's records and the
    /// replay's are each listed as many times as puts the replay at this
    /// share ([`mix_repeats`]). Without it the mix is the plain union.
    pub fn replay_share(mut self, share: f32) -> Self {
        self.replay_share = Some(share);
        self
    }

    /// Mix another chat dataset into training at `share` of the draws
    /// (0 < share < 1) of its own, beside the dataset and the [`Self::replay`]
    /// files: records a run must keep drawing at a set rate whatever else is
    /// trained, such as a base model's own answers replayed so the new data
    /// does not move it off them. May be called more than once; the shares
    /// of every weighted file and [`Self::replay_share`] together must leave
    /// the dataset some of the draws.
    pub fn replay_at(mut self, path: impl Into<PathBuf>, share: f32) -> Self {
        self.weighted_replay.push((path.into(), share));
        self
    }

    /// Micro-batches summed into each optimizer step (default 1). Rows are
    /// single examples, so this is the effective batch size.
    pub fn grad_accum(mut self, micro_batches: u32) -> Self {
        self.grad_accum = micro_batches;
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

    /// The adapter method: `lora` (default), `ia3` (learned output-channel
    /// scales) or `lokr` (a Kronecker-product delta, applied without forming
    /// it). A continued adapter must be continued with its own kind; the LoRA
    /// variants (`rank_stabilized`, `lora_plus`, `freeze_a`) apply to LoRA only.
    pub fn adapter_kind(mut self, kind: model::adapter::DeviceKind) -> Self {
        self.tuning.kind = kind;
        self
    }

    /// rsLoRA: scale the update by `alpha / sqrt(rank)` instead of
    /// `alpha / rank`, so a higher rank is not suppressed. A continued adapter
    /// must be continued with the setting it trained with.
    pub fn rank_stabilized(mut self, on: bool) -> Self {
        self.tuning.rank_stabilized = on;
        self
    }

    /// LoRA+: train the adapter's `B` matrices at `ratio` times the learning
    /// rate (default 1, plain LoRA).
    pub fn lora_plus(mut self, ratio: f32) -> Self {
        self.tuning.lr_ratio = ratio;
        self
    }

    /// LoRA-FA: keep the `A` matrices at their initial values and train only
    /// `B`, which halves the adapter's gradient and optimizer state.
    pub fn freeze_a(mut self, on: bool) -> Self {
        self.tuning.freeze_a = on;
        self
    }

    /// AdamW's decoupled weight decay on the adapter's matrices (default 0,
    /// as LoRA recipes use: the adapter starts at zero and decay only pulls
    /// it back there).
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Steps a run whose [`Self::patience`] runs out spends bringing the
    /// rate down to its floor before it stops (default a tenth of the
    /// steps): the anneal a run stopped on a plateau would otherwise never
    /// have, so the adapter it keeps is not the one trained at a high rate.
    pub fn cooldown(mut self, steps: u32) -> Self {
        self.cooldown_steps = Some(steps);
        self
    }

    /// Optimizer steps (default 100). The first twentieth warms the rate up,
    /// and it decays over all of them.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Peak learning rate (default 3e-4); it decays to a tenth of this.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Seeds the fresh adapter's initialisation and the batch order. Unset, a
    /// run continuing a saved state takes that state's seed and any other
    /// draws a random one; [`ChatFineTuneOutcome::seed`] and the training
    /// record carry the seed used, and passing it reproduces the run.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

    /// Whether the model is trained to reason before it answers (the
    /// default: records are rendered as the base's template renders them). Off,
    /// the model is trained for no-think mode: each supervised answer follows
    /// the block the base's template leaves a no-think prompt in
    /// ([`data::chat_template::ChatTemplate::no_think_block`]) and keeps it
    /// through rendering, so training sees what a model asked not to reason is
    /// asked from - without it, a reasoning model trained on bare answers and
    /// asked with its think block closed (or open) has never seen that state.
    /// A base that never reasons is unaffected. Scoring before and after
    /// renders the same way.
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// Train a record that supervises several assistant turns as ONE sequence
    /// ([`ChatSample::answers_packed`]) instead of one example per turn
    /// ([`ChatSample::answers_as_asked`]). Each turn's target is the same
    /// either way; packing renders the conversation once, so a long
    /// tool-using trajectory costs its length once instead of once per turn.
    /// A step's gradient is the mean over the supervised tokens of the rows it
    /// accumulates either way, so a token weighs the same; what changes is
    /// that a row is a whole conversation, so a step of `n` rows spans far
    /// more tokens and an epoch has fewer rows. A record with one supervised
    /// turn is unaffected.
    pub fn pack_answers(mut self, on: bool) -> Self {
        self.pack_answers = on;
        self
    }

    /// Hold the frozen base at bf16, half the bytes of fp32 (a 7B decoder
    /// then trains on one 24 GB card); the adapters and optimiser stay fp32.
    /// Scoring before and after runs at the same tier.
    pub fn bf16_base(mut self, on: bool) -> Self {
        self.bf16_base = on;
        self
    }

    /// Hold the frozen base as int8 weights (a byte per weight and a scale per
    /// 32, about a quarter of fp32), which leaves the room of a 7B decoder's
    /// other half for activations: a longer context on one card. The weights
    /// are decoded as the matrix kernels read them and the activations stay
    /// fp32; the adapters and optimiser stay fp32. Takes precedence over
    /// [`Self::bf16_base`].
    pub fn int8_base(mut self, on: bool) -> Self {
        self.int8_base = on;
        self
    }

    /// Hold the frozen base as 4-bit NormalFloat blocks (QLoRA): the 27B
    /// `qwen35` base then trains in 44 GiB instead of 80. `qwen35` only; the
    /// adapters and optimiser stay fp32. Takes precedence over
    /// [`Self::bf16_base`].
    pub fn nf4_base(mut self, on: bool) -> Self {
        self.nf4_base = on;
        self
    }

    /// Hold the frozen base in the named format (`f32`, `bf16`, `int8`, `nf4`, `fp8`, ...): the one
    /// spelling for every architecture, refused naming the architecture and the formats its trainer
    /// holds. Replaces the `bf16_base`, `int8_base` and `nf4_base` switches, which it excludes.
    pub fn base_format(mut self, spec: impl Into<String>) -> Self {
        self.base_format = Some(spec.into());
        self
    }

    /// Hold the frozen base at fp32, the exact reference tier, whatever the device could run natively.
    /// Without any of the `*_base` choices the tier is picked from what the device has and what fits
    /// ([`choose_base_tier`]) and recorded on the outcome ([`ChatFineTuneOutcome::base_tier`]).
    pub fn f32_base(mut self, on: bool) -> Self {
        self.f32_base = on;
        self
    }

    /// Arithmetic of the frozen-base GEMMs. The default, [`GemmPrecision::Auto`], runs them on
    /// tensor cores where the card has them (bf16 or NF4 base: bf16 operands; fp32 base: TF32
    /// operands, a 10-bit mantissa, no extra memory), fp32 accumulation, and on the exact fp32
    /// kernels elsewhere; [`GemmPrecision::Fp32Reference`] pins the fp32 kernels. The adapters
    /// and their small products stay exact fp32 in every tier.
    pub fn gemm_precision(mut self, precision: crate::GemmPrecision) -> Self {
        self.gemm_precision = precision;
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
        // A named architecture is refused before anything is read.
        let named = self.arch.as_deref().map(crate::study::arch).transpose()?;
        if let Some(arch) = named {
            arch.require(Capability::ChatLoraFineTune)?;
        }
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no out directory; call .out_dir(path)".to_string()))?;
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let arch = match named {
            Some(arch) => arch,
            None => {
                let arch = crate::study::arch_of_checkpoint(&model_dir).map_or_else(|| crate::study::arch("qwen3"), Ok)?;
                arch.require(Capability::ChatLoraFineTune)?;
                arch
            }
        };
        let open = qwen3::store_checkpoint_path(&weights, &model_dir);
        let weights_str = utf8(&open)?;
        let family = ChatFamily::of(arch.id, self.bf16_base, self.int8_base, self.nf4_base, self.base_format.as_deref())?;

        // Every file is checked against the base's own template before a
        // device is claimed.
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: self.pack_answers };
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
        let held_out = self.held_out.as_deref().map(|path| asked.read(path)).transpose()?;
        let held_out_text = self.held_out_text.as_deref().map(|path| asked.read(path)).transpose()?;
        let monitor = self.monitor.as_deref().map(|path| asked.read(path)).transpose()?;
        if self.eval_every == 0 && self.evaluations_dir.is_some() {
            return Err(Error::Backend("ChatFineTune: keeping the evaluations needs eval_every > 0".to_string()));
        }
        if self.eval_every == 0 && (self.patience > 0 || self.keep_best) {
            return Err(Error::Backend("ChatFineTune: selecting on the monitoring loss needs eval_every > 0".to_string()));
        }

        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha, self.tuning)?;

        let Mix { rows: training, replay_records, repeats } = mix_rows(train_samples, replay_samples, self.replay_share, weighted_samples)?;
        // The packed validation split is what a periodic evaluation scores:
        // the monitoring set, else the held-out set, else (unread, when
        // nothing is evaluated) the training set.
        let val = monitor.as_deref().or(held_out.as_deref()).unwrap_or(&training[..]);
        let vocab = family.vocab(&open)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        let prepared = data::chat::prepare_chat_samples(&training, val, &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        // The held-out sets of a family that scores them in the training process (qwen35) are prepared
        // like the monitoring split, and the training row has to hold their longest example too.
        let mut longest = prepared.longest_example;
        let mut held_out_dirs: Vec<PathBuf> = Vec::new();
        if matches!(family, ChatFamily::Qwen35(_)) {
            for (name, records) in [("prepared-held-out", held_out.as_deref()), ("prepared-held-out-text", held_out_text.as_deref())] {
                if let Some(records) = records {
                    let dir = out_dir.join(name);
                    let set = data::chat::prepare_chat_samples(records, records, &tok, &tmpl, render, vocab, &dir).map_err(|e| Error::Backend(format!("preparing {name}: {e}")))?;
                    longest = longest.max(set.longest_example);
                    held_out_dirs.push(dir);
                }
            }
        }
        let block = block_for(longest, self.max_block)?;

        let set = crate::device::resolve(&self.device)?;
        let explicit = self.bf16_base || self.int8_base || self.nf4_base || self.f32_base || self.base_format.is_some();
        let (family, base_tier_reason) = if explicit {
            (family, "asked for".to_string())
        } else {
            let cards: Vec<u64> = gpu_core::devices::gpus().iter().filter(|d| set.gpus.contains(&d.index)).map(|d| d.identity.vram_bytes).collect();
            let native = !cards.is_empty() && gpu_core::Gpu::new(&[]).caps().arch.compute_capability.is_some_and(|cc| cc >= gpu_core::select::I8_MMA_MIN_CC);
            let (params, layers, width) = family.base_shape(&open)?;
            let (choice, reason) = choose_base_tier(matches!(family, ChatFamily::Qwen35(_)), native, params, layers, width, block, family.quantised_base_bytes(params), &cards).map_err(Error::Backend)?;
            (family.with_tier(choice), reason)
        };
        eprintln!("ChatFineTune: base tier {} ({base_tier_reason})", family.tier_name());
        let scoring_tier = family.scoring_tier();
        let base_score = held_out.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, None, &tok, &tmpl, records, block, tier, render));
        let base_text_score = held_out_text.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, None, &tok, &tmpl, records, block, tier, render));

        if self.grad_accum == 0 {
            return Err(Error::Backend("ChatFineTune: grad_accum must be at least 1".to_string()));
        }
        let state_path = out_dir.join(STATE_FILE);
        let seed = resolve_seed(self.seed, Some(&state_path))?;
        let opts = model::FitOpts {
            grad_accum: self.grad_accum,
            eval_interval: self.eval_every,
            // Every monitoring record, once, each time.
            eval_batches: 0,
            // Keeping the best without a patience is a patience the run
            // never reaches.
            patience: if self.keep_best && self.patience == 0 { u32::MAX } else { self.patience },
            weight_decay: self.weight_decay,
            gemm_precision: self.gemm_precision,
            cooldown_steps: self.cooldown_steps.unwrap_or(self.steps / COOLDOWN_DIVISOR),
            ..fit_opts(self.steps, block, self.lr, seed)
        };
        let base_digest = base_digest(&open)?;
        let identity = serde_json::json!({
            "base": base_digest,
            "dataset": digest(dataset)?,
            "replay": self.replay.iter().map(|p| digest(p)).collect::<Result<Vec<_>>>()?,
            "weighted_replay": self.weighted_replay.iter().map(|(p, share)| Ok((digest(p)?, share.to_bits()))).collect::<Result<Vec<_>>>()?,
            "monitor": self.monitor.as_deref().map(digest).transpose()?,
            "parent": parent,
            "rank": rank,
            "alpha_bits": alpha.to_bits(),
            "adapter_kind": self.tuning.kind.name(),
            "rank_stabilized": self.tuning.rank_stabilized,
            "lr_ratio_bits": self.tuning.lr_ratio.to_bits(),
            "freeze_a": self.tuning.freeze_a,
            "weight_decay_bits": self.weight_decay.to_bits(),
            "repeats": repeats,
            "grad_accum": self.grad_accum,
        });
        let mut step_work: Vec<model::StepWork> = Vec::new();
        let mut on_step = |s: &model::StepReport| {
            step_work.push(s.work);
            on_progress(&FineTuneProgress { step: s.step, steps: s.steps, loss: s.loss, lr: s.lr });
            !cancel.is_cancelled()
        };
        let snapshot_card = format!("{base_id}:local:chat:step");
        let snapshots = match &self.evaluations_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| Error::Backend(format!("{}: {e}", dir.display())))?;
                Some(model::AdapterSnapshots { dir, card_id: &snapshot_card, base_id: &base_id, family: arch.family, lora: model::adapter::device::LoraCfg::new(rank, alpha, Vec::new()) })
            }
            None => None,
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity, snapshots, grad_sync: None };
        let held_out_refs: Vec<&Path> = held_out_dirs.iter().map(PathBuf::as_path).collect();
        let (report, trained, scored) = family.train(&open, &prepared_dir, &opts, rank, alpha, self.tuning, self.continue_from.as_deref(), control, &held_out_refs).map_err(|e| Error::Backend(format!("training: {e}")))?;
        // In-process scores (qwen35): the sets come back in the order they were given.
        let mut in_process = scored.into_iter();
        let mut take_scored = |present: bool| if present { in_process.next() } else { None };
        let (scored_held_out, scored_text) = (take_scored(held_out.is_some() && !held_out_dirs.is_empty()), take_scored(held_out_text.is_some() && !held_out_dirs.is_empty()));
        let as_score = |loss: Option<f32>, set: &model::ScoredSet| HeldOutScore { loss, token_accuracy: None, positions: set.positions, records: set.records, skipped: 0 };
        let base_score = base_score.or(scored_held_out.as_ref().map(|set| as_score(set.base_loss, set)));
        let base_text_score = base_text_score.or(scored_text.as_ref().map(|set| as_score(set.base_loss, set)));

        let (selected_step, selection) = match report.kept_best {
            Some(best) => (best.step, Selection::BestMonitorLoss),
            None => (report.steps_completed, Selection::LastStep),
        };
        let mut outcome = ChatFineTuneOutcome {
            status: FineTuneStatus::Completed,
            adapter: None,
            adapter_digest: None,
            record: None,
            resume_state: None,
            seed,
            steps: self.steps,
            steps_completed: report.steps_completed,
            resumed_at: report.resumed_at,
            initial_loss: Some(report.initial_loss),
            final_loss: report.final_loss,
            train_records: dataset_records,
            replay_records,
            monitor_records: monitor.as_ref().map_or(0, Vec::len),
            eval_every: self.eval_every,
            patience: self.patience,
            curve: report.evaluations.iter().map(|e| MonitorPoint { step: e.step, train_loss: e.train_loss, monitor_loss: e.eval_loss }).collect(),
            selected_step,
            selection,
            stopped_early: report.stopped_early,
            block,
            rank,
            alpha,
            lr: self.lr,
            weight_decay: self.weight_decay,
            trained_from: parent.clone(),
            base_digest: Some(base_digest.clone()),
            base_score,
            tuned_score: None,
            base_text_score,
            tuned_text_score: None,
            base_tier: family.tier_name().to_string(),
            base_tier_reason,
            step_work,
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
            seed,
            hyperparams: serde_json::json!({
                "rank": rank,
                "alpha": alpha,
                "steps": self.steps,
                "steps_completed": outcome.steps_completed,
                "selected_step": outcome.selected_step,
                "selection": outcome.selection,
                "eval_every": self.eval_every,
                "patience": self.patience,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "block": block,
                "train_records": outcome.train_records,
                "replay_records": outcome.replay_records,
                "monitor_records": outcome.monitor_records,
            }),
            environment: gpu_core::backend_name().to_string(),
            gate: None,
            trained_from: parent,
            base_digest: Some(base_digest),
            cycle: self.cycle,
        };
        let dataset_id = digest(dataset)?;
        trained.save(utf8(&adapter_path)?, &adapter_id, &base_id, Some(&dataset_id), provenance)
            .map_err(|e| Error::Backend(format!("{}: saving the adapter: {e}", adapter_path.display())))?;
        drop(trained);
        outcome.adapter_digest = Some(digest(&adapter_path)?);
        let adapter_str = utf8(&adapter_path)?;
        outcome.tuned_score = match scored_held_out {
            Some(set) => Some(as_score(set.tuned_loss, &set)),
            None => held_out.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, Some(adapter_str), &tok, &tmpl, records, block, tier, render)),
        };
        outcome.tuned_text_score = match scored_text {
            Some(set) => Some(as_score(set.tuned_loss, &set)),
            None => held_out_text.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, Some(adapter_str), &tok, &tmpl, records, block, tier, render)),
        };
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
    /// The seed of the fresh adapter's initialisation and the batch order:
    /// the one asked for, the one a continued state was saved with, or the
    /// one drawn. Passing it to [`ChatFineTune::seed`] reproduces the run.
    pub seed: u64,
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
    /// Records in the monitoring set ([`ChatFineTune::monitor`]); 0 without one.
    pub monitor_records: usize,
    /// Steps between evaluations; 0 when nothing was evaluated during training.
    pub eval_every: u32,
    /// Evaluations without improvement the run was allowed; 0 when it was
    /// never to stop early.
    pub patience: u32,
    /// Every evaluation taken during training, in order; empty without
    /// [`ChatFineTune::eval_every`].
    pub curve: Vec<MonitorPoint>,
    /// The step whose adapter was exported: the best evaluation's, or the
    /// last step trained.
    pub selected_step: u32,
    /// Why that step.
    pub selection: Selection,
    /// True when the patience ran out before the step budget did.
    pub stopped_early: bool,
    /// The training row length, sized to the longest record.
    pub block: u32,
    pub rank: u32,
    pub alpha: f32,
    /// The peak learning rate of the run.
    pub lr: f32,
    /// The AdamW weight decay of the run.
    pub weight_decay: f32,
    /// The digest of the adapter this run continued, if it continued one.
    pub trained_from: Option<String>,
    /// `sha256:<hex>` of the base checkpoint file the adapter was trained on -
    /// the digest its card records as `TrainingProvenance::base_digest`, so a
    /// caller binding the adapter to its base need not hash the base again.
    /// `None` when the run did not compute it.
    pub base_digest: Option<String>,
    /// The base on the held-out records; `None` without [`ChatFineTune::held_out`].
    pub base_score: Option<HeldOutScore>,
    /// Base plus the new adapter on the same records; `None` unless
    /// completed with a held-out set.
    pub tuned_score: Option<HeldOutScore>,
    /// The base on the [`ChatFineTune::held_out_text`] records.
    pub base_text_score: Option<HeldOutScore>,
    /// Base plus the new adapter on the same records.
    pub tuned_text_score: Option<HeldOutScore>,
    /// The tier the frozen base was held at (`f32`, `bf16`, `int8` or `nf4`) and why: asked for, or the
    /// automatic choice ([`choose_base_tier`]) with its reason.
    pub base_tier: String,
    pub base_tier_reason: String,
    /// What every completed step of this call measured, in step order: the
    /// seconds its phases took and the tokens it accounted for
    /// ([`model::StepWork`]). A resumed run reports the steps it took, not
    /// those the state it resumed already trained. The training record carries
    /// the same, with the totals, under `step_work` and `work`.
    pub step_work: Vec<model::StepWork>,
}

impl ChatFineTuneOutcome {
    fn record_json(&self, base_id: &str, dataset_id: &str) -> serde_json::Value {
        let score = |s: &Option<HeldOutScore>| {
            s.as_ref().map(|s| serde_json::json!({ "loss": s.loss, "token_accuracy": s.token_accuracy, "positions": s.positions, "records": s.records, "skipped": s.skipped }))
        };
        serde_json::json!({
            "base": base_id,
            "base_digest": self.base_digest,
            "dataset": dataset_id,
            "adapter_digest": self.adapter_digest,
            "seed": self.seed,
            "trained_from": self.trained_from,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "train_records": self.train_records,
            "replay_records": self.replay_records,
            "monitor_records": self.monitor_records,
            "eval_every": self.eval_every,
            "patience": self.patience,
            "curve": self.curve.iter().map(|p| serde_json::json!({ "step": p.step, "train_loss": p.train_loss, "monitor_loss": p.monitor_loss })).collect::<Vec<_>>(),
            "selected_step": self.selected_step,
            "selection": self.selection,
            "stopped_early": self.stopped_early,
            "block": self.block,
            "rank": self.rank,
            "alpha": self.alpha,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "base_score": score(&self.base_score),
            "tuned_score": score(&self.tuned_score),
            "base_text_score": score(&self.base_text_score),
            "tuned_text_score": score(&self.tuned_text_score),
            "base_tier": self.base_tier,
            "base_tier_reason": self.base_tier_reason,
            "step_work": self.step_work_json(),
            "work": self.work_json(),
        })
    }

    /// The first step this call trained (a resumed call continues the count).
    fn first_step(&self) -> u32 {
        self.resumed_at.unwrap_or(0) + 1
    }

    /// One object per completed step of this call: `step`, the four phases and
    /// their `step_s`, and the token accounting. Additive to the record; what
    /// [`Self::step_work`] measured, in the same order.
    fn step_work_json(&self) -> Vec<serde_json::Value> {
        self.step_work
            .iter()
            .enumerate()
            .map(|(i, w)| {
                serde_json::json!({
                    "step": self.first_step() + i as u32,
                    "prepare_s": w.prepare_secs,
                    "forward_s": w.forward_secs,
                    "backward_s": w.backward_secs,
                    "optimizer_s": w.optimizer_secs,
                    "step_s": w.step_secs(),
                    "useful_tokens": w.useful_tokens,
                    "computed_tokens": w.computed_tokens,
                    "padded_tokens": w.padded_tokens(),
                    "supervised_positions": w.supervised_positions,
                    "device_peak_bytes": w.device_memory.map(|m| m.peak_live),
                    "device_reserved_bytes": w.device_memory.map(|m| m.reserved),
                })
            })
            .collect()
    }

    /// The totals of [`Self::step_work_json`] and the rates that follow from
    /// them: the numbers an optimization of the step is compared on.
    fn work_json(&self) -> serde_json::Value {
        let sum_secs = |k: fn(&model::StepWork) -> f64| self.step_work.iter().map(k).sum::<f64>();
        let (prepare, forward, backward, optimizer) = (sum_secs(|w| w.prepare_secs), sum_secs(|w| w.forward_secs), sum_secs(|w| w.backward_secs), sum_secs(|w| w.optimizer_secs));
        let useful: u64 = self.step_work.iter().map(|w| w.useful_tokens).sum();
        let computed: u64 = self.step_work.iter().map(|w| w.computed_tokens).sum();
        // Each step's padding is what it paid over its own useful tokens, so a
        // step that computed fewer than it held is not credited against one
        // that overran.
        let padded: u64 = self.step_work.iter().map(model::StepWork::padded_tokens).sum();
        let supervised: u64 = self.step_work.iter().map(|w| w.supervised_positions).sum();
        let step_secs = prepare + forward + backward + optimizer;
        serde_json::json!({
            "steps": self.step_work.len(),
            "prepare_s": prepare,
            "forward_s": forward,
            "backward_s": backward,
            "optimizer_s": optimizer,
            "step_s": step_secs,
            "useful_tokens": useful,
            "computed_tokens": computed,
            "padded_tokens": padded,
            "supervised_positions": supervised,
            "useful_tokens_per_s": if step_secs > 0.0 { useful as f64 / step_secs } else { 0.0 },
            "computed_tokens_per_s": if step_secs > 0.0 { computed as f64 / step_secs } else { 0.0 },
        })
    }
}

/// One evaluation of the monitoring set during training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MonitorPoint {
    /// Steps completed when it was taken.
    pub step: u32,
    /// The mean training loss of the steps since the previous evaluation.
    pub train_loss: f32,
    /// The mean per-token loss over the monitoring records.
    pub monitor_loss: f32,
}

/// Which step's adapter a fine-tune exported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Selection {
    /// The last step trained: the run did not select on the monitoring loss.
    LastStep,
    /// The evaluation with the lowest monitoring loss.
    BestMonitorLoss,
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
    let open = qwen3::store_checkpoint_path(&weights, &model_dir);
    let weights_str = utf8(&open)?;
    qwen3::checkpoint_config(weights_str).map_err(Error::Backend)?;
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let records = Asked { tok: &tok, tmpl: &tmpl, no_think: None, render: data::chat::RenderOpts::default(), pack: false }.read(held_out)?;
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
    Ok(score_records(weights_str, adapter, &tok, &tmpl, &records, block_for(longest, None)?, qwen3::Dtype::F32, data::chat::RenderOpts::default()))
}

/// Per-token log-probabilities of `targets` given `tokens` under `base`
/// (optionally with `adapter` applied at run time, exactly as serving it
/// would), with every logit multiplied by `logit_scale` first: `1/T` scores
/// under the distribution a sampler at temperature `T` drew from, `1.0` the
/// plain model. `targets[i]` is the token that follows `tokens[i]`; an
/// [`crate::IGNORE_TARGET`] target scores `0.0`. Pass no `adapter` to score under the
/// frozen reference of a preference or policy-gradient objective: the same
/// base, adapter disabled.
///
/// `base` is resolved like [`ChatFineTune::from_pretrained`]'s. The scores are
/// the ones a trainer's `batch_token_logprobs` reads off its own forward, from
/// the full `logits_all` pass, so this is the always-correct oracle to compare
/// a training run against, not the fast path inside one.
pub fn score_tokens(base: &str, adapter: Option<&Path>, tokens: &[u32], targets: &[u32], logit_scale: f32, device: &Device) -> Result<Vec<f32>> {
    if tokens.is_empty() || tokens.len() != targets.len() {
        return Err(Error::MissingArgument(format!("score_tokens: need as many targets as tokens (got {} tokens, {} targets) and at least one", tokens.len(), targets.len())));
    }
    if !(logit_scale.is_finite() && logit_scale > 0.0) {
        return Err(Error::MissingArgument(format!("score_tokens: the logit scale must be finite and positive, got {logit_scale}")));
    }
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let open = qwen3::store_checkpoint_path(&weights, &model_dir);
    let weights_str = utf8(&open)?;
    let cfg = qwen3::checkpoint_config(weights_str).map_err(Error::Backend)?;
    if let Some(&bad) = tokens.iter().chain(targets.iter().filter(|&&t| t != model::IGNORE)).find(|&&t| t >= cfg.vocab) {
        return Err(Error::Backend(format!("score_tokens: token {bad} is outside the model's vocabulary of {}", cfg.vocab)));
    }
    crate::device::resolve(device)?;
    let mut scored = qwen3::Qwen::load_inference(weights_str, 1, tokens.len() as u32);
    if let Some(path) = adapter {
        scored.attach_adapter(utf8(path)?).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    }
    Ok(model::logprobs::token_logprobs_scaled(&scored, tokens, targets, logit_scale).expect("a language model has logits"))
}

#[allow(clippy::too_many_arguments)]
fn score_records(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, records: &[ChatSample], block: u32, tier: qwen3::Dtype, render: data::chat::RenderOpts) -> HeldOutScore {
    let s = qwen3::eval::score_chat_rendered(weights, adapter, tok, tmpl, records, block, tier, render);
    HeldOutScore { loss: (s.positions > 0).then_some(s.loss), token_accuracy: s.token_accuracy, positions: s.positions, records: s.samples, skipped: s.skipped }
}

/// One record's teacher-forced readout.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordReadout {
    /// Index of the record in the file (0-based, input order).
    pub record: usize,
    /// Tokens the record encoded to.
    pub tokens: usize,
    /// Its supervised positions, ascending: each the model's log-probability of
    /// the recorded token and the token it would have picked itself.
    pub positions: Vec<TokenScore>,
}

/// A teacher-forced scoring pass over a record set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenReadout {
    /// One entry per record, in the file's order.
    pub records: Vec<RecordReadout>,
    /// Supervised positions scored in all.
    pub positions: usize,
    /// Mean log-probability over those positions: higher means the model finds
    /// the records more likely (the mean log-probability, not the loss).
    pub mean_logprob: Option<f32>,
}

/// Teacher-forced per-position readouts of the records in `records` under the
/// 27B (`qwen35`) trainer's OWN forward, with `adapter` loaded and NO optimizer
/// step - the in-process side of a train-serve agreement measurement.
///
/// `base` is resolved like [`ChatFineTune::from_pretrained`]'s; the tokenizer
/// and chat template come from its directory, so a chat record is rendered the
/// way this checkpoint does (a file of `{"ids", "mask"}` token records is scored
/// as written instead - the form to use when the training run rendered a record
/// differently from its JSON). The records are laid out and walked by the same
/// batching a training step uses, so a position scored here is a position
/// trained there.
///
/// The model is built as a fine-tune builds it - the frozen base at bf16 with
/// LoRA adapters - and `adapter`'s rank and scale are read from its own card;
/// `tuning` supplies what the card does not carry (the adapter method and
/// whether its scale is rsLoRA's). With no adapter the fresh zero-delta
/// initialisation leaves the base's logits unchanged, so the base arm scores
/// the base exactly. The scoring row is the longest record (at least 64
/// tokens), one record at a time.
pub fn score_tokens_qwen35(base: &str, adapter: Option<&Path>, records: &Path, tuning: model::adapter::device::AdapterTuning, device: &Device) -> Result<TokenReadout> {
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let open = qwen3::store_checkpoint_path(&weights, &model_dir);
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let read = qwen35::score::read_records(records, &tok, &tmpl).map_err(Error::Backend)?;
    crate::device::resolve(device)?;
    let longest = read.iter().map(|r| r.ids.len()).max().unwrap_or(0);
    let block = u32::try_from(longest).map_err(|_| Error::Backend(format!("the longest record is {longest} tokens, past the largest supported row")))?.max(MIN_BLOCK);
    let seed = resolve_seed(None, None)?;
    let opts = fit_opts(0, block, SCORE_LR, seed);
    let run = qwen35::finetune::StreamedRun { dtype: qwen3::Dtype::BF16, recompute: true, long: qwen35::finetune::LongSeqOverrides::default() };
    let scored = qwen35::finetune::score_records_streamed(&open, adapter, tuning, &read, &opts, &run).map_err(|e| Error::Backend(format!("scoring {}: {e}", records.display())))?;
    let positions = scored.positions;
    let sum: f64 = scored.records.iter().flatten().map(|s| s.logprob as f64).sum();
    Ok(TokenReadout {
        records: read
            .iter()
            .zip(scored.records)
            .enumerate()
            .map(|(record, (r, positions))| RecordReadout { record, tokens: r.ids.len(), positions })
            .collect(),
        positions,
        mean_logprob: (positions > 0).then(|| (sum / positions as f64) as f32),
    })
}

/// The held-out split of a chat dataset, written as the ids and masks the
/// trainer encodes, one JSONL line per record.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldOutRecords {
    /// The records file written (`{"ids":[...],"mask":[...]}` per line).
    pub out: PathBuf,
    /// The manifest beside it: the dataset's digest, the rule, and the counts.
    pub manifest: PathBuf,
    /// Raw dataset records the rule held out (before a record is split into one
    /// example per supervised answer).
    pub held_out: usize,
    /// Lines written: the examples the trainer would encode for them.
    pub emitted: usize,
    /// Tokens and supervised positions in all the emitted examples.
    pub tokens: usize,
    pub positions: usize,
}

/// The fractions a run's held-out split is made by: a tenth of the examinable
/// records, whole groups, grown to eight groups within a quarter - the rule
/// `split_dataset_file` applies before a run carves its monitoring set, and so
/// the split whose positions a run reports as its held-out score.
const HELD_OUT_SHARE: f64 = 0.1;
const HELD_OUT_MIN_GROUPS: usize = 8;
const HELD_OUT_MOST: f64 = 0.25;
/// The fewest records the rule splits at all (one to train on, one to hold out).
const SPLIT_MIN_RECORDS: usize = 2;

/// What one dataset record says about where it belongs (see [`held_out_flags`]).
struct Membership {
    /// `metadata.group`: the source text records share. `None` is its own.
    group: Option<String>,
    /// `metadata.task` or a non-empty `metadata.experiences`: it has a question
    /// a model can be examined on. One that has not follows its group.
    examinable: bool,
    /// `metadata.split`: a side the producer fixed, which no rule moves.
    side: Option<FixedSide>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FixedSide {
    Train,
    HeldOut,
}

impl Membership {
    fn of(record: &str) -> Membership {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(record) else {
            return Membership { group: None, examinable: true, side: None };
        };
        let metadata = &value["metadata"];
        Membership {
            group: metadata["group"].as_str().map(str::to_string),
            examinable: metadata["task"].is_string() || metadata["experiences"].as_array().is_some_and(|e| !e.is_empty()),
            side: match metadata["split"].as_str() {
                Some("train") => Some(FixedSide::Train),
                Some("held_out") => Some(FixedSide::HeldOut),
                _ => None,
            },
        }
    }
}

/// Which records of `records` a run holds out: `None` when there are too few
/// records to split.
///
/// The rule is the one a training run applies before it carves its monitoring
/// set: whole groups only (a group divided between training and scoring would
/// train on what it is later judged by), a tenth of the examinable records or
/// one whole group at least, grown to [`HELD_OUT_MIN_GROUPS`] groups while the
/// set stays within [`HELD_OUT_MOST`] of them - walked from the LAST record
/// backwards, so the split is a pure function of the file, with no seed and no
/// clock in it. A record whose producer fixed its side stays there, and a
/// record sharing its group follows it.
fn held_out_flags(records: &[String]) -> Option<Vec<bool>> {
    if records.len() < SPLIT_MIN_RECORDS {
        return None;
    }
    let memberships: Vec<Membership> = records.iter().map(|r| Membership::of(r)).collect();
    let mut fixed: std::collections::HashMap<&str, FixedSide> = std::collections::HashMap::new();
    for m in &memberships {
        if let (Some(group), Some(side)) = (m.group.as_deref(), m.side) {
            let entry = fixed.entry(group).or_insert(side);
            if side == FixedSide::Train {
                *entry = FixedSide::Train;
            }
        }
    }
    let followed = |m: &Membership| m.group.as_deref().and_then(|g| fixed.get(g)).copied();
    let free: Vec<usize> = (0..memberships.len()).filter(|&n| memberships[n].side.is_none() && followed(&memberships[n]).is_none()).collect();
    let mut held: Vec<bool> = memberships.iter().map(|m| m.side.or_else(|| followed(m)) == Some(FixedSide::HeldOut)).collect();
    if free.len() < SPLIT_MIN_RECORDS {
        return Some(held);
    }
    let loose: Vec<&Membership> = free.iter().map(|&n| &memberships[n]).collect();
    for (&n, chosen) in free.iter().zip(choose_held_out(&loose)) {
        held[n] = chosen;
    }
    Some(held)
}

/// Which of `memberships` (no side fixed) the held-out rule takes out.
fn choose_held_out(memberships: &[&Membership]) -> Vec<bool> {
    let keys: Vec<String> = memberships.iter().enumerate().map(|(n, m)| m.group.clone().unwrap_or_else(|| format!("\u{0}sample-{n}"))).collect();
    // The units and the counts are the examinable records'.
    let units: Vec<usize> = (0..memberships.len()).filter(|&n| memberships[n].examinable).collect();
    let mut sizes: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for &n in &units {
        *sizes.entry(keys[n].as_str()).or_default() += 1;
    }
    let of_units = |share: f64| (units.len() as f64 * share).floor() as usize;
    let wanted = of_units(HELD_OUT_SHARE).max(1);
    let half = units.len() / 2;
    let most = of_units(HELD_OUT_MOST);
    let mut held: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut count = 0;
    for &n in units.iter().rev() {
        if count >= wanted && held.len() >= HELD_OUT_MIN_GROUPS {
            break;
        }
        let key = keys[n].as_str();
        if held.contains(key) {
            continue;
        }
        let size = sizes[key];
        let for_the_tenth = count < wanted && count + size <= half;
        let for_the_groups = held.len() < HELD_OUT_MIN_GROUPS && count + size <= most;
        if for_the_tenth || for_the_groups {
            held.insert(key);
            count += size;
        }
    }
    keys.iter().map(|key| held.contains(key.as_str())).collect()
}

/// The held-out split of the chat dataset at `dataset`, written as the ids and
/// masks the trainer encodes (`{"ids":[...],"mask":[...]}` per line, in dataset
/// order) to `out`, with a manifest naming the dataset, the rule and the counts
/// beside it.
///
/// This is the set a run reports as its held-out score: the records the rule
/// holds out whole, read exactly as `ChatFineTune` reads them for that run
/// (`.thinking(thinking)` decides whether a supervised answer follows the
/// model's no-think block, and a record with several supervised answers - a tool
/// trajectory - becomes one example per answer, as training encodes it). The
/// ids and masks are therefore the ones the evidence was measured on, not a
/// re-rendering of the chat JSON, and the same file scores the served and the
/// in-process side of a train-serve comparison.
///
/// The split is deterministic: the rule reads the dataset's `metadata.group`,
/// `metadata.task`/`metadata.experiences` and `metadata.split` in file order,
/// with no seed and no clock, so the same dataset emits the same bytes.
pub fn emit_held_out_records(base: &str, dataset: &Path, out: &Path, thinking: bool, models_dir: Option<&Path>) -> Result<HeldOutRecords> {
    let store_root = loader::model_dir::resolve(models_dir.and_then(|d| d.to_str()));
    let (_weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let text = std::fs::read_to_string(dataset).map_err(|e| Error::Backend(format!("{}: {e}", dataset.display())))?;
    let raw: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect();
    let flags = held_out_flags(&raw).ok_or_else(|| Error::Backend(format!("{}: holds {} record(s); a split needs at least {SPLIT_MIN_RECORDS}", dataset.display(), raw.len())))?;
    let held: Vec<&String> = raw.iter().zip(&flags).filter(|(_, &h)| h).map(|(r, _)| r).collect();
    if held.is_empty() {
        return Err(Error::Backend(format!(
            "{}: no whole group could be held out ({} record(s), {} group(s)): dividing a group would train on what a score measures",
            dataset.display(),
            raw.len(),
            raw.iter().map(|r| Membership::of(r).group).collect::<std::collections::BTreeSet<_>>().len()
        )));
    }
    let held_groups: std::collections::BTreeSet<Option<String>> = held.iter().map(|r| Membership::of(r).group).collect();
    let kept_groups: std::collections::BTreeSet<Option<String>> = raw.iter().zip(&flags).filter(|(_, &h)| !h).map(|(r, _)| Membership::of(r).group).collect();
    // The trainer reads its records from a file; so does this, through the SAME
    // reader, so the rendering (per-answer examples, the no-think block) cannot
    // differ from the run's.
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let tmp = std::env::temp_dir().join(format!("brain-held-out-{}-{}.jsonl", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&tmp, held.iter().map(|r| r.as_str()).collect::<Vec<_>>().join("\n")).map_err(|e| Error::Backend(format!("{}: {e}", tmp.display())))?;
    let no_think = if thinking { None } else { tmpl.no_think_block() };
    let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
    let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: false };
    let samples = asked.read(&tmp);
    std::fs::remove_file(&tmp).ok();
    let samples = samples?;
    let (mut tokens, mut positions, mut lines) = (0usize, 0usize, String::new());
    for (at, sample) in samples.iter().enumerate() {
        let (ids, mask) = sample.encode_with(&tok, &tmpl, render).map_err(|e| Error::Backend(format!("{}: held-out example {} cannot be encoded: {e}", dataset.display(), at + 1)))?;
        let supervised = mask.iter().enumerate().skip(1).filter(|(_, &m)| m).count();
        if supervised == 0 {
            return Err(Error::Backend(format!("{}: held-out example {} encodes to {} token(s) with no supervised position, so it would score nothing", dataset.display(), at + 1, ids.len())));
        }
        tokens += ids.len();
        positions += supervised;
        let ids_json: Vec<String> = ids.iter().map(u32::to_string).collect();
        let mask_json: Vec<&str> = mask.iter().map(|&m| if m { "true" } else { "false" }).collect();
        lines.push_str(&format!("{{\"ids\":[{}],\"mask\":[{}]}}\n", ids_json.join(","), mask_json.join(",")));
    }
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| Error::Backend(format!("{}: {e}", dir.display())))?;
    }
    std::fs::write(out, &lines).map_err(|e| Error::Backend(format!("{}: {e}", out.display())))?;
    let manifest = out.with_extension("manifest.json");
    let record = serde_json::json!({
        "emitter": "brain::emit_held_out_records",
        "code_revision": format!("brain {}", env!("CARGO_PKG_VERSION")),
        "dataset": { "path": dataset.display().to_string(), "sha256": digest(dataset)?, "records": raw.len() },
        "rule": {
            "by": "metadata.group, whole groups only",
            "share": HELD_OUT_SHARE,
            "min_groups": HELD_OUT_MIN_GROUPS,
            "most": HELD_OUT_MOST,
            "min_records": SPLIT_MIN_RECORDS,
            "walk": "last record backwards, no seed",
        },
        "held_out": {
            "records": held.len(),
            "groups": held_groups.len(),
            "train_groups": kept_groups.len(),
            "group_names": held_groups.iter().map(|g| g.clone().unwrap_or_else(|| "(ungrouped)".to_string())).collect::<Vec<_>>(),
        },
        "emitted": { "records": samples.len(), "tokens": tokens, "positions": positions, "thinking": thinking, "out": out.display().to_string() },
    });
    std::fs::write(&manifest, serde_json::to_vec_pretty(&record).map_err(|e| Error::Backend(e.to_string()))?).map_err(|e| Error::Backend(format!("{}: {e}", manifest.display())))?;
    Ok(HeldOutRecords { out: out.to_path_buf(), manifest, held_out: held.len(), emitted: samples.len(), tokens, positions })
}

/// The LoRA rank and alpha of a run, and the digest of the adapter it
/// continues: a fresh adapter is rank 8 (or `rank`) at alpha `2 * rank`; a
/// continued one is its own rank - a different requested rank is refused -
/// and its own alpha. `alpha` overrides either.
pub(crate) fn lora_shape(continue_from: Option<&Path>, rank: Option<u32>, alpha: Option<f32>, tuning: model::adapter::device::AdapterTuning) -> Result<(u32, f32, Option<String>)> {
    tuning.validate().map_err(Error::Backend)?;
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
            // The card holds the scale the adapter trained at; under rsLoRA the
            // training alpha that reproduces it is that scale times sqrt(rank).
            let card_alpha = adapter.alpha.unwrap_or(own as f32);
            let trained_alpha = if tuning.rank_stabilized { card_alpha / (own as f32).sqrt() } else { card_alpha };
            (own, trained_alpha, Some(digest(path)?))
        }
    };
    Ok((rank, alpha.unwrap_or(default_alpha), parent))
}

/// Validate `path` against the base's tokenizer and template, then read it.
/// How a fine-tune's records are asked for: the base's tokenizer and template,
/// and for a model trained for no-think mode the block its prompt leaves open.
pub(crate) struct Asked<'a> {
    pub tok: &'a dyn data::tokenizer::Tokenizer,
    pub tmpl: &'a ChatTemplate,
    pub no_think: Option<&'a str>,
    pub render: data::chat::RenderOpts,
    /// A record of several supervised answers is one packed example.
    pub pack: bool,
}

impl Asked<'_> {
    /// The samples of `path` as they are trained on, each checked to encode
    /// against the template and to supervise something. A dialogue is one
    /// example per answer, each as the model is asked for it
    /// ([`ChatSample::answers_as_asked`]), for a model trained for no-think
    /// mode and, in thinking mode, for a record whose supervised turns call
    /// tools or that supervises only its last message after tool turns
    /// (whole-record masking refuses those); any other record keeps its shape,
    /// its answers following the no-think block when there is one.
    pub(crate) fn read(&self, path: &Path) -> Result<Vec<ChatSample>> {
        let summary = crate::validate_chat_dataset(path).map_err(Error::Backend)?;
        if summary.trained_messages == 0 {
            return Err(Error::Backend(format!("{}: holds no supervised turns (`\"train\": true`)", path.display())));
        }
        let records = ChatSample::from_jsonl(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
        let mut samples = Vec::with_capacity(records.len());
        for (at, record) in records.iter().enumerate() {
            let invalid = |e: &dyn std::fmt::Display| Error::Backend(format!("{}: record {} cannot be encoded for training: {e}", path.display(), at + 1));
            // Whole-record masking is exact for plain answers in thinking mode
            // and refuses tool turns; everywhere else each answer is asked for,
            // and a record supervising only its last message has one boundary.
            if self.pack && record.messages.iter().filter(|m| m.train).count() > 1 {
                if let Some(packed) = record.answers_packed(self.tmpl, self.no_think.is_none()).map_err(|e| invalid(&e))? {
                    samples.push(packed);
                    continue;
                }
            }
            let per_answer = if self.no_think.is_some() || record.supervises_tool_calls() {
                record.answers_as_asked(self.tmpl, self.no_think.is_none()).map_err(|e| invalid(&e))?
            } else if record.supervises_only_final_turn() && record.context_holds_tool_turns() {
                record.final_turn_as_asked(self.tmpl, true).map_err(|e| invalid(&e))?.map(|asked| vec![asked])
            } else {
                None
            };
            match (per_answer, self.no_think) {
                (Some(asked), _) => samples.extend(asked),
                (None, None) => samples.push(record.clone()),
                (None, Some(block)) => samples.push(record.answering_without_thinking(block)),
            }
        }
        for (at, sample) in samples.iter().enumerate() {
            let (ids, mask) = sample.encode_with(self.tok, self.tmpl, self.render).map_err(|e| Error::Backend(format!("{}: example {} cannot be encoded for training: {e}", path.display(), at + 1)))?;
            if !mask.iter().any(|m| *m) {
                return Err(Error::Backend(format!("{}: example {} encodes to {} token(s) with none supervised, so training on it would be a no-op", path.display(), at + 1, ids.len())));
            }
        }
        Ok(samples)
    }
}

pub(crate) fn tokenizer_and_template(model_dir: &Path) -> Result<(QwenBpe, ChatTemplate)> {
    let tmpl = ChatTemplate::from_model_dir(model_dir).map_err(|e| Error::Backend(format!("{}: {e}", model_dir.display())))?;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = QwenBpe::from_file(utf8(&tok_path)?).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;
    Ok((tok, tmpl))
}

/// The training row for a dataset whose longest record is `longest` tokens:
/// exactly that long (at least [`MIN_BLOCK`]), capped at `max`. A record
/// longer than the cap is refused, since training on a row that cannot hold
/// it would supervise nothing of it. The row is not rounded up: attention
/// and every activation scale with it, so padding a 5646-token record to the
/// next power of two trains 8192 rows' worth of work for 5646 tokens of data.
pub(crate) fn block_for(longest: usize, max: Option<u32>) -> Result<u32> {
    if let Some(max) = max {
        if longest > max as usize {
            return Err(Error::Backend(format!("the longest record is {longest} tokens, past max_block {max}; raise max_block or shorten the record")));
        }
    }
    let block = u32::try_from(longest).map_err(|_| Error::Backend(format!("the longest record is {longest} tokens, past the largest supported row")))?.max(MIN_BLOCK);
    Ok(max.map_or(block, |m| block.min(m)))
}

/// The rows a fine-tune trains on, and how they were mixed.
pub(crate) struct Mix {
    /// Every row, each group listed as many times as its share needs.
    pub rows: Vec<ChatSample>,
    /// Records of the replay files and the weighted files, each counted once.
    pub replay_records: usize,
    /// How many times each group was listed, the dataset first.
    pub repeats: Vec<usize>,
}

/// The groups of the mix, the dataset first: the replay files at their share
/// when one is set, else in the dataset's own group as the plain union they
/// always were; then each weighted file.
pub(crate) fn mix_rows(mut train_samples: Vec<ChatSample>, mut replay_samples: Vec<ChatSample>, replay_share: Option<f32>, weighted_samples: Vec<(Vec<ChatSample>, f32)>) -> Result<Mix> {
    let replay_records = replay_samples.len() + weighted_samples.iter().map(|(s, _)| s.len()).sum::<usize>();
    let mut groups: Vec<(Vec<ChatSample>, f32)> = Vec::new();
    match replay_share {
        Some(share) => groups.push((std::mem::take(&mut replay_samples), share)),
        None => train_samples.append(&mut replay_samples),
    }
    groups.extend(weighted_samples);
    let shares: Vec<(usize, f32)> = std::iter::once((train_samples.len(), 0.0)).chain(groups.iter().map(|(s, share)| (s.len(), *share))).collect();
    let repeats = mix_repeats(&shares)?;
    let mut rows = Vec::new();
    for (samples, &times) in std::iter::once(&train_samples).chain(groups.iter().map(|(s, _)| s)).zip(&repeats) {
        for _ in 0..times {
            rows.extend(samples.iter().cloned());
        }
    }
    Ok(Mix { rows, replay_records, repeats })
}

/// The most times the smallest group of a mix is listed in search of a closer
/// fit: repeats are whole numbers, and a ratio like 1.6 listed once or twice
/// is far from itself, but at a few times more it rounds within the tolerance.
const MIX_MAX_SCALE: usize = 16;

/// How far a group's realised share of the rows may lie from the share asked
/// before the mix is listed at a larger scale.
const MIX_TOLERANCE: f64 = 0.01;

/// How many times each group of records is listed so that, drawn uniformly
/// with replacement from all the rows, a group's records come up its share
/// of the time. `groups` is `(records, share)`, the dataset first: its share
/// is whatever the others leave. Each group's weight per record is its share
/// over its records; the repeats are those weights over the smallest, scaled
/// by the smallest whole factor up to [`MIX_MAX_SCALE`] that puts every
/// realised share within [`MIX_TOLERANCE`] of what was asked (the closest
/// scale when none does). An empty group is listed once, which lists nothing.
fn mix_repeats(groups: &[(usize, f32)]) -> Result<Vec<usize>> {
    let Some(((dataset, _), replays)) = groups.split_first() else {
        return Ok(Vec::new());
    };
    let mut total = 0.0f64;
    for &(_, share) in replays {
        if !(share > 0.0 && share < 1.0) {
            return Err(Error::Backend(format!("ChatFineTune: a replay share must be between 0 and 1, got {share}")));
        }
        total += f64::from(share);
    }
    if total >= 1.0 {
        return Err(Error::Backend(format!("ChatFineTune: the replay shares add up to {total}, leaving the dataset no draws")));
    }
    let shares: Vec<(usize, f64)> = std::iter::once((*dataset, 1.0 - total)).chain(replays.iter().map(|&(n, s)| (n, f64::from(s)))).collect();
    // Only a group with records is weighted; one without is listed once and adds nothing.
    let weights: Vec<Option<f64>> = shares.iter().map(|&(n, share)| (n > 0).then(|| share / n as f64)).collect();
    let Some(least) = weights.iter().flatten().copied().min_by(f64::total_cmp) else {
        return Ok(vec![1; groups.len()]);
    };
    let at_scale = |scale: usize| -> Vec<usize> { weights.iter().map(|w| w.map_or(1, |w| ((w / least) * scale as f64).round().max(1.0) as usize)).collect() };
    let deviation = |repeats: &[usize]| -> f64 {
        let rows: f64 = repeats.iter().zip(&shares).map(|(&r, &(n, _))| (r * n) as f64).sum();
        repeats.iter().zip(&shares).filter(|(_, &(n, _))| n > 0).map(|(&r, &(n, share))| ((r * n) as f64 / rows - share).abs()).fold(0.0, f64::max)
    };
    let mut best = (f64::INFINITY, Vec::new());
    for scale in 1..=MIX_MAX_SCALE {
        let repeats = at_scale(scale);
        let off = deviation(&repeats);
        if off <= MIX_TOLERANCE {
            return Ok(repeats);
        }
        if off < best.0 {
            best = (off, repeats);
        }
    }
    Ok(best.1)
}

/// AdamW weight decay of a LoRA fine-tune unless named: none. The adapter
/// starts at zero, so decay only pulls what it has learned back towards it.
pub(crate) const LORA_WEIGHT_DECAY: f32 = 0.0;

/// The warmup is this share of the steps, as the divisor: a twentieth.
const WARMUP_DIVISOR: u32 = 20;

/// A run's cooldown after its patience runs out, unless named, is this
/// share of its steps, as the divisor: a tenth.
const COOLDOWN_DIVISOR: u32 = 10;

/// The seed a run trains with: the one asked for, else the one the saved
/// state at `state` was written with (the state only resumes under the
/// options it was saved with, the seed among them), else a random draw.
pub(crate) fn resolve_seed(requested: Option<u64>, state: Option<&Path>) -> Result<u64> {
    if let Some(seed) = requested {
        return Ok(seed);
    }
    let saved = match state {
        Some(path) => model::train::saved_seed(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?,
        None => None,
    };
    Ok(saved.unwrap_or_else(crate::random_seed))
}

pub(crate) fn fit_opts(steps: u32, block: u32, lr: f32, seed: u64) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: block,
        lr,
        min_lr: lr / 10.0,
        warmup: (steps / WARMUP_DIVISOR).max(1),
        decay_iters: steps,
        weight_decay: LORA_WEIGHT_DECAY,
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
        cooldown_steps: 0,
        adam: Default::default(),
        gemm_precision: Default::default(),
        ..Default::default()
    }
}

/// The content digest naming a base ([`brain_modelstore::fetch::base_digest_sidecar`]:
/// the file's own, or a `transformers` directory's config and weights), each file hashed once and then
/// answered from its validated sidecar: a fine-tune names the same base every round, and hashing a
/// multi-gigabyte base is the cost of reading it all.
pub(crate) fn base_digest(path: &Path) -> Result<String> {
    brain_modelstore::fetch::base_digest_sidecar(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
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

    /// A continued rsLoRA adapter trains on at the alpha that reproduces the
    /// scale its card records, and a LoRA+ ratio that cannot train is refused.
    #[test]
    fn a_continued_rslora_adapter_keeps_the_scale_its_card_records() {
        use model::adapter::device::{alpha_matches, AdapterTuning, LoraCfg};
        let path = std::env::temp_dir().join(format!("brain-sdk-rs-card-{}.safetensors", std::process::id()));
        let trained = LoraCfg::new(4, 8.0, vec!["wq".into()]).with_tuning(AdapterTuning { rank_stabilized: true, ..Default::default() });
        let mut card = checkpoint::st::ModelCard::new("test:adapter", "qwen");
        card.adapter = Some(checkpoint::st::Adapter { kind: "lora".into(), rank: Some(4), base: None, alpha: Some(trained.card_alpha()), targets: Some(vec!["wq".into()]), dataset_id: None, per_target: None });
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("a.lora_a".to_string(), vec![4, 2], vec![0.0; 8])], &serde_json::json!({}), Some(&card)).unwrap();
        let tuning = AdapterTuning { rank_stabilized: true, ..Default::default() };
        let (rank, alpha, parent) = lora_shape(Some(&path), None, None, tuning).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((rank, parent.is_some()), (4, true));
        assert!(alpha_matches(alpha, 8.0), "the training alpha is {alpha}");
        assert!(lora_shape(None, None, None, AdapterTuning { lr_ratio: -1.0, ..Default::default() }).is_err());
    }

    /// A fine-tune warms up over a twentieth of its steps, decays the rate
    /// over all of them, and decays no weights unless asked.
    #[test]
    fn the_recipe_warms_up_briefly_and_decays_no_weights_by_default() {
        let o = fit_opts(400, 256, 2e-4, 1);
        assert_eq!((o.warmup, o.decay_iters, o.weight_decay), (20, 400, 0.0));
        assert_eq!(fit_opts(10, 256, 2e-4, 1).warmup, 1);
    }

    /// A row is exactly as long as the longest record (a short dataset gets
    /// the minimum row), never longer than the caller's cap, and a record
    /// past the cap is refused by size, never truncated. A power-of-two row
    /// would pad a 5646-token record to 8192 rows.
    #[test]
    fn the_training_row_is_the_longest_record_within_the_cap() {
        assert_eq!(block_for(10, None).unwrap(), MIN_BLOCK);
        assert_eq!(block_for(65, None).unwrap(), 65);
        assert_eq!(block_for(5646, None).unwrap(), 5646);
        assert_eq!(block_for(5646, Some(5646)).unwrap(), 5646);
        assert_eq!(block_for(70, Some(512)).unwrap(), 70);
        assert_eq!(block_for(10, Some(32)).unwrap(), 32, "the cap wins over the minimum row");
        let refused = block_for(5647, Some(5646)).unwrap_err().to_string();
        assert!(refused.contains("max_block") && refused.contains("5647"), "{refused}");
    }

    /// The samples `Asked` reads from a dataset of `lines`, in the mode asked.
    fn read_lines(lines: &[&str], no_think: bool) -> Result<Vec<ChatSample>> {
        read_lines_packing(lines, no_think, false)
    }

    fn read_lines_packing(lines: &[&str], no_think: bool, pack: bool) -> Result<Vec<ChatSample>> {
        let tmpl = ChatTemplate::compile(&std::fs::read_to_string(brain_testutil::testdata("qwen3/chat_template.jinja")).unwrap()).unwrap();
        let chars: String = (32u8..127).map(char::from).chain(['\n']).collect();
        let tok = data::tokenizer::CharTokenizer::from_corpus(&chars);
        // One file per call: tests run in parallel and each removes its own.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("brain-asked-{}-{call}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        let no_think_block = no_think.then(|| tmpl.no_think_block().unwrap());
        let render = data::chat::RenderOpts { keep_reasoning: no_think };
        let read = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think_block.as_deref(), render, pack }.read(&path);
        std::fs::remove_file(&path).unwrap();
        read
    }

    const ANSWER: &str = r#"{"messages":[{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true}]}"#;
    const DIALOGUE: &str = r#"{"messages":[{"role":"user","content":"q1","train":false},{"role":"assistant","content":"a1","train":true},{"role":"user","content":"q2","train":false},{"role":"assistant","content":"a2","train":true}]}"#;
    const TOOL_CONTEXT_FINAL_ANSWER: &str = r#"{"messages":[{"role":"user","content":"w","train":false},{"role":"assistant","content":"","train":false,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","content":"r","tool_call_id":"c1","train":false},{"role":"assistant","content":"done","train":true}]}"#;
    const TOOL_CONVERSATION: &str = r#"{"messages":[{"role":"user","content":"w","train":false},{"role":"assistant","content":"","train":true,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","content":"r","tool_call_id":"c1","train":false},{"role":"assistant","content":"done","train":true}]}"#;

    /// A conversation with a supervised tool call trains turn by turn in both
    /// thinking modes, while a dataset of plain answers keeps the shape it has
    /// always had in thinking mode: one example per record, encoded whole.
    #[test]
    fn tool_conversations_train_per_turn_and_plain_ones_as_before() {
        for no_think in [false, true] {
            let samples = read_lines(&[TOOL_CONVERSATION], no_think).unwrap();
            assert_eq!(samples.len(), 2, "no_think={no_think}");
            assert!(samples.iter().all(|s| s.rendered.is_some()));
        }
        let plain = read_lines(&[ANSWER], false).unwrap();
        assert_eq!(plain.len(), 1);
        assert!(plain[0].rendered.is_none(), "thinking-mode plain answers are encoded as the record");
        assert_eq!(read_lines(&[DIALOGUE], true).unwrap().len(), 2);
    }

    /// Packing turns a record of several supervised answers into one example
    /// in either thinking mode, and leaves one-answer records as they are.
    #[test]
    fn packing_makes_one_example_of_a_conversation_of_answers() {
        for no_think in [false, true] {
            assert_eq!(read_lines_packing(&[TOOL_CONVERSATION], no_think, true).unwrap().len(), 1, "no_think={no_think}");
            assert_eq!(read_lines_packing(&[DIALOGUE], no_think, true).unwrap().len(), 1, "no_think={no_think}");
            assert_eq!(read_lines_packing(&[TOOL_CONTEXT_FINAL_ANSWER], no_think, true).unwrap().len(), 1);
        }
        assert!(read_lines_packing(&[ANSWER], false, true).unwrap()[0].rendered.is_none());
    }

    /// A record that supervises only its final answer after a tool call and
    /// its result is one example in either thinking mode, not a refusal.
    #[test]
    fn a_final_answer_after_unsupervised_tool_turns_is_one_example() {
        for no_think in [false, true] {
            let samples = read_lines(&[TOOL_CONTEXT_FINAL_ANSWER], no_think).unwrap();
            assert_eq!(samples.len(), 1, "no_think={no_think}");
            assert!(samples[0].rendered.is_some());
        }
    }

    /// The share each group's rows make of the mix `repeats` lists.
    fn realised(groups: &[(usize, f32)], repeats: &[usize]) -> Vec<f64> {
        let rows: Vec<f64> = groups.iter().zip(repeats).map(|(&(n, _), &r)| (n * r) as f64).collect();
        let total: f64 = rows.iter().sum();
        rows.iter().map(|r| r / total).collect()
    }

    /// A replay set many times the dataset must not take every draw, and one
    /// much smaller must still be drawn its share: each group is listed as
    /// often as its share needs, to within a hundredth; several weighted
    /// groups each get their own share; no replay or no share leaves the
    /// plain union; and a share that leaves the dataset nothing is refused.
    #[test]
    fn a_replay_share_sets_how_often_each_group_is_listed() {
        // 100 target + 900 replay at a quarter: 27 x 100 = 2700 target rows beside 900.
        let large = [(100, 0.0), (900, 0.25)];
        let repeats = mix_repeats(&large).unwrap();
        assert_eq!(repeats, [27, 1]);
        // 100 target + 10 replay at a half: the replay is listed ten times.
        let scarce = [(100, 0.0), (10, 0.5)];
        assert_eq!(mix_repeats(&scarce).unwrap(), [1, 10]);
        // A ratio that rounds badly once is listed at a larger scale until it fits.
        let awkward = [(278, 0.0), (150, 0.25)];
        let repeats = mix_repeats(&awkward).unwrap();
        let shares = realised(&awkward, &repeats);
        assert!((shares[1] - 0.25).abs() <= MIX_TOLERANCE, "{repeats:?} -> {shares:?}");
        // Two weighted groups and the dataset each at their share.
        let three = [(278, 0.0), (60, 0.25), (93, 0.25)];
        let repeats = mix_repeats(&three).unwrap();
        let shares = realised(&three, &repeats);
        for (got, want) in shares.iter().zip([0.5, 0.25, 0.25]) {
            assert!((got - want).abs() <= MIX_TOLERANCE, "{repeats:?} -> {shares:?}");
        }
        // Nothing to weight is the plain union; an empty group lists nothing.
        assert_eq!(mix_repeats(&[(100, 0.0)]).unwrap(), [1]);
        assert_eq!(mix_repeats(&[(100, 0.0), (0, 0.25)]).unwrap(), [1, 1]);
        assert!(mix_repeats(&[]).unwrap().is_empty());
        for bad in [0.0, 1.0, -0.1, f32::NAN] {
            assert!(mix_repeats(&[(1, 0.0), (1, bad)]).unwrap_err().to_string().contains("replay share"));
        }
        assert!(mix_repeats(&[(1, 0.0), (1, 0.5), (1, 0.5)]).unwrap_err().to_string().contains("add up"));
    }
}

#[cfg(test)]
mod base_tier_tests {
    use super::ChatFamily;
    use qwen3::Dtype;

    fn qwen3_tier(bf16: bool, int8: bool) -> Dtype {
        match ChatFamily::of("qwen3", bf16, int8, false, None).unwrap() {
            ChatFamily::Qwen3(tier) => tier,
            ChatFamily::Qwen35(_) => unreachable!(),
        }
    }

    fn qwen35_tier(bf16: bool, nf4: bool) -> Dtype {
        match ChatFamily::of("qwen35", bf16, false, nf4, None).unwrap() {
            ChatFamily::Qwen35(tier) => tier,
            ChatFamily::Qwen3(_) => unreachable!(),
        }
    }

    #[test]
    fn the_narrowest_requested_base_wins() {
        assert_eq!(qwen3_tier(false, false), Dtype::F32);
        assert_eq!(qwen3_tier(true, false), Dtype::BF16);
        assert_eq!(qwen3_tier(true, true), Dtype::I8);
        assert_eq!(qwen3_tier(false, true), Dtype::I8);
        assert_eq!(qwen35_tier(false, false), Dtype::F32);
        assert_eq!(qwen35_tier(true, false), Dtype::BF16);
        assert_eq!(qwen35_tier(true, true), Dtype::NF4);
    }

    #[test]
    fn one_format_flag_names_the_base_of_every_architecture() {
        let tier = |arch, spec| match ChatFamily::of(arch, false, false, false, Some(spec)).unwrap() {
            ChatFamily::Qwen3(t) | ChatFamily::Qwen35(t) => t,
        };
        assert_eq!(tier("qwen3", "nf4"), Dtype::NF4);
        assert_eq!(tier("qwen3", "fp8"), Dtype::F8E4M3);
        assert_eq!(tier("qwen35", "fp8"), Dtype::F8E4M3);
        assert_eq!(tier("qwen35", "bf16"), Dtype::BF16);
    }

    #[test]
    fn a_tier_the_architecture_lacks_is_refused() {
        let err = ChatFamily::of("qwen35", false, true, false, None).err().expect("qwen35 holds no int8 base");
        assert!(format!("{err:?}").contains("qwen35") && format!("{err:?}").contains("f32, bf16, nf4, fp8"), "{err:?}");
        assert!(ChatFamily::of("qwen3", false, false, false, Some("fp3")).is_err());
        assert!(ChatFamily::of("qwen3", true, false, false, Some("nf4")).is_err(), "a format and a switch name the same setting");
    }
}

#[cfg(test)]
mod tier_tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    // The 27B: about 27e9 parameters, 64 layers, width 5120.
    const BIG: (u64, u64, u64) = (27_000_000_000, 64, 5120);
    /// NF4 at 27e9 parameters: half a byte plus an f32 scale per 32 weights.
    const NF4_27B: u64 = 27_000_000_000 / 2 + 4 * (27_000_000_000 / 32);

    fn pick(single: bool, native: bool, shape: (u64, u64, u64), block: u32, cards: &[u64]) -> BaseTierChoice {
        choose_base_tier(single, native, shape.0, shape.1, shape.2, block, NF4_27B, cards).unwrap().0
    }

    #[test]
    fn a_device_with_native_tensor_cores_trains_at_bf16_when_the_base_fits() {
        assert_eq!(pick(true, true, BIG, 16384, &[96 * GIB]), BaseTierChoice::Bf16, "the 27B at 16K tokens needs about 78 GiB of 96");
        assert_eq!(pick(true, true, BIG, 8192, &[80 * GIB]), BaseTierChoice::Bf16);
    }

    #[test]
    fn a_base_that_does_not_fit_at_bf16_falls_back_to_the_quantised_tier_and_says_why() {
        let (choice, why) = choose_base_tier(true, true, BIG.0, BIG.1, BIG.2, 16384, NF4_27B, &[64 * GIB]).unwrap();
        assert_eq!(choice, BaseTierChoice::Quantised);
        assert!(why.contains("quantised"), "{why}");
        // Two cards do not help a single-card trainer, but do help a pipelining one.
        assert_eq!(pick(true, true, BIG, 16384, &[48 * GIB, 48 * GIB]), BaseTierChoice::Quantised);
        assert_eq!(pick(false, true, BIG, 16384, &[48 * GIB, 48 * GIB]), BaseTierChoice::Bf16);
    }

    /// Even the quantised base and the row do not fit: refuse before anything loads, with the arithmetic.
    #[test]
    fn a_base_that_fits_at_no_tier_is_refused_with_the_planners_arithmetic() {
        let err = choose_base_tier(true, true, BIG.0, BIG.1, BIG.2, 16384, NF4_27B, &[16 * GIB]).unwrap_err();
        assert!(err.contains("refusing to load") && err.contains("Weights: quantised base") && err.contains("Activations"), "{err}");
    }

    #[test]
    fn a_device_without_native_tensor_cores_keeps_the_fp32_reference() {
        assert_eq!(pick(true, false, BIG, 1024, &[96 * GIB]), BaseTierChoice::F32);
        assert_eq!(pick(false, false, (600_000_000, 28, 1024), 1024, &[]), BaseTierChoice::F32, "a CPU run");
    }
}
