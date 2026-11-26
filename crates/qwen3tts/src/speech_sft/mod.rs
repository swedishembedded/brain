// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Full-weight supervised fine-tuning of a Qwen3-TTS checkpoint (Talker and
//! code predictor) from recorded speech: the objective the reference trains
//! with, teacher forced, every gradient routed, over the shared fit loop.
//!
//! * [`speech_sft`]: the run - a base checkpoint, a training shard and an
//!   optional held-out one in; every trained tensor under its upstream name,
//!   the language and voice rows and the report out.
//! * [`model`]: the Talker and code predictor as one trainable model, and
//!   where each parameter lives.
//! * [`objective`]: the speech objective over a [`crate::shard::Shard`] for
//!   the shared fit loop (checkpoints, exact resume, step reports and
//!   cancellation come from [`::model::train::fit_controlled`]).
//! * [`rows`]: the rows of new languages and voices.
//!
//! The text embedding and the text projection are frozen in this release: a
//! row's gradient stops at the codec rows it summed. The speaker x-vector is a
//! frozen input.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// The arithmetic tier a build takes ([`SftShape::gemm`]).
pub use gpu_core::provider::library::GemmPrecision;
use serde::{Deserialize, Serialize};

pub mod calibrate;
pub mod distill;
pub mod model;
pub mod objective;
pub mod rows;
#[cfg(test)]
mod tables;
#[cfg(test)]
mod distill_tests;
#[cfg(test)]
mod tests;

pub use distill::{KdShape, KdStats, SpeechTeacher, TeacherMap, TextVocabMap};
pub use model::{DecoderStack, Example, ForwardStats, Logits, LossWeights, LrMultipliers, SftShape, SpeechSftModel};
pub use objective::{RowMap, SpeechObjective};
pub use rows::{RowSource, Rows, RowsConfig};

use crate::config::{MtpConfig, TalkerConfig};
use crate::prompt::TtsSpecials;
use crate::shard::Shard;

/// The learning-rate curve over the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Schedule {
    /// Warm up, then a cosine decay to a tenth of the rate by the last step.
    Cosine,
    /// Warm up, then the rate held.
    Constant,
}

/// The arithmetic of a run's products.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arithmetic {
    /// Tensor cores wherever the device has them (TF32 products of the fp32 weights, binary32 accumulation), the
    /// fp32 kernels elsewhere.
    #[default]
    TensorCores,
    /// The exact fp32 kernels everywhere: the reference the tensor-core runs are compared against.
    Fp32,
}

impl Arithmetic {
    fn gemm(self) -> GemmPrecision {
        match self {
            Arithmetic::TensorCores => GemmPrecision::Auto,
            Arithmetic::Fp32 => GemmPrecision::Fp32Reference,
        }
    }
}

/// One speech fine-tuning run. Unknown fields are refused.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechSftConfig {
    /// The base checkpoint: a Qwen3-TTS release directory (`config.json`,
    /// `model.safetensors`) on the machine that runs.
    pub base: String,
    /// Optimizer steps; exactly one of `steps` and `epochs`.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Passes over the training shard.
    #[serde(default)]
    pub epochs: Option<u32>,
    /// Peak learning rate (a starting point, not a tuned value).
    #[serde(default = "defaults::lr")]
    pub lr: f32,
    #[serde(default = "defaults::weight_decay")]
    pub weight_decay: f32,
    /// Steps of linear warmup.
    #[serde(default)]
    pub warmup: u32,
    #[serde(default = "defaults::schedule")]
    pub schedule: Schedule,
    /// Utterances per micro-batch.
    #[serde(default = "defaults::batch")]
    pub batch: u32,
    /// Micro-batches whose records are regrouped by length, so each micro-batch holds records of about one length
    /// and computes few padding rows (and the ranks of a data-parallel run get steps of about one cost). 1 draws
    /// records in the shuffled epoch order alone; the default, 16, is the window measured fastest on a card (a third
    /// more frames a second than unsorted batches of 1.5-30 s utterances). Every record still trains once an epoch;
    /// the steps then differ in their records' lengths rather than each mixing them.
    #[serde(default = "defaults::length_bucket_steps")]
    pub length_bucket_steps: u32,
    /// The most frames a record may hold (a longer one is refused, naming
    /// it): with the longest prompt it sizes the build.
    #[serde(default = "defaults::max_frames")]
    pub max_frames: u32,
    /// Micro-batches per optimizer step.
    #[serde(default = "defaults::one")]
    pub grad_accum: u32,
    /// The global gradient-norm clip, over every parameter group at once.
    #[serde(default = "defaults::grad_clip")]
    pub grad_clip: f32,
    /// The weight of the code predictor's loss (the reference's 0.3).
    #[serde(default = "defaults::mtp_weight")]
    pub mtp_weight: f32,
    /// The share of drawn records that train in the streaming text layout;
    /// the rest train non-streaming.
    #[serde(default = "defaults::streaming_fraction")]
    pub streaming_fraction: f32,
    #[serde(default)]
    pub lr_multipliers: LrMultipliers,
    /// The seed of the batch order and the layout draws; omitted, the saved
    /// run's when resuming, else a random one (reported back).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Which rows the shard's languages and voices take, and how new ones start.
    pub rows: RowsConfig,
    /// Score the held-out shard every this many steps (`0`: never).
    #[serde(default)]
    pub eval_interval: u32,
    /// Held-out batches per evaluation (`0`: every record).
    #[serde(default)]
    pub eval_batches: u32,
    /// Stop when the held-out loss has not improved for this many evaluations, and hand back the parameters of the
    /// best one (`0`: never stop early). Needs a held-out shard and an `eval_interval`.
    #[serde(default)]
    pub patience: u32,
    /// Hand back the parameters of the evaluation with the lowest held-out loss instead of the last step's, while
    /// still running every step (implied by a `patience`). Needs a held-out shard and an `eval_interval`.
    #[serde(default)]
    pub keep_best: bool,
    #[serde(default)]
    pub arithmetic: Arithmetic,
    /// The weight of the ground-truth objective (`CE_talker + mtp_weight * CE_code_predictor`) in the loss.
    #[serde(default = "defaults::unit")]
    pub gt_weight: f32,
    /// The frozen teacher of a distillation ([`distill`]): a Qwen3-TTS release directory of the same upstream layout
    /// on the machine that runs, held at bf16 beside the student on its card. Omitted, the run trains on the ground
    /// truth alone, and no `*_kd_*` setting may be named.
    #[serde(default)]
    pub teacher: Option<String>,
    /// The weight of the Talker's codebook-0 KL to the teacher (default 1).
    #[serde(default)]
    pub talker_kd_weight: Option<f32>,
    /// The weight of the code predictor's residual-codebook KLs to the teacher, summed over the codebooks (default
    /// 0.02: `mtp_weight`'s 0.3 spread over 15 codebooks).
    #[serde(default)]
    pub mtp_kd_weight: Option<f32>,
    /// The weight of the hidden-state matching (default 0: off); needs `hidden_kd_layers`.
    #[serde(default)]
    pub hidden_kd_weight: Option<f32>,
    /// `[student layer, teacher layer]` pairs whose residual streams are matched (the stream entering the layer; the
    /// layer count names the stream after the last layer).
    #[serde(default)]
    pub hidden_kd_layers: Vec<(u32, u32)>,
    /// The softmax temperature of the KLs (default 1); each KL is scaled by its square.
    #[serde(default)]
    pub kd_temperature: Option<f32>,
}

mod defaults {
    use super::Schedule;
    pub fn lr() -> f32 {
        2e-5
    }
    pub fn weight_decay() -> f32 {
        0.01
    }
    pub fn schedule() -> Schedule {
        Schedule::Cosine
    }
    pub fn batch() -> u32 {
        8
    }
    pub fn max_frames() -> u32 {
        400
    }
    pub fn one() -> u32 {
        1
    }
    pub fn length_bucket_steps() -> u32 {
        16
    }
    pub fn grad_clip() -> f32 {
        1.0
    }
    pub fn mtp_weight() -> f32 {
        0.3
    }
    pub fn streaming_fraction() -> f32 {
        0.5
    }
    pub fn unit() -> f32 {
        1.0
    }
}

/// `talker_kd_weight` when a teacher is named and the weight is not.
pub const DEFAULT_TALKER_KD_WEIGHT: f32 = 1.0;
/// `mtp_kd_weight` when a teacher is named and the weight is not: `mtp_weight`'s 0.3 spread over 15 codebooks.
pub const DEFAULT_MTP_KD_WEIGHT: f32 = 0.02;
/// `kd_temperature` when a teacher is named and the temperature is not.
pub const DEFAULT_KD_TEMPERATURE: f32 = 1.0;

impl SpeechSftConfig {
    /// Refuse a configuration no run can honour, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        match (self.steps, self.epochs) {
            (Some(0), _) | (_, Some(0)) => return Err("steps and epochs must be at least 1".into()),
            (Some(_), Some(_)) | (None, None) => return Err("name exactly one of steps and epochs".into()),
            _ => {}
        }
        for (name, v) in [("lr", self.lr), ("weight_decay", self.weight_decay), ("grad_clip", self.grad_clip), ("mtp_weight", self.mtp_weight)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("{name} must be finite and not negative, got {v}"));
            }
        }
        if !(0.0..=1.0).contains(&self.streaming_fraction) {
            return Err(format!("streaming_fraction must be in [0, 1], got {}", self.streaming_fraction));
        }
        for (name, v) in [("batch", self.batch), ("max_frames", self.max_frames), ("grad_accum", self.grad_accum), ("length_bucket_steps", self.length_bucket_steps)] {
            if v == 0 {
                return Err(format!("{name} must be at least 1"));
            }
        }
        if (self.patience > 0 || self.keep_best) && self.eval_interval == 0 {
            return Err("patience and keep_best select on the held-out loss: they need an eval_interval (and a held-out shard)".into());
        }
        if !(self.gt_weight.is_finite() && self.gt_weight >= 0.0) {
            return Err(format!("gt_weight must be finite and not negative, got {}", self.gt_weight));
        }
        match self.kd_shape(1) {
            None if self.talker_kd_weight.is_some() || self.mtp_kd_weight.is_some() || self.hidden_kd_weight.is_some() || self.kd_temperature.is_some() || !self.hidden_kd_layers.is_empty() => {
                return Err("the kd settings weigh a distillation: name a teacher".into());
            }
            None if self.gt_weight == 0.0 => return Err("gt_weight 0 without a teacher leaves nothing to learn".into()),
            Some(kd) if self.gt_weight == 0.0 && kd.talker == 0.0 && kd.mtp == 0.0 && kd.hidden == 0.0 => return Err("every weight of the loss is 0".into()),
            _ => {}
        }
        self.lr_multipliers.validate()
    }

    /// The distillation this run trains with, its teacher `teacher_d` wide; `None` without a teacher.
    pub fn kd_shape(&self, teacher_d: u32) -> Option<KdShape> {
        self.teacher.as_ref()?;
        Some(KdShape {
            talker: self.talker_kd_weight.unwrap_or(DEFAULT_TALKER_KD_WEIGHT),
            mtp: self.mtp_kd_weight.unwrap_or(DEFAULT_MTP_KD_WEIGHT),
            hidden: self.hidden_kd_weight.unwrap_or(0.0),
            temperature: self.kd_temperature.unwrap_or(DEFAULT_KD_TEMPERATURE),
            layers: self.hidden_kd_layers.clone(),
            teacher_d,
        })
    }

    /// The shared loop's patience for this run: the run's own, or one it never reaches when it keeps its best
    /// evaluation without stopping early ([`::model::train::FitOpts::patience`]).
    fn fit_patience(&self) -> u32 {
        match (self.patience, self.keep_best) {
            (0, true) => u32::MAX,
            (patience, _) => patience,
        }
    }
}

/// The distillation's measurements of a step or an evaluation ([`KdStats`]'s means).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distilled {
    /// The Talker's mean codebook-0 KL to the teacher.
    pub talker_kl: f32,
    /// The residual codebooks' KLs, each a mean over the frames, summed.
    pub code_predictor_kl: f32,
    /// The mapped pairs' hidden-state mean squared errors, summed.
    pub hidden_mse: f32,
}

/// One optimizer step, as the run reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechStep {
    pub step: u32,
    /// `gt_weight * (talker_ce + mtp_weight * code_predictor_ce)` plus the distillation's terms, averaged over the
    /// step's micro-batches.
    pub loss: f32,
    pub talker_ce: f32,
    pub code_predictor_ce: f32,
    /// What a distilling run measured against its teacher.
    pub distilled: Option<Distilled>,
    /// End-of-speech targets the Talker's greedy pick got right.
    pub eos_accuracy: f32,
    pub grad_norm: Option<f32>,
    pub lr: f32,
    /// Frames trained per second of step time.
    pub frames_per_s: f64,
    /// The held-out measurements taken after this step, when an evaluation was due.
    pub held_out: Option<HeldOut>,
}

/// Teacher-forced losses on the held-out shard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldOut {
    /// The selection metric: the loss, or the distillation's terms alone when the run distils.
    pub loss: f32,
    pub talker_ce: f32,
    pub code_predictor_ce: f32,
    pub eos_accuracy: f32,
    pub distilled: Option<Distilled>,
}

/// How the run went.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechSftReport {
    /// The shared loop's report (steps, resume, evaluations).
    pub fit: ::model::train::FitReport,
    /// Every step this call ran.
    pub steps: Vec<SpeechStep>,
    /// The seed the run trained with.
    pub seed: u64,
}

/// What a run produces.
#[derive(Clone, Debug)]
pub struct SpeechSftResult {
    /// Every trained tensor under its upstream name (`talker.model.layers.0...`, the run's own parameters such as a
    /// distillation's projections left out,
    /// `talker.codec_head.weight`, `talker.code_predictor.*`,
    /// `talker.model.codec_embedding.weight`), f32, in its upstream shape. Empty on a rank other than 0 of a
    /// data-parallel group: every rank trains the same parameters, and rank 0 hands them back.
    pub tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
    /// Every language the run trained, by name, and its codec row.
    pub languages: BTreeMap<String, u32>,
    /// Every named voice the run trained, by name, and its codec row.
    pub voices: BTreeMap<String, u32>,
    pub report: SpeechSftReport,
}

/// A base checkpoint read for training: its configuration and every tensor a
/// build reads, under upstream names.
pub struct Base {
    pub talker: TalkerConfig,
    pub mtp: MtpConfig,
    pub specials: TtsSpecials,
    /// The text projection's hidden width.
    pub text_inter: u32,
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<u64>>,
}

impl Base {
    /// Read the release at `dir`.
    pub fn open(dir: &Path) -> Result<Base, String> {
        let at = |what: &str, e: &dyn std::fmt::Display| format!("{}: {what}: {e}", dir.display());
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| at("config.json", &e))?;
        let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| at("config.json", &e))?;
        let path = dir.join("model.safetensors");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().ok_or_else(|| at("path", &"not UTF-8"))?).map_err(|e| at("model.safetensors", &e))?;
        let shape = |name: &str| reader.shape(name).map(<[u64]>::to_vec);
        let read = |name: &str| reader.tensor(name);
        Self::from_parts(&root, &shape, &read).map_err(|e| at("model.safetensors", &e))
    }

    /// A release whose `config.json` is `root` and whose tensors `shape` and
    /// `read` answer for by upstream name: what [`Self::open`] reads, for a
    /// checkpoint held in memory (an edited one).
    pub fn from_parts(root: &serde_json::Value, shape: &dyn Fn(&str) -> Option<Vec<u64>>, read: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Base, String> {
        let talker = TalkerConfig::from_json(root);
        let mtp = MtpConfig::from_json(root);
        let specials = TtsSpecials::from_config(root)?;
        let fc1_bias = model::TEXT_PROJECTION[1];
        let text_inter = shape(fc1_bias).and_then(|s| s.first().copied()).ok_or_else(|| format!("no {fc1_bias}"))? as u32;
        let shape_of = SftShape { talker: talker.clone(), mtp: mtp.clone(), specials: specials.clone(), batch: 1, block: 1, frames: 1, mtp_weight: 0.0, lr: LrMultipliers::default(), new_rows: Vec::new(), text_inter, gemm: GemmPrecision::Fp32Reference, gt_weight: 1.0, kd: None };
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, numel) in ::model::ModelConfig::param_list(&shape_of).into_iter().chain(shape_of.frozen_params()) {
            let values = read(&name).ok_or_else(|| format!("no tensor {name}"))?;
            if values.len() != numel {
                return Err(format!("{name} holds {} values, the configuration needs {numel}", values.len()));
            }
            shapes.insert(name.clone(), shape(&name).expect("read above"));
            tensors.insert(name, values);
        }
        Ok(Base { talker, mtp, specials, text_inter, tensors, shapes })
    }
}

/// The most frames any record of `shards` holds, refused past `max_frames`.
fn longest(shards: &[&Shard], max_frames: u32) -> Result<usize, String> {
    let mut most = 0;
    for shard in shards {
        for r in 0..shard.len() {
            let frames = shard.frames(r);
            if frames > max_frames as usize {
                return Err(format!("record {r} ({}) holds {frames} frames, past max_frames {max_frames}", shard.record(r).id.id));
            }
            most = most.max(frames);
        }
    }
    Ok(most)
}

/// Fine-tune the checkpoint `cfg.base` on `train`, scoring `held_out` every
/// `cfg.eval_interval` steps.
///
/// `control` is the shared loop's: its `on_step` sees every step and can stop
/// the run at a step boundary, its `state` makes the run resumable exactly
/// (this configuration joins its `identity`, so a state is never continued
/// under another one).
pub fn speech_sft(cfg: &SpeechSftConfig, train: &Shard, held_out: Option<&Shard>, control: ::model::train::FitControl<'_>) -> Result<SpeechSftResult, String> {
    cfg.validate()?;
    let Base { talker, mtp, specials, text_inter, mut tensors, shapes } = Base::open(Path::new(&cfg.base))?;
    let teacher = cfg.teacher.as_ref().map(|dir| Base::open(Path::new(dir)).map_err(|e| format!("teacher: {e}"))).transpose()?;
    if let Some(h) = held_out {
        if (&h.meta.languages, &h.meta.voices) != (&train.meta.languages, &train.meta.voices) {
            return Err("the held-out shard names other languages or voices than the training shard".to_string());
        }
    }
    let table = tensors.get_mut(model::TALKER_TABLE).expect("read by Base::open");
    let rows = rows::allocate(&cfg.rows, &specials, mtp.vocab, table, talker.d_model as usize, train)?;

    let shards: Vec<&Shard> = std::iter::once(train).chain(held_out).collect();
    let mut rows = rows;
    let writers: Vec<&Path> = cfg.teacher.iter().map(Path::new).collect();
    rows.map.text = shard_text_map(Path::new(&cfg.base), talker.text_vocab_size, &shards, &writers)?;
    rows.map.xvector = crate::speaker_projection::SpeakerProjection::read(Path::new(&cfg.base))?.map(std::sync::Arc::new);
    if let Some(dir) = &cfg.teacher {
        if crate::speaker_projection::SpeakerProjection::read(Path::new(dir))?.is_some() {
            return Err("teacher: a teacher reads the shards' own x-vectors, and this one is a cut that projects them".into());
        }
    }
    let frames = longest(&shards, cfg.max_frames)?;
    let mut block = 0;
    for s in &shards {
        block = block.max(objective::rows_needed(&specials, &rows.map, s)?);
    }
    let kd = teacher.as_ref().and_then(|t| cfg.kd_shape(t.talker.d_model));
    let shape = SftShape { talker, mtp, specials: specials.clone(), batch: cfg.batch, block: block as u32, frames: cfg.batch * frames as u32, mtp_weight: cfg.mtp_weight, lr: cfg.lr_multipliers, new_rows: rows.new_rows.clone(), text_inter, gemm: cfg.arithmetic.gemm(), gt_weight: cfg.gt_weight, kd };
    // The teacher is checked against the student and the shards before anything is built.
    let teaching = match (&teacher, &cfg.teacher) {
        (Some(t), Some(dir)) => Some(check_teacher(cfg, &shape, t, Path::new(dir), &rows, &shards)?),
        _ => None,
    };
    let seed = match (cfg.seed, control.state) {
        (Some(s), _) => s,
        (None, Some(state)) => ::model::train::saved_seed(state).map_err(|e| e.to_string())?.unwrap_or_else(data::rng::random_seed),
        (None, None) => data::rng::random_seed(),
    };
    let steps = match (cfg.steps, cfg.epochs) {
        (Some(s), _) => s,
        (None, Some(e)) => (u64::from(e) * train.len() as u64).div_ceil(u64::from(cfg.batch * cfg.grad_accum)).max(1) as u32,
        (None, None) => unreachable!("validated: exactly one of steps and epochs"),
    };
    let model = SpeechSftModel::build(shape, tensors, seed)?;
    let mut objective = SpeechObjective::new(train, held_out, rows.map.clone(), specials, cfg.batch as usize, cfg.streaming_fraction, cfg.mtp_weight)?.with_length_buckets(cfg.length_bucket_steps as usize).with_weights(model.loss_weights());
    if let (Some(map), Some(t)) = (teaching, teacher) {
        let s = model.shape();
        let tshape = SftShape { talker: t.talker, mtp: t.mtp, specials: t.specials, batch: s.batch, block: s.block, frames: s.frames, mtp_weight: 1.0, lr: LrMultipliers::default(), new_rows: Vec::new(), text_inter: t.text_inter, gemm: s.gemm, gt_weight: 1.0, kd: None };
        objective = objective.with_teacher(SpeechTeacher::new(&model, tshape, t.tensors, map)?);
    }
    let (min_lr, decay_iters) = match cfg.schedule {
        Schedule::Cosine => (cfg.lr / 10.0, steps),
        Schedule::Constant => (cfg.lr, 0),
    };
    let opts = ::model::train::FitOpts {
        steps,
        // The loop positions a resumed objective by records drawn: its batch is the objective's.
        batch_size: cfg.batch,
        block_size: block as u32,
        lr: cfg.lr,
        min_lr,
        warmup: cfg.warmup,
        decay_iters,
        weight_decay: cfg.weight_decay,
        grad_clip: cfg.grad_clip,
        grad_accum: cfg.grad_accum,
        eval_interval: if held_out.is_some() { cfg.eval_interval } else { 0 },
        eval_batches: cfg.eval_batches,
        seed,
        checkpoint_secs: 0,
        initial_probe_batches: 0,
        patience: cfg.fit_patience(),
        // The best evaluation's parameters are held in host memory (and in the resume state, under their `best:`
        // names): the run has no checkpoint path of its own to write them to at every improvement.
        best_hold_floats: usize::MAX,
        ..::model::train::FitOpts::default()
    };

    let mut history = Vec::new();
    let ::model::train::FitControl { on_step: mut caller, state, state_every, identity, snapshots, publish, grad_sync } = control;
    // A speech fine-tune trains every weight of the talker: it has no adapter to publish while it runs.
    if publish.is_some() {
        return Err("speech_sft: a full fine-tune of the talker publishes no adapter during the run".to_string());
    }
    let mut on_step = |r: &::model::train::StepReport| {
        history.push(step_of(r));
        caller.as_mut().is_none_or(|f| f(r))
    };
    let mut fingerprint = cfg.clone();
    fingerprint.seed = None;
    let identity = serde_json::json!({ "caller": identity, "speech_sft": serde_json::to_string(&fingerprint).map_err(|e| e.to_string())? });
    // Every rank of a data-parallel group ends with the same parameters; rank 0's are the run's.
    let writer = grad_sync.as_ref().is_none_or(|sync| sync.rank == 0);
    let control = ::model::train::FitControl { on_step: Some(&mut on_step), state, state_every, identity, snapshots, publish: None, grad_sync };
    let (fit, model) = ::model::train::fit_controlled(model, objective, &opts, None, control).map_err(|e| e.to_string())?;

    let tensors = ::model::Model::param_names(&model)
        .into_iter()
        .filter(|name| writer && !name.starts_with(distill::PRIVATE_PREFIX))
        .map(|name| {
            let values = ::model::Model::read_weight(&model, &name);
            (name.clone(), shapes[&name].clone(), values)
        })
        .collect();
    Ok(SpeechSftResult { tensors, languages: rows.languages, voices: rows.voices, report: SpeechSftReport { fit, steps: history, seed } })
}

/// How the text ids of `shards` read for the checkpoint at `target` (`target_vocab` text rows): unchanged when its own
/// tokenizer wrote them (or the shards name none), through the tokens of the checkpoint among `writers` whose tokenizer
/// wrote them (a teacher, or the full model a cut came from; a trimmed vocabulary renumbers its tokens), and refused
/// otherwise: their ids would silently name other tokens. The map is [`RowMap::text`]'s.
pub fn shard_text_map(target: &Path, target_vocab: u32, shards: &[&Shard], writers: &[&Path]) -> Result<Option<Vec<u32>>, String> {
    let digests: std::collections::BTreeSet<&str> = shards.iter().filter_map(|s| s.meta.producers.get("tokenizer").map(String::as_str)).collect();
    let Some(&shard_tokenizer) = digests.iter().next() else { return Ok(None) };
    if digests.len() > 1 {
        return Err(format!("the shards were tokenised by different tokenizers ({digests:?})"));
    }
    let own = crate::encode::tokenizer_digest(target)?;
    if own == shard_tokenizer {
        return Ok(None);
    }
    for dir in writers {
        if crate::encode::tokenizer_digest(dir)? == shard_tokenizer {
            let theirs = text_vocab_size(dir)?;
            let map = TextVocabMap::from_dirs(target, dir, target_vocab, theirs).map_err(|e| format!("{}: {e}", dir.display()))?;
            return Ok(Some(map.to_student(theirs)));
        }
    }
    Err(format!(
        "the shards' text ids are of another tokenizer ({shard_tokenizer}) than {}'s ({own}); tokenise them with its tokenizer, or name the checkpoint whose tokenizer wrote them",
        target.display()
    ))
}

/// A release's text vocabulary size (`talker_config.text_vocab_size` of its `config.json`).
fn text_vocab_size(dir: &Path) -> Result<u32, String> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let config: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}/config.json: {e}", dir.display()))?;
    config["talker_config"]["text_vocab_size"].as_u64().map(|v| v as u32).ok_or_else(|| format!("{}/config.json has no talker_config.text_vocab_size", dir.display()))
}

/// The teacher of `cfg` (its release `t` at `dir`) checked against the student `shape`, the shards' languages and voices
/// (the teacher must have each, at whatever row) and every text token of the shards: how the teacher reads the
/// student's batches.
fn check_teacher(cfg: &SpeechSftConfig, shape: &SftShape, t: &Base, dir: &Path, rows: &Rows, shards: &[&Shard]) -> Result<TeacherMap, String> {
    let text = TextVocabMap::from_dirs(Path::new(&cfg.base), dir, shape.talker.text_vocab_size, t.talker.text_vocab_size).map_err(|e| format!("teacher: {e}"))?;
    text.check_specials(&shape.specials, &t.specials).map_err(|e| format!("teacher: {e}"))?;
    let meta = &shards[0].meta;
    let mut codec_rows = HashMap::new();
    for (i, name) in meta.languages.iter().enumerate() {
        if let Some(row) = rows.map.languages[i] {
            let theirs = *t.specials.lang.get(name).ok_or_else(|| format!("teacher: the teacher has no language {name:?} to teach"))?;
            codec_rows.insert(row, theirs);
        }
    }
    for (i, name) in meta.voices.iter().enumerate() {
        let theirs = *t.specials.spk_id.get(name).ok_or_else(|| format!("teacher: the teacher has no voice {name:?} to teach"))?;
        codec_rows.insert(rows.map.voices[i], theirs);
    }
    for shard in shards {
        for r in 0..shard.len() {
            let rec = shard.record(r);
            // The teacher reads the x-vector the shard stores in the speaker slot (the student its projection of it):
            // only a teacher of the shards' x-vector width can.
            if rec.voice.is_none() && t.talker.d_model != shard.meta.xvector_dim {
                return Err(format!(
                    "teacher: record {r} ({}) speaks in the x-vector slot, and the teacher ({} wide) cannot read the shard's {}-wide x-vector",
                    rec.id.id, t.talker.d_model, shard.meta.xvector_dim
                ));
            }
            if let Some(Err(e)) = rec.text_ids.iter().map(|&id| rows.map.text_id(id).and_then(|s| text.get(s))).find(Result::is_err) {
                return Err(format!("teacher: record {r} ({}): {e}", rec.id.id));
            }
        }
    }
    Ok(TeacherMap { text, codec_rows })
}

/// A step report in this objective's terms.
fn step_of(r: &::model::train::StepReport) -> SpeechStep {
    let metric = |name: &str| r.metrics.iter().find(|(n, _)| *n == name).map_or(f32::NAN, |(_, v)| *v);
    let distilled = |prefix: &str| {
        let m = |name: &str| r.metrics.iter().find(|(n, _)| *n == format!("{prefix}{name}")).map(|(_, v)| *v);
        Some(Distilled { talker_kl: m("talker_kl")?, code_predictor_kl: m("code_predictor_kl")?, hidden_mse: m("hidden_mse")? })
    };
    SpeechStep {
        step: r.step,
        loss: r.loss,
        talker_ce: metric("talker_ce"),
        code_predictor_ce: metric("code_predictor_ce"),
        distilled: distilled(""),
        eos_accuracy: metric("eos_accuracy"),
        grad_norm: r.grad_norm,
        lr: r.lr,
        frames_per_s: r.work.useful_tokens_per_s(),
        held_out: r.eval_loss.map(|loss| HeldOut { loss, talker_ce: metric("held_out_talker_ce"), code_predictor_ce: metric("held_out_code_predictor_ce"), eos_accuracy: metric("held_out_eos_accuracy"), distilled: distilled("held_out_") }),
    }
}
