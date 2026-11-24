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

use serde::{Deserialize, Serialize};

pub mod model;
pub mod objective;
pub mod rows;
mod tables;
#[cfg(test)]
mod tests;

pub use model::{Example, ForwardStats, LrMultipliers, SftShape, SpeechSftModel};
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
    pub fn grad_clip() -> f32 {
        1.0
    }
    pub fn mtp_weight() -> f32 {
        0.3
    }
    pub fn streaming_fraction() -> f32 {
        0.5
    }
}

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
        for (name, v) in [("batch", self.batch), ("max_frames", self.max_frames), ("grad_accum", self.grad_accum)] {
            if v == 0 {
                return Err(format!("{name} must be at least 1"));
            }
        }
        self.lr_multipliers.validate()
    }
}

/// One optimizer step, as the run reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechStep {
    pub step: u32,
    /// `talker_ce + mtp_weight * code_predictor_ce`, averaged over the step's micro-batches.
    pub loss: f32,
    pub talker_ce: f32,
    pub code_predictor_ce: f32,
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
    pub loss: f32,
    pub talker_ce: f32,
    pub code_predictor_ce: f32,
    pub eos_accuracy: f32,
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
    /// Every trained tensor under its upstream name (`talker.model.layers.0...`,
    /// `talker.codec_head.weight`, `talker.code_predictor.*`,
    /// `talker.model.codec_embedding.weight`), f32, in its upstream shape.
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
        let talker = TalkerConfig::from_json(&root);
        let mtp = MtpConfig::from_json(&root);
        let specials = TtsSpecials::from_config_dir(dir.to_str().ok_or_else(|| at("path", &"not UTF-8"))?)?;
        let path = dir.join("model.safetensors");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().ok_or_else(|| at("path", &"not UTF-8"))?).map_err(|e| at("model.safetensors", &e))?;
        let fc1_bias = model::TEXT_PROJECTION[1];
        let text_inter = reader.shape(fc1_bias).and_then(|s| s.first().copied()).ok_or_else(|| at("model.safetensors", &format!("no {fc1_bias}")))? as u32;
        let shape = SftShape { talker: talker.clone(), mtp: mtp.clone(), specials: specials.clone(), batch: 1, block: 1, frames: 1, mtp_weight: 0.0, lr: LrMultipliers::default(), new_rows: Vec::new(), text_inter };
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, numel) in ::model::ModelConfig::param_list(&shape).into_iter().chain(shape.frozen_params()) {
            let values = reader.tensor(&name).ok_or_else(|| at("model.safetensors", &format!("no tensor {name}")))?;
            if values.len() != numel {
                return Err(at("model.safetensors", &format!("{name} holds {} values, the configuration needs {numel}", values.len())));
            }
            shapes.insert(name.clone(), reader.shape(&name).expect("read above").to_vec());
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
    if let Some(h) = held_out {
        if (&h.meta.languages, &h.meta.voices) != (&train.meta.languages, &train.meta.voices) {
            return Err("the held-out shard names other languages or voices than the training shard".to_string());
        }
    }
    let table = tensors.get_mut(model::TALKER_TABLE).expect("read by Base::open");
    let rows = rows::allocate(&cfg.rows, &specials, mtp.vocab, table, talker.d_model as usize, train)?;

    let shards: Vec<&Shard> = std::iter::once(train).chain(held_out).collect();
    let frames = longest(&shards, cfg.max_frames)?;
    let mut block = 0;
    for s in &shards {
        block = block.max(objective::rows_needed(&specials, &rows.map, s)?);
    }
    let shape = SftShape { talker, mtp, specials: specials.clone(), batch: cfg.batch, block: block as u32, frames: cfg.batch * frames as u32, mtp_weight: cfg.mtp_weight, lr: cfg.lr_multipliers, new_rows: rows.new_rows.clone(), text_inter };
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
    let objective = SpeechObjective::new(train, held_out, rows.map.clone(), specials, cfg.batch as usize, cfg.streaming_fraction, cfg.mtp_weight)?;
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
    let control = ::model::train::FitControl { on_step: Some(&mut on_step), state, state_every, identity, snapshots, publish: None, grad_sync };
    let (fit, model) = ::model::train::fit_controlled(model, objective, &opts, None, control).map_err(|e| e.to_string())?;

    let tensors = ::model::Model::param_names(&model)
        .into_iter()
        .map(|name| {
            let values = ::model::Model::read_weight(&model, &name);
            (name.clone(), shapes[&name].clone(), values)
        })
        .collect();
    Ok(SpeechSftResult { tensors, languages: rows.languages, voices: rows.voices, report: SpeechSftReport { fit, steps: history, seed } })
}

/// A step report in this objective's terms.
fn step_of(r: &::model::train::StepReport) -> SpeechStep {
    let metric = |name: &str| r.metrics.iter().find(|(n, _)| *n == name).map_or(f32::NAN, |(_, v)| *v);
    SpeechStep {
        step: r.step,
        loss: r.loss,
        talker_ce: metric("talker_ce"),
        code_predictor_ce: metric("code_predictor_ce"),
        eos_accuracy: metric("eos_accuracy"),
        grad_norm: r.grad_norm,
        lr: r.lr,
        frames_per_s: r.work.useful_tokens_per_s(),
        held_out: r.eval_loss.map(|loss| HeldOut { loss, talker_ce: metric("held_out_talker_ce"), code_predictor_ce: metric("held_out_code_predictor_ce"), eos_accuracy: metric("held_out_eos_accuracy") }),
    }
}
