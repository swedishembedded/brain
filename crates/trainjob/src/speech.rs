// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that teach speech
// synthesizers new languages and named voices from recorded speech, on the
// accelerators that serve them, for its clients. If your team needs expertise
// in text-to-speech fine-tuning as a service, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The `speech_sft` action: a Qwen3-TTS checkpoint fine-tuned on recorded speech, with new languages and named voices,
//! published in the upstream layout of its base.
//!
//! The datasets are training shards (`brain-tts-shard-v1`, [`qwen3tts::shard`]): the call's blobs `dataset` and
//! `held_out`, or digests of the artifact store. The run trains through the speech objective
//! ([`qwen3tts::speech_sft`]) under the `resume_key` contract the other jobs share ([`crate::runs`]), and the trained
//! tensors are exported over the base ([`qwen3tts::export`]): a checkpoint in the base's own names, dtypes and files,
//! with the new languages and voices in its `config.json`, that brain and the upstream package load as they are. It
//! leaves like every checkpoint of this service, as a manifest of its files by digest (`checkpoint_digest`), and with
//! `install_as` it is installed in the model store as `local/<install_as>`, where the server serves it as a model of
//! its own, under that id, with its voices as preset voices and its languages.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::{json, Value};

use qwen3tts::speech_sft::SpeechSftConfig;

use crate::consolidate::{check_install_name, check_resume, checkpoint_every_param, checkpoint_outcome_of, open_run, resume_key_param};
use crate::job::{self, bring, JobEnv, Workdir};
use crate::request::Source;
use crate::store::Digest;

/// The action's name.
pub const ACTION: &str = "speech_sft";
/// The architecture a base must be.
const ARCHITECTURE: &str = "Qwen3TTSForConditionalGeneration";
/// The largest `languages` or `voices` document accepted.
pub const MAX_NAMES_DOC: usize = 64 << 10;
/// The most languages or voices one run may name.
pub const MAX_NAMES: usize = 256;
/// The id a checkpoint carries in its manifest when it is not installed under a name.
const DEFAULT_ID: &str = "local/qwen3tts-speech";
/// The working set a run is charged on its card beside four copies of the trained weights (weights, gradients and the
/// two AdamW moments) and a micro-batch's activations: the kernels' workspaces and the step's fixed buffers.
const WORKSPACE_BYTES: u64 = 4 << 30;
/// Activation floats a codec frame of a micro-batch holds, per float of one layer's hidden and feed-forward widths
/// summed over the Talker's layers and the code predictor's (which runs every frame's codebooks as positions): fitted
/// to the device peaks of the 1.7B on a GH200 (62.9 GB at 16 utterances, 91.2 GB at 32, the longest of 375 frames).
const ACTIVATION_FLOATS_PER_WIDTH: f64 = 2.12;
/// The most utterances one micro-batch may hold.
const MAX_BATCH: f64 = 1_024.0;
/// The most frames a record may hold (about 22 minutes at 12.5 frames per second).
const MAX_FRAMES: f64 = 16_384.0;
/// The largest learning-rate multiplier of a parameter group.
const MAX_LR_MULTIPLIER: f64 = 1_000.0;

/// How the request names a language of the shards: one of the base's, or a new one initialised from a mix of the
/// base's.
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageInit {
    pub new: bool,
    pub init_from: Vec<String>,
}

/// The objective's own defaults: what a setting the request leaves out trains with.
fn objective_defaults() -> SpeechSftConfig {
    serde_json::from_value(json!({"base": "", "steps": 1, "rows": {}})).expect("the objective's defaults parse")
}

/// The objective's settings as served params, each bounded, with the objective's own default as its implied one.
fn training_params() -> Vec<ParamSpec> {
    let d = objective_defaults();
    let int = |name: &str, help: &str, max: f64| ParamSpec::new(name, ParamType::Int, help).min(1.0).max(max).step(1.0);
    let float = |name: &str, help: &str, max: f64, implied: f32| ParamSpec::new(name, ParamType::Float, help).min(0.0).max(max).implied_default(json!(implied));
    let multiplier = |name: &str, group: &str, implied: f32| float(name, &format!("learning-rate multiplier of {group}"), MAX_LR_MULTIPLIER, implied);
    vec![
        int("steps", "optimiser steps; exactly one of steps and epochs", capability::bounds::TRAIN_STEPS),
        int("epochs", "passes over the training shard, instead of steps", capability::bounds::TRAIN_COUNT),
        float("lr", "peak learning rate", 1.0, d.lr),
        float("weight_decay", "AdamW weight decay", 1.0, d.weight_decay),
        ParamSpec::new("warmup", ParamType::Int, "steps of linear learning-rate warmup").min(0.0).max(capability::bounds::TRAIN_COUNT).step(1.0).implied_default(json!(d.warmup)),
        ParamSpec::new("schedule", ParamType::Str, "the learning-rate curve after warmup: `cosine` (decay to a tenth by the last step) or `constant`").implied_default(json!(d.schedule)),
        int("batch_utterances", "utterances per micro-batch", MAX_BATCH).implied_default(json!(d.batch)),
        int("grad_accum", "micro-batches per optimiser step", 4096.0).implied_default(json!(d.grad_accum)),
        int("max_frames", "the most codec frames a record may hold; a longer one is refused by name, never cut (it sizes the build)", MAX_FRAMES).implied_default(json!(d.max_frames)),
        int("length_bucket_steps", "micro-batches whose utterances are regrouped by length, so each micro-batch pads little; 1 draws them unsorted", 4096.0).implied_default(json!(d.length_bucket_steps)),
        float("grad_clip", "the global gradient-norm clip", 1_000.0, d.grad_clip),
        float("residual_weight", "the weight of the code predictor's residual-codebook loss beside the Talker's (upstream trains with 0.3)", 10.0, d.mtp_weight),
        float("streaming_fraction", "the share of drawn records trained in the streaming text layout; the rest train non-streaming", 1.0, d.streaming_fraction),
        multiplier("talker_lr_multiplier", "the Talker decoder and its head", d.lr_multipliers.talker),
        multiplier("code_predictor_lr_multiplier", "the code predictor", d.lr_multipliers.code_predictor),
        multiplier("codec_table_lr_multiplier", "the codec table rows the base already trained", d.lr_multipliers.codec_tables),
        multiplier("new_row_lr_multiplier", "the rows of the new languages and voices", d.lr_multipliers.new_rows),
        float("voice_outlier_fraction", "the share of a new voice's x-vectors, least aligned with their mean direction, left out of the centroid its row starts at", 0.99, d.rows.voice_outlier_fraction),
        float("free_row_fraction", "a codec row is free for a new name when its norm is below this share of the median norm of the rows the base names", 0.99, d.rows.free_row_fraction),
        ParamSpec::new("eval_every", ParamType::Int, "score the held-out shard every this many steps (needs one)").min(1.0).max(capability::bounds::TRAIN_COUNT).step(1.0),
        ParamSpec::new("eval_batches", ParamType::Int, "held-out micro-batches per evaluation (0: every record)").min(0.0).max(capability::bounds::TRAIN_COUNT).step(1.0).implied_default(json!(d.eval_batches)),
        ParamSpec::new("patience", ParamType::Int, "stop when the held-out loss has not improved for this many evaluations, and publish the checkpoint of the best one (needs eval_every)").min(1.0).max(capability::bounds::TRAIN_COUNT).step(1.0),
        ParamSpec::new("keep_best", ParamType::Bool, "publish the checkpoint of the evaluation with the lowest held-out loss instead of the last step's, running every step (implied by patience; needs eval_every)").implied_default(json!(d.keep_best)),
        ParamSpec::new("seed", ParamType::Int, "seed of the batch order and the layout draws (random when omitted; the report carries the one used)").min(0.0).max(capability::bounds::EXACT_INTEGER),
    ]
}

/// What `speech_sft` is.
pub fn spec() -> ActionSpec {
    let blob = |name: &str, help: &str| BlobSpec::new(name, Media::Bytes, help);
    ActionSpec::new(ACTION, "fine-tune a Qwen3-TTS checkpoint on recorded speech (training shards), with new languages and named voices: the checkpoint leaves in its base's upstream layout by manifest digest in the artifact store (and optionally into the model store, where it is served as a model of its own)")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("base", ParamType::Str, "the Qwen3-TTS base, `<vendor>/<repo>` in the server's model store").required())
        .param(ParamSpec::new("dataset_digest", ParamType::Str, "the training shard (`brain-tts-shard-v1`), as a blake3 digest of the server's artifact store (or attach it as the input blob `dataset`; exactly one of the two)"))
        .param(ParamSpec::new("held_out_digest", ParamType::Str, "a held-out shard scored every eval_every steps, as a digest (or the input blob `held_out`; at most one of the two)"))
        .param(ParamSpec::new("languages", ParamType::Str, "JSON object naming every language the shards hold: name -> {\"new\": true, \"init_from\"?: [existing names]} for one the base lacks (its codec row starts at the mean of those, by default of every base language; every new language of a run names the same init_from), or {\"new\": false} for one of the base's; names are lowercase [a-z0-9_-]").required())
        .param(ParamSpec::new("voices", ParamType::Str, "JSON object naming every voice the shards train: name -> {\"new\": true} for a new voice (its row initialised from its speaker's x-vectors), or {\"new\": false} for one of the base's; a checkpoint with named voices is exported as `custom_voice`").default(json!("{}")))
        .params(training_params())
        .param(crate::parallel::cards_param("the whole Talker and code predictor with their optimizer state", "every speech fine-tune"))
        .param(crate::parallel::job_param("sharding replicated", ""))
        .param(ParamSpec::new("install_as", ParamType::Str, "also install the checkpoint in the server's model store as `local/<this name>` (letters, digits, `.`, `_`, `-`; it must not exist yet), where it is served as a model of its own with its voices and languages"))
        .param(resume_key_param())
        .param(checkpoint_every_param())
        .param(crate::consolidate::allow_cpu_param())
        .param(ParamSpec::new("validate_only", ParamType::Bool, "check the request, the base and the shards (formats, languages and voices against the names given) and answer with what the run would train, without training: host work that takes no card").default(json!(false)))
        .input(blob("dataset", "the training shard (`brain-tts-shard-v1`)"))
        .input(blob("held_out", "a held-out shard"))
}

/// A language as the request names it.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LanguageDoc {
    new: bool,
    #[serde(default)]
    init_from: Vec<String>,
}

/// A voice as the request names it.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VoiceDoc {
    new: bool,
}

/// The scalar settings of `speech_sft`, exactly the params its spec declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub base: String,
    pub dataset_digest: Option<String>,
    pub held_out_digest: Option<String>,
    pub languages: String,
    pub voices: String,
    pub steps: Option<u32>,
    pub epochs: Option<u32>,
    pub lr: Option<f32>,
    pub weight_decay: Option<f32>,
    pub warmup: Option<u32>,
    pub schedule: Option<String>,
    pub batch_utterances: Option<u32>,
    pub grad_accum: Option<u32>,
    pub max_frames: Option<u32>,
    pub length_bucket_steps: Option<u32>,
    pub grad_clip: Option<f32>,
    pub residual_weight: Option<f32>,
    pub streaming_fraction: Option<f32>,
    pub talker_lr_multiplier: Option<f32>,
    pub code_predictor_lr_multiplier: Option<f32>,
    pub codec_table_lr_multiplier: Option<f32>,
    pub new_row_lr_multiplier: Option<f32>,
    pub voice_outlier_fraction: Option<f32>,
    pub free_row_fraction: Option<f32>,
    pub eval_every: Option<u32>,
    pub eval_batches: Option<u32>,
    pub patience: Option<u32>,
    pub keep_best: Option<bool>,
    pub seed: Option<u64>,
    pub cards: u32,
    pub job: Option<String>,
    pub install_as: Option<String>,
    pub resume_key: Option<String>,
    pub checkpoint_every: Option<u32>,
    pub allow_cpu: bool,
    pub validate_only: bool,
}

/// A `speech_sft` request that passed the spec and every check that needs no file and no device.
#[derive(Debug)]
pub struct Request {
    pub params: Params,
    pub dataset: Source,
    pub held_out: Option<Source>,
    pub languages: BTreeMap<String, LanguageInit>,
    pub voices: BTreeMap<String, bool>,
    /// How the run is spread over this server's cards and, with a `job`, over several servers.
    pub spread: crate::parallel::Spread,
}

/// A language or voice name as upstream looks one up (lowercased).
fn check_name(what: &str, name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-') {
        return Err(format!("{ACTION}: {what} {name:?} is not a name (1 to 64 lowercase letters, digits, `_`, `-`)"));
    }
    Ok(())
}

fn names_doc<T: serde::de::DeserializeOwned>(field: &str, doc: &str) -> Result<BTreeMap<String, T>, String> {
    if doc.len() > MAX_NAMES_DOC {
        return Err(format!("{ACTION}: {field} is {} bytes, more than {MAX_NAMES_DOC}", doc.len()));
    }
    let map: BTreeMap<String, T> = serde_json::from_str(doc).map_err(|e| format!("{ACTION}: {field}: {e}"))?;
    if map.len() > MAX_NAMES {
        return Err(format!("{ACTION}: {field} names {} entries, more than {MAX_NAMES}", map.len()));
    }
    Ok(map)
}

fn source(inv: &Invocation, blob: &str, digest: Option<&str>, digest_param: &str) -> Result<Option<Source>, String> {
    match (digest, inv.blobs.contains_key(blob)) {
        (Some(d), false) => Digest::parse(d).map(|d| Some(Source::Digest(d))).map_err(|e| format!("{ACTION}: {digest_param}: {e}")),
        (None, true) => Ok(Some(Source::Blob(blob.to_string()))),
        (None, false) => Ok(None),
        (Some(_), true) => Err(format!("{ACTION}: name the {blob} once: as the input blob `{blob}` or as `{digest_param}`, not both")),
    }
}

impl Request {
    /// Reads the validated invocation `inv`.
    pub fn from_invocation(inv: &Invocation) -> Result<Request, String> {
        let params: Params = serde_json::from_value(inv.params.clone()).map_err(|e| format!("{ACTION}: {e}"))?;
        let dataset = source(inv, "dataset", params.dataset_digest.as_deref(), "dataset_digest")?.ok_or_else(|| format!("{ACTION}: name the training shard: the input blob `dataset` or `dataset_digest`"))?;
        let held_out = source(inv, "held_out", params.held_out_digest.as_deref(), "held_out_digest")?;
        if let Some(extra) = inv.blobs.keys().find(|k| !["dataset", "held_out"].contains(&k.as_str())) {
            return Err(format!("{ACTION}: the input blob \"{extra}\" is not an input of this action"));
        }
        if params.eval_every.is_some() && held_out.is_none() {
            return Err(format!("{ACTION}: eval_every scores a held-out shard; attach `held_out` or name `held_out_digest`"));
        }
        if (params.patience.is_some() || params.keep_best == Some(true)) && params.eval_every.is_none() {
            return Err(format!("{ACTION}: patience and keep_best select on the held-out loss; name eval_every (and a held-out shard)"));
        }
        let mut languages = BTreeMap::new();
        for (name, doc) in names_doc::<LanguageDoc>("languages", &params.languages)? {
            check_name("language", &name)?;
            if name.contains("dialect") {
                return Err(format!("{ACTION}: language {name:?}: upstream hides a language whose name says dialect"));
            }
            if !doc.new && !doc.init_from.is_empty() {
                return Err(format!("{ACTION}: language {name:?} is the base's own; init_from belongs to a new one"));
            }
            for from in &doc.init_from {
                check_name("language", from)?;
            }
            languages.insert(name, LanguageInit { new: doc.new, init_from: doc.init_from });
        }
        let mixes: std::collections::BTreeSet<&Vec<String>> = languages.values().filter(|l| l.new).map(|l| &l.init_from).collect();
        if mixes.len() > 1 {
            return Err(format!("{ACTION}: every new language of a run starts from the same mix; the languages name different init_from"));
        }
        if params.steps.is_some() == params.epochs.is_some() {
            return Err(format!("{ACTION}: name exactly one of steps and epochs"));
        }
        if let Some(schedule) = &params.schedule {
            serde_json::from_value::<qwen3tts::speech_sft::Schedule>(json!(schedule)).map_err(|_| format!("{ACTION}: schedule {schedule:?} is neither cosine nor constant"))?;
        }
        if languages.is_empty() {
            return Err(format!("{ACTION}: languages names no language; name every language the shards hold"));
        }
        let mut voices = BTreeMap::new();
        for (name, doc) in names_doc::<VoiceDoc>("voices", &params.voices)? {
            check_name("voice", &name)?;
            voices.insert(name, doc.new);
        }
        if let Some(name) = &params.install_as {
            check_install_name(name)?;
        }
        check_resume(ACTION, params.resume_key.as_deref(), params.checkpoint_every)?;
        let spread = crate::parallel::Spread::parse(ACTION, params.cards, params.job.as_deref(), params.seed, false)?;
        Ok(Request { params, dataset, held_out, languages, voices, spread })
    }

    /// Every digest the call names.
    pub fn digests(&self) -> Vec<Digest> {
        std::iter::once(&self.dataset).chain(&self.held_out).filter_map(|s| if let Source::Digest(d) = s { Some(*d) } else { None }).collect()
    }
}

/// The base of `request` in the model store, checked to be a Qwen3-TTS checkpoint that has every language and voice
/// the request calls the base's own, and none it calls new.
pub(crate) fn resolve_base(env: &JobEnv, request: &Request) -> Result<std::path::PathBuf, String> {
    let dir = job::resolve_base(env.models_dir.as_deref(), &request.params.base)?;
    let config: Value = std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).ok_or_else(|| format!("{ACTION}: base \"{}\" is not a Qwen3-TTS checkpoint (it has no readable config.json)", request.params.base))?;
    if !config["architectures"].as_array().is_some_and(|a| a.iter().any(|x| x == ARCHITECTURE)) {
        return Err(format!("{ACTION}: base \"{}\" is not a Qwen3-TTS checkpoint", request.params.base));
    }
    let specials = qwen3tts::TtsSpecials::from_config_dir(job::utf8(&dir)?).map_err(|e| format!("{ACTION}: base: {e}"))?;
    for (name, init) in &request.languages {
        match (init.new, specials.lang.contains_key(name)) {
            (true, true) => return Err(format!("{ACTION}: language {name:?} is the base's own; name it {{\"new\": false}}")),
            (false, false) => return Err(format!("{ACTION}: language {name:?} is not the base's; a new one is {{\"new\": true}}")),
            _ => {}
        }
        if let Some(from) = init.init_from.iter().find(|f| !specials.lang.contains_key(*f)) {
            return Err(format!("{ACTION}: language {name:?} is initialised from {from:?}, which is not one of the base's languages"));
        }
    }
    for (name, new) in &request.voices {
        match (*new, specials.spk_id.contains_key(name)) {
            (true, true) => return Err(format!("{ACTION}: voice {name:?} is the base's own; name it {{\"new\": false}}")),
            (false, false) => return Err(format!("{ACTION}: voice {name:?} is not the base's; a new one is {{\"new\": true}}")),
            _ => {}
        }
    }
    Ok(dir)
}

/// Bytes per trained parameter: the objective trains in f32, holding the weight, its gradient and the two AdamW
/// moments, 4 bytes each.
const BYTES_PER_TRAINED_PARAM: u64 = 16;

/// What one rank of a run of `request` is charged on its card: the Talker's and the code predictor's parameters at
/// [`BYTES_PER_TRAINED_PARAM`] (the codec and the speaker encoder are not trained), the workspace, and the activations
/// of a micro-batch of `batch_utterances` records of `max_frames` frames each, the most a micro-batch can hold (the
/// build is sized by the shard's longest record, which is not known before the shard is staged).
pub(crate) fn device_bytes(base_dir: &Path, request: &Request) -> u64 {
    let paths = qwen3tts::TtsPaths::new(base_dir, base_dir.to_string_lossy());
    let defaults = objective_defaults();
    let frames = u64::from(request.params.batch_utterances.unwrap_or(defaults.batch)) * u64::from(request.params.max_frames.unwrap_or(defaults.max_frames));
    let config: Value = std::fs::read(base_dir.join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    let (talker, mtp) = (qwen3tts::TalkerConfig::from_json(&config), qwen3tts::MtpConfig::from_json(&config));
    let width = u64::from(talker.n_layers) * u64::from(talker.d_model + talker.d_ff) + u64::from(mtp.num_code_groups) * u64::from(mtp.n_layers) * u64::from(mtp.d_model + mtp.d_ff);
    let activations = (frames as f64 * width as f64 * ACTIVATION_FLOATS_PER_WIDTH * 4.0) as u64;
    paths.talker_and_mtp_params().saturating_mul(BYTES_PER_TRAINED_PARAM).saturating_add(WORKSPACE_BYTES).saturating_add(activations)
}

/// The staged shards, each opened and checked against the base and the request: the codebooks and x-vector width of
/// the base, and exactly the languages and voices the request names.
fn open_shard(path: &Path, what: &str, base_dir: &Path, request: &Request) -> Result<qwen3tts::shard::Shard, String> {
    let shard = qwen3tts::shard::Shard::open(path).map_err(|e| format!("{ACTION}: {what}: {e}"))?;
    let config: Value = serde_json::from_slice(&std::fs::read(base_dir.join("config.json")).map_err(|e| format!("{ACTION}: base config: {e}"))?).map_err(|e| format!("{ACTION}: base config: {e}"))?;
    let groups = config["talker_config"]["num_code_groups"].as_u64();
    if groups != Some(u64::from(shard.meta.codebooks)) {
        return Err(format!("{ACTION}: {what} holds {} codebooks per frame and the base codes {groups:?}", shard.meta.codebooks));
    }
    let width = config["speaker_encoder_config"]["enc_dim"].as_u64();
    if width != Some(u64::from(shard.meta.xvector_dim)) {
        return Err(format!("{ACTION}: {what}'s x-vectors are {} wide and the base's speaker encoder writes {width:?}", shard.meta.xvector_dim));
    }
    if let Some(name) = shard.meta.languages.iter().find(|l| !request.languages.contains_key(*l)) {
        return Err(format!("{ACTION}: {what} holds language {name:?}, which languages does not name"));
    }
    if let Some(name) = shard.meta.voices.iter().find(|v| !request.voices.contains_key(*v)) {
        return Err(format!("{ACTION}: {what} trains voice {name:?}, which voices does not name"));
    }
    if let Some(name) = request.languages.keys().find(|l| !shard.meta.languages.contains(l)) {
        return Err(format!("{ACTION}: languages names {name:?}, which {what} does not hold"));
    }
    if let Some(name) = request.voices.keys().find(|v| !shard.meta.voices.contains(v)) {
        return Err(format!("{ACTION}: voices names {name:?}, which {what} does not train"));
    }
    if shard.is_empty() {
        return Err(format!("{ACTION}: {what} holds no record"));
    }
    Ok(shard)
}

/// A step's frame: what the objective reported for it (its loss, the Talker's and the code predictor's cross-entropy,
/// end-of-speech accuracy and frames, and after an evaluation the held-out ones).
fn step_frame(step: &model::train::StepReport, seconds: f64) -> Value {
    let frames_per_s = step.work.useful_tokens_per_s();
    let mut frame = json!({"type": "step", "step": step.step, "steps": step.steps, "loss": step.loss, "lr": step.lr, "grad_norm": step.grad_norm, "frames_per_s": if frames_per_s.is_finite() { json!(frames_per_s) } else { Value::Null }, "seconds": seconds});
    if let Some(eval) = step.eval_loss {
        frame["eval_loss"] = json!(eval);
    }
    for (name, value) in &step.metrics {
        frame[*name] = json!(value);
    }
    frame
}

/// The objective's configuration for `request` over the base at `base_dir`: the settings it names, the objective's own
/// defaults for the rest, validated as the run would ([`SpeechSftConfig::validate`]).
pub(crate) fn sft_config(request: &Request, base_dir: &Path, seed: Option<u64>) -> Result<SpeechSftConfig, String> {
    let p = &request.params;
    let source = |new: bool| if new { "new" } else { "base" };
    let mix = request.languages.values().find(|l| l.new && !l.init_from.is_empty()).map(|l| l.init_from.iter().map(|n| (n.clone(), 1.0 / l.init_from.len() as f32)).collect::<BTreeMap<String, f32>>());
    let mut rows = json!({
        "languages": request.languages.iter().map(|(n, l)| (n.clone(), json!(source(l.new)))).collect::<serde_json::Map<_, _>>(),
        "voices": request.voices.iter().map(|(n, new)| (n.clone(), json!(source(*new)))).collect::<serde_json::Map<_, _>>(),
        "language_init": mix,
    });
    let mut cfg = json!({"base": job::utf8(base_dir)?, "rows": {}});
    let set = |cfg: &mut Value, key: &str, v: Option<Value>| {
        if let Some(v) = v {
            cfg[key] = v;
        }
    };
    for (key, v) in [
        ("steps", p.steps.map(|v| json!(v))),
        ("epochs", p.epochs.map(|v| json!(v))),
        ("lr", p.lr.map(|v| json!(v))),
        ("weight_decay", p.weight_decay.map(|v| json!(v))),
        ("warmup", p.warmup.map(|v| json!(v))),
        ("schedule", p.schedule.clone().map(Value::from)),
        ("batch", p.batch_utterances.map(|v| json!(v))),
        ("grad_accum", p.grad_accum.map(|v| json!(v))),
        ("max_frames", p.max_frames.map(|v| json!(v))),
        ("length_bucket_steps", p.length_bucket_steps.map(|v| json!(v))),
        ("grad_clip", p.grad_clip.map(|v| json!(v))),
        ("mtp_weight", p.residual_weight.map(|v| json!(v))),
        ("streaming_fraction", p.streaming_fraction.map(|v| json!(v))),
        ("eval_interval", p.eval_every.map(|v| json!(v))),
        ("eval_batches", p.eval_batches.map(|v| json!(v))),
        ("patience", p.patience.map(|v| json!(v))),
        ("keep_best", p.keep_best.map(|v| json!(v))),
        ("seed", seed.or(p.seed).map(|v| json!(v))),
    ] {
        set(&mut cfg, key, v);
    }
    let mut multipliers = json!({});
    for (key, v) in [("talker", p.talker_lr_multiplier), ("code_predictor", p.code_predictor_lr_multiplier), ("codec_tables", p.codec_table_lr_multiplier), ("new_rows", p.new_row_lr_multiplier)] {
        set(&mut multipliers, key, v.map(|v| json!(v)));
    }
    cfg["lr_multipliers"] = multipliers;
    set(&mut rows, "voice_outlier_fraction", p.voice_outlier_fraction.map(|v| json!(v)));
    set(&mut rows, "free_row_fraction", p.free_row_fraction.map(|v| json!(v)));
    cfg["rows"] = rows;
    let cfg: SpeechSftConfig = serde_json::from_value(cfg).map_err(|e| format!("{ACTION}: {e}"))?;
    cfg.validate().map_err(|e| format!("{ACTION}: {e}"))?;
    Ok(cfg)
}

/// What the run reports about its training: the shared loop's totals, the last step's measurements, the held-out
/// curve, and which checkpoint the run published: the last step's (`"checkpoint": "last"`) or, selecting on held-out
/// loss, the best evaluation's (`"best"`, with its step and loss in `best`, and whether the run stopped early).
fn training_report(report: &qwen3tts::speech_sft::SpeechSftReport) -> Value {
    let fit = &report.fit;
    let last = report.steps.last().map(|s| {
        json!({"step": s.step, "loss": s.loss, "talker_ce": s.talker_ce, "code_predictor_ce": s.code_predictor_ce, "eos_accuracy": s.eos_accuracy, "grad_norm": s.grad_norm, "lr": s.lr, "frames_per_s": s.frames_per_s,
               "held_out": s.held_out.map(|h| json!({"loss": h.loss, "talker_ce": h.talker_ce, "code_predictor_ce": h.code_predictor_ce, "eos_accuracy": h.eos_accuracy}))})
    });
    let curve: Vec<Value> = fit.evaluations.iter().map(|e| json!({"step": e.step, "train_loss": e.train_loss, "held_out_loss": e.eval_loss})).collect();
    let best = fit.kept_best.map(|b| json!({"step": b.step, "held_out_loss": b.eval_loss}));
    json!({"initial_loss": fit.initial_loss, "final_loss": fit.final_loss, "steps_completed": fit.steps_completed, "evaluations": fit.evaluations.len(), "last_step": last,
           "held_out_curve": curve, "checkpoint": if best.is_some() { "best" } else { "last" }, "best": best, "stopped_early": fit.stopped_early})
}

/// Runs the `speech_sft` job `inv` on `devices`, the cards claimed for it: on the one this thread is scoped to, or as
/// one data-parallel rank per card of the request's spread ([`crate::parallel::train_as_ranks`]), every rank training
/// its share of the records with the same seed and every step's gradients summed over the group. Rank 0 streams the
/// steps, and its server exports and publishes the checkpoint.
pub fn run(env: &JobEnv, inv: &Invocation, devices: &[residency::Device], progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let request = Request::from_invocation(inv)?;
    let p = &request.params;
    let base_dir = resolve_base(env, &request)?;
    let work = Workdir::create(env)?;
    let total = p.steps.unwrap_or(0);
    progress(Progress::event(0, total, json!({"type": "stage", "stage": "staging the shards"})));
    let stage = |source: &Source, name: &str| -> Result<std::path::PathBuf, String> {
        let path = work.path().join(name);
        bring(source, &inv.blobs, env.store.as_ref(), work.staging(), &path)?;
        Ok(path)
    };
    let train = open_shard(&stage(&request.dataset, "dataset")?, "dataset", &base_dir, &request)?;
    let held_out = request.held_out.as_ref().map(|s| stage(s, "held_out").and_then(|path| open_shard(&path, "held_out", &base_dir, &request))).transpose()?;
    if p.validate_only {
        let mut outcome = Outcome::new();
        outcome.outputs = json!({
            "status": "valid", "objective": ACTION, "base": p.base, "steps": p.steps, "epochs": p.epochs,
            "records": train.len(), "held_out_records": held_out.as_ref().map(qwen3tts::shard::Shard::len),
            "languages": request.languages.keys().collect::<Vec<_>>(), "voices": request.voices.keys().collect::<Vec<_>>(),
            "device_bytes": device_bytes(&base_dir, &request),
        });
        return Ok(outcome);
    }
    let run = match open_run(env, ACTION, p.resume_key.as_deref(), inv, p.seed, &request.spread.layout(Some(brain::Sharding::Replicated)))? {
        Some(crate::runs::Opened::Completed(mut result)) => {
            result["resume"] = crate::runs::completed_report(p.resume_key.as_deref().unwrap_or_default());
            let mut outcome = Outcome::new();
            outcome.outputs = result;
            return Ok(outcome);
        }
        Some(crate::runs::Opened::Run(run)) => Some(*run),
        None => None,
    };
    // Every rank of a spread run draws the same initialisation and batch order: the resumable run's seed, the
    // request's, or one drawn here for all of them.
    let seed = run.as_ref().map(crate::runs::Run::seed).or(p.seed).or_else(|| request.spread.ranked().then(brain::random_seed));
    let cfg = sft_config(&request, &base_dir, seed)?;
    progress(Progress::event(0, total, json!({"type": "stage", "stage": "loading the base and training"})));
    let began = Instant::now();
    let state = run.as_ref().map(crate::runs::Run::state_path);
    let identity = json!({"base": p.base, "dataset": request.digests().iter().map(Digest::hex).collect::<Vec<_>>()});
    let state_every = p.checkpoint_every.unwrap_or(crate::runs::DEFAULT_CHECKPOINT_EVERY);
    // One rank's training: `rank` is its place in a spread run's group, `frames` where its steps are reported.
    let train_one = |rank: Option<&brain::DataParallel>, report: &mut dyn FnMut(Progress)| -> Result<qwen3tts::speech_sft::SpeechSftResult, String> {
        let mut on_step = |step: &model::train::StepReport| {
            report(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64())));
            match rank {
                // The ranks of a group stop after the same step.
                Some(rank) => rank.keep_going(step.step, inv.cancel.is_cancelled()),
                None => !inv.cancel.is_cancelled(),
            }
        };
        let control = model::train::FitControl { on_step: Some(&mut on_step), state: state.as_deref(), state_every, identity: identity.clone(), snapshots: None, publish: None, grad_sync: rank.map(brain::DataParallel::grad_sync) };
        qwen3tts::speech_sft::speech_sft(&cfg, &train, held_out.as_ref(), control).map_err(|e| format!("{ACTION}: {e}"))
    };
    let (trained, parallel) = if request.spread.ranked() {
        let rank = |_: usize, rank: brain::DataParallel, frames: Option<std::sync::mpsc::Sender<Progress>>| {
            let trained = train_one(Some(&rank), &mut |frame| {
                if let Some(frames) = &frames {
                    let _ = frames.send(frame);
                }
            })?;
            Ok((trained, rank.report()))
        };
        let mut ranks = crate::parallel::train_as_ranks(&request.spread, ACTION, env, devices, brain::Sharding::Replicated, rank, progress)?;
        let (trained, parallel) = ranks.swap_remove(0);
        (trained, Some(parallel))
    } else {
        (train_one(None, progress)?, None)
    };
    if trained.report.fit.interrupted {
        return Err(capability::failure::CANCELLED.to_string());
    }
    // A server that runs none of a job's rank 0 answers for its ranks only: the server of rank 0 publishes the run.
    if let Some(mut answer) = request.spread.answer_for_ranks(ACTION) {
        answer["steps_completed"] = json!(trained.report.fit.steps_completed);
        // Its part of the job is done: a resubmission answers this without joining the job again.
        if let Some(run) = run {
            run.finish(&mut answer, trained.report.fit.resumed_at)?;
        }
        let mut outcome = Outcome::new();
        outcome.outputs = answer;
        return Ok(outcome);
    }

    let id = p.install_as.as_ref().map_or_else(|| DEFAULT_ID.to_string(), |name| format!("local/{name}"));
    let out = work.path().join("release");
    let exported = qwen3tts::export::export(&base_dir, &out, &id, &qwen3tts::export::Trained { tensors: &trained.tensors, languages: &trained.languages, voices: &trained.voices }, &mut || !inv.cancel.is_cancelled())
        .map_err(|e| if e == capability::failure::CANCELLED { e } else { format!("{ACTION}: {e}") })?;
    let mut report = training_report(&trained.report);
    let resumed_at = trained.report.fit.resumed_at;
    report["status"] = json!("completed");
    report["objective"] = json!(ACTION);
    report["base"] = json!(p.base);
    report["seed"] = json!(trained.report.seed);
    report["languages"] = json!(trained.languages);
    report["voices"] = json!(trained.voices);
    report["tts_model_type"] = json!(match exported.kind {
        qwen3tts::prompt::CheckpointKind::Base => "base",
        qwen3tts::prompt::CheckpointKind::CustomVoice => "custom_voice",
        qwen3tts::prompt::CheckpointKind::VoiceDesign => "voice_design",
    });
    report["trained_tensors"] = json!(trained.tensors.len());
    if let Some(parallel) = parallel {
        report["parallel"] = serde_json::to_value(parallel).map_err(|e| format!("{ACTION}: {e}"))?;
    }
    work.ensure_within_quota()?;
    checkpoint_outcome_of(env, &out, p.install_as.as_deref(), report, run.map(|r| (r, resumed_at)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `params` as a remote caller's request, with `steps: 3` unless it names steps or epochs.
    fn validated(mut params: Value, blobs: &[&str]) -> Result<Request, String> {
        if params.get("steps").is_none() && params.get("epochs").is_none() {
            params["steps"] = json!(3);
        }
        let mut inv = Invocation::new();
        inv.params = params;
        for name in blobs {
            inv.blobs.insert((*name).to_string(), capability::Blob::new(Media::Bytes, b"x".to_vec()));
        }
        spec().validate_served(inv).and_then(|i| Request::from_invocation(&i))
    }

    fn digest(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    const SV: &str = r#"{"swedish": {"new": true, "init_from": ["english", "german"]}}"#;

    /// A step's frame carries what the objective reported: the step, the loss, the learning rate, the gradient norm,
    /// a held-out loss when one was measured, and the objective's own metrics by name.
    #[test]
    fn a_step_frame_carries_the_reports_fields() {
        let step = model::train::StepReport { step: 3, steps: 9, loss: 1.5, lr: 1e-4, grad_norm: Some(0.5), work: Default::default(), eval_loss: Some(2.0), metrics: vec![("talker_ce", 1.25), ("code_predictor_ce", 0.75), ("eos_accuracy", 0.5), ("held_out_talker_ce", 2.25)] };
        let frame = step_frame(&step, 7.0);
        assert_eq!(frame, json!({"type": "step", "step": 3, "steps": 9, "loss": 1.5, "lr": 1e-4_f32, "grad_norm": 0.5, "frames_per_s": 0.0, "seconds": 7.0, "eval_loss": 2.0, "talker_ce": 1.25, "code_predictor_ce": 0.75, "eos_accuracy": 0.5, "held_out_talker_ce": 2.25}));
    }

    /// A cancelled job stops before it reads anything.
    #[test]
    fn a_cancelled_job_stops_before_any_work() {
        let env = JobEnv { store: None, models_dir: None, work_root: std::env::temp_dir(), quotas: Default::default(), job_access: None, largest_card_bytes: None, bank: Default::default() };
        let mut inv = Invocation::new();
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        assert_eq!(run(&env, &inv, &[], &mut |_| {}).unwrap_err(), capability::failure::CANCELLED);
    }

    #[test]
    fn a_request_names_its_shards_languages_and_voices_and_refuses_what_it_cannot_take() {
        let ok = validated(json!({"base": "a/b", "languages": SV, "voices": r#"{"astrid": {"new": true}}"#, "install_as": "sv-1"}), &["dataset"]).unwrap();
        let cfg = sft_config(&ok, Path::new("base"), None).unwrap();
        assert_eq!((cfg.steps, cfg.batch, cfg.mtp_weight, cfg.rows.voices["astrid"]), (Some(3), objective_defaults().batch, 0.3, qwen3tts::speech_sft::RowSource::New));
        assert_eq!(cfg.rows.language_init, Some(BTreeMap::from([("english".to_string(), 0.5), ("german".to_string(), 0.5)])));
        let named = validated(json!({"base": "a/b", "languages": SV, "batch_utterances": 2, "residual_weight": 0.5, "new_row_lr_multiplier": 10.0, "streaming_fraction": 1.0, "warmup": 4, "schedule": "constant", "lr": 1e-4, "eval_batches": 2, "seed": 9, "length_bucket_steps": 16}), &["dataset"]).unwrap();
        let cfg = sft_config(&named, Path::new("base"), None).unwrap();
        assert_eq!((cfg.batch, cfg.mtp_weight, cfg.lr_multipliers.new_rows, cfg.streaming_fraction, cfg.warmup, cfg.seed, cfg.length_bucket_steps), (2, 0.5, 10.0, 1.0, 4, Some(9), 16));
        assert_eq!(ok.dataset, Source::Blob("dataset".into()));
        assert_eq!(ok.languages["swedish"], LanguageInit { new: true, init_from: vec!["english".into(), "german".into()] });
        assert_eq!((ok.voices["astrid"], ok.params.steps, ok.params.residual_weight), (true, Some(3), None));
        let spread = validated(json!({"base": "a/b", "languages": SV, "cards": 4, "eval_every": 5, "patience": 3, "seed": 1}), &["dataset", "held_out"]).unwrap();
        let cfg = sft_config(&spread, Path::new("base"), None).unwrap();
        assert_eq!((spread.spread.cards, spread.spread.ranked(), cfg.patience, cfg.eval_interval), (4, true, 3, 5));
        let by_digest = validated(json!({"base": "a/b", "languages": SV, "dataset_digest": digest(1), "held_out_digest": digest(2), "eval_every": 10}), &[]).unwrap();
        assert_eq!(by_digest.digests().len(), 2);
        let cases: Vec<(Value, Vec<&str>, &str)> = vec![
            (json!({"base": "a/b", "languages": SV}), vec![], "training shard"),
            (json!({"base": "a/b", "languages": SV, "dataset_digest": digest(1)}), vec!["dataset"], "not both"),
            (json!({"base": "a/b", "languages": SV, "dataset_digest": "../../etc/passwd"}), vec![], "dataset_digest"),
            (json!({"base": "a/b", "languages": SV}), vec!["dataset", "weights"], "unknown input"),
            (json!({"base": "a/b", "languages": SV, "eval_every": 5}), vec!["dataset"], "held-out"),
            (json!({"base": "a/b", "languages": "{}"}), vec!["dataset"], "no language"),
            (json!({"base": "a/b", "languages": r#"{"Swedish": {"new": true}}"#}), vec!["dataset"], "not a name"),
            (json!({"base": "a/b", "languages": r#"{"sv_dialect": {"new": true}}"#}), vec!["dataset"], "dialect"),
            (json!({"base": "a/b", "languages": r#"{"english": {"new": false, "init_from": ["german"]}}"#}), vec!["dataset"], "init_from"),
            (json!({"base": "a/b", "languages": r#"{"swedish": {"new": true, "path": "/x"}}"#}), vec!["dataset"], "unknown field"),
            (json!({"base": "a/b", "languages": SV, "voices": r#"{"astrid": {"new": true, "wav": "/x"}}"#}), vec!["dataset"], "unknown field"),
            (json!({"base": "a/b", "languages": " ".repeat(MAX_NAMES_DOC + 1)}), vec!["dataset"], "more than"),
            (json!({"base": "a/b", "languages": SV, "install_as": "../x"}), vec!["dataset"], "install_as"),
            (json!({"base": "a/b", "languages": SV, "checkpoint_every": 5}), vec!["dataset"], "resume_key"),
            (json!({"base": "a/b", "languages": SV, "cards": 0}), vec!["dataset"], "cards"),
            (json!({"base": "a/b", "languages": SV, "patience": 3}), vec!["dataset", "held_out"], "eval_every"),
            (json!({"base": "a/b", "languages": SV, "keep_best": true}), vec!["dataset", "held_out"], "eval_every"),
            (json!({"base": "a/b", "languages": SV, "job": "{}"}), vec!["dataset"], "job"),
            (json!({"base": "a/b", "languages": SV, "out": "x"}), vec!["dataset"], "unknown param"),
            (json!({"base": "a/b", "languages": SV, "steps": 0}), vec!["dataset"], "steps"),
            (json!({"base": "a/b", "languages": SV, "steps": 3, "epochs": 1}), vec!["dataset"], "exactly one of steps and epochs"),
            (json!({"base": "a/b", "languages": SV, "schedule": "linear"}), vec!["dataset"], "schedule"),
            (json!({"base": "a/b", "languages": SV, "batch_utterances": 0}), vec!["dataset"], "batch_utterances"),
            (json!({"base": "a/b", "languages": SV, "max_frames": 1_000_000}), vec!["dataset"], "max_frames"),
            (json!({"base": "a/b", "languages": SV, "streaming_fraction": 1.5}), vec!["dataset"], "streaming_fraction"),
            (json!({"base": "a/b", "languages": SV, "new_row_lr_multiplier": -1.0}), vec!["dataset"], "new_row_lr_multiplier"),
            (json!({"base": "a/b", "languages": r#"{"swedish": {"new": true, "init_from": ["english"]}, "norwegian": {"new": true, "init_from": ["german"]}}"#}), vec!["dataset"], "same mix"),
        ];
        for (params, blobs, why) in cases {
            let e = validated(params.clone(), &blobs).unwrap_err();
            assert!(e.contains(why), "{params}: expected {why:?}, got {e}");
        }
    }
}
