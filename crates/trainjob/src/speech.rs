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
//! ([`objective::speech_sft`]) under the `resume_key` contract the other jobs share ([`crate::runs`]), and the trained
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
/// two AdamW moments): activations of a step of utterances.
const ACTIVATION_BYTES: u64 = 2 << 30;

/// The trainer this action drives, `qwen3tts::speech_sft`'s interface. Until that objective is in the tree, this
/// stand-in keeps the base's tensors and only allocates rows for the new names, so every layer around it (staging,
/// resumption, export, publication, installation, serving) runs for real.
pub mod objective {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use qwen3tts::shard::Shard;

    /// How a language is trained: an existing one of the base, or a new one initialised from a mix of existing ones.
    #[derive(Clone, Debug, PartialEq)]
    pub struct LanguageInit {
        pub new: bool,
        pub init_from: Vec<String>,
    }

    /// The run's settings.
    #[derive(Clone, Debug)]
    pub struct SpeechSftConfig {
        /// The base checkpoint directory, as downloaded.
        pub base: PathBuf,
        pub steps: u32,
        pub lr: Option<f32>,
        pub grad_accum: u32,
        pub seed: u64,
        pub eval_every: Option<u32>,
        pub residual_weight: f32,
        pub languages: BTreeMap<String, LanguageInit>,
        /// Voice name -> whether it is new to the base.
        pub voices: BTreeMap<String, bool>,
    }

    /// What a run hands back: the trained tensors under their upstream names, the codec row of every language and
    /// voice the checkpoint names, and the run's report.
    pub struct SpeechSftResult {
        pub tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
        pub languages: BTreeMap<String, u32>,
        pub voices: BTreeMap<String, u32>,
        pub report: serde_json::Value,
    }

    /// The stand-in: no step is taken and the base's tensors are kept; each new language and voice gets the highest
    /// codec row no language, voice or special token of the base holds.
    pub fn speech_sft(cfg: &SpeechSftConfig, train: &Shard, _held_out: Option<&Shard>, control: model::train::FitControl<'_>) -> Result<SpeechSftResult, String> {
        let base = cfg.base.to_str().ok_or("the base directory is not UTF-8")?;
        let specials = qwen3tts::TtsSpecials::from_config_dir(base)?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(cfg.base.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let vocab = config["talker_config"]["vocab_size"].as_u64().ok_or("config.json has no talker_config.vocab_size")? as u32;
        let mut taken: std::collections::BTreeSet<u32> = specials.lang.values().chain(specials.spk_id.values()).copied().collect();
        taken.extend([specials.codec_nothink, specials.codec_think, specials.codec_think_bos, specials.codec_think_eos, specials.codec_pad, specials.codec_bos, specials.codec_eos]);
        let mut free = (0..vocab).rev().filter(|r| !taken.contains(r)).collect::<Vec<_>>().into_iter();
        let mut take = || free.next().ok_or("the codec vocabulary has no free row left");
        let mut languages = BTreeMap::new();
        for (name, init) in &cfg.languages {
            let row = if init.new { take()? } else { *specials.lang.get(name).ok_or_else(|| format!("language {name:?} is not the base's"))? };
            languages.insert(name.clone(), row);
        }
        let mut voices = BTreeMap::new();
        for (name, new) in &cfg.voices {
            let row = if *new { take()? } else { *specials.spk_id.get(name).ok_or_else(|| format!("voice {name:?} is not the base's"))? };
            voices.insert(name.clone(), row);
        }
        drop(control);
        Ok(SpeechSftResult { tensors: Vec::new(), languages, voices, report: serde_json::json!({"trainer": "stand-in: the base's tensors, unchanged", "records": train.len(), "steps_completed": 0}) })
    }
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
        .param(ParamSpec::new("languages", ParamType::Str, "JSON object naming every language the shards hold: name -> {\"new\": true, \"init_from\"?: [existing names]} for one the base lacks (its codec row initialised from the mix of those), or {\"new\": false} for one of the base's; names are lowercase [a-z0-9_-]").required())
        .param(ParamSpec::new("voices", ParamType::Str, "JSON object naming every voice the shards train: name -> {\"new\": true} for a new voice (its row initialised from its speaker's x-vectors), or {\"new\": false} for one of the base's; a checkpoint with named voices is exported as `custom_voice`").default(json!("{}")))
        .param(ParamSpec::new("steps", ParamType::Int, "optimiser steps").default(json!(100)).min(1.0).max(capability::bounds::TRAIN_STEPS).step(1.0))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate (default: the objective's)").min(0.0).max(1.0))
        .param(ParamSpec::new("grad_accum", ParamType::Int, "utterances averaged into one optimiser step").default(json!(1)).min(1.0).max(4096.0).step(1.0))
        .param(ParamSpec::new("eval_every", ParamType::Int, "score the held-out shard every this many steps (needs one)").min(1.0).max(capability::bounds::TRAIN_COUNT).step(1.0))
        .param(ParamSpec::new("residual_weight", ParamType::Float, "the weight of the code predictor's residual-codebook loss beside the Talker's (upstream trains with 0.3)").default(json!(0.3)).min(0.0).max(10.0))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the batch order and of new rows (random when omitted; the report carries the one used)").min(0.0).max(capability::bounds::EXACT_INTEGER))
        .param(ParamSpec::new("cards", ParamType::Int, "cards the run trains on; a speech fine-tune trains on one").default(json!(1)).min(1.0).max(1.0).step(1.0))
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
    pub steps: u32,
    pub lr: Option<f32>,
    pub grad_accum: u32,
    pub eval_every: Option<u32>,
    pub residual_weight: f32,
    pub seed: Option<u64>,
    pub cards: u32,
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
    pub languages: BTreeMap<String, objective::LanguageInit>,
    pub voices: BTreeMap<String, bool>,
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
            languages.insert(name, objective::LanguageInit { new: doc.new, init_from: doc.init_from });
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
        Ok(Request { params, dataset, held_out, languages, voices })
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

/// What a run is charged on its card: the Talker's and the code predictor's parameters at
/// [`BYTES_PER_TRAINED_PARAM`] (the codec and the speaker encoder are not trained) and a step's activations.
pub(crate) fn device_bytes(base_dir: &Path) -> u64 {
    let paths = qwen3tts::TtsPaths::new(base_dir, base_dir.to_string_lossy());
    paths.talker_and_mtp_params().saturating_mul(BYTES_PER_TRAINED_PARAM).saturating_add(ACTIVATION_BYTES)
}

/// The staged shards, each opened and checked against the base and the request: the codebooks and x-vector width of
/// the base, every language the shard holds named in `languages`, every voice in `voices`.
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
    if shard.is_empty() {
        return Err(format!("{ACTION}: {what} holds no record"));
    }
    Ok(shard)
}

/// A step's frame: what the objective reported for it.
fn step_frame(step: &model::train::StepReport, seconds: f64) -> Value {
    let mut frame = json!({"type": "step", "step": step.step, "steps": step.steps, "loss": step.loss, "lr": step.lr, "grad_norm": step.grad_norm, "seconds": seconds});
    if let Some(eval) = step.eval_loss {
        frame["eval_loss"] = json!(eval);
    }
    for (name, value) in &step.metrics {
        frame[*name] = json!(value);
    }
    frame
}

/// Runs the `speech_sft` job `inv` on the device this thread is scoped to.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let request = Request::from_invocation(inv)?;
    let p = &request.params;
    let base_dir = resolve_base(env, &request)?;
    let work = Workdir::create(env)?;
    progress(Progress::event(0, p.steps, json!({"type": "stage", "stage": "staging the shards"})));
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
            "status": "valid", "objective": ACTION, "base": p.base, "steps": p.steps, "grad_accum": p.grad_accum,
            "records": train.len(), "held_out_records": held_out.as_ref().map(qwen3tts::shard::Shard::len),
            "languages": request.languages.keys().collect::<Vec<_>>(), "voices": request.voices.keys().collect::<Vec<_>>(),
            "device_bytes": device_bytes(&base_dir),
        });
        return Ok(outcome);
    }
    let run = match open_run(env, ACTION, p.resume_key.as_deref(), inv, p.seed, &crate::runs::Layout::single())? {
        Some(crate::runs::Opened::Completed(mut result)) => {
            result["resume"] = crate::runs::completed_report(p.resume_key.as_deref().unwrap_or_default());
            let mut outcome = Outcome::new();
            outcome.outputs = result;
            return Ok(outcome);
        }
        Some(crate::runs::Opened::Run(run)) => Some(run),
        None => None,
    };
    let seed = run.as_ref().map(crate::runs::Run::seed).or(p.seed).unwrap_or_else(brain::random_seed);
    let cfg = objective::SpeechSftConfig {
        base: base_dir.clone(),
        steps: p.steps,
        lr: p.lr,
        grad_accum: p.grad_accum,
        seed,
        eval_every: p.eval_every,
        residual_weight: p.residual_weight,
        languages: request.languages.clone(),
        voices: request.voices.clone(),
    };
    progress(Progress::event(0, p.steps, json!({"type": "stage", "stage": "loading the base and training"})));
    let began = Instant::now();
    let mut on_step = |step: &model::train::StepReport| {
        progress(Progress::event(step.step, step.steps, step_frame(step, began.elapsed().as_secs_f64())));
        !inv.cancel.is_cancelled()
    };
    let state = run.as_ref().map(crate::runs::Run::state_path);
    let identity = json!({"base": p.base, "dataset": request.digests().iter().map(Digest::hex).collect::<Vec<_>>()});
    let control = model::train::FitControl { on_step: Some(&mut on_step), state: state.as_deref(), state_every: p.checkpoint_every.unwrap_or(crate::runs::DEFAULT_CHECKPOINT_EVERY), identity, snapshots: None, grad_sync: None };
    let trained = objective::speech_sft(&cfg, &train, held_out.as_ref(), control).map_err(|e| format!("{ACTION}: {e}"))?;
    inv.cancel.check()?;

    let id = p.install_as.as_ref().map_or_else(|| DEFAULT_ID.to_string(), |name| format!("local/{name}"));
    let out = work.path().join("release");
    let exported = qwen3tts::export::export(&base_dir, &out, &id, &qwen3tts::export::Trained { tensors: &trained.tensors, languages: &trained.languages, voices: &trained.voices }, &mut || !inv.cancel.is_cancelled())
        .map_err(|e| if e == capability::failure::CANCELLED { e } else { format!("{ACTION}: {e}") })?;
    let mut report = trained.report;
    report["status"] = json!("completed");
    report["objective"] = json!(ACTION);
    report["base"] = json!(p.base);
    report["seed"] = json!(seed);
    report["languages"] = json!(trained.languages);
    report["voices"] = json!(trained.voices);
    report["tts_model_type"] = json!(match exported.kind {
        qwen3tts::prompt::CheckpointKind::Base => "base",
        qwen3tts::prompt::CheckpointKind::CustomVoice => "custom_voice",
        qwen3tts::prompt::CheckpointKind::VoiceDesign => "voice_design",
    });
    report["trained_tensors"] = json!(trained.tensors.len());
    work.ensure_within_quota()?;
    checkpoint_outcome_of(env, &out, p.install_as.as_deref(), report, run.map(|r| (r, None)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validated(params: Value, blobs: &[&str]) -> Result<Request, String> {
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
        let step = model::train::StepReport { step: 3, steps: 9, loss: 1.5, lr: 1e-4, grad_norm: Some(0.5), work: Default::default(), eval_loss: Some(2.0), metrics: vec![("residual_loss", 0.25)] };
        let frame = step_frame(&step, 7.0);
        assert_eq!(frame, json!({"type": "step", "step": 3, "steps": 9, "loss": 1.5, "lr": 1e-4_f32, "grad_norm": 0.5, "seconds": 7.0, "eval_loss": 2.0, "residual_loss": 0.25}));
    }

    /// A cancelled job stops before it reads anything.
    #[test]
    fn a_cancelled_job_stops_before_any_work() {
        let env = JobEnv { store: None, models_dir: None, work_root: std::env::temp_dir(), quotas: Default::default(), job_access: None, largest_card_bytes: None };
        let mut inv = Invocation::new();
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        assert_eq!(run(&env, &inv, &mut |_| {}).unwrap_err(), capability::failure::CANCELLED);
    }

    #[test]
    fn a_request_names_its_shards_languages_and_voices_and_refuses_what_it_cannot_take() {
        let ok = validated(json!({"base": "a/b", "languages": SV, "voices": r#"{"astrid": {"new": true}}"#, "install_as": "sv-1"}), &["dataset"]).unwrap();
        assert_eq!(ok.dataset, Source::Blob("dataset".into()));
        assert_eq!(ok.languages["swedish"], objective::LanguageInit { new: true, init_from: vec!["english".into(), "german".into()] });
        assert_eq!((ok.voices["astrid"], ok.params.steps, ok.params.residual_weight), (true, 100, 0.3));
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
            (json!({"base": "a/b", "languages": SV, "cards": 2}), vec!["dataset"], "cards"),
            (json!({"base": "a/b", "languages": SV, "out": "x"}), vec!["dataset"], "unknown param"),
            (json!({"base": "a/b", "languages": SV, "steps": 0}), vec!["dataset"], "steps"),
        ];
        for (params, blobs, why) in cases {
            let e = validated(params.clone(), &blobs).unwrap_err();
            assert!(e.contains(why), "{params}: expected {why:?}, got {e}");
        }
    }
}
