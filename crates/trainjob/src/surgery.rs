// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained speech
// synthesizers into smaller models that still load in their upstream tooling,
// as a service on the accelerators that serve them, for its clients. If your
// team needs expertise in compressing text-to-speech models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The `speech_surgery` action: a Qwen3-TTS checkpoint made physically
//! smaller by a weight-surgery plan, chosen by calibration on a speech shard
//! and measured on a held-out one ([`brain::SpeechSurgery`]), published like
//! every checkpoint of this service in its base's upstream layout
//! (`checkpoint_digest`, optionally installed as `local/<install_as>`). With
//! `profile` every edit of the plan is measured alone on the base and no
//! checkpoint is written: the sensitivity profile a compression schedule is
//! planned from.

use std::path::Path;

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::consolidate::{check_install_name, checkpoint_outcome_of};
use crate::job::{self, bring, JobEnv, Workdir};
use crate::request::Source;
use crate::store::Digest;

/// The action's name.
pub const ACTION: &str = "speech_surgery";
/// The architecture a base must be.
const ARCHITECTURE: &str = "Qwen3TTSForConditionalGeneration";
/// The largest plan accepted.
const MAX_PLAN: usize = 64 << 10;
/// The largest corpus or probe text accepted.
const MAX_TEXT: usize = 256 << 20;
/// The most records calibration or measurement may take.
const MAX_RECORDS: f64 = 100_000.0;
/// The working set beside the calibration build (four f32 copies of the checkpoint, for the curvature's gradients).
const WORKSPACE_BYTES: u64 = 8 << 30;
const BYTES_PER_PARAM: u64 = 16;

/// What `speech_surgery` is.
pub fn spec() -> ActionSpec {
    let int = |name: &str, help: &str, max: f64| ParamSpec::new(name, ParamType::Int, help).min(0.0).max(max).step(1.0);
    let blob = |name: &str, help: &str| BlobSpec::new(name, Media::Bytes, help);
    ActionSpec::new(ACTION, "make a Qwen3-TTS checkpoint physically smaller by a weight-surgery plan (layers, MLP channels, KV groups, residual width, text path, text vocabulary), each edit chosen by calibration on a speech shard and measured (loss and KL from the unedited predictions) on a held-out one; the narrower checkpoint leaves in its base's upstream layout by manifest digest (and optionally into the model store); `profile` measures every edit alone and writes none")
        .training()
        .cancellable()
        .param(ParamSpec::new("base", ParamType::Str, "the Qwen3-TTS checkpoint, `<vendor>/<repo>` in the server's model store").required())
        .param(ParamSpec::new("plan", ParamType::Str, "JSON array of edits: {\"trim_vocabulary\": {\"vocabulary\": \"text\"}}, {\"fold_embedding\": {\"fold\": \"talker.text\", \"keep\": N, \"score\": S}}, {\"select_channels\": {\"dims\": \"talker.layers.*.mlp\" | \"*.kv_heads\", \"keep\": N, \"score\": S}}, {\"rotate_stream\": {\"stream\": \"talker\" | \"code_predictor\", \"width\": N, \"basis\": \"pca\" | \"identity\"}}, {\"remove_layers\": {\"stack\": \"talker.layers\" | \"code_predictor.layers\", \"count\": N, \"by\": \"kl\" | \"loss\" | \"cosine\"} or {\"layers\": [..]}; with \"greedy\": true (kl or loss) the layers are chosen one at a time, each measured with those already chosen removed}; S is act_rms, wanda, fisher or taylor").required())
        .param(ParamSpec::new("dataset_digest", ParamType::Str, "the calibration shard (`brain-tts-shard-v1`), as a digest of the server's artifact store (or the input blob `dataset`; exactly one)"))
        .param(ParamSpec::new("held_out_digest", ParamType::Str, "the shard the edits are measured on, as a digest (or the input blob `held_out`; default: the calibration shard)"))
        .param(int("records", "calibration records, spread over the shard (0: all)", MAX_RECORDS).implied_default(json!(64)))
        .param(int("held_out_records", "held-out records measured on (0: all)", MAX_RECORDS).implied_default(json!(64)))
        .param(int("batch_utterances", "records per calibration batch", 1_024.0).min(1.0).implied_default(json!(8)))
        .param(int("sample_rows", "rows per residual point and batch the stream and activation statistics take", 1_000_000.0).min(1.0).implied_default(json!(512)))
        .param(ParamSpec::new("shard_tokenizer", ParamType::Str, "the release whose tokenizer wrote the shards, `<vendor>/<repo>` in the server's model store, when the base was cut from it with a trimmed vocabulary: the shards' text ids are read through its tokens"))
        .param(ParamSpec::new("reference", ParamType::Str, "measure every edit against this release, `<vendor>/<repo>` in the server's model store (the model the base was itself cut from), instead of the unedited base; the baseline is the base's own distance to it"))
        .param(ParamSpec::new("profile", ParamType::Bool, "measure every edit alone on the base and publish no checkpoint").default(json!(false)))
        .param(ParamSpec::new("exact", ParamType::Bool, "exact fp32 products instead of tensor cores").default(json!(false)))
        .param(ParamSpec::new("install_as", ParamType::Str, "also install the checkpoint in the server's model store as `local/<this name>` (letters, digits, `.`, `_`, `-`; it must not exist yet)"))
        .param(crate::consolidate::allow_cpu_param())
        .input(blob("dataset", "the calibration shard"))
        .input(blob("held_out", "the shard the edits are measured on"))
        .input(blob("corpus", "UTF-8 text, one utterance per line, whose tokens a vocabulary trim keeps beside the shards' own"))
        .input(blob("probe", "UTF-8 text, one utterance per line, whose encodings the trim's report compares before and after"))
}

/// The settings of `speech_surgery`, exactly the params its spec declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub base: String,
    pub plan: String,
    pub dataset_digest: Option<String>,
    pub held_out_digest: Option<String>,
    pub records: Option<usize>,
    pub held_out_records: Option<usize>,
    pub batch_utterances: Option<usize>,
    pub sample_rows: Option<usize>,
    pub shard_tokenizer: Option<String>,
    pub reference: Option<String>,
    pub profile: bool,
    pub exact: bool,
    pub install_as: Option<String>,
    pub allow_cpu: bool,
}

/// A `speech_surgery` request that passed every check that needs no file and no device.
#[derive(Debug)]
pub struct Request {
    pub params: Params,
    pub plan: Vec<brain::surgery::Edit>,
    pub dataset: Source,
    pub held_out: Option<Source>,
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
        if params.plan.len() > MAX_PLAN {
            return Err(format!("{ACTION}: the plan is {} bytes, more than {MAX_PLAN}", params.plan.len()));
        }
        let plan: Vec<brain::surgery::Edit> = serde_json::from_str(&params.plan).map_err(|e| format!("{ACTION}: plan: {e}"))?;
        if plan.is_empty() {
            return Err(format!("{ACTION}: the plan names no edit"));
        }
        let dataset = source(inv, "dataset", params.dataset_digest.as_deref(), "dataset_digest")?.ok_or_else(|| format!("{ACTION}: name the calibration shard: the input blob `dataset` or `dataset_digest`"))?;
        let held_out = source(inv, "held_out", params.held_out_digest.as_deref(), "held_out_digest")?;
        if let Some(extra) = inv.blobs.keys().find(|k| !["dataset", "held_out", "corpus", "probe"].contains(&k.as_str())) {
            return Err(format!("{ACTION}: the input blob \"{extra}\" is not an input of this action"));
        }
        for name in ["corpus", "probe"] {
            if inv.blobs.get(name).is_some_and(|b| b.bytes.len() > MAX_TEXT) {
                return Err(format!("{ACTION}: {name} is more than {MAX_TEXT} bytes"));
            }
        }
        if inv.blobs.contains_key("corpus") && !plan.iter().any(|e| matches!(e, brain::surgery::Edit::TrimVocabulary { .. })) {
            return Err(format!("{ACTION}: a corpus only serves a trim_vocabulary edit, and the plan has none"));
        }
        if params.profile && params.install_as.is_some() {
            return Err(format!("{ACTION}: a profile publishes no checkpoint to install"));
        }
        if let Some(name) = &params.install_as {
            check_install_name(name)?;
        }
        Ok(Request { params, plan, dataset, held_out })
    }

    /// Every digest the call names.
    pub fn digests(&self) -> Vec<Digest> {
        std::iter::once(&self.dataset).chain(&self.held_out).filter_map(|s| if let Source::Digest(d) = s { Some(*d) } else { None }).collect()
    }
}

/// The base of `request` in the model store, checked to be a Qwen3-TTS checkpoint.
pub(crate) fn resolve_base(env: &JobEnv, request: &Request) -> Result<std::path::PathBuf, String> {
    let dir = job::resolve_base(env.models_dir.as_deref(), &request.params.base)?;
    let config: Value = std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).ok_or_else(|| format!("{ACTION}: base \"{}\" is not a Qwen3-TTS checkpoint (it has no readable config.json)", request.params.base))?;
    if !config["architectures"].as_array().is_some_and(|a| a.iter().any(|x| x == ARCHITECTURE)) {
        return Err(format!("{ACTION}: base \"{}\" is not a Qwen3-TTS checkpoint", request.params.base));
    }
    Ok(dir)
}

/// What a run is charged on its card: the calibration build (weights, gradients and two moments in f32) and the
/// workspace.
pub(crate) fn device_bytes(base_dir: &Path) -> u64 {
    let params = std::fs::metadata(base_dir.join("model.safetensors")).map_or(0, |m| m.len() / 2);
    params.saturating_mul(BYTES_PER_PARAM).saturating_add(WORKSPACE_BYTES)
}

/// The UTF-8 lines of an input blob that hold something.
fn lines(inv: &Invocation, name: &str) -> Result<Vec<String>, String> {
    let Some(blob) = inv.blobs.get(name) else { return Ok(Vec::new()) };
    let text = std::str::from_utf8(&blob.bytes).map_err(|_| format!("{ACTION}: {name} is not UTF-8 text"))?;
    Ok(text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
}

/// Runs the `speech_surgery` job `inv`.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let request = Request::from_invocation(inv)?;
    let p = &request.params;
    let base_dir = resolve_base(env, &request)?;
    let work = Workdir::create(env)?;
    let stage = |source: &Source, name: &str| -> Result<std::path::PathBuf, String> {
        let path = work.path().join(name);
        bring(source, &inv.blobs, env.store.as_ref(), work.staging(), &path)?;
        Ok(path)
    };
    let dataset = stage(&request.dataset, "dataset")?;
    let held_out = request.held_out.as_ref().map(|s| stage(s, "held_out")).transpose()?;
    let id = p.install_as.as_ref().map_or_else(|| "local/qwen3tts-surgery".to_string(), |name| format!("local/{name}"));
    let mut surgery = brain::SpeechSurgery::new(&base_dir, &dataset).plan(request.plan.clone()).profile(p.profile).exact(p.exact).corpus(lines(inv, "corpus")?).probe(lines(inv, "probe")?).out_dir(work.path().join("surgery")).id(id);
    if let Some(h) = &held_out {
        surgery = surgery.held_out(h);
    }
    if let Some(id) = &p.shard_tokenizer {
        surgery = surgery.shard_tokenizer(job::resolve_base(env.models_dir.as_deref(), id)?);
    }
    if let Some(id) = &p.reference {
        surgery = surgery.reference(job::resolve_base(env.models_dir.as_deref(), id)?);
    }
    for (set, v) in [(brain::SpeechSurgery::records as fn(_, _) -> _, p.records), (brain::SpeechSurgery::held_out_records, p.held_out_records), (brain::SpeechSurgery::batch, p.batch_utterances), (brain::SpeechSurgery::sample_rows, p.sample_rows)] {
        if let Some(n) = v {
            surgery = set(surgery, n);
        }
    }
    let mut stages = 0;
    let outcome = surgery
        .run_with(|stage| {
            stages += 1;
            progress(Progress::event(stages, 0, json!({"type": "stage", "stage": stage})));
        })
        .map_err(|e| format!("{ACTION}: {e}"))?;
    inv.cancel.check()?;
    let mut report = serde_json::to_value(&outcome).map_err(|e| format!("{ACTION}: {e}"))?;
    report["status"] = json!("completed");
    report["objective"] = json!(ACTION);
    report["base"] = json!(p.base);
    report["manifest"] = serde_json::from_slice(&std::fs::read(&outcome.manifest).map_err(|e| format!("{ACTION}: {e}"))?).map_err(|e| format!("{ACTION}: {e}"))?;
    match &outcome.checkpoint {
        None => {
            let mut out = Outcome::new();
            out.outputs = report;
            Ok(out)
        }
        Some(dir) => {
            work.ensure_within_quota()?;
            checkpoint_outcome_of(env, dir, p.install_as.as_deref(), report, None)
        }
    }
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

    const PLAN: &str = r#"[{"remove_layers": {"stack": "talker.layers", "count": 4, "by": "kl"}}]"#;

    #[test]
    fn a_request_names_its_plan_and_shards_and_refuses_what_it_cannot_take() {
        let ok = validated(json!({"base": "a/b", "plan": PLAN, "profile": true}), &["dataset"]).unwrap();
        assert_eq!((ok.plan.len(), ok.params.profile, ok.dataset.clone()), (1, true, Source::Blob("dataset".into())));
        let cases: Vec<(Value, Vec<&str>, &str)> = vec![
            (json!({"base": "a/b", "plan": PLAN}), vec![], "calibration shard"),
            (json!({"base": "a/b", "plan": "[]"}), vec!["dataset"], "no edit"),
            (json!({"base": "a/b", "plan": r#"[{"remove_layers": {"stack": "x", "every": 2}}]"#}), vec!["dataset"], "plan"),
            (json!({"base": "a/b", "plan": PLAN}), vec!["dataset", "corpus"], "corpus only serves"),
            (json!({"base": "a/b", "plan": PLAN, "profile": true, "install_as": "x"}), vec!["dataset"], "profile"),
            (json!({"base": "a/b", "plan": PLAN, "install_as": "../x"}), vec!["dataset"], "install_as"),
            (json!({"base": "a/b", "plan": PLAN}), vec!["dataset", "weights"], "unknown input"),
            (json!({"base": "a/b", "plan": PLAN, "out": "x"}), vec!["dataset"], "unknown param"),
        ];
        for (params, blobs, why) in cases {
            let e = validated(params.clone(), &blobs).unwrap_err();
            assert!(e.contains(why), "{params}: expected {why:?}, got {e}");
        }
    }
}
