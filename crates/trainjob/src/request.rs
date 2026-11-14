// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that accept a fine-tune as
// a typed, bounded request and refuse everything else at the boundary, for its
// clients. If your team needs expertise in exposing model training as a
// service without exposing the host, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The `train` action's contract: its [`ActionSpec`], and the typed request
//! built from an invocation that passed it.
//!
//! The scalar settings are ordinary params, bounded by the spec (a caller sees
//! and is held to the ranges). Where the data comes from is a document in the
//! `data` param, parsed into [`DataRefs`] with unknown fields refused: each
//! dataset is either an input blob of the call (a file descriptor over D-Bus)
//! or a digest in the server's artifact store. A caller never names a path.

use capability::{ActionSpec, BlobSpec, Invocation, Media, ParamSpec, ParamType};
use serde::Deserialize;
use serde_json::json;

use crate::store::Digest;

/// The model id the training workload is registered under.
pub const MODEL: &str = "brain/train";
/// The action's name.
pub const ACTION: &str = "train";

/// The most replay datasets (`replay` plus `weighted_replay`) a run mixes in.
pub const MAX_REPLAY_SETS: usize = 8;
/// The input blobs of the action, in the order a replay set takes them.
pub const REPLAY_BLOBS: [&str; MAX_REPLAY_SETS] = ["replay_0", "replay_1", "replay_2", "replay_3", "replay_4", "replay_5", "replay_6", "replay_7"];
/// The longest `data` document accepted, in bytes.
pub const MAX_DATA_DOCUMENT_BYTES: usize = 64 * 1024;
/// The most evaluations (`steps / eval_every`) one run takes, so its report, which lists them, is bounded.
pub const MAX_EVALUATIONS: u32 = 1024;
/// The longest adapter id accepted.
pub const MAX_ADAPTER_ID_LEN: usize = 200;

/// What the training action is.
pub fn action_spec() -> ActionSpec {
    let optional_f = |name: &str, help: &str, min: f64, max: f64| ParamSpec::new(name, ParamType::Float, help).min(min).max(max);
    let blob = |name: &str, help: &str| BlobSpec::new(name, Media::Bytes, help);
    let mut spec = ActionSpec::new(ACTION, "fine-tune a chat model's LoRA adapter on a chat dataset, by supervised imitation, by the clipped policy gradient on rollouts, or by direct preference optimization on chosen/rejected pairs (datasets in by blob or artifact digest, the adapter out by digest or blob)")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("base", ParamType::Str, "the base model, `<vendor>/<repo>` in the server's model store").required())
        .param(ParamSpec::new("data", ParamType::Str, "JSON document naming the datasets: {\"train\": SRC, \"held_out\"?, \"held_out_text\"?, \"monitor\"?, \"start_adapter\"?, \"replay\"?: [SRC], \"weighted_replay\"?: [{SRC fields, \"share\"}]} where SRC is {\"digest\": \"<blake3 hex>\"} (an artifact of the server's store) or {\"blob\": \"<input name>\"} (an input blob of this call)").required())
        .param(ParamSpec::new("objective", ParamType::Enum(vec!["sft".into(), "policy_gradient".into(), "dpo".into()]), "supervised imitation; the clipped policy gradient on rollout records that carry an advantage or a teacher context; or direct preference optimization on `generic-preference-v1` pairs (`dataset` and `held_out` hold pairs)").default(json!("sft")))
        .param(ParamSpec::new("steps", ParamType::Int, "optimiser steps").default(json!(100)).min(1.0).max(1_000_000.0).step(1.0))
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank of a fresh adapter; a continued adapter keeps its own (default 8)").min(1.0).max(256.0).step(1.0))
        .param(optional_f("alpha", "LoRA alpha (default 2 x rank)", 0.0, 1024.0))
        .param(optional_f("lr", "peak learning rate", 0.0, 1.0))
        .param(optional_f("weight_decay", "AdamW weight decay on the adapter", 0.0, 1.0))
        .param(ParamSpec::new("grad_accum", ParamType::Int, "records averaged into one optimiser step").default(json!(1)).min(1.0).max(4096.0).step(1.0))
        .param(ParamSpec::new("strict_updates", ParamType::Bool, "abort at the first optimiser step whose loss or gradient is not finite (off: such a step is skipped, and the run aborts on the third in a row); the adapter is never updated with a non-finite value").default(json!(false)))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of a fresh adapter's initialisation and the batch order (random when omitted; the report carries the one used)").min(0.0))
        .param(ParamSpec::new("max_block", ParamType::Int, "the longest training row in tokens: the memory reserved for the run is sized to it, and a record longer than it is refused by name").default(json!(8192)).min(64.0).max(1_048_576.0).step(1.0))
        .param(ParamSpec::new("thinking", ParamType::Bool, "train the model to reason before it answers (off: trained for no-think mode); sft and policy_gradient only").default(json!(true)))
        .param(ParamSpec::new("pack_answers", ParamType::Bool, "train a record that supervises several turns as one sequence; sft and policy_gradient only").default(json!(false)))
        .param(ParamSpec::new("arch", ParamType::Str, "the base's architecture (`qwen3`, `qwen35`) when its directory carries no config.json to tell it by; a Hugging Face checkpoint needs none"))
        .param(ParamSpec::new("base_format", ParamType::Str, "how the frozen base is held: f32, bf16, int8, nf4, fp8, ...; refused by name when the architecture's trainer does not hold it (default: the tier that fits the card)"))
        .param(ParamSpec::new("eval_every", ParamType::Int, "steps between evaluations of the monitoring and held-out records (0: none)").default(json!(0)).min(0.0).max(1_000_000.0).step(1.0))
        .param(ParamSpec::new("patience", ParamType::Int, "evaluations without improvement before the run stops (0: never)").default(json!(0)).min(0.0).max(1_000_000.0).step(1.0))
        .param(optional_f("replay_share", "the share of draws that come from the replay sets in all, in (0, 1)", 0.0, 1.0))
        .param(ParamSpec::new("adapter_id", ParamType::Str, "the id written on the adapter's card"))
        .param(ParamSpec::new("keep_evaluations", ParamType::Bool, "also publish the adapter of every evaluation (needs eval_every > 0)").default(json!(false)))
        .param(ParamSpec::new("return_adapter", ParamType::Bool, "also return the adapter as an output blob (a file descriptor over D-Bus); over HTTP take it by digest from the artifact store").default(json!(false)))
        .param(optional_f("clip_eps", "policy gradient: half-width of the clip band of the importance ratio", 0.0, 1.0))
        .param(optional_f("clip_eps_high", "policy gradient: the upper edge of the clip band is 1 + this (clip-higher), at least clip_eps; omitted, the band is symmetric", 0.0, 10.0))
        .param(ParamSpec::new("importance", ParamType::Str, "policy gradient: the correction for the sampling engine's gap to the trainer: `none`, `truncate:CAP` or `mask:CAP`; needs recorded behaviour log-probabilities"))
        .param(optional_f("max_step_kl", "policy gradient: skip (and log) an optimiser step whose mean KL to the reference is over this", 0.0, 1e9))
        .param(optional_f("max_step_grad_norm", "policy gradient: skip (and log) an optimiser step whose gradient norm is over this", 0.0, 1e12))
        .param(optional_f("kl_beta", "policy gradient: coefficient of the KL term to the reference policy", 0.0, 1000.0))
        .param(ParamSpec::new("reference", ParamType::Enum(vec!["base".into(), "start".into()]), "policy gradient: the policy the KL term is taken against"))
        .param(ParamSpec::new("behaviour", ParamType::Enum(vec!["recorded".into(), "start_of_run_proxy".into()]), "policy gradient: where the behaviour log-probabilities of the importance ratio come from: `recorded` (default) the sampler's own, required on every supervised token; `start_of_run_proxy` the start-of-run policy's where none was recorded, exact only if the rollouts were sampled from it"))
        .param(ParamSpec::new("loss_agg", ParamType::Enum(vec!["seq-mean".into(), "token-sum".into()]), "policy gradient: how the token terms are normalised"))
        .param(optional_f("temperature", "policy gradient: the temperature the rollouts were sampled at", 0.0, 100.0))
        .param(ParamSpec::new("teacher", ParamType::Enum(vec!["base".into(), "start".into()]), "policy gradient with teacher contexts: the frozen policy that scores the student's tokens in the privileged context"))
        .param(optional_f("advantage_bound", "policy gradient with teacher contexts: each token's advantage is clipped to [-bound, bound]", 0.0, 1e6))
        .param(optional_f("sequence_advantage_cap", "policy gradient with teacher contexts: the most the magnitudes of a sequence's token advantages may sum to", 0.0, 1e9))
        .param(optional_f("beta", "dpo: the temperature of the preference loss (default 0.1)", 0.0, 100.0))
        .param(optional_f("nll_weight", "dpo: the weight of the supervised likelihood of the chosen answer added to the loss (default 0)", 0.0, 1000.0))
        .param(ParamSpec::new("keep_reasoning", ParamType::Bool, "dpo: train on the reasoning of each candidate (the template drops a closed think block from an assistant turn otherwise)"))
        .input(blob("dataset", "the training records (`generic-messages-v2` JSONL; `generic-preference-v1` pairs for dpo)"))
        .input(blob("held_out", "records scored before and after, never trained on (preference pairs for dpo)"))
        .input(blob("held_out_text", "the writer's own text, scored before and after beside the held-out records"))
        .input(blob("monitor", "records scored every eval_every steps"))
        .input(blob("adapter", "the adapter to continue instead of starting a fresh one"));
    for name in REPLAY_BLOBS {
        spec = spec.input(blob(name, "a replay dataset"));
    }
    spec.output(BlobSpec::new("adapter", Media::Bytes, "the trained adapter (only when return_adapter is set)")).output(BlobSpec::new("record", Media::Text, "the training record the trainer wrote beside the adapter (JSON); published to the artifact store (record_digest) when there is one, returned when return_adapter is set or there is none"))
}

/// Where one dataset comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// An input blob of the call, by its name.
    Blob(String),
    /// An artifact of the server's store.
    Digest(Digest),
}

/// A source as it is written in the `data` document: exactly one of the two fields.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceDoc {
    digest: Option<String>,
    blob: Option<String>,
}

impl SourceDoc {
    fn resolve(self, what: &str) -> Result<Source, String> {
        match (self.digest, self.blob) {
            (Some(digest), None) => Digest::parse(&digest).map(Source::Digest).map_err(|e| format!("data.{what}.digest: {e}")),
            (None, Some(blob)) => Ok(Source::Blob(blob)),
            _ => Err(format!("data.{what}: name exactly one of \"digest\" and \"blob\"")),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightedDoc {
    digest: Option<String>,
    blob: Option<String>,
    share: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DataDoc {
    train: SourceDoc,
    held_out: Option<SourceDoc>,
    held_out_text: Option<SourceDoc>,
    monitor: Option<SourceDoc>,
    start_adapter: Option<SourceDoc>,
    #[serde(default)]
    replay: Vec<SourceDoc>,
    #[serde(default)]
    weighted_replay: Vec<WeightedDoc>,
}

/// The datasets of a run, and where each comes from.
#[derive(Clone, Debug, PartialEq)]
pub struct DataRefs {
    pub train: Source,
    pub held_out: Option<Source>,
    pub held_out_text: Option<Source>,
    pub monitor: Option<Source>,
    pub start_adapter: Option<Source>,
    pub replay: Vec<Source>,
    pub weighted_replay: Vec<(Source, f32)>,
}

impl DataRefs {
    /// Parses and checks the `data` document. `deny_unknown_fields` throughout; a replay set count past
    /// [`MAX_REPLAY_SETS`], a share outside (0, 1) and a blob that is not an input of the action are refused.
    pub fn parse(text: &str) -> Result<DataRefs, String> {
        if text.len() > MAX_DATA_DOCUMENT_BYTES {
            return Err(format!("data: the document is {} bytes; at most {MAX_DATA_DOCUMENT_BYTES} are accepted", text.len()));
        }
        let doc: DataDoc = serde_json::from_str(text).map_err(|e| format!("data: {e}"))?;
        if doc.replay.len() + doc.weighted_replay.len() > MAX_REPLAY_SETS {
            return Err(format!("data: at most {MAX_REPLAY_SETS} replay sets (replay and weighted_replay together)"));
        }
        let optional = |source: Option<SourceDoc>, what: &str| source.map(|s| s.resolve(what)).transpose();
        let refs = DataRefs {
            train: doc.train.resolve("train")?,
            held_out: optional(doc.held_out, "held_out")?,
            held_out_text: optional(doc.held_out_text, "held_out_text")?,
            monitor: optional(doc.monitor, "monitor")?,
            start_adapter: optional(doc.start_adapter, "start_adapter")?,
            replay: doc.replay.into_iter().enumerate().map(|(i, s)| s.resolve(&format!("replay[{i}]"))).collect::<Result<_, _>>()?,
            weighted_replay: doc
                .weighted_replay
                .into_iter()
                .enumerate()
                .map(|(i, w)| {
                    if !(w.share > 0.0 && w.share < 1.0) {
                        return Err(format!("data.weighted_replay[{i}].share: {} is not in (0, 1)", w.share));
                    }
                    SourceDoc { digest: w.digest, blob: w.blob }.resolve(&format!("weighted_replay[{i}]")).map(|s| (s, w.share))
                })
                .collect::<Result<_, _>>()?,
        };
        refs.check_blob_names()?;
        Ok(refs)
    }

    /// Every blob name the document refers to, in the order the run reads them.
    pub fn blob_names(&self) -> Vec<&str> {
        self.sources().into_iter().filter_map(|s| if let Source::Blob(name) = s { Some(name.as_str()) } else { None }).collect()
    }

    fn sources(&self) -> Vec<&Source> {
        [Some(&self.train), self.held_out.as_ref(), self.held_out_text.as_ref(), self.monitor.as_ref(), self.start_adapter.as_ref()]
            .into_iter()
            .flatten()
            .chain(self.replay.iter())
            .chain(self.weighted_replay.iter().map(|(s, _)| s))
            .collect()
    }

    /// Every digest the document refers to.
    pub fn digests(&self) -> Vec<Digest> {
        self.sources().into_iter().filter_map(|s| if let Source::Digest(d) = s { Some(*d) } else { None }).collect()
    }

    fn check_blob_names(&self) -> Result<(), String> {
        let known: Vec<&str> = ["dataset", "held_out", "held_out_text", "monitor", "adapter"].into_iter().chain(REPLAY_BLOBS).collect();
        let names = self.blob_names();
        for (i, name) in names.iter().enumerate() {
            if !known.contains(name) {
                return Err(format!("data: \"{name}\" is not an input of this action (inputs: {})", known.join(", ")));
            }
            if names[..i].contains(name) {
                return Err(format!("data: the input blob \"{name}\" is named twice"));
            }
        }
        Ok(())
    }
}

/// The objective the run trains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Sft,
    PolicyGradient,
    Dpo,
}

impl Objective {
    /// The name the spec and the report give it.
    pub fn name(self) -> &'static str {
        match self {
            Objective::Sft => "sft",
            Objective::PolicyGradient => "policy_gradient",
            Objective::Dpo => "dpo",
        }
    }

    /// The training rows one optimiser step holds at once: a preference step holds both candidates of a pair.
    pub fn rows_per_step(self) -> u32 {
        match self {
            Objective::Dpo => brain::PAIR_ROWS,
            Objective::Sft | Objective::PolicyGradient => 1,
        }
    }
}

/// The scalar settings, exactly the params the spec declares (the invocation was validated against it).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub base: String,
    pub data: String,
    pub objective: Objective,
    pub steps: u32,
    pub rank: Option<u32>,
    pub alpha: Option<f32>,
    pub lr: Option<f32>,
    pub weight_decay: Option<f32>,
    pub grad_accum: u32,
    pub strict_updates: bool,
    pub seed: Option<u64>,
    pub max_block: u32,
    pub thinking: bool,
    pub pack_answers: bool,
    pub arch: Option<String>,
    pub base_format: Option<String>,
    pub eval_every: u32,
    pub patience: u32,
    pub replay_share: Option<f32>,
    pub adapter_id: Option<String>,
    pub keep_evaluations: bool,
    pub return_adapter: bool,
    pub clip_eps: Option<f32>,
    pub clip_eps_high: Option<f32>,
    pub importance: Option<String>,
    pub max_step_kl: Option<f32>,
    pub max_step_grad_norm: Option<f32>,
    pub kl_beta: Option<f32>,
    pub reference: Option<String>,
    pub behaviour: Option<String>,
    pub loss_agg: Option<String>,
    pub temperature: Option<f32>,
    pub teacher: Option<String>,
    pub advantage_bound: Option<f32>,
    pub sequence_advantage_cap: Option<f32>,
    pub beta: Option<f32>,
    pub nll_weight: Option<f32>,
    pub keep_reasoning: Option<bool>,
}

/// A request that passed the spec and every check that needs no file and no device.
#[derive(Debug)]
pub struct Request {
    pub params: Params,
    pub data: DataRefs,
}

impl Request {
    /// Reads the validated invocation `inv` as a request, refusing what the spec alone cannot: settings that
    /// belong to the other objective, a `data` document that is malformed or names a blob the call does not
    /// carry (or carries one the document does not use), an adapter id that is not plain text.
    pub fn from_invocation(inv: &Invocation) -> Result<Request, String> {
        let params: Params = serde_json::from_value(inv.params.clone()).map_err(|e| format!("train: {e}"))?;
        let data = DataRefs::parse(&params.data)?;
        let refused = |what: &str| Err(format!("train: {what}"));
        let pg_named: Vec<&str> = [
            (params.clip_eps.is_some(), "clip_eps"),
            (params.clip_eps_high.is_some(), "clip_eps_high"),
            (params.importance.is_some(), "importance"),
            (params.max_step_kl.is_some(), "max_step_kl"),
            (params.max_step_grad_norm.is_some(), "max_step_grad_norm"),
            (params.kl_beta.is_some(), "kl_beta"),
            (params.reference.is_some(), "reference"),
            (params.behaviour.is_some(), "behaviour"),
            (params.loss_agg.is_some(), "loss_agg"),
            (params.temperature.is_some(), "temperature"),
            (params.teacher.is_some(), "teacher"),
            (params.advantage_bound.is_some(), "advantage_bound"),
            (params.sequence_advantage_cap.is_some(), "sequence_advantage_cap"),
        ]
        .into_iter()
        .filter_map(|(named, name)| named.then_some(name))
        .collect();
        if params.objective == Objective::Sft && !pg_named.is_empty() {
            return refused(&format!("{} configure the policy gradient; the objective is sft", pg_named.join(", ")));
        }
        let dpo_named: Vec<&str> = [(params.beta.is_some(), "beta"), (params.nll_weight.is_some(), "nll_weight"), (params.keep_reasoning.is_some(), "keep_reasoning")]
            .into_iter()
            .filter_map(|(named, name)| named.then_some(name))
            .collect();
        if params.objective != Objective::Dpo && !dpo_named.is_empty() {
            return refused(&format!("{} configure direct preference optimization; the objective is {}", dpo_named.join(", "), params.objective.name()));
        }
        if params.objective == Objective::Dpo {
            check_preference(&params, &data, &pg_named)?;
        }
        if params.keep_evaluations && params.eval_every == 0 {
            return refused("keep_evaluations needs eval_every > 0");
        }
        if params.eval_every > 0 && params.steps.div_ceil(params.eval_every) > MAX_EVALUATIONS {
            return refused(&format!("{} steps evaluated every {} is more than {MAX_EVALUATIONS} evaluations; raise eval_every", params.steps, params.eval_every));
        }
        if let Some(id) = &params.adapter_id {
            if id.is_empty() || id.len() > MAX_ADAPTER_ID_LEN || !id.chars().all(|c| c.is_ascii_alphanumeric() || "/:._-".contains(c)) {
                return refused(&format!("adapter_id must be 1 to {MAX_ADAPTER_ID_LEN} characters of letters, digits and / : . _ -"));
            }
        }
        check_attached(&data, inv)?;
        Ok(Request { params, data })
    }
}

/// Every blob `data` names is in the call, and every blob the call carries is used.
pub(crate) fn check_attached(data: &DataRefs, inv: &Invocation) -> Result<(), String> {
    let named = data.blob_names();
    for name in &named {
        if !inv.blobs.contains_key(*name) {
            return Err(format!("data: the input blob \"{name}\" is named but not attached"));
        }
    }
    if let Some(extra) = inv.blobs.keys().find(|k| !named.contains(&k.as_str())) {
        return Err(format!("data: the input blob \"{extra}\" is attached but the document does not use it"));
    }
    Ok(())
}

/// What a preference run does not take: every setting of the other objectives is refused by name rather than
/// ignored, so a caller who believes it set a replay mix or a monitor is told it did not.
fn check_preference(params: &Params, data: &DataRefs, pg_named: &[&str]) -> Result<(), String> {
    let mut unsupported: Vec<&str> = pg_named.to_vec();
    let sets = [
        (data.held_out_text.is_some(), "held_out_text"),
        (data.monitor.is_some(), "monitor"),
        (!data.replay.is_empty(), "replay"),
        (!data.weighted_replay.is_empty(), "weighted_replay"),
        (params.replay_share.is_some(), "replay_share"),
        (params.weight_decay.is_some(), "weight_decay"),
        (params.eval_every > 0, "eval_every"),
        (params.patience > 0, "patience"),
        (params.keep_evaluations, "keep_evaluations"),
        (params.pack_answers, "pack_answers"),
        (!params.thinking, "thinking"),
    ];
    unsupported.extend(sets.into_iter().filter_map(|(named, name)| named.then_some(name)));
    if unsupported.is_empty() {
        return Ok(());
    }
    Err(format!("train: {} are not settings of the dpo objective (it takes dataset, held_out and start_adapter pairs, beta, nll_weight, keep_reasoning and the optimiser settings)", unsupported.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Blob;

    fn digest(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    fn validated(params: serde_json::Value, blobs: &[&str]) -> Result<Invocation, String> {
        let mut inv = Invocation::new();
        inv.params = params;
        for name in blobs {
            inv.blobs.insert((*name).to_string(), Blob::new(Media::Bytes, b"{}".to_vec()));
        }
        action_spec().validate_served(inv)
    }

    fn request(params: serde_json::Value, blobs: &[&str]) -> Result<Request, String> {
        Request::from_invocation(&validated(params, blobs)?)
    }

    #[test]
    fn a_minimal_request_takes_the_specs_defaults() {
        let r = request(json!({"base": "Qwen/Qwen3.8-27B-FP8", "data": json!({"train": {"digest": digest(1)}}).to_string()}), &[]).unwrap();
        assert_eq!((r.params.objective, r.params.steps, r.params.max_block, r.params.grad_accum, r.params.thinking), (Objective::Sft, 100, 8192, 1, true));
        assert_eq!((r.params.rank, r.params.lr, r.params.seed), (None, None, None), "what the caller does not name stays the trainer's own default");
        assert_eq!(r.data.digests().len(), 1);
        assert!(!r.params.strict_updates, "a non-finite step is skipped by default");
        let strict = request(json!({"base": "Qwen/Qwen3.8-27B-FP8", "data": json!({"train": {"digest": digest(1)}}).to_string(), "strict_updates": true}), &[]).unwrap();
        assert!(strict.params.strict_updates);
    }

    #[test]
    fn a_dataset_is_a_blob_of_the_call_or_a_digest_never_a_path() {
        let ok = json!({"base": "a/b", "data": json!({"train": {"blob": "dataset"}, "held_out": {"digest": digest(2)}}).to_string()});
        let r = request(ok, &["dataset"]).unwrap();
        assert_eq!(r.data.train, Source::Blob("dataset".into()));
        for (doc, why) in [
            (json!({"train": {"path": "/etc/passwd"}}), "a path"),
            (json!({"train": "/etc/passwd"}), "a bare string"),
            (json!({"train": {"digest": "../../etc/passwd"}}), "a path-like digest"),
            (json!({"train": {"digest": digest(1).to_uppercase()}}), "an upper-case digest"),
            (json!({"train": {"digest": digest(1), "blob": "dataset"}}), "both"),
            (json!({"train": {}}), "neither"),
            (json!({"train": {"digest": digest(1)}, "extra": 1}), "an unknown field"),
            (json!({"held_out": {"digest": digest(1)}}), "no training set"),
            (json!({"train": {"blob": "../dataset"}}), "a blob that is not an input"),
        ] {
            assert!(request(json!({"base": "a/b", "data": doc.to_string()}), &["dataset"]).is_err(), "{why} must be refused");
        }
    }

    #[test]
    fn the_blobs_attached_and_the_blobs_named_must_agree() {
        let doc = json!({"train": {"blob": "dataset"}}).to_string();
        let missing = request(json!({"base": "a/b", "data": doc}), &[]).unwrap_err();
        assert!(missing.contains("named but not attached"), "{missing}");
        let unused = request(json!({"base": "a/b", "data": doc}), &["dataset", "monitor"]).unwrap_err();
        assert!(unused.contains("\"monitor\"") && unused.contains("does not use"), "{unused}");
        let twice = json!({"train": {"blob": "dataset"}, "monitor": {"blob": "dataset"}}).to_string();
        assert!(request(json!({"base": "a/b", "data": twice}), &["dataset"]).unwrap_err().contains("twice"));
    }

    #[test]
    fn replay_sets_are_bounded_and_their_shares_are_proper_fractions() {
        let many: Vec<_> = (0..=MAX_REPLAY_SETS).map(|i| json!({"digest": digest(i as u8)})).collect();
        let doc = json!({"train": {"digest": digest(1)}, "replay": many}).to_string();
        assert!(request(json!({"base": "a/b", "data": doc}), &[]).unwrap_err().contains("at most 8 replay sets"));
        for share in [0.0, 1.0, -0.5, 2.0] {
            let doc = json!({"train": {"digest": digest(1)}, "weighted_replay": [{"digest": digest(2), "share": share}]}).to_string();
            assert!(request(json!({"base": "a/b", "data": doc}), &[]).is_err(), "share {share}");
        }
        let doc = json!({"train": {"digest": digest(1)}, "replay": [{"digest": digest(2)}], "weighted_replay": [{"blob": "replay_1", "share": 0.25}]}).to_string();
        let r = request(json!({"base": "a/b", "data": doc}), &["replay_1"]).unwrap();
        assert_eq!((r.data.replay.len(), r.data.weighted_replay.len()), (1, 1));
    }

    #[test]
    fn an_oversized_document_is_refused_unparsed() {
        let doc = format!("{{\"train\":{{\"digest\":\"{}\"}},\"pad\":\"{}\"}}", digest(1), "x".repeat(MAX_DATA_DOCUMENT_BYTES));
        let e = request(json!({"base": "a/b", "data": doc}), &[]).unwrap_err();
        assert!(e.contains("at most 65536"), "{e}");
        // Deeply nested JSON is serde_json's own recursion limit, not a stack overflow.
        let nested = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
        assert!(request(json!({"base": "a/b", "data": nested}), &[]).is_err());
    }

    #[test]
    fn the_spec_bounds_every_scalar_before_anything_is_read() {
        let data = json!({"train": {"digest": digest(1)}}).to_string();
        for (name, value) in [
            ("steps", json!(0)), ("steps", json!(10_000_000)), ("rank", json!(0)), ("rank", json!(100_000)), ("grad_accum", json!(0)), ("max_block", json!(1)), ("max_block", json!(u64::MAX)),
            ("lr", json!(-1.0)), ("seed", json!(-1)), ("eval_every", json!(-1)), ("alpha", json!(1e12)), ("objective", json!("rl")), ("steps", json!("ten")), ("base", json!(5)),
        ] {
            let mut p = json!({"base": "a/b", "data": data});
            p[name] = value.clone();
            assert!(validated(p, &[]).is_err(), "{name}={value}");
        }
        for unknown in ["path", "out", "weights", "models_dir", "device", "save"] {
            let mut p = json!({"base": "a/b", "data": data});
            p[unknown] = json!("/etc/passwd");
            let e = validated(p, &[]).unwrap_err();
            assert!(e.contains("unknown param"), "{unknown}: {e}");
        }
    }

    #[test]
    fn policy_gradient_settings_belong_to_the_policy_gradient_objective() {
        let data = json!({"train": {"digest": digest(1)}}).to_string();
        let e = request(json!({"base": "a/b", "data": data, "kl_beta": 0.1, "teacher": "start"}), &[]).unwrap_err();
        assert!(e.contains("kl_beta, teacher") && e.contains("sft"), "{e}");
        assert!(request(json!({"base": "a/b", "data": data, "objective": "policy_gradient", "kl_beta": 0.1, "teacher": "start", "reference": "base", "loss_agg": "token-sum"}), &[]).is_ok());
        assert!(request(json!({"base": "a/b", "data": data, "keep_evaluations": true}), &[]).unwrap_err().contains("eval_every"));
        let e = request(json!({"base": "a/b", "data": data, "max_step_kl": 0.1}), &[]).unwrap_err();
        assert!(e.contains("max_step_kl") && e.contains("sft"), "the step guard belongs to the policy gradient: {e}");
        let stable = json!({"base": "a/b", "data": data, "objective": "policy_gradient", "clip_eps_high": 0.28, "importance": "truncate:2", "max_step_kl": 0.1, "max_step_grad_norm": 5.0});
        assert!(request(stable, &[]).is_ok());
    }

    #[test]
    fn the_evaluations_of_a_run_are_bounded_so_its_report_is() {
        let data = json!({"train": {"digest": digest(1)}}).to_string();
        assert!(request(json!({"base": "a/b", "data": data, "steps": 1_000_000, "eval_every": 1}), &[]).unwrap_err().contains("more than 1024 evaluations"));
        assert!(request(json!({"base": "a/b", "data": data, "steps": 1_000_000, "eval_every": 1000}), &[]).is_ok());
    }

    #[test]
    fn an_adapter_id_is_plain_text() {
        let data = json!({"train": {"digest": digest(1)}}).to_string();
        for bad in ["", "has space", "new\nline", "../../x y", &"a".repeat(MAX_ADAPTER_ID_LEN + 1)] {
            assert!(request(json!({"base": "a/b", "data": data, "adapter_id": bad}), &[]).is_err(), "{bad:?}");
        }
        assert!(request(json!({"base": "a/b", "data": data, "adapter_id": "Qwen/Qwen3.8-27B-FP8:local:chat:cand-1"}), &[]).is_ok());
    }

    #[test]
    fn a_preference_request_takes_pairs_and_its_own_settings_and_refuses_the_rest_by_name() {
        let data = json!({"train": {"digest": digest(1)}, "held_out": {"digest": digest(2)}, "start_adapter": {"digest": digest(3)}}).to_string();
        let ok = request(json!({"base": "a/b", "data": data, "objective": "dpo", "beta": 0.2, "nll_weight": 0.5, "keep_reasoning": true, "steps": 10, "grad_accum": 4, "strict_updates": true}), &[]).unwrap();
        assert_eq!((ok.params.objective, ok.params.beta, ok.params.nll_weight, ok.params.keep_reasoning), (Objective::Dpo, Some(0.2), Some(0.5), Some(true)));
        assert_eq!(Objective::Dpo.rows_per_step(), 2, "both candidates of a pair share a forward");
        let plain = request(json!({"base": "a/b", "data": data, "objective": "dpo"}), &[]).unwrap();
        assert_eq!((plain.params.beta, plain.params.nll_weight), (None, None), "the trainer's own defaults apply");
        for (extra, named) in [
            (json!({"clip_eps": 0.2}), "clip_eps"),
            (json!({"kl_beta": 0.1}), "kl_beta"),
            (json!({"eval_every": 5}), "eval_every"),
            (json!({"patience": 2}), "patience"),
            (json!({"weight_decay": 0.1}), "weight_decay"),
            (json!({"replay_share": 0.5}), "replay_share"),
            (json!({"pack_answers": true}), "pack_answers"),
            (json!({"thinking": false}), "thinking"),
        ] {
            let mut p = json!({"base": "a/b", "data": data, "objective": "dpo"});
            p.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let e = request(p, &[]).unwrap_err();
            assert!(e.contains(named) && e.contains("dpo"), "{named}: {e}");
        }
        for sets in [json!({"monitor": {"digest": digest(4)}}), json!({"held_out_text": {"digest": digest(4)}}), json!({"replay": [{"digest": digest(4)}]})] {
            let mut doc = json!({"train": {"digest": digest(1)}});
            doc.as_object_mut().unwrap().extend(sets.as_object().unwrap().clone());
            assert!(request(json!({"base": "a/b", "data": doc.to_string(), "objective": "dpo"}), &[]).unwrap_err().contains("not settings of the dpo objective"));
        }
        // Its own settings belong to it alone, and are bounded by the spec.
        let e = request(json!({"base": "a/b", "data": data, "beta": 0.1}), &[]).unwrap_err();
        assert!(e.contains("beta") && e.contains("sft"), "{e}");
        let e = request(json!({"base": "a/b", "data": data, "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "nll_weight": 1.0}), &[]).unwrap_err();
        assert!(e.contains("nll_weight") && e.contains("policy_gradient"), "{e}");
        for (name, value) in [("beta", json!(-0.1)), ("beta", json!(1e9)), ("nll_weight", json!(-1.0)), ("nll_weight", json!(1e9)), ("keep_reasoning", json!("yes"))] {
            let mut p = json!({"base": "a/b", "data": data, "objective": "dpo"});
            p[name] = value.clone();
            assert!(validated(p, &[]).is_err(), "{name}={value}");
        }
    }
}
