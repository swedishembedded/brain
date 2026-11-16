// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that create a model from
// nothing - vocabulary, chat format and initialization - on shared serving
// hardware, for its clients. If your team needs expertise in building small
// models of its own end to end, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `init_model`: a model from nothing, as a job of the service ([`brain::ModelInit`]). A byte-level BPE vocabulary is
//! learned from a corpus (by digest or blob) with the caller's special tokens, the caller's chat template is checked
//! against the contract training and serving rely on, and a checkpoint of the caller's configuration of a declaring
//! architecture is initialised from a seed. The checkpoint leaves like every checkpoint job's: a manifest of its
//! files in the artifact store and, with `install_as`, a model of the model store a later `pretrain` or `full_train`
//! names as its base.

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Media, ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::consolidate::{check_install_name, check_resume, checkpoint_outcome_of, open_run, resume_key_param};
use crate::job::{bring, trainer_failure, JobEnv, Workdir};
use crate::request::Source;
use crate::store::Digest;

/// The action's name.
pub const ACTION: &str = "init_model";
/// The longest `config` or `tokenizer` document accepted, in bytes.
pub const MAX_DOCUMENT_BYTES: usize = 64 * 1024;
/// The most parameters a model created here may have: what its initialization holds in host memory is bounded by it.
pub const MAX_PARAMS: u64 = 4 << 30;
/// The most corpus bytes the vocabulary is learned from.
pub const MAX_TOKENIZER_BYTES: f64 = (1u64 << 30) as f64;

/// What `init_model` is.
pub fn spec() -> ActionSpec {
    ActionSpec::new(ACTION, "create a model from nothing: learn a byte-level BPE vocabulary with declared special tokens on a text corpus, check a chat template, and initialise a checkpoint of a declared configuration of an architecture that declares it; the checkpoint leaves by manifest digest in the artifact store (and optionally into the model store)")
        .training()
        .cancellable()
        .param(ParamSpec::new("arch", ParamType::Str, "the architecture, a registry id whose trainer declares a from-scratch initialization (`ouro`, `qwen3`)").required())
        .param(ParamSpec::new("config", ParamType::Str, "the model's configuration, a JSON object in the architecture's config.json spelling; `vocab_size` may be left out (the tokenizer's is written)").required())
        .param(ParamSpec::new("tokenizer", ParamType::Str, "JSON document: {\"vocab_size\": N, \"special_tokens\": [...], \"eos_token\": T, \"bos_token\"?: T, \"pad_token\"?: T, \"split_digits\"?: bool}; the vocabulary is 256 bytes, the learned merges, then the special tokens in declared order; `split_digits` makes every digit its own token").required())
        .param(ParamSpec::new("chat_template", ParamType::Str, "the chat template (Jinja); without one the model is a base model that serves raw prompts"))
        .param(ParamSpec::new("corpus_digest", ParamType::Str, "the text corpus (JSONL, one {\"text\": ...} document per line), as a blake3 digest of the server's artifact store (or attach it as the input blob `corpus`; exactly one of the two)"))
        .param(ParamSpec::new("tokenizer_bytes", ParamType::Int, "learn the vocabulary from at most this many bytes of the corpus's first documents").default(json!(brain::from_scratch::DEFAULT_TOKENIZER_BYTES)).min(1.0).max(MAX_TOKENIZER_BYTES).step(1.0))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the initialization (random when omitted; the report carries the one used)").min(0.0).max(capability::bounds::EXACT_INTEGER))
        .param(ParamSpec::new("install_as", ParamType::Str, "also install the checkpoint in the server's model store as `local/<this name>` (letters, digits, `.`, `_`, `-`; it must not exist yet)"))
        .param(resume_key_param())
        .input(BlobSpec::new("corpus", Media::Bytes, "the text corpus, instead of `corpus_digest`"))
}

/// The vocabulary a request declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerDoc {
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub eos_token: String,
    pub bos_token: Option<String>,
    pub pad_token: Option<String>,
    /// Every digit its own pre-token ([`brain::ModelInit::split_digits`]).
    #[serde(default)]
    pub split_digits: bool,
}

/// The scalar settings, exactly the params the spec declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub arch: String,
    pub config: String,
    pub tokenizer: String,
    pub chat_template: Option<String>,
    pub corpus_digest: Option<String>,
    pub tokenizer_bytes: u64,
    pub seed: Option<u64>,
    pub install_as: Option<String>,
    pub resume_key: Option<String>,
}

/// A request that passed the spec and every check that needs no file.
#[derive(Debug)]
pub struct Request {
    pub params: Params,
    pub corpus: Source,
    /// The SDK pipeline it runs, every setting but the corpus and the out directory applied.
    pub init: brain::ModelInit,
    /// The parameters of the model it creates.
    pub params_count: u64,
}

fn document<T: serde::de::DeserializeOwned>(what: &str, text: &str) -> Result<T, String> {
    if text.len() > MAX_DOCUMENT_BYTES {
        return Err(format!("{ACTION}: {what}: the document is {} bytes; at most {MAX_DOCUMENT_BYTES} are accepted", text.len()));
    }
    serde_json::from_str(text).map_err(|e| format!("{ACTION}: {what}: {e}"))
}

impl Request {
    /// Reads the validated invocation `inv`: the corpus named exactly once, the documents well formed and bounded, the
    /// architecture one that declares the capability, the configuration its own and within [`MAX_PARAMS`].
    pub fn from_invocation(inv: &Invocation) -> Result<Request, String> {
        let params: Params = serde_json::from_value(inv.params.clone()).map_err(|e| format!("{ACTION}: {e}"))?;
        let corpus = match (&params.corpus_digest, inv.blobs.contains_key("corpus")) {
            (Some(digest), false) => Source::Digest(Digest::parse(digest).map_err(|e| format!("{ACTION}: corpus_digest: {e}"))?),
            (None, true) => Source::Blob("corpus".to_string()),
            _ => return Err(format!("{ACTION}: name the corpus exactly once: as the input blob `corpus` or as `corpus_digest`")),
        };
        if let Some(extra) = inv.blobs.keys().find(|k| k.as_str() != "corpus") {
            return Err(format!("{ACTION}: the input blob \"{extra}\" is not an input of this action"));
        }
        if let Some(name) = &params.install_as {
            check_install_name(name)?;
        }
        check_resume(ACTION, params.resume_key.as_deref(), None)?;
        let config: Value = document("config", &params.config)?;
        let tokenizer: TokenizerDoc = document("tokenizer", &params.tokenizer)?;
        let mut init = brain::ModelInit::new(params.arch.as_str())
            .config(config)
            .vocab_size(tokenizer.vocab_size)
            .special_tokens(tokenizer.special_tokens)
            .eos_token(tokenizer.eos_token)
            .split_digits(tokenizer.split_digits)
            .tokenizer_bytes(params.tokenizer_bytes);
        if let Some(t) = tokenizer.bos_token {
            init = init.bos_token(t);
        }
        if let Some(t) = tokenizer.pad_token {
            init = init.pad_token(t);
        }
        if let Some(t) = &params.chat_template {
            init = init.chat_template(t.as_str());
        }
        if let Some(seed) = params.seed {
            init = init.seed(seed);
        }
        let (_, params_count) = init.plan().map_err(|e| format!("{ACTION}: {e}"))?;
        if params_count > MAX_PARAMS {
            return Err(format!("{ACTION}: the configuration has {params_count} parameters; at most {MAX_PARAMS} are created here"));
        }
        Ok(Request { params, corpus, init, params_count })
    }

    /// The host memory the job holds: the initialised weights (and their copy as they are written) and the corpus text
    /// the vocabulary is learned from.
    pub fn host_bytes(&self) -> u64 {
        8 * self.params_count + 4 * self.params.tokenizer_bytes + (256 << 20)
    }
}

/// Runs the `init_model` job `inv`. Host work: it takes no card.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let request = Request::from_invocation(inv)?;
    // An initialization has no training state: a resumable one is idempotent, its recorded result answered again.
    let run = match open_run(env, ACTION, request.params.resume_key.as_deref(), inv, request.params.seed)? {
        Some(crate::runs::Opened::Completed(mut result)) => {
            result["resume"] = crate::runs::completed_report(request.params.resume_key.as_deref().unwrap_or_default());
            let mut outcome = capability::Outcome::new();
            outcome.outputs = result;
            return Ok(outcome);
        }
        Some(crate::runs::Opened::Run(run)) => Some(run),
        None => None,
    };
    let init = match &run {
        Some(r) => request.init.clone().seed(r.seed()),
        None => request.init.clone(),
    };
    let work = Workdir::create(env)?;
    progress(Progress::event(0, 1, json!({"type": "stage", "stage": "staging the corpus"})));
    let corpus = work.path().join("corpus.jsonl");
    bring(&request.corpus, &inv.blobs, env.store.as_ref(), work.staging(), &corpus)?;
    inv.cancel.check()?;
    progress(Progress::event(0, 1, json!({"type": "stage", "stage": "learning the vocabulary and initialising the weights"})));
    let out = work.path().join("model");
    let made = init.corpus(&corpus).out_dir(&out).run().map_err(trainer_failure)?;
    let report = json!({
        "status": "completed",
        "objective": "init",
        "arch": made.arch,
        "seed": made.seed,
        "vocab_size": made.vocab_size,
        "merges": made.merges,
        "params": made.params,
        "config": made.config,
        "tokenizer_digest": made.tokenizer_digest,
        "tokenizer_documents": made.tokenizer_documents,
        "tokenizer_bytes": made.tokenizer_bytes,
        "tool_calls": made.tool_calls,
    });
    work.ensure_within_quota()?;
    checkpoint_outcome_of(env, &made.checkpoint, request.params.install_as.as_deref(), report, run.map(|r| (r, None)))
}
