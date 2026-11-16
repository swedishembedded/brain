// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from
// random weights - vocabulary, chat format, initialization and pretraining -
// for its clients. If your team needs expertise in building a small model of
// its own end to end then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A model from nothing: [`ModelInit`] trains a byte-level BPE vocabulary on a corpus, declares its special tokens and
//! chat template and writes a randomly initialised checkpoint of a declared configuration; [`Pretrain`] trains any
//! checkpoint on raw text. Both are architecture-generic: the architecture is a registry id whose trainer declares
//! the capability ([`model::finetune::ChatTrainer::FROM_SCRATCH`] for an initialization, a full-parameter trainer
//! for pretraining), and every other one is refused by name. [`crate::from_scratch_architectures`] lists the first.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let made = brain::ModelInit::new("ouro")
//!     .config(serde_json::json!({"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4,
//!         "num_key_value_heads": 4, "head_dim": 64, "intermediate_size": 704, "max_position_embeddings": 2048,
//!         "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "total_ut_steps": 4, "tie_word_embeddings": false}))
//!     .corpus("corpus.jsonl")
//!     .vocab_size(4096)
//!     .special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
//!     .eos_token("<|im_end|>")
//!     .chat_template(std::fs::read_to_string("template.jinja")?)
//!     .out_dir("models/local/tiny")
//!     .run()?;
//! let trained = brain::Pretrain::from_pretrained(made.checkpoint.to_str().unwrap()).corpus("corpus.jsonl").steps(1000).out_dir("runs/pretrain").run()?;
//! # let _ = trained; Ok(()) }
//! ```
//!
//! **The corpus** is JSONL, one document per line as `{"text": "..."}` (unknown fields refused, blank lines skipped).
//! For pretraining each document is encoded with the checkpoint's own tokenizer, special tokens matched whole (so a
//! conversation serialized in the model's chat template trains the template's own tokens), and followed by the
//! separator token; the documents are concatenated into one stream and the run draws windows of `block + 1` tokens
//! from it, every position supervised. A held-out corpus is encoded the same way and scored every `eval_every` steps.
//!
//! **The template contract** a from-scratch model meets is checked before a weight is written: the template compiles,
//! a conversation renders with its last assistant turn a suffix of the prompt the model is asked from (the boundary
//! training masks by), the text it writes after an assistant turn contains the `eos_token` (so the model learns to
//! stop and serving stops it), and, when the template has a place for tools, an assistant tool-call turn renders and
//! the serving parser reads the same call back from it: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
//! (or the `<function=...>` form, when the template writes it).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::finetune::{base_digest, digest, fit_opts, resolve_seed, Asked};
use crate::full_finetune::{chat_entry, plan_before_load, FullRun};
use crate::{CancelToken, Device, Error, FineTuneProgress, FineTuneStatus, Result};

/// The fewest tokens a vocabulary holds beyond its 256 bytes and special tokens: one merge.
const MIN_MERGES: usize = 1;
/// The largest vocabulary [`ModelInit`] trains.
pub const MAX_VOCAB: usize = 1 << 20;
/// The most special tokens a vocabulary declares, and the longest one in bytes.
pub const MAX_SPECIAL_TOKENS: usize = 1024;
pub const MAX_SPECIAL_TOKEN_BYTES: usize = 128;
/// The longest chat template accepted, in bytes.
pub const MAX_TEMPLATE_BYTES: usize = 64 * 1024;
/// The corpus text a vocabulary is learned from by default: the first this-many bytes of documents.
pub const DEFAULT_TOKENIZER_BYTES: u64 = 64 << 20;
/// The separator a pretraining corpus's documents are followed by unless the run names one.
pub const DEFAULT_SEPARATOR: &str = data::bytebpe::END_OF_TEXT;

/// One document of a text corpus.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextRecord {
    text: String,
}

/// The documents of the JSONL corpus at `path`, at most `max_bytes` of text when bounded (the document that crosses
/// the bound is the last one read). A malformed line is refused naming it; an empty corpus is refused.
pub(crate) fn read_text_corpus(path: &Path, max_bytes: Option<u64>) -> Result<Vec<String>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    let (mut docs, mut bytes) = (Vec::new(), 0u64);
    for (at, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| Error::Backend(format!("{}: line {}: {e}", path.display(), at + 1)))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: TextRecord = serde_json::from_str(&line).map_err(|e| Error::Backend(format!("{}: line {}: not a {{\"text\": ...}} document: {e}", path.display(), at + 1)))?;
        bytes += record.text.len() as u64;
        docs.push(record.text);
        if max_bytes.is_some_and(|max| bytes >= max) {
            break;
        }
    }
    if docs.is_empty() {
        return Err(Error::Backend(format!("{}: holds no documents", path.display())));
    }
    Ok(docs)
}

/// A from-scratch model: vocabulary, chat format and random initialization. See the module doc.
#[derive(Clone, Debug)]
pub struct ModelInit {
    arch: String,
    config: serde_json::Value,
    corpus: Option<PathBuf>,
    tokenizer_bytes: u64,
    vocab_size: usize,
    special_tokens: Vec<String>,
    eos_token: Option<String>,
    bos_token: Option<String>,
    pad_token: Option<String>,
    chat_template: Option<String>,
    seed: Option<u64>,
    out_dir: Option<PathBuf>,
}

/// What [`ModelInit`] made.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInitOutcome {
    /// The checkpoint directory: the family's weights beside `tokenizer.json`, `tokenizer_config.json` and
    /// `generation_config.json`. A base like any other: [`Pretrain`] and [`crate::ChatFullFineTune`] train it and
    /// `brain serve` serves it.
    pub checkpoint: PathBuf,
    pub arch: String,
    /// The seed of the initialization: the one asked for or the one drawn. The same seed, corpus and settings
    /// write the same files.
    pub seed: u64,
    /// The vocabulary: 256 bytes, the merges, then the special tokens in declared order.
    pub vocab_size: usize,
    pub merges: usize,
    /// The parameters of the initialised model.
    pub params: u64,
    /// The configuration written, `vocab_size` included, in the family's `config.json` spelling.
    pub config: serde_json::Value,
    /// `sha256:<hex>` of the `tokenizer.json` written.
    pub tokenizer_digest: String,
    /// Corpus documents and bytes the vocabulary was learned from.
    pub tokenizer_documents: usize,
    pub tokenizer_bytes: u64,
    /// The chat template has a place for tools and its tool-call turn reads back through the serving parser.
    pub tool_calls: bool,
}

impl ModelInit {
    /// A model of architecture `arch` (a registry id, `ouro` or `qwen3`).
    pub fn new(arch: impl Into<String>) -> ModelInit {
        ModelInit {
            arch: arch.into(),
            config: serde_json::Value::Object(Default::default()),
            corpus: None,
            tokenizer_bytes: DEFAULT_TOKENIZER_BYTES,
            vocab_size: 0,
            special_tokens: Vec::new(),
            eos_token: None,
            bos_token: None,
            pad_token: None,
            chat_template: None,
            seed: None,
            out_dir: None,
        }
    }

    /// The model's configuration, a JSON object in the architecture's `config.json` spelling. `vocab_size` may be
    /// left out (the trained vocabulary's is written); given, it must equal [`Self::vocab_size`].
    pub fn config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }

    /// The JSONL text corpus (`{"text": ...}` per line) the vocabulary is learned from (required).
    pub fn corpus(mut self, path: impl Into<PathBuf>) -> Self {
        self.corpus = Some(path.into());
        self
    }

    /// Learn the vocabulary from at most this many bytes of the corpus's first documents (default 64 MiB).
    pub fn tokenizer_bytes(mut self, bytes: u64) -> Self {
        self.tokenizer_bytes = bytes;
        self
    }

    /// The whole vocabulary: 256 byte tokens, the learned merges and the special tokens (required).
    pub fn vocab_size(mut self, tokens: usize) -> Self {
        self.vocab_size = tokens;
        self
    }

    /// The special tokens, matched whole before byte-pair encoding and never learned as pieces: chat delimiters,
    /// tool-call tags, the document separator. They take the last ids, in this order.
    pub fn special_tokens<S: Into<String>>(mut self, tokens: impl IntoIterator<Item = S>) -> Self {
        self.special_tokens = tokens.into_iter().map(Into::into).collect();
        self
    }

    /// The token that ends a generation (required, one of the special tokens): the end of an assistant turn.
    pub fn eos_token(mut self, token: impl Into<String>) -> Self {
        self.eos_token = Some(token.into());
        self
    }

    /// The token a template may open a conversation with (`{{ bos_token }}`), one of the special tokens.
    pub fn bos_token(mut self, token: impl Into<String>) -> Self {
        self.bos_token = Some(token.into());
        self
    }

    /// The padding token, one of the special tokens.
    pub fn pad_token(mut self, token: impl Into<String>) -> Self {
        self.pad_token = Some(token.into());
        self
    }

    /// The chat template (Jinja, as a `tokenizer_config.json` carries it). Without one the model is a base model:
    /// it pretrains and serves raw prompts, and a chat fine-tune refuses it.
    pub fn chat_template(mut self, jinja: impl Into<String>) -> Self {
        self.chat_template = Some(jinja.into());
        self
    }

    /// The seed of the initialization (default: a random draw, reported on the outcome).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The checkpoint directory to write (required): absent or empty.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// Check what can be checked without reading the corpus: the architecture declares the capability, the
    /// vocabulary and its special tokens are well formed, the configuration is the architecture's. Returns the
    /// configuration with `vocab_size` set and its parameter count.
    pub fn plan(&self) -> Result<(serde_json::Value, u64)> {
        let chat = chat_entry(&self.arch)?;
        if !chat.declares_init {
            return Err(Error::Backend(format!("ModelInit: {}: the architecture does not declare a from-scratch initialization (those that do: {})", self.arch, crate::from_scratch_architectures().join(", "))));
        }
        let specials = &self.special_tokens;
        if specials.is_empty() || specials.len() > MAX_SPECIAL_TOKENS {
            return Err(Error::Backend(format!("ModelInit: declare 1 to {MAX_SPECIAL_TOKENS} special tokens (the eos token at least)")));
        }
        for (i, t) in specials.iter().enumerate() {
            if t.is_empty() || t.len() > MAX_SPECIAL_TOKEN_BYTES || t.chars().any(char::is_control) {
                return Err(Error::Backend(format!("ModelInit: special token {t:?} must be 1 to {MAX_SPECIAL_TOKEN_BYTES} bytes without control characters")));
            }
            if specials[..i].contains(t) {
                return Err(Error::Backend(format!("ModelInit: special token {t:?} is declared twice")));
            }
        }
        let eos = self.eos_token.as_deref().ok_or_else(|| Error::MissingArgument("ModelInit: no eos token; call .eos_token(...)".to_string()))?;
        for (what, token) in [("eos_token", Some(eos)), ("bos_token", self.bos_token.as_deref()), ("pad_token", self.pad_token.as_deref())] {
            if let Some(token) = token.filter(|t| !specials.iter().any(|s| s == t)) {
                return Err(Error::Backend(format!("ModelInit: {what} {token:?} is not one of the special tokens")));
            }
        }
        let floor = 256 + specials.len() + MIN_MERGES;
        if self.vocab_size < floor || self.vocab_size > MAX_VOCAB {
            return Err(Error::Backend(format!("ModelInit: vocab_size {} must be {floor} to {MAX_VOCAB} (256 bytes, {} special tokens and at least one merge)", self.vocab_size, specials.len())));
        }
        if let Some(t) = &self.chat_template {
            if t.len() > MAX_TEMPLATE_BYTES {
                return Err(Error::Backend(format!("ModelInit: the chat template is {} bytes; at most {MAX_TEMPLATE_BYTES} are accepted", t.len())));
            }
        }
        let serde_json::Value::Object(mut config) = self.config.clone() else { return Err(Error::Backend("ModelInit: config must be a JSON object".to_string())) };
        match config.get("vocab_size") {
            Some(v) if v.as_u64() != Some(self.vocab_size as u64) => return Err(Error::Backend(format!("ModelInit: config.vocab_size {v} is not the vocabulary's {}", self.vocab_size))),
            _ => {}
        }
        config.insert("vocab_size".to_string(), serde_json::json!(self.vocab_size));
        let config = serde_json::Value::Object(config);
        let params = (chat.init_params)(&config).map_err(|e| Error::Backend(format!("ModelInit: config: {e}")))?;
        Ok((config, params))
    }

    /// Write the checkpoint. Nothing is left in the out directory when a check fails.
    pub fn run(&self) -> Result<ModelInitOutcome> {
        let (config, params) = self.plan()?;
        let chat = chat_entry(&self.arch)?;
        self.corpus.as_deref().ok_or_else(|| Error::MissingArgument("ModelInit: no corpus; call .corpus(path)".to_string()))?;
        let out = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ModelInit: no out directory; call .out_dir(path)".to_string()))?;
        if out.exists() && std::fs::read_dir(out).map_err(|e| Error::Backend(format!("{}: {e}", out.display())))?.next().is_some() {
            return Err(Error::Backend(format!("ModelInit: {} is not empty", out.display())));
        }
        std::fs::create_dir_all(out).map_err(|e| Error::Backend(format!("{}: {e}", out.display())))?;
        let made = self.write(out, &chat, config, params);
        if made.is_err() {
            std::fs::remove_dir_all(out).ok();
        }
        made
    }

    fn write(&self, out: &Path, chat: &crate::chat_trainer::ChatEntry, config: serde_json::Value, params: u64) -> Result<ModelInitOutcome> {
        let corpus = self.corpus.as_deref().expect("checked by run");
        let docs = read_text_corpus(corpus, Some(self.tokenizer_bytes))?;
        let specials: Vec<&str> = self.special_tokens.iter().map(String::as_str).collect();
        let merges = self.vocab_size - 256 - specials.len();
        let bpe = data::bytebpe::ByteBpe::train_reserving(&docs.join("\n"), merges, &specials);
        if bpe.merge_count() < merges {
            return Err(Error::Backend(format!("ModelInit: the corpus yields {} merges and vocab_size {} needs {merges}: give more text or a smaller vocabulary", bpe.merge_count(), self.vocab_size)));
        }
        let tokenizer = bpe.to_hf_tokenizer_json(&specials).map_err(|e| Error::Backend(format!("ModelInit: {e}")))?;
        let tokenizer_bytes = serde_json::to_vec(&tokenizer).map_err(|e| Error::Backend(e.to_string()))?;
        let write = |name: &str, bytes: &[u8]| std::fs::write(out.join(name), bytes).map_err(|e| Error::Backend(format!("{}: {e}", out.join(name).display())));
        write("tokenizer.json", &tokenizer_bytes)?;
        let id_of = |t: &str| specials.iter().position(|s| *s == t).map(|i| 256 + merges + i);
        let eos = self.eos_token.as_deref().expect("checked by plan");
        let mut tokenizer_config = serde_json::json!({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": false,
            "eos_token": eos,
            "bos_token": self.bos_token,
            "pad_token": self.pad_token,
            "added_tokens_decoder": specials.iter().enumerate().map(|(i, s)| ((256 + merges + i).to_string(), serde_json::json!({"content": s, "lstrip": false, "normalized": false, "rstrip": false, "single_word": false, "special": true}))).collect::<serde_json::Map<_, _>>(),
        });
        if let Some(t) = &self.chat_template {
            tokenizer_config["chat_template"] = serde_json::json!(t);
        }
        write("tokenizer_config.json", &serde_json::to_vec_pretty(&tokenizer_config).map_err(|e| Error::Backend(e.to_string()))?)?;
        let generation = serde_json::json!({"eos_token_id": [id_of(eos)], "bos_token_id": self.bos_token.as_deref().and_then(id_of), "pad_token_id": self.pad_token.as_deref().and_then(id_of)});
        write("generation_config.json", &serde_json::to_vec_pretty(&generation).map_err(|e| Error::Backend(e.to_string()))?)?;
        let tool_calls = match &self.chat_template {
            Some(_) => check_template(out, eos)?,
            None => false,
        };
        let seed = self.seed.unwrap_or_else(crate::random_seed);
        (chat.init_checkpoint)(&config, seed, out).map_err(|e| Error::Backend(format!("ModelInit: {e}")))?;
        Ok(ModelInitOutcome {
            checkpoint: out.to_path_buf(),
            arch: self.arch.clone(),
            seed,
            vocab_size: self.vocab_size,
            merges,
            params,
            config,
            tokenizer_digest: digest(&out.join("tokenizer.json"))?,
            tokenizer_documents: docs.len(),
            tokenizer_bytes: docs.iter().map(|d| d.len() as u64).sum(),
            tool_calls,
        })
    }
}

/// Holds the chat template in `dir` to the contract of the module doc, read exactly as training and serving read it.
/// Returns whether it serves tool calls.
fn check_template(dir: &Path, eos: &str) -> Result<bool> {
    let refuse = |why: String| Error::Backend(format!("ModelInit: the chat template {why}"));
    let (tok, tmpl) = crate::finetune::tokenizer_and_template(dir).map_err(|e| refuse(format!("does not load: {e}")))?;
    let end = tmpl.assistant_turn_end().map_err(|e| refuse(e.to_string()))?;
    if !end.contains(eos) {
        return Err(refuse(format!("writes {end:?} after an assistant turn, which does not hold the eos token {eos:?}: the model would never learn to stop")));
    }
    let tools = tmpl.source().contains("tools");
    let plain = serde_json::json!({"messages": [{"role": "system", "content": "probe system", "train": false}, {"role": "user", "content": "probe question", "train": false}, {"role": "assistant", "content": "probe answer", "train": true}]});
    let tool = serde_json::json!({
        "tools": [{"type": "function", "function": {"name": "probe_tool", "description": "a probe", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
        "messages": [
            {"role": "user", "content": "probe question", "train": false},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "probe_tool", "arguments": "{\"path\": \"a.txt\"}"}}], "train": true},
        ],
    });
    let probes: Vec<&serde_json::Value> = if tools { vec![&plain, &tool] } else { vec![&plain] };
    let probe_path = dir.join(".template-probe.jsonl");
    std::fs::write(&probe_path, probes.iter().map(|p| p.to_string()).collect::<Vec<_>>().join("\n")).map_err(|e| Error::Backend(format!("{}: {e}", probe_path.display())))?;
    let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: None, render: data::chat::RenderOpts::default(), pack: false, images: &data::chat::TextOnly::TRAINER };
    let samples = asked.read(&probe_path);
    std::fs::remove_file(&probe_path).ok();
    let samples = samples.map_err(|e| refuse(format!("cannot render a conversation for training: {e}")))?;
    if !tools {
        return Ok(false);
    }
    // The tool-call turn as training supervises it, read back by the parser serving uses for this template.
    let format = qwen3::chat::ChatFormat::own_template(dir).map_err(refuse)?;
    let mut inv = capability::Invocation::new();
    inv.params = serde_json::json!({"messages": serde_json::json!([{"role": "user", "content": "probe question"}]).to_string(), "tools": tool["tools"].to_string()});
    let mut seen = Vec::new();
    for sample in &samples {
        let (ids, mask) = sample.encode_with(&tok, &tmpl, data::chat::RenderOpts::default()).map_err(|e| refuse(e.to_string()))?;
        let supervised: Vec<u32> = ids.iter().zip(&mask).filter(|(_, m)| **m).map(|(id, _)| *id).collect();
        let text = data::tokenizer::Tokenizer::decode(&tok, &supervised);
        let Ok(reply) = qwen3::chat::parse_reply_as(&format, &inv, &text) else { continue };
        let calls: Vec<serde_json::Value> = reply.outputs.get("tool_calls").and_then(|c| c.as_str()).and_then(|c| serde_json::from_str(c).ok()).unwrap_or_default();
        let read_back = calls.first().is_some_and(|c| c["name"] == "probe_tool" && serde_json::from_str::<serde_json::Value>(c["arguments"].as_str().unwrap_or("null")).ok() == Some(serde_json::json!({"path": "a.txt"})));
        if read_back {
            return Ok(true);
        }
        seen.push(text);
    }
    Err(refuse(format!("writes the tool-call turn as {seen:?}, which the serving parser does not read back as the call (it reads <tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call>)")))
}

/// Raw-text pretraining of a checkpoint. See the module doc.
#[derive(Clone, Debug)]
pub struct Pretrain {
    base: String,
    arch: Option<String>,
    models_dir: Option<String>,
    corpus: Option<PathBuf>,
    held_out: Option<PathBuf>,
    separator: Option<String>,
    steps: u32,
    batch: u32,
    block: u32,
    lr: f32,
    min_lr: Option<f32>,
    warmup: Option<u32>,
    weight_decay: f32,
    grad_clip: f32,
    grad_accum: u32,
    eval_every: u32,
    eval_batches: u32,
    strict_updates: bool,
    seed: Option<u64>,
    out_dir: Option<PathBuf>,
    device: Device,
    placed: bool,
    resume_state: Option<PathBuf>,
    checkpoint_every: u32,
}

/// What one pretraining run did.
#[derive(Clone, Debug, PartialEq)]
pub struct PretrainOutcome {
    pub status: FineTuneStatus,
    /// The trained checkpoint directory; `None` unless completed.
    pub checkpoint: Option<PathBuf>,
    /// `sha256:<hex>` of its weights file.
    pub weights_digest: Option<String>,
    pub base_digest: String,
    pub seed: u64,
    pub steps: u32,
    pub steps_completed: u32,
    pub resumed_at: Option<u32>,
    pub initial_loss: f32,
    pub final_loss: Option<f32>,
    /// `(step, train loss over the interval, held-out loss)` of every evaluation, those before a resume included.
    pub evaluations: Vec<(u32, f32, f32)>,
    pub documents: usize,
    pub tokens: usize,
    pub held_out_documents: usize,
    pub held_out_tokens: usize,
    /// The separator token and its id.
    pub separator: String,
    pub separator_id: u32,
    pub batch: u32,
    pub block: u32,
    pub lr: f32,
}

impl Pretrain {
    /// Pretraining of `base`: a checkpoint file or directory, or a `vendor/repo` reference in the model store.
    pub fn from_pretrained(base: impl Into<String>) -> Pretrain {
        Pretrain {
            base: base.into(),
            arch: None,
            models_dir: None,
            corpus: None,
            held_out: None,
            separator: None,
            steps: 100,
            batch: 8,
            block: 256,
            lr: 1e-3,
            min_lr: None,
            warmup: None,
            weight_decay: 0.1,
            grad_clip: 1.0,
            grad_accum: 1,
            eval_every: 0,
            eval_batches: 8,
            strict_updates: false,
            seed: None,
            out_dir: None,
            device: Device::default(),
            placed: false,
            resume_state: None,
            checkpoint_every: 0,
        }
    }

    /// The architecture of the base when the checkpoint does not tell it.
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = Some(id.into());
        self
    }
    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }
    /// The JSONL training corpus (required).
    pub fn corpus(mut self, path: impl Into<PathBuf>) -> Self {
        self.corpus = Some(path.into());
        self
    }
    /// A JSONL held-out corpus, scored every [`Self::eval_every`] steps and never trained on.
    pub fn held_out(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out = Some(path.into());
        self
    }
    /// The special token every document is followed by (default `<|endoftext|>`; the vocabulary must hold it).
    pub fn separator(mut self, token: impl Into<String>) -> Self {
        self.separator = Some(token.into());
        self
    }
    /// Optimizer steps (default 100).
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }
    /// Rows per micro-batch (default 8).
    pub fn batch(mut self, rows: u32) -> Self {
        self.batch = rows;
        self
    }
    /// Tokens per row (default 256), at most the model's positions.
    pub fn block(mut self, tokens: u32) -> Self {
        self.block = tokens;
        self
    }
    /// The peak learning rate (default 1e-3).
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }
    /// The rate the cosine decays to (default a tenth of the peak).
    pub fn min_lr(mut self, lr: f32) -> Self {
        self.min_lr = Some(lr);
        self
    }
    /// Linear warmup steps (default a twentieth of the run).
    pub fn warmup(mut self, steps: u32) -> Self {
        self.warmup = Some(steps);
        self
    }
    /// AdamW's decoupled weight decay (default 0.1).
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }
    /// The global gradient-norm clip (default 1; 0 turns it off).
    pub fn grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip;
        self
    }
    /// Micro-batches summed into one optimizer step (default 1).
    pub fn grad_accum(mut self, micro_batches: u32) -> Self {
        self.grad_accum = micro_batches;
        self
    }
    /// Score the held-out corpus every this-many steps (default 0: never; needs [`Self::held_out`]).
    pub fn eval_every(mut self, steps: u32) -> Self {
        self.eval_every = steps;
        self
    }
    /// Batches of held-out windows one evaluation averages (default 8).
    pub fn eval_batches(mut self, batches: u32) -> Self {
        self.eval_batches = batches;
        self
    }
    /// Abort at the first step whose loss or gradient is not finite (default: skip it).
    pub fn strict_updates(mut self, strict: bool) -> Self {
        self.strict_updates = strict;
        self
    }
    /// The seed of the window draws (default a random draw, or the one a resumed state was saved with).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Where the prepared corpus and the checkpoint go (required). One directory per run.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }
    /// The device the run is placed on.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    /// Run on the device this thread is already scoped to (see [`crate::ChatFullFineTune::placed`]).
    pub fn placed(mut self, on: bool) -> Self {
        self.placed = on;
        self
    }
    /// Exact resume through this training-state file (see [`crate::ChatFullFineTune::resume_state`]).
    pub fn resume_state(mut self, path: impl Into<PathBuf>) -> Self {
        self.resume_state = Some(path.into());
        self
    }
    /// Also save the resume state every this-many steps (default 0: only when stopped).
    pub fn checkpoint_every(mut self, steps: u32) -> Self {
        self.checkpoint_every = steps;
        self
    }

    /// Run to completion.
    pub fn run(&self) -> Result<PretrainOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step (and every evaluation, on its step) and stopping at the next step boundary
    /// once `cancel` fires.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<PretrainOutcome> {
        let corpus = self.corpus.as_deref().ok_or_else(|| Error::MissingArgument("Pretrain: no corpus; call .corpus(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("Pretrain: no out directory; call .out_dir(path)".to_string()))?;
        if self.eval_every > 0 && self.held_out.is_none() {
            return Err(Error::Backend("Pretrain: eval_every needs a held-out corpus".to_string()));
        }
        if self.batch == 0 || self.block < 2 || self.grad_accum == 0 {
            return Err(Error::Backend("Pretrain: batch and grad_accum must be at least 1 and block at least 2".to_string()));
        }
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, _) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let arch = match self.arch.as_deref() {
            Some(named) => named,
            None => crate::study::arch_of_checkpoint(&model_dir).map_or("qwen3", |a| a.id),
        };
        let chat = chat_entry(arch)?;
        let open = (chat.checkpoint_path)(&weights, &model_dir);
        if let Some(positions) = positions_of(&open) {
            if u64::from(self.block) > positions {
                return Err(Error::Backend(format!("Pretrain: block {} is longer than the model's {positions} positions", self.block)));
            }
        }
        let tok_path = model_dir.join("tokenizer.json");
        let tok = data::qwen_tokenizer::QwenBpe::from_file(crate::finetune::utf8(&tok_path)?).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;
        let separator = self.separator.clone().unwrap_or_else(|| DEFAULT_SEPARATOR.to_string());
        let separator_id = tok.special_id(&separator).ok_or_else(|| Error::Backend(format!("Pretrain: the vocabulary has no special token {separator:?} to separate documents with; name one with separator")))?;
        let vocab = (chat.vocab)(&open).map_err(Error::Backend)?;
        let encode = |path: &Path| -> Result<(usize, Vec<u32>)> {
            let docs = read_text_corpus(path, None)?;
            let mut ids = Vec::new();
            for doc in &docs {
                ids.extend(data::tokenizer::Tokenizer::encode(&tok, doc));
                ids.push(separator_id);
            }
            if let Some(id) = ids.iter().find(|id| **id as usize >= vocab) {
                return Err(Error::Backend(format!("{}: encodes to token {id}, past the model's vocabulary of {vocab}", path.display())));
            }
            Ok((docs.len(), ids))
        };
        let (documents, train) = encode(corpus)?;
        let (held_out_documents, val) = match &self.held_out {
            Some(path) => encode(path)?,
            None => (0, Vec::new()),
        };
        let prepared = out_dir.join("prepared");
        std::fs::create_dir_all(&prepared).map_err(|e| Error::Backend(format!("{}: {e}", prepared.display())))?;
        let io = |e: std::io::Error| Error::Backend(format!("{}: {e}", prepared.display()));
        data::binio::write_u32_bin(&prepared.join("train.u32.bin"), &train).map_err(io)?;
        data::binio::write_u32_bin(&prepared.join("val.u32.bin"), &val).map_err(io)?;
        std::fs::write(prepared.join("meta.json"), data::binio::Meta::vocab_only(vocab)).map_err(io)?;

        let set = if self.placed {
            gpu_core::devices::ComputeSet { gpus: gpu_core::devices::current_gpu().into_iter().collect(), ..gpu_core::devices::ambient_compute_set().clone() }
        } else {
            crate::device::resolve(&self.device)?
        };
        plan_before_load(&open, self.block, &set)?;
        let seed = resolve_seed(self.seed, self.resume_state.as_deref())?;
        let requested = model::FitOpts {
            batch_size: self.batch,
            min_lr: self.min_lr.unwrap_or(self.lr / 10.0),
            warmup: self.warmup.unwrap_or((self.steps / 20).max(1)),
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            grad_accum: self.grad_accum,
            eval_interval: self.eval_every,
            eval_batches: self.eval_batches,
            strict_updates: self.strict_updates,
            ..fit_opts(self.steps, self.block, self.lr, seed)
        };
        let opts = model::training_plan::ValidatedTrainingPlan::build(requested, &[]).map_err(|e| Error::Backend(format!("Pretrain: {e}")))?.into_opts();
        let base_digest = base_digest(&open)?;
        let identity = serde_json::json!({
            "objective": "pretrain",
            "base": base_digest,
            "corpus": digest(corpus)?,
            "held_out": self.held_out.as_deref().map(digest).transpose()?,
            "separator": separator,
        });
        let precision = model::TrainPrecision::default();
        let run = FullRun { chat: &chat, open: &open, tokenizer_dir: &model_dir, prepared_dir: &prepared, out_dir, opts: &opts, set: &set, precision: &precision, resume_state: self.resume_state.as_deref(), checkpoint_every: self.checkpoint_every, identity };
        let trained = run.train_and_export(cancel, &mut on_progress)?;
        Ok(PretrainOutcome {
            status: trained.status,
            checkpoint: trained.checkpoint,
            weights_digest: trained.weights_digest,
            base_digest,
            seed,
            steps: self.steps,
            steps_completed: trained.report.steps_completed,
            resumed_at: trained.report.resumed_at,
            initial_loss: trained.report.initial_loss,
            final_loss: trained.report.final_loss,
            evaluations: trained.report.evaluations.iter().map(|e| (e.step, e.train_loss, e.eval_loss)).collect(),
            documents,
            tokens: train.len(),
            held_out_documents,
            held_out_tokens: val.len(),
            separator,
            separator_id,
            batch: self.batch,
            block: self.block,
            lr: self.lr,
        })
    }
}

/// The positions the checkpoint at `open` holds (`max_position_embeddings`), when its configuration says.
fn positions_of(open: &Path) -> Option<u64> {
    let config: serde_json::Value = if open.is_dir() {
        serde_json::from_slice(&std::fs::read(open.join("config.json")).ok()?).ok()?
    } else {
        checkpoint::weightio::WeightReader::open(open.to_str()?).ok()?.config()
    };
    config.get("max_position_embeddings").and_then(|v| v.as_u64())
}
