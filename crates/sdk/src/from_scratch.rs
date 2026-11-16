// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from
// random weights - vocabulary, chat format, initialization and pretraining -
// for its clients. If your team needs expertise in building a small model of
// its own end to end then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A model from nothing: [`ModelInit`] trains a byte-level BPE vocabulary on a corpus, declares its special tokens and
//! chat template and writes a randomly initialised checkpoint of a declared configuration. It is
//! architecture-generic: the architecture is a registry id whose trainer declares the capability
//! ([`model::finetune::ChatTrainer::FROM_SCRATCH`]), and every other one is refused by name;
//! [`crate::from_scratch_architectures`] lists those that declare it.
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
//! println!("{} parameters in {}", made.params, made.checkpoint.display());
//! # Ok(()) }
//! ```
//!
//! **The corpus** is JSONL, one document per line as `{"text": "..."}` (unknown fields refused, blank lines skipped).
//! The vocabulary is learned from its first [`ModelInit::tokenizer_bytes`] bytes, the special tokens cut out first.
//!
//! **The template contract** a from-scratch model meets is checked before a weight is written: the template compiles,
//! a conversation renders with its last assistant turn a suffix of the prompt the model is asked from (the boundary
//! training masks by), the text it writes after an assistant turn contains the `eos_token` (so the model learns to
//! stop and serving stops it), and, when the template has a place for tools, an assistant tool-call turn renders and
//! the serving parser reads the same call back from it: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
//! (or the `<function=...>` form, when the template writes it).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::finetune::{digest, Asked};
use crate::full_finetune::chat_entry;
use crate::{Error, Result};

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
    /// `generation_config.json`. A base like any other: [`crate::ChatFullFineTune`] trains it and
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
