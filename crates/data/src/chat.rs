// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chat / tool-call fine-tuning data pipeline.
//!
//! Turns structured chat examples (system/user prompt + the assistant response
//! we want the model to learn) into brain's token-dataset layout with
//! **token-level supervision masking**: the prompt is masked (IGNORE targets)
//! and only the assistant span is trained — the correct recipe for function-call
//! and reasoning fine-tuning, where training on the (given) prompt would teach
//! the model to hallucinate user turns.
//!
//! Output (consumed unchanged by `model::fit` / `brain qwen finetune`):
//!   * `train.u32.bin` / `val.u32.bin` — `u32` token ids (Qwen's 151936 vocab).
//!   * `train.mask.bin` / `val.mask.bin` — `u8` per-token mask (1 = trainable).
//!   * `meta.json` — `{ vocab_size, token_width: 32 }`.
//!
//! Prompt and response are encoded **separately** and concatenated — exactly how
//! inference sees them (the prompt is encoded alone, the model then generates) —
//! so there is no train/inference tokenization skew across the boundary. Each
//! example is terminated by `<|endoftext|>` so windows can be sampled aligned to
//! example starts.

use std::io;
use std::path::Path;

use crate::binio;
use crate::qwen_tokenizer::QwenBpe;
use crate::tokenizer::Tokenizer;

/// `<|endoftext|>` — the document/example separator (also Qwen's pad/eos base).
pub const ENDOFTEXT: u32 = 151643;

/// One supervised chat example: a prompt (system optional + user) and the
/// assistant response the model must learn to produce.
#[derive(Clone, Debug)]
pub struct ChatExample {
    pub system: Option<String>,
    pub user: String,
    /// The assistant turn content to train on (e.g. a `<tool_call>…</tool_call>`
    /// block, or a `<think>…</think>` + answer). `<|im_end|>` is appended for you.
    pub assistant: String,
}

impl ChatExample {
    pub fn new(user: impl Into<String>, assistant: impl Into<String>) -> ChatExample {
        ChatExample { system: None, user: user.into(), assistant: assistant.into() }
    }
    pub fn with_system(system: impl Into<String>, user: impl Into<String>, assistant: impl Into<String>) -> ChatExample {
        ChatExample { system: Some(system.into()), user: user.into(), assistant: assistant.into() }
    }

    /// The prompt string through `<|im_start|>assistant\n` (what inference sees).
    pub fn prompt_str(&self, tok: &QwenBpe) -> String {
        let mut msgs: Vec<(&str, &str)> = Vec::new();
        if let Some(s) = &self.system {
            msgs.push(("system", s));
        }
        msgs.push(("user", &self.user));
        tok.apply_chat_template(&msgs, true)
    }

    /// Encode to `(ids, mask)`: prompt ids masked (false), response ids trained
    /// (true), then a masked `<|endoftext|>` separator.
    pub fn encode(&self, tok: &QwenBpe) -> (Vec<u32>, Vec<bool>) {
        let prompt = tok.encode(&self.prompt_str(tok));
        let resp = tok.encode(&format!("{}<|im_end|>\n", self.assistant));
        let mut ids = Vec::with_capacity(prompt.len() + resp.len() + 1);
        let mut mask = Vec::with_capacity(ids.capacity());
        ids.extend_from_slice(&prompt);
        mask.extend(std::iter::repeat_n(false, prompt.len()));
        ids.extend_from_slice(&resp);
        mask.extend(std::iter::repeat_n(true, resp.len()));
        ids.push(ENDOFTEXT);
        mask.push(false);
        (ids, mask)
    }
}

/// Encode a set of examples into one `(ids, mask)` stream.
pub fn encode_split(examples: &[ChatExample], tok: &QwenBpe) -> (Vec<u32>, Vec<bool>) {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for ex in examples {
        let (i, m) = ex.encode(tok);
        ids.extend_from_slice(&i);
        mask.extend_from_slice(&m);
    }
    (ids, mask)
}

/// Write a train/val split to `dir` as brain's masked token-dataset layout.
/// `vocab` is the MODEL's vocab (from its `config.json`), which must match the
/// checkpoint's `lm_head` — not the tokenizer's derived size.
pub fn prepare_chat(
    train: &[ChatExample],
    val: &[ChatExample],
    tok: &QwenBpe,
    vocab: usize,
    dir: &Path,
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (train_ids, train_mask) = encode_split(train, tok);
    let (val_ids, val_mask) = encode_split(val, tok);
    binio::write_u32_bin(&dir.join("train.u32.bin"), &train_ids)?;
    binio::write_mask_bin(&dir.join("train.mask.bin"), &train_mask)?;
    binio::write_u32_bin(&dir.join("val.u32.bin"), &val_ids)?;
    binio::write_mask_bin(&dir.join("val.mask.bin"), &val_mask)?;
    std::fs::write(dir.join("meta.json"), binio::Meta::vocab_only(vocab))?;
    Ok(())
}
