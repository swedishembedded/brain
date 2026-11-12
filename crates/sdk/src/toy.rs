// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving training methods on a model small enough to train in seconds,
// for its clients. If your team needs expertise in fast, falsifiable tests of fine-tuning, preference and
// reinforcement-learning pipelines then you can procure our services by sending an email to info@swedishembedded.com.

//! A toy chat checkpoint small enough to train and sample on the CPU backend in seconds, for tests of the training
//! methods that must show an effect (a held-out score that moves) and not only that a step ran.
//!
//! [`write_toy_chat_checkpoint`] writes a randomly initialised `qwen3` checkpoint (2 layers, 32 wide, a 264-token
//! vocabulary) with a byte-level tokenizer and a bare chat template: a user turn is its text followed by `=`, an
//! assistant turn its text followed by `<|im_end|>`, a system turn is dropped, and generation stops at `<|im_end|>`. A prompt `abc=` is answered by whatever the
//! weights make of the tokens after it, so a task whose answer is a function of the prompt (a cipher, say) is learnt by a
//! LoRA adapter of rank 16 in a few hundred steps, and a held-out prompt shows whether it generalised or memorised.
//!
//! The weights are drawn as [`qwen3::init::init_weights`] draws them and then widened ([`WEIGHT_SCALE`]): at the
//! standard 0.02 deviation the tied output embedding is so small that no adapter can lift a logit clear of 264 others
//! in a feasible number of steps.

use std::path::Path;

use crate::{Error, Result};

/// The factor the standard initialisation is widened by, norm gains excepted.
pub const WEIGHT_SCALE: f32 = 6.0;

/// The vocabulary the checkpoint declares: 256 bytes and three specials, rounded up.
const VOCAB: u32 = 264;
const CONTEXT: u32 = 256;

const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}={% elif m.role == 'assistant' %}{{ m.content }}<|im_end|>{% endif %}{% endfor %}";

/// Writes the toy checkpoint into `dir` (created if absent): `model.safetensors`, `tokenizer.json` and
/// `tokenizer_config.json`. The same `seed` writes the same weights.
///
/// # Errors
/// `dir` cannot be written.
pub fn write_toy_chat_checkpoint(dir: &Path, seed: u64) -> Result<()> {
    let io = |e: std::io::Error| Error::Backend(format!("toy checkpoint in {}: {e}", dir.display()));
    std::fs::create_dir_all(dir).map_err(io)?;
    let cfg = qwen3::QwenConfig { vocab: VOCAB, block_size: CONTEXT, max_position_embeddings: CONTEXT, d_model: 32, d_ff: 64, ..qwen3::QwenConfig::tiny() };
    let init = qwen3::init::init_weights(&cfg, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let drawn = &init[&name];
            let weights = if name.contains("norm") { drawn.clone() } else { drawn.iter().map(|w| w * WEIGHT_SCALE).collect() };
            (name, vec![numel as u64], weights)
        })
        .collect();
    let weights = dir.join("model.safetensors");
    let weights = weights.to_str().ok_or_else(|| Error::Backend("the toy checkpoint's path is not UTF-8".into()))?;
    checkpoint::st::save_safetensors(weights, &tensors, &cfg.to_json(), None).map_err(io)?;

    // One token per byte, spelled as GPT-2's byte-to-unicode table spells it.
    let printable: Vec<u32> = (0x21..=0x7e).chain(0xa1..=0xac).chain(0xae..=0xff).collect();
    let mut spare = 0;
    let mut vocab = serde_json::Map::new();
    for byte in 0u32..256 {
        let spelled = if printable.contains(&byte) {
            byte
        } else {
            spare += 1;
            255 + spare
        };
        vocab.insert(char::from_u32(spelled).unwrap_or('?').to_string(), byte.into());
    }
    let specials = serde_json::json!([
        {"content": "<|endoftext|>", "id": 256},
        {"content": "<|im_start|>", "id": 257},
        {"content": "<|im_end|>", "id": 258}
    ]);
    let tokenizer = serde_json::json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": specials});
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).map_err(io)?;
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({"chat_template": TEMPLATE, "eos_token": "<|im_end|>"}).to_string()).map_err(io)
}
