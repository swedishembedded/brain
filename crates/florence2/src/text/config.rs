// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2's BART-style text config - the real `text_config` block of
//! `microsoft/Florence-2-base`'s `config.json`, hardcoded the same way
//! [`crate::vision::DavitConfig::florence2_base`] hardcodes the vision side
//! (it's the model's own trained shape, not a design choice).

#[derive(Clone, Copy, Debug)]
pub struct BartConfig {
    pub d_model: u32,
    pub encoder_layers: u32,
    pub decoder_layers: u32,
    pub encoder_attention_heads: u32,
    pub decoder_attention_heads: u32,
    pub encoder_ffn_dim: u32,
    pub decoder_ffn_dim: u32,
    pub vocab_size: u32,
    /// `Florence2LearnedPositionalEmbedding`'s fixed offset - BART's own
    /// convention, unrelated to `pad_token_id`. Position `p` reads embedding
    /// row `p + POSITION_OFFSET`.
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub decoder_start_token_id: u32,
    pub layer_norm_eps: f32,
}

pub const POSITION_OFFSET: u32 = 2;

impl BartConfig {
    /// `microsoft/Florence-2-base`'s real `text_config`: `d_model=768`,
    /// 6 encoder + 6 decoder layers, 12 heads, `ffn_dim=3072`,
    /// `vocab_size=51289` (base 50265 BART/RoBERTa BPE + the 1024 location/
    /// task tokens - see `crate::tokenizer`), `scale_embedding=false` (no
    /// `sqrt(d_model)` embedding scale, unlike some other BART variants -
    /// verified against the checkpoint's own config, not assumed).
    pub fn florence2_base() -> BartConfig {
        BartConfig {
            d_model: 768,
            encoder_layers: 6,
            decoder_layers: 6,
            encoder_attention_heads: 12,
            decoder_attention_heads: 12,
            encoder_ffn_dim: 3072,
            decoder_ffn_dim: 3072,
            vocab_size: 51289,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            decoder_start_token_id: 2,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn head_dim(&self, heads: u32) -> u32 {
        self.d_model / heads
    }

    /// A small synthetic shape for M6's gradcheck/overfit tests - never the
    /// real checkpoint's dims (those are `florence2_base`'s job). Deliberately
    /// distinct encoder/decoder ffn widths (17/19) and a `d_model`/`heads`
    /// pair that gives a head_dim unlikely to collide with any other axis
    /// (12/3 -> head_dim 4), the same "catch an axis-swap bug the real
    /// checkpoint's coincidental dims would hide" reasoning
    /// `deepseek2::config::DeepseekV2Config::tiny`'s own doc gives.
    pub fn tiny() -> BartConfig {
        BartConfig {
            d_model: 12,
            encoder_layers: 2,
            decoder_layers: 2,
            encoder_attention_heads: 3,
            decoder_attention_heads: 3,
            encoder_ffn_dim: 17,
            decoder_ffn_dim: 19,
            vocab_size: 23,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            decoder_start_token_id: 2,
            layer_norm_eps: 1e-5,
        }
    }
}
