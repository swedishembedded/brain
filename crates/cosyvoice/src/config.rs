// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen2LM` (CosyVoice 2) speech-token LM configuration: the extra dims/ids
//! CosyVoice bolts onto a stock Qwen2.5-0.5B backbone, plus the
//! `qwen3::QwenConfig` adapter that hosts it - the same "my model's config ->
//! `qwen3::QwenConfig`" seam `qwen3tts::config::TalkerConfig::to_qwen`
//! established.
//!
//! CosyVoice 3's `CosyVoice3LM` (`sos`/`task_id` drawn from `speech_embedding`
//! instead of a separate `llm_embedding` table, a wider `speech_token_size +
//! 200` untied head, and a mandatory `<|endofprompt|>` marker in the text) is
//! a deliberate follow-up, **not** implemented here - see
//! `resources/cosyvoice/source/cosyvoice/llm/llm.py`'s `CosyVoice3LM` for the
//! delta.

/// `Qwen2LM` (CosyVoice 2) speech-token LM configuration.
#[derive(Clone, Debug)]
pub struct CosyVoiceLmConfig {
    pub llm_input_size: u32,
    pub llm_output_size: u32,
    /// FSQ codebook size (`3^8`), excluding the 3 special ids below.
    pub speech_token_size: u32,
    /// `llm_embedding` row for the sequence-start embedding.
    pub sos: u32,
    /// `llm_embedding` row for the task-boundary embedding.
    pub task_id: u32,
    /// `speech_token_size` - the first of the 3 special ids appended to the
    /// `speech_embedding`/`llm_decoder` vocab.
    pub eos_token: u32,
    /// `speech_token_size + 2` - the bistream fill marker (unused by the
    /// non-streaming `inference()` path this crate implements; reserved for
    /// `inference_bistream`, not yet ported).
    pub fill_token: u32,
    /// `{eos_token, eos_token+1, fill_token}` - any of these stops generation
    /// (`Qwen2LM.stop_token_ids`).
    pub stop_token_ids: [u32; 3],
    /// The hosted Qwen2.5-0.5B backbone (verified against the real
    /// `CosyVoice-BlankEN/config.json`: `hidden=896, layers=24, heads=14,
    /// kv_heads=2, head_dim=64, d_ff=4864, rope_theta=1e6, rms_eps=1e-6,
    /// vocab=151936, tie_embeddings=true`).
    pub qwen: qwen3::QwenConfig,
}

impl CosyVoiceLmConfig {
    /// The real `FunAudioLLM/CosyVoice2-0.5B` `Qwen2LM` configuration.
    pub fn cosyvoice2() -> CosyVoiceLmConfig {
        let speech_token_size = 6561;
        CosyVoiceLmConfig {
            llm_input_size: 896,
            llm_output_size: 896,
            speech_token_size,
            sos: 0,
            task_id: 1,
            eos_token: speech_token_size,
            fill_token: speech_token_size + 2,
            stop_token_ids: [speech_token_size, speech_token_size + 1, speech_token_size + 2],
            qwen: qwen3::QwenConfig::qwen2_0_5b(),
        }
    }

    /// The bolted-on `speech_embedding`/`llm_decoder` row count:
    /// `speech_token_size + 3` (6564 for the real model).
    pub fn speech_vocab(&self) -> u32 {
        self.speech_token_size + 3
    }

    /// A `qwen3::QwenConfig` sized for a `block_size`/`max_position_embeddings`
    /// of `ctx` - the prefill prefix length plus the AR decode budget this
    /// instance is built to serve. All other fields (layer/head/ff dims, RoPE,
    /// tied embeddings) come from `self.qwen` unchanged.
    pub fn to_qwen_config(&self, ctx: u32) -> qwen3::QwenConfig {
        let mut q = self.qwen.clone();
        q.block_size = ctx;
        q.max_position_embeddings = ctx;
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosyvoice2_matches_the_real_config_json() {
        let cfg = CosyVoiceLmConfig::cosyvoice2();
        assert_eq!(cfg.qwen.d_model, 896);
        assert_eq!(cfg.qwen.n_layers, 24);
        assert_eq!(cfg.qwen.n_heads, 14);
        assert_eq!(cfg.qwen.n_kv_heads, 2);
        assert_eq!(cfg.qwen.head_dim, 64);
        assert_eq!(cfg.qwen.d_ff, 4864);
        assert_eq!(cfg.qwen.rope_theta, 1.0e6);
        assert_eq!(cfg.qwen.rms_eps, 1e-6);
        assert_eq!(cfg.qwen.vocab, 151936);
        assert!(cfg.qwen.tie_embeddings);
        assert!(!cfg.qwen.qk_norm); // Qwen2, not Qwen3
        assert!(cfg.qwen.attn_bias); // Qwen2 q/k/v carry a bias

        assert_eq!(cfg.speech_token_size, 6561);
        assert_eq!(cfg.speech_vocab(), 6564);
        assert_eq!(cfg.eos_token, 6561);
        assert_eq!(cfg.fill_token, 6563);
        assert_eq!(cfg.stop_token_ids, [6561, 6562, 6563]);
    }

    #[test]
    fn to_qwen_config_overrides_only_context_size() {
        let cfg = CosyVoiceLmConfig::cosyvoice2();
        let q = cfg.to_qwen_config(256);
        assert_eq!(q.block_size, 256);
        assert_eq!(q.max_position_embeddings, 256);
        assert_eq!(q.d_model, cfg.qwen.d_model);
        assert_eq!(q.n_layers, cfg.qwen.n_layers);
    }
}
