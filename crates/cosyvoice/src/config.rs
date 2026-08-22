// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen2LM` (CosyVoice 2) / `CosyVoice3LM` (CosyVoice 3) speech-token LM
//! configuration: the extra dims/ids CosyVoice bolts onto a stock
//! Qwen2.5-0.5B backbone, plus the `qwen3::QwenConfig` adapter that hosts it -
//! the same "my model's config -> `qwen3::QwenConfig`" seam
//! `qwen3tts::config::TalkerConfig::to_qwen` established.
//!
//! The two generations share the same Qwen2.5-0.5B backbone and the same
//! `sos ++ text ++ task_id ++ prompt_speech_emb` prompt shape, but disagree on
//! ONE real thing (verified against `resources/cosyvoice/source/cosyvoice/
//! llm/llm.py`'s `Qwen2LM`/`CosyVoice3LM` line-by-line, not assumed): where
//! the `sos`/`task_id` embeddings come from. `Qwen2LM` reads them from a
//! small dedicated `llm_embedding: Embedding(2, d)` table; `CosyVoice3LM` has
//! no such table at all - `sos`/`task_id` are just ordinary (if unusual) rows
//! of the SAME `speech_embedding` table every speech token also indexes into
//! (`speech_token_size + 0` and `speech_token_size + 2`), and its
//! `speech_embedding`/`llm_decoder` are correspondingly wider
//! (`speech_token_size + 200`, not `+ 3`, to make room for the mix-ratio
//! bistream instruct-token machinery this port does not implement) and
//! `llm_decoder` carries NO bias (`bias=False`, confirmed absent from the
//! real checkpoint's own tensor names - CosyVoice 2's `llm_decoder` does
//! carry one). [`SpecialTokenSource`] expresses that one branch point without
//! duplicating the ~250 lines of prompt-assembly/generation logic
//! `crate::llm::CosyVoiceLm` hosts for both.

/// Where `sos`/`task_id` embeddings are read from - the one real branch point
/// between `Qwen2LM` (CosyVoice 2) and `CosyVoice3LM` (CosyVoice 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialTokenSource {
    /// `Qwen2LM`: a dedicated `llm_embedding: Embedding(2, d)` table, rows
    /// `[sos, task_id]`.
    LlmEmbedding,
    /// `CosyVoice3LM`: no `llm_embedding` table - `sos`/`task_id` are rows of
    /// the same `speech_embedding` table speech tokens use.
    SpeechEmbedding,
}

/// `Qwen2LM` (CosyVoice 2) / `CosyVoice3LM` (CosyVoice 3) speech-token LM
/// configuration.
#[derive(Clone, Debug)]
pub struct CosyVoiceLmConfig {
    pub llm_input_size: u32,
    pub llm_output_size: u32,
    /// FSQ codebook size (`3^8`), excluding the special ids below.
    pub speech_token_size: u32,
    /// Row index (within [`SpecialTokenSource`]'s table) of the
    /// sequence-start embedding.
    pub sos: u32,
    /// Row index of the task-boundary embedding.
    pub task_id: u32,
    /// `speech_token_size` (CV2) or `speech_token_size + 1` (CV3) - the
    /// `speech_embedding`/`llm_decoder` row that stops generation.
    pub eos_token: u32,
    /// The bistream fill marker (unused by the non-streaming `inference()`
    /// path this crate implements; reserved for `inference_bistream`, not yet
    /// ported).
    pub fill_token: u32,
    /// Any of these stops generation (`{Qwen2LM,CosyVoice3LM}.stop_token_ids`).
    /// CosyVoice 2's is `[eos_token, eos_token+1, fill_token]` (3 entries);
    /// CosyVoice 3's is `speech_token_size..speech_token_size+200` (200
    /// entries) - stored as a range rather than a fixed-size array so both
    /// fit the same field.
    pub stop_token_ids: std::ops::Range<u32>,
    /// Where `sos`/`task_id` are read from - see the module doc.
    pub special_token_source: SpecialTokenSource,
    /// `speech_token_size` + this = the `speech_embedding`/`llm_decoder` row
    /// count (`3` for CV2, `200` for CV3).
    pub speech_vocab_extra: u32,
    /// `llm_decoder` carries a bias (CV2: yes; CV3: no, confirmed absent from
    /// the real checkpoint's own tensor names).
    pub llm_decoder_has_bias: bool,
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
            stop_token_ids: speech_token_size..speech_token_size + 3,
            special_token_source: SpecialTokenSource::LlmEmbedding,
            speech_vocab_extra: 3,
            llm_decoder_has_bias: true,
            qwen: qwen3::QwenConfig::qwen2_0_5b(),
        }
    }

    /// The real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` `CosyVoice3LM`
    /// configuration (`sos = speech_token_size + 0`, `eos_token = +1`,
    /// `task_id = +2`, `fill_token = +3`, all read from `speech_embedding`;
    /// `speech_vocab_extra = 200`, `llm_decoder` bias-free).
    pub fn cosyvoice3() -> CosyVoiceLmConfig {
        let speech_token_size = 6561;
        CosyVoiceLmConfig {
            llm_input_size: 896,
            llm_output_size: 896,
            speech_token_size,
            sos: speech_token_size,
            task_id: speech_token_size + 2,
            eos_token: speech_token_size + 1,
            fill_token: speech_token_size + 3,
            stop_token_ids: speech_token_size..speech_token_size + 200,
            special_token_source: SpecialTokenSource::SpeechEmbedding,
            speech_vocab_extra: 200,
            llm_decoder_has_bias: false,
            qwen: qwen3::QwenConfig::qwen2_0_5b(),
        }
    }

    /// The bolted-on `speech_embedding`/`llm_decoder` row count
    /// (`speech_token_size + speech_vocab_extra`: 6564 for CV2, 6761 for CV3).
    pub fn speech_vocab(&self) -> u32 {
        self.speech_token_size + self.speech_vocab_extra
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
        assert_eq!(cfg.stop_token_ids, 6561..6564);
    }

    #[test]
    fn cosyvoice3_matches_the_real_config_and_dumper_findings() {
        let cfg = CosyVoiceLmConfig::cosyvoice3();
        assert_eq!(cfg.speech_token_size, 6561);
        assert_eq!(cfg.sos, 6561);
        assert_eq!(cfg.eos_token, 6562);
        assert_eq!(cfg.task_id, 6563);
        assert_eq!(cfg.fill_token, 6564);
        assert_eq!(cfg.speech_vocab(), 6761);
        assert_eq!(cfg.special_token_source, SpecialTokenSource::SpeechEmbedding);
        assert!(!cfg.llm_decoder_has_bias);
        assert_eq!(cfg.stop_token_ids, 6561..6761);
        assert!(cfg.stop_token_ids.contains(&cfg.eos_token));
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
