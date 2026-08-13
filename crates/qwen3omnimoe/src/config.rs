// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni-30B-A3B configuration, parsed from the released
//! `config.json` (`Qwen/Qwen3-Omni-30B-A3B-Instruct`).
//!
//! The nesting mirrors the checkpoint exactly: `thinker_config.{audio_config,
//! vision_config, text_config}`, `talker_config.{text_config,
//! code_predictor_config}`, `code2wav_config`. Every default below is the
//! REAL released value (dumped 2026-08-07),
//! not a guess: this file is the single place those numbers are
//! recorded as code, everything else derives from it.
//!
//! `code_predictor_config` is parsed by `qwen3tts::config::MtpConfig::from_json`
//! unchanged — Omni's code predictor is a 5-layer/16-codebook block, the same
//! shape `crates/tts` already models, at `talker_config.code_predictor_config`,
//! the exact path that parser already reads.

use serde_json::Value;

fn gu(o: &Value, k: &str, d: u32) -> u32 {
    o[k].as_u64().map(|x| x as u32).unwrap_or(d)
}
fn gf(o: &Value, k: &str, d: f32) -> f32 {
    o[k].as_f64().map(|x| x as f32).unwrap_or(d)
}
fn gb(o: &Value, k: &str, d: bool) -> bool {
    o[k].as_bool().unwrap_or(d)
}
fn mrope_section(o: &Value) -> Vec<u32> {
    o["rope_scaling"]["mrope_section"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect())
        .unwrap_or_else(|| vec![24, 20, 20])
}

/// `thinker_config.audio_config` — the AuT (audio tower) encoder. Same shape
/// `qwen3asr::config::AudioEncoderConfig` already models
/// (`conv2d1/2/3 -> conv_out -> proj1/2` stem, windowed transformer), at
/// Omni's larger scale.
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub num_mel_bins: u32,        // 128
    pub d_model: u32,             // 1280
    pub n_heads: u32,             // 20 (head_dim = d_model/n_heads = 64)
    pub ffn_dim: u32,             // 5120
    pub n_layers: u32,            // 32
    pub downsample_hidden: u32,   // 480
    pub output_dim: u32,          // 2048 (== thinker text hidden)
    pub n_window: u32,            // 50 (chunk = 2*n_window mel frames)
    pub n_window_infer: u32,      // 800
    pub eps: f32,                 // 1e-5 (LayerNorm)
}

impl AudioConfig {
    pub fn from_json(root: &Value) -> AudioConfig {
        let a = &root["thinker_config"]["audio_config"];
        AudioConfig {
            num_mel_bins: gu(a, "num_mel_bins", 128),
            d_model: gu(a, "d_model", 1280),
            n_heads: gu(a, "encoder_attention_heads", 20),
            ffn_dim: gu(a, "encoder_ffn_dim", 5120),
            n_layers: gu(a, "encoder_layers", 32),
            downsample_hidden: gu(a, "downsample_hidden_size", 480),
            output_dim: gu(a, "output_dim", 2048),
            n_window: gu(a, "n_window", 50),
            n_window_infer: gu(a, "n_window_infer", 800),
            eps: 1e-5,
        }
    }
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
    pub fn chunk_len(&self) -> u32 {
        2 * self.n_window
    }
}

/// `thinker_config.vision_config` — the ViT (vision tower). Same shape
/// `qwen3vl::config::VisionConfig` already models (PatchMerger + DeepStack),
/// at Omni's scale, plus `apply_vit_abs_pos_embed` and `gelu_pytorch_tanh`
/// (Qwen3-VL-4B uses `gelu` variants without the abs-pos-embed flag; Omni's
/// vision tower needs both).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: u32,                    // 27
    pub hidden: u32,                   // 1152
    pub num_heads: u32,                // 16 (head_dim = 72)
    pub intermediate: u32,             // 4304
    pub patch_size: u32,               // 16
    pub temporal_patch_size: u32,      // 2
    pub spatial_merge_size: u32,       // 2
    pub out_hidden_size: u32,          // 2048
    pub in_channels: u32,              // 3
    pub deepstack_indexes: Vec<u32>,   // [8,16,24]
    pub apply_vit_abs_pos_embed: bool, // true
    pub tokens_per_second: u32,        // 2 (video temporal sampling)
}

impl VisionConfig {
    pub fn from_json(root: &Value) -> VisionConfig {
        let v = &root["thinker_config"]["vision_config"];
        let deepstack = v["deepstack_visual_indexes"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect())
            .unwrap_or_else(|| vec![8, 16, 24]);
        VisionConfig {
            depth: gu(v, "depth", 27),
            hidden: gu(v, "hidden_size", 1152),
            num_heads: gu(v, "num_heads", 16),
            intermediate: gu(v, "intermediate_size", 4304),
            patch_size: gu(v, "patch_size", 16),
            temporal_patch_size: gu(v, "temporal_patch_size", 2),
            spatial_merge_size: gu(v, "spatial_merge_size", 2),
            out_hidden_size: gu(v, "out_hidden_size", 2048),
            in_channels: gu(v, "in_channels", 3),
            deepstack_indexes: deepstack,
            apply_vit_abs_pos_embed: gb(v, "apply_vit_abs_pos_embed", true),
            tokens_per_second: gu(v, "tokens_per_second", 2),
        }
    }
    pub fn head_dim(&self) -> u32 {
        self.hidden / self.num_heads
    }
    pub fn merge_unit(&self) -> u32 {
        self.spatial_merge_size * self.spatial_merge_size
    }
    pub fn patch_vec_dim(&self) -> u32 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// A Qwen3-MoE text decoder config — shared shape for the Thinker's
/// `thinker_config.text_config` (48L/128 experts top-8/no shared expert) and
/// the Talker's `talker_config.text_config` (20L/128 experts top-6/shared
/// expert 768). Both are plain-softmax top-k routers
/// (`model::moe::router_fwd`/`router_gate.wgsl`), not glm's sigmoid/bias/
/// group-limited variant.
#[derive(Clone, Debug)]
pub struct MoeTextConfig {
    pub n_layers: u32,
    pub hidden: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub moe_intermediate: u32,
    pub shared_expert_intermediate: u32, // 0 = no shared expert (Thinker)
    pub n_experts: u32,
    pub top_k: u32,
    pub norm_topk_prob: bool,
    pub use_qk_norm: bool,
    pub vocab: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub mrope_section: Vec<u32>, // [24,20,20] interleaved
    pub max_position_embeddings: u32,
}

impl MoeTextConfig {
    fn from_value(t: &Value, defaults: MoeTextConfig) -> MoeTextConfig {
        MoeTextConfig {
            n_layers: gu(t, "num_hidden_layers", defaults.n_layers),
            hidden: gu(t, "hidden_size", defaults.hidden),
            n_heads: gu(t, "num_attention_heads", defaults.n_heads),
            n_kv_heads: gu(t, "num_key_value_heads", defaults.n_kv_heads),
            head_dim: gu(t, "head_dim", defaults.head_dim),
            moe_intermediate: gu(t, "moe_intermediate_size", defaults.moe_intermediate),
            shared_expert_intermediate: gu(t, "shared_expert_intermediate_size", defaults.shared_expert_intermediate),
            n_experts: gu(t, "num_experts", defaults.n_experts),
            top_k: gu(t, "num_experts_per_tok", defaults.top_k),
            norm_topk_prob: gb(t, "norm_topk_prob", defaults.norm_topk_prob),
            use_qk_norm: gb(t, "use_qk_norm", defaults.use_qk_norm),
            vocab: gu(t, "vocab_size", defaults.vocab),
            rope_theta: gf(t, "rope_theta", defaults.rope_theta),
            rms_norm_eps: gf(t, "rms_norm_eps", defaults.rms_norm_eps),
            mrope_section: if t.get("rope_scaling").is_some() { mrope_section(t) } else { defaults.mrope_section },
            max_position_embeddings: gu(t, "max_position_embeddings", defaults.max_position_embeddings),
        }
    }

    /// `thinker_config.text_config` — 48L, hidden 2048, GQA 32/4, no shared
    /// expert, `use_qk_norm`, vocab 152064, theta 1e6.
    pub fn thinker_defaults() -> MoeTextConfig {
        MoeTextConfig {
            n_layers: 48,
            hidden: 2048,
            n_heads: 32,
            n_kv_heads: 4,
            head_dim: 128,
            moe_intermediate: 768,
            shared_expert_intermediate: 0,
            n_experts: 128,
            top_k: 8,
            norm_topk_prob: true,
            use_qk_norm: true,
            vocab: 152064,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            mrope_section: vec![24, 20, 20],
            max_position_embeddings: 65536,
        }
    }

    /// `talker_config.text_config` — 20L, hidden 1024, GQA 16/2, shared
    /// expert 768, vocab 3072 (codec ids), theta 1e6.
    ///
    /// `use_qk_norm: true` despite `talker_config.text_config` never setting
    /// a `use_qk_norm` JSON key (so `gb(t, "use_qk_norm", ...)` would fall
    /// through to this default either way): the Talker's decoder layer
    /// reuses `Qwen3OmniMoeThinkerTextAttention` verbatim, whose `q_norm`/
    /// `k_norm` are unconditional -- no config flag gates them at all, and
    /// the real checkpoint carries `talker.model.layers.*.self_attn.
    /// {q,k}_norm.weight` for every layer. `false` here would silently skip
    /// a real, weighted normalization step.
    pub fn talker_defaults() -> MoeTextConfig {
        MoeTextConfig {
            n_layers: 20,
            hidden: 1024,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 128,
            moe_intermediate: 384,
            shared_expert_intermediate: 768,
            n_experts: 128,
            top_k: 6,
            norm_topk_prob: true,
            use_qk_norm: true,
            vocab: 3072,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            mrope_section: vec![24, 20, 20],
            max_position_embeddings: 65536,
        }
    }

    pub fn thinker_from_json(root: &Value) -> MoeTextConfig {
        Self::from_value(&root["thinker_config"]["text_config"], Self::thinker_defaults())
    }
    pub fn talker_from_json(root: &Value) -> MoeTextConfig {
        Self::from_value(&root["talker_config"]["text_config"], Self::talker_defaults())
    }

    pub fn has_shared_expert(&self) -> bool {
        self.shared_expert_intermediate > 0
    }
    pub fn moe_shape(&self, rows: u32) -> model::moe::MoeShape {
        model::moe::MoeShape {
            rows,
            d_model: self.hidden,
            moe_ff: self.moe_intermediate,
            n_experts: self.n_experts,
            top_k: self.top_k,
        }
    }
}

/// `thinker_config`'s own scalar fields (special-token ids, cross-modal
/// timing) that sit alongside `audio_config`/`vision_config`/`text_config`
/// rather than inside any of them.
#[derive(Clone, Debug)]
pub struct ThinkerConfig {
    pub audio: AudioConfig,
    pub vision: VisionConfig,
    pub text: MoeTextConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub position_id_per_seconds: u32, // 13
    pub seconds_per_chunk: u32,       // 2
}

impl ThinkerConfig {
    /// The real released defaults (see this module's doc), with no
    /// `config.json` to read from - every field of [`Self::from_json`] falls
    /// back to its default when the corresponding key is absent, so parsing
    /// `Value::Null` (indexing it with anything yields `Value::Null` again,
    /// never a panic) produces exactly the same numbers
    /// `MoeTextConfig::thinker_defaults()` hand-rolls for its own field
    /// subset, without a second hand-synced copy of the special-token ids
    /// (`audio_token_id` etc.) that `crate::mm::build_multimodal_prompt`
    /// needs - a caller with no checkpoint directory on hand yet (e.g. an
    /// int8-only deployment, whose checkpoint is a single `.safetensors`
    /// with no `config.json` sibling) still gets a real, complete config.
    pub fn defaults() -> ThinkerConfig {
        Self::from_json(&Value::Null)
    }

    /// `self` with `text` replaced - a caller that already has a
    /// `MoeTextConfig` (real, imported from `config.json`, or a test's tiny
    /// synthetic shape) and just needs it wrapped in a full `ThinkerConfig`
    /// (special media token ids etc.) rather than re-deriving one field at a
    /// time.
    pub fn with_text(mut self, text: MoeTextConfig) -> ThinkerConfig {
        self.text = text;
        self
    }

    pub fn from_json(root: &Value) -> ThinkerConfig {
        let t = &root["thinker_config"];
        ThinkerConfig {
            audio: AudioConfig::from_json(root),
            vision: VisionConfig::from_json(root),
            text: MoeTextConfig::thinker_from_json(root),
            audio_start_token_id: gu(t, "audio_start_token_id", 151669),
            audio_end_token_id: gu(t, "audio_end_token_id", 151670),
            audio_token_id: gu(t, "audio_token_id", 151675),
            image_token_id: gu(t, "image_token_id", 151655),
            video_token_id: gu(t, "video_token_id", 151656),
            vision_start_token_id: gu(t, "vision_start_token_id", 151652),
            vision_end_token_id: gu(t, "vision_end_token_id", 151653),
            position_id_per_seconds: gu(t, "position_id_per_seconds", 13),
            seconds_per_chunk: gu(t, "seconds_per_chunk", 2),
        }
    }
}

/// `talker_config`'s own scalar fields, alongside `text_config` and
/// `code_predictor_config`.
#[derive(Clone, Debug)]
pub struct TalkerConfig {
    pub text: MoeTextConfig,
    pub code_predictor: qwen3tts::config::MtpConfig,
    pub accept_hidden_layer: u32, // 24 — which Thinker decoder layer's hidden state the Talker consumes
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub num_code_groups: u32, // 16
    pub thinker_hidden_size: u32, // 2048 — width of the hidden state accept_hidden_layer reads
    pub spatial_merge_size: u32,
    pub position_id_per_seconds: u32,
    pub seconds_per_chunk: u32,
    /// name -> codec speaker id, e.g. `{"chelsie": 2301, "ethan": 2302, "aiden": 2303}`.
    pub speaker_id: std::collections::BTreeMap<String, u32>,
}

impl TalkerConfig {
    pub fn from_json(root: &Value) -> TalkerConfig {
        let t = &root["talker_config"];
        let speaker_id = t["speaker_id"]
            .as_object()
            .map(|m| m.iter().filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32))).collect())
            .unwrap_or_else(|| {
                [("chelsie".to_string(), 2301), ("ethan".to_string(), 2302), ("aiden".to_string(), 2303)].into()
            });
        TalkerConfig {
            text: MoeTextConfig::talker_from_json(root),
            code_predictor: qwen3tts::config::MtpConfig::from_json(t),
            accept_hidden_layer: gu(t, "accept_hidden_layer", 24),
            audio_start_token_id: gu(t, "audio_start_token_id", 151669),
            audio_end_token_id: gu(t, "audio_end_token_id", 151670),
            audio_token_id: gu(t, "audio_token_id", 151675),
            image_token_id: gu(t, "image_token_id", 151655),
            video_token_id: gu(t, "video_token_id", 151656),
            vision_start_token_id: gu(t, "vision_start_token_id", 151652),
            codec_bos_id: gu(t, "codec_bos_id", 2149),
            codec_eos_token_id: gu(t, "codec_eos_token_id", 2150),
            codec_pad_id: gu(t, "codec_pad_id", 2148),
            codec_nothink_id: gu(t, "codec_nothink_id", 2155),
            codec_think_bos_id: gu(t, "codec_think_bos_id", 2156),
            codec_think_eos_id: gu(t, "codec_think_eos_id", 2157),
            num_code_groups: gu(t, "num_code_groups", 16),
            thinker_hidden_size: gu(t, "thinker_hidden_size", 2048),
            spatial_merge_size: gu(t, "spatial_merge_size", 2),
            position_id_per_seconds: gu(t, "position_id_per_seconds", 13),
            seconds_per_chunk: gu(t, "seconds_per_chunk", 2),
            speaker_id,
        }
    }
}

/// `code2wav_config` — RVQ decode + SEANet vocoder. Same shape
/// `mimi::config::CodecConfig` already models (Qwen3-TTS-12Hz codec decode
/// path), extended for Omni's wider pre-transformer (`hidden_size` 1024 vs
/// 512, `intermediate_size` 3072 vs 1024) and mean-pooled multi-codebook
/// input (`code_embedding` over `codebook_size * num_quantizers`, summed/
/// averaged across quantizers) rather than RVQ dequant.
#[derive(Clone, Debug)]
pub struct Code2WavConfig {
    pub num_quantizers: u32,            // 16 (1 semantic + 15 acoustic)
    pub num_semantic_quantizers: u32,   // 1
    pub codebook_size: u32,             // 2048 (acoustic)
    pub semantic_codebook_size: u32,    // 4096
    pub codebook_dim: u32,              // 512 (vector_quantization_hidden_dimension)
    pub hidden_size: u32,               // 1024 (pre-transformer + code_embedding width)
    pub intermediate_size: u32,         // 3072
    pub num_hidden_layers: u32,         // 8
    pub num_attention_heads: u32,       // 16
    pub num_key_value_heads: u32,       // 16
    pub sliding_window: u32,            // 72
    pub rope_theta: f32,                // 10000
    pub rms_norm_eps: f32,              // 1e-5
    pub layer_scale_initial_scale: f32, // 0.01
    pub decoder_dim: u32,               // 1536
    pub upsample_rates: Vec<u32>,       // [8,5,4,3]
    pub upsampling_ratios: Vec<u32>,    // [2,2]
    pub max_position_embeddings: u32,   // 8000
    pub output_sample_rate: u32,        // 24000
}

impl Code2WavConfig {
    pub fn from_json(root: &Value) -> Code2WavConfig {
        let c = &root["code2wav_config"];
        let vecu = |k: &str, d: &[u32]| -> Vec<u32> {
            c[k].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect())
                .unwrap_or_else(|| d.to_vec())
        };
        Code2WavConfig {
            num_quantizers: gu(c, "num_quantizers", 16),
            num_semantic_quantizers: gu(c, "num_semantic_quantizers", 1),
            codebook_size: gu(c, "codebook_size", 2048),
            semantic_codebook_size: gu(c, "semantic_codebook_size", 4096),
            codebook_dim: gu(c, "vector_quantization_hidden_dimension", 512),
            hidden_size: gu(c, "hidden_size", 1024),
            intermediate_size: gu(c, "intermediate_size", 3072),
            num_hidden_layers: gu(c, "num_hidden_layers", 8),
            num_attention_heads: gu(c, "num_attention_heads", 16),
            num_key_value_heads: gu(c, "num_key_value_heads", 16),
            sliding_window: gu(c, "sliding_window", 72),
            rope_theta: gf(c, "rope_theta", 10000.0),
            rms_norm_eps: gf(c, "rms_norm_eps", 1e-5),
            layer_scale_initial_scale: gf(c, "layer_scale_initial_scale", 0.01),
            decoder_dim: gu(c, "decoder_dim", 1536),
            upsample_rates: vecu("upsample_rates", &[8, 5, 4, 3]),
            upsampling_ratios: vecu("upsampling_ratios", &[2, 2]),
            max_position_embeddings: gu(c, "max_position_embeddings", 8000),
            output_sample_rate: 24000,
        }
    }
    /// Total temporal upsample from one code frame to output samples
    /// (`prod(upsample_rates) * prod(upsampling_ratios)` = 8*5*4*3*2*2 = 1920,
    /// i.e. 12.5 Hz code rate at 24 kHz output).
    pub fn total_upsample(&self) -> u32 {
        self.upsample_rates.iter().chain(self.upsampling_ratios.iter()).product()
    }
}

/// The full Qwen3-Omni-30B-A3B configuration.
#[derive(Clone, Debug)]
pub struct OmniConfig {
    pub thinker: ThinkerConfig,
    pub talker: TalkerConfig,
    pub code2wav: Code2WavConfig,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub assistant_token_id: u32,
    pub system_token_id: u32,
    pub user_token_id: u32,
    pub enable_audio_output: bool,
    /// Top-level (NOT under `talker_config`) — `Qwen3OmniMoeForConditionalGeneration
    /// .generate`'s own `self.config.tts_{bos,eos,pad}_token_id` (real
    /// checkpoint values: 151672/151673/151671), the ids whose
    /// `talker.text_projection`-projected embeddings frame the Talker
    /// prefill's assistant-text segment (`crate::talker_prompt`'s doc).
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
}

impl OmniConfig {
    pub fn from_json(root: &Value) -> OmniConfig {
        OmniConfig {
            thinker: ThinkerConfig::from_json(root),
            talker: TalkerConfig::from_json(root),
            code2wav: Code2WavConfig::from_json(root),
            im_start_token_id: gu(root, "im_start_token_id", 151644),
            im_end_token_id: gu(root, "im_end_token_id", 151645),
            assistant_token_id: gu(root, "assistant_token_id", 77091),
            system_token_id: gu(root, "system_token_id", 8948),
            user_token_id: gu(&root["thinker_config"], "user_token_id", 872),
            enable_audio_output: gb(root, "enable_audio_output", true),
            tts_bos_token_id: gu(root, "tts_bos_token_id", 151672),
            tts_eos_token_id: gu(root, "tts_eos_token_id", 151673),
            tts_pad_token_id: gu(root, "tts_pad_token_id", 151671),
        }
    }

    pub fn parse(json: &str) -> Result<OmniConfig, serde_json::Error> {
        Ok(OmniConfig::from_json(&serde_json::from_str(json)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed but structurally faithful inline copy of the released
    /// `config.json` — cross-checks the parser against the real nesting and
    /// the real released numbers, without
    /// needing the (70.5 GB) checkpoint present to run `cargo test`.
    const SAMPLE: &str = r#"{
        "im_start_token_id": 151644, "im_end_token_id": 151645,
        "assistant_token_id": 77091, "system_token_id": 8948,
        "enable_audio_output": true,
        "tts_bos_token_id": 151672, "tts_eos_token_id": 151673, "tts_pad_token_id": 151671,
        "code2wav_config": {
            "num_quantizers": 16, "num_semantic_quantizers": 1,
            "codebook_size": 2048, "semantic_codebook_size": 4096,
            "vector_quantization_hidden_dimension": 512,
            "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 8, "num_attention_heads": 16, "num_key_value_heads": 16,
            "sliding_window": 72, "rope_theta": 10000, "rms_norm_eps": 1e-5,
            "layer_scale_initial_scale": 0.01, "decoder_dim": 1536,
            "upsample_rates": [8, 5, 4, 3], "upsampling_ratios": [2, 2],
            "max_position_embeddings": 8000
        },
        "talker_config": {
            "accept_hidden_layer": 24, "audio_end_token_id": 151670,
            "audio_start_token_id": 151669, "audio_token_id": 151675,
            "image_token_id": 151655, "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "codec_bos_id": 2149, "codec_eos_token_id": 2150, "codec_pad_id": 2148,
            "codec_nothink_id": 2155, "codec_think_bos_id": 2156, "codec_think_eos_id": 2157,
            "num_code_groups": 16, "thinker_hidden_size": 2048, "spatial_merge_size": 2,
            "position_id_per_seconds": 13, "seconds_per_chunk": 2,
            "speaker_id": {"chelsie": 2301, "ethan": 2302, "aiden": 2303},
            "text_config": {
                "num_hidden_layers": 20, "hidden_size": 1024,
                "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
                "moe_intermediate_size": 384, "shared_expert_intermediate_size": 768,
                "num_experts": 128, "num_experts_per_tok": 6, "norm_topk_prob": true,
                "vocab_size": 3072, "rope_theta": 1000000,
                "rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": true},
                "max_position_embeddings": 65536
            },
            "code_predictor_config": {
                "num_hidden_layers": 5, "hidden_size": 1024, "head_dim": 128,
                "num_attention_heads": 16, "num_key_value_heads": 8, "intermediate_size": 3072,
                "vocab_size": 2048, "num_code_groups": 16,
                "rope_theta": 1000000, "rms_norm_eps": 1e-6
            }
        },
        "thinker_config": {
            "audio_start_token_id": 151669, "audio_end_token_id": 151670,
            "audio_token_id": 151675, "image_token_id": 151655, "video_token_id": 151656,
            "vision_start_token_id": 151652, "vision_end_token_id": 151653,
            "position_id_per_seconds": 13, "seconds_per_chunk": 2, "user_token_id": 872,
            "audio_config": {
                "num_mel_bins": 128, "d_model": 1280, "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120, "encoder_layers": 32, "downsample_hidden_size": 480,
                "output_dim": 2048, "n_window": 50, "n_window_infer": 800
            },
            "vision_config": {
                "depth": 27, "hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304,
                "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "out_hidden_size": 2048, "in_channels": 3,
                "deepstack_visual_indexes": [8, 16, 24],
                "apply_vit_abs_pos_embed": true, "tokens_per_second": 2
            },
            "text_config": {
                "num_hidden_layers": 48, "hidden_size": 2048,
                "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
                "moe_intermediate_size": 768, "shared_expert_intermediate_size": 0,
                "num_experts": 128, "num_experts_per_tok": 8, "norm_topk_prob": true,
                "use_qk_norm": true, "vocab_size": 152064, "rope_theta": 1000000,
                "rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": true},
                "max_position_embeddings": 65536
            }
        }
    }"#;

    #[test]
    fn parses_the_real_shape() {
        let c = OmniConfig::parse(SAMPLE).expect("parse");

        assert_eq!(c.tts_bos_token_id, 151672);
        assert_eq!(c.tts_eos_token_id, 151673);
        assert_eq!(c.tts_pad_token_id, 151671);

        assert_eq!(c.thinker.audio.n_layers, 32);
        assert_eq!(c.thinker.audio.d_model, 1280);
        assert_eq!(c.thinker.audio.head_dim(), 64);
        assert_eq!(c.thinker.audio.chunk_len(), 100);

        assert_eq!(c.thinker.vision.depth, 27);
        assert_eq!(c.thinker.vision.deepstack_indexes, vec![8, 16, 24]);
        assert_eq!(c.thinker.vision.head_dim(), 72);
        assert_eq!(c.thinker.vision.merge_unit(), 4);
        assert_eq!(c.thinker.vision.patch_vec_dim(), 3 * 2 * 16 * 16);

        assert_eq!(c.thinker.text.n_layers, 48);
        assert_eq!(c.thinker.text.n_experts, 128);
        assert_eq!(c.thinker.text.top_k, 8);
        assert!(!c.thinker.text.has_shared_expert());
        assert_eq!(c.thinker.text.mrope_section, vec![24, 20, 20]);

        assert_eq!(c.talker.text.n_layers, 20);
        assert_eq!(c.talker.text.n_experts, 128);
        assert_eq!(c.talker.text.top_k, 6);
        assert!(c.talker.text.has_shared_expert());
        assert_eq!(c.talker.text.shared_expert_intermediate, 768);
        // Reuses Qwen3OmniMoeThinkerTextAttention verbatim, whose q_norm/
        // k_norm are unconditional -- real weights confirm this (see
        // talker_defaults()'s doc comment).
        assert!(c.talker.text.use_qk_norm);
        assert_eq!(c.talker.accept_hidden_layer, 24);
        assert_eq!(c.talker.speaker_id.get("chelsie"), Some(&2301));
        assert_eq!(c.talker.speaker_id.get("ethan"), Some(&2302));
        assert_eq!(c.talker.speaker_id.get("aiden"), Some(&2303));

        assert_eq!(c.talker.code_predictor.n_layers, 5);
        assert_eq!(c.talker.code_predictor.num_code_groups, 16);
        assert_eq!(c.talker.code_predictor.vocab, 2048);
        assert_eq!(c.talker.code_predictor.n_residual(), 15);

        assert_eq!(c.code2wav.num_quantizers, 16);
        assert_eq!(c.code2wav.num_semantic_quantizers, 1);
        assert_eq!(c.code2wav.hidden_size, 1024);
        assert_eq!(c.code2wav.upsample_rates, vec![8, 5, 4, 3]);
        assert_eq!(c.code2wav.upsampling_ratios, vec![2, 2]);
        assert_eq!(c.code2wav.total_upsample(), 1920);

        assert_eq!(c.im_start_token_id, 151644);
        assert_eq!(c.assistant_token_id, 77091);
        assert!(c.enable_audio_output);
    }

    #[test]
    fn moe_shape_matches_model_moe() {
        let c = OmniConfig::parse(SAMPLE).expect("parse");
        let shape = c.thinker.text.moe_shape(4);
        assert_eq!(shape.rows, 4);
        assert_eq!(shape.d_model, 2048);
        assert_eq!(shape.moe_ff, 768);
        assert_eq!(shape.n_experts, 128);
        assert_eq!(shape.top_k, 8);
    }
}
