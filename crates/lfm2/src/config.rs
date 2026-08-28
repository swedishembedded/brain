// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder configuration + parameter layout.
//!
//! LFM2.5 (LiquidAI) is a hybrid stack: per layer either a **gated short-conv
//! mixer** (`in_proj` H→3H, `Bx = B·x`, depthwise conv1d k=3 with *symmetric*
//! padding, `y = C·conv`, `out_proj`) or **bidirectional GQA attention**
//! (per-head QK-RMSNorm over `head_dim`, RoPE base 1e6 half-split, no biases),
//! each followed by a SwiGLU FFN; RMSNorm pre-LN throughout, tied MLM head.
//! Which mixer a layer uses comes from the checkpoint's `layer_types`.

use serde_json::Value;

/// The mixer a layer runs before its FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Gated depthwise short-conv (`"conv"` in HF `layer_types`).
    Conv,
    /// Bidirectional GQA attention (`"full_attention"`).
    Attention,
}

#[derive(Clone, Debug)]
pub struct LfmConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Effective SwiGLU width (the HF `block_auto_adjust_ff_dim` rule is applied
    /// at import; checkpoints store the resolved value — 230M: 2560, 350M: 4608).
    pub d_ff: u32,
    /// Depthwise conv kernel width (HF `conv_L_cache`; 3 for both models).
    pub conv_k: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub tie_embeddings: bool,
    /// Per-layer mixer, index = layer. Length is the layer count.
    pub layer_types: Vec<LayerType>,
}

/// The HF FFN sizing rule (`Lfm2MLP.__init__`): when `block_auto_adjust_ff_dim`,
/// `d_ff = round_up(int(mult * int(2*d_ff/3)), multiple_of)`.
pub fn adjust_ff_dim(d_ff: u32, auto: bool, mult: f64, multiple_of: u32) -> u32 {
    if !auto {
        return d_ff;
    }
    let inter = (2 * d_ff) / 3; // int(2*x/3): both int-truncations coincide here
    let inter = (mult * inter as f64) as u32;
    multiple_of * inter.div_ceil(multiple_of)
}

impl LfmConfig {
    /// A tiny config for tests / gradient checks: exercises both mixers, GQA
    /// (4 q / 2 kv heads), QK-norm, RoPE-base, SwiGLU and the tied head.
    pub fn tiny() -> LfmConfig {
        LfmConfig {
            vocab: 23,
            block_size: 12,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            d_ff: 32,
            conv_k: 3,
            rope_theta: 1.0e6,
            norm_eps: 1e-5,
            tie_embeddings: true,
            layer_types: vec![LayerType::Conv, LayerType::Attention, LayerType::Conv],
        }
    }

    /// The published LFM2.5-Encoder-230M shape (from its `config.json`).
    pub fn lfm25_encoder_230m() -> LfmConfig {
        use LayerType::{Attention as A, Conv as C};
        LfmConfig {
            vocab: 65536,
            block_size: 8192,
            d_model: 1024,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 64,
            d_ff: 2560,
            conv_k: 3,
            rope_theta: 1.0e6,
            norm_eps: 1e-5,
            tie_embeddings: true,
            layer_types: vec![C, C, A, C, A, C, A, C, A, C, A, C, A, C],
        }
    }

    /// The published LFM2.5-Encoder-350M shape (`d_ff` post auto-adjust: 4608).
    pub fn lfm25_encoder_350m() -> LfmConfig {
        use LayerType::{Attention as A, Conv as C};
        LfmConfig {
            d_ff: 4608,
            layer_types: vec![C, C, A, C, C, A, C, C, A, C, A, C, A, C, A, C],
            ..Self::lfm25_encoder_230m()
        }
    }

    pub fn n_layers(&self) -> u32 {
        self.layer_types.len() as u32
    }
    /// Query projection width = `n_heads * head_dim` (= d_model for LFM2.5).
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    /// Key/Value projection width = `n_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
    /// Query heads per kv head (GQA grouping factor; 2 for LFM2.5).
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }
    /// The mlm head parameter name (tied -> the embedding table).
    pub fn head_weight(&self) -> &'static str {
        if self.tie_embeddings {
            "tok.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub fn to_json(&self) -> Value {
        let layers: Vec<&str> = self
            .layer_types
            .iter()
            .map(|t| match t {
                LayerType::Conv => "conv",
                LayerType::Attention => "full_attention",
            })
            .collect();
        serde_json::json!({
            "model": "lfm",
            "vocab_size": self.vocab, "block_size": self.block_size,
            "d_model": self.d_model, "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim, "d_ff": self.d_ff, "conv_k": self.conv_k,
            "rope_theta": self.rope_theta, "norm_eps": self.norm_eps,
            "tie_word_embeddings": self.tie_embeddings,
            "layer_types": layers,
        })
    }

    /// Every JSON key [`Self::from_json`] must find to read this config's
    /// real SHAPE rather than silently substitute an unrelated hardcoded
    /// default - see that function's own `g`/`gf` closures (a missing
    /// `layer_types` even falls back to a whole DIFFERENT config's field,
    /// `tiny().layer_types`). `tie_word_embeddings` stays optional (a
    /// sensible boolean default), same rationale as
    /// `qwen3::QwenConfig::SHAPE_KEYS`.
    pub const SHAPE_KEYS: &'static [&'static str] =
        &["vocab_size", "block_size", "d_model", "n_heads", "n_kv_heads", "head_dim", "d_ff", "conv_k", "rope_theta", "norm_eps", "layer_types"];

    /// Which of [`Self::SHAPE_KEYS`] `c` is missing.
    pub fn missing_shape_keys(c: &Value) -> Vec<&'static str> {
        Self::SHAPE_KEYS.iter().filter(|k| c.get(**k).is_none()).copied().collect()
    }

    /// [`Self::from_json`], but refuses a config that would silently default
    /// any shape-defining key instead of reading it.
    pub fn from_json_checked(c: &Value) -> Result<LfmConfig, String> {
        let missing = Self::missing_shape_keys(c);
        if !missing.is_empty() {
            return Err(format!(
                "config is missing shape key(s) {missing:?} - from_json would silently substitute an unrelated default for each rather than this checkpoint's real value"
            ));
        }
        Ok(Self::from_json(c))
    }

    pub fn from_json(c: &Value) -> LfmConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let layer_types = c["layer_types"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|t| match t.as_str() {
                        Some("full_attention") => LayerType::Attention,
                        _ => LayerType::Conv,
                    })
                    .collect()
            })
            .unwrap_or_else(|| LfmConfig::tiny().layer_types);
        LfmConfig {
            vocab: g("vocab_size", 23),
            block_size: g("block_size", 12),
            d_model: g("d_model", 16),
            n_heads: g("n_heads", 4),
            n_kv_heads: g("n_kv_heads", 2),
            head_dim: g("head_dim", 4),
            d_ff: g("d_ff", 32),
            conv_k: g("conv_k", 3),
            rope_theta: gf("rope_theta", 1.0e6),
            norm_eps: gf("norm_eps", 1e-5),
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(true),
            layer_types,
        }
    }

    /// Parameter list: `(name, numel)`, in forward order. Conv layers carry the
    /// gated-mixer tensors, attention layers the GQA + QK-norm set; every layer
    /// has two norms and a SwiGLU FFN. Tied checkpoints have no `lm_head`.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let hq = self.q_dim() as usize;
        let hkv = self.kv_dim() as usize;
        let hd = self.head_dim as usize;
        let k = self.conv_k as usize;

        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];
        for (l, ty) in self.layer_types.iter().enumerate() {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            match ty {
                LayerType::Conv => {
                    out.push((p("conv.in_proj.weight"), 3 * d * d));
                    out.push((p("conv.conv.weight"), d * k)); // depthwise [d,1,k]
                    out.push((p("conv.out_proj.weight"), d * d));
                }
                LayerType::Attention => {
                    out.push((p("attn.wq.weight"), hq * d));
                    out.push((p("attn.wk.weight"), hkv * d));
                    out.push((p("attn.wv.weight"), hkv * d));
                    out.push((p("attn.q_norm.weight"), hd));
                    out.push((p("attn.k_norm.weight"), hd));
                    out.push((p("attn.wo.weight"), d * hq));
                }
            }
            out.push((p("ln2.weight"), d));
            out.push((p("mlp.gate.weight"), ff * d));
            out.push((p("mlp.up.weight"), ff * d));
            out.push((p("mlp.down.weight"), d * ff));
        }
        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_checked_accepts_a_real_to_json_round_trip() {
        let c = LfmConfig::tiny().to_json();
        assert!(LfmConfig::missing_shape_keys(&c).is_empty());
        assert!(LfmConfig::from_json_checked(&c).is_ok());
    }

    #[test]
    fn from_json_checked_rejects_a_config_using_the_wrong_key_name() {
        let c = serde_json::json!({"vocab": 23, "block_size": 12, "d_model": 16, "n_heads": 4, "n_kv_heads": 2, "head_dim": 4});
        let err = LfmConfig::from_json_checked(&c).expect_err("missing vocab_size/d_ff/conv_k/rope_theta/norm_eps/layer_types must be refused");
        for key in ["vocab_size", "d_ff", "conv_k", "rope_theta", "norm_eps", "layer_types"] {
            assert!(err.contains(key), "error {err:?} should name the missing key {key:?}");
        }
    }

    #[test]
    fn ff_auto_adjust_matches_hf() {
        // 230M: adjust off -> literal. 350M: 6656 -> int(2*6656/3)=4437 -> *1.0
        // -> round up to 256 -> 4608 (verified against the real checkpoint).
        assert_eq!(adjust_ff_dim(2560, false, 1.0, 256), 2560);
        assert_eq!(adjust_ff_dim(6656, true, 1.0, 256), 4608);
    }

    #[test]
    fn param_list_roundtrip_and_counts() {
        let cfg = LfmConfig::tiny();
        let list = cfg.param_list();
        // Unique names, no zero sizes.
        let mut names: Vec<&String> = list.iter().map(|(n, _)| n).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), list.len());
        assert!(list.iter().all(|(_, n)| *n > 0));
        // JSON roundtrip preserves the layer stack.
        let back = LfmConfig::from_json(&cfg.to_json());
        assert_eq!(back.layer_types, cfg.layer_types);
        assert_eq!(back.d_ff, cfg.d_ff);
        assert_eq!(back.param_list(), list);
    }
}
