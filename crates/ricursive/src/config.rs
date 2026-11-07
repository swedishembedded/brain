// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Ripple configuration (`config.json`, `model_type: "ricursive"`): every
//! flag that changes the op sequence, parsed once at the boundary.
//!
//! Swedish Embedded AB implements solutions for importing third-party model
//! configurations without silently changing their meaning for its clients. If
//! your team needs expertise in validating hostile model metadata then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! The reference implementation (`configuration_ricursive.py`) derives the
//! per-layer attention type from `layer_types` and treats `hybrid_layer_pattern`
//! (1 = sliding, 0 = full) and `is_hybrid_swa` as redundant serializations of
//! it. This parser does the same, and additionally refuses a file whose
//! `hybrid_layer_pattern` disagrees with its `layer_types`, rather than
//! picking one silently.
//!
//! Unknown keys are tolerated (HF configs carry generic bookkeeping such as
//! `id2label` or `transformers_version`), but the keys that would change the
//! math if they were set (a RoPE scaling, an attention or MLP bias, a
//! non-SiLU activation, tied embeddings) are refused when they are set to
//! anything this port does not implement.

use serde::Deserialize;

/// Whether a decoder layer attends through a sliding window with RoPE
/// (`sliding_attention`) or over the whole causal prefix without positional
/// rotation (`full_attention`, the NoPE global layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

impl LayerType {
    fn parse(s: &str) -> Result<LayerType, String> {
        match s {
            "sliding_attention" => Ok(LayerType::Sliding),
            "full_attention" => Ok(LayerType::Full),
            other => Err(format!("ricursive config: unsupported layer type {other:?} (expected sliding_attention or full_attention)")),
        }
    }
}

/// The wire form: field names and defaults follow `RicursiveConfig.__init__`.
#[derive(Deserialize)]
struct Raw {
    model_type: Option<String>,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    rope_theta: f32,
    #[serde(default = "default_window")]
    sliding_window: usize,
    layer_types: Option<Vec<String>>,
    hybrid_layer_pattern: Option<Vec<u8>>,
    #[serde(default = "default_true")]
    use_gated_attention: bool,
    #[serde(default = "default_true")]
    gate_full_precision: bool,
    #[serde(default)]
    use_valres: bool,
    ffn_clamp_threshold: Option<f32>,
    embed_scale: Option<f32>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_hidden_act() -> String {
    "silu".into()
}
fn default_eps() -> f32 {
    1e-6
}
fn default_theta() -> f32 {
    10_000.0
}
fn default_window() -> usize {
    1024
}
fn default_true() -> bool {
    true
}

/// Keys that, if present and set, change the math in a way this port does not
/// implement. `null`, `false` and `0` count as unset.
const UNSUPPORTED_WHEN_SET: [&str; 5] = ["rope_scaling", "rope_parameters", "attention_bias", "mlp_bias", "attention_dropout"];

/// The model's shape and op-sequence flags.
#[derive(Clone, Debug, PartialEq)]
pub struct RicursiveConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Keys visible to a query on a sliding layer, itself included: key `j`
    /// is visible to query `i` iff `i - (sliding_window - 1) <= j <= i`.
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    /// `attn *= sigmoid(g_proj(x))` before `o_proj`.
    pub use_gated_attention: bool,
    /// Whether the gate's pre-activation is kept in f32. This port computes
    /// everything in f32, so the flag only has to be accepted.
    pub gate_full_precision: bool,
    /// Value residual: `v_proj` reads `lerp(x0, xn, sigmoid(value_res_lambda))`.
    pub use_valres: bool,
    /// Clipped SwiGLU: gate capped above, up clipped on both sides, at this
    /// magnitude. `None` disables the clipping (the reference treats 0 alike).
    pub ffn_clamp_threshold: Option<f32>,
    /// Multiplier applied to the embedding lookup; `None` for a checkpoint
    /// that folded it into `embed_tokens.weight`.
    pub embed_scale: Option<f32>,
}

impl RicursiveConfig {
    /// Parse and validate a `config.json` body.
    pub fn from_json(text: &str) -> Result<RicursiveConfig, String> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("ricursive config: not valid JSON: {e}"))?;
        for key in UNSUPPORTED_WHEN_SET {
            if let Some(v) = value.get(key) {
                let unset = v.is_null() || v == &serde_json::Value::Bool(false) || v.as_f64() == Some(0.0);
                if !unset {
                    return Err(format!("ricursive config: {key} = {v} is not supported by this port"));
                }
            }
        }
        let raw: Raw = serde_json::from_value(value).map_err(|e| format!("ricursive config: {e}"))?;
        if let Some(t) = &raw.model_type {
            if t != "ricursive" {
                return Err(format!("ricursive config: model_type is {t:?}, expected \"ricursive\""));
            }
        }
        if raw.hidden_act != "silu" {
            return Err(format!("ricursive config: hidden_act {:?} is not supported (only silu)", raw.hidden_act));
        }
        if raw.tie_word_embeddings {
            return Err("ricursive config: tie_word_embeddings is not supported (the checkpoint carries a separate lm_head)".into());
        }
        let n = raw.num_hidden_layers;
        let layer_types = match raw.layer_types {
            Some(names) => names.iter().map(|s| LayerType::parse(s)).collect::<Result<Vec<_>, _>>()?,
            // `RicursiveConfig`'s default: every fifth layer is global.
            None => (0..n).map(|i| if i % 5 == 4 { LayerType::Full } else { LayerType::Sliding }).collect(),
        };
        if layer_types.len() != n {
            return Err(format!("ricursive config: layer_types has {} entries, num_hidden_layers is {n}", layer_types.len()));
        }
        if let Some(pattern) = &raw.hybrid_layer_pattern {
            let derived: Vec<u8> = layer_types.iter().map(|t| u8::from(*t == LayerType::Sliding)).collect();
            if pattern != &derived {
                return Err("ricursive config: hybrid_layer_pattern disagrees with layer_types".into());
            }
        }
        let cfg = RicursiveConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: n,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            sliding_window: raw.sliding_window,
            layer_types,
            use_gated_attention: raw.use_gated_attention,
            gate_full_precision: raw.gate_full_precision,
            use_valres: raw.use_valres,
            ffn_clamp_threshold: raw.ffn_clamp_threshold.filter(|t| *t != 0.0),
            embed_scale: raw.embed_scale,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read and parse a `config.json` file.
    pub fn from_file(path: &str) -> Result<RicursiveConfig, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("ricursive config: reading {path}: {e}"))?;
        Self::from_json(&text)
    }

    fn validate(&self) -> Result<(), String> {
        let positive = [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
        ];
        for (name, v) in positive {
            if v == 0 {
                return Err(format!("ricursive config: {name} must be positive"));
            }
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(format!("ricursive config: num_attention_heads {} is not a multiple of num_key_value_heads {}", self.num_attention_heads, self.num_key_value_heads));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(format!("ricursive config: head_dim {} must be even (half-split RoPE)", self.head_dim));
        }
        if self.layer_types.contains(&LayerType::Sliding) && self.sliding_window == 0 {
            return Err("ricursive config: sliding_window must be positive when a layer is sliding".into());
        }
        if self.rms_norm_eps <= 0.0 || !self.rms_norm_eps.is_finite() {
            return Err(format!("ricursive config: rms_norm_eps {} must be finite and positive", self.rms_norm_eps));
        }
        Ok(())
    }

    /// Attention type of layer `layer`.
    pub fn layer_type(&self, layer: usize) -> LayerType {
        self.layer_types[layer]
    }

    /// Width of the query projection, `num_attention_heads * head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Width of the key and value projections, `num_key_value_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Query heads sharing one key/value head.
    pub fn group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// This config cut to the given source layers: layer `i` of the result is
    /// layer `source_layers[i]` of `self`. For running a few real layers of a
    /// checkpoint too large to hold, with their real types.
    pub fn truncated(&self, source_layers: &[usize]) -> Result<RicursiveConfig, String> {
        let mut layer_types = Vec::with_capacity(source_layers.len());
        for &l in source_layers {
            layer_types.push(*self.layer_types.get(l).ok_or_else(|| format!("ricursive config: source layer {l} out of range (model has {})", self.num_hidden_layers))?);
        }
        let cut = RicursiveConfig { num_hidden_layers: source_layers.len(), layer_types, ..self.clone() };
        cut.validate()?;
        Ok(cut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_HEAD: &str = r#"{"model_type":"ricursive","vocab_size":100352,"hidden_size":5120,"intermediate_size":40960,
        "num_hidden_layers":5,"num_attention_heads":80,"num_key_value_heads":8,"head_dim":128,"hidden_act":"silu",
        "rms_norm_eps":1e-06,"rope_theta":10000.0,"sliding_window":1024,"use_gated_attention":true,"gate_full_precision":true,
        "use_valres":true,"ffn_clamp_threshold":20.0,"embed_scale":71.55417527999327,"tie_word_embeddings":false,
        "transformers_version":"4.57.6","bos_token_id":100257"#;

    fn real(extra: &str) -> String {
        format!("{REAL_HEAD}{extra}}}")
    }

    #[test]
    fn the_released_shape_parses_with_the_default_hybrid_pattern() {
        let cfg = RicursiveConfig::from_json(&real("")).unwrap();
        assert_eq!((cfg.q_dim(), cfg.kv_dim(), cfg.group()), (10240, 1024, 10));
        let globals: Vec<usize> = (0..5).filter(|&i| cfg.layer_type(i) == LayerType::Full).collect();
        assert_eq!(globals, [4]);
        assert_eq!(cfg.ffn_clamp_threshold, Some(20.0));
    }

    #[test]
    fn a_zero_clamp_threshold_disables_clipping() {
        let text = real("").replace("\"ffn_clamp_threshold\":20.0", "\"ffn_clamp_threshold\":0");
        assert_eq!(RicursiveConfig::from_json(&text).unwrap().ffn_clamp_threshold, None);
    }

    #[test]
    fn math_changing_keys_and_inconsistent_metadata_are_refused() {
        let cases = [
            (r#","rope_scaling":{"type":"yarn"}"#, "rope_scaling"),
            (r#","attention_bias":true"#, "attention_bias"),
            (r#","hybrid_layer_pattern":[1,1,1,1,1]"#, "hybrid_layer_pattern"),
            (r#","layer_types":["sliding_attention"]"#, "layer_types has 1"),
            (r#","layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","chunked"]"#, "unsupported layer type"),
        ];
        for (extra, needle) in cases {
            let err = RicursiveConfig::from_json(&real(extra)).unwrap_err();
            assert!(err.contains(needle), "{extra}: {err}");
        }
        let tied = real("").replace("\"tie_word_embeddings\":false", "\"tie_word_embeddings\":true");
        assert!(RicursiveConfig::from_json(&tied).unwrap_err().contains("tie_word_embeddings"));
        let gelu = real("").replace("\"silu\"", "\"gelu\"");
        assert!(RicursiveConfig::from_json(&gelu).unwrap_err().contains("hidden_act"));
    }

    #[test]
    fn truncation_keeps_the_real_layer_types() {
        let cfg = RicursiveConfig::from_json(&real("")).unwrap();
        let cut = cfg.truncated(&[0, 4]).unwrap();
        assert_eq!(cut.layer_types, [LayerType::Sliding, LayerType::Full]);
        assert!(cfg.truncated(&[5]).is_err());
    }
}
