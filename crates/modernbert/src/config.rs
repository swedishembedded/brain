// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ModernBERT's shape and per-layer attention pattern.
//!
//! `layer_types` is read from the released `config.json` when present
//! (`answerdotai/ModernBERT-large`'s own 28-entry list), rather than always
//! re-derived from `global_attn_every_n_layers` - a released checkpoint's
//! alternation is a fact about that checkpoint, not a formula this crate
//! should assume holds for every future one, even though the formula
//! (`layer % global_attn_every_n_layers == 0` is full attention, everything
//! else is a local window) reproduces the real list exactly today.

use serde_json::Value;

/// One layer's attention shape: the whole span, or a local window that looks
/// `window` positions in EACH direction (bidirectional, unlike a causal
/// model's window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerAttn {
    Full,
    Local,
}

#[derive(Clone, Debug)]
pub struct ModernBertConfig {
    pub vocab: u32,
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    /// The GeGLU intermediate width. `mlp.wi.weight` is `[2*d_ff, d_model]`
    /// (gate and up fused in one matrix, split at runtime); `mlp.wo.weight`
    /// is `[d_model, d_ff]`.
    pub d_ff: u32,
    /// Rows of the RoPE table - the hard ceiling on one window.
    pub max_positions: u32,
    pub eps: f32,
    /// Per-layer attention pattern, `n_layers` long.
    pub layer_types: Vec<LayerAttn>,
    /// RoPE base for [`LayerAttn::Full`] layers - 160000 on the released
    /// checkpoint, far higher than [`Self::rope_theta_local`] so the rarer
    /// full-attention layers keep long-range positions distinguishable.
    pub rope_theta_full: f32,
    /// RoPE base for [`LayerAttn::Local`] layers - 10000, the conventional
    /// value, since a local layer never sees a position more than `window`
    /// away regardless of base.
    pub rope_theta_local: f32,
    /// Bidirectional local-attention radius: key `j` lives for query `i` on
    /// a [`LayerAttn::Local`] layer iff `|i-j| <= window`. The released
    /// config's own `local_attention: 128` is the FULL width, not this
    /// per-side radius - `window = local_attention / 2` - see
    /// `Self::from_json`'s own conversion.
    pub window: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    pub pad_token_id: u32,
    pub mask_token_id: u32,
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }

    /// Whether layer `l` needs to run the seeded backward's `attn_norm` -
    /// every layer except the first, whose input is the embedding's own
    /// LayerNorm output and therefore skips a redundant second one
    /// (`nn.Identity()` on the released model).
    pub fn has_attn_norm(&self, l: usize) -> bool {
        l != 0
    }

    /// `answerdotai/ModernBERT-large`, read off the released `config.json`:
    /// 28 layers, hidden 1024, 16 heads (head_dim 64), GeGLU intermediate
    /// 2624, vocab 50368, 8192 positions, `layer_norm_eps` 1e-5,
    /// `global_attn_every_n_layers` 3, `local_attention` 128 (so `window`
    /// 64), theta 160000/10000. Special token ids from the released
    /// `tokenizer/tokenizer.json`'s `added_tokens` (NOT assumed elsewhere -
    /// `Laya`'s multilingual/typed-decisions variants use a different
    /// tokenizer and different ids).
    pub fn modernbert_large() -> ModernBertConfig {
        ModernBertConfig {
            vocab: 50368,
            d_model: 1024,
            n_layers: 28,
            n_heads: 16,
            d_ff: 2624,
            max_positions: 8192,
            eps: 1e-5,
            layer_types: (0..28).map(|l| if l % 3 == 0 { LayerAttn::Full } else { LayerAttn::Local }).collect(),
            rope_theta_full: 160_000.0,
            rope_theta_local: 10_000.0,
            window: 64,
            cls_token_id: 50281,
            sep_token_id: 50282,
            pad_token_id: 50283,
            mask_token_id: 50284,
        }
    }

    /// A small synthetic shape for gradient checks and parity fixtures.
    /// Deliberately not a scaled copy of the real one - `d_ff` is not a
    /// multiple of `d_model`, matching `decide::EncoderConfig::tiny()`'s own
    /// reasoning. `n_layers: 4` with `global_every: 2` gives layer types
    /// `[Full, Local, Full, Local]`, so a probe run against this config
    /// exercises the no-attn_norm layer-0 case, BOTH thetas, and (with
    /// `window: 3` against a caller-chosen `seq_len` past `2*window`) the
    /// local window actually cutting off part of the span - the three
    /// properties a scaled-down-only config could accidentally leave
    /// untested. `d_model: 64` is required, not just convenient: the
    /// attention bind group slices `qkv` at each chunk's row offset and the
    /// device requires that offset 256-byte aligned, which only holds for
    /// every row when `d_model` is a multiple of 64 (see
    /// `crates/model/tests/chunked_bidir_fwd_win.rs`'s own note on the same
    /// constraint).
    pub fn tiny() -> ModernBertConfig {
        ModernBertConfig {
            vocab: 37,
            d_model: 64,
            n_layers: 4,
            n_heads: 4,
            d_ff: 19,
            max_positions: 64,
            eps: 1e-5,
            layer_types: (0..4).map(|l| if l % 2 == 0 { LayerAttn::Full } else { LayerAttn::Local }).collect(),
            rope_theta_full: 160_000.0,
            rope_theta_local: 10_000.0,
            window: 3,
            cls_token_id: 0,
            sep_token_id: 1,
            pad_token_id: 2,
            mask_token_id: 3,
        }
    }

    /// Every parameter and its shape, in one place: `ParamStore` sizing, the
    /// importer's coverage check and the initializer all read this.
    ///
    /// Layer 0 has no `attn_norm` entry - see [`Self::has_attn_norm`] - so
    /// this manifest is one tensor shorter than `n_layers` LayerNorms would
    /// suggest, and a caller iterating layers must check
    /// [`Self::has_attn_norm`] rather than assume every layer has one.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (h, ff) = (self.d_model as usize, self.d_ff as usize);
        let mut v = vec![("tok.weight".into(), vec![self.vocab as usize, h]), ("emb_norm.weight".into(), vec![h])];
        for l in 0..self.n_layers as usize {
            let p = format!("blocks.{l}");
            if self.has_attn_norm(l) {
                v.push((format!("{p}.attn_norm.weight"), vec![h]));
            }
            v.extend([
                (format!("{p}.qkv.weight"), vec![3 * h, h]),
                (format!("{p}.proj.weight"), vec![h, h]),
                (format!("{p}.mlp_norm.weight"), vec![h]),
                (format!("{p}.mlp.wi.weight"), vec![2 * ff, h]),
                (format!("{p}.mlp.wo.weight"), vec![h, ff]),
            ]);
        }
        v.push(("final_norm.weight".into(), vec![h]));
        v
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "vocab": self.vocab, "d_model": self.d_model, "n_layers": self.n_layers,
            "n_heads": self.n_heads, "d_ff": self.d_ff, "max_positions": self.max_positions,
            "eps": self.eps,
            "layer_types": self.layer_types.iter().map(|t| matches!(t, LayerAttn::Full)).collect::<Vec<_>>(),
            "rope_theta_full": self.rope_theta_full, "rope_theta_local": self.rope_theta_local,
            "window": self.window,
            "cls_token_id": self.cls_token_id, "sep_token_id": self.sep_token_id,
            "pad_token_id": self.pad_token_id, "mask_token_id": self.mask_token_id,
        })
    }

    pub fn from_json(v: &Value) -> ModernBertConfig {
        let u = |k: &str, d: u32| v.get(k).and_then(Value::as_u64).map(|x| x as u32).unwrap_or(d);
        let f = |k: &str, d: f32| v.get(k).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(d);
        let n_layers = u("n_layers", 28);
        let layer_types = v
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|x| x.as_bool().unwrap_or(false)).map(|full| if full { LayerAttn::Full } else { LayerAttn::Local }).collect())
            .unwrap_or_else(|| (0..n_layers).map(|l| if l % 3 == 0 { LayerAttn::Full } else { LayerAttn::Local }).collect());
        ModernBertConfig {
            vocab: u("vocab", 50368),
            d_model: u("d_model", 1024),
            n_layers,
            n_heads: u("n_heads", 16),
            d_ff: u("d_ff", 2624),
            max_positions: u("max_positions", 8192),
            eps: f("eps", 1e-5),
            layer_types,
            rope_theta_full: f("rope_theta_full", 160_000.0),
            rope_theta_local: f("rope_theta_local", 10_000.0),
            window: u("window", 64),
            cls_token_id: u("cls_token_id", 50281),
            sep_token_id: u("sep_token_id", 50282),
            pad_token_id: u("pad_token_id", 50283),
            mask_token_id: u("mask_token_id", 50284),
        }
    }

    /// The released `encoder/config.json` (HF `model_type: "modernbert"`)
    /// straight from the checkpoint, not this crate's own `to_json` shape -
    /// what the importer actually reads. `local_attention` is the FULL
    /// window width; `window` here is the per-side radius, so the halving
    /// happens exactly once, here, rather than at every call site that
    /// reads it.
    pub fn from_hf_json(v: &Value) -> Result<ModernBertConfig, String> {
        let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or_default();
        if model_type != "modernbert" {
            return Err(format!("expected model_type \"modernbert\", got {model_type:?}"));
        }
        let u = |k: &str| -> Result<u32, String> {
            v.get(k).and_then(Value::as_u64).map(|x| x as u32).ok_or_else(|| format!("missing or non-integer {k:?}"))
        };
        let n_layers = u("num_hidden_layers")?;
        let global_every = u("global_attn_every_n_layers")?;
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|x| x.as_str() == Some("full_attention"))
                .map(|full| if full { LayerAttn::Full } else { LayerAttn::Local })
                .collect(),
            None => (0..n_layers).map(|l| if l % global_every == 0 { LayerAttn::Full } else { LayerAttn::Local }).collect(),
        };
        let rope = v.get("rope_parameters");
        let theta = |key: &str, default: f32| -> f32 {
            rope.and_then(|r| r.get(key)).and_then(|r| r.get("rope_theta")).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(default)
        };
        let local_attention = u("local_attention").unwrap_or(128);
        Ok(ModernBertConfig {
            vocab: u("vocab_size")?,
            d_model: u("hidden_size")?,
            n_layers,
            n_heads: u("num_attention_heads")?,
            d_ff: u("intermediate_size")?,
            max_positions: u("max_position_embeddings")?,
            eps: v.get("layer_norm_eps").and_then(Value::as_f64).map(|x| x as f32).unwrap_or(1e-5),
            layer_types,
            rope_theta_full: theta("full_attention", 160_000.0),
            rope_theta_local: theta("sliding_attention", 10_000.0),
            window: local_attention / 2,
            cls_token_id: u("cls_token_id").unwrap_or(50281),
            sep_token_id: u("sep_token_id").unwrap_or(50282),
            pad_token_id: u("pad_token_id").unwrap_or(50283),
            // Not in the released `config.json` at all - read from the
            // tokenizer's own `added_tokens` at import time and overwritten
            // there; this default is ONLY what a config-only construction
            // (tests) sees.
            mask_token_id: u("mask_token_id").unwrap_or(50284),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modernbert_large_matches_the_released_checkpoints_own_parameter_count() {
        let cfg = ModernBertConfig::modernbert_large();
        let total: usize = cfg.tensor_manifest().iter().map(|(_, shape)| shape.iter().product::<usize>()).sum();
        // Computed independently from the released tensor shapes (verified
        // against the real `model.safetensors` header, not assumed): 27
        // layers carry `attn_norm` (layer 0 does not), every layer carries
        // qkv/proj/mlp_norm/mlp.wi/mlp.wo, plus the token embedding, the
        // embedding norm and the final norm.
        assert_eq!(total, 394_781_696, "encoder-only parameter count must match answerdotai/ModernBERT-large's own ~395M");
    }

    #[test]
    fn layer_zero_has_no_attn_norm_but_every_other_layer_does() {
        let cfg = ModernBertConfig::modernbert_large();
        let names: Vec<String> = cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect();
        assert!(!names.iter().any(|n| n == "blocks.0.attn_norm.weight"));
        for l in 1..cfg.n_layers as usize {
            let want = format!("blocks.{l}.attn_norm.weight");
            assert!(names.iter().any(|n| n == &want), "layer {l} must have attn_norm");
        }
    }

    #[test]
    fn layer_types_alternate_full_every_third_layer() {
        let cfg = ModernBertConfig::modernbert_large();
        for (l, t) in cfg.layer_types.iter().enumerate() {
            let want = if l % 3 == 0 { LayerAttn::Full } else { LayerAttn::Local };
            assert_eq!(*t, want, "layer {l}");
        }
    }

    #[test]
    fn from_hf_json_halves_local_attention_into_a_per_side_window() {
        let v = serde_json::json!({
            "model_type": "modernbert", "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "intermediate_size": 19, "max_position_embeddings": 64,
            "global_attn_every_n_layers": 2, "local_attention": 6,
        });
        let cfg = ModernBertConfig::from_hf_json(&v).unwrap();
        assert_eq!(cfg.window, 3, "local_attention is the FULL width; window is the per-side radius");
    }

    #[test]
    fn from_hf_json_refuses_a_non_modernbert_config() {
        let v = serde_json::json!({"model_type": "bert"});
        assert!(ModernBertConfig::from_hf_json(&v).is_err());
    }

    #[test]
    fn to_json_from_json_roundtrips() {
        let cfg = ModernBertConfig::tiny();
        let back = ModernBertConfig::from_json(&cfg.to_json());
        assert_eq!(back.tensor_manifest(), cfg.tensor_manifest());
        assert_eq!(back.layer_types, cfg.layer_types);
        assert_eq!(back.window, cfg.window);
    }
}
