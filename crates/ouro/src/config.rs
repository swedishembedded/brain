// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint-compatible configurations for
// recurrent-depth decoders for its clients. If your team needs expertise in
// importing a looped model's exact architecture and tensor layout then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Ouro configuration and parameter layout.
//!
//! The released `Ouro-2.6B-Thinking` config fixes the shape: 48 layers,
//! hidden 2048, 16 query and 16 KV heads of width 128, SwiGLU width 5632,
//! RoPE base 1e6, four recurrent steps, an untied LM head and a gated exit
//! projection. The **same** 48 layers run on every step, so the layer count is
//! a shape fact while `total_ut_steps` is the compute axis.

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct OuroConfig {
    pub vocab: u32,
    /// Positional capacity (`max_position_embeddings`), not the training length.
    pub block_size: u32,
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_ff: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Recurrent passes per token (`total_ut_steps`).
    pub total_ut_steps: u32,
    /// Cumulative exit probability a pass must reach to be selected
    /// (`early_exit_threshold`; 1.0 selects the final pass).
    pub early_exit_threshold: f32,
    pub tie_embeddings: bool,
}

const CONFIG_KEYS: &[&str] = &[
    "vocab_size",
    "max_position_embeddings",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "rope_theta",
    "rms_norm_eps",
    "total_ut_steps",
];

impl OuroConfig {
    /// A tiny config that exercises the full contract - weighted passes, the
    /// exit gate, an untied head - while fitting a gradient check.
    pub fn tiny() -> OuroConfig {
        OuroConfig {
            vocab: 29,
            block_size: 32,
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            d_ff: 32,
            rope_theta: 1.0e6,
            norm_eps: 1e-6,
            total_ut_steps: 3,
            early_exit_threshold: 1.0,
            tie_embeddings: false,
        }
    }

    /// `ByteDance/Ouro-2.6B-Thinking` as released.
    pub fn ouro_2_6b_thinking() -> OuroConfig {
        OuroConfig {
            vocab: 49152,
            block_size: 65536,
            d_model: 2048,
            n_layers: 48,
            n_heads: 16,
            n_kv_heads: 16,
            head_dim: 128,
            d_ff: 5632,
            rope_theta: 1.0e6,
            norm_eps: 1e-6,
            total_ut_steps: 4,
            early_exit_threshold: 1.0,
            tie_embeddings: false,
        }
    }

    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
    /// Query heads per KV head. Ouro-2.6B is plain MHA (1).
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }

    /// Parse a Hugging Face `config.json`. Unknown keys are not rejected here -
    /// the released file carries many (`layer_types`, `auto_map`,
    /// `transformers_version`) that brain does not need - but every key this
    /// crate relies on is required, and a missing one is named in the error.
    pub fn from_json(v: &Value) -> Result<OuroConfig, String> {
        let missing: Vec<&str> = CONFIG_KEYS.iter().copied().filter(|k| v.get(*k).is_none()).collect();
        if !missing.is_empty() {
            return Err(format!("ouro config: missing keys {missing:?}"));
        }
        let u = |k: &str| v[k].as_u64().map(|x| x as u32).ok_or_else(|| format!("ouro config: {k}"));
        let head_dim = match v.get("head_dim").and_then(|x| x.as_u64()) {
            Some(h) => h as u32,
            None => u("hidden_size")? / u("num_attention_heads")?,
        };
        let cfg = OuroConfig {
            vocab: u("vocab_size")?,
            block_size: u("max_position_embeddings")?,
            d_model: u("hidden_size")?,
            n_layers: u("num_hidden_layers")?,
            n_heads: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            head_dim,
            d_ff: u("intermediate_size")?,
            rope_theta: v["rope_theta"].as_f64().ok_or("ouro config: rope_theta")? as f32,
            norm_eps: v["rms_norm_eps"].as_f64().ok_or("ouro config: rms_norm_eps")? as f32,
            total_ut_steps: u("total_ut_steps").unwrap_or(4).max(1),
            early_exit_threshold: v["early_exit_threshold"].as_f64().unwrap_or(1.0) as f32,
            tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Structural checks the kernels rely on: a head split that divides
    /// evenly, an MHA/GQA group that divides evenly, non-zero dimensions.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("vocab_size", self.vocab),
            ("max_position_embeddings", self.block_size),
            ("hidden_size", self.d_model),
            ("num_hidden_layers", self.n_layers),
            ("num_attention_heads", self.n_heads),
            ("num_key_value_heads", self.n_kv_heads),
            ("head_dim", self.head_dim),
            ("intermediate_size", self.d_ff),
        ] {
            if v == 0 {
                return Err(format!("ouro config: {name} is zero"));
            }
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!(
                "ouro config: num_attention_heads {} is not a multiple of num_key_value_heads {}",
                self.n_heads, self.n_kv_heads
            ));
        }
        Ok(())
    }

    /// The released config's own spelling, for a checkpoint card.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "architectures": ["OuroForCausalLM"],
            "model_type": "ouro",
            "vocab_size": self.vocab,
            "max_position_embeddings": self.block_size,
            "hidden_size": self.d_model,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.d_ff,
            "hidden_act": "silu",
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.norm_eps,
            "total_ut_steps": self.total_ut_steps,
            "early_exit_threshold": self.early_exit_threshold,
            "tie_word_embeddings": self.tie_embeddings,
        })
    }

    /// `(name, numel)` in forward order. Every layer carries four RMSNorms
    /// (sandwich normalization: before **and** after attention and the MLP),
    /// the MHA projections and the SwiGLU FFN; the exit gate is one linear
    /// with a bias over the model width, and the head is untied for the
    /// released checkpoint.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let hq = self.q_dim() as usize;
        let hkv = self.kv_dim() as usize;

        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];
        for l in 0..self.n_layers as usize {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("attn.wq.weight"), hq * d));
            out.push((p("attn.wk.weight"), hkv * d));
            out.push((p("attn.wv.weight"), hkv * d));
            out.push((p("attn.wo.weight"), d * hq));
            out.push((p("ln1_2.weight"), d));
            out.push((p("ln2.weight"), d));
            out.push((p("mlp.gate.weight"), ff * d));
            out.push((p("mlp.up.weight"), ff * d));
            out.push((p("mlp.down.weight"), d * ff));
            out.push((p("ln2_2.weight"), d));
        }
        out.push(("norm.weight".to_string(), d));
        out.push(("gate.weight".to_string(), d));
        out.push(("gate.bias".to_string(), 1));
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
    fn tiny_round_trips_and_validates() {
        let cfg = OuroConfig::tiny();
        let back = OuroConfig::from_json(&cfg.to_json()).expect("tiny config parses");
        assert_eq!(back, cfg);
        assert_eq!(cfg.group(), 1);
    }

    #[test]
    fn the_released_shape_matches_the_checkpoint_manifest_exactly() {
        // Facts from the checkpoint's safetensors header: 533 tensors and
        // 2,667,974,657 parameters. A layout drift shows up here before a
        // 5 GiB import does.
        let cfg = OuroConfig::ouro_2_6b_thinking();
        let list = cfg.param_list();
        assert_eq!(list.len(), 533);
        let total: usize = list.iter().map(|(_, n)| *n).sum();
        assert_eq!(total, 2_667_974_657);
        assert_eq!(cfg.param_list().iter().map(|(n, _)| n).collect::<std::collections::BTreeSet<_>>().len(), 533);
    }

    #[test]
    fn a_missing_shape_key_is_named() {
        let v = serde_json::json!({"vocab_size": 29});
        let err = OuroConfig::from_json(&v).expect_err("incomplete config must be refused");
        assert!(err.contains("missing keys"), "{err}");
        assert!(err.contains("hidden_size"), "{err}");
    }

    #[test]
    fn an_indivisible_group_is_refused() {
        let mut cfg = OuroConfig::tiny();
        cfg.n_heads = 5;
        assert!(cfg.validate().unwrap_err().contains("multiple"));
    }
}
