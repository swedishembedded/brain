// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chronos-2 configuration and parameter list.
//!
//! The param NAMES here are the reference checkpoint's own `state_dict` keys
//! (e.g. `encoder.block.0.layer.0.self_attention.q.weight`), not a
//! brain-flavoured renaming — deliberately, so weight import is a 1:1 name match
//! against `amazon/chronos-2` and the T0 layout gate is a mechanical diff rather
//! than a hand-maintained translation table.
//!
//! Dims are verified from `amazon/chronos-2/config.json` (fetched 2026-07-20):
//! `d_model=768, d_kv=64, d_ff=3072, num_layers=12, num_heads=12`, native
//! `float32`. `inner_dim = num_heads * d_kv = 768 = d_model`, so the attention
//! projections are square `768×768`.
//!
//! Architecture invariants that matter for parity (see `docs` / the spec):
//! - T5-style **RMSNorm** (weight-only, no bias, no mean subtraction).
//! - **ReLU** FFN, not gated.
//! - **Unscaled** attention — there is NO `1/sqrt(d_kv)` factor.
//! - **RoPE** (half-split / NeoX style) on the *time* attention only; the group
//!   (cross-variate) attention uses no positional encoding.
//! - Patch embedding is a `ResidualBlock` over a 48-dim per-patch vector
//!   `[time_enc(16), values(16), mask(16)]`.
//! - Quantile head is a `ResidualBlock` `d_model -> num_quantiles*patch = 336`.

/// A named parameter with its shape (row-major, PyTorch `nn.Linear` `[out, in]`).
pub type Param = (String, Vec<usize>);

/// The 21 quantile levels Chronos-2 emits (non-persistent buffer in the
/// reference; hard-coded here).
pub const QUANTILES: [f32; 21] = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.99,
];

/// Chronos-2 core + task configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Chronos2Config {
    // transformer core
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub layer_norm_epsilon: f32,
    pub rope_theta: f32,
    pub vocab_size: usize, // 2: [PAD], [REG]
    pub reg_token_id: usize,
    // task
    pub context_length: usize,
    pub input_patch_size: usize,
    pub output_patch_size: usize,
    pub max_output_patches: usize,
    pub num_quantiles: usize,
    pub use_arcsinh: bool,
    pub use_reg_token: bool,
    pub time_encoding_scale: f32,
}

impl Default for Chronos2Config {
    /// The published `amazon/chronos-2` 120M checkpoint.
    fn default() -> Self {
        Chronos2Config {
            d_model: 768,
            d_kv: 64,
            d_ff: 3072,
            num_layers: 12,
            num_heads: 12,
            layer_norm_epsilon: 1e-6,
            rope_theta: 10000.0,
            vocab_size: 2,
            reg_token_id: 1,
            context_length: 8192,
            input_patch_size: 16,
            output_patch_size: 16,
            max_output_patches: 64,
            num_quantiles: 21,
            use_arcsinh: true,
            use_reg_token: true,
            time_encoding_scale: 8192.0,
        }
    }
}

impl Chronos2Config {
    /// A tiny config for gradient-checking and fast unit tests (same op
    /// structure, small dims). `inner_dim == d_model` is preserved, and
    /// `num_quantiles` stays 21 (the head geometry the forecaster's native-level
    /// interpolation assumes).
    pub fn tiny() -> Self {
        Chronos2Config {
            d_model: 16,
            d_kv: 4,
            d_ff: 32,
            num_layers: 2,
            num_heads: 4,
            input_patch_size: 4,
            output_patch_size: 4,
            max_output_patches: 8,
            context_length: 64,
            time_encoding_scale: 64.0,
            ..Chronos2Config::default()
        }
    }

    /// Inner attention dim = `num_heads * d_kv`.
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }

    /// The per-patch input feature width: `[time_enc, values, mask]` each
    /// `input_patch_size` wide.
    pub fn patch_feat_dim(&self) -> usize {
        self.input_patch_size * 3
    }

    /// The quantile head output width: `num_quantiles * output_patch_size`.
    pub fn head_out_dim(&self) -> usize {
        self.num_quantiles * self.output_patch_size
    }

    /// Parse from a Hugging Face `config.json` (`amazon/chronos-2`).
    pub fn from_hf(v: &serde_json::Value) -> Result<Chronos2Config, String> {
        let cc = &v["chronos_config"];
        let u = |x: &serde_json::Value, k: &str| -> Result<usize, String> {
            x[k].as_u64().map(|n| n as usize).ok_or_else(|| format!("missing/invalid {k}"))
        };
        let f = |x: &serde_json::Value, k: &str, d: f32| x[k].as_f64().map(|n| n as f32).unwrap_or(d);
        Ok(Chronos2Config {
            d_model: u(v, "d_model")?,
            d_kv: u(v, "d_kv")?,
            d_ff: u(v, "d_ff")?,
            num_layers: u(v, "num_layers")?,
            num_heads: u(v, "num_heads")?,
            layer_norm_epsilon: f(v, "layer_norm_epsilon", 1e-6),
            rope_theta: f(v, "rope_theta", 10000.0),
            vocab_size: u(v, "vocab_size").unwrap_or(2),
            reg_token_id: u(v, "reg_token_id").unwrap_or(1),
            context_length: u(cc, "context_length")?,
            input_patch_size: u(cc, "input_patch_size")?,
            output_patch_size: u(cc, "output_patch_size")?,
            max_output_patches: u(cc, "max_output_patches")?,
            // prefer an explicit count (brain's own container); fall back to the
            // quantiles array length (the HF config.json form).
            num_quantiles: cc["num_quantiles"]
                .as_u64()
                .map(|n| n as usize)
                .or_else(|| cc["quantiles"].as_array().map(|a| a.len()))
                .unwrap_or(21),
            use_arcsinh: cc["use_arcsinh"].as_bool().unwrap_or(true),
            use_reg_token: cc["use_reg_token"].as_bool().unwrap_or(true),
            time_encoding_scale: f(cc, "time_encoding_scale", 8192.0),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "d_model": self.d_model, "d_kv": self.d_kv, "d_ff": self.d_ff,
            "num_layers": self.num_layers, "num_heads": self.num_heads,
            "layer_norm_epsilon": self.layer_norm_epsilon, "rope_theta": self.rope_theta,
            "vocab_size": self.vocab_size, "reg_token_id": self.reg_token_id,
            "chronos_config": {
                "context_length": self.context_length,
                "input_patch_size": self.input_patch_size,
                "output_patch_size": self.output_patch_size,
                "max_output_patches": self.max_output_patches,
                "num_quantiles": self.num_quantiles,
                "quantiles": QUANTILES.to_vec(),
                "use_arcsinh": self.use_arcsinh,
                "use_reg_token": self.use_reg_token,
                "time_encoding_scale": self.time_encoding_scale,
            }
        })
    }

    /// The full parameter list, in the reference checkpoint's own key names and
    /// shapes. Device-free — diffable against the real safetensors header before
    /// a single kernel runs (the T0 layout gate).
    pub fn param_list(&self) -> Vec<Param> {
        let d = self.d_model;
        let f = self.d_ff;
        let i = self.inner_dim();
        let feat = self.patch_feat_dim();
        let head = self.head_out_dim();
        let mut p: Vec<Param> = Vec::new();

        // special-token embedding ([PAD], [REG])
        p.push(("shared.weight".into(), vec![self.vocab_size, d]));

        // input patch embedding: ResidualBlock over the 48-dim per-patch vector
        residual_block(&mut p, "input_patch_embedding", feat, f, d);

        // encoder blocks
        for b in 0..self.num_layers {
            let pre = format!("encoder.block.{b}");
            // layer.0 = time self-attention (RoPE)
            attention(&mut p, &format!("{pre}.layer.0"), d, i);
            // layer.1 = group self-attention (no RoPE)
            attention(&mut p, &format!("{pre}.layer.1"), d, i);
            // layer.2 = feed-forward (ReLU MLP)
            p.push((format!("{pre}.layer.2.mlp.wi.weight"), vec![f, d]));
            p.push((format!("{pre}.layer.2.mlp.wo.weight"), vec![d, f]));
            p.push((format!("{pre}.layer.2.layer_norm.weight"), vec![d]));
        }

        // encoder final norm
        p.push(("encoder.final_layer_norm.weight".into(), vec![d]));

        // output quantile head: ResidualBlock d_model -> num_quantiles*patch
        residual_block(&mut p, "output_patch_embedding", d, f, head);

        p
    }

    /// Total learnable parameter count.
    pub fn param_count(&self) -> usize {
        self.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

/// A `ResidualBlock`: `out = output_layer(relu(hidden_layer(x))) + residual_layer(x)`.
/// All three linears have bias. `hidden_layer: in->h`, `output_layer: h->out`,
/// `residual_layer: in->out`.
fn residual_block(p: &mut Vec<Param>, prefix: &str, in_dim: usize, h: usize, out: usize) {
    p.push((format!("{prefix}.hidden_layer.weight"), vec![h, in_dim]));
    p.push((format!("{prefix}.hidden_layer.bias"), vec![h]));
    p.push((format!("{prefix}.output_layer.weight"), vec![out, h]));
    p.push((format!("{prefix}.output_layer.bias"), vec![out]));
    p.push((format!("{prefix}.residual_layer.weight"), vec![out, in_dim]));
    p.push((format!("{prefix}.residual_layer.bias"), vec![out]));
}

/// One MHA sublayer: q/k/v/o projections (bias-free) + its pre-norm RMSNorm
/// weight. `q,k,v: d->inner`, `o: inner->d`.
fn attention(p: &mut Vec<Param>, prefix: &str, d: usize, inner: usize) {
    for proj in ["q", "k", "v"] {
        p.push((format!("{prefix}.self_attention.{proj}.weight"), vec![inner, d]));
    }
    p.push((format!("{prefix}.self_attention.o.weight"), vec![d, inner]));
    p.push((format!("{prefix}.layer_norm.weight"), vec![d]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn default_dims_match_the_published_checkpoint() {
        let c = Chronos2Config::default();
        assert_eq!((c.d_model, c.d_kv, c.d_ff, c.num_layers, c.num_heads), (768, 64, 3072, 12, 12));
        assert_eq!(c.inner_dim(), c.d_model, "H*Dk must equal d_model for this checkpoint");
        assert_eq!(c.patch_feat_dim(), 48);
        assert_eq!(c.head_out_dim(), 21 * 16);
    }

    #[test]
    fn param_list_has_the_right_keys_and_no_dupes() {
        let c = Chronos2Config::default();
        let pl = c.param_list();
        let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), pl.len(), "duplicate param keys");
        // spot-check the reference naming pattern
        assert!(keys.contains("shared.weight"));
        assert!(keys.contains("input_patch_embedding.hidden_layer.weight"));
        assert!(keys.contains("encoder.block.0.layer.0.self_attention.q.weight"));
        assert!(keys.contains("encoder.block.11.layer.1.self_attention.o.weight"));
        assert!(keys.contains("encoder.block.5.layer.2.mlp.wi.weight"));
        assert!(keys.contains("encoder.final_layer_norm.weight"));
        assert!(keys.contains("output_patch_embedding.output_layer.bias"));
        // 12 blocks × (2 attn × 5 + 3 mlp) = 12×13 = 156, + 1 shared + 6 in-embed
        // + 1 final_ln + 6 out-embed = 170
        assert_eq!(pl.len(), 1 + 6 + 12 * 13 + 1 + 6);
    }

    #[test]
    fn attention_projections_are_square_for_this_checkpoint() {
        let c = Chronos2Config::default();
        for (k, shape) in c.param_list() {
            if k.contains("self_attention") && k.ends_with(".weight") {
                assert_eq!(shape, vec![768, 768], "{k} should be 768x768");
            }
        }
    }

    #[test]
    fn total_param_count_is_about_120m() {
        let c = Chronos2Config::default();
        let n = c.param_count();
        // exact reconstruction: ~119.5M
        assert_eq!(n, 119_477_664, "param count drifted: {n}");
        assert!((119_000_000..121_000_000).contains(&n));
    }

    #[test]
    fn from_hf_roundtrips_the_real_config_shape() {
        // the real config.json structure (subset)
        let j = serde_json::json!({
            "d_model": 768, "d_kv": 64, "d_ff": 3072, "num_layers": 12, "num_heads": 12,
            "layer_norm_epsilon": 1e-6, "rope_theta": 10000.0, "vocab_size": 2, "reg_token_id": 1,
            "chronos_config": {
                "context_length": 8192, "input_patch_size": 16, "output_patch_size": 16,
                "max_output_patches": 64, "quantiles": QUANTILES.to_vec(),
                "use_arcsinh": true, "use_reg_token": true, "time_encoding_scale": 8192
            }
        });
        let c = Chronos2Config::from_hf(&j).unwrap();
        assert_eq!(c, Chronos2Config::default());
    }

    #[test]
    fn tiny_config_preserves_inner_dim_invariant() {
        let c = Chronos2Config::tiny();
        assert_eq!(c.inner_dim(), c.d_model);
        // param_list is well-formed at tiny scale too
        let pl = c.param_list();
        assert_eq!(pl.len(), 1 + 6 + c.num_layers * 13 + 1 + 6);
    }
}
