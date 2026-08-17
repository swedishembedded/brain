// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kronos configuration and parameter lists for BOTH nets: the BSQ tokenizer
//! (`KronosTokenizerConfig`) and the autoregressive decoder (`KronosConfig`).
//!
//! Param NAMES are the reference checkpoints' own `state_dict` keys, so import is
//! a 1:1 name match and the T0 layout gate is a mechanical diff (the same
//! discipline that took Chronos-2 to cosine=1.0).
//!
//! Dims verified from `NeoQuasar/Kronos-small/config.json` +
//! `NeoQuasar/Kronos-Tokenizer-base/config.json` (2026-07-20):
//! - tokenizer: d_in=6 (OHLCVA), d_model=256, n_heads=4, ff=512,
//!   n_enc_layers=n_dec_layers=4 → **3 encoder + 3 decoder blocks**
//!   (`range(n_layers-1)`), s1_bits=s2_bits=10, codebook_dim k=20, group_size=4.
//! - decoder (small): d_model=512, n_heads=8, ff=1024, n_layers=8,
//!   s1_bits=s2_bits=10 → s1/s2 vocab=1024, **learn_te=true** (learned calendar
//!   tables → key names `time_emb.<name>_embed.weight`, no `.emb`).
//!
//! Parity invariants (differ from Chronos-2!): attention is **CAUSAL + SCALED**
//! (1/√head_dim); RoPE is **half-split/NeoX** (reuse `rope_neox`); norm is
//! RMSNorm (eps 1e-5, weight-only); FFN is **SwiGLU** (`w2(silu(w1)·w3)`, no
//! bias). BSQ is parameter-free (`quant_embed` is the projection). The
//! `HierarchicalEmbedding` fusion scales `emb_s1/s2` by `√d_model`, but the
//! `emb_s1` reused as the dependency-layer sibling embedding is RAW (no scale).

/// A named parameter with its shape (row-major; PyTorch `nn.Linear` is `[out, in]`).
pub type Param = (String, Vec<usize>);

/// The BSQ tokenizer: OHLCV(+amount) bar ↔ hierarchical (s1, s2) tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct KronosTokenizerConfig {
    pub d_in: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_dim: usize,
    /// Actual TransformerBlock counts = `n_enc_layers - 1` / `n_dec_layers - 1`.
    pub n_enc_layers: usize,
    pub n_dec_layers: usize,
    pub s1_bits: usize,
    pub s2_bits: usize,
    pub group_size: usize,
}

impl Default for KronosTokenizerConfig {
    /// `NeoQuasar/Kronos-Tokenizer-base`.
    fn default() -> Self {
        KronosTokenizerConfig {
            d_in: 6,
            d_model: 256,
            n_heads: 4,
            ff_dim: 512,
            n_enc_layers: 4,
            n_dec_layers: 4,
            s1_bits: 10,
            s2_bits: 10,
            group_size: 4,
        }
    }
}

impl KronosTokenizerConfig {
    pub fn tiny() -> Self {
        KronosTokenizerConfig {
            d_in: 6,
            d_model: 16,
            n_heads: 4,
            ff_dim: 32,
            n_enc_layers: 3,
            n_dec_layers: 3,
            s1_bits: 4,
            s2_bits: 4,
            group_size: 2,
        }
    }
    /// Number of encoder TransformerBlocks (`n_enc_layers - 1`).
    pub fn enc_blocks(&self) -> usize {
        self.n_enc_layers.saturating_sub(1)
    }
    /// Number of decoder TransformerBlocks (`n_dec_layers - 1`).
    pub fn dec_blocks(&self) -> usize {
        self.n_dec_layers.saturating_sub(1)
    }
    /// BSQ codebook dimension `k = s1_bits + s2_bits`.
    pub fn codebook_dim(&self) -> usize {
        self.s1_bits + self.s2_bits
    }

    pub fn from_hf(v: &serde_json::Value) -> Result<KronosTokenizerConfig, String> {
        let u = |k: &str| v[k].as_u64().map(|n| n as usize).ok_or_else(|| format!("missing {k}"));
        Ok(KronosTokenizerConfig {
            d_in: u("d_in")?,
            d_model: u("d_model")?,
            n_heads: u("n_heads")?,
            ff_dim: u("ff_dim")?,
            n_enc_layers: u("n_enc_layers")?,
            n_dec_layers: u("n_dec_layers")?,
            s1_bits: u("s1_bits")?,
            s2_bits: u("s2_bits")?,
            group_size: v["group_size"].as_u64().map(|n| n as usize).unwrap_or(1),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "d_in": self.d_in, "d_model": self.d_model, "n_heads": self.n_heads,
            "ff_dim": self.ff_dim, "n_enc_layers": self.n_enc_layers,
            "n_dec_layers": self.n_dec_layers, "s1_bits": self.s1_bits,
            "s2_bits": self.s2_bits, "group_size": self.group_size,
        })
    }

    /// The tokenizer's learnable parameter list (reference `state_dict` names).
    /// The non-learnable BSQ buffers (`tokenizer.bsq.basis` etc.) are recomputed
    /// in code and excluded here (skipped by the T0 gate).
    pub fn param_list(&self) -> Vec<Param> {
        let d = self.d_model;
        let k = self.codebook_dim();
        let mut p: Vec<Param> = vec![
            ("embed.weight".into(), vec![d, self.d_in]),
            ("embed.bias".into(), vec![d]),
            ("head.weight".into(), vec![self.d_in, d]),
            ("head.bias".into(), vec![self.d_in]),
            ("quant_embed.weight".into(), vec![k, d]),
            ("quant_embed.bias".into(), vec![k]),
            ("post_quant_embed_pre.weight".into(), vec![d, self.s1_bits]),
            ("post_quant_embed_pre.bias".into(), vec![d]),
            ("post_quant_embed.weight".into(), vec![d, k]),
            ("post_quant_embed.bias".into(), vec![d]),
        ];

        for i in 0..self.enc_blocks() {
            transformer_block(&mut p, &format!("encoder.{i}"), d, self.ff_dim);
        }
        for i in 0..self.dec_blocks() {
            transformer_block(&mut p, &format!("decoder.{i}"), d, self.ff_dim);
        }
        p
    }

    pub fn param_count(&self) -> usize {
        self.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

/// The autoregressive decoder over (s1, s2) tokens, with the dual head.
#[derive(Clone, Debug, PartialEq)]
pub struct KronosConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ff_dim: usize,
    pub s1_bits: usize,
    pub s2_bits: usize,
    /// Learned calendar embeddings (`true`) vs frozen sinusoids (`false`).
    pub learn_te: bool,
    /// Dependency-layer head count (fixed 4 in the reference, independent of
    /// `n_heads`).
    pub dep_n_heads: usize,
    /// Max context in **bars** (timesteps), not subtokens.
    pub max_context: usize,
}

impl Default for KronosConfig {
    /// `NeoQuasar/Kronos-small` (24.7M).
    fn default() -> Self {
        KronosConfig {
            d_model: 512,
            n_layers: 8,
            n_heads: 8,
            ff_dim: 1024,
            s1_bits: 10,
            s2_bits: 10,
            learn_te: true,
            dep_n_heads: 4,
            max_context: 512,
        }
    }
}

impl KronosConfig {
    pub fn tiny() -> Self {
        KronosConfig {
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            ff_dim: 32,
            s1_bits: 4,
            s2_bits: 4,
            learn_te: true,
            dep_n_heads: 2,
            max_context: 64,
        }
    }
    pub fn s1_vocab(&self) -> usize {
        1 << self.s1_bits
    }
    pub fn s2_vocab(&self) -> usize {
        1 << self.s2_bits
    }

    pub fn from_hf(v: &serde_json::Value) -> Result<KronosConfig, String> {
        let u = |k: &str| v[k].as_u64().map(|n| n as usize).ok_or_else(|| format!("missing {k}"));
        Ok(KronosConfig {
            d_model: u("d_model")?,
            n_layers: u("n_layers")?,
            n_heads: u("n_heads")?,
            ff_dim: u("ff_dim")?,
            s1_bits: u("s1_bits")?,
            s2_bits: u("s2_bits")?,
            learn_te: v["learn_te"].as_bool().unwrap_or(true),
            dep_n_heads: v["dep_n_heads"].as_u64().map(|n| n as usize).unwrap_or(4),
            max_context: v["max_context"].as_u64().map(|n| n as usize).unwrap_or(512),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "d_model": self.d_model, "n_layers": self.n_layers, "n_heads": self.n_heads,
            "ff_dim": self.ff_dim, "s1_bits": self.s1_bits, "s2_bits": self.s2_bits,
            "learn_te": self.learn_te, "dep_n_heads": self.dep_n_heads,
            "max_context": self.max_context,
        })
    }

    /// The decoder's learnable parameter list (reference `state_dict` names).
    pub fn param_list(&self) -> Vec<Param> {
        let d = self.d_model;
        // hierarchical embedding
        let mut p: Vec<Param> = vec![
            ("embedding.emb_s1.weight".into(), vec![self.s1_vocab(), d]),
            ("embedding.emb_s2.weight".into(), vec![self.s2_vocab(), d]),
            ("embedding.fusion_proj.weight".into(), vec![d, 2 * d]),
            ("embedding.fusion_proj.bias".into(), vec![d]),
        ];

        // temporal (calendar) embeddings — 5 tables. `learn_te=true` → learned,
        // key `time_emb.<name>_embed.weight`; else frozen sinusoid, key
        // `time_emb.<name>_embed.emb.weight`.
        let suffix = if self.learn_te { "weight" } else { "emb.weight" };
        for (name, size) in
            [("minute", 60), ("hour", 24), ("weekday", 7), ("day", 32), ("month", 13)]
        {
            p.push((format!("time_emb.{name}_embed.{suffix}"), vec![size, d]));
        }

        // transformer blocks
        for i in 0..self.n_layers {
            transformer_block(&mut p, &format!("transformer.{i}"), d, self.ff_dim);
        }
        p.push(("norm.weight".into(), vec![d]));

        // dependency-aware cross-attention layer (n_heads=dep_n_heads, but the
        // projections are still d×d).
        cross_attention(&mut p, "dep_layer.cross_attn", d);
        p.push(("dep_layer.norm.weight".into(), vec![d]));

        // dual head
        p.push(("head.proj_s1.weight".into(), vec![self.s1_vocab(), d]));
        p.push(("head.proj_s1.bias".into(), vec![self.s1_vocab()]));
        p.push(("head.proj_s2.weight".into(), vec![self.s2_vocab(), d]));
        p.push(("head.proj_s2.bias".into(), vec![self.s2_vocab()]));

        p
    }

    pub fn param_count(&self) -> usize {
        self.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

/// One `TransformerBlock`: pre-norm RMSNorm → biased MHA (q/k/v/out) → residual
/// → pre-norm RMSNorm → SwiGLU FFN (w1/w3/w2, no bias) → residual. RoPE buffers
/// (`self_attn.rotary.inv_freq`) are recomputed, not stored here.
fn transformer_block(p: &mut Vec<Param>, prefix: &str, d: usize, ff: usize) {
    p.push((format!("{prefix}.norm1.weight"), vec![d]));
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        p.push((format!("{prefix}.self_attn.{proj}.weight"), vec![d, d]));
        p.push((format!("{prefix}.self_attn.{proj}.bias"), vec![d]));
    }
    p.push((format!("{prefix}.norm2.weight"), vec![d]));
    p.push((format!("{prefix}.ffn.w1.weight"), vec![ff, d]));
    p.push((format!("{prefix}.ffn.w3.weight"), vec![ff, d]));
    p.push((format!("{prefix}.ffn.w2.weight"), vec![d, ff]));
}

/// A biased cross-attention module (q/k/v/out projections d×d).
fn cross_attention(p: &mut Vec<Param>, prefix: &str, d: usize) {
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        p.push((format!("{prefix}.{proj}.weight"), vec![d, d]));
        p.push((format!("{prefix}.{proj}.bias"), vec![d]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn tokenizer_dims_and_block_counts() {
        let c = KronosTokenizerConfig::default();
        assert_eq!(c.enc_blocks(), 3);
        assert_eq!(c.dec_blocks(), 3);
        assert_eq!(c.codebook_dim(), 20);
        assert_eq!(c.group_size, 4);
        assert_eq!(20 % c.group_size, 0, "group_size must divide codebook_dim");
    }

    #[test]
    fn tokenizer_param_list_well_formed() {
        let c = KronosTokenizerConfig::default();
        let pl = c.param_list();
        let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), pl.len(), "dupe keys");
        assert!(keys.contains("embed.weight"));
        assert!(keys.contains("quant_embed.weight"));
        assert!(keys.contains("encoder.0.self_attn.q_proj.weight"));
        assert!(keys.contains("encoder.2.ffn.w2.weight"));
        assert!(keys.contains("decoder.0.norm1.weight"));
        assert!(keys.contains("post_quant_embed.weight"));
        // ~4M params (tokenizer file ~16MB fp32)
        let n = c.param_count();
        assert!((3_500_000..4_500_000).contains(&n), "tokenizer params {n}");
    }

    #[test]
    fn decoder_dims_and_vocab() {
        let c = KronosConfig::default();
        assert_eq!(c.s1_vocab(), 1024);
        assert_eq!(c.s2_vocab(), 1024);
        assert_eq!(c.max_context, 512, "bars, not subtokens");
        assert!(c.learn_te);
    }

    #[test]
    fn decoder_param_list_well_formed_and_learn_te_keys() {
        let c = KronosConfig::default();
        let pl = c.param_list();
        let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), pl.len(), "dupe keys");
        assert!(keys.contains("embedding.emb_s1.weight"));
        assert!(keys.contains("embedding.fusion_proj.weight"));
        assert!(keys.contains("transformer.7.ffn.w1.weight"));
        assert!(keys.contains("dep_layer.cross_attn.q_proj.weight"));
        assert!(keys.contains("head.proj_s2.bias"));
        assert!(keys.contains("norm.weight"));
        // learn_te=true -> learned calendar table key (no `.emb`)
        assert!(keys.contains("time_emb.minute_embed.weight"));
        assert!(!keys.contains("time_emb.minute_embed.emb.weight"));
        // ~24.7M for Kronos-small
        let n = c.param_count();
        assert!((23_000_000..26_000_000).contains(&n), "decoder params {n}");
    }

    #[test]
    fn learn_te_false_switches_key_names() {
        let c = KronosConfig { learn_te: false, ..KronosConfig::default() };
        let keys: HashSet<String> = c.param_list().into_iter().map(|(k, _)| k).collect();
        assert!(keys.contains("time_emb.minute_embed.emb.weight"));
        assert!(!keys.contains("time_emb.minute_embed.weight"));
    }

    #[test]
    fn from_hf_roundtrips() {
        let tc = KronosTokenizerConfig::default();
        assert_eq!(KronosTokenizerConfig::from_hf(&tc.to_json()).unwrap(), tc);
        let dc = KronosConfig::default();
        assert_eq!(KronosConfig::from_hf(&dc.to_json()).unwrap(), dc);
    }
}
