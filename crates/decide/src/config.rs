// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The state encoder's configuration and parameter layout.
//!
//! A BERT-family bidirectional encoder, as released by
//! `sentence-transformers/all-MiniLM-L6-v2`: learned absolute position
//! embeddings, a segment (`token_type`) embedding, **post-LayerNorm** residual
//! blocks (`LN(x + sublayer(x))`, not the pre-LN arrangement CLIP's text tower
//! uses), exact/erf GELU, and biases on every projection.
//!
//! Q/K/V are **fused at import** into one `[3H, H]` matrix. The checkpoint
//! ships them separately; the attention kernels read a fused `[rows, 3H]`
//! buffer with `q_off`/`k_off`/`v_off`, so joining them once at the boundary
//! buys one GEMM per layer instead of three and costs nothing at run time.

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub vocab: u32,
    /// Rows of the learned position table - the hard ceiling on one window.
    pub max_positions: u32,
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub d_ff: u32,
    /// Rows of the segment table. Two in every checkpoint of this family, and
    /// both are live here: the decision model separates its STATE and SLOT
    /// roles through this embedding rather than through a second tower.
    pub type_vocab: u32,
    pub eps: f32,
}

impl EncoderConfig {
    /// `sentence-transformers/all-MiniLM-L6-v2`, read off the released
    /// `config.json`: 6 layers, width 384, 12 heads (head_dim 32), FFN 1536,
    /// vocab 30522, 512 positions, `layer_norm_eps` 1e-12.
    pub fn mini_lm_l6() -> EncoderConfig {
        EncoderConfig {
            vocab: 30522,
            max_positions: 512,
            d_model: 384,
            n_layers: 6,
            n_heads: 12,
            d_ff: 1536,
            type_vocab: 2,
            eps: 1e-12,
        }
    }

    /// A small synthetic shape for gradient checks. Deliberately NOT a scaled
    /// copy of the real one: `d_model`/`n_heads` give `head_dim = 4`, and
    /// `d_ff` is not a multiple of `d_model`, so an axis swap that the real
    /// config's tidy ratios would hide shows up as a shape error.
    pub fn tiny() -> EncoderConfig {
        EncoderConfig {
            vocab: 23,
            max_positions: 12,
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            d_ff: 19,
            type_vocab: 2,
            eps: 1e-12,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }

    /// Every parameter and its shape, in one place: `ParamStore` sizing, the
    /// importer's coverage check and the initializer all read this, so a
    /// tensor cannot exist for one of them and not the others.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (h, ff) = (self.d_model as usize, self.d_ff as usize);
        let mut v = vec![
            ("tok.weight".into(), vec![self.vocab as usize, h]),
            ("pos.weight".into(), vec![self.max_positions as usize, h]),
            ("type.weight".into(), vec![self.type_vocab as usize, h]),
            ("emb_ln.weight".into(), vec![h]),
            ("emb_ln.bias".into(), vec![h]),
        ];
        for l in 0..self.n_layers {
            let p = format!("blocks.{l}");
            v.extend([
                (format!("{p}.qkv.weight"), vec![3 * h, h]),
                (format!("{p}.qkv.bias"), vec![3 * h]),
                (format!("{p}.proj.weight"), vec![h, h]),
                (format!("{p}.proj.bias"), vec![h]),
                (format!("{p}.ln1.weight"), vec![h]),
                (format!("{p}.ln1.bias"), vec![h]),
                (format!("{p}.fc1.weight"), vec![ff, h]),
                (format!("{p}.fc1.bias"), vec![ff]),
                (format!("{p}.fc2.weight"), vec![h, ff]),
                (format!("{p}.fc2.bias"), vec![h]),
                (format!("{p}.ln2.weight"), vec![h]),
                (format!("{p}.ln2.bias"), vec![h]),
            ]);
        }
        v
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "vocab": self.vocab,
            "max_positions": self.max_positions,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "type_vocab": self.type_vocab,
            "eps": self.eps,
        })
    }

    pub fn from_json(v: &Value) -> EncoderConfig {
        let u = |k: &str, d: u32| v.get(k).and_then(Value::as_u64).map(|x| x as u32).unwrap_or(d);
        EncoderConfig {
            vocab: u("vocab", 30522),
            max_positions: u("max_positions", 512),
            d_model: u("d_model", 384),
            n_layers: u("n_layers", 6),
            n_heads: u("n_heads", 12),
            d_ff: u("d_ff", 1536),
            type_vocab: u("type_vocab", 2),
            eps: v.get("eps").and_then(Value::as_f64).unwrap_or(1e-12) as f32,
        }
    }
}
