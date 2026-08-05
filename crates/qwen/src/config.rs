// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 decoder configuration + parameter layout.
//!
//! Qwen3 (0.6B reference): RMSNorm (no bias), RoPE (base 1e6, half-split), GQA
//! (`n_kv_heads < n_heads`), per-head QK-RMSNorm over `head_dim` applied to q/k
//! before RoPE, SwiGLU MLP, tied embeddings, **no** attention/MLP biases, and a
//! **decoupled `head_dim`** (e.g. hidden 1024 but 16 heads × 128 = 2048 ≠ 1024).

use serde_json::Value;

/// LoRA adapter configuration (low-rank fine-tuning). When present, the targeted
/// projections keep a frozen base weight plus trainable `A`/`B` adapters.
#[derive(Clone, Debug)]
pub struct LoraCfg {
    pub rank: u32,
    pub alpha: f32,
    /// Which projections get adapters (matched by the leaf name, e.g. "wq").
    pub targets: Vec<String>,
}

impl LoraCfg {
    /// The default Qwen LoRA target set: the four attention projections.
    pub fn attn(rank: u32, alpha: f32) -> LoraCfg {
        LoraCfg {
            rank,
            alpha,
            targets: ["wq", "wk", "wv", "wo"].iter().map(|s| s.to_string()).collect(),
        }
    }
    pub fn targets_leaf(&self, leaf: &str) -> bool {
        self.targets.iter().any(|t| t == leaf)
    }
}

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_ff: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub tie_embeddings: bool,
    /// Per-head QK-RMSNorm on q/k before RoPE. `true` for Qwen3; `false` for
    /// Qwen2 (which has no QK-norm). Governs both the forward and `param_list`.
    pub qk_norm: bool,
    /// Bias on the q/k/v projections. `false` for Qwen3 (bias-free); `true` for
    /// Qwen2 (q/k/v carry a bias, o/gate/up/down do not).
    pub attn_bias: bool,
    /// `Some` selects LoRA fine-tuning (frozen base + adapters); `None` is a
    /// full (all-parameter) model.
    pub lora: Option<LoraCfg>,
}

impl QwenConfig {
    /// A tiny config for tests / gradient checks. Exercises GQA (4 q / 2 kv
    /// heads, group 2), a decoupled `head_dim` (8, vs d_model/n_heads = 4),
    /// QK-norm, RoPE-base, SwiGLU and the tied head.
    pub fn tiny() -> QwenConfig {
        QwenConfig {
            vocab: 23,
            block_size: 12,
            n_layers: 2,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            d_ff: 32,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// The published Qwen3-0.6B shape (from its `config.json`).
    pub fn qwen3_0_6b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 28,
            d_model: 1024,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 3072,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// The published Qwen3-1.7B shape (from its `config.json`) — the text
    /// decoder used by Qwen3-ASR (`hidden_size` 2048, 28 layers, GQA 16/8,
    /// `head_dim` 128 so `q_dim` 2048, SwiGLU `d_ff` 6144, tied, θ 1e6).
    pub fn qwen3_1_7b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 28,
            d_model: 2048,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 6144,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// The published Qwen3-4B shape (from its `config.json`) — the text encoder
    /// used by Z-Image and FLUX.2 (`hidden_size` 2560, 36 layers, GQA 32/8,
    /// `head_dim` 128 so `q_dim` 4096 ≠ 2560, SwiGLU `d_ff` 9728, tied).
    pub fn qwen3_4b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 36,
            d_model: 2560,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 9728,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }


    /// The published Qwen3-8B shape (from its `config.json`) — the text encoder
    /// used by FLUX.2 Klein 9B (`hidden_size` 4096, 36 layers, GQA 32/8,
    /// `head_dim` 128, SwiGLU `d_ff` 12288, untied `lm_head`).
    pub fn qwen3_8b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 36,
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 12288,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: false,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// Apply derived defaults (head_dim = d_model/n_heads when unset).
    pub fn with_defaults(mut self) -> Self {
        if self.head_dim == 0 {
            self.head_dim = self.d_model / self.n_heads;
        }
        self
    }

    /// Query projection width = `n_heads * head_dim` (may differ from d_model).
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    /// Key/Value projection width = `n_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
    /// Query heads per kv head (GQA grouping factor).
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }
    /// The lm_head parameter name (tied -> the embedding table).
    pub fn head_weight(&self) -> &'static str {
        if self.tie_embeddings {
            "tok.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub fn to_json(&self) -> Value {
        let mut v = serde_json::json!({
            "model": "qwen",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim, "d_ff": self.d_ff,
            "rope_theta": self.rope_theta, "rms_norm_eps": self.rms_eps,
            "tie_word_embeddings": self.tie_embeddings
        });
        // A LoRA checkpoint must round-trip its adapter shape, or `param_list()`
        // rebuilds without the `.lora_a`/`.lora_b` names on load and the trained
        // adapters are silently dropped (see crates/qwen/tests/lora_roundtrip.rs).
        if let Some(l) = &self.lora {
            v["lora"] = serde_json::json!({
                "rank": l.rank, "alpha": l.alpha, "targets": l.targets,
            });
        }
        v
    }

    pub fn from_json(c: &Value) -> QwenConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        QwenConfig {
            vocab: g("vocab_size", 23),
            block_size: g("block_size", 12),
            n_layers: g("n_layers", 2),
            d_model: g("d_model", 16),
            n_heads: g("n_heads", 4),
            n_kv_heads: g("n_kv_heads", 2),
            head_dim: g("head_dim", 8),
            d_ff: g("d_ff", 32),
            rope_theta: gf("rope_theta", 1.0e6),
            rms_eps: gf("rms_norm_eps", 1e-6),
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(true),
            // Qwen3 default (QK-norm on, bias-free); a Qwen2 loader sets these.
            qk_norm: c["qk_norm"].as_bool().unwrap_or(true),
            attn_bias: c["attention_bias"].as_bool().unwrap_or(false),
            lora: c.get("lora").and_then(|l| l.as_object()).map(|l| LoraCfg {
                rank: l["rank"].as_u64().unwrap_or(0) as u32,
                alpha: l["alpha"].as_f64().unwrap_or(0.0) as f32,
                targets: l["targets"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }),
        }
        .with_defaults()
    }

    /// Qwen2 shape helper: QK-norm **off**, qkv **bias on** (the deltas vs Qwen3).
    /// `head_dim = d_model / n_heads`. Used by FastVLM's decoder.
    pub fn qwen2(vocab: u32, n_layers: u32, d_model: u32, n_heads: u32, n_kv_heads: u32, d_ff: u32, tie: bool) -> QwenConfig {
        QwenConfig {
            vocab,
            block_size: 2048,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim: d_model / n_heads,
            d_ff,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: tie,
            qk_norm: false,
            attn_bias: true,
            lora: None,
        }
    }

    /// `Qwen2-0.5B` (FastVLM-0.5B decoder).
    pub fn qwen2_0_5b() -> QwenConfig {
        Self::qwen2(151936, 24, 896, 14, 2, 4864, true)
    }
    /// `Qwen2-1.5B` (FastVLM-1.5B decoder).
    pub fn qwen2_1_5b() -> QwenConfig {
        Self::qwen2(151936, 28, 1536, 12, 2, 8960, true)
    }
    /// `Qwen2-7B` (FastVLM-7B decoder; untied head, vocab 152064).
    pub fn qwen2_7b() -> QwenConfig {
        Self::qwen2(152064, 28, 3584, 28, 4, 18944, false)
    }

    /// Parameter list: `(name, numel)`. With LoRA, targeted projections become a
    /// frozen base + trainable `A`/`B` adapters (see [`crate::model`]).
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let hq = self.q_dim() as usize;
        let hkv = self.kv_dim() as usize;
        let hd = self.head_dim as usize;
        let r = self.lora.as_ref().map(|l| l.rank as usize);

        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];
        // A linear `[out, in]` either as a plain trainable weight, or (LoRA on a
        // targeted leaf) as a frozen base + A[r,in] + B[out,r] adapters.
        let lin = |out_v: &mut Vec<(String, usize)>, name: String, leaf: &str, o: usize, i: usize| {
            let lora_here = r.is_some()
                && self
                    .lora
                    .as_ref()
                    .map(|l| l.targets_leaf(leaf))
                    .unwrap_or(false);
            if let (true, Some(rk)) = (lora_here, r) {
                out_v.push((name.clone(), o * i)); // frozen base
                out_v.push((format!("{name}.lora_a"), rk * i));
                out_v.push((format!("{name}.lora_b"), o * rk));
            } else {
                out_v.push((name, o * i));
            }
        };

        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            lin(&mut out, p("attn.wq.weight"), "wq", hq, d);
            lin(&mut out, p("attn.wk.weight"), "wk", hkv, d);
            lin(&mut out, p("attn.wv.weight"), "wv", hkv, d);
            if self.attn_bias {
                // Qwen2: q/k/v projections carry a bias (o/gate/up/down do not).
                out.push((p("attn.wq.bias"), hq));
                out.push((p("attn.wk.bias"), hkv));
                out.push((p("attn.wv.bias"), hkv));
            }
            if self.qk_norm {
                out.push((p("attn.q_norm.weight"), hd));
                out.push((p("attn.k_norm.weight"), hd));
            }
            lin(&mut out, p("attn.wo.weight"), "wo", d, hq);
            out.push((p("ln2.weight"), d));
            lin(&mut out, p("mlp.gate.weight"), "gate", ff, d);
            lin(&mut out, p("mlp.up.weight"), "up", ff, d);
            lin(&mut out, p("mlp.down.weight"), "down", d, ff);
        }
        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }
        out
    }
}
