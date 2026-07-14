// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM-5.2 (`glm_moe_dsa`) configuration + parameter layout.
//!
//! MLA attention (low-rank q/kv, decoupled nope/rope head split), a sigmoid
//! `noaux_tc` MoE router with a shared expert, and a `first_k_dense_replace`
//! dense→MoE schedule. Norms are RMSNorm; no projection biases
//! (`attention_bias=false` in GLM-5.2); untied `lm_head`.
//!
//! **Brain internal layout note (vs HuggingFace):** to keep every RoPE / matmul
//! target a *contiguous* buffer (no per-head `[nope|rope]` interleaving), brain
//! *splits* the fused HF projections into separate weights:
//!   * `q_b_proj`  → `q_b_nope` `[H*qk_nope, q_lora]` + `q_b_rope` `[H*qk_rope, q_lora]`
//!   * `kv_b_proj` → `kv_b_nope` `[H*qk_nope, kv_lora]` + `kv_b_v` `[H*v_head, kv_lora]`
//!   * `kv_a_proj_with_mqa` → `kv_a_c` `[kv_lora, d]` + `kv_a_rope` `[qk_rope, d]`
//! The HF→brain importer row-permutes the fused matrices into these splits.

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct GlmConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,

    // --- MLA (Multi-head Latent Attention) ---
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,

    // --- MoE ---
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32, // top-k
    pub moe_intermediate_size: u32,
    pub intermediate_size: u32, // dense-MLP inner dim (first_k_dense layers)
    pub first_k_dense_replace: u32,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,

    // --- shared ---
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub tie_embeddings: bool,

    // --- DSA indexer (Phase 2; ignored while `index_topk >= block_size`) ---
    pub index_topk: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    /// Per-layer indexer mode: `true` = `full` (runs its own indexer), `false` =
    /// `shared` (reuses the previous full layer's top-k). Empty ⇒ every layer full.
    pub indexer_full: Vec<bool>,
}

impl GlmConfig {
    /// Combined query/key head dim (nope + rope) used for attention scores.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// Width of the all-heads nope block (`H * qk_nope_head_dim`).
    pub fn nope_dim(&self) -> u32 {
        self.n_heads * self.qk_nope_head_dim
    }
    /// Width of the all-heads query rope block (`H * qk_rope_head_dim`).
    pub fn q_rope_dim(&self) -> u32 {
        self.n_heads * self.qk_rope_head_dim
    }
    /// Width of the all-heads value block (`H * v_head_dim`).
    pub fn v_dim(&self) -> u32 {
        self.n_heads * self.v_head_dim
    }
    /// Is layer `l` a dense-MLP layer (vs a MoE layer)?
    pub fn is_dense_layer(&self, l: u32) -> bool {
        l < self.first_k_dense_replace
    }
    /// Shared-expert inner dim (`moe_intermediate_size * n_shared_experts`).
    pub fn shared_ff(&self) -> u32 {
        self.moe_intermediate_size * self.n_shared_experts
    }
    /// The lm_head parameter name (tied ⇒ the embedding table).
    pub fn head_weight(&self) -> &'static str {
        if self.tie_embeddings {
            "tok.weight"
        } else {
            "lm_head.weight"
        }
    }

    /// A tiny config for tests / gradient checks. Exercises MLA (low-rank q/kv,
    /// nope/rope split), the sigmoid MoE router + shared expert, and one dense +
    /// one MoE layer. `index_topk >= block_size` ⇒ dense (all-pass) attention.
    pub fn tiny() -> GlmConfig {
        GlmConfig {
            vocab: 23,
            block_size: 12,
            n_layers: 2,
            d_model: 16,
            n_heads: 2,
            q_lora_rank: 12,
            kv_lora_rank: 8,
            qk_nope_head_dim: 6,
            qk_rope_head_dim: 4,
            v_head_dim: 8,
            n_routed_experts: 3,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            intermediate_size: 24,
            first_k_dense_replace: 1,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 2.5,
            norm_topk_prob: true,
            rope_theta: 1.0e4,
            rms_eps: 1e-5,
            tie_embeddings: false,
            index_topk: 4096,
            index_n_heads: 2,
            index_head_dim: 8,
            indexer_full: Vec::new(),
        }
    }

    /// The published GLM-5.2 shape (`configs/glm-5.2-config.json`). Not runnable
    /// locally at full size; used for import shape validation and reference.
    pub fn glm5_2() -> GlmConfig {
        GlmConfig {
            vocab: 154880,
            block_size: 4096,
            n_layers: 78,
            d_model: 6144,
            n_heads: 64,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_nope_head_dim: 192,
            qk_rope_head_dim: 64,
            v_head_dim: 256,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_intermediate_size: 2048,
            intermediate_size: 12288,
            first_k_dense_replace: 3,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 2.5,
            norm_topk_prob: true,
            rope_theta: 8.0e6,
            rms_eps: 1e-5,
            tie_embeddings: false,
            index_topk: 2048,
            index_n_heads: 32,
            index_head_dim: 128,
            indexer_full: Vec::new(),
        }
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "glm",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads,
            "q_lora_rank": self.q_lora_rank, "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim, "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "n_routed_experts": self.n_routed_experts, "n_shared_experts": self.n_shared_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size, "intermediate_size": self.intermediate_size,
            "first_k_dense_replace": self.first_k_dense_replace,
            "n_group": self.n_group, "topk_group": self.topk_group,
            "routed_scaling_factor": self.routed_scaling_factor, "norm_topk_prob": self.norm_topk_prob,
            "rope_theta": self.rope_theta, "rms_norm_eps": self.rms_eps,
            "tie_word_embeddings": self.tie_embeddings,
            "index_topk": self.index_topk, "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim
        })
    }

    pub fn from_json(c: &Value) -> GlmConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let gb = |k: &str, d: bool| c[k].as_bool().unwrap_or(d);
        GlmConfig {
            vocab: g("vocab_size", 23),
            block_size: g("block_size", 12),
            n_layers: g("n_layers", 2),
            d_model: g("d_model", 16),
            n_heads: g("n_heads", 2),
            q_lora_rank: g("q_lora_rank", 12),
            kv_lora_rank: g("kv_lora_rank", 8),
            qk_nope_head_dim: g("qk_nope_head_dim", 6),
            qk_rope_head_dim: g("qk_rope_head_dim", 4),
            v_head_dim: g("v_head_dim", 8),
            n_routed_experts: g("n_routed_experts", 3),
            n_shared_experts: g("n_shared_experts", 1),
            num_experts_per_tok: g("num_experts_per_tok", 2),
            moe_intermediate_size: g("moe_intermediate_size", 16),
            intermediate_size: g("intermediate_size", 24),
            first_k_dense_replace: g("first_k_dense_replace", 1),
            n_group: g("n_group", 1),
            topk_group: g("topk_group", 1),
            routed_scaling_factor: gf("routed_scaling_factor", 2.5),
            norm_topk_prob: gb("norm_topk_prob", true),
            rope_theta: gf("rope_theta", 1.0e4),
            rms_eps: gf("rms_norm_eps", 1e-5),
            tie_embeddings: gb("tie_word_embeddings", false),
            index_topk: g("index_topk", 4096),
            index_n_heads: g("index_n_heads", 2),
            index_head_dim: g("index_head_dim", 8),
            indexer_full: Vec::new(),
        }
    }

    /// Parameter list `(name, numel)` — defines naming + ordering for
    /// save/load/import and buffer allocation. Linear `[out, in]` row-major
    /// matches HF `nn.Linear.weight`, so imports never transpose.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let v = self.vocab as usize;
        let ql = self.q_lora_rank as usize;
        let kvl = self.kv_lora_rank as usize;
        let nope = self.nope_dim() as usize;
        let qrope = self.q_rope_dim() as usize;
        let rope1 = self.qk_rope_head_dim as usize; // shared MQA key rope (single head)
        let vd = self.v_dim() as usize;
        let moe_ff = self.moe_intermediate_size as usize;
        let dense_ff = self.intermediate_size as usize;
        let shared_ff = self.shared_ff() as usize;
        let e = self.n_routed_experts;

        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];
        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("input_ln.weight"), d));
            // MLA attention (no biases in GLM-5.2).
            out.push((p("attn.q_a.weight"), ql * d));
            out.push((p("attn.q_a_norm.weight"), ql));
            out.push((p("attn.q_b_nope.weight"), nope * ql));
            out.push((p("attn.q_b_rope.weight"), qrope * ql));
            out.push((p("attn.kv_a_c.weight"), kvl * d));
            out.push((p("attn.kv_a_rope.weight"), rope1 * d));
            out.push((p("attn.kv_a_norm.weight"), kvl));
            out.push((p("attn.kv_b_nope.weight"), nope * kvl));
            out.push((p("attn.kv_b_v.weight"), vd * kvl));
            out.push((p("attn.o.weight"), d * vd));
            out.push((p("post_ln.weight"), d));
            // MLP: dense for the first `first_k_dense_replace` layers, else MoE.
            if self.is_dense_layer(l) {
                out.push((p("mlp.gate.weight"), dense_ff * d));
                out.push((p("mlp.up.weight"), dense_ff * d));
                out.push((p("mlp.down.weight"), d * dense_ff));
            } else {
                out.push((p("moe.router.weight"), e as usize * d));
                out.push((p("moe.router.bias"), e as usize)); // e_score_correction_bias (selection only)
                for ei in 0..e {
                    let ep = |s: &str| format!("blocks.{l}.moe.experts.{ei}.{s}");
                    out.push((ep("gate.weight"), moe_ff * d));
                    out.push((ep("up.weight"), moe_ff * d));
                    out.push((ep("down.weight"), d * moe_ff));
                }
                out.push((p("moe.shared.gate.weight"), shared_ff * d));
                out.push((p("moe.shared.up.weight"), shared_ff * d));
                out.push((p("moe.shared.down.weight"), d * shared_ff));
            }
        }
        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }
        out
    }
}
