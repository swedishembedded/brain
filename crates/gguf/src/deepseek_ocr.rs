// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR's **language model** (`general.architecture = "deepseek2-ocr"`).
//!
//! A 12-layer, 1280-wide decoder: one leading dense block then 11 sparse-MoE
//! blocks (64 routed experts, top-6, plus shared experts), with plain MHA -
//! `head_count == head_count_kv == 10`, and the checkpoint carries square
//! `attn_q/attn_k/attn_v/attn_output` `[1280,1280]` weights with none of
//! DeepSeek-V2's MLA tensors (`attn_kv_a_mqa`/`attn_kv_b`/`attn_q_a`). It is a
//! `deepseek2`-family *config*, not a `deepseek2` *attention*.
//!
//! Verified against the real `DeepSeek-OCR-Q8_0.gguf` header (KV + tensor
//! name/shape manifest read directly off the file - see this crate's
//! `deepseek_ocr_real` test, which re-derives every number below from the
//! checkpoint rather than trusting this comment).
//!
//! ## Tensor mapping (GGUF dims are `ne[0]`-fastest; "torch" is the reversed,
//! row-major shape [`MmapGguf::shape`] already reports)
//!
//! | GGUF | torch shape | brain |
//! |---|---|---|
//! | `token_embd.weight` | `[129280,1280]` | `tok.weight` |
//! | `output.weight` | `[129280,1280]` | `lm_head.weight` (untied - the tensor exists) |
//! | `output_norm.weight` | `[1280]` | `norm.weight` |
//! | `blk.N.attn_norm.weight` | `[1280]` | `blocks.N.ln1.weight` |
//! | `blk.N.ffn_norm.weight` | `[1280]` | `blocks.N.ln2.weight` |
//! | `blk.N.attn_{q,k,v}.weight` | `[1280,1280]` | `blocks.N.self_attn.{q,k,v}_proj.weight` |
//! | `blk.N.attn_output.weight` | `[1280,1280]` | `blocks.N.self_attn.o_proj.weight` |
//! | `blk.0.ffn_{gate,up}.weight` | `[6848,1280]` | `blocks.0.mlp.{gate,up}.weight` (dense block) |
//! | `blk.0.ffn_down.weight` | `[1280,6848]` | `blocks.0.mlp.down.weight` |
//! | `blk.N.ffn_gate_inp.weight` | `[64,1280]` | `blocks.N.mlp.router.weight` |
//! | `blk.N.ffn_{gate,up}_exps.weight` | `[64,896,1280]` | `blocks.N.mlp.experts.{e}.{gate,up}.weight` (fan-out) |
//! | `blk.N.ffn_down_exps.weight` | `[64,1280,896]` | `blocks.N.mlp.experts.{e}.down.weight` |
//! | `blk.N.ffn_{gate,up}_shexp.weight` | `[1792,1280]` | `blocks.N.mlp.shared.{gate,up}.weight` |
//! | `blk.N.ffn_down_shexp.weight` | `[1280,1792]` | `blocks.N.mlp.shared.down.weight` |
//!
//! **The two shared experts stay fused, on purpose.** `expert_shared_count=2`
//! and `expert_feed_forward_length=896`, and the `*_shexp` tensors are
//! `2*896 = 1792` wide: llama.cpp concatenates the shared experts into one
//! MLP. There is no shared-expert *gate* tensor, so the shared experts are
//! summed unweighted - and an unweighted sum of two SwiGLU experts is
//! **exactly** one SwiGLU MLP of twice the width
//! (`Σ_s W_down^s · (act(W_gate^s x) ⊙ W_up^s x)` is the block-partitioned form
//! of the fused matmul). Splitting them would also be impossible to do
//! contiguously for `down` (its 1792 is the *inner* axis). So brain keeps the
//! fused form, which is arithmetically identical and one matmul instead of two.
//!
//! The brain-side names above mirror `crates/qwen35moe`'s scheme. The decoder
//! that consumes them (expected to live in `crates/deepseekv2`) is future
//! work; this module is the loader half only.

use checkpoint::gguf::MmapGguf;
use checkpoint::st::ModelCard;
use serde_json::Value;

use crate::import::{self, ImportStats, Mapped};
use crate::kv::ArchKv;

/// llama.cpp's `general.architecture` value for this model.
pub const GGUF_ARCHITECTURE: &str = "deepseek2-ocr";

/// DeepSeek-OCR's decoder hyperparameters, as carried by the GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekOcrConfig {
    pub vocab: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub max_position_embeddings: u32,
    pub rms_eps: f32,
    pub tie_embeddings: bool,

    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    /// Number of head dimensions RoPE rotates. See [`config_from_gguf`] for
    /// why this is not simply `{arch}.rope.dimension_count`.
    pub rotary_dim: u32,

    /// Leading **dense** blocks (`leading_dense_block_count`): blocks
    /// `[0, n_dense_layers)` carry a plain MLP, the rest carry the MoE.
    pub n_dense_layers: u32,
    /// The dense blocks' MLP width (`feed_forward_length`).
    pub ffn_hidden: u32,

    pub n_experts: u32,
    pub top_k: u32,
    pub moe_intermediate_size: u32,
    /// Shared experts, kept **fused** into a single `n_shared_experts *
    /// moe_intermediate_size` MLP (see this module's doc).
    pub n_shared_experts: u32,
    pub n_expert_groups: u32,
    pub n_expert_groups_used: u32,
}

impl DeepseekOcrConfig {
    /// Whether block `l` is sparse (MoE) rather than dense.
    pub fn is_moe_layer(&self, l: u32) -> bool {
        l >= self.n_dense_layers
    }

    /// The fused shared-expert MLP's width.
    pub fn shared_intermediate_size(&self) -> u32 {
        self.n_shared_experts * self.moe_intermediate_size
    }

    /// Total attention projection width (`n_heads * head_dim`).
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }

    /// The canonical output manifest: every brain-side tensor name and its
    /// element count. This is the import's coverage contract in both
    /// directions, and the future decoder's parameter layout.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let v = self.vocab as usize;
        let qd = self.q_dim() as usize;
        let kvd = (self.n_kv_heads * self.head_dim) as usize;

        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];
        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("self_attn.q_proj.weight"), qd * d));
            out.push((p("self_attn.k_proj.weight"), kvd * d));
            out.push((p("self_attn.v_proj.weight"), kvd * d));
            out.push((p("self_attn.o_proj.weight"), d * qd));
            out.push((p("ln2.weight"), d));

            if self.is_moe_layer(l) {
                let ff = self.moe_intermediate_size as usize;
                let sff = self.shared_intermediate_size() as usize;
                out.push((p("mlp.router.weight"), self.n_experts as usize * d));
                for e in 0..self.n_experts {
                    out.push((p(&format!("mlp.experts.{e}.gate.weight")), ff * d));
                    out.push((p(&format!("mlp.experts.{e}.up.weight")), ff * d));
                    out.push((p(&format!("mlp.experts.{e}.down.weight")), d * ff));
                }
                out.push((p("mlp.shared.gate.weight"), sff * d));
                out.push((p("mlp.shared.up.weight"), sff * d));
                out.push((p("mlp.shared.down.weight"), d * sff));
            } else {
                let ff = self.ffn_hidden as usize;
                out.push((p("mlp.gate.weight"), ff * d));
                out.push((p("mlp.up.weight"), ff * d));
                out.push((p("mlp.down.weight"), d * ff));
            }
        }
        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }
        out
    }

    /// The config as it is stored in the produced checkpoint's header.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "deepseek-ocr",
            "vocab_size": self.vocab,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_eps,
            "tie_word_embeddings": self.tie_embeddings,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rotary_dim": self.rotary_dim,
            "first_k_dense_replace": self.n_dense_layers,
            "intermediate_size": self.ffn_hidden,
            "n_routed_experts": self.n_experts,
            "num_experts_per_tok": self.top_k,
            "moe_intermediate_size": self.moe_intermediate_size,
            "n_shared_experts": self.n_shared_experts,
            "n_group": self.n_expert_groups,
            "topk_group": self.n_expert_groups_used,
        })
    }
}

/// Derive [`DeepseekOcrConfig`] from a GGUF's KV metadata, falling back to the
/// real tensor shapes where the KV is absent or wrong.
///
/// Two fields are not simple KV reads:
///
/// - **`head_dim`**: the file declares no `attention.key_length`, so it is
///   derived from `blk.0.attn_q.weight`'s own torch shape (`[n_heads *
///   head_dim, d_model]`) - the tensor is ground truth, `d_model / n_heads`
///   would only coincidentally agree.
/// - **`rotary_dim`**: the file declares `rope.dimension_count = 0`, so this
///   treats 0 as "unset" and uses the full `head_dim` (128). **Verified
///   against llama.cpp**, not assumed: the converter copies HF's
///   `qk_rope_head_dim`, which DeepSeek-OCR's `language_config` sets to 0
///   because it is not an MLA model, and llama.cpp's OCR attention branch
///   never reads `n_rot` - it ropes `n_embd / n_head = 128` dimensions
///   explicitly. That branch also asserts `freq_base == 10000` (the library
///   default this function's fallback matches, since the file carries no
///   `rope.freq_base`) and uses the **NEOX** rope layout (rotate-halves, not
///   interleaved pairs) - the detail the future decoder must match.
///
/// Absent entirely, and therefore NOT read here: `expert_weights_norm`,
/// `expert_weights_scale`, `expert_gating_func`, `rope.freq_base`. Their
/// absence is faithful, not lossy - HF's own config leaves every one of them
/// at its default, so llama.cpp's compiled-in defaults reproduce the
/// reference exactly: **softmax** router (the `NONE` gating default is
/// rewritten to softmax for this architecture), **no** top-k renormalization
/// (`norm_topk_prob = false`), routed scaling **1.0** (a `0.0` scale elides
/// the scale op), and, with `expert_group_count = 1`, no group masking - plain
/// top-6 of 64. Those four facts belong to the forward pass, not the loader,
/// which is why they are recorded here rather than in a config field.
pub fn config_from_gguf(mg: &MmapGguf) -> Result<DeepseekOcrConfig, String> {
    let kv = ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;

    let n_layers = kv.req_u32("block_count")?;
    let d_model = kv.req_u32("embedding_length")?;
    let n_heads = kv.req_u32("attention.head_count")?;
    let n_kv_heads = kv.u32_or("attention.head_count_kv", n_heads);

    let q_shape = mg.shape("blk.0.attn_q.weight").ok_or("deepseek-ocr: missing blk.0.attn_q.weight")?;
    let q_rows = *q_shape.first().ok_or("deepseek-ocr: blk.0.attn_q.weight has no leading dim")? as u32;
    if n_heads == 0 || !q_rows.is_multiple_of(n_heads) {
        return Err(format!("deepseek-ocr: attn_q rows {q_rows} not divisible by head_count {n_heads}"));
    }
    let head_dim = kv.u32("attention.key_length").unwrap_or(q_rows / n_heads);

    let rope_dim_count = kv.u32_or("rope.dimension_count", 0);
    let rotary_dim = if rope_dim_count == 0 { head_dim } else { rope_dim_count };

    let vocab = mg
        .shape("token_embd.weight")
        .and_then(|s| s.first().copied())
        .ok_or("deepseek-ocr: missing token_embd.weight")? as u32;
    if let Some(declared) = kv.u32("vocab_size") {
        if declared != vocab {
            return Err(format!("deepseek-ocr: vocab_size={declared} disagrees with token_embd rows {vocab}"));
        }
    }

    Ok(DeepseekOcrConfig {
        vocab,
        n_layers,
        d_model,
        max_position_embeddings: kv.u32_or("context_length", 8192),
        rms_eps: kv.f32_or("attention.layer_norm_rms_epsilon", 1e-6),
        tie_embeddings: !mg.names().iter().any(|n| n == "output.weight"),

        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta: kv.f32_or("rope.freq_base", 10_000.0),
        rotary_dim,

        n_dense_layers: kv.u32_or("leading_dense_block_count", 0),
        ffn_hidden: kv.req_u32("feed_forward_length")?,

        n_experts: kv.req_u32("expert_count")?,
        top_k: kv.req_u32("expert_used_count")?,
        moe_intermediate_size: kv.req_u32("expert_feed_forward_length")?,
        n_shared_experts: kv.u32_or("expert_shared_count", 0),
        n_expert_groups: kv.u32_or("expert_group_count", 1),
        n_expert_groups_used: kv.u32_or("expert_group_used_count", 1),
    })
}

/// Classify one GGUF tensor name. An unrecognized name is an **error**: this
/// checkpoint has no vision/MTP/auxiliary tensors to drop (the vision tower
/// ships as a separate mmproj file, see [`crate::deepseek_ocr_vision`]), so
/// anything unmapped means the converter changed and the import must stop.
pub fn classify(name: &str, cfg: &DeepseekOcrConfig) -> Result<Mapped, String> {
    match name {
        "token_embd.weight" => return Ok(Mapped::Simple("tok.weight".to_string())),
        "output.weight" => return Ok(Mapped::Simple("lm_head.weight".to_string())),
        "output_norm.weight" => return Ok(Mapped::Simple("norm.weight".to_string())),
        _ => {}
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return Err(format!("unrecognized top-level tensor {name:?}"));
    };
    let Some((idx, leaf)) = rest.split_once('.') else {
        return Err(format!("malformed block tensor name {name:?}"));
    };
    let l: u32 = idx.parse().map_err(|_| format!("malformed block index in {name:?}"))?;
    if l >= cfg.n_layers {
        return Err(format!("{name}: block index {l} beyond block_count {}", cfg.n_layers));
    }
    let p = |s: &str| format!("blocks.{l}.{s}");
    let moe = cfg.is_moe_layer(l);
    let n_experts = cfg.n_experts as usize;

    let m = match leaf {
        "attn_norm.weight" => Mapped::Simple(p("ln1.weight")),
        "ffn_norm.weight" => Mapped::Simple(p("ln2.weight")),
        "attn_q.weight" => Mapped::Simple(p("self_attn.q_proj.weight")),
        "attn_k.weight" => Mapped::Simple(p("self_attn.k_proj.weight")),
        "attn_v.weight" => Mapped::Simple(p("self_attn.v_proj.weight")),
        "attn_output.weight" => Mapped::Simple(p("self_attn.o_proj.weight")),
        // Dense block MLP.
        "ffn_gate.weight" if !moe => Mapped::Simple(p("mlp.gate.weight")),
        "ffn_up.weight" if !moe => Mapped::Simple(p("mlp.up.weight")),
        "ffn_down.weight" if !moe => Mapped::Simple(p("mlp.down.weight")),
        // MoE block.
        "ffn_gate_inp.weight" if moe => Mapped::Simple(p("mlp.router.weight")),
        "ffn_gate_exps.weight" if moe => Mapped::expert_stack(l as usize, "gate", n_experts),
        "ffn_up_exps.weight" if moe => Mapped::expert_stack(l as usize, "up", n_experts),
        "ffn_down_exps.weight" if moe => Mapped::expert_stack(l as usize, "down", n_experts),
        // Shared experts stay fused - see this module's doc.
        "ffn_gate_shexp.weight" if moe => Mapped::Simple(p("mlp.shared.gate.weight")),
        "ffn_up_shexp.weight" if moe => Mapped::Simple(p("mlp.shared.up.weight")),
        "ffn_down_shexp.weight" if moe => Mapped::Simple(p("mlp.shared.down.weight")),
        other => {
            let kind = if moe { "MoE" } else { "dense" };
            return Err(format!("unrecognized {kind}-block leaf {other:?} in {name:?}"));
        }
    };
    Ok(m)
}

/// Import a DeepSeek-OCR language-model GGUF into brain's native format.
pub fn import(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let cfg = config_from_gguf(mg)?;
    let params = cfg.param_list();

    let mut card = ModelCard::new(id_override.unwrap_or("deepseek-ocr"), "deepseek-ocr");
    card.context_length = Some(cfg.max_position_embeddings as u64);
    card.param_count = Some(params.iter().map(|(_, n)| *n as u64).sum());

    import::to_st(mg, &params, &|n| classify(n, &cfg), out_path, &cfg.to_json(), Some(&card), "deepseek-ocr")
}
