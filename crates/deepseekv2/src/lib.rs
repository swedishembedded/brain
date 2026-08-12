// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-V2-family **MHA** decoder - the language model DeepSeek-OCR reads
//! its document tokens out of.
//!
//! Despite the `deepseek2` family name this is **not** an MLA decoder. The real
//! checkpoint (`ggml-org/DeepSeek-OCR-GGUF`, `general.architecture =
//! "deepseek2-ocr"`) carries square `q_proj`/`k_proj`/`v_proj`/`o_proj` weights
//! with `head_count == head_count_kv`, no biases, and none of DeepSeek-V2's MLA
//! tensors - it is a `deepseek2`-family *config* over a plain *multi-head*
//! attention. `crates/glm` is the MLA + DSA-indexer + MTP decoder for a
//! different model family and shares no attention code with this crate.
//!
//! Per pre-norm block (RMSNorm everywhere, eps 1e-6, no bias anywhere):
//! ```text
//!   h    = RMSNorm(x)·ln1
//!   q,k,v = h·Wq^T, h·Wk^T, h·Wv^T           (MHA: n_kv_heads == n_heads)
//!   q,k  = RoPE_neox(q), RoPE_neox(k)        (half-split, theta 10000, FULL head_dim)
//!   x   += (softmax(qk^T/sqrt(head_dim) + causal)·v)·Wo^T
//!   h    = RMSNorm(x)·ln2
//!   x   += dense SwiGLU MLP                  (blocks [0, n_dense_layers))
//!        | Σ_topk gate_e·SwiGLU_e(h) + SwiGLU_shared(h)   (the rest)
//!   logits = lm_head·RMSNorm(x)·norm  (untied) ;  loss = masked cross-entropy
//! ```
//!
//! **Router policy.** The routed experts use a plain **softmax** router with
//! **no** top-k renormalisation and routed scaling **1.0** - llama.cpp's
//! compiled-in defaults for this architecture, which is what actually runs
//! since the GGUF carries no `expert_weights_norm`/`expert_weights_scale` key
//! at all. That pair (`norm_topk_prob` / `routed_scaling`) is exactly why this
//! decoder cannot reuse `crates/moe`'s router configuration, and both knobs are
//! spelled explicitly on [`config::DeepseekV2Config`] rather than defaulted, so
//! a forward and its backward can never disagree about them.
//!
//! **Shared experts stay fused.** The checkpoint's `*_shexp` tensors are one
//! `n_shared_experts * moe_intermediate_size`-wide SwiGLU with no shared-expert
//! gate tensor: the shared experts are summed **unweighted**, and an unweighted
//! sum of SwiGLU experts is exactly one SwiGLU of the summed width. This crate
//! therefore drives `model::moe::shared_expert_fwd`/`shared_expert_bwd`'s
//! **`None`** (ungated) arm over the fused weight - one matmul, not two, and
//! arithmetically identical to the reference.
//!
//! ## Scope
//!
//! Config + forward + backward + deterministic init + GGUF import + a gradient
//! check + `O(T²)`-recompute greedy decode
//! ([`model::DeepseekV2::generate_greedy`]). Deliberately **out of scope** for
//! this phase, and listed here rather than half-built: LoRA fine-tuning, INT8
//! quantization, cross-GPU tensor/expert sharding, and paged-KV incremental
//! serving - so of the two decode tiers the sibling decoders keep
//! (`sample::generate` recompute, `sample::generate_kv` cached), only the
//! recompute one exists here. `crates/qwen35moe` - the crate
//! this one's decoder shape is modelled on - has all four; this crate copies its
//! *decoder*, not its production surface, and each of those is additive on top
//! of the parameter layout below rather than a change to it.

pub mod config;
pub mod import;
pub mod init;
pub mod model;

pub use config::DeepseekV2Config;
pub use init::init_weights;
pub use model::{DeepseekV2, IGNORE, PIPELINES};
