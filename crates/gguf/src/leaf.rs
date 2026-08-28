// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! llama.cpp's per-block leaf-name vocabulary, shared by every decoder-LM
//! GGUF importer in this tree.
//!
//! [`crate::import::split_name`] already factors out the structure every
//! llama.cpp checkpoint shares (`token_embd`/`output`/`output_norm` plus a
//! flat `blk.{n}.{leaf}` space). This module factors out the next layer: the
//! leaf-name SPELLINGS themselves. `attn_q.weight`, `ffn_gate.weight`,
//! `ssm_alpha.weight` and so on mean the same thing in every architecture
//! llama.cpp converts them for - GQA, dense FFN, MoE FFN and Gated-DeltaNet/
//! SSM are building blocks reused across models, not per-model inventions, so
//! re-spelling this table inside every model's own `classify` is the exact
//! duplication AGENTS.md's "one implementation" rule exists to catch (`rmsnorm`
//! once existed seven times in this tree for the same reason).
//!
//! What stays per-model is [`Role`] -> that model's OWN brain-parameter
//! suffix (`qwen3` writes `attn.wq.weight`; `qwen35`/`qwen35moe` write
//! `self_attn.q_proj.weight`) and which roles an architecture even HAS (a
//! dense decoder has no [`Role::SsmAlpha`]; a non-MoE decoder has no
//! [`Role::FfnGateExps`]). This module only answers "what IS this leaf",
//! never "what does model X call it" or "does model X have one of these".
//!
//! One leaf spelling can be a genuine llama.cpp naming inconsistency across
//! architectures for the *same* structural role - `ffn_norm.weight` (plain
//! decoders) and `post_attention_norm.weight` (the Qwen3.5 family) are both
//! "the norm the residual sees before the FFN"; [`role`] accepts either
//! spelling for [`Role::FfnNorm`] rather than making every caller know both.

/// One structural position in a llama.cpp decoder-LM block, independent of
/// which architecture's checkpoint it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The norm before attention sees the residual.
    AttnNorm,
    /// The norm before the FFN/MoE sees the residual (llama.cpp spells this
    /// `ffn_norm.weight` on a plain decoder, `post_attention_norm.weight` on
    /// the Qwen3.5 family - both map here).
    FfnNorm,

    // -- full (GQA) attention --
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    AttnQNorm,
    AttnKNorm,
    AttnQBias,
    AttnKBias,
    AttnVBias,

    // -- dense FFN --
    FfnGate,
    FfnUp,
    FfnDown,

    // -- MoE FFN --
    FfnGateInp,
    FfnGateInpShexp,
    FfnGateShexp,
    FfnUpShexp,
    FfnDownShexp,
    /// `[n_experts, moe_ff, hidden]` - the caller fans this out per-expert
    /// with [`super::import::Mapped::expert_stack`]; this table only says
    /// which stack it is.
    FfnGateExps,
    FfnUpExps,
    FfnDownExps,

    // -- Gated DeltaNet / SSM (linear attention) --
    /// Fused qkv projection (`in_proj_qkv`).
    AttnQkv,
    /// The gated-RMSNorm's `z` input (`in_proj_z`) - spelled "gate" in
    /// llama.cpp's GGUF because it feeds a norm gate, not a router.
    AttnGate,
    SsmAlpha,
    SsmBeta,
    SsmConv1d,
    SsmA,
    SsmDtBias,
    SsmNorm,
    SsmOut,
}

/// Map one `blk.N.{leaf}` suffix (as split out by
/// [`super::import::split_name`]) to its structural [`Role`], or `None` if
/// this table has no opinion. The caller's classifier decides what an
/// unrecognized leaf means for ITS architecture: an error for a strict
/// importer, or a counted drop for a vision/MTP tensor the importer chooses
/// not to import.
pub fn role(leaf: &str) -> Option<Role> {
    Some(match leaf {
        "attn_norm.weight" => Role::AttnNorm,
        "ffn_norm.weight" | "post_attention_norm.weight" => Role::FfnNorm,

        "attn_q.weight" => Role::AttnQ,
        "attn_k.weight" => Role::AttnK,
        "attn_v.weight" => Role::AttnV,
        "attn_output.weight" => Role::AttnOutput,
        "attn_q_norm.weight" => Role::AttnQNorm,
        "attn_k_norm.weight" => Role::AttnKNorm,
        "attn_q.bias" => Role::AttnQBias,
        "attn_k.bias" => Role::AttnKBias,
        "attn_v.bias" => Role::AttnVBias,

        "ffn_gate.weight" => Role::FfnGate,
        "ffn_up.weight" => Role::FfnUp,
        "ffn_down.weight" => Role::FfnDown,

        "ffn_gate_inp.weight" => Role::FfnGateInp,
        "ffn_gate_inp_shexp.weight" => Role::FfnGateInpShexp,
        "ffn_gate_shexp.weight" => Role::FfnGateShexp,
        "ffn_up_shexp.weight" => Role::FfnUpShexp,
        "ffn_down_shexp.weight" => Role::FfnDownShexp,
        "ffn_gate_exps.weight" => Role::FfnGateExps,
        "ffn_up_exps.weight" => Role::FfnUpExps,
        "ffn_down_exps.weight" => Role::FfnDownExps,

        "attn_qkv.weight" => Role::AttnQkv,
        "attn_gate.weight" => Role::AttnGate,
        "ssm_alpha.weight" => Role::SsmAlpha,
        "ssm_beta.weight" => Role::SsmBeta,
        "ssm_conv1d.weight" => Role::SsmConv1d,
        "ssm_a" => Role::SsmA,
        "ssm_dt.bias" => Role::SsmDtBias,
        "ssm_norm.weight" => Role::SsmNorm,
        "ssm_out.weight" => Role::SsmOut,

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffn_norm_and_post_attention_norm_are_the_same_role() {
        assert_eq!(role("ffn_norm.weight"), Some(Role::FfnNorm));
        assert_eq!(role("post_attention_norm.weight"), Some(Role::FfnNorm));
    }

    /// Walks every `Role` variant against its real leaf spelling(s), so a
    /// variant nothing maps to (dead code) or a spelling that silently
    /// stopped matching cannot hide behind the match arm just "looking"
    /// exhaustive on inspection.
    #[test]
    fn every_role_variant_round_trips_through_its_real_leaf_spelling() {
        let cases: &[(Role, &str)] = &[
            (Role::AttnNorm, "attn_norm.weight"),
            (Role::FfnNorm, "ffn_norm.weight"),
            (Role::FfnNorm, "post_attention_norm.weight"),
            (Role::AttnQ, "attn_q.weight"),
            (Role::AttnK, "attn_k.weight"),
            (Role::AttnV, "attn_v.weight"),
            (Role::AttnOutput, "attn_output.weight"),
            (Role::AttnQNorm, "attn_q_norm.weight"),
            (Role::AttnKNorm, "attn_k_norm.weight"),
            (Role::AttnQBias, "attn_q.bias"),
            (Role::AttnKBias, "attn_k.bias"),
            (Role::AttnVBias, "attn_v.bias"),
            (Role::FfnGate, "ffn_gate.weight"),
            (Role::FfnUp, "ffn_up.weight"),
            (Role::FfnDown, "ffn_down.weight"),
            (Role::FfnGateInp, "ffn_gate_inp.weight"),
            (Role::FfnGateInpShexp, "ffn_gate_inp_shexp.weight"),
            (Role::FfnGateShexp, "ffn_gate_shexp.weight"),
            (Role::FfnUpShexp, "ffn_up_shexp.weight"),
            (Role::FfnDownShexp, "ffn_down_shexp.weight"),
            (Role::FfnGateExps, "ffn_gate_exps.weight"),
            (Role::FfnUpExps, "ffn_up_exps.weight"),
            (Role::FfnDownExps, "ffn_down_exps.weight"),
            (Role::AttnQkv, "attn_qkv.weight"),
            (Role::AttnGate, "attn_gate.weight"),
            (Role::SsmAlpha, "ssm_alpha.weight"),
            (Role::SsmBeta, "ssm_beta.weight"),
            (Role::SsmConv1d, "ssm_conv1d.weight"),
            (Role::SsmA, "ssm_a"),
            (Role::SsmDtBias, "ssm_dt.bias"),
            (Role::SsmNorm, "ssm_norm.weight"),
            (Role::SsmOut, "ssm_out.weight"),
        ];
        for (want, leaf) in cases {
            assert_eq!(role(leaf), Some(*want), "leaf {leaf:?} should map to {want:?}");
        }
    }

    #[test]
    fn an_unrecognized_leaf_has_no_role() {
        assert_eq!(role("attn_wibble.weight"), None);
    }
}
