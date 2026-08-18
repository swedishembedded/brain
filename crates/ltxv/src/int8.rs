// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 STORAGE format for the DiT's own linear weights (M9 slice: smaller
//! checkpoint on disk / in host RAM only - no compute-time change).
//!
//! This reuses `model::int8`'s shared per-channel symmetric int8 primitives
//! (`quantize_weight` / `dequantize_weight`) - the same ones zimage's DiT,
//! the Qwen encoder/decoder, and FLUX.2's DiT already use for exactly this
//! purpose. Unlike FLUX.2's own int8 tier, there is no compute-time DP4A
//! activation-quantization path here, and no new WGSL kernel: neither
//! `crate::dit::LtxDit::forward` nor `crate::dit::LtxAvDit::forward` dispatch
//! an int8 kernel of any kind. A quantized checkpoint is meant to be
//! dequantized back to plain f32 (see [`dequantize_tensors`]) before either
//! forward runs, unchanged from how it runs today.
//!
//! The never-quantize list is this port's own roadmap ledger's "upstream
//! never quantizes" set: `patchify_proj`, every `*adaln_single*` table,
//! `caption_projection`, `proj_out`, their `audio_` twins, `to_gate_logits`,
//! and `scale_shift_table` (every variant: the top-level output table, each
//! block's own `transformer_blocks.N.scale_shift_table` /
//! `prompt_scale_shift_table`, the audio twins, and the A<->V cross-attention
//! `scale_shift_table_a2v_ca_{video,audio}` tables) - modulation/conditioning
//! tables whose precision the whole per-token adaLN combine rides on, plus
//! the projections that set the model's numeric scale at the very first and
//! very last op.

use std::collections::HashMap;

use vae::blocks::Tensors;

/// True for a tensor name this port never quantizes - matched by substring
/// against the REAL names `crate::dit::dit_tensor_manifest` emits (e.g.
/// `patchify_proj.weight`, `adaln_single.linear.weight`,
/// `transformer_blocks.0.scale_shift_table`, `proj_out.bias`). The audio
/// stream and the A<->V cross-attention adaLN tables reuse these exact
/// substrings under an `audio_` / `av_ca_` prefix (e.g.
/// `audio_adaln_single...`, `av_ca_video_scale_shift_adaln_single...`,
/// `audio_scale_shift_table`), so one pattern list covers both streams
/// without a separate prefix check.
pub fn is_never_quantized(tensor_name: &str) -> bool {
    const NEVER_QUANTIZE_SUBSTRINGS: [&str; 6] =
        ["patchify_proj", "adaln_single", "caption_projection", "proj_out", "to_gate_logits", "scale_shift_table"];
    NEVER_QUANTIZE_SUBSTRINGS.iter().any(|pattern| tensor_name.contains(pattern))
}

/// One int8-eligible weight after [`quantize_tensors`]: `model::int8::
/// quantize_weight`'s packed `[n, k/4]` u32 words plus its per-row `[n]`
/// f32 scale, alongside the logical `[n, k]` shape - needed to dequantize,
/// since the packed shape alone cannot recover `k` (same reason `model::
/// int8::upload_dequantized` takes `n`/`k` as separate arguments).
pub struct QuantizedWeight {
    pub shape: Vec<usize>,
    pub packed: Vec<u32>,
    pub scale: Vec<f32>,
}

/// A DiT weight map split by [`quantize_tensors`] into its int8-eligible
/// tensors (packed) and everything else, untouched in `full`: every
/// never-quantized name, plus anything that fails the eligibility test
/// (rank != 2, or `k % 4 != 0`) - biases and 1D norm/scale vectors.
pub struct QuantizedTensors {
    pub full: Tensors,
    pub int8: HashMap<String, QuantizedWeight>,
}

/// A tensor is int8-storage-eligible iff it is a plain `[n, k]` matrix
/// (`k % 4 == 0`, the packing width `model::int8::quantize_weight` requires)
/// and its name is not on the never-quantize list. Every eligible weight in
/// this crate's real tensor manifest is an attention/MLP projection
/// (`to_q`/`to_k`/`to_v`/`to_out.0`, `ff.net.0.proj`/`ff.net.2`, and their
/// audio/cross-attention counterparts) - never a bias or a norm gain, since
/// those are 1D.
fn is_eligible(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && shape[1].is_multiple_of(4) && !is_never_quantized(name)
}

/// Quantize every eligible 2D weight in `w` via
/// `model::int8::quantize_weight` - storage format only, no kernel and no
/// change to either DiT's forward dispatch. Returns the eligible tensors
/// packed in `int8` and everything else (never-quantized names, biases, 1D
/// norm/scale vectors) copied as-is into `full`.
pub fn quantize_tensors(w: &Tensors) -> QuantizedTensors {
    let mut full = Tensors::new();
    let mut int8 = HashMap::new();
    for (name, (shape, data)) in w {
        if is_eligible(name, shape) {
            let (n, k) = (shape[0], shape[1]);
            let (packed, scale) = model::int8::quantize_weight(data, n, k);
            int8.insert(name.clone(), QuantizedWeight { shape: shape.clone(), packed, scale });
        } else {
            full.insert(name.clone(), (shape.clone(), data.clone()));
        }
    }
    QuantizedTensors { full, int8 }
}

/// The inverse of [`quantize_tensors`]: reconstruct a plain-f32 [`Tensors`]
/// map with every int8-eligible tensor dequantized back via `model::int8::
/// dequantize_weight`. This is what a loader hands to `crate::dit::
/// LtxDit::new` / `crate::dit::LtxAvDit::new` - neither forward dispatches
/// an int8 kernel of its own yet, so a quantized checkpoint must be expanded
/// to f32 before either constructor sees it.
pub fn dequantize_tensors(q: &QuantizedTensors) -> Tensors {
    let mut out = q.full.clone();
    for (name, qw) in &q.int8 {
        let (n, k) = (qw.shape[0], qw.shape[1]);
        let data = model::int8::dequantize_weight(&qw.packed, &qw.scale, n, k);
        out.insert(name.clone(), (qw.shape.clone(), data));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_quantized_names_are_excluded_from_eligibility() {
        for name in [
            "patchify_proj.weight",
            "audio_patchify_proj.weight",
            "adaln_single.linear.weight",
            "audio_adaln_single.linear.weight",
            "av_ca_video_scale_shift_adaln_single.linear.weight",
            "proj_out.weight",
            "audio_proj_out.weight",
            "scale_shift_table",
            "audio_scale_shift_table",
            "transformer_blocks.0.scale_shift_table",
            "transformer_blocks.0.prompt_scale_shift_table",
            "scale_shift_table_a2v_ca_video",
        ] {
            assert!(is_never_quantized(name), "{name} should be on the never-quantize list");
            assert!(!is_eligible(name, &[9, 64]), "{name} must not be int8-eligible even at a valid [n,k] shape");
        }
    }

    #[test]
    fn ordinary_projections_are_eligible_biases_and_norms_are_not() {
        assert!(is_eligible("transformer_blocks.0.attn1.to_q.weight", &[64, 64]));
        assert!(!is_eligible("transformer_blocks.0.attn1.to_q.bias", &[64])); // rank 1
        assert!(!is_eligible("transformer_blocks.0.attn1.q_norm.weight", &[64])); // rank 1
    }
}
