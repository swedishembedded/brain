// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 STORAGE format for the flow-matching DiT's weights (storage only -
//! a smaller checkpoint in host RAM / on disk, no compute-path change).
//!
//! This reuses `model::int8`'s shared per-channel symmetric int8 primitives
//! (`quantize_weight`/`dequantize_weight`) - the same ones `ltxv::int8`
//! already uses for a different DiT, and the same mechanism zimage's DiT,
//! the Qwen encoder/decoder, and FLUX.2's DiT use for the same purpose.
//! There is no compute-time DP4A activation-quantization path here and no
//! new WGSL kernel: `crate::dit::forward` dispatches no int8 kernel of any
//! kind. A quantized checkpoint is meant to be dequantized back to plain f32
//! (see [`dequantize_tensors`]) before `crate::dit::from_tensors` sees it,
//! unchanged from how the DiT is imported/run today.
//!
//! Unlike `ltxv::int8`, this module operates directly on
//! `checkpoint::safetensors::StTensor` (name+shape+data already together)
//! rather than a `HashMap<String, (shape, data)>` map - `crate::dit::
//! from_tensors` already consumes exactly that `Vec<StTensor>` shape, so
//! there is no separate `Tensors` type to invent here.
//!
//! The never-quantize list is `proj_in.weight` and `proj_out.weight` (the
//! model's first and last projections, which set its numeric scale) plus
//! `time_embed.linear_1.weight`/`time_embed.linear_2.weight` (the
//! timestep-conditioning MLP every block's modulation ultimately rides on) -
//! the same "boundary/modulation tables stay full precision" reasoning
//! `ltxv::int8`'s own module doc gives for its `patchify_proj`/
//! `adaln_single`/`proj_out`/`scale_shift_table` list. Every other 2D weight
//! this predicate can see is one of the DiT's 6 per-block linears
//! (`attn.to_q/to_k/to_v/to_out.0`, `ff_in.weight`, `ff_out.weight`) -
//! exactly the set `crate::dit_lora::linear_shape` already treats as this
//! model's LoRA-eligible linears, for the same underlying reason (every
//! other weight is either a 1D norm/bias, a rank-3 conv kernel, or a
//! boundary/conditioning table).

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

/// True for a tensor name this port never quantizes - matched by substring
/// against the REAL names `crate::dit::from_tensors` reads (e.g.
/// `proj_in.weight`, `proj_out.weight`, `time_embed.linear_1.weight`,
/// `time_embed.linear_2.weight`). `"proj_out"` does not accidentally match
/// a per-block name: attention's output projection is named
/// `attn.to_out.0.weight` (substring `to_out`, not `proj_out`).
pub fn is_never_quantized(tensor_name: &str) -> bool {
    const NEVER_QUANTIZE_SUBSTRINGS: [&str; 3] = ["proj_in", "proj_out", "time_embed"];
    NEVER_QUANTIZE_SUBSTRINGS.iter().any(|pattern| tensor_name.contains(pattern))
}

/// One int8-eligible weight after [`quantize_tensors`]: `model::int8::
/// quantize_weight`'s packed `[n, k/4]` u32 words plus its `[n, k/32]` f32
/// scale, alongside the logical `[n, k]` shape - needed to dequantize, since
/// the packed shape alone cannot recover `k`.
pub struct QuantizedWeight {
    pub shape: Vec<usize>,
    pub packed: Vec<u32>,
    pub scale: Vec<f32>,
}

/// The DiT's tensor list split by [`quantize_tensors`] into its int8-eligible
/// tensors (packed) and everything else, untouched in `full`: every
/// never-quantized name, plus anything that fails the eligibility test
/// (rank != 2, or `k % 4 != 0`) - the rank-3 conv kernels, `time_proj.weight`
/// (`k=1`), and every 1D norm/bias.
pub struct QuantizedTensors {
    pub full: Vec<StTensor>,
    pub int8: HashMap<String, QuantizedWeight>,
}

/// A tensor is int8-storage-eligible iff it is a plain `[n, k]` matrix
/// (`k % 32 == 0`, the scale-group width `model::int8::quantize_weight`
/// requires - `model::int8::GROUP`) and its name is not on the never-quantize
/// list. In this crate's real
/// tensor manifest that leaves exactly the DiT's 6 per-block linears
/// eligible - never a bias, a norm gain, `time_proj.weight` (`k=1`), or
/// either `k=1` conv kernel (rank 3).
fn is_eligible(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && shape[1].is_multiple_of(model::int8::GROUP) && !is_never_quantized(name)
}

/// Quantize every eligible 2D weight in `tensors` via `model::int8::
/// quantize_weight` - storage format only, no kernel and no change to
/// `crate::dit::forward`'s dispatch. Returns the eligible tensors packed in
/// `int8` and everything else (never-quantized names, biases, 1D norm/scale
/// vectors, conv kernels) cloned as-is into `full`.
pub fn quantize_tensors(tensors: &[StTensor]) -> QuantizedTensors {
    let mut full = Vec::new();
    let mut int8 = HashMap::new();
    for t in tensors {
        if is_eligible(&t.name, &t.shape) {
            let (n, k) = (t.shape[0], t.shape[1]);
            let (packed, scale) = model::int8::quantize_weight(&t.data, n, k);
            int8.insert(t.name.clone(), QuantizedWeight { shape: t.shape.clone(), packed, scale });
        } else {
            full.push(StTensor { name: t.name.clone(), shape: t.shape.clone(), data: t.data.clone() });
        }
    }
    QuantizedTensors { full, int8 }
}

/// The inverse of [`quantize_tensors`]: reconstruct a plain-f32
/// `Vec<StTensor>` with every int8-eligible tensor dequantized back via
/// `model::int8::dequantize_weight`. This is what a loader hands to
/// `crate::dit::from_tensors` - that constructor dispatches no int8 kernel
/// of its own, so a quantized checkpoint must be expanded to f32 before it
/// runs.
pub fn dequantize_tensors(q: &QuantizedTensors) -> Vec<StTensor> {
    let mut out: Vec<StTensor> = q.full.iter().map(|t| StTensor { name: t.name.clone(), shape: t.shape.clone(), data: t.data.clone() }).collect();
    for (name, qw) in &q.int8 {
        let (n, k) = (qw.shape[0], qw.shape[1]);
        let data = model::int8::dequantize_weight(&qw.packed, &qw.scale, n, k);
        out.push(StTensor { name: name.clone(), shape: qw.shape.clone(), data });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_quantized_names_are_excluded_from_eligibility() {
        for name in ["proj_in.weight", "proj_out.weight", "time_embed.linear_1.weight", "time_embed.linear_2.weight"] {
            assert!(is_never_quantized(name), "{name} should be on the never-quantize list");
            assert!(!is_eligible(name, &[8, 64]), "{name} must not be int8-eligible even at a valid [n,k] shape");
        }
        // Sanity: attention's output projection must NOT collide with the
        // "proj_out" substring - it is named "to_out", not "proj_out".
        assert!(!is_never_quantized("transformer_blocks.0.attn.to_out.0.weight"));
    }

    #[test]
    fn ordinary_linears_are_eligible_biases_and_norms_are_not() {
        assert!(is_eligible("transformer_blocks.0.attn.to_q.weight", &[8, 64]));
        assert!(!is_eligible("transformer_blocks.0.norm1.bias", &[8])); // rank 1
        assert!(!is_eligible("preprocess_conv.weight", &[14, 14, 1])); // rank 3
        // `k` must be a whole number of `model::int8::GROUP`s: a matrix that
        // is otherwise perfectly ordinary is kept in fp32 when it is not.
        assert!(!is_eligible("transformer_blocks.0.attn.to_q.weight", &[8, 8]));
    }
}
