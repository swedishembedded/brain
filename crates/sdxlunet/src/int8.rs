// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 STORAGE format for the SDXL UNet's weights - a HOST-MEMORY tier, not
//! a compute-time one.
//!
//! ## Why storage, not compute
//!
//! `crates/s3dit`/`crates/flux2`/`crates/flux1` are hand-rolled DiTs: every
//! linear is their own `Rec`/`model::dispatch` dispatch, so threading a
//! `Precision` through construction (`flux1::model::Flux1Model::new_with`)
//! also swaps the GEMM kernel for the DP4A one - a genuine compute-time
//! tier. `sdxlunet::model::Unet` is composed from `vae::blocks::Builder`
//! instead (this crate's own module doc: "adds no kernel and no block"),
//! and `Builder` is shared by roughly ten other architectures (the VAE
//! family, VQGAN, RRDBNet, CodeFormer, DIAMOND). Giving `Builder` a
//! genuine int8 GEMM dispatch would mean threading activation
//! quantization scratch through every one of them for a model that runs
//! on a UNIFIED-memory box with no discrete GPU - where the actual,
//! measured problem is that the checkpoint does not FIT in host+device
//! RAM at once, not that the matmuls are slow. So this crate takes the
//! same shape `crates/ltxv`'s own `int8.rs` already established for
//! exactly this situation: weights are packed 4x smaller
//! (`model::int8::quantize_weight`, the ONE shared quantizer every
//! weight-only int8 tier in this repo already uses - `qwen3::q8`,
//! `s3dit`, `flux1`/`flux2`, `ltxv`) and `vae::blocks::Builder::dev`
//! dequantizes ONE TENSOR AT A TIME at upload (`Builder::set_packed`),
//! never the whole checkpoint. The device buffer this produces is
//! bit-for-bit what the plain fp32 path uploads - no kernel changes, no
//! new dispatch - only the HOST-resident bytes shrink.
//!
//! This is what actually closes `crates/supir`'s measured OOM (see that
//! crate's own `int8.rs`): the combined trunk+adaptors+backbone import is
//! 15.6 GB fp32 on the host, and `Supir::new`'s device-side upload while
//! that is still live climbed past this box's 30 GB. Quantized storage
//! cuts the host side to roughly a quarter (~4 GB) without touching the
//! device upload at all, and the device upload alone (~15.6 GB, unified
//! memory) fits comfortably where 15.6 + 15.6 did not.
//!
//! ## What never gets quantized, and why
//!
//! [`is_never_quantized`] excludes `time_embedding`/`add_embedding`: the
//! two linears that turn the scalar timestep and the pooled-text/time-ids
//! vector into the SAME `emb` every one of the 17 resnets in this graph
//! reads (`crate::model::Rec::conditioning`). That is structurally the
//! same role `ltxv::int8::is_never_quantized` assigns its adaLN tables -
//! "modulation/conditioning tables whose precision the whole per-token
//! combine rides on" - and, like those tables, they are a tiny fraction of
//! the model's bytes (2.1 M params here vs 2.567 B total), so excluding
//! them costs nothing and removes one whole class of risk.
//!
//! Everything else eligible is an ordinary attention/MLP projection
//! (`attn1.qkv`, `attn2.to_q`/`attn2.kv`, `ff.hidden`/`ff.gate`/`ff.out`,
//! `proj_in`/`proj_out`, each resnet's `time_emb_proj`) - the same
//! category `ltxv::int8` and `checkpoint::quantize`'s own `Policy` both
//! already treat as the ordinary case. Convolutions (`conv_in`, every
//! resnet's `conv1`/`conv2`/`conv_shortcut`, the down/up-samplers,
//! `conv_out`) and every GroupNorm/LayerNorm gain are rank != 2 and are
//! therefore never even candidates - [`is_eligible`] checks that
//! structurally, the same rule `checkpoint::quantize::Policy::decide` and
//! `ltxv::int8::is_eligible` both use ("rank 2, `k` a whole number of
//! packing groups"), not a per-name guess.

use std::collections::HashMap;

use vae::blocks::{PackedTensors, PackedWeight, Tensors};

/// True for a tensor name this port never quantizes - matched by substring
/// against the real names [`crate::config::UNetConfig::tensor_manifest`]
/// emits (`time_embedding.linear_1.weight`, `add_embedding.linear_2.bias`,
/// …). See the module doc for why these two chains and nothing else.
pub fn is_never_quantized(tensor_name: &str) -> bool {
    const NEVER_QUANTIZE_SUBSTRINGS: [&str; 2] = ["time_embedding", "add_embedding"];
    NEVER_QUANTIZE_SUBSTRINGS.iter().any(|pattern| tensor_name.contains(pattern))
}

/// A tensor is int8-storage-eligible iff it is a plain `[n, k]` matrix
/// (`k % 4 == 0`, the packing width `model::int8::quantize_weight`
/// requires) and its name is not on the never-quantize list. Every conv
/// weight in this graph is rank 4 and every norm gain/bias is rank 1, so
/// this excludes them structurally rather than by name.
fn is_eligible(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && shape[1].is_multiple_of(4) && !is_never_quantized(name)
}

/// A [`Tensors`] map split into its int8-eligible weights (packed) and
/// everything else (`full`, untouched) - [`vae::blocks::Builder::set_packed`]
/// consumes `packed`, and the caller's `Rec::new`/`Builder::new` still take
/// `full`, so the two together cover the same manifest a plain fp32 build
/// would from one map, at a fraction of the resident host bytes.
pub struct QuantizedTensors {
    pub full: Tensors,
    pub packed: PackedTensors,
}

/// Split `w` into its int8-eligible weights (quantized via
/// `model::int8::quantize_weight`) and everything else, copied as-is into
/// `full`.
pub fn quantize_tensors(w: &Tensors) -> QuantizedTensors {
    let mut full = Tensors::new();
    let mut packed: PackedTensors = HashMap::new();
    for (name, (shape, data)) in w {
        if is_eligible(name, shape) {
            let (n, k) = (shape[0], shape[1]);
            let (p, scale) = model::int8::quantize_weight(data, n, k);
            packed.insert(name.clone(), PackedWeight { shape: shape.clone(), packed: p, scale });
        } else {
            full.insert(name.clone(), (shape.clone(), data.clone()));
        }
    }
    QuantizedTensors { full, packed }
}

/// The inverse of [`quantize_tensors`]: reconstruct a plain-f32 [`Tensors`]
/// map with every int8-eligible tensor dequantized back via
/// `model::int8::dequantize_weight`. Not on the real memory-saving path
/// (that is `Builder::set_packed`'s per-tensor dequantize-at-upload) - this
/// exists for a caller that genuinely wants one flat fp32 map back, e.g. a
/// test comparing a full round trip against the original.
pub fn dequantize_tensors(q: &QuantizedTensors) -> Tensors {
    let mut out = q.full.clone();
    for (name, pw) in &q.packed {
        let (n, k) = (pw.shape[0], pw.shape[1]);
        let data = model::int8::dequantize_weight(&pw.packed, &pw.scale, n, k);
        out.insert(name.clone(), (pw.shape.clone(), data));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UNetConfig;

    #[test]
    fn never_quantized_names_are_excluded_from_eligibility() {
        for name in [
            "time_embedding.linear_1.weight",
            "time_embedding.linear_2.bias",
            "add_embedding.linear_1.weight",
            "add_embedding.linear_2.bias",
        ] {
            assert!(is_never_quantized(name), "{name} should be on the never-quantize list");
            assert!(!is_eligible(name, &[8, 64]), "{name} must not be int8-eligible even at a valid [n,k] shape");
        }
    }

    /// Pinned against the REAL manifest names [`UNetConfig::tensor_manifest`]
    /// emits, so a typo'd substring that silently excludes nothing (or the
    /// wrong thing) does not pass unnoticed.
    #[test]
    fn never_quantize_predicate_matches_the_real_tensor_manifest() {
        let cfg = UNetConfig::sdxl_base();
        let manifest = cfg.tensor_manifest();
        let names: Vec<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();

        for category in ["time_embedding", "add_embedding"] {
            assert!(names.iter().any(|n| n.contains(category) && is_never_quantized(n)), "category '{category}' matched no real tensor name");
        }

        for must_quantize in [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.qkv.weight",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.kv.weight",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.hidden.weight",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.gate.weight",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.out.weight",
            "mid_block.resnets.0.time_emb_proj.weight",
        ] {
            assert!(names.contains(&must_quantize), "sanity: '{must_quantize}' should be in tensor_manifest");
            assert!(!is_never_quantized(must_quantize), "'{must_quantize}' must stay int8-eligible");
        }

        // Every rank-1/rank-4 tensor OUTSIDE the two conditioning chains
        // (norm gains/biases, every conv) is irrelevant to eligibility
        // either way (`is_eligible` excludes them structurally), but none
        // should accidentally collide with a never-quantize substring - a
        // canary for an over-broad pattern. `time_embedding`/`add_embedding`
        // biases legitimately DO match (their whole chain is on the list,
        // weights and biases alike), so those are excluded from the canary
        // rather than treated as a false positive.
        for (name, shape) in &manifest {
            if shape.len() != 2 && !name.contains("time_embedding") && !name.contains("add_embedding") {
                assert!(!is_never_quantized(name), "unexpected never-quantize match on a non-matrix tensor: {name}");
            }
        }
    }

    #[test]
    fn ordinary_projections_are_eligible_biases_and_norms_are_not() {
        assert!(is_eligible("down_blocks.1.attentions.0.transformer_blocks.0.ff.hidden.weight", &[256, 64]));
        assert!(!is_eligible("down_blocks.1.attentions.0.transformer_blocks.0.ff.hidden.bias", &[256])); // rank 1
        assert!(!is_eligible("down_blocks.0.resnets.0.norm1.weight", &[64])); // rank 1
        assert!(!is_eligible("conv_in.weight", &[64, 4, 3, 3])); // rank 4
    }

    fn tiny_tensors() -> Tensors {
        let cfg = UNetConfig::tiny();
        crate::init::init_weights(&cfg, 7)
    }

    #[test]
    fn quantize_then_dequantize_preserves_every_eligible_tensor() {
        let w = tiny_tensors();
        let q = quantize_tensors(&w);
        assert!(!q.packed.is_empty(), "tiny config must have at least one int8-eligible tensor");

        // Every never-quantized/ineligible tensor must survive bit-for-bit.
        for (name, (shape, data)) in &w {
            if !q.packed.contains_key(name) {
                let (fshape, fdata) = q.full.get(name).unwrap_or_else(|| panic!("'{name}' missing from quantize_tensors output"));
                assert_eq!(fshape, shape);
                assert_eq!(fdata, data, "never-quantized tensor '{name}' must be untouched");
            }
        }

        let deq = dequantize_tensors(&q);
        let mut worst = (1.0f64, String::new());
        for name in q.packed.keys() {
            let (_, orig) = w.get(name).expect("original tensor");
            let (_, got) = deq.get(name).expect("dequantized tensor");
            let c = cosine(orig, got);
            assert!(c >= 0.999, "'{name}': per-tensor round-trip cosine {c:.6} too low");
            if c < worst.0 {
                worst = (c, name.clone());
            }
        }
        println!("int8 storage: {} tensors round-tripped, worst per-tensor cosine {:.9} ({})", q.packed.len(), worst.0, worst.1);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            d += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        let den = na.sqrt() * nb.sqrt();
        if den <= 0.0 {
            0.0
        } else {
            d / den
        }
    }
}
