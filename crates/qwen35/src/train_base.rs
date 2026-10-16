// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a [`crate::model::Qwen35`] training build holds its frozen base in,
//! and the small tier predicates the trainer's linears dispatch on.
//!
//! A LoRA trainer never writes the base weights, so the base can live in any
//! tier a matmul and a `dX` matmul can read with floating-point activations:
//! fp32, bf16 (half the bytes) or - once `model::ops` provides the tier - NF4.
//! Adapters, norms, `A_log`, `dt_bias` and the causal-conv taps stay fp32.

use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::{TierPolicy, Weight};

use crate::config::Qwen35Config;

/// How to build a trainer. The default is the classic one: an fp32 base and
/// every layer's activations kept for the backward pass.
#[derive(Clone, Debug)]
pub struct TrainOpts {
    /// Storage tier of the per-layer mixer/MLP linears. Anything other than
    /// all-fp32 requires `cfg.lora` (a full-parameter build trains the base,
    /// which only fp32 can hold).
    pub base: TierPolicy,
    /// Keep only the layer-boundary residuals through the forward and rebuild
    /// each layer's activations inside the backward (one layer's activations
    /// alive at a time instead of all of them, for one extra forward).
    pub recompute: bool,
    /// Words of logits the chunked LM head keeps alive at once (it computes
    /// `rows x vocab` at a time instead of `tokens x vocab`); `None` takes
    /// the default of 2^28 words (1 GiB).
    pub head_budget_words: Option<u64>,
}

impl Default for TrainOpts {
    fn default() -> Self {
        TrainOpts { base: TierPolicy::uniform(Dtype::F32), recompute: false, head_budget_words: None }
    }
}

/// The tier a narrow-base LoRA build holds its frozen embedding table and LM
/// head at, or `None` when `name` stays an fp32 store tensor. Both are plain
/// `[vocab, d_model]` matrices the adapters never touch, and at the real
/// shape each is 5 GB in fp32 - more than one storage binding. bf16 halves
/// that and is the floor: unlike the layer linears they are not held at 4
/// bits, so a narrower request for the base gives them bf16. MTP shares the
/// tables with its own fp32 path and keeps them fp32.
pub(crate) fn table_tier(tier: &TierPolicy, cfg: &Qwen35Config, train: bool, name: &str) -> Option<Dtype> {
    let is_table = name == "tok.weight" || name == cfg.head_weight();
    (train && cfg.lora.is_some() && !cfg.mtp && tier.quantizes_anything() && is_table).then_some(Dtype::BF16)
}

/// `true` when `w` is read with fp32 (unquantized) activations: the tiers
/// whose matmul decodes the WEIGHT on the fly, so a LoRA correction computed
/// from the same fp32 activation is exact and the int8-packed activation view
/// would be dead work. The int8/int4/K-quant tiers read packed activations
/// and are inference-only.
pub(crate) fn reads_float_acts(w: &Weight) -> bool {
    !matches!(w, Weight::I8 { .. } | Weight::Q4 { .. } | Weight::KQuant { .. })
}

/// Host values of a weight held in the `Ops` weight map, for read-back
/// (`Qwen35::read_weight`). Only the float-read tiers have a lossless or
/// documented decode here.
pub(crate) fn read_weight_host(gpu: &Gpu, w: &Weight) -> Vec<f32> {
    let (n, k) = (w.n() as usize, w.k() as usize);
    match w {
        Weight::F32 { w, .. } => gpu.read(w, n * k),
        Weight::BF16 { w, .. } => unpack_bf16(gpu, w, n * k),
        Weight::NF4 { w, s, .. } => {
            let packed: Vec<u32> = gpu.read(w, n * k / 8).into_iter().map(f32::to_bits).collect();
            model::lut4::dequantize_weight_lut4(&packed, &gpu.read(s, n * k / 32), n, k, &model::lut4::NF4_LUT)
        }
        other => panic!("qwen35: reading back a {:?} base weight is not supported", other.dtype()),
    }
}

/// Inverse of `model::half::pack_bf16` over `len` elements: element `2i` in
/// the low half-word, `2i+1` in the high one.
fn unpack_bf16(gpu: &Gpu, packed: &DeviceBuffer, len: usize) -> Vec<f32> {
    let words = gpu.read(packed, len.div_ceil(2));
    let mut out: Vec<f32> = words.iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xffff_0000)]).collect();
    out.truncate(len);
    out
}

/// Overwrite a weight held in the `Ops` weight map with `data` (row-major
/// `[n, k]`, fp32 host values), re-encoding to the weight's own tier.
pub(crate) fn write_weight_host(gpu: &Gpu, w: &Weight, data: &[f32]) {
    assert_eq!(data.len(), w.n() as usize * w.k() as usize, "qwen35: write_weight length does not match the weight's [n, k]");
    match w {
        Weight::F32 { w, .. } => gpu.write(w, bytemuck::cast_slice(data)),
        Weight::BF16 { w, .. } => gpu.write(w, bytemuck::cast_slice(&model::half::pack_bf16(data))),
        other => panic!("qwen35: writing a {:?} base weight is not supported", other.dtype()),
    }
}
