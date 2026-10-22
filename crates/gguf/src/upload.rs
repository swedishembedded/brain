// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of quantized checkpoints in the
// precision they were released in for its clients. If your team needs
// expertise in running GGUF K-quant models on the GPUs it already owns, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A GGUF tensor onto the device at the tier that holds what it stores: the
//! one upload call a GGUF-sourced model makes for a linear, so every such
//! model serves a K-quant checkpoint as K-quants without code of its own.
//!
//! * Q4_K and Q5_K stay K-quants ([`Weight::KQuant`], relaid out by
//!   [`crate::kquant::try_kq_rect`]): a 4-bit checkpoint serves at about 4.6
//!   bits a weight instead of 9.
//! * Every other format is held at int8 ([`Dtype::I8`]): a Q8_0 tensor
//!   repacks exactly, and Q6_K/Q5_0/Q4_0 fit within it.
//!
//! The tier a tensor will be held at is answered without loading
//! ([`held_tier_of`]), so a memory plan charges what the upload will make.

use checkpoint::gguf::GgmlType;
use checkpoint::packed::BoundedWorkspace;
use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use model::ops::{KqScale, Ops, Weight};

/// The resident tier holding a tensor stored as `ty` without losing what it
/// holds - see this module's doc.
pub fn held_tier(ty: GgmlType) -> Dtype {
    match ty {
        GgmlType::Q4K => Dtype::Q4K,
        GgmlType::Q5K => Dtype::Q8K,
        _ => Dtype::I8,
    }
}

/// [`held_tier`] of tensor `name` as `src` lends its blocks; [`Dtype::I8`]
/// when it lends none (a float tensor, or one a value transform is in the way of).
pub fn held_tier_of(src: &dyn TensorSource, name: &str) -> Dtype {
    src.raw_block_layout(name).map_or(Dtype::I8, |layout| held_tier(layout.ty))
}

/// Upload tensor `name` of `src` as an `[n, k]` weight of tier `want`: the
/// K-quant tiers straight from the lent blocks, every other tier through
/// [`model::ops::upload_from_source`]. `None` when the source has no such
/// tensor, or `want` is a K-quant tier the tensor is not stored in (a K-quant
/// cannot be made from other values).
pub fn upload_weight(ops: &Ops, src: &dyn TensorSource, name: &str, n: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Option<Weight> {
    if !matches!(want, Dtype::Q4K | Dtype::Q8K) {
        return model::ops::upload_from_source(ops, src, name, n, k, want, staging);
    }
    let (wq, wsm, wd, layout) = crate::kquant::try_kq_rect(src, name, k, 0, n, 0, k)?;
    if held_tier(layout.ty) != want {
        return None;
    }
    let g = ops.gpu();
    let upload = |words: &[u32]| {
        let b = g.storage(words.len() as u64);
        g.write(&b, words);
        b
    };
    let sz = KqScale::Packed { wsm: upload(&wsm), wd: upload(&wd) };
    Some(Weight::KQuant { w: upload(&wq), sz, n: n as u32, k: k as u32, group: layout.group as u32, bits: layout.bits, affine: layout.affine })
}
