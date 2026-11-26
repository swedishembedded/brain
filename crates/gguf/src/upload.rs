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
    let rect = crate::kquant::try_kq_rect(src, name, k, 0, n, 0, k)?;
    (held_tier(rect.3.ty) == want).then(|| upload_rect(ops, rect))
}

/// Upload the tensors `names`, the parts of one fused projection, as one
/// `[n_total, k]` weight of tier `want`, the parts' rows in order. K-quant
/// parts are relaid out one by one and their rows joined on the host - a
/// K-quant row carries its own codes, sub-scales and super-block pair, so the
/// joined weight holds exactly what each part stores; every other tier goes
/// through [`model::ops::upload_fused_from_source`]. An error names the part
/// that is missing, not `k` wide, or stored in a format other than the one
/// `want` holds.
pub fn upload_fused_weight(ops: &Ops, src: &dyn TensorSource, names: &[&str], n_total: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Result<Weight, String> {
    if !matches!(want, Dtype::Q4K | Dtype::Q8K) {
        return model::ops::upload_fused_from_source(ops, src, names, n_total, k, want, staging);
    }
    let mut joined: Option<crate::kquant::KqRect> = None;
    for name in names {
        let numel = src.numel(name).ok_or_else(|| format!("{name}: missing from the checkpoint"))?;
        if k == 0 || !numel.is_multiple_of(k) {
            return Err(format!("{name}: {numel} elements are not rows of {k}"));
        }
        let (wq, wsm, wd, layout) = crate::kquant::try_kq_rect(src, name, k, 0, numel / k, 0, k).ok_or_else(|| format!("{name}: not stored as a K-quant of rows {k} wide"))?;
        if held_tier(layout.ty) != want {
            return Err(format!("{name}: stored as {}, which is not held at {want:?}", layout.ty.name()));
        }
        joined = Some(match joined {
            None => (wq, wsm, wd, layout),
            Some((mut jq, mut jsm, mut jd, jl)) => {
                if layout.ty != jl.ty {
                    return Err(format!("{names:?}: parts stored as {} and {} cannot be joined", jl.ty.name(), layout.ty.name()));
                }
                jq.extend(wq);
                jsm.extend(wsm);
                jd.extend(wd);
                (jq, jsm, jd, crate::kquant::KqLayout { n: jl.n + layout.n, ..jl })
            }
        });
    }
    let rect = joined.ok_or_else(|| "a fused projection needs at least one part".to_string())?;
    if rect.3.n != n_total {
        return Err(format!("{names:?}: {} rows, expected {n_total}", rect.3.n));
    }
    Ok(upload_rect(ops, rect))
}

/// A relaid-out K-quant rectangle as a device weight.
fn upload_rect(ops: &Ops, (wq, wsm, wd, layout): crate::kquant::KqRect) -> Weight {
    let g = ops.gpu();
    let upload = |words: &[u32]| {
        let b = g.storage(words.len() as u64);
        g.write(&b, words);
        b
    };
    let sz = KqScale::Packed { wsm: upload(&wsm), wd: upload(&wd) };
    Weight::KQuant { w: upload(&wq), sz, n: layout.n as u32, k: layout.k as u32, group: layout.group as u32, bits: layout.bits, affine: layout.affine }
}
