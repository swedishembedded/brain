// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of quantized
// checkpoints straight into device memory for its clients. If your team needs
// expertise in uploading packed weights tile by tile without ever expanding a
// tensor to full precision, you can procure our services by sending an email
// to info@swedishembedded.com.

//! [`upload_from_source`]: the one call a model builder makes to turn a named checkpoint tensor into
//! the resident [`Weight`] tier it asked for.
//!
//! When the source has a packed view (`checkpoint::TensorSource::packed`) the tensor is streamed tile
//! by tile through the load executor into a [`DeviceWeightSink`]: one tile of host staging leased from
//! the caller's workspace, Preserve or lossless Repack where the stored encoding is the tier, an
//! explicit Convert otherwise, never a whole-tensor f32 copy. A source without a packed view (or a tier
//! the executor cannot produce) falls back to the f32 interface, which is bounded by one tensor.

use checkpoint::packed::{BoundedWorkspace, LoadConfig, LoadPolicy, LoadRequest, QuantEncoding, TensorDescriptor};
use checkpoint::TensorSource;

use super::{DeviceWeightSink, Dtype, Ops, TierPolicy, Weight, FP8_BLOCK};

/// The encoding a resident tier holds, whatever the device: the tiers a tiled load can produce.
pub fn encoding_for(want: Dtype) -> Option<QuantEncoding> {
    Some(match want {
        Dtype::F32 => QuantEncoding::f32(),
        Dtype::BF16 => QuantEncoding::bf16(),
        Dtype::F16 => QuantEncoding::f16(),
        Dtype::I8 => QuantEncoding::int8_blocked(32),
        Dtype::NF4 => QuantEncoding::nf4_blocked(32),
        Dtype::F8E4M3 => QuantEncoding::fp8_e4m3_tiled(FP8_BLOCK),
        _ => return None,
    })
}

/// The destination encoding of `want`, when a tiled load can produce that tier exactly on this device.
fn encoding_of(ops: &Ops, want: Dtype) -> Option<QuantEncoding> {
    if want.promote(&ops.caps.numeric) != want {
        // The device would widen this tier: only the f32 path knows how to promote.
        return None;
    }
    encoding_for(want)
}

/// What a one-off conversion does to the stored tensor `desc` when asked for `tier`: `Convert` to the tier's
/// encoding when the tier has one and `desc`'s shape can take it (a blocked encoding needs a whole number of
/// blocks along the inner axis), otherwise `Preserve` (also for `None`, a tensor nobody asked to change).
pub fn conversion_policy(desc: &TensorDescriptor, tier: Option<Dtype>) -> LoadPolicy {
    let target = tier.and_then(encoding_for).filter(|enc| TensorDescriptor::new(&desc.name, desc.shape.clone(), enc.clone()).validate().is_ok());
    target.map_or(LoadPolicy::Preserve, LoadPolicy::Convert)
}

/// The resident tier that holds tensor `name` of `src` exactly as the source stores it, or `None` when the source has
/// no packed view of it or no tier is that encoding.
pub fn stored_tier(src: &dyn TensorSource, name: &str) -> Option<Dtype> {
    let stored = src.packed_descriptor(name)?.encoding;
    [Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::NF4, Dtype::F8E4M3].into_iter().find(|d| encoding_for(*d).as_ref() == Some(&stored))
}

/// The tier policy that keeps every tensor of `names` in the encoding `src` stores it in (so a bf16 or int8
/// checkpoint stays bf16 or int8 on the device and in a host-tier copy), and anything else at f32.
pub fn stored_tier_policy<'a>(src: &dyn TensorSource, names: impl IntoIterator<Item = &'a str>) -> TierPolicy {
    let mut policy = TierPolicy::uniform(Dtype::F32);
    for name in names {
        if let Some(tier) = stored_tier(src, name).filter(|t| *t != Dtype::F32) {
            policy = policy.with(&[name], tier);
        }
    }
    policy
}

fn upload_packed(ops: &Ops, src: &dyn TensorSource, names: &[&str], n: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Option<Weight> {
    let mut rows = 0u64;
    for name in names {
        let desc = src.packed_descriptor(name)?;
        if desc.shape.len() != 2 || desc.shape[1] != k as u64 {
            return None;
        }
        rows += desc.shape[0];
    }
    if rows != n as u64 {
        return None;
    }
    let enc = encoding_of(ops, want)?;
    let config = LoadConfig::default();
    // Lossless layout change first (Q8_0 to int8 blocks); an explicit numeric Convert otherwise.
    for policy in [LoadPolicy::Repack(enc.clone()), LoadPolicy::Convert(enc)] {
        let mut sink = DeviceWeightSink::new(ops);
        let req = LoadRequest { policy: &policy, workspace: staging, config: &config };
        match src.load_packed(names, &req, &mut sink) {
            Ok(Some(_)) => return Some(sink.take_weight().unwrap_or_else(|| panic!("upload {names:?}: a finished load produced no weight"))),
            Ok(None) => continue,
            Err(e) => panic!("upload {names:?}: {e}"),
        }
    }
    None
}

/// Upload tensor `name` of `src` as an `[n, k]` weight of tier `want`; `None` when the source has no such
/// tensor. `staging` bounds the host memory a packed load may hold at once.
pub fn upload_from_source(ops: &Ops, src: &dyn TensorSource, name: &str, n: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Option<Weight> {
    if let Some(w) = upload_packed(ops, src, &[name], n, k, want, staging) {
        return Some(w);
    }
    let mut built = None;
    src.with_tensor(name, &mut |raw| built = Some(Weight::upload(ops, raw, n, k, want)));
    built
}

/// Upload the row-wise concatenation of tensors `names` (a fused `q|k|v` or `gate|up` projection) as one
/// `[n_total, k]` weight of tier `want`. Parts stored in one encoding join without re-quantizing; parts stored
/// in different encodings are each converted to `want` tile by tile into their rows of the one weight (the
/// explicit `Convert` request is `want` itself), so a fused group never passes through a whole f32 tensor.
/// A source without a packed view of the parts, or whose parts are all stored as f32, concatenates them as f32.
/// Errors name the group: a part is missing, or the packed parts cannot be joined as `want` (a part's rows
/// are not whole tiles of the destination); load such a group as separate weights with
/// [`upload_group_preserved`].
pub fn upload_fused_from_source(ops: &Ops, src: &dyn TensorSource, names: &[&str], n_total: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Result<Weight, String> {
    let descs: Option<Vec<TensorDescriptor>> = names.iter().map(|n| src.packed_descriptor(n)).collect();
    if let Some(descs) = descs {
        if let Some(w) = upload_packed(ops, src, names, n_total, k, want, staging) {
            return Ok(w);
        }
        // Parts stored as f32 (a flat 1-D tensor the tiled load declines, say)
        // widen nothing on the f32 route below; a quantized part would.
        if !descs.iter().all(|d| d.encoding == QuantEncoding::f32()) {
            return Err(format!("{names:?} cannot be fused as {want:?} without an f32 round trip: load the parts as separate weights"));
        }
    }
    let mut parts: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    for name in names {
        let mut part = None;
        src.with_tensor(name, &mut |raw| part = Some(raw.to_vec()));
        parts.push(part.ok_or_else(|| format!("missing weight {name} among {names:?}"))?);
    }
    let refs: Vec<&[f32]> = parts.iter().map(Vec::as_slice).collect();
    Ok(Weight::upload(ops, &super::concat_rows_parallel(&refs), n_total, k, want))
}

/// A group of tensors that act as one fused projection, as resident weights.
pub enum FusedGroup {
    /// The parts share a stored encoding: one weight over all their rows.
    Joined(Weight),
    /// The parts are stored in different encodings: one weight per part, each in its stored encoding,
    /// in the order of the names. A fused GEMM over them is not possible; the model runs the parts separately.
    Split(Vec<Weight>),
}

/// Upload the group `names` (`ns[i]` rows each, `k` columns) in the encodings it is STORED in. Parts that
/// share an encoding become one [`FusedGroup::Joined`] weight; a group of mixed encodings (FP8 attention next to
/// bf16, NF4 `gate` next to FP8 `up`) is split into per-part weights, never widened to f32.
pub fn upload_group_preserved(ops: &Ops, src: &dyn TensorSource, names: &[&str], ns: &[usize], k: usize, staging: &BoundedWorkspace) -> Result<FusedGroup, String> {
    assert_eq!(names.len(), ns.len(), "one row count per part");
    let config = LoadConfig::default();
    let mut sink = DeviceWeightSink::new(ops);
    let req = LoadRequest { policy: &LoadPolicy::Preserve, workspace: staging, config: &config };
    if names.len() > 1 && src.load_packed(names, &req, &mut sink)?.is_some() {
        let weight = sink.take_weight().ok_or_else(|| format!("{names:?}: a finished load produced no weight"))?;
        if weight.n() as usize != ns.iter().sum::<usize>() || weight.k() as usize != k {
            return Err(format!("{names:?}: fused shape [{}, {}], expected [{}, {k}]", weight.n(), weight.k(), ns.iter().sum::<usize>()));
        }
        return Ok(FusedGroup::Joined(weight));
    }
    let parts = names.iter().zip(ns).map(|(name, n)| upload_preserved(ops, src, name, *n, k, staging)).collect::<Result<Vec<_>, _>>()?;
    Ok(if parts.len() == 1 { FusedGroup::Joined(parts.into_iter().next().expect("one part")) } else { FusedGroup::Split(parts) })
}

/// Upload tensor `name` of `src` as an `[n, k]` weight in the encoding it is STORED in - Preserve, no
/// numeric change - whichever resident tier that is: f32, bf16, f16, int8, NF4, FP8 (E4M3 or E5M2 with
/// tile scales), NVFP4, MXFP4 or MXFP8. This is how a streamed trainer keeps a checkpoint layer packed;
/// [`upload_from_source`] is the call for a tier chosen by the caller. A source with no packed view, a
/// shape that disagrees, or a stored encoding with no tier is an error: nothing is silently widened.
pub fn upload_preserved(ops: &Ops, src: &dyn TensorSource, name: &str, n: usize, k: usize, staging: &BoundedWorkspace) -> Result<Weight, String> {
    let desc = src.packed_descriptor(name).ok_or_else(|| format!("{name}: the source has no packed view of this tensor"))?;
    if desc.shape != [n as u64, k as u64] {
        return Err(format!("{name}: stored shape {:?}, expected [{n}, {k}]", desc.shape));
    }
    let config = LoadConfig::default();
    let mut sink = DeviceWeightSink::new(ops);
    let req = LoadRequest { policy: &LoadPolicy::Preserve, workspace: staging, config: &config };
    match src.load_packed(&[name], &req, &mut sink)? {
        Some(_) => sink.take_weight().ok_or_else(|| format!("{name}: a finished load produced no weight")),
        None => Err(format!("{name}: the source cannot serve this tensor packed (a value-changing binding?)")),
    }
}
