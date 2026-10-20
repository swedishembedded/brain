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

use checkpoint::packed::{BoundedWorkspace, LoadConfig, LoadPolicy, QuantEncoding};
use checkpoint::TensorSource;

use super::{DeviceWeightSink, Dtype, Ops, Weight, FP8_BLOCK};

/// The destination encoding of `want`, when a tiled load can produce that tier exactly on this device.
fn encoding_of(ops: &Ops, want: Dtype) -> Option<QuantEncoding> {
    if want.promote(&ops.caps.numeric) != want {
        // The device would widen this tier: only the f32 path knows how to promote.
        return None;
    }
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

fn upload_packed(ops: &Ops, src: &dyn TensorSource, name: &str, n: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Option<Weight> {
    let packed = src.packed()?;
    let desc = packed.descriptor(name)?;
    if desc.shape != [n as u64, k as u64] {
        return None;
    }
    let enc = encoding_of(ops, want)?;
    let config = LoadConfig::default();
    // Lossless layout change first (Q8_0 to int8 blocks); an explicit numeric Convert otherwise.
    for policy in [LoadPolicy::Repack(enc.clone()), LoadPolicy::Convert(enc)] {
        let mut sink = DeviceWeightSink::new(ops);
        match packed.load_into(name, &policy, &mut sink, staging, &config) {
            Ok(Some(_)) => return Some(sink.take_weight().unwrap_or_else(|| panic!("upload {name}: a finished load produced no weight"))),
            Ok(None) => continue,
            Err(e) => panic!("upload {name}: {e}"),
        }
    }
    None
}

/// Upload tensor `name` of `src` as an `[n, k]` weight of tier `want`; `None` when the source has no such
/// tensor. `staging` bounds the host memory a packed load may hold at once.
pub fn upload_from_source(ops: &Ops, src: &dyn TensorSource, name: &str, n: usize, k: usize, want: Dtype, staging: &BoundedWorkspace) -> Option<Weight> {
    if let Some(w) = upload_packed(ops, src, name, n, k, want, staging) {
        return Some(w);
    }
    let mut built = None;
    src.with_tensor(name, &mut |raw| built = Some(Weight::upload(ops, raw, n, k, want)));
    built
}
