// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Read facexlib's own `parsing_bisenet` checkpoint via
//! `checkpoint::safetensors`.
//!
//! The RELEASED file (`facexlib`'s GitHub release asset) is `.pth` in
//! torch's pre-1.6 LEGACY format - a bare pickle stream (`PROTO 2`, no zip
//! container) - which `checkpoint::torchpt` does not read (it expects the
//! zip-container format every OTHER `.pt`/`.pth` this workspace imports
//! uses, e.g. EVA-CLIP's). `tools/goldens/pulid_face_parsing_dump_reference.py`
//! converts it to `.safetensors` once (`torch.load` -> `safetensors.torch.
//! save_file`, dropping every `*.num_batches_tracked` scalar) - that
//! conversion is a data re-serialization, not a re-derivation, so the
//! weights this crate loads are byte-identical to the release.
//!
//! Every tensor this crate's graph reads keeps the checkpoint's OWN dotted
//! name verbatim (`cp.resnet.layer1.0.conv1.weight`, ...) - there is no
//! rename table, because [`crate::model`]'s block prefixes are written to
//! match the checkpoint, not the other way around.

use std::collections::HashMap;

pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Read `path` (`parsing_bisenet.safetensors`) into a name -> (shape, data)
/// map. Drops `conv_out16.*`/`conv_out32.*` (the two auxiliary,
/// training-only output heads - see `config.rs`'s doc on why they are not
/// part of this crate's graph at all); `*.num_batches_tracked` is already
/// absent from the conversion.
pub fn read(path: &str) -> Result<Tensors, String> {
    let tensors = checkpoint::safetensors::read(path)?;
    let mut out = Tensors::with_capacity(tensors.len());
    for t in tensors {
        if t.name.starts_with("conv_out16.") || t.name.starts_with("conv_out32.") {
            continue;
        }
        out.insert(t.name, (t.shape, t.data));
    }
    Ok(out)
}
