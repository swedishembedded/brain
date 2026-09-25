// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Read the two LPIPS files into one validated tensor set.
//!
//! * the trunk, torchvision's `alexnet-owt-7be5be79.pth`, is a zip-container
//!   `torch.save` that `checkpoint::torchpt` reads as released (a
//!   `.safetensors` copy of it reads just the same). Only the five
//!   `features.*` convolutions are kept; the classifier is not part of LPIPS.
//! * the heads, LPIPS v0.1's `alex.pth`, arrive in torch's legacy pickle
//!   format, which `checkpoint::torchpt` does not read, so the store holds
//!   them re-serialized as `alex.safetensors`
//!   (`tools/goldens/lpips_dump_reference.py`).
//!
//! Every tensor keeps its checkpoint name, which is also the name the graph
//! reads it under ([`crate::config::trunk_weight`] and friends), and every
//! shape is checked against [`crate::config::TRUNK`] here, once, so a wrong
//! file is refused by name rather than convolved.

use std::collections::HashMap;
use std::path::Path;

use crate::config::{head_weight, trunk_bias, trunk_weight, TRUNK};

/// Name -> (shape, data), exactly the fifteen tensors the metric reads.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// The shapes the metric reads, by name.
pub fn expected_shapes() -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::with_capacity(3 * TRUNK.len());
    for (i, c) in TRUNK.iter().enumerate() {
        v.push((trunk_weight(i), vec![c.cout as usize, c.cin as usize, c.k as usize, c.k as usize]));
        v.push((trunk_bias(i), vec![c.cout as usize]));
        v.push((head_weight(i), vec![1, c.cout as usize, 1, 1]));
    }
    v
}

/// Add the tensors of `path` the metric reads (named in `want`) to `out`.
fn read_into(path: &Path, want: &HashMap<String, Vec<usize>>, out: &mut Tensors) -> Result<(), String> {
    let p = path.to_str().ok_or_else(|| format!("lpips: {} is not a UTF-8 path", path.display()))?;
    let context = |e: String| format!("lpips: reading {}: {e}", path.display());
    let all: Vec<(String, Vec<usize>, Vec<f32>)> = if p.ends_with(".safetensors") {
        checkpoint::safetensors::read(p).map_err(context)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    } else {
        checkpoint::torchpt::read(p).map_err(context)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    };
    out.extend(all.into_iter().filter(|(name, _, _)| want.contains_key(name)).map(|(name, shape, data)| (name, (shape, data))));
    Ok(())
}

/// Read the trunk and the heads, keep the tensors the metric reads, and
/// check every one of them is present with its expected shape and finite.
pub fn read(trunk: &Path, heads: &Path) -> Result<Tensors, String> {
    let want: HashMap<String, Vec<usize>> = expected_shapes().into_iter().collect();
    let mut out = Tensors::with_capacity(want.len());
    read_into(trunk, &want, &mut out)?;
    read_into(heads, &want, &mut out)?;
    validate(&out)?;
    Ok(out)
}

/// Every expected tensor present, shaped as the graph wants and finite.
pub fn validate(t: &Tensors) -> Result<(), String> {
    for (name, shape) in expected_shapes() {
        let (have, data) = t.get(&name).ok_or_else(|| format!("lpips: no tensor {name} in the trunk or the heads"))?;
        if *have != shape {
            return Err(format!("lpips: {name} is {have:?}, expected {shape:?}"));
        }
        if let Some(bad) = data.iter().position(|v| !v.is_finite()) {
            return Err(format!("lpips: {name}[{bad}] is {}", data[bad]));
        }
    }
    Ok(())
}
