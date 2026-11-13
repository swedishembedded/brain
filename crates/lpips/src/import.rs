// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Read the two LPIPS files into one validated tensor set, for either [`Trunk`] (AlexNet, or VGG16 with torchvision's
//! `vgg16-397923af.pth` and LPIPS v0.1's `vgg.pth`, which the store holds the same way).
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

use crate::config::{head_weight, Trunk};

/// Name -> (shape, data), exactly the tensors the metric reads: the trunk's convolutions and the five heads.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// The shapes `trunk`'s metric reads, by name.
pub fn expected_shapes(trunk: Trunk) -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::new();
    for c in trunk.convs() {
        v.push((c.weight(), vec![c.cout as usize, c.cin as usize, c.k as usize, c.k as usize]));
        v.push((c.bias(), vec![c.cout as usize]));
    }
    for (i, ch) in trunk.tap_channels().into_iter().enumerate() {
        v.push((head_weight(i), vec![1, ch as usize, 1, 1]));
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

/// Read `trunk`'s files, keep the tensors the metric reads, and
/// check every one of them is present with its expected shape and finite.
pub fn read(trunk: Trunk, trunk_file: &Path, heads: &Path) -> Result<Tensors, String> {
    let want: HashMap<String, Vec<usize>> = expected_shapes(trunk).into_iter().collect();
    let mut out = Tensors::with_capacity(want.len());
    read_into(trunk_file, &want, &mut out)?;
    read_into(heads, &want, &mut out)?;
    validate(trunk, &out)?;
    Ok(out)
}

/// A random weight set of `trunk`'s released shapes, for tests and gradient checks that must not need the model store: He-normal
/// trunk convolutions (so activations keep their scale through the ReLU layers), small biases, and heads uniform in
/// `[0, 1)` (the released heads are non-negative, which keeps every tap's distance non-negative). Not a metric: its
/// distances mean nothing, but its arithmetic is the real one.
pub fn random(trunk: Trunk, seed: u64) -> Tensors {
    let mut rng = data::rng::Rng::new(seed);
    expected_shapes(trunk)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let fan_in = shape[1..].iter().product::<usize>() as f64;
            let data: Vec<f32> = if name.ends_with(".bias") {
                (0..n).map(|_| (0.05 * rng.next_gaussian()) as f32).collect()
            } else if name.starts_with("lin") {
                (0..n).map(|_| rng.next_f64() as f32).collect()
            } else {
                (0..n).map(|_| (rng.next_gaussian() * (2.0 / fan_in).sqrt()) as f32).collect()
            };
            (name, (shape, data))
        })
        .collect()
}

/// Every tensor `trunk`'s graph reads present, shaped as it wants and finite.
pub fn validate(trunk: Trunk, t: &Tensors) -> Result<(), String> {
    for (name, shape) in expected_shapes(trunk) {
        let (have, data) = t.get(&name).ok_or_else(|| format!("lpips: no tensor {name} in the {} trunk or heads", trunk.name()))?;
        if *have != shape {
            return Err(format!("lpips: {name} is {have:?}, expected {shape:?} for the {} trunk", trunk.name()));
        }
        if let Some(bad) = data.iter().position(|v| !v.is_finite()) {
            return Err(format!("lpips: {name}[{bad}] is {}", data[bad]));
        }
    }
    Ok(())
}
