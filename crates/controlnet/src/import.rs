// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a diffusers `ControlNetModel` checkpoint, with **two-way** coverage
//! validation.
//!
//! The remap loop itself is **`unet::import::remap_manifest`** — a ControlNet
//! checkpoint is a UNet-family checkpoint carrying the same three fused leaves
//! (`attn1`'s q/k/v, `attn2`'s k/v, the GEGLU `ff.net.0.proj`) under the same
//! module names, and a second copy here would be a second place those fusions
//! can be got wrong with nothing comparing the two. All this module owns is
//! finding the file and naming the model in the errors.

use std::collections::HashMap;

use crate::config::ControlNetConfig;

/// Host tensors by brain-side name: `(shape, row-major f32 data)`.
pub type Tensors = unet::import::Tensors;

/// Read `<dir>/diffusion_pytorch_model*.safetensors` (or the exact file when
/// `path` names one) and remap it onto `cfg`'s manifest.
pub fn load(path: &str, cfg: &ControlNetConfig) -> Result<Tensors, String> {
    cfg.validate()?;
    let p = std::path::Path::new(path);
    let file = if p.is_dir() {
        let mut cands: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("controlnet import: {path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|f| f.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        cands.sort();
        cands
            .iter()
            .find(|f| f.to_string_lossy().contains("fp16"))
            .or_else(|| cands.first())
            .ok_or_else(|| format!("controlnet import: no .safetensors in {path}"))?
            .clone()
    } else {
        p.to_path_buf()
    };
    let src = checkpoint::safetensors::read(file.to_str().ok_or("controlnet import: non-utf8 path")?)?;
    let raw: HashMap<String, (Vec<usize>, Vec<f32>)> =
        src.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    remap(raw, cfg)
}

/// The pure remap, so a synthetic checkpoint (tests, [`crate::init`]) exercises
/// exactly the code the real one does.
pub fn remap(
    raw: HashMap<String, (Vec<usize>, Vec<f32>)>,
    cfg: &ControlNetConfig,
) -> Result<Tensors, String> {
    unet::import::remap_manifest("controlnet", raw, &cfg.tensor_manifest())
}
