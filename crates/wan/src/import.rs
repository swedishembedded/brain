// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the Wan-VAE, in both shipped name spaces.
//!
//! The **canonical** names are the reference module tree's
//! (`encoder.downsamples.5.time_conv.weight`), because that tree is the math
//! authority and `WanVaeConfig::tensor_manifest` is derived from the same
//! schedule the graph walks. The diffusers export (`AutoencoderKLWan`) renames
//! every leaf and re-nests the decoder's flat block list two levels deep;
//! [`import_vae_diffusers`] maps it back.
//!
//! Both entry points validate in **both directions**: an expected tensor that is
//! absent errors by name, an unused source tensor errors by name, and a shape
//! mismatch errors with both shapes. Nothing is ever zero-filled - a VAE with
//! one zeroed norm gain still produces video, just wrong video, and the only
//! thing that would catch it is a parity run against a golden.
//!
//! The two files are bit-identical where they overlap (checked tensor by tensor
//! against `Wan2.1_VAE.pth`), so goldens dumped from one are valid for weights
//! imported from the other.

use checkpoint::safetensors::StTensor;
use std::collections::HashMap;
use vae::blocks::Tensors;

use crate::vae3d::WanVaeConfig;

/// Check a name→tensor map against the config's manifest in both directions.
fn validate(map: Tensors, cfg: &WanVaeConfig) -> Result<Tensors, String> {
    let manifest = cfg.tensor_manifest();
    for (name, shape) in &manifest {
        match map.get(name) {
            None => return Err(format!("wan vae import: missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("wan vae import: {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!(
                        "wan vae import: {name} has {} values, expected {n}",
                        d.len()
                    ));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> =
            manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("wan vae import: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// Import a checkpoint already in the reference module-tree names (the
/// `Wan2.1_VAE.pth` state dict, converted to safetensors).
pub fn import_vae_native(tensors: Vec<StTensor>, cfg: &WanVaeConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    validate(map, cfg)
}

/// Map one diffusers `AutoencoderKLWan` tensor name to its reference name.
///
/// Three kinds of difference, all mechanical:
///
/// * **leaf renames** - `conv_in`/`conv_out`/`norm_out` for the head convs,
///   `conv1`/`conv2`/`norm1`/`norm2`/`conv_shortcut` inside a residual block
///   (upstream addresses those by their `nn.Sequential` position:
///   `residual.{0,2,3,6}` and `shortcut`), `quant_conv`/`post_quant_conv` for
///   the two pointwise convs either side of the latent;
/// * **container renames** - `down_blocks` / `mid_block.{resnets,attentions}` /
///   `up_blocks` for `downsamples` / `middle.{0,1,2}` / `upsamples`;
/// * **re-nesting of the decoder's block list** - diffusers groups the
///   decoder's flat `upsamples.{0..14}` into `up_blocks.{i}.resnets.{r}` plus
///   `up_blocks.{i}.upsamplers.0`, at `num_res_blocks + 2` blocks per level.
///   The encoder's `down_blocks` keeps the flat indices, so only one side
///   needs the arithmetic.
fn diffusers_to_native(name: &str, cfg: &WanVaeConfig) -> Option<String> {
    // Leaf renames inside a residual block, applied after the container is
    // rewritten (both name spaces end in the same leaf grammar).
    fn leaf(rest: &str) -> String {
        for (d, n) in [
            ("norm1.", "residual.0."),
            ("conv1.", "residual.2."),
            ("norm2.", "residual.3."),
            ("conv2.", "residual.6."),
            ("conv_shortcut.", "shortcut."),
        ] {
            if let Some(tail) = rest.strip_prefix(d) {
                return format!("{n}{tail}");
            }
        }
        rest.to_string()
    }

    for (d, n) in [
        ("quant_conv.", "conv1."),
        ("post_quant_conv.", "conv2."),
        ("encoder.conv_in.", "encoder.conv1."),
        ("encoder.conv_out.", "encoder.head.2."),
        ("encoder.norm_out.", "encoder.head.0."),
        ("decoder.conv_in.", "decoder.conv1."),
        ("decoder.conv_out.", "decoder.head.2."),
        ("decoder.norm_out.", "decoder.head.0."),
    ] {
        if let Some(tail) = name.strip_prefix(d) {
            return Some(format!("{n}{tail}"));
        }
    }

    for side in ["encoder", "decoder"] {
        let mid = format!("{side}.mid_block.");
        if let Some(rest) = name.strip_prefix(&mid) {
            if let Some(r) = rest.strip_prefix("resnets.") {
                let (i, tail) = r.split_once('.')?;
                // mid resnets 0 and 1 are `middle.0` and `middle.2`; the
                // attention sits between them at `middle.1`.
                let j = match i {
                    "0" => 0,
                    "1" => 2,
                    _ => return None,
                };
                return Some(format!("{side}.middle.{j}.{}", leaf(tail)));
            }
            if let Some(r) = rest.strip_prefix("attentions.0.") {
                return Some(format!("{side}.middle.1.{r}"));
            }
            return None;
        }
    }

    if let Some(rest) = name.strip_prefix("encoder.down_blocks.") {
        let (i, tail) = rest.split_once('.')?;
        return Some(format!("encoder.downsamples.{i}.{}", leaf(tail)));
    }

    if let Some(rest) = name.strip_prefix("decoder.up_blocks.") {
        let (i, tail) = rest.split_once('.')?;
        let i: usize = i.parse().ok()?;
        let per = cfg.num_res_blocks as usize + 2;
        if let Some(r) = tail.strip_prefix("resnets.") {
            let (j, leafname) = r.split_once('.')?;
            let j: usize = j.parse().ok()?;
            return Some(format!("decoder.upsamples.{}.{}", i * per + j, leaf(leafname)));
        }
        if let Some(r) = tail.strip_prefix("upsamplers.0.") {
            return Some(format!("decoder.upsamples.{}.{}", i * per + per - 1, r));
        }
        return None;
    }

    None
}

/// Import a diffusers `AutoencoderKLWan` checkpoint.
pub fn import_vae_diffusers(tensors: Vec<StTensor>, cfg: &WanVaeConfig) -> Result<Tensors, String> {
    let mut map: Tensors = HashMap::new();
    for t in tensors {
        let native = diffusers_to_native(&t.name, cfg)
            .ok_or_else(|| format!("wan vae import: unmapped diffusers tensor {}", t.name))?;
        if map.insert(native.clone(), (t.shape, t.data)).is_some() {
            return Err(format!("wan vae import: two source tensors map to {native}"));
        }
    }
    validate(map, cfg)
}

/// Import from either name space, chosen by a name only one of them has.
pub fn import_vae(tensors: Vec<StTensor>, cfg: &WanVaeConfig) -> Result<Tensors, String> {
    let diffusers = tensors.iter().any(|t| t.name == "quant_conv.weight");
    if diffusers {
        import_vae_diffusers(tensors, cfg)
    } else {
        import_vae_native(tensors, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_names_map_onto_the_manifest() {
        let cfg = WanVaeConfig::wan21();
        for (n, want) in [
            ("encoder.conv_in.weight", "encoder.conv1.weight"),
            ("encoder.norm_out.gamma", "encoder.head.0.gamma"),
            ("encoder.conv_out.bias", "encoder.head.2.bias"),
            ("quant_conv.weight", "conv1.weight"),
            ("post_quant_conv.bias", "conv2.bias"),
            ("encoder.down_blocks.0.norm1.gamma", "encoder.downsamples.0.residual.0.gamma"),
            ("encoder.down_blocks.3.conv_shortcut.weight", "encoder.downsamples.3.shortcut.weight"),
            ("encoder.down_blocks.5.time_conv.bias", "encoder.downsamples.5.time_conv.bias"),
            ("encoder.down_blocks.8.resample.1.weight", "encoder.downsamples.8.resample.1.weight"),
            ("encoder.mid_block.resnets.1.conv2.weight", "encoder.middle.2.residual.6.weight"),
            ("encoder.mid_block.attentions.0.to_qkv.bias", "encoder.middle.1.to_qkv.bias"),
            ("decoder.up_blocks.0.resnets.2.conv1.weight", "decoder.upsamples.2.residual.2.weight"),
            ("decoder.up_blocks.0.upsamplers.0.time_conv.weight", "decoder.upsamples.3.time_conv.weight"),
            ("decoder.up_blocks.1.resnets.0.conv_shortcut.bias", "decoder.upsamples.4.shortcut.bias"),
            ("decoder.up_blocks.2.upsamplers.0.resample.1.bias", "decoder.upsamples.11.resample.1.bias"),
            ("decoder.up_blocks.3.resnets.2.conv2.weight", "decoder.upsamples.14.residual.6.weight"),
        ] {
            assert_eq!(diffusers_to_native(n, &cfg).as_deref(), Some(want), "mapping {n}");
        }
    }

    /// An unmapped name must be rejected, not silently dropped - otherwise a
    /// renamed upstream export would import as a set of missing tensors with
    /// the wrong error.
    #[test]
    fn unmapped_names_are_rejected() {
        let cfg = WanVaeConfig::wan21();
        assert_eq!(diffusers_to_native("decoder.up_blocks.0.somethingelse.weight", &cfg), None);
        assert_eq!(diffusers_to_native("encoder.mid_block.resnets.7.conv1.weight", &cfg), None);
        assert_eq!(diffusers_to_native("totally.unknown", &cfg), None);
    }

    /// Missing and unused tensors both error by name, and neither is filled in.
    #[test]
    fn validation_covers_both_directions() {
        let cfg = WanVaeConfig::wan21();
        let full: Tensors = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| {
                let len = s.iter().product();
                (n, (s, vec![0.0f32; len]))
            })
            .collect();
        assert!(validate(full.clone(), &cfg).is_ok());

        let mut missing = full.clone();
        missing.remove("decoder.upsamples.7.time_conv.weight");
        let e = validate(missing, &cfg).unwrap_err();
        assert!(e.contains("decoder.upsamples.7.time_conv.weight"), "{e}");

        let mut extra = full.clone();
        extra.insert("decoder.upsamples.11.time_conv.weight".into(), (vec![1], vec![0.0]));
        let e = validate(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");

        let mut wrong = full;
        wrong.insert("conv1.weight".into(), (vec![32, 32, 1, 1, 2], vec![0.0; 2048]));
        let e = validate(wrong, &cfg).unwrap_err();
        assert!(e.contains("conv1.weight") && e.contains("expected"), "{e}");
    }
}
