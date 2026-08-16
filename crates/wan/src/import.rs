// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the Wan-VAE and the Wan DiT, in both shipped name
//! spaces.
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

use crate::config::{Task, WanConfig};
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

// ---------------------------------------------------------------------------
// The diffusion transformer
// ---------------------------------------------------------------------------

/// Every tensor the DiT reads, with its shape, derived from the config.
///
/// This is the count the importer asserts against: 825 for T2V-1.3B (30 blocks
/// of 27, plus 12 embedding tensors and 3 in the head). A manifest derived from
/// the config rather than transcribed is what makes "missing" and "unused" both
/// answerable by name.
pub fn dit_manifest(cfg: &WanConfig) -> Vec<(String, Vec<usize>)> {
    let (dim, ffn, td, fd) = (cfg.dim, cfg.ffn_dim, cfg.text_dim, cfg.freq_dim);
    let (pt, ph, pw) = cfg.patch_size;
    let mut v: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |n: &str, s: Vec<usize>| v.push((n.to_string(), s));

    push("patch_embedding.weight", vec![dim, cfg.in_channels, pt, ph, pw]);
    push("patch_embedding.bias", vec![dim]);
    push("text_embedding.0.weight", vec![dim, td]);
    push("text_embedding.0.bias", vec![dim]);
    push("text_embedding.2.weight", vec![dim, dim]);
    push("text_embedding.2.bias", vec![dim]);
    push("time_embedding.0.weight", vec![dim, fd]);
    push("time_embedding.0.bias", vec![dim]);
    push("time_embedding.2.weight", vec![dim, dim]);
    push("time_embedding.2.bias", vec![dim]);
    push("time_projection.1.weight", vec![6 * dim, dim]);
    push("time_projection.1.bias", vec![6 * dim]);

    for l in 0..cfg.num_layers {
        let b = format!("blocks.{l}");
        push(&format!("{b}.modulation"), vec![1, 6, dim]);
        if cfg.cross_attn_norm {
            push(&format!("{b}.norm3.weight"), vec![dim]);
            push(&format!("{b}.norm3.bias"), vec![dim]);
        }
        for attn in ["self_attn", "cross_attn"] {
            for p in ["q", "k", "v", "o"] {
                push(&format!("{b}.{attn}.{p}.weight"), vec![dim, dim]);
                push(&format!("{b}.{attn}.{p}.bias"), vec![dim]);
            }
            if cfg.qk_norm {
                push(&format!("{b}.{attn}.norm_q.weight"), vec![dim]);
                push(&format!("{b}.{attn}.norm_k.weight"), vec![dim]);
            }
        }
        if cfg.task == Task::I2v {
            for p in ["k_img", "v_img"] {
                push(&format!("{b}.cross_attn.{p}.weight"), vec![dim, dim]);
                push(&format!("{b}.cross_attn.{p}.bias"), vec![dim]);
            }
            if cfg.qk_norm {
                push(&format!("{b}.cross_attn.norm_k_img.weight"), vec![dim]);
            }
        }
        push(&format!("{b}.ffn.0.weight"), vec![ffn, dim]);
        push(&format!("{b}.ffn.0.bias"), vec![ffn]);
        push(&format!("{b}.ffn.2.weight"), vec![dim, ffn]);
        push(&format!("{b}.ffn.2.bias"), vec![dim]);
    }

    push("head.head.weight", vec![pt * ph * pw * cfg.out_channels, dim]);
    push("head.head.bias", vec![pt * ph * pw * cfg.out_channels]);
    push("head.modulation", vec![1, 2, dim]);

    if cfg.task == Task::I2v {
        // `MLPProj`: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm over the
        // CLIP ViT-H/14 vision tower's 1280-wide tokens.
        for (n, s) in [
            ("img_emb.proj.0.weight", vec![1280]),
            ("img_emb.proj.0.bias", vec![1280]),
            ("img_emb.proj.1.weight", vec![1280, 1280]),
            ("img_emb.proj.1.bias", vec![1280]),
            ("img_emb.proj.3.weight", vec![dim, 1280]),
            ("img_emb.proj.3.bias", vec![dim]),
            ("img_emb.proj.4.weight", vec![dim]),
            ("img_emb.proj.4.bias", vec![dim]),
        ] {
            push(n, s);
        }
    }
    v
}

/// Check a name->tensor map against the DiT manifest in both directions.
fn validate_dit(map: Tensors, cfg: &WanConfig) -> Result<Tensors, String> {
    let manifest = dit_manifest(cfg);
    for (name, shape) in &manifest {
        match map.get(name) {
            None => return Err(format!("wan dit import: missing tensor {name}")),
            Some((s, d)) => {
                let n: usize = shape.iter().product();
                // A conv weight and its flattened form are the same tensor; the
                // element count is what the graph depends on, and reporting the
                // declared shape keeps a genuine width mismatch readable.
                if s.iter().product::<usize>() != n {
                    return Err(format!("wan dit import: {name} shape {s:?}, expected {shape:?}"));
                }
                if d.len() != n {
                    return Err(format!("wan dit import: {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("wan dit import: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// Leaf renames inside one diffusers block, longest-prefix first.
///
/// The dangerous row is the last pair: diffusers' `norm2` is the
/// **cross-attention** norm, which upstream calls `norm3`, and diffusers'
/// `norm3` is the FFN pre-norm, upstream's `norm2`. The two names are swapped,
/// so a mapping that passes them through imports cleanly, validates cleanly,
/// and normalises with the wrong learned affine at both sites.
const DIT_BLOCK_LEAVES: [(&str, &str); 19] = [
    ("attn1.to_q.", "self_attn.q."),
    ("attn1.to_k.", "self_attn.k."),
    ("attn1.to_v.", "self_attn.v."),
    ("attn1.to_out.0.", "self_attn.o."),
    ("attn1.norm_q.", "self_attn.norm_q."),
    ("attn1.norm_k.", "self_attn.norm_k."),
    ("attn2.to_q.", "cross_attn.q."),
    ("attn2.to_k.", "cross_attn.k."),
    ("attn2.to_v.", "cross_attn.v."),
    ("attn2.to_out.0.", "cross_attn.o."),
    ("attn2.norm_q.", "cross_attn.norm_q."),
    ("attn2.norm_k.", "cross_attn.norm_k."),
    ("attn2.add_k_proj.", "cross_attn.k_img."),
    ("attn2.add_v_proj.", "cross_attn.v_img."),
    ("attn2.norm_added_k.", "cross_attn.norm_k_img."),
    ("ffn.net.0.proj.", "ffn.0."),
    ("ffn.net.2.", "ffn.2."),
    ("norm2.", "norm3."),
    ("scale_shift_table", "modulation"),
];

const DIT_TOP_LEVEL: [(&str, &str); 12] = [
    ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
    ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
    ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
    ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
    ("condition_embedder.time_proj.", "time_projection.1."),
    ("condition_embedder.image_embedder.norm1.", "img_emb.proj.0."),
    ("condition_embedder.image_embedder.ff.net.0.proj.", "img_emb.proj.1."),
    ("condition_embedder.image_embedder.ff.net.2.", "img_emb.proj.3."),
    ("condition_embedder.image_embedder.norm2.", "img_emb.proj.4."),
    ("patch_embedding.", "patch_embedding."),
    ("proj_out.", "head.head."),
    ("scale_shift_table", "head.modulation"),
];

/// Map one diffusers `WanTransformer3DModel` tensor name to its reference name.
pub fn dit_diffusers_to_native(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("blocks.") {
        let (i, leaf) = rest.split_once('.')?;
        i.parse::<usize>().ok()?;
        for (d, n) in DIT_BLOCK_LEAVES {
            if let Some(tail) = leaf.strip_prefix(d) {
                return Some(format!("blocks.{i}.{n}{tail}"));
            }
        }
        return None;
    }
    for (d, n) in DIT_TOP_LEVEL {
        if let Some(tail) = name.strip_prefix(d) {
            return Some(format!("{n}{tail}"));
        }
    }
    None
}

/// Import a DiT checkpoint already in the reference module-tree names - the
/// canonical path, because `Wan-AI/Wan2.1-T2V-1.3B` is the registered default
/// reference and it is the repo that carries all four model roles.
pub fn import_dit_native(tensors: Vec<StTensor>, cfg: &WanConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    validate_dit(map, cfg)
}

/// Import a diffusers `WanTransformer3DModel` checkpoint.
pub fn import_dit_diffusers(tensors: Vec<StTensor>, cfg: &WanConfig) -> Result<Tensors, String> {
    let mut map: Tensors = HashMap::new();
    for t in tensors {
        let native = dit_diffusers_to_native(&t.name)
            .ok_or_else(|| format!("wan dit import: unmapped diffusers tensor {}", t.name))?;
        if map.insert(native.clone(), (t.shape, t.data)).is_some() {
            return Err(format!("wan dit import: two source tensors map to {native}"));
        }
    }
    validate_dit(map, cfg)
}

/// Import from either DiT name space, chosen by a name only one of them has.
pub fn import_dit(tensors: Vec<StTensor>, cfg: &WanConfig) -> Result<Tensors, String> {
    let diffusers = tensors.iter().any(|t| t.name == "blocks.0.scale_shift_table");
    if diffusers {
        import_dit_diffusers(tensors, cfg)
    } else {
        import_dit_native(tensors, cfg)
    }
}

/// Rename a whole reference-named map into the diffusers name space - the
/// inverse of [`dit_diffusers_to_native`], used to prove the mapping is a
/// bijection on a real checkpoint's names without a second checkpoint on disk.
pub fn dit_native_to_diffusers(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("blocks.") {
        let (i, leaf) = rest.split_once('.')?;
        i.parse::<usize>().ok()?;
        for (d, n) in DIT_BLOCK_LEAVES {
            if let Some(tail) = leaf.strip_prefix(n) {
                return Some(format!("blocks.{i}.{d}{tail}"));
            }
        }
        return None;
    }
    for (d, n) in DIT_TOP_LEVEL {
        if let Some(tail) = name.strip_prefix(n) {
            return Some(format!("{d}{tail}"));
        }
    }
    None
}

#[cfg(test)]
mod dit_tests {
    use super::*;
    use crate::config::WanConfig;

    /// 825 is the tensor count of the shipped 1.3B transformer index; the
    /// manifest is derived from the config, so this pins the derivation
    /// against a number read off the real checkpoint.
    #[test]
    fn manifest_counts_match_the_shipped_checkpoints() {
        let m = dit_manifest(&WanConfig::t2v_1_3b());
        assert_eq!(m.len(), 825, "T2V-1.3B: 30 blocks of 27 + 12 embed + 3 head");
        let m14 = dit_manifest(&WanConfig::t2v_14b());
        assert_eq!(m14.len(), 40 * 27 + 15);
        // I2V adds five per block (k_img/v_img weight+bias, norm_k_img) and
        // eight for `img_emb`.
        let i2v = dit_manifest(&WanConfig::i2v_14b_480p());
        assert_eq!(i2v.len(), 40 * 32 + 15 + 8);
    }

    #[test]
    fn diffusers_names_map_onto_the_manifest() {
        for (n, want) in [
            ("patch_embedding.weight", "patch_embedding.weight"),
            ("condition_embedder.time_proj.bias", "time_projection.1.bias"),
            ("condition_embedder.text_embedder.linear_1.weight", "text_embedding.0.weight"),
            ("proj_out.bias", "head.head.bias"),
            ("scale_shift_table", "head.modulation"),
            ("blocks.7.scale_shift_table", "blocks.7.modulation"),
            ("blocks.0.attn1.to_out.0.weight", "blocks.0.self_attn.o.weight"),
            ("blocks.29.attn2.norm_k.weight", "blocks.29.cross_attn.norm_k.weight"),
            ("blocks.3.ffn.net.0.proj.bias", "blocks.3.ffn.0.bias"),
            ("blocks.3.ffn.net.2.weight", "blocks.3.ffn.2.weight"),
            // The swap. diffusers norm2 -> upstream norm3 (cross-attn norm).
            ("blocks.1.norm2.weight", "blocks.1.norm3.weight"),
        ] {
            assert_eq!(dit_diffusers_to_native(n).as_deref(), Some(want), "mapping {n}");
        }
    }

    /// diffusers' `norm3` (the FFN pre-norm, affine-free) has NO parameters, so
    /// it must never appear in a checkpoint - and if a future export starts
    /// shipping it, mapping it through to upstream's `norm2` would be wrong in
    /// exactly the direction the swap makes hard to see. It maps to nothing.
    #[test]
    fn the_affine_free_norms_are_not_mapped() {
        assert_eq!(dit_diffusers_to_native("blocks.0.norm1.weight"), None);
        assert_eq!(dit_diffusers_to_native("blocks.0.norm3.weight"), None);
        assert_eq!(dit_diffusers_to_native("norm_out.weight"), None);
        assert_eq!(dit_diffusers_to_native("totally.unknown"), None);
        assert_eq!(dit_diffusers_to_native("blocks.x.attn1.to_q.weight"), None);
    }

    /// Every manifest name must survive a round trip through the diffusers
    /// name space - which is what makes `import_dit_diffusers` total on the
    /// real export rather than merely total on the names a test happened to
    /// list.
    #[test]
    fn the_mapping_is_a_bijection_over_the_whole_manifest() {
        for cfg in [WanConfig::t2v_1_3b(), WanConfig::i2v_14b_480p()] {
            for (name, _) in dit_manifest(&cfg) {
                let d = dit_native_to_diffusers(&name)
                    .unwrap_or_else(|| panic!("{}: no diffusers name for {name}", cfg.name));
                let back = dit_diffusers_to_native(&d)
                    .unwrap_or_else(|| panic!("{}: {d} does not map back", cfg.name));
                assert_eq!(back, name, "{}: round trip", cfg.name);
            }
        }
    }

    #[test]
    fn dit_validation_covers_both_directions() {
        let cfg = WanConfig::t2v_1_3b();
        let full: Tensors = dit_manifest(&cfg)
            .into_iter()
            .map(|(n, s)| {
                let len = s.iter().product();
                (n, (s, vec![0.0f32; len]))
            })
            .collect();
        assert!(validate_dit(full.clone(), &cfg).is_ok());

        let mut missing = full.clone();
        missing.remove("blocks.17.cross_attn.norm_k.weight");
        let e = validate_dit(missing, &cfg).unwrap_err();
        assert!(e.contains("blocks.17.cross_attn.norm_k.weight"), "{e}");

        let mut extra = full.clone();
        extra.insert("blocks.0.norm1.weight".into(), (vec![1536], vec![0.0; 1536]));
        let e = validate_dit(extra, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");

        let mut wrong = full;
        wrong.insert("head.modulation".into(), (vec![1, 6, 1536], vec![0.0; 6 * 1536]));
        let e = validate_dit(wrong, &cfg).unwrap_err();
        assert!(e.contains("head.modulation") && e.contains("expected"), "{e}");
    }
}
