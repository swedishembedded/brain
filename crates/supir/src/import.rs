// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the real SUPIR checkpoint, with **two-way** coverage validation.
//!
//! ## The trunk is CompVis/LDM-named, not diffusers-named
//! `model.diffusion_model.project_modules.*` and
//! `first_stage_model.denoise_encoder.*` use names that already match (or
//! trivially rename to) this crate's own manifests. `model.control_model.*`
//! does not: SUPIR ships its own from-scratch framework (`sgm/`, CompVis
//! lineage), so the trunk's OUTER structure is `input_blocks.{i}.{0,1}` /
//! `middle_block.{0,1,2}` / `time_embed` / `label_emb`, not diffusers'
//! `down_blocks`/`mid_block`/`time_embedding`/`add_embedding` - verified
//! directly against the real checkpoint header (every `input_blocks.*`/
//! `middle_block.*` key enumerated and cross-checked against SDXL's own
//! `layers_per_block`/`transformer_layers_per_block` schedule).
//!
//! The INNER transformer leaf names (`attn1.to_q`, `ff.net.0.proj`, `norm1`,
//! …) already match diffusers' own convention byte-for-byte - SUPIR's
//! `SpatialTransformer` reuses that naming. So [`remap_trunk`] does exactly
//! two things: (1) rename the OUTER LDM structure into
//! [`sdxlunet::config::UNetConfig::tensor_manifest`]'s own diffusers-style
//! names (a pure rename, no fusion - [`ldm_input_blocks`]'s job), then (2)
//! hand the renamed map to **`sdxlunet::import::remap_manifest`** for the
//! qkv/kv/GEGLU fusions - the SAME function `crates/controlnet` already
//! reuses for exactly this reason: a second copy of that fusion logic is a
//! second place it can drift with nothing to compare it against.
//!
//! ## `model.control_model.mask_LQ` is rejected, named, not dropped
//! A `[1,4,64,64]` leftover from an unreleased masking variant with no
//! counterpart in the released `GLVControl` code - present in the real
//! checkpoint but never read by anything the released repository ships
//! (upstream's own loader drops it with `strict=False`; the Python
//! reference dumper for this port's parity goldens asserts on its presence
//! the same way, rather than letting it vanish unnoticed). [`remap`] checks
//! for it BEFORE the generic two-way coverage pass and errors naming it
//! explicitly.

use std::collections::HashMap;

use sdxlunet::config::{BlockKind, UNetConfig};
use vae::config::VaeConfig;

use crate::adaptors::AdaptorConfig;
use crate::config::{denoise_encoder_manifest, SupirConfig, HINT_EMBEDDER};

/// Host tensors by brain-side name: `(shape, row-major f32 data)`.
pub type Tensors = sdxlunet::import::Tensors;

const CM_PREFIX: &str = "model.control_model.";
/// Strips only as far as `model.diffusion_model.` (not `...project_modules.`
/// too), because [`AdaptorConfig::tensor_manifest`]'s own brain-side names
/// already keep the `project_modules.` segment (matching the checkpoint's
/// own naming minus this prefix, per that function's doc) - stripping it
/// here too would leave every lookup below off by one path segment. The
/// real checkpoint's `model.diffusion_model.*` half carries nothing besides
/// `project_modules.*` (verified against its header: 118 tensors, all of
/// them), so this extracts exactly the adaptor half and nothing else; the
/// two-way coverage check below still catches it by name if that ever
/// changes.
const PM_PREFIX: &str = "model.diffusion_model.";
const DE_PREFIX: &str = "first_stage_model.denoise_encoder.";
const MASK_LQ: &str = "model.control_model.mask_LQ";

/// Read the SUPIR delta checkpoint (`SUPIR-v0Q_fp32.safetensors` or
/// equivalent) and remap it onto `cfg`'s manifest. The frozen SDXL backbone
/// is a SEPARATE file, loaded with `sdxlunet::import::load` against
/// `cfg.backbone` - this function only ever produces SUPIR's own 1035-tensor
/// delta.
pub fn load(path: &str, cfg: &SupirConfig) -> Result<Tensors, String> {
    let src = checkpoint::safetensors::read(path)?;
    let raw: HashMap<String, (Vec<usize>, Vec<f32>)> = src.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    remap(raw, cfg)
}

/// The pure remap, so a synthetic checkpoint (tests) exercises exactly the
/// code the real one does.
pub fn remap(mut raw: HashMap<String, (Vec<usize>, Vec<f32>)>, cfg: &SupirConfig) -> Result<Tensors, String> {
    if raw.remove(MASK_LQ).is_some() {
        return Err(format!(
            "supir import: {MASK_LQ} is a leftover from an unreleased masking variant with no \
             counterpart in the released GLVControl code - rejecting it explicitly rather than \
             silently dropping it"
        ));
    }

    let trunk_local = extract_prefixed(&mut raw, CM_PREFIX);
    let adaptor_local = extract_prefixed(&mut raw, PM_PREFIX);
    let denc_local = extract_prefixed(&mut raw, DE_PREFIX);

    if !raw.is_empty() {
        let mut extra: Vec<&String> = raw.keys().collect();
        extra.sort();
        return Err(format!(
            "supir import: {} source tensors are not under {CM_PREFIX}/{PM_PREFIX}/{DE_PREFIX}, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(8)]
        ));
    }

    let mut out = remap_trunk(trunk_local, &cfg.trunk)?;
    out.extend(remap_adaptors(adaptor_local, &cfg.adaptors)?);
    out.extend(remap_denoise_encoder(denc_local, &cfg.denoise_encoder)?);

    let want = cfg.tensor_manifest().len();
    if out.len() != want {
        return Err(format!("supir import: produced {} of {want} tensors", out.len()));
    }
    Ok(out)
}

/// Move every `{prefix}{rest}` key out of `raw` into a fresh map keyed by
/// `{rest}` alone.
fn extract_prefixed(raw: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let keys: Vec<String> = raw.keys().filter(|k| k.starts_with(prefix)).cloned().collect();
    let mut out = HashMap::with_capacity(keys.len());
    for k in keys {
        let v = raw.remove(&k).expect("just listed");
        out.insert(k[prefix.len()..].to_string(), v);
    }
    out
}

/// Exact shape equality - see `sdxlunet::import::check_shape`'s doc for why
/// element-count alone is the wrong, weaker check.
fn check_shape(name: &str, got: &[usize], want: &[usize]) -> Result<(), String> {
    if got != want {
        return Err(format!("supir import: {name} shape {got:?}, expected {want:?}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The trunk: LDM outer structure -> diffusers-style names, then the shared
// qkv/kv/GEGLU fusion.
// ---------------------------------------------------------------------------

/// One entry of `GLVControl`'s `input_blocks` (LDM numbering), derived from
/// `trunk`'s own `levels()`/`layers_per_block` - the SAME walk
/// `crate::adaptors::AdaptorConfig::for_backbone` and `Rec::down_path` use,
/// so a config where the two disagree cannot exist.
enum LdmBlock {
    ConvIn,
    Resnet { level: usize, layer: usize },
    Downsample { level: usize },
}

fn ldm_input_blocks(trunk: &UNetConfig) -> Vec<LdmBlock> {
    let mut v = vec![LdmBlock::ConvIn];
    for level in 0..trunk.levels() {
        for layer in 0..trunk.layers_per_block as usize {
            v.push(LdmBlock::Resnet { level, layer });
        }
        if level + 1 < trunk.levels() {
            v.push(LdmBlock::Downsample { level });
        }
    }
    v
}

/// Rename one `ResBlock`'s leaves: `in_layers.0/2` -> `norm1`/`conv1`,
/// `emb_layers.1` -> `time_emb_proj`, `out_layers.0/3` -> `norm2`/`conv2`,
/// `skip_connection` -> `conv_shortcut` (present only when the block changes
/// channel width - optional, unlike the other five).
fn rename_resnet(
    local: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
    renamed: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
    src: &str,
    dst: &str,
) -> Result<(), String> {
    for (s, d) in [
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("emb_layers.1", "time_emb_proj"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
    ] {
        for suf in ["weight", "bias"] {
            let sk = format!("{src}.{s}.{suf}");
            let v = local.remove(&sk).ok_or_else(|| format!("supir trunk import: missing {CM_PREFIX}{sk}"))?;
            renamed.insert(format!("{dst}.{d}.{suf}"), v);
        }
    }
    for suf in ["weight", "bias"] {
        let sk = format!("{src}.skip_connection.{suf}");
        if let Some(v) = local.remove(&sk) {
            renamed.insert(format!("{dst}.conv_shortcut.{suf}"), v);
        }
    }
    Ok(())
}

/// Move every leaf under `{src}.` to `{dst}.` unchanged - the transformer
/// sub-block, whose inner names already match diffusers (see the module
/// doc). Missing/extra leaves surface later, at
/// `sdxlunet::import::remap_manifest`'s own two-way check on the renamed
/// map, so nothing here needs to validate completeness itself.
fn rename_prefix_passthrough(local: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, renamed: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, src: &str, dst: &str) {
    let full = format!("{src}.");
    let keys: Vec<String> = local.keys().filter(|k| k.starts_with(&full)).cloned().collect();
    for k in keys {
        let v = local.remove(&k).expect("just listed");
        let suffix = &k[full.len()..];
        renamed.insert(format!("{dst}.{suffix}"), v);
    }
}

fn remap_trunk(mut local: HashMap<String, (Vec<usize>, Vec<f32>)>, trunk: &UNetConfig) -> Result<Tensors, String> {
    let mut renamed: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

    // The hint embedder is NOT part of `sdxlunet::config::UNetConfig::tensor_manifest`
    // (it has no counterpart in a vanilla UNet), so it never goes through
    // `sdxlunet::import::remap_manifest`'s fusion pass below - collected
    // separately and reattached at the end.
    let mut hint: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for suf in ["weight", "bias"] {
        let sk = format!("input_hint_block.0.{suf}");
        let v = local.remove(&sk).ok_or_else(|| format!("supir trunk import: missing {CM_PREFIX}{sk}"))?;
        hint.insert(format!("control_model.{HINT_EMBEDDER}.{suf}"), v);
    }

    for (src, dst) in [
        ("time_embed.0", "time_embedding.linear_1"),
        ("time_embed.2", "time_embedding.linear_2"),
        ("label_emb.0.0", "add_embedding.linear_1"),
        ("label_emb.0.2", "add_embedding.linear_2"),
    ] {
        for suf in ["weight", "bias"] {
            let sk = format!("{src}.{suf}");
            let v = local.remove(&sk).ok_or_else(|| format!("supir trunk import: missing {CM_PREFIX}{sk}"))?;
            renamed.insert(format!("{dst}.{suf}"), v);
        }
    }

    for (bi, blk) in ldm_input_blocks(trunk).into_iter().enumerate() {
        match blk {
            LdmBlock::ConvIn => {
                for suf in ["weight", "bias"] {
                    let sk = format!("input_blocks.{bi}.0.{suf}");
                    let v = local.remove(&sk).ok_or_else(|| format!("supir trunk import: missing {CM_PREFIX}{sk}"))?;
                    renamed.insert(format!("conv_in.{suf}"), v);
                }
            }
            LdmBlock::Resnet { level, layer } => {
                rename_resnet(&mut local, &mut renamed, &format!("input_blocks.{bi}.0"), &format!("down_blocks.{level}.resnets.{layer}"))?;
                if trunk.down_block_types[level] == BlockKind::CrossAttn {
                    rename_prefix_passthrough(&mut local, &mut renamed, &format!("input_blocks.{bi}.1"), &format!("down_blocks.{level}.attentions.{layer}"));
                }
            }
            LdmBlock::Downsample { level } => {
                for suf in ["weight", "bias"] {
                    let sk = format!("input_blocks.{bi}.0.op.{suf}");
                    let v = local.remove(&sk).ok_or_else(|| format!("supir trunk import: missing {CM_PREFIX}{sk}"))?;
                    renamed.insert(format!("down_blocks.{level}.downsamplers.0.conv.{suf}"), v);
                }
            }
        }
    }

    rename_resnet(&mut local, &mut renamed, "middle_block.0", "mid_block.resnets.0")?;
    rename_prefix_passthrough(&mut local, &mut renamed, "middle_block.1", "mid_block.attentions.0");
    rename_resnet(&mut local, &mut renamed, "middle_block.2", "mid_block.resnets.1")?;

    if !local.is_empty() {
        let mut extra: Vec<&String> = local.keys().collect();
        extra.sort();
        return Err(format!(
            "supir trunk import: {} unexpected {CM_PREFIX} tensors, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(8)]
        ));
    }

    let manifest = crate::config::trunk_manifest_unprefixed(trunk);
    let fused = sdxlunet::import::remap_manifest("supir trunk", renamed, &manifest)?;
    let mut out: Tensors = fused.into_iter().map(|(k, v)| (format!("control_model.{k}"), v)).collect();
    out.extend(hint);
    Ok(out)
}

// ---------------------------------------------------------------------------
// The 12 adaptors: a rename-free 1:1 map, except `ZeroCrossAttn`'s
// `to_k`/`to_v` -> one fused `kv`.
// ---------------------------------------------------------------------------

fn remap_adaptors(mut local: HashMap<String, (Vec<usize>, Vec<f32>)>, acfg: &AdaptorConfig) -> Result<Tensors, String> {
    let manifest = acfg.tensor_manifest();
    let mut out: Tensors = HashMap::new();
    for (name, shape) in &manifest {
        if let Some(base) = name.strip_suffix(".attn.kv.weight") {
            let mut data = Vec::with_capacity(shape.iter().product());
            let mut rows = 0usize;
            for leaf in ["to_k", "to_v"] {
                let src = format!("{base}.attn.{leaf}.weight");
                let (s, d) = local.remove(&src).ok_or_else(|| format!("supir adaptors import: missing {PM_PREFIX}{src}"))?;
                check_shape(&src, &s, &[shape[0] / 2, shape[1]])?;
                rows += s[0];
                data.extend_from_slice(&d);
            }
            check_shape(name, &[rows, shape[1]], shape)?;
            out.insert(name.clone(), (shape.clone(), data));
        } else {
            let (s, d) = local.remove(name).ok_or_else(|| format!("supir adaptors import: missing {PM_PREFIX}{name}"))?;
            check_shape(name, &s, shape)?;
            out.insert(name.clone(), (s, d));
        }
    }
    if !local.is_empty() {
        let mut extra: Vec<&String> = local.keys().collect();
        extra.sort();
        return Err(format!(
            "supir adaptors import: {} unexpected {PM_PREFIX} tensors, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(8)]
        ));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The denoise encoder: a plain 1:1 map (CompVis-named, no rename needed -
// see `crate::config::denoise_encoder_manifest`'s doc for why).
// ---------------------------------------------------------------------------

fn remap_denoise_encoder(mut local: HashMap<String, (Vec<usize>, Vec<f32>)>, vcfg: &VaeConfig) -> Result<Tensors, String> {
    let manifest = denoise_encoder_manifest(vcfg);
    let mut out: Tensors = HashMap::new();
    for (name, shape) in &manifest {
        let key = name.strip_prefix("denoise_encoder.").expect("denoise_encoder_manifest always prefixes this");
        let (s, d) = local.remove(key).ok_or_else(|| format!("supir denoise_encoder import: missing {DE_PREFIX}{key}"))?;
        check_shape(name, &s, shape)?;
        out.insert(name.clone(), (s, d));
    }
    if !local.is_empty() {
        let mut extra: Vec<&String> = local.keys().collect();
        extra.sort();
        return Err(format!(
            "supir denoise_encoder import: {} unexpected {DE_PREFIX} tensors, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(8)]
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic checkpoint under the REAL LDM/CompVis key layout,
    /// round-tripped through [`remap`] - the tiny-config coverage rung
    /// (porting.md §5 rung 1), and the only place this crate exercises the
    /// LDM->diffusers rename without a multi-GB real file.
    fn synthetic_checkpoint(cfg: &SupirConfig) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
        let mut raw: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        let mut put = |name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            raw.insert(name, (shape, vec![0.5f32; n]));
        };

        // ---- trunk, LDM-named ----
        let c0 = cfg.trunk.block_out_channels[0] as usize;
        put(format!("{CM_PREFIX}time_embed.0.weight"), vec![cfg.trunk.time_embed_dim as usize, c0]);
        put(format!("{CM_PREFIX}time_embed.0.bias"), vec![cfg.trunk.time_embed_dim as usize]);
        put(format!("{CM_PREFIX}time_embed.2.weight"), vec![cfg.trunk.time_embed_dim as usize, cfg.trunk.time_embed_dim as usize]);
        put(format!("{CM_PREFIX}time_embed.2.bias"), vec![cfg.trunk.time_embed_dim as usize]);
        put(format!("{CM_PREFIX}label_emb.0.0.weight"), vec![cfg.trunk.time_embed_dim as usize, cfg.trunk.projection_class_embeddings_input_dim as usize]);
        put(format!("{CM_PREFIX}label_emb.0.0.bias"), vec![cfg.trunk.time_embed_dim as usize]);
        put(format!("{CM_PREFIX}label_emb.0.2.weight"), vec![cfg.trunk.time_embed_dim as usize, cfg.trunk.time_embed_dim as usize]);
        put(format!("{CM_PREFIX}label_emb.0.2.bias"), vec![cfg.trunk.time_embed_dim as usize]);
        put(format!("{CM_PREFIX}input_hint_block.0.weight"), vec![c0, 4, 3, 3]);
        put(format!("{CM_PREFIX}input_hint_block.0.bias"), vec![c0]);

        let mut prev = c0;
        for (bi, blk) in ldm_input_blocks(&cfg.trunk).into_iter().enumerate() {
            match blk {
                LdmBlock::ConvIn => {
                    put(format!("{CM_PREFIX}input_blocks.{bi}.0.weight"), vec![c0, cfg.trunk.in_channels as usize, 3, 3]);
                    put(format!("{CM_PREFIX}input_blocks.{bi}.0.bias"), vec![c0]);
                }
                LdmBlock::Resnet { level, layer } => {
                    let cout = cfg.trunk.block_out_channels[level] as usize;
                    let cin = if layer == 0 { prev } else { cout };
                    let p = format!("{CM_PREFIX}input_blocks.{bi}.0");
                    put(format!("{p}.in_layers.0.weight"), vec![cin]);
                    put(format!("{p}.in_layers.0.bias"), vec![cin]);
                    put(format!("{p}.in_layers.2.weight"), vec![cout, cin, 3, 3]);
                    put(format!("{p}.in_layers.2.bias"), vec![cout]);
                    put(format!("{p}.emb_layers.1.weight"), vec![cout, cfg.trunk.time_embed_dim as usize]);
                    put(format!("{p}.emb_layers.1.bias"), vec![cout]);
                    put(format!("{p}.out_layers.0.weight"), vec![cout]);
                    put(format!("{p}.out_layers.0.bias"), vec![cout]);
                    put(format!("{p}.out_layers.3.weight"), vec![cout, cout, 3, 3]);
                    put(format!("{p}.out_layers.3.bias"), vec![cout]);
                    if cin != cout {
                        put(format!("{p}.skip_connection.weight"), vec![cout, cin, 1, 1]);
                        put(format!("{p}.skip_connection.bias"), vec![cout]);
                    }
                    if cfg.trunk.down_block_types[level] == BlockKind::CrossAttn {
                        let x = cfg.trunk.cross_attention_dim as usize;
                        let inner = 2 * (4 * cout);
                        let tp = format!("{CM_PREFIX}input_blocks.{bi}.1");
                        put(format!("{tp}.norm.weight"), vec![cout]);
                        put(format!("{tp}.norm.bias"), vec![cout]);
                        put(format!("{tp}.proj_in.weight"), vec![cout, cout]);
                        put(format!("{tp}.proj_in.bias"), vec![cout]);
                        for k in 0..cfg.trunk.transformer_layers_per_block[level] {
                            let b = format!("{tp}.transformer_blocks.{k}");
                            for nm in ["norm1", "norm2", "norm3"] {
                                put(format!("{b}.{nm}.weight"), vec![cout]);
                                put(format!("{b}.{nm}.bias"), vec![cout]);
                            }
                            for nm in ["to_q", "to_k", "to_v"] {
                                put(format!("{b}.attn1.{nm}.weight"), vec![cout, cout]);
                            }
                            put(format!("{b}.attn1.to_out.0.weight"), vec![cout, cout]);
                            put(format!("{b}.attn1.to_out.0.bias"), vec![cout]);
                            put(format!("{b}.attn2.to_q.weight"), vec![cout, cout]);
                            put(format!("{b}.attn2.to_k.weight"), vec![cout, x]);
                            put(format!("{b}.attn2.to_v.weight"), vec![cout, x]);
                            put(format!("{b}.attn2.to_out.0.weight"), vec![cout, cout]);
                            put(format!("{b}.attn2.to_out.0.bias"), vec![cout]);
                            put(format!("{b}.ff.net.0.proj.weight"), vec![inner, cout]);
                            put(format!("{b}.ff.net.0.proj.bias"), vec![inner]);
                            put(format!("{b}.ff.net.2.weight"), vec![cout, inner / 2]);
                            put(format!("{b}.ff.net.2.bias"), vec![cout]);
                        }
                        put(format!("{tp}.proj_out.weight"), vec![cout, cout]);
                        put(format!("{tp}.proj_out.bias"), vec![cout]);
                    }
                    prev = cout;
                }
                LdmBlock::Downsample { level } => {
                    let c = cfg.trunk.block_out_channels[level] as usize;
                    put(format!("{CM_PREFIX}input_blocks.{bi}.0.op.weight"), vec![c, c, 3, 3]);
                    put(format!("{CM_PREFIX}input_blocks.{bi}.0.op.bias"), vec![c]);
                }
            }
        }
        let cmid = *cfg.trunk.block_out_channels.last().unwrap() as usize;
        for (p, cin, cout) in [("middle_block.0", cmid, cmid), ("middle_block.2", cmid, cmid)] {
            let full = format!("{CM_PREFIX}{p}");
            put(format!("{full}.in_layers.0.weight"), vec![cin]);
            put(format!("{full}.in_layers.0.bias"), vec![cin]);
            put(format!("{full}.in_layers.2.weight"), vec![cout, cin, 3, 3]);
            put(format!("{full}.in_layers.2.bias"), vec![cout]);
            put(format!("{full}.emb_layers.1.weight"), vec![cout, cfg.trunk.time_embed_dim as usize]);
            put(format!("{full}.emb_layers.1.bias"), vec![cout]);
            put(format!("{full}.out_layers.0.weight"), vec![cout]);
            put(format!("{full}.out_layers.0.bias"), vec![cout]);
            put(format!("{full}.out_layers.3.weight"), vec![cout, cout, 3, 3]);
            put(format!("{full}.out_layers.3.bias"), vec![cout]);
        }
        {
            let level = cfg.trunk.levels() - 1;
            let x = cfg.trunk.cross_attention_dim as usize;
            let inner = 2 * (4 * cmid);
            let tp = format!("{CM_PREFIX}middle_block.1");
            put(format!("{tp}.norm.weight"), vec![cmid]);
            put(format!("{tp}.norm.bias"), vec![cmid]);
            put(format!("{tp}.proj_in.weight"), vec![cmid, cmid]);
            put(format!("{tp}.proj_in.bias"), vec![cmid]);
            for k in 0..cfg.trunk.transformer_layers_per_block[level] {
                let b = format!("{tp}.transformer_blocks.{k}");
                for nm in ["norm1", "norm2", "norm3"] {
                    put(format!("{b}.{nm}.weight"), vec![cmid]);
                    put(format!("{b}.{nm}.bias"), vec![cmid]);
                }
                for nm in ["to_q", "to_k", "to_v"] {
                    put(format!("{b}.attn1.{nm}.weight"), vec![cmid, cmid]);
                }
                put(format!("{b}.attn1.to_out.0.weight"), vec![cmid, cmid]);
                put(format!("{b}.attn1.to_out.0.bias"), vec![cmid]);
                put(format!("{b}.attn2.to_q.weight"), vec![cmid, cmid]);
                put(format!("{b}.attn2.to_k.weight"), vec![cmid, x]);
                put(format!("{b}.attn2.to_v.weight"), vec![cmid, x]);
                put(format!("{b}.attn2.to_out.0.weight"), vec![cmid, cmid]);
                put(format!("{b}.attn2.to_out.0.bias"), vec![cmid]);
                put(format!("{b}.ff.net.0.proj.weight"), vec![inner, cmid]);
                put(format!("{b}.ff.net.0.proj.bias"), vec![inner]);
                put(format!("{b}.ff.net.2.weight"), vec![cmid, inner / 2]);
                put(format!("{b}.ff.net.2.bias"), vec![cmid]);
            }
            put(format!("{tp}.proj_out.weight"), vec![cmid, cmid]);
            put(format!("{tp}.proj_out.bias"), vec![cmid]);
        }

        // ---- adaptors + denoise_encoder: already 1:1-named, so this can
        // just synthesise straight from the manifests.
        for (name, shape) in cfg.adaptors.tensor_manifest() {
            if let Some(base) = name.strip_suffix(".attn.kv.weight") {
                let half = shape[0] / 2;
                put(format!("{PM_PREFIX}{base}.attn.to_k.weight"), vec![half, shape[1]]);
                put(format!("{PM_PREFIX}{base}.attn.to_v.weight"), vec![half, shape[1]]);
            } else {
                put(format!("{PM_PREFIX}{name}"), shape);
            }
        }
        for (name, shape) in crate::config::denoise_encoder_manifest(&cfg.denoise_encoder) {
            let key = name.strip_prefix("denoise_encoder.").unwrap();
            put(format!("{DE_PREFIX}{key}"), shape);
        }

        raw
    }

    #[test]
    fn synthetic_checkpoint_round_trips_and_covers_the_manifest() {
        let cfg = SupirConfig::tiny();
        let raw = synthetic_checkpoint(&cfg);
        let out = remap(raw, &cfg).expect("remap should succeed on a well-formed synthetic checkpoint");
        let manifest = cfg.tensor_manifest();
        assert_eq!(out.len(), manifest.len());
        for (name, shape) in &manifest {
            let (s, _) = out.get(name).unwrap_or_else(|| panic!("missing {name} after remap"));
            assert_eq!(s, shape, "{name}");
        }
    }

    #[test]
    fn mask_lq_is_rejected_by_name() {
        let cfg = SupirConfig::tiny();
        // A checkpoint WITHOUT mask_LQ succeeds - the control this test's
        // real assertion needs, to prove the rejection below is about
        // mask_LQ specifically and not some other synthetic-checkpoint bug.
        let raw = synthetic_checkpoint(&cfg);
        assert!(remap(raw, &cfg).is_ok());

        let mut raw_with = synthetic_checkpoint(&cfg);
        raw_with.insert(MASK_LQ.to_string(), (vec![1, 4, 8, 8], vec![0.0; 256]));
        let err = remap(raw_with, &cfg).unwrap_err();
        assert!(err.contains("mask_LQ"), "error does not name mask_LQ: {err}");
    }

    #[test]
    fn a_missing_trunk_tensor_is_named_in_the_error() {
        let cfg = SupirConfig::tiny();
        let mut raw = synthetic_checkpoint(&cfg);
        raw.remove(&format!("{CM_PREFIX}time_embed.0.weight"));
        let err = remap(raw, &cfg).unwrap_err();
        assert!(err.contains("time_embed.0.weight"), "error does not name the missing tensor: {err}");
    }

    #[test]
    fn an_unexpected_extra_tensor_is_rejected() {
        let cfg = SupirConfig::tiny();
        let mut raw = synthetic_checkpoint(&cfg);
        raw.insert(format!("{PM_PREFIX}999.zero_conv.weight"), (vec![1], vec![0.0]));
        assert!(remap(raw, &cfg).is_err());
    }
}
