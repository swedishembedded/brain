// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a diffusers `UNet2DConditionModel` checkpoint into the brain-side
//! tensor set, with **two-way** coverage validation.
//!
//! Two-way means: every tensor the manifest declares must be produced (with the
//! declared shape), and every tensor in the source file must be consumed. A
//! one-way check passes a checkpoint whose extra tensors are silently ignored —
//! which is exactly how a variant checkpoint (a ControlNet, an inpainting UNet
//! with 9 input channels, an SD 1.5 UNet with conv `proj_in`) gets loaded as if
//! it were SDXL.
//!
//! The three host-side fusions/splits are documented on
//! [`crate::config::UNetConfig::tensor_manifest`]; they happen here, once, so
//! `model.rs` only ever binds whole buffers.

use std::collections::{HashMap, HashSet};

use crate::config::{BlockKind, UNetConfig};

/// Host tensors by brain-side name: `(shape, row-major f32 data)` — the exact
/// type `vae::blocks::Builder` consumes.
pub type Tensors = vae::blocks::Tensors;

/// Read `<dir>/diffusion_pytorch_model*.safetensors` (or the exact file when
/// `path` names one) and remap it onto `cfg`'s manifest.
pub fn load(path: &str, cfg: &UNetConfig) -> Result<Tensors, String> {
    let p = std::path::Path::new(path);
    let file = if p.is_dir() {
        let mut cands: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("unet import: {path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|f| f.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        cands.sort();
        // Prefer the fp16 variant when both are present: it is the only one the
        // released SDXL repo ships in full, and brain reads F16 exactly.
        cands
            .iter()
            .find(|f| f.to_string_lossy().contains("fp16"))
            .or_else(|| cands.first())
            .ok_or_else(|| format!("unet import: no .safetensors in {path}"))?
            .clone()
    } else {
        p.to_path_buf()
    };
    let src = checkpoint::safetensors::read(file.to_str().ok_or("unet import: non-utf8 path")?)?;
    let raw: HashMap<String, (Vec<usize>, Vec<f32>)> =
        src.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    remap(raw, cfg)
}

/// The pure remap, so a synthetic checkpoint (tests, `crate::init`) exercises
/// exactly the code the real one does.
pub fn remap(
    raw: HashMap<String, (Vec<usize>, Vec<f32>)>,
    cfg: &UNetConfig,
) -> Result<Tensors, String> {
    remap_manifest("unet", raw, &cfg.tensor_manifest())
}

/// [`remap`] driven by an explicit manifest rather than a [`UNetConfig`].
///
/// **Shared with `crates/controlnet`**, which is why it is public and takes
/// `who` for its error messages: a diffusers `ControlNetModel` checkpoint is a
/// UNet-family checkpoint carrying the *same three fused leaves* (`attn1`'s
/// q/k/v, `attn2`'s k/v, the GEGLU `ff.net.0.proj`) under the *same* module
/// names, so a second copy of this loop is a second place those three fusions
/// can be got wrong — and nothing would compare the copies. The
/// ControlNet-only tensors (the conditioning embedder, the zero-convs) map 1:1
/// and need no special case here.
pub fn remap_manifest(
    who: &str,
    mut raw: HashMap<String, (Vec<usize>, Vec<f32>)>,
    manifest: &[(String, Vec<usize>)],
) -> Result<Tensors, String> {
    let mut out: Tensors = HashMap::new();
    let mut used: HashSet<String> = HashSet::new();

    let take = |raw: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
                    used: &mut HashSet<String>,
                    name: &str|
     -> Result<(Vec<usize>, Vec<f32>), String> {
        used.insert(name.to_string());
        raw.remove(name).ok_or_else(|| format!("{who} import: missing source tensor {name}"))
    };

    // Everything that maps 1:1 by name is driven off the manifest itself, so a
    // manifest edit cannot forget to import its new tensor.
    let fused: HashSet<&str> = ["qkv.weight", "kv.weight", "ff.hidden", "ff.gate", "ff.out"]
        .into_iter()
        .collect();
    let is_fused = |n: &str| fused.iter().any(|f| n.contains(f)) || n.contains(".to_out.");

    for (name, shape) in manifest {
        if is_fused(name) {
            continue;
        }
        let (s, d) = take(&mut raw, &mut used, name)?;
        check_shape(name, &s, shape)?;
        out.insert(name.clone(), (s, d));
    }

    // ---- the fused / split / renamed leaves -------------------------------
    for (name, shape) in manifest {
        if !is_fused(name) {
            continue;
        }
        if let Some(base) = name.strip_suffix(".attn1.qkv.weight") {
            let mut data = Vec::with_capacity(shape.iter().product());
            let mut rows = 0usize;
            for leaf in ["to_q", "to_k", "to_v"] {
                let src = format!("{base}.attn1.{leaf}.weight");
                let (s, d) = take(&mut raw, &mut used, &src)?;
                // Per-piece, not just the concatenated total: three wrong
                // widths that happen to sum right would otherwise pass.
                check_shape(&src, &s, &[shape[0] / 3, shape[1]])?;
                rows += s[0];
                data.extend_from_slice(&d);
            }
            check_shape(name, &[rows, shape[1]], shape)?;
            out.insert(name.clone(), (shape.clone(), data));
        } else if let Some(base) = name.strip_suffix(".attn2.kv.weight") {
            let mut data = Vec::with_capacity(shape.iter().product());
            let mut rows = 0usize;
            for leaf in ["to_k", "to_v"] {
                let src = format!("{base}.attn2.{leaf}.weight");
                let (s, d) = take(&mut raw, &mut used, &src)?;
                check_shape(&src, &s, &[shape[0] / 2, shape[1]])?;
                rows += s[0];
                data.extend_from_slice(&d);
            }
            check_shape(name, &[rows, shape[1]], shape)?;
            out.insert(name.clone(), (shape.clone(), data));
        } else if let Some(base) = name.strip_suffix(".weight").or_else(|| name.strip_suffix(".bias")) {
            let is_w = name.ends_with(".weight");
            let leaf = if is_w { "weight" } else { "bias" };
            if let Some(b) = base.strip_suffix(".ff.hidden").or_else(|| base.strip_suffix(".ff.gate")) {
                // GEGLU: `proj` is [2I, C]; `chunk(2, dim=-1)` on its OUTPUT
                // takes the first I columns as `hidden` and the last I as
                // `gate`, which for a row-major [out, in] weight is the first I
                // ROWS and the last I rows. Splitting the wrong way is a silent
                // swap of the gate and the value.
                let src = format!("{b}.ff.net.0.proj.{leaf}");
                let (s, d) = raw.get(&src).ok_or_else(|| format!("{who} import: missing {src}"))?.clone();
                used.insert(src.clone());
                // The SOURCE is `[2I, C]` (or `[2I]`); check that before
                // slicing, so a wrong `C` fails naming `ff.net.0.proj` rather
                // than silently halving into two plausibly-shaped pieces.
                let mut want_src = shape.clone();
                want_src[0] *= 2;
                check_shape(&src, &s, &want_src)?;
                let half = s[0] / 2;
                let per = d.len() / s[0];
                let lo = if base.ends_with("hidden") { 0 } else { half * per };
                let piece = d[lo..lo + half * per].to_vec();
                out.insert(name.clone(), (shape.clone(), piece));
            } else if let Some(b) = base.strip_suffix(".ff.out") {
                let (s, d) = take(&mut raw, &mut used, &format!("{b}.ff.net.2.{leaf}"))?;
                check_shape(name, &s, shape)?;
                out.insert(name.clone(), (s, d));
            } else if let Some(b) = base.strip_suffix(".to_out") {
                let (s, d) = take(&mut raw, &mut used, &format!("{b}.to_out.0.{leaf}"))?;
                check_shape(name, &s, shape)?;
                out.insert(name.clone(), (s, d));
            } else {
                return Err(format!("{who} import: unhandled fused name {name}"));
            }
        } else {
            return Err(format!("{who} import: unhandled fused name {name}"));
        }
    }
    // The GEGLU split reads one source twice; drop it once, after both halves.
    for name in used.iter() {
        raw.remove(name);
    }

    if !raw.is_empty() {
        let mut extra: Vec<&String> = raw.keys().collect();
        extra.sort();
        return Err(format!(
            "{who} import: {} source tensors unused, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(8)]
        ));
    }
    if out.len() != manifest.len() {
        return Err(format!("{who} import: produced {} of {} tensors", out.len(), manifest.len()));
    }
    Ok(out)
}

const LDM_PREFIX: &str = "model.diffusion_model.";

/// Read a CompVis/LDM-format single-file SDXL checkpoint (the
/// `model.diffusion_model.*` half of the upstream-released
/// `sd_xl_base_1.0_*.safetensors`) and remap it onto `cfg`'s manifest - the
/// single-file sibling of [`load`], which only reads the diffusers `unet/`
/// directory layout.
///
/// Needed because `crates/supir` carries no frozen-backbone weights of its
/// own (its checkpoint is the SUPIR-over-SDXL DELTA only, per that crate's
/// module doc) - a real deployment loads the frozen UNet from the SAME
/// single-file release checkpoint the upstream Python reference does, and
/// there is no diffusers-layout `unet/` directory anywhere in that picture.
pub fn load_ldm(path: &str, cfg: &UNetConfig) -> Result<Tensors, String> {
    let src = checkpoint::safetensors::read(path)?;
    let raw: HashMap<String, (Vec<usize>, Vec<f32>)> = src
        .into_iter()
        .filter_map(|t| t.name.strip_prefix(LDM_PREFIX).map(|n| (n.to_string(), (t.shape, t.data))))
        .collect();
    remap_ldm(raw, cfg)
}

/// The pure remap behind [`load_ldm`]: rename the CompVis/LDM outer
/// structure (`input_blocks`/`middle_block`/`output_blocks`/`time_embed`/
/// `label_emb`/`out`, already stripped of [`LDM_PREFIX`]) into this crate's
/// diffusers-style names, then hand off to [`remap_manifest`] for the
/// qkv/kv/GEGLU fusion - the SAME fusion [`remap`] itself applies, so this
/// differs from `load`/`remap` only in the outer-structure rename, never in
/// how a fused leaf is split.
///
/// The down+mid half of this walk duplicates
/// `crates/supir::import::remap_trunk`'s own private LDM rename (that
/// crate needed it first, for `GLVControl`, before the frozen backbone's own
/// up path made a second, full-UNet version necessary here) - `crates/supir`
/// depends on this crate and not the other way around, so hoisting the
/// shared down+mid half out of `supir` and into this module, with `supir`
/// calling back into it, is the right fix but a separate change: this one
/// adds the up path this crate needed and did not have, not a refactor of
/// already-tested code in a crate two levels away.
pub fn remap_ldm(mut local: HashMap<String, (Vec<usize>, Vec<f32>)>, cfg: &UNetConfig) -> Result<Tensors, String> {
    let mut renamed: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

    for (src, dst) in [
        ("time_embed.0", "time_embedding.linear_1"),
        ("time_embed.2", "time_embedding.linear_2"),
        ("label_emb.0.0", "add_embedding.linear_1"),
        ("label_emb.0.2", "add_embedding.linear_2"),
    ] {
        for suf in ["weight", "bias"] {
            let sk = format!("{src}.{suf}");
            let v = local.remove(&sk).ok_or_else(|| format!("unet-ldm import: missing {LDM_PREFIX}{sk}"))?;
            renamed.insert(format!("{dst}.{suf}"), v);
        }
    }

    // ---- down: `input_blocks` -----------------------------------------
    enum DownBlk {
        ConvIn,
        Resnet { level: usize, layer: usize },
        Downsample { level: usize },
    }
    let mut down_walk = vec![DownBlk::ConvIn];
    for level in 0..cfg.levels() {
        for layer in 0..cfg.layers_per_block as usize {
            down_walk.push(DownBlk::Resnet { level, layer });
        }
        if level + 1 < cfg.levels() {
            down_walk.push(DownBlk::Downsample { level });
        }
    }
    for (bi, blk) in down_walk.into_iter().enumerate() {
        match blk {
            DownBlk::ConvIn => {
                for suf in ["weight", "bias"] {
                    let sk = format!("input_blocks.{bi}.0.{suf}");
                    let v = local.remove(&sk).ok_or_else(|| format!("unet-ldm import: missing {LDM_PREFIX}{sk}"))?;
                    renamed.insert(format!("conv_in.{suf}"), v);
                }
            }
            DownBlk::Resnet { level, layer } => {
                ldm_rename_resnet(&mut local, &mut renamed, &format!("input_blocks.{bi}.0"), &format!("down_blocks.{level}.resnets.{layer}"))?;
                if cfg.down_block_types[level] == BlockKind::CrossAttn {
                    ldm_rename_passthrough(&mut local, &mut renamed, &format!("input_blocks.{bi}.1"), &format!("down_blocks.{level}.attentions.{layer}"));
                }
            }
            DownBlk::Downsample { level } => {
                for suf in ["weight", "bias"] {
                    let sk = format!("input_blocks.{bi}.0.op.{suf}");
                    let v = local.remove(&sk).ok_or_else(|| format!("unet-ldm import: missing {LDM_PREFIX}{sk}"))?;
                    renamed.insert(format!("down_blocks.{level}.downsamplers.0.conv.{suf}"), v);
                }
            }
        }
    }

    // ---- mid: `middle_block` -------------------------------------------
    ldm_rename_resnet(&mut local, &mut renamed, "middle_block.0", "mid_block.resnets.0")?;
    ldm_rename_passthrough(&mut local, &mut renamed, "middle_block.1", "mid_block.attentions.0");
    ldm_rename_resnet(&mut local, &mut renamed, "middle_block.2", "mid_block.resnets.1")?;

    // ---- up: `output_blocks` -------------------------------------------
    // One flat `output_blocks` index `k` per (up_block, resnet-layer) pair,
    // exactly mirroring the down side's `input_blocks` numbering: submodule
    // `.0` is always the resnet, `.1` is the transformer when the up-block
    // has one, and an `Upsample` (verified against the real checkpoint
    // header: `output_blocks.{2,5}.2.conv.*`, i.e. AFTER the transformer,
    // for SDXL's own up_block_types) lands at whichever submodule index
    // comes next, on the LAST resnet of every up-block but the last.
    let mut k = 0usize;
    for i in 0..cfg.levels() {
        let skips = cfg.up_skips(i);
        let n_this = skips.len();
        let has_attn = cfg.up_block_types[i] == BlockKind::CrossAttn;
        for j in 0..n_this {
            ldm_rename_resnet(&mut local, &mut renamed, &format!("output_blocks.{k}.0"), &format!("up_blocks.{i}.resnets.{j}"))?;
            let mut next_sub = 1usize;
            if has_attn {
                ldm_rename_passthrough(&mut local, &mut renamed, &format!("output_blocks.{k}.1"), &format!("up_blocks.{i}.attentions.{j}"));
                next_sub = 2;
            }
            if j + 1 == n_this && i + 1 < cfg.levels() {
                for suf in ["weight", "bias"] {
                    let sk = format!("output_blocks.{k}.{next_sub}.conv.{suf}");
                    let v = local.remove(&sk).ok_or_else(|| format!("unet-ldm import: missing {LDM_PREFIX}{sk}"))?;
                    renamed.insert(format!("up_blocks.{i}.upsamplers.0.conv.{suf}"), v);
                }
            }
            k += 1;
        }
    }

    // ---- final head: `out.{0,2}` (GroupNorm, SiLU with no params, Conv2d) -
    for (src, dst) in [("out.0", "conv_norm_out"), ("out.2", "conv_out")] {
        for suf in ["weight", "bias"] {
            let sk = format!("{src}.{suf}");
            let v = local.remove(&sk).ok_or_else(|| format!("unet-ldm import: missing {LDM_PREFIX}{sk}"))?;
            renamed.insert(format!("{dst}.{suf}"), v);
        }
    }

    if !local.is_empty() {
        let mut extra: Vec<&String> = local.keys().collect();
        extra.sort();
        return Err(format!("unet-ldm import: {} unexpected {LDM_PREFIX} tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(8)]));
    }

    remap_manifest("unet-ldm", renamed, &cfg.tensor_manifest())
}

/// Rename one `ResBlock`'s leaves - the SAME rename
/// `crates/supir::import::remap_trunk`'s own private copy applies (see
/// [`remap_ldm`]'s doc for why this is not yet shared).
fn ldm_rename_resnet(
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
            let v = local.remove(&sk).ok_or_else(|| format!("unet-ldm import: missing {LDM_PREFIX}{sk}"))?;
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
/// sub-block, whose inner leaf names already match diffusers.
fn ldm_rename_passthrough(local: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, renamed: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, src: &str, dst: &str) {
    let full = format!("{src}.");
    let keys: Vec<String> = local.keys().filter(|k| k.starts_with(&full)).cloned().collect();
    for k in keys {
        let v = local.remove(&k).expect("just listed");
        let suffix = &k[full.len()..];
        renamed.insert(format!("{dst}.{suffix}"), v);
    }
}

/// Exact shape equality, not just element count.
///
/// Comparing `numel` only is the weaker check it looks like a shorthand for: it
/// accepts a TRANSPOSED weight wherever the two axes are equal, and every
/// square weight in SDXL is one — `attn1.to_q/to_k/to_v/to_out`,
/// `attn2.to_q/to_out`, `proj_in`, `proj_out`, and `time_embedding.linear_2`.
/// It also accepts a `[C, C, 1, 1]` conv `proj_in` (SD 1.5's
/// `use_linear_projection: false`) as if it were SDXL's `[C, C]` linear, which
/// is exactly the variant-checkpoint confusion this importer's two-way coverage
/// exists to reject.
pub(crate) fn check_shape(name: &str, got: &[usize], want: &[usize]) -> Result<(), String> {
    if got != want {
        return Err(format!("import: {name} shape {got:?}, expected {want:?}"));
    }
    Ok(())
}
