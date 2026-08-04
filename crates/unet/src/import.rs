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

use crate::config::UNetConfig;

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
    mut raw: HashMap<String, (Vec<usize>, Vec<f32>)>,
    cfg: &UNetConfig,
) -> Result<Tensors, String> {
    let mut out: Tensors = HashMap::new();
    let mut used: HashSet<String> = HashSet::new();

    let take = |raw: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
                    used: &mut HashSet<String>,
                    name: &str|
     -> Result<(Vec<usize>, Vec<f32>), String> {
        used.insert(name.to_string());
        raw.remove(name).ok_or_else(|| format!("unet import: missing source tensor {name}"))
    };

    // Everything that maps 1:1 by name is driven off the manifest itself, so a
    // manifest edit cannot forget to import its new tensor.
    let manifest = cfg.tensor_manifest();
    let fused: HashSet<&str> = ["qkv.weight", "kv.weight", "ff.hidden", "ff.gate", "ff.out"]
        .into_iter()
        .collect();
    let is_fused = |n: &str| fused.iter().any(|f| n.contains(f)) || n.contains(".to_out.");

    for (name, shape) in &manifest {
        if is_fused(name) {
            continue;
        }
        let (s, d) = take(&mut raw, &mut used, name)?;
        check_shape(name, &s, shape)?;
        out.insert(name.clone(), (s, d));
    }

    // ---- the fused / split / renamed leaves -------------------------------
    for (name, shape) in &manifest {
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
                let (s, d) = raw.get(&src).ok_or_else(|| format!("unet import: missing {src}"))?.clone();
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
                return Err(format!("unet import: unhandled fused name {name}"));
            }
        } else {
            return Err(format!("unet import: unhandled fused name {name}"));
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
            "unet import: {} source tensors unused, e.g. {:?}",
            extra.len(),
            &extra[..extra.len().min(8)]
        ));
    }
    if out.len() != manifest.len() {
        return Err(format!("unet import: produced {} of {} tensors", out.len(), manifest.len()));
    }
    Ok(out)
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
fn check_shape(name: &str, got: &[usize], want: &[usize]) -> Result<(), String> {
    if got != want {
        return Err(format!("unet import: {name} shape {got:?}, expected {want:?}"));
    }
    Ok(())
}
