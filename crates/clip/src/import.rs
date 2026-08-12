// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import with **two-way coverage validation**: every tensor the
//! config's manifest names is produced exactly once with the right shape, AND
//! no source tensor is left unused. A mismatch is an error naming the tensor -
//! never a silent zero-fill (the `flux2::import` / `qwen3::import` discipline).
//!
//! Two source layouts:
//!   * **HF `CLIPTextModel(WithProjection)`** (`text_model.*`) - q/k/v already
//!     split, so the only surgery is FUSING them into brain's `[3H, H]` qkv.
//!   * **EVA02-CLIP `.pt`** (`visual.*`) - three bias-asymmetric linears fused
//!     the same way (k's bias third is zero, because the reference's k linear
//!     genuinely has none), and the q/k rows permuted into brain's half-split
//!     RoPE channel order (see [`EvaVisionConfig::head_perm`]).
//!
//! An open_clip-native bigG checkpoint (fused `in_proj_weight`, `ln_final`,
//! `text_projection` as a bare Parameter) is a DIFFERENT layout and is not
//! handled here: SDXL/diffusers ship the HF one, which is what the goldens were
//! dumped through. Adding it means splitting the fused qkv at this boundary.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;
use checkpoint::torchpt::NamedTensor;

use crate::config::{ClipTextConfig, EvaVisionConfig};

/// name -> (shape, fp32 data), keyed by canonical brain names.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn validate(map: Tensors, manifest: &[(String, Vec<usize>)]) -> Result<Tensors, String> {
    for (name, shape) in manifest {
        match map.get(name) {
            None => return Err(format!("clip import: missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("clip import: {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!("clip import: {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> =
            manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> =
            map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("clip import: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// Insert, refusing to overwrite (a duplicate mapping is a bug, not a merge).
fn put(map: &mut Tensors, name: String, shape: Vec<usize>, data: Vec<f32>) -> Result<(), String> {
    if map.insert(name.clone(), (shape, data)).is_some() {
        return Err(format!("clip import: duplicate mapping onto {name}"));
    }
    Ok(())
}

/// Concatenate three `[out, in]` (or `[out]`) slices into brain's fused
/// `[3*out, in]` qkv order: q rows, then k rows, then v rows.
fn fuse3(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.len() + k.len() + v.len());
    out.extend_from_slice(q);
    out.extend_from_slice(k);
    out.extend_from_slice(v);
    out
}

// ---------------------------------------------------------------------------
// text towers
// ---------------------------------------------------------------------------

/// Read an SDXL-style `text_encoder*/` directory. Prefers the fp32
/// `model.safetensors` (or a sharded index); falls back to
/// `model.fp16.safetensors`, which is all the SDXL 1.0 release ships - the
/// reader widens F16 to f32 exactly, and the goldens were dumped from the same
/// bytes, so this is not a precision compromise relative to the reference.
pub fn read_text_encoder(dir: &std::path::Path) -> Result<Vec<StTensor>, String> {
    match checkpoint::safetensors::read_model_dir(dir) {
        Ok(t) => Ok(t),
        Err(primary) => {
            let fp16 = dir.join("model.fp16.safetensors");
            if fp16.exists() {
                return checkpoint::safetensors::read(
                    fp16.to_str().ok_or("clip import: non-utf8 path")?,
                );
            }
            Err(primary)
        }
    }
}

/// HF tensor names that are registered buffers, not parameters, and are
/// deliberately dropped. `position_ids` is `arange(max_positions)`; brain's
/// `pos_add` derives the position from the row index. Older `transformers`
/// exports ship it, newer ones do not - the manifest count is asserted either
/// way, so a drop can never hide a missing weight.
const TEXT_DROP: [&str; 1] = ["text_model.embeddings.position_ids"];

/// Map one HF `CLIPTextModel` tensor name to a brain name, or to the q/k/v slot
/// it occupies in the fused `blocks.N.qkv.*`.
enum TextSlot {
    Direct(String),
    /// `(fused brain name, slot)` - 0 = q, 1 = k, 2 = v.
    Qkv(String, usize),
    Drop,
}

fn text_slot(name: &str) -> Option<TextSlot> {
    if TEXT_DROP.contains(&name) {
        return Some(TextSlot::Drop);
    }
    let direct = |s: &str| Some(TextSlot::Direct(s.to_string()));
    match name {
        "text_model.embeddings.token_embedding.weight" => return direct("tok.weight"),
        "text_model.embeddings.position_embedding.weight" => return direct("pos.weight"),
        "text_model.final_layer_norm.weight" => return direct("final_norm.weight"),
        "text_model.final_layer_norm.bias" => return direct("final_norm.bias"),
        "text_projection.weight" => return direct("text_projection.weight"),
        _ => {}
    }
    let rest = name.strip_prefix("text_model.encoder.layers.")?;
    let (n, leaf) = rest.split_once('.')?;
    let slot = match leaf {
        "self_attn.q_proj.weight" => return Some(TextSlot::Qkv(format!("blocks.{n}.qkv.weight"), 0)),
        "self_attn.k_proj.weight" => return Some(TextSlot::Qkv(format!("blocks.{n}.qkv.weight"), 1)),
        "self_attn.v_proj.weight" => return Some(TextSlot::Qkv(format!("blocks.{n}.qkv.weight"), 2)),
        "self_attn.q_proj.bias" => return Some(TextSlot::Qkv(format!("blocks.{n}.qkv.bias"), 0)),
        "self_attn.k_proj.bias" => return Some(TextSlot::Qkv(format!("blocks.{n}.qkv.bias"), 1)),
        "self_attn.v_proj.bias" => return Some(TextSlot::Qkv(format!("blocks.{n}.qkv.bias"), 2)),
        "self_attn.out_proj.weight" => "proj.weight",
        "self_attn.out_proj.bias" => "proj.bias",
        "layer_norm1.weight" => "ln1.weight",
        "layer_norm1.bias" => "ln1.bias",
        "layer_norm2.weight" => "ln2.weight",
        "layer_norm2.bias" => "ln2.bias",
        "mlp.fc1.weight" => "fc1.weight",
        "mlp.fc1.bias" => "fc1.bias",
        "mlp.fc2.weight" => "fc2.weight",
        "mlp.fc2.bias" => "fc2.bias",
        _ => return None,
    };
    Some(TextSlot::Direct(format!("blocks.{n}.{slot}")))
}

/// The three q/k/v thirds of one fused attention projection, in `TextSlot::Qkv`
/// slot order, as they arrive from the checkpoint: `(shape, data)` each, `None`
/// until that third is seen.
type QkvThirds = [Option<(Vec<usize>, Vec<f32>)>; 3];

/// Import an HF `CLIPTextModel` / `CLIPTextModelWithProjection` checkpoint.
pub fn import_text(tensors: Vec<StTensor>, cfg: &ClipTextConfig) -> Result<Tensors, String> {
    let mut map: Tensors = HashMap::new();
    let mut qkv: HashMap<String, QkvThirds> = HashMap::new();

    for t in tensors {
        match text_slot(&t.name) {
            None => return Err(format!("clip import: unrecognized CLIP text tensor {}", t.name)),
            Some(TextSlot::Drop) => {}
            Some(TextSlot::Direct(brain)) => put(&mut map, brain, t.shape, t.data)?,
            Some(TextSlot::Qkv(fused, slot)) => {
                let e = qkv.entry(fused.clone()).or_default();
                if e[slot].is_some() {
                    return Err(format!("clip import: duplicate qkv third {}", t.name));
                }
                e[slot] = Some((t.shape, t.data));
            }
        }
    }

    for (name, thirds) in qkv {
        let [q, k, v] = thirds;
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            return Err(format!("clip import: incomplete q/k/v set for {name}"));
        };
        let shape = match q.0.len() {
            1 => vec![3 * q.0[0]],
            2 => vec![3 * q.0[0], q.0[1]],
            _ => return Err(format!("clip import: odd qkv rank for {name}: {:?}", q.0)),
        };
        if k.0 != q.0 || v.0 != q.0 {
            return Err(format!("clip import: q/k/v shape mismatch for {name}"));
        }
        put(&mut map, name, shape, fuse3(&q.1, &k.1, &v.1))?;
    }

    validate(map, &cfg.tensor_manifest())
}

// ---------------------------------------------------------------------------
// EVA02 image tower
// ---------------------------------------------------------------------------

/// What the EVA importer deliberately left behind, so a caller can assert it
/// rather than trust it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaImportReport {
    /// `visual.*` keys mapped into parameters.
    pub mapped: usize,
    /// `visual.*.rope.freqs_{cos,sin}` buffers skipped. The reference RECOMPUTES
    /// these in fp32 at construction and discards the checkpoint's fp16 copies,
    /// so importing them would be importing the wrong numbers. Expected
    /// `2 * (layers + 1)` = 50 for EVA02-L/336 (one pair per block + the tower's
    /// own shared pair).
    pub skipped_rope_buffers: usize,
    /// Keys outside `visual.` - the joint checkpoint's TEXT tower, a different
    /// model. Counted, not silently ignored.
    pub skipped_non_visual: usize,
}

/// Permute a `[heads*head_dim, in]` projection's ROWS (or a `[heads*head_dim]`
/// bias) within each head by `perm`: `out[h*hd + d] = src[h*hd + perm[d]]`.
fn permute_head_rows(src: &[f32], heads: usize, hd: usize, row: usize, perm: &[usize]) -> Vec<f32> {
    debug_assert_eq!(perm.len(), hd);
    debug_assert_eq!(src.len(), heads * hd * row);
    let mut out = vec![0.0f32; src.len()];
    for h in 0..heads {
        for (d, &pd) in perm.iter().enumerate() {
            let dst0 = (h * hd + d) * row;
            let src0 = (h * hd + pd) * row;
            out[dst0..dst0 + row].copy_from_slice(&src[src0..src0 + row]);
        }
    }
    out
}

/// Import the `visual.` tower of an EVA02-CLIP `.pt` checkpoint.
pub fn import_eva_visual(
    tensors: Vec<NamedTensor>,
    cfg: &EvaVisionConfig,
) -> Result<(Tensors, EvaImportReport), String> {
    let w = cfg.width as usize;
    let heads = cfg.heads as usize;
    let hd = cfg.head_dim() as usize;
    let perm = cfg.head_perm();

    let mut map: Tensors = HashMap::new();
    // fused qkv assembly: brain name -> [q, k, v]
    let mut qkv_w: HashMap<String, [Option<Vec<f32>>; 3]> = HashMap::new();
    let mut qkv_b: HashMap<String, [Option<Vec<f32>>; 3]> = HashMap::new();
    let mut rep = EvaImportReport { mapped: 0, skipped_rope_buffers: 0, skipped_non_visual: 0 };

    for t in tensors {
        let Some(rest) = t.name.strip_prefix("visual.") else {
            rep.skipped_non_visual += 1;
            continue;
        };
        // The recomputed-not-imported RoPE buffers (block-local and tower-level).
        if rest == "rope.freqs_cos"
            || rest == "rope.freqs_sin"
            || rest.ends_with(".rope.freqs_cos")
            || rest.ends_with(".rope.freqs_sin")
        {
            rep.skipped_rope_buffers += 1;
            continue;
        }
        rep.mapped += 1;

        // stem / head
        let direct: Option<(&str, Vec<usize>)> = match rest {
            // `[1,1,W]` in the checkpoint; brain binds a flat row.
            "cls_token" => Some(("cls_token", vec![w])),
            // `[1, seq, W]` -> `[seq, W]`.
            "pos_embed" => Some(("pos_embed", vec![cfg.seq_len() as usize, w])),
            "patch_embed.proj.weight" => {
                Some(("patch.weight", vec![w, 3, cfg.patch as usize, cfg.patch as usize]))
            }
            "patch_embed.proj.bias" => Some(("patch.bias", vec![w])),
            "norm.weight" => Some(("norm.weight", vec![w])),
            "norm.bias" => Some(("norm.bias", vec![w])),
            "head.weight" => Some(("head.weight", vec![cfg.embed_dim as usize, w])),
            "head.bias" => Some(("head.bias", vec![cfg.embed_dim as usize])),
            _ => None,
        };
        if let Some((brain, shape)) = direct {
            let n: usize = shape.iter().product();
            if t.data.len() != n {
                return Err(format!(
                    "clip import: {} has {} values, expected {n}",
                    t.name,
                    t.data.len()
                ));
            }
            put(&mut map, brain.to_string(), shape, t.data)?;
            continue;
        }

        let Some(brest) = rest.strip_prefix("blocks.") else {
            return Err(format!("clip import: unrecognized EVA visual tensor {}", t.name));
        };
        let Some((n, leaf)) = brest.split_once('.') else {
            return Err(format!("clip import: unrecognized EVA visual tensor {}", t.name));
        };
        // q/k/v: fuse, and permute the q and k HEAD CHANNELS into brain's
        // half-split RoPE order. v is NOT permuted - it never meets RoPE, and
        // the attention output (and therefore `proj`) must stay in reference
        // channel order.
        let fused = match leaf {
            "attn.q_proj.weight" => Some((format!("blocks.{n}.qkv.weight"), 0, true)),
            "attn.k_proj.weight" => Some((format!("blocks.{n}.qkv.weight"), 1, true)),
            "attn.v_proj.weight" => Some((format!("blocks.{n}.qkv.weight"), 2, false)),
            "attn.q_bias" => Some((format!("blocks.{n}.qkv.bias"), 0, true)),
            "attn.v_bias" => Some((format!("blocks.{n}.qkv.bias"), 2, false)),
            _ => None,
        };
        if let Some((name, slot, rotate)) = fused {
            let is_bias = name.ends_with(".bias");
            let row = if is_bias { 1 } else { w };
            if t.data.len() != heads * hd * row {
                return Err(format!(
                    "clip import: {} has {} values, expected {}",
                    t.name,
                    t.data.len(),
                    heads * hd * row
                ));
            }
            let data =
                if rotate { permute_head_rows(&t.data, heads, hd, row, &perm) } else { t.data };
            let slots = if is_bias { &mut qkv_b } else { &mut qkv_w };
            let e = slots.entry(name).or_default();
            if e[slot].is_some() {
                return Err(format!("clip import: duplicate qkv third {}", t.name));
            }
            e[slot] = Some(data);
            continue;
        }

        let m = cfg.mlp_hidden as usize;
        let (brain, shape) = match leaf {
            "norm1.weight" => ("norm1.weight", vec![w]),
            "norm1.bias" => ("norm1.bias", vec![w]),
            "attn.inner_attn_ln.weight" => ("inner_ln.weight", vec![w]),
            "attn.inner_attn_ln.bias" => ("inner_ln.bias", vec![w]),
            "attn.proj.weight" => ("proj.weight", vec![w, w]),
            "attn.proj.bias" => ("proj.bias", vec![w]),
            "norm2.weight" => ("norm2.weight", vec![w]),
            "norm2.bias" => ("norm2.bias", vec![w]),
            "mlp.w1.weight" => ("w1.weight", vec![m, w]),
            "mlp.w1.bias" => ("w1.bias", vec![m]),
            "mlp.w2.weight" => ("w2.weight", vec![m, w]),
            "mlp.w2.bias" => ("w2.bias", vec![m]),
            "mlp.ffn_ln.weight" => ("ffn_ln.weight", vec![m]),
            "mlp.ffn_ln.bias" => ("ffn_ln.bias", vec![m]),
            "mlp.w3.weight" => ("w3.weight", vec![w, m]),
            "mlp.w3.bias" => ("w3.bias", vec![w]),
            _ => return Err(format!("clip import: unrecognized EVA visual tensor {}", t.name)),
        };
        put(&mut map, format!("blocks.{n}.{brain}"), shape, t.data)?;
    }

    // k has NO bias in the reference (`F.linear(x, k_proj.weight, bias=None)`),
    // so its third of the fused bias is exactly zero. Materializing it is what
    // lets one `bias_add` serve the whole fused projection.
    for (name, thirds) in qkv_b.iter_mut() {
        if thirds[1].is_none() {
            thirds[1] = Some(vec![0.0f32; w]);
        }
        let [q, k, v] = std::mem::take(thirds);
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            return Err(format!("clip import: incomplete q/v bias set for {name}"));
        };
        put(&mut map, name.clone(), vec![3 * w], fuse3(&q, &k, &v))?;
    }
    for (name, thirds) in qkv_w.iter_mut() {
        let [q, k, v] = std::mem::take(thirds);
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            return Err(format!("clip import: incomplete q/k/v set for {name}"));
        };
        put(&mut map, name.clone(), vec![3 * w, w], fuse3(&q, &k, &v))?;
    }

    let expect_rope = 2 * (cfg.layers as usize + 1);
    if rep.skipped_rope_buffers != expect_rope {
        return Err(format!(
            "clip import: skipped {} rope freq buffers, expected {expect_rope} \
             (2 per block + the tower's own pair) - the skip set is asserted, not assumed",
            rep.skipped_rope_buffers
        ));
    }
    let map = validate(map, &cfg.tensor_manifest())?;
    Ok((map, rep))
}

// ---------------------------------------------------------------------------
// vanilla CLIP-L image tower (DeepSeek-OCR's mmproj)
// ---------------------------------------------------------------------------

/// The prefix `gguf::deepseek_ocr_vision` gives the CLIP branch's tensors in the
/// brain-native checkpoint it writes.
pub const DEEPSEEK_OCR_CLIP_PREFIX: &str = "vision.clip.";

/// Import the CLIP-L branch of a DeepSeek-OCR mmproj that
/// `gguf::deepseek_ocr_vision::import` has already converted to brain's native
/// `.safetensors`.
///
/// This is a **prefix strip and nothing else**: that module's brain-side names
/// are `vision.clip.<leaf>` and
/// [`ClipVisionConfig::tensor_manifest`](crate::config::ClipVisionConfig::tensor_manifest)
/// is that leaf list verbatim, asserted by
/// `config::tests::manifest_matches_the_gguf_param_list`. There is deliberately
/// no second name table here - a name mapping that exists twice is a name
/// mapping that will disagree.
///
/// Tensors outside the prefix (the SAM branch, the compressor, the projector)
/// belong to other crates and are IGNORED rather than rejected, so one imported
/// checkpoint feeds every stage of the tower. The usual two-way coverage still
/// holds over what is claimed: every manifest entry must be present with the
/// right shape, and no `vision.clip.*` tensor may be left over.
pub fn import_deepseek_ocr_vision(
    tensors: Vec<StTensor>,
    cfg: &crate::config::ClipVisionConfig,
) -> Result<Tensors, String> {
    let mut map: Tensors = HashMap::new();
    for t in tensors {
        let Some(leaf) = t.name.strip_prefix(DEEPSEEK_OCR_CLIP_PREFIX) else {
            continue; // SAM / compressor / projector - not this tower's tensors
        };
        put(&mut map, leaf.to_string(), t.shape, t.data)?;
    }
    validate(map, &cfg.tensor_manifest())
}

// ---------------------------------------------------------------------------
// vanilla CLIP-L image tower, STRAIGHT FROM THE mmproj GGUF
// ---------------------------------------------------------------------------

/// Import the CLIP branch **directly from an mmproj GGUF**, without first
/// materialising a brain-native `.safetensors` of the whole file.
///
/// [`import_deepseek_ocr_vision`] above is the safetensors-side prefix strip and
/// stays the path for a checkpoint already converted; this module is the
/// GGUF-side one, and it exists because converting the whole mmproj to fp32 just
/// to read one tower costs ~1.6 GB of disk and dequantizes the SAM branch and
/// the projector for nothing. It is deliberately the same shape as
/// `sam1::import` - the mmproj loader's own classifier, narrowed to one tower,
/// with every other tensor recorded as a `Mapped::Dropped` so the driver's
/// two-way coverage check still runs over all 476 source tensors.
pub mod gguf_mmproj {
    use std::collections::HashMap;

    use checkpoint::gguf::MmapGguf;
    use gguf::deepseek_ocr_vision as dsv;
    use gguf::Mapped;

    use crate::config::ClipVisionConfig;

    use super::DEEPSEEK_OCR_CLIP_PREFIX as PREFIX;

    const LABEL: &str = "clip-vision";

    /// Derive the tower's config from an open mmproj GGUF. Note this applies
    /// [`ClipVisionConfig::from_gguf`]'s LayerNorm-epsilon override - read that
    /// function's doc before assuming the file's key is what runs.
    pub fn config_from_gguf(mg: &MmapGguf) -> Result<ClipVisionConfig, String> {
        Ok(ClipVisionConfig::from_gguf(&dsv::config_from_gguf(mg)?))
    }

    /// `deepseek_ocr_vision::classify`, narrowed to the CLIP tower. A closure
    /// factory rather than an inline match so the dry run and the real load
    /// provably classify identically.
    fn clip_only(full: &dsv::DeepseekOcrVisionConfig) -> impl Fn(&str) -> Result<Mapped, String> + '_ {
        move |name: &str| match dsv::classify(name, full)? {
            Mapped::Simple(n) if n.starts_with(PREFIX) => Ok(Mapped::Simple(n)),
            _ => Ok(Mapped::Dropped("not part of the CLIP vision tower")),
        }
    }

    /// The manifest under the loader's prefixed names (its `param_list` is the
    /// leaf list with `vision.clip.` in front - asserted by
    /// `config::tests::manifest_matches_the_gguf_param_list`).
    fn prefixed(cfg: &ClipVisionConfig) -> Vec<(String, usize)> {
        cfg.tensor_manifest()
            .into_iter()
            .map(|(n, shape)| (format!("{PREFIX}{n}"), shape.iter().product()))
            .collect()
    }

    /// Header-only coverage check: every source tensor classified, every
    /// declared parameter produced, no tensor data read.
    pub fn dry_run(mg: &MmapGguf) -> Result<(ClipVisionConfig, gguf::ImportStats), String> {
        let full = dsv::config_from_gguf(mg)?;
        let cfg = ClipVisionConfig::from_gguf(&full);
        let stats = gguf::import::dry_run(mg, &prefixed(&cfg), &clip_only(&full), LABEL)?;
        Ok((cfg, stats))
    }

    /// Load the tower's weights into an init map keyed by the **leaf** names
    /// [`crate::model::ClipVision::new_on`] wants (the prefix is stripped after
    /// the coverage check, so the check still runs against the loader's own
    /// names).
    pub fn weights_from_gguf(mg: &MmapGguf) -> Result<(ClipVisionConfig, HashMap<String, Vec<f32>>), String> {
        let full = dsv::config_from_gguf(mg)?;
        let cfg = ClipVisionConfig::from_gguf(&full);
        let w = gguf::import::to_map(mg, &prefixed(&cfg), &clip_only(&full), LABEL)?;
        Ok((
            cfg,
            w.into_iter()
                .map(|(n, d)| {
                    let leaf = n.strip_prefix(PREFIX).unwrap_or(&n).to_string();
                    (leaf, d)
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_source(cfg: &ClipTextConfig, with_position_ids: bool) -> Vec<StTensor> {
        let h = cfg.hidden as usize;
        let i = cfg.intermediate as usize;
        let mut v = vec![
            StTensor {
                name: "text_model.embeddings.token_embedding.weight".into(),
                shape: vec![cfg.vocab as usize, h],
                data: vec![0.1; cfg.vocab as usize * h],
            },
            StTensor {
                name: "text_model.embeddings.position_embedding.weight".into(),
                shape: vec![cfg.max_positions as usize, h],
                data: vec![0.2; cfg.max_positions as usize * h],
            },
            StTensor {
                name: "text_model.final_layer_norm.weight".into(),
                shape: vec![h],
                data: vec![1.0; h],
            },
            StTensor {
                name: "text_model.final_layer_norm.bias".into(),
                shape: vec![h],
                data: vec![0.0; h],
            },
        ];
        if with_position_ids {
            v.push(StTensor {
                name: "text_model.embeddings.position_ids".into(),
                shape: vec![1, cfg.max_positions as usize],
                data: (0..cfg.max_positions).map(|x| x as f32).collect(),
            });
        }
        if let Some(p) = cfg.projection {
            v.push(StTensor {
                name: "text_projection.weight".into(),
                shape: vec![p as usize, h],
                data: vec![0.3; p as usize * h],
            });
        }
        for l in 0..cfg.layers {
            let p = format!("text_model.encoder.layers.{l}");
            for (slot, tag) in [("q", 1.0f32), ("k", 2.0), ("v", 3.0)] {
                v.push(StTensor {
                    name: format!("{p}.self_attn.{slot}_proj.weight"),
                    shape: vec![h, h],
                    data: vec![tag; h * h],
                });
                v.push(StTensor {
                    name: format!("{p}.self_attn.{slot}_proj.bias"),
                    shape: vec![h],
                    data: vec![tag; h],
                });
            }
            for (n, s) in [
                ("self_attn.out_proj.weight", vec![h, h]),
                ("self_attn.out_proj.bias", vec![h]),
                ("layer_norm1.weight", vec![h]),
                ("layer_norm1.bias", vec![h]),
                ("layer_norm2.weight", vec![h]),
                ("layer_norm2.bias", vec![h]),
                ("mlp.fc1.weight", vec![i, h]),
                ("mlp.fc1.bias", vec![i]),
                ("mlp.fc2.weight", vec![h, i]),
                ("mlp.fc2.bias", vec![h]),
            ] {
                let n_el: usize = s.iter().product();
                v.push(StTensor { name: format!("{p}.{n}"), shape: s, data: vec![0.5; n_el] });
            }
        }
        v
    }

    /// Toy dims so the coverage tests stay fast in debug builds.
    fn tiny_text() -> ClipTextConfig {
        ClipTextConfig {
            hidden: 8,
            intermediate: 16,
            layers: 2,
            heads: 2,
            max_positions: 6,
            vocab: 20,
            ..ClipTextConfig::clip_l()
        }
    }

    #[test]
    fn text_import_fuses_qkv_and_covers_both_directions() {
        let cfg = tiny_text();
        let map = import_text(text_source(&cfg, false), &cfg).unwrap();
        assert_eq!(map.len(), cfg.tensor_manifest().len());
        let h = cfg.hidden as usize;
        let (s, w) = &map["blocks.0.qkv.weight"];
        assert_eq!(s, &vec![3 * h, h]);
        assert_eq!((w[0], w[h * h], w[2 * h * h]), (1.0, 2.0, 3.0));
        let (s, b) = &map["blocks.1.qkv.bias"];
        assert_eq!(s, &vec![3 * h]);
        assert_eq!((b[0], b[h], b[2 * h]), (1.0, 2.0, 3.0));
        // the non-parameter buffer is dropped, not counted as coverage
        let map = import_text(text_source(&cfg, true), &cfg).unwrap();
        assert_eq!(map.len(), cfg.tensor_manifest().len());
    }

    #[test]
    fn text_import_errors_name_the_tensor() {
        let cfg = tiny_text();
        let mut short = text_source(&cfg, false);
        short.retain(|t| t.name != "text_model.encoder.layers.1.mlp.fc2.bias");
        let err = import_text(short, &cfg).unwrap_err();
        assert!(err.contains("blocks.1.fc2.bias"), "{err}");

        let mut extra = text_source(&cfg, false);
        extra.push(StTensor {
            name: "text_model.encoder.layers.0.mystery".into(),
            shape: vec![1],
            data: vec![0.0],
        });
        let err = import_text(extra, &cfg).unwrap_err();
        assert!(err.contains("mystery"), "{err}");

        let mut wrong = text_source(&cfg, false);
        for t in wrong.iter_mut() {
            if t.name == "text_model.final_layer_norm.weight" {
                t.shape = vec![cfg.hidden as usize + 1];
                t.data.push(0.0);
            }
        }
        let err = import_text(wrong, &cfg).unwrap_err();
        assert!(err.contains("final_norm.weight"), "{err}");
    }

    fn tiny_eva() -> EvaVisionConfig {
        EvaVisionConfig {
            image_size: 8,
            patch: 4,
            width: 8,
            layers: 2,
            heads: 2,
            mlp_hidden: 12,
            embed_dim: 6,
            ..EvaVisionConfig::eva02_l336()
        }
    }

    fn eva_source(cfg: &EvaVisionConfig) -> Vec<NamedTensor> {
        let w = cfg.width as usize;
        let m = cfg.mlp_hidden as usize;
        let p = cfg.patch as usize;
        let nt = |name: &str, shape: Vec<usize>, fill: f32| {
            let n: usize = shape.iter().product();
            NamedTensor { name: name.into(), shape, data: vec![fill; n] }
        };
        let mut v = vec![
            nt("visual.cls_token", vec![1, 1, w], 0.7),
            nt("visual.pos_embed", vec![1, cfg.seq_len() as usize, w], 0.8),
            nt("visual.patch_embed.proj.weight", vec![w, 3, p, p], 0.1),
            nt("visual.patch_embed.proj.bias", vec![w], 0.2),
            nt("visual.norm.weight", vec![w], 1.0),
            nt("visual.norm.bias", vec![w], 0.0),
            nt("visual.head.weight", vec![cfg.embed_dim as usize, w], 0.3),
            nt("visual.head.bias", vec![cfg.embed_dim as usize], 0.4),
            nt("visual.rope.freqs_cos", vec![cfg.num_patches() as usize, 4], 1.0),
            nt("visual.rope.freqs_sin", vec![cfg.num_patches() as usize, 4], 0.0),
            // a text-tower key from the same joint checkpoint
            nt("text.token_embedding.weight", vec![4, 4], 0.0),
        ];
        for l in 0..cfg.layers {
            let b = format!("visual.blocks.{l}");
            v.push(nt(&format!("{b}.attn.rope.freqs_cos"), vec![cfg.num_patches() as usize, 4], 1.0));
            v.push(nt(&format!("{b}.attn.rope.freqs_sin"), vec![cfg.num_patches() as usize, 4], 0.0));
            // q/k/v weights carry a per-channel ramp so the permutation is observable
            let mut qw = vec![0.0f32; w * w];
            for (r, row) in qw.chunks_mut(w).enumerate() {
                row.fill(r as f32);
            }
            v.push(NamedTensor {
                name: format!("{b}.attn.q_proj.weight"),
                shape: vec![w, w],
                data: qw.clone(),
            });
            v.push(NamedTensor {
                name: format!("{b}.attn.k_proj.weight"),
                shape: vec![w, w],
                data: qw,
            });
            v.push(nt(&format!("{b}.attn.v_proj.weight"), vec![w, w], 5.0));
            v.push(NamedTensor {
                name: format!("{b}.attn.q_bias"),
                shape: vec![w],
                data: (0..w).map(|i| i as f32).collect(),
            });
            v.push(nt(&format!("{b}.attn.v_bias"), vec![w], 9.0));
            for (n, s) in [
                ("norm1.weight", vec![w]),
                ("norm1.bias", vec![w]),
                ("attn.inner_attn_ln.weight", vec![w]),
                ("attn.inner_attn_ln.bias", vec![w]),
                ("attn.proj.weight", vec![w, w]),
                ("attn.proj.bias", vec![w]),
                ("norm2.weight", vec![w]),
                ("norm2.bias", vec![w]),
                ("mlp.w1.weight", vec![m, w]),
                ("mlp.w1.bias", vec![m]),
                ("mlp.w2.weight", vec![m, w]),
                ("mlp.w2.bias", vec![m]),
                ("mlp.ffn_ln.weight", vec![m]),
                ("mlp.ffn_ln.bias", vec![m]),
                ("mlp.w3.weight", vec![w, m]),
                ("mlp.w3.bias", vec![w]),
            ] {
                v.push(nt(&format!("{b}.{n}"), s, 0.5));
            }
        }
        v
    }

    #[test]
    fn eva_import_fuses_zero_fills_k_bias_and_permutes_qk() {
        let cfg = tiny_eva();
        let (map, rep) = import_eva_visual(eva_source(&cfg), &cfg).unwrap();
        assert_eq!(map.len(), cfg.tensor_manifest().len());
        assert_eq!(rep.skipped_rope_buffers, 2 * (cfg.layers as usize + 1));
        assert_eq!(rep.skipped_non_visual, 1);

        let w = cfg.width as usize;
        let hd = cfg.head_dim() as usize;
        let perm = cfg.head_perm();
        let (s, qkv) = &map["blocks.0.qkv.weight"];
        assert_eq!(s, &vec![3 * w, w]);
        // q rows permuted within each head; the ramp makes the source row visible
        for h in 0..cfg.heads as usize {
            for d in 0..hd {
                assert_eq!(qkv[(h * hd + d) * w], (h * hd + perm[d]) as f32);
            }
        }
        // v rows untouched
        assert_eq!(qkv[2 * w * w], 5.0);
        // k's bias third is exactly zero; q's is permuted; v's is untouched
        let (_, b) = &map["blocks.0.qkv.bias"];
        assert!(b[w..2 * w].iter().all(|&x| x == 0.0));
        for h in 0..cfg.heads as usize {
            for d in 0..hd {
                assert_eq!(b[h * hd + d], (h * hd + perm[d]) as f32);
            }
        }
        assert!(b[2 * w..].iter().all(|&x| x == 9.0));
    }

    #[test]
    fn eva_import_rejects_a_missing_tensor_and_a_short_rope_skip_set() {
        let cfg = tiny_eva();
        let mut short = eva_source(&cfg);
        short.retain(|t| t.name != "visual.blocks.1.mlp.w3.bias");
        let err = import_eva_visual(short, &cfg).unwrap_err();
        assert!(err.contains("blocks.1.w3.bias"), "{err}");

        let mut no_rope = eva_source(&cfg);
        no_rope.retain(|t| !t.name.ends_with("rope.freqs_cos"));
        let err = import_eva_visual(no_rope, &cfg).unwrap_err();
        assert!(err.contains("rope freq buffers"), "{err}");
    }

    fn tiny_vision() -> crate::config::ClipVisionConfig {
        crate::config::ClipVisionConfig {
            shape: gguf::deepseek_ocr_vision::ClipConfig {
                d_model: 6,
                n_layers: 2,
                n_heads: 2,
                ffn_hidden: 10,
                patch_size: 2,
                image_size: 6,
                n_positions: 10,
                layer_norm_eps: 1e-5,
            },
            act: crate::config::TextAct::QuickGelu,
        }
    }

    /// The mmproj source: every manifest tensor under `vision.clip.`, plus a
    /// SAM tensor that must be ignored rather than rejected.
    fn vision_source(cfg: &crate::config::ClipVisionConfig) -> Vec<StTensor> {
        let mut v: Vec<StTensor> = cfg
            .tensor_manifest()
            .into_iter()
            .enumerate()
            .map(|(i, (name, shape))| StTensor {
                name: format!("{DEEPSEEK_OCR_CLIP_PREFIX}{name}"),
                data: vec![i as f32; shape.iter().product()],
                shape,
            })
            .collect();
        v.push(StTensor { name: "vision.sam.pos_embed".into(), shape: vec![2, 3], data: vec![0.0; 6] });
        v
    }

    #[test]
    fn deepseek_ocr_vision_import_is_a_prefix_strip_with_two_way_coverage() {
        let cfg = tiny_vision();
        let map = import_deepseek_ocr_vision(vision_source(&cfg), &cfg).expect("import");
        assert_eq!(map.len(), cfg.tensor_manifest().len());
        assert_eq!(map["patch_embed.weight"].0, vec![6, 3, 2, 2]);
        assert_eq!(map["pos_embed"].0, vec![10, 6]);
        assert_eq!(map["blocks.1.attn.qkv.weight"].0, vec![18, 6]);
    }

    #[test]
    fn deepseek_ocr_vision_import_rejects_a_missing_and_an_extra_tensor() {
        let cfg = tiny_vision();
        let mut short = vision_source(&cfg);
        short.retain(|t| !t.name.ends_with("blocks.1.mlp.fc2.bias"));
        let err = import_deepseek_ocr_vision(short, &cfg).unwrap_err();
        assert!(err.contains("blocks.1.mlp.fc2.bias"), "{err}");

        let mut extra = vision_source(&cfg);
        extra.push(StTensor {
            name: format!("{DEEPSEEK_OCR_CLIP_PREFIX}post_norm.weight"),
            shape: vec![6],
            data: vec![0.0; 6],
        });
        let err = import_deepseek_ocr_vision(extra, &cfg).unwrap_err();
        assert!(err.contains("post_norm.weight"), "{err}");
    }
}
