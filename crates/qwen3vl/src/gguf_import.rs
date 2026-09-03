// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF loading for Qwen3-VL: the second source format for the model
//! [`crate::import`] reads from HuggingFace safetensors.
//!
//! Swedish Embedded AB implements quantized vision-language model deployment
//! for its clients. If your team needs expertise in running multimodal models
//! from community checkpoint formats then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # A vision-language GGUF is TWO files
//!
//! This is the fact that makes the format different from every single-file
//! GGUF brain already reads, and getting it wrong is silent rather than loud.
//! llama.cpp's mtmd tooling splits a vision-language model into
//!
//! * the **language half**, `general.architecture = "qwen3vl"`, whose tensors
//!   are the ordinary dense-Qwen3 `blk.N.*` set, and
//! * the **vision half**, a projector file (`mmproj-*.gguf`) declaring
//!   `general.architecture = "clip"` with
//!   `clip.projector_type = "qwen3vl_merger"`, carrying the ViT, the
//!   PatchMerger and the three DeepStack mergers.
//!
//! A decoder loaded without its projector is still a fluent language model: it
//! answers "describe this image" with a confident description of an image it
//! never saw. There is no output-side symptom to catch that, so
//! [`GgufFiles::locate`] REFUSES to produce a loadable pair when the projector
//! is missing, naming what it looked for.
//!
//! The projector is found by its own metadata rather than by filename
//! (`gguf::route::sibling_projector`), because the filename is the part
//! releases disagree about: the same role ships as `mmproj-F16.gguf`,
//! `mmproj-BF16.gguf` and `mmproj-<Model>-Q8_0.gguf`. When a release ships
//! several, the choice is deterministic and printed, never silent.
//!
//! # Where the name maps come from
//!
//! The **language half needs no new map at all**. Its tensors are the dense
//! Qwen3 set under llama.cpp's own spelling, so this module calls
//! `qwen3::gguf_import`'s map, which is transcribed from llama.cpp at a named
//! revision and gated bit-for-bit against the safetensors route. Re-deriving
//! it here is exactly the per-model duplication this crate is not going to
//! add: a swapped `k`/`v` is shape-compatible on every GQA layer and would not
//! be caught by loading.
//!
//! The **vision half's** map is below. It is checked, tensor by tensor, against
//! the released safetensors checkpoint by `tests/gguf_parity.rs`, which is the
//! same discipline for the same reason: the ViT's `qkv` is a single fused
//! `[3H, H]` matrix in both formats, so a wrong split or a transposed block
//! would load without complaint.
//!
//! | GGUF (mmproj) | HF (`Qwen3VLForConditionalGeneration`) |
//! |---|---|
//! | `v.patch_embd.weight` + `v.patch_embd.weight.1` | `model.visual.patch_embed.proj.weight` |
//! | `v.patch_embd.bias` | `model.visual.patch_embed.proj.bias` |
//! | `v.position_embd.weight` | `model.visual.pos_embed.weight` |
//! | `v.blk.N.ln1.{weight,bias}` | `…blocks.N.norm1.{weight,bias}` |
//! | `v.blk.N.ln2.{weight,bias}` | `…blocks.N.norm2.{weight,bias}` |
//! | `v.blk.N.attn_qkv.{weight,bias}` | `…blocks.N.attn.qkv.{weight,bias}` |
//! | `v.blk.N.attn_out.{weight,bias}` | `…blocks.N.attn.proj.{weight,bias}` |
//! | `v.blk.N.ffn_up.{weight,bias}` | `…blocks.N.mlp.linear_fc1.{weight,bias}` |
//! | `v.blk.N.ffn_down.{weight,bias}` | `…blocks.N.mlp.linear_fc2.{weight,bias}` |
//! | `v.post_ln.{weight,bias}` | `model.visual.merger.norm.{weight,bias}` |
//! | `mm.0.{weight,bias}` | `model.visual.merger.linear_fc1.{weight,bias}` |
//! | `mm.2.{weight,bias}` | `model.visual.merger.linear_fc2.{weight,bias}` |
//! | `v.deepstack.B.norm.{weight,bias}` | `…deepstack_merger_list.K.norm.{weight,bias}` |
//! | `v.deepstack.B.fc1.{weight,bias}` | `…deepstack_merger_list.K.linear_fc1.{weight,bias}` |
//! | `v.deepstack.B.fc2.{weight,bias}` | `…deepstack_merger_list.K.linear_fc2.{weight,bias}` |
//!
//! Three entries are not renames and are called out because they are where a
//! plausible-looking map would be wrong.
//!
//! The DeepStack mergers are indexed differently by the two formats, which is
//! the `B` versus `K` above. HF numbers them by TAP ORDINAL: the list is
//! `deepstack_merger_list.0/1/2`. GGUF numbers them by the ViT BLOCK they tap:
//! on the 4B, whose `deepstack_visual_indexes` are `[5, 11, 17]`, the tensors
//! are `v.deepstack.5/11/17`. Reading the ordinal as a block index (or the
//! reverse) finds no tensor on this checkpoint, but would silently pick the
//! wrong merger on any model whose taps started at 0, so the index comes from
//! the config's own tap list rather than from a counter.
//!
//! `v.post_ln` is the **merger's** input LayerNorm, not a trailing norm on the
//! ViT. Qwen3-VL's vision tower has no post-block norm of its own; llama.cpp
//! folds the merger's `norm` into the encoder's slot under that name. The
//! shapes decide it: the merger's norm is over the ViT width (1024 on the 4B),
//! the DeepStack mergers' norms are over width times merge-squared (4096), and
//! `v.post_ln` is the ViT width.
//!
//! `v.patch_embd` is a **3D convolution over two temporal slices**, and GGUF
//! carries it as two 4D tensors, the second suffixed `.1`. HF's contiguous
//! layout is `[out, in_channel, t, patch_h, patch_w]`, so the two slices
//! interleave per `(out, in_channel)` pair rather than concatenating. That is
//! the one place a shape check alone would pass while the weights were wrong:
//! both orderings hold the same element count.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use checkpoint::gguf::MmapGguf;
use data::qwen_tokenizer::QwenBpe;

use crate::config::{Qwen3VlConfig, VisionConfig};
use crate::import::ImportedWeights;

/// llama.cpp's `general.architecture` for the Qwen3-VL language half.
pub const GGUF_ARCHITECTURE: &str = "qwen3vl";
/// The `clip.projector_type` the matching `mmproj-*.gguf` declares.
pub const PROJECTOR_TYPE: &str = "qwen3vl_merger";

/// The two files a Qwen3-VL GGUF checkpoint is made of, both existence-checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufFiles {
    /// The language half.
    pub lm: PathBuf,
    /// The vision half (`mmproj-*.gguf`).
    pub mmproj: PathBuf,
}

impl GgufFiles {
    /// Resolve a Qwen3-VL GGUF checkpoint from a path that is either the
    /// language-half file itself or the directory holding it.
    ///
    /// Fails, naming the directory, when no projector sits beside the model:
    /// loading the decoder alone would produce captions that are fluent and
    /// unrelated to the image, which no downstream check can catch.
    pub fn locate(path: &Path) -> Result<GgufFiles, String> {
        let lm = if path.is_dir() { language_half_in(path)? } else { path.to_path_buf() };
        let route = route_of(&lm)?;
        if route.is_projector() {
            return Err(format!(
                "{}: this is the VISION half of a Qwen3-VL checkpoint (clip.projector_type {:?}), not the model. \
                 Point at the language-half GGUF instead; the projector is picked up from beside it.",
                lm.display(),
                route.projector.as_deref().unwrap_or("")
            ));
        }
        if route.tag != GGUF_ARCHITECTURE {
            return Err(format!(
                "{}: GGUF architecture {:?} is not {GGUF_ARCHITECTURE:?} (brain's {:?}); this file is a different model",
                lm.display(),
                route.tag,
                route.id()
            ));
        }
        let mmproj = gguf::route::sibling_projector(&lm).ok_or_else(|| {
            format!(
                "{}: no vision projector (mmproj) GGUF beside it. A Qwen3-VL GGUF release is TWO files: \
                 the language half and an `mmproj-*.gguf` vision tower. Without the projector the model \
                 cannot see the image at all, so this is refused rather than loaded. Fetch the mmproj from \
                 the same repo (`brain pull <repo-file-url>/mmproj-F16.gguf`) into {}.",
                lm.display(),
                lm.parent().unwrap_or(Path::new(".")).display()
            )
        })?;
        let proj_route = route_of(&mmproj)?;
        if proj_route.projector.as_deref() != Some(PROJECTOR_TYPE) {
            return Err(format!(
                "{}: clip.projector_type {:?}, expected {PROJECTOR_TYPE:?} -- this projector belongs to a different model",
                mmproj.display(),
                proj_route.projector.as_deref().unwrap_or("")
            ));
        }
        Ok(GgufFiles { lm, mmproj })
    }
}

fn route_of(p: &Path) -> Result<gguf::Route, String> {
    let s = p.to_str().ok_or_else(|| format!("{}: path is not valid UTF-8", p.display()))?;
    gguf::route_path(s)
}

/// The single language-half GGUF in `dir`.
///
/// Several `.gguf` files in one directory is the normal case (the projector,
/// and sometimes more than one quantization), so this reports what it saw
/// rather than picking arbitrarily: a deterministic choice is only defensible
/// where the alternatives are interchangeable, which two different
/// quantizations of the same model are not.
fn language_half_in(dir: &Path) -> Result<PathBuf, String> {
    let mut models: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .filter(|p| route_of(p).is_ok_and(|r| !r.is_projector()))
        .collect();
    models.sort();
    match models.len() {
        0 => Err(format!("{}: no language-half GGUF (only a projector, or no GGUF at all)", dir.display())),
        1 => Ok(models.remove(0)),
        _ => {
            let names: Vec<String> = models.iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned()).collect();
            Err(format!("{}: {} language-half GGUFs ({}); name the one to load", dir.display(), names.len(), names.join(", ")))
        }
    }
}

/// Derive the whole [`Qwen3VlConfig`] from the two files' own metadata.
///
/// The text half reuses `qwen3::gguf_import`'s reader against this
/// architecture's own KV prefix rather than restating llama.cpp's key names, so
/// a Qwen3-VL decoder and a plain Qwen3 decoder cannot disagree about what
/// `attention.key_length` means.
///
/// The four multimodal token ids have no GGUF KV of their own, so they are
/// resolved from the tokenizer by their literal content, which is where they
/// are ground truth. A vocabulary missing one is a loud error: an unresolved
/// `<|image_pad|>` BPEs into ordinary text and the image splice then lands on
/// text rows, producing a model that reads its prompt and ignores its input.
pub fn config(lm: &MmapGguf, mmproj: &MmapGguf, tok: &QwenBpe) -> Result<Qwen3VlConfig, String> {
    let kv = gguf::ArchKv::expect_architecture(lm, GGUF_ARCHITECTURE)?;
    let mut text = qwen3::gguf_import::config_from_kv(&kv, lm)?;
    // The decoder's run-parameter context, set by the caller at construction.
    text.block_size = 4096;

    let vision = vision_config(mmproj)?;
    let special = |name: &str| -> Result<u32, String> {
        tok.special_id(name).ok_or_else(|| format!("qwen3vl gguf: the embedded tokenizer has no reserved {name:?} token"))
    };
    Ok(Qwen3VlConfig {
        vision,
        text,
        mrope_section: mrope_section(&kv)?,
        image_token_id: special("<|image_pad|>")?,
        video_token_id: special("<|video_pad|>")?,
        vision_start_token_id: special("<|vision_start|>")?,
        vision_end_token_id: special("<|vision_end|>")?,
    })
}

/// M-RoPE's per-axis channel split, from `{arch}.rope.dimension_sections`.
///
/// llama.cpp declares four sections (the fourth is unused by this model and is
/// zero); brain's three axes are the first three.
fn mrope_section(kv: &gguf::ArchKv) -> Result<[u32; 3], String> {
    let v = kv.get("rope.dimension_sections").ok_or("qwen3vl gguf: missing rope.dimension_sections")?;
    let checkpoint::gguf::GgufValue::Array(items) = v else {
        return Err("qwen3vl gguf: rope.dimension_sections is not an array".to_string());
    };
    if items.len() < 3 {
        return Err(format!("qwen3vl gguf: rope.dimension_sections has {} entries, need at least 3", items.len()));
    }
    let mut out = [0u32; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = items[i].as_u64().ok_or("qwen3vl gguf: rope.dimension_sections entry is not an integer")? as u32;
    }
    Ok(out)
}

/// The ViT/merger config, from the projector's `clip.vision.*` KV plus the two
/// shapes the KV does not state.
fn vision_config(mmproj: &MmapGguf) -> Result<VisionConfig, String> {
    let kv = gguf::ArchKv::new(mmproj, "clip");
    let v = kv.scoped("vision");
    // Position-embedding count and input channels are tensor facts, and the
    // tensor is ground truth: an `image_size`-derived count that disagreed
    // with the table would build a model that cannot load its own weights.
    let pos = mmproj.shape("v.position_embd.weight").ok_or("qwen3vl gguf: mmproj has no v.position_embd.weight")?;
    let patch = mmproj.shape("v.patch_embd.weight").ok_or("qwen3vl gguf: mmproj has no v.patch_embd.weight")?;
    if patch.len() != 4 {
        return Err(format!("qwen3vl gguf: v.patch_embd.weight has shape {patch:?}, expected 4 dimensions"));
    }
    // GGUF splits the temporal axis of the patch-embed Conv3d into one tensor
    // per slice, so the count of slices IS the temporal patch size.
    let temporal_patch_size = (0..)
        .take_while(|t| mmproj.shape(&patch_embed_slice(*t)).is_some())
        .count() as u32;
    if temporal_patch_size == 0 {
        return Err("qwen3vl gguf: mmproj has no patch-embed slices".to_string());
    }
    Ok(VisionConfig {
        depth: v.req_u32("block_count")?,
        hidden: v.req_u32("embedding_length")?,
        num_heads: v.req_u32("attention.head_count")?,
        intermediate: v.req_u32("feed_forward_length")?,
        patch_size: v.req_u32("patch_size")?,
        temporal_patch_size,
        spatial_merge_size: v.req_u32("spatial_merge_size")?,
        num_position_embeddings: pos[0] as u32,
        out_hidden_size: v.req_u32("projection_dim")?,
        in_channels: patch[1] as u32,
        deepstack_indexes: deepstack_indexes(mmproj, &v)?,
        // Optional, same fallback and reasoning as `Qwen3VlConfig::from_hf`'s
        // own `tokens_per_second` field -- see that field's doc.
        tokens_per_second: v.u32_or("tokens_per_second", 2),
    })
}

/// Which ViT blocks feed a DeepStack merger, from
/// `clip.vision.is_deepstack_layers`, a per-block boolean mask.
///
/// The list is load-bearing twice over: it is the config field the model is
/// built from, AND it is how the merger tensors are addressed, since GGUF
/// names them by the block they tap.
fn deepstack_indexes(mmproj: &MmapGguf, v: &gguf::ArchKv) -> Result<Vec<u32>, String> {
    let value = v.get("is_deepstack_layers").ok_or("qwen3vl gguf: mmproj has no clip.vision.is_deepstack_layers")?;
    let checkpoint::gguf::GgufValue::Array(mask) = value else {
        return Err("qwen3vl gguf: clip.vision.is_deepstack_layers is not an array".to_string());
    };
    let idx: Vec<u32> = mask
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b, checkpoint::gguf::GgufValue::Bool(true)))
        .map(|(i, _)| i as u32)
        .collect();
    // The mask and the tensors are two statements of the same fact, and a
    // model built from one while loading the other is the failure worth
    // catching here rather than at the first merge.
    for b in &idx {
        if mmproj.shape(&format!("v.deepstack.{b}.fc1.weight")).is_none() {
            return Err(format!("qwen3vl gguf: is_deepstack_layers marks ViT block {b} but the mmproj has no v.deepstack.{b}.* merger"));
        }
    }
    let stray = mmproj.names().iter().filter(|n| n.starts_with("v.deepstack.") && n.ends_with(".fc1.weight")).count();
    if stray != idx.len() {
        return Err(format!("qwen3vl gguf: is_deepstack_layers marks {} blocks but the mmproj carries {stray} DeepStack mergers", idx.len()));
    }
    Ok(idx)
}

fn patch_embed_slice(t: usize) -> String {
    if t == 0 {
        "v.patch_embd.weight".to_string()
    } else {
        format!("v.patch_embd.weight.{t}")
    }
}

/// One GGUF tensor, dequantized, or a message naming what is missing.
fn tensor(mg: &MmapGguf, name: &str) -> Result<Vec<f32>, String> {
    mg.tensor(name).ok_or_else(|| format!("qwen3vl gguf: missing tensor {name:?}"))?
}

/// Read both halves into the four brain weight sets `Qwen3Vl` consumes.
///
/// Streams one tensor at a time out of each mapping, the same as the
/// safetensors route: peak host memory is the assembled model plus one
/// tensor's fp32 expansion, never a whole-file dequantization.
pub fn weights(files: &GgufFiles, cfg: &Qwen3VlConfig) -> Result<ImportedWeights, String> {
    let lm = MmapGguf::open(files.lm.to_str().ok_or("qwen3vl gguf: non-UTF8 lm path")?)?;
    let mmproj = MmapGguf::open(files.mmproj.to_str().ok_or("qwen3vl gguf: non-UTF8 mmproj path")?)?;
    let v = &cfg.vision;

    let mut vision: HashMap<String, Vec<f32>> = HashMap::new();
    // Patch embed: interleave the temporal slices into HF's
    // [out, in_channel, t, patch_h, patch_w] contiguous order.
    let plane = (v.patch_size * v.patch_size) as usize;
    let per_slice = (v.in_channels as usize) * plane;
    let t_max = v.temporal_patch_size as usize;
    let slices: Vec<Vec<f32>> = (0..t_max).map(|t| tensor(&mmproj, &patch_embed_slice(t))).collect::<Result<_, _>>()?;
    let out_dim = v.hidden as usize;
    let mut pe = vec![0f32; out_dim * per_slice * t_max];
    for o in 0..out_dim {
        for c in 0..v.in_channels as usize {
            for (t, slice) in slices.iter().enumerate() {
                let src = (o * v.in_channels as usize + c) * plane;
                let dst = ((o * v.in_channels as usize + c) * t_max + t) * plane;
                pe[dst..dst + plane].copy_from_slice(&slice[src..src + plane]);
            }
        }
    }
    vision.insert("patch_embed.weight".into(), pe);
    vision.insert("patch_embed.bias".into(), tensor(&mmproj, "v.patch_embd.bias")?);
    vision.insert("pos_embed".into(), tensor(&mmproj, "v.position_embd.weight")?);
    for n in 0..v.depth {
        for (gguf_leaf, brain_leaf) in VISION_BLOCK_LEAVES {
            vision.insert(format!("blocks.{n}.{brain_leaf}"), tensor(&mmproj, &format!("v.blk.{n}.{gguf_leaf}"))?);
        }
    }

    let mut main_merger: HashMap<String, Vec<f32>> = HashMap::new();
    for (gguf_name, brain_leaf) in MERGER_LEAVES {
        main_merger.insert((*brain_leaf).to_string(), tensor(&mmproj, gguf_name)?);
    }

    // GGUF names each DeepStack merger by the ViT BLOCK it taps, HF by its
    // ordinal in the list, so the tap list is what converts between them.
    let mut deepstack: Vec<HashMap<String, Vec<f32>>> = Vec::with_capacity(v.deepstack_indexes.len());
    for block in &v.deepstack_indexes {
        let mut m = HashMap::new();
        for (gguf_leaf, brain_leaf) in DEEPSTACK_LEAVES {
            m.insert((*brain_leaf).to_string(), tensor(&mmproj, &format!("v.deepstack.{block}.{gguf_leaf}"))?);
        }
        deepstack.push(m);
    }

    // The language half is a dense Qwen3 under llama.cpp's own spelling: reuse
    // that crate's gated map rather than restating it.
    let mut decoder: HashMap<String, Vec<f32>> = HashMap::new();
    for name in lm.names() {
        if let Some(brain) = qwen3::gguf_import::gguf_to_brain(name, cfg.text.tie_embeddings) {
            decoder.insert(brain, tensor(&lm, name)?);
        }
    }

    Ok(ImportedWeights { vision, main_merger, deepstack, decoder })
}

/// `(GGUF leaf, brain leaf)` for one ViT block. Bias and weight are listed
/// separately rather than derived, so a checkpoint missing one of a pair is an
/// error naming the tensor.
const VISION_BLOCK_LEAVES: &[(&str, &str)] = &[
    ("ln1.weight", "norm1.weight"),
    ("ln1.bias", "norm1.bias"),
    ("ln2.weight", "norm2.weight"),
    ("ln2.bias", "norm2.bias"),
    ("attn_qkv.weight", "qkv.weight"),
    ("attn_qkv.bias", "qkv.bias"),
    ("attn_out.weight", "proj.weight"),
    ("attn_out.bias", "proj.bias"),
    ("ffn_up.weight", "fc1.weight"),
    ("ffn_up.bias", "fc1.bias"),
    ("ffn_down.weight", "fc2.weight"),
    ("ffn_down.bias", "fc2.bias"),
];

/// `(GGUF name, brain leaf)` for the main PatchMerger. `v.post_ln` is the
/// merger's own input norm, not a ViT post-norm - see this module's doc.
const MERGER_LEAVES: &[(&str, &str)] = &[
    ("v.post_ln.weight", "ln.weight"),
    ("v.post_ln.bias", "ln.bias"),
    ("mm.0.weight", "fc1.weight"),
    ("mm.0.bias", "fc1.bias"),
    ("mm.2.weight", "fc2.weight"),
    ("mm.2.bias", "fc2.bias"),
];

/// `(GGUF leaf, brain leaf)` for one DeepStack merger.
const DEEPSTACK_LEAVES: &[(&str, &str)] = &[
    ("norm.weight", "ln.weight"),
    ("norm.bias", "ln.bias"),
    ("fc1.weight", "fc1.weight"),
    ("fc1.bias", "fc1.bias"),
    ("fc2.weight", "fc2.weight"),
    ("fc2.bias", "fc2.bias"),
];

/// The tokenizer the language half carries in its own `tokenizer.ggml.*` KV.
pub fn tokenizer(files: &GgufFiles) -> Result<QwenBpe, String> {
    let lm = MmapGguf::open(files.lm.to_str().ok_or("qwen3vl gguf: non-UTF8 lm path")?)?;
    let gt = lm.tokenizer().ok_or_else(|| format!("{}: no embedded tokenizer.ggml.* KV", files.lm.display()))?;
    QwenBpe::from_gguf(&gt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_leaf_tables_cover_every_tensor_of_their_stage() {
        // A ViT block is 4 norm params, a fused qkv (weight+bias), an output
        // projection (weight+bias) and two MLP linears (weight+bias).
        assert_eq!(VISION_BLOCK_LEAVES.len(), 12);
        assert_eq!(MERGER_LEAVES.len(), 6);
        assert_eq!(DEEPSTACK_LEAVES.len(), 6);
        // Every brain leaf a table produces is distinct: a duplicate would
        // silently overwrite one tensor with another of the same shape.
        for table in [VISION_BLOCK_LEAVES, MERGER_LEAVES, DEEPSTACK_LEAVES] {
            let mut brain: Vec<&str> = table.iter().map(|(_, b)| *b).collect();
            brain.sort_unstable();
            let n = brain.len();
            brain.dedup();
            assert_eq!(brain.len(), n, "duplicate brain leaf in a name table");
        }
    }

    #[test]
    fn patch_embed_slices_are_named_the_way_llama_cpp_splits_a_conv3d() {
        assert_eq!(patch_embed_slice(0), "v.patch_embd.weight");
        assert_eq!(patch_embed_slice(1), "v.patch_embd.weight.1");
    }
}
