// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The upstream `transformers` checkpoint** - `deepseek-ai/DeepSeek-OCR`'s
//! own safetensors release read into the same two halves [`crate::import`]
//! produces from the pre-converted `ggml-org/DeepSeek-OCR-GGUF` pair.
//!
//! ## Why this is a rename and nothing more
//!
//! The GGUF release is a conversion OF this checkpoint, so the two carry the
//! same 2710 tensors: 476 in the vision half (SAM tower + CLIP tower +
//! projector + the two learned image-block rows) and 2234 in the decoder.
//! [`classify`] is therefore the exact analogue of
//! `gguf::deepseek_ocr_vision::classify` + `gguf::deepseek_ocr::classify` -
//! a total map onto the same brain-side names, with no transpose, no slice
//! and no fusion, because a safetensors tensor and a GGUF tensor of the same
//! logical shape have the same flat element order (GGUF's `ne` is torch's
//! shape reversed, which leaves the row-major buffer identical).
//!
//! ## Derived from the checkpoint, then refused unless it is the preset
//!
//! [`config_from_shapes`] reads the tower geometry off the checkpoint's own
//! tensor SHAPES wherever a shape can carry it - which is nearly everywhere,
//! and matters because this repo's `config.json` cannot be trusted for it:
//! its `vision_config.mlp_ratio` is 3.7362, while both towers' real feed-forward
//! widths are 4x their model width. On the vision side only CLIP's
//! head count comes from `config.json`, because no tensor shape carries it
//! (its qkv projection is `[3*width, width]` whatever the head count is).
//! The result is then compared against
//! [`DeepseekOcrConfig::deepseek_ocr`] and anything else is refused, exactly
//! as [`crate::import::config`] does for the GGUF pair.
//!
//! The three brain-side constants that are not checkpoint facts at all
//! (the SAM tower's LayerNorm epsilon, its attention chunk, and CLIP's
//! epsilon) are not decided here: this module builds a
//! [`gguf::deepseek_ocr_vision::DeepseekOcrVisionConfig`] and hands it to the
//! SAME `SamViTConfig::from` / `ClipVisionConfig::from_gguf` adapters the
//! GGUF path uses, so there is one answer to "what epsilon does this tower
//! run at", not two.
//!
//! Swedish Embedded AB implements checkpoint-format importers like this one -
//! deriving a model's shape from the weights themselves and refusing anything
//! that does not match a known-good preset - for clients porting published
//! models onto their own inference stacks. If your team needs the same
//! discipline, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::weightio::WeightReader;

use clip::config::ClipVisionConfig;
use deepseek2::DeepseekV2Config;
// The decoder's own checkpoint shape. Aliased because this crate's composite
// config carries the same type NAME for a different thing.
use gguf::deepseek_ocr::DeepseekOcrConfig as DecoderShape;
use gguf::deepseek_ocr_vision::{ClipConfig, DeepseekOcrVisionConfig, SamConfig};
use sam1::SamViTConfig;
use serde_json::Value;

use crate::config::{DeepseekOcrConfig, IMAGE_NEWLINE, PROJECTOR_B, PROJECTOR_W, VIEW_SEPARATOR};

/// Which half of the composite one upstream tensor belongs to.
///
/// The two halves are separate [`checkpoint::TensorSource`]s at load time and
/// their name spaces genuinely overlap (both spell a stack `blocks.{i}.…`),
/// so the half is part of the answer, not something a caller may infer from
/// the name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Half {
    /// A name in [`crate::import::encoder_weights`]' map: `vision.sam.*`,
    /// CLIP's bare leaves, `projector.*`, and the two `vision.*` glue rows.
    Encoder(String),
    /// A name in the decoder's own `param_list`.
    Decoder(String),
    /// ONE expert's slice of a decoder MoE expert bank. Upstream ships each
    /// routed expert as its own `[out, in]` tensor; brain's decoder wants the
    /// three `blocks.{l}.mlp.experts.{gate,up,down}.weight` banks the GGUF
    /// release already stores fused, so `expert` says where in the bank this
    /// source tensor belongs and [`plans`] concatenates them in that order.
    DecoderExpert { bank: String, expert: u32 },
}

/// Shorthand for the one error shape this module reports.
fn unknown(full: &str) -> String {
    format!("{LABEL}: unrecognized tensor {full:?}")
}

const LABEL: &str = "deepseek-ocr hf import";

/// Split `"{index}.{leaf}"`, bounds-checking the index, so a checkpoint that
/// grew a block (or an expert) stops the import instead of landing silently
/// outside the parameter list.
fn split_indexed<'a>(rest: &'a str, full: &str, bound: u32, what: &str) -> Result<(u32, &'a str), String> {
    let (idx, leaf) = rest.split_once('.').ok_or_else(|| format!("{LABEL}: malformed {what} name {full:?}"))?;
    let i: u32 = idx.parse().map_err(|_| format!("{LABEL}: malformed {what} index in {full:?}"))?;
    if i >= bound {
        return Err(format!("{LABEL}: {full}: {what} index {i} beyond {bound}"));
    }
    Ok((i, leaf))
}

/// The brain-side home of one upstream HF tensor name.
pub fn classify(name: &str, cfg: &DeepseekOcrConfig) -> Result<Half, String> {
    match name {
        "model.projector.layers.weight" => return Ok(Half::Encoder(PROJECTOR_W.to_string())),
        "model.projector.layers.bias" => return Ok(Half::Encoder(PROJECTOR_B.to_string())),
        "model.image_newline" => return Ok(Half::Encoder(IMAGE_NEWLINE.to_string())),
        // `seperator` is the upstream spelling; brain's is spelled correctly.
        "model.view_seperator" => return Ok(Half::Encoder(VIEW_SEPARATOR.to_string())),
        "model.embed_tokens.weight" => return Ok(Half::Decoder("tok.weight".to_string())),
        "model.norm.weight" => return Ok(Half::Decoder("norm.weight".to_string())),
        "lm_head.weight" => return Ok(Half::Decoder("lm_head.weight".to_string())),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.sam_model.") {
        return sam_leaf(rest, name, cfg).map(Half::Encoder);
    }
    if let Some(rest) = name.strip_prefix("model.vision_model.") {
        return clip_leaf(rest, name, cfg).map(Half::Encoder);
    }
    if let Some(rest) = name.strip_prefix("model.layers.") {
        return decoder_leaf(rest, name, cfg);
    }
    Err(unknown(name))
}

/// The SAM ViT-B tower. Keeps its `vision.sam.` prefix, as the GGUF path's own
/// names do.
fn sam_leaf(rest: &str, full: &str, cfg: &DeepseekOcrConfig) -> Result<String, String> {
    let p = |s: &str| Ok(format!("vision.sam.{s}"));
    match rest {
        "pos_embed" => return p("pos_embed"),
        "patch_embed.proj.weight" => return p("patch_embed.weight"),
        "patch_embed.proj.bias" => return p("patch_embed.bias"),
        // `neck` is an `nn.Sequential(conv, LayerNorm2d, conv, LayerNorm2d)`,
        // so upstream indexes it positionally and brain names the four parts.
        "neck.0.weight" => return p("neck.conv1.weight"),
        "neck.1.weight" => return p("neck.norm1.weight"),
        "neck.1.bias" => return p("neck.norm1.bias"),
        "neck.2.weight" => return p("neck.conv2.weight"),
        "neck.3.weight" => return p("neck.norm2.weight"),
        "neck.3.bias" => return p("neck.norm2.bias"),
        // The two stride-2 compressor convs.
        "net_2.weight" => return p("compress.conv1.weight"),
        "net_3.weight" => return p("compress.conv2.weight"),
        _ => {}
    }
    let rest = rest.strip_prefix("blocks.").ok_or_else(|| unknown(full))?;
    let (l, leaf) = split_indexed(rest, full, cfg.sam.n_layers, "SAM block")?;
    let b = |s: &str| Ok(format!("vision.sam.blocks.{l}.{s}"));
    match leaf {
        "norm1.weight" | "norm1.bias" | "norm2.weight" | "norm2.bias" => b(leaf),
        "attn.qkv.weight" | "attn.qkv.bias" | "attn.rel_pos_h" | "attn.rel_pos_w" => b(leaf),
        "attn.proj.weight" | "attn.proj.bias" => b(leaf),
        "mlp.lin1.weight" => b("mlp.fc1.weight"),
        "mlp.lin1.bias" => b("mlp.fc1.bias"),
        "mlp.lin2.weight" => b("mlp.fc2.weight"),
        "mlp.lin2.bias" => b("mlp.fc2.bias"),
        _ => Err(unknown(full)),
    }
}

/// The CLIP-L/14 tower. Emitted as BARE leaves - the name space
/// `clip::ClipVisionConfig::tensor_manifest` declares and
/// [`crate::import::encoder_weights`] already produces by stripping the GGUF
/// path's `vision.clip.` prefix.
fn clip_leaf(rest: &str, full: &str, cfg: &DeepseekOcrConfig) -> Result<String, String> {
    match rest {
        "embeddings.class_embedding" => return Ok("class_embed".to_string()),
        "embeddings.patch_embedding.weight" => return Ok("patch_embed.weight".to_string()),
        "embeddings.position_embedding.weight" => return Ok("pos_embed".to_string()),
        // `pre_layrnorm` is upstream's own typo, carried by the real tensor.
        "pre_layrnorm.weight" => return Ok("pre_norm.weight".to_string()),
        "pre_layrnorm.bias" => return Ok("pre_norm.bias".to_string()),
        _ => {}
    }
    let rest = rest.strip_prefix("transformer.layers.").ok_or_else(|| unknown(full))?;
    let (l, leaf) = split_indexed(rest, full, cfg.clip.layers(), "CLIP block")?;
    let b = |s: &str| Ok(format!("blocks.{l}.{s}"));
    match leaf {
        "layer_norm1.weight" => b("norm1.weight"),
        "layer_norm1.bias" => b("norm1.bias"),
        "layer_norm2.weight" => b("norm2.weight"),
        "layer_norm2.bias" => b("norm2.bias"),
        // Already fused upstream, exactly as brain wants it - no q/k/v concat.
        "self_attn.qkv_proj.weight" => b("attn.qkv.weight"),
        "self_attn.qkv_proj.bias" => b("attn.qkv.bias"),
        "self_attn.out_proj.weight" => b("attn.proj.weight"),
        "self_attn.out_proj.bias" => b("attn.proj.bias"),
        "mlp.fc1.weight" | "mlp.fc1.bias" | "mlp.fc2.weight" | "mlp.fc2.bias" => b(leaf),
        _ => Err(unknown(full)),
    }
}

/// The DeepSeek-V2 decoder stack.
fn decoder_leaf(rest: &str, full: &str, cfg: &DeepseekOcrConfig) -> Result<Half, String> {
    let (l, leaf) = split_indexed(rest, full, cfg.decoder.n_layers(), "decoder layer")?;
    let p = |s: &str| Ok(Half::Decoder(format!("blocks.{l}.{s}")));
    match leaf {
        "input_layernorm.weight" => return p("ln1.weight"),
        "post_attention_layernorm.weight" => return p("ln2.weight"),
        "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" | "self_attn.o_proj.weight" => return p(leaf),
        _ => {}
    }
    let mlp = leaf.strip_prefix("mlp.").ok_or_else(|| unknown(full))?;
    // Which MLP a layer carries is a fact of the config (`first_k_dense_replace`),
    // so a router tensor on a dense layer is an error rather than a name brain
    // happens not to have.
    if !cfg.decoder.shape.is_moe_layer(l) {
        return match mlp {
            "gate_proj.weight" => p("mlp.gate.weight"),
            "up_proj.weight" => p("mlp.up.weight"),
            "down_proj.weight" => p("mlp.down.weight"),
            _ => Err(unknown(full)),
        };
    }
    match mlp {
        // NOT `mlp.gate_proj.weight`: on a MoE layer `mlp.gate` is the router.
        "gate.weight" => return p("mlp.router.weight"),
        "shared_experts.gate_proj.weight" => return p("mlp.shared.gate.weight"),
        "shared_experts.up_proj.weight" => return p("mlp.shared.up.weight"),
        "shared_experts.down_proj.weight" => return p("mlp.shared.down.weight"),
        _ => {}
    }
    let rest = mlp.strip_prefix("experts.").ok_or_else(|| unknown(full))?;
    let (e, eleaf) = split_indexed(rest, full, cfg.decoder.shape.n_experts, "expert")?;
    let which = match eleaf {
        "gate_proj.weight" => "gate",
        "up_proj.weight" => "up",
        "down_proj.weight" => "down",
        _ => return Err(unknown(full)),
    };
    Ok(Half::DecoderExpert { bank: format!("blocks.{l}.mlp.experts.{which}.weight"), expert: e })
}

/// The encoder half's manifest: every name [`classify`] may return as
/// [`Half::Encoder`], with its element count.
///
/// Composed from the three manifests that already exist rather than
/// re-declared, so this cannot drift from what `DeepEncoder::new` reads -
/// that constructor's own doc names these same three lists.
pub fn encoder_manifest(cfg: &DeepseekOcrConfig) -> Vec<(String, usize)> {
    let mut v = cfg.sam.param_list();
    v.extend(cfg.clip.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product::<usize>())));
    v.extend(cfg.glue_param_list());
    v
}

/// Split a checkpoint's tensor names into the two halves' fetch plans, keyed
/// by the brain-side name.
///
/// One pass over the SOURCE names (rather than looking each manifest entry up
/// by a hand-written inverse map) is what makes the coverage check meaningful
/// in both directions: an upstream tensor this crate does not understand is an
/// error here, and a manifest entry no upstream tensor produced is caught by
/// the per-half `validate` below.
///
/// `Fetch::Whole` for everything except the decoder's MoE expert banks: the
/// upstream CLIP tower already ships its qkv fused the way brain wants it, so
/// nothing is sliced, but upstream DOES ship each routed expert separately
/// while brain's decoder reads one `[n_experts, out, in]` bank per projection
/// (the layout the GGUF release stores natively), so those are a
/// `Fetch::Concat` of the experts in index order.
pub fn plans(names: &[String], cfg: &DeepseekOcrConfig) -> Result<(HashMap<String, Fetch>, HashMap<String, Fetch>), String> {
    let (mut enc, mut dec) = (HashMap::new(), HashMap::new());
    // bank name -> expert index -> source tensor. `BTreeMap` on the inner key
    // is what makes the concatenation expert-ordered by construction rather
    // than by a sort someone could later drop.
    let mut banks: BTreeMap<String, BTreeMap<u32, String>> = BTreeMap::new();
    for src in names {
        let (side, brain) = match classify(src, cfg)? {
            Half::Encoder(n) => (&mut enc, n),
            Half::Decoder(n) => (&mut dec, n),
            Half::DecoderExpert { bank, expert } => {
                if let Some(prev) = banks.entry(bank.clone()).or_default().insert(expert, src.clone()) {
                    return Err(format!("{LABEL}: two source tensors ({prev}, {src}) are expert {expert} of {bank}"));
                }
                continue;
            }
        };
        if side.insert(brain.clone(), Fetch::Whole(src.clone())).is_some() {
            return Err(format!("{LABEL}: two source tensors map to {brain}"));
        }
    }
    let n_experts = cfg.decoder.shape.n_experts;
    for (bank, parts) in banks {
        let missing: Vec<u32> = (0..n_experts).filter(|e| !parts.contains_key(e)).collect();
        if !missing.is_empty() {
            // Name each absent expert with its own per-expert leaf
            // (`...experts.{e}.gate.weight`), so the message points at the one
            // source tensor that is missing rather than only at the bank it
            // would have joined.
            let (head, tail) = bank.split_once("experts.").unwrap_or((bank.as_str(), ""));
            let named: Vec<String> = missing.iter().map(|e| format!("{head}experts.{e}.{tail}")).collect();
            return Err(format!(
                "{LABEL}: {bank}: this checkpoint produces no tensor for {} of {n_experts} experts: {named:?}",
                missing.len()
            ));
        }
        let fetch = Fetch::Concat(parts.into_values().map(Fetch::Whole).collect());
        if dec.insert(bank.clone(), fetch).is_some() {
            return Err(format!("{LABEL}: two source tensors map to {bank}"));
        }
    }
    for (label, plan, want) in [("encoder", &enc, encoder_manifest(cfg)), ("decoder", &dec, cfg.decoder.param_list())] {
        let missing: Vec<&str> = want.iter().map(|(n, _)| n.as_str()).filter(|n| !plan.contains_key(*n)).collect();
        if !missing.is_empty() {
            return Err(format!("{LABEL}: this checkpoint produces no {label} tensor for {} parameter(s): {missing:?}", missing.len()));
        }
    }
    Ok((enc, dec))
}

/// The encoder half's weights, read out of an open checkpoint under brain's
/// own names.
///
/// Eager, unlike the decoder: `DeepEncoder::new` uploads the whole tower at
/// once and the vision half is ~0.5 GB of the checkpoint, so this matches what
/// `crate::import::encoder_weights` already materializes from the mmproj.
pub fn encoder_weights(reader: &WeightReader, cfg: &DeepseekOcrConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    let (enc, _) = plans(&names, cfg)?;
    let mut out = HashMap::with_capacity(enc.len());
    for (brain, fetch) in enc {
        let Fetch::Whole(src) = fetch else { unreachable!("plans yields only whole tensors") };
        let data = reader.tensor(&src).ok_or_else(|| format!("{LABEL}: {src}: could not be read"))?;
        out.insert(brain, data);
    }
    Ok(out)
}

/// The decoder half as a streaming [`checkpoint::TensorSource`] under brain's
/// own names, borrowing `reader`.
///
/// Nothing is expanded to disk: each tensor is decoded from the checkpoint's
/// own BF16 on demand, so this path never builds the ~12 GB fp32 intermediate
/// the GGUF pair needs (`crate::import::expand_lm`).
pub fn decoder_source<'a>(reader: &'a WeightReader, cfg: &DeepseekOcrConfig) -> Result<RemapSource<'a>, String> {
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    let (_, dec) = plans(&names, cfg)?;
    let src = RemapSource::new(reader, dec);
    src.validate(&cfg.decoder.param_list())?;
    Ok(src)
}

/// Every tensor's shape, header-only - what [`config_from_shapes`] reads.
pub fn shapes(reader: &WeightReader) -> BTreeMap<String, Vec<usize>> {
    reader
        .names()
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|n| reader.shape(&n).map(|s| (n.clone(), s.iter().map(|d| *d as usize).collect())))
        .collect()
}

/// One tensor's extent along `axis`, or a message naming the tensor.
fn dim(shapes: &BTreeMap<String, Vec<usize>>, name: &str, axis: usize) -> Result<u32, String> {
    let s = shapes.get(name).ok_or_else(|| format!("{LABEL}: missing {name}"))?;
    let d = s.get(axis).ok_or_else(|| format!("{LABEL}: {name} has no axis {axis} (shape {s:?})"))?;
    Ok(*d as u32)
}

/// How many `{prefix}{i}.` stacks the checkpoint declares - one past the
/// highest index present, so a gap is caught by the coverage check rather
/// than silently shortening the stack.
fn stack_len(shapes: &BTreeMap<String, Vec<usize>>, prefix: &str) -> u32 {
    shapes
        .keys()
        .filter_map(|k| k.strip_prefix(prefix))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|idx| idx.parse::<u32>().ok())
        .max()
        .map_or(0, |m| m + 1)
}

/// A `config.json` number reached by a path of object keys.
fn cfg_u32(config: &Value, path: &[&str]) -> Result<u32, String> {
    let mut v = config;
    for k in path {
        v = v.get(k).ok_or_else(|| format!("{LABEL}: config.json has no {}", path.join(".")))?;
    }
    v.as_u64().map(|n| n as u32).ok_or_else(|| format!("{LABEL}: config.json {} is not an integer", path.join(".")))
}

/// Same, for a float with a documented upstream default when the field is
/// absent (this repo's `config.json` omits both of them).
fn cfg_f32(config: &Value, path: &[&str], default: f32) -> f32 {
    let mut v = config;
    for k in path {
        match v.get(k) {
            Some(next) => v = next,
            None => return default,
        }
    }
    v.as_f64().map_or(default, |n| n as f32)
}

/// Refuse when a number the checkpoint states twice disagrees with itself.
fn agree(what: &str, derived: u32, declared: u32) -> Result<(), String> {
    if derived != declared {
        return Err(format!("{LABEL}: {what}: the tensor shapes say {derived}, config.json says {declared}"));
    }
    Ok(())
}

/// The vision tower's shape, read off the checkpoint's own tensors.
///
/// Only ONE number here comes from `config.json`: CLIP's head count, which no
/// tensor shape carries (its qkv projection is `[3*width, width]` at any head
/// count). SAM's head count IS carried - by its relative-position tables'
/// second axis, which is the head dimension - and is cross-checked against
/// `config.json` rather than taken from it.
fn vision_from_shapes(shapes: &BTreeMap<String, Vec<usize>>, config: &Value) -> Result<DeepseekOcrVisionConfig, String> {
    let d_model = dim(shapes, "model.sam_model.patch_embed.proj.weight", 0)?;
    let patch_size = dim(shapes, "model.sam_model.patch_embed.proj.weight", 2)?;
    let (grid, grid_w) = (dim(shapes, "model.sam_model.pos_embed", 1)?, dim(shapes, "model.sam_model.pos_embed", 2)?);
    if grid != grid_w {
        return Err(format!("{LABEL}: SAM position grid is {grid}x{grid_w}; this tower is square"));
    }
    let n_layers = stack_len(shapes, "model.sam_model.blocks.");
    let ffn_hidden = dim(shapes, "model.sam_model.blocks.0.mlp.lin1.weight", 0)?;
    let head_dim = dim(shapes, "model.sam_model.blocks.0.attn.rel_pos_h", 1)?;
    if head_dim == 0 || d_model % head_dim != 0 {
        return Err(format!("{LABEL}: SAM width {d_model} is not a multiple of its head dim {head_dim}"));
    }
    let n_heads = d_model / head_dim;
    agree("SAM head count", n_heads, cfg_u32(config, &["vision_config", "width", "sam_vit_b", "heads"])?)?;
    agree("SAM depth", n_layers, cfg_u32(config, &["vision_config", "width", "sam_vit_b", "layers"])?)?;

    // Which blocks attend globally is carried by the relative-position tables
    // themselves: a global block spans the whole grid, a windowed one spans
    // its window. Both are `2 * extent - 1` rows.
    let (mut global_attn_layers, mut window) = (Vec::new(), None);
    for l in 0..n_layers {
        let rows = dim(shapes, &format!("model.sam_model.blocks.{l}.attn.rel_pos_h"), 0)?;
        if rows == 2 * grid - 1 {
            global_attn_layers.push(l);
            continue;
        }
        if rows % 2 == 0 {
            return Err(format!("{LABEL}: SAM block {l} has {rows} relative-position rows, which is not 2*extent-1"));
        }
        let w = rows.div_ceil(2);
        match window {
            None => window = Some(w),
            Some(prev) if prev != w => return Err(format!("{LABEL}: SAM blocks disagree about the attention window ({prev} vs {w})")),
            Some(_) => {}
        }
    }
    let window_size = window.ok_or_else(|| format!("{LABEL}: every SAM block attends globally; this tower has a windowed majority"))?;

    let clip_d = dim(shapes, "model.vision_model.embeddings.patch_embedding.weight", 0)?;
    let clip_patch = dim(shapes, "model.vision_model.embeddings.patch_embedding.weight", 2)?;
    let clip_layers = stack_len(shapes, "model.vision_model.transformer.layers.");
    let n_positions = dim(shapes, "model.vision_model.embeddings.position_embedding.weight", 0)?;
    // The native image size is the position table's own: one class token plus
    // a square patch grid. Derived rather than read, then cross-checked.
    let patches = n_positions.checked_sub(1).ok_or_else(|| format!("{LABEL}: CLIP position table is empty"))?;
    let side = (patches as f64).sqrt().round() as u32;
    if side * side != patches {
        return Err(format!("{LABEL}: CLIP has {patches} patch positions, which is not a square grid"));
    }
    let image_size = side * clip_patch;
    agree("CLIP native image size", image_size, cfg_u32(config, &["vision_config", "width", "clip-l-14-224", "image_size"])?)?;
    agree("CLIP depth", clip_layers, cfg_u32(config, &["vision_config", "width", "clip-l-14-224", "layers"])?)?;

    Ok(DeepseekOcrVisionConfig {
        sam: SamConfig {
            d_model,
            n_layers,
            n_heads,
            ffn_hidden,
            patch_size,
            grid,
            window_size,
            global_attn_layers,
            neck_channels: dim(shapes, "model.sam_model.neck.0.weight", 0)?,
            compress_mid: dim(shapes, "model.sam_model.net_2.weight", 0)?,
            compress_out: dim(shapes, "model.sam_model.net_3.weight", 0)?,
        },
        clip: ClipConfig {
            d_model: clip_d,
            n_layers: clip_layers,
            // The one number no tensor shape carries.
            n_heads: cfg_u32(config, &["vision_config", "width", "clip-l-14-224", "heads"])?,
            ffn_hidden: dim(shapes, "model.vision_model.transformer.layers.0.mlp.fc1.weight", 0)?,
            patch_size: clip_patch,
            image_size,
            n_positions,
            // Overwritten by `ClipVisionConfig::from_gguf`, which owns this
            // decision for both import paths; see that function's doc.
            layer_norm_eps: clip::config::DEEPSEEK_OCR_CLIP_EPS,
        },
        projector_in: dim(shapes, "model.projector.layers.weight", 1)?,
        projection_dim: dim(shapes, "model.projector.layers.weight", 0)?,
        // Inert here: the composite config carries neither, and
        // `crate::preprocess` owns the normalization constants. Kept at the
        // shipped values so this struct never claims something untrue.
        image_mean: vec![0.5; 3],
        image_std: vec![0.5; 3],
        // Selects the exact-erf form in `ClipVisionConfig::from_gguf`, the
        // same activation the shipped mmproj's `clip.use_gelu = true`
        // selects. Both import paths must agree, so this tracks that flag
        // rather than making a second, independent choice -- including where
        // that flag is WRONG: this checkpoint's own `deepencoder.py` runs
        // this tower on quick-GELU, which `ClipVisionConfig::deepseek_ocr`'s
        // docs now record. Fixing it there fixes it here; diverging here
        // would only make the two paths disagree.
        use_gelu: true,
        scale_factor: 1,
    })
}

/// The decoder's shape. Everything the tensors can carry is read off them;
/// the head counts, the router policy numbers and the two RoPE/norm constants
/// come from `config.json` (or, where this repo omits them, from the
/// documented `DeepseekV2Config` defaults its own
/// `configuration_deepseek_v2.py` declares).
fn decoder_from_shapes(shapes: &BTreeMap<String, Vec<usize>>, config: &Value, block_size: u32) -> Result<DeepseekV2Config, String> {
    let lang = if config.get("language_config").is_some() { &["language_config"][..] } else { &[][..] };
    let path = |k: &'static str| -> Vec<&str> { lang.iter().copied().chain(std::iter::once(k)).collect() };
    let get = |k: &'static str| cfg_u32(config, &path(k));

    let vocab = dim(shapes, "model.embed_tokens.weight", 0)?;
    let d_model = dim(shapes, "model.embed_tokens.weight", 1)?;
    let n_layers = stack_len(shapes, "model.layers.");
    agree("decoder width", d_model, get("hidden_size")?)?;
    agree("vocabulary", vocab, get("vocab_size")?)?;
    agree("decoder depth", n_layers, get("num_hidden_layers")?)?;

    let n_heads = get("num_attention_heads")?;
    let n_kv_heads = get("num_key_value_heads")?;
    let q_dim = dim(shapes, "model.layers.0.self_attn.q_proj.weight", 0)?;
    if n_heads == 0 || q_dim % n_heads != 0 {
        return Err(format!("{LABEL}: q projection is {q_dim} wide, which is not a multiple of {n_heads} heads"));
    }
    let head_dim = q_dim / n_heads;
    let kv_dim = dim(shapes, "model.layers.0.self_attn.k_proj.weight", 0)?;
    agree("key/value projection width", kv_dim, n_kv_heads * head_dim)?;

    // The dense-then-MoE schedule is visible in the tensors: a MoE layer has a
    // router, a dense one has a plain `gate_proj`.
    let n_dense_layers = (0..n_layers).take_while(|l| !shapes.contains_key(&format!("model.layers.{l}.mlp.gate.weight"))).count() as u32;
    agree("dense-layer prefix", n_dense_layers, get("first_k_dense_replace")?)?;
    if n_dense_layers >= n_layers {
        return Err(format!("{LABEL}: no layer carries a router; this is not a MoE decoder"));
    }
    let first_moe = n_dense_layers;
    let moe_intermediate_size = dim(shapes, &format!("model.layers.{first_moe}.mlp.experts.0.gate_proj.weight"), 0)?;
    let shared_width = dim(shapes, &format!("model.layers.{first_moe}.mlp.shared_experts.gate_proj.weight"), 0)?;
    if moe_intermediate_size == 0 || shared_width % moe_intermediate_size != 0 {
        return Err(format!("{LABEL}: the fused shared expert is {shared_width} wide, not a multiple of one expert's {moe_intermediate_size}"));
    }

    let shape = DecoderShape {
        vocab,
        n_layers,
        d_model,
        max_position_embeddings: get("max_position_embeddings")?,
        // Absent from this repo's config.json; both are `DeepseekV2Config`'s
        // own documented defaults.
        rms_eps: cfg_f32(config, &path("rms_norm_eps"), 1e-6),
        tie_embeddings: !shapes.contains_key("lm_head.weight"),
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta: cfg_f32(config, &path("rope_theta"), 10_000.0),
        // This checkpoint names no partial-rotary width, so the whole head
        // rotates - the same resolution `gguf::deepseek_ocr::config_from_gguf`
        // applies to the GGUF's `rope.dimension_count = 0`.
        rotary_dim: head_dim,
        n_dense_layers,
        ffn_hidden: dim(shapes, "model.layers.0.mlp.gate_proj.weight", 0)?,
        n_experts: dim(shapes, &format!("model.layers.{first_moe}.mlp.gate.weight"), 0)?,
        top_k: get("num_experts_per_tok")?,
        moe_intermediate_size,
        n_shared_experts: shared_width / moe_intermediate_size,
        n_expert_groups: get("n_group")?,
        n_expert_groups_used: get("topk_group")?,
    };
    agree("routed-expert count", shape.n_experts, get("n_routed_experts")?)?;
    agree("shared-expert count", shape.n_shared_experts, get("n_shared_experts")?)?;
    agree("expert width", shape.moe_intermediate_size, get("moe_intermediate_size")?)?;
    agree("dense feed-forward width", shape.ffn_hidden, get("intermediate_size")?)?;
    Ok(DeepseekV2Config::from_shape(shape, block_size))
}

/// The composite's config, derived from the checkpoint's own tensor shapes
/// (plus the handful of numbers only `config.json` carries) and then refused
/// unless it is [`DeepseekOcrConfig::deepseek_ocr`].
///
/// Deriving *and* comparing is the point, exactly as it is for the GGUF pair:
/// deriving alone would silently serve a differently-shaped checkpoint, and
/// hardcoding alone would not notice one at all.
pub fn config_from_shapes(shapes: &BTreeMap<String, Vec<usize>>, config: &Value, block_size: u32) -> Result<DeepseekOcrConfig, String> {
    let vision = vision_from_shapes(shapes, config)?;
    let cfg = DeepseekOcrConfig {
        sam: SamViTConfig::from(&vision.sam),
        clip: ClipVisionConfig::from_gguf(&vision),
        decoder: decoder_from_shapes(shapes, config, block_size)?,
        // No real-scale analogue; `check_real_scale_shaped` refuses it.
        patch_bypass: false,
    };
    let want = DeepseekOcrConfig::deepseek_ocr(block_size);
    if cfg != want {
        return Err(format!(
            "{LABEL}: this checkpoint's shape is not DeepSeek-OCR's documented preset \
             (derived token grid {:?}, clip {} wide x {} layers, decoder {} layers x d_model {}; \
             want {:?}, {} x {}, {} x {})",
            cfg.token_grid(),
            cfg.clip_width(),
            cfg.clip.layers(),
            cfg.decoder.n_layers(),
            cfg.decoder.d_model(),
            want.token_grid(),
            want.clip_width(),
            want.clip.layers(),
            want.decoder.n_layers(),
            want.decoder.d_model(),
        ));
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `deepseek-ai/DeepSeek-OCR` tensor header, transcribed from the
    /// live repo's own `model-00001-of-000001.safetensors` (2710 tensors, all
    /// BF16) rather than assumed from convention.
    ///
    /// A generator rather than a checked-in dump because the shapes are
    /// regular: 61 distinct name patterns, listed here in the same order the
    /// file declares them. It is deliberately written as an EMITTER while
    /// [`classify`] is written as a MATCHER, so a typo in either one fails the
    /// coverage test below rather than cancelling out.
    fn upstream_shapes() -> BTreeMap<String, Vec<usize>> {
        let mut m: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut put = |n: String, s: Vec<usize>| {
            assert!(m.insert(n.clone(), s).is_none(), "duplicate upstream name {n}");
        };

        // -- SAM ViT-B tower (179) ------------------------------------------
        put("model.sam_model.pos_embed".into(), vec![1, 64, 64, 768]);
        put("model.sam_model.patch_embed.proj.weight".into(), vec![768, 3, 16, 16]);
        put("model.sam_model.patch_embed.proj.bias".into(), vec![768]);
        for l in 0..12 {
            let b = |leaf: &str| format!("model.sam_model.blocks.{l}.{leaf}");
            // Global-attention blocks span the whole 64x64 grid (2*64-1 rows);
            // the rest span the 14x14 window (2*14-1).
            let rows = if [2, 5, 8, 11].contains(&l) { 127 } else { 27 };
            put(b("norm1.weight"), vec![768]);
            put(b("norm1.bias"), vec![768]);
            put(b("attn.rel_pos_h"), vec![rows, 64]);
            put(b("attn.rel_pos_w"), vec![rows, 64]);
            put(b("attn.qkv.weight"), vec![2304, 768]);
            put(b("attn.qkv.bias"), vec![2304]);
            put(b("attn.proj.weight"), vec![768, 768]);
            put(b("attn.proj.bias"), vec![768]);
            put(b("norm2.weight"), vec![768]);
            put(b("norm2.bias"), vec![768]);
            put(b("mlp.lin1.weight"), vec![3072, 768]);
            put(b("mlp.lin1.bias"), vec![3072]);
            put(b("mlp.lin2.weight"), vec![768, 3072]);
            put(b("mlp.lin2.bias"), vec![768]);
        }
        put("model.sam_model.neck.0.weight".into(), vec![256, 768, 1, 1]);
        put("model.sam_model.neck.1.weight".into(), vec![256]);
        put("model.sam_model.neck.1.bias".into(), vec![256]);
        put("model.sam_model.neck.2.weight".into(), vec![256, 256, 3, 3]);
        put("model.sam_model.neck.3.weight".into(), vec![256]);
        put("model.sam_model.neck.3.bias".into(), vec![256]);
        put("model.sam_model.net_2.weight".into(), vec![512, 256, 3, 3]);
        put("model.sam_model.net_3.weight".into(), vec![1024, 512, 3, 3]);

        // -- CLIP-L/14 tower (293) ------------------------------------------
        put("model.vision_model.embeddings.class_embedding".into(), vec![1024]);
        put("model.vision_model.embeddings.patch_embedding.weight".into(), vec![1024, 3, 14, 14]);
        put("model.vision_model.embeddings.position_embedding.weight".into(), vec![257, 1024]);
        put("model.vision_model.pre_layrnorm.weight".into(), vec![1024]);
        put("model.vision_model.pre_layrnorm.bias".into(), vec![1024]);
        for l in 0..24 {
            let b = |leaf: &str| format!("model.vision_model.transformer.layers.{l}.{leaf}");
            put(b("layer_norm1.weight"), vec![1024]);
            put(b("layer_norm1.bias"), vec![1024]);
            put(b("self_attn.qkv_proj.weight"), vec![3072, 1024]);
            put(b("self_attn.qkv_proj.bias"), vec![3072]);
            put(b("self_attn.out_proj.weight"), vec![1024, 1024]);
            put(b("self_attn.out_proj.bias"), vec![1024]);
            put(b("layer_norm2.weight"), vec![1024]);
            put(b("layer_norm2.bias"), vec![1024]);
            put(b("mlp.fc1.weight"), vec![4096, 1024]);
            put(b("mlp.fc1.bias"), vec![4096]);
            put(b("mlp.fc2.weight"), vec![1024, 4096]);
            put(b("mlp.fc2.bias"), vec![1024]);
        }

        // -- projector and the two learned image-block rows (4) -------------
        put("model.projector.layers.weight".into(), vec![1280, 2048]);
        put("model.projector.layers.bias".into(), vec![1280]);
        put("model.image_newline".into(), vec![1280]);
        // `seperator` is the upstream spelling; brain's is spelled correctly.
        put("model.view_seperator".into(), vec![1280]);

        // -- DeepSeek-V2 decoder (2234) -------------------------------------
        put("model.embed_tokens.weight".into(), vec![129280, 1280]);
        for l in 0..12 {
            let b = |leaf: &str| format!("model.layers.{l}.{leaf}");
            put(b("input_layernorm.weight"), vec![1280]);
            put(b("post_attention_layernorm.weight"), vec![1280]);
            for p in ["q", "k", "v", "o"] {
                put(b(&format!("self_attn.{p}_proj.weight")), vec![1280, 1280]);
            }
            if l == 0 {
                // `first_k_dense_replace = 1`: layer 0 is a plain dense MLP.
                put(b("mlp.gate_proj.weight"), vec![6848, 1280]);
                put(b("mlp.up_proj.weight"), vec![6848, 1280]);
                put(b("mlp.down_proj.weight"), vec![1280, 6848]);
            } else {
                put(b("mlp.gate.weight"), vec![64, 1280]);
                for e in 0..64 {
                    put(b(&format!("mlp.experts.{e}.gate_proj.weight")), vec![896, 1280]);
                    put(b(&format!("mlp.experts.{e}.up_proj.weight")), vec![896, 1280]);
                    put(b(&format!("mlp.experts.{e}.down_proj.weight")), vec![1280, 896]);
                }
                put(b("mlp.shared_experts.gate_proj.weight"), vec![1792, 1280]);
                put(b("mlp.shared_experts.up_proj.weight"), vec![1792, 1280]);
                put(b("mlp.shared_experts.down_proj.weight"), vec![1280, 1792]);
            }
        }
        put("model.norm.weight".into(), vec![1280]);
        put("lm_head.weight".into(), vec![129280, 1280]);

        m
    }

    /// The real `config.json`, transcribed from the live repo. Only the fields
    /// [`config_from_shapes`] actually reads are kept; the rest of that file
    /// duplicates `language_config` at the top level.
    fn upstream_config() -> Value {
        serde_json::json!({
            "architectures": ["DeepseekOCRForCausalLM"],
            "model_type": "deepseek_vl_v2",
            "language_config": {
                "vocab_size": 129280,
                "hidden_size": 1280,
                "intermediate_size": 6848,
                "moe_intermediate_size": 896,
                "num_hidden_layers": 12,
                "num_attention_heads": 10,
                "num_key_value_heads": 10,
                "n_routed_experts": 64,
                "num_experts_per_tok": 6,
                "n_shared_experts": 2,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 8192,
                "n_group": 1,
                "topk_group": 1,
            },
            "projector_config": { "input_dim": 2048, "n_embed": 1280, "projector_type": "linear" },
            "vision_config": {
                "image_size": 1024,
                "mlp_ratio": 3.7362,
                "model_name": "deeplip_b_l",
                "width": {
                    "clip-l-14-224": { "heads": 16, "image_size": 224, "layers": 24, "patch_size": 14, "width": 1024 },
                    "sam_vit_b": { "downsample_channels": [512, 1024], "global_attn_indexes": [2, 5, 8, 11], "heads": 12, "layers": 12, "width": 768 }
                }
            }
        })
    }

    /// The transcription itself is the thing most likely to be wrong, so it is
    /// pinned to the counts the real file declares before anything is derived
    /// from it.
    #[test]
    fn the_transcribed_upstream_header_has_the_shape_the_real_repo_publishes() {
        let s = upstream_shapes();
        assert_eq!(s.len(), 2710, "the real checkpoint declares 2710 tensors");
        let count = |p: &str| s.keys().filter(|k| k.starts_with(p)).count();
        assert_eq!(count("model.sam_model."), 179);
        assert_eq!(count("model.vision_model."), 293);
        assert_eq!(count("model.layers."), 2231);
        // 6.67 GB at BF16, matching the index's own `total_size`.
        let elems: usize = s.values().map(|v| v.iter().product::<usize>()).sum();
        assert_eq!(elems * 2, 6_672_212_480, "total_size from model.safetensors.index.json");
    }

    /// **Two-way coverage.** Every upstream tensor lands on a brain-side name,
    /// every brain-side name is produced exactly once, and the element counts
    /// agree - the same contract `gguf::import` enforces for the GGUF pair,
    /// checked here against the manifests that production code already
    /// declares, so a wrong rename cannot pass by agreeing with itself.
    #[test]
    fn every_upstream_tensor_maps_onto_the_two_manifests_exactly() {
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        let shapes = upstream_shapes();

        let mut enc: BTreeMap<String, usize> = BTreeMap::new();
        let mut dec: BTreeMap<String, usize> = BTreeMap::new();
        // Which experts each bank has been handed, so a bank's element count
        // below is the SUM of its parts and a duplicate or missing expert is
        // still a failure rather than an average that happens to fit.
        let mut bank_experts: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (name, shape) in &shapes {
            let numel: usize = shape.iter().product();
            match classify(name, &cfg).unwrap_or_else(|e| panic!("{e}")) {
                Half::Encoder(n) => assert!(enc.insert(n.clone(), numel).is_none(), "two upstream tensors map to encoder {n}"),
                Half::Decoder(n) => assert!(dec.insert(n.clone(), numel).is_none(), "two upstream tensors map to decoder {n}"),
                Half::DecoderExpert { bank, expert } => {
                    *dec.entry(bank.clone()).or_insert(0) += numel;
                    bank_experts.entry(bank).or_default().push(expert);
                }
            }
        }
        assert_eq!(bank_experts.len(), 11 * 3, "one bank per projection on each of the 11 MoE layers");
        for (bank, mut experts) in bank_experts {
            experts.sort_unstable();
            assert_eq!(experts, (0..cfg.decoder.shape.n_experts).collect::<Vec<u32>>(), "{bank}: every expert exactly once");
        }

        for (label, got, want) in [("encoder", &enc, encoder_manifest(&cfg)), ("decoder", &dec, cfg.decoder.param_list())] {
            let want: BTreeMap<String, usize> = want.into_iter().collect();
            let missing: Vec<&String> = want.keys().filter(|k| !got.contains_key(*k)).collect();
            let extra: Vec<&String> = got.keys().filter(|k| !want.contains_key(*k)).collect();
            assert!(missing.is_empty(), "{label}: manifest names no upstream tensor produced: {missing:?}");
            assert!(extra.is_empty(), "{label}: upstream tensors with no manifest entry: {extra:?}");
            for (name, numel) in &want {
                assert_eq!(got[name], *numel, "{label}: {name} element count");
            }
        }
    }

    /// The same coverage, through the function the loader actually calls -
    /// `plans`, not `classify` - because that is what decides whether a real
    /// `Session::load` succeeds, and it is where an INCOMPLETE checkpoint has
    /// to be refused rather than loaded with a silently missing tensor.
    #[test]
    fn plans_partition_the_checkpoint_and_refuse_an_incomplete_one() {
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        let names: Vec<String> = upstream_shapes().into_keys().collect();
        let (enc, dec) = plans(&names, &cfg).expect("the real checkpoint plans cleanly");
        assert_eq!(enc.len(), 476, "the vision half");
        // 2234 brain-side decoder names when every routed expert was its own
        // parameter; the 11 MoE layers' 64x3 experts are now 3 fused banks
        // each, so 2234 - 11 * (64 * 3 - 3) = 155.
        assert_eq!(dec.len(), 155, "the decoder half");
        assert_eq!(
            enc.len() + dec.len() + 11 * (64 - 1) * 3,
            names.len(),
            "every source tensor is claimed exactly once (each bank claims 64 of them, not 1)"
        );

        // Drop one expert. Every remaining tensor still classifies, so only
        // the manifest check can catch it.
        let short: Vec<String> = names.iter().filter(|n| !n.starts_with("model.layers.5.mlp.experts.17.")).cloned().collect();
        let e = plans(&short, &cfg).unwrap_err();
        assert!(e.contains("blocks.5.mlp.experts.17"), "the refusal must name what is missing: {e}");

        // A checkpoint carrying a tensor this crate does not understand stops
        // the import rather than being partly loaded.
        let mut extra = names.clone();
        extra.push("model.layers.0.self_attn.q_norm.weight".to_string());
        assert!(plans(&extra, &cfg).is_err(), "an unclassifiable tensor must stop the import");
    }

    /// The config is DERIVED from the checkpoint's own shapes, not read out of
    /// a preset - and the derivation reproduces the preset exactly.
    #[test]
    fn the_config_is_derived_from_the_checkpoints_own_shapes() {
        let got = config_from_shapes(&upstream_shapes(), &upstream_config(), 1).expect("the real checkpoint is the documented preset");
        assert_eq!(got, DeepseekOcrConfig::deepseek_ocr(1));
        got.check_real_scale_shaped();
    }

    /// A checkpoint of a different shape is REFUSED by name, never adapted to
    /// - deriving without comparing would silently serve a different model.
    #[test]
    fn a_checkpoint_that_is_not_the_preset_is_refused_by_name() {
        // A CLIP tower one block shallower: every individual tensor is still
        // well-formed, so only the comparison catches it.
        let mut shapes = upstream_shapes();
        for leaf in ["layer_norm1.weight", "layer_norm1.bias", "self_attn.qkv_proj.weight", "self_attn.qkv_proj.bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias", "layer_norm2.weight", "layer_norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"] {
            shapes.remove(&format!("model.vision_model.transformer.layers.23.{leaf}"));
        }
        let e = config_from_shapes(&shapes, &upstream_config(), 1).unwrap_err();
        assert!(e.contains("CLIP depth") && e.contains("23") && e.contains("24"), "the refusal must name both numbers that disagree: {e}");

        // CLIP's head count is the one number no tensor shape carries, so
        // nothing cross-checks it and the derivation stays self-consistent.
        // Only the comparison against the preset can catch this one -- which
        // is exactly why the comparison exists.
        let mut cfg = upstream_config();
        cfg["vision_config"]["width"]["clip-l-14-224"]["heads"] = Value::from(8);
        let e = config_from_shapes(&upstream_shapes(), &cfg, 1).unwrap_err();
        assert!(e.contains("preset"), "the refusal must say what it compared against: {e}");

        // A decoder whose config.json contradicts its own tensors is refused
        // by name rather than resolved in favour of either side.
        let mut cfg = upstream_config();
        cfg["language_config"]["num_attention_heads"] = Value::from(8);
        let e = config_from_shapes(&upstream_shapes(), &cfg, 1).unwrap_err();
        assert!(e.contains("key/value projection width"), "got {e}");
    }

    /// An unrecognized name is an error, never a silent drop: a checkpoint that
    /// grew a tensor must stop the import, not load a model missing it.
    #[test]
    fn an_unrecognized_upstream_tensor_is_an_error() {
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        for name in ["model.sam_model.blocks.0.attn.bias", "model.layers.0.self_attn.q_norm.weight", "model.vision_model.post_layernorm.weight", "model.layers.12.input_layernorm.weight"] {
            assert!(classify(name, &cfg).is_err(), "{name} must not classify");
        }
    }
}
