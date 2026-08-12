// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR's **vision tower** - the `mmproj` file llama.cpp's mtmd tooling
//! produces (`general.architecture = "clip"`, `clip.projector_type =
//! "deepseekocr"`).
//!
//! Every mmproj shares the architecture string `clip`, so the architecture key
//! alone cannot select a mapping; `clip.projector_type` is the secondary
//! discriminator (see [`crate::registry`]). This one carries four stages in a
//! single file (476 tensors):
//!
//! 1. **SAM ViT-B** (`v.sam.*`): 12 blocks, width 768, 12 heads, patch 16 over
//!    a 1024×1024 input (a 64×64 patch grid), decomposed relative position
//!    tables per block, and a 2-conv "neck" down to 256 channels. Blocks
//!    alternate windowed (`window_size = 14`) and global attention - which is
//!    readable straight off the tensors: a block's `attn.pos_h` table is
//!    `[2*14-1, 64] = [27,64]` when windowed and `[2*64-1, 64] = [127,64]`
//!    when global. In this checkpoint blocks 2, 5, 8, 11 are global.
//! 2. **The 16× compressor** (`v.sam.net_2`, `v.sam.net_3`): two stride-2 3×3
//!    convs, 256→512→1024, taking the neck's 64×64 map to 16×16 - matching
//!    CLIP's own 16×16 patch grid so the two towers' tokens line up.
//! 3. **CLIP-L** (`v.*`): 24 blocks, width 1024, 16 heads, patch 14 over
//!    224×224 (256 patches + a class token = the 257 learned positions).
//! 4. **The projector** (`mm.model.fc`): one linear `2048 → 1280` over the
//!    channel-concatenation of CLIP's 1024 and the compressor's 1024, into the
//!    language model's width, plus the learned `image_newline` and
//!    `view_seperator` [sic] row/view separator vectors.
//!
//! ## `clip.vision.feed_forward_length = 64` is wrong in this file
//!
//! The real MLP widths are in the tensors: `v.blk.N.ffn_up.weight` is torch
//! `[4096,1024]` (CLIP-L's 4×) and `v.sam.blk.N.mlp.lin1.weight` is torch
//! `[3072,768]` (SAM's 4×). The declared 64 matches neither. It is a
//! **converter bug**, confirmed in llama.cpp's DeepSeek-OCR vision conversion:
//! it computes `intermediate_size = heads * 4` where `width * 4` was meant, and
//! CLIP-L has 16 heads - 16×4 = 64 instead of 1024×4 = 4096. Nothing downstream
//! notices, because llama.cpp's own clip loader reads `n_ff` once to print it
//! and sizes every FFN from the tensor shapes instead. This module does the
//! same: **both** feed-forward widths come from tensor shapes and that key is
//! never read.
//!
//! The brain-side names below are this crate's proposal; the encoder that
//! consumes them (expected to compose `crates/sam1`-style SAM blocks with
//! `crates/clip`) is future work.

use checkpoint::gguf::MmapGguf;
use checkpoint::st::ModelCard;
use serde_json::Value;

use crate::import::{self, ImportStats, Mapped};
use crate::kv::ArchKv;

/// The `general.architecture` every mmproj declares.
pub const GGUF_ARCHITECTURE: &str = "clip";
/// The `clip.projector_type` that selects *this* mapping.
pub const PROJECTOR_TYPE: &str = "deepseekocr";

/// The SAM ViT-B half of the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct SamConfig {
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub ffn_hidden: u32,
    pub patch_size: u32,
    /// Patch grid side (`image_size / patch_size`), read off the position
    /// embedding rather than assumed - the file's `clip.vision.image_size`
    /// describes the CLIP branch, not this one.
    pub grid: u32,
    /// Local-attention window, in patches.
    pub window_size: u32,
    /// Blocks using **global** attention; the rest are windowed.
    pub global_attn_layers: Vec<u32>,
    /// Neck output channels (the 1×1 conv's width).
    pub neck_channels: u32,
    /// The compressor's two stride-2 conv widths (`net_2`, `net_3`).
    pub compress_mid: u32,
    pub compress_out: u32,
}

impl SamConfig {
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
    /// Relative-position table rows for block `l`: `2 * extent - 1`, where the
    /// extent is the window for a windowed block and the full grid for a
    /// global one.
    pub fn rel_pos_rows(&self, l: u32) -> u32 {
        let extent = if self.global_attn_layers.contains(&l) { self.grid } else { self.window_size };
        2 * extent - 1
    }
    pub fn image_size(&self) -> u32 {
        self.grid * self.patch_size
    }
}

/// The CLIP-L half of the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipConfig {
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub ffn_hidden: u32,
    pub patch_size: u32,
    pub image_size: u32,
    /// Learned position rows (class token + patches).
    pub n_positions: u32,
    pub layer_norm_eps: f32,
}

/// The whole `projector_type = deepseekocr` vision tower.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekOcrVisionConfig {
    pub sam: SamConfig,
    pub clip: ClipConfig,
    /// The projector's input width - CLIP's `d_model` concatenated with the
    /// compressor's output channels.
    pub projector_in: u32,
    /// The projector's output width: the language model's `d_model`.
    pub projection_dim: u32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    /// `clip.use_gelu`; false means the quick-GELU variant.
    pub use_gelu: bool,
    /// `clip.vision.projector.scale_factor` - the pixel-shuffle factor
    /// (1 = none) applied before the projector.
    pub scale_factor: u32,
}

impl DeepseekOcrVisionConfig {
    /// The canonical output manifest: every brain-side tensor and its element
    /// count.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = Vec::new();
        let s = &self.sam;
        let (sd, sff) = (s.d_model as usize, s.ffn_hidden as usize);
        let p = s.patch_size as usize;

        out.push(("vision.sam.patch_embed.weight".to_string(), sd * 3 * p * p));
        out.push(("vision.sam.patch_embed.bias".to_string(), sd));
        out.push(("vision.sam.pos_embed".to_string(), (s.grid * s.grid) as usize * sd));
        for l in 0..s.n_layers {
            let b = |leaf: &str| format!("vision.sam.blocks.{l}.{leaf}");
            let rel = s.rel_pos_rows(l) as usize * s.head_dim() as usize;
            out.push((b("norm1.weight"), sd));
            out.push((b("norm1.bias"), sd));
            out.push((b("attn.qkv.weight"), 3 * sd * sd));
            out.push((b("attn.qkv.bias"), 3 * sd));
            out.push((b("attn.proj.weight"), sd * sd));
            out.push((b("attn.proj.bias"), sd));
            out.push((b("attn.rel_pos_h"), rel));
            out.push((b("attn.rel_pos_w"), rel));
            out.push((b("norm2.weight"), sd));
            out.push((b("norm2.bias"), sd));
            out.push((b("mlp.fc1.weight"), sff * sd));
            out.push((b("mlp.fc1.bias"), sff));
            out.push((b("mlp.fc2.weight"), sd * sff));
            out.push((b("mlp.fc2.bias"), sd));
        }
        let neck = s.neck_channels as usize;
        out.push(("vision.sam.neck.conv1.weight".to_string(), neck * sd));
        out.push(("vision.sam.neck.norm1.weight".to_string(), neck));
        out.push(("vision.sam.neck.norm1.bias".to_string(), neck));
        out.push(("vision.sam.neck.conv2.weight".to_string(), neck * neck * 3 * 3));
        out.push(("vision.sam.neck.norm2.weight".to_string(), neck));
        out.push(("vision.sam.neck.norm2.bias".to_string(), neck));
        out.push(("vision.sam.compress.conv1.weight".to_string(), s.compress_mid as usize * neck * 3 * 3));
        out.push((
            "vision.sam.compress.conv2.weight".to_string(),
            s.compress_out as usize * s.compress_mid as usize * 3 * 3,
        ));

        let c = &self.clip;
        let (cd, cff) = (c.d_model as usize, c.ffn_hidden as usize);
        let cp = c.patch_size as usize;
        out.push(("vision.clip.class_embed".to_string(), cd));
        out.push(("vision.clip.patch_embed.weight".to_string(), cd * 3 * cp * cp));
        out.push(("vision.clip.pos_embed".to_string(), c.n_positions as usize * cd));
        out.push(("vision.clip.pre_norm.weight".to_string(), cd));
        out.push(("vision.clip.pre_norm.bias".to_string(), cd));
        for l in 0..c.n_layers {
            let b = |leaf: &str| format!("vision.clip.blocks.{l}.{leaf}");
            out.push((b("norm1.weight"), cd));
            out.push((b("norm1.bias"), cd));
            out.push((b("attn.qkv.weight"), 3 * cd * cd));
            out.push((b("attn.qkv.bias"), 3 * cd));
            out.push((b("attn.proj.weight"), cd * cd));
            out.push((b("attn.proj.bias"), cd));
            out.push((b("norm2.weight"), cd));
            out.push((b("norm2.bias"), cd));
            out.push((b("mlp.fc1.weight"), cff * cd));
            out.push((b("mlp.fc1.bias"), cff));
            out.push((b("mlp.fc2.weight"), cd * cff));
            out.push((b("mlp.fc2.bias"), cd));
        }

        let (pin, pout) = (self.projector_in as usize, self.projection_dim as usize);
        out.push(("vision.projector.fc.weight".to_string(), pout * pin));
        out.push(("vision.projector.fc.bias".to_string(), pout));
        out.push(("vision.image_newline".to_string(), pout));
        out.push(("vision.view_separator".to_string(), pout));
        out
    }

    /// The config as it is stored in the produced checkpoint's header.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "deepseek-ocr-vision",
            "projector_type": PROJECTOR_TYPE,
            "sam": {
                "d_model": self.sam.d_model,
                "n_layers": self.sam.n_layers,
                "n_heads": self.sam.n_heads,
                "ffn_hidden": self.sam.ffn_hidden,
                "patch_size": self.sam.patch_size,
                "image_size": self.sam.image_size(),
                "grid": self.sam.grid,
                "window_size": self.sam.window_size,
                "global_attn_layers": self.sam.global_attn_layers,
                "neck_channels": self.sam.neck_channels,
                "compress_mid": self.sam.compress_mid,
                "compress_out": self.sam.compress_out,
            },
            "clip": {
                "d_model": self.clip.d_model,
                "n_layers": self.clip.n_layers,
                "n_heads": self.clip.n_heads,
                "ffn_hidden": self.clip.ffn_hidden,
                "patch_size": self.clip.patch_size,
                "image_size": self.clip.image_size,
                "n_positions": self.clip.n_positions,
                "layer_norm_eps": self.clip.layer_norm_eps,
            },
            "projector_in": self.projector_in,
            "projection_dim": self.projection_dim,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "use_gelu": self.use_gelu,
            "scale_factor": self.scale_factor,
        })
    }
}

/// A tensor's leading (torch-outermost) dimension, or a named error.
fn rows(mg: &MmapGguf, name: &str) -> Result<u32, String> {
    let s = mg.shape(name).ok_or_else(|| format!("deepseek-ocr-vision: missing {name}"))?;
    Ok(*s.first().ok_or_else(|| format!("deepseek-ocr-vision: {name} is 0-dimensional"))? as u32)
}

/// Derive [`DeepseekOcrVisionConfig`] from the mmproj's KV, with tensor shapes
/// as the authority wherever the KV is missing or contradicted (both
/// feed-forward widths, the SAM grid, the neck/compressor channel counts, the
/// projector's input width, and which blocks use global attention).
pub fn config_from_gguf(mg: &MmapGguf) -> Result<DeepseekOcrVisionConfig, String> {
    let root = ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;
    let projector = root.str("projector_type").unwrap_or("");
    if projector != PROJECTOR_TYPE {
        return Err(format!("deepseek-ocr-vision: expected clip.projector_type={PROJECTOR_TYPE:?}, got {projector:?}"));
    }
    let v = root.scoped("vision");
    let sam_kv = v.scoped("sam");

    let sam_layers = sam_kv.req_u32("block_count")?;
    let sam_d = sam_kv.req_u32("embedding_length")?;
    let sam_heads = sam_kv.req_u32("head_count")?;
    if sam_heads == 0 || sam_d % sam_heads != 0 {
        return Err(format!("deepseek-ocr-vision: sam embedding_length {sam_d} not divisible by head_count {sam_heads}"));
    }
    let sam_head_dim = sam_d / sam_heads;
    let window_size = v.req_u32("window_size")?;

    // The SAM position embedding is torch [1, grid, grid, d_model].
    let pos = mg.shape("v.sam.pos_embd.weight").ok_or("deepseek-ocr-vision: missing v.sam.pos_embd.weight")?;
    let grid = *pos.get(1).ok_or("deepseek-ocr-vision: v.sam.pos_embd.weight is not [1,grid,grid,d]")? as u32;

    // Global vs windowed attention is decided by each block's own relative
    // position table extent - no KV key states it.
    let mut global_attn_layers = Vec::new();
    for l in 0..sam_layers {
        let r = rows(mg, &format!("v.sam.blk.{l}.attn.pos_h.weight"))?;
        if r == 2 * grid - 1 {
            global_attn_layers.push(l);
        } else if r != 2 * window_size - 1 {
            return Err(format!(
                "deepseek-ocr-vision: v.sam.blk.{l}.attn.pos_h.weight has {r} rows, expected {} (windowed) or {} (global)",
                2 * window_size - 1,
                2 * grid - 1
            ));
        }
    }

    let sam = SamConfig {
        d_model: sam_d,
        n_layers: sam_layers,
        n_heads: sam_heads,
        // Not `clip.vision.feed_forward_length` - see this module's doc.
        ffn_hidden: rows(mg, "v.sam.blk.0.mlp.lin1.weight")?,
        // The SAM patch size is not in the KV either (`clip.vision.patch_size`
        // is CLIP's 14); the patch-embed conv kernel is.
        patch_size: *mg
            .shape("v.sam.patch_embd.weight")
            .and_then(|s| s.get(2))
            .ok_or("deepseek-ocr-vision: v.sam.patch_embd.weight is not [d,3,k,k]")? as u32,
        grid,
        window_size,
        global_attn_layers,
        neck_channels: rows(mg, "v.sam.neck.0.weight")?,
        compress_mid: rows(mg, "v.sam.net_2.weight")?,
        compress_out: rows(mg, "v.sam.net_3.weight")?,
    };
    if sam.head_dim() != sam_head_dim {
        return Err("deepseek-ocr-vision: sam head_dim disagreement".to_string());
    }

    let clip = ClipConfig {
        d_model: v.req_u32("embedding_length")?,
        n_layers: v.req_u32("block_count")?,
        n_heads: v.req_u32("attention.head_count")?,
        ffn_hidden: rows(mg, "v.blk.0.ffn_up.weight")?,
        patch_size: v.req_u32("patch_size")?,
        image_size: v.req_u32("image_size")?,
        n_positions: rows(mg, "v.position_embd.weight")?,
        layer_norm_eps: v.f32_or("attention.layer_norm_epsilon", 1e-6),
    };
    let expect_positions = 1 + (clip.image_size / clip.patch_size).pow(2);
    if clip.n_positions != expect_positions {
        return Err(format!(
            "deepseek-ocr-vision: {} learned positions but image_size/patch_size imply {expect_positions}",
            clip.n_positions
        ));
    }

    let fc = mg.shape("mm.model.fc.weight").ok_or("deepseek-ocr-vision: missing mm.model.fc.weight")?;
    let projector_in = *fc.get(1).ok_or("deepseek-ocr-vision: mm.model.fc.weight is not 2-D")? as u32;
    let projection_dim = fc[0] as u32;
    if projector_in != clip.d_model + sam.compress_out {
        return Err(format!(
            "deepseek-ocr-vision: projector input {projector_in} != clip {} + compressor {}",
            clip.d_model, sam.compress_out
        ));
    }
    if let Some(declared) = v.u32("projection_dim") {
        if declared != projection_dim {
            return Err(format!("deepseek-ocr-vision: projection_dim={declared} disagrees with fc rows {projection_dim}"));
        }
    }

    Ok(DeepseekOcrVisionConfig {
        sam,
        clip,
        projector_in,
        projection_dim,
        image_mean: v.f32_array("image_mean").unwrap_or_else(|| vec![0.5; 3]),
        image_std: v.f32_array("image_std").unwrap_or_else(|| vec![0.5; 3]),
        use_gelu: root.bool("use_gelu").unwrap_or(false),
        scale_factor: v.u32_or("projector.scale_factor", 1),
    })
}

/// Classify one mmproj tensor name.
///
/// `v.sam.blk.N.pre_ln` → `norm1` and `post_ln` → `norm2` reads llama.cpp's
/// names positionally (SAM's ViT block is `x += attn(norm1(x)); x +=
/// mlp(norm2(x))`, so the "post" norm is still a pre-MLP norm).
/// `TODO(deepseek-ocr): re-confirm that pairing when the encoder forward is
/// written - a swapped pair would be silently wrong, not a crash.`
pub fn classify(name: &str, cfg: &DeepseekOcrVisionConfig) -> Result<Mapped, String> {
    // Top-level, non-block tensors first.
    let simple = |s: &str| Ok(Mapped::Simple(s.to_string()));
    match name {
        // `seperator` is llama.cpp's spelling; brain's name is spelled correctly.
        "v.view_seperator" => return simple("vision.view_separator"),
        "v.image_newline" => return simple("vision.image_newline"),
        "mm.model.fc.weight" => return simple("vision.projector.fc.weight"),
        "mm.model.fc.bias" => return simple("vision.projector.fc.bias"),
        "v.class_embd" => return simple("vision.clip.class_embed"),
        "v.patch_embd.weight" => return simple("vision.clip.patch_embed.weight"),
        "v.position_embd.weight" => return simple("vision.clip.pos_embed"),
        "v.pre_ln.weight" => return simple("vision.clip.pre_norm.weight"),
        "v.pre_ln.bias" => return simple("vision.clip.pre_norm.bias"),
        "v.sam.patch_embd.weight" => return simple("vision.sam.patch_embed.weight"),
        "v.sam.patch_embd.bias" => return simple("vision.sam.patch_embed.bias"),
        "v.sam.pos_embd.weight" => return simple("vision.sam.pos_embed"),
        "v.sam.neck.0.weight" => return simple("vision.sam.neck.conv1.weight"),
        "v.sam.neck.1.weight" => return simple("vision.sam.neck.norm1.weight"),
        "v.sam.neck.1.bias" => return simple("vision.sam.neck.norm1.bias"),
        "v.sam.neck.2.weight" => return simple("vision.sam.neck.conv2.weight"),
        "v.sam.neck.3.weight" => return simple("vision.sam.neck.norm2.weight"),
        "v.sam.neck.3.bias" => return simple("vision.sam.neck.norm2.bias"),
        "v.sam.net_2.weight" => return simple("vision.sam.compress.conv1.weight"),
        "v.sam.net_3.weight" => return simple("vision.sam.compress.conv2.weight"),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("v.sam.blk.") {
        let (l, leaf) = split_block(rest, name, cfg.sam.n_layers)?;
        let b = |s: &str| Mapped::Simple(format!("vision.sam.blocks.{l}.{s}"));
        return match leaf {
            "pre_ln.weight" => Ok(b("norm1.weight")),
            "pre_ln.bias" => Ok(b("norm1.bias")),
            "post_ln.weight" => Ok(b("norm2.weight")),
            "post_ln.bias" => Ok(b("norm2.bias")),
            "attn.qkv.weight" => Ok(b("attn.qkv.weight")),
            "attn.qkv.bias" => Ok(b("attn.qkv.bias")),
            "attn.out.weight" => Ok(b("attn.proj.weight")),
            "attn.out.bias" => Ok(b("attn.proj.bias")),
            "attn.pos_h.weight" => Ok(b("attn.rel_pos_h")),
            "attn.pos_w.weight" => Ok(b("attn.rel_pos_w")),
            "mlp.lin1.weight" => Ok(b("mlp.fc1.weight")),
            "mlp.lin1.bias" => Ok(b("mlp.fc1.bias")),
            "mlp.lin2.weight" => Ok(b("mlp.fc2.weight")),
            "mlp.lin2.bias" => Ok(b("mlp.fc2.bias")),
            other => Err(format!("unrecognized SAM block leaf {other:?} in {name:?}")),
        };
    }

    if let Some(rest) = name.strip_prefix("v.blk.") {
        let (l, leaf) = split_block(rest, name, cfg.clip.n_layers)?;
        let b = |s: &str| Mapped::Simple(format!("vision.clip.blocks.{l}.{s}"));
        return match leaf {
            "ln1.weight" => Ok(b("norm1.weight")),
            "ln1.bias" => Ok(b("norm1.bias")),
            "ln2.weight" => Ok(b("norm2.weight")),
            "ln2.bias" => Ok(b("norm2.bias")),
            "attn_qkv.weight" => Ok(b("attn.qkv.weight")),
            "attn_qkv.bias" => Ok(b("attn.qkv.bias")),
            "attn_out.weight" => Ok(b("attn.proj.weight")),
            "attn_out.bias" => Ok(b("attn.proj.bias")),
            "ffn_up.weight" => Ok(b("mlp.fc1.weight")),
            "ffn_up.bias" => Ok(b("mlp.fc1.bias")),
            "ffn_down.weight" => Ok(b("mlp.fc2.weight")),
            "ffn_down.bias" => Ok(b("mlp.fc2.bias")),
            other => Err(format!("unrecognized CLIP block leaf {other:?} in {name:?}")),
        };
    }

    Err(format!("unrecognized tensor {name:?}"))
}

/// Split `"{index}.{leaf}"`, bounds-checking the index against the tower's
/// depth so a converter that grows a block never lands silently outside the
/// parameter list.
fn split_block<'a>(rest: &'a str, full: &str, n_layers: u32) -> Result<(u32, &'a str), String> {
    let (idx, leaf) = rest.split_once('.').ok_or_else(|| format!("malformed block tensor name {full:?}"))?;
    let l: u32 = idx.parse().map_err(|_| format!("malformed block index in {full:?}"))?;
    if l >= n_layers {
        return Err(format!("{full}: block index {l} beyond block_count {n_layers}"));
    }
    Ok((l, leaf))
}

/// Import a DeepSeek-OCR mmproj GGUF into brain's native format.
pub fn import(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let cfg = config_from_gguf(mg)?;
    let params = cfg.param_list();

    let mut card = ModelCard::new(id_override.unwrap_or("deepseek-ocr-vision"), "deepseek-ocr");
    card.param_count = Some(params.iter().map(|(_, n)| *n as u64).sum());

    import::to_st(mg, &params, &|n| classify(n, &cfg), out_path, &cfg.to_json(), Some(&card), "deepseek-ocr-vision")
}
