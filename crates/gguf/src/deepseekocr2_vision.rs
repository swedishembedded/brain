// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2's **vision tower** mmproj (`general.architecture = "clip"`,
//! `clip.projector_type = "deepseekocr2"`).
//!
//! The predecessor's mmproj ([`crate::deepseek_ocr_vision`]) chains SAM ViT-B
//! into a 16x conv compressor into a whole second CLIP-L/14 tower. This one
//! drops CLIP entirely: SAM's neck now ends at the encoder's own width (896,
//! not CLIP's 1024 - `config.json`'s stated `downsample_channels` disagrees
//! with the real `net_3` tensor and is wrong), and SAM's token grid is fed
//! straight into a 24-layer GQA transformer that also carries two small
//! **learned query banks** (`v.resample_query_768`, 144 rows, for a 768x768
//! tile; `v.resample_query_1024`, 256 rows, for the 1024x1024 global view).
//! The image tokens and the matching query bank are concatenated (image
//! first), and only the query rows survive past the tower - a resampler, not
//! a plain encoder. A final norm (`v.post_ln`) and a single linear
//! (`mm.model.fc`) finish the tower; nothing here re-derives that forward
//! pass, only the tensors it needs.
//!
//! [`SamConfig`] is reused verbatim from [`crate::deepseek_ocr_vision`] - the
//! SAM half of this file is byte-for-byte the same tower under the same
//! `vision.sam.*` names `crates/sam1` already reads, so there is nothing
//! model-specific left to describe about it here.
//!
//! ## Two KV fields this file does not trust
//!
//! `clip.vision.attention.head_count_kv = 2` is real and load-bearing (this
//! tower is GQA, not the plain MHA the SAM/CLIP halves use elsewhere in this
//! crate) - confirmed independently by `attn_k.weight`'s row count (128 = 2 x
//! the 64-wide head). `clip.use_gelu = true`, by contrast, is inert: every
//! block's own tensors are a `ffn_gate`/`ffn_up`/`ffn_down` SwiGLU triple, and
//! nothing in llama.cpp's build for this projector type reads that flag. It
//! is not carried into [`Qwen2EncoderConfig`].
//!
//! Swedish Embedded AB ports vision-language checkpoints between inference
//! stacks for its clients. If your team needs a from-scratch import path for
//! a multimodal GGUF, you can procure our services at info@swedishembedded.com.

use checkpoint::gguf::MmapGguf;
use checkpoint::st::ModelCard;
use serde_json::Value;

use crate::deepseek_ocr_vision::SamConfig;
use crate::import::{self, ImportStats, Mapped};
use crate::kv::ArchKv;

/// The `general.architecture` every mmproj declares.
pub const GGUF_ARCHITECTURE: &str = "clip";
/// The `clip.projector_type` that selects *this* mapping - distinct from the
/// predecessor's `"deepseekocr"`, so the two never collide.
pub const PROJECTOR_TYPE: &str = "deepseekocr2";

/// The Qwen2-shaped resampler that replaces the predecessor's CLIP tower.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2EncoderConfig {
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub ffn_hidden: u32,
    pub layer_norm_eps: f32,
    /// Rows in the query bank paired with a 768x768 local tile.
    pub n_query_local: u32,
    /// Rows in the query bank paired with the 1024x1024 global view.
    pub n_query_global: u32,
}

impl Qwen2EncoderConfig {
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim()
    }
}

/// The whole `projector_type = deepseekocr2` vision tower.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekOcr2VisionConfig {
    pub sam: SamConfig,
    pub encoder: Qwen2EncoderConfig,
    /// The projector's input width - the encoder's own `d_model`, since
    /// there is no second tower to concatenate against here.
    pub projector_in: u32,
    /// The projector's output width: the language model's `d_model`.
    pub projection_dim: u32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

impl DeepseekOcr2VisionConfig {
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

        let e = &self.encoder;
        let (ed, eff, ekv) = (e.d_model as usize, e.ffn_hidden as usize, e.kv_dim() as usize);
        for l in 0..e.n_layers {
            let b = |leaf: &str| format!("vision.encoder.blocks.{l}.{leaf}");
            out.push((b("norm1.weight"), ed));
            out.push((b("norm2.weight"), ed));
            out.push((b("attn.q.weight"), ed * ed));
            out.push((b("attn.q.bias"), ed));
            out.push((b("attn.k.weight"), ekv * ed));
            out.push((b("attn.k.bias"), ekv));
            out.push((b("attn.v.weight"), ekv * ed));
            out.push((b("attn.v.bias"), ekv));
            out.push((b("attn.out.weight"), ed * ed));
            out.push((b("mlp.gate.weight"), eff * ed));
            out.push((b("mlp.up.weight"), eff * ed));
            out.push((b("mlp.down.weight"), ed * eff));
        }
        out.push(("vision.encoder.norm.weight".to_string(), ed));
        out.push(("vision.query_local.weight".to_string(), e.n_query_local as usize * ed));
        out.push(("vision.query_global.weight".to_string(), e.n_query_global as usize * ed));

        let (pin, pout) = (self.projector_in as usize, self.projection_dim as usize);
        out.push(("vision.projector.fc.weight".to_string(), pout * pin));
        out.push(("vision.projector.fc.bias".to_string(), pout));
        out.push(("vision.view_separator".to_string(), pout));
        out
    }

    /// The config as it is stored in the produced checkpoint's header.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "deepseek-ocr2-vision",
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
            "encoder": {
                "d_model": self.encoder.d_model,
                "n_layers": self.encoder.n_layers,
                "n_heads": self.encoder.n_heads,
                "n_kv_heads": self.encoder.n_kv_heads,
                "ffn_hidden": self.encoder.ffn_hidden,
                "layer_norm_eps": self.encoder.layer_norm_eps,
                "n_query_local": self.encoder.n_query_local,
                "n_query_global": self.encoder.n_query_global,
            },
            "projector_in": self.projector_in,
            "projection_dim": self.projection_dim,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        })
    }
}

/// A tensor's leading (torch-outermost) dimension, or a named error.
fn rows(mg: &MmapGguf, name: &str) -> Result<u32, String> {
    let s = mg.shape(name).ok_or_else(|| format!("deepseek-ocr2-vision: missing {name}"))?;
    Ok(*s.first().ok_or_else(|| format!("deepseek-ocr2-vision: {name} is 0-dimensional"))? as u32)
}

/// Derive [`DeepseekOcr2VisionConfig`] from the mmproj's KV, with tensor
/// shapes as the authority wherever the KV is silent or wrong (the
/// feed-forward width, the SAM grid, the neck/compressor channel counts, the
/// query-bank sizes, and which SAM blocks use global attention).
pub fn config_from_gguf(mg: &MmapGguf) -> Result<DeepseekOcr2VisionConfig, String> {
    let root = ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;
    let projector = root.str("projector_type").unwrap_or("");
    if projector != PROJECTOR_TYPE {
        return Err(format!("deepseek-ocr2-vision: expected clip.projector_type={PROJECTOR_TYPE:?}, got {projector:?}"));
    }
    let v = root.scoped("vision");
    let sam_kv = v.scoped("sam");

    let sam_layers = sam_kv.req_u32("block_count")?;
    let sam_d = sam_kv.req_u32("embedding_length")?;
    let sam_heads = sam_kv.req_u32("head_count")?;
    if sam_heads == 0 || sam_d % sam_heads != 0 {
        return Err(format!("deepseek-ocr2-vision: sam embedding_length {sam_d} not divisible by head_count {sam_heads}"));
    }
    let sam_head_dim = sam_d / sam_heads;
    let window_size = v.req_u32("window_size")?;

    // The SAM position embedding is torch [1, grid, grid, d_model].
    let pos = mg.shape("v.sam.pos_embd.weight").ok_or("deepseek-ocr2-vision: missing v.sam.pos_embd.weight")?;
    let grid = *pos.get(1).ok_or("deepseek-ocr2-vision: v.sam.pos_embd.weight is not [1,grid,grid,d]")? as u32;

    // Global vs windowed attention is decided by each block's own relative
    // position table extent - no KV key states it, same as the predecessor.
    let mut global_attn_layers = Vec::new();
    for l in 0..sam_layers {
        let r = rows(mg, &format!("v.sam.blk.{l}.attn.pos_h.weight"))?;
        if r == 2 * grid - 1 {
            global_attn_layers.push(l);
        } else if r != 2 * window_size - 1 {
            return Err(format!(
                "deepseek-ocr2-vision: v.sam.blk.{l}.attn.pos_h.weight has {r} rows, expected {} (windowed) or {} (global)",
                2 * window_size - 1,
                2 * grid - 1
            ));
        }
    }

    let sam = SamConfig {
        d_model: sam_d,
        n_layers: sam_layers,
        n_heads: sam_heads,
        ffn_hidden: rows(mg, "v.sam.blk.0.mlp.lin1.weight")?,
        patch_size: *mg
            .shape("v.sam.patch_embd.weight")
            .and_then(|s| s.get(2))
            .ok_or("deepseek-ocr2-vision: v.sam.patch_embd.weight is not [d,3,k,k]")? as u32,
        grid,
        window_size,
        global_attn_layers,
        neck_channels: rows(mg, "v.sam.neck.0.weight")?,
        compress_mid: rows(mg, "v.sam.net_2.weight")?,
        compress_out: rows(mg, "v.sam.net_3.weight")?,
    };
    if sam.head_dim() != sam_head_dim {
        return Err("deepseek-ocr2-vision: sam head_dim disagreement".to_string());
    }

    let e_layers = v.req_u32("block_count")?;
    let e_d = v.req_u32("embedding_length")?;
    let e_heads = v.req_u32("attention.head_count")?;
    if e_heads == 0 || e_d % e_heads != 0 {
        return Err(format!("deepseek-ocr2-vision: encoder embedding_length {e_d} not divisible by head_count {e_heads}"));
    }
    let e_head_dim = e_d / e_heads;
    let e_kv_dim = rows(mg, "v.blk.0.attn_k.weight")?;
    if e_kv_dim % e_head_dim != 0 {
        return Err(format!("deepseek-ocr2-vision: encoder attn_k width {e_kv_dim} not a multiple of head_dim {e_head_dim}"));
    }
    let encoder = Qwen2EncoderConfig {
        d_model: e_d,
        n_layers: e_layers,
        n_heads: e_heads,
        n_kv_heads: e_kv_dim / e_head_dim,
        // Not `clip.vision.feed_forward_length` - see this module's doc.
        ffn_hidden: rows(mg, "v.blk.0.ffn_up.weight")?,
        layer_norm_eps: v.f32_or("attention.layer_norm_epsilon", 1e-6),
        n_query_local: rows(mg, "v.resample_query_768.weight")?,
        n_query_global: rows(mg, "v.resample_query_1024.weight")?,
    };
    if let Some(declared) = v.u32("attention.head_count_kv") {
        if declared != encoder.n_kv_heads {
            return Err(format!("deepseek-ocr2-vision: attention.head_count_kv={declared} disagrees with attn_k's own width ({} kv heads)", encoder.n_kv_heads));
        }
    }
    if sam.compress_out != encoder.d_model {
        return Err(format!(
            "deepseek-ocr2-vision: SAM's compressor emits {} channels but the encoder expects {}",
            sam.compress_out, encoder.d_model
        ));
    }

    let fc = mg.shape("mm.model.fc.weight").ok_or("deepseek-ocr2-vision: missing mm.model.fc.weight")?;
    let projector_in = *fc.get(1).ok_or("deepseek-ocr2-vision: mm.model.fc.weight is not 2-D")? as u32;
    let projection_dim = fc[0] as u32;
    if projector_in != encoder.d_model {
        return Err(format!("deepseek-ocr2-vision: projector input {projector_in} != encoder d_model {}", encoder.d_model));
    }

    Ok(DeepseekOcr2VisionConfig {
        sam,
        encoder,
        projector_in,
        projection_dim,
        image_mean: v.f32_array("image_mean").unwrap_or_else(|| vec![0.5; 3]),
        image_std: v.f32_array("image_std").unwrap_or_else(|| vec![0.5; 3]),
    })
}

/// Classify one mmproj tensor name.
pub fn classify(name: &str, cfg: &DeepseekOcr2VisionConfig) -> Result<Mapped, String> {
    let simple = |s: &str| Ok(Mapped::Simple(s.to_string()));
    match name {
        "v.view_seperator" => return simple("vision.view_separator"),
        "mm.model.fc.weight" => return simple("vision.projector.fc.weight"),
        "mm.model.fc.bias" => return simple("vision.projector.fc.bias"),
        "v.post_ln.weight" => return simple("vision.encoder.norm.weight"),
        "v.resample_query_768.weight" => return simple("vision.query_local.weight"),
        "v.resample_query_1024.weight" => return simple("vision.query_global.weight"),
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
        let (l, leaf) = split_block(rest, name, cfg.encoder.n_layers)?;
        let b = |s: &str| Mapped::Simple(format!("vision.encoder.blocks.{l}.{s}"));
        return match leaf {
            "ln1.weight" => Ok(b("norm1.weight")),
            "ln2.weight" => Ok(b("norm2.weight")),
            "attn_q.weight" => Ok(b("attn.q.weight")),
            "attn_q.bias" => Ok(b("attn.q.bias")),
            "attn_k.weight" => Ok(b("attn.k.weight")),
            "attn_k.bias" => Ok(b("attn.k.bias")),
            "attn_v.weight" => Ok(b("attn.v.weight")),
            "attn_v.bias" => Ok(b("attn.v.bias")),
            "attn_out.weight" => Ok(b("attn.out.weight")),
            "ffn_gate.weight" => Ok(b("mlp.gate.weight")),
            "ffn_up.weight" => Ok(b("mlp.up.weight")),
            "ffn_down.weight" => Ok(b("mlp.down.weight")),
            other => Err(format!("unrecognized encoder block leaf {other:?} in {name:?}")),
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

/// Import a DeepSeek-OCR-2 mmproj GGUF into brain's native format.
pub fn import(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let cfg = config_from_gguf(mg)?;
    let params = cfg.param_list();

    let mut card = ModelCard::new(id_override.unwrap_or("deepseek-ocr2-vision"), "deepseek-ocr2");
    card.param_count = Some(params.iter().map(|(_, n)| *n as u64).sum());

    import::to_st(mg, &params, &|n| classify(n, &cfg), out_path, &cfg.to_json(), Some(&card), "deepseek-ocr2-vision")
}
