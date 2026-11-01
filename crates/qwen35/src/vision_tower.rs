// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the LLMs its clients serve on their own GPUs. If your team needs
// expertise in serving vision-language checkpoints at their published precision then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The vision tower of a released Qwen3.8 checkpoint, loaded for serving: the `model.visual.*` tensors of an HF
//! directory (all bf16 in the FP8 and NVFP4 releases), the checkpoint's own `preprocessor_config.json`, and a
//! device handle on the decoder's card.
//!
//! The tower is `qwen3vl`'s ViT encoder and patch merger, reused unchanged (parity-gated at the real dims against
//! the reference in `tests/vision_parity.rs`, and against real weights in `tests/hf_fp8_vision_real.rs`). This
//! checkpoint's vision config differs from Qwen3-VL's in two fields only: the merger projects to the decoder width
//! and no DeepStack tap feeds the decoder's early layers.
//!
//! One image becomes `grid_h * grid_w` rows of the decoder's width, in raster order of the merged grid: the rows
//! that stand at the image's placeholder tokens ([`crate::vision_frontend`]).

use std::path::Path;

use gpu_core::Gpu;
use imaging::pixels::Rgb8;
use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use qwen3vl::preprocess::ImageGeometry;
use serde_json::Value;

/// The most pixels an image is served at: larger ones are scaled down to it (the checkpoint's own bound is 16.7 MP, whose
/// 65536 patches would hold more than the cards can spare beside the decoder). Four megapixels is 4096 decoder rows.
pub const SERVING_MAX_IMAGE_PIXELS: u32 = 4 << 20;

/// Device bytes the tower takes beside the decoder: its weights as f32 (0.45 B parameters) and the activations of an
/// image of [`SERVING_MAX_IMAGE_PIXELS`] (16384 patches of 1152 features through the widest block).
pub const VISION_RESERVE_BYTES: u64 = 4 << 30;

/// The tensors a checkpoint's vision tower lives under.
const VISUAL_PREFIX: &str = "model.visual.";

/// The token ids a checkpoint's config assigns to an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionTokens {
    /// One of these stands at every row an image contributes.
    pub image: u32,
    pub start: u32,
    pub end: u32,
}

/// A checkpoint's vision settings, read from its directory without loading any weights.
#[derive(Clone, Debug)]
pub struct VisionSpec {
    pub config: VisionConfig,
    pub geometry: ImageGeometry,
    pub tokens: VisionTokens,
}

/// Why a directory has no vision, or the spec it has. `Ok(None)` is a text-only release (no `vision_config` in its
/// config, no `model.visual.*` tensors); a release that declares vision and cannot deliver it is an error.
pub fn read_spec(dir: &Path) -> Result<Option<VisionSpec>, String> {
    let config: Value = match std::fs::read(dir.join("config.json")) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| format!("config.json: {e}"))?,
        Err(_) => return Ok(None),
    };
    let Some(vc) = config.get("vision_config").filter(|v| v.is_object()) else { return Ok(None) };
    let has_tensors = checkpoint::weightio::WeightReader::open_hf_dir(dir).map(|r| r.names().any(|n| n.starts_with(VISUAL_PREFIX))).unwrap_or(false);
    if !has_tensors {
        return Ok(None);
    }
    for key in ["depth", "hidden_size", "num_heads", "intermediate_size", "patch_size", "temporal_patch_size", "spatial_merge_size", "num_position_embeddings", "out_hidden_size", "in_channels"] {
        if vc.get(key).and_then(Value::as_u64).filter(|&n| n > 0).is_none() {
            return Err(format!("config.json: vision_config.{key} must be a positive integer"));
        }
    }
    let mut vc = vc.clone();
    // This checkpoint family has no DeepStack; the key is absent from its config.
    vc["deepstack_visual_indexes"] = vc.get("deepstack_visual_indexes").cloned().unwrap_or_else(|| Value::Array(Vec::new()));
    let config_v = VisionConfig::from_hf(&vc);
    if !config_v.deepstack_indexes.is_empty() {
        return Err("config.json: vision_config.deepstack_visual_indexes must be empty: this decoder takes no DeepStack features".to_string());
    }
    let pre = std::fs::read(dir.join("preprocessor_config.json")).map_err(|e| format!("preprocessor_config.json: {e}"))?;
    let geometry = ImageGeometry::from_json(&serde_json::from_slice(&pre).map_err(|e| format!("preprocessor_config.json: {e}"))?)?;
    if (geometry.patch_size, geometry.merge_size, geometry.temporal_patch_size) != (config_v.patch_size, config_v.spatial_merge_size, config_v.temporal_patch_size) {
        return Err("preprocessor_config.json: patch, merge and temporal sizes disagree with config.json's vision_config".to_string());
    }
    let id = |key: &str| config.get(key).and_then(Value::as_u64).filter(|&n| n <= u32::MAX as u64).map(|n| n as u32).ok_or_else(|| format!("config.json: '{key}' must be a token id"));
    let tokens = VisionTokens { image: id("image_token_id")?, start: id("vision_start_token_id")?, end: id("vision_end_token_id")? };
    Ok(Some(VisionSpec { config: config_v, geometry, tokens }))
}

/// One image's rows for the decoder.
pub struct ImageRows {
    /// `[rows, d_model]`, raster order of the merged grid.
    pub rows: Vec<f32>,
    /// The merged grid, `(height, width)` in rows.
    pub grid: (u32, u32),
}

/// The vision tower on a card: weights uploaded once, shared by every request.
pub struct VisionTower {
    gpu: Gpu,
    encoder: VisionEncoder,
    merger: PatchMerger,
    spec: VisionSpec,
}

impl VisionTower {
    /// Load the tower of the checkpoint in `dir` onto the card `decoder` runs on. `d_model` is the decoder width the
    /// merger must project to; `max_pixels` lowers the checkpoint's own upper bound on an image's size (a serving
    /// limit: an image is scaled down to it, never refused for it).
    pub fn open(dir: &Path, spec: VisionSpec, decoder: &Gpu, d_model: u32, max_pixels: u32) -> Result<VisionTower, String> {
        if spec.config.out_hidden_size != d_model {
            return Err(format!("the vision merger projects to {} but the decoder is {d_model} wide", spec.config.out_hidden_size));
        }
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let weights = qwen3vl::import::load_vision(&reader, &spec.config)?;
        // Its own handle on the decoder's device, holding its freed blocks: on CUDA a free waits for every stream of
        // the process, so a tower that freed its temporaries per image would stall the decoder.
        let gpu = decoder.new_like(vision_pipelines());
        gpu.hold_freed_blocks(true);
        let encoder = VisionEncoder::new(&gpu, spec.config.clone(), &weights.vision);
        let merger = PatchMerger::new(&gpu, &weights.merger, spec.config.hidden, spec.config.spatial_merge_size, d_model, false);
        let spec = VisionSpec { geometry: spec.geometry.capped_at(max_pixels), ..spec };
        Ok(VisionTower { gpu, encoder, merger, spec })
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.spec
    }

    /// The decoder tokens an image of `height x width` pixels becomes, and the merged grid they form.
    pub fn tokens_for(&self, height: u32, width: u32) -> (u32, (u32, u32)) {
        self.spec.geometry.tokens(height, width)
    }

    /// Preprocess `img` as the checkpoint's processor does and run it through the tower and the merger.
    pub fn encode(&self, img: &Rgb8) -> ImageRows {
        let p = self.spec.geometry.patches(img);
        let g = &self.spec.geometry;
        let (gh, gw) = (p.h_bar / g.patch_size, p.w_bar / g.patch_size);
        let features = self.encoder.encode(&self.gpu, gh, gw, &p.pixels);
        let rows = self.merger.merge(&self.gpu, &features, gh * gw);
        ImageRows { rows, grid: (gh / g.merge_size, gw / g.merge_size) }
    }

    /// The tower's patch features before the merger (`[patches, hidden]`) for a prepared patch tensor: the stage the
    /// reference parity test compares.
    pub fn features(&self, grid_h: u32, grid_w: u32, pixels: &[f32]) -> Vec<f32> {
        self.encoder.encode(&self.gpu, grid_h, grid_w, pixels)
    }
}
