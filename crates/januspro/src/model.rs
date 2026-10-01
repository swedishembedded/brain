// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro's understanding path: DeepSeek-VL's composite with one SigLIP-L
//! tower, a plain `mlp_gelu` aligner, the `<|User|>`/`<|Assistant|>` roles,
//! and each image wrapped in `<begin_of_image>`/`<end_of_image>`.

use std::path::Path;

use checkpoint::weightio::WeightReader;
use deepseekvl::model::{Footprint, Parts, Placement, Vlm};
use deepseekvl::tower::SiglipTower;

use crate::config::JanusProConfig;

/// The understanding tower.
pub const TOWER_PREFIX: &str = clip::import::siglip::JANUS_PREFIX;
/// The VQ-16 image tokenizer (`brain-vqgan`'s LlamaGen schedule).
pub const VQ_PREFIX: &str = "gen_vision_model.";
/// The tags around each image in the token stream.
pub const IMAGE_START: &str = "<begin_of_image>";
pub const IMAGE_END: &str = "<end_of_image>";

/// Every component prefix of a Janus-Pro checkpoint.
pub const COMPONENTS: [&str; 7] = [
    TOWER_PREFIX,
    deepseekvl::import::ALIGNER_PREFIX,
    crate::gen::GEN_HEAD_PREFIX,
    crate::gen::GEN_ALIGNER_PREFIX,
    crate::gen::GEN_EMBED,
    VQ_PREFIX,
    "language_model.",
];

/// Open `dir` and check that every tensor belongs to a component.
pub fn open(dir: &Path) -> Result<(JanusProConfig, WeightReader), String> {
    let cfg = JanusProConfig::from_dir(dir)?;
    if cfg.vision.model_name != "siglip_large_patch16_384" {
        return Err(format!("understanding tower {} is not the SigLIP-L/16@384 this crate builds", cfg.vision.model_name));
    }
    let rd = WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    deepseekvl::import::check_coverage(&rd, &COMPONENTS)?;
    Ok((cfg, rd))
}

/// Janus-Pro's understanding tower on its device: SigLIP-L's and the
/// aligner's fp32 weights with the activations of one 384-pixel image,
/// measured on the released checkpoint.
pub const SIGLIP_TOWER_BYTES: u64 = 13 << 28;

/// The decoder's shape, from the checkpoint's own config.
pub fn decoder_config(dir: &Path) -> Result<qwen3::QwenConfig, String> {
    qwen3::hf::decoder_config_as(&JanusProConfig::from_dir(dir)?.language.to_string(), "llama")
}

/// The understanding build's footprint.
pub fn understanding_footprint(dir: &Path) -> Result<Footprint, String> {
    Ok(Footprint::of(&decoder_config(dir)?, SIGLIP_TOWER_BYTES))
}

/// Load the understanding path from `dir` with the decoder at `dtype`
/// (the checkpoint's own is bf16), its KV cache sized for `ctx` tokens, on the
/// ambient device.
pub fn load_understanding(dir: &Path, dtype: qwen3::Dtype, ctx: u32) -> Result<Vlm, String> {
    let (cfg, rd) = open(dir)?;
    let tower = Box::new(SiglipTower::load(&rd, TOWER_PREFIX, deepseekvl::import::ALIGNER_PREFIX, &cfg.aligner)?);
    Vlm::assemble(dir, &rd, Parts { tower, style: deepseekvl::prompt::JANUS, wrap: Some((IMAGE_START, IMAGE_END)), language: cfg.language }, dtype, ctx, None, false)
}

/// [`load_understanding`] as `placement` says.
pub fn load_understanding_placed(dir: &Path, dtype: qwen3::Dtype, placement: Placement, adapter: Option<&Path>) -> Result<Vlm, String> {
    let (cfg, rd) = open(dir)?;
    let tower = gpu_core::devices::with_gpu(placement.tower, || SiglipTower::load(&rd, TOWER_PREFIX, deepseekvl::import::ALIGNER_PREFIX, &cfg.aligner))??;
    let parts = Parts { tower: Box::new(tower), style: deepseekvl::prompt::JANUS, wrap: Some((IMAGE_START, IMAGE_END)), language: cfg.language };
    gpu_core::devices::with_gpu(placement.decoder, || Vlm::assemble(dir, &rd, parts, dtype, placement.context, adapter, true))?
}
