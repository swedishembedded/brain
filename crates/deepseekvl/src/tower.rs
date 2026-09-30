// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vision towers and their aligners: an image's `pixel_values` in, the
//! decoder-width rows that replace its placeholder out ([`VisionTower`]).
//!
//! [`HybridTower`] is DeepSeek-VL's:
//!
//! * High resolution: SAM-B over the whole `[3, 1024, 1024]` square (with the
//!   neck resize and HD branch, [`sam1::SamViTConfig::deepseek_vl`]), giving a
//!   `[1024, 24, 24]` map read as 576 rows.
//! * Low resolution: SigLIP-L over the square resized to 384, giving 576 rows.
//! * The split aligner projects each stream to half the decoder width,
//!   concatenates them (high first) and runs its GELU stack.
//!
//! The three stages share one device; the features cross between them
//! through the host, which is two `[576, 1024]` reads per image.
//!
//! [`SiglipTower`] is Janus-Pro's understanding tower: SigLIP-L over the
//! processor's already normalized 384-pixel square, then a plain `mlp_gelu`
//! aligner.

use checkpoint::weightio::WeightReader;
use clip::config::ClipVisionConfig;
use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};
use gpu_core::{DeviceBuffer, Gpu};
use model::projector::{MlpProjector, ProjectorConfig, PROJECTOR_PIPELINES};
use sam1::{SamEncoder, SamViTConfig};

use crate::config::DeepseekVlConfig;
use crate::preprocess;

/// One image through a tower: each feature stream the aligner reads (the
/// hybrid tower's high then low branch, or the single tower's one) and the
/// aligner's output.
pub struct Features {
    /// `[rows, width]` per stream.
    pub streams: Vec<Vec<f32>>,
    /// `[rows, n_embed]`: what the decoder reads.
    pub embeds: Vec<f32>,
}

/// A vision tower with its aligner.
pub trait VisionTower {
    /// Image rows per image.
    fn rows(&self) -> usize;
    /// The side of the square `pixel_values` the tower takes.
    fn image_size(&self) -> usize;
    /// Run one image's `pixel_values` (`[3, S, S]`, from
    /// [`preprocess::ImageProcessor::pixel_values`]).
    fn encode(&self, pixel_values: &[f32]) -> Features;
}

pub struct HybridTower {
    cfg: DeepseekVlConfig,
    sam: SamEncoder,
    siglip: ClipVision,
    gpu: Gpu,
    aligner: MlpProjector,
    high_in: DeviceBuffer,
    low_in: DeviceBuffer,
    rows: u32,
}

impl HybridTower {
    /// Build the tower from a DeepSeek-VL checkpoint on the ambient device.
    pub fn load(rd: &WeightReader, cfg: &DeepseekVlConfig) -> Result<HybridTower, String> {
        let sam_cfg = SamViTConfig::deepseek_vl();
        let siglip_cfg = ClipVisionConfig::siglip_large_patch16_384();
        for (branch, name, size) in [(&cfg.high, "sam_b_downsample", sam_cfg.image_h()), (&cfg.low, "siglip_large_patch16_384", siglip_cfg.image_size())] {
            if branch.model_name != name || branch.image_size != size || branch.select_layer != -1 {
                return Err(format!("vision branch {} at {} px (select_layer {}) is not the {name} tower at {size} px this crate builds", branch.model_name, branch.image_size, branch.select_layer));
            }
        }
        let rows = siglip_cfg.native_patches();
        let (gh, gw) = sam_cfg.compress_grid();
        if gh * gw != rows || sam_cfg.compress_out != cfg.high.output_dim || siglip_cfg.d_model() != cfg.low.output_dim {
            return Err(format!("the branches disagree: SAM gives {gh}x{gw}x{}, SigLIP {rows}x{}", sam_cfg.compress_out, siglip_cfg.d_model()));
        }
        let pcfg = ProjectorConfig::from_type(&cfg.aligner.projector_type, cfg.aligner.depth, cfg.aligner.input_dim, cfg.aligner.n_embed)?;
        let aligner_w = crate::import::aligner_weights(rd, crate::import::ALIGNER_PREFIX, &pcfg)?;
        let (siglip_w, _) = clip::import::siglip::import_timm(rd, clip::import::siglip::DEEPSEEK_VL_LOW_PREFIX, &siglip_cfg)?;
        let sam_src = sam1::hf::source(rd, rd.names(), crate::import::SAM_PREFIX, &sam_cfg, sam1::hf::Spelling::DeepseekVl)?;

        let gpu = Gpu::new(PROJECTOR_PIPELINES);
        let sam = SamEncoder::new_inference(gpu.new_like(sam1::PIPELINES), sam_cfg, &sam_src, 0);
        let siglip = ClipVision::new_on(gpu.new_like(CLIP_VISION_PIPELINES), siglip_cfg, 1, PatchSource::Pixels, &siglip_w);
        let aligner = MlpProjector::new(&gpu, pcfg, rows, &aligner_w)?;
        let high_in = gpu.storage((rows * cfg.high.output_dim) as u64);
        let low_in = gpu.storage((rows * cfg.low.output_dim) as u64);
        Ok(HybridTower { cfg: cfg.clone(), sam, siglip, gpu, aligner, high_in, low_in, rows })
    }

}

impl VisionTower for HybridTower {
    fn rows(&self) -> usize {
        self.rows as usize
    }

    fn image_size(&self) -> usize {
        self.cfg.high.image_size as usize
    }

    fn encode(&self, pixel_values: &[f32]) -> Features {
        let (high_cfg, low_cfg) = (&self.cfg.high, &self.cfg.low);
        let side = high_cfg.image_size as usize;
        assert_eq!(pixel_values.len(), 3 * side * side, "pixel_values is not [3, {side}, {side}]");
        let mut low_px = preprocess::low_resolution(pixel_values, side, low_cfg.image_size as usize);
        preprocess::normalize(&mut low_px, low_cfg.pixel_mean, low_cfg.pixel_std);
        let mut high_px = pixel_values.to_vec();
        preprocess::normalize(&mut high_px, high_cfg.pixel_mean, high_cfg.pixel_std);

        self.sam.write_image(&high_px);
        self.sam.run();
        let (rows, c) = (self.rows as usize, high_cfg.output_dim as usize);
        // `[C, H, W]` -> `[H*W, C]`.
        let map = self.sam.gpu.read(self.sam.output(), self.sam.out_len());
        let mut high = vec![0f32; rows * c];
        for ch in 0..c {
            for r in 0..rows {
                high[r * c + ch] = map[ch * rows + r];
            }
        }
        self.siglip.set_pixels(&low_px);
        self.siglip.forward();
        let low = self.siglip.read_output();

        self.gpu.write_f32(&self.high_in, &high);
        self.gpu.write_f32(&self.low_in, &low);
        self.gpu.submit(&[], &self.aligner.forward(&self.gpu, &[&self.high_in, &self.low_in]));
        let embeds = self.gpu.read(self.aligner.out(), rows * self.cfg.aligner.n_embed as usize);
        Features { streams: vec![high, low], embeds }
    }
}

/// SigLIP-L over the processor's normalized square, then a single-stream
/// aligner.
pub struct SiglipTower {
    siglip: ClipVision,
    gpu: Gpu,
    aligner: MlpProjector,
    features: DeviceBuffer,
    rows: u32,
    width: u32,
    n_embed: u32,
    image_size: u32,
}

impl SiglipTower {
    /// The tower under `tower_prefix` and the aligner under `aligner_prefix`
    /// of `rd`, the aligner described by `aligner`.
    pub fn load(rd: &WeightReader, tower_prefix: &str, aligner_prefix: &str, aligner: &crate::AlignerConfig) -> Result<SiglipTower, String> {
        let cfg = ClipVisionConfig::siglip_large_patch16_384();
        let pcfg = ProjectorConfig::from_type(&aligner.projector_type, aligner.depth, aligner.input_dim, aligner.n_embed)?;
        if pcfg.inputs() != 1 || pcfg.input_dim != cfg.d_model() {
            return Err(format!("a {} aligner over {} inputs does not read one {}-wide SigLIP stream", aligner.projector_type, aligner.input_dim, cfg.d_model()));
        }
        let aligner_w = crate::import::aligner_weights(rd, aligner_prefix, &pcfg)?;
        let (siglip_w, _) = clip::import::siglip::import_timm(rd, tower_prefix, &cfg)?;
        let (rows, width, image_size) = (cfg.native_patches(), cfg.d_model(), cfg.image_size());
        let gpu = Gpu::new(PROJECTOR_PIPELINES);
        let siglip = ClipVision::new_on(gpu.new_like(CLIP_VISION_PIPELINES), cfg, 1, PatchSource::Pixels, &siglip_w);
        let proj = MlpProjector::new(&gpu, pcfg, rows, &aligner_w)?;
        let features = gpu.storage((rows * width) as u64);
        Ok(SiglipTower { siglip, gpu, aligner: proj, features, rows, width, n_embed: pcfg.n_embed, image_size })
    }
}

impl VisionTower for SiglipTower {
    fn rows(&self) -> usize {
        self.rows as usize
    }

    fn image_size(&self) -> usize {
        self.image_size as usize
    }

    fn encode(&self, pixel_values: &[f32]) -> Features {
        let side = self.image_size as usize;
        assert_eq!(pixel_values.len(), 3 * side * side, "pixel_values is not [3, {side}, {side}]");
        self.siglip.set_pixels(pixel_values);
        self.siglip.forward();
        let feats = self.siglip.read_output();
        debug_assert_eq!(feats.len(), (self.rows * self.width) as usize);
        self.gpu.write_f32(&self.features, &feats);
        self.gpu.submit(&[], &self.aligner.forward(&self.gpu, &[&self.features]));
        let embeds = self.gpu.read(self.aligner.out(), (self.rows * self.n_embed) as usize);
        Features { streams: vec![feats], embeds }
    }
}
