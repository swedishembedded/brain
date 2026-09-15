// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full DaViT forward: chains the 4 stages' patch embeds and `depths[i]`
//! `(SpatialBlock, ChannelBlock)` pairs into one `forward_features_unpool`,
//! matching the reference's own `DaViT.forward_features_unpool`. Every
//! piece this composes (`PatchEmbed`, `SpatialBlock`, `ChannelBlock`) is
//! independently verified against the real checkpoint at cosine >=0.999.

use gpu_core::{DeviceBuffer, Gpu};
use vision::net::Shape;

use super::block::{ChannelBlock, ChannelBlockKernelIds, SpatialBlock, SpatialBlockKernelIds};
use super::config::DavitConfig;
use super::patch_embed::{PatchEmbed, PatchEmbedKernelIds};

pub struct DavitKernelIds {
    pub patch: PatchEmbedKernelIds,
    pub spatial: SpatialBlockKernelIds,
    pub channel: ChannelBlockKernelIds,
}

impl DavitKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> DavitKernelIds {
        DavitKernelIds {
            patch: PatchEmbedKernelIds::resolve(pipelines),
            spatial: SpatialBlockKernelIds::resolve(pipelines),
            channel: ChannelBlockKernelIds::resolve(pipelines),
        }
    }
}

struct Stage {
    patch: PatchEmbed,
    pairs: Vec<(SpatialBlock, ChannelBlock)>,
    grid_h: u32,
    grid_w: u32,
    dim: u32,
}

pub struct Davit {
    stages: Vec<Stage>,
}

impl Davit {
    /// `mlp_ratio`: DaViT's constant `4` (the reference never varies it per
    /// stage). `window_size`: shared across every stage (`cfg.window_size`).
    pub fn new(gpu: &Gpu, k: &DavitKernelIds, prefix: &str, cfg: &DavitConfig, mlp_ratio: u32, eps: f32, train: bool) -> Davit {
        let mut stages = Vec::with_capacity(cfg.stages.len());
        for (i, spec) in cfg.stages.iter().enumerate() {
            let in_shape = Shape::new(1, spec.dim_in, spec.in_hw.0, spec.in_hw.1);
            let patch = PatchEmbed::new(gpu, &k.spatial.conv_ids, &format!("{prefix}.convs.{i}"), in_shape, &spec.patch, spec.dim_out, eps, train);

            let mut pairs = Vec::with_capacity(spec.depth as usize);
            for p in 0..spec.depth {
                let pair_prefix = format!("{prefix}.blocks.{i}.{p}");
                let spatial = SpatialBlock::new(gpu, &k.spatial, &format!("{pair_prefix}.spatial_block"), spec.dim_out, spec.num_heads, cfg.window_size, spec.out_hw.0, spec.out_hw.1, mlp_ratio, eps, train);
                let channel = ChannelBlock::new(gpu, &k.channel, &format!("{pair_prefix}.channel_block"), spec.dim_out, spec.num_groups, spec.out_hw.0, spec.out_hw.1, mlp_ratio, eps, train);
                pairs.push((spatial, channel));
            }

            stages.push(Stage { patch, pairs, grid_h: spec.out_hw.0, grid_w: spec.out_hw.1, dim: spec.dim_out });
        }
        Davit { stages }
    }

    /// `pixel_values`: `[1,3,H,W]` NCHW, `H=W=` the config's image size.
    /// Returns the last stage's `[576,1024]`-shaped (for Florence-2-base)
    /// sequence output - the reference's `forward_features_unpool`.
    pub fn forward_features_unpool<'a>(&'a self, gpu: &Gpu, k: &DavitKernelIds, ps: &paramstore::ParamStore, pixel_values: &'a DeviceBuffer) -> &'a DeviceBuffer {
        let mut cur: &DeviceBuffer = pixel_values;
        for (i, stage) in self.stages.iter().enumerate() {
            let mut x = if i == 0 {
                stage.patch.forward_from_pixels(gpu, &k.patch, ps, cur)
            } else {
                let prev = &self.stages[i - 1];
                let prev_shape = Shape::new(1, prev.dim, prev.grid_h, prev.grid_w);
                stage.patch.forward_from_sequence(gpu, &k.patch, ps, cur, prev_shape)
            };
            for (spatial, channel) in &stage.pairs {
                x = spatial.forward(gpu, &k.spatial, ps, x);
                x = channel.forward(gpu, &k.channel, ps, x);
            }
            cur = x;
        }
        cur
    }
}
