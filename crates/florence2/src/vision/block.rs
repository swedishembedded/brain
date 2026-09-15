// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `SpatialBlock`: `conv1` (dwconv residual, if `conv_at_attn`) -> window
//! attention -> `conv2` (dwconv residual, if `conv_at_ffn`) -> MLP. Every
//! Florence2-base stage has both flags `true` (the reference's default),
//! so this always runs all four sublayers - see `vision::config`'s module
//! doc.

use gpu_core::Gpu;
use vision::ids::ConvKernelIds;
use vision::net::Shape;

use super::channel_attn::{ChannelAttn, ChannelAttnKernelIds};
use super::dwconv::DwConvResidual;
use super::mlp::{Mlp, MlpKernelIds};
use super::window_attn::{WindowAttn, WindowAttnKernelIds};

pub struct SpatialBlockKernelIds {
    pub conv_ids: ConvKernelIds,
    pub window: WindowAttnKernelIds,
    pub mlp: MlpKernelIds,
    pub nchw_nlc: usize,
    pub nlc_nchw: usize,
    pub add2: usize,
}

impl SpatialBlockKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> SpatialBlockKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        SpatialBlockKernelIds {
            conv_ids: ConvKernelIds::resolve(pipelines),
            window: WindowAttnKernelIds::resolve(pipelines),
            mlp: MlpKernelIds::resolve(pipelines),
            nchw_nlc: k("nchw_nlc"),
            nlc_nchw: k("nlc_nchw"),
            add2: k("add2"),
        }
    }
}

pub struct SpatialBlock {
    conv1: DwConvResidual,
    window_attn: WindowAttn,
    conv2: DwConvResidual,
    mlp: Mlp,
    dim: u32,
    heads: u32,
    rows: u32,
}

impl SpatialBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &Gpu, k: &SpatialBlockKernelIds, prefix: &str, dim: u32, heads: u32, window_size: u32, grid_h: u32, grid_w: u32, mlp_ratio: u32, eps: f32, train: bool) -> SpatialBlock {
        let shape = Shape::new(1, dim, grid_h, grid_w);
        let rows = grid_h * grid_w;
        SpatialBlock {
            conv1: DwConvResidual::new(gpu, &k.conv_ids, &format!("{prefix}.conv1.fn"), shape, train),
            window_attn: WindowAttn::new(gpu, &format!("{prefix}.window_attn"), dim, heads, window_size, grid_h, grid_w, eps),
            conv2: DwConvResidual::new(gpu, &k.conv_ids, &format!("{prefix}.conv2.fn"), shape, train),
            mlp: Mlp::new(gpu, &format!("{prefix}.ffn"), dim, dim * mlp_ratio, rows, eps),
            dim,
            heads,
            rows,
        }
    }

    pub fn forward<'a>(&'a self, gpu: &Gpu, k: &SpatialBlockKernelIds, ps: &paramstore::ParamStore, x_in: &'a gpu_core::DeviceBuffer) -> &'a gpu_core::DeviceBuffer {
        let x1 = self.conv1.forward(gpu, k.nchw_nlc, k.nlc_nchw, k.add2, ps, x_in);
        let x2 = self.window_attn.forward(gpu, &k.window, ps, x1, self.rows, self.heads);
        let x3 = self.conv2.forward(gpu, k.nchw_nlc, k.nlc_nchw, k.add2, ps, x2);
        self.mlp.forward(gpu, &k.mlp, ps, x3, self.rows)
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }
}

pub struct ChannelBlockKernelIds {
    pub conv_ids: ConvKernelIds,
    pub channel: ChannelAttnKernelIds,
    pub mlp: MlpKernelIds,
    pub nchw_nlc: usize,
    pub nlc_nchw: usize,
    pub add2: usize,
}

impl ChannelBlockKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> ChannelBlockKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        ChannelBlockKernelIds {
            conv_ids: ConvKernelIds::resolve(pipelines),
            channel: ChannelAttnKernelIds::resolve(pipelines),
            mlp: MlpKernelIds::resolve(pipelines),
            nchw_nlc: k("nchw_nlc"),
            nlc_nchw: k("nlc_nchw"),
            add2: k("add2"),
        }
    }
}

/// `ChannelBlock`: same `conv1 -> attn -> conv2 -> MLP` shape as
/// `SpatialBlock`, with channel attention (`ChannelAttn`) in place of
/// window attention.
pub struct ChannelBlock {
    conv1: DwConvResidual,
    channel_attn: ChannelAttn,
    conv2: DwConvResidual,
    mlp: Mlp,
    dim: u32,
    rows: u32,
}

impl ChannelBlock {
    /// `n`: spatial token count - see [`ChannelAttn::new`]'s doc on the
    /// caller's obligation to pre-scale the channel-attn qkv weight's Q
    /// rows by `n^-0.5` before this is constructed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &Gpu, k: &ChannelBlockKernelIds, prefix: &str, dim: u32, groups: u32, grid_h: u32, grid_w: u32, mlp_ratio: u32, eps: f32, train: bool) -> ChannelBlock {
        let shape = Shape::new(1, dim, grid_h, grid_w);
        let rows = grid_h * grid_w;
        ChannelBlock {
            conv1: DwConvResidual::new(gpu, &k.conv_ids, &format!("{prefix}.conv1.fn"), shape, train),
            channel_attn: ChannelAttn::new(gpu, &format!("{prefix}.channel_attn"), dim, groups, rows, eps),
            conv2: DwConvResidual::new(gpu, &k.conv_ids, &format!("{prefix}.conv2.fn"), shape, train),
            mlp: Mlp::new(gpu, &format!("{prefix}.ffn"), dim, dim * mlp_ratio, rows, eps),
            dim,
            rows,
        }
    }

    pub fn forward<'a>(&'a self, gpu: &Gpu, k: &ChannelBlockKernelIds, ps: &paramstore::ParamStore, x_in: &'a gpu_core::DeviceBuffer) -> &'a gpu_core::DeviceBuffer {
        let x1 = self.conv1.forward(gpu, k.nchw_nlc, k.nlc_nchw, k.add2, ps, x_in);
        let x2 = self.channel_attn.forward(gpu, &k.channel, ps, x1);
        let x3 = self.conv2.forward(gpu, k.nchw_nlc, k.nlc_nchw, k.add2, ps, x2);
        self.mlp.forward(gpu, &k.mlp, ps, x3, self.rows)
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }
}
