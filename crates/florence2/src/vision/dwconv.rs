// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The residual depthwise-conv insertion every DaViT spatial/channel block
//! carries (`conv1`/`conv2` in the reference's `SpatialBlock`/`ChannelBlock`,
//! per-block rather than once-per-stage the way `fastvlm::encoder::repcpe`
//! is - see `vision::config`'s module doc for why): `x = x + dwconv3x3(x)`,
//! no norm, in NCHW space with a sequence round trip either side (the block
//! itself stays in `[B,N,C]` sequence form throughout).

use gpu_core::{DeviceBuffer, Gpu, Step};
use vision::blocks::{Act, Conv, ConvNames, ConvSpec, Norm};
use vision::ids::ConvKernelIds;
use vision::net::{Ctx, Shape};

pub struct DwConvResidual {
    conv: Conv,
    conv_ids: ConvKernelIds,
    nchw_in: DeviceBuffer,
    conv_seq: DeviceBuffer,
    out: DeviceBuffer,
    shape: Shape,
}

impl DwConvResidual {
    /// `shape`: NCHW shape of the sequence this operates on (`shape.h*w`
    /// must equal the caller's row count).
    pub fn new(gpu: &Gpu, conv_ids: &ConvKernelIds, prefix: &str, shape: Shape, train: bool) -> DwConvResidual {
        let ctx = Ctx::new(gpu, conv_ids);
        let spec = ConvSpec::depthwise(shape.c, 3, 1, 1, Act::None).with_norm(Norm::None).with_bias();
        let names = ConvNames {
            bias: format!("{prefix}.dw.bias"),
            weight: format!("{prefix}.dw.weight"),
            gamma: String::new(),
            beta: String::new(),
            run_mean: String::new(),
            run_var: String::new(),
        };
        let conv = Conv::with_names(&ctx, &format!("{prefix}.dw"), names, shape, spec, train);
        let numel = shape.numel() as u64;
        DwConvResidual { conv_ids: *conv_ids, conv, nchw_in: gpu.storage(numel), conv_seq: gpu.storage(numel), out: gpu.storage(numel), shape }
    }

    /// `x_in`: `[rows, C]` sequence. Returns `x_in + dwconv3x3(x_in)`, also
    /// `[rows, C]`.
    pub fn forward(&self, gpu: &Gpu, nchw_nlc: usize, nlc_nchw: usize, add2: usize, ps: &paramstore::ParamStore, x_in: &DeviceBuffer) -> &DeviceBuffer {
        let rows = self.shape.h * self.shape.w;
        let total = rows * self.shape.c;

        let mut s: Vec<Step> = vec![gpu.step(nlc_nchw, &[x_in, &self.nchw_in], &[total, self.shape.c, rows], total)];
        gpu.submit(&[], &s);

        let ctx = Ctx::new(gpu, &self.conv_ids);
        self.conv.forward(&ctx, ps, &self.nchw_in);
        let conv_out = self.conv.out();

        s = vec![gpu.step(nchw_nlc, &[conv_out, &self.conv_seq], &[total, self.shape.c, rows], total)];
        s.push(gpu.step(add2, &[x_in, &self.conv_seq, &self.out], &[total], total));
        gpu.submit(&[], &s);
        &self.out
    }
}
