// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The HConvGLU mixer on the device: RMS norm, one GEMM into gate and value,
//! the gated product, one GEMM out. The Walsh transforms and the banded
//! convolutions are folded into the two weight matrices at upload.

use gpu_core::{DeviceBuffer, Step};
use model::ops::Weight;

use super::Device;
use crate::vision::encoder::HConv;

/// A resident HConvGLU mixer.
pub struct DeviceHConv {
    n: u32,
    norm: DeviceBuffer,
    bias: DeviceBuffer,
    w_in: Weight,
    w_out: Weight,
}

impl DeviceHConv {
    /// Fold and upload `host`.
    pub fn upload(dev: &Device, host: &HConv) -> DeviceHConv {
        let f = host.fold();
        let n = f.width;
        DeviceHConv {
            n: n as u32,
            norm: dev.gpu().storage_init("hconv_norm", &f.norm_weight),
            bias: dev.gpu().storage_init("hconv_bias", &f.gate_bias),
            w_in: dev.weight("hconv_in", &f.in_wt, 2 * n, n),
            w_out: dev.weight("hconv_out", &f.out_wt, n, n),
        }
    }

    /// Append the steps computing `y = HConvGLU(ZC(x))` for `rows` rows.
    pub fn encode(&self, dev: &Device, s: &mut Vec<Step>, x: &DeviceBuffer, rows: u32, y: &DeviceBuffer) {
        let g = dev.gpu();
        let n = self.n;
        let normed = g.storage((rows * n) as u64);
        let ab = g.storage((rows * 2 * n) as u64);
        let z = g.storage((rows * n) as u64);
        dev.zc_norm(s, x, &self.norm, rows, n, &normed);
        dev.linear(s, &self.w_in, &normed, rows, &ab);
        dev.hconv_glu(s, &ab, &self.bias, rows, n, &z);
        dev.linear(s, &self.w_out, &z, rows, y);
    }
}
