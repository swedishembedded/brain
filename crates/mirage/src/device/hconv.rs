// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The HConvGLU mixer on the device: RMS norm, sequency Walsh transform, the gate and value bands with the gated
//! product, the output band and the transform back.
//!
//! Two resident forms exist and are gated against each other: the structured form runs the Walsh butterfly
//! factors and the 65-tap bands on the original parameters (`3 * 65 * n` weights per mixer), the folded form
//! multiplies both transforms into two dense matrices (`3 * n^2`) and runs the selector's GEMMs.

use gpu_core::{DeviceBuffer, Step};
use model::ops::Weight;

use super::kron::Kron;
use super::{Device, WeightForm};
use crate::vision::encoder::HConv;

/// A resident HConvGLU mixer.
pub struct DeviceHConv {
    n: u32,
    norm: DeviceBuffer,
    bias: DeviceBuffer,
    form: Form,
}

enum Form {
    Folded { w_in: Weight, w_out: Weight },
    Structured { walsh: Kron, gate: DeviceBuffer, value: DeviceBuffer, out: DeviceBuffer, taps: u32 },
}

/// `[n, k]` per-channel taps as `[k, n]`, so a warp reads neighbouring channels of one tap.
fn tap_major(w: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), n * k);
    (0..k * n).map(|e| w[(e % n) * k + e / n]).collect()
}

impl DeviceHConv {
    /// Upload `host` in the device's weight form. `walsh` is the transform of the mixer's width, shared by all mixers
    /// (required in the structured form).
    pub fn upload(dev: &Device, host: &HConv, walsh: Option<&Kron>) -> DeviceHConv {
        let n = host.width;
        let norm: Vec<f32> = host.norm.iter().map(|s| 1.0 + s).collect();
        let form = match dev.form() {
            WeightForm::Folded => {
                let f = host.fold();
                Form::Folded { w_in: dev.weight("hconv_in", &f.in_wt, 2 * n, n), w_out: dev.weight("hconv_out", &f.out_wt, n, n) }
            }
            WeightForm::Structured => {
                let walsh = walsh.expect("the structured form needs the Walsh transform");
                assert_eq!(walsh.width() as usize, n, "the shared Walsh transform has the mixer's width");
                let k = host.kernel;
                Form::Structured {
                    walsh: walsh.clone(),
                    gate: dev.param_buffer("hconv_gate", &tap_major(&host.gate, n, k)),
                    value: dev.param_buffer("hconv_value", &tap_major(&host.value, n, k)),
                    out: dev.param_buffer("hconv_out_band", &tap_major(&host.out, n, k)),
                    taps: k as u32,
                }
            }
        };
        DeviceHConv { n: n as u32, norm: dev.gpu().storage_init("hconv_norm", &norm), bias: dev.gpu().storage_init("hconv_bias", &host.gate_bias), form }
    }

    /// Append the steps computing `y = HConvGLU(ZC(x))` for `rows` rows.
    pub fn encode(&self, dev: &Device, s: &mut Vec<Step>, x: &DeviceBuffer, rows: u32, y: &DeviceBuffer) {
        let g = dev.gpu();
        let n = self.n;
        let normed = g.storage_uninit((rows * n) as u64);
        dev.zc_norm(s, x, &self.norm, rows, n, &normed);
        match &self.form {
            Form::Folded { w_in, w_out } => {
                let ab = g.storage((rows * 2 * n) as u64);
                let z = g.storage((rows * n) as u64);
                dev.linear(s, w_in, &normed, rows, &ab);
                dev.hconv_glu(s, &ab, &self.bias, rows, n, &z);
                dev.linear(s, w_out, &z, rows, y);
            }
            Form::Structured { walsh, gate, value, out, taps } => {
                let u = g.storage_uninit((rows * n) as u64);
                let z = g.storage_uninit((rows * n) as u64);
                dev.kron(s, walsh, &normed, rows, &u, false);
                dev.hconv_band_glu(s, &u, gate, value, &self.bias, (rows, n, *taps), &z);
                dev.band_cols(s, &z, out, (rows, n, *taps), &normed);
                dev.kron(s, walsh, &normed, rows, y, false);
            }
        }
    }
}
