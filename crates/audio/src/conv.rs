// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 1D convolution Step-builders over the shared WGSL engine, plus tiny CPU
//! reference implementations used as test oracles.
//!
//! The codec conv encoder/decoder, the ECAPA speaker encoder, and the GAN
//! vocoder are all stacks of (transposed) 1D convolutions; these builders are
//! the audio analogue of `model::block`'s RMSNorm/RoPE/GQA/SwiGLU helpers. They
//! are pure dispatch assembly — shapes + buffers in, `Step`s out — and carry no
//! ParamStore / model concerns. Both NCL convolutions use grouping + dilation;
//! causal convs are expressed as a LEFT pad of `dilation*(K-1)` with `lo == l`.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Shape + hyperparameters of a 1D convolution (forward and both gradients share
/// this, since the kernels take an identical 10-word uniform).
#[derive(Clone, Copy, Debug)]
pub struct Conv1d {
    pub n: u32,
    pub cin: u32,
    pub l: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    pub dilation: u32,
    pub groups: u32,
    pub lo: u32,
}

/// Kernel-pipeline indices for the conv family a model supplies from its own
/// PIPELINES list (forward + input-grad + weight-grad).
#[derive(Clone, Copy)]
pub struct ConvKernels {
    pub fwd: usize,
    pub dx: usize,
    pub dw: usize,
}

impl Conv1d {
    fn params(&self) -> [u32; 10] {
        [self.n, self.cin, self.l, self.cout, self.k, self.stride, self.pad, self.dilation, self.groups, self.lo]
    }

    /// Output length of a standard (non-transposed) conv with the given low/high
    /// padding. The kernels only apply the LOW pad explicitly (high-side taps
    /// past the input are skipped = zero pad), so callers requesting symmetric
    /// padding pass `pad = pad_low` and size `lo` with this helper.
    pub fn out_len(l: u32, k: u32, stride: u32, pad_low: u32, pad_high: u32, dilation: u32) -> u32 {
        (l + pad_low + pad_high - dilation * (k - 1) - 1) / stride + 1
    }

    /// Output length of a transposed conv (upsampling).
    pub fn out_len_transposed(l: u32, k: u32, stride: u32, pad: u32, out_pad: u32, dilation: u32) -> u32 {
        (l - 1) * stride + dilation * (k - 1) + out_pad + 1 - 2 * pad
    }

    pub fn weight_numel(&self) -> usize {
        (self.cout * (self.cin / self.groups) * self.k) as usize
    }
    pub fn weight_numel_transposed(&self) -> usize {
        (self.cin * (self.cout / self.groups) * self.k) as usize
    }
}

/// `y = conv1d(x, w)` — `x:[N,Cin,L]`, `w:[Cout,Cin/G,K]`, `y:[N,Cout,Lo]`.
pub fn conv1d_fwd(g: &Gpu, k: &ConvKernels, c: &Conv1d, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer) -> Step {
    g.step(k.fwd, &[x, w, y], &c.params(), c.n * c.cout * c.lo)
}

/// conv1d backward: input grad `dx` (overwritten) and/or weight grad `dw`
/// (accumulated — zero it via `submit`'s `clears` first). Pass `None` to skip.
pub fn conv1d_bwd(
    g: &Gpu,
    k: &ConvKernels,
    c: &Conv1d,
    dy: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dx: Option<&DeviceBuffer>,
    dw: Option<&DeviceBuffer>,
) -> Vec<Step> {
    let mut s = Vec::new();
    if let Some(dx) = dx {
        s.push(g.step(k.dx, &[dy, w, dx], &c.params(), c.n * c.cin * c.l));
    }
    if let Some(dw) = dw {
        s.push(g.step(k.dw, &[dy, x, dw], &c.params(), c.cout * (c.cin / c.groups) * c.k));
    }
    s
}

/// `y = conv_transpose1d(x, w)` — `x:[N,Cin,L]`, `w:[Cin,Cout/G,K]`,
/// `y:[N,Cout,Lo]`.
pub fn convtr1d_fwd(g: &Gpu, k: &ConvKernels, c: &Conv1d, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer) -> Step {
    g.step(k.fwd, &[x, w, y], &c.params(), c.n * c.cout * c.lo)
}

/// Transposed-conv backward (mirrors [`conv1d_bwd`]; weight is `[Cin,Cout/G,K]`).
pub fn convtr1d_bwd(
    g: &Gpu,
    k: &ConvKernels,
    c: &Conv1d,
    dy: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dx: Option<&DeviceBuffer>,
    dw: Option<&DeviceBuffer>,
) -> Vec<Step> {
    let mut s = Vec::new();
    if let Some(dx) = dx {
        s.push(g.step(k.dx, &[dy, w, dx], &c.params(), c.n * c.cin * c.l));
    }
    if let Some(dw) = dw {
        s.push(g.step(k.dw, &[dy, x, dw], &c.params(), c.cin * (c.cout / c.groups) * c.k));
    }
    s
}

/// `weight[i,...] = weight_g[i] * weight_v[i,...] / ||weight_v[i,...]||_2` -
/// PyTorch `nn.utils.weight_norm(dim=0)`. `d0` is `weight_v`'s leading dim
/// (for `Conv1d` that is `Cout`; for `ConvTranspose1d`'s native `[Cin,
/// Cout/G, K]` weight layout it is `Cin` - `weight_norm`'s `dim=0` always
/// means dim 0 of the STORED tensor, whichever axis that happens to be for
/// the layer type; confirmed against a real checkpoint, where
/// `conv_t1.weight_g` has one scalar per `Cin` row, not per `Cout`). A
/// one-time host op at import time, not a hot-path kernel.
pub fn fold_weight_norm(g: &[f32], v: &[f32], d0: usize) -> Vec<f32> {
    assert_eq!(g.len(), d0, "weight_norm: weight_g has {} elements, expected d0={d0}", g.len());
    assert_eq!(v.len() % d0, 0, "weight_norm: weight_v length {} not divisible by d0={d0}", v.len());
    let rest = v.len() / d0;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..d0 {
        let row = &v[i * rest..(i + 1) * rest];
        let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let scale = (g[i] as f64 / norm.max(1e-12)) as f32;
        for (o, &x) in out[i * rest..(i + 1) * rest].iter_mut().zip(row) {
            *o = x * scale;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CPU reference oracles (kept tiny, used by tests; not on any hot path).
// ---------------------------------------------------------------------------

/// Reference forward for `conv1d` (matches `wgsl/conv1d.wgsl`).
pub fn conv1d_ref(c: &Conv1d, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (cin_g, cout_g) = (c.cin / c.groups, c.cout / c.groups);
    let mut y = vec![0.0f32; (c.n * c.cout * c.lo) as usize];
    for n in 0..c.n {
        for co in 0..c.cout {
            let g = co / cout_g;
            for lo in 0..c.lo {
                let mut acc = 0.0;
                for cl in 0..cin_g {
                    let ci = g * cin_g + cl;
                    for kw in 0..c.k {
                        let li_b = lo * c.stride + kw * c.dilation;
                        if li_b >= c.pad {
                            let li = li_b - c.pad;
                            if li < c.l {
                                let xi = ((n * c.cin + ci) * c.l + li) as usize;
                                let wi = ((co * cin_g + cl) * c.k + kw) as usize;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                }
                y[((n * c.cout + co) * c.lo + lo) as usize] = acc;
            }
        }
    }
    y
}

/// Reference forward for `convtr1d` (matches `wgsl/convtr1d.wgsl`).
pub fn convtr1d_ref(c: &Conv1d, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (cin_g, cout_g) = (c.cin / c.groups, c.cout / c.groups);
    let mut y = vec![0.0f32; (c.n * c.cout * c.lo) as usize];
    for n in 0..c.n {
        for co in 0..c.cout {
            let g = co / cout_g;
            let co_local = co - g * cout_g;
            for lo in 0..c.lo {
                let mut acc = 0.0;
                for kw in 0..c.k {
                    let num = lo + c.pad;
                    let sub = kw * c.dilation;
                    if num >= sub && (num - sub).is_multiple_of(c.stride) {
                        let li = (num - sub) / c.stride;
                        if li < c.l {
                            for cl in 0..cin_g {
                                let ci = g * cin_g + cl;
                                let xi = ((n * c.cin + ci) * c.l + li) as usize;
                                let wi = ((ci * cout_g + co_local) * c.k + kw) as usize;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                }
                y[((n * c.cout + co) * c.lo + lo) as usize] = acc;
            }
        }
    }
    y
}
