// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The step recorder of the device tape and one method per training kernel.
//!
//! Every method appends one [`Step`]; argument order follows the kernel headers in `crates/kernels/wgsl`.

use gpu_core::{Dispatch, DeviceBuffer, Step};
use model::ops::Weight;

use crate::device::prims::ConvShape;
use crate::device::Device;
use crate::vision::ops::NORM_EPS;

/// A word count that must fit a kernel parameter.
pub(super) fn word(n: usize) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| panic!("a device training dimension of {n} exceeds 32 bits"))
}

/// Pending forward and backward steps, and the buffers that must be zero when the submission starts.
pub(crate) struct Rec<'d> {
    pub(super) dev: &'d Device,
    steps: Vec<Step>,
    clears: Vec<DeviceBuffer>,
}

impl<'d> Rec<'d> {
    pub(super) fn new(dev: &'d Device) -> Rec<'d> {
        Rec { dev, steps: Vec::new(), clears: Vec::new() }
    }

    /// Run everything recorded so far.
    pub(super) fn submit(&mut self) {
        let clears: Vec<&DeviceBuffer> = self.clears.iter().collect();
        self.dev.gpu().submit(&clears, &self.steps);
        self.steps.clear();
        self.clears.clear();
    }

    /// A buffer of `len` values whose contents a kernel will fully write.
    pub(crate) fn alloc(&self, len: usize) -> DeviceBuffer {
        self.dev.gpu().storage(len.max(1) as u64)
    }

    /// A buffer of `len` values that reads as zero when the next submission starts.
    pub(crate) fn alloc_zeroed(&mut self, len: usize) -> DeviceBuffer {
        let b = self.alloc(len);
        self.clears.push(b.clone());
        b
    }

    fn threads(&mut self, kernel: &str, bufs: &[&DeviceBuffer], params: &[u32], n: usize) {
        let step = self.dev.gpu().dispatch(self.dev.k(kernel), bufs, params, Dispatch::Threads(word(n)));
        self.steps.push(step);
    }

    fn groups(&mut self, kernel: &str, bufs: &[&DeviceBuffer], params: &[u32], n: usize) {
        let step = self.dev.gpu().dispatch(self.dev.k(kernel), bufs, params, Dispatch::Workgroups(word(n)));
        self.steps.push(step);
    }

    /// `out = f(a, b)`: see `mirage_t_ew` for the modes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ew(&mut self, mode: u32, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer, total: usize, period: usize, c: f32) {
        self.threads("mirage_t_ew", &[a, b, out], &[word(total), word(period), mode, c.to_bits()], total);
    }

    /// `dst[i] += g[i] * o[k(i)]`, `k` per the mode of `mirage_t_mul_acc`.
    pub(crate) fn mul_acc(&mut self, g: &DeviceBuffer, o: &DeviceBuffer, dst: &DeviceBuffer, total: usize, period: usize, per_row: bool) {
        self.threads("mirage_t_mul_acc", &[g, o, dst], &[word(total), word(period), u32::from(per_row)], total);
    }

    /// `dst[c] += sum_r g[r, c] (* o[r, c])`.
    pub(crate) fn col_red(&mut self, g: &DeviceBuffer, o: &DeviceBuffer, dst: &DeviceBuffer, rows: usize, period: usize, use_o: bool) {
        self.threads("mirage_t_col_red", &[g, o, dst], &[word(rows), word(period), u32::from(use_o)], period);
    }

    /// `dst[r] (+)= sum_c a[r, c] b[r, c]`.
    pub(crate) fn row_red(&mut self, a: &DeviceBuffer, b: &DeviceBuffer, dst: &DeviceBuffer, rows: usize, d: usize, acc: bool) {
        self.groups("mirage_t_row_red", &[a, b, dst], &[word(rows), word(d), u32::from(acc)], rows);
    }

    /// `out[0] = c * sum(x[0..n])`.
    pub(crate) fn sum(&mut self, x: &DeviceBuffer, out: &DeviceBuffer, n: usize, c: f32) {
        self.groups("mirage_t_sum", &[x, out], &[word(n), c.to_bits()], 1);
    }

    /// `dst[i] += s[0] * c`.
    pub(crate) fn bcast_acc(&mut self, s: &DeviceBuffer, dst: &DeviceBuffer, total: usize, c: f32) {
        self.threads("mirage_t_bcast_acc", &[s, dst], &[word(total), c.to_bits()], total);
    }

    /// `dx += dy * f'(x)`; mode 0 SiLU, 1 sigmoid, 2 GELU.
    pub(crate) fn act_bwd(&mut self, x: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, total: usize, mode: u32) {
        self.threads("mirage_t_act_bwd", &[x, dy, dx], &[word(total), mode], total);
    }

    /// `dst[dst_off + i] (+)= src[src_off + i]` for `i < len`.
    pub(crate) fn span(&mut self, src: &DeviceBuffer, dst: &DeviceBuffer, len: usize, src_off: usize, dst_off: usize, acc: bool) {
        self.threads("mirage_t_span", &[src, dst], &[word(len), word(src_off), word(dst_off), u32::from(acc)], len);
    }

    /// One weighted-plan gather on the owner axis (see `mirage_t_plan`).
    pub(crate) fn plan(&mut self, ivals: &DeviceBuffer, list: &DeviceBuffer, other: &DeviceBuffer, out: &DeviceBuffer, total: usize, n_ivals: usize) {
        self.threads("mirage_t_plan", &[ivals, list, other, out], &[word(total), word(n_ivals)], total);
    }

    /// Row statistics of the RMS-norm backward into `stats[2 * rows]`.
    pub(crate) fn rms_stats(&mut self, x: &DeviceBuffer, g: &DeviceBuffer, w: &DeviceBuffer, stats: &DeviceBuffer, rows: usize, d: usize) {
        self.groups("mirage_t_rms_stats", &[x, g, w, stats], &[word(d), word(rows), NORM_EPS.to_bits()], rows);
    }

    /// RMS-norm input gradient.
    pub(crate) fn rms_dx(&mut self, x: &DeviceBuffer, g: &DeviceBuffer, w: &DeviceBuffer, stats: &DeviceBuffer, dx: &DeviceBuffer, total: usize, d: usize) {
        self.threads("mirage_t_rms_dx", &[x, g, w, stats, dx], &[word(total), word(d)], total);
    }

    /// RMS-norm scale gradient.
    pub(crate) fn rms_dscale(&mut self, x: &DeviceBuffer, g: &DeviceBuffer, stats: &DeviceBuffer, ds: &DeviceBuffer, rows: usize, d: usize) {
        self.threads("mirage_t_rms_dscale", &[x, g, stats, ds], &[word(rows), word(d)], d);
    }

    /// Row softmax.
    pub(crate) fn softmax(&mut self, x: &DeviceBuffer, out: &DeviceBuffer, rows: usize, d: usize) {
        self.groups("mirage_t_softmax", &[x, out], &[word(rows), word(d)], rows);
    }

    /// Softmax backward, accumulating.
    pub(crate) fn softmax_bwd(&mut self, p: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, rows: usize, d: usize) {
        self.groups("mirage_t_softmax_bwd", &[p, dy, dx], &[word(rows), word(d)], rows);
    }

    /// `dst += src` (the shared residual add).
    pub(crate) fn accumulate(&mut self, src: &DeviceBuffer, dst: &DeviceBuffer, total: usize) {
        self.threads("add", &[src, dst], &[word(total)], total);
    }

    /// `y += a * x`.
    pub(crate) fn axpy(&mut self, a: f32, x: &DeviceBuffer, y: &DeviceBuffer, total: usize) {
        self.threads("mirage_axpy", &[x, y], &[word(total), a.to_bits()], total);
    }

    /// `out = f(x)` with a shared unary kernel (`silu`, `sigmoid`, `gelu`).
    pub(crate) fn unary(&mut self, kernel: &str, x: &DeviceBuffer, out: &DeviceBuffer, total: usize) {
        self.threads(kernel, &[x, out], &[word(total)], total);
    }

    /// The zero-centred or unit RMS norm forward with weight `w` (`1 + scale` or ones).
    pub(crate) fn rmsnorm(&mut self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, rows: usize, d: usize) {
        self.groups("rmsnorm_rows", &[x, w, out], &[word(d), word(rows), NORM_EPS.to_bits()], rows);
    }

    /// `y[rows, out] = x[rows, in] W` for a dense `[in, out]` kernel.
    pub(crate) fn matmul_in_out(&mut self, w: &DeviceBuffer, x: &DeviceBuffer, y: &DeviceBuffer, rows: usize, inn: usize, out: usize) {
        let weight = Weight::dense_f32(w.clone(), word(inn), word(out));
        self.dev.ops().matmul_dx(&mut self.steps, &weight, x, word(rows), y, false);
    }

    /// `y[rows, n] = x[rows, k] W^T` for a dense `[n, k]` weight.
    pub(crate) fn matmul_nk(&mut self, w: &DeviceBuffer, x: &DeviceBuffer, y: &DeviceBuffer, rows: usize, n: usize, k: usize) {
        let weight = Weight::dense_f32(w.clone(), word(n), word(k));
        let act = self.dev.ops().act_f32(x, 0, word(rows), word(k));
        self.dev.ops().matmul(&mut self.steps, &weight, &act, y, 0);
    }

    /// `dx[rows, k] += dy[rows, n] W` for a dense `[n, k]` weight.
    pub(crate) fn matmul_dx_acc(&mut self, w: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, rows: usize, n: usize, k: usize) {
        let weight = Weight::dense_f32(w.clone(), word(n), word(k));
        self.dev.ops().matmul_dx(&mut self.steps, &weight, dy, word(rows), dx, true);
    }

    /// `dw[n, k] += a[rows, n]^T b[rows, k]`.
    pub(crate) fn matmul_dw(&mut self, a: &DeviceBuffer, b: &DeviceBuffer, dw: &DeviceBuffer, rows: usize, n: usize, k: usize) {
        // `Ops::matmul_dw(x, dy, ..)` computes `dw[n, k] += dy[m, n]^T x[m, k]`.
        self.dev.ops().matmul_dw(&mut self.steps, b, a, word(rows), word(n), word(k), dw);
    }

    /// Forward depthwise (`kernel [kh * kw, c]`) or dense (HWIO) convolution of one image, via the inference kernels.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_fwd(&mut self, g: Conv, dense_cout: Option<usize>, x: &DeviceBuffer, k: &DeviceBuffer, y: &DeviceBuffer) {
        let shape = ConvShape { batch: 1, h: word(g.h), w: word(g.w), kh: word(g.kh), kw: word(g.kw), sy: word(g.sy), sx: word(g.sx) };
        match dense_cout {
            Some(cout) => self.dev.conv2d(&mut self.steps, shape, word(g.c), word(cout), x, k, y),
            None => self.dev.dwconv(&mut self.steps, shape, word(g.c), x, k, y),
        }
    }

    /// Convolution input gradient: depthwise when `dense_cout` is `None`.
    pub(crate) fn conv_dx(&mut self, g: Conv, dense_cout: Option<usize>, gy: &DeviceBuffer, k: &DeviceBuffer, dx: &DeviceBuffer) {
        let (ho, wo) = (g.ho(), g.wo());
        let base = [word(g.h), word(g.w), word(g.c), word(ho), word(wo)];
        let tail = [word(g.kh), word(g.kw), word(g.sy), word(g.sx)];
        match dense_cout {
            Some(cout) => {
                let params: Vec<u32> = base.iter().copied().chain([word(cout)]).chain(tail).collect();
                self.threads("mirage_t_conv_dx", &[gy, k, dx], &params, g.h * g.w * g.c);
            }
            None => {
                let params: Vec<u32> = base.iter().copied().chain(tail).collect();
                self.threads("mirage_t_dwconv_dx", &[gy, k, dx], &params, g.h * g.w * g.c);
            }
        }
    }

    /// Convolution kernel gradient: depthwise when `dense_cout` is `None`.
    pub(crate) fn conv_dk(&mut self, g: Conv, dense_cout: Option<usize>, x: &DeviceBuffer, gy: &DeviceBuffer, dk: &DeviceBuffer) {
        let (ho, wo) = (g.ho(), g.wo());
        let base = [word(g.h), word(g.w), word(g.c), word(ho), word(wo)];
        let tail = [word(g.kh), word(g.kw), word(g.sy), word(g.sx)];
        match dense_cout {
            Some(cout) => {
                let params: Vec<u32> = base.iter().copied().chain([word(cout)]).chain(tail).collect();
                self.threads("mirage_t_conv_dk", &[x, gy, dk], &params, g.kh * g.kw * g.c * cout);
            }
            None => {
                let params: Vec<u32> = base.iter().copied().chain(tail).collect();
                self.threads("mirage_t_dwconv_dk", &[x, gy, dk], &params, g.kh * g.kw * g.c);
            }
        }
    }

    /// Banded channel convolution forward.
    pub(crate) fn band(&mut self, u: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer, rows: usize, n: usize, k: usize) {
        self.threads("mirage_t_band", &[u, w, y], &[word(rows), word(n), word(k)], rows * n);
    }

    /// Banded channel convolution input gradient.
    pub(crate) fn band_dx(&mut self, gy: &DeviceBuffer, w: &DeviceBuffer, du: &DeviceBuffer, rows: usize, n: usize, k: usize) {
        self.threads("mirage_t_band_dx", &[gy, w, du], &[word(rows), word(n), word(k)], rows * n);
    }

    /// Banded channel convolution weight gradient.
    pub(crate) fn band_dw(&mut self, u: &DeviceBuffer, gy: &DeviceBuffer, dw: &DeviceBuffer, rows: usize, n: usize, k: usize) {
        self.threads("mirage_t_band_dw", &[u, gy, dw], &[word(rows), word(n), word(k)], n * k);
    }

    /// Per-channel energy `sqrt(sum_valid y^2 + eps)` and the response-norm scale `gamma E / (mean(E) + eps) + 1`.
    pub(crate) fn grn_scale(&mut self, y: &DeviceBuffer, mask: &DeviceBuffer, gamma: &DeviceBuffer, energy: &DeviceBuffer, scale: &DeviceBuffer, rows: usize, d: usize) {
        let eps = NORM_EPS.to_bits();
        self.threads("mirage_grn_energy", &[y, mask, energy], &[1, word(rows), word(d), eps], d);
        self.threads("mirage_grn_scale", &[energy, gamma, scale], &[1, word(d), eps], d);
    }

    /// Response-norm backward statistics: `de` (assigned) and `dgamma` (accumulated).
    pub(crate) fn grn_de(&mut self, energy: &DeviceBuffer, gamma: &DeviceBuffer, gs: &DeviceBuffer, de: &DeviceBuffer, dgamma: &DeviceBuffer, d: usize) {
        self.groups("mirage_t_grn_de", &[energy, gamma, gs, de, dgamma], &[word(d), NORM_EPS.to_bits()], 1);
    }

    /// Response-norm input gradient.
    pub(crate) fn grn_dy(&mut self, y: &DeviceBuffer, mask: &DeviceBuffer, energy: &DeviceBuffer, de: &DeviceBuffer, dy: &DeviceBuffer, total: usize, d: usize) {
        self.threads("mirage_t_grn_dy", &[y, mask, energy, de, dy], &[word(total), word(d)], total);
    }
}

/// Geometry of one convolution of a single image (zero padding `k / 2`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv {
    pub(crate) h: usize,
    pub(crate) w: usize,
    /// Input channels.
    pub(crate) c: usize,
    pub(crate) kh: usize,
    pub(crate) kw: usize,
    pub(crate) sy: usize,
    pub(crate) sx: usize,
}

impl Conv {
    pub(crate) fn ho(&self) -> usize {
        (self.h + 2 * (self.kh / 2) - self.kh) / self.sy + 1
    }

    pub(crate) fn wo(&self) -> usize {
        (self.w + 2 * (self.kw / 2) - self.kw) / self.sx + 1
    }
}
