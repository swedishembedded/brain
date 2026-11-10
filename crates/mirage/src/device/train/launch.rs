// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The step recorder of the device tape and one method per training kernel.
//!
//! Every method appends one [`Step`]; argument order follows the kernel headers in `crates/kernels/wgsl`.

use gpu_core::{Dispatch, DeviceBuffer, Step};
use model::ops::Weight;

use crate::device::prims::ConvShape;
use crate::device::Device;
use crate::train::seq::AttnShape;
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
        // Columns per workgroup: 8, fewer for a narrow period so the rest of the 64 threads become row lanes.
        let cw = period.clamp(1, 8).next_power_of_two().min(8);
        let cw = if cw > period { cw / 2 } else { cw }.max(1);
        self.groups("mirage_t_col_red", &[g, o, dst], &[word(rows), word(period), u32::from(use_o), word(cw)], period.div_ceil(cw));
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
        self.groups("mirage_t_rms_dscale", &[x, g, stats, ds], &[word(rows), word(d)], d.div_ceil(8));
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
                self.groups("mirage_t_conv_dk", &[x, gy, dk], &params, (g.kh * g.kw * g.c * cout).div_ceil(8));
            }
            None => {
                let params: Vec<u32> = base.iter().copied().chain(tail).collect();
                self.groups("mirage_t_dwconv_dk", &[x, gy, dk], &params, (g.kh * g.kw * g.c).div_ceil(8));
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
    /// Rotary rotation with per-row-and-pair `(cos, sin)`: `y = R(sign) x`, assigned or accumulated.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope(&mut self, x: &DeviceBuffer, cs: &DeviceBuffer, y: &DeviceBuffer, rows: usize, heads: usize, hd: usize, sign: f32, acc: bool) {
        let total = rows * heads * (hd / 2);
        self.threads("mirage_t_rope", &[x, cs, y], &[word(total), word(heads), word(hd), sign.to_bits(), u32::from(acc)], total);
    }

    fn attn_params(s: &AttnShape) -> [u32; 8] {
        [word(s.tq), word(s.tk), word(s.hq), word(s.hk), word(s.qk), word(s.vh), u32::from(s.causal), (1.0 / (s.qk as f32).sqrt()).to_bits()]
    }

    /// Attention probabilities of every (query, head) row.
    pub(crate) fn attn_probs(&mut self, q: &DeviceBuffer, k: &DeviceBuffer, pr: &DeviceBuffer, s: &AttnShape) {
        self.groups("mirage_t_attn_probs", &[q, k, pr], &Self::attn_params(s), s.tq * s.hq);
    }

    /// Attention output `P v`.
    pub(crate) fn attn_out(&mut self, pr: &DeviceBuffer, v: &DeviceBuffer, out: &DeviceBuffer, s: &AttnShape) {
        self.threads("mirage_t_attn_out", &[pr, v, out], &Self::attn_params(s), s.tq * s.hq * s.vh);
    }

    /// Attention score gradient.
    pub(crate) fn attn_ds(&mut self, pr: &DeviceBuffer, go: &DeviceBuffer, v: &DeviceBuffer, ds: &DeviceBuffer, s: &AttnShape) {
        self.groups("mirage_t_attn_ds", &[pr, go, v, ds], &Self::attn_params(s), s.tq * s.hq);
    }

    /// Attention query gradient.
    pub(crate) fn attn_dq(&mut self, ds: &DeviceBuffer, k: &DeviceBuffer, dq: &DeviceBuffer, s: &AttnShape) {
        self.threads("mirage_t_attn_dq", &[ds, k, dq], &Self::attn_params(s), s.tq * s.hq * s.qk);
    }

    /// Attention key gradient.
    pub(crate) fn attn_dk(&mut self, ds: &DeviceBuffer, q: &DeviceBuffer, dk: &DeviceBuffer, s: &AttnShape) {
        self.threads("mirage_t_attn_dk", &[ds, q, dk], &Self::attn_params(s), s.tk * s.hk * s.qk);
    }

    /// Attention value gradient.
    pub(crate) fn attn_dv(&mut self, pr: &DeviceBuffer, go: &DeviceBuffer, dv: &DeviceBuffer, s: &AttnShape) {
        self.threads("mirage_t_attn_dv", &[pr, go, dv], &Self::attn_params(s), s.tk * s.hk * s.vh);
    }

    /// Causal dilated taps forward.
    pub(crate) fn taps(&mut self, x: &DeviceBuffer, taps: &DeviceBuffer, y: &DeviceBuffer, t: usize, w: usize, kt: usize, dil: usize) {
        self.threads("mirage_t_taps", &[x, taps, y], &[word(t), word(w), word(kt), word(dil)], t * w);
    }

    /// Causal dilated taps input gradient.
    pub(crate) fn taps_dx(&mut self, gy: &DeviceBuffer, taps: &DeviceBuffer, dx: &DeviceBuffer, t: usize, w: usize, kt: usize, dil: usize) {
        self.threads("mirage_t_taps_dx", &[gy, taps, dx], &[word(t), word(w), word(kt), word(dil)], t * w);
    }

    /// Causal dilated taps tap gradient.
    pub(crate) fn taps_dt(&mut self, x: &DeviceBuffer, gy: &DeviceBuffer, dt: &DeviceBuffer, t: usize, w: usize, kt: usize, dil: usize) {
        self.threads("mirage_t_taps_dt", &[x, gy, dt], &[word(t), word(w), word(kt), word(dil)], kt * w);
    }

    /// Sinkhorn forward over `blocks` blocks of `n x n`.
    pub(crate) fn sinkhorn(&mut self, x: &DeviceBuffer, y: &DeviceBuffer, blocks: usize, n: usize, iters: usize) {
        self.threads("mirage_t_sinkhorn", &[x, y], &[word(blocks), word(n), word(iters)], blocks);
    }

    /// Sinkhorn backward; `traj` is scratch for `blocks * 2 * iters * n * n` values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sinkhorn_bwd(&mut self, x: &DeviceBuffer, y: &DeviceBuffer, gy: &DeviceBuffer, traj: &DeviceBuffer, dl: &DeviceBuffer, blocks: usize, n: usize, iters: usize) {
        self.threads("mirage_t_sinkhorn_bwd", &[x, y, gy, traj, dl], &[word(blocks), word(n), word(iters)], blocks);
    }

    /// Lane mixing forward.
    pub(crate) fn mix(&mut self, h: &DeviceBuffer, x: &DeviceBuffer, y: &DeviceBuffer, dims: [usize; 4]) {
        let [t, a, _, d] = dims;
        self.threads("mirage_t_mix", &[h, x, y], &dims.map(word), t * a * d);
    }

    /// Lane mixing weight gradient.
    pub(crate) fn mix_dh(&mut self, gy: &DeviceBuffer, x: &DeviceBuffer, dh: &DeviceBuffer, dims: [usize; 4]) {
        let [t, a, b, _] = dims;
        self.threads("mirage_t_mix_dh", &[gy, x, dh], &dims.map(word), t * a * b);
    }

    /// Lane mixing source gradient.
    pub(crate) fn mix_dx(&mut self, h: &DeviceBuffer, gy: &DeviceBuffer, dx: &DeviceBuffer, dims: [usize; 4]) {
        let [t, _, b, d] = dims;
        self.threads("mirage_t_mix_dx", &[h, gy, dx], &dims.map(word), t * b * d);
    }

    /// `out[q, x] (+)= sum_y a[q, y] m[y sy + x sx]` over `q` rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kron_right(&mut self, a: &DeviceBuffer, m: &DeviceBuffer, out: &DeviceBuffer, q: usize, ny: usize, nx: usize, strides: (usize, usize), acc: bool) {
        let p = [word(q * nx), word(ny), word(nx), word(strides.0), word(strides.1), u32::from(acc)];
        self.threads("mirage_t_kron_right", &[a, m, out], &p, q * nx);
    }

    /// `out[r, x, l] (+)= sum_y m[y sy + x sx] a[r, y, l]` over rows viewed as `[fa, fb]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kron_left(&mut self, a: &DeviceBuffer, m: &DeviceBuffer, out: &DeviceBuffer, rows: usize, fab: (usize, usize), strides: (usize, usize), acc: bool) {
        let total = rows * fab.0 * fab.1;
        let p = [word(total), word(fab.0), word(fab.1), word(strides.0), word(strides.1), u32::from(acc)];
        self.threads("mirage_t_kron_left", &[a, m, out], &p, total);
    }

    /// Left-factor gradient of the Kronecker rotation.
    pub(crate) fn kron_da(&mut self, tmp: &DeviceBuffer, g: &DeviceBuffer, da: &DeviceBuffer, rows: usize, fa: usize, fb: usize) {
        self.threads("mirage_t_kron_da", &[tmp, g, da], &[word(rows), word(fa), word(fb)], fa * fa);
    }

    /// Right-factor gradient of the Kronecker rotation.
    pub(crate) fn kron_db(&mut self, z: &DeviceBuffer, dtmp: &DeviceBuffer, db: &DeviceBuffer, rows: usize, fa: usize, fb: usize) {
        self.threads("mirage_t_kron_db", &[z, dtmp, db], &[word(rows), word(fa), word(fb)], fb * fb);
    }

    /// `out[r, i] (+)= z[r, cols[i]]`.
    pub(crate) fn gather_cols(&mut self, z: &DeviceBuffer, cols: &DeviceBuffer, out: &DeviceBuffer, total: usize, n: usize, acc: bool) {
        self.threads("mirage_t_gather_cols", &[z, cols, out], &[word(total), word(n), u32::from(acc)], total);
    }
    /// Cross-entropy row losses and log-sum-exps.
    pub(crate) fn ce_rows(&mut self, x: &DeviceBuffer, target: &DeviceBuffer, w: &DeviceBuffer, rowloss: &DeviceBuffer, lse: &DeviceBuffer, rows: usize, v: usize) {
        self.groups("mirage_t_ce_rows", &[x, target, w, rowloss, lse], &[word(rows), word(v)], rows);
    }

    /// Cross-entropy logit gradient.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ce_bwd(&mut self, x: &DeviceBuffer, target: &DeviceBuffer, w: &DeviceBuffer, lse: &DeviceBuffer, g: &DeviceBuffer, dx: &DeviceBuffer, rows: usize, v: usize) {
        self.threads("mirage_t_ce_bwd", &[x, target, w, lse, g, dx], &[word(rows * v), word(v)], rows * v);
    }

    /// Focal loss per cell: `backward` accumulates the logit gradient, otherwise writes the term.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn focal(&mut self, x: &DeviceBuffer, target: &DeviceBuffer, weight: &DeviceBuffer, g: &DeviceBuffer, out: &DeviceBuffer, n: usize, alpha: f32, norm: f32, backward: bool) {
        self.threads("mirage_t_focal", &[x, target, weight, g, out], &[word(n), alpha.to_bits(), norm.to_bits(), u32::from(backward)], n);
    }

    /// Weighted L1 per element: `backward` accumulates the gradient, otherwise writes the term.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn l1(&mut self, x: &DeviceBuffer, target: &DeviceBuffer, weight: &DeviceBuffer, g: &DeviceBuffer, out: &DeviceBuffer, n: usize, norm: f32, backward: bool) {
        self.threads("mirage_t_l1", &[x, target, weight, g, out], &[word(n), norm.to_bits(), u32::from(backward)], n);
    }

    /// Mask loss forward into `out[0]` and `stats[0..3]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bce_dice(&mut self, x: &DeviceBuffer, target: &DeviceBuffer, out: &DeviceBuffer, stats: &DeviceBuffer, n: usize, bw: f32, dw: f32) {
        self.groups("mirage_t_bce_dice", &[x, target, out, stats], &[word(n), bw.to_bits(), dw.to_bits()], 1);
    }

    /// Mask loss logit gradient.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bce_dice_bwd(&mut self, x: &DeviceBuffer, target: &DeviceBuffer, stats: &DeviceBuffer, g: &DeviceBuffer, dx: &DeviceBuffer, n: usize, bw: f32, dw: f32) {
        self.threads("mirage_t_bce_dice_bwd", &[x, target, stats, g, dx], &[word(n), bw.to_bits(), dw.to_bits()], n);
    }

    /// CTC: log-softmax, the alpha-beta recursions (loss into `loss[0]`) and the logit gradient into `grad`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ctc(&mut self, x: &DeviceBuffer, ext: &DeviceBuffer, loss: &DeviceBuffer, grad: &DeviceBuffer, frames: usize, classes: usize, s_len: usize) {
        let lp = self.alloc(frames * classes);
        let (alpha, beta) = (self.alloc(frames * s_len), self.alloc(frames * s_len));
        self.groups("mirage_t_ctc_lp", &[x, &lp], &[word(frames), word(classes)], frames);
        let p = [word(frames), word(classes), word(s_len)];
        self.threads("mirage_t_ctc_dp", &[&lp, ext, &alpha, &beta, loss], &p, 1);
        self.threads("mirage_t_ctc_grad", &[&lp, ext, &alpha, &beta, grad], &p, frames * classes);
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
