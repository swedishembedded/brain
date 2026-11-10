// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device primitives of the Mirage vision path. Feature maps are NHWC
//! (`[batch, h, w, c]` flattened); a "row" is one pixel's channel vector.
//! Every method appends one step and takes the buffers in kernel order.

use gpu_core::{DeviceBuffer, Step};

use super::Device;
use crate::vision::ops::NORM_EPS;

/// A convolution's geometry: input `h x w`, `c` channels in, `kh x kw` kernel, strides.
#[derive(Debug, Clone, Copy)]
pub struct ConvShape {
    /// Images.
    pub batch: u32,
    /// Input height.
    pub h: u32,
    /// Input width.
    pub w: u32,
    /// Kernel height.
    pub kh: u32,
    /// Kernel width.
    pub kw: u32,
    /// Vertical stride.
    pub sy: u32,
    /// Horizontal stride.
    pub sx: u32,
}

impl ConvShape {
    /// Output height (zero padding `k / 2`).
    pub fn ho(&self) -> u32 {
        (self.h + 2 * (self.kh / 2) - self.kh) / self.sy + 1
    }

    /// Output width.
    pub fn wo(&self) -> u32 {
        (self.w + 2 * (self.kw / 2) - self.kw) / self.sx + 1
    }
}

/// Grouped-query attention shape for [`Device::attention`].
#[derive(Debug, Clone, Copy)]
pub struct AttnShape {
    /// Sequences.
    pub batch: u32,
    /// Queries per sequence.
    pub nq: u32,
    /// Query heads.
    pub heads: u32,
    /// Key/value heads.
    pub kvh: u32,
    /// Query/key head width.
    pub hd: u32,
    /// Value head width.
    pub vd: u32,
    /// Key rows per sequence in the buffers.
    pub nk: u32,
    /// Keys at index `>= visible` are ignored.
    pub visible: u32,
}

impl Device {
    fn push(&self, s: &mut Vec<Step>, name: &str, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) {
        s.push(self.gpu().step(self.k(name), bufs, params, threads));
    }

    /// Dense convolution, HWIO weights `[kh, kw, cin, cout]`.
    pub fn conv2d(&self, s: &mut Vec<Step>, g: ConvShape, cin: u32, cout: u32, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer) {
        let (ho, wo) = (g.ho(), g.wo());
        let p = [g.batch, g.h, g.w, cin, ho, wo, cout, g.kh, g.kw, g.sy, g.sx];
        self.push(s, "mirage_conv2d", &[x, w, y], &p, g.batch * ho * wo * cout);
    }

    /// Depthwise convolution, kernel `[kh * kw, c]`.
    pub fn dwconv(&self, s: &mut Vec<Step>, g: ConvShape, c: u32, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer) {
        let (ho, wo) = (g.ho(), g.wo());
        let p = [g.batch, g.h, g.w, c, ho, wo, g.kh, g.kw, g.sy, g.sx];
        self.push(s, "mirage_dwconv", &[x, w, y], &p, g.batch * ho * wo * c);
    }

    /// `x[r] *= m[r]` over `rows` rows of width `d`.
    pub fn rows_mask(&self, s: &mut Vec<Step>, x: &DeviceBuffer, m: &DeviceBuffer, rows: u32, d: u32) {
        self.push(s, "mirage_rows_mask", &[x, m], &[rows, d], rows * d);
    }

    /// `x[r] += m[r] * y[r]`.
    pub fn add_masked(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, m: &DeviceBuffer, rows: u32, d: u32) {
        self.push(s, "mirage_add_masked", &[x, y, m], &[rows, d], rows * d);
    }

    /// `x += y` over `total` values.
    pub fn add(&self, s: &mut Vec<Step>, y: &DeviceBuffer, x: &DeviceBuffer, total: u32) {
        self.push(s, "add", &[y, x], &[total], total);
    }

    /// `y = silu(x)`.
    pub fn silu(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, total: u32) {
        self.push(s, "silu", &[x, y], &[total], total);
    }

    /// `y = sigmoid(x)`.
    pub fn sigmoid(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, total: u32) {
        self.push(s, "sigmoid", &[x, y], &[total], total);
    }

    /// `x[r, c] += bias[c]`.
    pub fn bias_add(&self, s: &mut Vec<Step>, x: &DeviceBuffer, bias: &DeviceBuffer, rows: u32, n: u32) {
        self.push(s, "bias_add", &[x, bias], &[rows, n], rows * n);
    }

    /// `g = silu(g + bg) * (v + bv)`.
    pub fn gated_mix(&self, s: &mut Vec<Step>, g: &DeviceBuffer, v: &DeviceBuffer, bg: &DeviceBuffer, bv: &DeviceBuffer, rows: u32, d: u32) {
        self.push(s, "mirage_gated_mix", &[g, v, bg, bv], &[rows, d], rows * d);
    }

    /// Global response norm of `y` added into `x` (see the kernels): `x += m * (y * scale + beta)`,
    /// statistics over the valid rows of each of `batch` images of `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn grn_residual(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, m: &DeviceBuffer, gamma: &DeviceBuffer, beta: &DeviceBuffer, batch: u32, rows: u32, d: u32) {
        let g = self.gpu();
        let (energy, scale) = (g.storage((batch * d) as u64), g.storage((batch * d) as u64));
        let eps = crate::vision::ops::NORM_EPS.to_bits();
        self.push(s, "mirage_grn_energy", &[y, m, &energy], &[batch, rows, d, eps], batch * d);
        self.push(s, "mirage_grn_scale", &[&energy, gamma, &scale], &[batch, d, eps], batch * d);
        self.push(s, "mirage_grn_apply", &[x, y, m, &scale, beta], &[batch, rows, d], batch * rows * d);
    }

    /// `y += g[i mod d] * x`.
    pub fn chan_fma(&self, s: &mut Vec<Step>, x: &DeviceBuffer, g: &DeviceBuffer, y: &DeviceBuffer, total: u32, d: u32) {
        self.push(s, "mirage_chan_fma", &[x, g, y], &[total, d], total);
    }

    /// `y += a * x` for a scalar `a`.
    pub fn axpy(&self, s: &mut Vec<Step>, a: f32, x: &DeviceBuffer, y: &DeviceBuffer, total: u32) {
        self.push(s, "mirage_axpy", &[x, y], &[total, a.to_bits()], total);
    }

    /// `z = uv[:, :d] * silu(uv[:, d:])`.
    pub fn swiglu_rows(&self, s: &mut Vec<Step>, uv: &DeviceBuffer, z: &DeviceBuffer, rows: u32, d: u32) {
        self.push(s, "mirage_swiglu_rows", &[uv, z], &[rows, d], rows * d);
    }

    /// Mean over the height of an NHWC map.
    pub fn mean_h(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, batch: u32, h: u32, w: u32, c: u32) {
        self.push(s, "mirage_mean_h", &[x, y], &[batch, h, w, c], batch * w * c);
    }

    /// Nearest-neighbour 2x upsample of an NHWC map.
    pub fn upsample2(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, batch: u32, h: u32, w: u32, c: u32) {
        self.push(s, "mirage_upsample2", &[x, y], &[batch, h, w, c], batch * 4 * h * w * c);
    }

    /// 2D rotary embedding of the first `cells` tokens of every image.
    #[allow(clippy::too_many_arguments)]
    pub fn rope2d(&self, s: &mut Vec<Step>, t: &DeviceBuffer, batch: u32, tokens: u32, cells: u32, grid: u32, heads: u32, hd: u32, base: f32) {
        let rows = batch * tokens;
        self.push(s, "mirage_rope2d", &[t], &[tokens, cells, grid, heads, hd, base.to_bits(), rows], rows * heads * (hd / 2));
    }

    /// Masked grouped-query attention with the usual `1 / sqrt(hd)` scale, key-parallel:
    /// a scores pass, a row softmax and a weighted-sum pass, each with a thread per element.
    pub fn attention(&self, s: &mut Vec<Step>, a: AttnShape, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, key_mask: &DeviceBuffer, out: &DeviceBuffer) {
        self.attention_at(s, a, q, k, v, key_mask, &self.pos_dummy, out);
    }

    /// [`Device::attention`] where `a.visible == u32::MAX` means `pos[0] + 1` keys (position in a buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_at(&self, s: &mut Vec<Step>, a: AttnShape, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, key_mask: &DeviceBuffer, pos: &DeviceBuffer, out: &DeviceBuffer) {
        assert!(a.heads.is_multiple_of(a.kvh), "attention shape {a:?}");
        let scale = (1.0 / (a.hd as f32).sqrt()).to_bits();
        let rows = a.batch * a.nq * a.heads;
        let scores = self.alloc((rows * a.nk) as usize);
        let p = [a.batch, a.nq, a.heads, a.kvh, a.hd, a.nk, a.visible, scale];
        self.push(s, "mirage_attn_scores", &[q, k, key_mask, &scores, pos], &p, rows * a.nk);
        s.push(self.gpu().dispatch(self.k("mirage_attn_softmax"), &[&scores], &[rows, a.nk], gpu_core::Dispatch::Workgroups(rows)));
        let p = [a.batch, a.nq, a.heads, a.kvh, a.vd, a.nk, a.visible];
        self.push(s, "mirage_attn_apply", &[&scores, v, out, pos], &p, rows * a.vd);
    }

    /// The single-kernel reference attention (serial over keys); the parity oracle for [`Device::attention`].
    pub fn attention_reference(&self, s: &mut Vec<Step>, a: AttnShape, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, key_mask: &DeviceBuffer, out: &DeviceBuffer) {
        assert!(a.vd <= 128 && a.heads.is_multiple_of(a.kvh), "attention shape {a:?}");
        let scale = (1.0 / (a.hd as f32).sqrt()).to_bits();
        let p = [a.batch, a.nq, a.heads, a.kvh, a.hd, a.vd, a.nk, a.visible, scale];
        self.push(s, "mirage_attn", &[q, k, v, key_mask, out], &p, a.batch * a.nq * a.heads);
    }

    /// `y = a * b` elementwise.
    pub fn mul(&self, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, total: u32) {
        self.push(s, "mul", &[a, b, y], &[total], total);
    }

    /// `dst[b, dst_off + r] = src[b, src_off + r]` for `r < n` (rows of width `d`).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(&self, s: &mut Vec<Step>, src: &DeviceBuffer, (src_stride, src_off): (u32, u32), dst: &DeviceBuffer, (dst_stride, dst_off): (u32, u32), batch: u32, n: u32, d: u32) {
        self.push(s, "mirage_copy_rows", &[src, dst], &[batch, n, d, src_stride, src_off, dst_stride, dst_off], batch * n * d);
    }

    /// A resident vector.
    pub fn buf(&self, v: &[f32]) -> DeviceBuffer {
        self.gpu().storage_init("mirage", v)
    }

    /// The weight `1 + scale` of a zero-centred RMS norm, resident.
    pub fn norm_weight(&self, scale: &[f32]) -> DeviceBuffer {
        self.buf(&scale.iter().map(|s| 1.0 + s).collect::<Vec<_>>())
    }

    /// A fresh buffer of `n` values with undefined contents: for outputs a kernel writes completely.
    pub fn alloc(&self, n: usize) -> DeviceBuffer {
        self.gpu().storage_uninit(n as u64)
    }

    /// A fresh buffer of `n` zeros: for accumulators.
    pub fn alloc_zeroed(&self, n: usize) -> DeviceBuffer {
        self.gpu().storage(n as u64)
    }

    /// `zc_norm(x)` into a new buffer.
    pub fn zc_new(&self, s: &mut Vec<Step>, x: &DeviceBuffer, weight: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
        let y = self.alloc((rows * d) as usize);
        self.zc_norm(s, x, weight, rows, d, &y);
        y
    }

    /// `silu(x)` into a new buffer.
    pub fn silu_new(&self, s: &mut Vec<Step>, x: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let y = self.alloc(total as usize);
        self.silu(s, x, &y, total);
        y
    }

    /// `x W^T` into a new buffer.
    pub fn linear_new(&self, s: &mut Vec<Step>, w: &model::ops::Weight, x: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let y = self.alloc((rows * w.n()) as usize);
        self.linear(s, w, x, rows, &y);
        y
    }

    /// Submit `steps` and wait for them, so buffers they used may be dropped.
    pub fn flush(&self, steps: &mut Vec<Step>) {
        self.gpu().submit(&[], steps);
        self.gpu().poll_wait();
        steps.clear();
    }

    /// `out[b, j, :] = scale * emb[tokens[b], :]` for every lane `j`.
    pub fn embed_lanes(&self, s: &mut Vec<Step>, tokens: &DeviceBuffer, emb: &DeviceBuffer, out: &DeviceBuffer, (batch, n, d): (u32, u32, u32), scale: f32) {
        self.push(s, "mirage_embed_lanes", &[tokens, emb, out], &[batch, n, d, scale.to_bits()], batch * n * d);
    }

    /// Gather `tables` rows of width `sub` per sequence (`u32::MAX` rows are zero).
    pub fn gather_rows(&self, s: &mut Vec<Step>, rows: &DeviceBuffer, table: &DeviceBuffer, out: &DeviceBuffer, (batch, tables, sub): (u32, u32, u32)) {
        self.push(s, "mirage_gather_rows", &[rows, table, out], &[batch, tables, sub], batch * tables * sub);
    }

    /// The engram gate: `y += sigmoid(<rms(y), rms(key)> / sqrt(d)) * value`.
    pub fn engram_gate(&self, s: &mut Vec<Step>, y: &DeviceBuffer, key: &DeviceBuffer, value: &DeviceBuffer, rows: u32, d: u32) {
        let p = [d, rows, NORM_EPS.to_bits(), (1.0 / (d as f32).sqrt()).to_bits()];
        s.push(self.gpu().dispatch(self.k("mirage_engram_gate"), &[y, key, value], &p, gpu_core::Dispatch::Workgroups(rows)));
    }

    /// Lane maps of a hyper-connection layer (see the kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn mhc_maps(&self, s: &mut Vec<Step>, g: MhcShape, a: [f32; 3], phi: &DeviceBuffer, b: [&DeviceBuffer; 3], maps: &DeviceBuffer) {
        let p = [g.rows, g.n, g.lane_id, g.iters, a[0].to_bits(), a[1].to_bits(), a[2].to_bits()];
        self.push(s, "mirage_mhc_maps", &[phi, b[0], b[1], b[2], maps], &p, g.rows);
    }

    /// `u = sum_j h_pre[j] * lanes[j]`.
    pub fn mhc_mix(&self, s: &mut Vec<Step>, lanes: &DeviceBuffer, maps: &DeviceBuffer, u: &DeviceBuffer, (rows, n, d): (u32, u32, u32)) {
        self.push(s, "mirage_mhc_mix", &[lanes, maps, u], &[rows, n, d], rows * d);
    }

    /// `next = h_res lanes + h_post (y - u)`.
    #[allow(clippy::too_many_arguments)]
    pub fn mhc_post(&self, s: &mut Vec<Step>, lanes: &DeviceBuffer, y: &DeviceBuffer, u: &DeviceBuffer, maps: &DeviceBuffer, next: &DeviceBuffer, (rows, n, d): (u32, u32, u32)) {
        self.push(s, "mirage_mhc_post", &[lanes, y, u, maps, next], &[rows, n, d], rows * n * d);
    }

    /// Causal tap convolution at `pos`: `src` is the current raw row, which is also stored into `hist`.
    #[allow(clippy::too_many_arguments)]
    pub fn taps(&self, s: &mut Vec<Step>, src: &DeviceBuffer, hist: &DeviceBuffer, taps: &DeviceBuffer, out: &DeviceBuffer, pos: &DeviceBuffer, (rows, w, maxt): (u32, u32, u32), ntaps: u32, dilation: u32, (src_stride, src_off): (u32, u32)) {
        self.push(s, "mirage_taps", &[src, hist, taps, out, pos], &[rows, w, maxt, ntaps, dilation, src_stride, src_off], rows * w);
    }

    /// `out = att * sigmoid(gate)`.
    pub fn sigmoid_mul(&self, s: &mut Vec<Step>, att: &DeviceBuffer, gate: &DeviceBuffer, out: &DeviceBuffer, total: u32, w: u32, (gate_stride, gate_off): (u32, u32)) {
        self.push(s, "mirage_sigmoid_mul", &[att, gate, out], &[total, w, gate_stride, gate_off], total);
    }

    /// `hist[b, pos] = src[b]`.
    pub fn row_write(&self, s: &mut Vec<Step>, src: &DeviceBuffer, hist: &DeviceBuffer, pos: &DeviceBuffer, (rows, w, maxt): (u32, u32, u32)) {
        self.push(s, "mirage_row_write", &[src, hist, pos], &[rows, w, maxt], rows * w);
    }

    /// Rotary embedding of position `pos` on `rows` head vectors of width `hd`.
    pub fn rope_pos(&self, s: &mut Vec<Step>, t: &DeviceBuffer, pos: &DeviceBuffer, rows: u32, hd: u32, base: f32) {
        self.push(s, "mirage_rope_pos", &[t, pos], &[rows, hd, base.to_bits()], rows * (hd / 2));
    }

    /// The Hadamard-MLP middle stage: `zc` is `[rows, rank + n]` (conditioning scores then the first rotation), `out` is `[rows, n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn hada_mid(&self, s: &mut Vec<Step>, zc: &DeviceBuffer, cond_u: &DeviceBuffer, d2: &DeviceBuffer, b2: &DeviceBuffer, out: &DeviceBuffer, (rows, n, rank): (u32, u32, u32)) {
        self.push(s, "mirage_hada_mid", &[zc, cond_u, d2, b2, out], &[rows, n, rank], rows * n);
    }

    /// Mean over lanes.
    pub fn lane_mean(&self, s: &mut Vec<Step>, lanes: &DeviceBuffer, out: &DeviceBuffer, (rows, n, d): (u32, u32, u32)) {
        self.push(s, "mirage_lane_mean", &[lanes, out], &[rows, n, d], rows * d);
    }

    /// Row-wise argmax as `f32` indices.
    pub fn argmax(&self, s: &mut Vec<Step>, x: &DeviceBuffer, out: &DeviceBuffer, m: u32, n: u32) {
        s.push(self.gpu().dispatch(self.k("mirage_argmax_wg"), &[x, out], &[m, n], gpu_core::Dispatch::Workgroups(m)));
    }
    /// `tok[b] = hist[b, pos[0]]`.
    pub fn tok_at(&self, s: &mut Vec<Step>, hist: &DeviceBuffer, tok: &DeviceBuffer, pos: &DeviceBuffer, batch: u32, maxt: u32) {
        self.push(s, "mirage_tok_at", &[hist, tok, pos], &[batch, maxt], batch);
    }

    /// `hist[b, pos[0] + 1] = picked[b]` once `pos[0] + 1 >= first`.
    pub fn tok_put(&self, s: &mut Vec<Step>, picked: &DeviceBuffer, hist: &DeviceBuffer, pos: &DeviceBuffer, batch: u32, maxt: u32, first: u32) {
        self.push(s, "mirage_tok_put", &[picked, hist, pos], &[batch, maxt, first], batch);
    }

    /// `pos[0] += 1`.
    pub fn pos_inc(&self, s: &mut Vec<Step>, pos: &DeviceBuffer) {
        self.push(s, "mirage_pos_inc", &[pos], &[0], 1);
    }

    /// Engram table rows of position `pos`; `orders` has at most four entries.
    #[allow(clippy::too_many_arguments)]
    pub fn engram_rows(&self, s: &mut Vec<Step>, hist: &DeviceBuffer, rows: &DeviceBuffer, pos: &DeviceBuffer, (batch, maxt): (u32, u32), (tables, hashes, slots): (u32, u32, u32), orders: &[usize]) {
        assert!(orders.len() <= 4, "at most four engram orders");
        let o = |i: usize| orders.get(i).copied().unwrap_or(0) as u32;
        self.push(s, "mirage_engram_rows", &[hist, rows, pos], &[batch, maxt, tables, hashes, slots, o(0), o(1), o(2), o(3)], batch * tables);
    }
}

/// Geometry of one hyper-connection layer's maps.
#[derive(Debug, Clone, Copy)]
pub struct MhcShape {
    /// Sequences.
    pub rows: u32,
    /// Lanes (at most 4).
    pub n: u32,
    /// The lane this layer favours.
    pub lane_id: u32,
    /// Sinkhorn iterations.
    pub iters: u32,
}
