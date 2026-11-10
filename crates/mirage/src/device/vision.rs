// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vision half on the device, batched over images.
//!
//! One forward runs `batch` letterboxed images through the stem, the body,
//! the attention top, the dense head and (on demand) the mask branches. Every
//! image occupies the full canvas grid; the cells outside its resized image
//! are zeroed by per-image masks exactly where the host operators zero them,
//! so image `i` of a batch equals a batch of one. The attention top runs over
//! all `cells + registers` tokens of an image with a key mask instead of
//! gathering the valid cells: padding rows never reach a valid row, because
//! every other operator in the top is row-wise.

use gpu_core::{DeviceBuffer, Step};
use model::ops::Weight;

use super::prims::{AttnShape, ConvShape};
use super::{hconv::DeviceHConv, Device};
use crate::config::{MirageConfig, DENSE_BLOCKS, DENSE_OUT_CHANNELS, MASK_BRANCH_BLOCKS, MASK_RANK};
use crate::vision::dense::{Block as DenseBlock, DenseHead};
use crate::vision::encoder::Encoder;
use crate::vision::ops::{self, valid_cells, Linear, Map};
use crate::vision::VisionModel;

fn lin(dev: &Device, name: &str, l: &Linear) -> Weight {
    dev.linear_weight(name, l)
}

/// Per-image validity masks (`1` inside the resized image, `0` on padding) for
/// every grid size the vision path masks at.
pub struct Masks {
    sides: Vec<(usize, DeviceBuffer)>,
    /// `[batch, cells + registers]`: valid cells, registers zero (output rows).
    pub cells_out: DeviceBuffer,
    /// `[batch, cells + registers]`: valid cells and registers (attention keys).
    pub keys: DeviceBuffer,
}

impl Masks {
    /// Masks of `batch` images whose resized sizes are `sizes` (`(h, w)`).
    pub fn new(dev: &Device, cfg: &MirageConfig, sizes: &[(usize, usize)]) -> Masks {
        let side_mask = |side: usize| {
            let flat: Vec<f32> = sizes.iter().flat_map(|&(h, w)| valid_cells(h, w, side).into_iter().map(|v| f32::from(u8::from(v)))).collect();
            dev.buf(&flat)
        };
        let sides = (1..=5).map(|k| cfg.canvas >> k).map(|s| (s, side_mask(s))).collect();
        let grid = cfg.grid();
        let (mut out, mut keys) = (Vec::new(), Vec::new());
        for &(h, w) in sizes {
            let valid = valid_cells(h, w, grid);
            out.extend(valid.iter().map(|&v| f32::from(u8::from(v))).chain(std::iter::repeat_n(0.0, cfg.registers)));
            keys.extend(valid.iter().map(|&v| f32::from(u8::from(v))).chain(std::iter::repeat_n(1.0, cfg.registers)));
        }
        Masks { sides, cells_out: dev.buf(&out), keys: dev.buf(&keys) }
    }

    /// The mask of a `side x side` grid.
    pub fn side(&self, side: usize) -> &DeviceBuffer {
        &self.sides.iter().find(|(s, _)| *s == side).expect("a known grid size").1
    }
}

struct DStem {
    w0: DeviceBuffer,
    norms: Vec<DeviceBuffer>,
    dw: Vec<DeviceBuffer>,
    pw: Vec<Weight>,
    widths: Vec<usize>,
}

struct DBody {
    mix_norm: DeviceBuffer,
    dw: [DeviceBuffer; 3],
    b_value: DeviceBuffer,
    b_gate: DeviceBuffer,
    hconv: DeviceHConv,
    gamma: DeviceBuffer,
    beta: DeviceBuffer,
}

struct DTop {
    attn_norm: DeviceBuffer,
    qkv: [Weight; 3],
    gate: Weight,
    out: Weight,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    post_attn_norm: DeviceBuffer,
    attn_gate: f32,
    hconv: DeviceHConv,
    level_norm: DeviceBuffer,
    level_gates: DeviceBuffer,
}

struct DBlock {
    norm: DeviceBuffer,
    dw: DeviceBuffer,
    pw: Weight,
}

impl DBlock {
    fn new(dev: &Device, b: &DenseBlock) -> DBlock {
        DBlock { norm: dev.norm_weight(&b.norm), dw: dev.buf(&b.dw), pw: lin(dev, "dense_pw", &b.pw) }
    }
}

struct DBranch {
    blocks: Vec<DBlock>,
    final_norm: DeviceBuffer,
    head: Weight,
}

struct DDense {
    coarse: Weight,
    mid: Weight,
    fine: Weight,
    edge: Weight,
    neck_norm: DeviceBuffer,
    neck_dw: DeviceBuffer,
    neck_pw: Weight,
    blocks: Vec<DBlock>,
    final_norm: DeviceBuffer,
    out: Weight,
    bias: DeviceBuffer,
    proto: DBranch,
    coef: DBranch,
}

/// Encoder results of a batch, resident.
pub struct EncoderBatch {
    /// Images in the batch.
    pub batch: usize,
    /// Stem outputs, `[batch, h, w, c]` each, post-activation, padding zero.
    pub stages: Vec<(Map, DeviceBuffer)>,
    /// Body output `[batch, cells, dim]`.
    pub body: DeviceBuffer,
    /// Top tokens `[batch, cells + registers, dim]`.
    pub top: DeviceBuffer,
    /// Encoder memory `[batch, cells, dim]`, padding rows zero.
    pub encoder: DeviceBuffer,
}

/// Dense-head results of a batch, resident.
pub struct DenseBatch {
    /// `[batch, side * side, 128]`.
    pub head_in: DeviceBuffer,
    /// `[batch, side * side, 10]`.
    pub head: DeviceBuffer,
}

/// The vision weights resident on a device.
pub struct DeviceVision {
    cfg: MirageConfig,
    stem: DStem,
    body: Vec<DBody>,
    top: Vec<DTop>,
    registers: DeviceBuffer,
    enc_norm: DeviceBuffer,
    dense: DDense,
}

impl DeviceVision {
    /// Upload the host-prepared vision weights.
    pub fn upload(dev: &Device, vm: &VisionModel) -> DeviceVision {
        let cfg = vm.model_config().clone();
        let e: &Encoder = &vm.encoder;
        // One Walsh transform serves every mixer (nothing is uploaded when the mixers are folded).
        let walsh = (dev.form() == super::WeightForm::Structured).then(|| super::kron::Kron::walsh(dev, cfg.dim));
        let stem = DStem {
            w0: dev.buf(&e.stem.w0),
            norms: e.stem.norms.iter().map(|n| dev.norm_weight(n)).collect(),
            dw: e.stem.dw.iter().map(|w| dev.buf(w)).collect(),
            pw: e.stem.pw.iter().map(|l| lin(dev, "stem_pw", l)).collect(),
            widths: e.stem.widths.clone(),
        };
        let body = e
            .body
            .iter()
            .map(|b| DBody {
                mix_norm: dev.norm_weight(&b.mix_norm),
                dw: [dev.buf(&b.dw_gate), dev.buf(&b.dw_value), dev.buf(&b.dw_down)],
                b_value: dev.buf(&b.b_value),
                b_gate: dev.buf(&b.b_gate),
                hconv: DeviceHConv::upload(dev, &b.hconv, walsh.as_ref()),
                gamma: dev.buf(&b.gamma),
                beta: dev.buf(&b.beta),
            })
            .collect();
        let top = e
            .top
            .iter()
            .enumerate()
            .map(|(l, t)| DTop {
                attn_norm: dev.norm_weight(&t.attn_norm),
                qkv: [lin(dev, "top_q", &t.q), lin(dev, "top_k", &t.k), lin(dev, "top_v", &t.v)],
                gate: lin(dev, "top_gate", &t.gate),
                out: lin(dev, "top_out", &t.out),
                q_norm: dev.norm_weight(&t.q_norm),
                k_norm: dev.norm_weight(&t.k_norm),
                post_attn_norm: dev.norm_weight(&t.post_attn_norm),
                attn_gate: ops::sigmoid(t.attn_gate),
                hconv: DeviceHConv::upload(dev, &t.hconv, walsh.as_ref()),
                level_norm: dev.norm_weight(&e.level_norm[l]),
                level_gates: dev.buf(&e.level_gates[l * cfg.dim..(l + 1) * cfg.dim]),
            })
            .collect();
        let d: &DenseHead = &vm.dense;
        let branch = |b: &crate::vision::dense::Branch| DBranch {
            blocks: b.blocks.iter().map(|x| DBlock::new(dev, x)).collect(),
            final_norm: dev.norm_weight(&b.final_norm),
            head: lin(dev, "branch_head", &b.head),
        };
        let dense = DDense {
            coarse: lin(dev, "coarse", &d.coarse),
            mid: lin(dev, "mid", &d.mid),
            fine: lin(dev, "fine", &d.fine),
            edge: lin(dev, "edge", &d.edge),
            neck_norm: dev.norm_weight(&d.neck_norm),
            neck_dw: dev.buf(&d.neck_dw),
            neck_pw: lin(dev, "neck_pw", &d.neck_pw),
            blocks: d.blocks.iter().map(|b| DBlock::new(dev, b)).collect(),
            final_norm: dev.norm_weight(&d.final_norm),
            out: lin(dev, "out", &d.out),
            bias: dev.buf(&d.bias),
            proto: branch(&d.proto),
            coef: branch(&d.coef),
        };
        DeviceVision { stem, body, top, registers: dev.buf(&e.registers), enc_norm: dev.norm_weight(&e.enc_norm), dense, cfg }
    }

    fn conv(&self, h: usize, w: usize, batch: usize, k: usize, s: usize) -> ConvShape {
        ConvShape { batch: batch as u32, h: h as u32, w: w as u32, kh: k as u32, kw: k as u32, sy: s as u32, sx: s as u32 }
    }

    /// `x + dw3x3(x)`-free depthwise convenience: depthwise 3x3 stride 1 into a new buffer.
    fn dw3(&self, dev: &Device, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, batch: usize, side: usize, c: usize) -> DeviceBuffer {
        let y = dev.alloc(batch * side * side * c);
        dev.dwconv(s, self.conv(side, side, batch, 3, 1), c as u32, x, w, &y);
        y
    }

    /// Run the stem, body and top on `canvas` (`[batch, 512, 512, 3]`).
    pub fn encode(&self, dev: &Device, canvas: &DeviceBuffer, batch: usize, masks: &Masks) -> EncoderBatch {
        let cfg = &self.cfg;
        let mut s = Vec::new();
        let size = cfg.canvas;
        let mut stages = Vec::new();
        let st = &self.stem;
        let c0 = st.widths[0];
        let g0 = self.conv(size, size, batch, 3, 2);
        let mut cur = dev.alloc(batch * (size / 2) * (size / 2) * c0);
        dev.conv2d(&mut s, g0, 3, c0 as u32, canvas, &st.w0, &cur);
        let mut m = Map { h: size / 2, w: size / 2, c: c0 };
        for k in 0..st.widths.len() {
            if k > 0 {
                let ho = m.h / 2;
                let d = dev.alloc(batch * ho * ho * m.c);
                dev.dwconv(&mut s, self.conv(m.h, m.w, batch, 3, 2), m.c as u32, &cur, &st.dw[k - 1], &d);
                m = Map { h: ho, w: ho, c: st.widths[k] };
                cur = dev.linear_new(&mut s, &st.pw[k - 1], &d, (batch * ho * ho) as u32);
            }
            let rows = (batch * m.h * m.w) as u32;
            let z = dev.zc_new(&mut s, &cur, &st.norms[k], rows, m.c as u32);
            cur = dev.silu_new(&mut s, &z, rows * m.c as u32);
            dev.rows_mask(&mut s, &cur, masks.side(m.h), rows, m.c as u32);
            stages.push((m, cur.clone()));
            dev.flush(&mut s);
        }
        drop(cur);

        let (grid, d) = (cfg.grid(), cfg.dim);
        let cells = grid * grid;
        let rows = (batch * cells) as u32;
        let mut x = dev.alloc(batch * cells * d);
        dev.copy_rows(&mut s, &stages[stages.len() - 1].1, (cells as u32, 0), &x, (cells as u32, 0), batch as u32, cells as u32, (cfg.stem[cfg.stem.len() - 1]) as u32);
        assert_eq!(cfg.stem[cfg.stem.len() - 1], d, "the last stem stage feeds the body");
        dev.flush(&mut s);
        for b in &self.body {
            let mut hh = dev.zc_new(&mut s, &x, &b.mix_norm, rows, d as u32);
            dev.rows_mask(&mut s, &hh, masks.side(grid), rows, d as u32);
            let g = self.dw3(dev, &mut s, &hh, &b.dw[0], batch, grid, d);
            let v = self.dw3(dev, &mut s, &hh, &b.dw[1], batch, grid, d);
            dev.gated_mix(&mut s, &g, &v, &b.b_gate, &b.b_value, rows, d as u32);
            let down = self.dw3(dev, &mut s, &g, &b.dw[2], batch, grid, d);
            dev.add_masked(&mut s, &x, &down, masks.side(grid), rows, d as u32);
            let y = dev.alloc(batch * cells * d);
            b.hconv.encode(dev, &mut s, &x, rows, &y);
            dev.grn_residual(&mut s, &x, &y, masks.side(grid), &b.gamma, &b.beta, batch as u32, cells as u32, d as u32);
            hh = y;
            let _ = &hh;
            dev.flush(&mut s);
        }
        let body = dev.alloc(batch * cells * d);
        dev.copy_rows(&mut s, &x, (cells as u32, 0), &body, (cells as u32, 0), batch as u32, cells as u32, d as u32);

        let (top, encoder) = self.attention_top(dev, &mut s, &x, batch, masks);
        x = body.clone();
        drop(x);
        dev.flush(&mut s);
        EncoderBatch { batch, stages, body, top, encoder }
    }

    fn attention_top(&self, dev: &Device, s: &mut Vec<Step>, body: &DeviceBuffer, batch: usize, masks: &Masks) -> (DeviceBuffer, DeviceBuffer) {
        let c = &self.cfg;
        let (d, cells, regs) = (c.dim, c.grid() * c.grid(), c.registers);
        let tokens = cells + regs;
        let rows = (batch * tokens) as u32;
        let mut x = dev.alloc(batch * tokens * d);
        dev.copy_rows(s, body, (cells as u32, 0), &x, (tokens as u32, 0), batch as u32, cells as u32, d as u32);
        dev.copy_rows(s, &self.registers, (0, 0), &x, (tokens as u32, cells as u32), batch as u32, regs as u32, d as u32);
        let acc = dev.alloc_zeroed(batch * tokens * d);
        let (hq, hk, qk, vh) = (c.attn_heads, c.attn_kv_heads, c.attn_qk_dim, c.attn_v_dim);
        for t in &self.top {
            let level = dev.zc_new(s, &x, &t.level_norm, rows, d as u32);
            dev.chan_fma(s, &level, &t.level_gates, &acc, rows * d as u32, d as u32);
            let hh = dev.zc_new(s, &x, &t.attn_norm, rows, d as u32);
            let q0 = dev.linear_new(s, &t.qkv[0], &hh, rows);
            let k0 = dev.linear_new(s, &t.qkv[1], &hh, rows);
            let v = dev.linear_new(s, &t.qkv[2], &hh, rows);
            let q = dev.zc_new(s, &q0, &t.q_norm, rows * hq as u32, qk as u32);
            let k = dev.zc_new(s, &k0, &t.k_norm, rows * hk as u32, qk as u32);
            dev.rope2d(s, &q, batch as u32, tokens as u32, cells as u32, c.grid() as u32, hq as u32, qk as u32, c.enc_rope);
            dev.rope2d(s, &k, batch as u32, tokens as u32, cells as u32, c.grid() as u32, hk as u32, qk as u32, c.enc_rope);
            let att = dev.alloc(batch * tokens * hq * vh);
            let shape = AttnShape { batch: batch as u32, nq: tokens as u32, heads: hq as u32, kvh: hk as u32, hd: qk as u32, vd: vh as u32, nk: tokens as u32, visible: tokens as u32 };
            dev.attention(s, shape, &q, &k, &v, &masks.keys, &att);
            let gate = dev.linear_new(s, &t.gate, &hh, rows);
            let sg = dev.alloc(batch * tokens * hq * vh);
            dev.sigmoid(s, &gate, &sg, rows * (hq * vh) as u32);
            let gated = dev.alloc(batch * tokens * hq * vh);
            dev.mul(s, &att, &sg, &gated, rows * (hq * vh) as u32);
            let o0 = dev.linear_new(s, &t.out, &gated, rows);
            let o = dev.zc_new(s, &o0, &t.post_attn_norm, rows, d as u32);
            dev.axpy(s, t.attn_gate, &o, &x, rows * d as u32);
            let y = dev.alloc(batch * tokens * d);
            t.hconv.encode(dev, s, &x, rows, &y);
            dev.add(s, &x, &y, rows * d as u32);
            x = y;
            dev.flush(s);
        }
        let encoded = dev.zc_new(s, &x, &self.enc_norm, rows, d as u32);
        dev.add(s, &acc, &encoded, rows * d as u32);
        dev.rows_mask(s, &encoded, &masks.cells_out, rows, d as u32);
        let encoder = dev.alloc(batch * cells * d);
        dev.copy_rows(s, &encoded, (tokens as u32, 0), &encoder, (cells as u32, 0), batch as u32, cells as u32, d as u32);
        dev.flush(s);
        (x, encoder)
    }

    /// `silu(zc(y, norm))` into a new buffer.
    fn act(&self, dev: &Device, s: &mut Vec<Step>, y: &DeviceBuffer, norm: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
        let z = dev.zc_new(s, y, norm, rows, d);
        dev.silu_new(s, &z, rows * d)
    }

    /// The residual blocks and activated head of a mask branch. `y` is consumed.
    fn branch(&self, dev: &Device, s: &mut Vec<Step>, mut y: DeviceBuffer, b: &DBranch, batch: usize, side: usize, m: &DeviceBuffer) -> DeviceBuffer {
        let (d, rows) = (self.cfg.dense_dim, (batch * side * side) as u32);
        for blk in &b.blocks {
            let a = self.act(dev, s, &y, &blk.norm, rows, d as u32);
            let dw = self.dw3(dev, s, &a, &blk.dw, batch, side, d);
            let r = dev.linear_new(s, &blk.pw, &dw, rows);
            dev.add(s, &y, &r, rows * d as u32);
            dev.rows_mask(s, &r, m, rows, d as u32);
            y = r;
            dev.flush(s);
        }
        let a = self.act(dev, s, &y, &b.final_norm, rows, d as u32);
        dev.linear_new(s, &b.head, &a, rows)
    }

    /// The detection map for an encoded batch.
    pub fn dense(&self, dev: &Device, enc: &EncoderBatch, masks: &Masks) -> DenseBatch {
        let c = &self.cfg;
        let (dd, grid, batch) = (c.dense_dim, c.grid(), enc.batch);
        let (_, fine_level, mid_level) = c.dense_levels();
        let dn = &self.dense;
        let mut s = Vec::new();
        let (g32, g64) = (grid * 2, grid * 4);
        let (r32, r64) = ((batch * g32 * g32) as u32, (batch * g64 * g64) as u32);
        let coarse = dev.linear_new(&mut s, &dn.coarse, &enc.encoder, (batch * grid * grid) as u32);
        let d32 = dev.alloc(batch * g32 * g32 * dd);
        dev.upsample2(&mut s, &coarse, &d32, batch as u32, grid as u32, grid as u32, dd as u32);
        let (mid_map, mid_x) = &enc.stages[mid_level];
        let skip = dev.linear_new(&mut s, &dn.mid, mid_x, (batch * mid_map.h * mid_map.w) as u32);
        dev.add(&mut s, &skip, &d32, r32 * dd as u32);
        dev.rows_mask(&mut s, &d32, masks.side(g32), r32, dd as u32);
        let n = self.act(dev, &mut s, &d32, &dn.neck_norm, r32, dd as u32);
        let nd = self.dw3(dev, &mut s, &n, &dn.neck_dw, batch, g32, dd);
        let neck = dev.linear_new(&mut s, &dn.neck_pw, &nd, r32);
        dev.rows_mask(&mut s, &neck, masks.side(g32), r32, dd as u32);
        dev.add(&mut s, &neck, &d32, r32 * dd as u32);
        dev.flush(&mut s);

        let d64 = dev.alloc(batch * g64 * g64 * dd);
        dev.upsample2(&mut s, &d32, &d64, batch as u32, g32 as u32, g32 as u32, dd as u32);
        let (fine_map, fine_x) = &enc.stages[fine_level];
        let skip = dev.linear_new(&mut s, &dn.fine, fine_x, (batch * fine_map.h * fine_map.w) as u32);
        dev.add(&mut s, &skip, &d64, r64 * dd as u32);
        assert_eq!(dn.blocks.len(), DENSE_BLOCKS);
        for blk in &dn.blocks {
            let a = self.act(dev, &mut s, &d64, &blk.norm, r64, dd as u32);
            let dw = self.dw3(dev, &mut s, &a, &blk.dw, batch, g64, dd);
            let r = dev.linear_new(&mut s, &blk.pw, &dw, r64);
            dev.add(&mut s, &r, &d64, r64 * dd as u32);
            dev.flush(&mut s);
        }
        let head_in = self.act(dev, &mut s, &d64, &dn.final_norm, r64, dd as u32);
        dev.rows_mask(&mut s, &head_in, masks.side(g64), r64, dd as u32);
        let head = dev.linear_new(&mut s, &dn.out, &head_in, r64);
        dev.bias_add(&mut s, &head, &dn.bias, r64, DENSE_OUT_CHANNELS as u32);
        dev.flush(&mut s);
        DenseBatch { head_in, head }
    }

    /// Mask prototypes `[batch, 128 * 128, 32]` and coefficients `[batch, 64 * 64, 32]`.
    pub fn mask_maps(&self, dev: &Device, enc: &EncoderBatch, dense: &DenseBatch, masks: &Masks) -> (DeviceBuffer, DeviceBuffer) {
        let c = &self.cfg;
        let (dd, batch, side) = (c.dense_dim, enc.batch, c.canvas / c.stride);
        let (edge_level, _, _) = c.dense_levels();
        let dn = &self.dense;
        let mut s = Vec::new();
        let up = side * 2;
        let ru = (batch * up * up) as u32;
        let u = dev.alloc(batch * up * up * dd);
        dev.upsample2(&mut s, &dense.head_in, &u, batch as u32, side as u32, side as u32, dd as u32);
        let (edge_map, edge_x) = &enc.stages[edge_level];
        let skip = dev.linear_new(&mut s, &dn.edge, edge_x, (batch * edge_map.h * edge_map.w) as u32);
        dev.add(&mut s, &skip, &u, ru * dd as u32);
        dev.rows_mask(&mut s, &u, masks.side(up), ru, dd as u32);
        let protos = self.branch(dev, &mut s, u, &dn.proto, batch, up, masks.side(up));
        let rc = (batch * side * side) as u32;
        let c_in = dev.alloc(batch * side * side * dd);
        dev.copy_rows(&mut s, &dense.head_in, (1, 0), &c_in, (1, 0), 1, rc, dd as u32);
        dev.rows_mask(&mut s, &c_in, masks.side(side), rc, dd as u32);
        let coefs = self.branch(dev, &mut s, c_in, &dn.coef, batch, side, masks.side(side));
        assert!(MASK_BRANCH_BLOCKS == dn.proto.blocks.len() && MASK_RANK > 0);
        dev.flush(&mut s);
        (protos, coefs)
    }
}

/// The valid-cell grid of the encoder, for callers that need it on the host.
pub fn host_masks(cfg: &MirageConfig, h: usize, w: usize) -> Vec<bool> {
    valid_cells(h, w, cfg.grid())
}
