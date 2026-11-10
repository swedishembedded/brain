// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vision half of Mirage as a training graph: stem, body, attention top with level memory, dense and mask
//! heads, the text reader and the decoder memory. Each stage mirrors its inference operator in `crate::vision`
//! (the parity tests hold the two to the same values) and is built from the checked primitives.

use std::rc::Rc;

use super::graph::Graph;
use super::net::Net;
use super::ops::Seg;
use super::real::{r, Real};
use super::seq::AttnShape;
use super::tape::Id;
use crate::config::{DENSE_BLOCKS, DENSE_OUT_CHANNELS, MASK_BRANCH_BLOCKS, MASK_RANK, READER_MIX_KERNEL, READER_STAGES};
use crate::vision::masks::round_box;
use crate::vision::memory::{box_code, pooled_cells, sinusoid, Slot};
use crate::vision::ops::{valid_cells, Map};
use crate::vision::reader::FRAMES;

/// Encoder values later stages read.
#[derive(Clone)]
pub struct EncT {
    /// Stem stage outputs by stride, each with its map.
    pub stages: Vec<(Map, Id)>,
    /// The encoder memory `E`, `[grid^2, dim]`, padding rows zero.
    pub memory: Id,
    /// Token matrix of the top, `[grid^2 + registers, dim]`.
    pub top: Id,
}

/// Dense-head values.
pub struct DenseT {
    /// The head input, `[side^2, dense_dim]`.
    pub head_in: Id,
    /// The detection map, `[side^2, 10]`.
    pub head: Id,
}

impl<R: Real, T: Graph<R>> Net<'_, R, T> {
    /// `HConvGLU(ZC(x, hconv_norm))` of the block `prefix` on `rows` rows.
    pub fn hconv(&mut self, x: Id, prefix: &str) -> Id {
        let (n, k) = (self.cfg.dim, self.cfg.hconv_kernel);
        let (norm, gate, gate_bias, value, out) = (
            self.p(&format!("{prefix}/hconv_norm/scale")),
            self.p(&format!("{prefix}/hconv/gate_conv")),
            self.p(&format!("{prefix}/hconv/gate_bias")),
            self.p(&format!("{prefix}/hconv/value_conv")),
            self.p(&format!("{prefix}/hconv/out_conv")),
        );
        let u = self.tape.zc_norm(x, n, norm);
        let u = self.tape.walsh(u, n);
        let a = self.tape.band(u, n, gate, k);
        let v = self.tape.band(u, n, value, k);
        let a = self.tape.bias_add(a, gate_bias);
        let a = self.tape.gelu_tanh(a);
        let a = self.tape.mul(a, v);
        let y = self.tape.band(a, n, out, k);
        self.tape.walsh(y, n)
    }

    /// `silu(zc_norm(x))` over `d` channels with the norm `name`.
    fn norm_silu(&mut self, x: Id, d: usize, name: &str) -> Id {
        let s = self.p(name);
        let n = self.tape.zc_norm(x, d, s);
        self.tape.silu(n)
    }

    /// A recompute block with one output (see [`Net::block`]).
    pub(super) fn block1(&mut self, build: impl Fn(&mut Self) -> Id + 'static) -> Id {
        self.block(move |net| vec![build(net)])[0]
    }

    /// Stage `k` of the stem on `input` (the canvas for stage 0): a stride-2 convolution, or a depthwise +
    /// pointwise stride-2 pair, then the norm and re-masking. Padding stays zero.
    fn stem_stage(&mut self, k: usize, input: (Id, Map), h: usize, w: usize) -> (Id, Map) {
        let widths = self.cfg.stem.clone();
        let (x, m) = input;
        let (x, m) = if k == 0 {
            let w0 = self.p("stem/w0");
            self.tape.conv_dense(x, m, w0, 3, 3, widths[0], 2, 2)
        } else {
            let dw = self.p(&format!("stem/dw_{k}"));
            let (d, dm) = self.tape.depthwise(x, m, dw, 3, 3, 2, 2);
            let x = self.proj(d, &format!("stem/pw_{k}/kernel"), None, dm.h * dm.w, widths[k - 1], widths[k]);
            (x, Map { h: dm.h, w: dm.w, c: widths[k] })
        };
        let x = self.norm_silu(x, m.c, &format!("stem/norm_{k}/scale"));
        (self.tape.mask_rows(x, m.c, &valid_cells(h, w, m.h)), m)
    }

    /// The stem: five stages, each one recompute block.
    pub fn stem(&mut self, canvas: Id, h: usize, w: usize) -> Vec<(Map, Id)> {
        let stages = self.cfg.stem.len();
        let mut input = (canvas, Map { h: self.cfg.canvas, w: self.cfg.canvas, c: 3 });
        let mut out = Vec::with_capacity(stages);
        for k in 0..stages {
            let (x, m) = self.block_map(move |net| net.stem_stage(k, input, h, w));
            input = (x, m);
            out.push((m, x));
        }
        out
    }

    fn body_block(&mut self, x: Id, i: usize, gm: Map, valid: &[bool]) -> Id {
        let (d, mk) = (gm.c, self.cfg.mix_kernel);
        let p = format!("body_{i}");
        let mix_norm = self.p(&format!("{p}/mix_norm/scale"));
        let hh = self.tape.zc_norm(x, d, mix_norm);
        let hh = self.tape.mask_rows(hh, d, valid);
        let (dw_gate, dw_value, dw_down) = (self.p(&format!("{p}/mix/dw_gate")), self.p(&format!("{p}/mix/dw_value")), self.p(&format!("{p}/mix/dw_down")));
        let (b_gate, b_value) = (self.p(&format!("{p}/mix/b_gate")), self.p(&format!("{p}/mix/b_value")));
        let (g, _) = self.tape.depthwise(hh, gm, dw_gate, mk, mk, 1, 1);
        let (v, _) = self.tape.depthwise(hh, gm, dw_value, mk, mk, 1, 1);
        let g = self.tape.bias_add(g, b_gate);
        let g = self.tape.silu(g);
        let v = self.tape.bias_add(v, b_value);
        let gv = self.tape.mul(g, v);
        let (down, _) = self.tape.depthwise(gv, gm, dw_down, mk, mk, 1, 1);
        let down = self.tape.mask_rows(down, d, valid);
        let x = self.tape.add(x, down);

        let y = self.hconv(x, &p);
        let gamma = self.p(&format!("{p}/grn/gamma"));
        let beta = self.p(&format!("{p}/grn/beta"));
        let scale = self.tape.grn_scale(y, d, valid, gamma);
        let ys = self.tape.chan_mul(y, scale);
        let ys = self.tape.bias_add(ys, beta);
        let ys = self.tape.mask_rows(ys, d, valid);
        self.tape.add(x, ys)
    }

    /// The 2D rotary angles of the encoder's live cells (registers rotate by zero): `[tokens, pairs]`.
    fn encoder_angles(&self, live: &[usize], tokens: usize) -> Vec<f64> {
        let c = self.cfg;
        let (grid, hd) = (c.grid(), c.attn_qk_dim);
        let pairs = hd / 2;
        let per_axis = pairs / 2;
        // Same arithmetic as the inference operator: the frequency is formed in f32.
        let freq: Vec<f32> = (0..per_axis).map(|j| c.enc_rope.powf(-(j as f32) / per_axis as f32)).collect();
        let mut angles = vec![0f64; tokens * pairs];
        for (row, &cell) in live.iter().enumerate() {
            let (cy, cx) = ((cell / grid) as f32, (cell % grid) as f32);
            for j in 0..pairs {
                let pos = if j < per_axis { cy } else { cx };
                angles[row * pairs + j] = f64::from(pos * freq[j % per_axis]);
            }
        }
        angles
    }

    /// Attention layer `l` of the top on the token matrix `x` (`nv` live rows then the registers): the level-memory
    /// contribution added to `acc`, then gated attention and the HConvGLU. Returns the new tokens and accumulator.
    fn top_layer(&mut self, l: usize, x: Id, acc: Option<Id>, nv: usize, tokens: usize, angles: &[f64]) -> (Id, Id) {
        let c = self.cfg.clone();
        let d = c.dim;
        let (hq, hk, qk, vh) = (c.attn_heads, c.attn_kv_heads, c.attn_qk_dim, c.attn_v_dim);
        let p = format!("top_{l}");
        let level_gates = self.p("level_gates");
        let live_rows = self.tape.slice(x, 0, nv * d);
        let level_norm = self.p(&format!("level_norm_{l}/scale"));
        let level = self.tape.zc_norm(live_rows, d, level_norm);
        let gates = self.tape.slice(level_gates, l * d, d);
        let contribution = self.tape.chan_mul(level, gates);
        let acc = match acc {
            Some(a) => self.tape.add(a, contribution),
            None => contribution,
        };

        let attn_norm = self.p(&format!("{p}/attn_norm/scale"));
        let hh = self.tape.zc_norm(x, d, attn_norm);
        let q = self.proj(hh, &format!("{p}/self_attn/q_proj/kernel"), None, tokens, d, hq * qk);
        let k = self.proj(hh, &format!("{p}/self_attn/k_proj/kernel"), None, tokens, d, hk * qk);
        let v = self.proj(hh, &format!("{p}/self_attn/v_proj/kernel"), None, tokens, d, hk * vh);
        let (q_norm, k_norm) = (self.p(&format!("{p}/self_attn/q_norm/scale")), self.p(&format!("{p}/self_attn/k_norm/scale")));
        let q = self.tape.zc_norm(q, qk, q_norm);
        let k = self.tape.zc_norm(k, qk, k_norm);
        let q = self.tape.rope(q, hq, qk, angles);
        let k = self.tape.rope(k, hk, qk, angles);
        let att = self.tape.attention(q, k, v, AttnShape { tq: tokens, tk: tokens, hq, hk, qk, vh, causal: false });
        let gate = self.proj(hh, &format!("{p}/self_attn/gate_proj/kernel"), None, tokens, d, hq * vh);
        let gate = self.tape.sigmoid(gate);
        let gated = self.tape.mul(att, gate);
        let o = self.proj(gated, &format!("{p}/self_attn/out_proj/kernel"), None, tokens, hq * vh, d);
        let post = self.p(&format!("{p}/post_attn_norm/scale"));
        let o = self.tape.zc_norm(o, d, post);
        let attn_gate = self.p(&format!("{p}/attn_gate"));
        let a_gate = self.tape.sigmoid(attn_gate);
        let o = self.tape.scalar_mul(o, a_gate);
        let x2 = self.tape.add(x, o);
        let y = self.hconv(x2, &p);
        (self.tape.add(y, x2), acc)
    }

    /// Body, attention top and level memory over the stem's last map.
    pub fn encoder_top(&mut self, x: Id, h: usize, w: usize) -> (Id, Id) {
        let c = self.cfg.clone();
        let (d, grid) = (c.dim, c.grid());
        let cells = grid * grid;
        let valid = valid_cells(h, w, grid);
        let gm = Map { h: grid, w: grid, c: d };
        let mut x = x;
        let shared_valid = Rc::new(valid.clone());
        for i in 0..c.body_layers {
            let (input, v) = (x, Rc::clone(&shared_valid));
            x = self.block1(move |net| net.body_block(input, i, gm, &v));
        }
        let live: Vec<usize> = (0..cells).filter(|&i| valid[i]).collect();
        let (nv, regs) = (live.len(), c.registers);
        let tokens = nv + regs;
        let rows = self.tape.gather_rows(x, d, &live);
        let registers = self.p("registers");
        let mut x = self.tape.concat(rows, registers);
        let angles = self.encoder_angles(&live, tokens);
        let angles = Rc::new(angles);
        let mut acc: Option<Id> = None;
        for l in 0..c.attn_layers {
            let (input, previous, angles) = (x, acc, Rc::clone(&angles));
            let outs = self.block(move |net| {
                let (x, acc) = net.top_layer(l, input, previous, nv, tokens, &angles);
                vec![x, acc]
            });
            (x, acc) = (outs[0], Some(outs[1]));
        }

        let live_rows = self.tape.slice(x, 0, nv * d);
        let enc_norm = self.p("enc_norm/scale");
        let encoded = self.tape.zc_norm(live_rows, d, enc_norm);
        let encoded = match acc {
            Some(a) => self.tape.add(encoded, a),
            None => encoded,
        };
        let mut top_segs: Vec<Seg<R>> = live.iter().enumerate().map(|(row, &cell)| Seg::copy(cell * d, row * d, d)).collect();
        top_segs.push(Seg::copy(cells * d, nv * d, regs * d));
        let top = self.tape.plan(x, (cells + regs) * d, top_segs);
        let mem_segs: Vec<Seg<R>> = live.iter().enumerate().map(|(row, &cell)| Seg::copy(cell * d, row * d, d)).collect();
        let memory = self.tape.plan(encoded, cells * d, mem_segs);
        (top, memory)
    }

    /// The whole encoder on a letterboxed canvas `[canvas, canvas, 3]` (a leaf) whose image fills `h x w`.
    pub fn encoder(&mut self, canvas: Id, h: usize, w: usize) -> EncT {
        let stages = self.stem(canvas, h, w);
        let last = stages.last().expect("the stem has stages").1;
        let (top, memory) = self.encoder_top(last, h, w);
        EncT { stages, memory, top }
    }

    /// A residual block of the dense trunk or a mask branch (`norm`, `dw`, `pw` names given), re-masked when
    /// `valid` is set.
    fn trunk_block(&mut self, y: Id, m: Map, names: (String, String, String), valid: Option<&[bool]>) -> Id {
        let d = m.c;
        let a = self.norm_silu(y, d, &names.0);
        let dw = self.p(&names.1);
        let (a, _) = self.tape.depthwise(a, m, dw, 3, 3, 1, 1);
        let rr = self.proj(a, &names.2, None, m.h * m.w, d, d);
        let rr = self.tape.add(rr, y);
        match valid {
            Some(v) => self.tape.mask_rows(rr, d, v),
            None => rr,
        }
    }

    fn mask_branch(&mut self, y: Id, m: Map, valid: &[bool], pre: &str, out: &str) -> Id {
        let mut y = y;
        for i in 0..MASK_BRANCH_BLOCKS {
            y = self.trunk_block(y, m, (format!("dense/{pre}_norm_{i}/scale"), format!("dense/{pre}_dw_{i}"), format!("dense/{pre}_pw_{i}/kernel")), Some(valid));
        }
        let a = self.norm_silu(y, m.c, &format!("dense/{pre}_final_norm/scale"));
        self.proj(a, &format!("dense/{out}/kernel"), None, m.h * m.w, m.c, MASK_RANK)
    }

    /// The dense head: pyramid from the encoder memory and two stem skips, then the ten-channel detection map.
    /// The gradient the head sends back into the encoder is scaled by `head_grad` (see [`Tape::grad_scale`]).
    pub fn dense(&mut self, enc: &EncT, h: usize, w: usize) -> DenseT {
        let enc = enc.clone();
        let outs = self.block(move |net| {
            let d = net.dense_head(&enc, h, w);
            vec![d.head_in, d.head]
        });
        DenseT { head_in: outs[0], head: outs[1] }
    }

    fn dense_head(&mut self, enc: &EncT, h: usize, w: usize) -> DenseT {
        let c = self.cfg.clone();
        let (d, grid) = (c.dense_dim, c.grid());
        let side = c.canvas / c.stride;
        let (_, fine_level, mid_level) = c.dense_levels();
        let hg = f64::from(c.head_grad);
        let memory = self.tape.grad_scale(enc.memory, hg);
        let coarse = self.proj(memory, "dense/coarse/kernel", None, grid * grid, c.dim, d);
        let (d32, m32) = self.tape.upsample2(coarse, Map { h: grid, w: grid, c: d });
        let (mid_map, mid_x) = enc.stages[mid_level];
        let mid_x = self.tape.grad_scale(mid_x, hg);
        let skip = self.proj(mid_x, "dense/mid/kernel", None, mid_map.h * mid_map.w, mid_map.c, d);
        let d32 = self.tape.add(d32, skip);
        let d32 = self.tape.mask_rows(d32, d, &valid_cells(h, w, m32.h));
        let n = self.norm_silu(d32, d, "dense/neck_norm/scale");
        let neck_dw = self.p("dense/neck_dw");
        let (nd, _) = self.tape.depthwise(n, m32, neck_dw, 3, 3, 1, 1);
        let neck = self.proj(nd, "dense/neck_pw/kernel", None, m32.h * m32.w, d, d);
        let neck = self.tape.mask_rows(neck, d, &valid_cells(h, w, m32.h));
        let d32 = self.tape.add(d32, neck);

        let (d64, m64) = self.tape.upsample2(d32, m32);
        let (fine_map, fine_x) = enc.stages[fine_level];
        let fine_x = self.tape.grad_scale(fine_x, hg);
        let skip = self.proj(fine_x, "dense/fine/kernel", None, fine_map.h * fine_map.w, fine_map.c, d);
        let mut d64 = self.tape.add(d64, skip);
        for i in 0..DENSE_BLOCKS {
            let a = self.norm_silu(d64, d, &format!("dense/block_norm_{i}/scale"));
            let dw = self.p(&format!("dense/dw_{i}"));
            let (a, _) = self.tape.depthwise(a, m64, dw, 3, 3, 1, 1);
            let rr = self.proj(a, &format!("dense/pw_{i}/kernel"), None, m64.h * m64.w, d, d);
            d64 = self.tape.add(d64, rr);
        }
        let head_in = self.norm_silu(d64, d, "dense/final_norm/scale");
        let head_in = self.tape.mask_rows(head_in, d, &valid_cells(h, w, m64.h));
        let head = self.proj(head_in, "dense/out/kernel", None, side * side, d, DENSE_OUT_CHANNELS);
        let bias = self.p("dense/bias");
        let head = self.tape.bias_add(head, bias);
        DenseT { head_in, head }
    }

    /// Mask prototypes `[2 side, 2 side, 32]` and the coefficient map `[side, side, 32]`; each branch is one
    /// recompute block.
    pub fn mask_maps(&mut self, head_in: Id, enc: &EncT, h: usize, w: usize) -> (Id, Id) {
        let c = self.cfg.clone();
        let d = c.dense_dim;
        let side = c.canvas / c.stride;
        let (edge_level, _, _) = c.dense_levels();
        let (edge_map, edge_x) = enc.stages[edge_level];
        let head_grad = f64::from(c.head_grad);
        let protos = self.block1(move |net| {
            let edge_x = net.tape.grad_scale(edge_x, head_grad);
            let (u, um) = net.tape.upsample2(head_in, Map { h: side, w: side, c: d });
            let skip = net.proj(edge_x, "dense/edge/kernel", None, edge_map.h * edge_map.w, edge_map.c, d);
            let u = net.tape.add(u, skip);
            let v_proto = valid_cells(h, w, um.h);
            let u = net.tape.mask_rows(u, d, &v_proto);
            net.mask_branch(u, um, &v_proto, "proto", "protos")
        });
        let coefs = self.block1(move |net| {
            let v_coef = valid_cells(h, w, side);
            let c_in = net.tape.mask_rows(head_in, d, &v_coef);
            net.mask_branch(c_in, Map { h: side, w: side, c: d }, &v_coef, "coef", "coef")
        });
        (protos, coefs)
    }

    /// The reader's convolutional stages on one window: the per-frame feature rows `[frames, reader_dim]`.
    fn reader_stages(&mut self, window: Id) -> Id {
        let c = self.cfg.clone();
        let d = c.reader_dim;
        let widths = [READER_STAGES[0], READER_STAGES[1], READER_STAGES[2], d];
        let w0 = self.p("reader/w0");
        let (mut x, mut m) = self.tape.conv_dense(window, Map { h: crate::vision::reader::WINDOW_H, w: crate::vision::reader::WINDOW_W, c: 3 }, w0, 3, 3, widths[0], 2, 2);
        x = self.norm_silu(x, m.c, "reader/stage_norm_0/scale");
        for (i, &(sy, sx)) in [(2usize, 2usize), (2, 1), (2, 1)].iter().enumerate() {
            let dw = self.p(&format!("reader/dw_{i}"));
            let (dd, dm) = self.tape.depthwise(x, m, dw, 3, 3, sy, sx);
            x = self.proj(dd, &format!("reader/pw_{i}/kernel"), None, dm.h * dm.w, widths[i], widths[i + 1]);
            m = Map { h: dm.h, w: dm.w, c: widths[i + 1] };
            x = self.norm_silu(x, m.c, &format!("reader/stage_norm_{}/scale", i + 1));
        }
        // Mean over the remaining rows leaves one feature vector per frame.
        let inv = r::<R>(1.0 / m.h as f64);
        let segs: Vec<Seg<R>> = (0..m.h).map(|row| Seg { dst: 0, src: row * m.w * m.c, len: m.w * m.c, w: inv }).collect();
        let seq = self.tape.plan(x, m.w * m.c, segs);
        assert_eq!(m.w, FRAMES, "the reader makes {FRAMES} frames per window");
        seq
    }

    /// Mixer block `i` of the reader on the frame rows `seq`.
    fn reader_block(&mut self, seq: Id, i: usize) -> Id {
        let d = self.cfg.reader_dim;
        let frames = FRAMES;
        let norm = self.p(&format!("reader/norm_{i}/scale"));
        let n = self.tape.zc_norm(seq, d, norm);
        let mix = self.p(&format!("reader/mix_{i}"));
        let (hh, _) = self.tape.depthwise(n, Map { h: 1, w: frames, c: d }, mix, 1, READER_MIX_KERNEL, 1, 1);
        let uv = self.proj(hh, &format!("reader/up_{i}/kernel"), None, frames, d, 2 * d);
        let left: Vec<Seg<R>> = (0..frames).map(|f| Seg::copy(f * d, f * 2 * d, d)).collect();
        let right: Vec<Seg<R>> = (0..frames).map(|f| Seg::copy(f * d, f * 2 * d + d, d)).collect();
        let u = self.tape.plan(uv, frames * d, left);
        let g = self.tape.plan(uv, frames * d, right);
        let g = self.tape.silu(g);
        let gated = self.tape.mul(u, g);
        let down = self.proj(gated, &format!("reader/down_{i}/kernel"), None, frames, d, d);
        self.tape.add(seq, down)
    }

    /// The reader on one window of normalised pixels (`[48, 320, 3]`, a leaf): logits `[80, classes]`. The
    /// convolutional stages and each mixer block are recompute blocks.
    pub fn reader(&mut self, window: Id) -> Id {
        let c = self.cfg.clone();
        let d = c.reader_dim;
        let mut seq = self.block1(move |net| net.reader_stages(window));
        for i in 0..c.reader_blocks {
            let input = seq;
            seq = self.block1(move |net| net.reader_block(input, i));
        }
        let final_norm = self.p("reader/final_norm/scale");
        let n = self.tape.zc_norm(seq, d, final_norm);
        let logits = self.proj(n, "reader/out/kernel", None, FRAMES, d, crate::config::READER_CLASSES);
        let bias = self.p("reader/out/bias");
        self.tape.bias_add(logits, bias)
    }

    /// Decoder memory: the valid cells' encoder features plus their gated position code, then one row per slot.
    pub fn decoder_memory(&mut self, enc: &EncT, head_in: Id, valid: &[bool], slots: &[Slot]) -> Id {
        let c = self.cfg.clone();
        let (d, g) = (c.dim, c.grid());
        let side = c.canvas / c.stride;
        let head_dim = c.dense_dim;
        let live: Vec<usize> = (0..g * g).filter(|&i| valid[i]).collect();
        let half = d / 2;
        let pos: Vec<R> = (0..g * g)
            .flat_map(|cell| [sinusoid((cell / g) as f64, half), sinusoid((cell % g) as f64, half)].concat())
            .map(|v| r::<R>(f64::from(v)))
            .collect();
        let pos = self.constant(pos);
        let pe_gate = self.p("pe_gate");
        let gate = self.tape.sigmoid(pe_gate);
        let pos = self.tape.scalar_mul(pos, gate);
        let rows_e = self.tape.gather_rows(enc.memory, d, &live);
        let rows_p = self.tape.gather_rows(pos, d, &live);
        let cell_rows = self.tape.add(rows_e, rows_p);
        if slots.is_empty() {
            return cell_rows;
        }

        let n = slots.len();
        let mut centre = Vec::with_capacity(n);
        let mut codes: Vec<R> = Vec::with_capacity(n * 64);
        let mut pool_segs: Vec<Seg<R>> = Vec::new();
        let cell_px = (c.canvas / g) as f32;
        for (si, s) in slots.iter().enumerate() {
            let b = round_box(s.bbox);
            let cell = |lo: f32, hi: f32| ((side as f32 * (lo + hi) / 2.0) as usize).min(side - 1);
            centre.push(cell(b[1], b[3]) * side + cell(b[0], b[2]));
            codes.extend(box_code(b).into_iter().map(|v| r::<R>(f64::from(v))));
            let picked = pooled_cells(g, cell_px, valid, b);
            let wgt = r::<R>(1.0 / picked.len() as f64);
            pool_segs.extend(picked.iter().map(|&cell| Seg { dst: si * d, src: cell * d, len: d, w: wgt }));
        }
        // proj_in = [head feature | box code] per slot, so one kernel (and one adapter) covers both parts.
        let width = head_dim + 64;
        let gathered = self.tape.gather_rows(head_in, head_dim, &centre);
        let codes = self.constant(codes);
        let head_part = self.tape.plan(gathered, n * width, (0..n).map(|si| Seg::copy(si * width, si * head_dim, head_dim)).collect());
        let code_part = self.tape.plan(codes, n * width, (0..n).map(|si| Seg::copy(si * width + head_dim, si * 64, 64)).collect());
        let proj_in = self.tape.add(head_part, code_part);
        let u = self.proj(proj_in, "slot_proj/kernel", None, n, width, d);
        let pooled = self.tape.plan(enc.memory, n * d, pool_segs);
        let pool_proj = self.proj(pooled, "slot_pool/kernel", None, n, d, d);
        let u = self.tape.add(u, pool_proj);
        let slot_norm = self.p("slot_norm/scale");
        let m = self.tape.zc_norm(u, d, slot_norm);
        let embedding = self.p("embedding/embedding");
        let root = (d as f64).sqrt();
        let emb_segs: Vec<Seg<R>> = slots.iter().enumerate().map(|(si, s)| Seg { dst: si * d, src: s.token * d, len: d, w: r::<R>(root) }).collect();
        let emb = self.tape.plan(embedding, n * d, emb_segs);
        let m = self.tape.add(m, emb);
        self.tape.concat(cell_rows, m)
    }
}
