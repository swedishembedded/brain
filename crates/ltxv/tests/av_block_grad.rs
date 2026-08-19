// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gradcheck for ONE `LtxAvBlock` (`ltxv::av_grad`). The
//! gate the porting playbook sets for a block-level backward is **< 1e-4**,
//! the same bar `crates/ltxv/tests/block_grad.rs` uses for the video-only
//! block.
//!
//! Central differences in f64 along a random +/-1 direction per tensor. The
//! FD side shares no code with the analytic backward it checks: it only
//! ever calls `av_block_forward`, so a wrong backward cannot make itself
//! right.
//!
//! **Coverage is every AV block weight tensor plus every external input**
//! (`vx`/`ax`, both streams' `adaln_shared`/`ctx`, and the four model-shared
//! AV conditioning tables `av_video_ss`/`av_audio_ss`/`av_a2v_gate`/
//! `av_v2a_gate`) - the same "no decoration" discipline `block_grad.rs`'s
//! own doc explains, extended to the AV-specific tables this block reads
//! twice each (`ltxv::av_grad`'s module doc, point 4).
//!
//! Dims are deliberately non-coincidental throughout (lesson #4): video and
//! audio token/context counts, dims and head counts all differ from each
//! other and from the audio stream's own head geometry (which the AV
//! cross-attention runs at unconditionally).

use ltxv::av_grad::{av_block_backward, av_block_forward, AvBlockGrads, AvBlockW, AvCrossW, AvDims, CrossAttnW};
use ltxv::grad::{AttnW, Dims, Lin, LinNB};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

fn vec_of(n: usize, r: &mut impl FnMut() -> f64, scale: f64) -> Vec<f64> {
    (0..n).map(|_| r() * scale).collect()
}

fn gain_of(n: usize, r: &mut impl FnMut() -> f64) -> Vec<f64> {
    (0..n).map(|_| 1.0 + r() * 0.1).collect()
}

fn lin_of(out: usize, inn: usize, r: &mut impl FnMut() -> f64) -> Lin<f64> {
    Lin { w: vec_of(out * inn, r, 0.25), b: vec_of(out, r, 0.1) }
}

fn lin_nb_of(out: usize, inn: usize, r: &mut impl FnMut() -> f64) -> LinNB<f64> {
    LinNB { w: vec_of(out * inn, r, 0.25) }
}

fn attn_of(dim: usize, r: &mut impl FnMut() -> f64) -> AttnW<f64> {
    AttnW { q: lin_of(dim, dim, r), k: lin_of(dim, dim, r), v: lin_of(dim, dim, r), o: lin_of(dim, dim, r), qn: gain_of(dim, r), kn: gain_of(dim, r) }
}

fn cross_attn_of(q_dim: usize, kv_dim: usize, inner: usize, r: &mut impl FnMut() -> f64) -> CrossAttnW<f64> {
    CrossAttnW { q: lin_of(inner, q_dim, r), k: lin_of(inner, kv_dim, r), v: lin_of(inner, kv_dim, r), o: lin_of(q_dim, inner, r), qn: gain_of(inner, r), kn: gain_of(inner, r) }
}

/// Video: 2 heads x 4 (dim 8), 5 tokens, 4 text rows. Audio: 3 heads x 2
/// (dim 6), 3 tokens, 2 text rows. Every axis differs from every other.
fn dims() -> AvDims {
    AvDims { v: Dims { t: 5, te: 4, dim: 8, nh: 2, eps: 1e-6 }, a: Dims { t: 3, te: 2, dim: 6, nh: 3, eps: 1e-6 } }
}

struct Fixture {
    d: AvDims,
    w: AvBlockW<f64>,
    vx: Vec<f64>,
    ax: Vec<f64>,
    v_adaln_shared: Vec<f64>,
    a_adaln_shared: Vec<f64>,
    v_ctx: Vec<f64>,
    a_ctx: Vec<f64>,
    v_cos: Vec<f64>,
    v_sin: Vec<f64>,
    a_cos: Vec<f64>,
    a_sin: Vec<f64>,
    v_cross_cos: Vec<f64>,
    v_cross_sin: Vec<f64>,
    a_cross_cos: Vec<f64>,
    a_cross_sin: Vec<f64>,
    av_video_ss: Vec<f64>,
    av_audio_ss: Vec<f64>,
    av_a2v_gate: Vec<f64>,
    av_v2a_gate: Vec<f64>,
    v_target: Vec<f64>,
    a_target: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    let d = dims();
    let (vdim, adim) = (d.v.dim, d.a.dim);
    let (tv, ta) = (d.v.t, d.a.t);
    let (v_te, a_te) = (d.v.te, d.a.te);
    let (aheads, ahd) = (d.a.nh, d.a.hd());
    let mut r = rng(seed);

    let w = AvBlockW {
        v_scale_shift_table: vec_of(9 * vdim, &mut r, 0.3),
        v_prompt_scale_shift_table: vec_of(2 * vdim, &mut r, 0.3),
        v_attn1: attn_of(vdim, &mut r),
        v_attn2: attn_of(vdim, &mut r),
        v_ff1: lin_nb_of(4 * vdim, vdim, &mut r),
        v_ff2: lin_nb_of(vdim, 4 * vdim, &mut r),
        a_scale_shift_table: vec_of(9 * adim, &mut r, 0.3),
        a_prompt_scale_shift_table: vec_of(2 * adim, &mut r, 0.3),
        a_attn1: attn_of(adim, &mut r),
        a_attn2: attn_of(adim, &mut r),
        a_ff1: lin_of(4 * adim, adim, &mut r),
        a_ff2: lin_of(adim, 4 * adim, &mut r),
        av: AvCrossW {
            a2v: cross_attn_of(vdim, adim, adim, &mut r),
            v2a: cross_attn_of(adim, vdim, adim, &mut r),
            table_video: vec_of(5 * vdim, &mut r, 0.3),
            table_audio: vec_of(5 * adim, &mut r, 0.3),
        },
    };

    let half_v = d.v.hd() / 2;
    let half_a = d.a.hd() / 2;
    let cross_half = ahd / 2;
    Fixture {
        vx: vec_of(tv * vdim, &mut r, 1.0),
        ax: vec_of(ta * adim, &mut r, 1.0),
        v_adaln_shared: vec_of(tv * 9 * vdim, &mut r, 0.3),
        a_adaln_shared: vec_of(ta * 9 * adim, &mut r, 0.3),
        v_ctx: vec_of(v_te * vdim, &mut r, 1.0),
        a_ctx: vec_of(a_te * adim, &mut r, 1.0),
        v_cos: (0..d.v.nh * tv * half_v).map(|i| (i as f64 * 0.37).cos()).collect(),
        v_sin: (0..d.v.nh * tv * half_v).map(|i| (i as f64 * 0.37).sin()).collect(),
        a_cos: (0..d.a.nh * ta * half_a).map(|i| (i as f64 * 0.53).cos()).collect(),
        a_sin: (0..d.a.nh * ta * half_a).map(|i| (i as f64 * 0.53).sin()).collect(),
        v_cross_cos: (0..aheads * tv * cross_half).map(|i| (i as f64 * 0.61).cos()).collect(),
        v_cross_sin: (0..aheads * tv * cross_half).map(|i| (i as f64 * 0.61).sin()).collect(),
        a_cross_cos: (0..aheads * ta * cross_half).map(|i| (i as f64 * 0.71).cos()).collect(),
        a_cross_sin: (0..aheads * ta * cross_half).map(|i| (i as f64 * 0.71).sin()).collect(),
        av_video_ss: vec_of(tv * 4 * vdim, &mut r, 0.3),
        av_audio_ss: vec_of(ta * 4 * adim, &mut r, 0.3),
        av_a2v_gate: vec_of(vdim, &mut r, 0.3),
        av_v2a_gate: vec_of(adim, &mut r, 0.3),
        v_target: vec_of(tv * vdim, &mut r, 1.0),
        a_target: vec_of(ta * adim, &mut r, 1.0),
        d,
        w,
    }
}

/// Every trainable tensor of the block, in a fixed order.
fn params(w: &mut AvBlockW<f64>) -> Vec<(String, &mut Vec<f64>)> {
    vec![
        ("v_scale_shift_table".into(), &mut w.v_scale_shift_table),
        ("v_prompt_scale_shift_table".into(), &mut w.v_prompt_scale_shift_table),
        ("v_attn1.q.w".into(), &mut w.v_attn1.q.w),
        ("v_attn1.q.b".into(), &mut w.v_attn1.q.b),
        ("v_attn1.k.w".into(), &mut w.v_attn1.k.w),
        ("v_attn1.k.b".into(), &mut w.v_attn1.k.b),
        ("v_attn1.v.w".into(), &mut w.v_attn1.v.w),
        ("v_attn1.v.b".into(), &mut w.v_attn1.v.b),
        ("v_attn1.o.w".into(), &mut w.v_attn1.o.w),
        ("v_attn1.o.b".into(), &mut w.v_attn1.o.b),
        ("v_attn1.qn".into(), &mut w.v_attn1.qn),
        ("v_attn1.kn".into(), &mut w.v_attn1.kn),
        ("v_attn2.q.w".into(), &mut w.v_attn2.q.w),
        ("v_attn2.q.b".into(), &mut w.v_attn2.q.b),
        ("v_attn2.k.w".into(), &mut w.v_attn2.k.w),
        ("v_attn2.k.b".into(), &mut w.v_attn2.k.b),
        ("v_attn2.v.w".into(), &mut w.v_attn2.v.w),
        ("v_attn2.v.b".into(), &mut w.v_attn2.v.b),
        ("v_attn2.o.w".into(), &mut w.v_attn2.o.w),
        ("v_attn2.o.b".into(), &mut w.v_attn2.o.b),
        ("v_attn2.qn".into(), &mut w.v_attn2.qn),
        ("v_attn2.kn".into(), &mut w.v_attn2.kn),
        ("v_ff1.w".into(), &mut w.v_ff1.w),
        ("v_ff2.w".into(), &mut w.v_ff2.w),
        ("a_scale_shift_table".into(), &mut w.a_scale_shift_table),
        ("a_prompt_scale_shift_table".into(), &mut w.a_prompt_scale_shift_table),
        ("a_attn1.q.w".into(), &mut w.a_attn1.q.w),
        ("a_attn1.q.b".into(), &mut w.a_attn1.q.b),
        ("a_attn1.k.w".into(), &mut w.a_attn1.k.w),
        ("a_attn1.k.b".into(), &mut w.a_attn1.k.b),
        ("a_attn1.v.w".into(), &mut w.a_attn1.v.w),
        ("a_attn1.v.b".into(), &mut w.a_attn1.v.b),
        ("a_attn1.o.w".into(), &mut w.a_attn1.o.w),
        ("a_attn1.o.b".into(), &mut w.a_attn1.o.b),
        ("a_attn1.qn".into(), &mut w.a_attn1.qn),
        ("a_attn1.kn".into(), &mut w.a_attn1.kn),
        ("a_attn2.q.w".into(), &mut w.a_attn2.q.w),
        ("a_attn2.q.b".into(), &mut w.a_attn2.q.b),
        ("a_attn2.k.w".into(), &mut w.a_attn2.k.w),
        ("a_attn2.k.b".into(), &mut w.a_attn2.k.b),
        ("a_attn2.v.w".into(), &mut w.a_attn2.v.w),
        ("a_attn2.v.b".into(), &mut w.a_attn2.v.b),
        ("a_attn2.o.w".into(), &mut w.a_attn2.o.w),
        ("a_attn2.o.b".into(), &mut w.a_attn2.o.b),
        ("a_attn2.qn".into(), &mut w.a_attn2.qn),
        ("a_attn2.kn".into(), &mut w.a_attn2.kn),
        ("a_ff1.w".into(), &mut w.a_ff1.w),
        ("a_ff1.b".into(), &mut w.a_ff1.b),
        ("a_ff2.w".into(), &mut w.a_ff2.w),
        ("a_ff2.b".into(), &mut w.a_ff2.b),
        ("av.a2v.q.w".into(), &mut w.av.a2v.q.w),
        ("av.a2v.q.b".into(), &mut w.av.a2v.q.b),
        ("av.a2v.k.w".into(), &mut w.av.a2v.k.w),
        ("av.a2v.k.b".into(), &mut w.av.a2v.k.b),
        ("av.a2v.v.w".into(), &mut w.av.a2v.v.w),
        ("av.a2v.v.b".into(), &mut w.av.a2v.v.b),
        ("av.a2v.o.w".into(), &mut w.av.a2v.o.w),
        ("av.a2v.o.b".into(), &mut w.av.a2v.o.b),
        ("av.a2v.qn".into(), &mut w.av.a2v.qn),
        ("av.a2v.kn".into(), &mut w.av.a2v.kn),
        ("av.v2a.q.w".into(), &mut w.av.v2a.q.w),
        ("av.v2a.q.b".into(), &mut w.av.v2a.q.b),
        ("av.v2a.k.w".into(), &mut w.av.v2a.k.w),
        ("av.v2a.k.b".into(), &mut w.av.v2a.k.b),
        ("av.v2a.v.w".into(), &mut w.av.v2a.v.w),
        ("av.v2a.v.b".into(), &mut w.av.v2a.v.b),
        ("av.v2a.o.w".into(), &mut w.av.v2a.o.w),
        ("av.v2a.o.b".into(), &mut w.av.v2a.o.b),
        ("av.v2a.qn".into(), &mut w.av.v2a.qn),
        ("av.v2a.kn".into(), &mut w.av.v2a.kn),
        ("av.table_video".into(), &mut w.av.table_video),
        ("av.table_audio".into(), &mut w.av.table_audio),
    ]
}

/// Grad tensors in the SAME order as [`params`].
fn grad_of(g: &AvBlockGrads<f64>) -> Vec<(String, &Vec<f64>)> {
    vec![
        ("v_scale_shift_table".into(), &g.v_scale_shift_table),
        ("v_prompt_scale_shift_table".into(), &g.v_prompt_scale_shift_table),
        ("v_attn1.q.w".into(), &g.v_attn1.q.w),
        ("v_attn1.q.b".into(), &g.v_attn1.q.b),
        ("v_attn1.k.w".into(), &g.v_attn1.k.w),
        ("v_attn1.k.b".into(), &g.v_attn1.k.b),
        ("v_attn1.v.w".into(), &g.v_attn1.v.w),
        ("v_attn1.v.b".into(), &g.v_attn1.v.b),
        ("v_attn1.o.w".into(), &g.v_attn1.o.w),
        ("v_attn1.o.b".into(), &g.v_attn1.o.b),
        ("v_attn1.qn".into(), &g.v_attn1.qn),
        ("v_attn1.kn".into(), &g.v_attn1.kn),
        ("v_attn2.q.w".into(), &g.v_attn2.q.w),
        ("v_attn2.q.b".into(), &g.v_attn2.q.b),
        ("v_attn2.k.w".into(), &g.v_attn2.k.w),
        ("v_attn2.k.b".into(), &g.v_attn2.k.b),
        ("v_attn2.v.w".into(), &g.v_attn2.v.w),
        ("v_attn2.v.b".into(), &g.v_attn2.v.b),
        ("v_attn2.o.w".into(), &g.v_attn2.o.w),
        ("v_attn2.o.b".into(), &g.v_attn2.o.b),
        ("v_attn2.qn".into(), &g.v_attn2.qn),
        ("v_attn2.kn".into(), &g.v_attn2.kn),
        ("v_ff1.w".into(), &g.v_ff1.w),
        ("v_ff2.w".into(), &g.v_ff2.w),
        ("a_scale_shift_table".into(), &g.a_scale_shift_table),
        ("a_prompt_scale_shift_table".into(), &g.a_prompt_scale_shift_table),
        ("a_attn1.q.w".into(), &g.a_attn1.q.w),
        ("a_attn1.q.b".into(), &g.a_attn1.q.b),
        ("a_attn1.k.w".into(), &g.a_attn1.k.w),
        ("a_attn1.k.b".into(), &g.a_attn1.k.b),
        ("a_attn1.v.w".into(), &g.a_attn1.v.w),
        ("a_attn1.v.b".into(), &g.a_attn1.v.b),
        ("a_attn1.o.w".into(), &g.a_attn1.o.w),
        ("a_attn1.o.b".into(), &g.a_attn1.o.b),
        ("a_attn1.qn".into(), &g.a_attn1.qn),
        ("a_attn1.kn".into(), &g.a_attn1.kn),
        ("a_attn2.q.w".into(), &g.a_attn2.q.w),
        ("a_attn2.q.b".into(), &g.a_attn2.q.b),
        ("a_attn2.k.w".into(), &g.a_attn2.k.w),
        ("a_attn2.k.b".into(), &g.a_attn2.k.b),
        ("a_attn2.v.w".into(), &g.a_attn2.v.w),
        ("a_attn2.v.b".into(), &g.a_attn2.v.b),
        ("a_attn2.o.w".into(), &g.a_attn2.o.w),
        ("a_attn2.o.b".into(), &g.a_attn2.o.b),
        ("a_attn2.qn".into(), &g.a_attn2.qn),
        ("a_attn2.kn".into(), &g.a_attn2.kn),
        ("a_ff1.w".into(), &g.a_ff1.w),
        ("a_ff1.b".into(), &g.a_ff1.b),
        ("a_ff2.w".into(), &g.a_ff2.w),
        ("a_ff2.b".into(), &g.a_ff2.b),
        ("av.a2v.q.w".into(), &g.av.a2v.q.w),
        ("av.a2v.q.b".into(), &g.av.a2v.q.b),
        ("av.a2v.k.w".into(), &g.av.a2v.k.w),
        ("av.a2v.k.b".into(), &g.av.a2v.k.b),
        ("av.a2v.v.w".into(), &g.av.a2v.v.w),
        ("av.a2v.v.b".into(), &g.av.a2v.v.b),
        ("av.a2v.o.w".into(), &g.av.a2v.o.w),
        ("av.a2v.o.b".into(), &g.av.a2v.o.b),
        ("av.a2v.qn".into(), &g.av.a2v.qn),
        ("av.a2v.kn".into(), &g.av.a2v.kn),
        ("av.v2a.q.w".into(), &g.av.v2a.q.w),
        ("av.v2a.q.b".into(), &g.av.v2a.q.b),
        ("av.v2a.k.w".into(), &g.av.v2a.k.w),
        ("av.v2a.k.b".into(), &g.av.v2a.k.b),
        ("av.v2a.v.w".into(), &g.av.v2a.v.w),
        ("av.v2a.v.b".into(), &g.av.v2a.v.b),
        ("av.v2a.o.w".into(), &g.av.v2a.o.w),
        ("av.v2a.o.b".into(), &g.av.v2a.o.b),
        ("av.v2a.qn".into(), &g.av.v2a.qn),
        ("av.v2a.kn".into(), &g.av.v2a.kn),
        ("av.table_video".into(), &g.av.table_video),
        ("av.table_audio".into(), &g.av.table_audio),
    ]
}

fn mse(v_out: &[f64], v_target: &[f64], a_out: &[f64], a_target: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let n = (v_out.len() + a_out.len()) as f64;
    let l = v_out.iter().zip(v_target).chain(a_out.iter().zip(a_target)).map(|(a, b)| (a - b) * (a - b) / n).sum();
    let dv: Vec<f64> = v_out.iter().zip(v_target).map(|(a, b)| 2.0 * (a - b) / n).collect();
    let da: Vec<f64> = a_out.iter().zip(a_target).map(|(a, b)| 2.0 * (a - b) / n).collect();
    (l, dv, da)
}

struct Row {
    name: String,
    analytic: f64,
    numeric: f64,
}

impl Row {
    fn abs_err(&self) -> f64 {
        (self.analytic - self.numeric).abs()
    }
    fn rel_err(&self) -> f64 {
        self.abs_err() / self.analytic.abs().max(self.numeric.abs()).max(1e-6)
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_loss(f: &Fixture, w: &AvBlockW<f64>, vx: &[f64], ax: &[f64], v_adaln_shared: &[f64], a_adaln_shared: &[f64], v_ctx: &[f64], a_ctx: &[f64], av_video_ss: &[f64], av_audio_ss: &[f64], av_a2v_gate: &[f64], av_v2a_gate: &[f64]) -> f64 {
    let (vx_out, ax_out, _) = av_block_forward(
        f.d, w, vx, ax, v_adaln_shared, a_adaln_shared, v_ctx, a_ctx, &f.v_cos, &f.v_sin, &f.a_cos, &f.a_sin, &f.v_cross_cos, &f.v_cross_sin, &f.a_cross_cos, &f.a_cross_sin, av_video_ss,
        av_audio_ss, av_a2v_gate, av_v2a_gate,
    );
    mse(&vx_out, &f.v_target, &ax_out, &f.a_target).0
}

#[test]
fn av_block_analytic_grads_match_finite_differences() {
    let f = fixture(0xA1_5EAD);
    let w0 = f.w.clone();

    let (vx_out, ax_out, cache) = av_block_forward(
        f.d, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.v_cos, &f.v_sin, &f.a_cos, &f.a_sin, &f.v_cross_cos, &f.v_cross_sin, &f.a_cross_cos, &f.a_cross_sin,
        &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate,
    );
    let (_l, dvx_out, dax_out) = mse(&vx_out, &f.v_target, &ax_out, &f.a_target);
    let g = av_block_backward(f.d, &w0, &cache, &dvx_out, &dax_out);
    let analytic: Vec<(String, Vec<f64>)> = grad_of(&g).into_iter().map(|(n, v)| (n, v.clone())).collect();

    let eps = 1e-5;
    let mut dir = rng(0xD16_A001);
    let mut rows: Vec<Row> = Vec::new();
    for (pi, (name, ga)) in analytic.iter().enumerate() {
        let v: Vec<f64> = (0..ga.len()).map(|_| if dir() < 0.0 { -1.0 } else { 1.0 }).collect();
        let an: f64 = ga.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();
        let mut wp = w0.clone();
        for (p, &vi) in params(&mut wp)[pi].1.iter_mut().zip(&v) {
            *p += eps * vi;
        }
        let mut wm = w0.clone();
        for (p, &vi) in params(&mut wm)[pi].1.iter_mut().zip(&v) {
            *p -= eps * vi;
        }
        let lp = forward_loss(&f, &wp, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate);
        let lm = forward_loss(&f, &wm, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate);
        let numeric = (lp - lm) / (2.0 * eps);
        rows.push(Row { name: name.clone(), analytic: an, numeric });
    }

    // External inputs: vx, ax, both streams' adaln_shared/ctx, the four
    // model-shared AV conditioning tables.
    let inputs: Vec<(&str, &Vec<f64>, &Vec<f64>)> = vec![
        ("<input> vx", &f.vx, &g.dvx),
        ("<input> ax", &f.ax, &g.dax),
        ("<input> v_adaln_shared", &f.v_adaln_shared, &g.dv_adaln_shared),
        ("<input> a_adaln_shared", &f.a_adaln_shared, &g.da_adaln_shared),
        ("<input> v_ctx", &f.v_ctx, &g.dv_ctx),
        ("<input> a_ctx", &f.a_ctx, &g.da_ctx),
        ("<input> av_video_ss", &f.av_video_ss, &g.dav_video_ss),
        ("<input> av_audio_ss", &f.av_audio_ss, &g.dav_audio_ss),
        ("<input> av_a2v_gate", &f.av_a2v_gate, &g.dav_a2v_gate),
        ("<input> av_v2a_gate", &f.av_v2a_gate, &g.dav_v2a_gate),
    ];
    for (name, base, ga) in inputs {
        let v: Vec<f64> = (0..ga.len()).map(|_| if dir() < 0.0 { -1.0 } else { 1.0 }).collect();
        let an: f64 = ga.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();
        let bump = |sgn: f64| -> Vec<f64> { base.iter().zip(&v).map(|(&b, &vi)| b + sgn * eps * vi).collect() };
        let bp = bump(1.0);
        let bm = bump(-1.0);
        let (lp, lm) = match name {
            "<input> vx" => (
                forward_loss(&f, &w0, &bp, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &bm, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> ax" => (
                forward_loss(&f, &w0, &f.vx, &bp, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &bm, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> v_adaln_shared" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &bp, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &bm, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> a_adaln_shared" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &bp, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &bm, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> v_ctx" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &bp, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &bm, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> a_ctx" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &bp, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &bm, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> av_video_ss" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &bp, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &bm, &f.av_audio_ss, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> av_audio_ss" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &bp, &f.av_a2v_gate, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &bm, &f.av_a2v_gate, &f.av_v2a_gate),
            ),
            "<input> av_a2v_gate" => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &bp, &f.av_v2a_gate),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &bm, &f.av_v2a_gate),
            ),
            _ => (
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &bp),
                forward_loss(&f, &w0, &f.vx, &f.ax, &f.v_adaln_shared, &f.a_adaln_shared, &f.v_ctx, &f.a_ctx, &f.av_video_ss, &f.av_audio_ss, &f.av_a2v_gate, &bm),
            ),
        };
        rows.push(Row { name: name.into(), analytic: an, numeric: (lp - lm) / (2.0 * eps) });
    }

    let mut worst = 0.0f64;
    for r in &rows {
        println!("  {:<28} analytic={:+.8e} numeric={:+.8e} abs={:.2e} rel={:.2e}", r.name, r.analytic, r.numeric, r.abs_err(), r.rel_err());
        worst = worst.max(r.abs_err().min(r.rel_err()));
    }
    println!("AV block gradcheck: worst error {worst:.3e} over {} tensors", rows.len());
    let fails: Vec<&Row> = rows.iter().filter(|r| r.abs_err() > 1e-4 && r.rel_err() > 1e-4).collect();
    assert!(fails.is_empty(), "AV block FD gate (1e-4) failed for {:?}", fails.iter().map(|r| (&r.name, r.abs_err(), r.rel_err())).collect::<Vec<_>>());
}
