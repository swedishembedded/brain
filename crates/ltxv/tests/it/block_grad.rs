// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gradcheck for ONE video-only `LtxBlock` (`ltxv::grad`).
//! The gate the porting playbook sets for a block-level backward is
//! **< 1e-4**.
//!
//! Central differences in f64 along a random +/-1 direction per tensor. The
//! FD side shares no code with the analytic backward it checks: it only
//! ever calls `block_forward`, so a wrong backward cannot make itself right.
//!
//! **Coverage is all 24 block tensors plus all three inputs.** `dx` and
//! `dctx` are not decoration: the model backward chains `dx` through the
//! stack and would sum `dctx` across every block if a text encoder existed
//! upstream. `dadaln_shared` matters most of all - its adjoint is the
//! per-token modulation fold, simultaneously `d(scale_shift_table)`'s
//! row-sum source AND this block's contribution to the shared per-token
//! table every block reads (see `ltxv::grad`'s own module doc).

use ltxv::grad::{block_backward, block_forward, AttnW, BlockGrads, BlockW, Dims, Lin, LinNB};

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

/// Every trainable tensor of the block, in a fixed order.
fn params(w: &mut BlockW<f64>) -> Vec<(String, &mut Vec<f64>)> {
    vec![
        ("scale_shift_table".into(), &mut w.scale_shift_table),
        ("prompt_scale_shift_table".into(), &mut w.prompt_scale_shift_table),
        ("attn1.to_q.weight".into(), &mut w.attn1.q.w),
        ("attn1.to_q.bias".into(), &mut w.attn1.q.b),
        ("attn1.to_k.weight".into(), &mut w.attn1.k.w),
        ("attn1.to_k.bias".into(), &mut w.attn1.k.b),
        ("attn1.to_v.weight".into(), &mut w.attn1.v.w),
        ("attn1.to_v.bias".into(), &mut w.attn1.v.b),
        ("attn1.to_out.0.weight".into(), &mut w.attn1.o.w),
        ("attn1.to_out.0.bias".into(), &mut w.attn1.o.b),
        ("attn1.q_norm.weight".into(), &mut w.attn1.qn),
        ("attn1.k_norm.weight".into(), &mut w.attn1.kn),
        ("attn2.to_q.weight".into(), &mut w.attn2.q.w),
        ("attn2.to_q.bias".into(), &mut w.attn2.q.b),
        ("attn2.to_k.weight".into(), &mut w.attn2.k.w),
        ("attn2.to_k.bias".into(), &mut w.attn2.k.b),
        ("attn2.to_v.weight".into(), &mut w.attn2.v.w),
        ("attn2.to_v.bias".into(), &mut w.attn2.v.b),
        ("attn2.to_out.0.weight".into(), &mut w.attn2.o.w),
        ("attn2.to_out.0.bias".into(), &mut w.attn2.o.b),
        ("attn2.q_norm.weight".into(), &mut w.attn2.qn),
        ("attn2.k_norm.weight".into(), &mut w.attn2.kn),
        ("ff.net.0.proj.weight".into(), &mut w.ff1.w),
        ("ff.net.2.weight".into(), &mut w.ff2.w),
    ]
}

/// Grad tensors in the SAME order as [`params`].
fn grad_of(g: &BlockGrads<f64>) -> Vec<(String, &Vec<f64>)> {
    vec![
        ("scale_shift_table".into(), &g.scale_shift_table),
        ("prompt_scale_shift_table".into(), &g.prompt_scale_shift_table),
        ("attn1.to_q.weight".into(), &g.attn1.q.w),
        ("attn1.to_q.bias".into(), &g.attn1.q.b),
        ("attn1.to_k.weight".into(), &g.attn1.k.w),
        ("attn1.to_k.bias".into(), &g.attn1.k.b),
        ("attn1.to_v.weight".into(), &g.attn1.v.w),
        ("attn1.to_v.bias".into(), &g.attn1.v.b),
        ("attn1.to_out.0.weight".into(), &g.attn1.o.w),
        ("attn1.to_out.0.bias".into(), &g.attn1.o.b),
        ("attn1.q_norm.weight".into(), &g.attn1.qn),
        ("attn1.k_norm.weight".into(), &g.attn1.kn),
        ("attn2.to_q.weight".into(), &g.attn2.q.w),
        ("attn2.to_q.bias".into(), &g.attn2.q.b),
        ("attn2.to_k.weight".into(), &g.attn2.k.w),
        ("attn2.to_k.bias".into(), &g.attn2.k.b),
        ("attn2.to_v.weight".into(), &g.attn2.v.w),
        ("attn2.to_v.bias".into(), &g.attn2.v.b),
        ("attn2.to_out.0.weight".into(), &g.attn2.o.w),
        ("attn2.to_out.0.bias".into(), &g.attn2.o.b),
        ("attn2.q_norm.weight".into(), &g.attn2.qn),
        ("attn2.k_norm.weight".into(), &g.attn2.kn),
        ("ff.net.0.proj.weight".into(), &g.ff1.w),
        ("ff.net.2.weight".into(), &g.ff2.w),
    ]
}

/// Deliberately non-coincidental: 7 latent tokens against 5 text rows, dim
/// 12 (3 heads of 4).
fn dims() -> Dims {
    Dims { t: 7, te: 5, dim: 12, nh: 3, eps: 1e-6 }
}

struct Fixture {
    d: Dims,
    w: BlockW<f64>,
    x: Vec<f64>,
    adaln_shared: Vec<f64>,
    ctx: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    target: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    let d = dims();
    let dim = d.dim;
    let mut r = rng(seed);
    let w = BlockW {
        scale_shift_table: vec_of(9 * dim, &mut r, 0.3),
        prompt_scale_shift_table: vec_of(2 * dim, &mut r, 0.3),
        attn1: attn_of(dim, &mut r),
        attn2: attn_of(dim, &mut r),
        ff1: lin_nb_of(4 * dim, dim, &mut r),
        ff2: lin_nb_of(dim, 4 * dim, &mut r),
    };
    let half = d.hd() / 2;
    Fixture {
        x: vec_of(d.t * dim, &mut r, 1.0),
        adaln_shared: vec_of(d.t * 9 * dim, &mut r, 0.3),
        ctx: vec_of(d.te * dim, &mut r, 1.0),
        cos: (0..d.nh * d.t * half).map(|i| (i as f64 * 0.37).cos()).collect(),
        sin: (0..d.nh * d.t * half).map(|i| (i as f64 * 0.37).sin()).collect(),
        target: vec_of(d.t * dim, &mut r, 1.0),
        d,
        w,
    }
}

fn mse(out: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
    let n = out.len() as f64;
    let l = out.iter().zip(target).map(|(a, b)| (a - b) * (a - b) / n).sum();
    let dout = out.iter().zip(target).map(|(a, b)| 2.0 * (a - b) / n).collect();
    (l, dout)
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

#[test]
fn block_analytic_grads_match_finite_differences() {
    let f = fixture(0xB10C_6EAD);
    let w0 = f.w.clone();
    let loss_of = |w: &BlockW<f64>, x: &[f64], adaln_shared: &[f64], ctx: &[f64]| -> f64 {
        let (out, _) = block_forward(f.d, w, x, adaln_shared, ctx, &f.cos, &f.sin);
        mse(&out, &f.target).0
    };

    let (out, cache) = block_forward(f.d, &w0, &f.x, &f.adaln_shared, &f.ctx, &f.cos, &f.sin);
    let (_l, dout) = mse(&out, &f.target);
    let g = block_backward(f.d, &w0, &cache, &dout);
    let analytic: Vec<(String, Vec<f64>)> = grad_of(&g).into_iter().map(|(n, v)| (n, v.clone())).collect();
    assert_eq!(analytic.len(), 24, "an LTX video-only block has 24 trainable tensors");

    let eps = 1e-5;
    let mut dir = rng(0xD16_0001);
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
        let numeric = (loss_of(&wp, &f.x, &f.adaln_shared, &f.ctx) - loss_of(&wm, &f.x, &f.adaln_shared, &f.ctx)) / (2.0 * eps);
        rows.push(Row { name: name.clone(), analytic: an, numeric });
    }

    for (name, base, ga) in [("<input> x", &f.x, &g.dx), ("<input> adaln_shared", &f.adaln_shared, &g.dadaln_shared), ("<input> ctx", &f.ctx, &g.dctx)] {
        let v: Vec<f64> = (0..ga.len()).map(|_| if dir() < 0.0 { -1.0 } else { 1.0 }).collect();
        let an: f64 = ga.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();
        let bump = |sgn: f64| -> Vec<f64> { base.iter().zip(&v).map(|(&b, &vi)| b + sgn * eps * vi).collect() };
        let (lp, lm) = match name {
            "<input> x" => (loss_of(&w0, &bump(1.0), &f.adaln_shared, &f.ctx), loss_of(&w0, &bump(-1.0), &f.adaln_shared, &f.ctx)),
            "<input> adaln_shared" => (loss_of(&w0, &f.x, &bump(1.0), &f.ctx), loss_of(&w0, &f.x, &bump(-1.0), &f.ctx)),
            _ => (loss_of(&w0, &f.x, &f.adaln_shared, &bump(1.0)), loss_of(&w0, &f.x, &f.adaln_shared, &bump(-1.0))),
        };
        rows.push(Row { name: name.into(), analytic: an, numeric: (lp - lm) / (2.0 * eps) });
    }

    let mut worst = 0.0f64;
    for r in &rows {
        println!(
            "  {:<28} analytic={:+.8e} numeric={:+.8e} abs={:.2e} rel={:.2e}",
            r.name,
            r.analytic,
            r.numeric,
            r.abs_err(),
            r.rel_err()
        );
        worst = worst.max(r.abs_err().min(r.rel_err()));
    }
    println!("block gradcheck: worst error {worst:.3e} over {} tensors", rows.len());
    let fails: Vec<&Row> = rows.iter().filter(|r| r.abs_err() > 1e-4 && r.rel_err() > 1e-4).collect();
    assert!(
        fails.is_empty(),
        "block FD gate (1e-4) failed for {:?}",
        fails.iter().map(|r| (&r.name, r.abs_err(), r.rel_err())).collect::<Vec<_>>()
    );

    // `scale_shift_table`'s own grad is the ROW-SUM of the site gradient
    // (this block's own trained parameter); `adaln_shared`'s grad is the
    // UNREDUCED site gradient - so bumping `scale_shift_table` uniformly at
    // every row must move the loss identically to bumping every row of
    // `adaln_shared` by the same amount (the fold's operand is their sum,
    // broadcast). Checked directly, not assumed.
    let (t, dim) = (f.d.t, f.d.dim);
    let bump_table: Vec<f64> = vec_of(9 * dim, &mut rng(0xFEED), 0.01);
    let mut wm = w0.clone();
    for (p, e) in wm.scale_shift_table.iter_mut().zip(&bump_table) {
        *p += *e;
    }
    let mut bumped_shared = f.adaln_shared.clone();
    for r in 0..t {
        for i in 0..9 * dim {
            bumped_shared[r * 9 * dim + i] += bump_table[i];
        }
    }
    let a = loss_of(&wm, &f.x, &f.adaln_shared, &f.ctx);
    let b = loss_of(&w0, &f.x, &bumped_shared, &f.ctx);
    assert!((a - b).abs() < 1e-9, "scale_shift_table + adaln_shared must be the fold's only operand: {a} vs {b}");
}
