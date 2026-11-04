// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gradcheck for ONE Wan block ([`wan::grad`]). The gate the
//! porting playbook sets for a block-level backward is **< 1e-4**.
//!
//! Central differences in f64 along a random ±1 direction per tensor. The FD
//! side shares no code with the analytic backward it checks: it only ever calls
//! `block_forward`, so a wrong backward cannot make itself right.
//!
//! **Coverage is all 27 block tensors plus all three inputs.** `dx` and `dctx`
//! are not decoration: the model backward chains `dx` through the stack and sums
//! `dctx` across every block into `text_embedding`, so an unchecked input
//! adjoint would be a hole exactly where the whole-model gradient is built.
//! `e0` matters most of all - its adjoint is the modulation fold, which is both
//! `d(modulation)` and the block's contribution to the timestep path.

use wan::grad::{block_backward, block_forward, BlockGrads, BlockW, Dims, Lin};

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

/// Every trainable tensor of the block, in a fixed order.
fn params(w: &mut BlockW<f64>) -> Vec<(String, &mut Vec<f64>)> {
    vec![
        ("modulation".into(), &mut w.modulation),
        ("self_attn.q.weight".into(), &mut w.sq.w),
        ("self_attn.q.bias".into(), &mut w.sq.b),
        ("self_attn.k.weight".into(), &mut w.sk.w),
        ("self_attn.k.bias".into(), &mut w.sk.b),
        ("self_attn.v.weight".into(), &mut w.sv.w),
        ("self_attn.v.bias".into(), &mut w.sv.b),
        ("self_attn.o.weight".into(), &mut w.so.w),
        ("self_attn.o.bias".into(), &mut w.so.b),
        ("self_attn.norm_q.weight".into(), &mut w.snq),
        ("self_attn.norm_k.weight".into(), &mut w.snk),
        ("cross_attn.q.weight".into(), &mut w.cq.w),
        ("cross_attn.q.bias".into(), &mut w.cq.b),
        ("cross_attn.k.weight".into(), &mut w.ck.w),
        ("cross_attn.k.bias".into(), &mut w.ck.b),
        ("cross_attn.v.weight".into(), &mut w.cv.w),
        ("cross_attn.v.bias".into(), &mut w.cv.b),
        ("cross_attn.o.weight".into(), &mut w.co.w),
        ("cross_attn.o.bias".into(), &mut w.co.b),
        ("cross_attn.norm_q.weight".into(), &mut w.cnq),
        ("cross_attn.norm_k.weight".into(), &mut w.cnk),
        ("norm3.weight".into(), &mut w.norm3_w),
        ("norm3.bias".into(), &mut w.norm3_b),
        ("ffn.0.weight".into(), &mut w.ff1.w),
        ("ffn.0.bias".into(), &mut w.ff1.b),
        ("ffn.2.weight".into(), &mut w.ff2.w),
        ("ffn.2.bias".into(), &mut w.ff2.b),
    ]
}

/// Grad tensors in the SAME order as [`params`].
fn grad_of(g: &BlockGrads<f64>) -> Vec<(String, &Vec<f64>)> {
    vec![
        ("modulation".into(), &g.modulation),
        ("self_attn.q.weight".into(), &g.sq.w),
        ("self_attn.q.bias".into(), &g.sq.b),
        ("self_attn.k.weight".into(), &g.sk.w),
        ("self_attn.k.bias".into(), &g.sk.b),
        ("self_attn.v.weight".into(), &g.sv.w),
        ("self_attn.v.bias".into(), &g.sv.b),
        ("self_attn.o.weight".into(), &g.so.w),
        ("self_attn.o.bias".into(), &g.so.b),
        ("self_attn.norm_q.weight".into(), &g.snq),
        ("self_attn.norm_k.weight".into(), &g.snk),
        ("cross_attn.q.weight".into(), &g.cq.w),
        ("cross_attn.q.bias".into(), &g.cq.b),
        ("cross_attn.k.weight".into(), &g.ck.w),
        ("cross_attn.k.bias".into(), &g.ck.b),
        ("cross_attn.v.weight".into(), &g.cv.w),
        ("cross_attn.v.bias".into(), &g.cv.b),
        ("cross_attn.o.weight".into(), &g.co.w),
        ("cross_attn.o.bias".into(), &g.co.b),
        ("cross_attn.norm_q.weight".into(), &g.cnq),
        ("cross_attn.norm_k.weight".into(), &g.cnk),
        ("norm3.weight".into(), &g.norm3_w),
        ("norm3.bias".into(), &g.norm3_b),
        ("ffn.0.weight".into(), &g.ff1.w),
        ("ffn.0.bias".into(), &g.ff1.b),
        ("ffn.2.weight".into(), &g.ff2.w),
        ("ffn.2.bias".into(), &g.ff2.b),
    ]
}

/// Deliberately non-coincidental: 7 latent tokens against 5 text rows, dim 12
/// against ffn 9, 3 heads of 4. Equal dims are what hides a transposition.
fn dims() -> Dims {
    Dims { t: 7, te: 5, dim: 12, nh: 3, ffn: 9, eps: 1e-6 }
}

struct Fixture {
    d: Dims,
    w: BlockW<f64>,
    x: Vec<f64>,
    e0: Vec<f64>,
    ctx: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    target: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    let d = dims();
    let (dim, ffn) = (d.dim, d.ffn);
    let mut r = rng(seed);
    let w = BlockW {
        modulation: vec_of(6 * dim, &mut r, 0.3),
        sq: lin_of(dim, dim, &mut r),
        sk: lin_of(dim, dim, &mut r),
        sv: lin_of(dim, dim, &mut r),
        so: lin_of(dim, dim, &mut r),
        snq: gain_of(dim, &mut r),
        snk: gain_of(dim, &mut r),
        cq: lin_of(dim, dim, &mut r),
        ck: lin_of(dim, dim, &mut r),
        cv: lin_of(dim, dim, &mut r),
        co: lin_of(dim, dim, &mut r),
        cnq: gain_of(dim, &mut r),
        cnk: gain_of(dim, &mut r),
        norm3_w: gain_of(dim, &mut r),
        norm3_b: vec_of(dim, &mut r, 0.1),
        ff1: lin_of(ffn, dim, &mut r),
        ff2: lin_of(dim, ffn, &mut r),
    };
    let half = d.hd() / 2;
    Fixture {
        x: vec_of(d.t * dim, &mut r, 1.0),
        e0: vec_of(6 * dim, &mut r, 0.3),
        ctx: vec_of(d.te * dim, &mut r, 1.0),
        cos: (0..d.t * half).map(|i| (i as f64 * 0.37).cos()).collect(),
        sin: (0..d.t * half).map(|i| (i as f64 * 0.37).sin()).collect(),
        target: vec_of(d.t * dim, &mut r, 1.0),
        d,
        w,
    }
}

/// MSE of the block output against a fixed target, and its `dout`.
fn mse(out: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
    let n = out.len() as f64;
    let l = out.iter().zip(target).map(|(a, b)| (a - b) * (a - b) / n).sum();
    let dout = out.iter().zip(target).map(|(a, b)| 2.0 * (a - b) / n).collect();
    (l, dout)
}

/// One entry per FD comparison, so a failure names the tensor.
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
    let mut w0 = f.w.clone();
    let loss_of = |w: &BlockW<f64>, x: &[f64], e0: &[f64], ctx: &[f64]| -> f64 {
        let (out, _) = block_forward(f.d, w, x, e0, ctx, &f.cos, &f.sin);
        mse(&out, &f.target).0
    };

    let (out, cache) = block_forward(f.d, &w0, &f.x, &f.e0, &f.ctx, &f.cos, &f.sin);
    let (_l, dout) = mse(&out, &f.target);
    let g = block_backward(f.d, &w0, &cache, &dout);
    let analytic: Vec<(String, Vec<f64>)> = grad_of(&g).into_iter().map(|(n, v)| (n, v.clone())).collect();
    assert_eq!(analytic.len(), 27, "a Wan block has 27 trainable tensors");

    // eps 1e-5: with f64 central differences the truncation term is O(eps²)
    // ~1e-10 and the round-off term O(machine_eps/eps) ~1e-11 - three orders
    // below the 1e-4 gate, so a pass is the math being right, not the
    // tolerance being loose.
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
        let numeric = (loss_of(&wp, &f.x, &f.e0, &f.ctx) - loss_of(&wm, &f.x, &f.e0, &f.ctx)) / (2.0 * eps);
        rows.push(Row { name: name.clone(), analytic: an, numeric });
    }

    // The three input adjoints, by the same recipe.
    for (name, base, ga) in [("<input> x", &f.x, &g.dx), ("<input> e0", &f.e0, &g.modulation), ("<input> ctx", &f.ctx, &g.dctx)] {
        let v: Vec<f64> = (0..ga.len()).map(|_| if dir() < 0.0 { -1.0 } else { 1.0 }).collect();
        let an: f64 = ga.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();
        let bump = |sgn: f64| -> Vec<f64> { base.iter().zip(&v).map(|(&b, &vi)| b + sgn * eps * vi).collect() };
        let (lp, lm) = match name {
            "<input> x" => (loss_of(&w0, &bump(1.0), &f.e0, &f.ctx), loss_of(&w0, &bump(-1.0), &f.e0, &f.ctx)),
            "<input> e0" => (loss_of(&w0, &f.x, &bump(1.0), &f.ctx), loss_of(&w0, &f.x, &bump(-1.0), &f.ctx)),
            _ => (loss_of(&w0, &f.x, &f.e0, &bump(1.0)), loss_of(&w0, &f.x, &f.e0, &bump(-1.0))),
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
    // `params_mut`'s `e0` adjoint IS the modulation grad: perturbing `e0` and
    // perturbing `modulation` must move the loss identically, because the fold
    // only ever sees their sum. Checked directly, not assumed.
    let mut wm = w0.clone();
    for (p, e) in wm.modulation.iter_mut().zip(&f.e0) {
        *p += *e;
    }
    let zero = vec![0.0; f.e0.len()];
    let a = loss_of(&w0, &f.x, &f.e0, &f.ctx);
    let b = loss_of(&wm, &f.x, &zero, &f.ctx);
    assert_eq!(a, b, "modulation + e0 must be the fold's only operand");
    let _ = &mut w0;
}
