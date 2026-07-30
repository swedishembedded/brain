// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gradcheck for the FLUX.2 block backwards
//! ([`flux2::grad`]): the double block (both streams + joint attention) and
//! the single block (parallel attn ‖ SwiGLU, column-split linear2).
//!
//! Perturbs every trainable tensor, every modulation-site vector
//! (shift/scale/gate per site), and the block input `x` element-by-element and
//! confirms the analytic gradient matches the central difference. The scalar
//! objective is `L = Σ out·seed`, so `dout = seed` exactly and the backward is
//! checked in isolation. Pure host f64 — no GPU, runs in `make test`.

use flux2::grad::{
    double_backward, double_forward, single_backward, single_forward, Dims, DoubleMods, DoubleW,
    Mod, SingleW, StreamW,
};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

fn vof(n: usize, r: &mut impl FnMut() -> f64, scale: f64) -> Vec<f64> {
    (0..n).map(|_| r() * scale).collect()
}

fn stream_w(d: &Dims, r: &mut impl FnMut() -> f64) -> StreamW<f64> {
    let (dim, hd, mlp) = (d.d, d.hd(), d.mlp);
    StreamW {
        wq: vof(dim * dim, r, 0.3),
        wk: vof(dim * dim, r, 0.3),
        wv: vof(dim * dim, r, 0.3),
        nq: vof(hd, r, 0.2).iter().map(|v| 1.0 + v).collect(),
        nk: vof(hd, r, 0.2).iter().map(|v| 1.0 + v).collect(),
        wo: vof(dim * dim, r, 0.3),
        w1: vof(mlp * dim, r, 0.3),
        w3: vof(mlp * dim, r, 0.3),
        w2: vof(dim * mlp, r, 0.3),
    }
}

fn mk_mod(dim: usize, r: &mut impl FnMut() -> f64) -> Mod<f64> {
    Mod { shift: vof(dim, r, 0.2), scale: vof(dim, r, 0.2), gate: vof(dim, r, 0.5) }
}

/// Central-difference check of `analytic` against `param` for a scalar loss
/// `f(param[i] ± h)`. Asserts worst rel err < 1e-4 over a sampled stride.
fn check(name: &str, analytic: &[f64], param: &mut [f64], h: f64, mut f: impl FnMut(&mut [f64]) -> f64) {
    let mut worst = 0f64;
    let mut worst_i = 0;
    let step = (param.len() / 64).max(1);
    for i in (0..param.len()).step_by(step) {
        let orig = param[i];
        param[i] = orig + h;
        let lp = f(param);
        param[i] = orig - h;
        let lm = f(param);
        param[i] = orig;
        let num = (lp - lm) / (2.0 * h);
        let a = analytic[i];
        let denom = a.abs().max(num.abs()).max(1e-3);
        let rel = (a - num).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    eprintln!("  {name:14} worst rel err = {worst:.2e} (elem {worst_i}/{})", param.len());
    assert!(worst < 1e-4, "{name} gradcheck failed: {worst:.3e} at elem {worst_i}");
}

fn tables(n: usize, half: usize) -> (Vec<f64>, Vec<f64>) {
    (
        (0..n * half).map(|i| (i as f64 * 0.3).cos()).collect(),
        (0..n * half).map(|i| (i as f64 * 0.3).sin()).collect(),
    )
}

#[test]
fn double_block_backward_matches_finite_difference() {
    // 3 txt + 4 img rows, dim 8, 2 heads (hd 4), mlp 10.
    let d = Dims { nt: 3, ni: 4, d: 8, nh: 2, mlp: 10 };
    let (n, dim) = (d.n(), d.d);
    let mut r = rng(0xF1u64 << 32 | 0xD17_C0DE);
    let w = DoubleW { img: stream_w(&d, &mut r), txt: stream_w(&d, &mut r) };
    let m = DoubleMods {
        img1: mk_mod(dim, &mut r),
        img2: mk_mod(dim, &mut r),
        txt1: mk_mod(dim, &mut r),
        txt2: mk_mod(dim, &mut r),
    };
    let x = vof(n * dim, &mut r, 1.0);
    let (cos, sin) = tables(n, d.hd() / 2);
    let seed = vof(n * dim, &mut r, 1.0);

    let loss = |w: &DoubleW<f64>, x: &[f64], m: &DoubleMods<f64>| -> f64 {
        let (out, _) = double_forward(d, w, x, m, &cos, &sin);
        out.iter().zip(&seed).map(|(&o, &s)| o * s).sum()
    };
    let (_out, cache) = double_forward(d, &w, &x, &m, &cos, &sin);
    let g = double_backward(d, &w, &m, &cache, &seed);

    let h = 1e-4;
    // stream weights (both streams)
    macro_rules! ckw {
        ($stream:ident, $field:ident) => {{
            let an = g.$stream.$field.clone();
            let mut p = w.$stream.$field.clone();
            check(concat!(stringify!($stream), ".", stringify!($field)), &an, &mut p, h, |p| {
                let mut w2 = w.clone();
                w2.$stream.$field.copy_from_slice(p);
                loss(&w2, &x, &m)
            });
        }};
    }
    ckw!(img, wq);
    ckw!(img, wk);
    ckw!(img, wv);
    ckw!(img, nq);
    ckw!(img, nk);
    ckw!(img, wo);
    ckw!(img, w1);
    ckw!(img, w3);
    ckw!(img, w2);
    ckw!(txt, wq);
    ckw!(txt, wk);
    ckw!(txt, wv);
    ckw!(txt, nq);
    ckw!(txt, nk);
    ckw!(txt, wo);
    ckw!(txt, w1);
    ckw!(txt, w3);
    ckw!(txt, w2);
    // modulation sites (shift/scale/gate per site)
    macro_rules! ckm {
        ($site:ident, $field:ident) => {{
            let an = g.$site.$field.clone();
            let mut p = m.$site.$field.clone();
            check(concat!(stringify!($site), ".", stringify!($field)), &an, &mut p, h, |p| {
                let mut m2 = m.clone();
                m2.$site.$field.copy_from_slice(p);
                loss(&w, &x, &m2)
            });
        }};
    }
    ckm!(img1, shift);
    ckm!(img1, scale);
    ckm!(img1, gate);
    ckm!(img2, shift);
    ckm!(img2, scale);
    ckm!(img2, gate);
    ckm!(txt1, shift);
    ckm!(txt1, scale);
    ckm!(txt1, gate);
    ckm!(txt2, shift);
    ckm!(txt2, scale);
    ckm!(txt2, gate);
    // input
    let mut xc = x.clone();
    check("dx", &g.dx, &mut xc, h, |p| loss(&w, p, &m));
    eprintln!("FLUX.2 double-block backward: gradcheck PASSED (all tensors < 1e-4 rel err)");
}

#[test]
fn single_block_backward_matches_finite_difference() {
    let d = Dims { nt: 3, ni: 4, d: 8, nh: 2, mlp: 10 };
    let (n, dim, hd, mlp) = (d.n(), d.d, d.hd(), d.mlp);
    let mut r = rng(0x51_0B10);
    let w = SingleW {
        wq: vof(dim * dim, &mut r, 0.3),
        wk: vof(dim * dim, &mut r, 0.3),
        wv: vof(dim * dim, &mut r, 0.3),
        nq: vof(hd, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        nk: vof(hd, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        w1: vof(mlp * dim, &mut r, 0.3),
        w3: vof(mlp * dim, &mut r, 0.3),
        wo_a: vof(dim * dim, &mut r, 0.3),
        wo_b: vof(dim * mlp, &mut r, 0.3),
    };
    let m = mk_mod(dim, &mut r);
    let x = vof(n * dim, &mut r, 1.0);
    let (cos, sin) = tables(n, hd / 2);
    let seed = vof(n * dim, &mut r, 1.0);

    let loss = |w: &SingleW<f64>, x: &[f64], m: &Mod<f64>| -> f64 {
        let (out, _) = single_forward(d, w, x, m, &cos, &sin);
        out.iter().zip(&seed).map(|(&o, &s)| o * s).sum()
    };
    let (_out, cache) = single_forward(d, &w, &x, &m, &cos, &sin);
    let g = single_backward(d, &w, &m, &cache, &seed);

    let h = 1e-4;
    macro_rules! ckw {
        ($field:ident) => {{
            let an = g.$field.clone();
            let mut p = w.$field.clone();
            check(stringify!($field), &an, &mut p, h, |p| {
                let mut w2 = w.clone();
                w2.$field.copy_from_slice(p);
                loss(&w2, &x, &m)
            });
        }};
    }
    ckw!(wq);
    ckw!(wk);
    ckw!(wv);
    ckw!(nq);
    ckw!(nk);
    ckw!(w1);
    ckw!(w3);
    ckw!(wo_a);
    ckw!(wo_b);
    macro_rules! ckm {
        ($field:ident) => {{
            let an = g.m.$field.clone();
            let mut p = m.$field.clone();
            check(concat!("mod.", stringify!($field)), &an, &mut p, h, |p| {
                let mut m2 = m.clone();
                m2.$field.copy_from_slice(p);
                loss(&w, &x, &m2)
            });
        }};
    }
    ckm!(shift);
    ckm!(scale);
    ckm!(gate);
    let mut xc = x.clone();
    check("dx", &g.dx, &mut xc, h, |p| loss(&w, p, &m));
    eprintln!("FLUX.2 single-block backward: gradcheck PASSED (all tensors < 1e-4 rel err)");
}
