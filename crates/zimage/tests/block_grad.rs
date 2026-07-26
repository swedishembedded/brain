// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gradcheck for the S³-DiT block backward ([`zimage::grad`]).
//!
//! This is the correctness gate for DiT training: it perturbs every trainable
//! tensor (and the block inputs `x`, `c`) element-by-element and confirms the
//! analytic gradient matches the central-difference estimate. The scalar
//! objective is `L = Σ out·seed`, so `dout = seed` exactly and the backward is
//! checked in isolation. Pure host f64 — no GPU, runs in `make test`.

use zimage::grad::{backward, forward, Dims, Weights};

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

/// `L = Σ out·seed` for a given weights/x/c — the finite-difference probe.
fn loss(d: Dims, w: &Weights, x: &[f64], c: &[f64], cos: &[f64], sin: &[f64], seed: &[f64]) -> f64 {
    let (out, _) = forward(d, w, x, c, cos, sin);
    out.iter().zip(seed).map(|(&o, &s)| o * s).sum()
}

/// Central-difference check of `analytic` against `param` for a scalar loss
/// `f(param[i] ± h)`. Returns the worst relative error over all elements.
fn check(name: &str, analytic: &[f64], param: &mut [f64], h: f64, mut f: impl FnMut(&mut [f64]) -> f64) {
    let mut worst = 0f64;
    let mut worst_i = 0;
    // check a stride of elements (all if small, else sampled) to keep it quick.
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
    eprintln!("  {name:12} worst rel err = {worst:.2e} (elem {worst_i}/{})", param.len());
    assert!(worst < 1e-4, "{name} gradcheck failed: {worst:.3e} at elem {worst_i}");
}

#[test]
fn block_backward_matches_finite_difference() {
    // Small config: dim 8, 2 heads (hd 4), 3 tokens. cdim=8, hidden=21.
    let d = Dims::new(3, 8, 2);
    let (t, dim, hd, half) = (d.t, d.dim, d.hd, d.half());
    let mut r = rng(0xD17_C0DE);
    let w = Weights {
        wq: vec_of(dim * dim, &mut r, 0.3),
        wk: vec_of(dim * dim, &mut r, 0.3),
        wv: vec_of(dim * dim, &mut r, 0.3),
        wo: vec_of(dim * dim, &mut r, 0.3),
        w1: vec_of(d.hidden * dim, &mut r, 0.3),
        w2: vec_of(dim * d.hidden, &mut r, 0.3),
        w3: vec_of(d.hidden * dim, &mut r, 0.3),
        nq: vec_of(hd, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        nk: vec_of(hd, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        an1: vec_of(dim, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        an2: vec_of(dim, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        fn1: vec_of(dim, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        fn2: vec_of(dim, &mut r, 0.2).iter().map(|v| 1.0 + v).collect(),
        adaln_w: vec_of(4 * dim * d.cdim, &mut r, 0.1),
        adaln_b: vec_of(4 * dim, &mut r, 0.1),
    };
    let x = vec_of(t * dim, &mut r, 1.0);
    let c = vec_of(d.cdim, &mut r, 1.0);
    let cos: Vec<f64> = (0..t * half).map(|i| (i as f64 * 0.3).cos()).collect();
    let sin: Vec<f64> = (0..t * half).map(|i| (i as f64 * 0.3).sin()).collect();
    let seed = vec_of(t * dim, &mut r, 1.0);

    let (_out, cache) = forward(d, &w, &x, &c, &cos, &sin);
    let g = backward(d, &w, &cache, &seed);

    let h = 1e-4;
    // weights
    macro_rules! ckw {
        ($field:ident) => {{
            let an = g.$field.clone();
            let mut wc = w.clone();
            check(stringify!($field), &an, &mut wc.$field, h, |p| {
                let mut w2 = w.clone();
                w2.$field.copy_from_slice(p);
                loss(d, &w2, &x, &c, &cos, &sin, &seed)
            });
        }};
    }
    ckw!(wq);
    ckw!(wk);
    ckw!(wv);
    ckw!(wo);
    ckw!(w1);
    ckw!(w2);
    ckw!(w3);
    ckw!(nq);
    ckw!(nk);
    ckw!(an1);
    ckw!(an2);
    ckw!(fn1);
    ckw!(fn2);
    ckw!(adaln_w);
    ckw!(adaln_b);
    // inputs: dx, dc
    {
        let mut xc = x.clone();
        check("dx", &g.dx, &mut xc, h, |p| loss(d, &w, p, &c, &cos, &sin, &seed));
        let mut cc = c.clone();
        check("dc", &g.dc, &mut cc, h, |p| loss(d, &w, &x, p, &cos, &sin, &seed));
    }
    eprintln!("Z-Image block backward: gradcheck PASSED (all tensors < 1e-4 rel err)");
}
