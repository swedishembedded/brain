// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full-model training gates: (1) a finite-difference gradcheck of the entire
//! Z-Image S³-DiT under the flow-matching loss, and (2) overfit-one-batch to
//! ~zero. Together they prove the end-to-end training loop — block backward
//! chained across refiners + main layers, plus the timestep MLP, embedders, and
//! adaLN final layer, with `dc` accumulated across every modulated block — is
//! correct and optimizable. Pure host f64, no GPU.

use zimage::grad::{Grads, Weights};
use zimage::modelgrad::{backward, forward, loss, Cfg, ModelGrads, ModelWeights};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}
fn vof(n: usize, r: &mut impl FnMut() -> f64, s: f64) -> Vec<f64> {
    (0..n).map(|_| r() * s).collect()
}
fn gain(n: usize, r: &mut impl FnMut() -> f64, s: f64) -> Vec<f64> {
    (0..n).map(|_| 1.0 + r() * s).collect()
}

fn cfg() -> Cfg {
    Cfg { dim: 16, nh: 2, n_layers: 2, n_refiner: 1, cap_feat_dim: 12, in_channels: 4, patch: 2, h: 4, w: 4, ncap: 3, t_scale: 1000.0 }
}

fn block_w(c: &Cfg, r: &mut impl FnMut() -> f64) -> Weights {
    let (dim, hd) = (c.dim, c.dim / c.nh);
    let hidden = dim * 8 / 3;
    let cdim = c.dim.min(256);
    Weights {
        wq: vof(dim * dim, r, 0.1), wk: vof(dim * dim, r, 0.1), wv: vof(dim * dim, r, 0.1), wo: vof(dim * dim, r, 0.1),
        w1: vof(hidden * dim, r, 0.1), w2: vof(dim * hidden, r, 0.1), w3: vof(hidden * dim, r, 0.1),
        nq: gain(hd, r, 0.05), nk: gain(hd, r, 0.05),
        an1: gain(dim, r, 0.05), an2: gain(dim, r, 0.05), fn1: gain(dim, r, 0.05), fn2: gain(dim, r, 0.05),
        adaln_w: vof(4 * dim * cdim, r, 0.05), adaln_b: vof(4 * dim, r, 0.05),
    }
}

fn init(c: &Cfg, r: &mut impl FnMut() -> f64) -> ModelWeights {
    let (dim, cdim, pd) = (c.dim, c.dim.min(256), c.patch_dim());
    ModelWeights {
        t0_w: vof(1024 * 256, r, 0.05), t0_b: vof(1024, r, 0.02),
        t2_w: vof(cdim * 1024, r, 0.05), t2_b: vof(cdim, r, 0.02),
        xemb_w: vof(dim * pd, r, 0.1), xemb_b: vof(dim, r, 0.02),
        capn_w: gain(c.cap_feat_dim, r, 0.05), cap1_w: vof(dim * c.cap_feat_dim, r, 0.1), cap1_b: vof(dim, r, 0.02),
        noise_ref: (0..c.n_refiner).map(|_| block_w(c, r)).collect(),
        ctx_ref: (0..c.n_refiner).map(|_| block_w(c, r)).collect(),
        main: (0..c.n_layers).map(|_| block_w(c, r)).collect(),
        fadaln_w: vof(dim * cdim, r, 0.05), fadaln_b: vof(dim, r, 0.02),
        flin_w: vof(pd * dim, r, 0.05), flin_b: vof(pd, r, 0.02),
    }
}

fn bpm(b: &mut Weights) -> Vec<&mut Vec<f64>> {
    vec![&mut b.wq, &mut b.wk, &mut b.wv, &mut b.wo, &mut b.w1, &mut b.w2, &mut b.w3, &mut b.nq, &mut b.nk, &mut b.an1, &mut b.an2, &mut b.fn1, &mut b.fn2, &mut b.adaln_w, &mut b.adaln_b]
}
fn bgr(g: &Grads) -> Vec<&Vec<f64>> {
    vec![&g.wq, &g.wk, &g.wv, &g.wo, &g.w1, &g.w2, &g.w3, &g.nq, &g.nk, &g.an1, &g.an2, &g.fn1, &g.fn2, &g.adaln_w, &g.adaln_b]
}

fn params_mut(m: &mut ModelWeights) -> Vec<&mut Vec<f64>> {
    let mut v: Vec<&mut Vec<f64>> = vec![&mut m.t0_w, &mut m.t0_b, &mut m.t2_w, &mut m.t2_b, &mut m.xemb_w, &mut m.xemb_b, &mut m.capn_w, &mut m.cap1_w, &mut m.cap1_b];
    for b in m.noise_ref.iter_mut().chain(m.ctx_ref.iter_mut()).chain(m.main.iter_mut()) {
        v.extend(bpm(b));
    }
    v.extend([&mut m.fadaln_w, &mut m.fadaln_b, &mut m.flin_w, &mut m.flin_b]);
    v
}
fn grads_ref(g: &ModelGrads) -> Vec<&Vec<f64>> {
    let mut v: Vec<&Vec<f64>> = vec![&g.t0_w, &g.t0_b, &g.t2_w, &g.t2_b, &g.xemb_w, &g.xemb_b, &g.capn_w, &g.cap1_w, &g.cap1_b];
    for b in g.noise_ref.iter().chain(g.ctx_ref.iter()).chain(g.main.iter()) {
        v.extend(bgr(b));
    }
    v.extend([&g.fadaln_w, &g.fadaln_b, &g.flin_w, &g.flin_b]);
    v
}

struct Batch {
    latent: Vec<f64>,
    cap: Vec<f64>,
    t: f64,
    ic: Vec<f64>,
    is: Vec<f64>,
    cc: Vec<f64>,
    cs: Vec<f64>,
    target: Vec<f64>,
}
fn batch(c: &Cfg, r: &mut impl FnMut() -> f64) -> Batch {
    let half = (c.dim / c.nh) / 2;
    Batch {
        latent: vof(c.in_channels * c.h * c.w, r, 1.0),
        cap: vof(c.ncap * c.cap_feat_dim, r, 1.0),
        t: 0.3,
        ic: (0..c.n_img() * half).map(|i| (i as f64 * 0.2).cos()).collect(),
        is: (0..c.n_img() * half).map(|i| (i as f64 * 0.2).sin()).collect(),
        cc: (0..c.ncap * half).map(|i| (i as f64 * 0.2).cos()).collect(),
        cs: (0..c.ncap * half).map(|i| (i as f64 * 0.2).sin()).collect(),
        target: vof(c.n_img() * c.patch_dim(), r, 1.0),
    }
}
fn run_loss(c: &Cfg, w: &ModelWeights, b: &Batch) -> f64 {
    let (pred, _) = forward(c, w, &b.latent, &b.cap, b.t, &b.ic, &b.is, &b.cc, &b.cs);
    loss(&pred, &b.target).0
}

#[test]
fn full_model_gradcheck() {
    let c = cfg();
    let mut r = rng(0xABCD_01);
    let w0 = init(&c, &mut r);
    let b = batch(&c, &mut r);
    let (pred, cache) = forward(&c, &w0, &b.latent, &b.cap, b.t, &b.ic, &b.is, &b.cc, &b.cs);
    let (_l, dpred) = loss(&pred, &b.target);
    let g = backward(&c, &w0, &cache, &dpred);
    let analytic: Vec<Vec<f64>> = grads_ref(&g).into_iter().cloned().collect();

    let h = 1e-4;
    let mut worst_all = 0f64;
    let mut worst_name = 0usize;
    // number of top-level params for labeling
    let nparam = { let mut w = w0.clone(); params_mut(&mut w).len() };
    for pi in 0..nparam {
        let plen = { let mut w = w0.clone(); params_mut(&mut w)[pi].len() };
        let step = (plen / 6).max(1);
        let mut worst = 0f64;
        for i in (0..plen).step_by(step) {
            let mut wp = w0.clone();
            let orig = params_mut(&mut wp)[pi][i];
            params_mut(&mut wp)[pi][i] = orig + h;
            let lp = run_loss(&c, &wp, &b);
            params_mut(&mut wp)[pi][i] = orig - h;
            let lm = run_loss(&c, &wp, &b);
            let num = (lp - lm) / (2.0 * h);
            let a = analytic[pi][i];
            let denom = a.abs().max(num.abs()).max(1e-4);
            worst = worst.max((a - num).abs() / denom);
        }
        if worst > worst_all {
            worst_all = worst;
            worst_name = pi;
        }
    }
    eprintln!("Full-model gradcheck: worst rel err = {worst_all:.2e} (param #{worst_name})");
    assert!(worst_all < 1e-3, "full-model gradcheck failed: {worst_all:.3e}");
}

#[test]
fn full_model_overfits() {
    let c = cfg();
    let mut r = rng(0x1122_33);
    let mut w = init(&c, &mut r);
    let b = batch(&c, &mut r);

    let nparams: usize = { let mut wc = w.clone(); params_mut(&mut wc).iter().map(|p| p.len()).sum() };
    let mut m = vec![0f64; nparams];
    let mut v = vec![0f64; nparams];
    let (lr, b1, b2, eps): (f64, f64, f64, f64) = (3e-3, 0.9, 0.999, 1e-8);

    let l0 = run_loss(&c, &w, &b);
    let mut l = l0;
    for step in 1..=300 {
        let (pred, cache) = forward(&c, &w, &b.latent, &b.cap, b.t, &b.ic, &b.is, &b.cc, &b.cs);
        let (loss_v, dpred) = loss(&pred, &b.target);
        l = loss_v;
        let g = backward(&c, &w, &cache, &dpred);
        let grads: Vec<Vec<f64>> = grads_ref(&g).into_iter().cloned().collect();
        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        let mut off = 0;
        for (pi, param) in params_mut(&mut w).into_iter().enumerate() {
            for j in 0..param.len() {
                let gj = grads[pi][j];
                m[off] = b1 * m[off] + (1.0 - b1) * gj;
                v[off] = b2 * v[off] + (1.0 - b2) * gj * gj;
                param[j] -= lr * (m[off] / bc1) / ((v[off] / bc2).sqrt() + eps);
                off += 1;
            }
        }
        if step % 75 == 0 {
            eprintln!("  step {step:3}: loss = {l:.3e}");
        }
    }
    eprintln!("Full-model overfit: loss {l0:.3e} -> {l:.3e} ({nparams} params)");
    assert!(l < l0 * 1e-2, "full model did not overfit: {l0:.3e} -> {l:.3e}");
}
