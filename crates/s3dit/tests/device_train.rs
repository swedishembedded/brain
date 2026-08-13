// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device full-model training: (1) the GPU trainer's gradients match the
//! gradchecked host reference (`modelgrad`) at fp32, and (2) it overfits one
//! batch on the GPU. This is the end-to-end device training loop — the DiT block
//! stack on the GPU (persistent BlockDev engine) wrapped by the host timestep
//! MLP / embedders / final layer / flow-matching loss. Needs a GPU:
//! `BRAIN_DEV_GPU=1`.

use s3dit::grad::{Grads, Weights};
use s3dit::modelgrad::{self, Cfg, ModelGrads, ModelWeights};
use s3dit::train::{Batch, DeviceTrainer};

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
    let (dim, hd, hidden, cdim) = (c.dim, c.dim / c.nh, c.dim * 8 / 3, c.dim.min(256));
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
        t0_w: vof(1024 * 256, r, 0.05), t0_b: vof(1024, r, 0.02), t2_w: vof(cdim * 1024, r, 0.05), t2_b: vof(cdim, r, 0.02),
        xemb_w: vof(dim * pd, r, 0.1), xemb_b: vof(dim, r, 0.02),
        capn_w: gain(c.cap_feat_dim, r, 0.05), cap1_w: vof(dim * c.cap_feat_dim, r, 0.1), cap1_b: vof(dim, r, 0.02),
        noise_ref: (0..c.n_refiner).map(|_| block_w(c, r)).collect(),
        ctx_ref: (0..c.n_refiner).map(|_| block_w(c, r)).collect(),
        main: (0..c.n_layers).map(|_| block_w(c, r)).collect(),
        fadaln_w: vof(dim * cdim, r, 0.05), fadaln_b: vof(dim, r, 0.02), flin_w: vof(pd * dim, r, 0.05), flin_b: vof(pd, r, 0.02),
    }
}
fn batch(c: &Cfg, r: &mut impl FnMut() -> f64) -> Batch {
    let half = (c.dim / c.nh) / 2;
    Batch {
        latent: vof(c.in_channels * c.h * c.w, r, 1.0),
        cap: vof(c.ncap * c.cap_feat_dim, r, 1.0),
        t: 0.3,
        img_cos: (0..c.n_img() * half).map(|i| (i as f64 * 0.2).cos()).collect(),
        img_sin: (0..c.n_img() * half).map(|i| (i as f64 * 0.2).sin()).collect(),
        cap_cos: (0..c.ncap * half).map(|i| (i as f64 * 0.2).cos()).collect(),
        cap_sin: (0..c.ncap * half).map(|i| (i as f64 * 0.2).sin()).collect(),
        target: vof(c.n_img() * c.patch_dim(), r, 1.0),
    }
}

/// Relative L2 error, well-defined for zero vectors (unmodulated blocks'
/// adaLN grads are legitimately all-zero, where cosine is 0/0).
fn rel_l2(host: &[f64], dev: &[f64]) -> f64 {
    let nh = host.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff = host.iter().zip(dev).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    diff / nh.max(1e-9)
}

// Flat grad views in a fixed order (model tensors, then each block's 15, then final).
fn bgr(g: &Grads) -> Vec<&Vec<f64>> {
    vec![&g.wq, &g.wk, &g.wv, &g.wo, &g.w1, &g.w2, &g.w3, &g.nq, &g.nk, &g.an1, &g.an2, &g.fn1, &g.fn2, &g.adaln_w, &g.adaln_b]
}
fn mgr(g: &ModelGrads) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = vec![g.t0_w.clone(), g.t0_b.clone(), g.t2_w.clone(), g.t2_b.clone(), g.xemb_w.clone(), g.xemb_b.clone(), g.capn_w.clone(), g.cap1_w.clone(), g.cap1_b.clone()];
    for b in g.noise_ref.iter().chain(g.ctx_ref.iter()).chain(g.main.iter()) {
        v.extend(bgr(b).into_iter().cloned());
    }
    v.extend([g.fadaln_w.clone(), g.fadaln_b.clone(), g.flin_w.clone(), g.flin_b.clone()]);
    v
}
fn f2d(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
/// Flatten the device (f32-block) grads to f64, same order as `mgr`.
fn df(g: &s3dit::modelgrad::ModelGradsF32) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = vec![g.t0_w.clone(), g.t0_b.clone(), g.t2_w.clone(), g.t2_b.clone(), g.xemb_w.clone(), g.xemb_b.clone(), g.capn_w.clone(), g.cap1_w.clone(), g.cap1_b.clone()];
    for b in g.noise_ref.iter().chain(g.ctx_ref.iter()).chain(g.main.iter()) {
        v.extend([f2d(&b.wq), f2d(&b.wk), f2d(&b.wv), f2d(&b.wo), f2d(&b.w1), f2d(&b.w2), f2d(&b.w3), f2d(&b.nq), f2d(&b.nk), f2d(&b.an1), f2d(&b.an2), f2d(&b.fn1), f2d(&b.fn2), f2d(&b.adaln_w), f2d(&b.adaln_b)]);
    }
    v.extend([g.fadaln_w.clone(), g.fadaln_b.clone(), g.flin_w.clone(), g.flin_b.clone()]);
    v
}

#[test]
fn device_grads_match_host() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_DEV_GPU=1 (needs a GPU) for the device full-model training test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x9911);
    let w = init(&c, &mut r);
    let b = batch(&c, &mut r);

    // host reference (gradchecked)
    let (pred, cache) = modelgrad::forward(&c, &w, &b.latent, &b.cap, b.t, &b.img_cos, &b.img_sin, &b.cap_cos, &b.cap_sin);
    let (hl, dpred) = modelgrad::loss(&pred, &b.target);
    let hg = modelgrad::backward(&c, &w, &cache, &dpred);

    // device trainer
    let tr = DeviceTrainer::new(c);
    let (dl, dg) = tr.grads(&w.to_f32(), &b);

    eprintln!("loss host={hl:.6} device={dl:.6}");
    assert!((hl - dl).abs() / hl.abs().max(1e-9) < 1e-3, "loss mismatch");
    let (hv, dv) = (mgr(&hg), df(&dg));
    let mut worst = 0.0f64;
    let mut worst_i = 0;
    for (i, (h, d)) in hv.iter().zip(&dv).enumerate() {
        let rel = rel_l2(h, d);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    eprintln!("Device vs host grads: worst rel_l2 = {worst:.2e} (tensor #{worst_i} of {})", hv.len());
    assert!(worst < 5e-3, "device grad rel_l2 {worst:.3e} too high (tensor #{worst_i})");
    eprintln!("Device full-model training gradients match the gradchecked host reference.");
}

#[test]
fn device_model_overfits() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_DEV_GPU=1 for the device overfit test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x4242);
    let mut w = init(&c, &mut r);
    let b = batch(&c, &mut r);
    let tr = DeviceTrainer::new(c);

    // flat mutable param view + Adam
    fn bpm(b: &mut Weights) -> Vec<&mut Vec<f64>> {
        vec![&mut b.wq, &mut b.wk, &mut b.wv, &mut b.wo, &mut b.w1, &mut b.w2, &mut b.w3, &mut b.nq, &mut b.nk, &mut b.an1, &mut b.an2, &mut b.fn1, &mut b.fn2, &mut b.adaln_w, &mut b.adaln_b]
    }
    fn pm(m: &mut ModelWeights) -> Vec<&mut Vec<f64>> {
        let mut v: Vec<&mut Vec<f64>> = vec![&mut m.t0_w, &mut m.t0_b, &mut m.t2_w, &mut m.t2_b, &mut m.xemb_w, &mut m.xemb_b, &mut m.capn_w, &mut m.cap1_w, &mut m.cap1_b];
        for bl in m.noise_ref.iter_mut().chain(m.ctx_ref.iter_mut()).chain(m.main.iter_mut()) {
            v.extend(bpm(bl));
        }
        v.extend([&mut m.fadaln_w, &mut m.fadaln_b, &mut m.flin_w, &mut m.flin_b]);
        v
    }
    let nparams: usize = { let mut wc = w.clone(); pm(&mut wc).iter().map(|p| p.len()).sum() };
    let mut mm = vec![0f64; nparams];
    let mut vv = vec![0f64; nparams];
    let (lr, b1, b2, eps): (f64, f64, f64, f64) = (3e-3, 0.9, 0.999, 1e-8);

    let (l0, _) = tr.grads(&w.to_f32(), &b);
    let mut l = l0;
    for step in 1..=120 {
        let (loss, g) = tr.grads(&w.to_f32(), &b);
        l = loss;
        let grads: Vec<Vec<f64>> = df(&g);
        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        let mut off = 0;
        for (pi, param) in pm(&mut w).into_iter().enumerate() {
            for j in 0..param.len() {
                let gj = grads[pi][j];
                mm[off] = b1 * mm[off] + (1.0 - b1) * gj;
                vv[off] = b2 * vv[off] + (1.0 - b2) * gj * gj;
                param[j] -= lr * (mm[off] / bc1) / ((vv[off] / bc2).sqrt() + eps);
                off += 1;
            }
        }
        if step % 40 == 0 {
            eprintln!("  step {step:3}: loss = {l:.3e}");
        }
    }
    eprintln!("Device overfit: loss {l0:.3e} -> {l:.3e}");
    assert!(l < l0 * 5e-2, "device model did not overfit: {l0:.3e} -> {l:.3e}");
}
