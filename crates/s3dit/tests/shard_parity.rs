// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-sharding parity: cutting the main-layer stack and passing the residual
//! through a flat `[uni ‖ c]` (forward) / `[d_uni ‖ dc]` (backward) boundary must
//! produce the same gradients as the single-device path. This is the correctness
//! gate for training the full 6B pipeline-parallel: it proves (a) the boundary
//! slab is complete — only `[uni ‖ c]` crosses each cut — and (b) the two halves
//! are independent, so each can stream its layer slice on its own card with
//! weights in RAM (the memory-safe path). Needs a GPU: `BRAIN_DEV_GPU=1`.

use s3dit::modelgrad::{Cfg, ModelGradsF32, ModelWeights};
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
// 3 main layers so we can cut at 1 and 2.
fn cfg() -> Cfg {
    Cfg { dim: 16, nh: 2, n_layers: 3, n_refiner: 1, cap_feat_dim: 12, in_channels: 4, patch: 2, h: 4, w: 4, ncap: 3, t_scale: 1000.0 }
}
fn block_w(c: &Cfg, r: &mut impl FnMut() -> f64) -> s3dit::grad::Weights {
    let (dim, hd, hidden, cdim) = (c.dim, c.dim / c.nh, c.dim * 8 / 3, c.dim.min(256));
    s3dit::grad::Weights {
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

fn f2d(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
fn flat(g: &ModelGradsF32) -> Vec<Vec<f64>> {
    let mut v = vec![g.t0_w.clone(), g.t0_b.clone(), g.t2_w.clone(), g.t2_b.clone(), g.xemb_w.clone(), g.xemb_b.clone(), g.capn_w.clone(), g.cap1_w.clone(), g.cap1_b.clone()];
    for b in g.noise_ref.iter().chain(g.ctx_ref.iter()).chain(g.main.iter()) {
        v.extend([f2d(&b.wq), f2d(&b.wk), f2d(&b.wv), f2d(&b.wo), f2d(&b.w1), f2d(&b.w2), f2d(&b.w3), f2d(&b.nq), f2d(&b.nk), f2d(&b.an1), f2d(&b.an2), f2d(&b.fn1), f2d(&b.fn2), f2d(&b.adaln_w), f2d(&b.adaln_b)]);
    }
    v.extend([g.fadaln_w.clone(), g.fadaln_b.clone(), g.flin_w.clone(), g.flin_b.clone()]);
    v
}
fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let d = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / na.max(1e-9)
}

#[test]
fn pipeline_cut_matches_single_device() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the pipeline-sharding parity test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x5A5A);
    let w = init(&c, &mut r);
    let b = batch(&c, &mut r);
    let tr = DeviceTrainer::new(c);

    let (l0, g0) = tr.grads(&w.to_f32(), &b);
    let f0 = flat(&g0);
    for cut in 1..c.n_layers {
        let (l1, g1) = tr.grads_pipelined(&w.to_f32(), &b, cut);
        assert!((l0 - l1).abs() / l0.abs().max(1e-9) < 1e-5, "cut {cut}: loss {l0} vs {l1}");
        let f1 = flat(&g1);
        let mut worst = 0f64;
        for (a, b) in f0.iter().zip(&f1) {
            worst = worst.max(rel_l2(a, b));
        }
        eprintln!("cut {cut}: loss match, worst grad rel_l2 = {worst:.2e}");
        // f32 boundary (uni/c fwd, d_uni/dc bwd) → ~fp32 rounding, not bit-exact.
        assert!(worst < 1e-4, "cut {cut}: pipeline grad rel_l2 {worst:.3e} too high");
    }
    eprintln!("Pipeline sharding is grad-parity with single-device across all cuts.");
}
