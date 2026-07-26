// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two-card pipeline training end-to-end: `ShardTrainer` splits the DiT across
//! both P40s (front + main[0,cut) on card 0, main[cut,end) on card 1, head on
//! host), streaming each stage's layers so neither card nor RAM can overflow. We
//! check (1) its gradients match the single-device path and (2) it overfits a
//! batch — trained across two cards. Needs 2 GPUs: `BRAIN_DEV_GPU=1`.

use zimage::modelgrad::{Cfg, ModelGrads, ModelWeights};
use zimage::shard::ShardTrainer;
use zimage::train::{Batch, DeviceTrainer};

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
    Cfg { dim: 16, nh: 2, n_layers: 4, n_refiner: 1, cap_feat_dim: 12, in_channels: 4, patch: 2, h: 4, w: 4, ncap: 3, t_scale: 1000.0 }
}
fn block_w(c: &Cfg, r: &mut impl FnMut() -> f64) -> zimage::grad::Weights {
    let (dim, hd, hidden, cdim) = (c.dim, c.dim / c.nh, c.dim * 8 / 3, c.dim.min(256));
    zimage::grad::Weights {
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
fn flat(g: &ModelGrads) -> Vec<Vec<f64>> {
    let mut v = vec![g.t0_w.clone(), g.t2_w.clone(), g.xemb_w.clone(), g.cap1_w.clone(), g.fadaln_w.clone(), g.flin_w.clone()];
    for b in g.noise_ref.iter().chain(g.ctx_ref.iter()).chain(g.main.iter()) {
        v.extend([b.wq.clone(), b.w1.clone(), b.adaln_w.clone()]);
    }
    v
}
fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let d = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / na.max(1e-9)
}

#[test]
fn two_card_pipeline_matches_single_device() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_DEV_GPU=1 (needs 2 GPUs) for the 2-card pipeline test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x7C7C);
    let w = init(&c, &mut r);
    let b = batch(&c, &mut r);

    let single = DeviceTrainer::new(c);
    let (l0, g0) = single.grads(&w, &b);

    let pipe = ShardTrainer::new(c, 2); // split 4 main layers 2|2 across the cards
    let (l1, g1) = pipe.grads(&w, &b);

    assert!((l0 - l1).abs() / l0.abs().max(1e-9) < 1e-5, "loss: single {l0} vs 2-card {l1}");
    let (f0, f1) = (flat(&g0), flat(&g1));
    let mut worst = 0f64;
    for (a, b) in f0.iter().zip(&f1) {
        worst = worst.max(rel_l2(a, b));
    }
    eprintln!("2-card pipeline vs single-device: loss match, worst grad rel_l2 = {worst:.2e}");
    assert!(worst < 1e-4, "2-card grad rel_l2 {worst:.3e} too high");
    eprintln!("Z-Image trains pipeline-parallel across both P40s, grad-parity with single-device.");
}

#[test]
fn gpipe_microbatched_matches_summed_grads() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_DEV_GPU=1 (needs 2 GPUs) for the GPipe microbatch test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x9E9E);
    let w = init(&c, &mut r);
    let mbs: Vec<Batch> = (0..4).map(|_| batch(&c, &mut r)).collect();

    // ground truth: single-device, grads summed over the microbatches.
    let single = DeviceTrainer::new(c);
    let mut ref_loss = 0.0;
    let mut ref_g: Option<ModelGrads> = None;
    for mb in &mbs {
        let (l, g) = single.grads(&w, mb);
        ref_loss += l;
        ref_g = Some(match ref_g {
            None => g,
            Some(mut acc) => {
                add_grads(&mut acc, &g);
                acc
            }
        });
    }
    let ref_g = ref_g.unwrap();

    // GPipe: both cards concurrent, one thread each.
    let pipe = ShardTrainer::new(c, 2);
    let t0 = std::time::Instant::now();
    let (loss, g) = pipe.grads_microbatched(&w, &mbs);
    let dt = t0.elapsed().as_secs_f64();

    assert!((ref_loss - loss).abs() / ref_loss.abs().max(1e-9) < 1e-3, "loss: summed {ref_loss} vs gpipe {loss}");
    let (f0, f1) = (flat(&ref_g), flat(&g));
    let mut worst = 0f64;
    for (a, b) in f0.iter().zip(&f1) {
        worst = worst.max(rel_l2(a, b));
    }
    eprintln!("GPipe microbatched ({} mbs) vs summed single-device: worst grad rel_l2 = {worst:.2e}, {dt:.3}s", mbs.len());
    assert!(worst < 1e-3, "GPipe grad rel_l2 {worst:.3e} too high");
    eprintln!("GPipe pipeline (both P40s concurrent) is grad-parity with the summed single-device grads.");
}

fn add_grads(acc: &mut ModelGrads, g: &ModelGrads) {
    let gv = flat(g);
    for (a, b) in flat_mut(acc).into_iter().zip(gv.iter()) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += *y;
        }
    }
}
fn flat_mut(g: &mut ModelGrads) -> Vec<&mut Vec<f64>> {
    let mut v = vec![&mut g.t0_w, &mut g.t2_w, &mut g.xemb_w, &mut g.cap1_w, &mut g.fadaln_w, &mut g.flin_w];
    for b in g.noise_ref.iter_mut().chain(g.ctx_ref.iter_mut()).chain(g.main.iter_mut()) {
        v.extend([&mut b.wq, &mut b.w1, &mut b.adaln_w]);
    }
    v
}
