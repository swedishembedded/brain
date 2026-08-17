// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA training mechanism: with the base **frozen**, training only the low-rank
//! adapters (A,B) via the gradchecked device trainer drives the flow-matching
//! loss down, and the adapter round-trips through save/load. Needs a GPU:
//! `BRAIN_DEV_GPU=1` (uses the real `DeviceTrainer` path, ModelGradsF32).

use std::collections::HashMap;

use s3dit::grad::Weights;
use s3dit::lora::{LoraAdapter, LoraCfg};
use s3dit::modelgrad::{Cfg, ModelWeights};
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

#[test]
fn lora_only_overfits_with_base_frozen() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the LoRA training test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x51a7);
    let base = init(&c, &mut r).to_f32(); // FROZEN — never mutated below
    let b = batch(&c, &mut r);
    let tr = DeviceTrainer::new(c);
    let lc = LoraCfg::new(8);
    let mut ad = LoraAdapter::new(&c, lc);

    // B=0 at init → adapter is a no-op → same loss as the bare base.
    let (l_base, _) = tr.grads(&base, &b);
    let (l0, _) = tr.grads(&ad.apply(&base), &b);
    assert!((l_base - l0).abs() / l_base.max(1e-9) < 1e-6, "fresh adapter must be a no-op ({l_base} vs {l0})");

    let mut last = l0;
    let mut lmin = l0;
    let mut early = l0; // loss after the first 10 steps
    for i in 0..120 {
        let (loss, g) = tr.grads(&ad.apply(&base), &b);
        ad.step(&g, 0.03);
        lmin = lmin.min(loss);
        if i == 9 {
            early = loss;
        }
        last = loss;
    }
    // A rank-r adapter that only touches the `main` blocks' 7 linears (refiners /
    // embedders / adaLN / norms / final layer stay frozen) can't fully fit an
    // arbitrary target — it converges to a capacity floor and then jitters around
    // it. The mechanism is proven by a clear loss drop that then holds: gradients
    // flow through the weave into A,B and Adam updates them.
    eprintln!("LoRA-only overfit: loss {l0:.5} -> {last:.5} (min {lmin:.5}, {:.2}× lower)", l0 / lmin.max(1e-12));
    assert!(lmin < 0.75 * l0, "LoRA-only training barely moved the loss (min {lmin:.4} vs {l0:.4})");
    assert!(early < 0.9 * l0, "loss did not drop early ({early:.4} after 10 steps vs {l0:.4})");
    assert!(last < 0.85 * l0, "final loss did not hold near the floor ({last:.4} vs {l0:.4})");

    // Save/load round-trip: reloaded adapter reproduces the same effective weights.
    let tensors: HashMap<String, (Vec<usize>, Vec<f32>)> =
        ad.to_tensors().into_iter().map(|(n, s, d)| (n, (s, d))).collect();
    let re = LoraAdapter::from_tensors(&c, lc, &tensors).expect("reload");
    let (wa, wb) = (ad.apply(&base), re.apply(&base));
    let diff = wa.main[0].wq.iter().zip(&wb.main[0].wq).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(diff < 1e-6, "adapter save/load changed the weights (max diff {diff:.2e})");
    eprintln!("LoRA adapter save/load round-trips ({} tensors).", ad.to_tensors().len());
}
