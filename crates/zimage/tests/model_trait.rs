// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image as a brain `Model`: proves the adapter is a correct `Model` (param
//! round-trip) and that the *generic, architecture-agnostic* distributed
//! optimiser (`model::DdpOptimizer`, reducing through a `Collective`) drives it to
//! convergence — i.e. Z-Image now rides the same training machinery as every other
//! brain model, so multi-GPU / multi-machine / federated come for free. Needs a
//! GPU: `BRAIN_DEV_GPU=1`.

use model::{Collective, DdpOptimizer, HostCollective, Model};
use zimage::modelgrad::{Cfg, ModelWeights};
use zimage::train::{Batch, ZTrainModel};

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

#[test]
fn zimage_is_a_model_and_trains_via_ddp_optimizer() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_DEV_GPU=1 (needs a GPU) for the Z-Image Model-trait test");
        return;
    }
    let c = cfg();
    let mut r = rng(0x2222);
    let m = ZTrainModel::from_weights(c, init(&c, &mut r));

    // (1) param round-trip through the trait's named accessors.
    let names = m.param_names();
    assert!(names.contains(&"t0_w".to_string()) && names.contains(&"main.1.adaln_w".to_string()), "names: {}", names.len());
    let probe = &names[names.len() / 2];
    let before = m.read_weight(probe);
    let bumped: Vec<f32> = before.iter().map(|&x| x + 0.5).collect();
    m.write_weight(probe, &bumped);
    let after = m.read_weight(probe);
    for (a, b) in after.iter().zip(&bumped) {
        assert!((a - b).abs() < 1e-6, "round-trip mismatch");
    }
    m.write_weight(probe, &before); // restore

    // (2) train through the GENERIC distributed optimiser (world=1 → local AdamW
    // on the reduced grad). Proves Z-Image drives the same machinery every model
    // uses; scaling out is just a bigger Collective.
    m.load_batch(batch(&c, &mut r));
    let coll: std::sync::Arc<dyn Collective> = HostCollective::new(1);
    let mut opt = DdpOptimizer::new(&m);

    m.zero_grads();
    let l0 = m.forward();
    let mut l = l0;
    for t in 1..=120u32 {
        m.zero_grads();
        l = m.forward();
        m.backward();
        opt.step(&m, &*coll, 0, t, 3e-3, 0.0, None);
    }
    eprintln!("Z-Image via DdpOptimizer: loss {l0:.3e} -> {l:.3e}");
    assert!(l < l0 * 0.1, "Z-Image did not train through DdpOptimizer: {l0:.3e} -> {l:.3e}");
    eprintln!("Z-Image is a brain Model and trains through the generic distributed optimiser.");
}
