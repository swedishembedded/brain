// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Whole-model training gates for FLUX.2 ([`flux2::modelgrad`]):
//! (1) a finite-difference gradcheck of the entire DiT under the
//! rectified-flow velocity-MSE loss — including the conditioning path
//! (timestep embedding → time_in MLP → the three global modulation linears +
//! the final adaLN, with the site grads accumulated across the whole block
//! stack) — and (2) overfit-one-batch to prove the loop is optimizable.
//! Pure host f64, no GPU.

use flux2::modelgrad::{
    backward, forward, grad_views, init_model, loss, make_flow_batch, params_mut, Batch, Cfg,
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

fn batch(c: &Cfg, r: &mut impl FnMut() -> f64) -> Batch<f64> {
    let x0: Vec<f64> = (0..c.n_img() * c.in_channels).map(|_| r()).collect();
    let ctx: Vec<f64> = (0..c.txt_len * c.context_in_dim).map(|_| r()).collect();
    let noise: Vec<f64> = (0..x0.len()).map(|_| r()).collect();
    make_flow_batch(c, &x0, &ctx, 0.45, &noise)
}

fn run_loss(c: &Cfg, w: &flux2::modelgrad::ModelWeights<f64>, b: &Batch<f64>) -> f64 {
    let (pred, _) = forward(c, w, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    loss(&pred, &b.target).0
}

#[test]
fn full_model_gradcheck() {
    let c = Cfg::tiny();
    let w0 = init_model::<f64>(&c, 0xF1_ABCD);
    let mut r = rng(0xBEEF_02);
    let b = batch(&c, &mut r);

    let (pred, cache) = forward(&c, &w0, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    let (_l, dpred) = loss(&pred, &b.target);
    let g = backward(&c, &w0, &cache, &dpred);
    let analytic: Vec<(String, Vec<f64>)> =
        grad_views(&g).into_iter().map(|(n, v)| (n, v.clone())).collect();

    let h = 1e-4;
    let mut worst_all = 0f64;
    let mut worst_name = String::new();
    let nparam = { let mut w = w0.clone(); params_mut(&mut w).len() };
    assert_eq!(nparam, analytic.len(), "param/grad enumeration mismatch");
    for pi in 0..nparam {
        let plen = { let mut w = w0.clone(); params_mut(&mut w)[pi].1.len() };
        let step = (plen / 6).max(1);
        let mut worst = 0f64;
        for i in (0..plen).step_by(step) {
            let mut wp = w0.clone();
            let orig = params_mut(&mut wp)[pi].1[i];
            params_mut(&mut wp)[pi].1[i] = orig + h;
            let lp = run_loss(&c, &wp, &b);
            params_mut(&mut wp)[pi].1[i] = orig - h;
            let lm = run_loss(&c, &wp, &b);
            let num = (lp - lm) / (2.0 * h);
            let a = analytic[pi].1[i];
            let denom = a.abs().max(num.abs()).max(1e-4);
            worst = worst.max((a - num).abs() / denom);
        }
        if worst > worst_all {
            worst_all = worst;
            worst_name = analytic[pi].0.clone();
        }
    }
    eprintln!("FLUX.2 full-model gradcheck: worst rel err = {worst_all:.2e} ({worst_name})");
    assert!(worst_all < 1e-3, "full-model gradcheck failed: {worst_all:.3e} at {worst_name}");
}

#[test]
fn full_model_overfits() {
    let c = Cfg::tiny();
    let mut w = init_model::<f64>(&c, 0x11_2233);
    let mut r = rng(0x44_5566);
    let b = batch(&c, &mut r);

    let nparams: usize = { let mut wc = w.clone(); params_mut(&mut wc).iter().map(|(_, p)| p.len()).sum() };
    let mut m = vec![0f64; nparams];
    let mut v = vec![0f64; nparams];
    let (lr, b1, b2, eps): (f64, f64, f64, f64) = (3e-3, 0.9, 0.999, 1e-8);

    let l0 = run_loss(&c, &w, &b);
    let mut l = l0;
    for step in 1..=250 {
        let (pred, cache) = forward(&c, &w, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
        let (lv, dpred) = loss(&pred, &b.target);
        l = lv;
        let g = backward(&c, &w, &cache, &dpred);
        let grads: Vec<Vec<f64>> = grad_views(&g).into_iter().map(|(_, v)| v.clone()).collect();
        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        let mut off = 0;
        for (pi, (_, param)) in params_mut(&mut w).into_iter().enumerate() {
            for j in 0..param.len() {
                let gj = grads[pi][j];
                m[off] = b1 * m[off] + (1.0 - b1) * gj;
                v[off] = b2 * v[off] + (1.0 - b2) * gj * gj;
                param[j] -= lr * (m[off] / bc1) / ((v[off] / bc2).sqrt() + eps);
                off += 1;
            }
        }
        if step % 50 == 0 {
            eprintln!("  step {step:3}: loss = {l:.3e}");
        }
    }
    eprintln!("FLUX.2 full-model overfit: loss {l0:.3e} -> {l:.3e} ({nparams} params)");
    assert!(l < l0 * 1e-2, "full model did not overfit: {l0:.3e} -> {l:.3e}");
}
