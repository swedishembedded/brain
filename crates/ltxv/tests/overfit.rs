// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overfit-one-batch: the canonical "training actually works" gate, on the
//! WHOLE tiny video-only LTX DiT rather than one block.
//!
//! `block_grad.rs` and `gradcheck::check_ltxv` prove the analytic gradients
//! are *correct*; this proves they are *usable for optimization* - Adam
//! over every parameter drives the flow-matching velocity MSE toward zero on
//! a single fixed batch. Running it whole-model is what makes it cover the
//! conditioning path too: the timestep MLP, `adaln_single.linear`, every
//! block's own `scale_shift_table` and the output stage's `scale_shift_table`
//! only ever receive gradient through the per-token fold, so a
//! partially-wrong fold shows up here as a loss that stalls.

use ltxv::modelgrad::{grad_views, grads, init_model, make_flow_batch, params_mut, Cfg};

#[test]
fn the_whole_model_overfits_one_batch() {
    let cfg = Cfg::tiny();
    let mut w = init_model::<f64>(&cfg, 0xF17_0BEE);

    let x0: Vec<f64> = (0..cfg.t * cfg.in_channels).map(|i| ((i % 19) as f64 / 19.0 - 0.5) * 1.2).collect();
    let noise: Vec<f64> = (0..x0.len()).map(|i| ((i % 11) as f64 / 11.0 - 0.5) * 0.7).collect();
    let ctx: Vec<f64> = (0..cfg.context_len * cfg.dim).map(|i| ((i % 5) as f64 / 5.0 - 0.5) * 1.5).collect();
    let b = make_flow_batch(&cfg, &x0, &ctx, 0.4, &noise);

    // Adam state, flat over every parameter tensor.
    let nparams: usize = params_mut(&mut w).iter().map(|(_, p)| p.len()).sum();
    let mut m = vec![0f64; nparams];
    let mut v = vec![0f64; nparams];
    let (lr, b1, b2, eps) = (3e-3f64, 0.9f64, 0.999f64, 1e-8f64);

    let mut first = 0.0;
    let mut last = 0.0;
    for step in 1..=400u64 {
        let (l, g) = grads(&cfg, &w, &b);
        if step == 1 {
            first = l;
        }
        last = l;
        if step % 80 == 0 || step == 1 {
            println!("  step {step:>4}  loss {l:.8}");
        }
        let gv: Vec<Vec<f64>> = grad_views(&g).into_iter().map(|(_, x)| x.clone()).collect();
        let mut off = 0;
        for ((_, p), gt) in params_mut(&mut w).into_iter().zip(&gv) {
            for (i, (pi, &gi)) in p.iter_mut().zip(gt.iter()).enumerate() {
                let k = off + i;
                m[k] = b1 * m[k] + (1.0 - b1) * gi;
                v[k] = b2 * v[k] + (1.0 - b2) * gi * gi;
                let mh = m[k] / (1.0 - b1.powi(step as i32));
                let vh = v[k] / (1.0 - b2.powi(step as i32));
                *pi -= lr * mh / (vh.sqrt() + eps);
            }
            off += gt.len();
        }
    }
    println!("model overfit: loss {first:.8} -> {last:.8} over 400 Adam steps");
    assert!(last < first * 0.01, "the loss must collapse: {first} -> {last}");
    assert!(last < 1e-3, "the loss must approach zero, got {last}");
}
