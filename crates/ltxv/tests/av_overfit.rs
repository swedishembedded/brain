// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overfit-one-batch: the canonical "training actually works" gate, on the
//! WHOLE tiny audio+video LTX DiT rather than one block - the AV twin of
//! `crates/ltxv/tests/overfit.rs`.
//!
//! `av_block_grad.rs` and `gradcheck::check_ltxv_av`/`check_ltxv_av_
//! conditioning` prove the analytic gradients are *correct*; this proves
//! they are *usable for optimization* - Adam over every parameter (both
//! streams, the audio<->video cross-attention, all six timestep MLPs)
//! drives the combined flow-matching velocity MSE toward zero on a single
//! fixed batch.

use ltxv::av_modelgrad::{grad_views, grads, init_model, make_av_flow_batch, params_mut, AvCfg};

#[test]
fn the_whole_av_model_overfits_one_batch() {
    let cfg = AvCfg::tiny();
    let mut w = init_model::<f64>(&cfg, 0xF17_0BEE);

    let v_x0: Vec<f64> = (0..cfg.tv * cfg.v_in_channels).map(|i| ((i % 19) as f64 / 19.0 - 0.5) * 1.2).collect();
    let a_x0: Vec<f64> = (0..cfg.ta * cfg.a_in_channels).map(|i| ((i % 17) as f64 / 17.0 - 0.5) * 1.0).collect();
    let v_noise: Vec<f64> = (0..v_x0.len()).map(|i| ((i % 11) as f64 / 11.0 - 0.5) * 0.7).collect();
    let a_noise: Vec<f64> = (0..a_x0.len()).map(|i| ((i % 9) as f64 / 9.0 - 0.5) * 0.6).collect();
    let v_ctx: Vec<f64> = (0..cfg.v_context_len * cfg.vdim).map(|i| ((i % 5) as f64 / 5.0 - 0.5) * 1.5).collect();
    let a_ctx: Vec<f64> = (0..cfg.a_context_len * cfg.adim).map(|i| ((i % 3) as f64 / 3.0 - 0.5) * 1.3).collect();
    let b = make_av_flow_batch(&cfg, &v_x0, &a_x0, &v_ctx, &a_ctx, 0.4, 0.55, &v_noise, &a_noise);

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
    println!("AV model overfit: loss {first:.8} -> {last:.8} over 400 Adam steps");
    assert!(last < first * 0.01, "the loss must collapse: {first} -> {last}");
    assert!(last < 1e-3, "the loss must approach zero, got {last}");
}
