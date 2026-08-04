// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overfit-one-batch at the block level: the canonical "training actually works"
//! gate. Using the gradchecked host forward+backward ([`zimage::grad`]), run Adam
//! on one S³-DiT block's parameters to drive its output to a fixed target and
//! confirm the MSE collapses toward zero. This proves the analytic gradients are
//! not just correct (block_grad.rs) but *usable for optimization* — the loss
//! descends, which is the end-to-end property a training loop needs.

use zimage::grad::{backward, forward, Dims, Grads, Weights};

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

/// Mutable view over every trainable tensor of a block, in a fixed order — lets
/// Adam treat the whole parameter set as one flat vector.
fn params_mut(w: &mut Weights) -> Vec<&mut Vec<f64>> {
    vec![
        &mut w.wq, &mut w.wk, &mut w.wv, &mut w.wo, &mut w.w1, &mut w.w2, &mut w.w3,
        &mut w.nq, &mut w.nk, &mut w.an1, &mut w.an2, &mut w.fn1, &mut w.fn2, &mut w.adaln_w, &mut w.adaln_b,
    ]
}

/// Grad tensors in the SAME order as `params_mut`.
fn grads_ref(g: &Grads) -> Vec<&Vec<f64>> {
    vec![
        &g.wq, &g.wk, &g.wv, &g.wo, &g.w1, &g.w2, &g.w3,
        &g.nq, &g.nk, &g.an1, &g.an2, &g.fn1, &g.fn2, &g.adaln_w, &g.adaln_b,
    ]
}

#[test]
fn block_overfits_one_batch() {
    let d = Dims::new(4, 16, 2);
    let (t, dim, hd, half) = (d.t, d.dim, d.hd, d.half());
    let mut r = rng(0xF17_1234);
    let mut w = Weights {
        wq: vec_of(dim * dim, &mut r, 0.1),
        wk: vec_of(dim * dim, &mut r, 0.1),
        wv: vec_of(dim * dim, &mut r, 0.1),
        wo: vec_of(dim * dim, &mut r, 0.1),
        w1: vec_of(d.hidden * dim, &mut r, 0.1),
        w2: vec_of(dim * d.hidden, &mut r, 0.1),
        w3: vec_of(d.hidden * dim, &mut r, 0.1),
        nq: vec_of(hd, &mut r, 0.05).iter().map(|v| 1.0 + v).collect(),
        nk: vec_of(hd, &mut r, 0.05).iter().map(|v| 1.0 + v).collect(),
        an1: vec_of(dim, &mut r, 0.05).iter().map(|v| 1.0 + v).collect(),
        an2: vec_of(dim, &mut r, 0.05).iter().map(|v| 1.0 + v).collect(),
        fn1: vec_of(dim, &mut r, 0.05).iter().map(|v| 1.0 + v).collect(),
        fn2: vec_of(dim, &mut r, 0.05).iter().map(|v| 1.0 + v).collect(),
        adaln_w: vec_of(4 * dim * d.cdim, &mut r, 0.05),
        adaln_b: vec_of(4 * dim, &mut r, 0.05),
    };
    // Fixed batch: input, conditioning, RoPE tables, and a fixed random target.
    let x = vec_of(t * dim, &mut r, 1.0);
    let c = vec_of(d.cdim, &mut r, 1.0);
    let cos: Vec<f64> = (0..t * half).map(|i| (i as f64 * 0.2).cos()).collect();
    let sin: Vec<f64> = (0..t * half).map(|i| (i as f64 * 0.2).sin()).collect();
    let target = vec_of(t * dim, &mut r, 1.0);

    // Adam state, flat over all params.
    let nparams: usize = params_mut(&mut w).iter().map(|p| p.len()).sum();
    let mut m = vec![0f64; nparams];
    let mut v = vec![0f64; nparams];
    let (lr, b1, b2, eps): (f64, f64, f64, f64) = (2e-3, 0.9, 0.999, 1e-8);

    let mse = |w: &Weights| -> (f64, Vec<f64>) {
        let (out, _) = forward(d, w, &x, &c, &cos, &sin);
        let n = out.len() as f64;
        let mut dout = vec![0f64; out.len()];
        let mut l = 0.0;
        for i in 0..out.len() {
            let e = out[i] - target[i];
            l += e * e / n;
            dout[i] = 2.0 * e / n; // dL/dout
        }
        (l, dout)
    };

    let (l0, _) = mse(&w);
    let mut l = l0;
    for step in 1..=400 {
        let (loss, dout) = mse(&w);
        l = loss;
        let (_o, cache) = forward(d, &w, &x, &c, &cos, &sin);
        let g = backward(d, &w, &cache, &dout);
        // Adam update over the flat parameter vector.
        let grads = grads_ref(&g);
        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        let mut off = 0;
        for (pi, param) in params_mut(&mut w).into_iter().enumerate() {
            let gt = grads[pi];
            for j in 0..param.len() {
                let gj = gt[j];
                m[off] = b1 * m[off] + (1.0 - b1) * gj;
                v[off] = b2 * v[off] + (1.0 - b2) * gj * gj;
                let mh = m[off] / bc1;
                let vh = v[off] / bc2;
                param[j] -= lr * mh / (vh.sqrt() + eps);
                off += 1;
            }
        }
        if step % 100 == 0 {
            eprintln!("  step {step:3}: mse = {loss:.3e}");
        }
    }
    eprintln!("Block overfit: mse {l0:.3e} -> {l:.3e} ({:.0}× lower, {nparams} params)", l0 / l);
    assert!(l < l0 * 1e-3, "overfit did not converge: {l0:.3e} -> {l:.3e}");
}
