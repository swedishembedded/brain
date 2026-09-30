// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! AdamW's β1, β2 and ε are the caller's: two steps with non-default values
//! move a parameter exactly as a host AdamW with those values does. (At the
//! first step the moments' bias corrections cancel the betas out, so a
//! single step could not tell them apart.)

use qwen3::{Qwen, QwenConfig};

/// `torch.optim.AdamW`, one parameter, written out independently.
fn host_adamw(w: &mut [f32], grads: &[Vec<f32>], lr: f32, wd: f32, (b1, b2, eps): (f32, f32, f32)) {
    let (mut m, mut v) = (vec![0f32; w.len()], vec![0f32; w.len()]);
    for (t, g) in grads.iter().enumerate() {
        let t = t as i32 + 1;
        for i in 0..w.len() {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            let (mhat, vhat) = (m[i] / (1.0 - b1.powi(t)), v[i] / (1.0 - b2.powi(t)));
            w[i] -= lr * wd * w[i];
            w[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
    }
}

#[test]
fn betas_and_eps_are_honoured() {
    let cfg = QwenConfig::tiny();
    let model = Qwen::new(cfg.clone(), 1, 4, &qwen3::init_weights(&cfg, 7));
    let adam = optim::Adam { beta1: 0.8, beta2: 0.95, eps: 1e-3 };
    let (lr, wd, name) = (0.05f32, 0.01f32, "norm.weight");

    let start = model.read_weight(name);
    let mut want = start.clone();
    let n = want.len();
    let grads: Vec<Vec<f32>> = vec![(0..n).map(|i| 0.5 - i as f32 * 0.01).collect(), (0..n).map(|i| -0.2 + i as f32 * 0.003).collect()];
    host_adamw(&mut want, &grads, lr, wd, (adam.beta1, adam.beta2, adam.eps));

    for (t, g) in grads.iter().enumerate() {
        model.zero_grads();
        model.write_grad(name, g);
        model.adamw_step(t as u32 + 1, lr, wd, adam, None, 1.0);
    }
    let got = model.read_weight(name);
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst < 1e-6, "AdamW with β1 0.8, β2 0.95, ε 1e-3 is off the host reference by {worst}");

    // The same steps at the defaults land somewhere else: the values reach the update.
    let mut default_path = start;
    host_adamw(&mut default_path, &grads, lr, wd, (0.9, 0.999, 1e-8));
    assert!(default_path.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-4));
}
