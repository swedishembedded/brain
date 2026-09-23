// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The backward pass against finite differences.
//!
//! This is the gate the whole crate rests on. A wrong gradient does not
//! announce itself: training still runs, the loss still descends (toward the
//! wrong thing), and the first symptom is a policy that will not solve -
//! which looks exactly like a problem that is too hard. Every other number
//! this crate reports is only meaningful if this passes.

use std::collections::HashMap;

use solve::config::Config;
use solve::net::{init_weights, Net};

fn tiny() -> Config {
    // Deliberately not a scaled copy of anything real: d_ff is not a
    // multiple of d_model and no two axes share a value, so an index swapped
    // between them is a shape error rather than a silent transpose.
    Config { in_dim: 9, d_model: 6, d_ff: 5, blocks: 2, moves: 4 }
}

fn fixture(cfg: &Config, rows: u32) -> (Vec<f32>, Vec<u32>) {
    let mut rng = solve::data::Rng::new(0xC0FFEE);
    let mut x = vec![0.0f32; (rows * cfg.in_dim) as usize];
    // One-hot per row, which is the shape the stem is built for.
    for r in 0..rows as usize {
        let k = rng.below(cfg.in_dim as usize);
        x[r * cfg.in_dim as usize + k] = 1.0;
    }
    let labels = (0..rows).map(|_| rng.below(cfg.moves as usize) as u32).collect();
    (x, labels)
}

fn loss_at(net: &Net, w: &HashMap<String, Vec<f32>>, x: &[f32], y: &[u32]) -> f32 {
    for (n, v) in w {
        net.set_weight(n, v);
    }
    net.zero_grads();
    net.load_batch(x, y);
    net.accumulate();
    net.loss()
}

#[test]
fn every_parameter_gradient_matches_a_finite_difference() {
    let cfg = tiny();
    let rows = 4u32;
    let (x, y) = fixture(&cfg, rows);
    let w = init_weights(&cfg, 7);
    let net = Net::from_weights(cfg.clone(), rows, &w);

    // Analytic gradients for the unperturbed weights.
    let base = loss_at(&net, &w, &x, &y);
    assert!(base.is_finite() && base > 0.0, "loss should be a positive finite number, got {base}");
    let grads: HashMap<String, Vec<f32>> =
        cfg.tensor_manifest().iter().map(|(n, _)| (n.clone(), net.grad(n))).collect();

    // A central difference over a sample of coordinates in every tensor.
    // EPS is a compromise: too small and fp32 cancellation dominates, too
    // large and the quadratic term does.
    const EPS: f32 = 2e-3;
    let mut rng = solve::data::Rng::new(99);
    let mut checked = 0usize;

    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        for _ in 0..3 {
            let k = rng.below(n);
            let mut up = w.clone();
            up.get_mut(&name).expect("tensor").as_mut_slice()[k] += EPS;
            let l_up = loss_at(&net, &up, &x, &y);

            let mut dn = w.clone();
            dn.get_mut(&name).expect("tensor").as_mut_slice()[k] -= EPS;
            let l_dn = loss_at(&net, &dn, &x, &y);

            let fd = (l_up - l_dn) / (2.0 * EPS);
            let an = grads[&name][k];
            let scale = fd.abs().max(an.abs()).max(1e-3);
            let rel = (fd - an).abs() / scale;
            assert!(
                rel < 0.08,
                "{name}[{k}]: analytic {an:.6} vs finite-difference {fd:.6} (relative {rel:.3})"
            );
            checked += 1;
        }
    }
    assert!(checked >= 3 * cfg.tensor_manifest().len(), "every tensor must be sampled");
}

/// Gradients accumulate rather than replace, so several batches can make one
/// update. Two identical batches must give twice one batch's gradient.
#[test]
fn gradients_accumulate_across_batches() {
    let cfg = tiny();
    let rows = 4u32;
    let (x, y) = fixture(&cfg, rows);
    let w = init_weights(&cfg, 11);
    let net = Net::from_weights(cfg.clone(), rows, &w);

    net.zero_grads();
    net.load_batch(&x, &y);
    net.accumulate();
    let once = net.grad("head.weight");

    net.accumulate();
    let twice = net.grad("head.weight");

    for (a, b) in once.iter().zip(&twice) {
        assert!(
            (2.0 * a - b).abs() <= 1e-4 * a.abs().max(1.0),
            "a second identical batch should double the gradient: {a} then {b}"
        );
    }
}

/// The same check at a shape that routes to the REGISTER-TILED GEMMs.
///
/// [`tiny`] is below the tile threshold, so `pick_gemm` sends every matmul in
/// it to the naive kernel - which means the fast path, the one every real
/// training run uses, would be completely unchecked by the test above. A
/// wrong thread count or a swapped argument there produces a gradient that
/// is merely incomplete: the loss still falls, and the policy still fails.
#[test]
fn the_register_tiled_gemms_agree_with_finite_differences_too() {
    // Every output here is at least one 128x128 tile.
    let cfg = Config { in_dim: 130, d_model: 128, d_ff: 192, blocks: 2, moves: 20 };
    let rows = 128u32;
    let (x, y) = fixture(&cfg, rows);
    let w = init_weights(&cfg, 3);
    let net = Net::from_weights(cfg.clone(), rows, &w);

    let base = loss_at(&net, &w, &x, &y);
    assert!(base.is_finite(), "loss must be finite, got {base}");
    let grads: HashMap<String, Vec<f32>> =
        cfg.tensor_manifest().iter().map(|(n, _)| (n.clone(), net.grad(n))).collect();

    const EPS: f32 = 3e-3;
    let mut rng = solve::data::Rng::new(4242);
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        // The stem sees a one-hot, so most of its rows get no gradient at
        // all; perturbing a dead coordinate compares 0 against 0 and proves
        // nothing. Pick a column the batch actually lit up.
        let k = if name == "stem.weight" {
            let live = x.chunks(cfg.in_dim as usize).find_map(|r| r.iter().position(|&v| v == 1.0));
            live.expect("a one-hot row") * cfg.d_model as usize + rng.below(cfg.d_model as usize)
        } else {
            rng.below(n)
        };

        let mut up = w.clone();
        up.get_mut(&name).expect("tensor").as_mut_slice()[k] += EPS;
        let l_up = loss_at(&net, &up, &x, &y);
        let mut dn = w.clone();
        dn.get_mut(&name).expect("tensor").as_mut_slice()[k] -= EPS;
        let l_dn = loss_at(&net, &dn, &x, &y);

        let fd = (l_up - l_dn) / (2.0 * EPS);
        let an = grads[&name][k];
        // Absolute AND relative, because a pure relative bound on a small
        // gradient measures fp32 rounding rather than correctness: the loss
        // is a mean over rows near 3.0, so it carries roughly 3e-7 of
        // absolute noise, and dividing that by 2*EPS puts a floor near 1e-4
        // on any central difference here however right the gradient is.
        let tol = 5e-4 + 0.10 * fd.abs().max(an.abs());
        assert!(
            (fd - an).abs() <= tol,
            "{name}[{k}]: analytic {an:.6} vs finite-difference {fd:.6} (tolerance {tol:.6})"
        );
    }
}

/// A saved policy reloads and answers identically.
///
/// Not merely "the file parses": a checkpoint that loads and then rolls out
/// differently is the failure worth testing for, and it is invisible to any
/// check that only asserts the write succeeded.
#[test]
fn a_saved_policy_reloads_and_answers_identically() {
    let cfg = tiny();
    let rows = 4u32;
    let (x, _) = fixture(&cfg, rows);
    let net = Net::from_weights(cfg.clone(), rows, &init_weights(&cfg, 21));
    let before = net.policy(&x);

    let path = std::env::temp_dir().join(format!("solve-rt-{}.safetensors", std::process::id()));
    let p = path.to_str().expect("path");
    net.save(p, &serde_json::json!({"steps": 3, "depth": 5})).expect("save");

    let back = Net::load(p, rows).expect("a saved policy must load");
    assert_eq!(back.cfg, cfg, "the architecture must survive the round trip");
    let after = back.policy(&x);

    assert_eq!(after.len(), before.len());
    for (i, (a, b)) in after.iter().zip(&before).enumerate() {
        assert!((a - b).abs() < 1e-6, "probability {i} moved across the save: {b} -> {a}");
    }
    let _ = std::fs::remove_file(&path);
}
