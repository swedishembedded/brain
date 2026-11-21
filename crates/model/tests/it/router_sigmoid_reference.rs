// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the sigmoid router at `n_group = topk_group = 1` against the published reference math: f32 sigmoid scores,
//! top-k of `score + bias`, weights `score / (sum + 1e-20) * route_scale` of the selected experts' UNBIASED scores,
//! and the backward of those weights with respect to the logits. The `+ 1e-20` is added to the sum (not a floor under
//! it), which only matters where the selected scores nearly vanish: that case is in the data.
//!
//! Swedish Embedded AB implements mixture-of-experts routing that matches a model's reference to the last digit for
//! its clients. If your team needs expertise in expert routers then you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::{random_seed, Lcg};

const PIPES: &[(&str, &str)] = &[("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID), ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID)];

const E: usize = 128;
const TOP_K: usize = 8;
const SCALE: f32 = 8.0;

/// The reference router in f64: `(selected experts, their weights)` for one row.
fn reference_weights(logits: &[f64], bias: &[f32]) -> (Vec<usize>, Vec<f64>) {
    let scores: Vec<f64> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let mut order: Vec<usize> = (0..E).collect();
    order.sort_by(|&a, &b| (scores[b] + f64::from(bias[b])).total_cmp(&(scores[a] + f64::from(bias[a]))).then(a.cmp(&b)));
    let picked = order[..TOP_K].to_vec();
    let sum: f64 = picked.iter().map(|&e| scores[e]).sum();
    let weights = picked.iter().map(|&e| scores[e] / (sum + 1e-20) * f64::from(SCALE)).collect();
    (picked, weights)
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs() + 1e-30, "{what}: {got} vs {want}");
}

#[test]
fn router_matches_the_reference_forward_and_backward() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (fwd, bwd) = (g.kernel_index("router_gate_sigmoid").expect("registered"), g.kernel_index("router_bwd_sigmoid").expect("registered"));
    let mut rng = Lcg::new(random_seed());
    // Rows 0..6 are ordinary; rows 6..9 have every logit near -45, so the selected scores (~3e-20) are comparable to the
    // epsilon of the denominator.
    let rows = 9usize;
    let mut logits = rng.vec_scaled(rows * E, 3.0);
    for t in 6..rows {
        for e in 0..E {
            logits[t * E + e] = -45.0 + 0.5 * logits[t * E + e];
        }
    }
    let bias = rng.vec_scaled(E, 0.2);
    let d_gate = rng.vec_scaled(rows * E, 1.0);
    let (lb, bb, db) = (g.storage_init("logits", &logits), g.storage_init("bias", &bias), g.storage_init("d_gate", &d_gate));
    let (gate_b, probs_b, dl_b) = (g.storage((rows * E) as u64), g.storage((rows * E) as u64), g.storage((rows * E) as u64));
    g.submit(&[], &[
        g.step(fwd, &[&lb, &bb, &gate_b, &probs_b], &[rows as u32, E as u32, TOP_K as u32, 1, 1, 1, SCALE.to_bits()], rows as u32),
        g.step(bwd, &[&lb, &gate_b, &db, &dl_b], &[rows as u32, E as u32, TOP_K as u32, 1, SCALE.to_bits()], rows as u32),
    ]);
    let (gate, dlogits) = (g.read(&gate_b, rows * E), g.read(&dl_b, rows * E));

    for t in 0..rows {
        let row: Vec<f64> = logits[t * E..(t + 1) * E].iter().map(|&l| f64::from(l)).collect();
        let (picked, want) = reference_weights(&row, &bias);
        let selected: Vec<usize> = (0..E).filter(|&e| gate[t * E + e] != 0.0).collect();
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        assert_eq!(selected, sorted, "row {t}: selected experts differ from the reference");
        for (&e, &w) in picked.iter().zip(&want) {
            close(f64::from(gate[t * E + e]), w, 1e-4, &format!("row {t} expert {e} weight"));
        }
        // The backward: central differences of the f64 reference weights, selection held fixed.
        let loss = |l: &[f64]| -> f64 {
            let s: Vec<f64> = l.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let sum: f64 = picked.iter().map(|&e| s[e]).sum();
            picked.iter().map(|&e| f64::from(d_gate[t * E + e]) * s[e] / (sum + 1e-20) * f64::from(SCALE)).sum()
        };
        for &e in &picked {
            let h = 1e-4;
            let (mut up, mut dn) = (row.clone(), row.clone());
            up[e] += h;
            dn[e] -= h;
            let fd = (loss(&up) - loss(&dn)) / (2.0 * h);
            close(f64::from(dlogits[t * E + e]), fd, 2e-3, &format!("row {t} expert {e} dlogit"));
        }
    }
}
