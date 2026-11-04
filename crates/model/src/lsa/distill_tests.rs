// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::distill::*;
use super::indexer::{IndexGeometry, Indexer};
use super::*;
use data::rng::Lcg;

fn cfg() -> LsaConfig {
    LsaConfig { index_dim: 4, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 16, top_p: 4, top_k: 8, sink: 2, window: 16, early: None }
}

fn geometry() -> IndexGeometry {
    IndexGeometry { n_layers: 1, n_heads: 2, n_kv_heads: 1, head_dim: 8, rope_base: 0.0 }
}

/// A document whose teacher is what a hidden projection `truth` would score:
/// learnable exactly by the indexer.
fn sample(rng: &mut Lcg, truth: &[f32], t: usize) -> LayerSample {
    let (din, d) = (8usize, 4usize);
    let keys: Vec<f32> = (0..t * din).map(|_| rng.signed()).collect();
    let positions: Vec<u32> = (0..6).map(|i| (t - 1 - 3 * i) as u32).collect();
    let queries: Vec<f32> = (0..positions.len() * din).map(|_| rng.signed()).collect();
    let unit = |x: &[f32]| -> Vec<f64> {
        let v: Vec<f64> = (0..d).map(|c| (0..din).map(|i| x[i] as f64 * truth[i * d + c] as f64).sum()).collect();
        let n = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        v.into_iter().map(|a| a / n).collect()
    };
    let c = cfg();
    let targets = positions
        .iter()
        .enumerate()
        .map(|(s, &p)| {
            let q = unit(&queries[s * din..][..din]);
            let hi = p as usize + 1 - c.window;
            let z: Vec<f64> = (c.sink..hi).map(|j| q.iter().zip(unit(&keys[j * din..][..din])).map(|(a, b)| a * b).sum::<f64>() / 0.1).collect();
            let m = z.iter().copied().fold(f64::MIN, f64::max);
            let total: f64 = z.iter().map(|v| (v - m).exp()).sum();
            (c.sink..hi).zip(&z).map(|(j, v)| (j as u32, ((v - m).exp() / total) as f32)).collect()
        })
        .collect();
    LayerSample { keys, queries, positions, targets }
}

/// The analytic gradients of W_Q, W_K and ln tau agree with central finite
/// differences of the KL loss.
#[test]
fn distillation_gradients_match_finite_differences() {
    let mut rng = Lcg::random();
    let truth: Vec<f32> = (0..32).map(|_| rng.signed()).collect();
    let s = sample(&mut rng, &truth, 60);
    let init = Indexer::random(cfg(), geometry(), rng.next_u64()).unwrap();
    let d = LayerDistiller::new(init.layers[0].as_ref().unwrap(), 8, &cfg(), 0.3).unwrap();
    let (stats, gq, gk, gt) = d.loss_and_grads(&s);
    assert!(stats.queries > 0 && stats.kl > 0.0);
    let loss = |d: &LayerDistiller| d.loss_and_grads(&s).0.kl;
    let h = 1e-3f32;
    let fd = |f: &dyn Fn(&mut LayerDistiller, f32)| {
        let (mut up, mut down) = (d.clone(), d.clone());
        f(&mut up, h);
        f(&mut down, -h);
        (loss(&up) - loss(&down)) / (2.0 * h as f64)
    };
    for i in [0usize, 5, 13, 31] {
        let num = fd(&|m, e| m.wq[i] += e);
        assert!((num - gq[i] as f64).abs() < 2e-3 * (1.0 + num.abs()), "dW_Q[{i}]: analytic {} vs numeric {num}", gq[i]);
        let num = fd(&|m, e| m.wk[i] += e);
        assert!((num - gk[i] as f64).abs() < 2e-3 * (1.0 + num.abs()), "dW_K[{i}]: analytic {} vs numeric {num}", gk[i]);
    }
    let num = fd(&|m, e| m.log_tau += e);
    assert!((num - gt).abs() < 2e-3 * (1.0 + num.abs()), "d ln tau: analytic {gt} vs numeric {num}");
}

/// Trained on documents whose teacher a projection can express exactly, the
/// KL falls far below where it started on held-out documents.
#[test]
fn distillation_learns_a_learnable_teacher() {
    let mut rng = Lcg::random();
    let truth: Vec<f32> = (0..32).map(|_| rng.signed()).collect();
    let train: Vec<LayerSample> = (0..32).map(|_| sample(&mut rng, &truth, 70)).collect();
    let held: Vec<LayerSample> = (0..8).map(|_| sample(&mut rng, &truth, 70)).collect();
    let init = Indexer::random(cfg(), geometry(), rng.next_u64()).unwrap();
    let mut distiller = Distiller::new(&init, 0.5).unwrap();
    let held_kl = |d: &Distiller| -> f64 {
        let ix = d.indexer();
        let ld = LayerDistiller::new(ix.layers[0].as_ref().unwrap(), 8, &cfg(), d.temperatures()[0].unwrap()).unwrap();
        held.iter().map(|s| ld.loss_and_grads(s).0.kl).sum::<f64>() / held.len() as f64
    };
    let before = held_kl(&distiller);
    for epoch in 0..40 {
        for s in &train {
            distiller.step(&[Some(s.clone())], if epoch < 30 { 2e-2 } else { 5e-3 });
        }
    }
    let after = held_kl(&distiller);
    println!("held-out KL {before:.4} -> {after:.4}, tau {:?}", distiller.temperatures());
    assert!(after < 0.25 * before, "held-out KL only fell from {before} to {after}");
}
