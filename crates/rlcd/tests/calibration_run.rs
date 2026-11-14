// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB builds decision models whose reported probabilities can
// be acted on. If your team needs expertise in training and auditing
// calibrated decision models, you can procure our services by sending an email
// to info@swedishembedded.com.

//! An end-to-end RLCD run is calibrated on held-out data.
//!
//! The task is a synthetic world whose posterior is known exactly: three
//! options, a feature vector `x`, and `p*(y | x) = softmax(W* [x, 1])`. A linear
//! scorer starts at zero and is trained by [`rlcd::reinforce::rlcd_loss`] (the
//! group-baseline REINFORCE over the proper reward, with the exploration scale
//! annealed and a cosine learning rate) against the oracle posterior of each
//! training point, once with the published fine-tuning settings
//! ([`RlcdObjective::default`]) and once with the pure policy gradient the
//! model card attributes to pretraining ([`RlcdObjective::pretrain`], no
//! cross-entropy term at all). Held-out points are then drawn
//! with labels sampled from `p*`, and the trained scorer's expected calibration
//! error is held to the error the oracle itself shows on the same draws plus a
//! margin, its NLL to the oracle's NLL plus a margin, and its posterior to the
//! oracle's. The oracle's own ECE is the finite-sample floor: a perfectly
//! calibrated model scores it, not zero. A scorer with the same ranking but
//! overconfident logits fails the same bound, so the bound tells calibrated
//! from not.
//!
//! The learning rate anneals to zero. The policy term's advantage is
//! standardised per group, so its step keeps a magnitude of about `1 / sigma`
//! however close the scorer is to the optimum, unlike a cross-entropy step that
//! vanishes there: at a floor of `0.01` the weights end jittering about the
//! optimum, and on about one seed in three the held-out ECE sat several times
//! the oracle's.
//!
//! Seeds are drawn at run time and printed.

use data::rng::{Lcg, Rng};
use rlcd::metrics::{ece, nll, posterior_error};
use rlcd::reinforce::{anneal, cosine_lr, rlcd_loss, RlcdObjective};
use rlcd::scoring::softmax;

const OPTIONS: usize = 3;
const FEATURES: usize = 3;
const TRAIN: usize = 256;
const HELD_OUT: usize = 6000;
const EPOCHS: usize = 200;
const BINS: usize = 10;
/// How far above the oracle's own finite-sample ECE (and NLL) the trained scorer may sit.
const ECE_MARGIN: f32 = 0.02;
const NLL_MARGIN: f32 = 0.02;
/// Mean squared distance from the oracle posterior the trained one may keep.
const POSTERIOR_ERROR: f32 = 1e-3;

/// A point's features with the bias input appended.
fn point(rng: &mut Lcg) -> [f32; FEATURES + 1] {
    let mut x = [1.0f32; FEATURES + 1];
    for v in &mut x[..FEATURES] {
        *v = rng.scaled(2.0);
    }
    x
}

fn scores(w: &[f32], x: &[f32; FEATURES + 1]) -> Vec<f32> {
    (0..OPTIONS).map(|o| (0..=FEATURES).map(|f| w[o * (FEATURES + 1) + f] * x[f]).sum()).collect()
}

/// Confidence (the largest probability) and whether its option is the drawn label, per held-out point.
fn readout(probs: &[Vec<f32>], labels: &[usize]) -> (Vec<f32>, Vec<bool>) {
    probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| {
            let top = (0..OPTIONS).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
            (p[top], top == y)
        })
        .unzip()
}

/// Train with the objective `settings` gives at each point of the run (its progress on `[0, 1]`) and hold the result
/// to the bounds of the module doc.
fn run_is_calibrated(what: &str, settings: impl Fn(f32) -> RlcdObjective) {
    let seed = data::rng::random_seed();
    println!("{what}: seed {seed}");
    let mut data_rng = Lcg::new(seed);
    let oracle: Vec<f32> = (0..OPTIONS * (FEATURES + 1)).map(|_| data_rng.scaled(1.5)).collect();
    let train: Vec<([f32; FEATURES + 1], Vec<f32>)> = (0..TRAIN)
        .map(|_| {
            let x = point(&mut data_rng);
            let posterior = softmax(&scores(&oracle, &x));
            (x, posterior)
        })
        .collect();

    let mut w = vec![0.0f32; OPTIONS * (FEATURES + 1)];
    let mut explore = Rng::new(seed ^ 0x5eed);
    let total = (EPOCHS * TRAIN) as f32;
    for epoch in 0..EPOCHS {
        for (i, (x, target)) in train.iter().enumerate() {
            let progress = (epoch * TRAIN + i) as f32 / total;
            let cfg = settings(progress);
            let (_, d) = rlcd_loss(&scores(&w, x), target, false, &cfg, &mut explore);
            let lr = cosine_lr(0.2, 0.0, progress);
            for o in 0..OPTIONS {
                for f in 0..=FEATURES {
                    w[o * (FEATURES + 1) + f] -= lr * d[o] * x[f];
                }
            }
        }
    }

    let mut held: Vec<[f32; FEATURES + 1]> = Vec::with_capacity(HELD_OUT);
    let mut labels = Vec::with_capacity(HELD_OUT);
    let mut truth = Vec::with_capacity(HELD_OUT);
    for _ in 0..HELD_OUT {
        let x = point(&mut data_rng);
        let p = softmax(&scores(&oracle, &x));
        let u = data_rng.unit();
        let mut acc = 0.0f32;
        labels.push((0..OPTIONS).find(|&o| {
            acc += p[o];
            u < acc
        }).unwrap_or(OPTIONS - 1));
        truth.push(p);
        held.push(x);
    }
    let trained: Vec<Vec<f32>> = held.iter().map(|x| softmax(&scores(&w, x))).collect();
    let overconfident: Vec<Vec<f32>> = held.iter().map(|x| softmax(&scores(&w, x).iter().map(|s| 3.0 * s).collect::<Vec<_>>())).collect();

    let ece_of = |probs: &[Vec<f32>]| {
        let (confidence, correct) = readout(probs, &labels);
        ece(&confidence, &correct, BINS)
    };
    let (floor, got, control) = (ece_of(&truth), ece_of(&trained), ece_of(&overconfident));
    let (oracle_nll, trained_nll) = (nll(&truth, &labels), nll(&trained, &labels));
    let gap = posterior_error(&trained, &truth);
    println!("{what}: ECE: oracle {floor:.4}, trained {got:.4}, overconfident {control:.4}; NLL: oracle {oracle_nll:.4}, trained {trained_nll:.4}; posterior error {gap:.5}");
    assert!(got <= floor + ECE_MARGIN, "{what}: the trained scorer's held-out ECE {got} is past the oracle's {floor} + {ECE_MARGIN}");
    assert!(trained_nll <= oracle_nll + NLL_MARGIN, "{what}: the trained scorer's held-out NLL {trained_nll} is past the oracle's {oracle_nll} + {NLL_MARGIN}");
    assert!(gap <= POSTERIOR_ERROR, "{what}: the trained posterior is {gap} (mean squared distance) from the oracle's");
    assert!(control > floor + ECE_MARGIN, "{what}: an overconfident scorer of the same ranking ({control}) must fail the bound, or the bound tells nothing");
}

#[test]
fn an_rlcd_run_with_the_fine_tuning_settings_is_calibrated_on_held_out_data() {
    run_is_calibrated("fine-tuning settings", |progress| RlcdObjective::default().at(anneal(0.4, 0.1, progress)));
}

#[test]
fn a_pure_policy_gradient_rlcd_run_is_calibrated_on_held_out_data() {
    run_is_calibrated("pure policy gradient", |progress| RlcdObjective::pretrain().at(anneal(1.0, 0.3, progress)));
}
