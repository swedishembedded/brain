// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Central finite differences of a training-graph function in `f64`, the oracle every backward is held to.

use data::rng::Lcg;
use mirage::train::tape::{Id, Tape};

/// Step of the central difference.
const STEP: f64 = 1e-6;

/// Deterministic values in `[-scale, scale]`.
pub fn noise(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut rng = Lcg::new(seed); // seed: fixed filler, a check compares two derivatives of whatever it holds
    (0..n).map(|_| f64::from(rng.signed()) * scale).collect()
}

/// Largest deviation of the analytic gradient from the central difference, relative to the largest gradient entry,
/// over every input of `build` (which maps leaves to a vector output; the loss is a fixed random projection of it).
pub fn max_error(inputs: &[Vec<f64>], build: &dyn Fn(&mut Tape<f64>, &[Id]) -> Id) -> f64 {
    let loss_of = |ins: &[Vec<f64>]| -> (f64, Vec<Vec<f64>>) {
        let mut tape = Tape::<f64>::new();
        let ids: Vec<Id> = ins.iter().map(|v| tape.leaf(v.clone())).collect();
        let y = build(&mut tape, &ids);
        let proj = tape.leaf(noise(tape.len(y), 0xfd, 1.0));
        let weighted = tape.mul(y, proj);
        let loss = tape.sum_weighted(weighted, 1.0);
        let grads = tape.backward(loss);
        let g = ids.iter().zip(ins).map(|(id, v)| grads.get(*id).map_or_else(|| vec![0.0; v.len()], <[f64]>::to_vec)).collect();
        (tape.val(loss)[0], g)
    };
    let (_, analytic) = loss_of(inputs);
    let mut worst = 0f64;
    for (k, input) in inputs.iter().enumerate() {
        let mut fd = vec![0f64; input.len()];
        for i in 0..input.len() {
            let mut plus = inputs.to_vec();
            plus[k][i] += STEP;
            let mut minus = inputs.to_vec();
            minus[k][i] -= STEP;
            fd[i] = (loss_of(&plus).0 - loss_of(&minus).0) / (2.0 * STEP);
        }
        let scale = fd.iter().chain(&analytic[k]).fold(0f64, |m, v| m.max(v.abs())).max(1e-12);
        let err = fd.iter().zip(&analytic[k]).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max) / scale;
        worst = worst.max(err);
    }
    worst
}

/// Assert [`max_error`] stays under the block tolerance.
pub fn check(name: &str, inputs: &[Vec<f64>], build: &dyn Fn(&mut Tape<f64>, &[Id]) -> Id) {
    let err = max_error(inputs, build);
    assert!(err < 1e-4, "{name}: backward deviates from finite differences by {err:e}");
}
