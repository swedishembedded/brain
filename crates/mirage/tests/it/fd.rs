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
    eprintln!("fd {name}: {err:.2e}");
    assert!(err < 1e-4, "{name}: backward deviates from finite differences by {err:e}");
}

/// Outcome of [`model_check`].
pub struct ModelCheck {
    /// Worst relative error of a tensor's directional derivative.
    pub worst: f64,
    /// The tensor it occurred at.
    pub worst_name: String,
    /// Tensors with a vanishing gradient the finite difference cannot resolve.
    pub unresolvable: Vec<String>,
    /// Tensors that received no gradient at all.
    pub unreached: Vec<String>,
}

/// Directional finite-difference check of a scalar graph over every non-training-only tensor: one random
/// direction per tensor, Richardson-extrapolated central differences (steps `h` and `h/2`, so rounding stays at
/// the `h/2` level and truncation drops to O(h^4)). `loss` builds the graph and returns the scalar.
pub fn model_check(cfg: &mirage::config::MirageConfig, params: &mirage::train::net::Params<f64>, loss: &dyn Fn(&mut mirage::train::net::Net<'_, f64>) -> Id) -> ModelCheck {
    use mirage::train::net::Net;
    let eval = |p: &mirage::train::net::Params<f64>| {
        let mut net = Net::<f64>::new(cfg, p, None);
        let l = loss(&mut net);
        let value = net.tape.val(l)[0];
        let mut g = net.tape.backward(l);
        (value, net.param_grads(&mut g))
    };
    let (loss_value, grads) = eval(params);
    let mut out = ModelCheck { worst: 0.0, worst_name: String::new(), unresolvable: Vec::new(), unreached: Vec::new() };
    for spec in mirage::spec::Manifest::new(cfg).iter().filter(|s| !s.training_only) {
        let Some(g) = grads.get(&spec.name) else {
            out.unreached.push(spec.name.clone());
            continue;
        };
        if g.iter().all(|v| *v == 0.0) {
            out.unreached.push(spec.name.clone());
            continue;
        }
        let dir = noise(g.len(), 900 + spec.numel() as u64, 1.0);
        let analytic: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let shifted = |sign: f64, step: f64| {
            let mut p = params.clone();
            p.get_mut(&spec.name).unwrap().iter_mut().zip(&dir).for_each(|(v, d)| *v += sign * step * d);
            eval(&p).0
        };
        let h = 2e-4;
        let central = |step: f64| (shifted(1.0, step) - shifted(-1.0, step)) / (2.0 * step);
        let fd = (4.0 * central(h / 2.0) - central(h)) / 3.0;
        // Rounding of the loss itself limits a difference quotient to about |loss| * eps / h; a derivative within a
        // factor 50 of that cannot be resolved, and larger ones are judged net of it.
        let noise = 8.0 * loss_value.abs() * f64::EPSILON / h;
        let size = analytic.abs().max(fd.abs());
        if size < 50.0 * noise.max(1e-12) {
            out.unresolvable.push(spec.name.clone());
            continue;
        }
        let err = ((analytic - fd).abs() - noise).max(0.0) / size;
        if err > 1e-4 {
            eprintln!("  {}: analytic {analytic:.6e} fd {fd:.6e} err {err:.2e}", spec.name);
        }
        if err > out.worst {
            out.worst = err;
            out.worst_name = spec.name.clone();
        }
    }
    out
}
