// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight initialization for the YOLO conv blocks / head.
//!
//! Deterministic for a fixed `seed` (the FD gradient check relies on
//! reproducible inits). Conv weights get a small-std Gaussian; BN affine
//! `gamma`=1 (jittered for the gradient check so it is not a degenerate point),
//! `beta`=0; BN running stats start at `run_mean`=0, `run_var`=1. The running
//! stats carry no train-mode gradient, so their init only matters for eval.

use std::collections::HashMap;

use data::rng::Rng;

/// Initialize every parameter in `params` (`(name, numel)`), deterministic for
/// `seed`. `std` scales the conv-weight Gaussian.
pub fn init_params(params: &[(String, usize)], seed: u64, std: f32) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, numel) in params {
        let v: Vec<f32> = if name.ends_with("bn.gamma") {
            // gamma ~ 1 with small jitter (a nonzero, non-degenerate affine scale).
            (0..*numel).map(|_| 1.0 + 0.1 * rng.next_gaussian() as f32).collect()
        } else if name.ends_with("bn.beta") {
            (0..*numel).map(|_| 0.05 * rng.next_gaussian() as f32).collect()
        } else if name.ends_with("bn.run_mean") {
            vec![0.0; *numel]
        } else if name.ends_with("bn.run_var") {
            vec![1.0; *numel]
        } else if name.ends_with("cls.2.bias") {
            // Head class-bias prior (Ultralytics `Detect.bias_init`): start the
            // class logits at a low prior probability p≈0.01 so early training is
            // not swamped by the background-class imbalance. b = -log((1-p)/p).
            let p = 0.01f32;
            let b = -((1.0 - p) / p).ln();
            vec![b; *numel]
        } else if name.ends_with("reg.2.bias") {
            // Box/DFL distribution bias: zero (no prior on the box side).
            vec![0.0; *numel]
        } else {
            // conv / final-1x1 weights.
            (0..*numel).map(|_| std * rng.next_gaussian() as f32).collect()
        };
        w.insert(name.clone(), v);
    }
    w
}

/// Initialize the whole model's parameters (backbone + neck + head) from the
/// config's full `param_list`, deterministic for `seed`. Conv weights get a
/// small-std Kaiming-ish normal; BN affine `gamma`≈1 / `beta`=0; running stats
/// `run_mean`=0 / `run_var`=1 (see [`init_params`]).
pub fn init_model(cfg: &crate::YoloConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let params = <crate::YoloConfig as model::ModelConfig>::param_list(cfg);
    // A small std keeps the deep conv stack well-conditioned for the FD check
    // (larger weights drive SiLU/BN into curved, ill-conditioned regions, where
    // the central difference over the deep cascade disagrees with the analytic
    // directional derivative).
    init_params(&params, seed, 0.1)
}
