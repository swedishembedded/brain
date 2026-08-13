// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoencoder weight initialization:
//! - linear weights: Normal(0, `std`) with a fan-in-scaled `std` (Kaiming-ish),
//!   so the four matmuls keep activations in range across the bottleneck.
//! - all biases = 0.
//!
//! Deterministic for a fixed `seed` (the FD gradient check + benchmark both rely
//! on reproducible inits).

use std::collections::HashMap;

use data::rng::Rng;

use crate::model::AutoencoderConfig;

/// Build an initial weight map for `cfg`, deterministic for a fixed `seed`.
pub fn init_weights(cfg: &AutoencoderConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();

    // (name, numel, fan_in) for each linear; std = 1/sqrt(fan_in).
    let normal = |n: usize, fan_in: usize, rng: &mut Rng| -> Vec<f32> {
        let s = (1.0f32 / fan_in.max(1) as f32).sqrt();
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };

    for (name, numel) in cfg.param_list() {
        let v = if name.ends_with(".bias") {
            vec![0.0; numel]
        } else {
            // fan_in is the second matmul dimension (K) for `out = x @ W^T`,
            // recoverable from numel / output-rows; we encode it per-tensor.
            let fan_in = cfg.fan_in_of(&name);
            normal(numel, fan_in, &mut rng)
        };
        w.insert(name, v);
    }
    w
}
