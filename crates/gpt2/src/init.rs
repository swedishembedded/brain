// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPT weight initialization, matching nanogpt's `_init_weights`:
//! - embeddings & linear weights: Normal(0, 0.02)
//! - residual projections (`attn.out.weight`, `mlp.proj.weight`): Normal(0,
//!   0.02 / sqrt(2 * n_layers)) — the GPT-2 scaled init
//! - LayerNorm weight = 1, all biases = 0
//! - `lm_head.weight`: Normal(0, 0.02) (untied — see `model.rs`).

use std::collections::HashMap;

use data::rng::Rng;

use crate::model::GptConfig;

/// Build an initial weight map for `cfg`, deterministic for a fixed `seed`.
pub fn init_weights(cfg: &GptConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    let std = 0.02f32;
    let proj_std = 0.02f32 / ((2.0 * cfg.n_layers as f32).sqrt());

    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> {
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };

    for (name, numel) in cfg.param_list() {
        let v = if name.ends_with("ln1.weight")
            || name.ends_with("ln2.weight")
            || name == "ln.weight"
        {
            vec![1.0; numel] // LayerNorm gain = 1
        } else if name.ends_with(".bias") {
            vec![0.0; numel] // all biases = 0
        } else if name.ends_with("attn.out.weight") || name.ends_with("mlp.proj.weight") {
            normal(numel, proj_std, &mut rng) // GPT-2 scaled residual-projection init
        } else {
            normal(numel, std, &mut rng) // embeddings + other linear weights
        };
        w.insert(name, v);
    }
    w
}
