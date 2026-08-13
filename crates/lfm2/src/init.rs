// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM weight initialization (deterministic for a fixed seed) — for
//! from-scratch training and the gradient check. Mirrors the qwen recipe:
//! - RMSNorm gains: 1.0
//! - residual-output projections (`attn.wo`, `conv.out_proj`, `mlp.down`):
//!   Normal(0, 0.02/sqrt(2*L))
//! - everything else (linears, depthwise conv taps, embedding): Normal(0, 0.02)

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::LfmConfig;

fn is_norm_gain(name: &str) -> bool {
    name == "norm.weight"
        || name.ends_with("ln1.weight")
        || name.ends_with("ln2.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight")
}

fn is_residual_proj(name: &str) -> bool {
    name.ends_with("attn.wo.weight") || name.ends_with("conv.out_proj.weight") || name.ends_with("mlp.down.weight")
}

pub fn init_weights(cfg: &LfmConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let std = 0.02f32;
    let proj_std = 0.02f32 / ((2.0 * cfg.n_layers() as f32).sqrt());
    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> {
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };

    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if is_norm_gain(&name) {
            vec![1.0; numel]
        } else if is_residual_proj(&name) {
            normal(numel, proj_std, &mut rng)
        } else {
            normal(numel, std, &mut rng)
        };
        w.insert(name, v);
    }
    w
}
