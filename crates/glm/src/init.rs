// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM weight initialization (deterministic for a fixed seed):
//! - RMSNorm gains (`*norm.weight`, `*_ln.weight`): 1.0
//! - router selection bias (`moe.router.bias`): 0.0 (updated by a load-balance
//!   heuristic, never by backprop — matches the reference `e_score_correction_bias`)
//! - residual-output projections (`attn.o`, `*.down`): Normal(0, 0.02/sqrt(2L))
//! - all other linear weights + embedding + `lm_head`: Normal(0, 0.02)

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::GlmConfig;

/// True for an RMSNorm gain tensor (initialised to 1.0).
fn is_norm_gain(name: &str) -> bool {
    name == "norm.weight" || name.ends_with("_ln.weight") || name.ends_with("_norm.weight")
}

/// True for a residual-output projection (GPT-2 scaled init): the attention
/// output `attn.o` and any MLP/expert/shared `down` projection.
fn is_residual_proj(name: &str) -> bool {
    name.ends_with("attn.o.weight") || name.ends_with("down.weight")
}

pub fn init_weights(cfg: &GlmConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let std = 0.02f32;
    let proj_std = 0.02f32 / ((2.0 * cfg.n_layers as f32).sqrt());
    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> {
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };

    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if is_norm_gain(&name) {
            vec![1.0; numel]
        } else if name.ends_with(".bias") {
            vec![0.0; numel] // router selection bias + LayerNorm bias
        } else if is_residual_proj(&name) {
            normal(numel, proj_std, &mut rng)
        } else {
            normal(numel, std, &mut rng)
        };
        w.insert(name, v);
    }
    w
}
