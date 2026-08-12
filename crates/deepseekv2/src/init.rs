// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic fresh-weight init for tests / cold starts: fixed seed, no
//! external RNG crate (`data::rng::Lcg`) - this repo's standard convention
//! (`crates/qwen35moe/src/init.rs`, `crates/glm/src/init.rs`).
//!
//! Two families, matching every other RMSNorm decoder in this tree:
//!
//! - **Norm gains** (`ln1.weight`, `ln2.weight`, `norm.weight`) start at `1.0`
//!   - the FINAL per-channel multiplier this engine's shared `rmsnorm.wgsl`
//!   applies, not HF's `1 + weight` storage convention. That is also what a
//!   GGUF conversion bakes in, so an imported checkpoint and a fresh init mean
//!   the same thing by the same rule.
//! - **Everything else** (embeddings, the four attention projections, every
//!   dense/expert/shared SwiGLU linear, the router) gets `std = 0.02` normal
//!   noise.
//!
//! Note the router weight is initialised like any other linear rather than
//! zeroed: a zero router makes every expert's logit identical, so the top-k
//! selection is decided by tie-breaking order and the gate is uniform - a
//! degenerate starting point that hides router bugs instead of exercising them.

use std::collections::HashMap;

use data::rng::Lcg;

use crate::config::DeepseekV2Config;

pub fn init_weights(cfg: &DeepseekV2Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let std = 0.02f32;
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if name.ends_with("norm.weight") || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") {
            vec![1.0f32; numel]
        } else {
            rng.vec_scaled(numel, std)
        };
        w.insert(name, v);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every planned tensor is produced, at exactly the planned size, and the
    /// same seed produces the same weights (a non-deterministic init makes a
    /// gradient check unreproducible).
    #[test]
    fn init_covers_the_param_list_and_is_deterministic() {
        let cfg = DeepseekV2Config::tiny();
        let a = init_weights(&cfg, 7);
        let b = init_weights(&cfg, 7);
        assert_eq!(a.len(), cfg.param_list().len());
        for (name, numel) in cfg.param_list() {
            let v = a.get(&name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(v.len(), numel, "{name} size");
            assert_eq!(v, &b[&name], "{name} not deterministic");
            assert!(v.iter().all(|x| x.is_finite()), "{name} has non-finite entries");
        }
    }

    /// The three norm families start at the identity gain; nothing else does.
    #[test]
    fn norm_gains_start_at_one_and_other_tensors_do_not() {
        let cfg = DeepseekV2Config::tiny();
        let w = init_weights(&cfg, 3);
        for name in ["blocks.0.ln1.weight", "blocks.1.ln2.weight", "norm.weight"] {
            assert!(w[name].iter().all(|&x| x == 1.0), "{name} must be all-ones");
        }
        assert!(w["blocks.1.mlp.router.weight"].iter().any(|&x| x != 0.0), "a zero router is degenerate");
    }
}
