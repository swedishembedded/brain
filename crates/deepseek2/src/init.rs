// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic fresh-weight init for tests / cold starts: fixed seed, no
//! external RNG crate (`data::rng::Lcg`) - this repo's standard convention
//! (`crates/qwen35moe/src/init.rs`, `crates/glm/src/init.rs`).
//!
//! Three families, matching every other RMSNorm decoder in this tree:
//!
//! - **Norm gains** (`ln1.weight`, `ln2.weight`, `norm.weight`) start at `1.0` -
//!   the FINAL per-channel multiplier this engine's shared `rmsnorm.wgsl`
//!   applies, not HF's `1 + weight` storage convention. That is also what a
//!   GGUF conversion bakes in, so an imported checkpoint and a fresh init mean
//!   the same thing by the same rule.
//! - **LoRA `B`** (`*.lora_b`, present only when [`DeepseekV2Config::lora`] is
//!   set) starts at exactly zero, so a freshly-built adapter is a no-op delta
//!   (`config::DeepseekV2Config::param_list`'s doc has the shapes); **LoRA
//!   `A`** (`*.lora_a`) gets the same `std = 0.02` noise as everything else,
//!   so `A`'s own gradient is not degenerate merely because `B` starts at
//!   zero -- matches `qwen3::init::init_weights`'s identical convention for
//!   its own `LoraCfg`.
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
        } else if name.ends_with(".lora_b") {
            vec![0.0f32; numel] // zero-init so a fresh adapter starts as an exact no-op
        } else {
            rng.vec_scaled(numel, std)
        };
        w.insert(name, v);
    }
    w
}

/// Just the `.lora_a`/`.lora_b` tensors [`init_weights`] would produce for
/// `cfg.lora`'s targets -- what a caller merges over a REAL checkpoint's own
/// weight map (which never carries them; LoRA is trained after import) rather
/// than re-deriving fresh values for tensors the checkpoint already has.
/// Mirrors `qwen3::finetune::finetune`'s "fresh init for the whole
/// (possibly LoRA-extended) param set, then overwrite with the checkpoint's
/// own weights" merge, keeping only the half of that fresh init the merge
/// actually needs. Empty when `cfg.lora` is `None`.
pub fn init_adapters(cfg: &DeepseekV2Config, seed: u64) -> HashMap<String, Vec<f32>> {
    init_weights(cfg, seed).into_iter().filter(|(n, _)| n.ends_with(".lora_a") || n.ends_with(".lora_b")).collect()
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

    /// [`init_adapters`] produces exactly the tensors a LoRA-configured
    /// `param_list()` adds over the base -- `B` zero, `A` real noise -- and
    /// nothing when there is no LoRA config to draw them from.
    #[test]
    fn init_adapters_covers_only_the_lora_pair_b_zero_a_nonzero() {
        let base = DeepseekV2Config::tiny();
        assert!(init_adapters(&base, 7).is_empty(), "no lora configured -- there is nothing to init");

        let cfg = DeepseekV2Config { lora: Some(crate::config::lora_cfg(2, 4.0)), ..base.clone() };
        let adapters = init_adapters(&cfg, 7);
        let expected: std::collections::HashSet<String> =
            cfg.param_list().into_iter().filter(|(n, _)| n.ends_with(".lora_a") || n.ends_with(".lora_b")).map(|(n, _)| n).collect();
        assert!(!expected.is_empty());
        assert_eq!(adapters.keys().cloned().collect::<std::collections::HashSet<_>>(), expected);
        for (name, v) in &adapters {
            if name.ends_with(".lora_b") {
                assert!(v.iter().all(|&x| x == 0.0), "{name} must be all-zero at fresh init");
            } else {
                assert!(v.iter().any(|&x| x != 0.0), "{name} (lora_a) must not be degenerately zero");
            }
        }
        // Deterministic, and disjoint from the base map a real checkpoint fills.
        assert_eq!(adapters, init_adapters(&cfg, 7));
        for name in base.param_list().into_iter().map(|(n, _)| n) {
            assert!(!adapters.contains_key(&name), "{name}: init_adapters must not touch a base tensor");
        }
    }
}
