// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen weight initialization (deterministic for a fixed seed):
//! - RMSNorm gains (`*.weight` ending in `norm`/`ln`): 1.0
//! - residual projections (`attn.wo.weight`, `mlp.down.weight`): Normal(0, 0.02/sqrt(2*L))
//! - other linear weights + embedding: Normal(0, 0.02)
//! - LoRA `A`: Normal(0, 0.02); LoRA `B`: 0 (adapter starts as a no-op).

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::QwenConfig;

/// True for an RMSNorm gain tensor (initialised to 1.0).
fn is_norm_gain(name: &str) -> bool {
    name == "norm.weight"
        || name.ends_with("ln1.weight")
        || name.ends_with("ln2.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight")
}

/// True for a residual-output projection (GPT-2 scaled init).
fn is_residual_proj(name: &str) -> bool {
    name.ends_with("attn.wo.weight") || name.ends_with("mlp.down.weight")
}

pub fn init_weights(cfg: &QwenConfig, seed: u64) -> HashMap<String, Vec<f32>> {
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
        } else if name.ends_with(".lora_b") {
            vec![0.0; numel] // zero-init so the adapter starts as identity
        } else if name.ends_with(".lora_a") {
            normal(numel, std, &mut rng)
        } else if is_residual_proj(&name) {
            normal(numel, proj_std, &mut rng)
        } else {
            normal(numel, std, &mut rng)
        };
        w.insert(name, v);
    }
    w
}

/// The LoRA factors of `cfg` alone - what a fine-tune of an existing base
/// needs, without drawing (and holding) random numbers for every base
/// weight it is about to overwrite: `A` Normal(0, 0.02), `B` zero, so the
/// adapter starts as a no-op. Deterministic for a fixed seed.
pub fn init_adapter_weights(cfg: &QwenConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    cfg.param_list()
        .into_iter()
        .filter_map(|(name, numel)| {
            let v = if name.ends_with(".lora_b") {
                vec![0.0; numel]
            } else if name.ends_with(".lora_a") {
                (0..numel).map(|_| (rng.next_gaussian() as f32) * 0.02).collect()
            } else {
                return None;
            };
            Some((name, v))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LoraCfg;

    #[test]
    fn adapter_init_is_the_adapters_alone_and_a_no_op() {
        let cfg = QwenConfig { lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
        let w = init_adapter_weights(&cfg, 5);
        let adapters: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| n.contains(".lora_")).collect();
        assert!(!adapters.is_empty());
        let mut names: Vec<&String> = w.keys().collect();
        names.sort();
        let mut want: Vec<&String> = adapters.iter().collect();
        want.sort();
        assert_eq!(names, want, "exactly the adapter tensors, no base weight");
        for (name, v) in &w {
            if name.ends_with(".lora_b") {
                assert!(v.iter().all(|x| *x == 0.0), "{name}: B starts at zero");
            } else {
                assert!(v.iter().any(|x| *x != 0.0), "{name}: A is random");
            }
        }
        assert_eq!(w, init_adapter_weights(&cfg, 5), "deterministic for a seed");
        assert_ne!(w, init_adapter_weights(&cfg, 6), "and different for another");
    }
}
