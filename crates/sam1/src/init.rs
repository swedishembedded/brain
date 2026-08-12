// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic weight initialization for a fixed seed.
//!
//! Two entry points, and the difference is load-bearing rather than stylistic --
//! see [`init_dense`].

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::SamViTConfig;

/// LayerNorm / LayerNorm2d gain (initialised to 1.0).
fn is_norm_gain(name: &str) -> bool {
    name.ends_with("norm1.weight") || name.ends_with("norm2.weight")
}

/// Training-style init: norm gains 1, every bias 0, every other tensor
/// `Normal(0, 0.02)` with the residual-output projections (`attn.proj`,
/// `mlp.fc2`) scaled by `1/sqrt(2L)` -- the GPT-2 rule the ViT family inherits.
pub fn init_weights(cfg: &SamViTConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let proj_std = 0.02f32 / (2.0 * cfg.n_layers as f32).sqrt();
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if is_norm_gain(&name) {
            vec![1.0; numel]
        } else if name.ends_with(".bias") {
            vec![0.0; numel]
        } else {
            let s = if name.ends_with("attn.proj.weight") || name.ends_with("mlp.fc2.weight") { proj_std } else { 0.02 };
            (0..numel).map(|_| rng.next_gaussian() as f32 * s).collect()
        };
        w.insert(name, v);
    }
    w
}

/// Gradient-check init: **every** tensor is nonzero, gains sit near 1 and the
/// linear weights are scaled so the softmax stays unsaturated.
///
/// Not a cosmetic variant of [`init_weights`]. Three of its choices each defend
/// against a check that would pass vacuously:
///
///  * A **zero `attn.qkv.bias`** makes the window pad path invisible. A padded
///    position's input is exactly zero, so its key and value are the qkv
///    *bias* -- not zero -- and it genuinely participates in its window's
///    softmax. At bias 0 the two are the same tensor and a graph that dropped
///    the pad entirely would still check green.
///  * A **zero `mlp.fc*.bias` / `attn.proj.bias`** leaves `bias_grad`'s own
///    dispatch unexercised in the sense that matters: the bias still has a
///    gradient, but the forward it perturbs is degenerate.
///  * Gains at exactly 1 with zero betas make `layernorm_dgamma` /
///    `layernorm_dbeta` numerically indistinguishable from an identity path.
pub fn init_dense(cfg: &SamViTConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    fn u(n: usize, s: f32, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| s * (rng.next_f32() * 2.0 - 1.0)).collect()
    }
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if is_norm_gain(&name) {
            u(numel, 0.3, &mut rng).iter().map(|x| 1.0 + x).collect()
        } else if name.ends_with(".bias") {
            u(numel, 0.2, &mut rng)
        } else if name.contains("rel_pos") {
            u(numel, 0.5, &mut rng)
        } else {
            u(numel, 0.35, &mut rng)
        };
        w.insert(name, v);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_init_leaves_no_zero_tensor_and_is_deterministic() {
        let cfg = SamViTConfig::tiny();
        let a = init_dense(&cfg, 7);
        let b = init_dense(&cfg, 7);
        assert_eq!(a.len(), cfg.param_list().len());
        for (name, numel) in cfg.param_list() {
            let v = &a[&name];
            assert_eq!(v.len(), numel, "{name}");
            assert!(v.iter().any(|x| x.abs() > 1e-6), "{name} initialised to all zeros");
            assert_eq!(v, &b[&name], "{name} is not deterministic for a fixed seed");
        }
    }

    #[test]
    fn training_init_puts_gains_at_one_and_biases_at_zero() {
        let cfg = SamViTConfig::tiny();
        let w = init_weights(&cfg, 3);
        assert!(w["vision.sam.blocks.0.norm1.weight"].iter().all(|&x| x == 1.0));
        assert!(w["vision.sam.blocks.0.attn.qkv.bias"].iter().all(|&x| x == 0.0));
        assert!(w["vision.sam.neck.norm1.weight"].iter().all(|&x| x == 1.0));
    }
}
