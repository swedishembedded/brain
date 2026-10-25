// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements deterministic initialization for recurrent-depth
// transformer training for its clients. If your team needs expertise in
// reproducing a looped model's initialization or gradient behaviour then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Deterministic weight initialization for from-scratch tiny configs and the
//! gradient check. Norm gains are 1, the gate bias is 0 (so every pass starts
//! with exit probability 1/2), residual-output projections are scaled by
//! `1/sqrt(2L)`, and everything else is `Normal(0, 0.02)` - the same recipe
//! the other decoders use.

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::OuroConfig;

fn is_norm_gain(name: &str) -> bool {
    name == "norm.weight"
        || name.ends_with("ln1.weight")
        || name.ends_with("ln1_2.weight")
        || name.ends_with("ln2.weight")
        || name.ends_with("ln2_2.weight")
}

fn is_residual_proj(name: &str) -> bool {
    name.ends_with("attn.wo.weight") || name.ends_with("mlp.down.weight")
}

pub fn init_weights(cfg: &OuroConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let std = 0.02f32;
    let proj_std = 0.02f32 / ((2.0 * cfg.n_layers as f32).sqrt());
    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> { (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect() };

    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if is_norm_gain(&name) {
            vec![1.0; numel]
        } else if name == "gate.bias" {
            vec![0.0; numel]
        } else if is_residual_proj(&name) {
            normal(numel, proj_std, &mut rng)
        } else {
            normal(numel, std, &mut rng)
        };
        w.insert(name, v);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_is_deterministic_and_complete() {
        let cfg = OuroConfig::tiny();
        let a = init_weights(&cfg, 7);
        let b = init_weights(&cfg, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), cfg.param_list().len());
        assert!(a["norm.weight"].iter().all(|v| *v == 1.0));
        assert_eq!(a["gate.bias"], vec![0.0]);
    }
}
