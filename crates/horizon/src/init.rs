// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic initial weights.

use std::collections::HashMap;

use model::hostmath::gaussian;

use crate::config::HorizonConfig;

/// The initial log-hazard of every code: about one event per hundred units of
/// time at risk, a plausible order for annual adult mortality and harmless
/// elsewhere (the bias is learned first and fastest).
const INITIAL_LOG_HAZARD: f32 = -4.6;

/// Initial weights for `cfg`, a pure function of `seed`.
pub fn init_weights(cfg: &HorizonConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let residual = 1.0 / (2.0 * cfg.n_layers.max(1) as f32).sqrt();
    let fan_in = |name: &str| -> f32 {
        let d = cfg.d_model as f32;
        if name.ends_with("ffn.down.weight") {
            cfg.d_ff as f32
        } else if name == "hazard.code.weight" {
            cfg.rank as f32
        } else if name == "hazard.time.weight" || name == "additive.time.weight" {
            cfg.time_features() as f32
        } else {
            d
        }
    };
    let mut out = HashMap::new();
    for (i, (name, numel)) in cfg.param_list().into_iter().enumerate() {
        let noise = |std: f32| {
            gaussian(
                numel,
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64),
            )
            .into_iter()
            .map(|x| x * std)
            .collect::<Vec<f32>>()
        };
        let v = if name == "tok.gamma" {
            noise(0.02).into_iter().map(|x| 1.0 + x).collect()
        } else if name == "tok.beta" {
            noise(0.02)
        } else if name == "value_bins.weight" || name == "time_bins.weight" {
            noise(0.3)
        } else if name.ends_with(".bias") && name.starts_with("hazard.code") {
            vec![INITIAL_LOG_HAZARD; numel]
        } else if name.ends_with("ln1.weight")
            || name.ends_with("ln2.weight")
            || name == "ln_f.weight"
        {
            vec![1.0; numel]
        } else if name.ends_with(".bias") {
            vec![0.0; numel]
        } else if name.ends_with("attn.out.weight") || name.ends_with("ffn.down.weight") {
            noise(residual / fan_in(&name).sqrt())
        } else if name == "additive.state.weight" {
            // z is a SUM over a subject's tokens: start each token's
            // contribution to the log-hazard small.
            noise(0.01 / fan_in(&name).sqrt())
        } else if name == "hazard.code.weight"
            || name == "value_head.weight"
            || name == "additive.time.weight"
        {
            noise(0.1 / fan_in(&name).sqrt())
        } else {
            noise(1.0 / fan_in(&name).sqrt())
        };
        out.insert(name, v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialisation_is_deterministic_and_complete() {
        let cfg = HorizonConfig::tiny(10, 2);
        let a = init_weights(&cfg, 3);
        let b = init_weights(&cfg, 3);
        assert_eq!(a, b);
        assert_ne!(a["tok.beta"], init_weights(&cfg, 4)["tok.beta"]);
        for (name, numel) in cfg.param_list() {
            assert_eq!(a[&name].len(), numel, "{name}");
        }
        assert!(a["hazard.code.bias"]
            .iter()
            .all(|&x| x == INITIAL_LOG_HAZARD));
    }
}
