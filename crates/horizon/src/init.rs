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
        let v = if name == "visit.state" {
            visit_state(cfg.d_model as usize)
        } else if name == "tok.gamma" {
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

/// The continuous-time state's initial `[raw rate | population state]`: rates
/// log-spaced from one per tenth of a time unit to one per hundred units
/// (raw = the inverse of softplus), so some channels forget within a visit
/// interval and others carry a lifetime; the population state starts at 0.
fn visit_state(d: usize) -> Vec<f32> {
    let (fast, slow) = (10.0f64.ln(), 0.01f64.ln());
    let mut v: Vec<f32> = (0..d)
        .map(|c| {
            let r = (fast + (slow - fast) * c as f64 / (d.max(2) - 1) as f64).exp();
            (r.exp_m1().ln()) as f32
        })
        .collect();
    v.resize(2 * d, 0.0);
    v
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
        let mut cfg = cfg;
        cfg.visits = 3;
        let s = &init_weights(&cfg, 3)["visit.state"];
        let softplus = |x: f32| (1.0 + x.exp()).ln();
        let d = cfg.d_model as usize;
        assert!(
            (softplus(s[0]) - 10.0).abs() < 1e-3 && (softplus(s[d - 1]) - 0.01).abs() < 1e-5,
            "{s:?}"
        );
        assert!(s[d..].iter().all(|&m| m == 0.0));
    }
}
