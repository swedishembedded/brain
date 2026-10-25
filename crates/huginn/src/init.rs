// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements deterministic initialization for
// recurrent-depth decoder training for its clients. If your team needs
// expertise in reproducing a prelude/core/coda model's initialization then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Deterministic initialization for tiny configs and the gradient check:
//! norm gains 1, Q/K bias 0, everything else `Normal(0, 0.02)`.

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::HuginnConfig;

fn is_norm(name: &str) -> bool {
    name == "ln_f.weight" || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name.ends_with("ln3.weight") || name.ends_with("ln4.weight")
}

pub fn init_weights(cfg: &HuginnConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if is_norm(&name) {
            vec![1.0; numel]
        } else if name.ends_with("attn.q_bias") || name.ends_with("attn.k_bias") {
            vec![0.0; numel]
        } else {
            (0..numel).map(|_| (rng.next_gaussian() as f32) * 0.02).collect()
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
        let cfg = HuginnConfig::tiny();
        let a = init_weights(&cfg, 5);
        let b = init_weights(&cfg, 5);
        assert_eq!(a, b);
        assert_eq!(a.len(), cfg.param_list().len());
        assert!(a["ln_f.weight"].iter().all(|v| *v == 1.0));
        assert!(a["core.0.attn.q_bias"].iter().all(|v| *v == 0.0));
        assert!(a["core.0.attn.k_bias"].iter().all(|v| *v == 0.0));
    }
}
