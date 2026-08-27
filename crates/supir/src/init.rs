// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic synthetic weights for the weight-free smoke tests.
//!
//! A real SUPIR model is always imported ([`crate::import`]); this exists so
//! `trunk`/`adaptors`/`model` can be exercised at toy dims
//! ([`crate::config::SupirConfig::tiny`]) in about a second - the porting
//! playbook's §4 rung.
//!
//! Same departure from `sdxlunet::init` that `controlnet::init` documents:
//! the zero-convs (`zero_conv`/`zero_mul`/`zero_add`) are NOT zero here. A
//! freshly-initialised SUPIR has those at exactly 0 (the identity-at-init
//! property the roadmap's forward formula relies on), and a smoke test built
//! on that would pass with any trunk/adaptor graph at all - every adaptor
//! contribution would be zero whatever fed it. The released checkpoint is
//! trained and its zero-convs are not zero, so the synthetic ones are not
//! either.

use data::rng::Rng;

use crate::config::SupirConfig;

/// A full tensor set covering exactly `manifest`, deterministic for a fixed
/// `seed`. Generic over the manifest (rather than over [`SupirConfig`]
/// directly) so `trunk.rs`/`adaptors.rs`'s own tests can synthesise just
/// their half without pulling in the other's tensors.
pub fn init_weights_for(manifest: &[(String, Vec<usize>)], seed: u64) -> sdxlunet::import::Tensors {
    let mut rng = Rng::new(seed);
    let mut out: sdxlunet::import::Tensors = Default::default();
    for (name, shape) in manifest {
        let numel: usize = shape.iter().product();
        let fan_in: usize = if shape.len() > 1 { numel / shape[0] } else { 1 };
        let is_gain = name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name.ends_with("param_free_norm.weight");
        let s = if is_gain {
            0.1
        } else if name.ends_with(".bias") {
            0.05
        } else {
            1.0 / (fan_in as f32).sqrt()
        };
        let mut v: Vec<f32> = (0..numel).map(|_| rng.next_gaussian() as f32 * s).collect();
        if is_gain {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        out.insert(name.clone(), (shape.clone(), v));
    }
    out
}

/// The full [`SupirConfig`] delta (trunk + adaptors + denoise_encoder),
/// synthesised in one call.
pub fn init_weights(cfg: &SupirConfig, seed: u64) -> sdxlunet::import::Tensors {
    init_weights_for(&cfg.tensor_manifest(), seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The zero-convs must be NON-zero in the synthetic set - see the module
    /// header. If this ever regresses, every downstream smoke assertion
    /// about the adaptors' contribution becomes vacuous.
    #[test]
    fn synthetic_zero_convs_are_not_zero() {
        let cfg = SupirConfig::tiny();
        let t = init_weights(&cfg, 7);
        let j = cfg.adaptors.joins[0];
        for name in [
            format!("project_modules.{}.zero_conv.weight", j.pm_idx),
            format!("project_modules.{}.zero_mul.weight", j.pm_idx),
            format!("project_modules.{}.zero_add.weight", j.pm_idx),
        ] {
            let (_, d) = t.get(&name).unwrap_or_else(|| panic!("{name}"));
            assert!(d.iter().any(|v| v.abs() > 1e-6), "{name} is all zeros");
        }
    }

    #[test]
    fn init_covers_the_manifest_exactly() {
        let cfg = SupirConfig::tiny();
        let m = cfg.tensor_manifest();
        let t = init_weights(&cfg, 1);
        assert_eq!(t.len(), m.len());
        for (name, shape) in &m {
            let (s, d) = t.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(s, shape, "{name}");
            assert_eq!(d.len(), shape.iter().product::<usize>(), "{name}");
        }
    }
}
