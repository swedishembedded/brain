// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic synthetic weights for the tiny smoke test.
//!
//! A real ControlNet is always imported ([`crate::import`]); this exists so the
//! graph can be exercised at toy dims ([`ControlNetConfig::tiny`]) in ~a second
//! — the porting playbook's §4 rung.
//!
//! One deliberate difference from `unet::init`: the **zero-convs are not
//! zero here**. A freshly-initialised ControlNet has `controlnet_*` weights and
//! biases at exactly 0 (that is what makes it a no-op at the start of
//! training), and a smoke test built on that would pass with *any* down/mid
//! graph at all — every residual would be zero whatever went into it. The
//! released checkpoints are trained and their zero-convs are not zero, so the
//! synthetic ones are not either, and [`crate::init::init_weights`] says so.

use data::rng::Rng;

use crate::config::ControlNetConfig;
use crate::import::Tensors;

/// A full tensor set covering exactly [`ControlNetConfig::tensor_manifest`],
/// deterministic for a fixed `seed`.
pub fn init_weights(cfg: &ControlNetConfig, seed: u64) -> Tensors {
    let mut rng = Rng::new(seed);
    let mut out: Tensors = Default::default();
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        let fan_in: usize = if shape.len() > 1 { numel / shape[0] } else { 1 };
        let is_gain = name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name.ends_with("norm3.weight")
            || name.ends_with("norm.weight");
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
        out.insert(name, (shape, v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The zero-convs must be NON-zero in the synthetic set — see the module
    /// header. If this ever regresses, every downstream smoke assertion about
    /// the residuals becomes vacuous.
    #[test]
    fn synthetic_zero_convs_are_not_zero() {
        let cfg = ControlNetConfig::tiny();
        let t = init_weights(&cfg, 7);
        for name in ["controlnet_down_blocks.0.weight", "controlnet_mid_block.weight"] {
            let (_, d) = t.get(name).unwrap_or_else(|| panic!("{name}"));
            assert!(d.iter().any(|v| v.abs() > 1e-6), "{name} is all zeros");
        }
    }

    #[test]
    fn init_covers_the_manifest_exactly() {
        let cfg = ControlNetConfig::tiny();
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
