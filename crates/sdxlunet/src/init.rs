// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic synthetic weights for the tiny smoke test.
//!
//! A real SDXL UNet is always imported ([`crate::import`]); this exists so the
//! graph can be exercised at toy dims (`UNetConfig::tiny`) in ~a second, which
//! is the porting playbook's §4 rung: a weight-free forward that hits every
//! step kind and catches buffer-sizing and binding bugs long before a
//! real-weights parity run would hit them opaquely.
//!
//! `data::rng::Rng` is the workspace's only RNG.

use data::rng::Rng;

use crate::config::UNetConfig;
use crate::import::Tensors;

/// A full tensor set covering exactly [`UNetConfig::tensor_manifest`],
/// deterministic for a fixed `seed`.
///
/// Scales are `1/sqrt(fan_in)` for matrices and small for biases, with
/// GroupNorm/LayerNorm gains jittered around 1 — the same reasoning as
/// `clip::init`: an exactly-1 gain everywhere makes a whole class of indexing
/// mistakes invisible.
pub fn init_weights(cfg: &UNetConfig, seed: u64) -> Tensors {
    let mut rng = Rng::new(seed);
    let mut out: Tensors = Default::default();
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        // fan_in: the whole receptive field for a conv `[out, in, k, k]`, the
        // last axis for a `[out, in]` linear, and 1 for a vector.
        let fan_in: usize = if shape.len() > 1 { numel / shape[0] } else { 1 };
        let is_gain = name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name.ends_with("norm3.weight")
            || name.ends_with("norm.weight")
            || name == "conv_norm_out.weight";
        let s = if is_gain { 0.1 } else if name.ends_with(".bias") { 0.05 } else { 1.0 / (fan_in as f32).sqrt() };
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
