// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Encoder weight initialization, deterministic for a fixed seed - for the
//! gradient check and for from-scratch experiments. A real run starts from the
//! released checkpoint instead (`crate::import`).
//!
//! BERT's own recipe: every weight and embedding table `Normal(0, 0.02)` (the
//! checkpoint's `initializer_range`), every LayerNorm gain 1 and bias 0.
//!
//! The gains are deliberately NOT randomized. A LayerNorm initialized away
//! from 1 changes the scale every downstream layer sees, which would make a
//! from-scratch run diverge for a reason that has nothing to do with the
//! architecture - and the finite-difference check perturbs them regardless, so
//! leaving them at 1 costs no coverage.

use crate::Tensors;
use std::collections::HashMap;

use data::rng::Rng;

use crate::config::EncoderConfig;

/// BERT's `initializer_range`.
const STD: f32 = 0.02;

fn is_norm_gain(name: &str) -> bool {
    name.ends_with("_ln.weight") || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "head.ln.weight"
}

fn is_bias(name: &str) -> bool {
    name.ends_with(".bias")
}

pub fn init_weights(cfg: &EncoderConfig, seed: u64) -> Tensors {
    fill(cfg.tensor_manifest(), seed)
}

/// The head's weights. Always fresh: the head has no pretrained counterpart,
/// which is exactly why it takes a larger learning rate than the encoder.
pub fn init_head(cfg: &EncoderConfig, seed: u64) -> Tensors {
    fill(crate::head::tensor_manifest(cfg), seed)
}

fn fill(manifest: Vec<(String, Vec<usize>)>, seed: u64) -> Tensors {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, shape) in manifest {
        let numel: usize = shape.iter().product();
        let v = if is_norm_gain(&name) {
            vec![1.0; numel]
        } else if is_bias(&name) {
            vec![0.0; numel]
        } else {
            (0..numel).map(|_| rng.next_gaussian() as f32 * STD).collect()
        };
        w.insert(name, v);
    }
    w
}
