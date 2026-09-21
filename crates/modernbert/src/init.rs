// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic random weight init, for the structural test and any
//! from-scratch experiment. A real run starts from the released checkpoint
//! instead (Laya M4's importer).
//!
//! Mirrors `crates/decide/src/init.rs`'s own recipe (BERT's
//! `initializer_range`: every weight `Normal(0, 0.02)`, every LayerNorm gain
//! set to one) with one difference this crate's shape requires: there is no
//! bias anywhere in the trunk to leave at zero (`norm_bias`/`attention_bias`/
//! `mlp_bias` are all false), so `is_bias` has nothing to match here and is
//! simply absent rather than a dead branch.

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::ModernBertConfig;
use crate::laya::LayaConfig;

/// ModernBERT's own `initializer_range` (`answerdotai/ModernBERT-large`'s
/// released `config.json`; same value BERT used).
const STD: f32 = 0.02;

fn is_norm_gain(name: &str) -> bool {
    name.ends_with("_norm.weight")
}

pub fn init_weights(cfg: &ModernBertConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        let v = if is_norm_gain(&name) {
            vec![1.0; numel]
        } else {
            (0..numel).map(|_| rng.next_gaussian() as f32 * STD).collect()
        };
        w.insert(name, v);
    }
    w
}

/// Random weight init for [`LayaHead`](crate::laya::LayaHead)'s own
/// tensor manifest - the fixture-free structural test's source of weights.
/// Unlike the trunk above, the head has real biases (see `laya.rs`'s module
/// doc): every `*.bias` tensor starts at zero (`torch.nn.Linear`/
/// `torch.nn.LayerNorm`'s own default), every `*norm*.weight` gain starts at
/// one, everything else is `Normal(0, 0.02)`.
pub fn init_weights_laya(cfg: &LayaConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, shape) in crate::laya::tensor_manifest(cfg) {
        let numel: usize = shape.iter().product();
        let v = if name.ends_with(".bias") {
            vec![0.0; numel]
        } else if name.contains("norm") && name.ends_with(".weight") {
            vec![1.0; numel]
        } else {
            (0..numel).map(|_| rng.next_gaussian() as f32 * STD).collect()
        };
        w.insert(name, v);
    }
    w
}
