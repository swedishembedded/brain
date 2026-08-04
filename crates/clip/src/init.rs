// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Random weight initialization for a CLIP **text** tower.
//!
//! This exists for tests and gradient checks — a real CLIP-L / bigG tower is
//! always imported ([`crate::import`]), never initialized from scratch. The
//! scheme follows `transformers`' `CLIPPreTrainedModel._init_weights` in shape
//! (embeddings and linears normal, LayerNorm gain 1 / bias 0) but NOT in scale:
//! the reference's `factor * (d ** -0.5)` deviations are tiny, which at a
//! gradcheck's 16-channel config drives every activation into the linear regime
//! of `quick_gelu` and softmax and makes the FD comparison test almost nothing.
//! So the gains are jittered and the matrices use a plain `1/sqrt(fan_in)`.
//!
//! `deterministic for a fixed seed` is the contract the gradient check relies
//! on; `data::rng::Rng` is the workspace's only RNG.

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::ClipTextConfig;

/// Build an initial weight map for `cfg`, deterministic for a fixed `seed`.
///
/// Covers exactly [`ClipTextConfig::tensor_manifest`], so the result can be
/// handed straight to [`crate::model::ClipText::new_train_on`].
pub fn init_text_weights(cfg: &ClipTextConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    let normal = |n: usize, s: f32, rng: &mut Rng| -> Vec<f32> {
        (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
    };
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        // fan_in is the LAST axis for every 2-D tensor in the manifest
        // (`[out, in]`, matching `matmul`'s `[N, K]` weight layout).
        let fan_in = *shape.last().expect("non-empty shape");
        let v = if name.ends_with("norm.weight") || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") {
            // LayerNorm gain: 1 + jitter, so `dgamma` is not evaluated at a
            // point where every gain is identical.
            normal(numel, 0.1, &mut rng).iter().map(|x| 1.0 + x).collect()
        } else if name.ends_with(".bias") {
            normal(numel, 0.05, &mut rng)
        } else if name == "tok.weight" || name == "pos.weight" {
            normal(numel, 0.5, &mut rng)
        } else {
            normal(numel, 1.0 / (fan_in as f32).sqrt(), &mut rng)
        };
        assert_eq!(v.len(), numel, "{name}: init size");
        w.insert(name, v);
    }
    w
}

/// A fixed token batch for `cfg` with a valid EOS layout: `bos`, a deterministic
/// content span, then `eos` and right-padding.
///
/// `ClipText::set_tokens` pools at the FIRST argmax of the row and asserts it is
/// `cfg.eos_id`, so every content token must be strictly below `eos_id` — which
/// this guarantees. Sample `s` puts its EOS at position `1 + s` from the end, so
/// the batch exercises more than one pooling row.
pub fn fixed_tokens(cfg: &ClipTextConfig, b: u32, t: u32) -> Vec<u32> {
    assert!(t >= b + 2, "need room for bos + content + eos in every row");
    let mut ids = vec![cfg.pad_id; (b * t) as usize];
    for s in 0..b {
        let row = &mut ids[(s * t) as usize..((s + 1) * t) as usize];
        let eos_at = (t - 1 - s) as usize;
        row[0] = cfg.bos_id.min(cfg.eos_id - 1);
        for (i, slot) in row.iter_mut().enumerate().take(eos_at).skip(1) {
            *slot = ((i as u32 * 7 + s * 3) % (cfg.eos_id - 1)).max(1);
        }
        row[eos_at] = cfg.eos_id;
    }
    ids
}
