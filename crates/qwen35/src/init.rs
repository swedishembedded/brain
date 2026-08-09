// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic fresh-weight init for tests / cold starts, fixed-seed, no
//! external RNG crate (`data::rng::Lcg`) — this repo's standard convention
//! (`crates/glm/src/init.rs`, `crates/qwen/src/init.rs`).
//!
//! RMSNorm gains are initialised to `1.0` — the FINAL per-channel multiplier
//! this engine's shared `rmsnorm.wgsl`/`Qwen3_5MoeRMSNormGated`'s own
//! `self.weight * hidden_states` both assume — not the raw HF
//! `Qwen3_5MoeRMSNorm` convention of storing `weight` such that the applied
//! multiplier is `1+weight` (that class's own `_init_weights` zero-inits it
//! for exactly that reason). See `model.rs`'s module doc for the resulting
//! gap against a REAL imported checkpoint (not relevant to fresh-weight
//! tests, since there is no "raw HF value" here to be off by one from).
//!
//! `dt_bias`/`A_log` mirror the reference's own init exactly
//! (`Qwen3_5MoePreTrainedModel._init_weights`): `dt_bias` ones, `A_log =
//! log(Uniform(0,16))`.

use std::collections::HashMap;

use data::rng::Lcg;

use crate::config::Qwen35Config;

pub fn init_weights(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let std = 0.02f32;

    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if name.ends_with("norm.weight") {
            vec![1.0f32; numel]
        } else if name.ends_with(".dt_bias") {
            vec![1.0f32; numel]
        } else if name.ends_with(".A_log") {
            // A ~ Uniform(0,16), A_log = log(A). Floor A away from 0 so log
            // never returns -inf for a fresh (never-trained) init.
            (0..numel).map(|_| (rng.unit() * 16.0).max(1e-4).ln()).collect()
        } else {
            rng.vec_scaled(numel, std)
        };
        w.insert(name, v);
    }
    w
}
