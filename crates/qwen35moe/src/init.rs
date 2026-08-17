// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic fresh-weight init for tests / cold starts, fixed-seed, no
//! external RNG crate (`data::rng::Lcg`) — this repo's standard convention
//! (`crates/glm/src/init.rs`, `crates/qwen3/src/init.rs`).
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
//!
//! LoRA adapters (`cfg.lora` set — see `config::Qwen35Config::param_list`):
//! `.lora_a` gets the standard `std=0.02` normal init, `.lora_b` is
//! zero-initialised so a freshly-built adapter starts as an exact no-op
//! (`B @ A = 0`) — same convention as `qwen3::init::init_weights`.

use std::collections::HashMap;

use data::rng::Lcg;

use crate::config::Qwen35Config;

pub fn init_weights(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let std = 0.02f32;

    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if name.ends_with("norm.weight") || name.ends_with(".dt_bias") {
            vec![1.0f32; numel]
        } else if name.ends_with(".A_log") {
            // A ~ Uniform(0,16), A_log = log(A). Floor A away from 0 so log
            // never returns -inf for a fresh (never-trained) init.
            (0..numel).map(|_| (rng.unit() * 16.0).max(1e-4).ln()).collect()
        } else if name.ends_with(".lora_b") {
            vec![0.0f32; numel] // zero-init so the adapter starts as identity
        } else {
            // Everything else, `.lora_a` included: N(0, std). The adapter is
            // still identity at step 0 because `.lora_b` above is zero, so `A`
            // takes the same spread as a base weight.
            rng.vec_scaled(numel, std)
        };
        w.insert(name, v);
    }
    w
}

/// Like [`init_weights`] but produces ONLY the `.lora_a`/`.lora_b` adapter
/// tensors — every base tensor is skipped entirely (not generated, not
/// allocated). For a real checkpoint's LoRA smoke test
/// (`crates/qwen35moe/tests/lora_real_weight_smoke.rs`): merging real weights
/// into a FULL `init_weights(&lora_cfg, seed)` map would transiently hold two
/// full-size base copies at once (the freshly random one, and the real one
/// about to overwrite it) before the merge settles — at the real 35B-A3B
/// shape that's tens of GB of pure waste for values that are about to be
/// discarded. This produces only the small adapter tensors, so the caller's
/// `for (k, v) in real_base { adapters.insert(k, v); }` merge never
/// allocates a second full-size base copy.
pub fn init_lora_only(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let std = 0.02f32;
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        if name.ends_with(".lora_b") {
            w.insert(name, vec![0.0f32; numel]);
        } else if name.ends_with(".lora_a") {
            w.insert(name, rng.vec_scaled(numel, std));
        }
        // else: a base tensor -- the caller supplies the real weight.
    }
    w
}
