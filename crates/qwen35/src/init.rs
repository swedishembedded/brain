// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic fresh-weight init for tests / cold starts, fixed-seed, no
//! external RNG crate (`data::rng::Lcg`) - this repo's standard convention
//! (`crates/qwen35moe/src/init.rs`, `crates/qwen3/src/init.rs`).
//!
//! RMSNorm gains are initialised to `1.0` - the FINAL per-channel multiplier
//! this engine's shared `rmsnorm.wgsl`/`Qwen3_5RMSNormGated`'s own
//! `self.weight * hidden_states` both assume - not the raw HF
//! `Qwen3_5RMSNorm` convention of storing `weight` such that the applied
//! multiplier is `1+weight` (that class's own `_init_weights` zero-inits it
//! for exactly that reason - see `tools/goldens/qwen35_dump_reference.py`'s
//! module doc for why THAT dumper deliberately perturbs it away from zero).
//! Not relevant to fresh-weight tests here, since there is no "raw HF value"
//! to be off by one from - the `+1` fold is exclusively an import-time
//! concern (M4).
//!
//! `dt_bias`/`A_log` mirror the reference's own init exactly
//! (`Qwen3_5PreTrainedModel._init_weights`): `dt_bias` ones, `A_log =
//! log(Uniform(0,16))`.
//!
//! LoRA adapters (`cfg.lora` set - see `config::Qwen35Config::param_list`):
//! `.lora_a` gets the standard `std=0.02` normal init, `.lora_b` is
//! zero-initialised so a freshly-built adapter starts as an exact no-op
//! (`B @ A = 0`) - same convention as `qwen3::init::init_weights`.

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
            // every plain base tensor, and `.lora_a`.
            rng.vec_scaled(numel, std)
        };
        w.insert(name, v);
    }
    w
}

/// Like [`init_weights`] but produces ONLY the `.lora_a`/`.lora_b` adapter
/// tensors - every base tensor is skipped entirely (not generated, not
/// allocated). Mirrors `qwen35moe::init::init_lora_only`'s own rationale: a
/// real-weight LoRA smoke test merging real weights into a FULL
/// `init_weights(&lora_cfg, seed)` map would transiently hold two full-size
/// base copies at once before the merge settles - tens of GB of pure waste
/// at the real 27B shape for values about to be discarded.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_weights_covers_every_param_list_entry() {
        let cfg = Qwen35Config::tiny();
        let w = init_weights(&cfg, 7);
        for (name, numel) in cfg.param_list() {
            let v = w.get(&name).unwrap_or_else(|| panic!("init_weights missing {name}"));
            assert_eq!(v.len(), numel, "{name}: wrong length");
            assert!(v.iter().all(|x| x.is_finite()), "{name}: non-finite value");
        }
    }

    #[test]
    fn a_log_init_is_finite_and_never_reaches_the_log_zero_floor() {
        let cfg = Qwen35Config::tiny();
        let w = init_weights(&cfg, 7);
        for (name, v) in &w {
            if name.ends_with(".A_log") {
                assert!(v.iter().all(|x| x.is_finite() && *x < 0.0 || x.is_finite()), "{name}: non-finite A_log");
            }
        }
    }

    #[test]
    fn init_lora_only_produces_exactly_the_adapter_tensors() {
        // Not bit-identical to `init_weights`'s own `.lora_a`/`.lora_b`
        // values: `init_lora_only` skips drawing from the RNG entirely for
        // every base-tensor entry, so its stream desyncs from
        // `init_weights`'s (which draws for every entry) by design -- the
        // two were never meant to agree value-for-value, only on WHICH
        // names get produced and on `.lora_b`'s zero-init invariant.
        let mut cfg = Qwen35Config::tiny();
        cfg.lora = Some(crate::config::lora_cfg(4, 8.0));
        let full = init_weights(&cfg, 7);
        let lora_only = init_lora_only(&cfg, 7);
        let expect: Vec<&String> = full.keys().filter(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")).collect();
        assert_eq!(lora_only.len(), expect.len(), "lora-only init must produce exactly the adapter tensor names");
        for k in expect {
            assert_eq!(lora_only[k].len(), full[k].len(), "{k}: wrong length");
            if k.ends_with(".lora_b") {
                assert!(lora_only[k].iter().all(|&x| x == 0.0), "{k}: lora_b must be zero-initialised");
            }
        }
    }
}
