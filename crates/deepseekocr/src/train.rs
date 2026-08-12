// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA training glue for the composite.
//!
//! The adapter mechanism itself lives one layer down, in
//! `deepseekv2::config::LoraCfg` / `deepseekv2::model::DeepseekV2` (frozen
//! base + trainable `.lora_a`/`.lora_b` on the decoder's four attention
//! projections, composed entirely from the `matmul`/`matmul_dx`/`matmul_dw`/
//! `axpy`/`grad_scale` kernels the decoder's own forward/backward already
//! use -- see that crate's module docs). This file is only the
//! **composite-level seam**: [`DeepseekOcrConfig::decoder`]'s `lora` field is
//! threaded straight through `DeepseekV2::new_on`'s own role assignment, so
//! [`crate::DeepseekOcr::new`]/[`crate::DeepseekOcr::new_split`] need **no
//! change at all** to build a LoRA-adapted composite once `cfg.decoder.lora`
//! is set -- only the init map needs the one tensor family a real checkpoint
//! (or the checkpoint-free fixture) never carries.

use std::collections::HashMap;

use crate::config::DeepseekOcrConfig;

/// `base` overlaid with fresh `.lora_a`/`.lora_b` adapter tensors for
/// `cfg.decoder.lora`'s targets (`Normal(0, 0.02)` on `A`, zero on `B`, so the
/// adapter starts as an exact no-op delta) -- the composite-level counterpart
/// of `qwen3::finetune::finetune`'s "fresh init for the whole (possibly
/// LoRA-extended) param set, then overwrite with the checkpoint's own
/// weights" merge. `base` here is everything a real import (or the
/// checkpoint-free golden fixture's own `build_init`) already covers -- SAM,
/// CLIP, the glue and the decoder's own base weights -- none of which ever
/// carries a `.lora_a`/`.lora_b` tensor (LoRA is trained after import, never
/// part of the checkpoint's own manifest), so only the adapter half of a
/// fresh init needs to be produced and merged in.
///
/// Panics if `cfg.decoder.lora` is `None` (nothing to add), or if `base`
/// already carries one of the tensor names being added (a real checkpoint and
/// a fresh init should never collide; a collision here means `cfg` was built
/// against the wrong base).
pub fn lora_init_map(cfg: &DeepseekOcrConfig, base: &HashMap<String, Vec<f32>>, seed: u64) -> HashMap<String, Vec<f32>> {
    assert!(cfg.decoder.lora.is_some(), "lora_init_map: cfg.decoder.lora is None -- nothing to add");
    let mut init = base.clone();
    for (name, data) in deepseekv2::init::init_adapters(&cfg.decoder, seed) {
        assert!(
            init.insert(name.clone(), data).is_none(),
            "{name}: base already carries an adapter tensor -- checkpoint and fresh init collided"
        );
    }
    init
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The merge adds exactly the adapter tensors a LoRA-configured decoder's
    /// `param_list()` declares on top of a fixed base, and touches nothing
    /// already in `base`.
    #[test]
    fn lora_init_map_adds_only_the_adapter_tensors_and_leaves_base_untouched() {
        let mut cfg = DeepseekOcrConfig::tiny();
        cfg.decoder.lora = Some(deepseekv2::config::lora_cfg(2, 4.0));
        let base: HashMap<String, Vec<f32>> =
            cfg.decoder.shape.param_list().into_iter().map(|(n, sz)| (n, vec![1.0f32; sz])).collect();

        let adapters = deepseekv2::init::init_adapters(&cfg.decoder, 3);
        assert!(!adapters.is_empty(), "the tiny fixture's decoder must have LoRA-targetable attention leaves");

        let merged = lora_init_map(&cfg, &base, 3);
        assert_eq!(merged.len(), base.len() + adapters.len());
        for (name, data) in &base {
            assert_eq!(merged.get(name), Some(data), "{name}: base tensor was touched by the merge");
        }
        for (name, data) in &adapters {
            assert_eq!(merged.get(name), Some(data), "{name}: adapter tensor was not merged verbatim");
        }
    }

    /// Building the map against a config with no LoRA configured is refused
    /// loudly rather than silently returning `base` unchanged -- a caller that
    /// forgot to set `cfg.decoder.lora` should see this at the call site, not
    /// discover it later as a model with no trainable parameters at all.
    #[test]
    #[should_panic(expected = "cfg.decoder.lora is None")]
    fn lora_init_map_refuses_a_non_lora_config() {
        let cfg = DeepseekOcrConfig::tiny();
        let _ = lora_init_map(&cfg, &HashMap::new(), 0);
    }
}
