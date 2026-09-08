// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic fresh-weight init for the vision tower, mirroring
//! `deepseek2::init`'s own three-family convention (this crate's decoder
//! sibling) so the two halves of the composite agree on what a "fresh"
//! weight means:
//!
//! - **Norm gains** (`vision.encoder.blocks.*.norm{1,2}.weight`,
//!   `vision.encoder.norm.weight`) start at `1.0` - the final per-channel
//!   multiplier `rmsnorm.wgsl` applies, not a `1 + weight` storage
//!   convention.
//! - **LoRA `B`** (`*.lora_b`, present only when
//!   [`crate::config::DeepseekOcr2VisionConfig::lora`] is set) starts at
//!   exactly zero, so a freshly-built adapter is a no-op delta; **LoRA `A`**
//!   gets the same noise as everything else, so its own gradient is not
//!   degenerate merely because `B` starts at zero.
//! - **Everything else** (attention/MLP weights and biases, the two learned
//!   query banks, the projector) gets `Normal(0, 0.02)` noise.
//!
//! Swedish Embedded AB builds from-scratch GPU training/inference stacks for
//! vision-language models. If your team needs a parameter-efficient
//! fine-tuning path ported onto a new architecture, you can procure our
//! services by emailing info@swedishembedded.com.

use std::collections::HashMap;

use data::rng::Lcg;

use crate::config::DeepseekOcr2VisionConfig;

/// Every tensor [`DeepseekOcr2VisionConfig::param_list`] names, at a fresh
/// seeded value.
pub fn init_weights(cfg: &DeepseekOcr2VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let std = 0.02f32;
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let v = if name.ends_with("norm1.weight") || name.ends_with("norm2.weight") || name.ends_with("vision.encoder.norm.weight") {
            vec![1.0f32; numel]
        } else if name.ends_with(".lora_b") {
            vec![0.0f32; numel] // zero-init so a fresh adapter starts as an exact no-op
        } else {
            rng.vec_scaled(numel, std)
        };
        w.insert(name, v);
    }
    w
}

/// Just the `.lora_a`/`.lora_b` tensors [`init_weights`] would produce for
/// `cfg.lora`'s targets - what a caller merges over a REAL checkpoint's own
/// weight map (which never carries them; LoRA is trained after import)
/// rather than re-deriving fresh values for tensors the checkpoint already
/// has. Mirrors `deepseek2::init::init_adapters` exactly. Empty when
/// `cfg.lora` is `None`.
pub fn init_adapters(cfg: &DeepseekOcr2VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    init_weights(cfg, seed).into_iter().filter(|(n, _)| model::adapter::device::is_adapter_param(n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{lora_cfg, Qwen2EncoderConfig};
    use sam1::SamViTConfig;

    fn tiny(lora: Option<qwen3::LoraCfg>) -> DeepseekOcr2VisionConfig {
        let encoder = Qwen2EncoderConfig {
            d_model: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ffn_hidden: 11,
            rms_eps: model::block::RMSNORM_EPS,
            rope_theta: 10_000.0,
            n_query_local: 3,
            n_query_global: 5,
        };
        let sam = SamViTConfig { compress_out: 8, ..SamViTConfig::tiny() };
        DeepseekOcr2VisionConfig { sam, encoder, decoder_hidden: 6, lora }
    }

    /// Every planned tensor is produced, at exactly the planned size.
    #[test]
    fn init_weights_covers_every_planned_tensor_at_the_right_size() {
        let cfg = tiny(None);
        let w = init_weights(&cfg, 1);
        for (name, numel) in cfg.param_list() {
            assert_eq!(w.get(&name).map(Vec::len), Some(numel), "{name}: missing or wrong size");
        }
        assert_eq!(w.len(), cfg.param_list().len());
    }

    /// Norm gains start at exactly 1.0, never noise.
    #[test]
    fn norm_gains_start_at_one() {
        let cfg = tiny(None);
        let w = init_weights(&cfg, 2);
        assert!(w["vision.encoder.norm.weight"].iter().all(|&x| x == 1.0));
        assert!(w["vision.encoder.blocks.0.norm1.weight"].iter().all(|&x| x == 1.0));
    }

    /// With LoRA configured, `B` is degenerately zero and `A` is not -
    /// otherwise a fresh adapter's own gradient check would be starved on
    /// both sides at once.
    #[test]
    fn init_adapters_covers_only_the_lora_pair_b_zero_a_nonzero() {
        let base = tiny(None);
        assert!(init_adapters(&base, 7).is_empty(), "no lora configured -- there is nothing to init");

        let cfg = tiny(Some(lora_cfg(2, 4.0)));
        let adapters = init_adapters(&cfg, 7);
        assert!(!adapters.is_empty());
        for (name, v) in &adapters {
            if name.ends_with(".lora_b") {
                assert!(v.iter().all(|&x| x == 0.0), "{name} (lora_b) must start at exact zero");
            } else if name.ends_with(".lora_a") {
                assert!(v.iter().any(|&x| x != 0.0), "{name} (lora_a) must not be degenerately zero");
            }
        }
    }
}
