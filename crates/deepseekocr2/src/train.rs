// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA training glue for the composite - the two-tower analogue of
//! `deepseek2ocr::train`.
//!
//! Both towers' adapter mechanisms live one layer down and need no change
//! from this file: `deepseek2::config::DeepseekV2Config::lora` on the
//! decoder, `crate::config::DeepseekOcr2VisionConfig::lora` on the
//! resampler. `crate::model::DeepseekOcr2::new_on` already takes both
//! configs separately and threads each straight into its own tower's role
//! assignment, so building a LoRA-adapted composite needs no change there
//! either - only the init map needs the two tensor families a real
//! checkpoint (or a checkpoint-free fixture) never carries.
//!
//! Swedish Embedded AB builds from-scratch GPU training/inference stacks for
//! vision-language models. If your team needs LoRA wired across a
//! multi-tower composite, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::HashMap;

use deepseek2::DeepseekV2Config;

use crate::config::DeepseekOcr2VisionConfig;

/// `base` overlaid with fresh `.lora_a`/`.lora_b` tensors for whichever of
/// `vision_cfg.lora`/`decoder_cfg.lora` is set (either, or both - a caller
/// may adapt just the new vision tower, just the decoder, or both at
/// possibly different ranks, since the two configs are independent). Panics
/// if NEITHER is set (nothing to add), or if `base` already carries one of
/// the tensor names being added (a real checkpoint and a fresh init should
/// never collide; a collision here means a config was built against the
/// wrong base).
pub fn lora_init_map(vision_cfg: &DeepseekOcr2VisionConfig, decoder_cfg: &DeepseekV2Config, base: &HashMap<String, Vec<f32>>, seed: u64) -> HashMap<String, Vec<f32>> {
    assert!(vision_cfg.lora.is_some() || decoder_cfg.lora.is_some(), "lora_init_map: neither tower has a LoraCfg -- nothing to add");
    let mut init = base.clone();
    let mut merge = |name: String, data: Vec<f32>| {
        assert!(init.insert(name.clone(), data).is_none(), "{name}: base already carries an adapter tensor -- checkpoint and fresh init collided");
    };
    if vision_cfg.lora.is_some() {
        for (name, data) in crate::init::init_adapters(vision_cfg, seed) {
            merge(name, data);
        }
    }
    if decoder_cfg.lora.is_some() {
        // A distinct seed offset from the vision half's: the two towers'
        // adapter tensors never share a name (disjoint `vision.*` vs
        // `blocks.*` namespaces), so this is only to avoid the same LCG
        // stream producing coincidentally-identical `A` noise across towers.
        for (name, data) in deepseek2::init::init_adapters(decoder_cfg, seed ^ 0x5eed_5eed) {
            merge(name, data);
        }
    }
    init
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{lora_cfg, Qwen2EncoderConfig};
    use sam1::SamViTConfig;

    fn tiny_vision(lora: bool) -> DeepseekOcr2VisionConfig {
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
        DeepseekOcr2VisionConfig { sam, encoder, decoder_hidden: DeepseekV2Config::tiny().shape.d_model, lora: lora.then(|| lora_cfg(2, 4.0)) }
    }

    /// Adapting BOTH towers at once merges both families and touches
    /// nothing already in `base` - the two-tower generalization of
    /// `deepseek2ocr::train`'s own single-tower round-trip test.
    #[test]
    fn lora_init_map_adds_both_towers_adapters_and_leaves_base_untouched() {
        let vision_cfg = tiny_vision(true);
        let mut decoder_cfg = DeepseekV2Config::tiny();
        decoder_cfg.lora = Some(deepseek2::config::lora_cfg(2, 4.0));

        let base: HashMap<String, Vec<f32>> = crate::init::init_weights(&tiny_vision(false), 1)
            .into_iter()
            .chain(deepseek2::init_weights(&{
                let mut c = decoder_cfg.clone();
                c.lora = None;
                c
            }, 1))
            .collect();

        let vision_adapters = crate::init::init_adapters(&vision_cfg, 9);
        let decoder_adapters = deepseek2::init::init_adapters(&decoder_cfg, 9 ^ 0x5eed_5eed);
        assert!(!vision_adapters.is_empty());
        assert!(!decoder_adapters.is_empty());

        let merged = lora_init_map(&vision_cfg, &decoder_cfg, &base, 9);
        assert_eq!(merged.len(), base.len() + vision_adapters.len() + decoder_adapters.len());
        for (name, data) in &base {
            assert_eq!(merged.get(name), Some(data), "{name}: base tensor was touched by the merge");
        }
        for (name, data) in vision_adapters.iter().chain(&decoder_adapters) {
            assert_eq!(merged.get(name), Some(data), "{name}: adapter tensor was not merged verbatim");
        }
    }

    /// Building the map against two configs with no LoRA at all is refused
    /// loudly, the same as `deepseek2ocr::train`'s own refusal.
    #[test]
    #[should_panic(expected = "neither tower has a LoraCfg")]
    fn lora_init_map_refuses_two_non_lora_configs() {
        let vision_cfg = tiny_vision(false);
        let decoder_cfg = DeepseekV2Config::tiny();
        let _ = lora_init_map(&vision_cfg, &decoder_cfg, &HashMap::new(), 0);
    }
}
