// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapter-only save/load for a trained DeepSeek-V2 LoRA.
//!
//! [`DeepseekV2::save`] writes the WHOLE param store -- frozen base plus
//! `.lora_a`/`.lora_b` -- as a full-size checkpoint. That is correct for
//! resuming training, but wasteful for distribution and serving: a rank-8
//! adapter over the four attention projections is a few hundred KB against a
//! multi-gigabyte base. This module writes just the adapter tensors (with a
//! `ModelCard` describing them, so `model_dir::register` can catalog the
//! adapter as its own selectable model id) and can fold them into an
//! already-loaded base's weights for inference -- `W_eff = W +
//! (alpha/rank)*B*A`, applied once at load, so the forward pass pays zero
//! extra cost versus the unadapted base.
//!
//! The actual save/fold I/O and `fold_delta` math live once, generically, in
//! `model::lora::device_adapter` (self-improve roadmap P4 -- this used to be
//! a near-verbatim copy of `qwen3::lora`/`qwen35moe::lora`, as this file's
//! doc comment used to say). This module is now just the thin,
//! DeepseekV2-specific wiring: this crate's own `LoraCfg` type and the
//! `"deepseekv2"` family tag.

use std::collections::HashMap;

use crate::config::LoraCfg;
use crate::model::DeepseekV2;

/// Write only this model's `.lora_a`/`.lora_b` tensors -- never the frozen
/// base -- to `path`. See `model::lora::device_adapter::save_adapter`.
pub fn save_adapter(path: &str, model: &DeepseekV2, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
    let lora = model
        .cfg
        .lora
        .as_ref()
        .unwrap_or_else(|| panic!("save_adapter: model was not built with a LoraCfg"));
    model::lora::device_adapter::save_adapter(path, model, lora.rank, lora.alpha, &lora.targets, card_id, base_id, "deepseekv2", dataset_id)
}

/// Fold an adapter saved by [`save_adapter`] into a base model's host tensor
/// map (name -> row-major `[out, in]` data), in place. `base` must already
/// contain every targeted linear's weight under its plain name (e.g.
/// `blocks.0.self_attn.q_proj.weight`). See
/// `model::lora::device_adapter::fold_adapter_into`.
pub fn fold_adapter_into(base: &mut HashMap<String, Vec<f32>>, adapter_path: &str) -> std::io::Result<LoraCfg> {
    let (rank, alpha) = model::lora::device_adapter::fold_adapter_into(base, adapter_path)?;
    Ok(LoraCfg { rank, alpha, targets: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeepseekV2Config;
    use crate::model::PIPELINES;

    /// Save then fold round-trips a trained adapter's low-rank delta onto a
    /// base weight map -- the DeepseekV2 analogue of
    /// `qwen35moe::lora::tests::save_and_fold_round_trips_a_trained_adapter`.
    #[test]
    fn save_and_fold_round_trips_a_trained_adapter() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let mut cfg = DeepseekV2Config::tiny();
        cfg.lora = Some(crate::config::lora_cfg(2, 4.0));
        let init = crate::init::init_weights(&cfg, 3);
        let base_before: HashMap<String, Vec<f32>> = init
            .iter()
            .filter(|(n, _)| !n.ends_with(".lora_a") && !n.ends_with(".lora_b"))
            .map(|(n, v)| (n.clone(), v.clone()))
            .collect();
        let t = cfg.block_size;
        let model = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1, t, &init, true);
        // Move B off its zero init so the fold actually changes something.
        for name in model.param_names() {
            if name.ends_with(".lora_b") {
                let n = model.read_weight(&name).len();
                model.write_weight(&name, &vec![0.05f32; n]);
            }
        }

        let dir = std::env::temp_dir();
        let path = dir.join(format!("deepseekv2_lora_roundtrip_{}.safetensors", std::process::id()));
        let path_str = path.to_str().unwrap();
        save_adapter(path_str, &model, "test-adapter", "test-base", None).expect("save_adapter");

        let mut folded = base_before.clone();
        fold_adapter_into(&mut folded, path_str).expect("fold_adapter_into");
        std::fs::remove_file(&path).ok();

        // At least one targeted base weight must have actually changed.
        let mut any_changed = false;
        for (name, before) in &base_before {
            if let Some(after) = folded.get(name) {
                if after != before {
                    any_changed = true;
                    break;
                }
            }
        }
        assert!(any_changed, "fold_adapter_into left every base weight unchanged");
    }
}
