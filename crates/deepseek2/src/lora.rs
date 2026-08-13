// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapter-only save/load for a trained DeepSeek-V2 LoRA -- a direct port of
//! `qwen3::lora`'s design (`qwen35moe::lora` is the same port again, for its
//! own decoder), onto [`DeepseekV2`].
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

use std::collections::HashMap;

use checkpoint::st::{Adapter, ModelCard};

use crate::config::LoraCfg;
use crate::model::DeepseekV2;

/// Write only this model's `.lora_a`/`.lora_b` tensors -- never the frozen
/// base -- to `path`, carrying a `ModelCard` with `variant_of: base_id` and
/// an `Adapter` descriptor (rank/alpha/targets/dataset_id) so the adapter is
/// discoverable and reloadable without the base's shape being re-derived by
/// guesswork.
pub fn save_adapter(path: &str, model: &DeepseekV2, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
    let lora = model
        .cfg
        .lora
        .as_ref()
        .unwrap_or_else(|| panic!("save_adapter: model was not built with a LoraCfg"));

    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = model
        .param_names()
        .into_iter()
        .filter(|name| name.ends_with(".lora_a") || name.ends_with(".lora_b"))
        .map(|name| {
            let data = model.read_weight(&name);
            (name.clone(), vec![data.len() as u64], data)
        })
        .collect();
    assert!(!tensors.is_empty(), "save_adapter: no .lora_a/.lora_b tensors in the param store");

    let mut card = ModelCard::new(card_id, "deepseekv2");
    card.variant_of = Some(base_id.to_string());
    card.adapter = Some(Adapter {
        kind: "lora".to_string(),
        rank: Some(lora.rank),
        base: Some(base_id.to_string()),
        alpha: Some(lora.alpha),
        targets: Some(lora.targets.clone()),
        dataset_id: dataset_id.map(str::to_string),
    });

    let config = serde_json::json!({
        "rank": lora.rank, "alpha": lora.alpha, "targets": lora.targets,
    });
    checkpoint::st::save_safetensors(path, &tensors, &config, Some(&card))
}

/// Fold an adapter saved by [`save_adapter`] into a base model's host tensor
/// map (name -> row-major `[out, in]` data), in place. `base` must already
/// contain every targeted linear's weight under its plain name (e.g.
/// `blocks.0.self_attn.q_proj.weight`); this only reads the `.lora_a`/
/// `.lora_b` pair alongside it and adds the low-rank delta.
pub fn fold_adapter_into(base: &mut HashMap<String, Vec<f32>>, adapter_path: &str) -> std::io::Result<LoraCfg> {
    let st = checkpoint::st::load_safetensors(adapter_path)?;
    let card = st
        .card()
        .unwrap_or_else(|| panic!("fold_adapter_into: {adapter_path} has no ModelCard"));
    let a = card.adapter.as_ref().unwrap_or_else(|| panic!("fold_adapter_into: {adapter_path}'s card has no adapter descriptor"));
    let rank = a.rank.unwrap_or_else(|| panic!("fold_adapter_into: {adapter_path}'s adapter has no rank"));
    let alpha = a.alpha.unwrap_or(rank as f32);
    let scale = alpha / rank as f32;

    let mut names: Vec<&str> = st
        .tensors
        .keys()
        .filter_map(|n| n.strip_suffix(".lora_a"))
        .collect();
    names.sort();
    for base_name in names {
        let a_name = format!("{base_name}.lora_a");
        let b_name = format!("{base_name}.lora_b");
        let a_data = st.tensors.get(&a_name).unwrap_or_else(|| panic!("{adapter_path}: missing {a_name}"));
        let b_data = st.tensors.get(&b_name).unwrap_or_else(|| panic!("{adapter_path}: missing {b_name}"));
        let w = base
            .get_mut(base_name)
            .unwrap_or_else(|| panic!("fold_adapter_into: base has no weight named {base_name}"));
        fold_delta(w, a_data, b_data, rank as usize, scale);
    }

    Ok(LoraCfg { rank, alpha, targets: vec![] })
}

/// `W[o,i] += scale * sum_k B[o,k] * A[k,i]`, `A` is `[r,in]`, `B` is
/// `[out,r]`, both row-major -- the same convention `qwen3::lora::fold_delta`/
/// `qwen35moe::lora::fold_delta` use, and what `DeepseekV2::lora_fwd`'s
/// unfolded forward computes.
fn fold_delta(w: &mut [f32], a: &[f32], b: &[f32], r: usize, scale: f32) {
    let inn = a.len() / r;
    let out = b.len() / r;
    assert_eq!(w.len(), out * inn, "fold_delta: base weight shape does not match adapter rank/dims");
    for o in 0..out {
        let brow = &b[o * r..o * r + r];
        let wrow = &mut w[o * inn..o * inn + inn];
        for (k, &bok) in brow.iter().enumerate() {
            if bok == 0.0 {
                continue;
            }
            let bok = bok * scale;
            let arow = &a[k * inn..k * inn + inn];
            for i in 0..inn {
                wrow[i] += bok * arow[i];
            }
        }
    }
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
