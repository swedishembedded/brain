// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapter-only save/load for a trained Qwen LoRA.
//!
//! `Qwen::save` (see `model.rs`) writes the WHOLE param store -- base weights
//! plus `.lora_a`/`.lora_b` -- as a full-size checkpoint. That is correct for
//! resuming training, but wasteful for distribution and serving: a rank-8
//! adapter on Qwen3-0.6B is a few MB against a ~2.4 GB fp32 base. This module
//! writes just the adapter tensors (with a `ModelCard` describing them, so
//! `model_dir::register` can catalog the adapter as its own selectable model
//! id) and can fold them into an already-loaded base's weights for
//! inference -- `W_eff = W + (alpha/rank)*B*A`, applied once at load, so the
//! forward pass pays zero extra cost versus the unadapted base.

use std::collections::HashMap;

use checkpoint::st::{Adapter, ModelCard};

use crate::config::LoraCfg;
use crate::model::Qwen;

/// Write only this model's `.lora_a`/`.lora_b` tensors -- never the frozen
/// base -- to `path`, carrying a `ModelCard` with `variant_of: base_id` and
/// an `Adapter` descriptor (rank/alpha/targets/dataset_id) so the adapter is
/// discoverable and reloadable without the base's shape being re-derived by
/// guesswork.
pub fn save_adapter(path: &str, model: &Qwen, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
    let lora = model
        .cfg
        .lora
        .as_ref()
        .unwrap_or_else(|| panic!("save_adapter: model was not built with a LoraCfg"));

    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = model
        .ps
        .params
        .iter()
        .filter(|(name, _)| name.ends_with(".lora_a") || name.ends_with(".lora_b"))
        .map(|(name, _)| (name.clone(), vec![model.ps.numel(name) as u64], model.read_weight(name)))
        .collect();
    assert!(!tensors.is_empty(), "save_adapter: no .lora_a/.lora_b tensors in the param store");

    let mut card = ModelCard::new(card_id, "qwen");
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
/// `blocks.0.attn.wq.weight`); this only reads the `.lora_a`/`.lora_b` pair
/// alongside it and adds the low-rank delta.
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
/// `[out,r]`, both row-major -- the same convention `zimage::lora::Pair`
/// uses, and what `qwen3::model::Qwen::lora_fwd`'s unfolded forward computes.
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
