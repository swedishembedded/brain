// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapter-only save/load for a trained Qwen3.5/3.8-27B (dense) LoRA.
//!
//! `Qwen35::save` (see `model.rs`) writes the WHOLE param store - frozen base
//! plus `.lora_a`/`.lora_b` - as a full-size checkpoint. That is correct for
//! resuming training, but wasteful for distribution and serving: a rank-8
//! adapter over the 12 targeted projections (GDN's `in_proj_qkv`/`in_proj_z`/
//! `in_proj_b`/`in_proj_a`/`out_proj`, GQA's `q_proj`/`k_proj`/`v_proj`/
//! `o_proj`, the dense MLP's `gate`/`up`/`down` - see `crate::config::
//! lora_targets`) is a few million parameters against a multi-billion
//! parameter base. This module writes just the adapter tensors (with a
//! `ModelCard` describing them, so `model_dir::register` can catalog the
//! adapter as its own selectable model id) and can fold them into an
//! already-loaded base's weights for inference - `W_eff = W +
//! (alpha/rank)*B*A`, applied once at load, so the forward pass pays zero
//! extra cost versus the unadapted base.
//!
//! The actual save/fold I/O and `fold_delta` math live once, generically, in
//! `model::lora::device_adapter` (self-improve roadmap P4 - the same
//! primitive `qwen3::lora`/`qwen35moe::lora`/`deepseek2::lora` already wrap).
//! This module is the fourth, and last missing, thin wrapper: this crate's
//! own `LoraCfg` type (re-exported from `qwen3`, see `crate::config`) and the
//! `"qwen35"` family tag - must not collide with the MoE sibling's own
//! `"qwen35moe"` family, `crates/qwen35moe`. The one genuine architectural
//! difference from `qwen35moe::lora`: this model's MLP is plain dense SwiGLU
//! (no router, no experts), so its 12 LoRA-targetable leaves include
//! `gate`/`up`/`down` directly - `qwen35moe` deliberately excludes its
//! 256-expert MoE linears from LoRA and only targets the 9 GDN/GQA leaves.
//! `fold_adapter_into` itself needs no knowledge of this: it only reads
//! whichever `.lora_a`/`.lora_b` names the adapter file actually carries.
//!
//! Swedish Embedded AB implements LoRA adapter folding for edge-deployed LLM
//! serving. If your team needs expertise in on-device model fine-tuning or
//! efficient adapter distribution, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::HashMap;

use crate::config::LoraCfg;
use crate::model::Qwen35;

/// Write only this model's `.lora_a`/`.lora_b` tensors - never the frozen
/// base - to `path`. See `model::lora::device_adapter::save_adapter`.
pub fn save_adapter(path: &str, model: &Qwen35, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
    let lora = model
        .cfg
        .lora
        .as_ref()
        .unwrap_or_else(|| panic!("save_adapter: model was not built with a LoraCfg"));
    model::lora::device_adapter::save_adapter(path, model, lora.rank, lora.alpha, &lora.targets, card_id, base_id, "qwen35", dataset_id)
}

/// Fold an adapter saved by [`save_adapter`] into a base model's host tensor
/// map (name -> row-major `[out, in]` data), in place. `base` must already
/// contain every targeted linear's weight under its plain name (e.g.
/// `blocks.0.linear_attn.in_proj_qkv.weight`, `blocks.3.self_attn.q_proj.weight`,
/// `blocks.0.mlp.gate.weight`) - exactly what `checkpoint::load(weights)
/// .by_role("")` returns, so a caller can fold straight into the map that
/// feeds `Qwen35::new_on`/`new_i8`/`new_on_dt` with zero other changes. See
/// `model::lora::device_adapter::fold_adapter_into`.
pub fn fold_adapter_into(base: &mut HashMap<String, Vec<f32>>, adapter_path: &str) -> std::io::Result<LoraCfg> {
    let (rank, alpha) = model::lora::device_adapter::fold_adapter_into(base, adapter_path)?;
    Ok(LoraCfg { rank, alpha, targets: vec![] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::st::{Adapter, ModelCard};

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-qwen35-lora-unit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Hand-build an adapter safetensors file the way [`save_adapter`] would,
    /// but with fully deterministic, easy-to-hand-check `A`/`B` values -
    /// no GPU, no training, so this test runs everywhere `cargo test` does.
    fn write_synthetic_adapter(path: &std::path::Path, leaf: &str, rank: u32, alpha: f32, a: Vec<f32>, b: Vec<f32>) {
        let mut card = ModelCard::new("test-adapter", "qwen35");
        card.variant_of = Some("test-base".to_string());
        card.adapter = Some(Adapter {
            kind: "lora".to_string(),
            rank: Some(rank),
            base: Some("test-base".to_string()),
            alpha: Some(alpha),
            targets: Some(vec![leaf.to_string()]),
            dataset_id: None,
        });
        let tensors = vec![
            (format!("{leaf}.lora_a"), vec![a.len() as u64], a),
            (format!("{leaf}.lora_b"), vec![b.len() as u64], b),
        ];
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), Some(&card)).expect("write synthetic adapter");
    }

    /// RED-first spec: `fold_adapter_into` must apply EXACTLY
    /// `W += (alpha/rank)*B@A` to every targeted leaf, and must leave every
    /// OTHER leaf in the base map bit-identical - a fold that touches leaves
    /// it should not (or gets the scale/transpose wrong) is a silent
    /// correctness bug a "did anything change" smoke test would miss.
    #[test]
    fn fold_adapter_into_applies_exactly_scale_times_b_at_a_and_touches_nothing_else() {
        let dir = tmp("math");

        // A [r=2, in=3], B [out=4, r=2] - a small, hand-picked, non-trivial pair.
        let r = 2usize;
        let (out, inn) = (4usize, 3usize);
        let a: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5, -0.5, 2.0]; // [2,3]
        let b: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 2.0, -1.0, -3.0, 0.5]; // [4,2]
        let (rank, alpha) = (r as u32, 4.0f32); // scale = alpha/rank = 2.0
        let scale = alpha / rank as f32;

        let adapter_path = dir.join("adapter.safetensors");
        write_synthetic_adapter(&adapter_path, "blocks.0.mlp.gate.weight", rank, alpha, a.clone(), b.clone());

        let target_before: Vec<f32> = (0..out * inn).map(|i| i as f32 * 0.1).collect();
        let untouched_before: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut base: HashMap<String, Vec<f32>> = HashMap::new();
        base.insert("blocks.0.mlp.gate.weight".to_string(), target_before.clone());
        base.insert("blocks.0.other.weight".to_string(), untouched_before.clone());

        let cfg = fold_adapter_into(&mut base, adapter_path.to_str().unwrap()).expect("fold_adapter_into");
        assert_eq!(cfg.rank, rank);
        assert_eq!(cfg.alpha, alpha);

        // Expected: W[o,i] += scale * sum_k B[o,k]*A[k,i], row-major [out,in]/[out,r]/[r,in].
        let mut expected = target_before.clone();
        for o in 0..out {
            for i in 0..inn {
                let mut acc = 0.0f32;
                for k in 0..r {
                    acc += b[o * r + k] * a[k * inn + i];
                }
                expected[o * inn + i] += scale * acc;
            }
        }
        let got = base.get("blocks.0.mlp.gate.weight").unwrap();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-6, "folded weight disagrees with base + (alpha/rank)*B@A: got {got:?}, expected {expected:?}");
        }

        // The untouched leaf must be BIT-IDENTICAL - never merely "close".
        assert_eq!(base.get("blocks.0.other.weight").unwrap(), &untouched_before, "fold_adapter_into modified a leaf the adapter never targeted");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `fold_adapter_into` must refuse to fold silently if the base map is
    /// missing a leaf the adapter targets - a missing leaf means the caller
    /// loaded the wrong base checkpoint, and a silent skip there would serve
    /// a model that looks adapted but isn't.
    #[test]
    #[should_panic(expected = "base has no weight named")]
    fn fold_adapter_into_panics_on_a_target_missing_from_the_base_map() {
        let dir = tmp("missing-leaf");
        let adapter_path = dir.join("adapter.safetensors");
        write_synthetic_adapter(&adapter_path, "blocks.0.mlp.gate.weight", 2, 4.0, vec![0.0; 6], vec![0.0; 8]);

        let mut base: HashMap<String, Vec<f32>> = HashMap::new();
        base.insert("blocks.0.other.weight".to_string(), vec![1.0, 2.0]);
        let _ = fold_adapter_into(&mut base, adapter_path.to_str().unwrap());
    }
}
