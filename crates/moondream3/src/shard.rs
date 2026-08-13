// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream MoE expert sharding.
//!
//! The decoder's expert weights use the `blocks.<L>.moe.experts.<E>.…` naming that
//! [`federated::expert_id`](https://docs.rs/brain-federated) recognizes, so a
//! Moondream MoE checkpoint federates — `split` peels each expert into its own
//! hash-verified shard (the router and all dense/attention/vision tensors stay in
//! `shared.safetensors`), a worker trains one expert in isolation, and `assemble` folds
//! the overlay back — with **no Moondream-specific sharding code**. This module
//! documents that weave and enumerates the shardable expert tensors for a config.

use crate::config::MoondreamConfig;
use crate::import::moe_layer_keys;

/// All `blocks.<L>.moe.…` tensor keys across the MoE layers of `cfg` (router +
/// per-expert `w_h`/`w_g`/`w_down`). `federated::split` routes the router into
/// `shared.safetensors` and each `experts.<E>.…` tensor into that expert's shard.
pub fn moe_tensor_keys(cfg: &MoondreamConfig) -> Vec<String> {
    (0..cfg.n_layers).filter(|&l| cfg.is_moe_layer(l)).flat_map(|l| moe_layer_keys(l, cfg.moe.num_experts)).collect()
}

/// The number of independently-shardable experts per MoE layer.
pub fn experts_per_layer(cfg: &MoondreamConfig) -> u32 {
    cfg.moe.num_experts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_expert_key_count() {
        let cfg = MoondreamConfig::preview();
        // 20 MoE layers × (1 router + 64 experts × 3 tensors) = 20 × 193 = 3860.
        let keys = moe_tensor_keys(&cfg);
        assert_eq!(keys.len(), 20 * (1 + 64 * 3));
        assert!(keys.iter().any(|k| k == "blocks.4.moe.experts.0.w_h.weight"));
        assert!(keys.iter().any(|k| k == "blocks.23.moe.experts.63.w_down.weight"));
        // Every expert tensor is federated-recognizable; the router is not (shared).
        assert_eq!(federated::expert_id("blocks.4.moe.experts.7.w_h.weight"), Some(7));
        assert_eq!(federated::expert_id("blocks.4.moe.router.weight"), None);
    }

    #[test]
    fn moe_checkpoint_split_assemble_roundtrips() {
        let dir = std::env::temp_dir().join("brain_md_shard_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.safetensors");
        let base_s = base.to_str().unwrap();

        // A small Moondream-shaped MoE checkpoint: shared (tok/ln/router) + 2 experts
        // on layer 4, using the real w_h/w_g/w_down expert naming.
        let (d, inner) = (4usize, 3usize);
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("tok.weight".into(), vec![5, d as u64], vec![0.5; 5 * d]),
            ("blocks.4.ln.weight".into(), vec![d as u64], vec![1.0; d]),
            ("blocks.4.moe.router.weight".into(), vec![2, d as u64], vec![0.1; 2 * d]),
        ];
        for e in 0..2u32 {
            let v = e as f32 + 1.0;
            tensors.push((format!("blocks.4.moe.experts.{e}.w_h.weight"), vec![inner as u64, d as u64], vec![v; inner * d]));
            tensors.push((format!("blocks.4.moe.experts.{e}.w_g.weight"), vec![inner as u64, d as u64], vec![v * 2.0; inner * d]));
            tensors.push((format!("blocks.4.moe.experts.{e}.w_down.weight"), vec![d as u64, inner as u64], vec![v * 3.0; d * inner]));
        }
        checkpoint::save(base_s, serde_json::json!({"arch": "moondream"}), &tensors);

        // split → verify → merge_to_full, then assert byte-for-byte tensor identity.
        let split_dir = dir.join("split");
        let manifest = federated::split(base_s, &split_dir).unwrap();
        assert_eq!(manifest.experts, vec![0, 1]);
        federated::verify(&split_dir).unwrap();
        let merged = dir.join("merged.safetensors");
        federated::merge_to_full(&split_dir, merged.to_str().unwrap()).unwrap();

        let orig = checkpoint::load(base_s);
        let back = checkpoint::load(merged.to_str().unwrap());
        assert_eq!(orig.tensors.len(), back.tensors.len());
        for t in &orig.tensors {
            let b = back.tensors.iter().find(|x| x.name == t.name).unwrap_or_else(|| panic!("missing {}", t.name));
            assert_eq!(t.data, b.data, "tensor {} changed across split/merge", t.name);
        }

        // Single-expert overlay flow: a worker returns only expert 1.
        let overlay = dir.join("overlay1");
        let om = federated::split_filtered(base_s, &overlay, Some(&[1])).unwrap();
        assert_eq!(om.experts, vec![1]);
        let assembled = dir.join("assembled.safetensors");
        federated::assemble(&split_dir, &[&overlay], assembled.to_str().unwrap()).unwrap();
        let asm = checkpoint::load(assembled.to_str().unwrap());
        assert!(asm.tensors.iter().any(|t| t.name == "blocks.4.moe.experts.1.w_down.weight"));
    }
}
