// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `omni::import::hf_to_brain` against EVERY one of the real checkpoint's
//! 28 010 tensor names (not the hand-picked shape samples in
//! `import.rs`'s own unit tests) — the strongest coverage check available
//! without the weight bytes themselves, since `model.safetensors.index.json`
//! lists every name up front.
//!
//! Real-weight-adjacent, so it follows the engine's opt-in-env-var pattern:
//! skips cleanly when the checkpoint dir is not present.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test real_import_coverage -- --ignored`

use std::collections::HashSet;
use std::path::PathBuf;

fn hf_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("BRAIN_OMNI_HF_DIR").ok()?);
    d.join("model.safetensors.index.json").exists().then_some(d)
}

/// The audio-tower q/k/v leaves are consumed by `fuse_audio_qkv`, not
/// `hf_to_brain` directly (see `import.rs`'s doc) — the only names in the
/// whole real checkpoint this test expects `hf_to_brain` to legitimately
/// reject.
fn is_qkv_fuse_leaf(name: &str) -> bool {
    name.starts_with("thinker.audio_tower.layers.")
        && (name.ends_with("self_attn.q_proj.weight")
            || name.ends_with("self_attn.k_proj.weight")
            || name.ends_with("self_attn.v_proj.weight")
            || name.ends_with("self_attn.q_proj.bias")
            || name.ends_with("self_attn.k_proj.bias")
            || name.ends_with("self_attn.v_proj.bias"))
}

#[test]
#[ignore]
fn every_real_tensor_name_maps_or_is_a_known_qkv_leaf() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset or model.safetensors.index.json missing");
        return;
    };
    let idx_json = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("read index");
    let idx: serde_json::Value = serde_json::from_str(&idx_json).expect("parse index");
    let weight_map = idx["weight_map"].as_object().expect("weight_map object");
    assert_eq!(weight_map.len(), 28010, "index tensor count drifted from the recorded fact (docs/models/omni/status.md)");

    let mut unmapped = Vec::new();
    let mut seen_brain_names: HashSet<String> = HashSet::new();
    let mut collisions = Vec::new();
    for name in weight_map.keys() {
        if is_qkv_fuse_leaf(name) {
            continue;
        }
        match omni::import::hf_to_brain(name) {
            Some(bn) => {
                if !seen_brain_names.insert(bn.clone()) {
                    collisions.push(bn);
                }
            }
            None => unmapped.push(name.clone()),
        }
    }

    assert!(
        unmapped.is_empty(),
        "{} real tensor names have no mapping (first 20: {:?})",
        unmapped.len(),
        &unmapped[..unmapped.len().min(20)]
    );
    assert!(collisions.is_empty(), "two HF tensors mapped to the same brain name: {collisions:?}");

    println!("omni::import::hf_to_brain covers all {} non-qkv-leaf real tensor names, no collisions.", weight_map.len());
}
