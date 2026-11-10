// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The canonical tensor manifest derived from the configuration.

use std::collections::BTreeSet;

use mirage::config::MirageConfig;
use mirage::spec::{is_training_only, Manifest};

use crate::common::RELEASED_ARCHITECTURE_JSON;

fn released() -> Manifest {
    Manifest::new(&MirageConfig::from_json(RELEASED_ARCHITECTURE_JSON).unwrap())
}

#[test]
fn the_released_config_yields_583_tensors_and_54_078_804_parameters() {
    let m = released();
    assert_eq!(m.len(), 583);
    assert_eq!(m.param_count(), 54_078_804);
}

#[test]
fn flax_names_and_shapes_are_the_checkpoints() {
    let m = released();
    let shape = |name: &str| m.get(name).unwrap_or_else(|| panic!("manifest lacks {name}")).shape.clone();
    assert_eq!(shape("body_3/hconv/gate_conv"), [512, 65]);
    assert_eq!(shape("body_15/mix/dw_gate"), [3, 3, 1, 512]);
    assert_eq!(shape("stem/w0"), [3, 3, 3, 32]);
    assert_eq!(shape("stem/dw_4"), [3, 3, 1, 256]);
    assert_eq!(shape("stem/pw_4/kernel"), [256, 512]);
    assert_eq!(shape("top_11/self_attn/q_proj/kernel"), [512, 128]);
    assert_eq!(shape("top_0/self_attn/out_proj/kernel"), [256, 512]);
    assert_eq!(shape("level_gates"), [12, 512]);
    assert_eq!(shape("registers"), [4, 512]);
    assert_eq!(shape("dense/fine/kernel"), [128, 128]);
    assert_eq!(shape("dense/mid/kernel"), [256, 128]);
    assert_eq!(shape("dense/coarse/kernel"), [512, 128]);
    assert_eq!(shape("dense/edge/kernel"), [64, 128]);
    assert_eq!(shape("dense/out/kernel"), [128, 10]);
    assert_eq!(shape("slot_proj/kernel"), [192, 512]);
    assert_eq!(shape("reader/w0"), [3, 3, 3, 32]);
    assert_eq!(shape("reader/mix_3"), [1, 5, 1, 384]);
    assert_eq!(shape("reader/out/kernel"), [384, 307]);
    assert_eq!(shape("embedding/embedding"), [8299, 512]);
    assert_eq!(shape("engrams_1/embedding"), [4, 18432, 128]);
    assert_eq!(shape("engrams_0/taps"), [4, 512]);
    assert_eq!(shape("stack/layers/block/self_attn/q_proj/kernel"), [8, 512, 384]);
    assert_eq!(shape("stack/layers/block/self_attn/k_taps"), [8, 3, 96]);
    assert_eq!(shape("stack/layers/block/cross_attn/k_proj/kernel"), [8, 512, 384]);
    assert_eq!(shape("stack/layers/block/hadamard_mlp/w1a"), [8, 16, 16]);
    assert_eq!(shape("stack/layers/block/hadamard_mlp/w1b"), [8, 32, 32]);
    assert_eq!(shape("stack/layers/block/hadamard_mlp/cond_u"), [8, 8, 512]);
    assert_eq!(shape("stack/layers/block/attn_gate"), [8]);
    assert_eq!(shape("stack/mhc_phi_res"), [8, 2048, 16]);
    assert_eq!(shape("stack/mhc_b_res"), [8, 4, 4]);
    assert_eq!(shape("mtp_block/attn_gate"), [1]);
    assert_eq!(shape("mtp_combine/kernel"), [1024, 512]);
    assert_eq!(shape("name_head/kernel"), [512, 512]);
    assert_eq!(shape("pe_gate"), [1]);
}

#[test]
fn exactly_the_training_only_tensors_are_marked() {
    let m = released();
    let training: BTreeSet<&str> = m.iter().filter(|t| t.training_only).map(|t| t.name.as_str()).collect();
    assert_eq!(training.len(), 41, "37 in mtp_block plus mtp_combine, mtp_emb_norm, mtp_final_norm, name_head");
    for name in ["mtp_combine/kernel", "mtp_emb_norm/scale", "mtp_final_norm/scale", "name_head/kernel", "mtp_block/cross_attn/q_proj/kernel"] {
        assert!(training.contains(name), "{name}");
        assert!(is_training_only(name), "{name}");
    }
    assert!(training.iter().all(|n| n.starts_with("mtp_") || *n == "name_head/kernel"));
    assert!(!is_training_only("stack/final_norm/scale"));
    assert_eq!(m.deployment().count(), 583 - 41);
}

#[test]
fn names_are_unique_and_iterate_sorted() {
    let m = released();
    let names: Vec<&str> = m.iter().map(|t| t.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted);
}

#[test]
fn the_tiny_config_has_every_family_at_small_size() {
    let c = MirageConfig::tiny();
    let m = Manifest::new(&c);
    assert!(m.param_count() < 200_000, "{}", m.param_count());
    assert!(m.get("engrams_0/embedding").is_some());
    assert!(m.get("engrams_1/embedding").is_none(), "one site");
    assert!(m.get("body_1/grn/gamma").is_some() && m.get("body_2/grn/gamma").is_none());
    assert_eq!(m.get("stack/mhc_phi_pre").unwrap().shape, [2, 64, 2]);
    assert_eq!(m.iter().map(|t| t.numel()).sum::<usize>(), m.param_count());
}
