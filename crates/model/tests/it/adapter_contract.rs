// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for trustworthy model adapter
// delivery for its clients. If your team needs expertise in fine-tuned model
// artifacts then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The adapter card is a contract: the `targets` it declares are the linears
//! its tensors correct. A file whose card says one thing and whose tensors
//! another is refused at load, naming both sides, rather than served as an
//! adapter that silently corrects something other than what was trained.

use checkpoint::st::{Adapter, ModelCard};
use model::lora::device_adapter::read_adapter;

/// The adapter file, removed when the returned guard drops (also when the test panics).
fn write(name: &str, card_targets: Option<&[&str]>, tensor_bases: &[&str]) -> brain_testutil::Scratch {
    let path = brain_testutil::Scratch::file(&format!("brain-adapter-contract-{name}"), ".safetensors");
    let mut card = ModelCard::new("contract", "qwen");
    card.adapter = Some(Adapter {
        kind: "lora".to_string(),
        rank: Some(2),
        alpha: Some(4.0),
        targets: card_targets.map(|t| t.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    });
    let mut tensors = Vec::new();
    for base in tensor_bases {
        tensors.push((format!("{base}.lora_a"), vec![2, 4], vec![0.1f32; 8]));
        tensors.push((format!("{base}.lora_b"), vec![4, 2], vec![0.1f32; 8]));
    }
    checkpoint::st::save_safetensors(path.as_str(), &tensors, &serde_json::json!({"rank": 2, "alpha": 4.0}), Some(&card)).unwrap();
    path
}

#[test]
fn a_card_that_says_attention_over_mlp_tensors_is_refused_with_both_lists() {
    let path = write("mismatch", Some(&["wq", "wk"]), &["blocks.0.mlp.gate.weight", "blocks.0.mlp.up.weight"]);
    let err = read_adapter(path.as_str()).err().expect("refused").to_string();
    for needed in ["wq", "wk", "gate", "up"] {
        assert!(err.contains(needed), "the diagnostic names {needed}: {err}");
    }
    assert!(err.contains("card") && err.contains("tensors"), "{err}");
}

#[test]
fn a_card_naming_a_target_with_no_tensors_is_refused() {
    let path = write("missing", Some(&["wq", "wv"]), &["blocks.0.attn.wq.weight"]);
    let err = read_adapter(path.as_str()).err().expect("refused").to_string();
    assert!(err.contains("wv"), "{err}");
}

#[test]
fn a_card_that_matches_its_tensors_loads_and_a_card_without_targets_still_loads() {
    let ok = write("match", Some(&["wq", "wv"]), &["blocks.0.attn.wq.weight", "blocks.1.attn.wq.weight", "blocks.0.attn.wv.weight"]);
    assert_eq!(read_adapter(ok.as_str()).unwrap().sites.len(), 3);
    let legacy = write("legacy", None, &["blocks.0.attn.wq.weight"]);
    assert_eq!(read_adapter(legacy.as_str()).unwrap().sites.len(), 1);
}

/// An adapter of one site whose file holds `held` as saved tensors while its card lists `listed`.
fn write_saved(name: &str, listed: Option<&[&str]>, held: &[&str]) -> brain_testutil::Scratch {
    let path = brain_testutil::Scratch::file(&format!("brain-adapter-contract-{name}"), ".safetensors");
    let mut card = ModelCard::new("contract", "qwen");
    card.adapter = Some(Adapter {
        kind: "lora".to_string(),
        rank: Some(2),
        alpha: Some(4.0),
        modules_to_save: listed.map(|l| l.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    });
    let mut tensors = vec![("blocks.0.attn.wq.weight.lora_a".to_string(), vec![2, 4], vec![0.1f32; 8]), ("blocks.0.attn.wq.weight.lora_b".to_string(), vec![4, 2], vec![0.1f32; 8])];
    for base in held {
        tensors.push((format!("{base}.saved"), vec![3], vec![0.5f32; 3]));
    }
    checkpoint::st::save_safetensors(path.as_str(), &tensors, &serde_json::json!({"rank": 2, "alpha": 4.0}), Some(&card)).unwrap();
    path
}

#[test]
fn full_tensors_are_read_as_the_card_lists_them_and_a_disagreeing_file_is_refused() {
    let ok = write_saved("saved-ok", Some(&["blocks.0.ln.weight", "blocks.0.gate"]), &["blocks.0.ln.weight", "blocks.0.gate"]);
    let adapter = read_adapter(ok.as_str()).unwrap();
    assert_eq!(adapter.sites.len(), 1);
    assert_eq!(adapter.saved.iter().map(|t| t.base.as_str()).collect::<Vec<_>>(), ["blocks.0.gate", "blocks.0.ln.weight"], "sorted by name");
    assert!(adapter.saved.iter().all(|t| t.data == vec![0.5; 3]));

    let missing = read_adapter(write_saved("saved-missing", Some(&["blocks.0.ln.weight"]), &[]).as_str()).err().expect("refused").to_string();
    assert!(missing.contains("blocks.0.ln.weight") && missing.contains("modules_to_save"), "{missing}");
    let unlisted = read_adapter(write_saved("saved-unlisted", None, &["blocks.0.ln.weight"]).as_str()).err().expect("refused").to_string();
    assert!(unlisted.contains("blocks.0.ln.weight.saved") && unlisted.contains("does not list"), "{unlisted}");
    assert!(read_adapter(write("plain", Some(&["wq"]), &["blocks.0.attn.wq.weight"]).as_str()).unwrap().saved.is_empty(), "an adapter of factors only has no full tensors");
}

/// [`write`] with the config's recorded leaves (`leaves`: base name -> the leaf of its site).
fn write_leaves(name: &str, card_targets: &[&str], tensor_bases: &[&str], leaves: serde_json::Value) -> brain_testutil::Scratch {
    let path = brain_testutil::Scratch::file(&format!("brain-adapter-contract-{name}"), ".safetensors");
    let mut card = ModelCard::new("contract", "qwen");
    card.adapter = Some(Adapter { kind: "lora".to_string(), rank: Some(2), alpha: Some(4.0), targets: Some(card_targets.iter().map(|s| s.to_string()).collect()), ..Default::default() });
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = tensor_bases.iter().flat_map(|b| [(format!("{b}.lora_a"), vec![2, 4], vec![0.1f32; 8]), (format!("{b}.lora_b"), vec![4, 2], vec![0.1f32; 8])]).collect();
    checkpoint::st::save_safetensors(path.as_str(), &tensors, &serde_json::json!({"rank": 2, "alpha": 4.0, "leaves": leaves}), Some(&card)).unwrap();
    path
}

/// A site's leaf is the model's to say: a bank of queries stored as `attn_res.attn_queries.0` is an `attn_queries` site.
/// The writer records each pair's leaf and the reader checks the card against it; a recorded map that disagrees with the
/// pairs is refused by name.
#[test]
fn recorded_leaves_decide_what_a_pair_corrects() {
    let bases = ["model.attn_res.attn_queries.0", "model.attn_res.attn_queries.1", "model.layers.0.wq"];
    let leaves = serde_json::json!({ bases[0]: "attn_queries", bases[1]: "attn_queries", bases[2]: "wq" });
    let ok = write_leaves("leaves-ok", &["attn_queries", "wq"], &bases, leaves.clone());
    assert_eq!(read_adapter(ok.as_str()).unwrap().sites.len(), 3);
    let wrong_card = read_adapter(write_leaves("leaves-card", &["0", "1", "wq"], &bases, leaves).as_str()).err().expect("refused").to_string();
    assert!(wrong_card.contains("attn_queries"), "{wrong_card}");
    let unrecorded = read_adapter(write_leaves("leaves-missing", &["attn_queries", "wq"], &bases, serde_json::json!({ bases[0]: "attn_queries", bases[2]: "wq" })).as_str()).err().expect("refused").to_string();
    assert!(unrecorded.contains(bases[1]) && unrecorded.contains("no leaf"), "{unrecorded}");
    let stray = serde_json::json!({ bases[0]: "attn_queries", bases[1]: "attn_queries", bases[2]: "wq", "model.layers.9.wk": "wk" });
    let extra = read_adapter(write_leaves("leaves-extra", &["attn_queries", "wq"], &bases, stray).as_str()).err().expect("refused").to_string();
    assert!(extra.contains("model.layers.9.wk"), "{extra}");
    let malformed = read_adapter(write_leaves("leaves-malformed", &["attn_queries", "wq"], &bases, serde_json::json!(["attn_queries"])).as_str()).err().expect("refused").to_string();
    assert!(malformed.contains("leaves"), "{malformed}");
}
