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

fn write(name: &str, card_targets: Option<&[&str]>, tensor_bases: &[&str]) -> String {
    let path = std::env::temp_dir().join(format!("brain-adapter-contract-{name}-{}.safetensors", std::process::id()));
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
    let path = path.to_str().unwrap().to_string();
    checkpoint::st::save_safetensors(&path, &tensors, &serde_json::json!({"rank": 2, "alpha": 4.0}), Some(&card)).unwrap();
    path
}

#[test]
fn a_card_that_says_attention_over_mlp_tensors_is_refused_with_both_lists() {
    let path = write("mismatch", Some(&["wq", "wk"]), &["blocks.0.mlp.gate.weight", "blocks.0.mlp.up.weight"]);
    let err = read_adapter(&path).err().expect("refused").to_string();
    for needed in ["wq", "wk", "gate", "up"] {
        assert!(err.contains(needed), "the diagnostic names {needed}: {err}");
    }
    assert!(err.contains("card") && err.contains("tensors"), "{err}");
}

#[test]
fn a_card_naming_a_target_with_no_tensors_is_refused() {
    let path = write("missing", Some(&["wq", "wv"]), &["blocks.0.attn.wq.weight"]);
    let err = read_adapter(&path).err().expect("refused").to_string();
    assert!(err.contains("wv"), "{err}");
}

#[test]
fn a_card_that_matches_its_tensors_loads_and_a_card_without_targets_still_loads() {
    let ok = write("match", Some(&["wq", "wv"]), &["blocks.0.attn.wq.weight", "blocks.1.attn.wq.weight", "blocks.0.attn.wv.weight"]);
    assert_eq!(read_adapter(&ok).unwrap().sites.len(), 3);
    let legacy = write("legacy", None, &["blocks.0.attn.wq.weight"]);
    assert_eq!(read_adapter(&legacy).unwrap().sites.len(), 1);
}
