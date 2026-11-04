// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded serving of fine-tuned models for its clients. If your team needs
// expertise in serving LoRA fine-tunes of models too large to hold in host memory then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of `FoldedSource`: reading a base through an adapter folds the adapter into each targeted linear as it is
//! read, with exactly the arithmetic of `fold_adapter_into` on a host map, and leaves every other tensor alone.

use std::collections::HashMap;

use checkpoint::st::{Adapter, ModelCard};
use checkpoint::TensorSource;
use model::lora::device_adapter::{fold_adapter_into, FoldedSource};

const LEAF: &str = "blocks.0.mlp.gate.weight";

fn write_adapter(path: &std::path::Path, rank: u32, alpha: f32, a: Vec<f32>, b: Vec<f32>) {
    let mut card = ModelCard::new("test-adapter", "qwen3");
    card.variant_of = Some("test-base".to_string());
    card.adapter = Some(Adapter { kind: "lora".to_string(), rank: Some(rank), base: Some("test-base".to_string()), alpha: Some(alpha), targets: Some(vec!["gate".to_string()]), dataset_id: None, per_target: None });
    let tensors = vec![(format!("{LEAF}.lora_a"), vec![a.len() as u64], a), (format!("{LEAF}.lora_b"), vec![b.len() as u64], b)];
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), Some(&card)).expect("write adapter");
}

fn read(src: &dyn TensorSource, name: &str) -> Vec<f32> {
    let mut out = None;
    assert!(src.with_tensor(name, &mut |d| out = Some(d.to_vec())), "{name} not served");
    out.unwrap()
}

#[test]
fn a_folded_source_serves_the_fold_of_the_base_one_tensor_at_a_time() {
    let dir = std::env::temp_dir().join(format!("brain-folded-source-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapter.safetensors");
    // A [r=2, in=3], B [out=4, r=2]
    write_adapter(&path, 2, 4.0, vec![1.0, 2.0, -1.0, 0.5, -0.5, 2.0], vec![1.0, 0.0, 0.0, 1.0, 2.0, -1.0, -3.0, 0.5]);
    let base: HashMap<String, Vec<f32>> = [(LEAF.to_string(), (0..12).map(|i| i as f32 * 0.1).collect()), ("blocks.0.ln1.weight".to_string(), vec![1.0, 2.0, 3.0])].into_iter().collect();

    let mut want = base.clone();
    fold_adapter_into(&mut want, path.to_str().unwrap()).unwrap();

    let folded = FoldedSource::open(Box::new(base.clone()), path.to_str().unwrap()).expect("the adapter fits the base");
    assert_eq!(folded.rank_alpha(), (2, 4.0));
    for name in [LEAF, "blocks.0.ln1.weight"] {
        assert_eq!(read(&folded, name), want[name], "{name}");
    }
    assert_ne!(read(&folded, LEAF), base[LEAF], "the targeted linear must have moved");
    // Chunked reads see the same values; a folded linear offers no zero-copy or packed view of the stored bytes.
    let mut chunked = Vec::new();
    assert!(folded.with_tensor_chunks(LEAF, 5, &mut |_, c| chunked.extend_from_slice(c)));
    assert_eq!(chunked, want[LEAF]);
    assert!(folded.raw_words(LEAF).is_none());
    assert!(folded.raw_words("blocks.0.ln1.weight").is_some(), "an untouched tensor keeps the base's own view");
    assert!(!folded.with_tensor("absent", &mut |_| panic!("served an absent tensor")));

    // An adapter for another shape is refused when it is opened, not at the first read.
    let wrong: HashMap<String, Vec<f32>> = [(LEAF.to_string(), vec![0.0; 20])].into_iter().collect();
    let err = FoldedSource::open(Box::new(wrong), path.to_str().unwrap()).err().expect("a mismatched base is refused");
    assert!(err.to_string().contains("holds 20 values"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
