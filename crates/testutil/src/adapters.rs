// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures for tests of served adapter releases: a stand-in base checkpoint
//! and a Qwen3 LoRA adapter whose card names it. Shared by the crate that
//! verifies a release and the one that swaps it in under load.

use std::path::{Path, PathBuf};

use checkpoint::st::{Adapter, ModelCard, TrainingProvenance};

/// A fresh, empty scratch directory unique to this process and `name`.
pub fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-adapter-release-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A stand-in base file: verification reads only its bytes.
pub fn write_base(path: &Path, fill: f32) {
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("embed.weight".to_string(), vec![4], vec![fill; 4])], &serde_json::json!({}), None).unwrap();
}

/// A Qwen3 LoRA adapter file as `qwen3::lora::save_adapter_with_lineage`
/// writes it: a card naming `base_id` and, when given, the base digest
/// it was trained against.
pub fn write_adapter(path: &Path, id: &str, base_id: &str, base_digest: Option<String>, fill: f32) {
    let mut card = ModelCard::new(id, "qwen");
    card.adapter = Some(Adapter { kind: "lora".to_string(), rank: Some(2), base: Some(base_id.to_string()), alpha: Some(4.0), ..Default::default() });
    card.training = Some(TrainingProvenance {
        code_revision: "test".to_string(),
        regime: "sft_lora".to_string(),
        seed: 1,
        hyperparams: serde_json::Value::Null,
        environment: "cpu".to_string(),
        gate: None,
        trained_from: None,
        base_digest,
        cycle: 0,
    });
    let tensors = vec![
        ("blocks.0.attn.wq.lora_a".to_string(), vec![2, 4], vec![fill; 8]),
        ("blocks.0.attn.wq.lora_b".to_string(), vec![4, 2], vec![fill; 8]),
    ];
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({"rank": 2, "alpha": 4.0}), Some(&card)).unwrap();
}

pub fn file_digest(path: &Path) -> String {
    brain_modelstore::fetch::file_digest(path).unwrap()
}
