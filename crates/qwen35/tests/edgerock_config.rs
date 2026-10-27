// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! EdgeRock's HF `config.json` -> `Qwen35Config`: the published 4B shape reads
//! to `Qwen35Config::edgerock_4b()` and its parameter list has the tensor and
//! parameter counts the released checkpoint has; every inconsistent
//! `fixed_memory_*` / `independent_memory` field is refused by name.

use qwen35::config::{LayerType, Qwen35Config};
use serde_json::{json, Value};

/// `rockmatrix-ai/EdgeRock`'s config.json, the fields this importer reads.
fn published() -> Value {
    json!({
        "architectures": ["EdgeRockForCausalLM"],
        "attention_bias": false,
        "fixed_memory_revision": "qwen35-text-original-gdn-swa-independent-gqa-memory-v1",
        "fixed_memory_swa_layers": [3, 7, 11, 15, 19, 23, 27, 31],
        "fixed_memory_swa_window": 1024,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_size": 2560,
        "independent_memory": {"admission_rank": 256, "capacity": 1024, "max_commit_chunk": 256, "writer_intermediate_size": 2560},
        "intermediate_size": 9216,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "model_type": "edgerock",
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "tie_word_embeddings": true,
        "vocab_size": 248320
    })
}

fn read(config: &Value) -> Result<Qwen35Config, String> {
    let dir = std::env::temp_dir().join(format!("edgerock_config_{}_{}", std::process::id(), config.to_string().len()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let cfg = qwen35::fp8_source::hf_config(&dir);
    std::fs::remove_dir_all(&dir).ok();
    cfg
}

#[test]
fn the_published_config_reads_to_the_4b_preset() {
    let cfg = read(&published()).expect("the published config parses");
    assert_eq!(format!("{cfg:?}"), format!("{:?}", Qwen35Config { block_size: 262144, ..Qwen35Config::edgerock_4b() }));
    assert_eq!(cfg.memory_commit_block(), Some(256));
}

#[test]
fn the_4b_parameter_list_has_the_released_tensor_and_parameter_counts() {
    let params = Qwen35Config::edgerock_4b().param_list();
    assert_eq!(params.len(), 546, "the released checkpoint holds 546 tensors");
    let total: u64 = params.iter().map(|(_, n)| *n as u64).sum();
    assert!((4_570_000_000..4_590_000_000).contains(&total), "4.58B parameters, got {total}");
    assert!(!params.iter().any(|(n, _)| n == "lm_head.weight"), "tied head");
    let memory_layers: Vec<usize> = (0..32).filter(|l| Qwen35Config::edgerock_4b().layer_has_memory(*l)).collect();
    assert_eq!(memory_layers, [3, 7, 11, 15, 19, 23, 27, 31]);
    assert!(memory_layers.iter().all(|l| Qwen35Config::edgerock_4b().layer_types()[*l] == LayerType::Full));
}

#[test]
fn a_plain_qwen35_config_names_no_window_or_memory() {
    let mut c = published();
    for k in ["fixed_memory_revision", "fixed_memory_swa_layers", "fixed_memory_swa_window", "independent_memory"] {
        c.as_object_mut().unwrap().remove(k);
    }
    let cfg = read(&c).unwrap();
    assert!(cfg.swa_window.is_none() && cfg.anchor_memory.is_none());
}

#[test]
fn inconsistent_fixed_memory_fields_are_refused_by_name() {
    let refused = |edit: &dyn Fn(&mut Value), needle: &str| {
        let mut c = published();
        edit(&mut c);
        let err = read(&c).expect_err(needle);
        assert!(err.contains(needle), "expected {needle:?} in {err:?}");
    };
    refused(&|c| c["fixed_memory_revision"] = json!("qwen35-something-else"), "fixed_memory_revision");
    refused(&|c| c["fixed_memory_swa_layers"] = json!([3, 7, 11]), "fixed_memory_swa_layers");
    refused(&|c| c["fixed_memory_swa_window"] = json!(0), "fixed_memory_swa_window");
    refused(&|c| c["independent_memory"]["capacity"] = json!(0), "independent_memory.capacity");
    refused(&|c| drop(c["independent_memory"].as_object_mut().unwrap().remove("max_commit_chunk")), "independent_memory.max_commit_chunk");
    refused(&|c| drop(c.as_object_mut().unwrap().remove("independent_memory")), "independent_memory");
    refused(&|c| drop(c.as_object_mut().unwrap().remove("fixed_memory_revision")), "fixed_memory_revision");
}

#[test]
fn the_commit_block_is_capped_by_the_window() {
    let mut cfg = Qwen35Config::edgerock_tiny();
    assert_eq!(cfg.memory_commit_block(), Some(4));
    cfg.swa_window = Some(3);
    assert_eq!(cfg.memory_commit_block(), Some(3), "a window shorter than the chunk shortens the block");
    cfg.swa_window = None;
    assert!(cfg.check_window_and_memory().unwrap_err().contains("anchor_memory requires swa_window"));
}

#[test]
fn json_round_trip_keeps_the_window_and_memory() {
    let cfg = Qwen35Config::edgerock_tiny();
    let back = Qwen35Config::from_json(&cfg.to_json());
    assert_eq!(format!("{back:?}"), format!("{cfg:?}"));
}
