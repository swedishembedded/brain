// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The typed configuration: the released file parses completely, and every
//! way a file can be wrong is refused with the offending field named.

use mirage::config::{CheckpointConfig, ConfigError, MirageConfig};
use serde_json::{json, Value};

use crate::common::{released_or_skip, RELEASED_ARCHITECTURE_JSON};

fn released() -> MirageConfig {
    MirageConfig::from_json(RELEASED_ARCHITECTURE_JSON).expect("released architecture parses")
}

/// The released object with one key replaced (or removed when `value` is null).
fn mutated(key: &str, value: Value) -> String {
    let mut v: Value = serde_json::from_str(RELEASED_ARCHITECTURE_JSON).unwrap();
    let obj = v.as_object_mut().unwrap();
    if value.is_null() {
        obj.remove(key);
    } else {
        obj.insert(key.to_string(), value);
    }
    v.to_string()
}

fn rejected_field(key: &str, value: Value) -> String {
    match MirageConfig::from_json(&mutated(key, value)) {
        Err(ConfigError::Invalid { field, .. }) => field,
        other => panic!("`{key}` mutation should be Invalid, got {other:?}"),
    }
}

#[test]
fn the_released_architecture_parses_with_its_derived_dimensions() {
    let c = released();
    assert_eq!(c.dim, 512);
    assert_eq!(c.heads(), 8, "dim / v_dim");
    assert_eq!(c.grid(), 16, "canvas / 2^stem stages");
    assert_eq!(c.embedding_rows(), 8299, "text + 11 language tags + detection and region reference tokens");
    assert_eq!(c.engram_tables(), 4, "orders x hashes");
    assert_eq!(c.engram_sub_dim(), 128);
    assert_eq!(c.hadamard_width(), 512);
    assert_eq!(c.hadamard_factors(), (16, 32));
    assert_eq!(c.dense_levels(), (1, 2, 3), "stem indices of the edge, fine and mid feature maps");
}

#[test]
fn an_unknown_field_is_refused_by_name() {
    let err = MirageConfig::from_json(&mutated("surprise", json!(1))).unwrap_err();
    assert!(err.to_string().contains("surprise"), "{err}");
}

#[test]
fn a_missing_field_is_refused_by_name() {
    let err = MirageConfig::from_json(&mutated("dec_layers", Value::Null)).unwrap_err();
    assert!(err.to_string().contains("dec_layers"), "{err}");
}

#[test]
fn a_wrongly_typed_field_is_refused_by_name() {
    let err = MirageConfig::from_json(&mutated("dim", json!("512"))).unwrap_err();
    assert!(err.to_string().contains("dim"), "{err}");
}

#[test]
fn every_invariant_violation_names_its_field() {
    let cases: Vec<(&str, Value)> = vec![
        ("dim", json!(500)),               // not a power of two (HConvGLU needs a Walsh width)
        ("v_dim", json!(48)),              // dim not divisible by v_dim
        ("kv_heads", json!(3)),            // heads not divisible by kv_heads
        ("qk_dim", json!(47)),             // odd rotary width
        ("sites", json!([7, 3])),          // not ascending
        ("sites", json!([3, 8])),          // beyond dec_layers
        ("orders", json!([3, 2])),         // not ascending
        ("hashes", json!(3)),              // tables do not divide dim
        ("stem", json!([32, 64, 128, 256, 256])), // last stage must be dim
        ("canvas", json!(500)),            // not divisible by 2^stages
        ("stride", json!(6)),              // not a power of two
        ("buckets", json!([1024, 512, 1536])),
        ("buckets", json!([512, 1024, 2048])),   // exceeds max_tokens
        ("hconv_kernel", json!(64)),       // even band width
        ("mix_kernel", json!(4)),
        ("attn_kv_heads", json!(3)),       // attn_heads not divisible by it
        ("bos", json!(1)),                 // collides with eos
        ("dtype", json!("float64")),
        ("form_share", json!(1.5)),
        ("heat_alpha", json!(-0.1)),
        ("grn", json!(false)),             // layouts without GRN are not implemented
        ("level_memory", json!(false)),
        ("neck", json!(false)),
        ("pooled_slots", json!(false)),
        ("conv_norm", json!(false)),
        ("body_layers", json!(0)),
        ("detections", json!(0)),
    ];
    for (key, value) in cases {
        assert_eq!(rejected_field(key, value.clone()), key, "{key} = {value}");
    }
}

#[test]
fn the_tiny_test_configuration_is_valid_and_small() {
    let c = MirageConfig::tiny();
    c.validate().expect("tiny config is valid");
    assert!(c.dim <= 64 && c.dec_layers <= 2 && c.body_layers <= 2);
}

#[test]
fn a_config_round_trips_through_json() {
    let c = released();
    let again = MirageConfig::from_json(&c.to_json()).unwrap();
    assert_eq!(c, again);
}

#[test]
fn the_checkpoint_config_requires_every_envelope_field() {
    let full = json!({
        "model_type": "mirage",
        "architecture": serde_json::from_str::<Value>(RELEASED_ARCHITECTURE_JSON).unwrap(),
        "training_step": 1100000,
        "source_checkpoint_sha256": "dc8db0c4f1a9f516a9080224514de7b6b85193a8c6648a35b0f0486c5bc25224",
        "cact_bits": 2,
        "cact_bits_map": {"embedding": 4, "mhc": 4, "default": 2}
    });
    let c = CheckpointConfig::from_json(&full.to_string()).unwrap();
    assert_eq!(c.training_step, 1_100_000);
    assert_eq!(c.cact_bits_map.embedding, 4);

    let mut wrong_type = full.clone();
    wrong_type["model_type"] = json!("llama");
    let err = CheckpointConfig::from_json(&wrong_type.to_string()).unwrap_err();
    assert!(err.to_string().contains("model_type"), "{err}");

    let mut extra = full.clone();
    extra["cact_bits_map"]["surprise"] = json!(3);
    assert!(CheckpointConfig::from_json(&extra.to_string()).unwrap_err().to_string().contains("surprise"));

    let mut bad_bits = full;
    bad_bits["cact_bits"] = json!(7);
    let err = CheckpointConfig::from_json(&bad_bits.to_string()).unwrap_err();
    assert!(err.to_string().contains("cact_bits"), "{err}");
}

#[test]
fn the_released_config_json_parses_from_the_model_store() {
    let Some(dir) = released_or_skip("released config.json") else { return };
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let c = CheckpointConfig::from_json(&text).expect("released config.json parses with deny_unknown_fields");
    assert_eq!(c.architecture, released(), "the embedded copy in these tests is the released file");
}
