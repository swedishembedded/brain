// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint import boundaries that refuse a wrong file by naming the tensor, for its
// clients. If your team needs expertise in mapping a published release onto an engine's parameters exactly then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: the typed config reads the published `config.json` into the released geometry, and the name map derived from
//! it consumes the published header exactly once, tensor by tensor (shape and admitted encoding), in both directions.

use checkpoint::packed::PackedReader;
use inkling2::config::{ConfigError, LayerKind, LoopSpec};
use inkling2::{names, Family, Inkling2Config};

const REPO: &str = "thinkingmachines/Inkling-2-Mini";

fn published() -> Option<std::path::PathBuf> {
    let dir = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file());
    if dir.is_none() {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
    }
    dir
}

fn tiny() -> Option<std::path::PathBuf> {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if !dir.join("model.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return None;
    }
    Some(dir)
}

#[test]
fn the_published_release_is_consumed_exactly_once_by_the_name_map() {
    let Some(dir) = published() else { return };
    let cfg = Inkling2Config::from_dir(&dir).unwrap();
    assert_eq!(cfg.family, Family::Looped);
    assert_eq!((cfg.hidden, cfg.n_layers, cfg.vocab, cfg.valid_vocab), (2304, 24, 201_024, 200_058));
    assert_eq!(cfg.looped, Some(LoopSpec { start: 6, end: 18, passes: 3, gate_rank: 64 }));
    assert_eq!(cfg.visits(), 48);
    let full: Vec<u32> = (0..24).filter(|&i| cfg.kind(i) == LayerKind::Full).collect();
    assert_eq!(full, [5, 11, 17, 23]);

    let reader = PackedReader::open(&dir).unwrap();
    names::check(&cfg, &reader).unwrap();
    let m = names::manifest(&cfg);
    let params = |m: &[names::Entry]| m.iter().map(|e| e.shape.iter().product::<u64>()).sum::<u64>();
    assert_eq!(m.len(), 1116, "parameters of the published header (sidecars folded)");
    assert_eq!(params(&m), 26_388_641_280);
    assert_eq!(params(&names::text_manifest(&cfg)), 26_070_385_152, "the decoder's share");
}

#[test]
fn the_tiny_reference_checkpoint_is_consumed_exactly_once_and_a_wrong_config_is_refused_by_name() {
    let Some(dir) = tiny() else { return };
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg = Inkling2Config::from_json(&text).unwrap();
    assert_eq!(cfg.visits(), 10);
    let reader = PackedReader::open(&dir).unwrap();
    names::check(&cfg, &reader).unwrap();

    // The same file under a config without the loop: the message gates and queries are left over, named.
    let mut plain: serde_json::Value = serde_json::from_str(&text).unwrap();
    plain["model_type"] = "inkling2".into();
    plain["architectures"] = serde_json::json!(["Inkling2ForConditionalGeneration"]);
    plain["text_config"]["model_type"] = "inkling2_text".into();
    plain["text_config"].as_object_mut().unwrap().remove("loop_config");
    let plain = Inkling2Config::from_json(&plain.to_string()).unwrap();
    let err = names::check(&plain, &reader).unwrap_err();
    assert!(err.contains("model.llm.layers.2.controllers.attn.down_weight") && err.contains("middle_cycle_cross_pass_queries"), "{err}");

    // A geometry the file does not have: the first tensor that disagrees is named with both shapes.
    let mut wide: serde_json::Value = serde_json::from_str(&text).unwrap();
    wide["text_config"]["intermediate_size"] = 48.into();
    let err = names::check(&Inkling2Config::from_json(&wide.to_string()).unwrap(), &reader).unwrap_err();
    assert!(err.contains("model.llm.layers.0.mlp.experts.w13_weight") && err.contains("[8, 96, 64]"), "{err}");
}

#[test]
fn a_config_the_reference_would_read_differently_is_refused_naming_the_field() {
    let Some(dir) = tiny() else { return };
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let edit = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        f(&mut v);
        Inkling2Config::from_json(&v.to_string()).unwrap_err()
    };
    let invalid = |e: ConfigError| match e {
        ConfigError::Invalid { field, .. } => field,
        other => panic!("expected an invalid field, got {other}"),
    };
    // An unknown key at any level is a parse error naming it.
    let e = edit(&|v| v["text_config"]["loop_config"]["exit_gate"] = true.into());
    assert!(matches!(&e, ConfigError::Parse(m) if m.contains("exit_gate")), "{e}");
    // `layer_types` and `local_layer_ids` disagree.
    assert_eq!(invalid(edit(&|v| v["text_config"]["local_layer_ids"] = serde_json::json!([0, 1, 2, 4]))), "text_config.layer_types");
    // Without `layer_types`, the reference's (i + 1) % 6 rule decides; this list contradicts it.
    let e = edit(&|v| {
        v["text_config"].as_object_mut().unwrap().remove("layer_types");
    });
    assert_eq!(invalid(e), "text_config.local_layer_ids");
    // The core must be whole depth-attention blocks.
    assert_eq!(invalid(edit(&|v| v["text_config"]["loop_config"]["core_start"] = 1.into())), "text_config.loop_config.core_start");
    assert_eq!(invalid(edit(&|v| v["text_config"]["loop_config"]["is_varloop"] = true.into())), "text_config.loop_config.is_varloop");
    assert_eq!(invalid(edit(&|v| v["text_config"]["unpadded_vocab_size"] = 81.into())), "text_config.unpadded_vocab_size");
    assert_eq!(invalid(edit(&|v| v["text_config"]["n_shared_experts"] = 1.into())), "text_config.n_shared_experts");
}
