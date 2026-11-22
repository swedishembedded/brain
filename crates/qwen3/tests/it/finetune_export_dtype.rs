// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `transformers` directory a full fine-tune exports stores its weights at the tier the trained checkpoint
//! holds them in: a model trained with bf16 weights is exported bf16 (what the published Qwen3 checkpoints are),
//! not widened to f32 at twice the size. An fp32-trained model stays fp32.
//!
//! Swedish Embedded AB implements fine-tuning pipelines that export deployable checkpoints for its clients. If
//! your team needs expertise in training and shipping models, you can procure our services by sending an email to
//! info@swedishembedded.com.

use checkpoint::packed::{QuantEncoding, SafetensorsSink, Tile, TensorDescriptor, TensorSink};
use model::finetune::ChatTrainer;
use qwen3::finetune::Qwen3Trainer;
use qwen3::QwenConfig;
use std::path::{Path, PathBuf};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-export-dtype-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A brain checkpoint as a parameter-sharded or resident bf16 run saves it: linears and tables at `linear`, the
/// remaining parameters (norms) f32.
fn write_trained(path: &Path, cfg: &QwenConfig, weights: &std::collections::HashMap<String, Vec<f32>>, linear: QuantEncoding) {
    let descs: Vec<TensorDescriptor> = cfg
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let half = name == "tok.weight" || name == cfg.head_weight() || qwen3::q8::Q8::is_i8_linear(&name);
            TensorDescriptor::new(name, vec![numel as u64], if half { linear.clone() } else { QuantEncoding::f32() })
        })
        .collect();
    let mut sink = SafetensorsSink::create(path, &descs, &cfg.to_json()).unwrap();
    for d in &descs {
        let vals = &weights[&d.name];
        let bytes: Vec<u8> = if d.encoding == QuantEncoding::bf16() {
            vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect()
        } else {
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        sink.begin(d).unwrap();
        sink.write_tile(d, &Tile { elems: 0..vals.len() as u64, payload: &bytes, scales: Vec::new() }).unwrap();
        sink.finish(d).unwrap();
    }
    sink.commit().unwrap();
}

fn header_dtypes(file: &Path) -> std::collections::BTreeMap<String, String> {
    let bytes = std::fs::read(file).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    h.as_object().unwrap().iter().filter(|(k, _)| *k != "__metadata__").map(|(k, v)| (k.clone(), v["dtype"].as_str().unwrap().to_string())).collect()
}

fn export(tag: &str, linear: QuantEncoding) -> (QwenConfig, std::collections::HashMap<String, Vec<f32>>, PathBuf) {
    let cfg = QwenConfig::tiny();
    let weights = qwen3::init_weights(&cfg, 7);
    let dir = scratch(tag);
    let trained = dir.join("trained.safetensors");
    write_trained(&trained, &cfg, &weights, linear);
    let tok = dir.join("tok");
    std::fs::create_dir_all(&tok).unwrap();
    let out = dir.join("hf");
    Qwen3Trainer::export_checkpoint(&trained, &tok, &out, None, &mut || true).unwrap();
    (cfg, weights, out)
}

#[test]
fn a_bf16_trained_model_is_exported_bf16_and_reloads_within_bf16_rounding() {
    let (cfg, weights, out) = export("bf16", QuantEncoding::bf16());
    let dtypes = header_dtypes(&out.join("model.safetensors"));
    assert!(dtypes.values().all(|d| d == "BF16"), "every exported tensor is bf16 like the published checkpoints: {dtypes:?}");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap()["torch_dtype"], "bfloat16");
    let (_, src) = qwen3::open_checkpoint(out.to_str().unwrap()).unwrap();
    for (name, _) in cfg.param_list() {
        let want = &weights[&name];
        assert!(
            src.with_tensor(&name, &mut |got| {
                assert_eq!(got.len(), want.len(), "{name}");
                for (g, w) in got.iter().zip(want) {
                    assert!((g - w).abs() <= w.abs() * 2f32.powi(-8) + 1e-30, "{name}: {g} vs {w}");
                }
            }),
            "{name} present"
        );
    }
}

#[test]
fn an_f32_trained_model_is_exported_f32() {
    let (_, _, out) = export("f32", QuantEncoding::f32());
    let dtypes = header_dtypes(&out.join("model.safetensors"));
    assert!(dtypes.values().all(|d| d == "F32"), "{dtypes:?}");
}
