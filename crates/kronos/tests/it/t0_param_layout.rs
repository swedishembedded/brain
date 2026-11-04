// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T0 — the device-free parameter-layout gate for BOTH Kronos nets.
//!
//! Always-on: each `param_list()` is internally well-formed. Env-gated live
//! gates: with `BRAIN_KRONOS_TOKENIZER` / `BRAIN_KRONOS_DECODER` pointing at the
//! real safetensors (single file or HF dir), assert `param_list()` matches the
//! checkpoint's tensor header name-for-name and shape-for-shape (bar the known
//! non-persistent BSQ / RoPE buffers). Each config comes from the
//! checkpoint's own `config.json`, so the gate holds for any release tier and
//! `from_hf`'s parse is part of what it proves. This is the mechanical diff
//! that must pass before any kernel is written.

use kronos::{KronosConfig, KronosTokenizerConfig};
use std::collections::{HashMap, HashSet};

#[test]
fn param_lists_are_well_formed() {
    for (name, pl) in [
        ("tokenizer", KronosTokenizerConfig::default().param_list()),
        ("decoder", KronosConfig::default().param_list()),
    ] {
        let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), pl.len(), "{name}: duplicate keys");
        for (k, s) in &pl {
            assert!(!s.is_empty() && s.iter().all(|&d| d > 0), "{name}/{k}: bad shape {s:?}");
        }
    }
}

/// Buffers legitimately in the checkpoint but recomputed in code (not in
/// `param_list`): BSQ basis/codebook and RoPE inverse frequencies.
fn is_non_persistent(name: &str) -> bool {
    name.contains("inv_freq")
        || name.contains("bsq.basis")
        || name.contains("bsq.group_basis")
        || name.contains("group_codebook")
        || name.contains("rotary")
}

/// The `param_list` for whatever tier `path` actually is, read from the
/// checkpoint's OWN `config.json`.
///
/// This used to be `KronosConfig::default().param_list()` - the Kronos-small
/// tier, hardcoded - so pointing the gate at any other release reported
/// `shape mismatch for transformer.4.self_attn.v_proj.bias: left [832] right
/// [512]`, which reads like a layout defect and is really "this is a different
/// size of model". Deriving the config from the checkpoint makes the gate work
/// on every tier AND widens what it proves: `from_hf`'s parse of the real
/// `config.json` is now part of what has to line up, not just a constant.
fn config_param_list(path: &std::path::Path, label: &str) -> Option<HashMap<String, Vec<usize>>> {
    let cfg_path = if path.is_dir() { path.join("config.json") } else { path.parent()?.join("config.json") };
    let bytes = std::fs::read(&cfg_path).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(match label {
        "tokenizer" => KronosTokenizerConfig::from_hf(&v).ok()?.param_list().into_iter().collect(),
        _ => KronosConfig::from_hf(&v).ok()?.param_list().into_iter().collect(),
    })
}

fn live_gate(env: &str, label: &str) {
    let Ok(path) = std::env::var(env) else {
        return brain_testutil::skip(&format!("{env} unset; no live Kronos {label} layout gate"));
    };
    let p = std::path::Path::new(&path);
    let Some(ours) = config_param_list(p, label) else {
        return brain_testutil::skip(&format!("{env}={path} has no readable config.json; no live Kronos {label} layout gate"));
    };
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(&path)
    }
    .expect("read Kronos safetensors");
    let theirs: HashMap<String, Vec<usize>> =
        tensors.into_iter().map(|t| (t.name, t.shape)).collect();

    let mut missing = Vec::new();
    for (k, shape) in &ours {
        match theirs.get(k) {
            None => missing.push(k.clone()),
            Some(their) => assert_eq!(their, shape, "{label}: shape mismatch for `{k}`"),
        }
    }
    let mut extra: Vec<String> = theirs
        .keys()
        .filter(|k| !ours.contains_key(*k) && !is_non_persistent(k))
        .cloned()
        .collect();
    missing.sort();
    extra.sort();
    assert!(missing.is_empty(), "{label}: param_list missing {} tensors, e.g. {:?}", missing.len(), &missing[..missing.len().min(5)]);
    assert!(extra.is_empty(), "{label}: checkpoint has {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]);
}

#[test]
fn live_tokenizer_layout_matches_checkpoint() {
    live_gate("BRAIN_KRONOS_TOKENIZER", "tokenizer");
}

#[test]
fn live_decoder_layout_matches_checkpoint() {
    live_gate("BRAIN_KRONOS_DECODER", "decoder");
}
