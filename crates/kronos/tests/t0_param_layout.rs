// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T0 — the device-free parameter-layout gate for BOTH Kronos nets.
//!
//! Always-on: each `param_list()` is internally well-formed. Env-gated live
//! gates: with `KRONOS_TOKENIZER_CKPT` / `KRONOS_DECODER_CKPT` pointing at the
//! real safetensors (single file or HF dir), assert `param_list()` matches the
//! checkpoint's tensor header name-for-name and shape-for-shape (bar the known
//! non-persistent BSQ / RoPE buffers). This is the mechanical diff that must
//! pass before any kernel is written.

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

fn live_gate(env: &str, ours: HashMap<String, Vec<usize>>, label: &str) {
    let Ok(path) = std::env::var(env) else {
        eprintln!("{env} unset; skipping the live Kronos {label} layout gate");
        return;
    };
    let p = std::path::Path::new(&path);
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
    live_gate(
        "KRONOS_TOKENIZER_CKPT",
        KronosTokenizerConfig::default().param_list().into_iter().collect(),
        "tokenizer",
    );
}

#[test]
fn live_decoder_layout_matches_checkpoint() {
    live_gate(
        "KRONOS_DECODER_CKPT",
        KronosConfig::default().param_list().into_iter().collect(),
        "decoder",
    );
}
