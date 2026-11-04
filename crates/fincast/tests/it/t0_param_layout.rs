// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T0 — the device-free parameter-layout gate.
//!
//! Two levels:
//! - **Always-on**: `param_list()` is internally well-formed (unique keys, sane
//!   shapes, right count) and matches the committed golden header
//!   (`tests/golden/header.json`, dumped from the real `v1.pth`). Runs in CI with
//!   no checkpoint.
//! - **Env-gated live gate**: when `FINCAST_CKPT` points at the converted
//!   `model.safetensors` (see `tools/convert/fincast_convert.py`), assert `param_list()`
//!   matches the checkpoint's tensor header name-for-name and shape-for-shape,
//!   with nothing missing or extra (bar the known non-persistent gate buffers).
//!   This is the mechanical diff that must pass before any kernel is written.

use fincast::config::is_non_persistent;
use fincast::FincastConfig;
use std::collections::{HashMap, HashSet};

#[test]
fn param_list_is_well_formed() {
    let c = FincastConfig::default();
    let pl = c.param_list();
    let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys.len(), pl.len(), "duplicate keys in param_list");
    for (k, s) in &pl {
        assert!(!s.is_empty(), "{k} has empty shape");
        assert!(s.iter().all(|&d| d > 0), "{k} has a zero dim");
    }
}

/// The committed golden header is the reference `v1.pth` layout (name→shape),
/// minus the non-persistent buffers. `param_list()` must reproduce it exactly.
/// This grounds the layout in the real checkpoint even in CI (no download).
#[test]
fn param_list_matches_golden_header() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/header.json");
    let bytes = std::fs::read(path).expect("read golden header.json");
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let golden: HashMap<String, Vec<usize>> = v["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, s)| (k.clone(), s.as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect()))
        .collect();

    let ours: HashMap<String, Vec<usize>> = FincastConfig::default().param_list().into_iter().collect();
    let mut missing: Vec<String> = ours.keys().filter(|k| !golden.contains_key(*k)).cloned().collect();
    let mut extra: Vec<String> = golden.keys().filter(|k| !ours.contains_key(*k)).cloned().collect();
    missing.sort();
    extra.sort();
    assert!(missing.is_empty(), "param_list keys not in golden: {:?}", &missing[..missing.len().min(5)]);
    assert!(extra.is_empty(), "golden keys not in param_list: {:?}", &extra[..extra.len().min(5)]);
    for (k, shape) in &ours {
        assert_eq!(&golden[k], shape, "shape mismatch for `{k}`");
    }
}

#[test]
fn live_layout_matches_checkpoint() {
    let Ok(path) = std::env::var("FINCAST_CKPT") else {
        brain_testutil::skip("FINCAST_CKPT unset");
        return;
    };
    let p = std::path::Path::new(&path);
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(&path)
    }
    .expect("read FinCast safetensors");

    let theirs: HashMap<String, Vec<usize>> = tensors.into_iter().map(|t| (t.name, t.shape)).collect();
    let ours: HashMap<String, Vec<usize>> = FincastConfig::default().param_list().into_iter().collect();

    let mut missing = Vec::new();
    for (k, shape) in &ours {
        match theirs.get(k) {
            None => missing.push(k.clone()),
            Some(their_shape) => assert_eq!(their_shape, shape, "shape mismatch for `{k}`"),
        }
    }
    let mut extra: Vec<String> =
        theirs.keys().filter(|k| !ours.contains_key(*k) && !is_non_persistent(k)).cloned().collect();
    missing.sort();
    extra.sort();
    assert!(missing.is_empty(), "param_list missing {} tensors, e.g. {:?}", missing.len(), &missing[..missing.len().min(5)]);
    assert!(extra.is_empty(), "checkpoint has {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]);
}
