// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T0 - the device-free parameter-layout gate.
//!
//! Two levels:
//! - **Always-on**: `param_list()` is internally well-formed (unique keys, sane
//!   shapes, right count) and matches the committed golden header
//!   (`tests/golden/header.json`, read straight from the real
//!   `google/timesfm-3.0-pytorch` `model.safetensors` header - no download,
//!   no torch, just the safetensors JSON prefix). Runs in CI with no
//!   checkpoint.
//! - **Env-gated live gate**: when `BRAIN_TIMESFM3` points at a fetched
//!   checkpoint (`brain pull google/timesfm-3.0-pytorch`, a directory or the
//!   `model.safetensors` file itself), assert `param_list()` matches the
//!   checkpoint's tensor header name-for-name and shape-for-shape, with
//!   nothing missing or extra. This is the mechanical diff that must pass
//!   before any kernel is written.

use timesfm3::Timesfm3Config;
use std::collections::{HashMap, HashSet};

#[test]
fn param_list_is_well_formed() {
    let c = Timesfm3Config::default();
    let pl = c.param_list();
    let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys.len(), pl.len(), "duplicate keys in param_list");
    for (k, s) in &pl {
        assert!(!s.is_empty(), "{k} has empty shape");
        assert!(s.iter().all(|&d| d > 0), "{k} has a zero dim");
    }
}

/// The committed golden header is the real checkpoint's layout (name→shape).
/// `param_list()` must reproduce it exactly. This grounds the layout in the
/// real checkpoint even in CI (no download).
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
    assert_eq!(golden.len(), v["count"].as_u64().unwrap() as usize);

    let ours: HashMap<String, Vec<usize>> = Timesfm3Config::default().param_list().into_iter().collect();
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
    let Ok(path) = std::env::var("BRAIN_TIMESFM3") else {
        brain_testutil::skip("BRAIN_TIMESFM3 unset");
        return;
    };
    let p = std::path::Path::new(&path);
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(&path)
    }
    .expect("read TimesFM-3 safetensors");

    let theirs: HashMap<String, Vec<usize>> = tensors.into_iter().map(|t| (t.name, t.shape)).collect();
    let ours: HashMap<String, Vec<usize>> = Timesfm3Config::default().param_list().into_iter().collect();

    let mut missing = Vec::new();
    for (k, shape) in &ours {
        match theirs.get(k) {
            None => missing.push(k.clone()),
            Some(their_shape) => assert_eq!(their_shape, shape, "shape mismatch for `{k}`"),
        }
    }
    let mut extra: Vec<String> = theirs.keys().filter(|k| !ours.contains_key(*k)).cloned().collect();
    missing.sort();
    extra.sort();
    assert!(missing.is_empty(), "param_list missing {} tensors, e.g. {:?}", missing.len(), &missing[..missing.len().min(5)]);
    assert!(extra.is_empty(), "checkpoint has {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]);
}
