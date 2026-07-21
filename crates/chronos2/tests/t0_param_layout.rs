// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T0 — the device-free parameter-layout gate.
//!
//! Two levels:
//! - **Always-on**: `param_list()` is internally well-formed (unique keys,
//!   sane shapes, the right count). This runs in CI with no checkpoint.
//! - **Env-gated live gate**: when `CHRONOS2_CKPT` points at the real
//!   `amazon/chronos-2` safetensors (single file or an HF dir), assert
//!   `param_list()` matches the checkpoint's tensor header name-for-name and
//!   shape-for-shape, and that nothing is missing or extra (bar the known
//!   non-persistent buffers). This is the mechanical diff that must pass before
//!   any kernel is written — the reason `param_list()` uses the reference's own
//!   key names.

use chronos2::Chronos2Config;
use std::collections::{HashMap, HashSet};

#[test]
fn param_list_is_well_formed() {
    let c = Chronos2Config::default();
    let pl = c.param_list();
    let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys.len(), pl.len(), "duplicate keys in param_list");
    // every declared tensor has a non-empty shape
    for (k, s) in &pl {
        assert!(!s.is_empty(), "{k} has empty shape");
        assert!(s.iter().all(|&d| d > 0), "{k} has a zero dim");
    }
}

/// Non-persistent buffers that legitimately appear in the checkpoint header but
/// not in `param_list()` (recomputed in code): RoPE inverse frequencies and the
/// quantile-levels buffer.
fn is_non_persistent(name: &str) -> bool {
    name.contains("inv_freq") || name.ends_with("quantiles") || name.contains("rope_embed")
}

#[test]
fn live_layout_matches_checkpoint() {
    let Ok(path) = std::env::var("CHRONOS2_CKPT") else {
        eprintln!("CHRONOS2_CKPT unset; skipping the live Chronos-2 layout gate");
        return;
    };
    let p = std::path::Path::new(&path);
    let tensors = if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(&path)
    }
    .expect("read Chronos-2 safetensors");

    let theirs: HashMap<String, Vec<usize>> =
        tensors.into_iter().map(|t| (t.name, t.shape)).collect();
    let ours: HashMap<String, Vec<usize>> =
        Chronos2Config::default().param_list().into_iter().collect();

    // every declared param must be present with the exact shape
    let mut missing = Vec::new();
    for (k, shape) in &ours {
        match theirs.get(k) {
            None => missing.push(k.clone()),
            Some(their_shape) => {
                assert_eq!(their_shape, shape, "shape mismatch for `{k}`");
            }
        }
    }
    // nothing left over except known non-persistent buffers
    let mut extra: Vec<String> =
        theirs.keys().filter(|k| !ours.contains_key(*k) && !is_non_persistent(k)).cloned().collect();
    missing.sort();
    extra.sort();
    assert!(missing.is_empty(), "param_list missing {} tensors, e.g. {:?}", missing.len(), &missing[..missing.len().min(5)]);
    assert!(extra.is_empty(), "checkpoint has {} unmapped tensors, e.g. {:?}", extra.len(), &extra[..extra.len().min(5)]);
}
