// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The durable gate for the fully-qualified naming invariant (AGENTS.md,
//! `docs/models/naming.md`): every model `brain caps` advertises must be a
//! valid `ModelRef` under a reserved vendor, and none of them may collide
//! with a legacy short name in `modelref::alias`'s deprecation table -- a
//! `capability::Manifest.model` is always canonical, never a legacy alias.
//! A test, not a grep, so a future model that regresses this fails CI
//! directly rather than waiting to be noticed by inspection.

use std::process::Command;

use brain_modelref::{alias, ModelRef};

fn bin() -> String {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("brain");
    p.to_string_lossy().into_owned()
}

/// `brain caps --json` lists every statically-known model with no weights
/// loaded and no env configuration required -- safe to run anywhere, and the
/// same command a user runs to discover what brain can serve.
fn static_manifests() -> Vec<serde_json::Value> {
    let out = Command::new(bin()).args(["caps", "--json"]).output().expect("run brain caps --json");
    assert!(out.status.success(), "brain caps --json failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("brain caps --json produced invalid JSON");
    v.as_array().expect("brain caps --json: expected a top-level array").clone()
}

#[test]
fn every_static_model_id_is_a_valid_ref_under_a_reserved_vendor() {
    let manifests = static_manifests();
    assert!(!manifests.is_empty(), "brain caps --json returned no models");
    for m in &manifests {
        let id = m.get("model").and_then(|v| v.as_str()).unwrap_or_else(|| panic!("manifest missing a \"model\" field: {m}"));
        let r = ModelRef::parse(id).unwrap_or_else(|e| panic!("{id:?} is not a valid ModelRef: {e}"));
        assert!(r.is_reserved(), "{id:?} parses but its vendor {:?} is not reserved (brain/local/test)", r.vendor());
    }
}

#[test]
fn no_static_model_id_collides_with_a_legacy_alias_name() {
    // A Manifest.model is always canonical -- if a legacy short name equals a
    // real static id, the alias table and the catalog would disagree about
    // which one "mock" (say) actually means.
    let manifests = static_manifests();
    let legacy: Vec<&str> = alias::legacy_names().collect();
    for m in &manifests {
        let id = m.get("model").and_then(|v| v.as_str()).unwrap();
        assert!(!legacy.contains(&id), "{id:?} is both a static catalog id and a legacy alias name -- ambiguous");
    }
}

#[test]
fn every_legacy_alias_resolves_to_a_reserved_canonical_ref() {
    // The other direction: every row in the deprecation table must itself
    // point at something the grammar accepts under a reserved vendor (this
    // duplicates modelref::alias's own test, but as a CLI-level gate it
    // also catches a future row added there without running that crate's
    // tests).
    for name in alias::legacy_names() {
        let canon = alias::canonical(name).unwrap_or_else(|| panic!("{name:?} is not in the alias table"));
        let r = ModelRef::parse(canon).unwrap_or_else(|e| panic!("{canon:?}: {e}"));
        assert!(r.is_reserved(), "{canon:?} should be under a reserved vendor");
    }
}
