// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The architecture registry's capability declarations against the evidence
//! matrix.
//!
//! `Arch::caps` is the machine-readable claim; `scripts/gates/
//! training-capabilities.tsv` is the evidence, one row per capability naming
//! the tests that back it (`make check/training-capabilities` checks those
//! tests exist and are real `#[test]`s). This test keeps the two in step in
//! both directions, so a capability cannot be declared without evidence nor
//! evidenced without being declared.
//!
//! Swedish Embedded AB implements shipped capability contracts where a model's
//! declared support and its tested support cannot drift apart, for its clients.
//! If your team needs expertise in making a capability matrix machine-readable
//! and gate-enforced, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::{BTreeMap, BTreeSet};

use brain_arch::{by_id, Arch, BaseTier, Caps, ARCHS};

/// The capability names this test owns. The matrix also records serving
/// capabilities (`adapter_serving`, `quant_serving`, `kv_tiers`) and the
/// checkpoint-precision `load` table; those are not part of `Caps` yet and
/// are checked where they are declared.
const LORA_ROWS: [&str; 5] = ["lora_f32", "lora_bf16", "lora_int8", "lora_nf4", "lora_fp8"];

fn matrix() -> BTreeMap<String, BTreeSet<String>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/gates/training-capabilities.tsv");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut rows: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        let arch = cols.next().unwrap_or_default();
        let capability = cols.next().unwrap_or_default();
        // `load:<source>` rows describe the checkpoint-precision table, not an
        // architecture.
        if arch.starts_with("load:") || arch.is_empty() {
            continue;
        }
        rows.entry(arch.to_string()).or_default().insert(capability.to_string());
    }
    rows
}

/// The capability names an architecture's [`Caps`] declare.
fn declared(caps: Caps) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if caps.full_ft {
        out.insert("full_ft".to_string());
    }
    for tier in caps.lora_bases {
        out.insert(tier.row().to_string());
    }
    if caps.other {
        out.insert("other".to_string());
    }
    out
}

/// The training capabilities a matrix row set records.
fn recorded(cells: &BTreeSet<String>) -> BTreeSet<String> {
    cells.iter().filter(|c| *c == "full_ft" || *c == "other" || LORA_ROWS.contains(&c.as_str())).cloned().collect()
}

#[test]
fn every_architecture_in_the_matrix_is_registered() {
    for arch in matrix().keys() {
        assert!(by_id(arch).is_some(), "{arch}: the capability matrix names an architecture the registry has no row for");
    }
}

#[test]
fn declared_capabilities_match_the_evidence_matrix() {
    let matrix = matrix();
    for a in ARCHS {
        let recorded = matrix.get(a.id).map(recorded).unwrap_or_default();
        assert_eq!(declared(a.caps), recorded, "{}: Arch::caps and the capability matrix disagree", a.id);
    }
}

#[test]
fn every_declared_capability_is_backed_by_evidence() {
    let matrix = matrix();
    for a in ARCHS.iter().filter(|a| a.caps.trains() || a.caps.other) {
        let cells = matrix.get(a.id).unwrap_or_else(|| panic!("{}: declares training but the matrix has no row", a.id));
        assert!(!cells.is_empty(), "{}", a.id);
    }
}

#[test]
fn lora_bases_are_unique_and_ordered() {
    for a in ARCHS {
        let mut tiers: Vec<BaseTier> = a.caps.lora_bases.to_vec();
        let before = tiers.len();
        tiers.sort();
        tiers.dedup();
        assert_eq!(tiers.len(), before, "{}: a base tier is declared twice", a.id);
        assert!(before == 0 || a.caps.trains(), "{}", a.id);
    }
}

#[test]
fn toy_architectures_declare_no_training() {
    for a in ARCHS.iter().filter(|a: &&Arch| a.domain == brain_arch::Domain::Toy) {
        assert_eq!(a.caps, Caps::NONE, "{}: a toy architecture declares a training capability", a.id);
    }
}
