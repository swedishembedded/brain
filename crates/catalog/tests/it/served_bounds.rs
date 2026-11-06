// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers whose callers
// cannot ask them for unbounded work for its clients. If your team needs
// expertise in hardening model-serving APIs then you can procure our services
// by sending an email to info@swedishembedded.com.

//! Every number a caller may set on a served action has a ceiling, so no
//! request - from a client that is careless or one that is compromised - can
//! ask for a billion frames or a million steps. A param without one is listed
//! in `scripts/gates/unbounded-params.allow`, which only ever shrinks: a new
//! unbounded param fails this test, and so does an entry that is bounded now.

use std::collections::BTreeSet;

const ALLOW: &str = include_str!("../../../../scripts/gates/unbounded-params.allow");

#[test]
fn every_served_number_has_a_ceiling() {
    let mut unbounded = BTreeSet::new();
    // The served models depend on what the model store holds here, so an entry
    // is stale only when its model is served in this run and the param is bounded.
    let mut served = BTreeSet::new();
    for m in catalog::serving_manifests() {
        served.insert(m.model.clone());
        for a in &m.actions {
            for p in &a.params {
                if matches!(p.ty, capability::ParamType::Int | capability::ParamType::Float) && p.max.is_none() {
                    unbounded.insert(format!("{}\t{}\t{}", m.model, a.name, p.name));
                }
            }
        }
    }
    let allowed: BTreeSet<String> = ALLOW.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(str::to_string).collect();
    let new: Vec<&String> = unbounded.difference(&allowed).collect();
    let stale: Vec<&String> = allowed.difference(&unbounded).filter(|entry| entry.split('\t').next().is_some_and(|model| served.contains(model))).collect();
    assert!(new.is_empty(), "served numbers without a max (give them one: ParamSpec::max):\n{}", new.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
    assert!(stale.is_empty(), "bounded now, drop from scripts/gates/unbounded-params.allow:\n{}", stale.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
}
