// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that a fine-tune trains exactly the tensors it says it
// trains - no declared weight silently frozen, no adapter on a tensor that is not there - for its clients. If your
// team needs expertise in trainable-parameter audits for ML training code then you can procure our services by
// sending an email to info@swedishembedded.com.

//! C10 of the conformance suite ([`crate::conformance`]): the tensors a model trains are the tensors it declares.
//!
//! The architecture's own declaration ([`crate::ModelConfig::param_list`]: every tensor, by name and size) is the
//! expectation; the built model ([`Model::param_names`], [`Model::optimized_params`]) is the actual. The audit is
//! generic: it names no architecture, only what the contract says.
//!
//! * the built model holds exactly the declared tensors, at the declared sizes;
//! * a full fine-tune (`optimized_params` is `Some`) trains every declared tensor and nothing else;
//! * an adapter model trains only adapter tensors, each A with its B, at the suite's rank, on a
//!   base tensor the architecture declares, whose shape is the product of the pair's;
//! * no tensor is listed twice.

use std::collections::{BTreeMap, BTreeSet};

use crate::adapter::device::is_adapter_param;
use crate::adapter::kind::{LORA_A, LORA_B};
use crate::conformance::{build, trained_rank, Subject, RANK};
use crate::{Model, ModelConfig};

/// The tensors the architecture declares, by name, with their element counts.
fn declared<M: Model>(cfg: &M::Config) -> Result<BTreeMap<String, usize>, String> {
    let mut by_name = BTreeMap::new();
    for (name, size) in cfg.param_list() {
        if by_name.insert(name.clone(), size).is_some() {
            return Err(format!("the architecture declares {name} twice"));
        }
    }
    Ok(by_name)
}

fn no_duplicates(names: &[String], what: &str) -> Result<BTreeSet<String>, String> {
    let mut seen = BTreeSet::new();
    match names.iter().find(|n| !seen.insert((*n).clone())) {
        Some(twice) => Err(format!("{what} lists {twice} twice")),
        None => Ok(seen),
    }
}

/// C10: expected tensors against actual, for the plain model and for the adapter model.
pub fn trainable_set_matches_declaration<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let name = subject.name;
    let fail = |why: String| format!("{name}: {why}");

    // The plain model holds what the architecture declares and, when it exposes a trainable set, trains all of it.
    let (plain, _) = build(subject, None, 1, seed);
    let expected = declared::<M>(plain.config()).map_err(fail)?;
    let actual = no_duplicates(&plain.param_names(), "param_names").map_err(fail)?;
    let declared_names: BTreeSet<String> = expected.keys().cloned().collect();
    if actual != declared_names {
        let missing: Vec<_> = declared_names.difference(&actual).take(5).collect();
        let extra: Vec<_> = actual.difference(&declared_names).take(5).collect();
        return Err(fail(format!("the built model's tensors differ from the architecture's declaration: declared but absent {missing:?}, present but undeclared {extra:?}")));
    }
    for (tensor, size) in &expected {
        let held = plain.read_weight(tensor).len();
        if held != *size {
            return Err(fail(format!("{tensor} is declared with {size} values and the model holds {held}")));
        }
    }
    if let Some(trained) = plain.optimized_params() {
        let trained = no_duplicates(&trained, "optimized_params").map_err(fail)?;
        let frozen: Vec<_> = declared_names.difference(&trained).take(5).collect();
        let phantom: Vec<_> = trained.difference(&declared_names).take(5).collect();
        if !frozen.is_empty() || !phantom.is_empty() {
            return Err(fail(format!("a full fine-tune must train exactly the declared tensors: declared but not trained {frozen:?}, trained but undeclared {phantom:?}")));
        }
    }

    let Some(rank) = trained_rank(subject) else { return Ok(()) };
    let (adapted, _) = build(subject, Some(rank), 1, seed);
    let trained = adapted.optimized_params().ok_or_else(|| format!("{name}: declares adapters but exposes no trainable set"))?;
    let trained = no_duplicates(&trained, "optimized_params of the adapter model").map_err(fail)?;
    if let Some(base) = trained.iter().find(|t| !is_adapter_param(t)) {
        return Err(fail(format!("the adapter model trains {base}, which is not an adapter tensor")));
    }
    let sites: Vec<&String> = trained.iter().filter(|t| t.ends_with(LORA_A)).collect();
    if sites.is_empty() {
        return Err(fail("the adapter model has no adapter site".to_string()));
    }
    for a in sites {
        let base = a.trim_end_matches(LORA_A);
        let b = format!("{base}{LORA_B}");
        if !trained.contains(&b) {
            return Err(fail(format!("{a} has no {b}")));
        }
        let (a_len, b_len) = (adapted.read_weight(a).len(), adapted.read_weight(&b).len());
        let rank = RANK as usize;
        if a_len % rank != 0 || b_len % rank != 0 {
            return Err(fail(format!("{base}: the pair holds {a_len} and {b_len} values, not a multiple of the rank {rank}")));
        }
        let Some(size) = expected.get(base) else {
            return Err(fail(format!("{base}: the adapter sits on a tensor the architecture does not declare")));
        };
        if (a_len / rank) * (b_len / rank) != *size {
            return Err(fail(format!("{base}: a rank-{rank} pair of {a_len} and {b_len} values does not fit a {size}-value base tensor")));
        }
    }
    if let Some(orphan) = trained.iter().find(|t| t.ends_with(LORA_B) && !trained.contains(&format!("{}{LORA_A}", t.trim_end_matches(LORA_B)))) {
        return Err(fail(format!("{orphan} has no matching {LORA_A}")));
    }
    Ok(())
}
