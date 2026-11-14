// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-running inference and
// training jobs that must stop when their caller goes away. If your team needs
// expertise in cooperative cancellation across model serving stacks then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Every listed action observes cancellation, proved from one place.
//!
//! Two promises over [`catalog::manifests`], the list every serving surface is
//! derived from: every action declares `cancellable` (the manifest bit a caller
//! reads), and every action whose provider can be built without weights
//! refuses a token that fired before the call, with the contract's literal and
//! no progress, before it reads a weight or takes a lock.
//!
//! The providers are built from an assembly whose roles name a path that does
//! not exist, so an action that reached for a weight before its entry check
//! would fail with a load error, not the cancelled literal. The mid-run half
//! of the contract (a stop within one step) needs each model's own loop and
//! lives beside it, under `capability_mock::cancellation::certify`.

use std::path::PathBuf;

use capability::{Assembly, Invocation};
use catalog::{manifests, provider_from_assembly, resolver_spec_for};

/// An assembly for `model` whose every role names a path that does not exist.
fn absent_weights(model: &str) -> Assembly {
    let missing = PathBuf::from("/nonexistent/brain-cancellation-test");
    let roles = resolver_spec_for(model).map(|(_, spec)| spec.roles().iter().chain(spec.optional_roles()).map(|r| (r.to_string(), missing.join(r))).collect()).unwrap_or_default();
    Assembly { id: format!("local/{model}"), arch: resolver_spec_for(model).map(|(arch, _)| arch.to_string()).unwrap_or_default(), variant: None, roles, provenance: Vec::new() }
}

#[test]
fn every_listed_action_declares_that_it_observes_cancellation() {
    let undeclared: Vec<String> = manifests().iter().flat_map(|m| m.actions.iter().filter(|a| !a.cancellable).map(move |a| format!("{}/{}", m.model, a.name))).collect();
    assert!(undeclared.is_empty(), "these listed actions do not declare `cancellable`: {undeclared:?}");
}

#[test]
fn every_action_built_without_weights_refuses_a_token_fired_before_the_call() {
    let (mut proved, mut unbuilt, mut refused) = (0usize, Vec::new(), Vec::new());
    for m in manifests() {
        let provider = match provider_from_assembly(&m.model, &absent_weights(&m.model)) {
            Ok(p) => p,
            Err(e) => {
                unbuilt.push(format!("{}: {e}", m.model));
                continue;
            }
        };
        for spec in &m.actions {
            // An action a serving layer above the catalog attaches (the SDK's
            // decision training, say) is that layer's to prove.
            let Some(action) = provider.action(&spec.name) else {
                unbuilt.push(format!("{}/{}: not on the catalog's provider", m.model, spec.name));
                continue;
            };
            // Every action is checked, so one run names every offender.
            let subject = format!("{}/{}", m.model, spec.name);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| capability_mock::cancellation::admission(&subject, Invocation::new(), &|inv, p| action.run(inv, p)))) {
                Ok(()) => proved += 1,
                Err(e) => refused.push(e.downcast_ref::<String>().cloned().unwrap_or(subject)),
            }
        }
    }
    // Printed so a reader of a passing run sees what it covered; the models
    // listed here are covered by their own crates' tests on a loaded model.
    eprintln!("admission proved for {proved} action(s); providers that need weights to be built: {unbuilt:#?}");
    assert!(refused.is_empty(), "{} action(s) ran work under a token that fired before the call:\n{}", refused.len(), refused.join("\n"));
    assert!(proved > 0, "no provider could be built without weights");
}
