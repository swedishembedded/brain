// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for model servers that never report ready for what they cannot serve for
// its clients. If your team needs expertise in inference serving and memory planning then you can procure our services
// by sending an email to info@swedishembedded.com.

//! Ready means servable: before `brain serve` binds a surface, every served model is sized as its configuration makes
//! it (its batch, KV and adapter bank, through the model's own estimate) and checked against the devices this
//! process may use. Model-agnostic: the decision reads only the residency plan ([`residency::RunPlan::never_fits`]).
//!
//! * A model whose every action's default instance could never fit, even with nothing resident, is unservable. It is
//!   named with its size and withdrawn, so it is neither listed nor accepted (a request for it is "no model", never a
//!   `500` after ready).
//! * When an unservable instance carries more than one served model (adapters riding on their base: they share its
//!   instance), the adapter configuration itself is unservable: the server refuses to start, exit 1, naming each and
//!   its size, rather than serve the adapters' callers nothing.
//! * When nothing servable is left, the server refuses to start too.

use std::collections::{BTreeMap, BTreeSet};

/// One served model's action, sized at startup.
#[derive(Clone, Debug)]
pub struct SizedAction {
    pub model: String,
    /// The instance the action runs on: an adapter names its base's.
    pub instance: String,
    /// Bytes the instance needs on the device class it asks for (the largest of its device, host and NPU bytes).
    pub required: u64,
    pub never_fits: bool,
    /// The planner's own wording, for a person.
    pub refusal: Option<String>,
}

/// What the startup check decided.
#[derive(Debug, PartialEq, Eq)]
pub struct Decision {
    /// Models to withdraw before serving, each with the line that says why.
    pub withdrawn: Vec<(String, String)>,
    /// `Some(reason)`: the server must not start.
    pub refuse: Option<String>,
}

/// The decision for `sized` (every action of every served model). Pure: [`check`] gathers the plans and acts on it.
pub fn decide(sized: &[SizedAction]) -> Decision {
    let mut by_model: BTreeMap<&str, Vec<&SizedAction>> = BTreeMap::new();
    for action in sized {
        by_model.entry(action.model.as_str()).or_default().push(action);
    }
    let unservable: BTreeMap<&str, &SizedAction> = by_model
        .iter()
        .filter(|(_, actions)| actions.iter().all(|a| a.never_fits))
        .map(|(model, actions)| (*model, *actions.iter().max_by_key(|a| a.required).expect("a model has an action")))
        .collect();
    let withdrawn: Vec<(String, String)> = unservable
        .iter()
        .map(|(model, a)| {
            let why = a.refusal.as_deref().unwrap_or("no device this process may use could ever hold it");
            (model.to_string(), format!("{model} (instance {}) needs {} MiB: {why}", a.instance, a.required >> 20))
        })
        .collect();

    // Served models per instance: more than one on an unservable instance is a base with the adapters riding on it.
    let mut riders: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for action in sized {
        riders.entry(action.instance.as_str()).or_default().insert(action.model.as_str());
    }
    let shared: Vec<String> = unservable
        .values()
        .map(|a| a.instance.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|instance| riders.get(instance).is_some_and(|models| models.len() > 1))
        .map(|instance| format!("{instance} ({} models: {})", riders[instance].len(), riders[instance].iter().copied().collect::<Vec<_>>().join(", ")))
        .collect();
    let refuse = if !shared.is_empty() {
        Some(format!(
            "the instance(s) {} carry the served adapters and could never fit any device this process may use; the adapter bank (--adapter-slots, --adapter-rank-cap) and --max-batch are part of that size: lower them or serve fewer adapters per server",
            shared.join("; ")
        ))
    } else if !by_model.is_empty() && unservable.len() == by_model.len() {
        Some("no served model could ever fit any device this process may use".to_string())
    } else {
        None
    };
    Decision { withdrawn, refuse }
}

/// Size every model `executor` serves, report each unservable one, and withdraw it; exit 1 when [`decide`] refuses
/// the start. Runs before any surface binds, so a refused start never touches the ready file.
pub fn check(executor: &residency::Executor) {
    let mut sized = Vec::new();
    for manifest in executor.manifests().iter() {
        for action in &manifest.actions {
            let Ok(plan) = executor.plan(&manifest.model, &action.name, capability::Invocation::default()) else { continue };
            let required = plan.required.vram.max(plan.required.ram).max(plan.required.npu);
            sized.push(SizedAction {
                model: manifest.model.clone(),
                instance: format!("{}[{}]", plan.instance_model, plan.instance_key),
                required,
                never_fits: plan.never_fits,
                refusal: plan.refusal,
            });
        }
    }
    let decision = decide(&sized);
    for (_, line) in &decision.withdrawn {
        eprintln!("brain serve: UNSERVABLE {line}");
    }
    if let Some(reason) = decision.refuse {
        eprintln!("brain serve: refusing to start: {reason}");
        std::process::exit(1);
    }
    for (model, _) in &decision.withdrawn {
        executor.unregister(model);
        eprintln!("brain serve: {model} is not served (withdrawn before ready)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn action(model: &str, instance: &str, mib: u64, never_fits: bool) -> SizedAction {
        SizedAction { model: model.into(), instance: instance.into(), required: mib * MIB, never_fits, refusal: None }
    }

    #[test]
    fn a_configuration_whose_adapters_ride_on_an_instance_that_never_fits_refuses_to_start_naming_it() {
        let d = decide(&[
            action("base", "base[default]", 97_164, true),
            action("base:adapter-a", "base[default]", 97_164, true),
            action("other", "other[default]", 10, false),
        ]);
        let reason = d.refuse.expect("refused");
        assert!(reason.contains("base[default]") && reason.contains("base:adapter-a"), "{reason}");
        assert!(d.withdrawn.iter().any(|(m, line)| m == "base" && line.contains("97164 MiB")), "{:?}", d.withdrawn);
    }

    #[test]
    fn a_lone_model_that_never_fits_is_withdrawn_and_the_others_are_served() {
        let d = decide(&[action("huge", "huge[default]", 500_000, true), action("small", "small[default]", 10, false)]);
        assert_eq!(d.refuse, None);
        assert_eq!(d.withdrawn.iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(), ["huge"]);
    }

    #[test]
    fn a_model_with_one_action_that_fits_stays_served() {
        let d = decide(&[action("m", "m[long]", 500_000, true), action("m", "m[default]", 10, false)]);
        assert_eq!(d, Decision { withdrawn: vec![], refuse: None });
    }

    #[test]
    fn nothing_servable_left_refuses_to_start() {
        assert!(decide(&[action("huge", "huge[default]", 500_000, true)]).refuse.is_some());
    }
}
