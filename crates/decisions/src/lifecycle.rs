// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision-model lifecycle as served actions: `train` a checkpoint from
//! outcome-labelled examples, `eval` it on a held-out split, `promote` it
//! under a policy, `rollback` a promotion, and `inspect` a dataset.
//!
//! This module is the actions' declaration, the one command line
//! `brain decide <action>` is built from. The work (it needs the training
//! pipeline of the SDK, which sits above this crate) is done by the provider
//! the host attaches with [`crate::provider::DecisionProvider::with_lifecycle`]
//! (`brain::decision_run::lifecycle_provider`).
//!
//! **Host only.** Every input is a dataset, policy or checkpoint on the
//! machine that runs the action and `train` writes a checkpoint there, so all
//! of them are `.host_resolved()` and a serving process does not register
//! them ([`crate::provider::served_manifest`]): a remote caller shares no
//! filesystem with the host. `train` is marked `.training()`, so a surface
//! that did serve it would keep it behind `brain serve --training`.
//!
//! Swedish Embedded AB implements the tooling that turns recorded outcomes
//! into governed decision models, from the first training run to a promotion
//! gate. If your team needs expertise in decision-model lifecycle, you can
//! procure our services by sending an email to info@swedishembedded.com.

use capability::{ActionSpec, ParamSpec, ParamType};
use serde_json::json;

/// Action names of the lifecycle, in the order `--help` lists them.
pub const ACTIONS: [&str; 5] = ["train", "eval", "promote", "rollback", "inspect"];

/// Optimizer updates a run takes when the caller names none.
pub const DEFAULT_STEPS: i64 = 600;
/// Examples per optimizer update when the caller names none.
pub const DEFAULT_BATCH: i64 = 16;
/// Most examples one optimizer update may accumulate.
const MAX_BATCH: f64 = 4_096.0;
/// A seed is any `u64`, including the large ones a drawn seed echoes back.
const MAX_SEED: f64 = u64::MAX as f64;
/// The values the `activations` param accepts; the SDK's
/// `model::saved::SavedPolicy::ACTIVATIONS_CHOICES` is checked against it.
pub const ACTIVATIONS: [&str; 2] = ["keep", "recompute"];

fn host_path(name: &str, help: &str) -> ParamSpec {
    ParamSpec::new(name, ParamType::Str, help).host_resolved()
}

fn data() -> ParamSpec {
    host_path("data", "the JSONL file of outcome-labelled decision examples").required()
}

pub fn train_spec() -> ActionSpec {
    ActionSpec::new(
        "train",
        "Train a decision head on the `train` split of a JSONL file of outcome-labelled choices and write a checkpoint directory. The checkpoint's manifest records the base, data digest, split counts, hyperparameters, seed and brain commit, and says `unvalidated`: only `promote` can change that. A Laya base always trains its head with the trunk frozen and is referenced by id and digest, not copied.",
    )
    .training()
    .streaming()
    .cancellable()
    .param(data())
    .param(host_path("base", "the base model: a model-store id, or a checkpoint directory").required())
    .param(host_path("out", "the checkpoint directory to create; it must not exist yet").required())
    .param(ParamSpec::new("finetune_encoder", ParamType::Bool, "train the encoder as well as the head (refused for a Laya base)").default(json!(false)))
    .param(ParamSpec::new("steps", ParamType::Int, "optimizer updates").default(json!(DEFAULT_STEPS)).min(1.0).max(capability::bounds::TRAIN_STEPS))
    .param(ParamSpec::new("batch", ParamType::Int, "examples per optimizer update").default(json!(DEFAULT_BATCH)).min(1.0).max(MAX_BATCH))
    .param(ParamSpec::new("seed", ParamType::Int, "seeds the head's initialization and the batch order (omit for random; the outcome echoes it, and the same seed on the same data and base gives the same weights)").min(0.0).max(MAX_SEED))
    .param(ParamSpec::new(
        "activations",
        ParamType::Enum(ACTIVATIONS.map(String::from).to_vec()),
        "what the backward keeps of the forward: every layer's activations, or recomputed from each layer's input for the same gradients at lower memory (default: the backend's own)",
    ))
}

pub fn eval_spec() -> ActionSpec {
    ActionSpec::new(
        "eval",
        "Measure a checkpoint on one split: accuracy against the majority and chance baselines, NLL, Brier, ECE/AdaECE with reliability bins, coverage versus accuracy, option-order instability under a seeded permutation, cost-weighted regret when `costs` is given, and a per-family breakdown.",
    )
    .cancellable()
    .param(host_path("model", "the checkpoint: a model-store id, or a checkpoint directory").required())
    .param(data())
    .param(ParamSpec::new("split", ParamType::Enum(vec!["validation".into(), "test".into()]), "the held-out split to measure (the train split is not held out)").required())
    .param(host_path("costs", "a JSON file {\"false_exclusion\": X, \"wasted_tokens\": Y} for cost-weighted regret"))
    .param(ParamSpec::new("seed", ParamType::Int, "seeds the option permutation of the order-stability pass (omit for random; the outcome echoes it)").min(0.0).max(MAX_SEED))
}

pub fn promote_spec() -> ActionSpec {
    ActionSpec::new(
        "promote",
        "Re-evaluate a checkpoint on the `test` split under a policy and, only if every gate passes, mark it validated for the policy's profile: the manifest gains the validation and weights digest the `auto` selector requires, the evaluation record is stored beside it, and the checkpoint it supersedes becomes the last-known-good. A failed gate fails the action with how far it was missed and changes nothing. Numbers in the manifest are never trusted.",
    )
    .cancellable()
    .param(host_path("model", "the checkpoint to promote: a model-store id, or a checkpoint directory").required())
    .param(data())
    .param(host_path("policy", "the promotion policy: a JSON file naming the profile and its gates").required())
}

pub fn rollback_spec() -> ActionSpec {
    ActionSpec::new("rollback", "Make the last-known-good checkpoint the validated one again. `model` must be the profile's current checkpoint.")
        .cancellable()
        .param(host_path("model", "the profile's current checkpoint: a model-store id, or a checkpoint directory").required())
}

pub fn inspect_spec() -> ActionSpec {
    ActionSpec::new("inspect", "Validate a JSONL file of decision examples (schema, duplicate ids, leakage across splits) and summarize it: counts, label entropy and the majority-label baseline.")
        .cancellable()
        .param(data())
}

/// Every lifecycle action's declaration, in [`ACTIONS`] order.
pub fn specs() -> Vec<ActionSpec> {
    vec![train_spec(), eval_spec(), promote_spec(), rollback_spec(), inspect_spec()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_are_the_five_verbs_and_only_train_is_a_training_action() {
        let specs = specs();
        assert_eq!(specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ACTIONS);
        let training: Vec<&str> = specs.iter().filter(|s| s.training).map(|s| s.name.as_str()).collect();
        assert_eq!(training, ["train"]);
    }
}
