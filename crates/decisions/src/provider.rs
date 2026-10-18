// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The in-process decision provider - what `brain caps` lists and `brain decide`
//! runs - and the family model that answers "which decision models are here
//! and which would `auto` pick" without loading any of them.
//!
//! `brain/decision` has three actions:
//!
//! * `decide` - the capability itself, with a `model` param: a model id, a
//!   checkpoint directory on this machine, or `auto` (the default), which
//!   applies [`crate::select`]'s rule. The chosen checkpoint loads lazily on
//!   the first call and stays resident (one at a time; another id replaces
//!   it). Concurrent calls to one model serialize on its lock, which is also
//!   the bound: one forward at a time per loaded model.
//! * `models` - every known decision model with its state (`ready`,
//!   `not-installed`, `incompatible`), manifest and validation, read from the
//!   filesystem. Loads nothing.
//! * `select` - the model `auto` (or a named id) resolves to, with the rule
//!   and the reason. Loads nothing.
//!
//! A serving process schedules each checkpoint as its own model under its
//! store id with just the `decide` action
//! (`crates/catalog/src/resident_decision.rs`) and serves this family model
//! with [`served_manifest`], whose actions are the two that load nothing.
//!
//! Swedish Embedded AB implements model inventories and policy-driven
//! selection for its clients. If your team needs a decision model you can ask
//! "what is installed, and what would you use", you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use capability::{tag, Action, ActionResult, ActionSpec, FailureClass, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::{json, Value};

use crate::caps::{self, Session, FAMILY};
use crate::load::{self, pipelines};
use crate::store::{resolve_spec, Catalog};

/// `decide` on the family model: the checkpoint action plus the choice of model.
fn family_decide_spec() -> ActionSpec {
    // Host-resolved: which model answers on THIS machine, including a
    // checkpoint directory, is the local operator's choice. A remote caller
    // names a model by calling that model's own `decide`.
    caps::decide_spec().param(
        ParamSpec::new("model", ParamType::Str, "a decision model id, a checkpoint directory on this machine, or \"auto\" (a configured model, else a validated Laya model, else a validated decide model)")
            .default(json!("auto"))
            .host_resolved(),
    )
}

fn models_spec() -> ActionSpec {
    ActionSpec::new("models", "List the decision models this host knows, each with its state, manifest and validation. Loads nothing.")
}

fn select_spec() -> ActionSpec {
    ActionSpec::new("select", "Say which decision model `auto` (or a named id) resolves to, and why. Loads nothing.")
        .param(ParamSpec::new("model", ParamType::Str, "\"auto\" or a model id").default(json!("auto")))
}

/// The full family manifest, for discovery and the in-process path.
pub fn manifest() -> Manifest {
    Manifest::new(FAMILY, "Decision models: calibrated answers to typed questions over options supplied at request time, plus an inventory of the models that can answer.", vec![family_decide_spec(), models_spec(), select_spec()])
}

/// The family manifest as a serving process registers it: the actions that
/// load nothing. `decide` is served by each checkpoint's own model, which the
/// scheduler places and budgets.
pub fn served_manifest() -> Manifest {
    let mut m = manifest();
    m.actions.retain(|a| a.name != caps::ACTION);
    m
}

/// `{"models": [...]}` - the `models` action's output.
pub fn models_outcome(catalog: &Catalog) -> Outcome {
    Outcome::new().set("models", Value::Array(catalog.records().iter().map(|r| r.to_json()).collect()))
}

/// The `select` action's output.
pub fn select_outcome(catalog: &Catalog, requested: &str) -> Outcome {
    let s = catalog.select(requested);
    Outcome::new().set("chosen", json!(s.chosen)).set("rule", json!(s.rule.slug())).set("reason", json!(s.reason))
}

type Hot = Arc<Mutex<Option<(String, Arc<Mutex<Session>>)>>>;

/// The executable family model.
pub struct DecisionProvider {
    catalog: Catalog,
    hot: Hot,
}

impl DecisionProvider {
    pub fn new(catalog: Catalog) -> DecisionProvider {
        DecisionProvider { catalog, hot: Arc::new(Mutex::new(None)) }
    }

    /// The session for `model`, loading it on first use.
    fn session(&self, model: &str) -> Result<Arc<Mutex<Session>>, String> {
        let (id, dir): (String, PathBuf) = if model.starts_with('/') || model.starts_with('.') {
            resolve_spec(self.catalog.models_dir(), model)
        } else {
            let selection = self.catalog.select(model);
            let id = selection.chosen.ok_or_else(|| tag(FailureClass::NotInstalled, format!("no decision model is available: {}", selection.reason)))?;
            let rec = self.catalog.records().into_iter().find(|r| r.id == id);
            let dir = rec.map(|r| r.dir).ok_or_else(|| tag(FailureClass::NotInstalled, format!("model '{id}' is not installed")))?;
            (id, dir)
        };
        let mut hot = capability::lock_resident(&self.hot);
        if let Some((hot_id, session)) = hot.as_ref() {
            if *hot_id == id {
                return Ok(session.clone());
            }
        }
        *hot = None; // free the old build before loading another
        let loaded = load::load(&id, &dir, |backend| Ok(gpu_core::Gpu::new(pipelines(backend))))?;
        let session = Arc::new(Mutex::new(loaded.session));
        *hot = Some((id, session.clone()));
        Ok(session)
    }
}

impl Provider for DecisionProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let spec = manifest().actions.into_iter().find(|a| a.name == name)?;
        Some(Arc::new(FamilyAction { spec, provider: DecisionProvider { catalog: self.catalog.clone(), hot: self.hot.clone() } }))
    }
}

struct FamilyAction {
    spec: ActionSpec,
    provider: DecisionProvider,
}

impl Action for FamilyAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let model = inv.get_str("model").unwrap_or_else(|| "auto".into());
        match self.spec.name.as_str() {
            "models" => Ok(models_outcome(&self.provider.catalog)),
            "select" => Ok(select_outcome(&self.provider.catalog, &model)),
            "decide" => {
                let session = self.provider.session(&model)?;
                // `lock_resident`: a panic in one request must not brick the model.
                let mut slot = session.lock().map_err(|_| tag(FailureClass::BackendFailure, "the model lock was poisoned by an earlier fault"))?;
                slot.run(inv)
            }
            other => Err(format!("no action '{other}' on model '{FAMILY}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{write_laya, Scratch};
    use capability::classify;

    fn provider(root: &Scratch) -> DecisionProvider {
        DecisionProvider::new(Catalog::new(Some(root.to_path_buf()), vec![], None))
    }

    #[test]
    fn the_family_lists_three_actions_and_serves_the_two_that_load_nothing() {
        let names = |m: &Manifest| m.actions.iter().map(|a| a.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&manifest()), ["decide", "models", "select"]);
        assert_eq!(names(&served_manifest()), ["models", "select"]);
    }

    #[test]
    fn models_and_select_load_nothing_and_work_on_an_empty_host() {
        let s = Scratch::new("provider-empty");
        let p = provider(&s);
        let out = p.action("models").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(out.outputs["models"], json!([]));
        let sel = p.action("select").unwrap().run(&Invocation::new().set("model", json!("auto")), &mut |_| {}).unwrap();
        assert_eq!(sel.outputs["chosen"], Value::Null);
        assert_eq!(sel.outputs["rule"], "none");
        assert!(sel.outputs["reason"].as_str().unwrap().contains("no decision model"));
    }

    #[test]
    fn models_reports_an_installed_unvalidated_model_as_ready_and_not_validated() {
        let s = Scratch::new("provider-models");
        write_laya(&s.join("a/laya"), 48);
        let out = provider(&s).action("models").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let m = &out.outputs["models"][0];
        assert_eq!((m["id"].as_str(), m["state"].as_str(), m["validated"].as_bool()), (Some("a/laya"), Some("ready"), Some(false)));
    }

    #[test]
    fn decide_on_auto_with_nothing_validated_fails_as_not_installed_with_the_reason() {
        let s = Scratch::new("provider-auto");
        write_laya(&s.join("a/laya"), 48);
        let inv = Invocation::new().set("state", json!("s")).set("questions", json!(r#"[{"id":"q","type":"noul","instruction":"x"}]"#)).set("model", json!("auto"));
        let e = provider(&s).action("decide").unwrap().run(&inv, &mut |_| {}).expect_err("must fail");
        let (class, msg) = classify(&e).unwrap();
        assert_eq!(class, FailureClass::NotInstalled);
        assert!(msg.contains("a/laya") && msg.contains("no recorded validation"), "{msg}");
    }
}
