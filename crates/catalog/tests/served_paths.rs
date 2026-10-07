// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements secure model-serving APIs for its clients.
// If your team needs expertise in exposing machine-learning models to
// untrusted callers without exposing the machine then you can procure our
// services by sending an email to info@swedishembedded.com.

//! No served param names a file on the serving machine.
//!
//! A remote caller shares no filesystem with the host that runs an action, so
//! a path it supplies is at best unanswerable and at worst a probe: a dataset
//! folder read from anywhere, an adapter loaded from anywhere, a training
//! output written anywhere. Such a param is the host's to answer, declared
//! `.host_env(..)` or `.host_resolved()`, and `for_serving` drops it from
//! every off-machine surface.
//!
//! Both checks here are driven by the catalog's own manifests, never by a
//! list of known offenders, so a model added tomorrow with a plain path param
//! fails them without anyone having to remember it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use capability::{Invocation, Manifest, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

/// Whether `p` asks its caller for a location on some filesystem, read from
/// the declaration itself: a path-shaped name, or help text that says the
/// value is a path, file, folder or directory. Help text is what a human
/// reads to know what to send, so it is also what reveals that the answer is
/// a path. A param that is not a path but reads like one should be reworded;
/// that is cheaper than a remote caller reading the host's disk.
fn names_a_filesystem_path(p: &ParamSpec) -> bool {
    if p.ty != ParamType::Str {
        return false;
    }
    let name = p.name.to_ascii_lowercase();
    let path_name = ["weights", "tokenizer", "checkpoint", "ckpt", "path", "dir", "file", "folder"].contains(&name.as_str())
        || ["_path", "_dir", "_file", "_folder"].iter().any(|s| name.ends_with(s));
    let help = p.help.to_ascii_lowercase();
    let path_word = help
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .any(|w| matches!(w, "path" | "paths" | "folder" | "folders" | "directory" | "dir" | "file" | "files" | "filename" | "server-side"));
    let path_suffix = [".safetensors", ".gguf", ".pth", ".wav", ".png", ".jpg", ".jpeg"].iter().any(|s| help.contains(s));
    path_name || path_word || path_suffix
}

/// Every action of the served catalog, with every param a remote caller can
/// set. None may name a path.
#[test]
fn no_served_param_names_a_filesystem_path() {
    let mut offenders = Vec::new();
    for m in catalog::serving_manifests() {
        for a in &m.actions {
            for p in a.params.iter().filter(|p| names_a_filesystem_path(p)) {
                offenders.push(format!("{}:{}:{} ({})", m.model, a.name, p.name, p.help));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these params ask a remote caller for a path on the serving machine. Declare each `.host_env(\"BRAIN_...\")` \
         or `.host_resolved()` in its model's caps.rs (and take remote content as a blob input instead), or reword a \
         help text that only reads like a path:\n{}",
        offenders.join("\n")
    );
}

/// The predicate above must see the paths the catalog really declares, or
/// the test above passes by being blind.
#[test]
fn the_path_predicate_recognises_every_declared_host_path() {
    let host_declared: Vec<String> = catalog::manifests()
        .iter()
        .flat_map(|m| m.actions.iter().flat_map(move |a| a.params.iter().map(move |p| (m.model.clone(), a.name.clone(), p))))
        .filter(|(_, _, p)| p.host_env.is_some() || p.host_resolved)
        .filter(|(_, _, p)| !names_a_filesystem_path(p))
        .map(|(model, action, p)| format!("{model}:{action}:{} ({})", p.name, p.help))
        .collect();
    assert!(host_declared.is_empty(), "host-resolved params the path predicate does not recognise:\n{}", host_declared.join("\n"));
}

/// A catalog model's real manifest behind a stand-in instance that counts the
/// invocations that reach it.
struct Listed(Manifest, Arc<AtomicUsize>);
struct ListedInst(Arc<AtomicUsize>);

impl ResidentModel for Listed {
    fn manifest(&self) -> Manifest {
        self.0.clone()
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.0.model, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(ListedInst(Arc::clone(&self.1))))
    }
}

impl Instance for ListedInst {
    fn run(&mut self, _action: &str, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err("a stand-in never runs".into())
    }
}

/// `/v1/run` refuses every path-shaped param of every catalog action, named
/// by the caller, before anything reaches a model - including the ones the
/// served manifest no longer lists, which is the case that matters: a client
/// that ignores discovery and sends the param anyway.
#[tokio::test]
async fn v1_run_refuses_every_path_param_a_caller_names() {
    let reached = Arc::new(AtomicUsize::new(0));
    let full = catalog::manifests();
    let models: Vec<Arc<dyn ResidentModel>> = full.iter().map(|m| Arc::new(Listed(m.clone(), Arc::clone(&reached))) as Arc<dyn ResidentModel>).collect();
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let app = apiserve::router(apiserve::AppState::new(Executor::start(models, budgets, Policy::default()), "k", apiserve::Provider::OpenAI));

    let mut tried = 0;
    let mut accepted = Vec::new();
    for m in &full {
        for a in &m.actions {
            for p in a.params.iter().filter(|p| names_a_filesystem_path(p)) {
                tried += 1;
                let body = json!({"model": m.model, "action": a.name, "params": {p.name.as_str(): "/etc/passwd"}});
                let req = Request::builder()
                    .method(Method::POST)
                    .uri("/v1/run")
                    .header(header::AUTHORIZATION, "Bearer k")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap();
                let response = app.clone().oneshot(req).await.unwrap();
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
                let text = serde_json::from_slice::<Value>(&bytes).map(|v| v.to_string()).unwrap_or_default();
                if status != StatusCode::BAD_REQUEST || !text.contains(&format!("unknown param '{}'", p.name)) {
                    accepted.push(format!("{}:{}:{} -> {status} {text}", m.model, a.name, p.name));
                }
            }
        }
    }
    assert!(tried > 10, "the catalog declares only {tried} path params; the predicate went blind");
    assert!(accepted.is_empty(), "/v1/run did not refuse these path params as unknown:\n{}", accepted.join("\n"));
    assert_eq!(reached.load(Ordering::SeqCst), 0, "a refused call must never reach a model");
}
