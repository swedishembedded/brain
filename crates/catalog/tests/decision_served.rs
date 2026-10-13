// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decision models over the serving stack: store -> catalog -> residency ->
//! HTTP, on tiny random-weight checkpoints of both architectures, so the whole
//! path runs on every machine with no downloaded weights.
//!
//! What the tests prove is the CONTRACT, not the answers: the answers are
//! noise. A model is addressed by its store id; asking whether it is ready
//! loads nothing; the first call loads it; every way a call can fail comes
//! back as a distinct, machine-readable class; and `POST /v1/systemone` is a
//! thin view of the same capability.
//!
//! Swedish Embedded AB implements serving contracts that are proven end to
//! end rather than asserted per layer. If your team needs expertise in
//! verifying model-serving stacks, you can procure our services by sending an
//! email to info@swedishembedded.com.

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use catalog::resident_decision::{serve_all, DecisionServeConfig};
use decisions::fixtures::{write_decide, write_laya, Scratch};
use decisions::manifest::FILE as MANIFEST_FILE;
use residency::budget::Budgets;
use residency::{Device, Executor, Policy};
use serde_json::{json, Value};
use tower::ServiceExt;

struct Rig {
    app: Router,
    exec: Executor,
}

/// The serving stack over `store`, registered the way a serving process
/// registers it: each servable checkpoint as its own model, plus the family
/// model that answers inventory and selection.
fn rig(store: &Scratch, configured: &[&str], default: Option<&str>) -> Rig {
    let config = DecisionServeConfig { models: configured.iter().map(|s| s.to_string()).collect(), default: default.map(str::to_string) };
    let models = serve_all(&config, Some(&**store));
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let app = router(AppState::new(exec.clone(), "k", Provider::OpenAI));
    Rig { app, exec }
}

/// Decision checkpoints currently resident (the stateless family model also
/// shows up as a placement once called, and is not a loaded checkpoint).
fn loaded(r: &Rig) -> usize {
    r.exec.residency().placements.iter().filter(|p| p.key.model.starts_with("acme/")).count()
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let (status, text) = post_raw(app, uri, &body.to_string()).await;
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

/// `post` with the body as the caller wrote it: a `serde_json::Value` would
/// sort its keys, and the point of the wire is that the caller's order survives.
async fn post_raw(app: &Router, uri: &str, body: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::AUTHORIZATION, "Bearer k")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 22).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn run(app: &Router, model: &str, params: Value) -> (StatusCode, Value) {
    post(app, "/v1/run", json!({"model": model, "action": "decide", "params": params})).await
}

fn store_with_both() -> Scratch {
    let s = Scratch::new("served");
    write_laya(&s.join("acme/laya"), 48);
    write_decide(&s.join("acme/triage"), 3);
    s
}

const QUESTIONS: &str = r#"[
  {"id": "team", "type": "choice", "instruction": "who handles this",
   "options": [{"id": "billing", "description": "payments"}, "tech", "sales"]},
  {"id": "anger", "type": "score", "instruction": "how upset", "options": ["calm", "annoyed", "furious"]},
  {"id": "urgent", "type": "noul", "instruction": "is this urgent"}
]"#;

fn params() -> Value {
    json!({"state": "my payment failed again today", "questions": QUESTIONS})
}

fn code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or("")
}

// ------------------------------------------------------------------ discoverability

#[test]
fn brain_caps_lists_the_decision_family_and_its_inventory_runs_with_no_weights() {
    let listed = catalog::manifests().into_iter().find(|m| m.model == "brain/decision").expect("`brain caps` must list the decision family");
    let actions: Vec<&str> = listed.actions.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(actions, ["decide", "models", "select"]);
    // What a remote caller is shown never offers a model or path to choose.
    let served = catalog::serving_manifests().into_iter().find(|m| m.model == "brain/decision").unwrap();
    assert!(served.actions[0].params.iter().all(|p| p.name != "model"));

    let provider = catalog::provider("brain/decision").expect("constructible with no weights");
    let out = provider.action("models").unwrap().run(&capability::Invocation::new(), &mut |_| {}).expect("the inventory loads nothing");
    assert!(out.outputs["models"].is_array());
}

// ------------------------------------------------------------------ status

#[tokio::test]
async fn asking_what_is_ready_loads_nothing() {
    let s = store_with_both();
    let r = rig(&s, &["acme/gone"], None);
    let (status, body) = post(&r.app, "/v1/run", json!({"model": "brain/decision", "action": "models"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let models = body["outputs"]["models"].as_array().unwrap();
    let state_of = |id: &str| models.iter().find(|m| m["id"] == id).map(|m| m["state"].as_str().unwrap().to_string());
    assert_eq!(state_of("acme/laya").as_deref(), Some("ready"));
    assert_eq!(state_of("acme/triage").as_deref(), Some("ready"));
    assert_eq!(state_of("acme/gone").as_deref(), Some("not-installed"), "a named model that is missing is reported, not omitted");
    let laya = models.iter().find(|m| m["id"] == "acme/laya").unwrap();
    assert_eq!(laya["validated"], false, "no recorded validation, never reported as validated");
    assert_eq!(laya["backend"], "laya");
    assert_eq!(loaded(&r), 0, "a status query must not load a model");
}

// ------------------------------------------------------------------ addressing and lazy load

#[tokio::test]
async fn two_checkpoints_are_two_addressable_models_loaded_on_first_use() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    let listed: Vec<String> = r.exec.manifests().iter().map(|m| m.model.clone()).collect();
    for id in ["acme/laya", "acme/triage", "brain/decision"] {
        assert!(listed.iter().any(|m| m == id), "{id} must be registered: {listed:?}");
    }
    assert_eq!(loaded(&r), 0, "registering a model reads no weights");

    let (status, laya) = run(&r.app, "acme/laya", params()).await;
    assert_eq!(status, StatusCode::OK, "{laya}");
    assert_eq!((laya["outputs"]["model"].as_str(), laya["outputs"]["backend"].as_str()), (Some("acme/laya"), Some("laya")));
    assert_eq!(loaded(&r), 1, "the first call loads exactly the model it named");

    let (status, triage) = run(&r.app, "acme/triage", params()).await;
    assert_eq!(status, StatusCode::OK, "{triage}");
    assert_eq!((triage["outputs"]["model"].as_str(), triage["outputs"]["backend"].as_str()), (Some("acme/triage"), Some("decide")));
    assert_eq!(loaded(&r), 2);
}

#[tokio::test]
async fn the_served_family_offers_inventory_and_selection_but_not_a_decide_that_would_bypass_the_scheduler() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    let m = r.exec.manifests().iter().find(|m| m.model == "brain/decision").cloned().unwrap();
    let actions: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(actions, ["models", "select"]);
}

// ------------------------------------------------------------------ failure classes

#[tokio::test]
async fn every_way_a_call_can_fail_is_a_distinct_machine_readable_class() {
    let s = store_with_both();
    // Incompatible: weights present, not a safetensors file.
    write_laya(&s.join("acme/corrupt"), 48);
    std::fs::write(s.join("acme/corrupt/model.safetensors"), b"not weights").unwrap();
    // Unsupported: a model whose manifest says it answers choices only.
    write_laya(&s.join("acme/choices-only"), 48);
    std::fs::write(s.join("acme/choices-only").join(MANIFEST_FILE), r#"{"schema": 1, "question_types": ["choice"]}"#).unwrap();

    let r = rig(&s, &["acme/gone", "acme/corrupt"], None);

    let (status, body) = run(&r.app, "acme/gone", params()).await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_installed"), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("brain pull acme/gone"));

    let (status, body) = run(&r.app, "acme/corrupt", params()).await;
    assert_eq!((status, code(&body)), (StatusCode::INTERNAL_SERVER_ERROR, "failed_to_load"), "{body}");
    assert!(!body["error"]["message"].as_str().unwrap().contains(s.to_str().unwrap()), "no path reaches a remote caller: {body}");

    let only_noul = json!({"state": "s", "questions": r#"[{"id": "u", "type": "noul", "instruction": "x"}]"#});
    let (status, body) = run(&r.app, "acme/choices-only", only_noul).await;
    assert_eq!((status, code(&body)), (StatusCode::UNPROCESSABLE_ENTITY, "unsupported_capability"), "{body}");

    let too_many: Vec<String> = (0..256).map(|i| format!("\"o{i}\"")).collect();
    let q = format!(r#"[{{"id": "q", "type": "choice", "instruction": "x", "options": [{}]}}]"#, too_many.join(","));
    let (status, body) = run(&r.app, "acme/laya", json!({"state": "s", "questions": q})).await;
    assert_eq!((status, code(&body)), (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request"), "{body}");
}

// ------------------------------------------------------------------ selection

fn validate(model_dir: &std::path::Path) {
    let digest = brain_modelstore::fetch::file_digest(&model_dir.join("model.safetensors")).unwrap();
    let manifest = format!(r#"{{"schema": 1, "weights_sha256": "{digest}", "validation": {{"status": "validated", "profile": "routing"}}}}"#);
    std::fs::write(model_dir.join(MANIFEST_FILE), manifest).unwrap();
}

async fn select(app: &Router) -> Value {
    post(app, "/v1/run", json!({"model": "brain/decision", "action": "select", "params": {}})).await.1["outputs"].clone()
}

#[tokio::test]
async fn auto_selects_nothing_when_no_model_was_validated_and_says_why() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    let sel = select(&r.app).await;
    assert_eq!(sel["chosen"], Value::Null);
    assert_eq!(sel["rule"], "none");
    let reason = sel["reason"].as_str().unwrap();
    assert!(reason.contains("acme/laya") && reason.contains("no recorded validation"), "{reason}");
}

#[tokio::test]
async fn auto_prefers_a_validated_laya_over_a_validated_decide_model() {
    let s = store_with_both();
    validate(&s.join("acme/laya"));
    validate(&s.join("acme/triage"));
    let sel = select(&rig(&s, &[], None).app).await;
    assert_eq!((sel["chosen"].as_str(), sel["rule"].as_str()), (Some("acme/laya"), Some("validated-laya")));

    // Remove the Laya validation: the validated decide model is next.
    std::fs::remove_file(s.join("acme/laya").join(MANIFEST_FILE)).unwrap();
    let sel = select(&rig(&s, &[], None).app).await;
    assert_eq!((sel["chosen"].as_str(), sel["rule"].as_str()), (Some("acme/triage"), Some("validated-decide")));
}

#[tokio::test]
async fn an_explicitly_configured_default_beats_validation_and_is_reported() {
    let s = store_with_both();
    validate(&s.join("acme/laya"));
    let sel = select(&rig(&s, &["acme/triage"], Some("acme/triage")).app).await;
    assert_eq!((sel["chosen"].as_str(), sel["rule"].as_str()), (Some("acme/triage"), Some("configured")));
    assert!(sel["reason"].as_str().unwrap().contains("explicitly configured"));
}

#[tokio::test]
async fn a_validation_that_no_longer_matches_the_weights_fails_to_load_rather_than_vouching() {
    let s = store_with_both();
    validate(&s.join("acme/laya"));
    // The weights change after the validation was recorded.
    let weights = s.join("acme/laya/model.safetensors");
    let mut bytes = std::fs::read(&weights).unwrap();
    bytes.push(0);
    std::fs::write(&weights, bytes).unwrap();
    let r = rig(&s, &[], None);
    let (status, body) = run(&r.app, "acme/laya", params()).await;
    assert_eq!((status, code(&body)), (StatusCode::INTERNAL_SERVER_ERROR, "failed_to_load"), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("does not match the manifest"), "{body}");
}
