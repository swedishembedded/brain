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
use decisions::engine::Backend;
use decisions::fixtures::{write_decide, write_laya, write_laya_head_only, Scratch};
use decisions::manifest::{weights_identity, FILE as MANIFEST_FILE};
use residency::budget::Budgets;
use residency::{Device, Executor, Policy};
use serde_json::{json, Value};
use tower::ServiceExt;

struct Rig {
    app: Router,
    exec: Executor,
}

impl Drop for Rig {
    /// Stop the scheduler and let its resident models release their devices
    /// in an orderly way: a device still alive at process exit can crash the
    /// driver's worker threads and fail a test binary whose tests all passed.
    fn drop(&mut self) {
        self.exec.shutdown();
    }
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

// ------------------------------------------------------------------ /v1/systemone

/// A request written the way a caller would: state keys NOT alphabetical,
/// options NOT alphabetical.
const STATE_TEXT: &str = r#"{"ticket": "payments failing", "days": 3}"#;
const SYSTEMONE: &str = r#"{
  "model": "acme/laya",
  "state": {"ticket": "payments failing", "days": 3},
  "questions": {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"technical": "Outages", "billing": "Payments", "sales": "Pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}"#;

#[tokio::test]
async fn systemone_answers_in_the_shared_wire_shape_in_the_callers_order() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    let (status, text) = post_raw(&r.app, "/v1/systemone", SYSTEMONE).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let doc: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["model"], "acme/laya");
    assert_eq!(doc["usage"], json!({"output_tokens": 0}));
    assert_eq!(doc["brain"]["backend"], "laya");

    let a = &doc["answers"];
    assert_eq!(a["is_urgent"]["type"], "noul");
    assert!(a["is_urgent"].get("confidence").is_none(), "a noul is its own confidence");
    assert_eq!(a["department"]["type"], "choice");
    assert!(a["department"]["confidence"].is_number());
    assert_eq!(a["frustration"]["type"], "score");
    assert_eq!(a["frustration"]["legend"], json!({"0": "Calm", "1": "Frustrated", "2": "Very angry"}));
    assert_eq!(a["frustration"]["probabilities"].as_object().unwrap().len(), 3);

    // Order is part of the wire: questions as asked, options as listed.
    let at = |needle: &str| text.find(needle).unwrap_or_else(|| panic!("{needle} missing from {text}"));
    assert!(at("\"is_urgent\"") < at("\"department\"") && at("\"department\"") < at("\"frustration\""), "{text}");
    assert!(at("\"technical\": 0") < at("\"billing\": 0") && at("\"billing\": 0") < at("\"sales\": 0"), "options keep the caller's order: {text}");
}

#[tokio::test]
async fn systemone_and_the_capability_return_the_same_numbers_for_the_same_request() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    for model in ["acme/laya", "acme/triage"] {
        let body = SYSTEMONE.replace("acme/laya", model);
        let (status, text) = post_raw(&r.app, "/v1/systemone", &body).await;
        assert_eq!(status, StatusCode::OK, "{model}: {text}");
        let one: Value = serde_json::from_str(&text).unwrap();

        let questions = r#"{"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "department": {"type": "choice", "instructions": "Which team?",
                           "criteria": {"technical": "Outages", "billing": "Payments", "sales": "Pricing"}},
            "frustration": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]}}"#;
        let (status, full) = run(&r.app, model, json!({"state": STATE_TEXT, "state_format": "json", "questions": questions})).await;
        assert_eq!(status, StatusCode::OK, "{model}: {full}");
        let answers = full["outputs"]["answers"].as_array().unwrap();

        let dept = answers.iter().find(|a| a["id"] == "department").unwrap();
        assert_eq!(one["answers"]["department"]["choice"], dept["selected"], "{model}");
        for cand in dept["ranked"].as_array().unwrap() {
            let id = cand["id"].as_str().unwrap();
            assert_eq!(one["answers"]["department"]["probabilities"][id], cand["probability"], "{model}: {id}");
        }
        let urgent = answers.iter().find(|a| a["id"] == "is_urgent").unwrap();
        assert_eq!(one["answers"]["is_urgent"]["noul"], urgent["noul"], "{model}");
        let anger = answers.iter().find(|a| a["id"] == "frustration").unwrap();
        assert_eq!(one["answers"]["frustration"]["score"], anger["score"], "{model}");
    }
}

#[tokio::test]
async fn systemone_refuses_what_is_outside_the_published_limits_with_a_422() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    let many = |n: usize| (0..n).map(|i| format!("\"o{i}\": \"d\"")).collect::<Vec<_>>().join(", ");
    let levels = |n: usize| (0..n).map(|i| format!("\"l{i}\"")).collect::<Vec<_>>().join(", ");
    let req = |q: String| format!(r#"{{"model": "acme/laya", "state": "s", "questions": {{"q": {q}}}}}"#);
    let cases = [
        ("256 options", req(format!(r#"{{"type": "choice", "instructions": "x", "criteria": {{{}}}}}"#, many(256)))),
        ("11 levels", req(format!(r#"{{"type": "score", "instructions": "x", "criteria": [{}]}}"#, levels(11)))),
        ("1 level", req(format!(r#"{{"type": "score", "instructions": "x", "criteria": [{}]}}"#, levels(1)))),
        ("unknown type", req(r#"{"type": "vibes", "instructions": "x"}"#.to_string())),
        ("missing instructions", req(r#"{"type": "noul"}"#.to_string())),
    ];
    for (what, body) in cases {
        let (status, text) = post_raw(&r.app, "/v1/systemone", &body).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{what}: {text}");
        let doc: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(code(&doc), "invalid_request", "{what}: {text}");
        assert!(doc["error"]["message"].as_str().unwrap().contains("\"q\""), "{what}: the failing question is named: {text}");
    }
    // The limits are inclusive at the wire (`decisions::wire`'s tests take 255
    // options). How many options a model can HOLD is bounded by its context,
    // so a model that cannot hold them says how many fit.
    let ok = req(format!(r#"{{"type": "score", "instructions": "x", "criteria": [{}]}}"#, levels(10)));
    assert_eq!(post_raw(&r.app, "/v1/systemone", &ok).await.0, StatusCode::OK);
    let too_long = req(format!(r#"{{"type": "choice", "instructions": "x", "criteria": {{{}}}}}"#, many(255)));
    let (status, text) = post_raw(&r.app, "/v1/systemone", &too_long).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{text}");
    assert!(text.contains("token context") && text.contains("of the 255 options"), "the refusal says why: {text}");
}

#[tokio::test]
async fn systemone_distinguishes_a_malformed_body_a_missing_field_and_an_unknown_model() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    assert_eq!(post_raw(&r.app, "/v1/systemone", "not json").await.0, StatusCode::BAD_REQUEST);
    assert_eq!(post_raw(&r.app, "/v1/systemone", r#"{"questions": {}}"#).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(post_raw(&r.app, "/v1/systemone", r#"{"model": "nobody/home", "state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}"#).await.0, StatusCode::NOT_FOUND);
}

// ------------------------------------------------------------------ selection

fn validate(model_dir: &std::path::Path) {
    let digest = brain_modelstore::fetch::file_digest(&model_dir.join("model.safetensors")).unwrap();
    let manifest = format!(r#"{{"schema": 1, "weights_sha256": "{digest}", "validation": {{"status": "validated", "profile": "routing"}}}}"#);
    std::fs::write(model_dir.join(MANIFEST_FILE), manifest).unwrap();
}

const AUTO: &str = r#"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}"#;

#[tokio::test]
async fn auto_refuses_to_answer_from_a_model_nobody_validated_and_says_why() {
    let s = store_with_both();
    let r = rig(&s, &[], None);
    let (status, text) = post_raw(&r.app, "/v1/systemone", AUTO).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{text}");
    let doc: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(code(&doc), "not_installed");
    let msg = doc["error"]["message"].as_str().unwrap();
    assert!(msg.contains("acme/laya") && msg.contains("no recorded validation"), "{msg}");
}

#[tokio::test]
async fn auto_prefers_a_validated_laya_over_a_validated_decide_model() {
    let s = store_with_both();
    validate(&s.join("acme/laya"));
    validate(&s.join("acme/triage"));
    let r = rig(&s, &[], None);
    let (status, text) = post_raw(&r.app, "/v1/systemone", AUTO).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "acme/laya");

    // Remove the Laya validation: the validated decide model is next.
    std::fs::remove_file(s.join("acme/laya").join(MANIFEST_FILE)).unwrap();
    let r = rig(&s, &[], None);
    let (status, text) = post_raw(&r.app, "/v1/systemone", AUTO).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "acme/triage");
}

#[tokio::test]
async fn an_explicitly_configured_default_beats_validation_and_is_reported() {
    let s = store_with_both();
    validate(&s.join("acme/laya"));
    let r = rig(&s, &["acme/triage"], Some("acme/triage"));
    let (status, text) = post_raw(&r.app, "/v1/systemone", AUTO).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "acme/triage");
    let (_, sel) = post(&r.app, "/v1/run", json!({"model": "brain/decision", "action": "select", "params": {}})).await;
    assert_eq!(sel["outputs"]["rule"], "configured");
    assert!(sel["outputs"]["reason"].as_str().unwrap().contains("explicitly configured"));
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

#[tokio::test]
async fn a_trained_head_whose_file_or_base_changed_fails_to_load_rather_than_answering() {
    let s = Scratch::new("served-head-only");
    write_laya(&s.join("convai/laya"), 48);
    write_laya_head_only(&s.join("acme/tuned"), &s.join("convai/laya"), "convai/laya", 9);
    let identity = weights_identity(&s.join("acme/tuned"), Backend::Laya).unwrap();
    std::fs::write(s.join("acme/tuned").join(MANIFEST_FILE), format!(r#"{{"schema": 1, "weights_sha256": "{identity}"}}"#)).unwrap();

    let r = rig(&s, &["acme/tuned"], None);
    let (status, body) = run(&r.app, "acme/tuned", params()).await;
    assert_eq!(status, StatusCode::OK, "a head over an installed base serves like any checkpoint: {body}");
    drop(r);

    // The base is a different model than the head was trained on.
    let weights = s.join("convai/laya/model.safetensors");
    let original = std::fs::read(&weights).unwrap();
    let mut changed = original.clone();
    *changed.last_mut().unwrap() ^= 1;
    std::fs::write(&weights, changed).unwrap();
    let r = rig(&s, &["acme/tuned"], None);
    let (status, body) = run(&r.app, "acme/tuned", params()).await;
    assert_eq!((status, code(&body)), (StatusCode::INTERNAL_SERVER_ERROR, "failed_to_load"), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("not the one this head was trained on"), "{body}");
    drop(r);

    // The head itself was edited after the manifest recorded its identity.
    std::fs::write(&weights, original).unwrap();
    let head = s.join("acme/tuned/head.safetensors");
    let mut bytes = std::fs::read(&head).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&head, bytes).unwrap();
    let r = rig(&s, &["acme/tuned"], None);
    let (status, body) = run(&r.app, "acme/tuned", params()).await;
    assert_eq!((status, code(&body)), (StatusCode::INTERNAL_SERVER_ERROR, "failed_to_load"), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("does not match the manifest"), "{body}");
}
