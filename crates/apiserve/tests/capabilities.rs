// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The brain-native routes that reach every model class: the capability
//! catalogue, a synchronous `run`, and background jobs for work too long to hold
//! a request open. Drives the OpenAI surface with `tower::ServiceExt::oneshot`
//! against `apiserve::router(state)` -- no socket, no GPU.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use apiserve::{router, ApiError, AppState, Authenticator, Call, CallResult, Principal, Provider, RequestHooks, Ticket};
use axum::body::Body;
use axum::http::{header, HeaderMap, Method, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

// ------------------------------------------------------------------ fake models

/// `transcribe(lang) <- audio -> text`: a model whose input is a blob.
struct Echo;
struct EchoInst;

fn echo_manifest() -> Manifest {
    Manifest::new(
        "brain-echo",
        "an audio model",
        vec![ActionSpec::new("transcribe", "audio in, text out")
            .param(ParamSpec::new("lang", ParamType::Str, "language").default(json!("en")))
            .input(BlobSpec::new("audio", Media::Audio, "the clip").required())
            .output(BlobSpec::new("text", Media::Text, "the transcript"))],
    )
}

impl ResidentModel for Echo {
    fn manifest(&self) -> Manifest {
        echo_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-echo", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(EchoInst))
    }
}

impl Instance for EchoInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let clip = inv.blobs.get("audio").ok_or("no audio")?;
        let lang = inv.params["lang"].as_str().unwrap_or("?").to_string();
        Ok(Outcome::new()
            .set("bytes", json!(clip.bytes.len()))
            .set("prompt_tokens", json!(clip.bytes.len()))
            .blob("text", Blob::new(Media::Text, format!("{lang}:{}", clip.bytes.len()).into_bytes())))
    }
}

/// `infer(prompt)` whose weights path is the host's fact (`BRAIN_APISERVE_TEST_WEIGHTS`),
/// not a caller's choice. It reports back the path it was run with.
struct Hosted;
struct HostedInst;

const HOSTED_WEIGHTS_VAR: &str = "BRAIN_APISERVE_TEST_WEIGHTS";

fn hosted_manifest() -> Manifest {
    Manifest::new(
        "brain-hosted",
        "a model whose weights live on the host",
        vec![ActionSpec::new("infer", "text in, text out")
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("weights", ParamType::Str, "path to the checkpoint").host_env(HOSTED_WEIGHTS_VAR))
            .param(ParamSpec::new("checkpoint", ParamType::Str, "path to an alternative checkpoint").default(json!("")).host_resolved())
            .output(BlobSpec::new("text", Media::Text, "the answer"))],
    )
}

impl ResidentModel for Hosted {
    fn manifest(&self) -> Manifest {
        hosted_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-hosted", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(HostedInst))
    }
}

impl Instance for HostedInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = inv.params["weights"].as_str().unwrap_or("").to_string();
        Ok(Outcome::new().blob("text", Blob::new(Media::Text, weights.into_bytes())))
    }
}

/// `render(steps) -> video`: slow, reports progress, and honours cancellation.
struct Slow(Arc<AtomicBool>);
struct SlowInst(Arc<AtomicBool>);

fn slow_manifest() -> Manifest {
    Manifest::new(
        "brain-slow",
        "a video model",
        vec![ActionSpec::new("render", "render a clip")
            .streaming()
            .param(ParamSpec::new("steps", ParamType::Int, "steps").default(json!(10)).max(1000.0))
            .output(BlobSpec::new("video", Media::Video, "the clip"))],
    )
}

impl ResidentModel for Slow {
    fn manifest(&self) -> Manifest {
        slow_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-slow", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(SlowInst(Arc::clone(&self.0))))
    }
}

impl Instance for SlowInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let steps = inv.params["steps"].as_u64().unwrap_or(10) as u32;
        for step in 1..=steps {
            if inv.cancel.is_cancelled() {
                self.0.store(true, Ordering::SeqCst);
                return Err("cancelled".into());
            }
            progress(Progress::step(step, steps, "render"));
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(Outcome::new().blob("video", Blob::new(Media::Video, vec![1, 2, 3]).with_meta(json!({"frames": 1}))))
    }
}

/// `train(steps)`: emits a structured frame per step, as a fine-tune reports its loss and statistics, and a
/// plain tick before the first.
struct Trainer(Arc<AtomicUsize>);
struct TrainerInst(Arc<AtomicUsize>);

impl ResidentModel for Trainer {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-trainer",
            "a workload that reports each step",
            vec![ActionSpec::new("train", "train").streaming().param(ParamSpec::new("steps", ParamType::Int, "steps").default(json!(3)).max(100.0)).param(ParamSpec::new("fail", ParamType::Bool, "fail at the end").default(json!(false)))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-trainer", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TrainerInst(Arc::clone(&self.0))))
    }
}

impl Instance for TrainerInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        let steps = inv.params["steps"].as_u64().unwrap_or(3) as u32;
        progress(Progress::step(0, steps, "loading"));
        for step in 1..=steps {
            if inv.cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            progress(Progress::event(step, steps, json!({"step": step, "steps": steps, "loss": 1.0 / step as f64})));
            std::thread::sleep(Duration::from_millis(20));
        }
        if inv.params["fail"].as_bool() == Some(true) {
            return Err("the optimiser diverged".into());
        }
        Ok(Outcome::new().set("steps", json!(steps)))
    }
}

// ------------------------------------------------------------------ the seams

/// A caller named by its key; its scope is its own name.
struct ByName;
struct Named(String);

impl Authenticator for ByName {
    fn authenticate(&self, provider: Provider, headers: &HeaderMap) -> Result<Principal, ApiError> {
        let key = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
        match key {
            Some(name) if !name.is_empty() => Ok(Arc::new(Named(name.to_string()))),
            _ => Err(ApiError::unauthorized(provider, "no key")),
        }
    }
    fn scope(&self, principal: &Principal) -> String {
        principal.downcast_ref::<Named>().map(|n| n.0.clone()).unwrap_or_default()
    }
}

#[derive(Default)]
struct Ledger {
    begun: usize,
    settled: Vec<String>,
}

struct Meter(Arc<Mutex<Ledger>>);
struct MeterTicket(Arc<Mutex<Ledger>>);

impl RequestHooks for Meter {
    fn begin(&self, call: &Call<'_>) -> Result<Box<dyn Ticket>, ApiError> {
        if call.caller.and_then(|p| p.downcast_ref::<Named>()).is_some_and(|n| n.0 == "broke") {
            return Err(ApiError::payment_required(call.provider, "insufficient balance"));
        }
        self.0.lock().unwrap().begun += 1;
        Ok(Box::new(MeterTicket(Arc::clone(&self.0))))
    }
}

impl Ticket for MeterTicket {
    fn finish(self: Box<Self>, result: CallResult<'_>) -> Result<(), ApiError> {
        let what = match result {
            CallResult::Done(outcome) => format!("done:{}", outcome.blobs.keys().cloned().collect::<Vec<_>>().join(",")),
            CallResult::Failed(_) => "failed".to_string(),
            CallResult::Refused => "refused".to_string(),
        };
        self.0.lock().unwrap().settled.push(what);
        Ok(())
    }
}

// ------------------------------------------------------------------ harness

struct Rig {
    app: Router,
    ledger: Arc<Mutex<Ledger>>,
    saw_cancel: Arc<AtomicBool>,
    trainer_runs: Arc<AtomicUsize>,
}

fn rig() -> Rig {
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let trainertrainer_runs = Arc::new(AtomicUsize::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Echo), Arc::new(Slow(Arc::clone(&saw_cancel))), Arc::new(Hosted), Arc::new(Trainer(Arc::clone(&trainertrainer_runs)))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let ledger = Arc::new(Mutex::new(Ledger::default()));
    let state = AppState::new(exec, "unused", Provider::OpenAI).with_authenticator(Arc::new(ByName)).with_hooks(Arc::new(Meter(Arc::clone(&ledger))));
    Rig { app: router(state), ledger, saw_cancel, trainer_runs: trainertrainer_runs }
}

async fn call(app: &Router, method: Method, uri: &str, key: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri).header(header::AUTHORIZATION, format!("Bearer {key}"));
    let body = match body {
        Some(v) => {
            req = req.header(header::CONTENT_TYPE, "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let response = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 22).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Polls a job until it leaves `running`.
async fn settled_job(app: &Router, id: &str, key: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (_, status) = call(app, Method::GET, &format!("/v1/jobs/{id}"), key, None).await;
        if status["status"] != "running" {
            return status;
        }
        assert!(Instant::now() < deadline, "the job never finished: {status}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// One server-sent event.
#[derive(Debug, PartialEq)]
struct Sse {
    id: String,
    event: String,
    data: Value,
}

/// Reads an event stream to its end (the body ends when the job does).
async fn events(app: &Router, uri: &str, key: &str, last_event_id: Option<&str>) -> (StatusCode, Vec<Sse>) {
    let mut req = Request::builder().method(Method::GET).uri(uri).header(header::AUTHORIZATION, format!("Bearer {key}"));
    if let Some(id) = last_event_id {
        req = req.header("last-event-id", id);
    }
    let response = tokio::time::timeout(Duration::from_secs(20), app.clone().oneshot(req.body(Body::empty()).unwrap())).await.expect("the response").unwrap();
    let status = response.status();
    let bytes = tokio::time::timeout(Duration::from_secs(20), axum::body::to_bytes(response.into_body(), 1 << 22)).await.expect("the stream ended").unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let parsed = text
        .split("\n\n")
        .filter(|block| block.lines().any(|l| l.starts_with("event:")))
        .map(|block| {
            let field = |name: &str| block.lines().find_map(|l| l.strip_prefix(name)).map(|v| v.trim().to_string()).unwrap_or_default();
            Sse { id: field("id:"), event: field("event:"), data: serde_json::from_str(&field("data:")).unwrap_or(Value::Null) }
        })
        .collect();
    (status, parsed)
}

const AUDIO: &str = "AAECAwQF"; // base64 of the six bytes 0..=5

fn run_body(model: &str, action: &str, params: Value, blobs: Value) -> Value {
    json!({"model": model, "action": action, "params": params, "blobs": blobs})
}

fn audio_blob() -> Value {
    json!({"audio": {"media": "audio", "data": AUDIO}})
}

// ------------------------------------------------------------------ the specs

#[tokio::test]
async fn the_catalogue_lists_every_action_with_its_contract() {
    let rig = rig();
    let (status, body) = call(&rig.app, Method::GET, "/v1/capabilities", "alice", None).await;
    assert_eq!(status, StatusCode::OK);
    let entries = body["data"].as_array().expect("a list");
    let transcribe = entries.iter().find(|e| e["model"] == "brain-echo" && e["action"] == "transcribe").expect("brain-echo transcribe is listed");
    assert_eq!(transcribe["inputs"][0], json!({"name": "audio", "media": "audio", "required": true}));
    assert_eq!(transcribe["outputs"][0]["name"], "text");
    assert_eq!(transcribe["params"][0]["name"], "lang");
    assert_eq!(transcribe["streaming"], false);
    let render = entries.iter().find(|e| e["action"] == "render").expect("render is listed");
    assert_eq!(render["streaming"], true);
}

/// The devices the executor schedules on are listed with their memory as it accounts it; the host adds what each one
/// is, and nothing it adds is a path or an address. No key is no answer, as for every route.
#[tokio::test]
async fn the_devices_are_listed_with_their_memory_and_what_the_host_says_they_are() {
    struct Host;
    impl apiserve::DeviceInventory for Host {
        fn host(&self) -> serde_json::Map<String, Value> {
            json!({"backend": "cpu"}).as_object().unwrap().clone()
        }
        fn device(&self, _device: Device) -> serde_json::Map<String, Value> {
            json!({"name": "host", "class": "cpu"}).as_object().unwrap().clone()
        }
    }
    let rig = rig();
    let (status, body) = call(&rig.app, Method::GET, "/v1/devices", "alice", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"], json!([{"id": "cpu", "kind": "cpu", "memory": {"total_bytes": 8u64 << 30, "reserved_bytes": 0, "used_bytes": 0, "free_bytes": 8u64 << 30}}]));
    assert_eq!(body["host"], json!({}), "a host that describes nothing adds nothing");

    let described = AppState::new(Executor::start(Vec::new(), { let mut b = Budgets::new(); b.set(Device::Cpu, 1 << 30, 1 << 20); b }, Policy::default()), "key", Provider::OpenAI).with_devices(Some(Arc::new(Host)));
    let app = router(described);
    let (status, body) = call(&app, Method::GET, "/v1/devices", "key", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!((body["host"]["backend"].as_str(), body["data"][0]["name"].as_str(), body["data"][0]["memory"]["free_bytes"].as_u64()), (Some("cpu"), Some("host"), Some((1 << 30) - (1 << 20))));
    let text = body.to_string();
    assert!(!text.contains('/') && !text.to_lowercase().contains("uuid") && !text.contains("pci"), "no path, address or identifier of the machine: {text}");

    let bare = app.clone().oneshot(Request::builder().uri("/v1/devices").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(bare.status(), StatusCode::UNAUTHORIZED, "the route is behind the key like every other");
    let wrong = app.oneshot(Request::builder().uri("/v1/devices").header(header::AUTHORIZATION, "Bearer wrong").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn the_catalogue_does_not_offer_what_only_the_host_may_choose() {
    let rig = rig();
    let (_, body) = call(&rig.app, Method::GET, "/v1/capabilities", "alice", None).await;
    let infer = body["data"].as_array().unwrap().iter().find(|e| e["model"] == "brain-hosted").expect("brain-hosted is listed");
    let names: Vec<&str> = infer["params"].as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["prompt"], "a weights path is the host's, never offered to a caller");
}

#[tokio::test]
async fn a_caller_cannot_name_what_only_the_host_may_choose() {
    let rig = rig();
    for param in ["weights", "checkpoint"] {
        let (status, response) = call(&rig.app, Method::POST, "/v1/run", "alice", Some(run_body("brain-hosted", "infer", json!({"prompt": "hi", param: "/etc/passwd"}), json!({})))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "naming '{param}' must be refused: {response}");
    }
    assert_eq!(rig.ledger.lock().unwrap().begun, 0, "a refused call never reaches the hooks");
}

#[tokio::test]
async fn the_host_still_answers_for_what_the_caller_may_not() {
    std::env::set_var(HOSTED_WEIGHTS_VAR, "/host/own/weights");
    let rig = rig();
    let (status, body) = call(&rig.app, Method::POST, "/v1/run", "alice", Some(run_body("brain-hosted", "infer", json!({"prompt": "hi"}), json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["blobs"]["text"]["data"], "L2hvc3Qvb3duL3dlaWdodHM=", "base64 of the host's own path");
}

#[tokio::test]
async fn run_executes_any_action_with_blobs_in_and_blobs_out() {
    let rig = rig();
    let (status, body) = call(&rig.app, Method::POST, "/v1/run", "alice", Some(run_body("brain-echo", "transcribe", json!({"lang": "sv"}), audio_blob()))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outputs"]["bytes"], 6);
    assert_eq!(body["blobs"]["text"]["media"], "text");
    assert_eq!(body["blobs"]["text"]["data"], "c3Y6Ng==", "base64 of \"sv:6\"");
    assert_eq!(rig.ledger.lock().unwrap().settled, ["done:text"], "a run is metered like any other call");
}

#[tokio::test]
async fn run_refuses_what_the_action_does_not_accept_before_anythingtrainer_runs() {
    let rig = rig();
    let cases = [
        (run_body("brain-echo", "transcribe", json!({"lang": "sv", "nope": 1}), audio_blob()), StatusCode::BAD_REQUEST, "an unknown param"),
        (run_body("brain-echo", "transcribe", json!({}), json!({})), StatusCode::BAD_REQUEST, "a missing required blob"),
        (run_body("brain-echo", "transcribe", json!({}), json!({"audio": {"media": "image", "data": AUDIO}})), StatusCode::BAD_REQUEST, "the wrong media"),
        (run_body("brain-echo", "transcribe", json!({}), json!({"audio": {"media": "audio", "data": "***"}})), StatusCode::BAD_REQUEST, "bad base64"),
        (run_body("brain-echo", "no-such-action", json!({}), json!({})), StatusCode::NOT_FOUND, "an unknown action"),
        (run_body("no-such-model", "transcribe", json!({}), json!({})), StatusCode::NOT_FOUND, "an unknown model"),
    ];
    for (body, expected, why) in cases {
        let (status, response) = call(&rig.app, Method::POST, "/v1/run", "alice", Some(body)).await;
        assert_eq!(status, expected, "{why}: {response}");
    }
    assert_eq!(rig.ledger.lock().unwrap().begun, 0, "nothing invalid reaches the hooks");
}

#[tokio::test]
async fn a_jobtrainer_runs_in_the_background_and_its_result_is_fetched_later() {
    let rig = rig();
    let (status, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-slow", "render", json!({"steps": 5}), json!({})))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{created}");
    let id = created["id"].as_str().expect("an id").to_string();

    let (status, pending) = call(&rig.app, Method::GET, &format!("/v1/jobs/{id}/result"), "alice", None).await;
    if pending["status"] == "running" {
        assert_eq!(status, StatusCode::CONFLICT, "a result is not served before it exists");
    }

    let done = settled_job(&rig.app, &id, "alice").await;
    assert_eq!(done["status"], "succeeded", "{done}");

    let (status, result) = call(&rig.app, Method::GET, &format!("/v1/jobs/{id}/result"), "alice", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["blobs"]["video"]["media"], "video");
    assert_eq!(result["blobs"]["video"]["meta"], json!({"frames": 1}));
    assert_eq!(rig.ledger.lock().unwrap().settled, ["done:video"]);
}

#[tokio::test]
async fn a_running_job_reports_its_progress() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-slow", "render", json!({"steps": 200}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (_, status) = call(&rig.app, Method::GET, &format!("/v1/jobs/{id}"), "alice", None).await;
        if status["progress"]["step"].as_u64().is_some_and(|s| s > 0) {
            assert_eq!(status["status"], "running");
            assert_eq!(status["progress"]["total"], 200);
            break;
        }
        assert!(Instant::now() < deadline, "no progress was ever reported: {status}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let _ = call(&rig.app, Method::DELETE, &format!("/v1/jobs/{id}"), "alice", None).await;
}

#[tokio::test]
async fn a_job_can_be_cancelled_and_the_model_sees_it() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-slow", "render", json!({"steps": 500}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();

    let (status, _) = call(&rig.app, Method::DELETE, &format!("/v1/jobs/{id}"), "alice", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(settled_job(&rig.app, &id, "alice").await["status"], "cancelled");

    let deadline = Instant::now() + Duration::from_secs(10);
    while !rig.saw_cancel.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the model never saw the cancellation");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (status, _) = call(&rig.app, Method::GET, &format!("/v1/jobs/{id}/result"), "alice", None).await;
    assert_ne!(status, StatusCode::OK, "a cancelled job has no result");
    assert_eq!(rig.ledger.lock().unwrap().settled.len(), 1, "settled once, as a call that did not complete");
}

/// Deleting a finished job forgets it and its result at once, so a caller
/// that has read what it needs gets its slot back instead of waiting out the
/// result's time to live.
#[tokio::test]
async fn deleting_a_finished_job_forgets_it() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-slow", "render", json!({"steps": 1}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    assert_eq!(settled_job(&rig.app, &id, "alice").await["status"], "succeeded");
    let (status, body) = call(&rig.app, Method::DELETE, &format!("/v1/jobs/{id}"), "alice", None).await;
    assert_eq!((status, body["status"].as_str()), (StatusCode::OK, Some("deleted")));
    let (status, _) = call(&rig.app, Method::GET, &format!("/v1/jobs/{id}"), "alice", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_job_belongs_to_the_caller_that_made_it() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-slow", "render", json!({"steps": 3}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    let _ = settled_job(&rig.app, &id, "alice").await;

    for (method, suffix) in [(Method::GET, ""), (Method::GET, "/result"), (Method::DELETE, "")] {
        let (status, _) = call(&rig.app, method.clone(), &format!("/v1/jobs/{id}{suffix}"), "mallory", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {suffix}: another caller must not even learn the job exists");
    }
    let (status, _) = call(&rig.app, Method::GET, &format!("/v1/jobs/{id}/result"), "alice", None).await;
    assert_eq!(status, StatusCode::OK, "its owner still can");
}

#[tokio::test]
async fn a_caller_who_cannot_pay_is_refused_when_the_job_is_created_not_after() {
    let rig = rig();
    let (status, body) = call(&rig.app, Method::POST, "/v1/jobs", "broke", Some(run_body("brain-slow", "render", json!({"steps": 3}), json!({})))).await;
    assert_eq!(status, StatusCode::PAYMENT_REQUIRED, "{body}");
    assert_eq!(rig.ledger.lock().unwrap().begun, 0);
}

#[tokio::test]
async fn an_unknown_job_is_not_found() {
    let rig = rig();
    let (status, _) = call(&rig.app, Method::GET, "/v1/jobs/00000000-0000-0000-0000-000000000000", "alice", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = call(&rig.app, Method::GET, "/v1/jobs/not-a-uuid", "alice", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ------------------------------------------------------------------ the event stream

#[tokio::test]
async fn a_jobs_events_replay_every_step_then_end_with_its_status() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-trainer", "train", json!({"steps": 4}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    // Followed while it runs.
    let (status, live) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", None).await;
    assert_eq!(status, StatusCode::OK);
    let frames: Vec<&Sse> = live.iter().filter(|e| e.event == "frame").collect();
    assert_eq!(frames.len(), 4, "{live:?}");
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.data["step"], i as u64 + 1);
        assert_eq!(frame.data["loss"], 1.0 / (i as f64 + 1.0));
    }
    let last = live.last().unwrap();
    assert_eq!((last.event.as_str(), &last.data), ("done", &json!({"status": "succeeded"})));
    let ids: Vec<u64> = live.iter().map(|e| e.id.parse().unwrap()).collect();
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "event ids only grow: {ids:?}");

    // Followed after it finished: the same events, from the log.
    let (_, late) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", None).await;
    assert_eq!(late, live);
}

#[tokio::test]
async fn a_client_that_reconnects_resumes_after_the_last_event_it_got() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-trainer", "train", json!({"steps": 4}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    let (_, all) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", None).await;
    let third = &all[2].id;
    let (_, rest) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", Some(third)).await;
    assert_eq!(rest, all[3..], "exactly the events after the third");
    let (_, none) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", Some(&all.last().unwrap().id)).await;
    assert!(none.is_empty(), "nothing follows the terminal event");
}

#[tokio::test]
async fn a_failed_job_ends_its_stream_with_the_reason_and_a_cancelled_one_says_so() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-trainer", "train", json!({"steps": 2, "fail": true}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    let (_, failed) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", None).await;
    let end = failed.last().unwrap();
    assert_eq!((end.event.as_str(), end.data["status"].as_str()), ("done", Some("failed")), "{failed:?}");
    assert!(end.data["error"]["message"].is_string());

    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-trainer", "train", json!({"steps": 100}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    let follower = {
        let (app, uri) = (rig.app.clone(), format!("/v1/jobs/{id}/events"));
        tokio::spawn(async move { events(&app, &uri, "alice", None).await })
    };
    let (status, _) = call(&rig.app, Method::DELETE, &format!("/v1/jobs/{id}"), "alice", None).await;
    assert_eq!(status, StatusCode::OK);
    let (_, cancelled) = follower.await.unwrap();
    assert_eq!(cancelled.last().unwrap().data, json!({"status": "cancelled"}));
}

#[tokio::test]
async fn a_jobs_events_are_its_owners_alone() {
    let rig = rig();
    let (_, created) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-trainer", "train", json!({"steps": 2}), json!({})))).await;
    let id = created["id"].as_str().unwrap().to_string();
    let (status, body) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "mallory", None).await;
    assert_eq!((status, body.len()), (StatusCode::NOT_FOUND, 0), "another caller must not learn the job exists");
    for uri in ["/v1/jobs/not-a-uuid/events", "/v1/jobs/00000000-0000-0000-0000-000000000000/events"] {
        assert_eq!(events(&rig.app, uri, "alice", None).await.0, StatusCode::NOT_FOUND, "{uri}");
    }
    for bad in ["abc", "-1", "1.5", ""] {
        let (status, _) = events(&rig.app, &format!("/v1/jobs/{id}/events"), "alice", Some(bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "Last-Event-ID {bad:?}");
    }
}

#[tokio::test]
async fn the_event_stream_is_not_reachable_without_a_key() {
    let rig = rig();
    let response = rig.app.clone().oneshot(Request::builder().method(Method::GET).uri("/v1/jobs/00000000-0000-0000-0000-000000000000/events").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn a_client_that_hangs_up_while_its_job_waits_for_a_lane_takes_the_job_with_it() {
    let rig = rig();
    // The lane is held by a long render, so the training job can only wait.
    let (_, held) = call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-slow", "render", json!({"steps": 1000}), json!({})))).await;
    let held = held["id"].as_str().unwrap().to_string();
    let waiting = tokio::time::timeout(Duration::from_millis(400), call(&rig.app, Method::POST, "/v1/jobs", "alice", Some(run_body("brain-trainer", "train", json!({"steps": 3}), json!({})))));
    assert!(waiting.await.is_err(), "the job was admitted while the lane was held: the premise of the test is gone");
    // The client is gone. Free the lane: the abandoned job must not run.
    let _ = call(&rig.app, Method::DELETE, &format!("/v1/jobs/{held}"), "alice", None).await;
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(rig.trainer_runs.load(Ordering::SeqCst), 0, "a job whose client hung up while it was queued ran anyway");
}
