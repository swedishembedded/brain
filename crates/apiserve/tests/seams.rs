// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two seams an embedding application plugs into: who the caller is
//! ([`apiserve::Authenticator`]) and what a call may do and costs
//! ([`apiserve::RequestHooks`]). Drives each dialect with
//! `tower::ServiceExt::oneshot` against `apiserve::router(state)` -- no socket,
//! no GPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use apiserve::{router, ApiError, AppState, Authenticator, Call, CallResult, Principal, Provider, RequestHooks, Ticket};
use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use futures::StreamExt;
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

// ------------------------------------------------------------------ a fake model

/// A streaming chat model that counts how often it actually runs.
struct FakeChat(Arc<AtomicUsize>);
struct FakeChatInst(Arc<AtomicUsize>);

fn chat_manifest() -> Manifest {
    Manifest::new(
        "brain-chat",
        "a chat model",
        vec![ActionSpec::new("generate", "generate text")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .output(BlobSpec::new("text", Media::Text, "generated text"))],
    )
}

impl ResidentModel for FakeChat {
    fn manifest(&self) -> Manifest {
        chat_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-chat", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeChatInst(Arc::clone(&self.0))))
    }
}

impl Instance for FakeChatInst {
    fn run(&mut self, _a: &str, _i: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        for (i, piece) in ["Hello", " world"].iter().enumerate() {
            progress(Progress::token(i as u32, 2, *piece));
        }
        Ok(Outcome::new()
            .set("prompt_tokens", json!(5))
            .set("completion_tokens", json!(2))
            .set("finish_reason", json!("stop"))
            .blob("text", Blob::new(Media::Text, b"Hello world".to_vec())))
    }
}

// ------------------------------------------------------------------ the seams under test

/// Who a key belongs to.
struct Caller(&'static str);

/// Recognises three keys; anything else, including a surface's own static key,
/// is refused.
struct Callers;

impl Authenticator for Callers {
    fn authenticate(&self, provider: Provider, headers: &HeaderMap) -> Result<Principal, ApiError> {
        let presented = match provider {
            Provider::Anthropic => headers.get("x-api-key").and_then(|v| v.to_str().ok()),
            Provider::OpenAI | Provider::OpenRouter => headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")),
        };
        match presented {
            Some("alice-key") => Ok(Arc::new(Caller("alice"))),
            Some("broke-key") => Ok(Arc::new(Caller("broke"))),
            Some("unrecordable-key") => Ok(Arc::new(Caller("unrecordable"))),
            _ => Err(ApiError::unauthorized(provider, "no such key")),
        }
    }
}

/// What a ticket was settled with.
#[derive(Debug, PartialEq, Eq)]
enum Settled {
    Done { prompt: i64, completion: i64 },
    Failed,
    Refused,
}

#[derive(Default)]
struct Log {
    begun: Vec<(String, String, String)>,
    settled: Vec<Settled>,
    /// When set, the hooks do not list any model.
    hidden: bool,
}

/// Records every call it sees; refuses `broke` at the door, and fails to record
/// a settlement for `unrecordable`.
struct Recorder(Arc<Mutex<Log>>);
struct RecorderTicket {
    log: Arc<Mutex<Log>>,
    caller: &'static str,
}

impl RequestHooks for Recorder {
    fn listed(&self, _model: &str) -> bool {
        !self.0.lock().unwrap().hidden
    }

    fn begin(&self, call: &Call<'_>) -> Result<Box<dyn Ticket>, ApiError> {
        let caller = call.caller.and_then(|p| p.downcast_ref::<Caller>()).map_or("anonymous", |c| c.0);
        if caller == "broke" {
            return Err(ApiError::payment_required(call.provider, "insufficient balance"));
        }
        self.0.lock().unwrap().begun.push((caller.to_string(), call.model.to_string(), call.action.to_string()));
        Ok(Box::new(RecorderTicket { log: Arc::clone(&self.0), caller }))
    }
}

impl Ticket for RecorderTicket {
    fn finish(self: Box<Self>, result: CallResult<'_>) -> Result<(), ApiError> {
        let settled = match result {
            CallResult::Done(outcome) => Settled::Done {
                prompt: outcome.outputs.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(-1),
                completion: outcome.outputs.get("completion_tokens").and_then(Value::as_i64).unwrap_or(-1),
            },
            CallResult::Failed(_) => Settled::Failed,
            CallResult::Refused => Settled::Refused,
        };
        self.log.lock().unwrap().settled.push(settled);
        if self.caller == "unrecordable" {
            return Err(ApiError::internal(Provider::OpenAI, "the charge could not be recorded"));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ harness

struct Rig {
    app: Router,
    log: Arc<Mutex<Log>>,
    runs: Arc<AtomicUsize>,
}

fn rig(provider: Provider) -> Rig {
    let runs = Arc::new(AtomicUsize::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeChat(Arc::clone(&runs)))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let log = Arc::new(Mutex::new(Log::default()));
    let state = AppState::new(exec, "static-key", provider)
        .with_authenticator(Arc::new(Callers))
        .with_hooks(Arc::new(Recorder(Arc::clone(&log))));
    Rig { app: router(state), log, runs }
}

fn chat_request(key: &str, stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::AUTHORIZATION, format!("Bearer {key}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({"model": "brain-chat", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}).to_string()))
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Waits until `check` holds, so a settlement that happens on the executor's
/// thread after the response is visible without a fixed sleep.
async fn eventually(what: &str, check: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !check() {
        assert!(Instant::now() < deadline, "never happened: {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// ------------------------------------------------------------------ the specs

#[tokio::test]
async fn a_pluggable_authenticator_replaces_the_static_key() {
    let rig = rig(Provider::OpenAI);
    let get = |key: &str| Request::builder().uri("/models").header(header::AUTHORIZATION, format!("Bearer {key}")).body(Body::empty()).unwrap();

    let (status, _) = json_body(rig.app.clone().oneshot(get("alice-key")).await.unwrap()).await;
    assert_eq!(status, StatusCode::OK, "a key the authenticator knows is let in");

    let (status, body) = json_body(rig.app.clone().oneshot(get("nobody-key")).await.unwrap()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "invalid_api_key", "the refusal keeps the dialect's shape");

    let (status, _) = json_body(rig.app.clone().oneshot(get("static-key")).await.unwrap()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "the surface's own key no longer opens anything");
}

#[tokio::test]
async fn hooks_see_the_caller_the_call_and_the_settled_outcome_exactly_once() {
    let rig = rig(Provider::OpenAI);
    let (status, _) = json_body(rig.app.clone().oneshot(chat_request("alice-key", false)).await.unwrap()).await;
    assert_eq!(status, StatusCode::OK);

    let log = rig.log.lock().unwrap();
    assert_eq!(log.begun, [("alice".to_string(), "brain-chat".to_string(), "generate".to_string())]);
    assert_eq!(log.settled, [Settled::Done { prompt: 5, completion: 2 }]);
}

#[tokio::test]
async fn a_hook_refusal_stops_the_request_before_the_model_runs() {
    let rig = rig(Provider::OpenAI);
    let (status, body) = json_body(rig.app.clone().oneshot(chat_request("broke-key", false)).await.unwrap()).await;
    assert_eq!(status, StatusCode::PAYMENT_REQUIRED);
    assert_eq!(body["error"]["code"], "insufficient_quota");
    assert_eq!(rig.runs.load(Ordering::SeqCst), 0, "a refused call never reaches the model");
    let log = rig.log.lock().unwrap();
    assert!(log.begun.is_empty() && log.settled.is_empty(), "a refusal leaves nothing to settle");
}

#[tokio::test]
async fn a_completed_stream_is_settled_once_with_its_final_counts() {
    let rig = rig(Provider::OpenAI);
    let response = rig.app.clone().oneshot(chat_request("alice-key", true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();

    eventually("the stream's settlement", || rig.log.lock().unwrap().settled.len() == 1).await;
    assert_eq!(rig.log.lock().unwrap().settled, [Settled::Done { prompt: 5, completion: 2 }]);
}

#[tokio::test]
async fn a_stream_the_client_abandons_is_still_settled_exactly_once() {
    let rig = rig(Provider::OpenAI);
    let response = rig.app.clone().oneshot(chat_request("alice-key", true)).await.unwrap();
    let mut body = response.into_body().into_data_stream();
    let _ = body.next().await; // one frame, then the client walks away
    drop(body);

    eventually("the abandoned stream's settlement", || !rig.log.lock().unwrap().settled.is_empty()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(rig.log.lock().unwrap().settled.len(), 1, "settled once, never twice and never not at all");
}

#[tokio::test]
async fn a_settlement_that_cannot_be_recorded_fails_the_request() {
    let rig = rig(Provider::OpenAI);
    let (status, body) = json_body(rig.app.clone().oneshot(chat_request("unrecordable-key", false)).await.unwrap()).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "an unbilled answer is not served: {body}");
    assert_eq!(rig.log.lock().unwrap().settled.len(), 1);
}

#[tokio::test]
async fn the_anthropic_dialect_authenticates_by_x_api_key_and_settles_the_same_way() {
    let rig = rig(Provider::Anthropic);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("x-api-key", "alice-key")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({"model": "brain-chat", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}).to_string()))
        .unwrap();
    let (status, _) = json_body(rig.app.clone().oneshot(request).await.unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    let log = rig.log.lock().unwrap();
    assert_eq!(log.begun.len(), 1);
    assert_eq!(log.begun[0].0, "alice");
    assert_eq!(log.settled, [Settled::Done { prompt: 5, completion: 2 }]);
}

#[tokio::test]
async fn hooks_decide_which_models_are_listed() {
    let rig = rig(Provider::OpenAI);
    let get = |uri: &str| Request::builder().uri(uri.to_string()).header(header::AUTHORIZATION, "Bearer alice-key").body(Body::empty()).unwrap();

    let (_, body) = json_body(rig.app.clone().oneshot(get("/models")).await.unwrap()).await;
    assert_eq!(body["data"].as_array().unwrap().len(), 1, "a model is listed unless the hooks say otherwise");
    let (_, body) = json_body(rig.app.clone().oneshot(get("/v1/capabilities")).await.unwrap()).await;
    assert_eq!(body["data"].as_array().unwrap().len(), 1);

    rig.log.lock().unwrap().hidden = true;

    let (_, body) = json_body(rig.app.clone().oneshot(get("/models")).await.unwrap()).await;
    assert!(body["data"].as_array().unwrap().is_empty(), "a hidden model is not listed: {body}");
    let (status, _) = json_body(rig.app.clone().oneshot(get("/models/brain-chat")).await.unwrap()).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "nor can it be looked up by name");
    let (_, body) = json_body(rig.app.clone().oneshot(get("/v1/capabilities")).await.unwrap()).await;
    assert!(body["data"].as_array().unwrap().is_empty());
}
