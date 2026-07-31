// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P4 apiserve tests — auth, `/models` shape (validated against the vendored
//! OpenAPI specs), the 501 stubs, and the 404 route table. Drives each provider
//! with `tower::ServiceExt::oneshot` against `apiserve::router(state)` — no socket,
//! no GPU. `Executor::start(...).manifests()` works CPU-only.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

// ------------------------------------------------------------------ fake models

/// A carded resident model with a fixed manifest; never actually run in P4 tests.
struct Carded(Manifest);
struct Never;
impl ResidentModel for Carded {
    fn manifest(&self) -> Manifest {
        self.0.clone()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.0.model.clone(), "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(Never))
    }
}
impl Instance for Never {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        Err("not implemented in P4 tests".into())
    }
}

/// A chat model: streaming `generate(prompt) -> Text`.
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

/// An embed-only model: `embed(text) -> embedding bytes`.
fn embed_manifest() -> Manifest {
    Manifest::new(
        "brain-embed",
        "an embedding model",
        vec![ActionSpec::new("embed", "embed text")
            .param(ParamSpec::new("text", ParamType::Str, "input text").required())
            .output(BlobSpec::new("embedding", Media::Bytes, "embedding vector"))],
    )
}

fn executor() -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Carded(chat_manifest())), Arc::new(Carded(embed_manifest()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

/// A runnable fake chat model: `generate` emits two token deltas ("Hello", " world")
/// then returns a canned outcome — text blob "Hello world" + prompt/completion
/// counts + finish_reason "stop". Zero-cost so it schedules on a CPU-only budget.
struct FakeChat;
struct FakeChatInst;
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
        Ok(Box::new(FakeChatInst))
    }
}
impl Instance for FakeChatInst {
    fn run(&mut self, _a: &str, _i: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        for (i, piece) in ["Hello", " world"].iter().enumerate() {
            progress(Progress::step(i as u32, 2, "phase")); // coarse step: ignored by SSE
            progress(Progress::token(i as u32, 2, *piece));
        }
        Ok(Outcome::new()
            .set("prompt_tokens", json!(5))
            .set("completion_tokens", json!(2))
            .set("finish_reason", json!("stop"))
            .blob("text", Blob::new(Media::Text, b"Hello world".to_vec())))
    }
}

/// An executor whose chat model is the RUNNABLE [`FakeChat`], plus a carded (never
/// run) embed model so "non-chat model -> 404" can be exercised.
fn chat_executor() -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeChat), Arc::new(Carded(embed_manifest()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

/// The fixed mean-pooled vector the [`FakeEmbed`] model returns for any input.
const FAKE_EMBED: [f32; 4] = [0.1, 0.2, 0.3, 0.4];

/// A runnable fake embeddings model: `embed` returns a fixed `mean` vector plus a
/// `tokens`/`dim` count, mirroring the LFM encoder's outcome shape. Zero-cost so it
/// schedules on a CPU-only budget.
struct FakeEmbed;
struct FakeEmbedInst;
impl ResidentModel for FakeEmbed {
    fn manifest(&self) -> Manifest {
        embed_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-embed", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeEmbedInst))
    }
}
impl Instance for FakeEmbedInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        Ok(Outcome::new()
            .set("tokens", json!(3))
            .set("dim", json!(FAKE_EMBED.len()))
            .set("mean", json!(FAKE_EMBED.to_vec()))
            .blob("embeddings", Blob::new(Media::Bytes, Vec::new())))
    }
}

/// An executor with the RUNNABLE [`FakeEmbed`] model plus the RUNNABLE [`FakeChat`]
/// (so "non-embeddings model -> 404" can be exercised on a real chat model).
fn embed_executor() -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeEmbed), Arc::new(FakeChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

fn embed_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(embed_executor(), key.clone(), provider);
    (router(state), key)
}

fn chat_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(chat_executor(), key.clone(), provider);
    (router(state), key)
}

fn build_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(executor(), key.clone(), provider);
    (router(state), key)
}

// --------------------------------------------------------------- http helpers

fn auth(req: axum::http::request::Builder, provider: Provider, key: &str) -> axum::http::request::Builder {
    match provider {
        Provider::Anthropic => req.header("x-api-key", key),
        Provider::OpenAI | Provider::OpenRouter => req.header(header::AUTHORIZATION, format!("Bearer {key}")),
    }
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap() };
    (status, body)
}

const ALL: [Provider; 3] = [Provider::OpenAI, Provider::Anthropic, Provider::OpenRouter];

// ------------------------------------------------------------- jsonschema help

fn spec(file: &str) -> Value {
    let p: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/specs").join(file);
    let mut root: Value = serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))).unwrap();
    sanitize(&mut root);
    root
}

/// Make a vendored OpenAPI 3.x spec compile + validate under strict Draft 2020-12:
/// the vendored docs use OpenAPI's `nullable: true` (which 2020-12 ignores, so a
/// legitimately-null value like a streaming `finish_reason` would be rejected) and
/// one carries an invalid `"type": null` (Anthropic's `Model`, which won't even
/// compile). Rewrite both into standard 2020-12 (widen `type`/`enum` to admit null,
/// wrap a nullable `$ref`, drop the bad `type`).
fn sanitize(v: &mut Value) {
    match v {
        Value::Object(map) => {
            if map.get("type") == Some(&Value::Null) {
                map.remove("type");
            }
            let nullable = map.remove("nullable") == Some(Value::Bool(true));
            if nullable {
                allow_null(map);
            }
            for child in map.values_mut() {
                sanitize(child);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(sanitize),
        _ => {}
    }
}

/// Extend a schema object so JSON `null` is a valid instance.
fn allow_null(map: &mut serde_json::Map<String, Value>) {
    if let Some(Value::Array(e)) = map.get_mut("enum") {
        if !e.iter().any(Value::is_null) {
            e.push(Value::Null);
        }
    }
    match map.get("type").cloned() {
        Some(Value::String(s)) => {
            map.insert("type".into(), json!([s, "null"]));
        }
        Some(Value::Array(mut arr)) => {
            if !arr.iter().any(|x| x == &json!("null")) {
                arr.push(json!("null"));
            }
            map.insert("type".into(), Value::Array(arr));
        }
        _ if map.contains_key("$ref") => {
            let r = map.remove("$ref").unwrap();
            map.insert("anyOf".into(), json!([{ "$ref": r }, { "type": "null" }]));
        }
        _ => {
            if let Some(Value::Array(a)) = map.get_mut("anyOf") {
                a.push(json!({ "type": "null" }));
            } else if let Some(Value::Array(a)) = map.get_mut("oneOf") {
                a.push(json!({ "type": "null" }));
            }
        }
    }
}

/// Validate `instance` against `#/components/schemas/<schema>` of vendored `file`
/// (2020-12; the OpenAPI docs are 3.1). Uses the sibling-`$ref` trick so internal
/// `#/components/schemas/...` refs resolve within the same document.
fn assert_valid(file: &str, schema: &str, instance: &Value) {
    let mut root = spec(file);
    root.as_object_mut().unwrap().insert("$ref".into(), json!(format!("#/components/schemas/{schema}")));
    let validator = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(&root)
        .unwrap_or_else(|e| panic!("compiling {file}#{schema}: {e}"));
    if !validator.is_valid(instance) {
        let errs: Vec<String> = validator.iter_errors(instance).map(|e| format!("  - {e} (at {})", e.instance_path())).collect();
        panic!("{file}#{schema} rejected:\n{}\ninstance = {}", errs.join("\n"), serde_json::to_string_pretty(instance).unwrap());
    }
}

/// The vendored spec + canonical error schema for a provider.
fn error_schema(p: Provider) -> (&'static str, &'static str) {
    match p {
        Provider::OpenAI => ("openai.json", "ErrorResponse"),
        Provider::Anthropic => ("anthropic.json", "ErrorResponse"),
        Provider::OpenRouter => ("openrouter.json", "InternalServerResponse"),
    }
}

// -------------------------------------------------------------------- tests

#[tokio::test]
async fn auth_missing_blank_wrong_are_401_and_correct_is_200() {
    for p in ALL {
        let (app, key) = build_app(p);
        // missing key
        let (st, body) = send(&app, Request::builder().uri("/models").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: missing key must 401");
        let (file, schema) = if p == Provider::OpenRouter { ("openrouter.json", "UnauthorizedResponse") } else { error_schema(p) };
        assert_valid(file, schema, &body);

        // blank key
        let (st, _) = send(&app, auth(Request::builder().uri("/models"), p, "").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: blank key must 401");

        // wrong key
        let (st, _) = send(&app, auth(Request::builder().uri("/models"), p, "nope").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: wrong key must 401");

        // correct key
        let (st, _) = send(&app, auth(Request::builder().uri("/models"), p, &key).body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::OK, "{p}: correct key must 200");
    }
}

#[tokio::test]
async fn models_list_validates_and_honors_capability_filter() {
    // OpenAI: both models exposed; validate against ListModelsResponse.
    let (app, key) = build_app(Provider::OpenAI);
    let (st, body) = send(&app, auth(Request::builder().uri("/v1/models"), Provider::OpenAI, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "ListModelsResponse", &body);
    let ids: Vec<&str> = body["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"brain-chat"), "openai must list the chat model: {ids:?}");
    assert!(ids.contains(&"brain-embed"), "openai must list the embed model: {ids:?}");

    // OpenRouter: both exposed; validate against ModelsListResponse.
    let (app, key) = build_app(Provider::OpenRouter);
    let (st, body) = send(&app, auth(Request::builder().uri("/models"), Provider::OpenRouter, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openrouter.json", "ModelsListResponse", &body);

    // Anthropic: chat exposed, embed NOT (chat-only surface). No vendored list
    // schema, so validate the item shape structurally.
    let (app, key) = build_app(Provider::Anthropic);
    let (st, body) = send(&app, auth(Request::builder().uri("/v1/models"), Provider::Anthropic, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    let data = body["data"].as_array().unwrap();
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"brain-chat"), "anthropic must list the chat model: {ids:?}");
    assert!(!ids.contains(&"brain-embed"), "anthropic must NOT list the embed-only model: {ids:?}");
    for m in data {
        assert_eq!(m["type"], "model");
        assert!(m["id"].is_string() && m["display_name"].is_string() && m["created_at"].is_string());
    }
}

#[tokio::test]
async fn get_model_by_id_and_404_when_not_exposed() {
    // OpenAI: chat model card validates against Model; unknown id -> 404.
    let (app, key) = build_app(Provider::OpenAI);
    let (st, body) = send(&app, auth(Request::builder().uri("/v1/models/brain-chat"), Provider::OpenAI, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "Model", &body);

    let (st, body) = send(&app, auth(Request::builder().uri("/v1/models/nope"), Provider::OpenAI, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_valid("openai.json", "ErrorResponse", &body);

    // Anthropic: the embed-only model is not exposed here -> 404.
    let (app, key) = build_app(Provider::Anthropic);
    let (st, body) = send(&app, auth(Request::builder().uri("/v1/models/brain-embed"), Provider::Anthropic, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "anthropic must 404 an embed-only model");
    assert_valid("anthropic.json", "ErrorResponse", &body);
}

#[tokio::test]
async fn embeddings_images_are_501_with_spec_valid_bodies() {
    // Chat, Anthropic count_tokens, and embeddings are now implemented; image
    // generation remains 501. (provider, path)
    let cases = [
        (Provider::OpenAI, "/v1/images/generations"),
        (Provider::OpenRouter, "/images/generations"),
    ];
    for (p, path) in cases {
        let (app, key) = build_app(p);
        let req = auth(Request::builder().method(Method::POST).uri(path), p, &key).body(Body::from("{}")).unwrap();
        let (st, body) = send(&app, req).await;
        assert_eq!(st, StatusCode::NOT_IMPLEMENTED, "{p} {path} must 501");
        let (file, schema) = error_schema(p);
        assert_valid(file, schema, &body);
    }
}

#[tokio::test]
async fn unknown_path_is_404() {
    for p in ALL {
        let (app, key) = build_app(p);
        let (st, _) = send(&app, auth(Request::builder().uri("/no/such/route"), p, &key).body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::NOT_FOUND, "{p}: unknown path must 404");
    }
}

#[test]
fn api_caps_derives_from_manifest_shape() {
    let chat = apiserve::api_caps(&chat_manifest());
    assert!(chat.chat && !chat.embeddings && !chat.image);
    let embed = apiserve::api_caps(&embed_manifest());
    assert!(embed.embeddings && !embed.chat && !embed.image);
}

// ---------------------------------------------------------------- chat helpers

/// POST `path` with a JSON `body` (auth for `p`), returning `(status, text)`.
async fn post_text(app: &Router, p: Provider, key: &str, path: &str, body: &Value) -> (StatusCode, String) {
    let req = auth(Request::builder().method(Method::POST).uri(path), p, key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// POST `path` with a JSON `body`, returning `(status, json)`.
async fn post_json(app: &Router, p: Provider, key: &str, path: &str, body: &Value) -> (StatusCode, Value) {
    let (st, text) = post_text(app, p, key, path, body).await;
    let v = if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap() };
    (st, v)
}

/// The `data:` payloads of an SSE body, in order (`event:` names dropped).
fn sse_data(body: &str) -> Vec<String> {
    body.lines().filter_map(|l| l.strip_prefix("data: ").map(|s| s.to_string())).collect()
}

/// `(event, data)` pairs of an SSE body, in order.
fn sse_events(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut ev = String::new();
    for l in body.lines() {
        if let Some(e) = l.strip_prefix("event: ") {
            ev = e.to_string();
        } else if let Some(d) = l.strip_prefix("data: ") {
            out.push((ev.clone(), d.to_string()));
        }
    }
    out
}

// ----------------------------------------------------------- chat: non-stream

#[tokio::test]
async fn openai_chat_nonstream_validates_usage_and_finish() {
    let (app, key) = chat_app(Provider::OpenAI);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "CreateChatCompletionResponse", &v);
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 5);
    assert_eq!(v["usage"]["completion_tokens"], 2);
    assert_eq!(v["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn anthropic_messages_nonstream_validates_and_maps_stop_reason() {
    let (app, key) = chat_app(Provider::Anthropic);
    let body = json!({"model": "brain-chat", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("anthropic.json", "Message", &v);
    assert_eq!(v["type"], "message");
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "Hello world");
    assert_eq!(v["stop_reason"], "end_turn"); // contract "stop" -> anthropic "end_turn"
    assert_eq!(v["usage"]["input_tokens"], 5);
    assert_eq!(v["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn openrouter_chat_nonstream_validates_and_has_native_finish_reason() {
    let (app, key) = chat_app(Provider::OpenRouter);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openrouter.json", "ChatResult", &v);
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["native_finish_reason"], "stop"); // mirrors finish_reason
}

#[tokio::test]
async fn anthropic_count_tokens_returns_input_tokens_approximation() {
    let (app, key) = chat_app(Provider::Anthropic);
    // 40 chars of content -> heuristic ~ chars/4.
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "0123456789012345678901234567890123456789"}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages/count_tokens", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("anthropic.json", "BetaCountMessageTokensResponse", &v);
    assert!(v["input_tokens"].as_i64().unwrap() > 0, "approx count must be positive: {v}");
}

// -------------------------------------------------------------- chat: errors

#[tokio::test]
async fn chat_unknown_and_non_chat_models_are_404() {
    // OpenAI: unknown model id -> 404.
    let (app, key) = chat_app(Provider::OpenAI);
    let body = json!({"model": "nope", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "unknown model must 404");
    assert_valid("openai.json", "ErrorResponse", &v);

    // OpenAI: an existing but non-chat (embed-only) model -> 404.
    let body = json!({"model": "brain-embed", "messages": [{"role": "user", "content": "hi"}]});
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "non-chat model must 404");

    // Anthropic: same, on /v1/messages.
    let (app, key) = chat_app(Provider::Anthropic);
    let body = json!({"model": "nope", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_valid("anthropic.json", "ErrorResponse", &v);
}

#[tokio::test]
async fn chat_bad_bodies_are_400() {
    // OpenAI: malformed JSON, missing model, missing messages, n > 1.
    let (app, key) = chat_app(Provider::OpenAI);
    let cases: [Value; 3] = [
        json!({"messages": [{"role": "user", "content": "hi"}]}),   // no model
        json!({"model": "brain-chat"}),                              // no messages
        json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}], "n": 2}), // n > 1
    ];
    for body in cases {
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "must 400: {body}");
        assert_valid("openai.json", "ErrorResponse", &v);
    }
    // Malformed JSON body.
    let req = auth(Request::builder().method(Method::POST).uri("/v1/chat/completions"), Provider::OpenAI, &key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{ not json"))
        .unwrap();
    let (st, _) = send(&app, req).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    // Anthropic requires max_tokens.
    let (app, key) = chat_app(Provider::Anthropic);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, _) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "anthropic must 400 without max_tokens");
}

// -------------------------------------------------------------- chat: SSE

#[tokio::test]
async fn openai_chat_stream_orders_chunks_and_concatenates() {
    let (app, key) = chat_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let datas = sse_data(&text);
    assert_eq!(datas.last().map(String::as_str), Some("[DONE]"), "stream must terminate with [DONE]");

    // First chunk announces the assistant role.
    let first: Value = serde_json::from_str(&datas[0]).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

    let mut content = String::new();
    let mut saw_finish = false;
    let mut saw_usage = false;
    for d in datas.iter().filter(|d| d.as_str() != "[DONE]") {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_valid("openai.json", "CreateChatCompletionStreamResponse", &v);
        assert_eq!(v["object"], "chat.completion.chunk");
        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
            content.push_str(c);
        }
        if v["choices"][0]["finish_reason"].as_str() == Some("stop") {
            saw_finish = true;
        }
        if v.get("usage").map(|u| !u.is_null()).unwrap_or(false) {
            saw_usage = true;
            assert_eq!(v["usage"]["total_tokens"], 7);
        }
    }
    assert_eq!(content, "Hello world", "concatenated deltas must equal the full text");
    assert!(saw_finish, "a terminal chunk must carry finish_reason=stop");
    assert!(saw_usage, "include_usage must emit a usage chunk");
}

#[tokio::test]
async fn anthropic_messages_stream_orders_events_and_concatenates() {
    let (app, key) = chat_app(Provider::Anthropic);
    let body = json!({
        "model": "brain-chat",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let (st, text) = post_text(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK);
    let events = sse_events(&text);
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();

    // Exact terminal + presence of each phase.
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    let pos = |name: &str| names.iter().position(|n| *n == name).unwrap_or_else(|| panic!("missing event {name}: {names:?}"));
    // Ordering: start < block_start < first delta < block_stop < message_delta < stop.
    assert!(pos("message_start") < pos("content_block_start"));
    assert!(pos("content_block_start") < pos("content_block_delta"));
    assert!(pos("content_block_delta") < pos("content_block_stop"));
    assert!(pos("content_block_stop") < pos("message_delta"));
    assert!(pos("message_delta") < pos("message_stop"));

    // Each event validates against its schema.
    for (e, d) in &events {
        let v: Value = serde_json::from_str(d).unwrap();
        let schema = match e.as_str() {
            "message_start" => "MessageStartEvent",
            "content_block_start" => "ContentBlockStartEvent",
            "content_block_delta" => "ContentBlockDeltaEvent",
            "content_block_stop" => "ContentBlockStopEvent",
            "message_delta" => "MessageDeltaEvent",
            "message_stop" => "MessageStopEvent",
            other => panic!("unexpected event {other}"),
        };
        assert_valid("anthropic.json", schema, &v);
    }

    // Concatenated text_delta pieces == the full text.
    let cat: String = events
        .iter()
        .filter(|(e, _)| e == "content_block_delta")
        .map(|(_, d)| serde_json::from_str::<Value>(d).unwrap()["delta"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(cat, "Hello world");

    // message_delta carries the mapped stop_reason + cumulative output tokens.
    let (_, md) = events.iter().find(|(e, _)| e == "message_delta").unwrap();
    let v: Value = serde_json::from_str(md).unwrap();
    assert_eq!(v["delta"]["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn openrouter_chat_stream_validates_and_carries_native_finish_reason() {
    let (app, key) = chat_app(Provider::OpenRouter);
    let body = json!({
        "model": "brain-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let (st, text) = post_text(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let datas = sse_data(&text);
    assert_eq!(datas.last().map(String::as_str), Some("[DONE]"));
    let mut saw_native = false;
    for d in datas.iter().filter(|d| d.as_str() != "[DONE]") {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_valid("openrouter.json", "ChatStreamChunk", &v);
        if v["choices"][0]["finish_reason"].as_str() == Some("stop") {
            assert_eq!(v["choices"][0]["native_finish_reason"], "stop");
            saw_native = true;
        }
    }
    assert!(saw_native, "final chunk must carry native_finish_reason");
}

// ------------------------------------------------------------- P9: embeddings

/// OpenAI single-string input: a spec-valid `CreateEmbeddingResponse` with one
/// float embedding, index 0, and non-zero usage.
#[tokio::test]
async fn openai_embeddings_single_string_validates() {
    let (app, key) = embed_app(Provider::OpenAI);
    let body = json!({"model": "brain-embed", "input": "hello world"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::OK, "single-string embeddings must 200: {v}");
    assert_valid("openai.json", "CreateEmbeddingResponse", &v);
    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "brain-embed");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let emb: Vec<f64> = data[0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(emb.len(), 4);
    assert!((emb[0] - 0.1).abs() < 1e-6 && (emb[3] - 0.4).abs() < 1e-6, "float embedding must match the model vector: {emb:?}");
    assert_eq!(v["usage"]["prompt_tokens"], 3);
    assert_eq!(v["usage"]["total_tokens"], 3);
}

/// OpenAI array-of-strings input: one `data` entry per input, indices 0..n, and
/// usage summed across inputs.
#[tokio::test]
async fn openai_embeddings_array_of_strings_indices_and_usage() {
    let (app, key) = embed_app(Provider::OpenAI);
    let body = json!({"model": "brain-embed", "input": ["a", "b", "c"]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "CreateEmbeddingResponse", &v);
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 3, "one embedding per input string");
    let idx: Vec<i64> = data.iter().map(|d| d["index"].as_i64().unwrap()).collect();
    assert_eq!(idx, vec![0, 1, 2], "indices must be request order");
    assert_eq!(v["usage"]["prompt_tokens"], 9, "3 tokens * 3 inputs summed");
    assert_eq!(v["usage"]["total_tokens"], 9);
}

/// `encoding_format: "base64"` yields a base64 string that decodes to the exact
/// same floats as the float form.
#[tokio::test]
async fn openai_embeddings_base64_decodes_to_same_floats() {
    let (app, key) = embed_app(Provider::OpenAI);
    let body = json!({"model": "brain-embed", "input": "hi", "encoding_format": "base64"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::OK);
    let s = v["data"][0]["embedding"].as_str().expect("base64 embedding must be a string");
    let floats = events::bytes::decode_f32(s).expect("must decode as LE-f32");
    assert_eq!(floats, vec![0.1_f32, 0.2, 0.3, 0.4], "base64 must decode to the model vector");
}

/// `dimensions` less than the vector length truncates; greater than it is a 400.
#[tokio::test]
async fn openai_embeddings_dimensions_truncate_and_overflow() {
    let (app, key) = embed_app(Provider::OpenAI);
    // Truncate 4 -> 2.
    let body = json!({"model": "brain-embed", "input": "hi", "dimensions": 2});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "CreateEmbeddingResponse", &v);
    let emb: Vec<f64> = v["data"][0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(emb.len(), 2, "dimensions=2 must truncate to the first 2 dims");
    assert!((emb[0] - 0.1).abs() < 1e-6 && (emb[1] - 0.2).abs() < 1e-6);

    // Too many dims -> 400.
    let body = json!({"model": "brain-embed", "input": "hi", "dimensions": 8});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "dimensions > vector length must 400");
    assert_valid("openai.json", "ErrorResponse", &v);
}

/// Unknown model -> 404; an existing non-embeddings (chat) model -> 404.
#[tokio::test]
async fn openai_embeddings_unknown_and_non_embed_models_are_404() {
    let (app, key) = embed_app(Provider::OpenAI);
    let body = json!({"model": "nope", "input": "hi"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "unknown model must 404");
    assert_valid("openai.json", "ErrorResponse", &v);

    let body = json!({"model": "brain-chat", "input": "hi"});
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "a non-embeddings (chat) model must 404");
}

/// Bad bodies -> 400: malformed JSON, missing model, missing input, empty input,
/// token-array input, and a bad encoding_format.
#[tokio::test]
async fn openai_embeddings_bad_bodies_are_400() {
    let (app, key) = embed_app(Provider::OpenAI);
    let cases: [Value; 5] = [
        json!({"input": "hi"}),                                          // no model
        json!({"model": "brain-embed"}),                                 // no input
        json!({"model": "brain-embed", "input": ""}),                    // empty string
        json!({"model": "brain-embed", "input": [1212, 318, 257]}),      // token array
        json!({"model": "brain-embed", "input": "hi", "encoding_format": "bogus"}), // bad format
    ];
    for body in cases {
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/embeddings", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "must 400: {body}");
        assert_valid("openai.json", "ErrorResponse", &v);
    }
    // Malformed JSON body.
    let req = auth(Request::builder().method(Method::POST).uri("/v1/embeddings"), Provider::OpenAI, &key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{ not json"))
        .unwrap();
    let (st, _) = send(&app, req).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
}

/// OpenRouter `/embeddings` works the same (same grammar, spec-valid response).
#[tokio::test]
async fn openrouter_embeddings_works_the_same() {
    let (app, key) = embed_app(Provider::OpenRouter);
    let body = json!({"model": "brain-embed", "input": ["x", "y"]});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/embeddings", &body).await;
    assert_eq!(st, StatusCode::OK, "openrouter embeddings must 200: {v}");
    // The response is the OpenAI embeddings shape; validate against that schema.
    assert_valid("openai.json", "CreateEmbeddingResponse", &v);
    assert_eq!(v["data"].as_array().unwrap().len(), 2);

    // Unknown model still 404 (OpenRouter error shape).
    let body = json!({"model": "nope", "input": "hi"});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/embeddings", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_valid("openrouter.json", "InternalServerResponse", &v);
}

// ------------------------------------------------- P6: admission + backpressure

/// A shared gate the test uses to (a) pin the single CPU lane and (b) count how
/// many jobs did REAL work (passed the cancel guard and started running). A
/// well-behaved model drops a cancelled job before touching the gate, so a job that
/// was shed at admission (its token cancelled) never bumps `started`.
#[derive(Clone)]
struct Gate {
    inner: Arc<(Mutex<bool>, Condvar)>,
    started: Arc<AtomicUsize>,
}
impl Gate {
    fn new() -> Gate {
        Gate { inner: Arc::new((Mutex::new(false), Condvar::new())), started: Arc::new(AtomicUsize::new(0)) }
    }
    /// Block the caller (the lane thread) until [`Gate::release`].
    fn wait(&self) {
        let (m, cv) = &*self.inner;
        let mut open = m.lock().unwrap();
        while !*open {
            open = cv.wait(open).unwrap();
        }
    }
    /// Let every waiting lane run to completion.
    fn release(&self) {
        let (m, cv) = &*self.inner;
        *m.lock().unwrap() = true;
        cv.notify_all();
    }
    fn started(&self) -> usize {
        self.started.load(Ordering::SeqCst)
    }
}

/// A chat model whose `run` blocks the lane until the gate is released — pinning the
/// single CPU lane so a second request cannot be admitted. A cancelled invocation
/// (e.g. one shed at admission whose token was cancelled) returns immediately WITHOUT
/// bumping `started`, proving a timed-out job never runs wastefully.
struct SlowChat(Gate);
struct SlowChatInst(Gate);
impl ResidentModel for SlowChat {
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
        Ok(Box::new(SlowChatInst(self.0.clone())))
    }
}
impl Instance for SlowChatInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        // Drop cancelled work before doing anything (as a real model would).
        if inv.cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        self.0.started.fetch_add(1, Ordering::SeqCst);
        self.0.wait(); // pin the lane until the test releases us
        Ok(Outcome::new()
            .set("prompt_tokens", json!(1))
            .set("completion_tokens", json!(1))
            .set("finish_reason", json!("stop"))
            .blob("text", Blob::new(Media::Text, b"done".to_vec())))
    }
}

/// A single-CPU-lane executor whose one chat model blocks on `gate`.
fn slow_executor(gate: Gate) -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(SlowChat(gate))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

/// POST returning `(status, headers, json)` — keeps the response headers so the test
/// can assert `Retry-After` + `content-type` (the shared `send`/`post_json` helpers
/// discard them).
async fn post_full(app: &Router, p: Provider, key: &str, path: &str, body: &Value) -> (StatusCode, axum::http::HeaderMap, Value) {
    let req = auth(Request::builder().method(Method::POST).uri(path), p, key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap() };
    (status, headers, v)
}

/// The single CPU lane is pinned by one in-flight request; a second request that
/// cannot be ADMITTED within the (short) deadline is shed with a 429 — promptly, not
/// hanging — carrying `Retry-After` and a spec-valid provider error body. The same
/// holds for a streaming request: it gets a PLAIN 429 JSON, not an SSE stream that
/// then errors. A shed job's token is cancelled, so it never runs once the lane frees.
#[tokio::test]
async fn admit_deadline_sheds_saturated_lane_with_429_and_cancels() {
    let gate = Gate::new();
    let key = "sk-brain-test-key".to_string();
    let deadline = Duration::from_millis(200);
    let state = AppState::new(slow_executor(gate.clone()), key.clone(), Provider::OpenAI).with_admit_deadline(deadline);
    let app = router(state);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});

    // Job 1 occupies the single CPU lane. Its response won't resolve until we release
    // the gate, so drive it on a background task.
    let (app1, key1, b1) = (app.clone(), key.clone(), body.clone());
    let job1 = tokio::spawn(async move { post_json(&app1, Provider::OpenAI, &key1, "/v1/chat/completions", &b1).await });

    // Wait until job 1 is actually RUNNING on the lane (past the cancel guard).
    let until = Instant::now() + Duration::from_secs(5);
    while gate.started() == 0 {
        assert!(Instant::now() < until, "job 1 never started on the lane");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Job 2 (non-stream): the lane is pinned, so it cannot be admitted in 200ms -> 429.
    let t0 = Instant::now();
    let (st, headers, v) = post_full(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    let elapsed = t0.elapsed();
    assert_eq!(st, StatusCode::TOO_MANY_REQUESTS, "saturated lane must shed with 429: {v}");
    assert!(elapsed < Duration::from_secs(2), "429 must return near the deadline, not hang: {elapsed:?}");
    assert!(headers.contains_key(header::RETRY_AFTER), "429 must carry Retry-After: {headers:?}");
    assert_valid("openai.json", "ErrorResponse", &v);

    // Job 3 (stream): the SSE admit race must ALSO shed BEFORE any event-stream body —
    // a plain 429 JSON, not text/event-stream.
    let stream_body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, headers, v) = post_full(&app, Provider::OpenAI, &key, "/v1/chat/completions", &stream_body).await;
    assert_eq!(st, StatusCode::TOO_MANY_REQUESTS, "streaming request on a saturated lane must shed with 429: {v}");
    assert!(headers.contains_key(header::RETRY_AFTER), "streamed 429 must carry Retry-After");
    let ctype = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
    assert!(ctype.starts_with("application/json"), "shed stream must be a plain JSON 429, not SSE: {ctype}");
    assert_valid("openai.json", "ErrorResponse", &v);

    // Only ONE job ever did real work; the shed jobs were cancelled and, even after
    // the lane frees, must not run.
    assert_eq!(gate.started(), 1, "no shed job may run before the lane frees");
    gate.release();
    let (st1, _v1) = job1.await.unwrap();
    assert_eq!(st1, StatusCode::OK, "the admitted job 1 must complete once released");

    // Give the dispatcher a moment to (re)visit the queue now the lane is free: the
    // cancelled jobs must NOT have run.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(gate.started(), 1, "a timed-out (cancelled) job must not run after the lane frees");
}
