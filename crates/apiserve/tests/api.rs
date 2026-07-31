// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P4 apiserve tests — auth, `/models` shape (validated against the vendored
//! OpenAPI specs), the 501 stubs, and the 404 route table. Drives each provider
//! with `tower::ServiceExt::oneshot` against `apiserve::router(state)` — no socket,
//! no GPU. `Executor::start(...).manifests()` works CPU-only.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress};
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
    serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))).unwrap()
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
async fn chat_embeddings_images_are_501_with_spec_valid_bodies() {
    // (provider, path)
    let cases = [
        (Provider::OpenAI, "/v1/chat/completions"),
        (Provider::OpenAI, "/v1/embeddings"),
        (Provider::OpenAI, "/v1/images/generations"),
        (Provider::OpenRouter, "/chat/completions"),
        (Provider::OpenRouter, "/embeddings"),
        (Provider::OpenRouter, "/images/generations"),
        (Provider::Anthropic, "/v1/messages"),
        (Provider::Anthropic, "/v1/messages/count_tokens"),
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
