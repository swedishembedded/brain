// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P4 apiserve tests — auth, `/models` shape (validated against the vendored
//! OpenAPI specs), the 501 stubs, and the 404 route table. Drives each provider
//! with `tower::ServiceExt::oneshot` against `apiserve::router(state)` — no socket,
//! no GPU. `Executor::start(...).manifests()` works CPU-only.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use futures::StreamExt;
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
            .param(ParamSpec::new("seed", ParamType::Int, "sampling seed"))
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

/// An image model: streaming `text2image(prompt, width, height, seed) -> Image`.
fn image_manifest() -> Manifest {
    Manifest::new(
        "brain-image",
        "an image model",
        vec![ActionSpec::new("text2image", "generate an image from a prompt")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("width", ParamType::Int, "width").default(json!(1024)))
            // The tallest image this model draws.
            .param(ParamSpec::new("height", ParamType::Int, "height").default(json!(1024)).max(1024.0))
            .param(ParamSpec::new("seed", ParamType::Int, "seed"))
            .output(BlobSpec::new("image", Media::Image, "the generated image"))],
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

/// The fixed 2×2 RGB image the [`FakeImage`] model returns, as interleaved HWC f32
/// in `[0,1]`: red, green, blue, white. Chosen from {0.0, 1.0} so the 8-bit quantise
/// is exact (0 / 255), letting a test compare the served PNG byte-for-byte.
const FAKE_IMG_HWC: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
/// The same pixels quantised to RGB8 — the input the handler PNG-encodes.
const FAKE_IMG_RGB8: [u8; 12] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];

/// A runnable fake image model: `text2image` emits two coarse denoise-step progress
/// ticks (delta `None`) then returns a 2×2 raw HWC-f32 image blob (the brain image
/// wire format). Zero-cost so it schedules on a CPU-only budget.
struct FakeImage;
struct FakeImageInst;
impl ResidentModel for FakeImage {
    fn manifest(&self) -> Manifest {
        image_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-image", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeImageInst))
    }
}
impl Instance for FakeImageInst {
    fn run(&mut self, _a: &str, _i: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Two denoise steps (coarse, delta None) → two partial_image events on SSE.
        progress(Progress::step(1, 2, "denoise"));
        progress(Progress::step(2, 2, "denoise"));
        Ok(Outcome::new()
            .set("width", json!(2))
            .set("height", json!(2))
            .blob("image", capability::blob::image_blob(&FAKE_IMG_HWC, 2, 2, 3)))
    }
}

/// An executor with the RUNNABLE [`FakeImage`] plus the RUNNABLE [`FakeChat`] (so
/// "non-image (chat) model -> 404" can be exercised on a real chat model).
fn image_executor() -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeImage), Arc::new(FakeChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

fn image_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(image_executor(), key.clone(), provider);
    (router(state), key)
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

/// A model that reports its REAL serving capacity (`Manifest::max_context_tokens`,
/// e.g. `QwenResident` advertising the engine's actual configured
/// capacity — see `resident_llm.rs`) must have that exact number reach `/models`,
/// so a client reading it can trust a prompt within it will actually be admitted.
/// A model that does NOT report it (the plain `chat_manifest()` used elsewhere in
/// this file) must fall back sanely: `context_length` omitted entirely on OpenAI
/// (matching the real API, which has no such field), the conservative default on
/// OpenRouter (which documents `context_length` as a real field).
#[tokio::test]
async fn model_card_advertises_real_capacity_when_known() {
    let capped_manifest = chat_manifest();
    assert_eq!(capped_manifest.max_context_tokens, None, "sanity: chat_manifest() itself reports nothing");
    let capped = capped_manifest.with_max_context_tokens(2048);
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Carded(capped)), Arc::new(Carded(embed_manifest()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);

    // OpenAI: the capped model carries context_length; the uncapped one omits it.
    let state = AppState::new(Executor::start(models, budgets, Policy::default()), key.clone(), Provider::OpenAI);
    let app = router(state);
    let (st, body) =
        send(&app, auth(Request::builder().uri("/v1/models/brain-chat"), Provider::OpenAI, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "Model", &body);
    assert_eq!(body["context_length"], 2048, "the real engine capacity must reach the OpenAI card: {body}");

    let (st, body) =
        send(&app, auth(Request::builder().uri("/v1/models/brain-embed"), Provider::OpenAI, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "Model", &body);
    assert!(body.get("context_length").is_none(), "an unreported capacity must be omitted, not guessed: {body}");

    // OpenRouter: the capped model carries the real number; the uncapped one gets
    // the conservative fallback, not a silently-wrong large default.
    let models: Vec<Arc<dyn ResidentModel>> =
        vec![Arc::new(Carded(chat_manifest().with_max_context_tokens(2048))), Arc::new(Carded(embed_manifest()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let state = AppState::new(Executor::start(models, budgets, Policy::default()), key.clone(), Provider::OpenRouter);
    let app = router(state);
    let (st, body) =
        send(&app, auth(Request::builder().uri("/models/brain-chat"), Provider::OpenRouter, &key).body(Body::empty()).unwrap())
            .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(body["context_length"], 2048);
    assert_eq!(body["top_provider"]["context_length"], 2048);

    let (st, body) =
        send(&app, auth(Request::builder().uri("/models/brain-embed"), Provider::OpenRouter, &key).body(Body::empty()).unwrap())
            .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(body["context_length"], 4096, "unreported capacity falls back to the conservative default");
}

/// An agent-grade chat model: `generate` declares `tools` and a hard `max_new` bound.
fn agent_chat_manifest() -> Manifest {
    Manifest::new(
        "brain-agent",
        "a tool-calling chat model",
        vec![ActionSpec::new("generate", "generate text")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("max_new", ParamType::Int, "new tokens").default(json!(128)).max(8192.0))
            .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions"))
            .output(BlobSpec::new("text", Media::Text, "generated text"))],
    )
}

/// An app over `models` for `provider`, plus the executor it serves from, so a test
/// can change what is attached to a model the way a hot swap does.
fn app_over(models: Vec<Manifest>, provider: Provider) -> (Router, Executor, String) {
    let models: Vec<Arc<dyn ResidentModel>> = models.into_iter().map(|m| Arc::new(Carded(m)) as Arc<dyn ResidentModel>).collect();
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let key = "sk-brain-test-key".to_string();
    (router(AppState::new(exec.clone(), key.clone(), provider)), exec, key)
}

async fn get_json(app: &Router, provider: Provider, uri: &str, key: &str) -> Value {
    let (st, body) = send(app, auth(Request::builder().uri(uri), provider, key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK, "{provider} GET {uri}");
    body
}

/// The card of an attached adapter names it by identity (`id`, `digest`) and tracks a
/// hot swap; with nothing attached the field is absent (an unknown is not a guess);
/// the card never carries a filesystem path. Both dialects stay schema-valid.
#[tokio::test]
async fn model_card_names_the_attached_adapter_and_tracks_a_swap() {
    for (provider, list_uri, one_uri, model_schema) in [
        (Provider::OpenAI, "/v1/models", "/v1/models/brain-agent", ("openai.json", "Model")),
        (Provider::OpenRouter, "/models", "/models/brain-agent", ("openrouter.json", "Model")),
    ] {
        let (app, exec, key) = app_over(vec![agent_chat_manifest(), embed_manifest()], provider);

        let card = get_json(&app, provider, one_uri, &key).await;
        assert_valid(model_schema.0, model_schema.1, &card);
        assert!(card.get("adapter").is_none(), "{provider}: no adapter attached, so none is advertised: {card}");

        let first = residency::AdapterIdentity { id: "acme/tuned".into(), digest: format!("sha256:{}", "a".repeat(64)) };
        exec.set_served_adapter("brain-agent", Some(first.clone()));
        let card = get_json(&app, provider, one_uri, &key).await;
        assert_valid(model_schema.0, model_schema.1, &card);
        assert_eq!(card["adapter"], json!({ "id": first.id, "digest": first.digest }), "{provider}: {card}");
        let listed = get_json(&app, provider, list_uri, &key).await;
        let entry = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-agent").unwrap();
        assert_eq!(entry["adapter"]["digest"], json!(first.digest), "{provider}: the list carries it too");
        let embed = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-embed").unwrap();
        assert!(embed.get("adapter").is_none(), "{provider}: an adapter is advertised on its own model only");

        let second = residency::AdapterIdentity { id: "acme/tuned".into(), digest: format!("sha256:{}", "b".repeat(64)) };
        exec.set_served_adapter("brain-agent", Some(second.clone()));
        let card = get_json(&app, provider, one_uri, &key).await;
        assert_eq!(card["adapter"]["digest"], json!(second.digest), "{provider}: a hot swap is visible on the next read");

        let text = card.to_string();
        assert!(!text.contains(".safetensors") && !text.contains("/tmp") && !text.contains("/home"), "{provider}: no path in the card: {text}");
        assert_eq!(card["adapter"].as_object().unwrap().len(), 2, "{provider}: the adapter object is exactly id + digest");

        exec.set_served_adapter("brain-agent", None);
        let card = get_json(&app, provider, one_uri, &key).await;
        assert!(card.get("adapter").is_none(), "{provider}: detaching removes the field: {card}");
    }
}

/// `capabilities` comes from the manifest's action shapes and `max_output_tokens` from
/// the `generate` action's declared `max_new` bound; neither is invented when the
/// manifest does not say.
#[tokio::test]
async fn model_card_derives_capabilities_and_output_limit_from_the_manifest() {
    for (provider, uri_prefix, model_schema) in [
        (Provider::OpenAI, "/v1/models", ("openai.json", "Model")),
        (Provider::OpenRouter, "/models", ("openrouter.json", "Model")),
    ] {
        let (app, _exec, key) = app_over(vec![agent_chat_manifest(), chat_manifest(), embed_manifest()], provider);

        let agent = get_json(&app, provider, &format!("{uri_prefix}/brain-agent"), &key).await;
        assert_valid(model_schema.0, model_schema.1, &agent);
        assert_eq!(agent["capabilities"], json!(["chat", "tools"]), "{provider}: {agent}");
        assert_eq!(agent["max_output_tokens"], 8192, "{provider}: {agent}");

        let plain = get_json(&app, provider, &format!("{uri_prefix}/brain-chat"), &key).await;
        assert_valid(model_schema.0, model_schema.1, &plain);
        assert_eq!(plain["capabilities"], json!(["chat"]), "{provider}: a chat model without tools does not claim them");
        assert!(plain.get("max_output_tokens").is_none(), "{provider}: an undeclared bound is absent, not 0: {plain}");

        let embed = get_json(&app, provider, &format!("{uri_prefix}/brain-embed"), &key).await;
        assert_eq!(embed["capabilities"], json!(["embeddings"]), "{provider}: {embed}");
    }
}

/// A single model card is as key-gated as the list: it now names what is attached.
#[tokio::test]
async fn model_card_route_requires_the_key() {
    for (p, uri) in [(Provider::OpenAI, "/v1/models/brain-chat"), (Provider::OpenRouter, "/models/brain-chat")] {
        let (app, key) = build_app(p);
        let (st, _) = send(&app, Request::builder().uri(uri).body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: {uri} without a key");
        let (st, _) = send(&app, auth(Request::builder().uri(uri), p, "wrong").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: {uri} with a wrong key");
        let (st, _) = send(&app, auth(Request::builder().uri(uri), p, &key).body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::OK, "{p}: {uri} with the key");
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
    let image = apiserve::api_caps(&image_manifest());
    assert!(image.image && !image.chat && !image.embeddings);
}

/// A FastVLM-shaped model: a streaming `caption` action taking `prompt` and
/// emitting `Text` — same shape as `chat_manifest` except the action name. The
/// chat handlers always dispatch the literal action `"generate"`
/// (`openai.rs`/`anthropic.rs`), so a model whose only prompt-shaped action is
/// named something else must NOT be advertised as chat: it would be listed by
/// `/v1/models` and then fail `ActionSpec::validate` on `messages`/`temp` (unknown
/// params) and the missing required `image` input this fake omits for brevity.
fn caption_manifest() -> Manifest {
    Manifest::new(
        "brain-caption",
        "an image captioning model",
        vec![ActionSpec::new("caption", "caption an image")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt"))
            .output(BlobSpec::new("text", Media::Text, "the caption"))],
    )
}

/// A CodeFormer/Real-ESRGAN/VQGAN-shaped model: an action emitting an `Image`
/// output but requiring a source image input and taking no `prompt` — image
/// *editing*, not text-to-image. `/images/generations` only ever dispatches a
/// pure text-to-image action ([`text2image_action`]'s contract), so this must
/// NOT be advertised as `image`: it would be listed and then 404 on that route.
fn image_edit_manifest() -> Manifest {
    Manifest::new(
        "brain-restore",
        "a face restoration model",
        vec![ActionSpec::new("restore_face", "restore a degraded face")
            .input(BlobSpec::new("image", Media::Image, "the degraded face").required())
            .output(BlobSpec::new("image", Media::Image, "the restored face"))],
    )
}

/// An ArcFace-shaped model: an action literally named `embed`, right name, but it
/// takes a required `image` input and NO `text` param — `/v1/embeddings`
/// (`openai.rs::handle_embeddings`) always dispatches `embed` with a `text` param
/// and never a blob, so this would 400 on the missing required input if advertised.
fn image_embed_manifest() -> Manifest {
    Manifest::new(
        "brain-arcface",
        "a face embedding model",
        vec![ActionSpec::new("embed", "512-d face identity embedding")
            .input(BlobSpec::new("image", Media::Image, "the face").required())
            .output(BlobSpec::new("embedding", Media::Bytes, "the embedding"))],
    )
}

/// A CLIP-shaped model: an action shaped like an embedder (a `Bytes` output whose
/// name contains "embed") but NOT literally named `embed` — `/v1/embeddings`
/// always dispatches the literal action `"embed"`, so this would fail with "no
/// action 'embed'" if advertised on the output-blob-name-alone rule the classifier
/// used to have.
fn misnamed_embed_manifest() -> Manifest {
    Manifest::new(
        "brain-clip",
        "a text embedding model with the wrong action name",
        vec![ActionSpec::new("embed_text", "embed a string")
            .param(ParamSpec::new("text", ParamType::Str, "input text").required())
            .output(BlobSpec::new("embedding", Media::Bytes, "the embedding"))],
    )
}

#[test]
fn api_caps_rejects_shape_matches_that_are_not_actually_dispatchable() {
    // Prompt-shaped + streaming + Text output, but the wrong action name: not chat.
    let caption = apiserve::api_caps(&caption_manifest());
    assert!(!caption.chat, "a non-\"generate\" action must not be advertised as chat");
    assert!(!caption.image && !caption.embeddings);

    // Emits an Image output, but it's an edit (required input, no `prompt`): not image.
    let edit = apiserve::api_caps(&image_edit_manifest());
    assert!(!edit.image, "an image-editing action (required input blob) must not be advertised as text-to-image");
    assert!(!edit.chat && !edit.embeddings);

    // Named "embed", but it wants an image, not text: not embeddings.
    let face = apiserve::api_caps(&image_embed_manifest());
    assert!(!face.embeddings, "an 'embed' action with no 'text' param must not be advertised as embeddings");
    assert!(!face.chat && !face.image);

    // Embedding-shaped output, but the action isn't named "embed": not embeddings.
    let clip = apiserve::api_caps(&misnamed_embed_manifest());
    assert!(!clip.embeddings, "an action not literally named 'embed' must not be advertised as embeddings, however embedding-shaped its output");
    assert!(!clip.chat && !clip.image);
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
    // OpenAI: malformed JSON, missing model, missing messages, n > 1, a
    // sampling parameter brain cannot honour, an out-of-range top_k,
    // temperature, top_p or completion budget.
    let (app, key) = chat_app(Provider::OpenAI);
    let msg = json!([{"role": "user", "content": "hi"}]);
    let cases: [Value; 11] = [
        json!({"messages": msg}),                                    // no model
        json!({"model": "brain-chat"}),                              // no messages
        json!({"model": "brain-chat", "messages": msg, "n": 2}),     // n > 1
        json!({"model": "brain-chat", "messages": msg, "presence_penalty": 3.0}),
        json!({"model": "brain-chat", "messages": msg, "top_logprobs": 5}),
        json!({"model": "brain-chat", "messages": msg, "return_token_ids": "yes"}),
        json!({"model": "brain-chat", "messages": msg, "top_k": 5000}),
        json!({"model": "brain-chat", "messages": msg, "temperature": 3.0}),
        json!({"model": "brain-chat", "messages": msg, "top_p": 1.5}),
        json!({"model": "brain-chat", "messages": msg, "max_tokens": 0}),
        json!({"model": "brain-chat", "messages": msg, "max_completion_tokens": -1}),
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
    let body = json!({"model": "brain-chat", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "seed": "x"});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "anthropic must 400 on a non-integer seed");
    assert_valid("anthropic.json", "ErrorResponse", &v);
    for bad in [json!({"max_tokens": 0}), json!({"max_tokens": 8, "temperature": -1.0}), json!({"max_tokens": 8, "top_p": 2.0})] {
        let mut body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut().unwrap().extend(bad.as_object().unwrap().clone());
        let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "anthropic must 400: {body}");
        assert_valid("anthropic.json", "ErrorResponse", &v);
    }
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

/// A provider that fails AFTER admission (`HTTP 200` + SSE headers already
/// committed) must surface a NAMED `error` SSE event carrying a generic,
/// non-leaking error body, then terminate with `[DONE]` — never an unnamed
/// `data:` frame a client has no signal to key off (see `StreamMsg::Err` in
/// `openai.rs`'s `render_chat_stream`). This is the streaming counterpart of
/// `runtime_error_is_not_reflected_to_client` (which only covers the
/// non-streaming path) and closes the gap noted there: nothing previously
/// exercised a mid-stream `StreamMsg::Err` for any provider.
#[tokio::test]
async fn openai_chat_stream_error_frame_is_named_and_generic() {
    let (app, key) = failing_chat_app_after(Provider::OpenAI, std::time::Duration::from_millis(600));
    let body = json!({
        "model": "brain-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    // The failure surfaces after the response has begun (the model fails 600 ms in, past
    // the moment a streamed request gets to fail with a plain status - see
    // `an_engine_refusal_is_an_immediate_http_error_streamed_or_not`) - the response is a
    // normal 200 + SSE body, not a 4xx.
    assert_eq!(st, StatusCode::OK);

    let events = sse_events(&text);
    let (ev, data) = events
        .iter()
        .find(|(_, d)| d != "[DONE]" && d.contains("\"error\""))
        .expect("an error frame must be present in the stream");
    assert_eq!(ev, "error", "the error frame must be a NAMED SSE event, not the default");

    // The error frame is intentionally NOT a CreateChatCompletionStreamResponse:
    // there is no finish_reason value in the real OpenAI schema that means
    // "error" (stop/length/tool_calls/content_filter/function_call only - see
    // tests/specs/openai.json), and real OpenAI sends the same bare
    // `{"error": ...}` shape for a mid-stream failure. Validate it as an
    // error body instead - the same shape the non-streaming path already
    // validates in `runtime_error_is_not_reflected_to_client`.
    let v: Value = serde_json::from_str(data).unwrap();
    assert_valid("openai.json", "ErrorResponse", &v);
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert_eq!(msg, "the model failed to process the request", "client gets the generic message");
    assert!(!msg.contains("secret"), "internal path must not leak to client: {msg}");
    assert!(!msg.contains("weights.gguf"), "internal path must not leak to client: {msg}");
    assert!(!msg.contains("panic"), "internal panic text must not leak to client: {msg}");

    let datas = sse_data(&text);
    assert_eq!(datas.last().map(String::as_str), Some("[DONE]"), "stream must still terminate with [DONE]");
}

/// A request that fails inside the model must not ALSO look like a successful
/// empty answer.
///
/// The role chunk (`{"role":"assistant","content":""}`) used to be emitted
/// unconditionally, before anything was known about the request. A failure
/// after admission - a lane panic, which a real 30B load reproduced by OOMing
/// a card mid-generation - therefore produced a stream containing a
/// syntactically valid, empty assistant message followed by `[DONE]`, and a
/// client that accumulates `choices` and stops at `[DONE]` reported
/// `success=true` with empty text and no tool calls. The error frame was
/// there; nothing forced a client to look at it.
///
/// So: a stream that produced no output before failing carries NO
/// `chat.completion.chunk` at all. There is no empty success left to mistake
/// the failure for, whether or not the client inspects SSE event names.
#[tokio::test]
async fn openai_chat_stream_failure_emits_no_empty_assistant_message() {
    let (app, key) = failing_chat_app_after(Provider::OpenAI, std::time::Duration::from_millis(600));
    let body = json!({
        "model": "brain-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);

    for d in sse_data(&text) {
        if d == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(&d).unwrap_or(Value::Null);
        assert_ne!(
            v["object"].as_str(),
            Some("chat.completion.chunk"),
            "a failed request must emit no assistant-message chunk at all, got: {d}"
        );
        assert!(v.get("error").is_some(), "the only payload of a failed stream is the error body, got: {d}");
    }
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

// ---------------------------------------------------------- P10: image generation

/// The 8-byte PNG signature.
const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// OpenAI non-stream: a spec-valid `ImagesResponse` whose `b64_json` base64-decodes
/// to the exact PNG the handler encodes from the model's raw image blob.
#[tokio::test]
async fn openai_images_nonstream_validates_and_decodes_to_png() {
    let (app, key) = image_app(Provider::OpenAI);
    let body = json!({"model": "brain-image", "prompt": "a red cat", "size": "1024x1024"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::OK, "images must 200: {v}");
    assert_valid("openai.json", "ImagesResponse", &v);
    assert!(v["created"].is_i64(), "created must be a unix timestamp: {v}");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "n defaults to 1 image");
    let b64 = data[0]["b64_json"].as_str().expect("b64_json must be a string");
    let bytes = events::base64::decode(b64).expect("b64_json must be valid base64");
    assert_eq!(&bytes[0..8], &PNG_SIG, "b64_json must decode to PNG bytes");
    assert_eq!(bytes, apiserve::png::encode_rgb8(&FAKE_IMG_RGB8, 2, 2), "b64_json must be the PNG of the model's image");
}

/// `n` produces one image per requested count; `response_format: "url"` is accepted
/// but still answered with `b64_json` (brain has no object store).
#[tokio::test]
async fn openai_images_n_and_url_format_return_b64() {
    let (app, key) = image_app(Provider::OpenAI);
    let body = json!({"model": "brain-image", "prompt": "cats", "n": 3, "response_format": "url"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::OK, "n=3 + url format must 200: {v}");
    assert_valid("openai.json", "ImagesResponse", &v);
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 3, "n=3 must return 3 images");
    for d in data {
        assert!(d["b64_json"].is_string(), "url format still returns b64_json: {d}");
        assert!(d.get("url").is_none(), "brain never returns a url");
    }
}

/// `precision` is a non-standard, optional field (mirroring `seed`) forwarded
/// to the resident model as an Invocation param (`s3dit::caps` reads it to
/// pick int8 vs. fp32 DiT precision) -- both accepted enum values must 200,
/// and omitting it entirely must still work exactly as before this field
/// existed (already covered by `openai_images_nonstream_validates_and_decodes_to_png`).
#[tokio::test]
async fn openai_images_precision_int8_and_fp32_are_accepted() {
    let (app, key) = image_app(Provider::OpenAI);
    for precision in ["int8", "fp32"] {
        let body = json!({"model": "brain-image", "prompt": "a red cat", "precision": precision});
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
        assert_eq!(st, StatusCode::OK, "precision={precision} must 200: {v}");
        assert_valid("openai.json", "ImagesResponse", &v);
    }
}

/// Unknown model -> 404; an existing non-image (chat) model -> 404.
#[tokio::test]
async fn openai_images_unknown_and_non_image_models_are_404() {
    let (app, key) = image_app(Provider::OpenAI);
    let body = json!({"model": "nope", "prompt": "hi"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "unknown model must 404");
    assert_valid("openai.json", "ErrorResponse", &v);

    let body = json!({"model": "brain-chat", "prompt": "hi"});
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "a non-image (chat) model must 404");
}

/// Every whitelisted size a served model generates at is accepted, 384x384
/// (Janus-Pro's only size) included.
#[tokio::test]
async fn openai_images_accept_the_whitelisted_small_sizes() {
    let (app, key) = image_app(Provider::OpenAI);
    for size in ["256x256", "384x384", "512x512"] {
        let body = json!({"model": "brain-image", "prompt": "a red cat", "size": size});
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
        assert_eq!(st, StatusCode::OK, "{size} must 200: {v}");
        assert_valid("openai.json", "ImagesResponse", &v);
    }
}

/// A whitelisted size the model itself does not draw (its text2image action
/// bounds `width`/`height`) is refused up front, naming the bound, rather
/// than queued behind the model's load and refused there.
#[tokio::test]
async fn openai_images_refuse_a_size_outside_the_models_own_bounds() {
    let (app, key) = image_app(Provider::OpenAI);
    let body = json!({"model": "brain-image", "prompt": "a tall cat", "size": "1024x1536"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "must 400: {v}");
    assert_valid("openai.json", "ErrorResponse", &v);
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("height") && msg.contains("1024"), "{msg}");
}

/// Bad bodies -> 400: missing model, missing prompt, unsupported size, n out of
/// range, a bad response_format, a bad precision, and malformed JSON.
#[tokio::test]
async fn openai_images_bad_bodies_are_400() {
    let (app, key) = image_app(Provider::OpenAI);
    let cases: [Value; 7] = [
        json!({"prompt": "hi"}),                                                  // no model
        json!({"model": "brain-image"}),                                          // no prompt
        json!({"model": "brain-image", "prompt": "hi", "size": "3x3"}),           // unsupported size
        json!({"model": "brain-image", "prompt": "hi", "n": 0}),                  // n < 1
        json!({"model": "brain-image", "prompt": "hi", "n": 11}),                 // n > 10
        json!({"model": "brain-image", "prompt": "hi", "response_format": "xyz"}), // bad format
        json!({"model": "brain-image", "prompt": "hi", "precision": "fp16"}),     // bad precision
    ];
    for body in cases {
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "must 400: {body}");
        assert_valid("openai.json", "ErrorResponse", &v);
    }
    let req = auth(Request::builder().method(Method::POST).uri("/v1/images/generations"), Provider::OpenAI, &key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{ not json"))
        .unwrap();
    let (st, _) = send(&app, req).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "malformed JSON must 400");
}

/// OpenRouter `/images/generations` works the same (shared handler): a spec-valid
/// image response, and an unknown model still 404s in the OpenRouter error shape.
#[tokio::test]
async fn openrouter_images_work_the_same() {
    let (app, key) = image_app(Provider::OpenRouter);
    let body = json!({"model": "brain-image", "prompt": "a dog"});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/images/generations", &body).await;
    assert_eq!(st, StatusCode::OK, "openrouter images must 200: {v}");
    assert_valid("openai.json", "ImagesResponse", &v);
    assert_eq!(v["data"].as_array().unwrap().len(), 1);

    let body = json!({"model": "nope", "prompt": "hi"});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/images/generations", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_valid("openrouter.json", "InternalServerResponse", &v);
}

/// `stream: true` maps each denoise-step progress tick to a spec-valid
/// `image_generation.partial_image` event (indexed, no pixels) and the final image to
/// a spec-valid `image_generation.completed` event whose `b64_json` decodes to the PNG.
#[tokio::test]
async fn openai_images_stream_emits_partials_and_final_image() {
    let (app, key) = image_app(Provider::OpenAI);
    let body = json!({"model": "brain-image", "prompt": "a cat", "size": "1024x1024", "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::OK);
    let frames: Vec<Value> = sse_data(&text).iter().map(|d| serde_json::from_str(d).unwrap()).collect();

    // One partial_image event per denoise step (the fake emits two), each indexed and
    // schema-valid.
    let partials: Vec<&Value> = frames.iter().filter(|e| e["type"] == "image_generation.partial_image").collect();
    assert!(partials.len() >= 2, "a partial_image event per denoise step: {frames:?}");
    for (i, p) in partials.iter().enumerate() {
        assert_valid("openai.json", "ImageGenPartialImageEvent", p);
        assert_eq!(p["partial_image_index"], i as i64, "partial indices count up");
    }

    // Exactly one terminal completed event carrying the real PNG.
    let completed: Vec<&Value> = frames.iter().filter(|e| e["type"] == "image_generation.completed").collect();
    assert_eq!(completed.len(), 1, "one terminal completed event: {frames:?}");
    assert_valid("openai.json", "ImageGenCompletedEvent", completed[0]);
    let b64 = completed[0]["b64_json"].as_str().unwrap();
    let bytes = events::base64::decode(b64).unwrap();
    assert_eq!(&bytes[0..8], &PNG_SIG);
    assert_eq!(bytes, apiserve::png::encode_rgb8(&FAKE_IMG_RGB8, 2, 2), "completed event carries the model's PNG");
}

// ------------------------------------------- P11: OpenRouter model-id resolution

/// OpenRouter strips a leading `"<provider>/"` namespace: a request for
/// `anything/brain-chat` resolves to the local `brain-chat` and returns a spec-valid
/// 200 (whereas the exact id is not in the catalog).
#[tokio::test]
async fn openrouter_chat_strips_provider_prefix() {
    let (app, key) = chat_app(Provider::OpenRouter);
    let body = json!({"model": "anything/brain-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "prefixed model must resolve + 200: {v}");
    assert_valid("openrouter.json", "ChatResult", &v);
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
}

/// OpenRouter honours the `models` fallback array: the primary `model` and the first
/// fallback entry don't resolve, but the second (`prefix/brain-chat`) does after prefix
/// stripping — so the request succeeds via that entry.
#[tokio::test]
async fn openrouter_chat_models_fallback_array_resolves_second() {
    let (app, key) = chat_app(Provider::OpenRouter);
    let body = json!({
        "model": "does/not-exist",
        "models": ["nope/x", "prefix/brain-chat"],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "second fallback entry must resolve + 200: {v}");
    assert_valid("openrouter.json", "ChatResult", &v);
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
}

/// OpenRouter tolerates its own extra request fields (`provider`, `route`,
/// `transforms`, `plugins`): they parse and are ignored, so the request still 200s
/// instead of 400-ing on the unknown keys.
#[tokio::test]
async fn openrouter_chat_tolerates_or_only_fields() {
    let (app, key) = chat_app(Provider::OpenRouter);
    let body = json!({
        "model": "brain-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "provider": {"order": ["brain"], "allow_fallbacks": false},
        "route": "fallback",
        "transforms": ["middle-out"],
        "plugins": [{"id": "web"}],
    });
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "OpenRouter-only extras must not 400: {v}");
    assert_valid("openrouter.json", "ChatResult", &v);
}

/// The prefix-strip + fallback behaviour is OpenRouter-only: on the OpenAI surface a
/// slashed model id is NOT stripped and a `models` array is ignored, so a request for a
/// prefixed id that only matches after stripping still 404s.
#[tokio::test]
async fn openai_chat_does_not_strip_prefix_or_use_models_fallback() {
    let (app, key) = chat_app(Provider::OpenAI);
    // Slashed id: OpenAI must not strip -> 404.
    let body = json!({"model": "anything/brain-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "OpenAI must not strip a slashed model: {v}");
    assert_valid("openai.json", "ErrorResponse", &v);

    // `models` fallback array is an OpenRouter-only field: ignored on OpenAI, so an
    // unresolvable primary `model` still 404s even if a `models` entry would resolve.
    let body = json!({
        "model": "does/not-exist",
        "models": ["brain-chat"],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "OpenAI must ignore the models fallback array");
}

/// The prefix strip + `models` fallback apply to OpenRouter embeddings and images too
/// (shared resolution): a prefixed embeddings id resolves, and an images `models`
/// fallback entry resolves after stripping.
#[tokio::test]
async fn openrouter_embeddings_and_images_resolve_prefix_and_fallback() {
    // Embeddings: prefixed id strips to the local embed model.
    let (app, key) = embed_app(Provider::OpenRouter);
    let body = json!({"model": "any/brain-embed", "input": "hi"});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/embeddings", &body).await;
    assert_eq!(st, StatusCode::OK, "prefixed embeddings model must resolve: {v}");
    assert_valid("openai.json", "CreateEmbeddingResponse", &v);

    // Images: primary doesn't resolve, fallback entry strips to the local image model.
    let (app, key) = image_app(Provider::OpenRouter);
    let body = json!({"model": "nope/x", "models": ["vendor/brain-image"], "prompt": "a dog"});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/images/generations", &body).await;
    assert_eq!(st, StatusCode::OK, "images fallback entry must resolve: {v}");
    assert_valid("openai.json", "ImagesResponse", &v);
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
    let message = v["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("--admit-deadline-ms") && message.contains("200"), "the 429 tells the client what to change: {message:?}");

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

/// A chat model that works until its request is cancelled (bounded, so a bug cannot hang the suite) and reports whether it was.
struct UntilCancelled {
    started: Arc<AtomicUsize>,
    cancelled: Arc<AtomicUsize>,
}
struct UntilCancelledInst(Arc<AtomicUsize>, Arc<AtomicUsize>);
impl ResidentModel for UntilCancelled {
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
        Ok(Box::new(UntilCancelledInst(self.started.clone(), self.cancelled.clone())))
    }
}
impl Instance for UntilCancelledInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        let until = Instant::now() + Duration::from_secs(5);
        while Instant::now() < until {
            if inv.cancel.is_cancelled() {
                self.1.fetch_add(1, Ordering::SeqCst);
                return Err("cancelled".into());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(Outcome::new().set("prompt_tokens", json!(1)).set("completion_tokens", json!(1)).set("finish_reason", json!("stop")).blob("text", Blob::new(Media::Text, b"done".to_vec())))
    }
}

/// **A client that hangs up on a one-shot request cancels its generation.** A non-streaming request is one handler future; when the
/// client disconnects the server drops it, and the generation it started must stop (and give its KV back) instead of running to
/// its token limit for nobody - a timed-out client otherwise leaves the work behind on the card.
#[tokio::test]
async fn dropping_a_one_shot_request_cancels_its_generation() {
    let (started, cancelled) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(UntilCancelled { started: started.clone(), cancelled: cancelled.clone() })];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let key = "sk-brain-test-key".to_string();
    let app = router(AppState::new(Executor::start(models, budgets, Policy::default()), key.clone(), Provider::OpenAI));
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});

    // The client gives up after 300 ms: the future is dropped, as the server does when the connection closes.
    let request = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body);
    assert!(tokio::time::timeout(Duration::from_millis(300), request).await.is_err(), "the model works until cancelled, so the call does not finish");
    assert_eq!(started.load(Ordering::SeqCst), 1, "the generation had started");
    let until = Instant::now() + Duration::from_secs(2);
    while cancelled.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < until, "the generation kept running after its client hung up");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A chat model whose `run_batch` pays its (mocked) generation cost ONCE per
/// batch, not once per invocation — a stand-in for "one shared forward pass
/// serves every sequence in it", exactly what `qwen3::serve::Scheduler` does for
/// real — measured with a real model, TTFA at concurrency 2 came in LOWER
/// than at concurrency 1 through this same router.
/// The framework always calls `run_batch` (never `run` directly — see
/// `residency::executor::run_group`), so `run` here just routes through it.
struct BatchingChat {
    ms: u64,
    max_batch_seen: Arc<AtomicUsize>,
}
struct BatchingChatInst {
    ms: u64,
    max_batch_seen: Arc<AtomicUsize>,
}
impl ResidentModel for BatchingChat {
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
        Ok(Box::new(BatchingChatInst { ms: self.ms, max_batch_seen: self.max_batch_seen.clone() }))
    }
}
impl Instance for BatchingChatInst {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_, p| progress(p)).pop().expect("one result for one invocation")
    }
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.max_batch_seen.fetch_max(invs.len(), Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(self.ms)); // ONE sleep for the whole batch
        invs.iter()
            .map(|_| {
                Ok(Outcome::new()
                    .set("prompt_tokens", json!(1))
                    .set("completion_tokens", json!(1))
                    .set("finish_reason", json!("stop"))
                    .blob("text", Blob::new(Media::Text, b"done".to_vec())))
            })
            .collect()
    }
}

fn batching_executor(ms: u64, max_batch_seen: Arc<AtomicUsize>) -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(BatchingChat { ms, max_batch_seen })];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

/// REGRESSION for the concurrent-request-batching investigation's core claim,
/// at the layer that had NEVER been exercised before this workstream: the real HTTP
/// router. `residency::executor`'s own `same_model_batches_and_evicts` already
/// proves the dispatcher groups same-key queued jobs into one claim
/// (`Stats::max_batch`); this proves that grouping is actually REACHABLE by N
/// concurrent `/v1/chat/completions` requests arriving over HTTP, and that it
/// actually saves wall time (not just an internal counter) — the audit's
/// finding was precisely that nothing upstream of the model ever produced more
/// than one queued invocation at a time in practice.
#[tokio::test]
async fn n_concurrent_chat_requests_batch_through_the_real_router_not_serialize() {
    const N: usize = 4;
    const BATCH_MS: u64 = 300;
    let max_batch_seen = Arc::new(AtomicUsize::new(0));
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(batching_executor(BATCH_MS, max_batch_seen.clone()), key.clone(), Provider::OpenAI);
    let app = router(state);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});

    let t0 = Instant::now();
    let mut jobs = Vec::with_capacity(N);
    for _ in 0..N {
        let (app, key, body) = (app.clone(), key.clone(), body.clone());
        jobs.push(tokio::spawn(async move { post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await }));
    }
    for j in jobs {
        let (st, v) = j.await.unwrap();
        assert_eq!(st, StatusCode::OK, "{v}");
    }
    let elapsed = t0.elapsed();

    assert!(
        max_batch_seen.load(Ordering::SeqCst) >= 2,
        "{N} concurrent requests to the same resident must batch into one dispatcher claim, not admit \
         one invocation at a time; saw max_batch={}",
        max_batch_seen.load(Ordering::SeqCst)
    );
    // Batched: ~1x BATCH_MS regardless of N (fully grouped) or a small multiple of it
    // (grouped across a couple of dispatcher rounds, if the requests didn't all land
    // in the queue before the first claim). Serialized (the pre-fix behavior): N x
    // BATCH_MS. The bound below is a fraction of fully-serial and deliberately
    // generous - this test asserts "batching saves real wall time", not a
    // specific batch-count-per-round -
    // so it stays non-flaky under scheduler jitter while still failing hard against a
    // regression back to one-invocation-at-a-time admission.
    let serial = BATCH_MS * N as u64;
    assert!(
        elapsed < Duration::from_millis(serial * 3 / 4),
        "{N} concurrent requests took {elapsed:?}, expected well under the fully-serial {serial}ms \
         (batching should make this well under {}ms)",
        serial * 3 / 4
    );
}

/// An image model whose `run_batch` draws every image of a batch together and
/// records the largest batch it was handed.
struct BatchingImage {
    max_batch_seen: Arc<AtomicUsize>,
}
struct BatchingImageInst {
    max_batch_seen: Arc<AtomicUsize>,
}
impl ResidentModel for BatchingImage {
    fn manifest(&self) -> Manifest {
        image_manifest()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-image", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(BatchingImageInst { max_batch_seen: self.max_batch_seen.clone() }))
    }
}
impl Instance for BatchingImageInst {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_, p| progress(p)).pop().expect("one result for one invocation")
    }
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.max_batch_seen.fetch_max(invs.len(), Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(200)); // the batch's one pass
        invs.iter().map(|_| Ok(Outcome::new().blob("image", capability::blob::image_blob(&[0.0; 12], 2, 2, 3)))).collect()
    }
}

/// The `n` images of one request are submitted together, so a model that can
/// draw several at once is handed them as one batch instead of one at a time.
#[tokio::test]
async fn the_images_of_one_request_reach_the_model_as_one_batch() {
    let max_batch_seen = Arc::new(AtomicUsize::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(BatchingImage { max_batch_seen: max_batch_seen.clone() })];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let key = "sk-brain-test-key".to_string();
    let app = router(AppState::new(Executor::start(models, budgets, Policy::default()), key.clone(), Provider::OpenAI));
    let body = json!({"model": "brain-image", "prompt": "a red cat", "n": 3, "size": "1024x1024"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/images/generations", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["data"].as_array().map(Vec::len), Some(3));
    assert!(max_batch_seen.load(Ordering::SeqCst) >= 2, "3 images of one request must be dispatched together; the largest batch was {}", max_batch_seen.load(Ordering::SeqCst));
}

// ------------------------------------------------------------ security (P17)

/// A chat model whose `generate` always fails with an internal error string that
/// embeds a filesystem path — the kind of detail that must NEVER reach a client.
struct Failing(std::time::Duration);
struct FailingInst(std::time::Duration);
impl ResidentModel for Failing {
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
        Ok(Box::new(FailingInst(self.0)))
    }
}
impl Instance for FailingInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        std::thread::sleep(self.0);
        Err("backend panic at /home/secret/models/weights.gguf: kernel exploded".into())
    }
}

/// The failing chat model, failing at once.
fn failing_chat_app(provider: Provider) -> (Router, String) {
    failing_chat_app_after(provider, std::time::Duration::ZERO)
}

/// The failing chat model, failing `after` its request started. A streamed request that fails
/// within the first moments is answered with the HTTP status of the failure; one that fails
/// later - after the response has begun - surfaces it as an `error` event, so the tests of that
/// event fail the model slowly.
fn failing_chat_app_after(provider: Provider, after: std::time::Duration) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Failing(after))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let state = AppState::new(Executor::start(models, budgets, Policy::default()), key.clone(), provider);
    (router(state), key)
}

/// A chat model whose `generate` is refused by the engine at once, with the engine's own
/// wording (what `llmserve`'s scheduler glue returns for a request it cannot serve).
struct Refusing(&'static str);
struct RefusingInst(&'static str);
impl ResidentModel for Refusing {
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
        Ok(Box::new(RefusingInst(self.0)))
    }
}
impl Instance for RefusingInst {
    fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        Err(self.0.into())
    }
}

/// A request the engine refuses at admission is answered with an HTTP error status and an
/// OpenAI-style error body, immediately, streamed or not - never a 200 event stream whose
/// only event is the error (a client that waits for a `[DONE]` that never comes hangs on it)
/// and never silence. The over-context refusal is `context_length_exceeded`; the engine's
/// other refusals (a token outside the vocabulary, an adapter it does not hold, a prompt
/// the chat template cannot render) are 4xx errors that leak nothing.
#[tokio::test]
async fn an_engine_refusal_is_an_immediate_http_error_streamed_or_not() {
    let cases: [(&str, &str); 4] = [
        ("qwen35: needs 33805 tokens, engine capacity is 32768", "context_length_exceeded"),
        ("qwen35: token 70000 is outside the vocabulary (65000)", "invalid_request"),
        ("qwen35: adapter 3 is not available", "invalid_request"),
        ("qwen35: the chat template cannot render this conversation", "invalid_request"),
    ];
    for (refusal, code) in cases {
        for stream in [false, true] {
            let key = "sk-brain-test-key".to_string();
            let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Refusing(refusal))];
            let mut budgets = Budgets::new();
            budgets.set(Device::Cpu, 8 << 30, 0);
            let state = AppState::new(Executor::start(models, budgets, Policy::default()), key.clone(), Provider::OpenAI);
            let app = router(state);
            let body = json!({"model": "brain-chat", "stream": stream, "messages": [{"role": "user", "content": "hi"}]});
            let answered = tokio::time::timeout(std::time::Duration::from_secs(5), post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body)).await;
            let (st, v) = answered.unwrap_or_else(|_| panic!("stream={stream}: {refusal:?} was not answered within 5 s"));
            assert_eq!(st, StatusCode::BAD_REQUEST, "stream={stream}: {refusal:?} must be an HTTP 400, got {st}: {v}");
            assert_valid("openai.json", "ErrorResponse", &v);
            assert_eq!(v["error"]["code"], code, "stream={stream}: {refusal:?}: {v}");
            let msg = v["error"]["message"].as_str().unwrap_or("");
            assert!(!msg.contains("qwen35:"), "the engine's internal prefix does not reach the client: {msg}");
        }
    }
}

/// An UNAUTHENTICATED request to an unknown path must be 401 (not 404): the key layer
/// wraps the fallback, so a caller with no key cannot even enumerate which routes
/// exist. (Regression for "auth covers every route incl. fallback".)
#[tokio::test]
async fn unauthenticated_unknown_path_is_401_not_404() {
    for p in ALL {
        let (app, _key) = build_app(p);
        // No auth header at all, unknown path.
        let (st, _) = send(&app, Request::builder().uri("/no/such/route").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: unauth unknown path must 401, not reveal 404");
        // A known route, still unauthenticated, is also 401 — no route is outside auth.
        let (st, _) = send(&app, Request::builder().method(Method::POST).uri("/v1/chat/completions").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: unauth known route must 401");
    }
}

/// A wrong key with the SAME byte length as the real key is rejected, and the correct
/// key is accepted. Exercises the constant-time comparison on the equal-length path
/// (a plain `==` and a constant-time compare must agree on the RESULT).
#[tokio::test]
async fn wrong_key_of_equal_length_is_401_correct_is_200() {
    for p in ALL {
        let (app, key) = build_app(p);
        let mut wrong: Vec<u8> = key.clone().into_bytes();
        *wrong.last_mut().unwrap() ^= 0x01; // flip one byte; length unchanged
        let wrong = String::from_utf8(wrong).unwrap();
        assert_eq!(wrong.len(), key.len(), "constructed wrong key must match length");

        let (st, _) = send(&app, auth(Request::builder().uri("/models"), p, &wrong).body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{p}: equal-length wrong key must 401");
        let (st, _) = send(&app, auth(Request::builder().uri("/models"), p, &key).body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::OK, "{p}: correct key must 200");
    }
}

/// A request body larger than `MAX_BODY_BYTES` is rejected (413) before any handler
/// buffers it — the explicit body-size ceiling that prevents an OOM.
#[tokio::test]
async fn oversized_request_body_is_413() {
    let (app, key) = chat_app(Provider::OpenAI);
    let huge = "a".repeat(apiserve::MAX_BODY_BYTES + 1024);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": huge}]});
    let req = auth(Request::builder().method(Method::POST).uri("/v1/chat/completions"), Provider::OpenAI, &key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE, "an over-limit body must 413");
}

/// An internal runtime/activation error string from the executor must NOT be
/// reflected into the client-facing error body (no paths / panic text). The client
/// gets a generic message; the detail is only logged server-side.
#[tokio::test]
async fn runtime_error_is_not_reflected_to_client() {
    let (app, key) = failing_chat_app(Provider::OpenAI);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "a runtime failure surfaces as a 4xx: {v}");
    assert_valid("openai.json", "ErrorResponse", &v);
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(!msg.contains("secret"), "internal path must not leak to client: {msg}");
    assert!(!msg.contains("weights.gguf"), "internal path must not leak to client: {msg}");
    assert!(!msg.contains("panic"), "internal panic text must not leak to client: {msg}");
    assert_eq!(msg, "the model failed to process the request", "client gets the generic message");
}

/// `--api-keys-out` writes the key file with owner-only (0600) permissions — never
/// world- or group-readable.
#[cfg(unix)]
#[tokio::test]
async fn write_keys_file_is_owner_only_0600() {
    use std::os::unix::fs::PermissionsExt;
    let addr = "127.0.0.1:0".parse().unwrap();
    let surfaces = vec![apiserve::Surface::new(Provider::OpenAI, addr, "sk-brain-deadbeef")];
    static N: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!("brain-keys-{}-{}.json", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    // Pre-create the file world-readable to prove write_keys RE-tightens it.
    std::fs::write(&path, "{}").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

    apiserve::write_keys(&surfaces, &path).unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    let _ = std::fs::remove_file(&path);
    assert_eq!(mode, 0o600, "keys file must be owner-only 0600, got {mode:o}");
}

// ------------------------------------------------------- auto-fetch wiring

/// A stub [`residency::ModelSupplier`] that exercises the HTTP-layer wiring
/// (`bridge::ensure_and_recheck`) without a real network/model-store dependency:
/// `"vendor/fetchable-chat"` classifies as fetchable and "fetching" it registers
/// a runnable chat resident under that exact name; every other model is
/// `Unknown`. Counts `ensure` calls so a test can assert a classify-only
/// (`Unknown`) outcome never triggers one — the zero-network-I/O invariant
/// required for a name the store would refuse.
struct StubSupplier {
    ensure_calls: AtomicUsize,
}
impl residency::ModelSupplier for StubSupplier {
    fn classify(&self, model: &str) -> residency::Supply {
        if model == "vendor/fetchable-chat" {
            residency::Supply::Fetchable
        } else {
            residency::Supply::Unknown(format!("{model}: not in the stub catalog"))
        }
    }
    fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        self.ensure_calls.fetch_add(1, Ordering::SeqCst);
        progress("model.safetensors", 512, 1503);
        exec.register(Arc::new(FetchedChat(model.to_string())));
        Ok(())
    }
}

/// Like [`FakeChat`] but its manifest name is chosen at construction — what a
/// real `StoreSupplier::ensure` produces (a resident carded under the
/// fully-qualified `vendor/repo` the client asked for, not a fixed built-in name).
struct FetchedChat(String);
impl ResidentModel for FetchedChat {
    fn manifest(&self) -> Manifest {
        let mut m = chat_manifest();
        m.model = self.0.clone();
        m
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.0.clone(), "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeChatInst))
    }
}

/// An unresolved model that the supplier classifies `Fetchable` is fetched (via
/// `ensure`) and the ORIGINAL request then succeeds against the newly-registered
/// resident — the end-to-end "ask for a name that isn't resident yet, it just
/// works" path `bridge::ensure_and_recheck` exists for.
#[tokio::test]
async fn unresolved_chat_model_auto_fetches_via_the_supplier_then_succeeds() {
    let key = "sk-brain-test-key".to_string();
    let supplier = Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
    let state = AppState::new(chat_executor(), key.clone(), Provider::OpenAI).with_supplier(Some(supplier.clone() as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "vendor/fetchable-chat", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "auto-fetched model must serve the request: {v:?}");
    assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 1);
}

/// The STREAMING sibling of the test above: the SSE body opens immediately (no
/// blocking wait before the response starts), fetch progress from the
/// supplier's `progress` callback arrives as SSE COMMENT lines (`: BRAIN …`,
/// legal per the SSE spec, invisible to `sse_data`/`sse_events` and to every
/// conformant client's `data:`/`event:` parsing), and the real
/// `chat.completion.chunk` stream still follows once the model is ready.
#[tokio::test]
async fn unresolved_chat_model_streams_fetch_progress_as_sse_comments_then_the_real_stream() {
    let key = "sk-brain-test-key".to_string();
    let supplier = Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
    let state = AppState::new(chat_executor(), key.clone(), Provider::OpenAI).with_supplier(Some(supplier.clone() as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "vendor/fetchable-chat", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 1);

    // A comment line for the fetch tick, carrying the model name and the
    // percentage computed from the stub's `progress(512, 1503)` callback.
    assert!(text.lines().any(|l| l.starts_with(": BRAIN") && l.contains("vendor/fetchable-chat") && l.contains("34%")), "no fetch-progress comment line in:\n{text}");

    // The comment line(s) are invisible to standard SSE data/event parsing --
    // the real chat stream underneath is completely unaffected.
    let chunks = sse_data(&text);
    assert!(chunks.iter().any(|d| d.contains("\"role\":\"assistant\"") || d.contains("\"role\": \"assistant\"")), "missing role chunk: {chunks:?}");
    assert_eq!(chunks.last().unwrap(), "[DONE]");
}

/// A malicious/malformed HF filename in the fetch-progress callback (the
/// `name` argument comes from the store's `Step::Download { dest_name, .. }`,
/// itself derived from whatever file names the remote repo's owner chose --
/// attacker-influenced, not attacker-controlled by the requesting client, but
/// still untrusted) must not crash the SSE renderer: `axum::sse::Event::
/// comment` PANICS on an embedded newline/CR, so `stream_with_autofetch`
/// strips them before building the comment line. Regression for exactly that.
#[tokio::test]
async fn a_newline_in_the_fetch_progress_name_does_not_panic_the_sse_stream() {
    struct NewlineSupplier;
    impl residency::ModelSupplier for NewlineSupplier {
        fn classify(&self, model: &str) -> residency::Supply {
            if model == "vendor/fetchable-chat" { residency::Supply::Fetchable } else { residency::Supply::Unknown("n/a".into()) }
        }
        fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
            progress("evil\r\nfilename\nhere.safetensors", 1, 2);
            exec.register(Arc::new(FetchedChat(model.to_string())));
            Ok(())
        }
    }
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(chat_executor(), key.clone(), Provider::OpenAI).with_supplier(Some(Arc::new(NewlineSupplier) as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "vendor/fetchable-chat", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "must not panic/500 on a newline-carrying fetch-progress name");
    assert_eq!(sse_data(&text).last().unwrap(), "[DONE]", "stream must still complete normally: {text}");
}

/// The STREAMING sibling of `unknown_model_with_a_supplier_present_still_404s_
/// with_no_fetch_attempt`: an `Unknown` model must stay a plain, non-SSE 404
/// body -- never an event-stream that opens and then immediately errors --
/// and `ensure` is never called.
#[tokio::test]
async fn unresolved_chat_model_streaming_request_for_an_unknown_model_is_a_plain_404() {
    let key = "sk-brain-test-key".to_string();
    let supplier = Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
    let state = AppState::new(chat_executor(), key.clone(), Provider::OpenAI).with_supplier(Some(supplier.clone() as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "brain/reserved-or-nonsense", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert!(!text.starts_with("data:") && !text.starts_with(":"), "must not open an SSE body: {text}");
    assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 0);
}

/// Same streaming-auto-fetch mechanism as the OpenAI test above, on the
/// Anthropic dialect: SSE comment lines for fetch progress, then the normal
/// `message_start..message_stop` event sequence.
#[tokio::test]
async fn anthropic_unresolved_chat_model_streams_fetch_progress_then_the_real_stream() {
    let key = "sk-brain-test-key".to_string();
    let supplier = Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
    let state = AppState::new(chat_executor(), key.clone(), Provider::Anthropic).with_supplier(Some(supplier.clone() as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "vendor/fetchable-chat", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 1);
    assert!(text.lines().any(|l| l.starts_with(": BRAIN") && l.contains("34%")), "no fetch-progress comment line in:\n{text}");

    let events = sse_events(&text);
    assert!(events.iter().any(|(ev, _)| ev == "message_start"));
    assert!(events.iter().any(|(ev, _)| ev == "message_stop"));
}

/// A model the supplier classifies `Unknown` (the shape a reserved-vendor or
/// malformed ref gets) still 404s -- and, critically, `ensure` is NEVER called,
/// proving `classify` alone gates any network/filesystem I/O.
#[tokio::test]
async fn unknown_model_with_a_supplier_present_still_404s_with_no_fetch_attempt() {
    let key = "sk-brain-test-key".to_string();
    let supplier = Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
    let state = AppState::new(chat_executor(), key.clone(), Provider::OpenAI).with_supplier(Some(supplier.clone() as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "brain/reserved-or-nonsense", "messages": [{"role": "user", "content": "hi"}]});
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 0, "classify-only Unknown must never call ensure");
}

/// A supplier whose `ensure` fails must still just 404 -- the raw failure reason
/// (which could carry a hub URL or a filesystem path) is never reflected into the
/// response body.
#[tokio::test]
async fn a_failed_fetch_404s_without_leaking_the_internal_error_reason() {
    struct AlwaysFails;
    impl residency::ModelSupplier for AlwaysFails {
        fn classify(&self, _model: &str) -> residency::Supply {
            residency::Supply::Fetchable
        }
        fn ensure(&self, _model: &str, _exec: &Executor, _progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
            Err("hub error: /abs/workspace/secret-internal-path unreachable".to_string())
        }
    }
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(chat_executor(), key.clone(), Provider::OpenAI).with_supplier(Some(Arc::new(AlwaysFails) as Arc<dyn residency::ModelSupplier>));
    let app = router(state);

    let body = json!({"model": "vendor/will-fail", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let body_text = v.to_string();
    assert!(!body_text.contains("secret-internal-path"), "internal fetch error leaked into the response: {body_text}");
}

/// With NO supplier configured (every test above this one, and every
/// `AppState::new` that doesn't call `.with_supplier`), an unresolved model is a
/// plain 404 -- the pre-auto-fetch behavior, unchanged.
#[tokio::test]
async fn no_supplier_configured_is_a_plain_404_unchanged() {
    let (app, key) = chat_app(Provider::OpenAI);
    let body = json!({"model": "vendor/would-be-fetchable", "messages": [{"role": "user", "content": "hi"}]});
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
}

// ============================================================ tool calling

/// A chat model that ROUND-TRIPS `tools`/`enable_thinking` (echoed into
/// `reasoning_content`, so a test can assert what actually reached the
/// invocation without a second introspection channel) and streams a scripted
/// `reasoning` event plus TWO parallel tool calls, through the exact neutral
/// `Progress::event` `{"kind":...}` shapes `crates/catalog/src/resident_llm.rs`'s
/// `emit_chat_events` / `crates/catalog/src/resident_mock.rs`'s `generate_tool_call`
/// use — so `crates/apiserve`'s handling is exercised against a model that isn't
/// `resident_mock` itself, while staying byte-for-byte faithful to what a real
/// model's `ChatScanner` actually emits.
struct FakeToolCallChat;
struct FakeToolCallChatInst;
impl ResidentModel for FakeToolCallChat {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-toolcall",
            "a tool-calling chat model",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt"))
                .param(ParamSpec::new("messages", ParamType::Str, "chat messages"))
                .param(ParamSpec::new("tools", ParamType::Str, "tool definitions"))
                .param(ParamSpec::new("tool_choice", ParamType::Str, "tool choice"))
                .param(ParamSpec::new("enable_thinking", ParamType::Bool, "thinking").default(json!(true)))
                .param(ParamSpec::new("preserve_thinking", ParamType::Bool, "preserve history reasoning"))
                .param(ParamSpec::new("template_flavor", ParamType::Str, "chat template flavor"))
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-toolcall", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeToolCallChatInst))
    }
}
/// The scripted parallel calls: `(name, arguments)`, streamed as two argument
/// fragments each so fragment-concatenation is actually exercised.
const TOOLCALL_SCRIPT: [(&str, &str); 2] = [("get_weather", r#"{"location": "Paris"}"#), ("set_timer", r#"{"minutes": 5}"#)];
impl Instance for FakeToolCallChatInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Round-trip proof: what `to_invocation` actually set, echoed into
        // `reasoning_content` (itself under test for streaming/non-stream parity).
        let tools_len = inv.get_str("tools").map(|s| s.len()).unwrap_or(0);
        let enable_thinking = inv.get_bool("enable_thinking").map(|b| b.to_string()).unwrap_or_else(|| "unset".to_string());
        let preserve_thinking = inv.get_bool("preserve_thinking").map(|b| b.to_string()).unwrap_or_else(|| "unset".to_string());
        let template_flavor = inv.get_str("template_flavor").unwrap_or_else(|| "unset".to_string());
        let image_wh = inv.get_blob("image").map(|b| format!("{}x{}", b.meta["w"], b.meta["h"])).unwrap_or_else(|| "none".to_string());
        let reasoning = format!("deciding (tools_len:{tools_len};enable_thinking:{enable_thinking};preserve_thinking:{preserve_thinking};template_flavor:{template_flavor};image:{image_wh})");
        progress(Progress::event(0, 1, json!({ "kind": "reasoning", "text": reasoning })));

        let mut calls: Vec<Value> = Vec::with_capacity(TOOLCALL_SCRIPT.len());
        for (index, (name, arguments)) in TOOLCALL_SCRIPT.iter().enumerate() {
            let index = index as u32;
            let id = format!("call_{index}");
            progress(Progress::event(0, 1, json!({ "kind": "tool_call_start", "index": index, "id": id, "name": name })));
            let mid = arguments.len() / 2;
            progress(Progress::event(0, 1, json!({ "kind": "tool_call_args", "index": index, "text": &arguments[..mid] })));
            progress(Progress::event(0, 1, json!({ "kind": "tool_call_args", "index": index, "text": &arguments[mid..] })));
            progress(Progress::event(0, 1, json!({ "kind": "tool_call_end", "index": index })));
            calls.push(json!({ "id": id, "name": name, "arguments": arguments }));
        }

        Ok(Outcome::new()
            .set("prompt_tokens", json!(5))
            .set("completion_tokens", json!(2))
            .set("finish_reason", json!("tool_calls"))
            .set("reasoning_content", json!(reasoning))
            .set("tool_calls", json!(serde_json::to_string(&calls).unwrap()))
            .blob("text", Blob::new(Media::Text, Vec::new())))
    }
}
fn toolcall_executor() -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeToolCallChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}
fn toolcall_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(toolcall_executor(), key.clone(), provider);
    (router(state), key)
}

/// The sibling of [`FakeToolCallChat`]: prose only, reporting the contract
/// finish_reason a resident emits when a `required`/`named` tool_choice demand
/// went unmet (see `qwen3::chat::SeqState::finish`). Exercises the surface
/// mapping: OpenAI's enum collapses it to `stop`, while the OpenRouter
/// surface's `native_finish_reason` carries it verbatim.
struct FakeUnmetToolChoiceChat;
struct FakeUnmetToolChoiceChatInst;
impl ResidentModel for FakeUnmetToolChoiceChat {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-unmet",
            "a chat model with an unmet tool_choice demand",
            vec![ActionSpec::new("generate", "generate text").streaming().param(ParamSpec::new("prompt", ParamType::Str, "the prompt")).output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-unmet", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeUnmetToolChoiceChatInst))
    }
}
impl Instance for FakeUnmetToolChoiceChatInst {
    fn run(&mut self, _a: &str, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        Ok(Outcome::new()
            .set("prompt_tokens", json!(3))
            .set("completion_tokens", json!(7))
            .set("finish_reason", json!("tool_choice_unmet"))
            .blob("text", Blob::new(Media::Text, b"the weather in Paris is lovely".to_vec())))
    }
}
fn unmet_executor() -> Executor {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeUnmetToolChoiceChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}
fn unmet_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(unmet_executor(), key.clone(), provider);
    (router(state), key)
}

#[tokio::test]
async fn openai_chat_tool_calls_nonstream_validates_content_null_and_arguments_parse() {
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({"model": "brain-toolcall", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openai.json", "CreateChatCompletionResponse", &v);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert!(v["choices"][0]["message"]["content"].is_null(), "content must be null alongside tool_calls: {v}");
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().expect("tool_calls array present");
    assert_eq!(calls.len(), 2, "both scripted calls present");
    for c in calls {
        assert_eq!(c["type"], "function");
        assert!(c["id"].as_str().unwrap().starts_with("call_"));
        let args = c["function"]["arguments"].as_str().expect("arguments is a JSON string");
        let _: Value = serde_json::from_str(args).expect("arguments must itself parse as JSON");
    }
    // reasoning_content appears on the non-stream message (chunk of the
    // streaming/non-stream parity requirement covered by the stream test below).
    assert!(!v["choices"][0]["message"]["reasoning_content"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn openai_chat_tool_calls_stream_validates_indices_fragments_and_no_markup_leak() {
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({"model": "brain-toolcall", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let datas = sse_data(&text);
    assert_eq!(datas.last().map(String::as_str), Some("[DONE]"));

    let mut first_seen: std::collections::HashMap<i64, (String, String)> = std::collections::HashMap::new();
    let mut frags: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
    let mut saw_finish = false;
    let mut saw_reasoning = false;
    for d in datas.iter().filter(|d| d.as_str() != "[DONE]") {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_valid("openai.json", "CreateChatCompletionStreamResponse", &v);
        let delta = &v["choices"][0]["delta"];
        if let Some(c) = delta["content"].as_str() {
            assert!(!c.contains("<tool_call>") && !c.contains("<think>"), "raw markup leaked into delta.content: {c:?}");
        }
        if let Some(r) = delta["reasoning_content"].as_str() {
            if !r.is_empty() {
                saw_reasoning = true;
            }
        }
        if let Some(tcs) = delta["tool_calls"].as_array() {
            for tc in tcs {
                let index = tc["index"].as_i64().expect("tool_calls[].index present");
                if let Some(id) = tc["id"].as_str() {
                    // First chunk for this index: id+type+function.name+EMPTY arguments.
                    assert_eq!(tc["type"], "function");
                    let name = tc["function"]["name"].as_str().unwrap().to_string();
                    assert_eq!(tc["function"]["arguments"].as_str(), Some(""), "first chunk's arguments must be empty");
                    first_seen.insert(index, (id.to_string(), name));
                } else {
                    // Later chunk: index + argument fragment ONLY (no id/type/name).
                    assert!(tc.get("type").is_none(), "later chunk must not repeat 'type'");
                    assert!(tc["function"].get("name").is_none(), "later chunk must not repeat 'function.name'");
                    let frag = tc["function"]["arguments"].as_str().expect("later chunk carries an arguments fragment");
                    frags.entry(index).or_default().push_str(frag);
                }
            }
        }
        if v["choices"][0]["finish_reason"].as_str() == Some("tool_calls") {
            saw_finish = true;
        }
    }
    assert!(saw_reasoning, "reasoning_content must appear in at least one streaming delta");
    assert!(saw_finish, "terminal chunk must carry finish_reason=tool_calls");

    // Parallel calls get distinct sequential indices.
    assert_eq!(first_seen.len(), 2, "two distinct tool-call starts: {first_seen:?}");
    assert_eq!(first_seen.get(&0).map(|(_, n)| n.as_str()), Some("get_weather"));
    assert_eq!(first_seen.get(&1).map(|(_, n)| n.as_str()), Some("set_timer"));

    // Concatenated argument fragments parse as JSON, per index.
    let a0: Value = serde_json::from_str(frags.get(&0).expect("index 0 fragments")).expect("index 0 arguments parse as JSON");
    assert_eq!(a0["location"], "Paris");
    let a1: Value = serde_json::from_str(frags.get(&1).expect("index 1 fragments")).expect("index 1 arguments parse as JSON");
    assert_eq!(a1["minutes"], 5);
}

#[tokio::test]
async fn openai_chat_tools_and_enable_thinking_reach_the_invocation() {
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "enable_thinking": false,
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(!reasoning.contains("tools_len:0"), "a non-empty 'tools' must reach the invocation: {reasoning}");
    assert!(reasoning.contains("enable_thinking:false"), "'enable_thinking' must reach the invocation: {reasoning}");
}

#[tokio::test]
async fn openai_chat_preserve_thinking_reaches_the_invocation() {
    // `preserve_thinking` (the Qwen3.8 chat-template kwarg that gates whether
    // prior assistant turns keep their <think> blocks in the rendered
    // history) follows the same extension point as `reasoning_effort`:
    // nested under `chat_template_kwargs` or, tolerated, top-level.
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"preserve_thinking": false},
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("preserve_thinking:false"), "'preserve_thinking' must reach the invocation: {reasoning}");
    // ...and the top-level spelling is tolerated too.
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": "hi"}],
        "preserve_thinking": true,
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("preserve_thinking:true"), "{reasoning}");
}

#[tokio::test]
async fn openai_chat_template_flavor_reaches_the_invocation() {
    // `template_flavor` (which upstream chat template the resident renders,
    // and with it the tool-call wire form) follows the same extension point
    // as the other chat-template kwargs: nested under `chat_template_kwargs`
    // or, tolerated, top-level.
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"template_flavor": "qwen3.8"},
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("template_flavor:qwen3.8"), "'template_flavor' must reach the invocation: {reasoning}");
    // ...and the top-level spelling is tolerated too.
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": "hi"}],
        "template_flavor": "qwen3",
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("template_flavor:qwen3"), "{reasoning}");
    // Unset stays unset: the resident owns the flavor default (a Qwen3.8
    // model defaults its own flavor; the API surface must not silently pick
    // one for it).
    let body = json!({"model": "brain-toolcall", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("template_flavor:unset"), "{reasoning}");
}

#[tokio::test]
async fn openai_chat_no_tools_means_unset_enable_thinking_in_the_invocation() {
    // The sibling of the round-trip test above: when the client sends neither
    // `tools` nor `enable_thinking` (nor `preserve_thinking`), the invocation
    // must carry neither (an absent optional param, not a false/empty default
    // silently injected).
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({"model": "brain-toolcall", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("tools_len:0"));
    assert!(reasoning.contains("enable_thinking:unset"));
    assert!(reasoning.contains("preserve_thinking:unset"));
}

#[tokio::test]
async fn openai_chat_image_url_content_part_reaches_the_invocation_as_a_blob() {
    // The regression test for the "image_url/input_audio content parts are
    // silently dropped" bug -- content_text() used to keep only "text"
    // parts; apiserve::media now decodes an inline data: image_url into a
    // real HWC-f32 blob and attaches it to the Invocation under "image".
    // 1x1 white PPM (P6), same fixture as apiserve::media's own unit tests.
    const TINY_PPM_B64: &str = "UDYKMSAxCjI1NQr///8=";
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": format!("data:image/x-ppm;base64,{TINY_PPM_B64}")}},
        ]}],
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    let reasoning = v["choices"][0]["message"]["reasoning_content"].as_str().unwrap();
    assert!(reasoning.contains("image:1x1"), "the 1x1 image blob must reach the invocation: {reasoning}");
}

#[tokio::test]
async fn openai_chat_external_image_url_is_400_not_a_silent_drop() {
    let (app, key) = toolcall_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-toolcall",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}],
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{v}");
}

#[tokio::test]
async fn openrouter_chat_tool_calls_validate_against_the_openrouter_spec() {
    let (app, key) = toolcall_app(Provider::OpenRouter);

    // Non-streaming: ChatResult (not the raw OpenAI schema).
    let body = json!({"model": "brain-toolcall", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_valid("openrouter.json", "ChatResult", &v);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(v["choices"][0]["native_finish_reason"], "tool_calls");
    assert!(v["choices"][0]["message"]["tool_calls"].as_array().unwrap().len() == 2);

    // Streaming: ChatStreamChunk.
    let body = json!({"model": "brain-toolcall", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let datas = sse_data(&text);
    let mut saw_tool_calls = false;
    for d in datas.iter().filter(|d| d.as_str() != "[DONE]") {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_valid("openrouter.json", "ChatStreamChunk", &v);
        if v["choices"][0]["delta"]["tool_calls"].is_array() {
            saw_tool_calls = true;
        }
    }
    assert!(saw_tool_calls, "openrouter stream must carry tool_calls deltas");
}

/// `finish_reason: "tool_choice_unmet"` from the resident (a `required`/`named`
/// tool_choice demand the model did not satisfy - see
/// `qwen3::chat::SeqState::finish`) survives the surface mapping: OpenAI's enum
/// collapses it to `stop`, while OpenRouter's `native_finish_reason` carries the
/// contract value verbatim (non-streaming AND the streaming terminal chunk).
#[tokio::test]
async fn unmet_tool_choice_collapses_to_stop_but_surfaces_under_native_finish_reason() {
    let body = json!({"model": "brain-unmet", "messages": [{"role": "user", "content": "hi"}]});

    // Non-streaming, OpenRouter surface.
    let (app, key) = unmet_app(Provider::OpenRouter);
    let (st, v) = post_json(&app, Provider::OpenRouter, &key, "/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "OpenAI's enum has no tool_choice_unmet");
    assert_eq!(v["choices"][0]["native_finish_reason"], "tool_choice_unmet");
    assert!(v["choices"][0]["message"]["tool_calls"].is_null());
    assert_eq!(v["choices"][0]["message"]["content"], "the weather in Paris is lovely");

    // Non-streaming, plain OpenAI surface: no native field at all.
    let (app, key) = unmet_app(Provider::OpenAI);
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["choices"][0].get("native_finish_reason").is_none());

    // Streaming: the terminal chunk carries the same pair.
    let (app, key) = unmet_app(Provider::OpenRouter);
    let body_stream = json!({"model": "brain-unmet", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenRouter, &key, "/chat/completions", &body_stream).await;
    assert_eq!(st, StatusCode::OK);
    let mut saw_terminal = false;
    for d in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if d == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(d).unwrap();
        if v["choices"][0]["finish_reason"].is_string() {
            saw_terminal = true;
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
            assert_eq!(v["choices"][0]["native_finish_reason"], "tool_choice_unmet");
        }
    }
    assert!(saw_terminal, "stream must carry a terminal finish_reason chunk");
}

#[tokio::test]
async fn anthropic_messages_with_tools_is_400() {
    let (app, key) = chat_app(Provider::Anthropic);
    let body = json!({
        "model": "brain-chat", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    });
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "tools present must 400 on the Anthropic surface");
    assert_valid("anthropic.json", "ErrorResponse", &v);
}

#[tokio::test]
async fn anthropic_tool_result_content_block_is_400() {
    let (app, key) = chat_app(Provider::Anthropic);
    let body = json!({
        "model": "brain-chat", "max_tokens": 16,
        "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "22C"}]}],
    });
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "a tool_result content block must 400");
    assert_valid("anthropic.json", "ErrorResponse", &v);

    // The sibling case: a tool_use block in an assistant message.
    let body = json!({
        "model": "brain-chat", "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "get_weather", "input": {}}]},
        ],
    });
    let (st, _) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "a tool_use content block must 400");
}

#[tokio::test]
async fn openai_chat_tools_bounds_are_400_not_panics_or_500s() {
    let (app, key) = chat_app(Provider::OpenAI);

    let cases: Vec<(&str, Value)> = vec![
        ("tools not an array", json!({"model":"brain-chat","messages":[{"role":"user","content":"hi"}],"tools":"nope"})),
        (
            "tool missing function.name",
            json!({"model":"brain-chat","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{}}]}),
        ),
        (
            "tool function.name over 64 chars",
            json!({"model":"brain-chat","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"a".repeat(65)}}]}),
        ),
        (
            "more than 128 tools",
            json!({
                "model":"brain-chat","messages":[{"role":"user","content":"hi"}],
                "tools": (0..129).map(|i| json!({"type":"function","function":{"name": format!("t{i}")}})).collect::<Vec<_>>(),
            }),
        ),
        (
            "oversized tools payload",
            json!({
                "model":"brain-chat","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"t","description":"x".repeat(300 * 1024)}}],
            }),
        ),
        (
            "bad tool_choice string",
            json!({"model":"brain-chat","messages":[{"role":"user","content":"hi"}],"tool_choice":"whatever"}),
        ),
        (
            "tool_choice object missing function.name",
            json!({"model":"brain-chat","messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"function"}}),
        ),
        (
            "tool message missing tool_call_id",
            json!({"model":"brain-chat","messages":[{"role":"tool","content":"22C"}]}),
        ),
    ];
    for (label, body) in cases {
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{label} must 400, got {st}: {v}");
        assert_valid("openai.json", "ErrorResponse", &v);
    }
}

// =========================================== step-only (non-token-streaming) chat

/// The full reply [`StepOnlyChat`] returns in its terminal outcome - never as a
/// token delta.
const STEP_ONLY_TEXT: &str = "the whole answer arrives at the end";

/// A chat model that emits ONLY coarse `Progress::step` ticks (delta `None`) and
/// carries its whole reply in the terminal `Outcome`'s `text` blob - the shape
/// `crates/omni` (Qwen3-Omni) actually has. The SSE renderers used to drop such a
/// reply entirely, producing a well-formed stream with zero content.
struct StepOnlyChat;
struct StepOnlyChatInst;
impl ResidentModel for StepOnlyChat {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-steponly",
            "a chat model that reports only coarse progress",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt"))
                .param(ParamSpec::new("messages", ParamType::Str, "chat messages"))
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-steponly", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(StepOnlyChatInst))
    }
}
impl Instance for StepOnlyChatInst {
    fn run(&mut self, _a: &str, _i: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Coarse steps only - `Progress::step` sets `delta: None`.
        progress(Progress::step(0, 2, "prefill"));
        progress(Progress::step(1, 2, "decode"));
        Ok(Outcome::new()
            .set("prompt_tokens", json!(7))
            .set("completion_tokens", json!(9))
            .set("finish_reason", json!("stop"))
            .blob("text", Blob::new(Media::Text, STEP_ONLY_TEXT.as_bytes().to_vec())))
    }
}
fn step_only_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(StepOnlyChat), Arc::new(FakeChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let state = AppState::new(exec, key.clone(), provider);
    (router(state), key)
}

/// Concatenated `delta.content` across an OpenAI chat SSE body.
fn openai_stream_content(text: &str) -> String {
    sse_data(text)
        .iter()
        .filter(|d| d.as_str() != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect()
}

/// Concatenated `text_delta` text across an Anthropic messages SSE body.
fn anthropic_stream_content(text: &str) -> String {
    sse_events(text)
        .iter()
        .filter(|(ev, _)| ev == "content_block_delta")
        .filter_map(|(_, d)| serde_json::from_str::<Value>(d).ok())
        .filter_map(|v| v["delta"]["text"].as_str().map(str::to_string))
        .collect()
}

#[tokio::test]
async fn openai_chat_stream_sends_keepalive_comments_while_the_model_is_silent() {
    // A model that sits silent on the lane (a long local prefill is minutes of
    // exactly this) while the client reads the SSE body frame by frame: the
    // connection must carry SSE comment keepalives during the silence, so a
    // client's idle watchdog (any compliant SSE parser resets on bytes, and
    // comments are invisible to `data:`/`event:` parsing) never declares the
    // stream stale. Before the keep-alive the body emitted NOTHING between the
    // response headers and the first token.
    let gate = Gate::new();
    let key = "sk-brain-test-key".to_string();
    let state = AppState::new(slow_executor(gate.clone()), key.clone(), Provider::OpenAI);
    let app = router(state);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let req = auth(Request::builder().method(Method::POST).uri("/v1/chat/completions"), Provider::OpenAI, &key)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let mut frames = resp.into_body().into_data_stream();

    // Wait for the model to actually be on the lane, then hold it silent.
    let until = Instant::now() + Duration::from_secs(5);
    while gate.started() == 0 {
        assert!(Instant::now() < until, "job never started on the lane");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // The keepalive interval is 15s; a comment frame must arrive within it
    // while the model is still blocked. (First frame may be nothing at all -
    // there is no role chunk until the first delta - so keep reading until a
    // comment shows up, bounded by the interval itself.) Each frame wait is
    // itself bounded so a body that stays silent forever fails the assert
    // instead of hanging the test.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(25);
    let mut saw_comment = false;
    loop {
        match tokio::time::timeout_at(deadline, frames.next()).await {
            Err(_elapsed) => break,
            Ok(None) => panic!("stream ended before any keepalive"),
            Ok(Some(frame)) => {
                let bytes = frame.expect("stream frame");
                if bytes.contains(&b':') {
                    saw_comment = true;
                    break;
                }
            }
        }
    }
    assert!(saw_comment, "no SSE comment keepalive arrived while the model was silent");
    gate.release();
}

#[tokio::test]
async fn openai_stream_emits_the_outcome_text_when_the_model_streams_no_deltas() {
    let (app, key) = step_only_app(Provider::OpenAI);
    let body = json!({"model": "brain-steponly", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(openai_stream_content(&text), STEP_ONLY_TEXT, "the whole outcome text must be streamed as content: {text}");
    // Still a well-formed stream: every chunk validates and it terminates properly.
    let datas = sse_data(&text);
    assert_eq!(datas.last().map(String::as_str), Some("[DONE]"));
    for d in datas.iter().filter(|d| d.as_str() != "[DONE]") {
        assert_valid("openai.json", "CreateChatCompletionStreamResponse", &serde_json::from_str(d).unwrap());
    }
    assert!(
        datas.iter().filter_map(|d| serde_json::from_str::<Value>(d).ok()).any(|v| v["choices"][0]["finish_reason"] == "stop"),
        "terminal finish_reason chunk still present: {text}"
    );
}

#[tokio::test]
async fn anthropic_stream_emits_the_outcome_text_when_the_model_streams_no_deltas() {
    let (app, key) = step_only_app(Provider::Anthropic);
    let body = json!({"model": "brain-steponly", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(anthropic_stream_content(&text), STEP_ONLY_TEXT, "the whole outcome text must be streamed as a text_delta: {text}");
    // The synthetic delta must land INSIDE the content block, before its stop.
    let names: Vec<String> = sse_events(&text).into_iter().map(|(e, _)| e).collect();
    let last_delta = names.iter().rposition(|n| n == "content_block_delta").expect("a content_block_delta was emitted");
    let stop = names.iter().position(|n| n == "content_block_stop").expect("content_block_stop present");
    assert!(last_delta < stop, "the synthetic delta must precede content_block_stop: {names:?}");
    assert_eq!(names.last().map(String::as_str), Some("message_stop"));
}

#[tokio::test]
async fn a_model_that_does_stream_deltas_gets_no_synthetic_duplicate_chunk() {
    // The other half of the fix: `FakeChat` streams real token deltas AND
    // returns the same text in its outcome. The fallback must stay invisible -
    // "Hello world" exactly once, not twice.
    let (app, key) = step_only_app(Provider::OpenAI);
    let body = json!({"model": "brain-chat", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(openai_stream_content(&text), "Hello world", "no duplicated tail: {text}");

    let (app, key) = step_only_app(Provider::Anthropic);
    let body = json!({"model": "brain-chat", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let (st, text) = post_text(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(anthropic_stream_content(&text), "Hello world", "no duplicated tail: {text}");
}

// ====================================================== multimodal content parts

/// A chat model that echoes a DESCRIPTOR of the multimodal blobs that reached
/// its invocation into its reply text - byte-for-byte the format
/// `crates/catalog/src/resident_mock.rs::media_suffix` uses (`" [image:{w}x{h}]"`,
/// `" [audio:{n}samples@16k]"`), so the HTTP surfaces are exercised against the
/// same observable a `BRAIN_MOCK=1 brain serve` exposes.
///
/// The point is the POSITIVE case: `apiserve::media`'s own unit tests only cover
/// decoding in isolation plus the negative (bad format) paths, and nothing proved
/// that a decoded `input_audio` / Anthropic `image` block actually survives
/// `to_invocation` → executor → resident.
struct MediaEchoChat;
struct MediaEchoChatInst;
impl ResidentModel for MediaEchoChat {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-mediaecho",
            "a chat model that echoes attached media",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt"))
                .param(ParamSpec::new("messages", ParamType::Str, "chat messages"))
                .input(BlobSpec::new("audio", Media::Audio, "optional 16 kHz mono f32-LE PCM"))
                .input(BlobSpec::new("image", Media::Image, "optional HWC-f32 image"))
                .input(BlobSpec::new("image1", Media::Image, "a further image"))
                .input(BlobSpec::new("image2", Media::Image, "a further image"))
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-mediaecho", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(MediaEchoChatInst))
    }
}
impl Instance for MediaEchoChatInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut text = "seen:".to_string();
        for key in ["image", "image1", "image2"] {
            if let Some(b) = inv.get_blob(key) {
                text.push_str(&format!(" [{key}:{}x{}]", b.meta["w"], b.meta["h"]));
            }
        }
        if let Some(b) = inv.get_blob("audio") {
            text.push_str(&format!(" [audio:{}samples@16k]", b.bytes.len() / 4));
        }
        progress(Progress::token(0, 1, text.clone()));
        Ok(Outcome::new()
            .set("prompt_tokens", json!(1))
            .set("completion_tokens", json!(1))
            .set("finish_reason", json!("stop"))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}
fn media_echo_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(MediaEchoChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let state = AppState::new(exec, key.clone(), provider);
    (router(state), key)
}

/// A base64 WAV file: 16-bit PCM, mono, `rate` Hz, `n` samples. Built with the
/// workspace's own encoder so the fixture can't silently drift from the parser.
fn wav_b64(n: usize, rate: u32) -> String {
    let samples: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - 0.5).collect();
    events::base64::encode(&audio::wav::encode(&samples, rate))
}

/// A 1x1 white binary PPM (P6), base64 - the same fixture `apiserve::media`'s
/// unit tests use.
const TINY_PPM_B64: &str = "UDYKMSAxCjI1NQr///8=";

#[tokio::test]
async fn openai_input_audio_content_part_reaches_the_model() {
    // 320 samples at 16 kHz stay 320 after the (identity) resample, so the
    // descriptor pins the whole decode: base64 → WAV parse → 16 kHz f32 PCM blob.
    let (app, key) = media_echo_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-mediaecho",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what did you hear?"},
            {"type": "input_audio", "input_audio": {"data": wav_b64(320, 16000), "format": "wav"}},
        ]}],
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "seen: [audio:320samples@16k]", "{v}");
}

/// A 2x1 binary PPM, so the images of a request are told apart by width.
const WIDE_PPM_B64: &str = "UDYKMiAxCjI1NQr///8AAAA=";

#[tokio::test]
async fn every_image_of_a_chat_request_reaches_the_model_as_its_own_blob() {
    let ppm = |b64: &str| json!({"type": "image_url", "image_url": {"url": format!("data:image/x-ppm;base64,{b64}")}});
    let (app, key) = media_echo_app(Provider::OpenAI);
    let body = json!({"model": "brain-mediaecho", "messages": [
        {"role": "user", "content": [ppm(TINY_PPM_B64), {"type": "text", "text": "and"}, ppm(WIDE_PPM_B64)]},
    ]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "seen: [image:1x1] [image1:2x1]", "{v}");

    let (app, key) = media_echo_app(Provider::Anthropic);
    let block = |b64: &str| json!({"type": "image", "source": {"type": "base64", "media_type": "image/x-ppm", "data": b64}});
    let body = json!({"model": "brain-mediaecho", "max_tokens": 16, "messages": [{"role": "user", "content": [block(WIDE_PPM_B64), block(TINY_PPM_B64)]}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["text"], "seen: [image:2x1] [image1:1x1]", "{v}");
}

#[tokio::test]
async fn a_chat_request_with_more_images_than_the_cap_is_a_400_before_any_model_runs() {
    let part = json!({"type": "image_url", "image_url": {"url": format!("data:image/x-ppm;base64,{TINY_PPM_B64}")}});
    let parts: Vec<_> = (0..9).map(|_| part.clone()).collect();
    let (app, key) = media_echo_app(Provider::OpenAI);
    let body = json!({"model": "brain-mediaecho", "messages": [{"role": "user", "content": parts}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("at most 8 images"), "{v}");
}

#[tokio::test]
async fn openai_input_audio_is_resampled_to_16khz_before_it_reaches_the_model() {
    // An 8 kHz clip must arrive at 16 kHz (twice the samples) - the model side
    // is fixed at 16 kHz, so a pass-through would be silently wrong.
    let (app, key) = media_echo_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-mediaecho",
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav_b64(160, 8000), "format": "wav"}}]}],
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "seen: [audio:320samples@16k]", "{v}");
}

/// A chat model that echoes the raw contract `messages` JSON verbatim --
/// lets a test see EXACTLY what `to_invocation`/`flatten_message` built,
/// the same round-trip-proof pattern `FakeToolCallChatInst`'s `reasoning`
/// echo already uses for `tools`/`enable_thinking`.
struct MessagesEchoChat;
struct MessagesEchoChatInst;
impl ResidentModel for MessagesEchoChat {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-messagesecho",
            "a chat model that echoes the raw contract 'messages' JSON",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt"))
                .param(ParamSpec::new("messages", ParamType::Str, "chat messages"))
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-messagesecho", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(MessagesEchoChatInst))
    }
}
impl Instance for MessagesEchoChatInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let text = inv.get_str("messages").unwrap_or_default();
        progress(Progress::token(0, 1, text.clone()));
        Ok(Outcome::new()
            .set("prompt_tokens", json!(1))
            .set("completion_tokens", json!(1))
            .set("finish_reason", json!("stop"))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}
fn messages_echo_app(provider: Provider) -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(MessagesEchoChat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let state = AppState::new(exec, key.clone(), provider);
    (router(state), key)
}

/// Gap A's regression test: a genuinely multimodal message's `content` must
/// reach the chat-templated `messages` JSON as a TYPED ARRAY (so the real
/// Jinja template can place each medium's placeholder at its own position -
/// see `omni::caps::render_chat_prompt`'s doc), not `content_text`'s old
/// always-flatten-to-a-string behavior, which silently dropped every
/// non-"text" part before the template ever saw it. Also proves the real
/// base64 PAYLOAD is stripped, not duplicated into this JSON string -- the
/// separately-extracted blob (`openai_chat_image_url_content_part_reaches_
/// the_invocation_as_a_blob`) is still its only carrier.
#[tokio::test]
async fn openai_chat_multimodal_content_preserves_typed_parts_for_the_chat_template() {
    let (app, key) = messages_echo_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-messagesecho",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": format!("data:image/x-ppm;base64,{TINY_PPM_B64}")}},
            {"type": "input_audio", "input_audio": {"data": wav_b64(320, 16000), "format": "wav"}},
        ]}],
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    let echoed = v["choices"][0]["message"]["content"].as_str().expect("echoed messages JSON string");
    let messages: Value = serde_json::from_str(echoed).expect("echoed text is valid JSON");
    let content = &messages[0]["content"];
    assert!(content.is_array(), "multimodal content must stay a typed array, not flatten to a string: {content}");
    let parts = content.as_array().unwrap();
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");
    assert!(parts[1]["image_url"].is_object(), "image_url part must keep its 'image_url' key -- the template checks 'image_url' in content");
    assert!(!echoed.contains(TINY_PPM_B64), "the real image payload must not be duplicated into the messages JSON -- it already lives in the separately-extracted blob");
    assert_eq!(parts[2]["type"], "input_audio");
    // Gap B: the normalized key the real chat template's own audio detection
    // (`content.type == 'audio' or 'audio' in content or 'audio_url' in
    // content` -- read directly off the real chat_template.json) needs,
    // since OpenAI's real `input_audio` shape matches none of the three as-is.
    assert_eq!(parts[2]["audio_url"], true);
    assert_eq!(parts[2]["audio"], true);
}

/// Minimal-blast-radius guarantee: a message whose content is an array of
/// ONLY "text" parts (no image/audio) keeps flattening to a plain string,
/// byte-identical to this session's pre-fix behavior -- every OTHER model's
/// own `messages` parser (`qwen3::chat::parse_chat_messages`,
/// `capability::last_user_text`, etc.) sees no shape change at all for the
/// common, non-multimodal case; only a genuinely multimodal message's
/// `content` becomes an array.
#[tokio::test]
async fn openai_chat_text_only_multipart_content_still_flattens_to_a_string() {
    let (app, key) = messages_echo_app(Provider::OpenAI);
    let body = json!({
        "model": "brain-messagesecho",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}],
    });
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    let echoed = v["choices"][0]["message"]["content"].as_str().expect("echoed messages JSON string");
    let messages: Value = serde_json::from_str(echoed).expect("echoed text is valid JSON");
    assert_eq!(messages[0]["content"], "hello world");
}

#[tokio::test]
async fn anthropic_image_content_block_reaches_the_model() {
    let (app, key) = media_echo_app(Provider::Anthropic);
    let body = json!({
        "model": "brain-mediaecho",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": TINY_PPM_B64}},
        ]}],
    });
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_valid("anthropic.json", "Message", &v);
    assert_eq!(v["content"][0]["text"], "seen: [image:1x1]", "{v}");
}

#[tokio::test]
async fn a_text_only_request_attaches_no_media_blobs() {
    // The negative control: the fake reports exactly what arrived, so this
    // proves the surfaces don't invent an empty image/audio blob.
    let (app, key) = media_echo_app(Provider::OpenAI);
    let body = json!({"model": "brain-mediaecho", "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "seen:", "{v}");
}

// ============================================================== completions

/// A completion model: `generate` declares `suffix` (fill-in-the-middle) and
/// echoes what reached it - the prompt, the suffix and whether a chat
/// template was asked for - streamed as two token deltas. A prompt of
/// `"no-fim"` fails the way a checkpoint without FIM tokens does.
struct FakeFim;
struct FakeFimInst;
impl ResidentModel for FakeFim {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-fim",
            "a completion model with fill-in-the-middle",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt"))
                .param(ParamSpec::new("suffix", ParamType::Str, "text after the insertion point"))
                .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template"))
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-fim", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeFimInst))
    }
}
impl Instance for FakeFimInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = inv.get_str("prompt").unwrap_or_default();
        if prompt == "no-fim" {
            return Err("qwen: this checkpoint has no fill-in-the-middle tokens".into());
        }
        let head = format!("[{prompt}|");
        let tail = format!("{}|chat={}]", inv.get_str("suffix").unwrap_or_else(|| "-".into()), inv.get_bool("chat").map_or("-".to_string(), |c| c.to_string()));
        progress(Progress::token(0, 2, head.clone()));
        progress(Progress::token(1, 2, tail.clone()));
        Ok(Outcome::new()
            .set("prompt_tokens", json!(3))
            .set("completion_tokens", json!(2))
            .set("finish_reason", json!("length"))
            .blob("text", Blob::new(Media::Text, format!("{head}{tail}").into_bytes())))
    }
}
fn completion_app() -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeFim), Arc::new(FakeChat), Arc::new(Carded(embed_manifest()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    (router(AppState::new(exec, key.clone(), Provider::OpenAI)), key)
}

/// A completion continues the raw prompt - no chat template - and answers in
/// the legacy `text_completion` shape.
#[tokio::test]
async fn openai_completion_continues_the_raw_prompt() {
    let (app, key) = completion_app();
    for path in ["/v1/completions", "/completions"] {
        let body = json!({"model": "brain-fim", "prompt": "def f(", "max_tokens": 2});
        let (st, v) = post_json(&app, Provider::OpenAI, &key, path, &body).await;
        assert_eq!(st, StatusCode::OK, "{path}: {v}");
        assert_valid("openai.json", "CreateCompletionResponse", &v);
        assert_eq!(v["object"], "text_completion");
        assert!(v["id"].as_str().unwrap().starts_with("cmpl-"), "{v}");
        assert_eq!(v["choices"][0]["text"], "[def f(|-|chat=false]", "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["total_tokens"], 5);
    }
    // A one-element prompt array is the same request.
    let body = json!({"model": "brain-fim", "prompt": ["def f("]});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/completions", &body).await;
    assert_eq!((st, v["choices"][0]["text"].as_str()), (StatusCode::OK, Some("[def f(|-|chat=false]")), "{v}");
}

/// `suffix` reaches a model that declares fill-in-the-middle; a model that
/// does not, or whose vocabulary has no FIM tokens, is a 400 naming it.
#[tokio::test]
async fn a_suffix_is_fill_in_the_middle_or_a_400() {
    let (app, key) = completion_app();
    let body = json!({"model": "brain-fim", "prompt": "def f(", "suffix": "\n    return x"});
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["text"], "[def f(|\n    return x|chat=false]");

    for model in ["brain-chat", "brain-fim"] {
        let prompt = if model == "brain-fim" { "no-fim" } else { "def f(" };
        let body = json!({"model": model, "prompt": prompt, "suffix": "x"});
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/completions", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{model}: {v}");
        assert_valid("openai.json", "ErrorResponse", &v);
        assert!(v["error"]["message"].as_str().unwrap().contains("suffix"), "{model}: {v}");
    }
}

/// Streamed `text_completion` frames add up to the text, the last carries
/// the finish reason, usage follows when asked for, then `[DONE]`.
#[tokio::test]
async fn openai_completion_stream_concatenates_to_the_text() {
    let (app, key) = completion_app();
    let body = json!({"model": "brain-fim", "prompt": "a", "stream": true, "stream_options": {"include_usage": true}});
    let (st, text) = post_text(&app, Provider::OpenAI, &key, "/v1/completions", &body).await;
    assert_eq!(st, StatusCode::OK);
    let datas = sse_data(&text);
    assert_eq!(datas.last().map(String::as_str), Some("[DONE]"));
    let (mut content, mut finish, mut usage) = (String::new(), None, None);
    for d in datas.iter().filter(|d| d.as_str() != "[DONE]") {
        let v: Value = serde_json::from_str(d).unwrap();
        assert_eq!(v["object"], "text_completion", "{v}");
        // A frame before the last has no reason yet: `null`, as OpenAI streams
        // it, which the vendored (non-streaming) schema does not admit.
        if v["choices"].get(0).is_none_or(|c| !c["finish_reason"].is_null()) {
            assert_valid("openai.json", "CreateCompletionResponse", &v);
        }
        if let Some(c) = v["choices"].get(0) {
            content.push_str(c["text"].as_str().unwrap());
            if let Some(f) = c["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
        if !v["usage"].is_null() {
            usage = Some(v["usage"]["total_tokens"].clone());
        }
    }
    assert_eq!(content, "[a|-|chat=false]");
    assert_eq!(finish.as_deref(), Some("length"));
    assert_eq!(usage, Some(json!(5)));
}

#[tokio::test]
async fn completion_bad_bodies_are_400() {
    let (app, key) = completion_app();
    let cases: [Value; 9] = [
        json!({"prompt": "a"}),                                     // no model
        json!({"model": "brain-fim"}),                              // no prompt
        json!({"model": "brain-fim", "prompt": ["a", "b"]}),        // several prompts
        json!({"model": "brain-fim", "prompt": [1, 2, 3]}),         // token ids
        json!({"model": "brain-fim", "prompt": "a", "n": 2}),
        json!({"model": "brain-fim", "prompt": "a", "best_of": 2}),
        json!({"model": "brain-fim", "prompt": "a", "echo": true}),
        json!({"model": "brain-fim", "prompt": "a", "logprobs": 1}),
        json!({"model": "brain-fim", "prompt": "a", "max_tokens": 0}),
    ];
    for body in cases {
        let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/completions", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "must 400: {body}");
        assert_valid("openai.json", "ErrorResponse", &v);
    }
    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/completions", &json!({"model": "brain-embed", "prompt": "a"})).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "a model with no generate action");
    // The route is behind the key like every other.
    let (st, _) = post_json(&app, Provider::OpenAI, "nope", "/v1/completions", &json!({"model": "brain-fim", "prompt": "a"})).await;
    assert_eq!(st, StatusCode::UNAUTHORIZED);
}

// ======================================================= anthropic thinking

/// A reasoning chat model: `generate` streams its reasoning as `reasoning`
/// events, then the answer as token deltas, the way a Qwen3/R1 resident
/// does. With `enable_thinking: false` it answers without reasoning; an
/// assistant turn in the history carrying `reasoning_content` is echoed back
/// so its arrival is observable.
struct FakeReasoner;
struct FakeReasonerInst;
impl ResidentModel for FakeReasoner {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-reasoner",
            "a reasoning chat model",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("messages", ParamType::Str, "chat messages"))
                .param(ParamSpec::new("enable_thinking", ParamType::Bool, "reason first"))
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-reasoner", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeReasonerInst))
    }
}
impl Instance for FakeReasonerInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let history: Value = serde_json::from_str(&inv.get_str("messages").unwrap_or_default()).unwrap_or(Value::Null);
        let earlier = history.as_array().into_iter().flatten().find_map(|m| m["reasoning_content"].as_str().map(str::to_string));
        let thinking = inv.get_bool("enable_thinking").unwrap_or(true);
        let mut reasoning = String::new();
        if thinking {
            for piece in ["Let me ", "think."] {
                reasoning.push_str(piece);
                progress(Progress::event(0, 2, json!({"kind": "reasoning", "text": piece})));
            }
        }
        let text = match earlier {
            Some(r) => format!("earlier: {r}"),
            None => "Hello world".to_string(),
        };
        progress(Progress::token(1, 2, text.clone()));
        let mut out = Outcome::new().set("prompt_tokens", json!(4)).set("completion_tokens", json!(6)).set("finish_reason", json!("stop"));
        if !reasoning.is_empty() {
            out = out.set("reasoning_content", json!(reasoning));
        }
        Ok(out.blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}
fn reasoner_app() -> (Router, String) {
    let key = "sk-brain-test-key".to_string();
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeReasoner)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    (router(AppState::new(exec, key.clone(), Provider::Anthropic)), key)
}

/// The vendored Anthropic schema predates extended thinking, so a thinking
/// block is checked by its documented shape and the rest of the message by
/// the schema.
fn assert_thinking_block(block: &Value, thinking: &str) {
    assert_eq!(block["type"], "thinking", "{block}");
    assert_eq!(block["thinking"], thinking, "{block}");
    assert!(block["signature"].is_string(), "a thinking block carries a signature: {block}");
}

/// Reasoning is a `thinking` block ahead of the answer's `text` block.
#[tokio::test]
async fn anthropic_reasoning_becomes_a_thinking_block() {
    let (app, key) = reasoner_app();
    let body = json!({"model": "brain-reasoner", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_thinking_block(&v["content"][0], "Let me think.");
    assert_eq!(v["content"][1], json!({"type": "text", "text": "Hello world"}));
    let mut rest = v.clone();
    rest["content"] = json!([v["content"][1].clone()]);
    assert_valid("anthropic.json", "Message", &rest);

    // Thinking turned off: no thinking block at all.
    let off = json!({"model": "brain-reasoner", "max_tokens": 64, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &off).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["content"], json!([{"type": "text", "text": "Hello world"}]));
}

/// A thinking block in the history reaches the model as that turn's
/// reasoning, for the chat template to keep or drop.
#[tokio::test]
async fn anthropic_thinking_history_reaches_the_model_as_reasoning() {
    let (app, key) = reasoner_app();
    let body = json!({"model": "brain-reasoner", "max_tokens": 64, "thinking": {"type": "disabled"}, "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "they greeted me", "signature": ""}, {"type": "text", "text": "Hello"}]},
        {"role": "user", "content": "again"},
    ]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["text"], "earlier: they greeted me");
}

/// Streamed: a thinking block (index 0) of `thinking_delta`s closed by a
/// `signature_delta`, then the text block (index 1).
#[tokio::test]
async fn anthropic_stream_carries_thinking_then_text() {
    let (app, key) = reasoner_app();
    let body = json!({"model": "brain-reasoner", "max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "hi"}]});
    let (st, text) = post_text(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK);
    let events = sse_events(&text);
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    let (mut thinking, mut answer, mut signed) = (String::new(), String::new(), false);
    for (name, data) in &events {
        let v: Value = serde_json::from_str(data).unwrap();
        match (name.as_str(), v["delta"]["type"].as_str(), v["content_block"]["type"].as_str()) {
            ("content_block_start", _, Some("thinking")) => {
                assert_eq!(v["index"], 0);
                assert_thinking_block(&v["content_block"], "");
            }
            ("content_block_delta", Some("thinking_delta"), _) => {
                assert_eq!(v["index"], 0);
                thinking.push_str(v["delta"]["thinking"].as_str().unwrap());
            }
            ("content_block_delta", Some("signature_delta"), _) => {
                assert!(v["delta"]["signature"].is_string());
                signed = true;
            }
            ("content_block_delta", Some("text_delta"), _) => {
                assert_eq!(v["index"], 1, "the text block follows the thinking block");
                answer.push_str(v["delta"]["text"].as_str().unwrap());
                assert_valid("anthropic.json", "ContentBlockDeltaEvent", &v);
            }
            ("content_block_start", _, Some("text")) => {
                assert_eq!(v["index"], 1);
                assert_valid("anthropic.json", "ContentBlockStartEvent", &v);
            }
            _ => {}
        }
    }
    assert_eq!((thinking.as_str(), answer.as_str(), signed), ("Let me think.", "Hello world", true), "{names:?}");
    let stops = events.iter().filter(|(e, _)| e == "content_block_stop").count();
    assert_eq!(stops, 2, "{names:?}");
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
}

/// `thinking` is `{"type": "enabled", "budget_tokens": N}` with
/// 1024 <= N < max_tokens, or `{"type": "disabled"}`; anything else is a 400.
#[tokio::test]
async fn anthropic_thinking_config_is_validated() {
    let (app, key) = reasoner_app();
    let ok = json!({"type": "enabled", "budget_tokens": 1024});
    let body = json!({"model": "brain-reasoner", "max_tokens": 2048, "thinking": ok, "messages": [{"role": "user", "content": "hi"}]});
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    for bad in [
        json!({"type": "enabled"}),
        json!({"type": "enabled", "budget_tokens": 1000}),
        json!({"type": "enabled", "budget_tokens": 4096}),
        json!({"type": "sometimes"}),
        json!("enabled"),
    ] {
        let body = json!({"model": "brain-reasoner", "max_tokens": 2048, "thinking": bad, "messages": [{"role": "user", "content": "hi"}]});
        let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{bad}: {v}");
        assert_valid("anthropic.json", "ErrorResponse", &v);
        assert!(v["error"]["message"].as_str().unwrap().contains("thinking"), "{v}");
    }
}

// ------------------------------------------------- adapters as models of their own
//
// A LoRA adapter is served as a model id beside its base: the same resident
// instance, a card of its own naming the adapter by identity, and a reserved
// param the host stamps so the instance knows which id was asked for. The
// resident layer (`crates/catalog`) owns the real thing; these fakes pin what
// the API surface must do with it, whoever builds it.

const ADAPTER_BASE: &str = "brain-chat";
const ADAPTER_ID: &str = "brain-chat:lab:tuned:v1";
const ADAPTER_PARAM: &str = "__adapter";

/// A chat id served by the one shared instance: the base (`adapter: None`) or
/// a view riding on it.
struct SiblingChat {
    name: String,
    adapter: Option<residency::AdapterIdentity>,
}

struct SiblingChatInst;

impl ResidentModel for SiblingChat {
    fn manifest(&self) -> Manifest {
        let mut m = chat_manifest();
        m.model = self.name.clone();
        m
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(ADAPTER_BASE, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(SiblingChatInst))
    }
    fn stamp(&self, inv: &mut Invocation) {
        let Some(params) = inv.params.as_object_mut() else { return };
        match &self.adapter {
            Some(_) => params.insert(ADAPTER_PARAM.to_string(), json!(self.name)),
            None => params.remove(ADAPTER_PARAM),
        };
    }
    fn adapter_identity(&self) -> Option<residency::AdapterIdentity> {
        self.adapter.clone()
    }
}

impl Instance for SiblingChatInst {
    /// The answer names who answered; a seeded decode is a pure function of the
    /// seed, and an unseeded request draws a fresh one, as the real models do.
    /// The seed used and the adapter that answered are reported in the outcome.
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        static DRAWN: AtomicI64 = AtomicI64::new(1_000);
        let requested = inv.get_i64("seed");
        let seed = requested.unwrap_or_else(|| DRAWN.fetch_add(1, Ordering::SeqCst));
        let adapter = inv.get_str(ADAPTER_PARAM);
        let who = adapter.clone().unwrap_or_else(|| "the base".to_string());
        let mut out = Outcome::new().set("prompt_tokens", json!(5)).set("completion_tokens", json!(2)).set("finish_reason", json!("stop")).set("seed", json!(seed));
        if let Some(id) = adapter {
            out = out.set("adapter", json!({"id": id, "digest": adapter_identity().digest}));
        }
        if inv.get_bool("logprobs") == Some(true) || inv.get_bool("return_token_ids") == Some(true) {
            out = out.set("completion_token_ids", json!([11, 12])).set("completion_logprobs", json!([-0.5, -0.25]));
        }
        if inv.get_bool("return_token_ids") == Some(true) {
            out = out.set("prompt_token_ids", json!([1, 2, 3, 4, 5]));
        }
        let answer = match requested {
            Some(seed) => format!("answered by {who} with seed {seed}"),
            None => format!("answered by {who}"),
        };
        Ok(out.blob("text", Blob::new(Media::Text, answer.into_bytes())))
    }
}

fn adapter_identity() -> residency::AdapterIdentity {
    residency::AdapterIdentity { id: ADAPTER_ID.into(), digest: format!("sha256:{}", "c".repeat(64)) }
}

/// The base and one adapter riding on it, behind `provider`.
fn adapter_app(provider: Provider) -> (Router, Executor, String) {
    let models: Vec<Arc<dyn ResidentModel>> = vec![
        Arc::new(SiblingChat { name: ADAPTER_BASE.into(), adapter: None }),
        Arc::new(SiblingChat { name: ADAPTER_ID.into(), adapter: Some(adapter_identity()) }),
        Arc::new(Carded(embed_manifest())),
    ];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let key = "sk-brain-test-key".to_string();
    (router(AppState::new(exec.clone(), key.clone(), provider)), exec, key)
}

fn chat_body(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
}

/// An adapter is listed as a model of its own, named by identity, and its base
/// is listed exactly as before; no card carries a path.
#[tokio::test]
async fn an_adapter_is_a_listed_model_named_by_identity_and_never_by_a_path() {
    for (provider, list_uri, one_uri) in [(Provider::OpenAI, "/v1/models", format!("/v1/models/{ADAPTER_ID}")), (Provider::OpenRouter, "/models", format!("/models/{ADAPTER_ID}"))] {
        let (app, _exec, key) = adapter_app(provider);
        let listed = get_json(&app, provider, list_uri, &key).await;
        let cards = listed["data"].as_array().unwrap();
        let base = cards.iter().find(|m| m["id"] == ADAPTER_BASE).unwrap_or_else(|| panic!("{provider}: base missing: {listed}"));
        let adapter = cards.iter().find(|m| m["id"] == ADAPTER_ID).unwrap_or_else(|| panic!("{provider}: adapter missing: {listed}"));
        assert!(base.get("adapter").is_none(), "{provider}: the base entry names no adapter: {base}");
        assert_eq!(adapter["adapter"], json!({"id": ADAPTER_ID, "digest": adapter_identity().digest}), "{provider}: {adapter}");

        let card = get_json(&app, provider, &one_uri, &key).await;
        assert_eq!(card["adapter"]["digest"], json!(adapter_identity().digest), "{provider}: {card}");
        let text = listed.to_string();
        for path_like in [".safetensors", "/tmp", "/home", "/data", "\\"] {
            assert!(!text.contains(path_like), "{provider}: no path in any card ({path_like}): {text}");
        }
        let embed = cards.iter().find(|m| m["id"] == "brain-embed").unwrap();
        assert!(embed.get("adapter").is_none(), "{provider}: only an adapter view names an adapter");
    }
    // The Anthropic list is the same models, with its unchanged card shape.
    let (app, _exec, key) = adapter_app(Provider::Anthropic);
    let listed = get_json(&app, Provider::Anthropic, "/v1/models", &key).await;
    let ids: Vec<&str> = listed["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&ADAPTER_ID) && ids.contains(&ADAPTER_BASE), "{ids:?}");
}

/// An adapter id is a model like any other for authentication: its card and its
/// chat route need the key, and what an unauthenticated caller sees for it is
/// what it sees for an id that does not exist, so no adapter can be enumerated.
#[tokio::test]
async fn an_adapter_model_id_requires_the_key_and_its_absence_of_one_reveals_nothing() {
    for (provider, card_uri, chat_path, ghost_uri) in [
        (Provider::OpenAI, format!("/v1/models/{ADAPTER_ID}"), "/v1/chat/completions", "/v1/models/brain-chat:lab:ghost:v9"),
        (Provider::Anthropic, format!("/v1/models/{ADAPTER_ID}"), "/v1/messages", "/v1/models/brain-chat:lab:ghost:v9"),
        (Provider::OpenRouter, format!("/models/{ADAPTER_ID}"), "/chat/completions", "/models/brain-chat:lab:ghost:v9"),
    ] {
        let (app, _exec, _key) = adapter_app(provider);
        for key in [None, Some(""), Some("not-the-key")] {
            let builder = |uri: &str| match key {
                Some(k) => auth(Request::builder().uri(uri), provider, k),
                None => Request::builder().uri(uri),
            };
            let (st_card, body_card) = send(&app, builder(&card_uri).body(Body::empty()).unwrap()).await;
            let (st_ghost, body_ghost) = send(&app, builder(ghost_uri).body(Body::empty()).unwrap()).await;
            assert_eq!(st_card, StatusCode::UNAUTHORIZED, "{provider}: the adapter's card needs the key ({key:?})");
            assert_eq!((st_card, &body_card), (st_ghost, &body_ghost), "{provider}: an adapter id and an absent id are indistinguishable without the key");

            let post = |model: &str| {
                let mut req = builder(chat_path).method(Method::POST).header(header::CONTENT_TYPE, "application/json");
                if provider == Provider::Anthropic {
                    req = req.header("anthropic-version", "2023-06-01");
                }
                let mut body = chat_body(model);
                body["max_tokens"] = json!(8);
                req.body(Body::from(body.to_string())).unwrap()
            };
            let (st, _) = send(&app, post(ADAPTER_ID)).await;
            assert_eq!(st, StatusCode::UNAUTHORIZED, "{provider}: chatting with the adapter needs the key ({key:?})");
        }
    }
}

/// Requests for the base and for an adapter, in flight together on the one
/// instance, are each answered as the id they named.
#[tokio::test]
async fn mixed_base_and_adapter_requests_are_answered_as_the_ids_they_named() {
    let (app, exec, key) = adapter_app(Provider::OpenAI);
    let mut asked = Vec::new();
    for i in 0..8 {
        let model = if i % 2 == 0 { ADAPTER_BASE } else { ADAPTER_ID };
        let app = app.clone();
        let key = key.clone();
        asked.push((model, tokio::spawn(async move { post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &chat_body(model)).await })));
    }
    for (model, handle) in asked {
        let (st, v) = handle.await.unwrap();
        assert_eq!(st, StatusCode::OK, "{model}: {v}");
        let want = if model == ADAPTER_BASE { "answered by the base".to_string() } else { format!("answered by {ADAPTER_ID}") };
        assert_eq!(v["choices"][0]["message"]["content"], json!(want), "{model} must be answered as itself");
    }
    assert_eq!(exec.stats().builds, 1, "one instance served both ids");

    // The same through the Anthropic surface.
    let (app, _exec, key) = adapter_app(Provider::Anthropic);
    let mut body = chat_body(ADAPTER_ID);
    body["max_tokens"] = json!(8);
    let (st, v) = post_json(&app, Provider::Anthropic, &key, "/v1/messages", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["text"], json!(format!("answered by {ADAPTER_ID}")));
}

/// An adapter that is not served - never loaded, or withdrawn - is exactly an
/// unknown model: the same status and the same body shape, and nothing about
/// adapters in it.
#[tokio::test]
async fn an_unserved_adapter_id_gets_the_same_404_as_an_unknown_model() {
    for provider in ALL {
        let (app, exec, key) = adapter_app(provider);
        let (chat, card) = match provider {
            Provider::OpenAI => ("/v1/chat/completions", "/v1/models/"),
            Provider::Anthropic => ("/v1/messages", "/v1/models/"),
            Provider::OpenRouter => ("/chat/completions", "/models/"),
        };
        let body_for = |model: &str| {
            let mut b = chat_body(model);
            b["max_tokens"] = json!(8);
            b
        };
        // Shape of the answer with the id replaced by a placeholder.
        let masked = |v: &Value, id: &str| v.to_string().replace(id, "<id>");

        let (st_unknown, unknown) = post_json(&app, provider, &key, chat, &body_for("no-such-model")).await;
        assert_eq!(st_unknown, StatusCode::NOT_FOUND);
        let (st_card_unknown, card_unknown) = send(&app, auth(Request::builder().uri(format!("{card}no-such-model")), provider, &key).body(Body::empty()).unwrap()).await;

        // Withdrawn after being served.
        exec.unregister(ADAPTER_ID);
        let mut candidates = vec!["brain-chat:lab:ghost:v9", "brain-chat:lab:../x:v1", "brain-chat:a:b:c:d", ADAPTER_ID];
        candidates.push("Qwen/Qwen3-0.6B:lab:never-trained:latest");
        for id in candidates {
            let (st, v) = post_json(&app, provider, &key, chat, &body_for(id)).await;
            assert_eq!(st, StatusCode::NOT_FOUND, "{provider}: {id}: {v}");
            assert_eq!(masked(&v, id), masked(&unknown, "no-such-model"), "{provider}: {id} must read as an unknown model");
            // A slash in an id travels percent-encoded in a path segment.
            let encoded = id.replace('/', "%2F");
            let (st, v) = send(&app, auth(Request::builder().uri(format!("{card}{encoded}")), provider, &key).body(Body::empty()).unwrap()).await;
            assert_eq!((st, masked(&v, id)), (st_card_unknown, masked(&card_unknown, "no-such-model")), "{provider}: GET card for {id}");
        }
        let listed = get_json(&app, provider, if provider == Provider::OpenRouter { "/models" } else { "/v1/models" }, &key).await;
        assert!(!listed.to_string().contains(ADAPTER_ID), "{provider}: a withdrawn adapter is no longer listed");
    }
}

/// The param that carries the selection is the host's: a caller cannot set it,
/// on any route, and a caller reaching the base never lands on an adapter.
#[tokio::test]
async fn the_reserved_adapter_param_cannot_be_set_by_a_caller() {
    let (app, _exec, key) = adapter_app(Provider::OpenAI);

    // Extra body fields on a chat request are not carried to the action.
    let mut body = chat_body(ADAPTER_BASE);
    body[ADAPTER_PARAM] = json!(ADAPTER_ID);
    body["adapter"] = json!(ADAPTER_ID);
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &body).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], json!("answered by the base"), "a forged selection must not reach an adapter");

    // The generic routes validate against the served manifest, which declares
    // neither name, so both are refused as unknown params before anything runs.
    for model in [ADAPTER_BASE, ADAPTER_ID] {
        for forged in [ADAPTER_PARAM, "adapter"] {
            let run = json!({"model": model, "action": "generate", "params": {"prompt": "hi", forged: ADAPTER_ID}, "blobs": {}});
            for path in ["/v1/run", "/v1/jobs"] {
                let (st, v) = post_json(&app, Provider::OpenAI, &key, path, &run).await;
                assert_eq!(st, StatusCode::BAD_REQUEST, "{model} {path} {forged}: {v}");
                assert!(v.to_string().contains("unknown param"), "{model} {path} {forged}: {v}");
            }
        }
    }

    // The served catalogue never advertises it either.
    let (st, caps) = send(&app, auth(Request::builder().uri("/v1/capabilities"), Provider::OpenAI, &key).body(Body::empty()).unwrap()).await;
    assert_eq!(st, StatusCode::OK);
    assert!(!caps.to_string().contains(ADAPTER_PARAM), "{caps}");
}

// ------------------------------------------------- seed echo and served identity

/// A reply names the seed it decoded with, so a caller that did not choose one
/// can reproduce the decode; the seed it chose comes back unchanged and repeats
/// the same text. Both chat and the legacy completions endpoint say so, in the
/// `brain` extension object - never in a field a standard client validates.
#[tokio::test]
async fn a_reply_names_the_seed_it_used_and_a_seeded_request_repeats() {
    let (app, _exec, key) = adapter_app(Provider::OpenAI);
    for (path, body) in [("/v1/chat/completions", chat_body(ADAPTER_BASE)), ("/v1/completions", json!({"model": ADAPTER_BASE, "prompt": "hi"}))] {
        let mut seeded = body.clone();
        seeded["seed"] = json!(77);
        let (st, first) = post_json(&app, Provider::OpenAI, &key, path, &seeded).await;
        assert_eq!(st, StatusCode::OK, "{path}: {first}");
        let (_, second) = post_json(&app, Provider::OpenAI, &key, path, &seeded).await;
        assert_eq!(first["brain"]["seed"], json!(77), "{path}: {first}");
        assert_eq!(first["choices"][0], second["choices"][0], "{path}: the same seed gives the same text");

        let (_, a) = post_json(&app, Provider::OpenAI, &key, path, &body).await;
        let (_, b) = post_json(&app, Provider::OpenAI, &key, path, &body).await;
        let (drawn_a, drawn_b) = (a["brain"]["seed"].as_i64(), b["brain"]["seed"].as_i64());
        assert!(drawn_a.is_some() && drawn_b.is_some() && drawn_a != drawn_b, "{path}: an unseeded reply names the seed it drew: {a} {b}");
    }
}

/// The base names no adapter; an adapter model names itself by identity (id and
/// content digest) in every reply - the same pair its card carries - so a run
/// can be tied to the exact weights that answered. Streamed replies say so too,
/// on the terminal chunk.
#[tokio::test]
async fn a_reply_names_the_adapter_that_answered_by_digest() {
    let (app, _exec, key) = adapter_app(Provider::OpenAI);
    let (_, base) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &chat_body(ADAPTER_BASE)).await;
    assert!(base["brain"].get("adapter").is_none(), "the base names no adapter: {base}");
    let (_, tuned) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &chat_body(ADAPTER_ID)).await;
    assert_eq!(tuned["brain"]["adapter"], json!({"id": ADAPTER_ID, "digest": adapter_identity().digest}), "{tuned}");
    assert_eq!(tuned["model"], json!(ADAPTER_ID));

    let mut streamed = chat_body(ADAPTER_ID);
    streamed["stream"] = json!(true);
    streamed["seed"] = json!(5);
    let (_, sse) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &streamed).await;
    let frames: Vec<Value> = sse_data(&sse).iter().filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
    let with_brain: Vec<&Value> = frames.iter().filter(|f| f.get("brain").is_some()).collect();
    assert_eq!(with_brain.len(), 1, "exactly the terminal chunk carries the extension: {sse}");
    assert_eq!(with_brain[0]["brain"], json!({"seed": 5, "adapter": {"id": ADAPTER_ID, "digest": adapter_identity().digest}}));
    assert_eq!(with_brain[0]["choices"][0]["finish_reason"], json!("stop"));
}

/// A chat request that asks for them gets the sampled token ids and each
/// token's log-probability on the choice, and the prompt's token ids at the top
/// level of the reply (streamed: on the terminal chunk); a request that does
/// not ask gets neither. The legacy completions endpoint keeps refusing.
#[tokio::test]
async fn a_chat_reply_returns_token_ids_and_logprobs_only_when_asked() {
    let (app, _exec, key) = adapter_app(Provider::OpenAI);
    let mut asked = chat_body(ADAPTER_BASE);
    asked["logprobs"] = json!(true);
    asked["return_token_ids"] = json!(true);
    let (st, v) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &asked).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["token_ids"], json!([11, 12]));
    assert_eq!(v["prompt_token_ids"], json!([1, 2, 3, 4, 5]));
    let content = v["choices"][0]["logprobs"]["content"].as_array().expect("logprobs.content");
    let logprobs: Vec<f64> = content.iter().map(|t| t["logprob"].as_f64().unwrap()).collect();
    assert_eq!(logprobs, vec![-0.5, -0.25]);
    assert_eq!(content[0]["token"], json!("token_id:11"));

    let (_, plain) = post_json(&app, Provider::OpenAI, &key, "/v1/chat/completions", &chat_body(ADAPTER_BASE)).await;
    assert!(plain["choices"][0]["logprobs"].is_null() && plain["choices"][0].get("token_ids").is_none() && plain.get("prompt_token_ids").is_none(), "{plain}");

    asked["stream"] = json!(true);
    let (_, sse) = post_text(&app, Provider::OpenAI, &key, "/v1/chat/completions", &asked).await;
    let frames: Vec<Value> = sse_data(&sse).iter().filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
    let terminal: Vec<&Value> = frames.iter().filter(|f| f["choices"][0]["finish_reason"].is_string()).collect();
    assert_eq!(terminal.len(), 1, "{sse}");
    assert_eq!(terminal[0]["choices"][0]["token_ids"], json!([11, 12]));
    assert_eq!(terminal[0]["prompt_token_ids"], json!([1, 2, 3, 4, 5]));
    assert_eq!(terminal[0]["choices"][0]["logprobs"]["content"].as_array().map(Vec::len), Some(2));

    let (st, _) = post_json(&app, Provider::OpenAI, &key, "/v1/completions", &json!({"model": ADAPTER_BASE, "prompt": "hi", "logprobs": 1})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
}
