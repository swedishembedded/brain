// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving open models behind the APIs existing agent tools already
// speak, for its clients. If your team needs expertise in keeping an inference server compatible with the clients
// that call it, then you can procure our services by sending an email to info@swedishembedded.com.

//! SPEC: the request bodies the clients that call a served chat model really send are admitted, field for field, by
//! the HTTP routes and the served Qwen3.8 `generate` spec they are validated against (`apiserve::bridge::admit`), and
//! each setting they carry reaches the model. One fixture per client shape, each taken from the client's own request
//! builder:
//!
//! - an agent runtime's OpenAI-compatible driver pointed at brain: `stream_options` on every request, `max_tokens`,
//!   the sampler it was given (`temperature`, `top_p`, `top_k`, `seed`), `policy_trace` where other servers get
//!   `logprobs` and `return_token_ids`, `tools` and `tool_choice: "none"`, the thinking switch in
//!   `chat_template_kwargs` and a `thinking_budget`; assistant turns with `reasoning_content` and `tool_calls`, tool
//!   turns, content parts;
//! - the same driver against any other OpenAI-compatible server: `logprobs` and `return_token_ids`;
//! - an exam harness's plain client: `chat_template_kwargs.enable_thinking` beside `temperature` and `max_tokens`
//!   (a gateway in front of brain forwards a body as sent, rewriting only `model`);
//! - Claude Code (2.1.325, recorded) on the Anthropic route: `system` as blocks with `cache_control`,
//!   `thinking: {"type": "adaptive", "display": "omitted"}`, `context_management` clearing no thinking,
//!   `output_config: {"effort": ...}`, `metadata.user_id`, `tool_use`/`tool_result`/`thinking` blocks, tools with
//!   `input_schema`, `cache_control` and `defer_loading`, `?beta=true` on the path.
//!
//! The model is the Qwen3.8 GGUF/FP8 resident's manifest over an instance that answers with the invocation it was
//! handed, so the test sees what admission let through and what it filled in, with no weights.

use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, Blob, Invocation, Manifest, Media, Outcome, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "Qwen/Qwen3.8-27B-FP8";

/// The served Qwen3.8 resident's manifest, answering every call with the invocation it received.
struct Echo(Manifest);

struct EchoInst;

impl ResidentModel for Echo {
    fn manifest(&self) -> Manifest {
        self.0.clone()
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
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
        let out = Outcome::new().set("prompt_tokens", json!(5)).set("completion_tokens", json!(1)).set("finish_reason", json!("stop"));
        Ok(out.blob("text", Blob::new(Media::Text, inv.params.to_string().into_bytes())))
    }
}

fn app(provider: Provider) -> (Router, Executor) {
    let resident = qwen35::int8_gguf_resident::Qwen35GgufResident::new("never-opened".to_string(), vec![(Device::Gpu(0), 96 << 30)], 4096, model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8)).with_model_id(MODEL);
    let mut manifest = resident.manifest();
    manifest.model = MODEL.to_string();
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Echo(manifest))];
    let exec = Executor::start(models, budgets, Policy::default());
    (router(AppState::new(exec.clone(), "k", provider)), exec)
}

async fn post(app: &Router, provider: Provider, path: &str, body: &Value) -> (StatusCode, String) {
    let auth = match provider {
        Provider::Anthropic => (header::HeaderName::from_static("x-api-key"), "k".to_string()),
        _ => (header::AUTHORIZATION, "Bearer k".to_string()),
    };
    let mut req = Request::builder().method(Method::POST).uri(path).header(auth.0, auth.1).header(header::CONTENT_TYPE, "application/json");
    if provider == Provider::Anthropic {
        req = req.header("anthropic-version", "2023-06-01").header("anthropic-beta", "claude-code-20250219,interleaved-thinking-2025-05-14");
    }
    let response = app.clone().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The invocation the model received for `body`, from the reply's text (streamed: the text deltas joined); the body is
/// sent as given and with `stream` flipped, and must be admitted both ways.
async fn received(provider: Provider, path: &str, body: &Value) -> Value {
    let (app, exec) = app(provider);
    let mut texts = Vec::new();
    for stream in [false, true] {
        let mut sent = body.clone();
        sent["stream"] = json!(stream);
        let (status, reply) = post(&app, provider, path, &sent).await;
        assert_eq!(status, StatusCode::OK, "{path} stream={stream}: {reply}\nsent {sent}");
        let text = if stream {
            reply
                .lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|d| serde_json::from_str::<Value>(d).ok())
                .filter_map(|f| f["choices"][0]["delta"]["content"].as_str().or(f["delta"]["text"].as_str()).map(str::to_string))
                .collect::<String>()
        } else {
            let v: Value = serde_json::from_str(&reply).unwrap();
            v["choices"][0]["message"]["content"].as_str().or(v["content"][0]["text"].as_str()).unwrap_or_default().to_string()
        };
        texts.push(serde_json::from_str::<Value>(&text).unwrap_or_else(|e| panic!("{path} stream={stream}: the echo is not JSON ({e}): {text:?}\n{reply}")));
    }
    exec.shutdown();
    assert_eq!(texts[0], texts[1], "the streamed call is the same call");
    texts.remove(0)
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "read_file", "description": "read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])
}

/// An agent turn as the OpenAI-compatible driver writes it: system, user content parts, an assistant turn with its
/// reasoning and a tool call, the tool's result.
fn agent_messages() -> Value {
    json!([
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": [{"type": "text", "text": "show main.rs"}]},
        {"role": "assistant", "reasoning_content": "I should read it.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"main.rs\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"},
    ])
}

#[tokio::test]
async fn an_agent_runtime_driving_brain_is_admitted_with_every_setting_it_sends() {
    let body = json!({
        "model": MODEL, "messages": agent_messages(), "stream_options": {"include_usage": true}, "max_tokens": 64,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 7, "policy_trace": true, "tools": tools(),
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let got = received(Provider::OpenAI, "/v1/chat/completions", &body).await;
    assert_eq!((got["temp"].clone(), got["top_k"].clone(), got["top_p"].clone(), got["seed"].clone()), (json!(1.0), json!(0), json!(1.0), json!(7)), "{got}");
    assert_eq!((got["max_new"].clone(), got["policy_trace"].clone(), got["enable_thinking"].clone()), (json!(64), json!(true), json!(false)), "{got}");
    assert!(got["tools"].as_str().is_some_and(|t| t.contains("read_file")) && got["messages"].as_str().is_some_and(|m| m.contains("call_1")), "{got}");

    // Tools withheld, and a thinking budget instead of a trace (a budget reshapes the distribution a trace records).
    let mut budgeted = body.clone();
    budgeted.as_object_mut().unwrap().remove("policy_trace");
    budgeted["tool_choice"] = json!("none");
    budgeted["chat_template_kwargs"] = json!({"enable_thinking": true});
    budgeted["thinking_budget"] = json!(256);
    let got = received(Provider::OpenAI, "/v1/chat/completions", &budgeted).await;
    assert_eq!((got["enable_thinking"].clone(), got["tool_choice"].clone()), (json!(true), json!("\"none\"")), "{got}");
    assert!(got.as_object().unwrap().values().any(|v| v == &json!(256)), "the thinking budget reaches the model: {got}");
}

#[tokio::test]
async fn the_same_driver_asking_for_token_logprobs_is_admitted() {
    let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream_options": {"include_usage": true}, "max_tokens": 8, "logprobs": true, "return_token_ids": true});
    let got = received(Provider::OpenAI, "/v1/chat/completions", &body).await;
    assert_eq!((got["logprobs"].clone(), got["return_token_ids"].clone()), (json!(true), json!(true)), "{got}");
}

#[tokio::test]
async fn an_exam_harness_switching_thinking_off_is_admitted() {
    let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 32, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": false}});
    let got = received(Provider::OpenAI, "/v1/chat/completions", &body).await;
    assert_eq!((got["enable_thinking"].clone(), got["temp"].clone(), got["max_new"].clone()), (json!(false), json!(0.0), json!(32)), "{got}");
}

#[tokio::test]
async fn claude_code_on_the_anthropic_route_is_admitted() {
    let cached = json!({"type": "ephemeral"});
    let body = json!({
        "model": MODEL, "max_tokens": 32000, "stream": true, "temperature": 1,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.220"},
            {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK.", "cache_control": cached},
            {"type": "text", "text": "Working directory: /work", "cache_control": cached},
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "show main.rs", "cache_control": cached}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should read it.", "signature": "c2ln"},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "main.rs"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "fn main() {}"}], "cache_control": cached}]},
        ],
        "tools": [
            {"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"], "additionalProperties": false, "$schema": "http://json-schema.org/draft-07/schema#"}, "cache_control": cached},
            {"name": "DeferredToolPlaceholder", "description": "found by tool search", "input_schema": {"type": "object", "properties": {}}, "defer_loading": true},
        ],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "adaptive", "display": "omitted"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "output_config": {"effort": "high"},
        "metadata": {"user_id": "{\"device_id\":\"0f\",\"account_uuid\":\"\",\"session_id\":\"4756df20\"}"},
    });
    let got = received(Provider::Anthropic, "/v1/messages?beta=true", &body).await;
    assert_eq!(got["max_new"], json!(32000), "{got}");
    assert!(got["messages"].as_str().is_some_and(|m| m.contains("toolu_1") && m.contains("I should read it.")), "the turns reach the model: {got}");
    assert!(got["tools"].as_str().is_some_and(|t| t.contains("file_path") && !t.contains("DeferredToolPlaceholder")), "a deferred tool is not shown to the model: {got}");
    assert_eq!(got["reasoning_effort"], json!("xhigh"), "Anthropic's default effort is the model's default: {got}");

    // A lower effort reaches the model's reasoning-effort setting; a member brain does not do is refused by name.
    let mut low = body.clone();
    low["output_config"] = json!({"effort": "low"});
    assert_eq!(received(Provider::Anthropic, "/v1/messages", &low).await["reasoning_effort"], json!("low"));
    let (rig, exec) = app(Provider::Anthropic);
    let mut structured = body.clone();
    structured["stream"] = json!(false);
    structured["output_config"] = json!({"format": {"type": "json_schema", "schema": {}}});
    let (status, reply) = post(&rig, Provider::Anthropic, "/v1/messages", &structured).await;
    assert!(status == StatusCode::BAD_REQUEST && reply.contains("output_config.format"), "{status} {reply}");
    // A context edit that drops part of the prompt asks for what brain does not do.
    let mut edited = structured.clone();
    edited["output_config"] = json!({"effort": "high"});
    edited["context_management"] = json!({"edits": [{"type": "clear_tool_uses_20250919"}]});
    let (status, reply) = post(&rig, Provider::Anthropic, "/v1/messages", &edited).await;
    exec.shutdown();
    assert!(status == StatusCode::BAD_REQUEST && reply.contains("context_management.edits[0]"), "{status} {reply}");
}

/// The same agent runtime's native Anthropic driver: a request-wide `cache_control` and system blocks.
#[tokio::test]
async fn an_agent_runtime_anthropic_driver_is_admitted() {
    let body = json!({
        "model": MODEL, "max_tokens": 512, "temperature": 0.7, "top_p": 0.9, "top_k": 20, "cache_control": {"type": "ephemeral"},
        "system": [{"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}, {"type": "text", "text": "Today is Tuesday."}],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "read_file", "description": "read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
    });
    let got = received(Provider::Anthropic, "/v1/messages", &body).await;
    assert_eq!((got["temp"].clone(), got["top_k"].clone(), got["max_new"].clone()), (json!(0.7), json!(20), json!(512)), "{got}");
    assert!(got["system"].as_str().is_some_and(|s| s.contains("coding agent") && s.contains("Tuesday")), "{got}");
}
