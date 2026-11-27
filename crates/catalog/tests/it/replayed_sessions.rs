// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving agent sessions whose every request extends the tokens the
// model was given and sampled before, for its clients. If your team needs expertise in exact multi-turn prompts for
// training and serving language models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! SPEC (HTTP contract): every chat reply's assistant message carries `sampled` (also in a stream's terminal chunk),
//! and every reply reports `brain.history`. A client that echoes `sampled` back on the assistant message gets a prompt
//! that holds the answer as the ids that were sampled: the history is `append_only`, and the prefix cache serves the
//! previous prompt's full blocks. A message whose blob is missing or tampered with is rendered
//! by the template, and the reply names the turn and why. Tiny Qwen3 and Qwen3.8 engines on the CPU backend, behind
//! the real OpenAI route, with a byte-level ChatML tokenizer.

use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use capability::{Invocation, Manifest};
use data::qwen_tokenizer::QwenBpe;
use llmserve::{ScheduledInstance, TextFrontend};
use model::serve::Scheduler;
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const VOCAB: u32 = 264;
const SPECIALS: [&str; 7] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>"];

/// Byte-level ids `0..256` and ChatML's markers after them, inside the tiny models' vocabulary.
fn chatml_tokenizer() -> QwenBpe {
    let vocab: serde_json::Map<String, Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let added: Vec<Value> = SPECIALS.iter().enumerate().map(|(i, s)| json!({"id": 256 + i, "content": s, "special": true})).collect();
    let tokenizer = json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}, "added_tokens": added});
    QwenBpe::from_json_bytes(tokenizer.to_string().as_bytes()).unwrap()
}

fn front() -> TextFrontend {
    let tok = chatml_tokenizer();
    let stop = vec![tok.special_id("<|im_end|>").unwrap()];
    TextFrontend::new(tok, stop, "replay", Arc::new(qwen3::chat::parse_request))
}

#[derive(Clone, Copy)]
enum Arch {
    Qwen3,
    Qwen38,
}

/// A served tiny model: the chat `generate` action over a scheduler of the arch's engine.
struct Tiny(Arch);

impl ResidentModel for Tiny {
    fn manifest(&self) -> Manifest {
        let spec = llmserve::generate_spec("generate text (tiny)", true).params(qwen3::chat::template_params(data::qwen_chat::TemplateFlavor::Qwen3));
        Manifest::new("tiny", "a tiny chat model", vec![spec])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("tiny", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(match self.0 {
            Arch::Qwen3 => {
                let cfg = qwen3::config::QwenConfig { vocab: VOCAB, ..qwen3::config::QwenConfig::tiny() };
                let weights = qwen3::init_weights(&cfg, 5);
                let engine = qwen3::serve::Engine::from_map(cfg, &weights, 8, 128, 4, 24, 16, false, false);
                Box::new(ScheduledInstance::new(front(), Scheduler::new(engine, 4), 4))
            }
            Arch::Qwen38 => {
                let cfg = qwen35::config::Qwen35Config { vocab: VOCAB, ..qwen35::config::Qwen35Config::tiny() };
                let weights = qwen35::init::init_weights(&cfg, 11);
                let engine = qwen35::serve::Engine::from_map_on(&gpu_core::Gpu::new_cpu(qwen35::model::pipelines()), cfg, &weights, 256, 4);
                Box::new(ScheduledInstance::new(front(), Scheduler::new(engine, 4), 4))
            }
        })
    }
}

fn app(arch: Arch) -> Router {
    served(arch, Provider::OpenAI)
}

fn served(arch: Arch, provider: Provider) -> Router {
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Tiny(arch))];
    router(AppState::new(Executor::start(models, budgets, Policy::default()), "k", provider))
}

async fn post(app: &Router, body: &Value) -> Value {
    post_to(app, "/v1/chat/completions", body).await
}

async fn post_to(app: &Router, uri: &str, body: &Value) -> Value {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("x-api-key", "k")
        .header(header::AUTHORIZATION, "Bearer k")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    assert_eq!(status, StatusCode::OK, "{text}");
    if body["stream"] == json!(true) {
        // The stream's chunks (an OpenAI stream's, an Anthropic stream's events), each a JSON object.
        let chunks: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
        return json!(chunks);
    }
    serde_json::from_str(&text).unwrap()
}

fn request(messages: &[Value], stream: bool) -> Value {
    json!({"model": "tiny", "messages": messages, "max_tokens": 6, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}, "return_token_ids": true, "stream": stream})
}

/// A two-call session through the route: the second call echoes the first answer as the reply gave it.
async fn two_calls(app: &Router) -> (Value, Vec<Value>, Value) {
    let mut messages = vec![json!({"role": "system", "content": "You are an agent."}), json!({"role": "user", "content": "List the files."})];
    let first = post(app, &request(&messages, false)).await;
    let message = first["choices"][0]["message"].clone();
    assert!(message["sampled"].is_object(), "a chat reply's message carries what was sampled: {first}");
    assert_eq!(first["brain"]["history"]["append_only"], json!(true), "a first call has no earlier answer: {first}");
    messages.push(message);
    messages.push(json!({"role": "user", "content": "Count them."}));
    let second = post(app, &request(&messages, false)).await;
    (first, messages, second)
}

fn ids(v: &Value) -> Vec<u32> {
    data::replay::decode_ids(v.as_str().unwrap()).unwrap()
}

#[tokio::test]
async fn an_echoed_answer_is_replayed_and_the_prefix_cache_serves_it() {
    for arch in [Arch::Qwen3, Arch::Qwen38] {
        let app = app(arch);
        let (first, _, second) = two_calls(&app).await;
        assert_eq!(second["brain"]["history"], json!({"append_only": true, "diverged_turn": null, "reason": null}), "{second}");
        let prompt1: Vec<u32> = serde_json::from_value(first["prompt_token_ids"].clone()).unwrap();
        let generated = ids(&first["choices"][0]["message"]["sampled"]["token_ids"]);
        let prompt2: Vec<u32> = serde_json::from_value(second["prompt_token_ids"].clone()).unwrap();
        let extended: Vec<u32> = prompt1.iter().chain(&generated).copied().collect();
        assert!(prompt2.starts_with(&extended), "the second prompt is the first, its answer and the turns after it");
        let cached = second["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() as usize;
        println!("prefix cache: {cached} of {} prompt ids served; first prompt {} ids, its answer {} ids", prompt2.len(), prompt1.len(), generated.len());
        // The tiny Qwen3.8 engine has no prefix cache; the Qwen3 engine keeps a finished sequence's blocks through its
        // answer, so the second call reuses every full block of the first prompt and its answer. The answer's last id
        // was sampled and never fed back, so it has no KV row.
        if matches!(arch, Arch::Qwen3) {
            assert_eq!(first["choices"][0]["finish_reason"], json!("length"), "{first}");
            assert_eq!(cached, (prompt1.len() + generated.len() - 1) / 8 * 8, "the second call reuses the first prompt's and its answer's full blocks");
        }
    }
}

#[tokio::test]
async fn a_missing_or_tampered_blob_renders_the_turn_by_the_template_and_says_so() {
    let app = app(Arch::Qwen3);
    let (_, messages, _) = two_calls(&app).await;
    let mut missing = messages.clone();
    missing[2].as_object_mut().unwrap().remove("sampled");
    let reply = post(&app, &request(&missing, false)).await;
    assert_eq!(reply["brain"]["history"], json!({"append_only": false, "diverged_turn": 2, "reason": "missing"}), "{reply}");
    let mut tampered = messages.clone();
    tampered[2]["sampled"]["context_hash"] = json!("sha256:00");
    let reply = post(&app, &request(&tampered, false)).await;
    assert_eq!(reply["brain"]["history"], json!({"append_only": false, "diverged_turn": 2, "reason": "context"}), "{reply}");
}

#[tokio::test]
async fn a_stream_carries_the_sampled_blob_in_its_terminal_chunk() {
    let app = app(Arch::Qwen3);
    let messages = vec![json!({"role": "user", "content": "List the files."})];
    let chunks = post(&app, &request(&messages, true)).await;
    let last = chunks.as_array().unwrap().iter().rev().find(|c| c["choices"][0]["finish_reason"].is_string()).expect("a terminal chunk").clone();
    assert!(last["choices"][0]["delta"]["sampled"]["token_ids"].is_string(), "{last}");
    assert_eq!(last["brain"]["history"]["append_only"], json!(true), "{last}");
}

/// SPEC (HTTP contract, Anthropic dialect): `/v1/messages` carries the answer's `sampled` blob on the reply and on a
/// stream's closing `message_delta`, takes it back on the assistant message a client sends, and reports
/// `brain.history`, as the OpenAI route does; the echoed answer is served from the prefix cache.
#[tokio::test]
async fn the_anthropic_route_carries_and_replays_the_sampled_answer() {
    let app = served(Arch::Qwen3, Provider::Anthropic);
    let ask = |messages: &[Value], stream: bool| json!({"model": "tiny", "max_tokens": 6, "temperature": 0, "system": "You are an agent.", "messages": messages, "stream": stream});
    let mut messages = vec![json!({"role": "user", "content": "List the files."})];
    let first = post_to(&app, "/v1/messages", &ask(&messages, false)).await;
    assert!(first["sampled"]["token_ids"].is_string(), "a reply carries what was sampled: {first}");
    assert_eq!(first["brain"]["history"]["append_only"], json!(true), "{first}");
    messages.push(json!({"role": "assistant", "content": first["content"], "sampled": first["sampled"]}));
    messages.push(json!({"role": "user", "content": "Count them."}));
    let second = post_to(&app, "/v1/messages", &ask(&messages, false)).await;
    assert_eq!(second["brain"]["history"], json!({"append_only": true, "diverged_turn": null, "reason": null}), "{second}");
    let cached = second["usage"]["cache_read_input_tokens"].as_u64().unwrap();
    let generated = ids(&first["sampled"]["token_ids"]).len() as u64;
    let prompt1 = first["usage"]["input_tokens"].as_u64().unwrap();
    assert_eq!(cached, (prompt1 + generated - 1) / 8 * 8, "the echoed answer is served from the prefix cache: {second}");
    let mut dropped = messages.clone();
    dropped[1].as_object_mut().unwrap().remove("sampled");
    let rendered = post_to(&app, "/v1/messages", &ask(&dropped, false)).await;
    assert_eq!(rendered["brain"]["history"]["reason"], json!("missing"), "{rendered}");
    let events = post_to(&app, "/v1/messages", &ask(&messages, true)).await;
    let closing = events.as_array().unwrap().iter().find(|e| e["type"] == "message_delta").expect("a closing message_delta").clone();
    assert!(closing["delta"]["sampled"]["token_ids"].is_string(), "{closing}");
    assert_eq!(closing["brain"]["history"]["append_only"], json!(true), "{closing}");
}
