// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements per-request fine-tuned variants of looped mixture-of-experts models, served from one
// resident copy of their published weights, for its clients. If your team needs expertise in training an adapter and
// putting it behind the same API as its base then you can procure our services by sending an email to
// info@swedishembedded.com.

//! An adapter the Inkling 2 trainer saves, served beside its base by the resident:
//!
//! - trained with the trainer on the pulled tiny checkpoint (every site: linears, message gates, query banks), saved,
//!   verified against the base and offered as a model of its own, a request for it decodes the trainer's adapted
//!   forward (greedy tokens its argmax, sampled log-probabilities its log-softmax) while a request for the base in the
//!   same instance decodes the base's forward;
//! - over HTTP the adapter is a listed model named by identity and never by a path, selected by its model id on every
//!   surface; a request without the key, or for an adapter that is not offered, learns nothing an unknown model would
//!   not tell it, and the reserved param that selects the adapter is the host's alone: a caller who sets it is refused.

use std::path::Path;
use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use capability::Invocation;
use catalog::resident_inkling2::Inkling2Resident;
use catalog::resident_qwen35::Qwen35Sizing;
use inkling2::Inkling2Train;
use llmserve::adapter_registry::{AdapterBankSettings, AdapterHost, ADAPTER_PARAM};
use model::finetune::{AdapterLineage, ChatTrainer};
use model::Model;
use residency::budget::Budgets;
use residency::{Device, Executor, InstanceKey, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::inkling2_served::{log_softmax_at, pulled_dir, VALID};

const BASE: &str = "local/inkling2";
const ADAPTER: &str = "local/inkling2:lab:tuned:v1";
/// Completion tokens each request decodes.
const NEW: usize = 4;

fn sizing() -> Qwen35Sizing {
    Qwen35Sizing { ctx: 256, ctx_explicit: true, max_batch: 2, ..Qwen35Sizing::default() }
}

fn chat() -> Invocation {
    Invocation::new()
        .set("messages", json!(json!([{"role": "user", "content": "tell me"}]).to_string()))
        .set("temp", json!(0.0))
        .set("max_new", json!(NEW))
        .set("ignore_eos", json!(true))
        .set("logprobs", json!(true))
        .set("return_token_ids", json!(true))
}

/// The prompt ids the resident renders `chat()` to.
fn prompt_ids(dir: &Path) -> Vec<u32> {
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let format = qwen3::chat::ChatFormat::own_template(dir).unwrap();
    let escaped = chat().escape_client_text(&|t| data::tokenizer::Tokenizer::escape_untrusted(&tok, t));
    let rendered = qwen3::chat::render_prompt_as(&format, &escaped).unwrap();
    data::tokenizer::Tokenizer::encode(&tok, &rendered.text)
}

/// A rank-2 adapter over every site the trainer adapts on this checkpoint, trained for a few steps on one row of `len`
/// tokens and saved at `path` as the card `ADAPTER` over `BASE`. Returns the trained build.
fn train_adapter(dir: &Path, len: usize, path: &Path) -> Inkling2Train {
    let lora = model::adapter::device::LoraCfg::new(2, 4.0, inkling2::train::targets(false));
    let tier = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
    // seed: any `A` draw; both sides read the file the trainer saves.
    let m = Inkling2Train::from_dir_lora(dir, lora, &inkling2::train::BuildOptions { batch: 1, seq: len as u32, base: &tier, seed: 7, continue_from: None }).unwrap();
    let tokens: Vec<u32> = (0..len as u32).map(|i| (i * 37 + 11) % 250 + 3).collect();
    let targets: Vec<u32> = tokens[1..].iter().copied().chain([model::IGNORE]).collect();
    let mut first = None;
    let mut last = 0.0;
    for step in 1..=6 {
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &tokens, targets: &targets });
        last = m.forward();
        first.get_or_insert(last);
        m.backward();
        m.adamw_step(step, 5e-2, 0.0, model::Adam::default(), Some(1.0), 1.0);
    }
    assert!(last < first.unwrap(), "the adapter trains: loss {first:?} -> {last}");
    let lineage = AdapterLineage { card_id: ADAPTER, base_id: BASE, dataset_id: None, provenance: None };
    <inkling2::finetune::Inkling2Trainer as ChatTrainer>::save_adapter(&m, path, &lineage).unwrap();
    m
}

/// The resident over `dir` with an adapter bank, and the adapter at `path` verified against it and offered as a view.
fn served(dir: &Path, path: &Path) -> (Arc<Inkling2Resident>, llmserve::adapter_registry::AdapterView<Inkling2Resident>) {
    let resident = Arc::new(Inkling2Resident::from_hf_dir(dir.to_str().unwrap(), BASE, sizing()).unwrap().with_adapter_bank(Some(AdapterBankSettings::default())).unwrap());
    let loaded = llmserve::adapter_release::verify_adapter(path, &resident.served_base()).expect("the trainer's adapter verifies against its base");
    let view = resident.adapter_view(loaded).expect("the bank holds the adapter");
    (resident, view)
}

/// Every served token of `out` against the forward `logits` (`[seq, VALID]`) of the prompt and the tokens served before
/// it: greedy, the argmax; sampled, the log-softmax at the token drawn. Returns the completion.
fn check_against(out: &Value, prompt: &[u32], logits: &dyn Fn(&[u32]) -> Vec<f32>, sampled: bool, what: &str) -> Vec<u32> {
    let served_prompt: Vec<u32> = serde_json::from_value(out["prompt_token_ids"].clone()).unwrap();
    assert_eq!(served_prompt, prompt, "{what}: the served prompt is the template's");
    let completion: Vec<u32> = serde_json::from_value(out["completion_token_ids"].clone()).unwrap();
    let logprobs: Vec<f64> = serde_json::from_value(out["completion_logprobs"].clone()).unwrap();
    assert_eq!(completion.len(), NEW, "{what}: {out}");
    let seq: Vec<u32> = prompt.iter().chain(&completion[..NEW - 1]).copied().collect();
    let all = logits(&seq);
    for (i, (&t, &lp)) in completion.iter().zip(&logprobs).enumerate() {
        let row = &all[(prompt.len() - 1 + i) * VALID..(prompt.len() + i) * VALID];
        if sampled {
            let want = log_softmax_at(row, t as usize);
            assert!((lp - want).abs() < 1e-4, "{what}: token {i}: served log-probability {lp}, forward {want}");
        } else {
            let argmax = row.iter().enumerate().fold((0, f32::MIN), |b, (j, &x)| if x > b.1 { (j, x) } else { b }).0 as u32;
            assert_eq!(t, argmax, "{what}: token {i}: the served greedy token is the forward's argmax");
        }
    }
    completion
}

#[test]
fn a_trained_adapter_is_served_beside_its_base_at_the_trainers_adapted_logits() {
    let Some(dir) = pulled_dir("adapter") else { return };
    let prompt = prompt_ids(&dir);
    let scratch = brain_testutil::Scratch::dir("inkling2-served-adapter");
    let path = scratch.path().join("adapter.safetensors");
    gpu_core::devices::with_host_tier(|| {
        let trained = train_adapter(&dir, prompt.len() + NEW - 1, &path);
        let base = inkling2::Inkling2::load(inkling2::serve::device_ops().unwrap(), &dir).unwrap();
        let (resident, view) = served(&dir, &path);
        let manifest = view.manifest();
        assert_eq!(manifest.model, ADAPTER, "the adapter is a model of its own");
        assert_eq!(view.adapter_identity().map(|a| a.id), Some(ADAPTER.to_string()));
        let mut instance = resident.activate(&InstanceKey::new(BASE, "default"), Device::Cpu).expect("the resident activates with its bank");
        let as_adapter = |inv: Invocation| {
            let mut inv = inv;
            view.stamp(&mut inv);
            inv
        };
        let adapted = |seq: &[u32]| trained.logits(seq).unwrap();
        let plain = |seq: &[u32]| base.forward(seq).unwrap();
        let sampled = |inv: Invocation| inv.set("temp", json!(1.0)).set("top_k", json!(0)).set("top_p", json!(1.0)).set("seed", json!(5));
        let mut completions = Vec::new();
        for random in [false, true] {
            let pick = |inv: Invocation| if random { sampled(inv) } else { inv };
            let out = instance.run("generate", &as_adapter(pick(chat())), &mut |_| {}).expect("the adapter is served").outputs;
            assert_eq!(out["adapter"]["id"], json!(ADAPTER), "the outcome names the adapter that answered: {out}");
            completions.push(check_against(&out, &prompt, &adapted, random, "adapter"));
            let out = instance.run("generate", &pick(chat()), &mut |_| {}).expect("the base is served").outputs;
            assert!(out.get("adapter").is_none_or(Value::is_null), "the base answers as the base: {out}");
            check_against(&out, &prompt, &plain, random, "base");
        }
        // The adapter moved the function it serves: its forward is not the base's at the prompt's last position.
        let (a, b) = (adapted(&prompt), plain(&prompt));
        let at = (prompt.len() - 1) * VALID;
        let moved = a[at..at + VALID].iter().zip(&b[at..at + VALID]).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-3, "the trained adapter moves the logits ({moved:.3e})");
        // Both in one batch: a run of the two requests together answers each as alone.
        let both = instance.run_batch("generate", &[as_adapter(chat()), chat()], &mut |_, _| {});
        let first: Vec<u32> = serde_json::from_value(both[0].as_ref().unwrap().outputs["completion_token_ids"].clone()).unwrap();
        assert_eq!(first, completions[0], "the adapter's greedy completion in a mixed batch is its completion alone");
    });
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------------------------------------------------------- over the HTTP surface

struct Rig {
    app: axum::Router,
    exec: Executor,
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.exec.shutdown();
    }
}

const KEY: &str = "sk-inkling2-adapters";

async fn send(rig: &Rig, req: Request<Body>) -> (StatusCode, Value) {
    let response = rig.app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn post(path: &str, key: Option<&str>, body: &Value) -> Request<Body> {
    let mut req = Request::builder().method(Method::POST).uri(path).header(header::CONTENT_TYPE, "application/json");
    if let Some(k) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {k}"));
    }
    req.body(Body::from(body.to_string())).unwrap()
}

fn get(path: &str, key: Option<&str>) -> Request<Body> {
    let mut req = Request::builder().uri(path);
    if let Some(k) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {k}"));
    }
    req.body(Body::empty()).unwrap()
}

fn chat_body(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "tell me"}], "max_tokens": NEW, "temperature": 0.0})
}

#[tokio::test(flavor = "multi_thread")]
async fn the_adapter_is_selected_by_its_model_id_and_its_selection_is_the_hosts_alone() {
    let Some(dir) = pulled_dir("adapter-http") else { return };
    let prompt = prompt_ids(&dir);
    let scratch = brain_testutil::Scratch::dir("inkling2-http-adapter");
    let path = scratch.path().join("adapter.safetensors");
    gpu_core::devices::with_host_tier(|| drop(train_adapter(&dir, prompt.len() + NEW - 1, &path)));
    let (resident, view) = served(&dir, &path);
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 64 << 30, 0);
    let models: Vec<Arc<dyn ResidentModel>> = vec![resident.clone(), Arc::new(view)];
    let exec = Executor::start(models, budgets, Policy::default());
    let rig = Rig { app: router(AppState::new(exec.clone(), KEY, Provider::OpenAI)), exec };

    // Listed by identity: the adapter's entry names its id and digest, no entry names a path.
    let (st, listed) = send(&rig, get("/v1/models", Some(KEY))).await;
    assert_eq!(st, StatusCode::OK, "{listed}");
    let cards = listed["data"].as_array().unwrap();
    let adapter = cards.iter().find(|m| m["id"] == ADAPTER).unwrap_or_else(|| panic!("the adapter is listed: {listed}"));
    assert_eq!(adapter["adapter"]["id"], json!(ADAPTER), "{adapter}");
    assert!(adapter["adapter"]["digest"].as_str().is_some_and(|d| d.starts_with("sha256:")), "{adapter}");
    assert!(cards.iter().any(|m| m["id"] == BASE && m.get("adapter").is_none()), "the base entry names no adapter: {listed}");
    let text = listed.to_string();
    for path_like in [".safetensors", "/tmp", "/home", "\\\\"] {
        assert!(!text.contains(path_like), "no card names a path ({path_like}): {text}");
    }

    // Selected by its model id: the adapter's greedy answer is the adapter's, the base's the base's.
    let (st, by_adapter) = send(&rig, post("/v1/chat/completions", Some(KEY), &chat_body(ADAPTER))).await;
    assert_eq!(st, StatusCode::OK, "{by_adapter}");
    let (st, by_base) = send(&rig, post("/v1/chat/completions", Some(KEY), &chat_body(BASE))).await;
    assert_eq!(st, StatusCode::OK, "{by_base}");
    assert_eq!(by_adapter["model"], json!(ADAPTER), "{by_adapter}");

    // No key: the adapter's card and chat are refused exactly as an adapter that does not exist.
    let ghost = "local/inkling2:lab:ghost:v9";
    for key in [None, Some("not-the-key")] {
        let (st_card, body_card) = send(&rig, get(&format!("/v1/models/{ADAPTER}"), key)).await;
        let (st_ghost, body_ghost) = send(&rig, get(&format!("/v1/models/{ghost}"), key)).await;
        assert_eq!(st_card, StatusCode::UNAUTHORIZED, "{body_card}");
        assert_eq!((st_card, &body_card), (st_ghost, &body_ghost), "an offered adapter and an absent one are indistinguishable without the key");
        let (st, _) = send(&rig, post("/v1/chat/completions", key, &chat_body(ADAPTER))).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "chatting with the adapter needs the key");
    }
    // An adapter that is not offered is an unknown model, whatever spelling.
    let (st_ghost, body_ghost) = send(&rig, post("/v1/chat/completions", Some(KEY), &chat_body(ghost))).await;
    let (st_none, body_none) = send(&rig, post("/v1/chat/completions", Some(KEY), &chat_body("local/no-such-model"))).await;
    assert_eq!(st_ghost, StatusCode::NOT_FOUND, "{body_ghost}");
    assert_eq!(st_ghost, st_none, "{body_ghost} vs {body_none}");
    assert_eq!(body_ghost["error"]["type"], body_none["error"]["type"], "{body_ghost} vs {body_none}");

    // The param that selects the adapter is the host's: a caller who names it, on the base or on the adapter, is
    // refused as an unknown param before anything runs; the chat surface has no field that reaches it.
    for model in [BASE, ADAPTER] {
        let run = json!({"model": model, "action": "generate", "params": {"messages": json!([{"role": "user", "content": "x"}]).to_string(), ADAPTER_PARAM: ADAPTER, "max_new": 1}});
        let (st, body) = send(&rig, post("/v1/run", Some(KEY), &run)).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{model}: {body}");
        assert!(body["error"]["message"].as_str().is_some_and(|m| m.contains(ADAPTER_PARAM)), "{model}: the refusal names the param: {body}");
    }
    let mut smuggled = chat_body(BASE);
    smuggled[ADAPTER_PARAM] = json!(ADAPTER);
    let (st, body) = send(&rig, post("/v1/chat/completions", Some(KEY), &smuggled)).await;
    if st == StatusCode::OK {
        assert_eq!(body["choices"][0]["message"]["content"], by_base["choices"][0]["message"]["content"], "a body field never selects the adapter: {body}");
    } else {
        assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
    }
    drop(rig);
    std::fs::remove_dir_all(&dir).ok();
}
