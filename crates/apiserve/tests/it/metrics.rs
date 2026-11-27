// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing what a server that
// trains and serves on shared accelerators is doing, for its clients. If your
// team needs expertise in observable LLM serving then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Conformance: the server's metrics document (`brain.server.metrics` v1) has exactly the fields its `.fields` file
//! pins (each one documented, `make check/metrics-fields`), and says what happened: a build, an eviction, answers by
//! kind, a training job with its own report. A field may be added (pin and document it); one that is renamed or goes
//! fails here. The `scheduler` object of a replica is pinned where it is made (`llmserve`), a training job's `report`
//! where it is made (`trainjob::observe`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-brain-test-key";
const GB: u64 = 1 << 30;

/// Every field of `doc` (`brain_testutil::schema::field_paths`), a cause-keyed `refused` map as one field.
pub fn field_paths(doc: &Value) -> Vec<String> {
    brain_testutil::schema::field_paths(doc, &["served.refused"])
}

/// Fail unless `live` is exactly what the `.fields` file `fields` pins: a pinned field that is missing was removed or
/// renamed, a live one not pinned is new and must be pinned (and documented, `make check/metrics-fields`).
pub fn assert_pinned(what: &str, live: &[String], fields: &str) {
    let pinned = brain_testutil::schema::pinned(fields, "");
    let gone: Vec<&String> = pinned.iter().filter(|p| !live.contains(p)).collect();
    let unpinned: Vec<&String> = live.iter().filter(|p| !pinned.contains(p)).collect();
    assert!(gone.is_empty(), "{what}: pinned fields missing (a field must never go or be renamed): {gone:?}");
    assert!(unpinned.is_empty(), "{what}: fields not pinned (pin and document a new field): {unpinned:?}");
}

/// A text model: the scheduler it publishes is a placeholder here (its fields are pinned in `llmserve`).
struct Text {
    name: &'static str,
    vram: u64,
}
struct TextInst;
impl ResidentModel for Text {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name, "a text model", vec![ActionSpec::new("generate", "generate")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.name, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(self.vram, 0)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TextInst))
    }
}
impl Instance for TextInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.params.get("fail").is_some() {
            return Err(capability::failure::tag(capability::FailureClass::InvalidRequest, "asked to fail"));
        }
        let mut out = Outcome::new();
        out.outputs = json!({"prompt_tokens": 7, "completion_tokens": 11});
        Ok(out)
    }
    fn metrics(&self) -> Vec<(String, Value)> {
        vec![("scheduler".into(), json!({"kv": {"total_tokens": 4096, "claimable_tokens": 4000}, "admission": {"admitted_width": 8}, "sequences": {"running": 0, "waiting": 0}}))]
    }
}

/// A training workload that reports about itself.
struct Trainer;
struct TrainerInst;
impl ResidentModel for Trainer {
    fn manifest(&self) -> Manifest {
        Manifest::new("brain/train", "a workload", vec![ActionSpec::new("train", "train")])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain/train", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::new(2 * GB, 0)
    }
    fn transient(&self) -> bool {
        true
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TrainerInst))
    }
}
impl Instance for TrainerInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.report.publish(json!({"current_phase": "ended"}));
        Ok(Outcome::new())
    }
}

async fn get(app: &Router, uri: &str) -> Value {
    let req = Request::builder().uri(uri).header(header::AUTHORIZATION, format!("Bearer {KEY}")).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET {uri}");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// The fields of `doc`, the insides of a replica's `scheduler` and a training job's `report` left out.
fn fields(doc: &Value) -> Vec<String> {
    field_paths(doc).into_iter().filter(|p| !p.starts_with("models[].replicas[].scheduler.") && !p.starts_with("training_jobs[].report.")).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_metrics_document_has_exactly_the_pinned_documented_fields_and_says_what_happened() {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Cpu, 8 * GB, 0);
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Text { name: "small", vram: 10 * GB }), Arc::new(Text { name: "large", vram: 20 * GB }), Arc::new(Trainer)];
    let exec = Executor::start(models, budgets, Policy::default());
    let app = router(AppState::new(exec.clone(), KEY.to_string(), Provider::OpenAI));
    let run = |model: &'static str, inv: Invocation| {
        let exec = exec.clone();
        tokio::task::spawn_blocking(move || exec.run_blocking(model, if model == "brain/train" { "train" } else { "generate" }, inv, |_| {}))
    };
    run("small", Invocation::new()).await.unwrap().expect("served");
    // The large model needs the card: the small one is evicted.
    run("large", Invocation::new()).await.unwrap().expect("served");
    run("large", Invocation::new().set("fail", json!(true))).await.unwrap().expect_err("refused");
    run("brain/train", Invocation::new()).await.unwrap().expect("trained");
    // The large model's scheduler reaches the stats with the next refresh.
    let deadline = Instant::now() + Duration::from_secs(10);
    let doc = loop {
        let doc = get(&app, "/v1/metrics").await;
        let replicas = doc["models"].as_array().unwrap().iter().find(|m| m["model"] == "large").map_or(0, |m| m["replicas"].as_array().map_or(0, Vec::len));
        if replicas > 0 || Instant::now() > deadline {
            break doc;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!((doc["schema"].as_str(), doc["schema_version"].as_u64()), (Some(apiserve::metrics::SCHEMA), Some(1)));

    assert_pinned("brain.server.metrics", &fields(&doc), include_str!("fields/server-metrics-v1.fields"));

    // And it says what happened.
    let model = |name: &str| doc["models"].as_array().unwrap().iter().find(|m| m["model"] == name).cloned().unwrap_or_else(|| panic!("{name} listed: {doc:#}"));
    assert!(doc["models"].as_array().unwrap().iter().all(|m| m["model"] != "brain/train"), "a workload is a training job, not a model");
    let small = model("small");
    assert_eq!(small["residency"]["state"], "evicted", "{small:#}");
    assert_eq!(small["residency"]["last_build"]["succeeded"], true);
    assert!(small["residency"]["last_eviction"]["at_unix_ms"].as_u64() >= small["residency"]["last_build"]["ended_unix_ms"].as_u64());
    assert_eq!(small["served"], json!({"requests_completed": 1, "prompt_tokens": 7, "completion_tokens": 11, "cancelled": 0, "refused": {}}));
    let large = model("large");
    assert_eq!((large["residency"]["state"].as_str(), large["residency"]["devices"].clone()), (Some("resident"), json!(["gpu0"])));
    assert_eq!(large["served"]["refused"], json!({"invalid_request": 1}));
    assert_eq!(large["totals"]["kv_total_tokens"], 4096, "{large:#}");
    assert_eq!(large["totals"]["admitted_width"], 8);
    let card = doc["cards"].as_array().unwrap().iter().find(|c| c["device"] == "gpu0").cloned().expect("the card");
    assert_eq!((card["total_bytes"].as_u64(), card["used_bytes"].as_u64(), card["training_reservation_bytes"].as_u64(), card["serving_held"].as_bool()), (Some(24 * GB), Some(20 * GB), Some(0), Some(false)));
    let job = &doc["training_jobs"][0];
    assert_eq!((job["model"].as_str(), job["action"].as_str(), job["state"].as_str()), (Some("brain/train"), Some("train"), Some("completed")));
    assert_eq!(job["report"], json!({"current_phase": "ended"}), "the job's own report");
    assert!(job["wall_secs"].as_f64().unwrap() >= job["queued_secs"].as_f64().unwrap());

    // `/metrics` answers the same document.
    assert_eq!(get(&app, "/metrics").await["schema"], apiserve::metrics::SCHEMA);
    exec.shutdown();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_health_answer_has_exactly_the_pinned_documented_fields() {
    let exec = Executor::start(Vec::new(), Budgets::new(), Policy::default());
    let app = router(AppState::new(exec.clone(), KEY.to_string(), Provider::OpenAI));
    let health = get(&app, "/v1/health").await;
    assert_eq!((health["schema"].as_str(), health["schema_version"].as_u64()), (Some(apiserve::health::SCHEMA), Some(1)));
    assert_pinned("brain.server.health", &field_paths(&health), include_str!("fields/server-health-v1.fields"));
    exec.shutdown();
}
