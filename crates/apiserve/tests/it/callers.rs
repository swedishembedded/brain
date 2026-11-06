// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving several applications
// from one inference server without letting one of them, compromised, take
// the others' share for its clients. If your team needs expertise in
// multi-tenant model serving then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Named callers: each holds its own key, may do what its role allows and no
//! more, sees only its own jobs, and holds at most its own share of the work
//! in flight. A caller that is compromised is contained to that.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use apiserve::{router, AppState, CallerSpec, Callers, Provider, Role};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const APP: &str = "sk-brain-app-key";
const TRAINER: &str = "sk-brain-trainer-key";

/// A model with a training action and a slow inference one; counts runs.
struct Model(Arc<AtomicUsize>);
struct ModelInstance(Arc<AtomicUsize>);

impl ResidentModel for Model {
    fn manifest(&self) -> Manifest {
        let out = || BlobSpec::new("text", Media::Text, "what ran");
        Manifest::new("brain-model", "a model", vec![ActionSpec::new("lora_train", "train").training().output(out()), ActionSpec::new("describe", "say what it is").output(out())])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-model", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(ModelInstance(self.0.clone())))
    }
}

impl Instance for ModelInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        // Long enough for a second request to arrive while this one is in flight.
        let until = std::time::Instant::now() + Duration::from_millis(400);
        while std::time::Instant::now() < until && !inv.cancel.is_cancelled() {
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(Outcome::new().blob("text", Blob::new(Media::Text, action.as_bytes().to_vec())))
    }
}

fn state(runs: &Arc<AtomicUsize>, app_inflight: usize) -> AppState {
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    let exec = Executor::start(vec![Arc::new(Model(runs.clone())) as Arc<dyn ResidentModel>], budgets, Policy::default());
    let callers = Callers::new(vec![
        CallerSpec { name: "app".into(), role: Role::Inference, key: APP.into(), max_in_flight: app_inflight },
        CallerSpec { name: "trainer".into(), role: Role::Training, key: TRAINER.into(), max_in_flight: 4 },
    ])
    .expect("two well-formed callers");
    AppState::new(exec, "unused-static-key", Provider::OpenAI).with_authenticator(Arc::new(callers)).with_training(true)
}

async fn call(app: &axum::Router, key: &str, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(path).header(header::AUTHORIZATION, format!("Bearer {key}")).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn run_body(action: &str) -> Option<Value> {
    Some(json!({"model": "brain-model", "action": action}))
}

#[tokio::test(flavor = "multi_thread")]
async fn each_caller_has_its_own_key_and_only_its_roles_rights() {
    let runs = Arc::new(AtomicUsize::new(0));
    let app = router(state(&runs, 4));
    assert_eq!(call(&app, "unused-static-key", "POST", "/v1/run", run_body("describe")).await.0, StatusCode::UNAUTHORIZED, "the static key is not a caller's");
    assert_eq!(call(&app, APP, "POST", "/v1/run", run_body("describe")).await.0, StatusCode::OK);
    assert_eq!(call(&app, APP, "POST", "/v1/run", run_body("lora_train")).await.0, StatusCode::FORBIDDEN, "an inference caller may not train, even on a training server");
    assert_eq!(call(&app, APP, "POST", "/v1/jobs", run_body("lora_train")).await.0, StatusCode::FORBIDDEN);
    assert_eq!(call(&app, TRAINER, "POST", "/v1/run", run_body("lora_train")).await.0, StatusCode::OK);
    assert_eq!(runs.load(Ordering::SeqCst), 2, "refused calls never reached the model");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_caller_sees_and_cancels_only_its_own_jobs() {
    let runs = Arc::new(AtomicUsize::new(0));
    let app = router(state(&runs, 4));
    let (status, job) = call(&app, APP, "POST", "/v1/jobs", run_body("describe")).await;
    assert!(status.is_success(), "{status} {job}");
    let id = job["id"].as_str().expect("a job id").to_string();
    for (method, path) in [("GET", format!("/v1/jobs/{id}")), ("DELETE", format!("/v1/jobs/{id}"))] {
        assert_eq!(call(&app, TRAINER, method, &path, None).await.0, StatusCode::NOT_FOUND, "{method} {path} from another caller");
    }
    let (_, listed) = call(&app, TRAINER, "GET", "/v1/jobs", None).await;
    assert!(!listed.to_string().contains(&id), "another caller's job is not listed: {listed}");
    assert_eq!(call(&app, APP, "GET", &format!("/v1/jobs/{id}"), None).await.0, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_caller_holds_at_most_its_own_share_in_flight() {
    let runs = Arc::new(AtomicUsize::new(0));
    let app = router(state(&runs, 1));
    let (status, _) = call(&app, APP, "POST", "/v1/jobs", run_body("describe")).await;
    assert!(status.is_success());
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(call(&app, APP, "POST", "/v1/run", run_body("describe")).await.0, StatusCode::TOO_MANY_REQUESTS, "over its budget while its job runs");
    assert_eq!(call(&app, TRAINER, "POST", "/v1/run", run_body("describe")).await.0, StatusCode::OK, "another caller's share is untouched");
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(call(&app, APP, "POST", "/v1/run", run_body("describe")).await.0, StatusCode::OK, "the share is back once the job ended");
}

#[test]
fn callers_must_be_distinct_and_well_formed() {
    let spec = |name: &str, key: &str| CallerSpec { name: name.into(), role: Role::Inference, key: key.into(), max_in_flight: 1 };
    assert!(Callers::new(vec![]).is_err(), "no callers");
    assert!(Callers::new(vec![spec("a", "k1"), spec("a", "k2")]).is_err(), "a name twice");
    assert!(Callers::new(vec![spec("a", "k1"), spec("b", "k1")]).is_err(), "a key twice");
    assert!(Callers::new(vec![spec("has space", "k1")]).is_err(), "a name that is not [a-z0-9_-]");
    assert!(Callers::new(vec![spec("a", "")]).is_err(), "an empty key");
    assert!(Callers::new(vec![CallerSpec { max_in_flight: 0, ..spec("a", "k1") }]).is_err(), "a budget of nothing");
}

/// A supplier that would fetch anything, and counts how often it was asked to.
struct CountingSupplier(Arc<AtomicUsize>);

impl residency::ModelSupplier for CountingSupplier {
    fn classify(&self, _model: &str) -> residency::Supply {
        residency::Supply::Fetchable
    }
    fn ensure(&self, _model: &str, _exec: &Executor, _progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err("nothing to fetch in a test".into())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn only_an_admin_caller_makes_the_server_fetch() {
    let (runs, fetches) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    let exec = Executor::start(vec![Arc::new(Model(runs.clone())) as Arc<dyn ResidentModel>], budgets, Policy::default());
    let callers = Callers::new(vec![
        CallerSpec { name: "app".into(), role: Role::Training, key: APP.into(), max_in_flight: 4 },
        CallerSpec { name: "ops".into(), role: Role::Admin, key: TRAINER.into(), max_in_flight: 4 },
    ])
    .unwrap();
    let supplier: Arc<dyn residency::ModelSupplier> = Arc::new(CountingSupplier(fetches.clone()));
    let app = router(AppState::new(exec, "unused", Provider::OpenAI).with_authenticator(Arc::new(callers)).with_supplier(Some(supplier)));
    let chat = |model: &str| Some(json!({"model": model, "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(call(&app, APP, "POST", "/v1/chat/completions", chat("vendor/not-here")).await.0, StatusCode::NOT_FOUND);
    assert_eq!(fetches.load(Ordering::SeqCst), 0, "a caller below admin never starts a fetch");
    assert_eq!(call(&app, TRAINER, "POST", "/v1/chat/completions", chat("vendor/not-here")).await.0, StatusCode::NOT_FOUND);
    assert_eq!(fetches.load(Ordering::SeqCst), 1, "an admin's request does");
}
