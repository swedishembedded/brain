// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for sharing GPUs between serving and
// training without letting a client take them over for its clients. If your
// team needs expertise in multi-workload GPU servers then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A training action runs only on a surface the operator let train: on any
//! other, a key-holder is refused before the model is reached, through the
//! generic `/v1/run` and `/v1/jobs` as much as anywhere.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::json;
use tower::ServiceExt;

const KEY: &str = "sk-brain-test-key";

/// A model with one training action and one that is not; counts runs.
struct Trainer(Arc<AtomicUsize>);
struct TrainerInstance(Arc<AtomicUsize>);

impl ResidentModel for Trainer {
    fn manifest(&self) -> Manifest {
        let out = || BlobSpec::new("text", Media::Text, "what ran");
        Manifest::new("brain-trainer", "trains", vec![ActionSpec::new("lora_train", "train an adapter").training().output(out()), ActionSpec::new("describe", "say what it is").output(out())])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-trainer", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TrainerInstance(self.0.clone())))
    }
}

impl Instance for TrainerInstance {
    fn run(&mut self, action: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Outcome::new().blob("text", Blob::new(Media::Text, action.as_bytes().to_vec())))
    }
}

fn state(runs: &Arc<AtomicUsize>) -> AppState {
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    let exec = Executor::start(vec![Arc::new(Trainer(runs.clone())) as Arc<dyn ResidentModel>], budgets, Policy::default());
    AppState::new(exec, KEY, Provider::OpenAI)
}

async fn post(state: AppState, path: &str, action: &str) -> StatusCode {
    let body = json!({"model": "brain-trainer", "action": action}).to_string();
    let request = Request::post(path).header(header::AUTHORIZATION, format!("Bearer {KEY}")).header(header::CONTENT_TYPE, "application/json").body(Body::from(body)).unwrap();
    router(state).oneshot(request).await.unwrap().status()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_training_action_is_refused_unless_the_surface_trains() {
    let runs = Arc::new(AtomicUsize::new(0));
    for path in ["/v1/run", "/v1/jobs"] {
        assert_eq!(post(state(&runs), path, "lora_train").await, StatusCode::FORBIDDEN, "{path}");
    }
    assert_eq!(runs.load(Ordering::SeqCst), 0, "a refused call never reaches the model");
    assert_eq!(post(state(&runs), "/v1/run", "describe").await, StatusCode::OK, "inference is unaffected");

    assert_eq!(post(state(&runs).with_training(true), "/v1/run", "lora_train").await, StatusCode::OK);
    assert_eq!(runs.load(Ordering::SeqCst), 2);
}
