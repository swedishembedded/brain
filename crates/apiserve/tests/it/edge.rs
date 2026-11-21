// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers that admit as many
// concurrent agents as their batch carries for its clients. If your team needs
// expertise in admission control and backpressure then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The edge ceiling of the HTTP surfaces is sized from the served models' decode slots, not a constant: a model
//! that carries 300 sequences at once is offered more than 256 requests before any is shed (knowledge 538 measured
//! 768 of 1,024 agent conversations refused with 503 against a batch of 512 that had room for them).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::admission::{edge_concurrency, EDGE_CONCURRENCY};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::json;
use tower::ServiceExt;

const SLOTS: usize = 300;

/// Held until the test opens the gate: every request it is given waits there.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    turned: Condvar,
    started: AtomicUsize,
}

struct Wide(Arc<Gate>);
struct WideInst(Arc<Gate>);

impl ResidentModel for Wide {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-wide",
            "a chat model that carries many sequences",
            vec![crate::chat_contract(ActionSpec::new("generate", "generate text").streaming().param(ParamSpec::new("prompt", ParamType::Str, "the prompt")).output(BlobSpec::new("text", Media::Text, "generated text")))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-wide", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(SLOTS)
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(WideInst(Arc::clone(&self.0))))
    }
}

impl Instance for WideInst {
    fn run(&mut self, _a: &str, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.started.fetch_add(1, Ordering::SeqCst);
        let mut open = self.0.open.lock().unwrap();
        while !*open {
            open = self.0.turned.wait(open).unwrap();
        }
        Ok(Outcome::new().set("prompt_tokens", json!(1)).set("completion_tokens", json!(1)).set("finish_reason", json!("stop")).blob("text", Blob::new(Media::Text, b"ok".to_vec())))
    }
}

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::AUTHORIZATION, "Bearer sk-brain-test-key")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({"model": "brain-wide", "messages": [{"role": "user", "content": "hi"}]}).to_string()))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_up_to_the_models_slots_are_held_not_shed_at_a_fixed_256() {
    let gate = Arc::new(Gate::default());
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(vec![Arc::new(Wide(Arc::clone(&gate)))], budgets, Policy::default());
    let state = AppState::new(exec, "sk-brain-test-key", Provider::OpenAI);
    assert_eq!(state.edge_concurrency, edge_concurrency(SLOTS));
    assert!(state.edge_concurrency > EDGE_CONCURRENCY);
    let app = router(state);

    let calls: Vec<_> = (0..=EDGE_CONCURRENCY + 20)
        .map(|_| {
            let app = app.clone();
            tokio::spawn(async move { app.oneshot(chat_request()).await.unwrap().status() })
        })
        .collect();
    // While the gate is shut nothing can finish: a call that has already returned was shed at the edge.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let early: Vec<usize> = (0..calls.len()).filter(|&i| calls[i].is_finished()).collect();

    *gate.open.lock().unwrap() = true;
    gate.turned.notify_all();
    let mut statuses = Vec::new();
    for call in calls {
        statuses.push(call.await.unwrap());
    }
    let early: Vec<StatusCode> = early.into_iter().map(|i| statuses[i]).collect();
    assert!(early.is_empty(), "{} of {} requests were answered before any could run: {early:?}", early.len(), statuses.len());
    assert!(statuses.iter().all(|s| *s == StatusCode::OK), "{statuses:?}");
}
