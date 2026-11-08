// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The job over HTTP: create (`POST /v1/jobs`), follow (`GET /v1/jobs/{id}/events`), result, cancel, behind
//! the surface's key, with datasets and the adapter moving by digest and never through the API.

use std::time::Duration;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::support::{data_digest, rig, rollouts, work_is_clean, BASE};

const KEY: &str = "sk-brain-test";

async fn call(app: &Router, method: Method, uri: &str, key: Option<&str>, body: Option<Value>) -> (StatusCode, Vec<u8>) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(key) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let body = match body {
        Some(v) => {
            req = req.header(header::CONTENT_TYPE, "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let response = tokio::time::timeout(Duration::from_secs(600), app.clone().oneshot(req.body(body).unwrap())).await.expect("the response").unwrap();
    let status = response.status();
    let bytes = tokio::time::timeout(Duration::from_secs(600), axum::body::to_bytes(response.into_body(), 1 << 26)).await.expect("the body").unwrap();
    (status, bytes.to_vec())
}

fn json_of(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or(Value::Null)
}

fn job_body(params: Value) -> Value {
    json!({"model": trainjob::MODEL, "action": trainjob::ACTION, "params": params})
}

#[tokio::test(flavor = "multi_thread")]
async fn a_training_job_over_http_is_created_followed_and_collected_by_digest() {
    let rig = rig("http-job", 16 << 30, true);
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let train = data_digest(&rig, &rollouts());
    let params = json!({"base": BASE, "arch": "qwen35", "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()});

    let (status, created) = call(&app, Method::POST, "/v1/jobs", Some(KEY), Some(job_body(params))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{}", String::from_utf8_lossy(&created));
    let id = json_of(&created)["id"].as_str().unwrap().to_string();

    // The stream carries each step and ends with the status.
    let (status, stream) = call(&app, Method::GET, &format!("/v1/jobs/{id}/events"), Some(KEY), None).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(stream).unwrap();
    let frames: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data:")).filter_map(|d| serde_json::from_str(d.trim()).ok()).collect();
    let steps: Vec<&Value> = frames.iter().filter(|f| f["type"] == "step").collect();
    assert_eq!(steps.len(), 2, "{text}");
    assert!(steps.iter().all(|f| f["pg"]["kl"].is_number() && f["pg"]["clip_fraction"].is_number()), "{text}");
    assert!(text.contains("event: done") && frames.last().unwrap()["status"] == "succeeded", "{text}");

    // The result names the adapter by digest and carries no adapter bytes.
    let (status, result) = call(&app, Method::GET, &format!("/v1/jobs/{id}/result"), Some(KEY), None).await;
    assert_eq!(status, StatusCode::OK);
    let result = json_of(&result);
    let digest = trainjob::Digest::parse(result["outputs"]["adapter_digest"].as_str().expect("an adapter digest")).unwrap();
    assert!(result["blobs"].get("adapter").is_none(), "the adapter does not travel through the API unless asked: {result}");
    assert!(result["blobs"].get("record").is_none(), "nor does the record: it is in the store too");
    let record_digest = trainjob::Digest::parse(result["outputs"]["record_digest"].as_str().expect("a record digest")).unwrap();
    let record: Value = serde_json::from_slice(&std::fs::read(rig.store.root().join(record_digest.hex())).unwrap()).unwrap();
    assert_eq!(record["policy_gradient"]["steps"].as_array().map(Vec::len), Some(2), "every step's statistics");
    let stored = std::fs::read(rig.store.root().join(digest.hex())).unwrap();
    assert_eq!(trainjob::store::digest_of(&stored), digest);
    assert!(work_is_clean(&rig.work_root));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_job_api_is_closed_to_those_without_the_key_and_to_every_input_it_does_not_define() {
    let rig = rig("http-security", 16 << 30, true);
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let train = data_digest(&rig, &rollouts());
    let good = json!({"base": BASE, "arch": "qwen35", "data": json!({"train": {"digest": train.hex()}}).to_string()});

    // Authentication: nothing is reachable, nothing is enumerable.
    for (method, uri) in [(Method::POST, "/v1/jobs"), (Method::GET, "/v1/jobs/00000000-0000-0000-0000-000000000000"), (Method::GET, "/v1/jobs/00000000-0000-0000-0000-000000000000/events"), (Method::DELETE, "/v1/jobs/00000000-0000-0000-0000-000000000000"), (Method::GET, "/v1/jobs/nope/events")] {
        for key in [None, Some("wrong"), Some("")] {
            let (status, body) = call(&app, method.clone(), uri, key, Some(job_body(good.clone()))).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{method} {uri} with {key:?}");
            assert!(!String::from_utf8_lossy(&body).contains(KEY));
        }
    }

    // Unknown fields, paths and digests that are not digests: each a 400 that names the problem and starts nothing.
    let mut refusals: Vec<(String, Value)> = Vec::new();
    for extra in ["out_dir", "weights", "models_dir", "path", "save", "device"] {
        let mut p = good.clone();
        p[extra] = json!("../escape");
        refusals.push((format!("param {extra}"), job_body(p)));
    }
    for bad in ["../../etc/passwd", "/etc/shadow", "G".repeat(64).as_str(), "", "blake3:abc"] {
        let mut p = good.clone();
        p["data"] = json!(json!({"train": {"digest": bad}}).to_string());
        refusals.push((format!("digest {bad:?}"), job_body(p)));
    }
    let mut p = good.clone();
    p["data"] = json!(json!({"train": {"path": "/etc/passwd"}}).to_string());
    refusals.push(("a path in the data document".into(), job_body(p)));
    let mut p = good.clone();
    p["data"] = json!("x".repeat(70_000));
    refusals.push(("an oversized data document".into(), job_body(p)));
    let mut p = good.clone();
    p["steps"] = json!(i64::MAX);
    refusals.push(("an absurd step count".into(), job_body(p)));
    let mut p = good.clone();
    p["max_block"] = json!(u64::MAX);
    refusals.push(("an absurd row length".into(), job_body(p)));
    refusals.push(("an unknown top-level field".into(), json!({"model": trainjob::MODEL, "action": trainjob::ACTION, "params": good, "path": "/etc"})));
    for (why, body) in refusals {
        let (status, response) = call(&app, Method::POST, "/v1/jobs", Some(KEY), Some(body)).await;
        assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY, "{why}: {status} {}", String::from_utf8_lossy(&response));
        assert!(!String::from_utf8_lossy(&response).contains(rig.dir.to_str().unwrap()), "{why}: the refusal names a path of this machine");
    }

    // A digest that parses but is not in the store is refused by name, and nothing else is read in its place.
    let absent = trainjob::store::digest_of(b"absent");
    let mut p = good.clone();
    p["data"] = json!(json!({"train": {"digest": absent.hex()}}).to_string());
    let (status, response) = call(&app, Method::POST, "/v1/jobs", Some(KEY), Some(job_body(p))).await;
    let response = String::from_utf8(response).unwrap();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(response.contains(&absent.hex()) && response.contains("not in the artifact store"), "{response}");
    assert!(!response.contains(rig.dir.to_str().unwrap()), "{response}");

    // A base the model store does not hold is refused the same way.
    let mut p = good.clone();
    p["base"] = json!("nobody/nothing");
    let (status, response) = call(&app, Method::POST, "/v1/jobs", Some(KEY), Some(job_body(p))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{}", String::from_utf8_lossy(&response));
}
