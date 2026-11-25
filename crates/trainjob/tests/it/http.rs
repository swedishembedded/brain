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

use crate::support::{data_digest, plain, rig, rig_qwen3, rollouts, work_is_clean, Frames, BASE};

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

    // The caller's listing names the run, so a client that loses the id can find it again.
    let (status, listing) = call(&app, Method::GET, "/v1/jobs", Some(KEY), None).await;
    assert_eq!(status, StatusCode::OK);
    let listing = json_of(&listing);
    assert!(listing["data"].as_array().unwrap().iter().any(|j| j["id"] == id.as_str() && j["model"] == trainjob::MODEL && j["action"] == trainjob::ACTION), "{listing}");

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
    for (method, uri) in [(Method::POST, "/v1/jobs"), (Method::GET, "/v1/jobs"), (Method::GET, "/v1/jobs/00000000-0000-0000-0000-000000000000"), (Method::GET, "/v1/jobs/00000000-0000-0000-0000-000000000000/events"), (Method::DELETE, "/v1/jobs/00000000-0000-0000-0000-000000000000"), (Method::GET, "/v1/jobs/nope/events")] {
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

/// `check_dataset` over `POST /v1/run`: the per-record counts are the ones the executor returns for the same call (the
/// bus and this route share the spec, the door and the action), and a page over the cap is a 400 naming `limit`.
#[tokio::test(flavor = "multi_thread")]
async fn check_dataset_lists_per_record_counts_over_http_and_refuses_an_unbounded_page() {
    let rig = rig_qwen3("http-check", 16 << 30);
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let mut dataset = plain();
    dataset.extend_from_slice(b"\nnot json\n");
    let params = json!({"base": BASE, "per_record": true});
    let body = |params: &Value| json!({"model": trainjob::MODEL, "action": "check_dataset", "params": params, "blobs": {"dataset": {"media": "bytes", "data": events::base64::encode(&dataset)}}});

    let (status, response) = call(&app, Method::POST, "/v1/run", Some(KEY), Some(body(&params))).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&response));
    let over_http = json_of(&response)["outputs"].clone();
    let direct = crate::support::run_action(&rig, "check_dataset", &trainjob::dryrun::spec(), params, &[("dataset", dataset.clone())], &Frames::default()).unwrap().outputs;
    assert_eq!(over_http, direct);
    assert_eq!((over_http["per_record"].as_array().map(Vec::len), over_http["per_record_refused"][0]["line"].as_u64()), (Some(4), Some(5)));

    let (status, response) = call(&app, Method::POST, "/v1/run", Some(KEY), Some(body(&json!({"base": BASE, "per_record": true, "limit": u64::MAX})))).await;
    assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY, "{status}");
    assert!(String::from_utf8_lossy(&response).contains("limit"), "{}", String::from_utf8_lossy(&response));
}

/// A caller records the full settings of a run from `GET /v1/capabilities`: every objective setting the request may
/// omit declares the default the trainer applies, the same map the trainer's configuration gives.
#[tokio::test(flavor = "multi_thread")]
async fn the_capabilities_declare_the_defaults_the_trainer_applies() {
    let rig = rig("http-caps", 16 << 30, true);
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let (status, body) = call(&app, Method::GET, "/v1/capabilities", Some(KEY), None).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let caps = json_of(&body);
    let action = caps["data"].as_array().expect("a list of actions").iter().find(|a| a["model"] == trainjob::MODEL && a["action"] == trainjob::ACTION).expect("brain/train's train action is listed");
    let declared: serde_json::Map<String, Value> = action["params"].as_array().unwrap().iter().filter_map(|p| Some((p["name"].as_str()?.to_string(), p.get("implied_default")?.clone()))).collect();
    assert_eq!(declared, trainjob::request::implied_defaults());
    assert_eq!(declared["clip_eps"], json!(0.2), "a decimal, not the widened f32");
}

/// A setting that needs another is refused at submission naming both, not failed once queued as a model fault the
/// caller cannot read.
#[tokio::test(flavor = "multi_thread")]
async fn selecting_without_evaluations_is_refused_at_submission_naming_the_setting() {
    let rig = rig_qwen3("http-select-needs-eval", 16 << 30);
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let train = data_digest(&rig, &plain());
    let data = json!({"train": {"digest": train.hex()}}).to_string();
    let body = json!({"model": trainjob::MODEL, "action": "train", "params": {"base": BASE, "data": data, "steps": 2, "patience": 2}});
    let (status, refused) = call(&app, Method::POST, "/v1/run", Some(KEY), Some(body)).await;
    let text = String::from_utf8_lossy(&refused);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("patience") && text.contains("eval_every"), "{text}");
    assert!(work_is_clean(&rig.work_root));
}

/// A full fine-tune whose precision its architecture does not run is a 400 at submission, not a job that fails once
/// queued; `validate_only` answers the settings the run would train with.
#[tokio::test(flavor = "multi_thread")]
async fn a_full_fine_tune_the_trainer_would_refuse_is_refused_at_submission() {
    let rig = rig_qwen3("http-full-check", 16 << 30);
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let train = data_digest(&rig, &plain());
    let data = json!({"train": {"digest": train.hex()}}).to_string();
    let body = |params: Value| json!({"model": trainjob::MODEL, "action": trainjob::consolidate::FULL_ACTION, "params": params});

    let (status, refused) = call(&app, Method::POST, "/v1/run", Some(KEY), Some(body(json!({"base": BASE, "data": data, "train_precision": "weights=f32"})))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{}", String::from_utf8_lossy(&refused));
    assert!(String::from_utf8_lossy(&refused).contains("weights=f32"));

    let (status, checked) = call(&app, Method::POST, "/v1/run", Some(KEY), Some(body(json!({"base": BASE, "data": data, "max_block": 128, "validate_only": true})))).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&checked));
    assert_eq!(json_of(&checked)["outputs"]["status"], "valid");
    assert!(work_is_clean(&rig.work_root));
}

/// `speech_sft` over HTTP: the shard moves by digest, the job streams its stages and ends with the checkpoint's
/// manifest digest; a param that names a path, an unknown key and a digest the store does not hold are 400s that start
/// nothing and name no path of this machine.
#[tokio::test(flavor = "multi_thread")]
async fn a_speech_fine_tune_over_http_moves_its_shard_by_digest_and_refuses_what_it_does_not_define() {
    use crate::speech::{shard, write_tts_base, LANGUAGES, TTS_BASE, VOICES};
    let rig = rig("http-speech", 16 << 30, true);
    write_tts_base(&rig.dir.join("models"));
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let digest = data_digest(&rig, &shard());
    let body = |params: Value| json!({"model": trainjob::MODEL, "action": trainjob::speech::ACTION, "params": params});
    let good = json!({"base": TTS_BASE, "languages": LANGUAGES, "voices": VOICES, "dataset_digest": digest.hex(), "steps": 1, "allow_cpu": true});

    let mut refusals = Vec::new();
    for extra in ["out", "weights_dir", "ckpt", "path", "dataset"] {
        let mut p = good.clone();
        p[extra] = json!("/etc/passwd");
        refusals.push((format!("param {extra}"), p));
    }
    let mut p = good.clone();
    p["dataset_digest"] = json!(trainjob::store::digest_of(b"absent").hex());
    refusals.push(("an absent digest".into(), p));
    let mut p = good.clone();
    p["languages"] = json!("x".repeat(trainjob::speech::MAX_NAMES_DOC + 1));
    refusals.push(("an oversized languages document".into(), p));
    for (why, p) in refusals {
        let (status, response) = call(&app, Method::POST, "/v1/jobs", Some(KEY), Some(body(p))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{why}: {}", String::from_utf8_lossy(&response));
        assert!(!String::from_utf8_lossy(&response).contains(rig.dir.to_str().unwrap()), "{why}: the refusal names a path of this machine");
    }

    let (status, created) = call(&app, Method::POST, "/v1/jobs", Some(KEY), Some(body(good))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{}", String::from_utf8_lossy(&created));
    let id = json_of(&created)["id"].as_str().unwrap().to_string();
    let (_, stream) = call(&app, Method::GET, &format!("/v1/jobs/{id}/events"), Some(KEY), None).await;
    let text = String::from_utf8(stream).unwrap();
    assert!(text.contains("\"stage\"") && text.contains("event: done"), "{text}");
    let (status, result) = call(&app, Method::GET, &format!("/v1/jobs/{id}/result"), Some(KEY), None).await;
    assert_eq!(status, StatusCode::OK);
    let result = json_of(&result);
    assert!(trainjob::Digest::parse(result["outputs"]["checkpoint_digest"].as_str().unwrap_or_default()).is_ok(), "{result}");
    assert!(!result.to_string().contains(rig.dir.to_str().unwrap()), "the result names no path of this machine: {result}");
}
