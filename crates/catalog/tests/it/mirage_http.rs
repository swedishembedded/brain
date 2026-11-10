// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for exposing perception models over
// hardened HTTP APIs for its clients. If your team needs expertise in securing
// inference endpoints then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Mirage over the generic `/v1/run` route: the boundary refuses what it must
//! (oversized or inconsistent images, malformed and oversized forms, unknown
//! params, any param that names a path) before a device is touched, and a real
//! call answers with the JSON the session produces.

use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use catalog::resident_mirage::MirageResident;
use residency::budget::Budgets;
use residency::{Device, Executor, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "Cactus-Compute/mirage";

struct Rig {
    app: Router,
    exec: Executor,
    _stub: Option<std::path::PathBuf>,
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.exec.shutdown();
        if let Some(dir) = &self._stub {
            std::fs::remove_dir_all(dir).ok();
        }
    }
}

fn serve(resident: MirageResident, stub: Option<std::path::PathBuf>) -> Rig {
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 24 << 30, 0);
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(resident)];
    let exec = Executor::start(models, budgets, Policy::default());
    Rig { app: router(AppState::new(exec.clone(), "k", Provider::OpenAI)), exec, _stub: stub }
}

/// A server whose checkpoint directory holds empty stand-ins: enough for every refusal, which happens before the
/// model is built.
fn stub_rig() -> Rig {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!("brain-mirage-http-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    for f in mirage::resolver::RELEASE_FILES {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }
    serve(MirageResident::new(dir.to_string_lossy()).unwrap(), Some(dir))
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/run")
        .header(header::AUTHORIZATION, "Bearer k")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// A `side x side` RGB f32 image as the wire blob of `/v1/run`.
fn wire_image(side: u32) -> Value {
    let bytes = vec![0u8; (side * side * 12) as usize];
    json!({"media": "image", "data": events::base64::encode(&bytes), "meta": {"w": side, "h": side, "c": 3}})
}

fn call(action: &str, params: Value, image: Value) -> Value {
    json!({"model": MODEL, "action": action, "params": params, "blobs": {"image": image}})
}

fn message(body: &Value) -> String {
    body["error"]["message"].as_str().unwrap_or("").to_string()
}

#[tokio::test]
async fn the_catalogue_lists_both_actions_and_no_path_param() {
    let rig = stub_rig();
    let req = Request::builder().uri("/v1/capabilities").header(header::AUTHORIZATION, "Bearer k").body(Body::empty()).unwrap();
    let response = rig.app.clone().oneshot(req).await.unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
    let listing: Value = serde_json::from_slice(&bytes).unwrap();
    let text = listing.to_string();
    assert!(text.contains(MODEL) && text.contains("\"describe\"") && text.contains("\"extract\""), "{text}");
    for forbidden in ["\"weights\"", "\"checkpoint\"", "\"path\"", "\"dir\""] {
        assert!(!text.contains(forbidden), "no param may name a host path: {forbidden}");
    }
}

#[tokio::test]
async fn an_image_past_the_pixel_bound_is_refused_from_its_header() {
    let rig = stub_rig();
    let huge = json!({"media": "image", "data": events::base64::encode(&[0u8; 12]), "meta": {"w": 20_000, "h": 20_000, "c": 3}});
    let (status, body) = post(&rig.app, call("describe", json!({}), huge)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(message(&body).contains("pixels"), "{body}");
}

#[tokio::test]
async fn an_image_whose_payload_disagrees_with_its_header_is_refused() {
    let rig = stub_rig();
    let lying = json!({"media": "image", "data": events::base64::encode(&[0u8; 48]), "meta": {"w": 64, "h": 64, "c": 3}});
    let (status, body) = post(&rig.app, call("describe", json!({}), lying)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(message(&body).contains("payload"), "{body}");
}

#[tokio::test]
async fn an_oversized_request_body_is_refused() {
    let rig = stub_rig();
    let blob = "A".repeat(70 << 20);
    let body = json!({"model": MODEL, "action": "describe", "params": {}, "blobs": {"image": {"media": "image", "data": blob, "meta": {"w": 2, "h": 2, "c": 3}}}});
    let (status, _) = post(&rig.app, body).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn forms_that_are_malformed_oversized_or_unsupported_are_refused_with_the_param_named() {
    let rig = stub_rig();
    let image = wire_image(8);
    let refuse = |form: String| {
        let image = image.clone();
        let app = rig.app.clone();
        async move { post(&app, call("extract", json!({"form": form}), image)).await }
    };
    let (status, body) = refuse("{not json".to_string()).await;
    assert_eq!((status, message(&body).contains("form")), (StatusCode::BAD_REQUEST, true), "{body}");
    let (status, body) = refuse("[1, 2]".to_string()).await;
    assert_eq!((status, message(&body).contains("form")), (StatusCode::BAD_REQUEST, true), "{body}");
    let big = format!("{{\"a\": {{\"enum\": [\"{}\"]}}}}", "x".repeat(mirage::caps::MAX_FORM_BYTES));
    let (status, body) = refuse(big).await;
    assert_eq!((status, message(&body).contains("bytes")), (StatusCode::BAD_REQUEST, true), "{body}");
    let fields: serde_json::Map<String, Value> = (0..=mirage::caps::MAX_FORM_FIELDS).map(|i| (format!("f{i}"), json!({"enum": ["a", "b"]}))).collect();
    let (status, body) = refuse(Value::Object(fields).to_string()).await;
    assert_eq!((status, message(&body).contains("fields")), (StatusCode::BAD_REQUEST, true), "{body}");
    let values: Vec<Value> = (0..200).map(|i| json!(format!("v{i}"))).collect();
    let bomb: serde_json::Map<String, Value> = (0..8).map(|i| (format!("f{i}"), json!({"enum": values}))).collect();
    let (status, body) = refuse(Value::Object(bomb).to_string()).await;
    assert_eq!((status, message(&body).contains("candidate")), (StatusCode::BAD_REQUEST, true), "{body}");
    let (status, body) = refuse(r#"{"b": {"type": "string", "format": "box"}}"#.to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = post(&rig.app, call("extract", json!({}), image.clone())).await;
    assert_eq!((status, message(&body).contains("form")), (StatusCode::BAD_REQUEST, true), "the form is required: {body}");
}

#[tokio::test]
async fn unknown_params_and_path_like_params_are_refused() {
    let rig = stub_rig();
    for name in ["weights", "checkpoint", "path", "dir", "model_path", "mirage", "cact", "bogus"] {
        let (status, body) = post(&rig.app, call("describe", json!({name: "/etc/passwd"}), wire_image(8))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "param {name}: {body}");
        assert!(message(&body).contains(name), "the refusal names the param {name}: {body}");
    }
    let (status, body) = post(&rig.app, call("describe", json!({"sections": "objects,bogus"}), wire_image(8))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = post(&rig.app, call("describe", json!({"max_tokens": 1_000_000}), wire_image(8))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "above the declared ceiling: {body}");
    let (status, body) = post(&rig.app, call("describe", json!({"language": "klingon"}), wire_image(8))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = post(&rig.app, call("describe", json!({"form": "{}"}), wire_image(8))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "form is not a describe param: {body}");
    let (status, _) = post(&rig.app, call("train", json!({}), wire_image(8))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "an unknown action");
}

/// The same request through HTTP and through the session gives the same JSON.
#[tokio::test]
async fn a_served_describe_and_extract_equal_the_session_answers() {
    let Some(dir) = brain_testutil::model_dir(MODEL).filter(|d| mirage::resolver::RELEASE_FILES.iter().all(|f| std::path::Path::new(d).join(f).is_file())) else {
        brain_testutil::skip("mirage http: Cactus-Compute/mirage is not in the model store");
        return;
    };
    let Some(golden) = ({
        let path = brain_testutil::testdata("golden/mirage/goldens/cats.safetensors");
        std::path::Path::new(&path).is_file().then(|| checkpoint::st::load_safetensors(&path).expect("golden loads").tensors)
    }) else {
        brain_testutil::skip("mirage http: the cats golden is absent");
        return;
    };
    let rgb: Vec<u8> = golden["input_rgb"].iter().map(|&v| v as u8).collect();
    let hwc: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
    let blob = capability::blob::image_blob(&hwc, 640, 480, 3);
    let wire = json!({"media": "image", "data": events::base64::encode(&blob.bytes), "meta": blob.meta});
    let form = r#"{"animal": {"enum": ["cat", "dog", "bird"]}}"#;

    let rig = serve(MirageResident::new(&dir).unwrap(), None);
    let (status, described) = post(&rig.app, call("describe", json!({"max_tokens": 600, "mask_format": "bitmap"}), wire.clone())).await;
    assert_eq!(status, StatusCode::OK, "{described}");
    let (status, extracted) = post(&rig.app, call("extract", json!({"form": form}), wire.clone())).await;
    assert_eq!(status, StatusCode::OK, "{extracted}");

    let session = mirage::caps::MirageSession::load(&dir, gpu_core::Gpu::new(&mirage::caps::MirageSession::kernels())).unwrap();
    let inv = capability::Invocation::new().blob("image", blob).set("max_tokens", json!(600)).set("mask_format", json!("bitmap"));
    let direct = session.run_batch("describe", std::slice::from_ref(&inv)).pop().unwrap().unwrap();
    let strip = |mut v: Value| {
        v.as_object_mut().unwrap().remove("usage");
        v
    };
    assert_eq!(strip(described["outputs"].clone()), strip(direct.outputs.clone()));
    let direct_form = session.run_batch("extract", &[capability::Invocation::new().blob("image", inv.blobs["image"].clone()).set("form", json!(form))]).pop().unwrap().unwrap();
    assert_eq!(strip(extracted["outputs"].clone()), strip(direct_form.outputs));
    assert!(described["outputs"]["usage"]["decoder_tokens"].as_u64().unwrap() > 20);
}
