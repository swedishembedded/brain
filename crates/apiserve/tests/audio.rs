// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `POST /v1/audio/transcriptions` on the OpenAI and OpenRouter surfaces: a
//! speech-to-text model is discovered from its manifest alone, listed with
//! audio-in/text-out modalities, and transcribes an uploaded recording sent
//! either as an OpenAI multipart form or as OpenRouter's JSON `input_audio`.
//! Everything a client can get wrong is refused before the model runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-brain-test-key";
const BOUNDARY: &str = "brain-test-boundary";

/// An ASR-only model built on the shared `transcribe` contract. It answers
/// with the duration it received, so a test sees exactly what was decoded,
/// and counts its runs: a refused request must leave the count alone.
struct FakeAsr(Arc<AtomicUsize>);
struct FakeAsrInst(Arc<AtomicUsize>);
impl ResidentModel for FakeAsr {
    fn manifest(&self) -> Manifest {
        Manifest::new("brain-asr", "a speech-to-text model", vec![audio::asr_caps::transcribe_spec()])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-asr", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeAsrInst(self.0.clone())))
    }
}
impl Instance for FakeAsrInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        let samples = audio::asr_caps::wav_from_blob(inv.get_blob("audio").ok_or("no audio")?)?;
        Ok(audio::asr_caps::transcription_outcome(format!("heard {} ms", samples.len() / 16), &[1, 2]))
    }
}

/// A chat model, so "transcribing with a chat model" has something to refuse.
struct Chat;
impl ResidentModel for Chat {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-chat",
            "a chat model",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-chat", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Err("never activated".into())
    }
}

fn app_counting(provider: Provider) -> (Router, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeAsr(runs.clone())), Arc::new(Chat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    (router(AppState::new(Executor::start(models, budgets, Policy::default()), KEY.to_string(), provider)), runs)
}

fn app(provider: Provider) -> Router {
    app_counting(provider).0
}

/// `seconds` of a quiet tone as a 16-bit WAV at `rate`.
fn wav(seconds: f32, rate: u32) -> Vec<u8> {
    let n = (seconds * rate as f32) as usize;
    let samples: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
    audio::wav::encode(&samples, rate)
}

fn multipart(fields: &[(&str, &str)], file: Option<&[u8]>) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").into_bytes());
    }
    if let Some(bytes) = file {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip\"\r\nContent-Type: application/octet-stream\r\n\r\n").into_bytes());
        body.extend_from_slice(bytes);
        body.extend(b"\r\n");
    }
    body.extend(format!("--{BOUNDARY}--\r\n").into_bytes());
    body
}

async fn send(app: &Router, uri: &str, content_type: &str, body: Vec<u8>, key: Option<&str>) -> (StatusCode, Value) {
    let mut req = Request::post(uri).header(header::CONTENT_TYPE, content_type);
    if let Some(key) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let resp = app.clone().oneshot(req.body(Body::from(body)).unwrap()).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn upload(app: &Router, fields: &[(&str, &str)], file: Option<&[u8]>) -> (StatusCode, Value) {
    let content_type = format!("multipart/form-data; boundary={BOUNDARY}");
    send(app, "/v1/audio/transcriptions", &content_type, multipart(fields, file), Some(KEY)).await
}

async fn models(app: &Router, provider: Provider) -> Value {
    let req = match provider {
        Provider::Anthropic => Request::get("/v1/models").header("x-api-key", KEY),
        _ => Request::get("/v1/models").header(header::AUTHORIZATION, format!("Bearer {KEY}")),
    };
    let req = req.body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap()
}

fn card<'a>(list: &'a Value, id: &str) -> Option<&'a Value> {
    list["data"].as_array().unwrap().iter().find(|m| m["id"] == id)
}

#[tokio::test]
async fn a_speech_model_is_listed_from_its_manifest_alone() {
    let list = models(&app(Provider::OpenAI), Provider::OpenAI).await;
    let asr = card(&list, "brain-asr").expect("an ASR-only model must be listed");
    assert_eq!(asr["capabilities"], json!(["transcription"]));

    let list = models(&app(Provider::OpenRouter), Provider::OpenRouter).await;
    let asr = card(&list, "brain-asr").expect("listed on OpenRouter too");
    assert_eq!(asr["architecture"]["input_modalities"], json!(["audio"]));
    assert_eq!(asr["architecture"]["output_modalities"], json!(["text"]));
    assert_eq!(asr["architecture"]["modality"], "audio->text");
    let chat = card(&list, "brain-chat").unwrap();
    assert_eq!(chat["architecture"]["modality"], "text->text");

    // The Anthropic surface is chat-only.
    let list = models(&app(Provider::Anthropic), Provider::Anthropic).await;
    assert!(card(&list, "brain-asr").is_none());
}

#[tokio::test]
async fn a_multipart_upload_is_transcribed_on_both_surfaces() {
    for provider in [Provider::OpenAI, Provider::OpenRouter] {
        // A 48 kHz WAV: decoded and resampled to 16 kHz first.
        let (status, body) = upload(&app(provider), &[("model", "brain-asr"), ("language", "en")], Some(&wav(1.5, 48_000))).await;
        assert_eq!(status, StatusCode::OK, "{provider}: {body}");
        assert_eq!(body["text"], "heard 1500 ms");
        assert_eq!(body["usage"], json!({ "type": "duration", "seconds": 1.5 }));
    }
}

#[tokio::test]
async fn openrouter_json_input_audio_is_transcribed() {
    let body = json!({ "model": "brain-asr", "input_audio": { "data": events::base64::encode(&wav(2.0, 16_000)), "format": "wav" } });
    for uri in ["/v1/audio/transcriptions", "/audio/transcriptions"] {
        let (status, out) = send(&app(Provider::OpenRouter), uri, "application/json", serde_json::to_vec(&body).unwrap(), Some(KEY)).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {out}");
        assert_eq!(out["text"], "heard 2000 ms");
    }
}

#[tokio::test]
async fn everything_a_client_can_get_wrong_is_refused_before_the_model_runs() {
    let (app, runs) = app_counting(Provider::OpenAI);
    let clip = wav(1.0, 16_000);
    let cases: Vec<(&str, StatusCode, (StatusCode, Value))> = vec![
        ("not audio", StatusCode::BAD_REQUEST, upload(&app, &[("model", "brain-asr")], Some(b"ID3 an mp3 file")).await),
        ("no file", StatusCode::BAD_REQUEST, upload(&app, &[("model", "brain-asr")], None).await),
        ("no model", StatusCode::BAD_REQUEST, upload(&app, &[], Some(&clip)).await),
        ("empty audio", StatusCode::BAD_REQUEST, upload(&app, &[("model", "brain-asr")], Some(&wav(0.0, 16_000))).await),
        ("too long", StatusCode::BAD_REQUEST, upload(&app, &[("model", "brain-asr")], Some(&wav(301.0, 8_000))).await),
        ("bad language", StatusCode::BAD_REQUEST, upload(&app, &[("model", "brain-asr"), ("language", "english please")], Some(&clip)).await),
        ("unsupported format", StatusCode::BAD_REQUEST, upload(&app, &[("model", "brain-asr"), ("response_format", "srt")], Some(&clip)).await),
        ("unknown model", StatusCode::NOT_FOUND, upload(&app, &[("model", "nope")], Some(&clip)).await),
        ("chat model", StatusCode::NOT_FOUND, upload(&app, &[("model", "brain-chat")], Some(&clip)).await),
        ("not a form or JSON", StatusCode::BAD_REQUEST, send(&app, "/v1/audio/transcriptions", "text/plain", clip.clone(), Some(KEY)).await),
        (
            "no key",
            StatusCode::UNAUTHORIZED,
            send(&app, "/v1/audio/transcriptions", &format!("multipart/form-data; boundary={BOUNDARY}"), multipart(&[("model", "brain-asr")], Some(&clip)), None).await,
        ),
    ];
    for (what, expected, (status, body)) in cases {
        assert_eq!(status, expected, "{what}: {body}");
    }
    let (_, body) = upload(&app, &[("model", "brain-asr")], Some(b"ID3 an mp3 file")).await;
    assert!(body["error"]["message"].as_str().unwrap().contains("wav, ogg (opus), webm (opus)"), "{body}");
    assert_eq!(runs.load(Ordering::SeqCst), 0, "a refused request must never reach the model");
}
