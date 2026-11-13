// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `POST /v1/audio/speech` on the OpenAI and OpenRouter surfaces: a text-to-speech model is discovered from its
//! manifest alone (a `speak` action taking `text` and returning `audio`), listed as `speech`, and answers a WAV or raw
//! PCM. Everything a client can get wrong, and every field brain cannot honour, is refused before the model runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-brain-test-key";
const RATE: u32 = 24_000;

/// What the fake model says: a ramp, as raw f32 little-endian PCM like the real `speak` action.
const SAMPLES: [f32; 4] = [0.0, 0.5, -0.5, 1.0];

/// A TTS model with the contract of `qwen3tts`'s `speak`. It reports the `max_frames` it was given as the number of
/// samples' worth of ramp repeated, so a test sees the bound the route applied, and counts its runs.
struct FakeTts(Arc<AtomicUsize>);
struct FakeTtsInst(Arc<AtomicUsize>);

impl ResidentModel for FakeTts {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-tts",
            "a text-to-speech model",
            vec![ActionSpec::new("speak", "synthesize speech")
                .streaming()
                .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
                .param(ParamSpec::new("max_frames", ParamType::Int, "length cap").min(0.0).max(16384.0).default(json!(256)))
                .output(BlobSpec::new("audio", Media::Audio, "raw mono f32 LE PCM"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-tts", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeTtsInst(self.0.clone())))
    }
}

impl Instance for FakeTtsInst {
    fn run(&mut self, _a: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        let frames = inv.get_i64("max_frames").unwrap_or(0) as usize;
        let samples: Vec<f32> = SAMPLES.iter().copied().cycle().take(SAMPLES.len() * frames.clamp(1, 3)).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(Outcome::new().blob("audio", Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": RATE}))))
    }
}

struct Chat;
impl ResidentModel for Chat {
    fn manifest(&self) -> Manifest {
        Manifest::new("brain-chat", "a chat model", vec![ActionSpec::new("generate", "generate").streaming().param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required()).output(BlobSpec::new("text", Media::Text, "text"))])
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

fn app(provider: Provider) -> (Router, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeTts(runs.clone())), Arc::new(Chat)];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    (router(AppState::new(Executor::start(models, budgets, Policy::default()), KEY.to_string(), provider)), runs)
}

async fn post(app: &Router, uri: &str, body: &Value) -> (StatusCode, Vec<(String, String)>, Vec<u8>) {
    let request = Request::builder().method("POST").uri(uri).header(header::AUTHORIZATION, format!("Bearer {KEY}")).header(header::CONTENT_TYPE, "application/json").body(Body::from(body.to_string())).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().iter().map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string())).collect();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap().to_vec();
    (status, headers, bytes)
}

fn header_of<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

#[tokio::test]
async fn speech_is_a_wav_by_default_and_raw_pcm_on_request_at_the_models_own_rate() {
    for (provider, uri) in [(Provider::OpenAI, "/v1/audio/speech"), (Provider::OpenRouter, "/audio/speech")] {
        let (app, runs) = app(provider);
        let (status, headers, body) = post(&app, uri, &json!({"model": "brain-tts", "input": "hello"})).await;
        assert_eq!(status, StatusCode::OK, "{provider:?}");
        assert_eq!((header_of(&headers, "content-type"), header_of(&headers, "x-sample-rate")), (Some("audio/wav"), Some("24000")));
        let wav = audio::wav::parse(&body).expect("a WAV");
        assert_eq!(wav.sample_rate, RATE);
        assert_eq!(wav.samples.len(), SAMPLES.len() * 3, "the route capped the frames the model was asked for");
        for (got, want) in wav.samples.iter().zip(SAMPLES.iter().cycle()) {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }

        let (status, headers, pcm) = post(&app, uri, &json!({"model": "brain-tts", "input": "hello", "response_format": "pcm"})).await;
        assert_eq!((status, header_of(&headers, "content-type")), (StatusCode::OK, Some("audio/pcm")));
        let first: Vec<i16> = pcm.as_chunks::<2>().0.iter().take(4).map(|c| i16::from_le_bytes(*c)).collect();
        assert_eq!(first, vec![0, 16384, -16384, 32767], "16-bit signed little-endian, clamped");
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }
}

/// Every refusal happens before the model is asked: the run count stays at zero.
#[tokio::test]
async fn a_speech_request_the_server_cannot_honour_is_refused_by_name_before_any_run() {
    let (app, runs) = app(Provider::OpenAI);
    let long = "x".repeat(apiserve::speech::MAX_INPUT_CHARS + 1);
    let cases: Vec<(Value, StatusCode, &str)> = vec![
        (json!({"model": "brain-tts", "input": "hi", "voice": "alloy"}), StatusCode::BAD_REQUEST, "voice"),
        (json!({"model": "brain-tts", "input": "hi", "speed": 1.5}), StatusCode::BAD_REQUEST, "speed"),
        (json!({"model": "brain-tts", "input": "hi", "instructions": "calm"}), StatusCode::BAD_REQUEST, "instructions"),
        (json!({"model": "brain-tts", "input": "hi", "response_format": "mp3"}), StatusCode::BAD_REQUEST, "mp3"),
        (json!({"model": "brain-tts", "input": "hi", "path": "/etc/passwd"}), StatusCode::BAD_REQUEST, "path"),
        (json!({"model": "brain-tts", "input": ""}), StatusCode::BAD_REQUEST, "input"),
        (json!({"model": "brain-tts", "input": long}), StatusCode::BAD_REQUEST, "input"),
        (json!({"model": "brain-tts"}), StatusCode::BAD_REQUEST, "input"),
        (json!({"input": "hi"}), StatusCode::BAD_REQUEST, "model"),
        (json!({"model": "brain-chat", "input": "hi"}), StatusCode::NOT_FOUND, ""),
        (json!({"model": "nobody", "input": "hi"}), StatusCode::NOT_FOUND, ""),
    ];
    for (body, expected, mention) in cases {
        let (status, _, reply) = post(&app, "/v1/audio/speech", &body).await;
        assert_eq!(status, expected, "{body}");
        assert!(String::from_utf8_lossy(&reply).contains(mention), "{body}: {}", String::from_utf8_lossy(&reply));
    }
    let request = Request::builder().method("POST").uri("/v1/audio/speech").header(header::AUTHORIZATION, format!("Bearer {KEY}")).body(Body::from("{ not json")).unwrap();
    assert_eq!(app.clone().oneshot(request).await.unwrap().status(), StatusCode::BAD_REQUEST);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn the_speech_route_is_behind_the_key_and_the_model_is_listed_as_speech() {
    let (app, _) = app(Provider::OpenAI);
    for auth in [None, Some("Bearer wrong")] {
        let mut request = Request::builder().method("POST").uri("/v1/audio/speech").header(header::CONTENT_TYPE, "application/json");
        if let Some(auth) = auth {
            request = request.header(header::AUTHORIZATION, auth);
        }
        let response = app.clone().oneshot(request.body(Body::from(json!({"model": "brain-tts", "input": "hi"}).to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    let request = Request::builder().uri("/v1/models").header(header::AUTHORIZATION, format!("Bearer {KEY}")).body(Body::empty()).unwrap();
    let body = axum::body::to_bytes(app.clone().oneshot(request).await.unwrap().into_body(), 1 << 20).await.unwrap();
    let listed: Value = serde_json::from_slice(&body).unwrap();
    let tts = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-tts").expect("the speech model is listed");
    assert_eq!(tts["capabilities"], json!(["speech"]));
}
