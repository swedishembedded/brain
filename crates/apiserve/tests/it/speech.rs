// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `POST /v1/audio/speech` on the OpenAI and OpenRouter surfaces: a text-to-speech model is discovered from its
//! manifest alone (an action declaring a speech contract and returning `audio`), listed as `speech`, and answers a WAV
//! or raw PCM. A reference voice, an instruction, a preset voice and a seed go to the action whose contract takes
//! them. Everything a client can get wrong, and every field brain cannot honour, is refused before the model runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use capability::speech::{SpeechContract, Transcript};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-brain-test-key";
const RATE: u32 = 24_000;

/// What the fake model says: a ramp, as raw f32 little-endian PCM like the real `speak` action.
const SAMPLES: [f32; 4] = [0.0, 0.5, -0.5, 1.0];

/// A TTS model with a plain speech contract (text and a length cap). It reports the `max_frames` it was given as the number of
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
                .speech(SpeechContract::new("text").length("max_frames"))
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

/// What a [`Voices`] action was handed: its name, its params, and each input blob's bytes.
type Seen = Arc<Mutex<Vec<(String, Value, Vec<(String, Vec<u8>)>)>>>;

/// A model shaped like Qwen3-TTS (`synth`, `clone` with an optional transcript, `design` with an instruction and a
/// preset voice) or, with `transcript_required`, like CosyVoice (one `synth` that clones and needs the transcript).
struct Voices {
    id: &'static str,
    transcript_required: bool,
    seen: Seen,
}
struct VoicesInst(Seen);

fn speech_action(name: &str) -> ActionSpec {
    ActionSpec::new(name, name)
        .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
        .param(ParamSpec::new("seed", ParamType::Int, "seed").min(0.0))
        .param(ParamSpec::new("lang", ParamType::Str, "language"))
        .output(BlobSpec::new("audio", Media::Audio, "a WAV"))
}

impl ResidentModel for Voices {
    fn manifest(&self) -> Manifest {
        let base = || SpeechContract::new("text").seed("seed").language("lang");
        let actions = if self.transcript_required {
            vec![speech_action("synth")
                .param(ParamSpec::new("ref_text", ParamType::Str, "transcript").required())
                .input(BlobSpec::new("ref_audio", Media::Audio, "the voice").required())
                .speech(base().reference("ref_audio", Transcript::Required("ref_text".into())))]
        } else {
            vec![
                speech_action("synth").speech(base()),
                speech_action("clone")
                    .param(ParamSpec::new("ref_text", ParamType::Str, "transcript").default(json!("")))
                    .input(BlobSpec::new("ref", Media::Bytes, "the voice").required())
                    .speech(base().reference("ref", Transcript::Optional("ref_text".into()))),
                speech_action("design")
                    .param(ParamSpec::new("instruct", ParamType::Str, "description").default(json!("")))
                    .param(ParamSpec::new("speaker", ParamType::Str, "preset").default(json!("")))
                    .speech(base().instructions("instruct").voice("speaker")),
            ]
        };
        Manifest::new(self.id, "voices", actions)
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(VoicesInst(self.seen.clone())))
    }
}

impl Instance for VoicesInst {
    fn run(&mut self, action: &str, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let blobs = ["ref", "ref_audio"].iter().filter_map(|k| inv.get_blob(k).map(|b| (k.to_string(), b.bytes.clone()))).collect();
        self.0.lock().unwrap().push((action.to_string(), inv.without_blobs().params.clone(), blobs));
        let seed = inv.get_i64("seed").unwrap_or(77);
        Ok(Outcome::new().set("seed", json!(seed)).blob("audio", Blob::new(Media::Audio, audio::wav::encode(&SAMPLES, RATE))))
    }
}

struct Chat;
impl ResidentModel for Chat {
    fn manifest(&self) -> Manifest {
        Manifest::new("brain-chat", "a chat model", vec![crate::chat_contract(ActionSpec::new("generate", "generate").streaming().param(ParamSpec::new("prompt", ParamType::Str, "the prompt")).output(BlobSpec::new("text", Media::Text, "text")))])
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
    let (router, runs, _) = app_with_voices(provider);
    (router, runs)
}

fn app_with_voices(provider: Provider) -> (Router, Arc<AtomicUsize>, Seen) {
    let runs = Arc::new(AtomicUsize::new(0));
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let models: Vec<Arc<dyn ResidentModel>> = vec![
        Arc::new(FakeTts(runs.clone())),
        Arc::new(Voices { id: "brain-voices", transcript_required: false, seen: seen.clone() }),
        Arc::new(Voices { id: "brain-cloner", transcript_required: true, seen: seen.clone() }),
        Arc::new(Chat),
    ];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    (router(AppState::new(Executor::start(models, budgets, Policy::default()), KEY.to_string(), provider)), runs, seen)
}

/// A second of a 10 kHz whistle at 48 kHz, as a WAV data URL: content a 16 kHz copy would lose.
fn reference_wav() -> (Vec<f32>, String) {
    let whistle: Vec<f32> = (0..48_000).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 10_000.0 * i as f32 / 48_000.0).sin()).collect();
    let url = format!("data:audio/wav;base64,{}", b64(&audio::wav::encode(&whistle, 48_000)));
    (whistle, url)
}

fn b64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (*b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { A[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
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
    let (app, _, _) = app_with_voices(Provider::OpenAI);
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
    assert_eq!(tts["speech"]["response_formats"], json!(["wav", "pcm"]));
    assert_eq!(tts["speech"]["modes"], json!([{"reference_voice": false, "reference_transcript": "none", "instructions": false, "preset_voices": false, "seed": false, "language": false}]));
    let cloner = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-cloner").expect("listed");
    assert_eq!(cloner["speech"]["modes"][0]["reference_transcript"], "required", "a client can tell the transcript is needed");
    let chat = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-chat").unwrap();
    assert!(chat.get("speech").is_none(), "a model that does not speak says nothing about speech");

    let (app, _, _) = app_with_voices(Provider::OpenRouter);
    let request = Request::builder().uri("/v1/models").header(header::AUTHORIZATION, format!("Bearer {KEY}")).body(Body::empty()).unwrap();
    let body = axum::body::to_bytes(app.clone().oneshot(request).await.unwrap().into_body(), 1 << 20).await.unwrap();
    let listed: Value = serde_json::from_slice(&body).unwrap();
    let tts = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-voices").expect("listed on OpenRouter");
    assert_eq!(tts["architecture"]["output_modalities"], json!(["speech"]), "OpenRouter's name for a text-to-speech model");
    assert_eq!(tts["architecture"]["modality"], "text+audio->speech", "it reads a recorded voice");
    let plain = listed["data"].as_array().unwrap().iter().find(|m| m["id"] == "brain-tts").unwrap();
    assert_eq!(plain["architecture"]["modality"], "text->speech");
    assert_eq!(tts["speech"]["modes"].as_array().map(Vec::len), Some(3));
}

/// Each request goes to the narrowest action whose contract takes everything it carries, with every field in the
/// action's own param; a reference recording arrives as a WAV at its own rate, and the seed used comes back.
#[tokio::test]
async fn a_speech_request_is_routed_by_what_it_steers_and_hands_the_reference_over_whole() {
    let (app, _, seen) = app_with_voices(Provider::OpenRouter);
    let (whistle, url) = reference_wav();
    let take = |seen: &Seen| seen.lock().unwrap().pop().expect("the model ran");

    let (status, headers, _) = post(&app, "/v1/audio/speech", &json!({"model": "brain-voices", "input": "plain", "seed": 5})).await;
    assert_eq!((status, header_of(&headers, "x-seed")), (StatusCode::OK, Some("5")));
    let (action, params, _) = take(&seen);
    assert_eq!((action.as_str(), &params["text"], &params["seed"]), ("synth", &json!("plain"), &json!(5)));

    let body = json!({"model": "brain-voices", "input": "cloned", "input_references": [
        {"type": "input_audio", "input_audio": {"data": url}}, {"type": "text", "text": "what it says"}]});
    assert_eq!(post(&app, "/v1/audio/speech", &body).await.0, StatusCode::OK);
    let (action, params, blobs) = take(&seen);
    assert_eq!((action.as_str(), &params["ref_text"]), ("clone", &json!("what it says")));
    let (name, bytes) = &blobs[0];
    let wav = audio::wav::parse(bytes).expect("the reference arrives as a WAV");
    assert_eq!((name.as_str(), wav.sample_rate, wav.samples.len()), ("ref", 48_000, whistle.len()), "kept at its own rate and length");

    let body = json!({"model": "brain-voices", "input": "x", "input_references": [{"type": "input_audio", "input_audio": {"data": url}}]});
    assert_eq!(post(&app, "/v1/audio/speech", &body).await.0, StatusCode::OK);
    assert_eq!(take(&seen).0, "clone", "an optional transcript may be left out");

    assert_eq!(post(&app, "/v1/audio/speech", &json!({"model": "brain-voices", "input": "x", "instructions": "a tired, whispering voice"})).await.0, StatusCode::OK);
    let (action, params, _) = take(&seen);
    assert_eq!((action.as_str(), &params["instruct"]), ("design", &json!("a tired, whispering voice")));

    assert_eq!(post(&app, "/v1/audio/speech", &json!({"model": "brain-voices", "input": "x", "voice": "ryan"})).await.0, StatusCode::OK);
    let (action, params, _) = take(&seen);
    assert_eq!((action.as_str(), &params["speaker"]), ("design", &json!("ryan")));

    let body = json!({"model": "brain-voices", "input": "hej", "voice": "astrid", "language": "swedish"});
    assert_eq!(post(&app, "/v1/audio/speech", &body).await.0, StatusCode::OK);
    let (action, params, _) = take(&seen);
    assert_eq!((action.as_str(), &params["speaker"], &params["lang"]), ("design", &json!("astrid"), &json!("swedish")), "the language goes to the action's own param");

    let body = json!({"model": "brain-cloner", "input": "x", "input_references": [
        {"type": "input_audio", "input_audio": {"data": url}}, {"type": "text", "text": "what it says"}]});
    assert_eq!(post(&app, "/v1/audio/speech", &body).await.0, StatusCode::OK);
    let (action, params, blobs) = take(&seen);
    assert_eq!((action.as_str(), &params["ref_text"], blobs[0].0.as_str()), ("synth", &json!("what it says"), "ref_audio"));
}

/// Steering a model cannot take is refused naming the field, before anything runs.
#[tokio::test]
async fn steering_a_model_cannot_take_is_refused_by_name_before_any_run() {
    let (app, _, seen) = app_with_voices(Provider::OpenAI);
    let (_, url) = reference_wav();
    let audio_part = json!({"type": "input_audio", "input_audio": {"data": url}});
    let text_part = json!({"type": "text", "text": "what it says"});
    let long: Vec<f32> = vec![0.0; 16_000 * (apiserve::speech::MAX_REFERENCE_SECONDS as usize + 1)];
    let long_part = json!({"type": "input_audio", "input_audio": {"data": b64(&audio::wav::encode(&long, 16_000))}});
    let cases: Vec<(Value, &str)> = vec![
        (json!({"model": "brain-tts", "input": "hi", "voice": "alloy"}), "voice"),
        (json!({"model": "brain-tts", "input": "hi", "seed": 1}), "seed"),
        (json!({"model": "brain-tts", "input": "hi", "language": "swedish"}), "language"),
        (json!({"model": "brain-voices", "input": "hi", "language": "x".repeat(apiserve::speech::MAX_LANGUAGE_CHARS + 1)}), "language"),
        (json!({"model": "brain-tts", "input": "hi", "input_references": [audio_part.clone()]}), "input_references"),
        (json!({"model": "brain-cloner", "input": "hi"}), "input_references"),
        (json!({"model": "brain-cloner", "input": "hi", "input_references": [audio_part.clone()]}), "transcript"),
        (json!({"model": "brain-cloner", "input": "hi", "instructions": "calm", "input_references": [audio_part.clone(), text_part.clone()]}), "instructions"),
        (json!({"model": "brain-voices", "input": "hi", "voice": {"id": "voice_1"}}), "voice"),
        (json!({"model": "brain-voices", "input": "hi", "voice": 7}), "voice"),
        (json!({"model": "brain-voices", "input": "hi", "input_references": [audio_part.clone(), audio_part.clone()]}), "more than one input_audio"),
        (json!({"model": "brain-voices", "input": "hi", "input_references": [text_part.clone()]}), "transcript"),
        (json!({"model": "brain-voices", "input": "hi", "input_references": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}), "image_url"),
        (json!({"model": "brain-voices", "input": "hi", "input_references": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}}]}), "mp3"),
        (json!({"model": "brain-voices", "input": "hi", "input_references": [{"type": "input_audio", "input_audio": {"data": "bm90IGF1ZGlv"}}]}), "input_audio"),
        (json!({"model": "brain-voices", "input": "hi", "input_references": [long_part]}), "longer than"),
        (json!({"model": "brain-voices", "input": "hi", "instructions": ""}), "instructions"),
    ];
    for (body, mention) in cases {
        let (status, _, reply) = post(&app, "/v1/audio/speech", &body).await;
        let reply = String::from_utf8_lossy(&reply);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {reply}");
        assert!(reply.contains(mention), "{body}: {reply}");
    }
    assert!(seen.lock().unwrap().is_empty(), "a refused request ran the model");
}
