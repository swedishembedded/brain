// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving speech synthesis behind
// standard inference APIs for its clients. If your team needs expertise in
// production text-to-speech services then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `POST /audio/speech` (with and without `/v1`): text to speech on the OpenAI and OpenRouter surfaces, the
//! counterpart of `/audio/transcriptions`.
//!
//! The body is `{"model", "input", "response_format"?}`, and nothing else: a field brain cannot honour (`voice`,
//! `speed`, `instructions`, a streaming format) is a 400 naming it, never ignored. `response_format` is `wav` (the
//! default: a 16-bit mono WAV) or `pcm` (raw 16-bit signed little-endian mono at the model's rate, as OpenAI's
//! `pcm`); a compressed format is a 400, since brain writes none. The model is the one whose manifest has a `speak`
//! action that takes `text` and returns an `audio` blob ([`crate::catalog::resolve_speech`]); the call is held to that
//! action's own contract before any job exists, and the reply's length is bounded by the action's frame cap.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use capability::{Blob, Invocation};
use serde::Deserialize;
use serde_json::json;

use crate::auth::CallerExt;
use crate::bridge;
use crate::catalog;
use crate::error::ApiError;
use crate::state::AppState;

/// The longest `input` accepted, in characters (OpenAI's own bound).
pub const MAX_INPUT_CHARS: usize = 4096;
/// The most codec frames one request may ask a model for: about a minute and a half of speech, so a reply is
/// megabytes, never hundreds of them, whatever the model's own ceiling.
pub const MAX_FRAMES: i64 = 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechRequest {
    model: String,
    input: String,
    #[serde(default)]
    response_format: Option<String>,
}

/// What a client asked for in the reply's container.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Wav,
    Pcm,
}

/// Fields OpenAI defines that brain cannot honour, refused by name before the body is read as a request.
const UNSUPPORTED: [&str; 4] = ["voice", "speed", "instructions", "stream_format"];

fn parse(provider: crate::surface::Provider, body: &Bytes) -> Result<(SpeechRequest, Format), ApiError> {
    let invalid = |m: String| ApiError::invalid_request(provider, m);
    let value: serde_json::Value = serde_json::from_slice(body).map_err(|e| invalid(format!("invalid JSON body: {e}")))?;
    if let Some(name) = UNSUPPORTED.iter().find(|n| value.get(**n).is_some_and(|v| !v.is_null())) {
        return Err(invalid(format!("'{name}' is not supported: brain's speech models take the text and nothing that steers the voice")));
    }
    let req: SpeechRequest = serde_json::from_value(value).map_err(|e| invalid(format!("invalid request: {e}")))?;
    if req.input.trim().is_empty() || req.input.chars().count() > MAX_INPUT_CHARS {
        return Err(invalid(format!("'input' must be 1 to {MAX_INPUT_CHARS} characters")));
    }
    let format = match req.response_format.as_deref() {
        None | Some("wav") => Format::Wav,
        Some("pcm") => Format::Pcm,
        Some(other) => return Err(invalid(format!("'response_format' {other:?} is not supported: brain writes wav and pcm"))),
    };
    Ok((req, format))
}

/// The mono samples and rate of a model's `audio` blob: a WAV container, or raw f32 little-endian PCM at
/// `meta.sample_rate`.
fn samples_of(blob: &Blob) -> Result<(Vec<f32>, u32), String> {
    if blob.bytes.starts_with(b"RIFF") {
        let wav = audio::wav::parse(&blob.bytes).map_err(|e| e.to_string())?;
        return Ok((wav.samples, wav.sample_rate));
    }
    let rate = blob.meta.get("sample_rate").and_then(|v| v.as_u64()).filter(|r| (8_000..=192_000).contains(r)).ok_or("the model's audio carries no usable sample_rate")? as u32;
    if !blob.bytes.len().is_multiple_of(4) {
        return Err("the model's audio is not whole f32 samples".to_string());
    }
    Ok((blob.bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(), rate))
}

/// `POST /audio/speech`.
pub async fn speech(State(state): State<AppState>, caller: CallerExt, body: Bytes) -> Response {
    let state = state.scoped(caller);
    let provider = state.provider;
    let (req, format) = match parse(provider, &body) {
        Ok(parsed) => parsed,
        Err(e) => return e.into_response(),
    };
    let body_model = json!({ "model": req.model });
    let manifests = state.exec.manifests();
    let (model, action) = match catalog::resolve_model(provider, &body_model, |id| catalog::resolve_speech(&manifests, id)) {
        Some((id, action)) => (id, action),
        None => match bridge::ensure_and_recheck(&state, provider, &req.model, |id| catalog::resolve_speech(&state.exec.manifests(), id)).await {
            Ok(action) => (req.model.clone(), action),
            Err(e) => return e.into_response(),
        },
    };
    let manifests = state.exec.manifests();
    let Some(spec) = manifests.iter().find(|m| m.model == model).and_then(|m| m.actions.iter().find(|a| a.name == action)) else {
        return ApiError::model_not_found(provider, &model).into_response();
    };
    let mut call = Invocation::new().set("text", json!(req.input));
    if spec.params.iter().any(|p| p.name == "max_frames") {
        call = call.set("max_frames", json!(MAX_FRAMES));
    }
    let call = match spec.validate_served(call) {
        Ok(call) => call,
        Err(e) => return ApiError::invalid_request(provider, e).into_response(),
    };
    let outcome = match bridge::submit(&state, &model, &action, call).await {
        Ok(o) => o,
        Err(e) => return e.into_response(),
    };
    let Some(blob) = outcome.blobs.get("audio") else {
        return ApiError::invalid_request(provider, "the model returned no audio").into_response();
    };
    let (samples, rate) = match samples_of(blob) {
        Ok(decoded) => decoded,
        Err(e) => return ApiError::invalid_request(provider, e).into_response(),
    };
    let (bytes, content_type) = match format {
        Format::Wav => (audio::wav::encode(&samples, rate), "audio/wav"),
        Format::Pcm => (samples.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect::<Vec<u8>>(), "audio/pcm"),
    };
    let mut response = (StatusCode::OK, bytes).into_response();
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    // A client reading raw PCM needs the rate: it is the model's, not a constant.
    if let Ok(value) = HeaderValue::from_str(&rate.to_string()) {
        response.headers_mut().insert("x-sample-rate", value);
    }
    response
}
