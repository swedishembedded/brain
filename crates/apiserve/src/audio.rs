// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving speech recognition
// behind standard inference APIs for its clients. If your team needs expertise
// in production speech-to-text services then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `POST /audio/transcriptions` (with and without `/v1`): speech to text on the
//! OpenAI and OpenRouter surfaces, one handler for both.
//!
//! Two request shapes are accepted, the two the providers document:
//! - an OpenAI multipart form: `file` (the recording), `model`, and the
//!   optional `language`, `response_format`, `temperature` and `prompt`;
//! - OpenRouter's JSON: `{"model", "input_audio": {"data": base64, "format"},
//!   "language"?}`.
//!
//! The recording's container is taken from its bytes (`audio::decode`), never
//! from `format` or a filename, and decoded to 16 kHz mono before the model is
//! resolved or scheduled, so a malformed, unsupported or over-long upload is a
//! 400 that costs no model time. The answer is `{"text", "usage": {"type":
//! "duration", "seconds"}}`, the duration being what was actually decoded.

use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use capability::Invocation;
use serde_json::{json, Value};

use crate::auth::CallerExt;
use crate::bridge;
use crate::catalog;
use crate::error::ApiError;
use crate::state::AppState;
use crate::surface::Provider;

/// The longest recording one request may carry. Long enough for a dictated
/// paragraph or a voice note; anything longer is split by the client (the
/// providers' own upstreams time out at about a minute of work).
pub const MAX_TRANSCRIPTION_SECONDS: u32 = 300;

/// One transcription request, whichever shape it arrived in.
struct TranscriptionRequest {
    model: String,
    audio: Vec<u8>,
    language: Option<String>,
}

/// `POST /audio/transcriptions`.
pub async fn transcriptions(State(state): State<AppState>, caller: CallerExt, request: Request) -> Response {
    let state = state.scoped(caller);
    let provider = state.provider;
    let mut req = match read_request(provider, request).await {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let recording = std::mem::take(&mut req.audio);
    let decoded = crate::preprocess::off_runtime(provider, move || audio::decode::decode_speech(&recording, MAX_TRANSCRIPTION_SECONDS)).await;
    let samples = match decoded {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return ApiError::invalid_request(provider, e.to_string()).into_response(),
        Err(e) => return e.into_response(),
    };

    let body = json!({ "model": req.model });
    let manifests = state.exec.manifests();
    let (model, takes_language) = match catalog::resolve_model(provider, &body, |id| catalog::resolve_transcription(&manifests, id)) {
        Some(found) => found,
        None => match bridge::ensure_and_recheck(&state, provider, &req.model, |id| catalog::resolve_transcription(&state.exec.manifests(), id)).await {
            Ok(takes_language) => (req.model.clone(), takes_language),
            Err(e) => return e.into_response(),
        },
    };

    let mut inv = Invocation::new().blob("audio", audio::asr_caps::audio_blob_from_samples(&samples));
    if let (true, Some(language)) = (takes_language, &req.language) {
        inv = inv.set("language", json!(language));
    }
    let outcome = match bridge::submit(&state, &model, "transcribe", inv).await {
        Ok(o) => o,
        Err(e) => return e.into_response(),
    };
    let text = outcome.outputs.get("text").and_then(Value::as_str).unwrap_or_default();
    let seconds = samples.len() as f64 / audio::asr_caps::ASR_SAMPLE_RATE as f64;
    Json(json!({ "text": text, "usage": { "type": "duration", "seconds": seconds } })).into_response()
}

async fn read_request(provider: Provider, request: Request) -> Result<TranscriptionRequest, ApiError> {
    let content_type = request.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or_default().to_ascii_lowercase();
    let invalid = |m: String| ApiError::invalid_request(provider, m);
    if content_type.starts_with("multipart/form-data") {
        let form = Multipart::from_request(request, &()).await.map_err(|e| invalid(format!("invalid multipart body: {e}")))?;
        read_form(provider, form).await
    } else if content_type.starts_with("application/json") {
        let bytes = Bytes::from_request(request, &()).await.map_err(|e| invalid(format!("unreadable body: {e}")))?;
        let body: Value = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("invalid JSON body: {e}")))?;
        read_json(provider, &body)
    } else {
        Err(invalid("send the recording as multipart/form-data (file, model) or as JSON (model, input_audio)".into()))
    }
}

async fn read_form(provider: Provider, mut form: Multipart) -> Result<TranscriptionRequest, ApiError> {
    let invalid = |m: String| ApiError::invalid_request(provider, m);
    let (mut model, mut audio, mut language, mut format) = (None, None, None, None);
    while let Some(field) = form.next_field().await.map_err(|e| invalid(format!("invalid multipart body: {e}")))? {
        let name = field.name().unwrap_or_default().to_string();
        let value = field.bytes().await.map_err(|e| invalid(format!("invalid multipart field '{name}': {e}")))?;
        let text = || String::from_utf8_lossy(&value).trim().to_string();
        match name.as_str() {
            "file" => audio = Some(value.to_vec()),
            "model" => model = Some(text()),
            "language" => language = Some(text()),
            "response_format" => format = Some(text()),
            // Accepted for compatibility, with no effect on brain's models.
            _ => {}
        }
    }
    let audio = audio.ok_or_else(|| invalid("'file' is required".into()))?;
    finish(provider, model, audio, language, format.as_deref())
}

fn read_json(provider: Provider, body: &Value) -> Result<TranscriptionRequest, ApiError> {
    let invalid = |m: String| ApiError::invalid_request(provider, m);
    let data = body
        .get("input_audio")
        .and_then(|a| a.get("data"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("'input_audio.data' (base64) is required".into()))?;
    let audio = crate::b64::decode(data).map_err(|e| invalid(format!("'input_audio.data' is not base64: {e}")))?;
    let text = |key: &str| body.get(key).and_then(Value::as_str).map(str::to_string);
    finish(provider, text("model"), audio, text("language"), body.get("response_format").and_then(Value::as_str))
}

fn finish(provider: Provider, model: Option<String>, audio: Vec<u8>, language: Option<String>, format: Option<&str>) -> Result<TranscriptionRequest, ApiError> {
    let invalid = |m: String| ApiError::invalid_request(provider, m);
    let model = model.filter(|m| !m.is_empty()).ok_or_else(|| invalid("'model' is required".into()))?;
    if !matches!(format, None | Some("json")) {
        return Err(invalid(format!("response_format {:?} is not supported; use \"json\"", format.unwrap_or_default())));
    }
    let language = language.filter(|l| !l.is_empty());
    if let Some(l) = &language {
        if !(2..=3).contains(&l.len()) || !l.bytes().all(|b| b.is_ascii_lowercase()) {
            return Err(invalid(format!("language {l:?} is not an ISO-639-1 code such as \"en\"")));
        }
    }
    Ok(TranscriptionRequest { model, audio, language })
}
