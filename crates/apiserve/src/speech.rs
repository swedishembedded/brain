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
//! The body is `{"model", "input"}` plus what steers the voice, each mapped onto the action through its
//! [`capability::speech::SpeechContract`], never guessed from param names:
//!
//! - `input_references`: the voice to imitate, as the OpenRouter-style media parts an agent runtime sends - one
//!   `input_audio` part (`{"data": "data:audio/wav;base64,..."}` or bare base64, with an optional `format`) and,
//!   optionally, one `text` part holding what the recording says. The recording is decoded at its own rate (WAV,
//!   Ogg/Opus or WebM/Opus, told from its bytes; [`audio::decode::decode_recording`]) and handed to the model as a WAV,
//!   so the model resamples it itself for each of its parts.
//! - `instructions`: a free-text description of the voice and delivery (OpenAI's field).
//! - `voice`: one of the model's preset voices, by name. A custom voice object (`{"id": ...}`) is refused: brain keeps
//!   no stored voices, and a reference goes in `input_references`.
//! - `seed`: the run's seed (brain's extension), echoed in the `x-seed` header with the seed actually used.
//!
//! The model's speech action is the one whose contract has a place for everything the request carries
//! ([`crate::catalog::speech_action_for`]); a field no action of the model can take is a 400 naming it, never
//! dropped. `speed` and `stream_format` are refused by name. `response_format` is `wav` (the default: a 16-bit mono
//! WAV) or `pcm` (raw 16-bit signed little-endian mono at the model's rate, as OpenAI's `pcm`); a compressed format is
//! a 400, since brain writes none. The call is held to the action's own contract before any job exists, and the
//! reply's length is bounded by the action's length cap.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use capability::speech::SpeechRequest as Steering;
use capability::{Blob, Invocation};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::CallerExt;
use crate::bridge;
use crate::catalog;
use crate::error::ApiError;
use crate::state::AppState;

/// The longest `input` accepted, in characters (OpenAI's own bound).
pub const MAX_INPUT_CHARS: usize = 4096;
/// The longest `instructions` accepted, in characters (OpenAI's own bound).
pub const MAX_INSTRUCTIONS_CHARS: usize = 4096;
/// The longest reference transcript accepted, in characters.
pub const MAX_TRANSCRIPT_CHARS: usize = 4096;
/// The longest voice name accepted, in characters.
pub const MAX_VOICE_CHARS: usize = 64;
/// The longest reference recording accepted: a voice is imitated from seconds of speech, and every second of it is
/// prompt the model attends to at every step it writes.
pub const MAX_REFERENCE_SECONDS: u32 = 30;
/// The containers a reply may be asked for in.
pub const RESPONSE_FORMATS: [&str; 2] = ["wav", "pcm"];
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
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    voice: Option<Value>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    input_references: Option<Vec<ReferencePart>>,
}

/// One part of `input_references`.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ReferencePart {
    InputAudio { input_audio: InputAudio },
    Text { text: String },
    /// Read only to be refused by name: a picture is no voice to imitate.
    ImageUrl {
        #[serde(rename = "image_url")]
        _image_url: serde::de::IgnoredAny,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputAudio {
    data: String,
    #[serde(default)]
    format: Option<String>,
}

/// The voice a request asks to imitate, still encoded.
struct Reference {
    recording: Vec<u8>,
    transcript: Option<String>,
}

/// A parsed request: what is said, how the voice is steered, and the reply's container.
struct Parsed {
    model: String,
    input: String,
    format: Format,
    instructions: Option<String>,
    voice: Option<String>,
    seed: Option<u64>,
    reference: Option<Reference>,
}

impl Parsed {
    fn steering(&self) -> Steering {
        Steering {
            reference: self.reference.is_some(),
            transcript: self.reference.as_ref().is_some_and(|r| r.transcript.is_some()),
            instructions: self.instructions.is_some(),
            voice: self.voice.is_some(),
        }
    }
}

/// What a client asked for in the reply's container.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Wav,
    Pcm,
}

/// Fields OpenAI defines that brain cannot honour, refused by name before the body is read as a request.
const UNSUPPORTED: [&str; 2] = ["speed", "stream_format"];

fn parse(provider: crate::surface::Provider, body: &Bytes) -> Result<Parsed, ApiError> {
    let invalid = |m: String| ApiError::invalid_request(provider, m);
    let value: Value = serde_json::from_slice(body).map_err(|e| invalid(format!("invalid JSON body: {e}")))?;
    if let Some(name) = UNSUPPORTED.iter().find(|n| value.get(**n).is_some_and(|v| !v.is_null())) {
        return Err(invalid(format!("'{name}' is not supported: brain's speech models do not take it")));
    }
    let req: SpeechRequest = serde_json::from_value(value).map_err(|e| invalid(format!("invalid request: {e}")))?;
    let bounded = |field: &str, text: &str, max: usize| {
        if text.trim().is_empty() || text.chars().count() > max {
            Err(invalid(format!("'{field}' must be 1 to {max} characters")))
        } else {
            Ok(())
        }
    };
    bounded("input", &req.input, MAX_INPUT_CHARS)?;
    if let Some(i) = &req.instructions {
        bounded("instructions", i, MAX_INSTRUCTIONS_CHARS)?;
    }
    let format = match req.response_format.as_deref() {
        None | Some("wav") => Format::Wav,
        Some("pcm") => Format::Pcm,
        Some(other) => return Err(invalid(format!("'response_format' {other:?} is not supported: brain writes wav and pcm"))),
    };
    let voice = match req.voice {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) => {
            bounded("voice", &name, MAX_VOICE_CHARS)?;
            Some(name)
        }
        Some(Value::Object(_)) => return Err(invalid("'voice' as a custom voice object is not supported: brain keeps no stored voices; send the recording to imitate as an input_audio part of 'input_references'".into())),
        Some(_) => return Err(invalid("'voice' must be the name of one of the model's preset voices".into())),
    };
    let reference = match req.input_references {
        None => None,
        Some(parts) => reference_of(parts).map_err(invalid)?,
    };
    Ok(Parsed { model: req.model, input: req.input, format, instructions: req.instructions, voice, seed: req.seed, reference })
}

/// The one recording and the optional transcript `input_references` may carry, still encoded.
fn reference_of(parts: Vec<ReferencePart>) -> Result<Option<Reference>, String> {
    let mut recording = None;
    let mut transcript = None;
    for part in parts {
        match part {
            ReferencePart::InputAudio { input_audio } => {
                if recording.is_some() {
                    return Err("'input_references' carries more than one input_audio part: brain imitates one recording".into());
                }
                if let Some(f) = input_audio.format.as_deref().filter(|f| !matches!(*f, "wav" | "ogg" | "opus" | "webm")) {
                    return Err(format!("input_audio: format {f:?} is not supported; accepted formats are {}", audio::decode::ACCEPTED_FORMATS));
                }
                let payload = capability::blob::data_url_payload(&input_audio.data).unwrap_or(&input_audio.data);
                recording = Some(crate::b64::decode(payload).map_err(|e| format!("input_audio: the data is not base64: {e}"))?);
            }
            ReferencePart::Text { text } => {
                if transcript.is_some() {
                    return Err("'input_references' carries more than one text part: the one text part is the recording's transcript".into());
                }
                if text.trim().is_empty() || text.chars().count() > MAX_TRANSCRIPT_CHARS {
                    return Err(format!("the transcript text part must be 1 to {MAX_TRANSCRIPT_CHARS} characters"));
                }
                transcript = Some(text);
            }
            ReferencePart::ImageUrl { .. } => return Err("'input_references' image_url parts are not supported: brain's speech models imitate a voice from a recording".into()),
        }
    }
    match (recording, transcript) {
        (Some(recording), transcript) => Ok(Some(Reference { recording, transcript })),
        (None, Some(_)) => Err("a transcript text part goes with the input_audio recording it transcribes".into()),
        (None, None) => Ok(None),
    }
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
    let parsed = match parse(provider, &body) {
        Ok(parsed) => parsed,
        Err(e) => return e.into_response(),
    };
    let body_model = json!({ "model": parsed.model });
    let manifests = state.exec.manifests();
    let model = match catalog::resolve_model(provider, &body_model, |id| catalog::resolve_speech(&manifests, id)) {
        Some((_, id)) => id,
        None => match bridge::ensure_and_recheck(&state, provider, &parsed.model, |id| catalog::resolve_speech(&state.exec.manifests(), id)).await {
            Ok(id) => id,
            Err(e) => return e.into_response(),
        },
    };
    let manifests = state.exec.manifests();
    let Some(manifest) = manifests.iter().find(|m| m.model == model) else {
        return ApiError::model_not_found(provider, &model).into_response();
    };
    let spec = match catalog::speech_action_for(manifest, parsed.steering()) {
        Ok(spec) => spec,
        Err(e) => return ApiError::invalid_request(provider, e).into_response(),
    };
    let Some(contract) = spec.speech.clone() else {
        return ApiError::model_not_found(provider, &model).into_response();
    };
    if parsed.seed.is_some() && contract.seed.is_none() {
        return ApiError::invalid_request(provider, format!("'seed' is not supported by model '{model}'")).into_response();
    }
    let mut call = Invocation::new().set(&contract.text, json!(parsed.input));
    if let Some(length) = &contract.length {
        call = call.set(length, json!(MAX_FRAMES));
    }
    if let (Some(param), Some(seed)) = (&contract.seed, parsed.seed) {
        call = call.set(param, json!(seed));
    }
    if let (Some(param), Some(text)) = (&contract.instructions, &parsed.instructions) {
        call = call.set(param, json!(text));
    }
    if let (Some(param), Some(name)) = (&contract.voice, &parsed.voice) {
        call = call.set(param, json!(name));
    }
    if let (Some(slot), Some(reference)) = (&contract.reference, parsed.reference) {
        let Some(media) = spec.inputs.iter().find(|b| b.name == slot.input).map(|b| b.media) else {
            return ApiError::invalid_request(provider, format!("model '{model}' declares no input '{}' for its reference voice", slot.input)).into_response();
        };
        let recording = reference.recording;
        let decoded = crate::preprocess::off_runtime(provider, move || audio::decode::decode_recording(&recording, MAX_REFERENCE_SECONDS)).await;
        let (samples, rate) = match decoded {
            Ok(Ok(decoded)) => decoded,
            Ok(Err(e)) => return ApiError::invalid_request(provider, format!("input_audio: {e}")).into_response(),
            Err(e) => return e.into_response(),
        };
        let wav = Blob::new(media, audio::wav::encode(&samples, rate)).with_meta(json!({"format": "wav", "sample_rate": rate, "channels": 1}));
        call = call.blob(&slot.input, wav);
        if let (Some(param), Some(text)) = (slot.transcript.param(), reference.transcript) {
            call = call.set(param, json!(text));
        }
    }
    let action = spec.name.clone();
    let outcome = match bridge::admit(&state, &model, &action, call, bridge::Naming::Params) {
        Ok(call) => bridge::submit(&state, call).await,
        Err(e) => Err(e),
    };
    let outcome = match outcome {
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
    let (bytes, content_type) = match parsed.format {
        Format::Wav => (audio::wav::encode(&samples, rate), "audio/wav"),
        Format::Pcm => (samples.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect::<Vec<u8>>(), "audio/pcm"),
    };
    let mut response = (StatusCode::OK, bytes).into_response();
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    // A client reading raw PCM needs the rate: it is the model's, not a constant.
    if let Ok(value) = HeaderValue::from_str(&rate.to_string()) {
        response.headers_mut().insert("x-sample-rate", value);
    }
    // The seed the take was drawn with, so a client can ask for the same take again.
    if let Some(seed) = outcome.outputs.get("seed").and_then(Value::as_u64) {
        if let Ok(value) = HeaderValue::from_str(&seed.to_string()) {
            response.headers_mut().insert("x-seed", value);
        }
    }
    response
}
