// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decoding OpenAI `image_url`/`input_audio` content parts and Anthropic
//! `image` content blocks into brain's own blob wire format
//! (`capability::blob::image_blob`, `audio::asr_caps`'s raw-16kHz-PCM
//! convention) — the fix for the "multimodal content parts are silently
//! dropped" gap `openai.rs`/`anthropic.rs`'s own `content_text` functions
//! have always had (M11/M12 flagged it, left it open; still true for every
//! OTHER model, not just `brain/omni` — this is a generic content-part fix,
//! not omni-specific).
//!
//! **Scope**: inline `data:` URLs / base64 payloads only — no external URL
//! fetching. A plain `http(s)://` `image_url` is valid per OpenAI's own
//! schema, but this server does not fetch third-party URLs on a client's
//! behalf (the same boundary this codebase draws elsewhere for outbound
//! network calls) — it errors with a clear message instead of silently
//! dropping the image as before. At most ONE image and ONE audio clip are
//! extracted per request (the first found, scanning all messages in
//! order) — matches the single-image/single-audio-input shape
//! `omni::caps::generate_spec()`'s `audio`/`image` blob inputs already
//! declare; a model that wants more would need a richer wire shape this
//! module doesn't attempt to invent.

use capability::{Blob, Media};
use serde_json::Value;

use crate::b64;

/// The (at most one image, at most one audio) blobs found across a
/// request's messages, ready to attach to an `Invocation` via `.blob(...)`.
#[derive(Default, Debug)]
pub struct ExtractedMedia {
    pub image: Option<Blob>,
    pub audio: Option<Blob>,
}

/// Decode a `data:<mime>;base64,<payload>` URL's payload, or `None` if `url`
/// isn't a data URL (a real `http(s)://` URL — out of scope, see module doc).
fn data_url_payload(url: &str) -> Option<&str> {
    let rest = url.strip_prefix("data:")?;
    let (_, payload) = rest.split_once(";base64,")?;
    Some(payload)
}

/// Decode one image content part's bytes (PNG/JPEG/PPM, via
/// `imaging::codec::decode`) into brain's HWC-f32 image blob.
fn decode_image_data_url(data_url: &str) -> Result<Blob, String> {
    let payload = data_url_payload(data_url).ok_or_else(|| "image_url: only inline 'data:...;base64,...' URLs are supported (no external fetch)".to_string())?;
    let bytes = b64::decode(payload)?;
    let rgb = imaging::codec::decode(&bytes)?;
    Ok(capability::blob::image_blob(&rgb.to_hwc_unit(), rgb.w, rgb.h, 3))
}

/// Decode one image content block's raw base64 payload (Anthropic's own
/// `source.data` — no `data:` URL wrapper, unlike OpenAI's `image_url`).
fn decode_image_base64(payload: &str) -> Result<Blob, String> {
    let bytes = b64::decode(payload)?;
    let rgb = imaging::codec::decode(&bytes)?;
    Ok(capability::blob::image_blob(&rgb.to_hwc_unit(), rgb.w, rgb.h, 3))
}

/// Decode one `input_audio` part's base64 payload (a whole WAV/MP3 FILE per
/// OpenAI's schema — not raw PCM) into brain's raw-16kHz-mono-PCM audio
/// blob (`audio::asr_caps`'s wire convention). Only WAV is actually
/// decodable today — no MP3 decoder exists in this workspace — an MP3
/// payload errors clearly rather than silently producing garbage/empty
/// audio.
fn decode_input_audio(b64_data: &str, format: &str) -> Result<Blob, String> {
    if format != "wav" {
        return Err(format!("input_audio: only format 'wav' is supported (no MP3 decoder in this workspace), got {format:?}"));
    }
    let bytes = b64::decode(b64_data)?;
    let wav = audio::wav::parse(&bytes).map_err(|e| format!("input_audio: {e}"))?;
    let samples = audio::resample_linear(&wav.samples, wav.sample_rate, 16000);
    let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Ok(Blob::new(Media::Audio, pcm).with_meta(serde_json::json!({"sample_rate": 16000})))
}

/// Scan OpenAI-shaped `messages` (the RAW pre-flatten request array) for the
/// first `image_url`/`input_audio` content part across all messages, in
/// order.
pub fn extract_openai(messages: &[Value]) -> Result<ExtractedMedia, String> {
    let mut out = ExtractedMedia::default();
    for m in messages {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        for p in parts {
            match p.get("type").and_then(|v| v.as_str()) {
                Some("image_url") if out.image.is_none() => {
                    if let Some(url) = p.get("image_url").and_then(|u| u.get("url")).and_then(|v| v.as_str()) {
                        out.image = Some(decode_image_data_url(url)?);
                    }
                }
                Some("input_audio") if out.audio.is_none() => {
                    let ia = p.get("input_audio");
                    let data = ia.and_then(|a| a.get("data")).and_then(|v| v.as_str());
                    let format = ia.and_then(|a| a.get("format")).and_then(|v| v.as_str()).unwrap_or("wav");
                    if let Some(data) = data {
                        out.audio = Some(decode_input_audio(data, format)?);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(out)
}

/// Scan Anthropic-shaped `messages` for the first `image` content block
/// across all messages (`{"type":"image","source":{"type":"base64",
/// "media_type":...,"data":...}}` — `source.type` other than `"base64"`,
/// e.g. a URL source, is out of scope for the same reason OpenAI's
/// external `image_url` is).
pub fn extract_anthropic(messages: &[Value]) -> Result<ExtractedMedia, String> {
    let mut out = ExtractedMedia::default();
    for m in messages {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        for p in parts {
            if out.image.is_some() {
                break;
            }
            if p.get("type").and_then(|v| v.as_str()) != Some("image") {
                continue;
            }
            let source = p.get("source");
            if source.and_then(|s| s.get("type")).and_then(|v| v.as_str()) != Some("base64") {
                continue; // a URL source: out of scope, see this function's doc
            }
            if let Some(data) = source.and_then(|s| s.get("data")).and_then(|v| v.as_str()) {
                out.image = Some(decode_image_base64(data)?);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A 1x1 white binary PPM (P6) — `imaging::codec::decode` supports P6
    /// alongside PNG/JPEG (`crates/cli/src/image_io.rs`'s own doc), and a
    /// hand-built PPM is trivially verifiable byte for byte (unlike a
    /// memorized PNG base64 string, which risks an invalid fixture the
    /// test would then silently never really exercise the decoder with).
    /// `P6\n1 1\n255\n` + 3 RGB bytes (255,255,255), base64-encoded.
    const TINY_PNG_B64: &str = "UDYKMSAxCjI1NQr///8=";

    #[test]
    fn extracts_an_openai_image_url_data_uri() {
        let messages = json!([
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{TINY_PNG_B64}")}}
            ]}
        ]);
        let got = extract_openai(messages.as_array().unwrap()).expect("extract");
        let img = got.image.expect("image found");
        assert_eq!(img.media, Media::Image);
        assert_eq!(img.meta["w"], 1);
        assert_eq!(img.meta["h"], 1);
    }

    #[test]
    fn an_external_http_image_url_errors_clearly_instead_of_silently_dropping() {
        let messages = json!([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}
        ]);
        let err = extract_openai(messages.as_array().unwrap()).unwrap_err();
        assert!(err.contains("data:"), "error should explain the data: URL requirement, got: {err}");
    }

    #[test]
    fn extracts_an_anthropic_base64_image_block() {
        let messages = json!([
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": TINY_PNG_B64}}
            ]}
        ]);
        let got = extract_anthropic(messages.as_array().unwrap()).expect("extract");
        assert!(got.image.is_some());
    }

    #[test]
    fn a_plain_text_only_message_extracts_nothing() {
        let messages = json!([{"role": "user", "content": "just text"}]);
        let got = extract_openai(messages.as_array().unwrap()).expect("extract");
        assert!(got.image.is_none() && got.audio.is_none());
    }

    #[test]
    fn input_audio_rejects_non_wav_format_clearly() {
        let messages = json!([
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}}]}
        ]);
        let err = extract_openai(messages.as_array().unwrap()).unwrap_err();
        assert!(err.contains("wav"), "error should name the supported format, got: {err}");
    }
}
