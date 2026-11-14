// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decoding OpenAI `image_url`/`input_audio` content parts and Anthropic
//! `image` content blocks into brain's own blob wire format
//! (`capability::blob::image_blob`, `audio::asr_caps`'s raw-16kHz-PCM
//! convention) - the fix for the "multimodal content parts are silently
//! dropped" gap `openai.rs`/`anthropic.rs`'s own `content_text` functions
//! have always had (previously flagged and left open; still true for every
//! OTHER model, not just `brain/qwen3omnimoe` - this is a generic content-part fix,
//! not omni-specific).
//!
//! **Scope**: inline `data:` URLs / base64 payloads only - no external URL
//! fetching. A plain `http(s)://` `image_url` is valid per OpenAI's own
//! schema, but this server does not fetch third-party URLs on a client's
//! behalf (the same boundary this codebase draws elsewhere for outbound
//! network calls) - it errors with a clear message instead of silently
//! dropping the image as before. Every image of a request is extracted (up to
//! [`MAX_IMAGES`], scanning all messages in order) and attached as the blobs
//! `image`, `image1`, ... ([`image_key`]), the naming a model that takes
//! several images declares (`brain-deepseekvl`'s `generate`); a model that
//! declares only `image` reads the first. At most ONE audio clip is
//! extracted (the first found), matching the single `audio` blob input
//! `qwen3omnimoe::caps::generate_spec()` declares.

use capability::Blob;
use serde_json::Value;

use crate::b64;

/// The most images one request may carry. Each is decoded to a float image
/// held in memory while the request runs, so the count is bounded before
/// anything is decoded past it.
pub use capability::blob::MAX_IMAGES;

pub use capability::blob::{MAX_IMAGE_PIXELS, MAX_REQUEST_PIXELS};

use capability::blob::image_key;

/// The images (in order) and at most one audio clip found across a request's
/// messages, ready to attach to an `Invocation` via `.blob(...)`.
#[derive(Default, Debug)]
pub struct ExtractedMedia {
    pub images: Vec<Blob>,
    pub audio: Option<Blob>,
    /// Pixels of the images so far, against [`MAX_REQUEST_PIXELS`].
    pixels: u64,
}

impl ExtractedMedia {
    /// Attach every image to `inv` under [`image_key`]'s names.
    pub fn attach_images(&mut self, mut inv: capability::Invocation) -> capability::Invocation {
        for (i, img) in self.images.drain(..).enumerate() {
            inv = inv.blob(&image_key(i), img);
        }
        inv
    }

    /// Decode the next image's bytes (PNG/JPEG/PPM) into brain's HWC-f32 image
    /// blob. The image is refused from its header, before it is decoded, when
    /// it is over [`MAX_IMAGE_PIXELS`] or over what is left of the request's
    /// [`MAX_REQUEST_PIXELS`]: checking the total only after decoding let one
    /// request decode a whole further image past it.
    fn push_image(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.images.len() == MAX_IMAGES {
            return Err(format!("a request may carry at most {MAX_IMAGES} images"));
        }
        let left = MAX_REQUEST_PIXELS - self.pixels;
        let rgb = imaging::codec::decode_bounded(bytes, MAX_IMAGE_PIXELS.min(left)).map_err(|e| {
            if left < MAX_IMAGE_PIXELS {
                format!("{e} (the images of a request may hold at most {MAX_REQUEST_PIXELS} pixels together)")
            } else {
                e
            }
        })?;
        self.pixels += u64::from(rgb.w) * u64::from(rgb.h);
        self.images.push(capability::blob::image_blob(&rgb.to_hwc_unit(), rgb.w, rgb.h, 3));
        Ok(())
    }
}

/// Decode one image content part's bytes (PNG/JPEG/PPM, via
/// `imaging::codec::decode`): its bytes, for [`ExtractedMedia::push_image`].
fn image_data_url_bytes(data_url: &str) -> Result<Vec<u8>, String> {
    let payload = capability::blob::data_url_payload(data_url).ok_or_else(|| "image_url: only inline 'data:...;base64,...' URLs are supported (no external fetch)".to_string())?;
    b64::decode(payload)
}

/// Decode one `input_audio` part's base64 payload (a whole recording FILE per
/// OpenAI's schema - not raw PCM) into brain's raw-16kHz-mono-PCM audio blob
/// (`audio::asr_caps`'s wire convention), through the same decoder as
/// `/audio/transcriptions`: WAV, Ogg/Opus or WebM/Opus, told from the bytes.
/// A `format` naming anything else (MP3, for which no decoder exists here) is
/// refused clearly rather than guessed at.
fn decode_input_audio(b64_data: &str, format: &str) -> Result<Blob, String> {
    if !matches!(format, "wav" | "ogg" | "opus" | "webm") {
        return Err(format!("input_audio: format {format:?} is not supported; accepted formats are {}", audio::decode::ACCEPTED_FORMATS));
    }
    let bytes = b64::decode(b64_data)?;
    let samples = audio::decode::decode_speech(&bytes, crate::audio::MAX_TRANSCRIPTION_SECONDS).map_err(|e| format!("input_audio: {e}"))?;
    Ok(audio::asr_caps::audio_blob_from_samples(&samples))
}

/// Whether a chat body carries media to decode: an OpenAI `image_url` or
/// `input_audio` part, or an Anthropic `image` block. A request that does is
/// parsed through [`crate::preprocess::off_runtime`]; one that does not stays
/// on the async worker, where parsing text costs nothing worth bounding.
pub fn carries_media(body: &Value) -> bool {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else { return false };
    fn any_media(parts: &[Value]) -> bool {
        parts.iter().any(|part| match part.get("type").and_then(Value::as_str) {
            Some("image_url" | "input_audio" | "image") => true,
            // An Anthropic tool result carries its own content blocks.
            Some("tool_result") => part.get("content").and_then(Value::as_array).is_some_and(|inner| any_media(inner)),
            _ => false,
        })
    }
    messages.iter().filter_map(|m| m.get("content").and_then(Value::as_array)).any(|parts| any_media(parts))
}

/// Scan OpenAI-shaped `messages` (the RAW pre-flatten request array) for
/// every `image_url` content part and the first `input_audio` one across all
/// messages, in order.
pub fn extract_openai(messages: &[Value]) -> Result<ExtractedMedia, String> {
    let mut out = ExtractedMedia::default();
    for m in messages {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        for p in parts {
            match p.get("type").and_then(|v| v.as_str()) {
                Some("image_url") => {
                    if let Some(url) = p.get("image_url").and_then(|u| u.get("url")).and_then(|v| v.as_str()) {
                        out.push_image(&image_data_url_bytes(url)?)?;
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

/// Scan Anthropic-shaped `messages` for every `image` content block
/// across all messages (`{"type":"image","source":{"type":"base64",
/// "media_type":...,"data":...}}` - `source.type` other than `"base64"`,
/// e.g. a URL source, is out of scope for the same reason OpenAI's
/// external `image_url` is).
pub fn extract_anthropic(messages: &[Value]) -> Result<ExtractedMedia, String> {
    let mut out = ExtractedMedia::default();
    for m in messages {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        anthropic_images(parts, &mut out)?;
    }
    Ok(out)
}

/// The base64 images among Anthropic content `parts`, in order, including
/// those a `tool_result` block carries (a tool that read an image file).
fn anthropic_images(parts: &[Value], out: &mut ExtractedMedia) -> Result<(), String> {
    for p in parts {
        match p.get("type").and_then(|v| v.as_str()) {
            Some("tool_result") => {
                if let Some(inner) = p.get("content").and_then(|c| c.as_array()) {
                    anthropic_images(inner, out)?;
                }
            }
            Some("image") => {
                let source = p.get("source");
                if source.and_then(|s| s.get("type")).and_then(|v| v.as_str()) != Some("base64") {
                    continue; // a URL source: out of scope, see this function's doc
                }
                if let Some(data) = source.and_then(|s| s.get("data")).and_then(|v| v.as_str()) {
                    // Anthropic's raw base64 `source.data`, no `data:` URL wrapper.
                    out.push_image(&b64::decode(data)?)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Media;
    use serde_json::json;

    /// A 1x1 white binary PPM (P6) - `imaging::codec::decode` supports P6
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
        let img = got.images.first().expect("image found");
        assert_eq!(img.media, Media::Image);
        assert_eq!(img.meta["w"], 1);
        assert_eq!(img.meta["h"], 1);
    }

    /// `P6\n2 1\n255\n` + 6 bytes: a 2x1 image, so the order the images come
    /// back in is visible.
    const WIDE_PPM_B64: &str = "UDYKMiAxCjI1NQr///8AAAA=";

    fn image_part(b64: &str) -> serde_json::Value {
        json!({"type": "image_url", "image_url": {"url": format!("data:image/x-ppm;base64,{b64}")}})
    }

    #[test]
    fn every_image_of_a_request_is_extracted_in_order() {
        let messages = json!([
            {"role": "user", "content": [image_part(TINY_PNG_B64), {"type": "text", "text": "and"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [image_part(WIDE_PPM_B64)]}
        ]);
        let got = extract_openai(messages.as_array().unwrap()).expect("extract");
        let sizes: Vec<(u64, u64)> = got.images.iter().map(|i| (i.meta["w"].as_u64().unwrap(), i.meta["h"].as_u64().unwrap())).collect();
        assert_eq!(sizes, [(1, 1), (2, 1)], "across messages, in the order they were sent");
        let anthropic = json!([{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/x-ppm", "data": WIDE_PPM_B64}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/x-ppm", "data": TINY_PNG_B64}}
        ]}]);
        let got = extract_anthropic(anthropic.as_array().unwrap()).expect("extract");
        assert_eq!(got.images.iter().map(|i| i.meta["w"].as_u64().unwrap()).collect::<Vec<_>>(), [2, 1]);
    }

    #[test]
    fn a_request_may_not_carry_more_images_than_the_cap() {
        let parts: Vec<_> = (0..=MAX_IMAGES).map(|_| image_part(TINY_PNG_B64)).collect();
        let err = extract_openai(json!([{"role": "user", "content": parts}]).as_array().unwrap()).unwrap_err();
        assert!(err.contains(&format!("at most {MAX_IMAGES} images")), "{err}");
        let blocks: Vec<_> = (0..=MAX_IMAGES).map(|_| json!({"type": "image", "source": {"type": "base64", "media_type": "image/x-ppm", "data": TINY_PNG_B64}})).collect();
        let err = extract_anthropic(json!([{"role": "user", "content": blocks}]).as_array().unwrap()).unwrap_err();
        assert!(err.contains("at most"), "{err}");
        let within: Vec<_> = (0..MAX_IMAGES).map(|_| image_part(TINY_PNG_B64)).collect();
        assert_eq!(extract_openai(json!([{"role": "user", "content": within}]).as_array().unwrap()).unwrap().images.len(), MAX_IMAGES);
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
        assert_eq!(got.images.len(), 1);
    }

    #[test]
    fn a_plain_text_only_message_extracts_nothing() {
        let messages = json!([{"role": "user", "content": "just text"}]);
        let got = extract_openai(messages.as_array().unwrap()).expect("extract");
        assert!(got.images.is_empty() && got.audio.is_none());
    }

    #[test]
    fn input_audio_rejects_non_wav_format_clearly() {
        let messages = json!([
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}}]}
        ]);
        let err = extract_openai(messages.as_array().unwrap()).unwrap_err();
        assert!(err.contains("wav"), "error should name the supported format, got: {err}");
    }

    /// A PNG that is a signature, a valid IHDR declaring `w x h` and the start of the pixel stream: all a bomb needs.
    fn png_declaring(w: u32, h: u32) -> Vec<u8> {
        let crc = |bytes: &[u8]| {
            let mut c = 0xFFFF_FFFFu32;
            for &b in bytes {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = (c >> 1) ^ (0xEDB8_8320 & (c & 1).wrapping_neg());
                }
            }
            !c
        };
        let mut ihdr = b"IHDR".to_vec();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let idat = b"IDAT\x78\x01".to_vec();
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&crc(&ihdr).to_be_bytes());
        png.extend_from_slice(&[0, 0, 0, 2]);
        png.extend_from_slice(&idat);
        png.extend_from_slice(&crc(&idat).to_be_bytes());
        png
    }

    #[test]
    fn an_image_declaring_more_pixels_than_the_limit_is_refused_before_it_is_decoded() {
        let url = format!("data:image/png;base64,{}", crate::b64::encode(&png_declaring(65_535, 65_535)));
        let messages = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]);
        let err = extract_openai(messages.as_array().unwrap()).unwrap_err();
        assert!(err.contains("limit"), "{err}");
        let blocks = json!([{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": crate::b64::encode(&png_declaring(65_535, 65_535))}}]}]);
        assert!(extract_anthropic(blocks.as_array().unwrap()).unwrap_err().contains("limit"));
    }

    /// What is left of the request's budget bounds the next image from its
    /// header: the PNG below has no pixel data, so a refusal that names the
    /// budget proves nothing was decoded first.
    #[test]
    fn the_images_of_one_request_share_a_pixel_budget_checked_before_decoding() {
        let mut media = ExtractedMedia { pixels: MAX_REQUEST_PIXELS - 1_000, ..ExtractedMedia::default() };
        let err = media.push_image(&png_declaring(100, 100)).unwrap_err();
        assert!(err.contains("limit") && err.contains("together"), "{err}");
        assert!(media.images.is_empty());
    }
}
