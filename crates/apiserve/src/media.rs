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
//! several images declares (`brain-deepseekvl`'s `generate`). A request
//! carries at most ONE audio clip, matching the single `audio` blob input
//! `qwen3omnimoe::caps::generate_spec()` declares; a second is refused.
//!
//! **Every content part is accounted for**: it is text the prompt keeps, a
//! medium decoded here, or a conversation block the provider module reads
//! (tool calls, reasoning). Any other part, and a media part this server
//! cannot read (a URL source, a missing payload), refuses the request rather
//! than being dropped, so a reply never looks as if it had seen input it
//! did not. `crate::bridge::refuse_unread_media` then checks that the model's
//! action declares an input for every medium attached.

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
/// every `image_url` content part and the one `input_audio` part across all
/// messages, in order. A part of any other type than `text`, `refusal` (an
/// assistant turn's), `image_url` or `input_audio` is refused, as is a media
/// part without its payload and a second audio clip.
pub fn extract_openai(messages: &[Value]) -> Result<ExtractedMedia, String> {
    let mut out = ExtractedMedia::default();
    for m in messages {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        for p in parts {
            match p.get("type").and_then(|v| v.as_str()) {
                Some("text" | "refusal") => {}
                Some("image_url") => {
                    let url = p.get("image_url").and_then(|u| u.get("url")).and_then(|v| v.as_str()).ok_or("image_url: the part has no image_url.url")?;
                    out.push_image(&image_data_url_bytes(url)?)?;
                }
                Some("input_audio") => {
                    if out.audio.is_some() {
                        return Err("a request may carry at most one input_audio clip".to_string());
                    }
                    let ia = p.get("input_audio");
                    let data = ia.and_then(|a| a.get("data")).and_then(|v| v.as_str()).ok_or("input_audio: the part has no input_audio.data")?;
                    let format = ia.and_then(|a| a.get("format")).and_then(|v| v.as_str()).unwrap_or("wav");
                    out.audio = Some(decode_input_audio(data, format)?);
                }
                other => return Err(unsupported_part(other, "text, image_url, input_audio")),
            }
        }
    }
    Ok(out)
}

/// The refusal for a content part of a type this server does not read.
fn unsupported_part(kind: Option<&str>, accepted: &str) -> String {
    match kind {
        Some(kind) => format!("content part type {kind:?} is not supported; accepted types are {accepted}"),
        None => format!("a content part has no \"type\"; accepted types are {accepted}"),
    }
}

/// Scan Anthropic-shaped `messages` for every `image` content block
/// across all messages (`{"type":"image","source":{"type":"base64",
/// "media_type":...,"data":...}}`). A `source.type` other than `"base64"`,
/// e.g. a URL source, is refused for the same reason OpenAI's external
/// `image_url` is, and so is a block of a type the conversation does not
/// carry (a `document`, say).
pub fn extract_anthropic(messages: &[Value]) -> Result<ExtractedMedia, String> {
    let mut out = ExtractedMedia::default();
    for m in messages {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        anthropic_images(parts, &mut out)?;
    }
    Ok(out)
}

/// The block types an Anthropic conversation carries: text, images, and the
/// tool and reasoning blocks `crate::anthropic` maps onto the contract.
const ANTHROPIC_BLOCKS: &str = "text, image, tool_use, tool_result, thinking, redacted_thinking";

/// The base64 images among Anthropic content `parts`, in order, including
/// those a `tool_result` block carries (a tool that read an image file).
fn anthropic_images(parts: &[Value], out: &mut ExtractedMedia) -> Result<(), String> {
    for p in parts {
        match p.get("type").and_then(|v| v.as_str()) {
            Some("text" | "tool_use" | "thinking" | "redacted_thinking") => {}
            Some("tool_result") => {
                if let Some(inner) = p.get("content").and_then(|c| c.as_array()) {
                    anthropic_images(inner, out)?;
                }
            }
            Some("image") => {
                let source = p.get("source");
                match source.and_then(|s| s.get("type")).and_then(|v| v.as_str()) {
                    Some("base64") => {}
                    other => return Err(format!("image: source type {} is not supported; send the image inline as a \"base64\" source (no external fetch)", other.map_or("(none)".to_string(), |t| format!("{t:?}")))),
                }
                let data = source.and_then(|s| s.get("data")).and_then(|v| v.as_str()).ok_or("image: the base64 source has no data")?;
                // Anthropic's raw base64 `source.data`, no `data:` URL wrapper.
                out.push_image(&b64::decode(data)?)?;
            }
            other => return Err(unsupported_part(other, ANTHROPIC_BLOCKS)),
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

    /// Every part is read, decoded, or refused: none is dropped on the way to
    /// the model, so a reply never looks as if it had seen input it did not.
    #[test]
    fn a_part_the_server_cannot_read_refuses_the_request() {
        let wav = crate::b64::encode(&audio::wav::encode(&[0.0; 1600], 16_000));
        let clip = json!({"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}});
        let two_clips = json!([{"role": "user", "content": [clip.clone(), clip.clone()]}]);
        assert!(extract_openai(two_clips.as_array().unwrap()).unwrap_err().contains("at most one input_audio"));
        let one_clip = json!([{"role": "user", "content": [clip]}]);
        assert!(extract_openai(one_clip.as_array().unwrap()).unwrap().audio.is_some());
        let file = json!([{"role": "user", "content": [{"type": "file", "file": {"file_id": "f"}}]}]);
        assert!(extract_openai(file.as_array().unwrap()).unwrap_err().contains("\"file\" is not supported"));
        let no_payload = json!([{"role": "user", "content": [{"type": "input_audio", "input_audio": {"format": "wav"}}]}]);
        assert!(extract_openai(no_payload.as_array().unwrap()).unwrap_err().contains("no input_audio.data"));

        let url_image = json!([{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]}]);
        assert!(extract_anthropic(url_image.as_array().unwrap()).unwrap_err().contains("source type \"url\""));
        let document = json!([{"role": "user", "content": [{"type": "document", "source": {"type": "text", "data": "x"}}]}]);
        assert!(extract_anthropic(document.as_array().unwrap()).unwrap_err().contains("\"document\" is not supported"));
        let conversation = json!([
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": ""}, {"type": "tool_use", "id": "a", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "ok"}]}, {"type": "text", "text": "go on"}]}
        ]);
        assert!(extract_anthropic(conversation.as_array().unwrap()).unwrap().images.is_empty());
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
