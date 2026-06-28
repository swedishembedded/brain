// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! JSONL event protocol between the brain runtime and its host.
//!
//! Each event is one line of JSON, internally tagged on an `"event"` field
//! (e.g. `{"event":"user_text","text":"hi"}`). [`encode_line`] serializes one
//! [`Event`] to such a line; [`decode_line`] parses one back. The codec is hand
//! rolled over `serde_json::Value` to keep the dependency surface to
//! `serde_json` only.
//!
//! Frame payloads ([`Event::CameraFrame`]) carry either inline base64 RGB8 (or
//! PPM P6) bytes in `data`, or a `path` to a file holding the same. The tiny,
//! dependency-free [`base64`] and [`ppm`] helpers here decode those into a raw
//! `Vec<u8>` of interleaved RGB; see [`decode_frame`]. PNG/JPEG are intentionally
//! unsupported (no decoder dependency) and return a clear error rather than
//! panicking.

use serde_json::{json, Value};

/// One protocol event. `#[non_exhaustive]` so adding variants in later phases is
/// not a breaking change for downstream `match`es.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// User-typed text fed into the brain.
    UserText { text: String },
    /// A streamed chunk of the brain's text response. `seq` orders chunks;
    /// `done` marks the final chunk of a response.
    BrainTextChunk { text: String, seq: u32, done: bool },
    /// A camera frame, either inlined (`data`, base64) or referenced (`path`).
    CameraFrame {
        format: String,
        w: u32,
        h: u32,
        data: Option<String>,
        path: Option<String>,
    },
    /// Object detections: `dets[i]` = `[x1, y1, x2, y2, score, class]`, with an
    /// optional parallel `labels` of class names.
    ObjectDetected { dets: Vec<[f32; 6]>, labels: Vec<String> },
    /// User request to synthesize speech from `text`, optionally voice-cloning a
    /// reference utterance (`ref_audio` path + its transcript `ref_text`) in a
    /// given `language`. The TTS analogue of [`Event::UserText`].
    UserSynthRequest {
        text: String,
        ref_audio: Option<String>,
        ref_text: Option<String>,
        language: Option<String>,
    },
    /// A streamed chunk of synthesized audio: base64 little-endian f32 PCM mono
    /// at `sample_rate`. `seq` orders chunks; `done` marks the final chunk of a
    /// response. The audio analogue of [`Event::BrainTextChunk`].
    AudioChunk { pcm_b64: String, sample_rate: u32, seq: u32, done: bool },
    /// Host-requested cancellation of the in-flight operation.
    Cancel,
    /// The runtime has finished initializing and is ready for input.
    Ready,
    /// A fatal/handled error condition.
    Error { message: String },
    /// A free-form diagnostic log line.
    Log { message: String },
}

/// A protocol line: one [`Event`] plus an OPTIONAL request-correlation id.
///
/// A client may stamp a request with a `req_id`; the runtime echoes that same
/// `req_id` on EVERY event emitted while handling that request, so multiple
/// in-flight requests over one stdio stream can be demultiplexed. Lines without
/// a `req_id` behave exactly as before (`req_id == None`), keeping the wire
/// format fully back-compatible.
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    /// Optional request-correlation id, serialized as a top-level `"req_id"`
    /// field only when `Some`.
    pub req_id: Option<String>,
    /// The wrapped protocol event.
    pub event: Event,
}

impl Envelope {
    /// An envelope with no correlation id (the back-compat default).
    pub fn bare(event: Event) -> Envelope {
        Envelope { req_id: None, event }
    }

    /// An envelope tagged with the given `req_id`.
    pub fn with_id(req_id: Option<String>, event: Event) -> Envelope {
        Envelope { req_id, event }
    }
}

/// Encode one [`Envelope`] as a single JSONL line. The event's own fields are
/// emitted exactly as [`encode_line`] would; a top-level `"req_id"` is added
/// only when `req_id` is `Some`.
pub fn encode_envelope(env: &Envelope) -> String {
    let mut v = event_to_value(&env.event);
    if let Some(id) = &env.req_id {
        // `v` is always a JSON object (every event encodes to one).
        if let Value::Object(map) = &mut v {
            map.insert("req_id".to_string(), json!(id));
        }
    }
    v.to_string()
}

/// Decode one JSONL line into an [`Envelope`]: the inner [`Event`] plus an
/// optional top-level `req_id` (a non-string or absent `req_id` ⇒ `None`).
/// Returns `Err` on malformed input or an unknown `"event"` tag.
pub fn decode_envelope(line: &str) -> Result<Envelope, String> {
    let v: Value = serde_json::from_str(line.trim()).map_err(|e| e.to_string())?;
    let req_id = v["req_id"].as_str().map(|s| s.to_string());
    let event = decode_event_value(&v)?;
    Ok(Envelope { req_id, event })
}

/// Encode one event as a single JSONL line (no trailing newline).
///
/// Back-compat shim over [`encode_envelope`] with no `req_id`.
pub fn encode_line(ev: &Event) -> String {
    encode_envelope(&Envelope::bare(ev.clone()))
}

/// Serialize one [`Event`] to its `serde_json::Value` (always a JSON object).
fn event_to_value(ev: &Event) -> Value {
    match ev {
        Event::UserText { text } => json!({ "event": "user_text", "text": text }),
        Event::BrainTextChunk { text, seq, done } => {
            json!({ "event": "brain_text_chunk", "text": text, "seq": seq, "done": done })
        }
        Event::CameraFrame { format, w, h, data, path } => json!({
            "event": "camera_frame", "format": format, "w": w, "h": h,
            "data": data, "path": path,
        }),
        Event::ObjectDetected { dets, labels } => {
            let dets: Vec<Value> = dets.iter().map(|d| json!(d.to_vec())).collect();
            json!({ "event": "object_detected", "dets": dets, "labels": labels })
        }
        Event::UserSynthRequest { text, ref_audio, ref_text, language } => json!({
            "event": "user_synth_request", "text": text,
            "ref_audio": ref_audio, "ref_text": ref_text, "language": language,
        }),
        Event::AudioChunk { pcm_b64, sample_rate, seq, done } => json!({
            "event": "audio_chunk", "pcm_b64": pcm_b64,
            "sample_rate": sample_rate, "seq": seq, "done": done,
        }),
        Event::Cancel => json!({ "event": "cancel" }),
        Event::Ready => json!({ "event": "ready" }),
        Event::Error { message } => json!({ "event": "error", "message": message }),
        Event::Log { message } => json!({ "event": "log", "message": message }),
    }
}

/// Decode one JSONL line into an [`Event`], ignoring any `req_id`. Returns `Err`
/// with a human-readable message on malformed input or an unknown `"event"` tag
/// (never panics).
pub fn decode_line(line: &str) -> Result<Event, String> {
    let v: Value = serde_json::from_str(line.trim()).map_err(|e| e.to_string())?;
    decode_event_value(&v)
}

/// Decode the inner [`Event`] from an already-parsed protocol object.
fn decode_event_value(v: &Value) -> Result<Event, String> {
    let tag = v["event"].as_str().ok_or_else(|| "missing \"event\" tag".to_string())?;
    let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
    let u = |k: &str| v[k].as_u64().unwrap_or_default() as u32;
    match tag {
        "user_text" => Ok(Event::UserText { text: s("text") }),
        "brain_text_chunk" => Ok(Event::BrainTextChunk {
            text: s("text"),
            seq: u("seq"),
            done: v["done"].as_bool().unwrap_or(false),
        }),
        "camera_frame" => Ok(Event::CameraFrame {
            format: {
                let f = s("format");
                if f.is_empty() { "rgb8".to_string() } else { f }
            },
            w: u("w"),
            h: u("h"),
            data: v["data"].as_str().map(|x| x.to_string()),
            path: v["path"].as_str().map(|x| x.to_string()),
        }),
        "object_detected" => {
            let dets = v["dets"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|row| {
                            let mut d = [0.0f32; 6];
                            if let Some(r) = row.as_array() {
                                for (i, x) in r.iter().take(6).enumerate() {
                                    d[i] = x.as_f64().unwrap_or(0.0) as f32;
                                }
                            }
                            d
                        })
                        .collect()
                })
                .unwrap_or_default();
            let labels = v["labels"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            Ok(Event::ObjectDetected { dets, labels })
        }
        "user_synth_request" => Ok(Event::UserSynthRequest {
            text: s("text"),
            ref_audio: v["ref_audio"].as_str().map(|x| x.to_string()),
            ref_text: v["ref_text"].as_str().map(|x| x.to_string()),
            language: v["language"].as_str().map(|x| x.to_string()),
        }),
        "audio_chunk" => Ok(Event::AudioChunk {
            pcm_b64: s("pcm_b64"),
            sample_rate: u("sample_rate"),
            seq: u("seq"),
            done: v["done"].as_bool().unwrap_or(false),
        }),
        "cancel" => Ok(Event::Cancel),
        "ready" => Ok(Event::Ready),
        "error" => Ok(Event::Error { message: s("message") }),
        "log" => Ok(Event::Log { message: s("message") }),
        other => Err(format!("unknown event tag: {other:?}")),
    }
}

/// Decode a [`Event::CameraFrame`]'s pixels into interleaved RGB8.
///
/// Resolution order:
///   1. `data` present → base64-decode, then treat the bytes as a PPM (P6) if
///      they start with the `P6` magic, else as raw RGB8.
///   2. else `path` present → read the file; PPM if it starts with `P6`, else raw.
///
/// Validates that the result has `w*h*3` bytes for raw RGB8. PNG/JPEG (detected by
/// magic bytes) return a clear unsupported error. Returns `Err` (never panics).
pub fn decode_frame(ev: &Event) -> Result<Vec<u8>, String> {
    let Event::CameraFrame { format, w, h, data, path } = ev else {
        return Err("decode_frame: not a camera_frame event".to_string());
    };
    if format != "rgb8" {
        return Err(format!("decode_frame: unsupported format {format:?} (only \"rgb8\")"));
    }
    let bytes: Vec<u8> = if let Some(b64) = data {
        base64::decode(b64)?
    } else if let Some(p) = path {
        std::fs::read(p).map_err(|e| format!("decode_frame: reading {p}: {e}"))?
    } else {
        return Err("decode_frame: camera_frame has neither data nor path".to_string());
    };
    decode_pixels(&bytes, *w, *h)
}

/// Interpret a raw payload (already de-base64'd / read) as RGB8 pixels.
fn decode_pixels(bytes: &[u8], w: u32, h: u32) -> Result<Vec<u8>, String> {
    if bytes.starts_with(b"\x89PNG") {
        return Err("decode_frame: PNG is not supported (no decoder); send rgb8 or PPM".to_string());
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        return Err("decode_frame: JPEG is not supported (no decoder); send rgb8 or PPM".to_string());
    }
    if bytes.starts_with(b"P6") {
        let (px, pw, ph) = ppm::decode_p6(bytes)?;
        if pw != w || ph != h {
            return Err(format!(
                "decode_frame: PPM dims {pw}x{ph} disagree with frame {w}x{h}"
            ));
        }
        return Ok(px);
    }
    let expect = (w as usize) * (h as usize) * 3;
    if bytes.len() != expect {
        return Err(format!(
            "decode_frame: raw rgb8 length {} != w*h*3 = {expect}",
            bytes.len()
        ));
    }
    Ok(bytes.to_vec())
}

/// Dependency-free standard base64 (RFC 4648, `+/` alphabet, `=` padding).
pub mod base64 {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// Encode bytes to a standard base64 string (with padding).
    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[((n >> 6) & 63) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    /// Decode a standard base64 string. Whitespace is ignored; invalid characters
    /// or bad padding return `Err` (never panics).
    pub fn decode(input: &str) -> Result<Vec<u8>, String> {
        // Reverse map: byte value -> 6-bit sextet, 255 = invalid.
        let val = |c: u8| -> Option<u8> {
            match c {
                b'A'..=b'Z' => Some(c - b'A'),
                b'a'..=b'z' => Some(c - b'a' + 26),
                b'0'..=b'9' => Some(c - b'0' + 52),
                b'+' => Some(62),
                b'/' => Some(63),
                _ => None,
            }
        };
        let mut sextets: Vec<u8> = Vec::with_capacity(input.len());
        let mut pad = 0usize;
        for &c in input.as_bytes() {
            match c {
                b' ' | b'\n' | b'\r' | b'\t' => continue,
                b'=' => pad += 1,
                _ => {
                    let v = val(c).ok_or_else(|| format!("base64: invalid char {:?}", c as char))?;
                    if pad > 0 {
                        return Err("base64: data character after padding".to_string());
                    }
                    sextets.push(v);
                }
            }
        }
        let mut out = Vec::with_capacity(sextets.len() / 4 * 3);
        for group in sextets.chunks(4) {
            let n = match group.len() {
                4 => {
                    ((group[0] as u32) << 18)
                        | ((group[1] as u32) << 12)
                        | ((group[2] as u32) << 6)
                        | (group[3] as u32)
                }
                3 => ((group[0] as u32) << 18) | ((group[1] as u32) << 12) | ((group[2] as u32) << 6),
                2 => ((group[0] as u32) << 18) | ((group[1] as u32) << 12),
                _ => return Err("base64: truncated input (orphan sextet)".to_string()),
            };
            out.push((n >> 16) as u8);
            if group.len() >= 3 {
                out.push((n >> 8) as u8);
            }
            if group.len() == 4 {
                out.push(n as u8);
            }
        }
        Ok(out)
    }
}

/// Minimal binary PPM (P6) reader/writer for RGB8 frames.
pub mod ppm {
    /// Encode interleaved RGB8 `px` (`w*h*3` bytes) as a binary P6 PPM.
    pub fn encode_p6(px: &[u8], w: u32, h: u32) -> Vec<u8> {
        let header = format!("P6\n{w} {h}\n255\n");
        let mut out = Vec::with_capacity(header.len() + px.len());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(px);
        out
    }

    /// Decode a binary P6 PPM into `(pixels, w, h)`. Only maxval 255 is supported.
    /// Malformed headers return `Err` (never panics).
    pub fn decode_p6(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
        if !bytes.starts_with(b"P6") {
            return Err("ppm: not a P6 file".to_string());
        }
        // Tokenize the ASCII header: magic, width, height, maxval. Whitespace
        // separated; `#` begins a comment to end of line. The pixel blob starts
        // right after the single whitespace following maxval.
        let mut pos = 2usize;
        let mut tokens: Vec<u32> = Vec::with_capacity(3);
        while tokens.len() < 3 {
            // skip whitespace and comments
            loop {
                if pos >= bytes.len() {
                    return Err("ppm: header ended early".to_string());
                }
                match bytes[pos] {
                    b' ' | b'\n' | b'\r' | b'\t' => pos += 1,
                    b'#' => {
                        while pos < bytes.len() && bytes[pos] != b'\n' {
                            pos += 1;
                        }
                    }
                    _ => break,
                }
            }
            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos == start {
                return Err("ppm: expected an integer in header".to_string());
            }
            let tok: u32 = std::str::from_utf8(&bytes[start..pos])
                .ok()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| "ppm: bad integer in header".to_string())?;
            tokens.push(tok);
        }
        let (w, h, maxval) = (tokens[0], tokens[1], tokens[2]);
        if maxval != 255 {
            return Err(format!("ppm: only maxval 255 supported (got {maxval})"));
        }
        // exactly one whitespace byte separates maxval from the data
        if pos >= bytes.len() {
            return Err("ppm: missing pixel data".to_string());
        }
        pos += 1;
        let need = (w as usize) * (h as usize) * 3;
        let data = &bytes[pos..];
        if data.len() < need {
            return Err(format!(
                "ppm: pixel data too short: have {}, need {need}",
                data.len()
            ));
        }
        Ok((data[..need].to_vec(), w, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<Event> {
        vec![
            Event::UserText { text: "hi there".into() },
            Event::BrainTextChunk { text: "tok".into(), seq: 7, done: false },
            Event::BrainTextChunk { text: "".into(), seq: 8, done: true },
            Event::CameraFrame {
                format: "rgb8".into(),
                w: 2,
                h: 1,
                data: Some("AAAA".into()),
                path: None,
            },
            Event::CameraFrame {
                format: "rgb8".into(),
                w: 4,
                h: 4,
                data: None,
                path: Some("/tmp/frame.ppm".into()),
            },
            Event::ObjectDetected {
                dets: vec![[1.0, 2.0, 3.0, 4.0, 0.9, 5.0], [0.0, 0.0, 1.0, 1.0, 0.5, 0.0]],
                labels: vec!["cat".into(), "dog".into()],
            },
            Event::UserSynthRequest {
                text: "hello world".into(),
                ref_audio: Some("voice.wav".into()),
                ref_text: Some("reference".into()),
                language: Some("english".into()),
            },
            Event::UserSynthRequest {
                text: "no reference".into(),
                ref_audio: None,
                ref_text: None,
                language: None,
            },
            Event::AudioChunk { pcm_b64: "AAAA".into(), sample_rate: 24000, seq: 0, done: false },
            Event::AudioChunk { pcm_b64: "".into(), sample_rate: 24000, seq: 3, done: true },
            Event::Cancel,
            Event::Ready,
            Event::Error { message: "boom".into() },
            Event::Log { message: "hello".into() },
        ]
    }

    #[test]
    fn roundtrip_every_variant() {
        for ev in all_variants() {
            let line = encode_line(&ev);
            assert!(!line.contains('\n'), "encoded line must be single-line: {line}");
            let back = decode_line(&line).unwrap();
            assert_eq!(back, ev, "roundtrip mismatch for {ev:?}");
        }
    }

    #[test]
    fn golden_bytes_user_text_and_chunk() {
        assert_eq!(
            encode_line(&Event::UserText { text: "hi".into() }),
            r#"{"event":"user_text","text":"hi"}"#
        );
        assert_eq!(
            encode_line(&Event::BrainTextChunk { text: "a".into(), seq: 0, done: true }),
            r#"{"done":true,"event":"brain_text_chunk","seq":0,"text":"a"}"#
        );
    }

    #[test]
    fn unknown_tag_errors_not_panics() {
        assert!(decode_line(r#"{"event":"nope"}"#).is_err());
        assert!(decode_line(r#"{"no_event":1}"#).is_err());
        assert!(decode_line("not json at all").is_err());
        // unknown tag still errors cleanly through the envelope path too.
        assert!(decode_envelope(r#"{"req_id":"x","event":"nope"}"#).is_err());
    }

    #[test]
    fn envelope_roundtrip_with_and_without_req_id() {
        for ev in all_variants() {
            // No req_id: round-trips and matches the bare encode_line bytes.
            let bare = Envelope::bare(ev.clone());
            let line = encode_envelope(&bare);
            assert!(!line.contains("req_id"), "bare envelope must omit req_id: {line}");
            assert_eq!(line, encode_line(&ev), "bare envelope must match encode_line");
            assert_eq!(decode_envelope(&line).unwrap(), bare);

            // With req_id: round-trips and the inner event is unchanged.
            let tagged = Envelope::with_id(Some("abc".into()), ev.clone());
            let tline = encode_envelope(&tagged);
            let back = decode_envelope(&tline).unwrap();
            assert_eq!(back, tagged, "tagged roundtrip mismatch");
            assert_eq!(back.req_id.as_deref(), Some("abc"));
            assert_eq!(back.event, ev);
        }
    }

    #[test]
    fn decode_envelope_reads_optional_req_id() {
        let with = decode_envelope(r#"{"req_id":"r1","event":"user_text","text":"hi"}"#).unwrap();
        assert_eq!(with.req_id.as_deref(), Some("r1"));
        assert_eq!(with.event, Event::UserText { text: "hi".into() });

        let without = decode_envelope(r#"{"event":"user_text","text":"hi"}"#).unwrap();
        assert_eq!(without.req_id, None);
        assert_eq!(without.event, Event::UserText { text: "hi".into() });
    }

    #[test]
    fn golden_envelope_bytes_with_req_id() {
        let env = Envelope::with_id(
            Some("r1".into()),
            Event::BrainTextChunk { text: "a".into(), seq: 0, done: true },
        );
        assert_eq!(
            encode_envelope(&env),
            r#"{"done":true,"event":"brain_text_chunk","req_id":"r1","seq":0,"text":"a"}"#
        );
    }

    #[test]
    fn base64_roundtrip_and_known_vectors() {
        assert_eq!(base64::encode(b""), "");
        assert_eq!(base64::encode(b"f"), "Zg==");
        assert_eq!(base64::encode(b"fo"), "Zm8=");
        assert_eq!(base64::encode(b"foo"), "Zm9v");
        assert_eq!(base64::encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64::encode(b"foobar"), "Zm9vYmFy");
        for s in [b"".to_vec(), b"f".to_vec(), b"foobar".to_vec(), (0u8..=255).collect()] {
            assert_eq!(base64::decode(&base64::encode(&s)).unwrap(), s);
        }
    }

    #[test]
    fn base64_decode_errors() {
        assert!(base64::decode("****").is_err()); // invalid chars
        assert!(base64::decode("Zg=v").is_err()); // data after padding
        assert!(base64::decode("Z").is_err()); // orphan sextet
    }

    #[test]
    fn ppm_roundtrip_and_errors() {
        let px: Vec<u8> = vec![10, 20, 30, 40, 50, 60]; // 2x1
        let enc = ppm::encode_p6(&px, 2, 1);
        let (back, w, h) = ppm::decode_p6(&enc).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(back, px);
        // comment + extra whitespace in header still parses
        let with_comment = b"P6\n# a comment\n2 1\n255\n\x0a\x14\x1e\x28\x32\x3c".to_vec();
        let (b2, _, _) = ppm::decode_p6(&with_comment).unwrap();
        assert_eq!(b2, px);
        assert!(ppm::decode_p6(b"P3 2 1 255").is_err()); // not P6
        assert!(ppm::decode_p6(b"P6\n2 1\n128\n").is_err()); // maxval != 255
        assert!(ppm::decode_p6(b"P6\n2 1\n255\n\x01").is_err()); // truncated pixels
    }

    #[test]
    fn decode_frame_inline_rgb8_and_ppm() {
        let px = vec![1u8, 2, 3, 4, 5, 6]; // 2x1 rgb8
        let raw = Event::CameraFrame {
            format: "rgb8".into(),
            w: 2,
            h: 1,
            data: Some(base64::encode(&px)),
            path: None,
        };
        assert_eq!(decode_frame(&raw).unwrap(), px);

        let ppm_bytes = ppm::encode_p6(&px, 2, 1);
        let framed = Event::CameraFrame {
            format: "rgb8".into(),
            w: 2,
            h: 1,
            data: Some(base64::encode(&ppm_bytes)),
            path: None,
        };
        assert_eq!(decode_frame(&framed).unwrap(), px);
    }

    #[test]
    fn decode_frame_errors_not_panics() {
        // wrong raw length
        let bad = Event::CameraFrame {
            format: "rgb8".into(),
            w: 2,
            h: 2,
            data: Some(base64::encode(&[1u8, 2, 3])),
            path: None,
        };
        assert!(decode_frame(&bad).is_err());
        // unsupported format
        let png = Event::CameraFrame {
            format: "png".into(),
            w: 1,
            h: 1,
            data: Some(base64::encode(b"\x89PNG\r\n")),
            path: None,
        };
        assert!(decode_frame(&png).is_err());
        // PNG magic under rgb8 format → clear unsupported error
        let pngraw = Event::CameraFrame {
            format: "rgb8".into(),
            w: 1,
            h: 1,
            data: Some(base64::encode(b"\x89PNG\r\n\x1a\x0a")),
            path: None,
        };
        assert!(decode_frame(&pngraw).is_err());
        // neither data nor path
        let empty =
            Event::CameraFrame { format: "rgb8".into(), w: 1, h: 1, data: None, path: None };
        assert!(decode_frame(&empty).is_err());
    }

    #[test]
    fn decode_frame_from_path() {
        let px = vec![9u8, 8, 7, 6, 5, 4];
        let dir = std::env::temp_dir();
        let path = dir.join(format!("brain_evt_frame_{}.ppm", std::process::id()));
        std::fs::write(&path, ppm::encode_p6(&px, 2, 1)).unwrap();
        let ev = Event::CameraFrame {
            format: "rgb8".into(),
            w: 2,
            h: 1,
            data: None,
            path: Some(path.to_str().unwrap().to_string()),
        };
        assert_eq!(decode_frame(&ev).unwrap(), px);
        std::fs::remove_file(&path).ok();
    }
}
