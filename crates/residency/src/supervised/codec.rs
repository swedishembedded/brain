// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for carrying model requests,
// progress and media between a server and its isolated workers for its
// clients. If your team needs expertise in inter-process protocols for model
// serving then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The JSON a hosted model's frames carry, both ends: run envelopes, answers
//! and progress updates, with their blobs in sealed memfds beside the frame
//! and a text too long for a line moved into a descriptor of its own.

use std::collections::{BTreeMap, VecDeque};
use std::os::fd::OwnedFd;

use capability::{ActionResult, Blob, Media, Outcome, Progress};
use serde_json::{json, Map, Value};

/// A JSON text for a frame: inline when it fits a line, else in a sealed
/// memfd put first among `fds`, with `-` on the line.
pub(super) fn pack(text: String, fds: &mut Vec<OwnedFd>) -> Result<String, String> {
    // Room for the id, the verb and the action on the same line.
    if text.len() + 4096 < gpu_supervisor::MAX_REQUEST_BYTES.min(gpu_supervisor::MAX_REPLY_BYTES) {
        return Ok(text);
    }
    if fds.len() >= gpu_supervisor::wire::MAX_FDS {
        return Err(format!("more than {} blobs in one frame", gpu_supervisor::wire::MAX_FDS - 1));
    }
    fds.insert(0, gpu_supervisor::wire::seal("envelope", text.as_bytes()).map_err(|e| format!("envelope: {e}"))?);
    Ok("-".to_string())
}

/// The JSON a frame carries: its text, or for `-` the first of `fds`.
pub(super) fn unpack(text: &str, fds: &mut VecDeque<OwnedFd>) -> Result<Value, String> {
    let owned;
    let text = if text == "-" {
        let fd = fds.pop_front().ok_or("an envelope in a descriptor that did not arrive")?;
        let bytes = gpu_supervisor::wire::read_sealed(&fd, gpu_supervisor::wire::MAX_BLOB_BYTES).map_err(|e| format!("envelope: {e}"))?;
        owned = String::from_utf8(bytes).map_err(|_| "an envelope that is not UTF-8")?;
        owned.as_str()
    } else {
        text
    };
    serde_json::from_str(text).map_err(|e| format!("not JSON ({e})"))
}

/// The keys of a JSON object, checked against the ones a record allows.
pub(super) fn object<'a>(v: &'a Value, what: &str, allowed: &[&str]) -> Result<&'a Map<String, Value>, String> {
    let map = v.as_object().ok_or_else(|| format!("{what} is not an object"))?;
    if let Some(k) = map.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(format!("{what} has an unknown field {k:?}"));
    }
    Ok(map)
}

/// Blobs described in `list`, their bytes read from `fds` in order.
pub(super) fn read_blobs(list: Option<&Value>, fds: &mut VecDeque<OwnedFd>) -> Result<BTreeMap<String, Blob>, String> {
    let mut out = BTreeMap::new();
    let Some(list) = list else { return Ok(out) };
    for d in list.as_array().ok_or("blobs is not a list")? {
        let d = object(d, "a blob", &["name", "media", "meta"])?;
        let name = d.get("name").and_then(Value::as_str).ok_or("a blob without a name")?;
        let media = d.get("media").and_then(Value::as_str).and_then(Media::parse).ok_or_else(|| format!("blob {name:?} has no known media"))?;
        let fd = fds.pop_front().ok_or_else(|| format!("blob {name:?} has no descriptor"))?;
        let bytes = gpu_supervisor::wire::read_sealed(&fd, gpu_supervisor::wire::MAX_BLOB_BYTES).map_err(|e| format!("blob {name:?}: {e}"))?;
        let meta = d.get("meta").cloned().unwrap_or(Value::Null);
        if out.insert(name.to_string(), Blob { media, bytes, meta }).is_some() {
            return Err(format!("blob {name:?} appears twice"));
        }
    }
    Ok(out)
}

/// Blobs as the list a frame describes them in, and their descriptors.
pub(super) fn write_blobs<'a>(blobs: impl IntoIterator<Item = (&'a String, &'a Blob)>, fds: &mut Vec<OwnedFd>) -> Result<Vec<Value>, String> {
    blobs
        .into_iter()
        .map(|(name, blob)| {
            fds.push(gpu_supervisor::wire::seal(name, &blob.bytes).map_err(|e| format!("blob {name:?}: {e}"))?);
            Ok(json!({ "name": name, "media": blob.media.name(), "meta": blob.meta }))
        })
        .collect()
}

pub(super) fn no_leftovers(fds: &VecDeque<OwnedFd>) -> Result<(), String> {
    if fds.is_empty() {
        Ok(())
    } else {
        Err(format!("{} descriptor(s) that no blob names", fds.len()))
    }
}

/// An `ok` answer's outcome.
pub(super) fn decode_outcome(payload: &str, fds: Vec<OwnedFd>) -> ActionResult {
    let mut fds = VecDeque::from(fds);
    let v = unpack(payload, &mut fds)?;
    let map = object(&v, "the answer", &["outputs", "blobs"])?;
    let outputs = map.get("outputs").cloned().unwrap_or_else(|| json!({}));
    if !outputs.is_object() {
        return Err("the outputs are not a JSON object".into());
    }
    let blobs = read_blobs(map.get("blobs"), &mut fds)?;
    no_leftovers(&fds)?;
    Ok(Outcome { outputs, blobs })
}

/// A progress frame's update.
pub(super) fn decode_progress(payload: &str, fds: Vec<OwnedFd>) -> Result<Progress, String> {
    let mut fds = VecDeque::from(fds);
    let v = unpack(payload, &mut fds)?;
    let map = object(&v, "a progress frame", &["step", "total", "message", "delta", "event", "chunk"])?;
    let count = |k: &str| map.get(k).map_or(Ok(0), |n| n.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("{k} is not a count")));
    let text = |k: &str| map.get(k).map(|t| t.as_str().map(str::to_string).ok_or_else(|| format!("{k} is not text"))).transpose();
    let chunk = match map.get("chunk") {
        Some(c) => read_blobs(Some(&json!([c])), &mut fds)?.into_iter().next(),
        None => None,
    };
    no_leftovers(&fds)?;
    Ok(Progress { step: count("step")?, total: count("total")?, message: text("message")?.unwrap_or_default(), delta: text("delta")?, event: map.get("event").cloned(), chunk })
}

pub(super) fn encode_progress(p: &Progress, fds: &mut Vec<OwnedFd>) -> Result<String, String> {
    let mut v = json!({ "step": p.step, "total": p.total, "message": p.message });
    if let Some(d) = &p.delta {
        v["delta"] = json!(d);
    }
    if let Some(e) = &p.event {
        v["event"] = e.clone();
    }
    if let Some((name, blob)) = &p.chunk {
        v["chunk"] = write_blobs([(name, blob)], fds)?.remove(0);
    }
    pack(v.to_string(), fds)
}


#[cfg(test)]
mod tests {
    use super::*;
    use capability::Media;

    /// A progress update and an outcome survive the trip through their frame
    /// encoding, blobs included, and an envelope too long for a line goes in
    /// a descriptor.
    #[test]
    fn progress_outcomes_and_long_envelopes_cross_the_frame_encoding() {
        let chunk = Blob::new(Media::Audio, vec![1, 2, 3]).with_meta(json!({"rate": 16000}));
        let p = Progress { step: 2, total: 5, message: "decoding".into(), delta: Some("he".into()), event: Some(json!({"k": 1})), chunk: Some(("clip".into(), chunk)) };
        let mut fds = Vec::new();
        let payload = encode_progress(&p, &mut fds).unwrap();
        let back = decode_progress(&payload, fds).unwrap();
        assert_eq!((back.step, back.total, back.message.as_str(), back.delta.as_deref()), (2, 5, "decoding", Some("he")));
        assert_eq!(back.event, Some(json!({"k": 1})));
        let (name, blob) = back.chunk.unwrap();
        assert_eq!((name.as_str(), blob.media, blob.bytes, blob.meta), ("clip", Media::Audio, vec![1, 2, 3], json!({"rate": 16000})));

        let long = "x".repeat(gpu_supervisor::MAX_REQUEST_BYTES);
        let mut fds = Vec::new();
        let text = pack(json!({ "outputs": { "text": long }, "blobs": [] }).to_string(), &mut fds).unwrap();
        assert_eq!(text, "-");
        let outcome = decode_outcome(&text, fds).unwrap();
        assert_eq!(outcome.outputs["text"].as_str().map(str::len), Some(gpu_supervisor::MAX_REQUEST_BYTES));
        assert!(decode_outcome(r#"{"outputs": {}, "extra": 1}"#, Vec::new()).is_err(), "an unknown field is refused");
    }
}
