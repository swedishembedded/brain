// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for carrying model requests,
// progress and media between a server and its isolated workers for its
// clients. If your team needs expertise in inter-process protocols for model
// serving then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The JSON a hosted model's frames carry, both ends: job envelopes, results
//! and progress updates.
//!
//! A frame's blobs travel in one sealed memfd beside it (a bundle: their
//! bytes one after another, each blob's length in the JSON), so a group of
//! any size crosses in one frame with at most two descriptors: the bundle,
//! and first, a JSON text too long for a line, with `-` on the line.

use std::collections::{BTreeMap, VecDeque};
use std::os::fd::OwnedFd;

use capability::{ActionResult, Blob, CancelToken, Invocation, Media, Outcome, Progress};
use serde_json::{json, Map, Value};

/// A JSON text for a frame: inline when it fits a line, else in a sealed
/// memfd put first among `fds`, with `-` on the line.
pub(super) fn pack(text: String, fds: &mut Vec<OwnedFd>) -> Result<String, String> {
    // Room for the id, the verb and the action on the same line.
    if text.len() + 4096 < gpu_supervisor::MAX_REQUEST_BYTES.min(gpu_supervisor::MAX_REPLY_BYTES) {
        return Ok(text);
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

/// The blobs of one frame on their way out: described in its JSON, their
/// bytes gathered for one bundle.
#[derive(Default)]
pub(super) struct BlobsOut<'a> {
    parts: Vec<&'a [u8]>,
}

impl<'a> BlobsOut<'a> {
    /// `blobs` as the list a frame describes them in.
    pub(super) fn add(&mut self, blobs: impl IntoIterator<Item = (&'a String, &'a Blob)>) -> Vec<Value> {
        blobs
            .into_iter()
            .map(|(name, blob)| {
                self.parts.push(&blob.bytes);
                json!({ "name": name, "media": blob.media.name(), "meta": blob.meta, "len": blob.bytes.len() })
            })
            .collect()
    }

    /// The bundle's descriptor appended to `fds`, if there is anything in it.
    pub(super) fn seal(self, fds: &mut Vec<OwnedFd>) -> Result<(), String> {
        if !self.parts.is_empty() {
            fds.push(gpu_supervisor::wire::seal_parts("blobs", &self.parts).map_err(|e| format!("blobs: {e}"))?);
        }
        Ok(())
    }
}

/// The blobs of one frame on their way in: the bundle's bytes, handed out in
/// the order the frame lists them.
pub(super) struct BlobsIn {
    bytes: Vec<u8>,
    at: usize,
}

impl BlobsIn {
    /// The bundle among the frame's descriptors left after its JSON: none, or
    /// exactly one.
    pub(super) fn take(mut fds: VecDeque<OwnedFd>) -> Result<BlobsIn, String> {
        let bytes = match fds.pop_front() {
            Some(fd) => gpu_supervisor::wire::read_sealed(&fd, gpu_supervisor::wire::MAX_BLOB_BYTES).map_err(|e| format!("blobs: {e}"))?,
            None => Vec::new(),
        };
        if !fds.is_empty() {
            return Err(format!("{} descriptor(s) beside the blobs that no blob names", fds.len()));
        }
        Ok(BlobsIn { bytes, at: 0 })
    }

    /// The blobs `list` describes.
    pub(super) fn read(&mut self, list: Option<&Value>) -> Result<BTreeMap<String, Blob>, String> {
        let mut out = BTreeMap::new();
        let Some(list) = list else { return Ok(out) };
        for d in list.as_array().ok_or("blobs is not a list")? {
            let (name, blob) = self.one(d)?;
            if out.insert(name.clone(), blob).is_some() {
                return Err(format!("blob {name:?} appears twice"));
            }
        }
        Ok(out)
    }

    fn one(&mut self, d: &Value) -> Result<(String, Blob), String> {
        let d = object(d, "a blob", &["name", "media", "meta", "len"])?;
        let name = d.get("name").and_then(Value::as_str).ok_or("a blob without a name")?;
        let media = d.get("media").and_then(Value::as_str).and_then(Media::parse).ok_or_else(|| format!("blob {name:?} has no known media"))?;
        let len = d.get("len").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()).ok_or_else(|| format!("blob {name:?} has no length"))?;
        let end = self.at.checked_add(len).filter(|&e| e <= self.bytes.len()).ok_or_else(|| format!("blob {name:?} runs past the bundle"))?;
        let bytes = self.bytes[self.at..end].to_vec();
        self.at = end;
        let meta = d.get("meta").cloned().unwrap_or(Value::Null);
        Ok((name.to_string(), Blob { media, bytes, meta }))
    }

    /// Every byte of the bundle was named by a blob.
    pub(super) fn done(&self) -> Result<(), String> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(format!("{} byte(s) of blobs that no blob names", self.bytes.len() - self.at))
        }
    }
}

/// The frame text and descriptors for a group of jobs: `{"jobs": [{params,
/// scope, blobs}, ...]}`.
pub(super) fn encode_jobs(invs: &[Invocation], fds: &mut Vec<OwnedFd>) -> Result<String, String> {
    let mut blobs = BlobsOut::default();
    let jobs: Vec<Value> = invs.iter().map(|inv| json!({ "params": inv.params, "scope": inv.scope, "blobs": blobs.add(&inv.blobs) })).collect();
    blobs.seal(fds)?;
    pack(json!({ "jobs": jobs }).to_string(), fds)
}

/// The jobs [`encode_jobs`] wrote, each with the cancellation token the
/// worker registered for it, in order.
pub(super) fn decode_jobs(text: &str, fds: Vec<OwnedFd>, tokens: Vec<CancelToken>) -> Result<Vec<Invocation>, String> {
    let mut fds = VecDeque::from(fds);
    let v = unpack(text, &mut fds)?;
    let mut blobs = BlobsIn::take(fds)?;
    let jobs = object(&v, "the jobs", &["jobs"])?.get("jobs").and_then(Value::as_array).ok_or("the jobs are not a list")?;
    if jobs.len() != tokens.len() {
        return Err(format!("{} jobs where the frame announced {}", jobs.len(), tokens.len()));
    }
    let invs = jobs
        .iter()
        .zip(tokens)
        .map(|(job, cancel)| {
            let map = object(job, "a job", &["params", "scope", "blobs"])?;
            let params = map.get("params").cloned().unwrap_or_else(|| json!({}));
            let scope = map.get("scope").and_then(Value::as_str).unwrap_or("").to_string();
            Ok(Invocation { params, blobs: blobs.read(map.get("blobs"))?, cancel, scope })
        })
        .collect::<Result<Vec<_>, String>>()?;
    blobs.done()?;
    Ok(invs)
}

/// One job's result as JSON, its blobs added to `blobs`.
fn result_json<'a>(result: &'a ActionResult, blobs: &mut BlobsOut<'a>) -> Value {
    match result {
        Ok(o) => json!({ "outputs": o.outputs, "blobs": blobs.add(&o.blobs) }),
        Err(e) => json!({ "error": e }),
    }
}

fn result_from(v: &Value, blobs: &mut BlobsIn, extra: &[&str]) -> Result<ActionResult, String> {
    let allowed: Vec<&str> = ["outputs", "blobs", "error"].into_iter().chain(extra.iter().copied()).collect();
    let map = object(v, "a result", &allowed)?;
    if let Some(e) = map.get("error") {
        return Ok(Err(e.as_str().ok_or("an error that is not text")?.to_string()));
    }
    let outputs = map.get("outputs").cloned().unwrap_or_else(|| json!({}));
    if !outputs.is_object() {
        return Err("the outputs are not a JSON object".into());
    }
    Ok(Ok(Outcome { outputs, blobs: blobs.read(map.get("blobs"))? }))
}

/// The answer to a group: `{"results": [...]}`, one per job.
pub(super) fn encode_results(results: &[ActionResult], fds: &mut Vec<OwnedFd>) -> Result<String, String> {
    let mut blobs = BlobsOut::default();
    let list: Vec<Value> = results.iter().map(|r| result_json(r, &mut blobs)).collect();
    blobs.seal(fds)?;
    pack(json!({ "results": list }).to_string(), fds)
}

/// The results [`encode_results`] wrote. A worker that broke the protocol is
/// the outer error; a job that failed is its own `Err`.
pub(super) fn decode_results(payload: &str, fds: Vec<OwnedFd>) -> Result<Vec<ActionResult>, String> {
    let mut fds = VecDeque::from(fds);
    let v = unpack(payload, &mut fds)?;
    let mut blobs = BlobsIn::take(fds)?;
    let list = object(&v, "the answer", &["results"])?.get("results").and_then(Value::as_array).ok_or("the results are not a list")?;
    let results = list.iter().map(|r| result_from(r, &mut blobs, &[])).collect::<Result<Vec<_>, String>>()?;
    blobs.done()?;
    Ok(results)
}

/// A progress frame's job and update.
pub(super) fn decode_progress(payload: &str, fds: Vec<OwnedFd>) -> Result<(usize, Progress), String> {
    let mut fds = VecDeque::from(fds);
    let v = unpack(payload, &mut fds)?;
    let mut blobs = BlobsIn::take(fds)?;
    let map = object(&v, "a progress frame", &["job", "step", "total", "message", "delta", "event", "chunk"])?;
    let job = map.get("job").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()).ok_or("a progress frame names no job")?;
    let count = |k: &str| map.get(k).map_or(Ok(0), |n| n.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("{k} is not a count")));
    let text = |k: &str| map.get(k).map(|t| t.as_str().map(str::to_string).ok_or_else(|| format!("{k} is not text"))).transpose();
    let chunk = map.get("chunk").map(|c| blobs.one(c)).transpose()?;
    blobs.done()?;
    Ok((job, Progress { step: count("step")?, total: count("total")?, message: text("message")?.unwrap_or_default(), delta: text("delta")?, event: map.get("event").cloned(), chunk }))
}

/// Job `job`'s progress update as a frame's payload.
pub(super) fn encode_progress(job: usize, p: &Progress, fds: &mut Vec<OwnedFd>) -> Result<String, String> {
    let mut v = json!({ "job": job, "step": p.step, "total": p.total, "message": p.message });
    if let Some(d) = &p.delta {
        v["delta"] = json!(d);
    }
    if let Some(e) = &p.event {
        v["event"] = e.clone();
    }
    let mut blobs = BlobsOut::default();
    if let Some((name, blob)) = &p.chunk {
        v["chunk"] = blobs.add([(name, blob)]).remove(0);
    }
    blobs.seal(fds)?;
    pack(v.to_string(), fds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Media;

    /// A progress update and a group's results survive the trip through their
    /// frame encoding, blobs included, and an answer too long for a line goes
    /// in a descriptor.
    #[test]
    fn progress_results_and_long_answers_cross_the_frame_encoding() {
        let chunk = Blob::new(Media::Audio, vec![1, 2, 3]).with_meta(json!({"rate": 16000}));
        let p = Progress { step: 2, total: 5, message: "decoding".into(), delta: Some("he".into()), event: Some(json!({"k": 1})), chunk: Some(("clip".into(), chunk)) };
        let mut fds = Vec::new();
        let payload = encode_progress(3, &p, &mut fds).unwrap();
        let (job, back) = decode_progress(&payload, fds).unwrap();
        assert_eq!((job, back.step, back.total, back.message.as_str(), back.delta.as_deref()), (3, 2, 5, "decoding", Some("he")));
        assert_eq!(back.event, Some(json!({"k": 1})));
        let (name, blob) = back.chunk.unwrap();
        assert_eq!((name.as_str(), blob.media, blob.bytes, blob.meta), ("clip", Media::Audio, vec![1, 2, 3], json!({"rate": 16000})));

        let long = "x".repeat(gpu_supervisor::MAX_REQUEST_BYTES);
        let results: Vec<ActionResult> = vec![
            Ok(Outcome::new().set("text", json!(long)).blob("a", Blob::new(Media::Bytes, vec![9; 5]))),
            Err("job 1 failed".into()),
            Ok(Outcome::new().blob("b", Blob::new(Media::Image, vec![7; 3]))),
        ];
        let mut fds = Vec::new();
        let text = encode_results(&results, &mut fds).unwrap();
        assert_eq!((text.as_str(), fds.len()), ("-", 2), "the answer and the blobs each in one descriptor");
        let back = decode_results(&text, fds).unwrap();
        assert_eq!(back[0].as_ref().unwrap().outputs["text"].as_str().map(str::len), Some(gpu_supervisor::MAX_REQUEST_BYTES));
        assert_eq!(back[0].as_ref().unwrap().blobs["a"].bytes, vec![9; 5]);
        assert_eq!(back[1].as_ref().err().map(String::as_str), Some("job 1 failed"));
        assert_eq!(back[2].as_ref().unwrap().blobs["b"].bytes, vec![7; 3]);
        assert!(decode_results(r#"{"results": [{"outputs": {}, "extra": 1}]}"#, Vec::new()).is_err(), "an unknown field is refused");
        assert!(decode_results(r#"{"results": [{"blobs": [{"name": "x", "media": "image", "len": 4}]}]}"#, Vec::new()).is_err(), "a blob with no bytes behind it is refused");
    }
}
