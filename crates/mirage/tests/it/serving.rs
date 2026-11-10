// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served actions end to end on the released weights: a session answers
//! `describe` and `extract` like the model does, merges concurrent requests
//! into one device batch without changing any answer, reuses a decoder memory
//! between the two actions, and drops cancelled requests.

use std::sync::OnceLock;

use capability::Invocation;
use mirage::caps::{MirageSession, PREPROCESSING_VERSION};
use mirage::describe::Options;
use mirage::device::{Device, Image};
use serde_json::{json, Value};

use crate::common::released_dir;

const FORM: &str = r#"{"animal": {"enum": ["cat", "dog", "bird"]}, "count": {"type": "integer", "minimum": 0, "maximum": 6}}"#;

/// The cats photograph of the tracked goldens: `(rgb, height, width)`.
fn cats() -> Option<(Vec<u8>, usize, usize)> {
    let g = crate::decoder::golden("serving", "cats")?;
    Some((g["input_rgb"].iter().map(|&v| v as u8).collect(), 480, 640))
}

fn session() -> Option<&'static MirageSession> {
    static CELL: OnceLock<Option<MirageSession>> = OnceLock::new();
    CELL.get_or_init(|| {
        let dir = released_dir()?;
        Some(MirageSession::load(dir.to_str()?, gpu_core::Gpu::new(&Device::kernels())).expect("the released checkpoint loads"))
    })
    .as_ref()
}

fn blob(rgb: &[u8], h: usize, w: usize) -> capability::Blob {
    let hwc: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
    capability::blob::image_blob(&hwc, w as u32, h as u32, 3)
}

fn describe_inv(rgb: &[u8]) -> Invocation {
    Invocation::new().blob("image", blob(rgb, 480, 640)).set("mask_format", json!("bitmap")).set("max_tokens", json!(600))
}

fn extract_inv(rgb: &[u8]) -> Invocation {
    Invocation::new().blob("image", blob(rgb, 480, 640)).set("form", json!(FORM))
}

/// The outputs of an outcome without the accounting.
fn answer(outcome: &capability::Outcome) -> Value {
    let mut v = outcome.outputs.clone();
    v.as_object_mut().unwrap().remove("usage");
    v
}

#[test]
fn describe_answers_like_the_model_and_reports_usage() {
    let (Some(session), Some((rgb, h, w))) = (session(), cats()) else {
        brain_testutil::skip("serving describe: the released checkpoint or the cats golden is absent");
        return;
    };
    let served = session.run_batch("describe", &[describe_inv(&rgb)]).pop().unwrap().expect("describes");
    let opts = Options::from_json(r#"{"mask_format":"bitmap","max_tokens":600}"#).unwrap();
    let direct = session.model().describe_batch(&[Image { rgb: &rgb, height: h, width: w }], &opts).pop().unwrap().unwrap();
    assert_eq!(answer(&served), serde_json::to_value(&direct).unwrap(), "the served sections are the model's result");
    let usage = &served.outputs["usage"];
    assert_eq!(usage["images"], 1);
    assert!(usage["decoder_tokens"].as_u64().unwrap() > 20, "{usage}");
    for k in ["vision_ms", "decode_ms", "post_ms", "batch"] {
        assert!(usage[k].is_number(), "usage.{k} in {usage}");
    }
}

/// Sixteen identical-option requests are one device batch and each answer is the single-request answer.
#[test]
fn concurrent_requests_share_one_batch_and_keep_their_answers() {
    let (Some(session), Some((rgb, _, _))) = (session(), cats()) else {
        brain_testutil::skip("serving batch: the released checkpoint or the cats golden is absent");
        return;
    };
    let alone = answer(&session.run_batch("describe", &[describe_inv(&rgb)]).pop().unwrap().unwrap());
    let before = session.stats();
    // Different images in one batch: the 16 requests alternate between the photograph and a darker copy of it.
    let dark: Vec<u8> = rgb.iter().map(|&b| b / 2).collect();
    let invs: Vec<Invocation> = (0..16).map(|i| describe_inv(if i % 2 == 0 { &rgb } else { &dark })).collect();
    let results = session.run_batch("describe", &invs);
    let after = session.stats();
    assert_eq!(after.batches - before.batches, 1, "16 requests with one set of options are one device batch");
    assert_eq!(after.sizes.last(), Some(&16));
    assert!(after.widest >= 16);
    let dark_alone = answer(&session.run_batch("describe", &[describe_inv(&dark)]).pop().unwrap().unwrap());
    for (i, r) in results.iter().enumerate() {
        let got = answer(r.as_ref().expect("describes"));
        assert_eq!(got, if i % 2 == 0 { alone.clone() } else { dark_alone.clone() }, "request {i} equals its single-request answer");
        assert_eq!(r.as_ref().unwrap().outputs["usage"]["batch"], 16);
    }
}

/// Requests with different options are separate calls of one batch; each is answered by its own options.
#[test]
fn different_options_run_as_separate_groups() {
    let (Some(session), Some((rgb, _, _))) = (session(), cats()) else {
        brain_testutil::skip("serving groups: the released checkpoint or the cats golden is absent");
        return;
    };
    let objects_only = describe_inv(&rgb).set("sections", json!("objects"));
    let results = session.run_batch("describe", &[describe_inv(&rgb), objects_only]);
    let (full, only) = (results[0].as_ref().unwrap(), results[1].as_ref().unwrap());
    assert!(full.outputs.get("scene").is_some() || full.outputs.get("objects").is_some());
    assert!(only.outputs.get("scene").is_none() && only.outputs.get("text").is_none(), "only the objects section is returned: {}", answer(only));
}

#[test]
fn extract_reuses_the_memory_a_describe_left_and_gives_the_same_answer() {
    let (Some(session), Some((rgb, _, _))) = (session(), cats()) else {
        brain_testutil::skip("serving reuse: the released checkpoint or the cats golden is absent");
        return;
    };
    // A fresh image content so nothing earlier in this process cached it.
    let fresh: Vec<u8> = rgb.iter().map(|&b| b.saturating_sub(7)).collect();
    let (_, bytes_before) = session.cached_memories();
    let cold = session.run_batch("extract", &[extract_inv(&fresh)]).pop().unwrap().unwrap();
    assert_eq!(cold.outputs["usage"]["memory_reused"], false);
    let hits = session.stats().memory_hits;
    let warm = session.run_batch("extract", &[extract_inv(&fresh)]).pop().unwrap().unwrap();
    assert_eq!(warm.outputs["usage"]["memory_reused"], true, "the second extract of the same image skips the vision half");
    assert_eq!(session.stats().memory_hits, hits + 1);
    assert_eq!(answer(&cold), answer(&warm));
    assert!(session.cached_memories().1 > bytes_before);

    // Describe leaves its memory behind for an extract under the same proposal options.
    let other: Vec<u8> = rgb.iter().map(|&b| b.saturating_sub(13)).collect();
    session.run_batch("describe", &[describe_inv(&other)]).pop().unwrap().unwrap();
    let after_describe = session.run_batch("extract", &[extract_inv(&other)]).pop().unwrap().unwrap();
    assert_eq!(after_describe.outputs["usage"]["memory_reused"], true, "describe populated the cache");
    let direct = session.model().extract_batch(&[Image { rgb: &other, height: 480, width: 640 }], &[serde_json::from_str(FORM).unwrap()], &Options::default()).pop().unwrap().unwrap();
    assert_eq!(answer(&after_describe)["form"], serde_json::to_value(&direct).unwrap()["form"], "reuse does not change the answer");

    // Other proposal options are a different key: no reuse.
    let tighter = extract_inv(&other).set("conf", json!(0.6));
    assert_eq!(session.run_batch("extract", &[tighter]).pop().unwrap().unwrap().outputs["usage"]["memory_reused"], false);
    assert_eq!(PREPROCESSING_VERSION, 1, "bumping the preprocessing version invalidates cached memories (the key carries it)");
}

#[test]
fn a_cancelled_request_fails_alone_and_does_not_join_the_batch() {
    let (Some(session), Some((rgb, _, _))) = (session(), cats()) else {
        brain_testutil::skip("serving cancel: the released checkpoint or the cats golden is absent");
        return;
    };
    let mut doomed = describe_inv(&rgb);
    doomed.cancel = capability::CancelToken::armed();
    doomed.cancel.cancel();
    let before = session.stats();
    let results = session.run_batch("describe", &[doomed, describe_inv(&rgb)]);
    assert_eq!(results[0].as_ref().unwrap_err(), "cancelled");
    assert!(results[1].is_ok(), "the other request is unaffected");
    let after = session.stats();
    assert_eq!(after.images - before.images, 1, "the cancelled request took no slot in the batch");

    // Cancelled while the batch decodes: the request still fails, the others finish.
    let mut late = describe_inv(&rgb);
    late.cancel = capability::CancelToken::armed();
    let flip = late.cancel.clone();
    let invs = vec![late, describe_inv(&rgb)];
    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            flip.cancel();
        });
        let results = session.run_batch("describe", &invs);
        assert!(results[1].is_ok());
        // The flag may land before or after the last window; either way the answer is an error or a full result.
        if let Err(e) = &results[0] {
            assert_eq!(e, "cancelled");
        }
    });
}

#[test]
fn bad_requests_fail_alone_with_the_field_named() {
    let (Some(session), Some((rgb, _, _))) = (session(), cats()) else {
        brain_testutil::skip("serving errors: the released checkpoint or the cats golden is absent");
        return;
    };
    let bad = describe_inv(&rgb).set("sections", json!("objects,nonsense"));
    let results = session.run_batch("describe", &[bad, describe_inv(&rgb)]);
    let err = results[0].as_ref().unwrap_err();
    assert!(err.contains("sections") || err.contains("nonsense"), "{err}");
    assert!(results[1].is_ok());
    let boxy = Invocation::new().blob("image", blob(&rgb, 480, 640)).set("form", json!(r#"{"b": {"type": "string", "format": "box"}}"#));
    assert!(session.run_batch("extract", &[boxy]).pop().unwrap().is_err());
}
