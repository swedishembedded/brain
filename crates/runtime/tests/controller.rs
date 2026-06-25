// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end controller tests: scripted `feed_line` turns over fake models,
//! asserting the exact ordered event stream. No trained model, no GPU.

use events::{base64, ppm, Envelope, Event};
use runtime::{Controller, FakeDetectModel, FakeInferModel, GenConfig, Registry};

/// Project the inner [`Event`]s out of a returned envelope stream (dropping
/// req_ids) so the existing event-stream assertions read unchanged.
fn events_of(out: &[Envelope]) -> Vec<Event> {
    out.iter().map(|e| e.event.clone()).collect()
}

fn controller_with(text: &str) -> Controller {
    let infer = Box::new(FakeInferModel::echoing(text));
    let detect = Box::new(FakeDetectModel::default());
    let reg = Registry::with_models(infer, detect);
    // greedy + eos=256 (FakeInferModel's terminator); plenty of max_new headroom.
    let cfg = GenConfig { max_new: 64, temperature: 0.0, top_k: 0, eos: Some(256), seed: 0 };
    Controller::with_config(reg, cfg)
}

#[test]
fn user_text_streams_one_chunk_per_token_then_done() {
    let mut ctrl = controller_with("hey");
    let env_out = ctrl.feed_line(r#"{"event":"user_text","text":"go"}"#);
    let out = events_of(&env_out);

    // Expect: chunk("h",0), chunk("e",1), chunk("y",2), terminal done chunk.
    let chunks: Vec<(&str, u32, bool)> = out
        .iter()
        .filter_map(|e| match e {
            Event::BrainTextChunk { text, seq, done } => Some((text.as_str(), *seq, *done)),
            _ => None,
        })
        .collect();

    assert_eq!(
        chunks,
        vec![("h", 0, false), ("e", 1, false), ("y", 2, false), ("", 3, true)],
        "got events: {out:?}"
    );
    // exactly one terminal chunk
    assert_eq!(chunks.iter().filter(|c| c.2).count(), 1);
}

#[test]
fn camera_frame_emits_one_object_detected() {
    let mut ctrl = controller_with("x");
    // 2x1 rgb8 frame inline.
    let px = vec![1u8, 2, 3, 4, 5, 6];
    let frame = Event::CameraFrame {
        format: "rgb8".into(),
        w: 2,
        h: 1,
        data: Some(base64::encode(&px)),
        path: None,
    };
    let out = events_of(&ctrl.feed_line(&events::encode_line(&frame)));

    let dets: Vec<&Event> =
        out.iter().filter(|e| matches!(e, Event::ObjectDetected { .. })).collect();
    assert_eq!(dets.len(), 1, "expected exactly one detection, got {out:?}");
    match dets[0] {
        Event::ObjectDetected { dets, labels } => {
            assert_eq!(dets.len(), 1);
            assert_eq!(dets[0], [10.0, 20.0, 110.0, 220.0, 0.99, 0.0]);
            assert_eq!(labels, &vec!["object".to_string()]);
        }
        _ => unreachable!(),
    }
}

#[test]
fn camera_frame_via_ppm_path() {
    let mut ctrl = controller_with("x");
    let px = vec![9u8, 8, 7, 6, 5, 4];
    let frame = Event::CameraFrame {
        format: "rgb8".into(),
        w: 2,
        h: 1,
        data: Some(base64::encode(&ppm::encode_p6(&px, 2, 1))),
        path: None,
    };
    let out = events_of(&ctrl.feed_line(&events::encode_line(&frame)));
    assert!(out.iter().any(|e| matches!(e, Event::ObjectDetected { .. })), "{out:?}");
}

#[test]
fn cancel_faults_the_machine() {
    let mut ctrl = controller_with("hi");
    let out = events_of(&ctrl.feed_line(r#"{"event":"cancel"}"#));
    assert!(
        out.iter().any(|e| matches!(e, Event::Error { .. })),
        "cancel should emit an error from Faulted: {out:?}"
    );
    // Once faulted, further input is swallowed (terminal sink): no new chunks.
    let out2 = events_of(&ctrl.feed_line(r#"{"event":"user_text","text":"more"}"#));
    assert!(
        !out2.iter().any(|e| matches!(e, Event::BrainTextChunk { .. })),
        "faulted machine must not stream: {out2:?}"
    );
}

#[test]
fn returns_to_idle_and_can_stream_again() {
    let mut ctrl = controller_with("ab");
    let first = events_of(&ctrl.feed_line(r#"{"event":"user_text","text":"q"}"#));
    assert!(first.iter().any(|e| matches!(e, Event::BrainTextChunk { done: true, .. })));
    // A second turn must stream again from seq 0 (we re-create the model each
    // call would differ; here the same fake re-emits its script).
    let second = events_of(&ctrl.feed_line(r#"{"event":"user_text","text":"q"}"#));
    let seqs: Vec<u32> = second
        .iter()
        .filter_map(|e| match e {
            Event::BrainTextChunk { seq, .. } => Some(*seq),
            _ => None,
        })
        .collect();
    assert_eq!(seqs.first(), Some(&0), "second turn restarts seq: {second:?}");
}

#[test]
fn bad_frame_emits_error_not_panic() {
    let mut ctrl = controller_with("x");
    // raw rgb8 wrong length for 2x2
    let frame = Event::CameraFrame {
        format: "rgb8".into(),
        w: 2,
        h: 2,
        data: Some(base64::encode(&[1u8, 2, 3])),
        path: None,
    };
    let out = events_of(&ctrl.feed_line(&events::encode_line(&frame)));
    assert!(out.iter().any(|e| matches!(e, Event::Error { .. })), "{out:?}");
}

#[test]
fn real_tiny_gpt_pumps_without_panicking() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    // Random-weight tiny GPT (no training): the pump must advance tokens and the
    // controller must terminate without panic. We don't assert the text.
    use gpt::{Gpt, GptConfig};
    let cfg = GptConfig::tiny();
    let itos: Vec<char> = (0..cfg.vocab).map(|i| char::from_u32(b'a' as u32 + i).unwrap_or('?')).collect();
    let init = gpt::init_weights(&cfg, 1234);
    let block = cfg.block_size;
    let model = Gpt::new(cfg, 1, block, &init);
    let infer = Box::new(runtime::GptInfer::from_parts(model, Some(itos)));
    let detect = Box::new(FakeDetectModel::default());
    let reg = Registry::with_models(infer, detect);
    // small max_new so the test is cheap; greedy; no eos (stops at max_new).
    let mut ctrl = Controller::with_config(
        reg,
        GenConfig { max_new: 4, temperature: 0.0, top_k: 0, eos: None, seed: 7 },
    );
    let out = events_of(&ctrl.feed_line(r#"{"event":"user_text","text":"ab"}"#));
    let n_chunks = out.iter().filter(|e| matches!(e, Event::BrainTextChunk { .. })).count();
    assert!(n_chunks >= 1, "expected at least one chunk + terminal: {out:?}");
    assert!(out.iter().any(|e| matches!(e, Event::BrainTextChunk { done: true, .. })));
}

#[test]
fn user_text_with_req_id_echoes_it_on_every_chunk() {
    let mut ctrl = controller_with("hey");
    let out = ctrl.feed_line(r#"{"req_id":"abc","event":"user_text","text":"go"}"#);
    // Every emitted chunk (streaming + terminal done) carries req_id "abc".
    let chunks: Vec<&Envelope> =
        out.iter().filter(|e| matches!(e.event, Event::BrainTextChunk { .. })).collect();
    assert!(!chunks.is_empty(), "expected chunks: {out:?}");
    for env in &out {
        assert_eq!(env.req_id.as_deref(), Some("abc"), "every event must echo req_id: {env:?}");
    }
    assert!(out.iter().any(|e| matches!(e.event, Event::BrainTextChunk { done: true, .. })));
}

#[test]
fn camera_frame_with_req_id_echoes_it_on_object_detected() {
    let mut ctrl = controller_with("x");
    let px = vec![1u8, 2, 3, 4, 5, 6];
    let frame = Event::CameraFrame {
        format: "rgb8".into(),
        w: 2,
        h: 1,
        data: Some(base64::encode(&px)),
        path: None,
    };
    // Tag the request by encoding through the envelope.
    let line = events::encode_envelope(&Envelope::with_id(Some("img7".into()), frame));
    let out = ctrl.feed_line(&line);
    let det: Vec<&Envelope> =
        out.iter().filter(|e| matches!(e.event, Event::ObjectDetected { .. })).collect();
    assert_eq!(det.len(), 1, "expected one detection: {out:?}");
    assert_eq!(det[0].req_id.as_deref(), Some("img7"));
}

#[test]
fn successive_requests_get_correctly_tagged_responses() {
    let mut ctrl = controller_with("ab");
    let first = ctrl.feed_line(r#"{"req_id":"r1","event":"user_text","text":"q"}"#);
    let second = ctrl.feed_line(r#"{"req_id":"r2","event":"user_text","text":"q"}"#);
    assert!(first.iter().all(|e| e.req_id.as_deref() == Some("r1")), "{first:?}");
    assert!(second.iter().all(|e| e.req_id.as_deref() == Some("r2")), "{second:?}");
    // And both actually streamed.
    assert!(first.iter().any(|e| matches!(e.event, Event::BrainTextChunk { .. })));
    assert!(second.iter().any(|e| matches!(e.event, Event::BrainTextChunk { .. })));
}

#[test]
fn request_without_req_id_carries_none() {
    let mut ctrl = controller_with("hi");
    let out = ctrl.feed_line(r#"{"event":"user_text","text":"q"}"#);
    assert!(!out.is_empty());
    assert!(out.iter().all(|e| e.req_id.is_none()), "untagged request must carry None: {out:?}");
}
