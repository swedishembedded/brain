// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end controller tests: scripted `feed_line` turns over fake models,
//! asserting the exact ordered event stream. No trained model, no GPU.

use events::{base64, ppm, Envelope, Event};
use runtime::{
    Controller, FakeDetectModel, FakeInferModel, FakeSynthModel, GenConfig, Registry,
};

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
fn cancel_is_recoverable_not_terminal() {
    let mut ctrl = controller_with("hi");
    let out = events_of(&ctrl.feed_line(r#"{"event":"cancel"}"#));
    // Cancel now emits a terminal `cancelled` ack, NOT a fault.
    assert!(
        out.iter().any(|e| matches!(e, Event::Cancelled)),
        "cancel should emit a recoverable `cancelled` ack: {out:?}"
    );
    assert!(
        !out.iter().any(|e| matches!(e, Event::Error { .. })),
        "cancel must not fault: {out:?}"
    );
    // The session is still alive: a following request streams normally.
    let out2 = events_of(&ctrl.feed_line(r#"{"event":"user_text","text":"more"}"#));
    assert!(
        out2.iter().any(|e| matches!(e, Event::BrainTextChunk { done: true, .. })),
        "controller must keep serving after a cancel: {out2:?}"
    );
}

/// A [`Control`] that lets `n` chunks through, then requests cancel forever.
struct CancelAfter(usize);
impl runtime::Control for CancelAfter {
    fn poll(&mut self) -> Option<Event> {
        if self.0 == 0 {
            Some(Event::Cancel)
        } else {
            self.0 -= 1;
            None
        }
    }
}

#[test]
fn cancel_mid_stream_stops_early_and_recovers() {
    let mut ctrl = synth_controller();
    // ~100 chars -> ~100_000 samples -> ~5 chunks at 24_000/chunk if run to end.
    let text = "the quick brown fox jumps over the lazy dog, then the lazy dog jumps over the quick brown fox again!";
    let line = events::encode_line(&Event::UserSynthRequest {
        text: text.into(),
        ref_audio: None,
        ref_text: None,
        language: None,
    });
    let mut out: Vec<Envelope> = Vec::new();
    // Cancel after exactly 2 chunks have streamed.
    ctrl.feed_line_streaming(&line, &mut out, &mut CancelAfter(2));
    let evs = events_of(&out);

    let n_chunks = evs.iter().filter(|e| matches!(e, Event::AudioChunk { .. })).count();
    assert_eq!(n_chunks, 2, "cancel must pre-empt after exactly 2 chunks: {evs:?}");
    // No terminal done:true chunk — the stream was interrupted, not completed.
    assert!(
        !evs.iter().any(|e| matches!(e, Event::AudioChunk { done: true, .. })),
        "interrupted stream must not emit a normal terminal chunk: {evs:?}"
    );
    // Instead it ends with the recoverable `cancelled` ack.
    assert!(matches!(evs.last(), Some(Event::Cancelled)), "must end with cancelled: {evs:?}");

    // And the controller recovered: a fresh synth turn runs to completion.
    let again = events_of(&ctrl.feed_line(&line));
    assert!(
        again.iter().any(|e| matches!(e, Event::AudioChunk { done: true, .. })),
        "controller must stream again after a mid-stream cancel: {again:?}"
    );
}

#[test]
fn streaming_sink_matches_buffered_output() {
    // The streaming path with a Vec sink and no control must produce exactly the
    // same envelope sequence as the buffered feed_line — streaming is a superset,
    // not a behavior change, when nothing cancels.
    let mut a = controller_with("hello");
    let buffered = a.feed_line(r#"{"req_id":"r1","event":"user_text","text":"go"}"#);

    let mut b = controller_with("hello");
    let mut streamed: Vec<Envelope> = Vec::new();
    b.feed_line_streaming(r#"{"req_id":"r1","event":"user_text","text":"go"}"#, &mut streamed, &mut ());

    assert_eq!(buffered, streamed, "streaming and buffered output must match");
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
    use gpt2::{Gpt, GptConfig};
    let cfg = GptConfig::tiny();
    let itos: Vec<char> = (0..cfg.vocab).map(|i| char::from_u32(b'a' as u32 + i).unwrap_or('?')).collect();
    let init = gpt2::init_weights(&cfg, 1234);
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
fn successive_camera_frames_each_emit_a_detection() {
    // Regression: after a detection the controller must return to Idle so a
    // SECOND frame is handled too (multiple in-flight requests over one stream).
    let mut ctrl = controller_with("x");
    let px = vec![1u8, 2, 3, 4, 5, 6];
    let frame = |id: &str| {
        events::encode_envelope(&Envelope::with_id(
            Some(id.into()),
            Event::CameraFrame {
                format: "rgb8".into(),
                w: 2,
                h: 1,
                data: Some(base64::encode(&px)),
                path: None,
            },
        ))
    };
    for id in ["f1", "f2", "f3"] {
        let out = ctrl.feed_line(&frame(id));
        let det: Vec<&Envelope> =
            out.iter().filter(|e| matches!(e.event, Event::ObjectDetected { .. })).collect();
        assert_eq!(det.len(), 1, "frame {id} should emit one detection: {out:?}");
        assert_eq!(det[0].req_id.as_deref(), Some(id), "frame {id} req_id echo");
    }
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

/// Build a controller wired with a fake TTS model (no real model, no GPU).
fn synth_controller() -> Controller {
    let reg = Registry {
        synth: Some(Box::new(FakeSynthModel::default())),
        ..Default::default()
    };
    Controller::new(reg)
}

#[test]
fn user_synth_request_streams_audio_chunks_then_done() {
    let mut ctrl = synth_controller();
    // ~60 chars -> 60_000 samples -> 3 chunks at 24_000/chunk + a terminal done.
    let text = "the quick brown fox jumps over the lazy dog twice and then again!";
    let line = events::encode_line(&Event::UserSynthRequest {
        text: text.into(),
        ref_audio: None,
        ref_text: None,
        language: Some("english".into()),
    });
    let out = events_of(&ctrl.feed_line(&line));

    let chunks: Vec<(&str, u32, u32, bool)> = out
        .iter()
        .filter_map(|e| match e {
            Event::AudioChunk { pcm_b64, sample_rate, seq, done } => {
                Some((pcm_b64.as_str(), *sample_rate, *seq, *done))
            }
            _ => None,
        })
        .collect();

    // At least one chunk plus exactly one terminal done:true.
    assert!(chunks.len() >= 2, "expected >=1 data chunk + terminal done: {out:?}");
    assert_eq!(chunks.iter().filter(|c| c.3).count(), 1, "exactly one terminal done");
    let (_, _, _, last_done) = *chunks.last().unwrap();
    assert!(last_done, "the final chunk must be the terminal done:true");
    // Every chunk carries the 24 kHz rate; seqs increment from 0.
    assert!(chunks.iter().all(|c| c.1 == 24000));
    assert_eq!(chunks.first().unwrap().2, 0, "first seq is 0");
    // Non-terminal chunks decode back to f32 PCM cleanly.
    for (b64, _, _, done) in &chunks {
        if !done {
            assert!(runtime::pump::decode_pcm(b64).is_ok(), "chunk must decode");
        }
    }
}

#[test]
fn user_synth_request_with_req_id_echoes_it() {
    let mut ctrl = synth_controller();
    let ev = Event::UserSynthRequest {
        text: "hello".into(),
        ref_audio: Some("voice.wav".into()),
        ref_text: Some("ref".into()),
        language: None,
    };
    let line = events::encode_envelope(&Envelope::with_id(Some("s1".into()), ev));
    let out = ctrl.feed_line(&line);
    let chunks: Vec<&Envelope> =
        out.iter().filter(|e| matches!(e.event, Event::AudioChunk { .. })).collect();
    assert!(!chunks.is_empty(), "expected audio chunks: {out:?}");
    for env in &out {
        assert_eq!(env.req_id.as_deref(), Some("s1"), "every event echoes req_id: {env:?}");
    }
    assert!(out.iter().any(|e| matches!(e.event, Event::AudioChunk { done: true, .. })));
}

#[test]
fn synth_without_model_emits_terminal_done_only() {
    // No synth model in the registry: the request must still complete cleanly
    // with a single terminal done chunk (never panic / hang).
    let mut ctrl = Controller::new(Registry::new());
    let line = events::encode_line(&Event::UserSynthRequest {
        text: "anything".into(),
        ref_audio: None,
        ref_text: None,
        language: None,
    });
    let out = events_of(&ctrl.feed_line(&line));
    let chunks: Vec<&Event> =
        out.iter().filter(|e| matches!(e, Event::AudioChunk { .. })).collect();
    assert_eq!(chunks.len(), 1, "exactly the terminal chunk: {out:?}");
    assert!(matches!(chunks[0], Event::AudioChunk { done: true, .. }));
}

#[test]
fn synth_returns_to_idle_and_text_still_works() {
    // After a synth turn the controller must return to Idle so a following text
    // turn streams normally (the seam doesn't break the existing path).
    let infer = Box::new(FakeInferModel::echoing("hi"));
    let reg = Registry {
        infer: Some(infer),
        synth: Some(Box::new(FakeSynthModel::default())),
        ..Default::default()
    };
    let cfg = GenConfig { max_new: 64, temperature: 0.0, top_k: 0, eos: Some(256), seed: 0 };
    let mut ctrl = Controller::with_config(reg, cfg);

    let s = events_of(&ctrl.feed_line(&events::encode_line(&Event::UserSynthRequest {
        text: "speak".into(),
        ref_audio: None,
        ref_text: None,
        language: None,
    })));
    assert!(s.iter().any(|e| matches!(e, Event::AudioChunk { done: true, .. })));

    let t = events_of(&ctrl.feed_line(r#"{"event":"user_text","text":"go"}"#));
    assert!(
        t.iter().any(|e| matches!(e, Event::BrainTextChunk { done: true, .. })),
        "text path must still stream after a synth turn: {t:?}"
    );
}

#[test]
fn request_without_req_id_carries_none() {
    let mut ctrl = controller_with("hi");
    let out = ctrl.feed_line(r#"{"event":"user_text","text":"q"}"#);
    assert!(!out.is_empty());
    assert!(out.iter().all(|e| e.req_id.is_none()), "untagged request must carry None: {out:?}");
}
