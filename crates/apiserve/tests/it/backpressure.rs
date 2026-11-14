// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers whose memory
// stays bounded however slowly their clients read for its clients. If your
// team needs expertise in streaming backpressure and cancellation then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A streamed generation against a client that reads slowly or not at all.
//!
//! The model here produces deltas as fast as it can; the response body is
//! read in process (`tower::ServiceExt::oneshot`), so an unread body is an
//! unpolled stream and nothing between the model and the reader buffers on
//! its behalf. What is pinned: the model is held back by the reader (at most
//! the documented queue bound is ever ahead of it), a slow reader still gets
//! every byte in order, a reader that stops is cancelled after the stall
//! deadline with a terminal error event, and a reader that disconnects frees
//! a held-back model at once. Each call is settled with the hooks exactly once.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use apiserve::bridge::{STREAM_COALESCE_BYTES, STREAM_QUEUE_MESSAGES};
use apiserve::{router, ApiError, AppState, Call, CallResult, Provider, RequestHooks, Ticket};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use futures::StreamExt;
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

/// Bytes in each delta [`Flood`] produces (`"00042 "`).
const PIECE_BYTES: usize = 6;

/// The delta a [`Flood`] produces at step `i`.
fn piece(i: usize) -> String {
    format!("{:05} ", i % 100_000)
}

/// What the model has done, as the test observes it.
#[derive(Default)]
struct Meter {
    /// Bytes of delta text handed to the progress callback so far.
    emitted: AtomicUsize,
    /// Set when `run` has returned, by finishing or by observing its cancel.
    stopped: AtomicBool,
}

/// A chat model that streams `pieces` deltas as fast as its progress callback
/// returns, checking its cancel token before each one.
struct Flood {
    pieces: usize,
    meter: Arc<Meter>,
}
struct FloodInst {
    pieces: usize,
    meter: Arc<Meter>,
}

impl ResidentModel for Flood {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-chat",
            "a chat model that never waits for its tokens",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-chat", "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FloodInst { pieces: self.pieces, meter: Arc::clone(&self.meter) }))
    }
}

impl Instance for FloodInst {
    fn run(&mut self, _a: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut text = String::new();
        let mut result = Err("cancelled".to_string());
        for i in 0..self.pieces {
            if inv.cancel.is_cancelled() {
                break;
            }
            let p = piece(i);
            progress(Progress::token(i as u32, 0, &p));
            self.meter.emitted.fetch_add(p.len(), Ordering::SeqCst);
            text.push_str(&p);
        }
        if !inv.cancel.is_cancelled() {
            result = Ok(Outcome::new()
                .set("prompt_tokens", json!(1))
                .set("completion_tokens", json!(self.pieces))
                .set("finish_reason", json!("stop"))
                .blob("text", Blob::new(Media::Text, text.into_bytes())));
        }
        self.meter.stopped.store(true, Ordering::SeqCst);
        result
    }
}

/// How each call was settled with the hooks.
#[derive(Debug, PartialEq, Eq)]
enum Settled {
    Done,
    Failed,
    Refused,
}

struct Recorder(Arc<Mutex<Vec<Settled>>>);
struct RecorderTicket(Arc<Mutex<Vec<Settled>>>);

impl RequestHooks for Recorder {
    fn begin(&self, _call: &Call<'_>) -> Result<Box<dyn Ticket>, ApiError> {
        Ok(Box::new(RecorderTicket(Arc::clone(&self.0))))
    }
}

impl Ticket for RecorderTicket {
    fn finish(self: Box<Self>, result: CallResult<'_>) -> Result<(), ApiError> {
        let settled = match result {
            CallResult::Done(_) => Settled::Done,
            CallResult::Failed(_) => Settled::Failed,
            CallResult::Refused => Settled::Refused,
        };
        self.0.lock().unwrap().push(settled);
        Ok(())
    }
}

struct Rig {
    app: Router,
    state: AppState,
    meter: Arc<Meter>,
    settled: Arc<Mutex<Vec<Settled>>>,
}

fn rig(pieces: usize, stall: Duration) -> Rig {
    let meter = Arc::new(Meter::default());
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Flood { pieces, meter: Arc::clone(&meter) })];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    let settled = Arc::new(Mutex::new(Vec::new()));
    let state = AppState::new(exec, "sk-brain-test-key", Provider::OpenAI).with_hooks(Arc::new(Recorder(Arc::clone(&settled)))).with_stream_stall(stall);
    Rig { app: router(state.clone()), state, meter, settled }
}

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::AUTHORIZATION, "Bearer sk-brain-test-key")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({"model": "brain-chat", "stream": true, "messages": [{"role": "user", "content": "hi"}]}).to_string()))
        .unwrap()
}

/// The `data:` payloads of an SSE body, in order.
fn sse_data(body: &str) -> Vec<Value> {
    body.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect()
}

/// The assistant text an OpenAI chat stream carried, concatenated.
fn streamed_text(body: &str) -> String {
    sse_data(body).iter().filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_string)).collect()
}

/// Waits until `check` holds, failing the test after `within`.
async fn eventually(what: &str, within: Duration, check: impl Fn() -> bool) {
    let deadline = Instant::now() + within;
    while !check() {
        assert!(Instant::now() < deadline, "never happened: {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The most delta text the bridge may hold for one stream: a full queue of
/// coalesced deltas, each at most one piece past the coalescing limit, plus the
/// one item the SSE layer has taken and not yet yielded.
fn queue_bound_bytes() -> usize {
    (STREAM_QUEUE_MESSAGES + 1) * (STREAM_COALESCE_BYTES + PIECE_BYTES)
}

#[tokio::test]
async fn a_slow_reader_holds_the_generation_back_and_still_receives_every_byte() {
    // Several times the queue bound, so a model that is not held back would
    // finish during the pause below.
    let pieces = 8 * queue_bound_bytes() / PIECE_BYTES;
    let rig = rig(pieces, Duration::from_secs(30));
    let response = rig.app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut raw = String::new();
    let mut frames = 0usize;
    while let Some(chunk) = body.next().await {
        raw.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
        frames += 1;
        if frames == 3 {
            // The reader pauses: the model fills the queue and then waits.
            tokio::time::sleep(Duration::from_millis(300)).await;
            let held = rig.meter.emitted.load(Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert_eq!(rig.meter.emitted.load(Ordering::SeqCst), held, "a full queue holds the model back");
            assert!(!rig.meter.stopped.load(Ordering::SeqCst), "the model is waiting, not finished");
            let ahead = held - streamed_text(&raw).len();
            assert!(ahead <= queue_bound_bytes(), "{ahead} bytes ahead of the reader exceeds the bound {}", queue_bound_bytes());
        } else if frames.is_multiple_of(97) {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }
    let expected: String = (0..pieces).map(piece).collect();
    assert_eq!(streamed_text(&raw), expected, "the concatenated deltas are the generation, in order");
    assert!(raw.trim_end().ends_with("data: [DONE]"), "the stream ends normally");
    eventually("the settlement", Duration::from_secs(10), || !rig.settled.lock().unwrap().is_empty()).await;
    assert_eq!(*rig.settled.lock().unwrap(), [Settled::Done]);
}

#[tokio::test]
async fn a_reader_that_stops_reading_is_cancelled_after_the_stall_deadline() {
    let rig = rig(usize::MAX, Duration::from_millis(200));
    let response = rig.app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut raw = String::new();
    raw.push_str(std::str::from_utf8(&body.next().await.unwrap().unwrap()).unwrap());

    // The body is kept but not read: the stall deadline cancels the generation.
    eventually("the stalled generation stops", Duration::from_secs(10), || rig.meter.stopped.load(Ordering::SeqCst)).await;
    eventually("the settlement", Duration::from_secs(10), || !rig.settled.lock().unwrap().is_empty()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(*rig.settled.lock().unwrap(), [Settled::Failed], "settled once, as a failure");
    assert!(rig.meter.emitted.load(Ordering::SeqCst) <= queue_bound_bytes() + 2 * PIECE_BYTES, "nothing past the bound was produced for a reader that never came back");

    // The reader comes back: what was queued, then a terminal error event.
    while let Some(chunk) = body.next().await {
        raw.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
    }
    let errors: Vec<Value> = sse_data(&raw).into_iter().filter(|v| v.get("error").is_some()).collect();
    assert_eq!(errors.len(), 1, "one terminal error event: {raw}");
    assert_eq!(errors[0]["error"]["code"], "timeout");
    assert!(raw.trim_end().ends_with("data: [DONE]"));
    drop(body);
    assert_eq!(rig.state.active(), 0, "the request's registration is released");
}

#[tokio::test]
async fn a_reader_that_disconnects_frees_a_held_back_generation_at_once() {
    let rig = rig(usize::MAX, Duration::from_secs(30));
    let response = rig.app.clone().oneshot(chat_request()).await.unwrap();
    let mut body = response.into_body().into_data_stream();
    let _ = body.next().await;
    eventually("the queue fills", Duration::from_secs(10), || rig.meter.emitted.load(Ordering::SeqCst) > STREAM_QUEUE_MESSAGES * PIECE_BYTES).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!rig.meter.stopped.load(Ordering::SeqCst));

    let dropped = Instant::now();
    drop(body);
    eventually("the generation stops", Duration::from_secs(5), || rig.meter.stopped.load(Ordering::SeqCst)).await;
    assert!(dropped.elapsed() < Duration::from_secs(5), "well inside the stall deadline");
    eventually("the settlement", Duration::from_secs(10), || !rig.settled.lock().unwrap().is_empty()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(*rig.settled.lock().unwrap(), [Settled::Failed]);
    assert_eq!(rig.state.active(), 0);
}
