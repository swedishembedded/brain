// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Brain runtime: an event-driven, HSM-driven controller that wires loaded
//! models to the JSONL [`events`] protocol.
//!
//! Models plug in through two object-safe traits — [`InferModel`] (text/token
//! inference) and [`DetectModel`] (object detection) — so the controller is
//! written once against the traits and a [`Registry`] holds whichever concrete
//! models were loaded. A real GPT plugs in via [`GptInfer`]; a real YOLO adapter
//! drops in later behind [`DetectModel`] (tests use [`FakeDetectModel`]).
//!
//! ## State hierarchy
//! ```text
//! Root
//! ├── Operational
//! │   ├── Idle        (waiting for input)
//! │   ├── Chatting    (streaming a text response, one token per Tick)
//! │   └── Detecting   (one-shot object detection)
//! └── Faulted         (error sink)
//! ```
//! Built on [`hfsm`], applying the embedded state-machine skill's patterns:
//!   * **Reminder** — `Chatting::on_entry` seeds the [`StreamPump`] and posts a
//!     `Tick`; each `Tick` pumps exactly one token, emits one `brain_text_chunk`,
//!     and re-posts `Tick`. At EOS it emits the terminal `done:true` chunk and
//!     transitions back to `Idle`. RTC guarantees the self-posted `Tick`s are
//!     processed in order, one per dispatch.
//!   * **Behavioural inheritance** — `cancel` is handled once in `Operational`
//!     (→ `Faulted`) and inherited by every operational substate.
//!   * **LCA-correct entry/exit** — `Chatting::on_exit` frees the pump exactly
//!     once, guaranteed by the engine's exit-chain on any outbound transition.

use events::{Envelope, Event};
use hfsm::{Disp, Hsm, Machine};

pub mod pump;
pub mod sample;

pub use pump::StreamPump;

/// A text/token inference model (e.g. a GPT decoder) behind an object-safe seam.
pub trait InferModel {
    /// Per-position logits for one token sequence, row-major `[len * vocab]`.
    fn logits_all(&self, tokens: &[u32]) -> Vec<f32>;
    /// Maximum context length.
    fn block_size(&self) -> u32;
    /// Vocabulary size.
    fn vocab(&self) -> u32;
    /// Index-to-char table for char tokenizers, if any.
    fn itos(&self) -> Option<&[char]>;
}

/// An object-detection model (e.g. YOLO) behind an object-safe seam.
pub trait DetectModel {
    /// Detect objects in an RGB8 image; returns `[x1, y1, x2, y2, score, class]`
    /// rows.
    fn detect(&self, rgb: &[u8], w: u32, h: u32) -> Vec<[f32; 6]>;
    /// Class-id → label table (parallel to the `class` field of a detection).
    fn labels(&self) -> &[String] {
        &[]
    }
}

/// A sink for events emitted by the controller during a turn. Tests capture into
/// a `Vec`; the CLI writes JSONL to stdout.
pub trait Emit {
    fn emit(&mut self, ev: Event);
}

impl Emit for Vec<Event> {
    fn emit(&mut self, ev: Event) {
        self.push(ev);
    }
}

// ---- GPT adapter ----------------------------------------------------------

/// Wraps a loaded [`gpt::Gpt`] (sized for `B=1`) plus its char vocab as an
/// [`InferModel`]. Construct via [`GptInfer::load`] (from a checkpoint path) or
/// [`GptInfer::from_parts`].
pub struct GptInfer {
    model: gpt::Gpt,
    itos: Option<Vec<char>>,
}

impl GptInfer {
    /// Load a GPT checkpoint and its embedded `itos` (if char-level). The model is
    /// sized `B=1 × T=block_size` for single-sequence inference.
    pub fn load(path: &str) -> GptInfer {
        let itos = gpt::Gpt::load_itos(path);
        // size T to the model's own block size by peeking the config.
        let block = {
            let c = checkpoint::load(path);
            gpt::GptConfig::from_json(&c.header["config"]).block_size
        };
        let model = gpt::Gpt::load(path, 1, block);
        GptInfer { model, itos }
    }

    pub fn from_parts(model: gpt::Gpt, itos: Option<Vec<char>>) -> GptInfer {
        GptInfer { model, itos }
    }
}

impl InferModel for GptInfer {
    fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        self.model.logits_all(tokens)
    }
    fn block_size(&self) -> u32 {
        self.model.cfg.block_size
    }
    fn vocab(&self) -> u32 {
        self.model.cfg.vocab
    }
    fn itos(&self) -> Option<&[char]> {
        self.itos.as_deref()
    }
}

// ---- YOLO adapter ---------------------------------------------------------

/// Wraps a loaded [`yolo::Yolo`] (sized for `B=1`) as a [`DetectModel`]. The
/// controller hands it an RGB8 frame; the adapter normalises it to `[0,1]` HWC
/// floats and calls [`yolo::Yolo::detect`], which letterboxes to the model's
/// input, runs the eval-mode forward, decodes + NMS, and returns boxes in the
/// frame's own pixel coordinates. Construct via [`YoloDetect::load`].
pub struct YoloDetect {
    model: yolo::Yolo,
    labels: Vec<String>,
    conf: f32,
    iou: f32,
}

impl YoloDetect {
    /// Load a YOLO checkpoint, sized `B=1` for single-frame inference. Labels are
    /// numeric (`"0".."nc-1"`) — the synthetic detector carries no class names.
    pub fn load(path: &str) -> YoloDetect {
        let model = yolo::Yolo::load(path, 1);
        YoloDetect::from_model(model)
    }

    /// Wrap an already-built model (used by tests with a random-weight tiny YOLO).
    pub fn from_model(model: yolo::Yolo) -> YoloDetect {
        let labels = (0..model.cfg.nc).map(|c| c.to_string()).collect();
        // Inference is eval-only: pin eval mode so the per-Conv BatchNorm-eval
        // collapse (`sb`) is computed ONCE and reused, instead of being
        // invalidated by detect_batch's eval->train flip every frame. That flip
        // otherwise re-runs pack_sb (4 host readbacks per Conv block, each a full
        // GPU sync) every frame — ~200 syncs/frame, the dominant GPU cost.
        model.set_eval(true);
        YoloDetect { model, labels, conf: 0.25, iou: 0.45 }
    }

    /// Override the confidence / IoU thresholds (defaults 0.25 / 0.45).
    pub fn with_thresholds(mut self, conf: f32, iou: f32) -> YoloDetect {
        self.conf = conf;
        self.iou = iou;
        self
    }
}

impl DetectModel for YoloDetect {
    fn detect(&self, rgb: &[u8], w: u32, h: u32) -> Vec<[f32; 6]> {
        // RGB8 -> normalised f32 HWC (the layout `Yolo::detect` expects).
        let src: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
        self.model.detect(&src, w, h, self.conf, self.iou)
    }
    fn labels(&self) -> &[String] {
        &self.labels
    }
}

// ---- Fake / echo models for tests and the no-checkpoint CLI path ----------

/// An [`InferModel`] that ignores its input and emits a fixed token sequence,
/// each token's logits one-hot so greedy sampling reproduces the script exactly.
/// The sequence ends in `eos`; the pump stops when it samples `eos`.
pub struct FakeInferModel {
    pub script: Vec<u32>,
    pub eos: u32,
    pub itos: Vec<char>,
}

impl FakeInferModel {
    /// Build a fake whose `script` spells `text` over a simple ASCII char vocab,
    /// terminated by an EOS sentinel one past the vocab's text range.
    pub fn echoing(text: &str) -> FakeInferModel {
        // vocab: index == byte value (0..=255), plus EOS at 256.
        let itos: Vec<char> = (0u32..=256).map(|i| char::from_u32(i).unwrap_or('?')).collect();
        let mut script: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let eos = 256;
        script.push(eos);
        FakeInferModel { script, eos, itos }
    }
}

impl InferModel for FakeInferModel {
    fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        // We only need the LAST row to be a one-hot of the next scripted token.
        // The pump always calls with the full growing context; we use its length
        // (minus the seeded prompt) to index the script. To stay self-contained,
        // we instead key off how many of our own tokens are already present at the
        // tail. Simplest robust scheme: count trailing tokens that appear in the
        // script and pick the next one.
        let v = self.vocab() as usize;
        let len = tokens.len();
        let mut logits = vec![0.0f32; len * v];
        // Determine how many scripted tokens we've already produced by matching
        // the tail of `tokens` against `script`.
        let produced = trailing_match_len(tokens, &self.script);
        let next = self.script.get(produced).copied().unwrap_or(self.eos);
        // one-hot the last row
        let base = (len - 1) * v;
        logits[base + next as usize] = 1.0;
        logits
    }
    fn block_size(&self) -> u32 {
        256
    }
    fn vocab(&self) -> u32 {
        257
    }
    fn itos(&self) -> Option<&[char]> {
        Some(&self.itos)
    }
}

/// Longest suffix of `tokens` that equals a prefix of `script` — i.e. how many
/// scripted tokens have already been appended. Lets [`FakeInferModel`] be
/// stateless yet deterministic regardless of the seed prompt.
fn trailing_match_len(tokens: &[u32], script: &[u32]) -> usize {
    let max = tokens.len().min(script.len());
    for k in (0..=max).rev() {
        if tokens[tokens.len() - k..] == script[..k] {
            return k;
        }
    }
    0
}

/// A [`DetectModel`] returning one fixed, deterministic box. Stands in for YOLO.
pub struct FakeDetectModel {
    pub det: [f32; 6],
    pub label: String,
}

impl Default for FakeDetectModel {
    fn default() -> FakeDetectModel {
        FakeDetectModel { det: [10.0, 20.0, 110.0, 220.0, 0.99, 0.0], label: "object".into() }
    }
}

impl DetectModel for FakeDetectModel {
    fn detect(&self, _rgb: &[u8], _w: u32, _h: u32) -> Vec<[f32; 6]> {
        vec![self.det]
    }
    fn labels(&self) -> &[String] {
        std::slice::from_ref(&self.label)
    }
}

// ---- Registry -------------------------------------------------------------

/// Holds the models loaded for a session.
#[derive(Default)]
pub struct Registry {
    pub infer: Option<Box<dyn InferModel>>,
    pub detect: Option<Box<dyn DetectModel>>,
}

impl Registry {
    pub fn new() -> Registry {
        Registry::default()
    }

    /// In-memory constructor for tests: plug in any boxed models.
    pub fn with_models(
        infer: Box<dyn InferModel>,
        detect: Box<dyn DetectModel>,
    ) -> Registry {
        Registry { infer: Some(infer), detect: Some(detect) }
    }
}

// ---- The concrete controller HSM ------------------------------------------

#[derive(Copy, Clone, PartialEq, Debug)]
enum St {
    Root,
    Operational,
    Idle,
    Chatting,
    Detecting,
    Faulted,
}

/// Internal events the machine reacts to. Wraps the external [`Event`] plus the
/// synthetic reminders the controller bridges into the engine.
enum Ev {
    External(Event),
    /// Reminder self-posted by `Chatting`/`Detecting` to advance one step.
    Tick,
    /// Completion reminder: return to `Idle` from an operational substate.
    GoIdle,
}

/// Streaming config (temperature/top-k/max-new/eos) for the text pump.
#[derive(Clone)]
pub struct GenConfig {
    pub max_new: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub eos: Option<u32>,
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> GenConfig {
        GenConfig { max_new: 256, temperature: 0.0, top_k: 0, eos: None, seed: 0 }
    }
}

/// The [`Machine`] implementation: holds the models, the active pump, the output
/// sink, and the streaming sequence counter.
struct Brain {
    registry: Registry,
    cfg: GenConfig,
    pump: Option<StreamPump>,
    seq: u32,
    out: Vec<Envelope>,
    /// Prompt stashed by the controller before entering `Chatting` (entry actions
    /// see no event).
    pending_prompt: Option<String>,
    /// Frame stashed before entering `Detecting`.
    pending_frame: Option<Event>,
    /// Set by a handler/entry action to request another `Tick` (Reminder).
    pending_tick: bool,
    /// Set when the current operational substate has finished and wants `Idle`.
    want_idle: bool,
    /// Correlation id of the request being handled this turn. Echoed onto every
    /// event emitted while it is `Some`. Dispatch is synchronous run-to-completion,
    /// so this is well-defined for the whole turn (all streaming chunks + the
    /// terminal `done`, or the single `object_detected`).
    active_req_id: Option<String>,
}

impl Brain {
    /// Emit one event, stamped with the active request's `req_id` (if any).
    fn emit(&mut self, ev: Event) {
        let req_id = self.active_req_id.clone();
        self.out.push(Envelope { req_id, event: ev });
    }
}

impl Machine for Brain {
    type State = St;
    type Event = Ev;

    fn parent(&self, s: St) -> Option<St> {
        match s {
            St::Root => None,
            St::Operational | St::Faulted => Some(St::Root),
            St::Idle | St::Chatting | St::Detecting => Some(St::Operational),
        }
    }

    fn dispatch(&mut self, state: St, ev: &Ev) -> Disp<St> {
        match state {
            St::Idle => match ev {
                Ev::External(Event::UserText { .. }) => Disp::Tran(St::Chatting),
                Ev::External(Event::CameraFrame { .. }) => Disp::Tran(St::Detecting),
                _ => Disp::Unhandled, // bubble (cancel, etc.)
            },
            St::Chatting => match ev {
                Ev::Tick => {
                    self.pump_one_token();
                    Disp::Handled
                }
                Ev::GoIdle => Disp::Tran(St::Idle),
                // ignore new input while streaming (kept simple); cancel bubbles
                Ev::External(Event::UserText { .. }) | Ev::External(Event::CameraFrame { .. }) => {
                    Disp::Handled
                }
                _ => Disp::Unhandled,
            },
            St::Detecting => match ev {
                // entry action already did the detection + emit; complete to Idle.
                Ev::GoIdle => Disp::Tran(St::Idle),
                Ev::Tick => Disp::Handled,
                _ => Disp::Unhandled,
            },
            // `cancel`/error path handled once here, inherited by all substates.
            St::Operational => match ev {
                Ev::External(Event::Cancel) => Disp::Tran(St::Faulted),
                Ev::External(Event::Error { .. }) => Disp::Tran(St::Faulted),
                _ => Disp::Unhandled,
            },
            St::Faulted => Disp::Handled, // terminal: swallow everything
            St::Root => Disp::Unhandled,
        }
    }

    fn on_entry(&mut self, s: St) {
        match s {
            St::Chatting => self.start_chat(),
            St::Detecting => self.run_detection(),
            St::Faulted => self.emit(Event::Error { message: "controller faulted".into() }),
            _ => {}
        }
    }

    fn on_exit(&mut self, s: St) {
        if s == St::Chatting {
            // free the pump exactly once on leaving Chatting
            self.pump = None;
        }
    }
}

impl Brain {
    /// `Chatting::on_entry`: seed the pump with the last user prompt and kick off
    /// the streaming loop by posting a `Tick`. The prompt is recovered from the
    /// most recent `UserText` that drove the transition — but `on_entry` sees no
    /// event, so the controller stashes it in `pending_prompt` before posting.
    fn start_chat(&mut self) {
        let prompt = self.pending_prompt.take().unwrap_or_default();
        if let Some(infer) = self.registry.infer.as_deref() {
            self.pump = Some(StreamPump::new(infer, &prompt, self.cfg.clone()));
        }
        self.seq = 0;
        // Reminder: drive the first token next RTC step.
        self.pending_tick = true;
    }

    /// `Chatting` Tick handler: pump exactly one token, emit one chunk, re-post.
    fn pump_one_token(&mut self) {
        let infer = match self.registry.infer.as_deref() {
            Some(i) => i,
            None => {
                self.emit(Event::BrainTextChunk { text: String::new(), seq: self.seq, done: true });
                self.want_idle = true;
                return;
            }
        };
        let step = self.pump.as_mut().map(|p| p.step(infer)).unwrap_or(None);
        match step {
            Some(delta) => {
                self.emit(Event::BrainTextChunk { text: delta, seq: self.seq, done: false });
                self.seq += 1;
                self.pending_tick = true; // re-post: keep streaming
            }
            None => {
                // EOS / max_new: terminal chunk then back to Idle.
                self.emit(Event::BrainTextChunk { text: String::new(), seq: self.seq, done: true });
                self.want_idle = true;
            }
        }
    }

    /// `Detecting::on_entry`: run the detector on the pending frame and emit one
    /// `object_detected`, then request the completion Tick back to Idle.
    fn run_detection(&mut self) {
        let frame = self.pending_frame.take();
        let result = frame.as_ref().map(events::decode_frame);
        match (result, self.registry.detect.as_deref()) {
            (Some(Ok(rgb)), Some(det)) => {
                let (w, h) = match frame {
                    Some(Event::CameraFrame { w, h, .. }) => (w, h),
                    _ => (0, 0),
                };
                let dets = det.detect(&rgb, w, h);
                let labels = det.labels().to_vec();
                self.emit(Event::ObjectDetected { dets, labels });
                // Completion → return to Idle so the next frame is handled. (A
                // `Tick` is a no-op in `Detecting`; `GoIdle` is what transitions,
                // matching the chat-EOS / error completion paths.)
                self.want_idle = true;
            }
            (Some(Err(e)), _) => {
                self.emit(Event::Error { message: format!("frame decode: {e}") });
                self.want_idle = true;
            }
            (None, _) => {
                self.emit(Event::Error { message: "no pending frame".into() });
                self.want_idle = true;
            }
            (Some(Ok(_)), None) => {
                self.emit(Event::Error { message: "no detector loaded".into() });
                self.want_idle = true;
            }
        }
    }
}

/// Drives the runtime: consumes protocol lines, emits protocol events. Owns the
/// [`Hsm`] over the [`Brain`] machine and bridges its self-posting flags into the
/// engine's RTC queue.
pub struct Controller {
    hsm: Hsm<Brain>,
}

impl Controller {
    /// Build a controller from a registry, with default generation config.
    pub fn new(registry: Registry) -> Controller {
        Controller::with_config(registry, GenConfig::default())
    }

    pub fn with_config(registry: Registry, cfg: GenConfig) -> Controller {
        let brain = Brain {
            registry,
            cfg,
            pump: None,
            seq: 0,
            out: Vec::new(),
            pending_prompt: None,
            pending_frame: None,
            pending_tick: false,
            want_idle: false,
            active_req_id: None,
        };
        let mut hsm = Hsm::new(brain, St::Idle);
        // Enter the initial Idle chain (Root→Operational→Idle).
        hsm.init();
        Controller { hsm }
    }

    /// Feed one JSONL protocol line; return every event emitted during that turn,
    /// each wrapped in an [`Envelope`] carrying the request's `req_id` (if the line
    /// supplied one). Lines without a `req_id` yield envelopes with `req_id: None`,
    /// i.e. identical behavior to before this field existed.
    ///
    /// Decodes the envelope, sets the active `req_id`, posts the event (and any
    /// reminder follow-ups) to the HSM, runs to completion, and drains the sink.
    /// A decode error surfaces an `error` event (no `req_id`) without faulting.
    pub fn feed_line(&mut self, line: &str) -> Vec<Envelope> {
        match events::decode_envelope(line) {
            Ok(env) => self.feed_event_with_id(env.req_id, env.event),
            Err(e) => {
                // Surface a protocol error without faulting the machine. The line
                // failed to parse, so we have no req_id to echo.
                vec![Envelope::bare(Event::Error { message: format!("decode: {e}") })]
            }
        }
    }

    /// Post an already-decoded [`Event`] (no correlation id) and run to completion.
    pub fn feed_event(&mut self, ev: Event) -> Vec<Envelope> {
        self.feed_event_with_id(None, ev)
    }

    /// Post an already-decoded [`Event`] tagged with `req_id` and run to
    /// completion. Every emitted event for this turn carries `req_id`.
    pub fn feed_event_with_id(&mut self, req_id: Option<String>, ev: Event) -> Vec<Envelope> {
        self.hsm.machine_mut().active_req_id = req_id;
        // Stash payloads the entry actions need (they see no event).
        match &ev {
            Event::UserText { text } => {
                self.hsm.machine_mut().pending_prompt = Some(text.clone());
            }
            Event::CameraFrame { .. } => {
                self.hsm.machine_mut().pending_frame = Some(ev.clone());
            }
            _ => {}
        }
        self.hsm.post(Ev::External(ev));
        self.pump_until_settled();
        let out = std::mem::take(&mut self.hsm.machine_mut().out);
        // The turn is over; clear the active id so any later untagged emit can't
        // accidentally inherit it.
        self.hsm.machine_mut().active_req_id = None;
        out
    }

    /// Bridge the machine's self-post flags (`pending_tick`, `want_idle`) into the
    /// engine and drain until no more synthetic work remains. This realises the
    /// Reminder pattern over the generic engine: each reminder is processed fully
    /// (one token / one completion step) before the next is posted, and the run is
    /// non-reentrant, so RTC ordering holds.
    fn pump_until_settled(&mut self) {
        // Cap iterations as a safety net against a misbehaving pump.
        for _ in 0..(self.hsm.machine().cfg.max_new + 8).max(8) {
            self.hsm.run();
            let m = self.hsm.machine_mut();
            if m.want_idle {
                m.want_idle = false;
                m.pending_tick = false;
                self.hsm.post(Ev::GoIdle);
                continue;
            }
            if m.pending_tick {
                m.pending_tick = false;
                self.hsm.post(Ev::Tick);
                continue;
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_decode_error_is_surfaced() {
        let mut ctrl = Controller::new(Registry::new());
        let out = ctrl.feed_line("garbage");
        assert!(matches!(
            out.as_slice(),
            [Envelope { req_id: None, event: Event::Error { .. } }]
        ));
    }
}
