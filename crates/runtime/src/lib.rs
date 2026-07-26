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
//! │   ├── Idle         (waiting for input)
//! │   ├── Chatting     (streaming a text response, one token per Tick)
//! │   ├── Detecting    (one-shot object detection)
//! │   ├── Synthesizing (streaming synthesized audio, one chunk per Tick)
//! │   ├── Forecasting  (one-shot forecast: forecast_request -> forecast_result)
//! │   ├── Backtesting  (one-shot rolling-origin backtest -> backtest_result)
//! │   └── Cancelled    (one-shot: emit `cancelled` ack, back to Idle — RECOVERABLE)
//! └── Faulted          (error sink)
//! ```
//! The forecasting seam is the fourth model trait ([`forecast::ForecastModel`],
//! alongside [`InferModel`]/[`DetectModel`]/[`SynthModel`]) and, unlike the
//! others, is a **named multi-model map** in the [`Registry`] — the client
//! selects a model per request and negotiates capabilities. Forecast/backtest
//! are one-shot (entry action computes + emits, like `Detecting`); forecast
//! errors take the recoverable emit-and-return-to-`Idle` path, never `Faulted`.
//! Built on [`hfsm`], applying the embedded state-machine skill's patterns:
//!   * **Reminder** — `Chatting::on_entry` seeds the [`StreamPump`] and posts a
//!     `Tick`; each `Tick` pumps exactly one token, emits one `brain_text_chunk`,
//!     and re-posts `Tick`. At EOS it emits the terminal `done:true` chunk and
//!     transitions back to `Idle`. RTC guarantees the self-posted `Tick`s are
//!     processed in order, one per dispatch.
//!   * **Behavioural inheritance** — `cancel` is handled once in `Operational`
//!     (→ recoverable `Cancelled`) and inherited by every operational substate;
//!     a genuine `error` still routes to the terminal `Faulted` sink.
//!   * **Streaming sink** — the pump flushes each emission to a caller-supplied
//!     `&mut dyn Emit` as it is produced and polls a `&mut dyn Control` between
//!     reminders, so chunks reach the wire live and a `Cancel` pre-empts the next
//!     token instead of waiting for the whole turn to finish.
//!   * **LCA-correct entry/exit** — `Chatting::on_exit` frees the pump exactly
//!     once, guaranteed by the engine's exit-chain on any outbound transition.

use events::{Envelope, Event};
use forecast::{BacktestReport, BacktestSpec, ForecastModel, ForecastSpec, Panel};
use hfsm::{Disp, Hsm, Machine};
use std::collections::HashMap;
use std::sync::Arc;

pub mod pump;
pub mod sample;

pub use pump::{AudioStreamPump, StreamPump};

/// How many PCM samples each streamed `audio_chunk` carries (24 kHz · 1 s). The
/// whole waveform is produced up front, then sliced into chunks of this size.
pub const AUDIO_CHUNK_SAMPLES: usize = 24000;

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

/// A resolved text-to-speech request handed to a [`SynthModel`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SynthRequest {
    /// The text to synthesize.
    pub text: String,
    /// Optional reference-audio path for voice cloning (x-vector timbre).
    pub ref_audio: Option<String>,
    /// Optional transcript of the reference audio (ICL path).
    pub ref_text: Option<String>,
    /// Optional language tag (e.g. `"english"`); the model picks a default.
    pub language: Option<String>,
}

/// A text-to-speech model behind an object-safe seam (the TTS analogue of
/// [`InferModel`]). The controller calls [`synth`](SynthModel::synth) once per
/// request to produce the whole waveform, which is then streamed out in chunks.
///
/// A real Qwen3-TTS model plugs in here by wrapping [`tts::pipeline`]: an adapter
/// holds the loaded [`tts::TtsPaths`] + [`tts::GenOpts`] and, in `synth`, calls
/// `tts::pipeline::synth` (no reference) or `tts::pipeline::clone` (with
/// `ref_audio`/`ref_text`), returning the 24 kHz waveform. It is intentionally
/// NOT wired into `brain run` here to keep the runtime's build/deps light (the
/// TTS stack pulls the whole codec+speaker+talker graph); the seam + a
/// [`FakeSynthModel`] test is sufficient.
pub trait SynthModel {
    /// Synthesize a waveform for `req`. The companion [`sample_rate`] gives its
    /// rate (Hz).
    fn synth(&self, req: &SynthRequest) -> Vec<f32>;
    /// Output sample rate (Hz). Defaults to 24 kHz (the Qwen3-TTS codec rate).
    fn sample_rate(&self) -> u32 {
        24000
    }
}

/// A live sink for envelopes emitted by the controller during a streaming turn.
/// Each emission is delivered as it is produced (one token / one chunk), not
/// buffered until the turn ends — so the CLI/socket can flush to the wire
/// incrementally. Tests capture into a `Vec<Envelope>`; the server adapts it to
/// a JSONL line writer.
pub trait Emit {
    fn emit(&mut self, env: Envelope);
}

impl Emit for Vec<Envelope> {
    fn emit(&mut self, env: Envelope) {
        self.push(env);
    }
}

/// An out-of-band control source polled *between* streaming steps, so a `Cancel`
/// (or a structured `Error`) can interrupt a long generation without blocking the
/// pump. Returns `None` when nothing is pending. The unit type is the no-control
/// source — it never interrupts — used by the buffered [`Controller::feed_event`]
/// path and by transports that don't (yet) supply a side channel.
pub trait Control {
    fn poll(&mut self) -> Option<Event>;
}

impl Control for () {
    fn poll(&mut self) -> Option<Event> {
        None
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

/// A [`SynthModel`] returning a fixed, deterministic waveform whose length scales
/// with the request text, so tests can assert chunking without a real TTS model.
pub struct FakeSynthModel {
    /// Samples of synthesized audio per input character (deterministic ramp).
    pub samples_per_char: usize,
    pub sample_rate: u32,
}

impl Default for FakeSynthModel {
    fn default() -> FakeSynthModel {
        // ~2.5 chunks of audio for a 60-char prompt at the default chunk size,
        // enough to exercise multi-chunk streaming + the terminal `done`.
        FakeSynthModel { samples_per_char: 1000, sample_rate: 24000 }
    }
}

impl SynthModel for FakeSynthModel {
    fn synth(&self, req: &SynthRequest) -> Vec<f32> {
        let n = (req.text.chars().count().max(1)) * self.samples_per_char;
        // Deterministic low-amplitude ramp; content is irrelevant to the tests.
        (0..n).map(|i| ((i % 256) as f32 / 256.0) - 0.5).collect()
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
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
///
/// The text/image/audio seams are single-slot (one model per kind); the
/// forecasting seam is a **named map** because the forecasting API is
/// multi-model by design — a client selects a model by name per request, and
/// capability negotiation enumerates them. Models are `Arc` so a future
/// multi-threaded server can share one instance across workers.
#[derive(Default)]
pub struct Registry {
    pub infer: Option<Box<dyn InferModel>>,
    pub detect: Option<Box<dyn DetectModel>>,
    pub synth: Option<Box<dyn SynthModel>>,
    pub forecast: HashMap<String, Arc<dyn ForecastModel>>,
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
        Registry {
            infer: Some(infer),
            detect: Some(detect),
            synth: None,
            forecast: HashMap::new(),
        }
    }

    /// Register a forecasting model under its own [`Capabilities::name`]. A later
    /// registration under the same name replaces the earlier one.
    pub fn register_forecast(&mut self, model: Arc<dyn ForecastModel>) -> &mut Self {
        let name = model.capabilities().name;
        self.forecast.insert(name, model);
        self
    }

    /// Look up a forecasting model by name.
    pub fn forecast_model(&self, name: &str) -> Option<Arc<dyn ForecastModel>> {
        self.forecast.get(name).cloned()
    }

    /// All registered forecasting models' capabilities, sorted by name — the
    /// payload of a `capabilities_result`.
    pub fn capabilities(&self) -> Vec<forecast::Capabilities> {
        let mut v: Vec<_> = self.forecast.values().map(|m| m.capabilities()).collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
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
    Synthesizing,
    /// One-shot forecast: the entry action computes and emits a `forecast_result`
    /// (or a structured `error`), then returns to `Idle`. Follows `Detecting`.
    Forecasting,
    /// One-shot rolling-origin backtest: the entry action runs the whole backtest
    /// and emits a `backtest_result`, then returns to `Idle`.
    Backtesting,
    /// Recoverable cancellation: the entry action emits a terminal `cancelled`
    /// acknowledgement and returns to `Idle`. Distinct from [`St::Faulted`] —
    /// a cancel does NOT brick the session (the counterpart to `Faulted` for the
    /// benign, host-requested stop). Follows `Detecting`'s one-shot shape.
    Cancelled,
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

/// Steps a rolling-origin backtest **one model at a time** so the controller can
/// stream `backtest_chunk` progress. Each model's rolling-origin evaluation is
/// independent (per-`(model, metric)` rows over the same deterministic origins),
/// so running them one per `step()` and concatenating the rows reproduces exactly
/// the all-models report — while letting the caller see partial aggregates as each
/// model finishes.
struct BacktestPump {
    models: Vec<(String, Arc<dyn ForecastModel>)>,
    panel: Panel,
    spec: BacktestSpec,
    next: usize,
    report: BacktestReport,
}

impl BacktestPump {
    /// Evaluate the next model, merge its rows into the running report, and return
    /// `(report_so_far, is_last)`. `None` once every model has been evaluated.
    fn step(&mut self) -> Option<(BacktestReport, bool)> {
        let (name, model) = self.models.get(self.next)?;
        let one = [(name.clone(), model.as_ref())];
        let r = fcbench::backtest::run(&one, &self.panel, &self.spec);
        self.report.rows.extend(r.rows);
        self.next += 1;
        Some((self.report.clone(), self.next >= self.models.len()))
    }
}

/// The [`Machine`] implementation: holds the models, the active pump, the output
/// sink, and the streaming sequence counter.
struct Brain {
    registry: Registry,
    /// Generic capability providers (Z-Image, …), driving `ActionRequest` /
    /// `ManifestRequest` through the model-agnostic [`capability`] interface.
    caps: capability::Registry,
    cfg: GenConfig,
    pump: Option<StreamPump>,
    /// Active audio pump while in `Synthesizing` (parallel to `pump`).
    audio_pump: Option<AudioStreamPump>,
    /// Active backtest pump while in `Backtesting`: steps one model per `Tick`,
    /// emitting an aggregates-so-far `backtest_chunk` after each, then the final
    /// `backtest_result`.
    backtest_pump: Option<BacktestPump>,
    seq: u32,
    out: Vec<Envelope>,
    /// Prompt stashed by the controller before entering `Chatting` (entry actions
    /// see no event).
    pending_prompt: Option<String>,
    /// Frame stashed before entering `Detecting`.
    pending_frame: Option<Event>,
    /// Synth request stashed before entering `Synthesizing`.
    pending_synth: Option<SynthRequest>,
    /// Forecast request `(model, panel, spec)` stashed before `Forecasting`.
    pending_forecast: Option<(String, Panel, ForecastSpec)>,
    /// Backtest request `(panel, spec)` stashed before `Backtesting`.
    pending_backtest: Option<(Panel, BacktestSpec)>,
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
    /// Run a generic [`Event::ActionRequest`] through the [`capability::Registry`]:
    /// build a validated invocation from the wire params + base64 blobs, execute
    /// the action (streaming `ActionProgress` inline), and emit `ActionResult`
    /// (or a structured `Error`). Model-agnostic — every model's actions flow here.
    fn run_action(&mut self, model: &str, action: &str, params: &serde_json::Value, blobs: &[events::WireBlob]) {
        let Some(act) = self.caps.find(model, action) else {
            self.emit(Event::Error { message: format!("no action '{action}' on model '{model}'") });
            return;
        };
        // wire → capability::Invocation (decode base64 blobs).
        let mut inv = capability::Invocation { params: params.clone(), blobs: Default::default() };
        for wb in blobs {
            let bytes = match events::base64::decode(&wb.b64) {
                Ok(b) => b,
                Err(e) => {
                    self.emit(Event::Error { message: format!("blob '{}': {e}", wb.name) });
                    return;
                }
            };
            let media = capability::Media::parse(&wb.media).unwrap_or(capability::Media::Bytes);
            inv.blobs.insert(wb.name.clone(), capability::Blob { media, bytes, meta: wb.meta.clone() });
        }
        let inv = match act.spec().validate(inv) {
            Ok(i) => i,
            Err(e) => {
                self.emit(Event::Error { message: e });
                return;
            }
        };
        // run — progress streams inline (act is an owned Arc, so the closure may
        // borrow self mutably).
        let mut progress = Vec::new();
        let res = act.run(&inv, &mut |p: capability::Progress| progress.push(p));
        for p in progress {
            self.emit(Event::ActionProgress { step: p.step, total: p.total, message: p.message });
        }
        match res {
            Ok(outcome) => {
                let wire_blobs = outcome
                    .blobs
                    .into_iter()
                    .map(|(name, b)| events::WireBlob { name, media: b.media.name().to_string(), b64: events::base64::encode(&b.bytes), meta: b.meta })
                    .collect();
                self.emit(Event::ActionResult { outputs: outcome.outputs, blobs: wire_blobs });
            }
            Err(e) => self.emit(Event::Error { message: e }),
        }
    }

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
            St::Idle
            | St::Chatting
            | St::Detecting
            | St::Synthesizing
            | St::Forecasting
            | St::Backtesting
            | St::Cancelled => Some(St::Operational),
        }
    }

    fn dispatch(&mut self, state: St, ev: &Ev) -> Disp<St> {
        match state {
            St::Idle => match ev {
                Ev::External(Event::UserText { .. }) => Disp::Tran(St::Chatting),
                Ev::External(Event::CameraFrame { .. }) => Disp::Tran(St::Detecting),
                Ev::External(Event::UserSynthRequest { .. }) => Disp::Tran(St::Synthesizing),
                Ev::External(Event::ForecastRequest { .. }) => Disp::Tran(St::Forecasting),
                Ev::External(Event::BacktestRequest { .. }) => Disp::Tran(St::Backtesting),
                // Capability negotiation is instantaneous and stateless: answer in
                // place without a state change.
                Ev::External(Event::CapabilitiesRequest) => {
                    let models = self.registry.capabilities();
                    self.emit(Event::CapabilitiesResult { models });
                    Disp::Handled
                }
                // Generic capability discovery + invocation — the model-agnostic
                // path. Both are synchronous run-to-completion (progress streams
                // inline), so they answer in Idle with no state change.
                Ev::External(Event::ManifestRequest) => {
                    let manifests = serde_json::Value::Array(self.caps.manifests().iter().map(|m| m.to_json()).collect());
                    self.emit(Event::ManifestResult { manifests });
                    Disp::Handled
                }
                Ev::External(Event::ActionRequest { model, action, params, blobs }) => {
                    self.run_action(model, action, params, blobs);
                    Disp::Handled
                }
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
            // one-shot: entry action computed + emitted; complete to Idle.
            St::Forecasting | St::Cancelled => match ev {
                Ev::GoIdle => Disp::Tran(St::Idle),
                Ev::Tick => Disp::Handled,
                _ => Disp::Unhandled,
            },
            // streamed: each Tick evaluates one model, emitting a `backtest_chunk`,
            // until the last model emits the terminal `backtest_result`.
            St::Backtesting => match ev {
                Ev::Tick => {
                    self.pump_one_backtest();
                    Disp::Handled
                }
                Ev::GoIdle => Disp::Tran(St::Idle),
                _ => Disp::Unhandled,
            },
            St::Synthesizing => match ev {
                Ev::Tick => {
                    self.pump_one_chunk();
                    Disp::Handled
                }
                Ev::GoIdle => Disp::Tran(St::Idle),
                // ignore new input while synthesizing; cancel bubbles to Operational.
                Ev::External(Event::UserText { .. })
                | Ev::External(Event::CameraFrame { .. })
                | Ev::External(Event::UserSynthRequest { .. }) => Disp::Handled,
                _ => Disp::Unhandled,
            },
            // `cancel`/error path handled once here, inherited by all substates.
            // Cancel is a benign, host-requested stop → the RECOVERABLE `Cancelled`
            // state (emit ack, return to Idle). A genuine `error` still faults.
            St::Operational => match ev {
                Ev::External(Event::Cancel) => Disp::Tran(St::Cancelled),
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
            St::Synthesizing => self.start_synth(),
            St::Forecasting => self.run_forecast(),
            St::Backtesting => self.start_backtest(),
            // Emit a terminal `cancelled` ack, then complete back to Idle. The
            // active pump (if any) was already freed by the exit chain leaving the
            // streaming substate, so this is a clean stop, not a fault.
            St::Cancelled => {
                self.emit(Event::Cancelled);
                self.want_idle = true;
            }
            St::Faulted => self.emit(Event::Error { message: "controller faulted".into() }),
            _ => {}
        }
    }

    fn on_exit(&mut self, s: St) {
        match s {
            // free each pump exactly once on leaving its streaming state.
            St::Chatting => self.pump = None,
            St::Synthesizing => self.audio_pump = None,
            St::Backtesting => self.backtest_pump = None,
            _ => {}
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

    /// `Synthesizing::on_entry`: synthesize the whole waveform for the stashed
    /// request, seed the [`AudioStreamPump`], and kick off streaming with a `Tick`.
    /// Mirrors [`start_chat`](Self::start_chat). With no synth model loaded the
    /// pump stays `None` and the first Tick emits just the terminal `done`.
    fn start_synth(&mut self) {
        let req = self.pending_synth.take().unwrap_or_default();
        if let Some(synth) = self.registry.synth.as_deref() {
            let pcm = synth.synth(&req);
            self.audio_pump =
                Some(AudioStreamPump::new(pcm, synth.sample_rate(), AUDIO_CHUNK_SAMPLES));
        }
        self.seq = 0;
        self.pending_tick = true;
    }

    /// `Synthesizing` Tick handler: emit exactly one `audio_chunk`, re-post. At
    /// the end of the waveform emit the terminal `done:true` chunk and go `Idle`.
    fn pump_one_chunk(&mut self) {
        let sr = self.audio_pump.as_ref().map(|p| p.sample_rate()).unwrap_or(24000);
        let next = self.audio_pump.as_mut().and_then(|p| p.step());
        match next {
            Some(pcm_b64) => {
                self.emit(Event::AudioChunk { pcm_b64, sample_rate: sr, seq: self.seq, done: false });
                self.seq += 1;
                self.pending_tick = true; // keep streaming
            }
            None => {
                // Drained (or no model): terminal empty chunk, then back to Idle.
                self.emit(Event::AudioChunk {
                    pcm_b64: String::new(),
                    sample_rate: sr,
                    seq: self.seq,
                    done: true,
                });
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

    /// `Forecasting::on_entry`: resolve the requested model, validate + forecast
    /// the stashed panel, and emit one `forecast_result`. On any failure emit a
    /// structured `error` (code/retryable) and return to `Idle` — a forecast
    /// error is recoverable and never faults the session (mirrors the
    /// frame-decode error path in `run_detection`).
    fn run_forecast(&mut self) {
        let Some((model_name, panel, spec)) = self.pending_forecast.take() else {
            self.emit(Event::Error { message: "no pending forecast".into() });
            self.want_idle = true;
            return;
        };
        match self.registry.forecast_model(&model_name) {
            None => self.emit(Event::ForecastError {
                error: forecast::ForecastError::unknown_model(&model_name),
            }),
            Some(model) => match model.forecast(&panel, &spec) {
                Ok(forecast) => self.emit(Event::ForecastResult { forecast }),
                Err(error) => self.emit(Event::ForecastError { error }),
            },
        }
        self.want_idle = true;
    }

    /// `Backtesting::on_entry`: resolve every requested model, run a
    /// rolling-origin backtest over the stashed panel, and emit one
    /// `backtest_result`. Unknown model names are skipped with a `log` note
    /// rather than failing the whole request. Buffered (not streamed) in P0.
    /// `Backtesting::on_entry`: resolve the requested models to resident instances
    /// and seed the [`BacktestPump`], then kick off streaming with a `Tick`. Bad
    /// requests (no pending backtest / no known models) emit an error and complete
    /// immediately, exactly as the one-shot version did.
    fn start_backtest(&mut self) {
        let Some((panel, spec)) = self.pending_backtest.take() else {
            self.emit(Event::Error { message: "no pending backtest".into() });
            self.want_idle = true;
            return;
        };
        // resolve model names -> resident instances
        let models: Vec<(String, Arc<dyn ForecastModel>)> = spec
            .models
            .iter()
            .filter_map(|name| self.registry.forecast_model(name).map(|m| (name.clone(), m)))
            .collect();
        for name in &spec.models {
            if self.registry.forecast_model(name).is_none() {
                self.emit(Event::Log { message: format!("backtest: unknown model {name} skipped") });
            }
        }
        if models.is_empty() {
            self.emit(Event::ForecastError {
                error: forecast::ForecastError::bad_request("no known models in backtest request"),
            });
            self.want_idle = true;
            return;
        }
        self.backtest_pump =
            Some(BacktestPump { models, panel, spec, next: 0, report: BacktestReport::default() });
        self.seq = 0;
        self.pending_tick = true;
    }

    /// `Backtesting` Tick handler: evaluate the next model. Each non-final model
    /// emits a `backtest_chunk` (aggregates so far) and re-posts; the final model
    /// emits the terminal `backtest_result` and returns to Idle. A single-model
    /// backtest emits just the result (no chunks) — behavior identical to before.
    fn pump_one_backtest(&mut self) {
        match self.backtest_pump.as_mut().and_then(|p| p.step()) {
            Some((report, false)) => {
                self.emit(Event::BacktestChunk { report, seq: self.seq, done: false });
                self.seq += 1;
                self.pending_tick = true; // keep evaluating
            }
            Some((report, true)) => {
                self.emit(Event::BacktestResult { report });
                self.want_idle = true;
            }
            None => {
                // no models (shouldn't happen — start_backtest guards) — complete.
                self.emit(Event::BacktestResult { report: BacktestReport::default() });
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
            caps: capability::Registry::new(),
            cfg,
            pump: None,
            audio_pump: None,
            backtest_pump: None,
            seq: 0,
            out: Vec::new(),
            pending_prompt: None,
            pending_frame: None,
            pending_synth: None,
            pending_forecast: None,
            pending_backtest: None,
            pending_tick: false,
            want_idle: false,
            active_req_id: None,
        };
        let mut hsm = Hsm::new(brain, St::Idle);
        // Enter the initial Idle chain (Root→Operational→Idle).
        hsm.init();
        Controller { hsm }
    }

    /// Register a generic capability [`Provider`](capability::Provider) (e.g.
    /// Z-Image), making its actions reachable over the event API via
    /// `ManifestRequest` / `ActionRequest` — no new event variants per model.
    pub fn register_provider(&mut self, p: std::sync::Arc<dyn capability::Provider>) {
        self.hsm.machine_mut().caps.register(p);
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
        let mut out: Vec<Envelope> = Vec::new();
        self.feed_line_streaming(line, &mut out, &mut ());
        out
    }

    /// Streaming twin of [`feed_line`](Self::feed_line): emit each envelope to
    /// `out` **as it is produced** (one token / one chunk), and poll `ctl` between
    /// steps so a `Cancel` can interrupt mid-generation. A decode error surfaces an
    /// `error` envelope (no `req_id`) without faulting.
    pub fn feed_line_streaming(&mut self, line: &str, out: &mut dyn Emit, ctl: &mut dyn Control) {
        match events::decode_envelope(line) {
            Ok(env) => self.feed_event_streaming(env.req_id, env.event, out, ctl),
            Err(e) => out.emit(Envelope::bare(Event::Error { message: format!("decode: {e}") })),
        }
    }

    /// Post an already-decoded [`Event`] (no correlation id) and run to completion.
    pub fn feed_event(&mut self, ev: Event) -> Vec<Envelope> {
        self.feed_event_with_id(None, ev)
    }

    /// Post an already-decoded [`Event`] tagged with `req_id`, buffering the whole
    /// turn's output. Thin wrapper over [`feed_event_streaming`](Self::feed_event_streaming)
    /// with a `Vec` sink and no control source — behavior identical to before the
    /// streaming seam existed.
    pub fn feed_event_with_id(&mut self, req_id: Option<String>, ev: Event) -> Vec<Envelope> {
        let mut out: Vec<Envelope> = Vec::new();
        self.feed_event_streaming(req_id, ev, &mut out, &mut ());
        out
    }

    /// The streaming core: stash the payload, post the external event, then drive
    /// the pump one reminder at a time — flushing every emission to `out` as it
    /// happens and polling `ctl` for an out-of-band `Cancel` before each `Tick`.
    /// A cancel routes through the recoverable `Cancelled` state (terminal
    /// `cancelled` ack, back to `Idle`), so the controller keeps serving.
    pub fn feed_event_streaming(
        &mut self,
        req_id: Option<String>,
        ev: Event,
        out: &mut dyn Emit,
        ctl: &mut dyn Control,
    ) {
        self.hsm.machine_mut().active_req_id = req_id;
        // Stash payloads the entry actions need (they see no event).
        match &ev {
            Event::UserText { text } => {
                self.hsm.machine_mut().pending_prompt = Some(text.clone());
            }
            Event::CameraFrame { .. } => {
                self.hsm.machine_mut().pending_frame = Some(ev.clone());
            }
            Event::UserSynthRequest { text, ref_audio, ref_text, language } => {
                self.hsm.machine_mut().pending_synth = Some(SynthRequest {
                    text: text.clone(),
                    ref_audio: ref_audio.clone(),
                    ref_text: ref_text.clone(),
                    language: language.clone(),
                });
            }
            Event::ForecastRequest { model, panel, spec } => {
                self.hsm.machine_mut().pending_forecast =
                    Some((model.clone(), panel.clone(), spec.clone()));
            }
            Event::BacktestRequest { panel, spec } => {
                self.hsm.machine_mut().pending_backtest = Some((panel.clone(), spec.clone()));
            }
            _ => {}
        }
        self.hsm.post(Ev::External(ev));
        self.pump_streaming(out, ctl);
        // The turn is over; clear the active id so any later untagged emit can't
        // accidentally inherit it.
        self.hsm.machine_mut().active_req_id = None;
    }

    /// Move everything the machine has emitted so far out to the live sink. Called
    /// after every engine step so streamed chunks reach the caller incrementally
    /// rather than in one batch at the end of the turn.
    fn flush(&mut self, out: &mut dyn Emit) {
        for env in std::mem::take(&mut self.hsm.machine_mut().out) {
            out.emit(env);
        }
    }

    /// Bridge the machine's self-post flags (`pending_tick`, `want_idle`) into the
    /// engine and drive until no more synthetic work remains, flushing to `out`
    /// after each step and polling `ctl` between reminders. This realises the
    /// Reminder pattern over the generic engine: each reminder is processed fully
    /// (one token / one completion step) before the next is posted, and the run is
    /// non-reentrant, so RTC ordering holds. A `Cancel` from `ctl` is injected as
    /// an external event (handled by `Operational` → recoverable `Cancelled`),
    /// pre-empting the next `Tick`.
    fn pump_streaming(&mut self, out: &mut dyn Emit, ctl: &mut dyn Control) {
        // Cap iterations as a safety net against a misbehaving pump. The floor is
        // generous so legitimate multi-chunk audio streams aren't truncated, while
        // still bounding a pump that never clears `pending_tick` (each text token
        // and each audio chunk advances deterministically toward completion).
        for _ in 0..(self.hsm.machine().cfg.max_new + 8).max(100_000) {
            self.hsm.run();
            self.flush(out);
            let m = self.hsm.machine_mut();
            if m.want_idle {
                m.want_idle = false;
                m.pending_tick = false;
                self.hsm.post(Ev::GoIdle);
                continue;
            }
            if m.pending_tick {
                m.pending_tick = false;
                // Poll for an out-of-band cancel BEFORE producing the next chunk,
                // so a long stream stops promptly. The injected event is handled
                // like any external one (Operational → Cancelled), then the loop
                // flushes the ack and settles back to Idle.
                if let Some(ctl_ev) = ctl.poll() {
                    self.hsm.post(Ev::External(ctl_ev));
                    continue;
                }
                self.hsm.post(Ev::Tick);
                continue;
            }
            break;
        }
        self.flush(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forecast::{Representation, Variate};

    #[test]
    fn empty_decode_error_is_surfaced() {
        let mut ctrl = Controller::new(Registry::new());
        let out = ctrl.feed_line("garbage");
        assert!(matches!(
            out.as_slice(),
            [Envelope { req_id: None, event: Event::Error { .. } }]
        ));
    }

    fn registry_with_naive() -> Registry {
        let mut reg = Registry::new();
        reg.register_forecast(Arc::new(fcbench::RandomWalk));
        reg
    }

    // ---- generic capability actions over the event API ----

    struct DemoAction;
    impl capability::Action for DemoAction {
        fn spec(&self) -> capability::ActionSpec {
            use capability::{ActionSpec, BlobSpec, Media, ParamSpec, ParamType};
            ActionSpec::new("echo", "echo text N times")
                .param(ParamSpec::new("text", ParamType::Str, "text").required())
                .param(ParamSpec::new("times", ParamType::Int, "count").default(serde_json::json!(1)))
                .output(BlobSpec::new("result", Media::Text, "the echoed text"))
        }
        fn run(&self, inv: &capability::Invocation, progress: &mut dyn FnMut(capability::Progress)) -> capability::ActionResult {
            progress(capability::Progress { step: 1, total: 1, message: "echoing".into() });
            let s = inv.get_str("text").unwrap_or_default().repeat(inv.get_i64("times").unwrap_or(1) as usize);
            Ok(capability::Outcome::new().set("chars", serde_json::json!(s.len())).blob("result", capability::Blob::new(capability::Media::Text, s.into_bytes())))
        }
    }
    struct DemoProvider;
    impl capability::Provider for DemoProvider {
        fn manifest(&self) -> capability::Manifest {
            use capability::Action as _;
            capability::Manifest::new("demo", "demo", vec![DemoAction.spec()])
        }
        fn action(&self, name: &str) -> Option<Arc<dyn capability::Action>> {
            (name == "echo").then(|| Arc::new(DemoAction) as Arc<dyn capability::Action>)
        }
    }

    #[test]
    fn generic_manifest_and_action_over_event_api() {
        let mut ctrl = Controller::new(Registry::new());
        ctrl.register_provider(Arc::new(DemoProvider));

        // discovery
        let out = ctrl.feed_line(r#"{"event":"manifest_request"}"#);
        match &out[0].event {
            Event::ManifestResult { manifests } => {
                assert_eq!(manifests[0]["model"], "demo");
                assert_eq!(manifests[0]["actions"][0]["name"], "echo");
            }
            e => panic!("expected manifest_result, got {e:?}"),
        }

        // invocation → progress + result (result blob is base64)
        let out = ctrl.feed_line(r#"{"event":"action_request","model":"demo","action":"echo","params":{"text":"ab","times":3}}"#);
        assert!(matches!(&out[0].event, Event::ActionProgress { total: 1, .. }));
        match &out[1].event {
            Event::ActionResult { outputs, blobs } => {
                assert_eq!(outputs["chars"], 6);
                let b = &blobs[0];
                assert_eq!(b.name, "result");
                assert_eq!(events::base64::decode(&b.b64).unwrap(), b"ababab");
            }
            e => panic!("expected action_result, got {e:?}"),
        }

        // validation error surfaces as a structured error, not a panic
        let out = ctrl.feed_line(r#"{"event":"action_request","model":"demo","action":"echo","params":{}}"#);
        assert!(matches!(&out[0].event, Event::Error { .. }));
        // unknown model
        let out = ctrl.feed_line(r#"{"event":"action_request","model":"nope","action":"echo","params":{}}"#);
        assert!(matches!(&out[0].event, Event::Error { .. }));
    }

    fn forecast_request(model: &str, req: Option<&str>) -> Envelope {
        let panel = Panel::single("1d", "AAPL", vec![Variate::target("close", vec![1.0, 2.0, 3.0])]);
        let spec = ForecastSpec {
            horizon: 3,
            representations: vec![Representation::Quantiles, Representation::Point],
            quantile_levels: vec![0.1, 0.5, 0.9],
            num_samples: 0,
            seed: 0,
        };
        Envelope::with_id(
            req.map(|s| s.to_string()),
            Event::ForecastRequest { model: model.to_string(), panel, spec },
        )
    }

    #[test]
    fn forecast_request_yields_a_result_and_returns_to_idle() {
        let mut ctrl = Controller::with_config(registry_with_naive(), GenConfig::default());
        let env = forecast_request("naive", Some("r1"));
        let out = ctrl.feed_event_with_id(env.req_id, env.event);
        // exactly one forecast_result, carrying the req_id
        assert_eq!(out.len(), 1, "{out:?}");
        match &out[0] {
            Envelope { req_id, event: Event::ForecastResult { forecast } } => {
                assert_eq!(req_id.as_deref(), Some("r1"));
                assert_eq!(forecast.model, "naive");
                assert_eq!(forecast.targets[0].name, "close");
                assert!(forecast.targets[0].quantiles.is_some());
            }
            other => panic!("expected forecast_result, got {other:?}"),
        }
        // a second request still works -> we returned to Idle, not Faulted
        let env2 = forecast_request("naive", Some("r2"));
        let out2 = ctrl.feed_event_with_id(env2.req_id, env2.event);
        assert!(matches!(out2.as_slice(), [Envelope { event: Event::ForecastResult { .. }, .. }]));
    }

    #[test]
    fn unknown_model_yields_structured_error_and_stays_operational() {
        let mut ctrl = Controller::with_config(registry_with_naive(), GenConfig::default());
        let env = forecast_request("does_not_exist", Some("r1"));
        let out = ctrl.feed_event_with_id(env.req_id, env.event);
        match out.as_slice() {
            [Envelope { event: Event::ForecastError { error }, .. }] => {
                assert_eq!(error.code, "unknown_model");
            }
            other => panic!("expected forecast error, got {other:?}"),
        }
        // still operational: a valid request now succeeds
        let env2 = forecast_request("naive", None);
        let out2 = ctrl.feed_event(env2.event);
        assert!(matches!(out2.as_slice(), [Envelope { event: Event::ForecastResult { .. }, .. }]));
    }

    #[test]
    fn capabilities_request_enumerates_registered_models() {
        let mut ctrl = Controller::with_config(registry_with_naive(), GenConfig::default());
        let out = ctrl.feed_event(Event::CapabilitiesRequest);
        match out.as_slice() {
            [Envelope { event: Event::CapabilitiesResult { models }, .. }] => {
                assert_eq!(models.len(), 1);
                assert_eq!(models[0].name, "naive");
            }
            other => panic!("expected capabilities_result, got {other:?}"),
        }
    }

    #[test]
    fn backtest_request_yields_an_aggregated_report() {
        let mut reg = Registry::new();
        reg.register_forecast(Arc::new(fcbench::RandomWalk));
        reg.register_forecast(Arc::new(fcbench::Drift));
        let mut ctrl = Controller::with_config(reg, GenConfig::default());
        // a longer series so the rolling origin has room
        let series: Vec<f32> = (0..120).map(|i| 100.0 + (i as f32) * 0.1).collect();
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", series)]);
        let spec = BacktestSpec {
            models: vec!["naive".into(), "drift".into()],
            horizon: 5,
            origins: 10,
            stride: 2,
            metrics: vec!["mase".into(), "wql".into()],
            quantile_levels: vec![0.1, 0.5, 0.9],
            seed: 0,
        };
        let out = ctrl.feed_event(Event::BacktestRequest { panel, spec });
        // Two models now stream: one `backtest_chunk` (partial, done:false) then a
        // terminal `backtest_result` with BOTH models' rows.
        let chunks: Vec<&Event> =
            out.iter().map(|e| &e.event).filter(|e| matches!(e, Event::BacktestChunk { .. })).collect();
        assert_eq!(chunks.len(), 1, "expected one interim chunk for the first model: {out:?}");
        assert!(matches!(chunks[0], Event::BacktestChunk { done: false, .. }));
        match out.last().map(|e| &e.event) {
            Some(Event::BacktestResult { report }) => {
                assert!(report.get("naive", "mase").is_some());
                assert!(report.get("drift", "mase").is_some());
            }
            other => panic!("expected terminal backtest_result, got {other:?}"),
        }
    }

    #[test]
    fn single_model_backtest_emits_only_a_result() {
        // One model -> no interim chunks, just the terminal result (unchanged from
        // the pre-streaming behavior).
        let mut ctrl = Controller::with_config(registry_with_naive(), GenConfig::default());
        let series: Vec<f32> = (0..80).map(|i| 100.0 + (i as f32) * 0.1).collect();
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", series)]);
        let spec = BacktestSpec {
            models: vec!["naive".into()],
            horizon: 5,
            origins: 8,
            stride: 2,
            metrics: vec!["mase".into()],
            quantile_levels: vec![0.5],
            seed: 0,
        };
        let out = ctrl.feed_event(Event::BacktestRequest { panel, spec });
        assert!(
            !out.iter().any(|e| matches!(e.event, Event::BacktestChunk { .. })),
            "single-model backtest must not emit chunks: {out:?}"
        );
        assert!(matches!(out.last().map(|e| &e.event), Some(Event::BacktestResult { .. })));
    }

    #[test]
    fn backtest_can_be_cancelled_between_models_and_recovers() {
        // Cancel after the first model's chunk: the backtest stops early with a
        // `cancelled` ack, and the controller returns to Idle (a later request
        // works). Uses the streaming pump + a control that cancels after 1 chunk.
        struct CancelAfterOne(bool);
        impl Control for CancelAfterOne {
            fn poll(&mut self) -> Option<Event> {
                if self.0 {
                    Some(Event::Cancel)
                } else {
                    self.0 = true;
                    None
                }
            }
        }
        let mut reg = Registry::new();
        reg.register_forecast(Arc::new(fcbench::RandomWalk));
        reg.register_forecast(Arc::new(fcbench::Drift));
        let mut ctrl = Controller::with_config(reg, GenConfig::default());
        let series: Vec<f32> = (0..120).map(|i| 100.0 + (i as f32) * 0.1).collect();
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", series)]);
        // First poll (before model 0) primes to None; the second (after model 0's
        // chunk, before model 1) cancels — so exactly one chunk streams, then stop.
        let spec = BacktestSpec {
            models: vec!["naive".into(), "drift".into()],
            horizon: 5,
            origins: 10,
            stride: 2,
            metrics: vec!["mase".into()],
            quantile_levels: vec![0.5],
            seed: 0,
        };
        let mut out: Vec<Envelope> = Vec::new();
        ctrl.feed_event_streaming(
            None,
            Event::BacktestRequest { panel, spec },
            &mut out,
            &mut CancelAfterOne(false),
        );
        // stopped before the final result; ended with a cancelled ack.
        assert!(!out.iter().any(|e| matches!(e.event, Event::BacktestResult { .. })), "{out:?}");
        assert!(matches!(out.last().map(|e| &e.event), Some(Event::Cancelled)), "{out:?}");
        // recovered: a fresh capabilities request is answered.
        let again = ctrl.feed_event(Event::CapabilitiesRequest);
        assert!(again.iter().any(|e| matches!(e.event, Event::CapabilitiesResult { .. })));
    }
}
