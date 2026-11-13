// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Running a world model without a window: building the DIAMOND or fake
//! model a request names, rolling it out over an action script, verifying a
//! recorded episode dataset against it, and timing it.
//!
//! Everything here is plain data in and results out, so the served actions
//! ([`crate::caps`]) and a windowed front end drive one implementation. The
//! rollout is deterministic for a given model, seed and action script: DIAMOND
//! draws exactly one frame of normals per step, so the noise stream of a
//! recording is reproduced by the same seed and denoising-step count.
//!
//! Swedish Embedded AB implements world-model simulators that run on its
//! clients' own embedded hardware. If your team needs expertise in
//! action-conditioned video models and their verification then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;

use capability::{CancelToken, Progress};
use data::episode::EpisodeDataset;
use wm_core::{FakeWorldModel, WorldModel};
use wm_display::keymap::{Key, KeyChordMap, KeySet};
use wm_display::pacing::{FixedTimestep, WallClock};
use wm_display::record::RecorderSink;
use wm_display::sink::{fnv1a, FrameSink, HashSink, HeadlessSink, Hud, PpmDirSink};
use wm_display::{play_loop, InputSource, PlayReport, PolledInput, ScriptInput, SplitIo};

use crate::DiamondWorldModel;

/// Which world model a request runs.
#[derive(Clone, Debug)]
pub enum Kind {
    /// The pretrained Atari-100k EDM world model, from a brain checkpoint.
    Diamond { weights: String },
    /// The deterministic, GPU-free test model.
    Fake,
}

/// Everything that fixes the model a run starts from.
#[derive(Clone, Debug)]
pub struct ModelRequest {
    pub kind: Kind,
    /// An ONNX export of the UNet: the UNet then runs as a compiled OpenVINO
    /// graph on the Intel NPU and the sampler stays on the host.
    pub npu_onnx: Option<String>,
    pub seed: u64,
    /// Denoising steps per frame; 0 keeps the model's default.
    pub denoise_steps: u32,
    /// A directory of `.ppm` frames to start the world from, oldest first.
    pub seed_context: Option<String>,
}

/// A built model and where it runs.
pub struct Built {
    pub model: Box<dyn WorldModel>,
    pub backend: String,
}

/// The [`WorldModel::set_nfe`] quality code for `denoise_steps` (0 = default
/// 3 steps, 1 = 2 steps, 2 = 1 step); `None` leaves the model's default.
fn nfe_code(denoise_steps: u32) -> Option<u32> {
    match denoise_steps {
        0 => None,
        n if n >= 3 => Some(0),
        2 => Some(1),
        _ => Some(2),
    }
}

/// Build the model `req` names, reset to its start state.
pub fn build(req: &ModelRequest) -> Result<Built, String> {
    let (mut model, backend): (Box<dyn WorldModel>, String) = match &req.kind {
        Kind::Fake => {
            let mut m = FakeWorldModel::new();
            // Deterministic start position derived from the seed via the
            // context hash contract of FakeWorldModel::reset.
            let ctx: Vec<f32> = (0..8).map(|i| (req.seed >> (i * 8)) as u8 as f32 / 255.0).collect();
            m.reset(&ctx, &[]);
            (Box::new(m), "fake".into())
        }
        Kind::Diamond { weights } => match &req.npu_onnx {
            Some(onnx) => {
                let mut m = crate::npu::DiamondNpuWorldModel::load(weights, onnx, req.seed).map_err(|e| format!("cannot start the diamond NPU world model: {e}"))?;
                let backend = format!("npu ({})", m.device());
                m.reset(&[], &[]);
                (Box::new(m), backend)
            }
            None => {
                let (cfg, tensors) = crate::import::load(weights)?;
                let mut m = DiamondWorldModel::new(crate::DiamondUNet::new(cfg, &tensors, None), req.seed);
                m.reset(&[], &[]);
                (Box::new(m), "engine".into())
            }
        },
    };
    if let Some(code) = nfe_code(req.denoise_steps) {
        model.set_nfe(code);
    }
    if let Some(dir) = &req.seed_context {
        let (c, h, w) = model.frame_shape();
        let ctx = load_seed_context(dir, c, h, w)?;
        model.reset(&ctx, &[]);
    }
    Ok(Built { model, backend })
}

/// Sorted `*.ppm` frames of `dir` as trait-convention context (CHW f32
/// [0,1], oldest first, concatenated).
fn load_seed_context(dir: &str, c: u32, h: u32, w: u32) -> Result<Vec<f32>, String> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("seed context {dir}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ppm"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("seed context {dir}: no .ppm files"));
    }
    let mut out = vec![];
    for p in paths {
        let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let img = imaging::decode(&bytes).map_err(|e| format!("{}: {e}", p.display()))?;
        if (img.w, img.h) != (w, h) {
            return Err(format!("{}: expected {w}x{h}, got {}x{}", p.display(), img.w, img.h));
        }
        // HWC u8 -> CHW f32 [0,1]: the cast and the permutation are separate
        // steps in `imaging` precisely so neither hides inside the other.
        let chw = imaging::pixels::hwc_to_chw(&img.to_hwc_unit(), 3, h as usize, w as usize);
        // Frames are RGB; a model configured for more channels leaves the rest zero.
        let mut padded = vec![0.0f32; (c * h * w) as usize];
        let n = chw.len().min(padded.len());
        padded[..n].copy_from_slice(&chw[..n]);
        out.extend(padded);
    }
    Ok(out)
}

/// Atari keymap (Breakout-style 4 actions): Space=FIRE, D=RIGHT, A=LEFT.
pub fn atari_keymap(num_actions: u32) -> KeyChordMap {
    let mut chords = vec![];
    if num_actions > 1 {
        chords.push((KeySet::of(&[Key::Space]), 1));
    }
    if num_actions > 2 {
        chords.push((KeySet::of(&[Key::D]), 2));
        chords.push((KeySet::of(&[Key::Right]), 2));
    }
    if num_actions > 3 {
        chords.push((KeySet::of(&[Key::A]), 3));
        chords.push((KeySet::of(&[Key::Left]), 3));
    }
    KeyChordMap::new(chords, 0)
}

/// The chord map a model is played with: Atari keys for DIAMOND, WASD for the
/// fake model.
pub fn keymap_for(kind: &Kind, num_actions: u32) -> KeyChordMap {
    match kind {
        Kind::Diamond { .. } => atari_keymap(num_actions),
        Kind::Fake => KeyChordMap::wasd(0),
    }
}

/// The action ids in `text`, separated by whitespace and/or commas.
pub fn parse_actions(text: &str) -> Result<Vec<u32>, String> {
    text.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().map_err(|_| format!("bad action id {t:?}")))
        .collect()
}

/// Where a rollout's frames go besides the clip it returns.
#[derive(Clone, Debug, Default)]
pub struct RolloutTargets {
    /// A directory to write `frame_NNNNNN.ppm` into.
    pub ppm_dir: Option<String>,
    /// An episode-dataset directory to record the session into.
    pub record_dir: Option<String>,
}

/// What a rollout produced.
pub struct Rollout {
    pub report: PlayReport,
    /// Every frame as RGB8, `w * h * 3` bytes each.
    pub frames: Vec<Vec<u8>>,
    pub frame_hashes: Vec<u64>,
    pub width: u32,
    pub height: u32,
    /// Frames recorded into the episode dataset, when one was asked for.
    pub recorded: Option<usize>,
}

impl Rollout {
    /// One hash over every frame's hash: equal for equal rollouts.
    pub fn rollout_hash(&self) -> u64 {
        let bytes: Vec<u8> = self.frame_hashes.iter().flat_map(|h| h.to_le_bytes()).collect();
        fnv1a(&bytes)
    }
}

/// Collects every frame, hashes it, and feeds the optional PPM dump and
/// episode recorder, so a rollout is ONE sink pipeline however it is asked.
struct RolloutSink<'a> {
    hashes: HashSink,
    frames: Vec<Vec<u8>>,
    ppm: Option<PpmDirSink>,
    recorder: Option<RecorderSink>,
    total: u32,
    progress: &'a mut dyn FnMut(Progress),
}

impl FrameSink for RolloutSink<'_> {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        self.hashes.frame(rgb, w, h, hud);
        if let Some(p) = &mut self.ppm {
            p.frame(rgb, w, h, hud);
        }
        if let Some(r) = &mut self.recorder {
            r.frame(rgb, w, h, hud);
        }
        self.frames.push(rgb.to_vec());
        (self.progress)(Progress::step(hud.step as u32, self.total, format!("frame {}", hud.step)));
    }
}

/// Ends a scripted input early once `cancel` is armed.
struct CancellableScript<'a> {
    script: ScriptInput,
    cancel: &'a CancelToken,
}

impl InputSource for CancellableScript<'_> {
    fn poll(&mut self) -> PolledInput {
        if self.cancel.is_cancelled() {
            return PolledInput { quit: true, ..Default::default() };
        }
        self.script.poll()
    }
}

/// Roll `model` out over `actions`, unpaced. `Err("cancelled")` if `cancel`
/// fired; a recording started before that is not published.
pub fn rollout(
    model: &mut dyn WorldModel,
    model_name: &str,
    actions: Vec<u32>,
    fps: u32,
    targets: &RolloutTargets,
    cancel: &CancelToken,
    progress: &mut dyn FnMut(Progress),
) -> Result<Rollout, String> {
    if let Some(bad) = actions.iter().find(|&&a| a >= model.num_actions()) {
        return Err(format!("action {bad} is out of range: the model has {} actions", model.num_actions()));
    }
    let steps = actions.len() as u64;
    let ppm = targets.ppm_dir.as_deref().map(|d| PpmDirSink::new(d).map_err(|e| format!("cannot create {d}: {e}"))).transpose()?;
    let recorder = targets.record_dir.as_deref().map(|d| RecorderSink::new(d, model.num_actions(), fps));
    let mut sink = RolloutSink { hashes: HashSink::default(), frames: vec![], ppm, recorder, total: steps as u32, progress };
    let mut input = CancellableScript { script: ScriptInput::new(actions), cancel };
    let pacer = FixedTimestep::new(WallClock::new(), UNPACED_FPS, false);
    let report = {
        let mut io = SplitIo { input: &mut input, sink: &mut sink };
        play_loop(model, &mut io, &KeyChordMap::wasd(0), pacer, fps, model_name, Some(steps))
    };
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let recorded = match sink.recorder.take() {
        Some(rec) => {
            let n = rec.frames_recorded();
            rec.finalize()?;
            Some(n)
        }
        None => None,
    };
    let (_, height, width) = model.frame_shape();
    Ok(Rollout { report, frames: sink.frames, frame_hashes: sink.hashes.hashes, width, height, recorded })
}

/// A fixed-timestep target so high that headless runs never sleep.
const UNPACED_FPS: u32 = 100_000;

/// The result of comparing a model's regenerated frames with a recording.
#[derive(Debug)]
pub struct Verification {
    pub checked: usize,
    pub max_mad: f32,
    pub worst_frame: usize,
    pub failed: usize,
}

/// How to verify an episode dataset.
pub struct VerifyRequest {
    pub tolerance: f32,
    /// Steps burned (frames discarded) before the recorded frames are loaded
    /// as context, so verification starts at that frame with the noise stream
    /// of a stochastic model still aligned. Only models whose reset is a pure
    /// context load (diamond) match with a non-zero value; the fake model
    /// derives its state from a context hash and verifies with 0.
    pub context: usize,
}

/// Re-generate `ds` through `model` and compare it frame by frame (mean
/// absolute difference over RGB8). Exact by construction for deterministic
/// models, and for DIAMOND when the seed and denoising steps match the
/// recording's.
pub fn verify(model: &mut dyn WorldModel, ds: &EpisodeDataset, req: &VerifyRequest) -> Result<Verification, String> {
    let (c, h, w) = model.frame_shape();
    if (c, h, w) != (ds.c, ds.h, ds.w) {
        return Err(format!("model frames are {c}x{h}x{w} but the dataset is {}x{}x{}", ds.c, ds.h, ds.w));
    }
    if req.context >= ds.n {
        return Err(format!("context {} leaves no frames to verify (n = {})", req.context, ds.n));
    }
    let actions = ds.actions().to_vec();
    if let Some(bad) = actions.iter().find(|&&a| a >= model.num_actions()) {
        return Err(format!("the recording uses action {bad}, outside the model's {} actions", model.num_actions()));
    }
    if req.context > 0 {
        // Burn the context steps so a stochastic model's rng lands where it
        // was at record time, then load the recorded frames as context.
        for &act in &actions[..req.context] {
            let _ = model.step(act);
        }
        let mut ctx = Vec::with_capacity(req.context * ds.frame_len());
        for i in 0..req.context {
            ctx.extend(ds.frame_f32(i)?);
        }
        model.reset(&ctx, &actions[..req.context]);
    }
    let mut v = Verification { checked: ds.n - req.context, max_mad: 0.0, worst_frame: req.context, failed: 0 };
    for (i, &action) in actions.iter().enumerate().take(ds.n).skip(req.context) {
        let f = model.step(action);
        // The exact conversion the recorder applied: f32 -> RGB8 -> CHW u8.
        let rgb = imaging::pixels::chw_to_rgb8(&f, w, h, c as usize, imaging::ChannelPolicy::RequireRgb)?.px;
        let got = imaging::pixels::hwc_to_chw(&rgb, 3, h as usize, w as usize);
        let want = ds.frame(i)?;
        let sum: u64 = got.iter().zip(&want).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum();
        let mad = sum as f32 / (got.len() as f32 * 255.0);
        if mad > v.max_mad {
            v.max_mad = mad;
            v.worst_frame = i;
        }
        if mad > req.tolerance {
            v.failed += 1;
        }
    }
    Ok(v)
}

/// Open the episode dataset at `dir`.
pub fn open_dataset(dir: &str) -> Result<EpisodeDataset, String> {
    EpisodeDataset::open(Path::new(dir)).map_err(|e| format!("cannot open {dir}: {e}"))
}

/// Per-frame timings of an unpaced run of `frames` steps cycling through the
/// model's actions.
pub struct BenchResult {
    pub report: PlayReport,
    pub total_s: f32,
}

pub fn bench(model: &mut dyn WorldModel, model_name: &str, frames: u64) -> BenchResult {
    let na = model.num_actions() as u64;
    let mut input = ScriptInput::new((0..frames).map(|i| (i % na) as u32).collect());
    let mut sink = HeadlessSink;
    let mut io = SplitIo { input: &mut input, sink: &mut sink };
    let t0 = std::time::Instant::now();
    let report = play_loop(model, &mut io, &KeyChordMap::wasd(0), FixedTimestep::new(WallClock::new(), UNPACED_FPS, false), 0, model_name, Some(frames));
    BenchResult { report, total_s: t0.elapsed().as_secs_f32() }
}

/// One UNet forward's per-kernel timings, `(kernel, ms, dispatches)`, with
/// one submit per step: for ranking kernels, not for absolute time.
pub fn profile_unet(weights: &str) -> Result<Vec<(String, f64, u32)>, String> {
    let (cfg, tensors) = crate::import::load(weights)?;
    let unet = crate::DiamondUNet::new(cfg.clone(), &tensors, None);
    let n = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let noisy = vec![0.1f32; n];
    let obs = vec![0.0f32; n * cfg.num_steps_conditioning as usize];
    unet.set_context(&obs);
    let actions = vec![0u32; cfg.num_steps_conditioning as usize];
    // Warm up (JIT compile / pipeline setup), then profile.
    let _ = unet.forward(&noisy, 0.1, &actions);
    Ok(unet.profile_forward(&noisy, 0.1, &actions).into_iter().map(|(name, ms, count)| (name.to_string(), ms, count)).collect())
}
