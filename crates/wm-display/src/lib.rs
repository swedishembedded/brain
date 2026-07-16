// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Realtime display + input for world models.
//!
//! A [`wm_core::WorldModel`] steps once per paced tick; frames go to a
//! [`sink::FrameSink`] (the SDL window, a hash sink for golden tests, a PPM
//! recorder, or a tee of several). Input is a latest-state pressed-key
//! bitmask mapped through a longest-chord-wins [`keymap::KeyChordMap`], so
//! human latency is at most one frame period and a step never consumes a
//! stale queued action.
//!
//! The SDL window (feature `sdl`, default on) uses SDL2's SOFTWARE renderer
//! deliberately: presentation stays on the CPU and the whole iGPU compute
//! budget belongs to the model.

pub mod keymap;
pub mod pacing;
pub mod record;
pub mod sink;
mod sys;
pub mod window;

use keymap::{KeyChordMap, UxKey};
use pacing::{Clock, FixedTimestep};
use sink::{FrameSink, Hud};
use wm_core::WorldModel;

/// Convert a CHW f32 [0,1] frame to interleaved RGB8.
pub fn chw_to_rgb8(frame: &[f32], c: u32, h: u32, w: u32) -> Vec<u8> {
    assert_eq!(frame.len(), (c * h * w) as usize);
    assert!(c >= 3, "need at least 3 channels for RGB, got {c}");
    let plane = (h * w) as usize;
    let mut rgb = vec![0u8; plane * 3];
    for i in 0..plane {
        for ch in 0..3 {
            let v = frame[ch * plane + i].clamp(0.0, 1.0);
            rgb[i * 3 + ch] = (v * 255.0 + 0.5) as u8;
        }
    }
    rgb
}

/// Outcome of a play session.
#[derive(Clone, Debug, Default)]
pub struct PlayReport {
    pub steps: u64,
    pub fps: f32,
    /// Mean work ms per model step.
    pub work_ms_mean: f32,
    /// p95 work ms per model step.
    pub work_ms_p95: f32,
}

/// Input source for the play loop: the window, or a scripted list for
/// headless runs.
pub trait InputSource {
    /// Poll current input; `None` action means "use the chord map on keys".
    fn poll(&mut self) -> PolledInput;
}

#[derive(Clone, Debug, Default)]
pub struct PolledInput {
    pub pressed: keymap::KeySet,
    pub ux: Vec<UxKey>,
    pub quit: bool,
    /// Scripted override: when set, bypasses the chord map entirely.
    pub forced_action: Option<u32>,
}

/// Scripted input: one action id per step, then quit. Deterministic driver
/// for golden-hash tests and `--headless` runs.
pub struct ScriptInput {
    actions: Vec<u32>,
    i: usize,
}

impl ScriptInput {
    pub fn new(actions: Vec<u32>) -> ScriptInput {
        ScriptInput { actions, i: 0 }
    }
}

impl InputSource for ScriptInput {
    fn poll(&mut self) -> PolledInput {
        if self.i >= self.actions.len() {
            return PolledInput { quit: true, ..Default::default() };
        }
        let a = self.actions[self.i];
        self.i += 1;
        PolledInput { forced_action: Some(a), ..Default::default() }
    }
}

/// Combined input+output endpoint of the play loop. The SDL window is one
/// object serving both roles; headless runs pair a script with a sink via
/// [`SplitIo`].
pub trait PlayIo {
    fn poll(&mut self) -> PolledInput;
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud);
}

/// Pairs any [`InputSource`] with any [`FrameSink`] as one [`PlayIo`].
pub struct SplitIo<'a, I: InputSource + ?Sized, S: FrameSink + ?Sized> {
    pub input: &'a mut I,
    pub sink: &'a mut S,
}

impl<I: InputSource + ?Sized, S: FrameSink + ?Sized> PlayIo for SplitIo<'_, I, S> {
    fn poll(&mut self) -> PolledInput {
        self.input.poll()
    }
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        self.sink.frame(rgb, w, h, hud);
    }
}

impl PlayIo for window::SdlWindow {
    fn poll(&mut self) -> PolledInput {
        let i = self.pump();
        PolledInput { pressed: i.pressed, ux: i.ux, quit: i.quit, forced_action: None }
    }
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        FrameSink::frame(self, rgb, w, h, hud);
    }
}

/// The play loop: pump input, map to an action, step the model once per
/// paced tick, hand the frame to the sink. Runs until quit or `max_steps`.
pub fn play_loop<C: Clock>(
    model: &mut dyn WorldModel,
    io: &mut dyn PlayIo,
    keymap: &KeyChordMap,
    mut pacer: FixedTimestep<C>,
    target_fps: u32,
    model_name: &str,
    max_steps: Option<u64>,
) -> PlayReport {
    let (c, h, w) = model.frame_shape();
    let mut hud = Hud { model: model_name.to_string(), target_fps, ..Default::default() };
    let mut works: Vec<u64> = vec![];
    let mut quality: i64 = 0;
    // Set when a UxKey::Reset was applied; carried on the NEXT emitted frame
    // as `hud.reset` (the first frame of the new episode), surviving paused
    // ticks in between.
    let mut pending_reset = false;

    loop {
        if let Some(m) = max_steps {
            if hud.step >= m {
                break;
            }
        }
        let polled = io.poll();
        if polled.quit {
            break;
        }
        for ux in &polled.ux {
            match ux {
                UxKey::Pause => hud.paused = !hud.paused,
                UxKey::Reset => {
                    // Rewind to the initial seed context (re-seeding the RNG),
                    // not a blank/random restart — Enter returns to the start.
                    model.reset_initial();
                    pending_reset = true;
                }
                UxKey::QualityDown => {
                    quality = (quality - 1).max(-3);
                    apply_quality(model, quality);
                    hud.quality = (-quality) as u32;
                }
                UxKey::QualityUp => {
                    quality = (quality + 1).min(0);
                    apply_quality(model, quality);
                    hud.quality = (-quality) as u32;
                }
                UxKey::StepOnce | UxKey::Quit => {}
            }
        }
        let step_once = polled.ux.contains(&UxKey::StepOnce);
        if hud.paused && !step_once {
            // Keep pumping/presenting while paused, without stepping.
            pacer.tick(|_| {});
            continue;
        }

        let action = polled
            .forced_action
            .unwrap_or_else(|| keymap.action(polled.pressed));

        let mut frame: Vec<f32> = vec![];
        let tick = pacer.tick(|_| {
            frame = model.step(action);
        });
        works.push(tick.work_ms);
        hud.fps = tick.fps;
        hud.step += 1;
        if let Some(dq) = tick.quality_delta {
            quality = (quality + dq as i64).clamp(-3, 0);
            apply_quality(model, quality);
            hud.quality = (-quality) as u32;
        }

        hud.action = action;
        hud.reset = pending_reset;
        pending_reset = false;
        let rgb = chw_to_rgb8(&frame, c, h, w);
        io.frame(&rgb, w, h, &hud);
    }

    works.sort_unstable();
    let mean = if works.is_empty() {
        0.0
    } else {
        works.iter().sum::<u64>() as f32 / works.len() as f32
    };
    let p95 = works
        .get((works.len().saturating_sub(1)) * 95 / 100)
        .copied()
        .unwrap_or(0) as f32;
    PlayReport { steps: hud.step, fps: hud.fps, work_ms_mean: mean, work_ms_p95: p95 }
}

/// Quality level `q<=0` -> `set_nfe(base >> -q)` convention: level 0 is the
/// model's default; each level halves the NFE knob. Models clamp internally.
fn apply_quality(model: &mut dyn WorldModel, q: i64) {
    // 0 => leave default; negative => degrade. The knob is model-defined;
    // we pass a small code the model interprets (0 = default).
    let nfe_code = if q >= 0 { 0 } else { (-q) as u32 };
    model.set_nfe(nfe_code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pacing::MockClock;
    use sink::HashSink;
    use wm_core::FakeWorldModel;

    fn run_scripted(actions: Vec<u32>) -> (PlayReport, Vec<u64>) {
        let mut model = FakeWorldModel::new();
        model.reset(&[], &[]);
        let mut input = ScriptInput::new(actions);
        let mut sink = HashSink::default();
        let pacer = FixedTimestep::new(MockClock { now: 0 }, 15, false);
        let km = KeyChordMap::wasd(0);
        let mut io = SplitIo { input: &mut input, sink: &mut sink };
        let report = play_loop(&mut model, &mut io, &km, pacer, 15, "fake", None);
        (report, sink.hashes)
    }

    #[test]
    fn playloop_scripted_run_is_deterministic() {
        let (r1, h1) = run_scripted(vec![1, 1, 2, 3, 4, 0, 4, 4]);
        let (r2, h2) = run_scripted(vec![1, 1, 2, 3, 4, 0, 4, 4]);
        assert_eq!(r1.steps, 8);
        assert_eq!(r2.steps, 8);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 8);
    }

    #[test]
    fn playloop_actions_change_frames() {
        let (_, still) = run_scripted(vec![0, 0, 0, 0]);
        let (_, moving) = run_scripted(vec![4, 4, 4, 4]);
        assert_ne!(still, moving);
    }

    #[test]
    fn playloop_chw_to_rgb8_clamps_and_rounds() {
        // 1x1 frame, 3 channels: -0.5 -> 0, 0.5 -> 128, 2.0 -> 255.
        let rgb = chw_to_rgb8(&[-0.5, 0.5, 2.0], 3, 1, 1);
        assert_eq!(rgb, vec![0, 128, 255]);
    }
}
