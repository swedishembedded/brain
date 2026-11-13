// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `play` workflow: the world model in an SDL window, one step per paced
//! tick, with an optional recording of the session as an episode dataset.
//!
//! The model is built by `diamond::worldrun::build` from the same params the
//! served `rollout` takes, so a window session and a headless rollout of one
//! seed start from the same world.
//!
//! Swedish Embedded AB implements interactive simulators on learned world
//! models for its clients. If your team needs expertise in real-time model
//! rollouts with human input then you can procure our services by sending an
//! email to info@swedishembedded.com.

use diamond::worldrun::{build, keymap_for};
use wm_display::pacing::{FixedTimestep, WallClock};
use wm_display::record::RecorderSink;
use wm_display::sink::{FrameSink, Hud};
use wm_display::window::{Renderer, SdlWindow};
use wm_display::{play_loop, PlayIo, PolledInput};

/// The served `rollout` params a window also takes: the model, the seed, the
/// sampler quality, the NPU graph, the starting frames and the recording.
const SHARED_PARAMS: [&str; 7] = ["model", "seed", "denoise_steps", "onnx", "seed_context", "fps", "record"];

/// The default window scale: DIAMOND frames are 64x64.
const DEFAULT_SCALE: u32 = 10;

/// What the window takes off the command line before the rest is read as
/// `rollout`'s.
#[derive(Debug)]
struct WindowFlags {
    scale: u32,
    max_steps: Option<u64>,
    adaptive: bool,
    renderer: Renderer,
}

fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    let Some(i) = args.iter().position(|a| a == flag) else { return Ok(None) };
    let v = args.get(i + 1).cloned().ok_or_else(|| format!("{flag} needs a value"))?;
    args.drain(i..=i + 1);
    Ok(Some(v))
}

fn take_count(args: &mut Vec<String>, flag: &str) -> Result<Option<u64>, String> {
    take_value(args, flag)?.map(|v| v.parse::<u64>().map_err(|_| format!("{flag} {v}: not a count"))).transpose()
}

fn window_flags(args: &mut Vec<String>) -> Result<WindowFlags, String> {
    let scale = take_count(args, "--scale")?.map_or(DEFAULT_SCALE, |s| s.clamp(1, 64) as u32);
    let max_steps = take_count(args, "--frames")?.filter(|&n| n > 0);
    let renderer = take_value(args, "--renderer")?.map_or(Ok(Renderer::Auto), |v| Renderer::parse(&v))?;
    let adaptive = args.iter().position(|a| a == "--adaptive").map(|i| args.remove(i)).is_some();
    Ok(WindowFlags { scale, max_steps, adaptive, renderer })
}

/// The SDL window serves input and display while an optional recorder tees
/// the frames off into an episode dataset.
struct RecordingWindow<'a> {
    window: &'a mut SdlWindow,
    recorder: Option<&'a mut RecorderSink>,
}

impl PlayIo for RecordingWindow<'_> {
    fn poll(&mut self) -> PolledInput {
        PlayIo::poll(self.window)
    }
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        FrameSink::frame(self.window, rgb, w, h, hud);
        if let Some(recorder) = self.recorder.as_deref_mut() {
            recorder.frame(rgb, w, h, hud);
        }
    }
}

pub fn play(call: &invoke::WorkflowCall) -> Result<(), String> {
    let mut rest = call.args.to_vec();
    let flags = window_flags(&mut rest)?;
    let mut spec = diamond::caps::rollout_spec();
    spec.params.retain(|p| SHARED_PARAMS.contains(&p.name.as_str()));
    spec.outputs.clear();
    let parsed = invoke::args::check(call.program, &spec, &rest).map_err(|_| "see the flag error above".to_string())?;
    let inv = &parsed.invocation;

    let (request, _seed) = diamond::caps::model_request(inv, call.assembly.role_path("weights").ok(), "play")?;
    let mut built = build(&request)?;
    eprintln!("{}: world model on {}", call.program, built.backend);
    let model = built.model.as_mut();
    let (_, h, w) = model.frame_shape();
    let fps = inv.get_i64("fps").unwrap_or(15).max(1) as u32;
    let keymap = keymap_for(&request.kind, model.num_actions());
    let mut recorder = inv.get_str("record").filter(|d| !d.is_empty()).map(|dir| RecorderSink::new(dir, model.num_actions(), fps));

    let mut window = SdlWindow::with_renderer("brain diamond", w, h, flags.scale, flags.renderer).map_err(|e| format!("cannot open an SDL window: {e} (headless? try SDL_VIDEODRIVER=dummy)"))?;
    let report = {
        let mut io = RecordingWindow { window: &mut window, recorder: recorder.as_mut() };
        let pacer = FixedTimestep::new(WallClock::new(), fps, flags.adaptive);
        play_loop(model, &mut io, &keymap, pacer, fps, inv.get_str("model").as_deref().unwrap_or("diamond"), flags.max_steps)
    };
    if let Some(recorder) = recorder {
        let frames = recorder.frames_recorded();
        recorder.finalize()?;
        println!("recorded {frames} frames");
    }
    println!("steps={} fps={:.1} work_ms_mean={:.2} work_ms_p95={:.2}", report.steps, report.fps, report.work_ms_mean, report.work_ms_p95);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_flags_are_taken_off_before_the_actions_own() {
        let mut args: Vec<String> = ["--scale", "4", "--adaptive", "--renderer", "software", "--frames", "30", "--seed", "7"].map(String::from).to_vec();
        let flags = window_flags(&mut args).unwrap();
        assert_eq!((flags.scale, flags.max_steps, flags.adaptive, flags.renderer), (4, Some(30), true, Renderer::Software));
        assert_eq!(args, ["--seed", "7"]);
    }

    #[test]
    fn a_bad_window_flag_is_named() {
        assert!(window_flags(&mut vec!["--renderer".into(), "vulkan".into()]).unwrap_err().contains("vulkan"));
        assert!(window_flags(&mut vec!["--scale".into(), "big".into()]).unwrap_err().contains("--scale"));
    }
}
