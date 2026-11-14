// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A deterministic, weight-free **mock video model**, served beside the other
//! mocks by `brain serve --mock`, so the whole video path - contract mapping,
//! the executor, encoding, the spool and every HTTP route above it - runs on a
//! machine with no GPU and no checkpoint.
//!
//! Registered as [`MODEL`], it declares the two shapes a real video model has:
//!
//! * **`t2v`** - text to video. Each frame is one flat colour derived from the
//!   seed and the frame index ([`frame_rgb`]), which survives AV1 compression
//!   well enough that a test can decode the delivered file and check that the
//!   frames it holds are the frames the mock drew. `audio=true` adds a sine
//!   track as a WAV blob, as the real audio-visual models do.
//! * **`extend`** - an edit: appends frames continuing the input clip's own
//!   colour walk, so a test can tell the original part from the new one.
//!
//! Both report one progress step per frame and poll the cancel token between
//! them, so progress and cancellation reach every layer above.
//!
//! Swedish Embedded AB implements solutions for testing generative-media
//! services end to end without GPUs for its clients. If your team needs
//! expertise in media pipeline testing then you can procure our services by
//! sending an email to info@swedishembedded.com.

use capability::video::{Audio, Fps, VideoContract};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

/// The catalog id the mock video model registers under.
pub const MODEL: &str = "brain/mock-video";

/// The sample rate of the mock's sound track.
pub const SAMPLE_RATE: u32 = 16_000;

/// The flat colour of frame `index` of a clip drawn with `seed`, as `[0,255]`.
/// Steps are large (32 per frame) so neighbouring frames stay distinguishable
/// after lossy coding.
pub fn frame_rgb(seed: i64, index: u32) -> [u8; 3] {
    let s = seed.unsigned_abs();
    [((s.wrapping_mul(37) + u64::from(index) * 32) % 256) as u8, ((s.wrapping_mul(91)) % 256) as u8, 128]
}

fn contract() -> VideoContract {
    VideoContract::new("frames", (1, 8), Fps::Param { name: "fps".into(), default: 8 }, ("width", "height"), 16, 864 * 480, (1.0, 8.0))
        .audio(Audio::Param { name: "audio".into(), default: false })
        .seed("seed")
}

fn t2v_spec() -> ActionSpec {
    ActionSpec::new("t2v", "deterministic mock text-to-video: one flat colour per frame").cancellable()
        .streaming()
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt (only its presence is checked)").required())
        .param(ParamSpec::new("frames", ParamType::Int, "frames, of the form 1 + 8k").default(json!(9)).min(1.0).max(321.0))
        .param(ParamSpec::new("width", ParamType::Int, "width, px, a multiple of 16").default(json!(256)).min(16.0).max(864.0))
        .param(ParamSpec::new("height", ParamType::Int, "height, px, a multiple of 16").default(json!(144)).min(16.0).max(864.0))
        .param(ParamSpec::new("fps", ParamType::Int, "frame rate").default(json!(8)).min(1.0).max(60.0))
        .param(ParamSpec::new("seed", ParamType::Int, "colour seed (omitted: 0, the mock is deterministic by design)"))
        .param(ParamSpec::new("audio", ParamType::Bool, "also make a sine sound track").default(json!(false)))
        .output(BlobSpec::new("video", Media::Video, "the clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required())
        .output(BlobSpec::new("audio", Media::Audio, "the sound track when audio=true: a mono WAV"))
        .video(contract())
}

fn extend_spec() -> ActionSpec {
    ActionSpec::new("extend", "deterministic mock clip extension: the clip followed by new frames continuing its colour walk").cancellable()
        .streaming()
        .input(BlobSpec::new("video", Media::Video, "the clip to continue, in the video blob wire format").required())
        .param(ParamSpec::new("prompt", ParamType::Str, "what the new part shows (only its presence is checked)").required())
        .param(ParamSpec::new("frames", ParamType::Int, "frames to add").default(json!(8)).min(1.0).max(240.0))
        .param(ParamSpec::new("seed", ParamType::Int, "colour seed of the new frames (omitted: 0)"))
        .output(BlobSpec::new("video", Media::Video, "the clip followed by the new frames").required())
        .video(contract().clip_input("video").audio(Audio::None))
}

/// The mock video model (stateless: every action is a pure function).
pub struct MockVideoResident;

impl ResidentModel for MockVideoResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(MODEL, "deterministic mock video model (text-to-video and extend)", vec![t2v_spec(), extend_spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(16 << 20, 0)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(MockVideoInstance))
    }
}

struct MockVideoInstance;

impl Instance for MockVideoInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        match action {
            "t2v" => t2v(inv, progress),
            "extend" => extend(inv, progress),
            other => Err(format!("mock-video: unknown action '{other}'")),
        }
    }
}

fn int(inv: &Invocation, name: &str) -> Result<u32, String> {
    inv.get_i64(name).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| format!("mock-video: '{name}' must be a non-negative integer"))
}

fn flat_frame(rgb: [u8; 3], w: u32, h: u32) -> (Vec<f32>, u32, u32) {
    let px = [f32::from(rgb[0]) / 255.0, f32::from(rgb[1]) / 255.0, f32::from(rgb[2]) / 255.0];
    (px.iter().copied().cycle().take(w as usize * h as usize * 3).collect(), w, h)
}

/// Draw `count` frames starting at colour index `first`, one progress step
/// each, stopping early on cancel.
fn draw(seed: i64, first: u32, count: u32, w: u32, h: u32, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
    let mut frames = Vec::with_capacity(count as usize);
    for i in 0..count {
        inv.cancel.check()?;
        frames.push(flat_frame(frame_rgb(seed, first + i), w, h));
        progress(Progress::step(i + 1, count, "frame"));
    }
    Ok(frames)
}

fn t2v(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let (frames, w, h, fps) = (int(inv, "frames")?, int(inv, "width")?, int(inv, "height")?, int(inv, "fps")?);
    if w % 16 != 0 || h % 16 != 0 {
        return Err(format!("mock-video: {w}x{h} is not a multiple of 16"));
    }
    let seed = inv.get_i64("seed").unwrap_or(0);
    let drawn = draw(seed, 0, frames, w, h, inv, progress)?;
    let mut video = capability::blob::video_blob(&drawn)?;
    video.meta["fps"] = json!(fps);
    let mut outcome = Outcome::new().set("seed", json!(seed)).blob("video", video);
    if inv.get_bool("audio").unwrap_or(false) {
        let seconds = f64::from(frames) / f64::from(fps);
        let samples: Vec<f32> = (0..(seconds * f64::from(SAMPLE_RATE)) as usize).map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3).collect();
        outcome = outcome.blob("audio", Blob::new(Media::Audio, audio::wav::encode(&samples, SAMPLE_RATE)).with_meta(json!({"format": "wav", "sample_rate": SAMPLE_RATE, "channels": 1})));
    }
    Ok(outcome)
}

fn extend(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let mut clip = capability::blob::decode_video(inv, "video")?;
    let fps = inv.get_blob("video").and_then(|b| b.meta.get("fps").cloned()).unwrap_or(json!(8));
    let (w, h) = clip.first().map(|(_, w, h)| (*w, *h)).ok_or("mock-video: the clip has no frames")?;
    let seed = inv.get_i64("seed").unwrap_or(0);
    let start = clip.len() as u32;
    clip.extend(draw(seed, start, int(inv, "frames")?, w, h, inv, progress)?);
    let mut video = capability::blob::video_blob(&clip)?;
    video.meta["fps"] = fps;
    Ok(Outcome::new().set("seed", json!(seed)).blob("video", video))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(action: &str, inv: Invocation) -> Outcome {
        let m = MockVideoResident;
        let spec = m.manifest().actions.into_iter().find(|a| a.name == action).unwrap();
        let inv = spec.validate(inv).unwrap();
        MockVideoInstance.run(action, &inv, &mut |_| {}).unwrap()
    }

    #[test]
    fn t2v_draws_the_requested_clip_with_its_rate_and_optional_sound() {
        let inv = Invocation::new().set("prompt", json!("a ball")).set("frames", json!(9)).set("width", json!(32)).set("height", json!(16)).set("seed", json!(3)).set("audio", json!(true));
        let out = run("t2v", inv);
        let v = &out.blobs["video"];
        assert_eq!((v.meta["frames"].as_u64(), v.meta["w"].as_u64(), v.meta["fps"].as_u64()), (Some(9), Some(32), Some(8)));
        let frames = capability::blob::decode_video(&Invocation::new().blob("video", v.clone()), "video").unwrap();
        assert_eq!((frames[4].0[0] * 255.0).round() as u8, frame_rgb(3, 4)[0]);
        assert!(audio::asr_caps::is_wav(&out.blobs["audio"].bytes));
    }

    #[test]
    fn extend_keeps_the_clip_and_continues_its_colour_walk() {
        let clip = run("t2v", Invocation::new().set("prompt", json!("x")).set("width", json!(16)).set("height", json!(16)));
        let out = run("extend", Invocation::new().set("prompt", json!("more")).set("frames", json!(4)).blob("video", clip.blobs["video"].clone()));
        let frames = capability::blob::decode_video(&Invocation::new().blob("video", out.blobs["video"].clone()), "video").unwrap();
        assert_eq!(frames.len(), 13);
        assert_eq!((frames[10].0[0] * 255.0).round() as u8, frame_rgb(0, 10)[0]);
    }

    /// Every served model's video contract names params and inputs its
    /// action really has: a renamed param would otherwise make `/videos`
    /// send one the action refuses.
    #[test]
    fn every_video_contract_matches_its_action() {
        let mut manifests = crate::manifests();
        manifests.push(MockVideoResident.manifest());
        let contracts: Vec<String> = manifests.iter().flat_map(|m| m.actions.iter().filter(|a| a.video.is_some()).map(move |a| format!("{}/{}", m.model, a.name))).collect();
        assert!(contracts.iter().any(|c| c.ends_with("ltxv/t2v")), "LTX t2v declares a contract: {contracts:?}");
        let problems: Vec<String> = manifests.iter().flat_map(|m| m.actions.iter().flat_map(|a| a.video.as_ref().map(|c| c.problems(a)).unwrap_or_default())).collect();
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn both_actions_declare_a_video_contract() {
        let m = MockVideoResident.manifest();
        assert!(m.actions.iter().all(|a| a.video.is_some()));
        assert_eq!(m.actions[1].video.as_ref().unwrap().clip_input.as_deref(), Some("video"));
    }
}
