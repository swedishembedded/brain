// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan's capabilities, declared through the generalized [`capability`]
//! interface - what makes `brain caps brain/wan`, the event API and the D-Bus
//! surface work with no Wan-specific plumbing in the CLI or the runtime.
//!
//! The one path this does NOT reach is a capability-dispatched CLI verb:
//! `wan` has a dedicated `_cli.rs` handler, and the resolver gives those
//! precedence over the generic `ARCH_TO_MODEL` dispatch (an id in both would
//! make the generic path silently dead, which its own invariant test
//! forbids). So `brain wan t2v` runs `wan_cli`, not [`WanAction`] - the same
//! `crate::pipeline` underneath either way. `flux2-klein` is in exactly this
//! state for the same reason.
//!
//! The manifest is **static** (no weights needed) so capability *discovery* is
//! free; only [`WanProvider`] (execution) loads anything. One action today,
//! `t2v`, whose parameter defaults are [`WanConfig`]'s - i.e. upstream's own
//! `generate.py` defaults, so an invocation that names only a prompt runs what
//! upstream would run.
//!
//! ## Two things this action does that a short one does not have to
//!
//! * **It polls `inv.cancel` every denoise step.** A 480p run is measured in
//!   tens of minutes; an action that length which cannot be aborted is not
//!   servable, whatever else it does. The poll lives in [`crate::pipeline`]'s
//!   loop, which is shared by every caller.
//! * **It holds the DiT resident across calls** ([`crate::pipeline::HotDit`]),
//!   keyed on the only things that fix the built graphs: the variant, the
//!   latent extent and the device. A cold call pays ~20 s of load plus 5.7 GB
//!   of upload at 1.3B; a second request at the same size pays neither.
//!
//! The execution helpers below are `pub fn`s shared by BOTH [`WanProvider`]
//! and the residency adapter (`crates/cli/src/resident_wan.rs`) - one
//! implementation of param decoding, generation and outcome shaping, the
//! `flux2::caps` pattern.

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

use crate::config::WanConfig;
use crate::pipeline::{GenOpts, HotDit, Paths, Solver};

/// The model id used on the CLI (`brain do brain/wan …`) and the event API.
pub const MODEL: &str = "brain/wan";

/// The variant enum, in manifest order. T2V only: I2V needs a 36-channel
/// input and the CLIP vision tower, neither of which this crate has yet, and
/// advertising an action that cannot run is worse than not advertising it.
const VARIANTS: [&str; 2] = ["t2v-1.3B", "t2v-14B"];

/// The multistep solver enum. Both are gated bit-exactly against the
/// reference; they do not share a starting sigma, so this is a real choice.
const SOLVERS: [&str; 2] = ["unipc", "dpm++"];

/// The variant a name selects, or an error naming what IS available.
pub fn config_from_name(name: &str) -> Result<WanConfig, String> {
    match name {
        "t2v-1.3B" => Ok(WanConfig::t2v_1_3b()),
        "t2v-14B" => Ok(WanConfig::t2v_14b()),
        other => Err(format!("unknown wan variant {other:?} ({})", VARIANTS.join(", "))),
    }
}

/// The full, static capability manifest - safe to build with no weights
/// loaded. Defaults are the 1.3B variant's, i.e. [`GenOpts::from_config`]'s.
pub fn manifest() -> Manifest {
    let d = GenOpts::from_config(&WanConfig::t2v_1_3b());
    let t2v = ActionSpec::new(
        "t2v",
        "generate a video clip from a text prompt (umT5-XXL conditioning, flow-matching denoise over a 3D latent volume with CFG, causal 3D VAE decode)",
    )
    .streaming()
    .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired clip").required())
    .param(ParamSpec::new("negative_prompt", ParamType::Str, "what to avoid; omit for upstream's own sample_neg_prompt, pass \"\" for none"))
    .param(ParamSpec::new("frames", ParamType::Int, "video frames; must be of the form 1 + 4k (the causal VAE gives the first frame its own latent frame)").default(json!(d.frames)))
    .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 16)").default(json!(d.width)))
    .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 16)").default(json!(d.height)))
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps").default(json!(d.steps)))
    .param(ParamSpec::new("shift", ParamType::Float, "flow-matching sigma shift").default(json!(d.shift)))
    .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance; <= 1.0 runs ONE forward per step instead of two").default(json!(d.guidance)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise seed (omit for 0)").default(json!(0)))
    .param(ParamSpec::new("fps", ParamType::Int, "frame rate reported with the clip").default(json!(d.fps)))
    .param(ParamSpec::new("solver", ParamType::Enum(SOLVERS.iter().map(|s| s.to_string()).collect()), "multistep flow-matching solver").default(json!("unipc")))
    .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "model variant; the checkpoint at BRAIN_WAN_DIT must match").default(json!("t2v-1.3B")))
    .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required());

    Manifest::new(
        MODEL,
        "Wan2.1 (Alibaba) - text-to-video diffusion transformer over a 3D latent volume: umT5-XXL text conditioning, flow-matching UniPC/DPM++ sampling with CFG, causal 3D VAE at (4, 8, 8) stride.",
        vec![t2v],
    )
}

// ===================== shared execution helpers =====================
//
// Both the hot-DiT [`WanProvider`] and the residency adapter
// (`crates/cli/src/resident_wan.rs`) run `t2v` through these - ONE
// implementation of param decoding, generation and outcome shaping.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, Invocation, Outcome, Progress, Provider};

/// A decoded `t2v` request: the variant config, its name, and the per-call
/// [`GenOpts`].
#[derive(Debug)]
pub struct GenParams {
    pub cfg: WanConfig,
    pub variant: String,
    pub opts: GenOpts,
}

/// Decode + validate `t2v`'s params from an invocation. Every geometric
/// constraint is checked HERE, before a byte of the 17.6 GB checkpoint is
/// read: a bad frame count or a size the patch grid cannot tile must not cost
/// a weight load to discover.
pub fn gen_params_from(inv: &Invocation) -> Result<GenParams, String> {
    let variant = inv.get_str("variant").unwrap_or_else(|| "t2v-1.3B".into());
    let cfg = config_from_name(&variant)?;
    let d = GenOpts::from_config(&cfg);
    let opts = GenOpts {
        frames: inv.get_i64("frames").unwrap_or(d.frames as i64).max(1) as usize,
        width: inv.get_i64("width").unwrap_or(d.width as i64).max(16) as usize,
        height: inv.get_i64("height").unwrap_or(d.height as i64).max(16) as usize,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        shift: inv.get_f64("shift").unwrap_or(d.shift as f64) as f32,
        guidance: inv.get_f64("guidance").unwrap_or(d.guidance as f64) as f32,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        // Absent = upstream's own negative prompt; an explicitly empty string
        // is a real request for none, so it must survive as `Some("")`.
        negative_prompt: inv.get_str("negative_prompt"),
        solver: Solver::from_name(&inv.get_str("solver").unwrap_or_else(|| "unipc".into()))?,
        fps: inv.get_i64("fps").unwrap_or(d.fps as i64).max(1) as usize,
        device: None,
        te_device: None,
    };
    if cfg.latent_shape(opts.frames, opts.width, opts.height).is_none() {
        return Err(format!("frames must be of the form 1 + 4k (1, 5, 9, … 81); got {}", opts.frames));
    }
    let (_, ph, pw) = cfg.patch_size;
    let (_, sh, sw) = cfg.vae_stride;
    if !opts.width.is_multiple_of(sw * pw) || !opts.height.is_multiple_of(sh * ph) {
        return Err(format!("{}x{} is not a multiple of {}x{} (VAE stride x patch size)", opts.width, opts.height, sw * pw, sh * ph));
    }
    Ok(GenParams { cfg, variant, opts })
}

/// Run one generation against a caller-owned resident DiT and wrap the result
/// as a video-output [`Outcome`] (the shared `capability::blob` wire format).
/// Cancellation rides in `inv.cancel` - [`crate::pipeline`] polls it per
/// denoise step.
pub fn generate_on(paths: &Paths, hot: &mut Option<HotDit>, inv: &Invocation, p: &GenParams, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let (video, timings) = crate::pipeline::generate_hot(&p.cfg, paths, &prompt, &p.opts, &inv.cancel, hot, |done, total, phase| {
        progress(Progress::step(done, total, phase.to_string()))
    })?;
    Ok(video_outcome(&video, &timings))
}

/// Wrap a generated clip as a video-output [`Outcome`] - ONE implementation,
/// shared by the provider and the residency adapter.
///
/// The frames go out through `capability::blob::video_blob` (brain's one
/// clip wire format), with `fps` added to its metadata: a clip without its
/// frame rate is not playable, and the alternative - a scalar output the
/// blob writer would have to be taught to look for - splits one fact across
/// two places.
pub fn video_outcome(video: &crate::pipeline::Video, timings: &crate::pipeline::Timings) -> Outcome {
    let frames: Vec<(Vec<f32>, u32, u32)> =
        video.frames.iter().map(|px| (px.iter().map(|&b| b as f32 / 255.0).collect::<Vec<f32>>(), video.width, video.height)).collect();
    let mut blob = match capability::blob::video_blob(&frames) {
        Ok(b) => b,
        // Unreachable for a real generation (the pipeline already checked the
        // buffer length against w*h*3 per frame), so this reports rather than
        // panics inside a serving thread.
        Err(e) => return Outcome::new().set("error", json!(e)),
    };
    if let Some(m) = blob.meta.as_object_mut() {
        m.insert("fps".to_string(), json!(video.fps));
    }
    Outcome::new()
        .set("frames", json!(video.frames.len()))
        .set("width", json!(video.width))
        .set("height", json!(video.height))
        .set("fps", json!(video.fps))
        .set("seconds_per_forward", json!(timings.secs_per_forward()))
        .blob("video", blob)
}

// ===================== execution (hot-DiT provider) =====================

/// The executable Wan model behind the manifest. Holds a **hot DiT** so a
/// long-lived process (`brain run` / the event server) loads and uploads the
/// transformer once per (variant, latent extent, device) and reuses it across
/// `ActionRequest`s. Weight paths come from the environment
/// (`BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}`).
pub struct WanProvider {
    hot: Arc<Mutex<Option<HotDit>>>,
}

impl WanProvider {
    pub fn new() -> WanProvider {
        WanProvider { hot: Arc::new(Mutex::new(None)) }
    }
}

impl Default for WanProvider {
    fn default() -> Self {
        WanProvider::new()
    }
}

impl Provider for WanProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest()
            .actions
            .iter()
            .any(|a| a.name == name)
            .then(|| Arc::new(WanAction { name: name.to_string(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

/// One Wan action, dispatched through the shared helpers above.
struct WanAction {
    name: String,
    hot: Arc<Mutex<Option<HotDit>>>,
}

impl Action for WanAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.name.as_str() {
            "t2v" => {
                // Params before the weights-env check: a request that could
                // never run must not read "you forgot to export BRAIN_WAN_DIT".
                let p = gen_params_from(inv)?;
                let paths = Paths::from_env()?;
                let mut guard = self.hot.lock().map_err(|_| "hot DiT lock poisoned")?;
                generate_on(&paths, &mut guard, inv, &p, progress)
            }
            other => Err(format!("wan '{other}': unknown action")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["t2v"]);
        let t2v = &m.actions[0];
        // Long-running and progress-bearing: a silent hour is a hang.
        assert!(t2v.streaming);
        // Only the prompt is required; everything else defaults to upstream's
        // own generate.py values, which is the whole "name a prompt and go"
        // contract.
        let required: Vec<&str> = t2v.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        let def = |name: &str| t2v.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no param {name}")).default.clone();
        assert_eq!(def("frames"), Some(json!(81)));
        assert_eq!(def("width"), Some(json!(832)));
        assert_eq!(def("height"), Some(json!(480)));
        assert_eq!(def("steps"), Some(json!(50)));
        assert_eq!(def("shift"), Some(json!(5.0)));
        assert_eq!(def("guidance"), Some(json!(5.0)));
        assert_eq!(def("seed"), Some(json!(0)));
        assert_eq!(def("fps"), Some(json!(16)));
        assert_eq!(def("solver"), Some(json!("unipc")));
        assert_eq!(def("variant"), Some(json!("t2v-1.3B")));
        // The two enums list exactly what the decoders accept.
        let ty = |name: &str| t2v.params.iter().find(|p| p.name == name).unwrap().ty.clone();
        assert!(matches!(ty("variant"), ParamType::Enum(v) if v == VARIANTS.map(String::from).to_vec()));
        assert!(matches!(ty("solver"), ParamType::Enum(v) if v == SOLVERS.map(String::from).to_vec()));
        // One output: the clip itself, as a video blob a remote client can
        // actually retrieve (a server-side path would be useless to it).
        assert_eq!(t2v.outputs.len(), 1);
        assert_eq!(t2v.outputs[0].name, "video");
        assert_eq!(t2v.outputs[0].media, Media::Video);
        assert!(t2v.outputs[0].required);
        assert!(t2v.inputs.is_empty(), "t2v takes no binary input");
        // The whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["model"], MODEL);
        assert_eq!(j["actions"].as_array().unwrap().len(), 1);
        assert_eq!(j["actions"][0]["outputs"][0]["media"], "video");
        assert_eq!(j["actions"][0]["streaming"], true);
        assert_eq!(j["actions"][0]["params"][0]["name"], "prompt");
        assert_eq!(j["actions"][0]["params"][0]["required"], true);
    }

    /// Every enum value the manifest advertises must decode, and the manifest
    /// defaults must survive `validate` -> `gen_params_from` unchanged. This
    /// is the join the two halves can drift at: a default in the spec that the
    /// decoder does not accept is a model that fails on its own advertised
    /// settings.
    #[test]
    fn the_advertised_defaults_and_enums_decode() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let inv = spec.validate(Invocation::new().set("prompt", json!("a cat"))).unwrap();
        let p = gen_params_from(&inv).unwrap();
        assert_eq!(p.variant, "t2v-1.3B");
        assert_eq!((p.opts.frames, p.opts.width, p.opts.height), (81, 832, 480));
        assert_eq!((p.opts.steps, p.opts.shift, p.opts.guidance), (50, 5.0, 5.0));
        assert_eq!(p.opts.solver, Solver::UniPc);
        assert_eq!(p.opts.negative_prompt, None, "absent means upstream's own sample_neg_prompt");
        for v in VARIANTS {
            assert!(config_from_name(v).is_ok(), "{v} is advertised but not constructible");
        }
        for s in SOLVERS {
            assert!(Solver::from_name(s).is_ok(), "{s} is advertised but not decodable");
        }
        assert!(config_from_name("i2v-14B").is_err(), "an unadvertised variant must not resolve");
    }

    /// The geometric rules are checked from the params alone, so a request
    /// that cannot possibly run never costs a 17.6 GB weight load to reject.
    #[test]
    fn impossible_geometry_is_rejected_before_any_weight_is_read() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let decode = |inv: Invocation| gen_params_from(&spec.validate(inv).unwrap());
        let base = || Invocation::new().set("prompt", json!("x"));
        // 80 frames is not 1 + 4k; rounding it would silently truncate the clip.
        let e = decode(base().set("frames", json!(80))).unwrap_err();
        assert!(e.contains("1 + 4k"), "{e}");
        // A width the (VAE stride x patch) grid cannot tile.
        let e = decode(base().set("width", json!(130))).unwrap_err();
        assert!(e.contains("multiple of"), "{e}");
        // An explicitly empty negative prompt is a real request for none.
        let p = decode(base().set("negative_prompt", json!(""))).unwrap();
        assert_eq!(p.opts.negative_prompt.as_deref(), Some(""));
        // A small clip that IS representable decodes cleanly.
        let p = decode(base().set("frames", json!(9)).set("width", json!(256)).set("height", json!(256)).set("steps", json!(2))).unwrap();
        assert_eq!((p.opts.frames, p.opts.width, p.opts.height, p.opts.steps), (9, 256, 256, 2));
    }

    /// An unarmed cancel token never fires, so a short/streaming caller that
    /// does not care pays nothing - and an armed one aborts. The loop-level
    /// behaviour is `pipeline`'s own test; this pins that the ACTION carries
    /// the token at all, which is what the serving contract requires.
    #[test]
    fn a_cancelled_invocation_is_refused_without_touching_the_weights() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let mut inv = Invocation::new().set("prompt", json!("x"));
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        let inv = spec.validate(inv).unwrap();
        assert!(inv.cancel.is_cancelled(), "validate must carry the token through to the action");
    }

    /// The video output is the wire format the shared codec reads back, with
    /// the frame rate carried in the blob's own metadata.
    #[test]
    fn video_outcome_round_trips_through_the_shared_clip_codec() {
        let video = crate::pipeline::Video { width: 2, height: 1, fps: 24, frames: vec![vec![255, 0, 0, 255, 0, 0], vec![0, 255, 0, 0, 255, 0]] };
        let out = video_outcome(&video, &crate::pipeline::Timings::default());
        assert_eq!(out.outputs["frames"], json!(2));
        assert_eq!(out.outputs["fps"], json!(24));
        let blob = &out.blobs["video"];
        assert_eq!(blob.media, Media::Video);
        assert_eq!(blob.meta["fps"], json!(24));
        let inv = Invocation::new().blob("video", blob.clone());
        let back = capability::blob::decode_video(&inv, "video").unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].1, 2);
        assert_eq!(back[0].0[0], 1.0, "red channel of the first pixel");
    }
}
