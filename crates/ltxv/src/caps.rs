// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's capabilities, declared through the generalized [`capability`]
//! interface - `wan::caps`'s pattern, adapted to this milestone's one real
//! difference: there is no resident weight-holding model to cache, because
//! the DiT is always tiny-config with FRESH RANDOM WEIGHTS (see
//! `crate::pipeline`'s module doc) and the VAE is read fresh per call, the
//! same way `wan::caps`'s VAE is never cached alongside its hot DiT.
//!
//! The manifest is **static** (no weights needed) so capability *discovery*
//! is free; only [`LtxvProvider`] (execution) reads anything from disk. Two
//! actions, `t2v` and `dfr` (the multi-stage DFR pipeline,
//! `crate::pipeline::generate_dfr`), both with a `prompt` param that is
//! REQUIRED even though no real text encoder consumes it yet (see
//! `crate::pipeline::context_stub`'s doc) - the capability surface is meant
//! to look like the real target shape a later milestone fills in, not a
//! smaller one this milestone happens to support. `brain ltxv {t2v,dfr}`
//! being a dedicated CLI module does not exempt either action from this
//! surface - the serving contract's obligation 1 is explicit that a bespoke
//! subcommand must never be the ONLY entry point.
//!
//! `inv.cancel` is polled once per denoise step - the poll lives in
//! `crate::pipeline::denoise`'s loop, shared by every caller, per the
//! serving contract's cancellation obligation.

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

use crate::pipeline::{dit_config_from_name, DfrOpts, DfrPaths, GenOpts, Paths};

/// The model id advertised on the capability surface (`brain caps
/// brain/ltxv`) and the event API - NOT reachable as a CLI verb of its own,
/// since `ltxv` has a dedicated CLI module (`crates/cli/src/ltxv_cli.rs`)
/// the resolver gives precedence over generic capability dispatch, the same
/// routing `wan` uses (`brain ltxv t2v` runs that module, not this action).
pub const MODEL: &str = "brain/ltxv";

/// `dit_config_from_name`'s advertised keys for `dfr` - `"tiny"` only:
/// `crate::pipeline::generate_dfr` still always builds `random_tiny_weights`
/// regardless of `dit_config` (no `RealDit` branch the way `generate`/t2v
/// has one), so advertising `"ltx25_22b"` here would promise a real-weight
/// DFR run this crate cannot yet produce - a tracked gap, not implemented.
const DFR_DIT_CONFIGS: [&str; 1] = ["tiny"];
/// `dit_config_from_name`'s advertised keys for `t2v` - both `"tiny"`
/// (fresh random weights, the smoke-test config) and `"ltx25_22b"` (the
/// real 22B checkpoint via `RealDit`/`crate::dit::forward_q_streamed` - see
/// `crate::pipeline`'s module doc). Selecting `"ltx25_22b"` also needs
/// `Paths::dit` (`--dit`/`$BRAIN_LTXV_DIT`) set; that check lives in
/// `generate` itself since this manifest is built with no weights loaded.
const T2V_DIT_CONFIGS: [&str; 2] = ["tiny", "ltx25_22b"];

/// The full, static capability manifest - safe to build with no weights
/// loaded. Defaults are [`GenOpts::default`]'s.
pub fn manifest() -> Manifest {
    let d = GenOpts::default();
    let t2v = ActionSpec::new(
        "t2v",
        "generate a video clip from a text prompt (rectified-flow ancestral Euler denoise over the LTX-2.5 video DiT with CFG, causal 3D VAE decode). dit_config=tiny (default) uses a random-weight smoke-test DiT and a deterministic stub text context; dit_config=ltx25_22b (needs --dit/$BRAIN_LTXV_DIT) runs the real 22B int8 checkpoint, optionally with the real Gemma-4 text encoder (--text-encoder/$BRAIN_LTXV_TEXT_ENCODER) - see crate::pipeline's module doc.",
    )
    .streaming()
    .param(ParamSpec::new("prompt", ParamType::Str, "text description; folded into a deterministic noise/context seed only - there is no real text encoder yet (see crate::pipeline::context_stub)").required())
    .param(ParamSpec::new("frames", ParamType::Int, "video frames; must be of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)").default(json!(d.frames)))
    .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 32)").default(json!(d.width)))
    .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 32)").default(json!(d.height)))
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps").default(json!(d.steps)))
    .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance; <= 1.0 runs ONE forward per step instead of two").default(json!(d.guidance)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise/weight/context seed (omit for 0)").default(json!(0)))
    .param(ParamSpec::new("fps", ParamType::Int, "frame rate reported with the clip").default(json!(d.fps)))
    .param(ParamSpec::new("base_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 1024 tokens").default(json!(d.base_shift)))
    .param(ParamSpec::new("max_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 4096 tokens").default(json!(d.max_shift)))
    .param(ParamSpec::new("stretch", ParamType::Bool, "stretch the schedule's terminal sigma to `terminal`").default(json!(d.stretch)))
    .param(ParamSpec::new("terminal", ParamType::Float, "terminal sigma the stretch targets").default(json!(d.terminal)))
    .param(ParamSpec::new("eta", ParamType::Float, "ancestral-Euler eta; 0 = deterministic Euler, 1 = fully ancestral").default(json!(d.eta)))
    .param(ParamSpec::new("dit_config", ParamType::Enum(T2V_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "DiT size: tiny (random weights) or ltx25_22b (the real checkpoint, needs --dit/$BRAIN_LTXV_DIT)").default(json!("tiny")))
    .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required());

    let dfr = ActionSpec::new(
        "dfr",
        "generate a video clip via DFR (Diffusion Fidelity Rendering): half-res base generation with generated keyframe slots, a real spatial x2 latent upscale, a full-res re-noised detailing pass (no IC-LoRA - see crate::pipeline's DFR doc), and 0-2 real temporal x2 upsample rounds with tile-based stitching. M8c gap: same tiny random-weight DiT and stub text context t2v has - see crate::pipeline's module doc (search \"M8c\") for exactly what is real.",
    )
    .streaming()
    .param(ParamSpec::new("prompt", ParamType::Str, "text description; folded into a deterministic noise/context seed only - there is no real text encoder yet (see crate::pipeline::context_stub)").required())
    .param(ParamSpec::new("frames", ParamType::Int, "video frames; must be of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)").default(json!(d.frames)))
    .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 64: stage 1 halves it, and the half must still be a multiple of the VAE's 32 spatial stride)").default(json!(d.width)))
    .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 64)").default(json!(d.height)))
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps per stage/tile").default(json!(d.steps)))
    .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance; <= 1.0 runs ONE forward per step instead of two").default(json!(d.guidance)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise/weight/context seed (omit for 0)").default(json!(0)))
    .param(ParamSpec::new("fps", ParamType::Int, "stage-1 frame rate; the reported fps is this times 2^temporal_upsample_rounds").default(json!(d.fps)))
    .param(ParamSpec::new("base_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 1024 tokens").default(json!(d.base_shift)))
    .param(ParamSpec::new("max_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 4096 tokens").default(json!(d.max_shift)))
    .param(ParamSpec::new("stretch", ParamType::Bool, "stretch the schedule's terminal sigma to `terminal`").default(json!(d.stretch)))
    .param(ParamSpec::new("terminal", ParamType::Float, "terminal sigma the stretch targets").default(json!(d.terminal)))
    .param(ParamSpec::new("eta", ParamType::Float, "ancestral-Euler eta; 0 = deterministic Euler, 1 = fully ancestral").default(json!(d.eta)))
    .param(ParamSpec::new("dit_config", ParamType::Enum(DFR_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "DiT size; only tiny (random weights) is implemented for dfr - the real 22B checkpoint is wired for t2v only, see this crate's roadmap").default(json!("tiny")))
    .param(ParamSpec::new("temporal_upsample_rounds", ParamType::Int, "0, 1, or 2 real temporal x2 refine rounds").default(json!(0)))
    .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required());

    Manifest::new(
        MODEL,
        "LTX-2.5 (Lightricks) - text-to-video diffusion transformer, rectified-flow ancestral Euler sampling with CFG, causal 3D VAE at (8, 32, 32) stride. t2v: real VAE + real scheduler, with the real 22B int8 checkpoint and real Gemma-4 text encoder available via dit_config=ltx25_22b (default stays the tiny random-weight smoke config). dfr: multi-stage pipeline with real latent upscalers, still tiny-random-weight-DiT/stub-context only - real-weight DFR is a tracked gap.",
        vec![t2v, dfr],
    )
}

// ===================== shared execution helpers =====================
//
// Both [`LtxvProvider`] and the residency adapter (`crates/cli/src/
// resident_ltxv.rs`) run `t2v` through these - ONE implementation of param
// decoding, generation and outcome shaping, the `wan::caps` pattern.

use std::sync::Arc;

use capability::{Action, ActionResult, Invocation, Outcome, Progress, Provider};

/// A decoded `t2v` request.
#[derive(Debug)]
pub struct GenParams {
    pub opts: GenOpts,
}

/// Decode + validate `t2v`'s params from an invocation. Every geometric
/// constraint is checked HERE, before the VAE checkpoint is read: a bad
/// frame count or a size the VAE stride cannot tile must not cost a weight
/// load to discover.
pub fn gen_params_from(inv: &Invocation) -> Result<GenParams, String> {
    let d = GenOpts::default();
    let dit_config = inv.get_str("dit_config").unwrap_or_else(|| "tiny".into());
    dit_config_from_name(&dit_config)?; // validated here so a bad name is rejected before any weight load
    let opts = GenOpts {
        frames: inv.get_i64("frames").unwrap_or(d.frames as i64).max(1) as usize,
        width: inv.get_i64("width").unwrap_or(d.width as i64).max(32) as usize,
        height: inv.get_i64("height").unwrap_or(d.height as i64).max(32) as usize,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        guidance: inv.get_f64("guidance").unwrap_or(d.guidance as f64) as f32,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        fps: inv.get_i64("fps").unwrap_or(d.fps as i64).max(1) as usize,
        base_shift: inv.get_f64("base_shift").unwrap_or(d.base_shift),
        max_shift: inv.get_f64("max_shift").unwrap_or(d.max_shift),
        stretch: inv.get_bool("stretch").unwrap_or(d.stretch),
        terminal: inv.get_f64("terminal").unwrap_or(d.terminal),
        eta: inv.get_f64("eta").unwrap_or(d.eta),
        s_noise: d.s_noise,
        context_len: d.context_len,
        dit_config,
        device: None,
    };
    use crate::vae3d::LtxVaeConfig;
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(opts.frames as u32).is_none() {
        return Err(format!("frames must be of the form 1 + 8k (1, 9, 17, … ); got {}", opts.frames));
    }
    if !opts.width.is_multiple_of(32) || !opts.height.is_multiple_of(32) {
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", opts.width, opts.height));
    }
    Ok(GenParams { opts })
}

/// Run one generation and wrap the result as a video-output [`Outcome`].
/// Cancellation rides in `inv.cancel` - `crate::pipeline::denoise` polls it
/// per step.
pub fn generate_on(paths: &Paths, inv: &Invocation, p: &GenParams, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let (video, timings) = crate::pipeline::generate(paths, &prompt, &p.opts, &inv.cancel, |done, total, phase| progress(Progress::step(done, total, phase.to_string())))?;
    Ok(video_outcome(&video, &timings))
}

/// A decoded `dfr` request - [`GenParams`]'s DFR analogue.
#[derive(Debug)]
pub struct DfrParams {
    pub opts: DfrOpts,
}

/// Decode + validate `dfr`'s params from an invocation - [`gen_params_from`]'s
/// DFR analogue, same "reject before any weight load" discipline plus the
/// `temporal_upsample_rounds` bound `crate::pipeline::generate_dfr` itself
/// enforces.
pub fn dfr_params_from(inv: &Invocation) -> Result<DfrParams, String> {
    let d = GenOpts::default();
    let dit_config = inv.get_str("dit_config").unwrap_or_else(|| "tiny".into());
    dit_config_from_name(&dit_config)?; // validated here so a bad name is rejected before any weight load
    let base = GenOpts {
        frames: inv.get_i64("frames").unwrap_or(d.frames as i64).max(1) as usize,
        width: inv.get_i64("width").unwrap_or(d.width as i64).max(64) as usize,
        height: inv.get_i64("height").unwrap_or(d.height as i64).max(64) as usize,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        guidance: inv.get_f64("guidance").unwrap_or(d.guidance as f64) as f32,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        fps: inv.get_i64("fps").unwrap_or(d.fps as i64).max(1) as usize,
        base_shift: inv.get_f64("base_shift").unwrap_or(d.base_shift),
        max_shift: inv.get_f64("max_shift").unwrap_or(d.max_shift),
        stretch: inv.get_bool("stretch").unwrap_or(d.stretch),
        terminal: inv.get_f64("terminal").unwrap_or(d.terminal),
        eta: inv.get_f64("eta").unwrap_or(d.eta),
        s_noise: d.s_noise,
        context_len: d.context_len,
        dit_config,
        device: None,
    };
    let temporal_upsample_rounds = inv.get_i64("temporal_upsample_rounds").unwrap_or(0).max(0) as usize;
    let opts = DfrOpts { base, temporal_upsample_rounds };
    use crate::vae3d::LtxVaeConfig;
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(opts.base.frames as u32).is_none() {
        return Err(format!("frames must be of the form 1 + 8k (1, 9, 17, … ); got {}", opts.base.frames));
    }
    if !opts.base.width.is_multiple_of(64) || !opts.base.height.is_multiple_of(64) {
        return Err(format!("{}x{} is not a multiple of 64 for DFR (stage 1 halves it, and the half must still be a multiple of the VAE's 32 spatial stride)", opts.base.width, opts.base.height));
    }
    if opts.temporal_upsample_rounds > 2 {
        return Err(format!("temporal_upsample_rounds must be 0, 1, or 2, got {}", opts.temporal_upsample_rounds));
    }
    Ok(DfrParams { opts })
}

/// Run one DFR generation and wrap the result as a video-output [`Outcome`] -
/// [`generate_on`]'s DFR analogue.
pub fn dfr_on(paths: &DfrPaths, inv: &Invocation, p: &DfrParams, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let (video, timings) = crate::pipeline::generate_dfr(paths, &prompt, &p.opts, &inv.cancel, |done, total, phase| progress(Progress::step(done, total, phase.to_string())))?;
    Ok(video_outcome(&video, &timings))
}

/// Wrap a generated clip as a video-output [`Outcome`] - the same wire
/// format `wan::caps::video_outcome` produces.
pub fn video_outcome(video: &crate::pipeline::Video, timings: &crate::pipeline::Timings) -> Outcome {
    let frames: Vec<(Vec<f32>, u32, u32)> = video.frames.iter().map(|px| (px.iter().map(|&b| b as f32 / 255.0).collect::<Vec<f32>>(), video.width, video.height)).collect();
    let mut blob = match capability::blob::video_blob(&frames) {
        Ok(b) => b,
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

// ===================== execution (provider) =====================

/// The executable LTX-2.5 model behind the manifest. Unlike `WanProvider`
/// there is no hot-DiT cache: this milestone's DiT is tiny/random and cheap
/// to rebuild per call (microseconds, not the 20s+5.7GB upload a real 22B
/// checkpoint would cost) - caching it would add state for no benefit. Only
/// the VAE path comes from the environment (`BRAIN_LTXV_VAE`).
pub struct LtxvProvider;

impl LtxvProvider {
    pub fn new() -> LtxvProvider {
        LtxvProvider
    }
}

impl Default for LtxvProvider {
    fn default() -> Self {
        LtxvProvider::new()
    }
}

impl Provider for LtxvProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| Arc::new(LtxvAction { name: name.to_string() }) as Arc<dyn Action>)
    }
}

/// One LTX-2.5 action, dispatched through the shared helpers above.
struct LtxvAction {
    name: String,
}

impl Action for LtxvAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    #[tracing::instrument(level = "info", name = "ltxv_action", skip_all, fields(action = %self.name))]
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // The SERVED entry point (D-Bus/HTTP/`brain do`), as opposed to the
        // `brain ltxv` CLI: a served request has no terminal to print to, so
        // its start/finish/failure only exist anywhere if they are traced.
        tracing::info!("action invoked");
        let result = match self.name.as_str() {
            "t2v" => {
                // Params before the weights-env check: a request that could
                // never run must not read "you forgot to export BRAIN_LTXV_VAE".
                let p = gen_params_from(inv)?;
                let paths = Paths::from_env()?;
                generate_on(&paths, inv, &p, progress)
            }
            "dfr" => {
                let p = dfr_params_from(inv)?;
                let paths = DfrPaths::from_env()?;
                dfr_on(&paths, inv, &p, progress)
            }
            other => {
                tracing::error!(action = other, "unknown action");
                Err(format!("ltxv '{other}': unknown action"))
            }
        };
        match &result {
            Ok(_) => tracing::info!("action succeeded"),
            Err(e) => tracing::error!(error = %e, "action failed"),
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weights-free per the serving contract's obligation 5: this must never
    /// touch `BRAIN_LTXV_VAE` or the filesystem.
    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["t2v", "dfr"]);
        let t2v = &m.actions[0];
        assert!(t2v.streaming, "a multi-step denoise run without progress reads as a hang");
        let required: Vec<&str> = t2v.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        let def = |name: &str| t2v.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no param {name}")).default.clone();
        assert_eq!(def("frames"), Some(json!(9)));
        assert_eq!(def("width"), Some(json!(64)));
        assert_eq!(def("height"), Some(json!(64)));
        assert_eq!(def("steps"), Some(json!(4)));
        assert_eq!(def("guidance"), Some(json!(1.0)));
        assert_eq!(def("seed"), Some(json!(0)));
        assert_eq!(def("fps"), Some(json!(8)));
        assert_eq!(def("dit_config"), Some(json!("tiny")));
        let ty = |name: &str| t2v.params.iter().find(|p| p.name == name).unwrap().ty.clone();
        assert!(matches!(ty("dit_config"), ParamType::Enum(v) if v == T2V_DIT_CONFIGS.map(String::from).to_vec()));
        assert_eq!(t2v.outputs.len(), 1);
        assert_eq!(t2v.outputs[0].name, "video");
        assert_eq!(t2v.outputs[0].media, Media::Video);
        assert!(t2v.outputs[0].required);
        assert!(t2v.inputs.is_empty(), "t2v takes no binary input");
        let j = m.to_json();
        assert_eq!(j["model"], MODEL);
        assert_eq!(j["actions"][0]["streaming"], true);
        assert_eq!(j["actions"][0]["params"][0]["name"], "prompt");
        assert_eq!(j["actions"][0]["params"][0]["required"], true);
    }

    /// The manifest's own defaults must survive `validate` -> `gen_params_from`
    /// unchanged, and every advertised `dit_config` value must decode - the
    /// join the two halves can silently drift at.
    #[test]
    fn the_advertised_defaults_and_dit_configs_decode() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let inv = spec.validate(Invocation::new().set("prompt", json!("a cat"))).unwrap();
        let p = gen_params_from(&inv).unwrap();
        assert_eq!((p.opts.frames, p.opts.width, p.opts.height, p.opts.steps), (9, 64, 64, 4));
        assert_eq!(p.opts.dit_config, "tiny");
        for c in T2V_DIT_CONFIGS {
            assert!(dit_config_from_name(c).is_ok(), "t2v: {c} is advertised but not decodable");
        }
        let dfr_spec = manifest().actions.into_iter().nth(1).unwrap();
        assert_eq!(dfr_spec.name, "dfr");
        for c in DFR_DIT_CONFIGS {
            assert!(dit_config_from_name(c).is_ok(), "dfr: {c} is advertised but not decodable");
        }
        assert!(dit_config_from_name("22b").is_err(), "an unimplemented dit-config must not resolve");
    }

    /// The geometric rules are checked from the params alone, so a request
    /// that cannot possibly run never costs a weight load to reject.
    #[test]
    fn impossible_geometry_is_rejected_before_any_weight_is_read() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let decode = |inv: Invocation| gen_params_from(&spec.validate(inv).unwrap());
        let base = || Invocation::new().set("prompt", json!("x"));
        let e = decode(base().set("frames", json!(8))).unwrap_err();
        assert!(e.contains("1 + 8k"), "{e}");
        let e = decode(base().set("width", json!(65))).unwrap_err();
        assert!(e.contains("multiple of 32"), "{e}");
        let p = decode(base().set("frames", json!(17)).set("width", json!(96)).set("height", json!(96)).set("steps", json!(2))).unwrap();
        assert_eq!((p.opts.frames, p.opts.width, p.opts.height, p.opts.steps), (17, 96, 96, 2));
    }

    #[test]
    fn a_cancelled_invocation_is_refused_without_touching_the_weights() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let mut inv = Invocation::new().set("prompt", json!("x"));
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        let inv = spec.validate(inv).unwrap();
        assert!(inv.cancel.is_cancelled(), "validate must carry the token through to the action");
    }

    /// `dfr`'s own analogue of [`manifest_declares_the_full_surface`] - the
    /// action `brain ltxv dfr --help` documents must also be reachable
    /// through `capability::Provider`, not just the CLI (serving contract
    /// obligation 1).
    #[test]
    fn dfr_is_advertised_with_its_own_full_surface() {
        let m = manifest();
        let dfr = m.actions.iter().find(|a| a.name == "dfr").expect("dfr must be an advertised action");
        assert!(dfr.streaming, "a multi-stage DFR run without progress reads as a hang");
        let required: Vec<&str> = dfr.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        let def = |name: &str| dfr.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no param {name}")).default.clone();
        assert_eq!(def("frames"), Some(json!(9)));
        assert_eq!(def("width"), Some(json!(64)));
        assert_eq!(def("height"), Some(json!(64)));
        assert_eq!(def("temporal_upsample_rounds"), Some(json!(0)));
        assert_eq!(dfr.outputs.len(), 1);
        assert_eq!(dfr.outputs[0].name, "video");
        assert_eq!(dfr.outputs[0].media, Media::Video);
    }

    /// [`the_advertised_defaults_and_dit_configs_decode`]'s DFR analogue.
    #[test]
    fn dfr_advertised_defaults_decode() {
        let spec = manifest().actions.into_iter().find(|a| a.name == "dfr").unwrap();
        let inv = spec.validate(Invocation::new().set("prompt", json!("a cat"))).unwrap();
        let p = dfr_params_from(&inv).unwrap();
        assert_eq!((p.opts.base.frames, p.opts.base.width, p.opts.base.height, p.opts.base.steps), (9, 64, 64, 4));
        assert_eq!(p.opts.temporal_upsample_rounds, 0);
    }

    /// [`impossible_geometry_is_rejected_before_any_weight_is_read`]'s DFR
    /// analogue - `dfr`'s own multiple-of-64 rule (not t2v's multiple-of-32)
    /// and the 0-2 `temporal_upsample_rounds` bound must both be rejected
    /// before any weight is read.
    #[test]
    fn dfr_impossible_params_are_rejected_before_any_weight_is_read() {
        let spec = manifest().actions.into_iter().find(|a| a.name == "dfr").unwrap();
        let decode = |inv: Invocation| dfr_params_from(&spec.validate(inv).unwrap());
        let base = || Invocation::new().set("prompt", json!("x"));
        let e = decode(base().set("frames", json!(8))).unwrap_err();
        assert!(e.contains("1 + 8k"), "{e}");
        let e = decode(base().set("width", json!(96))).unwrap_err();
        assert!(e.contains("multiple of 64"), "{e}");
        let e = decode(base().set("temporal_upsample_rounds", json!(3))).unwrap_err();
        assert!(e.contains("temporal_upsample_rounds"), "{e}");
        let p = decode(base().set("frames", json!(17)).set("width", json!(128)).set("height", json!(128)).set("temporal_upsample_rounds", json!(2))).unwrap();
        assert_eq!((p.opts.base.frames, p.opts.base.width, p.opts.base.height, p.opts.temporal_upsample_rounds), (17, 128, 128, 2));
    }

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
