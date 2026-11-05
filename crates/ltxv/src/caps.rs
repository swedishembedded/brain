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

use crate::pipeline::{dit_config_from_name, DfrOpts, DfrPaths, GenOpts, Named, Paths, Sampler};

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
    .params(guidance_params())
    .param(ParamSpec::new("start_frame", ParamType::Str, "optional path to a PNG/JPEG still: encoded through the real video VAE and used as frame-0 conditioning while the rest of the clip denoises around it (VideoConditionByLatentIndex on its own, VideoConditionByKeyframeIndex when end_frame is also given - see crate::pipeline::conditioned_latent). Needs dit_config=ltx25_22b (the real VAE). Runs under either sampler: eta>0 follows the reference's ancestral loop (LTX-2.5 distilled's own stage-1 sampler), eta=0 the deterministic one.").host_resolved())
    .param(ParamSpec::new("end_frame", ParamType::Str, "optional path to a PNG/JPEG still: a clean image-guidance latent appended at the clip's LAST pixel frame's timestamp, not a replacement of it - the generated token there still denoises freely, only strongly guided toward the still via attention (start_frame given alone is the one case that IS a literal replacement; combined with end_frame/mid_frame it switches to this same appended-guide mechanism too). Same image as start_frame for a clip that loops seamlessly (the generated content in between connects the still to itself); a different image for a clip that morphs from one still to another. Same eta/dit_config requirements as start_frame.").host_resolved())
    .param(ParamSpec::new("mid_frame", ParamType::Str, "optional path to a PNG/JPEG still: the same appended image-guidance latent end_frame uses, anchored at ONE INTERIOR instant of the clip, so a single generation can be guided at its start, its middle and its end at once. Same eta/dit_config requirements as start_frame.").host_resolved())
    .param(ParamSpec::new("mid_frame_at", ParamType::Int, "which pixel frame mid_frame anchors, strictly between 0 and frames-1. Unset takes the reference's own position for a single interior keyframe (evenly_spaced_keyframe_positions), (frames-1)/2. Only read when mid_frame is set."))
    .param(ParamSpec::new("conditioning_strength", ParamType::Float, "ImageConditioningInput.strength for every given still: 1.0 pins the conditioned tokens to the encoded image exactly, lower values (the reference's own CLI example is 0.8) leave the model room to move. Only read when one of start_frame/mid_frame/end_frame is set.").default(json!(d.conditioning_strength)))
    .param(ParamSpec::new("frames", ParamType::Int, "video frames; must be of the form 1 + 8k (the causal VAE gives the first frame its own latent frame). Unset, the duration head predicts the length from the prompt (1-20 s) when the real checkpoint, text encoder and duration head are configured; otherwise 121 with the real checkpoint, 9 for the tiny smoke test"))
    .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 32); unset, 768 with the real checkpoint, 64 for the smoke test"))
    .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 32); unset, 512 with the real checkpoint, 64 for the smoke test"))
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps").default(json!(d.steps)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise/weight/context seed (omit for random)"))
    .param(ParamSpec::new("fps", ParamType::Int, "frame rate; unset, 24 with the real checkpoint, 8 for the smoke test"))
    .param(ParamSpec::new("base_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 1024 tokens").default(json!(d.base_shift)))
    .param(ParamSpec::new("max_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 4096 tokens").default(json!(d.max_shift)))
    .param(ParamSpec::new("stretch", ParamType::Bool, "stretch the schedule's terminal sigma to `terminal`").default(json!(d.stretch)))
    .param(ParamSpec::new("terminal", ParamType::Float, "terminal sigma the stretch targets").default(json!(d.terminal)))
    .params(sampler_params())
    .param(ParamSpec::new("dit_config", ParamType::Enum(T2V_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "DiT size: ltx25_22b (the real checkpoint - the default when it is configured) or tiny (random weights, a smoke test - the default without it)"))
    .param(ParamSpec::new(
        "audio",
        ParamType::Bool,
        "also generate the clip's SOUND. LTX-2.5 is natively audio-visual: one transformer denoises a video-latent and an audio-latent stream together, coupled every block by cross-attention, and this runs both instead of only the video half. Needs dit_config=ltx25_22b, a real text encoder ($BRAIN_LTXV_TEXT_ENCODER - the audio stream has its OWN text projection the stub cannot stand in for) and $BRAIN_LTXV_AUDIO_VAE; all three are checked before any weight is read. Off by default because of cost: it runs on the same streamed/quantized/device-resident path the video-only stream does, so a joint step costs modestly more than a video-only one at the same shape rather than several times more, but it is still more. Any legal frame count: past one denoising window the audio latent crosses each seam beside the video one, which constrains where a seam may fall (see crate::audio's module doc) and so plans a different window split from the same request without sound.",
    )
    .default(json!(d.audio)))
    .param(ParamSpec::new("audio_source", ParamType::Str, "server-side path to a WAV soundtrack to generate the video FOR (audio-to-video): encoded through the audio VAE and held as the clip's frozen audio stream while the video denoises against it, then returned as the clip's audio. Needs audio=true; past one denoising window each window holds its own slice of it.").host_resolved())
    .param(ParamSpec::new("adapter", ParamType::Str, "server-side path to a LoRA adapter to apply: the official LTX trainer's / a published IC-LoRA's ComfyUI .safetensors, or one brain trained. Runs beside the quantized weights, never folded into them.").host_resolved())
    .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier (ComfyUI strength_model); 1.0 = the reference default. Only read when adapter is set.").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
    .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required())
    // Not `.required()`: a silent generation is the default, and an output a
    // caller can legitimately not receive must not be declared as one that
    // always arrives.
    .output(BlobSpec::new(
        "audio",
        Media::Audio,
        "the clip's own sound track when audio=true: a complete 48 kHz stereo WAV, generated by the SAME forwards as the frames and covering the same time window. Absent when audio=false.",
    ));

    // What `/videos` maps onto t2v: LTX-2.5's 1 + 8k frame grid at the
    // requested rate, sides on the VAE's 32-pixel stride, the 1-20 s the
    // duration head predicts within, sound on request.
    let t2v = t2v.video(video_contract());

    let dfr = ActionSpec::new(
        "dfr",
        "generate a video clip via DFR (Diffusion Fidelity Rendering): half-res base generation with generated keyframe slots, a real spatial x2 latent upscale, a full-res re-noised detailing pass (no IC-LoRA - see crate::pipeline's DFR doc), and 0-2 real temporal x2 upsample rounds with tile-based stitching. M8c gap: same tiny random-weight DiT and stub text context t2v has - see crate::pipeline's module doc (search \"M8c\") for exactly what is real.",
    )
    .streaming()
    .param(ParamSpec::new("prompt", ParamType::Str, "text description; folded into a deterministic noise/context seed only - there is no real text encoder yet (see crate::pipeline::context_stub)").required())
    .params(guidance_params())
    .param(ParamSpec::new("frames", ParamType::Int, "video frames; must be of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)").default(json!(d.frames)))
    .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 64: stage 1 halves it, and the half must still be a multiple of the VAE's 32 spatial stride)").default(json!(d.width)))
    .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 64)").default(json!(d.height)))
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps per stage/tile").default(json!(d.steps)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise/weight/context seed (omit for random)"))
    .param(ParamSpec::new("fps", ParamType::Int, "stage-1 frame rate; the reported fps is this times 2^temporal_upsample_rounds").default(json!(d.fps)))
    .param(ParamSpec::new("base_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 1024 tokens").default(json!(d.base_shift)))
    .param(ParamSpec::new("max_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 4096 tokens").default(json!(d.max_shift)))
    .param(ParamSpec::new("stretch", ParamType::Bool, "stretch the schedule's terminal sigma to `terminal`").default(json!(d.stretch)))
    .param(ParamSpec::new("terminal", ParamType::Float, "terminal sigma the stretch targets").default(json!(d.terminal)))
    .params(sampler_params())
    .param(ParamSpec::new("dit_config", ParamType::Enum(DFR_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "DiT size; only tiny (random weights) is implemented for dfr - the real 22B checkpoint is wired for t2v only, see this crate's roadmap").default(json!("tiny")))
    .param(ParamSpec::new("temporal_upsample_rounds", ParamType::Int, "0, 1, or 2 real temporal x2 refine rounds").default(json!(0)))
    .param(ParamSpec::new("adapter", ParamType::Str, "server-side path to a LoRA adapter to apply: the official LTX trainer's / a published IC-LoRA's ComfyUI .safetensors, or one brain trained. Runs beside the quantized weights, never folded into them.").host_resolved())
    .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier (ComfyUI strength_model); 1.0 = the reference default. Only read when adapter is set.").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
    .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required());

    let a = crate::pipeline::AudioOpts::default();
    let t2a = ActionSpec::new(
        "t2a",
        "generate a sound clip from a text prompt (ltx_pipelines.t2a_one_stage): the LTX-2.5 checkpoint's audio stream denoised alone - the video half is never read - then the audio VAE and vocoder with bandwidth extension. Needs the real checkpoint ($BRAIN_LTXV_DIT), text encoder ($BRAIN_LTXV_TEXT_ENCODER) and audio VAE ($BRAIN_LTXV_AUDIO_VAE).",
    )
    .streaming()
    .param(ParamSpec::new("prompt", ParamType::Str, "what the clip should sound like").required())
    .param(ParamSpec::new("seconds", ParamType::Float, "clip length in seconds; unset, the duration head predicts it from the prompt (1-20 s), or 5 s without one").min(0.04))
    .params(guidance_params())
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps (a non-distilled checkpoint's; the distilled one has a fixed schedule)").default(json!(a.base.steps)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise seed (omit for random)"))
    .params(sampler_params())
    .param(ParamSpec::new("adapter", ParamType::Str, "server-side path to a LoRA adapter to apply beside the quantized weights").host_resolved())
    .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier; 1.0 = the reference default. Only read when adapter is set.").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
    .output(BlobSpec::new("audio", Media::Audio, "the generated clip: a complete 48 kHz stereo WAV").required());

    let clip_in = |spec: ActionSpec| -> ActionSpec {
        spec.streaming()
            .input(BlobSpec::new("video", Media::Video, "the source clip, in the standard video wire convention (capability::blob::decode_video): every frame one (w,h), each a multiple of 32").required())
            .input(BlobSpec::new("soundtrack", Media::Audio, "the clip's own soundtrack, a WAV container; needed when audio=true"))
            .param(ParamSpec::new("fps", ParamType::Int, "the clip's frame rate; unset, the video blob's meta.fps, else 24"))
            .param(ParamSpec::new("prompt", ParamType::Str, "what the generated part shows").required())
            .params(guidance_params())
            .params(sampler_params())
            .param(ParamSpec::new("seed", ParamType::Int, "noise seed (omit for random)"))
            .param(ParamSpec::new("steps", ParamType::Int, "denoise steps (non-distilled checkpoints)").default(json!(d.steps)))
            .param(ParamSpec::new("dit_config", ParamType::Enum(T2V_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "ltx25_22b (default when the checkpoint is configured) or tiny (random-weight smoke test)"))
            .param(ParamSpec::new("adapter", ParamType::Str, "server-side path to a LoRA adapter to apply beside the quantized weights").host_resolved())
            .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier; 1.0 = the reference default").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
            .output(BlobSpec::new("video", Media::Video, "the resulting clip: N interleaved-HWC f32 RGB frames, meta {frames,w,h,c,fps}").required())
            .output(BlobSpec::new("audio", Media::Audio, "the resulting soundtrack when audio=true: a complete WAV"))
    };
    // An edit takes its length and size from the clip; the contract says
    // which blob carries it and what else maps.
    let edit = |spec: ActionSpec, sound: bool| -> ActionSpec {
        let contract = video_contract().clip_input("video");
        spec.video(if sound { contract } else { contract.audio(capability::video::Audio::None) })
    };
    let v2v = clip_in(ActionSpec::new(
        "v2v",
        "retake part of a clip (ltx_pipelines.retake): a time window is regenerated against the prompt while the rest is held, the picture, the sound or both",
    ))
    .param(ParamSpec::new("start", ParamType::Float, "the window's start, seconds").required().min(0.0))
    .param(ParamSpec::new("end", ParamType::Float, "the window's end, seconds").required().min(0.0))
    .param(ParamSpec::new("audio", ParamType::Bool, "also run the clip's sound through the same forwards").default(json!(false)))
    .param(ParamSpec::new("hold_video", ParamType::Bool, "regenerate no frame (with audio: video-to-audio)").default(json!(false)))
    .param(ParamSpec::new("hold_audio", ParamType::Bool, "keep the soundtrack everywhere").default(json!(false)));
    let extend = clip_in(ActionSpec::new(
        "extend",
        "continue a clip (video_extend_lora's forward extension): its last frames held as a clean prefix while new frames after them denoise; returns the clip followed by the new frames",
    ))
    .param(ParamSpec::new("frames", ParamType::Int, "frames to add, rounded up to a multiple of 8").default(json!(crate::pipeline::ExtendOpts::default().frames)).min(1.0))
    .param(ParamSpec::new("context_frames", ParamType::Int, "how much of the clip's end the continuation sees").default(json!(crate::longform::CONTEXT_FRAMES)).min(1.0))
    .param(ParamSpec::new("audio", ParamType::Bool, "extend the soundtrack too (needs the soundtrack input)").default(json!(false)));
    let upscale = clip_in(ActionSpec::new(
        "upscale",
        "re-render a finished clip at twice its resolution: the real latent x2 upscaler, then the distilled refinement pass against the prompt",
    ))
    .param(ParamSpec::new("refine_steps", ParamType::Int, "refinement steps, a suffix of the distilled stage-2 schedule").default(json!(crate::pipeline::UpscaleOpts::default().refine_steps)).min(1.0));

    let (v2v, extend, upscale) = (edit(v2v, true), edit(extend, true), edit(upscale, false));

    Manifest::new(
        MODEL,
        "LTX-2.5 (Lightricks) - text-to-video diffusion transformer, rectified-flow ancestral Euler sampling with CFG, causal 3D VAE at (8, 32, 32) stride. t2v: real VAE + real scheduler, with the real 22B int8 checkpoint and real Gemma-4 text encoder available via dit_config=ltx25_22b (default stays the tiny random-weight smoke config). dfr: multi-stage pipeline with real latent upscalers, still tiny-random-weight-DiT/stub-context only - real-weight DFR is a tracked gap.",
        vec![t2v, dfr, t2a, v2v, extend, upscale],
    )
}

/// The video contract of LTX-2.5's generation and edit actions - see
/// `capability::video`.
fn video_contract() -> capability::video::VideoContract {
    use capability::video::{Audio, Fps, VideoContract};
    VideoContract::new("frames", (1, 8), Fps::Param { name: "fps".into(), default: 24 }, ("width", "height"), 32, 1920 * 1088, (1.0, 20.0))
        .audio(Audio::Param { name: "audio".into(), default: GenOpts::default().audio })
        .seed("seed")
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
    /// Which settings the request named. The rest are resolved against the
    /// configured checkpoints when it runs ([`crate::pipeline::resolve_defaults`]),
    /// and an unnamed length is the duration head's to predict when it can
    /// run ([`crate::pipeline::can_predict_frames`]).
    pub named: Named,
}

/// Which of the settings [`crate::pipeline::resolve_defaults`] fills an
/// invocation names itself.
pub fn named_in(inv: &Invocation) -> Named {
    let has = |k: &str| inv.params.get(k).is_some_and(|v| !v.is_null());
    Named { dit_config: has("dit_config"), frames: has("frames"), size: has("width") || has("height"), fps: has("fps") }
}

/// The generation `p` describes, with every setting it left unnamed
/// resolved against `paths` - and an unnamed length predicted by the
/// duration head when it can run.
pub fn resolved_opts(paths: &Paths, prompt: &str, p: &GenParams) -> Result<GenOpts, String> {
    let mut o = p.opts.clone();
    crate::pipeline::resolve_defaults(&mut o, paths, p.named);
    if !p.named.frames && crate::pipeline::can_predict_frames(paths, &o) {
        o.frames = crate::pipeline::predict_frames(paths, prompt, &o, crate::pipeline::prediction_streams(&o))?;
    }
    Ok(o)
}

/// The invocation's LoRA adapter (`adapter` at `lora_scale`), if it names one.
fn adapters_from(inv: &Invocation) -> Vec<model::adapter::AdapterSpec> {
    let scale = inv.get_f64("lora_scale").unwrap_or(1.0) as f32;
    inv.get_str("adapter").map(|path| model::adapter::AdapterSpec { path, scale }).into_iter().collect()
}

/// Decode + validate `t2v`'s params from an invocation. Every geometric
/// constraint is checked HERE, before the VAE checkpoint is read: a bad
/// frame count or a size the VAE stride cannot tile must not cost a weight
/// load to discover.
pub fn gen_params_from(inv: &Invocation) -> Result<GenParams, String> {
    let d = GenOpts::default();
    let dit_config = inv.get_str("dit_config").unwrap_or_else(|| "tiny".into());
    dit_config_from_name(&dit_config)?; // validated here so a bad name is rejected before any weight load
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let (sampler, eta) = sampler_from(inv)?;
    let opts = GenOpts {
        frames: inv.get_i64("frames").unwrap_or(d.frames as i64).max(1) as usize,
        width: inv.get_i64("width").unwrap_or(d.width as i64).max(32) as usize,
        height: inv.get_i64("height").unwrap_or(d.height as i64).max(32) as usize,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        guidance: guidance_from(inv)?,
        seed,
        fps: inv.get_i64("fps").unwrap_or(d.fps as i64).max(1) as usize,
        base_shift: inv.get_f64("base_shift").unwrap_or(d.base_shift),
        max_shift: inv.get_f64("max_shift").unwrap_or(d.max_shift),
        stretch: inv.get_bool("stretch").unwrap_or(d.stretch),
        terminal: inv.get_f64("terminal").unwrap_or(d.terminal),
        sampler,
        eta,
        s_noise: d.s_noise,
        context_len: d.context_len,
        dit_config,
        audio: inv.get_bool("audio").unwrap_or(d.audio),
        audio_source: inv.get_str("audio_source").filter(|p| !p.is_empty()).map(|p| crate::audio::read_wav(&p)).transpose()?,
        device: None,
        start_frame: inv.get_str("start_frame"),
        end_frame: inv.get_str("end_frame"),
        mid_frame: inv.get_str("mid_frame"),
        mid_frame_at: inv.get_i64("mid_frame_at").filter(|&n| n >= 0).map(|n| n as usize),
        conditioning_strength: inv.get_f64("conditioning_strength").unwrap_or(d.conditioning_strength as f64).clamp(0.0, 1.0) as f32,
        // Not an invocation parameter: placement is the SERVER's decision,
        // not the caller's. A resident instance overrides it (see
        // `crates/catalog/src/resident_ltxv.rs`, which hands each concurrently
        // admitted generation its own card); a one-shot CLI run keeps the
        // default, which uses both cards for one generation's CFG pair.
        future_anchors: Vec::new(),
        devices: d.devices,
        adapters: adapters_from(inv),
        reference: None,
        refine_in_context: false,
    };
    use crate::vae3d::LtxVaeConfig;
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(opts.frames as u32).is_none() {
        return Err(format!("frames must be of the form 1 + 8k (1, 9, 17, … ); got {}", opts.frames));
    }
    if !opts.width.is_multiple_of(32) || !opts.height.is_multiple_of(32) {
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", opts.width, opts.height));
    }
    if opts.mid_frame.is_some() {
        crate::pipeline::mid_anchor_frame(opts.frames, opts.mid_frame_at)?;
    }
    Ok(GenParams { opts, named: named_in(inv) })
}

/// Run one generation and wrap the result as a video-output [`Outcome`].
/// Cancellation rides in `inv.cancel` - `crate::pipeline::denoise` polls it
/// per step.
pub fn generate_on(paths: &Paths, inv: &Invocation, p: &GenParams, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    // Through `generate_long`, which IS `generate` for every request that
    // fits one denoising window and rolls a latent context across several
    // when it does not. No new action parameter: the context size is the
    // reference's own default (`crate::longform::CONTEXT_LATENT_FRAMES`) and
    // the per-window ceiling comes from the environment, so a long `frames`
    // simply works instead of failing at buffer creation.
    let base = resolved_opts(paths, &prompt, p)?;
    let seed = base.seed;
    let o = crate::pipeline::LongOpts { max_window_tokens: crate::longform::max_window_tokens_from_env(), base, ..Default::default() };
    let (video, timings) = crate::pipeline::generate_long(paths, &prompt, &o, &inv.cancel, |done, total, phase| progress(Progress::step(done, total, phase.to_string())))?;
    Ok(video_outcome(&video, &timings, seed))
}

/// Decode + validate `t2a`'s params from an invocation, before any weight is
/// read.
pub fn t2a_params_from(inv: &Invocation) -> Result<crate::pipeline::AudioOpts, String> {
    let d = crate::pipeline::AudioOpts::default();
    let (sampler, eta) = sampler_from(inv)?;
    let seconds = inv.get_f64("seconds");
    if let Some(s) = seconds.filter(|s| !(s.is_finite() && *s > 0.0)) {
        return Err(format!("seconds must be a positive number, got {s}"));
    }
    let base = GenOpts {
        steps: inv.get_i64("steps").unwrap_or(d.base.steps as i64).max(1) as usize,
        guidance: guidance_from(inv)?,
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        sampler,
        eta,
        adapters: adapters_from(inv),
        ..d.base
    };
    Ok(crate::pipeline::AudioOpts { seconds, base })
}

/// Run one text-to-audio generation and wrap the clip as an audio-output
/// [`Outcome`].
pub fn t2a_on(paths: &Paths, inv: &Invocation, opts: &crate::pipeline::AudioOpts, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let (clip, timings) = crate::pipeline::generate_audio(paths, &prompt, opts, &inv.cancel, |done, total, phase| progress(Progress::step(done, total, phase.to_string())))?;
    let out = Outcome::new().set("seed", json!(opts.base.seed)).set("seconds_per_forward", json!(timings.secs_per_forward()));
    Ok(audio_outcome(out, &clip))
}

/// The source clip of a video-input action: its frames as RGB8 at the
/// request's frame rate, and its soundtrack when one was sent.
pub fn clip_from(inv: &Invocation) -> Result<crate::pipeline::Video, String> {
    let frames = capability::blob::decode_video(inv, "video")?;
    let (w, h) = (frames[0].1, frames[0].2);
    let meta_fps = inv.get_blob("video").and_then(|b| b.meta.get("fps")).and_then(|v| v.as_u64());
    let fps = inv.get_i64("fps").map(|f| f as u64).or(meta_fps).unwrap_or(crate::pipeline::PRODUCTION_FPS as u64);
    if fps == 0 {
        return Err("fps must be at least 1".into());
    }
    let audio = match inv.get_blob("soundtrack") {
        Some(b) => {
            let (sample_rate, channels) = audio::wav::parse_planar(&b.bytes).map_err(|e| format!("'soundtrack': {e}"))?;
            Some(crate::audio::AudioClip { channels, sample_rate })
        }
        None => None,
    };
    let to_u8 = |px: Vec<f32>| px.into_iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    Ok(crate::pipeline::Video { width: w, height: h, fps: fps as usize, frames: frames.into_iter().map(|(px, _, _)| to_u8(px)).collect(), audio })
}

/// The generation settings every video-input action shares, resolved
/// against `paths`: the clip fixes the shape, the request the rest.
fn clip_gen_opts(paths: &Paths, inv: &Invocation, clip: &crate::pipeline::Video) -> Result<GenOpts, String> {
    let d = GenOpts::default();
    let (sampler, eta) = sampler_from(inv)?;
    let mut o = GenOpts {
        width: clip.width as usize,
        height: clip.height as usize,
        frames: clip.frames.len(),
        fps: clip.fps,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        guidance: guidance_from(inv)?,
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        sampler,
        eta,
        dit_config: inv.get_str("dit_config").unwrap_or(d.dit_config.clone()),
        audio: inv.get_bool("audio").unwrap_or(false),
        audio_source: clip.audio.clone(),
        adapters: adapters_from(inv),
        ..d
    };
    dit_config_from_name(&o.dit_config)?;
    crate::pipeline::resolve_defaults(&mut o, paths, Named { dit_config: named_in(inv).dit_config, frames: true, size: true, fps: true });
    Ok(o)
}

/// Run one video-input action (`v2v`, `extend`, `upscale`) and wrap the
/// resulting clip as an [`Outcome`]. `place` applies the server's own
/// placement decision (device, device plan) - a request does not make it.
pub fn clip_action_on(action: &str, paths: &Paths, inv: &Invocation, place: &dyn Fn(&mut GenOpts), progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let clip = clip_from(inv)?;
    let mut base = clip_gen_opts(paths, inv, &clip)?;
    place(&mut base);
    let seed = base.seed;
    let mut report = |done: u32, total: u32, phase: &str| progress(Progress::step(done, total, phase.to_string()));
    let (video, timings) = match action {
        "v2v" => {
            let window = (inv.get_f64("start").ok_or("'start' is required")? as f32, inv.get_f64("end").ok_or("'end' is required")? as f32);
            let o = crate::pipeline::MaskedOpts {
                window: Some(window),
                regenerate_video: !inv.get_bool("hold_video").unwrap_or(false),
                regenerate_audio: !inv.get_bool("hold_audio").unwrap_or(false),
                base,
                ..Default::default()
            };
            crate::pipeline::generate_masked(paths, &prompt, &clip, &o, &inv.cancel, &mut report)?
        }
        "extend" => {
            let d = crate::pipeline::ExtendOpts::default();
            let o = crate::pipeline::ExtendOpts {
                frames: inv.get_i64("frames").map_or(d.frames, |n| n.max(1) as usize),
                context_frames: inv.get_i64("context_frames").map_or(d.context_frames, |n| n.max(1) as usize),
                base,
            };
            crate::pipeline::extend(paths, &prompt, &clip, &o, &inv.cancel, &mut report)?
        }
        "upscale" => {
            let d = crate::pipeline::UpscaleOpts::default();
            let o = crate::pipeline::UpscaleOpts { refine_steps: inv.get_i64("refine_steps").map_or(d.refine_steps, |n| n.max(1) as usize), base, ..d };
            crate::pipeline::upscale(paths, &prompt, &clip, &o, &inv.cancel, &mut report)?
        }
        other => return Err(format!("ltxv '{other}' takes no source clip")),
    };
    Ok(video_outcome(&video, &timings, seed))
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
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let (sampler, eta) = sampler_from(inv)?;
    let base = GenOpts {
        frames: inv.get_i64("frames").unwrap_or(d.frames as i64).max(1) as usize,
        width: inv.get_i64("width").unwrap_or(d.width as i64).max(64) as usize,
        height: inv.get_i64("height").unwrap_or(d.height as i64).max(64) as usize,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        guidance: guidance_from(inv)?,
        seed,
        fps: inv.get_i64("fps").unwrap_or(d.fps as i64).max(1) as usize,
        base_shift: inv.get_f64("base_shift").unwrap_or(d.base_shift),
        max_shift: inv.get_f64("max_shift").unwrap_or(d.max_shift),
        stretch: inv.get_bool("stretch").unwrap_or(d.stretch),
        terminal: inv.get_f64("terminal").unwrap_or(d.terminal),
        sampler,
        eta,
        s_noise: d.s_noise,
        context_len: d.context_len,
        dit_config,
        audio: inv.get_bool("audio").unwrap_or(d.audio),
        audio_source: None,
        device: None,
        // DFR's own temporal/canvas layout doesn't route through
        // `generate`'s single-frame-conditioning path - image conditioning
        // is `t2v`-only this pass.
        start_frame: None,
        end_frame: None,
        mid_frame: None,
        mid_frame_at: None,
        conditioning_strength: d.conditioning_strength,
        future_anchors: Vec::new(),
        devices: d.devices,
        adapters: adapters_from(inv),
        reference: None,
        refine_in_context: false,
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
    Ok(video_outcome(&video, &timings, p.opts.base.seed))
}

/// Wrap a generated clip as a video-output [`Outcome`] - the same wire
/// format `wan::caps::video_outcome` produces. `seed` is the one the clip was
/// drawn with, so the outcome alone reproduces it.
pub fn video_outcome(video: &crate::pipeline::Video, timings: &crate::pipeline::Timings, seed: u64) -> Outcome {
    let frames: Vec<(Vec<f32>, u32, u32)> = video.frames.iter().map(|px| (px.iter().map(|&b| b as f32 / 255.0).collect::<Vec<f32>>(), video.width, video.height)).collect();
    let mut blob = match capability::blob::video_blob(&frames) {
        Ok(b) => b,
        Err(e) => return Outcome::new().set("error", json!(e)),
    };
    if let Some(m) = blob.meta.as_object_mut() {
        m.insert("fps".to_string(), json!(video.fps));
    }
    let mut out = Outcome::new()
        .set("frames", json!(video.frames.len()))
        .set("width", json!(video.width))
        .set("height", json!(video.height))
        .set("fps", json!(video.fps))
        .set("seconds_per_forward", json!(timings.secs_per_forward()))
        .set("seed", json!(seed))
        .blob("video", blob);
    if let Some(a) = &video.audio {
        out = audio_outcome(out, a);
    }
    out
}

/// `out` with `clip` attached as its `audio` blob.
///
/// A complete WAV rather than raw samples, matching every other audio output
/// on this workspace's capability surface (`cosyvoice::caps`): the container
/// carries the rate and the channel count with the bytes, so a consumer
/// cannot reconstruct the wrong signal from correct samples.
fn audio_outcome(out: Outcome, clip: &crate::audio::AudioClip) -> Outcome {
    let planes: Vec<&[f32]> = clip.channels.iter().map(Vec::as_slice).collect();
    let bytes = audio::wav::encode_multi(&planes, clip.sample_rate);
    out.set("audio_samples", json!(clip.samples_per_channel()))
        .set("audio_sample_rate", json!(clip.sample_rate))
        .set("audio_seconds", json!(clip.seconds()))
        .blob("audio", capability::Blob::new(Media::Audio, bytes).with_meta(json!({"format": "wav", "sample_rate": clip.sample_rate, "channels": clip.channels.len()})))
}

// ===================== execution (provider) =====================

/// The executable LTX-2.5 model behind the manifest. Unlike `WanProvider`
/// there is no hot-DiT cache: this milestone's DiT is tiny/random and cheap
/// to rebuild per call (microseconds, not the seconds and 5.7 GB of upload a real 22B
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
            "t2a" => {
                let opts = t2a_params_from(inv)?;
                let paths = Paths::from_env()?;
                t2a_on(&paths, inv, &opts, progress)
            }
            action @ ("v2v" | "extend" | "upscale") => {
                let paths = Paths::from_env()?;
                clip_action_on(action, &paths, inv, &|_| {}, progress)
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


/// The sampler parameters every generation action takes ([`Sampler`]).
fn sampler_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("sampler", ParamType::Enum(vec!["euler".into(), "res2s".into()]), "denoising loop: euler (one guided evaluation per step, ancestral when eta > 0 - the distilled pipeline's) or res2s (second-order, two guided evaluations per step with SDE noise - the reference's high-quality pipeline's)").default(json!("euler")),
        ParamSpec::new("eta", ParamType::Float, "sampler noise strength: euler 0 = deterministic, 1 = fully ancestral (default 1); res2s step-level SDE strength (default 0.5)"),
    ]
}

/// The sampler and its eta: an explicit `eta` wins, otherwise the sampler's
/// own reference default.
fn sampler_from(inv: &Invocation) -> Result<(Sampler, f64), String> {
    let sampler = inv.get_str("sampler").map(|s| Sampler::parse(&s)).transpose()?.unwrap_or_default();
    Ok((sampler, sampler.resolve_eta(inv.get_f64("eta"))?))
}

/// The guidance parameters both generation actions take
/// ([`crate::guidance`]): `guidance_preset` picks a starting point, every
/// other parameter overrides its own term. Unset, every stream runs a single
/// conditional forward per step - what the distilled checkpoint denoises with.
fn guidance_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("guidance_preset", ParamType::Enum(vec!["none".into(), "reference".into(), "hq".into()]), "starting guidance: none (one conditional forward per step - the distilled checkpoint), reference (the reference pipeline's non-distilled guidance: CFG 3 video / 7 audio, STG 1 on block 28, rescale 0.7, modality guidance 3) or hq (its high-quality res_2s preset: no STG, rescale 0.45 video / 1.0 audio; pair with sampler res2s)").default(json!("none")),
        ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance scale on both streams; 1 is off (overrides the preset)"),
        ParamSpec::new("audio_guidance", ParamType::Float, "classifier-free guidance scale on the audio stream alone (overrides guidance)"),
        ParamSpec::new("stg", ParamType::Float, "spatio-temporal guidance scale (self-attention bypassed in stg_blocks); 0 is off"),
        ParamSpec::new("audio_stg", ParamType::Float, "spatio-temporal guidance scale on the audio stream (overrides stg)"),
        ParamSpec::new("stg_blocks", ParamType::Str, "comma-separated transformer blocks the STG pass perturbs, e.g. 28"),
        ParamSpec::new("rescale", ParamType::Float, "how far the guided prediction is scaled back to the conditional one's standard deviation, 0..1"),
        ParamSpec::new("audio_rescale", ParamType::Float, "rescale on the audio stream (overrides rescale)"),
        ParamSpec::new("modality_guidance", ParamType::Float, "guidance against the pass with audio<->video cross-attention cut, video stream (a2v); 1 is off"),
        ParamSpec::new("audio_modality_guidance", ParamType::Float, "the same for the audio stream (v2a); 1 is off"),
    ]
}

/// Decode [`guidance_params`] into both streams' guidance.
fn guidance_from(inv: &Invocation) -> Result<crate::guidance::Guidance, String> {
    let mut g = crate::guidance::Guidance::preset(inv.get_str("guidance_preset").as_deref().unwrap_or("none"))?;
    // The broad terms first, so a per-stream one overrides them.
    for name in ["guidance", "stg", "rescale", "stg_blocks", "modality_guidance", "audio_guidance", "audio_stg", "audio_rescale", "audio_modality_guidance"] {
        let value = match inv.params.get(name) {
            Some(serde_json::Value::String(v)) => v.clone(),
            Some(v) => v.to_string(),
            None => continue,
        };
        g.set(name, &value)?;
    }
    g.validate()?;
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A video-input action reads its clip off the wire: RGB8 frames from the
    /// f32 blob, the rate from `fps` over the blob's own `meta.fps`, and the
    /// soundtrack from its WAV; a request without a clip is refused.
    #[test]
    fn a_clip_action_decodes_its_source_clip() {
        let frames: Vec<(Vec<f32>, u32, u32)> = (0..3).map(|k| (vec![k as f32 / 2.0; 2 * 2 * 3], 2, 2)).collect();
        let mut blob = capability::blob::video_blob(&frames).unwrap();
        blob.meta["fps"] = json!(12);
        let wav = audio::wav::encode_multi(&[&[0.0f32, 0.5, -0.5][..]], 16_000);
        let inv = Invocation::new().blob("video", blob).blob("soundtrack", capability::Blob::new(Media::Audio, wav));
        let clip = clip_from(&inv).unwrap();
        assert_eq!((clip.width, clip.height, clip.fps, clip.frames.len()), (2, 2, 12, 3));
        assert_eq!(clip.frames[1][0], 128, "0.5 maps to the nearest byte");
        assert_eq!(clip.audio.as_ref().map(|a| (a.sample_rate, a.samples_per_channel())), Some((16_000, 3)));
        assert_eq!(clip_from(&inv.clone().set("fps", json!(30))).unwrap().fps, 30);
        assert!(clip_from(&Invocation::new()).is_err());
        let m = manifest();
        for name in ["v2v", "extend", "upscale"] {
            let a = m.actions.iter().find(|a| a.name == name).unwrap_or_else(|| panic!("{name} is declared"));
            assert!(a.inputs.iter().any(|i| i.name == "video" && i.required), "{name} takes a source clip");
        }
    }

    /// A `t2v` request that names no `frames` asks for the duration head's
    /// length; one that names them keeps them.
    #[test]
    fn an_unnamed_length_is_the_duration_heads_to_predict() {
        let base = Invocation::new().set("prompt", json!("a lighthouse in a storm"));
        assert!(!gen_params_from(&base).unwrap().named.frames);
        let named = gen_params_from(&base.clone().set("frames", json!(17))).unwrap();
        assert!(named.named.frames);
        assert_eq!(named.opts.frames, 17);
    }

    /// `t2a` takes a prompt and a length and returns a WAV; its sampler's
    /// eta defaults to the chosen loop's reference value, and a length that
    /// is not a positive number is refused before any weight is read.
    #[test]
    fn t2a_is_declared_and_decoded() {
        let m = manifest();
        let t2a = m.actions.iter().find(|a| a.name == "t2a").expect("t2a is declared");
        assert!(t2a.streaming);
        let required: Vec<&str> = t2a.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        assert!(t2a.outputs.iter().any(|o| o.name == "audio" && o.required));
        let inv = |pairs: &[(&str, serde_json::Value)]| pairs.iter().fold(Invocation::new().set("prompt", json!("rain on a tin roof")), |inv, (k, v)| inv.set(k, v.clone()));
        let o = t2a_params_from(&inv(&[("seconds", json!(2.5)), ("sampler", json!("res2s")), ("seed", json!(3))])).unwrap();
        assert_eq!((o.seconds, o.base.sampler, o.base.eta, o.base.seed), (Some(2.5), Sampler::Res2s, 0.5, 3));
        assert_eq!(t2a_params_from(&inv(&[])).unwrap().seconds, None, "unset, the length is the duration head's to predict");
        assert_eq!(o.base.dit_config, "ltx25_22b");
        assert!(o.base.audio);
        for bad in [json!(0.0), json!(-1.0)] {
            assert!(t2a_params_from(&inv(&[("seconds", bad.clone())])).is_err(), "seconds {bad} must be refused");
        }
    }

    /// Weights-free per the serving contract's obligation 5: this must never
    /// touch `BRAIN_LTXV_VAE` or the filesystem.
    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["t2v", "dfr", "t2a", "v2v", "extend", "upscale"]);
        let t2v = &m.actions[0];
        assert!(t2v.streaming, "a multi-step denoise run without progress reads as a hang");
        let required: Vec<&str> = t2v.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        let def = |name: &str| t2v.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no param {name}")).default.clone();
        // The clip's shape and the DiT have no static default: they depend
        // on whether the real checkpoint is configured
        // (`crate::pipeline::resolve_defaults`).
        for name in ["frames", "width", "height", "fps", "dit_config"] {
            assert_eq!(def(name), None, "{name}");
        }
        assert_eq!(def("steps"), Some(json!(4)));
        assert_eq!(def("guidance_preset"), Some(json!("none")));
        assert_eq!(def("guidance"), None, "unset, the preset decides");
        // No declared default any more: an omitted seed resolves to a random
        // value at the resolver (`data::rng::random_seed`), not a fixed 0 -
        // see `gen_params_from`.
        assert_eq!(def("seed"), None);
        let ty = |name: &str| t2v.params.iter().find(|p| p.name == name).unwrap().ty.clone();
        assert!(matches!(ty("dit_config"), ParamType::Enum(v) if v == T2V_DIT_CONFIGS.map(String::from).to_vec()));
        // Two: the picture, always, and the sound when `audio` is set.
        assert_eq!(t2v.outputs.len(), 2);
        assert_eq!(t2v.outputs[0].name, "video");
        assert_eq!(t2v.outputs[0].media, Media::Video);
        assert!(t2v.outputs[0].required);
        assert_eq!(t2v.outputs[1].name, "audio");
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

    /// An unseeded request draws its own seed (a different one each time), a
    /// given seed is kept, and a seed that is not a non-negative integer is
    /// refused instead of being clamped to another one.
    #[test]
    fn t2v_and_dfr_draw_a_seed_when_none_is_given_and_refuse_a_malformed_one() {
        let t2v = |inv: Invocation| gen_params_from(&inv).map(|p| p.opts.seed);
        assert_ne!(t2v(Invocation::new()), t2v(Invocation::new()), "two unseeded requests must not share a seed");
        assert_eq!(t2v(Invocation::new().set("seed", json!(42))), Ok(42));
        assert!(t2v(Invocation::new().set("seed", json!(-1))).is_err());
        let dfr = |inv: Invocation| dfr_params_from(&inv.set("width", json!(64)).set("height", json!(64))).map(|p| p.opts.base.seed);
        assert_ne!(dfr(Invocation::new()), dfr(Invocation::new()));
        assert_eq!(dfr(Invocation::new().set("seed", json!(7))), Ok(7));
        assert!(dfr(Invocation::new().set("seed", json!("x"))).is_err());
    }

    #[test]
    fn video_outcome_round_trips_through_the_shared_clip_codec() {
        let video = crate::pipeline::Video { width: 2, height: 1, fps: 24, frames: vec![vec![255, 0, 0, 255, 0, 0], vec![0, 255, 0, 0, 255, 0]], audio: None };
        let out = video_outcome(&video, &crate::pipeline::Timings::default(), 99);
        assert_eq!(out.outputs["frames"], json!(2));
        assert_eq!(out.outputs["seed"], json!(99), "the outcome reports the seed the clip was drawn with");
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
