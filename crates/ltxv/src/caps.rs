// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's capabilities, declared through the generalized [`capability`]
//! interface - `wan::caps`'s pattern. The manifest is **static** (no weights
//! needed), so capability *discovery* is free; only [`LtxvProvider`]
//! (execution) reads anything from disk. `t2v` runs the real checkpoint when
//! the model store has it and the tiny random-weight smoke test otherwise;
//! `dfr` (`crate::pipeline::generate_dfr`) takes `t2v`'s parameters on the
//! real checkpoint only. `brain ltxv {t2v,dfr}` being a dedicated CLI module
//! does not exempt either action from this surface - the serving contract's
//! obligation 1 is explicit that a bespoke subcommand must never be the ONLY
//! entry point.
//!
//! `inv.cancel` is polled once per denoise step - the poll lives in
//! `crate::pipeline::denoise`'s loop, shared by every caller, per the
//! serving contract's cancellation obligation.

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

use crate::pipeline::{dit_config_from_name, DfrOpts, GenOpts, Named, Paths, Sampler};

/// The model id on the capability surface (`brain caps brain/ltxv`, `brain
/// ltxv <action>`) and the event API.
pub const MODEL: &str = "brain/ltxv";

/// `dit_config_from_name`'s advertised keys for `t2v` - both `"tiny"`
/// (fresh random weights, the smoke-test config) and `"ltx25_22b"` (the
/// real 22B checkpoint via `RealDit`/`crate::dit::forward_q_streamed` - see
/// `crate::pipeline`'s module doc). Selecting `"ltx25_22b"` also needs
/// `Paths::dit` (the model store or `--dit`) set; that check lives in
/// `generate` itself since this manifest is built with no weights loaded.
const T2V_DIT_CONFIGS: [&str; 2] = ["tiny", "ltx25_22b"];

/// The full, static capability manifest - safe to build with no weights
/// loaded. Defaults are [`GenOpts::default`]'s.
pub fn manifest() -> Manifest {
    let d = GenOpts::default();
    let t2v = ActionSpec::new(
        "t2v",
        "generate a video clip from a text prompt (rectified-flow ancestral Euler denoise over the LTX-2.5 video DiT with CFG, causal 3D VAE decode). With the real checkpoint in the model store it runs the real 22B int8 DiT and Gemma-4 text encoder at the reference's defaults; without it, dit_config=tiny's random-weight smoke test with a stub text context - see crate::pipeline's module doc.",
    )
    .streaming()
    .cancellable()
    .param(ParamSpec::new("prompt", ParamType::Str, "what the clip shows (required unless scenes gives one prompt per scene)"))
    .params(guidance_params())
    .param(ParamSpec::new("start_frame", ParamType::Str, "optional path to a PNG/JPEG still: encoded through the real video VAE and used as frame-0 conditioning while the rest of the clip denoises around it (VideoConditionByLatentIndex on its own, VideoConditionByKeyframeIndex when end_frame is also given - see crate::pipeline::conditioned_latent). Needs dit_config=ltx25_22b (the real VAE). Runs under either sampler: eta>0 follows the reference's ancestral loop (LTX-2.5 distilled's own stage-1 sampler), eta=0 the deterministic one.").host_resolved())
    .param(ParamSpec::new("end_frame", ParamType::Str, "optional path to a PNG/JPEG still: a clean image-guidance latent appended at the clip's LAST pixel frame's timestamp, not a replacement of it - the generated token there still denoises freely, only strongly guided toward the still via attention (start_frame given alone is the one case that IS a literal replacement; combined with end_frame/mid_frame it switches to this same appended-guide mechanism too). Same image as start_frame for a clip that loops seamlessly (the generated content in between connects the still to itself); a different image for a clip that morphs from one still to another. Same eta/dit_config requirements as start_frame.").host_resolved())
    .param(ParamSpec::new("mid_frame", ParamType::Str, "optional path to a PNG/JPEG still anchoring ONE INTERIOR instant of the clip (mid_frame_at): a clean image-guidance latent appended at that instant, exactly as end_frame is at the last - one more of keyframes. Same eta/dit_config requirements as start_frame.").host_resolved())
    .param(ParamSpec::new("mid_frame_at", ParamType::Int, "which pixel frame mid_frame anchors, strictly between 0 and frames-1. Unset takes the reference's own position for a single interior keyframe (evenly_spaced_keyframe_positions), (frames-1)/2. Only read when mid_frame is set.").min(0.0).max(capability::bounds::VIDEO_FRAMES))
    .param(ParamSpec::new("keyframes", ParamType::Str, "interior keyframes, comma-separated FRAME:PATH pairs (e.g. 40:a.png,80:b.png): each server-side PNG/JPEG still anchors the pixel frame it names, strictly between 0 and frames-1, as a clean image-guidance latent appended at that instant; the clip moves through every one of them in a single generation. Same eta/dit_config requirements as start_frame.").host_resolved())
    .param(ParamSpec::new("keyframe_strength", ParamType::Float, "ImageConditioningInput.strength for the interior keyframes (mid_frame, keyframes) alone; start_frame and end_frame keep conditioning_strength").min(0.0).max(capability::bounds::UNIT).default(json!(d.keyframe_strength)))
    .param(ParamSpec::new("conditioning_strength", ParamType::Float, "ImageConditioningInput.strength for every given still: 1.0 pins the conditioned tokens to the encoded image exactly, lower values (the reference's own CLI example is 0.8) leave the model room to move. Only read when one of start_frame/mid_frame/keyframes/end_frame is set.").min(0.0).max(capability::bounds::UNIT).default(json!(d.conditioning_strength)))
    // perf-number: duration-head range, a model contract
    .param(ParamSpec::new("frames", ParamType::Int, "video frames; must be of the form 1 + 8k (the causal VAE gives the first frame its own latent frame). Unset, the duration head predicts the length from the prompt (1-20 s) when the real checkpoint, text encoder and duration head are configured; otherwise 121 with the real checkpoint, 9 for the tiny smoke test").min(0.0).max(capability::bounds::VIDEO_FRAMES))
    .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 32); unset, 768 with the real checkpoint, 64 for the smoke test").min(0.0).max(capability::bounds::SIDE_PX))
    .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 32); unset, 512 with the real checkpoint, 64 for the smoke test").min(0.0).max(capability::bounds::SIDE_PX))
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps").min(0.0).max(capability::bounds::DENOISE_STEPS).default(json!(d.steps)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise/weight/context seed (omit for random)"))
    .param(ParamSpec::new("fps", ParamType::Int, "frame rate; unset, 24 with the real checkpoint, 8 for the smoke test").min(0.0).max(capability::bounds::FPS))
    .param(ParamSpec::new("base_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 1024 tokens").default(json!(d.base_shift)))
    .param(ParamSpec::new("max_shift", ParamType::Float, "LTX2Scheduler token-count shift anchor at 4096 tokens").default(json!(d.max_shift)))
    .param(ParamSpec::new("stretch", ParamType::Bool, "stretch the schedule's terminal sigma to `terminal`").default(json!(d.stretch)))
    .param(ParamSpec::new("terminal", ParamType::Float, "terminal sigma the stretch targets").default(json!(d.terminal)))
    .params(sampler_params())
    .param(ParamSpec::new("dit_config", ParamType::Enum(T2V_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "DiT size: ltx25_22b (the real checkpoint - the default when it is configured) or tiny (random weights, a smoke test - the default without it)"))
    .param(ParamSpec::new(
        "audio",
        ParamType::Bool,
        "also generate the clip's SOUND. LTX-2.5 is natively audio-visual: one transformer denoises a video-latent and an audio-latent stream together, coupled every block by cross-attention, and this runs both instead of only the video half. Needs the real checkpoint, text encoder (the audio stream has its OWN text projection the stub cannot stand in for) and audio VAE in the model store; all three are checked before any weight is read. Off by default because of cost: it runs on the same streamed/quantized/device-resident path the video-only stream does, so a joint step costs modestly more than a video-only one at the same shape rather than several times more, but it is still more. Any legal frame count: past one denoising window the audio latent crosses each seam beside the video one, which constrains where a seam may fall (see crate::audio's module doc) and so plans a different window split from the same request without sound.",
    )
    .default(json!(d.audio)))
    .param(ParamSpec::new("audio_source", ParamType::Str, "server-side path to a WAV soundtrack to generate the video FOR (audio-to-video): encoded through the audio VAE and held as the clip's frozen audio stream while the video denoises against it, then returned as the clip's audio. Needs audio=true; past one denoising window each window holds its own slice of it.").host_resolved())
    .param(ParamSpec::new("adapter", ParamType::Str, model::adapter::ADAPTER_LIST_HELP).host_resolved())
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
    // duration head predicts within, sound on request, and a request's first
    // and last frame images as the start and end stills.
    let t2v = t2v.video(video_contract().frame_images(Some("start_frame"), Some("end_frame")));

    // DFR takes everything t2v does - stills, sound, adapters, an unnamed
    // length the duration head predicts - on the real checkpoint only.
    let dfr = ActionSpec {
        name: "dfr".into(),
        summary: "generate a video clip via DFR (Diffusion Fidelity Rendering) on the real LTX-2.5 checkpoint: a keyframe-segment canvas, half-resolution stage 1 with generated keyframe slots, then a spatial x2 upscale and a full-resolution detailing pass under the Refine-Details IC-LoRA with the stage-1 video as its reference. Needs the real checkpoint, text encoder, spatial upscaler and the detailing IC-LoRA in the model store.".into(),
        params: t2v.params.iter().filter(|p| p.name != "dit_config").cloned().collect(),
        presentation: Default::default(),
        ..t2v.clone()
    }
    .param(ParamSpec::new("temporal_upsample_rounds", ParamType::Int, "0, 1, or 2 temporal x2 rounds, each doubling the frame rate").min(0.0).max(8.0).default(json!(0)))
    .param(ParamSpec::new("spatial_upscalings", ParamType::Int, "1: stage 1 at half the output resolution, the detailing pass at it; 2: stage 1 at a quarter, the detailing pass and temporal rounds at half, then a spatial epilogue at the output resolution (width and height multiples of 128)").min(1.0).max(2.0).default(json!(1)));

    // What t2v takes beyond the parameters it shares with dfr.
    let long = crate::pipeline::LongOpts::default();
    let t2v = t2v
        .params(runtime_params())
        .param(ParamSpec::new("scenes", ParamType::Str, "a clip of several scenes instead of prompt/frames: `<frames>:<prompt>` per scene, separated by `|`; each scene is generated with nothing carried from the one before it"))
        .param(ParamSpec::new("context_frames", ParamType::Int, "how much of the clip so far each further denoising window of a long clip sees (1 + 8k)").min(1.0).max(capability::bounds::VIDEO_FRAMES).default(json!(crate::longform::CONTEXT_FRAMES)))
        .param(ParamSpec::new("max_window_tokens", ParamType::Int, "the most latent tokens one denoising window holds; a longer clip rolls several windows").min(1.0).max(capability::bounds::TOKENS).default(json!(long.max_window_tokens)))
        .param(ParamSpec::new("reference_strength", ParamType::Float, "how strongly the reference video conditions the clip (IC-LoRA video reference), 0..1").min(0.0).max(1.0).step(0.05).default(json!(1.0)))
        .param(ParamSpec::new("reference_attention_strength", ParamType::Float, "the weight of attention to the reference, 0..1 (scaled per pixel by reference_attention_mask)").min(0.0).max(1.0).step(0.05).default(json!(1.0)))
        .param(ParamSpec::new("stage_2_ic_lora", ParamType::Bool, "also condition the refinement stage in context (the reference's two-stage IC-LoRA pipeline)").default(json!(false)))
        .input(BlobSpec::new("reference", Media::Video, "a reference video the clip follows (an IC-LoRA's control video: pose, depth, edges)").clip())
        .input(BlobSpec::new("reference_attention_mask", Media::Video, "a grey clip weighting attention to the reference per pixel; needs reference").clip());
    // DFR has no scenes: a prompt is what it renders.
    let mut dfr = dfr.params(runtime_params());
    if let Some(p) = dfr.params.iter_mut().find(|p| p.name == "prompt") {
        p.required = true;
    }

    let a = crate::pipeline::AudioOpts::default();
    let t2a = ActionSpec::new(
        "t2a",
        "generate a sound clip from a text prompt (ltx_pipelines.t2a_one_stage): the LTX-2.5 checkpoint's audio stream denoised alone - the video half is never read - then the audio VAE and vocoder with bandwidth extension. Needs the real checkpoint, text encoder and audio VAE in the model store.",
    )
    .streaming()
    .cancellable()
    .param(ParamSpec::new("prompt", ParamType::Str, "what the clip should sound like").required())
    // perf-number: duration-head range and default length, a model contract
    .param(ParamSpec::new("seconds", ParamType::Float, "clip length in seconds; unset, the duration head predicts it from the prompt (1-20 s), or 5 s without one").max(capability::bounds::SECONDS).min(0.04))
    .params(guidance_params())
    .param(ParamSpec::new("steps", ParamType::Int, "denoise steps (a non-distilled checkpoint's; the distilled one has a fixed schedule)").min(0.0).max(capability::bounds::DENOISE_STEPS).default(json!(a.base.steps)))
    .param(ParamSpec::new("seed", ParamType::Int, "initial-noise seed (omit for random)"))
    .params(sampler_params())
    .param(ParamSpec::new("adapter", ParamType::Str, model::adapter::ADAPTER_LIST_HELP).host_resolved())
    .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier; 1.0 = the reference default. Only read when adapter is set.").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
    .output(BlobSpec::new("audio", Media::Audio, "the generated clip: a complete 48 kHz stereo WAV").required());

    let clip_in = |spec: ActionSpec| -> ActionSpec {
        spec.streaming()
            .input(BlobSpec::new("video", Media::Video, "the source clip, in the standard video wire convention (capability::blob::decode_video): every frame one (w,h), each a multiple of 32").required())
            .input(BlobSpec::new("soundtrack", Media::Audio, "the clip's own soundtrack, a WAV container; needed when audio=true"))
            .param(ParamSpec::new("fps", ParamType::Int, "the clip's frame rate; unset, the video blob's meta.fps, else 24").min(0.0).max(capability::bounds::FPS))
            .param(ParamSpec::new("prompt", ParamType::Str, "what the generated part shows").required())
            .params(guidance_params())
            .params(sampler_params())
            .param(ParamSpec::new("seed", ParamType::Int, "noise seed (omit for random)"))
            .param(ParamSpec::new("steps", ParamType::Int, "denoise steps (non-distilled checkpoints)").min(0.0).max(capability::bounds::DENOISE_STEPS).default(json!(d.steps)))
            .param(ParamSpec::new("dit_config", ParamType::Enum(T2V_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "ltx25_22b (default when the checkpoint is configured) or tiny (random-weight smoke test)"))
            .param(ParamSpec::new("adapter", ParamType::Str, model::adapter::ADAPTER_LIST_HELP).host_resolved())
            .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier; 1.0 = the reference default").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
            .params(runtime_params())
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
    ).cancellable())
    .param(ParamSpec::new("start", ParamType::Float, "the retake window's start, seconds (with end; or give mask instead)").min(0.0).max(3600.0))
    .param(ParamSpec::new("end", ParamType::Float, "the retake window's end, seconds").min(0.0).max(3600.0))
    .input(BlobSpec::new("mask", Media::Video, "instead of a window: the region to regenerate on every frame, a mask sequence (`sam2 track`)"))
    .param(ParamSpec::new("strength", ParamType::Float, "how hard the held region stays the source's, 0..1; 1 holds it exactly").min(0.0).max(1.0).step(0.05).default(json!(crate::pipeline::MaskedOpts::default().strength)))
    .param(ParamSpec::new("keyframes", ParamType::Str, "stills guiding frames of the regenerated region, comma-separated FRAME:PATH pairs of server-side PNG/JPEG files").host_resolved())
    .param(ParamSpec::new("keyframe_strength", ParamType::Float, "how strongly the keyframes guide, 0..1").min(0.0).max(capability::bounds::UNIT).default(json!(GenOpts::default().keyframe_strength)))
    .param(ParamSpec::new("audio", ParamType::Bool, "also run the clip's sound through the same forwards").default(json!(false)))
    .param(ParamSpec::new("hold_video", ParamType::Bool, "regenerate no frame (with audio: video-to-audio)").default(json!(false)))
    .param(ParamSpec::new("hold_audio", ParamType::Bool, "keep the soundtrack everywhere").default(json!(false)));
    let extend = clip_in(ActionSpec::new(
        "extend",
        "continue a clip (video_extend_lora's forward extension): its last frames held as a clean prefix while new frames after them denoise; returns the clip followed by the new frames",
    ).cancellable())
    .param(ParamSpec::new("frames", ParamType::Int, "frames to add, rounded up to a multiple of 8").max(capability::bounds::VIDEO_FRAMES).default(json!(crate::pipeline::ExtendOpts::default().frames)).min(1.0))
    .param(ParamSpec::new("context_frames", ParamType::Int, "how much of the clip's end the continuation sees").max(capability::bounds::VIDEO_FRAMES).default(json!(crate::longform::CONTEXT_FRAMES)).min(1.0))
    .param(ParamSpec::new("seconds", ParamType::Float, "instead of frames: the time to add, at the clip's frame rate").min(0.04).max(capability::bounds::SECONDS))
    .param(ParamSpec::new("audio", ParamType::Bool, "extend the soundtrack too (needs the soundtrack input)").default(json!(false)));
    let upscale = clip_in(ActionSpec::new(
        "upscale",
        "re-render a finished clip at twice its resolution: the real latent x2 upscaler, then the distilled refinement pass against the prompt",
    ).cancellable())
    .param(ParamSpec::new("context_frames", ParamType::Int, "clean frames each refinement pass carries into the next (1 + 8k); fewer buys passes at the cost of continuity").min(1.0).max(capability::bounds::VIDEO_FRAMES).default(json!(crate::longform::CONTEXT_FRAMES)))
    .param(ParamSpec::new("refine_steps", ParamType::Int, "refinement steps, a suffix of the distilled stage-2 schedule").max(capability::bounds::DENOISE_STEPS).default(json!(crate::pipeline::UpscaleOpts::default().refine_steps)).min(1.0));

    let (v2v, extend, upscale) = (edit(v2v, true), edit(extend, true), edit(upscale, false));

    Manifest::new(
        MODEL,
        "LTX-2.5 (Lightricks) - text-to-video diffusion transformer, rectified-flow ancestral Euler sampling with CFG, causal 3D VAE at (8, 32, 32) stride. t2v: real VAE + real scheduler, with the real 22B int8 checkpoint and real Gemma-4 text encoder available via dit_config=ltx25_22b (default stays the tiny random-weight smoke config). dfr: multi-stage pipeline with real latent upscalers, still tiny-random-weight-DiT/stub-context only - real-weight DFR is a tracked gap.",
        vec![t2v, dfr, t2a, v2v, extend, upscale, replace_spec()],
    )
}

/// `replace`: the person clicked on in a clip replaced by the character from
/// a sheet ([`crate::replace`]).
fn replace_spec() -> ActionSpec {
    let t = crate::replace::Tuning::default();
    ActionSpec::new(
        "replace",
        "replace a person in a clip with the character from a sheet, keeping their path, timing and the whole background: SAM 2.1 tracks the person, FLUX.2 paints the character over them on frame 0 and on guide frames, LTX-2.5 regenerates their region guided by those stills, and the result is composited over the source",
    )
    .streaming()
    .cancellable()
    .input(BlobSpec::new("video", Media::Video, "the source clip, any size and length: trimmed at its end to 1 + 8k frames and centre-cropped to multiples of 32 unless conform=false").required().clip())
    .input(BlobSpec::new("character", Media::Image, "the character sheet: who the person becomes").required())
    .input(BlobSpec::new("actor_mask", Media::Video, "the person's track as a mask sequence (`sam2 track`), instead of clicks"))
    .input(BlobSpec::new("soundtrack", Media::Audio, "the sound to carry into the result, a WAV cut to the clip's length; without it the result is silent"))
    .param(ParamSpec::new("prompt", ParamType::Str, "the clip as it should look: the character's looks and clothes, what they do, the scene").required())
    .param(ParamSpec::new("points", ParamType::Str, "clicks on the person in source pixels, 'x,y;x,y' - a few over head, coat and legs pick the whole person"))
    .param(ParamSpec::new("exclude", ParamType::Str, "clicks on things that are NOT the person, 'x,y;x,y'"))
    .param(ParamSpec::new("prompt_frame", ParamType::Int, "the frame the clicks sit on").min(0.0).max(capability::video::MAX_EDIT_CLIP_FRAMES as f64).default(json!(0)))
    .param(ParamSpec::new("edit_prompt", ParamType::Str, "the instruction for the still edits (default: put the sheet's character in the person's place, pose and size, the rest unchanged)"))
    .param(ParamSpec::new("fps", ParamType::Int, "the clip's frame rate when its container states none").min(1.0).max(capability::bounds::FPS))
    .param(ParamSpec::new("conform", ParamType::Bool, "trim and crop a clip onto LTX-2.5's grid; false refuses one that is not").default(json!(true)))
    .param(ParamSpec::new("guide_frames", ParamType::Str, "the frames besides 0 painted as guides, comma-separated, or 'none' (default: every guide_every frames and the last)"))
    .param(ParamSpec::new("guide_every", ParamType::Int, "without guide_frames, a guide every this many frames").min(1.0).max(capability::video::MAX_EDIT_CLIP_FRAMES as f64).default(json!(t.guide_every)))
    .param(ParamSpec::new("guide_strength", ParamType::Float, "how hard the video holds each guide; at 1 it cross-fades between stills instead of moving").min(0.0).max(1.0).step(0.05).default(json!(t.guide_strength)))
    .param(ParamSpec::new("edit_attempts", ParamType::Int, "edits made per still; with ArcFace in the store all are made and the one whose face is closest to the sheet's is kept").min(1.0).max(64.0).default(json!(t.edit_attempts)))
    .param(ParamSpec::new("identity", ParamType::Bool, "keep stills by how close their face is to the sheet's (ArcFace); false keeps them by overlap alone").default(json!(true)))
    .param(ParamSpec::new("min_overlap", ParamType::Float, "the least overlap (IoU) of painted figure and person a still needs").min(0.0).max(1.0).step(0.05).default(json!(t.min_overlap)))
    .param(ParamSpec::new("edit_size", ParamType::Int, "the long side, px, the stills are painted at").min(64.0).max(capability::bounds::SIDE_PX).default(json!(t.edit_long_side)))
    .param(ParamSpec::new("repaint_grow", ParamType::Float, "how far past the person's outline an edit paints, as a fraction of their width").min(0.0).max(4.0).default(json!(t.repaint_grow)))
    .param(ParamSpec::new("body_ratio", ParamType::Float, "a standing person's height over their silhouette's median width: how much body a still paints below a frame that cuts the person off").min(0.1).max(20.0).default(json!(t.body_ratio)))
    .param(ParamSpec::new("paste_grow", ParamType::Int, "the margin, px, a still is pasted back in").min(0.0).max(1024.0).default(json!(t.paste_grow_px)))
    .param(ParamSpec::new("region_grow", ParamType::Int, "the margin, px, around person and figure the video pass regenerates").min(0.0).max(1024.0).default(json!(t.region_grow_px)))
    .param(ParamSpec::new("frame0_inset", ParamType::Int, "how far inside the figure, px, frame 0 is redrawn").min(0.0).max(1024.0).default(json!(t.frame0_inset_px)))
    .param(ParamSpec::new("feather", ParamType::Int, "the feather, px, across every composite edge").min(0.0).max(256.0).default(json!(t.feather_px)))
    .param(ParamSpec::new("seed", ParamType::Int, "the seed of the edits and the video pass (omit for random)").min(0.0))
    .param(ParamSpec::new("steps", ParamType::Int, "video denoising steps (default: the checkpoint's)").min(0.0).max(capability::bounds::DENOISE_STEPS))
    .param(ParamSpec::new("dit_config", ParamType::Enum(T2V_DIT_CONFIGS.iter().map(|s| s.to_string()).collect()), "ltx25_22b (default when the checkpoint is configured) or tiny (random-weight smoke test)"))
    .param(ParamSpec::new("adapter", ParamType::Str, model::adapter::ADAPTER_LIST_HELP).host_resolved())
    .param(ParamSpec::new("lora_scale", ParamType::Float, "strength of every video adapter that names none").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
    .param(ParamSpec::new("edit_adapter", ParamType::Str, "server-side LoRA adapter(s) for the FLUX.2 still edits: `path` or `path@strength`, comma-separated").host_resolved())
    .param(ParamSpec::new("edit_lora_scale", ParamType::Float, "strength of every edit adapter that names none").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
    .param(ParamSpec::new("components", ParamType::Str, "the other models' checkpoints named outright, `<model>.<role>=<path>` comma-separated: flux2.dit, flux2.vae, flux2.text_encoder, flux2.tokenizer, flux2.variant, sam2.weights, sam2.variant, arcface.weights").host_resolved())
    .param(ParamSpec::new("work_dir", ParamType::Str, "server-side directory for the intermediates (the track, the stills, the region, the video pass) a repeated run resumes from (default: one under the system's temporary directory named by the clip and the sheet)").host_resolved())
    .param(ParamSpec::new("fresh", ParamType::Bool, "redo every step instead of reusing the work directory's").default(json!(false)))
    .output(BlobSpec::new("video", Media::Video, "the clip with the person replaced, with the soundtrack when one was given").required())
    .output(BlobSpec::new("audio", Media::Audio, "the soundtrack carried into the result, a complete WAV"))
}

/// Run `replace` ([`crate::replace::run`]) with the other models from
/// `components`.
pub fn replace_on(paths: &Paths, components: &dyn capability::Components, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let clip = clip_from(inv)?;
    let req = crate::replace::Request::from_invocation(inv, &clip)?;
    let seed = req.seed;
    let done = crate::replace::run(paths, components, &req, clip, &inv.cancel, &mut |step, total, phase| progress(Progress::step(step, total, phase.to_string())))?;
    Ok(video_outcome(&done.video, &crate::pipeline::Timings::default(), seed).set("notes", json!(done.notes)).set("work_dir", json!(done.work.to_string_lossy())))
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
    /// A clip of several scenes (`scenes`), else empty: one clip of
    /// `opts.frames` from the prompt.
    pub scenes: Vec<crate::longform::Scene>,
    /// Latent frames each further window of a long clip sees, and the most
    /// tokens one window holds.
    pub context_latent_frames: usize,
    pub max_window_tokens: usize,
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
        o.frames = crate::pipeline::predict_frames(paths, prompt, &o, crate::pipeline::prediction_streams())?;
    }
    Ok(o)
}

/// The invocation's LoRA adapters (`adapter`, each at its own strength or
/// `lora_scale`).
fn adapters_from(inv: &Invocation) -> Result<Vec<model::adapter::AdapterSpec>, String> {
    model::adapter::AdapterSpec::parse_list(&inv.get_str("adapter").unwrap_or_default(), inv.get_f64("lora_scale").unwrap_or(1.0) as f32)
}

/// The execution knobs every generating action takes: where the weights
/// live while it runs and how the text encoder is held. Host-only - they
/// spend the serving machine's memory, which a remote caller does not
/// decide.
fn runtime_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("resident_blocks", ParamType::Int, "DiT blocks kept on the card between steps; the rest stream from host memory (unset: as many as fit)").min(0.0).max(1024.0).host_resolved(),
        ParamSpec::new("two_stage", ParamType::Bool, "run the reference's two-stage pipeline (half-resolution pass, upscale, refine) instead of one stage (unset: the checkpoint's choice)").host_resolved(),
        ParamSpec::new("vae_tiling", ParamType::Bool, "decode the VAE in tiles to bound its memory (unset: when the clip needs it)").host_resolved(),
        ParamSpec::new("text_precision", ParamType::Enum(vec!["fp32".into(), "int8".into()]), "the Gemma-4 text encoder's tier (unset: what fits)").host_resolved(),
        ParamSpec::new("text_cache", ParamType::Bool, "reuse encoded prompts from the on-disk text cache").default(json!(true)).host_resolved(),
        ParamSpec::new("text_cache_max_bytes", ParamType::Int, "bytes the text cache is pruned to after a write").min(1.0).max(1e15).host_resolved(),
        ParamSpec::new("latent_dump", ParamType::Str, "server-side directory each stage's latents are written to, for inspection").host_resolved(),
        ParamSpec::new("av_fp32_reference", ParamType::Bool, "run the audio-video stack at fp32 (the parity reference, slower)").default(json!(false)).host_resolved(),
    ]
}

/// [`runtime_params`] read into `rt`.
fn runtime_from(inv: &Invocation, rt: &mut crate::pipeline::Runtime) -> Result<(), String> {
    if let Some(n) = inv.get_i64("resident_blocks") {
        rt.resident_blocks = Some(u32::try_from(n.max(0)).map_err(|e| format!("resident_blocks: {e}"))?);
    }
    rt.two_stage = inv.get_bool("two_stage").or(rt.two_stage);
    rt.vae_tiling = inv.get_bool("vae_tiling").or(rt.vae_tiling);
    if let Some(p) = inv.get_str("text_precision") {
        rt.text_precision = Some(match p.as_str() {
            "fp32" => gemma4::Precision::Fp32,
            "int8" => gemma4::Precision::Int8,
            other => return Err(format!("text_precision: '{other}' is not fp32 or int8")),
        });
    }
    rt.text_cache.enabled = inv.get_bool("text_cache").unwrap_or(rt.text_cache.enabled);
    if let Some(n) = inv.get_i64("text_cache_max_bytes") {
        rt.text_cache.max_bytes = n.max(1) as u64;
    }
    rt.latent_dump = inv.get_str("latent_dump").filter(|d| !d.is_empty()).or(rt.latent_dump.take());
    rt.av_fp32_reference = inv.get_bool("av_fp32_reference").unwrap_or(rt.av_fp32_reference);
    Ok(())
}

/// A clip input as the pipeline takes it, at `fps`.
fn video_input(inv: &Invocation, name: &str, fps: usize) -> Result<Option<crate::pipeline::Video>, String> {
    if inv.get_blob(name).is_none() {
        return Ok(None);
    }
    let frames = capability::blob::decode_video(inv, name)?;
    let (w, h) = (frames[0].1, frames[0].2);
    let to_u8 = |px: Vec<f32>| px.into_iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    Ok(Some(crate::pipeline::Video { width: w, height: h, fps, frames: frames.into_iter().map(|(px, _, _)| to_u8(px)).collect(), audio: None }))
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
    let mut opts = GenOpts {
        frames: inv.get_i64("frames").unwrap_or(d.frames as i64).max(1) as usize,
        width: inv.get_i64("width").unwrap_or(d.width as i64).max(32) as usize,
        height: inv.get_i64("height").unwrap_or(d.height as i64).max(32) as usize,
        steps: inv.get_i64("steps").unwrap_or(d.steps as i64).max(1) as usize,
        guidance: guidance_from(inv)?,
        negative_prompt: negative_prompt_from(inv),
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
        keyframes: Vec::new(),
        keyframe_strength: inv.get_f64("keyframe_strength").unwrap_or(d.keyframe_strength as f64).clamp(0.0, 1.0) as f32,
        conditioning_strength: inv.get_f64("conditioning_strength").unwrap_or(d.conditioning_strength as f64).clamp(0.0, 1.0) as f32,
        // Not an invocation parameter: placement is the SERVER's decision,
        // not the caller's. A resident instance overrides it (see
        // `crates/catalog/src/resident_ltxv.rs`, which hands each concurrently
        // admitted generation its own card); a one-shot CLI run keeps the
        // default, which uses both cards for one generation's CFG pair.
        future_anchors: Vec::new(),
        devices: d.devices,
        adapters: adapters_from(inv)?,
        reference: None,
        refine_in_context: inv.get_bool("stage_2_ic_lora").unwrap_or(false),
        runtime: Default::default(),
    };
    runtime_from(inv, &mut opts.runtime)?;
    let mut named = named_in(inv);
    let scenes: Vec<crate::longform::Scene> = match inv.get_str("scenes").filter(|s| !s.trim().is_empty()) {
        Some(list) => {
            if inv.get_str("prompt").is_some() || named.frames {
                return Err("scenes carries every scene's frames and prompt, so it cannot be combined with prompt/frames".into());
            }
            let scenes = list.split('|').map(crate::longform::Scene::parse).collect::<Result<Vec<_>, _>>()?;
            opts.frames = scenes.iter().map(|s| s.frames).sum();
            named.frames = true;
            scenes
        }
        None if inv.get_str("prompt").is_none() => return Err("'prompt' is required (or scenes, one prompt per scene)".into()),
        None => Vec::new(),
    };
    if let Some(clip) = video_input(inv, "reference", opts.fps)? {
        let attention_mask = video_input(inv, "reference_attention_mask", opts.fps)?;
        let reference = crate::pipeline::VideoReference::new(clip, inv.get_f64("reference_strength").unwrap_or(1.0) as f32);
        opts.reference = Some(crate::pipeline::VideoReference { attention_strength: inv.get_f64("reference_attention_strength").unwrap_or(1.0) as f32, attention_mask, ..reference });
    } else if inv.get_blob("reference_attention_mask").is_some() || inv.get_f64("reference_attention_strength").is_some_and(|s| s != 1.0) {
        return Err("reference_attention_strength and reference_attention_mask weight a reference video, and there is none".into());
    }
    let context_frames = inv.get_i64("context_frames").unwrap_or(crate::longform::CONTEXT_FRAMES as i64).max(1) as u32;
    let context_latent_frames = crate::vae3d::LtxVaeConfig::conv25().latent_frames(context_frames).ok_or_else(|| format!("context_frames {context_frames} is not of the form 1 + 8k"))? as usize;
    let max_window_tokens = inv.get_i64("max_window_tokens").map_or(crate::pipeline::LongOpts::default().max_window_tokens, |n| n.max(1) as usize);
    use crate::vae3d::LtxVaeConfig;
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(opts.frames as u32).is_none() {
        return Err(format!("frames must be of the form 1 + 8k (1, 9, 17, … ); got {}", opts.frames));
    }
    if !opts.width.is_multiple_of(32) || !opts.height.is_multiple_of(32) {
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", opts.width, opts.height));
    }
    opts.keyframes = keyframes_from(inv, opts.frames)?;
    Ok(GenParams { opts, named, scenes, context_latent_frames, max_window_tokens })
}

/// `t2v`'s interior keyframes: the `keyframes` list and the `mid_frame`
/// still at its resolved instant, in timeline order and checked against the
/// clip ([`crate::pipeline::check_keyframes`]).
fn keyframes_from(inv: &Invocation, frames: usize) -> Result<Vec<(usize, String)>, String> {
    let mut keyframes = crate::pipeline::parse_keyframes(&inv.get_str("keyframes").unwrap_or_default())?;
    if let Some(path) = inv.get_str("mid_frame").filter(|p| !p.is_empty()) {
        let at = inv.get_i64("mid_frame_at").filter(|&n| n >= 0).map(|n| n as usize);
        keyframes.push((crate::pipeline::mid_anchor_frame(frames, at)?, path));
    }
    keyframes.sort_by_key(|(at, _)| *at);
    crate::pipeline::check_keyframes(frames, &keyframes)?;
    Ok(keyframes)
}

/// Run one generation and wrap the result as a video-output [`Outcome`].
/// Cancellation rides in `inv.cancel` - `crate::pipeline::denoise` polls it
/// per step.
pub fn generate_on(paths: &Paths, inv: &Invocation, p: &GenParams, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    if !p.scenes.is_empty() {
        let mut base = p.opts.clone();
        crate::pipeline::resolve_defaults(&mut base, paths, p.named);
        let seed = base.seed;
        let o = crate::pipeline::LongOpts { context_latent_frames: p.context_latent_frames, max_window_tokens: p.max_window_tokens, base };
        let (video, timings) = crate::pipeline::generate_scenes(paths, &p.scenes, &o, &inv.cancel, |done, total, phase| progress(Progress::step(done, total, phase.to_string())))?;
        return Ok(video_outcome(&video, &timings, seed));
    }
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    // Through `generate_long`, which IS `generate` for every request that
    // fits one denoising window and rolls a latent context across several
    // when it does not. No new action parameter: the context size is the
    // reference's own default (`crate::longform::CONTEXT_LATENT_FRAMES`) and
    // the per-window ceiling comes from the environment, so a long `frames`
    // simply works instead of failing at buffer creation.
    let base = resolved_opts(paths, &prompt, p)?;
    let seed = base.seed;
    let o = crate::pipeline::LongOpts { context_latent_frames: p.context_latent_frames, max_window_tokens: p.max_window_tokens, base };
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
        negative_prompt: negative_prompt_from(inv),
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        sampler,
        eta,
        adapters: adapters_from(inv)?,
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
    // A container's rate arrives as a float (`30.0`, `29.97`), which `as_u64`
    // would read as no rate at all.
    let meta_fps = inv.get_blob("video").and_then(|b| b.meta.get("fps")).and_then(|v| v.as_f64()).filter(|f| *f > 0.0).map(|f| f.round() as u64);
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
        negative_prompt: negative_prompt_from(inv),
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        sampler,
        eta,
        dit_config: inv.get_str("dit_config").unwrap_or(d.dit_config.clone()),
        audio: inv.get_bool("audio").unwrap_or(false),
        audio_source: clip.audio.clone(),
        adapters: adapters_from(inv)?,
        ..d
    };
    dit_config_from_name(&o.dit_config)?;
    runtime_from(inv, &mut o.runtime)?;
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
            let window = match (inv.get_f64("start"), inv.get_f64("end")) {
                (Some(a), Some(b)) => Some((a as f32, b as f32)),
                (None, None) => None,
                _ => return Err("v2v: a retake window needs both start and end".into()),
            };
            let mask = inv.get_blob("mask").map(imaging::maskseq::MaskClip::from_blob).transpose()?;
            if window.is_some() == mask.is_some() {
                return Err("v2v regenerates either a time window (start, end) or a mask sequence (mask), exactly one of the two".into());
            }
            let mut base = base;
            base.keyframe_strength = inv.get_f64("keyframe_strength").map_or(base.keyframe_strength, |v| v.clamp(0.0, 1.0) as f32);
            base.future_anchors = crate::pipeline::parse_keyframes(&inv.get_str("keyframes").unwrap_or_default())?;
            let o = crate::pipeline::MaskedOpts {
                window,
                mask,
                strength: inv.get_f64("strength").map_or(crate::pipeline::MaskedOpts::default().strength, |v| v.clamp(0.0, 1.0) as f32),
                regenerate_video: !inv.get_bool("hold_video").unwrap_or(false),
                regenerate_audio: !inv.get_bool("hold_audio").unwrap_or(false),
                base,
            };
            crate::pipeline::generate_masked(paths, &prompt, &clip, &o, &inv.cancel, &mut report)?
        }
        "extend" => {
            let d = crate::pipeline::ExtendOpts::default();
            let frames = match inv.get_f64("seconds") {
                Some(s) => (s * clip.fps as f64).ceil().max(1.0) as usize,
                None => inv.get_i64("frames").map_or(d.frames, |n| n.max(1) as usize),
            };
            let o = crate::pipeline::ExtendOpts {
                frames,
                context_frames: inv.get_i64("context_frames").map_or(d.context_frames, |n| n.max(1) as usize),
                base,
            };
            crate::pipeline::extend(paths, &prompt, &clip, &o, &inv.cancel, &mut report)?
        }
        "upscale" => {
            let d = crate::pipeline::UpscaleOpts::default();
            let context_frames = inv.get_i64("context_frames").unwrap_or(crate::longform::CONTEXT_FRAMES as i64).max(1) as u32;
            let context_latent_frames = crate::vae3d::LtxVaeConfig::conv25().latent_frames(context_frames).ok_or_else(|| format!("context_frames {context_frames} is not of the form 1 + 8k"))? as usize;
            let o = crate::pipeline::UpscaleOpts { refine_steps: inv.get_i64("refine_steps").map_or(d.refine_steps, |n| n.max(1) as usize), context_latent_frames, base, ..d };
            crate::pipeline::upscale(paths, &prompt, &clip, &o, &inv.cancel, &mut report)?
        }
        other => return Err(format!("ltxv '{other}' takes no source clip")),
    };
    Ok(video_outcome(&video, &timings, seed))
}

/// A decoded `dfr` request: `t2v`'s own parameters plus DFR's rounds.
#[derive(Debug)]
pub struct DfrParams {
    pub gen: GenParams,
    pub temporal_upsample_rounds: usize,
    pub spatial_upscalings: usize,
}

/// Decode + validate `dfr`'s params from an invocation - [`gen_params_from`]
/// on the real checkpoint, plus DFR's own constraints, all before any weight
/// load.
pub fn dfr_params_from(inv: &Invocation) -> Result<DfrParams, String> {
    let mut gen = gen_params_from(&inv.clone().set("dit_config", json!("ltx25_22b")))?;
    gen.named.dit_config = true;
    let temporal_upsample_rounds = inv.get_i64("temporal_upsample_rounds").unwrap_or(0).max(0) as usize;
    if temporal_upsample_rounds > 2 {
        return Err(format!("temporal_upsample_rounds must be 0, 1, or 2, got {temporal_upsample_rounds}"));
    }
    let spatial_upscalings = inv.get_i64("spatial_upscalings").unwrap_or(1).max(1) as usize;
    if spatial_upscalings > 2 {
        return Err(format!("spatial_upscalings must be 1 or 2, got {spatial_upscalings}"));
    }
    let (o, divisor) = (&gen.opts, 32usize << spatial_upscalings);
    if gen.named.size && (!o.width.is_multiple_of(divisor) || !o.height.is_multiple_of(divisor)) {
        return Err(format!("{}x{} is not a multiple of {divisor} for DFR at {spatial_upscalings} spatial upscaling(s)", o.width, o.height));
    }
    Ok(DfrParams { gen, temporal_upsample_rounds, spatial_upscalings })
}

/// Run one DFR generation and wrap the result as a video-output [`Outcome`] -
/// [`generate_on`]'s DFR analogue, resolving what the request left unnamed
/// the same way.
pub fn dfr_on(paths: &Paths, inv: &Invocation, p: &DfrParams, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let base = resolved_opts(paths, &prompt, &p.gen)?;
    let opts = DfrOpts { base, temporal_upsample_rounds: p.temporal_upsample_rounds, spatial_upscalings: p.spatial_upscalings };
    let (video, timings) = crate::pipeline::generate_dfr(paths, &prompt, &opts, &inv.cancel, |done, total, phase| progress(Progress::step(done, total, phase.to_string())))?;
    Ok(video_outcome(&video, &timings, opts.base.seed))
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

/// The executable LTX-2.5 model behind the manifest, over the weights the
/// model store resolved for it ([`Paths::from_assembly`]). Every action
/// rebuilds what it needs per call; a resident instance that keeps weights
/// warm across calls is `crates/catalog/src/resident_ltxv.rs`.
pub struct LtxvProvider {
    paths: Arc<Paths>,
    /// Where `replace` finds the other models; without it `replace` refuses.
    components: Option<Arc<dyn capability::Components>>,
}

impl LtxvProvider {
    pub fn from_paths(paths: Paths) -> LtxvProvider {
        LtxvProvider { paths: Arc::new(paths), components: None }
    }

    /// The provider whose `replace` finds SAM 2, FLUX.2 and ArcFace through
    /// `components` ([`crate::replace::component_specs`] names the specs it needs).
    pub fn with_components(paths: Paths, components: Arc<dyn capability::Components>) -> LtxvProvider {
        LtxvProvider { paths: Arc::new(paths), components: Some(components) }
    }
}

/// The provider over a resolved [`capability::Assembly`], with the host's
/// `components` for `replace`.
pub fn provider(assembly: &capability::Assembly, components: Arc<dyn capability::Components>) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(LtxvProvider::with_components(Paths::from_assembly(assembly, None)?, components)))
}

impl Provider for LtxvProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| Arc::new(LtxvAction { name: name.to_string(), paths: self.paths.clone(), components: self.components.clone() }) as Arc<dyn Action>)
    }
}

/// One LTX-2.5 action, dispatched through the shared helpers above.
struct LtxvAction {
    name: String,
    paths: Arc<Paths>,
    components: Option<Arc<dyn capability::Components>>,
}

impl Action for LtxvAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    #[tracing::instrument(level = "info", name = "ltxv_action", skip_all, fields(action = %self.name))]
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        // The SERVED entry point (D-Bus/HTTP/`brain do`), as opposed to the
        // `brain ltxv` CLI: a served request has no terminal to print to, so
        // its start/finish/failure only exist anywhere if they are traced.
        tracing::info!("action invoked");
        let paths = self.paths.as_ref();
        let result = match self.name.as_str() {
            "t2v" => generate_on(paths, inv, &gen_params_from(inv)?, progress),
            "dfr" => dfr_on(paths, inv, &dfr_params_from(inv)?, progress),
            "t2a" => t2a_on(paths, inv, &t2a_params_from(inv)?, progress),
            action @ ("v2v" | "extend" | "upscale") => clip_action_on(action, paths, inv, &|_| {}, progress),
            "replace" => match &self.components {
                Some(c) => replace_on(paths, c.as_ref(), inv, progress),
                None => Err("ltxv replace: this host resolves no other models (SAM 2, FLUX.2, ArcFace) for it".into()),
            },
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
        ParamSpec::new("negative_prompt", ParamType::Str, "what classifier-free guidance steers away from; read only when some guidance scale is not 1 (default: the reference pipelines' artifact list)"),
    ]
}

/// The `negative_prompt` parameter, else the reference's default.
fn negative_prompt_from(inv: &Invocation) -> String {
    inv.get_str("negative_prompt").unwrap_or_else(|| crate::guidance::DEFAULT_NEGATIVE_PROMPT.into())
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

    /// CFG guides against the request's `negative_prompt`, and against the
    /// reference pipelines' own default when it names none.
    #[test]
    fn the_negative_prompt_defaults_to_the_references() {
        let base = Invocation::new().set("prompt", json!("a lighthouse in a storm"));
        assert_eq!(gen_params_from(&base).unwrap().opts.negative_prompt, crate::guidance::DEFAULT_NEGATIVE_PROMPT);
        let named = gen_params_from(&base.set("negative_prompt", json!("fog"))).unwrap();
        assert_eq!(named.opts.negative_prompt, "fog");
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
    /// touch the model store or the filesystem.
    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["t2v", "dfr", "t2a", "v2v", "extend", "upscale", "replace"]);
        let t2v = &m.actions[0];
        assert!(t2v.streaming, "a multi-step denoise run without progress reads as a hang");
        let required: Vec<&str> = t2v.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert!(required.is_empty(), "prompt or scenes, checked when decoded: {required:?}");
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
        let inputs: Vec<(&str, bool)> = t2v.inputs.iter().map(|b| (b.name.as_str(), b.required)).collect();
        assert_eq!(inputs, [("reference", false), ("reference_attention_mask", false)], "t2v's only binary inputs are the optional reference video and its mask");
        let j = m.to_json();
        assert_eq!(j["model"], MODEL);
        assert_eq!(j["actions"][0]["streaming"], true);
        assert_eq!(j["actions"][0]["params"][0]["name"], "prompt");
        assert_eq!(j["actions"][0]["params"][0]["required"], false, "prompt or scenes");
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
        let t2v = m.actions.iter().find(|a| a.name == "t2v").unwrap();
        let names = |a: &ActionSpec| a.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        const T2V_ONLY: [&str; 7] = ["dit_config", "scenes", "context_frames", "max_window_tokens", "reference_strength", "reference_attention_strength", "stage_2_ic_lora"];
        let mut want = names(t2v);
        want.retain(|n| !T2V_ONLY.contains(&n.as_str()));
        want.push("temporal_upsample_rounds".into());
        want.push("spatial_upscalings".into());
        let (mut got, mut want) = (names(dfr), want);
        got.sort();
        want.sort();
        assert_eq!(got, want, "dfr takes t2v's parameters on the real checkpoint, plus its rounds");
        assert!(dfr.params.iter().any(|p| p.name == "prompt" && p.required), "dfr renders a prompt, never scenes");
        let outputs = |a: &ActionSpec| a.outputs.iter().map(|b| (b.name.clone(), b.required)).collect::<Vec<_>>();
        assert_eq!(outputs(dfr), outputs(t2v), "dfr returns what t2v does, sound included");
    }

    /// [`the_advertised_defaults_and_dit_configs_decode`]'s DFR analogue.
    #[test]
    fn dfr_advertised_defaults_decode() {
        let spec = manifest().actions.into_iter().find(|a| a.name == "dfr").unwrap();
        let inv = spec.validate(Invocation::new().set("prompt", json!("a cat"))).unwrap();
        let p = dfr_params_from(&inv).unwrap();
        assert_eq!(p.gen.opts.dit_config, "ltx25_22b", "DFR runs the real checkpoint");
        assert!(!p.gen.named.frames, "an unnamed length is the duration head's to predict");
        assert_eq!(p.temporal_upsample_rounds, 0);
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
        let e = decode(base().set("width", json!(96)).set("height", json!(64))).unwrap_err();
        assert!(e.contains("multiple of 64"), "{e}");
        let e = decode(base().set("temporal_upsample_rounds", json!(3))).unwrap_err();
        assert!(e.contains("temporal_upsample_rounds"), "{e}");
        let p = decode(base().set("frames", json!(17)).set("width", json!(128)).set("height", json!(128)).set("temporal_upsample_rounds", json!(2))).unwrap();
        assert_eq!((p.gen.opts.frames, p.gen.opts.width, p.gen.opts.height, p.temporal_upsample_rounds), (17, 128, 128, 2));
    }

    /// An unseeded request draws its own seed (a different one each time), a
    /// given seed is kept, and a seed that is not a non-negative integer is
    /// refused instead of being clamped to another one.
    #[test]
    fn t2v_and_dfr_draw_a_seed_when_none_is_given_and_refuse_a_malformed_one() {
        let t2v = |inv: Invocation| gen_params_from(&inv.set("prompt", json!("p"))).map(|p| p.opts.seed);
        assert_ne!(t2v(Invocation::new()), t2v(Invocation::new()), "two unseeded requests must not share a seed");
        assert_eq!(t2v(Invocation::new().set("seed", json!(42))), Ok(42));
        assert!(t2v(Invocation::new().set("seed", json!(-1))).is_err());
        let dfr = |inv: Invocation| dfr_params_from(&inv.set("prompt", json!("p")).set("width", json!(64)).set("height", json!(64))).map(|p| p.gen.opts.seed);
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
