// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's capability surface - the [`capability`] `Provider`/`Action`
//! wiring, declared through the generalized `capability` interface -
//! `wan::caps`'s pattern (the closest precedent, read in full before writing
//! this module), adapted to H3's own two real differences from it.
//!
//! ## Two things this module does that `wan::caps` does not have to
//!
//! * **`H3Transformer::forward` takes a packed sequence of ANY length per
//!   call - no latent-extent-sized graph is baked in at `load` time**, unlike
//!   `wan::pipeline::HotDit` (a compiled RoPE/kernel graph genuinely sized to
//!   `(frames, width, height)`) or `ltxv`'s own per-forward-rebuilt streamed
//!   blocks. So the expensive resident object (the loaded DiT's device
//!   buffers) does not need to be rebuilt per request shape at all - one
//!   resident [`crate::model::H3Transformer`] serves every `(canvas,
//!   num_frames)` a checkpoint's device placement is asked for. See
//!   [`crate::pipeline::t2va_hot`]/[`crate::pipeline::fl2va_hot`]'s own doc.
//! * **Two coupled latent extents, not one.** A request fixes BOTH a video
//!   latent extent (`num_latent_frames x latent_height x latent_width`) and
//!   an audio latent extent (`num_audio_latents`) from the SAME aligned pixel
//!   frame count - `crate::pipeline::video_latent_num_frames`/
//!   `audio_latent_num_frames` are two different closed forms over one
//!   `aligned_num_frames`, not a shared divisor. `ltxv`'s own instance key
//!   (`crates/cli/src/resident_ltxv.rs::parse_key`, checked before writing
//!   this module) does NOT solve this: its `audio: bool` request parameter is
//!   not part of its instance key at all - LTX's audio latent track is
//!   derived from the SAME video frame/fps pair with no extra degree of
//!   freedom the key needs to carry, so omitting it from the key is correct
//!   there, not an analogous solution to copy. What `ltxv` DOES have worth
//!   copying is a DIFFERENT pattern, from `crates/cli/src/resident_ltxv.rs`'s
//!   own module doc: its resident weight cache is keyed on checkpoint
//!   identity + device, DECOUPLED from `InstanceKey`, precisely because its
//!   own resident object (the block-weight cache) does not depend on the
//!   request shape either. [`crate::caps::instance_key`] below follows that
//!   same split: the key still names both latent extents (task + video
//!   extent + audio extent + dtype, matching this workspace's residency-key
//!   convention of naming what a request needs to BUILD/ESTIMATE against, and
//!   forward-compatible with a future compiled-graph device backend that
//!   would genuinely need one graph per extent), while
//!   `crates/catalog/src/resident_minimaxh3.rs`'s actual hot [`crate::model::
//!   H3Transformer`] is held ONE PER DEVICE, shared across every key -
//!   switching request shape never forces a reload of the 33B DiT.
//!
//! ## Why this gate exists
//!
//! The MiniMax H3 Community License Agreement is not Apache-2.0. It carries:
//! a territorial carve-out (the ordinary grant excludes the EU, UK, South
//! Korea and the US - an organization in one of those regions needs
//! MiniMax's separate authorization), a >$20M/yr revenue registration
//! clause, and a mandatory "MiniMax H3" UI attribution requirement for any
//! commercial product or service. This is a stricter license than
//! `minimaxmusic3`'s (which has no commercial-use restriction, so that
//! crate carries only a prose note in its own user-facing docs page, no
//! runtime gate) - modeled instead on the one runtime license gate that
//! exists in this workspace, `flux2::caps::check_license`, byte-for-byte.
//!
//! This crate's Rust *implementation* is Apache-2.0, ported from the
//! Apache-2.0 `diffusers`/`transformers` reference (see this crate's module
//! doc). What this gate protects is the *weights*: brain never vendors or
//! auto-fetches them (`arch::ARCHS`'s `minimaxh3` row has `default_ref:
//! None`), and running against a checkpoint the operator obtained themself
//! still requires an explicit opt-in - the same "the code is Apache, the
//! weights are the operator's problem to clear" split `supir`/`flux2`'s 9B
//! variant already establish in this tree.

use std::sync::Once;

/// Refuse to run against MiniMax-H3 weights unless the operator has
/// confirmed they are authorized under the MiniMax H3 Community License
/// Agreement (including, if applicable, its territorial restrictions) -
/// then print the attribution notice once per process. `allowed` is that
/// confirmation as the request carried it (`allow_community`, the CLI's
/// `--allow-community`), so each request answers for itself.
///
/// Called from every entry point that would touch real H3 weights (the
/// eventual `gen_params_from`-equivalent in this module, and each CLI
/// handler), the same call-site discipline `flux2::caps::check_license`
/// uses so no served surface can bypass it.
pub fn check_license(allowed: bool) -> Result<(), String> {
    if !allowed {
        return Err(
            "MiniMax-H3 weights are released under the MiniMax H3 Community License Agreement \
             (territorial restrictions apply - the ordinary grant excludes the EU, UK, South \
             Korea and the US; a >$20M/yr revenue registration clause and a mandatory \"MiniMax \
             H3\" UI attribution requirement also apply). Pass --allow-community \
             (allow_community: true in a request) to confirm you are authorized to use \
             these weights under that license."
                .into(),
        );
    }
    static NOTICE: Once = Once::new();
    NOTICE.call_once(|| {
        eprintln!(
            "minimaxh3: MiniMax H3 Community License weights enabled - territorial and \
             revenue-cap restrictions apply"
        );
    });
    Ok(())
}

// ============================================================================
// The manifest - static, weight-free (safe to build with nothing loaded).
// ============================================================================

use capability::{ActionSpec, BlobSpec, Manifest, Media, ParamSpec, ParamType};
use serde_json::json;

use crate::config::H3TransformerConfig;
use crate::pipeline;
use crate::video_vae::VideoVaeConfig;

/// The model id used on the CLI (`brain do brain/minimaxh3 …`) and the event
/// API.
pub const MODEL: &str = "brain/minimaxh3";

/// The task enum, in manifest order - `ref2va` is deliberately absent: it is
/// not implemented (`crate::pipeline`'s own module doc, "deliberately not
/// implemented here"), and advertising an action that cannot run is worse
/// than not advertising it, the same rule `wan::caps` states for I2V.
const TASKS: [&str; 3] = ["t2va", "fl2va", "ref2va"];

/// `video_latent_num_frames`/`audio_latent_num_frames`/the canvas-multiple
/// check all need the real DiT/VAE config numbers, and every one of those is
/// pure data - no weight has to be read to compute them. Shared by the
/// manifest's own defaults, [`gen_params_from`]'s geometry validation, and
/// [`instance_key`].
fn real_configs() -> (H3TransformerConfig, VideoVaeConfig) {
    (H3TransformerConfig::real(), VideoVaeConfig::real())
}

/// The smallest `num_frames` that clears MiniMax-H3's own 5-15 second
/// duration bound at [`pipeline::FPS`] once aligned - the manifest's own
/// `num_frames` default, so a request naming only a prompt runs a legal
/// duration rather than failing on the bound `pipeline::generate` itself
/// enforces (`crate::pipeline`'s own tiny-config test picks the same value
/// for the identical reason).
fn default_num_frames() -> u32 {
    let (_, vae_cfg) = real_configs();
    pipeline::align_num_frames((pipeline::MIN_DURATION_S * pipeline::FPS) as u32, vae_cfg.clip_length, vae_cfg.tokens_chunk_size()).expect("default_num_frames: align_num_frames")
}

/// The full, static capability manifest.
pub fn manifest() -> Manifest {
    let default_frames = default_num_frames();
    let common = |a: ActionSpec| -> ActionSpec {
        a.streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired clip's video and sound").required())
            .param(ParamSpec::new("width", ParamType::Int, "canvas width, px - must be given together with height (a multiple of the VAE spatial stride x the DiT's spatial patch size); omit both to resolve a 16:9 canvas at MiniMax-H3's own short-edge/max-pixels defaults").min(0.0).max(capability::bounds::SIDE_PX))
            .param(ParamSpec::new("height", ParamType::Int, "canvas height, px - see width").min(0.0).max(capability::bounds::SIDE_PX))
            // perf-number: frame-rate constant / memory ratio, not a measured timing
            .param(ParamSpec::new("num_frames", ParamType::Int, "video frames at 24fps before alignment; snapped up to the VAE's own chunk grid, and the aligned duration must land in MiniMax-H3's 5-15 second bound").min(0.0).max(capability::bounds::VIDEO_FRAMES).default(json!(default_frames)))
            .param(ParamSpec::new("steps", ParamType::Int, "denoise steps (shared by both the video and the audio schedule, one forward per step - no CFG, this model is guidance-distilled)").min(1.0).max(capability::bounds::DENOISE_STEPS).default(json!(20)))
            .param(ParamSpec::new("seed", ParamType::Int, "initial-noise seed (omit for random; the outcome reports the one used)"))
            .param(ParamSpec::new("int8_activation_levels", ParamType::Int, "int8 levels the DiT's linears quantize their activations to against its int8 weights: 1 (the int8 tier's ordinary arithmetic), or 2 (also quantizes what the first level rounds away - closer to the fp32 model, at a second product per linear)").min(1.0).max(2.0).default(json!(1)))
            .param(ParamSpec::new("dit_placement", ParamType::Enum(pipeline::DitPlacement::NAMES.iter().map(|n| n.to_string()).collect()), "where the DiT's weights live while it denoises: resident (held on the cards at int8, refused naming the bytes when they cannot hold it with this request's workspace) or streamed (one block at a time at fp32, read from the checkpoint every step: any card holds it, many times slower)").default(json!("resident")))
            .param(ParamSpec::new("adapter", ParamType::Str, "server-side path to a LoRA adapter for this action's DiT (a diffusers PEFT save, such as a few-step distillation); folded into the weights before they are quantized").host_resolved())
            .param(ParamSpec::new("lora_scale", ParamType::Float, "LoRA strength multiplier on top of the adapter's own alpha/rank; 1.0 = as trained. Only read when adapter is set.").default(json!(1.0)).min(0.0).max(4.0).step(0.05))
            .param(ParamSpec::new("vsa_sparsity", ParamType::Float, "video sparse attention: the share of the video's key tiles each video query drops (0.75 keeps a quarter), selected per head by tile-mean scores; text, references and audio always stay dense. Omitted: the preset's own, else 0 (dense attention)").min(0.0).max(0.99).step(0.05))
            .param(ParamSpec::new("vsa_gate", ParamType::Str, "server-side path to the trained to_gate_compress of every DiT block (a FastVideo VSA checkpoint or a gate-only extract): adds the coarse branch video sparse attention was distilled with").host_resolved())
            .param(ParamSpec::new("allow_community", ParamType::Bool, "confirm you are authorized to use the MiniMax-H3 weights under the MiniMax H3 Community License Agreement (territorial, revenue-cap and attribution terms apply); every request is refused without it").default(json!(false)))
            .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames in [0,1], meta {frames,w,h,c,fps}").required())
            .output(BlobSpec::new("audio", Media::Audio, "the clip's own stereo sound track: a complete 32 kHz WAV covering the same time window, generated by the SAME forward as the frames - MiniMax-H3 is natively audio-visual, so this is never absent").required())
    };

    // What `/videos` maps onto t2va: the VAE's chunk grid at a fixed frame rate,
    // the released canvas multiple and pixel budget, the released duration range, sound always.
    // fl2va and ref2va render the same clips; the server writes a request's
    // frame images and references to private files and fills their path
    // params, so a caller never names a path.
    let (_, vae_cfg) = real_configs();
    let contract = capability::video::VideoContract::new(
        "num_frames",
        (vae_cfg.tokens_chunk_size(), vae_cfg.clip_length),
        capability::video::Fps::Fixed(pipeline::FPS as u32),
        ("width", "height"),
        canvas_multiple(),
        u64::from(pipeline::CANVAS_MAX_PIXELS),
        (f64::from(pipeline::MIN_DURATION_S), f64::from(pipeline::MAX_DURATION_S)),
    )
    .audio(capability::video::Audio::Always)
    .seed("seed");
    let t2va = common(ActionSpec::new(
        TASKS[0],
        "generate a video clip AND its own soundtrack from a text prompt (Qwen3-VL text conditioning, one packed self-attention DiT denoising video and audio rows together under two independent shifted-sigma Euler schedules in the SAME forward, causal 3D video VAE + DAC/BigVGAN audio VAE decode)",
    ))
    .video(contract.clone());

    let fl2va = common(ActionSpec::new(
        TASKS[1],
        "t2va, plus the clip anchored to still images at its first frame, its last frame and/or interior frames: the given stills are VAE-encoded, lightly noised, and prepended to the packed sequence as extra conditioning rows, each at its own frame's rotary time, that the denoise loop never resamples - the generated clip is guided through every one of them",
    ))
    // `host_resolved`: these name still files on the machine that holds the
    // weights - `ltxv::caps`' `start_frame`/`end_frame`/`mid_frame` pattern.
    // A caller standing on the host (`brain do`) sets them; every off-machine
    // surface refuses them and would take the still's content as a blob
    // instead.
    .param(ParamSpec::new("first_frame", ParamType::Str, "server-side path to a PNG/JPEG still the clip's FIRST frame is anchored to. At least one of first_frame/last_frame/keyframes is required.").host_resolved())
    .param(ParamSpec::new("last_frame", ParamType::Str, "server-side path to a PNG/JPEG still the clip's LAST frame is anchored to. At least one of first_frame/last_frame/keyframes is required.").host_resolved())
    .param(ParamSpec::new("keyframes", ParamType::Str, "interior keyframes, comma-separated FRAME:PATH pairs in timeline order (e.g. 40:a.png,80:b.png): each server-side PNG/JPEG still anchors the pixel frame it names, strictly between the clip's first and last").host_resolved())
    .video(contract.clone().frame_images(Some("first_frame"), Some("last_frame")));

    let ref2va = common(ActionSpec::new(
        TASKS[2],
        "t2va conditioned on reference media - images, videos and soundtracks the clip should draw its subjects, look and sound from (MiniMax-H3's separately trained transformer_ref). Each reference is labelled in the Qwen3-VL presentation and packed, at its own resolution, ahead of the generated rows; the prompt can name them as <Picture i>, <Video k> and <Audio j>",
    ))
    .param(
        ParamSpec::new(
            "references",
            ParamType::Str,
            "server-side reference paths, comma-separated, in the order the prompt numbers them: PNG/JPEG/WebP images, video files, WAV soundtracks; a video's soundtrack is attached as video.mp4+track.wav. At most 9 images, 3 videos, 3 soundtracks and 12 in total, and soundtracks never alone.",
        )
        .required()
        .host_resolved(),
    )
    .param(ParamSpec::new("preset", ParamType::Enum(PRESETS.iter().map(|p| p.to_string()).collect()), "what the DiT runs with, from the model store beside the release: base (the release alone), turbo-4step (lightx2v's 4-step ref2v adapter: run it at steps 4) or turbo-4step-vsa (that adapter and FastVideo's trained sparse-attention gates, at vsa_sparsity 0.75 unless one is asked)").default(json!("base")))
    .video(contract.clone().references("references", ',', &[Media::Image, Media::Video, Media::Audio], pipeline::MAX_REFERENCES));

    Manifest::new(
        MODEL,
        "MiniMax-H3 (MiniMax) - a 33B joint video+audio rectified-flow diffusion transformer: one packed self-attention stack denoises video, audio and text-conditioned rows together (not a two-stream architecture like ltxv), under two independent shifted-sigma Euler schedules in one forward, no CFG (guidance-distilled). Qwen3-VL text conditioning truncated to an intermediate hidden layer; causal 3D video VAE + DAC-lineage/BigVGAN audio VAE. t2va: text only. fl2va: text plus still images anchoring its first, last and/or interior frames. ref2va: text plus up to 12 image/video/audio references.",
        vec![t2va, fl2va, ref2va],
    )
}

// ===================== shared param decoding =====================

/// The names `ref2va`'s `preset` takes.
pub const PRESETS: [&str; 3] = ["base", "turbo-4step", "turbo-4step-vsa"];
/// `(vendor, repo, file)` of the adapter the turbo presets run with:
/// lightx2v's 4-step distillation of `transformer_ref`.
pub const PRESET_ADAPTER: (&str, &str, &str) = ("lightx2v", "Minimax-h3-Turbo", "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors");
/// `(vendor, repo, file)` of the VSA gates `turbo-4step-vsa` runs with,
/// extracted from FastVideo's FastH3 4-step VSA checkpoint.
pub const PRESET_GATES: (&str, &str, &str) = ("barelymining", "ComfyUI-MiniMax-H3-FastVideo", "fasth3_vsa_gate.safetensors");
/// The sparsity `turbo-4step-vsa` runs at unless a request asks another.
pub const PRESET_VSA_SPARSITY: f32 = 0.75;

/// One decoded `t2va`/`fl2va` request's numeric options - everything BOTH
/// actions take. [`fl2va_keyframes_from`] decodes `fl2va`'s own extra
/// `first_frame`/`last_frame` params separately, since only `fl2va` has them.
#[derive(Debug)]
pub struct GenParams {
    pub opts: pipeline::GenOpts,
    /// The LoRA adapter the DiT runs with (`crate::lora`), `None` for the
    /// base model.
    pub adapter: Option<model::adapter::AdapterSpec>,
    /// `lora_scale`: the strength of the adapter, explicit or the preset's.
    adapter_scale: f32,
    /// The blocks' trained VSA gates (`crate::vsa`), `None` without them.
    pub vsa_gate: Option<String>,
    /// The request's `preset`, resolved onto `adapter`/`vsa_gate` by
    /// [`resolve_preset`]; `"base"` asks nothing.
    pub preset: String,
    /// Whether the request named its own `vsa_sparsity`.
    sparsity_asked: bool,
}

/// The adapter and gate files a request runs its DiT with, its preset
/// resolved as a run resolves it - what a residency plan prices before the
/// request runs.
pub fn requested_extras(inv: &capability::Invocation, paths: &Paths) -> Result<(Option<String>, Option<String>), String> {
    let mut params = decode_gen_params(inv)?;
    resolve_preset(&mut params, paths)?;
    Ok((params.adapter.map(|a| a.path), params.vsa_gate))
}

/// Fill `params`' adapter, gates and sparsity from its preset: pinned files
/// of model-store repos, found under the store that holds the release
/// `paths` was resolved from. A preset beside a host-side adapter or gate is
/// refused - which of them should run is ambiguous.
pub fn resolve_preset(params: &mut GenParams, paths: &Paths) -> Result<(), String> {
    let (adapter, gates) = match params.preset.as_str() {
        "base" => return Ok(()),
        "turbo-4step" => (true, false),
        "turbo-4step-vsa" => (true, true),
        other => return Err(format!("minimaxh3: unknown preset '{other}' ({})", PRESETS.join(", "))),
    };
    if params.adapter.is_some() || params.vsa_gate.is_some() {
        return Err(format!("minimaxh3: preset '{}' names its own adapter and gates; drop the explicit adapter/vsa_gate or the preset", params.preset));
    }
    let release = std::path::Path::new(&paths.text_encoder).parent().filter(|r| crate::spec::is_h3_pipeline_root(r)).ok_or("minimaxh3: a preset is served from a model-store release")?;
    let store = release.parent().and_then(std::path::Path::parent).ok_or("minimaxh3: the release is not inside a model store")?;
    let file = |(vendor, repo, name): (&str, &str, &str)| {
        let path = brain_modelstore::Store::new(store).repo_dir(&brain_modelref::ModelRef::new(vendor, repo, None)).join(name);
        match path.is_file() {
            true => path.to_str().map(str::to_string).ok_or_else(|| format!("minimaxh3: {} is not UTF-8", path.display())),
            false => Err(format!("minimaxh3: preset '{}' runs {name} of {vendor}/{repo}, which the model store beside the release does not hold", params.preset)),
        }
    };
    if adapter {
        params.adapter = Some(model::adapter::AdapterSpec { path: file(PRESET_ADAPTER)?, scale: params.adapter_scale });
    }
    if gates {
        params.vsa_gate = Some(file(PRESET_GATES)?);
        if !params.sparsity_asked {
            params.opts.vsa_sparsity = PRESET_VSA_SPARSITY;
        }
    }
    Ok(())
}

/// Decode + validate the shared params from an invocation, and check the
/// license gate. Every geometric/duration constraint MiniMax-H3 itself
/// enforces is checked HERE too, from the real config's own numbers (pure
/// data, no weight read) - a request that could never run must not cost a
/// checkpoint load to reject, the same discipline `wan::caps::gen_params_from`
/// documents.
pub fn gen_params_from(inv: &capability::Invocation) -> Result<GenParams, String> {
    check_license(inv.get_bool("allow_community").unwrap_or(false))?;
    decode_gen_params(inv)
}

/// [`gen_params_from`]'s body, minus the license check - split out so the
/// pure decode/validation logic is testable on its own.
fn decode_gen_params(inv: &capability::Invocation) -> Result<GenParams, String> {
    let (dit_cfg, vae_cfg) = real_configs();

    let width = inv.get_i64("width");
    let height = inv.get_i64("height");
    let canvas = match (width, height) {
        (Some(w), Some(h)) => {
            let canvas_multiple = vae_cfg.spatial_compression_ratio() * dit_cfg.patch_size[2];
            let (w, h) = (w.max(1) as u32, h.max(1) as u32);
            if !w.is_multiple_of(canvas_multiple) || !h.is_multiple_of(canvas_multiple) {
                return Err(format!("gen_params_from: {w}x{h} is not a multiple of {canvas_multiple} (video VAE spatial stride x DiT patch size)"));
            }
            Some((h, w))
        }
        (None, None) => None,
        _ => return Err("gen_params_from: width and height must be given together".to_string()),
    };

    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let num_frames = inv.get_i64("num_frames").unwrap_or(default_num_frames() as i64).max(1) as u32;
    let aligned = pipeline::align_num_frames(num_frames, vae_cfg.clip_length, vae_cfg.tokens_chunk_size())?;
    let duration = aligned as f32 / pipeline::FPS;
    if !(pipeline::MIN_DURATION_S..=pipeline::MAX_DURATION_S).contains(&duration) {
        return Err(format!(
            "gen_params_from: MiniMax-H3 generates between {}s and {}s at {}fps, got num_frames={num_frames} (aligned {aligned}, {duration}s)",
            pipeline::MIN_DURATION_S,
            pipeline::MAX_DURATION_S,
            pipeline::FPS
        ));
    }

    let opts = pipeline::GenOpts {
        canvas,
        num_frames,
        vsa_sparsity: match inv.get_f64("vsa_sparsity").unwrap_or(0.0) {
            v if (0.0..1.0).contains(&v) => v as f32,
            v => return Err(format!("minimaxh3: vsa_sparsity must be in [0, 1), got {v}")),
        },
        denoise_steps: match inv.get_i64("steps").unwrap_or(20) {
            n if n >= 1 => n as usize,
            n => return Err(format!("minimaxh3: steps must be at least 1 (one transformer forward per step), got {n}")),
        },
        seed,
        // Not an invocation parameter: placement is the SERVER's decision,
        // not the caller's - `ltxv::caps::gen_params_from`'s own identical
        // comment. A resident instance overrides this after decoding
        // (`crates/catalog/src/resident_minimaxh3.rs`); the direct provider
        // above leaves it `None` (the host default).
        device: None,
        act_levels: match inv.get_i64("int8_activation_levels").unwrap_or(1) {
            1 => crate::block::ActLevels::One,
            2 => crate::block::ActLevels::Two,
            n => return Err(format!("gen_params_from: int8_activation_levels must be 1 or 2, got {n}")),
        },
        dit_placement: pipeline::DitPlacement::parse(&inv.get_str("dit_placement").unwrap_or_else(|| "resident".into()))?,
    };
    let adapter_scale = match inv.get_f64("lora_scale").unwrap_or(1.0) {
        scale if scale.is_finite() && scale >= 0.0 => scale as f32,
        scale => return Err(format!("gen_params_from: lora_scale must be a non-negative number, got {scale}")),
    };
    let adapter = inv.get_str("adapter").filter(|p| !p.trim().is_empty()).map(|path| model::adapter::AdapterSpec { path: path.to_string(), scale: adapter_scale });
    let vsa_gate = inv.get_str("vsa_gate").filter(|p| !p.trim().is_empty()).map(|p| p.to_string());
    let preset = inv.get_str("preset").unwrap_or_else(|| "base".into()).to_string();
    let sparsity_asked = inv.get_f64("vsa_sparsity").is_some();
    Ok(GenParams { opts, adapter, adapter_scale, vsa_gate, preset, sparsity_asked })
}

/// One `fl2va` keyframe request: which instant of the clip it anchors, and the
/// server-side path to the still image ([`encode_keyframes`] loads and
/// encodes it; `crate::pipeline::Anchor`/`KeyframeCondition`'s own doc for
/// what happens to it after that).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlKeyframeSpec {
    pub anchor: pipeline::Anchor,
    pub path: String,
}

/// Decode `fl2va`'s own `first_frame`/`keyframes`/`last_frame` params, in
/// timeline order. At least one still is required - checked here so a
/// request naming none is rejected before any weight is read, matching
/// [`gen_params_from`]'s own discipline. Whether the interior frames fit the
/// clip is [`pipeline::check_anchors`]' question once the clip is aligned.
pub fn fl2va_keyframes_from(inv: &capability::Invocation) -> Result<Vec<FlKeyframeSpec>, String> {
    let mut out = Vec::new();
    if let Some(path) = inv.get_str("first_frame").filter(|s| !s.is_empty()) {
        out.push(FlKeyframeSpec { anchor: pipeline::Anchor::First, path });
    }
    for pair in inv.get_str("keyframes").unwrap_or_default().split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (at, path) = pair.split_once(':').ok_or_else(|| format!("fl2va: keyframe '{pair}' is not FRAME:PATH"))?;
        let at = at.trim().parse::<u32>().map_err(|e| format!("fl2va: keyframe frame '{at}': {e}"))?;
        out.push(FlKeyframeSpec { anchor: pipeline::Anchor::At(at), path: path.trim().to_string() });
    }
    if let Some(path) = inv.get_str("last_frame").filter(|s| !s.is_empty()) {
        out.push(FlKeyframeSpec { anchor: pipeline::Anchor::Last, path });
    }
    if out.is_empty() {
        return Err("fl2va: at least one of first_frame/last_frame/keyframes is required".to_string());
    }
    Ok(out)
}

/// `fl2va`'s stills on the canvas, each with the instant of the clip it anchors.
pub type Keyframes = Vec<(pipeline::Anchor, imaging::Rgb8)>;

/// What a request conditions on besides its prompt, decoded from disk and
/// fitted to the request's geometry - no weight is read for it.
pub enum Conditioning {
    /// `t2va`: the prompt alone.
    Prompt,
    /// `fl2va`'s keyframes on the canvas, in presentation order.
    Keyframes(Keyframes),
    /// `ref2va`'s references on the model's own rates and resolutions.
    References(Vec<crate::references::NormalizedReference>),
}

/// One decoded request: its options (an `fl2va` canvas resolved from its
/// first keyframe), its prompt and its conditioning.
pub struct Request {
    pub params: GenParams,
    pub prompt: String,
    pub conditioning: Conditioning,
}

/// The canvas multiple of the released checkpoint: the video VAE's spatial
/// stride times the DiT's spatial patch.
fn canvas_multiple() -> u32 {
    let (dit_cfg, vae_cfg) = real_configs();
    vae_cfg.spatial_compression_ratio() * dit_cfg.patch_size[2]
}

/// Decode `action`'s request from `inv` and load the media it names,
/// before any weight is read: a request that cannot run is refused first.
pub fn request_from(action: &str, inv: &capability::Invocation, paths: &Paths) -> Result<Request, String> {
    let mut params = gen_params_from(inv)?;
    resolve_preset(&mut params, paths)?;
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let conditioning = match action {
        "t2va" => Conditioning::Prompt,
        "fl2va" => {
            let (canvas, keyframes) = load_keyframes(&fl2va_keyframes_from(inv)?, params.opts.canvas)?;
            params.opts.canvas = Some(canvas);
            Conditioning::Keyframes(keyframes)
        }
        "ref2va" => {
            let (_, vae_cfg) = real_configs();
            let aligned = pipeline::align_num_frames(params.opts.num_frames, vae_cfg.clip_length, vae_cfg.tokens_chunk_size())?;
            let media = references_from(inv)?;
            Conditioning::References(crate::references::normalize_references(&media, aligned, &crate::references::ReferenceCanvas::released(canvas_multiple()))?)
        }
        other => return Err(format!("minimaxh3: unknown action '{other}'")),
    };
    Ok(Request { params, prompt, conditioning })
}

/// Load `fl2va`'s stills onto the request's canvas (`MiniMaxH3ResizeStep`):
/// with none given, the canvas follows the first still's own aspect. The
/// first still is LANCZOS-stretched onto it; a second is cover-cropped -
/// scaled (rounding half to even) to cover it, then centred.
pub fn load_keyframes(specs: &[FlKeyframeSpec], canvas: Option<(u32, u32)>) -> Result<((u32, u32), Keyframes), String> {
    let stills = specs.iter().map(|s| imaging::load(&s.path).map(|img| (s.anchor, img)).map_err(|e| format!("fl2va: loading {}: {e}", s.path))).collect::<Result<Vec<_>, _>>()?;
    let first = &stills.first().ok_or("fl2va: at least one of first_frame/last_frame/keyframes is required")?.1;
    let (h, w) = match canvas {
        Some(c) => c,
        None => pipeline::resolve_canvas_size(first.w as f64, first.h as f64, canvas_multiple(), pipeline::CANVAS_SHORT_EDGE, pipeline::CANVAS_MAX_PIXELS)?,
    };
    let fitted = stills
        .into_iter()
        .enumerate()
        .map(|(i, (anchor, img))| {
            let img = if (img.w, img.h) == (w, h) {
                img
            } else if i == 0 {
                imaging::host::resize_lanczos_pil(&img, w as usize, h as usize)
            } else {
                let scale = (w as f64 / img.w as f64).max(h as f64 / img.h as f64);
                let rw = w.max((img.w as f64 * scale).round_ties_even() as u32);
                let rh = h.max((img.h as f64 * scale).round_ties_even() as u32);
                let covered = imaging::host::resize_lanczos_pil(&img, rw as usize, rh as usize);
                let (left, top) = (((rw - w) / 2) as usize, ((rh - h) / 2) as usize);
                let px = (0..h as usize).flat_map(|y| covered.px[((top + y) * rw as usize + left) * 3..][..w as usize * 3].iter().copied()).collect();
                imaging::Rgb8 { w, h, px }
            };
            (anchor, img)
        })
        .collect();
    Ok(((h, w), fitted))
}

/// `ref2va`'s `references`: comma-separated server-side paths, kinds read
/// off the extension - a WAV is a soundtrack, a PNG/JPEG/WebP an image,
/// anything else a video, whose own soundtrack is attached as
/// `video+track.wav` (brain decodes WAV, not a container's audio stream).
pub fn references_from(inv: &capability::Invocation) -> Result<Vec<crate::references::ReferenceMedia>, String> {
    use crate::references::{ReferenceMedia, Soundtrack};
    let list = inv.get_str("references").filter(|s| !s.trim().is_empty()).ok_or("ref2va: 'references' is required")?;
    let soundtrack = |path: &str| -> Result<Soundtrack, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("ref2va: reading {path}: {e}"))?;
        let (sample_rate, channels) = audio::wav::parse_planar(&bytes).map_err(|e| format!("ref2va: {path}: {e}"))?;
        Ok(Soundtrack { channels, sample_rate })
    };
    let extension = |path: &str| std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    list.split(',')
        .map(str::trim)
        .map(|entry| {
            let (path, track) = entry.split_once('+').map_or((entry, None), |(v, a)| (v, Some(a)));
            match extension(path).as_str() {
                "wav" if track.is_none() => Ok(ReferenceMedia::Audio(soundtrack(path)?)),
                "png" | "jpg" | "jpeg" | "webp" if track.is_none() => imaging::load(path).map(ReferenceMedia::Image).map_err(|e| format!("ref2va: loading {path}: {e}")),
                "wav" | "png" | "jpg" | "jpeg" | "webp" => Err(format!("ref2va: only a video takes an attached soundtrack, got '{entry}'")),
                _ => {
                    let frames = imaging::video::decode_frames_rgb8(std::path::Path::new(path), &imaging::video::VideoDecodeOpts { fps: None, max_frames: 0, spread: 0 })?;
                    let fps = imaging::video::probe_fps(std::path::Path::new(path)).ok_or_else(|| format!("ref2va: {path} reports no frame rate"))?;
                    Ok(ReferenceMedia::Video { frames, fps, soundtrack: track.map(soundtrack).transpose()? })
                }
            }
        })
        .collect()
}

/// The most packed rows `ref2va`'s references can take, before any is
/// decoded: an image at the size it normalizes to, read off its header (at
/// the widest 4:1 it is accepted at when the header cannot be read), a video
/// on a canvas no larger than the clip's own `target_video_rows`, and every
/// soundtrack no longer than the clip's `target_audio_rows`. Zero for every
/// other action. What a resident workspace is sized from.
pub fn reference_rows_bound(action: &str, inv: &capability::Invocation, target_video_rows: u32, target_audio_rows: u32) -> u32 {
    if action != "ref2va" {
        return 0;
    }
    let (dit_cfg, vae_cfg) = real_configs();
    let side = |edge: u32| edge / vae_cfg.spatial_compression_ratio() / dit_cfg.patch_size[1];
    let widest = side(pipeline::REFERENCE_IMAGE_SHORT_EDGE) * side(4 * pipeline::REFERENCE_IMAGE_SHORT_EDGE);
    let canvas = crate::references::ReferenceCanvas::released(canvas_multiple());
    let image_rows = |path: &str| {
        imaging::dimensions(path).ok().and_then(|(w, h)| crate::references::normalized_image_size(w, h, &canvas).ok()).map_or(widest, |(w, h)| side(w) * side(h))
    };
    let list = inv.get_str("references").unwrap_or_default();
    list.split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .map(|entry| {
            let (path, track) = entry.split_once('+').map_or((entry, None), |(v, a)| (v, Some(a)));
            let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
            match ext.as_str() {
                "wav" => target_audio_rows,
                "png" | "jpg" | "jpeg" | "webp" => image_rows(path),
                _ => target_video_rows + if track.is_some() { target_audio_rows } else { 0 },
            }
        })
        .sum()
}

// ===================== text conditioning =====================

/// The best per-layer-linear storage tier the AMBIENT device (`--device`/
/// `BRAIN_DEVICE`) can actually deliver, cheapest-real-cost first: int8 (if
/// `caps.numeric.int8_dot`), else bf16 (if `caps.numeric.bf16_storage` -
/// perf-number: frame-rate constant / memory ratio, not a measured timing
/// storage-only, decoded to fp32 inside the kernel, but still a genuine 2x
/// on bytes moved/resident), else fp32.
///
/// **Why this exists, not a hardcoded request**: `gpu_core::select::Dtype::
/// promote` silently demotes an unsupported tier back to fp32 -
/// `Weight::upload`'s own doc says as much ("read `Qwen::linear_dtype` for
/// what actually landed, never this request"), and the text tower once
/// asked for `Dtype::I8` and never checked. On the CPU JIT backend
/// `NumericSupport.int8_dot` is `false` (`backend-cpu`'s own doc: "the
/// multi-barrier packed-int8 GEMMs are outside the JIT's single-barrier
/// model, and there is no VNNI fast path yet") - so EVERY prior real-weight
/// run in this port's history asked for int8 and silently got fp32 instead,
/// which is the actual reason a 50-layer-truncated encoder (expected ~25GB
/// at int8) climbed to ~140GB (matching a ~100GB fp32 destination plus
/// transient overhead, not a leak - confirmed via `/proc/<pid>/smaps_rollup`
/// showing the growth as anonymous/private-dirty, not file-backed page
/// cache). `bf16_storage` IS `true` on the CPU backend (host RAM holds any
/// byte layout - see that struct's own comment), so this is a REAL,
/// non-lying 2x reduction there, not a second silent no-op.
pub fn best_linear_dtype() -> gpu_core::select::Dtype {
    use gpu_core::select::Dtype;
    let probe = gpu_core::Gpu::new(&[]);
    let n = probe.caps().numeric;
    if n.int8_dot {
        Dtype::I8
    } else if n.bf16_storage || n.bf16 {
        Dtype::BF16
    } else {
        Dtype::F32
    }
}

/// The conditioning a request's presentation gives: the hidden state after
/// [`pipeline::TEXT_ENCODER_LAYER`] layers of the Qwen3-VL conditioner, over
/// the prompt and whatever keyframes or references it carries
/// ([`crate::presentation`]). The language tower is built for this
/// presentation alone and dropped, placed over as many cards as its ~27 GB
/// of int8 linears need at [`best_linear_dtype`]'s tier; the vision tower
/// runs on `device`.
pub fn encode_presentation(paths: &Paths, prompt: &str, items: &[crate::presentation::Presented], device: Option<&str>) -> Result<pipeline::TextConditioning, String> {
    let encoder = crate::presentation::PresentationEncoder::open(&paths.text_encoder, &paths.tokenizer, paths.processor.as_deref(), pipeline::TEXT_ENCODER_LAYER, best_linear_dtype(), device)?;
    encoder.encode(prompt, items)
}

// ===================== outcome shaping =====================

/// Wrap one generated clip+soundtrack as an [`capability::Outcome`] - the
/// shared `capability::blob::video_blob`/`audio::wav::encode_multi` wire
/// format every other video/audio-producing model in this workspace uses
/// (`wan::caps::video_outcome`/`ltxv::caps::video_outcome`'s pattern). `seed`
/// is the one the clip was drawn with, so the outcome alone reproduces it.
pub fn av_outcome(av: &pipeline::GeneratedAv, seed: u64) -> capability::Outcome {
    let plane = (av.height * av.width) as usize;
    let t = av.num_video_frames as usize;
    let mut frames: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(t);
    for ti in 0..t {
        let mut hwc = vec![0f32; 3 * plane];
        for c in 0..3usize {
            let src = &av.video[(c * t + ti) * plane..(c * t + ti + 1) * plane];
            for (i, &v) in src.iter().enumerate() {
                hwc[i * 3 + c] = v;
            }
        }
        frames.push((hwc, av.width, av.height));
    }
    let mut blob = match capability::blob::video_blob(&frames) {
        Ok(b) => b,
        // Unreachable for a real generation (the buffer length already
        // matches w*h*3 per frame by construction above), so this reports
        // rather than panics inside a serving thread - `wan::caps::
        // video_outcome`'s own precedent.
        Err(e) => return capability::Outcome::new().set("error", json!(e)),
    };
    if let Some(m) = blob.meta.as_object_mut() {
        m.insert("fps".to_string(), json!(pipeline::FPS));
    }

    let planes: Vec<&[f32]> = av.audio.iter().map(Vec::as_slice).collect();
    let wav = audio::wav::encode_multi(&planes, av.sample_rate);
    let audio_blob = capability::Blob::new(Media::Audio, wav).with_meta(json!({"format": "wav", "sample_rate": av.sample_rate, "channels": av.audio.len()}));

    capability::Outcome::new()
        .set("frames", json!(av.num_video_frames))
        .set("width", json!(av.width))
        .set("height", json!(av.height))
        .set("fps", json!(pipeline::FPS))
        .set("audio_samples", json!(av.audio.first().map(Vec::len).unwrap_or(0)))
        .set("audio_sample_rate", json!(av.sample_rate))
        .set("seed", json!(seed))
        .blob("video", blob)
        .blob("audio", audio_blob)
}

// ===================== resident weight source =====================

/// Every component directory one MiniMax-H3 checkout needs. Weights resolve
/// through the model-store resolver (`crate::spec::H3Spec`, wired into
/// `crates/catalog`'s `ModelEntry`) via [`Paths::from_assembly`], or from
/// one checkout root a caller names via [`Paths::resolve`]. See [`Paths::dit`]'s own doc for [`crate::model::H3Transformer::
/// load`]'s real, tracked naming gap.
#[derive(Clone, Debug)]
pub struct Paths {
    /// The `transformer/` component directory - real-checkpoint tensor
    /// names there are not yet confirmed to agree with
    /// [`crate::model::H3Transformer::load`]'s own naming (`to_q.weight`
    /// etc., diffusers' `MiniMaxH3Attention` module names); a real-checkpoint
    /// DiT importer bridging the two, if a bridge turns out to be needed
    /// at all, is separate, not-yet-built work.
    pub dit: String,
    /// The `vae/` component directory `crate::import::import_video_vae`
    /// reads.
    pub video_vae: String,
    /// The `audio_vae/` component directory
    /// `crate::import::import_audio_vae_decoder` reads.
    pub vocoder: String,
    /// The `text_encoder/` component directory - `config.json` plus the
    /// sharded Qwen3-VL weights [`encode_prompt`] reads.
    pub text_encoder: String,
    /// The `tokenizer/` component directory - `tokenizer.json`/`vocab.json`/
    /// `merges.txt`, separate from `text_encoder/` in the root layout.
    pub tokenizer: String,
    /// The release's `processor/` directory - the Qwen3-VL image and video
    /// processor configs that present keyframes and references. `None`
    /// outside a release root, which can still present a prompt alone.
    pub processor: Option<String>,
    /// The release's `transformer_ref/` directory - `ref2va`'s separately
    /// trained DiT, same architecture as [`Self::dit`]. `None` when the
    /// checkout does not carry it, which refuses `ref2va` alone.
    pub dit_ref: Option<String>,
}

impl Paths {
    /// Every sub-path one checkout `root` implies - each a fixed,
    /// real-checkpoint-confirmed subdirectory name under it (this crate's
    /// roadmap ledger's "Checkpoint layout" section), not independently
    /// overridable per role the way `ltxv`'s recipe is (H3's fetch side,
    /// `modelstore::recipe::H3Recipe`, has no per-role granularity either).
    pub fn resolve(root: &str) -> Paths {
        let (processor, dit_ref) = Paths::release_extras(std::path::Path::new(root));
        Paths {
            dit: format!("{root}/transformer"),
            video_vae: format!("{root}/vae"),
            vocoder: format!("{root}/audio_vae"),
            text_encoder: format!("{root}/text_encoder"),
            tokenizer: format!("{root}/tokenizer"),
            processor,
            dit_ref,
        }
    }

    /// The release root's `processor/` and `transformer_ref/` directories,
    /// each when present. They are no role of their own: they only exist
    /// as part of a release, beside its `text_encoder/` and `transformer/`.
    fn release_extras(root: &std::path::Path) -> (Option<String>, Option<String>) {
        let dir = |name: &str| Some(root.join(name)).filter(|p| p.is_dir()).and_then(|p| p.to_str().map(str::to_string));
        (dir("processor"), dir("transformer_ref"))
    }

    /// Extract the five component directories from a resolved model-store
    /// [`capability::Assembly`] - the SAME five roles `loader::supply`'s
    /// `"minimaxh3"` convert arm writes into the compound manifest and
    /// [`crate::spec::H3Spec`] declares (`wan::pipeline::Paths::from_assembly`'s
    /// own pattern; role names match [`Paths`]'s field names, so the mapping
    /// is a straight rename-free read).
    pub fn from_assembly(assembly: &capability::Assembly) -> Result<Paths, String> {
        Paths::from_roles(&|role| Paths::component(assembly, role))
    }

    /// [`Paths`] from the five roles `role` resolves, whichever lane they
    /// came from (a store assembly, a served compound manifest). The
    /// `processor/` and `transformer_ref/` extras sit in the release root the
    /// text encoder came from, so they are found there rather than resolved.
    pub fn from_roles(role: &dyn Fn(&str) -> Result<String, String>) -> Result<Paths, String> {
        let text_encoder = role("text_encoder")?;
        let root = std::path::Path::new(&text_encoder).parent().filter(|p| crate::spec::is_h3_pipeline_root(p));
        let (processor, dit_ref) = root.map(Paths::release_extras).unwrap_or_default();
        Ok(Paths {
            dit: role("dit")?,
            video_vae: role("video_vae")?,
            vocoder: role("vocoder")?,
            text_encoder,
            tokenizer: role("tokenizer")?,
            processor,
            dit_ref,
        })
    }

    /// The DiT `action` runs: `ref2va`'s own `transformer_ref/`, every other
    /// action's `transformer/`.
    pub fn dit_for(&self, action: &str) -> Result<&str, String> {
        if action != "ref2va" {
            return Ok(&self.dit);
        }
        self.dit_ref.as_deref().ok_or_else(|| format!("minimaxh3 ref2va: runs on the release's transformer_ref/, which the checkout beside {} does not carry - fetch it with the rest of MiniMaxAI/MiniMax-H3", self.text_encoder))
    }

    /// The path one role of `assembly` names. A role satisfied by the
    /// released pipeline root (`crate::spec`'s `model_index.json` lane) names
    /// that root, and the component is its fixed subfolder.
    pub fn component(assembly: &capability::Assembly, role: &str) -> Result<String, String> {
        let path = assembly.roles.get(role).and_then(|p| p.to_str()).ok_or_else(|| format!("minimaxh3: assembly '{}' has no {role} role", assembly.id))?;
        if !crate::spec::is_h3_pipeline_root(std::path::Path::new(path)) {
            return Ok(path.to_string());
        }
        let p = Paths::resolve(path);
        Ok(match role {
            "dit" => p.dit,
            "video_vae" => p.video_vae,
            "vocoder" => p.vocoder,
            "text_encoder" => p.text_encoder,
            "tokenizer" => p.tokenizer,
            other => return Err(format!("minimaxh3: '{other}' is not a role of this architecture")),
        })
    }
}

/// Open a streaming reader over the DiT checkpoint at `path` - a bare file
/// or a sharded `<name>.safetensors.index.json` directory, no name
/// remapping (the real checkpoint's own tensor names already match
/// [`crate::model::H3Transformer::load`]'s naming - see [`Paths::dit`]'s own
/// doc). `pub` so `crates/catalog/src/resident_minimaxh3.rs` can open the DiT
/// reader ONCE (into its own resident [`crate::model::H3Transformer`])
/// without going through [`LoadedWeights::load`], which would also re-read
/// the VAEs.
///
/// A streaming [`checkpoint::weightio::WeightReader`], not the eager
/// `read_model_dir` this crate's VAE loaders still use: at ~33B params,
/// materializing this checkpoint's tensors into a whole-map host copy
/// before [`crate::model::H3Transformer::load`] uploads a single byte would
/// cost ~132GB on its own (bf16-on-disk decoded to fp32) - the exact mistake
/// `qwen3vl::Qwen3Vl::from_hf` used to make on the (much smaller) text
/// encoder, which measurably OOMed a 150GB container cap before it was
/// fixed the same way.
pub fn open_dit_reader(path: &str) -> Result<checkpoint::weightio::WeightReader, String> {
    checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(path)).map_err(|e| format!("minimaxh3: open DiT checkpoint {path}: {e}"))
}

/// The video/audio VAE weights [`pipeline::t2va`]/[`pipeline::fl2va`] need,
/// loaded once and held by the caller - the DiT is deliberately NOT part of
/// this struct. `wan::caps`'s own VAE deliberately never rides alongside its
/// hot DiT either; here the split additionally lets
/// `crates/catalog/src/resident_minimaxh3.rs` hold the (33B, expensive-to-reload)
/// DiT resident while still re-reading these (hundreds of MB, cheap) VAEs
/// per call - the same "not everything worth loading is worth CACHING"
/// judgment `resident_wan.rs`'s own module doc states for its own VAE/T5.
/// The per-channel VAE latent normalization is read from each VAE's own
/// `config.json` by [`read_latent_stats`] - see that function's doc for why a
/// missing file is an error here rather than a `mean=0`/`std=1` fallback.
pub struct VaeWeights {
    pub video_vae_tensors: vae::blocks::Tensors,
    pub video_vae_cfg: VideoVaeConfig,
    pub vocoder_tensors: vae::blocks::Tensors,
    pub vocoder_cfg: crate::vocoder::VocoderConfig,
    pub video_latents_mean: Vec<f32>,
    pub video_latents_std: Vec<f32>,
    pub audio_latents_mean: Vec<f32>,
    pub audio_latents_std: Vec<f32>,
}

/// Read a VAE component's own per-channel latent normalization out of its
/// `config.json` (`latents_mean`/`latents_std`), checking both arrays are
/// exactly `channels` long.
///
/// The reference applies these on both sides of the DiT - `encode_vae_condition`
/// normalizes with `(latents - mean) / std` and `MiniMaxH3VideoDecodeStep` /
/// `MiniMaxH3AudioDecodeStep` denormalize with `latents * std + mean` - so the
/// diffusion model works in a whitened latent space and the VAE decoder never
/// sees anything but the raw one.
///
/// **A missing or malformed `config.json` is an error, deliberately, and not a
/// `mean=0`/`std=1` fallback.** Those two values are the IDENTITY of that
/// affine transform, so falling back to them does not degrade the decode a
/// little - it deletes the denormalization step entirely and hands the decoder
/// a whitened latent it was never trained on, producing a wrong picture with
/// no error anywhere. A default that is indistinguishable from a correct load
/// but silently wrong is worse than a refusal, so this refuses. The real
/// values are per-channel and nowhere near the identity: the released video
/// VAE's means span -1.37..1.07 and its standard deviations 0.45..3.28.
pub fn read_latent_stats(dir: &str, channels: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let path = format!("{dir}/config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("minimaxh3: cannot read {path} for the VAE latent normalization: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("minimaxh3: cannot parse {path}: {e}"))?;
    let field = |name: &str| -> Result<Vec<f32>, String> {
        let array = json.get(name).and_then(|v| v.as_array()).ok_or_else(|| format!("minimaxh3: {path} has no `{name}` array - it cannot be the VAE config this port expects"))?;
        let values: Vec<f32> = array.iter().map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| format!("minimaxh3: {path}: `{name}` holds a non-numeric entry"))).collect::<Result<_, _>>()?;
        if values.len() != channels {
            return Err(format!("minimaxh3: {path}: `{name}` has {} entries, expected {channels} (one per latent channel)", values.len()));
        }
        Ok(values)
    };
    // Not named `std`: that would shadow the crate root for the rest of this
    // function.
    let stds = field("latents_std")?;
    if let Some(bad) = stds.iter().position(|&s| !(s.is_finite() && s > 0.0)) {
        return Err(format!("minimaxh3: {path}: `latents_std[{bad}]` is {}, which cannot divide the encode side", stds[bad]));
    }
    let means = field("latents_mean")?;
    if let Some(bad) = means.iter().position(|m| !m.is_finite()) {
        return Err(format!("minimaxh3: {path}: `latents_mean[{bad}]` is {}", means[bad]));
    }
    Ok((means, stds))
}

impl VaeWeights {
    /// Load both VAEs from `paths`, at the real config numbers - two-way
    /// import coverage via `crate::import`'s own functions - together with
    /// each one's per-channel latent normalization ([`read_latent_stats`]).
    pub fn load(paths: &Paths) -> Result<VaeWeights, String> {
        let video_vae_cfg = VideoVaeConfig::real();
        let vocoder_cfg = crate::vocoder::VocoderConfig::h3_32khz();
        let video_vae_tensors = crate::import::import_video_vae(&paths.video_vae, &video_vae_cfg)?;
        let vocoder_tensors = crate::import::import_audio_vae_decoder(&paths.vocoder, &vocoder_cfg)?;
        let (video_latents_mean, video_latents_std) = read_latent_stats(&paths.video_vae, video_vae_cfg.latent_channels as usize)?;
        let (audio_latents_mean, audio_latents_std) = read_latent_stats(&paths.vocoder, vocoder_cfg.vae_latent_channels as usize)?;
        Ok(VaeWeights {
            video_latents_mean,
            video_latents_std,
            audio_latents_mean,
            audio_latents_std,
            video_vae_tensors,
            video_vae_cfg,
            vocoder_tensors,
            vocoder_cfg,
        })
    }

    /// A borrowing [`pipeline::H3Checkpoint`] over `self` plus a
    /// caller-supplied `dit_tensors`/`dit_cfg` - the DiT half lives outside
    /// this struct (see its own doc), so callers that already hold a
    /// resident [`crate::model::H3Transformer`] (which never reads
    /// `ckpt.dit_tensors` - see [`pipeline::t2va_hot`]'s doc) can pass an
    /// empty placeholder map rather than keeping a real one around just to
    /// satisfy this field.
    pub fn as_checkpoint<'a>(&'a self, dit_tensors: &'a dyn checkpoint::TensorSource, dit_cfg: H3TransformerConfig, (dit_lora, dit_gates): DitExtras<'a>) -> pipeline::H3Checkpoint<'a> {
        pipeline::H3Checkpoint {
            dit_tensors,
            dit_lora,
            dit_gates,
            dit_cfg,
            video_vae_tensors: &self.video_vae_tensors,
            video_vae_cfg: self.video_vae_cfg.clone(),
            vocoder_tensors: &self.vocoder_tensors,
            vocoder_cfg: self.vocoder_cfg.clone(),
            video_latents_mean: self.video_latents_mean.clone(),
            video_latents_std: self.video_latents_std.clone(),
            audio_latents_mean: self.audio_latents_mean.clone(),
            audio_latents_std: self.audio_latents_std.clone(),
        }
    }
}

/// Every weight [`pipeline::t2va`]/[`pipeline::fl2va`] need, loaded once and
/// held by the caller - the owning counterpart of [`pipeline::H3Checkpoint`],
/// which only borrows. The DIRECT (non-resident) [`MiniMaxH3Provider`]'s own
/// per-call bundle; the residency-scheduled path
/// (`crates/catalog/src/resident_minimaxh3.rs`) uses [`VaeWeights`] plus its
/// own resident DiT instead, so the (33B) DiT is not re-read from disk per
/// call.
pub struct LoadedWeights {
    /// A streaming reader over the DiT's own (~33B-param) checkpoint - never
    /// a whole-map host copy. See [`open_dit_reader`]'s own doc for why this
    /// is not [`read_tensors`], which is still the right (eager) call for
    /// the VAEs (hundreds of MB, not a memory risk).
    pub dit_reader: checkpoint::weightio::WeightReader,
    pub dit_cfg: H3TransformerConfig,
    pub vae: VaeWeights,
}

impl LoadedWeights {
    /// Load every weight from `paths` - a streaming reader for the DiT at
    /// `dit` ([`Paths::dit_for`]; see [`open_dit_reader`]'s own doc) plus
    /// [`VaeWeights::load`].
    pub fn load(paths: &Paths, dit: &str) -> Result<LoadedWeights, String> {
        Ok(LoadedWeights { dit_reader: open_dit_reader(dit)?, dit_cfg: H3TransformerConfig::real(), vae: VaeWeights::load(paths)? })
    }

    /// A borrowing [`pipeline::H3Checkpoint`] over these tensors - built fresh
    /// per call since [`pipeline::H3Checkpoint`] borrows rather than owns.
    pub fn as_checkpoint<'a>(&'a self, extras: DitExtras<'a>) -> pipeline::H3Checkpoint<'a> {
        self.vae.as_checkpoint(&self.dit_reader, self.dit_cfg, extras)
    }
}

/// What a generation runs its DiT with besides the checkpoint: the adapter
/// folded into its weights and every block's VSA gate.
pub type DitExtras<'a> = (Option<&'a model::lora::RuntimeLora>, Option<&'a dyn checkpoint::TensorSource>);

/// The VSA gate file at `path`, every block's gate checked before a weight
/// is read.
pub fn open_vsa_gates(path: &str, cfg: &H3TransformerConfig) -> Result<checkpoint::mmap::MmapSafetensors, String> {
    let gates = checkpoint::mmap::MmapSafetensors::open(path).map_err(|e| format!("minimaxh3: vsa_gate {path}: {e}"))?;
    crate::vsa::check_gates(&gates, cfg)?;
    Ok(gates)
}

// ===================== execution =====================

/// Generate `req` and shape its outcome - the one entry point the direct
/// provider below and the residency-scheduled adapter
/// (`crates/catalog/src/resident_minimaxh3.rs`) both funnel through
/// ([`run_request`]), so the presentation, the conditioning and the outcome
/// cannot drift between them. The text is encoded first and its tower
/// dropped before the DiT loads, so the two never share the cards.
/// `cancel` is polled every denoise step and `progress` reports each step
/// and decode.
pub fn generate_on(ckpt: &pipeline::H3Checkpoint, req: &Request, paths: &Paths, cancel: &capability::CancelToken, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
    use crate::presentation::Presented;
    let opts = &req.params.opts;
    let items: Vec<Presented> = match &req.conditioning {
        Conditioning::Prompt => Vec::new(),
        Conditioning::Keyframes(keyframes) => keyframes.iter().map(|(_, img)| Presented::Image(img)).collect(),
        Conditioning::References(references) => references.iter().map(Presented::of_reference).collect(),
    };
    // Under way from the start: the text encode alone can take many minutes.
    progress(Progress::step(0, opts.denoise_steps as u32 + 2, "encode text".to_string()));
    let text = encode_presentation(paths, &req.prompt, &items, opts.device.as_deref())?;
    let report = |done: u32, total: u32, phase: &str| progress(Progress::step(done, total, phase.to_string()));
    let av = match &req.conditioning {
        Conditioning::Prompt => pipeline::t2va(ckpt, &text, opts, cancel, report)?,
        Conditioning::Keyframes(keyframes) => pipeline::fl2va(ckpt, &text, &encode_keyframes(ckpt, keyframes, opts), opts, cancel, report)?,
        Conditioning::References(references) => {
            let encoded = encode_references(ckpt, paths, references, opts)?;
            pipeline::ref2va(ckpt, &text, &encoded, opts, cancel, report)?
        }
    };
    Ok(av_outcome(&av, opts.seed))
}

/// Run `action` for `inv` on `device` (`None`: every schedulable card): the
/// request and its media checked, the adapter and gates resolved and checked,
/// all before any weight is read - a request that could never run must not
/// cost a checkpoint load to refuse (`wan::caps::WanAction::run`'s
/// ordering) - then [`generate_on`].
pub fn run_request(paths: &Paths, action: &str, inv: &Invocation, device: Option<&str>, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let mut req = request_from(action, inv, paths)?;
    req.params.opts.device = device.map(str::to_string);
    let cfg = H3TransformerConfig::real();
    let lora = req.params.adapter.as_ref().map(|a| crate::lora::runtime_adapter(a, &cfg)).transpose()?;
    let gates = req.params.vsa_gate.as_deref().map(|p| open_vsa_gates(p, &cfg)).transpose()?;
    let weights = LoadedWeights::load(paths, paths.dit_for(action)?)?;
    let extras = (lora.as_ref(), gates.as_ref().map(|g| g as &dyn checkpoint::TensorSource));
    generate_on(&weights.as_checkpoint(extras), &req, paths, &inv.cancel, progress)
}

/// VAE-encode `fl2va`'s fitted keyframes into the conditioning rows the
/// pipeline packs.
pub fn encode_keyframes(ckpt: &pipeline::H3Checkpoint, keyframes: &[(pipeline::Anchor, imaging::Rgb8)], opts: &pipeline::GenOpts) -> Vec<pipeline::KeyframeCondition> {
    keyframes
        .iter()
        .map(|(anchor, img)| {
            let pixels = crate::references::planar_pixels(&[img]);
            pipeline::encode_keyframe_condition(&ckpt.video_vae_cfg, ckpt.video_vae_tensors, opts.device.as_deref(), &pixels, img.h, img.w, &ckpt.video_latents_mean, &ckpt.video_latents_std, ckpt.dit_cfg.patch_size, *anchor, opts.seed)
        })
        .collect()
}

/// Encode `ref2va`'s references: frames through the checkpoint's video VAE,
/// soundtracks through the audio VAE's encoder half, read from the release's
/// `audio_vae/` for this request.
fn encode_references(ckpt: &pipeline::H3Checkpoint, paths: &Paths, references: &[crate::references::NormalizedReference], opts: &pipeline::GenOpts) -> Result<Vec<pipeline::EncodedReference>, String> {
    let audio_cfg = crate::audio_encoder::AudioEncoderConfig::h3();
    let audio_tensors = if references.iter().any(|r| !matches!(r, crate::references::NormalizedReference::Image(_) | crate::references::NormalizedReference::Video { soundtrack: None, .. })) {
        crate::audio_encoder::import_audio_vae_encoder(&paths.vocoder, &audio_cfg)?
    } else {
        Default::default()
    };
    let visual = pipeline::VisualConditionRequest {
        video_vae_cfg: &ckpt.video_vae_cfg,
        video_vae_tensors: ckpt.video_vae_tensors,
        device: opts.device.as_deref(),
        video_latents_mean: &ckpt.video_latents_mean,
        video_latents_std: &ckpt.video_latents_std,
        patch_size: ckpt.dit_cfg.patch_size,
        seed: opts.seed,
    };
    let audio = crate::references::SoundtrackEncoder { cfg: &audio_cfg, tensors: &audio_tensors, latents_mean: &ckpt.audio_latents_mean, latents_std: &ckpt.audio_latents_std, device: opts.device.as_deref() };
    Ok(crate::references::encode_references(references, &visual, &audio, ckpt.video_vae_cfg.clip_length, ckpt.video_vae_cfg.tokens_chunk_size()))
}

// ===================== execution (direct provider) =====================

use std::sync::Arc;

use capability::{Action, ActionResult, Invocation, Progress, Provider};

/// The executable MiniMax-H3 model behind the manifest, for the DIRECT
/// (non-resident) serving path - `brain do brain/minimaxh3 t2va ...` and
/// `crates/catalog`'s `ModelEntry`, whose provider closure resolves the
/// weight paths through the model store via [`Paths::from_assembly`]
/// (`wan`/`s3dit`'s own provider shape). The residency-scheduled adapter is
/// `crates/catalog/src/resident_minimaxh3.rs`'s `MiniMaxH3Resident` -
/// `wan`'s own direct-provider/resident-adapter split. Loads fresh weights
/// every call (see [`run_hot_dit`]'s doc) - no caching here, since caching
/// across calls is exactly what the residency-scheduled path exists to do
/// instead.
pub struct MiniMaxH3Provider {
    paths: Paths,
}

impl MiniMaxH3Provider {
    /// The resolved-assembly constructor: every weight path comes from the
    /// model store, nothing from the environment.
    pub fn from_paths(paths: Paths) -> MiniMaxH3Provider {
        MiniMaxH3Provider { paths }
    }
}

impl Provider for MiniMaxH3Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest()
            .actions
            .iter()
            .any(|a| a.name == name)
            .then(|| Arc::new(MiniMaxH3Action { name: name.to_string(), paths: self.paths.clone() }) as Arc<dyn Action>)
    }
}

struct MiniMaxH3Action {
    name: String,
    paths: Paths,
}

impl Action for MiniMaxH3Action {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        run_request(&self.paths, &self.name, inv, None, progress)
    }
}

/// The model store's MiniMax-H3 checkout root, for the real-weight tests -
/// `None` when the store holds none.
#[cfg(test)]
pub(crate) fn store_root() -> Option<String> {
    brain_testutil::model_dir("MiniMaxAI/MiniMax-H3").filter(|root| std::path::Path::new(root).is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generation conditions on the real conditioner or not at all: a
    /// checkout without it is refused, never handed placeholder embeddings.
    #[test]
    fn a_missing_text_encoder_is_refused() {
        let paths = Paths::resolve("/nonexistent-minimaxh3-root");
        assert!(encode_presentation(&paths, "a red fox running through snow", &[], Some("cpu")).is_err());
    }

    #[test]
    fn gated_unless_opted_in() {
        let err = check_license(false).unwrap_err();
        assert!(err.contains("--allow-community") && err.contains("allow_community"), "error must name the opt-in on both surfaces: {err}");
        assert!(err.contains("Community License"), "error must name the license: {err}");
        assert!(check_license(true).is_ok());
    }

    /// `gen_params_from` itself must be license-gated - every request needs
    /// the gate here (unlike `flux2::caps`, there is no free variant), so a
    /// caller cannot skip it by going through the caps-layer decoder instead
    /// of `check_license` directly. The opt-in travels with the request.
    #[test]
    fn gen_params_from_is_license_gated() {
        let inv = Invocation::new().set("prompt", json!("a cat"));
        let err = gen_params_from(&inv).unwrap_err();
        assert!(err.contains("allow_community"), "{err}");
        let declined = Invocation::new().set("prompt", json!("a cat")).set("allow_community", json!(false));
        assert!(gen_params_from(&declined).is_err());
        let accepted = Invocation::new().set("prompt", json!("a cat")).set("allow_community", json!(true));
        gen_params_from(&accepted).unwrap_or_else(|e| panic!("an opted-in request must pass the gate: {e}"));
        let opt_in = manifest().actions.iter().map(|a| a.params.iter().find(|p| p.name == "allow_community").map(|p| p.default.clone())).collect::<Vec<_>>();
        assert!(opt_in.iter().all(|d| d == &Some(Some(json!(false)))), "every action declares allow_community, off by default: {opt_in:?}");
    }

    /// The activation precision is the request's: one int8 level unless it
    /// asks for two, and nothing else is accepted.
    /// The DiT is held resident unless a request asks for it streamed, and
    /// a placement it does not know is refused, never read as either.
    #[test]
    fn a_request_chooses_where_its_dit_lives() {
        let inv = Invocation::new().set("prompt", json!("a"));
        let placement = |v: Option<&str>| decode_gen_params(&v.map_or(inv.clone(), |v| inv.clone().set("dit_placement", json!(v)))).map(|p| p.opts.dit_placement);
        assert_eq!(placement(None), Ok(pipeline::DitPlacement::Resident));
        assert_eq!(placement(Some("streamed")), Ok(pipeline::DitPlacement::Streamed));
        let err = placement(Some("auto")).unwrap_err();
        assert!(err.contains("dit_placement"), "{err}");
    }

    #[test]
    fn a_request_chooses_its_int8_activation_levels() {
        let levels = |v: Option<i64>| {
            let inv = Invocation::new().set("prompt", json!("a cat"));
            decode_gen_params(&v.map_or(inv.clone(), |v| inv.set("int8_activation_levels", json!(v)))).map(|p| p.opts.act_levels)
        };
        assert_eq!(levels(None), Ok(crate::block::ActLevels::One));
        assert_eq!(levels(Some(1)), Ok(crate::block::ActLevels::One));
        assert_eq!(levels(Some(2)), Ok(crate::block::ActLevels::Two));
        let err = levels(Some(3)).unwrap_err();
        assert!(err.contains("int8_activation_levels"), "{err}");
    }

    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, TASKS);

        let t2va = &m.actions[0];
        assert!(t2va.streaming, "a multi-step joint denoise run without progress reads as a hang");
        let required: Vec<&str> = t2va.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        let def = |a: &ActionSpec, name: &str| a.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no param {name}")).default.clone();
        assert_eq!(def(t2va, "num_frames"), Some(json!(default_num_frames())));
        assert_eq!(def(t2va, "steps"), Some(json!(20)));
        assert_eq!(def(t2va, "seed"), None, "an omitted seed is drawn, not defaulted");
        // Both outputs always present: MiniMax-H3 is natively audio-visual,
        // never a video-only or optional-audio model like `ltxv`.
        assert_eq!(t2va.outputs.len(), 2);
        assert_eq!(t2va.outputs[0].name, "video");
        assert_eq!(t2va.outputs[0].media, Media::Video);
        assert!(t2va.outputs[0].required);
        assert_eq!(t2va.outputs[1].name, "audio");
        assert_eq!(t2va.outputs[1].media, Media::Audio);
        assert!(t2va.outputs[1].required, "the soundtrack is never optional for this model");
        assert!(t2va.inputs.is_empty(), "t2va takes no binary input");

        // `fl2va`: same shared surface plus its own keyframe params, neither
        // of which is required on its own (the pair is - see
        // `fl2va_keyframes_from`'s own test below).
        let fl2va = &m.actions[1];
        assert!(fl2va.streaming);
        let required: Vec<&str> = fl2va.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt"]);
        assert!(fl2va.params.iter().any(|p| p.name == "first_frame" && !p.required));
        assert!(fl2va.params.iter().any(|p| p.name == "last_frame" && !p.required));
        assert_eq!(fl2va.outputs.len(), 2);

        let ref2va = &m.actions[2];
        let required: Vec<&str> = ref2va.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["prompt", "references"]);
        assert_eq!(ref2va.outputs.len(), 2);

        // The whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["model"], MODEL);
        assert_eq!(j["actions"].as_array().unwrap().len(), 3);
        assert_eq!(j["actions"][0]["streaming"], true);
        assert_eq!(j["actions"][0]["params"][0]["name"], "prompt");
        assert_eq!(j["actions"][0]["params"][0]["required"], true);
    }

    /// `fl2va`'s stills reach the canvas as the reference fits them: with no
    /// canvas asked for it follows the first still's aspect, the first
    /// still is stretched onto it and a second one cover-cropped - a
    /// one-pixel-wide red column centred in a 3:1 still survives a crop to
    /// the canvas's 16:9.
    #[test]
    fn keyframes_are_fitted_to_the_canvas_as_the_reference_fits_them() {
        let dir = std::env::temp_dir().join(format!("minimaxh3-keyframes-{}-{}", std::process::id(), data::rng::random_seed()));
        std::fs::create_dir_all(&dir).unwrap();
        let still = |name: &str, w: u32, h: u32, px: Vec<u8>| {
            let path = dir.join(name);
            imaging::save_png(&path, &imaging::Rgb8 { w, h, px }).unwrap();
            path.to_string_lossy().into_owned()
        };
        let first = still("first.png", 1600, 900, vec![90; 1600 * 900 * 3]);
        let last = still("last.png", 2700, 900, (0..2700 * 900).flat_map(|i| if i % 2700 == 1350 { [255, 0, 0] } else { [0, 0, 0] }).collect());
        let specs = [FlKeyframeSpec { anchor: pipeline::Anchor::First, path: first }, FlKeyframeSpec { anchor: pipeline::Anchor::Last, path: last }];

        let ((h, w), fitted) = load_keyframes(&specs, None).unwrap();
        assert_eq!((h, w), pipeline::resolve_canvas_size(16.0, 9.0, canvas_multiple(), pipeline::CANVAS_SHORT_EDGE, pipeline::CANVAS_MAX_PIXELS).unwrap());
        assert!(fitted.iter().all(|(_, img)| (img.h, img.w) == (h, w)));
        assert_eq!(fitted[0].0, pipeline::Anchor::First);
        let crop = &fitted[1].1;
        let red_column = (0..w as usize).max_by_key(|&x| crop.px[x * 3]).unwrap();
        assert!((red_column as i64 - w as i64 / 2).abs() <= 1, "the cover crop is centred (red column at {red_column} of {w})");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `ref2va` references are typed by their extension, a soundtrack rides
    /// on a video only, and the action refuses a checkout without
    /// `transformer_ref/` before reading any weight.
    #[test]
    fn references_are_read_by_kind_and_ref2va_needs_its_own_transformer() {
        let dir = std::env::temp_dir().join(format!("minimaxh3-refs-{}-{}", std::process::id(), data::rng::random_seed()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("a.png");
        imaging::save_png(&png, &imaging::Rgb8 { w: 4, h: 4, px: vec![7; 48] }).unwrap();
        let wav = dir.join("b.wav");
        audio::wav::write(&wav, &[0.1; 3200], 32000).unwrap();
        let refs = |list: String| references_from(&Invocation::new().set("references", json!(list)));
        let media = refs(format!("{},{}", png.display(), wav.display())).unwrap();
        assert!(matches!(media[0], crate::references::ReferenceMedia::Image(_)));
        assert!(matches!(media[1], crate::references::ReferenceMedia::Audio(_)));
        assert!(refs(format!("{}+{}", png.display(), wav.display())).unwrap_err().contains("only a video"));
        assert!(refs(String::new()).unwrap_err().contains("required"));
        let paths = Paths::resolve(dir.to_str().unwrap());
        assert!(paths.dit_for("ref2va").unwrap_err().contains("transformer_ref"));
        assert_eq!(paths.dit_for("t2va").unwrap(), paths.dit);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A served ref2va request picks a preset by name; each resolves to
    /// pinned files of model-store repos beside the release it is served
    /// from, never to a path the caller names. A preset the store lacks is
    /// refused naming the repo to pull, and a preset beside a host-side
    /// adapter or gate is ambiguous and refused.
    #[test]
    fn a_preset_resolves_to_pinned_store_files_beside_the_release() {
        let store = std::env::temp_dir().join(format!("brain-minimaxh3-preset-{}", std::process::id()));
        let release = store.join("MiniMaxAI/MiniMax-H3");
        std::fs::create_dir_all(release.join("text_encoder")).unwrap();
        std::fs::write(release.join("model_index.json"), r#"{"_class_name": "MiniMaxH3ModularPipeline"}"#).unwrap();
        let paths = Paths::resolve(release.to_str().unwrap());
        let decoded = |preset: &str, extra: &[(&str, serde_json::Value)]| {
            let mut inv = Invocation::new().set("prompt", json!("x")).set("preset", json!(preset));
            for (k, v) in extra {
                inv = inv.set(k, v.clone());
            }
            let mut p = decode_gen_params(&inv)?;
            resolve_preset(&mut p, &paths)?;
            Ok::<GenParams, String>(p)
        };

        let base = decoded("base", &[]).expect("base needs nothing");
        assert!(base.adapter.is_none() && base.vsa_gate.is_none() && base.opts.vsa_sparsity == 0.0);
        let e = decoded("turbo-4step", &[]).expect_err("not in the store");
        assert!(e.contains("lightx2v/Minimax-h3-Turbo"), "{e}");

        for (vendor, repo, file) in [PRESET_ADAPTER, PRESET_GATES] {
            let dir = store.join(vendor).join(repo);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(file), b"").unwrap();
        }
        let turbo = decoded("turbo-4step", &[]).expect("resolves");
        assert!(turbo.adapter.as_ref().unwrap().path.ends_with(PRESET_ADAPTER.2) && turbo.vsa_gate.is_none());
        let vsa = decoded("turbo-4step-vsa", &[]).expect("resolves");
        assert!(vsa.vsa_gate.as_ref().unwrap().ends_with(PRESET_GATES.2));
        assert_eq!(vsa.opts.vsa_sparsity, PRESET_VSA_SPARSITY, "the preset's own sparsity unless one is asked");
        assert_eq!(decoded("turbo-4step-vsa", &[("vsa_sparsity", json!(0.5))]).unwrap().opts.vsa_sparsity, 0.5);
        let e = decoded("turbo-4step", &[("adapter", json!("/elsewhere.safetensors"))]).expect_err("ambiguous");
        assert!(e.contains("preset"), "{e}");
        std::fs::remove_dir_all(&store).ok();
    }

    /// A reference image the bound can read reserves the rows its
    /// normalized size packs - a square one 64 x 64 - and one it cannot read
    /// the widest an image is accepted at.
    #[test]
    fn a_readable_reference_image_reserves_its_own_rows() {
        let dir = std::env::temp_dir().join(format!("brain-minimaxh3-refrows-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let square = dir.join("square.png");
        imaging::save_png(&square, &imaging::Rgb8::new(768, 768, vec![128; 768 * 768 * 3]).unwrap()).unwrap();
        let bound = |list: String| reference_rows_bound("ref2va", &Invocation::new().set("references", json!(list)), 0, 0);
        assert_eq!(bound(square.to_str().unwrap().to_string()), 64 * 64);
        assert_eq!(bound(dir.join("absent.png").to_str().unwrap().to_string()), 64 * 256, "the widest 4:1 image");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The manifest's own advertised defaults must survive `validate` ->
    /// `decode_gen_params` unchanged - the join the two halves can drift at.
    #[test]
    fn the_advertised_defaults_decode() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let inv = spec.validate(Invocation::new().set("prompt", json!("a cat"))).unwrap();
        let p = decode_gen_params(&inv).unwrap();
        assert_eq!(p.opts.num_frames, default_num_frames());
        assert_eq!(p.opts.denoise_steps, 20);
        assert_eq!(p.opts.canvas, None);
        assert_eq!(p.opts.device, None, "placement is the server's decision, not an invocation param");
    }

    /// An unseeded request draws its own seed (a different one each time), a
    /// given seed is kept, and a seed that is not a non-negative integer is
    /// refused instead of being clamped to another one.
    #[test]
    fn an_unseeded_request_draws_a_seed_and_a_malformed_one_is_refused() {
        let seed_of = |inv: Invocation| decode_gen_params(&inv).map(|p| p.opts.seed);
        assert_ne!(seed_of(Invocation::new()), seed_of(Invocation::new()), "two unseeded requests must not share a seed");
        assert_eq!(seed_of(Invocation::new().set("seed", json!(42))), Ok(42));
        assert!(seed_of(Invocation::new().set("seed", json!(-1))).is_err());
    }

    /// Every geometric/duration constraint is checked from the params alone,
    /// so a request that could never run never costs a checkpoint load to
    /// reject - `wan::caps`'s own identically-named test.
    #[test]
    fn impossible_geometry_is_rejected_before_any_weight_is_read() {
        let base = || Invocation::new().set("prompt", json!("x"));
        // width without height.
        let e = decode_gen_params(&base().set("width", json!(768))).unwrap_err();
        assert!(e.contains("together"), "{e}");
        // A size the (VAE stride x patch) grid cannot tile.
        let e = decode_gen_params(&base().set("width", json!(100)).set("height", json!(100))).unwrap_err();
        assert!(e.contains("multiple of"), "{e}");
        // A frame count whose aligned duration falls outside 5-15s.
        let e = decode_gen_params(&base().set("num_frames", json!(1))).unwrap_err();
        assert!(e.contains("5s") && e.contains("15s"), "{e}");
        // A request that IS representable decodes cleanly.
        let (_, vae_cfg) = real_configs();
        let m = vae_cfg.spatial_compression_ratio() * H3TransformerConfig::real().patch_size[2];
        let p = decode_gen_params(&base().set("width", json!(m)).set("height", json!(m)).set("steps", json!(2))).unwrap();
        assert_eq!(p.opts.canvas, Some((m, m)));
        assert_eq!(p.opts.denoise_steps, 2);
        // `steps` counts forwards; none at all is refused, not clamped.
        let e = decode_gen_params(&base().set("steps", json!(0))).unwrap_err();
        assert!(e.contains("steps"), "{e}");
    }

    #[test]
    fn fl2va_keyframes_from_requires_at_least_one_anchor() {
        let e = fl2va_keyframes_from(&Invocation::new()).unwrap_err();
        assert!(e.contains("at least one"), "{e}");

        let first = fl2va_keyframes_from(&Invocation::new().set("first_frame", json!("keyframe_a.png"))).unwrap();
        assert_eq!(first, vec![FlKeyframeSpec { anchor: pipeline::Anchor::First, path: "keyframe_a.png".to_string() }]);

        let both = fl2va_keyframes_from(&Invocation::new().set("first_frame", json!("keyframe_a.png")).set("last_frame", json!("keyframe_b.png"))).unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].anchor, pipeline::Anchor::First);
        assert_eq!(both[1].anchor, pipeline::Anchor::Last);

        // An explicitly empty path is treated as absent, same as every other
        // `filter(|s| !s.is_empty())` optional-string param in this workspace.
        let e = fl2va_keyframes_from(&Invocation::new().set("first_frame", json!(""))).unwrap_err();
        assert!(e.contains("at least one"), "{e}");

        // Interior keyframes sit between the two ends, in the order given.
        let all = fl2va_keyframes_from(&Invocation::new().set("first_frame", json!("a.png")).set("keyframes", json!("40:b.png, 80:c.png")).set("last_frame", json!("d.png"))).unwrap();
        let anchors: Vec<_> = all.iter().map(|k| (k.anchor, k.path.as_str())).collect();
        assert_eq!(anchors, vec![(pipeline::Anchor::First, "a.png"), (pipeline::Anchor::At(40), "b.png"), (pipeline::Anchor::At(80), "c.png"), (pipeline::Anchor::Last, "d.png")]);
        let alone = fl2va_keyframes_from(&Invocation::new().set("keyframes", json!("60:b.png"))).unwrap();
        assert_eq!(alone, vec![FlKeyframeSpec { anchor: pipeline::Anchor::At(60), path: "b.png".to_string() }]);
        for bad in ["b.png", "x:b.png", "-3:b.png"] {
            assert!(fl2va_keyframes_from(&Invocation::new().set("keyframes", json!(bad))).is_err(), "{bad} must be refused");
        }
    }

    /// The real Qwen3-VL conditioner, loaded from the model store's real
    /// checkout, must produce real, finite, non-trivial conditioning for a
    /// prompt alone. This is a genuinely large load (tens of GB); skips
    /// cleanly, like every other real-weight test in this crate, when the
    /// checkout is not present.
    #[test]
    #[ignore = "needs the real MiniMax-H3 checkout in the model store"]
    fn a_prompt_presents_through_the_real_encoder_when_weights_are_present() {
        let Some(root) = crate::caps::store_root() else {
            brain_testutil::skip("no MiniMax-H3 checkout in the model store to encode from");
            return;
        };
        let paths = Paths::resolve(&root);
        if !checkpoint::safetensors::has_model_weights(std::path::Path::new(&paths.text_encoder)) {
            brain_testutil::skip(&format!("{} has no text encoder weights - not (yet) downloaded", paths.text_encoder));
            return;
        }
        let text = encode_presentation(&paths, "a red fox running through snow at dusk", &[], Some("cpu")).unwrap_or_else(|e| panic!("real text conditioning: {e}"));
        assert_eq!(text.embeds.len() % 5120, 0, "embeds must be a whole number of text_dim=5120 rows");
        assert_eq!(text.token_tags.len(), text.embeds.len() / 5120);
        assert!(text.token_tags.iter().all(|&t| t == crate::config::TAG_TEXT), "a plain t2va prompt has no vision rows");
        assert!(text.embeds.iter().all(|v| v.is_finite()), "real encoder output must be finite");
        let max_abs = text.embeds.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_abs > 1.0, "a hidden state at this depth has a large spread (max_abs={max_abs})");
    }

    /// The VAE latent normalization must come from the checkpoint, and must
    /// not be the identity.
    ///
    /// This is the regression gate for a real defect: [`VaeWeights::load`]
    /// used to hand `pipeline::generate` `mean=0`/`std=1` placeholders, which
    /// are the IDENTITY of the affine transform the reference applies, so the
    /// video VAE decoded a whitened latent it was never trained on and every
    /// generated frame came out wrong - with nothing failing anywhere. The
    /// two assertions below are the two halves of that bug: that the values
    /// are read at all, and that they are not the identity that made the read
    /// look unnecessary.
    #[test]
    #[ignore = "needs the real MiniMax-H3 checkout in the model store"]
    fn vae_latent_normalization_is_read_from_the_checkpoint_and_is_not_the_identity() {
        let Some(root) = crate::caps::store_root() else {
            brain_testutil::skip("no MiniMax-H3 checkout in the model store to read VAE latent stats from");
            return;
        };
        let paths = Paths::resolve(&root);
        if !std::path::Path::new(&paths.video_vae).join("config.json").is_file() {
            brain_testutil::skip(&format!("{}/config.json not found - video VAE not (yet) downloaded", paths.video_vae));
            return;
        }

        let video_channels = VideoVaeConfig::real().latent_channels as usize;
        let (mean, std) = read_latent_stats(&paths.video_vae, video_channels).unwrap_or_else(|e| panic!("video VAE latent stats: {e}"));
        assert_eq!(mean.len(), video_channels);
        assert_eq!(std.len(), video_channels);
        assert!(mean.iter().all(|v| v.is_finite()) && std.iter().all(|v| v.is_finite()));
        assert!(std.iter().all(|&s| s > 0.0), "every latent_std must be positive - it divides the encode side");
        // The placeholders this test exists to keep out. Real per-channel
        // stats are nowhere near the identity: the released video VAE's means
        // span roughly -1.37..1.07 and its stds 0.45..3.28.
        assert!(mean.iter().any(|&m| m.abs() > 0.1), "latents_mean looks like the all-zero placeholder: {mean:?}");
        assert!(std.iter().any(|&s| (s - 1.0).abs() > 0.1), "latents_std looks like the all-one placeholder: {std:?}");

        let audio_channels = crate::vocoder::VocoderConfig::h3_32khz().vae_latent_channels as usize;
        if std::path::Path::new(&paths.vocoder).join("config.json").is_file() {
            let (amean, astd) = read_latent_stats(&paths.vocoder, audio_channels).unwrap_or_else(|e| panic!("audio VAE latent stats: {e}"));
            assert_eq!(amean.len(), audio_channels);
            assert_eq!(astd.len(), audio_channels);
            assert!(astd.iter().all(|&s| s > 0.0));
            assert!(amean.iter().any(|&m| m.abs() > 0.01) || astd.iter().any(|&s| (s - 1.0).abs() > 0.1), "audio latents_mean/std look like the placeholders");
        }

        // A wrong channel count is a hard error, never a silent truncation.
        assert!(read_latent_stats(&paths.video_vae, video_channels + 1).is_err(), "a channel-count mismatch must be refused");
    }

    /// `av_outcome`'s wire format round-trips through the shared clip codec,
    /// same discipline as `wan::caps::video_outcome_round_trips_through_the_
    /// shared_clip_codec`, plus the audio side this model always carries.
    #[test]
    fn av_outcome_round_trips_through_the_shared_codecs() {
        let av = pipeline::GeneratedAv {
            video: vec![
                1.0, 0.0, // frame0 red channel, 2 px
                0.0, 1.0, // frame0 green channel
                0.0, 0.0, // frame0 blue channel
            ],
            num_video_frames: 1,
            height: 1,
            width: 2,
            audio: vec![vec![0.1, -0.2, 0.3], vec![-0.1, 0.2, -0.3]],
            sample_rate: 32000,
        };
        let out = av_outcome(&av, 17);
        assert_eq!(out.outputs["seed"], json!(17), "the outcome reports the seed the clip was drawn with");
        assert_eq!(out.outputs["frames"], json!(1));
        assert_eq!(out.outputs["width"], json!(2));
        assert_eq!(out.outputs["height"], json!(1));
        assert_eq!(out.outputs["audio_samples"], json!(3));
        assert_eq!(out.outputs["audio_sample_rate"], json!(32000));

        let video_blob = &out.blobs["video"];
        assert_eq!(video_blob.media, Media::Video);
        assert_eq!(video_blob.meta["fps"], json!(pipeline::FPS));
        let inv = Invocation::new().blob("video", video_blob.clone());
        let back = capability::blob::decode_video(&inv, "video").unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].1, 2, "width");
        assert_eq!(back[0].0[0], 1.0, "red channel of the first pixel");
        assert_eq!(back[0].0[4], 1.0, "green channel of the second pixel");

        let audio_blob = &out.blobs["audio"];
        assert_eq!(audio_blob.media, Media::Audio);
        assert_eq!(audio_blob.meta["channels"], json!(2));
        assert_eq!(audio_blob.meta["sample_rate"], json!(32000));
        assert!(!audio_blob.bytes.is_empty());
    }

    /// `LoadedWeights::load` must fail cleanly (not panic) when the DiT path
    /// does not exist - a real, if trivial, error path.
    #[test]
    fn loaded_weights_load_reports_a_missing_dit_file_cleanly() {
        let paths = Paths::resolve("/nonexistent-minimaxh3-root");
        assert!(LoadedWeights::load(&paths, &paths.dit).is_err());
    }

    /// [`Paths::resolve`] derives every sub-path from one root, each a fixed,
    /// real-checkpoint-confirmed subdirectory name - no field is ever empty
    /// or shares another field's value.
    #[test]
    fn resolve_derives_every_role_from_one_root() {
        let p = Paths::resolve("/some/root");
        assert_eq!(p.dit, "/some/root/transformer");
        assert_eq!(p.video_vae, "/some/root/vae");
        assert_eq!(p.vocoder, "/some/root/audio_vae");
        assert_eq!(p.text_encoder, "/some/root/text_encoder");
        assert_eq!(p.tokenizer, "/some/root/tokenizer");
    }
}
