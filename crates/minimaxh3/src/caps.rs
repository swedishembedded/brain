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
//!   `crates/cli/src/resident_minimaxh3.rs`'s actual hot [`crate::model::
//!   H3Transformer`] is held ONE PER DEVICE, shared across every key -
//!   switching request shape never forces a reload of the 33B DiT.
//!
//! ## Real, tracked gaps this module inherits from earlier phases
//!
//! * **Real DiT weight import does not exist yet.** [`crate::model::
//!   H3Transformer::load`]'s own doc states it accepts a `Tensors` source
//!   "named after the reference module's own attribute paths ... or any
//!   other Tensors source" - this module reads [`Paths::dit`] as exactly
//!   such a source (a bare `safetensors` file already in this crate's own
//!   naming: `to_q`/`to_k`/`to_v` split, not the real checkpoint's fused
//!   `qkv_proj`). Splitting the real checkpoint's fused QKV into that shape
//!   is a real-checkpoint IMPORT step that has not been built (out of THIS
//!   phase's scope, tracked separately) - until it exists, no raw
//!   `MiniMaxAI/MiniMax-H3` `transformer/` directory satisfies `$BRAIN_MINIMAXH3_DIT`
//!   directly.
//! * **No real Qwen3-VL text encoder is wired into this serving path.**
//!   Building `crate::pipeline::TextConditioning` for real needs a resident
//!   `qwen3vl::Qwen3Vl` plus a tokenizer, and the loader that assembles one
//!   from a raw HF directory is private to `qwen3vl::caps`
//!   (`load_hf_resident`) - not something this module can call today.
//!   [`text_conditioning_stub`] below stands in, deterministic and
//!   prompt-seeded but not semantically real - the exact same class of gap
//!   `ltxv::pipeline::context_stub` documents and ships with for its own
//!   still-missing real text encoder path. `t2va`/`fl2va` still run a real
//!   forward, real dual-schedule denoise loop, and real VAE/vocoder decode
//!   end to end against this stub; only the semantic content of what the
//!   video/audio actually depict is not driven by the prompt's real meaning
//!   yet.
//! * **`fl2va`'s two-keyframe text-presentation gap** (`crate::pipeline::
//!   TextConditioning`'s own doc) still applies; on top of it, THIS module's
//!   text-conditioning stub does not splice a keyframe's vision tokens into
//!   the text presentation AT ALL (0 or 1 or 2 keyframes all read identically
//!   to the text encoder) - a strict superset of the pipeline-level gap,
//!   present only because the real Qwen3-VL wiring above is itself missing.
//!   The DiT-side keyframe conditioning rows (`crate::pipeline::
//!   build_packed_sequence`/`encode_keyframe_condition`, the mechanism that
//!   actually anchors the generated video to the keyframes) are unaffected -
//!   they never touch the text encoder, exactly as `TextConditioning`'s own
//!   doc already states.
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
/// then print the attribution notice once per process.
///
/// Called from every entry point that would touch real H3 weights (the
/// eventual `gen_params_from`-equivalent in this module, and each CLI
/// handler), the same call-site discipline `flux2::caps::check_license`
/// uses so no served surface can bypass it.
pub fn check_license() -> Result<(), String> {
    if std::env::var("BRAIN_MINIMAXH3_ALLOW_COMMUNITY").ok().as_deref() != Some("1") {
        return Err(
            "MiniMax-H3 weights are released under the MiniMax H3 Community License Agreement \
             (territorial restrictions apply - the ordinary grant excludes the EU, UK, South \
             Korea and the US; a >$20M/yr revenue registration clause and a mandatory \"MiniMax \
             H3\" UI attribution requirement also apply). Set \
             BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1 to confirm you are authorized to use these \
             weights under that license."
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
const TASKS: [&str; 2] = ["t2va", "fl2va"];

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
            .param(ParamSpec::new("width", ParamType::Int, "canvas width, px - must be given together with height (a multiple of the VAE spatial stride x the DiT's spatial patch size); omit both to resolve a 16:9 canvas at MiniMax-H3's own short-edge/max-pixels defaults"))
            .param(ParamSpec::new("height", ParamType::Int, "canvas height, px - see width"))
            .param(ParamSpec::new("num_frames", ParamType::Int, "video frames at 24fps before alignment; snapped up to the VAE's own chunk grid, and the aligned duration must land in MiniMax-H3's 5-15 second bound").default(json!(default_frames)))
            .param(ParamSpec::new("steps", ParamType::Int, "denoise steps (shared by both the video and the audio schedule, one forward per step - no CFG, this model is guidance-distilled)").default(json!(20)))
            .param(ParamSpec::new("seed", ParamType::Int, "initial-noise seed (omit for 0)").default(json!(0)))
            .output(BlobSpec::new("video", Media::Video, "the generated clip: N interleaved-HWC f32 RGB frames in [0,1], meta {frames,w,h,c,fps}").required())
            .output(BlobSpec::new("audio", Media::Audio, "the clip's own stereo sound track: a complete 32 kHz WAV covering the same time window, generated by the SAME forward as the frames - MiniMax-H3 is natively audio-visual, so this is never absent").required())
    };

    let t2va = common(ActionSpec::new(
        TASKS[0],
        "generate a video clip AND its own soundtrack from a text prompt (Qwen3-VL text conditioning, one packed self-attention DiT denoising video and audio rows together under two independent shifted-sigma Euler schedules in the SAME forward, causal 3D video VAE + DAC/BigVGAN audio VAE decode)",
    ));

    let fl2va = common(ActionSpec::new(
        TASKS[1],
        "t2va, plus one and/or the other end of the clip anchored to a still image: the given still(s) are VAE-encoded, lightly noised, and prepended to the packed sequence as extra conditioning rows the denoise loop never resamples - the generated clip is guided to start and/or end at them",
    ))
    .param(ParamSpec::new("first_frame", ParamType::Str, "server-side path to a PNG/JPEG still the clip's FIRST frame is anchored to. At least one of first_frame/last_frame is required."))
    .param(ParamSpec::new("last_frame", ParamType::Str, "server-side path to a PNG/JPEG still the clip's LAST frame is anchored to. At least one of first_frame/last_frame is required."));

    Manifest::new(
        MODEL,
        "MiniMax-H3 (MiniMax) - a 33B joint video+audio rectified-flow diffusion transformer: one packed self-attention stack denoises video, audio and text-conditioned rows together (not a two-stream architecture like ltxv), under two independent shifted-sigma Euler schedules in one forward, no CFG (guidance-distilled). Qwen3-VL text conditioning truncated to an intermediate hidden layer; causal 3D video VAE + DAC-lineage/BigVGAN audio VAE. t2va: text only. fl2va: text plus a first/last-frame image anchor. ref2va (up to 12 mixed image/video/audio references) is not yet implemented and is not advertised.",
        vec![t2va, fl2va],
    )
}

// ===================== shared param decoding =====================

/// One decoded `t2va`/`fl2va` request's numeric options - everything BOTH
/// actions take. [`fl2va_keyframes_from`] decodes `fl2va`'s own extra
/// `first_frame`/`last_frame` params separately, since only `fl2va` has them.
#[derive(Debug)]
pub struct GenParams {
    pub opts: pipeline::GenOpts,
}

/// Decode + validate the shared params from an invocation, and check the
/// license gate. Every geometric/duration constraint MiniMax-H3 itself
/// enforces is checked HERE too, from the real config's own numbers (pure
/// data, no weight read) - a request that could never run must not cost a
/// checkpoint load to reject, the same discipline `wan::caps::gen_params_from`
/// documents.
pub fn gen_params_from(inv: &capability::Invocation) -> Result<GenParams, String> {
    check_license()?;
    decode_gen_params(inv)
}

/// [`gen_params_from`]'s body, minus the license check - split out so the
/// pure decode/validation logic is testable without setting
/// `BRAIN_MINIMAXH3_ALLOW_COMMUNITY` (every request needs that gate, unlike
/// `flux2::caps`'s variant-scoped one, whose default variant is free -
/// setting env vars races other tests in the same binary, so this crate's
/// tests exercise the decode logic directly instead).
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
        num_inference_steps: inv.get_i64("steps").unwrap_or(20).max(1) as usize,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        // Not an invocation parameter: placement is the SERVER's decision,
        // not the caller's - `ltxv::caps::gen_params_from`'s own identical
        // comment. A resident instance overrides this after decoding
        // (`crates/cli/src/resident_minimaxh3.rs`); the direct provider
        // above leaves it `None` (the host default).
        device: None,
    };
    Ok(GenParams { opts })
}

/// One `fl2va` keyframe request: which end of the clip it anchors, and the
/// server-side path to the still image (`crate::pipeline::Anchor`/
/// `KeyframeCondition`'s own doc for what happens to it once loaded - loading
/// the PNG/JPEG itself is the CALLER's job, same split
/// `ltxv::pipeline::load_still_chw`'s call sites keep, since this crate has
/// no `image`-decoding dependency of its own).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlKeyframeSpec {
    pub anchor: pipeline::Anchor,
    pub path: String,
}

/// Decode `fl2va`'s own `first_frame`/`last_frame` params. At least one is
/// required - checked here so a request naming neither is rejected before
/// any weight is read, matching [`gen_params_from`]'s own discipline.
pub fn fl2va_keyframes_from(inv: &capability::Invocation) -> Result<Vec<FlKeyframeSpec>, String> {
    let mut out = Vec::new();
    if let Some(path) = inv.get_str("first_frame").filter(|s| !s.is_empty()) {
        out.push(FlKeyframeSpec { anchor: pipeline::Anchor::First, path });
    }
    if let Some(path) = inv.get_str("last_frame").filter(|s| !s.is_empty()) {
        out.push(FlKeyframeSpec { anchor: pipeline::Anchor::Last, path });
    }
    if out.is_empty() {
        return Err("fl2va: at least one of first_frame/last_frame is required".to_string());
    }
    Ok(out)
}

// ===================== text conditioning stub =====================

/// FNV-1a, matching `ltxv::pipeline`'s own private copy byte-for-byte - the
/// prompt-to-seed mix [`text_conditioning_stub`] uses.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// A deterministic, prompt-SEEDED placeholder [`pipeline::TextConditioning`],
/// standing in until a real Qwen3-VL text encoder is wired into this crate's
/// serving path (this module's own doc - "Real, tracked gaps" - names exactly
/// what is missing and why). Mirrors `ltxv::pipeline::context_stub`'s own
/// precedent for the identical class of gap: the same prompt and seed always
/// produce the same conditioning (so `--seed` runs stay reproducible), and a
/// different prompt or seed produces a different one (so this is not simply
/// ignoring the prompt) - but the values carry no real semantic content, so
/// what the generated video/audio actually DEPICT is not yet driven by the
/// prompt's real meaning.
pub fn text_conditioning_stub(prompt: &str, text_dim: u32, seed: u64) -> pipeline::TextConditioning {
    let num_tokens = prompt.split_whitespace().count().clamp(1, 64);
    let mix = seed ^ fnv1a(prompt);
    let mut rng = data::rng::Rng::new(mix);
    let embeds: Vec<f32> = (0..num_tokens * text_dim as usize).map(|_| 0.5 * rng.next_gaussian() as f32).collect();
    pipeline::TextConditioning { embeds, token_tags: vec![crate::config::TAG_TEXT; num_tokens] }
}

/// Whether `paths` names a checkout with a real Qwen3-VL text encoder to
/// load - both `text_encoder/config.json` and `tokenizer/tokenizer.json`
/// present (the two files [`build_text_encoder`] reads first). A cheap,
/// pre-load check so a caller can fall back to
/// [`text_conditioning_stub`] without paying for (and failing on) a partial
/// checkout.
pub fn has_real_text_encoder(paths: &Paths) -> bool {
    std::path::Path::new(&paths.text_encoder).join("config.json").is_file() && std::path::Path::new(&paths.tokenizer).join("tokenizer.json").is_file()
}

/// A short, fixed context length for one prompt - MiniMax-H3 prompts are a
/// caption, not a long document, and this is far above any real one while
/// staying well under the checkpoint's own 262144-token ceiling (no reason
/// to size a resident buffer for headroom this crate never uses).
const TEXT_ENCODER_SEQ_LEN: u32 = 512;

/// Load a real [`qwen3vl::Qwen3Vl`] text-only conditioner (`n_visual=0` -
/// see [`qwen3vl::Qwen3Vl::encode_hidden`]'s own doc for why that is only
/// valid for a splice-free instance) plus its BPE tokenizer, from `paths`.
/// Mirrors `qwen3vl::caps`'s own `load_hf_resident` (not reusable directly:
/// private to that module) at fp32, no vision capacity - this crate never
/// splices a keyframe into the SAME instance a plain `t2va` request uses,
/// so there is no reason to pay for DeepStack/splice buffers here. A REAL
/// load: the checkpoint is tens of GB, so this is deliberately not called
/// on every request without [`has_real_text_encoder`] gating it first.
pub fn build_text_encoder(paths: &Paths) -> Result<(qwen3vl::Qwen3Vl, data::qwen_tokenizer::QwenBpe), String> {
    let cfg_path = format!("{}/config.json", paths.text_encoder);
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("minimaxh3 text encoder: cannot read {cfg_path}: {e}"))?;
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_text).map_err(|e| format!("minimaxh3 text encoder: cannot parse {cfg_path}: {e}"))?;
    let cfg = qwen3vl::Qwen3VlConfig::from_hf(&cfg_json);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(&paths.tokenizer).map_err(|e| format!("minimaxh3 text encoder: tokenizer: {e}"))?;
    let qwen = qwen3vl::Qwen3Vl::from_hf(&paths.text_encoder, cfg.vision, cfg.text, TEXT_ENCODER_SEQ_LEN, cfg.image_token_id, 0, 0, cfg.mrope_section, gpu_core::select::Dtype::F32)?;
    Ok((qwen, tok))
}

/// [`pipeline::encode_text`]'s `t2va` case (no keyframe image): tokenize
/// `prompt` verbatim - `get_qwen3vl_prompt_embeds`'s own real caller
/// (`encoders.py`'s `MiniMaxH3TextEncoderStep`) tokenizes with
/// `add_special_tokens=False`, no chat template, and `data::Tokenizer::
/// encode`'s own contract is exactly that (special tokens recognized WITHIN
/// the text, none added around it) - so a plain `tok.encode(prompt)` already
/// matches, no template step needed here unlike `flux2`'s own Qwen3 caption
/// encoder.
pub fn encode_text_real(qwen: &qwen3vl::Qwen3Vl, tok: &data::qwen_tokenizer::QwenBpe, prompt: &str) -> pipeline::TextConditioning {
    use data::Tokenizer;
    let tokens = tok.encode(prompt);
    let token_tags = vec![crate::config::TAG_TEXT; tokens.len()];
    pipeline::encode_text(qwen, &tokens, &token_tags, None)
}

/// [`text_conditioning_stub`] unless `paths` names a real, present Qwen3-VL
/// checkout, in which case the real encoder is built and run instead - the
/// one seam every `t2va`/`fl2va` entry point below shares, so real-weight
/// availability decides real-vs-stub in exactly one place. Deliberately does
/// NOT fall back to the stub when a checkout IS present but fails to load -
/// that would silently swap a real bug for a fake-but-"successful"
/// generation, exactly the failure mode this crate's own real-weight tests
/// elsewhere refuse to hide.
fn text_conditioning(paths: &Paths, prompt: &str, text_dim: u32, seed: u64) -> Result<pipeline::TextConditioning, String> {
    if !has_real_text_encoder(paths) {
        return Ok(text_conditioning_stub(prompt, text_dim, seed));
    }
    let (qwen, tok) = build_text_encoder(paths)?;
    Ok(encode_text_real(&qwen, &tok, prompt))
}

// ===================== outcome shaping =====================

/// Wrap one generated clip+soundtrack as an [`capability::Outcome`] - the
/// shared `capability::blob::video_blob`/`audio::wav::encode_multi` wire
/// format every other video/audio-producing model in this workspace uses
/// (`wan::caps::video_outcome`/`ltxv::caps::video_outcome`'s pattern).
pub fn av_outcome(av: &pipeline::GeneratedAv) -> capability::Outcome {
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
        .blob("video", blob)
        .blob("audio", audio_blob)
}

// ===================== resident weight source =====================

/// Every component directory one `BRAIN_MINIMAXH3_DIR` checkout needs,
/// derived from that single root - matching `arch::ARCHS`'s own
/// `weights_env: &[("BRAIN_MINIMAXH3_DIR", "dir")]` row (one `dir` role, not
/// per-role env vars: this replaces an earlier draft that read three
/// separate `BRAIN_MINIMAXH3_{DIT,VIDEO_VAE,VOCODER}` vars nothing in
/// `arch::ARCHS` ever registered, so real serving could never actually reach
/// it). See [`Paths::dit`]'s own doc for [`crate::model::H3Transformer::
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
    /// sharded Qwen3-VL weights [`build_text_encoder`] loads.
    pub text_encoder: String,
    /// The `tokenizer/` component directory - `tokenizer.json`/`vocab.json`/
    /// `merges.txt`, separate from `text_encoder/` in the root layout.
    pub tokenizer: String,
}

impl Paths {
    /// `None` (not registered) unless `BRAIN_MINIMAXH3_DIR` is set - every
    /// sub-path is a fixed, real-checkpoint-confirmed subdirectory name
    /// under it (this crate's roadmap ledger's "Checkpoint layout" section),
    /// not independently overridable per role the way `ltxv`'s recipe is
    /// (H3's fetch side, `modelstore::recipe::H3Recipe`, has no per-role
    /// granularity either - see that type's own doc for why).
    pub fn from_env() -> Result<Paths, String> {
        let root = std::env::var("BRAIN_MINIMAXH3_DIR").map_err(|_| "minimaxh3: $BRAIN_MINIMAXH3_DIR not set".to_string())?;
        Ok(Paths::resolve(&root))
    }

    /// [`Paths::from_env`]'s pure half - every sub-path `root` implies,
    /// callable directly by tests without the env var.
    pub fn resolve(root: &str) -> Paths {
        Paths {
            dit: format!("{root}/transformer"),
            video_vae: format!("{root}/vae"),
            vocoder: format!("{root}/audio_vae"),
            text_encoder: format!("{root}/text_encoder"),
            tokenizer: format!("{root}/tokenizer"),
        }
    }
}

/// Read a bare file OR a sharded `<name>.safetensors.index.json` directory
/// into a [`vae::blocks::Tensors`] map with NO name remapping -
/// `checkpoint::safetensors::read_model_dir` plus the same
/// `HashMap`-from-`Vec<StTensor>` collection every other import in this crate
/// starts from, minus any fold/rename step (there is none to do here - see
/// [`Paths::dit`]'s doc for whether the real checkpoint's own tensor names
/// need one). `pub` so `crates/cli/src/resident_minimaxh3.rs` can read the
/// DiT tensors ONCE (into its own resident [`crate::model::H3Transformer`])
/// without going through [`LoadedWeights::load`], which would also re-read
/// the VAEs.
pub fn read_tensors(path: &str) -> Result<vae::blocks::Tensors, String> {
    let raw = checkpoint::safetensors::read_model_dir(std::path::Path::new(path))?;
    Ok(raw.into_iter().map(|t| (t.name, (t.shape, t.data))).collect())
}

/// The video/audio VAE weights [`pipeline::t2va`]/[`pipeline::fl2va`] need,
/// loaded once and held by the caller - the DiT is deliberately NOT part of
/// this struct. `wan::caps`'s own VAE deliberately never rides alongside its
/// hot DiT either; here the split additionally lets
/// `crates/cli/src/resident_minimaxh3.rs` hold the (33B, expensive-to-reload)
/// DiT resident while still re-reading these (hundreds of MB, cheap) VAEs
/// per call - the same "not everything worth loading is worth CACHING"
/// judgment `resident_wan.rs`'s own module doc states for its own VAE/T5.
/// Per-channel VAE latent normalization defaults to `mean=0/std=1` (the
/// reference's own registered default) - real per-channel values are not
/// available from this port's current checkpoint download
/// (`pipeline::encode_keyframe_condition`'s own doc states the same gap).
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

impl VaeWeights {
    /// Load both VAEs from `paths`, at the real config numbers - two-way
    /// import coverage via `crate::import`'s own functions.
    pub fn load(paths: &Paths) -> Result<VaeWeights, String> {
        let video_vae_cfg = VideoVaeConfig::real();
        let vocoder_cfg = crate::vocoder::VocoderConfig::h3_32khz();
        let video_vae_tensors = crate::import::import_video_vae(&paths.video_vae, &video_vae_cfg)?;
        let vocoder_tensors = crate::import::import_audio_vae_decoder(&paths.vocoder, &vocoder_cfg)?;
        Ok(VaeWeights {
            video_latents_mean: vec![0.0; video_vae_cfg.latent_channels as usize],
            video_latents_std: vec![1.0; video_vae_cfg.latent_channels as usize],
            audio_latents_mean: vec![0.0; vocoder_cfg.vae_latent_channels as usize],
            audio_latents_std: vec![1.0; vocoder_cfg.vae_latent_channels as usize],
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
    pub fn as_checkpoint<'a>(&'a self, dit_tensors: &'a vae::blocks::Tensors, dit_cfg: H3TransformerConfig) -> pipeline::H3Checkpoint<'a> {
        pipeline::H3Checkpoint {
            dit_tensors,
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
/// (`crates/cli/src/resident_minimaxh3.rs`) uses [`VaeWeights`] plus its own
/// resident DiT instead, so the (33B) DiT is not re-read from disk per call.
pub struct LoadedWeights {
    pub dit_tensors: vae::blocks::Tensors,
    pub dit_cfg: H3TransformerConfig,
    pub vae: VaeWeights,
}

impl LoadedWeights {
    /// Load every weight from `paths` - a bare read for the DiT (see
    /// [`Paths::dit`]'s doc) plus [`VaeWeights::load`].
    pub fn load(paths: &Paths) -> Result<LoadedWeights, String> {
        Ok(LoadedWeights { dit_tensors: read_tensors(&paths.dit)?, dit_cfg: H3TransformerConfig::real(), vae: VaeWeights::load(paths)? })
    }

    /// A borrowing [`pipeline::H3Checkpoint`] over these tensors - built fresh
    /// per call since [`pipeline::H3Checkpoint`] borrows rather than owns.
    pub fn as_checkpoint(&self) -> pipeline::H3Checkpoint<'_> {
        self.vae.as_checkpoint(&self.dit_tensors, self.dit_cfg)
    }
}

// ===================== execution =====================

/// Run one `t2va` against a caller-held resident [`crate::model::
/// H3Transformer`] (see [`pipeline::t2va_hot`]'s doc for why one resident
/// transformer serves every request shape) - the entry point BOTH
/// [`t2va_on`] (cold: builds `model` fresh) and
/// `crates/cli/src/resident_minimaxh3.rs` (hot: reuses one across calls)
/// funnel through, so param decoding, text conditioning and outcome shaping
/// cannot drift between the two. `paths` decides real-vs-stub text
/// conditioning (see [`text_conditioning`]'s own doc) - the qwen3vl
/// checkpoint is reloaded fresh per call when real, matching this crate's
/// documented not-yet-residency-optimized state (the VAEs already do the
/// same; see [`VaeWeights`]'s own doc). Neither cancellation nor per-step
/// progress is threaded through `crate::pipeline`'s denoise loop yet, a
/// real, tracked gap (`wan::pipeline`/`ltxv::pipeline` both poll
/// `inv.cancel` and report progress per step; H3's loop does neither yet), so
/// `inv.cancel` rides along unpolled and `ActionSpec::streaming()` above
/// currently only means "long-running", not "reports intermediate progress".
pub fn t2va_hot_on(ckpt: &pipeline::H3Checkpoint, inv: &capability::Invocation, p: &GenParams, paths: &Paths, model: &crate::model::H3Transformer) -> capability::ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    let text = text_conditioning(paths, &prompt, ckpt.dit_cfg.text_dim, p.opts.seed)?;
    let av = pipeline::t2va_hot(ckpt, &text, &p.opts, model)?;
    Ok(av_outcome(&av))
}

/// [`t2va_hot_on`]'s `fl2va` analogue. `keyframes` are already-encoded
/// (`crate::pipeline::encode_keyframe_condition`, run by the caller once the
/// still images named by [`fl2va_keyframes_from`] have been loaded and
/// resized onto the target canvas). Real text conditioning here is still
/// `t2va`'s own no-image path (see [`text_conditioning`]'s doc and
/// [`pipeline::TextConditioning`]'s own two-keyframe gap) - the keyframes'
/// influence on the generation flows entirely through `keyframes` here, not
/// through the text encoder.
pub fn fl2va_hot_on(ckpt: &pipeline::H3Checkpoint, inv: &capability::Invocation, p: &GenParams, paths: &Paths, keyframes: &[pipeline::KeyframeCondition], model: &crate::model::H3Transformer) -> capability::ActionResult {
    let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
    if keyframes.is_empty() {
        return Err("fl2va: at least one keyframe is required - use t2va for a text-only request".to_string());
    }
    let text = text_conditioning(paths, &prompt, ckpt.dit_cfg.text_dim, p.opts.seed)?;
    let av = pipeline::fl2va_hot(ckpt, &text, keyframes, &p.opts, model)?;
    Ok(av_outcome(&av))
}

/// [`t2va_hot_on`]'s COLD equivalent: load a fresh [`crate::model::
/// H3Transformer`] from `weights.dit_tensors` and run once. Used by the
/// direct (non-resident) [`MiniMaxH3Provider`] below - real residency is
/// `crates/cli/src/resident_minimaxh3.rs`'s job.
pub fn t2va_on(weights: &LoadedWeights, inv: &capability::Invocation, p: &GenParams, paths: &Paths) -> capability::ActionResult {
    let ckpt = weights.as_checkpoint();
    let model = crate::model::H3Transformer::load(&weights.dit_tensors, weights.dit_cfg, p.opts.device.as_deref());
    t2va_hot_on(&ckpt, inv, p, paths, &model)
}

/// [`t2va_on`]'s `fl2va` analogue.
pub fn fl2va_on(weights: &LoadedWeights, inv: &capability::Invocation, p: &GenParams, paths: &Paths, keyframes: &[pipeline::KeyframeCondition]) -> capability::ActionResult {
    let ckpt = weights.as_checkpoint();
    let model = crate::model::H3Transformer::load(&weights.dit_tensors, weights.dit_cfg, p.opts.device.as_deref());
    fl2va_hot_on(&ckpt, inv, p, paths, keyframes, &model)
}

// ===================== execution (direct provider) =====================

use std::sync::Arc;

use capability::{Action, ActionResult, Invocation, Progress, Provider};

/// The executable MiniMax-H3 model behind the manifest, for the DIRECT
/// (non-resident) serving path - `brain do brain/minimaxh3 t2va ...` and the
/// catalog's `always!` entry (`resident: None`, see `crates/catalog/src/
/// lib.rs`; the residency-scheduled adapter is
/// `crates/cli/src/resident_minimaxh3.rs`'s `MiniMaxH3Resident`, registered
/// separately in `crates/cli/src/resident.rs` - `wan`/`ltxv`'s own
/// direct-provider/resident-adapter split). Loads fresh weights every call
/// (see [`run_hot_dit`]'s doc) - no caching here, since caching across calls
/// is exactly what the residency-scheduled path exists to do instead.
pub struct MiniMaxH3Provider;

impl MiniMaxH3Provider {
    pub fn new() -> MiniMaxH3Provider {
        MiniMaxH3Provider
    }
}

impl Default for MiniMaxH3Provider {
    fn default() -> Self {
        MiniMaxH3Provider::new()
    }
}

impl Provider for MiniMaxH3Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| Arc::new(MiniMaxH3Action { name: name.to_string() }) as Arc<dyn Action>)
    }
}

struct MiniMaxH3Action {
    name: String,
}

impl Action for MiniMaxH3Action {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Params before the weights-env check: a request that could never
        // run must not read "you forgot to export BRAIN_MINIMAXH3_DIT" -
        // `wan::caps::WanAction::run`'s own ordering.
        let p = gen_params_from(inv)?;
        match self.name.as_str() {
            "t2va" => {
                let paths = Paths::from_env()?;
                let weights = LoadedWeights::load(&paths)?;
                t2va_on(&weights, inv, &p, &paths)
            }
            "fl2va" => {
                let _keyframe_specs = fl2va_keyframes_from(inv)?;
                // Loading the named stills into pixel buffers needs an
                // `image`-decoding dependency this crate deliberately does
                // not carry (see `Paths`' own doc for the identical split on
                // the DiT importer); the residency-scheduled path
                // (`crates/cli/src/resident_minimaxh3.rs`, which DOES depend
                // on `image`) is `fl2va`'s real, working entry point today.
                Err("minimaxh3 fl2va: not runnable through the direct provider yet - the residency-scheduled adapter (crates/cli/src/resident_minimaxh3.rs) loads keyframe stills; a direct-provider path needs the same image-decoding wiring added here".to_string())
            }
            other => Err(format!("minimaxh3 '{other}': unknown action")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_real_text_encoder_is_false_for_a_nonexistent_root() {
        assert!(!has_real_text_encoder(&Paths::resolve("/nonexistent-minimaxh3-root")));
    }

    /// [`text_conditioning`] must fall back to the stub - not error, not
    /// hang trying to load anything - when no real text encoder is present,
    /// so every existing weight-free `t2va`/`fl2va` caller keeps working.
    #[test]
    fn text_conditioning_falls_back_to_the_stub_without_a_real_encoder() {
        let paths = Paths::resolve("/nonexistent-minimaxh3-root");
        let text = text_conditioning(&paths, "a red fox running through snow", 16, 7).expect("stub path must not error");
        assert!(!text.embeds.is_empty());
        assert_eq!(text.token_tags.len(), text.embeds.len() / 16);
        // Reproducible at the same seed - `text_conditioning_stub`'s own contract.
        let again = text_conditioning(&paths, "a red fox running through snow", 16, 7).unwrap();
        assert_eq!(text.embeds, again.embeds);
    }

    #[test]
    fn gated_unless_opted_in() {
        // The gate reads the env var per call; only assert the refusing path
        // here (setting env vars in tests races other tests in the binary -
        // same discipline as `flux2::caps`'s equivalent test).
        if std::env::var("BRAIN_MINIMAXH3_ALLOW_COMMUNITY").ok().as_deref() != Some("1") {
            let err = check_license().unwrap_err();
            assert!(err.contains("BRAIN_MINIMAXH3_ALLOW_COMMUNITY"), "error must name the opt-in: {err}");
            assert!(err.contains("Community License"), "error must name the license: {err}");
        }
    }

    /// `gen_params_from` itself must be license-gated - every request needs
    /// the gate here (unlike `flux2::caps`, there is no free variant), so a
    /// caller cannot skip it by going through the caps-layer decoder instead
    /// of `check_license` directly.
    #[test]
    fn gen_params_from_is_license_gated() {
        if std::env::var("BRAIN_MINIMAXH3_ALLOW_COMMUNITY").ok().as_deref() != Some("1") {
            let inv = Invocation::new().set("prompt", json!("a cat"));
            let err = gen_params_from(&inv).unwrap_err();
            assert!(err.contains("BRAIN_MINIMAXH3_ALLOW_COMMUNITY"), "{err}");
        }
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
        assert_eq!(def(t2va, "seed"), Some(json!(0)));
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

        // ref2va must not be advertised - it is not implemented.
        assert!(!names.contains(&"ref2va"));

        // The whole manifest round-trips to JSON for discovery.
        let j = m.to_json();
        assert_eq!(j["model"], MODEL);
        assert_eq!(j["actions"].as_array().unwrap().len(), 2);
        assert_eq!(j["actions"][0]["streaming"], true);
        assert_eq!(j["actions"][0]["params"][0]["name"], "prompt");
        assert_eq!(j["actions"][0]["params"][0]["required"], true);
    }

    /// The manifest's own advertised defaults must survive `validate` ->
    /// `decode_gen_params` unchanged - the join the two halves can drift at.
    #[test]
    fn the_advertised_defaults_decode() {
        let spec = manifest().actions.into_iter().next().unwrap();
        let inv = spec.validate(Invocation::new().set("prompt", json!("a cat"))).unwrap();
        let p = decode_gen_params(&inv).unwrap();
        assert_eq!(p.opts.num_frames, default_num_frames());
        assert_eq!(p.opts.num_inference_steps, 20);
        assert_eq!(p.opts.seed, 0);
        assert_eq!(p.opts.canvas, None);
        assert_eq!(p.opts.device, None, "placement is the server's decision, not an invocation param");
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
        assert_eq!(p.opts.num_inference_steps, 2);
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
    }

    /// The stub is deterministic and prompt/seed-sensitive - never a
    /// no-op, and reproducible across identical calls (so `--seed` runs
    /// stay reproducible even though the semantics are placeholder).
    #[test]
    fn text_conditioning_stub_is_deterministic_and_prompt_seeded() {
        let a = text_conditioning_stub("a cat on a skateboard", 12, 7);
        let b = text_conditioning_stub("a cat on a skateboard", 12, 7);
        assert_eq!(a.embeds, b.embeds, "same prompt+seed must reproduce exactly");
        assert_eq!(a.token_tags, b.token_tags);
        assert!(a.token_tags.iter().all(|&t| t == crate::config::TAG_TEXT));
        assert_eq!(a.embeds.len(), a.token_tags.len() * 12);

        let c = text_conditioning_stub("a completely different prompt", 12, 7);
        assert_ne!(a.embeds, c.embeds, "a different prompt must change the stub");
        let d = text_conditioning_stub("a cat on a skateboard", 12, 8);
        assert_ne!(a.embeds, d.embeds, "a different seed must change the stub");
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
        let out = av_outcome(&av);
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
        assert!(LoadedWeights::load(&paths).is_err());
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

    #[test]
    fn paths_from_env_requires_brain_minimaxh3_dir() {
        // No assertion on the ambient environment (another test/process may
        // have it set) - only that the function does not panic and that a
        // `Paths` it returns carries every field non-empty.
        if let Ok(p) = Paths::from_env() {
            assert!(!p.dit.is_empty() && !p.video_vae.is_empty() && !p.vocoder.is_empty() && !p.text_encoder.is_empty() && !p.tokenizer.is_empty());
        }
    }
}
