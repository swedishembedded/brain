// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end LTX-2.5 text-to-video: noise -> a rectified-flow ancestral
//! Euler denoise loop over the tiny video-only DiT ([`crate::dit`]) with
//! classifier-free guidance -> the real causal 3D VAE decode
//! ([`crate::vae3d`]) -> RGB frames. Mirrors `wan::pipeline`'s shape
//! (`GenOpts`/`Paths`/`generate`/a private `Denoiser` trait for testing the
//! CFG fold against a fake) as closely as the two architectures' real
//! differences allow.
//!
//! ## Real weights are opt-in, not the default (read before assuming this
//! ## generates anything at production quality)
//!
//! By default `GenOpts::dit_config == "tiny"` and `Paths::text_encoder ==
//! None`, and [`generate`] runs the ORIGINAL smoke-test path this module
//! shipped with: the DiT is [`crate::config::LtxDitConfig::tiny`] with
//! FRESH RANDOM WEIGHTS ([`crate::dit::random_tiny_weights`], seeded so
//! `--seed` is reproducible) and [`context_stub`] fabricates a
//! deterministic-but-meaningless `[context_len, cross_attention_dim]`
//! context from the prompt string's hash folded into the seed - the
//! "prompt" changes the output only because it changes the stub, carrying
//! no semantic meaning whatsoever.
//!
//! Setting `GenOpts::dit_config = "ltx25_22b"` plus [`Paths::dit`]
//! (`--dit` or the model store) switches to the real 22B distilled
//! checkpoint, loaded straight off its GGUF via
//! [`crate::gguf_src::LtxvGgufSource`] and run at int8 compute
//! ([`RealDit`], [`crate::dit::forward_q_streamed`]) - real weights were
//! only ever proven correct at REDUCED DEPTH / one block before this, so
//! treat a real-weight clip as a wiring proof, not a quality claim, exactly
//! the framing this module always carried for the tiny path. Setting
//! [`Paths::text_encoder`] (`--text-encoder` or the model store)
//! independently switches [`context_stub`] out for [`real_text_context`]'s
//! real Gemma-4 encoding. The two switches are independent (a real DiT with
//! a stub context, or vice versa, both run) but only "real DiT + real text
//! encoder" is an honest end-to-end real-weight generation.
//!
//! [`GenOpts::audio`] runs LTX-2.5's AUDIO stream beside its video one, in
//! the same forwards, on the same int8/streamed/device-resident path
//! ([`RealAvDit`], [`crate::dit::av_forward_q_streamed_in`]). The eager
//! host-fp32 arm that one is gated and measured against is still reachable,
//! behind [`Runtime::av_fp32_reference`]; which of the two
//! a run took is named in its own banner ([`av_arm_label`]).
//!
//! Everything else is real: [`ltx2_sigmas`]'s token-count-dependent shift,
//! [`euler_ancestral_step`]'s rectified-flow ancestral formula, the RoPE
//! position-bounds construction, the classifier-free guidance fold, and the
//! VAE decode (real weights, the same `vae3d`/`import` this port's own
//! parity tests gate).
//!
//! ## One stage or two, and why that is not a free choice
//!
//! [`generate`] denoises in ONE stage at the requested resolution up to
//! [`SINGLE_STAGE_MAX_TOKENS`] video tokens and in the reference's TWO above
//! it ([`should_two_stage`]). That is not an optimization: the distilled
//! checkpoint's fixed sigma table is only ever asked, upstream, to build a
//! clip from noise at HALF the requested resolution, and past ~6k tokens
//! asking it for the whole thing in one stage measurably disintegrates the
//! END of the clip while the beginning stays correct.
//! [`SINGLE_STAGE_MAX_TOKENS`]'s doc carries the measurement, the failure's
//! exact shape in latent space, and the experiment that told the latent's
//! defect apart from the decoder's.
//!
//! Two-stage runs need [`Paths::spatial_upsampler`] and both axes on a
//! multiple of 64; below the ceiling neither is read and nothing this port
//! shipped before changes.
//!
//! ## One simplification inside the guidance fold
//!
//! Real LTX-2.5 sampling (`ltx_pipelines.utils.samplers`) can layer STG
//! (spatiotemporal guidance), audio/video joint guidance, and CFG rescale on
//! top of plain CFG. None of that is implemented here: this pipeline runs
//! ONLY `uncond + guidance·(cond - uncond)` on the model's predicted
//! velocity (algebraically identical to the roadmap's `cond +
//! (guidance-1)·(cond-uncond)` spelling), the same fold `wan::pipeline` uses.
//! STG needs the perturbation machinery this port hasn't built; audio/video
//! joint guidance needs the audio stream, which is a separate, later
//! milestone entirely. `guidance <= 1.0` skips the unconditional forward
//! exactly as `wan::pipeline` does (the combination collapses to the
//! conditional prediction).
//!
//! ## Token layout: the DiT and the VAE disagree, on purpose
//!
//! [`crate::dit::LtxDit::forward`] reads/writes **token-major** `[T,
//! in_channels]` (one contiguous channel vector per token, `T = lat_t·lh·lw`
//! in `(frame, height, width)` raster order - `ltx_core`'s own
//! `VideoLatentPatchifier` convention). [`crate::vae3d::LtxVaeDecoder::decode`]
//! reads/writes **channel-major** `[C, lat_t, lh, lw]` (the video-VAE
//! convention every `Builder3d` op in this crate shares with `wan`'s VAE).
//! [`tc_to_chw`] is the one transpose this pipeline needs, applied ONCE after
//! the denoise loop, right before the VAE boundary - the noise latent itself
//! is generated directly in the DiT's own token-major layout, so no
//! transpose is needed going in.

use std::time::Instant;

use vae::blocks::Tensors;

use crate::config::{LtxAvDitConfig, LtxDitConfig};
use crate::dit::{random_tiny_weights, LtxDit};
use crate::import::VideoVaeArch;
use crate::longform::Scene;
use crate::upsampler::{LatentUpsampler, LatentUpsamplerConfig};
use crate::vae3d::{LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder};
use diffusion::scheduler::{euler_ancestral_step, ltx2_sigmas, LTX2_DISTILLED_SIGMAS, LTX2_STAGE2_DISTILLED_SIGMAS};

/// The real distilled schedule's own step count (`LTX2_DISTILLED_SIGMAS.len() -
/// 1`), exposed so a caller (e.g. `crates/cli/src/ltxv_cli.rs`'s own
/// progress line) can report it without hardcoding a number that would drift
/// from the table itself.
pub const LTX2_DISTILLED_STEPS: usize = LTX2_DISTILLED_SIGMAS.len() - 1;

/// The refinement stage's own step count - `STAGE_2_DISTILLED_SIGMAS`'s
/// table length minus its terminal zero, the same relation
/// [`LTX2_DISTILLED_STEPS`] has to its own table. Three, for the real
/// checkpoint.
pub const LTX2_STAGE2_STEPS: usize = LTX2_STAGE2_DISTILLED_SIGMAS.len() - 1;

/// Read a checkpoint from a directory of safetensors shards or one
/// safetensors file - the real VAE ships as a single file, so this is a
/// strict subset of `wan::pipeline::read_any` (no `.pth` support needed
/// here; add it if a future role needs one).
fn read_any(path: &str) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        return checkpoint::safetensors::read_model_dir(p);
    }
    if !p.exists() {
        return Err(format!("{path} does not exist"));
    }
    checkpoint::safetensors::read(path)
}

/// Where the real weights live. `vae` is the one REQUIRED role (see this
/// module's doc for why the DiT/text-encoder were always tiny-config/stub
/// until this milestone). `dit`/`text_encoder` are OPTIONAL: absent,
/// [`generate`] falls back exactly to the tiny random-weight DiT /
/// [`context_stub`] path every earlier milestone shipped; present, it loads
/// the real 22B GGUF / real Gemma-4 checkpoint instead - see
/// [`dit_config_from_name`] and [`generate`]'s own body for exactly which
/// value of `GenOpts::dit_config` selects which path.
#[derive(Clone, Debug)]
pub struct Paths {
    pub vae: String,
    /// The real 22B distilled DiT GGUF (`ltx-2.5-22b-distilled-transformer-
    /// {Q8_0,Q4_K_M}.gguf`) - required only when `GenOpts::dit_config` names
    /// a real config (`"ltx25_22b"`), read via
    /// `crate::gguf_src::LtxvGgufSource`.
    pub dit: Option<String>,
    /// The real Gemma-4-12B text encoder (`gemma4-*-with-proj-ltx-2.5-
    /// bf16.safetensors`, tokenizer embedded as its own `tokenizer_json`
    /// tensor - see `gemma4::tokenizer`'s doc) - when absent, [`generate`]
    /// keeps using [`context_stub`].
    pub text_encoder: Option<String>,
    /// The real spatial x2 latent upscaler
    /// (`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`), needed
    /// only for requests above [`SINGLE_STAGE_MAX_TOKENS`], where
    /// [`generate`] runs the reference's own two-stage shape and this is what
    /// carries stage 1's latent up to the requested resolution. Below that
    /// threshold nothing reads it.
    pub spatial_upsampler: Option<String>,
    /// The real temporal x2 latent upscaler
    /// (`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`), which
    /// only DFR's temporal rounds read ([`DfrPaths`]).
    pub temporal_upsampler: Option<String>,
    /// The real audio VAE (`ltx-2.5-audio-vae-bf16.safetensors`), which
    /// carries the audio VAE decoder AND the vocoder as two disjoint tensor
    /// subsets of one file. Required only when `GenOpts::audio` is set;
    /// without it a request for sound is refused before any weight is read,
    /// rather than generating an audio latent nothing can decode.
    pub audio_vae: Option<String>,
    /// The duration head (`ltx-2.5-duration-head-bf16.safetensors`), which
    /// predicts a clip's natural length from its prompt when the request
    /// names none ([`predict_frames`]). Resolved from the model store only;
    /// without it a request that names no length takes the default.
    pub duration_head: Option<String>,
    /// DFR's detailing IC-LoRA (`ltx-2.5-22b-ic-lora-refine-details-*`),
    /// which its detailing pass applies at [`DETAILING_LORA_STRENGTH`].
    pub detailing_lora: Option<String>,
}

impl Paths {
    /// Only the VAE, the one required role: every other role absent, which is
    /// the tiny smoke-test configuration.
    pub fn new(vae: impl Into<String>) -> Paths {
        Paths { vae: vae.into(), dit: None, text_encoder: None, spatial_upsampler: None, temporal_upsampler: None, audio_vae: None, duration_head: None, detailing_lora: None }
    }

    /// Build `Paths` from an already-resolved [`capability::Assembly`]
    /// (`crate::spec::LtxvSpec`'s roles) - `vae` is the only required role
    /// (the resolver never returns an assembly without it), every other one
    /// absent when the store holds none. `spatial_upsampler` names one
    /// outright, ahead of the store's.
    pub fn from_assembly(assembly: &capability::Assembly, spatial_upsampler: Option<&str>) -> Result<Paths, String> {
        let get = |role: &str| -> Option<String> { assembly.roles.get(role).map(|p| p.to_string_lossy().into_owned()) };
        let vae = get("vae").ok_or_else(|| format!("ltxv: assembly '{}' has no vae role", assembly.id))?;
        let spatial_upsampler = spatial_upsampler.filter(|s| !s.is_empty()).map(str::to_string).or_else(|| get("spatial_upsampler"));
        Ok(Paths {
            vae,
            dit: get("dit"),
            text_encoder: get("text_encoder"),
            spatial_upsampler,
            temporal_upsampler: get("temporal_upsampler"),
            audio_vae: get("audio_vae"),
            duration_head: get("duration_head"),
            detailing_lora: get("detailing_lora"),
        })
    }
}

/// The largest video-token count the distilled checkpoint's own fixed
/// 8-sigma schedule still produces a coherent clip at **in one stage**, from
/// pure noise, measured on this hardware against the real 22B Q8_0 DiT.
///
/// # Why there is a ceiling at all
///
/// `ltx_pipelines.distilled.DistilledPipeline.__call__` never runs its
/// `DISTILLED_SIGMAS` table at the requested resolution. It runs it at
/// `width // 2, height // 2`, then carries that latent up with the spatial
/// x2 upscaler and spends three more steps (`STAGE_2_DISTILLED_SIGMAS`,
/// starting at sigma 0.909375) refining at full resolution. So the table is
/// only ever asked to build structure from noise at a QUARTER of the tokens
/// the output has, and upstream's largest shipped preset
/// (`LTX_2_3_HQ_PARAMS`, 1088x1920 out) puts that at 544x960 = 2040 video
/// tokens.
///
/// This port ran one stage at the full requested resolution, which is fine
/// while the token count stays near what the table was distilled for and is
/// not fine past it. Measured end to end - real Q8_0 22B DiT, real Gemma-4
/// encoder, real conv VAE, one prompt, one seed, one conditioning still,
/// everything but the resolution held fixed, scored with
/// [`crate::clipmetric::blowup_ratio`] (max over median frame-to-frame
/// difference; ~1 for a clip with steady motion however fast):
///
/// | request | video tokens | blowup ratio |
/// |---|---:|---:|
/// | 512x512 | 1024 | 1.06 |
/// | 960x544 | 2040 | 1.03 |
/// | 1280x704 | 3520 | 1.04 |
/// | 1600x896 | 5600 | 1.04 |
/// | 1920x1088 | **8160** | **14.66** |
///
/// The ceiling is set BETWEEN the largest measured-good count and the
/// measured-broken one - the same discipline
/// [`crate::vae3d::WHOLE_DECODE_MAX_PIXELS`] uses - so every shape this port
/// already ran keeps its exact behaviour and only the one that disintegrates
/// changes path. Where in `(5600, 8160)` the real cliff sits is not measured
/// and this constant does not pretend to know; it only has to separate them.
///
/// The failure is not a gradual softening - it is the LAST latent frame's
/// content collapsing while the rest of the clip stays correct. In the
/// 8160-token latent, latent frame 3's standard deviation falls to 0.911
/// (the other three sit at 1.07/0.98/1.01) while its distance from latent
/// frame 2 rises to 0.630 against 0.386 for the pair before it; in every
/// good latent above, the last frame's deviation returns to ~1.07 and the
/// adjacent distances DECREASE monotonically. Decoding the bad latent with
/// latent frame 3 replaced by latent frame 2 takes the clip's blowup ratio
/// from 17.43 to 1.31, which is what makes this the latent's defect rather
/// than the decoder's.
///
/// [`Runtime::two_stage`] (`--two-stage on|off`) forces the choice either way, which is also
/// how the two paths get compared at a shape that does not need it.
pub const SINGLE_STAGE_MAX_TOKENS: usize = 6144;

/// Whether a request of `tokens` video tokens at `width`x`height` should be
/// generated as the reference's TWO stages rather than one.
///
/// Three conditions, all necessary:
///
/// * the token count exceeds [`SINGLE_STAGE_MAX_TOKENS`] (below it the
///   single-stage path is measured-good and is left exactly as it was);
/// * both axes are multiples of 64, so halving them lands on the VAE's own
///   32-pixel spatial stride - upstream asserts the same thing for exactly
///   the same reason (`assert_resolution(..., is_two_stage=True)`);
/// * a real distilled checkpoint is in play at all, since the schedule this
///   is about is that checkpoint's.
///
/// `forced` ([`Runtime::two_stage`]) overrides the token test (never the
/// divisibility one, which is a hard geometric requirement).
pub fn should_two_stage(tokens: usize, width: usize, height: usize, real_distilled: bool, forced: Option<bool>) -> bool {
    if !width.is_multiple_of(64) || !height.is_multiple_of(64) {
        return false;
    }
    forced.unwrap_or(real_distilled && tokens > SINGLE_STAGE_MAX_TOKENS)
}

/// Which settings a request named itself - every other one is
/// [`resolve_defaults`]'s to fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Named {
    pub dit_config: bool,
    pub frames: bool,
    pub size: bool,
    pub fps: bool,
}

/// The reference pipelines' own defaults (`PipelineParams`): a 768x512
/// clip of 121 frames at 24 fps.
pub const PRODUCTION_WIDTH: usize = 768;
pub const PRODUCTION_HEIGHT: usize = 512;
pub const PRODUCTION_FRAMES: usize = 121;
pub const PRODUCTION_FPS: usize = 24;

/// Fill what a request left unnamed from what is configured: with the real
/// checkpoint resolved, the real DiT and the reference's own clip size,
/// length and frame rate; without it, [`GenOpts::default`]'s tiny smoke test
/// stays, which is the only thing that can run.
///
/// [`GenOpts::default`] is the smoke-test configuration every unit test
/// builds on; a production request goes through this instead, so a plain
/// command with the real weights in the store renders the real model rather
/// than random weights.
pub fn resolve_defaults(o: &mut GenOpts, paths: &Paths, named: Named) {
    if !named.dit_config && paths.dit.is_some() {
        o.dit_config = "ltx25_22b".into();
    }
    if o.dit_config == "tiny" {
        return;
    }
    if !named.size {
        (o.width, o.height) = (PRODUCTION_WIDTH, PRODUCTION_HEIGHT);
    }
    if !named.fps {
        o.fps = PRODUCTION_FPS;
    }
    if !named.frames {
        o.frames = PRODUCTION_FRAMES;
    }
}

/// Which DiT config to build. `"tiny"` (the default) is the fresh-random-
/// weight smoke-test config every earlier milestone shipped. `"ltx25_22b"`
/// is the real 22B checkpoint's own config - selecting it requires
/// [`Paths::dit`] (`--dit` or the model store) to also be set, checked in
/// [`generate`] itself (this function only maps a name to a shape; it does
/// not know about weight paths).
pub fn dit_config_from_name(name: &str) -> Result<LtxDitConfig, String> {
    match name {
        "tiny" => Ok(LtxDitConfig::tiny()),
        "ltx25_22b" => Ok(LtxDitConfig::ltx25_22b()),
        other => Err(format!("unknown ltxv dit-config {other:?} (tiny, ltx25_22b)")),
    }
}

/// The text encoder's arithmetic tiers, as [`Runtime::text_precision`] names them.
pub use gemma4::Precision as TextPrecision;

/// How a generation runs, as opposed to what it asks for: placement and
/// arithmetic choices the caller makes explicitly and that travel with the
/// request, so nothing about a run is decided by process-global state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Runtime {
    /// DiT blocks each card holds resident across a generation's forwards;
    /// `None` plans from the card's free VRAM ([`crate::devres::planned_slots`]),
    /// `Some(0)` streams every block on every forward.
    pub resident_blocks: Option<u32>,
    /// Force the reference's two-stage shape on or off; `None` decides from
    /// the request ([`should_two_stage`]).
    pub two_stage: Option<bool>,
    /// Denoise an audio-visual request on the eager host-fp32 reference arm
    /// ([`crate::av_stream::AvDenoiser`]) instead of the quantized,
    /// device-resident one - the same model at higher precision, for A/B
    /// measurement.
    pub av_fp32_reference: bool,
    /// Force the tiled VAE decode on or off; `None` is the measured policy
    /// ([`crate::vae3d::should_tile`]).
    pub vae_tiling: Option<bool>,
    /// The text encoder's arithmetic; `None` follows the checkpoint (int8 for
    /// a quantized GGUF, fp32 for safetensors).
    pub text_precision: Option<gemma4::Precision>,
    /// The encoded-prompt disk cache ([`crate::text_cache`]).
    pub text_cache: crate::text_cache::Policy,
    /// Write every latent about to be VAE-decoded to this path
    /// ([`crate::latentdump`]), so the decode can be rerun on its own.
    pub latent_dump: Option<String>,
}

/// Everything a single generation varies.
#[derive(Clone, Debug)]
pub struct GenOpts {
    /// Video frames. Must be `1 + 8k` - the causal VAE gives the first frame
    /// its own latent frame (stride 8 temporally), so 16 is not
    /// representable and 17 is.
    pub frames: usize,
    /// Pixels, must be a multiple of 32 (the VAE's spatial stride).
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Each stream's guidance ([`crate::guidance`]): classifier-free, STG,
    /// isolated-modality and rescale. The default is a single conditional
    /// forward per step - what the distilled checkpoint denoises with.
    pub guidance: crate::guidance::Guidance,
    /// The caption the negative-prompt guidance pass is conditioned on -
    /// encoded only when some stream's guidance runs that pass
    /// ([`crate::guidance::Guidance::needs_uncond`]). Defaults to the
    /// reference's [`crate::guidance::DEFAULT_NEGATIVE_PROMPT`].
    pub negative_prompt: String,
    pub seed: u64,
    pub fps: usize,
    /// [`ltx2_sigmas`]'s four shape parameters. Defaults are `LTX2Scheduler.
    /// execute`'s own signature defaults.
    pub base_shift: f64,
    pub max_shift: f64,
    pub stretch: bool,
    pub terminal: f64,
    /// The denoising loop every stage runs ([`Sampler`]).
    pub sampler: Sampler,
    /// The sampler's noise strength. Euler: [`euler_ancestral_step`]'s
    /// `eta`, `1.0` (upstream's distilled-pipeline default, `ANCESTRAL_ETA`)
    /// fully ancestral, `0.0` a plain deterministic Euler step. res_2s: the
    /// step-level SDE strength, `0.5` its reference default
    /// ([`Sampler::default_eta`]).
    pub eta: f64,
    /// [`euler_ancestral_step`]'s `s_noise`; res_2s has none.
    pub s_noise: f64,
    /// The stub text context's token count (see [`context_stub`] - there is
    /// no real tokenizer, so this is just a shape parameter).
    pub context_len: usize,
    /// [`dit_config_from_name`]'s key. `"tiny"` is the only value
    /// implemented this milestone.
    pub dit_config: String,
    /// Generate the clip's SOUND as well as its picture.
    ///
    /// LTX-2.5 is natively audio-visual: one transformer denoises a video
    /// stream and an audio stream together, coupled every block by
    /// cross-attention, and two thirds of the distilled checkpoint's tensors
    /// are the audio half. Setting this runs the model that is in the file;
    /// leaving it off runs only its video half and the clip is silent.
    ///
    /// Off by default, and the default is about COST, not about correctness.
    /// The audio-extended block runs on the same streamed/quantized/
    /// device-resident path the video-only one does ([`RealAvDit`]), so a
    /// joint step costs modestly more than a video-only step at the same
    /// shape rather than several times more - but it is still more, and the
    /// audio-visual path still has no training support and no bandwidth
    /// extension, so turning sound on stays something a caller asks for.
    ///
    /// Requires `dit_config = "ltx25_22b"` and [`Paths::audio_vae`]; both are
    /// checked before any weight is read.
    pub audio: bool,
    /// A soundtrack to generate the video FOR (audio-to-video,
    /// `ltx_pipelines.a2vid_two_stage`): encoded through the audio VAE and
    /// held as the frozen, clean audio stream every stage denoises the video
    /// against, and returned as the clip's own audio. Needs [`Self::audio`].
    pub audio_source: Option<crate::audio::AudioClip>,
    pub device: Option<String>,
    /// First-frame image conditioning (PNG/JPEG), resized to
    /// `width`x`height` and encoded through the real video VAE - see
    /// [`conditioned_latent`]'s doc for which reference conditioning item
    /// this becomes (it depends on whether `end_frame` is also set) and
    /// [`Frozen`]'s for how it composes with `eta`. `None` is
    /// pure text-to-video (every token noise, this pipeline's original and
    /// still-default behavior).
    pub start_frame: Option<String>,
    /// Last-pixel-frame image conditioning at pixel-frame `frames - 1` - may
    /// be the SAME path as `start_frame`, for a clip that loops seamlessly
    /// since the generated content in between has to connect the still to
    /// itself; or a DIFFERENT path, for a clip that morphs from one still to
    /// another. At least one of `start_frame`/`end_frame` being set is what
    /// turns image conditioning on at all.
    ///
    /// **Passing the SAME image at both ends usually produces a STATIC
    /// clip, and that is the model answering the question rather than a
    /// defect.** "Start at this image and end at this same image" has a
    /// correct trivial solution. Measured on the real 22B checkpoint with
    /// `crates/ltxv/tests/it/motion_real.rs`'s metric (peak excursion from
    /// frame 0; anything >= 18 visibly animates, anything <= 9 is the
    /// pinned still repeated): 7.3 at 640x320/25 frames, and 40.0 for the
    /// same shape, seed, prompt and code with only the end still changed to
    /// a mirrored copy. See that file's table for the full matrix,
    /// including the levers that do NOT change it (strength, sampler,
    /// conditioning mechanism, clip length, conditioning-image
    /// compression). For a clip that has to move, anchor two DIFFERENT
    /// instants, or use `start_frame` alone.
    pub end_frame: Option<String>,
    /// Image conditioning at one INTERIOR instant of the clip, so a single
    /// generation can be pinned at its start, its middle and its end at once.
    ///
    /// Mechanically this is a PIN, not the appended guiding block `end_frame`
    /// uses: [`conditioned_latent`] overwrites the latent slot containing
    /// this instant with the encoded still and freezes it there (the
    /// deviation from the reference is documented on that function - an
    /// appended guide measurably bends the trajectory toward the instant
    /// without ever realising the still's content). The end still stays an
    /// appended guide; see [`mid_anchor_frame`] for where the default
    /// position comes from.
    pub mid_frame: Option<String>,
    /// Which pixel frame [`Self::mid_frame`] anchors. `None` takes the
    /// reference's own single-interior-keyframe position - see
    /// [`mid_anchor_frame`].
    pub mid_frame_at: Option<usize>,
    /// Extra appended guiding blocks for LONG-FORM windows, as `(local pixel
    /// frame, image path)` pairs in the WINDOW's own coordinate frame: an
    /// anchor whose instant lies beyond this window's own decoded span,
    /// conditioned so the window bends toward it instead of running wherever
    /// its prompt alone takes it. The guide mechanism is
    /// [`conditioned_latent`]'s appended block, whose RoPE position is the
    /// raw pixel-frame offset the reference's `frame_idx` is - bounded by
    /// nothing - so a guide may name an instant the window's own tokens never
    /// reach. Set only by [`window_gen_opts`], which routes the clip-global
    /// anchors per window; a single-clip request carries none, and a caller
    /// setting this by hand must keep the positions inside the window's own
    /// coordinate frame and in ascending order. Each guide costs one latent
    /// frame's worth of tokens on top of the window's own planned budget -
    /// the window ceilings bound the PLAN, not the conditioning riding with
    /// it.
    pub future_anchors: Vec<(usize, String)>,
    /// `ImageConditioningInput.strength` for every given still - `1.0` pins
    /// the conditioned tokens to the encoded image exactly, `0.0` ignores
    /// them. See [`conditioned_latent`]'s doc for the two places it lands
    /// and why the reference makes it a REQUIRED per-image CLI argument
    /// (`--image PATH FRAME_IDX STRENGTH`, its own help example: `0.8`).
    ///
    /// `1.0` is this crate's default because "hold exactly this frame" is
    /// what `--start-frame` alone promises, and because it is the value the
    /// port used before this became a parameter at all. Lowering it does
    /// visibly change the output (mean |delta| 6.4 at 0.8, 14.3 at 0.5,
    /// against the same run at 1.0) - but it is NOT a motion knob: see
    /// `end_frame`'s doc for the one thing that is.
    pub conditioning_strength: f32,
    /// Which physical card each stage of this generation runs on - see
    /// [`crate::devplan`]. [`DevicePlan::Auto`] (the default) runs the
    /// conditional and unconditional DiT forwards of every CFG step
    /// concurrently on two cards when the machine has two schedulable ones,
    /// and is byte-for-byte the old single-device behaviour when it does not.
    ///
    /// It affects placement and nothing else: the two forwards are
    /// independent computations over independent inputs, so the folded
    /// velocity is bit-identical either way (gated by
    /// `crates/ltxv/tests/it/cfg_parallel.rs`).
    pub devices: crate::devplan::DevicePlan,
    /// LoRA adapters every denoiser forward applies, in the reference's own
    /// ComfyUI layout (the official trainer's and the published IC-LoRAs'
    /// format), each at its own strength - see [`crate::lora::runtime_adapter`].
    /// The checkpoint's weights are never rewritten: quantized blocks run the
    /// correction beside their packed weights, fp32 weights fold it exactly.
    pub adapters: Vec<model::adapter::AdapterSpec>,
    /// An IC-LoRA reference video (`--reference-video`): appended to every
    /// in-context stage as clean tokens at their own positions, for an
    /// adapter in [`Self::adapters`] trained to read them - see
    /// [`crate::refcond`]. Needs an adapter: the base checkpoint alone does
    /// not know what to do with them.
    pub reference: Option<VideoReference>,
    /// Keep the adapters and the reference on the refinement stage of a
    /// two-stage run too (`--stage-2-ic-lora`). Off by default, as in the
    /// reference's own IC-LoRA recipe: stage 1 runs in context, stage 2 runs
    /// the bare checkpoint.
    pub refine_in_context: bool,
    /// How the generation runs ([`Runtime`]).
    pub runtime: Runtime,
}

/// A reference video for IC-LoRA conditioning: the decoded clip, how
/// strongly its tokens are held (`1.0` keeps them perfectly clean), and how
/// strongly they and the generated tokens attend to each other
/// (`ICLoraPipeline`'s `conditioning_attention_strength` and
/// `conditioning_attention_mask`).
#[derive(Clone)]
pub struct VideoReference {
    pub clip: Video,
    pub strength: f32,
    /// The attention weight, in `[0, 1]`: `1.0` attends fully, `0.0` cuts
    /// the reference off from the clip. Scales [`Self::attention_mask`] when
    /// one is given.
    pub attention_strength: f32,
    /// A grayscale mask video setting the weight per region and frame
    /// (white attends, black does not): its first `frames` frames, at half
    /// the output resolution as the reference loads it, averaged onto the
    /// reference's latent grid.
    pub attention_mask: Option<Video>,
}

impl VideoReference {
    /// A reference at full attention strength and no mask.
    pub fn new(clip: Video, strength: f32) -> VideoReference {
        VideoReference { clip, strength, attention_strength: 1.0, attention_mask: None }
    }
}

impl std::fmt::Debug for VideoReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoReference")
            .field("frames", &self.clip.frames.len())
            .field("width", &self.clip.width)
            .field("height", &self.clip.height)
            .field("strength", &self.strength)
            .field("attention_strength", &self.attention_strength)
            .field("attention_mask_frames", &self.attention_mask.as_ref().map(|m| m.frames.len()))
            .finish()
    }
}

impl Default for GenOpts {
    /// A clip small enough to run as a smoke test in seconds: 9 frames (2
    /// latent frames), 64x64 (2x2 latent grid) = 8 DiT tokens, 4 steps.
    fn default() -> GenOpts {
        GenOpts {
            frames: 9,
            width: 64,
            height: 64,
            steps: 4,
            guidance: crate::guidance::Guidance::default(),
            negative_prompt: crate::guidance::DEFAULT_NEGATIVE_PROMPT.into(),
            seed: data::rng::random_seed(),
            fps: 8,
            base_shift: 0.95,
            max_shift: 2.05,
            stretch: true,
            terminal: 0.1,
            sampler: Sampler::Euler,
            eta: Sampler::Euler.default_eta(),
            s_noise: 1.0,
            context_len: 8,
            dit_config: "tiny".to_string(),
            audio: false,
            audio_source: None,
            device: None,
            start_frame: None,
            end_frame: None,
            mid_frame: None,
            mid_frame_at: None,
            future_anchors: Vec::new(),
            conditioning_strength: 1.0,
            devices: crate::devplan::DevicePlan::default(),
            adapters: Vec::new(),
            reference: None,
            refine_in_context: false,
            runtime: Runtime::default(),
        }
    }
}

/// A generated clip: `frames` interleaved RGB8 images, each `width * height *
/// 3` bytes - the same shape `wan::pipeline::Video` uses (this crate has no
/// imaging dependency; the CLI turns frames into a container, see
/// `crates/cli/src/ltxv_cli.rs`).
#[derive(Clone)]
pub struct Video {
    pub width: u32,
    pub height: u32,
    pub fps: usize,
    pub frames: Vec<Vec<u8>>,
    /// The clip's own sound track, generated by the SAME forward pass that
    /// generated the frames and covering the same time window - `None` when
    /// `GenOpts::audio` was off, which is the only reason a clip is silent.
    pub audio: Option<crate::audio::AudioClip>,
}

/// Per-phase wall clock.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub build_dit: f32,
    /// Prompt -> text context: the whole [`real_text_context`] call, or 0
    /// when the stub context stands in for it.
    ///
    /// This field exists because its absence was a real, expensive defect,
    /// not because a breakdown is nice to have. Every previously published
    /// number for this pipeline printed `build + denoise + vae` against a
    /// wall clock those three summed to roughly half of, and the missing
    /// half WAS this stage - so two optimization passes went into the
    /// second-largest stage while the largest had never been measured once.
    /// A breakdown must either account for its own total or name what it is
    /// missing; [`Timings::unattributed`] is the other half of that rule.
    pub text_encode: f32,
    pub denoise: f32,
    pub decode: f32,
    /// Audio latent -> waveform: the audio VAE decoder plus the vocoder.
    /// Zero on a silent generation. Its own row rather than folded into
    /// `decode`, because the two stages have completely different fixes and
    /// a breakdown that hid one inside the other is what this struct's own
    /// `text_encode` field exists to prevent happening again.
    pub audio_decode: f32,
    pub steps: usize,
    pub tokens: usize,
    pub forwards_per_step: usize,
}

impl Timings {
    /// Everything this struct actually attributes. Compare against a caller's
    /// own wall clock via [`Self::unattributed`] rather than presenting this
    /// as the run's total.
    pub fn total(&self) -> f32 {
        self.build_dit + self.text_encode + self.denoise + self.decode + self.audio_decode
    }

    /// The part of `wall` no field here explains - printed as its own row so
    /// a stage nobody has instrumented yet is visible instead of invisible.
    pub fn unattributed(&self, wall: f32) -> f32 {
        (wall - self.total()).max(0.0)
    }

    pub fn secs_per_forward(&self) -> f32 {
        let n = (self.steps * self.forwards_per_step).max(1);
        self.denoise / n as f32
    }
}

/// FNV-1a, 64-bit - folds the prompt string into the noise/context seed (see
/// this module's doc: there is no real text encoder, so this is the whole
/// extent to which the prompt affects the output). Not cryptographic, not
/// meant to be; just a cheap deterministic mix.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// What the AUDIO stream's noise seeds are derived from, so one `--seed`
/// reproduces a whole audio-visual run while the two streams never draw the
/// same numbers. `"AUDIO"`, which is also what makes it recognisable in a
/// trace.
const AUDIO_SEED_SALT: u64 = 0x41_55_44_49_4f;

/// Salt for the NA/diffusion VAE decoder's own pixel-space noise draw, so
/// one `--seed` still reproduces a whole run without the decode reusing the
/// exact stream the DiT's latent noise already consumed. Only
/// [`crate::import::VideoVaeArch::NaDiffusion`] draws at all - the conv
/// decoder is deterministic in the latent (see [`decode_video`]).
const VAE_DECODE_SEED_SALT: u64 = 0x5641_455f_4445_4300;

/// Standard-normal noise from a seeded SplitMix64 stream - the same
/// generator `wan::pipeline::seeded_noise` uses, deliberately not torch's
/// Philox (see that function's doc for why bit-for-bit reproduction was
/// never a goal here).
fn seeded_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Rng::new(seed);
    (0..n).map(|_| rng.next_gaussian() as f32).collect()
}

/// A deterministic, meaningless `[context_len, dim]` text-context stand-in -
/// see this module's doc for why. `0.5*N(0,1)`, matching the scale
/// `tools/goldens/ltxv_dit_dump_reference.py`'s own fake context uses, so the
/// magnitude the cross-attention sees is at least in the range a real
/// projected caption would be.
pub fn context_stub(context_len: usize, dim: usize, seed: u64) -> Vec<f32> {
    seeded_noise(context_len * dim, seed).into_iter().map(|v| 0.5 * v).collect()
}

/// Round `n` up to a multiple of `cfg.connector_num_learnable_registers`
/// when `cfg.use_embeddings_connector` is set - `crate::block::
/// EmbeddingsConnector::forward` requires its own input sequence length to
/// be an EXACT multiple of its own register count (register substitution
/// tiles the registers round-robin over invalid positions, see that
/// function's doc). [`GenOpts::context_len`]'s stub size is not naturally
/// shaped that way, which is what this function is actually for.
///
/// **Not used for the real Gemma-4 encoder path** - an earlier version of
/// this doc claimed a real tokenizer's prompt length is not naturally
/// register-shaped either, and used THAT as the justification for reusing
/// this same round-up here too. That premise was never checked against the
/// reference and was wrong: `resources/ltxv/source/.../gemma_assets.py`'s
/// `TOKENIZER_MAX_LENGTH = 1024` is a FIXED length the reference tokenizer
/// pads (or truncates) every prompt to, regardless of the prompt's own
/// token count - `PromptEncoder.__call__` -> `text_encoder.encode()`'s own
/// doc: every prompt's per-item slice stays `[1, 1024, D]`, never trimmed
/// down to its real content length, and `EmbeddingsProcessor.
/// process_hidden_states` runs the connector over that full width. Rounding
/// a SHORT prompt's own token count up to the nearest multiple of 128
/// (e.g. 128 for a 20-token prompt) silently fed the connector - and every
/// downstream cross-attention softmax - a context roughly 8x shorter than
/// the real checkpoint was ever calibrated against. See
/// `GEMMA4_MAX_PROMPT_TOKENS`/[`real_text_context`] for the fix; this
/// function now only backs the stub/testing path.
fn padded_context_len(cfg: &LtxDitConfig, n: usize) -> usize {
    if cfg.use_embeddings_connector && cfg.connector_num_learnable_registers > 0 {
        let m = cfg.connector_num_learnable_registers as usize;
        n.div_ceil(m).max(1) * m
    } else {
        n
    }
}

/// Real Gemma-4 text conditioning, replacing [`context_stub`] when
/// [`Paths::text_encoder`] is set: open the checkpoint for streaming, extract
/// its embedded tokenizer (`gemma4::tokenizer`'s doc - it is the checkpoint's
/// own `tokenizer_json` tensor, not a separate file), tokenize `prompt`
/// (and `negative`, the negative prompt, for the unconditional branch - only
/// when some stream guides against it), run `gemma4::forward_streamed`, and project the full
/// `hidden_states` tuple through `gemma4::AggregateEmbed`'s video head -
/// the real `[t, cross_attention_dim]` context `LtxDit::forward`/
/// `crate::dit::forward_q_streamed` expect.
///
/// The projection is not a bare `Linear`: see
/// `gemma4::AggregateEmbed::forward`'s doc for the per-token/per-state RMS
/// normalization, interleaved column order and rescale that
/// `ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2`
/// applies around it, and for what this pipeline produced while they were
/// missing. The `<bos>` this function's `tokenize` prepends belongs to the
/// same reference-fidelity fix.
///
/// Returns `(ctx_cond, ctx_uncond, context_valid, context_len)`.
/// `context_len` is the reference's own fixed Gemma-4 tokenizer width
/// (`GEMMA4_MAX_PROMPT_TOKENS`'s doc - `TOKENIZER_MAX_LENGTH = 1024`, NOT a
/// multiple of `dit_cfg.connector_num_learnable_registers` derived from the
/// prompt's own token count, an earlier and unverified assumption) when
/// `dit_cfg.use_embeddings_connector` is set (NOT [`GenOpts::context_len`],
/// which only sizes [`context_stub`]'s fake context). `crate::block::
/// EmbeddingsConnector::forward` still requires its own input length to be
/// an exact multiple of its register count (register substitution tiles the
/// registers round-robin over the INVALID positions - see that function's
/// doc); `1024` happens to already be an exact multiple of the real
/// checkpoint's `connector_num_learnable_registers` (128), so no further
/// rounding is needed on top of it. Each branch is framed by
/// [`frame_prompt`] - right-padded, as the reference's connector reads it - and carries
/// its own mask of real (`1.0`) vs padded (`0.0`) positions, so the
/// connector substitutes its learnable registers into exactly that branch's
/// padding rather than the DiT reading zeros as caption content. The padded
/// rows are zero because the connector rewrites them before any block reads
/// them (see `EmbeddingsConnector::forward`'s step 1). Without a
/// negative-prompt pass the unconditional branch is an all-zero vector of
/// the same shape that nothing reads.
///
/// What an encode hands the denoise loop, and what [`crate::text_cache`]
/// stores.
///
/// The audio pair is `Some` only when the caller asked for sound. It is a
/// SEPARATE projection of the same text tower output rather than a reshape of
/// the video one - the checkpoint carries
/// `text_embedding_projection.audio_aggregate_embed` next to the video head,
/// and the audio stream's own embeddings connector is built for that head's
/// narrower output width. Feeding the video context to the audio connector is
/// not a size mismatch that fails loudly in every case; at the same width it
/// would silently condition the sound on the wrong projection.
#[derive(Clone, Debug, Default)]
pub struct TextContext {
    pub cond: Vec<f32>,
    pub uncond: Vec<f32>,
    /// Which of `cond`'s rows are the prompt's own - see [`frame_prompt`].
    pub valid: Vec<f32>,
    /// The same for `uncond`, whose prompt is the negative one: each branch
    /// is masked by its own token count, as the reference encodes them.
    pub uncond_valid: Vec<f32>,
    pub len: usize,
    pub a_cond: Option<Vec<f32>>,
    pub a_uncond: Option<Vec<f32>>,
}

#[tracing::instrument(level = "info", name = "text_encode", skip_all, fields(prompt_chars = prompt.len(), negative = negative.is_some()))]
fn real_text_context(path: &str, prompt: &str, dit_cfg: &LtxDitConfig, negative: Option<&str>, device: Option<&str>, want_audio: bool, runtime: &Runtime) -> Result<TextContext, String> {
    let cross_attention_dim = dit_cfg.cross_attention_dim as usize;

    let cfg = gemma4::Gemma4Config::gemma4_12b();
    let hidden = cfg.hidden_size as usize;
    let n_states = cfg.num_hidden_layers as usize + 1;

    // A `.gguf` here is a quantized text tower produced by `brain quantize`.
    // It is read on demand, one layer at a time, and its projections run in
    // int8 where the device has a packed-dot path - so this branch is both
    // about half the bytes of the bf16 safetensors and a different tier of
    // arithmetic. A `.safetensors` path keeps the original behaviour exactly.
    let quantized = path.ends_with(".gguf");
    // `Runtime::text_precision` forces a tier: fp32 over a quantized
    // checkpoint compares the two tiers' output on the SAME file (the only
    // cheap whole-encoder fp32-vs-int8 number, since both reads hit the same
    // bytes) or serves a platform whose fast path is fp32. `Precision::
    // for_device` still has the last word in the other direction - it can
    // refuse int8, never impose it.
    let precision = runtime.text_precision.unwrap_or(if quantized { gemma4::Precision::Int8 } else { gemma4::Precision::Fp32 });

    // The cache is checked BEFORE anything is read. On a hit this whole
    // stage - the largest in the pipeline - costs a few megabytes of file
    // read, because the encode is a pure function of the key below.
    let (encoder_len, encoder_mtime) = crate::text_cache::encoder_identity(path);
    let cache_key = crate::text_cache::Key {
        prompt: prompt.to_string(),
        encoder_path: path.to_string(),
        encoder_len,
        encoder_mtime,
        precision: format!("{precision:?}"),
        cross_attention_dim,
        connector_registers: dit_cfg.connector_num_learnable_registers,
        use_connector: dit_cfg.use_embeddings_connector,
        negative_prompt: negative.map(str::to_string),
        audio: want_audio,
        encode_revision: crate::text_cache::ENCODE_REVISION,
    };
    // A silent request is also answered by the entry an audio request (or a
    // duration prediction) wrote: the same encode plus one more projection.
    let hit = crate::text_cache::load(&cache_key, runtime.text_cache).or_else(|| {
        let with_audio = crate::text_cache::Key { audio: true, ..cache_key.clone() };
        (!want_audio).then(|| crate::text_cache::load(&with_audio, runtime.text_cache)).flatten().map(|hit| crate::text_cache::Encoded { a_ctx_cond: None, a_ctx_uncond: None, ..hit })
    });
    if let Some(hit) = hit {
        return Ok(TextContext { cond: hit.ctx_cond, uncond: hit.ctx_uncond, valid: hit.context_valid, uncond_valid: hit.uncond_valid, len: hit.context_len, a_cond: hit.a_ctx_cond, a_uncond: hit.a_ctx_uncond });
    }

    // Sub-stage timings, not one opaque block. The stage as a whole was
    // unmeasured until recently and turned out to be the largest in the
    // pipeline; measuring only its total would repeat that mistake one level
    // down, because "read the checkpoint" and "run the forward" have
    // completely different fixes and only a split says which one to attack.
    let source_bytes: u64 = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let t_read = std::time::Instant::now();

    // Both file types are read on demand, one layer at a time, through the
    // same streamed forward: a `.gguf` (`brain quantize`'s output) and the
    // `.safetensors` the release ships, memory-mapped in place. Neither is
    // ever materialized whole, so a text encode reads each layer once and
    // holds one layer at a time.
    //
    // The AUDIO head is loaded only when it will be used: it is a second
    // projection of the same width class as the video one, so building it
    // unconditionally would add its whole parameter count to every silent
    // generation's peak for nothing.
    let (encoder, tok): (Box<dyn checkpoint::TensorSource>, _) = if quantized {
        let src = gemma4::Gemma4GgufSource::open(path, &cfg)?;
        let tok = src.tokenizer()?;
        (Box::new(src), tok)
    } else {
        (Box::new(gemma4::open_safetensors(path, &cfg)?), gemma4::safetensors_tokenizer(path)?)
    };
    let t_agg_load = std::time::Instant::now();
    let agg = aggregate_head_from_source(encoder.as_ref(), hidden, n_states, "video")?;
    let agg_a = want_audio.then(|| aggregate_head_from_source(encoder.as_ref(), hidden, n_states, "audio")).transpose()?;
    tracing::info!(secs = t_agg_load.elapsed().as_secs_f32(), audio = want_audio, "aggregate-embed head(s) loaded");
    let read_s = t_read.elapsed().as_secs_f32();
    tracing::info!(
        secs = read_s,
        bytes = source_bytes,
        mib_per_s = source_bytes as f32 / read_s.max(1e-6) / (1024.0 * 1024.0),
        quantized,
        "text encoder opened"
    );

    // `resources/ltxv/source/packages/ltx-core/src/ltx_core/text_encoders/
    // gemma/gemma_assets.py::TOKENIZER_MAX_LENGTH = 1024`, `truncation=True`
    // in `LTXGemmaTokenizer.tokenize_with_weights` - the reference truncates
    // every prompt to at most 1024 tokens BEFORE encoding, never fewer, see
    // `GEMMA4_CONTEXT_LEN`'s doc for why the padded side of this also has to
    // be exactly 1024, not a shape-derived guess.
    const GEMMA4_MAX_PROMPT_TOKENS: usize = 1024;
    // **The leading `<bos>` is the reference's, not an embellishment.**
    // `LTXGemmaTokenizer.tokenize_with_weights` (`ltx_core/text_encoders/
    // gemma/tokenizer.py`) prepends it unconditionally, and says why in its
    // own class doc: "Gemma 3 already emits it via post_processor; Gemma 4
    // does not, so we prepend". `encode_raw` adds no token, so it is added
    // here; without it every caption token sits one position off from where
    // the checkpoint was trained to read it.
    let bos_id = tok.bos_id();
    let tokenize = |s: &str| -> Vec<u32> {
        // `text.strip()`, same line of the reference.
        let mut ids = tok.encode_raw(s.trim());
        if ids.first() != Some(&bos_id) {
            ids.insert(0, bos_id);
        }
        // Never empty: an empty prompt is `<bos>` alone.
        ids.truncate(GEMMA4_MAX_PROMPT_TOKENS);
        ids
    };

    // One closure covers both loaders and both prompts, so the conditional
    // and unconditional branches cannot drift apart.
    //
    // Both heads read the SAME `hidden_states`, so the audio projection costs
    // one extra matrix product, never a second 12B tower forward - which is
    // exactly the reference's own arrangement (`FeatureExtractorV2.forward`
    // returns both from one pass).
    let encode = |ids: &[u32]| -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
        let n = ids.len();
        let t_fwd = std::time::Instant::now();
        let hidden_states = gemma4::forward_streamed(&cfg, encoder.as_ref(), device, precision, ids)?.hidden_states;
        tracing::info!(secs = t_fwd.elapsed().as_secs_f32(), tokens = n, layers = cfg.num_hidden_layers, ?precision, "text tower forward");
        let t_agg = std::time::Instant::now();
        let out = agg.forward(&hidden_states, n, hidden);
        let out_a = agg_a.as_ref().map(|a| a.forward(&hidden_states, n, hidden));
        tracing::info!(secs = t_agg.elapsed().as_secs_f32(), tokens = n, in_dim = hidden * n_states, audio = out_a.is_some(), "aggregate-embed projection");
        Ok((out, out_a))
    };

    let ids_cond = tokenize(prompt);
    let n_cond = ids_cond.len();
    // The reference's own fixed tokenizer width (`GEMMA4_MAX_PROMPT_TOKENS`'s
    // doc), not a rounded-up multiple of the connector's register count -
    // `padded_context_len` is for the stub/testing path only, see its doc.
    // A disabled connector still reads `context` as real caption content at
    // its own natural length, matching `padded_context_len`'s existing
    // pass-through behavior in that case.
    let context_len = if dit_cfg.use_embeddings_connector { GEMMA4_MAX_PROMPT_TOKENS } else { n_cond };
    let (raw_cond, raw_cond_a) = encode(&ids_cond)?;
    if raw_cond.len() != n_cond * cross_attention_dim {
        return Err(format!(
            "ltxv real text encoder: aggregate-embed produced {} values, expected {} ({n_cond} tokens x {cross_attention_dim} cross_attention_dim) - checkpoint/config mismatch",
            raw_cond.len(),
            n_cond * cross_attention_dim
        ));
    }
    // One framing for both streams and both branches, so a width that
    // differs cannot make the four paths differ in any other way.
    let framed = |raw: &[f32], rows: usize, dim: usize| frame_prompt(raw, rows, dim, context_len).0;
    let (ctx_cond, context_valid) = frame_prompt(&raw_cond, n_cond, cross_attention_dim, context_len);
    let a_dim = raw_cond_a.as_ref().map(|v| v.len() / n_cond.max(1));
    let a_ctx_cond = raw_cond_a.as_ref().zip(a_dim).map(|(v, d)| framed(v, n_cond, d));

    let (ctx_uncond, uncond_valid, a_ctx_uncond) = match negative {
        Some(negative) => {
            let ids_u = tokenize(negative);
            let (raw_u, raw_u_a) = encode(&ids_u)?;
            let (v, valid) = frame_prompt(&raw_u, ids_u.len(), cross_attention_dim, context_len);
            let a = raw_u_a.as_ref().zip(a_dim).map(|(x, d)| framed(x, ids_u.len(), d));
            (v, valid, a)
        }
        // The all-zero stand-in, at each stream's own width - never the
        // video vector reused, which would be a shape error on the audio side
        // and a wrong conditioning if the two widths ever coincided. No pass
        // reads it: the negative-prompt pass is exactly what is not run.
        None => (vec![0f32; context_len * cross_attention_dim], context_valid.clone(), a_dim.map(|d| vec![0f32; context_len * d])),
    };

    crate::text_cache::store(
        &cache_key,
        &crate::text_cache::Encoded {
            ctx_cond: ctx_cond.clone(),
            ctx_uncond: ctx_uncond.clone(),
            context_valid: context_valid.clone(),
            uncond_valid: uncond_valid.clone(),
            context_len,
            a_ctx_cond: a_ctx_cond.clone(),
            a_ctx_uncond: a_ctx_uncond.clone(),
        },
        runtime.text_cache,
    );
    Ok(TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, uncond_valid, len: context_len, a_cond: a_ctx_cond, a_uncond: a_ctx_uncond })
}

/// `rows` encoded `[rows, dim]` prompt tokens framed into the connector's
/// fixed `[context_len, dim]` input, with the mask of which rows are the
/// prompt's own.
///
/// RIGHT-padded: the prompt fills the FIRST `rows` positions and the padding
/// the connector replaces with its registers follows. That is the layout the
/// reference's connector reads, whatever side its tokenizer pads on -
/// `EmbeddingsProcessor.create_embeddings` stably sorts the valid tokens
/// before the pads ahead of both connectors. The side is not cosmetic: the
/// connector substitutes the register tiled at each padded position's own
/// index and runs rotary attention over raw sequence indices, so the prompt
/// framed at the other end comes out a different conditioning (row cosines
/// down to -0.4 against the reference). A prompt longer than the frame keeps
/// its first `context_len` tokens, as the reference truncates.
fn frame_prompt(raw: &[f32], rows: usize, dim: usize, context_len: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = rows.min(context_len);
    let mut framed = vec![0f32; context_len * dim];
    framed[..rows * dim].copy_from_slice(&raw[..rows * dim]);
    let mut valid = vec![0f32; context_len];
    valid[..rows].fill(1.0);
    (framed, valid)
}

/// `gemma4::AggregateEmbed` over a streaming source. The eager path builds
/// this from a whole-model map it already has; a streamed one has to pull
/// the head's own two tensors, which is the ONE place a `TensorSource` is
/// asked for something outside the layer loop.
fn aggregate_head_from_source(src: &dyn checkpoint::TensorSource, hidden: usize, n_states: usize, stream: &str) -> Result<gemma4::AggregateEmbed, String> {
    let get = |name: &str| -> Result<Vec<f32>, String> {
        let mut out = None;
        if !src.with_tensor(name, &mut |d| out = Some(d.to_vec())) {
            return Err(format!("ltxv real text encoder: text encoder has no tensor {name}"));
        }
        Ok(out.expect("with_tensor reported found, so the callback ran"))
    };
    let weight = get(&format!("text_embedding_projection.{stream}_aggregate_embed.weight"))?;
    let bias = get(&format!("text_embedding_projection.{stream}_aggregate_embed.bias"))?;
    let out_dim = bias.len();
    Ok(gemma4::AggregateEmbed::new(weight, bias, hidden * n_states, out_dim))
}

/// `[3, T, 2]` row-major RoPE position bounds for a `(f, h, w)` latent grid -
/// `ltx_core.components.patchifiers.VideoLatentPatchifier.
/// get_patch_grid_bounds`'s construction at `patch_size=1` (latent-token
/// space): integer grid coordinates, `end = start + 1`, flattened `(f h w)`
/// (frame outermost, width innermost) - the SAME token order
/// [`tc_to_chw`]/[`chw_to_tc`] assume and the golden dumper's own
/// `det_video_modality` builds (`torch.meshgrid(..., indexing="ij")` then
/// flatten in that order).
pub fn grid_positions(f: usize, h: usize, w: usize) -> Vec<f32> {
    let t = f * h * w;
    let mut out = vec![0f32; 3 * t * 2];
    let mut tok = 0usize;
    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                for (axis, v) in [fi, hi, wi].into_iter().enumerate() {
                    out[(axis * t + tok) * 2] = v as f32;
                    out[(axis * t + tok) * 2 + 1] = v as f32 + 1.0;
                }
                tok += 1;
            }
        }
    }
    out
}

/// The real VAE's downsample factors (`ltx_core.types.SpatioTemporalScaleFactors.
/// default()`): 1 latent frame covers 8 pixel frames (except the first,
/// see [`real_pixel_positions`]'s doc), 1 latent cell covers 32x32 pixels.
pub const VAE_TEMPORAL_SCALE: usize = 8;
pub const VAE_SPATIAL_SCALE: usize = 32;

/// `[3, T, 2]` row-major RoPE position bounds for a `(f, h, w)` latent grid,
/// in the REAL production pipeline's own units - `ltx_core.tools.
/// VideoLatentTools.create_initial_state`'s `get_pixel_coords(latent_coords,
/// scale_factors, causal_fix=True)` followed by `positions[:, 0, ...] /=
/// fps`, NOT [`grid_positions`]'s raw latent-grid integers.
///
/// This is a REAL, confirmed correctness gap [`grid_positions`] and this
/// pipeline's earlier real-weight tests never caught: `LTXModel.forward`
/// runs the same RoPE formula on whatever position values it is given, so
/// feeding it raw latent-grid integers instead of these pixel-scaled ones
/// produces no crash and no NaN - only a garbled, physically-meaningless
/// spatial/temporal coordinate system for a model whose
/// `positional_embedding_max_pos: [20, 2048, 2048]` (pixel-scale maximums,
/// not latent-grid-scale ones - a 2048 latent-index maximum would be absurd
/// for a model this size) was calibrated against the real ones. Every
/// earlier real-weight DiT parity gate in this crate compared against a
/// golden built from RAW meshgrid positions on BOTH sides (`ltxv_real_dit_
/// dump_reference.py`'s own `det_video_modality`, a deliberate scope cut
/// documented there as proving PORT correctness given arbitrary positions,
/// not proving `generate`'s own position CONSTRUCTION) - so this gap was
/// invisible to every cosine-similarity check run so far, on either side.
///
/// Height/width axes: `[hi*32, (hi+1)*32)` / `[wi*32, (wi+1)*32)` - a
/// straight latent-to-pixel scale, no causal fix (only the temporal axis is
/// causal). Frame axis: `get_pixel_coords`'s `causal_fix=True` branch
/// rewrites `[fi*8, (fi+1)*8)` to `[max(0, fi*8+1-8), max(0, (fi+1)*8+1-8))`
/// BEFORE the `/fps` divide - this is not an approximation, it is exactly
/// why `fi=0` maps to `[0, 1)` (the causal VAE's own first-latent-frame ==
/// 1-pixel-frame rule) while every later `fi` maps to a genuine 8-pixel-frame
/// window (`[fi*8-7, fi*8+1)`), both then divided by `fps`.
pub fn real_pixel_positions(f: usize, h: usize, w: usize, fps: f64) -> Vec<f32> {
    let t = f * h * w;
    let mut out = vec![0f32; 3 * t * 2];
    let mut tok = 0usize;
    for fi in 0..f {
        let (pixel_start, pixel_end) = ((fi * VAE_TEMPORAL_SCALE) as f64, ((fi + 1) * VAE_TEMPORAL_SCALE) as f64);
        let (fixed_start, fixed_end) = ((pixel_start + 1.0 - VAE_TEMPORAL_SCALE as f64).max(0.0), (pixel_end + 1.0 - VAE_TEMPORAL_SCALE as f64).max(0.0));
        let (f_start, f_end) = (fixed_start / fps, fixed_end / fps);
        for hi in 0..h {
            for wi in 0..w {
                let axis_vals = [(f_start, f_end), ((hi * VAE_SPATIAL_SCALE) as f64, ((hi + 1) * VAE_SPATIAL_SCALE) as f64), ((wi * VAE_SPATIAL_SCALE) as f64, ((wi + 1) * VAE_SPATIAL_SCALE) as f64)];
                for (axis, &(s, e)) in axis_vals.iter().enumerate() {
                    out[(axis * t + tok) * 2] = s as f32;
                    out[(axis * t + tok) * 2 + 1] = e as f32;
                }
                tok += 1;
            }
        }
    }
    out
}

/// `[3, lh*lw, 2]` row-major RoPE position bounds for a single-pixel-frame
/// keyframe/conditioning image inserted at pixel-frame `frame_idx` within a
/// clip's own timeline - `ltx_core.conditioning.types.keyframe_cond.
/// VideoConditionByKeyframeIndex.apply_to`'s own position formula
/// (`get_pixel_coords` on the item's own local `[0,1)` latent bound, with
/// `causal_fix = (frame_idx == 0)`, then `+= frame_idx`, then narrowed to
/// `[start, start+1)` since this crate only ever inserts whole-pixel-frame
/// still images, then `/= fps`). Working through both branches of that
/// `causal_fix` conditional by hand, they converge to the SAME final
/// result for a single-pixel-frame item: `[frame_idx/fps, (frame_idx+1)/
/// fps)` on the frame axis, unconditionally - the branch only matters for
/// an INTERMEDIATE representation the final narrow-to-one-frame step
/// erases. [`real_pixel_positions`]`(1, lh, lw, fps)` (this crate's
/// existing "own video's frame 0" builder) is exactly this function called
/// at `frame_idx = 0` - both compute `[0, 1/fps)` - but `real_pixel_
/// positions` cannot express `frame_idx > 0` (its own multi-frame loop
/// always applies the causal fix, which is only correct for a video's own
/// sequential frame axis, not an independently-inserted keyframe elsewhere
/// in the timeline).
pub fn keyframe_conditioning_positions(pixel_frame_idx: usize, lh: usize, lw: usize, fps: f64) -> Vec<f32> {
    let t = lh * lw;
    let mut out = vec![0f32; 3 * t * 2];
    let (f_start, f_end) = (pixel_frame_idx as f64 / fps, (pixel_frame_idx + 1) as f64 / fps);
    let mut tok = 0usize;
    for hi in 0..lh {
        for wi in 0..lw {
            let axis_vals = [(f_start, f_end), ((hi * VAE_SPATIAL_SCALE) as f64, ((hi + 1) * VAE_SPATIAL_SCALE) as f64), ((wi * VAE_SPATIAL_SCALE) as f64, ((wi + 1) * VAE_SPATIAL_SCALE) as f64)];
            for (axis, &(s, e)) in axis_vals.iter().enumerate() {
                out[(axis * t + tok) * 2] = s as f32;
                out[(axis * t + tok) * 2 + 1] = e as f32;
            }
            tok += 1;
        }
    }
    out
}

#[cfg(test)]
mod keyframe_conditioning_positions_tests {
    use super::*;

    #[test]
    fn frame_idx_zero_matches_real_pixel_positions_own_frame_zero() {
        let (lh, lw, fps) = (2usize, 3usize, 8.0);
        let a = keyframe_conditioning_positions(0, lh, lw, fps);
        let b = real_pixel_positions(1, lh, lw, fps);
        assert_eq!(a, b, "frame_idx=0 must agree with real_pixel_positions(1, lh, lw, fps) exactly - both describe the same instant");
    }

    #[test]
    fn nonzero_frame_idx_is_a_plain_pixel_frame_over_fps_window() {
        let (lh, lw, fps) = (1usize, 1usize, 8.0);
        let p = keyframe_conditioning_positions(64, lh, lw, fps);
        // frame axis: [64/8, 65/8) = [8.0, 8.125)
        assert_eq!((p[0], p[1]), (8.0, 8.125));
        // height/width axes: still a plain 32-to-1 latent-to-pixel scale for the one token.
        assert_eq!((p[2], p[3]), (0.0, 32.0));
        assert_eq!((p[4], p[5]), (0.0, 32.0));
    }
}

/// Which pixel frame [`GenOpts::mid_frame`] anchors in a `frames`-long clip.
///
/// `at` is the caller's own index when they gave one. With none, the position
/// is the reference's own answer for a single INTERIOR keyframe:
/// `ltx_pipelines.utils.helpers.evenly_spaced_keyframe_positions(num_keyframes
/// = 1, num_frames)` is `torch.linspace(0, num_frames - 1, 3).round()[1:-1]`,
/// i.e. `(frames - 1) / 2` - `[60]` for a 121-frame clip. That division is
/// exact for every legal clip length, since `1 + 8k` makes `frames - 1` even,
/// so nothing here depends on which way a tie rounds.
///
/// **The position is NOT snapped to a latent-frame boundary, and that is the
/// reference's behaviour rather than an omission here.**
/// `VideoConditionByKeyframeIndex.apply_to` adds `frame_idx` straight onto the
/// RoPE time coordinate (`positions[:, 0, ...] += self.frame_idx`, then
/// `/= fps`) of an APPENDED token block - the guide never occupies a slot on
/// the generated video's latent grid, so there is no grid for it to land on.
/// The one pixel-to-latent mapping in the reference
/// (`ltx_pipelines.dfr_layout.pixel_to_latent_index`) *raises* on a position
/// that is not already on the x8 border rather than rounding to it, and it is
/// used only for DFR's own generated-keyframe grid. The `1 + 8k` rule
/// constrains the clip's LENGTH, which [`generate`] already enforces; it does
/// not constrain where a guide may point inside it.
///
/// Refused outside `0 < at < frames - 1`: frame 0 and frame `frames - 1` are
/// what `start_frame`/`end_frame` already name, and a clip needs at least
/// three frames to have an interior at all (the reference raises the same way,
/// `num_frames < num_keyframes + 2`).
pub fn mid_anchor_frame(frames: usize, at: Option<usize>) -> Result<usize, String> {
    if frames < 3 {
        return Err(format!("a {frames}-frame clip has no interior frame for a mid-frame anchor to sit at (it needs at least 3)"));
    }
    let Some(at) = at else {
        return Ok((frames - 1) / 2);
    };
    if at == 0 || at >= frames - 1 {
        return Err(format!("a mid-frame anchor at pixel frame {at} is not INSIDE a {frames}-frame clip: frame 0 is --start-frame's and frame {} is --end-frame's, so this one has to sit strictly between them", frames - 1));
    }
    Ok(at)
}

/// Which latent frame's own pixel span contains `pixel_frame` - the inverse of
/// [`real_pixel_positions`]' causal fix, where latent frame 0 covers exactly
/// one pixel frame and every later one covers [`VAE_TEMPORAL_SCALE`].
///
/// It names the instant of the clip a caller is pointing at - the log line
/// and the window plan both read it - and, since the mid anchor moved from
/// an appended guide to a pin, it is also the slot [`conditioned_latent`]
/// overwrites with the mid still (see that function's deviation note). The
/// pixel frame itself is still not snapped to an x8 border (see
/// [`mid_anchor_frame`]): the pin lands on the slot CONTAINING the named
/// instant, so the still's content renders across the slot's whole pixel
/// span.
pub fn latent_frame_containing(pixel_frame: usize) -> usize {
    if pixel_frame == 0 {
        0
    } else {
        (pixel_frame - 1) / VAE_TEMPORAL_SCALE + 1
    }
}

#[cfg(test)]
mod mid_anchor_frame_tests {
    use super::*;

    /// The default position is the reference's own interior keyframe position,
    /// and the latent frame it lands in is the one whose pixel span really
    /// contains it - checked against [`real_pixel_positions`]' own bounds so
    /// the two formulas cannot drift apart.
    #[test]
    fn the_default_position_is_the_references_own_single_interior_keyframe() {
        // `evenly_spaced_keyframe_positions(1, 121) == [60]`.
        assert_eq!(mid_anchor_frame(121, None), Ok(60));
        assert_eq!(mid_anchor_frame(9, None), Ok(4));
        assert_eq!(mid_anchor_frame(17, None), Ok(8));
        assert_eq!(mid_anchor_frame(3, None), Ok(1));

        // 121 frames is 16 latent frames; pixel frame 60 sits in latent frame
        // 8, whose span is [57, 65).
        assert_eq!(latent_frame_containing(60), 8);
        let fps = 8.0;
        let p = real_pixel_positions(16, 1, 1, fps);
        let (start, end) = (p[8 * 2] as f64 * fps, p[8 * 2 + 1] as f64 * fps);
        assert!((start..end).contains(&60.0), "latent frame 8 spans [{start}, {end}), which must contain pixel frame 60");
        // The two ends the other flags already name, for the same walk.
        assert_eq!(latent_frame_containing(0), 0);
        assert_eq!(latent_frame_containing(120), 15, "the last pixel frame of a 121-frame clip is the last of its 16 latent frames");
    }

    /// An explicit position is taken verbatim - no snapping to the x8 latent
    /// border, because the reference does not snap either (see
    /// [`mid_anchor_frame`]'s doc).
    #[test]
    fn an_explicit_position_is_taken_verbatim_and_only_the_ends_are_refused() {
        assert_eq!(mid_anchor_frame(121, Some(37)), Ok(37), "37 is not on the x8 border and is still accepted");
        assert_eq!(mid_anchor_frame(121, Some(1)), Ok(1));
        assert_eq!(mid_anchor_frame(121, Some(119)), Ok(119));
        assert!(mid_anchor_frame(121, Some(0)).is_err(), "frame 0 is --start-frame's");
        assert!(mid_anchor_frame(121, Some(120)).is_err(), "the last frame is --end-frame's");
        assert!(mid_anchor_frame(121, Some(500)).is_err(), "outside the clip");
        assert!(mid_anchor_frame(1, None).is_err(), "a one-frame clip has no interior");
    }
}

#[cfg(test)]
mod real_pixel_positions_tests {
    use super::*;

    #[test]
    fn frame_axis_is_causal_fixed_then_divided_by_fps() {
        let fps = 8.0;
        let p = real_pixel_positions(3, 1, 1, fps);
        // fi=0: pixel [0,8) -> causal_fix [max(0,-7), max(0,1)) = [0,1) -> /8 = [0, 0.125)
        assert_eq!((p[0], p[1]), (0.0, 0.125));
        // fi=1: pixel [8,16) -> causal_fix [max(0,1), max(0,9)) = [1,9) -> /8 = [0.125, 1.125)
        assert_eq!((p[2], p[3]), (0.125, 1.125));
        // fi=2: pixel [16,24) -> causal_fix [max(0,9), max(0,17)) = [9,17) -> /8 = [1.125, 2.125)
        assert_eq!((p[4], p[5]), (1.125, 2.125));
    }

    #[test]
    fn height_and_width_axes_are_a_straight_32x_latent_to_pixel_scale() {
        let p = real_pixel_positions(1, 2, 3, 8.0);
        let t = 2 * 3;
        // hi=0,wi=0 (tok 0): height [0,32), width [0,32)
        assert_eq!((p[t * 2], p[t * 2 + 1]), (0.0, 32.0));
        assert_eq!((p[2 * t * 2], p[2 * t * 2 + 1]), (0.0, 32.0));
        // hi=1,wi=2 (tok 5, the last one - h outer, w inner): height [32,64), width [64,96)
        assert_eq!((p[(t + 5) * 2], p[(t + 5) * 2 + 1]), (32.0, 64.0));
        assert_eq!((p[(2 * t + 5) * 2], p[(2 * t + 5) * 2 + 1]), (64.0, 96.0));
    }
}

/// VAE channel-major `[C, T, H, W]` -> DiT token-major `[T*H*W, C]` (unused
/// by [`generate`] directly - the noise latent is already generated
/// token-major - but kept for symmetry/testing and any future image-
/// conditioning path that would need to encode a real frame into the DiT's
/// input layout).
pub fn chw_to_tc(x: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let n_tok = t * h * w;
    assert_eq!(x.len(), c * n_tok, "chw_to_tc: {} values, expected {}", x.len(), c * n_tok);
    let mut out = vec![0f32; n_tok * c];
    for ci in 0..c {
        for tok in 0..n_tok {
            out[tok * c + ci] = x[ci * n_tok + tok];
        }
    }
    out
}

/// The inverse of [`chw_to_tc`]: DiT token-major `[T*H*W, C]` -> VAE
/// channel-major `[C, T, H, W]` - what [`generate`] applies once, right
/// before the VAE decode (see this module's doc).
pub fn tc_to_chw(x: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let n_tok = t * h * w;
    assert_eq!(x.len(), n_tok * c, "tc_to_chw: {} values, expected {}", x.len(), n_tok * c);
    let mut out = vec![0f32; c * n_tok];
    for tok in 0..n_tok {
        for ci in 0..c {
            out[ci * n_tok + tok] = x[tok * c + ci];
        }
    }
    out
}

/// The one VAE-decode call site both [`generate`] and [`generate_dfr`] use:
/// decode a `[C, lat_t, lh, lw]` latent through whichever decoder the loaded
/// checkpoint actually carries, taking the WHOLE-clip path when it fits and
/// the overlapping-tile path when it does not.
///
/// `arch` comes from [`crate::import::import_vae`], which detects it from
/// the file: the two real Lightricks LTX-2.5 video-VAE releases share one
/// conv encoder but ship structurally different decoders under the same
/// `decoder.*` name, and both are valid downloads. Both produce the same
/// `(1 + 8*(lat_t-1), lh*32, lw*32)` `[3, frames, h, w]` volume, which is
/// what lets one call site serve either without the caller knowing.
///
/// [`VideoVaeArch::Conv3d`] picks between the whole and tiled paths by
/// [`crate::vae3d::should_tile`]'s measured output-pixel-volume policy (with
/// [`Runtime::vae_tiling`] as the override). Every shape this port shipped
/// before that change stays on the exact whole path bit for bit; what the
/// tiled path adds is the shapes that used to abort with a `wgpu`
/// out-of-memory - 25 frames at 1080p above all, which is 52.2 Mpx against a
/// measured ~35 Mpx ceiling on a 24 GiB card.
///
/// [`VideoVaeArch::NaDiffusion`] has no tiled path at all - upstream's
/// `diffusion_tiling.py` overlapping-tile blend is not ported (see
/// [`crate::na_decoder`]'s header), so its own volume ceiling is a named
/// refusal from [`crate::na_decoder::check_decode`] rather than a silent
/// driver failure. That is why this function returns a `Result` where the
/// conv-only version could not fail.
///
/// Borrows `vweights`: the tiled path needs them across several graph builds
/// (one per distinct tile shape), and [`upscale`] decodes several segments of
/// one clip against the same weights, so neither can be handed ownership
/// without a ~3 GB host copy.
#[allow(clippy::too_many_arguments)]
fn decode_video(vcfg: &LtxVaeConfig, arch: VideoVaeArch, vweights: &vae::blocks::Tensors, lat_t: u32, lh: u32, lw: u32, latent: &[f32], o: &GenOpts) -> Result<(Vec<f32>, usize), String> {
    let (device, seed) = (o.device.as_deref(), o.seed ^ VAE_DECODE_SEED_SALT);
    let frames = 1 + 8 * (lat_t - 1);
    let (h, w) = (lh * 32, lw * 32);
    if let Some(path) = &o.runtime.latent_dump {
        crate::latentdump::dump(path, crate::latentdump::LatentShape { c: (latent.len() / (lat_t * lh * lw) as usize) as u32, t: lat_t, h: lh, w: lw }, latent);
    }
    if arch == VideoVaeArch::NaDiffusion {
        let cfg = crate::na_decoder::NaDecoderConfig::ltx25();
        let gpu = gpu_core::Gpu::open(device, &crate::na_decoder::KERNELS);
        tracing::info!(frames, h, w, "VAE decode: NA/diffusion decoder (whole clip, no tiling)");
        let px = crate::na_decoder::decode(&gpu, vweights, &cfg, latent, lat_t, lh, lw, seed)?;
        return Ok((px, frames as usize));
    }
    if crate::vae3d::should_tile(frames, h, w, o.runtime.vae_tiling) {
        let dec = crate::vae3d::LtxVaeTiledDecoder::auto(vcfg, vweights, lat_t, lh, lw, device);
        let n = dec.plan().tiles().len();
        tracing::info!(tiles = n, waste = dec.plan().overlap_waste(), frames, h, w, "VAE decode: tiled (whole-clip decode would not fit)");
        let px = dec.decode_with(latent, |done, total| tracing::debug!(done, total, "vae tile"));
        Ok((px, dec.frames() as usize))
    } else {
        let dec = LtxVaeDecoder::build(vcfg, vweights, lat_t, lh, lw, device);
        Ok((dec.decode(latent), dec.frames() as usize))
    }
}

/// Geometry (positions/masks) and initial content for appending one or more
/// `lh*lw`-token image-conditioning blocks after `base_t` noise tokens - one
/// block per `(pixel_frame_idx, cond_latent_tokens)` pair in `blocks`, each
/// with its OWN encoded content: `GenOpts::start_frame`/`end_frame` may be
/// the SAME image (the loop case: the generated content in between has to
/// connect the still to itself) or two DIFFERENT ones (a morph from one
/// still to another) - `ltx_core`'s `VideoConditionByKeyframeIndex(frame_idx,
/// strength=1.0)`, one instance per block, `marked=false` (an ordinary
/// image, not a generated-keyframe slot) is why `keyframes_mask` stays
/// all-zero across the APPENDED range, unlike [`crate::dfr::
/// keyframe_slots`]'s `marked=true` case. The BASE range keeps whatever
/// marker it already had: `ltx_core.conditioning.mask_utils.
/// extend_keyframes_mask` *extends* `latent_state.keyframes_mask`, it does
/// not rebuild it, and the base state's mask always carries
/// `VideoLatentTools._first_frame_keyframes_mask`'s unconditional
/// first-latent-frame ones (see [`generate`]'s own `keyframes_mask`
/// construction). Rebuilding the mask from scratch here instead of copying
/// the base range in would drop that marker, i.e. turning image conditioning
/// on would also, invisibly, turn OFF a positional-embedding term the same
/// clip has in the unconditioned path - which is why `base_keyframes_mask`
/// is a parameter rather than something this function reconstructs.
///
/// Position units match [`keyframe_conditioning_positions`] exactly - see
/// that function's own doc for the exact reference formula and why
/// `frame_idx=0` and [`real_pixel_positions`]`(1, lh, lw, fps)` agree.
struct ImageConditioning {
    /// `[3, base_t + n*lh*lw, 2]`, `n = blocks.len()`.
    positions: Vec<f32>,
    /// `[base_t + n*lh*lw]` - the caller's own base mask verbatim on
    /// `[0, base_t)`, zero on the appended range (see this struct's doc).
    keyframes_mask: Vec<f32>,
    /// `[base_t + n*lh*lw]` - `1.0` (denoise fully) on `[0, base_t)`,
    /// `1 - strength` (`0.0` = fully frozen, see [`Frozen`]) on the appended
    /// range.
    denoise_mask: Vec<f32>,
    /// `[base_t + n*lh*lw, channels]` - each appended block holds its own
    /// real encoded image latent (token-major, [`chw_to_tc`]'s layout); the
    /// base range is never read (`denoise_mask` is `1.0` there) and is left
    /// zeroed.
    clean: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn append_image_conditioning(base_t: usize, base_positions: &[f32], base_keyframes_mask: &[f32], lh: usize, lw: usize, channels: usize, fps: f64, appended_denoise_mask: f32, blocks: &[(usize, &[f32])]) -> ImageConditioning {
    assert_eq!(base_positions.len(), 3 * base_t * 2, "append_image_conditioning: base_positions has {} values, expected {}", base_positions.len(), 3 * base_t * 2);
    assert_eq!(base_keyframes_mask.len(), base_t, "append_image_conditioning: base_keyframes_mask has {} values, expected {base_t}", base_keyframes_mask.len());
    let block_t = lh * lw;
    for (_, tokens) in blocks {
        assert_eq!(tokens.len(), block_t * channels, "append_image_conditioning: a block has {} values, expected {}", tokens.len(), block_t * channels);
    }
    let n = blocks.len();
    let cond_t = n * block_t;
    let total_t = base_t + cond_t;

    let mut positions = vec![0f32; 3 * total_t * 2];
    for axis in 0..3 {
        positions[axis * total_t * 2..axis * total_t * 2 + base_t * 2].copy_from_slice(&base_positions[axis * base_t * 2..(axis + 1) * base_t * 2]);
    }
    for (bi, &(frame_idx, _)) in blocks.iter().enumerate() {
        let block_positions = keyframe_conditioning_positions(frame_idx, lh, lw, fps);
        let dst_off = base_t + bi * block_t;
        for axis in 0..3 {
            let dst = axis * total_t * 2 + dst_off * 2;
            positions[dst..dst + block_t * 2].copy_from_slice(&block_positions[axis * block_t * 2..(axis + 1) * block_t * 2]);
        }
    }

    let mut keyframes_mask = vec![0f32; total_t];
    keyframes_mask[..base_t].copy_from_slice(base_keyframes_mask);

    let mut denoise_mask = vec![appended_denoise_mask; total_t];
    denoise_mask[..base_t].fill(1.0);

    let mut clean = vec![0f32; total_t * channels];
    for (bi, &(_, tokens)) in blocks.iter().enumerate() {
        let dst_off = (base_t + bi * block_t) * channels;
        clean[dst_off..dst_off + block_t * channels].copy_from_slice(tokens);
    }

    ImageConditioning { positions, keyframes_mask, denoise_mask, clean }
}

#[cfg(test)]
mod image_conditioning_tests {
    use super::*;

    /// The base video's own `keyframes_mask` - `VideoLatentTools.
    /// _first_frame_keyframes_mask`'s unconditional first-latent-frame ones,
    /// exactly what [`generate`] builds. `base_t = 2*1*3` with one latent
    /// frame worth of tokens = `1*3`.
    fn base_mask(base_t: usize, tokens_per_latent_frame: usize) -> Vec<f32> {
        let mut m = vec![0f32; base_t];
        m[..tokens_per_latent_frame].fill(1.0);
        m
    }

    #[test]
    fn appends_lh_lw_tokens_after_base_with_frame0_bounds_and_frozen_mask() {
        let (base_t, lh, lw, channels, fps) = (6usize, 2usize, 2usize, 3usize, 8.0);
        let base_positions = real_pixel_positions(2, 1, 3, fps); // base_t = 2*1*3 = 6, arbitrary but matching
        let base_km = base_mask(base_t, 3);
        let cond_tokens: Vec<f32> = (0..lh * lw * channels).map(|i| i as f32).collect();

        let ic = append_image_conditioning(base_t, &base_positions, &base_km, lh, lw, channels, fps, 0.0, &[(0, &cond_tokens)]);

        let cond_t = lh * lw;
        let total_t = base_t + cond_t;
        assert_eq!(ic.positions.len(), 3 * total_t * 2);
        assert_eq!(&ic.keyframes_mask[..base_t], &base_km[..], "the base video keeps its own first-latent-frame marker: extend_keyframes_mask EXTENDS the state's mask, it does not rebuild it");
        assert_eq!(&ic.keyframes_mask[base_t..], &vec![0f32; cond_t][..], "ordinary image conditioning is marked=false");
        assert_eq!(&ic.denoise_mask[..base_t], &vec![1.0f32; base_t][..], "base tokens denoise fully");
        assert_eq!(&ic.denoise_mask[base_t..], &vec![0.0f32; cond_t][..], "conditioning tokens are frozen");
        assert_eq!(&ic.clean[..base_t * channels], &vec![0.0f32; base_t * channels][..], "base range of `clean` is never read");
        assert_eq!(&ic.clean[base_t * channels..], &cond_tokens[..], "conditioning tokens carry the real encoded content");

        // Base positions copied verbatim into [0, base_t) of every axis.
        for axis in 0..3 {
            assert_eq!(&ic.positions[axis * total_t * 2..axis * total_t * 2 + base_t * 2], &base_positions[axis * base_t * 2..(axis + 1) * base_t * 2]);
        }
        // Appended positions match a standalone real_pixel_positions(1, lh,
        // lw, fps) - frame axis bounds [0, 1/fps) for every appended token
        // (frame_idx=0, causal-fixed the same way the base video's own
        // frame 0 is).
        let expect_cond = real_pixel_positions(1, lh, lw, fps);
        for axis in 0..3 {
            assert_eq!(&ic.positions[axis * total_t * 2 + base_t * 2..(axis + 1) * total_t * 2], &expect_cond[axis * cond_t * 2..(axis + 1) * cond_t * 2]);
        }
        assert_eq!(&ic.positions[0..2], &[0.0, 0.125], "frame axis, first base token: unaffected by the appended range");
        assert_eq!(&ic.positions[base_t * 2..base_t * 2 + 2], &[0.0, 0.125], "frame axis, first appended token: frame_idx=0, same instant as the base video's own frame 0");
    }

    #[test]
    fn loop_conditioning_appends_two_blocks_of_the_same_image_at_two_frame_indices() {
        let (base_t, lh, lw, channels, fps) = (6usize, 2usize, 2usize, 3usize, 8.0);
        let base_positions = real_pixel_positions(2, 1, 3, fps);
        let base_km = base_mask(base_t, 3);
        let cond_tokens: Vec<f32> = (0..lh * lw * channels).map(|i| i as f32).collect();
        let last_pixel_frame = 8usize; // e.g. a 9-frame clip's last pixel-frame index

        let ic = append_image_conditioning(base_t, &base_positions, &base_km, lh, lw, channels, fps, 0.0, &[(0, &cond_tokens), (last_pixel_frame, &cond_tokens)]);

        let block_t = lh * lw;
        let total_t = base_t + 2 * block_t;
        assert_eq!(ic.positions.len(), 3 * total_t * 2);
        assert_eq!(&ic.keyframes_mask[..base_t], &base_km[..]);
        assert_eq!(&ic.keyframes_mask[base_t..], &vec![0f32; 2 * block_t][..]);
        assert_eq!(&ic.denoise_mask[..base_t], &vec![1.0f32; base_t][..]);
        assert_eq!(&ic.denoise_mask[base_t..], &vec![0.0f32; 2 * block_t][..], "BOTH appended blocks are frozen");
        // Both blocks carry the SAME encoded image - one real VAE encode,
        // reused at both timeline positions (the loop's whole point: same
        // content at start and end).
        assert_eq!(&ic.clean[base_t * channels..(base_t + block_t) * channels], &cond_tokens[..]);
        assert_eq!(&ic.clean[(base_t + block_t) * channels..], &cond_tokens[..]);

        // First appended block: frame_idx=0 -> [0, 0.125).
        assert_eq!(&ic.positions[base_t * 2..base_t * 2 + 2], &[0.0, 0.125]);
        // Second appended block: frame_idx=8 -> [1.0, 1.125).
        let second_off = base_t + block_t;
        assert_eq!(&ic.positions[second_off * 2..second_off * 2 + 2], &[1.0, 1.125]);
    }

    #[test]
    fn start_and_end_conditioning_may_carry_two_different_images() {
        let (base_t, lh, lw, channels, fps) = (6usize, 2usize, 2usize, 3usize, 8.0);
        let base_positions = real_pixel_positions(2, 1, 3, fps);
        let base_km = base_mask(base_t, 3);
        let start_tokens: Vec<f32> = (0..lh * lw * channels).map(|i| i as f32).collect();
        let end_tokens: Vec<f32> = (0..lh * lw * channels).map(|i| 100.0 + i as f32).collect();

        let ic = append_image_conditioning(base_t, &base_positions, &base_km, lh, lw, channels, fps, 0.0, &[(0, &start_tokens), (8, &end_tokens)]);

        let block_t = lh * lw;
        assert_eq!(&ic.clean[base_t * channels..(base_t + block_t) * channels], &start_tokens[..], "the start block keeps the start image's own content");
        assert_eq!(&ic.clean[(base_t + block_t) * channels..], &end_tokens[..], "the end block keeps the end image's own (different) content");
    }
}

/// One generation's conditioned latent state - what [`generate`] hands the
/// denoise loop once the given stills (if any) have been encoded.
struct ConditionedLatent {
    /// `[t, channels]` token-major initial latent. Frozen tokens hold their
    /// clean content, never noise: `ltx_core.components.noisers.
    /// GaussianNoiser.__call__`'s second `torch.lerp(clean_latent, noised,
    /// denoise_mask)` is exactly `clean` wherever `denoise_mask == 0`.
    latent: Vec<f32>,
    /// `[3, t, 2]` RoPE bounds.
    positions: Vec<f32>,
    /// `[t]` first-latent-frame marker, extended over any appended range.
    keyframes_mask: Vec<f32>,
    /// `[t]`, `1.0` denoise / `0.0` frozen - the reference's `denoise_mask`,
    /// which is BOTH what [`post_process_latent`] re-pins with and what the
    /// per-token timesteps are a product of (see [`denoise`]).
    denoise_mask: Vec<f32>,
    /// `[t, channels]` clean content for the frozen tokens (zero elsewhere,
    /// never read there).
    clean: Vec<f32>,
    /// Token count, `>= base_t` (each appended block adds `lh*lw`).
    t: usize,
}

/// Build [`ConditionedLatent`] from the base noise latent plus whichever
/// stills were given, **choosing the reference mechanism the requested
/// combination actually calls for**. There are two of them and they are not
/// interchangeable:
///
/// * **One still at frame 0 only** - image-to-video. `ltx_pipelines.utils.
///   helpers.combined_image_conditionings`' `img.frame_idx == 0` branch:
///   `VideoConditionByLatentIndex(latent_idx=0)`, which OVERWRITES the base
///   video's own first latent frame in place (`conditioning/types/
///   latent_cond.py`'s `apply_to`: `clean_latent[:, start:stop] = tokens`,
///   `denoise_mask[:, start:stop] = 1 - strength`). No token is appended;
///   the clip's own frame 0 IS the still. This is `ti2vid_one_stage` /
///   `distilled`'s conditioning, and it is right for "start from this
///   image".
/// * **Any other combination** - two ends, an interior anchor, or all three -
///   keyframe interpolation, which is a DIFFERENT reference pipeline with a
///   DIFFERENT conditioning builder.
///   `ltx_pipelines.keyframe_interpolation.KeyframeInterpolationPipeline.
///   __call__` uses `helpers.image_conditionings_by_adding_guiding_latent`,
///   which wraps EVERY image - `frame_idx == 0` included, with no special
///   case - in `VideoConditionByKeyframeIndex`, i.e. APPENDS a guiding
///   token block per still and leaves every one of the generated video's own
///   tokens denoising freely. `frame_idx` is a raw pixel-frame offset onto
///   the RoPE time coordinate, so a block may point at any instant of the
///   clip, and the reference caps neither the count nor the positions.
///   **The one deliberate deviation from the reference is the interior
///   anchor's mechanism.** The reference would append the mid still as a
///   third guide like any other; measured on the real 22B checkpoint
///   (`crates/ltxv/tests/it/longform_mid_real.rs`), a guide BENDS the
///   trajectory toward its instant (delta 11.62 at the mid still, against
///   25-29 everywhere else) but never REALISES its content - attention-only
///   pull lets the model compromise, which reads on screen as "the motion
///   changes direction but nothing resembles the mid frame". So this port
///   PINS the mid still into the base grid's own latent slot - the slot
///   [`latent_frame_containing`] names - with the same in-place
///   overwrite-and-freeze the image-to-video builder applies at frame 0
///   (`VideoConditionByLatentIndex`'s `clean_latent[:, start:stop] = tokens`,
///   `denoise_mask[:, start:stop] = 1 - strength`; the same freezing a masked
///   video-to-video pass trains on). The pinned slot keeps the base grid's
///   own positions and keyframes mask - it is a structural frame of the
///   clip, not an appended token block - and the causal VAE renders its
///   content across the slot's whole pixel span, so the trajectory must pass
///   through the mid image rather than merely bend toward it. Two further
///   variants were MEASURED and rejected against the plain pin
///   (`crates/ltxv/tests/it/longform_mid_real.rs`, real 22B checkpoint, mid
///   delta at its instant): the still also appended as a guide at the
///   owner's instant changed nothing (9.25 either way - the pinned latent's
///   decode dominates its pixel span, so the guide's attention-only pull has
///   nothing left to move), and encoding the still through the video path
///   with its span repeated (the interior slot's native 8-frame aggregation)
///   produced a bit-identical trajectory - a constant signal's temporal
///   aggregation IS its one-frame latent, so the cheap image-path encode is
///   already the right pin content. The measured delta floor at the instant
///   (~9) is the prompt-driven motion the clip keeps rendering AROUND the
///   still's content inside the pinned span - a pinned interior still
///   competes with the scene's own movement, where the start pin covers
///   every pixel - and it holds across the slot's whole pixel span plus one
///   frame of causal bleed backward, which is what "the clip passes through
///   the mid image" looks like on screen. The end stills stay pure appended
///   guides: those measurably work, and a pinned last latent frame would
///   hold the clip's final second motionless.
///
/// **What the difference buys, and what it does NOT.** Overwriting latent
/// frame 0 freezes a whole latent frame of the GENERATED sequence: the
/// causal VAE's first latent frame covers one pixel frame, but it still
/// costs `lh*lw` tokens - `1/lat_t` of the clip, which is 1-in-16 at the
/// 121-frame shapes the image-to-video pipelines run and HALF of a 9-frame
/// clip. Appending instead leaves every generated token free and makes the
/// still pure guidance. That is a real structural difference and it is what
/// the reference does here, so this is what the port does.
///
/// It is **not** what decides whether a both-ends clip animates. Measured
/// on the real 22B checkpoint at 9 frames / 384x192 with the same still at
/// both ends: peak excursion from frame 0 was 5.07 with the overwrite
/// mechanism and 4.64 with this one - both frozen, both reproducing the
/// still in every frame. What decides that is whether the two anchors are
/// the same picture; see `GenOpts::end_frame` and
/// `crates/ltxv/tests/it/motion_real.rs`'s table.
///
/// **Which of the two runs is decided by how many stills were given, not by
/// which ones.** One still at frame 0 and nothing else is image-to-video and
/// takes the overwrite; every other request - including any [`GenOpts::
/// mid_frame`] anchor - is keyframe interpolation. There the end stills are
/// appended guides, `frame_idx == 0` included, because that is what
/// `image_conditionings_by_adding_guiding_latent` does - and the mid anchor
/// is pinned into its slot instead (see the deviation above). Nothing about
/// the existing one- and two-still requests changes.
///
/// `start`/`mid`/`end` are already-encoded `[lh*lw, channels]` latent token
/// blocks (one real VAE encode each; the SAME image passed at both ends is
/// encoded once and reused). `frames` is the clip's pixel-frame count - the
/// end still conditions pixel-frame `frames - 1`, and `mid` carries the pixel
/// frame it conditions ([`mid_anchor_frame`]). At least one still must be
/// present.
///
/// `noise` is the whole sequence's initial Gaussian draw, `[(base_t +
/// blocks*lh*lw), channels]` ([`conditioning_block_count`] gives `blocks`) -
/// the reference draws ONE noise tensor over the full post-conditioning
/// sequence (`GaussianNoiser._sample_noise` runs after every conditioning
/// item has been applied), so the appended range gets real noise, not zeros.
/// It only matters at `strength < 1.0`; at `1.0` the conditioned tokens are
/// exactly `clean` and the appended noise is multiplied out.
///
/// `strength` is `ImageConditioningInput.strength`, the per-image knob every
/// reference pipeline takes as a REQUIRED CLI argument (`--image PATH
/// FRAME_IDX STRENGTH`, `ltx_pipelines.utils.args.ImageAction`, whose own
/// help text's example is `0.8`). Both conditioning items turn it into the
/// same two things:
///
/// * `denoise_mask = 1 - strength` over the conditioned tokens
///   (`latent_cond.py`'s `apply_to`, `keyframe_cond.py`'s `torch.full(...,
///   1.0 - self.strength)`) - which is both what [`post_process_latent`]
///   blends with and, through `timesteps_from_mask`, the timestep those
///   tokens are announced at;
/// * the initial latent there, `GaussianNoiser`'s `torch.lerp(clean_latent,
///   noised, denoise_mask)` = `(1-m)*clean + m*noise`.
///
/// `1.0` (this crate's default) is the hardest possible pin: mask `0`,
/// timestep `0`, the token nailed to `clean` for the whole trajectory and
/// pulling every free token toward it through self-attention. That is the
/// right setting for "the clip starts as exactly this frame"; it is NOT the
/// only setting the reference supports, and a clip conditioned at both ends
/// at `1.0` is being asked for a much harder thing than the same clip at
/// `0.8`.
///
/// `future` carries the long-form windows' additional anchors - stills whose
/// named pixel instants may lie BEYOND this window's own generated span (see
/// [`GenOpts::future_anchors`]). They are appended guiding blocks like any
/// other, at the raw pixel-frame positions they name; the reference's
/// `frame_idx` RoPE offset bounds nothing, and RoPE attention is relative, so
/// a guide beyond the last generated token is the same mechanism a guide at
/// frame 0 is, naming an instant the window's own tokens must approach. A
/// start still alongside any future anchor takes the interpolation path, not
/// the overwrite: more than one anchor means the base video's own tokens all
/// stay free, exactly as a mid or end anchor already forces.
#[allow(clippy::too_many_arguments)]
fn conditioned_latent(noise: Vec<f32>, base_positions: &[f32], base_keyframes_mask: &[f32], base_t: usize, lh: usize, lw: usize, channels: usize, frames: usize, fps: f64, start: Option<&[f32]>, mid: Option<(usize, &[f32])>, end: Option<&[f32]>, future: &[(usize, &[f32])], strength: f32) -> ConditionedLatent {
    assert!(start.is_some() || mid.is_some() || end.is_some() || !future.is_empty(), "conditioned_latent: at least one still must be given");
    assert!((0.0..=1.0).contains(&strength), "conditioned_latent: strength {strength} is outside [0, 1]");
    let block_t = lh * lw;
    let blocks = conditioning_block_count(start.is_some(), mid.is_some(), end.is_some(), future.len());
    let total_t = base_t + blocks * block_t;
    assert_eq!(noise.len(), total_t * channels, "conditioned_latent: noise has {} values, expected {}", noise.len(), total_t * channels);
    let m = 1.0 - strength;
    // `GaussianNoiser`'s second lerp, for one token range: `(1-m)*clean +
    // m*noise`. At `m == 0` (strength 1.0) this is exactly `clean`.
    let mix = |latent: &mut [f32], clean_tokens: &[f32], off_tokens: usize| {
        let off = off_tokens * channels;
        for (i, &c) in clean_tokens.iter().enumerate() {
            latent[off + i] = (1.0 - m) * c + m * latent[off + i];
        }
    };
    // The mid anchor's slot on the BASE grid: the latent frame whose pixel
    // span contains the instant it names. It must name a real interior frame
    // - pixel frame 0 is the start still's own slot - and fit the grid the
    // caller built; both violations are mis-routed requests, not user error.
    let mid_slot = mid.map(|(px, _)| {
        assert!(px >= 1, "conditioned_latent: mid anchor names pixel frame 0, which is the start still's own slot");
        let slot = latent_frame_containing(px);
        assert!((slot + 1) * block_t <= base_t, "conditioned_latent: mid anchor at pixel frame {px} lands in latent frame {slot}, beyond this pass's {0}-latent-frame grid", base_t / block_t);
        slot
    });

    // Image-to-video: one still at frame 0 and nothing else overwrites the
    // base video's own first latent frame.
    if let (Some(s), None, None) = (start, mid, end) {
        if future.is_empty() {
            let mut latent = noise;
            mix(&mut latent, s, 0);
            let mut denoise_mask = vec![1.0f32; base_t];
            denoise_mask[..block_t].fill(m);
            let mut clean = vec![0f32; base_t * channels];
            clean[..block_t * channels].copy_from_slice(s);
            return ConditionedLatent { latent, positions: base_positions.to_vec(), keyframes_mask: base_keyframes_mask.to_vec(), denoise_mask, clean, t: base_t };
        }
    }

    // Keyframe interpolation: the end stills appended as their own guiding
    // blocks in timeline order, the base video otherwise untouched (see this
    // function's doc). The future anchors come last: their named instants lie
    // beyond this window's own span, and every in-window anchor sits inside
    // it. The mid anchor is NOT among them - it takes the pin below.
    let anchors: Vec<(usize, &[f32])> = [start.map(|s| (0usize, s)), end.map(|e| (frames - 1, e))].into_iter().flatten().chain(future.iter().copied()).collect();
    let mut ic = append_image_conditioning(base_t, base_positions, base_keyframes_mask, lh, lw, channels, fps, m, &anchors);
    let mut latent = noise;
    for (bi, (_, tokens)) in anchors.iter().enumerate() {
        mix(&mut latent, tokens, base_t + bi * block_t);
    }
    // The mid pin: overwrite the slot on the base grid whose pixel span
    // contains the named instant, freeze it there (the same `1 - strength`
    // algebra every conditioning item applies). The slot keeps the base
    // grid's own positions and keyframes mask - it is a structural frame of
    // the clip the trajectory must pass through, not an appended guide the
    // model may compromise with. `mid.is_some()` and `mid_slot.is_none()`
    // cannot disagree: both derive from the same `mid`.
    if let (Some((_, tokens)), Some(slot)) = (mid, mid_slot) {
        mix(&mut latent, tokens, slot * block_t);
        let (s0, s1) = (slot * block_t, (slot + 1) * block_t);
        ic.denoise_mask[s0..s1].fill(m);
        ic.clean[s0 * channels..s1 * channels].copy_from_slice(tokens);
    }
    ConditionedLatent { latent, positions: ic.positions, keyframes_mask: ic.keyframes_mask, denoise_mask: ic.denoise_mask, clean: ic.clean, t: total_t }
}

/// How many `lh*lw`-token conditioning blocks [`conditioned_latent`] will
/// APPEND for a given request - `0` for image-to-video (a lone start still
/// overwrites latent frame 0 in place), otherwise one per still given,
/// start included. The future anchors count as stills: a start still beside
/// any one of them is interpolation, not the overwrite. The mid anchor
/// contributes NO block of its own - it pins into the base grid's latent
/// slot - but its PRESENCE still flips a start still onto the interpolation
/// mechanism, so it takes part in the decision without adding to the count.
/// [`generate`] needs this before the stills are encoded, to draw the
/// initial noise at the full post-conditioning length in one go (see
/// [`conditioned_latent`]'s `noise`).
fn conditioning_block_count(start: bool, mid: bool, end: bool, future: usize) -> usize {
    if start && !mid && !end && future == 0 {
        0
    } else {
        usize::from(start) + usize::from(end) + future
    }
}

#[cfg(test)]
mod conditioned_latent_tests {
    use super::*;

    /// A 9-frame clip's geometry at a 2x2 latent grid: 2 latent frames of
    /// `lh*lw = 4` tokens each.
    const LH: usize = 2;
    const LW: usize = 2;
    const CH: usize = 3;
    const FRAMES: usize = 9;
    const FPS: f64 = 8.0;

    /// `(noise, base_positions, base_keyframes_mask, base_t)` for a clip
    /// with `blocks` appended conditioning blocks - the noise vector is
    /// drawn at the full post-conditioning length, exactly the way
    /// [`generate`] draws it.
    fn base(blocks: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let lat_t = 2usize;
        let base_t = lat_t * LH * LW;
        let positions = real_pixel_positions(lat_t, LH, LW, FPS);
        let mut km = vec![0f32; base_t];
        km[..LH * LW].fill(1.0);
        let noise: Vec<f32> = (0..(base_t + blocks * LH * LW) * CH).map(|i| -(i as f32) - 1.0).collect();
        (noise, positions, km, base_t)
    }

    fn tokens(bias: f32) -> Vec<f32> {
        (0..LH * LW * CH).map(|i| bias + i as f32).collect()
    }

    /// The reference's interpolation pipeline
    /// (`image_conditionings_by_adding_guiding_latent`) appends a guiding
    /// block per still - `frame_idx == 0` included - and freezes NONE of the
    /// generated video's own tokens. A port that reuses the image-to-video
    /// builder's `frame_idx == 0` overwrite instead pins half of a 9-frame
    /// clip's latent frames.
    #[test]
    fn both_ends_append_two_guiding_blocks_and_freeze_no_generated_token() {
        let (latent, positions, km, base_t) = base(2);
        let (s, e) = (tokens(0.0), tokens(100.0));

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, Some(&e), &[], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + 2 * block_t, "one appended guiding block per still");
        assert_eq!(&c.denoise_mask[..base_t], &vec![1.0f32; base_t][..], "every token of the GENERATED video denoises freely - VideoConditionByKeyframeIndex appends, it never overwrites");
        assert_eq!(&c.denoise_mask[base_t..], &vec![0.0f32; 2 * block_t][..], "both appended guiding blocks are frozen (strength 1.0)");
        assert_eq!(&c.latent[..base_t * CH], &latent[..base_t * CH], "the base video keeps its own noise: nothing is overwritten");
        assert_eq!(&c.latent[base_t * CH..(base_t + block_t) * CH], &s[..]);
        assert_eq!(&c.latent[(base_t + block_t) * CH..], &e[..]);
        assert_eq!(c.positions.len(), 3 * c.t * 2);
        assert_eq!(c.keyframes_mask.len(), c.t);
        assert_eq!(c.clean.len(), c.t * CH);
        // Guiding blocks land at pixel-frame 0 and `frames - 1`.
        assert_eq!(&c.positions[base_t * 2..base_t * 2 + 2], &[0.0, 0.125]);
        let second = base_t + block_t;
        assert_eq!(&c.positions[second * 2..second * 2 + 2], &[1.0, 1.125], "the end still conditions pixel-frame 8 of a 9-frame clip");
    }

    /// Start-only stays image-to-video: `combined_image_conditionings`'
    /// `frame_idx == 0` branch really does overwrite latent frame 0, and
    /// that is the right mechanism when the caller asked for "the clip
    /// starts as this image".
    #[test]
    fn start_only_overwrites_the_first_latent_frame_in_place() {
        let (latent, positions, km, base_t) = base(0);
        let s = tokens(0.0);

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, None, &[], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t, "nothing is appended");
        assert_eq!(&c.latent[..block_t * CH], &s[..], "the still replaces the first latent frame's noise");
        assert_eq!(&c.latent[block_t * CH..], &latent[block_t * CH..], "every later token keeps its noise");
        assert_eq!(&c.denoise_mask[..block_t], &vec![0.0f32; block_t][..]);
        assert_eq!(&c.denoise_mask[block_t..], &vec![1.0f32; base_t - block_t][..]);
        assert_eq!(&c.clean[..block_t * CH], &s[..]);
        assert_eq!(c.positions, positions, "the base video's own positions are unchanged");
    }

    /// End-only appends exactly one guiding block at `frames - 1`.
    #[test]
    fn end_only_appends_one_guiding_block_at_the_last_pixel_frame() {
        let (latent, positions, km, base_t) = base(1);
        let e = tokens(100.0);

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, None, Some(&e), &[], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + block_t);
        assert_eq!(&c.denoise_mask[..base_t], &vec![1.0f32; base_t][..]);
        assert_eq!(&c.denoise_mask[base_t..], &vec![0.0f32; block_t][..]);
        assert_eq!(&c.latent[..base_t * CH], &latent[..base_t * CH]);
        assert_eq!(&c.latent[base_t * CH..], &e[..]);
        assert_eq!(&c.positions[base_t * 2..base_t * 2 + 2], &[1.0, 1.125]);
    }

    /// `strength < 1.0` is the reference's own per-image knob and it has to
    /// reach BOTH of the places `ImageConditioningInput.strength` reaches:
    /// the `1 - strength` denoise mask (which is also the conditioned
    /// tokens' timestep, through `timesteps_from_mask`), and
    /// `GaussianNoiser`'s `lerp(clean, noise, mask)` initial latent. A port
    /// that only softened the mask would announce "partly noisy" for a token
    /// it had nonetheless pinned to a clean image.
    #[test]
    fn strength_below_one_softens_both_the_mask_and_the_initial_latent() {
        let (latent, positions, km, base_t) = base(2);
        let (s, e) = (tokens(0.0), tokens(100.0));
        let strength = 0.8f32;
        let m = 1.0 - strength;

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, Some(&e), &[], strength);

        let block_t = LH * LW;
        assert_eq!(&c.denoise_mask[base_t..], &vec![m; 2 * block_t][..], "denoise_mask over a conditioned token is 1 - strength");
        assert_eq!(&c.denoise_mask[..base_t], &vec![1.0f32; base_t][..]);
        // clean is unaffected by strength - it is the target, not the blend.
        assert_eq!(&c.clean[base_t * CH..(base_t + block_t) * CH], &s[..]);
        for (i, &v) in c.latent[base_t * CH..(base_t + block_t) * CH].iter().enumerate() {
            let want = (1.0 - m) * s[i] + m * latent[base_t * CH + i];
            assert!((v - want).abs() < 1e-5, "token {i}: initial latent is lerp(clean, noise, 1-strength): got {v}, want {want}");
        }
    }

    /// The block count `generate` needs BEFORE the stills are encoded, to
    /// draw one noise vector at the full post-conditioning length. The mid
    /// anchor never appends - it pins into its owning latent slot - but its
    /// presence still moves a start still onto the interpolation mechanism.
    #[test]
    fn appended_block_count_matches_the_mechanism_each_request_uses() {
        assert_eq!(conditioning_block_count(false, false, false, 0), 0, "unconditioned");
        assert_eq!(conditioning_block_count(true, false, false, 0), 0, "image-to-video overwrites in place");
        assert_eq!(conditioning_block_count(false, false, true, 0), 1);
        assert_eq!(conditioning_block_count(true, false, true, 0), 2, "keyframe interpolation appends BOTH end stills");
        assert_eq!(conditioning_block_count(false, true, false, 0), 0, "a lone mid anchor pins and appends nothing");
        assert_eq!(conditioning_block_count(true, true, false, 0), 1, "the mid pin moves the start still onto an appended guide: one block, the start's");
        assert_eq!(conditioning_block_count(true, true, true, 0), 2, "the two end stills append; the mid pins");
        assert_eq!(conditioning_block_count(true, false, false, 1), 2, "a future anchor turns the start still into an appended guide: it appends, and so does the start");
        assert_eq!(conditioning_block_count(false, false, false, 2), 2, "future anchors alone are their own appended blocks");
    }

    /// **The mid anchor is both pinned and appended; start and end are pure
    /// guides.** The two end stills each get their own appended guiding
    /// block (the reference's `image_conditionings_by_adding_guiding_latent`
    /// semantics); the interior anchor is written INTO the latent slot
    /// containing its pixel instant, frozen there - structural, not advisory,
    /// and appended nowhere.
    #[test]
    fn three_anchors_pin_the_mid_and_append_the_two_ends() {
        let (latent, positions, km, base_t) = base(2);
        let (s, m, e) = (tokens(0.0), tokens(50.0), tokens(100.0));
        let mid_at = mid_anchor_frame(FRAMES, None).expect("a 9-frame clip has an interior");
        assert_eq!(mid_at, 4);

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), Some((mid_at, &m)), Some(&e), &[], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + 2 * block_t, "only the two end stills are appended; the mid pin costs no tokens");
        assert_eq!(&c.denoise_mask[base_t..], &vec![0.0f32; 2 * block_t][..], "both appended guiding blocks are frozen");
        // Pixel frame 4 of a 9-frame clip sits in latent frame 1 (px [1, 9)):
        // the pin occupies exactly that slot of the base grid.
        let slot = latent_frame_containing(mid_at) * block_t;
        assert_eq!(slot, block_t);
        assert_eq!(&c.latent[slot * CH..(slot + block_t) * CH], &m[..], "the pinned slot carries the mid still, replacing its noise");
        assert_eq!(&c.denoise_mask[slot..slot + block_t], &vec![0.0f32; block_t][..], "the pinned slot is frozen at strength 1.0");
        assert_eq!(&c.clean[slot * CH..(slot + block_t) * CH], &m[..], "the pinned slot's clean target is the still");
        assert_eq!(&c.denoise_mask[..slot], &vec![1.0f32; slot][..], "every other generated token still denoises freely");
        assert_eq!(&c.denoise_mask[slot + block_t..base_t], &vec![1.0f32; base_t - slot - block_t][..]);
        assert_eq!(&c.latent[..slot * CH], &latent[..slot * CH], "tokens outside the pin keep their own noise");
        assert_eq!(&c.latent[(slot + block_t) * CH..base_t * CH], &latent[(slot + block_t) * CH..base_t * CH]);
        // The pinned slot keeps the video's OWN grid positions - it is a
        // structural frame of the clip, not an appended guide with its own
        // one-frame-wide RoPE span.
        for axis in 0..3 {
            let src = &positions[axis * base_t * 2..(axis + 1) * base_t * 2];
            let dst = &c.positions[axis * c.t * 2..(axis + 1) * c.t * 2];
            assert_eq!(&dst[slot * 2..(slot + block_t) * 2], &src[slot * 2..(slot + block_t) * 2], "axis {axis}: the pinned slot uses the base grid's own positions");
        }
        // The end stills land at their own instants as appended guides.
        assert_eq!(&c.positions[base_t * 2..base_t * 2 + 2], &[0.0, 0.125]);
        let second = base_t + block_t;
        assert_eq!(&c.positions[second * 2..second * 2 + 2], &[1.0, 1.125]);
    }

    /// A middle anchor with no still at either end is a pin into its slot and
    /// nothing else - the mechanism does not need company, and no block is
    /// appended for it.
    #[test]
    fn a_lone_mid_anchor_pins_into_its_slot() {
        let (latent, positions, km, base_t) = base(0);
        let m = tokens(50.0);

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, Some((4, &m)), None, &[], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t, "nothing is appended");
        let slot = latent_frame_containing(4) * block_t;
        assert_eq!(&c.latent[slot * CH..(slot + block_t) * CH], &m[..]);
        assert_eq!(&c.denoise_mask[slot..slot + block_t], &vec![0.0f32; block_t][..]);
        assert_eq!(&c.clean[slot * CH..(slot + block_t) * CH], &m[..]);
        assert_eq!(&c.denoise_mask[..slot], &vec![1.0f32; slot][..]);
        assert_eq!(c.positions, positions, "the base grid's positions are untouched - the pin reuses its own slot");
    }

    /// A pinned mid at strength < 1.0 blends both the mask and the initial
    /// latent exactly like every other conditioning item - the pin is the
    /// same `1 - strength` algebra, applied to a slot of the base grid.
    #[test]
    fn a_pinned_mid_blends_at_strength_below_one() {
        let (latent, positions, km, base_t) = base(0);
        let m_img = tokens(50.0);
        let strength = 0.8f32;
        let m = 1.0 - strength;

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, Some((4, &m_img)), None, &[], strength);

        let block_t = LH * LW;
        let slot = latent_frame_containing(4) * block_t;
        assert_eq!(&c.denoise_mask[slot..slot + block_t], &vec![m; block_t][..]);
        for (i, &v) in c.latent[slot * CH..(slot + block_t) * CH].iter().enumerate() {
            let want = (1.0 - m) * m_img[i] + m * latent[slot * CH + i];
            assert!((v - want).abs() < 1e-5, "token {i}: the pinned slot's initial latent is lerp(clean, noise, 1-strength): got {v}, want {want}");
        }
    }

    /// Pixel frame 0 is the start still's own slot: a mid anchor naming it is
    /// a mis-routed request, refused loudly rather than silently pinning the
    /// clip's first latent frame twice.
    #[test]
    #[should_panic(expected = "mid anchor names pixel frame 0")]
    fn a_pinned_mid_refuses_pixel_frame_zero() {
        let (latent, positions, km, base_t) = base(0);
        let m = tokens(50.0);
        let _ = conditioned_latent(latent, &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, Some((0, &m)), None, &[], 1.0);
    }

    /// Adding a middle anchor to a `--start-frame` run moves the start still
    /// off the in-place overwrite and onto an appended guide, while the mid
    /// takes the pin - the two stills condition different mechanisms, and the
    /// generated video's own latent frame 0 must be released for the start
    /// guide, or the clip would be pinned twice at the same instant.
    #[test]
    fn a_mid_anchor_moves_the_start_still_from_an_overwrite_to_a_guide() {
        let (latent, positions, km, base_t) = base(1);
        let (s, m) = (tokens(0.0), tokens(50.0));

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), Some((4, &m)), None, &[], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + block_t, "only the start still is appended");
        assert_eq!(&c.denoise_mask[..base_t - block_t], &vec![1.0f32; base_t - block_t][..], "latent frame 0 is no longer overwritten");
        let slot = latent_frame_containing(4) * block_t;
        assert_eq!(&c.clean[base_t * CH..(base_t + block_t) * CH], &s[..], "the start still is the appended guide");
        assert_eq!(&c.clean[slot * CH..(slot + block_t) * CH], &m[..], "the mid still is the pin");
    }

    /// A future anchor rides along as its own appended block at the RAW pixel
    /// position it names - the reference's `frame_idx` RoPE offset, which
    /// bounds nothing, here naming an instant beyond the window's own
    /// generated span. Nothing else about the request changes: the block is
    /// frozen at the conditioning strength and carries its own encoded
    /// content, and the base video's tokens, positions and mask are verbatim.
    #[test]
    fn a_future_anchor_is_an_appended_block_at_its_own_raw_position() {
        let (latent, positions, km, base_t) = base(1);
        let f = tokens(7.0);

        let c = conditioned_latent(latent, &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, None, None, &[(40, &f)], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + block_t);
        assert_eq!(&c.denoise_mask[base_t..], &vec![0.0f32; block_t][..], "the guide is frozen");
        assert_eq!(&c.clean[base_t * CH..], &f[..], "the guide carries the encoded content");
        let expect = keyframe_conditioning_positions(40, LH, LW, FPS);
        for axis in 0..3 {
            assert_eq!(
                &c.positions[axis * c.t * 2 + base_t * 2..(axis + 1) * c.t * 2],
                &expect[axis * block_t * 2..(axis + 1) * block_t * 2],
                "the guide sits at its own raw pixel position, not snapped to the span it points into"
            );
        }
        assert_eq!(&c.keyframes_mask[..base_t], &km[..], "the base video keeps its own first-latent-frame marker");
    }

    /// A start still beside a future anchor is keyframe interpolation, not
    /// image-to-video: there is more than one anchor, so the start still joins
    /// the appended guides and the base video's own tokens all stay free -
    /// the same rule a mid or end anchor already triggers.
    #[test]
    fn a_start_still_beside_a_future_anchor_is_appended_not_overwritten() {
        let (latent, positions, km, base_t) = base(2);
        let (s, f) = (tokens(3.0), tokens(7.0));

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, None, &[(40, &f)], 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + 2 * block_t);
        assert_eq!(&c.latent[..base_t * CH], &latent[..base_t * CH], "every base token keeps its own noise - nothing was overwritten");
        assert_eq!(&c.clean[..base_t * CH], &vec![0.0f32; base_t * CH][..]);
        assert_eq!(&c.clean[base_t * CH..(base_t + block_t) * CH], &s[..], "first appended block: the start still at frame 0");
        assert_eq!(&c.clean[(base_t + block_t) * CH..], &f[..], "second appended block: the future anchor");
    }
}

/// The only thing the denoise loop asks of a model: a velocity prediction at
/// PER-TOKEN timesteps. A trait, private, so the CFG fold and the
/// cancellation/step bookkeeping are testable against a fake instead of a
/// real (if tiny) GPU forward - the `wan::pipeline::Denoiser` pattern.
///
/// `timesteps` is `[t]`, one RAW sigma per token (both real implementations
/// apply `timestep_scale_multiplier` internally, matching the golden's own
/// convention - see `crate::dit::LtxDit::forward`'s doc). [`denoise`] builds
/// it as the reference's `timesteps_from_mask(denoise_mask, sigma)`
/// (`ltx_pipelines.utils.helpers`), which is `sigma` broadcast uniformly
/// only when nothing is frozen.
///
/// # Why the two CFG branches are their own method
///
/// [`Self::forward_cfg_pair`] exists because the pair - not the individual
/// forward - is the unit a placement decision applies to. Both branches read
/// the SAME latent, timesteps, positions and mask; only the text context
/// differs, and nothing produced by one is read by the other. Whether they
/// run one after another on one card or at the same time on two is therefore
/// a property of the DENOISER (does its forward open its own device?), not of
/// the loop, and this is where a denoiser states it. The default is the
/// sequential pair every call site had before device plans existed.
trait Denoiser {
    fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32>;

    /// True when this denoiser carries LTX-2.5's audio stream as well as its
    /// video one, so [`denoise`] should step both.
    fn has_audio(&self) -> bool {
        false
    }

    /// One JOINT audio+video forward, returning `(video velocity, audio
    /// velocity)`.
    ///
    /// Not two forwards: the A<->V cross-attention makes each stream's answer
    /// depend on the other stream's input every block, so there is no "audio
    /// forward" that could be run on its own or afterwards. A denoiser that
    /// returns `false` from [`Self::has_audio`] is never asked.
    fn forward_av(&self, _i: &StepInputs, _a: &AudioInputs, _context: Caption, _a_context: Caption) -> (Vec<f32>, Vec<f32>) {
        unreachable!("forward_av on a video-only denoiser - denoise checks has_audio first")
    }

    /// One CFG step's conditional and unconditional forwards.
    ///
    /// Default: sequentially, on whatever device this denoiser already runs
    /// on - byte-for-byte the old behaviour, and the only correct answer for
    /// a denoiser (like [`LtxDit`]) that is `!Sync` and cannot be shared
    /// across the two threads at all. [`RealDit`] overrides it: it resolves
    /// its open device from the card the call is SCOPED to, so scoping the
    /// call is enough to move the whole forward onto the other card.
    fn forward_cfg_pair(&self, i: &StepInputs, cond: Caption, uncond: Caption) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((self.forward(i, cond), self.forward(i, uncond)))
    }

    /// Give back every device resource held BETWEEN forwards, leaving the
    /// denoiser usable - the next forward rebuilds whatever it needs.
    ///
    /// [`generate`] does not need this: it drops the whole denoiser before its
    /// VAE decode, because a resident weight window and a decode that needs up
    /// to ~16.5 GiB do not both fit on one card (see `crate::devres::
    /// planned_slots`, which caps the window at a quarter of the card for the
    /// same reason). A window loop CANNOT drop it - the next window denoises
    /// with it - so it releases the same thing instead.
    ///
    /// Default: nothing held, nothing to give back, which is right for every
    /// denoiser that opens its device per forward.
    fn release_devices(&self) {}
}

/// One guidance branch's text context as the embeddings connector reads it:
/// the prompt's `[context_len, dim]` rows and which of them are the prompt's
/// own (`1.0`) rather than padding the connector fills with its registers.
///
/// The two travel together because each branch encodes its OWN prompt, and
/// the reference masks each by its own token count
/// (`EmbeddingsProcessor.process_hidden_states` per prompt): an
/// unconditional branch read under the conditional prompt's mask treats
/// padding as caption and caption as padding.
#[derive(Clone, Copy)]
struct Caption<'a> {
    rows: &'a [f32],
    valid: &'a [f32],
}

impl<'a> Caption<'a> {
    fn of(rows: &'a [f32], valid: &'a [f32]) -> Caption<'a> {
        Caption { rows, valid }
    }
}

/// Everything one denoise-step forward needs that is IDENTICAL across the
/// conditional and unconditional branches. The text context is what differs
/// between them, so it stays a separate argument; grouping the rest is what
/// makes "run this pair, wherever" expressible as one call instead of a
/// nine-argument closure repeated twice.
#[derive(Clone, Copy)]
struct StepInputs<'a> {
    latent: &'a [f32],
    timesteps: &'a [f32],
    positions: &'a [f32],
    keyframes_mask: &'a [f32],
    context_len: usize,
    t: usize,
    /// The schedule's SCALAR sigma for this step. Only the joint AV forward
    /// reads it: each stream's cross-attention gate is modulated by the OTHER
    /// stream's sigma, which is a per-request scalar rather than the
    /// per-token `timesteps` above.
    sigma: f32,
    /// What this forward's guidance pass perturbs (STG, isolated modality) -
    /// [`crate::block::NO_PERTURBATIONS`] for the conditional pass.
    perturbations: &'a crate::block::Perturbations,
    /// A weighted IC-LoRA reference's self-attention bias over these `t`
    /// tokens, `None` when the stage carries no reference or carries it at
    /// full strength.
    reference_attention: Option<&'a crate::refcond::ReferenceAttention>,
}

impl<'a> StepInputs<'a> {
    /// The same step under one guidance pass's perturbations.
    fn with_perturbations<'b>(self, perturbations: &'b crate::block::Perturbations) -> StepInputs<'b>
    where
        'a: 'b,
    {
        let shorter: StepInputs<'b> = self;
        StepInputs { perturbations, ..shorter }
    }
}

/// The audio stream's own half of one joint forward.
struct AudioInputs<'a> {
    latent: &'a [f32],
    timesteps: &'a [f32],
    positions: &'a [f32],
    ta: usize,
    /// The audio stream's scalar sigma - see [`stream_sigma`].
    sigma: f32,
}

/// A stream's scalar sigma for one step: the schedule's, or `0` when every
/// token of the stream is held (`modality_from_latent_state`: a frozen
/// state's `Modality.sigma` is forced to 0, so its prompt AdaLN and the
/// other stream's cross-modal gate read a clean stream as clean).
fn stream_sigma(sigma: f64, mask: Option<&[f32]>) -> f32 {
    match mask {
        Some(m) if !m.is_empty() && m.iter().all(|&v| v == 0.0) => 0.0,
        _ => sigma as f32,
    }
}

/// The audio stream's state across a denoising stage - what [`denoise`]
/// carries beside the video latent when the denoiser generates sound.
///
/// The two contexts are the AUDIO stream's own text projection, not the
/// video one: see [`TextContext`]'s doc. They are BORROWED rather than
/// copied: a stage is handed the same projection every window of a long-form
/// clip, and re-copying it per stage is work a seam does not need to do.
struct AudioState<'a> {
    latent: Vec<f32>,
    positions: Vec<f32>,
    ta: usize,
    ctx_cond: &'a [f32],
    ctx_uncond: &'a [f32],
    /// `(denoise_mask, clean)` over the tokens carried in from the previous
    /// long-form WINDOW, `None` when this stage carries nothing.
    ///
    /// The audio counterpart of the video half's own [`Frozen`]: mask `0` on
    /// the carried prefix (so its per-token timestep is `0` and
    /// [`to_denoised`] is the identity there) and `1` on everything this
    /// stage generates, with the carried content re-pinned every step. The
    /// same three things the video's carried prefix gets, on the stream whose
    /// tokens are a different length of time.
    frozen: Option<(Vec<f32>, Vec<f32>)>,
}

impl Denoiser for LtxDit {
    fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
        assert_eq!(i.perturbations, &crate::block::NO_PERTURBATIONS, "the tiny host DiT runs no perturbed guidance pass");
        let inp = crate::dit::ForwardInputs { latent: i.latent, timesteps: i.timesteps, positions: i.positions, keyframes_mask: i.keyframes_mask, context: context.rows, context_len: i.context_len, t: i.t, context_valid: context.valid };
        self.forward_referenced(&inp, i.reference_attention).out
    }
}

/// One joint denoise step's inputs, in the one form BOTH audio-visual arms
/// take.
///
/// The mapping from what [`denoise`] holds (a [`StepInputs`], an
/// [`AudioInputs`] and the two streams' text projections) onto the sixteen
/// fields [`crate::dit::AvStreamedStep`] wants is written exactly once, here,
/// because that is the shape of the mistake it prevents: the two streams'
/// contexts, positions, timesteps and latents are four pairs of same-typed
/// slices, and handing one arm the audio member where the video one belongs
/// produces a plausible clip rather than an error. Gated by
/// `an_av_step_carries_each_streams_own_inputs`.
///
/// Each stream carries its own scalar sigma: both run one schedule in
/// lockstep (`ltx_pipelines.utils.samplers.euler_denoising_loop` steps them
/// together off one `sigmas` tensor), so the two are equal - unless one
/// stream is wholly frozen, which reads as sigma 0 ([`stream_sigma`]). Each
/// stream's cross-modal gate is keyed on the OTHER stream's sigma.
fn av_step<'a>(i: &StepInputs<'a>, a: &AudioInputs<'a>, v_context: Caption<'a>, a_context: Caption<'a>) -> crate::dit::AvStreamedStep<'a> {
    crate::dit::AvStreamedStep {
        v_latent: i.latent,
        v_timesteps: i.timesteps,
        v_positions: i.positions,
        v_keyframes_mask: i.keyframes_mask,
        v_context: v_context.rows,
        v_context_len: i.context_len,
        tv: i.t,
        v_sigma: i.sigma,
        v_context_valid: v_context.valid,
        a_latent: a.latent,
        a_timesteps: a.timesteps,
        a_positions: a.positions,
        a_context: a_context.rows,
        a_context_len: i.context_len,
        ta: a.ta,
        a_sigma: a.sigma,
        a_context_valid: a_context.valid,
        perturbations: i.perturbations,
        reference_attention: i.reference_attention,
    }
}

impl Denoiser for crate::av_stream::AvDenoiser {
    /// A video-only forward against an audio-visual model is not something
    /// this pipeline ever wants: dropping the audio stream would silently
    /// change the VIDEO too, because the A2V cross-attention writes into the
    /// video residual every block. [`denoise`] checks [`Self::has_audio`] and
    /// takes [`Self::forward_av`] instead.
    fn forward(&self, _i: &StepInputs, _context: Caption) -> Vec<f32> {
        unreachable!("AvDenoiser::forward - denoise takes forward_av for an audio-visual denoiser")
    }

    fn has_audio(&self) -> bool {
        true
    }

    fn forward_av(&self, i: &StepInputs, a: &AudioInputs, context: Caption, a_context: Caption) -> (Vec<f32>, Vec<f32>) {
        crate::av_stream::AvDenoiser::forward(self, &av_step(i, a, context, a_context))
    }
}

/// The real 22B checkpoint's audio-visual [`Denoiser`] - [`RealDit`]'s twin,
/// and what an `--audio` generation denoises through.
///
/// Field for field the same design, for the same reasons, over the
/// audio-extended block: an open GGUF source, the non-block "head" tensors
/// (both streams' patchify/proj_out pairs, all eight model-level adaLN groups
/// and both embeddings connectors - [`crate::dit::load_av_head_tensors_from_source`]),
/// the CHECKPOINT-scoped host weight cache, and one
/// [`crate::devres::AvDitSession`] per card holding that card's resident
/// window of already-quantized AV blocks. Every step of the generation then
/// re-reads nothing and re-uploads only what the card does not already hold.
///
/// Two things differ from [`RealDit`], both because there are two streams:
///
/// * There is no [`Denoiser::forward_cfg_pair`] override, so a CFG pair runs
///   sequentially on one card. [`denoise`]'s audio arm calls
///   [`Denoiser::forward_av`] directly - there is no joint-forward pair seam
///   on the trait to place a two-card dispatch behind - and the real
///   distilled checkpoint denoises at guidance 1, where no pair is issued at
///   all. Giving the AV path a two-card pair is a change to the trait, not to
///   this type.
/// * An AV block is bigger than a video one, so the same card affords fewer
///   resident slots. That is the residency policy's arithmetic
///   ([`crate::devres::slots_for`]), not a decision made here.
struct RealAvDit {
    cfg: LtxAvDitConfig,
    src: crate::gguf_src::LtxvGgufSource,
    head: Tensors,
    device: Option<String>,
    cache: crate::block::GenerationCache,
    /// One open device plus its resident AV weight window per card, keyed on
    /// the card the calling thread is scoped to - see [`RealDit::sessions`],
    /// whose contract this shares exactly.
    sessions: std::sync::Mutex<std::collections::HashMap<Option<u32>, std::sync::Arc<crate::devres::AvDitSession>>>,
    /// Block-linear LoRA corrections every session attaches ([`block_adapter`]).
    adapter: Option<SharedAdapter>,
    /// [`Runtime::resident_blocks`].
    resident_blocks: Option<u32>,
}

impl RealAvDit {
    /// This thread's card's session, built on first use and sized from the
    /// VIDEO token count of the forward that built it.
    ///
    /// A long-form window boundary and the stage-1 -> stage-2 boundary both
    /// change that count, and both already call [`Denoiser::release_devices`]
    /// before the next forward - so the next session is planned from the
    /// shape it will actually run, rather than inheriting the previous
    /// stage's. A caller that changes shape WITHOUT releasing is still
    /// correct, because [`crate::devres::AvDitSession::prefill`] keys its
    /// window on the shape and rebuilds it; it only keeps a slot count
    /// planned for the older one.
    fn session(&self, tv: usize) -> std::sync::Arc<crate::devres::AvDitSession> {
        let key = gpu_core::devices::current_gpu();
        let mut map = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(key)
            .or_insert_with(|| {
                let session = match self.resident_blocks {
                    Some(n) => crate::devres::AvDitSession::resident_with_slots(self.device.as_deref(), n.min(self.cfg.video.num_layers)),
                    None => crate::devres::AvDitSession::resident(&self.cfg, crate::block::QTier::Int8, self.device.as_deref(), tv),
                };
                std::sync::Arc::new(session.with_adapter(self.adapter.clone()))
            })
            .clone()
    }
}

impl Denoiser for RealAvDit {
    /// Same reason [`crate::av_stream::AvDenoiser`]'s is unreachable: the A2V
    /// cross-attention writes into the video residual every block, so there is
    /// no video-only forward of this model that means anything.
    fn forward(&self, _i: &StepInputs, _context: Caption) -> Vec<f32> {
        unreachable!("RealAvDit::forward - denoise takes forward_av for an audio-visual denoiser")
    }

    fn has_audio(&self) -> bool {
        true
    }

    fn forward_av(&self, i: &StepInputs, a: &AudioInputs, context: Caption, a_context: Caption) -> (Vec<f32>, Vec<f32>) {
        let session = self.session(i.t);
        crate::dit::av_forward_q_streamed_in(&session, &self.cfg, &self.src, &self.head, crate::block::QTier::Int8, &av_step(i, a, context, a_context), &self.cache)
    }

    /// Drop every card's AV session, and with it the resident weight window.
    /// [`Self::session`] rebuilds one on the next forward, sized from THAT
    /// forward's token count. The weights are not re-read - they come back
    /// from the checkpoint-scoped host cache this denoiser still holds.
    fn release_devices(&self) {
        let dropped: Vec<_> = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).drain().collect();
        if !dropped.is_empty() {
            tracing::info!(cards = dropped.len(), "releasing the audio-visual DiT's device sessions");
        }
    }
}

/// The real 22B checkpoint's own [`Denoiser`] - `LtxDit`'s counterpart when
/// [`Paths::dit`] is set: holds an open GGUF source plus the small resident
/// "head" tensors ([`crate::dit::load_head_tensors_from_source`]) and, on a
/// generation's first forward call, streams each of the 48 blocks from `src`
/// via [`crate::dit::forward_q_streamed`] - see that function's doc for the
/// memory bound this buys over materializing the whole model as host fp32.
/// Int8 compute (not int4): this milestone's own "start small, prove it
/// works first" choice - see [`generate`]'s doc for why.
///
/// `cache`: a handle onto THE CHECKPOINT's host-side weight cache
/// ([`crate::weightcache`]), not this instance's own - obtained from a
/// process-wide registry keyed on the checkpoint's identity (path + byte
/// length + mtime) and shared by reference across every one of `denoise`'s
/// forward calls (both the conditional and unconditional branch when CFG is
/// on, and every one of the run's denoise steps). It holds the two things
/// `forward_q_streamed` would otherwise recompute identically every call:
/// each block's already-quantized weight bytes (the GGUF read + CPU quantize
/// Phase 8 measured as the dominant share of one real denoise step) and the
/// embeddings-connector routing.
///
/// The scope is the point. A `RealDit` is still per-generation, but the store
/// behind this handle is not: it outlives the `RealDit`, the `generate()`
/// call and the resident instance, so a SECOND generation against the same
/// checkpoint starts warm on its first forward instead of re-reading ~22 GB
/// off a rotational disk at whatever that disk's sequential rate happens to be. What bounds it is the process-wide
/// host ceiling `--limit-ram-total` publishes, evicted by the residency
/// layer's own cost-aware policy - not the lifetime of any one call. Sharing
/// entries across generations is safe for the same reason sharing them across
/// steps was: the entries are a pure function of immutable checkpoint bytes,
/// and the identity key changes whenever the file does.
///
/// Its interior synchronization (an `RwLock`, handing back `Arc`s) is what
/// lets [`Denoiser::forward`] keep taking `&self`, and additionally makes the
/// whole type `Sync` - so a later phase can dispatch the two CFG branches
/// concurrently on two cards without redesigning this.
struct RealDit {
    /// The checkpoint stream this denoiser runs: a video-only checkpoint's
    /// video, or an AV checkpoint's audio alone (`cfg` then being
    /// [`LtxAvDitConfig::audio_stream`]).
    stream: crate::config::Stream,
    cfg: LtxDitConfig,
    src: crate::gguf_src::LtxvGgufSource,
    head: Tensors,
    device: Option<String>,
    cache: crate::block::GenerationCache,
    /// Which card each CFG branch runs on - see [`crate::devplan`]. Resolved
    /// once, in [`generate`], so every step of one generation makes the same
    /// placement decision.
    place: crate::devplan::Placement,
    /// One open device plus its resident weight window PER CARD, created on
    /// that card's first forward and held for the whole generation - see
    /// [`crate::devres`]. Keyed on the card this thread is currently scoped to
    /// (`devices::current_gpu()`, which `crate::devplan::on_gpu` sets), so the
    /// concurrent CFG pair gets two sessions on two cards rather than
    /// contending for one, and a `Single` plan gets exactly one.
    ///
    /// `Mutex<HashMap<_, Arc<_>>>` rather than a plain map: the lock is held
    /// only long enough to clone the `Arc`, never across a forward, so the two
    /// branches never wait on each other. Dropping this whole `RealDit` -
    /// which [`generate`] already does before the VAE decode opens its own
    /// device - releases every card's resident weights.
    sessions: std::sync::Mutex<std::collections::HashMap<Option<u32>, std::sync::Arc<crate::devres::DitSession>>>,
    /// Block-linear LoRA corrections every session attaches ([`block_adapter`]).
    adapter: Option<SharedAdapter>,
    /// [`Runtime::resident_blocks`].
    resident_blocks: Option<u32>,
}

impl RealDit {
    /// This thread's card's session, built on first use.
    ///
    /// Sized from the token count the FIRST forward on that card runs at,
    /// which is the token count every later step of the same generation runs
    /// at too. A caller that changes the shape mid-session gets the window
    /// rebuilt rather than a mismatch (see `crate::devres`).
    fn session(&self, t: usize) -> std::sync::Arc<crate::devres::DitSession> {
        let key = gpu_core::devices::current_gpu();
        let mut map = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(key)
            .or_insert_with(|| {
                let session = match self.resident_blocks {
                    Some(n) => crate::devres::DitSession::resident_with_slots(self.device.as_deref(), n.min(self.cfg.num_layers)),
                    None => crate::devres::DitSession::resident(&self.cfg, crate::block::QTier::Int8, self.device.as_deref(), t),
                };
                std::sync::Arc::new(session.with_adapter(self.adapter.clone()))
            })
            .clone()
    }
}

impl Denoiser for RealDit {
    fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
        let session = self.session(i.t);
        let step = crate::dit::StreamedStep {
            stream: self.stream,
            latent: i.latent,
            timesteps: i.timesteps,
            positions: i.positions,
            keyframes_mask: i.keyframes_mask,
            context: context.rows,
            context_len: i.context_len,
            t: i.t,
            context_valid: context.valid,
            perturbations: i.perturbations,
            reference_attention: i.reference_attention,
        };
        crate::dit::forward_q_streamed_in(&session, &self.cfg, &self.src, &self.head, crate::block::QTier::Int8, &step, &self.cache)
    }

    /// The two forwards, one per card, at the same time.
    ///
    /// Safe to share `&self` across the two threads for reasons that are
    /// properties of the fields, not assumptions about them:
    ///
    /// * `src` is a `MmapGguf` - an immutable mapping, read-only from both;
    /// * `head` is an owned tensor map, never written after construction;
    /// * `cache` is the checkpoint-scoped [`crate::weightcache`] store, whose
    ///   whole `RwLock`+`Arc` design exists so a reader holds no lock across
    ///   its device upload. Both branches therefore hit the SAME already-
    ///   quantized host bytes and upload them independently to their own
    ///   card, which is why two cards cost one checkpoint read, not two;
    /// * `cfg`/`device`/`place` are plain `Copy`/immutable data.
    ///
    /// The connector-routing half of the cache is read by both branches too,
    /// and they genuinely differ there (the conditional and unconditional
    /// contexts are different inputs, so they occupy two of the store's four
    /// connector slots) - it is `MAX_CONNECTOR_ENTRIES = 4` precisely so one
    /// generation's pair fits without either branch evicting the other's.
    ///
    /// * `sessions` is a `Mutex` map keyed on the card each branch is scoped
    ///   to, so each branch builds and then reuses ITS OWN open device and its
    ///   own resident weight window. The lock is held only long enough to
    ///   clone an `Arc`, never across a forward.
    ///
    /// No device handle crosses a thread boundary: each branch resolves its
    /// session from the card `crate::devplan::on_gpu` scoped it to, and both
    /// cards' VRAM is released together when this `RealDit` drops.
    fn forward_cfg_pair(&self, i: &StepInputs, cond: Caption, uncond: Caption) -> Result<(Vec<f32>, Vec<f32>), String> {
        dispatch_cfg_pair(self, &self.place, i, cond, uncond)
    }

    /// Drop every card's session: the open `Gpu` and, with it, the resident
    /// weight window that session was holding. [`Self::session`] rebuilds one
    /// on the next forward, sized from THAT forward's token count.
    ///
    /// The weights themselves are not re-read - they come back from the
    /// checkpoint-scoped host cache this `RealDit` still holds - so what a
    /// release costs is a device open plus the window's uploads, and what it
    /// buys is a card with nothing of this denoiser's on it.
    fn release_devices(&self) {
        let dropped: Vec<_> = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).drain().collect();
        if !dropped.is_empty() {
            tracing::info!(cards = dropped.len(), "releasing the DiT's device sessions");
        }
    }
}

/// Run one CFG step's two forwards under `place`: sequentially when both
/// branches land on the same device, and concurrently - one thread per card,
/// each scoped with `with_gpu` - when they do not.
///
/// A free function rather than a method body so the gate
/// (`the_concurrent_cfg_pair_is_bit_identical_to_the_sequential_one`) drives
/// THIS code with a per-call-device denoiser it can build in milliseconds,
/// instead of a copy of it that could drift from what `RealDit` really runs.
///
/// `D: Sync` is the whole safety argument, checked by the compiler at every
/// call site rather than asserted in prose: a denoiser that could not be
/// shared across the two threads cannot reach this function.
fn dispatch_cfg_pair<D: Denoiser + Sync>(dit: &D, place: &crate::devplan::Placement, i: &StepInputs, cond: Caption, uncond: Caption) -> Result<(Vec<f32>, Vec<f32>), String> {
    if !place.cfg_is_parallel() {
        return Ok((dit.forward(i, cond), dit.forward(i, uncond)));
    }
    let (cond_gpu, uncond_gpu) = (place.cond, place.uncond);
    std::thread::scope(|s| {
        let u = s.spawn(move || crate::devplan::on_gpu(uncond_gpu, || dit.forward(i, uncond)));
        let c = crate::devplan::on_gpu(cond_gpu, || dit.forward(i, cond));
        // Join before propagating either error: an early return would drop
        // the scope's guard and block on the same join anyway, and reporting
        // the conditional branch's failure while the unconditional one is
        // still uploading 13 GB reads as a hang.
        let u = u.join().map_err(|_| format!("ltxv: the unconditional CFG branch panicked on gpu{uncond_gpu:?}"))?;
        Ok((c?, u?))
    })
}

/// `to_denoised` (`ltx_core.utils.to_denoised`): the model predicts a
/// velocity; the denoised (x0) estimate the stepper needs is `sample -
/// velocity * timestep`.
///
/// **`timesteps` is PER TOKEN, not the schedule's scalar sigma.** The
/// reference applies this conversion inside the model wrapper -
/// `ltx_core.model.transformer.model.X0Model.forward` calls
/// `to_denoised(video.latent, vx, video.timesteps)`, and `Modality.timesteps`
/// is `timesteps_from_mask(denoise_mask, sigma)` = `denoise_mask * sigma`,
/// shape `(B, T, 1)`, broadcast over the channel axis of the `(B, T, C)`
/// latent. A frozen image-conditioning token therefore converts at timestep
/// `0`, where the formula is the IDENTITY: its clean content passes through
/// untouched.
///
/// Feeding the scalar sigma here instead is exactly correct for plain
/// text-to-video (`denoise_mask` is all ones, so every token's timestep IS
/// the sigma) and silently wrong the moment anything is frozen - see
/// [`a_frozen_token_survives_the_terminal_step_exactly`](self) for the
/// mechanism and the ledger entry for the measured damage.
///
/// `channels` is the latent width; `timesteps.len() * channels ==
/// sample.len()`.
fn to_denoised(sample: &[f32], velocity: &[f32], timesteps: &[f32], channels: usize) -> Vec<f32> {
    assert!(channels > 0, "to_denoised: channels must be nonzero");
    debug_assert_eq!(sample.len(), timesteps.len() * channels);
    debug_assert_eq!(velocity.len(), sample.len());
    sample
        .chunks_exact(channels)
        .zip(velocity.chunks_exact(channels))
        .zip(timesteps)
        .flat_map(|((x, v), &ts)| x.iter().zip(v).map(move |(&x, &v)| (x as f64 - v as f64 * ts as f64) as f32))
        .collect()
}

/// A held-fixed image-conditioning range within the denoised latent -
/// `mask[tok] < 1.0` pulls that token toward `clean[tok]` every step
/// (`ltx_pipelines.utils.helpers.post_process_latent`: `denoised*mask +
/// clean*(1-mask)`). `mask`/mask-implicit token count is `clean.len() /
/// channels`; every token not covered here denoises normally (`mask[tok] ==
/// 1.0`).
///
/// **HOW MANY TIMES per step it is applied depends on the sampler**, and the
/// reference's two loops genuinely differ:
///
/// * Deterministic Euler (`samplers.euler_denoising_loop` -> `_step_state`):
///   ONCE, on the model's x0 ESTIMATE, before the step formula runs on it.
/// * Ancestral Euler (`samplers._ancestral_euler_denoising_loop`, which is
///   what `euler_ancestral_denoising_loop` and therefore LTX-2.5's own
///   distilled stage 1 run): TWICE. Once on the x0 estimate
///   (`_ModalityStep.from_modality_result`, for every step including the
///   terminal one it short-circuits to), and again on the STEPPED latent
///   `x_next` after the renoise term has been added (`if draw_noise: x_next =
///   post_process_latent(x_next, ...)`, skipped on the terminal step because
///   there is no stepped latent there).
///
/// [`denoise`] implements both, selected by `eta`. Dropping the ancestral
/// loop's second application would leave freshly injected noise sitting on a
/// token that is supposed to be clean; dropping either loop's first one lets
/// a PARTIALLY conditioned token (`mask` strictly between 0 and 1) step from
/// an unblended estimate. Neither is visible at `mask == 0` or `mask == 1`,
/// which is why the gate that pins it
/// ([`tests::a_partially_conditioned_token_is_pulled_to_its_clean_content_under_both_samplers`])
/// had to be built at a strength in between.
struct Frozen<'a> {
    mask: &'a [f32],
    clean: &'a [f32],
    channels: usize,
    /// The self-attention bias a weighted IC-LoRA reference among these
    /// tokens carries ([`crate::refcond::reference_attention`]); `None` when
    /// every conditioning token attends at full strength.
    attention: Option<&'a crate::refcond::ReferenceAttention>,
}

fn post_process_latent(denoised: &mut [f32], frozen: &Frozen) {
    let t = frozen.mask.len();
    debug_assert_eq!(denoised.len(), t * frozen.channels);
    debug_assert_eq!(frozen.clean.len(), t * frozen.channels);
    for tok in 0..t {
        let m = frozen.mask[tok];
        if m == 1.0 {
            continue;
        }
        let off = tok * frozen.channels;
        for c in 0..frozen.channels {
            denoised[off + c] = m * denoised[off + c] + (1.0 - m) * frozen.clean[off + c];
        }
    }
}

/// [`post_process_latent`] on the res_2s loop's f64 anchors.
fn post_process_latent_f64(x: &mut [f64], frozen: &Frozen) {
    for (tok, &m) in frozen.mask.iter().enumerate() {
        if m == 1.0 {
            continue;
        }
        let off = tok * frozen.channels;
        for (v, &c) in x[off..off + frozen.channels].iter_mut().zip(&frozen.clean[off..off + frozen.channels]) {
            *v = m as f64 * *v + (1.0 - m as f64) * c as f64;
        }
    }
}

/// Which of the reference's denoising loops a stage runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sampler {
    /// `euler_denoising_loop` at `eta = 0`, `euler_ancestral_denoising_loop`
    /// above it - one guided evaluation per step. What the distilled
    /// checkpoint's own pipeline runs.
    #[default]
    Euler,
    /// `res2s_audio_video_denoising_loop` - a second-order exponential
    /// integrator: two guided evaluations per step (the second at the
    /// geometric-midpoint sigma) with SDE noise at both, plus one final
    /// evaluation that removes the last of the noise. What the reference's
    /// high-quality two-stage pipeline (`ti2vid_two_stages_hq`) runs.
    Res2s,
}

impl Sampler {
    /// The name the CLI and the capability take.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "euler" => Ok(Self::Euler),
            "res2s" | "res_2s" => Ok(Self::Res2s),
            other => Err(format!("unknown sampler {other:?}: expected euler or res2s")),
        }
    }

    /// The reference loop's own default noise strength: `ANCESTRAL_ETA = 1.0`
    /// for the ancestral Euler loop, `eta = 0.5` for res_2s.
    pub fn default_eta(self) -> f64 {
        match self {
            Self::Euler => 1.0,
            Self::Res2s => 0.5,
        }
    }

    /// `eta` when one was given, otherwise [`Self::default_eta`]; either way
    /// within `0..=1`, the range both loops define it on.
    pub fn resolve_eta(self, eta: Option<f64>) -> Result<f64, String> {
        let eta = eta.unwrap_or(self.default_eta());
        if !(0.0..=1.0).contains(&eta) {
            return Err(format!("eta must be within 0..=1, got {eta}"));
        }
        Ok(eta)
    }
}

/// How one stage steps its schedule: the loop, its noise strength and the
/// seed its noise is drawn from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sampling {
    sampler: Sampler,
    /// Euler: [`euler_ancestral_step`]'s eta, `0` a plain Euler step. res_2s:
    /// the step-level SDE strength (`sigma_up = eta * sigma_next`); its
    /// substep always injects at `0.5`, as the reference fixes.
    eta: f64,
    /// [`euler_ancestral_step`]'s `s_noise`; res_2s has none.
    s_noise: f64,
    seed: u64,
}

#[cfg(test)]
impl Sampling {
    fn euler(eta: f64, s_noise: f64, seed: u64) -> Self {
        Self { sampler: Sampler::Euler, eta, s_noise, seed }
    }
}

/// The guided model at one point of the schedule: every guidance pass of
/// [`crate::guidance`] run at a given latent and sigma, each turned into its
/// x0 estimate, combined per stream and pinned to the conditioning
/// ([`post_process_latent`]). Every sampler is built on this one unit, so
/// the per-token timesteps, the pass set and the masking are the same
/// whichever loop steps the result.
struct GuidedModel<'a> {
    dit: &'a dyn Denoiser,
    positions: &'a [f32],
    keyframes_mask: &'a [f32],
    ctx_cond: Caption<'a>,
    ctx_uncond: Caption<'a>,
    context_len: usize,
    t: usize,
    /// The video latent's channel width - the axis `Modality.timesteps`
    /// `(B, T, 1)` broadcasts over in [`to_denoised`].
    channels: usize,
    guidance: &'a crate::guidance::Guidance,
    passes: Vec<crate::guidance::Pass>,
    frozen: Option<&'a Frozen<'a>>,
    /// The audio stream's fixed inputs; its latent is passed per evaluation.
    audio: Option<&'a AudioState<'a>>,
    a_channels: usize,
}

impl GuidedModel<'_> {
    /// `timesteps_from_mask(denoise_mask, sigma)` (`ltx_pipelines.utils.
    /// helpers`, reached from `modality_from_latent_state`): the model is
    /// told, PER TOKEN, how noisy that token is. A frozen token holds clean
    /// content, so its timestep is `0 * sigma == 0`. Getting this wrong is not
    /// a small error: the DiT's AdaLN modulation is computed from this
    /// per-token timestep (`crate::dit::ada_layer_norm_single` builds one
    /// shift/scale ROW per token), so a clean token labelled "as noisy as
    /// everything else" is modulated as if it were pure noise - and, through
    /// self-attention, drags every genuinely-noisy token with it. The same
    /// holds for a long-form window's carried audio prefix.
    fn timesteps(mask: Option<&[f32]>, tokens: usize, sigma: f64) -> Vec<f32> {
        match mask {
            Some(m) => m.iter().map(|&m| m * sigma as f32).collect(),
            None => vec![sigma as f32; tokens],
        }
    }

    fn audio_mask(&self) -> Option<&[f32]> {
        self.audio.and_then(|a| a.frozen.as_ref().map(|(m, _)| m.as_slice()))
    }

    /// The masked, guided x0 of both streams at `latent` (and `audio_latent`
    /// when the model generates audio) and `sigma`. `step` only labels
    /// traces and errors.
    fn x0(&self, latent: &[f32], audio_latent: Option<&[f32]>, sigma: f64, step: usize) -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
        use crate::block::NO_PERTURBATIONS;
        use crate::guidance::Pass;
        let timesteps = Self::timesteps(self.frozen.map(|f| f.mask), self.t, sigma);
        let base = StepInputs {
            latent,
            timesteps: &timesteps,
            positions: self.positions,
            keyframes_mask: self.keyframes_mask,
            context_len: self.context_len,
            t: self.t,
            sigma: stream_sigma(sigma, self.frozen.map(|f| f.mask)),
            perturbations: &NO_PERTURBATIONS,
            reference_attention: self.frozen.and_then(|f| f.attention),
        };
        let x0 = |v: Vec<f32>| to_denoised(latent, &v, &timesteps, self.channels);
        // The conditional pass first; the others only when some stream's
        // guidance needs them.
        let passes = &self.passes;
        let (cond, others): (Latents, Vec<Latents>) = match (self.audio, audio_latent) {
            (Some(a), Some(a_latent)) => {
                let a_timesteps = Self::timesteps(self.audio_mask(), a.ta, sigma);
                let ai = AudioInputs { latent: a_latent, timesteps: &a_timesteps, positions: &a.positions, ta: a.ta, sigma: stream_sigma(sigma, self.audio_mask()) };
                let a_x0 = |v: Vec<f32>| to_denoised(a_latent, &v, &a_timesteps, self.a_channels);
                let run = |ptb: &crate::block::Perturbations, v_ctx: Caption, a_ctx: Caption| {
                    let (v, av) = self.dit.forward_av(&base.with_perturbations(ptb), &ai, v_ctx, a_ctx);
                    (x0(v), Some(a_x0(av)))
                };
                // Both streams' projections of one prompt share its mask.
                let (a_cond, a_uncond) = (Caption::of(a.ctx_cond, self.ctx_cond.valid), Caption::of(a.ctx_uncond, self.ctx_uncond.valid));
                tracing::trace!(step, forwards = 1 + passes.len(), sigma, "joint audio+video guidance passes starting");
                let cond = run(&NO_PERTURBATIONS, self.ctx_cond, a_cond);
                let others = passes
                    .iter()
                    .map(|p| match p {
                        Pass::Uncond => run(&NO_PERTURBATIONS, self.ctx_uncond, a_uncond),
                        Pass::Perturbed(ptb) | Pass::Isolated(ptb) => run(ptb, self.ctx_cond, a_cond),
                    })
                    .collect();
                (cond, others)
            }
            (None, None) if passes.as_slice() == [Pass::Uncond] => {
                // CFG alone keeps its two-card pair seam.
                tracing::trace!(step, branch = "cond+uncond", sigma, "forward pair starting");
                let (cond, uncond) = self.dit.forward_cfg_pair(&base, self.ctx_cond, self.ctx_uncond)?;
                ((x0(cond), None), vec![(x0(uncond), None)])
            }
            (None, None) => {
                tracing::trace!(step, forwards = 1 + passes.len(), sigma, "guidance passes starting");
                let cond = (x0(self.dit.forward(&base, self.ctx_cond)), None);
                let others = passes
                    .iter()
                    .map(|p| match p {
                        Pass::Uncond => (x0(self.dit.forward(&base, self.ctx_uncond)), None),
                        Pass::Perturbed(ptb) | Pass::Isolated(ptb) => (x0(self.dit.forward(&base.with_perturbations(ptb), self.ctx_cond)), None),
                    })
                    .collect();
                (cond, others)
            }
            _ => unreachable!("an audio latent is passed exactly when the model carries an audio stream"),
        };
        let pass_of = |want: fn(&Pass) -> bool| passes.iter().position(want);
        let (uncond_at, ptb_at, iso_at) = (pass_of(|p| matches!(p, Pass::Uncond)), pass_of(|p| matches!(p, Pass::Perturbed(_))), pass_of(|p| matches!(p, Pass::Isolated(_))));
        let video_pass = |at: Option<usize>| at.map(|k| others[k].0.as_slice());
        // Without an audio stream there is nothing to cut, so the isolated
        // pass would be the conditional one itself and the modality term is
        // zero - which is what passing `cond` for it says, without the forward.
        let video_isolated = if self.audio.is_some() { video_pass(iso_at) } else { Some(cond.0.as_slice()) };
        let mut denoised = self.guidance.video.combine(&cond.0, video_pass(uncond_at), video_pass(ptb_at), video_isolated, self.frozen.map(|f| f.mask), self.channels);
        if !denoised.iter().all(|v| v.is_finite()) {
            let bad = denoised.iter().filter(|v| !v.is_finite()).count();
            tracing::error!(step, sigma, non_finite = bad, of = denoised.len(), "the denoiser produced non-finite values");
            return Err(format!("the denoiser produced non-finite values at step {step} (sigma = {sigma:.4})"));
        }
        // Every reference loop masks the x0 estimate before stepping it
        // (`_step_state`, `_ModalityStep.from_modality_result`, and both
        // res_2s evaluations), including the terminal one.
        if let Some(f) = self.frozen {
            post_process_latent(&mut denoised, f);
        }
        let audio_denoised = match &cond.1 {
            Some(a_cond) => {
                let audio_pass = |at: Option<usize>| at.map(|k| others[k].1.as_deref().expect("a joint pass carries the audio stream"));
                let mut a = self.guidance.audio.combine(a_cond, audio_pass(uncond_at), audio_pass(ptb_at), audio_pass(iso_at), self.audio_mask(), self.a_channels);
                if !a.iter().all(|v| v.is_finite()) {
                    return Err(format!("the denoiser produced non-finite audio values at step {step} (sigma = {sigma:.4})"));
                }
                if let Some((mask, clean)) = self.audio.and_then(|s| s.frozen.as_ref()) {
                    post_process_latent(&mut a, &Frozen { mask, clean, channels: self.a_channels, attention: None });
                }
                Some(a)
            }
            None => None,
        };
        Ok((denoised, audio_denoised))
    }

    /// The audio stream's frozen prefix as a [`Frozen`], when it has one.
    fn audio_frozen(&self) -> Option<Frozen<'_>> {
        self.audio.and_then(|a| a.frozen.as_ref()).map(|(mask, clean)| Frozen { mask, clean, channels: self.a_channels, attention: None })
    }
}

/// Denoise `latent` (and, beside it, `audio`'s latent) down `sigmas` with
/// the guided model and the loop `sampling` names.
///
/// Both streams step ONE schedule in lockstep - the reference's loops
/// advance `video_state` and `audio_state` off the same `sigmas` at the same
/// step. That lockstep is not a simplification: it is what keeps the two
/// streams' sigmas consistent with the cross-modality gates each block
/// computes from the OTHER stream's sigma.
///
/// Traced (`--trace-ltxv`): the loop is one span, each step a `debug!` with
/// its sigma pair and running seconds-per-step, each individual forward a
/// `trace!`, a cancellation a `warn!` naming the step it stopped at, and a
/// non-finite prediction an `error!` - so a run that diverged or stalled can
/// be pinpointed from the trace alone instead of re-run under a profiler.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "info", name = "denoise", skip_all, fields(steps = sigmas.len().saturating_sub(1), tokens = t, sampler = ?sampling.sampler, eta = sampling.eta))]
fn denoise(
    dit: &dyn Denoiser,
    sigmas: &[f64],
    latent: Vec<f32>,
    positions: &[f32],
    keyframes_mask: &[f32],
    ctx_cond: Caption,
    ctx_uncond: Caption,
    context_len: usize,
    t: usize,
    guidance: &crate::guidance::Guidance,
    sampling: Sampling,
    total: u32,
    frozen: Option<&Frozen>,
    cancel: &capability::CancelToken,
    audio: Option<&mut AudioState<'_>>,
    progress: &mut impl FnMut(u32, u32, &str),
) -> Result<Vec<f32>, String> {
    guidance.validate()?;
    if audio.is_some() && !dit.has_audio() {
        return Err("ltxv: an audio stream was supplied to a denoiser that does not generate audio".into());
    }
    // The audio latent is stepped beside the video's; the rest of the
    // stream's state is the model's fixed input.
    let mut audio = audio;
    let a_latent = audio.as_deref_mut().map(|a| std::mem::take(&mut a.latent));
    let model = GuidedModel {
        dit,
        positions,
        keyframes_mask,
        ctx_cond,
        ctx_uncond,
        context_len,
        t,
        channels: latent.len().checked_div(t).unwrap_or(1),
        guidance,
        passes: guidance.passes(audio.is_some()),
        frozen,
        audio: audio.as_deref(),
        a_channels: audio.as_deref().map(|a| a_latent.as_ref().map_or(0, Vec::len).checked_div(a.ta).unwrap_or(1)).unwrap_or(1),
    };
    let t0 = Instant::now();
    tracing::info!(steps = sigmas.len().saturating_sub(1), forwards_per_step = 1 + model.passes.len(), sigma_first = sigmas.first().copied().unwrap_or(0.0), sigma_last = sigmas.last().copied().unwrap_or(0.0), frozen_tokens = frozen.map(|f| f.mask.iter().filter(|&&m| m == 0.0).count()), "denoise loop starting");
    let mut on_step = |i: usize, sigma: f64, sigma_next: f64| {
        let steps = sigmas.len().saturating_sub(1);
        let per = t0.elapsed().as_secs_f32() / (i + 1) as f32;
        tracing::debug!(step = i + 1, steps, sigma, sigma_next, secs_per_step = per, "step done");
        progress(i as u32 + 1, total, &format!("denoise sigma={sigma:.3} {per:.2}s/step"));
    };
    let (latent, a_latent) = match sampling.sampler {
        Sampler::Euler => euler_loop(&model, sigmas, latent, a_latent, sampling, cancel, &mut on_step)?,
        Sampler::Res2s => res2s_loop(&model, sigmas, latent, a_latent, sampling, cancel, &mut on_step)?,
    };
    if let (Some(a), Some(l)) = (audio, a_latent) {
        a.latent = l;
    }
    tracing::info!(secs = t0.elapsed().as_secs_f32(), "denoise loop done");
    Ok(latent)
}

/// A stream pair being stepped: the video latent and, when the model
/// generates audio, the audio latent beside it.
type Latents = (Vec<f32>, Option<Vec<f32>>);

/// The check every loop makes between steps: a forward is one submit of the
/// whole block stack and is not interruptible from inside, same reasoning as
/// `wan::pipeline::denoise`.
fn check_cancel(cancel: &capability::CancelToken, step: usize, steps: usize) -> Result<(), String> {
    if cancel.is_cancelled() {
        tracing::warn!(step, steps, "cancelled at a step boundary; aborting the denoise loop");
        return Err("cancelled".into());
    }
    Ok(())
}

/// The reference's two Euler loops, selected by `eta`.
///
/// LTX-2.5's distilled stage 1 is `euler_ancestral_denoising_loop` with
/// `ANCESTRAL_ETA = 1.0` / `ANCESTRAL_S_NOISE = 1.0` (`ltx_pipelines.
/// distilled` picks it for every checkpoint at or above `(2, 5)`), not the
/// deterministic loop. The difference matters most exactly where
/// conditioning is strongest: the first four sigmas of the distilled
/// schedule move by 0.006 each, which is a near-no-op without a renoise
/// term, and a deterministic trajectory pulled by clean conditioning tokens
/// has nothing to push it off the "hold the still" solution.
///
/// The ancestral loop steps the whole latent with the SCALAR schedule and
/// then re-applies `post_process_latent` to the STEPPED result (skipped on
/// the terminal step), which needs no per-token sigma. See [`Frozen`]'s doc
/// for the two orderings.
fn euler_loop(model: &GuidedModel, sigmas: &[f64], mut latent: Vec<f32>, mut a_latent: Option<Vec<f32>>, sampling: Sampling, cancel: &capability::CancelToken, on_step: &mut dyn FnMut(usize, f64, f64)) -> Result<Latents, String> {
    let Sampling { eta, s_noise, seed, .. } = sampling;
    let ancestral = eta > 0.0;
    let mut noise_rng = data::rng::Rng::new(seed);
    let mut draw = |n: usize| if ancestral { Some((0..n).map(|_| noise_rng.next_gaussian() as f32).collect::<Vec<f32>>()) } else { None };
    let steps = sigmas.len().saturating_sub(1);
    for i in 0..steps {
        check_cancel(cancel, i + 1, steps)?;
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let (denoised, a_denoised) = model.x0(&latent, a_latent.as_deref(), sigma, i + 1)?;
        let noise = draw(latent.len());
        latent = euler_ancestral_step(&latent, &denoised, sigma, sigma_next, eta, s_noise, noise.as_deref());
        if let (Some(f), true) = (model.frozen, ancestral && sigma_next != 0.0) {
            post_process_latent(&mut latent, f);
        }
        // The audio stream takes the SAME step with its own draw of
        // ancestral noise. Its noise comes from the same generator, after the
        // video's, so a seed still reproduces the whole run exactly - and the
        // two streams never share numbers, which they would if one draw were
        // reused.
        if let (Some(a), Some(a_denoised)) = (a_latent.as_mut(), a_denoised) {
            let a_noise = draw(a.len());
            *a = euler_ancestral_step(a, &a_denoised, sigma, sigma_next, eta, s_noise, a_noise.as_deref());
            if let (Some(f), true) = (model.audio_frozen(), ancestral && sigma_next != 0.0) {
                post_process_latent(a, &f);
            }
        }
        on_step(i, sigma, sigma_next);
    }
    Ok((latent, a_latent))
}

/// `_get_new_noise` (`ltx_pipelines.utils.samplers`): a standard-normal
/// draw re-standardised to exactly zero mean and unit (unbiased) standard
/// deviation over the whole latent. The reference's following
/// `_channelwise_normalize` over the `(tokens, channels)` axes of a
/// single-batch latent is that same standardisation again.
fn standardized_noise(rng: &mut data::rng::Rng, n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
    if n > 1 {
        let mean = v.iter().sum::<f64>() / n as f64;
        let std = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64).sqrt();
        v.iter_mut().for_each(|x| *x = (*x - mean) / std);
    }
    v
}

/// `res2s_audio_video_denoising_loop` (`ltx_pipelines.utils.samplers`) in
/// its defaults: `c2 = 0.5`, bong iteration on, `legacy_mode = True` (the
/// SDE step uses the scalar schedule and the conditioning is re-pinned on
/// the stepped latent afterwards).
///
/// Per step, from the anchor `x` at `sigma`:
///
/// 1. `x0_1` = the guided model at `(x, sigma)`, `eps_1 = x0_1 - x`.
/// 2. `x_mid = x + h*a21*eps_1` with `h = ln(sigma/sigma_next)`, then SDE
///    noise from `sigma` down to `sub = sqrt(sigma*sigma_next)` at eta 0.5
///    (the substep's own generator, seeded `seed + 10000`).
/// 3. Bong iteration (`h < 0.5`, `sigma > 0.03`): 100 rounds of re-solving
///    the anchor so that `x_mid` is exactly what step 2 would produce from
///    it, which folds the substep noise back into the anchor.
/// 4. `x0_2` = the guided model at `(x_mid, sub)`, `eps_2 = x0_2 - x`.
/// 5. `x_next = x + h*(b1*eps_1 + b2*eps_2)`, then SDE noise to `sigma_next`
///    at `eta` from the step generator.
///
/// A schedule ending at `0` has its last sigma replaced by `[0.0011, 0]` (a
/// zero would make `h` infinite) and ends with one more evaluation at
/// `0.0011` whose x0 is the result. Anchors are f64, as the reference keeps
/// them; the latent between steps is f32.
fn res2s_loop(model: &GuidedModel, sigmas: &[f64], latent: Vec<f32>, a_latent: Option<Vec<f32>>, sampling: Sampling, cancel: &capability::CancelToken, on_step: &mut dyn FnMut(usize, f64, f64)) -> Result<Latents, String> {
    const SUBSTEP_ETA: f64 = 0.5;
    const BONG_ITERATIONS: usize = 100;
    let steps = sigmas.len().saturating_sub(1);
    let terminal = sigmas.last() == Some(&0.0);
    let mut sigmas = sigmas.to_vec();
    if terminal {
        sigmas.pop();
        sigmas.extend([0.0011, 0.0]);
    }
    let mut step_rng = data::rng::Rng::new(sampling.seed);
    let mut substep_rng = data::rng::Rng::new(sampling.seed.wrapping_add(10_000));
    let widen = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let narrow = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    // The SDE injection both stages share: `Res2sDiffusionStep.step` from
    // `anchor` toward `target`, then the conditioning re-pinned on the
    // result (`legacy_mode`).
    let inject = |rng: &mut data::rng::Rng, anchor: &[f64], target: &[f64], sigma: f64, sigma_next: f64, eta: f64, frozen: Option<&Frozen>| {
        let noise = standardized_noise(rng, anchor.len());
        let mut x = diffusion::scheduler::res2s_sde_step(anchor, target, sigma, sigma_next, eta, &noise);
        if let Some(f) = frozen {
            post_process_latent_f64(&mut x, f);
        }
        x
    };
    let mut streams: Vec<Vec<f64>> = std::iter::once(widen(&latent)).chain(a_latent.as_deref().map(widen)).collect();
    let frozen_of = |k: usize| if k == 0 { model.frozen.map(|f| Frozen { mask: f.mask, clean: f.clean, channels: f.channels, attention: f.attention }) } else { model.audio_frozen() };
    for i in 0..steps {
        check_cancel(cancel, i + 1, steps)?;
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let h = -(sigma_next / sigma).ln();
        let c = diffusion::scheduler::res2s_coefficients(h);
        let sub_sigma = (sigma * sigma_next).sqrt();

        let (v1, a1) = model.x0(&narrow(&streams[0]), streams.get(1).map(|a| narrow(a)).as_deref(), sigma, i + 1)?;
        let x0_1: Vec<Vec<f64>> = std::iter::once(widen(&v1)).chain(a1.as_deref().map(widen)).collect();
        let mut anchors = streams.clone();
        let mut eps_1: Vec<Vec<f64>> = x0_1.iter().zip(&anchors).map(|(d, x)| d.iter().zip(x).map(|(d, x)| d - x).collect()).collect();
        let mids: Vec<Vec<f64>> = (0..streams.len())
            .map(|k| {
                let target: Vec<f64> = anchors[k].iter().zip(&eps_1[k]).map(|(x, e)| x + h * c.a21 * e).collect();
                inject(&mut substep_rng, &anchors[k], &target, sigma, sub_sigma, SUBSTEP_ETA, frozen_of(k).as_ref())
            })
            .collect();
        if h < 0.5 && sigma > 0.03 {
            for k in 0..streams.len() {
                for _ in 0..BONG_ITERATIONS {
                    anchors[k] = mids[k].iter().zip(&eps_1[k]).map(|(m, e)| m - h * c.a21 * e).collect();
                    eps_1[k] = x0_1[k].iter().zip(&anchors[k]).map(|(d, x)| d - x).collect();
                }
            }
        }

        let (v2, a2) = model.x0(&narrow(&mids[0]), mids.get(1).map(|a| narrow(a)).as_deref(), sub_sigma, i + 1)?;
        let x0_2: Vec<Vec<f64>> = std::iter::once(widen(&v2)).chain(a2.as_deref().map(widen)).collect();
        for k in 0..streams.len() {
            let target: Vec<f64> = anchors[k].iter().zip(&eps_1[k]).zip(&x0_2[k]).map(|((x, e1), d2)| x + h * (c.b1 * e1 + c.b2 * (d2 - x))).collect();
            let next = inject(&mut step_rng, &anchors[k], &target, sigma, sigma_next, sampling.eta, frozen_of(k).as_ref());
            // The reference stores the stepped latent in the model's dtype.
            streams[k] = next.iter().map(|&x| x as f32 as f64).collect();
        }
        on_step(i, sigma, sigma_next);
    }
    let (mut latent, mut a_latent) = (narrow(&streams[0]), streams.get(1).map(|a| narrow(a)));
    if terminal {
        check_cancel(cancel, steps + 1, steps)?;
        (latent, a_latent) = model.x0(&latent, a_latent.as_deref(), sigmas[steps], steps + 1)?;
    }
    Ok((latent, a_latent))
}

/// Load and resize one conditioning still to the VAE's `[-1, 1]` input
/// range, `[3, height, width]` row-major - the part of the conditioning
/// encodes that does not care how many frames the VAE will be fed.
fn load_still_chw(path: &str, width: usize, height: usize) -> Result<Vec<f32>, String> {
    let img = image::open(path).map_err(|e| format!("{path}: {e}"))?.resize_exact(width as u32, height as u32, image::imageops::FilterType::Lanczos3).to_rgb8();
    Ok(imaging::pixels::hwc_to_chw(img.as_raw(), 3, height, width).into_iter().map(vae_input).collect())
}

/// `[0,255] -> [-1,1]`, the video VAE's own input range (see
/// `LtxVaeEncoder::encode`'s doc).
fn vae_input(v: u8) -> f32 {
    v as f32 / 127.5 - 1.0
}

/// Encode one still through the real video VAE at `width`x`height`, as the
/// `[lh*lw, channels]` token block a conditioning item takes.
///
/// A free function rather than [`generate`]'s old inline closure because a
/// two-stage generation encodes the SAME still twice, once per stage, at that
/// stage's own resolution - which is what
/// `ltx_pipelines.distilled.DistilledPipeline.__call__` does (it calls
/// `combined_image_conditionings` separately for stage 1 and stage 2, with
/// `stage_1_h/stage_1_w` and then `height/width`).
///
/// This is also the right content for the mid anchor's PIN, which sits in an
/// interior latent slot whose decoder normally aggregates 8 pixel frames:
/// fed the same still repeated across its span instead, the encoder produced
/// a bit-identical trajectory end to end (a constant signal's temporal
/// aggregation IS its one-frame latent), so the pin keeps the cheap
/// image-path encode.
fn encode_still(vcfg: &LtxVaeConfig, vweights: &vae::blocks::Tensors, path: &str, width: usize, height: usize, channels: usize, device: Option<&str>) -> Result<Vec<f32>, String> {
    let (lh, lw) = (height / 32, width / 32);
    let img_t = Instant::now();
    let img_chw = load_still_chw(path, width, height)?;
    let enc = LtxVaeEncoder::build(vcfg, vweights, 1, height as u32, width as u32, device);
    let cond_latent_chw = enc.encode(&img_chw);
    let cond_tokens = chw_to_tc(&cond_latent_chw, channels, 1, lh, lw);
    tracing::info!(path, width, height, secs = img_t.elapsed().as_secs_f32(), cond_tokens = cond_tokens.len() / channels, "conditioning image encoded");
    Ok(cond_tokens)
}

/// Folded into the seed of an IC-LoRA reference's own noise draw, so it never
/// repeats the base latent's numbers.
const REFERENCE_SEED_SALT: u64 = 0x52_45_46_56_49_44; // "REFVID"
/// Folded into the noise generated keyframe slots start from.
const SLOT_SEED_SALT: u64 = 0x53_4c_4f_54; // "SLOT"

/// A reference clip's `[f*h*w, channels]` latent tokens and its latent
/// `(f, h, w)`.
type ReferenceTokens = (Vec<f32>, (usize, usize, usize));

/// The IC-LoRA reference clip as the conditioning item takes it
/// (`append_ic_lora_reference_video_conditionings`): its first `frames`
/// frames, resized to `width/downscale x height/downscale`, temporally
/// subsampled (frame 0, then every `temporal`-th), VAE-encoded. Returns the
/// `[f*h*w, channels]` tokens and the latent `(f, h, w)`.
#[allow(clippy::too_many_arguments)]
fn encode_reference(vcfg: &LtxVaeConfig, vweights: &vae::blocks::Tensors, r: &VideoReference, frames: usize, width: usize, height: usize, downscale: usize, temporal: usize, channels: usize, device: Option<&str>) -> Result<ReferenceTokens, String> {
    if !width.is_multiple_of(32 * downscale) || !height.is_multiple_of(32 * downscale) {
        return Err(format!("{width}x{height} at reference_downscale_factor {downscale} gives a {}x{} reference, which is not on the VAE's 32-pixel grid", width / downscale, height / downscale));
    }
    if r.clip.frames.len() < frames {
        return Err(format!("the reference video has {} frames, fewer than the {frames} being generated", r.clip.frames.len()));
    }
    let (rw, rh) = (width / downscale, height / downscale);
    let (sw, sh) = (r.clip.width, r.clip.height);
    let picked: Vec<usize> = std::iter::once(0).chain((1..frames).step_by(temporal)).collect();
    let n = picked.len();
    let lat_f = vcfg.latent_frames(n as u32).ok_or_else(|| format!("the reference keeps {n} frames after temporal_scale_factor {temporal}, which is not of the form 1 + 8k"))? as usize;
    let resized: Vec<Vec<u8>> = picked
        .iter()
        .map(|&i| {
            let img = image::RgbImage::from_raw(sw, sh, r.clip.frames[i].clone()).ok_or_else(|| format!("reference frame {i} is not {sw}x{sh} RGB8"))?;
            Ok(image::imageops::resize(&img, rw as u32, rh as u32, image::imageops::FilterType::Lanczos3).into_raw())
        })
        .collect::<Result<_, String>>()?;
    let enc = LtxVaeEncoder::build(vcfg, vweights, n as u32, rh as u32, rw as u32, device);
    let latent = enc.encode(&rgb8_to_chw(&resized, rh, rw));
    let (lh, lw) = (rh / 32, rw / 32);
    Ok((chw_to_tc(&latent, channels, lat_f, lh, lw), (lat_f, lh, lw)))
}

/// The per-token attention weights of a reference whose latent is
/// `(lat_f, lat_h, lat_w)`, or `None` when it attends at full strength
/// (`append_ic_lora_reference_video_conditionings`): the mask video's
/// grayscale - its first `frames` frames, resized to half of `output`
/// (`_load_mask_video`) and area-averaged onto the latent grid - times the
/// strength, or the strength alone when it is below 1 and there is no mask.
fn reference_cross_mask(r: &VideoReference, frames: usize, output: (usize, usize), (lat_f, lat_h, lat_w): (usize, usize, usize)) -> Result<Option<Vec<f32>>, String> {
    let strength = r.attention_strength;
    if !(0.0..=1.0).contains(&strength) {
        return Err(format!("the reference's attention strength {strength} is outside [0, 1]"));
    }
    let Some(mask) = &r.attention_mask else {
        return Ok((strength < 1.0).then(|| vec![strength; lat_f * lat_h * lat_w]));
    };
    if mask.frames.len() < frames {
        return Err(format!("the attention mask video has {} frames, fewer than the {frames} being generated", mask.frames.len()));
    }
    if lat_f > 1 && !(frames - 1).is_multiple_of(lat_f - 1) {
        return Err(format!("{frames} mask frames do not average onto {lat_f} reference latent frames ((frames - 1) must divide by {})", lat_f - 1));
    }
    let (w, h) = (output.0 / 2, output.1 / 2);
    let mut gray = Vec::with_capacity(frames * h * w);
    for (i, frame) in mask.frames[..frames].iter().enumerate() {
        let img = image::RgbImage::from_raw(mask.width, mask.height, frame.clone()).ok_or_else(|| format!("attention mask frame {i} is not {}x{} RGB8", mask.width, mask.height))?;
        let px = image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3).into_raw();
        gray.extend(px.as_chunks::<3>().0.iter().map(|[r, g, b]| ((*r as f32 + *g as f32 + *b as f32) / (3.0 * 255.0)).clamp(0.0, 1.0)));
    }
    let latent = crate::refcond::downsample_mask_to_latent(&gray, frames, h, w, lat_f, lat_h, lat_w);
    Ok(Some(latent.into_iter().map(|v| v * strength).collect()))
}

/// Everything one denoising stage needs that does NOT vary between the two
/// stages of a two-stage generation: the model, the VAE weights the
/// conditioning encode needs, the text context, and the clip's frame count.
struct StageCtx<'a> {
    dit: &'a dyn Denoiser,
    /// The AUDIO stream's own text projection, `Some` exactly when this
    /// generation is audio-visual (see [`TextContext`]'s doc for why it is a
    /// separate projection rather than the video one reused).
    a_ctx_cond: &'a [f32],
    a_ctx_uncond: &'a [f32],
    vcfg: &'a LtxVaeConfig,
    vweights: &'a vae::blocks::Tensors,
    o: &'a GenOpts,
    lat_t: usize,
    in_channels: usize,
    ctx_cond: Caption<'a>,
    ctx_uncond: Caption<'a>,
    context_len: usize,
    cancel: &'a capability::CancelToken,
    /// The IC-LoRA reference video a stage run with this context appends, with
    /// the adapters' `(downscale, temporal_scale)` factors. `None` for every
    /// generation without one, and for a stage that runs out of context (the
    /// refinement of an IC-LoRA run, by default).
    reference: Option<StageReference<'a>>,
}

/// A stage's IC-LoRA reference (`VideoConditionByReferenceLatent`).
#[derive(Clone, Copy)]
enum StageReference<'a> {
    /// A reference video, encoded at the stage's own resolution, with the
    /// `(downscale, temporal_scale)` factors its adapter declares.
    Video(&'a VideoReference, (usize, usize)),
    /// An already-encoded `[C, f, h, w]` latent appended at strength 1 - the
    /// DFR detailing pass's half-resolution stage-1 video, its adapter's
    /// `downscale` apart from the stage's own grid.
    Latent { chw: &'a [f32], dims: (usize, usize, usize), downscale: usize },
}

/// Clean latent frames carried in from the PREVIOUS long-form window,
/// occupying the first [`Self::frames`] latent frames of a stage's own token
/// sequence and held at sigma 0 for the whole trajectory.
///
/// This is `--start-frame`'s `VideoConditionByLatentIndex(latent_idx=0)`
/// mechanism with the frame count unpinned - the reference's own class
/// asserts only `(B, C, H, W)` on its latent and writes `clean_latent[:,
/// start:stop]` / `denoise_mask[:, start:stop]` over whatever range it
/// covers, and its trainer spells the multi-frame case out as prefix
/// conditioning (`crate::longform`'s module doc carries the citation). What
/// makes it a CONTINUATION rather than an image conditioning is only where
/// the content comes from: [`crate::longform::carry_tail`]'s slice of the
/// previous window's own final latent, never a decoded-and-re-encoded frame.
struct LatentContext<'a> {
    /// `[C, frames, lh, lw]` at THIS stage's own resolution - a two-stage
    /// window's half-resolution stage 1 and its full-resolution stage 2 each
    /// get the previous window's latent at the matching size, not one
    /// rescaled copy.
    chw: &'a [f32],
    frames: usize,
}

/// A SOURCE clip's latent plus the per-token mask that says which of its
/// tokens this stage carries verbatim and which it regenerates.
///
/// `ltx_core.conditioning.types.mask_cond.VideoConditionByMask`, ported and
/// parity-gated in [`crate::maskcond`]. Unlike [`LatentContext`] (a contiguous
/// prefix of latent FRAMES) and unlike image conditioning (whole latent frames,
/// appended or overwritten), this pins an arbitrary per-token region that may
/// differ on every latent frame - which is what a moving subject needs.
struct MaskedSource<'a> {
    /// `[C, lat_t, lh, lw]` - the source clip VAE-encoded at this stage's
    /// resolution.
    chw: &'a [f32],
    /// `[lat_t * lh * lw]` in the reference's own polarity: `1` conditions
    /// (carried), `0` generates. See [`crate::maskcond`]'s module doc - it is
    /// inverted from every human-facing mask convention in this repo, and
    /// [`crate::maskcond::MaskSeqPolarity::to_conditioning`] is the one place
    /// the flip happens.
    mask: &'a [f32],
    /// `VideoConditionByMask::strength`. `1.0` freezes the conditioned tokens
    /// at the source's own latent for the whole trajectory.
    strength: f32,
}

/// One stage's own inputs: the resolution it runs at, its schedule, its
/// sampler, and - for a refinement stage - the latent it starts from.
struct Stage<'a> {
    width: usize,
    height: usize,
    sigmas: &'a [f64],
    /// A long-form continuation window's carried latent prefix; `None` for
    /// every window that starts from nothing, which is every stage this
    /// pipeline ran before long-form generation existed.
    context: Option<LatentContext<'a>>,
    /// The [`GenOpts::sampler`]'s eta for THIS stage. The reference's
    /// distilled stage 2 is deterministic whatever stage 1 used: "Stage 2 is
    /// always deterministic -- its 3-step refinement schedule is too short to
    /// remove freshly injected noise" (`ltx_pipelines.distilled`).
    eta: f64,
    /// `[C, lat_t, lh, lw]` content this stage starts from, partially
    /// re-noised to `sigmas[0]`; `None` draws pure noise, which is what a
    /// first stage does.
    seed_chw: Option<&'a [f32]>,
    /// Folded into the noise seed so two stages of one generation never draw
    /// the same numbers.
    seed_salt: u64,
    /// Steps already reported to `progress` before this stage.
    done_before: u32,
    label: &'static str,
    /// Per-token masked conditioning over a source clip
    /// ([`MaskedSource`]); `None` for every stage that generates from
    /// nothing. Mutually exclusive with [`Self::context`] and with image
    /// conditioning - all three write the same two tensors, and
    /// [`denoise_stage`] refuses the combination rather than letting the last
    /// one silently win.
    mask: Option<MaskedSource<'a>>,
    /// The audio stream's half of this stage; `None` on a video-only
    /// generation.
    audio: Option<AudioStage>,
    /// Generated keyframe slots denoised beside the clip (DFR); `None` for
    /// every other stage.
    slots: Option<KeyframeSlots<'a>>,
}

/// `VideoGeneratedKeyframeSlots`: one latent frame's worth of tokens per
/// `pixel_frames` entry, appended FULLY denoised (`denoise_mask = 1`) and
/// marked in `keyframes_mask`, at a RoPE time span of exactly one pixel
/// frame - the model generates a single sharp frame there and conditions the
/// clip around it on it.
#[derive(Clone, Copy)]
struct KeyframeSlots<'a> {
    /// Strictly increasing pixel frames on the stage's own canvas.
    pixel_frames: &'a [usize],
    /// `[C, K, lh, lw]` content the slots start from (a refinement stage's
    /// upscaled keyframes), re-noised to the stage's first sigma; `None`
    /// starts them from noise.
    initial_chw: Option<&'a [f32]>,
}

/// The base state [`append_keyframe_slots`] extends.
struct KeyframeSlotsInput<'a> {
    base_t: usize,
    base_positions: &'a [f32],
    base_keyframes_mask: &'a [f32],
    lh: usize,
    lw: usize,
    channels: usize,
    fps: f64,
}

/// What [`append_keyframe_slots`] produces: the whole state's positions and
/// keyframes mask, and the appended tokens' starting latent.
struct AppendedSlots {
    total_t: usize,
    positions: Vec<f32>,
    keyframes_mask: Vec<f32>,
    /// `[K * lh * lw, channels]` - `lerp(initial, noise, sigma0)`, the
    /// `GaussianNoiser` start of a `denoise_mask = 1` token.
    latent: Vec<f32>,
}

/// [`KeyframeSlots`] appended to a `[3, base_t, 2]` state - positions from
/// [`keyframe_conditioning_positions`] (`_slot_positions`: the spatial grid,
/// time `[p, p + 1) / fps`), keyframes mask `1` (marked), start latent the
/// initial content lerped toward noise by `sigma0`.
fn append_keyframe_slots(b: KeyframeSlotsInput<'_>, slots: &KeyframeSlots<'_>, sigma0: f32, seed: u64) -> Result<AppendedSlots, String> {
    let k = slots.pixel_frames.len();
    if k == 0 || slots.pixel_frames.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!("keyframe slots need strictly increasing pixel frames, got {:?}", slots.pixel_frames));
    }
    let block_t = b.lh * b.lw;
    let total_t = b.base_t + k * block_t;
    let mut positions = vec![0f32; 3 * total_t * 2];
    for axis in 0..3 {
        positions[axis * total_t * 2..axis * total_t * 2 + b.base_t * 2].copy_from_slice(&b.base_positions[axis * b.base_t * 2..(axis + 1) * b.base_t * 2]);
    }
    for (si, &p) in slots.pixel_frames.iter().enumerate() {
        let sp = keyframe_conditioning_positions(p, b.lh, b.lw, b.fps);
        for axis in 0..3 {
            let dst = axis * total_t * 2 + (b.base_t + si * block_t) * 2;
            positions[dst..dst + block_t * 2].copy_from_slice(&sp[axis * block_t * 2..(axis + 1) * block_t * 2]);
        }
    }
    let mut keyframes_mask = b.base_keyframes_mask.to_vec();
    keyframes_mask.resize(total_t, 1.0);
    let initial = match slots.initial_chw {
        Some(chw) if chw.len() != b.channels * k * block_t => return Err(format!("{} initial keyframe values for {k} slots of {block_t} tokens x {} channels", chw.len(), b.channels)),
        Some(chw) => chw_to_tc(chw, b.channels, k, b.lh, b.lw),
        None => vec![0f32; k * block_t * b.channels],
    };
    let noise = seeded_noise(initial.len(), seed);
    let latent = initial.iter().zip(&noise).map(|(&x, &n)| (1.0 - sigma0) * x + sigma0 * n).collect();
    Ok(AppendedSlots { total_t, positions, keyframes_mask, latent })
}

/// The audio stream's own half of one stage's inputs - the counterpart of
/// [`Stage::seed_chw`] plus [`Stage::context`], which the audio stream needs
/// in exactly the same two roles and at only one resolution.
struct AudioStage {
    /// `[ta, crate::audio::TOKEN_DIM]` the stage starts from.
    ///
    /// The FIRST stage of a window passes freshly drawn noise and a
    /// refinement stage passes the previous stage's own denoised audio
    /// latent, re-noised to this stage's starting sigma - which is what
    /// `ltx_pipelines.distilled` does (`ModalitySpec(context=audio_context,
    /// noise_scale=stage_2_sigmas[0], initial_latent=audio_state.latent)`).
    /// Audio is NOT spatially upscaled between stages: it has no spatial
    /// axes, so a refinement stage refines the same token count it was
    /// handed.
    latent: Vec<f32>,
    /// The previous WINDOW's own final audio tokens, written over the head of
    /// [`Self::latent`] and held at sigma 0 for the whole trajectory - the
    /// audio counterpart of [`LatentContext`], on the stream whose tokens are
    /// a different length of time (see `crate::audio`'s module doc).
    ///
    /// Empty for every stage of a window that carries nothing, which is every
    /// stage this pipeline ran before a clip could be longer than one window.
    /// It is applied AFTER the re-noise above, exactly as the video half
    /// writes its own carried prefix over the upscaled stage-1 copy of the
    /// same frames: what a continuation window holds fixed is the previous
    /// window's own output, never a re-noised or round-tripped copy of it.
    context: Vec<f32>,
    /// A per-token `(denoise_mask, clean)` over the whole stream instead of
    /// a carried prefix: a retake holds the source soundtrack fixed outside
    /// the regenerated window (`TemporalRegionMask`). `None` for every stage
    /// that carries at most a prefix.
    frozen: Option<(Vec<f32>, Vec<f32>)>,
}

/// Run one denoising stage at one latent resolution and return its final
/// latent as `[C, lat_t, lh, lw]`.
///
/// This is [`generate`]'s original denoise body, made resolution-parametric
/// and given a seed input, so the same code serves a single-stage run and
/// both stages of a two-stage one. Nothing about WHAT it does changed.
fn denoise_stage(sc: &StageCtx<'_>, mut st: Stage<'_>, total: u32, progress: &mut impl FnMut(u32, u32, &str)) -> Result<StageOut, String> {
    let o = sc.o;
    let (lh, lw) = (st.height / 32, st.width / 32);
    let t = sc.lat_t * lh * lw;
    let c = sc.in_channels;
    // `real_pixel_positions`, not `grid_positions` - the real production
    // pipeline's own `VideoLatentTools.create_initial_state` builds RoPE
    // positions in pixel-scale units (`get_pixel_coords`, causal-fixed,
    // divided by fps), not raw latent-grid integers. See that function's
    // own doc for why this was never caught by an earlier cosine-similarity
    // check on either side. They are rebuilt per stage because the SPATIAL
    // axes are pixel-scaled, so a half-resolution stage has its own grid;
    // the frame axis is identical in both.
    let positions = real_pixel_positions(sc.lat_t, lh, lw, o.fps as f64);
    // The causal VAE's first latent frame covers exactly ONE pixel frame
    // (every later one covers `VAE_TEMPORAL_SCALE`), making it "the same
    // token class as a generated keyframe slot" - `ltx_core.tools.
    // VideoLatentTools._first_frame_keyframes_mask`'s own doc: marked
    // UNCONDITIONALLY, independent of whether any real image conditioning
    // is present. The mask marks a TOKEN CLASS
    // (first-latent-frame-is-narrower), not "this token is externally
    // conditioned", and `dit_cfg.use_keyframes_abs_pos_embedding` is `true`
    // for the real checkpoint, so leaving it all-zero would silently omit a
    // real positional-embedding addition on every generation.
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[..lh * lw].fill(1.0);

    // One draw over the WHOLE post-conditioning sequence, matching
    // `GaussianNoiser._sample_noise`, which runs after every conditioning
    // item has appended its tokens. The mid anchor adds no block: it pins
    // into the base grid instead of appending (see [`conditioned_latent`]).
    let blocks = conditioning_block_count(o.start_frame.is_some(), o.mid_frame.is_some(), o.end_frame.is_some(), o.future_anchors.len());
    let mut latent0 = seeded_noise((t + blocks * lh * lw) * c, o.seed ^ st.seed_salt);
    if let Some(seed_chw) = st.seed_chw {
        // `GaussianNoiser.__call__`'s partial re-noise, `lerp(latent, noise,
        // sigma0)` - upstream's `ModalitySpec::noise_scale`, set to
        // `stage_2_sigmas[0]` - over the whole sequence. The base video's
        // latent is the seed; an appended conditioning block's is the item's
        // zero placeholder, so its noise is scaled by `sigma0` too before its
        // clean content is mixed in below at `1 - strength`.
        let seed_tc = chw_to_tc(seed_chw, c, sc.lat_t, lh, lw);
        assert_eq!(seed_tc.len(), t * c, "stage seed is {} values, expected {}", seed_tc.len(), t * c);
        let s0 = st.sigmas[0] as f32;
        for (dst, &s) in latent0[..t * c].iter_mut().zip(&seed_tc) {
            *dst = (1.0 - s0) * s + s0 * *dst;
        }
        latent0[t * c..].iter_mut().for_each(|n| *n *= s0);
    }

    let (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen) = if let Some(ms) = &st.mask {
        // `VideoConditionByMask` applied to the state `create_initial_state`
        // builds: base clean all zeros, base denoise mask all ones. Both
        // terms of both blends are still evaluated - `crate::maskcond` is the
        // reference's algebra, not a specialisation of it, so this stays
        // correct if a caller ever hands it a state something else conditioned
        // first.
        if o.start_frame.is_some() || o.mid_frame.is_some() || o.end_frame.is_some() || !o.future_anchors.is_empty() || st.context.is_some() {
            return Err("a masked video-to-video pass was given a conditioning still or a carried latent context as well: masked conditioning already writes every token's clean latent and denoise mask, so the two cannot both apply".into());
        }
        assert_eq!(ms.mask.len(), t, "masked stage: mask has {} tokens, expected {t}", ms.mask.len());
        let cond_tc = chw_to_tc(ms.chw, c, sc.lat_t, lh, lw);
        let mc = crate::maskcond::apply_video_condition_by_mask(&vec![0f32; t * c], &vec![1f32; t], &cond_tc, ms.mask, c, ms.strength);
        // `GaussianNoiser`, which the reference runs after every conditioning
        // item. `latent0` IS this stage's noise draw, so it stands in for both
        // the state's own latent and the fresh draw at `noise_scale = 1.0`,
        // where the noiser's first lerp is the identity.
        let latent = crate::maskcond::noised_initial_latent(&latent0, &latent0, &mc.clean, &mc.denoise_mask, c, 1.0);
        let carried = ms.mask.iter().filter(|&&m| m >= 1.0).count();
        tracing::info!(stage = st.label, tokens = t, carried, strength = ms.strength, "masked conditioning applied");
        (latent, positions, keyframes_mask, t, Some((mc.denoise_mask, mc.clean)))
    } else {
        // Merged context + image branch: context pins a prefix over
        // an already-built conditioning state.  The two write provably
        // disjoint index ranges: context writes [0, ctx_tokens) and
        // image conditioning writes [base_t, base_t+blocks*lh*lw),
        // where ctx_tokens = ctx.frames * lh * lw and base_t = t.
        //
        // One refusal: context + lone --start-frame is a genuine
        // collision - start-frame's overwrite of latent frame 0 sits
        // inside the carried prefix.
        if st.context.is_some() && o.start_frame.is_some() && o.mid_frame.is_none() && o.end_frame.is_none() {
            return Err("a carried latent context and a lone --start-frame cannot compose: start-frame's overwrite of latent frame 0 sits inside the carried prefix".into());
        }
        let has_image = o.start_frame.is_some() || o.mid_frame.is_some() || o.end_frame.is_some() || !o.future_anchors.is_empty();
        let (mut latent, positions_d, keyframes_mask_d, denoise_t_count, mut frozen) = if has_image {
            // WHICH mechanism runs is [`conditioned_latent`]'s decision - the
            // reference has two conditioning builders for these two cases
            // (image-to-video's in-place overwrite of latent frame 0, keyframe
            // interpolation's appended guiding blocks). See that function's doc
            // for the reference citations and for where
            // `conditioning_strength` lands.
            //
            // The same path passed for both ends is encoded ONCE: a real VAE
            // encode is not free, and the loop case (one still at both ends) is
            // the common one.
            let enc = |p: &str| encode_still(sc.vcfg, sc.vweights, p, st.width, st.height, c, o.device.as_deref());
            let start_tokens = o.start_frame.as_deref().map(enc).transpose()?;
            let end_tokens = match (&o.end_frame, &o.start_frame) {
                (Some(e), Some(s)) if e == s => start_tokens.clone(),
                (Some(e), _) => Some(enc(e.as_str())?),
                (None, _) => None,
            };
            // Resolved from the clip's frame count, which is the same in both
            // stages of a two-stage run, so the anchor names the same instant at
            // half resolution and at full.
            let mid_at = o.mid_frame.as_deref().map(|_| mid_anchor_frame(o.frames, o.mid_frame_at)).transpose()?;
            let mid_tokens = o.mid_frame.as_deref().map(enc).transpose()?;
            let mid = mid_at.zip(mid_tokens.as_deref());
            if let Some((at, _)) = mid {
                tracing::info!(stage = st.label, mid_frame = at, latent_frame = latent_frame_containing(at), of_frames = o.frames, "mid-frame anchor pinned into its latent slot");
            }
            // The mid pin writes INSIDE the base range, at the pinned slot's
            // token range, while a carried context overwrites [0, ctx_tokens)
            // of that same range - so the pin must sit beyond the carried
            // prefix. Window routing only ever hands this pass a mid owned by
            // its EMITTED span, which starts a whole latent frame past the
            // context; the refusal keeps that invariant loud instead of
            // silently letting the context overwrite the pin.
            if let (Some(ctx), Some((at, _))) = (&st.context, mid) {
                let slot = latent_frame_containing(at);
                let ctx_tokens = ctx.frames * lh * lw;
                if slot * lh * lw < ctx_tokens {
                    return Err(format!("the mid anchor pinned at window pixel frame {at} lands in latent frame {slot}, inside the carried context's {} latent frames: window routing must own the mid instant in this window's emitted span", ctx.frames));
                }
            }
            // The long-form windows' future anchors: appended guides at the raw
            // local pixel positions they name, which may lie beyond this
            // window's own span (see [`GenOpts::future_anchors`]).
            let mut future_tokens = Vec::with_capacity(o.future_anchors.len());
            for (_, path) in &o.future_anchors {
                future_tokens.push(enc(path)?);
            }
            let future: Vec<(usize, &[f32])> = o.future_anchors.iter().zip(&future_tokens).map(|((at, _), tokens)| (*at, tokens.as_slice())).collect();
            if !future.is_empty() {
                tracing::info!(stage = st.label, future_anchors = ?future.iter().map(|(at, _)| *at).collect::<Vec<_>>(), of_frames = o.frames, "future anchors appended as guiding blocks");
            }
            let cl = conditioned_latent(latent0, &positions, &keyframes_mask, t, lh, lw, c, o.frames, o.fps as f64, start_tokens.as_deref(), mid, end_tokens.as_deref(), &future, o.conditioning_strength);
            tracing::info!(stage = st.label, strength = o.conditioning_strength, tokens = cl.t, base_tokens = t, appended_blocks = blocks, "image conditioning applied");
            (cl.latent, cl.positions, cl.keyframes_mask, cl.t, Some((cl.denoise_mask, cl.clean)))
        } else {
            (latent0, positions, keyframes_mask, t, None)
        };
        // Overlay the carried latent context prefix, if present.
        if let Some(ctx) = &st.context {
            let ctx_tokens = ctx.frames * lh * lw;
            if ctx.frames > sc.lat_t {
                return Err(format!("a {}-latent-frame context does not fit a {}-latent-frame window", ctx.frames, sc.lat_t));
            }
            let ctx_tc = chw_to_tc(ctx.chw, c, ctx.frames, lh, lw);
            latent[..ctx_tokens * c].copy_from_slice(&ctx_tc);
            let (ref mut denoise_mask, ref mut clean) = frozen.get_or_insert_with(|| {
                (vec![1f32; t], vec![0f32; t * c])
            });
            clean[..ctx_tokens * c].copy_from_slice(&ctx_tc);
            denoise_mask[..ctx_tokens].fill(0.0);
            tracing::info!(stage = st.label, context_latent_frames = ctx.frames, context_tokens = ctx_tokens, tokens = t, "latent context frozen");
        }
        (latent, positions_d, keyframes_mask_d, denoise_t_count, frozen)
    };
    // Generated keyframe slots, after the images and before the reference -
    // the order DFR's stages list their conditionings in.
    let slot_start = denoise_t_count;
    let (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen) = match &st.slots {
        None => (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen),
        Some(slots) => {
            if st.mask.is_some() {
                return Err("a masked video-to-video pass cannot also generate keyframe slots".into());
            }
            let sigma0 = if st.seed_chw.is_some() { st.sigmas[0] as f32 } else { 1.0 };
            let base = KeyframeSlotsInput { base_t: denoise_t_count, base_positions: &positions_d, base_keyframes_mask: &keyframes_mask_d, lh, lw, channels: c, fps: o.fps as f64 };
            let ks = append_keyframe_slots(base, slots, sigma0, o.seed ^ st.seed_salt ^ SLOT_SEED_SALT)?;
            let mut latent = latent0;
            latent.extend_from_slice(&ks.latent);
            let (mut mask, mut clean) = frozen.unwrap_or_else(|| (vec![1f32; denoise_t_count], vec![0f32; denoise_t_count * c]));
            mask.resize(ks.total_t, 1.0);
            clean.resize(ks.total_t * c, 0.0);
            tracing::info!(stage = st.label, slots = slots.pixel_frames.len(), slot_tokens = ks.total_t - denoise_t_count, seeded = slots.initial_chw.is_some(), sigma0, "generated keyframe slots appended");
            (latent, ks.positions, ks.keyframes_mask, ks.total_t, Some((mask, clean)))
        }
    };
    let slot_end = denoise_t_count;
    // The IC-LoRA reference, appended LAST - after every other conditioning
    // item, as `ICLoraPipeline` orders them - at this stage's own resolution.
    let mut reference_attention = None;
    let (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen) = match sc.reference {
        None => (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen),
        Some(reference) => {
            if st.mask.is_some() {
                return Err("a masked video-to-video pass cannot also take an IC-LoRA reference video".into());
            }
            let (tokens, dims, downscale, temporal, strength, cross_mask) = match reference {
                StageReference::Video(reference, (downscale, temporal)) => {
                    let (tokens, dims) = encode_reference(sc.vcfg, sc.vweights, reference, o.frames, st.width, st.height, downscale, temporal, c, o.device.as_deref())?;
                    let cross_mask = reference_cross_mask(reference, o.frames, (o.width, o.height), dims)?;
                    (tokens, dims, downscale, temporal, reference.strength, cross_mask)
                }
                StageReference::Latent { chw, dims: (f, h, w), downscale } => (chw_to_tc(chw, c, f, h, w), (f, h, w), downscale, 1, 1.0, None),
            };
            let rc = crate::refcond::append_reference_video_conditioning(denoise_t_count, &positions_d, &keyframes_mask_d, c, o.fps as f64, dims, &tokens, downscale, temporal, strength, cross_mask.as_deref());
            reference_attention = rc.cross_mask.as_deref().map(|cm| crate::refcond::reference_attention(t, denoise_t_count, cm));
            let total_t = rc.keyframes_mask.len();
            // `GaussianNoiser`: the appended tokens start at
            // `lerp(clean, noise, denoise_mask)`, which is exactly the clean
            // reference at strength 1.
            let noise = seeded_noise((total_t - denoise_t_count) * c, o.seed ^ st.seed_salt ^ REFERENCE_SEED_SALT);
            let mut latent = latent0;
            for (i, n) in noise.iter().enumerate() {
                let (tok, m) = (denoise_t_count + i / c, rc.denoise_mask[denoise_t_count + i / c]);
                latent.push((1.0 - m) * rc.clean[tok * c + i % c] + m * n);
            }
            let (mut mask, mut clean) = frozen.unwrap_or_else(|| (vec![1f32; denoise_t_count], vec![0f32; denoise_t_count * c]));
            mask.extend_from_slice(&rc.denoise_mask[denoise_t_count..]);
            clean.extend_from_slice(&rc.clean[denoise_t_count * c..]);
            tracing::info!(stage = st.label, reference_tokens = total_t - denoise_t_count, dims = ?dims, downscale, temporal, strength, weighted_attention = reference_attention.is_some(), "IC-LoRA reference appended");
            (latent, rc.positions, rc.keyframes_mask, total_t, Some((mask, clean)))
        }
    };
    let frozen_ref = frozen.as_ref().map(|(mask, clean)| Frozen { mask, clean, channels: c, attention: reference_attention.as_ref() });
    tracing::info!(stage = st.label, width = st.width, height = st.height, tokens = denoise_t_count, base_tokens = t, steps = st.sigmas.len() - 1, eta = st.eta, seeded = st.seed_chw.is_some(), "stage denoising");
    let done_before = st.done_before;
    let dim = crate::audio::TOKEN_DIM as usize;
    let mut audio_state = st.audio.take().map(|a| {
        let AudioStage { mut latent, context, frozen: given } = a;
        let ta = latent.len() / dim;
        // A REFINEMENT stage is handed the previous stage's own DENOISED
        // audio latent and has to re-noise it to this stage's starting sigma
        // - the same `lerp(clean, noise, sigma0)` the video half two blocks
        // up does with its upscaled seed, and the reference's own
        // `ModalitySpec(noise_scale=stage_2_sigmas[0],
        // initial_latent=audio_state.latent)`. Handing a clean latent to a
        // loop that believes it is at sigma 0.909 is not a small error: every
        // step's `to_denoised` would be scaled for noise that is not there.
        //
        // A FIRST stage is handed noise already, and `seed_chw` is `None`
        // there, so this does not run.
        if st.seed_chw.is_some() {
            let s0 = st.sigmas[0] as f32;
            let mut rng = data::rng::Rng::new(o.seed ^ AUDIO_SEED_SALT ^ st.seed_salt);
            for v in &mut latent {
                let n = rng.next_gaussian() as f32;
                *v = (1.0 - s0) * *v + s0 * n;
            }
            tracing::info!(stage = st.label, sigma0 = s0, audio_tokens = ta, "audio latent re-noised for a refinement stage");
        }
        // The carried prefix goes on LAST, over whatever the re-noise above
        // left there: a continuation window is pinned to the previous
        // window's own output, and a re-noised copy of it is a different
        // tensor.
        latent[..context.len()].copy_from_slice(&context);
        if let Some((mask, clean)) = &given {
            for (i, v) in latent.iter_mut().enumerate() {
                let m = mask[i / dim];
                *v = m * *v + (1.0 - m) * clean[i];
            }
        }
        let frozen = given.or_else(|| (!context.is_empty()).then(|| {
            let mut mask = vec![1f32; ta];
            mask[..context.len() / dim].fill(0.0);
            let mut clean = vec![0f32; ta * dim];
            clean[..context.len()].copy_from_slice(&context);
            (mask, clean)
        }));
        AudioState { latent, positions: crate::audio::positions(ta), ta, ctx_cond: sc.a_ctx_cond, ctx_uncond: sc.a_ctx_uncond, frozen }
    });
    if let Some(a) = &audio_state {
        tracing::info!(stage = st.label, audio_tokens = a.ta, carried_tokens = a.frozen.as_ref().map_or(0, |(m, _)| m.iter().filter(|&&m| m == 0.0).count()), "audio stream denoising jointly with the video stream");
    }
    let final_latent = denoise(
        sc.dit,
        st.sigmas,
        latent0,
        &positions_d,
        &keyframes_mask_d,
        sc.ctx_cond,
        sc.ctx_uncond,
        sc.context_len,
        denoise_t_count,
        &o.guidance,
        Sampling { sampler: o.sampler, eta: st.eta, s_noise: o.s_noise, seed: o.seed ^ 0x4e_4f_49_53_45 ^ st.seed_salt },
        total,
        frozen_ref.as_ref(),
        sc.cancel,
        audio_state.as_mut(),
        &mut |done, tot, phase| progress(done_before + done, tot, phase),
    )?;
    let slots_chw = st.slots.as_ref().map(|s| tc_to_chw(&final_latent[slot_start * c..slot_end * c], c, s.pixel_frames.len(), lh, lw));
    Ok(StageOut { video_chw: tc_to_chw(&final_latent[..t * c], c, sc.lat_t, lh, lw), slots_chw, audio: audio_state.map(|a| a.latent) })
}

/// One stage's result: the video latent it produced and, on an audio-visual
/// generation, the audio latent the SAME forwards produced beside it.
struct StageOut {
    /// `[C, lat_t, lh, lw]`.
    video_chw: Vec<f32>,
    /// `[ta, crate::audio::TOKEN_DIM]`, `None` on a video-only generation.
    audio: Option<Vec<f32>>,
    /// The generated keyframe slots' final content, `[C, K, lh, lw]` -
    /// `None` for a stage without [`Stage::slots`].
    slots_chw: Option<Vec<f32>>,
}

/// One refinement pass: carry `latent_chw` up with the real spatial x2 latent
/// upscaler, then denoise it at the doubled resolution.
///
/// This is the tail of the reference's distilled two-stage generation
/// (`ltx_pipelines.distilled.DistilledPipeline.__call__`) and, because it is
/// the same operation whether the input latent came from a stage-1 denoise or
/// from VAE-encoding a clip that finished rendering an hour ago, it is also
/// the whole of [`upscale`]. Both reach it here rather than each carrying
/// their own copy - the un-normalize sandwich below is a defect this crate
/// has already shipped once (see [`crate::upsampler::upsample_video`]), and
/// one call site is one place for it to be right.
struct Refine<'a> {
    /// The run's shared spatial x2 upscaler cache, plus the path to fill it
    /// from - see [`SpatialUpsampler`] for why it is a cache and not either a
    /// bare path or already-imported weights.
    upsampler: &'a SpatialUpsampler,
    upsampler_path: &'a str,
    /// Generated keyframe slots carried up with the video (DFR's detailing
    /// pass): the previous stage's `[C, K, lh1, lw1]` keyframes, upscaled by
    /// the same upscaler as one K-frame latent and seeding the slots.
    slots: Option<(&'a [usize], &'a [f32])>,
    /// The previous stage's audio latent plus whatever this window carries
    /// from the one before it - `None` on a video-only generation. See
    /// [`AudioStage`].
    audio: Option<AudioStage>,
    /// `[C, lat_t, lh1, lw1]` - the latent to carry up.
    latent_chw: &'a [f32],
    lat_t: usize,
    lh1: usize,
    lw1: usize,
    /// The PIXEL resolution this refinement runs at, i.e. twice the latent
    /// grid `lh1`/`lw1` describes.
    width: usize,
    height: usize,
    /// [`stage2_sigmas`]'s output - a suffix of the distilled refinement
    /// table, never an interpolated schedule (see that function's doc).
    sigmas: &'a [f64],
    /// The refinement's sampler eta: [`refine_eta`] for every refinement
    /// but DFR's detailing pass, which samples as its first stage does.
    eta: f64,
    /// A long-form continuation window's carried latent prefix, at THIS
    /// stage's (full) resolution - it overwrites the upscaled stage-1 copy of
    /// the same frames, so what the refinement holds fixed is the previous
    /// window's own final latent rather than a round trip through the x2
    /// upscaler.
    context: Option<LatentContext<'a>>,
    /// Folded into this refinement's noise seed on top of its own `0x5332`,
    /// so two windows of one long-form generation never draw the same
    /// refinement noise.
    seed_salt: u64,
    done_before: u32,
    label: &'static str,
}

/// A refinement stage's eta. Under Euler, `0`: upstream's distilled
/// refinement stage is always the DETERMINISTIC sampler, "its 3-step
/// refinement schedule is too short to remove freshly injected noise".
/// res_2s refines with its own SDE noise, as `ti2vid_two_stages_hq`'s stage
/// 2 does.
fn refine_eta(o: &GenOpts) -> f64 {
    if o.sampler == Sampler::Res2s {
        o.eta
    } else {
        0.0
    }
}

fn upscale_and_refine(sc: &StageCtx<'_>, mut r: Refine<'_>, total: u32, progress: &mut impl FnMut(u32, u32, &str)) -> Result<StageOut, String> {
    let (lh, lw) = (r.height / 32, r.width / 32);
    progress(r.done_before, total, "spatial upscale");
    tracing::info!(latent_h = r.lh1, latent_w = r.lw1, "real x2 latent upscale");
    let scfg = LatentUpsamplerConfig::spatial_x2();
    let ups = LatentUpsampler::build(&scfg, r.upsampler.get(r.upsampler_path)?, r.lat_t as u32, r.lh1 as u32, r.lw1 as u32, sc.o.device.as_deref());
    // Through `upsample_video`, NOT `upsample` directly: the upscaler was
    // trained on raw VAE latents and needs the per-channel un-normalize/
    // re-normalize sandwich around it. Skipping it costs half the latent's
    // variance and decodes to a blurred clip - see that function's own doc
    // for the measurement.
    let (pc_mean, pc_std) = crate::vae3d::per_channel_statistics(sc.vweights);
    let upscaled = crate::upsampler::upsample_video(&ups, &pc_mean, &pc_std, r.latent_chw);
    let (_, _, up_lh, up_lw) = ups.out_shape();
    drop(ups);
    // `upsampler(stacked_keyframes)`: the planes go through the upscaler as
    // one K-frame latent, as the reference stacks them.
    let upscaled_keyframes = match r.slots {
        None => None,
        Some((frames, keyframes)) => {
            let kups = LatentUpsampler::build(&scfg, r.upsampler.get(r.upsampler_path)?, frames.len() as u32, r.lh1 as u32, r.lw1 as u32, sc.o.device.as_deref());
            Some(crate::upsampler::upsample_video(&kups, &pc_mean, &pc_std, keyframes))
        }
    };
    if (up_lh as usize, up_lw as usize) != (lh, lw) {
        tracing::error!(got_h = up_lh, got_w = up_lw, want_h = lh, want_w = lw, "spatial upscaler produced the wrong latent grid");
        return Err(format!("spatial upscaler produced a {up_lh}x{up_lw} latent grid, expected {lh}x{lw} for {}x{}", r.width, r.height));
    }

    denoise_stage(
        sc,
        Stage {
            mask: None,
            width: r.width,
            height: r.height,
            sigmas: r.sigmas,
            eta: r.eta,
            seed_chw: Some(&upscaled),
            context: r.context.as_ref().map(|c| LatentContext { chw: c.chw, frames: c.frames }),
            // "S2" - the refinement must not draw the same noise as whatever
            // produced the latent it starts from.
            seed_salt: 0x5332 ^ r.seed_salt,
            done_before: r.done_before,
            label: r.label,
            // The audio latent is carried across the stage boundary as-is and
            // re-noised inside the loop by the schedule's own first sigma -
            // never spatially upscaled, because it has no spatial axes. This
            // mirrors the reference's own stage-2 audio spec. A long-form
            // window's carried prefix then goes over the head of it, exactly
            // as the video half's `context` does.
            audio: r.audio.take(),
            slots: r.slots.zip(upscaled_keyframes.as_deref()).map(|((pixel_frames, _), initial)| KeyframeSlots { pixel_frames, initial_chw: Some(initial) }),
        },
        total,
        progress,
    )
}

/// The spatial x2 latent upscaler's weights, read at most ONCE per run and
/// not until a refinement actually needs them.
///
/// Both halves matter and they pull in opposite directions. **Once**: every
/// refinement in a run uses the same weights - only
/// [`LatentUpsampler::build`] is sized by the window it runs on - so
/// re-reading and re-dequantizing a gigabyte of checkpoint at each seam is
/// pure per-window cost, and a long-form clip pays it per window while an
/// [`upscale`] pays it per pass. **Not until needed**: a generation already
/// holds a host cache of the whole transformer's block weights while it
/// denoises, so the upscaler's own expansion must not sit beside it through a
/// stage that does not use it.
#[derive(Default)]
struct SpatialUpsampler {
    weights: std::cell::OnceCell<vae::blocks::Tensors>,
}

impl SpatialUpsampler {
    /// The imported weights, reading them on the first call and returning the
    /// same map on every later one.
    fn get(&self, path: &str) -> Result<&vae::blocks::Tensors, String> {
        if self.weights.get().is_none() {
            let t0 = Instant::now();
            let w = crate::import::import_upsampler(read_any(path)?, &LatentUpsamplerConfig::spatial_x2())?;
            tracing::info!(secs = t0.elapsed().as_secs_f32(), tensors = w.len(), "spatial x2 latent upscaler loaded");
            let _ = self.weights.set(w);
        }
        Ok(self.weights.get().expect("just set"))
    }
}

/// The refinement schedule for `steps` steps: the LAST `steps + 1` entries of
/// `STAGE_2_DISTILLED_SIGMAS`.
///
/// A SUFFIX, not a resampling. The distilled checkpoint denoises correctly
/// only at the specific sigma values distillation baked in (see [`generate`]'s
/// own `sigmas` construction for the measurement that says so), so "fewer
/// refinement steps" can only mean "start further down the same table", never
/// "the same span in fewer, larger hops". Asking for the full
/// [`LTX2_STAGE2_STEPS`] reproduces upstream's own stage 2 exactly.
fn stage2_sigmas(steps: usize) -> Result<Vec<f64>, String> {
    if steps == 0 || steps > LTX2_STAGE2_STEPS {
        return Err(format!(
            "{steps} refinement steps is outside 1..={LTX2_STAGE2_STEPS} (the distilled refinement table has {} entries and this can only take a suffix of it)",
            LTX2_STAGE2_DISTILLED_SIGMAS.len()
        ));
    }
    Ok(LTX2_STAGE2_DISTILLED_SIGMAS[LTX2_STAGE2_DISTILLED_SIGMAS.len() - steps - 1..].iter().map(|&s| s as f64).collect())
}

/// Build the denoiser [`generate`] and [`upscale`] both run: the tiny
/// random-weight config, or the real 22B checkpoint streamed int8-compute
/// when `GenOpts::dit_config` names it (needs [`Paths::dit`]).
///
/// `adapters` is passed apart from `o` so a stage that runs out of context
/// (an IC-LoRA run's bare refinement) can build the same model without them.
fn build_denoiser(paths: &Paths, dit_cfg: LtxDitConfig, o: &GenOpts, adapters: &[model::adapter::AdapterSpec], place: crate::devplan::Placement) -> Result<Box<dyn Denoiser>, String> {
    let adapter = if adapters.is_empty() { None } else { Some(crate::lora::runtime_adapter(adapters)?) };
    if o.dit_config == "tiny" {
        // Not a real model: random weights, so any output is a wiring proof
        // and nothing else. Worth a warning rather than an info line - a run
        // that silently produced noise because a checkpoint path was unset is
        // the most expensive way to discover this.
        tracing::warn!("--dit-config tiny: building a RANDOM-weight DiT, output is a smoke test and carries no semantics");
        let weight_seed = o.seed ^ 0x4c_54_58_76_44_49_54; // "LTXvDIT" folded into the seed, so the same --seed reproduces the same weights
        let weights: Tensors = random_tiny_weights(&dit_cfg, weight_seed);
        let dit = LtxDit::new(dit_cfg, weights, o.device.as_deref());
        return Ok(Box::new(match adapter {
            Some(a) => dit.with_adapter(a)?,
            None => dit,
        }));
    }
    let dit_path = paths.dit.as_ref().ok_or_else(|| {
        tracing::error!(dit_config = %o.dit_config, "no real DiT checkpoint configured for a real dit-config");
        format!("ltxv dit-config {:?} needs a real checkpoint: put it in the model store or pass --dit <path>", o.dit_config)
    })?;
    tracing::info!(path = %dit_path, "opening the real DiT GGUF");
    let src = crate::gguf_src::LtxvGgufSource::open(dit_path).inspect_err(|e| tracing::error!(path = %dit_path, error = %e, "opening the DiT GGUF failed"))?;
    let real_cfg = src.config().video;
    if real_cfg != dit_cfg {
        tracing::error!(path = %dit_path, dit_config = %o.dit_config, "the checkpoint's embedded config does not match the named build config");
        return Err(format!("ltxv: {dit_path}'s own embedded config does not match LtxDitConfig::{:?}() - checkpoint/build mismatch", o.dit_config));
    }
    if o.audio {
        // The audio-visual denoiser is a DIFFERENT object, not the video one
        // with a flag: both streams are computed in ONE forward, because the
        // A<->V cross-attention couples them every block.
        let cfg = *src.config();
        // Keyed on the checkpoint, not on this call - `RealDit`'s own doc has
        // the reasoning, and it is the same store: an AV block and a video
        // block of the same checkpoint are different entries, so the two
        // arms share one cache without either seeing the other's bytes.
        let cache = crate::block::GenerationCache::for_checkpoint(dit_path);
        if o.runtime.av_fp32_reference {
            if o.reference.as_ref().is_some_and(|r| r.attention_strength < 1.0 || r.attention_mask.is_some()) {
                return Err("--av-fp32-reference does not weight a reference's attention: drop it, or the --reference-attention-* flags".into());
            }
            tracing::warn!(
                gib = crate::av_stream::host_floats(&cfg) * 4 / (1 << 30),
                "denoising on the eager host-fp32 audio-visual REFERENCE arm, expanding the whole checkpoint (drop --av-fp32-reference for the quantized, device-resident path)"
            );
            let mut w = crate::av_stream::AvWeights::load(&src, cfg)?;
            if let Some(a) = &adapter {
                fold_adapter(a, &mut w.tensors)?;
            }
            return Ok(Box::new(crate::av_stream::AvDenoiser::new(w, o.device.as_deref())));
        }
        let mut head = crate::dit::load_av_head_tensors_from_source(&src, &cfg);
        let adapter = adapter.map(|a| block_adapter(a, &src, &mut head)).transpose()?;
        let cs = cache.stats();
        tracing::info!(
            layers = cfg.video.num_layers,
            video_dim = cfg.video.inner_dim,
            audio_dim = cfg.audio.inner_dim,
            head_tensors = head.len(),
            cached_blocks = cs.blocks,
            cached_bytes = cs.bytes,
            "real audio-visual DiT ready (both streams, int8 blocks stream per forward)"
        );
        return Ok(Box::new(RealAvDit { cfg, src, head, device: o.device.clone(), cache, sessions: Default::default(), adapter, resident_blocks: o.runtime.resident_blocks }));
    }
    let mut head = crate::dit::load_head_tensors_from_source(&src, &real_cfg);
    let adapter = adapter.map(|a| block_adapter(a, &src, &mut head)).transpose()?;
    tracing::info!(layers = real_cfg.num_layers, inner_dim = real_cfg.inner_dim, head_tensors = head.len(), "real DiT ready (blocks stream per forward)");
    // Keyed on the checkpoint, not on this call: the same store is handed to
    // the next generation against the same file (see `RealDit`'s doc and
    // `crate::weightcache`).
    let cache = crate::block::GenerationCache::for_checkpoint(dit_path);
    let cs = cache.stats();
    tracing::info!(cached_blocks = cs.blocks, cached_bytes = cs.bytes, "checkpoint weight cache attached");
    Ok(Box::new(RealDit { stream: crate::config::Stream::Video, cfg: real_cfg, src, head, device: o.device.clone(), cache, place, sessions: Default::default(), adapter, resident_blocks: o.runtime.resident_blocks }))
}

/// Fold every correction of `adapter` into the fp32 `weights` - exact at
/// fp32, so this is how the eager reference arms apply one. Every target must
/// exist at the adapter's shape.
fn fold_adapter(adapter: &model::lora::RuntimeLora, weights: &mut Tensors) -> Result<(), String> {
    adapter.validate(&|k| weights.get(k).map(|(_, d)| d.len()))?;
    crate::lora::fold_held(adapter, weights);
    Ok(())
}

/// Split `adapter` for the real streamed checkpoint: head targets fold into
/// the fp32 `head` tensors now; block targets ride with the sessions and
/// attach beside each quantized block as it is built, and connector targets
/// fold in whenever a forward reads a connector. Every target is checked
/// against the checkpoint's own tensor sizes first, and a target that is
/// none of those is refused - an adapter that silently skipped one would
/// serve the base model.
fn block_adapter(adapter: model::lora::RuntimeLora, src: &crate::gguf_src::LtxvGgufSource, head: &mut Tensors) -> Result<SharedAdapter, String> {
    use checkpoint::TensorSource;
    adapter.validate(&|k| src.numel(k))?;
    if let Some(d) = adapter.deltas().iter().find(|d| !crate::dit::is_block_key(&d.key) && !crate::dit::is_connector_key(&d.key) && !head.contains_key(&d.key)) {
        return Err(format!("ltxv: the adapter targets {}, a model-level tensor this denoiser does not load", d.key));
    }
    crate::lora::fold_held(&adapter, head);
    Ok(std::sync::Arc::new(std::sync::Mutex::new(adapter)))
}

/// A runtime adapter shared by every card's session: claims are recorded per
/// block build, so it is locked only while a block attaches its corrections.
pub type SharedAdapter = std::sync::Arc<std::sync::Mutex<model::lora::RuntimeLora>>;

/// The text conditioning both [`generate`] and [`upscale`] denoise against:
/// real Gemma-4 when [`Paths::text_encoder`] is set ([`real_text_context`]),
/// otherwise the deterministic-but-meaningless stub every earlier milestone
/// used (see this module's doc on [`context_stub`]). `context_len` therefore
/// comes from the real tokenizer's own output length in the former case, not
/// [`GenOpts::context_len`] (which only ever sized the stub).
///
/// `text_encode_secs` is written rather than returned so a caller's
/// [`Timings`] field is filled in exactly when the real encoder ran.
fn build_context(paths: &Paths, prompt: &str, dit_cfg: LtxDitConfig, o: &GenOpts, place: crate::devplan::Placement, text_encode_secs: &mut f32) -> Result<TextContext, String> {
    let Some(te_path) = &paths.text_encoder else {
        // Same class of silent-nonsense as the tiny DiT: the prompt reaches
        // the model only as a hash-derived stub.
        tracing::warn!("no text encoder configured: the prompt is being replaced by a deterministic STUB context and carries no meaning");
        let prompt_mix = o.seed ^ fnv1a(prompt);
        let dim = dit_cfg.cross_attention_dim as usize;
        let n = o.context_len;
        // Padded exactly like [`real_text_context`]'s own real context (see
        // [`padded_context_len`]'s doc) - a no-op (`context_len == n`) for
        // every config whose connector is disabled, e.g. `LtxDitConfig::tiny`,
        // so this stays byte-identical to the pre-real-DiT behavior there.
        let context_len = padded_context_len(&dit_cfg, n);
        let stub = context_stub(n, dim, prompt_mix);
        let mut ctx_cond = vec![0f32; context_len * dim];
        ctx_cond[..stub.len()].copy_from_slice(&stub);
        // The "unconditional" branch has no real empty-prompt encoding
        // either; an all-zero context is the closest honest stand-in (most
        // text encoders map an empty string close to zero after their own
        // normalization) and, crucially, is DIFFERENT from `ctx_cond` - so
        // the CFG fold in `denoise` is exercised for real rather than folding
        // two identical branches.
        let ctx_uncond = vec![0f32; context_len * dim];
        // Real for the first `n` positions, invalid (register-substituted by
        // the connector when enabled) for the padded tail - all-valid when
        // `context_len == n` (connector disabled), unchanged from the
        // pre-padding behavior.
        let mut context_valid = vec![0f32; context_len];
        context_valid[..n].fill(1.0);
        // No audio projection on the stub path: the stub carries no semantic
        // content at all, so a fabricated "audio context" would be a second
        // meaningless vector rather than a stand-in for anything. A request
        // for sound is refused before it gets here (see `generate`).
        return Ok(TextContext { cond: ctx_cond, uncond: ctx_uncond, uncond_valid: context_valid.clone(), valid: context_valid, len: context_len, a_cond: None, a_uncond: None });
    };
    tracing::info!(path = %te_path, "encoding the prompt with the real text encoder");
    let te_t = std::time::Instant::now();
    // On the card the conditional DiT forward will NOT use, when the plan has
    // one to spare: the 12B encoder's own device footprint is then released
    // from a card that is not about to hold the denoise loop's activations.
    // See `crate::devplan`.
    let r = crate::devplan::on_gpu(place.text, || real_text_context(te_path, prompt, &dit_cfg, o.guidance.needs_uncond(o.audio).then_some(o.negative_prompt.as_str()), o.device.as_deref(), o.audio, &o.runtime))?
        .inspect(|c| tracing::info!(context_len = c.len, audio = c.a_cond.is_some(), "prompt encoded"))
        .inspect_err(|e| tracing::error!(path = %te_path, error = %e, "text encoding failed"))?;
    *text_encode_secs = te_t.elapsed().as_secs_f32();
    tracing::info!(secs = *text_encode_secs, "text encode done");
    Ok(r)
}

/// Everything an audio request needs, checked before any weight is read.
///
/// One function because [`generate`] and [`generate_long`] are two entry
/// points into the same requirement, and a check that lived in only one of
/// them would let the other produce an audio latent nothing can decode - or
/// fill a host weight cache first and fail after it.
fn check_audio_request(paths: &Paths, o: &GenOpts) -> Result<(), String> {
    check_audio_inputs(paths, o)?;
    if o.audio {
        check_av_host_memory(&LtxAvDitConfig::ltx25(), o.runtime.av_fp32_reference)?;
    }
    Ok(())
}

/// [`check_audio_request`]'s checkpoint and option half - all of it that
/// text-to-audio needs, since it holds only the audio stream's blocks rather
/// than the joint model's.
fn check_audio_inputs(paths: &Paths, o: &GenOpts) -> Result<(), String> {
    if !o.audio {
        return match o.audio_source {
            Some(_) => Err("an audio source is the clip's audio stream: it needs --audio".into()),
            None => Ok(()),
        };
    }
    if o.dit_config != "ltx25_22b" {
        return Err(format!(
            "audio generation needs the real audio-visual checkpoint: dit_config is {:?}, and the tiny smoke-test DiT has random weights whose audio stream would be noise, not sound",
            o.dit_config
        ));
    }
    if paths.dit.is_none() {
        return Err("audio generation needs the real DiT checkpoint: put it in the model store or pass --dit <path>".into());
    }
    if paths.audio_vae.is_none() {
        return Err("audio generation needs the audio VAE + vocoder (ltx-2.5-audio-vae-bf16.safetensors): put it in the model store or pass --audio-vae <path>".into());
    }
    if paths.text_encoder.is_none() {
        return Err(
            "audio generation needs the real text encoder: the audio stream is conditioned through its OWN text projection (text_embedding_projection.audio_aggregate_embed), which the stub context cannot stand in for - put it in the model store or pass --text-encoder <path>"
                .into(),
        );
    }
    Ok(())
}

/// What a run's banner says its denoiser is, as a pure function of the two
/// things that decide it - so the line a user reads and the object
/// [`build_denoiser`] returns cannot disagree about which arm ran.
///
/// `real` is "this is a real checkpoint config, not the tiny random-weight
/// one"; `fp32` is [`av_fp32_reference`].
pub fn av_arm_label(real: bool, audio: bool, fp32: bool) -> &'static str {
    match (real, audio, fp32) {
        (false, _, _) => "tiny random-weight DiT",
        (true, false, _) => "REAL checkpoint DiT (int8 compute, video stream only)",
        (true, true, false) => "REAL checkpoint audio+video DiT (int8 compute, both streams)",
        (true, true, true) => "REAL checkpoint audio+video DiT (host fp32 REFERENCE arm, both streams)",
    }
}

/// Refuse an audio request this machine cannot hold, against the requirement
/// of the arm that will actually run.
///
/// The two arms differ by an order of magnitude and the guard has to follow
/// the switch, or it quotes a number the code no longer has: the fp32 arm
/// needs the whole manifest expanded to f32
/// ([`crate::av_stream::AvWeights::fits_in_host_memory`]), while the quantized
/// arm holds one already-quantized copy of each block plus the non-block head
/// tensors, and streams everything else off the mapped checkpoint.
///
/// The quantized figure is derived, not written down: `cached_av_block_bytes`
/// is the same function the residency policy sizes a slot with, and the head
/// half is the manifest minus its `transformer_blocks.` prefix - so a config
/// change that grew the model cannot leave this check reading a stale size.
fn check_av_host_memory(cfg: &LtxAvDitConfig, fp32: bool) -> Result<(), String> {
    if fp32 {
        return crate::av_stream::AvWeights::fits_in_host_memory(cfg);
    }
    let want = quantized_av_host_bytes(cfg);
    let Some(avail) = crate::av_stream::available_host_bytes() else { return Ok(()) };
    let margin = want / 8;
    if avail < want + margin {
        return Err(format!(
            "ltxv: audio generation holds the audio+video DiT's blocks as int8 in host RAM ({} GiB, plus headroom) but only {} GiB is available",
            want / (1 << 30),
            avail / (1 << 30)
        ));
    }
    Ok(())
}

/// Host bytes the quantized audio-visual arm holds: one cached int8 copy of
/// every transformer block, plus the non-block head tensors as fp32.
fn quantized_av_host_bytes(cfg: &LtxAvDitConfig) -> u64 {
    let blocks = crate::block::cached_av_block_bytes(&cfg.video, &cfg.audio, crate::block::QTier::Int8) * u64::from(cfg.video.num_layers);
    let head: u64 = crate::dit::av_dit_tensor_manifest(cfg)
        .iter()
        .filter(|(name, _)| !name.starts_with("transformer_blocks."))
        .map(|(_, s)| s.iter().product::<usize>() as u64 * 4)
        .sum();
    blocks + head
}

/// Text to video. `progress(done, total, phase)` mirrors `wan::pipeline::
/// generate`'s contract; `cancel` is polled once per denoise step.
#[tracing::instrument(level = "info", name = "generate", skip_all, fields(frames = o.frames, width = o.width, height = o.height, steps = o.steps, seed = o.seed, guidance = ?o.guidance, dit_config = %o.dit_config))]
pub fn generate(paths: &Paths, prompt: &str, o: &GenOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    tracing::info!(
        prompt_chars = prompt.chars().count(),
        device = o.device.as_deref().unwrap_or("(ambient)"),
        real_dit = paths.dit.is_some(),
        real_text_encoder = paths.text_encoder.is_some(),
        "text-to-video generation starting"
    );
    let vcfg = LtxVaeConfig::conv25();
    let lat_t = vcfg.latent_frames(o.frames as u32).ok_or_else(|| {
        tracing::error!(frames = o.frames, "frame count is not 1 + 8k, which the causal VAE requires");
        format!("{} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)", o.frames)
    })?;
    if !o.width.is_multiple_of(32) || !o.height.is_multiple_of(32) {
        tracing::error!(width = o.width, height = o.height, "resolution is not a multiple of the VAE spatial stride");
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", o.width, o.height));
    }
    if o.steps == 0 {
        tracing::error!("--steps must be at least 1");
        return Err("--steps must be at least 1".into());
    }
    // Before any weight is read: an anchor pointed outside the clip is a
    // typo, and a typo should cost milliseconds rather than a model build.
    if o.mid_frame.is_some() {
        mid_anchor_frame(o.frames, o.mid_frame_at)?;
    }
    check_audio_request(paths, o)?;
    let (lh, lw) = (o.height / 32, o.width / 32);
    let (lat_t, lh, lw) = (lat_t as usize, lh, lw);
    let t = lat_t * lh * lw;

    let dit_cfg = dit_config_from_name(&o.dit_config).inspect_err(|e| tracing::error!(dit_config = %o.dit_config, error = %e, "unknown DiT config"))?;
    if dit_cfg.in_channels != vcfg.latent_channels {
        tracing::error!(in_channels = dit_cfg.in_channels, latent_channels = vcfg.latent_channels, "DiT/VAE latent width mismatch");
        return Err(format!("ltxv dit-config {:?} has in_channels {} but the VAE latent width is {}", o.dit_config, dit_cfg.in_channels, vcfg.latent_channels));
    }
    tracing::debug!(latent_frames = lat_t, latent_h = lh, latent_w = lw, tokens = t, in_channels = dit_cfg.in_channels, "latent layout resolved");
    let in_channels = dit_cfg.in_channels as usize;
    // The real `ltx25_22b` checkpoint is the DISTILLED variant
    // (`ltx-2.5-22b-distilled-transformer-*.gguf`) - distillation trains the
    // model to denoise correctly at a small set of SPECIFIC, non-uniformly-
    // spaced sigma values baked in during training
    // (`diffusion::scheduler::LTX2_DISTILLED_SIGMAS`, matched bit-exactly
    // against source), not at arbitrary intermediate noise levels a generic
    // continuous-time schedule formula would produce. Using the generic
    // `ltx2_sigmas` formula (this pipeline's ONLY schedule until now, per
    // this module's own doc) with the real checkpoint produces reasonable-
    // looking sigma numbers that the distilled weights were never trained
    // to denoise from - confirmed empirically to produce non-recognizable
    // output. `--steps`/`--base-shift`/`--max-shift`/`--stretch`/
    // `--terminal` are therefore ignored for `ltx25_22b`: the real schedule
    // has a fixed shape, not a user-tunable one. The tiny random-weight path
    // is unaffected (never distilled, so the generic formula is the correct
    // choice there, same as always).
    let is_real_distilled = o.dit_config == "ltx25_22b";
    let sigmas: Vec<f64> = if is_real_distilled {
        LTX2_DISTILLED_SIGMAS.iter().map(|&s| s as f64).collect()
    } else {
        ltx2_sigmas(t, o.steps, o.base_shift, o.max_shift, o.stretch, o.terminal)
    };
    // Whether this request takes the reference's two-stage shape, decided
    // here rather than at the denoise call site because the progress total
    // has to know: a two-stage run reports stage 1's steps, one upscale, then
    // stage 2's.
    let two_stage = should_two_stage(t, o.width, o.height, is_real_distilled, o.runtime.two_stage);
    // Phases: build, every stage-1 step, (two-stage only) one upscale and
    // every stage-2 step, decode.
    let stage2_phases = if two_stage { LTX2_STAGE2_STEPS as u32 + 1 } else { 0 };
    let total = sigmas.len() as u32 - 1 + stage2_phases + 2;
    // `--steps` is IGNORED for the distilled checkpoint (see above). That is
    // a silent override of something the user typed, so say so.
    if is_real_distilled && o.steps != sigmas.len() - 1 {
        tracing::warn!(requested_steps = o.steps, schedule_steps = sigmas.len() - 1, "the distilled checkpoint has a fixed sigma schedule; --steps is ignored");
    }
    tracing::debug!(schedule = if is_real_distilled { "ltx2_distilled" } else { "ltx2_shifted" }, steps = sigmas.len() - 1, "sigma schedule built");
    let mut timings = Timings::default();

    // ---- build the DiT: tiny config, random weights (this pipeline's
    // original path); or the real 22B checkpoint, streamed int8-compute,
    // when `--dit-config ltx25_22b` names it (needs `Paths::dit` - see this
    // module's doc and `RealDit`'s doc) ----
    // Resolve the placement ONCE, before anything is built: every stage of
    // this generation must agree about which card it is on, and a plan
    // re-resolved per step could drift if the thread's ambient selection
    // changed underneath it (see `crate::devplan`).
    let place = o.devices.resolve(o.device.as_deref());
    tracing::info!(cond_gpu = ?place.cond, uncond_gpu = ?place.uncond, text_gpu = ?place.text, cfg_parallel = place.cfg_is_parallel() && o.guidance.needs_uncond(o.audio), "device placement resolved");
    // An IC-LoRA reference is read by the adapter that was trained on it,
    // with the geometry that adapter declares - checked before any weight.
    let reference = match &o.reference {
        None => None,
        Some(_) if o.adapters.is_empty() => return Err("--reference-video needs the IC-LoRA trained to read it (--adapter): the base checkpoint has no use for reference tokens".into()),
        Some(r) => Some(StageReference::Video(r, crate::lora::reference_scale_factors(&o.adapters)?)),
    };
    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, o, &o.adapters, place)?;
    if o.audio && !dit.has_audio() {
        return Err("ltxv: audio was requested but the built denoiser carries no audio stream".into());
    }
    timings.build_dit = build_t.elapsed().as_secs_f32();
    tracing::info!(secs = timings.build_dit, "transformer built");
    if cancel.is_cancelled() {
        tracing::warn!(phase = "after build", "cancelled");
        return Err("cancelled".into());
    }

    // ---- denoise ----------------------------------------------------------
    // Moved, never cloned: the projected caption is the widest host buffer
    // this function holds besides the latents themselves.
    let TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, uncond_valid, len: context_len, a_cond: text_a_cond, a_uncond: text_a_uncond } =
        build_context(paths, prompt, dit_cfg, o, place, &mut timings.text_encode)?;

    // ---- the stage plan ---------------------------------------------------
    // `vraw`/`vweights` are loaded here (not at decode time below, which
    // reuses them) because image conditioning needs a real VAE ENCODE before
    // any denoising, once per stage at that stage's own resolution.
    let vraw = read_any(&paths.vae)?;
    let crate::import::VideoVae { arch: varch, weights: vweights } = crate::import::import_vae(vraw, &vcfg)?;
    // The audio stream's own initial noise, drawn from a seed derived from -
    // but never equal to - the video stream's, so one `--seed` reproduces the
    // whole audio-visual run and the two streams never start from the same
    // numbers.
    let ta = crate::audio::latent_frames(o.frames, o.fps);
    let audio_latent0 = o.audio.then(|| {
        tracing::info!(audio_tokens = ta, token_dim = crate::audio::TOKEN_DIM, seconds = o.frames as f32 / o.fps as f32, "audio latent shape resolved from the clip's own duration");
        seeded_noise(ta * crate::audio::TOKEN_DIM as usize, o.seed ^ AUDIO_SEED_SALT)
    });
    // An audio source is the whole audio stream, carried as a frozen
    // context: clean, at timestep 0 and re-pinned every step, exactly like a
    // long-form window's carried prefix - only it covers every token.
    let audio_frozen: Vec<f32> = match &o.audio_source {
        Some(source) => {
            let path = paths.audio_vae.as_deref().ok_or("ltxv: an audio source is encoded through the audio VAE, and no audio VAE path is set")?;
            let (_, vae_raw): (Vec<_>, Vec<_>) = checkpoint::safetensors::read(path)?.into_iter().partition(|t| t.name.starts_with("vocoder."));
            let vae_w = crate::import::import_audio_vae(vae_raw, &crate::audio_vae::AudioVaeConfig::ltx25())?;
            crate::audio::encode(&vae_w, source, ta, o.device.as_deref())?
        }
        None => Vec::new(),
    };
    let (a_ctx_cond, a_ctx_uncond) = (text_a_cond.unwrap_or_default(), text_a_uncond.unwrap_or_default());
    let stage_ctx = StageCtx {
        dit: dit.as_ref(),
        vcfg: &vcfg,
        vweights: &vweights,
        o,
        lat_t,
        in_channels,
        ctx_cond: Caption::of(&ctx_cond, &context_valid),
        ctx_uncond: Caption::of(&ctx_uncond, &uncond_valid),
        context_len,
        a_ctx_cond: &a_ctx_cond,
        a_ctx_uncond: &a_ctx_uncond,
        cancel,
        reference,
    };
    let denoise_t = Instant::now();
    let stage_out = if two_stage {
        // The reference's own shape for the distilled checkpoint
        // (`ltx_pipelines.distilled.DistilledPipeline.__call__`): build the
        // clip at HALF resolution on the full distilled schedule, carry it up
        // with the real spatial x2 latent upscaler, then spend three more
        // deterministic steps detailing at the requested resolution. See
        // [`SINGLE_STAGE_MAX_TOKENS`] for the measurement that says a single
        // stage stops working past 4096 video tokens and what it looks like
        // when it does.
        let upsampler_path = paths.spatial_upsampler.as_deref().ok_or_else(|| {
            tracing::error!(tokens = t, width = o.width, height = o.height, "a two-stage request needs the spatial latent upscaler");
            format!(
                "{}x{} is {t} video tokens, past the {SINGLE_STAGE_MAX_TOKENS}-token ceiling the distilled schedule holds in ONE stage, so it needs the reference's two-stage path and its spatial x2 latent upscaler (ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors): put it in the model store or pass --spatial-upsampler <path>. Forcing --two-stage off runs one stage instead and is known to disintegrate the end of the clip at this size.",
                o.width, o.height
            )
        })?;
        // One refinement in this path, so the cache below is filled exactly
        // once - it exists so `Refine` has one shape for all three callers,
        // and so this generation does not hold the upscaler's expansion
        // beside the transformer through a stage that never touches it.
        let upsampler = SpatialUpsampler::default();
        let (w1, h1) = (o.width / 2, o.height / 2);
        let (lh1, lw1) = (h1 / 32, w1 / 32);
        let stage1 = denoise_stage(
            &stage_ctx,
            Stage {
                mask: None,
                width: w1,
                height: h1,
                sigmas: &sigmas,
                eta: o.eta,
                seed_chw: None,
                context: None,
                seed_salt: 0,
                done_before: 0,
                label: "stage1",
                audio: audio_latent0.map(|latent| AudioStage { latent, context: audio_frozen.clone(), frozen: None }),
                slots: None,
            },
            total,
            &mut progress,
        )?;

        // An IC-LoRA run refines OUT of context by default, as the
        // reference's own recipe does: the bare checkpoint, no reference
        // tokens. Same checkpoint, so the quantized-weight cache is shared and
        // nothing is re-read; only the adapters are left off.
        let bare = match (&reference, o.refine_in_context) {
            (Some(_), false) => {
                dit.release_devices();
                Some(build_denoiser(paths, dit_cfg, o, &[], place)?)
            }
            _ => None,
        };
        let refine_ctx = match &bare {
            Some(b) => StageCtx { dit: b.as_ref(), reference: None, ..stage_ctx },
            None => StageCtx { ..stage_ctx },
        };
        upscale_and_refine(
            &refine_ctx,
            Refine {
                upsampler: &upsampler,
                upsampler_path,
                slots: None,
                audio: stage1.audio.map(|latent| AudioStage { latent, context: audio_frozen.clone(), frozen: None }),
                latent_chw: &stage1.video_chw,
                lat_t,
                lh1,
                lw1,
                width: o.width,
                height: o.height,
                sigmas: &stage2_sigmas(LTX2_STAGE2_STEPS)?,
                eta: refine_eta(o),
                context: None,
                seed_salt: 0,
                done_before: sigmas.len() as u32 - 1,
                label: "stage2",
            },
            total,
            &mut progress,
        )?
    } else {
        denoise_stage(
            &stage_ctx,
            Stage {
                mask: None,
                width: o.width,
                height: o.height,
                sigmas: &sigmas,
                eta: o.eta,
                seed_chw: None,
                context: None,
                seed_salt: 0,
                done_before: 0,
                label: "single",
                audio: audio_latent0.map(|latent| AudioStage { latent, context: audio_frozen.clone(), frozen: None }),
                slots: None,
            },
            total,
            &mut progress,
        )?
    };
    let final_chw = stage_out.video_chw;
    let audio_latent = stage_out.audio;
    // Release the DiT's own device context (for `RealDit`, its resident
    // `Gpu`) before the VAE decode below opens its own - real device memory
    // is not this pipeline's to hold onto once the denoise loop is done
    // with it.
    drop(dit);
    timings.denoise = denoise_t.elapsed().as_secs_f32();
    // `sigmas.len() - 1`, not `o.steps`: the real distilled schedule ignores
    // `--steps` entirely (see where `sigmas` is built, above). A two-stage
    // run adds its own refinement steps on top, and reporting only stage 1's
    // would make `secs_per_forward` a fiction.
    timings.steps = sigmas.len() - 1 + if two_stage { LTX2_STAGE2_DISTILLED_SIGMAS.len() - 1 } else { 0 };
    timings.tokens = t;
    timings.forwards_per_step = o.guidance.forwards_per_step(o.audio);
    tracing::info!(secs = timings.denoise, steps = timings.steps, secs_per_forward = timings.secs_per_forward(), "denoise done");

    if cancel.is_cancelled() {
        tracing::warn!(phase = "after denoise", "cancelled");
        return Err("cancelled".into());
    }

    // ---- decode -------------------------------------------------------------
    progress(total - 1, total, "vae decode");
    tracing::info!(path = %paths.vae, latent_frames = lat_t, "VAE decode starting");
    let decode_t = Instant::now();
    // `denoise_stage` already stripped any appended image-conditioning tokens
    // and returned the `[C, lat_t, lh, lw]` video latent - the conditioning
    // frame is the source image itself, not a new frame to render.
    let (pixels, frames) = decode_video(&vcfg, varch, &vweights, lat_t as u32, lh as u32, lw as u32, &final_chw, o)?;
    let (w, h) = (o.width, o.height);
    if pixels.len() != 3 * frames * h * w {
        return Err(format!("VAE returned {} values, expected {}", pixels.len(), 3 * frames * h * w));
    }
    let out = chw_to_rgb8(&pixels, frames, h, w);
    timings.decode = decode_t.elapsed().as_secs_f32();
    tracing::info!(secs = timings.decode, frames, "VAE decode done");

    // ---- decode the sound -------------------------------------------------
    // Same forwards, same conditioning, same time window: the audio latent
    // here came out of the joint denoise above, not a second pass.
    let audio = match (audio_latent, &o.audio_source) {
        // The source soundtrack IS the clip's audio: the denoise held it
        // frozen, so decoding its latent would only add the VAE's loss.
        (Some(_), Some(source)) => {
            let mut clip = source.clone();
            let video_seconds = frames as f32 / o.fps as f32;
            let keep = (video_seconds * clip.sample_rate as f32).round() as usize;
            clip.channels.iter_mut().for_each(|c| c.truncate(keep));
            clip.pad_to_seconds(video_seconds);
            Some(clip)
        }
        (Some(latent), None) => {
            let path = paths.audio_vae.as_deref().expect("generate: an audio request is refused earlier without an audio VAE path");
            let audio_t = Instant::now();
            let mut clip = decode_audio_latent(path, &latent, ta, o.device.as_deref())?;
            // The causal audio VAE's first latent frame covers one mel frame
            // rather than four, so the decode is exactly three mel frames
            // short of the clip. Hold the last sample over that gap so the
            // two tracks are the same length (see `crate::audio`'s doc).
            let video_seconds = frames as f32 / o.fps as f32;
            let before = clip.seconds();
            clip.pad_to_seconds(video_seconds);
            timings.audio_decode = audio_t.elapsed().as_secs_f32();
            tracing::info!(secs = timings.audio_decode, samples = clip.samples_per_channel(), rate = clip.sample_rate, decoded_seconds = before, video_seconds, "audio decode done");
            Some(clip)
        }
        (None, _) => None,
    };
    progress(total, total, "done");
    tracing::info!(frames, width = w, height = h, fps = o.fps, audio = audio.is_some(), total_secs = timings.total(), "generation done");
    Ok((Video { width: w as u32, height: h as u32, fps: o.fps, frames: out, audio }, timings))
}

/// Whether [`predict_frames`] can run for a request: the duration head, the
/// real checkpoint whose connectors it reads, and the text encoder are all
/// configured. A request that names no length takes its default otherwise.
pub fn can_predict_frames(paths: &Paths, o: &GenOpts) -> bool {
    paths.duration_head.is_some() && paths.dit.is_some() && paths.text_encoder.is_some() && o.dit_config == "ltx25_22b"
}

/// The streams whose connector outputs a video generation's duration
/// prediction pools: both, with or without sound. Every reference video
/// pipeline hands the head both encodings (`ti2vid_*`, `distilled`, `dfr`),
/// and a silent clip is the same generation with its soundtrack dropped, so
/// it takes the same length. The head is not robust to losing a stream:
/// over the video encoding alone it predicts a single quick action four
/// times longer than a multi-beat scene.
pub fn prediction_streams() -> &'static [crate::config::Stream] {
    &[crate::config::Stream::Video, crate::config::Stream::Audio]
}

/// The clip length the duration head predicts for `prompt`, in frames at
/// `o.fps` - `resolve_num_frames` with the default `AutoDuration`: one to
/// twenty seconds, on the causal VAE's `8k + 1` grid.
pub fn predict_frames(paths: &Paths, prompt: &str, o: &GenOpts, streams: &[crate::config::Stream]) -> Result<usize, String> {
    let seconds = predict_seconds(paths, prompt, o, streams)?;
    let frames = crate::duration_head::frames_for_seconds(seconds, o.fps as f64, crate::duration_head::MIN_SECONDS, crate::duration_head::MAX_SECONDS);
    tracing::info!(predicted_seconds = seconds, frames, fps = o.fps, "duration head predicted the clip length");
    Ok(frames)
}

/// The duration head's raw, unclamped prediction for `prompt`, in seconds,
/// over [`prompt_encodings`]'s `streams`.
pub fn predict_seconds(paths: &Paths, prompt: &str, o: &GenOpts, streams: &[crate::config::Stream]) -> Result<f64, String> {
    let head_path = paths.duration_head.as_deref().ok_or("predicting the clip length needs the duration head")?;
    let (video, audio) = prompt_encodings(paths, prompt, o, streams)?;
    let head_cfg = crate::duration_head::DurationHeadConfig::ltx25();
    let head_w = crate::import::import_duration_head(checkpoint::safetensors::read(head_path)?, &head_cfg)?;
    Ok(crate::duration_head::DurationHead::new(head_cfg, &head_w).forward(video.as_deref(), audio.as_deref()) as f64)
}

/// `prompt`'s connector outputs for `streams` - `[context_len, dim]` each,
/// the reference's `EmbeddingsProcessorOutput.{video,audio}_encoding` and
/// what the duration head pools: the DiT's own view of the prompt.
///
/// The prompt is encoded with exactly the flags the generation will use (the
/// text cache then answers the generation's own encode), and routed through
/// the checkpoint's connectors into the checkpoint cache the first forward
/// reads - a prediction costs the head itself and nothing twice. A video
/// generation's head sees both streams ([`prediction_streams`]) and
/// text-to-audio ([`Stream::Audio`] alone) the audio stream, as the
/// reference's pipelines pass them. The audio stream's projection is built
/// whenever it is asked for, sound or not; a silent generation's own encode
/// is then answered from that larger cache entry.
///
/// [`Stream::Audio`]: crate::config::Stream::Audio
#[allow(clippy::type_complexity)]
pub fn prompt_encodings(paths: &Paths, prompt: &str, o: &GenOpts, streams: &[crate::config::Stream]) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>), String> {
    use crate::config::Stream;
    let o = &GenOpts { audio: o.audio || streams.contains(&Stream::Audio), ..o.clone() };
    let text = prompt_context(paths, prompt, o)?;
    let route = |stream: Stream| -> Result<Vec<f32>, String> {
        let rows = match stream {
            Stream::Video => &text.cond,
            Stream::Audio => text.a_cond.as_ref().ok_or("routing the prompt through the audio connector needs the prompt's audio projection")?,
        };
        route_prompt(paths, o, stream, rows, &text.valid)
    };
    let video = streams.contains(&Stream::Video).then(|| route(Stream::Video)).transpose()?;
    let audio = streams.contains(&Stream::Audio).then(|| route(Stream::Audio)).transpose()?;
    Ok((video, audio))
}

/// `prompt` encoded for `o` by the real text encoder - the
/// [`TextContext`] a generation denoises against, before any connector.
pub fn prompt_context(paths: &Paths, prompt: &str, o: &GenOpts) -> Result<TextContext, String> {
    if !can_predict_frames(paths, o) {
        return Err("predicting the clip length needs the duration head, the real DiT checkpoint (dit_config ltx25_22b) and the real text encoder".into());
    }
    let src = crate::gguf_src::LtxvGgufSource::open(paths.dit.as_deref().unwrap_or_default())?;
    let mut encode_secs = 0.0;
    build_context(paths, prompt, src.config().video, o, o.devices.resolve(o.device.as_deref()), &mut encode_secs)
}

/// One stream's framed prompt `rows` (masked by `valid`) routed through the
/// real checkpoint's embeddings connector for that stream, through the
/// checkpoint cache the generation's first forward reads.
pub fn route_prompt(paths: &Paths, o: &GenOpts, stream: crate::config::Stream, rows: &[f32], valid: &[f32]) -> Result<Vec<f32>, String> {
    use crate::config::Stream;
    let dit_path = paths.dit.as_deref().ok_or("routing a prompt needs the real DiT checkpoint")?;
    let src = crate::gguf_src::LtxvGgufSource::open(dit_path)?;
    let av_cfg = *src.config();
    let cfg = match stream {
        Stream::Video => av_cfg.video,
        Stream::Audio => av_cfg.audio_stream(),
    };
    let gpu = gpu_core::Gpu::open(o.device.as_deref(), &crate::block::KERNELS);
    let cache = crate::block::GenerationCache::for_checkpoint(dit_path);
    Ok(crate::dit::routed_context(&gpu, &src, &cfg, stream, rows, valid, valid.len(), &cache, None))
}

/// Text-to-audio options.
#[derive(Clone, Debug)]
pub struct AudioOpts {
    /// The clip's length; the audio latent runs at
    /// [`crate::audio::LATENTS_PER_SECOND`]. `None` lets the duration head
    /// predict it from the prompt ([`predict_frames`], on `base.fps`'s frame
    /// grid as the reference snaps it), or takes [`Self::DEFAULT_SECONDS`]
    /// when no duration head is configured.
    pub seconds: Option<f64>,
    /// Everything a generation shares. Only the audio stream's guidance
    /// (`guidance.audio`) applies; the video-only fields go unread.
    pub base: GenOpts,
}

impl AudioOpts {
    /// The length a request that names none takes without a duration head.
    pub const DEFAULT_SECONDS: f64 = 5.0;
    /// The frame rate a predicted length snaps on - the reference pipelines'
    /// own `--frame-rate` default.
    pub const FRAME_RATE: usize = 24;
}

impl Default for AudioOpts {
    fn default() -> AudioOpts {
        AudioOpts { seconds: None, base: GenOpts { dit_config: "ltx25_22b".into(), audio: true, fps: AudioOpts::FRAME_RATE, ..GenOpts::default() } }
    }
}

/// Text-to-audio: an AV checkpoint's audio stream denoised ALONE
/// (`ltx_pipelines.t2a_one_stage`), then decoded to a waveform.
///
/// The video half is never read: the denoiser is the single-stream forward
/// at [`LtxAvDitConfig::audio_stream`], whose blocks are the audio stream's
/// own (`LTXAudioOnlyModelConfigurator`). The stream takes the audio
/// guidance ([`crate::guidance::Guidance::audio`]) with the cross-modal term
/// off, as the reference's `modality_scale=1.0` sets it - there is no video
/// to guide against. The schedule is the checkpoint's: the distilled table
/// for the distilled checkpoint, otherwise `LTX2Scheduler` at its default
/// token count (`MAX_SHIFT_ANCHOR`, since an audio latent has no spatial
/// size to shift by).
pub fn generate_audio(paths: &Paths, prompt: &str, opts: &AudioOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(crate::audio::AudioClip, Timings), String> {
    if let Some(seconds) = opts.seconds.filter(|s| !(s.is_finite() && *s > 0.0)) {
        return Err(format!("the clip length must be a positive number of seconds, got {seconds}"));
    }
    let o = GenOpts { audio: true, ..opts.base.clone() };
    if o.audio_source.is_some() {
        return Err("text-to-audio generates the soundtrack; an audio source has nothing left to generate".into());
    }
    check_audio_inputs(paths, &o)?;
    let seconds = match opts.seconds {
        Some(s) => s,
        None if can_predict_frames(paths, &o) => predict_frames(paths, prompt, &o, &[crate::config::Stream::Audio])? as f64 / o.fps as f64,
        None => AudioOpts::DEFAULT_SECONDS,
    };
    let ta = (seconds * crate::audio::LATENTS_PER_SECOND).round().max(1.0) as usize;
    let sigmas: Vec<f64> = if o.dit_config == "ltx25_22b" {
        LTX2_DISTILLED_SIGMAS.iter().map(|&s| s as f64).collect()
    } else {
        ltx2_sigmas(diffusion::scheduler::LTX2_MAX_SHIFT_ANCHOR as usize, o.steps, o.base_shift, o.max_shift, o.stretch, o.terminal)
    };
    let total = sigmas.len() as u32 - 1 + 2;
    let mut timings = Timings::default();
    let place = o.devices.resolve(o.device.as_deref());

    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit_path = paths.dit.as_deref().expect("check_audio_inputs refuses a request without a DiT path");
    let src = crate::gguf_src::LtxvGgufSource::open(dit_path)?;
    let av_cfg = *src.config();
    let cfg = av_cfg.audio_stream();
    let mut head = crate::dit::load_av_head_tensors_from_source(&src, &av_cfg);
    let adapter = if o.adapters.is_empty() { None } else { Some(block_adapter(crate::lora::runtime_adapter(&o.adapters)?, &src, &mut head)?) };
    let dit = RealDit { stream: crate::config::Stream::Audio, cfg, src, head, device: o.device.clone(), cache: crate::block::GenerationCache::for_checkpoint(dit_path), place, sessions: Default::default(), adapter, resident_blocks: o.runtime.resident_blocks };
    timings.build_dit = build_t.elapsed().as_secs_f32();
    tracing::info!(audio_tokens = ta, seconds, layers = cfg.num_layers, inner_dim = cfg.inner_dim, "audio-only DiT ready (the video half is never read)");

    let TextContext { valid: context_valid, uncond_valid, len: context_len, a_cond, a_uncond, .. } = build_context(paths, prompt, av_cfg.video, &o, place, &mut timings.text_encode)?;
    let a_cond = a_cond.ok_or("text-to-audio needs the text encoder's audio projection")?;
    let a_uncond = a_uncond.unwrap_or_default();
    let audio_guidance = crate::guidance::GuiderParams { modality: 1.0, ..o.guidance.audio.clone() };
    let guidance = crate::guidance::Guidance { video: audio_guidance.clone(), audio: audio_guidance };

    let latent0 = seeded_noise(ta * crate::audio::TOKEN_DIM as usize, o.seed ^ AUDIO_SEED_SALT);
    let positions = crate::audio::positions(ta);
    let denoise_t = Instant::now();
    let sampling = Sampling { sampler: o.sampler, eta: o.eta, s_noise: o.s_noise, seed: o.seed ^ 0x4e_4f_49_53_45 ^ AUDIO_SEED_SALT };
    #[rustfmt::skip]
    let latent = denoise(&dit, &sigmas, latent0, &positions, &vec![0.0; ta], Caption::of(&a_cond, &context_valid), Caption::of(&a_uncond, &uncond_valid), context_len, ta,
        &guidance, sampling, total, None, cancel, None, &mut |done, tot, phase| progress(1 + done, tot, phase))?;
    timings.denoise = denoise_t.elapsed().as_secs_f32();
    drop(dit);

    let path = paths.audio_vae.as_deref().expect("check_audio_inputs refuses a request without an audio VAE path");
    let audio_t = Instant::now();
    let mut clip = decode_audio_latent(path, &latent, ta, o.device.as_deref())?;
    // The causal VAE's first latent frame decodes to one mel frame rather
    // than four (see `crate::audio`'s doc); hold the last sample over the gap
    // so the clip is the length asked for.
    clip.pad_to_seconds(seconds as f32);
    timings.audio_decode = audio_t.elapsed().as_secs_f32();
    progress(total, total, "done");
    tracing::info!(seconds = clip.seconds(), rate = clip.sample_rate, total_secs = timings.total(), "text-to-audio done");
    Ok((clip, timings))
}

/// Read the audio VAE + vocoder from one checkpoint and decode an audio
/// latent all the way to a waveform.
///
/// Read here rather than beside the video VAE at the top of [`generate`]
/// because nothing before the denoise loop needs it: a silent generation must
/// not pay for a checkpoint it never touches, and an audio-visual one is
/// holding the fp32 AV DiT until the loop ends anyway.
fn decode_audio_latent(path: &str, latent: &[f32], ta: usize, device: Option<&str>) -> Result<crate::audio::AudioClip, String> {
    let acfg = crate::audio_vae::AudioVaeConfig::ltx25();
    // ONE read, split by prefix: the audio VAE and the vocoder (base and
    // bandwidth extension) are disjoint subsets of one file, and `StTensor`
    // is not `Clone`.
    let (voc_raw, vae_raw): (Vec<_>, Vec<_>) = checkpoint::safetensors::read(path)?.into_iter().partition(|t| t.name.starts_with("vocoder."));
    let vae_w = crate::import::import_audio_vae(vae_raw, &acfg)?;
    let voc_w = crate::import::import_vocoder_with_bwe(voc_raw)?;
    Ok(crate::audio::decode(&vae_w, &voc_w, latent, ta, device))
}

// ============================================================================
// Post-hoc upscaling of an already-rendered clip
// ============================================================================

/// `[3, frames, h, w]` VAE output in `[-1,1]` to [`Video::frames`]'
/// interleaved RGB8.
///
/// No clamp is applied by the model itself - upstream clamps to `[-1,1]`
/// outside it, same convention `wan::pipeline` follows for its own VAE.
fn chw_to_rgb8(pixels: &[f32], frames: usize, h: usize, w: usize) -> Vec<Vec<u8>> {
    let plane = frames * h * w;
    (0..frames)
        .map(|f| {
            let mut px = vec![0u8; h * w * 3];
            for c in 0..3 {
                let base = c * plane + f * h * w;
                for i in 0..h * w {
                    px[i * 3 + c] = (127.5 * (pixels[base + i].clamp(-1.0, 1.0) + 1.0)) as u8;
                }
            }
            px
        })
        .collect()
}

/// The inverse of [`chw_to_rgb8`] over a slice of a clip's frames: interleaved
/// RGB8 to the `[3, frames, h, w]` `[-1,1]` volume [`LtxVaeEncoder::encode`]
/// takes. Same `[0,255] -> [-1,1]` map [`encode_still`] uses for a single
/// conditioning image.
fn rgb8_to_chw(frames: &[Vec<u8>], h: usize, w: usize) -> Vec<f32> {
    // The frames stacked as one `frames * h`-row image: its planar layout IS
    // the `[3, frames, h, w]` volume.
    imaging::pixels::hwc_to_chw(&frames.concat(), 3, frames.len() * h, w).into_iter().map(vae_input).collect()
}

/// The largest video-token count [`upscale`] will put through ONE refinement
/// pass before it splits the clip instead.
///
/// **Derived, not measured, and deliberately conservative.** The DiT's
/// per-forward adaLN table is `[t, 9 * inner_dim]` fp32 - 147456 bytes per
/// token at the real checkpoint's `inner_dim = 4096` - against a
/// `max_storage_buffer_binding_size` this box's Tesla P40 reports as 2047
/// MiB, which that ONE table crosses at t ~= 14556. This constant sits below
/// that with room for the other per-forward slabs, and above 8160, the
/// largest refinement token count this crate has a recorded real run at (the
/// two-stage 1080p path). Where between those the real limit sits is not
/// measured and this number does not pretend to know; the binding size is
/// also a per-adapter figure, so a card that reports less would move the
/// crossover down.
///
/// Two things it is NOT. It is not [`SINGLE_STAGE_MAX_TOKENS`], which is a
/// QUALITY ceiling on building structure from noise; refinement starts from
/// content and is not subject to it. And it does not bound [`generate`],
/// whose stage 2 runs whatever the requested resolution implies - exceeding
/// the binding limit there fails LOUDLY at buffer creation
/// (`Buffer size N is greater than the maximum buffer size`), so this is
/// about not spending an hour to reach that abort, not about avoiding silent
/// corruption.
pub const REFINE_MAX_TOKENS: usize = 12288;

/// How [`upscale`] splits a clip whose refinement will not fit in one pass:
/// one [`crate::longform::Window`] per refinement pass, in order.
///
/// **This is [`crate::longform::window_plan`] and nothing else.** A clip too
/// long to refine in one pass and a clip too long to GENERATE in one window
/// pose the same question - how much of the previous pass does the next one
/// have to see - and it already has a measured answer: the previous pass's own
/// last latent frames, frozen at sigma 0, which held a seam at ratio 0.99
/// against naive chaining's 0.85. So refinement plans with the same function,
/// carries with the same [`crate::longform::carry_tail`], and freezes with the
/// same [`LatentContext`]. Each pass VAE-encodes the source range
/// [`crate::longform::Window::source_first_frame`] names, carries it up with
/// the x2 upscaler, overwrites the leading `context` latent frames with the
/// previous pass's own refined output, and refines - so what the model sees at
/// a boundary is real refined content, not the same source frames re-imagined
/// from scratch.
///
/// The one thing refinement adds is [`crate::longform::fitted_context`]: an
/// upscaled grid is four times as dense per latent frame as the grid it came
/// from, so `context + 1` latent frames do not always fit and the context is
/// reduced to what does rather than the clip being refused. See that function
/// for why refinement compromises where generation refuses.
///
/// A clip that fits one pass is one pass, carrying nothing - byte for byte the
/// refinement a single-pass upscale already ran.
pub fn refine_plan(frames: usize, out_lh: usize, out_lw: usize, context: usize, max_tokens: usize) -> Result<Vec<crate::longform::Window>, String> {
    let fitted = crate::longform::fitted_context(out_lh, out_lw, context, max_tokens)?;
    crate::longform::window_plan(frames, out_lh, out_lw, fitted, max_tokens)
}

/// Everything [`upscale`] varies beyond what a generation already varies.
#[derive(Clone, Debug)]
pub struct UpscaleOpts {
    /// Spatial factor. Only `2` - that is the factor the official checkpoint
    /// (`ltx-2.5-latent-spatial-upscaler-x2-*`) implements, and there is no
    /// second one to select.
    pub factor: usize,
    /// Refinement steps, `1..=`[`LTX2_STAGE2_STEPS`]. See [`stage2_sigmas`]
    /// for why this can only pick a suffix of the distilled table.
    pub refine_steps: usize,
    /// Clean latent frames carried from each refinement pass into the next,
    /// [`crate::longform::CONTEXT_LATENT_FRAMES`] by default and reduced by
    /// [`crate::longform::fitted_context`] when the output grid cannot hold
    /// that many plus a frame to refine.
    ///
    /// Lowering it is a real lever and costs continuity to buy passes: a pass
    /// spends `context` of its latent-frame budget before it refines anything,
    /// so at a tight grid `context + 1 == max_lat` and every frame carried is
    /// a frame not refined this pass.
    pub context_latent_frames: usize,
    /// Per-pass video-token ceiling, [`REFINE_MAX_TOKENS`] by default.
    pub max_refine_tokens: usize,
    /// Seed, guidance, device placement, `dit_config`. `frames`, `width` and
    /// `height` are IGNORED - they come from the input clip, which is the
    /// whole point of this entry point.
    pub base: GenOpts,
}

impl Default for UpscaleOpts {
    fn default() -> UpscaleOpts {
        UpscaleOpts {
            factor: 2,
            refine_steps: LTX2_STAGE2_STEPS,
            context_latent_frames: crate::longform::CONTEXT_LATENT_FRAMES,
            max_refine_tokens: REFINE_MAX_TOKENS,
            base: GenOpts::default(),
        }
    }
}

/// What a refinement pass's index is folded into its noise seed with, so two
/// passes of one clip never draw the same refinement noise.
///
/// Multiplied by the index rather than XORed, for
/// [`crate::longform::SCENE_SEED_SALT`]'s reason: pass 0 - the only pass a
/// clip that fits has - keeps the caller's seed EXACTLY, so a single-pass
/// upscale stays bit for bit the run it already was.
const REFINE_SEED_SALT: u64 = 0x5245_4649_4e45_0001;

/// Re-render an already-finished clip at twice its spatial resolution:
/// VAE-encode it, carry the latent up with the official x2 latent spatial
/// upscaler, refine at the new size, VAE-decode.
///
/// This is [`generate`]'s two-stage tail with the generation removed - the
/// same [`upscale_and_refine`], the same upscaler, the same un-normalize
/// sandwich, the same distilled refinement schedule - applied to a latent
/// that came from pixels on disk instead of from stage 1. Nothing about the
/// mechanics is re-derived here; what this function adds is the encode, the
/// plan ([`refine_plan`]) and the reassembly.
///
/// **A clip too long for one refinement pass is refined in several, and they
/// are one clip rather than several.** Each pass after the first freezes the
/// previous pass's own last latent frames at the head of its sequence, exactly
/// as [`generate_long`]'s continuation windows do - see [`refine_plan`]. The
/// rolling state is one latent slab of at most `context_latent_frames` frames,
/// so a ten-minute clip costs the same host memory as a ten-second one.
///
/// **`prompt` matters.** The refinement is a diffusion pass, not a
/// deterministic filter: the DiT is asked what this content should look like
/// with more detail, and it answers against the text context. Passing the
/// clip's original generation prompt is worth doing; passing an empty string
/// is legal and refines against an empty context, which costs detail. There
/// is no way to recover a prompt from a video file, so the caller has to
/// supply it.
#[tracing::instrument(level = "info", name = "upscale", skip_all, fields(frames = clip.frames.len(), width = clip.width, height = clip.height, factor = o.factor, seed = o.base.seed))]
pub fn upscale(paths: &Paths, prompt: &str, clip: &Video, o: &UpscaleOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    if o.base.reference.is_some() {
        return Err("--reference-video is applied by a single-window generation (`t2v` within one denoising window); upscale does not route a reference".into());
    }
    if o.factor != 2 {
        return Err(format!(
            "factor {} is not implemented: the official LTX-2.5 latent spatial upscaler is an x2 network and this pipeline runs that network, not a resampler",
            o.factor
        ));
    }
    let (w, h) = (clip.width as usize, clip.height as usize);
    let frames = clip.frames.len();
    if w == 0 || h == 0 || frames == 0 {
        return Err("the input clip is empty".into());
    }
    if !w.is_multiple_of(32) || !h.is_multiple_of(32) {
        return Err(format!("{w}x{h} is not a multiple of 32 (the VAE's spatial stride) - the input clip has to be VAE-representable before it can be upscaled"));
    }
    if let Some(bad) = clip.frames.iter().position(|f| f.len() != w * h * 3) {
        return Err(format!("frame {bad} is {} bytes, expected {} for {w}x{h} RGB8", clip.frames[bad].len(), w * h * 3));
    }
    let (out_w, out_h) = (w * o.factor, h * o.factor);
    let (lh, lw) = (out_h / 32, out_w / 32);
    let plan = refine_plan(frames, lh, lw, o.context_latent_frames, o.max_refine_tokens)?;
    let sigmas = stage2_sigmas(o.refine_steps)?;

    let vcfg = LtxVaeConfig::conv25();
    let dit_cfg = dit_config_from_name(&o.base.dit_config).inspect_err(|e| tracing::error!(dit_config = %o.base.dit_config, error = %e, "unknown DiT config"))?;
    if dit_cfg.in_channels != vcfg.latent_channels {
        return Err(format!("ltxv dit-config {:?} has in_channels {} but the VAE latent width is {}", o.base.dit_config, dit_cfg.in_channels, vcfg.latent_channels));
    }
    // One cache for the whole run: every pass refines with the same weights,
    // read on the first pass and reused by the rest (see
    // [`SpatialUpsampler`]).
    let upsampler = SpatialUpsampler::default();
    let upsampler_path = paths.spatial_upsampler.as_deref().ok_or_else(|| {
        tracing::error!("upscaling needs the spatial latent upscaler");
        "upscaling IS the spatial latent upscaler (ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors): put it in the model store or pass --upsampler-spatial <path>".to_string()
    })?;
    let carried = plan.get(1).map(|w| w.context).unwrap_or(0);
    tracing::info!(passes = plan.len(), carried_latent_frames = carried, out_width = out_w, out_height = out_h, refine_steps = o.refine_steps, "upscale planned");
    if carried < o.context_latent_frames && plan.len() > 1 {
        // The one quality compromise this path makes silently otherwise, and
        // it is forced by the output grid rather than chosen - see
        // `longform::fitted_context`.
        tracing::warn!(
            requested = o.context_latent_frames,
            carried,
            tokens_per_pass = o.max_refine_tokens,
            tokens_per_latent_frame = lh * lw,
            "a {out_w}x{out_h} refinement pass cannot hold the requested latent context; each pass carries what fits of the previous one instead"
        );
    }

    // Phases: build, then per pass one encode, one upscale, its refinement
    // steps and one decode.
    let per_pass = 3 + o.refine_steps as u32;
    let total = 1 + plan.len() as u32 * per_pass;
    let mut timings = Timings::default();
    let place = o.base.devices.resolve(o.base.device.as_deref());
    tracing::info!(cond_gpu = ?place.cond, uncond_gpu = ?place.uncond, text_gpu = ?place.text, "device placement resolved");

    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, &o.base, &o.base.adapters, place)?;
    timings.build_dit = build_t.elapsed().as_secs_f32();
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    if prompt.trim().is_empty() {
        // The one quality lever a caller can accidentally leave at zero, and
        // the output looks plausible either way - see this function's doc.
        tracing::warn!("no prompt: the refinement pass will denoise against an empty text context, which costs detail it would otherwise recover");
    }
    let text = build_context(paths, prompt, dit_cfg, &o.base, place, &mut timings.text_encode)?;
    let (ctx_cond, ctx_uncond, context_valid, uncond_valid, context_len) = (text.cond, text.uncond, text.valid, text.uncond_valid, text.len);

    let crate::import::VideoVae { arch: varch, weights: vweights } = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
    let in_channels = dit_cfg.in_channels as usize;
    let work_t = Instant::now();
    let mut out_frames: Vec<Vec<u8>> = Vec::with_capacity(frames);
    // VAE encode and VAE decode both land in `Timings::decode` - it is the
    // VAE's share of the run, and this entry point runs the encoder too.
    let mut vae_secs = 0.0f32;

    // The rolling state, and the whole of it: at most `carried` latent frames
    // at the OUTPUT resolution. It does not grow with the clip's length.
    let mut carried_latent: Option<(Vec<f32>, usize)> = None;

    for (si, pass) in plan.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let done_before = 1 + si as u32 * per_pass;
        let lat_t = pass.latent_frames();
        // The pass's WHOLE decode, carried context included: a latent frame
        // cannot be decoded without the frames around it, so the frames the
        // context covers are read, refined and thrown away rather than
        // stitched in from the source.
        let (start, len) = (pass.source_first_frame(), pass.decoded_frames());
        progress(done_before, total, "vae encode");
        tracing::info!(pass = si, of = plan.len(), first_frame = start, frames = len, latent_frames = lat_t, carried = pass.context, new = pass.new, "encoding a refinement pass");
        let enc_t = Instant::now();
        let encoder = LtxVaeEncoder::build(&vcfg, &vweights, len as u32, h as u32, w as u32, o.base.device.as_deref());
        let latent = encoder.encode(&rgb8_to_chw(&clip.frames[start..start + len], h, w));
        drop(encoder);
        vae_secs += enc_t.elapsed().as_secs_f32();

        // `frames` is THIS pass's length, which is what a stage reads it for.
        // `width`/`height` are the resolution the refinement runs at -
        // `denoise_stage` takes those from its own `Stage`, not from here, but
        // a `GenOpts` that disagreed with the stage it accompanies would be a
        // trap for the next reader. Image conditioning is cleared: this entry
        // point's content comes from the clip, not from a still.
        let pass_opts = GenOpts { frames: len, width: out_w, height: out_h, start_frame: None, mid_frame: None, end_frame: None, ..o.base.clone() };
        let sc = StageCtx {
            // A refinement pass never denoises an audio stream (its content
            // comes from a rendered file), so these are never read - empty
            // rather than a fabricated projection.
            a_ctx_cond: &[],
            a_ctx_uncond: &[],
            dit: dit.as_ref(),
            vcfg: &vcfg,
            vweights: &vweights,
            o: &pass_opts,
            lat_t,
            in_channels,
            ctx_cond: Caption::of(&ctx_cond, &context_valid),
            ctx_uncond: Caption::of(&ctx_uncond, &uncond_valid),
            context_len,
            cancel,
            reference: None,
        };
        // The plan's own context count and what the previous pass could
        // actually supply have to be the same number - the emitted-frame
        // arithmetic is derived from the former and the freeze from the
        // latter, so a disagreement would silently shift the clip.
        if carried_latent.as_ref().map(|(_, n)| *n).unwrap_or(0) != pass.context {
            return Err(format!("pass {si} plans a {}-latent-frame context but the previous pass carried {:?}", pass.context, carried_latent.as_ref().map(|(_, n)| *n)));
        }
        let refined = upscale_and_refine(
            &sc,
            Refine {
                // A refinement reads its content from a video FILE, which
                // carries no audio latent to refine and whose own sound stays
                // in the input file (see `ltxv_cli::upscale`).
                audio: None,
                upsampler: &upsampler,
                upsampler_path,
                slots: None,
                latent_chw: &latent,
                lat_t,
                lh1: h / 32,
                lw1: w / 32,
                width: out_w,
                height: out_h,
                sigmas: &sigmas,
                eta: refine_eta(&o.base),
                // The upscaled copy of these frames is overwritten by the
                // previous pass's own refined latent, so what the pass holds
                // fixed is real refined content rather than the same source
                // frames about to be refined a second, different way.
                context: carried_latent.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                seed_salt: REFINE_SEED_SALT.wrapping_mul(si as u64),
                done_before: done_before + 1,
                label: "refine",
            },
            total,
            &mut progress,
        )?
        .video_chw;
        carried_latent = Some((crate::longform::carry_tail(&refined, in_channels, lat_t, lh, lw, carried.min(lat_t)), carried.min(lat_t)));

        // Same reason [`generate_long`]'s window loop does it, and the same
        // reason [`generate`] drops the denoiser outright before its decode:
        // a pass loop holds the DiT across N decodes, and a resident weight
        // window does not fit alongside one.
        dit.release_devices();
        progress(done_before + per_pass - 1, total, "vae decode");
        let dec_t = Instant::now();
        let (pixels, got) = decode_video(&vcfg, varch, &vweights, lat_t as u32, lh as u32, lw as u32, &refined, &o.base)?;
        vae_secs += dec_t.elapsed().as_secs_f32();
        if got != len || pixels.len() != 3 * got * out_h * out_w {
            return Err(format!("pass {si} decoded to {got} frames / {} values, expected {len} / {}", pixels.len(), 3 * len * out_h * out_w));
        }
        let rgb = chw_to_rgb8(&pixels, got, out_h, out_w);
        out_frames.extend(rgb.into_iter().skip(pass.dropped_frames()));
    }
    drop(dit);
    timings.decode = vae_secs;
    timings.denoise = (work_t.elapsed().as_secs_f32() - vae_secs).max(0.0);
    timings.steps = o.refine_steps * plan.len();
    timings.tokens = plan.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    timings.forwards_per_step = o.base.guidance.forwards_per_step(o.base.audio);

    if out_frames.len() != frames {
        return Err(format!("reassembled {} frames from {} passes, expected {frames}", out_frames.len(), plan.len()));
    }
    progress(total, total, "done");
    tracing::info!(frames, width = out_w, height = out_h, passes = plan.len(), total_secs = timings.total(), "upscale done");
    Ok((Video { width: out_w as u32, height: out_h as u32, fps: clip.fps, frames: out_frames, audio: None }, timings))
}

#[cfg(test)]
mod masked_stage_tests {
    use super::*;

    /// The whole claim [`generate_masked`] makes, proven against the loop's own
    /// machinery and with NO weights: a conditioning position starts at the
    /// source latent, is announced to the model at timestep 0, passes through
    /// [`to_denoised`] as the identity, and is re-pinned to the source by
    /// [`post_process_latent`] after every step - so whatever the model says
    /// about it is discarded and it leaves the sampler bit-exactly as it
    /// entered. A generated position is left entirely alone by all three.
    ///
    /// This is the part of the path that can be gated without the 22B
    /// checkpoint or the real VAE, and it is the part where being wrong would
    /// look plausible: a run that merely STARTS a region at the source and then
    /// lets it denoise produces a picture that resembles the input, which is
    /// exactly what a broken preservation guarantee looks like.
    #[test]
    fn a_conditioned_token_survives_the_whole_step_untouched() {
        let (t, c) = (6usize, 4usize);
        let source: Vec<f32> = (0..t * c).map(|i| 0.25 * i as f32 - 1.0).collect();
        let noise: Vec<f32> = (0..t * c).map(|i| 7.0 - 0.5 * i as f32).collect();
        // Tokens 0, 2, 4 are carried; 1, 3, 5 are regenerated.
        let mask: Vec<f32> = (0..t).map(|i| f32::from(i % 2 == 0)).collect();

        let mc = crate::maskcond::apply_video_condition_by_mask(&vec![0f32; t * c], &vec![1f32; t], &source, &mask, c, 1.0);
        let latent = crate::maskcond::noised_initial_latent(&noise, &noise, &mc.clean, &mc.denoise_mask, c, 1.0);

        // The sigma the loop announces per token is `denoise_mask * sigma`
        // (`timesteps_from_mask`), so a carried token is at 0 and `to_denoised`
        // is the identity there.
        let sigma = 0.909f32;
        let timesteps: Vec<f32> = mc.denoise_mask.iter().map(|m| m * sigma).collect();
        // A velocity the model might have produced - deliberately not zero, so
        // "the token did not move" cannot be true by accident.
        let velocity: Vec<f32> = (0..t * c).map(|i| 3.0 + i as f32).collect();
        let mut denoised = to_denoised(&latent, &velocity, &timesteps, c);
        post_process_latent(&mut denoised, &Frozen { mask: &mc.denoise_mask, clean: &mc.clean, channels: c, attention: None });

        for tok in 0..t {
            let (a, b) = (tok * c, tok * c + c);
            if mask[tok] == 1.0 {
                assert_eq!(&latent[a..b], &source[a..b], "token {tok} must START at the source latent");
                assert_eq!(&denoised[a..b], &source[a..b], "token {tok} must END the step at the source latent, whatever the model said");
                assert_eq!(timesteps[tok], 0.0, "token {tok} must be announced at timestep 0");
            } else {
                assert_eq!(&latent[a..b], &noise[a..b], "token {tok} must start at pure noise");
                assert_eq!(timesteps[tok], sigma, "token {tok} must be announced at the full sigma");
                assert_ne!(&denoised[a..b], &noise[a..b], "token {tok} must actually move");
            }
        }
    }

    /// An all-conditioning mask is the identity on the picture, and
    /// [`generate_masked`] refuses it rather than spending a generation to
    /// return the input. Checked here as the algebra it rests on.
    #[test]
    fn an_all_conditioning_mask_is_the_identity() {
        let (t, c) = (4usize, 3usize);
        let source: Vec<f32> = (0..t * c).map(|i| i as f32).collect();
        let noise = vec![99f32; t * c];
        let mc = crate::maskcond::apply_video_condition_by_mask(&vec![0f32; t * c], &vec![1f32; t], &source, &vec![1f32; t], c, 1.0);
        assert_eq!(mc.denoise_mask, vec![0f32; t]);
        assert_eq!(crate::maskcond::noised_initial_latent(&noise, &noise, &mc.clean, &mc.denoise_mask, c, 1.0), source);
    }
}

// ============================================================================
// Masked video-to-video: regenerate part of a clip, carry the rest verbatim
// ============================================================================

/// Everything a masked video-to-video pass varies beyond what a generation
/// already varies.
#[derive(Clone, Debug)]
pub struct MaskedOpts {
    /// A `brain/sam2-maskseq/1` mask-sequence directory - see
    /// [`crate::maskcond::read_mask_sequence`] for the format, the polarity
    /// rule, and everything it refuses rather than reconciling. Empty when
    /// [`Self::window`] says what to regenerate instead.
    pub mask_dir: String,
    /// A retake (`ltx_pipelines.retake`): regenerate `[start, end)` seconds
    /// of the clip and hold the rest - `TemporalRegionMask` on both streams.
    pub window: Option<(f32, f32)>,
    /// Whether the video inside the region is regenerated; `false` holds
    /// every frame, so with sound on only the audio is generated
    /// (video-to-audio).
    pub regenerate_video: bool,
    /// Whether the audio inside the region is regenerated; `false` holds
    /// the source soundtrack ([`GenOpts::audio_source`]) unchanged. A clip
    /// with no source soundtrack generates its whole audio stream.
    pub regenerate_audio: bool,
    /// `VideoConditionByMask::strength`. `1.0` is the point of this path: the
    /// conditioned region leaves the sampler bit-exactly the source's own
    /// latent, because its per-token timestep is 0 and `post_process_latent`
    /// re-pins it after every step. Lower values let the carried region drift
    /// toward the prompt, which is a softer and quite different effect.
    pub strength: f32,
    /// Seed, guidance, steps, fps, `dit_config`, device placement. `frames`,
    /// `width` and `height` are IGNORED - they come from the input clip,
    /// which is the whole point of this entry point.
    pub base: GenOpts,
}

impl Default for MaskedOpts {
    fn default() -> MaskedOpts {
        MaskedOpts { mask_dir: String::new(), window: None, regenerate_video: true, regenerate_audio: true, strength: 1.0, base: GenOpts::default() }
    }
}

/// Regenerate the masked-out region of an existing clip and carry everything
/// else through untouched.
///
/// This is [`generate`]'s denoise loop with the initial latent and the
/// per-token denoise mask coming from a source clip plus a mask sequence
/// instead of from pure noise: VAE-encode the clip, reduce the pixel-space
/// mask to the latent grid, apply
/// `ltx_core.conditioning.types.mask_cond.VideoConditionByMask`, denoise,
/// VAE-decode.
///
/// **What it buys over the IC-LoRA path ([`crate::refcond`]).** An IC-LoRA
/// appends a reference as extra tokens and depends on an ADAPTER having been
/// trained to attend across them - and LTX-2.5 has exactly one published
/// IC-LoRA, a pixel spatial upscaler, so that route is closed on this
/// checkpoint. Masked conditioning needs no adapter: `clean_latent` and
/// `denoise_mask` are inputs the base model already honours. It is also
/// stronger where it applies - a conditioned token at `strength = 1.0` is
/// preserved BIT-exactly rather than structurally.
///
/// **One stage, at the clip's own resolution.** There is nothing to upscale
/// here: the content already exists at the size it is wanted, so the
/// two-stage half-resolution-then-detail shape [`generate`] uses would round
/// trip the carried region through a downscale it does not need. Past
/// [`SINGLE_STAGE_MAX_TOKENS`] this warns rather than refusing - the ceiling
/// is a quality observation about generating a clip from noise, and most of
/// this sequence is not being generated.
///
/// **No audio.** A masked pass conditions a video latent; the source's own
/// sound track stays in the source file, and re-attaching it is one ffmpeg
/// mux (the same split [`upscale`] documents).
///
/// `prompt` describes the WHOLE frame, not just the replacement: the DiT
/// denoises the generated region in the context of the carried one through
/// self-attention, and answers to the text context while doing it.
#[tracing::instrument(level = "info", name = "v2v", skip_all, fields(frames = clip.frames.len(), width = clip.width, height = clip.height, strength = o.strength, seed = o.base.seed))]
pub fn generate_masked(paths: &Paths, prompt: &str, clip: &Video, o: &MaskedOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    if o.base.reference.is_some() {
        return Err("--reference-video is applied by a single-window generation (`t2v` within one denoising window); a masked video-to-video pass does not route a reference".into());
    }
    let (w, h) = (clip.width as usize, clip.height as usize);
    let frames = clip.frames.len();
    if w == 0 || h == 0 || frames == 0 {
        return Err("the input clip is empty".into());
    }
    if !w.is_multiple_of(32) || !h.is_multiple_of(32) {
        return Err(format!("{w}x{h} is not a multiple of 32 (the VAE's spatial stride) - the input clip has to be VAE-representable before part of it can be regenerated"));
    }
    if let Some(bad) = clip.frames.iter().position(|f| f.len() != w * h * 3) {
        return Err(format!("frame {bad} is {} bytes, expected {} for {w}x{h} RGB8", clip.frames[bad].len(), w * h * 3));
    }
    if !(0.0..=1.0).contains(&o.strength) {
        return Err(format!("strength {} is outside [0, 1]", o.strength));
    }
    let vcfg = LtxVaeConfig::conv25();
    let lat_t = vcfg.latent_frames(frames as u32).ok_or_else(|| format!("{frames} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)"))? as usize;
    let (lh, lw) = (h / 32, w / 32);
    let t = lat_t * lh * lw;

    let dit_cfg = dit_config_from_name(&o.base.dit_config).inspect_err(|e| tracing::error!(dit_config = %o.base.dit_config, error = %e, "unknown DiT config"))?;
    if dit_cfg.in_channels != vcfg.latent_channels {
        return Err(format!("ltxv dit-config {:?} has in_channels {} but the VAE latent width is {}", o.base.dit_config, dit_cfg.in_channels, vcfg.latent_channels));
    }
    let in_channels = dit_cfg.in_channels as usize;

    // The mask is read and reduced BEFORE any weight is touched: a mask
    // sequence that does not match the clip, or whose polarity is not
    // declared, should cost milliseconds rather than a 22B model build.
    check_audio_request(paths, &o.base)?;
    // `1` on a held token, `0` on a regenerated one - the mask sequence's own
    // polarity, the complement of the denoise mask.
    let mask: Vec<f32> = match (o.mask_dir.is_empty(), o.window) {
        (false, None) => crate::maskcond::read_mask_sequence(std::path::Path::new(&o.mask_dir), frames, w, h, lat_t, lh, lw)?,
        (true, Some((start, end))) => {
            if start >= end {
                return Err(format!("the retake window starts at {start} s, at or after its end {end} s"));
            }
            let region = crate::maskcond::temporal_region(&real_pixel_positions(lat_t, lh, lw, clip.fps as f64), t, start, end);
            region.iter().map(|&r| if o.regenerate_video { 1.0 - r } else { 1.0 }).collect()
        }
        _ => return Err("a masked pass regenerates either what a mask sequence marks or a time window, exactly one of the two".into()),
    };
    // The audio stream of a sound-on pass: the source soundtrack (when there
    // is one) encoded and held outside the region, regenerated inside it.
    let ta = crate::audio::latent_frames(frames, clip.fps);
    let audio_frozen: Option<(Vec<f32>, Vec<f32>)> = match (o.base.audio, &o.base.audio_source) {
        (false, _) => None,
        (true, None) if !o.regenerate_audio => return Err("holding the audio needs a source soundtrack (--audio-source) to hold".into()),
        (true, None) => None,
        (true, Some(source)) => {
            let path = paths.audio_vae.as_deref().expect("an audio request is refused above without an audio VAE path");
            let (_, vae_raw): (Vec<_>, Vec<_>) = checkpoint::safetensors::read(path)?.into_iter().partition(|t| t.name.starts_with("vocoder."));
            let vae_w = crate::import::import_audio_vae(vae_raw, &crate::audio_vae::AudioVaeConfig::ltx25())?;
            let clean = crate::audio::encode(&vae_w, source, ta, o.base.device.as_deref())?;
            let region = match o.window {
                Some((start, end)) => crate::maskcond::temporal_region(&crate::audio::positions(ta), ta, start, end),
                None => vec![1.0; ta],
            };
            let denoise: Vec<f32> = region.iter().map(|&r| if o.regenerate_audio { r } else { 0.0 }).collect();
            Some((denoise, clean))
        }
    };
    let audio_generates = o.base.audio && audio_frozen.as_ref().is_none_or(|(m, _)| m.iter().any(|&m| m > 0.0));
    let generated = mask.iter().filter(|&&m| m < 1.0).count();
    if generated == 0 && !audio_generates {
        return Err(format!(
            "every one of the {t} latent tokens is a conditioning position, so this run would return the input clip unchanged - the mask sequence in {} marks nothing to regenerate (check its declared polarity)",
            o.mask_dir
        ));
    }
    tracing::info!(latent_tokens = t, touched = generated, "mask reduced to the latent grid");

    let is_real_distilled = o.base.dit_config == "ltx25_22b";
    let sigmas: Vec<f64> = if is_real_distilled {
        LTX2_DISTILLED_SIGMAS.iter().map(|&s| s as f64).collect()
    } else {
        ltx2_sigmas(t, o.base.steps, o.base.base_shift, o.base.max_shift, o.base.stretch, o.base.terminal)
    };
    if t > SINGLE_STAGE_MAX_TOKENS {
        tracing::warn!(tokens = t, ceiling = SINGLE_STAGE_MAX_TOKENS, "this pass runs as ONE stage at the clip's own resolution, past the token count the distilled schedule was tuned to generate from noise; the carried region is unaffected");
    }
    // Phases: build, encode, every step, decode.
    let total = sigmas.len() as u32 + 2;
    let mut timings = Timings::default();
    let place = o.base.devices.resolve(o.base.device.as_deref());

    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, &o.base, &o.base.adapters, place)?;
    timings.build_dit = build_t.elapsed().as_secs_f32();
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let text = build_context(paths, prompt, dit_cfg, &o.base, place, &mut timings.text_encode)?;

    let crate::import::VideoVae { arch: varch, weights: vweights } = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
    progress(1, total, "vae encode");
    let enc_t = Instant::now();
    let source_chw = {
        let encoder = LtxVaeEncoder::build(&vcfg, &vweights, frames as u32, h as u32, w as u32, o.base.device.as_deref());
        encoder.encode(&rgb8_to_chw(&clip.frames, h, w))
    };
    let mut vae_secs = enc_t.elapsed().as_secs_f32();
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }

    // `frames`/`width`/`height` come from the clip; image conditioning and
    // audio are cleared, because this entry point's content comes from the
    // source and its sound stays in the source file.
    let pass_opts = GenOpts { frames, width: w, height: h, start_frame: None, mid_frame: None, end_frame: None, ..o.base.clone() };
    let audio_noise = o.base.audio.then(|| seeded_noise(ta * crate::audio::TOKEN_DIM as usize, o.base.seed ^ AUDIO_SEED_SALT));
    let sc = StageCtx {
        a_ctx_cond: text.a_cond.as_deref().unwrap_or_default(),
        a_ctx_uncond: text.a_uncond.as_deref().unwrap_or_default(),
        dit: dit.as_ref(),
        vcfg: &vcfg,
        vweights: &vweights,
        o: &pass_opts,
        lat_t,
        in_channels,
        ctx_cond: Caption::of(&text.cond, &text.valid),
        ctx_uncond: Caption::of(&text.uncond, &text.uncond_valid),
        context_len: text.len,
        cancel,
        reference: None,
    };
    let work_t = Instant::now();
    let stage_out = denoise_stage(
        &sc,
        Stage {
            mask: Some(MaskedSource { chw: &source_chw, mask: &mask, strength: o.strength }),
            width: w,
            height: h,
            sigmas: &sigmas,
            eta: o.base.eta,
            seed_chw: None,
            context: None,
            seed_salt: 0,
            done_before: 2,
            label: "masked",
            audio: audio_noise.map(|latent| AudioStage { latent, context: Vec::new(), frozen: audio_frozen.clone() }),
            slots: None,
        },
        total,
        &mut progress,
    )?;
    timings.denoise = work_t.elapsed().as_secs_f32();
    // Same reason [`generate`] does it before its own decode: the decoder
    // opens its own device context and a resident weight window does not fit
    // alongside one.
    drop(dit);

    progress(total - 1, total, "vae decode");
    let dec_t = Instant::now();
    let (pixels, got) = decode_video(&vcfg, varch, &vweights, lat_t as u32, lh as u32, lw as u32, &stage_out.video_chw, &o.base)?;
    vae_secs += dec_t.elapsed().as_secs_f32();
    if got != frames || pixels.len() != 3 * got * h * w {
        return Err(format!("decoded to {got} frames / {} values, expected {frames} / {}", pixels.len(), 3 * frames * h * w));
    }
    timings.decode = vae_secs;
    timings.steps = sigmas.len() - 1;
    timings.tokens = t;
    timings.forwards_per_step = o.base.guidance.forwards_per_step(o.base.audio);

    // The sound: the source soundtrack when the pass held all of it, else
    // the denoised stream decoded.
    let audio = match stage_out.audio {
        Some(_) if !audio_generates => o.base.audio_source.clone(),
        Some(latent) => {
            let path = paths.audio_vae.as_deref().expect("an audio request is refused above without an audio VAE path");
            let mut sound = decode_audio_latent(path, &latent, ta, o.base.device.as_deref())?;
            sound.pad_to_seconds(frames as f32 / clip.fps as f32);
            Some(sound)
        }
        None => None,
    };
    progress(total, total, "done");
    tracing::info!(frames, width = w, height = h, total_secs = timings.total(), audio = audio.is_some(), "masked video-to-video done");
    Ok((Video { width: w as u32, height: h as u32, fps: clip.fps, frames: chw_to_rgb8(&pixels, got, h, w), audio }, timings))
}

// ============================================================================
// Extension: more of an existing clip
// ============================================================================

/// What [`extend`] adds to a clip.
#[derive(Clone, Debug)]
pub struct ExtendOpts {
    /// Frames to add after the clip's last one. Rounded up to a whole latent
    /// frame ([`VAE_TEMPORAL_SCALE`]), which is the unit a continuation is
    /// generated in.
    pub frames: usize,
    /// How much of the clip's end the continuation is conditioned on, in
    /// pixel frames - the reference extension recipe's prefix
    /// ([`crate::longform::CONTEXT_FRAMES`]), shortened to the clip when the
    /// clip is shorter and snapped down onto the `8k + 1` grid.
    pub context_frames: usize,
    /// The generation settings. `audio` extends the soundtrack too, which
    /// needs the clip's own soundtrack as `audio_source`.
    pub base: GenOpts,
}

impl Default for ExtendOpts {
    fn default() -> ExtendOpts {
        ExtendOpts { frames: 8 * VAE_TEMPORAL_SCALE, context_frames: crate::longform::CONTEXT_FRAMES, base: GenOpts::default() }
    }
}

/// The clip [`extend`] regenerates: how many of the source's last frames it
/// holds, how many it adds, and the time window it generates.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ExtensionPlan {
    context: usize,
    added: usize,
    window: (f32, f32),
}

fn extension_plan(clip_frames: usize, o: &ExtendOpts, fps: usize) -> Result<ExtensionPlan, String> {
    if clip_frames == 0 {
        return Err("the input clip is empty".into());
    }
    if o.frames == 0 {
        return Err("an extension adds at least one frame".into());
    }
    if o.context_frames == 0 || fps == 0 {
        return Err("an extension is conditioned on at least one frame of the clip, at a nonzero frame rate".into());
    }
    let step = VAE_TEMPORAL_SCALE;
    let context = (o.context_frames.min(clip_frames) - 1) / step * step + 1;
    let added = o.frames.div_ceil(step) * step;
    Ok(ExtensionPlan { context, added, window: (context as f32 / fps as f32, (context + added) as f32 / fps as f32) })
}

/// Continue `clip` for `o.frames` more frames (`video_extend_lora`'s
/// forward extension): the clip's last `context` frames are held as a clean
/// prefix while the frames after them denoise from noise, guided by
/// `prompt`.
///
/// That is a retake ([`generate_masked`]) of a time window lying past the
/// clip's end: the source is the clip's tail followed by placeholder frames
/// the window regenerates entirely, so the held prefix reaches the model
/// exactly as the reference's `prefix` condition puts it - frozen tokens at
/// timestep 0 at the front of the sequence. The returned clip is the
/// original, untouched, followed by the new frames; with `audio`, the
/// original soundtrack followed by the sound generated for them.
pub fn extend(paths: &Paths, prompt: &str, clip: &Video, o: &ExtendOpts, cancel: &capability::CancelToken, progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    let plan = extension_plan(clip.frames.len(), o, clip.fps)?;
    let tail_start = clip.frames.len() - plan.context;
    let mut source = Video { width: clip.width, height: clip.height, fps: clip.fps, frames: clip.frames[tail_start..].to_vec(), audio: None };
    let last = source.frames.last().expect("the context holds at least one frame").clone();
    source.frames.resize(plan.context + plan.added, last);
    let clip_seconds = clip.frames.len() as f32 / clip.fps as f32;
    let tail_seconds = tail_start as f32 / clip.fps as f32;
    let audio_source = match (o.base.audio, &o.base.audio_source) {
        (false, _) => None,
        (true, None) => return Err("extending the soundtrack too needs the clip's own soundtrack as the audio source".into()),
        (true, Some(sound)) => {
            let mut held = sound.clone();
            held.channels.iter_mut().for_each(|c| {
                c.truncate((clip_seconds * sound.sample_rate as f32).round() as usize);
                c.drain(..((tail_seconds * sound.sample_rate as f32).round() as usize).min(c.len()));
            });
            // Silence over the added span: the window regenerates it, so only
            // its length matters.
            held.channels.iter_mut().for_each(|c| c.resize(((plan.window.1) * sound.sample_rate as f32).round() as usize, 0.0));
            Some(held)
        }
    };
    let masked = MaskedOpts {
        mask_dir: String::new(),
        window: Some(plan.window),
        regenerate_video: true,
        regenerate_audio: true,
        strength: 1.0,
        base: GenOpts { audio_source, ..o.base.clone() },
    };
    tracing::info!(clip_frames = clip.frames.len(), context = plan.context, added = plan.added, window_start = plan.window.0, window_end = plan.window.1, "extending the clip");
    let (generated, timings) = generate_masked(paths, prompt, &source, &masked, cancel, progress)?;
    let mut out = Video { width: clip.width, height: clip.height, fps: clip.fps, frames: clip.frames.clone(), audio: None };
    out.frames.extend(generated.frames.into_iter().skip(plan.context));
    if let (Some(original), Some(mut new)) = (o.base.audio_source.as_ref(), generated.audio) {
        // The generated sound covers the held tail too; keep it only from the
        // clip's own end on, after the original soundtrack.
        let mut sound = if original.sample_rate == new.sample_rate {
            original.clone()
        } else {
            crate::audio::AudioClip { channels: original.channels.iter().map(|c| audio::resample::sinc_hann(c, original.sample_rate, new.sample_rate)).collect(), sample_rate: new.sample_rate }
        };
        let rate = new.sample_rate as f32;
        // A mono original is spread over the generated channels; otherwise
        // the two share the channels both have.
        if let [mono] = sound.channels.as_slice() {
            sound.channels = vec![mono.clone(); new.channels.len()];
        }
        let channels = sound.channels.len().min(new.channels.len());
        sound.channels.truncate(channels);
        sound.pad_to_seconds(clip_seconds);
        let skip = ((clip_seconds - tail_seconds) * rate).round() as usize;
        for (held, fresh) in sound.channels.iter_mut().zip(&mut new.channels) {
            held.truncate((clip_seconds * rate).round() as usize);
            held.extend(fresh.drain(skip.min(fresh.len())..));
        }
        out.audio = Some(sound);
    }
    Ok((out, timings))
}

// ============================================================================
// Long-form generation: several denoising windows, one rolling latent context
// ============================================================================

/// Everything [`generate_long`] varies beyond what a generation already
/// varies.
#[derive(Clone, Debug)]
pub struct LongOpts {
    /// Clean latent frames carried across each window boundary - see
    /// [`crate::longform::CONTEXT_LATENT_FRAMES`] for where the default comes
    /// from and for the hypothesis (the VAE's temporal receptive field) that
    /// cannot set it.
    pub context_latent_frames: usize,
    /// Per-window video-token ceiling, [`crate::longform::LONGFORM_MAX_TOKENS`]
    /// by default.  Governs stage 1's half-resolution plan.
    pub max_window_tokens: usize,
    /// Per-pass video-token ceiling for stage 2's refinement, mirroring
    /// [`crate::pipeline::REFINE_MAX_TOKENS`].  Defaults to
    /// [`crate::pipeline::REFINE_MAX_TOKENS`].
    pub max_refine_tokens: usize,
    /// Run a video-only clip stage-major (stage 1 across the whole clip at
    /// half resolution, then stage 2 refining slices of it) when its stage-2
    /// plan splits; `false` keeps the window-major loop. On by default.
    pub stage_major: bool,
    /// `frames` is the WHOLE clip's length, not one window's.
    pub base: GenOpts,
}

impl Default for LongOpts {
    fn default() -> LongOpts {
        LongOpts {
            context_latent_frames: crate::longform::CONTEXT_LATENT_FRAMES,
            max_window_tokens: crate::longform::LONGFORM_MAX_TOKENS,
            max_refine_tokens: REFINE_MAX_TOKENS,
            stage_major: true,
            base: GenOpts::default(),
        }
    }
}

/// Two independent window plans for a stage-major long-form generation:
/// stage 1 at half resolution (where motion is decided) and stage 2 at full
/// resolution (where refinement detail is carried).
///
/// Stage 1's plan determines the clip's motion structure; stage 2's plan
/// fragments only as much as memory demands, with no quality cost because the
/// motion is already fixed by stage 1.
#[derive(Clone, Debug)]
pub struct TwoStageLongPlan {
    /// Half-resolution window plan (where motion is decided).
    pub stage1: Vec<crate::longform::Window>,
    /// Full-resolution window plan (refinement detail).
    pub stage2: Vec<crate::longform::Window>,
    /// Latent frames in the global half-resolution latent buffer (sum of
    /// stage1 windows' `new` fields).
    pub lat_t: usize,
    /// Stage 2's context: the number of full-res latent frames carried across
    /// each stage-2 seam (may differ from stage 1's context if the grids
    /// differ).
    pub stage2_context: usize,
}

/// Build two independent window plans for a stage-major long-form generation.
///
/// Calls [`crate::longform::window_plan_aligned`] twice, once at the
/// half-resolution grid (stage 1) and once at the full-resolution grid
/// (stage 2).  Stage 2's context is capped to what fits under
/// `max_stage2_tokens` on the denser grid.
///
/// Returns `Err` when either grid cannot express the clip.
pub fn two_stage_long_plan(
    frames: usize,
    lh1: usize,
    lw1: usize,
    lh: usize,
    lw: usize,
    context: usize,
    max_stage1_tokens: usize,
    max_stage2_tokens: usize,
    align: usize,
) -> Result<TwoStageLongPlan, String> {
    let stage1 = crate::longform::window_plan_aligned(frames, lh1, lw1, context, max_stage1_tokens, align)?;
    let stage2_ctx = crate::longform::fitted_context(lh, lw, context, max_stage2_tokens)?;
    let stage2 = crate::longform::window_plan_aligned(frames, lh, lw, stage2_ctx, max_stage2_tokens, align)?;
    let lat_t = stage1.iter().map(|w| w.new).sum();
    Ok(TwoStageLongPlan { stage1, stage2, lat_t, stage2_context: stage2_ctx })
}

/// Build per-window generation options from a clip-global base, routing
/// `start_frame`, `end_frame`, and `mid_frame` to the window that owns each,
/// and handing every window the anchors still ahead of it as future guides.
///
/// - `start_frame`: window 0 only (pixel frame 0 only exists there).
/// - `end_frame`: last window only (its `frames = w.decoded_frames()` already
///   re-expresses the anchor with no arithmetic).
/// - `mid_frame` / `mid_frame_at`: routed via [`crate::longform::route_mid_anchor`]
///   to the window whose decoded range contains the pixel frame, and re-expressed
///   as the local pixel frame within that window.
/// - Everything still AHEAD of a window - a mid anchor a later window owns,
///   the end anchor on any window but the last - becomes that window's
///   [`GenOpts::future_anchors`] entry: an appended guide at its local pixel
///   position, so the window bends toward an anchor it cannot reach inside its
///   own span. An anchor a window has already emitted is its past; the carried
///   context carries its effect and nothing re-conditions it.
///
/// Returns `Err` when the base options carry a mid anchor that doesn't land
/// inside the clip (out-of-bounds `mid_frame_at`), since no window could
/// honour it as either owner or future.
pub fn window_gen_opts(
    base: &GenOpts,
    plan: &[crate::longform::Window],
    wi: usize,
) -> Result<GenOpts, String> {
    let w = &plan[wi];
    let start_frame = if wi == 0 { base.start_frame.clone() } else { None };
    let end_frame = if wi == plan.len() - 1 { base.end_frame.clone() } else { None };
    // The mid anchor's clip-global instant, resolved and validated against the
    // FULL clip: an explicit `mid_frame_at` outside it has no owning window and
    // no future for any window to guide toward.
    let mid_global = base.mid_frame.as_ref().map(|_| mid_anchor_frame(base.frames, base.mid_frame_at)).transpose()?;
    // Which window owns the instant - always some window, because the decoded
    // ranges tile the clip (see `crate::longform`'s `route_mid_anchor`).
    let owner = mid_global.and_then(|m| crate::longform::route_mid_anchor(plan, m).map(|(owning_wi, _)| (m, owning_wi)));
    let (mid_frame, mid_frame_at) = match owner {
        Some((m, owning_wi)) if owning_wi == wi => {
            // The global pixel frame falls inside this window's decoded range
            // and this window emits it: re-express as the local pixel frame.
            (base.mid_frame.clone(), Some(m - w.source_first_frame()))
        }
        _ => (None, None),
    };
    let decoded_end = w.source_first_frame() + w.decoded_frames();
    let mut future_anchors = Vec::new();
    if let (Some(ref path), Some((m, _))) = (&base.mid_frame, owner) {
        // The owner is the only window whose emitted span contains the
        // instant; every window whose span ends before it sees it in future.
        if m >= decoded_end {
            future_anchors.push((m - w.source_first_frame(), path.clone()));
        }
    }
    if let Some(ref path) = base.end_frame {
        if wi != plan.len() - 1 && base.frames > decoded_end {
            future_anchors.push((base.frames - 1 - w.source_first_frame(), path.clone()));
        }
    }
    Ok(GenOpts {
        frames: w.decoded_frames(),
        start_frame,
        end_frame,
        mid_frame,
        mid_frame_at,
        future_anchors,
        ..base.clone()
    })
}

#[cfg(test)]
mod window_gen_opts_tests {
    use super::*;

    /// A 41-frame clip as two windows: the head owns decoded [0, 33), the
    /// continuation emits [33, 41) on top of a 2-latent-frame context whose
    /// pixels it re-decodes but never re-emits.
    fn two_window_plan() -> Vec<crate::longform::Window> {
        vec![
            crate::longform::Window { context: 0, new: 5, first_frame: 0 },
            crate::longform::Window { context: 2, new: 1, first_frame: 33 },
        ]
    }

    fn base_with_anchors(at: Option<usize>) -> GenOpts {
        GenOpts {
            frames: 41,
            start_frame: Some("start.png".into()),
            mid_frame: Some("mid.png".into()),
            mid_frame_at: at,
            end_frame: Some("end.png".into()),
            ..GenOpts::default()
        }
    }

    /// An anchor whose instant lies beyond a window's own decoded span rides
    /// along as a future guide at its local pixel position - the bending
    /// force a window needs to approach an anchor it cannot reach in its own
    /// span. The owning window conditions it in-window instead, and the last
    /// window, with everything behind or inside it, carries nothing.
    #[test]
    fn an_anchor_beyond_a_windows_span_rides_along_as_a_future_guide() {
        let plan = two_window_plan();
        let b = base_with_anchors(Some(36)); // the mid anchor's instant is window 1's to own

        let w0 = window_gen_opts(&b, &plan, 0).expect("a clip-global mid anchor inside the clip routes");
        assert_eq!(w0.start_frame.as_deref(), Some("start.png"));
        assert_eq!(w0.mid_frame, None, "the mid anchor is not window 0's to own in-window");
        assert_eq!(w0.end_frame, None, "the end anchor belongs to the last window in-window");
        assert_eq!(
            w0.future_anchors,
            vec![(36, "mid.png".to_string()), (40, "end.png".to_string())],
            "both later anchors, in timeline order, at local pixel positions beyond window 0's 33-frame span"
        );

        let w1 = window_gen_opts(&b, &plan, 1).expect("a clip-global mid anchor inside the clip routes");
        assert_eq!(w1.start_frame, None, "the start still conditioned window 0 and is window 1's past");
        assert_eq!(w1.mid_frame.as_deref(), Some("mid.png"), "the mid anchor's instant is inside window 1's emitted span");
        assert_eq!(w1.mid_frame_at, Some(12), "pixel 36 minus window 1's source_first_frame 24");
        assert_eq!(w1.end_frame.as_deref(), Some("end.png"), "the last window owns the end anchor in-window");
        assert!(w1.future_anchors.is_empty(), "nothing lies beyond the last window");
    }

    /// An anchor a window has already emitted is its PAST: the carried
    /// context carries its effect, and re-conditioning it would pin two
    /// different instants of the window's own coordinate frame to the same
    /// still.
    #[test]
    fn an_anchor_a_window_has_already_emitted_is_not_re_conditioned() {
        let plan = two_window_plan();
        let b = base_with_anchors(Some(20)); // owned by window 0

        let w0 = window_gen_opts(&b, &plan, 0).expect("routes");
        assert_eq!(w0.mid_frame.as_deref(), Some("mid.png"));
        assert_eq!(w0.mid_frame_at, Some(20));
        assert_eq!(w0.future_anchors, vec![(40, "end.png".to_string())], "only the end anchor is still ahead of window 0");

        let w1 = window_gen_opts(&b, &plan, 1).expect("routes");
        assert_eq!(w1.mid_frame, None, "the mid anchor's instant is in window 1's past");
        assert!(w1.future_anchors.is_empty());
    }

    /// Without an explicit position the mid anchor belongs at the middle of
    /// the FULL clip - routed to the window that owns that instant, and seen
    /// by no other window as anything but past or future.
    #[test]
    fn the_default_mid_anchor_is_clip_global_and_reaches_no_window_but_its_owner_in_window() {
        let plan = two_window_plan();
        let b = base_with_anchors(None); // the default (41-1)/2 = 20 is window 0's to own

        let w0 = window_gen_opts(&b, &plan, 0).expect("routes");
        assert_eq!(w0.mid_frame.as_deref(), Some("mid.png"));
        assert_eq!(w0.mid_frame_at, Some(20), "the clip's own middle, not this window's");
        assert_eq!(w0.future_anchors, vec![(40, "end.png".to_string())]);
    }

    /// A mid anchor outside the clip is refused where it is routed - no
    /// window's decoded range contains it and no future guide can name an
    /// instant the clip never reaches.
    #[test]
    fn a_mid_anchor_outside_the_clip_is_refused_where_it_is_routed() {
        let plan = two_window_plan();
        let b = base_with_anchors(Some(50));
        let err = window_gen_opts(&b, &plan, 0).expect_err("beyond the clip");
        assert!(err.contains("not INSIDE"), "the refusal names the shape problem: {err}");
    }
}

/// Text to video of arbitrary length: one clip built out of several
/// consecutive denoising windows, each conditioned on the previous window's
/// own last latent frames.
///
/// **A request that fits one window is handed straight to [`generate`]**, so
/// every shape this crate already generated keeps its exact behaviour, bit
/// for bit, and this entry point costs nothing below the ceiling.
///
/// # Window-major path (single-stage and audio-visual)
///
/// Above it ([`crate::longform::window_plan`]), the loop is: generate window
/// 0 normally; slice its last `context_latent_frames` latent frames out of
/// the final latent with [`crate::longform::carry_tail`] BEFORE anything is
/// decoded; freeze them at the head of window 1's sequence
/// ([`LatentContext`]) so only the new frames get a denoising schedule;
/// repeat. The rolling state is two latent slabs of at most
/// `context_latent_frames` frames each (one per stage of a two-stage window),
/// so a ten-minute clip costs the same host memory as a ten-second one.
///
/// **What crosses a seam never becomes a picture on the way.** Chaining by
/// decoding a window, taking its last RGB frame and re-encoding it as the
/// next window's `--start-frame` is continuous in position and discontinuous
/// in velocity: one frame cannot say what was moving or how fast, so motion
/// is re-invented at every boundary. Here the carried tensor is the denoised
/// latent itself.
///
/// **On an audio-visual request the AUDIO latent crosses every seam too**, by
/// the same mechanism and in the same place: the previous window's own last
/// audio tokens ([`crate::audio::carry_tail`]) frozen at sigma 0 at the head
/// of the next window's audio sequence, with only the new tokens on a
/// denoising schedule. The windows contribute tokens to ONE latent, decoded
/// once when the loop ends - never per window and butted together, which
/// would put a causal-VAE boundary and a waveform join at every seam.
///
/// What the two streams cannot share is a time resolution, so an audio-visual
/// plan is a DIFFERENT window split: [`crate::longform::window_plan_aligned`]
/// places seams only where both grids have a boundary, and
/// [`crate::audio::audio_plan`] re-derives the token layout from the finished
/// plan and refuses rather than rounding. `crate::audio`'s module doc carries
/// the rule and why a rounded seam is the failure mode that matters.
///
/// **Scope (window-major).** This path is what serves audio-visual clips,
/// clips whose stage-2 refinement plan fits one window (see the stage-major
/// dispatch below), and anything with [`LongOpts::stage_major`]
/// off. Clip-global anchors route through [`window_gen_opts`] exactly as
/// they do on the stage-major path: `--start-frame` conditions window 0, the
/// end and mid stills condition the windows that own their instants, and every
/// window also receives the anchors still ahead of it as future guides.
///
/// # Stage-major path (two-stage, video-only, default)
///
/// A video-only multi-window request goes to [`run_stage_major`] when its
/// STAGE-2 refinement plan - the full-res grid under `max_refine_tokens`,
/// not the dispatch plan under `max_window_tokens` - splits into at least two
/// windows. Anything else falls through to the window-major loop above, which
/// has always served exactly those shapes. There the generation is
/// restructured to run stage 1 at half resolution for the whole clip, then
/// stage 2 at full resolution:
///
/// 1. **Stage 1** builds a global half-resolution latent using
///    [`two_stage_long_plan`]'s independent half-res window plan.  This lets
///    stage 1 decide the clip's motion in as few, as global a pass as
///    possible, with fragments only when memory demands it.
///
/// 2. **Stage 2** refines each window from a slice of stage 1's output, using
///    the full-res window plan.  The motion is already fixed by stage 1, so
///    stage 2 can fragment freely with no quality cost.
///
/// Clip-global anchors (`--start-frame`, `--end-frame`, `--mid-frame`) are
/// routed to their owning window via [`window_gen_opts`], and every window
/// receives the anchors still ahead of it as future guides; a mid anchor
/// outside the clip is refused rather than silently dropped.  Audio-visual
/// clips keep the window-major path (Phase F defers that design).
///
/// Long-form generation is otherwise the ordinary generation path - same
/// schedule, same CFG fold, same two-stage decision per window, same VAE.
#[tracing::instrument(level = "info", name = "generate_long", skip_all, fields(frames = o.base.frames, width = o.base.width, height = o.base.height, seed = o.base.seed, context = o.context_latent_frames))]
pub fn generate_long(paths: &Paths, prompt: &str, o: &LongOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(o.base.frames as u32).is_none() {
        return Err(format!("{} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)", o.base.frames));
    }
    if !o.base.width.is_multiple_of(32) || !o.base.height.is_multiple_of(32) {
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", o.base.width, o.base.height));
    }
    let (lh, lw) = (o.base.height / 32, o.base.width / 32);
    // An audio-visual clip's seams have to land on whole audio tokens, which
    // constrains how far a window may advance - see `crate::audio`'s module
    // doc. A silent clip has one stream and no such constraint, and plans
    // exactly as it always did.
    let align = if o.base.audio { crate::audio::window_latent_frame_quantum(o.base.fps) } else { 1 };
    let plan = crate::longform::window_plan_aligned(o.base.frames, lh, lw, o.context_latent_frames, o.max_window_tokens, align)
        .map_err(|e| if o.base.audio { format!("{e}. {}", crate::audio::quantum_note(o.base.fps)) } else { e })?;
    if plan.len() == 1 {
        // Byte-for-byte the path this request already took: one window is a
        // generation, and this entry point must not become a second way of
        // spelling it.
        return generate(paths, prompt, &o.base, cancel, progress);
    }
    if o.base.reference.is_some() {
        return Err("--reference-video is applied by a single-window generation; a clip this long is generated as several windows, which do not route a reference yet".into());
    }
    // Stage-major long-form: when the kill switch is enabled (default) and the
    // clip is video-only, route to the stage-major loop which runs stage 1 at
    // half resolution for the whole clip, then stage 2 at full resolution.
    // Audio-visual clips keep today's window-major loop (Phase F defers that).
    let stage_major_enabled = o.stage_major && !o.base.audio;
    // The stage-major loop needs its OWN stage-2 plan to split: it refines the
    // half-res motion across >= 2 full-resolution windows, and a clip whose
    // stage-2 plan fits one window is exactly what the window-major loop below
    // has always served. The dispatch plan above is cut at `max_window_tokens`
    // on the full-res grid, while stage 2 plans at `max_refine_tokens` - close
    // but not the same ceiling, so "the dispatch plan splits" does not imply
    // "the stage-2 plan splits" - and dispatching on the wrong one made every
    // such request die in run_stage_major's own refusal instead of generating.
    if stage_major_enabled
        && two_stage_long_plan(o.base.frames, lh / 2, lw / 2, lh, lw, o.context_latent_frames, o.max_window_tokens, o.max_refine_tokens, align)
            .map(|t| t.stage2.len() >= 2)
            .unwrap_or(false)
    {
        return run_stage_major(paths, prompt, o, cancel, progress);
    }
    check_audio_request(paths, &o.base)?;
    if o.base.steps == 0 {
        return Err("--steps must be at least 1".into());
    }

    let dit_cfg = dit_config_from_name(&o.base.dit_config)?;
    if dit_cfg.in_channels != vcfg.latent_channels {
        return Err(format!("ltxv dit-config {:?} has in_channels {} but the VAE latent width is {}", o.base.dit_config, dit_cfg.in_channels, vcfg.latent_channels));
    }
    let in_channels = dit_cfg.in_channels as usize;
    let is_real_distilled = o.base.dit_config == "ltx25_22b";
    let context = o.context_latent_frames;

    // Whether this plan runs the reference's two-stage shape is decided ONCE,
    // from its largest window, and then applies to every window.
    //
    // Not per window, deliberately. Two windows of one clip built different
    // ways would change the clip's construction half way through; and a
    // two-stage window's stage 1 carries the previous window's HALF-resolution
    // latent, which a single-stage window never produces - so a mixed plan
    // would silently hand a stage 1 either nothing or a stale tail from two
    // windows back. Resolved before the first forward so a plan that needs
    // the upscaler and cannot reach it fails now rather than in an hour.
    let widest = plan.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    let two_stage = should_two_stage(widest, o.base.width, o.base.height, is_real_distilled, o.base.runtime.two_stage);
    if two_stage && paths.spatial_upsampler.is_none() {
        return Err(format!(
            "the widest window of this {}x{} plan is {widest} video tokens, past the {SINGLE_STAGE_MAX_TOKENS}-token ceiling the distilled schedule holds in ONE stage, so it needs the reference's two-stage path and its spatial x2 latent upscaler: put it in the model store or pass --spatial-upsampler <path>",
            o.base.width, o.base.height
        ));
    }
    if two_stage && (!o.base.width.is_multiple_of(64) || !o.base.height.is_multiple_of(64)) {
        return Err(format!("{}x{} cannot take the two-stage path (both axes must be a multiple of 64), and this plan needs it", o.base.width, o.base.height));
    }

    let sigmas: Vec<f64> = if is_real_distilled {
        LTX2_DISTILLED_SIGMAS.iter().map(|&s| s as f64).collect()
    } else {
        // One schedule for every window: the shift is a function of the token
        // count, and windows of one plan differ in length, so deriving it per
        // window would denoise the clip's own halves on different schedules.
        ltx2_sigmas(plan[0].tokens(lh, lw), o.base.steps, o.base.base_shift, o.base.max_shift, o.base.stretch, o.base.terminal)
    };
    let steps = sigmas.len() - 1;

    // Phases: build, then per window its stage-1 steps, (two-stage only) one
    // upscale plus the refinement steps, and one decode.
    let per_window = steps as u32 + if two_stage { LTX2_STAGE2_STEPS as u32 + 1 } else { 0 } + 1;
    let total = 1 + plan.len() as u32 * per_window;
    tracing::info!(windows = plan.len(), context_latent_frames = context, max_window_tokens = o.max_window_tokens, widest_tokens = widest, two_stage, frames = o.base.frames, "long-form generation planned");

    // The audio stream's own layout over the same plan, derived and CHECKED
    // before any weight is read: a seam that does not land on a whole audio
    // token is refused here rather than approximated later (see
    // `crate::audio::audio_plan`).
    let aplan = o.base.audio.then(|| crate::audio::audio_plan(&plan, context, o.base.frames, o.base.fps)).transpose()?;
    if let Some(a) = &aplan {
        tracing::info!(
            audio_tokens = a.total,
            carried_tokens = a.context,
            per_window = ?a.per_window,
            latent_frame_quantum = align,
            "audio stream planned across the window seams"
        );
    }

    // Resolved once, before the first forward rather than per window: it is
    // the same table every time, and a refinement schedule this run cannot
    // spell should cost milliseconds rather than a window of device time.
    let refine_sigmas = if two_stage { stage2_sigmas(LTX2_STAGE2_STEPS)? } else { Vec::new() };

    let mut timings = Timings::default();
    let place = o.base.devices.resolve(o.base.device.as_deref());
    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, &o.base, &o.base.adapters, place)?;
    if o.base.audio && !dit.has_audio() {
        return Err("ltxv: audio was requested but the built denoiser carries no audio stream".into());
    }
    timings.build_dit = build_t.elapsed().as_secs_f32();
    let TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, uncond_valid, len: context_len, a_cond, a_uncond } =
        build_context(paths, prompt, dit_cfg, &o.base, place, &mut timings.text_encode)?;
    let (a_ctx_cond, a_ctx_uncond) = (a_cond.unwrap_or_default(), a_uncond.unwrap_or_default());
    let crate::import::VideoVae { arch: varch, weights: vweights } = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
    // One cache for the whole clip, filled by the first window that refines
    // and reused by every later one (see [`SpatialUpsampler`]).
    let upsampler = SpatialUpsampler::default();

    let mut out_frames: Vec<Vec<u8>> = Vec::with_capacity(o.base.frames);
    // The rolling state, and the whole of it: at most `context` latent frames
    // per stage, plus - on an audio-visual clip - the same seam's worth of
    // audio tokens. It does not grow with the clip's length.
    let mut carried_full: Option<(Vec<f32>, usize)> = None;
    let mut carried_half: Option<(Vec<f32>, usize)> = None;
    let mut carried_audio_full: Vec<f32> = Vec::new();
    let mut carried_audio_half: Vec<f32> = Vec::new();
    // The clip's whole audio latent, assembled window by window and decoded
    // ONCE at the end. Decoding per window instead would put a causal-VAE
    // boundary at every seam - the audio VAE's first latent frame covers one
    // mel frame rather than four - and then need those pieces butted together
    // in the waveform, which is exactly the click this design avoids by never
    // creating it.
    let mut audio_latent: Vec<f32> = Vec::new();
    // An audio source is the whole clip's audio stream, encoded once: each
    // window holds its own slice of it frozen - clean, at timestep 0 - while
    // the picture denoises against it, the single-window audio-to-video
    // conditioning cut along the same seams the generated audio would cross.
    let source_latent: Option<Vec<f32>> = match (&o.base.audio_source, &aplan) {
        (Some(source), Some(a)) => {
            let path = paths.audio_vae.as_deref().ok_or("ltxv: an audio source is encoded through the audio VAE, and no audio VAE path is set")?;
            let (_, vae_raw): (Vec<_>, Vec<_>) = checkpoint::safetensors::read(path)?.into_iter().partition(|t| t.name.starts_with("vocoder."));
            let vae_w = crate::import::import_audio_vae(vae_raw, &crate::audio_vae::AudioVaeConfig::ltx25())?;
            Some(crate::audio::encode(&vae_w, source, a.total, o.base.device.as_deref())?)
        }
        _ => None,
    };
    let mut vae_secs = 0.0f32;
    let work_t = Instant::now();
    let mut done_before = 1u32;

    for (wi, w) in plan.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let lat_t = w.latent_frames();
        // Routed per window like every other path's: the anchors whose
        // instants this window owns condition it in-window, and the anchors
        // still ahead of it ride along as future guides (see
        // [`window_gen_opts`]). The inline options this replaces set
        // `end_frame: None` everywhere and passed the clip-global mid anchor
        // through verbatim - its position reinterpreted as each window's OWN
        // local frame - so the end still vanished and the mid still landed at
        // the wrong instant in every window but its owner's.
        let win_opts = window_gen_opts(&o.base, &plan, wi)?;
        let sc = StageCtx {
            a_ctx_cond: &a_ctx_cond,
            a_ctx_uncond: &a_ctx_uncond,
            dit: dit.as_ref(),
            vcfg: &vcfg,
            vweights: &vweights,
            o: &win_opts,
            lat_t,
            in_channels,
            ctx_cond: Caption::of(&ctx_cond, &context_valid),
            ctx_uncond: Caption::of(&ctx_uncond, &uncond_valid),
            context_len,
            cancel,
            reference: None,
        };
        // The plan's own context count and what the previous window could
        // actually supply have to be the same number - the emitted-frame
        // arithmetic is derived from the former and the freeze from the
        // latter, so a disagreement would silently shift the clip.
        if carried_full.as_ref().map(|(_, n)| *n).unwrap_or(0) != w.context {
            return Err(format!("window {wi} plans a {}-latent-frame context but the previous window carried {:?}", w.context, carried_full.as_ref().map(|(_, n)| *n)));
        }
        // Distinct per window, so two windows of one clip never draw the same
        // noise and repeat each other's content.
        let seed_salt = 0x57_49_4e_44_00_00_00_00u64 ^ wi as u64;
        tracing::info!(window = wi, of = plan.len(), latent_frames = lat_t, carried = w.context, new = w.new, tokens = w.tokens(lh, lw), "window starting");

        // This window's fresh audio noise, sized by the plan. The carried
        // tokens go at the head of it inside the stage, exactly as the video
        // half's carried latent frames do.
        let audio_noise = aplan.as_ref().map(|a| seeded_noise(a.per_window[wi] * crate::audio::TOKEN_DIM as usize, o.base.seed ^ AUDIO_SEED_SALT ^ seed_salt));
        // This window's slice of the source soundtrack: it starts where the
        // window's audio sequence does, `context` tokens before the window's
        // own first new token.
        let held_audio: Option<(Vec<f32>, Vec<f32>)> = match (&source_latent, &aplan) {
            (Some(src), Some(a)) => {
                let dim = crate::audio::TOKEN_DIM as usize;
                let start = audio_latent.len() / dim - if wi == 0 { 0 } else { a.context };
                Some((vec![0.0; a.per_window[wi]], src[start * dim..(start + a.per_window[wi]) * dim].to_vec()))
            }
            _ => None,
        };
        // The same agreement the video context is held to, on the other
        // stream: what the plan says this window carries and what the previous
        // window actually handed over have to be the same number, or the two
        // streams slide against each other by exactly the difference - which
        // is not a failure, it is audible drift.
        if let Some(a) = &aplan {
            let dim = crate::audio::TOKEN_DIM as usize;
            let want = if wi == 0 { 0 } else { a.context * dim };
            if carried_audio_full.len() != want || (two_stage && carried_audio_half.len() != want) {
                return Err(format!(
                    "window {wi} plans a {}-token audio context but the previous window carried {} (its first stage carried {})",
                    want / dim,
                    carried_audio_full.len() / dim,
                    carried_audio_half.len() / dim
                ));
            }
        }

        let stage_out = if two_stage {
            let (w1, h1) = (o.base.width / 2, o.base.height / 2);
            let (lh1, lw1) = (h1 / 32, w1 / 32);
            let stage1 = denoise_stage(
                &sc,
                Stage {
                    mask: None,
                    width: w1,
                    height: h1,
                    sigmas: &sigmas,
                    eta: o.base.eta,
                    seed_chw: None,
                    // Stage 1 builds this window's structure, so this is
                    // where the motion history has to be - at stage 1's own
                    // half resolution, taken from the previous window's own
                    // stage-1 latent rather than a downsampled full-res one
                    // (there is no x0.5 latent downsampler, and inventing one
                    // would put content into stage 1 that no LTX-2.5
                    // component produced).
                    context: carried_half.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                    seed_salt,
                    done_before,
                    label: "stage1",
                    audio: audio_noise.map(|latent| AudioStage { latent, context: std::mem::take(&mut carried_audio_half), frozen: held_audio.clone() }),
                    slots: None,
                },
                total,
                &mut progress,
            )?;
            let stage1_chw = stage1.video_chw;
            carried_half = Some((crate::longform::carry_tail(&stage1_chw, in_channels, lat_t, lh1, lw1, context.min(lat_t)), context.min(lat_t)));
            // The audio stream has no half-resolution copy of itself - it has
            // no spatial axes at all - so stage 1's tail is carried on exactly
            // the same terms the video's is, into the NEXT window's stage 1.
            if let (Some(latent), Some(a)) = (&stage1.audio, &aplan) {
                carried_audio_half = crate::audio::carry_tail(latent, a.context);
            }
            // Stage 2 is a different token count, so the resident window has
            // to be rebuilt for it either way (`crate::devres::DitSession::
            // prefill`). Releasing the whole session instead of rebuilding
            // inside it costs one more device open and buys two things: the x2
            // upscaler about to be built gets a card with none of the DiT's
            // weights on it, and the new session's slot count is planned from
            // stage 2's OWN token count rather than inherited from stage 1's
            // much smaller one.
            dit.release_devices();
            upscale_and_refine(
                &sc,
                Refine {
                    upsampler: &upsampler,
                    upsampler_path: paths.spatial_upsampler.as_deref().expect("checked before the first forward"),
                    slots: None,
                    latent_chw: &stage1_chw,
                    lat_t,
                    lh1,
                    lw1,
                    width: o.base.width,
                    height: o.base.height,
                    sigmas: &refine_sigmas,
                    eta: refine_eta(&o.base),
                    context: carried_full.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                    seed_salt,
                    done_before: done_before + steps as u32,
                    label: "stage2",
                    // Stage 1's own audio latent, re-noised inside the stage,
                    // with the previous WINDOW's final audio tokens written
                    // over its head - the audio counterpart of `context`
                    // above, and for the same reason: what a continuation
                    // window is pinned to is the previous window's own final
                    // output, at this stage's own place in the schedule.
                    audio: stage1.audio.map(|latent| AudioStage { latent, context: std::mem::take(&mut carried_audio_full), frozen: held_audio.clone() }),
                },
                total,
                &mut progress,
            )?
        } else {
            // One stage, so this window's only audio sequence is the one
            // built above - carried prefix and all.
            denoise_stage(
                &sc,
                Stage {
                    mask: None,
                    width: o.base.width,
                    height: o.base.height,
                    sigmas: &sigmas,
                    eta: o.base.eta,
                    seed_chw: None,
                    context: carried_full.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                    seed_salt,
                    done_before,
                    label: "single",
                    audio: audio_noise.map(|latent| AudioStage { latent, context: std::mem::take(&mut carried_audio_full), frozen: held_audio.clone() }),
                    slots: None,
                },
                total,
                &mut progress,
            )?
        };
        let final_chw = stage_out.video_chw;
        carried_full = Some((crate::longform::carry_tail(&final_chw, in_channels, lat_t, lh, lw, context.min(lat_t)), context.min(lat_t)));
        // Everything past the carried prefix is this window's own contribution
        // to the clip's single audio latent, and the prefix itself was already
        // contributed by the window that generated it - the exact counterpart
        // of dropping this window's leading decoded pixel frames below.
        if let (Some(latent), Some(a)) = (&stage_out.audio, &aplan) {
            let carried_vals = if wi == 0 { 0 } else { a.context * crate::audio::TOKEN_DIM as usize };
            audio_latent.extend_from_slice(&latent[carried_vals..]);
            carried_audio_full = crate::audio::carry_tail(latent, a.context);
        }

        done_before += per_window - 1;
        // What [`generate`] achieves with `drop(dit)` before its own decode,
        // and the reason it does: a VAE decode opens its own device and needs
        // up to ~16.5 GiB at the shapes this pipeline supports, which does not
        // fit alongside a resident weight window. A window loop cannot drop
        // the denoiser - the next window needs it - so it hands the card back
        // and lets the next window's first forward re-open it.
        dit.release_devices();
        progress(done_before, total, "vae decode");
        let dec_t = Instant::now();
        let (pixels, got) = decode_video(&vcfg, varch, &vweights, lat_t as u32, lh as u32, lw as u32, &final_chw, &o.base)?;
        vae_secs += dec_t.elapsed().as_secs_f32();
        if got != w.decoded_frames() || pixels.len() != 3 * got * o.base.height * o.base.width {
            return Err(format!("window {wi} decoded to {got} frames / {} values, expected {} / {}", pixels.len(), w.decoded_frames(), 3 * w.decoded_frames() * o.base.height * o.base.width));
        }
        // The carried frames are decoded because a latent frame cannot be
        // decoded without its neighbours - that is what makes the pixel seam
        // continuous - and then dropped. They are emitted exactly once, by
        // the window that generated them, and never re-encoded.
        let rgb = chw_to_rgb8(&pixels, got, o.base.height, o.base.width);
        out_frames.extend(rgb.into_iter().skip(w.dropped_frames()));
        done_before += 1;
    }
    drop(dit);

    if out_frames.len() != o.base.frames {
        return Err(format!("reassembled {} frames from {} windows, expected {}", out_frames.len(), plan.len(), o.base.frames));
    }
    timings.decode = vae_secs;
    timings.denoise = (work_t.elapsed().as_secs_f32() - vae_secs).max(0.0);
    timings.steps = plan.len() * (steps + if two_stage { LTX2_STAGE2_STEPS } else { 0 });
    timings.tokens = plan.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    timings.forwards_per_step = o.base.guidance.forwards_per_step(o.base.audio);

    // ---- decode the sound, ONCE over the whole clip ------------------------
    // The windows contributed tokens to one latent, not waveforms to be butted
    // together, so the audio VAE and the vocoder see a single continuous
    // sequence and there is no decode boundary at a seam to smooth over. The
    // token count is the clip's own, so this is the same arithmetic - and the
    // same sample count - a single-window clip of this length produces.
    let audio = match &aplan {
        // The source soundtrack IS the clip's audio: every window held it
        // frozen, so decoding its latent would only add the VAE's loss.
        Some(_) if o.base.audio_source.is_some() => {
            let mut clip = o.base.audio_source.clone().expect("checked by the guard");
            let video_seconds = out_frames.len() as f32 / o.base.fps as f32;
            let keep = (video_seconds * clip.sample_rate as f32).round() as usize;
            clip.channels.iter_mut().for_each(|c| c.truncate(keep));
            clip.pad_to_seconds(video_seconds);
            Some(clip)
        }
        Some(a) => {
            let dim = crate::audio::TOKEN_DIM as usize;
            if audio_latent.len() != a.total * dim {
                return Err(format!("reassembled {} audio tokens from {} windows, expected {}", audio_latent.len() / dim, plan.len(), a.total));
            }
            let path = paths.audio_vae.as_deref().expect("an audio request is refused earlier without an audio VAE path");
            let audio_t = Instant::now();
            let mut clip = decode_audio_latent(path, &audio_latent, a.total, o.base.device.as_deref())?;
            let video_seconds = out_frames.len() as f32 / o.base.fps as f32;
            let before = clip.seconds();
            clip.pad_to_seconds(video_seconds);
            timings.audio_decode = audio_t.elapsed().as_secs_f32();
            tracing::info!(secs = timings.audio_decode, samples = clip.samples_per_channel(), rate = clip.sample_rate, decoded_seconds = before, video_seconds, "audio decode done");
            Some(clip)
        }
        None => None,
    };
    progress(total, total, "done");
    tracing::info!(frames = out_frames.len(), windows = plan.len(), audio = audio.is_some(), total_secs = timings.total(), "long-form generation done");
    Ok((Video { width: o.base.width as u32, height: o.base.height as u32, fps: o.base.fps, frames: out_frames, audio }, timings))
}

/// Stage-major long-form generation: stage 1 builds the clip's motion
/// structure at half resolution in as few, as global a pass as possible;
/// stage 2 refines at full resolution, reading slices of stage 1's output.
///
/// This is the path [`LongOpts::stage_major`] selects (on by default for
/// video-only clips).  Audio-visual clips keep the window-major
/// loop in [`generate_long`].
///
/// Stage 1's plan is sized by the half-resolution token count, so clips whose
/// entire stage-1 trajectory fits in one window get a single global pass.
/// Stage 2 fragments as much as memory demands, with no quality cost because
/// the motion is already fixed by stage 1.
fn run_stage_major(
    paths: &Paths,
    prompt: &str,
    o: &LongOpts,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(u32, u32, &str),
) -> Result<(Video, Timings), String> {
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(o.base.frames as u32).is_none() {
        return Err(format!("{} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)", o.base.frames));
    }
    if !o.base.width.is_multiple_of(32) || !o.base.height.is_multiple_of(32) {
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", o.base.width, o.base.height));
    }
    let (lh, lw) = (o.base.height / 32, o.base.width / 32);
    let (w1, h1) = (o.base.width / 2, o.base.height / 2);
    let (lh1, lw1) = (h1 / 32, w1 / 32);
    let align = 1u32; // video-only; audio clips never reach here

    let tsp = two_stage_long_plan(
        o.base.frames, lh1, lw1, lh, lw,
        o.context_latent_frames, o.max_window_tokens, o.max_refine_tokens,
        align as usize,
    )?;
    if tsp.stage2.len() < 2 {
        return Err("stage-major path requires >= 2 stage-2 windows (otherwise generate_long's fast path handles it)".into());
    }
    // A mid anchor that routes into no window would be honoured by none of
    // them - silently dropped, the defect the window-major path refuses
    // rather than commits. Every pixel frame inside the clip lands in exactly
    // one window's decoded range (the ranges tile the clip), so only an
    // out-of-clip anchor can fail to route: refuse it here, before any weight
    // is read.
    if let (Some(_), Some(global_at)) = (&o.base.mid_frame, o.base.mid_frame_at) {
        if crate::longform::route_mid_anchor(&tsp.stage1, global_at).is_none() {
            return Err(format!("--mid-frame-at {global_at} lands outside this {}-frame clip: no window's decoded range contains it, so no window could honour it", o.base.frames));
        }
    }

    check_audio_request(paths, &o.base)?;
    if o.base.steps == 0 {
        return Err("--steps must be at least 1".into());
    }

    let dit_cfg = dit_config_from_name(&o.base.dit_config)?;
    if dit_cfg.in_channels != vcfg.latent_channels {
        return Err(format!("ltxv dit-config {:?} has in_channels {} but the VAE latent width is {}", o.base.dit_config, dit_cfg.in_channels, vcfg.latent_channels));
    }
    let in_channels = dit_cfg.in_channels as usize;
    let is_real_distilled = o.base.dit_config == "ltx25_22b";
    let context = o.context_latent_frames;

    let sigmas: Vec<f64> = if is_real_distilled {
        LTX2_DISTILLED_SIGMAS.iter().map(|&s| s as f64).collect()
    } else {
        ltx2_sigmas(tsp.stage1[0].tokens(lh1, lw1), o.base.steps, o.base.base_shift, o.base.max_shift, o.base.stretch, o.base.terminal)
    };
    let steps = sigmas.len() - 1;
    let refine_sigmas = stage2_sigmas(LTX2_STAGE2_STEPS)?;

    // Phases: build, stage-1 windows, stage-2 windows, one decode.
    let per_s1 = steps as u32;
    let per_s2 = LTX2_STAGE2_STEPS as u32 + 1;
    let total = 1 + tsp.stage1.len() as u32 * per_s1 + tsp.stage2.len() as u32 * per_s2 + 1;
    tracing::info!(
        stage1_windows = tsp.stage1.len(),
        stage2_windows = tsp.stage2.len(),
        stage1_max_tokens = o.max_window_tokens,
        stage2_max_tokens = o.max_refine_tokens,
        stage2_context = tsp.stage2_context,
        frames = o.base.frames,
        "stage-major long-form generation planned"
    );

    let mut timings = Timings::default();
    let place = o.base.devices.resolve(o.base.device.as_deref());
    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, &o.base, &o.base.adapters, place)?;
    if o.base.audio && !dit.has_audio() {
        return Err("ltxv: audio was requested but the built denoiser carries no audio stream".into());
    }
    timings.build_dit = build_t.elapsed().as_secs_f32();
    let TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, uncond_valid, len: context_len, .. } =
        build_context(paths, prompt, dit_cfg, &o.base, place, &mut timings.text_encode)?;
    let crate::import::VideoVae { arch: varch, weights: vweights } = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
    let upsampler = SpatialUpsampler::default();

    let mut out_frames: Vec<Vec<u8>> = Vec::with_capacity(o.base.frames);
    let mut vae_secs = 0.0f32;
    let work_t = Instant::now();
    let mut done_before = 1u32;

    // ---- Stage 1: build the global half-res latent -------------------------
    let mut global_half = vec![0f32; in_channels * tsp.lat_t * lh1 * lw1];
    let mut s1_carried: Option<(Vec<f32>, usize)> = None;
    for (wi, w) in tsp.stage1.iter().enumerate() {
        if cancel.is_cancelled() { return Err("cancelled".into()); }
        let lat_t = w.latent_frames();
        let win_opts = window_gen_opts(&o.base, &tsp.stage1, wi)?;
        let sc = StageCtx {
            a_ctx_cond: &[],
            a_ctx_uncond: &[],
            dit: dit.as_ref(),
            vcfg: &vcfg,
            vweights: &vweights,
            o: &win_opts,
            lat_t,
            in_channels,
            ctx_cond: Caption::of(&ctx_cond, &context_valid),
            ctx_uncond: Caption::of(&ctx_uncond, &uncond_valid),
            context_len,
            cancel,
            reference: None,
        };
        let seed_salt = 0x57_49_4e_44_00_00_00_00u64 ^ wi as u64;
        tracing::info!(window = wi, of = tsp.stage1.len(), latent_frames = lat_t, carried = w.context, new = w.new, tokens = w.tokens(lh1, lw1), "stage1 window starting");

        let stage_out = denoise_stage(
            &sc,
            Stage {
                mask: None,
                width: w1,
                height: h1,
                sigmas: &sigmas,
                eta: o.base.eta,
                seed_chw: None,
                context: s1_carried.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                seed_salt,
                done_before,
                label: "s1",
                audio: None,
                slots: None,
            },
            total,
            &mut progress,
        )?;
        let stage1_chw = stage_out.video_chw;
        s1_carried = Some((crate::longform::carry_tail(&stage1_chw, in_channels, lat_t, lh1, lw1, context.min(lat_t)), context.min(lat_t)));
        // Write this window's NEW frames into the global half-res buffer.
        let first_lat = w.first_latent_frame(if wi == 0 { 0 } else { tsp.stage1[..wi].iter().map(|w| w.new).sum() });
        crate::longform::write_latent_window(&mut global_half, in_channels, tsp.lat_t, lh1, lw1, first_lat, &stage1_chw[(w.context * lh1 * lw1 * in_channels)..]);
        done_before += per_s1;
    }

    // ---- Stage 2: refine each stage-2 window from the global half-res ------
    let mut carried_full: Option<(Vec<f32>, usize)> = None;
    for (wi, w) in tsp.stage2.iter().enumerate() {
        if cancel.is_cancelled() { return Err("cancelled".into()); }
        // Stage 2 reads its NEW frames from the global half-res latent. The
        // window's shape comes from the PLAN (`w.context + w.new`): the refine
        // input must be exactly this window's own `latent_frames()` so every
        // decoded frame count the plan promised holds, and the context slots
        // hold the half-res frames immediately before this window's own -
        // never some other window's tail.
        let new_count = w.new;
        let first_lat = w.first_latent_frame(if wi == 0 { 0 } else { tsp.stage2[..wi].iter().map(|w| w.new).sum() });
        let s1_new = crate::longform::latent_window(&global_half, in_channels, tsp.lat_t, lh1, lw1, first_lat, new_count);
        // `ctx_count` is THIS window's own plan context: 0 for the first
        // stage-2 window, `tsp.stage2_context` for the rest.  The context
        // slots are filled from the global half-res buffer (the frames just
        // before this window's own), giving the upscaler real content to
        // chew on; for every window with context > 0 the `carried_full`
        // overlay inside `denoise_stage` then overwrites them with the
        // previous stage-2 window's own refined output.
        let ctx_count = w.context;
        let lat_t = ctx_count + new_count;
        let mut s1_slice = vec![0f32; in_channels * lat_t * lh1 * lw1];
        if ctx_count > 0 {
            let ctx_first = first_lat.saturating_sub(ctx_count);
            let ctx_data = crate::longform::latent_window(&global_half, in_channels, tsp.lat_t, lh1, lw1, ctx_first, ctx_count);
            s1_slice[..ctx_count * lh1 * lw1 * in_channels].copy_from_slice(&ctx_data);
        }
        s1_slice[ctx_count * lh1 * lw1 * in_channels..].copy_from_slice(&s1_new);
        // Stage-2 keyframes route per-window, exactly like stage 1: an anchor
        // belongs at the position where its instant lives in THIS window's own
        // sequence. Blasting every keyframe into every window instead pins the
        // start still onto position 0 of windows whose position 0 is a frozen
        // mid-video context frame - conditioning that contradicts the carried
        // context and re-anchors each segment, which is one way a long-form
        // clip fractures into per-window trajectories.
        let win_opts = window_gen_opts(&o.base, &tsp.stage2, wi)?;
        let sc = StageCtx {
            a_ctx_cond: &[],
            a_ctx_uncond: &[],
            dit: dit.as_ref(),
            vcfg: &vcfg,
            vweights: &vweights,
            o: &win_opts,
            lat_t,
            in_channels,
            ctx_cond: Caption::of(&ctx_cond, &context_valid),
            ctx_uncond: Caption::of(&ctx_uncond, &uncond_valid),
            context_len,
            cancel,
            reference: None,
        };
        let seed_salt = 0x57_49_4e_44_00_00_00_00u64 ^ (tsp.stage1.len() + wi) as u64;
        tracing::info!(window = wi, of = tsp.stage2.len(), latent_frames = lat_t, carried = w.context, new = w.new, tokens = w.tokens(lh, lw), "stage2 window starting");

        dit.release_devices();
        let stage_out = upscale_and_refine(
            &sc,
            Refine {
                upsampler: &upsampler,
                upsampler_path: paths.spatial_upsampler.as_deref().expect("checked before the first forward"),
                slots: None,
                latent_chw: &s1_slice,
                lat_t,
                lh1,
                lw1,
                width: o.base.width,
                height: o.base.height,
                sigmas: &refine_sigmas,
                eta: refine_eta(&o.base),
                context: carried_full.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                seed_salt,
                done_before,
                label: "s2",
                audio: None,
            },
            total,
            &mut progress,
        )?;
        let final_chw = stage_out.video_chw;
        carried_full = Some((crate::longform::carry_tail(&final_chw, in_channels, lat_t, lh, lw, tsp.stage2_context.min(lat_t)), tsp.stage2_context.min(lat_t)));
        done_before += per_s2;

        dit.release_devices();
        progress(done_before, total, "vae decode");
        let dec_t = Instant::now();
        let (pixels, got) = decode_video(&vcfg, varch, &vweights, lat_t as u32, lh as u32, lw as u32, &final_chw, &o.base)?;
        vae_secs += dec_t.elapsed().as_secs_f32();
        if got != w.decoded_frames() || pixels.len() != 3 * got * o.base.height * o.base.width {
            return Err(format!("window {wi} decoded to {got} frames / {} values, expected {} / {}", pixels.len(), w.decoded_frames(), 3 * w.decoded_frames() * o.base.height * o.base.width));
        }
        let rgb = chw_to_rgb8(&pixels, got, o.base.height, o.base.width);
        out_frames.extend(rgb.into_iter().skip(w.dropped_frames()));
        done_before += 1;
    }
    drop(dit);

    if out_frames.len() != o.base.frames {
        return Err(format!("reassembled {} frames from {} stage-2 windows, expected {}", out_frames.len(), tsp.stage2.len(), o.base.frames));
    }
    timings.decode = vae_secs;
    timings.denoise = (work_t.elapsed().as_secs_f32() - vae_secs).max(0.0);
    timings.steps = tsp.stage1.len() * steps + tsp.stage2.len() * LTX2_STAGE2_STEPS;
    timings.tokens = tsp.stage2.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    timings.forwards_per_step = o.base.guidance.forwards_per_step(o.base.audio);
    progress(total, total, "done");
    tracing::info!(frames = out_frames.len(), stage1_windows = tsp.stage1.len(), stage2_windows = tsp.stage2.len(), total_secs = timings.total(), "stage-major long-form generation done");
    Ok((Video { width: o.base.width as u32, height: o.base.height as u32, fps: o.base.fps, frames: out_frames, audio: None }, timings))
}

/// One scene's own [`LongOpts`] within a [`generate_scenes`] run - `frames`
/// and `seed` become the scene's own, `start_frame` conditions scene 0's
/// opening only, and everything else (including `end_frame`/`mid_frame`)
/// passes through from the caller's request untouched.
///
/// **Do not add an unconditional `end_frame: None` (or `mid_frame: None`)
/// here.** `generate_scenes` calls this for the `scenes.len() == 1` case
/// too - the ordinary, no-`--scene` `t2v` request - and an earlier version
/// of this function zeroed `end_frame` for every scene regardless of `si`,
/// which silently discarded `--end-frame` on every single-scene call ever
/// made. The multi-scene
/// refusal in `generate_scenes`, run before this is ever called more than
/// once, is what keeps `end_frame`/`mid_frame` at `None` once more than one
/// scene is in play; this function has no case of its own left to enforce.
fn scene_gen_opts(o: &LongOpts, s: &Scene, si: usize) -> LongOpts {
    LongOpts {
        base: GenOpts {
            frames: s.frames,
            // The still conditions the clip's own opening, which is the first
            // scene's first window and nowhere else.
            start_frame: if si == 0 { o.base.start_frame.clone() } else { None },
            seed: o.base.seed ^ crate::longform::SCENE_SEED_SALT.wrapping_mul(si as u64),
            ..o.base.clone()
        },
        ..o.clone()
    }
}

#[cfg(test)]
mod scene_gen_opts_tests {
    use super::*;

    fn scene(frames: usize, prompt: &str) -> Scene {
        Scene { frames, prompt: prompt.into() }
    }

    /// The exact defect this function was extracted to stop from coming
    /// back: a single-scene request (`si == 0`, the ordinary `t2v` path)
    /// must keep the caller's own `end_frame`/`mid_frame`, not silently
    /// drop them.
    #[test]
    fn a_single_scene_keeps_the_callers_end_and_mid_frame() {
        let o = LongOpts { base: GenOpts { end_frame: Some("end.png".into()), mid_frame: Some("mid.png".into()), mid_frame_at: Some(4), ..GenOpts::default() }, ..LongOpts::default() };
        let got = scene_gen_opts(&o, &scene(9, "a shot"), 0);
        assert_eq!(got.base.end_frame.as_deref(), Some("end.png"), "single-scene end_frame must survive unchanged");
        assert_eq!(got.base.mid_frame.as_deref(), Some("mid.png"), "single-scene mid_frame must survive unchanged");
        assert_eq!(got.base.mid_frame_at, Some(4));
    }

    /// `start_frame` conditions scene 0's opening only - unchanged behaviour,
    /// pinned down alongside the fix above so the two do not drift apart
    /// again under a future edit.
    #[test]
    fn only_scene_zero_keeps_the_start_frame() {
        let o = LongOpts { base: GenOpts { start_frame: Some("start.png".into()), ..GenOpts::default() }, ..LongOpts::default() };
        assert_eq!(scene_gen_opts(&o, &scene(9, "a"), 0).base.start_frame.as_deref(), Some("start.png"));
        assert_eq!(scene_gen_opts(&o, &scene(9, "b"), 1).base.start_frame, None);
    }

    /// Each scene gets the request's own frame count and a distinct,
    /// derived seed - the two fields this function derives rather than
    /// copying verbatim.
    #[test]
    fn each_scene_gets_its_own_frame_count_and_a_distinct_seed() {
        let o = LongOpts { base: GenOpts::default(), ..LongOpts::default() };
        let s0 = scene_gen_opts(&o, &scene(9, "a"), 0);
        let s1 = scene_gen_opts(&o, &scene(17, "b"), 1);
        assert_eq!(s0.base.frames, 9);
        assert_eq!(s1.base.frames, 17);
        assert_ne!(s0.base.seed, s1.base.seed, "each scene must denoise from its own seed");
    }
}

/// Progress units one scene of a [`generate_scenes`] run is worth.
///
/// A scene's own phase count is not known until [`generate_long`] has planned
/// it, and planning every scene twice just to total them up would duplicate
/// the stage decision. Each scene's `(done, total)` is rescaled into its own
/// slice of this instead, so the reported fraction is monotonic and correct
/// without a second planner.
const SCENE_PROGRESS_UNITS: u32 = 100;

/// Text to video across SEVERAL scenes: one clip, one file, a different prompt
/// per scene, and a genuine cut between them.
///
/// **This is [`generate_long`] run once per scene.** Inside a scene everything
/// is exactly what that function does - the rolling latent context, the window
/// plan, the two-stage decision, the per-window device release - and a scene
/// boundary is a reset because a fresh `generate_long` call's first window
/// carries nothing. There is no second window loop and no second sampler here;
/// what this function adds is the up-front plan over all scenes, the per-scene
/// seed, and the concatenation.
///
/// **Why the reset is the default and not an option.** A continuation window
/// is HARD-conditioned on real content at sigma 0 (see [`crate::longform`]),
/// which is what keeps one shot's motion continuous and is exactly what would
/// stop a new scene from becoming a different scene. A caller who wants the
/// content to continue is asking for one scene, and says so by writing one.
///
/// **A single-scene call is handed straight to [`generate_long`]**, so a
/// request that names one scene is bit-for-bit the run it already was -
/// including `end_frame`/`mid_frame`, which [`scene_gen_opts`] must not
/// touch: the multi-scene refusal below is what keeps them `None` by the
/// time more than one scene is in play, so this function has no case of its
/// own left to enforce.
///
/// [`LongOpts::base`]'s own `frames` is NOT read - each [`Scene`] brings its
/// own length, and the clip's length is their sum. `start_frame` conditions
/// the first scene's opening and nowhere else; `end_frame` and `mid_frame` are
/// refused for the same reasons [`generate_long`] refuses them.
#[tracing::instrument(level = "info", name = "generate_scenes", skip_all, fields(scenes = scenes.len(), width = o.base.width, height = o.base.height, seed = o.base.seed))]
pub fn generate_scenes(paths: &Paths, scenes: &[Scene], o: &LongOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    let Some(first) = scenes.first() else {
        return Err("a generation needs at least one scene".into());
    };
    let scene_opts = |s: &Scene, si: usize| scene_gen_opts(o, s, si);
    if scenes.len() == 1 {
        return generate_long(paths, &first.prompt, &scene_opts(first, 0), cancel, progress);
    }
    if !o.base.width.is_multiple_of(32) || !o.base.height.is_multiple_of(32) {
        return Err(format!("{}x{} is not a multiple of 32 (the VAE's spatial stride)", o.base.width, o.base.height));
    }
    if o.base.end_frame.is_some() {
        return Err("--end-frame is not supported for a multi-scene clip: it pins the last frame of ONE window, and the last window of a multi-scene plan is the last scene's, which is not what a caller asking for a final still means".into());
    }
    if o.base.mid_frame.is_some() {
        return Err("--mid-frame is not supported for a multi-scene clip: \"the middle of the clip\" is a position in a timeline that is now a sequence of scenes, and which scene should own it has not been designed - anchor the scene you mean by generating it on its own".into());
    }
    // A window seam inside a scene carries both streams; a SCENE boundary
    // deliberately carries neither, which is what makes the next scene free to
    // be a different subject. For the picture that is a cut and it is the
    // point; for the sound it is a restart, and a scene's token count is
    // `round(scene_frames / fps * rate)`, so the scenes' counts do not sum to
    // the clip's own and the two tracks would not even be the same length.
    // What sound should do at a deliberate visual cut is a design question,
    // not an arithmetic one, and it is refused rather than guessed.
    if o.base.audio {
        return Err(
            "--audio is not supported for a multi-scene clip: a scene boundary deliberately carries nothing across, so the sound would restart at every cut, and the per-scene token counts do not sum to the clip's own. Ask for one scene (any length - a long single-scene clip carries its sound across every window seam), or drop --audio."
                .into(),
        );
    }
    let (lh, lw) = (o.base.height / 32, o.base.width / 32);
    // Every scene, before the first weight is read: a five-scene request whose
    // fourth scene cannot be planned fails now, not three scenes of device
    // time later.
    let plan = crate::longform::scene_plan(scenes, lh, lw, o.context_latent_frames, o.max_window_tokens)?;
    let total_frames: usize = scenes.iter().map(|s| s.frames).sum();
    tracing::info!(scenes = scenes.len(), windows = plan.iter().map(Vec::len).sum::<usize>(), frames = total_frames, "multi-scene generation planned");

    let n = scenes.len();
    let total = n as u32 * SCENE_PROGRESS_UNITS;
    let mut out_frames: Vec<Vec<u8>> = Vec::with_capacity(total_frames);
    let mut timings = Timings::default();
    for (si, s) in scenes.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        tracing::info!(scene = si + 1, of = n, frames = s.frames, windows = plan[si].len(), prompt = %s.prompt, "scene starting");
        let done_before = si as u32 * SCENE_PROGRESS_UNITS;
        let (clip, t) = generate_long(paths, &s.prompt, &scene_opts(s, si), cancel, |done, scene_total, phase| {
            progress(done_before + (done * SCENE_PROGRESS_UNITS) / scene_total.max(1), total, &format!("scene {}/{n}: {phase}", si + 1));
        })?;
        if clip.frames.len() != s.frames {
            return Err(format!("scene {} came back as {} frames, expected {}", si + 1, clip.frames.len(), s.frames));
        }
        out_frames.extend(clip.frames);
        timings.build_dit += t.build_dit;
        timings.text_encode += t.text_encode;
        timings.denoise += t.denoise;
        timings.decode += t.decode;
        timings.steps += t.steps;
        timings.tokens = timings.tokens.max(t.tokens);
        timings.forwards_per_step = t.forwards_per_step;
    }

    if out_frames.len() != total_frames {
        return Err(format!("reassembled {} frames from {n} scenes, expected {total_frames}", out_frames.len()));
    }
    progress(total, total, "done");
    tracing::info!(frames = out_frames.len(), scenes = n, total_secs = timings.total(), "multi-scene generation done");
    Ok((Video { width: o.base.width as u32, height: o.base.height as u32, fps: o.base.fps, frames: out_frames, audio: None }, timings))
}

// ============================================================================
// DFR (Diffusion Fidelity Rendering)
// ============================================================================

use crate::dfr;

/// The strength DFR's detailing pass applies its IC-LoRA at
/// (`ltx_pipelines.utils.constants.DETAILING_LORA_STRENGTH`).
pub const DETAILING_LORA_STRENGTH: f32 = 0.5;

/// One DFR generation (`ltx_pipelines.dfr_pipeline.DFRPipeline`): everything
/// [`GenOpts`] carries - `width`/`height` the OUTPUT resolution, `frames` the
/// requested length - plus DFR's own rounds.
#[derive(Clone, Debug, Default)]
pub struct DfrOpts {
    pub base: GenOpts,
    /// 0, 1 or 2 temporal x2 rounds, each doubling the frame rate.
    pub temporal_upsample_rounds: usize,
}

/// DFR text to video on the real checkpoint (`DFRPipeline.__call__`):
///
/// 1. the request is laid out on a keyframe-segment canvas
///    ([`dfr::resolve_canvas`]): its length padded to whole 24- or 32-frame
///    segments, one generated keyframe slot at every segment border;
/// 2. stage 1 denoises the clip at half resolution with those slots
///    ([`KeyframeSlots`]) from noise, on the distilled schedule;
/// 3. stage 2 carries the video and the slots up with the spatial x2
///    upscaler and re-denoises at full resolution under the detailing
///    IC-LoRA ([`Paths::detailing_lora`] at [`DETAILING_LORA_STRENGTH`]),
///    the half-resolution stage-1 video appended as its reference latent,
///    on the stage-2 distilled schedule;
/// 4. the canvas is trimmed back to the requested frames and decoded.
///
/// Both stages sample with the generation's own sampler and eta - DFR's
/// detailing pass is ancestral where the distilled pipeline's refinement is
/// not. The soundtrack, when asked for, is stage 1's: stage 2 re-noises and
/// denoises audio only because the video reads it through the cross-modal
/// attention, and does not ship it.
#[tracing::instrument(level = "info", name = "generate_dfr", skip_all, fields(frames = o.base.frames, width = o.base.width, height = o.base.height, seed = o.base.seed, temporal_rounds = o.temporal_upsample_rounds))]
pub fn generate_dfr(paths: &Paths, prompt: &str, o: &DfrOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    let base = &o.base;
    if base.reference.is_some() {
        return Err("DFR appends its own IC-LoRA reference (the stage-1 video); --reference-video does not apply".into());
    }
    if o.temporal_upsample_rounds > 2 {
        return Err(format!("temporal_upsample_rounds must be 0, 1, or 2, got {}", o.temporal_upsample_rounds));
    }
    if o.temporal_upsample_rounds > 0 {
        return Err("DFR's temporal rounds are not available yet: run without --temporal-upsample-rounds".into());
    }
    if base.dit_config != "ltx25_22b" || paths.dit.is_none() {
        return Err("DFR needs the real LTX-2.5 checkpoint (dit_config ltx25_22b): its keyframe slots and detailing IC-LoRA are trained for it".into());
    }
    let detailing = paths.detailing_lora.clone().ok_or("DFR's detailing pass needs the Refine-Details IC-LoRA (ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors): put it in the model store or pass --detailing-lora <path>")?;
    let upsampler_path = paths.spatial_upsampler.as_deref().ok_or("DFR needs the spatial x2 latent upscaler: put it in the model store or pass --spatial-upsampler <path>")?;
    if !base.width.is_multiple_of(64) || !base.height.is_multiple_of(64) {
        return Err(format!("{}x{} is not a multiple of 64 for DFR (stage 1 halves it, and the half must still be a multiple of the VAE's 32 spatial stride)", base.width, base.height));
    }
    check_audio_request(paths, base)?;
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(base.frames as u32).is_none() {
        return Err(format!("{} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)", base.frames));
    }
    let (canvas_frames, segment, slot_frames) = dfr::resolve_canvas(base.frames, dfr::VIDEO_TEMPORAL_SCALE)?;
    let lat_t = vcfg.latent_frames(canvas_frames as u32).expect("resolve_canvas returns a 1 + 8k frame count") as usize;
    let o1 = GenOpts { frames: canvas_frames, ..base.clone() };
    let dit_cfg = dit_config_from_name(&o1.dit_config)?;
    let in_channels = dit_cfg.in_channels as usize;
    tracing::info!(requested_frames = base.frames, canvas_frames, segment, slots = ?slot_frames, "DFR canvas resolved");

    let sigmas1: Vec<f64> = LTX2_DISTILLED_SIGMAS.iter().map(|&s| s as f64).collect();
    let sigmas2 = stage2_sigmas(LTX2_STAGE2_STEPS)?;
    let total = (sigmas1.len() - 1 + sigmas2.len() - 1) as u32 + 3;
    let mut timings = Timings::default();
    let place = o1.devices.resolve(o1.device.as_deref());

    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, &o1, &o1.adapters, place)?;
    timings.build_dit = build_t.elapsed().as_secs_f32();
    let text = build_context(paths, prompt, dit_cfg, &o1, place, &mut timings.text_encode)?;
    let crate::import::VideoVae { arch: varch, weights: vweights } = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
    let ta = crate::audio::latent_frames(canvas_frames, o1.fps);
    let audio_noise = o1.audio.then(|| seeded_noise(ta * crate::audio::TOKEN_DIM as usize, o1.seed ^ AUDIO_SEED_SALT));
    let stage_ctx = StageCtx {
        dit: dit.as_ref(),
        vcfg: &vcfg,
        vweights: &vweights,
        o: &o1,
        lat_t,
        in_channels,
        ctx_cond: Caption::of(&text.cond, &text.valid),
        ctx_uncond: Caption::of(&text.uncond, &text.uncond_valid),
        context_len: text.len,
        a_ctx_cond: text.a_cond.as_deref().unwrap_or_default(),
        a_ctx_uncond: text.a_uncond.as_deref().unwrap_or_default(),
        cancel,
        reference: None,
    };
    let denoise_t = Instant::now();
    let (w1, h1) = (base.width / 2, base.height / 2);
    let stage1 = denoise_stage(
        &stage_ctx,
        Stage {
            mask: None,
            width: w1,
            height: h1,
            sigmas: &sigmas1,
            eta: o1.eta,
            seed_chw: None,
            context: None,
            seed_salt: DFR_STAGE1_SEED_SALT,
            done_before: 1,
            label: "dfr stage1",
            audio: audio_noise.map(|latent| AudioStage { latent, context: Vec::new(), frozen: None }),
            slots: Some(KeyframeSlots { pixel_frames: &slot_frames, initial_chw: None }),
        },
        total,
        &mut progress,
    )?;
    let stage1_keyframes = stage1.slots_chw.ok_or("DFR stage 1 returned no keyframe slots")?;

    // The detailing pass: the same checkpoint under the detailing IC-LoRA,
    // stacked on the request's own adapters as `with_loras` stacks them.
    dit.release_devices();
    let detailing_spec = model::adapter::AdapterSpec { path: detailing, scale: DETAILING_LORA_STRENGTH };
    let (downscale, _) = crate::lora::reference_scale_factors(std::slice::from_ref(&detailing_spec))?;
    let adapters: Vec<_> = o1.adapters.iter().cloned().chain(std::iter::once(detailing_spec)).collect();
    let detailing_dit = build_denoiser(paths, dit_cfg, &o1, &adapters, place)?;
    let (lh1, lw1) = (h1 / 32, w1 / 32);
    let detailing_ctx = StageCtx {
        dit: detailing_dit.as_ref(),
        reference: Some(StageReference::Latent { chw: &stage1.video_chw, dims: (lat_t, lh1, lw1), downscale }),
        ..stage_ctx
    };
    let stage2 = upscale_and_refine(
        &detailing_ctx,
        Refine {
            upsampler: &SpatialUpsampler::default(),
            upsampler_path,
            slots: Some((&slot_frames, &stage1_keyframes)),
            audio: stage1.audio.clone().map(|latent| AudioStage { latent, context: Vec::new(), frozen: None }),
            latent_chw: &stage1.video_chw,
            lat_t,
            lh1,
            lw1,
            width: base.width,
            height: base.height,
            sigmas: &sigmas2,
            eta: o1.eta,
            context: None,
            seed_salt: DFR_STAGE2_SEED_SALT,
            done_before: sigmas1.len() as u32,
            label: "dfr stage2",
        },
        total,
        &mut progress,
    )?;
    drop(detailing_dit);
    timings.denoise = denoise_t.elapsed().as_secs_f32();
    timings.steps = sigmas1.len() + sigmas2.len() - 2;
    timings.tokens = lat_t * (base.height / 32) * (base.width / 32);
    timings.forwards_per_step = o1.guidance.forwards_per_step(o1.audio);
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }

    // The canvas padded the tail; the request gets its own length back.
    let (lh, lw) = (base.height / 32, base.width / 32);
    let keep = vcfg.latent_frames(base.frames as u32).expect("checked above") as usize;
    let video_chw = dfr::slice_time_chw(&stage2.video_chw, in_channels, lat_t, lh, lw, 0, keep);
    progress(total - 1, total, "vae decode");
    let decode_t = Instant::now();
    let (pixels, frames) = decode_video(&vcfg, varch, &vweights, keep as u32, lh as u32, lw as u32, &video_chw, base)?;
    let out = chw_to_rgb8(&pixels, frames, base.height, base.width);
    timings.decode = decode_t.elapsed().as_secs_f32();
    let audio = match stage1.audio {
        Some(latent) => {
            let path = paths.audio_vae.as_deref().expect("an audio request is refused earlier without an audio VAE path");
            let mut clip = decode_audio_latent(path, &latent, ta, base.device.as_deref())?;
            let video_seconds = frames as f32 / base.fps as f32;
            clip.channels.iter_mut().for_each(|c| c.truncate((video_seconds * clip.sample_rate as f32).round() as usize));
            clip.pad_to_seconds(video_seconds);
            Some(clip)
        }
        None => None,
    };
    progress(total, total, "done");
    tracing::info!(frames, width = base.width, height = base.height, audio = audio.is_some(), total_secs = timings.total(), "DFR generation done");
    Ok((Video { width: base.width as u32, height: base.height as u32, fps: base.fps, frames: out, audio }, timings))
}

/// Folded into DFR stage 1's and stage 2's noise, so neither draws what
/// another stage or pipeline does.
const DFR_STAGE1_SEED_SALT: u64 = 0x4446_5231; // "DFR1"
const DFR_STAGE2_SEED_SALT: u64 = 0x4446_5232; // "DFR2"

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the real distilled schedule's step count so a future edit to
    /// `LTX2_DISTILLED_SIGMAS` (or to this constant) cannot silently drift
    /// the two apart - `ltxv_cli`'s progress line and `generate`'s own
    /// `total` computation both depend on this staying in sync with the
    /// table's real length.
    #[test]
    fn distilled_steps_matches_the_real_sigma_table() {
        assert_eq!(LTX2_DISTILLED_STEPS, LTX2_DISTILLED_SIGMAS.len() - 1);
        assert_eq!(LTX2_DISTILLED_STEPS, 8, "the real LTX2_DISTILLED_SIGMAS table has 9 entries (8 steps) - update this pin if that table is deliberately changed");
    }

    /// The shapes this port actually runs, against the measurement in
    /// [`SINGLE_STAGE_MAX_TOKENS`]'s doc. The three that were measured GOOD
    /// on one stage must keep taking one stage (so nothing this port already
    /// shipped changes), and the one that was measured BROKEN must take two.
    #[test]
    fn the_stage_policy_matches_the_shapes_that_were_measured() {
        let toks = |w: usize, h: usize| 4 * (h / 32) * (w / 32);
        for (w, h) in [(512, 512), (960, 544), (1280, 704), (1600, 896)] {
            let t = toks(w, h);
            assert!(t <= SINGLE_STAGE_MAX_TOKENS, "{w}x{h} is {t} tokens, which the measurement says is a single-stage shape");
            assert!(!should_two_stage(t, w, h, true, None), "{w}x{h} ({t} tokens) was measured good on ONE stage and must not change path");
        }
        let (w, h) = (1920, 1088);
        let t = toks(w, h);
        assert_eq!(t, 8160);
        assert!(should_two_stage(t, w, h, true, None), "1920x1088 is {t} tokens and was measured to disintegrate on one stage");
    }

    /// The two conditions that are NOT about the token count: a
    /// non-distilled config has no distilled schedule to outgrow, and an
    /// axis that is not a multiple of 64 cannot be halved onto the VAE's
    /// 32-pixel stride (upstream asserts the same thing).
    #[test]
    fn the_stage_policy_refuses_a_shape_it_cannot_halve_and_a_config_it_does_not_apply_to() {
        let big = SINGLE_STAGE_MAX_TOKENS + 1;
        assert!(!should_two_stage(big, 1920, 1088, false, None), "the tiny random-weight config runs the generic shifted schedule, not the distilled table");
        // 1088 is a multiple of 64; 1056 is a multiple of 32 but not 64, so
        // halving it lands off the VAE's spatial stride.
        assert!(!1056usize.is_multiple_of(64) && 1056usize.is_multiple_of(32));
        assert!(!should_two_stage(big, 1920, 1056, true, None), "an axis that cannot be halved onto the 32-pixel stride must stay single-stage");
        assert!(!should_two_stage(big, 1888, 1088, true, None), "the width axis has the same requirement as the height axis");
        assert!(should_two_stage(64, 1920, 1088, false, Some(true)), "forcing two stages overrides the token test");
        assert!(!should_two_stage(big, 1920, 1088, true, Some(false)), "forcing one stage overrides it the other way");
        assert!(!should_two_stage(big, 1920, 1056, true, Some(true)), "no override makes an axis halvable that is not");
    }

    #[test]
    fn gen_opts_defaults_are_a_representable_tiny_clip() {
        let o = GenOpts::default();
        let vcfg = LtxVaeConfig::conv25();
        assert!(vcfg.latent_frames(o.frames as u32).is_some(), "{} frames must be 1+8k", o.frames);
        assert!(o.width.is_multiple_of(32) && o.height.is_multiple_of(32));
        assert_eq!(o.dit_config, "tiny");
    }

    /// Every role comes from the resolved assembly, absent ones as `None`,
    /// and an explicitly named spatial upscaler outranks the store's.
    #[test]
    fn paths_are_the_assemblys_roles() {
        let roles: std::collections::BTreeMap<String, std::path::PathBuf> =
            [("vae", "/s/vae"), ("dit", "/s/dit.gguf"), ("spatial_upsampler", "/s/up.st"), ("temporal_upsampler", "/s/tup.st")].iter().map(|(r, p)| (r.to_string(), p.into())).collect();
        let a = capability::Assembly { id: "local/ltxv".into(), arch: "ltxv".into(), variant: None, roles, provenance: Vec::new() };
        let p = Paths::from_assembly(&a, None).unwrap();
        assert_eq!((p.vae.as_str(), p.dit.as_deref(), p.spatial_upsampler.as_deref(), p.temporal_upsampler.as_deref()), ("/s/vae", Some("/s/dit.gguf"), Some("/s/up.st"), Some("/s/tup.st")));
        assert_eq!((p.text_encoder, p.audio_vae, p.duration_head, p.detailing_lora), (None, None, None, None));
        assert_eq!(Paths::from_assembly(&a, Some("/flag/up.st")).unwrap().spatial_upsampler.as_deref(), Some("/flag/up.st"));
    }

    #[test]
    fn a_bad_frame_count_is_rejected_before_any_weight_is_read() {
        let paths = Paths::new("/nope");
        let o = GenOpts { frames: 8, ..GenOpts::default() };
        let e = generate(&paths, "x", &o, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("1 + 8k"), "{e}");

        let o = GenOpts { width: 65, ..GenOpts::default() };
        let e = generate(&paths, "x", &o, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("multiple of 32"), "{e}");
    }

    /// An audio request that cannot possibly produce sound must be refused in
    /// milliseconds, naming the ONE thing that is missing.
    ///
    /// Not a nicety: the audio path opens the checkpoint and fills a host
    /// weight cache before it can discover any of this, so a check that ran
    /// later would cost minutes to say "no text encoder". Each
    /// prerequisite is asserted separately, because a single "audio needs more
    /// setup" message would send a caller looking in the wrong place.
    #[test]
    fn an_audio_request_that_cannot_produce_sound_is_refused_before_any_weight_is_read() {
        let audio = GenOpts { audio: true, dit_config: "ltx25_22b".into(), ..GenOpts::default() };
        let bare = Paths::new("/nope");

        // Wrong config: the tiny DiT's audio stream is random weights.
        let tiny = GenOpts { dit_config: "tiny".into(), ..audio.clone() };
        let e = generate(&bare, "x", &tiny, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("real audio-visual checkpoint"), "{e}");

        // No DiT checkpoint.
        let e = generate(&bare, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("--dit"), "{e}");

        // An audio source is the audio stream: without --audio it has none
        // to be.
        let source = crate::audio::AudioClip { channels: vec![vec![0.0; 1600]], sample_rate: 16_000 };
        let silent_with_source = GenOpts { audio: false, audio_source: Some(source), ..audio.clone() };
        let e = generate(&bare, "x", &silent_with_source, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("--audio"), "{e}");

        // A masked pass regenerates what ONE of a mask sequence or a window
        // marks, and holding a soundtrack needs one to hold.
        let clip = Video { width: 32, height: 32, fps: 8, frames: vec![vec![0u8; 32 * 32 * 3]; 9], audio: None };
        let both = MaskedOpts { mask_dir: "/m".into(), window: Some((0.0, 1.0)), ..Default::default() };
        let e = generate_masked(&bare, "x", &clip, &both, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("exactly one"), "{e}");

        // No audio VAE - the sound would have nothing to decode it.
        let with_dit = Paths { dit: Some("/dit".into()), ..bare.clone() };
        let e = generate(&with_dit, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("--audio-vae"), "{e}");

        // No text encoder - the audio stream has its OWN text projection, so
        // the stub context cannot stand in for it the way it can for video.
        let with_av = Paths { audio_vae: Some("/avae".into()), ..with_dit.clone() };
        let e = generate(&with_av, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("--text-encoder"), "{e}");
        assert!(e.contains("audio_aggregate_embed"), "the message must name WHY the stub cannot stand in: {e}");

        // And with every prerequisite named, the refusal is gone: the run
        // proceeds far enough to fail on the bogus paths themselves, which is
        // a different error entirely.
        let full = Paths { text_encoder: Some("/te".into()), ..with_av };
        let e = generate(&full, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("the fake paths still fail");
        assert!(!e.contains("put it in the model store"), "no prerequisite should still be reported missing: {e}");
    }

    /// [`av_step`] is the ONE place a denoise step's two streams are mapped
    /// onto the sixteen fields both audio-visual arms read, so a swap between
    /// them is a swap in every arm at once - and a swap between two same-typed
    /// slices produces a plausible clip, not an error.
    ///
    /// Every field is given a value distinguishable from every other, so the
    /// assertions below are identity checks rather than shape checks. The
    /// two contexts and the two position sets are the pairs that matter: the
    /// audio stream's connector is built for a NARROWER head than the video
    /// stream's, and the two streams' RoPE spaces have different axis counts.
    #[test]
    fn an_av_step_carries_each_streams_own_inputs() {
        let (v_latent, a_latent) = (vec![1.0f32, 1.5], vec![2.0f32]);
        let (v_ts, a_ts) = (vec![3.0f32, 3.5], vec![4.0f32]);
        let (v_pos, a_pos) = (vec![5.0f32; 12], vec![6.0f32; 2]);
        let kf = vec![7.0f32, 7.5];
        let (v_valid, a_valid) = (vec![8.0f32; 4], vec![8.5f32; 4]);
        let (v_ctx, a_ctx) = (vec![9.0f32; 4], vec![10.0f32; 3]);
        let i = StepInputs { latent: &v_latent, timesteps: &v_ts, positions: &v_pos, keyframes_mask: &kf, context_len: 4, t: 2, sigma: 0.25, perturbations: &crate::block::NO_PERTURBATIONS, reference_attention: None };
        let a = AudioInputs { latent: &a_latent, timesteps: &a_ts, positions: &a_pos, ta: 1, sigma: 0.0 };
        let s = av_step(&i, &a, Caption::of(&v_ctx, &v_valid), Caption::of(&a_ctx, &a_valid));

        assert_eq!(s.v_latent, &v_latent[..], "the video stream must carry the VIDEO latent");
        assert_eq!(s.a_latent, &a_latent[..], "the audio stream must carry the AUDIO latent");
        assert_eq!(s.v_timesteps, &v_ts[..]);
        assert_eq!(s.a_timesteps, &a_ts[..]);
        assert_eq!(s.v_positions, &v_pos[..]);
        assert_eq!(s.a_positions, &a_pos[..]);
        assert_eq!(s.v_keyframes_mask, &kf[..]);
        assert_eq!(s.v_context, &v_ctx[..], "the video stream must carry the VIDEO text projection");
        assert_eq!(s.a_context, &a_ctx[..], "the audio stream must carry the AUDIO text projection");
        assert_eq!(s.v_context_valid, &v_valid[..]);
        assert_eq!(s.a_context_valid, &a_valid[..]);
        assert_eq!((s.tv, s.ta), (2, 1));
        assert_eq!((s.v_context_len, s.a_context_len), (4, 4));
        // Each stream carries its own sigma: each stream's cross-modal gate
        // reads the OTHER one's, and a frozen stream's is 0.
        assert_eq!((s.v_sigma, s.a_sigma), (0.25, 0.0));
    }

    /// A wholly held stream reads as clean (`Modality.sigma = 0` for a frozen
    /// state); a partly held one, or one with nothing held, keeps the
    /// schedule's sigma.
    #[test]
    fn a_wholly_frozen_stream_reads_as_sigma_zero() {
        assert_eq!(stream_sigma(0.7, Some(&[0.0, 0.0])), 0.0);
        assert_eq!(stream_sigma(0.7, Some(&[0.0, 1.0])), 0.7f64 as f32);
        assert_eq!(stream_sigma(0.7, None), 0.7f64 as f32);
    }

    /// The banner a run prints and the object `build_denoiser` returns must
    /// name the same arm. `av_arm_label` is the shared pure function, so this
    /// pins its four cases; `build_denoiser` routing on the same predicate is
    /// what makes pinning the label meaningful.
    #[test]
    fn the_arm_label_names_the_arm_that_will_run() {
        assert!(av_arm_label(false, false, false).contains("tiny"));
        assert!(av_arm_label(false, true, true).contains("tiny"), "the tiny DiT has no audio arm to choose between");
        let video = av_arm_label(true, false, false);
        assert!(video.contains("int8") && video.contains("video stream only"), "{video}");
        let quant = av_arm_label(true, true, false);
        assert!(quant.contains("int8") && quant.contains("both streams"), "{quant}");
        let fp32 = av_arm_label(true, true, true);
        assert!(fp32.contains("fp32") && fp32.contains("both streams"), "{fp32}");
        assert_ne!(quant, fp32, "the two audio-visual arms must be distinguishable in the banner");
    }

    /// The host-memory guard has to follow the arm switch. The fp32 reference
    /// arm needs the whole manifest expanded to f32; the quantized arm holds
    /// one int8 copy of each block plus the fp32 head tensors, and streams the
    /// rest off the mapped checkpoint. A guard that quoted the fp32
    /// requirement on the quantized path would refuse machines that can run it
    /// comfortably.
    #[test]
    fn the_quantized_arm_is_budgeted_against_its_own_footprint() {
        let cfg = LtxAvDitConfig::ltx25();
        let fp32 = crate::av_stream::host_floats(&cfg) * 4;
        let quant = quantized_av_host_bytes(&cfg);
        assert!(quant < fp32 / 2, "the quantized arm must be budgeted well below the fp32 expansion ({quant} vs {fp32})");
        assert!(quant > 0);
        // And a tiny config must never be refused on either arm, on any
        // machine that can run this test at all.
        let tiny = LtxAvDitConfig::tiny();
        check_av_host_memory(&tiny, false).expect("a tiny AV config must fit on the quantized arm anywhere");
        check_av_host_memory(&tiny, true).expect("a tiny AV config must fit on the fp32 arm anywhere");
    }

    #[test]
    fn grid_positions_are_frame_major_width_minor_integer_bounds() {
        let p = grid_positions(2, 1, 2); // T = 4 tokens
        // token order: (f=0,h=0,w=0), (f=0,h=0,w=1), (f=1,h=0,w=0), (f=1,h=0,w=1)
        let t = 4;
        let get = |axis: usize, tok: usize| (p[(axis * t + tok) * 2], p[(axis * t + tok) * 2 + 1]);
        assert_eq!(get(0, 0), (0.0, 1.0)); // frame axis, token 0
        assert_eq!(get(0, 2), (1.0, 2.0)); // frame axis, token 2 (f=1)
        assert_eq!(get(2, 1), (1.0, 2.0)); // width axis, token 1 (w=1)
        assert_eq!(get(2, 2), (0.0, 1.0)); // width axis, token 2 (w=0)
    }

    #[test]
    fn chw_tc_round_trip_and_transpose_correctly() {
        let (c, t, h, w) = (3usize, 2usize, 2usize, 2usize);
        let n_tok = t * h * w;
        let chw: Vec<f32> = (0..c * n_tok).map(|i| i as f32).collect();
        let tc = chw_to_tc(&chw, c, t, h, w);
        assert_eq!(tc.len(), chw.len());
        // token 0's channel vector must be [chw[0], chw[n_tok], chw[2*n_tok]].
        assert_eq!(&tc[0..c], &[chw[0], chw[n_tok], chw[2 * n_tok]]);
        let back = tc_to_chw(&tc, c, t, h, w);
        assert_eq!(back, chw);
    }

    #[test]
    fn context_stub_is_seeded_and_prompt_independent_at_the_function_level() {
        let a = context_stub(4, 8, 42);
        assert_eq!(a, context_stub(4, 8, 42), "same seed, same stub");
        assert_ne!(a, context_stub(4, 8, 43), "different seed, different stub");
        assert_eq!(a.len(), 32);
    }

    /// Records which context each forward saw (by its first element - the
    /// two stubs used here are constant vectors so this fully identifies
    /// which branch ran), and answers with a constant vector - so the loop's
    /// CFG bookkeeping is observable without a real DiT.
    #[derive(Default)]
    struct FakeDit {
        seen: std::cell::RefCell<Vec<f32>>,
        /// Every `timesteps` slice the loop handed the model, in call order -
        /// what `timesteps_are_the_denoise_mask_times_sigma` inspects.
        timesteps_seen: std::cell::RefCell<Vec<Vec<f32>>>,
        /// Every `latent` the loop handed the model - i.e. what the previous
        /// step's sampler produced, which is the only window a weight-free
        /// test has onto the step ORDER.
        latents_seen: std::cell::RefCell<Vec<Vec<f32>>>,
    }
    impl Denoiser for FakeDit {
        fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
            self.seen.borrow_mut().push(context.rows[0]);
            self.timesteps_seen.borrow_mut().push(i.timesteps.to_vec());
            self.latents_seen.borrow_mut().push(i.latent.to_vec());
            vec![context.rows[0]; i.latent.len()]
        }
    }

    /// LTX-2.5's distilled stage 1 is `euler_ancestral_denoising_loop` at
    /// `eta = 1.0` (`ltx_pipelines.distilled`: `ANCESTRAL_SAMPLER_SINCE_
    /// VERSION = (2, 5)`, `ANCESTRAL_ETA = 1.0`), and image conditioning has
    /// to work UNDER that sampler, not only under the deterministic one.
    /// `samplers._ancestral_euler_denoising_loop` re-applies
    /// `post_process_latent` to the STEPPED latent - after the renoise term -
    /// so a conditioned token is clean again at the top of the next step.
    /// Skip that and every conditioned token carries a full step's worth of
    /// freshly injected noise into the next forward while still being
    /// announced at timestep 0.
    #[test]
    fn the_ancestral_sampler_re_pins_conditioned_tokens_after_the_renoise_term() {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit::default();
        let (t, channels) = (2usize, 1usize);
        let positions = grid_positions(t, 1, 1);
        let keyframes_mask = vec![0.0f32; t];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        // Token 0 denoises; token 1 is conditioned, and its clean content is
        // a value no arithmetic in the step could land on by accident.
        let mask = vec![1.0f32, 0.0];
        let clean = vec![0.0f32, 5.0];
        let frozen = Frozen { mask: &mask, clean: &clean, channels, attention: None };
        // `GaussianNoiser`: a fully conditioned token starts AT its clean
        // content, never at noise.
        let latent0 = vec![0.0f32, 5.0];

        let out = denoise(&dit, &sigmas, latent0, &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(1.0, 1.0, 7), 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("eta > 0 with conditioned tokens must run, not be refused");

        let latents = dit.latents_seen.borrow().clone();
        assert_eq!(latents.len(), 2, "one forward per step");
        assert_eq!(latents[0][1], 5.0, "step 0 sees the conditioned token at its clean content");
        assert_eq!(latents[1][1], 5.0, "step 1 must see it clean AGAIN: the renoise term ran over the whole latent and post_process_latent has to undo it there");
        assert_ne!(latents[1][0], latents[0][0], "the free token really was stepped and renoised");
        // The terminal step short-circuits to the raw x0 estimate with no
        // re-pin, exactly as `_ancestral_euler_denoising_loop` does. That is
        // safe for a frozen token only because the x0 conversion runs at that
        // token's OWN timestep (zero), where it is the identity - see
        // `a_frozen_token_survives_the_terminal_step_exactly`, which pins the
        // resulting value.
        assert_eq!(out.len(), t);
        assert_eq!(out[1], 5.0, "the conditioned token ends at its clean content even though the terminal step never re-pinned it");
    }

    /// A PARTIALLY conditioned token (`--conditioning-strength` below 1, so
    /// `denoise_mask` is strictly between 0 and 1) has to be pulled toward its
    /// clean content by exactly the reference's amount under BOTH samplers.
    ///
    /// `samplers._ModalityStep.from_modality_result` runs `post_process_latent`
    /// on the model's x0 ESTIMATE for every ancestral step, terminal one
    /// included - the same masking `_step_state` does for the deterministic
    /// loop - and the ancestral loop THEN masks the stepped latent again after
    /// the renoise term. Both, not one or the other. The distinction is
    /// invisible at `mask == 0` (the x0 conversion runs at that token's own
    /// zero timestep, so the estimate already IS the clean content) and at
    /// `mask == 1` (the blend is the identity), which is why every earlier
    /// gate here passes either way; it is a real difference for every strength
    /// in between, and `eta = 1.0` is this pipeline's default.
    #[test]
    fn a_partially_conditioned_token_is_pulled_to_its_clean_content_under_both_samplers() {
        // One terminal step, so what comes out IS the masked x0 estimate with
        // no further arithmetic to hide a missing blend.
        let sigmas = vec![0.5, 0.0];
        let (t, channels) = (2usize, 1usize);
        let positions = grid_positions(t, 1, 1);
        let keyframes_mask = vec![0.0f32; t];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        // Token 0 denoises freely; token 1 is conditioned at strength 0.5.
        let mask = vec![1.0f32, 0.5];
        let clean = vec![0.0f32, 10.0];
        let frozen = Frozen { mask: &mask, clean: &clean, channels, attention: None };
        let latent0 = vec![0.0f32, 10.0];
        // `FakeDit` predicts velocity 1.0 everywhere, and token 1's own
        // timestep is `0.5 * 0.5`, so its x0 estimate is `10 - 0.25`; the
        // reference's blend then lands it half way back to `clean`.
        let want = 0.5 * (10.0 - 0.25) + 0.5 * 10.0;

        for eta in [0.0f64, 1.0] {
            let dit = FakeDit::default();
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(eta, 1.0, 7), 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
            assert_eq!(out[1], want, "eta={eta}: a half-conditioned token must end half way between its x0 estimate and its clean content");
        }
    }

    /// A fully frozen conditioning token must come out of the whole sampler
    /// bit-exactly equal to its clean content - **including on the terminal
    /// step**, which is the one step that never re-pins.
    ///
    /// This is not a property of `post_process_latent`; it is a property of
    /// the velocity -> x0 conversion. The reference does that conversion
    /// INSIDE the model wrapper (`ltx_core.model.transformer.model.X0Model.
    /// forward`: `to_denoised(video.latent, vx, video.timesteps)`) against the
    /// PER-TOKEN `timesteps` tensor - `denoise_mask * sigma`, so `0` on a
    /// frozen token - not against the schedule's scalar sigma. At timestep 0
    /// the conversion is the identity, so the anchor survives the terminal
    /// step untouched.
    ///
    /// Using the scalar sigma instead is invisible in plain text-to-video
    /// (every token's timestep IS the sigma there) and silently multiplies a
    /// frozen anchor by roughly `1 + sigma_terminal` - the real distilled
    /// schedule's last sigma is 0.421875, so a real `--start-frame` clip's
    /// anchor latent came out too large by exactly that factor, which the causal VAE
    /// decoder's temporal receptive field then smeared across the frames
    /// after it. See this phase's ledger entry for the measured curves.
    #[test]
    fn a_frozen_token_survives_the_terminal_step_exactly() {
        // Ends on the real distilled schedule's own terminal pair, so the
        // number this test would be wrong by is the number a real run was
        // wrong by.
        let sigmas = vec![1.0, 0.421875, 0.0];
        let dit = FakeDit::default();
        let (t, channels) = (2usize, 1usize);
        let positions = grid_positions(t, 1, 1);
        let keyframes_mask = vec![0.0f32; t];
        // A velocity of 1.0 on every token (`FakeDit` echoes `context.rows[0]`) -
        // the model does NOT conveniently predict zero at a frozen token, and
        // the fix must not depend on it doing so.
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let mask = vec![1.0f32, 0.0];
        let clean = vec![0.0f32, 5.0];
        let frozen = Frozen { mask: &mask, clean: &clean, channels, attention: None };
        let latent0 = vec![0.0f32, 5.0];

        for eta in [0.0f64, 1.0] {
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(eta, 1.0, 7), 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
            assert_eq!(out[1], 5.0, "eta={eta}: the frozen anchor must end at exactly its clean content, not at clean - velocity*sigma_terminal ({})", 5.0 - 0.421875);
            assert_ne!(out[0], 0.0, "eta={eta}: the free token really was denoised (guards against a fix that froze everything)");
        }
    }

    /// **A frozen range survives the whole trajectory bit-identically
    /// wherever it sits in the sequence** - at the head, in the interior, or
    /// past the end of the generated video - on every token and every channel.
    ///
    /// The head case is the long-form seam, stated as the property that makes
    /// it one. It is the half of the continuity chain that lives in the
    /// sampler; the other half - that what goes in is a verbatim slice of the
    /// previous window's own final latent - is
    /// `crates/ltxv/tests/it/longform.rs`'s
    /// `the_carried_tail_is_the_previous_windows_own_last_latent_frames`.
    /// Together: window `n`'s last K latent frames ARE window `n + 1`'s first
    /// K, with no decode, no re-encode and no drift, which is what
    /// [`generate_long`] means by carrying latent context rather than a
    /// re-encoded picture.
    ///
    /// The APPENDED case is `--end-frame`'s and `--mid-frame`'s guiding
    /// blocks, and the INTERIOR case is the one nothing else covers: every
    /// frozen range this crate had before sat at one of the two ends of the
    /// sequence, so a step that walked a range from the wrong end, or that
    /// only ever re-pinned a prefix, would have passed. Position is the whole
    /// variable here - the sampler's own arithmetic is per token and must not
    /// know where in the sequence a token lives.
    ///
    /// Deliberately wider than
    /// [`a_frozen_token_survives_the_terminal_step_exactly`](self): several
    /// frames, several tokens per frame and several channels, so a fix that
    /// happened to hold for one scalar token - or that transposed a range -
    /// cannot pass.
    #[test]
    fn a_frozen_range_survives_the_whole_trajectory_wherever_it_sits() {
        let sigmas = vec![1.0, 0.421875, 0.0];
        // A 4-latent-frame window on a 1x2 grid (2 tokens per frame, 3
        // channels), plus one appended guiding block of the same width - the
        // shape `conditioned_latent` produces for a clip with one still.
        let (lh, lw, channels) = (1usize, 2usize, 3usize);
        let (lat_t, block_t) = (4usize, lh * lw);
        let base_t = lat_t * block_t;
        let t = base_t + block_t;
        let base_positions = real_pixel_positions(lat_t, lh, lw, 8.0);
        let mut base_km = vec![0.0f32; base_t];
        base_km[..block_t].fill(1.0);
        let guide = vec![0.0f32; block_t * channels];
        let ic = append_image_conditioning(base_t, &base_positions, &base_km, lh, lw, channels, 8.0, 0.0, &[(4, &guide)]);
        let (positions, keyframes_mask) = (ic.positions, ic.keyframes_mask);
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];

        // Head: a 2-latent-frame carried context. Interior: one latent frame
        // in the middle of the generated video. Tail: the appended guiding
        // block. Nothing in the sampler may distinguish them.
        let frozen_ranges = [(0usize, 2 * block_t), (3 * block_t, 4 * block_t), (t - block_t, t)];
        let mut mask = vec![1.0f32; t];
        let mut clean = vec![0.0f32; t * channels];
        for &(lo, hi) in &frozen_ranges {
            mask[lo..hi].fill(0.0);
            // Every frozen value distinct, so an off-by-one or a transpose
            // inside a range cannot come back equal.
            for (i, v) in clean[lo * channels..hi * channels].iter_mut().enumerate() {
                *v = 100.0 + (lo * channels + i) as f32;
            }
        }
        let frozen = Frozen { mask: &mask, clean: &clean, channels, attention: None };
        let mut latent0 = seeded_noise(t * channels, 11);
        for &(lo, hi) in &frozen_ranges {
            latent0[lo * channels..hi * channels].copy_from_slice(&clean[lo * channels..hi * channels]);
        }

        let dit = FakeDit::default();
        for eta in [0.0f64, 1.0] {
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(eta, 1.0, 7), 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
            for &(lo, hi) in &frozen_ranges {
                assert_eq!(&out[lo * channels..hi * channels], &clean[lo * channels..hi * channels], "eta={eta}: the frozen range [{lo}, {hi}) must come out exactly as it went in");
            }
            let free: Vec<usize> = (0..t).filter(|tok| mask[*tok] == 1.0).collect();
            assert!(free.iter().any(|&tok| out[tok * channels..(tok + 1) * channels] != latent0[tok * channels..(tok + 1) * channels]), "eta={eta}: the free tokens really were denoised");
        }
        // Every step announced the frozen tokens at timestep 0 and everything
        // else at the schedule's sigma - the AdaLN modulation the free tokens
        // are generated under depends on it.
        for ts in dit.timesteps_seen.borrow().iter() {
            for (tok, &m) in mask.iter().enumerate() {
                if m == 0.0 {
                    assert_eq!(ts[tok], 0.0, "a frozen token was announced as noisy");
                } else {
                    assert!(ts[tok] > 0.0, "a free token was announced as clean");
                }
            }
        }
    }

    /// The per-token conversion degenerates to the scalar one when nothing is
    /// frozen: `timesteps_from_mask(ones, sigma)` IS `sigma` everywhere, so
    /// the unconditioned trajectory must be bit-identical to what it was
    /// before the per-token x0 conversion existed.
    #[test]
    fn to_denoised_is_per_token_and_collapses_to_the_scalar_form_when_nothing_is_frozen() {
        let latent = [1.0f32, 2.0, 3.0, 4.0];
        let velocity = [0.5f32, 0.5, 0.5, 0.5];
        // channels = 2, so two tokens of two channels each.
        let uniform = to_denoised(&latent, &velocity, &[0.25, 0.25], 2);
        assert_eq!(uniform, vec![0.875f32, 1.875, 2.875, 3.875], "one timestep for every token is the plain `sample - velocity*sigma`");
        // Token 0 frozen (timestep 0), token 1 at sigma - the image-
        // conditioning case, and the whole point: the frozen token passes
        // through untouched while its neighbour is still corrected.
        let mixed = to_denoised(&latent, &velocity, &[0.0, 0.25], 2);
        assert_eq!(mixed, vec![1.0f32, 2.0, 2.875, 3.875], "a timestep-0 token is returned unchanged, channel by channel");
    }

    /// The reference builds `Modality.timesteps` as `timesteps_from_mask(
    /// denoise_mask, sigma)` (`ltx_pipelines.utils.helpers`, via
    /// `modality_from_latent_state`), NOT as the schedule's sigma broadcast
    /// uniformly. A frozen image-conditioning token carries clean VAE
    /// content, so the model must be told its noise level is zero - the
    /// DiT's AdaLN builds one modulation row PER TOKEN from this value, so a
    /// clean token mislabelled "fully noisy" corrupts its own modulation and,
    /// through self-attention, the whole sequence.
    #[test]
    fn timesteps_are_the_denoise_mask_times_sigma() {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit::default();
        let (t, channels) = (4usize, 1usize);
        let positions = grid_positions(t, 1, 1);
        let keyframes_mask = vec![0.0f32; t];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        // Tokens 2 and 3 are a frozen conditioning block; 0 and 1 denoise.
        let mask = vec![1.0f32, 1.0, 0.0, 0.0];
        let clean = vec![0.0f32; t * channels];
        let frozen = Frozen { mask: &mask, clean: &clean, channels, attention: None };
        denoise(&dit, &sigmas, vec![0.0; t * channels], &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(0.0, 1.0, 7), 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");

        let seen = dit.timesteps_seen.borrow().clone();
        assert_eq!(seen.len(), 2, "one forward per step at guidance <= 1");
        assert_eq!(seen[0], vec![1.0, 1.0, 0.0, 0.0], "step 0 (sigma=1.0): mask * sigma");
        assert_eq!(seen[1], vec![0.5, 0.5, 0.0, 0.0], "step 1 (sigma=0.5): mask * sigma - the frozen block stays at zero for the whole schedule");
    }

    /// With nothing frozen, `timesteps_from_mask` degenerates to the
    /// schedule's sigma on every token - the unconditioned path must be
    /// byte-identical to what it was before per-token timesteps existed.
    #[test]
    fn unconditioned_timesteps_are_the_scalar_sigma_on_every_token() {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit::default();
        let t = 3usize;
        let positions = grid_positions(t, 1, 1);
        let keyframes_mask = vec![0.0f32; t];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        denoise(&dit, &sigmas, vec![0.0; t], &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(0.0, 1.0, 7), 4, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
        assert_eq!(*dit.timesteps_seen.borrow(), vec![vec![1.0f32; t], vec![0.5f32; t]]);
    }

    /// A model whose x0 estimate is `target` wherever the token is noisy:
    /// velocity `(x - target) / timestep`, and `0` on a timestep-0 (frozen)
    /// token. On the audio stream the target is `-target`.
    struct ConstantX0 {
        target: f32,
        timesteps_seen: std::cell::RefCell<Vec<f32>>,
    }
    impl ConstantX0 {
        fn new(target: f32) -> Self {
            Self { target, timesteps_seen: Default::default() }
        }
        fn velocity(latent: &[f32], timesteps: &[f32], target: f32) -> Vec<f32> {
            let channels = latent.len() / timesteps.len();
            latent.iter().enumerate().map(|(k, &x)| match timesteps[k / channels] {
                0.0 => 0.0,
                ts => (x - target) / ts,
            }).collect()
        }
    }
    impl Denoiser for ConstantX0 {
        fn forward(&self, i: &StepInputs, _context: Caption) -> Vec<f32> {
            self.timesteps_seen.borrow_mut().push(i.sigma);
            Self::velocity(i.latent, i.timesteps, self.target)
        }
        fn has_audio(&self) -> bool {
            true
        }
        fn forward_av(&self, i: &StepInputs, a: &AudioInputs, _context: Caption, _a_context: Caption) -> (Vec<f32>, Vec<f32>) {
            self.timesteps_seen.borrow_mut().push(i.sigma);
            (Self::velocity(i.latent, i.timesteps, self.target), Self::velocity(a.latent, a.timesteps, -self.target))
        }
    }

    /// A text-to-audio request that cannot run is refused before any weight
    /// is read: a length that is not a positive number of seconds, a
    /// soundtrack to hold when the soundtrack is what is asked for, and any
    /// missing checkpoint (the paths below name nothing that exists).
    #[test]
    fn a_text_to_audio_request_that_cannot_run_is_refused_up_front() {
        let paths = Paths::new("/missing/vae");
        let run = |o: AudioOpts| generate_audio(&paths, "rain", &o, &Default::default(), |_, _, _| {}).map(|_| ()).unwrap_err();
        for seconds in [0.0, -2.0, f64::NAN] {
            assert!(run(AudioOpts { seconds: Some(seconds), ..AudioOpts::default() }).contains("positive number of seconds"));
        }
        let held = AudioOpts { base: GenOpts { audio_source: Some(crate::audio::AudioClip { channels: vec![vec![0.0; 16]], sample_rate: 16_000 }), ..AudioOpts::default().base }, ..AudioOpts::default() };
        assert!(run(held).contains("nothing left to generate"));
        assert!(run(AudioOpts::default()).contains("real DiT checkpoint"));
    }

    /// An extension holds the clip's last 57 frames (the reference recipe's
    /// 8 latent frames, fewer for a shorter clip, always on the `8k + 1`
    /// grid), adds whole latent frames, and regenerates exactly the latent
    /// frames past the held ones: the window's start is the first added
    /// frame's time, so no held latent frame overlaps it.
    #[test]
    fn an_extension_regenerates_exactly_the_latent_frames_past_the_held_tail() {
        let o = |frames, context_frames| ExtendOpts { frames, context_frames, ..ExtendOpts::default() };
        let plan = extension_plan(121, &o(20, crate::longform::CONTEXT_FRAMES), 24).unwrap();
        assert_eq!((plan.context, plan.added), (57, 24), "20 frames round up to three latent frames");
        assert_eq!(extension_plan(30, &o(8, 57), 24).unwrap().context, 25, "a 30-frame clip holds its last 25");
        assert!(extension_plan(0, &o(8, 57), 24).is_err());
        assert!(extension_plan(30, &o(0, 57), 24).is_err());

        let (lh, lw) = (2, 3);
        let lat_t = (plan.context + plan.added - 1) / VAE_TEMPORAL_SCALE + 1;
        let t = lat_t * lh * lw;
        let region = crate::maskcond::temporal_region(&real_pixel_positions(lat_t, lh, lw, 24.0), t, plan.window.0, plan.window.1);
        let held_latents = (plan.context - 1) / VAE_TEMPORAL_SCALE + 1;
        for (tok, &m) in region.iter().enumerate() {
            let latent_frame = tok / (lh * lw);
            assert_eq!(m, if latent_frame < held_latents { 0.0 } else { 1.0 }, "latent frame {latent_frame}");
        }
    }

    /// A request that names nothing runs the real model at the reference's
    /// own size, length and rate when the checkpoint is configured, and the
    /// tiny smoke test only when it is not; whatever a request names is kept.
    #[test]
    fn unnamed_settings_take_the_production_defaults_with_a_real_checkpoint() {
        let bare = Paths::new("/vae");
        let real = Paths { dit: Some("/dit.gguf".into()), ..bare.clone() };
        let mut o = GenOpts::default();
        resolve_defaults(&mut o, &bare, Named::default());
        assert_eq!((o.dit_config.as_str(), o.frames, o.width), ("tiny", GenOpts::default().frames, GenOpts::default().width));
        let mut o = GenOpts::default();
        resolve_defaults(&mut o, &real, Named::default());
        assert_eq!((o.dit_config.as_str(), o.width, o.height, o.frames, o.fps), ("ltx25_22b", 768, 512, 121, 24));
        let mut o = GenOpts { frames: 17, width: 256, height: 192, ..GenOpts::default() };
        resolve_defaults(&mut o, &real, Named { frames: true, size: true, ..Named::default() });
        assert_eq!((o.frames, o.width, o.height, o.fps), (17, 256, 192, 24));
        let mut o = GenOpts::default();
        resolve_defaults(&mut o, &real, Named { dit_config: true, ..Named::default() });
        assert_eq!(o.dit_config, "tiny", "an explicit --dit-config tiny stays the smoke test");
    }

    fn res2s(eta: f64, seed: u64) -> Sampling {
        Sampling { sampler: Sampler::Res2s, eta, s_noise: 1.0, seed }
    }

    /// `res2s_audio_video_denoising_loop`'s evaluation schedule: per step one
    /// evaluation at the step's sigma and one at the geometric midpoint
    /// `sqrt(sigma * sigma_next)`, and - for a schedule ending at 0 - the
    /// last sigma moved to `0.0011` and one final evaluation there.
    #[test]
    fn res2s_evaluates_each_step_at_its_sigma_and_midpoint_then_once_at_the_end() {
        let dit = ConstantX0::new(0.5);
        let (positions, keyframes_mask, context_valid) = (grid_positions(1, 1, 1), vec![0.0f32; 1], vec![1.0f32; 1]);
        denoise(&dit, &[1.0, 0.5, 0.0], vec![0.3; 2], &positions, &keyframes_mask, Caption::of(&[1.0], &context_valid), Caption::of(&[0.0], &context_valid), 1, 1, &crate::guidance::Guidance::cfg(1.0), res2s(0.5, 7), 2, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("res2s runs");
        let want = [1.0, 0.5f64.sqrt(), 0.5, (0.5f64 * 0.0011).sqrt(), 0.0011];
        let seen = dit.timesteps_seen.borrow();
        assert_eq!(seen.len(), want.len(), "{seen:?}");
        for (s, w) in seen.iter().zip(want) {
            assert!((*s as f64 - w).abs() < 1e-6, "{seen:?} vs {want:?}");
        }
    }

    /// Driven by a model whose x0 never moves, the exponential integrator
    /// lands every noisy token on it whatever noise the SDE terms injected
    /// along the way (small steps, so bong iteration runs too), holds a
    /// frozen token at its clean content bit-exactly, and steps the audio
    /// stream beside the video toward its own x0.
    #[test]
    fn res2s_lands_on_the_models_x0_and_holds_frozen_tokens() {
        let dit = ConstantX0::new(0.25);
        let (t, channels) = (3usize, 2usize);
        let positions = grid_positions(t, 1, 1);
        let (keyframes_mask, context_valid) = (vec![0.0f32; t], vec![1.0f32; 1]);
        let mask = vec![0.0f32, 1.0, 1.0];
        let clean = vec![2.0f32, -2.0, 0.0, 0.0, 0.0, 0.0];
        let frozen = Frozen { mask: &mask, clean: &clean, channels, attention: None };
        let latent0 = vec![2.0f32, -2.0, 0.7, -0.4, 1.3, 0.1];
        let (a_ctx, ta) = (vec![0.0f32; 1], 2usize);
        let mut audio = AudioState { latent: vec![0.9, -1.1, 0.2, 0.4], positions: crate::audio::positions(ta), ta, ctx_cond: &a_ctx, ctx_uncond: &a_ctx, frozen: None };
        let sigmas = [1.0, 0.9, 0.75, 0.6, 0.0];
        for eta in [0.0, 0.5, 1.0] {
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, Caption::of(&[1.0], &context_valid), Caption::of(&[0.0], &context_valid), 1, t, &crate::guidance::Guidance::cfg(1.0), res2s(eta, 11), 4, Some(&frozen), &Default::default(), Some(&mut audio), &mut |_, _, _: &str| {}).expect("res2s runs");
            assert_eq!(&out[..channels], &clean[..channels], "eta={eta}: the frozen token is the clean content exactly");
            for v in &out[channels..] {
                assert!((v - 0.25).abs() < 1e-5, "eta={eta}: {out:?}");
            }
            for v in &audio.latent {
                assert!((v + 0.25).abs() < 1e-5, "eta={eta}: audio {:?}", audio.latent);
            }
            audio.latent = vec![0.9, -1.1, 0.2, 0.4];
        }
    }

    /// One deterministic step (`eta = 0`, and `h = ln 2 >= 0.5` so no bong
    /// iteration) of a model whose x0 never moves is the exact solution of
    /// the flow ODE, `target + (sigma_next/sigma) * (x - target)`: the
    /// substep's noise only moves the point the second evaluation is taken
    /// at, and `b1 + b2 = phi1(-h)`.
    #[test]
    fn a_deterministic_res2s_step_solves_the_flow_ode_for_a_fixed_x0() {
        let dit = ConstantX0::new(0.25);
        let (positions, keyframes_mask, context_valid) = (grid_positions(2, 1, 1), vec![0.0f32; 2], vec![1.0f32; 1]);
        let x = vec![0.75f32, -1.75];
        let out = denoise(&dit, &[1.0, 0.5], x.clone(), &positions, &keyframes_mask, Caption::of(&[1.0], &context_valid), Caption::of(&[0.0], &context_valid), 1, 2, &crate::guidance::Guidance::cfg(1.0), res2s(0.0, 5), 1, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("res2s runs");
        for (o, x) in out.iter().zip(&x) {
            let want = 0.25 + 0.5 * (x - 0.25);
            assert!((o - want).abs() < 1e-6, "{out:?}");
        }
    }

    /// Two runs of one seed are identical, two seeds differ: the SDE noise is
    /// drawn from the run's seed and nowhere else.
    #[test]
    fn res2s_noise_is_reproducible_from_the_seed() {
        // A model that does not converge in one evaluation, so the noise
        // shows in the result.
        struct Halfway;
        impl Denoiser for Halfway {
            fn forward(&self, i: &StepInputs, _context: Caption) -> Vec<f32> {
                i.latent.iter().map(|&x| 0.5 * x).collect()
            }
        }
        let (positions, keyframes_mask, context_valid) = (grid_positions(4, 1, 1), vec![0.0f32; 4], vec![1.0f32; 1]);
        let run = |seed| denoise(&Halfway, &[1.0, 0.7, 0.4, 0.0], vec![0.5, -0.5, 1.0, -1.0], &positions, &keyframes_mask, Caption::of(&[1.0], &context_valid), Caption::of(&[0.0], &context_valid), 1, 4, &crate::guidance::Guidance::cfg(1.0), res2s(0.5, seed), 3, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("res2s runs");
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    fn run_loop(guidance: f32, eta: f64) -> (Vec<f32>, Vec<f32>) {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit::default();
        let positions = grid_positions(1, 1, 1);
        let keyframes_mask = vec![0.0f32];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let out = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, 1, &crate::guidance::Guidance::cfg(guidance), Sampling::euler(eta, 1.0, 7), 4, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
        let seen = dit.seen.borrow().clone();
        (seen, out)
    }

    /// The CFG fold is `uncond + g·(cond - uncond)` on the VELOCITY, and each
    /// branch must upload (here: pass) its own context - with the markers
    /// above (1.0 cond, 0.0 uncond) the forwards alternate 1,0,1,0 across the
    /// two steps, and the folded velocity is exactly `g`.
    #[test]
    fn cfg_runs_two_forwards_per_step_and_folds_them_on_the_velocity() {
        let (seen, _) = run_loop(5.0, 0.0);
        assert_eq!(seen, vec![1.0, 0.0, 1.0, 0.0], "each branch must see its own context");
    }

    /// `guidance <= 1.0` collapses to the conditional prediction exactly, so
    /// only ONE forward runs per step.
    #[test]
    fn guidance_of_one_runs_a_single_forward_per_step() {
        let (seen, _) = run_loop(1.0, 0.0);
        assert_eq!(seen, vec![1.0, 1.0], "one forward per step, always conditional");
    }

    /// DFR's two conditioned states against `tools/goldens/
    /// ltxv_dfr_conditioning_dump_reference.py`, which applies the official
    /// conditioning items to the reference's own initial states: stage 1's
    /// generated keyframe slots, and stage 2's slots seeded with keyframes
    /// plus the half-resolution reference latent. Composed as
    /// [`denoise_stage`] composes them - base grid, slots, reference - and
    /// compared before the noiser (`sigma0 = 0`), on every tensor the state
    /// carries.
    #[test]
    fn dfr_stage_conditioning_matches_the_reference() {
        let path = brain_testutil::testdata("golden/ltxv/dfr_conditioning/dfr_conditioning.safetensors");
        if !std::path::Path::new(&path).exists() {
            brain_testutil::skip(&format!("{path} absent - run tools/goldens/ltxv_dfr_conditioning_dump_reference.py"));
            return;
        }
        let fx = checkpoint::safetensors::read(&path).expect("read golden");
        let get = |n: &str| fx.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("no golden {n}")).data.clone();
        let close = |label: &str, got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
            let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst <= 1e-6, "{label}: max |diff| {worst}");
        };
        let (frames, fps, c) = (49usize, 24.0, 128usize);
        let (canvas, _, slot_frames) = crate::dfr::resolve_canvas(frames, crate::dfr::VIDEO_TEMPORAL_SCALE).unwrap();
        assert_eq!((canvas, slot_frames.as_slice()), (49, &[24usize, 48][..]));
        let lat_t = LtxVaeConfig::conv25().latent_frames(canvas as u32).unwrap() as usize;
        for (stage, (h, w)) in [("stage1", (64 / 2, 128 / 2)), ("stage2", (64, 128))] {
            let (lh, lw) = (h / 32, w / 32);
            let t = lat_t * lh * lw;
            let positions = real_pixel_positions(lat_t, lh, lw, fps);
            let mut keyframes_mask = vec![0f32; t];
            keyframes_mask[..lh * lw].fill(1.0);
            let seeded = stage == "stage2";
            let video_in = if seeded { chw_to_tc(&get("stage2.video_in"), c, lat_t, lh, lw) } else { vec![0f32; t * c] };
            let keyframes = seeded.then(|| get("stage2.keyframes_in"));
            let slots = KeyframeSlots { pixel_frames: &slot_frames, initial_chw: keyframes.as_deref() };
            let base = KeyframeSlotsInput { base_t: t, base_positions: &positions, base_keyframes_mask: &keyframes_mask, lh, lw, channels: c, fps };
            let ks = append_keyframe_slots(base, &slots, 0.0, 1).unwrap();
            let mut latent = video_in;
            latent.extend_from_slice(&ks.latent);
            let (mut positions, mut keyframes_mask, mut denoise_mask, mut clean) = (ks.positions, ks.keyframes_mask, vec![1f32; ks.total_t], vec![0f32; ks.total_t * c]);
            if seeded {
                let (rf, rh, rw) = (lat_t, lh / 2, lw / 2);
                let tokens = chw_to_tc(&get("stage2.reference_in"), c, rf, rh, rw);
                let rc = crate::refcond::append_reference_video_conditioning(ks.total_t, &positions, &keyframes_mask, c, fps, (rf, rh, rw), &tokens, 2, 1, 1.0, None);
                latent.resize(rc.keyframes_mask.len() * c, 0.0);
                denoise_mask.extend_from_slice(&rc.denoise_mask[ks.total_t..]);
                clean.extend_from_slice(&rc.clean[ks.total_t * c..]);
                (positions, keyframes_mask) = (rc.positions, rc.keyframes_mask);
            }
            close(&format!("{stage} positions"), &positions, &get(&format!("{stage}.positions")));
            close(&format!("{stage} keyframes_mask"), &keyframes_mask, &get(&format!("{stage}.keyframes_mask")));
            close(&format!("{stage} denoise_mask"), &denoise_mask, &get(&format!("{stage}.denoise_mask")));
            // Clean content is read only where a token is held (`1 - mask`
            // weights it in `post_process_latent`); the reference also keeps
            // the initial latent on fully denoised rows, which nothing reads.
            let held = |v: &[f32]| v.chunks(c).zip(&denoise_mask).flat_map(|(r, &m)| r.iter().map(move |x| x * (1.0 - m))).collect::<Vec<f32>>();
            close(&format!("{stage} clean_latent (held tokens)"), &held(&clean), &held(&get(&format!("{stage}.clean_latent"))));
            close(&format!("{stage} latent"), &latent, &get(&format!("{stage}.latent")));
        }
    }

    /// A prompt is framed RIGHT-padded, as the reference's connector reads
    /// it: its rows open the frame and the mask marks exactly those, and one
    /// longer than the frame keeps its leading tokens.
    #[test]
    fn a_prompt_is_framed_right_padded() {
        let (rows, valid) = frame_prompt(&[1.0, 2.0, 3.0, 4.0], 2, 2, 4);
        assert_eq!(rows, vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(valid, vec![1.0, 1.0, 0.0, 0.0]);
        let (rows, valid) = frame_prompt(&[1.0, 2.0, 3.0], 3, 1, 2);
        assert_eq!((rows, valid), (vec![1.0, 2.0], vec![1.0, 1.0]));
    }

    /// Each CFG branch reaches the denoiser under its OWN prompt's mask - the
    /// negative prompt has its own token count, and reading it under the
    /// conditional prompt's mask would treat its padding as caption.
    #[test]
    fn each_cfg_branch_is_masked_by_its_own_prompt() {
        struct Masks(std::cell::RefCell<Vec<Vec<f32>>>);
        impl Denoiser for Masks {
            fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
                self.0.borrow_mut().push(context.valid.to_vec());
                vec![0.0; i.latent.len()]
            }
        }
        let dit = Masks(Default::default());
        let (cond_valid, uncond_valid) = (vec![0.0f32, 1.0], vec![1.0f32, 1.0]);
        let (positions, keyframes_mask) = (grid_positions(1, 1, 1), vec![0.0f32; 1]);
        let (cond, uncond) = (Caption::of(&[1.0, 1.0], &cond_valid), Caption::of(&[0.5, 0.5], &uncond_valid));
        denoise(&dit, &[1.0, 0.0], vec![0.3; 1], &positions, &keyframes_mask, cond, uncond, 2, 1, &crate::guidance::Guidance::cfg(3.0), Sampling::euler(0.0, 1.0, 7), 1, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("a guided step runs");
        assert_eq!(*dit.0.borrow(), vec![cond_valid, uncond_valid]);
    }

    /// Under the reference guidance a video-only step runs the conditional
    /// forward, the negative-prompt one and the STG one - in that order, the
    /// STG pass alone perturbed, on the configured block - and the isolated-
    /// modality pass not at all, since there is no audio stream to cut.
    #[test]
    fn reference_guidance_runs_each_pass_once_per_step() {
        struct Recorder(std::cell::RefCell<Vec<(f32, crate::block::Perturbations)>>);
        impl Denoiser for Recorder {
            fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
                self.0.borrow_mut().push((context.rows[0], i.perturbations.clone()));
                (0..i.latent.len()).map(|k| 0.1 * context.rows[0] * (k + 1) as f32).collect()
            }
        }
        let dit = Recorder(Default::default());
        let (positions, keyframes_mask, context_valid) = (vec![0.0f32; 6], vec![0.0f32; 1], vec![1.0f32; 1]);
        let guidance = crate::guidance::Guidance::reference();
        #[rustfmt::skip]
        denoise(&dit, &[1.0, 0.0], vec![0.3; 2], &positions, &keyframes_mask, Caption::of(&[1.0], &context_valid), Caption::of(&[2.0], &context_valid), 1, 1, &guidance, Sampling::euler(0.0, 1.0, 7), 1, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("a guided step runs");
        let seen = dit.0.borrow();
        let stg = crate::block::Perturbations { video_self_attn_blocks: vec![28], audio_self_attn_blocks: vec![], cross_modal: false };
        assert_eq!(*seen, vec![(1.0, crate::block::Perturbations::default()), (2.0, crate::block::Perturbations::default()), (1.0, stg)]);
    }

    /// The last step's `sigma_next == 0`, so [`euler_ancestral_step`] returns
    /// the denoised sample directly - and with `velocity == 0` (the fake's
    /// constant-0 unconditional branch at `guidance<=1` is never reached
    /// here; use guidance=1, cond=uncond=... - simplest: assert the loop
    /// completes and returns a finite, correctly-sized vector).
    #[test]
    fn a_two_step_schedule_runs_to_completion_and_returns_the_right_shape() {
        let (_, out) = run_loop(1.0, 1.0);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    /// The denoise loop aborts at the next step boundary once cancelled, and
    /// reports the exact string the D-Bus `Cancel` surface expects.
    #[test]
    fn a_cancelled_denoise_aborts_at_the_next_step_and_says_so() {
        let sigmas = vec![1.0, 0.75, 0.5, 0.25, 0.0];
        let dit = FakeDit::default();
        let positions = grid_positions(1, 1, 1);
        let keyframes_mask = vec![0.0f32];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let cancel = capability::CancelToken::armed();
        let handle = cancel.clone();
        let err = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, Caption::of(&cond, &context_valid), Caption::of(&uncond, &context_valid), 1, 1, &crate::guidance::Guidance::cfg(1.0), Sampling::euler(0.0, 1.0, 7), 6, None, &cancel, None, &mut |step, _, _: &str| {
            if step == 2 {
                handle.cancel();
            }
        })
        .expect_err("must abort");
        assert_eq!(err, "cancelled");
        // Steps 0 and 1 ran (forwards 1,2), step 2's progress callback
        // flipped the token, step 3 refused: two forwards, not four.
        assert_eq!(dit.seen.borrow().len(), 2);
    }

    /// A denoiser with the one property that makes concurrent CFG dispatch
    /// possible at all, and the one `FakeDit` cannot have: **the device is
    /// chosen INSIDE the forward call**, so scoping the call moves the whole
    /// forward onto another card. `LtxDit::new` opens its own `Gpu`, so
    /// building a fresh one per call reproduces exactly what
    /// `dit::forward_q_streamed` does for the real 22B checkpoint, at a
    /// config that runs in milliseconds and needs no fixture.
    ///
    /// Real kernels on a real device, deliberately - a host-arithmetic fake
    /// would gate the thread plumbing while saying nothing about whether two
    /// physical cards agree bit for bit, which is the claim under test.
    struct PerCallDeviceDit {
        cfg: LtxDitConfig,
        w: Tensors,
        place: crate::devplan::Placement,
    }
    impl Denoiser for PerCallDeviceDit {
        fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
            Denoiser::forward(&LtxDit::new(self.cfg, self.w.clone(), None), i, context)
        }
        fn forward_cfg_pair(&self, i: &StepInputs, cond: Caption, uncond: Caption) -> Result<(Vec<f32>, Vec<f32>), String> {
            dispatch_cfg_pair(self, &self.place, i, cond, uncond)
        }
    }

    /// The gate for concurrent CFG dispatch: running the conditional and
    /// unconditional forwards at the same time on two cards must produce
    /// **bit-identical** results to running them one after another on one.
    ///
    /// Not a tolerance, and not `assert_eq!` on `f32` (which calls two NaNs
    /// unequal): a bit-pattern comparison, because the claim is exactness.
    /// It should hold trivially - the two forwards are independent
    /// computations, not two halves of one reduction, so no sum is
    /// reassociated by moving one of them - and if it ever does not, that is
    /// a real defect (a nondeterministic kernel, an uninitialised read)
    /// worth failing on rather than papering over with a wider bound.
    ///
    /// Runs on whatever this box has: with two schedulable cards it really
    /// dispatches across both; with one (or none) `DevicePlan::Auto` resolves
    /// `Single` and the test degenerates to "the same thing twice", which
    /// still gates the fold and the plumbing.
    #[test]
    fn the_concurrent_cfg_pair_is_bit_identical_to_the_sequential_one() {
        let cfg = LtxDitConfig::tiny();
        let w = random_tiny_weights(&cfg, 0xC0FFEE);
        let (t, ctx_len) = (8usize, 4usize);
        let inputs_owned = (
            (0..t * cfg.in_channels as usize).map(|i| (i as f32 * 0.013).sin()).collect::<Vec<f32>>(),
            vec![0.7f32; t],
            grid_positions(2, 2, 2),
            vec![0.0f32; t],
            vec![1.0f32; ctx_len],
        );
        let i = StepInputs {
            sigma: 1.0,
            latent: &inputs_owned.0,
            timesteps: &inputs_owned.1,
            positions: &inputs_owned.2,
            keyframes_mask: &inputs_owned.3,
            context_len: ctx_len,
            t,
            perturbations: &crate::block::NO_PERTURBATIONS,
            reference_attention: None,
        };
        let d = cfg.cross_attention_dim as usize;
        let cond: Vec<f32> = (0..ctx_len * d).map(|k| (k as f32 * 0.021).cos()).collect();
        let uncond: Vec<f32> = (0..ctx_len * d).map(|k| (k as f32 * 0.017).sin()).collect();

        let seq = PerCallDeviceDit { cfg, w: w.clone(), place: crate::devplan::Placement::single() };
        let (cond, uncond) = (Caption::of(&cond, &inputs_owned.4), Caption::of(&uncond, &inputs_owned.4));
        let (c0, u0) = seq.forward_cfg_pair(&i, cond, uncond).expect("the sequential pair always runs");

        let place = crate::devplan::DevicePlan::Auto.resolve(None);
        let par = PerCallDeviceDit { cfg, w, place };
        let (c1, u1) = par.forward_cfg_pair(&i, cond, uncond).expect("the concurrent pair must run, not error");

        assert_eq!(c0.len(), c1.len());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&c0), bits(&c1), "the conditional branch changed value by being placed on {:?} (cfg_parallel={})", place.cond, place.cfg_is_parallel());
        assert_eq!(bits(&u0), bits(&u1), "the unconditional branch changed value by being placed on {:?}", place.uncond);
        // The two branches must genuinely differ - a gate where cond == uncond
        // would pass even if the dispatch handed the same context to both.
        assert_ne!(bits(&c0), bits(&u0), "the two contexts must produce different velocities, or this gate proves nothing");
    }

    /// `denoise`'s CFG step must route through `forward_cfg_pair`, not
    /// through two bare `forward` calls - otherwise a denoiser that CAN use
    /// two cards silently keeps using one, and the whole placement decision
    /// becomes dead code that no test would notice.
    #[test]
    fn the_cfg_step_routes_through_the_pair_method() {
        #[derive(Default)]
        struct PairCounter {
            pairs: std::sync::atomic::AtomicUsize,
        }
        impl Denoiser for PairCounter {
            fn forward(&self, i: &StepInputs, context: Caption) -> Vec<f32> {
                vec![context.rows[0]; i.latent.len()]
            }
            fn forward_cfg_pair(&self, i: &StepInputs, cond: Caption, uncond: Caption) -> Result<(Vec<f32>, Vec<f32>), String> {
                self.pairs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok((self.forward(i, cond), self.forward(i, uncond)))
            }
        }
        let dit = PairCounter::default();
        let sigmas = vec![1.0, 0.5, 0.0];
        let positions = grid_positions(1, 1, 1);
        denoise(&dit, &sigmas, vec![0.0; 1], &positions, &[0.0f32], Caption::of(&[1.0f32], &[1.0f32]), Caption::of(&[0.0f32], &[1.0f32]), 1, 1, &crate::guidance::Guidance::cfg(5.0), Sampling::euler(0.0, 1.0, 7), 4, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
        assert_eq!(dit.pairs.load(std::sync::atomic::Ordering::Relaxed), 2, "one pair dispatch per CFG step");
    }
}
