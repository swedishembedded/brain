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
//! (`--dit`/`$BRAIN_LTXV_DIT`) switches to the real 22B distilled
//! checkpoint, loaded straight off its GGUF via
//! [`crate::gguf_src::LtxvGgufSource`] and run at int8 compute
//! ([`RealDit`], [`crate::dit::forward_q_streamed`]) - real weights were
//! only ever proven correct at REDUCED DEPTH / one block before this, so
//! treat a real-weight clip as a wiring proof, not a quality claim, exactly
//! the framing this module always carried for the tiny path. Setting
//! [`Paths::text_encoder`] (`--text-encoder`/`$BRAIN_LTXV_TEXT_ENCODER`)
//! independently switches [`context_stub`] out for [`real_text_context`]'s
//! real Gemma-4 encoding. The two switches are independent (a real DiT with
//! a stub context, or vice versa, both run) but only "real DiT + real text
//! encoder" is an honest end-to-end real-weight generation.
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

use crate::config::LtxDitConfig;
use crate::dit::{random_tiny_weights, LtxDit};
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
    /// The real audio VAE (`ltx-2.5-audio-vae-bf16.safetensors`), which
    /// carries the audio VAE decoder AND the vocoder as two disjoint tensor
    /// subsets of one file. Required only when `GenOpts::audio` is set;
    /// without it a request for sound is refused before any weight is read,
    /// rather than generating an audio latent nothing can decode.
    pub audio_vae: Option<String>,
}

/// `(variable, human name)` - kept as a table (one row today) for the same
/// reason `wan::pipeline::PATH_VARS` is: the env reader and the "you are
/// missing X" error must never disagree about the spelling. Covers only the
/// REQUIRED role; see [`OPTIONAL_PATH_VARS`] for the two real-checkpoint
/// roles that are opt-in, not required.
pub const PATH_VARS: [(&str, &str); 1] = [("BRAIN_LTXV_VAE", "VAE")];

/// `(variable, human name)` for the two OPTIONAL real-checkpoint roles
/// [`Paths::resolve`] also reads - kept out of [`PATH_VARS`] (whose own
/// callers, e.g. `ltxv_cli`'s doc self-check, document every entry as a
/// REQUIRED weight) since missing either of these is not an error, only a
/// fallback to this module's tiny-DiT/stub-context path. Spellings match
/// `crates/arch`'s own `ltxv` row's `weights_env` (`"dit"`/`"text_encoder"`).
pub const OPTIONAL_PATH_VARS: [(&str, &str); 4] = [
    ("BRAIN_LTXV_DIT", "real DiT checkpoint"),
    ("BRAIN_LTXV_TEXT_ENCODER", "real text encoder checkpoint"),
    ("BRAIN_LTXV_UPSAMPLER_SPATIAL", "spatial x2 latent upscaler (required above SINGLE_STAGE_MAX_TOKENS)"),
    ("BRAIN_LTXV_AUDIO_VAE", "audio VAE + vocoder (required for audio generation)"),
];

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        Paths::resolve(None, None, None, None)
    }

    /// The explicit flag wins over the environment variable, same precedence
    /// as every other weight path in this workspace. `dit`/`text_encoder`
    /// are optional in both forms (flag and env) - see this struct's doc.
    pub fn resolve(vae: Option<&str>, dit: Option<&str>, text_encoder: Option<&str>, spatial_upsampler: Option<&str>) -> Result<Paths, String> {
        let (var, role) = PATH_VARS[0];
        let vae = match vae.filter(|s| !s.is_empty()) {
            Some(v) => v.to_string(),
            None => match std::env::var(var) {
                Ok(v) if !v.is_empty() => v,
                _ => return Err(format!("no {role} weights: pass --vae <path> or set {var}")),
            },
        };
        let optional = |flag: Option<&str>, var: &str| -> Option<String> {
            flag.filter(|s| !s.is_empty()).map(str::to_string).or_else(|| std::env::var(var).ok().filter(|v| !v.is_empty()))
        };
        let dit = optional(dit, OPTIONAL_PATH_VARS[0].0);
        let text_encoder = optional(text_encoder, OPTIONAL_PATH_VARS[1].0);
        // `brain ltxv t2v` passes `None` here and reaches the upscaler by
        // environment only: on that command it is not a choice a caller
        // makes, only a fixed member of the LTX-2.5 checkpoint set that the
        // two-stage path requires and the single-stage path never touches.
        // `brain ltxv upscale`, whose whole subject IS the upscaler, and
        // `brain ltxv dfr` (through its own `DfrPaths`) take it as a flag.
        let spatial_upsampler = optional(spatial_upsampler, OPTIONAL_PATH_VARS[2].0);
        // Environment-only for the same reason as the upscaler above: the
        // audio VAE is not a per-run choice, it is a fixed member of the
        // LTX-2.5 checkpoint set that the audio path requires.
        let audio_vae = optional(None, OPTIONAL_PATH_VARS[3].0);
        Ok(Paths { vae, dit, text_encoder, spatial_upsampler, audio_vae })
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
/// `BRAIN_LTXV_TWO_STAGE=1`/`0` forces the choice either way, which is also
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
/// `BRAIN_LTXV_TWO_STAGE=1`/`0` overrides the token test (never the
/// divisibility one, which is a hard geometric requirement).
pub fn should_two_stage(tokens: usize, width: usize, height: usize, real_distilled: bool) -> bool {
    if !real_distilled || !width.is_multiple_of(64) || !height.is_multiple_of(64) {
        return false;
    }
    match std::env::var("BRAIN_LTXV_TWO_STAGE").ok().as_deref() {
        Some("1") | Some("on") | Some("true") => true,
        Some("0") | Some("off") | Some("false") => false,
        _ => tokens > SINGLE_STAGE_MAX_TOKENS,
    }
}

/// Which DiT config to build. `"tiny"` (the default) is the fresh-random-
/// weight smoke-test config every earlier milestone shipped. `"ltx25_22b"`
/// is the real 22B checkpoint's own config - selecting it requires
/// [`Paths::dit`] (`--dit`/`$BRAIN_LTXV_DIT`) to also be set, checked in
/// [`generate`] itself (this function only maps a name to a shape; it does
/// not know about weight paths).
pub fn dit_config_from_name(name: &str) -> Result<LtxDitConfig, String> {
    match name {
        "tiny" => Ok(LtxDitConfig::tiny()),
        "ltx25_22b" => Ok(LtxDitConfig::ltx25_22b()),
        other => Err(format!("unknown ltxv dit-config {other:?} (tiny, ltx25_22b)")),
    }
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
    /// Classifier-free guidance. `<= 1.0` skips the unconditional forward
    /// entirely (exact, see this module's doc).
    pub guidance: f32,
    pub seed: u64,
    pub fps: usize,
    /// [`ltx2_sigmas`]'s four shape parameters. Defaults are `LTX2Scheduler.
    /// execute`'s own signature defaults.
    pub base_shift: f64,
    pub max_shift: f64,
    pub stretch: bool,
    pub terminal: f64,
    /// [`euler_ancestral_step`]'s `eta`/`s_noise`. `eta=1.0` (upstream's own
    /// distilled-pipeline default, `ANCESTRAL_ETA`) is fully ancestral;
    /// `eta=0.0` collapses to a plain deterministic Euler step.
    pub eta: f64,
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
    /// The audio-extended block has no streamed/quantized/device-resident
    /// implementation the way the video-only one does (see
    /// `crate::av_stream`'s module doc), so an audio-visual generation holds
    /// the model as host fp32 and re-uploads it per forward. Until that gap
    /// closes, turning sound on has to be something a caller asks for.
    ///
    /// Requires `dit_config = "ltx25_22b"` and [`Paths::audio_vae`]; both are
    /// checked before any weight is read.
    pub audio: bool,
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
    /// `crates/ltxv/tests/motion_real.rs`'s metric (peak excursion from
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
    /// Mechanically this is the same appended guiding block `end_frame`
    /// already uses - `ltx_core.conditioning.types.keyframe_cond.
    /// VideoConditionByKeyframeIndex(frame_idx=N)`, whose `frame_idx` is a raw
    /// pixel-frame offset added to the RoPE time coordinate with no snapping
    /// and no bound on how many items a request carries. The reference's own
    /// `--image PATH FRAME_IDX STRENGTH` is repeatable for exactly this
    /// reason; see [`mid_anchor_frame`] for where the default position comes
    /// from.
    pub mid_frame: Option<String>,
    /// Which pixel frame [`Self::mid_frame`] anchors. `None` takes the
    /// reference's own single-interior-keyframe position - see
    /// [`mid_anchor_frame`].
    pub mid_frame_at: Option<usize>,
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
    /// `crates/ltxv/tests/cfg_parallel.rs`).
    pub devices: crate::devplan::DevicePlan,
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
            guidance: 1.0,
            seed: 0,
            fps: 8,
            base_shift: 0.95,
            max_shift: 2.05,
            stretch: true,
            terminal: 0.1,
            eta: 1.0,
            s_noise: 1.0,
            context_len: 8,
            dit_config: "tiny".to_string(),
            audio: false,
            device: None,
            start_frame: None,
            end_frame: None,
            mid_frame: None,
            mid_frame_at: None,
            conditioning_strength: 1.0,
            devices: crate::devplan::DevicePlan::default(),
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
/// [`Paths::text_encoder`] is set: read the checkpoint, extract its
/// embedded tokenizer (`gemma4::tokenizer`'s doc - it is the checkpoint's
/// own `tokenizer_json` tensor, not a separate file), tokenize `prompt`
/// (and, only if `guidance > 1.0`, the empty string for the unconditional
/// branch), run `gemma4::Gemma4Model::forward`, and project the full
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
/// rounding is needed on top of it. `context_valid` marks the real (`1.0`)
/// vs padded (`0.0`) positions so the connector substitutes its own
/// learnable registers into the padded tail rather than the DiT reading
/// zeros as if they were real caption content - the padded rows in
/// `ctx_cond`/`ctx_uncond` are zero exactly because the connector rewrites
/// them before any block reads them (see `EmbeddingsConnector::forward`'s
/// step 1).
///
/// `denoise`'s own CFG fold shares ONE `context_len`/`context_valid` pair
/// across both branches, so when `guidance > 1.0` the empty-prompt
/// encoding is zero-padded/truncated to match `ctx_cond`'s length - a
/// documented simplification real per-branch attention masking would
/// avoid, not exercised by this pipeline's own default (`guidance <= 1.0`
/// skips the unconditional forward entirely, per this module's doc);
/// `ctx_uncond` there is an all-zero vector of the same shape,
/// [`context_stub`]'s own "closest honest stand-in" convention.
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
    pub valid: Vec<f32>,
    pub len: usize,
    pub a_cond: Option<Vec<f32>>,
    pub a_uncond: Option<Vec<f32>>,
}

#[tracing::instrument(level = "info", name = "text_encode", skip_all, fields(prompt_chars = prompt.len(), guidance = guidance))]
fn real_text_context(path: &str, prompt: &str, dit_cfg: &LtxDitConfig, guidance: f32, device: Option<&str>, want_audio: bool) -> Result<TextContext, String> {
    use data::Tokenizer as _;
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
    // `BRAIN_LTXV_TEXT_PRECISION=fp32` forces the portable tier over a
    // quantized checkpoint. Two real uses: comparing the two tiers' output
    // on the SAME file (which is the only cheap way to get a whole-encoder
    // fp32-vs-int8 number, since both reads then hit the same bytes), and a
    // platform whose fast path is fp32 and which therefore wants the smaller
    // file without the int8 arithmetic. `Precision::for_device` still has
    // the last word in the other direction - it can refuse int8, never
    // impose it.
    let precision = match std::env::var("BRAIN_LTXV_TEXT_PRECISION").ok().as_deref().map(str::trim) {
        Some("fp32") | Some("f32") => gemma4::Precision::Fp32,
        Some("int8") | Some("i8") => gemma4::Precision::Int8,
        Some(other) if !other.is_empty() => {
            return Err(format!("ltxv real text encoder: BRAIN_LTXV_TEXT_PRECISION='{other}' is not one of fp32, int8"));
        }
        _ if quantized => gemma4::Precision::Int8,
        _ => gemma4::Precision::Fp32,
    };

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
        uncond_encoded: guidance > 1.0,
        audio: want_audio,
        encode_revision: crate::text_cache::ENCODE_REVISION,
    };
    if let Some(hit) = crate::text_cache::load(&cache_key) {
        return Ok(TextContext { cond: hit.ctx_cond, uncond: hit.ctx_uncond, valid: hit.context_valid, len: hit.context_len, a_cond: hit.a_ctx_cond, a_uncond: hit.a_ctx_uncond });
    }

    // Sub-stage timings, not one opaque block. The stage as a whole was
    // unmeasured until recently and turned out to be the largest in the
    // pipeline; measuring only its total would repeat that mistake one level
    // down, because "read the checkpoint" and "run the forward" have
    // completely different fixes and only a split says which one to attack.
    let source_bytes: u64 = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let t_read = std::time::Instant::now();

    // Two loaders, one downstream shape: both end up as "something that can
    // run a forward" plus the aggregate-embed head plus the checkpoint's own
    // tokenizer, so everything past this point is identical.
    //
    // Both are built ONCE, before the closure below, and neither the weight
    // map nor the 770-million-parameter head is ever copied per call. The
    // eager map is the whole model as f32; cloning it to encode a second
    // prompt would double the largest allocation in the process.
    enum Encoder {
        Gguf(gemma4::Gemma4GgufSource),
        Eager(gemma4::Gemma4Model),
    }
    // The AUDIO head is loaded only when it will be used: it is a second
    // projection of the same width class as the video one, so building it
    // unconditionally would add its whole parameter count to every silent
    // generation's peak for nothing.
    let (encoder, agg, agg_a, tok) = if quantized {
        let src = gemma4::Gemma4GgufSource::open(path, &cfg)?;
        let tok = src.tokenizer()?;
        let t_agg_load = std::time::Instant::now();
        let agg = aggregate_head_from_source(&src, hidden, n_states, "video")?;
        let agg_a = want_audio.then(|| aggregate_head_from_source(&src, hidden, n_states, "audio")).transpose()?;
        tracing::info!(secs = t_agg_load.elapsed().as_secs_f32(), audio = want_audio, "aggregate-embed head(s) loaded");
        (Encoder::Gguf(src), agg, agg_a, tok)
    } else {
        let raw = checkpoint::safetensors::read(path)?;
        let tok = gemma4::load_tokenizer(&raw)?;
        let weights = gemma4::import_gemma4(raw, &cfg)?;
        // Built by reference BEFORE `Gemma4Model::new` takes ownership of the
        // map - the head's own two tensors are cloned out, not borrowed.
        let agg = gemma4::AggregateEmbed::from_weights(&weights, hidden, n_states);
        let agg_a = want_audio.then(|| gemma4::AggregateEmbed::from_weights_audio(&weights, hidden, n_states));
        (Encoder::Eager(gemma4::Gemma4Model::new(cfg, weights, device)), agg, agg_a, tok)
    };
    let read_s = t_read.elapsed().as_secs_f32();
    tracing::info!(
        secs = read_s,
        bytes = source_bytes,
        mib_per_s = source_bytes as f32 / read_s.max(1e-6) / (1024.0 * 1024.0),
        quantized,
        "text encoder opened"
    );

    // A real tokenizer can legitimately return zero tokens for an empty (or
    // whitespace-only) string; `T=0` has no representable RoPE/attention
    // shape anywhere in this pipeline, so fall back to token id 0 (the
    // tokenizer's own id-0 entry, whatever it is) exactly the way an
    // all-zero fallback context already stands in for "no real content" in
    // the guidance<=1.0 branch below.
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
    // does not, so we prepend". This crate's `data::qwen_tokenizer::QwenBpe`
    // is deliberately template-free (`template_prefix`'s doc - callers that
    // want an HF-equivalent encoding prepend it themselves), so nothing added
    // it here and every prompt was encoded one token short of what the
    // checkpoint was trained on, with each caption token sitting at the
    // position the next one down should hold.
    //
    // Looked up by content rather than hard-coded: an id constant would be a
    // second, unverifiable copy of the checkpoint's own vocabulary. A
    // checkpoint whose tokenizer declares no `<bos>` at all keeps the old
    // behaviour rather than inventing a token - it is not this function's
    // place to decide what such a file meant.
    let bos_id = tok.special_id("<bos>");
    if bos_id.is_none() {
        tracing::warn!("the text encoder's tokenizer declares no <bos>; encoding without the leading BOS the reference prepends");
    }
    let tokenize = |s: &str| -> Vec<u32> {
        // `text.strip()`, same line of the reference.
        let mut ids = tok.encode(s.trim());
        ids.truncate(GEMMA4_MAX_PROMPT_TOKENS);
        match bos_id {
            Some(bos) if ids.first() != Some(&bos) => {
                ids.insert(0, bos);
                ids.truncate(GEMMA4_MAX_PROMPT_TOKENS);
            }
            _ => {}
        }
        if ids.is_empty() {
            vec![0u32]
        } else {
            ids
        }
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
        let hidden_states = match &encoder {
            Encoder::Gguf(src) => gemma4::forward_streamed(&cfg, src, device, precision, ids)?.hidden_states,
            Encoder::Eager(model) => model.forward(ids).hidden_states,
        };
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
    // Pad an encode's `[n, dim]` rows into the fixed `[context_len, dim]`
    // frame the connector reads. One helper for both streams and both
    // branches, so a width that differs cannot make the four paths differ in
    // any other way.
    let framed = |raw: &[f32], rows: usize, dim: usize| -> Vec<f32> {
        let mut v = vec![0f32; context_len * dim];
        let rows = rows.min(context_len);
        v[..rows * dim].copy_from_slice(&raw[..rows * dim]);
        v
    };
    let ctx_cond = framed(&raw_cond, n_cond, cross_attention_dim);
    let a_dim = raw_cond_a.as_ref().map(|v| v.len() / n_cond.max(1));
    let a_ctx_cond = raw_cond_a.as_ref().zip(a_dim).map(|(v, d)| framed(v, n_cond, d));
    let mut context_valid = vec![0f32; context_len];
    context_valid[..n_cond].fill(1.0);

    let (ctx_uncond, a_ctx_uncond) = if guidance > 1.0 {
        let ids_u = tokenize("");
        let (raw_u, raw_u_a) = encode(&ids_u)?;
        let v = framed(&raw_u, ids_u.len(), cross_attention_dim);
        let a = raw_u_a.as_ref().zip(a_dim).map(|(x, d)| framed(x, ids_u.len(), d));
        (v, a)
    } else {
        // The same all-zero stand-in the video branch has always used, at
        // each stream's own width - never the video vector reused, which
        // would be a shape error on the audio side and a wrong conditioning
        // if the two widths ever coincided.
        (vec![0f32; context_len * cross_attention_dim], a_dim.map(|d| vec![0f32; context_len * d]))
    };

    crate::text_cache::store(
        &cache_key,
        &crate::text_cache::Encoded {
            ctx_cond: ctx_cond.clone(),
            ctx_uncond: ctx_uncond.clone(),
            context_valid: context_valid.clone(),
            context_len,
            a_ctx_cond: a_ctx_cond.clone(),
            a_ctx_uncond: a_ctx_uncond.clone(),
        },
    );
    Ok(TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, len: context_len, a_cond: a_ctx_cond, a_uncond: a_ctx_uncond })
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
/// Reported rather than enforced: an appended guide block carries its own RoPE
/// position and does not overwrite this latent frame (see [`mid_anchor_frame`]
/// on why nothing is snapped). It names the instant of the clip a caller is
/// pointing at, which is what a log line and a window plan both need.
fn latent_frame_containing(pixel_frame: usize) -> usize {
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
/// decode a `[C, lat_t, lh, lw]` latent, taking the WHOLE-clip path when it
/// fits and the overlapping-tile path when it does not.
///
/// Which path runs is [`crate::vae3d::should_tile`]'s measured output-pixel-volume
/// policy (with `BRAIN_LTXV_VAE_TILE` as the override). Every shape this port
/// shipped before this change stays on the exact whole path bit for bit; what
/// the tiled path adds is the shapes that used to abort with a `wgpu`
/// out-of-memory - 25 frames at 1080p above all, which is 52.2 Mpx against a
/// measured ~35 Mpx ceiling on a 24 GiB card.
///
/// Borrows `vweights`: the tiled path needs them across several graph builds
/// (one per distinct tile shape), and [`upscale`] decodes several segments of
/// one clip against the same weights, so neither can be handed ownership
/// without a ~3 GB host copy.
fn decode_video(vcfg: &LtxVaeConfig, vweights: &vae::blocks::Tensors, lat_t: u32, lh: u32, lw: u32, device: Option<&str>, latent: &[f32]) -> (Vec<f32>, usize) {
    let frames = 1 + 8 * (lat_t - 1);
    let (h, w) = (lh * 32, lw * 32);
    crate::latentdump::dump_if_requested(crate::latentdump::LatentShape { c: (latent.len() / (lat_t * lh * lw) as usize) as u32, t: lat_t, h: lh, w: lw }, latent);
    if crate::vae3d::should_tile(frames, h, w) {
        let dec = crate::vae3d::LtxVaeTiledDecoder::auto(vcfg, vweights, lat_t, lh, lw, device);
        let n = dec.plan().tiles().len();
        tracing::info!(tiles = n, waste = dec.plan().overlap_waste(), frames, h, w, "VAE decode: tiled (whole-clip decode would not fit)");
        let px = dec.decode_with(latent, |done, total| tracing::debug!(done, total, "vae tile"));
        (px, dec.frames() as usize)
    } else {
        let dec = LtxVaeDecoder::build(vcfg, vweights, lat_t, lh, lw, device);
        (dec.decode(latent), dec.frames() as usize)
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
    assert!(!blocks.is_empty(), "append_image_conditioning: blocks must be non-empty");
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
/// `crates/ltxv/tests/motion_real.rs`'s table.
///
/// **Which of the two runs is decided by how many stills were given, not by
/// which ones.** One still at frame 0 and nothing else is image-to-video and
/// takes the overwrite; every other request - including any [`GenOpts::
/// mid_frame`] anchor - is keyframe interpolation and appends every still it
/// was given, `frame_idx == 0` included, because that is what
/// `image_conditionings_by_adding_guiding_latent` does. Nothing about the
/// existing one- and two-still requests changes.
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
#[allow(clippy::too_many_arguments)]
fn conditioned_latent(noise: Vec<f32>, base_positions: &[f32], base_keyframes_mask: &[f32], base_t: usize, lh: usize, lw: usize, channels: usize, frames: usize, fps: f64, start: Option<&[f32]>, mid: Option<(usize, &[f32])>, end: Option<&[f32]>, strength: f32) -> ConditionedLatent {
    assert!(start.is_some() || mid.is_some() || end.is_some(), "conditioned_latent: at least one still must be given");
    assert!((0.0..=1.0).contains(&strength), "conditioned_latent: strength {strength} is outside [0, 1]");
    let block_t = lh * lw;
    let blocks = conditioning_block_count(start.is_some(), mid.is_some(), end.is_some());
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

    // Image-to-video: one still at frame 0 and nothing else overwrites the
    // base video's own first latent frame.
    if let (Some(s), None, None) = (start, mid, end) {
        let mut latent = noise;
        mix(&mut latent, s, 0);
        let mut denoise_mask = vec![1.0f32; base_t];
        denoise_mask[..block_t].fill(m);
        let mut clean = vec![0f32; base_t * channels];
        clean[..block_t * channels].copy_from_slice(s);
        return ConditionedLatent { latent, positions: base_positions.to_vec(), keyframes_mask: base_keyframes_mask.to_vec(), denoise_mask, clean, t: base_t };
    }

    // Keyframe interpolation: every still appended as its own guiding block in
    // timeline order, the base video untouched (see this function's doc).
    let anchors: Vec<(usize, &[f32])> = [start.map(|s| (0usize, s)), mid, end.map(|e| (frames - 1, e))].into_iter().flatten().collect();
    let ic = append_image_conditioning(base_t, base_positions, base_keyframes_mask, lh, lw, channels, fps, m, &anchors);
    let mut latent = noise;
    for (bi, (_, tokens)) in anchors.iter().enumerate() {
        mix(&mut latent, tokens, base_t + bi * block_t);
    }
    ConditionedLatent { latent, positions: ic.positions, keyframes_mask: ic.keyframes_mask, denoise_mask: ic.denoise_mask, clean: ic.clean, t: total_t }
}

/// How many `lh*lw`-token conditioning blocks [`conditioned_latent`] will
/// APPEND for a given request - `0` for image-to-video (a lone start still
/// overwrites latent frame 0 in place), otherwise one per still given.
/// [`generate`] needs this before the stills are encoded, to draw the initial
/// noise at the full post-conditioning length in one go (see
/// [`conditioned_latent`]'s `noise`).
fn conditioning_block_count(start: bool, mid: bool, end: bool) -> usize {
    match (start, mid, end) {
        (true, false, false) => 0,
        _ => usize::from(start) + usize::from(mid) + usize::from(end),
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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, Some(&e), 1.0);

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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, None, 1.0);

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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, None, Some(&e), 1.0);

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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, Some(&e), strength);

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
    /// draw one noise vector at the full post-conditioning length.
    #[test]
    fn appended_block_count_matches_the_mechanism_each_request_uses() {
        assert_eq!(conditioning_block_count(false, false, false), 0, "unconditioned");
        assert_eq!(conditioning_block_count(true, false, false), 0, "image-to-video overwrites in place");
        assert_eq!(conditioning_block_count(false, false, true), 1);
        assert_eq!(conditioning_block_count(true, false, true), 2, "keyframe interpolation appends BOTH stills");
        assert_eq!(conditioning_block_count(false, true, false), 1, "a lone mid anchor is one appended guide");
        assert_eq!(conditioning_block_count(true, true, false), 2, "adding an interior anchor stops the start still from being an in-place overwrite");
        assert_eq!(conditioning_block_count(true, true, true), 3);
    }

    /// **Three anchors in one pass.** Start, middle and end each get their own
    /// appended guiding block, carrying their own encoded content, at their
    /// own pixel-frame position, with every token of the generated video still
    /// free - the reference's `image_conditionings_by_adding_guiding_latent`
    /// wraps EVERY image in `VideoConditionByKeyframeIndex` with no special
    /// case for frame 0 and no cap on how many items a request carries.
    #[test]
    fn three_anchors_append_three_guiding_blocks_at_their_own_instants() {
        let (latent, positions, km, base_t) = base(3);
        let (s, m, e) = (tokens(0.0), tokens(50.0), tokens(100.0));
        let mid_at = mid_anchor_frame(FRAMES, None).expect("a 9-frame clip has an interior");
        assert_eq!(mid_at, 4);

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), Some((mid_at, &m)), Some(&e), 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + 3 * block_t);
        assert_eq!(&c.denoise_mask[..base_t], &vec![1.0f32; base_t][..], "every token of the generated video still denoises freely");
        assert_eq!(&c.denoise_mask[base_t..], &vec![0.0f32; 3 * block_t][..], "all three guiding blocks are frozen");
        assert_eq!(&c.latent[..base_t * CH], &latent[..base_t * CH], "nothing is overwritten");
        // Timeline order, each block holding its OWN image.
        assert_eq!(&c.clean[base_t * CH..(base_t + block_t) * CH], &s[..]);
        assert_eq!(&c.clean[(base_t + block_t) * CH..(base_t + 2 * block_t) * CH], &m[..]);
        assert_eq!(&c.clean[(base_t + 2 * block_t) * CH..], &e[..]);
        // And each at its own instant: 0, 4 and 8 pixel frames at 8 frames/second.
        for (bi, want) in [(0usize, 0.0f32), (1, 0.5), (2, 1.0)] {
            let off = base_t + bi * block_t;
            assert_eq!(c.positions[off * 2], want, "guiding block {bi} sits at the wrong instant");
        }
    }

    /// A middle anchor with no still at either end is one appended guide and
    /// nothing else - the mechanism does not need company.
    #[test]
    fn a_lone_mid_anchor_appends_one_guiding_block_at_its_own_instant() {
        let (latent, positions, km, base_t) = base(1);
        let m = tokens(50.0);

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, Some((4, &m)), None, 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + block_t);
        assert_eq!(&c.denoise_mask[..base_t], &vec![1.0f32; base_t][..]);
        assert_eq!(&c.clean[base_t * CH..], &m[..]);
        assert_eq!(&c.positions[base_t * 2..base_t * 2 + 2], &[0.5, 0.625], "pixel frame 4 at 8 frames/second is a one-frame-wide [4/8, 5/8) span");
    }

    /// Adding a middle anchor to a `--start-frame` run moves the start still
    /// off the in-place overwrite and onto an appended guide, which is what
    /// the reference's interpolation builder does with every image it is
    /// given. The generated video's own latent frame 0 must be released when
    /// that happens, or the clip would be pinned twice at the same instant.
    #[test]
    fn a_mid_anchor_moves_the_start_still_from_an_overwrite_to_a_guide() {
        let (latent, positions, km, base_t) = base(2);
        let (s, m) = (tokens(0.0), tokens(50.0));

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), Some((4, &m)), None, 1.0);

        let block_t = LH * LW;
        assert_eq!(c.t, base_t + 2 * block_t);
        assert_eq!(&c.denoise_mask[..base_t], &vec![1.0f32; base_t][..], "latent frame 0 is no longer overwritten");
        assert_eq!(&c.latent[..base_t * CH], &latent[..base_t * CH], "the base video keeps its own noise");
        assert_eq!(&c.clean[base_t * CH..(base_t + block_t) * CH], &s[..]);
        assert_eq!(&c.clean[(base_t + block_t) * CH..], &m[..]);
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
    fn forward(&self, i: &StepInputs, context: &[f32]) -> Vec<f32>;

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
    fn forward_av(&self, _i: &StepInputs, _a: &AudioInputs, _context: &[f32], _a_context: &[f32]) -> (Vec<f32>, Vec<f32>) {
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
    fn forward_cfg_pair(&self, i: &StepInputs, cond: &[f32], uncond: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
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

/// Everything one denoise-step forward needs that is IDENTICAL across the
/// conditional and unconditional branches. The text context is what differs
/// between them, so it stays a separate argument; grouping the rest is what
/// makes "run this pair, wherever" expressible as one call instead of a
/// nine-argument closure repeated twice.
struct StepInputs<'a> {
    latent: &'a [f32],
    timesteps: &'a [f32],
    positions: &'a [f32],
    keyframes_mask: &'a [f32],
    context_len: usize,
    context_valid: &'a [f32],
    t: usize,
    /// The schedule's SCALAR sigma for this step. Only the joint AV forward
    /// reads it: each stream's cross-attention gate is modulated by the OTHER
    /// stream's sigma, which is a per-request scalar rather than the
    /// per-token `timesteps` above.
    sigma: f32,
}

/// The audio stream's own half of one joint forward.
struct AudioInputs<'a> {
    latent: &'a [f32],
    timesteps: &'a [f32],
    positions: &'a [f32],
    ta: usize,
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
    fn forward(&self, i: &StepInputs, context: &[f32]) -> Vec<f32> {
        LtxDit::forward(self, i.latent, i.timesteps, i.positions, i.keyframes_mask, context, i.context_len, i.t, i.context_valid).out
    }
}

impl Denoiser for crate::av_stream::AvDenoiser {
    /// A video-only forward against an audio-visual model is not something
    /// this pipeline ever wants: dropping the audio stream would silently
    /// change the VIDEO too, because the A2V cross-attention writes into the
    /// video residual every block. [`denoise`] checks [`Self::has_audio`] and
    /// takes [`Self::forward_av`] instead.
    fn forward(&self, _i: &StepInputs, _context: &[f32]) -> Vec<f32> {
        unreachable!("AvDenoiser::forward - denoise takes forward_av for an audio-visual denoiser")
    }

    fn has_audio(&self) -> bool {
        true
    }

    fn forward_av(&self, i: &StepInputs, a: &AudioInputs, context: &[f32], a_context: &[f32]) -> (Vec<f32>, Vec<f32>) {
        crate::av_stream::AvDenoiser::forward(
            self,
            &crate::av_stream::AvStepInputs {
                v_latent: i.latent,
                v_timesteps: i.timesteps,
                v_positions: i.positions,
                v_keyframes_mask: i.keyframes_mask,
                tv: i.t,
                a_latent: a.latent,
                a_timesteps: a.timesteps,
                a_positions: a.positions,
                ta: a.ta,
                sigma: i.sigma,
                context_len: i.context_len,
                context_valid: i.context_valid,
            },
            context,
            a_context,
        )
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
            .or_insert_with(|| std::sync::Arc::new(crate::devres::DitSession::resident(&self.cfg, crate::block::QTier::Int8, self.device.as_deref(), t)))
            .clone()
    }
}

impl Denoiser for RealDit {
    fn forward(&self, i: &StepInputs, context: &[f32]) -> Vec<f32> {
        let session = self.session(i.t);
        crate::dit::forward_q_streamed_in(
            &session,
            &self.cfg,
            &self.src,
            &self.head,
            crate::block::QTier::Int8,
            i.latent,
            i.timesteps,
            i.positions,
            i.keyframes_mask,
            context,
            i.context_len,
            i.t,
            i.context_valid,
            &self.cache,
        )
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
    fn forward_cfg_pair(&self, i: &StepInputs, cond: &[f32], uncond: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
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
fn dispatch_cfg_pair<D: Denoiser + Sync>(dit: &D, place: &crate::devplan::Placement, i: &StepInputs, cond: &[f32], uncond: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
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

/// The rectified-flow ancestral Euler denoise loop, `wan::pipeline::denoise`'s
/// shape: per step, one conditional and (if `guidance > 1`) one
/// unconditional forward at the SAME latent, folded as `uncond +
/// guidance·(cond - uncond)` **on the velocity** (matching
/// `wan::pipeline`'s own fold point), converted to a denoised (x0) estimate,
/// then one [`euler_ancestral_step`].
///
/// Traced (`--trace-ltxv`): the loop is one span, each step a `debug!` with
/// its sigma pair and running seconds-per-step, each individual forward a
/// `trace!`, a cancellation a `warn!` naming the step it stopped at, and a
/// non-finite prediction an `error!` - so a run that diverged or stalled can
/// be pinpointed from the trace alone instead of re-run under a profiler.
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

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "info", name = "denoise", skip_all, fields(steps = sigmas.len().saturating_sub(1), tokens = t, guidance = guidance, eta = eta))]
fn denoise(
    dit: &dyn Denoiser,
    sigmas: &[f64],
    mut latent: Vec<f32>,
    positions: &[f32],
    keyframes_mask: &[f32],
    ctx_cond: &[f32],
    ctx_uncond: &[f32],
    context_len: usize,
    context_valid: &[f32],
    t: usize,
    guidance: f32,
    eta: f64,
    s_noise: f64,
    noise_seed: u64,
    total: u32,
    frozen: Option<&Frozen>,
    cancel: &capability::CancelToken,
    audio: Option<&mut AudioState<'_>>,
    progress: &mut impl FnMut(u32, u32, &str),
) -> Result<Vec<f32>, String> {
    // Which of the reference's two loops this step follows. `ltx_pipelines.
    // distilled` picks the ancestral one for every checkpoint at or above
    // `ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)` - i.e. for the LTX-2.5
    // distilled weights this pipeline runs, stage 1 is
    // `euler_ancestral_denoising_loop` with `ANCESTRAL_ETA = 1.0` /
    // `ANCESTRAL_S_NOISE = 1.0`, not the deterministic Euler loop. The
    // difference matters most exactly where conditioning is strongest: the
    // first four sigmas of the distilled schedule move by 0.006 each, which
    // is a near-no-op without a renoise term, and a deterministic
    // trajectory pulled by clean conditioning tokens has nothing to push it
    // off the "hold the still" solution.
    //
    // An earlier version of this function REFUSED `eta > 0` whenever any
    // token was conditioned, on the grounds that a frozen token's renoise
    // term needs a per-token sigma. That requirement is real for
    // `samplers._inject_sde_noise` (the res2s loop's SDE injection, which
    // does build `stack([timesteps_from_mask(mask, sigma),
    // timesteps_from_mask(mask, sigma_next)])`) - and it is NOT how
    // `_ancestral_euler_denoising_loop` works. That loop steps the whole
    // latent with the SCALAR schedule and then re-applies
    // `post_process_latent` to the STEPPED result, which needs no per-token
    // sigma at all. See [`Frozen`]'s doc for the two orderings.
    let ancestral = eta > 0.0;
    let cfg_on = guidance > 1.0;
    // Both streams step ONE schedule in lockstep - the reference's own
    // `euler_denoising_loop` advances `video_state` and `audio_state` off the
    // same `sigmas` tensor at the same `step_idx`. That lockstep is not a
    // simplification: it is what keeps the two streams' sigmas consistent
    // with the cross-modality gates each block computes from the OTHER
    // stream's sigma.
    let mut audio = audio;
    let audio_on = audio.is_some();
    if audio_on && !dit.has_audio() {
        return Err("ltxv: an audio stream was supplied to a denoiser that does not generate audio".into());
    }
    // The latent's channel width - the axis `Modality.timesteps` `(B, T, 1)`
    // broadcasts over in [`to_denoised`].
    let channels = if t == 0 { 1 } else { latent.len() / t };
    let steps = sigmas.len().saturating_sub(1);
    let mut noise_rng = data::rng::Rng::new(noise_seed);
    let t0 = Instant::now();
    tracing::info!(steps, cfg = cfg_on, forwards_per_step = if cfg_on { 2 } else { 1 }, sigma_first = sigmas.first().copied().unwrap_or(0.0), sigma_last = sigmas.last().copied().unwrap_or(0.0), frozen_tokens = frozen.map(|f| f.mask.iter().filter(|&&m| m == 0.0).count()), "denoise loop starting");
    for i in 0..steps {
        // Once per step: a forward is one submit of the whole block stack
        // and is not interruptible from inside, same reasoning as
        // `wan::pipeline::denoise`.
        if cancel.is_cancelled() {
            tracing::warn!(step = i + 1, steps, "cancelled at a step boundary; aborting the denoise loop");
            return Err("cancelled".into());
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // `timesteps_from_mask(denoise_mask, sigma)` (`ltx_pipelines.utils.
        // helpers`, reached from `modality_from_latent_state`, which is what
        // every reference denoiser builds its `Modality.timesteps` with):
        // the model is told, PER TOKEN, how noisy that token is. A frozen
        // image-conditioning token holds CLEAN VAE content, so its timestep
        // is `0 * sigma == 0`, not the schedule's sigma. Getting this wrong
        // is not a small error: the DiT's AdaLN modulation is computed from
        // this per-token timestep (`crate::dit::ada_layer_norm_single`
        // builds one shift/scale ROW per token), so a clean token labelled
        // "as noisy as everything else" is modulated as if it were pure
        // noise - and, through self-attention, drags every genuinely-noisy
        // token in the sequence with it. The frozen token itself would still
        // decode correctly (`post_process_latent` re-pins it every step
        // regardless), so the damage lands entirely on the frames the model
        // is actually generating.
        let timesteps: Vec<f32> = match frozen {
            Some(f) => f.mask.iter().map(|&m| m * sigma as f32).collect(),
            None => vec![sigma as f32; t],
        };
        let inputs = StepInputs { latent: &latent, timesteps: &timesteps, positions, keyframes_mask, context_len, context_valid, t, sigma: sigma as f32 };
        // A single-window clip's audio stream has no conditioning items and
        // nothing frozen, so every one of its tokens carries the schedule's
        // own sigma. A long-form continuation window's carried prefix is
        // frozen exactly as the video's is, and gets timestep 0 for the same
        // reason: the DiT's adaLN modulation is built from this per-token
        // timestep, so a clean token labelled as noisy as the rest is
        // modulated as pure noise and drags every generated token with it
        // through self-attention.
        let a_timesteps: Vec<f32> = match audio.as_deref() {
            Some(a) => match &a.frozen {
                Some((mask, _)) => mask.iter().map(|&m| m * sigma as f32).collect(),
                None => vec![sigma as f32; a.ta],
            },
            None => Vec::new(),
        };
        let (velocity, a_velocity) = match audio.as_deref() {
            Some(a) => {
                let ai = AudioInputs { latent: &a.latent, timesteps: &a_timesteps, positions: &a.positions, ta: a.ta };
                if cfg_on {
                    tracing::trace!(step = i + 1, branch = "cond+uncond", sigma, "joint audio+video forward pair starting");
                    // Sequentially, never the parallel-card pair: a joint AV
                    // forward holds the model as host fp32 and both branches
                    // would want it at once (see `crate::av_stream`).
                    let (vc, ac) = dit.forward_av(&inputs, &ai, ctx_cond, a.ctx_cond);
                    let (vu, au) = dit.forward_av(&inputs, &ai, ctx_uncond, a.ctx_uncond);
                    let v: Vec<f32> = vc.iter().zip(&vu).map(|(&c, &u)| u + guidance * (c - u)).collect();
                    let aa: Vec<f32> = ac.iter().zip(&au).map(|(&c, &u)| u + guidance * (c - u)).collect();
                    (v, Some(aa))
                } else {
                    tracing::trace!(step = i + 1, branch = "cond", sigma, "joint audio+video forward starting");
                    let (v, aa) = dit.forward_av(&inputs, &ai, ctx_cond, a.ctx_cond);
                    (v, Some(aa))
                }
            }
            None if cfg_on => {
                tracing::trace!(step = i + 1, branch = "cond+uncond", sigma, "forward pair starting");
                let (cond, uncond) = dit.forward_cfg_pair(&inputs, ctx_cond, ctx_uncond)?;
                (cond.iter().zip(&uncond).map(|(&c, &u)| u + guidance * (c - u)).collect(), None)
            }
            None => {
                tracing::trace!(step = i + 1, branch = "cond", sigma, "forward starting");
                (dit.forward(&inputs, ctx_cond), None)
            }
        };
        if !velocity.iter().all(|v| v.is_finite()) {
            let bad = velocity.iter().filter(|v| !v.is_finite()).count();
            tracing::error!(step = i + 1, sigma, non_finite = bad, of = velocity.len(), "the denoiser produced non-finite values");
            return Err(format!("the denoiser produced non-finite values at step {} (sigma = {sigma:.4})", i + 1));
        }
        // The SAME per-token `timesteps` the model was just told about, not
        // the schedule's scalar sigma - `X0Model.forward` converts against
        // `Modality.timesteps` (see [`to_denoised`]), which makes the
        // conversion the identity on a frozen token instead of scaling it by
        // `1 + sigma`.
        let mut denoised = to_denoised(&latent, &velocity, &timesteps, channels);
        // Mask the x0 estimate before it is stepped - `_step_state` for the
        // deterministic loop, `_ModalityStep.from_modality_result` for the
        // ancestral one, which does it for every step including the terminal
        // one it then short-circuits to. The ancestral loop masks a SECOND
        // time below, on the stepped latent, after the renoise term.
        if let Some(f) = frozen {
            post_process_latent(&mut denoised, f);
        }
        let noise = if ancestral { Some((0..latent.len()).map(|_| noise_rng.next_gaussian() as f32).collect::<Vec<f32>>()) } else { None };
        latent = euler_ancestral_step(&latent, &denoised, sigma, sigma_next, eta, s_noise, noise.as_deref());
        // The audio stream takes the SAME step, with its own per-token
        // timesteps (uniform) and its own draw of ancestral noise. Its noise
        // comes from the same generator, after the video's, so a seed still
        // reproduces the whole run exactly - and the two streams never share
        // numbers, which they would if one draw were reused.
        if let (Some(a), Some(av)) = (audio.as_deref_mut(), a_velocity) {
            if !av.iter().all(|v| v.is_finite()) {
                return Err(format!("the denoiser produced non-finite audio values at step {} (sigma = {sigma:.4})", i + 1));
            }
            let a_channels = if a.ta == 0 { 1 } else { a.latent.len() / a.ta };
            let mut a_denoised = to_denoised(&a.latent, &av, &a_timesteps, a_channels);
            // Both applications the video half takes, for the same reasons -
            // once on the x0 estimate before the step, and again on the
            // stepped latent after the ancestral renoise term, which would
            // otherwise leave fresh noise sitting on a token that is supposed
            // to be the previous window's own clean content. See [`Frozen`].
            if let Some((mask, clean)) = &a.frozen {
                post_process_latent(&mut a_denoised, &Frozen { mask, clean, channels: a_channels });
            }
            let a_noise = if ancestral { Some((0..a.latent.len()).map(|_| noise_rng.next_gaussian() as f32).collect::<Vec<f32>>()) } else { None };
            let mut a_next = euler_ancestral_step(&a.latent, &a_denoised, sigma, sigma_next, eta, s_noise, a_noise.as_deref());
            if ancestral && sigma_next != 0.0 {
                if let Some((mask, clean)) = &a.frozen {
                    post_process_latent(&mut a_next, &Frozen { mask, clean, channels: a_channels });
                }
            }
            a.latent = a_next;
        }
        if let (Some(f), true) = (frozen, ancestral) {
            if sigma_next != 0.0 {
                post_process_latent(&mut latent, f);
            }
        }
        let per = t0.elapsed().as_secs_f32() / (i + 1) as f32;
        tracing::debug!(step = i + 1, steps, sigma, sigma_next, secs_per_step = per, "step done");
        progress(i as u32 + 1, total, &format!("denoise sigma={sigma:.3} {per:.2}s/step"));
    }
    tracing::info!(steps, secs = t0.elapsed().as_secs_f32(), "denoise loop done");
    Ok(latent)
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
#[allow(clippy::too_many_arguments)]
fn encode_still(vcfg: &LtxVaeConfig, vweights: &vae::blocks::Tensors, path: &str, width: usize, height: usize, channels: usize, device: Option<&str>) -> Result<Vec<f32>, String> {
    let (lh, lw) = (height / 32, width / 32);
    let img_t = Instant::now();
    let img = image::open(path).map_err(|e| format!("{path}: {e}"))?.resize_exact(width as u32, height as u32, image::imageops::FilterType::Lanczos3).to_rgb8();
    let mut img_chw = vec![0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x as u32, y as u32).0;
            let idx = y * width + x;
            for c in 0..3 {
                // `[0,255] -> [-1,1]`, the VAE's own input range (see
                // `LtxVaeEncoder::encode`'s doc).
                img_chw[c * height * width + idx] = (p[c] as f32 / 127.5) - 1.0;
            }
        }
    }
    let enc = LtxVaeEncoder::build(vcfg, vweights, 1, height as u32, width as u32, device);
    let cond_latent_chw = enc.encode(&img_chw);
    let cond_tokens = chw_to_tc(&cond_latent_chw, channels, 1, lh, lw);
    tracing::info!(path, width, height, secs = img_t.elapsed().as_secs_f32(), cond_tokens = cond_tokens.len() / channels, "conditioning image encoded");
    Ok(cond_tokens)
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
    ctx_cond: &'a [f32],
    ctx_uncond: &'a [f32],
    context_valid: &'a [f32],
    context_len: usize,
    cancel: &'a capability::CancelToken,
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
    /// [`euler_ancestral_step`]'s eta for THIS stage. The reference's stage 2
    /// is deterministic whatever stage 1 used: "Stage 2 is always
    /// deterministic -- its 3-step refinement schedule is too short to remove
    /// freshly injected noise" (`ltx_pipelines.distilled`).
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
    /// The audio stream's half of this stage; `None` on a video-only
    /// generation.
    audio: Option<AudioStage>,
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
    // item has appended its tokens.
    let blocks = conditioning_block_count(o.start_frame.is_some(), o.mid_frame.is_some(), o.end_frame.is_some());
    let mut latent0 = seeded_noise((t + blocks * lh * lw) * c, o.seed ^ st.seed_salt);
    if let Some(seed_chw) = st.seed_chw {
        // `GaussianNoiser.__call__`'s partial re-noise, `lerp(seed, noise,
        // sigma0)` - upstream's `ModalitySpec::noise_scale`, set to
        // `stage_2_sigmas[0]`. Only the BASE video range is seeded; any
        // appended conditioning block stays pure noise here and is
        // overwritten by its own clean content below, exactly as upstream's
        // conditioning items are applied after the noiser.
        let seed_tc = chw_to_tc(seed_chw, c, sc.lat_t, lh, lw);
        assert_eq!(seed_tc.len(), t * c, "stage seed is {} values, expected {}", seed_tc.len(), t * c);
        let s0 = st.sigmas[0] as f32;
        for (dst, &s) in latent0[..t * c].iter_mut().zip(&seed_tc) {
            *dst = (1.0 - s0) * s + s0 * *dst;
        }
    }

    let (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen) = if let Some(ctx) = &st.context {
        // Pinned exactly as `--start-frame` pins latent frame 0, over
        // `ctx.frames` latent frames instead of one: mask 0 (so the per-token
        // timestep is 0 and `to_denoised` is the identity there), content
        // written straight into the initial latent rather than noised
        // (`GaussianNoiser`'s `lerp(clean, noised, denoise_mask)` at mask 0
        // is exactly `clean`), and re-pinned every step by
        // `post_process_latent`.
        if o.start_frame.is_some() || o.mid_frame.is_some() || o.end_frame.is_some() {
            return Err("a long-form continuation window carries a latent context AND was given a conditioning still: the two both claim latent frame 0 and cannot be applied together".into());
        }
        let ctx_tokens = ctx.frames * lh * lw;
        if ctx.frames > sc.lat_t {
            return Err(format!("a {}-latent-frame context does not fit a {}-latent-frame window", ctx.frames, sc.lat_t));
        }
        let ctx_tc = chw_to_tc(ctx.chw, c, ctx.frames, lh, lw);
        let mut latent = latent0;
        latent[..ctx_tokens * c].copy_from_slice(&ctx_tc);
        let mut clean = vec![0f32; t * c];
        clean[..ctx_tokens * c].copy_from_slice(&ctx_tc);
        let mut denoise_mask = vec![1f32; t];
        denoise_mask[..ctx_tokens].fill(0.0);
        tracing::info!(stage = st.label, context_latent_frames = ctx.frames, context_tokens = ctx_tokens, tokens = t, "latent context frozen");
        (latent, positions, keyframes_mask, t, Some((denoise_mask, clean)))
    } else if o.start_frame.is_some() || o.mid_frame.is_some() || o.end_frame.is_some() {
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
        let start_tokens = o.start_frame.as_deref().map(&enc).transpose()?;
        let end_tokens = match (&o.end_frame, &o.start_frame) {
            (Some(e), Some(s)) if e == s => start_tokens.clone(),
            (Some(e), _) => Some(enc(e.as_str())?),
            (None, _) => None,
        };
        // Resolved from the clip's frame count, which is the same in both
        // stages of a two-stage run, so the anchor names the same instant at
        // half resolution and at full.
        let mid_at = o.mid_frame.as_deref().map(|_| mid_anchor_frame(o.frames, o.mid_frame_at)).transpose()?;
        let mid_tokens = o.mid_frame.as_deref().map(&enc).transpose()?;
        let mid = mid_at.zip(mid_tokens.as_deref());
        if let Some((at, _)) = mid {
            tracing::info!(stage = st.label, mid_frame = at, latent_frame = latent_frame_containing(at), of_frames = o.frames, "mid-frame anchor placed");
        }
        let cl = conditioned_latent(latent0, &positions, &keyframes_mask, t, lh, lw, c, o.frames, o.fps as f64, start_tokens.as_deref(), mid, end_tokens.as_deref(), o.conditioning_strength);
        tracing::info!(stage = st.label, strength = o.conditioning_strength, tokens = cl.t, base_tokens = t, appended_blocks = blocks, "image conditioning applied");
        (cl.latent, cl.positions, cl.keyframes_mask, cl.t, Some((cl.denoise_mask, cl.clean)))
    } else {
        (latent0, positions, keyframes_mask, t, None)
    };
    let frozen_ref = frozen.as_ref().map(|(mask, clean)| Frozen { mask, clean, channels: c });
    tracing::info!(stage = st.label, width = st.width, height = st.height, tokens = denoise_t_count, base_tokens = t, steps = st.sigmas.len() - 1, eta = st.eta, seeded = st.seed_chw.is_some(), "stage denoising");
    let done_before = st.done_before;
    let dim = crate::audio::TOKEN_DIM as usize;
    let mut audio_state = st.audio.take().map(|a| {
        let AudioStage { mut latent, context } = a;
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
        let frozen = (!context.is_empty()).then(|| {
            let mut mask = vec![1f32; ta];
            mask[..context.len() / dim].fill(0.0);
            let mut clean = vec![0f32; ta * dim];
            clean[..context.len()].copy_from_slice(&context);
            (mask, clean)
        });
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
        sc.context_valid,
        denoise_t_count,
        o.guidance,
        st.eta,
        o.s_noise,
        o.seed ^ 0x4e_4f_49_53_45 ^ st.seed_salt,
        total,
        frozen_ref.as_ref(),
        sc.cancel,
        audio_state.as_mut(),
        &mut |done, tot, phase| progress(done_before + done, tot, phase),
    )?;
    Ok(StageOut { video_chw: tc_to_chw(&final_latent[..t * c], c, sc.lat_t, lh, lw), audio: audio_state.map(|a| a.latent) })
}

/// One stage's result: the video latent it produced and, on an audio-visual
/// generation, the audio latent the SAME forwards produced beside it.
struct StageOut {
    /// `[C, lat_t, lh, lw]`.
    video_chw: Vec<f32>,
    /// `[ta, crate::audio::TOKEN_DIM]`, `None` on a video-only generation.
    audio: Option<Vec<f32>>,
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
    if (up_lh as usize, up_lw as usize) != (lh, lw) {
        tracing::error!(got_h = up_lh, got_w = up_lw, want_h = lh, want_w = lw, "spatial upscaler produced the wrong latent grid");
        return Err(format!("spatial upscaler produced a {up_lh}x{up_lw} latent grid, expected {lh}x{lw} for {}x{}", r.width, r.height));
    }

    // `eta = 0`: upstream's refinement stage is always the DETERMINISTIC
    // sampler, "its 3-step refinement schedule is too short to remove freshly
    // injected noise".
    denoise_stage(
        sc,
        Stage {
            width: r.width,
            height: r.height,
            sigmas: r.sigmas,
            eta: 0.0,
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
/// [`upscale`] pays it per pass. **Not until needed**: an audio-visual
/// generation holds the whole transformer as host fp32 while it denoises, so
/// the upscaler's own expansion must not sit beside it through a stage that
/// does not use it.
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
fn build_denoiser(paths: &Paths, dit_cfg: LtxDitConfig, o: &GenOpts, place: crate::devplan::Placement) -> Result<Box<dyn Denoiser>, String> {
    if o.dit_config == "tiny" {
        // Not a real model: random weights, so any output is a wiring proof
        // and nothing else. Worth a warning rather than an info line - a run
        // that silently produced noise because a checkpoint path was unset is
        // the most expensive way to discover this.
        tracing::warn!("--dit-config tiny: building a RANDOM-weight DiT, output is a smoke test and carries no semantics");
        let weight_seed = o.seed ^ 0x4c_54_58_76_44_49_54; // "LTXvDIT" folded into the seed, so the same --seed reproduces the same weights
        let weights: Tensors = random_tiny_weights(&dit_cfg, weight_seed);
        return Ok(Box::new(LtxDit::new(dit_cfg, weights, o.device.as_deref())));
    }
    let dit_path = paths.dit.as_ref().ok_or_else(|| {
        tracing::error!(dit_config = %o.dit_config, "no real DiT checkpoint configured for a real dit-config");
        format!("ltxv dit-config {:?} needs a real checkpoint: pass --dit <path> or set BRAIN_LTXV_DIT", o.dit_config)
    })?;
    tracing::info!(path = %dit_path, "opening the real DiT GGUF");
    let src = crate::gguf_src::LtxvGgufSource::open(dit_path).inspect_err(|e| tracing::error!(path = %dit_path, error = %e, "opening the DiT GGUF failed"))?;
    if o.audio {
        // The audio-visual denoiser is a DIFFERENT object, not the video one
        // with a flag: it runs `LtxAvDit`, which computes both streams in one
        // forward because the A<->V cross-attention couples them every block.
        // It also expands the whole checkpoint to host fp32 rather than
        // streaming int8 blocks - see `crate::av_stream`'s module doc for why
        // that is the only route today and what it costs.
        let cfg = *src.config();
        tracing::warn!(gib = crate::av_stream::host_floats(&cfg) * 4 / (1 << 30), "audio-visual generation: expanding the whole checkpoint to host fp32 (the audio stream has no streamed/quantized path yet)");
        let w = crate::av_stream::AvWeights::load(&src, cfg)?;
        return Ok(Box::new(crate::av_stream::AvDenoiser::new(w, o.device.as_deref())));
    }
    let real_cfg = src.config().video;
    if real_cfg != dit_cfg {
        tracing::error!(path = %dit_path, dit_config = %o.dit_config, "the checkpoint's embedded config does not match the named build config");
        return Err(format!("ltxv: {dit_path}'s own embedded config does not match LtxDitConfig::{:?}() - checkpoint/build mismatch", o.dit_config));
    }
    let head = crate::dit::load_head_tensors_from_source(&src, &real_cfg);
    tracing::info!(layers = real_cfg.num_layers, inner_dim = real_cfg.inner_dim, head_tensors = head.len(), "real DiT ready (blocks stream per forward)");
    // Keyed on the checkpoint, not on this call: the same store is handed to
    // the next generation against the same file (see `RealDit`'s doc and
    // `crate::weightcache`).
    let cache = crate::block::GenerationCache::for_checkpoint(dit_path);
    let cs = cache.stats();
    tracing::info!(cached_blocks = cs.blocks, cached_bytes = cs.bytes, "checkpoint weight cache attached");
    Ok(Box::new(RealDit { cfg: real_cfg, src, head, device: o.device.clone(), cache, place, sessions: Default::default() }))
}

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
        return Ok(TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, len: context_len, a_cond: None, a_uncond: None });
    };
    tracing::info!(path = %te_path, "encoding the prompt with the real text encoder");
    let te_t = std::time::Instant::now();
    // On the card the conditional DiT forward will NOT use, when the plan has
    // one to spare: the 12B encoder's own device footprint is then released
    // from a card that is not about to hold the denoise loop's activations.
    // See `crate::devplan`.
    let r = crate::devplan::on_gpu(place.text, || real_text_context(te_path, prompt, &dit_cfg, o.guidance, o.device.as_deref(), o.audio))?
        .inspect(|c| tracing::info!(context_len = c.len, audio = c.a_cond.is_some(), "prompt encoded"))
        .inspect_err(|e| tracing::error!(path = %te_path, error = %e, "text encoding failed"))?;
    *text_encode_secs = te_t.elapsed().as_secs_f32();
    tracing::info!(secs = *text_encode_secs, "text encode done");
    Ok(r)
}

/// Text to video. `progress(done, total, phase)` mirrors `wan::pipeline::
/// generate`'s contract; `cancel` is polled once per denoise step. `prompt`
/// only ever reaches [`context_stub`] (see this module's doc - there is no
/// real text encoder).
#[tracing::instrument(level = "info", name = "generate", skip_all, fields(frames = o.frames, width = o.width, height = o.height, steps = o.steps, seed = o.seed, guidance = o.guidance, dit_config = %o.dit_config))]
/// Everything an audio request needs, checked before any weight is read.
///
/// One function because [`generate`] and [`generate_long`] are two entry
/// points into the same requirement, and a check that lived in only one of
/// them would let the other produce an audio latent nothing can decode - or
/// spend a whole-checkpoint host expansion first and fail after it.
fn check_audio_request(paths: &Paths, o: &GenOpts) -> Result<(), String> {
    if !o.audio {
        return Ok(());
    }
    if o.dit_config != "ltx25_22b" {
        return Err(format!(
            "audio generation needs the real audio-visual checkpoint: dit_config is {:?}, and the tiny smoke-test DiT has random weights whose audio stream would be noise, not sound",
            o.dit_config
        ));
    }
    if paths.dit.is_none() {
        let (var, _) = OPTIONAL_PATH_VARS[0];
        return Err(format!("audio generation needs the real DiT checkpoint: pass --dit <path> or set {var}"));
    }
    if paths.audio_vae.is_none() {
        let (var, role) = OPTIONAL_PATH_VARS[3];
        return Err(format!("audio generation needs the {role}: set {var} to ltx-2.5-audio-vae-bf16.safetensors"));
    }
    if paths.text_encoder.is_none() {
        let (var, _) = OPTIONAL_PATH_VARS[1];
        return Err(format!(
            "audio generation needs the real text encoder: the audio stream is conditioned through its OWN text projection (text_embedding_projection.audio_aggregate_embed), which the stub context cannot stand in for - set {var}"
        ));
    }
    crate::av_stream::AvWeights::fits_in_host_memory(&crate::config::LtxAvDitConfig::ltx25())
}

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
    let two_stage = should_two_stage(t, o.width, o.height, is_real_distilled);
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
    tracing::info!(cond_gpu = ?place.cond, uncond_gpu = ?place.uncond, text_gpu = ?place.text, cfg_parallel = place.cfg_is_parallel() && o.guidance > 1.0, "device placement resolved");
    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit = build_denoiser(paths, dit_cfg, o, place)?;
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
    let TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, len: context_len, a_cond: text_a_cond, a_uncond: text_a_uncond } =
        build_context(paths, prompt, dit_cfg, o, place, &mut timings.text_encode)?;

    // ---- the stage plan ---------------------------------------------------
    // `vraw`/`vweights` are loaded here (not at decode time below, which
    // reuses them) because image conditioning needs a real VAE ENCODE before
    // any denoising, once per stage at that stage's own resolution.
    let vraw = read_any(&paths.vae)?;
    let vweights = crate::import::import_vae(vraw, &vcfg)?;
    // The audio stream's own initial noise, drawn from a seed derived from -
    // but never equal to - the video stream's, so one `--seed` reproduces the
    // whole audio-visual run and the two streams never start from the same
    // numbers.
    let ta = crate::audio::latent_frames(o.frames, o.fps);
    let audio_latent0 = o.audio.then(|| {
        tracing::info!(audio_tokens = ta, token_dim = crate::audio::TOKEN_DIM, seconds = o.frames as f32 / o.fps as f32, "audio latent shape resolved from the clip's own duration");
        seeded_noise(ta * crate::audio::TOKEN_DIM as usize, o.seed ^ AUDIO_SEED_SALT)
    });
    let (a_ctx_cond, a_ctx_uncond) = (text_a_cond.unwrap_or_default(), text_a_uncond.unwrap_or_default());
    let stage_ctx = StageCtx {
        dit: dit.as_ref(),
        vcfg: &vcfg,
        vweights: &vweights,
        o,
        lat_t,
        in_channels,
        ctx_cond: &ctx_cond,
        ctx_uncond: &ctx_uncond,
        context_valid: &context_valid,
        context_len,
        a_ctx_cond: &a_ctx_cond,
        a_ctx_uncond: &a_ctx_uncond,
        cancel,
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
            let (var, role) = OPTIONAL_PATH_VARS[2];
            tracing::error!(tokens = t, width = o.width, height = o.height, "a two-stage request needs the spatial latent upscaler");
            format!(
                "{}x{} is {t} video tokens, past the {SINGLE_STAGE_MAX_TOKENS}-token ceiling the distilled schedule holds in ONE stage, so it needs the reference's two-stage path - set {var} to the {role} (ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors). Forcing BRAIN_LTXV_TWO_STAGE=0 runs one stage instead and is known to disintegrate the end of the clip at this size.",
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
                width: w1,
                height: h1,
                sigmas: &sigmas,
                eta: o.eta,
                seed_chw: None,
                context: None,
                seed_salt: 0,
                done_before: 0,
                label: "stage1",
                audio: audio_latent0.map(|latent| AudioStage { latent, context: Vec::new() }),
            },
            total,
            &mut progress,
        )?;

        upscale_and_refine(
            &stage_ctx,
            Refine {
                upsampler: &upsampler,
                upsampler_path,
                audio: stage1.audio.map(|latent| AudioStage { latent, context: Vec::new() }),
                latent_chw: &stage1.video_chw,
                lat_t,
                lh1,
                lw1,
                width: o.width,
                height: o.height,
                sigmas: &stage2_sigmas(LTX2_STAGE2_STEPS)?,
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
                width: o.width,
                height: o.height,
                sigmas: &sigmas,
                eta: o.eta,
                seed_chw: None,
                context: None,
                seed_salt: 0,
                done_before: 0,
                label: "single",
                audio: audio_latent0.map(|latent| AudioStage { latent, context: Vec::new() }),
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
    timings.forwards_per_step = if o.guidance > 1.0 { 2 } else { 1 };
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
    let (pixels, frames) = decode_video(&vcfg, &vweights, lat_t as u32, lh as u32, lw as u32, o.device.as_deref(), &final_chw);
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
    let audio = match audio_latent {
        Some(latent) => {
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
        None => None,
    };
    progress(total, total, "done");
    tracing::info!(frames, width = w, height = h, fps = o.fps, audio = audio.is_some(), total_secs = timings.total(), "generation done");
    Ok((Video { width: w as u32, height: h as u32, fps: o.fps, frames: out, audio }, timings))
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
    let vcfg = crate::vocoder::VocoderConfig::ltx25();
    // ONE read, split by prefix. The two importers want disjoint subsets of
    // one file and `StTensor` is not `Clone`, which is why the parity suite
    // reads it twice - but a `partition` on the prefix each importer already
    // selects by hands each of them its own half with no clone and no second
    // pass over the checkpoint. `vocoder.bwe_generator.*`/`vocoder.mel_stft.*`
    // land on the vocoder side and are skipped there exactly as before.
    let (voc_raw, vae_raw): (Vec<_>, Vec<_>) = checkpoint::safetensors::read(path)?.into_iter().partition(|t| t.name.starts_with("vocoder."));
    let vae_w = crate::import::import_audio_vae(vae_raw, &acfg)?;
    let voc_w = crate::import::import_vocoder(voc_raw, &vcfg)?;
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
    let plane = frames.len() * h * w;
    let mut out = vec![0f32; 3 * plane];
    for (f, px) in frames.iter().enumerate() {
        for i in 0..h * w {
            for c in 0..3 {
                out[c * plane + f * h * w + i] = (px[i * 3 + c] as f32 / 127.5) - 1.0;
            }
        }
    }
    out
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
        let (var, role) = OPTIONAL_PATH_VARS[2];
        tracing::error!("upscaling needs the spatial latent upscaler");
        format!("upscaling IS the spatial latent upscaler: pass --upsampler-spatial <path> or set {var} to the {role} (ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)")
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
    let dit = build_denoiser(paths, dit_cfg, &o.base, place)?;
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
    let (ctx_cond, ctx_uncond, context_valid, context_len) = (text.cond, text.uncond, text.valid, text.len);

    let vweights = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
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
            ctx_cond: &ctx_cond,
            ctx_uncond: &ctx_uncond,
            context_valid: &context_valid,
            context_len,
            cancel,
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
                latent_chw: &latent,
                lat_t,
                lh1: h / 32,
                lw1: w / 32,
                width: out_w,
                height: out_h,
                sigmas: &sigmas,
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
        let (pixels, got) = decode_video(&vcfg, &vweights, lat_t as u32, lh as u32, lw as u32, o.base.device.as_deref(), &refined);
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
    timings.forwards_per_step = if o.base.guidance > 1.0 { 2 } else { 1 };

    if out_frames.len() != frames {
        return Err(format!("reassembled {} frames from {} passes, expected {frames}", out_frames.len(), plan.len()));
    }
    progress(total, total, "done");
    tracing::info!(frames, width = out_w, height = out_h, passes = plan.len(), total_secs = timings.total(), "upscale done");
    Ok((Video { width: out_w as u32, height: out_h as u32, fps: clip.fps, frames: out_frames, audio: None }, timings))
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
    /// by default.
    pub max_window_tokens: usize,
    /// `frames` is the WHOLE clip's length, not one window's.
    pub base: GenOpts,
}

impl Default for LongOpts {
    fn default() -> LongOpts {
        LongOpts { context_latent_frames: crate::longform::CONTEXT_LATENT_FRAMES, max_window_tokens: crate::longform::LONGFORM_MAX_TOKENS, base: GenOpts::default() }
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
/// **Scope.** `--start-frame` conditions window 0, as it would a single
/// clip. `--end-frame` is refused: a continuation window's latent context and
/// an appended keyframe block both want to be the thing the window is pinned
/// to, and "the clip ends on this still" over a multi-window plan has not
/// been designed. `--mid-frame` is refused for a second reason on top of that
/// one: its position is a pixel frame of the WHOLE clip, and routing it means
/// finding the window whose emitted range covers it and re-expressing it in
/// that window's own frame numbering. Long-form generation is otherwise the
/// ordinary generation path - same schedule, same CFG fold, same two-stage
/// decision per window, same VAE.
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
    check_audio_request(paths, &o.base)?;
    if o.base.end_frame.is_some() {
        return Err("--end-frame is not supported for a multi-window clip: it pins the last frame of ONE window, and pinning the end of a rolling plan has not been designed".into());
    }
    if o.base.mid_frame.is_some() {
        return Err(format!(
            "--mid-frame is not supported for a multi-window clip: this {}-frame request is {} windows, and an anchor at a clip-wide pixel frame has to be routed to the window that covers it and re-expressed in that window's own frame numbering, which has not been designed",
            o.base.frames,
            plan.len()
        ));
    }
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
    let two_stage = should_two_stage(widest, o.base.width, o.base.height, is_real_distilled);
    if two_stage && paths.spatial_upsampler.is_none() {
        let (var, role) = OPTIONAL_PATH_VARS[2];
        return Err(format!(
            "the widest window of this {}x{} plan is {widest} video tokens, past the {SINGLE_STAGE_MAX_TOKENS}-token ceiling the distilled schedule holds in ONE stage, so it needs the reference's two-stage path - set {var} to the {role}",
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
    let dit = build_denoiser(paths, dit_cfg, &o.base, place)?;
    if o.base.audio && !dit.has_audio() {
        return Err("ltxv: audio was requested but the built denoiser carries no audio stream".into());
    }
    timings.build_dit = build_t.elapsed().as_secs_f32();
    let TextContext { cond: ctx_cond, uncond: ctx_uncond, valid: context_valid, len: context_len, a_cond, a_uncond } =
        build_context(paths, prompt, dit_cfg, &o.base, place, &mut timings.text_encode)?;
    let (a_ctx_cond, a_ctx_uncond) = (a_cond.unwrap_or_default(), a_uncond.unwrap_or_default());
    let vweights = crate::import::import_vae(read_any(&paths.vae)?, &vcfg)?;
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
    let mut vae_secs = 0.0f32;
    let work_t = Instant::now();
    let mut done_before = 1u32;

    for (wi, w) in plan.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let lat_t = w.latent_frames();
        let win_opts = GenOpts {
            frames: w.decoded_frames(),
            // The still conditions the clip's own opening, which is window 0
            // and nowhere else.
            start_frame: if wi == 0 { o.base.start_frame.clone() } else { None },
            end_frame: None,
            ..o.base.clone()
        };
        let sc = StageCtx {
            a_ctx_cond: &a_ctx_cond,
            a_ctx_uncond: &a_ctx_uncond,
            dit: dit.as_ref(),
            vcfg: &vcfg,
            vweights: &vweights,
            o: &win_opts,
            lat_t,
            in_channels,
            ctx_cond: &ctx_cond,
            ctx_uncond: &ctx_uncond,
            context_valid: &context_valid,
            context_len,
            cancel,
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
                    audio: audio_noise.map(|latent| AudioStage { latent, context: std::mem::take(&mut carried_audio_half) }),
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
                    latent_chw: &stage1_chw,
                    lat_t,
                    lh1,
                    lw1,
                    width: o.base.width,
                    height: o.base.height,
                    sigmas: &refine_sigmas,
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
                    audio: stage1.audio.map(|latent| AudioStage { latent, context: std::mem::take(&mut carried_audio_full) }),
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
                    width: o.base.width,
                    height: o.base.height,
                    sigmas: &sigmas,
                    eta: o.base.eta,
                    seed_chw: None,
                    context: carried_full.as_ref().map(|(chw, n)| LatentContext { chw, frames: *n }),
                    seed_salt,
                    done_before,
                    label: "single",
                    audio: audio_noise.map(|latent| AudioStage { latent, context: std::mem::take(&mut carried_audio_full) }),
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
        let (pixels, got) = decode_video(&vcfg, &vweights, lat_t as u32, lh as u32, lw as u32, o.base.device.as_deref(), &final_chw);
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
    timings.forwards_per_step = if o.base.guidance > 1.0 { 2 } else { 1 };

    // ---- decode the sound, ONCE over the whole clip ------------------------
    // The windows contributed tokens to one latent, not waveforms to be butted
    // together, so the audio VAE and the vocoder see a single continuous
    // sequence and there is no decode boundary at a seam to smooth over. The
    // token count is the clip's own, so this is the same arithmetic - and the
    // same sample count - a single-window clip of this length produces.
    let audio = match &aplan {
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
/// request that names one scene is bit-for-bit the run it already was.
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
    let scene_opts = |s: &Scene, si: usize| LongOpts {
        base: GenOpts {
            frames: s.frames,
            // The still conditions the clip's own opening, which is the first
            // scene's first window and nowhere else.
            start_frame: if si == 0 { o.base.start_frame.clone() } else { None },
            end_frame: None,
            seed: o.base.seed ^ crate::longform::SCENE_SEED_SALT.wrapping_mul(si as u64),
            ..o.base.clone()
        },
        ..o.clone()
    };
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

/// DFR-specific weight paths: the video VAE (decode only, same file
/// [`Paths`] already names) plus the two real latent upscalers
/// (spatial x2 always required, temporal x2 only when
/// [`DfrOpts::temporal_upsample_rounds`] is nonzero). Kept as its OWN struct
/// rather than adding fields to [`Paths`], so [`generate`]/[`Paths`] stay
/// exactly what they already were - DFR EXTENDS the pipeline, it does
/// not touch [`generate`]'s own surface (see this crate's module doc).
///
/// ## What's real in [`generate_dfr`], precisely (read before assuming this
/// ## generates anything real)
///
/// Same honesty bar [`generate`]'s own doc sets, extended for DFR's
/// own additional gaps. REAL:
///
/// * The canvas/keyframe-segment geometry ([`crate::dfr::resolve_canvas`]),
///   the generated-keyframe-slot token append + `keyframes_mask`
///   construction ([`crate::dfr::keyframe_slots`], landing squarely on the
///   `keyframes_mask` seam [`crate::dit::LtxDit::forward`] has accepted),
///   the tile-boundary/lead-in/stitch math
///   ([`crate::dfr::tile_ranges`]/[`crate::dfr::stitch_tile_latents`]), and
///   the final frame-count contract ([`crate::dfr::target_frame_count`]).
/// * The two real-weight latent upscalers ([`crate::upsampler`]) -
///   stage 1's half-res video AND its generated keyframe slots are BOTH
///   really spatially upscaled x2, and each temporal round really runs the
///   real temporal x2 upscaler before tiling.
/// * Stage 2 and every temporal-round tile genuinely RE-NOISE a real seed
///   (the upscaled stage-1 result, a tile's temporally-upsampled local
///   segment) via the same `torch.lerp(seed, noise, sigma0)` formula
///   `GaussianNoiser` uses, not a fresh unrelated noise draw - see
///   [`noised_seed`].
/// * The tiny random-weight DiT ([`generate`]'s own stand-in), the same
///   real `LTX2Scheduler`/CFG-fold/ancestral-Euler [`denoise`] loop
///   [`generate`] uses (called once per stage/tile), and the real VAE
///   conv-decoder decode.
///
/// NOT real, by explicit scope (see this crate's module doc):
///
/// * **No IC-LoRA at all.** Stage 2's real spatial-detailing adapter does
///   not exist in this repo (a LoRA on the real 22B model this hardware
///   cannot run regardless) - `_detailing_lora`/`VideoConditionByReferenceLatent`
///   have no counterpart here. Stage 2 here is "re-noise the upscaled
///   result and denoise again with the SAME tiny DiT", which exercises the
///   real re-noise/keyframe-reseed mechanics but claims no detailing
///   quality whatsoever.
/// * **No per-token/partial-strength anchor-keyframe carry-forward.** Real
///   DFR pins a temporal round's seam keyframes at `strength=0.95`
///   (`_ANCHOR_KEYFRAME_STRENGTH`) via genuinely PER-TOKEN timesteps -
///   `denoise`'s own doc already records that this whole pipeline broadcasts
///   one scalar sigma to every token (no partial/keyframe denoising), and
///   building real per-token stepping was judged out of scope for a
///   smoke-level milestone (it would touch the already-parity-tested Euler-
///   ancestral step machinery, not just pipeline orchestration). Concretely:
///   each round's tiles regenerate their seam region as ordinary noise
///   re-seeded only from the temporally-upsampled video (not from a
///   carried-forward anchor still), so seam continuity across tiles is not
///   modeled here even though [`crate::dfr::TileRange::anchor_kf_global`]
///   computes the real anchor positions a future change could wire in.
/// * **The NA diffusion decoder is not wired in as an alternative
///   decode path.** [`generate_dfr`] decodes through the same real conv
///   decoder [`generate`] uses. `na_decoder::NADecoder`'s tiling/scale
///   requirements (overlapping-tile chunked decode, `w_chunks`) differ
///   enough from this decoder's single-shot call that wiring it in was
///   judged a separate, nontrivial integration - the same "land what's
///   solid" judgment call made for the NA decoder's own stage-5/full-chain
///   question.
/// * **No real distilled-schedule sigma tables.** Real DFR uses fixed
///   `DISTILLED_SIGMAS`/`STAGE_2_DISTILLED_SIGMAS`/a `DISTILLED_SIGMAS[4:]`
///   subrange for temporal rounds; every stage here instead calls the same
///   generic [`ltx2_sigmas`] at [`GenOpts::steps`] steps, for the same
///   "one real schedule generator already proven, no second one" reasoning
///   [`generate`] follows.
/// * **Position units.** See `crate::dfr`'s module doc: keyframe-slot RoPE
///   positions live in this port's own fractional-latent-grid convention,
///   not real LTX's fps-normalized pixel time.
#[derive(Clone, Debug)]
pub struct DfrPaths {
    pub vae: String,
    pub spatial_upsampler: String,
    /// Required only when `DfrOpts::temporal_upsample_rounds > 0`.
    pub temporal_upsampler: Option<String>,
}

/// `(variable, human name)` - the DFR analogue of [`PATH_VARS`], one row
/// added per new weight role this milestone needs.
pub const DFR_PATH_VARS: [(&str, &str); 3] = [("BRAIN_LTXV_VAE", "VAE"), ("BRAIN_LTXV_UPSAMPLER_SPATIAL", "spatial latent upscaler"), ("BRAIN_LTXV_UPSAMPLER_TEMPORAL", "temporal latent upscaler")];

impl DfrPaths {
    pub fn from_env() -> Result<DfrPaths, String> {
        DfrPaths::resolve(None, None, None)
    }

    /// The explicit flag wins over the environment variable, same precedence
    /// [`Paths::resolve`] uses. The temporal upscaler path is OPTIONAL (only
    /// `--temporal-upsample-rounds > 0` requires it) - callers that need it
    /// check `Option::is_none()` themselves, same as
    /// [`DfrOpts::temporal_upsample_rounds`]'s own validation in
    /// [`generate_dfr`].
    pub fn resolve(vae: Option<&str>, spatial_upsampler: Option<&str>, temporal_upsampler: Option<&str>) -> Result<DfrPaths, String> {
        let required = |flag: Option<&str>, var: &str, role: &str| -> Result<String, String> {
            match flag.filter(|s| !s.is_empty()) {
                Some(v) => Ok(v.to_string()),
                None => match std::env::var(var) {
                    Ok(v) if !v.is_empty() => Ok(v),
                    _ => Err(format!("no {role} weights: pass the matching flag or set {var}")),
                },
            }
        };
        let vae = required(vae, DFR_PATH_VARS[0].0, DFR_PATH_VARS[0].1)?;
        let spatial_upsampler = required(spatial_upsampler, DFR_PATH_VARS[1].0, DFR_PATH_VARS[1].1)?;
        let temporal_upsampler = match temporal_upsampler.filter(|s| !s.is_empty()) {
            Some(v) => Some(v.to_string()),
            None => std::env::var(DFR_PATH_VARS[2].0).ok().filter(|v| !v.is_empty()),
        };
        Ok(DfrPaths { vae, spatial_upsampler, temporal_upsampler })
    }
}

/// Everything a single DFR generation varies: the same knobs [`GenOpts`]
/// already exposes (`width`/`height` are the FULL, stage-2/final resolution
/// - stage 1 runs at half that), plus DFR's own multi-round knob.
#[derive(Clone, Debug, Default)]
pub struct DfrOpts {
    pub base: GenOpts,
    /// 0, 1, or 2 temporal x2 refine rounds - `dfr_pipeline.py`'s own
    /// `temporal_upsample_rounds` contract. `> 0` requires
    /// [`DfrPaths::temporal_upsampler`]. Defaults to 0: no refine round.
    pub temporal_upsample_rounds: usize,
}

/// `torch.lerp(seed, noise, sigma0)`: `(1-sigma0)*seed + sigma0*noise` -
/// `GaussianNoiser.__call__`'s own partial re-noise formula
/// (`ltx_core.components.noisers`), used everywhere DFR seeds a stage from a
/// previous stage's REAL content (the upscaled stage-1 video/keyframes for
/// stage 2, a tile's temporally-upsampled local segment for a round) instead
/// of the pure noise [`seeded_noise`] draws for stage 1 (which has no prior
/// content to seed from - `sigma0` there is always effectively 1.0, so this
/// formula would collapse to plain noise anyway).
fn noised_seed(seed_content: &[f32], sigma0: f32, seed: u64) -> Vec<f32> {
    let noise = seeded_noise(seed_content.len(), seed);
    seed_content.iter().zip(&noise).map(|(&c, &n)| (1.0 - sigma0) * c + sigma0 * n).collect()
}

/// DFR text to video: half-res base + keyframe slots -> real spatial x2
/// upscale -> full-res re-noised detailing pass -> 0-2 real temporal x2
/// upscale + tile-stitch rounds -> real VAE decode. See this section's own
/// doc (right above [`DfrPaths`]) for exactly what's real and what isn't.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "info", name = "generate_dfr", skip_all, fields(frames = o.base.frames, width = o.base.width, height = o.base.height, steps = o.base.steps, seed = o.base.seed, temporal_rounds = o.temporal_upsample_rounds))]
pub fn generate_dfr(paths: &DfrPaths, prompt: &str, o: &DfrOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<(Video, Timings), String> {
    let base = &o.base;
    tracing::info!(prompt_chars = prompt.chars().count(), device = base.device.as_deref().unwrap_or("(ambient)"), "DFR generation starting");
    if o.temporal_upsample_rounds > 2 {
        tracing::error!(rounds = o.temporal_upsample_rounds, "temporal_upsample_rounds out of range");
        return Err(format!("temporal_upsample_rounds must be 0, 1, or 2, got {}", o.temporal_upsample_rounds));
    }
    if o.temporal_upsample_rounds > 0 && paths.temporal_upsampler.is_none() {
        return Err("temporal_upsample_rounds > 0 requires a temporal upsampler path".into());
    }
    let vcfg = LtxVaeConfig::conv25();
    if vcfg.latent_frames(base.frames as u32).is_none() {
        return Err(format!("{} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)", base.frames));
    }
    if !base.width.is_multiple_of(64) || !base.height.is_multiple_of(64) {
        return Err(format!("{}x{} is not a multiple of 64 for DFR (stage 1 halves it, and the half must still be a multiple of the VAE's 32 spatial stride)", base.width, base.height));
    }
    if base.steps == 0 {
        return Err("--steps must be at least 1".into());
    }

    let dit_cfg = dit_config_from_name(&base.dit_config)?;
    if dit_cfg.in_channels != vcfg.latent_channels {
        return Err(format!("ltxv dit-config {:?} has in_channels {} but the VAE latent width is {}", base.dit_config, dit_cfg.in_channels, vcfg.latent_channels));
    }
    let in_channels = dit_cfg.in_channels as usize;

    let (canvas_frames, _segment, kf_positions) = dfr::resolve_canvas(base.frames, dfr::VIDEO_TEMPORAL_SCALE)?;
    let lat_t = vcfg.latent_frames(canvas_frames as u32).expect("resolve_canvas always returns a 1+8k frame count") as usize;
    let k = kf_positions.len();

    let mut timings = Timings::default();
    let total_phases = (3 + o.temporal_upsample_rounds + 1) as u32; // build, stage1, stage2, N rounds, decode

    // Every per-stage `denoise` call below gets its OWN no-op progress
    // closure (a fresh literal each call site, not a shared `&mut` binding -
    // `&mut impl FnMut` is not `Copy`/`Clone`, so one binding could not be
    // reused across the several `denoise` calls this function makes). This
    // function reports progress itself, once per stage/tile, instead of
    // forwarding `denoise`'s own per-step granularity - a deliberate
    // simplification for a multi-stage pipeline (see this section's doc).

    // ---- build the DiT once - shared by every stage/tile (see this
    // section's doc: no detailing LoRA, so stage 2 reuses the SAME weights
    // stage 1 and every round tile use) ----
    progress(0, total_phases, "build transformer");
    let build_t = Instant::now();
    let weight_seed = base.seed ^ 0x4c_54_58_76_44_49_54; // "LTXvDIT"
    let weights: Tensors = random_tiny_weights(&dit_cfg, weight_seed);
    let dit = LtxDit::new(dit_cfg, weights, base.device.as_deref());
    timings.build_dit = build_t.elapsed().as_secs_f32();
    tracing::info!(secs = timings.build_dit, canvas_frames, keyframe_slots = k, "DFR transformer built (random tiny weights)");
    if cancel.is_cancelled() {
        tracing::warn!(phase = "after build", "cancelled");
        return Err("cancelled".into());
    }

    let prompt_mix = base.seed ^ fnv1a(prompt);
    let ctx_cond = context_stub(base.context_len, dit_cfg.cross_attention_dim as usize, prompt_mix);
    let ctx_uncond = vec![0f32; base.context_len * dit_cfg.cross_attention_dim as usize];

    let denoise_t = Instant::now();

    // ---- Stage 1: half-res base + keyframe slots, pure noise (no prior
    // stage exists to seed from) ----
    progress(1, total_phases, "stage1 denoise");
    tracing::info!(stage = "stage1", "half-res base denoise");
    let (lh1, lw1) = (base.height / 2 / 32, base.width / 2 / 32);
    let t0_1 = lat_t * lh1 * lw1;
    let base_positions_1 = grid_positions(lat_t, lh1, lw1);
    let layout1 = dfr::keyframe_slots(t0_1, &base_positions_1, lh1, lw1, &kf_positions, dfr::VIDEO_TEMPORAL_SCALE, true)?;
    let t1 = layout1.total_tokens;
    let sigmas1 = ltx2_sigmas(t1, base.steps, base.base_shift, base.max_shift, base.stretch, base.terminal);
    let latent1_0 = seeded_noise(t1 * in_channels, base.seed ^ 0x53_31);
    let ctx_valid = vec![1.0f32; base.context_len]; // DFR's DiT is always tiny-config (connector disabled) - see this section's doc.
    let final1 = denoise(&dit, &sigmas1, latent1_0, &layout1.positions, &layout1.keyframes_mask, &ctx_cond, &ctx_uncond, base.context_len, &ctx_valid, t1, base.guidance, base.eta, base.s_noise, base.seed ^ 0x4e_31, base.steps as u32, None, cancel, None, &mut |_, _, _: &str| {})?;
    if cancel.is_cancelled() {
        tracing::warn!(stage = "stage1", "cancelled");
        return Err("cancelled".into());
    }

    let reserved_half_res_video = tc_to_chw(&final1[..t0_1 * in_channels], in_channels, lat_t, lh1, lw1);
    let slot1_chw = tc_to_chw(&final1[t0_1 * in_channels..], in_channels, k, lh1, lw1);

    // ---- real spatial x2 upscale of BOTH the video and its slots ----
    progress(1, total_phases, "spatial upscale");
    tracing::info!(stage = "spatial_upscale", path = %paths.spatial_upsampler, "real x2 latent upscale");
    // The VAE is imported HERE rather than at decode time because the latent
    // upscalers need its `per_channel_statistics` (see
    // `crate::upsampler::upsample_video`); the decode below reuses it, so the
    // file is still read exactly once.
    let vraw = read_any(&paths.vae)?;
    let vweights = crate::import::import_vae(vraw, &vcfg)?;
    let sraw = read_any(&paths.spatial_upsampler)?;
    let scfg = LatentUpsamplerConfig::spatial_x2();
    let sweights = crate::import::import_upsampler(sraw, &scfg)?;
    let video_upsampler = LatentUpsampler::build(&scfg, &sweights, lat_t as u32, lh1 as u32, lw1 as u32, base.device.as_deref());
    // `upsample_video`, not `upsample`: the per-channel un-normalize/
    // re-normalize sandwich the upscaler was trained inside. This call site
    // predates that helper and was missing it, which cost half the latent's
    // variance here too.
    let (pc_mean, pc_std) = crate::vae3d::per_channel_statistics(&vweights);
    let upscaled_video_chw = crate::upsampler::upsample_video(&video_upsampler, &pc_mean, &pc_std, &reserved_half_res_video);
    let (_, _, lh2u, lw2u) = video_upsampler.out_shape();
    let slots_upsampler = LatentUpsampler::build(&scfg, &sweights, k as u32, lh1 as u32, lw1 as u32, base.device.as_deref());
    let upscaled_slots_chw = crate::upsampler::upsample_video(&slots_upsampler, &pc_mean, &pc_std, &slot1_chw);
    drop(sweights);

    let (lh2, lw2) = (lh2u as usize, lw2u as usize);
    let (want_lh2, want_lw2) = (base.height / 32, base.width / 32);
    if (lh2, lw2) != (want_lh2, want_lw2) {
        tracing::error!(got_h = lh2, got_w = lw2, want_h = want_lh2, want_w = want_lw2, "spatial upscaler produced the wrong latent grid");
        return Err(format!("spatial upscaler produced a {lh2}x{lw2} latent grid, expected {want_lh2}x{want_lw2} for {}x{}", base.width, base.height));
    }

    // ---- Stage 2: full-res detailing, RE-NOISED from the real upscaled
    // seed (no IC-LoRA - see this section's doc) ----
    progress(2, total_phases, "stage2 denoise");
    tracing::info!(stage = "stage2", latent_h = lh2, latent_w = lw2, "full-res detailing denoise");
    let t0_2 = lat_t * lh2 * lw2;
    let base_positions_2 = grid_positions(lat_t, lh2, lw2);
    let layout2 = dfr::keyframe_slots(t0_2, &base_positions_2, lh2, lw2, &kf_positions, dfr::VIDEO_TEMPORAL_SCALE, true)?;
    let t2 = layout2.total_tokens;
    let sigmas2 = ltx2_sigmas(t2, base.steps, base.base_shift, base.max_shift, base.stretch, base.terminal);
    let sigma2_0 = sigmas2[0] as f32;
    let mut seed2 = chw_to_tc(&upscaled_video_chw, in_channels, lat_t, lh2, lw2);
    seed2.extend_from_slice(&chw_to_tc(&upscaled_slots_chw, in_channels, k, lh2, lw2));
    // The per-stage seed tags are ASCII ("S2", "N2"), written unsplit because
    // `0x53_32` reads to clippy as `0x53` with a mistyped `i32` suffix.
    let latent2_0 = noised_seed(&seed2, sigma2_0, base.seed ^ 0x5332);
    let final2 = denoise(&dit, &sigmas2, latent2_0, &layout2.positions, &layout2.keyframes_mask, &ctx_cond, &ctx_uncond, base.context_len, &ctx_valid, t2, base.guidance, base.eta, base.s_noise, base.seed ^ 0x4e32, base.steps as u32, None, cancel, None, &mut |_, _, _: &str| {})?;
    if cancel.is_cancelled() {
        tracing::warn!(stage = "stage2", "cancelled");
        return Err("cancelled".into());
    }

    let mut video_chw = tc_to_chw(&final2[..t0_2 * in_channels], in_channels, lat_t, lh2, lw2);
    let mut cur_lat_t = lat_t;

    // ---- 0-2 temporal x2 upsample rounds, tile-based ----
    if o.temporal_upsample_rounds > 0 {
        let traw = read_any(paths.temporal_upsampler.as_ref().expect("checked above"))?;
        let tcfg = LatentUpsamplerConfig::temporal_x2();
        let tweights = crate::import::import_upsampler(traw, &tcfg)?;

        for round_idx in 1..=o.temporal_upsample_rounds {
            if cancel.is_cancelled() {
                tracing::warn!(round = round_idx, "cancelled");
                return Err("cancelled".into());
            }
            progress(2 + round_idx as u32, total_phases, &format!("temporal round {round_idx}"));

            let tup = LatentUpsampler::build(&tcfg, &tweights, cur_lat_t as u32, lh2 as u32, lw2 as u32, base.device.as_deref());
            // Same contract as the spatial upscaler above: raw VAE latent
            // space in and out (`crate::upsampler::upsample_video`).
            let upsampled_video = crate::upsampler::upsample_video(&tup, &pc_mean, &pc_std, &video_chw);
            let (_, ut, _, _) = tup.out_shape();
            let new_lat_t = ut as usize;

            let round_num_frames = dfr::target_frame_count(canvas_frames, round_idx as u32);
            let seam_positions: Vec<usize> = kf_positions.iter().map(|&p| p * (1usize << round_idx)).collect();
            let num_tiles = 1usize << round_idx;
            let tiles = dfr::tile_ranges(&seam_positions, round_num_frames, num_tiles, dfr::VIDEO_TEMPORAL_SCALE, dfr::TILE_LEAD_SEGMENTS)?;

            let mut tile_video_outputs: Vec<Vec<f32>> = Vec::with_capacity(tiles.len());
            tracing::info!(stage = "temporal_round", round = round_idx, tiles = tiles.len(), latent_frames = new_lat_t, "temporal x2 upscale round");
            for (tile_index, tile) in tiles.iter().enumerate() {
                if cancel.is_cancelled() {
                    tracing::warn!(round = round_idx, tile = tile_index, "cancelled");
                    return Err("cancelled".into());
                }
                tracing::debug!(round = round_idx, tile = tile_index, latent_start = tile.latent_start, latent_end = tile.latent_end_exclusive, "tile denoise");
                let lat_t_local = tile.latent_end_exclusive - tile.latent_start;
                let tile_video_chw = dfr::slice_time_chw(&upsampled_video, in_channels, new_lat_t, lh2, lw2, tile.latent_start, tile.latent_end_exclusive);
                let tile_video_tc = chw_to_tc(&tile_video_chw, in_channels, lat_t_local, lh2, lw2);

                let local_slot_positions = dfr::remap_positions_to_local(&tile.slot_kf_global, tile.pixel_start);
                let base_positions_tile = grid_positions(lat_t_local, lh2, lw2);
                let t0_tile = lat_t_local * lh2 * lw2;
                let layout_tile = dfr::keyframe_slots(t0_tile, &base_positions_tile, lh2, lw2, &local_slot_positions, dfr::VIDEO_TEMPORAL_SCALE, true)?;
                let t_tile = layout_tile.total_tokens;

                // Slot content seeds from the nearest local video latent
                // frame's tokens (`_slot_initials_from_video`'s own
                // nearest-frame rule).
                let mut seed_tile = tile_video_tc.clone();
                let hw = lh2 * lw2;
                for &lp in &local_slot_positions {
                    let fi = ((lp as f64 / dfr::VIDEO_TEMPORAL_SCALE as f64).round() as usize).min(lat_t_local.saturating_sub(1));
                    let start = fi * hw * in_channels;
                    seed_tile.extend_from_slice(&tile_video_tc[start..start + hw * in_channels]);
                }

                let sigmas_tile = ltx2_sigmas(t_tile, base.steps, base.base_shift, base.max_shift, base.stretch, base.terminal);
                let sigma_tile_0 = sigmas_tile[0] as f32;
                let reseed_seed = base.seed ^ 0x52_53 ^ (1000 * round_idx as u64) ^ tile_index as u64;
                let latent_tile_0 = noised_seed(&seed_tile, sigma_tile_0, reseed_seed);
                // Tiles are positionally identical across rounds, so a
                // shared noise seed would inject byte-identical ancestral
                // noise into every one of them - offset per round/tile, the
                // same reasoning `dfr_pipeline.py`'s own `noise_seed=seed +
                // 1000*round_idx + tile_index` documents.
                let noise_seed_tile = base.seed ^ 0x54_49 ^ (1000 * round_idx as u64) ^ tile_index as u64;
                let final_tile = denoise(
                    &dit,
                    &sigmas_tile,
                    latent_tile_0,
                    &layout_tile.positions,
                    &layout_tile.keyframes_mask,
                    &ctx_cond,
                    &ctx_uncond,
                    base.context_len,
                    &ctx_valid,
                    t_tile,
                    base.guidance,
                    base.eta,
                    base.s_noise,
                    noise_seed_tile,
                    base.steps as u32,
                    None,
                    cancel,
                    None,
                    &mut |_, _, _: &str| {},
                )?;
                tile_video_outputs.push(tc_to_chw(&final_tile[..t0_tile * in_channels], in_channels, lat_t_local, lh2, lw2));
            }

            video_chw = dfr::stitch_tile_latents(&tile_video_outputs, &tiles, in_channels, lh2, lw2)?;
            let expected_t = (round_num_frames - 1) / dfr::VIDEO_TEMPORAL_SCALE + 1;
            let actual_t = video_chw.len() / (in_channels * lh2 * lw2);
            if actual_t != expected_t {
                return Err(format!("temporal round {round_idx}: stitched latent T={actual_t} != expected {expected_t}"));
            }
            cur_lat_t = actual_t;
        }
    }

    // ---- trim to the caller's frame-count contract (see
    // `dfr::target_frame_count`'s doc) - applies even at rounds=0, since the
    // canvas may have padded past the requested frame count on its own ----
    let target_frames = dfr::target_frame_count(base.frames, o.temporal_upsample_rounds as u32);
    let keep_latents = (target_frames - 1) / dfr::VIDEO_TEMPORAL_SCALE + 1;
    if keep_latents > cur_lat_t {
        return Err(format!("target {target_frames} frames exceeds the generated canvas ({cur_lat_t} latent frames)"));
    }
    if keep_latents != cur_lat_t {
        video_chw = dfr::slice_time_chw(&video_chw, in_channels, cur_lat_t, lh2, lw2, 0, keep_latents);
        cur_lat_t = keep_latents;
    }
    timings.denoise = denoise_t.elapsed().as_secs_f32();
    timings.steps = base.steps;
    timings.tokens = t2;
    timings.forwards_per_step = if base.guidance > 1.0 { 2 } else { 1 };

    // ---- decode -------------------------------------------------------------
    progress(total_phases - 1, total_phases, "vae decode");
    let decode_t = Instant::now();
    let (pixels, frames) = decode_video(&vcfg, &vweights, cur_lat_t as u32, lh2 as u32, lw2 as u32, base.device.as_deref(), &video_chw);
    let (w, h) = (base.width, base.height);
    if pixels.len() != 3 * frames * h * w {
        return Err(format!("VAE returned {} values, expected {}", pixels.len(), 3 * frames * h * w));
    }
    let plane = frames * h * w;
    let out: Vec<Vec<u8>> = (0..frames)
        .map(|f| {
            let mut px = vec![0u8; h * w * 3];
            for c in 0..3 {
                // Deliberately NOT named `base` (this closure's own plane
                // offset) - `generate_dfr`'s `base: &GenOpts` is in scope
                // here, and this repo's own porting playbook flags exactly
                // this kind of unrelated shadow as a readability trap.
                let plane_base = c * plane + f * h * w;
                for i in 0..h * w {
                    px[i * 3 + c] = (127.5 * (pixels[plane_base + i].clamp(-1.0, 1.0) + 1.0)) as u8;
                }
            }
            px
        })
        .collect();
    timings.decode = decode_t.elapsed().as_secs_f32();
    let fps = base.fps * (1usize << o.temporal_upsample_rounds);
    progress(total_phases, total_phases, "done");
    Ok((Video { width: w as u32, height: h as u32, fps, frames: out, audio: None }, timings))
}

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
    ///
    /// Guarded against the environment: `BRAIN_LTXV_TWO_STAGE` would
    /// otherwise let a stray export make this pass or fail for the wrong
    /// reason.
    #[test]
    fn the_stage_policy_matches_the_shapes_that_were_measured() {
        if std::env::var("BRAIN_LTXV_TWO_STAGE").is_ok() {
            return;
        }
        let toks = |w: usize, h: usize| 4 * (h / 32) * (w / 32);
        for (w, h) in [(512, 512), (960, 544), (1280, 704), (1600, 896)] {
            let t = toks(w, h);
            assert!(t <= SINGLE_STAGE_MAX_TOKENS, "{w}x{h} is {t} tokens, which the measurement says is a single-stage shape");
            assert!(!should_two_stage(t, w, h, true), "{w}x{h} ({t} tokens) was measured good on ONE stage and must not change path");
        }
        let (w, h) = (1920, 1088);
        let t = toks(w, h);
        assert_eq!(t, 8160);
        assert!(should_two_stage(t, w, h, true), "1920x1088 is {t} tokens and was measured to disintegrate on one stage");
    }

    /// The two conditions that are NOT about the token count: a
    /// non-distilled config has no distilled schedule to outgrow, and an
    /// axis that is not a multiple of 64 cannot be halved onto the VAE's
    /// 32-pixel stride (upstream asserts the same thing).
    #[test]
    fn the_stage_policy_refuses_a_shape_it_cannot_halve_and_a_config_it_does_not_apply_to() {
        if std::env::var("BRAIN_LTXV_TWO_STAGE").is_ok() {
            return;
        }
        let big = SINGLE_STAGE_MAX_TOKENS + 1;
        assert!(!should_two_stage(big, 1920, 1088, false), "the tiny random-weight config runs the generic shifted schedule, not the distilled table");
        // 1088 is a multiple of 64; 1056 is a multiple of 32 but not 64, so
        // halving it lands off the VAE's spatial stride.
        assert!(!1056usize.is_multiple_of(64) && 1056usize.is_multiple_of(32));
        assert!(!should_two_stage(big, 1920, 1056, true), "an axis that cannot be halved onto the 32-pixel stride must stay single-stage");
        assert!(!should_two_stage(big, 1888, 1088, true), "the width axis has the same requirement as the height axis");
    }

    #[test]
    fn gen_opts_defaults_are_a_representable_tiny_clip() {
        let o = GenOpts::default();
        let vcfg = LtxVaeConfig::conv25();
        assert!(vcfg.latent_frames(o.frames as u32).is_some(), "{} frames must be 1+8k", o.frames);
        assert!(o.width.is_multiple_of(32) && o.height.is_multiple_of(32));
        assert_eq!(o.dit_config, "tiny");
    }

    #[test]
    fn an_explicit_vae_path_beats_the_environment_variable() {
        std::env::set_var("BRAIN_LTXV_VAE", "env-vae");
        let p = Paths::resolve(None, None, None, None).expect("from env");
        assert_eq!(p.vae, "env-vae");
        // Deliberately no assertion about `dit`/`text_encoder` here: this test
        // sets only `BRAIN_LTXV_VAE`, so asserting those were `None` would be
        // asserting about variables it never controlled - it failed for anyone
        // who happened to have a real checkpoint exported. Their resolution is
        // owned by the sibling test below, which does clear them first.
        let p = Paths::resolve(Some("/flag/vae"), None, None, None).expect("flag wins");
        assert_eq!(p.vae, "/flag/vae");
        let p = Paths::resolve(Some(""), None, None, None).expect("empty flag falls through");
        assert_eq!(p.vae, "env-vae");
        std::env::remove_var("BRAIN_LTXV_VAE");
        let e = Paths::resolve(None, None, None, None).unwrap_err();
        assert!(e.contains("--vae") && e.contains("BRAIN_LTXV_VAE"), "{e}");
    }

    /// [`Paths::dit`]/[`Paths::text_encoder`] follow the exact same
    /// flag-over-env, optional (not error-on-absent) resolution
    /// [`OPTIONAL_PATH_VARS`] documents.
    #[test]
    fn the_optional_real_checkpoint_paths_resolve_flag_over_env_and_are_none_when_absent() {
        std::env::remove_var("BRAIN_LTXV_DIT");
        std::env::remove_var("BRAIN_LTXV_TEXT_ENCODER");
        let p = Paths::resolve(Some("/vae"), None, None, None).expect("vae only");
        assert_eq!(p.dit, None);
        assert_eq!(p.text_encoder, None);

        std::env::set_var("BRAIN_LTXV_DIT", "env-dit");
        std::env::set_var("BRAIN_LTXV_TEXT_ENCODER", "env-te");
        let p = Paths::resolve(Some("/vae"), None, None, None).expect("from env");
        assert_eq!(p.dit, Some("env-dit".to_string()));
        assert_eq!(p.text_encoder, Some("env-te".to_string()));
        let p = Paths::resolve(Some("/vae"), Some("/flag-dit"), Some("/flag-te"), None).expect("flag wins");
        assert_eq!(p.dit, Some("/flag-dit".to_string()));
        assert_eq!(p.text_encoder, Some("/flag-te".to_string()));
        std::env::remove_var("BRAIN_LTXV_DIT");
        std::env::remove_var("BRAIN_LTXV_TEXT_ENCODER");
    }

    #[test]
    fn a_bad_frame_count_is_rejected_before_any_weight_is_read() {
        let paths = Paths { vae: "/nope".into(), dit: None, text_encoder: None, spatial_upsampler: None, audio_vae: None };
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
    /// Not a nicety: the audio path expands the whole checkpoint to host fp32
    /// before it can discover any of this, so a check that ran later would
    /// cost minutes and most of a machine's RAM to say "no text encoder". Each
    /// prerequisite is asserted separately, because a single "audio needs more
    /// setup" message would send a caller looking in the wrong place.
    #[test]
    fn an_audio_request_that_cannot_produce_sound_is_refused_before_any_weight_is_read() {
        let audio = GenOpts { audio: true, dit_config: "ltx25_22b".into(), ..GenOpts::default() };
        let bare = Paths { vae: "/nope".into(), dit: None, text_encoder: None, spatial_upsampler: None, audio_vae: None };

        // Wrong config: the tiny DiT's audio stream is random weights.
        let tiny = GenOpts { dit_config: "tiny".into(), ..audio.clone() };
        let e = generate(&bare, "x", &tiny, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("real audio-visual checkpoint"), "{e}");

        // No DiT checkpoint.
        let e = generate(&bare, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("BRAIN_LTXV_DIT"), "{e}");

        // No audio VAE - the sound would have nothing to decode it.
        let with_dit = Paths { dit: Some("/dit".into()), ..bare.clone() };
        let e = generate(&with_dit, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("BRAIN_LTXV_AUDIO_VAE"), "{e}");

        // No text encoder - the audio stream has its OWN text projection, so
        // the stub context cannot stand in for it the way it can for video.
        let with_av = Paths { audio_vae: Some("/avae".into()), ..with_dit.clone() };
        let e = generate(&with_av, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("BRAIN_LTXV_TEXT_ENCODER"), "{e}");
        assert!(e.contains("audio_aggregate_embed"), "the message must name WHY the stub cannot stand in: {e}");

        // And with every prerequisite named, the refusal is gone: the run
        // proceeds far enough to fail on the bogus paths themselves, which is
        // a different error entirely.
        let full = Paths { text_encoder: Some("/te".into()), ..with_av };
        let e = generate(&full, "x", &audio, &Default::default(), |_, _, _| {}).err().expect("the fake paths still fail");
        assert!(!e.contains("BRAIN_LTXV_"), "no prerequisite should still be reported missing: {e}");
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
        fn forward(&self, i: &StepInputs, context: &[f32]) -> Vec<f32> {
            self.seen.borrow_mut().push(context[0]);
            self.timesteps_seen.borrow_mut().push(i.timesteps.to_vec());
            self.latents_seen.borrow_mut().push(i.latent.to_vec());
            vec![context[0]; i.latent.len()]
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
        let frozen = Frozen { mask: &mask, clean: &clean, channels };
        // `GaussianNoiser`: a fully conditioned token starts AT its clean
        // content, never at noise.
        let latent0 = vec![0.0f32, 5.0];

        let out = denoise(&dit, &sigmas, latent0, &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, 1.0, 1.0, 7, 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("eta > 0 with conditioned tokens must run, not be refused");

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
        let frozen = Frozen { mask: &mask, clean: &clean, channels };
        let latent0 = vec![0.0f32, 10.0];
        // `FakeDit` predicts velocity 1.0 everywhere, and token 1's own
        // timestep is `0.5 * 0.5`, so its x0 estimate is `10 - 0.25`; the
        // reference's blend then lands it half way back to `clean`.
        let want = 0.5 * (10.0 - 0.25) + 0.5 * 10.0;

        for eta in [0.0f64, 1.0] {
            let dit = FakeDit::default();
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, eta, 1.0, 7, 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
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
        // A velocity of 1.0 on every token (`FakeDit` echoes `context[0]`) -
        // the model does NOT conveniently predict zero at a frozen token, and
        // the fix must not depend on it doing so.
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let mask = vec![1.0f32, 0.0];
        let clean = vec![0.0f32, 5.0];
        let frozen = Frozen { mask: &mask, clean: &clean, channels };
        let latent0 = vec![0.0f32, 5.0];

        for eta in [0.0f64, 1.0] {
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, eta, 1.0, 7, 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
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
    /// `crates/ltxv/tests/longform.rs`'s
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
        let frozen = Frozen { mask: &mask, clean: &clean, channels };
        let mut latent0 = seeded_noise(t * channels, 11);
        for &(lo, hi) in &frozen_ranges {
            latent0[lo * channels..hi * channels].copy_from_slice(&clean[lo * channels..hi * channels]);
        }

        let dit = FakeDit::default();
        for eta in [0.0f64, 1.0] {
            let out = denoise(&dit, &sigmas, latent0.clone(), &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, eta, 1.0, 7, 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
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
        let frozen = Frozen { mask: &mask, clean: &clean, channels };
        denoise(&dit, &sigmas, vec![0.0; t * channels], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, 0.0, 1.0, 7, 4, Some(&frozen), &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");

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
        denoise(&dit, &sigmas, vec![0.0; t], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, 0.0, 1.0, 7, 4, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
        assert_eq!(*dit.timesteps_seen.borrow(), vec![vec![1.0f32; t], vec![0.5f32; t]]);
    }

    fn run_loop(guidance: f32, eta: f64) -> (Vec<f32>, Vec<f32>) {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit::default();
        let positions = grid_positions(1, 1, 1);
        let keyframes_mask = vec![0.0f32];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let out = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, 1, guidance, eta, 1.0, 7, 4, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
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
        let err = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, 1, 1.0, 0.0, 1.0, 7, 6, None, &cancel, None, &mut |step, _, _: &str| {
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
        fn forward(&self, i: &StepInputs, context: &[f32]) -> Vec<f32> {
            Denoiser::forward(&LtxDit::new(self.cfg, self.w.clone(), None), i, context)
        }
        fn forward_cfg_pair(&self, i: &StepInputs, cond: &[f32], uncond: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
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
            context_valid: &inputs_owned.4,
            t,
        };
        let d = cfg.cross_attention_dim as usize;
        let cond: Vec<f32> = (0..ctx_len * d).map(|k| (k as f32 * 0.021).cos()).collect();
        let uncond: Vec<f32> = (0..ctx_len * d).map(|k| (k as f32 * 0.017).sin()).collect();

        let seq = PerCallDeviceDit { cfg, w: w.clone(), place: crate::devplan::Placement::single() };
        let (c0, u0) = seq.forward_cfg_pair(&i, &cond, &uncond).expect("the sequential pair always runs");

        let place = crate::devplan::DevicePlan::Auto.resolve(None);
        let par = PerCallDeviceDit { cfg, w, place };
        let (c1, u1) = par.forward_cfg_pair(&i, &cond, &uncond).expect("the concurrent pair must run, not error");

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
            fn forward(&self, i: &StepInputs, context: &[f32]) -> Vec<f32> {
                vec![context[0]; i.latent.len()]
            }
            fn forward_cfg_pair(&self, i: &StepInputs, cond: &[f32], uncond: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
                self.pairs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok((self.forward(i, cond), self.forward(i, uncond)))
            }
        }
        let dit = PairCounter::default();
        let sigmas = vec![1.0, 0.5, 0.0];
        let positions = grid_positions(1, 1, 1);
        denoise(&dit, &sigmas, vec![0.0; 1], &positions, &[0.0f32], &[1.0f32], &[0.0f32], 1, &[1.0f32], 1, 5.0, 0.0, 1.0, 7, 4, None, &Default::default(), None, &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
        assert_eq!(dit.pairs.load(std::sync::atomic::Ordering::Relaxed), 2, "one pair dispatch per CFG step");
    }
}
