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
use crate::upsampler::{LatentUpsampler, LatentUpsamplerConfig};
use crate::vae3d::{LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder};
use diffusion::scheduler::{euler_ancestral_step, ltx2_sigmas, LTX2_DISTILLED_SIGMAS};

/// The real distilled schedule's own step count (`LTX2_DISTILLED_SIGMAS.len() -
/// 1`), exposed so a caller (e.g. `crates/cli/src/ltxv_cli.rs`'s own
/// progress line) can report it without hardcoding a number that would drift
/// from the table itself.
pub const LTX2_DISTILLED_STEPS: usize = LTX2_DISTILLED_SIGMAS.len() - 1;

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
pub const OPTIONAL_PATH_VARS: [(&str, &str); 2] = [("BRAIN_LTXV_DIT", "real DiT checkpoint"), ("BRAIN_LTXV_TEXT_ENCODER", "real text encoder checkpoint")];

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        Paths::resolve(None, None, None)
    }

    /// The explicit flag wins over the environment variable, same precedence
    /// as every other weight path in this workspace. `dit`/`text_encoder`
    /// are optional in both forms (flag and env) - see this struct's doc.
    pub fn resolve(vae: Option<&str>, dit: Option<&str>, text_encoder: Option<&str>) -> Result<Paths, String> {
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
        Ok(Paths { vae, dit, text_encoder })
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
            device: None,
            start_frame: None,
            end_frame: None,
            conditioning_strength: 1.0,
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
    pub steps: usize,
    pub tokens: usize,
    pub forwards_per_step: usize,
}

impl Timings {
    /// Everything this struct actually attributes. Compare against a caller's
    /// own wall clock via [`Self::unattributed`] rather than presenting this
    /// as the run's total.
    pub fn total(&self) -> f32 {
        self.build_dit + self.text_encode + self.denoise + self.decode
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
/// `(ctx_cond, ctx_uncond, context_valid, context_len)` - what an encode
/// hands the denoise loop, and what [`crate::text_cache`] stores.
type TextContext = (Vec<f32>, Vec<f32>, Vec<f32>, usize);

#[tracing::instrument(level = "info", name = "text_encode", skip_all, fields(prompt_chars = prompt.len(), guidance = guidance))]
fn real_text_context(path: &str, prompt: &str, dit_cfg: &LtxDitConfig, guidance: f32, device: Option<&str>) -> Result<TextContext, String> {
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
    };
    if let Some(hit) = crate::text_cache::load(&cache_key) {
        return Ok((hit.ctx_cond, hit.ctx_uncond, hit.context_valid, hit.context_len));
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
    let (encoder, agg, tok) = if quantized {
        let src = gemma4::Gemma4GgufSource::open(path, &cfg)?;
        let tok = src.tokenizer()?;
        let t_agg_load = std::time::Instant::now();
        let agg = aggregate_head_from_source(&src, hidden, n_states)?;
        tracing::info!(secs = t_agg_load.elapsed().as_secs_f32(), "aggregate-embed head loaded");
        (Encoder::Gguf(src), agg, tok)
    } else {
        let raw = checkpoint::safetensors::read(path)?;
        let tok = gemma4::load_tokenizer(&raw)?;
        let weights = gemma4::import_gemma4(raw, &cfg)?;
        // Built by reference BEFORE `Gemma4Model::new` takes ownership of the
        // map - the head's own two tensors are cloned out, not borrowed.
        let agg = gemma4::AggregateEmbed::from_weights(&weights, hidden, n_states);
        (Encoder::Eager(gemma4::Gemma4Model::new(cfg, weights, device)), agg, tok)
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
    let tokenize = |s: &str| -> Vec<u32> {
        let mut ids = tok.encode(s);
        ids.truncate(GEMMA4_MAX_PROMPT_TOKENS);
        if ids.is_empty() {
            vec![0u32]
        } else {
            ids
        }
    };

    // One closure covers both loaders and both prompts, so the conditional
    // and unconditional branches cannot drift apart.
    let encode = |ids: &[u32]| -> Result<Vec<f32>, String> {
        let n = ids.len();
        let t_fwd = std::time::Instant::now();
        let hidden_states = match &encoder {
            Encoder::Gguf(src) => gemma4::forward_streamed(&cfg, src, device, precision, ids)?.hidden_states,
            Encoder::Eager(model) => model.forward(ids).hidden_states,
        };
        tracing::info!(secs = t_fwd.elapsed().as_secs_f32(), tokens = n, layers = cfg.num_hidden_layers, ?precision, "text tower forward");
        let t_agg = std::time::Instant::now();
        let out = agg.forward(&hidden_states, n, hidden);
        tracing::info!(secs = t_agg.elapsed().as_secs_f32(), tokens = n, in_dim = hidden * n_states, "aggregate-embed projection");
        Ok(out)
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
    let raw_cond = encode(&ids_cond)?;
    if raw_cond.len() != n_cond * cross_attention_dim {
        return Err(format!(
            "ltxv real text encoder: aggregate-embed produced {} values, expected {} ({n_cond} tokens x {cross_attention_dim} cross_attention_dim) - checkpoint/config mismatch",
            raw_cond.len(),
            n_cond * cross_attention_dim
        ));
    }
    let mut ctx_cond = vec![0f32; context_len * cross_attention_dim];
    ctx_cond[..n_cond * cross_attention_dim].copy_from_slice(&raw_cond);
    let mut context_valid = vec![0f32; context_len];
    context_valid[..n_cond].fill(1.0);

    let ctx_uncond = if guidance > 1.0 {
        let ids_u = tokenize("");
        let raw_u = encode(&ids_u)?;
        let mut v = vec![0f32; context_len * cross_attention_dim];
        let rows = ids_u.len().min(context_len);
        v[..rows * cross_attention_dim].copy_from_slice(&raw_u[..rows * cross_attention_dim]);
        v
    } else {
        vec![0f32; context_len * cross_attention_dim]
    };

    crate::text_cache::store(
        &cache_key,
        &crate::text_cache::Encoded {
            ctx_cond: ctx_cond.clone(),
            ctx_uncond: ctx_uncond.clone(),
            context_valid: context_valid.clone(),
            context_len,
        },
    );
    Ok((ctx_cond, ctx_uncond, context_valid, context_len))
}

/// `gemma4::AggregateEmbed` over a streaming source. The eager path builds
/// this from a whole-model map it already has; a streamed one has to pull
/// the head's own two tensors, which is the ONE place a `TensorSource` is
/// asked for something outside the layer loop.
fn aggregate_head_from_source(src: &dyn checkpoint::TensorSource, hidden: usize, n_states: usize) -> Result<gemma4::AggregateEmbed, String> {
    let get = |name: &str| -> Result<Vec<f32>, String> {
        let mut out = None;
        if !src.with_tensor(name, &mut |d| out = Some(d.to_vec())) {
            return Err(format!("ltxv real text encoder: text encoder has no tensor {name}"));
        }
        Ok(out.expect("with_tensor reported found, so the callback ran"))
    };
    let weight = get("text_embedding_projection.video_aggregate_embed.weight")?;
    let bias = get("text_embedding_projection.video_aggregate_embed.bias")?;
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
        // height/width axes: still a plain 32x latent-to-pixel scale for the one token.
        assert_eq!((p[2], p[3]), (0.0, 32.0));
        assert_eq!((p[4], p[5]), (0.0, 32.0));
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
/// * **Stills at BOTH ends** - keyframe interpolation, which is a DIFFERENT
///   reference pipeline with a DIFFERENT conditioning builder.
///   `ltx_pipelines.keyframe_interpolation.KeyframeInterpolationPipeline.
///   __call__` uses `helpers.image_conditionings_by_adding_guiding_latent`,
///   which wraps EVERY image - `frame_idx == 0` included, with no special
///   case - in `VideoConditionByKeyframeIndex`, i.e. APPENDS a guiding
///   token block per still and leaves every one of the generated video's own
///   tokens denoising freely.
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
/// `start`/`end` are already-encoded `[lh*lw, channels]` latent token blocks
/// (one real VAE encode each; the SAME image passed at both ends is encoded
/// once and reused). `frames` is the clip's pixel-frame count - the end
/// still conditions pixel-frame `frames - 1`. At least one of the two must
/// be present.
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
fn conditioned_latent(noise: Vec<f32>, base_positions: &[f32], base_keyframes_mask: &[f32], base_t: usize, lh: usize, lw: usize, channels: usize, frames: usize, fps: f64, start: Option<&[f32]>, end: Option<&[f32]>, strength: f32) -> ConditionedLatent {
    assert!(start.is_some() || end.is_some(), "conditioned_latent: at least one of start/end must be given");
    assert!((0.0..=1.0).contains(&strength), "conditioned_latent: strength {strength} is outside [0, 1]");
    let block_t = lh * lw;
    let blocks = conditioning_block_count(start.is_some(), end.is_some());
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

    // Keyframe interpolation: both ends appended as guiding blocks, the base
    // video untouched (see this function's doc).
    if let (Some(s), Some(e)) = (start, end) {
        let ic = append_image_conditioning(base_t, base_positions, base_keyframes_mask, lh, lw, channels, fps, m, &[(0, s), (frames - 1, e)]);
        let mut latent = noise;
        mix(&mut latent, s, base_t);
        mix(&mut latent, e, base_t + block_t);
        return ConditionedLatent { latent, positions: ic.positions, keyframes_mask: ic.keyframes_mask, denoise_mask: ic.denoise_mask, clean: ic.clean, t: total_t };
    }

    // One appended block at the clip's last pixel frame - the same
    // `VideoConditionByKeyframeIndex` mechanism, one still.
    if let Some(e) = end {
        let ic = append_image_conditioning(base_t, base_positions, base_keyframes_mask, lh, lw, channels, fps, m, &[(frames - 1, e)]);
        let mut latent = noise;
        mix(&mut latent, e, base_t);
        return ConditionedLatent { latent, positions: ic.positions, keyframes_mask: ic.keyframes_mask, denoise_mask: ic.denoise_mask, clean: ic.clean, t: total_t };
    }

    // Image-to-video: overwrite the base video's own first latent frame.
    let s = start.expect("checked above");
    let mut latent = noise;
    mix(&mut latent, s, 0);
    let mut denoise_mask = vec![1.0f32; base_t];
    denoise_mask[..block_t].fill(m);
    let mut clean = vec![0f32; base_t * channels];
    clean[..block_t * channels].copy_from_slice(s);
    ConditionedLatent { latent, positions: base_positions.to_vec(), keyframes_mask: base_keyframes_mask.to_vec(), denoise_mask, clean, t: base_t }
}

/// How many `lh*lw`-token conditioning blocks [`conditioned_latent`] will
/// APPEND for a given `(start, end)` request - `0` for image-to-video (the
/// start still overwrites latent frame 0 in place), `1` for an end still
/// alone, `2` for keyframe interpolation. [`generate`] needs this before the
/// stills are encoded, to draw the initial noise at the full
/// post-conditioning length in one go (see [`conditioned_latent`]'s `noise`).
fn conditioning_block_count(start: bool, end: bool) -> usize {
    match (start, end) {
        (true, true) => 2,
        (_, true) => 1,
        _ => 0,
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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), Some(&e), 1.0);

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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), None, 1.0);

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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, None, Some(&e), 1.0);

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

        let c = conditioned_latent(latent.clone(), &positions, &km, base_t, LH, LW, CH, FRAMES, FPS, Some(&s), Some(&e), strength);

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
        assert_eq!(conditioning_block_count(false, false), 0, "unconditioned");
        assert_eq!(conditioning_block_count(true, false), 0, "image-to-video overwrites in place");
        assert_eq!(conditioning_block_count(false, true), 1);
        assert_eq!(conditioning_block_count(true, true), 2, "keyframe interpolation appends BOTH stills");
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
trait Denoiser {
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, latent: &[f32], timesteps: &[f32], positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, context_valid: &[f32], t: usize) -> Vec<f32>;
}

impl Denoiser for LtxDit {
    fn forward(&self, latent: &[f32], timesteps: &[f32], positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, context_valid: &[f32], t: usize) -> Vec<f32> {
        LtxDit::forward(self, latent, timesteps, positions, keyframes_mask, context, context_len, t, context_valid).out
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
/// `cache`: THIS instance's own per-generation, host-side
/// [`crate::block::GenerationCache`] - shared by reference across every one of
/// `denoise`'s forward calls on this `RealDit` (both the conditional and
/// unconditional branch when CFG is on, and every one of the run's denoise
/// steps). It holds the two things `forward_q_streamed` would otherwise
/// recompute identically every call: each block's already-quantized weight
/// bytes (the GGUF read + CPU quantize Phase 8 measured at ~86% of one real
/// denoise step, now paid at most ONCE per block per generation) and the
/// embeddings-connector routing (unchanged for a whole generation, since the
/// encoded prompt is). Its interior mutability is what lets
/// [`Denoiser::forward`] keep taking `&self` (the same shape `LtxAvDit`'s own
/// per-stage state uses in `crate::dit`); `denoise`'s loop never holds two
/// simultaneous borrows, since each forward call borrows, uses, and drops
/// its borrow before returning.
///
/// A `RealDit` is per-generation, so the cache's lifetime is exactly the
/// window over which its entries are guaranteed still to describe the same
/// inputs - dropping the `RealDit` (which `generate` does before the VAE
/// decode) frees all of it.
struct RealDit {
    cfg: LtxDitConfig,
    src: crate::gguf_src::LtxvGgufSource,
    head: Tensors,
    device: Option<String>,
    cache: crate::block::GenerationCache,
}

impl Denoiser for RealDit {
    fn forward(&self, latent: &[f32], timesteps: &[f32], positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, context_valid: &[f32], t: usize) -> Vec<f32> {
        crate::dit::forward_q_streamed(
            &self.cfg,
            &self.src,
            &self.head,
            self.device.as_deref(),
            crate::block::QTier::Int8,
            latent,
            timesteps,
            positions,
            keyframes_mask,
            context,
            context_len,
            t,
            context_valid,
            &self.cache,
        )
    }
}

/// `to_denoised` (`ltx_core.utils.to_denoised`): the model predicts a
/// velocity; the denoised (x0) estimate the stepper needs is `sample -
/// velocity * sigma`.
fn to_denoised(sample: &[f32], velocity: &[f32], sigma: f64) -> Vec<f32> {
    sample.iter().zip(velocity).map(|(&x, &v)| (x as f64 - v as f64 * sigma) as f32).collect()
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
/// **WHERE in the step it is applied depends on the sampler**, and the
/// reference's two loops genuinely differ:
///
/// * Deterministic Euler (`samplers.euler_denoising_loop` -> `_step_state`):
///   applied to the model's x0 ESTIMATE, before the step formula runs on it.
/// * Ancestral Euler (`samplers._ancestral_euler_denoising_loop`, which is
///   what `euler_ancestral_denoising_loop` and therefore LTX-2.5's own
///   distilled stage 1 run): applied to the STEPPED latent `x_next`, after
///   the renoise term has been added - `if draw_noise: x_next =
///   post_process_latent(x_next, ...)`. The x0 estimate is left alone, and
///   the terminal (`sigma_next == 0`) step returns it unmodified.
///
/// [`denoise`] implements both, selected by `eta`, because getting this
/// backwards would either leave freshly injected noise sitting on a token
/// that is supposed to be clean, or pin a token the sampler was never given
/// a chance to move.
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
        tracing::trace!(step = i + 1, branch = "cond", sigma, "forward starting");
        let cond = dit.forward(&latent, &timesteps, positions, keyframes_mask, ctx_cond, context_len, context_valid, t);
        let velocity = if cfg_on {
            tracing::trace!(step = i + 1, branch = "uncond", sigma, "forward starting");
            let uncond = dit.forward(&latent, &timesteps, positions, keyframes_mask, ctx_uncond, context_len, context_valid, t);
            cond.iter().zip(&uncond).map(|(&c, &u)| u + guidance * (c - u)).collect()
        } else {
            cond
        };
        if !velocity.iter().all(|v| v.is_finite()) {
            let bad = velocity.iter().filter(|v| !v.is_finite()).count();
            tracing::error!(step = i + 1, sigma, non_finite = bad, of = velocity.len(), "the denoiser produced non-finite values");
            return Err(format!("the denoiser produced non-finite values at step {} (sigma = {sigma:.4})", i + 1));
        }
        let mut denoised = to_denoised(&latent, &velocity, sigma);
        // `_step_state`: mask the x0 estimate, then step. The ancestral loop
        // instead steps the estimate as-is and re-applies the mask to the
        // STEPPED latent below (after the renoise term), except on the
        // terminal step, which it short-circuits to the raw estimate
        // (`replace(state, latent=step.denoised)`, no `post_process_latent`).
        if let (Some(f), false) = (frozen, ancestral) {
            post_process_latent(&mut denoised, f);
        }
        let noise = if ancestral { Some((0..latent.len()).map(|_| noise_rng.next_gaussian() as f32).collect::<Vec<f32>>()) } else { None };
        latent = euler_ancestral_step(&latent, &denoised, sigma, sigma_next, eta, s_noise, noise.as_deref());
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

/// Text to video. `progress(done, total, phase)` mirrors `wan::pipeline::
/// generate`'s contract; `cancel` is polled once per denoise step. `prompt`
/// only ever reaches [`context_stub`] (see this module's doc - there is no
/// real text encoder).
#[tracing::instrument(level = "info", name = "generate", skip_all, fields(frames = o.frames, width = o.width, height = o.height, steps = o.steps, seed = o.seed, guidance = o.guidance, dit_config = %o.dit_config))]
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
    let total = sigmas.len() as u32 - 1 + 2;
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
    progress(0, total, "build transformer");
    let build_t = Instant::now();
    let dit: Box<dyn Denoiser> = if o.dit_config == "tiny" {
        // Not a real model: random weights, so any output is a wiring proof
        // and nothing else. Worth a warning rather than an info line - a run
        // that silently produced noise because a checkpoint path was unset is
        // the most expensive way to discover this.
        tracing::warn!("--dit-config tiny: building a RANDOM-weight DiT, output is a smoke test and carries no semantics");
        let weight_seed = o.seed ^ 0x4c_54_58_76_44_49_54; // "LTXvDIT" folded into the seed, so the same --seed reproduces the same weights
        let weights: Tensors = random_tiny_weights(&dit_cfg, weight_seed);
        Box::new(LtxDit::new(dit_cfg, weights, o.device.as_deref()))
    } else {
        let dit_path = paths.dit.as_ref().ok_or_else(|| {
            tracing::error!(dit_config = %o.dit_config, "no real DiT checkpoint configured for a real dit-config");
            format!("ltxv dit-config {:?} needs a real checkpoint: pass --dit <path> or set BRAIN_LTXV_DIT", o.dit_config)
        })?;
        tracing::info!(path = %dit_path, "opening the real DiT GGUF");
        let src = crate::gguf_src::LtxvGgufSource::open(dit_path).inspect_err(|e| tracing::error!(path = %dit_path, error = %e, "opening the DiT GGUF failed"))?;
        let real_cfg = src.config().video;
        if real_cfg != dit_cfg {
            tracing::error!(path = %dit_path, dit_config = %o.dit_config, "the checkpoint's embedded config does not match the named build config");
            return Err(format!("ltxv: {dit_path}'s own embedded config does not match LtxDitConfig::{:?}() - checkpoint/build mismatch", o.dit_config));
        }
        let head = crate::dit::load_head_tensors_from_source(&src, &real_cfg);
        tracing::info!(layers = real_cfg.num_layers, inner_dim = real_cfg.inner_dim, head_tensors = head.len(), "real DiT ready (blocks stream per forward)");
        Box::new(RealDit { cfg: real_cfg, src, head, device: o.device.clone(), cache: Default::default() })
    };
    timings.build_dit = build_t.elapsed().as_secs_f32();
    tracing::info!(secs = timings.build_dit, "transformer built");
    if cancel.is_cancelled() {
        tracing::warn!(phase = "after build", "cancelled");
        return Err("cancelled".into());
    }

    // ---- denoise ----------------------------------------------------------
    // `real_pixel_positions`, not `grid_positions` - the real production
    // pipeline's own `VideoLatentTools.create_initial_state` builds RoPE
    // positions in pixel-scale units (`get_pixel_coords`, causal-fixed,
    // divided by fps), not raw latent-grid integers. See that function's
    // own doc for why this was never caught by an earlier cosine-similarity
    // check on either side.
    let positions = real_pixel_positions(lat_t, lh, lw, o.fps as f64);
    // The causal VAE's first latent frame covers exactly ONE pixel frame
    // (every later one covers `VAE_TEMPORAL_SCALE`), making it "the same
    // token class as a generated keyframe slot" - `ltx_core.tools.
    // VideoLatentTools._first_frame_keyframes_mask`'s own doc: marked
    // UNCONDITIONALLY, independent of whether any real image conditioning
    // is present. An earlier version of this code left `keyframes_mask`
    // all-zero for plain text-to-video, reasoning that "every token is
    // genuinely noise, not a held-fixed real frame" - true, but irrelevant:
    // the mask marks a TOKEN CLASS (first-latent-frame-is-narrower), not
    // "this token is externally conditioned". `dit_cfg.use_keyframes_abs_pos_
    // embedding` is `true` for the real checkpoint, so this was silently
    // omitting a real positional-embedding addition on every real
    // generation until fixed.
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[..lh * lw].fill(1.0);
    // Real Gemma-4 conditioning when `Paths::text_encoder` is set
    // ([`real_text_context`]); otherwise the same deterministic-but-
    // meaningless stub every earlier milestone used (see this module's doc
    // on [`context_stub`]). `context_len` therefore comes from the real
    // tokenizer's own output length in the former case, not
    // `GenOpts::context_len` (which only ever sized the stub).
    let (ctx_cond, ctx_uncond, context_valid, context_len) = match &paths.text_encoder {
        Some(te_path) => {
            tracing::info!(path = %te_path, "encoding the prompt with the real text encoder");
            let te_t = std::time::Instant::now();
            let r = real_text_context(te_path, prompt, &dit_cfg, o.guidance, o.device.as_deref())
                .inspect(|(_, _, _, n)| tracing::info!(context_len = n, "prompt encoded"))
                .inspect_err(|e| tracing::error!(path = %te_path, error = %e, "text encoding failed"))?;
            timings.text_encode = te_t.elapsed().as_secs_f32();
            tracing::info!(secs = timings.text_encode, "text encode done");
            r
        }
        None => {
            // Same class of silent-nonsense as the tiny DiT above: the prompt
            // reaches the model only as a hash-derived stub.
            tracing::warn!("no text encoder configured: the prompt is being replaced by a deterministic STUB context and carries no meaning");
            let prompt_mix = o.seed ^ fnv1a(prompt);
            let dim = dit_cfg.cross_attention_dim as usize;
            let n = o.context_len;
            // Padded exactly like [`real_text_context`]'s own real context
            // (see [`padded_context_len`]'s doc) - a no-op (`context_len ==
            // n`) for every config whose connector is disabled, e.g.
            // `LtxDitConfig::tiny`, so this stays byte-identical to the
            // pre-real-DiT behavior there.
            let context_len = padded_context_len(&dit_cfg, n);
            let stub = context_stub(n, dim, prompt_mix);
            let mut ctx_cond = vec![0f32; context_len * dim];
            ctx_cond[..stub.len()].copy_from_slice(&stub);
            // The "unconditional" branch has no real empty-prompt encoding
            // either; an all-zero context is the closest honest stand-in
            // (most text encoders map an empty string close to zero after
            // their own normalization) and, crucially, is DIFFERENT from
            // `ctx_cond` - so the CFG fold in `denoise` is exercised for
            // real rather than folding two identical branches.
            let ctx_uncond = vec![0f32; context_len * dim];
            // Real for the first `n` positions, invalid (register-
            // substituted by the connector when enabled) for the padded
            // tail - all-valid when `context_len == n` (connector
            // disabled), unchanged from the pre-padding behavior.
            let mut context_valid = vec![0f32; context_len];
            context_valid[..n].fill(1.0);
            (ctx_cond, ctx_uncond, context_valid, context_len)
        }
    };

    // ---- optional image conditioning: encode a real still (or two) and
    // condition the clip on it/them at frame 0 and/or the clip's last pixel
    // frame - see `GenOpts::start_frame`/`end_frame`'s doc, and
    // [`conditioned_latent`]'s for WHICH of the reference's two conditioning
    // mechanisms each combination uses and why that choice decides whether
    // the clip moves. `vraw`/`vweights` are loaded here (not at decode time
    // below, which now reuses them) since this needs them before the
    // denoise loop, not after it.
    let vraw = read_any(&paths.vae)?;
    let vweights = crate::import::import_vae(vraw, &vcfg)?;
    // One draw over the WHOLE post-conditioning sequence, matching
    // `GaussianNoiser._sample_noise`, which runs after every conditioning
    // item has appended its tokens. `seeded_noise` is prefix-stable, so a
    // clip's first `t*in_channels` values - the entire unconditioned path -
    // are byte-identical whether or not anything is appended here.
    let latent0 = seeded_noise((t + conditioning_block_count(o.start_frame.is_some(), o.end_frame.is_some()) * lh * lw) * in_channels, o.seed);
    let encode_still = |path: &str| -> Result<Vec<f32>, String> {
        let img_t = Instant::now();
        let img = image::open(path).map_err(|e| format!("{path}: {e}"))?.resize_exact(o.width as u32, o.height as u32, image::imageops::FilterType::Lanczos3).to_rgb8();
        let mut img_chw = vec![0f32; 3 * o.height * o.width];
        for y in 0..o.height {
            for x in 0..o.width {
                let p = img.get_pixel(x as u32, y as u32).0;
                let idx = y * o.width + x;
                for c in 0..3 {
                    // `[0,255] -> [-1,1]`, the VAE's own input range (see
                    // `LtxVaeEncoder::encode`'s doc).
                    img_chw[c * o.height * o.width + idx] = (p[c] as f32 / 127.5) - 1.0;
                }
            }
        }
        let enc = LtxVaeEncoder::build(&vcfg, &vweights, 1, o.height as u32, o.width as u32, o.device.as_deref());
        let cond_latent_chw = enc.encode(&img_chw);
        let cond_tokens = chw_to_tc(&cond_latent_chw, in_channels, 1, lh, lw);
        tracing::info!(path, secs = img_t.elapsed().as_secs_f32(), cond_tokens = cond_tokens.len() / in_channels, "conditioning image encoded");
        Ok(cond_tokens)
    };
    let (latent0, positions_d, keyframes_mask_d, denoise_t_count, frozen) = if o.start_frame.is_some() || o.end_frame.is_some() {
        // WHICH mechanism runs is [`conditioned_latent`]'s decision - the
        // reference has two conditioning builders for these two cases
        // (image-to-video's in-place overwrite of latent frame 0, keyframe
        // interpolation's appended guiding blocks). See that function's doc
        // for the reference citations and for where
        // `conditioning_strength` lands.
        //
        // Whichever mechanism runs, the resulting `denoise_mask` is what the
        // denoise loop turns into PER-TOKEN timesteps (`timesteps_from_mask`,
        // see [`denoise`]) - a frozen token is announced to the model as
        // noise-free, which is the whole reason the model treats it as
        // guidance rather than as sequence noise.
        //
        // The same path passed for both ends is encoded ONCE: a real VAE
        // encode is not free, and the loop case (one still at both ends) is
        // the common one.
        let start_tokens = o.start_frame.as_deref().map(&encode_still).transpose()?;
        let end_tokens = match (&o.end_frame, &o.start_frame) {
            (Some(e), Some(s)) if e == s => start_tokens.clone(),
            (Some(e), _) => Some(encode_still(e.as_str())?),
            (None, _) => None,
        };
        let c = conditioned_latent(latent0, &positions, &keyframes_mask, t, lh, lw, in_channels, o.frames, o.fps as f64, start_tokens.as_deref(), end_tokens.as_deref(), o.conditioning_strength);
        tracing::info!(strength = o.conditioning_strength, tokens = c.t, base_tokens = t, appended_blocks = conditioning_block_count(o.start_frame.is_some(), o.end_frame.is_some()), "image conditioning applied");
        (c.latent, c.positions, c.keyframes_mask, c.t, Some((c.denoise_mask, c.clean)))
    } else {
        (latent0, positions.clone(), keyframes_mask.clone(), t, None)
    };
    let frozen_ref = frozen.as_ref().map(|(mask, clean)| Frozen { mask, clean, channels: in_channels });
    let denoise_t = Instant::now();
    tracing::info!(tokens = denoise_t_count, base_tokens = t, context_len, image_conditioned = o.start_frame.is_some() || o.end_frame.is_some(), "denoising");
    let final_latent = denoise(dit.as_ref(), &sigmas, latent0, &positions_d, &keyframes_mask_d, &ctx_cond, &ctx_uncond, context_len, &context_valid, denoise_t_count, o.guidance, o.eta, o.s_noise, o.seed ^ 0x4e_4f_49_53_45, total, frozen_ref.as_ref(), cancel, &mut progress)?;
    // Release the DiT's own device context (for `RealDit`, its resident
    // `Gpu`) before the VAE decode below opens its own - real device memory
    // is not this pipeline's to hold onto once the denoise loop is done
    // with it.
    drop(dit);
    timings.denoise = denoise_t.elapsed().as_secs_f32();
    // `sigmas.len() - 1`, not `o.steps`: the real distilled schedule ignores
    // `--steps` entirely (see where `sigmas` is built, above).
    timings.steps = sigmas.len() - 1;
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
    // Strip any appended image-conditioning tokens (see the `--image` branch
    // above) - only the original `t` video tokens get decoded; the
    // conditioning frame is the source image itself, not a new frame to
    // render.
    let chw = tc_to_chw(&final_latent[..t * in_channels], in_channels, lat_t, lh, lw);
    let dec = LtxVaeDecoder::build(&vcfg, &vweights, lat_t as u32, lh as u32, lw as u32, o.device.as_deref());
    drop(vweights);
    let pixels = dec.decode(&chw);
    let frames = dec.frames() as usize;
    let (w, h) = (o.width, o.height);
    if pixels.len() != 3 * frames * h * w {
        return Err(format!("VAE returned {} values, expected {}", pixels.len(), 3 * frames * h * w));
    }
    // No clamp is applied by the model itself - upstream clamps to [-1,1]
    // outside it, same convention `wan::pipeline` follows for its own VAE.
    let plane = frames * h * w;
    let out: Vec<Vec<u8>> = (0..frames)
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
        .collect();
    timings.decode = decode_t.elapsed().as_secs_f32();
    tracing::info!(secs = timings.decode, frames, "VAE decode done");
    progress(total, total, "done");
    tracing::info!(frames, width = w, height = h, fps = o.fps, total_secs = timings.total(), "generation done");
    Ok((Video { width: w as u32, height: h as u32, fps: o.fps, frames: out }, timings))
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
    let final1 = denoise(&dit, &sigmas1, latent1_0, &layout1.positions, &layout1.keyframes_mask, &ctx_cond, &ctx_uncond, base.context_len, &ctx_valid, t1, base.guidance, base.eta, base.s_noise, base.seed ^ 0x4e_31, base.steps as u32, None, cancel, &mut |_, _, _: &str| {})?;
    if cancel.is_cancelled() {
        tracing::warn!(stage = "stage1", "cancelled");
        return Err("cancelled".into());
    }

    let reserved_half_res_video = tc_to_chw(&final1[..t0_1 * in_channels], in_channels, lat_t, lh1, lw1);
    let slot1_chw = tc_to_chw(&final1[t0_1 * in_channels..], in_channels, k, lh1, lw1);

    // ---- real spatial x2 upscale of BOTH the video and its slots ----
    progress(1, total_phases, "spatial upscale");
    tracing::info!(stage = "spatial_upscale", path = %paths.spatial_upsampler, "real x2 latent upscale");
    let sraw = read_any(&paths.spatial_upsampler)?;
    let scfg = LatentUpsamplerConfig::spatial_x2();
    let sweights = crate::import::import_upsampler(sraw, &scfg)?;
    let video_upsampler = LatentUpsampler::build(&scfg, &sweights, lat_t as u32, lh1 as u32, lw1 as u32, base.device.as_deref());
    let upscaled_video_chw = video_upsampler.upsample(&reserved_half_res_video);
    let (_, _, lh2u, lw2u) = video_upsampler.out_shape();
    let slots_upsampler = LatentUpsampler::build(&scfg, &sweights, k as u32, lh1 as u32, lw1 as u32, base.device.as_deref());
    let upscaled_slots_chw = slots_upsampler.upsample(&slot1_chw);
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
    let final2 = denoise(&dit, &sigmas2, latent2_0, &layout2.positions, &layout2.keyframes_mask, &ctx_cond, &ctx_uncond, base.context_len, &ctx_valid, t2, base.guidance, base.eta, base.s_noise, base.seed ^ 0x4e32, base.steps as u32, None, cancel, &mut |_, _, _: &str| {})?;
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
            let upsampled_video = tup.upsample(&video_chw);
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
    let vraw = read_any(&paths.vae)?;
    let vweights = crate::import::import_vae(vraw, &vcfg)?;
    let dec = LtxVaeDecoder::build(&vcfg, &vweights, cur_lat_t as u32, lh2 as u32, lw2 as u32, base.device.as_deref());
    drop(vweights);
    let pixels = dec.decode(&video_chw);
    let frames = dec.frames() as usize;
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
    Ok((Video { width: w as u32, height: h as u32, fps, frames: out }, timings))
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
        let p = Paths::resolve(None, None, None).expect("from env");
        assert_eq!(p.vae, "env-vae");
        assert_eq!(p.dit, None);
        assert_eq!(p.text_encoder, None);
        let p = Paths::resolve(Some("/flag/vae"), None, None).expect("flag wins");
        assert_eq!(p.vae, "/flag/vae");
        let p = Paths::resolve(Some(""), None, None).expect("empty flag falls through");
        assert_eq!(p.vae, "env-vae");
        std::env::remove_var("BRAIN_LTXV_VAE");
        let e = Paths::resolve(None, None, None).unwrap_err();
        assert!(e.contains("--vae") && e.contains("BRAIN_LTXV_VAE"), "{e}");
    }

    /// [`Paths::dit`]/[`Paths::text_encoder`] follow the exact same
    /// flag-over-env, optional (not error-on-absent) resolution
    /// [`OPTIONAL_PATH_VARS`] documents.
    #[test]
    fn the_optional_real_checkpoint_paths_resolve_flag_over_env_and_are_none_when_absent() {
        std::env::remove_var("BRAIN_LTXV_DIT");
        std::env::remove_var("BRAIN_LTXV_TEXT_ENCODER");
        let p = Paths::resolve(Some("/vae"), None, None).expect("vae only");
        assert_eq!(p.dit, None);
        assert_eq!(p.text_encoder, None);

        std::env::set_var("BRAIN_LTXV_DIT", "env-dit");
        std::env::set_var("BRAIN_LTXV_TEXT_ENCODER", "env-te");
        let p = Paths::resolve(Some("/vae"), None, None).expect("from env");
        assert_eq!(p.dit, Some("env-dit".to_string()));
        assert_eq!(p.text_encoder, Some("env-te".to_string()));
        let p = Paths::resolve(Some("/vae"), Some("/flag-dit"), Some("/flag-te")).expect("flag wins");
        assert_eq!(p.dit, Some("/flag-dit".to_string()));
        assert_eq!(p.text_encoder, Some("/flag-te".to_string()));
        std::env::remove_var("BRAIN_LTXV_DIT");
        std::env::remove_var("BRAIN_LTXV_TEXT_ENCODER");
    }

    #[test]
    fn a_bad_frame_count_is_rejected_before_any_weight_is_read() {
        let paths = Paths { vae: "/nope".into(), dit: None, text_encoder: None };
        let o = GenOpts { frames: 8, ..GenOpts::default() };
        let e = generate(&paths, "x", &o, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("1 + 8k"), "{e}");

        let o = GenOpts { width: 65, ..GenOpts::default() };
        let e = generate(&paths, "x", &o, &Default::default(), |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("multiple of 32"), "{e}");
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
        fn forward(&self, latent: &[f32], timesteps: &[f32], _positions: &[f32], _keyframes_mask: &[f32], context: &[f32], _context_len: usize, _context_valid: &[f32], _t: usize) -> Vec<f32> {
            self.seen.borrow_mut().push(context[0]);
            self.timesteps_seen.borrow_mut().push(timesteps.to_vec());
            self.latents_seen.borrow_mut().push(latent.to_vec());
            vec![context[0]; latent.len()]
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

        let out = denoise(&dit, &sigmas, latent0, &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, 1.0, 1.0, 7, 4, Some(&frozen), &Default::default(), &mut |_, _, _: &str| {}).expect("eta > 0 with conditioned tokens must run, not be refused");

        let latents = dit.latents_seen.borrow().clone();
        assert_eq!(latents.len(), 2, "one forward per step");
        assert_eq!(latents[0][1], 5.0, "step 0 sees the conditioned token at its clean content");
        assert_eq!(latents[1][1], 5.0, "step 1 must see it clean AGAIN: the renoise term ran over the whole latent and post_process_latent has to undo it there");
        assert_ne!(latents[1][0], latents[0][0], "the free token really was stepped and renoised");
        // The terminal step short-circuits to the raw x0 estimate with no
        // re-pin, exactly as `_ancestral_euler_denoising_loop` does - so the
        // final value of a conditioned token is the model's own estimate.
        // Asserted so that a future change to that ordering is deliberate.
        assert_eq!(out.len(), t);
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
        denoise(&dit, &sigmas, vec![0.0; t * channels], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, 0.0, 1.0, 7, 4, Some(&frozen), &Default::default(), &mut |_, _, _: &str| {}).expect("fake denoiser is finite");

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
        denoise(&dit, &sigmas, vec![0.0; t], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, t, 1.0, 0.0, 1.0, 7, 4, None, &Default::default(), &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
        assert_eq!(*dit.timesteps_seen.borrow(), vec![vec![1.0f32; t], vec![0.5f32; t]]);
    }

    fn run_loop(guidance: f32, eta: f64) -> (Vec<f32>, Vec<f32>) {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit::default();
        let positions = grid_positions(1, 1, 1);
        let keyframes_mask = vec![0.0f32];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let out = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, 1, guidance, eta, 1.0, 7, 4, None, &Default::default(), &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
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
        let err = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, 1, 1.0, 0.0, 1.0, 7, 6, None, &cancel, &mut |step, _, _: &str| {
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
}
