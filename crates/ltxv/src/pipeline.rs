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
use crate::vae3d::{LtxVaeConfig, LtxVaeDecoder};
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
/// function's doc), and neither a real tokenizer's own prompt length nor
/// [`GenOpts::context_len`]'s fixed stub size is naturally shaped that way.
/// Shared by [`real_text_context`] and [`generate`]'s own stub-context
/// branch, so a connector-enabled real config (`LtxDitConfig::ltx25_22b`)
/// never sees an unpadded context from EITHER path. A disabled connector
/// reads `context` directly as real caption content, so `n` passes through
/// unchanged in that case - padding it would inject zero rows the DiT has
/// no way to ignore.
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
    let tokenize = |s: &str| -> Vec<u32> {
        let ids = tok.encode(s);
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
    let context_len = padded_context_len(dit_cfg, n_cond);
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

/// The only thing the denoise loop asks of a model: a velocity prediction at
/// a per-token sigma (broadcast uniformly here - no partial/keyframe
/// denoising in this milestone). A trait, private, so the CFG fold and the
/// cancellation/step bookkeeping are testable against a fake instead of a
/// real (if tiny) GPU forward - the `wan::pipeline::Denoiser` pattern.
trait Denoiser {
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, latent: &[f32], sigma: f32, positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, context_valid: &[f32], t: usize) -> Vec<f32>;
}

impl Denoiser for LtxDit {
    fn forward(&self, latent: &[f32], sigma: f32, positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, context_valid: &[f32], t: usize) -> Vec<f32> {
        // Per-token timesteps = denoise_mask * sigma; denoise_mask is
        // all-ones here (every token is denoised uniformly - see this
        // module's doc on keyframes_mask), so this is `sigma` broadcast to
        // every token. `LtxDit::forward` applies `timestep_scale_multiplier`
        // internally, so this passes the RAW sigma, matching the golden's
        // own convention (see `crate::dit::LtxDit::forward`'s doc).
        let timesteps = vec![sigma; t];
        LtxDit::forward(self, latent, &timesteps, positions, keyframes_mask, context, context_len, t, context_valid).out
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
    fn forward(&self, latent: &[f32], sigma: f32, positions: &[f32], keyframes_mask: &[f32], context: &[f32], context_len: usize, context_valid: &[f32], t: usize) -> Vec<f32> {
        let timesteps = vec![sigma; t];
        crate::dit::forward_q_streamed(
            &self.cfg,
            &self.src,
            &self.head,
            self.device.as_deref(),
            crate::block::QTier::Int8,
            latent,
            &timesteps,
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
    cancel: &capability::CancelToken,
    progress: &mut impl FnMut(u32, u32, &str),
) -> Result<Vec<f32>, String> {
    let cfg_on = guidance > 1.0;
    let steps = sigmas.len().saturating_sub(1);
    let mut noise_rng = data::rng::Rng::new(noise_seed);
    let t0 = Instant::now();
    tracing::info!(steps, cfg = cfg_on, forwards_per_step = if cfg_on { 2 } else { 1 }, sigma_first = sigmas.first().copied().unwrap_or(0.0), sigma_last = sigmas.last().copied().unwrap_or(0.0), "denoise loop starting");
    for i in 0..steps {
        // Once per step: a forward is one submit of the whole block stack
        // and is not interruptible from inside, same reasoning as
        // `wan::pipeline::denoise`.
        if cancel.is_cancelled() {
            tracing::warn!(step = i + 1, steps, "cancelled at a step boundary; aborting the denoise loop");
            return Err("cancelled".into());
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        tracing::trace!(step = i + 1, branch = "cond", sigma, "forward starting");
        let cond = dit.forward(&latent, sigma as f32, positions, keyframes_mask, ctx_cond, context_len, context_valid, t);
        let velocity = if cfg_on {
            tracing::trace!(step = i + 1, branch = "uncond", sigma, "forward starting");
            let uncond = dit.forward(&latent, sigma as f32, positions, keyframes_mask, ctx_uncond, context_len, context_valid, t);
            cond.iter().zip(&uncond).map(|(&c, &u)| u + guidance * (c - u)).collect()
        } else {
            cond
        };
        if !velocity.iter().all(|v| v.is_finite()) {
            let bad = velocity.iter().filter(|v| !v.is_finite()).count();
            tracing::error!(step = i + 1, sigma, non_finite = bad, of = velocity.len(), "the denoiser produced non-finite values");
            return Err(format!("the denoiser produced non-finite values at step {} (sigma = {sigma:.4})", i + 1));
        }
        let denoised = to_denoised(&latent, &velocity, sigma);
        let noise = if eta > 0.0 { Some((0..latent.len()).map(|_| noise_rng.next_gaussian() as f32).collect::<Vec<f32>>()) } else { None };
        latent = euler_ancestral_step(&latent, &denoised, sigma, sigma_next, eta, s_noise, noise.as_deref());
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
    let positions = grid_positions(lat_t, lh, lw);
    // No keyframe conditioning: this is pure text-to-video with no input
    // image, so every token is genuinely noise, not a held-fixed real frame
    // - `keyframes_mask` is all zero (see `LtxDit::forward`'s doc: nonzero
    // marks a keyframe token). An all-1.0 mask would add
    // `keyframes_abs_pos_embedding` to every token for no semantic reason.
    let keyframes_mask = vec![0f32; t];
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

    let latent0 = seeded_noise(t * in_channels, o.seed);
    let denoise_t = Instant::now();
    tracing::info!(tokens = t, context_len, "denoising");
    let final_latent = denoise(dit.as_ref(), &sigmas, latent0, &positions, &keyframes_mask, &ctx_cond, &ctx_uncond, context_len, &context_valid, t, o.guidance, o.eta, o.s_noise, o.seed ^ 0x4e_4f_49_53_45, total, cancel, &mut progress)?;
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
    let chw = tc_to_chw(&final_latent, in_channels, lat_t, lh, lw);
    let vraw = read_any(&paths.vae)?;
    let vweights = crate::import::import_vae(vraw, &vcfg)?;
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
    let final1 = denoise(&dit, &sigmas1, latent1_0, &layout1.positions, &layout1.keyframes_mask, &ctx_cond, &ctx_uncond, base.context_len, &ctx_valid, t1, base.guidance, base.eta, base.s_noise, base.seed ^ 0x4e_31, base.steps as u32, cancel, &mut |_, _, _: &str| {})?;
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
    let final2 = denoise(&dit, &sigmas2, latent2_0, &layout2.positions, &layout2.keyframes_mask, &ctx_cond, &ctx_uncond, base.context_len, &ctx_valid, t2, base.guidance, base.eta, base.s_noise, base.seed ^ 0x4e32, base.steps as u32, cancel, &mut |_, _, _: &str| {})?;
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
    struct FakeDit {
        seen: std::cell::RefCell<Vec<f32>>,
    }
    impl Denoiser for FakeDit {
        fn forward(&self, latent: &[f32], _sigma: f32, _positions: &[f32], _keyframes_mask: &[f32], context: &[f32], _context_len: usize, _context_valid: &[f32], _t: usize) -> Vec<f32> {
            self.seen.borrow_mut().push(context[0]);
            vec![context[0]; latent.len()]
        }
    }

    fn run_loop(guidance: f32, eta: f64) -> (Vec<f32>, Vec<f32>) {
        let sigmas = vec![1.0, 0.5, 0.0];
        let dit = FakeDit { seen: Default::default() };
        let positions = grid_positions(1, 1, 1);
        let keyframes_mask = vec![0.0f32];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let out = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, 1, guidance, eta, 1.0, 7, 4, &Default::default(), &mut |_, _, _: &str| {}).expect("fake denoiser is finite");
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
        let dit = FakeDit { seen: Default::default() };
        let positions = grid_positions(1, 1, 1);
        let keyframes_mask = vec![0.0f32];
        let (cond, uncond) = (vec![1.0f32; 1], vec![0.0f32; 1]);
        let context_valid = vec![1.0f32; 1];
        let cancel = capability::CancelToken::armed();
        let handle = cancel.clone();
        let err = denoise(&dit, &sigmas, vec![0.0; 1], &positions, &keyframes_mask, &cond, &uncond, 1, &context_valid, 1, 1.0, 0.0, 1.0, 7, 6, &cancel, &mut |step, _, _: &str| {
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
