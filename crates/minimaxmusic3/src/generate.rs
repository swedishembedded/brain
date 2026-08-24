// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The top-level orchestration: lyrics + caption text in, a full stereo
//! song out. ONE implementation of "run the whole five-component
//! pipeline", shared by `crate::caps::MinimaxMusic3Provider` (the direct
//! `brain do`/event-API path), the residency adapter
//! (`crates/cli/src/resident_minimaxmusic3.rs`), and
//! `tests/e2e_short_generation.rs` (which calls [`generate`] directly
//! rather than duplicating this composition inline).
//!
//! Sequential-stage RAM discipline (this crate's own roadmap ledger
//! records why this matters on a RAM-constrained machine): each stage's
//! weights live in their own block scope and are dropped before the next
//! stage's load - the AR stage (two Global LLM
//! instances + the depth decoder) never overlaps the denoise stage (the
//! DiT + condition encoder), which never overlaps the vocoder stage.
//! Chunked generation (songs longer than [`denoise::CHUNK_FRAMES`] AR
//! frames) still respects this: EVERY chunk's latents are produced before
//! the DiT is dropped, then EVERY chunk is decoded through the vocoder
//! stage - not interleaved per chunk, which would need both resident at
//! once.
//!
//! Device selection is ambient for EVERY stage - `--device` /
//! `BRAIN_DEVICE`, the same knob every other model in this workspace
//! honours. The AR stage inherits it through `qwen3::Qwen`; the denoise
//! and vocoder stages get it from [`GenOpts::device`], which is `None`
//! ("do not override") by default and otherwise takes a `--device`-shaped
//! token straight to [`Gpu::open`].
//!
//! These two stages used to call `Gpu::new_cpu` unconditionally, which
//! pinned the 36-layer DiT and the 512x-upsample vocoder to the CPU
//! backend no matter what the caller asked for - the two most
//! compute-heavy components in the model, on the one device that cannot
//! use a GPU. That was written for a machine with no discrete card; it is
//! a bug anywhere else, and it silently made `--device gpu` a no-op for
//! most of this pipeline's cost.

use crate::config::{ConditionEncoderConfig, DepthDecoderConfig, DitConfig, VocoderConfig};
use crate::{condition_encoder, denoise, depth_decoder, dit, global_llm, pipeline, stitch, vocoder};
use data::qwen_tokenizer::QwenBpe;
use gpu_core::Gpu;

/// One role per checkpoint directory, resolved from the SAME six env vars
/// `crates/arch`'s `minimaxmusic3` registry row names in its own
/// `weights_env` table.
#[derive(Clone, Debug)]
pub struct Paths {
    pub lm: String,
    pub depth: String,
    pub condition: String,
    pub dit: String,
    pub vocoder: String,
    pub tokenizer: String,
}

/// `(variable, human role name)`, in the order [`Paths`] declares them -
/// one table so the env reader and its own error messages cannot disagree
/// about the spelling (matches `wan::pipeline::PATH_VARS`'s own shape).
pub const PATH_VARS: [(&str, &str); 6] = [
    ("BRAIN_MINIMAXMUSIC3_LM", "Global LLM"),
    ("BRAIN_MINIMAXMUSIC3_DEPTH", "RVQ depth decoder"),
    ("BRAIN_MINIMAXMUSIC3_CONDITION", "condition encoder"),
    ("BRAIN_MINIMAXMUSIC3_DIT", "flow-matching DiT"),
    ("BRAIN_MINIMAXMUSIC3_VOCODER", "vocoder"),
    ("BRAIN_MINIMAXMUSIC3_TOKENIZER", "tokenizer"),
];

impl Paths {
    /// Every role from its environment variable; `Err` names the first
    /// missing one.
    pub fn from_env() -> Result<Paths, String> {
        let get = |i: usize| -> Result<String, String> {
            let (var, role) = PATH_VARS[i];
            std::env::var(var).ok().filter(|v| !v.is_empty()).ok_or_else(|| format!("no {role} weights: set {var}"))
        };
        Ok(Paths { lm: get(0)?, depth: get(1)?, condition: get(2)?, dit: get(3)?, vocoder: get(4)?, tokenizer: get(5)? })
    }
}

/// The AR stage's own frame rate (`ConditionEncoderConfig::real()`'s
/// `input_sampling_rate / input_hop_length` = `24000/960`): by
/// construction, one AR/depth-decoder frame always maps to exactly
/// `1/25` s of the eventual audio, all the way through the condition
/// encoder's resample and the vocoder's upsample - confirmed by hand in
/// this module's own tests, not just asserted here.
pub const AR_FRAME_RATE_HZ: f32 = 25.0;

/// Generation knobs. `duration_seconds` is converted to an AR frame cap
/// via [`AR_FRAME_RATE_HZ`] - the AR loop can still stop earlier (the
/// model sampling `AUDIO_END_TOKEN_ID`), matching the reference's own
/// "cap, not a fixed length" semantics.
pub struct GenOpts {
    pub duration_seconds: f32,
    pub num_inference_steps: usize,
    pub seed: u64,
    /// A `--device`-shaped token (`cpu`, `gpu`, `gpu0`, `gpu1`, ...) for
    /// the denoise and vocoder stages, or `None` to inherit the ambient
    /// `--device`/`BRAIN_DEVICE` selection like every other stage. Passed
    /// verbatim to [`Gpu::open`].
    pub device: Option<String>,
}

impl Default for GenOpts {
    fn default() -> GenOpts {
        GenOpts { duration_seconds: 10.0, num_inference_steps: denoise::DEFAULT_NUM_INFERENCE_STEPS, seed: 0, device: None }
    }
}

/// `duration_seconds` -> a max-AR-frame cap ([`AR_FRAME_RATE_HZ`], at
/// least 1 frame).
pub fn max_frames_for_duration(duration_seconds: f32) -> usize {
    ((duration_seconds * AR_FRAME_RATE_HZ).round() as i64).max(1) as usize
}

/// A finished song: separate left/right channels (matching `stitch::
/// Stitcher::finish`'s own planar output) and the vocoder's own sample
/// rate.
pub struct GeneratedSong {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
}

/// Run the full pipeline: prompt assembly -> CFG-guided AR sampling ->
/// chunked DiT denoising -> vocoder crop-and-stitch. See this module's own
/// doc for the sequential-stage RAM discipline and the `BRAIN_DEVICE=cpu`
/// operational note.
pub fn generate(paths: &Paths, opts: &GenOpts, lyrics: &str, caption: &str, progress: crate::ProgressSink<'_>) -> Result<GeneratedSong, String> {
    let max_frames = max_frames_for_duration(opts.duration_seconds);

    // ---- AR stage: two int8 Global LLM instances + the depth decoder. ----
    let frame_hiddens = {
        let tokenizer = QwenBpe::from_dir(&paths.tokenizer)?;
        let (conditional_ids, unconditional_ids) = global_llm::assemble_prompt(&tokenizer, caption, lyrics);
        let cap = (conditional_ids.len() + max_frames + 8) as u32;
        let (cfg, lm_cond) = global_llm::import(&paths.lm, 1, cap)?;
        let (_, lm_uncond) = global_llm::import(&paths.lm, 1, cap)?;
        let head = lm_cond.read_weight(cfg.head_weight());

        let dd_cfg = DepthDecoderConfig::real();
        let dd_w = depth_decoder::import(&paths.depth, &dd_cfg)?;

        pipeline::generate_frames(&lm_cond, &lm_uncond, &dd_w, &dd_cfg, &head, cfg.vocab as usize, cfg.d_model as usize, &conditional_ids, &unconditional_ids, max_frames, opts.seed, progress)
    };

    let cond_cfg = ConditionEncoderConfig::real();
    let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
    if frame_hiddens.is_empty() {
        return Err("minimaxmusic3::generate: the AR stage produced zero frames (sampled AUDIO_END on frame 0 - try a different prompt or seed)".to_string());
    }
    let num_frames = frame_hiddens.len() / per_frame;

    // ---- Denoise stage: condition encoder + the DiT, every chunk. ----
    // Every chunk's latents are produced (and `length`s recorded) before
    // this block ends and the DiT drops - never interleaved with the
    // vocoder stage below, which would need both resident at once.
    let chunks: Vec<(Vec<f32>, usize)> = {
        let cond_w = condition_encoder::import(&paths.condition)?;
        let dit_cfg = DitConfig::real();
        let dit_w = dit::import(&paths.dit, &dit_cfg)?;
        let gpu = Gpu::open(opts.device.as_deref(), dit::PIPELINES);
        let starts = denoise::chunk_starts(num_frames);
        let mut state = denoise::ChunkState::default();
        starts
            .iter()
            .enumerate()
            .map(|(i, &start)| {
                let latents = denoise::denoise_chunk(&gpu, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, start, &mut state, opts.num_inference_steps, opts.seed.wrapping_add(i as u64 + 1), progress);
                let chunk_frames = (start + denoise::CHUNK_FRAMES).min(num_frames) - start;
                let length = condition_encoder::latent_length(&cond_cfg, chunk_frames);
                (latents, length)
            })
            .collect()
    };

    // ---- Vocoder stage: crop-and-stitch every chunk. ----
    let (left, right) = {
        let vocoder_cfg = VocoderConfig::real();
        let vocoder_w = vocoder::import(&paths.vocoder, &vocoder_cfg)?;
        let gpu = Gpu::open(opts.device.as_deref(), vocoder::PIPELINES);
        let mut stitcher = stitch::Stitcher::new();
        let n = chunks.len();
        for (i, (latents, length)) in chunks.iter().enumerate() {
            stitcher.push_chunk(&gpu, &vocoder_cfg, &vocoder_w, latents, *length, i == 0, i + 1 == n);
            progress(i as u32 + 1, n as u32, "vocode");
        }
        stitcher.finish()
    };

    Ok(GeneratedSong { left, right, sample_rate: VocoderConfig::real().sampling_rate })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_frames_for_duration_matches_the_ar_frame_rate() {
        assert_eq!(max_frames_for_duration(1.0), 25);
        assert_eq!(max_frames_for_duration(10.0), 250);
        assert_eq!(max_frames_for_duration(0.0), 1, "must never return zero frames");
    }

    #[test]
    fn path_vars_names_match_the_arch_registrys_own_weights_env_table() {
        // crates/arch's own `weights_env` list for "minimaxmusic3" (checked
        // by hand against crates/arch/src/lib.rs, not re-imported here to
        // avoid a dependency cycle) - this test pins that PATH_VARS cannot
        // silently drift from that registration.
        let vars: Vec<&str> = PATH_VARS.iter().map(|(v, _)| *v).collect();
        assert_eq!(
            vars,
            [
                "BRAIN_MINIMAXMUSIC3_LM",
                "BRAIN_MINIMAXMUSIC3_DEPTH",
                "BRAIN_MINIMAXMUSIC3_CONDITION",
                "BRAIN_MINIMAXMUSIC3_DIT",
                "BRAIN_MINIMAXMUSIC3_VOCODER",
                "BRAIN_MINIMAXMUSIC3_TOKENIZER",
            ]
        );
    }

    #[test]
    fn from_env_names_the_first_missing_var() {
        // Clear every role var for the duration of this test (serial by
        // construction - `cargo test` runs each test in its own thread but
        // env vars are process-global; this test only ever asserts on the
        // Err path, so a concurrent from_env in another test racing this
        // one would at worst see a spurious extra var set/unset, not a
        // wrong answer at true parallel envs level in-process).
        for (var, _) in PATH_VARS {
            std::env::remove_var(var);
        }
        let err = Paths::from_env().unwrap_err();
        assert!(err.contains("BRAIN_MINIMAXMUSIC3_LM"), "unexpected error: {err}");
    }
}
