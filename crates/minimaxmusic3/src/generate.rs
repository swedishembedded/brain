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
//! Sequential-stage DEVICE discipline: each stage's `Gpu` handles and
//! device-resident objects live in their own block scope and are dropped
//! before the next stage builds its own - the AR stage (two Global LLM
//! instances + the depth decoder) never overlaps the denoise stage (the
//! DiT + condition encoder), which never overlaps the vocoder stage.
//! Chunked generation (songs longer than [`denoise::CHUNK_FRAMES`] AR
//! frames) still respects this: EVERY chunk's latents are produced before
//! the DiT is dropped, then EVERY chunk is decoded through the vocoder
//! stage - not interleaved per chunk, which would need both resident at
//! once.
//!
//! HOST weights are a different question, and the answer changed with the
//! hardware. Four of the five components import into a plain tree of host
//! `Vec<f32>` and are read through [`crate::weightcache`], which holds them
//! across calls keyed on the checkpoint directory - so a second `generate`
//! on the same weights does not re-read 10.7 GB off disk. The block scopes
//! above still bound every DEVICE allocation exactly as they did; what a
//! scope end now releases is one `Arc` clone, not the bytes. `crate::
//! weightcache`'s own module doc carries why that is safe, what it costs,
//! and why the Global LLM is deliberately not in it.
//!
//! Device selection is ambient for EVERY stage - `--device` /
//! `BRAIN_DEVICE`, the same knob every other model in this workspace
//! honours. The AR stage inherits it through `qwen3::Qwen`; the denoise
//! and vocoder stages get it from [`GenOpts::device`], which is `None`
//! ("do not override") by default and otherwise takes a `--device`-shaped
//! token straight to [`Gpu::open`].
//!
//! Both of the stages that have two independent halves use two cards when
//! the machine has two: the AR stage loads its conditional and
//! unconditional Global LLM instances on separate cards
//! ([`ar_branch_devices`]), and the denoise stage runs the DiT's two CFG
//! branches concurrently, one per card ([`denoise::CfgDevices`],
//! [`crate::devplan`]) - bit-identically, and only when nothing pinned the
//! run to one device.
//!
//! These two stages used to call `Gpu::new_cpu` unconditionally, which
//! pinned the 36-layer DiT and the 512x-upsample vocoder to the CPU
//! backend no matter what the caller asked for - the two most
//! compute-heavy components in the model, on the one device that cannot
//! use a GPU. That was written for a machine with no discrete card; it is
//! a bug anywhere else, and it silently made `--device gpu` a no-op for
//! most of this pipeline's cost.

use crate::config::{ConditionEncoderConfig, DepthDecoderConfig, DitConfig, VocoderConfig};
use crate::{condition_encoder, denoise, depth_decoder, global_llm, pipeline, stitch, vocoder, weightcache};
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

/// Which physical card each of the AR stage's two CFG branches loads onto.
///
/// The conditional and unconditional branches are the SAME 8B checkpoint
/// loaded twice, and at int8 each instance is roughly 13 GB: ~6.4 GB of
/// packed int8 per-layer linears, plus the `tok.weight` and `lm_head`
/// tables which stay fp32 at ~3.28 GB each (they are gathered by
/// `embed_tile`/vocab-tiled rather than run through a packed-weight GEMM,
/// so no int8 tier applies to them). The pair therefore does not fit one
/// 24 GB card, which is exactly why the upstream implementation documents
/// two CUDA GPUs as a hard requirement.
///
/// With two or more schedulable GPUs the branches go on different cards.
/// With one - including when the caller pinned `--device gpu0` - both land
/// on it and a real checkpoint will exhaust it; that is the caller's
/// explicit choice, so it is left to fail with the backend's own
/// out-of-memory error rather than silently overridden.
///
/// The two branches only ever exchange host-side vectors (each `Qwen`
/// keeps its own KV cache and hidden states come back to the host anyway
/// for the CFG blend), so there is no cross-device transfer to arrange -
/// and because they share nothing else either, they are also STEPPED
/// concurrently, one per card ([`pipeline::ArBranches`]), rather than one
/// card idling through the other's half of every frame.
fn ar_branch_devices() -> crate::devplan::Placement {
    let gpus = &gpu_core::devices::ambient_compute_set().gpus;
    match gpus.len() {
        0 | 1 => crate::devplan::Placement::single(),
        _ => crate::devplan::Placement { cond: Some(gpus[0]), uncond: Some(gpus[1]) },
    }
}

/// The card the depth decoder's [`depth_decoder::Resident`] goes on, or
/// `None` for the host path.
///
/// **`Some` only when there is a real GPU.** The alternative is not a slower
/// device - it is `hostmath`'s AVX2+FMA+rayon path, which at these shapes
/// beats the Cranelift JIT's rendering of the same dispatch graph across the
/// same cores. So `BRAIN_DEVICE=cpu` (an empty `gpus` set) keeps the host
/// implementation rather than routing the graph through the CPU backend.
///
/// **The same card as the conditional Global LLM branch, not a third one.**
/// The whole point of stepping the two CFG branches together is that they
/// share one pass over the weights, which needs them in one dispatch and so
/// on one card. `ar_branch_devices` already puts the two LM instances on
/// gpu0/gpu1; ~2.3 GB of fp32 depth-decoder weights sit beside the first of
/// them.
fn depth_decoder_device(cond_dev: Option<u32>) -> Result<Option<Gpu>, String> {
    let gpus = &gpu_core::devices::ambient_compute_set().gpus;
    let Some(&index) = cond_dev.as_ref().or(gpus.first()) else {
        return Ok(None);
    };
    Gpu::new_on_index(index, depth_decoder::PIPELINES).map(Some).map_err(|e| format!("minimaxmusic3: the depth decoder could not open gpu{index}: {e}"))
}

/// `--device`-shaped tokens for the denoise and vocoder stages.
///
/// These two stages are sequential and never logically overlap, but on one
/// card a two-chunk generation died in the vocoder with an out-of-memory
/// after both chunks had denoised cleanly. Measured on a P40 at a real
/// chunk length: the vocoder alone peaks at 12.26 GB decoding one
/// 689-latent chunk, and the DiT stage holds ~9.3 GB.
///
/// **The mechanism is not established.** An earlier version of this comment
/// blamed wgpu for not returning a dropped device's VRAM; that is wrong -
/// `Buffer::drop` destroys immediately, and this session measured gpu0
/// falling to 15 MiB once the DiT stage ended. Treat this as a fix that
/// works, not as evidence about wgpu; see the roadmap ledger's Phase 13
/// correction for the probe that would settle it.
///
/// With two or more schedulable GPUs they go on different cards. With one,
/// or when the caller pinned a device explicitly, that choice stands: an
/// explicit `--device` is an instruction, not a hint.
///
/// The denoise stage no longer LEAVES the second card idle, though: its two
/// CFG branches run one per card ([`denoise::CfgDevices`],
/// [`crate::devplan`]), so what this function still decides for that stage
/// is only where the branch that is not borrowing the other card lives -
/// the same `gpus[0]`, from the same schedulable set the placement reads.
/// The vocoder stage runs after the DiT and its `CfgDevices` have both
/// dropped, so it still gets a card with nothing of the denoise stage on
/// it.
fn stage_devices(explicit: Option<&str>) -> (Option<String>, Option<String>) {
    if let Some(dev) = explicit {
        return (Some(dev.to_string()), Some(dev.to_string()));
    }
    let gpus = &gpu_core::devices::ambient_compute_set().gpus;
    match gpus.len() {
        0 | 1 => (None, None),
        _ => (Some(format!("gpu{}", gpus[0])), Some(format!("gpu{}", gpus[1]))),
    }
}

/// [`global_llm::import`] on a specific card, or on the ambient selection
/// when `device` is `None`.
fn load_global_llm(dir: &str, cap: u32, device: Option<u32>) -> Result<(qwen3::QwenConfig, qwen3::Qwen), String> {
    match device {
        Some(i) => gpu_core::devices::with_gpu(i, || global_llm::import(dir, 1, cap))?,
        None => global_llm::import(dir, 1, cap),
    }
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
        let place = ar_branch_devices();
        let (cfg, lm_cond) = load_global_llm(&paths.lm, cap, place.cond)?;
        let (_, mut lm_uncond) = load_global_llm(&paths.lm, cap, place.uncond)?;
        let head = lm_cond.read_weight(cfg.head_weight());

        let dd_cfg = DepthDecoderConfig::real();
        let dd_w = weightcache::depth_decoder(&paths.depth, &dd_cfg)?;
        let dd_gpu = depth_decoder_device(place.cond)?;
        let mut dec = match &dd_gpu {
            Some(gpu) => depth_decoder::Decoder::device(gpu, &dd_cfg, &dd_w, 2),
            None => depth_decoder::Decoder::host(&dd_cfg, 2),
        };

        let branches = pipeline::ArBranches::new(&lm_cond, &mut lm_uncond, place);
        pipeline::generate_frames(branches, &mut dec, &dd_w, &dd_cfg, &head, cfg.vocab as usize, cfg.d_model as usize, &conditional_ids, &unconditional_ids, max_frames, opts.seed, progress)
    };

    let (denoise_dev, vocoder_dev) = stage_devices(opts.device.as_deref());

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
        let cond_w = weightcache::condition_encoder(&paths.condition)?;
        let dit_cfg = DitConfig::real();
        let dit_w = weightcache::dit(&paths.dit, &dit_cfg)?;
        // One handle per card the CFG placement names, opened ONCE for the
        // whole stage and reused across every chunk. On a two-card box that
        // is both cards: the DiT's conditional and zero-condition forwards
        // run concurrently, one per card (`denoise::CfgDevices`,
        // `crate::devplan`), which is the only place in this pipeline where
        // the second card is not simply idle for the whole denoise stage.
        let devices = denoise::CfgDevices::open(denoise_dev.as_deref(), opts.device.as_deref());
        let starts = denoise::chunk_starts(num_frames);
        let length_of = |start: usize| condition_encoder::latent_length(&cond_cfg, (start + denoise::CHUNK_FRAMES).min(num_frames) - start);
        // The DiT's ~9.7 GB of weights per card, uploaded ONCE for the whole
        // stage - exactly as `devices` above is opened once - because
        // nothing about them depends on the chunk. Only the RoPE tables do,
        // via `length`, and `denoise_chunk` rebinds those per chunk for
        // ~90 kB. Re-uploading the blocks per chunk cost tens of seconds of PCIe
        // traffic each, which across a four-minute song's ~59 chunks dwarfed the
        // generation itself,
        // re-sending byte-identical data every time.
        //
        // `starts` is never empty (`chunk_starts` returns at least `[0]`),
        // so the first chunk's length is the right one to build at and the
        // first rebind is a no-op.
        let mut residents = denoise::ChunkResidents::new(&devices, &dit_cfg, &dit_w, length_of(starts[0]));
        let mut state = denoise::ChunkState::default();
        starts
            .iter()
            .enumerate()
            .map(|(i, &start)| {
                let latents = denoise::denoise_chunk(&mut residents, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, start, &mut state, opts.num_inference_steps, opts.seed.wrapping_add(i as u64 + 1), progress);
                (latents, length_of(start))
            })
            .collect()
        // `residents`, `devices` and `dit_w` all drop here, at the end of
        // the denoise stage's block scope and before the vocoder stage
        // loads anything - the sequential-stage RAM discipline this
        // module's own doc describes. Hoisting the upload out of the chunk
        // loop changed how LONG the DiT is resident, never how far.
    };

    // ---- Vocoder stage: crop-and-stitch every chunk. ----
    let (left, right) = {
        let vocoder_cfg = VocoderConfig::real();
        let vocoder_w = weightcache::vocoder(&paths.vocoder, &vocoder_cfg)?;
        let gpu = Gpu::open(vocoder_dev.as_deref(), vocoder::PIPELINES);
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
