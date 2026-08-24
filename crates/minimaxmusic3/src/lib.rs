// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax Music 3: lyrics + a structured music-description caption in,
//! a full song (up to 5 minutes, 44.1 kHz stereo) out.
//! (`MiniMaxAI/MiniMax-Music3`, no official upstream inference code - ported
//! from an unmerged `diffusers` PR, commit `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`).
//!
//! Five chained components:
//!
//! ```text
//! lyrics + caption
//!        |
//!    [tokenize: data::qwen_tokenizer::QwenBpe]
//!        v
//! Global LLM (real Qwen3-8B, reused verbatim from `crates/qwen3`)
//!    autoregressive, CFG-guided: one semantic RVQ code per 25 Hz frame
//!        |
//!        +--> RVQ depth decoder (this crate: `depth_decoder`)
//!        |        4-layer causal transformer, autoregressively predicts the
//!        |        7 residual codebooks per frame from the LLM's hidden state
//!        v
//! per-frame hidden states (LLM + 7 depth steps, 8 total)
//!        |
//!    [condition encoder: this crate: `condition_encoder`]
//!        softmax layer-mix -> conv proj -> nearest-resample 25 Hz -> latent rate
//!        v
//! Flow-matching DiT (this crate: `dit`/`block`)
//!    36-layer, partial-RoPE self-attn + gated FFN, denoises Flow-VAE latents
//!    in 200-frame chunks (100-frame hop, 172-latent overlap splice)
//!        v
//! Vocoder (this crate: `vocoder`, DAC-style)
//!    SnakeBeta + weight-normalized conv/conv-transpose -> 44.1 kHz stereo wav
//! ```
//!
//! Reused as-is rather than reimplemented: `crates/qwen3` for the Global LLM
//! (a real Qwen3-8B architecture, `hidden=4096, layers=36, heads=32,
//! kv_heads=8, head_dim=128, vocab=200000` - confirmed against the real
//! checkpoint's own `language_model/config.json`, NOT the smaller published
//! Qwen3-8B's `vocab=151936` preset - see `global_llm::import`'s own doc
//! for why `language_model/` specifically, not the checkpoint's other,
//! same-shaped but architecturally different `qwen_7B/qwen_7B/` directory),
//! `data::qwen_tokenizer::QwenBpe` for tokenization,
//! `audio::conv` (forward AND backward) for every conv/conv-transpose in the
//! vocoder, `model::block`'s `Bidir`/`rope2d_partial`/`LayerNorm`/`swiglu`/
//! `kv_expand` Step-builders for the DiT's bidirectional partial-RoPE
//! transformer blocks, and `diffusion::FlowMatchEulerScheduler` (with its new
//! `invert_sigmas` option - the reference scheduler is constructed with
//! `invert_sigmas=True, num_train_timesteps=1, shift=1.0`) for the DiT's
//! sampling loop.
//!
//! Status: condition encoder, vocoder, the RVQ depth decoder, and the DiT's
//! forward are all landed with real-weight parity cosine 1.0. The vocoder,
//! the depth decoder, AND the DiT (`dit_train::Trainer`, device-dispatched
//! through the same `model::block` builders `dit::forward` uses) are fully
//! trainable - full fine-tune (gradcheck on every named parameter, a real
//! overfit demonstration) and LoRA (exact no-op at zero-init, a directional
//! FD check on every adapter, and a training demonstration with the base
//! weights provably frozen) for all three, including the DiT
//! (`dit_lora`, fold-then-run through a fresh `dit_train::Trainer` per
//! step). The vocoder also has a from-scratch STFT-magnitude adversarial
//! discriminator (gradchecked end to end, including through the STFT, and
//! shown to learn). The DiT also has an INT8 STORAGE tier (`dit_int8`,
//! smaller checkpoint only - no compute-path change) and `model::Shardable`
//! pipeline-parallel sharding (`dit_shard::DitStage` - inference-time
//! layer-range splitting across devices, no backward through the pipeline;
//! `dit_train::Trainer` remains this crate's real, single-device DiT
//! training path). The Global LLM (`global_llm`) has streamed real-weight
//! import (real-layer parity cosine 1.0 against `transformers`) and an
//! audio-code-restricted training objective (`Batch::LmWeighted`, proven
//! at `QwenConfig::tiny()` scale). `pipeline::generate_frames` wires it
//! together with the depth decoder into the full CFG-guided AR
//! generation loop (two `qwen3::Qwen` instances, one per CFG branch,
//! stepped in lockstep), and `denoise::denoise_chunk` turns that AR
//! output into Flow-VAE latents chunk by chunk (the 200-frame/100-hop
//! window, the DiT's own zero-condition CFG, and the 172-latent overlap
//! splice carried between chunks via `denoise::ChunkState`).
//! `stitch::Stitcher` decodes each chunk's latents through the vocoder
//! and crops/concatenates them into one continuous stereo waveform
//! (`audio::wav::write_multi` for the delivered file), and
//! `global_llm::assemble_prompt` builds the AR stage's own token-id
//! prompts from raw caption/lyrics text. `generate::generate` is the ONE
//! implementation of the full pipeline (prompt assembly through
//! multi-chunk denoise and stitch), shared by `tests/
//! e2e_short_generation.rs`, `caps::MinimaxMusic3Provider` (the direct/
//! `brain do` path), and `crates/cli::resident_minimaxmusic3` (the
//! D-Bus/scheduler path) - `brain minimaxmusic3 generate` and the
//! `brain/minimaxmusic3` D-Bus model are both live and reachable. What
//! **could not be validated on this development machine** is a real run
//! all the way through: whole-8B-model residency fails on both of this
//! machine's backends (CPU-JIT's `int8` request silently promotes to
//! fp32; this machine's own Vulkan iGPU caps single buffers below the
//! embedding/head tensors' size), both pre-existing `qwen3`/`gpu_core`/
//! `backend-cpu` limits, not a defect here - see this crate's own
//! roadmap ledger for the measured diagnosis.

pub mod caps;
pub mod condition_encoder;
pub mod config;
pub mod denoise;
pub mod depth_decoder;
pub mod depth_lora;
pub mod discriminator;
pub mod dit;
pub mod dit_int8;
pub mod dit_lora;
pub mod dit_shard;
pub mod dit_train;
pub mod generate;
pub mod global_llm;
pub mod lora;
pub mod pipeline;
pub mod stitch;
pub mod train;
pub mod vocoder;

pub use config::{ConditionEncoderConfig, DepthDecoderConfig, DitConfig, VocoderConfig};

/// Where a long generation reports how far along it is: `(done, total,
/// stage)`, with `stage` one of `"ar"`, `"denoise"` or `"vocode"`.
///
/// Deliberately NOT `capability::Progress`. `pipeline`/`denoise`/`stitch`
/// are library math and have no business depending on the serving
/// surface's types; `caps::generate_action` is the one place that adapts
/// this into a `Progress`. It also means an in-process caller (a bench, a
/// test) can observe stage progress without constructing a serving
/// invocation.
///
/// This exists because `generate` is a multi-MINUTE call. The `generate`
/// action has always declared `.streaming()` - and `caps`'s own test says
/// why, that "a multi-minute generation with no progress reporting is a
/// hang, not a feature" - but every layer bound the callback as `_progress`
/// and dropped it, so the declaration was a promise nothing kept. It is
/// also what `brain perf` derives its entire `submit -> first -> done`
/// timeline from, so without it this model cannot be benchmarked at all.
pub type ProgressSink<'a> = &'a mut dyn FnMut(u32, u32, &str);

/// A [`ProgressSink`] that discards everything - for callers that do not
/// care (tests, one-shot library use).
pub fn ignore_progress() -> impl FnMut(u32, u32, &str) {
    |_, _, _| {}
}
