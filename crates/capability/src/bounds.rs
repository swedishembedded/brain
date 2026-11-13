// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for model-serving APIs whose
// callers cannot ask for unbounded work for its clients. If your team needs
// expertise in hardening inference servers then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The ceilings a served param of the same meaning shares across models.
//!
//! A number a caller sets sizes work: tokens to generate, pixels to render,
//! frames, denoising or training steps. Each such param declares a ceiling
//! (`ParamSpec::max`), and a param that means what another model's means
//! uses the same one from here, so "how much may a caller ask for" is one
//! decision per meaning, not one per model. A ceiling that only one model's
//! semantics fix (an enum-like code, a level count) stays with that model.
//! `crates/catalog/tests/it/served_bounds.rs` holds every served number to
//! having one.

/// Tokens a call may generate or read: `max_new`, `max_tokens`,
/// `thinking_budget`. The model's own context refuses less than this first.
pub const TOKENS: f64 = 262_144.0;
/// The top-k filter of a sampler (`top_k`, `topk`).
pub const TOP_K: f64 = 1_000.0;
/// One side of an image or a video frame a caller asks to be rendered, or a
/// coordinate in one, in pixels (`width`, `height`, `x`, `y`, `w`, `h`).
pub const SIDE_PX: f64 = 8_192.0;
/// A size in pixels a model resamples its input to, or a tile or margin of it
/// (`size`, `tile`, `input`, outpaint margins).
pub const SIZE_PX: f64 = 4_096.0;
/// Frames of a clip, or a frame index in one.
pub const VIDEO_FRAMES: f64 = 2_048.0;
/// Denoising steps of one generation.
pub const DENOISE_STEPS: f64 = 1_000.0;
/// Optimizer steps or iterations of one training run.
pub const TRAIN_STEPS: f64 = 100_000.0;
/// Other counts of a training run: warmup steps, steps between checkpoints
/// or evaluations, patience, minimum events.
pub const TRAIN_COUNT: f64 = 1_000_000.0;
/// A training row or context length, in tokens (`block`, `seq_len`).
pub const CONTEXT_TOKENS: f64 = 131_072.0;
/// The context window an inference resident is built for (`ctx`): the
/// checkpoint's own declared window and the device's memory refuse less than
/// this first.
pub const CONTEXT_WINDOW: f64 = 16_777_216.0;
/// The rank of a LoRA adapter.
pub const LORA_RANK: f64 = 1_024.0;
/// Frames per second of a clip.
pub const FPS: f64 = 240.0;
/// Seconds of generated audio or video.
pub const SECONDS: f64 = 600.0;
/// An audio sample rate, in Hz.
pub const SAMPLE_RATE: f64 = 384_000.0;
/// A probability, a confidence or a strength on a 0-1 scale.
pub const UNIT: f64 = 1.0;
