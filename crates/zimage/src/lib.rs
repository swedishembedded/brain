// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image (Tongyi) text-to-image — the S³-DiT single-stream diffusion
//! transformer.
//!
//! Forward-first: latent-diffusion **inference** is the immediate goal (the DiT
//! runs a few flow-matching steps over Qwen3-4B caption features + a noisy
//! latent, then the VAE decodes). Training (hand-written backward + gradcheck +
//! `Shardable` + LoRA) builds on the same forward graph.
//!
//! Assembled from the shared crates — `dit` (multi-axis RoPE), `diffusion`
//! (scheduler), `vae` (decode), and the `qwen` encoder — plus direct dispatch of
//! brain's kernels (the same pattern as `crates/vae`). The one structural trick:
//! Z-Image's global adaLN modulation (per-channel scale/gate from the timestep
//! embedding) is **folded into the RMSNorm weights on the host** each forward —
//! `rmsnorm(x,w)·scale = rmsnorm(x, w·scale)` and `gate·rmsnorm(y,w) =
//! rmsnorm(y, w·gate)` — so no scale/gate kernels are needed.

pub mod block;
pub mod dev;
pub mod import;
pub mod int8;
pub mod model;

pub use dev::{ZImageDit, ZImageDitShard};

pub use block::{BlockDims, Tensors, ZImageBlock};
pub use model::{ZImageConfig, ZImageModel};
