// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3: a 33B joint video+audio rectified-flow diffusion transformer.
//!
//! Unlike [`ltxv`](../ltxv/index.html) (two streams coupled by bidirectional
//! A/V cross-attention), H3 is **one packed self-attention stack**: video,
//! audio and condition rows share every block, and modality differences come
//! from input/output projections, modality-tagged AdaLN and positional
//! coordinates - not modality-specific transformer blocks. Confirmed
//! structurally from the real checkpoint's own tensor names: no A/V
//! cross-attention weights anywhere in the shard.
//!
//! A single 50-block transformer produces both a video and an audio
//! prediction per forward, denoised under **two independent shifted-sigma
//! Euler schedules** (`shift=12` video, `shift=3` audio) - never two
//! separate `denoise()` loops. Text conditioning comes from a Qwen3-VL text
//! encoder truncated to decoder layer 50 (confirmed from the real
//! `diffusers` reference), refined by 2 token-refiner blocks (plain
//! pre-norm, no AdaLN, no RoPE) before packing. The model is
//! guidance-distilled: no CFG pass, no `guidance_scale`.
//!
//! ## Reference material
//!
//! Unlike every prior port in this workspace, the math authority is **not**
//! MiniMax's own repository (community-licensed): `model_index.json` sources
//! every component from Apache-2.0 `diffusers`/`transformers`
//! (`MiniMaxH3DiTModel`, `MiniMaxH3VideoVAE`, `MiniMaxH3AudioVAE`,
//! `MiniMaxH3Qwen3VLHFEncoder`). The audio VAE additionally ships its own
//! Apache-2.0/MIT Python source directly inside the checkpoint. This crate's
//! implementation is ported from those Apache-2.0 sources, never translated
//! from MiniMax's own community-licensed repository - see [`caps::check_license`]
//! for what that means for the weights themselves.
//!
//! ## Hardware fit
//!
//! `adaln_proj` is 260M params/block x 50 blocks = 13.0B of the 33B total
//! (confirmed from the real shard-0 safetensors header). Folding it into
//! small precomputed per-(step,modality) modulation tables
//! ([`precompute_adaln`]) turns the DiT into a ~20.2B backbone (measured at
//! the tiny-config scale and extrapolated to the real config's dimensions -
//! see that module) that fits resident at int8 across two 24GB cards.
//!
//! ## Status
//!
//! This crate is a fresh port in progress. Every module through the DiT
//! core, both VAEs, the vocoder and the dual schedule has a real numeric
//! parity gate against the actual installed `diffusers==0.40.0` reference at
//! tiny config, green; the audio VAE additionally has real-checkpoint
//! numeric parity. `t2va`/`fl2va` run end to end at tiny config, weight-free.
//! `ref2va` is not implemented. Real-checkpoint import for the DiT itself
//! (splitting the checkpoint's fused `qkv_proj`) and a real Qwen3-VL text
//! encoder wired into serving (a deterministic stub stands in today) are
//! both open - an entry existing in this module does not by itself mean the
//! capability it names is real-weight-complete.

pub mod block;
pub mod caps;
pub mod config;
pub mod import;
pub mod model;
pub mod precompute_adaln;
pub mod rope;
pub mod schedule;
pub mod vocoder;
