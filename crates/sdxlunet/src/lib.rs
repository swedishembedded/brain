// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL's `UNet2DConditionModel`: down / mid / up blocks with skip connections,
//! ResBlocks with an **additive** timestep embedding, spatial transformers
//! (self-attention + cross-attention to the text encoding + a GEGLU
//! feed-forward), and SDXL's `text_time` added conditioning.
//!
//! * [`config`] — the reference configuration and the canonical tensor manifest.
//! * [`import`] — diffusers checkpoint import with two-way coverage validation.
//! * [`hostemb`] — SDXL's added-conditioning concat (the only host math local
//!   to this crate; the sinusoid itself is `model::hostmath::timestep_embedding`).
//! * [`init`] — deterministic synthetic weights for the smoke test.
//! * [`model`] — the forward graph.
//!
//! **Adds no kernel and no block.** Everything convolutional is
//! `vae::blocks::Builder`; everything transformer is `model::block` plus the
//! existing attention kernels. That was the measured finding that put the UNet
//! family ahead of a bigger DiT, and it holds.
//!
//! Note the crate description's original wording — "ResBlocks with timestep
//! scale-shift" — was wrong: SDXL ships `resnet_time_scale_shift: "default"`,
//! which is a per-channel ADD. See [`model`] for the five conventions this
//! graph pins and how each was verified.
//!
//! The discrete schedulers this model samples with live in
//! `diffusion::discrete` (DDIM / Euler / Euler-ancestral / DPM-Solver++, ε and
//! v-prediction), not here — they are architecture-independent host math and
//! belong next to `FlowMatchEulerScheduler`. [`sampler`] is the SDXL-specific
//! denoise loop built on top of them (seeding, the CFG pair, the per-step
//! forward seam `controlnet` composes onto); [`textenc`] is the dual CLIP
//! conditioning both `pipeline::Sdxl` and `controlnet::caps::Controlled` share.
//!
//! Backward and `check_unet` are done (`train`). The serving contract (a
//! `capability::Provider`, a residency adapter, D-Bus, an example) is met -
//! see [`caps`] and `crates/cli/src/resident_sdxl.rs`.

pub mod caps;
pub mod config;
pub mod hostemb;
pub mod import;
pub mod init;
pub mod model;
pub mod pipeline;
pub mod sampler;
pub mod textenc;
pub mod train;

pub use config::{BlockKind, UNetConfig};
pub use model::Unet;
