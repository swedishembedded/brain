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
//! family into `docs/imaging/plan.md` §2 ahead of a bigger DiT, and it holds.
//!
//! Note the crate description's original wording — "ResBlocks with timestep
//! scale-shift" — was wrong: SDXL ships `resnet_time_scale_shift: "default"`,
//! which is a per-channel ADD. See [`model`] for the five conventions this
//! graph pins and how each was verified.
//!
//! The discrete schedulers this model samples with live in
//! `diffusion::discrete` (DDIM / Euler / Euler-ancestral / DPM-Solver++, ε and
//! v-prediction), not here — they are architecture-independent host math and
//! belong next to `FlowMatchEulerScheduler`.
//!
//! **Forward only.** Backward/`check_unet`, the serving contract (a
//! `capability::Provider`, a residency adapter, `run_batch`, D-Bus, an
//! example), the VAE / text-encoder glue and a sampling CLI are all deferred.

pub mod config;
pub mod hostemb;
pub mod import;
pub mod init;
pub mod model;

pub use config::{BlockKind, UNetConfig};
pub use model::Unet;
