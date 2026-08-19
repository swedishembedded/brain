// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **ControlNet as a backbone-agnostic seam**, and the SDXL implementation of
//! it.
//!
//! A ControlNet is a trainable copy of a diffusion backbone's early blocks
//! whose zero-conv outputs are added as residuals at *named injection points*.
//! That structure is not UNet-specific — a FLUX ControlNet injects into
//! double-stream blocks the same way — so what this crate exports first is the
//! contract ([`adapter`]) and only then a model that satisfies it ([`model`]).
//! A crate that only worked for SDXL would be the wrong deliverable even at
//! perfect parity.
//!
//! * [`adapter`] — [`InjectionPoint`], [`Residuals`], the [`ControlAdapter`]
//!   (backbone) / [`ControlSource`] (control model) pair, and the by-name
//!   matching that is the whole seam. `sdxlunet::Unet` is the first
//!   [`ControlAdapter`]; the FLUX DiTs are the intended second.
//! * [`config`] — [`ControlNetConfig`], which **holds** a `UNetConfig` rather
//!   than restating it, plus the canonical tensor manifest.
//! * [`import`] — diffusers `ControlNetModel` import, two-way covered, reusing
//!   `sdxlunet::import::remap_manifest`.
//! * [`cond`] — preparing the conditioning image (including from a ZipDepth
//!   map).
//! * [`init`] — deterministic synthetic weights for the smoke test.
//! * [`model`] — the forward graph.
//!
//! **Adds no block, and one kernel *slot* rather than one kernel.** The
//! trainable copy is `sdxlunet::model::Rec` verbatim; the conditioning embedder and
//! the zero-convs are `vae::blocks::Builder` convolutions; `conditioning_scale`
//! is the existing `scale_chan` with `c = 1`. [`model::KERNELS`] is
//! `sdxlunet::model::KERNELS` plus that one entry, and being a strict
//! prefix-extension is what lets ONE device drive a UNet and a ControlNet
//! together.
//!
//! **Forward only.** `check_controlnet`, LoRA/finetuning, INT8, batch > 1, the
//! serving contract (a `capability::Provider`, a residency adapter,
//! `run_batch`, D-Bus, an example) and a sampling CLI are all deferred.

pub mod adapter;
pub mod caps;
pub mod cond;
pub mod config;
pub mod import;
pub mod init;
pub mod model;

pub use adapter::{
    check_compatible, order_for, ControlAdapter, ControlSource, InjectionPoint, Layout, Residuals,
};
pub use config::ControlNetConfig;
pub use model::ControlNet;
