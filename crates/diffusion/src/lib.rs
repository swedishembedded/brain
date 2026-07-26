// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-agnostic diffusion core for brain's image-generation models.
//!
//! This crate owns the parts of a diffusion model that do **not** depend on the
//! denoiser architecture: the noise/timestep schedule, the sampler that turns a
//! sequence of model outputs into a clean latent, the training-time timestep
//! sampling, the flow-matching (rectified-flow) loss, and classifier-free
//! guidance. Per-model DiT crates (`zimage`, `flux2`, `hidream`) call into this
//! the same way every model calls the shared `model` trainer — the analogue of
//! how `bench` is model-agnostic.
//!
//! Today it provides the [`FlowMatchEulerScheduler`] used by Z-Image and FLUX.2
//! (rectified flow, `x_{t} = (1-σ)·x_0 + σ·ε`, Euler integration of the learned
//! velocity). It is deliberately pure host math (no `gpu_core` dependency) so it
//! is trivially unit-testable and reusable on CPU and GPU paths alike; the
//! device-touching pieces (velocity loss, CFG combine) land alongside as they
//! are wired to the DiT.

pub mod scheduler;

pub use scheduler::{default_z_image_sigmas, FlowMatchConfig, FlowMatchEulerScheduler};
