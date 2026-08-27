// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-agnostic diffusion core for brain's image-generation models.
//!
//! This crate owns the parts of a diffusion model that do **not** depend on the
//! denoiser architecture: the noise/timestep schedule, the sampler that turns a
//! sequence of model outputs into a clean latent, the training-time timestep
//! sampling, the flow-matching (rectified-flow) loss, and classifier-free
//! guidance. Per-model DiT crates (`s3dit`, `flux2`, `hidream`) call into this
//! the same way every model calls the shared `model` trainer — the analogue of
//! how `bench` is model-agnostic.
//!
//! Two scheduler families live here, one per forward process:
//!
//! * [`scheduler`] — **flow matching / rectified flow** ([`FlowMatchEulerScheduler`],
//!   Z-Image and FLUX.2): `x_σ = (1-σ)·x_0 + σ·ε`, the denoiser predicts a
//!   velocity, and Euler integrates it.
//! * [`discrete`] — the **DDPM variance-preserving chain** (SD / SDXL,
//!   `crates/sdxlunet`): `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε`, with
//!   [`DdimScheduler`], [`EulerScheduler`], [`EulerAncestralScheduler`] and
//!   [`DpmSolverPlusPlusScheduler`], each in the ε- and v-prediction
//!   parameterisations.
//! * [`flowsolvers`] - **multistep** solvers in the flow-matching
//!   parameterisation ([`FlowUniPcScheduler`], [`FlowDpmSolverPlusPlusScheduler`],
//!   Wan2.1's `unipc` and `dpm++`): same `α = 1-σ` forward process as
//!   [`scheduler`], but the update reuses the previous steps' model outputs
//!   instead of integrating each step independently.
//!
//! Both are deliberately pure host math (no `gpu_core` dependency) so they are
//! trivially unit-testable and reusable on CPU and GPU paths alike; the
//! device-touching pieces (velocity loss, CFG combine) land alongside as they
//! are wired to the denoiser.

pub mod discrete;
pub mod flowsolvers;
pub mod scheduler;

pub use discrete::{
    BetaSchedule, DdimScheduler, DiscreteConfig, DpmSolverPlusPlusScheduler,
    EulerAncestralScheduler, EulerScheduler, Prediction, Sigmas, SolverType, TimestepSpacing,
};
pub use flowsolvers::{
    BhSolver, FlowDpmSolverConfig, FlowDpmSolverPlusPlusScheduler, FlowUniPcConfig,
    FlowUniPcScheduler,
};
pub use scheduler::{default_z_image_sigmas, flow_shift, FlowMatchConfig, FlowMatchEulerScheduler};
