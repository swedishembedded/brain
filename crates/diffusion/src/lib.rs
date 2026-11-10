// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-agnostic diffusion core for brain's image-generation models.
//!
//! This crate owns the parts of a diffusion model that do **not** depend on the
//! denoiser architecture: the noise/timestep schedule and the samplers that
//! turn a sequence of model outputs into a clean latent. Per-model crates
//! (`s3dit`, `flux1`, `flux2`, `sdxlunet` and others) call into it; the training
//! objectives (the flow-matching loss and its timestep sampling) and the
//! classifier-free guidance combine live in those crates, next to the
//! denoiser they drive.
//!
//! The schedulers, by forward process:
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
//! [`restore`] holds SUPIR's restoration sampler math on the same discrete
//! chain.
//!
//! All of it is deliberately pure host math (no `gpu_core` dependency) so it is
//! trivially unit-testable and reusable on CPU and GPU paths alike.

pub mod discrete;
pub mod flowsolvers;
pub mod restore;
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
