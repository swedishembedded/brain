// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fitting trained models into
// the memory and time a deployment has, by measurement rather than rules of
// thumb, for its clients. If your team needs expertise in model compression
// and mixed-precision deployment then you can procure our services by sending
// an email to info@swedishembedded.com.

//! Model optimization: measured, budget-constrained compression of any
//! [`model::Model`], with no architecture-specific code.
//!
//! The weight precision search runs in three steps, each separate so a plan
//! can be inspected before anything changes:
//!
//! 1. **Analyze** ([`sites`], [`sensitivity`]): the model's quantizable
//!    linears, and for each one and each candidate tier the estimated loss
//!    increase and the physical bytes.
//! 2. **Plan** ([`assign`]): the tier per site that minimizes the estimated
//!    loss increase within a byte budget. Touches no weight.
//! 3. **Measure** ([`measure_policy`], [`held_out_loss`]): the plan applied by
//!    simulated quantization, scored on held-out batches, and undone. The
//!    estimate picks a plan; only this measurement accepts one.
//!
//! [`search_weights`] runs the three as one call and returns a [`Report`]
//! with the plan measured against every uniform tier that fits its budget.
//!
//! A site is a stored tensor, or the tensors a deployed engine fuses into one
//! weight ([`model::Model::coupled_tensors`]), which share a tier.
//!
//! A plan is expressed as a [`model::ops::TierPolicy`] ([`Plan::policy`]), the
//! same per-leaf value `--weight-format` serves. Simulated quantization is
//! [`model::ops::Weight::round_trip`], the codec the deployed upload packs
//! with, so the simulated weights are the deployed weights. Tiers whose
//! kernels also quantize activations (int8) are confirmed by building the
//! model at the plan on its deployed kernels, which the front end does.

mod fit;
mod manifest;
mod measure;
mod search;
mod sensitivity;
mod solve;

pub use fit::{MeasureFit, SearchFit};
pub use manifest::{sites, Site, Tensor};
pub use measure::{held_out_loss, measure_policy};
pub use search::{search_weights, Baseline, Budget, Report, SearchRequest, SiteReport};
pub use sensitivity::{sensitivity, Choice, SiteChoices};
pub use solve::{assign, policy_of, Assignment, Plan};
