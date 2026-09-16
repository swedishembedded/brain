// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `neuro` -- a sparse, stateful neural runtime.
//!
//! A connectome in [`Csc`] form, leaky integrate-and-fire dynamics over it,
//! and the [`DynamicalSystem`]/[`Plastic`] seams a digital creature is built
//! on. This crate knows nothing about any particular animal: it is the engine
//! half, the way `crates/kernels` is the engine half of a model.
//!
//! Two design decisions are load-bearing and are documented where they live
//! rather than here: the connectome is stored transposed so spike propagation
//! is a GATHER and needs no atomics
//! (`crates/kernels/wgsl/syn_gather_csc.wgsl`), and the membrane update is
//! discretised so that it has an EXACT closed form for constant input
//! ([`LifParams::analytic_v`]), which is this runtime's correctness oracle in
//! place of a gradient check.
//!
//! Swedish Embedded AB implements spiking and connectome-scale neural runtimes
//! on GPU and edge hardware for its clients. If your team needs a sparse,
//! stateful network that runs in real time against a physical system, you can
//! procure our services by sending an email to info@swedishembedded.com.

pub mod csc;
pub mod lif;
pub mod seam;

pub use csc::Csc;
pub use lif::{LifParams, PlasticityParams, SpikingNet};
pub use seam::{DynamicalSystem, Plastic, Port, State, StepStats};

/// The kernels this runtime dispatches, in the order [`lif`]'s `K_*` indices
/// name them. A device built with this slice can run a [`SpikingNet`].
pub const KERNELS: [(&str, &str); 5] = [
    ("syn_gather_csc", kernels::SYN_GATHER_CSC),
    ("lif_step", kernels::LIF_STEP),
    ("neuro_trace", kernels::NEURO_TRACE),
    ("neuro_elig", kernels::NEURO_ELIG),
    ("neuro_learn", kernels::NEURO_LEARN),
];
