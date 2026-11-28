// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a timeline model is built: a parameter of every call that builds one ([`Placement`]), never a guess from
//! the process's environment. A library caller (an embedding application, a test) names the device; only a brain
//! process whose own `--device`/`--backend` flags configured it builds on that configuration
//! ([`Placement::Process`]).
//!
//! Swedish Embedded AB implements long-horizon risk models trained and served on hardware its clients choose. If your
//! team needs expertise in placing model work on shared machines without taking cards another job holds, you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

/// The device a timeline model is built on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    /// The CPU backend: the default, so a caller that names nothing takes no card.
    #[default]
    Cpu,
    /// A `--device`-shaped token: `gpu`, `gpu<i>`, `wgpu`, `cpu` (resolved by [`Gpu::open`]).
    Device(String),
    /// The compute set this brain process was configured with by its own `--device`/`--backend` flags (a served
    /// model, a CLI action). Never for a library caller: a process without those flags has no configuration to use.
    Process,
}

impl Placement {
    /// A handle on this placement for `kernels`.
    ///
    /// # Panics
    /// A [`Placement::Device`] token that does not parse or names hardware this machine does not have: running
    /// somewhere else instead would be a silent substitution. [`Placement::Process`] in a process no flags
    /// configured.
    #[must_use]
    pub fn open(&self, kernels: &[(&str, &str)]) -> Gpu {
        match self {
            Placement::Cpu => Gpu::new_cpu(kernels),
            Placement::Device(token) => Gpu::open(Some(token), kernels),
            Placement::Process => {
                // Only a process its own flags configured has a compute set to build on; anywhere else the
                // fallback would be a guess from the environment, which is exactly what this type replaces.
                if gpu_core::devices::published_compute_set().is_none() {
                    panic!(
                        "a timeline model was asked to build on the process's configured device, but this process \
                         was never configured with one (brain's --device/--backend, or devices::select_compute): \
                         name a Placement instead"
                    );
                }
                Gpu::new(kernels)
            }
        }
    }
}
