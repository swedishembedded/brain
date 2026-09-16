// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resolve a [`crate::Device`] selection against real hardware and apply it
//! process-wide, the same way `crates/cli/src/main.rs` applies `--device`
//! before any model builds. Shared by every surface that builds a
//! device-resident model with no CLI startup path to have done this for it
//! (`apply` layers `resolve`'s model-placement policy on top, for every
//! surface that also selects `resolve`; `creature` has no such placement
//! layer and calls `resolve` directly).

use crate::{Device, Error, Result};

/// Probe the real hardware, resolve `device` against it, apply the result as
/// this process's ambient [`gpu_core::ComputeSet`], and hand the resolved
/// set back so a caller that needs it (placement, diagnostics) doesn't have
/// to resolve a second time.
pub(crate) fn resolve(device: &Device) -> Result<gpu_core::devices::ComputeSet> {
    let probe = gpu_core::Inventory::probe();
    let set = device.resolve(&probe).map_err(Error::Backend)?;
    set.apply().map_err(Error::Backend)?;
    gpu_core::publish_compute_set(set.clone());
    Ok(set)
}

/// [`resolve`] plus installing [`loader::install_default_placer`], so
/// automatic GPU/CPU model-shard placement narrows to exactly what was
/// asked for. Every surface that selects `resolve` (real, potentially
/// multi-GB, potentially multi-GPU checkpoints) needs this; `creature`
/// (a single connectome's worth of GPU memory, no shard placement) does not
/// and calls [`resolve`] alone.
#[cfg(feature = "resolve")]
pub(crate) fn apply(device: &Device) -> Result<()> {
    let set = resolve(device)?;
    let gpus: Option<std::collections::HashSet<u32>> = Some(set.gpus.iter().copied().collect());
    let cpu_allowed = set.cpu_enabled();
    loader::install_default_placer(gpus, cpu_allowed);
    Ok(())
}
