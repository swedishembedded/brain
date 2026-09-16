// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resolve a [`crate::Device`] selection against real hardware and apply it
//! process-wide, the same way `crates/cli/src/main.rs` applies `--device`
//! before any model builds. Shared by every surface that builds a
//! device-resident model with no CLI startup path to have done this for it
//! (`pipeline::apply_device` layers `resolve`'s model-placement policy on
//! top, for the one surface that also selects `resolve`; `creature` has no
//! such placement layer and calls this directly).

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
