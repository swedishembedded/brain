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
    let mut set = device.resolve(&probe).map_err(Error::Backend)?;
    gpu_core::devices::apply_requested_backend(&mut set).map_err(Error::Backend)?;
    // Not `ComputeSet::apply`: narrowing the CPU pins the whole process to the
    // chosen cores and sizes the global pool, which is for a program's start-up
    // (the CLI, `appopts`), not for a library building a pipeline among other
    // threads. The backend and the card pin are applied; CPU cores stay the
    // process's.
    set.apply_backend().map_err(Error::Backend)?;
    gpu_core::publish_compute_set(set.clone());
    Ok(set)
}

/// [`resolve`] plus installing [`loader::install_default_placer`], so
/// automatic GPU/CPU model-shard placement narrows to exactly what was
/// asked for. Every surface that selects `resolve` (real, potentially
/// multi-GB, potentially multi-GPU checkpoints) needs this; `creature`
/// (a single connectome's worth of GPU memory, no shard placement) does not
/// and calls [`resolve`] alone.
/// Returns the resolved set, so a pipeline that can run on an NPU sees
/// whether one was selected.
#[cfg(feature = "resolve")]
pub(crate) fn apply(device: &Device) -> Result<gpu_core::devices::ComputeSet> {
    let set = resolve(device)?;
    let gpus: Option<std::collections::HashSet<u32>> = Some(set.gpus.iter().copied().collect());
    let cpu_allowed = set.cpu_enabled();
    loader::install_default_placer(gpus, cpu_allowed);
    Ok(set)
}

/// Drive the hardware with `backend` (`wgpu | vulkan | cuda | cpu`) in every
/// pipeline this process builds from now on: the `--backend` flag, for a
/// program that is not the brain CLI. Call it before building a pipeline.
///
/// # Errors
/// `backend` is not a backend name.
pub fn select_backend(backend: &str) -> Result<()> {
    gpu_core::devices::request_backend(backend).map_err(Error::Backend)
}

/// One GPU's memory right now, as the driver reports it: what a caller needs
/// to say, before it loads a model, whether the model will fit on a card or be
/// placed on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMemory {
    /// The card's index, the `i` of `--device gpu<i>`.
    pub index: u32,
    /// The card's size in bytes.
    pub total_bytes: u64,
    /// Bytes free right now, other processes' allocations included.
    pub free_bytes: u64,
    /// False when the free figure could not be read from the driver and was
    /// assumed, so a caller does not present a guess as a measurement.
    pub measured: bool,
}

impl From<gpu_core::capacity::GpuMem> for GpuMemory {
    fn from(card: gpu_core::capacity::GpuMem) -> Self {
        Self { index: card.index, total_bytes: card.total, free_bytes: card.free, measured: card.measured }
    }
}

/// The memory of every GPU in this machine, empty when it has none. Reads the
/// driver; loads no model and allocates nothing on a card.
pub fn gpu_memory() -> Vec<GpuMemory> {
    gpu_core::capacity::probe_gpus().into_iter().map(GpuMemory::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cards_memory_keeps_its_index_sizes_and_whether_it_was_measured() {
        let card = gpu_core::capacity::GpuMem { index: 1, total: 24 << 30, free: 3 << 30, measured: true };
        assert_eq!(GpuMemory::from(card), GpuMemory { index: 1, total_bytes: 24 << 30, free_bytes: 3 << 30, measured: true });
        let guessed = gpu_core::capacity::GpuMem { measured: false, ..card };
        assert!(!GpuMemory::from(guessed).measured);
    }
}
