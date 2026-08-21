// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! brain's automatic model **residency + scheduling** layer.
//!
//! It lets `brain serve` (and every other execution path — the CLI, the event
//! runtime, the D-Bus surface) expose many models at once without the caller
//! managing memory: hot weights live on the GPU, recently-used ones spill to RAM,
//! cold ones stay memory-mapped on disk, and jobs are scheduled to reuse hot paths.
//!
//! This crate is split into small, testable pieces:
//! - [`budget`] — per-device memory budgets with a reserved headroom.
//! - [`lru`] — the resident-instance table (last-use tracking for eviction).
//! - [`place`] — placement (which device) + eviction (which LRU victims).
//! - (later phases) tiered weight store, the `ResidentModel` trait, the `Executor`.
//!
//! P0 (this file + the three above) is pure CPU logic — the memory model — with no
//! GPU or model dependency, so it is fully unit-testable.

pub mod admission;
pub mod bridge;
pub mod budget;
pub mod devpool;
pub mod executor;
pub mod jobs;
pub mod log;
pub mod lru;
pub mod manager;
pub mod model;
pub mod multi;
pub mod place;
pub mod scheduler;
pub mod supply;

pub use devpool::DevicePool;
pub use executor::{Executor, InFlightJob, Job};
pub use manager::{DeviceBudget, InstancePlacement, ResidencyManager, ResidencyReport};
pub use model::{Instance, ResidentModel};
pub use multi::{MultiDeviceCost, MultiDeviceResidentModel};
pub use scheduler::Policy;
pub use supply::{ModelSupplier, Supply};

/// A device that can hold a hot model instance. Canonical definition lives in
/// `memauth` (the process-wide memory authority both this crate and
/// `weightset`'s within-instance weight window depend on) — re-exported here
/// so every existing `residency::Device` path keeps compiling unchanged.
/// `Gpu(u32)` is the physical card `gpu<i>` names in the device registry
/// (`gpu_core::devices`, PCI-bus order); `Cpu` is RAM-resident/CPU-executed;
/// `Npu(u32)` is a whole-graph (OpenVINO) device.
pub use memauth::Device;

/// The memory footprint of a model instance when it is **Hot**.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MemCost {
    /// GPU bytes held while hot on a GPU (weights + resident activation scratch).
    pub vram: u64,
    /// Host bytes held (warm weights, staging, or a CPU-resident model).
    pub ram: u64,
    /// NPU device bytes held while hot on the NPU (the compiled OpenVINO graph +
    /// weights). Non-zero **iff** the model has an NPU path — this is precisely how a
    /// model advertises NPU-eligibility, so a CPU-only model (npu == 0) is never
    /// placed on the NPU even when one is budgeted. `MemCost::new` keeps `npu = 0`.
    pub npu: u64,
    /// Reclaimable, page-cache-backed bytes (a `Tier::Cold`/mmap footprint) —
    /// separate from `ram` because the kernel can evict these under memory
    /// pressure the way it never evicts a live allocation, so a governor
    /// must not treat them as equally "in use". Zero for every model that
    /// doesn't override [`Instance::demote`] to a mapped tier.
    pub mapped: u64,
}

impl MemCost {
    pub fn new(vram: u64, ram: u64) -> MemCost {
        MemCost { vram, ram, npu: 0, mapped: 0 }
    }
    /// Add an NPU footprint (marks the instance as NPU-placeable). The NPU's compiled
    /// blob + I/O live in shared host memory, so this is a host-memory figure — but
    /// kept a separate field from `ram` so NPU-eligibility is explicit (a CPU-only
    /// model reports `ram > 0, npu == 0` and is never placed on the NPU).
    pub fn with_npu(mut self, npu: u64) -> MemCost {
        self.npu = npu;
        self
    }
    /// Record a `Tier::Cold` mapped-but-not-loaded footprint alongside this
    /// cost (see [`Self::mapped`]'s doc).
    pub fn with_mapped(mut self, mapped: u64) -> MemCost {
        self.mapped = mapped;
        self
    }
    /// The bytes this instance occupies on `device` (VRAM on a GPU, RAM on the CPU,
    /// NPU bytes on an NPU).
    pub fn on(&self, device: Device) -> u64 {
        match device {
            Device::Gpu(_) => self.vram,
            Device::Cpu => self.ram,
            Device::Npu(_) => self.npu,
        }
    }
}

/// Residency tier of a model's weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    /// Memory-mapped on disk (not loaded).
    Cold,
    /// Converted host buffers in RAM (loaded, not on a GPU).
    Warm,
    /// Built on a GPU, ready to run.
    Hot,
}

/// Identity of a resident model instance: a model plus the configuration fingerprint
/// that fixes its build (size / precision / adapter — the model's "hot key"). Two
/// jobs with the same `InstanceKey` can share one resident instance (and batch).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InstanceKey {
    pub model: String,
    pub config: String,
}

impl InstanceKey {
    pub fn new(model: impl Into<String>, config: impl Into<String>) -> InstanceKey {
        InstanceKey { model: model.into(), config: config.into() }
    }
}

impl std::fmt::Display for InstanceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]", self.model, self.config)
    }
}
