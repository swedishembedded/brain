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

pub mod budget;
pub mod lru;
pub mod model;
pub mod place;

pub use model::{Instance, ResidentModel};

/// A device that can hold a hot model instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Device {
    /// A GPU by index (the physical card `BRAIN_GPU_INDEX` selects).
    Gpu(u32),
    /// RAM-resident, CPU-executed (e.g. the CPU encoder path). Bounded by the RAM budget.
    Cpu,
}

/// The memory footprint of a model instance when it is **Hot**.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MemCost {
    /// GPU bytes held while hot on a GPU (weights + resident activation scratch).
    pub vram: u64,
    /// Host bytes held (warm weights, staging, or a CPU-resident model).
    pub ram: u64,
}

impl MemCost {
    pub fn new(vram: u64, ram: u64) -> MemCost {
        MemCost { vram, ram }
    }
    /// The bytes this instance occupies on `device` (VRAM on a GPU, RAM on the CPU).
    pub fn on(&self, device: Device) -> u64 {
        match device {
            Device::Gpu(_) => self.vram,
            Device::Cpu => self.ram,
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
