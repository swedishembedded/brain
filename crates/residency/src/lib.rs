// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! brain's automatic model **residency + scheduling** layer.
//!
//! It lets `brain serve` (and every other execution path — the CLI, the event
//! runtime, the D-Bus surface) expose many models at once without the caller
//! managing memory: hot weights live on the GPU, recently-used ones spill to RAM,
//! cold ones stay memory-mapped on disk, and jobs are scheduled to reuse hot paths.
//!
//! Swedish Embedded AB implements memory-budgeted, multi-accelerator serving
//! for teams whose models do not all fit resident at once - which is most
//! teams, once more than one model is in production. If your team needs
//! expertise in model residency, eviction policy and scheduling work across
//! several devices without overcommitting any of them, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! This crate is split into small, testable pieces:
//! - [`budget`] — per-device memory budgets with a reserved headroom.
//! - [`lru`] — the resident-instance table (last-use tracking for eviction).
//! - [`place`] — placement (which device) + eviction (which LRU victims).
//! - (later phases) tiered weight store, the `ResidentModel` trait, the `Executor`.
//!
//! P0 (this file + the three above) is pure CPU logic — the memory model — with no
//! GPU or model dependency, so it is fully unit-testable.

pub mod activate;
pub mod adapter;
pub mod affinity;
pub mod admission;
pub mod bridge;
pub mod budget;
pub mod contract;
pub mod devices;
pub mod devpool;
pub mod executor;
pub mod jobs;
pub mod log;
pub mod lru;
pub mod manager;
pub mod model;
pub mod multi;
pub mod observe;
pub mod place;
pub mod plan;
pub mod runplan;
pub mod scheduler;
pub mod staged;
pub mod supply;
pub mod supervised;


pub use activate::on_device;
/// Where a device build inside [`on_device`] lands (a caller that must not run on the CPU by accident checks it).
pub use gpu_core::{build_target, BuildTarget};
pub use adapter::{AdapterIdentity, PolicyLineage};
pub use devpool::DevicePool;
pub use executor::{Executor, InFlightJob, Job};
pub use manager::{BuildRecord, DeviceBudget, EvictionRecord, InstanceLife, InstancePlacement, LifeState, ResidencyManager, ResidencyReport};
pub use model::{Alias, Instance, JobFeed, OfArchitecture, ResidentModel, ADMITTED_WIDTH, METRICS_PUBLISH_INTERVAL};
pub use multi::{MultiDeviceCost, MultiDeviceResidentModel};
pub use runplan::{Eviction, PlanError, Placement, RunPlan};
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
pub use devices::DeviceInventory;

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
    /// Device bytes the instance allocates beyond `vram` while it runs its
    /// widest step (the forward buffers of a decode batch or a prefill chunk,
    /// attention scratch, a recorded step's launch workspace), as the model
    /// estimates them from its own shapes ([`Self::with_activation`]). `None`
    /// means the model declares no estimate: the placement then charges the
    /// operator's measured fallback in its place
    /// ([`crate::budget::Budgets::activation_fallback`]).
    pub activation: Option<u64>,
    /// Host bytes a process of the instance's own holds (the worker that hosts
    /// it: its binary's mappings, its runtime, its own copy of what the model
    /// keeps on the host), [`Self::with_process_host`]. None of it is shared
    /// with the serving process, so it is charged to the host budget beside
    /// whatever the instance is charged wherever it is placed. `0` for an
    /// instance that runs in the serving process.
    pub process_host: u64,
    /// Device bytes, counted in [`Self::vram`] while Hot, of a device context of the instance's own that stays on
    /// its card when the instance is demoted: the worker process that hosts it holds that context for as long as it
    /// lives ([`Self::with_kept_context`]). Its demotion then frees `vram` less these, and a Warm or Cold instance
    /// stays charged them on its card. Zero for an instance whose demotion frees its card.
    pub context: u64,
    /// The instance is host-side work ([`Self::host_work`]): bookkeeping a
    /// serving process does in its own memory (checking a dataset, counting
    /// tokens, folding files), not a model computing on the CPU. It is placed on
    /// the host whatever accelerators `--device` names, because the host is
    /// where it runs and no accelerator could take it instead.
    pub host_side: bool,
}

impl MemCost {
    pub fn new(vram: u64, ram: u64) -> MemCost {
        MemCost { vram, ram, npu: 0, mapped: 0, activation: None, process_host: 0, context: 0, host_side: false }
    }
    /// `ram` bytes of host-side work (see [`Self::host_side`]): placed on the
    /// host's memory even when the CPU is excluded from model compute
    /// ([`crate::budget::Budgets::exclude_cpu_compute`]), never refused because
    /// only accelerators were named.
    pub fn host_work(ram: u64) -> MemCost {
        MemCost { host_side: true, ..MemCost::new(0, ram) }
    }
    /// Whether the instance may go to the host's memory when the CPU computes no
    /// model: host-side work, or an instance that holds no memory anywhere (host
    /// glue, which every budget can take).
    pub fn runs_on_host_regardless(&self) -> bool {
        self.host_side || (self.vram == 0 && self.ram == 0 && self.npu == 0)
    }
    /// Declare the instance's peak activation bytes, beyond [`Self::vram`]
    /// (see [`Self::activation`]). `0` declares that `vram` already covers
    /// them, so no fallback is charged.
    pub fn with_activation(mut self, bytes: u64) -> MemCost {
        self.activation = Some(bytes);
        self
    }
    /// The cost a placement charges: `vram` plus the declared activation, or
    /// plus `fallback` for a model that declares none and holds device bytes
    /// (a model with no device bytes runs no device batch). The result
    /// declares what it added, so placing it again adds nothing.
    pub fn placed(&self, fallback: u64) -> MemCost {
        let activation = self.activation.unwrap_or(if self.vram > 0 { fallback } else { 0 });
        MemCost { vram: self.vram + activation, activation: Some(activation), ..*self }
    }
    /// Add `bytes` of device memory a device context of the instance's own holds beside it (the worker process that
    /// hosts it), charged with [`Self::vram`] so [`Self::placed`] and every budget see it. Nothing is added for an
    /// instance that holds no device memory, whose worker opens no device.
    pub fn with_device_context(mut self, bytes: u64) -> MemCost {
        if self.vram > 0 || self.activation.is_some_and(|a| a > 0) {
            self.vram += bytes;
        }
        self
    }
    /// Add `bytes` of host memory a process of the instance's own holds beside it (see [`Self::process_host`]).
    pub fn with_process_host(mut self, bytes: u64) -> MemCost {
        self.process_host += bytes;
        self
    }
    /// Declare `bytes` of [`Self::vram`] that a demotion keeps on the instance's card (see [`Self::context`]).
    pub fn with_kept_context(mut self, bytes: u64) -> MemCost {
        self.context = bytes;
        self
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

    /// The bytes an instance **resident on** `device` charges that device.
    ///
    /// Identical to [`Self::on`] for an accelerator, and for a model that only
    /// ever declared host bytes. It differs for the one case [`Self::on`] gets
    /// wrong: a weight-holding model reports its footprint as `vram` (that is
    /// what `est_vram` builds), and when such a model is placed on the HOST
    /// tier - a GPU-less box, or a card too contended to hold it - those
    /// weights are held in RAM. `on(Cpu)` reports its `ram` field, which for
    /// such a model is `0`, so the host tier was charged NOTHING for a model
    /// physically occupying tens of GiB of it and would accept an unbounded
    /// number of them.
    ///
    /// It is a `max` rather than a sum for the reason
    /// [`crate::place::host_need`] documents: a model does not stage a second
    /// copy of itself beside itself.
    pub fn resident_on(&self, device: Device) -> u64 {
        match device {
            Device::Cpu => self.ram.max(self.vram).max(self.npu),
            d => self.on(d),
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

    /// The marker a replica's config carries; replica 0 is the key itself.
    const REPLICA: &'static str = "#replica=";

    /// Replica `i` of the build this key names: the same model and config on its own instance, which the
    /// manager places and runs separately (see [`ResidentModel::replicas`]). Replica 0 is `self`.
    pub fn replica(&self, i: usize) -> InstanceKey {
        let base = self.base();
        if i == 0 {
            return base;
        }
        InstanceKey { model: base.model, config: format!("{}{}{i}", base.config, Self::REPLICA) }
    }

    /// Which replica this key names (0 for a key that is not a replica's).
    pub fn replica_index(&self) -> usize {
        self.config.rsplit_once(Self::REPLICA).and_then(|(_, i)| i.parse().ok()).unwrap_or(0)
    }

    /// The key of replica 0 of this key's build: the key the model itself derived.
    pub fn base(&self) -> InstanceKey {
        match self.config.rsplit_once(Self::REPLICA) {
            Some((base, i)) if i.parse::<usize>().is_ok() => InstanceKey { model: self.model.clone(), config: base.to_string() },
            _ => self.clone(),
        }
    }
}

impl std::fmt::Display for InstanceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]", self.model, self.config)
    }
}

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before the unit tests run.
#[cfg(test)]
brain_testutil::configure_test_binary!();
