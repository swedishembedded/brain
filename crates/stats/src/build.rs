// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Live [`StatsSource`]s that read the running system, plus the one-call
//! [`snapshot_from_executor`] convenience the D-Bus surface uses.
//!
//! [`ExecutorSource`] is the primary wiring: one cheap `Executor` clone yields the
//! whole picture — counters ([`Executor::stats`]), the model catalog
//! ([`Executor::manifests`]), and the residency + budget report
//! ([`Executor::residency`]). From those three it fills, entirely from data:
//!
//! - **accelerators** — one row per device that has a budget (CPU + every GPU +
//!   every NPU), with total/used/reserved memory. The set adapts to whatever the
//!   machine budgeted; nothing is hardcoded.
//! - **models** — the manifest catalog joined with per-instance placement, so each
//!   model shows where (if anywhere) it is resident.
//! - **executor** — the scheduler counters.
//!
//! `requests`/`connections` are left to dedicated sources (a `JobRegistry`-backed
//! request source can be added without touching this one).

use capability::Manifest;
use residency::{Device, DeviceBudget, Executor, InstancePlacement, ResidencyReport, Tier};

use crate::snapshot::{Accelerator, ExecutorStat, Instance, ModelStat, StatsSnapshot};
use crate::source::{Assembler, StatsSource};

/// The `--device`-style id for a device (`cpu`, `gpu0`, `npu1`).
pub fn device_id(d: Device) -> String {
    match d {
        Device::Cpu => "cpu".to_string(),
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
    }
}

/// `(kind, index, name)` for a device.
fn device_facets(d: Device) -> (&'static str, u32, String) {
    match d {
        Device::Cpu => ("cpu", 0, "CPU".to_string()),
        Device::Gpu(i) => ("gpu", i, format!("GPU {i}")),
        Device::Npu(i) => ("npu", i, format!("NPU {i}")),
    }
}

fn tier_name(t: Tier) -> &'static str {
    match t {
        Tier::Hot => "hot",
        Tier::Warm => "warm",
        Tier::Cold => "cold",
    }
}

fn accelerator_from_budget(b: &DeviceBudget) -> Accelerator {
    let (kind, index, name) = device_facets(b.device);
    Accelerator {
        id: device_id(b.device),
        kind: kind.to_string(),
        name,
        index,
        mem_total: b.total,
        mem_used: b.used,
        mem_reserved: b.reserved,
        util: None,
        extra: Default::default(),
    }
}

/// Join the manifest catalog with per-instance placement into `ModelStat`s.
pub fn models_from(manifests: &[Manifest], placements: &[InstancePlacement]) -> Vec<ModelStat> {
    let mut models: Vec<ModelStat> = manifests
        .iter()
        .map(|m| {
            let instances: Vec<Instance> = placements
                .iter()
                .filter(|p| p.key.model == m.model)
                .map(|p| Instance { device: device_id(p.device), tier: tier_name(p.tier).to_string(), mem: p.mem, extra: Default::default() })
                .collect();
            ModelStat {
                id: m.model.clone(),
                family: m.summary.clone(),
                capabilities: m.actions.iter().map(|a| a.name.clone()).collect(),
                resident: !instances.is_empty(),
                instances,
                extra: Default::default(),
            }
        })
        .collect();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

fn executor_stat(s: &residency::executor::Stats) -> ExecutorStat {
    ExecutorStat {
        builds: s.builds,
        evictions: s.evictions,
        batches: s.batches,
        jobs: s.jobs,
        resident: s.resident as u64,
        queue_peak: s.queue_peak as u64,
        max_batch: s.max_batch as u64,
        max_parallel: s.max_parallel as u64,
        extra: Default::default(),
    }
}

/// The live source that reads a residency [`Executor`]. Holds a cheap clone, so it
/// can be registered once and sampled repeatedly (e.g. by the D-Bus stream task).
pub struct ExecutorSource {
    exec: Executor,
}

impl ExecutorSource {
    pub fn new(exec: Executor) -> ExecutorSource {
        ExecutorSource { exec }
    }
}

impl StatsSource for ExecutorSource {
    fn contribute(&self, snap: &mut StatsSnapshot) {
        let stats = self.exec.stats();
        let ResidencyReport { placements, budgets } = self.exec.residency();
        snap.executor = executor_stat(&stats);
        // Accelerators come straight from the device budgets — one row per
        // budgeted device, so the set adapts to any machine (0..N GPUs/NPUs).
        snap.accelerators.extend(budgets.iter().map(accelerator_from_budget));
        snap.models.extend(models_from(self.exec.manifests(), &placements));
    }
}

/// Build a full snapshot from a single [`Executor`] — the D-Bus `StatsSnapshot`
/// method and `StatsStream` signal both use this. Assembles via the standard
/// [`Assembler`] so additional sources can be layered in later without changing
/// call sites.
pub fn snapshot_from_executor(exec: &Executor) -> StatsSnapshot {
    Assembler::new().register(ExecutorSource::new(exec.clone())).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use residency::budget::Budgets;
    use residency::{Instance as RInstance, InstanceKey, MemCost, Policy, ResidentModel};
    use capability::{ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, Progress};
    use std::sync::Arc;

    const GB: u64 = 1 << 30;

    /// A minimal stub resident model (no GPU): advertises a couple of actions and
    /// an activation that returns a trivial instance.
    struct Stub {
        name: &'static str,
        caps: Vec<&'static str>,
        vram: u64,
    }
    struct StubInst;
    impl ResidentModel for Stub {
        fn manifest(&self) -> Manifest {
            Manifest::new(self.name, "stub model", self.caps.iter().map(|c| ActionSpec::new(c, c)).collect())
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn RInstance>, String> {
            Ok(Box::new(StubInst))
        }
    }
    impl RInstance for StubInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    fn stub_executor() -> Executor {
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Cpu, 8 * GB, 0);
        let models: Vec<Arc<dyn ResidentModel>> = vec![
            Arc::new(Stub { name: "alpha", caps: vec!["generate", "embed"], vram: 10 * GB }),
            Arc::new(Stub { name: "beta", caps: vec!["transcribe"], vram: 4 * GB }),
        ];
        Executor::start(models, budgets, Policy::default())
    }

    #[test]
    fn snapshot_from_running_executor_is_well_formed_and_data_driven() {
        let exec = stub_executor();
        // Nothing resident yet: accelerators still enumerate from budgets, models
        // list from the catalog, none resident.
        let snap = snapshot_from_executor(&exec);
        assert_eq!(snap.schema, crate::snapshot::SCHEMA_VERSION);
        // Two budgeted devices (CPU + GPU 0) → two accelerator rows, from data.
        assert_eq!(snap.accelerators.len(), 2);
        assert!(snap.accelerators.iter().any(|a| a.id == "cpu" && a.kind == "cpu"));
        let gpu = snap.accelerators.iter().find(|a| a.id == "gpu0").expect("gpu0 row");
        assert_eq!(gpu.mem_total, 24 * GB);
        assert_eq!(gpu.mem_reserved, 2 * GB);
        // Both catalog models present, sorted, capabilities carried through.
        assert_eq!(snap.models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["alpha", "beta"]);
        let alpha = snap.models.iter().find(|m| m.id == "alpha").unwrap();
        assert_eq!(alpha.capabilities, vec!["generate", "embed"]);
        assert!(!alpha.resident);
        assert!(alpha.instances.is_empty());

        // Run alpha → it becomes resident on GPU 0; the next snapshot reflects it.
        exec.run_blocking("alpha", "generate", Invocation::new(), |_| {}).unwrap();
        let snap = snapshot_from_executor(&exec);
        let alpha = snap.models.iter().find(|m| m.id == "alpha").unwrap();
        assert!(alpha.resident, "alpha must be resident after a run");
        let inst = alpha.instances.first().expect("one instance");
        assert_eq!(inst.device, "gpu0");
        assert_eq!(inst.tier, "hot");
        assert_eq!(inst.mem, 10 * GB);
        // The executor section reflects the resident instance. (Cumulative
        // counters like `builds`/`batches` sync on the lane's lagging `Done`
        // message, so they are eventually-consistent and not asserted here; the
        // residency report — read through the dispatcher — is the consistent one.)
        assert!(snap.executor.resident >= 1);
        // GPU 0's used memory now includes the resident instance.
        let gpu = snap.accelerators.iter().find(|a| a.id == "gpu0").unwrap();
        assert!(gpu.mem_used >= 10 * GB);
        // The whole thing serializes to a parseable JSON document.
        let json = snap.to_json_string();
        assert!(StatsSnapshot::from_json_str(&json).is_ok());
    }
}
