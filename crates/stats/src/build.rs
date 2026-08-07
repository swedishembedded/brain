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
//! - **requests** — the executor's in-flight jobs ([`Executor::in_flight`]), each
//!   queued or running, mapped to a `RequestStat` (model/action/phase/since_ms).
//!
//! `connections` is still left to a dedicated source: per-transport socket tracking
//! is a front-end concern, not something the executor sees.

use capability::Manifest;
use residency::{Device, DeviceBudget, Executor, InFlightJob, InstancePlacement, ResidencyReport, Tier};

use crate::snapshot::{Accelerator, ExecutorStat, Instance, ModelStat, RequestStat, StatsSnapshot};
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
    let mut extra = std::collections::BTreeMap::new();
    // GPU 0 only — see `gpu_sysfs_extra`'s doc for why a second GPU is left
    // alone rather than guessed at.
    if kind == "gpu" && index == 0 {
        gpu_sysfs_extra(&mut extra);
    }
    Accelerator {
        id: device_id(b.device),
        kind: kind.to_string(),
        name,
        index,
        mem_total: b.total,
        mem_used: b.used,
        mem_reserved: b.reserved,
        // Real occupancy needs the i915 PMU (CAP_PERFMON, not available to
        // this process by default) or an equivalent vendor counter — nothing
        // this crate can read unprivileged is honestly "utilization", so
        // this typed field stays None. What sysfs freq/throttle state IS
        // readable unprivileged lands in `extra` below, clearly labelled.
        util: None,
        extra,
    }
}

/// Best-effort GPU frequency/throttle state from `/sys/class/drm`, landed in
/// `extra` rather than the typed `util` field — see `accelerator_from_budget`'s
/// comment for why. Deliberately duplicated from (not imported from)
/// `crates/perf/src/devicetel.rs`: `brain-stats` is kept dependency-light on
/// purpose (see this crate's own doc comment) so `braintop`/D-Bus can depend
/// on it without pulling in the model/backend graph `brain-perf` needs; both
/// readers are independently small and pure-`std`, so the duplication costs
/// little and preserves that layering — the same trade this repo already
/// makes for the PCI/UUID identity query duplicated between `backend-wgpu`
/// and `backend-vulkan`.
///
/// Only ever asked for `gpu0`: correlating a DRM card index with brain's own
/// `Device::Gpu(i)` indexing (which comes from wgpu/Vulkan adapter
/// enumeration, not `/sys/class/drm/cardN`) is not attempted here, so a
/// second budgeted GPU is left with no sysfs fields rather than one
/// possibly attributed to the wrong card.
fn gpu_sysfs_extra(extra: &mut std::collections::BTreeMap<String, serde_json::Value>) {
    let Some(card) = intel_card_dir() else { return };
    let gt0 = card.join("gt/gt0");
    let read_u32 = |rel: &str| -> Option<u32> { std::fs::read_to_string(gt0.join(rel)).ok()?.trim().parse().ok() };
    let read_bool01 = |rel: &str| -> Option<bool> { read_u32(rel).map(|v| v != 0) };
    if let Some(v) = read_u32("rps_act_freq_mhz") {
        extra.insert("gpu_freq_mhz".to_string(), serde_json::json!(v));
    }
    if let Some(v) = read_bool01("throttle_reason_pl1") {
        extra.insert("gpu_throttled_pl1".to_string(), serde_json::json!(v));
    }
    if let Some(v) = read_bool01("throttle_reason_thermal") {
        extra.insert("gpu_throttled_thermal".to_string(), serde_json::json!(v));
    }
}

/// Find the first Intel (`vendor == 0x8086`) DRM card directory — discovered,
/// never hardcoded, so this works whichever index the kernel assigned.
fn intel_card_dir() -> Option<std::path::PathBuf> {
    let mut cards: Vec<std::path::PathBuf> = std::fs::read_dir("/sys/class/drm")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
        })
        .collect();
    cards.sort();
    cards.into_iter().find(|c| {
        std::fs::read_to_string(c.join("device/vendor")).map(|v| v.trim() == "0x8086").unwrap_or(false)
    })
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
        admitted: s.admitted,
        queue_depth: s.queue_depth as u64,
        extra: Default::default(),
    }
}

/// Map one executor in-flight job into a snapshot [`RequestStat`]. `provider` stays
/// `None` — it is a transport concept (which front-end submitted), not something the
/// executor knows; the executor only sees model/action/phase.
fn request_from_inflight(j: InFlightJob) -> RequestStat {
    RequestStat {
        id: j.id.to_string(),
        provider: None,
        model: Some(j.model),
        action: Some(j.action),
        phase: j.phase,
        since_ms: j.since_ms,
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
        // multi_placements (multi-device resident instances) is not yet
        // rendered here -- `ModelStat`/`Instance`'s stats schema is
        // single-device by construction (one `device` field), same as
        // `InstancePlacement`; showing a multi-device instance for real
        // needs a schema change (a device LIST per instance), not just
        // reading this field. No multi-device model exists in production
        // yet (`crates/residency/src/multi.rs` is this session's new
        // foundation), so "not shown yet" is an honest, low-urgency gap, not
        // a silently wrong number. See `.todo/omni-int8-dual-gpu-residency.md`.
        let ResidencyReport { placements, budgets, multi_placements: _ } = self.exec.residency();
        snap.executor = executor_stat(&stats);
        // Accelerators come straight from the device budgets — one row per
        // budgeted device, so the set adapts to any machine (0..N GPUs/NPUs).
        snap.accelerators.extend(budgets.iter().map(accelerator_from_budget));
        snap.models.extend(models_from(&self.exec.manifests(), &placements));
        // In-flight work → `requests`, straight from the executor's live queue +
        // running set. `connections` stays empty: per-transport socket tracking is a
        // front-end concern, out of scope for this executor-backed source.
        snap.requests.extend(self.exec.in_flight().into_iter().map(request_from_inflight));
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

    #[test]
    fn gpu_sysfs_extra_never_panics_and_only_inserts_bools_and_a_u32() {
        // Environment-dependent (this box happens to have a real Intel GPU,
        // a CI runner may not) -- the contract under test is "never panics,
        // and whatever it inserts is honestly typed," not a specific value.
        let mut extra = std::collections::BTreeMap::new();
        gpu_sysfs_extra(&mut extra);
        if let Some(v) = extra.get("gpu_freq_mhz") {
            assert!(v.is_u64(), "gpu_freq_mhz must be a plain number, got {v:?}");
        }
        for k in ["gpu_throttled_pl1", "gpu_throttled_thermal"] {
            if let Some(v) = extra.get(k) {
                assert!(v.is_boolean(), "{k} must be a bool, got {v:?}");
            }
        }
    }

    #[test]
    fn only_gpu0_gets_sysfs_extra_a_second_gpu_is_left_alone() {
        let gpu1 = accelerator_from_budget(&DeviceBudget { device: Device::Gpu(1), total: 24 * GB, reserved: 0, used: 0 });
        assert!(gpu1.extra.is_empty(), "a second GPU must not get gpu0's sysfs data attributed to it");
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

    /// A snapshot built from an executor with in-flight work carries a non-empty
    /// `requests` section, mapped from the executor's live jobs (model/action/phase/
    /// since_ms; provider stays None). A gated model holds one job running while the
    /// snapshot is taken, so the assertion is deterministic (no timing guesswork).
    #[test]
    fn snapshot_has_requests_when_executor_has_in_flight_work() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        use std::sync::mpsc::channel;
        use std::time::{Duration, Instant};

        struct Gated {
            entered: Arc<AtomicU32>,
            release: Arc<AtomicBool>,
        }
        struct GatedInst {
            entered: Arc<AtomicU32>,
            release: Arc<AtomicBool>,
        }
        impl ResidentModel for Gated {
            fn manifest(&self) -> Manifest {
                Manifest::new("g", "gated", vec![ActionSpec::new("run", "run")])
            }
            fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
                InstanceKey::new("g", "default")
            }
            fn estimate(&self, _k: &InstanceKey) -> MemCost {
                MemCost::new(GB, 0)
            }
            fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn RInstance>, String> {
                Ok(Box::new(GatedInst { entered: self.entered.clone(), release: self.release.clone() }))
            }
        }
        impl RInstance for GatedInst {
            fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
                self.entered.fetch_add(1, Ordering::SeqCst);
                let start = Instant::now();
                while !self.release.load(Ordering::SeqCst) {
                    if start.elapsed() > Duration::from_secs(5) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
            }
        }

        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        let entered = Arc::new(AtomicU32::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Gated { entered: entered.clone(), release: release.clone() })];
        let exec = Executor::start(models, budgets, Policy::default());

        let (tx, rx) = channel();
        exec.submit(residency::Job::new("g", "run", Invocation::new()).reply(move |r| { let _ = tx.send(r); }));
        // Wait until the job is genuinely running on the lane before sampling.
        let start = Instant::now();
        while entered.load(Ordering::SeqCst) == 0 {
            assert!(start.elapsed() < Duration::from_secs(5), "gated run never started");
            std::thread::sleep(Duration::from_millis(2));
        }

        let snap = snapshot_from_executor(&exec);
        assert!(!snap.requests.is_empty(), "expected a non-empty requests section");
        let r = &snap.requests[0];
        assert_eq!(r.model.as_deref(), Some("g"));
        assert_eq!(r.action.as_deref(), Some("run"));
        assert_eq!(r.phase, "running");
        assert_eq!(r.provider, None, "provider is a transport concept, not set here");
        assert!(r.id.parse::<u64>().is_ok(), "id should be the numeric job id: {}", r.id);

        release.store(true, Ordering::SeqCst);
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    }
}
