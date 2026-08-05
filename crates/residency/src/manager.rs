// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`ResidencyManager`]: given `(model, action, invocation)`, ensure the right
//! model instance is **Hot** on a device — placing it on the emptiest GPU that fits,
//! or evicting least-recently-used instances to make room — then run it. Dropping an
//! evicted [`Instance`] frees its device memory (RAII). This is the "use all the
//! memory automatically" core; the scheduler (next) drives concurrency and batching
//! on top of it.
//!
//! Single-device instances are handled directly. A model that spans multiple devices
//! (e.g. z-image's encoder + DiT on two cards) reports the extra footprint via
//! [`ResidentModel::estimate`]'s `ram`/secondary accounting and pins its own cards;
//! the manager still tracks the primary-device budget.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::budget::Budgets;
use crate::lru::Residents;
use crate::place::{no_exclude, pick_device, plan_eviction};
use crate::{Device, Instance, InstanceKey, ResidentModel, Tier};

/// A hot instance handle: the (mutex-guarded) instance plus the device it lives on.
/// The scheduler runs it outside the manager lock; the key stays pinned meanwhile.
pub type InstanceHandle = Arc<Mutex<Box<dyn Instance>>>;

/// Why a claim could not produce a runnable instance. The executor MUST treat
/// these differently: `NoCapacity` is transient (retry when a lane frees a
/// device); `Activate` is permanent for the key — the queued jobs must be
/// failed, or they wait forever and wedge the group.
#[derive(Debug)]
pub enum ClaimError {
    /// No free device can host the instance right now.
    NoCapacity(String),
    /// The model/instance itself is unusable (unknown model, activation error).
    Activate(String),
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimError::NoCapacity(e) | ClaimError::Activate(e) => write!(f, "{e}"),
        }
    }
}

impl From<ClaimError> for String {
    fn from(e: ClaimError) -> String {
        e.to_string()
    }
}

/// What a successful claim yields: an already-hot instance, or a placed,
/// pre-accounted, pinned slot whose **build is deferred to the caller's
/// thread**. Deferring matters: `activate()` can take seconds (weight load,
/// NPU graph compile) or hang outright, and it must never run on the
/// dispatcher thread where it would freeze ALL scheduling. The caller runs
/// [`ResidentModel::activate`] and then reports
/// [`ResidencyManager::adopt`] (success) or
/// [`ResidencyManager::build_failed`] (unwind the accounting).
pub enum Claimed {
    Hot(InstanceHandle),
    Build(Arc<dyn ResidentModel>),
}

/// One resident instance's placement — the per-model residency the stats
/// subsystem renders (which model is Hot, on which device, at what memory cost).
#[derive(Clone, Debug)]
pub struct InstancePlacement {
    pub key: InstanceKey,
    pub device: Device,
    pub tier: Tier,
    /// Bytes this instance occupies **on its device** (VRAM/RAM/NPU as applicable).
    pub mem: u64,
}

/// One device's live budget (total capacity, reserved headroom, bytes in use) —
/// the accelerator memory picture the stats subsystem renders (nvidia-smi-like).
#[derive(Clone, Copy, Debug)]
pub struct DeviceBudget {
    pub device: Device,
    pub total: u64,
    pub reserved: u64,
    pub used: u64,
}

/// A point-in-time residency + budget snapshot: every placed instance plus every
/// device's budget. Produced by [`ResidencyManager::report`] and surfaced through
/// the [`Executor`](crate::Executor) residency accessor so callers outside the
/// dispatcher thread (stats, D-Bus) can render the live memory/residency tree
/// without reaching into the manager's internals. Deterministically ordered.
#[derive(Clone, Debug, Default)]
pub struct ResidencyReport {
    pub placements: Vec<InstancePlacement>,
    pub budgets: Vec<DeviceBudget>,
}

/// Total order over devices for deterministic reporting: CPU, then GPUs by index,
/// then NPUs by index (HashMap iteration order is otherwise unstable).
fn device_order(d: Device) -> (u8, u32) {
    match d {
        Device::Cpu => (0, 0),
        Device::Gpu(i) => (1, i),
        Device::Npu(i) => (2, i),
    }
}

/// Owns the resident model instances, their budgets, and the LRU. Not thread-safe by
/// itself — the scheduler owns one behind its worker(s).
pub struct ResidencyManager {
    models: HashMap<String, Arc<dyn ResidentModel>>,
    budgets: Budgets,
    residents: Residents,
    instances: HashMap<InstanceKey, InstanceHandle>,
    /// Eviction/promotion audit log (most recent last) for reporting/tests.
    pub events: Vec<String>,
    /// Cumulative counters (never reset) — instance builds and evictions.
    pub builds: u64,
    pub evictions: u64,
}

impl ResidencyManager {
    pub fn new(budgets: Budgets) -> ResidencyManager {
        ResidencyManager { models: HashMap::new(), budgets, residents: Residents::new(), instances: HashMap::new(), events: Vec::new(), builds: 0, evictions: 0 }
    }

    /// Number of resident (budget-accounted) instances. Counted from the
    /// accounting map, not the built-instance map: a deferred build is already
    /// resident (placed, budgeted, pinned) while its lane is still activating.
    pub fn resident_count(&self) -> usize {
        self.residents.iter().count()
    }

    pub fn register(&mut self, model: Arc<dyn ResidentModel>) {
        self.models.insert(model.manifest().model.clone(), model);
    }

    pub fn manifests(&self) -> Vec<Manifest> {
        self.models.values().map(|m| m.manifest()).collect()
    }

    /// The instance key for `(model, action, inv)`, or `None` if the model is unknown.
    pub fn instance_key_for(&self, model: &str, action: &str, inv: &Invocation) -> Option<InstanceKey> {
        self.models.get(model).map(|m| m.instance_key(action, inv))
    }

    /// Could `key` run **now** on a device not in `exclude`? A resident instance is
    /// runnable iff its device is free; a cold one iff it can be placed (or evicted
    /// into) on some free device. Used by the parallel scheduler to skip groups whose
    /// only device is busy, without mutating anything.
    pub fn placeable(&self, key: &InstanceKey, model: &str, exclude: &HashSet<Device>) -> bool {
        if let Some(e) = self.residents.get(key) {
            return !exclude.contains(&e.device);
        }
        let m = match self.models.get(model) {
            Some(m) => m,
            None => return false,
        };
        let cost = m.estimate(key);
        pick_device(&cost, &self.budgets, exclude).is_some()
            || plan_eviction(&cost, &self.budgets, &self.residents, std::slice::from_ref(key), exclude).is_some()
    }

    pub fn models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.models.keys().cloned().collect();
        v.sort();
        v
    }

    /// Current tier of each resident instance (for `Residency` reporting).
    pub fn residency(&self) -> Vec<(InstanceKey, Device, Tier)> {
        self.residents.iter().map(|(k, e)| (k.clone(), e.device, Tier::Hot)).collect()
    }

    /// A full residency + budget snapshot for stats/reporting: every placed
    /// instance (with its device-memory cost) plus every device's budget,
    /// deterministically ordered. This is the data-source the stats subsystem and
    /// the D-Bus `StatsSnapshot`/`StatsStream` surface render from — it is
    /// computed inside the dispatcher thread (which owns the manager) and shipped
    /// out via the [`Executor`](crate::Executor) residency accessor.
    pub fn report(&self) -> ResidencyReport {
        let mut placements: Vec<InstancePlacement> = self
            .residents
            .iter()
            .map(|(k, e)| InstancePlacement { key: k.clone(), device: e.device, tier: Tier::Hot, mem: e.cost.on(e.device) })
            .collect();
        placements.sort_by(|a, b| (a.key.model.clone(), a.key.config.clone(), device_order(a.device)).cmp(&(b.key.model.clone(), b.key.config.clone(), device_order(b.device))));
        let mut budgets: Vec<DeviceBudget> = self
            .budgets
            .devices()
            .filter_map(|d| self.budgets.get(d).map(|b| DeviceBudget { device: d, total: b.total, reserved: b.reserved, used: b.used }))
            .collect();
        budgets.sort_by_key(|b| device_order(b.device));
        ResidencyReport { placements, budgets }
    }

    pub fn budgets(&self) -> &Budgets {
        &self.budgets
    }

    /// Ensure the instance for `(model, action, inv)` is Hot, then run the action.
    /// Promotes (evicting LRU as needed) automatically. Pins the instance while it
    /// runs so a concurrent request can't evict it mid-job. (Synchronous path —
    /// deferred builds run inline on this thread.)
    pub fn run(&mut self, model: &str, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (handle, key) = self.claim_built(model, action, inv)?;
        let out = handle.lock().unwrap().run(action, inv, progress);
        self.release(&key);
        out
    }

    /// [`claim`](Self::claim) + build inline when needed — for synchronous callers
    /// that are not a scheduler dispatcher.
    fn claim_built(&mut self, model: &str, action: &str, inv: &Invocation) -> Result<(InstanceHandle, InstanceKey), String> {
        let (claimed, device, key) = self.claim(model, action, inv, &no_exclude()).map_err(String::from)?;
        let handle = match claimed {
            Claimed::Hot(h) => h,
            Claimed::Build(m) => match m.activate(&key, device) {
                Ok(inst) => self.adopt(&key, Arc::new(Mutex::new(inst))),
                Err(e) => {
                    self.build_failed(&key);
                    return Err(e);
                }
            },
        };
        Ok((handle, key))
    }

    /// Place + **pin** the instance for `(model, action, inv)`, returning either a
    /// hot handle or a deferred build (see [`Claimed`]). The caller runs the handle
    /// (outside the manager lock, so other lanes proceed) and MUST call
    /// [`release`](Self::release) after — or, for a deferred build,
    /// [`adopt`](Self::adopt) / [`build_failed`](Self::build_failed) first.
    /// `exclude` names devices a concurrent lane is already using (so this
    /// placement avoids them).
    pub fn claim(
        &mut self,
        model: &str,
        action: &str,
        inv: &Invocation,
        exclude: &HashSet<Device>,
    ) -> Result<(Claimed, Device, InstanceKey), ClaimError> {
        let m = self
            .models
            .get(model)
            .ok_or_else(|| ClaimError::Activate(format!("no model '{model}'")))?
            .clone();
        let key = m.instance_key(action, inv);
        if self.instances.contains_key(&key) {
            self.residents.touch(&key);
            self.residents.set_pinned(&key, true);
            let handle = self.instances.get(&key).expect("hot").clone();
            let device = self.residents.get(&key).expect("resident").device;
            return Ok((Claimed::Hot(handle), device, key));
        }
        // Cold: place + pre-account + pin NOW (so nothing steals the budget or
        // evicts the slot), but defer the potentially slow/hanging activate() to
        // the caller's thread.
        let cost = m.estimate(&key);
        let device = match pick_device(&cost, &self.budgets, exclude) {
            Some(d) => d,
            None => {
                let plan = plan_eviction(&cost, &self.budgets, &self.residents, std::slice::from_ref(&key), exclude)
                    .ok_or_else(|| {
                        ClaimError::NoCapacity(format!(
                            "{key} ({} MiB) is too large for any available device",
                            cost.vram >> 20
                        ))
                    })?;
                for victim in &plan.victims {
                    self.evict(victim);
                }
                plan.device
            }
        };
        self.budgets.alloc(device, cost.on(device));
        self.residents.insert(key.clone(), cost, device);
        self.residents.set_pinned(&key, true);
        self.events.push(format!("promote {key} -> {device:?} (building)"));
        Ok((Claimed::Build(m), device, key))
    }

    /// A deferred build succeeded: adopt the instance so later claims find it hot.
    pub fn adopt(&mut self, key: &InstanceKey, handle: InstanceHandle) -> InstanceHandle {
        self.instances.insert(key.clone(), handle.clone());
        self.builds += 1;
        self.events.push(format!("built {key}"));
        handle
    }

    /// A deferred build failed: unwind the pre-accounted budget + resident slot.
    /// The claim is over — do NOT also call [`release`](Self::release).
    pub fn build_failed(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.residents.remove(key) {
            self.budgets.release(entry.device, entry.cost.on(entry.device));
        }
        self.instances.remove(key);
        self.events.push(format!("build-failed {key}"));
    }

    /// Unpin an instance after a run and mark it most-recently-used.
    pub fn release(&mut self, key: &InstanceKey) {
        self.residents.set_pinned(key, false);
        self.residents.touch(key);
    }

    /// Run several same-key invocations of one action on a single hot instance —
    /// the hot-path-reuse batch. The instance is promoted once and pinned for the
    /// whole group (so it can't be evicted between jobs), then its `run_batch` runs
    /// them (a model with real batch support does one forward; others loop).
    pub fn run_batch(&mut self, model: &str, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(Progress)) -> Result<Vec<ActionResult>, String> {
        let first = invs.first().ok_or("empty batch")?;
        let (handle, key) = self.claim_built(model, action, first)?;
        let out = handle.lock().unwrap().run_batch(action, invs, &mut |_i, p| progress(p));
        self.release(&key);
        Ok(out)
    }

    /// Demote (drop) an instance, freeing its device memory.
    fn evict(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.residents.remove(key) {
            self.budgets.release(entry.device, entry.cost.on(entry.device));
            self.instances.remove(key); // drops the Instance → frees the GPU
            self.evictions += 1;
            self.events.push(format!("evict {key} <- {:?}", entry.device));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemCost;
    use capability::{ActionResult, ActionSpec, Blob, Media, Outcome};
    use std::sync::atomic::{AtomicU32, Ordering};

    const GB: u64 = 1 << 30;

    /// A fake model whose instances count live GPU builds (via a shared counter), so
    /// a test can watch automatic swap free memory without a GPU.
    struct Fake {
        name: String,
        vram: u64,
        live: Arc<AtomicU32>,
    }
    struct FakeInst {
        live: Arc<AtomicU32>,
    }
    impl Drop for FakeInst {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl ResidentModel for Fake {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }
    impl crate::Instance for FakeInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
    }

    #[test]
    fn three_models_on_one_gpu_swap_by_lru() {
        // One 24 GB card, 2 GB reserved → 22 usable. Three 10 GB models: two fit, the
        // third forces the LRU one out.
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB);
        let mut mgr = ResidencyManager::new(budgets);
        for n in ["a", "b", "c"] {
            mgr.register(Arc::new(Fake { name: n.into(), vram: 10 * GB, live: live.clone() }));
        }

        // Run a, then b — both fit (20 GB <= 22).
        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 2);
        assert_eq!(mgr.residency().len(), 2);

        // Run c → needs 10, only 2 free → evict LRU (a) → 2 resident, c hot, a gone.
        mgr.run("c", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 2, "one instance evicted, memory freed");
        let hot: Vec<String> = mgr.residency().into_iter().map(|(k, _, _)| k.model).collect();
        assert!(hot.contains(&"c".to_string()) && hot.contains(&"b".to_string()) && !hot.contains(&"a".to_string()));
        assert!(mgr.events.iter().any(|e| e.contains("evict a")), "events: {:?}", mgr.events);

        // Re-running b (still hot) is a no-op promotion.
        let before = live.load(Ordering::SeqCst);
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), before);
    }

    #[test]
    fn balances_across_two_gpus() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        for n in ["a", "b"] {
            mgr.register(Arc::new(Fake { name: n.into(), vram: 20 * GB, live: live.clone() }));
        }
        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        // Both resident, one per card (b placed on the emptier GPU 1).
        let devs: Vec<Device> = mgr.residency().into_iter().map(|(_, d, _)| d).collect();
        assert!(devs.contains(&Device::Gpu(0)) && devs.contains(&Device::Gpu(1)));
        assert_eq!(live.load(Ordering::SeqCst), 2);
    }
}
