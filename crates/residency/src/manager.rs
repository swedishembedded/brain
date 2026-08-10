// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`ResidencyManager`]: given `(model, action, invocation)`, ensure the right
//! model instance is **Hot** on a device — placing it on the emptiest GPU that fits,
//! or evicting least-recently-used instances to make room — then run it. Dropping an
//! evicted [`Instance`] frees its device memory (RAII). This is the "use all the
//! memory automatically" core; the scheduler (next) drives concurrency and batching
//! on top of it.
//!
//! Single-device instances are handled directly (`claim`/`run`/`evict`, tracked
//! in [`crate::lru::Residents`]). A model that spans multiple devices AT ONCE
//! (e.g. an int8 MoE model layer-sharded across two GPUs) registers separately
//! via [`ResidencyManager::register_multi`] and is placed by
//! [`ResidencyManager::claim_multi`] — real, honest per-device accounting via
//! [`crate::multi::MultiDeviceCost`]/[`crate::multi::pick_devices`] against the
//! SAME [`Budgets`] every single-device instance shares (so a multi-device
//! instance's bytes are never invisible to a single-device claim's budget
//! check, and vice versa).
//!
//! **What the multi-device path does NOT do (a deliberate, documented scope
//! limit, not an oversight)**: multi-device instances are NOT tracked in
//! [`crate::lru::Residents`] (whose `Entry` is single-device by construction)
//! and are therefore never chosen as LRU/cost-aware eviction VICTIMS — once
//! claimed, a multi-device instance stays resident until explicitly
//! `release_multi`'d/evicted by its own caller, not auto-evicted to make room
//! for something else. `claim_multi`'s OWN eviction fallback still works (it
//! can evict single-device LRU victims per needed device to make room for
//! itself), so a multi-device claim is not stuck behind stale single-device
//! residents — the gap is one-directional: nothing evicts a multi-device
//! instance automatically. Acceptable for the intended shape (one big model
//! held resident for the process lifetime, e.g. an int8-sharded Thinker), and
//! precisely the honest boundary this crate's own "gates that lie" discipline
//! prefers over silently pretending full LRU parity exists. Extending
//! `Residents` to a multi-device `Entry` (so eviction scoring can consider
//! multi-device victims too) is real, separate follow-up work if a future
//! caller genuinely needs it — not attempted here, since the original
//! dual-GPU residency work this integration grew out of is now closed;
//! `crate::executor` layers the async `Executor` dispatch this module's
//! synchronous `claim_multi` needed on top.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::budget::Budgets;
use crate::lru::Residents;
use crate::multi::{pick_devices, MultiDeviceCost, MultiDeviceResidentModel};
use crate::place::{could_ever_fit, no_exclude, pick_device, plan_eviction_with, CostAware, EvictionPolicy};
use crate::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

/// A hot instance handle: the (mutex-guarded) instance plus the device it lives on.
/// The scheduler runs it outside the manager lock; the key stays pinned meanwhile.
pub type InstanceHandle = Arc<Mutex<Box<dyn Instance>>>;

/// Why a claim could not produce a runnable instance. The executor MUST treat
/// these differently: `NoCapacity` is transient (retry when a lane frees a
/// device); `TooLarge` and `Activate` are permanent for the key — the queued
/// jobs must be failed, or they wait forever and wedge the group.
#[derive(Debug)]
pub enum ClaimError {
    /// No free device can host the instance right now (SOME device's usable
    /// budget could hold it, but not without evicting more than is
    /// currently evictable — try again once something frees).
    NoCapacity(String),
    /// The instance exceeds EVERY device's usable budget even fully empty —
    /// no eviction, however aggressive, could ever make room. Checked
    /// BEFORE planning any eviction, so a claim that can never succeed never
    /// costs anything else its residency (see `place::could_ever_fit`).
    TooLarge(String),
    /// The model/instance itself is unusable (unknown model, activation error).
    Activate(String),
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimError::NoCapacity(e) | ClaimError::TooLarge(e) | ClaimError::Activate(e) => write!(f, "{e}"),
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
    /// The instance is Warm (a prior eviction called
    /// [`Instance::demote`] instead of dropping it) — the caller must run
    /// [`Instance::promote`] on `device` (deferred to its own thread, same
    /// reason `Build`'s `activate` is: it can be slow) and report
    /// [`ResidencyManager::adopt`]/[`build_failed`](ResidencyManager::build_failed)
    /// exactly as for a fresh build. The existing `Instance` is reused, not
    /// rebuilt from the checkpoint.
    Promote(InstanceHandle),
}

/// [`Claimed`]'s multi-device sibling — carries a [`MultiDeviceResidentModel`]
/// (whose `activate_multi` takes a device SET) instead of a `ResidentModel`,
/// since [`ResidencyManager::claim_multi`] needs a different build contract,
/// not just a different placement.
pub enum ClaimedMulti {
    Hot(InstanceHandle),
    Build(Arc<dyn MultiDeviceResidentModel>),
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

/// One multi-device resident instance's placement — [`InstancePlacement`]'s
/// sibling for an instance that spans several devices instead of one.
#[derive(Clone, Debug)]
pub struct MultiInstancePlacement {
    pub key: InstanceKey,
    /// `(device, bytes on that device)` — every device this instance
    /// occupies, each with its own real byte count (never summed into one
    /// figure that could be mistaken for a single-device cost).
    pub devices: Vec<(Device, u64)>,
    pub tier: Tier,
}

/// A point-in-time residency + budget snapshot: every placed instance plus every
/// device's budget. Produced by [`ResidencyManager::report`] and surfaced through
/// the [`Executor`](crate::Executor) residency accessor so callers outside the
/// dispatcher thread (stats, D-Bus) can render the live memory/residency tree
/// without reaching into the manager's internals. Deterministically ordered.
#[derive(Clone, Debug, Default)]
pub struct ResidencyReport {
    pub placements: Vec<InstancePlacement>,
    /// Multi-device placements — kept SEPARATE from `placements` rather than
    /// folded in (an `InstancePlacement` has exactly one `device: Device`
    /// field, singular by construction; forcing a multi-device instance into
    /// that shape would mean picking one device to report and hiding the
    /// rest, exactly the kind of lying figure this crate's `multi` module
    /// exists to avoid). A renderer that ignores this field simply doesn't
    /// show multi-device instances, which is honest (empty), not wrong.
    pub multi_placements: Vec<MultiInstancePlacement>,
    pub budgets: Vec<DeviceBudget>,
}

/// A single-device [`MemCost`] naming `need` bytes on exactly `d`'s class
/// (VRAM for a GPU, NPU bytes for an NPU, host RAM for the CPU) and nothing
/// else — what [`plan_eviction_with`] needs to evaluate ONE specific device
/// of a multi-device cost in isolation. Shared by [`ResidencyManager::
/// claim_multi`] (which actually evicts) and [`ResidencyManager::
/// placeable_multi`] (which only checks feasibility) so the two stay in
/// lock-step by construction rather than by two independently-maintained
/// `match`es.
fn synth_cost_for(d: Device, need: u64) -> MemCost {
    match d {
        Device::Gpu(_) => MemCost::new(need, 0),
        Device::Npu(_) => MemCost::new(0, 0).with_npu(need),
        Device::Cpu => MemCost::new(0, need),
    }
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
    /// Models placeable across several devices at once — a SEPARATE registry
    /// from `models` (see this file's own module doc): `claim_multi` looks
    /// here, `claim` never does. A model may be registered in both maps
    /// under the same name if it wants to be reachable either way (not
    /// required by anything here).
    multi_models: HashMap<String, Arc<dyn MultiDeviceResidentModel>>,
    budgets: Budgets,
    residents: Residents,
    /// Multi-device residents' bookkeeping — parallel to `residents` but
    /// keyed on the same `InstanceKey` space. A name must not be claimed
    /// both ways at once, and BOTH claim paths enforce it: `claim` refuses a
    /// key resident here, `claim_multi` refuses a key resident in
    /// `residents` (each with a clean `ClaimError::Activate`, never a
    /// dispatcher-killing panic).
    multi_residents: HashMap<InstanceKey, MultiEntry>,
    instances: HashMap<InstanceKey, InstanceHandle>,
    /// Eviction/promotion audit log (most recent last) for reporting/tests.
    pub events: Vec<String>,
    /// Cumulative counters (never reset) — instance builds and evictions.
    pub builds: u64,
    pub evictions: u64,
    /// Which resident to evict first when a claim needs room. Defaults to
    /// [`CostAware`] (GDSF: `uses * reload_cost / age`) rather than strict LRU
    /// -- swapping a large model back in costs far more than a small one, and
    /// `brain perf residency` measured `CostAware` beating strict LRU on hit
    /// rate (54.3% vs 50.0%) under a shifting Zipf load. See `place.rs`'s
    /// module doc for the measurement this generalizes from.
    eviction: Box<dyn EvictionPolicy>,
}

/// One multi-device resident instance's bookkeeping — parallel to
/// [`crate::lru::Entry`], but spanning several devices and (see this file's
/// module doc) deliberately NOT part of the LRU/cost-aware eviction pool.
struct MultiEntry {
    cost: MultiDeviceCost,
    devices: Vec<Device>,
    /// True while a job is actively running — must not be evicted/dropped.
    pinned: bool,
}

impl ResidencyManager {
    pub fn new(budgets: Budgets) -> ResidencyManager {
        ResidencyManager {
            models: HashMap::new(),
            multi_models: HashMap::new(),
            budgets,
            residents: Residents::new(),
            multi_residents: HashMap::new(),
            instances: HashMap::new(),
            events: Vec::new(),
            builds: 0,
            evictions: 0,
            eviction: Box::new(CostAware),
        }
    }

    /// Override the eviction policy (builder-style) -- e.g. `Lru` for an A/B
    /// comparison, or a test wanting strict recency semantics.
    pub fn with_eviction_policy(mut self, policy: Box<dyn EvictionPolicy>) -> ResidencyManager {
        self.eviction = policy;
        self
    }

    /// Number of resident (budget-accounted) instances. Counted from the
    /// accounting map, not the built-instance map: a deferred build is already
    /// resident (placed, budgeted, pinned) while its lane is still activating.
    pub fn resident_count(&self) -> usize {
        self.residents.iter().count()
    }

    /// Every currently-built instance's own [`Instance::metrics`], keyed by
    /// `InstanceKey`. Polled by the DISPATCHER thread, so this must NEVER
    /// block: a `try_lock` skips any instance a lane is mid-`run_batch` on
    /// (its handle is locked for the whole batch — see `run_group`) rather
    /// than stalling the dispatcher until that lane frees. A skipped
    /// instance's metrics are simply stale until the next poll finds it
    /// free, which is correct for best-effort observability and was NOT
    /// correct as a blocking `.lock()` (confirmed live: it froze the
    /// dispatcher for the length of a running job, reproduced by
    /// `in_flight_reports_queued_and_running_jobs_with_monotonic_ids`
    /// timing out instead of returning promptly).
    pub fn all_metrics(&self) -> HashMap<InstanceKey, Vec<(String, serde_json::Value)>> {
        self.instances.iter().filter_map(|(k, h)| h.try_lock().ok().map(|inst| (k.clone(), inst.metrics()))).collect()
    }

    pub fn register(&mut self, model: Arc<dyn ResidentModel>) {
        self.models.insert(model.manifest().model.clone(), model);
    }

    /// Register a model reachable via [`Self::claim_multi`] — a SEPARATE
    /// registry from [`Self::register`] (see this file's own module doc).
    pub fn register_multi(&mut self, model: Arc<dyn MultiDeviceResidentModel>) {
        self.multi_models.insert(model.manifest().model.clone(), model);
    }

    pub fn manifests(&self) -> Vec<Manifest> {
        self.models.values().map(|m| m.manifest()).collect()
    }

    /// The instance key for `(model, action, inv)`, or `None` if the model is
    /// unknown under EITHER registry. Falls back to `multi_models` only when
    /// `models` doesn't have it — strictly additive: every model reachable
    /// before this fallback existed resolves exactly as it did (the single-
    /// device registry is always checked first), and this only adds names
    /// that previously produced `None` here (and, downstream, the executor's
    /// `Msg::Submit` handler's `no model 'x'` reply).
    pub fn instance_key_for(&self, model: &str, action: &str, inv: &Invocation) -> Option<InstanceKey> {
        self.models
            .get(model)
            .map(|m| m.instance_key(action, inv))
            .or_else(|| self.multi_models.get(model).map(|m| m.instance_key(action, inv)))
    }

    /// Whether `model` is registered as a [`MultiDeviceResidentModel`] — the
    /// executor's `assign` uses this to pick between the `claim`/`placeable`
    /// and `claim_multi`/`placeable_multi` branches for a queued group.
    pub fn is_multi(&self, model: &str) -> bool {
        self.multi_models.contains_key(model)
    }

    /// Number of resident multi-device instances (parallel to
    /// [`Self::resident_count`], which only counts single-device ones).
    pub fn resident_multi_count(&self) -> usize {
        self.multi_residents.len()
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
            || plan_eviction_with(&*self.eviction, &cost, &self.budgets, &self.residents, std::slice::from_ref(key), exclude).is_some()
    }

    /// [`Self::placeable`]'s multi-device sibling — the executor's scheduling
    /// filter for a [`MultiDeviceResidentModel`] group, mirroring exactly what
    /// [`Self::claim_multi`] can achieve (direct fit OR its own per-device LRU
    /// eviction fallback) WITHOUT mutating anything, the same relationship
    /// `placeable`/`claim` already have.
    ///
    /// Returns `true` — deliberately, though it reads like the wrong answer —
    /// when `estimate_multi` names ZERO devices. An empty cost is that
    /// method's documented "this model is unavailable right now" signal (see
    /// its own doc); filtering the group out HERE would mean it is never
    /// `placeable` on any future round either (the cost won't change), so its
    /// jobs would sit in the queue forever with no error and no explanation.
    /// Returning `true` instead lets the group reach [`Self::claim_multi`],
    /// which turns the same empty cost into a real, per-job
    /// [`ClaimError::Activate`] — a clean failure instead of a silent hang.
    pub fn placeable_multi(&self, key: &InstanceKey, model: &str, exclude: &HashSet<Device>) -> bool {
        if let Some(e) = self.multi_residents.get(key) {
            return e.devices.iter().all(|d| !exclude.contains(d));
        }
        let m = match self.multi_models.get(model) {
            Some(m) => m,
            None => return false,
        };
        let cost = m.estimate_multi(key);
        let wanted: Vec<Device> = cost.devices().collect();
        if wanted.is_empty() {
            return true; // unavailable model -- see doc above
        }
        if pick_devices(&cost, &self.budgets, exclude).is_some() {
            return true;
        }
        // Mirror claim_multi's per-device eviction fallback, read-only: every
        // named device must independently either already fit or be evictable
        // (single-device LRU victims on THAT device alone -- `only_d` excludes
        // every other device so `plan_eviction_with` cannot "succeed" by
        // picking a different card than the one actually needed).
        let every_device: HashSet<Device> = self.budgets.devices().collect();
        wanted.iter().all(|&d| {
            if exclude.contains(&d) {
                return false;
            }
            let need = cost.on(d);
            let b = match self.budgets.get(d) {
                Some(b) => b,
                None => return false,
            };
            if b.usable() < need {
                return false;
            }
            if b.fits(need) {
                return true;
            }
            let mut only_d = every_device.clone();
            only_d.remove(&d);
            plan_eviction_with(&*self.eviction, &synth_cost_for(d, need), &self.budgets, &self.residents, &[], &only_d).is_some()
        })
    }

    pub fn models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.models.keys().cloned().collect();
        v.sort();
        v
    }

    /// Current tier of each resident instance (for `Residency` reporting).
    pub fn residency(&self) -> Vec<(InstanceKey, Device, Tier)> {
        self.residents.iter().map(|(k, e)| (k.clone(), e.device, e.tier)).collect()
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
            .map(|(k, e)| InstancePlacement { key: k.clone(), device: e.device, tier: e.tier, mem: e.cost.on(e.device) })
            .collect();
        placements.sort_by(|a, b| (a.key.model.clone(), a.key.config.clone(), device_order(a.device)).cmp(&(b.key.model.clone(), b.key.config.clone(), device_order(b.device))));
        let mut multi_placements: Vec<MultiInstancePlacement> = self
            .multi_residents
            .iter()
            .map(|(k, e)| MultiInstancePlacement {
                key: k.clone(),
                devices: e.devices.iter().map(|&d| (d, e.cost.on(d))).collect(),
                tier: Tier::Hot,
            })
            .collect();
        multi_placements.sort_by(|a, b| (a.key.model.clone(), a.key.config.clone()).cmp(&(b.key.model.clone(), b.key.config.clone())));
        let mut budgets: Vec<DeviceBudget> = self
            .budgets
            .devices()
            .filter_map(|d| self.budgets.get(d).map(|b| DeviceBudget { device: d, total: b.total, reserved: b.reserved, used: b.used }))
            .collect();
        budgets.sort_by_key(|b| device_order(b.device));
        ResidencyReport { placements, multi_placements, budgets }
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
            Claimed::Promote(h) => {
                let result = h.lock().unwrap().promote(device);
                match result {
                    Ok(()) => self.adopt(&key, h),
                    Err(e) => {
                        self.build_failed(&key);
                        return Err(e);
                    }
                }
            }
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
        // Cross-registry guard: a key resident as a MULTI-device instance
        // must never be claimed through the single-device path — before this
        // check, the handle lookup below found the shared `instances` entry,
        // then the `residents` expect() panicked ON THE DISPATCHER THREAD,
        // killing scheduling for the whole server.
        if self.multi_residents.contains_key(&key) {
            return Err(ClaimError::Activate(format!("{key}: resident as a multi-device instance — claim it via claim_multi, not claim")));
        }
        // The instance object existing is still the real guard (matches the
        // pre-Warm invariant exactly): a cold build's `residents.insert`
        // pre-accounts the slot before `self.instances` gets the handle
        // (via `adopt`, on the caller's thread) — so `residents` can have a
        // NOT-yet-adopted entry for `key` while a build is in flight, and
        // that in-flight window must keep falling into the cold-build path
        // below (which is itself made a no-op-ish re-place by the budget
        // already being charged), never this branch.
        if let Some(handle) = self.instances.get(&key).cloned() {
            // Never expect() here: this runs on the dispatcher thread, where a
            // panic kills scheduling for every model. A handle with no
            // residency entry is a registry-wiring bug — fail the one claim.
            let entry = *self
                .residents
                .get(&key)
                .ok_or_else(|| ClaimError::Activate(format!("{key}: instance handle exists but has no single-device residency entry (registry mismatch)")))?;
            if entry.tier == Tier::Hot {
                self.residents.touch(&key);
                self.residents.set_pinned(&key, true);
                return Ok((Claimed::Hot(handle), entry.device, key));
            }
            // Warm: place it like a cold build (pick a device, evict if
            // needed — the entry itself is never a candidate victim of its
            // own placement, same as a cold build's `keep`), but hand back
            // `Claimed::Promote` so the caller reuses the existing
            // `Instance` via `promote()` instead of rebuilding it.
            let hot_cost = m.estimate(&key);
            let device = match pick_device(&hot_cost, &self.budgets, exclude) {
                Some(d) => d,
                None => {
                    if !could_ever_fit(&hot_cost, &self.budgets) {
                        return Err(ClaimError::TooLarge(format!(
                            "{key} ({} MiB) exceeds every device's usable budget even fully empty",
                            hot_cost.vram.max(hot_cost.ram).max(hot_cost.npu) >> 20
                        )));
                    }
                    let plan = plan_eviction_with(&*self.eviction, &hot_cost, &self.budgets, &self.residents, std::slice::from_ref(&key), exclude)
                        .ok_or_else(|| {
                            ClaimError::NoCapacity(format!(
                                "{key} ({} MiB) has no room right now — nothing currently evictable frees enough",
                                hot_cost.vram.max(hot_cost.ram).max(hot_cost.npu) >> 20
                            ))
                        })?;
                    for victim in &plan.victims {
                        self.evict(victim);
                    }
                    plan.device
                }
            };
            self.budgets.release(entry.device, entry.cost.on(entry.device));
            self.budgets.alloc(device, hot_cost.on(device));
            self.residents.retier(&key, hot_cost, device, Tier::Hot);
            self.residents.set_pinned(&key, true);
            self.events.push(format!("promote {key} -> {device:?} (warm->hot)"));
            return Ok((Claimed::Promote(handle), device, key));
        }
        // Cold: place + pre-account + pin NOW (so nothing steals the budget or
        // evicts the slot), but defer the potentially slow/hanging activate() to
        // the caller's thread.
        let cost = m.estimate(&key);
        let device = match pick_device(&cost, &self.budgets, exclude) {
            Some(d) => d,
            None => {
                // Checked BEFORE planning any eviction: a claim that could never
                // fit on any device even fully empty must fail cleanly right now,
                // not after evicting everything else and still not fitting.
                if !could_ever_fit(&cost, &self.budgets) {
                    return Err(ClaimError::TooLarge(format!(
                        "{key} ({} MiB) exceeds every device's usable budget even fully empty",
                        cost.vram.max(cost.ram).max(cost.npu) >> 20
                    )));
                }
                let plan = plan_eviction_with(&*self.eviction, &cost, &self.budgets, &self.residents, std::slice::from_ref(&key), exclude)
                    .ok_or_else(|| {
                        ClaimError::NoCapacity(format!(
                            "{key} ({} MiB) has no room right now — nothing currently evictable frees enough",
                            cost.vram.max(cost.ram).max(cost.npu) >> 20
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

    /// [`Self::claim`]'s multi-device sibling — places (or finds hot) an
    /// instance of a [`MultiDeviceResidentModel`] registered via
    /// [`Self::register_multi`]. See this file's own module doc for what the
    /// eviction fallback here does and does not cover.
    pub fn claim_multi(
        &mut self,
        model: &str,
        action: &str,
        inv: &Invocation,
        exclude: &HashSet<Device>,
    ) -> Result<(ClaimedMulti, Vec<Device>, InstanceKey), ClaimError> {
        let m = self
            .multi_models
            .get(model)
            .ok_or_else(|| ClaimError::Activate(format!("no multi-device model '{model}'")))?
            .clone();
        let key = m.instance_key(action, inv);
        // Cross-registry guard, symmetric to `claim`'s: a key resident as a
        // SINGLE-device instance must not be claimed through the multi path
        // (the old expect("multi-resident") below panicked the dispatcher).
        if self.residents.get(&key).is_some() {
            return Err(ClaimError::Activate(format!("{key}: resident as a single-device instance — claim it via claim, not claim_multi")));
        }
        if let Some(handle) = self.instances.get(&key).cloned() {
            let entry = self
                .multi_residents
                .get_mut(&key)
                .ok_or_else(|| ClaimError::Activate(format!("{key}: instance handle exists but has no multi-device residency entry (registry mismatch)")))?;
            entry.pinned = true;
            let devices = entry.devices.clone();
            return Ok((ClaimedMulti::Hot(handle), devices, key));
        }
        // A build for this key is already reserved (present in `multi_residents`,
        // budget already allocated on every device) but not yet adopted (absent
        // from `instances`) -- i.e. between a PRIOR `claim_multi`'s `Build` result
        // and its caller's `adopt_multi`. Re-running the placement logic below
        // would allocate budget a SECOND time on every device for the same
        // instance. The `Executor` never triggers this in practice (its `running`
        // set keeps a key out of `group_rows` for exactly this window), but
        // `ResidencyManager` itself has no other guard against a caller that
        // isn't protected that way, so make the double-claim impossible here
        // rather than relying on every future caller getting it right.
        if self.multi_residents.contains_key(&key) {
            return Err(ClaimError::NoCapacity(format!("{key}: build already in flight")));
        }
        let cost = m.estimate_multi(&key);
        let wanted: Vec<Device> = cost.devices().collect();
        if wanted.is_empty() {
            return Err(ClaimError::Activate(format!("{key}: estimate_multi named zero devices")));
        }
        let devices = match pick_devices(&cost, &self.budgets, exclude) {
            Some(d) => d,
            None => {
                // Per-device eviction fallback: for EACH device this instance
                // needs, evict single-device LRU victims on THAT device
                // specifically to make room — never touching another
                // multi-device resident (this manager's own module doc names
                // that as the deliberate scope limit). Reuses the existing
                // single-device eviction planner per device by excluding
                // every other device, so it cannot "succeed" by picking a
                // different one than the one actually needed.
                let every_device: HashSet<Device> = self.budgets.devices().collect();
                for &d in &wanted {
                    if exclude.contains(&d) {
                        return Err(ClaimError::NoCapacity(format!("{key}: device {d:?} is excluded")));
                    }
                    let need = cost.on(d);
                    let b = self
                        .budgets
                        .get(d)
                        .ok_or_else(|| ClaimError::NoCapacity(format!("{key}: device {d:?} has no budget")))?;
                    if b.usable() < need {
                        return Err(ClaimError::NoCapacity(format!(
                            "{key} ({} MiB on {d:?}) is too large for that device's usable budget",
                            need >> 20
                        )));
                    }
                    if b.fits(need) {
                        continue; // already fits on this device, nothing to evict here
                    }
                    let mut only_d = every_device.clone();
                    only_d.remove(&d);
                    let plan = plan_eviction_with(&*self.eviction, &synth_cost_for(d, need), &self.budgets, &self.residents, &[], &only_d)
                        .ok_or_else(|| ClaimError::NoCapacity(format!("{key}: cannot free {} MiB on {d:?}", need >> 20)))?;
                    for victim in &plan.victims {
                        self.evict(victim);
                    }
                }
                pick_devices(&cost, &self.budgets, exclude)
                    .ok_or_else(|| ClaimError::NoCapacity(format!("{key}: does not fit even after eviction")))?
            }
        };
        for &d in &devices {
            self.budgets.alloc(d, cost.on(d));
        }
        self.multi_residents.insert(key.clone(), MultiEntry { cost, devices: devices.clone(), pinned: true });
        self.events.push(format!("promote {key} -> {devices:?} (building, multi-device)"));
        Ok((ClaimedMulti::Build(m), devices, key))
    }

    /// [`Self::adopt`]'s multi-device sibling.
    pub fn adopt_multi(&mut self, key: &InstanceKey, handle: InstanceHandle) -> InstanceHandle {
        self.instances.insert(key.clone(), handle.clone());
        self.builds += 1;
        self.events.push(format!("built {key} (multi-device)"));
        handle
    }

    /// [`Self::build_failed`]'s multi-device sibling: unwinds the
    /// pre-accounted budget on EVERY device this claim reserved.
    pub fn build_failed_multi(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.multi_residents.remove(key) {
            for &d in &entry.devices {
                self.budgets.release(d, entry.cost.on(d));
            }
        }
        self.instances.remove(key);
        self.events.push(format!("build-failed {key} (multi-device)"));
    }

    /// [`Self::release`]'s multi-device sibling.
    pub fn release_multi(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.multi_residents.get_mut(key) {
            entry.pinned = false;
        }
    }

    /// Demote (drop) a multi-device instance, freeing its memory on EVERY
    /// device it occupied. Public (unlike the single-device [`Self::evict`]):
    /// nothing auto-evicts a multi-device resident (this file's own module
    /// doc explains why), so a caller that genuinely wants one gone —
    /// swapping to a different checkpoint, a shutdown path — calls this
    /// directly.
    ///
    /// Refuses (returns `false`, evicts nothing) while the instance is
    /// PINNED — a lane is actively running a job against it (`claim_multi`
    /// pins on every claim; `release_multi` unpins after). Evicting out from
    /// under a running job would drop the `Instance` (freeing its GPU memory)
    /// while a lane still holds a strong `Arc` to it and is mid-call — the
    /// budget would say the memory is free while the lane is still using it,
    /// exactly the kind of lying figure this crate's `multi` module exists to
    /// avoid. Returns `true` if an unpinned entry was found and evicted;
    /// `true` is also NOT returned for a key that was never resident (no-op,
    /// same as `false`) — check separately if the caller needs to
    /// distinguish "refused" from "nothing to evict".
    pub fn evict_multi(&mut self, key: &InstanceKey) -> bool {
        match self.multi_residents.get(key) {
            None => false,
            Some(entry) if entry.pinned => false,
            Some(_) => {
                let entry = self.multi_residents.remove(key).expect("checked Some above");
                for &d in &entry.devices {
                    self.budgets.release(d, entry.cost.on(d));
                }
                self.instances.remove(key);
                self.evictions += 1;
                self.events.push(format!("evict {key} <- {:?} (multi-device)", entry.devices));
                true
            }
        }
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
    /// Free `key`'s device slot to make room for something else. Tries a
    /// soft demotion to `Tier::Warm` first (releases the device buffers,
    /// keeps the `Instance` and its host bytes alive, so a later claim for
    /// the same key can [`Instance::promote`] straight back instead of
    /// rebuilding from the checkpoint) — this is the entire "made real"
    /// part of `Tier::Warm`: every existing caller of `evict` gets it for
    /// free, with zero behaviour change for a model that hasn't opted in,
    /// since `demote` defaults to `Err` and this falls straight through to
    /// the original full-drop below whenever it does.
    fn evict(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.residents.get(key).copied() {
            if entry.tier == Tier::Hot {
                if let (Some(handle), Some(model)) = (self.instances.get(key).cloned(), self.models.get(&key.model).cloned()) {
                    let warm_cost = model.estimate_at(key, Tier::Warm);
                    // The Warm copy is a real host-RAM charge — it must FIT
                    // (pool-aware: on a unified-memory box the HOST_POOL is
                    // the same physical bytes the accelerators use). Checked
                    // BEFORE `demote()` releases anything: repeated multi-GB
                    // demotions that nothing refuses are exactly the swap
                    // cliff memauth's doc warns about. When it doesn't fit,
                    // fall through to the full drop below — freeing the
                    // device slot is the caller's actual requirement; keeping
                    // a warm copy is only an optimization. (Conservative for
                    // a CPU-Hot instance, whose own Hot bytes are not counted
                    // as freed here; demoting CPU→CPU-warm is not a shape any
                    // current caller produces.)
                    if self.budgets.fits_on(Device::Cpu, warm_cost.on(Device::Cpu)) && handle.lock().unwrap().demote(Tier::Warm).is_ok() {
                        self.budgets.release(entry.device, entry.cost.on(entry.device));
                        self.budgets.alloc(Device::Cpu, warm_cost.on(Device::Cpu));
                        self.residents.retier(key, warm_cost, Device::Cpu, Tier::Warm);
                        self.events.push(format!("demote {key} <- {:?} (warm)", entry.device));
                        return;
                    }
                }
            }
        }
        // Full drop: today's only behaviour, and the fallback whenever
        // `demote` isn't supported, the entry was already below Hot, or the
        // model/instance lookup above came up empty.
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

    /// A model whose `Instance` implements `demote`/`promote` for real: `live`
    /// counts the `Instance` object's whole lifetime (activate..Drop); `hot`
    /// counts device residency specifically (activate/promote add, demote
    /// subtracts) — the two diverging is exactly the property a Warm
    /// demotion has that a full evict doesn't.
    struct DemotableFake {
        name: String,
        vram: u64,
        warm_ram: u64,
        live: Arc<AtomicU32>,
        hot: Arc<AtomicU32>,
    }
    struct DemotableInst {
        live: Arc<AtomicU32>,
        hot: Arc<AtomicU32>,
    }
    impl Drop for DemotableInst {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl ResidentModel for DemotableFake {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(self.vram, 0)
        }
        fn estimate_at(&self, _k: &InstanceKey, tier: Tier) -> MemCost {
            match tier {
                Tier::Hot => MemCost::new(self.vram, 0),
                Tier::Warm | Tier::Cold => MemCost::new(0, self.warm_ram),
            }
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            self.hot.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(DemotableInst { live: self.live.clone(), hot: self.hot.clone() }))
        }
    }
    impl crate::Instance for DemotableInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new().blob("out", Blob::new(Media::Bytes, vec![1])))
        }
        fn demote(&mut self, tier: Tier) -> Result<(), String> {
            if tier == Tier::Hot {
                return Err("Hot is not a demotion target".to_string());
            }
            self.hot.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
        fn promote(&mut self, _device: Device) -> Result<(), String> {
            self.hot.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// The actual "made real" wiring: evicting a model that opts into
    /// `demote` releases its device slot WITHOUT dropping the `Instance`
    /// (`live` unchanged, `hot` drops) — and a later claim for the same key
    /// promotes it straight back (`hot` rises again, `live` STILL
    /// unchanged: no second `activate()`/checkpoint reload ever happened).
    /// A model that hasn't opted in (`Fake`/`FakeInst`) keeps dropping on
    /// eviction exactly as `three_models_on_one_gpu_swap_by_lru` already
    /// proves — this test is the other half.
    #[test]
    fn evict_demotes_to_warm_when_the_model_opts_in_and_a_later_claim_promotes_it_back() {
        let live = Arc::new(AtomicU32::new(0));
        let hot = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB); // 22 GB usable
        budgets.set(Device::Cpu, 64 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(DemotableFake { name: "a".into(), vram: 10 * GB, warm_ram: GB, live: live.clone(), hot: hot.clone() }));
        mgr.register(Arc::new(Fake { name: "b".into(), vram: 10 * GB, live: live.clone() }));
        mgr.register(Arc::new(Fake { name: "c".into(), vram: 10 * GB, live: live.clone() }));

        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 2);
        assert_eq!(hot.load(Ordering::SeqCst), 1);
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().used, 0);

        // c needs 10, only 2 GB free -> evict LRU (a) -> demote to Warm,
        // not a full drop.
        mgr.run("c", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(hot.load(Ordering::SeqCst), 0, "a's device slot must be released");
        // a (demoted, still alive) + b (hot) + c (just activated) = 3.
        assert_eq!(live.load(Ordering::SeqCst), 3, "a's Instance must still be alive -- demoted, not dropped");
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().used, GB, "the Warm RAM charge must be tracked");
        assert!(mgr.events.iter().any(|e| e.contains("demote a")), "events: {:?}", mgr.events);
        let tiers: HashMap<String, Tier> = mgr.residency().into_iter().map(|(k, _, t)| (k.model, t)).collect();
        assert_eq!(tiers.get("a"), Some(&Tier::Warm));
        assert_eq!(tiers.get("c"), Some(&Tier::Hot));

        // Claiming "a" again promotes it back -- same Instance (live
        // unchanged: no second activate), device-resident again (hot back
        // to 1). This itself needs room: c (10) + b (10) = 20 <= 22, so a's
        // 10 GB forces one more LRU eviction (b, non-demotable -> full drop).
        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(hot.load(Ordering::SeqCst), 1, "a must be device-resident again");
        // b's full drop (3 -> 2) proves the eviction-for-room still happened;
        // a itself contributed no change to `live` (promote, not activate).
        assert_eq!(live.load(Ordering::SeqCst), 2, "promoting a reused the existing Instance, no rebuild");
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().used, 0, "the Warm RAM charge must be released on promote");
        assert!(mgr.events.iter().any(|e| e.contains("warm->hot")), "events: {:?}", mgr.events);
        let tiers: HashMap<String, Tier> = mgr.residency().into_iter().map(|(k, _, t)| (k.model, t)).collect();
        assert_eq!(tiers.get("a"), Some(&Tier::Hot));
    }

    /// SPEC (audit F2): a Warm demotion charges host RAM, so it must CHECK the
    /// host budget first. When the warm bytes don't fit, eviction falls through
    /// to a full drop — never an unrefused overcommit (the swap cliff on
    /// unified-memory boxes).
    #[test]
    fn warm_demotion_that_does_not_fit_host_ram_falls_back_to_a_full_drop() {
        let live = Arc::new(AtomicU32::new(0));
        let hot = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB); // 22 GB usable
        budgets.set(Device::Cpu, 4 * GB, 0); // too small for an 8 GB warm cache
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(DemotableFake { name: "a".into(), vram: 20 * GB, warm_ram: 8 * GB, live: live.clone(), hot: hot.clone() }));
        mgr.register(Arc::new(Fake { name: "c".into(), vram: 20 * GB, live: live.clone() }));

        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        // c forces an eviction of a; the 8 GB warm copy exceeds the 4 GB CPU
        // budget, so a must be fully dropped, not demoted.
        mgr.run("c", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 1, "a must be fully dropped when its warm copy cannot fit host RAM");
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().used, 0, "no host charge may be left behind");
        assert!(mgr.events.iter().any(|e| e.contains("evict a")), "events: {:?}", mgr.events);
        assert!(!mgr.events.iter().any(|e| e.contains("demote a")), "events: {:?}", mgr.events);
    }

    /// SPEC (audit F3): one name resident through one registry must be REFUSED
    /// (clean ClaimError) when claimed through the other — both directions.
    /// Before the guard existed, each direction panicked the dispatcher thread
    /// via an expect() on the other registry's bookkeeping.
    #[test]
    fn cross_registry_claims_are_refused_cleanly_not_panics() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        // The same name registered BOTH ways (explicitly permitted).
        mgr.register(Arc::new(Fake { name: "x".into(), vram: GB, live: live.clone() }));
        mgr.register_multi(Arc::new(MultiFake { name: "x".into(), per_gpu: GB, live: live.clone() }));

        // Resident via the multi path -> the single-device claim must refuse.
        let (_handle, key) = claim_multi_built(&mut mgr, "x").unwrap();
        let err = mgr.claim("x", "run", &Invocation::new(), &no_exclude()).err().expect("must be refused");
        assert!(matches!(err, ClaimError::Activate(_)), "expected Activate, got {err:?}");
        assert!(err.to_string().contains("claim_multi"), "{err}");
        mgr.release_multi(&key);
        mgr.evict_multi(&key);

        // Resident via the single path -> the multi claim must refuse.
        mgr.run("x", "run", &Invocation::new(), &mut |_| {}).unwrap();
        let err = mgr.claim_multi("x", "run", &Invocation::new(), &HashSet::new()).err().expect("must be refused");
        assert!(matches!(err, ClaimError::Activate(_)), "expected Activate, got {err:?}");
        assert!(err.to_string().contains("via claim"), "{err}");
    }

    /// A model bigger than every device's usable budget, even fully empty,
    /// must fail cleanly with `ClaimError::TooLarge` and cost NOTHING else
    /// its residency — no eviction plan is even attempted, because none
    /// could ever succeed (the "larger than every tier" scenario).
    #[test]
    fn too_large_for_any_device_fails_cleanly_without_evicting_anything() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB); // 22 GB usable, ever.
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(Fake { name: "small".into(), vram: 10 * GB, live: live.clone() }));
        mgr.register(Arc::new(Fake { name: "huge".into(), vram: 100 * GB, live: live.clone() }));

        // A resident already occupies the card -- if TooLarge were mistakenly
        // planned as an eviction, this would be the (wrong) victim.
        mgr.run("small", "run", &Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 1);

        let err = mgr.claim("huge", "run", &Invocation::new(), &no_exclude()).err().expect("must be refused");
        assert!(matches!(err, ClaimError::TooLarge(_)), "expected TooLarge, got {err:?}");

        // Nothing was evicted or touched to serve a claim that could never succeed.
        assert_eq!(mgr.evictions, 0);
        assert_eq!(live.load(Ordering::SeqCst), 1, "the existing resident must be untouched");
        let hot: Vec<String> = mgr.residency().into_iter().map(|(k, _, _)| k.model).collect();
        assert_eq!(hot, vec!["small".to_string()]);
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

    /// A fake model that spans BOTH gpu0 and gpu1 at once — occupies real,
    /// distinct bytes on each, the shape `MultiDeviceCost` exists for.
    struct MultiFake {
        name: String,
        per_gpu: u64,
        live: Arc<AtomicU32>,
    }
    impl ResidentModel for MultiFake {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "fake multi", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0) // not the path claim_multi uses; never consulted there
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            Err("MultiFake: single-device activate is not this model's contract".to_string())
        }
    }
    impl crate::multi::MultiDeviceResidentModel for MultiFake {
        fn estimate_multi(&self, _k: &InstanceKey) -> crate::multi::MultiDeviceCost {
            crate::multi::MultiDeviceCost::new(vec![(Device::Gpu(0), self.per_gpu), (Device::Gpu(1), self.per_gpu)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, devices: &[Device]) -> Result<Box<dyn crate::Instance>, String> {
            assert_eq!(devices, [Device::Gpu(0), Device::Gpu(1)], "activate_multi must see exactly the devices estimate_multi named");
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }

    fn claim_multi_built(mgr: &mut ResidencyManager, model: &str) -> Result<(InstanceHandle, InstanceKey), String> {
        let (claimed, devices, key) = mgr.claim_multi(model, "run", &Invocation::new(), &HashSet::new()).map_err(String::from)?;
        let handle = match claimed {
            ClaimedMulti::Hot(h) => h,
            ClaimedMulti::Build(m) => match m.activate_multi(&key, &devices) {
                Ok(inst) => mgr.adopt_multi(&key, Arc::new(Mutex::new(inst))),
                Err(e) => {
                    mgr.build_failed_multi(&key);
                    return Err(e);
                }
            },
        };
        Ok((handle, key))
    }

    #[test]
    fn multi_device_claim_reserves_real_bytes_on_every_named_device() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register_multi(Arc::new(MultiFake { name: "int8thinker".into(), per_gpu: 15 * GB, live: live.clone() }));

        let (handle, key) = claim_multi_built(&mut mgr, "int8thinker").unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 1);
        assert!(handle.lock().unwrap().run("run", &Invocation::new(), &mut |_| {}).is_ok());

        // Real per-device accounting: 15 GB used on EACH card, not double
        // counted, not folded into one figure.
        assert_eq!(mgr.budgets().get(Device::Gpu(0)).unwrap().used, 15 * GB);
        assert_eq!(mgr.budgets().get(Device::Gpu(1)).unwrap().used, 15 * GB);

        let report = mgr.report();
        assert_eq!(report.multi_placements.len(), 1);
        assert_eq!(report.multi_placements[0].key, key);
        let mut devs = report.multi_placements[0].devices.clone();
        devs.sort_by_key(|&(d, _)| match d {
            Device::Gpu(i) => i,
            _ => u32::MAX,
        });
        assert_eq!(devs, vec![(Device::Gpu(0), 15 * GB), (Device::Gpu(1), 15 * GB)]);
        assert!(report.placements.is_empty(), "a multi-device instance must not also appear in the single-device placements list");

        mgr.release_multi(&key);
        drop(handle); // drop this test's OWN strong ref -- evict_multi's removal from `instances` is not the last one otherwise
        mgr.evict_multi(&key);
        assert_eq!(live.load(Ordering::SeqCst), 0, "evict_multi must free every device");
        assert_eq!(mgr.budgets().get(Device::Gpu(0)).unwrap().used, 0);
        assert_eq!(mgr.budgets().get(Device::Gpu(1)).unwrap().used, 0);
    }

    #[test]
    fn multi_device_claim_evicts_single_device_lru_victims_to_make_room() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        // Fill both cards with ordinary single-device residents first.
        for n in ["a", "b"] {
            mgr.register(Arc::new(Fake { name: n.into(), vram: 20 * GB, live: live.clone() }));
        }
        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap(); // -> gpu0
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap(); // -> gpu1
        assert_eq!(live.load(Ordering::SeqCst), 2);

        // A 15 GB-per-card multi-device model needs room neither card has
        // free right now (24 - 20 = 4 GB each) -- must evict on BOTH.
        mgr.register_multi(Arc::new(MultiFake { name: "int8thinker".into(), per_gpu: 15 * GB, live: live.clone() }));
        let (_handle, _key) = claim_multi_built(&mut mgr, "int8thinker").unwrap();
        assert_eq!(live.load(Ordering::SeqCst), 1, "both single-device residents evicted, one multi-device instance now live");
        assert!(mgr.events.iter().any(|e| e.contains("evict a")), "events: {:?}", mgr.events);
        assert!(mgr.events.iter().any(|e| e.contains("evict b")), "events: {:?}", mgr.events);
    }

    #[test]
    fn multi_device_cost_too_large_for_one_device_is_a_clean_error_not_silent_partial_placement() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        // 30 GB on gpu1 alone can never fit a 24 GB card, however much gets evicted.
        mgr.register_multi(Arc::new(MultiFakeUneven { live: live.clone() }));
        let err = match claim_multi_built(&mut mgr, "uneven") {
            Ok(_) => panic!("expected a capacity error"),
            Err(e) => e,
        };
        assert!(err.contains("too large"), "{err}");
        assert_eq!(live.load(Ordering::SeqCst), 0, "a failed claim must not have activated anything");
    }

    struct MultiFakeUneven {
        live: Arc<AtomicU32>,
    }
    impl ResidentModel for MultiFakeUneven {
        fn manifest(&self) -> Manifest {
            Manifest::new("uneven", "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("uneven", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            Err("not this model's contract".to_string())
        }
    }
    impl crate::multi::MultiDeviceResidentModel for MultiFakeUneven {
        fn estimate_multi(&self, _k: &InstanceKey) -> crate::multi::MultiDeviceCost {
            crate::multi::MultiDeviceCost::new(vec![(Device::Gpu(0), 1 * GB), (Device::Gpu(1), 30 * GB)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }
}
