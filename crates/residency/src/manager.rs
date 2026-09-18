// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`ResidencyManager`]: given `(model, action, invocation)`, ensure the right
//! model instance is **Hot** on a device - placing it on the emptiest GPU that fits,
//! or evicting least-recently-used instances to make room - then run it. Dropping an
//! evicted [`Instance`] frees its device memory (RAII). This is the "use all the
//! memory automatically" core; the scheduler (next) drives concurrency and batching
//! on top of it.
//!
//! Single-device instances are handled directly (`claim`/`run`/`evict`, tracked
//! in [`crate::lru::Residents`]). A model that spans multiple devices AT ONCE
//! (e.g. an int8 MoE model layer-sharded across two GPUs) registers separately
//! via [`ResidencyManager::register_multi`] and is placed by
//! [`ResidencyManager::claim_multi`] - real, honest per-device accounting via
//! [`crate::multi::MultiDeviceCost`]/[`crate::multi::pick_devices`] against the
//! SAME [`Budgets`] every single-device instance shares (so a multi-device
//! instance's bytes are never invisible to a single-device claim's budget
//! check, and vice versa).
//!
//! **What the multi-device path does NOT do (a deliberate, documented scope
//! limit, not an oversight)**: multi-device instances are NOT tracked in
//! [`crate::lru::Residents`] (whose `Entry` is single-device by construction)
//! and are therefore never chosen as LRU/cost-aware eviction VICTIMS - once
//! claimed, a multi-device instance stays resident until explicitly
//! `release_multi`'d/evicted by its own caller, not auto-evicted to make room
//! for something else. `claim_multi`'s OWN eviction fallback still works (it
//! can evict single-device LRU victims per needed device to make room for
//! itself), so a multi-device claim is not stuck behind stale single-device
//! residents - the gap is one-directional: nothing evicts a multi-device
//! instance automatically. Acceptable for the intended shape (one big model
//! held resident for the process lifetime, e.g. an int8-sharded Thinker), and
//! precisely the honest boundary this crate's own "gates that lie" discipline
//! prefers over silently pretending full LRU parity exists. Extending
//! `Residents` to a multi-device `Entry` (so eviction scoring can consider
//! multi-device victims too) is real, separate follow-up work if a future
//! caller genuinely needs it - not attempted here, since the original
//! dual-GPU residency work this integration grew out of is now closed;
//! `crate::executor` layers the async `Executor` dispatch this module's
//! synchronous `claim_multi` needed on top.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::budget::Budgets;
use crate::lru::Residents;
use crate::multi::{pick_devices, MultiDeviceCost, MultiDeviceResidentModel};
use crate::runplan::{Eviction, PlanError, Placement, RunPlan};
use crate::place::{could_ever_fit, no_exclude, pick_device, plan_eviction_with, CostAware, EvictionPolicy};
use crate::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

/// A hot instance handle: the (mutex-guarded) instance plus the device it lives on.
/// The scheduler runs it outside the manager lock; the key stays pinned meanwhile.
pub type InstanceHandle = Arc<Mutex<Box<dyn Instance>>>;

/// Why a claim could not produce a runnable instance. The executor MUST treat
/// these differently: `NoCapacity` is transient (retry when a lane frees a
/// device); `TooLarge` and `Activate` are permanent for the key - the queued
/// jobs must be failed, or they wait forever and wedge the group.
#[derive(Debug)]
pub enum ClaimError {
    /// No free device can host the instance right now (SOME device's usable
    /// budget could hold it, but not without evicting more than is
    /// currently evictable - try again once something frees).
    NoCapacity(String),
    /// The instance exceeds EVERY device's usable budget even fully empty -
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
    /// [`Instance::demote`] instead of dropping it) - the caller must run
    /// [`Instance::promote`] on `device` (deferred to its own thread, same
    /// reason `Build`'s `activate` is: it can be slow) and report
    /// [`ResidencyManager::adopt`]/[`build_failed`](ResidencyManager::build_failed)
    /// exactly as for a fresh build. The existing `Instance` is reused, not
    /// rebuilt from the checkpoint.
    Promote(InstanceHandle),
}

/// [`Claimed`]'s multi-device sibling - carries a [`MultiDeviceResidentModel`]
/// (whose `activate_multi` takes a device SET) instead of a `ResidentModel`,
/// since [`ResidencyManager::claim_multi`] needs a different build contract,
/// not just a different placement.
pub enum ClaimedMulti {
    Hot(InstanceHandle),
    Build(Arc<dyn MultiDeviceResidentModel>),
}

/// One resident instance's placement - the per-model residency the stats
/// subsystem renders (which model is Hot, on which device, at what memory cost).
#[derive(Clone, Debug)]
pub struct InstancePlacement {
    pub key: InstanceKey,
    pub device: Device,
    pub tier: Tier,
    /// Bytes this instance occupies **on its device** (VRAM/RAM/NPU as applicable).
    pub mem: u64,
}

/// One device's live budget (total capacity, reserved headroom, bytes in use) -
/// the accelerator memory picture the stats subsystem renders (nvidia-smi-like).
#[derive(Clone, Copy, Debug)]
pub struct DeviceBudget {
    pub device: Device,
    pub total: u64,
    pub reserved: u64,
    pub used: u64,
}

/// One multi-device resident instance's placement - [`InstancePlacement`]'s
/// sibling for an instance that spans several devices instead of one.
#[derive(Clone, Debug)]
pub struct MultiInstancePlacement {
    pub key: InstanceKey,
    /// `(device, bytes on that device)` - every device this instance
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
    /// Multi-device placements - kept SEPARATE from `placements` rather than
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
/// else - what [`plan_eviction_with`] needs to evaluate ONE specific device
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
/// itself - the scheduler owns one behind its worker(s).
pub struct ResidencyManager {
    models: HashMap<String, Arc<dyn ResidentModel>>,
    /// Models placeable across several devices at once - a SEPARATE registry
    /// from `models` (see this file's own module doc): `claim_multi` looks
    /// here, `claim` never does. A model may be registered in both maps
    /// under the same name if it wants to be reachable either way (not
    /// required by anything here).
    multi_models: HashMap<String, Arc<dyn MultiDeviceResidentModel>>,
    budgets: Budgets,
    residents: Residents,
    /// Multi-device residents' bookkeeping - parallel to `residents` but
    /// keyed on the same `InstanceKey` space. A name must not be claimed
    /// both ways at once, and BOTH claim paths enforce it: `claim` refuses a
    /// key resident here, `claim_multi` refuses a key resident in
    /// `residents` (each with a clean `ClaimError::Activate`, never a
    /// dispatcher-killing panic).
    multi_residents: HashMap<InstanceKey, MultiEntry>,
    instances: HashMap<InstanceKey, InstanceHandle>,
    /// Eviction/promotion audit log (most recent last) for reporting/tests.
    /// BOUNDED (a ring of the last [`Self::MAX_EVENTS`]): a long-lived server
    /// churning instances used to grow this Vec forever - an unbounded audit
    /// log with no production reader is a slow leak, not observability.
    pub events: std::collections::VecDeque<String>,
    /// Cumulative counters (never reset) - instance builds and evictions.
    pub builds: u64,
    pub evictions: u64,
    /// Which resident to evict first when a claim needs room. Defaults to
    /// [`CostAware`] (GDSF: `uses * reload_cost / age`) rather than strict LRU
    /// -- swapping a large model back in costs far more than a small one, and
    /// `brain perf residency` measured `CostAware` beating strict LRU on hit
    /// rate under a shifting Zipf load. See `place.rs`'s
    /// module doc for the measurement this generalizes from.
    eviction: Box<dyn EvictionPolicy>,
}

/// One multi-device resident instance's bookkeeping - parallel to
/// [`crate::lru::Entry`], but spanning several devices and (see this file's
/// module doc) deliberately NOT part of the LRU/cost-aware eviction pool.
struct MultiEntry {
    cost: MultiDeviceCost,
    devices: Vec<Device>,
    /// True while a job is actively running - must not be evicted/dropped.
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
            events: std::collections::VecDeque::new(),
            builds: 0,
            evictions: 0,
            eviction: Box::new(CostAware),
        }
    }

    /// The audit ring's bound - see `events`.
    const MAX_EVENTS: usize = 256;

    fn event(&mut self, e: String) {
        crate::log::info(&e);
        if self.events.len() >= Self::MAX_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(e);
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
    /// (its handle is locked for the whole batch - see `run_group`) rather
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

    /// Register a model reachable via [`Self::claim_multi`] - a SEPARATE
    /// registry from [`Self::register`] (see this file's own module doc).
    pub fn register_multi(&mut self, model: Arc<dyn MultiDeviceResidentModel>) {
        self.multi_models.insert(model.manifest().model.clone(), model);
    }

    pub fn manifests(&self) -> Vec<Manifest> {
        self.models.values().map(|m| m.manifest()).collect()
    }

    /// The instance key for `(model, action, inv)`, or `None` if the model is
    /// unknown under EITHER registry. Falls back to `multi_models` only when
    /// `models` doesn't have it - strictly additive: every model reachable
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

    /// Whether `model` is registered as a [`MultiDeviceResidentModel`] - the
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

    /// **What would happen if this job started?** - the same computation
    /// [`Self::placeable`] performs, with its reasoning kept instead of
    /// collapsed into a bool.
    ///
    /// Answers required memory, where it already is (if anywhere), which
    /// device it would go to, exactly which residents would be evicted to
    /// make room and what that frees, every device it could run on at all,
    /// and the bytes that would cross into device memory to activate it.
    ///
    /// Mutates nothing and reserves nothing. See [`crate::runplan`]'s own
    /// module doc on why a plan is a prediction rather than a reservation,
    /// and why that distinction is stated rather than hidden.
    ///
    /// # Errors
    ///
    /// [`PlanError`] when no such model is registered, or when the model
    /// cannot derive an instance key for this action and these parameters.
    /// A model that IS registered and CAN derive a key but cannot fit
    /// anywhere is not an error: that is a successful plan whose `refusal`
    /// says so, which is a strictly more useful answer than a failure.
    pub fn plan(&self, model: &str, action: &str, inv: &Invocation) -> Result<RunPlan, PlanError> {
        let key = self
            .instance_key_for(model, action, inv)
            .ok_or_else(|| PlanError::UnknownModel { model: model.to_string() })?;
        let exclude: crate::runplan::Excluded = HashSet::new();

        // A multi-device model is planned through its own cost shape - it
        // names several devices at once, so `pick_device`'s single-device
        // answer would be meaningless for it.
        if let Some(m) = self.multi_models.get(model) {
            let cost = m.estimate_multi(&key);
            let devices: Vec<Device> = cost.devices().collect();
            // `MultiDeviceCost` reports accelerator bytes per device plus
            // one host figure, so the aggregate is the sum of the per-device
            // shares and the single `ram()` - never a per-device ram that
            // does not exist.
            let mut required = MemCost::new(0, cost.ram());
            for device in &devices {
                required.vram += cost.on(*device);
            }
            let resident = self.multi_residents.get(&key).map(|e| Placement {
                // A multi-device instance has no single device; the first
                // in its own declared order is the one a caller means by
                // "where is it", and the rest are reachable from the
                // residency report. Named rather than invented.
                device: e.devices.first().copied().unwrap_or(Device::Cpu),
                tier: Tier::Hot,
            });
            let placeable = resident.is_some()
                || pick_devices(&cost, &self.budgets, &exclude).is_some();
            return Ok(RunPlan {
                model: model.to_string(),
                action: action.to_string(),
                instance_key: key.config.clone(),
                required,
                resident_on: resident.clone(),
                device: devices.first().copied(),
                // Multi-device eviction is per-device inside `claim_multi`
                // and is not projected here rather than being guessed at.
                evict: Vec::new(),
                supported_devices: devices.clone(),
                estimated_transfer: if resident.is_some() { 0 } else { required.vram },
                refusal: if placeable || devices.is_empty() {
                    None
                } else {
                    Some(format!(
                        "'{model}' needs {} devices this host cannot currently supply together",
                        cost.devices().count()
                    ))
                },
            });
        }

        let m = self
            .models
            .get(model)
            .ok_or_else(|| PlanError::UnknownModel { model: model.to_string() })?;
        let cost = m.estimate(&key);
        let supported = crate::runplan::supported_devices(&cost, &self.budgets);

        if let Some(entry) = self.residents.get(&key) {
            return Ok(RunPlan {
                model: model.to_string(),
                action: action.to_string(),
                instance_key: key.config.clone(),
                required: cost,
                resident_on: Some(Placement { device: entry.device, tier: entry.tier }),
                device: Some(entry.device),
                evict: Vec::new(),
                supported_devices: supported,
                // Already there: nothing crosses a bus to start this job.
                estimated_transfer: 0,
                refusal: None,
            });
        }

        // Fits as things stand: no eviction, no disruption.
        if let Some(device) = pick_device(&cost, &self.budgets, &exclude) {
            return Ok(RunPlan {
                model: model.to_string(),
                action: action.to_string(),
                instance_key: key.config.clone(),
                required: cost,
                resident_on: None,
                device: Some(device),
                evict: Vec::new(),
                supported_devices: supported,
                estimated_transfer: crate::runplan::transfer_for(&cost, Some(device)),
                refusal: None,
            });
        }

        // Does not fit as things stand. What would it cost to make it fit?
        // The victims come from the SAME policy `claim` would apply, in the
        // same order, so the answer is a prediction of what will actually
        // happen rather than a plausible-looking one.
        if let Some(plan) = plan_eviction_with(
            &*self.eviction,
            &cost,
            &self.budgets,
            &self.residents,
            std::slice::from_ref(&key),
            &exclude,
        ) {
            let evict = plan
                .victims
                .iter()
                .map(|victim| Eviction {
                    model: crate::runplan::model_of(victim),
                    instance_key: victim.config.clone(),
                    device: plan.device,
                    frees: self
                        .residents
                        .get(victim)
                        .map(|e| match plan.device {
                            Device::Cpu => e.cost.ram,
                            Device::Gpu(_) => e.cost.vram,
                            Device::Npu(_) => e.cost.npu,
                        })
                        .unwrap_or(0),
                })
                .collect();
            return Ok(RunPlan {
                model: model.to_string(),
                action: action.to_string(),
                instance_key: key.config.clone(),
                required: cost,
                resident_on: None,
                device: Some(plan.device),
                evict,
                supported_devices: supported,
                estimated_transfer: crate::runplan::transfer_for(&cost, Some(plan.device)),
                refusal: None,
            });
        }

        // Nowhere to put it, and nothing that could be moved out of the way.
        // The refusal names the reason a caller can act on: too big for any
        // device this host has, versus every device being occupied by
        // something unevictable.
        let refusal = if could_ever_fit(&cost, &self.budgets) {
            format!(
                "'{model}' needs {} device bytes, and every device large enough is fully \
                 occupied by instances that cannot be evicted right now",
                cost.vram.max(cost.ram)
            )
        } else {
            format!(
                "'{model}' needs {} device bytes, more than any device on this host has",
                cost.vram.max(cost.ram)
            )
        };
        Ok(RunPlan {
            model: model.to_string(),
            action: action.to_string(),
            instance_key: key.config.clone(),
            required: cost,
            resident_on: None,
            device: None,
            evict: Vec::new(),
            supported_devices: supported,
            estimated_transfer: 0,
            refusal: Some(refusal),
        })
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
        if pick_device(&cost, &self.budgets, exclude).is_some()
            || plan_eviction_with(&*self.eviction, &cost, &self.budgets, &self.residents, std::slice::from_ref(key), exclude).is_some()
        {
            return true;
        }
        // Neither fits now nor is evictable into. If it could never fit even on a
        // fully empty device (e.g. `--device npu` excluded the CPU/GPU this model
        // actually needs), returning `false` here would leave it permanently
        // unplaceable: the cost never changes, so every future round would repeat
        // this same "not placeable" verdict and the group would sit in the queue
        // forever with no error and no explanation. Let it through instead so it
        // reaches `claim()`, which turns the same unplaceable cost into a real,
        // per-job `ClaimError::TooLarge` -- a clean failure instead of a silent
        // hang. Mirrors `placeable_multi`'s identical fix (see its own doc).
        !could_ever_fit(&cost, &self.budgets)
    }

    /// [`Self::placeable`]'s multi-device sibling - the executor's scheduling
    /// filter for a [`MultiDeviceResidentModel`] group, mirroring exactly what
    /// [`Self::claim_multi`] can achieve (direct fit OR its own per-device LRU
    /// eviction fallback) WITHOUT mutating anything, the same relationship
    /// `placeable`/`claim` already have.
    ///
    /// Returns `true` - deliberately, though it reads like the wrong answer -
    /// when `estimate_multi` names ZERO devices. An empty cost is that
    /// method's documented "this model is unavailable right now" signal (see
    /// its own doc); filtering the group out HERE would mean it is never
    /// `placeable` on any future round either (the cost won't change), so its
    /// jobs would sit in the queue forever with no error and no explanation.
    /// Returning `true` instead lets the group reach [`Self::claim_multi`],
    /// which turns the same empty cost into a real, per-job
    /// [`ClaimError::Activate`] - a clean failure instead of a silent hang.
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
            if self.budgets.get(d).is_none() {
                return false;
            }
            // Pool-clamped (`usable_on`/`fits_on`), never the raw per-device
            // `Budget` - see `multi::pick_devices` for why.
            if self.budgets.usable_on(d) < need {
                // PERMANENTLY too large for this device. Reporting that as
                // "not placeable right now" is what `placeable`'s own escape
                // hatch exists to avoid: the cost never changes, so every
                // future round repeats the verdict and the group sits in the
                // queue forever with no error and no reply. Let it through to
                // `claim_multi`, which turns the same figure into a real
                // per-job `ClaimError` - a clean failure instead of a hang.
                return true;
            }
            if self.budgets.fits_on(d, need) {
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
    /// the D-Bus `StatsSnapshot`/`StatsStream` surface render from - it is
    /// computed inside the dispatcher thread (which owns the manager) and shipped
    /// out via the [`Executor`](crate::Executor) residency accessor.
    pub fn report(&self) -> ResidencyReport {
        let mut placements: Vec<InstancePlacement> = self
            .residents
            .iter()
            .map(|(k, e)| InstancePlacement { key: k.clone(), device: e.device, tier: e.tier, mem: e.cost.resident_on(e.device) })
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
    /// runs so a concurrent request can't evict it mid-job. (Synchronous path -
    /// deferred builds run inline on this thread.)
    pub fn run(&mut self, model: &str, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (handle, key) = self.claim_built(model, action, inv)?;
        let out = handle.lock().unwrap().run(action, inv, progress);
        self.release(&key);
        out
    }

    /// [`claim`](Self::claim) + build inline when needed - for synchronous callers
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
    /// [`release`](Self::release) after - or, for a deferred build,
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
        // must never be claimed through the single-device path - before this
        // check, the handle lookup below found the shared `instances` entry,
        // then the `residents` expect() panicked ON THE DISPATCHER THREAD,
        // killing scheduling for the whole server.
        if self.multi_residents.contains_key(&key) {
            return Err(ClaimError::Activate(format!("{key}: resident as a multi-device instance - claim it via claim_multi, not claim")));
        }
        // The instance object existing is still the real guard (matches the
        // pre-Warm invariant exactly): a cold build's `residents.insert`
        // pre-accounts the slot before `self.instances` gets the handle
        // (via `adopt`, on the caller's thread) - so `residents` can have a
        // NOT-yet-adopted entry for `key` while a build is in flight, and
        // that in-flight window must keep falling into the cold-build path
        // below (which is itself made a no-op-ish re-place by the budget
        // already being charged), never this branch.
        if let Some(handle) = self.instances.get(&key).cloned() {
            // Never expect() here: this runs on the dispatcher thread, where a
            // panic kills scheduling for every model. A handle with no
            // residency entry is a registry-wiring bug - fail the one claim.
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
            // needed - the entry itself is never a candidate victim of its
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
                    let plan = plan_eviction_with(&*self.eviction, &hot_cost, &self.budgets, &self.residents, std::slice::from_ref(&key), exclude);
                    match self.accelerator_before_host(plan, &hot_cost, exclude) {
                        Some(d) => d,
                        None => {
                            return Err(ClaimError::NoCapacity(format!(
                                "{key} ({} MiB) has no room right now - nothing currently evictable frees enough",
                                hot_cost.vram.max(hot_cost.ram).max(hot_cost.npu) >> 20
                            )))
                        }
                    }
                }
            };
            self.budgets.release(entry.device, entry.cost.resident_on(entry.device));
            self.budgets.alloc(device, hot_cost.resident_on(device));
            self.residents.retier(&key, hot_cost, device, Tier::Hot);
            self.residents.set_pinned(&key, true);
            self.event(format!("promote {key} -> {device:?} (warm->hot)"));
            return Ok((Claimed::Promote(handle), device, key));
        }
        // Cold: place + pre-account + pin NOW (so nothing steals the budget or
        // evicts the slot), but defer the potentially slow/hanging activate() to
        // the caller's thread.
        //
        // A key can be in `residents` WITHOUT being in `instances` while such
        // a deferred build is in flight (insert happens here, adopt happens on
        // the lane). Re-claiming it then would double-charge the budget and
        // overwrite the LRU entry without releasing the old cost - previously
        // documented as a "no-op-ish re-place" (it wasn't) and unreachable
        // only because the Executor's `running` set happens to serialize
        // same-key groups. The manager now enforces its own invariant.
        if self.residents.get(&key).is_some() {
            return Err(ClaimError::NoCapacity(format!(
                "{key}: a deferred build for this key is already in flight - retry when it adopts or fails"
            )));
        }
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
                let plan = plan_eviction_with(&*self.eviction, &cost, &self.budgets, &self.residents, std::slice::from_ref(&key), exclude);
                match self.accelerator_before_host(plan, &cost, exclude) {
                    Some(d) => d,
                    None => {
                        return Err(ClaimError::NoCapacity(format!(
                            "{key} ({} MiB) has no room right now - nothing currently evictable frees enough",
                            cost.vram.max(cost.ram).max(cost.npu) >> 20
                        )))
                    }
                }
            }
        };
        self.budgets.alloc(device, cost.resident_on(device));
        self.residents.insert(key.clone(), cost, device);
        self.residents.set_pinned(&key, true);
        self.event(format!("promote {key} -> {device:?} (building)"));
        Ok((Claimed::Build(m), device, key))
    }

    /// A deferred build succeeded: adopt the instance so later claims find it hot.
    pub fn adopt(&mut self, key: &InstanceKey, handle: InstanceHandle) -> InstanceHandle {
        self.instances.insert(key.clone(), handle.clone());
        self.builds += 1;
        self.event(format!("built {key}"));
        handle
    }

    /// A deferred build failed: unwind the pre-accounted budget + resident slot.
    /// The claim is over - do NOT also call [`release`](Self::release).
    pub fn build_failed(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.residents.remove(key) {
            self.budgets.release(entry.device, entry.cost.resident_on(entry.device));
        }
        self.instances.remove(key);
        self.event(format!("build-failed {key}"));
    }

    /// Unpin an instance after a run and mark it most-recently-used.
    pub fn release(&mut self, key: &InstanceKey) {
        self.residents.set_pinned(key, false);
        // Recency only: `claim` already counted this job's use (see
        // `Residents::touch_recency`).
        self.residents.touch_recency(key);
    }

    /// [`Self::claim`]'s multi-device sibling - places (or finds hot) an
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
            return Err(ClaimError::Activate(format!("{key}: resident as a single-device instance - claim it via claim, not claim_multi")));
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
                // specifically to make room - never touching another
                // multi-device resident (this manager's own module doc names
                // that as the deliberate scope limit). Reuses the existing
                // single-device eviction planner per device by excluding
                // every other device, so it cannot "succeed" by picking a
                // different one than the one actually needed.
                // TWO passes, and the split is the whole point. The first pass
                // only PLANS - it mutates nothing - so a device that turns out
                // to be impossible is discovered before any resident has been
                // destroyed for it. This loop used to evict device by device
                // as it went, which meant a claim wanting `[(gpu0, 15 GiB),
                // (gpu1, 30 GiB)]` on 24 GiB cards destroyed gpu0's 20 GiB
                // resident and only THEN discovered gpu1 could never hold its
                // share - and the executor's retry then did it again to
                // whatever had taken its place. An eviction thrash loop that
                // got worse on every retry, and the exact inverse of the
                // invariant `too_large_for_any_device_fails_cleanly_without_
                // evicting_anything` pins for the single-device path.
                //
                // Planning per device is exact rather than merely
                // conservative: `only_d` restricts each plan to victims on
                // that ONE device, so no two devices' plans can name the same
                // victim and the plans cannot interfere.
                let every_device: HashSet<Device> = self.budgets.devices().collect();
                let mut victims: Vec<InstanceKey> = Vec::new();
                for &d in &wanted {
                    if exclude.contains(&d) {
                        return Err(ClaimError::NoCapacity(format!("{key}: device {d:?} is excluded")));
                    }
                    let need = cost.on(d);
                    if self.budgets.get(d).is_none() {
                        return Err(ClaimError::TooLarge(format!("{key}: device {d:?} has no budget")));
                    }
                    // Pool-clamped, so a unified-memory box cannot book the
                    // same physical bytes twice (see `multi::pick_devices`).
                    if self.budgets.usable_on(d) < need {
                        // PERMANENT: no eviction, however aggressive, can make
                        // this device hold this share. `TooLarge`, not
                        // `NoCapacity` - the executor retries the latter
                        // forever, and `placeable_multi` deliberately lets a
                        // group through to here precisely so it gets a real,
                        // final error instead of sitting in the queue.
                        return Err(ClaimError::TooLarge(format!(
                            "{key} ({} MiB on {d:?}) is too large for that device's usable budget even fully empty",
                            need >> 20
                        )));
                    }
                    if self.budgets.fits_on(d, need) {
                        continue; // already fits on this device, nothing to evict here
                    }
                    let mut only_d = every_device.clone();
                    only_d.remove(&d);
                    let plan = plan_eviction_with(&*self.eviction, &synth_cost_for(d, need), &self.budgets, &self.residents, &[], &only_d)
                        .ok_or_else(|| ClaimError::NoCapacity(format!("{key}: cannot free {} MiB on {d:?}", need >> 20)))?;
                    victims.extend(plan.victims);
                }
                // Every device is satisfiable; only now destroy anything.
                for victim in &victims {
                    self.evict_entry(victim);
                }
                pick_devices(&cost, &self.budgets, exclude)
                    .ok_or_else(|| ClaimError::NoCapacity(format!("{key}: does not fit even after eviction")))?
            }
        };
        // The host bytes are a real charge and must FIT before anything is
        // reserved - checked here rather than inside `charge_multi_host` so a
        // refusal happens before the per-device budgets are touched.
        if cost.ram() > 0 && !devices.contains(&Device::Cpu) && self.budgets.get(Device::Cpu).is_some() && !self.budgets.fits_on(Device::Cpu, cost.ram()) {
            return Err(ClaimError::NoCapacity(format!("{key}: {} MiB of host RAM does not fit right now", cost.ram() >> 20)));
        }
        for &d in &devices {
            self.budgets.alloc(d, cost.on(d));
        }
        self.charge_multi_host(&cost, &devices);
        self.multi_residents.insert(key.clone(), MultiEntry { cost, devices: devices.clone(), pinned: true });
        self.event(format!("promote {key} -> {devices:?} (building, multi-device)"));
        Ok((ClaimedMulti::Build(m), devices, key))
    }

    /// [`Self::adopt`]'s multi-device sibling.
    pub fn adopt_multi(&mut self, key: &InstanceKey, handle: InstanceHandle) -> InstanceHandle {
        self.instances.insert(key.clone(), handle.clone());
        self.builds += 1;
        self.event(format!("built {key} (multi-device)"));
        handle
    }

    /// [`Self::build_failed`]'s multi-device sibling: unwinds the
    /// pre-accounted budget on EVERY device this claim reserved.
    pub fn build_failed_multi(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.multi_residents.remove(key) {
            for &d in &entry.devices {
                self.budgets.release(d, entry.cost.on(d));
            }
            self.refund_multi_host(&entry.cost, &entry.devices);
        }
        self.instances.remove(key);
        self.event(format!("build-failed {key} (multi-device)"));
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
    /// doc explains why), so a caller that genuinely wants one gone -
    /// swapping to a different checkpoint, a shutdown path - calls this
    /// directly.
    ///
    /// Refuses (returns `false`, evicts nothing) while the instance is
    /// PINNED - a lane is actively running a job against it (`claim_multi`
    /// pins on every claim; `release_multi` unpins after). Evicting out from
    /// under a running job would drop the `Instance` (freeing its GPU memory)
    /// while a lane still holds a strong `Arc` to it and is mid-call - the
    /// budget would say the memory is free while the lane is still using it,
    /// exactly the kind of lying figure this crate's `multi` module exists to
    /// avoid. Returns `true` if an unpinned entry was found and evicted;
    /// `true` is also NOT returned for a key that was never resident (no-op,
    /// same as `false`) - check separately if the caller needs to
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
                self.refund_multi_host(&entry.cost, &entry.devices);
                self.instances.remove(key);
                self.evictions += 1;
                self.event(format!("evict {key} <- {:?} (multi-device)", entry.devices));
                true
            }
        }
    }

    /// Run several same-key invocations of one action on a single hot instance -
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

    /// Public single-device sibling of [`Self::evict_multi`] - same
    /// pinned-refusal contract: refuses (returns `false`, evicts nothing)
    /// while a job is actively running against `key`, or if it isn't
    /// resident at all. For a caller that genuinely wants a specific
    /// instance gone right now - e.g. swapping in a newly-trained LoRA
    /// adapter (self-improve roadmap P4/P5's continuous-training hot-swap:
    /// bump the served model-card's adapter path, then call this so the
    /// NEXT claim rebuilds against it instead of reusing the stale Hot/Warm
    /// instance) - as opposed to [`Self::evict_entry`] below, the automatic
    /// make-room-for-something-else path, whose only caller already
    /// excludes pinned candidates upstream (`Residents::lru_on`) and so
    /// never needed its own check.
    pub fn evict(&mut self, key: &InstanceKey) -> bool {
        match self.residents.get(key) {
            None => false,
            Some(e) if e.pinned => false,
            Some(_) => {
                self.evict_entry(key);
                true
            }
        }
    }

    /// Demote (drop) an instance, freeing its device memory.
    /// Free `key`'s device slot to make room for something else. Tries a
    /// soft demotion to `Tier::Warm` first (releases the device buffers,
    /// keeps the `Instance` and its host bytes alive, so a later claim for
    /// the same key can [`Instance::promote`] straight back instead of
    /// rebuilding from the checkpoint) - this is the entire "made real"
    /// part of `Tier::Warm`: every existing caller of `evict_entry` gets it
    /// for free, with zero behaviour change for a model that hasn't opted
    /// in, since `demote` defaults to `Err` and this falls straight through
    /// to the original full-drop below whenever it does.
    ///
    /// Not pinned-checked: its only caller (the eviction-planner path above)
    /// already excludes pinned candidates before this ever runs - see
    /// [`Self::evict`] for the pinned-safe public entry point.
    /// Re-size each accelerator's budget from a LIVE measurement of how much
    /// of it is free, so a neighbouring process taking or releasing VRAM
    /// changes what this scheduler believes it may use.
    ///
    /// `free` is `(device, bytes free right now)` as the driver reports it -
    /// which already excludes both this process's own allocations and every
    /// other process's. The budget's `total` is therefore set to
    /// `used + free`: what we hold plus what is genuinely still available. Not
    /// `free` alone, which would double-count our own residents (they are
    /// already charged in `used` AND absent from `free`) and evict them one by
    /// one on every refresh.
    ///
    /// The `reserved` headroom is preserved: it is a policy figure
    /// (`--reserve-gb`), not a measurement.
    ///
    /// A device the probe does not report is left exactly as it is - a probe
    /// that cannot see a card must not silently zero it. Likewise the host
    /// tier, which this never touches: `Device::Cpu`'s budget is host RAM, a
    /// different measurement with a different meaning.
    ///
    /// The budget can legitimately shrink BELOW what is already charged, when
    /// a neighbour takes bytes we were counting on. That is not an error to
    /// hide: `Budget::free` reports 0, nothing new is placed, and the eviction
    /// planner starts reclaiming - which is the correct response to the card
    /// having been taken out from under us.
    pub fn refresh_accelerator_capacity(&mut self, free: &[(Device, u64)]) {
        for &(d, free_now) in free {
            if matches!(d, Device::Cpu) {
                continue;
            }
            let Some(b) = self.budgets.get_mut(d) else { continue };
            let want = b.used.saturating_add(free_now);
            if want != b.total {
                b.total = want;
                b.reserved = b.reserved.min(b.total);
            }
        }
    }

    /// Execute an eviction plan, but try a CARD before accepting the host
    /// tier.
    ///
    /// `plan_eviction_with` offers the host tier as its last class, which is
    /// what makes "never fail" true. It is still the slowest possible answer,
    /// so before taking it, try the one victim class that planner cannot see:
    /// an unpinned multi-device resident sitting on a card
    /// ([`Self::evict_multi_to_fit`]). Order of preference, fastest first:
    /// single-device eviction on an accelerator, then multi-device eviction on
    /// an accelerator, then the host tier.
    fn accelerator_before_host(&mut self, plan: Option<crate::place::EvictionPlan>, cost: &MemCost, exclude: &HashSet<Device>) -> Option<Device> {
        if let Some(p) = &plan {
            if p.device != Device::Cpu {
                let victims = p.victims.clone();
                let device = p.device;
                for victim in &victims {
                    self.evict_entry(victim);
                }
                return Some(device);
            }
        }
        if let Some(d) = self.evict_multi_to_fit(cost, exclude) {
            return Some(d);
        }
        let p = plan?;
        for victim in &p.victims {
            self.evict_entry(victim);
        }
        Some(p.device)
    }

    /// Free a CARD for a single-device claim by dropping unpinned
    /// multi-device residents that occupy it.
    ///
    /// `plan_eviction_with` reads `self.residents` only; `multi_residents` is
    /// a separate map it has never seen. So once a large multi-device model
    /// was resident, every later single-device claim that needed its card got
    /// `NoCapacity` - forever, with no automatic recovery, because the only
    /// thing that could release it was an explicit external
    /// `Executor::evict_multi` call that nothing on the serving path makes.
    ///
    /// Plans before it destroys anything, same discipline as `claim_multi`.
    /// Victims are taken cheapest-first (smallest bytes on that card), which
    /// is the closest thing to a policy available here: `MultiEntry` carries
    /// no recency or use count, so `EvictionPolicy` cannot score it. Taking
    /// the least is at least a bounded, explicable loss. Wiring
    /// multi-residents into `Residents` so the real policy applies to them is
    /// the proper fix and is deliberately not attempted here.
    fn evict_multi_to_fit(&mut self, cost: &MemCost, exclude: &HashSet<Device>) -> Option<Device> {
        if cost.vram == 0 {
            return None;
        }
        let mut cards: Vec<Device> = self.budgets.gpus().into_iter().filter(|d| !exclude.contains(d)).collect();
        // Emptiest card first: fewest victims to reach the target.
        cards.sort_by_key(|&d| (std::cmp::Reverse(self.budgets.free_on(d)), format!("{d:?}")));
        for d in cards {
            if self.budgets.usable_on(d) < cost.vram {
                continue; // can never fit here, even emptied
            }
            let mut victims: Vec<(InstanceKey, u64)> = self
                .multi_residents
                .iter()
                .filter(|(_, e)| !e.pinned && e.devices.contains(&d))
                .map(|(k, e)| (k.clone(), e.cost.on(d)))
                .collect();
            victims.sort_by_key(|(k, bytes)| (*bytes, format!("{k}")));
            // Dry run: how far down the list do we have to go, and is it enough?
            let mut freed = 0u64;
            let mut take = 0usize;
            let deficit = cost.vram.saturating_sub(self.budgets.free_on(d));
            for (_, bytes) in &victims {
                if freed >= deficit {
                    break;
                }
                freed += bytes;
                take += 1;
            }
            if freed < deficit {
                continue; // this card cannot be freed enough; destroy nothing
            }
            for (k, _) in victims.into_iter().take(take) {
                self.evict_multi(&k);
            }
            if self.budgets.fits_on(d, cost.vram) {
                return Some(d);
            }
        }
        None
    }

    /// Host RAM a multi-device instance holds "regardless of accelerator
    /// placement" - `MultiDeviceCost::ram`'s own words. It was declared by
    /// every multi-device model in the repo and charged by nothing: the field
    /// had no `alloc` call site anywhere, so a model staging GiB-scale bytes
    /// on the host was invisible to the host budget and an unbounded number of
    /// them could be admitted.
    ///
    /// Skipped when the cost already names `Device::Cpu` explicitly - that
    /// entry is charged by the per-device loop, and charging both would
    /// double-count one model's own host bytes.
    fn charge_multi_host(&mut self, cost: &crate::multi::MultiDeviceCost, devices: &[Device]) {
        let ram = cost.ram();
        if ram > 0 && !devices.contains(&Device::Cpu) && self.budgets.get(Device::Cpu).is_some() {
            self.budgets.alloc(Device::Cpu, ram);
        }
    }

    /// The exact inverse of [`Self::charge_multi_host`] - same condition, so
    /// the two cannot drift and leave the host budget permanently short.
    fn refund_multi_host(&mut self, cost: &crate::multi::MultiDeviceCost, devices: &[Device]) {
        let ram = cost.ram();
        if ram > 0 && !devices.contains(&Device::Cpu) && self.budgets.get(Device::Cpu).is_some() {
            self.budgets.release(Device::Cpu, ram);
        }
    }

    fn evict_entry(&mut self, key: &InstanceKey) {
        if let Some(entry) = self.residents.get(key).copied() {
            if entry.tier == Tier::Hot {
                if let (Some(handle), Some(model)) = (self.instances.get(key).cloned(), self.models.get(&key.model).cloned()) {
                    let warm_cost = model.estimate_at(key, Tier::Warm);
                    // The Warm copy is a real host-RAM charge - it must FIT
                    // (pool-aware: on a unified-memory box the HOST_POOL is
                    // the same physical bytes the accelerators use). Checked
                    // BEFORE `demote()` releases anything: repeated multi-GB
                    // demotions that nothing refuses are exactly the swap
                    // cliff memauth's doc warns about. When it doesn't fit,
                    // fall through to the full drop below - freeing the
                    // device slot is the caller's actual requirement; keeping
                    // a warm copy is only an optimization. (Conservative for
                    // a CPU-Hot instance, whose own Hot bytes are not counted
                    // as freed here; demoting CPU→CPU-warm is not a shape any
                    // current caller produces.)
                    if self.budgets.fits_on(Device::Cpu, warm_cost.resident_on(Device::Cpu)) && handle.lock().unwrap().demote(Tier::Warm).is_ok() {
                        self.budgets.release(entry.device, entry.cost.resident_on(entry.device));
                        self.budgets.alloc(Device::Cpu, warm_cost.resident_on(Device::Cpu));
                        self.residents.retier(key, warm_cost, Device::Cpu, Tier::Warm);
                        self.event(format!("demote {key} <- {:?} (warm)", entry.device));
                        return;
                    }
                }
            }
        }
        // Full drop: today's only behaviour, and the fallback whenever
        // `demote` isn't supported, the entry was already below Hot, or the
        // model/instance lookup above came up empty.
        if let Some(entry) = self.residents.remove(key) {
            self.budgets.release(entry.device, entry.cost.resident_on(entry.device));
            self.instances.remove(key); // drops the Instance → frees the GPU
            self.evictions += 1;
            self.event(format!("evict {key} <- {:?}", entry.device));
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

    /// The whole point of `plan`: the prediction it makes is the decision
    /// `claim` then actually takes, victim for victim. If these two could
    /// disagree, a caller showing "starting this will evict Qwen3 14B"
    /// would be lying, which is worse than saying nothing.
    #[test]
    fn a_plan_predicts_exactly_the_eviction_that_then_happens() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB);
        let mut mgr = ResidencyManager::new(budgets);
        for n in ["a", "b", "c"] {
            mgr.register(Arc::new(Fake { name: n.into(), vram: 10 * GB, live: live.clone() }));
        }

        // Nothing loaded: `a` fits outright and disturbs nobody.
        let plan = mgr.plan("a", "run", &Invocation::new()).expect("a is registered");
        assert!(plan.runnable());
        assert!(!plan.disturbs_residents(), "an empty card evicts nothing");
        assert_eq!(plan.device, Some(Device::Gpu(0)));
        assert_eq!(plan.required.vram, 10 * GB);
        assert_eq!(plan.estimated_transfer, 10 * GB, "10 GB has to cross the bus");
        assert_eq!(plan.resident_on, None);

        mgr.run("a", "run", &Invocation::new(), &mut |_| {}).unwrap();
        mgr.run("b", "run", &Invocation::new(), &mut |_| {}).unwrap();

        // Already resident: no load, no transfer, no disruption.
        let plan = mgr.plan("a", "run", &Invocation::new()).unwrap();
        assert_eq!(
            plan.resident_on.as_ref().map(|p| p.device),
            Some(Device::Gpu(0)),
            "a is hot, and the plan says where"
        );
        assert_eq!(plan.estimated_transfer, 0, "nothing crosses a bus for a hot instance");

        // 20 of 22 GB used. `c` needs 10, so something must go -- and the
        // plan names WHICH, before anything happens.
        let plan = mgr.plan("c", "run", &Invocation::new()).unwrap();
        assert!(plan.runnable(), "it can run, at a price");
        assert!(plan.disturbs_residents());
        let predicted: Vec<&str> = plan.evict.iter().map(|e| e.model.as_str()).collect();
        assert_eq!(predicted, vec!["a"], "LRU picks a, and the plan says so");
        assert_eq!(plan.evict[0].frees, 10 * GB);
        assert_eq!(plan.evict[0].device, Device::Gpu(0));

        // Now actually run it, and check the prediction held.
        mgr.run("c", "run", &Invocation::new(), &mut |_| {}).unwrap();
        let hot: Vec<String> = mgr.residency().into_iter().map(|(k, _, _)| k.model).collect();
        assert!(!hot.contains(&"a".to_string()), "the predicted victim is the real one");
        assert!(hot.contains(&"c".to_string()) && hot.contains(&"b".to_string()));
    }

    /// A model too big for any device is a successful plan that says so,
    /// not an error -- "it cannot run here" is exactly the answer a caller
    /// asked for, and a caller that gets an error instead has to guess
    /// whether the model is missing or merely too large.
    #[test]
    fn a_model_that_cannot_fit_anywhere_plans_a_refusal_rather_than_erroring() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 8 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(Fake { name: "huge".into(), vram: 40 * GB, live }));

        let plan = mgr.plan("huge", "run", &Invocation::new()).expect("it IS registered");
        assert!(!plan.runnable());
        assert_eq!(plan.device, None);
        assert!(plan.evict.is_empty(), "nothing to evict when nothing would help");
        assert!(
            plan.supported_devices.is_empty(),
            "no device on this host could ever hold it"
        );
        let refusal = plan.refusal.as_deref().expect("just asserted not runnable");
        assert!(
            refusal.contains("more than any device on this host has"),
            "the refusal must distinguish too-big from merely-occupied: {refusal}"
        );
    }

    /// An unregistered model is the one genuine error: there is no estimate
    /// to make, so there is nothing honest to say about it.
    #[test]
    fn planning_an_unregistered_model_is_an_error_not_an_empty_plan() {
        let mgr = ResidencyManager::new(Budgets::new());
        let err = mgr.plan("nope", "run", &Invocation::new()).unwrap_err();
        assert_eq!(err, PlanError::UnknownModel { model: "nope".to_string() });
        assert!(err.to_string().contains("no model 'nope' is registered"));
    }

    /// The JSON a remote caller sees carries the answer, the reasoning and
    /// the device names brain's own `--device` flag accepts -- so a caller
    /// can hand a plan's answer straight back as a request.
    #[test]
    fn the_wire_shape_carries_the_reasoning_and_reusable_device_names() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(1), 24 * GB, 2 * GB);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(Fake { name: "m".into(), vram: 10 * GB, live }));

        let v = mgr.plan("m", "run", &Invocation::new()).unwrap().to_json();
        assert_eq!(v["model"], "m");
        assert_eq!(v["device"], "gpu1", "the same spelling --device accepts");
        assert_eq!(v["required"]["vram"], 10 * GB);
        assert_eq!(v["estimated_transfer"], 10 * GB);
        assert_eq!(v["runnable"], true);
        assert_eq!(v["evict"], serde_json::json!([]));
        assert_eq!(v["supported_devices"], serde_json::json!(["gpu1"]));
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

        // Run a, then b - both fit (20 GB <= 22).
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
    /// subtracts) - the two diverging is exactly the property a Warm
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
    /// (`live` unchanged, `hot` drops) - and a later claim for the same key
    /// promotes it straight back (`hot` rises again, `live` STILL
    /// unchanged: no second `activate()`/checkpoint reload ever happened).
    /// A model that hasn't opted in (`Fake`/`FakeInst`) keeps dropping on
    /// eviction exactly as `three_models_on_one_gpu_swap_by_lru` already
    /// proves - this test is the other half.
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
    /// to a full drop - never an unrefused overcommit (the swap cliff on
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
    /// (clean ClaimError) when claimed through the other - both directions.
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
    /// its residency - no eviction plan is even attempted, because none
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

    /// `--device npu` on a model with no NPU path (`MemCost.npu == 0`, e.g. an
    /// LLM): the job can never be placed (no NPU-eligible cost, CPU/GPU excluded
    /// from compute), so it must fail fast with `ClaimError::TooLarge` -- not
    /// hang in the queue forever. Regression for `placeable()` lacking the
    /// `could_ever_fit` escape hatch `placeable_multi()` already had: without
    /// it, `Executor::assign` never calls `claim()` for this group (it never
    /// appears in `placeable`), so no error is ever produced and the caller
    /// just times out with a generic 429 once the HTTP layer's own admission
    /// deadline expires.
    #[test]
    fn placeable_lets_a_permanently_unplaceable_job_through_so_claim_can_fail_it_cleanly() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Npu(0), 8 * GB, 0); // only an NPU is schedulable ...
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(Fake { name: "llm".into(), vram: 4 * GB, live })); // ... but this model has no NPU path (npu cost == 0)

        let key = InstanceKey::new("llm", "default");
        assert!(
            mgr.placeable(&key, "llm", &no_exclude()),
            "must let the group through to claim() rather than hang it in the queue forever"
        );

        let err = mgr.claim("llm", "run", &Invocation::new(), &no_exclude()).err().expect("must be refused");
        assert!(matches!(err, ClaimError::TooLarge(_)), "expected TooLarge, got {err:?}");
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

    /// A fake model that spans BOTH gpu0 and gpu1 at once - occupies real,
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
            crate::multi::MultiDeviceCost::new(vec![(Device::Gpu(0), GB), (Device::Gpu(1), 30 * GB)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }

    /// **Rollback.** `claim_multi` used to evict device by device as it went,
    /// so a claim wanting `[(gpu0, 1 GB), (gpu1, 30 GB)]` destroyed gpu0's
    /// resident and only THEN discovered gpu1 could never hold its share. The
    /// executor retries `NoCapacity`, so every retry destroyed more - an
    /// eviction thrash loop that got worse, not better. Nothing may be
    /// evicted for a claim that cannot succeed.
    #[test]
    fn a_multi_device_claim_that_cannot_succeed_evicts_nothing_on_the_way() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        // A single-device resident filling gpu0 - the victim the old code took.
        mgr.register(Arc::new(Fake { name: "victim".into(), vram: 23 * GB, live: live.clone() }));
        mgr.run("victim", "run", &Invocation::new(), &mut |_| {}).expect("victim resident");
        assert_eq!(mgr.resident_count(), 1);

        // ...and a multi model whose gpu0 share NEEDS that resident evicted
        // (10 GB against 1 GB free) while its gpu1 share can never fit a
        // 24 GB card at all. The old loop planned and evicted gpu0 first, then
        // discovered gpu1, and returned with the victim already destroyed.
        mgr.register_multi(Arc::new(MultiFakeRollback { live: live.clone() }));
        let err = match claim_multi_built(&mut mgr, "rollback") {
            Ok(_) => panic!("gpu1's 30 GB share can never fit a 24 GB card"),
            Err(e) => e,
        };
        assert!(err.contains("too large"), "{err}");
        assert_eq!(mgr.resident_count(), 1, "the impossible claim must not have destroyed the resident");
    }

    /// **Multi-device residents are eviction victims too.** `plan_eviction_with`
    /// reads `residents` only; `multi_residents` is a separate map it has never
    /// seen. So once a large multi-device model was resident, every later
    /// single-device claim needing its card got `NoCapacity` forever, with no
    /// automatic recovery - the only thing that could free it was an explicit
    /// external `evict_multi` call the serving path never makes.
    #[test]
    fn a_resident_multi_device_model_can_be_evicted_for_a_single_device_claim() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register_multi(Arc::new(MultiFake { name: "spanner".into(), per_gpu: 20 * GB, live: live.clone() }));
        let (_, mkey) = claim_multi_built(&mut mgr, "spanner").expect("multi resident");
        mgr.release_multi(&mkey); // unpinned: a lane finished with it
        assert_eq!(mgr.resident_multi_count(), 1);

        // 16 GB fits neither card while the spanner holds 20 GB of each, and
        // there is no single-device resident anywhere to evict.
        mgr.register(Arc::new(Fake { name: "later".into(), vram: 16 * GB, live: live.clone() }));
        mgr.run("later", "run", &Invocation::new(), &mut |_| {}).expect("the spanner must be evictable");
        assert_eq!(mgr.resident_multi_count(), 0, "the multi-device resident was the only thing in the way");
        let where_ = mgr.residents.get(&InstanceKey::new("later", "default")).map(|e| e.device);
        assert!(matches!(where_, Some(Device::Gpu(_))), "the claim must get a CARD, not the host tier: {where_:?}");
    }

    /// ...but a PINNED one is never taken out from under a running lane.
    #[test]
    fn a_pinned_multi_device_resident_is_never_evicted() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        budgets.set(Device::Cpu, 128 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register_multi(Arc::new(MultiFake { name: "spanner".into(), per_gpu: 20 * GB, live: live.clone() }));
        claim_multi_built(&mut mgr, "spanner").expect("multi resident"); // still pinned
        mgr.register(Arc::new(Fake { name: "later".into(), vram: 16 * GB, live: live.clone() }));
        mgr.run("later", "run", &Invocation::new(), &mut |_| {}).expect("the host tier absorbs it instead");
        assert_eq!(mgr.resident_multi_count(), 1, "a pinned resident must survive");
        assert_eq!(mgr.residents.get(&InstanceKey::new("later", "default")).map(|e| e.device), Some(Device::Cpu), "slower, but it ran");
    }

    /// **Host RAM declared by a multi-device model is charged.**
    /// `MultiDeviceCost::ram` is documented as bytes held "regardless of
    /// accelerator placement" and had no `alloc` call site anywhere, so an
    /// unbounded number of GiB-scale host stagings could be admitted against a
    /// host budget that never moved.
    #[test]
    fn a_multi_device_models_host_ram_is_charged_and_refunded() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0).set(Device::Gpu(1), 24 * GB, 0);
        budgets.set(Device::Cpu, 64 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register_multi(Arc::new(MultiFakeHost { live: live.clone() }));
        let (_, key) = claim_multi_built(&mut mgr, "hosty").expect("resident");
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().used, 10 * GB, "the declared host bytes must be charged");
        mgr.release_multi(&key);
        assert!(mgr.evict_multi(&key));
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().used, 0, "and released again, exactly");
    }

    /// gpu0's share needs an eviction to fit; gpu1's can never fit at all.
    struct MultiFakeRollback {
        live: Arc<AtomicU32>,
    }
    impl ResidentModel for MultiFakeRollback {
        fn manifest(&self) -> Manifest {
            Manifest::new("rollback", "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("rollback", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            Err("not this model's contract".to_string())
        }
    }
    impl crate::multi::MultiDeviceResidentModel for MultiFakeRollback {
        fn estimate_multi(&self, _k: &InstanceKey) -> crate::multi::MultiDeviceCost {
            crate::multi::MultiDeviceCost::new(vec![(Device::Gpu(0), 10 * GB), (Device::Gpu(1), 30 * GB)], 0)
        }
        fn activate_multi(&self, _k: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }

    struct MultiFakeHost {
        live: Arc<AtomicU32>,
    }
    impl ResidentModel for MultiFakeHost {
        fn manifest(&self) -> Manifest {
            Manifest::new("hosty", "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new("hosty", "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            Err("not this model's contract".to_string())
        }
    }
    impl crate::multi::MultiDeviceResidentModel for MultiFakeHost {
        fn estimate_multi(&self, _k: &InstanceKey) -> crate::multi::MultiDeviceCost {
            crate::multi::MultiDeviceCost::new(vec![(Device::Gpu(0), GB), (Device::Gpu(1), GB)], 10 * GB)
        }
        fn activate_multi(&self, _k: &InstanceKey, _devices: &[Device]) -> Result<Box<dyn crate::Instance>, String> {
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { live: self.live.clone() }))
        }
    }

    /// **The daemon adapts to VRAM it does not own.** Budgets used to be frozen
    /// at process start, from the card's TOTAL size, so every byte a
    /// neighbouring process took or released was invisible: the scheduler
    /// placed onto a card somebody else had filled, and never noticed when
    /// they left. `refresh_accelerator_capacity` re-sizes each card from a
    /// live measurement - what we hold plus what is genuinely still free.
    #[test]
    fn a_live_capacity_refresh_tracks_a_neighbouring_process_both_ways() {
        let live = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB);
        budgets.set(Device::Cpu, 128 * GB, 0);
        let mut mgr = ResidencyManager::new(budgets);
        mgr.register(Arc::new(Fake { name: "m".into(), vram: 8 * GB, live: live.clone() }));
        mgr.run("m", "run", &Invocation::new(), &mut |_| {}).expect("8 GB fits an empty card");
        assert_eq!(mgr.budgets.free_on(Device::Gpu(0)), 14 * GB);

        // A neighbour takes 12 GB: 4 GB is really free, and we hold 8.
        mgr.refresh_accelerator_capacity(&[(Device::Gpu(0), 4 * GB)]);
        assert_eq!(mgr.budgets.free_on(Device::Gpu(0)), 2 * GB, "12 used by us+them, 2 GB reserve, 4 GB free -> 2 GB placeable");

        // The neighbour exits. Capacity comes BACK without a restart - this is
        // the "always recover again once vram becomes available again" half.
        mgr.refresh_accelerator_capacity(&[(Device::Gpu(0), 16 * GB)]);
        assert_eq!(mgr.budgets.free_on(Device::Gpu(0)), 14 * GB);

        // A device the probe cannot see is left exactly as it was, and the
        // host tier is never touched by an accelerator probe.
        mgr.refresh_accelerator_capacity(&[(Device::Cpu, GB)]);
        assert_eq!(mgr.budgets.get(Device::Cpu).unwrap().total, 128 * GB);
    }
}
