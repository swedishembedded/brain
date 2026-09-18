// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **"Will this run, and what happens if I start it?"** answered BEFORE
//! anything is loaded, allocated or evicted.
//!
//! Swedish Embedded AB implements resource-aware model placement and
//! admission control for its clients. If your team needs expertise in
//! predicting whether a model fits the hardware you have - and saying so
//! honestly before a job starts rather than after it fails - you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! # The problem this exists to remove
//!
//! The residency manager has always KNOWN the answer. [`crate::ResidencyManager::placeable`]
//! computes exactly this - estimate the instance, try to place it, fall back
//! to planning an eviction - and then throws every detail away and returns a
//! bool. So a caller that wants to tell a user "this needs 21.8 GB, it fits
//! on P40 #1, and starting it will evict Qwen3 14B" had two options: guess,
//! or reimplement brain's memory accounting outside brain. Both are wrong,
//! and the second is worse, because it drifts silently the moment a model's
//! estimate changes.
//!
//! [`RunPlan`] is the same computation with its reasoning kept. Nothing here
//! is new arithmetic; it is the existing decision, made visible.
//!
//! # A plan is a prediction, not a reservation
//!
//! Computing one mutates nothing and reserves nothing. Between planning and
//! starting, another job may claim the device the plan named. That is
//! inherent to any pre-flight answer and is stated rather than papered over:
//! a caller uses a plan to decide and to explain, never to assume. The
//! actual admission decision stays [`crate::ResidencyManager::claim`]'s, which
//! is the only thing that reserves.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::budget::Budgets;
use crate::{Device, InstanceKey, MemCost, Tier};

/// What starting one job would require and disturb.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunPlan {
    /// The model this plan is about.
    pub model: String,
    /// The action, as asked for.
    pub action: String,
    /// The instance fingerprint the model derived from the parameters - two
    /// requests sharing this key share one hot instance, which is why a plan
    /// for a 512x512 image and one for 1024x1024 can differ entirely.
    pub instance_key: String,
    /// What this instance costs while hot: device bytes, host bytes, NPU
    /// bytes and reclaimable mapped bytes, kept separate because a governor
    /// must not treat them as interchangeable.
    pub required: MemCost,
    /// Where it already is, if it is already there. `Some` means starting
    /// this job loads nothing and evicts nothing.
    pub resident_on: Option<Placement>,
    /// Where it would go. `None` with an empty [`Self::evict`] and a
    /// populated [`Self::refusal`] means it cannot go anywhere.
    pub device: Option<Device>,
    /// What would have to be evicted first, in the order the eviction policy
    /// would actually pick them. Empty when the model fits without
    /// disturbing anything - which is the difference between "this will run"
    /// and "this will run, and cost you what is currently loaded".
    pub evict: Vec<Eviction>,
    /// Every device this instance could be placed on if that device were
    /// empty. A caller offering "run somewhere else" needs the set of
    /// somewhere-elses, and it is not the same as the set of devices that
    /// exist: a model with `npu == 0` has no NPU path at all.
    pub supported_devices: Vec<Device>,
    /// Bytes that would move into device memory to activate. Zero when
    /// already resident. This is the transfer a caller is about to pay for,
    /// and it is what makes "it fits" and "it fits quickly" different
    /// answers.
    pub estimated_transfer: u64,
    /// Why this cannot run, when it cannot. `None` means it can - either as
    /// it stands, or after the evictions above.
    pub refusal: Option<String>,
}

impl RunPlan {
    /// Whether this job can start at all, evictions included.
    #[must_use]
    pub fn runnable(&self) -> bool {
        self.refusal.is_none()
    }

    /// Whether starting it would cost the caller something already loaded.
    #[must_use]
    pub fn disturbs_residents(&self) -> bool {
        !self.evict.is_empty()
    }
}

impl RunPlan {
    /// This plan as JSON, for a caller on the other side of a wire.
    ///
    /// Hand-written rather than derived: this crate has `serde_json` but not
    /// `serde`, `MemCost`/`Device`/`Tier` carry no derives of their own, and
    /// the same hand-written convention is what `capability::Manifest`
    /// already uses for every manifest this host serves. One less derive is
    /// also one less way for a field rename to silently change a wire shape.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "action": self.action,
            "instance_key": self.instance_key,
            "required": mem_json(&self.required),
            "resident_on": self.resident_on.as_ref().map(|p| json!({
                "device": device_name(p.device),
                "tier": tier_name(p.tier),
            })),
            "device": self.device.map(device_name),
            "evict": self.evict.iter().map(|e| json!({
                "model": e.model,
                "instance_key": e.instance_key,
                "device": device_name(e.device),
                "frees": e.frees,
            })).collect::<Vec<_>>(),
            "supported_devices": self.supported_devices.iter().copied().map(device_name).collect::<Vec<_>>(),
            "estimated_transfer": self.estimated_transfer,
            "runnable": self.runnable(),
            "refusal": self.refusal,
        })
    }
}

fn mem_json(c: &MemCost) -> Value {
    json!({ "vram": c.vram, "ram": c.ram, "npu": c.npu, "mapped": c.mapped })
}

/// A device's stable wire name: `"cpu"`, `"gpu0"`, `"npu0"` - the same
/// spelling brain's own `--device` flag accepts, so a caller can hand a
/// plan's answer straight back as a request.
#[must_use]
pub fn device_name(d: Device) -> String {
    match d {
        Device::Cpu => "cpu".to_string(),
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
    }
}

fn tier_name(t: Tier) -> &'static str {
    match t {
        Tier::Cold => "cold",
        Tier::Warm => "warm",
        Tier::Hot => "hot",
    }
}

/// Where an instance sits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub device: Device,
    pub tier: Tier,
}

/// One instance that would be evicted to make room.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eviction {
    /// The victim, named the way a caller can show it: its model id and the
    /// configuration fingerprint that distinguishes it from its siblings.
    pub model: String,
    /// The full instance key, for a caller that wants to act on it.
    pub instance_key: String,
    /// Which device it is being evicted from.
    pub device: Device,
    /// What evicting it frees on that device.
    pub frees: u64,
}

/// Why a plan could not be produced at all - distinct from a plan that says
/// the job cannot run, which is itself a useful answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// No model is registered under this id.
    UnknownModel { model: String },
    /// The model is registered but declares no such action, or rejected the
    /// parameters before an instance key could be derived.
    UnknownAction { model: String, action: String },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::UnknownModel { model } => {
                write!(f, "no model '{model}' is registered on this host")
            }
            PlanError::UnknownAction { model, action } => write!(
                f,
                "model '{model}' does not serve an action '{action}' with these parameters"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// The devices `cost` could be placed on if each were empty.
///
/// Reads the budget's TOTAL rather than its free bytes on purpose: this
/// answers "is this device even a candidate for this model", a property of
/// the hardware and the model, not of what happens to be loaded right now.
/// What is loaded right now is [`RunPlan::evict`]'s job.
pub(crate) fn supported_devices(cost: &MemCost, budgets: &Budgets) -> Vec<Device> {
    let mut out = Vec::new();
    for device in budgets.devices() {
        let Some(budget) = budgets.get(device) else {
            continue;
        };
        let need = match device {
            Device::Cpu => cost.ram,
            Device::Gpu(_) => cost.vram,
            // Zero NPU bytes is precisely how a model says it has no NPU
            // path (see `MemCost::npu`), so it is never an NPU candidate -
            // not even on an NPU with room to spare.
            Device::Npu(_) => {
                if cost.npu == 0 {
                    continue;
                }
                cost.npu
            }
        };
        if budget.total.saturating_sub(budget.reserved) >= need {
            out.push(device);
        }
    }
    // `Device` is not `Ord` (an enum of card indices has no natural total
    // order), so sort on the wire name instead -- which is the order a
    // caller displays them in anyway, and is stable across runs.
    out.sort_by_key(|d| device_name(*d));
    out
}

/// Bytes that would move into device memory to activate `cost` on `device`.
pub(crate) fn transfer_for(cost: &MemCost, device: Option<Device>) -> u64 {
    match device {
        Some(Device::Gpu(_)) => cost.vram,
        Some(Device::Npu(_)) => cost.npu,
        // Nothing crosses a bus for a CPU-resident instance: the host bytes
        // ARE the instance. Reporting `ram` here would tell a caller it is
        // about to pay a transfer cost it will not pay.
        Some(Device::Cpu) | None => 0,
    }
}

/// The devices a caller wants excluded from consideration.
pub(crate) type Excluded = HashSet<Device>;

/// The model id embedded in an instance key, for display.
pub(crate) fn model_of(key: &InstanceKey) -> String {
    key.model.clone()
}
