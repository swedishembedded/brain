// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`StatsSnapshot`] tree and its typed sections.
//!
//! Every section is a plain serde struct so the whole snapshot serializes to a
//! stable JSON document. Two rules keep it future-proof and data-driven:
//!
//! 1. **Collections, not fixed slots.** Accelerators, models, instances, requests,
//!    and connections are `Vec`s keyed by an `id`, so N of anything renders from
//!    the data (one GPU or eight; zero models or fifty).
//! 2. **An open `extra` at every level.** New leaf metrics go into
//!    `extra: BTreeMap<String, Value>` with no schema change; a typed view renders
//!    known fields and a generic tree view renders whatever is in `extra`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Bumped when the *typed* shape changes incompatibly. braintop reads this to pick
/// a renderer; `extra` growth never bumps it (that is the whole point of `extra`).
pub const SCHEMA_VERSION: u32 = 1;

/// The root of the stats tree. A single point-in-time view of the running system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatsSnapshot {
    /// Typed-schema version (see [`SCHEMA_VERSION`]).
    pub schema: u32,
    /// Every schedulable accelerator (CPU + each GPU + each NPU present). Adapts to
    /// any count — nvidia-smi-like memory rows.
    pub accelerators: Vec<Accelerator>,
    /// The model catalog, each with where (if anywhere) it is currently resident.
    pub models: Vec<ModelStat>,
    /// The scheduler/executor counters.
    pub executor: ExecutorStat,
    /// In-flight work (may be empty until the request registry is wired).
    pub requests: Vec<RequestStat>,
    /// Front-end connections (kept for forward-compat; may be empty for now).
    pub connections: Vec<ConnStat>,
    /// Open bag of snapshot-level metrics that have no typed home yet.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

impl Default for StatsSnapshot {
    fn default() -> StatsSnapshot {
        StatsSnapshot {
            schema: SCHEMA_VERSION,
            accelerators: Vec::new(),
            models: Vec::new(),
            executor: ExecutorStat::default(),
            requests: Vec::new(),
            connections: Vec::new(),
            extra: BTreeMap::new(),
        }
    }
}

impl StatsSnapshot {
    /// An empty snapshot stamped with the current schema version.
    pub fn new() -> StatsSnapshot {
        StatsSnapshot::default()
    }

    /// Serialize to a compact JSON string (the D-Bus wire form).
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".into())
    }

    /// Serialize to a `serde_json::Value`.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Parse a snapshot from its JSON string form.
    pub fn from_json_str(s: &str) -> Result<StatsSnapshot, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// One schedulable accelerator — a self-describing memory row (nvidia-smi-like).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Accelerator {
    /// Stable id, matching `--device` naming (`cpu`, `gpu0`, `npu0`).
    pub id: String,
    /// `"cpu"` | `"gpu"` | `"npu"`.
    pub kind: String,
    /// Human-facing name (e.g. `"GPU 0"`).
    pub name: String,
    /// Index within its kind.
    pub index: u32,
    pub mem_total: u64,
    pub mem_used: u64,
    pub mem_reserved: u64,
    /// Utilization percent when known (0..=100); `None` when not sampled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub util: Option<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

/// One model in the catalog, with its live residency (where it is Hot, per device).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelStat {
    /// Model id (the served name).
    pub id: String,
    /// Family / short description (grouping label for the model view).
    pub family: String,
    /// Advertised action names.
    pub capabilities: Vec<String>,
    /// True when at least one instance is currently resident.
    pub resident: bool,
    /// Each resident instance's placement (empty when cold).
    pub instances: Vec<Instance>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

/// One resident instance of a model on one device.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Instance {
    /// Device id (`cpu`, `gpu0`, `npu0`).
    pub device: String,
    /// Residency tier (`hot` | `warm` | `cold`).
    pub tier: String,
    /// Bytes held on that device.
    pub mem: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

/// Scheduler/executor counters (from `Executor::stats`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExecutorStat {
    pub builds: u64,
    pub evictions: u64,
    pub batches: u64,
    pub jobs: u64,
    pub resident: u64,
    pub queue_peak: u64,
    pub max_batch: u64,
    pub max_parallel: u64,
    /// Cumulative jobs admitted onto a lane (as opposed to `jobs`, which only
    /// counts once its group's `Done` arrives) -- moves the instant work starts.
    pub admitted: u64,
    /// LIVE queued-job count, unlike `queue_peak`'s never-resetting high-water
    /// mark -- what a dashboard actually wants to watch change over time.
    pub queue_depth: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

/// One in-flight request (in-progress work). Fields are optional so partial
/// knowledge still renders.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RequestStat {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Coarse phase (`queued` | `running` | ...).
    pub phase: String,
    /// Milliseconds since the request was observed.
    pub since_ms: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

/// One front-end connection (forward-compat placeholder — may be empty for now).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConnStat {
    pub id: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_snapshot_serializes_stably() {
        let snap = StatsSnapshot::new();
        let v = snap.to_value();
        // Every typed section is present; collections are empty arrays, not null.
        assert_eq!(v["schema"], json!(SCHEMA_VERSION));
        assert_eq!(v["accelerators"], json!([]));
        assert_eq!(v["models"], json!([]));
        assert_eq!(v["requests"], json!([]));
        assert_eq!(v["connections"], json!([]));
        assert_eq!(v["executor"]["builds"], json!(0));
        // Empty `extra` is omitted (skip_serializing_if), keeping the wire lean.
        assert!(v.get("extra").is_none());
    }

    #[test]
    fn adapts_to_n_accelerators_and_models_from_data() {
        let mut snap = StatsSnapshot::new();
        for i in 0..5 {
            snap.accelerators.push(Accelerator {
                id: format!("gpu{i}"),
                kind: "gpu".into(),
                name: format!("GPU {i}"),
                index: i,
                mem_total: 24 << 30,
                ..Default::default()
            });
        }
        for n in ["a", "b", "c"] {
            snap.models.push(ModelStat { id: n.into(), ..Default::default() });
        }
        let v = snap.to_value();
        assert_eq!(v["accelerators"].as_array().unwrap().len(), 5);
        assert_eq!(v["models"].as_array().unwrap().len(), 3);
        // A round-trip preserves the counts — the renderer reads them from data.
        let back = StatsSnapshot::from_json_str(&snap.to_json_string()).unwrap();
        assert_eq!(back.accelerators.len(), 5);
        assert_eq!(back.models.len(), 3);
    }

    #[test]
    fn extra_round_trips_at_every_level() {
        let mut snap = StatsSnapshot::new();
        snap.extra.insert("uptime_ms".into(), json!(1234));
        snap.executor.extra.insert("policy".into(), json!("default"));
        let mut acc = Accelerator { id: "gpu0".into(), kind: "gpu".into(), ..Default::default() };
        acc.extra.insert("temp_c".into(), json!(61));
        snap.accelerators.push(acc);

        let s = snap.to_json_string();
        let back = StatsSnapshot::from_json_str(&s).unwrap();
        assert_eq!(back.extra["uptime_ms"], json!(1234));
        assert_eq!(back.executor.extra["policy"], json!("default"));
        assert_eq!(back.accelerators[0].extra["temp_c"], json!(61));
    }
}
