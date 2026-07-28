// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The result artifact — `brain.perf/1`.
//!
//! One JSON file per run. The schema is fixed so `compare` can diff runs from
//! different machines, months apart, without a migration. Two rules:
//!
//! * **Unmeasured fields are `null`, never omitted and never `0`.** "We did not
//!   measure device utilisation" and "device utilisation was zero" are different
//!   facts and a comparison that confuses them is wrong.
//! * **`valid: false` excludes a result from comparison.** A performance number
//!   whose correctness gate failed is not a slower-but-honest number; it is a
//!   measurement of a different, broken computation.

use serde_json::{json, Map, Value};

use crate::env::Env;
use crate::stats::r3;
use crate::target::TargetInfo;

pub const SCHEMA: &str = "brain.perf/1";

/// A complete run result.
pub struct Artifact {
    pub scenario: String,
    pub valid: bool,
    pub invalid_reason: Option<String>,
    pub smoke: bool,
    pub env: Env,
    pub target: TargetInfo,
    pub workload: Value,
    pub performance: Value,
    pub scheduling: Value,
    pub memory: Value,
    pub reliability: Value,
    pub resources: Value,
    pub correctness: Value,
    /// `sweep` only: one entry per concurrency level.
    pub curve: Option<Vec<Value>>,
    /// `mixed` only: per-traffic-class blocks.
    pub per_class: Option<Vec<Value>>,
    pub best_of_n: usize,
    pub spread_pct: Option<f64>,
    /// What this run could NOT measure and why. A scenario that is limited by a
    /// missing engine capability says so here rather than reporting a confident
    /// number it did not observe.
    pub notes: Option<String>,
}

impl Artifact {
    pub fn new(scenario: &str, env: Env, target: TargetInfo) -> Artifact {
        Artifact {
            scenario: scenario.to_string(),
            valid: true,
            invalid_reason: None,
            smoke: false,
            env,
            target,
            workload: Value::Null,
            performance: Value::Null,
            scheduling: Value::Null,
            memory: empty_memory(),
            reliability: empty_reliability(),
            resources: empty_resources(),
            correctness: not_checked(),
            curve: None,
            per_class: None,
            best_of_n: 1,
            spread_pct: None,
            notes: None,
        }
    }

    /// Mark the run invalid. Used by the correctness gate; `compare` then skips it.
    pub fn invalidate(&mut self, reason: &str) {
        self.valid = false;
        self.invalid_reason = Some(reason.to_string());
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "scenario": self.scenario,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason.clone().map(Value::from).unwrap_or(Value::Null),
            "smoke": self.smoke,
            "env": self.env.to_json(),
            "target": self.target.to_json(),
            "workload": self.workload,
            "performance": self.performance,
            "scheduling": self.scheduling,
            "memory": self.memory,
            "reliability": self.reliability,
            "resources": self.resources,
            "correctness": self.correctness,
            "best_of_n": self.best_of_n,
            "spread_pct": self.spread_pct.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "notes": self.notes.clone().map(Value::from).unwrap_or(Value::Null),
            "curve": self.curve.clone().map(Value::from).unwrap_or(Value::Null),
            "per_class": self.per_class.clone().map(Value::from).unwrap_or(Value::Null),
        })
    }

    /// Default artifact path: `results/perf-<scenario>-<model>-<device>-<seed>.json`.
    pub fn default_path(&self, seed: u64) -> String {
        format!(
            "results/perf-{}-{}-{}-{}.json",
            self.scenario, self.target.model, self.env.device_requested, seed
        )
    }

    pub fn write(&self, path: &str) -> std::io::Result<()> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let text = serde_json::to_string_pretty(&self.to_json()).map_err(std::io::Error::other)?;
        std::fs::write(path, text + "\n")
    }
}

/// Engine counters merged into the `memory` block by a target that has them.
pub fn memory_with(counters: &[(String, Value)]) -> Value {
    let mut m = match empty_memory() {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    for (k, v) in counters {
        m.insert(k.clone(), v.clone());
    }
    Value::Object(m)
}

pub fn empty_memory() -> Value {
    json!({
        "kv_effective_capacity_artifacts": Value::Null,
        "kv_theoretical_artifacts": Value::Null,
        "kv_hit_rate": Value::Null,
        "eviction_regret": Value::Null,
        "recomputed_artifacts": Value::Null,
        "fragmentation": Value::Null,
        "bytes_moved_per_artifact": Value::Null,
        "peak_device_mb": Value::Null,
        "peak_host_mb": Value::Null
    })
}

pub fn empty_reliability() -> Value {
    json!({
        "cancelled_compute_waste": Value::Null,
        "failure_detect_ms": Value::Null,
        "recovery_ms": Value::Null,
        "lost_requests": 0,
        "corrupted_responses": 0,
        "errors": 0,
        "rejections": 0,
        "timeouts": 0,
        "ooms": 0
    })
}

pub fn empty_resources() -> Value {
    json!({
        "device_util": Value::Null,
        "host_cpu_util": Value::Null,
        "host_mem_mb": Value::Null,
        "storage_read_mb": Value::Null,
        "energy_j": Value::Null,
        "j_per_output_artifact": Value::Null
    })
}

/// The correctness block when no gate ran. `passed: null` — *not* `true`, so an
/// ungated run can never be mistaken for a verified one.
pub fn not_checked() -> Value {
    json!({
        "gate": Value::Null,
        "reference": Value::Null,
        "greedy_token_match": Value::Null,
        "mean_logprob_error": Value::Null,
        "structured_validity": Value::Null,
        "protocol_errors": Value::Null,
        "passed": Value::Null
    })
}

/// A passing correctness gate.
pub fn checked(gate: &str, reference: &str, match_rate: f64) -> Value {
    json!({
        "gate": gate,
        "reference": reference,
        "greedy_token_match": r3(match_rate),
        "mean_logprob_error": Value::Null,
        "structured_validity": Value::Null,
        "protocol_errors": 0,
        "passed": true
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Env;
    use crate::target::TargetInfo;

    fn art() -> Artifact {
        Artifact::new("latency", Env::capture("cpu"), TargetInfo::new("qwen", "token"))
    }

    #[test]
    fn unmeasured_fields_serialise_as_null() {
        let a = art();
        let j = a.to_json();
        assert!(j["memory"]["kv_hit_rate"].is_null());
        assert!(j["resources"]["energy_j"].is_null());
        assert!(j["curve"].is_null());
    }

    #[test]
    fn an_ungated_run_is_never_reported_as_verified() {
        let j = art().to_json();
        assert!(j["correctness"]["passed"].is_null(), "no gate must read as null, not true");
    }

    #[test]
    fn invalidate_records_the_reason() {
        let mut a = art();
        assert!(a.to_json()["valid"].as_bool().unwrap());
        a.invalidate("greedy mismatch 0.87 < 0.9999");
        let j = a.to_json();
        assert!(!j["valid"].as_bool().unwrap());
        assert!(j["invalid_reason"].as_str().unwrap().contains("greedy"));
    }

    #[test]
    fn schema_and_path_are_stable() {
        let a = art();
        assert_eq!(a.to_json()["schema"], SCHEMA);
        let p = a.default_path(1234);
        assert!(p.starts_with("results/perf-latency-qwen-cpu-1234"), "got {p}");
    }

    #[test]
    fn counters_merge_into_memory_without_dropping_the_null_fields() {
        let m = memory_with(&[("kv_hit_rate".into(), json!(0.61))]);
        assert_eq!(m["kv_hit_rate"], 0.61);
        assert!(m["fragmentation"].is_null(), "unset fields must survive the merge as null");
    }

    #[test]
    fn artifact_round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("perf-schema-test-{}", std::process::id()));
        let path = dir.join("a.json");
        let a = art();
        a.write(path.to_str().unwrap()).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        let v: Value = serde_json::from_str(&text).expect("parse");
        assert_eq!(v["schema"], SCHEMA);
        assert_eq!(v["scenario"], "latency");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
