// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing what a server that
// trains and serves on shared accelerators is doing, for its clients. If your
// team needs expertise in observable LLM serving and training then you can
// procure our services by sending an email to info@swedishembedded.com.

//! `GET /metrics` (+ `/v1/metrics`): the server's serving and training state as one versioned JSON document
//! ([`SCHEMA`] version [`SCHEMA_VERSION`]), for an observer of a learning loop or an operator. Every field is
//! documented in the metrics and health reference; fields are only ever added (a conformance test pins the set), never
//! renamed or given another unit.
//!
//! Per model: its residency (resident, building, warm, evicted, unservable with the reason; its last build and
//! eviction), its jobs, each replica's scheduler (KV pool, back-off, sequences, windowed token rates) and what it has
//! answered. Per card: memory as the residency manager accounts it, the training reservation, whether a workload
//! holds it. Per training job: its state, its wait, and its own account of its wall time by phase.
//!
//! Behind the key like every route. It names models and devices (as `/v1/models` and `/v1/capabilities` do), never
//! a path or request content.

use std::collections::BTreeMap;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use residency::observe::{ModelServed, Observation, WorkloadJob};
use residency::{InstanceKey, InstanceLife, LifeState};
use serde_json::{json, Value};

use crate::state::AppState;

/// The document's schema name.
pub const SCHEMA: &str = "brain.server.metrics";
/// The document's schema version: bumped only when a field changes meaning or goes; adding fields keeps it.
pub const SCHEMA_VERSION: u32 = 1;

/// The metrics routes (registered on every provider surface, outside the edge limit).
pub fn routes() -> Router<AppState> {
    Router::new().route("/metrics", get(metrics)).route("/v1/metrics", get(metrics))
}

async fn metrics(State(state): State<AppState>) -> Json<Value> {
    let exec = state.exec.clone();
    // The cards and lives round-trip through the dispatcher: off the async runtime.
    let observation = tokio::task::spawn_blocking(move || exec.observe()).await.unwrap_or_default();
    Json(document(&state, &observation))
}

/// Seconds as a float.
fn secs(d: std::time::Duration) -> f64 {
    d.as_secs_f64()
}

/// The whole document, from `observation` and what the executor counts.
pub fn document(state: &AppState, observation: &Observation) -> Value {
    let exec = &state.exec;
    let stats = exec.stats();
    let served = exec.served();
    let mut models = Vec::new();
    for manifest in exec.manifests().iter() {
        let name = &manifest.model;
        if exec.is_workload(name) {
            continue;
        }
        let group = exec.slot_group(name).unwrap_or_else(|| name.clone());
        let lives: Vec<&InstanceLife> = observation.lives.iter().filter(|l| l.key.model == group).collect();
        let unservable = stats.unservable.iter().find(|(k, _)| k.model == group).map(|(_, why)| why.clone());
        let mut entry = json!({
            "model": name,
            "slot_group": group,
            "residency": residency(&lives, unservable),
            "served": served_json(served.get(name)),
        });
        // The instance's own figures are on its slot group's entry: an adapter shares its base's.
        if &group == name {
            let mut keys: Vec<&InstanceKey> = stats.metrics.keys().filter(|k| k.model == group).collect();
            keys.sort_by(|a, b| a.config.cmp(&b.config));
            let replicas: Vec<Value> = keys.iter().filter_map(|k| replica(k, &stats.metrics[*k])).collect();
            let (queued, running) = stats.load.iter().filter(|(k, _)| k.model == group).fold((0, 0), |(q, r), (_, l)| (q + l.queued, r + l.running));
            entry["jobs"] = json!({"queued": queued, "running": running});
            entry["totals"] = totals(&replicas);
            entry["replicas"] = Value::Array(replicas);
        }
        models.push(entry);
    }
    json!({
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "instance": crate::health::instance(),
        "time_unix_ms": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
        "queue_depth": stats.queue_depth,
        "serving_held": stats.serving_held,
        "models": models,
        "cards": observation.cards.iter().map(|c| json!({
            "device": residency::observe::device_name(c.device),
            "total_bytes": c.total_bytes,
            "reserved_bytes": c.reserved_bytes,
            "used_bytes": c.used_bytes,
            "free_bytes": c.free_bytes,
            "ceiling_bytes": c.ceiling_bytes,
            "training_reservation_bytes": c.workload_reservation_bytes,
            "serving_held": c.serving_held,
        })).collect::<Vec<_>>(),
        "training_jobs": exec.workloads().iter().map(training_job).collect::<Vec<_>>(),
    })
}

/// A model's residency from its instances' lives: the most available state any replica is in, and its most recent
/// build and eviction.
fn residency(lives: &[&InstanceLife], unservable: Option<String>) -> Value {
    let rank = |s: LifeState| match s {
        LifeState::Resident => 0,
        LifeState::Building => 1,
        LifeState::Warm => 2,
        LifeState::Evicted => 3,
        LifeState::NotLoaded => 4,
    };
    let best = lives.iter().map(|l| l.state).min_by_key(|&s| rank(s));
    let state = match (best, &unservable) {
        (Some(LifeState::Resident), _) => "resident",
        (_, Some(_)) => "unservable",
        (Some(s), None) => s.slug(),
        (None, None) => LifeState::NotLoaded.slug(),
    };
    let mut devices: Vec<String> = lives.iter().flat_map(|l| l.devices.iter().map(|&d| residency::observe::device_name(d))).collect();
    devices.sort();
    devices.dedup();
    let last_build = lives.iter().filter_map(|l| l.last_build).max_by_key(|b| b.ended_unix_ms);
    let last_eviction = lives.iter().filter_map(|l| l.last_eviction).max_by_key(|e| e.at_unix_ms);
    json!({
        "state": state,
        "unservable_reason": unservable,
        "replicas_resident": lives.iter().filter(|l| l.state == LifeState::Resident).count(),
        "devices": devices,
        "building_secs": lives.iter().filter_map(|l| l.building_for).max().map(secs),
        "last_build": last_build.map(|b| json!({"ended_unix_ms": b.ended_unix_ms, "duration_secs": secs(b.duration), "succeeded": b.succeeded})),
        "last_eviction": last_eviction.map(|e| json!({"at_unix_ms": e.at_unix_ms, "resident_secs": e.resident_for.map(secs), "demoted": e.demoted})),
    })
}

/// One replica: its instance key's config and the scheduler object it publishes; `None` for an instance without a
/// scheduler (a model that does not batch through one).
fn replica(key: &InstanceKey, metrics: &[(String, Value)]) -> Option<Value> {
    let scheduler = metrics.iter().find(|(n, _)| n == SCHEDULER_METRIC).map(|(_, v)| v.clone())?;
    Some(json!({"config": key.config, "scheduler": scheduler}))
}

/// The metric a scheduler-backed instance publishes its observation under (`llmserve::SCHEDULER_METRIC`).
const SCHEDULER_METRIC: &str = "scheduler";

/// Sums over a model's replicas of what a router or dashboard reads first.
fn totals(replicas: &[Value]) -> Value {
    let sum = |path: &[&str]| -> Value {
        let values: Vec<&Value> = replicas.iter().filter_map(|r| path.iter().try_fold(&r["scheduler"], |v, k| v.get(*k))).collect();
        if values.is_empty() {
            return Value::Null;
        }
        match values.iter().all(|v| v.is_u64()) {
            true => json!(values.iter().filter_map(|v| v.as_u64()).sum::<u64>()),
            false => json!(values.iter().filter_map(|v| v.as_f64()).sum::<f64>()),
        }
    };
    json!({
        "kv_total_tokens": sum(&["kv", "total_tokens"]),
        "kv_claimable_tokens": sum(&["kv", "claimable_tokens"]),
        "kv_total_blocks": sum(&["kv", "total_blocks"]),
        "kv_free_blocks": sum(&["kv", "free_blocks"]),
        "prefix_cached_blocks": sum(&["kv", "prefix_cached_blocks"]),
        "admitted_width": sum(&["admission", "admitted_width"]),
        "running_sequences": sum(&["sequences", "running"]),
        "waiting_sequences": sum(&["sequences", "waiting"]),
        "decode_tokens_per_s": sum(&["throughput", "decode_tokens_per_s"]),
        "prefill_tokens_per_s": sum(&["throughput", "prefill_tokens_per_s"]),
    })
}

fn served_json(served: Option<&ModelServed>) -> Value {
    let empty = ModelServed::default();
    let s = served.unwrap_or(&empty);
    json!({
        "requests_completed": s.completed,
        "prompt_tokens": s.prompt_tokens,
        "completion_tokens": s.completion_tokens,
        "cancelled": s.cancelled,
        "refused": s.refused.iter().collect::<BTreeMap<_, _>>(),
    })
}

fn training_job(job: &WorkloadJob) -> Value {
    json!({
        "id": job.id,
        "model": job.model,
        "action": job.action,
        "state": job.state.slug(),
        "submitted_unix_ms": job.submitted_unix_ms,
        "started_unix_ms": job.started_unix_ms,
        "ended_unix_ms": job.ended_unix_ms,
        "queued_secs": secs(job.queued_for),
        "wall_secs": secs(job.wall),
        "report": job.report,
    })
}
