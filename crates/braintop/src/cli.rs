// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--cli` flat batch mode: turn a [`StatsSnapshot`] into stable, shell-parseable
//! `path.to.metric=value` lines.
//!
//! Keys are **stable and data-driven**: collections are keyed by their id (never
//! by position), so a line for a given GPU or model keeps the same path across
//! runs regardless of ordering — `accelerator.gpu0.mem_used=…`,
//! `model.qwen.instances.gpu0.tier=hot`. Open `extra` maps at every level are
//! flattened generically (nested objects dotted, arrays indexed), so metrics added
//! later show up with no code change here.

use std::collections::BTreeMap;

use brain_stats::StatsSnapshot;
use serde_json::Value;

/// Flatten a whole snapshot into deterministic `path=value` lines.
pub fn flatten_snapshot(s: &StatsSnapshot) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!("schema={}", s.schema));

    // Accelerators — keyed by id (cpu / gpu0 / npu0).
    for a in &s.accelerators {
        let p = format!("accelerator.{}", a.id);
        out.push(format!("{p}.kind={}", a.kind));
        out.push(format!("{p}.name={}", a.name));
        out.push(format!("{p}.index={}", a.index));
        out.push(format!("{p}.mem_total={}", a.mem_total));
        out.push(format!("{p}.mem_used={}", a.mem_used));
        out.push(format!("{p}.mem_reserved={}", a.mem_reserved));
        if let Some(u) = a.util {
            out.push(format!("{p}.util={u}"));
        }
        flatten_extra(&format!("{p}.extra"), &a.extra, &mut out);
    }

    // Models — keyed by id; instances keyed by device.
    for m in &s.models {
        let p = format!("model.{}", m.id);
        out.push(format!("{p}.family={}", m.family));
        out.push(format!("{p}.resident={}", m.resident));
        out.push(format!("{p}.capabilities={}", m.capabilities.join(",")));
        for inst in &m.instances {
            let ip = format!("{p}.instances.{}", inst.device);
            out.push(format!("{ip}.tier={}", inst.tier));
            out.push(format!("{ip}.mem={}", inst.mem));
            flatten_extra(&format!("{ip}.extra"), &inst.extra, &mut out);
        }
        flatten_extra(&format!("{p}.extra"), &m.extra, &mut out);
    }

    // Executor counters.
    let e = &s.executor;
    out.push(format!("executor.builds={}", e.builds));
    out.push(format!("executor.evictions={}", e.evictions));
    out.push(format!("executor.batches={}", e.batches));
    out.push(format!("executor.jobs={}", e.jobs));
    out.push(format!("executor.resident={}", e.resident));
    out.push(format!("executor.queue_peak={}", e.queue_peak));
    out.push(format!("executor.max_batch={}", e.max_batch));
    out.push(format!("executor.max_parallel={}", e.max_parallel));
    flatten_extra("executor.extra", &e.extra, &mut out);

    // Requests — keyed by id (empty for now; populates automatically later).
    for r in &s.requests {
        let p = format!("request.{}", r.id);
        if let Some(v) = &r.provider {
            out.push(format!("{p}.provider={v}"));
        }
        if let Some(v) = &r.model {
            out.push(format!("{p}.model={v}"));
        }
        if let Some(v) = &r.action {
            out.push(format!("{p}.action={v}"));
        }
        out.push(format!("{p}.phase={}", r.phase));
        out.push(format!("{p}.since_ms={}", r.since_ms));
        flatten_extra(&format!("{p}.extra"), &r.extra, &mut out);
    }

    // Connections — keyed by id (empty for now).
    for c in &s.connections {
        let p = format!("connection.{}", c.id);
        flatten_extra(&format!("{p}.extra"), &c.extra, &mut out);
    }

    // Snapshot-level open metrics.
    flatten_extra("extra", &s.extra, &mut out);
    out
}

/// Convenience: the flattened lines joined with `\n` (what `--cli` prints).
pub fn render_cli(s: &StatsSnapshot) -> String {
    flatten_snapshot(s).join("\n")
}

/// Flatten an `extra` map under `prefix` (skips an empty map so lean snapshots
/// stay lean).
fn flatten_extra(prefix: &str, extra: &BTreeMap<String, Value>, out: &mut Vec<String>) {
    for (k, v) in extra {
        flatten_value(&format!("{prefix}.{k}"), v, out);
    }
}

/// Recursively flatten an arbitrary JSON value: objects dotted (BTreeMap key
/// order — deterministic), arrays indexed by position, scalars emitted as leaves.
fn flatten_value(prefix: &str, v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                flatten_value(&format!("{prefix}.{k}"), val, out);
            }
        }
        Value::Array(items) => {
            for (i, val) in items.iter().enumerate() {
                flatten_value(&format!("{prefix}.{i}"), val, out);
            }
        }
        Value::String(s) => out.push(format!("{prefix}={s}")),
        Value::Bool(b) => out.push(format!("{prefix}={b}")),
        Value::Number(n) => out.push(format!("{prefix}={n}")),
        Value::Null => out.push(format!("{prefix}=null")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests_support::sample_snapshot;

    #[test]
    fn flattens_known_keys_deterministically() {
        let snap = sample_snapshot();
        let lines = flatten_snapshot(&snap);

        // Schema + an accelerator memory leaf, keyed by id (not position).
        assert!(lines.contains(&"schema=1".to_string()));
        assert!(lines.contains(&"accelerator.gpu0.mem_used=8589934592".to_string()));
        assert!(lines.contains(&"accelerator.cpu.kind=cpu".to_string()));

        // A model instance keyed by device — the documented shape.
        assert!(lines.contains(&"model.qwen.instances.gpu0.tier=hot".to_string()));
        assert!(lines.contains(&"model.qwen.instances.cpu.tier=warm".to_string()));
        assert!(lines.contains(&"model.qwen.resident=true".to_string()));

        // Executor counters + a generic `extra` leaf (nested value, dotted).
        assert!(lines.contains(&"executor.builds=3".to_string()));
        assert!(lines.contains(&"accelerator.gpu0.extra.temp_c=61".to_string()));
        assert!(lines.contains(&"extra.uptime_ms=1234".to_string()));

        // A non-resident model still emits its typed leaves.
        assert!(lines.contains(&"model.tts.resident=false".to_string()));
    }

    #[test]
    fn output_is_stable_across_calls() {
        let snap = sample_snapshot();
        assert_eq!(render_cli(&snap), render_cli(&snap));
    }
}
