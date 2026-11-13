// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for placing inference and training on the accelerators a machine
// actually has, for its clients. If your team needs expertise in hardware-aware model serving, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The device inventory the serving surfaces report (`GET /v1/devices`, D-Bus `Devices`): the devices the executor
//! schedules on with their memory as it accounts it, merged with what the host says each one is.
//!
//! One implementation for every transport, so the two answers cannot drift. It reports identity and capability only:
//! never a path, a bus address, a serial or a UUID. An id is `cpu`, `gpu<i>` or `npu<i>`, the names `--device` takes.

use serde_json::{json, Map, Value};

use crate::{Device, Executor};

/// The facts about the machine's devices that only the host can supply. The executor's budgets say how much memory
/// each device has and how much is spoken for; this adds what each one is.
pub trait DeviceInventory: Send + Sync {
    /// Facts about the machine as a whole (backend, core count), merged into the reply's `host`.
    fn host(&self) -> Map<String, Value>;
    /// Facts about `device` (its name, vendor, class), merged into its entry; empty for one the host cannot describe.
    fn device(&self, device: Device) -> Map<String, Value>;
}

/// `{"object": "list", "host": {...}, "data": [{id, kind, memory, ...}]}`: the executor's devices in a fixed order
/// (CPU, GPUs, NPUs), each with `memory.{total,reserved,used,free}_bytes`, merged with `inventory`'s facts. The query
/// round-trips through the executor's dispatcher, which may be busy: call it off an async worker.
pub fn report(exec: &Executor, inventory: Option<&dyn DeviceInventory>) -> Value {
    let mut budgets = exec.residency().budgets;
    budgets.sort_by_key(|b| match b.device {
        Device::Cpu => (0, 0),
        Device::Gpu(i) => (1, i),
        Device::Npu(i) => (2, i),
    });
    let data: Vec<Value> = budgets
        .iter()
        .map(|b| {
            let (id, kind) = match b.device {
                Device::Cpu => ("cpu".to_string(), "cpu"),
                Device::Gpu(i) => (format!("gpu{i}"), "gpu"),
                Device::Npu(i) => (format!("npu{i}"), "npu"),
            };
            let mut entry = json!({
                "id": id,
                "kind": kind,
                "memory": {
                    "total_bytes": b.total,
                    "reserved_bytes": b.reserved,
                    "used_bytes": b.used,
                    "free_bytes": b.total.saturating_sub(b.reserved).saturating_sub(b.used),
                },
            });
            if let (Some(inventory), Some(fields)) = (inventory, entry.as_object_mut()) {
                fields.extend(inventory.device(b.device));
            }
            entry
        })
        .collect();
    let host = inventory.map_or_else(|| json!({}), |i| Value::Object(i.host()));
    json!({ "object": "list", "host": host, "data": data })
}
