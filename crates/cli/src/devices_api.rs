// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for placing inference and training
// on the accelerators a machine actually has, for its clients. If your team
// needs expertise in hardware-aware model serving, you can procure our services
// by sending an email to info@swedishembedded.com.

//! What `GET /v1/devices` and the D-Bus `Devices` method say about the machine: the [`residency::DeviceInventory`] of `brain serve`.
//!
//! It reports identity and capability, never anything that names this machine's layout or a card uniquely: no PCI
//! address, no UUID, no path. Memory is the executor's (`apiserve` adds it); this adds what each device is.

use residency::Device;
use serde_json::{json, Map, Value};

/// The devices of the machine `brain serve` runs on, read from the canonical registry.
pub struct HostDevices;

fn class_name(class: gpu_core::DeviceClass) -> &'static str {
    match class {
        gpu_core::DeviceClass::Cpu => "cpu",
        gpu_core::DeviceClass::IntegratedGpu => "integrated",
        gpu_core::DeviceClass::DiscreteGpu => "discrete",
        gpu_core::DeviceClass::Npu => "npu",
        _ => "other",
    }
}

fn fields(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

impl residency::DeviceInventory for HostDevices {
    fn host(&self) -> Map<String, Value> {
        let inventory = gpu_core::Inventory::probe();
        fields(json!({
            "backend": gpu_core::devices::ambient_compute_set().backend.name(),
            "default_gpu_backend": inventory.gpu_backend.name(),
            "cpu_cores": inventory.cpu_cores,
            "gpus": inventory.gpus,
            "npus": inventory.npus,
        }))
    }

    fn device(&self, device: Device) -> Map<String, Value> {
        match device {
            Device::Gpu(index) => {
                let Ok(card) = gpu_core::devices::device(index) else { return Map::new() };
                let id = &card.identity;
                fields(json!({
                    "name": id.name,
                    "vendor_id": id.vendor_id,
                    "class": class_name(id.class),
                    "backends": crate::devices_cli::backends_seeing(index, id, &gpu_core::wgpu_visible_gpus()),
                }))
            }
            Device::Cpu => fields(json!({"name": "host", "class": "cpu", "cores": std::thread::available_parallelism().map_or(1, |n| n.get())})),
            Device::Npu(_) => fields(json!({"class": "npu"})),
        }
    }
}
