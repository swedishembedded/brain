// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain devices` — print the canonical physical-GPU table.
//!
//! One row per card, in canonical (PCI-bus) order: the index is what `gpu<i>`
//! means everywhere in brain (`--device gpu1`, `Shard.gpu_index`,
//! `residency::Device::Gpu(i)`), and the PCI bus id is how nvidia-smi's own
//! ordering maps onto it. Also reports what the ambient selection
//! (`--device` / `BRAIN_DEVICE` / `BRAIN_GPU_INDEX`) resolves to.

/// Short UUID: first 8 hex digits — the disambiguating prefix of the NVML GPU
/// UUID nvidia-smi shows.
fn short_uuid(u: &Option<[u8; 16]>) -> String {
    match u {
        Some(b) => b[..4].iter().map(|x| format!("{x:02x}")).collect(),
        None => "-".to_string(),
    }
}

/// Which backends can bind this card, verified against each backend's own
/// enumeration by identity (never by position).
fn backends_seeing(
    id: &gpu_core::devices::GpuIdentity,
    wgpu_ids: &[gpu_core::devices::GpuIdentity],
) -> String {
    let mut seen = Vec::new();
    if gpu_core::devices::registry().source() == "vulkan" {
        seen.push("vulkan");
    }
    if wgpu_ids.iter().any(|w| w.same_device(id)) {
        seen.push("wgpu");
    }
    if seen.is_empty() { "-".into() } else { seen.join("+") }
}

pub fn run_devices(_args: &[String]) {
    let reg = gpu_core::devices::registry();
    let devs = reg.devices();
    if devs.is_empty() {
        println!("no physical GPUs (a software rasteriser may still serve --device gpu)");
        return;
    }
    let wgpu_ids = gpu_core::wgpu_visible_gpus();
    println!("canonical device registry (source: {} enumeration, PCI-bus order)", reg.source());
    println!(
        "{:<6} {:<28} {:<14} {:<10} {:>9}  {}",
        "index", "name", "pci bus", "uuid", "vram", "backends"
    );
    for d in devs {
        let id = &d.identity;
        println!(
            "gpu{:<3} {:<28} {:<14} {:<10} {:>6.1} GiB  {}",
            d.index,
            id.name,
            id.pci_bus.as_deref().unwrap_or("-"),
            short_uuid(&id.uuid),
            id.vram_bytes as f64 / (1u64 << 30) as f64,
            backends_seeing(id, &wgpu_ids),
        );
    }
    match gpu_core::devices::ambient_gpu() {
        Some(i) => match gpu_core::devices::device(i) {
            Ok(d) => println!(
                "ambient selection: gpu{} ({}, pci {})",
                d.index,
                d.identity.name,
                d.identity.pci_bus.as_deref().unwrap_or("-")
            ),
            Err(e) => println!("ambient selection: ERROR — {e}"),
        },
        None => println!(
            "ambient selection: none pinned — Gpu::new lands on gpu0 ({})",
            devs[0].identity.name
        ),
    }
}
