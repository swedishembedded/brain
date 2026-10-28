// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain devices` — print the canonical physical-GPU table.
//!
//! One row per card, in canonical (PCI-bus) order: the index is what `gpu<i>`
//! means everywhere in brain (`--device gpu1`, `Shard.gpu_index`,
//! `residency::Device::Gpu(i)`), and the PCI bus id is how nvidia-smi's own
//! ordering maps onto it. Also reports what the ambient selection
//! (`--device` / `BRAIN_DEVICE` / `BRAIN_GPU_INDEX`) resolves to, and how each
//! pair of cards reaches the other: the path a split model's boundary
//! transfers take and the bandwidth measured on it (`gpu_core::link`).

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
    index: u32,
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
    let hw = gpu_core::defaults::Hardware::detected();
    if hw.cuda_unusable.is_none() && hw.cards.iter().any(|c| c.index == index && c.compute_capability.is_some()) {
        seen.push("cuda");
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
        "{:<6} {:<28} {:<14} {:<10} {:>9}  backends",
        "index", "name", "pci bus", "uuid", "vram"
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
            backends_seeing(d.index, id, &wgpu_ids),
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
        // Nothing pinned no longer means "card 0": with no --device, brain
        // asks the placement policy which card can actually hold a model
        // right now (`gpu_core::devices::selected_device`). Report where a
        // `Gpu::new` would REALLY land, not where it used to.
        None => {
            let d = gpu_core::devices::selected_device().unwrap_or(&devs[0]);
            println!("ambient selection: none pinned - Gpu::new lands on gpu{} ({})", d.index, d.identity.name);
        }
    }
    let (backend, why) = gpu_core::defaults::candidates(gpu_core::defaults::BACKEND, gpu_core::defaults::Hardware::detected());
    println!("default backend: {} ({why}; --backend overrides)", backend[0].name());
    if let Some(why) = &gpu_core::defaults::Hardware::detected().cuda_unusable {
        println!("cuda: not usable here - {why}");
    }
    print_links(devs);
    print_npus();
}

/// How each pair of cards moves data between them. Opening a handle per card
/// is what makes the answer the backend's own (a CUDA driver's peer query, not
/// a guess from the PCI table), and the bandwidth is measured once and cached.
fn print_links(devs: &[gpu_core::devices::DeviceId]) {
    if devs.len() < 2 {
        return;
    }
    println!();
    println!("device links (the path a transfer between two cards takes; bandwidth is measured, then cached)");
    let open = |d: &gpu_core::devices::DeviceId| {
        std::panic::catch_unwind(|| gpu_core::Gpu::new_on(d, &[])).map_err(|_| format!("gpu{} could not be opened", d.index))
    };
    for (i, a) in devs.iter().enumerate() {
        for b in &devs[i + 1..] {
            let (ga, gb) = match (open(a), open(b)) {
                (Ok(ga), Ok(gb)) => (ga, gb),
                (Err(e), _) | (_, Err(e)) => {
                    println!("  gpu{}-gpu{}  unknown: {e}", a.index, b.index);
                    continue;
                }
            };
            let link = ga.peer_link(&gb);
            let how = match (&link.reason, link.performance_rank) {
                (None, Some(rank)) => format!("direct peer access (driver performance rank {rank})"),
                (None, None) => "direct peer access".to_string(),
                (Some(why), _) => format!("through host memory: {why}"),
            };
            let rate = |from: &gpu_core::Gpu, to: &gpu_core::Gpu| match gpu_core::link::bandwidth(from, to) {
                Some(bw) => format!("{:.1} GB/s{}", bw.used_gbs(), if bw.peer_gbs.is_some() { "" } else { " (host-staged)" }),
                None => "unmeasured".to_string(),
            };
            println!(
                "  gpu{}-gpu{}  {how}; gpu{}->gpu{} {}, gpu{}->gpu{} {}",
                a.index, b.index, a.index, b.index, rate(&ga, &gb), b.index, a.index, rate(&gb, &ga)
            );
        }
    }
}

/// NPU section: separate from the GPU table above because "present" and
/// "usable" are two different questions here. The device node
/// (`/dev/accel/accel*`) shows up as soon as the `intel_vpu` kernel driver
/// binds, which says nothing about whether OpenVINO can actually schedule on
/// it — that also needs host NPU firmware (`/lib/firmware/intel/vpu`, loaded
/// by the HOST kernel, invisible to `/dev/accel`'s mere presence) and a
/// working OpenVINO runtime. Reporting both lets `brain devices` explain a
/// "node present but `--device npu` still errors" machine instead of just
/// omitting the NPU as if it didn't exist — see
/// `scripts/build/setup-npu-runtime.sh` for the same two-question diagnostic.
fn print_npus() {
    let npu_nodes = gpu_core::Inventory::probe().npus;
    println!();
    if npu_nodes == 0 {
        println!("no NPU device node found (expected /dev/accel/accel* or /dev/rknpu)");
        return;
    }
    println!("NPU: {npu_nodes} device node(s) present (/dev/accel/accel*)");
    match npu::openvino::available_devices() {
        Ok(devs) if devs.iter().any(|d| d == "NPU" || d.starts_with("NPU.")) => {
            println!("  npu0   usable — OpenVINO reports it (available_devices: {})", devs.join(", "));
        }
        Ok(devs) => {
            println!(
                "  npu0   device node present but OpenVINO does NOT report NPU (available_devices: {}) \
                 — likely missing host NPU firmware (/lib/firmware/intel/vpu on the HOST, not any \
                 container); see scripts/build/setup-npu-runtime.sh",
                devs.join(", ")
            );
        }
        Err(e) => println!("  npu0   device node present but OpenVINO could not be queried: {e}"),
    }
}
