// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the machine can schedule: the cards, NPUs and RAM a serving process
//! budgets models against, narrowed to the `--device` set it was given.
//!
//! Swedish Embedded AB implements hardware-aware model serving for its
//! clients. If your team needs expertise in budgeting models onto the GPUs,
//! NPUs and RAM a machine really has, you can procure our services by sending
//! an email to info@swedishembedded.com.

use gpu_core::ComputeSet;

/// The schedulable compute of this machine, in the shapes
/// [`crate::build_executor`] budgets against.
#[derive(Clone, Debug, Default)]
pub struct Machine {
    /// `(index, free bytes)` per schedulable GPU.
    pub gpus: Vec<(u32, u64)>,
    /// `(index, budget bytes)` per schedulable NPU. An NPU shares host RAM.
    pub npus: Vec<(u32, u64)>,
    /// The GPU indices among [`Self::gpus`] whose bytes ARE host RAM (an
    /// integrated GPU, or the no-discrete-GPU fallback).
    pub unified_gpus: Vec<u32>,
    /// What `Device::Cpu` may be budgeted for compute: the host RAM available,
    /// or `0` when the CPU was excluded from compute. Host RAM stays a spill
    /// tier either way, which is why [`Self::pool_ram`] is separate.
    pub cpu_compute_ram: u64,
    /// The physical host RAM that the CPU, every unified GPU and every NPU draw
    /// from together. Never zeroed by excluding the CPU from compute.
    pub pool_ram: u64,
    /// The `--device` text the set came from, or `"all"`.
    pub compute: String,
}

impl Machine {
    /// Discovers what this machine can schedule, narrowed to `compute` (the
    /// resolved `--device` set; `None` means everything).
    ///
    /// Prints, to stderr, what it found and why a device was left out, because
    /// the serving process's own log is the only place an operator can read it.
    pub fn probe(compute: Option<&ComputeSet>) -> Machine {
        // Discover the GPUs' capacity so the scheduler can budget/evict against real VRAM,
        // then narrow to what `--device` made schedulable. With no `--device` the set is
        // every device, which is exactly the "use all the hardware wisely" default.
        // FREE bytes, not total. `--reserve-gb` is then carved out of what is
        // actually available, so a card a neighbouring process is already holding
        // 18 GiB of is budgeted at 6 GiB rather than 24. Budgeting from the card's
        // SIZE is what let the daemon plan a placement the driver then refused -
        // the scheduler's own accounting said the card was empty. Same probe the
        // one-shot placer uses (`gpu_core::capacity`), so the two halves of this
        // process can no longer disagree about the same card at the same instant.
        let mut all_gpus = gpu_core::capacity::available_gpus();
        // No NVIDIA GPU, but the wgpu backend can drive an integrated GPU (e.g. Intel
        // Arc on Meteor Lake): budget it as a schedulable `Gpu` lane. Integrated GPUs
        // have no dedicated VRAM - they share system RAM - so size the budget like the
        // NPU (a modest fraction of RAM). This is what makes `--device gpu` (and the
        // all-devices default) actually schedule onto the iGPU on such boxes.
        // Devices this fallback creates ALWAYS share physical RAM with the CPU
        // (that is the case it exists for) - tracked so they can be declared into
        // the same memauth pool as Device::Cpu below, instead of budgeted as an
        // independent-but-physically-identical pool of bytes.
        let mut fallback_unified_gpus: Vec<u32> = Vec::new();
        if all_gpus.is_empty() {
            // Not `discrete_gpu_count` (that's 0 by definition on an integrated-only
            // box): `visible_gpu_count` counts the iGPU too, which is exactly the
            // case this fallback exists for.
            let n = gpu_core::visible_gpu_count();
            if n > 0 {
                // The real ceiling is the shared RAM pool declared below, not a
                // fraction reserved here - this device budget only needs to be AT
                // LEAST the pool's total so the pool (not a smaller guessed
                // fraction) is always the binding constraint.
                let ram = loader::placement::host_ram_available();
                all_gpus = (0..n as u32).map(|i| (i, ram)).collect();
                fallback_unified_gpus = (0..n as u32).collect();
                eprintln!("brain serve: no NVIDIA GPU; budgeting {n} integrated GPU(s), sharing the {} GB RAM pool (schedulable)", ram >> 30);
            }
        }
        let set = compute;
        let gpus: Vec<(u32, u64)> = match set {
            Some(s) => all_gpus.iter().copied().filter(|(i, _)| s.gpus.contains(i)).collect(),
            None => all_gpus.clone(),
        };
        let cpu_schedulable = set.map(|s| s.cpu_enabled()).unwrap_or(true);
        // Devices whose bytes physically ARE the CPU's RAM: this fallback's
        // synthesized GPUs, plus any real GPU the device registry classifies as
        // integrated (an Arc/Xe iGPU reporting real VRAM via query_gpu_mem - the
        // common case on this box - never goes through the fallback above, so it
        // needs its own check here). A discrete GPU with dedicated VRAM is not
        // included. See `memauth`'s module doc for why declaring this matters:
        // without it, a GPU-side allocation and a CPU-side one are budgeted as
        // if they came from two separate pools of memory, when they are the same
        // physical bytes.
        let unified_gpus: Vec<u32> = gpus
            .iter()
            .map(|&(i, _)| i)
            .filter(|i| {
                fallback_unified_gpus.contains(i)
                    || gpu_core::devices::gpus().iter().any(|d| d.index == *i && d.identity.class == backend_api::DeviceClass::IntegratedGpu)
            })
            .collect();

        // Schedulable NPUs: `--device` narrows to `set.npus`; with no `--device`, any NPU
        // present is scheduled. The Meteor-Lake-class NPU shares system RAM, so it gets a
        // modest per-device budget. A model with an NPU path (MemCost.npu > 0) is then
        // auto-placed on the NPU in preference to CPU/GPU (see place::pick_device).
        let npu_indices: Vec<u32> = match set {
            Some(s) => s.npus.clone(),
            None if npu::openvino::npu_present() => vec![0],
            None => vec![],
        };
        let ram = loader::placement::host_ram_available();
        // NPUs always share system RAM (see the comment above); their device
        // budget only needs to be at least the pool's total, same reasoning as
        // the iGPU fallback - `resident::build_executor` declares them into the
        // shared pool alongside `unified_gpus` and Device::Cpu.
        let npus: Vec<(u32, u64)> = npu_indices.iter().map(|&i| (i, ram)).collect();

        // What is actually schedulable is `gpus`/`npus`/`cpu_schedulable`, not just
        // `gpus` - a prior version of this message said "scheduling on CPU only"
        // purely from `gpus.is_empty()`, which was wrong on two counts whenever an
        // NPU was involved: `--device npu` schedules on the NPU (never CPU - CPU
        // compute is excluded, see `cpu_compute_ram` below), and `--device npu,cpu`
        // schedules on both, not "CPU only".
        if gpus.is_empty() && npus.is_empty() {
            if all_gpus.is_empty() {
                eprintln!("brain serve: no GPUs or NPUs detected; serving with CPU-only budget");
            } else {
                eprintln!("brain serve: --device excluded every GPU; scheduling on CPU only");
            }
        } else if gpus.is_empty() && !npus.is_empty() {
            if cpu_schedulable {
                eprintln!("brain serve: --device excluded every GPU; scheduling on NPU + CPU");
            } else {
                eprintln!("brain serve: --device restricted to NPU; scheduling on NPU only (CPU and GPU excluded)");
            }
        }
        let cpu_compute_ram = if cpu_schedulable { ram } else { 0 };
    Machine {
        gpus,
        npus,
        unified_gpus,
        cpu_compute_ram,
        pool_ram: ram,
        compute: set.map(|s| s.to_string()).unwrap_or_else(|| "all".into()),
    }
    }
}
