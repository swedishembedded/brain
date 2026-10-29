// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native **CUDA Driver API** support for brain.
//!
//! Swedish Embedded AB implements native GPU compute backends for its clients.
//! If your team needs expertise in driving NVIDIA hardware from Rust without a
//! CUDA build-time dependency, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # What this crate does today
//!
//! It loads `libcuda.so.1` at run time ([`driver`]), enumerates the devices
//! that driver exposes, and reports each one as a
//! [`backend_api::GpuIdentity`] keyed on `cuDeviceGetUuid` - the same 16 bytes
//! Vulkan reports as `deviceUUID`, which is what lets a CUDA ordinal resolve
//! to brain's canonical `gpu<i>` index without trusting either enumeration's
//! order.
//!
//! On top of that it implements [`backend_api::Backend`]
//! ([`backend::CudaBackend`]): device allocations, host transfers, dispatches
//! whose CUDA source is generated from the registered WGSL and compiled by
//! NVRTC **on first use of each kernel**, and a capability report built from
//! device queries. What it does NOT have is a native kernel: every
//! dispatch it serves today is the generated (`ImplSource::Generated`) tier,
//! and a kernel outside that generator's supported subset is refused by name
//! at the dispatch that needed it - never approximated and never diverted to
//! another device, because an explicitly requested backend is a hard
//! contract.
//!
//! It also states the **tier policy** ([`policy`]): which operators are
//! required to reach which implementation tier, on which queried compute
//! capability. That lands before any kernel does, so the first native kernel
//! arrives into something that checks the claim it makes about itself -
//! every tier computes the same answers, so nothing else would.
//!
//! # Nothing about the hardware is compiled in
//!
//! Device count, compute capability, VRAM and SM count are queried per device.
//! A card's properties are an answer this crate asks for at run time, never a
//! constant, and never a permanent ceiling - see [`driver::CudaDevice`].

pub mod aot;
pub mod backend;
pub mod cublas;
pub mod driver;
pub mod exec;
mod graph;
pub mod live;
pub mod nccl;
pub mod nvrtc;
pub mod nvtx;
pub mod policy;
mod upload_ring;

pub use backend::{CudaBackend, LaunchStats, PASS_FIRST_FLUSH_STEPS, PASS_FLUSH_STEPS};
pub use driver::{driver, CudaDevice};
pub use live::{call_totals, live_resources, zeroed_by_size, CallTotals, LiveResources};

/// Every CUDA-visible device as a brain [`backend_api::GpuIdentity`], in CUDA
/// ordinal order - the shape `gpu_core::devices` consumes, matching
/// `backend_vulkan::enumerate_physical_gpus` and `backend_wgpu::enumerate_gpus`.
///
/// `Err` on any box where the driver cannot be loaded or initialised, which is
/// an ordinary, expected outcome (no NVIDIA driver installed) and not a fault.
/// `Ok(vec![])` means the driver loaded and exposes no device - a different
/// situation, e.g. `CUDA_VISIBLE_DEVICES=` filtering everything out.
pub fn enumerate_physical_gpus() -> Result<Vec<backend_api::GpuIdentity>, String> {
    let d = driver::driver().map_err(|e| e.to_string())?;
    Ok(d.devices()?.iter().map(|dev| dev.identity()).collect())
}

/// A card the CUDA backend can run on.
#[derive(Clone, Debug)]
pub struct UsableGpu {
    pub identity: backend_api::GpuIdentity,
    /// `(major, minor)`, as the driver reports it.
    pub compute_capability: (u32, u32),
}

/// The cards the CUDA backend can run on: the driver enumerates them AND
/// kernels can be had for them - NVRTC to compile, or an ahead-of-time store to
/// load. `Err` says which is missing, the ordinary answer on a box without the
/// NVIDIA driver or toolkit.
pub fn usable_gpus() -> Result<Vec<UsableGpu>, String> {
    let d = driver::driver().map_err(|e| e.to_string())?;
    let cards: Vec<_> = d.devices()?.iter().map(|dev| UsableGpu { identity: dev.identity(), compute_capability: (dev.cc_major, dev.cc_minor) }).collect();
    if cards.is_empty() {
        return Err("the CUDA driver exposes no device".into());
    }
    let aot = aot::global().ok().flatten().is_some_and(|store| !store.entries().is_empty());
    match nvrtc::version() {
        Ok(_) => Ok(cards),
        Err(_) if aot => Ok(cards),
        Err(e) => Err(format!("no NVRTC to compile kernels and no ahead-of-time kernel store ({e})")),
    }
}
