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
//! device queries. What it does NOT have is a hand-written kernel: every
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

pub mod backend;
pub mod driver;
pub mod exec;
pub mod nvrtc;
pub mod policy;

pub use backend::{CudaBackend, LaunchStats};
pub use driver::{driver, CudaDevice};

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
