// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The compiled cooperative-matrix SPIR-V blob, and nothing else.
//!
//! **M8.9 (kernel-performance.md): the pipeline creation/dispatch logic that
//! used to live in this file (`MatmulBackend`, `matmul`, `scalar_matmul`,
//! `coopmat_matmul`, `build_pipeline`/`destroy_pipeline`, the standalone
//! `cooperative_matmul_demo`) moved to `crates/backend-vulkan/src/coopmat.rs`.**
//! Every one of those functions built its own, SEPARATE `VkContext` (a second
//! logical device) rather than sharing the real Vulkan backend's - which
//! meant a coopmat dispatch forfeited M6.1's per-buffer dependency tracking
//! and M6.2's asynchronous submission entirely (both live on
//! `backend-vulkan`'s device, not a second one this crate opened for a demo).
//! Moving the pipeline machinery onto `backend-vulkan`'s own `VkContext` via
//! `Backend::register_native`/`step_native` (M8.3's ABI) is what lets a
//! coopmat dispatch become a real `Step` on the caller's tape instead of an
//! out-of-band `upload; dispatch; fence-wait; download` round trip.
//!
//! What stays here: compiling `shaders_vk/matmul_coopmat.comp` to SPIR-V at
//! BUILD time (`build.rs`, GLSL -> SPIR-V via glslc/glslangValidator - no
//! device involved) and handing the compiled bytes to whoever registers them
//! (`backend_vulkan::coopmat::register`). `print_vk_info` (this crate's own
//! capability probe, [`crate::print_vk_info`]) still opens its own throwaway
//! `VkContext` - that is fine for a read-only capability query that dispatches
//! nothing.

/// Precompiled cooperative-matrix SPIR-V, present only when build.rs found a
/// GLSL compiler. `have_coopmat_spv` is set by build.rs in that case.
#[cfg(have_coopmat_spv)]
const COOPMAT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_coopmat.spv"));

/// The baked coopmat SPIR-V, or `None` if it was not compiled at build time
/// (no `glslc`/`glslangValidator` on `PATH` - see `build.rs`'s own doc).
pub fn coopmat_spv() -> Option<&'static [u8]> {
    #[cfg(have_coopmat_spv)]
    {
        Some(COOPMAT_SPV)
    }
    #[cfg(not(have_coopmat_spv))]
    {
        None
    }
}
