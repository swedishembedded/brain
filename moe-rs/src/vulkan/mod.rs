// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native-Vulkan (ash) execution path, gated behind the `vulkan-coopmat`
//! Cargo feature. SEPARATE from the wgpu path in `crate::gpu` -- the default
//! build never compiles this module.
//!
//! What this provides (see the module docs for detail):
//!   * `context::VkContext` -- ash instance/device/queue + buffers + dispatch,
//!     and the `VK_KHR_cooperative_matrix` capability query.
//!   * `shader::wgsl_to_spirv` -- compile the existing WGSL kernels to SPIR-V
//!     via `naga` (WGSL stays the source of truth for scalar kernels).
//!   * `matmul` -- `out = x @ W^T` with a `MatmulBackend` that picks the NVIDIA
//!     tensor-core kernel (GLSL `matmul_coopmat.comp`, compiled by build.rs)
//!     when available, else the naga-compiled scalar `matmul.wgsl`.
//!
//! SCOPE: matmul + runtime + capability + fallback. Porting the full PID
//! forward pass to this runtime is a documented follow-up (README_VULKAN.md).
//!
//! HARDWARE NOTE: developed against software Vulkan (llvmpipe), which lacks
//! cooperative matrix, so only the scalar fallback executes here. The
//! tensor-core path is correct-by-construction and must be validated on NVIDIA
//! hardware (Turing sm_75+ for f16; Pascal sm_61 has no coop matrix and falls
//! back to scalar).

pub mod context;
pub mod matmul;
pub mod shader;

// Re-exported as the module's public surface (used by the CLI entries below and
// intended for the documented follow-up forward port). Not all are referenced
// inside this crate yet.
#[allow(unused_imports)]
pub use context::{component_type_name, CoopMatCaps, CoopMatShape, VkContext};
#[allow(unused_imports)]
pub use matmul::{cooperative_matmul_demo, matmul, MatmulBackend};

/// Print the selected adapter and its cooperative-matrix capabilities. Backs
/// the `moe pid vk-info` CLI entry.
pub fn print_vk_info() {
    let ctx = match VkContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("vk-info: could not init Vulkan: {e}");
            return;
        }
    };
    println!("Vulkan adapter: {}", ctx.adapter_name);
    println!("VK_KHR_cooperative_matrix extension present: {}", ctx.caps.extension_present);
    println!("cooperativeMatrix feature enabled: {}", ctx.caps.feature_supported);
    println!("selected matmul backend: {:?}", MatmulBackend::select(&ctx));
    println!(
        "coopmat SPIR-V baked in at build time: {}",
        matmul::coopmat_spv().is_some()
    );

    if ctx.caps.shapes.is_empty() {
        println!("supported cooperative-matrix shapes: none");
        println!(
            "  (expected on llvmpipe / Pascal sm_61; tensor-core matmul will use \
             the scalar fallback)"
        );
    } else {
        println!("supported cooperative-matrix shapes (M x N x K  A*B->C/result  scope):");
        for s in &ctx.caps.shapes {
            println!(
                "  {:>3} x {:>3} x {:>3}   {}*{}->{}/{}   sat={}  scope={:?}",
                s.m,
                s.n,
                s.k,
                component_type_name(s.a_type),
                component_type_name(s.b_type),
                component_type_name(s.c_type),
                component_type_name(s.result_type),
                s.saturating_accumulation,
                s.scope,
            );
        }
        println!(
            "f16*f16->f32 tensor-core usable: {}",
            ctx.caps.supports_f16_tensorcore()
        );
    }
}
