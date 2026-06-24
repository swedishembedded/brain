// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WGSL -> SPIR-V via `naga`, and `vk::ShaderModule` creation.
//!
//! WGSL remains the source of truth for every scalar kernel: we reuse the exact
//! same `src/shaders/*.wgsl` text the wgpu path uses, run it through
//! `naga::front::wgsl` -> `naga::back::spv` here, and hand the resulting SPIR-V
//! to ash. naga maps each WGSL `@group(N) @binding(M)` straight to a SPIR-V
//! `DescriptorSet = N, Binding = M` decoration, so the descriptor-set layout we
//! build in `matmul.rs` (set 0: uniform at 0, storage at 1..) lines up with the
//! kernel by construction.

use ash::vk;

/// Compile a WGSL compute kernel string to SPIR-V words.
///
/// Returns `Err` with a human-readable message on parse/validate/emit failure.
pub fn wgsl_to_spirv(src: &str) -> Result<Vec<u32>, String> {
    // Front end: WGSL -> naga IR.
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| format!("WGSL parse error: {e:?}"))?;

    // Validate (required to produce the ModuleInfo the SPV backend needs).
    // Allow the full capability set so e.g. f16/subgroup kernels would pass;
    // our scalar matmul needs none of them.
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error: {e:?}"))?;

    // Back end: naga IR -> SPIR-V. Target Vulkan 1.1-class SPIR-V 1.3 so the
    // module is consumable by a 1.3 device. Default Options maps group/binding
    // -> descriptor set/binding directly (empty binding_map + fake bindings).
    let mut options = naga::back::spv::Options::default();
    options.lang_version = (1, 3);
    // Strip debug info regardless of build profile for a lean module.
    options.flags.remove(naga::back::spv::WriterFlags::DEBUG);

    let words = naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("SPIR-V emit error: {e:?}"))?;
    Ok(words)
}

/// Create a `vk::ShaderModule` from SPIR-V words.
///
/// # Safety
/// `device` must outlive the returned module; the caller owns it and must call
/// `destroy_shader_module`.
pub unsafe fn make_shader_module(
    device: &ash::Device,
    spirv: &[u32],
) -> Result<vk::ShaderModule, String> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    device
        .create_shader_module(&info, None)
        .map_err(|e| format!("create_shader_module failed: {e}"))
}
