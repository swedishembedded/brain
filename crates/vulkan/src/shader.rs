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

/// Whether kernels are compiled with naga's runtime checks, mirroring
/// `backend-wgpu`'s `ShaderRuntimeChecks` decision and reading the SAME
/// `BRAIN_GPU_CHECKED` switch, so the two GPU backends cannot silently compile
/// the same WGSL under different rules.
///
/// Off by default. The safety argument is the one `backend-wgpu` and
/// `backend-cpu` already make for the identical choice (`create_shader_module_
/// trusted` / Cranelift's `MemFlags::trusted()`): every kernel self-bounds on
/// its uniform (`if (idx >= total) { return; }`), every loop is counted by a
/// uniform field rather than by data, and buffer sizes are fixed by the model.
/// The device additionally enables `robustBufferAccess`, so an out-of-range
/// access is bounded by the hardware rather than being undefined - the same
/// backstop wgpu relies on when it selects `BoundsCheckPolicy::Unchecked`.
///
/// This is not a micro-optimisation. Leaving naga's defaults on cost this
/// backend HALF its arithmetic throughput on a Tesla P40: the fp32 FMA roofline
/// probe measured 5.05 TFLOP/s against the wgpu backend's 10.6 TFLOP/s from the
/// identical WGSL, and the packed-int8 probe 20.4 vs 43.2 TOP/s, with DRAM
/// bandwidth identical on both - the gap was entirely the per-iteration guard
/// counter and the per-access clamp that this backend alone was emitting.
fn runtime_checked() -> bool {
    std::env::var("BRAIN_GPU_CHECKED").map(|v| v != "0").unwrap_or(false)
}

fn bounds_check_policies() -> naga::proc::BoundsCheckPolicies {
    use naga::proc::BoundsCheckPolicy;
    let p = if runtime_checked() { BoundsCheckPolicy::Restrict } else { BoundsCheckPolicy::Unchecked };
    naga::proc::BoundsCheckPolicies { index: p, buffer: p, image_load: p, binding_array: p }
}

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
    let mut flags = naga::back::spv::Options::default().flags;
    // Strip debug info regardless of build profile for a lean module.
    flags.remove(naga::back::spv::WriterFlags::DEBUG);
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags,
        bounds_check_policies: bounds_check_policies(),
        // naga defaults this ON, which wraps every loop in a 64-bit
        // guard counter decremented per iteration. See `runtime_checked`.
        force_loop_bounding: runtime_checked(),
        ..Default::default()
    };

    let words = naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("SPIR-V emit error: {e:?}"))?;
    Ok(words)
}

/// Reflect a WGSL kernel's `@group(0)` resource bindings as `(binding, is_uniform)`
/// pairs, sorted by binding index. naga maps `@group(0) @binding(N)` to descriptor
/// set 0 binding N, so this is exactly the descriptor-set-layout a generic backend
/// must build to match the SPIR-V (uniform at binding 0, storage at 1..). Bindings
/// in groups other than 0 are ignored (brain kernels use a single bind group).
pub fn wgsl_bindings(src: &str) -> Result<Vec<(u32, bool)>, String> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| format!("WGSL parse error: {e:?}"))?;
    let mut out = Vec::new();
    for (_h, gv) in module.global_variables.iter() {
        if let Some(rb) = &gv.binding {
            if rb.group != 0 {
                continue;
            }
            let is_uniform = matches!(gv.space, naga::AddressSpace::Uniform);
            out.push((rb.binding, is_uniform));
        }
    }
    out.sort_by_key(|(b, _)| *b);
    Ok(out)
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
