// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cooperative-matrix (`VK_KHR_cooperative_matrix`) pipeline: creation
//! from raw SPIR-V (`Backend::register_native`'s implementation) and the
//! host-side f16 packing a caller needs to feed it (M8.9,
//! `kernel-performance.md`).
//!
//! **Why this lives here and not in `crates/vulkan`.** `crates/vulkan`'s own
//! `VkContext` is a SEPARATE Vulkan device from `backend-vulkan`'s - the real,
//! always-on backend every model dispatch actually goes through (M6.1's
//! per-buffer dependency tracking, M6.2's asynchronous submission via a
//! shared timeline semaphore, per-kernel device timing). A pipeline created
//! against `crates/vulkan`'s device could never be dispatched through any of
//! that; it would need its own upload/dispatch/fence-wait/download round trip
//! outside `Backend::step`/`submit`, forfeiting the tape-capture and
//! async-submission machinery entirely - exactly the failure mode
//! `Backend::register_native`'s own doc comment warns about. Building the
//! pipeline here, against `self.ctx` (the SAME `VkContext` every WGSL
//! dispatch already shares), is what lets [`VulkanBackend::step_native`]
//! record an ordinary [`VkStep`] that `flush`/`submit` handle exactly like
//! any WGSL one.
//!
//! Swedish Embedded AB implements solutions for bringing a from-scratch GPU
//! kernel (Vulkan cooperative matrix, custom SPIR-V, or otherwise) onto a
//! production async-submission pipeline without forking the device it runs
//! against. If your team needs this kind of low-level GPU integration work,
//! you can reach us at info@swedishembedded.com.

use ash::vk;

use backend_api::{BindKind, NativeSpec};

use vulkan::context::VkContext;

/// Cooperative-matrix tile size `matmul_coopmat.comp` was authored against
/// (must match the `.comp`'s own `TILE_M`/`TILE_N`/`TILE_K` consts).
pub const TILE: u32 = 16;

/// Round `v` up to the next multiple of `m` - the coopmat kernel assumes
/// tile-aligned `M`/`N`/`K` extents (see the `.comp`'s own doc comment); a
/// caller pads to this before packing.
pub fn round_up(v: u32, m: u32) -> u32 {
    v.div_ceil(m) * m
}

/// Whether `m`/`n`/`k` are already tile-aligned - the shape gate
/// `gpu_core::provider::coopmat::CoopMatProvider::accepts` uses to decline a
/// shape it would otherwise have to silently pad (padding belongs to a real
/// caller that KNOWS it is feeding this kernel, not to a provider guessing at
/// scratch ownership).
pub fn is_tile_aligned(m: u32, n: u32, k: u32) -> bool {
    m.is_multiple_of(TILE) && n.is_multiple_of(TILE) && k.is_multiple_of(TILE)
}

/// Pack `x` to IEEE-754 binary16 bits, round-to-nearest-even, correct for
/// subnormals and infinities alike.
///
/// **The bug this replaces.** The pre-M8.9 hand-rolled version
/// (`vulkan::matmul::f32_to_f16_bits`) flushed every subnormal-in-f16-range
/// f32 value (`exp <= 0` after rebiasing, i.e. magnitude below f16's smallest
/// normal, ~6.1e-5) to a SIGNED ZERO instead of an f16 subnormal or a
/// correctly-rounded normal - silently discarding real precision a weight
/// tensor's small values legitimately carry, and doing so QUIETLY (no
/// truncation warning, just a wrong number). `half::f16::from_f32` is the
/// same crate `backend-wgpu`/`model`/`gguf`/`checkpoint`/`ltxv`/`gpu-core`
/// already depend on for this exact conversion elsewhere in this workspace
/// (checked before adding a new dependency here, per this milestone's own
/// brief) - reusing it fixes the bug by construction rather than hand-fixing
/// the bit-twiddling a second time.
pub fn pack_f16(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

/// Pack `x[M,K]` (row-major, arbitrary `M`/`K`) into a `[round_up(M,TILE),
/// round_up(K,TILE)]` row-major f16 buffer, zero-padded. Shared by the demo
/// and by a real provider's own operand-repacking path.
pub fn pack_padded_f16(x: &[f32], m: u32, k: u32) -> (Vec<u16>, u32, u32) {
    let mp = round_up(m, TILE);
    let kp = round_up(k, TILE);
    let mut out = vec![0u16; (mp * kp) as usize];
    for r in 0..m {
        for c in 0..k {
            out[(r * kp + c) as usize] = pack_f16(x[(r * k + c) as usize]);
        }
    }
    (out, mp, kp)
}

/// One kernel registered via [`crate::VulkanBackend::register_native`]:
/// everything [`crate::VulkanBackend::resolve_kernel`] needs to dispatch it,
/// plus its own per-kernel profiling accumulator (kept alongside the
/// pipeline, not in the catalogue-sized `VkProfile::acc`, so registering a
/// kernel at runtime never needs to grow that Vec in lockstep with this one
/// under a SEPARATE lock - see [`crate::VulkanBackend::record_timing`]).
pub(crate) struct NativeEntry {
    pub(crate) module: vk::ShaderModule,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) bindings: Vec<vulkan::shader::WgslBinding>,
    /// The divisor `backend_api::grid_ws(threads, wgsize)` uses to turn a
    /// caller's `threads` into a workgroup count. A `NativeSpec` kernel has no
    /// reflected `@workgroup_size` for the generic engine to read, so this
    /// crate defines the convention itself: **`threads` for a native kernel
    /// IS the workgroup count directly** (`wgsize == 1`), never a per-
    /// invocation thread count divided by a workgroup size. Every
    /// `NativeEntry` this backend builds today uses that convention; a future
    /// native kernel authored differently would set this to whatever value
    /// makes `grid_ws` compute the workgroup count it actually wants.
    pub(crate) wgsize: u32,
    pub(crate) name: String,
    pub(crate) ms: f64,
    pub(crate) calls: u64,
}

impl NativeEntry {
    /// Destroy this entry's Vulkan objects. Called from
    /// `VulkanBackend`'s own `Drop` - native entries are per-handle (never
    /// `Arc`-shared the way the WGSL catalogue's `VkPipelineSet` is with a
    /// `share()` sibling), so there is exactly one owner and no refcount to
    /// check first.
    ///
    /// # Safety
    /// `dev` must be the device this entry's objects were created against,
    /// and must not have been destroyed yet.
    pub(crate) unsafe fn destroy(&self, dev: &ash::Device) {
        dev.destroy_pipeline(self.pipeline, None);
        dev.destroy_pipeline_layout(self.layout, None);
        dev.destroy_descriptor_set_layout(self.set_layout, None);
        dev.destroy_shader_module(self.module, None);
    }
}

/// Build a compute pipeline from `spec` against `ctx`'s device - the
/// implementation of [`crate::VulkanBackend::register_native`].
///
/// Returns `Err` (never panics) on any REAL failure (bad SPIR-V, layout/
/// pipeline creation rejected by the driver) - `register_native`'s own doc
/// contract is `None` in that case, "true of every backend today" already
/// covers a backend with no native path at all; this is the same contract
/// one level down, where the backend DOES have a native path but the
/// specific device rejects the specific kernel outright.
///
/// **What this function does NOT prove, measured on real hardware.** A
/// device without the capability this SPIR-V needs does not reliably reject
/// pipeline creation here: on this workspace's own sandbox (an Intel ANV
/// iGPU, no `VK_KHR_cooperative_matrix` shapes, no validation layer active),
/// `vkCreateComputePipelines` for the real coopmat SPIR-V SUCCEEDS - the
/// driver accepts a pipeline it may not correctly execute. So this function
/// returning `Ok` is NOT proof the device can run the kernel correctly; the
/// authoritative gate is `Requirement.matrix` (checked by
/// `gpu_core::provider::ProviderRegistry::resolve` BEFORE a real caller ever
/// reaches `register_native`), not whether pipeline creation happened to
/// succeed. See `crates/gpu-core/tests/coopmat_provider_declines_on_this_box.rs`
/// for the measurement.
pub(crate) fn build_pipeline(ctx: &VkContext, spec: &NativeSpec) -> Result<NativeEntry, String> {
    let NativeSpec::SpirV { code, entry, bindings } = spec else {
        // `HostFn` is a CPU-ISA-pack provider's shape, not this GPU backend's -
        // see `NativeSpec::HostFn`'s own doc ("the backend that accepts this
        // decides how it actually runs").
        return Err("VulkanBackend::register_native: NativeSpec::HostFn has no GPU dispatch path".to_string());
    };
    if !code.len().is_multiple_of(4) {
        return Err(format!("SPIR-V blob is not word-aligned ({} bytes)", code.len()));
    }
    let words: Vec<u32> = code.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let wgsl_bindings: Vec<vulkan::shader::WgslBinding> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| vulkan::shader::WgslBinding {
            binding: i as u32,
            is_uniform: matches!(b, BindKind::Uniform),
            is_write: matches!(b, BindKind::StorageReadWrite),
        })
        .collect();
    let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = wgsl_bindings
        .iter()
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b.binding)
                .descriptor_type(if b.is_uniform { vk::DescriptorType::UNIFORM_BUFFER } else { vk::DescriptorType::STORAGE_BUFFER })
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();

    let dev = &ctx.device;
    unsafe {
        let module = vulkan::shader::make_shader_module(dev, &words)?;
        let set_layout = match dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings), None) {
            Ok(l) => l,
            Err(e) => {
                dev.destroy_shader_module(module, None);
                return Err(format!("descriptor set layout: {e}"));
            }
        };
        let set_layouts = [set_layout];
        let layout = match dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None) {
            Ok(l) => l,
            Err(e) => {
                dev.destroy_descriptor_set_layout(set_layout, None);
                dev.destroy_shader_module(module, None);
                return Err(format!("pipeline layout: {e}"));
            }
        };
        let entry_c = std::ffi::CString::new(*entry).map_err(|_| "entry point name contains a NUL byte".to_string())?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_c);
        let pipeline = match dev.create_compute_pipelines(
            ctx.pipeline_cache(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
            None,
        ) {
            Ok(p) => p[0],
            Err((_, e)) => {
                dev.destroy_pipeline_layout(layout, None);
                dev.destroy_descriptor_set_layout(set_layout, None);
                dev.destroy_shader_module(module, None);
                return Err(format!(
                    "compute pipeline declined by this device (expected on hardware without the \
                     capability this SPIR-V needs): {e}"
                ));
            }
        };
        Ok(NativeEntry {
            module,
            set_layout,
            layout,
            pipeline,
            bindings: wgsl_bindings,
            wgsize: 1,
            name: format!("native-spirv#{entry}"),
            ms: 0.0,
            calls: 0,
        })
    }
}

/// The coopmat kernel's binding layout, in bind order (binding 0 = the
/// `Params{m,k,n}` uniform, matching every other kernel's convention in this
/// crate; bindings 1/2 = the packed-f16 `x`/`w` operands; binding 3 = the f32
/// output).
pub const BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// The [`NativeSpec`] for `matmul_coopmat.comp` (f16 x f16 -> f32, 16x16x16
/// subgroup tiles) - `None` when no GLSL compiler was on `PATH` at build time
/// (`vulkan::matmul::coopmat_spv`'s own doc). Entry point is always `"main"`
/// (the `.comp`'s own `void main()`).
pub fn spec() -> Option<NativeSpec> {
    vulkan::matmul::coopmat_spv().map(|code| NativeSpec::SpirV { code, entry: "main", bindings: BINDINGS })
}

/// `brain toypid vk-matmul` (behind the `vulkan-coopmat` CLI feature): register
/// the coopmat kernel on a REAL [`crate::VulkanBackend`] (the same device/
/// dependency-tracking/async-submission path a served model dispatches
/// through - never a second, throwaway device) and run one small tile-aligned
/// GEMM through it, or report exactly why the device declined.
pub fn demo() {
    let backend = match crate::VulkanBackend::try_new(&[]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("vk-matmul: could not open the native Vulkan backend: {e}");
            return;
        }
    };
    println!("adapter: native Vulkan backend (shared VkContext, not a second device)");
    let Some(spec) = spec() else {
        println!("vk-matmul: no coopmat SPIR-V baked in at build time (no glslc/glslangValidator on PATH)");
        return;
    };
    let Some(id) = backend.register_native(&spec) else {
        println!(
            "vk-matmul: this device declined the coopmat pipeline (expected without \
             VK_KHR_cooperative_matrix + a usable f16x f16->f32 shape - e.g. Pascal sm_61 or \
             an Intel iGPU). The scalar WGSL matmul kernel is what every real dispatch uses \
             on this hardware instead."
        );
        return;
    };

    // out = x @ W^T, M=N=K=32 (2x2 tiles): x[r,c] = (r+c)*0.01, W = I so out == x.
    let (m, k, n) = (32u32, 32u32, 32u32);
    let mut x = vec![0f32; (m * k) as usize];
    for r in 0..m {
        for c in 0..k {
            x[(r * k + c) as usize] = ((r + c) as f32) * 0.01;
        }
    }
    let mut w = vec![0f32; (n * k) as usize];
    for i in 0..n.min(k) {
        w[(i * k + i) as usize] = 1.0;
    }
    let (xf, mp, kp) = pack_padded_f16(&x, m, k);
    let (wf, np, kp2) = pack_padded_f16(&w, n, k);
    debug_assert_eq!(kp, kp2);

    // `storage(n)` allocates `n` f32-WORDS; `xf`/`wf` are packed f16 (2 bytes
    // each), so their word count is half their element count (always exact -
    // `mp`/`kp`/`np` are TILE=16 multiples, so the byte length is always a
    // multiple of 4). `write` takes raw `u32` words and uploads their bytes
    // verbatim - `bytemuck::cast_slice` reinterprets the `u16` pairs as `u32`s
    // without touching a single bit, which is exactly the raw byte upload a
    // packed-f16 buffer needs (never a numeric f16->f32 conversion).
    let xbuf = backend.storage((xf.len() / 2) as u64);
    backend.write(&xbuf, bytemuck::cast_slice(&xf));
    let wbuf = backend.storage((wf.len() / 2) as u64);
    backend.write(&wbuf, bytemuck::cast_slice(&wf));
    let obuf = backend.storage((mp * np) as u64);

    let tiles = (mp / TILE) * (np / TILE);
    let params = [mp, kp, np];
    let step = backend.step_native(id, &[&xbuf, &wbuf, &obuf], &params, tiles).expect("id was just registered");
    backend.submit(&[], &[step]);
    let padded = backend.read(&obuf, (mp * np) as usize);
    let mut out = vec![0f32; (m * n) as usize];
    for r in 0..m {
        for c in 0..n {
            out[(r * n + c) as usize] = padded[(r * np + c) as usize];
        }
    }
    println!("out[0..6]      = {:?}", &out[0..6]);
    println!("expected[0..6] = {:?}", &x[0..6]);
    let max_err = out.iter().zip(x.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("max abs error vs identity-matmul = {max_err:.3e} (expect ~1e-2 from f16 rounding)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_pads_to_the_next_tile_multiple() {
        assert_eq!(round_up(0, TILE), 0);
        assert_eq!(round_up(1, TILE), TILE);
        assert_eq!(round_up(TILE, TILE), TILE);
        assert_eq!(round_up(TILE + 1, TILE), 2 * TILE);
    }

    #[test]
    fn is_tile_aligned_rejects_any_non_multiple_dimension() {
        assert!(is_tile_aligned(32, 32, 32));
        assert!(!is_tile_aligned(33, 32, 32));
        assert!(!is_tile_aligned(32, 33, 32));
        assert!(!is_tile_aligned(32, 32, 33));
    }

    /// **The real precision bug this milestone fixes.** The pre-M8.9 hand-
    /// rolled `f32_to_f16_bits` flushed every subnormal-in-f16-range f32
    /// value (rebiased exponent `<= 0`, i.e. magnitude below f16's smallest
    /// NORMAL, `2^-14`) to a signed zero - discarding real, representable
    /// precision. `2^-15` is exactly half of `2^-14`: not representable as an
    /// f16 NORMAL, but exactly representable as an f16 SUBNORMAL (one bit of
    /// mantissa), so a correct round-to-nearest-even conversion must return
    /// it EXACTLY, not zero.
    #[test]
    fn pack_f16_preserves_an_f16_subnormal_the_old_conversion_flushed_to_zero() {
        let x = 2f32.powi(-15);
        let bits = pack_f16(x);
        assert_ne!(bits, 0, "a real f16-representable subnormal must not flush to signed zero");
        assert_eq!(half::f16::from_bits(bits).to_f32(), x, "must round-trip exactly at this magnitude");
    }

    /// The smallest positive f16 subnormal (`2^-24`) must also survive -
    /// the old conversion's `exp <= 0` branch returned `sign` (zero) for
    /// this and everything below it too, with no distinction from a value
    /// that actually underflows f16 entirely (anything `< 2^-24`, which
    /// legitimately rounds to zero).
    #[test]
    fn pack_f16_preserves_the_smallest_f16_subnormal() {
        let smallest_subnormal = 2f32.powi(-24);
        let bits = pack_f16(smallest_subnormal);
        assert_ne!(bits, 0);
        assert_eq!(half::f16::from_bits(bits).to_f32(), smallest_subnormal);
    }

    /// An ordinary normal value still round-trips (this was never broken -
    /// pinned so a future change to `pack_f16` cannot regress the common
    /// case while "fixing" the subnormal one).
    #[test]
    fn pack_f16_round_trips_an_ordinary_normal_value() {
        let x = 1.5f32;
        assert_eq!(half::f16::from_bits(pack_f16(x)).to_f32(), x);
    }

    #[test]
    fn pack_padded_f16_zero_pads_and_preserves_the_real_elements() {
        // 2x3, not tile-aligned -> pads to 16x16, zeros elsewhere.
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (packed, mp, kp) = pack_padded_f16(&x, 2, 3);
        assert_eq!((mp, kp), (TILE, TILE));
        assert_eq!(packed.len(), (TILE * TILE) as usize);
        for r in 0..2u32 {
            for c in 0..3u32 {
                let v = half::f16::from_bits(packed[(r * kp + c) as usize]).to_f32();
                assert_eq!(v, x[(r * 3 + c) as usize]);
            }
        }
        // Padding stays zero.
        assert_eq!(packed[3], 0, "row 0, col 3 is padding (only 3 real cols)");
        assert_eq!(packed[(2 * kp) as usize], 0, "row 2 is padding (only 2 real rows)");
    }
}
