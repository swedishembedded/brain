// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Matmul backend selection (tensor-core vs scalar) on the ash runtime, plus a
//! self-contained smoke demo.
//!
//! Backend choice (`MatmulBackend::select`):
//!   * `Coopmat` -- iff the device reports a usable f16xf16->f32 cooperative-
//!     matrix shape AND a precompiled `matmul_coopmat.spv` is baked in (the
//!     build.rs only sets `have_coopmat_spv` when glslc/glslang produced it).
//!   * `Scalar` -- otherwise: the WGSL `matmul.wgsl`, compiled to SPIR-V by
//!     naga at runtime. This is what runs on llvmpipe here and on Pascal sm_61.
//!
//! Both backends share the descriptor layout (set 0: uniform Params{m,k,n} at
//! binding 0, storage x/w/out at bindings 1/2/3) and the `out = x @ W^T`
//! semantics, so the host code below is backend-agnostic apart from input dtype
//! (f32 for scalar, f16-packed for coopmat) and the dispatch grid.

use ash::vk;

use super::context::VkContext;
use super::shader;

const MATMUL_WGSL: &str = kernels::MATMUL;

/// Precompiled cooperative-matrix SPIR-V, present only when build.rs found a
/// GLSL compiler. `have_coopmat_spv` is set by build.rs in that case.
#[cfg(have_coopmat_spv)]
const COOPMAT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_coopmat.spv"));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulBackend {
    /// NVIDIA tensor-core kernel (f16xf16->f32). Validate on hardware.
    Coopmat,
    /// Scalar fp32 fallback (WGSL via naga). Runs anywhere.
    Scalar,
}

impl MatmulBackend {
    /// Decide which backend to use for `ctx`.
    pub fn select(ctx: &VkContext) -> MatmulBackend {
        if coopmat_spv().is_some() && ctx.caps.supports_f16_tensorcore() {
            MatmulBackend::Coopmat
        } else {
            MatmulBackend::Scalar
        }
    }
}

/// The baked coopmat SPIR-V, or `None` if it was not compiled at build time.
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

/// Cooperative-matrix tile size the GLSL kernel uses (must match the .comp).
const TILE: u32 = 16;

/// Pack f32 -> IEEE-754 binary16 (round-to-nearest-even, minimal). Used only to
/// feed the coopmat kernel; the scalar kernel takes f32 directly.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        // subnormal/zero -> flush to signed zero (adequate for the demo).
        sign
    } else if exp >= 0x1f {
        sign | 0x7c00 // inf/overflow
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

/// Round `v` up to the next multiple of `m`.
fn round_up(v: u32, m: u32) -> u32 {
    v.div_ceil(m) * m
}

/// Run `out = x @ W^T` on the ash runtime with the selected backend and read
/// the result back. Shapes: x[M,K], W[N,K], out[M,N]. Returns row-major out.
///
/// For the coopmat backend, M/N/K are padded up to multiples of TILE (the
/// kernel assumes tile-aligned extents); the padding rows/cols are dropped on
/// read-back. The scalar backend handles arbitrary extents directly.
pub fn matmul(
    ctx: &VkContext,
    backend: MatmulBackend,
    x: &[f32],
    w: &[f32],
    m: u32,
    k: u32,
    n: u32,
) -> Vec<f32> {
    assert_eq!(x.len(), (m * k) as usize, "x shape");
    assert_eq!(w.len(), (n * k) as usize, "w shape");

    match backend {
        MatmulBackend::Scalar => scalar_matmul(ctx, x, w, m, k, n),
        MatmulBackend::Coopmat => coopmat_matmul(ctx, x, w, m, k, n),
    }
}

// ---- shared pipeline plumbing ----

struct Pipeline {
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

/// Build a compute pipeline with set 0 = [uniform@0, storage@1, storage@2,
/// storage@3] and allocate one descriptor set bound to the given buffers.
unsafe fn build_pipeline(
    ctx: &VkContext,
    spirv: &[u32],
    ubuf: &super::context::VkBuffer,
    x: &super::context::VkBuffer,
    w: &super::context::VkBuffer,
    out: &super::context::VkBuffer,
) -> Pipeline {
    let dev = &ctx.device;
    let module = shader::make_shader_module(dev, spirv).expect("shader module");

    let bindings = [
        descriptor(0, vk::DescriptorType::UNIFORM_BUFFER),
        descriptor(1, vk::DescriptorType::STORAGE_BUFFER),
        descriptor(2, vk::DescriptorType::STORAGE_BUFFER),
        descriptor(3, vk::DescriptorType::STORAGE_BUFFER),
    ];
    let set_layout = dev
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
        .expect("set layout");

    let set_layouts = [set_layout];
    let layout = dev
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
            None,
        )
        .expect("pipeline layout");

    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let pipeline = dev
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        )
        .expect("compute pipeline")[0];

    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3),
    ];
    let pool = dev
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
        .expect("descriptor pool");
    let set = dev
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts),
        )
        .expect("descriptor set")[0];

    // Wire the four buffers to bindings 0..3.
    let u_info = [vk::DescriptorBufferInfo::default().buffer(ubuf.buffer).range(ubuf.size)];
    let x_info = [vk::DescriptorBufferInfo::default().buffer(x.buffer).range(x.size)];
    let w_info = [vk::DescriptorBufferInfo::default().buffer(w.buffer).range(w.size)];
    let o_info = [vk::DescriptorBufferInfo::default().buffer(out.buffer).range(out.size)];
    let writes = [
        write_buf(set, 0, vk::DescriptorType::UNIFORM_BUFFER, &u_info),
        write_buf(set, 1, vk::DescriptorType::STORAGE_BUFFER, &x_info),
        write_buf(set, 2, vk::DescriptorType::STORAGE_BUFFER, &w_info),
        write_buf(set, 3, vk::DescriptorType::STORAGE_BUFFER, &o_info),
    ];
    dev.update_descriptor_sets(&writes, &[]);

    Pipeline {
        module,
        layout,
        set_layout,
        pipeline,
        pool,
        set,
    }
}

unsafe fn destroy_pipeline(ctx: &VkContext, p: Pipeline) {
    let dev = &ctx.device;
    dev.destroy_pipeline(p.pipeline, None);
    dev.destroy_pipeline_layout(p.layout, None);
    dev.destroy_descriptor_set_layout(p.set_layout, None);
    dev.destroy_descriptor_pool(p.pool, None);
    dev.destroy_shader_module(p.module, None);
}

fn descriptor(binding: u32, ty: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn write_buf<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    info: &'a [vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty)
        .buffer_info(info)
}

/// Params{m,k,n} uniform, std140-padded to 16 bytes (matches Gpu::uniform).
fn params_bytes(m: u32, k: u32, n: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&m.to_ne_bytes());
    v.extend_from_slice(&k.to_ne_bytes());
    v.extend_from_slice(&n.to_ne_bytes());
    v.extend_from_slice(&0u32.to_ne_bytes());
    v
}

// ---- scalar backend (WGSL matmul.wgsl via naga) ----

fn scalar_matmul(ctx: &VkContext, x: &[f32], w: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
    let spirv = shader::wgsl_to_spirv(MATMUL_WGSL).expect("naga compile matmul.wgsl");

    let ubuf = ctx.storage(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
    ctx.upload(&ubuf, &params_bytes(m, k, n));
    let xbuf = ctx.storage((x.len() * 4) as u64, vk::BufferUsageFlags::empty());
    ctx.upload(&xbuf, bytemuck::cast_slice(x));
    let wbuf = ctx.storage((w.len() * 4) as u64, vk::BufferUsageFlags::empty());
    ctx.upload(&wbuf, bytemuck::cast_slice(w));
    let total = (m * n) as usize;
    let obuf = ctx.storage((total * 4) as u64, vk::BufferUsageFlags::empty());

    unsafe {
        let p = build_pipeline(ctx, &spirv, &ubuf, &xbuf, &wbuf, &obuf);
        // WGSL matmul: workgroup_size(64), one thread per output element.
        let groups = (total as u32).div_ceil(64).max(1);
        ctx.dispatch(p.pipeline, p.layout, p.set, (groups, 1, 1));
        destroy_pipeline(ctx, p);
    }

    let bytes = ctx.download(&obuf, total * 4);
    let out = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();

    ctx.destroy_buffer(ubuf);
    ctx.destroy_buffer(xbuf);
    ctx.destroy_buffer(wbuf);
    ctx.destroy_buffer(obuf);
    out
}

// ---- cooperative-matrix backend (GLSL matmul_coopmat.comp) ----

fn coopmat_matmul(ctx: &VkContext, x: &[f32], w: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
    let spv_bytes = coopmat_spv().expect("coopmat backend selected without baked SPIR-V");
    // SPIR-V is little-endian u32 words.
    assert!(spv_bytes.len() % 4 == 0, "coopmat spv not word-aligned");
    let spirv: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Pad extents to TILE multiples; the kernel iterates tile-aligned ranges.
    let mp = round_up(m, TILE);
    let kp = round_up(k, TILE);
    let np = round_up(n, TILE);

    // Repack inputs into padded f16 buffers (zeros in the padding).
    let mut xf = vec![0u16; (mp * kp) as usize];
    for r in 0..m {
        for c in 0..k {
            xf[(r * kp + c) as usize] = f32_to_f16_bits(x[(r * k + c) as usize]);
        }
    }
    let mut wf = vec![0u16; (np * kp) as usize];
    for r in 0..n {
        for c in 0..k {
            wf[(r * kp + c) as usize] = f32_to_f16_bits(w[(r * k + c) as usize]);
        }
    }

    let ubuf = ctx.storage(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
    ctx.upload(&ubuf, &params_bytes(mp, kp, np));
    let xbuf = ctx.storage((xf.len() * 2) as u64, vk::BufferUsageFlags::empty());
    ctx.upload(&xbuf, bytemuck::cast_slice(&xf));
    let wbuf = ctx.storage((wf.len() * 2) as u64, vk::BufferUsageFlags::empty());
    ctx.upload(&wbuf, bytemuck::cast_slice(&wf));
    let padded_out = (mp * np) as usize;
    let obuf = ctx.storage((padded_out * 4) as u64, vk::BufferUsageFlags::empty());

    unsafe {
        let p = build_pipeline(ctx, &spirv, &ubuf, &xbuf, &wbuf, &obuf);
        // One workgroup per TILE_M x TILE_N output tile.
        ctx.dispatch(p.pipeline, p.layout, p.set, (mp / TILE, np / TILE, 1));
        destroy_pipeline(ctx, p);
    }

    let bytes = ctx.download(&obuf, padded_out * 4);
    let padded = bytemuck::cast_slice::<u8, f32>(&bytes);
    // Drop padding back to [M, N].
    let mut out = vec![0f32; (m * n) as usize];
    for r in 0..m {
        for c in 0..n {
            out[(r * n + c) as usize] = padded[(r * np + c) as usize];
        }
    }
    ctx.destroy_buffer(ubuf);
    ctx.destroy_buffer(xbuf);
    ctx.destroy_buffer(wbuf);
    ctx.destroy_buffer(obuf);
    out
}

/// Smoke entry: allocate two small matrices, run whichever backend is selected,
/// read the result back, and print a summary. On this machine (llvmpipe) it
/// runs the scalar backend; on NVIDIA Turing+ it runs the tensor-core kernel.
pub fn cooperative_matmul_demo() {
    let ctx = match VkContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("vk-matmul: could not init Vulkan: {e}");
            return;
        }
    };
    let backend = MatmulBackend::select(&ctx);
    println!("adapter: {}", ctx.adapter_name);
    println!("matmul backend: {backend:?}");

    // out = x @ W^T, with M=N=K=32 (>= one 16x16x16 tile in every dim).
    let (m, k, n) = (32u32, 32u32, 32u32);
    // x[r,c] = (r+c) scaled small; W = identity-ish so out is easy to eyeball.
    let mut x = vec![0f32; (m * k) as usize];
    for r in 0..m {
        for c in 0..k {
            x[(r * k + c) as usize] = ((r + c) as f32) * 0.01;
        }
    }
    // W = I_n (n == k here): out = x @ I^T = x.
    let mut w = vec![0f32; (n * k) as usize];
    for i in 0..n.min(k) {
        w[(i * k + i) as usize] = 1.0;
    }

    let out = matmul(&ctx, backend, &x, &w, m, k, n);
    println!("out[0..6]      = {:?}", &out[0..6]);
    println!("expected[0..6] = {:?}", &x[0..6]);
    let max_err = out
        .iter()
        .zip(x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max abs error vs identity-matmul = {max_err:.3e}");
    match backend {
        MatmulBackend::Scalar => {
            println!("(scalar fp32 path -- exact; this is the fallback on llvmpipe / Pascal)")
        }
        MatmulBackend::Coopmat => println!(
            "(tensor-core f16 path -- expect ~1e-2 error from f16 rounding; \
             validate numerics on NVIDIA hardware)"
        ),
    }
}
