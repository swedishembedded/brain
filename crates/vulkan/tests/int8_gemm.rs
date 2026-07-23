// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 DP4A inference GEMM — correctness (quantized parity) + speed vs fp32.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-vulkan --test int8_gemm -- --ignored --nocapture
//! ```
//!
//! Answers the two questions the fp32 GEMM work answered, for the INT8 path:
//!
//! * **correct** — the DP4A kernel's dequantized output matches an *exact* int32
//!   reference computed on the host from the same quantized bytes (fp32 rounding
//!   only), and it matches the original fp32 matmul to within int8 quantization
//!   error (cosine ≥ 0.999 on well-conditioned data), and
//! * **fast** — GOP/s and the speedup over the fp32 software-pipelined
//!   `matmul_reg2`, on identical M/K/N.
//!
//! Runs on the raw `VkContext` (which enables `shaderIntegerDotProduct`), the
//! same path the peak-throughput bench used to reach 43.8 TOPS of DP4A.

use ash::vk;
use std::time::Instant;

use vulkan::context::{VkBuffer, VkContext};
use vulkan::shader;

const I8: &str = kernels::MATMUL_I8;
const F32: &str = kernels::MATMUL_REG2;

struct Shape { label: &'static str, m: usize, k: usize, n: usize }

const SHAPES: &[Shape] = &[
    // Qwen3-TTS Talker 0.6B dominant linears (prefill T=256):
    Shape { label: "tts talker q_proj 256x1024->2048", m: 256, k: 1024, n: 2048 },
    Shape { label: "tts talker ffn-up 256x1024->3072", m: 256, k: 1024, n: 3072 },
    Shape { label: "tts talker ffn-dn 256x3072->1024", m: 256, k: 3072, n: 1024 },
    Shape { label: "glm ffn           512x6144->2048", m: 512, k: 6144, n: 2048 },
    Shape { label: "square 2048",                       m: 2048, k: 2048, n: 2048 },
];

/// A compute pipeline with `nstorage` storage bindings after the uniform.
struct Pipe {
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

unsafe fn build(ctx: &VkContext, spirv: &[u32], ubuf: &VkBuffer, storages: &[&VkBuffer]) -> Pipe {
    let dev = &ctx.device;
    let module = dev
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
        .expect("module");
    let mut bindings = vec![vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    for i in 0..storages.len() {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding((i + 1) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
    }
    let set_layout = dev
        .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
        .expect("set layout");
    let set_layouts = [set_layout];
    let layout = dev
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)
        .expect("layout");
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry);
    let pipeline = dev
        .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
        .expect("pipeline")[0];
    let pool_sizes = [
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(storages.len() as u32),
    ];
    let pool = dev
        .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None)
        .expect("pool");
    let set = dev
        .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts))
        .expect("set")[0];
    let u_info = [vk::DescriptorBufferInfo::default().buffer(ubuf.buffer).range(ubuf.size)];
    let mut writes = vec![vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u_info)];
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = storages
        .iter()
        .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).range(b.size)])
        .collect();
    for (i, info) in infos.iter().enumerate() {
        writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding((i + 1) as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info));
    }
    dev.update_descriptor_sets(&writes, &[]);
    Pipe { module, layout, set_layout, pipeline, pool, set }
}

unsafe fn destroy(ctx: &VkContext, p: Pipe) {
    let d = &ctx.device;
    d.destroy_pipeline(p.pipeline, None);
    d.destroy_pipeline_layout(p.layout, None);
    d.destroy_descriptor_set_layout(p.set_layout, None);
    d.destroy_descriptor_pool(p.pool, None);
    d.destroy_shader_module(p.module, None);
}

fn bytes_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() }

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 17) % 97) as f32 / 97.0) - 0.5).collect()
}

/// Per-tensor symmetric int8 quantization: returns (int8 as i32, scale).
fn quantize(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(1e-8f32, |m, &v| m.max(v.abs()));
    let s = amax / 127.0;
    (x.iter().map(|&v| (v / s).round().clamp(-127.0, 127.0) as i8).collect(), s)
}

/// Pack a [rows, K] int8 matrix into [rows, K/4] u32 (4 int8 per u32, LE).
fn pack(q: &[i8], rows: usize, k: usize) -> Vec<u32> {
    let kg = k / 4;
    let mut out = vec![0u32; rows * kg];
    for r in 0..rows {
        for g in 0..kg {
            let mut w = 0u32;
            for b in 0..4 {
                w |= ((q[r * k + g * 4 + b] as u8) as u32) << (8 * b);
            }
            out[r * kg + g] = w;
        }
    }
    out
}

fn reg_groups(m: usize, n: usize) -> u32 { (m.div_ceil(128) * n.div_ceil(128)) as u32 }

/// std140 uniform for matmul_i8 Params{m,kg,n,sx,sw} padded to 32 B.
fn uni_i8(m: u32, kg: u32, n: u32, sx: f32, sw: f32) -> Vec<u8> {
    let mut v = Vec::new();
    for w in [m, kg, n, sx.to_bits(), sw.to_bits(), 0, 0, 0] { v.extend_from_slice(&w.to_ne_bytes()); }
    v
}
fn uni_f32(m: u32, k: u32, n: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for w in [m, k, n, 0] { v.extend_from_slice(&w.to_ne_bytes()); }
    v
}

fn time_dispatch(ctx: &VkContext, p: &Pipe, groups: u32, reps: usize) -> f64 {
    ctx.dispatch(p.pipeline, p.layout, p.set, (groups, 1, 1)); // warm
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        ctx.dispatch(p.pipeline, p.layout, p.set, (groups, 1, 1));
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn int8_gemm() {
    let ctx = match VkContext::new() { Ok(c) => c, Err(e) => { eprintln!("no vulkan: {e}"); return; } };
    if !ctx.prec.dp4a { eprintln!("device has no accelerated DP4A; skipping"); return; }
    println!("\n=== INT8 DP4A GEMM on {} ===", ctx.adapter_name);
    println!("{:<30} {:>10} {:>10} {:>8} {:>9} {:>9}", "shape", "int8 GOP/s", "f32 GOP/s", "speedup", "cos(f32)", "kernel-rel");

    let i8_spv = shader::wgsl_to_spirv(I8).expect("naga matmul_i8");
    let f32_spv = shader::wgsl_to_spirv(F32).expect("naga matmul_reg2");
    let reps = 5;

    for s in SHAPES {
        let (m, k, n) = (s.m, s.k, s.n);
        assert!(k % 4 == 0);
        let xf = fill(m * k, 1);
        let wf = fill(n * k, 2);

        // Quantize (per-tensor symmetric) + pack.
        let (xq, sx) = quantize(&xf);
        let (wq, sw) = quantize(&wf);
        let xp = pack(&xq, m, k);
        let wp = pack(&wq, n, k);

        // ---- INT8 GPU kernel ----
        let uq = ctx.storage(32, vk::BufferUsageFlags::UNIFORM_BUFFER);
        ctx.upload(&uq, &uni_i8(m as u32, (k / 4) as u32, n as u32, sx, sw));
        let xb = ctx.storage((xp.len() * 4) as u64, vk::BufferUsageFlags::empty());
        ctx.upload(&xb, &bytes_u32(&xp));
        let wb = ctx.storage((wp.len() * 4) as u64, vk::BufferUsageFlags::empty());
        ctx.upload(&wb, &bytes_u32(&wp));
        let ob = ctx.storage((m * n * 4) as u64, vk::BufferUsageFlags::empty());
        let (got, t_i8) = unsafe {
            let p = build(&ctx, &i8_spv, &uq, &[&xb, &wb, &ob]);
            let t = time_dispatch(&ctx, &p, reg_groups(m, n), reps);
            let bytes = ctx.download(&ob, m * n * 4);
            destroy(&ctx, p);
            (bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(), t)
        };

        // ---- exact host int32 reference from the SAME quantized bytes ----
        // (kernel correctness: GPU must equal this up to fp32 rounding of the
        //  final scale multiply).
        let mut kern_rel = 0f32;
        // ---- fp32 reference from original floats (quantization quality) ----
        let mut dot_gg = 0f64; let mut dot_gf = 0f64; let mut dot_ff = 0f64;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0i32;
                for ki in 0..k { acc += xq[mi * k + ki] as i32 * wq[ni * k + ki] as i32; }
                let exact = acc as f32 * sx * sw;
                let g = got[mi * n + ni];
                let d = (g - exact).abs() / exact.abs().max(1e-3);
                if d > kern_rel { kern_rel = d; }
                let mut f = 0f32;
                for ki in 0..k { f += xf[mi * k + ki] * wf[ni * k + ki]; }
                dot_gg += (g * g) as f64; dot_ff += (f * f) as f64; dot_gf += (g * f) as f64;
            }
        }
        let cos = dot_gf / (dot_gg.sqrt() * dot_ff.sqrt() + 1e-12);

        // ---- fp32 reg2 for the speed baseline ----
        let uf = ctx.storage(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
        ctx.upload(&uf, &uni_f32(m as u32, k as u32, n as u32));
        let xfb = ctx.storage((xf.len() * 4) as u64, vk::BufferUsageFlags::empty());
        ctx.upload(&xfb, bytemuck::cast_slice(&xf));
        let wfb = ctx.storage((wf.len() * 4) as u64, vk::BufferUsageFlags::empty());
        ctx.upload(&wfb, bytemuck::cast_slice(&wf));
        let ofb = ctx.storage((m * n * 4) as u64, vk::BufferUsageFlags::empty());
        let t_f32 = unsafe {
            let p = build(&ctx, &f32_spv, &uf, &[&xfb, &wfb, &ofb]);
            let t = time_dispatch(&ctx, &p, reg_groups(m, n), reps);
            destroy(&ctx, p);
            t
        };

        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        println!(
            "{:<30} {:>10.0} {:>10.0} {:>7.2}x {:>9.5} {:>9.1e}",
            s.label, gflop / t_i8, gflop / t_f32, t_f32 / t_i8, cos, kern_rel
        );

        // kernel must reproduce the exact int math (only fp32 rounding of the scale).
        assert!(kern_rel < 1e-3, "{}: int8 kernel disagrees with exact int reference (rel {kern_rel:.2e})", s.label);
        // quantization must preserve direction (well-conditioned inputs).
        assert!(cos > 0.999, "{}: int8 result not aligned with fp32 (cos {cos:.5})", s.label);
        // int8 should not be slower than the tuned fp32 kernel.
        assert!(t_i8 <= t_f32 * 1.05, "{}: int8 slower than fp32 (i8 {t_i8:.4}s f32 {t_f32:.4}s)", s.label);

        for b in [uq, xb, wb, ob, uf, xfb, wfb, ofb] { ctx.destroy_buffer(b); }
    }
}
