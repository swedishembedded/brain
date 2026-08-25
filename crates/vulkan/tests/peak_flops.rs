// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Peak arithmetic-throughput benchmark — drives each Tesla P40 to its rated
//! per-precision compute ceiling and reports the achieved fraction.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-vulkan --test peak_flops -- --ignored --nocapture
//! ```
//!
//! These are the P40 (GP102) datasheet peaks. The values themselves live in
//! the `PEAK_*` constants below, because the test computes against them; what
//! belongs here is where each one comes from:
//!
//! | precision | derivation |
//! |---|---|
//! | FP32 | 30 SM x 128 FP32 cores x 2 (FMA) x 1.531 GHz |
//! | FP64 | FP32 / 32 (Pascal consumer double-rate) |
//! | FP16 | FP32 / 64 (GP102 has no fast-fp16 unit) |
//! | INT8 | DP4A: 4 int8 lanes per FP32 lane, 8 int-ops per dot |
//!
//! A GEMM cannot reach these - it is bounded by memory and by the fraction of
//! peak a real dependency graph sustains (even cuBLAS tops out well short of
//! the rated figure).
//! The rated numbers are a *pure-ALU* property: a kernel that does nothing but
//! back-to-back fused-multiply-adds out of registers, with enough independent
//! accumulator chains to cover FMA latency and enough threads to fill every SM.
//! That is exactly what this measures — the honest "can we actually drive the
//! silicon to its spec" number, per card.
//!
//! Requires the non-fp32 device features `VkContext` now enables
//! (`shaderFloat64`, `shaderFloat16`, `shaderIntegerDotProduct`); a precision the
//! device does not expose is reported as skipped, not failed.

use ash::vk;
use std::time::Instant;

use vulkan::context::{VkBuffer, VkContext};
use vulkan::shader;

// ---- peak kernels (NOT in the shared registry: they use f16/f64/dp4a the CPU
// JIT and the wgpu device do not compile). Each thread runs `iters` iterations
// of independent FMA/DP4A chains from registers — no memory traffic in the hot
// loop — then writes once (so nothing is dead-code-eliminated). ----

/// 8 independent vec4 chains = 32-way ILP, so 64 FLOP/thread/iter (4 mul + 4
/// add per chain).
const FP32: &str = r#"
struct P { iters: u32, n: u32, m: f32, c: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let s = f32(g & 255u) * 1e-3;
    let m = vec4<f32>(p.m);
    let c = vec4<f32>(p.c + s);
    var a0 = vec4<f32>(s + 0.1); var a1 = vec4<f32>(s + 0.2);
    var a2 = vec4<f32>(s + 0.3); var a3 = vec4<f32>(s + 0.4);
    var a4 = vec4<f32>(s + 0.5); var a5 = vec4<f32>(s + 0.6);
    var a6 = vec4<f32>(s + 0.7); var a7 = vec4<f32>(s + 0.8);
    // 8 independent chains x 4-way inner unroll: 32 vec4 fma / iter, so the
    // loop control is amortized four ways. fma() forces the fused op (1 issue
    // slot, 2 FLOP).
    for (var i = 0u; i < p.iters; i = i + 1u) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    let r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[g] = r.x + r.y + r.z + r.w;
}
"#;

/// 16 independent f64 chains, 16×2=32 FLOP/thread/iter.
const FP64: &str = r#"
struct P { iters: u32, n: u32, m: f32, c: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let s = f64(g & 255u) * 1e-3;
    let m = f64(p.m);
    let c = f64(p.c) + s;
    var a0 = s + 0.1; var a1 = s + 0.2; var a2 = s + 0.3; var a3 = s + 0.4;
    var a4 = s + 0.5; var a5 = s + 0.6; var a6 = s + 0.7; var a7 = s + 0.8;
    var a8 = s + 0.9; var a9 = s + 1.0; var aa = s + 1.1; var ab = s + 1.2;
    var ac = s + 1.3; var ad = s + 1.4; var ae = s + 1.5; var af = s + 1.6;
    for (var i = 0u; i < p.iters; i = i + 1u) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        a8 = fma(a8, m, c); a9 = fma(a9, m, c); aa = fma(aa, m, c); ab = fma(ab, m, c);
        ac = fma(ac, m, c); ad = fma(ad, m, c); ae = fma(ae, m, c); af = fma(af, m, c);
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        a8 = fma(a8, m, c); a9 = fma(a9, m, c); aa = fma(aa, m, c); ab = fma(ab, m, c);
        ac = fma(ac, m, c); ad = fma(ad, m, c); ae = fma(ae, m, c); af = fma(af, m, c);
    }
    let r = (a0+a1+a2+a3) + (a4+a5+a6+a7) + (a8+a9+aa+ab) + (ac+ad+ae+af);
    out[g] = f32(r);
}
"#;

/// 8 independent vec4<f16> chains, 64 FLOP/thread/iter (fp16 lanes).
const FP16: &str = r#"
enable f16;
struct P { iters: u32, n: u32, m: f32, c: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let s = f16(f32(g & 15u) * 1e-2);
    let m = vec4<f16>(f16(p.m));
    let c = vec4<f16>(s);
    var a0 = vec4<f16>(s + 0.1h); var a1 = vec4<f16>(s + 0.2h);
    var a2 = vec4<f16>(s + 0.3h); var a3 = vec4<f16>(s + 0.4h);
    var a4 = vec4<f16>(s + 0.5h); var a5 = vec4<f16>(s + 0.6h);
    var a6 = vec4<f16>(s + 0.7h); var a7 = vec4<f16>(s + 0.8h);
    for (var i = 0u; i < p.iters; i = i + 1u) {
        a0 = a0 * m + c; a1 = a1 * m + c; a2 = a2 * m + c; a3 = a3 * m + c;
        a4 = a4 * m + c; a5 = a5 * m + c; a6 = a6 * m + c; a7 = a7 * m + c;
    }
    let r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[g] = f32(r.x + r.y + r.z + r.w);
}
"#;

/// 16 independent DP4A chains, 16×8=128 int-ops/thread/iter.
const INT8: &str = r#"
struct P { iters: u32, n: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let a = p.a + g; let b = p.b ^ g;
    var s0 = 0i; var s1 = 0i; var s2 = 0i; var s3 = 0i;
    var s4 = 0i; var s5 = 0i; var s6 = 0i; var s7 = 0i;
    var s8 = 0i; var s9 = 0i; var sa = 0i; var sb = 0i;
    var sc = 0i; var sd = 0i; var se = 0i; var sf = 0i;
    // 16 chains x 2-way unroll = 32 dots/iter. dot4I8Packed = 8 int ops.
    for (var i = 0u; i < p.iters; i = i + 1u) {
        s0 = s0 + dot4I8Packed(a, b); s1 = s1 + dot4I8Packed(a, b);
        s2 = s2 + dot4I8Packed(a, b); s3 = s3 + dot4I8Packed(a, b);
        s4 = s4 + dot4I8Packed(a, b); s5 = s5 + dot4I8Packed(a, b);
        s6 = s6 + dot4I8Packed(a, b); s7 = s7 + dot4I8Packed(a, b);
        s8 = s8 + dot4I8Packed(a, b); s9 = s9 + dot4I8Packed(a, b);
        sa = sa + dot4I8Packed(a, b); sb = sb + dot4I8Packed(a, b);
        sc = sc + dot4I8Packed(a, b); sd = sd + dot4I8Packed(a, b);
        se = se + dot4I8Packed(a, b); sf = sf + dot4I8Packed(a, b);
        s0 = s0 + dot4I8Packed(a, b); s1 = s1 + dot4I8Packed(a, b);
        s2 = s2 + dot4I8Packed(a, b); s3 = s3 + dot4I8Packed(a, b);
        s4 = s4 + dot4I8Packed(a, b); s5 = s5 + dot4I8Packed(a, b);
        s6 = s6 + dot4I8Packed(a, b); s7 = s7 + dot4I8Packed(a, b);
        s8 = s8 + dot4I8Packed(a, b); s9 = s9 + dot4I8Packed(a, b);
        sa = sa + dot4I8Packed(a, b); sb = sb + dot4I8Packed(a, b);
        sc = sc + dot4I8Packed(a, b); sd = sd + dot4I8Packed(a, b);
        se = se + dot4I8Packed(a, b); sf = sf + dot4I8Packed(a, b);
    }
    out[g] = (s0+s1+s2+s3) + (s4+s5+s6+s7) + (s8+s9+sa+sb) + (sc+sd+se+sf);
}
"#;

const PEAK_FP32: f64 = 11_760.0; // GFLOP/s
const PEAK_FP64: f64 = 367.5;
const PEAK_FP16: f64 = 183.7;
const PEAK_INT8: f64 = 47_000.0; // GOP/s (47 TOPS)

const WORKGROUPS: u32 = 4096;
const THREADS: u32 = WORKGROUPS * 256;

/// Minimal 2-binding (uniform + one storage) compute pipeline for a peak kernel.
struct Peak {
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

unsafe fn build(ctx: &VkContext, spirv: &[u32], ubuf: &VkBuffer, obuf: &VkBuffer) -> Peak {
    let dev = &ctx.device;
    let module = dev
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
        .expect("shader module");
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_layout = dev
        .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
        .expect("set layout");
    let set_layouts = [set_layout];
    let layout = dev
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)
        .expect("pipeline layout");
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let pipeline = dev
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
            None,
        )
        .expect("compute pipeline")[0];
    let pool_sizes = [
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1),
    ];
    let pool = dev
        .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None)
        .expect("pool");
    let set = dev
        .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts))
        .expect("set")[0];
    let u_info = [vk::DescriptorBufferInfo::default().buffer(ubuf.buffer).range(ubuf.size)];
    let o_info = [vk::DescriptorBufferInfo::default().buffer(obuf.buffer).range(obuf.size)];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&o_info),
    ];
    dev.update_descriptor_sets(&writes, &[]);
    Peak { module, layout, set_layout, pipeline, pool, set }
}

unsafe fn destroy(ctx: &VkContext, p: Peak) {
    let d = &ctx.device;
    d.destroy_pipeline(p.pipeline, None);
    d.destroy_pipeline_layout(p.layout, None);
    d.destroy_descriptor_set_layout(p.set_layout, None);
    d.destroy_descriptor_pool(p.pool, None);
    d.destroy_shader_module(p.module, None);
}

/// uniform: {iters:u32, n:u32, m/a:u32, c/b:u32} padded to 16 B.
fn uni(iters: u32, n: u32, w2: u32, w3: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&iters.to_ne_bytes());
    v.extend_from_slice(&n.to_ne_bytes());
    v.extend_from_slice(&w2.to_ne_bytes());
    v.extend_from_slice(&w3.to_ne_bytes());
    v
}

/// Compile + run one kernel, return best GOP/s over `reps` (ops_per_iter counts
/// the fused mul+add as 2). Returns None if naga/pipeline rejects the kernel.
fn run(ctx: &VkContext, src: &str, iters: u32, ops_per_thread_iter: f64, w2: u32, w3: u32, reps: usize) -> Option<f64> {
    let spirv = shader::wgsl_to_spirv(src).map_err(|e| eprintln!("  naga: {e}")).ok()?;
    let ubuf = ctx.storage(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
    ctx.upload(&ubuf, &uni(iters, THREADS, w2, w3));
    let obuf = ctx.storage((THREADS as u64) * 4, vk::BufferUsageFlags::empty());
    let best = unsafe {
        let p = build(ctx, &spirv, &ubuf, &obuf);
        // warm-up
        ctx.dispatch(p.pipeline, p.layout, p.set, (WORKGROUPS, 1, 1));
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            ctx.dispatch(p.pipeline, p.layout, p.set, (WORKGROUPS, 1, 1));
            best = best.min(t.elapsed().as_secs_f64());
        }
        destroy(ctx, p);
        best
    };
    ctx.destroy_buffer(ubuf);
    ctx.destroy_buffer(obuf);
    let ops = THREADS as f64 * iters as f64 * ops_per_thread_iter;
    Some(ops / best / 1e9)
}

#[test]
#[ignore]
fn peak_flops() {
    let ndev: usize = std::env::var("BRAIN_VK_NDEV").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let reps = 5;
    let iters = 20_000u32;

    for dev in 0..ndev {
        std::env::set_var("BRAIN_VK_DEVICE", dev.to_string());
        let ctx = match VkContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("device {dev}: no Vulkan context ({e})");
                continue;
            }
        };
        println!("\n=== GPU {dev}: {} ===", ctx.adapter_name);
        println!("  precisions enabled: f64={} f16={} dp4a={}", ctx.prec.f64, ctx.prec.f16, ctx.prec.dp4a);
        println!("  {:<6} {:>12} {:>12} {:>8}", "prec", "achieved", "peak", "%peak");

        let report = |name: &str, got: Option<f64>, peak: f64, unit: &str| {
            match got {
                Some(g) => println!("  {:<6} {:>9.1} {} {:>9.1} {} {:>6.1}%", name, g, unit, peak, unit, 100.0 * g / peak),
                None => println!("  {:<6} {:>12} (kernel/feature unavailable)", name, "skipped"),
            }
        };

        report("FP32", run(&ctx, FP32, iters, 256.0, 1.0f32.to_bits(), (1e-6f32).to_bits(), reps), PEAK_FP32, "GFLOP/s");
        report("FP64", if ctx.prec.f64 { run(&ctx, FP64, iters, 64.0, 1.0f32.to_bits(), (1e-6f32).to_bits(), reps) } else { None }, PEAK_FP64, "GFLOP/s");
        report("FP16", if ctx.prec.f16 { run(&ctx, FP16, iters, 64.0, 1.0f32.to_bits(), (1e-3f32).to_bits(), reps) } else { None }, PEAK_FP16, "GFLOP/s");
        // DP4A: each dot4I8Packed = 4 mul + 4 add = 8 int ops.
        report("INT8", if ctx.prec.dp4a { run(&ctx, INT8, iters, 256.0, 0x01020304, 0x04030201, reps) } else { None }, PEAK_INT8, "GOP/s");
    }
}
