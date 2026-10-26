// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native stored-format GEMVs (`kernels_cuda`, `cu/fp8_nvfp4_gemv.cu`) that
//! `gpu_core::native_upgrade` substitutes for their WGSL siblings on a CUDA device: `matmul_fp8_gemv`,
//! `matmul_nvfp4_gemv` and `moe_nvfp4_gemv_gather`.
//!
//! Swedish Embedded AB implements native kernels for FP8 and NVFP4 LLM decode. If your team needs expertise in
//! streaming 4-bit and 8-bit weights at the speed of the memory system then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! The native kernels decode the codes exactly and sum with fused multiply-adds in a different order than the WGSL
//! tier, so the claim is not equal bits but agreement with an f64 oracle of the decoded weights to f32 rounding of the
//! running sum, for the native kernel and for the WGSL tier alike (`<name>_ref` is the same source under a name the
//! upgrade table does not know, so both run on the same device from the same inputs). The selection test makes the
//! comparison honest: the native kernel must actually be the one that ran.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};
use std::sync::OnceLock;

fn specialised(base: &'static str, src: &'static str, maxm: u32) -> (&'static str, &'static str) {
    kernels::template::interned(base, src, &[("MAXM", maxm)]).expect("the GEMV kernels declare MAXM")
}

/// `[(name, source)]` for the registered pipelines: each kernel and its `_ref` alias.
fn kernel_list() -> &'static [(&'static str, &'static str)] {
    static LIST: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    LIST.get_or_init(|| {
        let (f8, f8_src) = specialised("matmul_fp8_gemv", kernels::MATMUL_FP8_GEMV, 8);
        let (f4, f4_src) = specialised("matmul_nvfp4_gemv", kernels::MATMUL_NVFP4_GEMV, 8);
        let (f8_1, f8_1_src) = specialised("matmul_fp8_gemv", kernels::MATMUL_FP8_GEMV, 1);
        let (f4_1, f4_1_src) = specialised("matmul_nvfp4_gemv", kernels::MATMUL_NVFP4_GEMV, 1);
        vec![
            (f8, f8_src),
            ("matmul_fp8_gemv_ref", f8_src),
            (f4, f4_src),
            ("matmul_nvfp4_gemv_ref", f4_src),
            ("moe_nvfp4_gemv_gather", kernels::MOE_NVFP4_GEMV_GATHER),
            ("moe_nvfp4_gemv_gather_ref", kernels::MOE_NVFP4_GEMV_GATHER),
            (f8_1, f8_1_src),
            (f4_1, f4_1_src),
        ]
    })
}

const FP8: usize = 0;
const FP8_REF: usize = 1;
const FP4: usize = 2;
const FP4_REF: usize = 3;
const GATHER: usize = 4;
const GATHER_REF: usize = 5;
/// The single-row builds (what a decode step dispatches): their own kernels.
const FP8_M1: usize = 6;
const FP4_M1: usize = 7;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

fn e4m3(b: u8) -> f32 {
    let (e, m) = (u32::from(b >> 3) & 15, u32::from(b & 7));
    let v = if e == 0 { m as f32 / 512.0 } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) };
    if b & 128 != 0 { -v } else { v }
}

fn e2m1(c: u32) -> f32 {
    let v = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> backend_api::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

/// Valid E4M3 bytes (no NaN code).
fn fp8_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| loop { let b = r.next_u32() as u8; if b & 0x7f != 0x7f { break b; } }).collect()
}

/// E4M3 scale bytes of ordinary positive magnitude.
fn scale_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| 0x30 + (r.next_u32() % 0x20) as u8).collect()
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (i, b)| w | u32::from(*b) << (8 * i))).collect()
}

/// Assert `got` equals the oracle within f32 rounding of the running sum, measured against each output's sum of squared terms.
fn assert_matches(label: &str, got: &[f32], want: &[(f64, f64)]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, (w, scale))) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}: output {i} not written");
        assert!((f64::from(*g) - w).abs() <= 2e-5 * scale + 1e-12, "{label}: output {i} is {g}, oracle {w} (scale {scale})");
    }
}

struct Fp8Case {
    m: usize,
    k: usize,
    n: usize,
    first: usize,
    count: usize,
    block: usize,
}

fn run_fp8(gpu: &Gpu, c: &Fp8Case, seed: u64) {
    let Fp8Case { m, k, n, first, count, block } = *c;
    let w = fp8_bytes(n * k, seed);
    let cb = k.div_ceil(block);
    let grid: Vec<f32> = Lcg::new(seed + 1).vec_scaled(n.div_ceil(block) * cb, 0.05).iter().map(|v| v.abs() + 0.01).collect();
    let x = Lcg::new(seed + 2).vec_scaled((m + 2) * k, 1.0);
    let xr0 = 2usize; // activation rows start after two rows that must not be read
    let (xb, wb, sb) = (gpu.storage_init("x", &x), upload_u32(gpu, &words_of(&w)), gpu.storage_init("s", &grid));
    let params = [m as u32, k as u32, count as u32, first as u32, block as u32, cb as u32, xr0 as u32, 1];
    let (native, reference) = (gpu.storage((m * count) as u64), gpu.storage((m * count) as u64));
    gpu.write(&native, &vec![0x7fc0_dead; m * count]);
    gpu.write(&reference, &vec![0x7fc0_dead; m * count]);
    let (kind, expect) = if m == 1 { (FP8_M1, "matmul_fp8_gemv_m1") } else { (FP8, "matmul_fp8_gemv") };
    assert_eq!(gpu.native_kernel_for(kind, &params), Some(expect), "the native kernel must be the one that runs");
    assert_eq!(gpu.native_kernel_for(FP8_REF, &params), None);
    let step = |kind: usize, y: &backend_api::DeviceBuffer, blocks: u32| gpu.dispatch(kind, &[&xb, &wb, &sb, y], &params, Dispatch::Workgroups(blocks));
    gpu.submit(&[], &[step(kind, &native, count.div_ceil(16) as u32), step(FP8_REF, &reference, count.div_ceil(4) as u32)]);
    gpu.poll_wait();
    let mut want = Vec::new();
    for r in 0..m {
        for col in 0..count {
            let row = first + col;
            let terms: Vec<f64> = (0..k).map(|i| f64::from(x[(xr0 + r) * k + i]) * f64::from(e4m3(w[row * k + i])) * f64::from(grid[(row / block) * cb + i / block])).collect();
            want.push((terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt()));
        }
    }
    let label = format!("fp8 m={m} k={k} n={n} first={first} count={count} block={block}");
    assert_matches(&format!("native {label}"), &gpu.read(&native, m * count), &want);
    assert_matches(&format!("wgsl {label}"), &gpu.read(&reference, m * count), &want);
}

struct Fp4Case {
    m: usize,
    k: usize,
    count: usize,
    first: usize,
}

/// The decoded value of weight `i` of row `row` of an NVFP4 matrix whose codes, scale bytes and tensor scale are given.
fn fp4_weight(codes: &[u32], scales: &[u8], global: f32, k: usize, row: usize, i: usize) -> f64 {
    f64::from(e2m1(codes[row * k / 8 + i / 8] >> (4 * (i % 8)))) * f64::from(e4m3(scales[row * k / 16 + i / 16])) * f64::from(global)
}

fn run_fp4(gpu: &Gpu, c: &Fp4Case, seed: u64) {
    let Fp4Case { m, k, count, first } = *c;
    let n = first + count + 3;
    let mut r = Lcg::new(seed);
    let codes: Vec<u32> = (0..n * k / 8).map(|_| r.next_u32()).collect();
    let scales = scale_bytes(n * k / 16, seed + 1);
    let global = 0.007f32;
    let x = Lcg::new(seed + 2).vec_scaled((m + 1) * k, 1.0);
    let xr0 = 1usize;
    let (xb, wb, sb) = (gpu.storage_init("x", &x), upload_u32(gpu, &codes), upload_u32(gpu, &words_of(&scales)));
    let params = [m as u32, k as u32, count as u32, first as u32, xr0 as u32, global.to_bits(), 1, 0];
    let (native, reference) = (gpu.storage((m * count) as u64), gpu.storage((m * count) as u64));
    let (kind, expect) = if m == 1 { (FP4_M1, "matmul_nvfp4_gemv_m1") } else { (FP4, "matmul_nvfp4_gemv") };
    assert_eq!(gpu.native_kernel_for(kind, &params), Some(expect));
    assert_eq!(gpu.native_kernel_for(FP4_REF, &params), None);
    let step = |kind: usize, y: &backend_api::DeviceBuffer, blocks: u32| gpu.dispatch(kind, &[&xb, &wb, &sb, y], &params, Dispatch::Workgroups(blocks));
    gpu.submit(&[], &[step(kind, &native, count.div_ceil(16) as u32), step(FP4_REF, &reference, count.div_ceil(4) as u32)]);
    gpu.poll_wait();
    let mut want = Vec::new();
    for rr in 0..m {
        for col in 0..count {
            let terms: Vec<f64> = (0..k).map(|i| f64::from(x[(xr0 + rr) * k + i]) * fp4_weight(&codes, &scales, global, k, first + col, i)).collect();
            want.push((terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt()));
        }
    }
    let label = format!("nvfp4 m={m} k={k} count={count} first={first}");
    assert_matches(&format!("native {label}"), &gpu.read(&native, m * count), &want);
    assert_matches(&format!("wgsl {label}"), &gpu.read(&reference, m * count), &want);
}

fn run_gather(gpu: &Gpu, slots: usize, k: usize, n: usize, xdiv: usize, experts: usize, seed: u64) {
    let rows = slots.div_ceil(xdiv);
    let mut r = Lcg::new(seed);
    let codes: Vec<u32> = (0..experts * n * k / 8).map(|_| r.next_u32()).collect();
    let scales = scale_bytes(experts * n * k / 16, seed + 1);
    let global: Vec<f32> = (0..experts).map(|e| 0.002 + 0.001 * e as f32).collect();
    let x = Lcg::new(seed + 2).vec_scaled(rows * k, 1.0);
    let ids: Vec<u32> = (0..slots).map(|_| r.next_u32() % experts as u32).collect();
    let (xb, ib, wb, sb, gb) = (gpu.storage_init("x", &x), upload_u32(gpu, &ids), upload_u32(gpu, &codes), upload_u32(gpu, &words_of(&scales)), gpu.storage_init("g", &global));
    let params = [slots as u32, k as u32, n as u32, xdiv as u32];
    let (native, reference) = (gpu.storage((slots * n) as u64), gpu.storage((slots * n) as u64));
    assert_eq!(gpu.native_kernel_for(GATHER, &params), Some("moe_nvfp4_gemv_gather"));
    assert_eq!(gpu.native_kernel_for(GATHER_REF, &params), None);
    let step = |kind: usize, out: &backend_api::DeviceBuffer, blocks: u32| gpu.dispatch(kind, &[&xb, &ib, &wb, &sb, &gb, out], &params, Dispatch::Workgroups(blocks));
    gpu.submit(&[], &[step(GATHER, &native, (slots * n.div_ceil(16)) as u32), step(GATHER_REF, &reference, (slots * n.div_ceil(4)) as u32)]);
    gpu.poll_wait();
    let mut want = Vec::new();
    for s in 0..slots {
        let e = ids[s] as usize;
        for col in 0..n {
            let terms: Vec<f64> = (0..k).map(|i| f64::from(x[(s / xdiv) * k + i]) * fp4_weight(&codes, &scales, global[e], k, e * n + col, i)).collect();
            want.push((terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt()));
        }
    }
    let label = format!("gather slots={slots} k={k} n={n} xdiv={xdiv} experts={experts}");
    assert_matches(&format!("native {label}"), &gpu.read(&native, slots * n), &want);
    assert_matches(&format!("wgsl {label}"), &gpu.read(&reference, slots * n), &want);
}

#[test]
fn the_native_fp8_gemv_matches_the_oracle_at_every_row_count_window_and_scale_layout() {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    if !is_cuda(&gpu) || !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the native stored-format GEMVs need a CUDA device");
    }
    let base = data::rng::random_seed();
    for (i, c) in [
        Fp8Case { m: 1, k: 2048, n: 300, first: 0, count: 300, block: 128 },
        Fp8Case { m: 2, k: 2048, n: 300, first: 5, count: 200, block: 128 },
        Fp8Case { m: 8, k: 4096, n: 70, first: 0, count: 70, block: 128 },
        Fp8Case { m: 3, k: 144, n: 33, first: 1, count: 20, block: 16 },   // ragged k (9 chunks: fewer than lanes), tiny scale tiles
        Fp8Case { m: 5, k: 512, n: 130, first: 0, count: 130, block: 128 }, // a partial last scale tile
        Fp8Case { m: 1, k: 6144, n: 17, first: 0, count: 17, block: 128 },
    ]
    .iter()
    .enumerate()
    {
        run_fp8(&gpu, c, base.wrapping_add(i as u64 * 13));
    }
}

#[test]
fn the_native_nvfp4_gemv_matches_the_oracle_at_every_row_count_and_window() {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    if !is_cuda(&gpu) || !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the native stored-format GEMVs need a CUDA device");
    }
    let base = data::rng::random_seed();
    for (i, c) in [
        Fp4Case { m: 1, k: 2048, count: 300, first: 0 },
        Fp4Case { m: 4, k: 2048, count: 64, first: 7 },
        Fp4Case { m: 8, k: 576, count: 33, first: 0 },  // 18 chunks: two lanes take a second
        Fp4Case { m: 2, k: 64, count: 17, first: 3 },   // two chunks: most lanes idle
        Fp4Case { m: 1, k: 4096, count: 1000, first: 0 },
    ]
    .iter()
    .enumerate()
    {
        run_fp4(&gpu, c, base.wrapping_add(i as u64 * 17));
    }
}

#[test]
fn the_native_nvfp4_expert_gather_matches_the_oracle() {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    if !is_cuda(&gpu) || !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the native stored-format GEMVs need a CUDA device");
    }
    let base = data::rng::random_seed();
    for (i, &(slots, k, n, xdiv, experts)) in [(9usize, 2048usize, 64usize, 9usize, 17usize), (90, 2048, 36, 9, 17), (40, 512, 48, 1, 17), (37, 192, 6, 1, 5), (30, 576, 12, 2, 7), (1, 64, 1, 1, 1)].iter().enumerate() {
        run_gather(&gpu, slots, k, n, xdiv, experts, base.wrapping_add(i as u64 * 19));
    }
}

/// The fused gate + up + SwiGLU kernel equals `silu(gate) * up` of two gathers' oracles.
#[test]
fn the_fused_gate_up_silu_matches_the_oracle() {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    if !is_cuda(&gpu) || !gpu.caps().workgroup_reductions || !gpu.has_fused(gpu_core::Fused::MoeNvfp4GateUp) {
        return brain_testutil::skip_unavailable("the fused gate/up kernel needs a CUDA device");
    }
    let base = data::rng::random_seed();
    for (i, &(slots, k, n, xdiv, experts)) in [(9usize, 2048usize, 64usize, 9usize, 17usize), (18, 576, 37, 9, 5), (1, 64, 1, 1, 1)].iter().enumerate() {
        let seed = base.wrapping_add(i as u64 * 23);
        let rows = slots.div_ceil(xdiv);
        let mut r = Lcg::new(seed);
        let bank = |salt: u64| -> (Vec<u32>, Vec<u8>, Vec<f32>) {
            let mut r = Lcg::new(seed + salt);
            ((0..experts * n * k / 8).map(|_| r.next_u32()).collect(), scale_bytes(experts * n * k / 16, seed + salt + 1), (0..experts).map(|e| 0.002 + 0.0007 * e as f32 + salt as f32 * 1e-4).collect())
        };
        let (cg, sg, gg) = bank(100);
        let (cu, su, gu) = bank(200);
        let x = Lcg::new(seed + 2).vec_scaled(rows * k, 1.0);
        let ids: Vec<u32> = (0..slots).map(|_| r.next_u32() % experts as u32).collect();
        let (xb, ib) = (gpu.storage_init("x", &x), upload_u32(&gpu, &ids));
        let (wg, sgb, ggb) = (upload_u32(&gpu, &cg), upload_u32(&gpu, &words_of(&sg)), gpu.storage_init("gg", &gg));
        let (wu, sub, gub) = (upload_u32(&gpu, &cu), upload_u32(&gpu, &words_of(&su)), gpu.storage_init("gu", &gu));
        let out = gpu.storage((slots * n) as u64);
        let step = gpu.fused_step(gpu_core::Fused::MoeNvfp4GateUp, &[&xb, &ib, &wg, &sgb, &ggb, &wu, &sub, &gub, &out], &[slots as u32, k as u32, n as u32, xdiv as u32]).expect("offered");
        gpu.submit(&[], &[step]);
        gpu.poll_wait();
        let got = gpu.read(&out, slots * n);
        for s in 0..slots {
            let e = ids[s] as usize;
            for col in 0..n {
                let dot = |codes: &[u32], scales: &[u8], global: f32| -> (f64, f64) {
                    let terms: Vec<f64> = (0..k).map(|t| f64::from(x[(s / xdiv) * k + t]) * fp4_weight(codes, scales, global, k, e * n + col, t)).collect();
                    (terms.iter().sum(), terms.iter().map(|v| v * v).sum::<f64>().sqrt())
                };
                let ((g, gs), (u, us)) = (dot(&cg, &sg, gg[e]), dot(&cu, &su, gu[e]));
                let want = g / (1.0 + (-g).exp()) * u;
                // silu is 1-Lipschitz-ish times u: the error scale is the larger of the two terms' scales.
                let tol = 2e-5 * (gs * u.abs() + us * (g / (1.0 + (-g).exp())).abs() + gs * us * 0.5) + 1e-9;
                assert!((f64::from(got[s * n + col]) - want).abs() <= tol, "slot {s} col {col}: fused {} oracle {want} (k={k})", got[s * n + col]);
            }
        }
    }
}

/// The native router head selects the reference's experts and agrees with its weights to f32 rounding.
#[test]
fn the_native_router_topk_selects_the_reference_experts() {
    let list: &[(&str, &str)] = &[("moe_router_topk", kernels::MOE_ROUTER_TOPK), ("moe_router_topk_ref", kernels::MOE_ROUTER_TOPK)];
    let gpu = gpu_core::testgpu::dev(list);
    if !is_cuda(&gpu) || !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the native router head needs a CUDA device");
    }
    let mut r = Lcg::new(data::rng::random_seed());
    for &(rows, n, top_k, shared, norm, scale) in &[
        (1usize, 256usize, 8usize, 1usize, 1u32, 1.0f32),
        (5, 256, 8, 1, 1, 1.0),
        (3, 17, 2, 0, 1, 1.0),
        (40, 1000, 16, 1, 1, 1.0),
        (2, 256, 1, 1, 1, 1.0),
        // DeepSeek-V2's raw, scaled routing.
        (3, 64, 6, 0, 0, 1.0),
        (3, 64, 6, 0, 0, 2.5),
    ] {
        let width = n + shared;
        let logits: Vec<f32> = (0..rows * width).map(|_| (r.next_u32() % 20001) as f32 / 2500.0 - 4.0).collect();
        let lb = gpu.storage_init("l", &logits);
        let out_w = top_k + shared;
        let (ia, wa) = (gpu.storage((rows * out_w) as u64), gpu.storage((rows * out_w) as u64));
        let (ib, wb) = (gpu.storage((rows * out_w) as u64), gpu.storage((rows * out_w) as u64));
        let params = [rows as u32, n as u32, top_k as u32, shared as u32, norm, scale.to_bits()];
        assert_eq!(gpu.native_kernel_for(0, &params), Some("moe_router_topk"));
        assert_eq!(gpu.native_kernel_for(1, &params), None);
        gpu.submit(&[], &[gpu.dispatch(0, &[&lb, &ia, &wa], &params, Dispatch::Workgroups(rows as u32)), gpu.dispatch(1, &[&lb, &ib, &wb], &params, Dispatch::Workgroups(rows as u32))]);
        gpu.poll_wait();
        let (ids_n, ids_r) = (gpu.read(&ia, rows * out_w), gpu.read(&ib, rows * out_w));
        assert_eq!(ids_n.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), ids_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "rows={rows} n={n} k={top_k}: chosen experts differ");
        for (a, b) in gpu.read(&wa, rows * out_w).iter().zip(gpu.read(&wb, rows * out_w)) {
            assert!((a - b).abs() <= 1e-5 * b.abs() + 1e-9, "rows={rows} n={n}: weight {a} vs {b}");
        }
    }
}
