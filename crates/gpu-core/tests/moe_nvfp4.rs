// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The NVFP4 sparse-MoE expert GEMMs (`moe_nvfp4_gemv_gather`, `moe_nvfp4_grouped`).
//!
//! Swedish Embedded AB implements fast serving of NVFP4 mixture-of-experts models for its clients. If your team
//! needs expertise in 4-bit expert GEMMs then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! * the gather kernel is the decode regime's expert GEMV over a fused bank: its output is the f32 product of the
//!   decoded weights (E2M1 code x E4M3 block scale x per-expert global scale, the layout of a ModelOpt
//!   `W4A16_NVFP4` checkpoint; `k` a multiple of 64) with the unquantised activations, within f32 summation error of a host reference;
//! * the grouped kernel agrees with the gather kernel on the same bank, ids and activations to f32 rounding of the
//!   running sum (on a CUDA device the gather kernel is its native twin, which sums with fused multiply-adds in another
//!   order; the bits are equal only where both are the WGSL tier).

use data::rng::{random_seed, Lcg};
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("moe_nvfp4_gemv_gather", kernels::MOE_NVFP4_GEMV_GATHER),
    ("moe_route_count", kernels::MOE_ROUTE_COUNT),
    ("moe_route_scan", kernels::MOE_ROUTE_SCAN),
    ("moe_route_emit", kernels::MOE_ROUTE_EMIT),
    ("moe_nvfp4_grouped", kernels::MOE_NVFP4_GROUPED),
];
const K_GATHER: usize = 0;
const K_COUNT: usize = 1;
const K_SCAN: usize = 2;
const K_EMIT: usize = 3;
const K_GROUPED: usize = 4;
const MR: u32 = 8;
const COLS: u32 = 4;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1(code: u32) -> f32 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

fn e4m3(b: u32) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = if e == 0 { m as f32 / 512.0 } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) };
    if b & 128 != 0 { -v } else { v }
}

struct Bank {
    words: Vec<u32>,
    scale_words: Vec<u32>,
    scale_bytes: Vec<u8>,
    global: Vec<f32>,
}

/// A random bank of `blocks` matrices of `[n, k]`: codes, E4M3 scale bytes of normal magnitude, global scales.
fn random_bank(blocks: usize, n: usize, k: usize, seed: u64) -> Bank {
    let mut r = Lcg::new(seed);
    let words: Vec<u32> = (0..blocks * n * k / 8).map(|_| r.next_u32()).collect();
    // E4M3 bytes with exponent 4..=10 and a random mantissa and sign-free: scales are positive in practice.
    let scale_bytes: Vec<u8> = (0..blocks * n * k / 16).map(|_| (((4 + r.next_u32() % 7) << 3) | (r.next_u32() % 8)) as u8).collect();
    let scale_words = scale_bytes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (i, b)| w | (*b as u32) << (8 * i))).collect();
    let global = (0..blocks).map(|_| 0.002 + (r.next_u32() % 100) as f32 * 1e-4).collect();
    Bank { words, scale_words, scale_bytes, global }
}

fn weight(bank: &Bank, row: usize, k: usize, i: usize) -> f32 {
    let w = bank.words[row * k / 8 + i / 8];
    e2m1(w >> (4 * (i % 8))) * e4m3(bank.scale_bytes[row * k / 16 + i / 16] as u32)
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

fn read_bits(gpu: &Gpu, buf: &gpu_core::DeviceBuffer, n: usize) -> Vec<u32> {
    gpu.read(buf, n).iter().map(|f| f.to_bits()).collect()
}

fn route(gpu: &Gpu, ids: &[u32], ne: u32) -> (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer) {
    let slots = ids.len() as u32;
    let ids_b = upload_u32(gpu, ids);
    let counts = gpu.storage(ne as u64);
    let tab = gpu.storage(2 * (ne as u64 + 1));
    let perm = gpu.storage(slots.max(1) as u64);
    gpu.submit(
        &[],
        &[
            gpu.dispatch(K_COUNT, &[&ids_b, &counts], &[slots, ne], Dispatch::Workgroups(ne)),
            gpu.dispatch(K_SCAN, &[&counts, &tab], &[ne, MR], Dispatch::Workgroups(1)),
            gpu.dispatch(K_EMIT, &[&ids_b, &tab, &perm], &[slots, ne], Dispatch::Workgroups(ne)),
        ],
    );
    (tab, perm)
}

struct Case {
    slots: usize,
    k: usize,
    n: usize,
    xdiv: usize,
    ne: usize,
}

fn run_case(gpu: &Gpu, c: &Case, seed: u64) {
    let Case { slots, k, n, xdiv, ne } = *c;
    let rows = slots.div_ceil(xdiv);
    let mut rng = Lcg::new(seed);
    let x: Vec<f32> = (0..rows * k).map(|_| (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();
    let ids: Vec<u32> = (0..slots).map(|_| rng.next_u32() % ne as u32).collect();
    let bank = random_bank(ne, n, k, seed + 1);
    let (xb, idb) = (gpu.storage_init("x", &x), upload_u32(gpu, &ids));
    let (wb, sb, gb) = (upload_u32(gpu, &bank.words), upload_u32(gpu, &bank.scale_words), gpu.storage_init("gl", &bank.global));
    let gather = gpu.storage((slots * n) as u64);
    let grouped = gpu.storage((slots * n) as u64);
    gpu.write(&grouped, &vec![0x7fc0_dead; slots * n]);
    let (tab, perm) = route(gpu, &ids, ne as u32);
    let max_tiles = ne as u32 + (slots as u32).div_ceil(MR);
    gpu.submit(
        &[],
        &[
            gpu.dispatch(K_GATHER, &[&xb, &idb, &wb, &sb, &gb, &gather], &[slots as u32, k as u32, n as u32, xdiv as u32], Dispatch::Workgroups(slots as u32 * (n as u32).div_ceil(COLS))),
            gpu.dispatch(K_GROUPED, &[&xb, &tab, &perm, &wb, &sb, &gb, &grouped], &[max_tiles, k as u32, n as u32, xdiv as u32, ne as u32], Dispatch::Workgroups(max_tiles * (n as u32).div_ceil(COLS))),
        ],
    );
    gpu.poll_wait();
    let (gather_bits, grouped_bits) = (read_bits(gpu, &gather, slots * n), read_bits(gpu, &grouped, slots * n));
    if gpu.native_kernel_for(K_GATHER, &[slots as u32, k as u32, n as u32, xdiv as u32]).is_none() {
        assert_eq!(grouped_bits, gather_bits, "on the WGSL tier grouped is bit-identical to gather (slots={slots} k={k} n={n})");
    }
    for s in 0..slots {
        let e = ids[s] as usize;
        for col in 0..n {
            let row = e * n + col;
            let want: f64 = (0..k).map(|i| weight(&bank, row, k, i) as f64 * x[(s / xdiv) * k + i] as f64).sum::<f64>() * bank.global[e] as f64;
            let got = f32::from_bits(gather_bits[s * n + col]) as f64;
            let tol = 1e-4 * (1.0 + want.abs()) + 2e-5 * (k as f64).sqrt();
            assert!((got - want).abs() <= tol, "slot {s} col {col}: gather {got}, host reference {want} (slots={slots} k={k} n={n})");
            let grouped = f32::from_bits(grouped_bits[s * n + col]) as f64;
            assert!((grouped - want).abs() <= tol, "slot {s} col {col}: grouped {grouped}, host reference {want} (slots={slots} k={k} n={n})");
        }
    }
}

#[test]
fn the_nvfp4_expert_gemms_match_the_decoded_reference_and_each_other() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("needs workgroup barriers");
    }
    let base = random_seed();
    for (i, c) in [
        Case { slots: 9, k: 2048, n: 64, xdiv: 9, ne: 17 },   // decode: a token's routed experts plus the shared one
        Case { slots: 90, k: 2048, n: 36, xdiv: 9, ne: 17 },  // ten tokens
        Case { slots: 40, k: 512, n: 48, xdiv: 1, ne: 17 },   // down projection (k = moe_intermediate_size)
        Case { slots: 37, k: 192, n: 6, xdiv: 1, ne: 5 },     // ragged n, 6 chunks (fewer than lanes)
        Case { slots: 30, k: 576, n: 12, xdiv: 2, ne: 7 },    // 18 chunks: two lanes take a second
        Case { slots: 1, k: 64, n: 1, xdiv: 1, ne: 1 },
    ]
    .iter()
    .enumerate()
    {
        run_case(&gpu, c, base.wrapping_add(i as u64 * 17));
    }
}
