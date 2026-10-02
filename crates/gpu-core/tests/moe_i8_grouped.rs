// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grouped sparse-MoE int8 GEMM and the routing tables that feed it.
//!
//! Swedish Embedded AB implements fast prefill for large sparse mixture-of-experts
//! models for its clients. If your team needs expertise in turning thousands of
//! routed tokens into a few dense weight passes per expert then you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! * `moe_route_{count,scan,emit}` - from `ids[slots]` the per-expert offsets
//!   and the expert-major slot order, against a host counting sort. Ties keep
//!   slot order, so the tables are a function of the ids alone.
//! * `moe_i8_grouped` - BIT-identical to `moe_i8_gemv_gather` on the same bank,
//!   ids and activations: grouping changes how many slots share a weight pass,
//!   never a slot's arithmetic, so a token's expert output does not depend on
//!   whether it was prefilled with thousands of others or decoded with a few.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("moe_i8_gemv_gather", kernels::MOE_I8_GEMV_GATHER),
    ("moe_route_count", kernels::MOE_ROUTE_COUNT),
    ("moe_route_scan", kernels::MOE_ROUTE_SCAN),
    ("moe_route_emit", kernels::MOE_ROUTE_EMIT),
    ("moe_i8_grouped", kernels::MOE_I8_GROUPED),
];
const K_GATHER: usize = 0;
const K_COUNT: usize = 1;
const K_SCAN: usize = 2;
const K_EMIT: usize = 3;
const K_GROUPED: usize = 4;

/// Slots per tile of `moe_i8_grouped.wgsl`.
const MR: u32 = 8;
const GATHER_COLS: u32 = 4;
const GROUPED_COLS: u32 = 4;

fn upload_u32(gpu: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

fn upload_f32(gpu: &Gpu, v: &[f32]) -> gpu_core::DeviceBuffer {
    gpu.storage_init("test", v)
}

fn read_u32(gpu: &Gpu, buf: &gpu_core::DeviceBuffer, n: usize) -> Vec<u32> {
    gpu.read(buf, n).iter().map(|f| f.to_bits()).collect()
}

fn random_packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                w |= u32::from((r.next_u32() % 256) as u8) << (8 * lane);
            }
            w
        })
        .collect()
}

fn random_scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect()
}

/// The tables on the device: `(tab, perm)`.
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

/// A host counting sort: the contract of the three kernels.
fn route_reference(ids: &[u32], ne: usize) -> (Vec<u32>, Vec<u32>) {
    let mut counts = vec![0u32; ne];
    for &i in ids {
        counts[i as usize] += 1;
    }
    let (mut tab, mut pos, mut tiles) = (vec![0u32; 2 * (ne + 1)], 0u32, 0u32);
    for e in 0..ne {
        tab[e] = pos;
        tab[ne + 1 + e] = tiles;
        pos += counts[e];
        tiles += counts[e].div_ceil(MR);
    }
    tab[ne] = pos;
    tab[2 * ne + 1] = tiles;
    let mut perm = vec![0u32; ids.len()];
    let mut next = tab[..ne].to_vec();
    for (s, &e) in ids.iter().enumerate() {
        perm[next[e as usize] as usize] = s as u32;
        next[e as usize] += 1;
    }
    (tab, perm)
}

#[test]
fn the_routing_tables_are_a_stable_counting_sort_of_the_ids() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("needs workgroup barriers");
    }
    let mut rng = Lcg::new(77);
    // (slots, experts, skew): the real 257 experts at decode and prefill sizes,
    // few experts with many slots each, one expert taking everything, and slot
    // counts that do not divide into the 256 runs.
    for &(slots, ne, skew) in &[(9u32, 257u32, false), (4608, 257, false), (1000, 5, false), (777, 257, true), (1, 3, false), (300, 257, false)] {
        let ids: Vec<u32> = (0..slots).map(|_| if skew { 3 } else { rng.next_u32() % ne }).collect();
        let (tab, perm) = route(&gpu, &ids, ne);
        gpu.poll_wait();
        let (want_tab, want_perm) = route_reference(&ids, ne as usize);
        assert_eq!(read_u32(&gpu, &tab, want_tab.len()), want_tab, "offsets for {slots} slots over {ne} experts");
        assert_eq!(read_u32(&gpu, &perm, ids.len()), want_perm, "order for {slots} slots over {ne} experts");
    }
}

/// The grouped kernel against the gather kernel, over the same inputs.
fn grouped_matches_gather(gpu: &Gpu, slots: usize, k: usize, n: usize, xdiv: usize, ne: usize, skew: usize, seed: u64) {
    let kg = k / 4;
    let rows = slots.div_ceil(xdiv);
    let xq = random_packed(rows * kg, seed + 1);
    let sx = random_scales(rows, seed + 2);
    let mut rng = Lcg::new(seed + 3);
    // `skew` > 0 sends that percentage of the slots to expert 0, leaving others empty or thin.
    let ids: Vec<u32> = (0..slots).map(|_| if (rng.next_u32() % 100) < skew as u32 { 0 } else { rng.next_u32() % ne as u32 }).collect();
    let wq = random_packed(ne * n * kg, seed + 4);
    let sw = random_scales(ne * n * (kg / 8), seed + 5);

    let (xb, sxb, idb, wb, swb) = (upload_u32(gpu, &xq), upload_f32(gpu, &sx), upload_u32(gpu, &ids), upload_u32(gpu, &wq), upload_f32(gpu, &sw));
    let want = gpu.storage((slots * n) as u64);
    let got = gpu.storage((slots * n) as u64);
    gpu.write(&got, &vec![0x7fc0_dead; slots * n]);
    let (tab, perm) = route(gpu, &ids, ne as u32);
    let max_tiles = ne as u32 + (slots as u32).div_ceil(MR);
    gpu.submit(
        &[],
        &[
            gpu.dispatch(K_GATHER, &[&xb, &sxb, &idb, &wb, &swb, &want], &[slots as u32, kg as u32, n as u32, xdiv as u32], Dispatch::Workgroups(slots as u32 * (n as u32).div_ceil(GATHER_COLS))),
            gpu.dispatch(K_GROUPED, &[&xb, &sxb, &tab, &perm, &wb, &swb, &got], &[max_tiles, kg as u32, n as u32, xdiv as u32, ne as u32], Dispatch::Workgroups(max_tiles * (n as u32).div_ceil(GROUPED_COLS))),
        ],
    );
    gpu.poll_wait();
    let (want, got) = (read_u32(gpu, &want, slots * n), read_u32(gpu, &got, slots * n));
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).take(5).collect();
    assert!(
        bad.is_empty(),
        "slots={slots} k={k} n={n} xdiv={xdiv} experts={ne}: first mismatches at {bad:?}: grouped {:?} gather {:?}",
        bad.iter().map(|&i| f32::from_bits(got[i])).collect::<Vec<_>>(),
        bad.iter().map(|&i| f32::from_bits(want[i])).collect::<Vec<_>>()
    );
}

#[test]
fn the_grouped_gemm_is_bit_identical_to_the_gather_gemv() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().numeric.int8_dot || !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("needs a packed int8 dot and workgroup barriers");
    }
    // (slots, k, n, xdiv, experts, % of slots sent to expert 0)
    for (i, &(slots, k, n, xdiv, ne, skew)) in [
        (9, 2048, 512, 9, 257, 0),    // decode: one token, a slot per expert, almost every tile a single slot
        (2304, 2048, 512, 9, 257, 0), // a 256-token prefill round: ~9 slots an expert
        (2304, 512, 2048, 1, 257, 0), // its down projection
        (900, 2048, 512, 9, 257, 60), // one hot expert (many tiles), the rest thin or empty
        (37, 64, 6, 1, 5, 0),         // ragged n and a ragged last tile
        (50, 32, 4, 1, 3, 0),         // one 32-group: fewer groups than lanes
        (40, 544, 12, 2, 7, 30),      // 17 groups: one lane takes a second
        (1, 96, 1, 1, 1, 0),
        (8, 64, 8, 1, 2, 100), // exactly one full tile
    ]
    .iter()
    .enumerate()
    {
        grouped_matches_gather(&gpu, slots, k, n, xdiv, ne, skew, 5000 + i as u64);
    }
}
