// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `moe_i8_grouped_mma` kernel (`kernels_cuda`,
//! `cu/moe_i8_grouped_mma.cu`) that `gpu_core::native_upgrade` substitutes for
//! the WGSL `moe_i8_grouped` on a CUDA device with int8 tensor cores.
//!
//! Swedish Embedded AB implements bit-exact tensor-core kernels for quantised
//! sparse-MoE inference. If your team needs expertise in moving an expert GEMM
//! onto tensor cores without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The substitution is invisible to every caller, so the claim it has to hold is
//! that the RAW BITS are identical to the WGSL tier. That is achievable because
//! a scale group's integer dot product is exact in either, and the native kernel
//! keeps the WGSL kernel's 16 lane accumulators and its fold order. The
//! reference is `moe_i8_grouped_ref`, the same source under a name the upgrade
//! table does not know, so both sides run on the same device from the same
//! inputs, differing in which kernel ran.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("moe_i8_grouped", kernels::MOE_I8_GROUPED),
    ("moe_i8_grouped_ref", kernels::MOE_I8_GROUPED),
    ("moe_route_count", kernels::MOE_ROUTE_COUNT),
    ("moe_route_scan", kernels::MOE_ROUTE_SCAN),
    ("moe_route_emit", kernels::MOE_ROUTE_EMIT),
];
const K_GROUPED: usize = 0;
const K_REF: usize = 1;
const K_COUNT: usize = 2;
const K_SCAN: usize = 3;
const K_EMIT: usize = 4;

const MR: u32 = 8;
const WGSL_COLS: u32 = 4;
/// Written around every output so an out-of-place write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda"
}

fn packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = Lcg::new(seed);
    (0..words)
        .map(|_| (0..4).fold(0u32, |w, lane| w | u32::from((r.next_u32() % 256) as u8) << (8 * lane)))
        .collect()
}

fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5 - if r.next_u32() % 7 == 0 { 2e-3 } else { 0.0 }).collect()
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

struct Case {
    slots: u32,
    k: u32,
    n: u32,
    xdiv: u32,
    ne: u32,
    /// Percentage of slots sent to expert 0 (a hot expert, many tiles; the rest thin or empty).
    skew: u32,
}

/// `(native, reference)` output bits for the same inputs.
fn run(gpu: &Gpu, c: &Case, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let (slots, kg, n, xdiv, ne) = (c.slots as usize, (c.k / 4) as usize, c.n as usize, c.xdiv as usize, c.ne as usize);
    let rows = slots.div_ceil(xdiv);
    let mut rng = Lcg::new(seed);
    let ids: Vec<u32> = (0..slots).map(|_| if rng.next_u32() % 100 < c.skew { 0 } else { rng.next_u32() % c.ne }).collect();
    let xq = upload_u32(gpu, &packed(rows * kg, seed + 1));
    let sx = gpu.storage_init("sx", &scales(rows, seed + 2));
    let wq = upload_u32(gpu, &packed(ne * n * kg, seed + 3));
    let sw = gpu.storage_init("sw", &scales(ne * n * (kg / 8), seed + 4));

    let ids_b = upload_u32(gpu, &ids);
    let counts = gpu.storage(ne as u64);
    let tab = gpu.storage(2 * (ne as u64 + 1));
    let perm = gpu.storage(slots.max(1) as u64);
    gpu.submit(
        &[],
        &[
            gpu.dispatch(K_COUNT, &[&ids_b, &counts], &[c.slots, c.ne], Dispatch::Workgroups(c.ne)),
            gpu.dispatch(K_SCAN, &[&counts, &tab], &[c.ne, MR], Dispatch::Workgroups(1)),
            gpu.dispatch(K_EMIT, &[&ids_b, &tab, &perm], &[c.slots, c.ne], Dispatch::Workgroups(c.ne)),
        ],
    );
    let tiles = c.ne + c.slots.div_ceil(MR);
    let params = [tiles, kg as u32, c.n, c.xdiv, c.ne];
    let go = |kind: usize| -> Vec<u32> {
        let out = gpu.storage((slots * n) as u64);
        gpu.write(&out, &vec![SENTINEL; slots * n]);
        gpu.submit(&[], &[gpu.dispatch(kind, &[&xq, &sx, &tab, &perm, &wq, &sw, &out], &params, Dispatch::Workgroups(tiles * c.n.div_ceil(WGSL_COLS)))]);
        gpu.poll_wait();
        gpu.read(&out, slots * n).iter().map(|f| f.to_bits()).collect()
    };
    (go(K_GROUPED), go(K_REF))
}

#[test]
fn the_native_kernel_is_selected_exactly_for_the_shapes_it_serves() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions || !gpu.caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("moe_i8_grouped needs workgroup barriers and a packed int8 dot");
    }
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_GROUPED, &[40, 512, 2048, 9, 257]), None, "only a CUDA device takes a native kernel");
        return brain_testutil::skip_unavailable("not a CUDA device");
    }
    if gpu.native_kernel_for(K_GROUPED, &[40, 512, 2048, 9, 257]).is_none() {
        return brain_testutil::skip_unavailable("this device has no int8 tensor cores (or native kernels are off)");
    }
    assert_eq!(gpu.native_kernel_for(K_GROUPED, &[40, 512, 2048, 9, 257]), Some("moe_i8_grouped_mma"));
    assert_eq!(gpu.native_kernel_for(K_GROUPED, &[40, 516, 2048, 9, 257]), None, "K not a whole number of scale groups");
    // The reference alias is deliberately not upgraded; it is what makes the
    // comparison a comparison.
    assert_eq!(gpu.native_kernel_for(K_REF, &[40, 512, 2048, 9, 257]), None);
}

/// BYTE-identical to the WGSL tier over every shape class the kernel's
/// arithmetic and addressing can get wrong.
#[test]
fn the_native_kernel_is_byte_identical_to_the_wgsl_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("native grouped MoE GEMM needs a CUDA device");
    }
    if gpu.native_kernel_for(K_GROUPED, &[1, 64, 8, 1, 1]).is_none() {
        return brain_testutil::skip_unavailable("this device has no int8 tensor cores (or native kernels are off)");
    }
    // k: 32 = one group (fewer than 16 lanes); 544 = 17 groups (one lane takes a
    // second, the second 16-block is ragged); 2048 = the real gate/up (4 blocks
    // of 16); 512 = the real down (one block). n: ragged against the warp's 16
    // and the block's 64. Tiles: a lone slot, full tiles, a ragged last tile, a
    // hot expert and empty experts.
    let cases = [
        Case { slots: 9, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 0 },
        Case { slots: 2304, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 0 },
        Case { slots: 2304, k: 512, n: 2048, xdiv: 1, ne: 257, skew: 0 },
        Case { slots: 900, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 60 },
        Case { slots: 37, k: 64, n: 6, xdiv: 1, ne: 5, skew: 0 },
        Case { slots: 50, k: 32, n: 4, xdiv: 1, ne: 3, skew: 0 },
        Case { slots: 40, k: 544, n: 12, xdiv: 2, ne: 7, skew: 30 },
        Case { slots: 77, k: 160, n: 100, xdiv: 3, ne: 9, skew: 20 },
        Case { slots: 1, k: 96, n: 1, xdiv: 1, ne: 1, skew: 0 },
        Case { slots: 8, k: 64, n: 17, xdiv: 1, ne: 2, skew: 100 },
        Case { slots: 64, k: 1056, n: 70, xdiv: 1, ne: 4, skew: 0 },
    ];
    for (i, c) in cases.iter().enumerate() {
        let (got, want) = run(&gpu, c, 9000 + i as u64);
        let bad: Vec<usize> = (0..got.len()).filter(|&j| got[j] != want[j]).take(5).collect();
        assert!(
            bad.is_empty(),
            "slots={} k={} n={} xdiv={} experts={}: first mismatches at {bad:?}: native {:?} wgsl {:?} - both fold the same terms in the same order, so this is a defect, not rounding",
            c.slots,
            c.k,
            c.n,
            c.xdiv,
            c.ne,
            bad.iter().map(|&j| f32::from_bits(got[j])).collect::<Vec<_>>(),
            bad.iter().map(|&j| f32::from_bits(want[j])).collect::<Vec<_>>()
        );
    }
}

/// Seeded random shapes, so the grid above is not the only place the tails and
/// the tile geometry have been looked at.
#[test]
fn random_shapes_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("native grouped MoE GEMM needs a CUDA device");
    }
    if gpu.native_kernel_for(K_GROUPED, &[1, 64, 8, 1, 1]).is_none() {
        return brain_testutil::skip_unavailable("this device has no int8 tensor cores (or native kernels are off)");
    }
    let mut r = Lcg::new(0x600d_5eed);
    for i in 0..40 {
        let c = Case {
            slots: 1 + r.next_u32() % 300,
            k: 32 * (1 + r.next_u32() % 70),
            n: 1 + r.next_u32() % 150,
            xdiv: 1 + r.next_u32() % 9,
            ne: 1 + r.next_u32() % 40,
            skew: r.next_u32() % 90,
        };
        let (got, want) = run(&gpu, &c, 500 + i);
        assert_eq!(got, want, "native grouped MoE GEMM differs at slots={} k={} n={} xdiv={} experts={} skew={}", c.slots, c.k, c.n, c.xdiv, c.ne, c.skew);
    }
}
