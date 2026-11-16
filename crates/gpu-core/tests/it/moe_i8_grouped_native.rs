// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate of the native kernels `gpu_core::native_upgrade` substitutes for the WGSL `moe_i8_grouped` on a CUDA
//! device with int8 tensor cores (`kernels_cuda`, `cu/moe_i8_grouped_mma.cu`), as this device selects them: the
//! one-pass kernel, the staged runs, and the four-tile form the tuner may choose for a dense round.
//!
//! Swedish Embedded AB implements bit-exact tensor-core kernels for quantised
//! sparse-MoE inference. If your team needs expertise in moving an expert GEMM
//! onto tensor cores without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The exact kernels keep the WGSL kernel's 16 lane accumulators and its fold order, so the claim they hold is the RAW
//! BITS of the WGSL tier: a scale group's integer dot product is exact in either. The four-tile form reassociates the
//! sum and is held to the tolerance gate instead (`common/moe_grouped.rs`). The reference is `moe_i8_grouped_ref`, the
//! same source under a name the upgrade table does not know, so both sides run on the same device from the same
//! inputs, differing in which kernel ran. `moe_i8_grouped_candidates.rs` forces every candidate in turn.

use data::rng::{random_seed, Lcg};
use gpu_core::Gpu;

#[path = "../common/moe_grouped.rs"]
mod moe_grouped;
use moe_grouped::{run, Case, KERNELS, K_GROUPED, K_REF, MR};

/// The candidate that reassociates its per-group sum: held to the tolerance gate, every other one to the WGSL bits.
const REASSOCIATES: &str = "moe_i8_grouped_mma_wide";

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda"
}

/// The kernel this device's selection (tuned or default) runs `c` on, and whether its output matched the WGSL tier as
/// that kernel promises: bit for bit, or within the tolerance gate for [`REASSOCIATES`].
fn check(gpu: &Gpu, c: &Case, seed: u64) -> Result<&'static str, String> {
    let kernel = gpu.native_kernel_for(K_GROUPED, &c.params()).unwrap_or("moe_i8_grouped (wgsl)");
    let round = run(gpu, c, seed);
    if kernel == REASSOCIATES {
        round.within_tolerance(seed).map(|_| kernel)
    } else {
        let bad: Vec<usize> = (0..round.got.len()).filter(|&j| round.got[j] != round.want[j]).take(5).collect();
        match bad.is_empty() {
            true => Ok(kernel),
            false => Err(format!(
                "first mismatches at {bad:?}: native {:?} wgsl {:?} - both fold the same terms in the same order, so this is a defect, not rounding",
                bad.iter().map(|&j| f32::from_bits(round.got[j])).collect::<Vec<_>>(),
                bad.iter().map(|&j| f32::from_bits(round.want[j])).collect::<Vec<_>>()
            )),
        }
    }
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
    // A prefill round has many tiles of each expert: its weight rows are staged once and every tile of a run of the
    // expert reads them there. The 35B-A3B's 4096-row round is 36864 slots over 257 experts, its gate/up K 2048 and its
    // down K 512; a decode step of 64 rows (576 slots) keeps one pass of the weights per tile.
    // Which of the round's candidates runs is the tuner's measured choice (`gpu_core::native_tune`, the `moe_i8_grouped`
    // slot); a decode step has no class and keeps the exact one-pass kernel.
    let dense = 257 + 36864 / MR;
    let gate_up = gpu.native_kernel_for(K_GROUPED, &[dense, 512, 512, 9, 257]);
    assert!(matches!(gate_up, Some("moe_i8_grouped_mma_runs" | "moe_i8_grouped_mma_wide" | "moe_i8_grouped_mma")), "{gate_up:?}");
    let down = gpu.native_kernel_for(K_GROUPED, &[dense, 128, 2048, 1, 257]);
    assert!(matches!(down, Some("moe_i8_grouped_mma_runs_k512" | "moe_i8_grouped_mma_runs" | "moe_i8_grouped_mma_wide" | "moe_i8_grouped_mma")), "{down:?}");
    assert_eq!(gpu.native_kernel_for(K_GROUPED, &[257 + 576 / MR, 512, 512, 9, 257]), Some("moe_i8_grouped_mma"));
    let wide_k = gpu.native_kernel_for(K_GROUPED, &[dense, 520, 512, 9, 257]);
    assert!(matches!(wide_k, Some("moe_i8_grouped_mma_wide" | "moe_i8_grouped_mma")), "K past the staged rows' room: {wide_k:?}");
    // The reference alias is deliberately not upgraded; it is what makes the
    // comparison a comparison.
    assert_eq!(gpu.native_kernel_for(K_REF, &[40, 512, 2048, 9, 257]), None);
}

/// What the selected kernel promises - the WGSL tier's bytes, or the tolerance gate for the one that reassociates -
/// over every shape class the kernels' arithmetic and addressing can get wrong.
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
        // Rounds dense enough for the staged runs: the real 4096-row gate/up and down, a hot expert beside thin
        // ones, and experts of two or three tiles each, so runs span expert boundaries and end mid-expert.
        Case { slots: 36864, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 0 },
        Case { slots: 36864, k: 512, n: 2048, xdiv: 1, ne: 257, skew: 0 },
        Case { slots: 9000, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 50 },
        Case { slots: 1000, k: 544, n: 70, xdiv: 3, ne: 50, skew: 0 },
        Case { slots: 333, k: 96, n: 33, xdiv: 1, ne: 7, skew: 10 },
    ];
    for (i, c) in cases.iter().enumerate() {
        if let Err(why) = check(&gpu, c, 9000 + i as u64) {
            panic!("slots={} k={} n={} xdiv={} experts={}: {why}", c.slots, c.k, c.n, c.xdiv, c.ne);
        }
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
    let mut r = Lcg::new(random_seed());
    for i in 0..40 {
        let c = Case {
            slots: 1 + r.next_u32() % 300,
            k: 32 * (1 + r.next_u32() % 70),
            n: 1 + r.next_u32() % 150,
            xdiv: 1 + r.next_u32() % 9,
            ne: 1 + r.next_u32() % 40,
            skew: r.next_u32() % 90,
        };
        if let Err(why) = check(&gpu, &c, 500 + i) {
            panic!("native grouped MoE GEMM at slots={} k={} n={} xdiv={} experts={} skew={}: {why}", c.slots, c.k, c.n, c.xdiv, c.ne, c.skew);
        }
    }
}
