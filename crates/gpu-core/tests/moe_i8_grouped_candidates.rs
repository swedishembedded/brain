// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every candidate of the grouped expert GEMM's tuned slot (`moe_i8_grouped`), each forced in turn on the rounds it
//! serves: the exact kernels give the WGSL tier's bytes, the four-tile form that reassociates its per-group sum passes
//! the tolerance gate (`common/moe_grouped.rs`). The tuner may choose any of them for a dense round, so each one is
//! held to its promise here whatever this device's measurement picked. A binary of its own: a forced candidate is
//! process-wide.
//!
//! Swedish Embedded AB implements solutions for selecting the fastest correct kernel per device for its clients. If
//! your team needs expertise in sparse mixture-of-experts inference on tensor cores then you can procure our services
//! by sending an email to info@swedishembedded.com.

#[path = "common/moe_grouped.rs"]
mod moe_grouped;
use moe_grouped::{run, Case, KERNELS, K_GROUPED};

/// The slot's candidates and whether each keeps the WGSL tier's bits.
const CANDIDATES: [(&str, bool); 4] = [("moe_i8_grouped_mma_runs_k512", true), ("moe_i8_grouped_mma_runs", true), ("moe_i8_grouped_mma_wide", false), ("moe_i8_grouped_mma", true)];

#[test]
fn every_candidate_keeps_its_promise_on_the_rounds_it_serves() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || !gpu.caps().numeric.int8_dot || gpu.native_kernel_for(K_GROUPED, &[1, 64, 8, 1, 1]).is_none() {
        return brain_testutil::skip_unavailable("the native grouped MoE GEMM needs a CUDA device with int8 tensor cores");
    }
    // Dense rounds (two or more tiles an expert): the 35B-A3B's 4096-row gate/up and down, a hot expert beside thin
    // ones, runs that span expert boundaries, ragged K, N and tiles, and a K past the staged runs' room.
    let cases = [
        Case { slots: 36864, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 0 },
        Case { slots: 36864, k: 512, n: 2048, xdiv: 1, ne: 257, skew: 0 },
        Case { slots: 9000, k: 2048, n: 512, xdiv: 9, ne: 257, skew: 50 },
        Case { slots: 1000, k: 544, n: 70, xdiv: 3, ne: 50, skew: 0 },
        Case { slots: 333, k: 96, n: 33, xdiv: 1, ne: 7, skew: 10 },
        Case { slots: 700, k: 2240, n: 40, xdiv: 7, ne: 9, skew: 0 },
        Case { slots: 50, k: 32, n: 4, xdiv: 1, ne: 3, skew: 0 },
    ];
    let mut exercised = vec![0usize; CANDIDATES.len()];
    for (ci, &(name, exact)) in CANDIDATES.iter().enumerate() {
        let _pin = gpu_core::pin_candidate("moe_i8_grouped", name);
        for (i, c) in cases.iter().enumerate() {
            // A candidate that does not serve the round is not what the dispatch runs: skip it.
            if gpu.native_kernel_for(K_GROUPED, &c.params()) != Some(name) {
                continue;
            }
            exercised[ci] += 1;
            let round = run(&gpu, c, 7000 + i as u64);
            let shape = format!("{name} at slots={} k={} n={} xdiv={} experts={} skew={}", c.slots, c.k, c.n, c.xdiv, c.ne, c.skew);
            if exact {
                let bad = (0..round.got.len()).filter(|&j| round.got[j] != round.want[j]).count();
                assert_eq!(bad, 0, "{shape}: {bad} outputs differ from the WGSL tier's bits");
            } else {
                let (rms, max) = round.within_tolerance(7000 + i as u64).unwrap_or_else(|why| panic!("{shape}: {why}"));
                println!("{shape}: error {rms:.2} (rms) / {max:.2} (max) of the WGSL tier's");
            }
        }
    }
    for (&(name, _), &n) in CANDIDATES.iter().zip(&exercised) {
        assert!(n > 0, "{name} served none of the rounds: the gate checked nothing of it");
    }
}

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before any test runs.
brain_testutil::configure_test_binary!();
