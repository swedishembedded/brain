// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed cost of the sparse-MoE int8 expert GEMMs at the Qwen3.6-35B-A3B
//! layer shape: the one-slot-per-pass gather kernel at decode batch sizes and the
//! grouped kernel at prefill-round sizes.
//!
//! Swedish Embedded AB implements measured throughput contracts for sparse-MoE
//! inference kernels. If your team needs expertise in knowing how close an expert
//! GEMM runs to the compute and bandwidth ceilings of your GPU, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a gate (that is `moe_i8_grouped.rs`):
//!
//! ```text
//! cargo test --release -p brain-gpu-core --test bench_moe_grouped -- --ignored --nocapture
//! ```
//!
//! Time is the kernel's own, from the backend's device events, never the host
//! clock (the card is shared). Routing is uniform random over the 256 routed
//! experts plus the shared one (every token), which is what a trained router
//! approximates. `GB/s` counts the weight bytes of the experts that actually got
//! a slot, once; `TMAC/s` counts every slot's multiply-accumulates.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("moe_i8_gemv_gather", kernels::MOE_I8_GEMV_GATHER),
    ("moe_route_count", kernels::MOE_ROUTE_COUNT),
    ("moe_route_scan", kernels::MOE_ROUTE_SCAN),
    ("moe_route_emit", kernels::MOE_ROUTE_EMIT),
    ("moe_i8_grouped", kernels::MOE_I8_GROUPED),
];

const NE: u32 = 257;
const TOP_K: u32 = 8;
const MR: u32 = 8;
const TRIALS: usize = 7;

fn timed(gpu: &Gpu, steps: &[gpu_core::Step], what: &str) -> Vec<f64> {
    gpu.submit(&[], steps);
    gpu.poll_wait();
    let mut per_call = Vec::new();
    for _ in 0..TRIALS {
        gpu.reset_kernel_times();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing");
        let (_, ms, calls) = rows.iter().find(|(name, _, _)| name.contains(what)).expect("the kernel ran");
        per_call.push(ms / *calls as f64);
    }
    per_call.sort_by(f64::total_cmp);
    per_call
}

fn run(gpu: &Gpu, tokens: u32, k: u32, n: u32, xdiv_per_row: bool) {
    let slots_per_row = TOP_K + 1;
    let slots = tokens * slots_per_row;
    let kg = k / 4;
    let mut rng = Lcg::new(7);
    let mut ids = Vec::with_capacity(slots as usize);
    for _ in 0..tokens {
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < TOP_K as usize {
            let e = rng.next_u32() % (NE - 1);
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend(chosen);
        ids.push(NE - 1);
    }
    let touched = {
        let mut seen = vec![false; NE as usize];
        ids.iter().for_each(|&e| seen[e as usize] = true);
        seen.iter().filter(|&&s| s).count() as u64
    };
    let xdiv = if xdiv_per_row { slots_per_row } else { 1 };
    let rows = slots.div_ceil(xdiv);
    let xq = gpu.storage(u64::from(rows * kg));
    gpu.write(&xq, &vec![0x01020304u32; (rows * kg) as usize]);
    let sx = gpu.storage(u64::from(rows));
    gpu.write_f32(&sx, &vec![1.0; rows as usize]);
    let wq = gpu.storage(u64::from(NE) * u64::from(n) * u64::from(kg));
    let sw = gpu.storage(u64::from(NE) * u64::from(n) * u64::from(kg / 8));
    let out = gpu.storage(u64::from(slots) * u64::from(n));
    let ids_b = gpu.storage(u64::from(slots));
    gpu.write(&ids_b, &ids);
    let counts = gpu.storage(u64::from(NE));
    let tab = gpu.storage(2 * (u64::from(NE) + 1));
    let perm = gpu.storage(u64::from(slots));

    let weight_bytes = touched * u64::from(n) * (u64::from(kg) * 4 + u64::from(kg / 8) * 4);
    let macs = f64::from(slots) * f64::from(n) * f64::from(k);
    let report = |name: &str, t: &[f64]| {
        println!(
            "{tokens:>6} tokens k={k:<5} n={n:<5} {name:<20} best {:>8.1} us  median {:>8.1} us   {:>6.0} GB/s  {:>6.2} TMAC/s",
            t[0] * 1e3,
            t[t.len() / 2] * 1e3,
            weight_bytes as f64 / (t[0] * 1e-3) / 1e9,
            macs / (t[0] * 1e-3) / 1e12
        );
    };

    let gather = [gpu.dispatch(0, &[&xq, &sx, &ids_b, &wq, &sw, &out], &[slots, kg, n, xdiv], Dispatch::Workgroups(slots * n.div_ceil(4)))];
    report("gather", &timed(gpu, &gather, "moe_i8_gemv_gather"));

    gpu.submit(
        &[],
        &[
            gpu.dispatch(1, &[&ids_b, &counts], &[slots, NE], Dispatch::Workgroups(NE)),
            gpu.dispatch(2, &[&counts, &tab], &[NE, MR], Dispatch::Workgroups(1)),
            gpu.dispatch(3, &[&ids_b, &tab, &perm], &[slots, NE], Dispatch::Workgroups(NE)),
        ],
    );
    let tiles = NE + slots.div_ceil(MR);
    let grouped = [gpu.dispatch(4, &[&xq, &sx, &tab, &perm, &wq, &sw, &out], &[tiles, kg, n, xdiv, NE], Dispatch::Workgroups(tiles * n.div_ceil(4)))];
    report("grouped", &timed(gpu, &grouped, "moe_i8_grouped"));

    let route = [
        gpu.dispatch(1, &[&ids_b, &counts], &[slots, NE], Dispatch::Workgroups(NE)),
        gpu.dispatch(2, &[&counts, &tab], &[NE, MR], Dispatch::Workgroups(1)),
        gpu.dispatch(3, &[&ids_b, &tab, &perm], &[slots, NE], Dispatch::Workgroups(NE)),
    ];
    gpu.submit(&[], &route);
    gpu.poll_wait();
    let mut total = 0.0;
    for name in ["moe_route_count", "moe_route_scan", "moe_route_emit"] {
        total += timed(gpu, &route, name)[0];
    }
    println!("{tokens:>6} tokens routing tables (count+scan+emit)       best {:>8.1} us", total * 1e3);
}

#[test]
#[ignore]
fn expert_gemm_cost_at_the_35b_a3b_layer_shape() {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    let sizes: Vec<u32> = std::env::var("BRAIN_BENCH_TOKENS").ok().map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect()).unwrap_or_else(|| vec![1, 16, 32, 256, 1024, 4096]);
    for tokens in sizes {
        run(&gpu, tokens, 2048, 512, true); // gate / up
        run(&gpu, tokens, 512, 2048, false); // down
    }
}
