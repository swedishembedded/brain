// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for host-side launch overhead of
// GPU inference servers for its clients. If your team needs expertise in
// making a dispatch thread keep a GPU busy then you can procure our services
// by sending an email to info@swedishembedded.com.

//! What the host does per chunked-prefill round of the INT8 decoder on CUDA:
//! driver launch calls, graph replays and allocations, from the process-wide
//! call totals. A served 27B prompt is hundreds of such rounds, and the
//! dispatch thread, not the card, bounds them once a round costs the host more
//! than the device; this census is where a round that starts allocating, or
//! stops replaying, shows up as a number.

use backend_cuda::live::call_totals;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

#[test]
fn a_steady_state_prefill_round_allocates_nothing_on_the_device() {
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let cfg = Qwen35Config { block_size: 128, max_position_embeddings: 128, ..Qwen35Config::tiny_i8() };
    let init = qwen35::init::init_weights(&cfg, 7); // seed: any weights serve, the census counts calls
    let m = Qwen35::new_on_dt(gpu.share(), cfg.clone(), 1, cfg.block_size, &init, &TierPolicy::uniform(Dtype::I8));
    let tokens: Vec<u32> = (0..cfg.block_size).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    const ROUND: usize = 16;

    // Warm-up: kernels compile, scratch and uniforms are created.
    m.reset_decode_cache();
    m.prefill_chunked(&tokens[..ROUND], ROUND as u32);

    let rounds = tokens.len().div_ceil(ROUND) as u64;
    for pass in 0..3 {
        m.reset_decode_cache();
        let before = call_totals();
        let t = std::time::Instant::now();
        m.prefill_chunked(&tokens, ROUND as u32);
        let wall = t.elapsed();
        let after = call_totals();
        let per = |a: u64, b: u64| (a - b) / rounds;
        if pass > 0 {
            assert_eq!(after.device_alloc_calls, before.device_alloc_calls, "pass {pass}: a steady-state prefill round called cuMemAlloc, which synchronises the device");
        }
        println!(
            "pass {pass}: per round of {ROUND}: {} launches, {} graph replays, {} cuMemAlloc, {} pool allocs; {:?} per round",
            per(after.host_launches, before.host_launches),
            per(after.graph_replays, before.graph_replays),
            per(after.device_alloc_calls, before.device_alloc_calls),
            per(after.pool_alloc_calls, before.pool_alloc_calls),
            wall / rounds as u32
        );
    }
}
