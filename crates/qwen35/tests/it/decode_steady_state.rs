// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A steady-state Qwen3.8 decode token costs the host a few graph launches, not
//! thousands of driver calls, and computes exactly what it computed before.
//!
//! Swedish Embedded AB implements low-latency single-stream inference for its
//! clients. If your team needs expertise in making a token loop device-bound
//! instead of driver-bound, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Measured on the real 27B INT8 resident before this was fixed: 2500 kernel
//! launches per token, 2500 `cuMemAlloc`/`cuMemFree` pairs (each free waits for
//! the whole device, so the host and the card ran in lockstep), and a device
//! that was busy for a quarter of the wall time. Two properties together cure
//! it and both are asserted here, on the CUDA backend only (it skips
//! elsewhere):
//!
//! - **no driver allocation in a steady-state token** - the step's temporaries
//!   and its per-token inputs live at the same device addresses every token;
//! - **only replayed graphs and no per-dispatch launch** - the step is recorded
//!   once (in a few chunks, so the card starts before the host has built the
//!   rest) and replayed, the same chunks every token.
//!
//! The second test is what keeps the first two honest: reusing buffers without
//! zeroing them is only legal if no kernel reads what it did not write, so the
//! same token sequence decoded twice through the same recycled buffers must be
//! BIT-identical, not close.

use backend_cuda::call_totals;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use std::sync::{Mutex, MutexGuard};

/// The call totals are process-global.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn model(gpu: Gpu) -> (Qwen35, Vec<u32>) {
    let cfg = Qwen35Config::tiny_i8();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, t, &init, &TierPolicy::uniform(Dtype::I8));
    let tokens = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    (m, tokens)
}

#[test]
fn a_steady_state_decode_token_allocates_nothing_and_only_replays_graphs() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let (m, tokens) = model(gpu);
    m.reset_decode_cache();
    // One eager token, one capturing token, and one more for anything that is
    // allocated lazily on first use.
    for &t in &tokens[..4] {
        m.step(t);
    }
    let before = call_totals();
    let measured = 6;
    for &t in &tokens[4..4 + measured] {
        m.step(t);
    }
    let after = call_totals();
    assert_eq!(after.device_alloc_calls - before.device_alloc_calls, 0, "a steady-state token called cuMemAlloc");
    assert_eq!(after.host_launches - before.host_launches, 0, "a steady-state token launched kernels one at a time");
    // A token is a few graphs, not one: the layer stack is issued in chunks so
    // the card starts before the host has built the whole step. What matters is
    // that it is replays and nothing else, the same number every token.
    let replays = after.graph_replays - before.graph_replays;
    assert!(replays >= measured as u64, "a token must replay at least one graph");
    assert_eq!(replays % measured as u64, 0, "{replays} replays over {measured} tokens: the chunks are not cut the same way every token");
}

#[test]
fn a_decode_through_recycled_buffers_is_bit_identical_to_the_first() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let (m, tokens) = model(gpu);
    let run = |m: &Qwen35| {
        m.reset_decode_cache();
        tokens.iter().map(|&t| m.step(t)).collect::<Vec<_>>()
    };
    let first = run(&m);
    let second = run(&m);
    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert!(a.iter().all(|x| x.is_finite()), "position {i}: non-finite hidden state");
        assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "position {i}: a second decode through recycled scratch differs from the first");
    }
}
