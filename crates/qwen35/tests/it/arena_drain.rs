// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pooled prefill round that does not drain the device between layers computes EXACTLY what the same round
//! computes with the drain.
//!
//! Swedish Embedded AB implements low-latency GPU inference for its clients, where an idle gap between two layers
//! of a prefill round is throughput nobody gets back. If your team needs expertise in keeping the device fed
//! while a host records the next layer, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The scratch arena recycles each layer's temporaries for the next layer. On a backend that executes one handle's
//! submissions in order (CUDA) the drain between layers is not what keeps that safe, so the pooled round skips it.
//! This holds the claim at the level of the round's output bits, over several chunks of a prompt (so the second
//! chunk's recycled slots follow the first's) and with the fused front end on and off. The device-level hazard
//! test is `gpu-core/tests/cuda_arena_reuse.rs`.
//!
//! CUDA backend only; skips elsewhere.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

const ROWS: u32 = 16;
const CHUNKS: u32 = 3;

fn prefill(m: &Qwen35, tokens: &[u32], drain: Option<bool>, fusion: bool) -> Vec<f32> {
    m.set_chunk_arena_drain(drain);
    m.set_decode_fusion(fusion);
    m.reset_decode_cache();
    m.prefill_chunked(tokens, ROWS)
}

#[test]
fn a_pooled_prefill_round_without_the_layer_drain_is_bit_identical_to_one_with_it() {
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    assert!(gpu.submissions_ordered(), "the CUDA backend orders one handle's submissions");
    let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny_i8() };
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, ROWS * CHUNKS, &init, &TierPolicy::uniform(Dtype::I8));
    m.set_chunk_arena_min_rows(1);
    let tokens: Vec<u32> = (0..ROWS * CHUNKS).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    for fusion in [true, false] {
        let drained = prefill(&m, &tokens, Some(true), fusion);
        let undrained = prefill(&m, &tokens, Some(false), fusion);
        let default = prefill(&m, &tokens, None, fusion);
        assert!(drained.iter().all(|x| x.is_finite()), "non-finite hidden state");
        assert!(drained.iter().zip(&undrained).all(|(a, b)| a.to_bits() == b.to_bits()), "fusion={fusion}: dropping the drain changed the round");
        assert!(drained.iter().zip(&default).all(|(a, b)| a.to_bits() == b.to_bits()), "fusion={fusion}: the default differs from the drained round");
    }
}
