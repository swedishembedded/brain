// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A prefill round with the native fused front end and quantisation epilogues computes EXACTLY what the same round
//! computes as the chain of dispatches they replace, in fewer launches.
//!
//! Swedish Embedded AB implements bit-exact fused kernels for quantised LLM inference. If your team needs
//! expertise in cutting the launch count and memory traffic of a prefill round without moving a single output bit,
//! you can procure our services by sending an email to info@swedishembedded.com.
//!
//! `decode_fusion.rs` holds the decode tape to the same claim; a prefill round shares the kernels but not the
//! tape (it carries the next layer's normed activation across the layer boundary, and fuses the SwiGLU product's
//! quantisation), so it owes its own gate. The identity is bitwise: the hidden state after the round is the same
//! with the fusion on and off. The second test checks the fusion engaged, since the identity holds trivially on a
//! device that was never offered the kernels.
//!
//! CUDA backend only; skips elsewhere.

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

const ROWS: u32 = 24;

fn model(gpu: Gpu, cfg: &Qwen35Config) -> (Qwen35, Vec<u32>) {
    let init = qwen35::init::init_weights(cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, cfg.block_size, &init, &TierPolicy::uniform(Dtype::I8));
    // A pooled round (the production regime): the scratch arena is open for every layer.
    m.set_chunk_arena_min_rows(1);
    let tokens = (0..ROWS).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    (m, tokens)
}

fn prefill(m: &Qwen35, tokens: &[u32], fusion: bool) -> Vec<f32> {
    m.set_decode_fusion(fusion);
    m.reset_decode_cache();
    m.prefill_chunked(tokens, ROWS)
}

#[test]
fn the_fused_prefill_round_is_bit_identical_to_the_unfused_one() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let (m, tokens) = model(gpu, &Qwen35Config::tiny_i8());
    let fused = prefill(&m, &tokens, true);
    let unfused = prefill(&m, &tokens, false);
    assert!(fused.iter().all(|x| x.is_finite()), "non-finite hidden state");
    assert!(fused.iter().zip(&unfused).all(|(x, y)| x.to_bits() == y.to_bits()), "the fused prefill round differs from the unfused chain");
}

#[test]
fn the_fused_prefill_round_launches_fewer_kernels() {
    let _s = serial();
    if !gpu_core::native_kernels_enabled() {
        brain_testutil::skip_unavailable("native kernels are withheld in this process");
        return;
    }
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let cfg = Qwen35Config::tiny_i8();
    let (m, tokens) = model(gpu, &cfg);
    let launches = |fusion: bool| {
        prefill(&m, &tokens, fusion); // the first round pays kernel compilation and capture; count the next
        let before = call_totals().host_launches;
        prefill(&m, &tokens, fusion);
        call_totals().host_launches - before
    };
    let (fused, unfused) = (launches(true), launches(false));
    // Each layer folds away at least the three launches its second norm front is, and the first layer's front is
    // one launch instead of three.
    assert!(unfused >= fused + 3 * cfg.n_layers as u64, "the fused round launched {fused} kernels, the unfused one {unfused}: the fusion did not engage");
}
