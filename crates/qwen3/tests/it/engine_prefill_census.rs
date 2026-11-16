// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping an accelerator busy through the prefill of a served language
// model for its clients. If your team needs expertise in GPU inference servers then you can procure our services by
// sending an email to info@swedishembedded.com.

//! What the host does per prefill of the paged serving engine on CUDA, from the call totals of the engine's own
//! handle. A half-precision engine rounds each chunk's activations to a bf16 scratch for the library GEMM; with the
//! handle's freed blocks going straight back to the driver, every one of those was a `cuMemAlloc` and a `cuMemFree`
//! (which waits for the whole device), and a 16K-token Qwen3-4B prefill left the card idle for over half its time. A
//! steady-state prefill takes its temporaries from blocks the handle keeps.
//!
//! Each layer's temporaries are also the next layer's: built as one tape, an 8K-row chunk of the 4B held every layer's
//! bf16 casts at once, the stream-ordered pool grew by gigabytes before the first kernel ran and gave them back to the
//! driver at the next synchronise, so every served prompt paid the growth again (knowledge 476). A steady-state
//! prefill takes no block from the pool at all.

use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen3::serve::Engine;
use qwen3::{Dtype, QwenConfig};

#[test]
fn a_steady_state_prefill_does_not_allocate_from_the_driver() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    // Library-eligible widths (16-element rows), an untied head, a prompt of several chunks.
    let cfg = QwenConfig { d_model: 64, d_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 64, block_size: 256, max_position_embeddings: 256, tie_embeddings: false, ..QwenConfig::tiny() };
    let w = qwen3::init_weights(&cfg, 11); // seed: any weights serve, the census counts calls
    const CHUNK: u32 = 32;
    let mut e = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 64, 1, 16, CHUNK, true, Dtype::BF16);
    let prompt: Vec<u32> = (0..(4 * CHUNK)).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let calls = |e: &Engine| e.gpu().call_totals().expect("the CUDA backend counts its host calls");

    // Every pass reads the whole prompt: the prefix cache is emptied after each, so no pass is a one-block tail.
    let cold_prefill = |e: &mut Engine| {
        let mut t = BlockTable::new();
        let hidden = e.prefill_for_perf(&mut t, &prompt);
        e.release_for_perf(&mut t);
        e.reclaim_prefix(u32::MAX);
        hidden
    };
    // Warm-up: kernels compile and the prefill's blocks are created once.
    let first = cold_prefill(&mut e);

    for pass in 1..4 {
        let before = calls(&e);
        let hidden = cold_prefill(&mut e);
        let after = calls(&e);
        assert_eq!(hidden, first, "pass {pass}: the same prompt gave another hidden state");
        assert_eq!(after.device_alloc_calls, before.device_alloc_calls, "pass {pass}: a steady-state prefill called cuMemAlloc, which waits for the device on the matching free");
        assert_eq!(after.pool_alloc_calls, before.pool_alloc_calls, "pass {pass}: a steady-state prefill took blocks from the pool, so a layer's temporaries were not the next layer's");
    }
}

/// A server records and freezes its decode steps before the first request (`PagedDecoder::prepare_decode`): a decode
/// step at a prepared row count then records and freezes nothing, where the first step at each new row count used to
/// capture and instantiate a graph while the card waited.
#[test]
fn a_decode_step_at_a_prepared_row_count_records_nothing() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let cfg = QwenConfig { d_model: 64, d_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 64, block_size: 256, max_position_embeddings: 256, tie_embeddings: false, ..QwenConfig::tiny() };
    let w = qwen3::init_weights(&cfg, 13); // seed: any weights serve
    let mut e = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 64, 4, 16, 32, true, Dtype::BF16);
    e.prepare_decode(4);
    assert_eq!(e.decode_tapes(), (4, 4), "every row count up to the engine's batch is recorded and frozen");
    let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
    for (i, t) in tables.iter_mut().enumerate() {
        let prompt: Vec<u32> = (0..(5 + i as u32)).map(|j| (j * 7 + i as u32) % cfg.vocab).collect();
        e.prefill_for_perf(t, &prompt);
    }
    let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
    let prepared = e.forward_batched_greedy(&mut refs, &[1, 2, 3]);
    assert_eq!(e.decode_tapes(), (4, 4), "the step recorded a tape");

    // The same step on an engine that records on first use gives the same tokens.
    let mut fresh = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 64, 4, 16, 32, true, Dtype::BF16);
    let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
    for (i, t) in tables.iter_mut().enumerate() {
        let prompt: Vec<u32> = (0..(5 + i as u32)).map(|j| (j * 7 + i as u32) % cfg.vocab).collect();
        fresh.prefill_for_perf(t, &prompt);
    }
    let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
    assert_eq!(fresh.forward_batched_greedy(&mut refs, &[1, 2, 3]), prepared, "a prepared tape decodes as a tape recorded on first use");
}

/// A batch wider than the decode regime is recorded too: an engine of 40 sequences records and freezes a decode tape
/// for every row count up to 40, and a 40-row step replays one (it used to launch every dispatch on its own, about 2400
/// launches a step for Qwen3-4B at 64 rows), with the tokens a step recorded on first use gives.
#[test]
fn a_decode_batch_wider_than_the_decode_regime_is_recorded() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let cfg = QwenConfig { d_model: 64, d_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 64, block_size: 256, max_position_embeddings: 256, tie_embeddings: false, ..QwenConfig::tiny() };
    let w = qwen3::init_weights(&cfg, 13); // seed: any weights serve
    let rows = 40usize;
    let step = |e: &mut Engine| {
        let mut tables: Vec<BlockTable> = (0..rows).map(|_| BlockTable::new()).collect();
        for (i, t) in tables.iter_mut().enumerate() {
            let prompt: Vec<u32> = (0..(5 + (i % 7) as u32)).map(|j| (j * 7 + i as u32) % cfg.vocab).collect();
            e.prefill_for_perf(t, &prompt);
        }
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let inputs: Vec<u32> = (0..rows as u32).map(|i| i % cfg.vocab).collect();
        e.forward_batched_greedy(&mut refs, &inputs)
    };
    let mut e = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 256, rows as u32, 16, 32, true, Dtype::BF16);
    e.prepare_decode(rows);
    assert_eq!(e.decode_tapes(), (rows, rows), "every row count up to the engine's batch is recorded and frozen");
    let prepared = step(&mut e);
    assert_eq!(e.decode_tapes(), (rows, rows), "the 40-row step replayed its prepared tape");
    let mut fresh = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 256, rows as u32, 16, 32, true, Dtype::BF16);
    assert_eq!(step(&mut fresh), prepared, "a prepared tape decodes as a tape recorded on first use");
    assert_eq!(fresh.decode_tapes().0, 1, "the step recorded its row count on first use");
}
