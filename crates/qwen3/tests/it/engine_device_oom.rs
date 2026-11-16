// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving language models that keep every admitted request through a
// full card for its clients. If your team needs expertise in GPU inference servers then you can procure our services
// by sending an email to info@swedishembedded.com.

//! Spec: a device allocation that fails inside a step of the paged Qwen3 engine never takes the process down and never
//! fails a request that fits the card alone. The engine's step unwinds with the typed device-out-of-memory payload, the
//! scheduler sends the sequences the step carried back to the queue, and once the card has room they complete with the
//! tokens a card that never ran out gives (knowledge 435). A request that fails while it is alone on the card is
//! refused with a reason a client can retry on, and the engine serves the next one.

use model::serve::{RejectReason, Request, Scheduler};
use qwen3::serve::Engine;
use qwen3::{Dtype, QwenConfig};
use std::collections::HashMap;

fn cfg() -> QwenConfig {
    QwenConfig { d_model: 64, d_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 64, block_size: 256, max_position_embeddings: 256, tie_embeddings: false, ..QwenConfig::tiny() }
}

/// A fresh engine of 4 sequences and 32-row prefill rounds, so each prompt below is read in several rounds.
fn scheduler(gpu: &gpu_core::Gpu) -> Scheduler<Engine> {
    let c = cfg();
    let w = qwen3::init_weights(&c, 17); // seed: any weights serve, the test compares two runs of the same engine
    Scheduler::new(Engine::from_map_tier_on(gpu, c, &w, 16, 64, 4, 16, 32, true, Dtype::BF16), 4)
}

const PROMPTS: [usize; 3] = [70, 45, 90];

fn submit(sched: &mut Scheduler<Engine>) -> Vec<u64> {
    PROMPTS.iter().map(|&len| sched.submit(Request { prompt: (0..len as u32).map(|i| (i * 7 + len as u32) % 64).collect(), max_new: 6, stop: Vec::new() })).collect()
}

/// What a run ended with: every completion by id, and every refusal.
type Outcome = (HashMap<u64, Vec<u32>>, Vec<(u64, RejectReason)>);

fn drain(sched: &mut Scheduler<Engine>) -> Outcome {
    let (mut done, mut rejected) = (HashMap::new(), Vec::new());
    for _ in 0..1000 {
        if !sched.pending() {
            return (done, rejected);
        }
        let report = sched.step_report();
        rejected.extend(report.rejected);
        done.extend(report.completed);
    }
    panic!("the scheduler did not go idle: {} waiting, {} running", sched.waiting_len(), sched.running_len());
}

/// The card is full at the first prefill round: no allocation past what the built engine holds. The step is sent back,
/// and once the card has room every request completes with the tokens of a run that never ran out.
#[test]
fn a_prefill_round_that_runs_the_card_out_of_memory_is_sent_back_and_completes() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let mut reference = scheduler(&gpu);
    let ids = submit(&mut reference);
    let (expected, rejected) = drain(&mut reference);
    assert!(rejected.is_empty(), "{rejected:?}");

    let mut sched = scheduler(&gpu);
    let built = sched.decoder_mut().gpu().memory_usage().expect("the CUDA backend accounts for its allocations").live;
    sched.decoder_mut().gpu().set_device_cap(Some(built));
    let again = submit(&mut sched);
    let report = sched.step_report();
    assert!(report.completed.is_empty() && report.rejected.is_empty(), "the full card completed or refused a request: {:?}", report.rejected);
    assert_eq!(sched.waiting_len(), PROMPTS.len(), "every sequence the failed round carried is back in the queue");

    sched.decoder_mut().gpu().set_device_cap(None);
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "a request that fits the card was refused: {rejected:?}");
    for (id, first) in again.iter().zip(&ids) {
        assert_eq!(done[id], expected[first], "request {id} gets the tokens a card with room gives it");
    }
}

/// A request that runs the card out while it is alone is refused with the retryable device-out-of-memory reason, and
/// the engine serves the next request once the card has room.
#[test]
fn a_request_the_full_card_cannot_hold_alone_is_refused_and_the_engine_serves_on() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let mut reference = scheduler(&gpu);
    let first = reference.submit(Request { prompt: (0..70).map(|i| (i * 5 + 1) % 64).collect(), max_new: 6, stop: Vec::new() });
    let (expected, _) = drain(&mut reference);

    let mut sched = scheduler(&gpu);
    let built = sched.decoder_mut().gpu().memory_usage().expect("the CUDA backend accounts for its allocations").live;
    sched.decoder_mut().gpu().set_device_cap(Some(built));
    let doomed = sched.submit(Request { prompt: (0..70).map(|i| (i * 5 + 1) % 64).collect(), max_new: 6, stop: Vec::new() });
    let (done, rejected) = drain(&mut sched);
    assert!(done.is_empty(), "a request completed on a card with no room for its prefill");
    assert!(matches!(rejected.as_slice(), [(id, RejectReason::DeviceOutOfMemory { .. })] if *id == doomed), "{rejected:?}");

    sched.decoder_mut().gpu().set_device_cap(None);
    let next = sched.submit(Request { prompt: (0..70).map(|i| (i * 5 + 1) % 64).collect(), max_new: 6, stop: Vec::new() });
    let (done, rejected) = drain(&mut sched);
    assert!(rejected.is_empty(), "{rejected:?}");
    assert_eq!(done[&next], expected[&first], "the engine serves the next request as an engine that never ran out");
}
