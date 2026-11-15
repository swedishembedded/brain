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

use model::paged::BlockTable;
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

    // Warm-up: kernels compile and the prefill's blocks are created once.
    let mut t = BlockTable::new();
    let first = e.prefill_for_perf(&mut t, &prompt);
    e.release_for_perf(&mut t);

    for pass in 1..3 {
        let before = calls(&e);
        let mut t = BlockTable::new();
        let hidden = e.prefill_for_perf(&mut t, &prompt);
        let after = calls(&e);
        e.release_for_perf(&mut t);
        assert_eq!(hidden, first, "pass {pass}: the same prompt gave another hidden state");
        assert_eq!(after.device_alloc_calls, before.device_alloc_calls, "pass {pass}: a steady-state prefill called cuMemAlloc, which waits for the device on the matching free");
    }
}
