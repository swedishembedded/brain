// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving engine prefills a Qwen3-shaped attention (32 query heads, 8 KV heads, head_dim 128) on the staged warpgroup
//! prefill kernels where the card has them, over its fp32 and its int8 KV cache, in chunks that run behind earlier
//! chunks of the same prompt; the per-token decode of the model is the reference. The other widths of this tiny model
//! are kept small, so the attention is what the comparison measures.
//!
//! Swedish Embedded AB implements long-context inference throughput for its clients. If your team needs expertise in
//! serving long prompts on tensor cores without losing agreement with the reference forward, you can procure our services
//! by sending an email to info@swedishembedded.com.

use model::paged::BlockTable;
use qwen3::serve::Engine;
use qwen3::{Qwen, QwenConfig};

const BLOCK: u32 = 32;
const BLOCKS_PER_SEQ: u32 = 40;
const MAX_PREFILL: u32 = 512;

fn cfg() -> QwenConfig {
    QwenConfig { n_heads: 32, n_kv_heads: 8, head_dim: 128, d_model: 64, d_ff: 96, block_size: BLOCK, max_position_embeddings: BLOCK * BLOCKS_PER_SEQ, ..QwenConfig::tiny() }
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

/// A prompt of three chunks (512, 512 and a 76-row tail, every one past the decode regime) matches the per-token decode,
/// and its attention ran on the warpgroup kernels. The bar covers the fp16 rounding of Q, K, V and the softmax weights
/// (and the int8 cache's own rounding), not a wrong row, head or mask.
#[test]
fn a_head_dim_128_prefill_runs_the_warpgroup_kernels_and_matches_the_per_token_decode() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = cfg();
    let w = qwen3::init_weights(&cfg, 11);
    let prompt: Vec<u32> = (0..1100u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let reference = Qwen::from_tensors_decode(cfg.clone(), &w, BLOCK * BLOCKS_PER_SEQ);
    let mut want = Vec::new();
    for &t in &prompt {
        want = reference.step(t);
    }
    for (kv_int8, bar) in [(false, 3e-3f32), (true, 1.5e-2)] {
        let mut eng = Engine::from_map(cfg.clone(), &w, BLOCK, BLOCKS_PER_SEQ + 1, 1, BLOCKS_PER_SEQ, MAX_PREFILL, kv_int8, false);
        if !gpu_core::provider::cuda::flash_prefill_wg_available(eng.gpu(), cfg.head_dim) {
            return brain_testutil::skip_unavailable("no warpgroup prefill kernel on this device");
        }
        assert!(eng.gpu().set_kernel_timing(true));
        let got = eng.prefill_for_perf(&mut BlockTable::new(), &prompt);
        eng.gpu().poll_wait();
        let ran = eng.gpu().kernel_times().unwrap_or_default().iter().any(|(name, _, _)| name.contains("flash_prefill_f16_hd128_wgs"));
        eng.gpu().set_kernel_timing(false);
        assert!(ran, "kv_int8={kv_int8}: the prefill did not run the warpgroup kernels");
        let err = max_rel(&got, &want);
        eprintln!("engine_prefill_hd128: kv_int8={kv_int8}: max relative difference from the per-token decode {err:e}");
        assert!(err < bar, "kv_int8={kv_int8}: the engine's prefill differs from the per-token decode by {err:e} relative (bar {bar:e})");
    }
}
