// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **CPU-vs-wgpu agreement for this decoder's own dispatch sequence.**
//!
//! Every other test in this crate runs ONE backend at a time: `gradcheck.rs`
//! and `parity.rs` pin the CPU JIT, `chunked_prefill.rs` runs the CPU and the
//! ambient default separately and compares each against ITSELF (a chunked
//! replay against a per-token replay on the same device). None of them can see
//! a kernel that is correct on one backend and wrong on the other, because no
//! comparison in this crate ever crosses that line.
//!
//! That gap matters for this architecture specifically. The decode path
//! dispatches a sparse MoE - `model::moe::router_fwd_kind` (a top-k selection
//! with a per-row reduction), 64 per-expert SwiGLU triples, and
//! `model::moe::shared_expert_fwd` - plus `model::block::gqa_chunk_step`'s
//! cached attention. A reduction written with a workgroup barrier can return
//! a silently wrong (or all-zero) result on exactly one backend, and a suite
//! that only ever compares a backend against itself passes anyway.
//!
//! So this file asserts the two backends agree, over BOTH of this decoder's
//! two independent tapes:
//!
//! 1. `logits_all` - the flat batched forward (`ROPE`/`gqa_fwd`, dense
//!    `moe::expert_fwd` over every expert).
//! 2. `prefill_chunked` + `step` - the KV-cached decode tape
//!    (`ROPE_AT`/`gqa_chunk_step`), which shares no dispatch with (1).
//!
//! and that greedy decode picks the SAME token ids on both, which is the
//! property a served run actually depends on: a logit difference below the
//! argmax margin is fp32 noise, one above it is a different document.
//!
//! `DeepseekV2Config::tiny()` keeps every dimension distinct (`d_model` 12 =
//! 3 heads x head_dim 4, 5 routed experts top-2, 2 shared, dense ff 21 vs MoE
//! ff 7, vocab 19), so a transposed or swapped axis cannot hide behind two
//! equal numbers.
//!
//! Skipped, never failed, when this box has no GPU to compare against
//! (`MOE_SKIP_GPU_TESTS`, or `Gpu::new_wgpu` resolving to no real card).

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::{DeepseekV2, Sizes, PIPELINES};
use gpu_core::Gpu;

/// The bound both backends must agree inside.
///
/// MEASURED, not guessed: the worst disagreement over all three tests below
/// is 2.24e-8 (batched forward), with the two decode tapes at 7.45e-9 to
/// 1.49e-8 - i.e. one to three fp32 ulps at these magnitudes, which is what a
/// different reduction order inside a GEMM costs and nothing more.
///
/// perf-number: a numerical-tolerance ratio, not a throughput claim - 1e-6
/// sits ~45x above that measured worst case, and still an order of magnitude
/// BELOW the weakest real signal `chunked_prefill.rs`'s mutation table
/// records for this decoder (5.4e-6 for a chunk-relative RoPE), so a real
/// divergence cannot hide under it.
const BOUND: f32 = 1e-6;

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// `Some(gpu)` when a real wgpu device is worth comparing against.
fn wgpu_or_skip() -> Option<Gpu> {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip("MOE_SKIP_GPU_TESTS");
        return None;
    }
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip("no discrete GPU - a software rasteriser would compare CPU against CPU");
        return None;
    }
    Some(Gpu::new_wgpu(PIPELINES))
}

/// The prompt both backends run. Long enough to cross a MoE layer's router
/// several times with different top-k picks, short enough for the tiny
/// fixture's 13-position ceiling.
fn prompt() -> Vec<u32> {
    let vocab = DeepseekV2Config::tiny().vocab();
    (0..9).map(|i| (i * 3 + 1) % vocab).collect()
}

/// The batched tape (`logits_all`): dense `moe::expert_fwd` over every expert,
/// `ROPE` + `gqa_fwd`, one flat `[t, vocab]` slab.
#[test]
fn the_batched_forward_agrees_across_backends() {
    let Some(gpu) = wgpu_or_skip() else { return };
    let cfg = DeepseekV2Config::tiny();
    let init = deepseek2::init_weights(&cfg, 7);
    let ids = prompt();

    let m_cpu = DeepseekV2::new_on(Gpu::new_cpu(PIPELINES), cfg.clone(), 1, ids.len() as u32, &init, false);
    let want = m_cpu.logits_all(&ids);

    let m_gpu = DeepseekV2::new_on(gpu, cfg, 1, ids.len() as u32, &init, false);
    let got = m_gpu.logits_all(&ids);

    assert!(got.iter().all(|x| x.is_finite()), "the wgpu batched forward produced non-finite logits");
    let err = maxabs(&got, &want);
    println!("batched forward, cpu vs wgpu: maxabs {err:e} over {} logits", got.len());
    assert!(err < BOUND, "batched forward diverges between backends: maxabs {err:e}");
}

/// The decode tape (`prefill_chunked` + `step`): `ROPE_AT`, `gqa_chunk_step`
/// over the persistent KV cache, and the same MoE layer at one row per step.
/// Shares no dispatch with the batched tape above, so a kernel that is wrong
/// on wgpu in only one of the two is still caught.
#[test]
fn the_kv_cached_decode_agrees_across_backends() {
    let Some(gpu) = wgpu_or_skip() else { return };
    let cfg = DeepseekV2Config::tiny();
    let init = deepseek2::init_weights(&cfg, 7);
    let ids = prompt();
    let tail: Vec<u32> = (0..3).map(|i| (i * 5 + 2) % cfg.vocab()).collect();
    let sizes = Sizes { b: 1, t: 1, ctx: 13, chunk: 4, batched: false };

    let build = |g: Gpu| DeepseekV2::new_sized(g, cfg.clone(), sizes, &init, false);
    let m_cpu = build(Gpu::new_cpu(PIPELINES));
    let m_gpu = build(gpu);

    let want_prefill = m_cpu.prefill_chunked(&ids);
    let got_prefill = m_gpu.prefill_chunked(&ids);
    assert!(got_prefill.iter().all(|x| x.is_finite()), "the wgpu chunked prefill produced non-finite logits");
    let err = maxabs(&got_prefill, &want_prefill);
    println!("chunked prefill, cpu vs wgpu: maxabs {err:e}");
    assert!(err < BOUND, "chunked prefill diverges between backends: maxabs {err:e}");

    for (i, &tok) in tail.iter().enumerate() {
        let want = m_cpu.step(tok);
        let got = m_gpu.step(tok);
        let err = maxabs(&got, &want);
        println!("decode step {i}, cpu vs wgpu: maxabs {err:e}");
        assert!(err < BOUND, "decode step {i} diverges between backends: maxabs {err:e}");
    }
}

/// The property a served run depends on: not "the logits are close" but "the
/// same tokens come out". A per-logit difference under the argmax margin is
/// fp32 noise; one over it is a different document, and only this assertion
/// can tell them apart.
#[test]
fn greedy_decode_picks_the_same_ids_on_both_backends() {
    let Some(gpu) = wgpu_or_skip() else { return };
    let cfg = DeepseekV2Config::tiny();
    let init = deepseek2::init_weights(&cfg, 7);
    let ids = prompt();
    let n_new = 4u32;
    let sizes = Sizes { b: 1, t: 1, ctx: 13, chunk: 4, batched: false };

    let want = DeepseekV2::new_sized(Gpu::new_cpu(PIPELINES), cfg.clone(), sizes, &init, false).generate_greedy_kv(&ids, n_new);
    let got = DeepseekV2::new_sized(gpu, cfg, sizes, &init, false).generate_greedy_kv(&ids, n_new);

    assert_eq!(&got[..ids.len()], &ids[..], "the prompt must come back verbatim");
    assert_eq!(got, want, "wgpu greedy decode picked different ids than the CPU JIT");
    println!("greedy decode agrees on all {} ids", got.len());
}
