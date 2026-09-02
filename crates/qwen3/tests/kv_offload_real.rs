// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What host-RAM KV offload actually costs and buys on a REAL checkpoint and
//! a real card - the numbers `model::kv_offload`'s design rests on, measured
//! rather than derived.
//!
//! Swedish Embedded AB implements memory-tiered LLM serving for its clients.
//! If your team needs expertise in serving more concurrent sessions than the
//! VRAM you own would otherwise allow, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! `crates/qwen3/src/serve.rs`'s own tests pin CORRECTNESS (a demoted and
//! restored sequence decodes bit-identically, and a pool too small for the
//! batch still produces identical outputs) at a toy shape, which is the right
//! place for a bit-exactness gate and the wrong place to learn what a swap
//! costs: the transfer is bandwidth-bound, and a three-layer model moves a
//! thousandth of the bytes a real one does.
//!
//! Two things are measured here, both on the real 8B checkpoint:
//!
//! 1. **What one swap costs.** Demote and promote a real sequence's KV,
//!    timed end to end (gather/scatter dispatch + PCIe transfer), reported as
//!    ms and achieved GB/s against the bytes actually moved.
//! 2. **What it buys.** The same eight-request workload served twice against
//!    the SAME KV pool: once with the batch capped at what the pool can hold
//!    resident (no offload - today's only option), once with every request
//!    admitted at once and the pool's overflow swapped to host RAM. Wall
//!    time, tokens, and swap volume for both.
//!
//! ```text
//! BRAIN_QWEN3_CKPT=/path/to/model.brain.safetensors \
//!   cargo test --release -p brain-qwen3 --test kv_offload_real -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::time::Instant;

use model::kv_offload::KvOffload;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen3::config::QwenConfig;
use qwen3::serve::{Engine, Request, Scheduler};

/// Tokens of context each synthetic session holds - a realistic
/// retrieval-augmented prompt, and long enough that its KV is a real transfer
/// rather than a latency probe.
const PROMPT: usize = 1024;
/// Tokens each session generates.
const MAX_NEW: usize = 16;
/// Sessions offered.
const SESSIONS: usize = 8;
/// KV block size (tokens) - the serving default.
const BLOCK: u32 = 16;
/// Per-sequence context ceiling, in blocks.
const BLOCKS_PER_SEQ: u32 = 128; // 2048 tokens

fn prompt_of(seed: u32, vocab: u32) -> Vec<u32> {
    (0..PROMPT as u32).map(|i| (seed.wrapping_mul(7919).wrapping_add(i.wrapping_mul(31)) % (vocab - 1)) + 1).collect()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[test]
#[ignore = "hardware probe: needs a real Qwen3 checkpoint (BRAIN_QWEN3_CKPT) and a discrete GPU"]
fn host_kv_offload_swap_cost_and_concurrency_on_a_real_checkpoint() {
    let Ok(path) = std::env::var("BRAIN_QWEN3_CKPT") else {
        brain_testutil::skip("set BRAIN_QWEN3_CKPT to a brain-format Qwen3 checkpoint to run this");
        return;
    };

    // A pool deliberately smaller than the offered load: 320 blocks = 5120
    // cached tokens, against eight sessions of PROMPT+MAX_NEW each. Sized this
    // way on purpose - it is the shape a real box is in (the serving sizing in
    // `cli::resident_llm::pool_sizing` gives the pool room for about two full
    // contexts however many sessions the batch admits), just reached with a
    // small workload instead of a long one.
    let num_blocks = 320u32;
    let t0 = Instant::now();
    let mut eng = Engine::load(&path, BLOCK, num_blocks, SESSIONS as u32, BLOCKS_PER_SEQ, 512, true, true);
    let load_s = t0.elapsed().as_secs_f64();
    let vocab = PagedDecoder::vocab(&eng) as u32;
    let per_token = eng.kv_offload_bytes_per_token();
    let pool_tokens = num_blocks as u64 * BLOCK as u64;
    eprintln!(
        "\n=== host-RAM KV offload on {path} ===\nloaded in {load_s:.1}s | KV pool {:.2} GiB for {pool_tokens} cached tokens | \
         {:.1} KiB/token | one {PROMPT}+{MAX_NEW}-token session = {:.1} MiB of KV",
        gib(PagedDecoder::kv_pool_bytes(&eng)),
        per_token / 1024.0,
        per_token * (PROMPT + MAX_NEW) as f64 / (1 << 20) as f64,
    );

    // ---- 1. what one swap costs -------------------------------------
    eng.set_kv_offload_bytes(16 << 30);
    let mut table = BlockTable::new();
    let hidden = eng.prefill(&mut table, &prompt_of(1, vocab));
    let first = eng.admit_greedy(&hidden);
    let blocks = table.blocks().len() as u64;
    let bytes = blocks * BLOCK as u64 * (per_token as u64);

    let t = Instant::now();
    eng.demote_kv(1, &mut table).expect("demote");
    let out_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut table = eng.promote_kv(1).expect("promote");
    let in_s = t.elapsed().as_secs_f64();
    eprintln!(
        "swap out (device -> host): {:7.1} ms  {:5.2} GB/s\nswap in  (host -> device): {:7.1} ms  {:5.2} GB/s\n\
         one {PROMPT}-token session costs {:.1} MiB of host RAM and {:.0} ms to park + revive",
        out_s * 1e3,
        bytes as f64 / out_s / 1e9,
        in_s * 1e3,
        bytes as f64 / in_s / 1e9,
        bytes as f64 / (1 << 20) as f64,
        (out_s + in_s) * 1e3,
    );

    // The real-checkpoint rung of the correctness gate: decoding on restored
    // KV must produce exactly what decoding without the round trip does.
    let mut straight = BlockTable::new();
    let hidden = eng.prefill(&mut straight, &prompt_of(1, vocab));
    assert_eq!(eng.admit_greedy(&hidden), first, "the same prompt must admit the same first token");
    let mut a = (first, Vec::new());
    let mut b = (first, Vec::new());
    let steps = 8;
    let t = Instant::now();
    for _ in 0..steps {
        a.0 = eng.forward_batched_greedy(&mut [&mut straight], &[a.0])[0];
        a.1.push(a.0);
    }
    let decode_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    for _ in 0..steps {
        b.0 = eng.forward_batched_greedy(&mut [&mut table], &[b.0])[0];
        b.1.push(b.0);
    }
    assert_eq!(b.1, a.1, "decode after a real demote/promote round trip must be token-identical");
    // The denominator the whole design turns on: what ONE decode step costs
    // with this sequence's KV resident, against what moving that KV across the
    // bus costs. A per-block scheme inside the decode loop pays the second per
    // token; sequence-level offload pays it once per scheduling transition,
    // amortised over every token the rest of the batch produces meanwhile.
    eprintln!(
        "one decode step (batch 1, {PROMPT}-token context): {decode_ms:6.1} ms\nfetching this sequence's KV per token \
         instead would cost {:.0} ms - {:.0}x the step it would be serving",
        in_s * 1e3,
        in_s * 1e3 / decode_ms,
    );
    eng.release_table(&mut table);
    eng.release_table(&mut straight);

    // ---- 2. what it buys --------------------------------------------
    // Both arms serve the SAME eight sessions against the SAME engine and the
    // same pool. The only difference is whether the pool's overflow is a hard
    // cap on the batch (arm A) or spills to host RAM (arm B).
    let resident_capacity = (pool_tokens / (PROMPT + MAX_NEW) as u64) as usize;
    eprintln!("\nthe pool holds {resident_capacity} of the {SESSIONS} sessions resident at once");

    let reqs = |vocab: u32| {
        (0..SESSIONS as u32)
            .map(|i| Request { prompt: prompt_of(i + 10, vocab), max_new: MAX_NEW, eos: None })
            .collect::<Vec<_>>()
    };

    // Arm A: no offload; the batch is capped at what fits, so the rest wait.
    eng.set_kv_offload_bytes(0);
    let mut sched = Scheduler::new(eng, resident_capacity);
    for r in reqs(vocab) {
        sched.submit(r);
    }
    let t = Instant::now();
    let out_a = sched.run();
    let a_s = t.elapsed().as_secs_f64();
    let mut eng = sched.into_decoder();

    // Arm B: every session admitted at once, overflow parked in host RAM.
    eng.set_kv_offload_bytes(16 << 30);
    let mut sched = Scheduler::new(eng, SESSIONS);
    for r in reqs(vocab) {
        sched.submit(r);
    }
    let t = Instant::now();
    let out_b = sched.run();
    let b_s = t.elapsed().as_secs_f64();
    let stats = sched.offload_stats();

    assert_eq!(out_a.len(), SESSIONS, "arm A must serve every session");
    assert_eq!(out_b.len(), SESSIONS, "arm B must serve every session");
    let mut ids: Vec<&u64> = out_a.keys().collect();
    ids.sort();
    for id in ids {
        assert_eq!(out_b.get(id), out_a.get(id), "session {id}: swapping must not change a single token");
    }

    let tokens = (SESSIONS * MAX_NEW) as f64;
    eprintln!(
        "\narm A  batch capped at {resident_capacity} (no offload): {a_s:6.2} s  {:5.1} tok/s\n\
         arm B  all {SESSIONS} admitted, overflow in host RAM  : {b_s:6.2} s  {:5.1} tok/s\n\
         swaps: {} out / {} in, {} blocks moved each way, peak host RAM {:.1} MiB\n\
         outputs identical across both arms.",
        tokens / a_s,
        tokens / b_s,
        stats.demotions,
        stats.promotions,
        stats.blocks_out,
        stats.peak_bytes as f64 / (1 << 20) as f64,
    );
    assert!(stats.demotions > 0, "arm B must genuinely have swapped, or it measured nothing");
}

/// Qwen3-8B's exact KV geometry - 36 layers, 8 KV heads, `head_dim` 128, int8
/// KV - on a deliberately tiny body (small `d_model`/`d_ff`/vocab, random
/// weights).
///
/// A cached token's KV cost is a function of ONLY those numbers and the KV
/// dtype, so this measures the real transfer at the real size (74.2 KiB/token,
/// the figure the 8B checkpoint itself reports in the test above) without
/// needing that checkpoint's 33 GiB of weights - or its 2.49 GB fp32 embedding
/// table, which exceeds this card's 2 GiB storage-binding limit and is being
/// fixed elsewhere.
///
/// What it deliberately does NOT measure is serving throughput: this body does
/// almost no arithmetic, so any tok/s taken from it would describe the fixture
/// rather than a model. Swap cost, host bytes and bandwidth are the whole
/// claim.
fn kv_geometry_of_qwen3_8b(vocab: u32) -> QwenConfig {
    QwenConfig {
        vocab,
        block_size: 4096,
        n_layers: 36,
        d_model: 256,
        n_heads: 8,
        n_kv_heads: 8,
        head_dim: 128,
        d_ff: 256,
        rope_theta: 1.0e6,
        rms_eps: 1e-6,
        max_position_embeddings: 4096,
        tie_embeddings: true,
        qk_norm: true,
        attn_bias: false,
        lora: None,
    }
}

fn random_weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Rng::new(5);
    cfg.param_list()
        .into_iter()
        .map(|(name, count)| {
            let v = if name.contains("norm") {
                vec![1.0f32; count]
            } else {
                (0..count).map(|_| rng.next_gaussian() as f32 * 0.02).collect()
            };
            (name, v)
        })
        .collect()
}

#[test]
#[ignore = "hardware probe: needs a discrete GPU and an idle bus"]
fn what_one_kv_swap_costs_at_qwen3_8b_geometry() {
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("kv_offload_real: no discrete GPU on this box");
        return;
    }
    let cfg = kv_geometry_of_qwen3_8b(1024);
    let map = random_weights(&cfg);
    let (bs, blocks_per_seq, num_blocks) = (16u32, 256u32, 1024u32);
    let mut eng = Engine::from_map(cfg, &map, bs, num_blocks, 2, blocks_per_seq, 512, true, false);
    eng.set_kv_offload_bytes(8 << 30);
    let per_token = eng.kv_offload_bytes_per_token();
    let vocab = PagedDecoder::vocab(&eng) as u32;

    eprintln!(
        "\n=== one KV swap at Qwen3-8B geometry (36 layers, 8 KV heads, head_dim 128, int8 KV) ===\n\
         {:.1} KiB/token | KV pool {:.2} GiB for {} cached tokens",
        per_token / 1024.0,
        PagedDecoder::kv_pool_bytes(&eng) as f64 / (1u64 << 30) as f64,
        num_blocks as u64 * bs as u64,
    );

    // Warm up: the FIRST swap of an engine's life also builds the staging
    // buffers, and charging a one-time allocation to a bandwidth number is how
    // a probe reports a rate the mechanism does not have.
    {
        let mut warm = BlockTable::new();
        eng.prefill(&mut warm, &[7u32; 64]);
        eng.demote_kv(0, &mut warm).expect("warm-up demote");
        let mut warm = eng.promote_kv(0).expect("warm-up promote");
        eng.release_table(&mut warm);
    }

    for &tokens in &[1024usize, 4096] {
        let prompt: Vec<u32> = (0..tokens as u32).map(|i| (i * 31 + 7) % (vocab - 1) + 1).collect();
        let mut table = BlockTable::new();
        eng.prefill(&mut table, &prompt);
        let bytes = tokens as f64 * per_token;

        // Best of three round trips: a swap is a bandwidth measurement, and
        // the interesting number is what the mechanism can do, not what one
        // sample of a shared bus did.
        let (mut out_s, mut in_s) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..3 {
            let t = Instant::now();
            eng.demote_kv(1, &mut table).expect("demote");
            out_s = out_s.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            table = eng.promote_kv(1).expect("promote");
            in_s = in_s.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "{tokens:5} tokens = {:6.1} MiB of KV | park {:7.1} ms ({:4.2} GB/s) | revive {:7.1} ms ({:4.2} GB/s)",
            bytes / (1 << 20) as f64,
            out_s * 1e3,
            bytes / out_s / 1e9,
            in_s * 1e3,
            bytes / in_s / 1e9,
        );
        // A rate above what any host bus on this class of machine can carry
        // would mean the transfer never happened - the same impossibility
        // bound `gpu-core/tests/pcie_handoff.rs` guards its own probe with.
        assert!(bytes / out_s / 1e9 < 64.0 && bytes / in_s / 1e9 < 64.0, "a swap cannot beat the host bus");
        assert_eq!(eng.kv_offload_stats().bytes_resident, 0, "the round trips must leave nothing behind");
        eng.release_table(&mut table);
    }
}
