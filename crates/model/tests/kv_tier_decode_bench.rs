// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One GQA layer's decode attention at a long context, per KV tier: the fused
//! split-key kernel on f32, bf16 and int8 planes against the scores/softmax/
//! apply triad it replaces - interleaved min-of-N, the repo's convention for a
//! measurement on a card other processes share (a kernel that was descheduled
//! mid-run reports the other tenant's time; the minimum is the run it was not).
//!
//! Swedish Embedded AB implements long-context inference serving for clients
//! whose GPU decides how many users one card carries. If your team needs
//! expertise in decode attention that runs at the memory roofline over a
//! compact KV cache, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a gate (the gates are `kv_tier.rs`). Run
//! on a GPU with room for `batch * context` rows of K and V per tier:
//!
//! ```text
//! BENCH_CONTEXT=131072 BENCH_BATCHES=1,4 BENCH_REPS=21 \
//!   cargo test --release -p brain-model --test kv_tier_decode_bench -- --ignored --nocapture
//! ```
//!
//! Qwen3.8-27B's attention shape (24 heads, 4 kv heads, head_dim 256) at
//! `BENCH_CONTEXT` tokens (default 131072), the layer's K and V planes
//! allocated for `batch` sequences each (one block per sequence, the layout
//! the GGUF resident uses). Reports per-layer milliseconds, the bytes the
//! layer reads, and the effective bandwidth against them; a decode step is 16
//! such layers.

use std::time::Instant;

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::paged_attention_fused;
use model::kv_tier::{kernel_list, FlashDecodeShape, KvAppend, KvKernels, KvPlane, KvTier};
use model::ops::PagedDecodeShape;

const N_HEADS: u32 = 24;
const N_KV: u32 = 4;
const HEAD_DIM: u32 = 256;
const KV_STRIDE: u32 = N_KV * HEAD_DIM;

fn env_list(name: &str, default: &[u32]) -> Vec<u32> {
    std::env::var(name).ok().map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect()).unwrap_or_else(|| default.to_vec())
}

fn u32s(g: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = g.storage(v.len().max(1) as u64);
    g.write(&b, v);
    b
}

/// A plane of `rows` token rows holding repeating random data: `chunk` random
/// rows appended again and again, so building 128k rows costs a few dispatches.
fn plane(g: &Gpu, tier: KvTier, rows: u32, seed: u64) -> KvPlane {
    let plane = KvPlane::new(g, tier, rows as u64, KV_STRIDE as u64, HEAD_DIM as u64);
    let kernels = KvKernels::resolve(g, tier).unwrap();
    let chunk = 4096u32.min(rows);
    let src = g.storage_init("src", &Lcg::new(seed).vec_scaled((chunk * KV_STRIDE) as usize, 1.0));
    let mut at = 0u32;
    while at < rows {
        let n = chunk.min(rows - at);
        let blocks = u32s(g, &vec![0u32; n as usize]);
        let offsets = u32s(g, &(at..at + n).collect::<Vec<_>>());
        g.submit(&[], &[kernels.append(g, &plane, &src, &blocks, &offsets, KvAppend { batch: n, kv_stride: KV_STRIDE, block_size: rows, head_dim: HEAD_DIM })]);
        at += n;
    }
    g.poll_wait();
    plane
}

/// One submit's wall time, `poll_wait`-bracketed, in milliseconds.
fn once(g: &Gpu, steps: &[gpu_core::Step]) -> f64 {
    let t0 = Instant::now();
    g.submit(&[], steps);
    g.poll_wait();
    t0.elapsed().as_secs_f64() * 1e3
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "a measurement; needs a GPU with room for batch x context rows of K and V"]
fn fused_decode_against_the_triad_per_layer_at_a_long_context() {
    let context = env_list("BENCH_CONTEXT", &[131_072])[0];
    let batches = env_list("BENCH_BATCHES", &[1, 4]);
    let reps = env_list("BENCH_REPS", &[21])[0] as usize;

    let mut list = kernel_list(&[]);
    list.push(("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED));
    let g = Gpu::new_gpu(&list);
    if !paged_attention_fused(&g, true, false, HEAD_DIM, 0) {
        brain_testutil::skip_unavailable("this device does not run the fused decode kernel");
        return;
    }

    println!("one GQA layer, {N_HEADS} heads / {N_KV} kv heads / head_dim {HEAD_DIM}, context {context}, min of {reps} interleaved reps");
    println!("{:>5} {:>6} {:>10} {:>10} {:>9} {:>9}", "batch", "arm", "ms/layer", "MiB read", "GB/s", "x triad");
    for &batch in &batches {
        let rows = context; // one block of `context` rows per sequence
        let pool_rows = batch * rows;
        let lens = vec![context; batch as usize];
        let block_tables = u32s(&g, &(0..batch).collect::<Vec<_>>());
        let seq_lens = u32s(&g, &lens);
        let q = g.storage_init("q", &Lcg::new(7).vec_scaled((batch * N_HEADS * HEAD_DIM) as usize, 0.5));
        let ctx = g.storage((batch * N_HEADS * HEAD_DIM) as u64);

        // Arms: the f32 triad, then the fused kernel on each tier.
        struct Arm {
            name: &'static str,
            steps: Vec<gpu_core::Step>,
            bytes: f64,
            ms: Vec<f64>,
        }
        let mut arms: Vec<Arm> = Vec::new();
        let mut planes = Vec::new();
        for tier in KvTier::ALL {
            let (k, v) = (plane(&g, tier, pool_rows, 11), plane(&g, tier, pool_rows, 13));
            let kernels = KvKernels::resolve(&g, tier).unwrap();
            // Bytes the layer READS: every live row of K and of V, once.
            let bytes = 2.0 * tier.plane_bytes((batch * context) as u64, KV_STRIDE as u64, HEAD_DIM as u64) as f64;
            if tier == KvTier::F32 {
                let scores = g.storage((batch * N_HEADS * context) as u64);
                let probs = g.storage((batch * N_HEADS * context) as u64);
                let shape = PagedDecodeShape { batch, n_heads: N_HEADS, group: N_HEADS / N_KV, head_dim: HEAD_DIM, block_size: rows, kv_stride: KV_STRIDE, cap: context, max_bt: 1, scale: 1.0 / 16.0 };
                let softmax = g.kernel_index("decode_softmax_batched").unwrap();
                let steps = vec![
                    kernels.scores(&g, &q, &k, &block_tables, &seq_lens, &scores, shape),
                    g.step(softmax, &[&scores, &seq_lens, &probs], &[batch, N_HEADS, context], batch * N_HEADS),
                    kernels.apply(&g, &probs, &v, &block_tables, &seq_lens, &ctx, shape),
                ];
                arms.push(Arm { name: "triad", steps, bytes, ms: Vec::new() });
            }
            let fused = FlashDecodeShape { batch, n_heads: N_HEADS, n_kv_heads: N_KV, head_dim: HEAD_DIM, block_size: rows, max_bt: 1, cap: context };
            let name = match tier {
                KvTier::F32 => "f32",
                KvTier::Bf16 => "bf16",
                KvTier::Int8 => "int8",
            };
            arms.push(Arm { name, steps: kernels.flash_decode(&g, &q, &k, &v, &block_tables, &seq_lens, &ctx, fused), bytes, ms: Vec::new() });
            planes.push((k, v));
        }

        for arm in arms.iter() {
            once(&g, &arm.steps); // compile and warm
        }
        for _ in 0..reps {
            for arm in arms.iter_mut() {
                let ms = once(&g, &arm.steps);
                arm.ms.push(ms);
            }
        }
        let triad_ms = min(&arms[0].ms);
        for arm in &arms {
            let ms = min(&arm.ms);
            println!("{:>5} {:>6} {:>10.3} {:>10.0} {:>9.0} {:>8.1}x", batch, arm.name, ms, arm.bytes / (1 << 20) as f64, arm.bytes / ms / 1e6, triad_ms / ms);
        }
        drop(planes);
    }
}
