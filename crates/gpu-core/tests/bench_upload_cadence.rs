// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the wall-clock goes in a **streaming weight loader**: the per-call
//! fixed costs of `poll_wait` and `write_at`, measured directly rather than
//! inferred from a whole model load's total.
//!
//! `crates/gpu-core/tests/vram_overhead.rs` answered "how many bytes does an
//! upload cost". This answers the orthogonal question its throughput probes
//! opened: **how much does each CALL cost, independent of the bytes it
//! moves** - because a loader that streams one transformer block at a time
//! makes thousands of them per forward, and a fixed cost per call is exactly
//! the shape that leaves a GPU idle while a host core stays pinned.
//!
//! ```text
//! cargo test --release -p brain-gpu-core \
//!   -- --ignored --nocapture --test-threads=1 bench_upload_cadence
//! ```
//!
//! Every timed region is fenced with a real readback (`Gpu::read`), never a
//! bare `write`/`submit` return - `vram_overhead.rs`'s own history records
//! probes that measured nothing because `write_buffer` had not been submitted
//! yet. `#[ignore]`d and `--test-threads=1` for the one-device-per-process
//! rule, same shape as the other `bench_*` files here.

use std::time::Instant;

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[];

/// MiniMax-H3 at real dimensions (`H3TransformerConfig::real()`): the tensor
/// set one `transformer_blocks.{i}` load puts on the card, which a streaming
/// forward repeats 50 times. Sizes in f32 words.
const HIDDEN: usize = 5376;
const FFN: usize = 14336;
const INNER: usize = 56 * 128;

fn block_tensors() -> Vec<(&'static str, usize)> {
    vec![
        ("wq", INNER * HIDDEN),
        ("wk", INNER * HIDDEN),
        ("wv", INNER * HIDDEN),
        ("wo", HIDDEN * INNER),
        ("norm_q", 128),
        ("norm_k", 128),
        ("norm1", HIDDEN),
        ("norm2", HIDDEN),
        ("fc1_gate", FFN * HIDDEN),
        ("fc1_value", FFN * HIDDEN),
        ("fc2", HIDDEN * FFN),
    ]
}

fn gpu() -> Option<Gpu> {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return None;
    }
    Some(Gpu::new_wgpu(KERNELS))
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Probe 1: what does a bare `poll_wait` cost with nothing in flight, versus
/// one right after a real chunk write? If the idle call is already expensive
/// the cost is fixed per CALL, and calling it once per tensor rather than
/// once per block is a cadence bug, not a byte-throughput problem.
#[test]
#[ignore]
fn poll_wait_fixed_cost() {
    let Some(g) = gpu() else { return };
    let scratch = g.storage(1 << 20);
    let chunk = vec![0f32; 1 << 20];

    // Warm: open, allocate, and get one full submit+fence behind us so the
    // first-call driver setup is not attributed to the steady state.
    g.write_f32_at(&scratch, 0, &chunk);
    let _ = g.read(&scratch, 1);

    const N: usize = 200;
    let t = Instant::now();
    for _ in 0..N {
        g.poll_wait();
    }
    let idle = t.elapsed() / N as u32;

    let t = Instant::now();
    for _ in 0..N {
        g.write_f32_at(&scratch, 0, &chunk);
        g.poll_wait();
    }
    let after_write = t.elapsed() / N as u32;

    let t = Instant::now();
    for _ in 0..N {
        g.write_f32_at(&scratch, 0, &chunk);
    }
    let _ = g.read(&scratch, 1);
    let write_only = t.elapsed() / N as u32;

    println!("\n== probe 1: per-call fixed cost ==");
    println!("  bare poll_wait, nothing in flight : {:8.4} ms/call", ms(idle));
    println!("  write_at(4MiB) + poll_wait        : {:8.4} ms/call", ms(after_write));
    println!("  write_at(4MiB) alone, fenced once : {:8.4} ms/call", ms(write_only));
    println!("  => poll_wait's share when paired  : {:8.4} ms/call", ms(after_write) - ms(write_only));
}

/// Probe 2: does splitting a 308 MB tensor into ~75 four-MiB `write_at` calls
/// cost meaningfully more than fewer, larger calls? Sweeps the chunk size
/// across the value `paramstore::UPLOAD_CHUNK_WORDS` currently pins (1<<20
/// words = 4 MiB) up to a single unchunked write, each fenced.
#[test]
#[ignore]
fn upload_chunk_size_sweep() {
    let Some(g) = gpu() else { return };
    let numel = FFN * HIDDEN; // fc1_gate: 77.07 M words, 308 MB
    let data = vec![1.5f32; numel];
    let bytes = 4.0 * numel as f64;

    println!("\n== probe 2: chunk-size sweep, one 308 MB tensor ==");
    println!("  {:>12}  {:>10}  {:>8}  {:>9}  {:>10}", "chunk", "calls", "ms", "GB/s", "ms/call");
    // Repeat each arm so a cold staging-pool size class is visible as a
    // first-iteration outlier rather than being averaged into the result.
    for chunk_words in [1usize << 20, 1 << 22, 1 << 24, numel] {
        let calls = numel.div_ceil(chunk_words);
        for rep in 0..3 {
            let buf = g.storage(numel as u64);
            let t = Instant::now();
            g.write_f32_chunked(&buf, &data, chunk_words);
            let _ = g.read(&buf, 1); // fence: force submit + device wait
            let el = t.elapsed();
            let label = if rep == 0 { format!("{} MiB", chunk_words / (1 << 18)) } else { String::new() };
            println!(
                "  {:>12}  {:>10}  {:>8.1}  {:>9.2}  {:>10.4}{}",
                label,
                calls,
                ms(el),
                bytes / el.as_secs_f64() / 1e9,
                ms(el) / calls as f64,
                if rep == 0 { "  (cold)" } else { "" }
            );
        }
    }
}

/// Probe 3: the loader shape itself. One MiniMax-H3 block's 11 tensors,
/// uploaded then dropped, five times over - the inner loop of a streaming
/// forward. Compares `poll_wait` once per TENSOR (what
/// `paramstore::upload::Uploader::account` does today) against once per
/// BLOCK, holding everything else equal. The difference is the cadence cost
/// in isolation; the repeat count is what shows whether staging reuse warms
/// up or misses every iteration.
#[test]
#[ignore]
fn per_tensor_versus_per_block_poll_cadence() {
    let Some(g) = gpu() else { return };
    let tensors = block_tensors();
    let total_bytes: f64 = tensors.iter().map(|(_, n)| 4.0 * *n as f64).sum();
    let biggest = tensors.iter().map(|(_, n)| *n).max().unwrap();
    let data = vec![0.75f32; biggest];
    let chunk = 1usize << 20; // UPLOAD_CHUNK_WORDS

    println!("\n== probe 3: one block ({:.2} GB, {} tensors), load-then-drop ==", total_bytes / 1e9, tensors.len());

    for (label, poll_per_tensor) in [("poll per tensor", true), ("poll per block", false)] {
        // Warm the pool with one untimed block so both arms start equal.
        {
            let bufs: Vec<_> = tensors
                .iter()
                .map(|(_, n)| {
                    let b = g.storage(*n as u64);
                    g.write_f32_chunked(&b, &data[..*n], chunk);
                    b
                })
                .collect();
            let _ = g.read(&bufs[0], 1);
        }
        println!("  -- {label} --");
        for rep in 0..5 {
            let t = Instant::now();
            let mut t_write = std::time::Duration::ZERO;
            let mut t_poll = std::time::Duration::ZERO;
            let bufs: Vec<_> = tensors
                .iter()
                .map(|(_, n)| {
                    let b = g.storage(*n as u64);
                    let tw = Instant::now();
                    g.write_f32_chunked(&b, &data[..*n], chunk);
                    t_write += tw.elapsed();
                    if poll_per_tensor {
                        let tp = Instant::now();
                        g.poll_wait();
                        t_poll += tp.elapsed();
                    }
                    b
                })
                .collect();
            let tp = Instant::now();
            let _ = g.read(&bufs[0], 1); // the per-block fence both arms share
            t_poll += tp.elapsed();
            drop(bufs);
            let el = t.elapsed();
            println!(
                "    rep {rep}: total {:8.1} ms  ({:.2} GB/s) | write {:8.1} ms | poll/fence {:8.1} ms",
                ms(el),
                total_bytes / el.as_secs_f64() / 1e9,
                ms(t_write),
                ms(t_poll)
            );
        }
    }
}
