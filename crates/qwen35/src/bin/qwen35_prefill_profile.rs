// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prefill (time-to-first-token) throughput of `qwen35::serve::Engine`'s two
//! prompt-replay shapes on the SAME model instance: the pre-M25 per-token
//! dispatch loop (`Qwen35::step`, one full `run_decode_step` per prompt token)
//! against M25's chunked rounds (`Qwen35::prefill_chunked`).
//!
//! Companion to `qwen35_decode_profile`, which prices the `n = 1` STEADY-STATE
//! token loop. Prefill is the opposite regime: many rows share one weight
//! stream, so the question is how much of the per-token dispatch and
//! weight-read cost the batched shape amortises away - a question only a
//! before/after on the same weights and the same device answers.
//!
//! Weights are SYNTHETIC (`qwen35::init::init_weights`), at the real
//! Qwen3.8-27B PER-LAYER shape (`d_model = 5120`, `intermediate_size =
//! 17408`, 24 query / 4 KV heads of 256, 48 GDN value heads of 128) but with a
//! reduced layer count and vocabulary, because `serve::Engine` builds a plain
//! FP32 `Qwen35` and the real 64-layer FP32 model is ~108 GB - past this box's
//! two 24 GB P40s regardless of how the prompt is replayed. The layer-type mix
//! is exact (`full_attention_interval = 4`, so one GQA layer per three GDN
//! ones), so the per-layer arithmetic each path pays is the real model's; only
//! the layer COUNT is scaled. Numbers from this tool are per-layer-honest and
//! whole-model-indicative, not a served 27B time-to-first-token.
//!
//! Usage:
//!   qwen35_prefill_profile [prompt_tokens] [chunk] [n_layers]
//!
//! Defaults: 512 prompt tokens, chunk 256 (`serve::MAX_PREFILL_TOKENS`), 4
//! layers.

use std::time::Instant;

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let prompt_len: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let chunk: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_layers: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    // Real per-layer shape, reduced depth and vocabulary - see module doc.
    let cfg = Qwen35Config { n_layers, vocab: 4096, block_size: prompt_len + 64, ..Qwen35Config::qwen38_27b() };
    println!(
        "config: {n_layers} layers (d_model {}, ff {}, {} q-heads x {} + {} GDN v-heads x {}), vocab {}, cap {}",
        cfg.d_model, cfg.intermediate_size, cfg.n_heads, cfg.head_dim, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.vocab, cfg.block_size
    );

    let t0 = Instant::now();
    let init = qwen35::init::init_weights(&cfg, 7);
    let floats: usize = init.values().map(|v| v.len()).sum();
    println!("synthetic weights: {:.2} GB fp32, generated in {:.1}s", floats as f64 * 4.0 / 1e9, t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let m = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &init);
    drop(init);
    println!("upload + pipeline build: {:.1}s", t0.elapsed().as_secs_f64());

    let prompt: Vec<u32> = (0..prompt_len).map(|i| (i * 7 + 3) % cfg.vocab).collect();

    // Warm the device (first dispatch of a shape pays pipeline/scratch costs
    // neither path should be charged for).
    m.reset_decode_cache();
    let _ = m.prefill_chunked(&prompt[..chunk.min(prompt_len) as usize], chunk);

    // Pre-M25: one full decode-step dispatch per prompt token.
    m.reset_decode_cache();
    let t0 = Instant::now();
    let mut per_token_last = Vec::new();
    for &tok in &prompt {
        per_token_last = m.step(tok);
    }
    let per_token = t0.elapsed().as_secs_f64();

    // M25: bounded chunked rounds.
    m.reset_decode_cache();
    let t0 = Instant::now();
    let chunked_last = m.prefill_chunked(&prompt, chunk);
    let chunked = t0.elapsed().as_secs_f64();

    let err = per_token_last.iter().zip(&chunked_last).fold(0.0f32, |mx, (x, y)| mx.max((x - y).abs()));
    println!();
    println!("prompt {prompt_len} tokens, chunk {chunk}");
    println!("  per-token replay : {per_token:8.3} s   {:9.1} tok/s   {:7.2} ms/token", prompt_len as f64 / per_token, per_token * 1e3 / prompt_len as f64);
    println!("  chunked prefill  : {chunked:8.3} s   {:9.1} tok/s   {:7.2} ms/token", prompt_len as f64 / chunked, chunked * 1e3 / prompt_len as f64);
    println!("  speedup          : {:.2}x", per_token / chunked);
    println!("  last hidden state agreement (maxabs): {err:e}");
}
