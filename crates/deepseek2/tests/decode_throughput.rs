// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The served decoder's real shape, on both backends: same ids, measured
//! cost.**
//!
//! `tests/generate.rs` gates the decode LOOP against llama.cpp, but at a
//! 2-token prompt and a 10-token context - a shape that says nothing about
//! what a page of OCR costs. `tests/parity.rs` gates one forward. Neither
//! builds what `deepseek2ocr::caps::Session::load` builds: `ctx = 8192`,
//! `chunk = 512`, `batched = false`, a ~283-row prompt (the 273-row image
//! block plus BOS plus an instruction), then a run of KV-cached steps.
//!
//! This file does, on the CPU Cranelift JIT and on a real wgpu/Vulkan card,
//! and asserts the two produce the SAME ids. That is the assertion that
//! matters for a backend switch: a served page must not change because the
//! decoder moved to the GPU.
//!
//! It also PRINTS prefill seconds, seconds/token and tokens/second for each
//! backend. Those are reported, never asserted - a throughput floor baked into
//! a test binary becomes a flaky gate on the first busy machine, and the point
//! of measuring here is to have a number to cite, from the same build the
//! correctness assertion ran on, rather than an estimate.
//!
//! The prompt is synthetic token ids, not a real page: the decoder half of
//! this model has no image in it (the vision tower's output enters through
//! `enable_mm_splice`, which does not change the dispatch count of a single
//! row), so a synthetic prompt of the right LENGTH costs exactly what a real
//! one does. `crates/deepseek2ocr/tests/real_weight_long_context.rs` makes the
//! same argument for its constant fill image.

use std::time::Instant;

use deepseek2::model::{DeepseekV2, Sizes};

/// `deepseek2ocr::caps::default_ctx_len()`'s own default - the checkpoint's
/// declared `max_position_embeddings`.
const CTX: u32 = 8192;
/// `deepseek2ocr::caps::default_chunk_len()`'s own default.
const CHUNK: u32 = 512;
/// BOS + the real 273-row image block + a real instruction's tokens, i.e. the
/// prompt length `crates/deepseek2ocr/src/prompt.rs` actually assembles.
const PROMPT_ROWS: usize = 283;
/// Generated tokens to time. Long enough that the per-step cost dominates the
/// timer's own resolution on either backend, short enough that the CPU arm
/// finishes.
const STEPS: u32 = 32;

/// The model-store lookup, the one-off fp32 expansion and the inference build.
#[path = "common/real_lm.rs"]
mod real_lm;

struct Run {
    ids: Vec<u32>,
    prefill_s: f64,
    decode_s: f64,
}

fn run_on(gpu: gpu_core::Gpu, backend: &str) -> Option<Run> {
    let Some((gguf, st)) = real_lm::paths() else {
        brain_testutil::skip(&format!("{}/{} not in the model store", real_lm::STORE, real_lm::GGUF));
        return None;
    };
    let cfg = deepseek2::import::config_from_file(gguf.to_str().expect("utf-8 path"), 1).unwrap_or_else(|e| panic!("config_from_file: {e}"));
    let src = checkpoint::weightio::WeightReader::open(&real_lm::expanded(&gguf, &st, &cfg)).unwrap_or_else(|e| panic!("open expansion: {e}"));

    let t0 = Instant::now();
    let m = DeepseekV2::new_sized(gpu, cfg.clone(), Sizes { b: 1, t: 1, ctx: CTX, chunk: CHUNK, batched: false }, &src, false);
    let build_s = t0.elapsed().as_secs_f64();
    drop(src);
    println!("  [{backend}] built (ctx {CTX}, chunk {CHUNK}) in {build_s:.1} s");
    brain_testutil::mem(&format!("{backend}: decoder built"));

    // Deterministic, in-vocab, and not a constant run - a repeated id would
    // let a broken causal mask look right.
    let prompt: Vec<u32> = (0..PROMPT_ROWS).map(|i| ((i as u32) * 7919 + 11) % cfg.vocab()).collect();

    let t1 = Instant::now();
    let logits = m.prefill_chunked(&prompt);
    let prefill_s = t1.elapsed().as_secs_f64();
    assert!(logits.iter().all(|x| x.is_finite()), "[{backend}] prefill produced non-finite logits");

    let mut ids = Vec::with_capacity(STEPS as usize);
    let mut next = argmax(&logits);
    let t2 = Instant::now();
    for _ in 0..STEPS {
        ids.push(next);
        next = argmax(&m.step(next));
    }
    let decode_s = t2.elapsed().as_secs_f64();

    println!(
        "  [{backend}] prefill {PROMPT_ROWS} rows in {prefill_s:.2} s ({:.1} rows/s); decode {STEPS} tokens in {decode_s:.2} s = {:.3} s/token ({:.2} tok/s)",
        PROMPT_ROWS as f64 / prefill_s,
        decode_s / STEPS as f64,
        STEPS as f64 / decode_s,
    );
    Some(Run { ids, prefill_s, decode_s })
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).expect("nonempty").0 as u32
}

#[ignore = "real 2.9 B-parameter checkpoint at the served ctx/chunk, on BOTH backends: ~12 GB of host RAM and ~14 GB of VRAM, several minutes (the CPU arm dominates). Slow lane only. `cargo test --release -p brain-deepseek2 --test decode_throughput -- --nocapture --ignored`."]
#[test]
fn the_served_decode_shape_agrees_across_backends_and_reports_its_cost() {
    let Some(gpu) = real_lm::wgpu() else { return };
    println!("== deepseekv2 served decode shape: ctx {CTX}, chunk {CHUNK}, prompt {PROMPT_ROWS} rows, {STEPS} greedy steps");

    let Some(w) = run_on(gpu, "wgpu") else { return };
    let Some(c) = run_on(real_lm::cpu(), "cpu") else { return };

    println!(
        "  speedup wgpu vs cpu: prefill {:.1}x, decode {:.1}x",
        c.prefill_s / w.prefill_s,
        c.decode_s / w.decode_s
    );

    // The one assertion. Everything above is a measurement; this is the gate:
    // moving the decoder between backends must not change what the model says.
    assert_eq!(w.ids, c.ids, "wgpu and cpu decoded different ids at the served shape");
    println!("  both backends decoded the same {} ids", w.ids.len());
}
