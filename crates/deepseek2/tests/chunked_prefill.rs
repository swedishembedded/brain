// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `DeepseekV2::prefill_chunked`/`decode_rows` (`batched = false`, multi-row
//! prefill rounds), gated two ways that catch two DIFFERENT classes of bug:
//!
//! 1. **Chunk-width dispatch consistency** (`run`, below): the SAME
//!    `decode_rows` a token-by-token `step` replay uses, called at `n > 1`
//!    instead of `n = 1` every time, must leave the persistent KV cache in a
//!    state that later `step` calls cannot tell apart from the per-token
//!    replay's. Since `step`'s reference run and `prefill_chunked`'s run
//!    under test go through the literal same function, this is blind to a
//!    bug baked identically into `decode_rows` regardless of `n` (e.g. always
//!    using position 0) - both sides would be equally, self-consistently
//!    wrong. What it DOES catch, and was measured catching: a KV-cache
//!    WRITE-OFFSET bug, because `kv_cache_fill_at`'s destination depends on
//!    how many rows a call writes AND at what round each call happens - a
//!    "write at 0 always" bug corrupts a `chunk=4` prefill's cache
//!    differently than a `chunk=1` per-token replay's (four rounds each
//!    clobbering rows `0..4`, vs every single token clobbering row `0`
//!    alone), which is exactly the divergence this file measured (see the
//!    mutation table below).
//! 2. **Independent-implementation cross-check** (`a_splice_run_that_
//!    straddles_a_chunk_boundary_is_recut`, below): `prefill_chunked`'s
//!    `[vocab]` output compared against `DeepseekV2::logits_all` - the
//!    batched, `ROPE`/`gqa_fwd`-based tape, which shares NO dispatch with
//!    `decode_rows`'s `ROPE_AT`/`gqa_chunk_step`. This is what actually
//!    proves `decode_rows`'s position/causal-mask math is right, independent
//!    of chunk width - and it is the one that caught the two mutations (1)
//!    could not.
//!
//! The prompt is 9 tokens; `run` replays it at chunk widths 4 (`4+4+1`,
//! ragged) and 16 (one whole-prompt round, proving the fresh-sequence/empty-
//! cache path independently of round-to-round continuation).
//!
//! `DeepseekV2Config::tiny()` already has TWO layers, both with a REAL
//! attention sublayer (plain MHA runs on every layer regardless of the
//! dense/MoE FFN split - unlike qwen35's mixed GDN/GQA stack, there is no
//! "only every Nth layer attends" here), so a non-final row's wrong
//! computation in layer 0 flows into layer 1's cache entry for that same
//! position, which a later row attends to - the multi-layer case the real
//! 12-layer checkpoint actually runs.
//!
//! ## Measured, not guessed: what each test actually catches
//!
//! Three deliberately broken variants were built and measured against BOTH
//! gates, each reverted immediately after measuring. Baseline (correct
//! implementation): `run` measures EXACTLY 0 on both the CPU Cranelift JIT
//! and a real Vulkan GPU (Tesla P40, `BRAIN_DEVICE=vulkan`); the independent
//! cross-check measures 7.45e-9 (the honest fp32 noise floor between two
//! genuinely different dispatch sequences over the same math).
//!
//! | mutation | `run` (chunk=4) | independent cross-check |
//! |---|---|---|
//! | `kv_cache_fill_at`'s `start` forced to `0` | **1.8e-3** | 5.6e-3 |
//! | `ROPE_AT`'s `pos_base` forced to `0` (chunk-relative, not absolute, position) | 0 (blind - see below) | **5.4e-6** |
//! | `seq_lens[i] = pos_start+i+1` flattened to `i+1` (chunk-relative causal mask) | 0 (blind - see below) | **5.3e-3** |
//!
//! The two "blind" results are not a bound problem, they are the exact
//! shared-code limitation described above: forcing position/mask to be
//! chunk-relative is wrong by the SAME amount whether `decode_rows` is called
//! once at `n=9` or nine times at `n=1`, so `run`'s two replays stay in
//! (wrong) agreement. `BOUND_CHUNK` (`1e-4`) and `BOUND_INDEPENDENT` (`1e-6`)
//! are each set with the largest margin the WEAKEST real signal for that gate
//! allows - `run`'s only real signal is the KV-offset bug (1.8e-3, ~18x
//! `BOUND_CHUNK`), the independent cross-check's weakest is the RoPE bug
//! (5.4e-6, ~5.4x `BOUND_INDEPENDENT` and ~725x its own 7.45e-9 noise floor).

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::{DeepseekV2, Sizes, PIPELINES};
use gpu_core::Gpu;

const BOUND_CHUNK: f32 = 1e-4;
const BOUND_INDEPENDENT: f32 = 1e-6;

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// A fresh `batched = false` instance: `ctx` wide enough for the whole run,
/// `chunk` the round width under test.
fn build(gpu: Gpu, chunk: u32, ctx: u32) -> DeepseekV2 {
    let cfg = DeepseekV2Config::tiny();
    let init = deepseek2::init_weights(&cfg, 7);
    DeepseekV2::new_sized(gpu, cfg, Sizes { b: 1, t: 1, ctx, chunk, batched: false }, &init, false)
}

/// Replay `prompt` through both paths at round width `chunk`, then continue
/// both with the same `tail` tokens one at a time and compare. One instance,
/// reset between the two phases - `reset_cache` only rewinds the decode
/// position, but both phases write the exact same absolute positions before
/// ever reading them back, so nothing stale survives from phase 1 into phase
/// 2 (the same pattern `generate_greedy_kv_matches_recompute` already relies
/// on for this crate's own re-run-on-one-instance tests).
fn run(gpu: Gpu, chunk: u32) {
    let vocab = DeepseekV2Config::tiny().vocab();
    let ctx = 13; // DeepseekV2Config::tiny()'s own max_position_embeddings
    let prompt: Vec<u32> = (0..9).map(|i| (i * 3 + 1) % vocab).collect();
    let tail: Vec<u32> = (0..3).map(|i| (i * 5 + 2) % vocab).collect();

    let m = build(gpu, chunk, ctx);

    // Reference: the existing one-dispatch-per-token replay.
    let mut want_last = Vec::new();
    for &tok in &prompt {
        want_last = m.step(tok);
    }
    let want_tail: Vec<Vec<f32>> = tail.iter().map(|&tok| m.step(tok)).collect();
    assert_eq!(m.cache_pos(), (prompt.len() + tail.len()) as u32);

    // Under test: same instance, cache reset, the same prompt through
    // chunked prefill, then the SAME per-token continuation.
    m.reset_cache();
    let got_last = m.prefill_chunked(&prompt);
    assert_eq!(got_last.len(), vocab as usize, "prefill_chunked must return one [vocab] logits row");
    assert!(got_last.iter().all(|x| x.is_finite()), "prefill_chunked produced non-finite logits");
    assert_eq!(m.cache_pos(), prompt.len() as u32, "prefill_chunked must leave the decode position at the prompt length");
    let got_tail: Vec<Vec<f32>> = tail.iter().map(|&tok| m.step(tok)).collect();
    assert_eq!(m.cache_pos(), (prompt.len() + tail.len()) as u32);

    let last_err = maxabs(&got_last, &want_last);
    assert!(last_err < BOUND_CHUNK, "chunk={chunk}: prompt's last-token logits maxabs={last_err}");
    let mut worst = last_err;
    for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
        let err = maxabs(got, want);
        worst = worst.max(err);
        assert!(err < BOUND_CHUNK, "chunk={chunk}: continuation token {i} logits maxabs={err} (chunked prefill left the cache wrong)");
    }
    println!("chunked_prefill(chunk={chunk}): worst maxabs over prompt-last + {} continuation steps = {worst:e}", tail.len());
}

#[test]
fn chunked_prefill_matches_token_by_token_replay_cpu() {
    run(Gpu::new_cpu(PIPELINES), 4);
}

#[test]
fn chunked_prefill_matches_token_by_token_replay_default_backend() {
    run(Gpu::new(PIPELINES), 4);
}

/// The single-round case: `chunk >= prompt.len()`, so the whole prompt is one
/// dispatch round starting from a fresh (empty) KV cache. Separated from the
/// multi-round test above because a bug in the round-to-round cache
/// continuation passes this one and fails that one, and vice versa for a bug
/// in the fresh-sequence seeding.
#[test]
fn whole_prompt_single_chunk_matches_token_by_token_replay() {
    run(Gpu::new_cpu(PIPELINES), 16);
}

/// The independent cross-check (see module doc): a splice run that does not
/// align with a chunk boundary, `row0 = 3, n_rows = 3` inside a `chunk = 4`
/// round width, so a naive `0, 4, 8, ...` chunk cut would end round 0 at
/// position 4 - one row INSIDE the run `[3, 6)`. `chunk_plan`'s re-cutting
/// (round 0 ends at `row0`, the run becomes its own round) is what a
/// straddling boundary would otherwise break silently: `decode_rows` asserts
/// a round never partially overlaps a spliced run, so a `chunk_plan`
/// regression here panics rather than producing a quietly wrong image
/// splice. Comparing against `logits_all` (the batched, un-chunked tape) is
/// also what makes this THE gate for `decode_rows`'s position/mask math
/// itself - see the module doc's measured mutation table.
#[test]
fn a_splice_run_that_straddles_a_chunk_boundary_is_recut() {
    let cfg = DeepseekV2Config::tiny();
    let d = cfg.d_model();
    let init = deepseek2::init_weights(&cfg, 7);
    let ctx = 13;
    let chunk = 4;
    let prompt: Vec<u32> = (0..9).map(|i| (i * 3 + 1) % cfg.vocab()).collect();
    let (row0, n_rows) = (3u32, 3u32);
    let img: Vec<f32> = (0..(n_rows * d)).map(|i| (i as f32) * 0.01 - 0.3).collect();

    // Reference: the batched (flat, un-chunked) tape over the whole prompt,
    // which shares no dispatch with `decode_rows`.
    let mut mb = DeepseekV2::new_on(Gpu::new_cpu(PIPELINES), cfg.clone(), 1, prompt.len() as u32, &init, false);
    mb.enable_mm_splice(row0, n_rows);
    mb.write_img_embeds(&img);
    let want = mb.logits_all(&prompt);
    let want_last = &want[want.len() - cfg.vocab() as usize..];

    // Under test: batched=false, chunked prefill, the splice straddling a
    // chunk boundary.
    let mut mc = DeepseekV2::new_sized(Gpu::new_cpu(PIPELINES), cfg.clone(), Sizes { b: 1, t: 1, ctx, chunk, batched: false }, &init, false);
    mc.enable_mm_splice(row0, n_rows);
    mc.write_img_embeds(&img);
    let got_last = mc.prefill_chunked(&prompt);

    let err = maxabs(got_last.as_slice(), want_last);
    assert!(err < BOUND_INDEPENDENT, "straddling splice: chunked vs batched logits maxabs={err}");
}
