// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35::serve::Engine`/`Scheduler` must reproduce `Qwen35::step`'s
//! single-sequence decode EXACTLY - going through the paged
//! `Scheduler`/`Engine`/`BlockTable` machinery must not change the actual
//! numbers versus the already-proven-correct decode path
//! (`decode_step.rs` gates `Qwen35::step` itself against whole-sequence
//! `logits_all`; this test's job is only to gate the NEW paged wiring this
//! module adds on top of it). Mirrors `qwen35moe/tests/serve.rs` exactly.
//!
//! Admits one request, drives it to completion via
//! `model::serve::Scheduler<qwen35::serve::Engine>`, and compares its
//! generated tokens token-for-token against `qwen35::sample::generate_kv`
//! (greedy, `temperature=0.0`) run directly over a second `Qwen35` instance
//! built from the SAME weights. Runs on both the CPU JIT and the default GPU
//! backend.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use model::serve::{PagedDecoder, Request};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use qwen35::serve::{Engine, Scheduler};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
    qwen35::init::init_weights(cfg, seed)
}

fn run(make_gpu: fn(&[(&str, &str)]) -> Gpu) {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 11);
    let prompt = vec![1u32, 5, 3, 9, 2, 7];
    let max_new = 6usize;
    let max_seq_len = (prompt.len() + max_new) as u32;

    // Reference: `Qwen35::step`'s own single-sequence decode, driven by
    // `crate::sample::generate_kv` (greedy) - already proven correct against
    // `logits_all` by `decode_step.rs`.
    let reference = Qwen35::new_on(make_gpu(pipelines()), cfg.clone(), 1, max_seq_len, &init);
    let mut rng = Rng::new(1);
    let want = qwen35::sample::generate_kv(&reference, &prompt, max_new, 0.0, 0, 1.0, &[], &mut rng);
    assert_eq!(want.len(), max_new, "greedy decode with no eos must always produce exactly max_new tokens");

    // Under test: the SAME prompt, on the SAME weights, through the paged
    // Scheduler/Engine/BlockTable machinery this file adds.
    // `max_concurrent=2`: exercises a pool wider than the one sequence this
    // test actually admits, so a bug that only shows up when
    // `blocks()[0] != 0` (i.e. hard-coded to the first physical block) would
    // have a chance to surface if a later change reordered allocation.
    let engine = Engine::from_map_on(&make_gpu(pipelines()), cfg, &init, max_seq_len, 2);
    println!("kv_pool_bytes={} kv_pool_capacity_tokens={}", engine.kv_pool_bytes(), engine.kv_pool_capacity_tokens());
    let mut sched = Scheduler::new(engine, 1);
    let id = sched.submit(Request { prompt: prompt.clone(), max_new, eos: None });
    let out = sched.run();
    let got = out.get(&id).expect("the admitted request must complete");

    assert_eq!(got, &want, "paged Scheduler/Engine decode must exactly match Qwen35::step's single-sequence decode");
    println!("serve engine matches Qwen35::step over {max_new} greedy tokens: {got:?}");
}

#[test]
fn scheduler_decode_matches_step_cpu() {
    run(Gpu::new_cpu);
}

/// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
/// backend otherwise - run this under both `BRAIN_DEVICE=cpu` and unset
/// (the default GPU backend). `scheduler_decode_matches_step_cpu` above pins
/// the CPU JIT explicitly regardless of `BRAIN_DEVICE` so the CPU path is
/// always exercised even when this one runs against the GPU.
#[test]
fn scheduler_decode_matches_step_default_backend() {
    run(Gpu::new);
}

/// M3.4: `Engine::prefill` used to call `self.model.gpu.read(&h, d)` on
/// EVERY prompt token, discarding all but the last iteration's result - a
/// host-synchronising readback per PROMPT TOKEN to recover exactly one
/// `[d_model]` row. It now chains `run_decode_step`'s device buffer across
/// the whole loop and reads back exactly once, after it ends - so a
/// `prefill` call must cost exactly ONE readback, regardless of how many
/// tokens the prompt has. Mirrors `qwen3::serve::Engine`'s own
/// `prefill_submits_scale_with_chunks_not_with_token_count` gate (M3.1),
/// adapted to per-token (not per-chunk) granularity since this engine has no
/// multi-token batched dispatch to chunk over.
#[test]
fn prefill_reads_back_exactly_once_regardless_of_prompt_length() {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 7);
    let max_seq_len = 20u32;

    let probe = Engine::from_map(cfg.clone(), &init, max_seq_len, 1);
    if probe.device_stats().is_none() {
        brain_testutil::skip_unavailable("this backend does not count device readbacks");
        return;
    }
    drop(probe);

    let reads_for = |prompt: &[u32]| -> u64 {
        let mut engine = Engine::from_map(cfg.clone(), &init, max_seq_len, 1);
        let mut t = model::paged::BlockTable::new();
        let before = engine.device_stats().expect("probed available above").readbacks;
        engine.prefill(&mut t, prompt);
        let after = engine.device_stats().expect("probed available above").readbacks;
        after - before
    };

    let short = vec![1u32, 5, 3];
    let long: Vec<u32> = (0..12).map(|i| (i % (cfg.vocab - 1)) + 1).collect();
    let reads_short = reads_for(&short);
    let reads_long = reads_for(&long);
    assert_eq!(reads_short, 1, "a {}-token prefill must read back exactly once, got {reads_short}", short.len());
    assert_eq!(reads_long, 1, "a {}-token prefill must read back exactly once, got {reads_long}", long.len());
}

/// M3.4: `Engine::forward_batched_topk` now extracts its top-`k` candidates
/// entirely on the device (`Qwen35::head_topk_dev`: iterative `argmax_part`/
/// `argmax_final` + `topk_extract_step`) instead of sorting a host-computed
/// `[vocab]` logits vector. Ties/reduction order between a tiled device
/// matmul and a scalar host dot product are real (this campaign's own gate
/// wording - see `qwen3::serve::Engine`'s
/// `admission_head_matches_a_true_host_matvec_within_tolerance`), so this
/// compares VALUES within a tolerance and ids only where values are not
/// near-tied, rather than asserting exact equality.
#[test]
fn forward_batched_topk_matches_an_independent_host_matvec_within_tolerance() {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 13);
    let max_seq_len = 16u32;
    let prompt = vec![2u32, 4, 6];
    let k = 5usize;

    let mut engine = Engine::from_map(cfg.clone(), &init, max_seq_len, 1);
    let mut t = model::paged::BlockTable::new();
    let hidden = engine.prefill(&mut t, &prompt);

    // Device path under test: one more decode step, top-k extracted on the
    // device from that step's own hidden state.
    let next_tok = 1u32;
    let got = engine.forward_batched_topk(&mut [&mut t], std::slice::from_ref(&next_tok), k).pop().expect("one row");

    // Independent reference: replay the SAME prompt + next token through a
    // second `Qwen35` instance driven by `Qwen35::step` (not through
    // `Engine`/`DecodeCaches` at all), then a plain host `matvec_par` + sort -
    // no device kernel this test is trying to verify is reused here.
    let reference = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, max_seq_len, &init);
    let mut ref_hidden = Vec::new();
    for &tok in &prompt {
        ref_hidden = reference.step(tok);
    }
    // Same hidden state, to fp32 tolerance rather than bit-for-bit: the two
    // sides get here by DIFFERENT dispatch shapes (a chunked `n=3` prefill
    // round against `Engine`'s pooled KV, versus three one-token
    // `Qwen35::step` calls against this instance's own cache), and a
    // reduction's summation order is a property of the dispatch shape. The
    // chunked-vs-per-token equivalence itself is `tests/chunked_prefill.rs`'s
    // gate, at the same 1e-5; this assert only needs the two runs to be on the
    // same sequence before the top-k comparison below means anything.
    let seed_err = ref_hidden.iter().zip(&hidden).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(seed_err < 1e-5, "the reference replay must land on the SAME prefill hidden state `Engine::prefill` returned (maxabs={seed_err})");
    let ref_hidden_next = reference.step(next_tok);
    let head = reference.read_weight(cfg.head_weight());
    let ref_logits = model::hostmath::matvec_par(&head, &ref_hidden_next, cfg.vocab as usize, cfg.d_model as usize);
    let mut ref_ranked: Vec<(u32, f32)> = ref_logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    ref_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ref_ranked.truncate(k);

    assert_eq!(got.len(), k, "forward_batched_topk must return exactly k candidates, got {}", got.len());
    for (i, (&(got_id, got_v), &(ref_id, ref_v))) in got.iter().zip(ref_ranked.iter()).enumerate() {
        assert!(
            (got_v - ref_v).abs() < 1e-3,
            "candidate {i}: device value {got_v} vs host reference value {ref_v} (ids {got_id} vs {ref_id})"
        );
        assert_eq!(got_id, ref_id, "candidate {i}: device id {got_id} vs host reference id {ref_id} at value {got_v}");
    }
}

/// **The batched-decode spec.** Several concurrent sequences, at DIFFERENT
/// prompt lengths, decoded through one `Engine::forward_batched_greedy` call
/// per step - one set of GPU dispatches carrying every sequence's row - must
/// produce exactly the tokens each sequence produces when it is the only
/// request the engine has.
///
/// This is the property the whole batched hybrid decode path exists to
/// preserve, and the one a single-sequence test structurally cannot see.
/// Batching this architecture means two different kinds of per-sequence state
/// share a dispatch: the full-attention layers' KV history (now one pooled
/// buffer per layer, addressed by each sequence's own physical block) and the
/// Gated-DeltaNet layers' recurrent state and conv window (staged into and out
/// of a contiguous batch slab). A row reading or writing another row's slice
/// of either shows up here and nowhere else.
///
/// Different prompt lengths are the point, not incidental: equal lengths make
/// a stale-`seq_lens` or off-by-one causal bound invisible, because every row
/// would then want the same bound anyway. The greedy token ids are compared
/// EXACTLY - they are discrete, so this is a strictly sharper statement than
/// any tolerance on the hidden states would be.
#[test]
fn batched_decode_matches_each_sequence_decoded_alone() {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 23);
    let prompts: Vec<Vec<u32>> = vec![vec![1u32, 5, 3], vec![2u32, 8], vec![4u32, 6, 1, 7, 3], vec![9u32, 2, 5, 8]];
    let steps = 5usize;
    let max_seq_len = 16u32;

    // Reference: each sequence ALONE in its own engine, one at a time.
    let mut want: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        let mut engine = Engine::from_map(cfg.clone(), &init, max_seq_len, 1);
        let mut t = model::paged::BlockTable::new();
        let hidden = engine.prefill(&mut t, p);
        let mut tok = engine.admit_greedy(&hidden);
        let mut out = vec![tok];
        for _ in 1..steps {
            tok = engine.forward_batched_greedy(&mut [&mut t], &[tok]).pop().expect("one row");
            out.push(tok);
        }
        want.push(out);
    }

    // Under test: all of them resident at once, decoded together.
    let mut engine = Engine::from_map(cfg.clone(), &init, max_seq_len, prompts.len() as u32);
    let mut tables: Vec<model::paged::BlockTable> = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for p in &prompts {
        let mut t = model::paged::BlockTable::new();
        let hidden = engine.prefill(&mut t, p);
        cur.push(engine.admit_greedy(&hidden));
        tables.push(t);
    }
    for (o, &tok) in got.iter_mut().zip(&cur) {
        o.push(tok);
    }
    for _ in 1..steps {
        let mut refs: Vec<&mut model::paged::BlockTable> = tables.iter_mut().collect();
        cur = engine.forward_batched_greedy(&mut refs, &cur);
        for (o, &tok) in got.iter_mut().zip(&cur) {
            o.push(tok);
        }
    }

    for (b, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "sequence {b} (prompt len {}) decoded differently in a batch of {} than alone", prompts[b].len(), prompts.len());
    }
    println!("batched decode of {} sequences (prompt lens {:?}) matches each decoded alone over {steps} steps", prompts.len(), prompts.iter().map(|p| p.len()).collect::<Vec<_>>());

    // The same batch through the top-k path, which shares the batched decode
    // step but a different device head reduction.
    let mut refs: Vec<&mut model::paged::BlockTable> = tables.iter_mut().collect();
    let topk = engine.forward_batched_topk(&mut refs, &cur, 3);
    assert_eq!(topk.len(), prompts.len(), "forward_batched_topk must return one candidate list per sequence");
    for (b, row) in topk.iter().enumerate() {
        assert_eq!(row.len(), 3, "sequence {b}: forward_batched_topk must return exactly k candidates");
        assert!(row.windows(2).all(|w| w[0].1 >= w[1].1), "sequence {b}: top-k candidates must come back best-first, got {row:?}");
    }
}
