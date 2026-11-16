// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35moe::serve::Engine`/`Scheduler` must reproduce `Qwen35::step`'s
//! single-sequence decode EXACTLY -- going through the paged
//! `Scheduler`/`Engine`/`BlockTable` machinery must not change the actual
//! numbers versus the already-proven-correct P11b decode path
//! (`decode_step.rs` gates `Qwen35::step` itself against whole-sequence
//! `logits_all`; this test's job is only to gate the NEW paged wiring this
//! module adds on top of it).
//!
//! Admits one request, drives it to completion via
//! `model::serve::Scheduler<qwen35moe::serve::Engine>`, and compares its
//! generated tokens token-for-token against `qwen35moe::sample::generate_kv`
//! (greedy, `temperature=0.0`) run directly over a second `Qwen35` instance
//! built from the SAME weights -- the same "one model instance for the
//! reference, one independent path for the thing under test" structure
//! `decode_step.rs` itself uses. Runs on both the CPU JIT and the default
//! GPU backend, since a barrier-crossing kernel can silently misbehave on
//! exactly one backend.

use std::collections::HashMap;

use data::rng::{random_seed, Rng};
use gpu_core::Gpu;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, Request};
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{Qwen35, pipelines};
use qwen35moe::serve::{Engine, EngineOptions, Scheduler};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> HashMap<String, Vec<f32>> {
    qwen35moe::init::init_weights(cfg, seed)
}

fn run(make_gpu: fn(&[(&str, &str)]) -> Gpu) {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 11);
    let prompt = vec![1u32, 5, 3, 9, 2, 7];
    let max_new = 6usize;
    let max_seq_len = (prompt.len() + max_new) as u32;

    // Reference: `Qwen35::step`'s own single-sequence decode, driven by
    // `crate::sample::generate_kv` (greedy) -- P11b, already proven correct
    // against `logits_all` by `decode_step.rs`. `t_ref` just needs to be
    // positive (`gdn_chunk_size` always returns a divisor of its own input),
    // so the prompt+max_new length is as good a choice as any.
    let reference = Qwen35::new_on(make_gpu(pipelines()), cfg.clone(), 1, max_seq_len, &init);
    let mut rng = Rng::new(random_seed());
    let want = qwen35moe::sample::generate_kv(&reference, &prompt, max_new, 0.0, 0, 1.0, &[], &mut rng);
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
    let id = sched.submit(Request { prompt: prompt.clone(), max_new, stop: Vec::new() });
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
/// backend otherwise -- run this under both `BRAIN_DEVICE=cpu` and unset
/// (the default GPU backend).
/// `scheduler_decode_matches_step_cpu` above pins the CPU JIT explicitly
/// regardless of `BRAIN_DEVICE` so the CPU path is always exercised even when
/// this one runs against the GPU.
#[test]
fn scheduler_decode_matches_step_default_backend() {
    run(Gpu::new);
}

/// **The batched-decode spec.** Several concurrent sequences, at DIFFERENT
/// prompt lengths, decoded through one `Engine::forward_batched_greedy` call
/// per step - one set of GPU dispatches carrying every sequence's row - must
/// produce exactly the tokens each sequence produces when it is the only
/// request the engine has.
///
/// The `qwen35moe` twin of `qwen35/tests/serve.rs`'s gate of the same name, on
/// the same shared primitives (`model::block::gqa_decode_batched_step` for the
/// pooled full-attention KV, `model::gdn_mixer::gdn_mixer_decode_fwd` for the
/// recurrent state and conv window) - which is the point: a caller batching
/// this model gets the same contract as one batching the dense model, so both
/// have to be gated the same way. What this file adds over its twin is the
/// sparse MoE sublayer under the batch: the router picks experts PER ROW, so a
/// batched step routes rows wanting different experts through one dispatch.
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
    println!(
        "batched decode of {} sequences (prompt lens {:?}) matches each decoded alone over {steps} steps",
        prompts.len(),
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );
}

/// Rows of a KV block in the paged-pool specs below: small enough that the tiny
/// model's short prompts span several blocks.
const SPEC_BLOCK_ROWS: u32 = 4;

/// **The paged-pool spec.** Sequences draw fixed-size KV blocks from ONE pool
/// sized in rows, independent of `max_concurrent * max_seq_len`: two sequences
/// of different lengths take exactly the blocks their tokens cover, take more as
/// decode crosses block boundaries, and between them fill a pool far smaller
/// than two whole contexts. Each sequence's greedy tokens equal what it gives
/// served alone in an engine that reserves it one whole-context block, so the
/// block indirection changes no number.
#[test]
fn sequences_of_different_lengths_draw_blocks_from_one_pool() {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 31);
    let prompts: Vec<Vec<u32>> = vec![vec![3u32, 1, 4, 1, 5], vec![2u32, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9]];
    let steps = 8u32;
    let max_seq_len = 32u32;
    // Prompt 5 + 8 appended = 13 rows (4 blocks), prompt 13 + 8 = 21 rows (6 blocks): the pool's ten blocks exactly.
    let pool_rows = 10 * SPEC_BLOCK_ROWS;

    let alone = |p: &[u32]| -> Vec<u32> {
        let opts = EngineOptions::new(max_seq_len, 1).with_kv_block_rows(max_seq_len);
        let mut engine = Engine::from_source(cfg.clone(), &init, opts);
        let mut t = BlockTable::new();
        let hidden = engine.prefill(&mut t, p);
        let mut tok = engine.admit_greedy(&hidden);
        let mut out = vec![tok];
        for _ in 0..steps {
            tok = engine.forward_batched_greedy(&mut [&mut t], &[tok]).pop().expect("one row");
            out.push(tok);
        }
        engine.release_table(&mut t);
        out
    };
    let want: Vec<Vec<u32>> = prompts.iter().map(|p| alone(p)).collect();

    let opts = EngineOptions::new(max_seq_len, 2).with_kv_block_rows(SPEC_BLOCK_ROWS).with_kv_pool_rows(pool_rows);
    let mut engine = Engine::from_source(cfg.clone(), &init, opts);
    assert_eq!(engine.kv_pool_capacity_tokens(), pool_rows as u64, "the pool holds the rows it was sized for, not max_concurrent * max_seq_len");
    assert!(pool_rows < 2 * max_seq_len, "the spec's pool is smaller than two whole contexts");
    assert_eq!(engine.free_blocks(), 10);

    let mut tables: Vec<BlockTable> = Vec::new();
    let mut cur = Vec::new();
    for p in &prompts {
        let mut t = BlockTable::new();
        let hidden = engine.prefill(&mut t, p);
        assert_eq!(t.blocks().len() as u32, (p.len() as u32).div_ceil(SPEC_BLOCK_ROWS), "a prompt takes the blocks its tokens cover");
        cur.push(engine.admit_greedy(&hidden));
        tables.push(t);
    }
    assert_eq!(engine.free_blocks(), 10 - 2 - 4, "both prompts drew from one pool");
    let mut got: Vec<Vec<u32>> = cur.iter().map(|&t| vec![t]).collect();
    for _ in 0..steps {
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        cur = engine.forward_batched_greedy(&mut refs, &cur);
        for (o, &t) in got.iter_mut().zip(&cur) {
            o.push(t);
        }
    }
    assert_eq!(tables.iter().map(|t| t.blocks().len()).collect::<Vec<_>>(), vec![4, 6], "decode takes a block as a sequence crosses into it");
    assert_eq!(engine.free_blocks(), 0);
    let mut seen = std::collections::HashSet::new();
    assert!(tables.iter().flat_map(|t| t.blocks()).all(|b| seen.insert(*b)), "no block backs two sequences");
    for (b, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "sequence {b} (prompt {} tokens) decoded from the shared pool differently than alone", prompts[b].len());
    }
    for t in tables.iter_mut() {
        engine.release_table(t);
    }
    assert_eq!(engine.free_blocks(), 10, "a released sequence gives every block back");
}

/// Admission is on blocks free, not on whole contexts: a pool that holds ONE
/// whole context runs four short requests at once, and each completes with the
/// tokens it gives alone.
#[test]
fn short_prompts_run_together_in_a_pool_of_one_context() {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 37);
    let max_seq_len = 32u32;
    let prompts: Vec<Vec<u32>> = vec![vec![1u32, 2, 3], vec![4u32, 5, 6, 7], vec![8u32, 9], vec![3u32, 3, 5, 1, 2]];
    let max_new = 4usize;
    let solo = |p: &[u32]| {
        let engine = Engine::from_source(cfg.clone(), &init, EngineOptions::new(max_seq_len, 1));
        let mut sched = Scheduler::new(engine, 1);
        let id = sched.submit(Request { prompt: p.to_vec(), max_new, stop: Vec::new() });
        sched.run().remove(&id).expect("completes")
    };
    let want: Vec<Vec<u32>> = prompts.iter().map(|p| solo(p)).collect();

    let opts = EngineOptions::new(max_seq_len, prompts.len() as u32).with_kv_block_rows(SPEC_BLOCK_ROWS).with_kv_pool_rows(max_seq_len);
    let engine = Engine::from_source(cfg.clone(), &init, opts);
    let mut sched = Scheduler::new(engine, prompts.len());
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new, stop: Vec::new() })).collect();
    let (mut out, mut widest) = (HashMap::new(), 0);
    while sched.pending() {
        out.extend(sched.step());
        widest = widest.max(sched.running_len());
    }
    assert_eq!(widest, prompts.len(), "every short request ran at once in a pool of one whole context");
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(out.get(id), Some(&want[i]), "request {i} decoded in the shared pool differently than alone");
    }
}

/// Admission picks its first token where the head is: the greedy pick and the top-`k` candidates are the best of the
/// prefill's logit row, as the host would find them by scanning or sorting the row.
#[test]
fn admission_picks_on_the_device_what_the_logit_row_ranks_first() {
    let cfg = Qwen35Config::tiny();
    let init = init_weights(&cfg, 23);
    let mut engine = Engine::from_map(cfg.clone(), &init, 16, 1);
    let mut table = BlockTable::new();
    let hidden = engine.prefill(&mut table, &[4, 1, 8, 2, 6]);
    let row = engine.logits(&hidden);
    let mut ranked: Vec<(u32, f32)> = row.iter().copied().enumerate().map(|(i, v)| (i as u32, v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    assert_eq!(engine.admit_greedy(&hidden), ranked[0].0, "the greedy pick is not the row's best token");
    let k = 5;
    let got = engine.admit_topk(&hidden, k);
    let bits = |v: &[(u32, f32)]| v.iter().map(|&(t, l)| (t, l.to_bits())).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(&ranked[..k]), "the candidates are not the row's best {k}");
    engine.release_table(&mut table);
}
