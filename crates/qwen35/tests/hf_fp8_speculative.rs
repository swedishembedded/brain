// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding of FP8 LLM releases with their own multi-token-prediction head
// for its clients. If your team needs expertise in serving FP8 checkpoints faster than one token per weight read, then
// you can procure our services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! Speculative decoding of the paged resident engine served from an HF FP8 directory, drafting with the directory's own
//! `mtp.*` tensors, on a fixture with the 27B's recurrent geometry (the verify kernels' shape).
//!
//! Spec:
//!   * a mixed forward whose decode rows are a verify step ([`PagedDecoder::forward_mixed_verify`]) scores the same
//!     rows as the separate verify pass and reads the prompts in as the separate prefill does;
//!   * speculation leaves greedy output as plain decoding gives it, with prompts arriving while sequences decode
//!     (verify rows beside prompt slices) and with every sequence drafting from its own head state.
//!
//! Run with `-- --device gpu1 --backend cuda`; without `--device` the binary reports the tests skipped.

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillJob, PrefillProgress, Request, Scheduler, SpecPolicy};
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident, SpecServing, SpeculativeMode};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

const CAP: u32 = 512;

fn engine() -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    // The verify kernels take the 27B's recurrent geometry (key and value heads of 128, a width-4 conv).
    let cfg = Qwen35Config { linear_num_key_heads: 1, linear_num_value_heads: 2, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, ..fp8_resident_cfg() };
    let dir = write_checkpoint_conditioned(&cfg, &format!("speculative{n}"), 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, CAP, tier)
        .with_max_batch(8)
        .with_prefix_slots(4)
        .with_kv_block_rows(16)
        .with_kv_tier(KvTier::F32)
        .with_speculation(SpecServing { mode: SpeculativeMode::Mtp, scratch_rows: 32, ..SpecServing::default() });
    let (mut engine, _, _) = resident.activate_engine(&[device]).expect("the tiny FP8 directory loads");
    std::fs::remove_dir_all(dir).ok();
    engine.set_mixed_step(true);
    assert!(engine.instance().has_mtp() && engine.instance().speculation_rows() > 0, "the directory's own head drafts");
    engine
}

fn prompt_of(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 3 + (seed * 5 + i * 7 + i * i) % 58).collect()
}

/// Staggered requests: the long prompts arrive while the short ones decode, so their slices ride in iterations whose
/// decode rows verify drafts. Returns every request's tokens and what speculation did.
fn tokens(speculate: bool) -> (Vec<Vec<u32>>, u64) {
    let mut sched = Scheduler::new(engine(), 8);
    sched.set_speculation(SpecPolicy { enabled: speculate, ..SpecPolicy::default() });
    let mut ids = Vec::new();
    for (seed, len, max_new) in [(1u32, 20u32, 40usize), (2, 33, 40)] {
        ids.push(sched.submit(Request { prompt: prompt_of(seed, len), max_new, stop: Vec::new() }));
    }
    for _ in 0..4 {
        sched.step();
    }
    for (seed, len, max_new) in [(3u32, 200u32, 20usize), (4, 150, 20), (5, 90, 20)] {
        ids.push(sched.submit(Request { prompt: prompt_of(seed, len), max_new, stop: Vec::new() }));
    }
    let mut done = sched.run();
    let verified = sched.spec_stats().verify_steps;
    (ids.iter().map(|id| done.remove(id).expect("the request completes")).collect(), verified)
}

/// Greedy output under speculation is plain decoding's, to the numerics of a verify row against a decode row: every
/// request starts with the same tokens (a wrong rollback or a wrongly read prompt diverges at once).
fn speculating_beside_arriving_prompts_gives_the_plain_tokens() {
    const AGREE: usize = 12;
    let (plain, none) = tokens(false);
    let (spec, verified) = tokens(true);
    assert_eq!(none, 0);
    assert!(verified > 0, "speculation ran");
    for (n, (s, p)) in spec.into_iter().zip(plain).enumerate() {
        assert_eq!(s.len(), p.len(), "request {n}");
        let agree = s.iter().zip(&p).take_while(|(a, b)| a == b).count();
        assert!(agree >= AGREE.min(s.len()), "request {n} diverges after {agree} tokens: {s:?} against {p:?}");
    }
}

/// One mixed verify forward against the two passes on twin engines, and the rollback after it: rows of three running
/// sequences (a next input and two proposals each) beside two fresh prompts.
fn one_mixed_verify_forward_matches_a_prefill_round_and_a_verify_step() {
    let (mut a, mut b) = (engine(), engine());
    let running: Vec<Vec<u32>> = (0..3).map(|i| prompt_of(10 + i, 20 + 7 * i)).collect();
    let fresh = [prompt_of(20, 100), prompt_of(21, 60)];
    let start = |e: &mut Qwen35GgufEngine| -> (Vec<BlockTable>, Vec<Vec<u32>>) {
        let mut tables = Vec::new();
        let mut rows = Vec::new();
        for p in &running {
            let mut t = BlockTable::new();
            let h = e.prefill(&mut t, p);
            let next = e.admit_greedy(&h);
            rows.push(vec![next, (next + 3) % 64, (next + 9) % 64]);
            tables.push(t);
        }
        (tables, rows)
    };
    let (mut ta, ra) = start(&mut a);
    let (mut tb, rb) = start(&mut b);
    assert_eq!(ra, rb);
    let (mut na, mut nb) = ([BlockTable::new(), BlockTable::new()], [BlockTable::new(), BlockTable::new()]);
    let mut refs_a: Vec<&mut BlockTable> = ta.iter_mut().collect();
    let [n0, n1] = &mut na;
    let mut jobs_a = [PrefillJob { table: n0, prompt: &fresh[0], budget: 100, embeds: None }, PrefillJob { table: n1, prompt: &fresh[1], budget: 60, embeds: None }];
    let mixed = a.forward_mixed_verify(&mut refs_a, &ra, 4, &mut jobs_a);
    let [m0, m1] = &mut nb;
    let mut jobs_b = [PrefillJob { table: m0, prompt: &fresh[0], budget: 100, embeds: None }, PrefillJob { table: m1, prompt: &fresh[1], budget: 60, embeds: None }];
    let prefilled = b.prefill_chunks(&mut jobs_b);
    let mut refs_b: Vec<&mut BlockTable> = tb.iter_mut().collect();
    let scored = b.forward_batched_verify(&mut refs_b, &rb, 4);
    for (seq, (x, y)) in mixed.scored.iter().zip(&scored).enumerate() {
        assert_eq!(x.len(), y.len());
        for (row, (cx, cy)) in x.iter().zip(y).enumerate() {
            assert_eq!(cx[0].0, cy[0].0, "sequence {seq} row {row}: the top token differs");
            for (px, py) in cx.iter().zip(cy) {
                assert!((px.1 - py.1).abs() <= 5e-2 * py.1.abs().max(1.0), "sequence {seq} row {row}: logit {} against {}", px.1, py.1);
            }
        }
    }
    for (job, (x, y)) in mixed.prefilled.iter().zip(&prefilled).enumerate() {
        match (x, y) {
            (PrefillProgress::Done(hx), PrefillProgress::Done(hy)) => {
                let (num, den) = hx.iter().zip(hy).fold((0f64, 0f64), |(n, d), (&p, &q)| (n + ((p - q) as f64).powi(2), d + (q as f64).powi(2)));
                assert!((num / den).sqrt() < 5e-2, "job {job}: hidden rel-L2 {}", (num / den).sqrt());
            }
            (PrefillProgress::Partial { consumed: cx }, PrefillProgress::Partial { consumed: cy }) => assert_eq!(cx, cy, "job {job}: slices differ"),
            _ => panic!("job {job}: one way finished the prompt and the other did not"),
        }
    }
    // Settle both the same way (the first sequence rejects its proposals, the second keeps one, the third all) and decode one
    // plain step from the rolled-back states: the recurrent state the rollback leaves is what the next token reads.
    let keep = [1usize, 2, 3];
    a.commit_verified(&mut refs_a, &keep);
    b.commit_verified(&mut refs_b, &keep);
    let inputs = [7u32, 8, 9];
    for (t, &k) in refs_a.iter_mut().zip(&keep) {
        let len = t.len();
        t.truncate(len - 3 + k as u32, a.alloc_mut());
    }
    for (t, &k) in refs_b.iter_mut().zip(&keep) {
        let len = t.len();
        t.truncate(len - 3 + k as u32, b.alloc_mut());
    }
    let (da, db) = (a.forward_batched_topk(&mut refs_a, &inputs, 4), b.forward_batched_topk(&mut refs_b, &inputs, 4));
    for (row, (x, y)) in da.iter().zip(&db).enumerate() {
        assert_eq!(x[0].0, y[0].0, "after the rollback, decode row {row}: the top token differs");
    }
}

gpu_core::card_tests!(speculating_beside_arriving_prompts_gives_the_plain_tokens, one_mixed_verify_forward_matches_a_prefill_round_and_a_verify_step);
