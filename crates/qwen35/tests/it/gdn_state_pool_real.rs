// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-precision serving of hybrid recurrent language models for clients whose GPU
// memory bandwidth decides their cost per token. If your team needs expertise in storing recurrent state in fewer
// bytes without changing what the model says, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The quality gate for storing the Gated DeltaNet recurrent state of the served pool as bf16 (`--gdn-state bf16`)
//! instead of f32, on the REAL Qwen3.8-27B FP8 checkpoint.
//!
//! The pool is the path that matters: a served sequence's state lives in the per-slot pool between decode steps, so a
//! bf16 pool rounds every sequence's state to bf16 after every token and carries that rounding forward through the whole
//! generation. (The working copy a prefill or the single-sequence parity tool reads is always f32, so
//! `qwen35_fp8_parity` cannot see this.) The gate therefore runs several sequences through the pooled batched decode:
//!
//! 1. the f32-pool engine serves [`SEQS`] long prompts to [`NEW_TOKENS`] greedy tokens each (the Scheduler's served
//!    path); that stream is the reference;
//! 2. the f32 pool and the bf16 pool each score that SAME stream teacher-forced through the pooled batched decode
//!    (`teacher_forced_logits_batch`: prefill into the pool, then one batched step per position), so their logits
//!    compare position for position even where free-running generation would have diverged;
//! 3. the bf16 pool is also served free-running, and the length of the identical prefix of each sequence is reported.
//!
//! Reported: top-1 agreement of bf16 against f32 over all positions, the mean KL(f32 || bf16) in nats, and the worst
//! sequence. Self-skips when the checkpoint is not in the model store (`brain pull Qwen/Qwen3.8-27B-FP8`).

use model::kv_tier::KvTier;
use model::policy::GdnState;
use model::serve::{Request, Scheduler};
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};

const STORE: &str = "Qwen/Qwen3.8-27B-FP8";
/// Sequences decoded together through the pool.
const SEQS: usize = 4;
/// Greedy tokens each sequence generates: far enough that the state's rounding has compounded over its whole length.
const NEW_TOKENS: usize = 1024;
const CAP: u32 = 3072;
/// Bytes kept free per card, `brain serve`'s default reserve.
const RESERVE: u64 = 2 << 30;

/// Least top-1 agreement of the bf16 pool with the f32 pool over the teacher-forced stream.
///
/// Measured on the real checkpoint, 4 sequences x 1024 positions through the pooled batched decode (2026-10-31):
/// top-1 agreement of the bf16 pool with the f32 pool sits well above the run-to-run spread of `qwen35_fp8_parity`, which the other
/// precision changes of this engine are held to; a bf16 pool must not be looser than that spread, so the bar sits half a point above it.
// perf-number: the bar itself, a test threshold, not a measurement
/// 98.5 % (one position in 4096 moves the figure by 0.02 points, its binomial standard error is 0.1).
const MIN_TOP1: f64 = 0.985;
/// Most mean KL(f32 pool || bf16 pool) in nats.
///
/// Measured over the same positions:
/// perf-number: measured accuracy evidence, mean KL of the bf16 pool against the f32 pool
/// 1.075e-4 nats (the sequences 0.98 - 1.19e-4). The other precision changes of this engine spread over 1.5 - 3e-3 nats, so
/// 1e-3 sits under the least of them and about ten times over the measurement.
const MAX_MEAN_KL: f64 = 1e-3;

fn engine(path: &str, gdn: GdnState) -> Option<Qwen35GgufEngine> {
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE);
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM");
        return None;
    }
    let tier = held_tiers_at(path).expect("the directory is a servable checkpoint");
    let device = devices[0].0;
    let resident = Qwen35GgufResident::new(path.to_string(), devices, CAP, tier).with_kv_tier(KvTier::INT8).with_max_batch(SEQS as u32).with_prefix_slots(0).with_kv_block_rows(qwen35::int8_gguf_resident::KV_BLOCK_ROWS).with_gdn_state(gdn);
    let (engine, _, _) = resident.activate_engine(&[device]).unwrap_or_else(|e| panic!("activate {path}: {e}"));
    Some(engine)
}

/// Long-form prompts built from the fixture passages: the instruction asks for an answer long enough that the model
/// keeps writing sensible prose for the whole budget.
fn prompts(engine: &Qwen35GgufEngine) -> Vec<Vec<u32>> {
    include_str!("../fixtures/real_prompts.txt")
        .split("\n=====\n")
        .take(SEQS)
        .map(|body| {
            let text = format!(
                "<|im_start|>user\n{}\n\nWrite a long, detailed, section by section explanation of the text above for a new engineer: what it says, why it matters, what could go wrong, and how to check each claim.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                body.trim()
            );
            engine.instance().tokenize(&text)
        })
        .collect()
}

/// The served greedy continuations (no stop tokens: every sequence runs the whole budget).
fn serve(engine: Qwen35GgufEngine, prompts: &[Vec<u32>]) -> (Qwen35GgufEngine, Vec<Vec<u32>>) {
    let mut sched = Scheduler::new(engine, SEQS);
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: NEW_TOKENS, stop: Vec::new() })).collect();
    let mut done = sched.run();
    let streams = ids.iter().map(|id| done.remove(id).expect("the request completes")).collect();
    (sched.into_decoder(), streams)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

/// KL(softmax(p) || softmax(q)) in nats.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::MIN, f32::max) as f64;
        m + v.iter().map(|x| ((*x as f64) - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(a, b)| {
            let pa = ((*a as f64) - lp).exp();
            if pa < 1e-12 { 0.0 } else { pa * (((*a as f64) - lp) - ((*b as f64) - lq)) }
        })
        .sum()
}

#[test]
fn a_bf16_state_pool_serves_long_generations_like_the_f32_pool() {
    let Some(path) = brain_testutil::model_dir(STORE).filter(|d| std::path::Path::new(d).join("config.json").exists()) else {
        brain_testutil::skip(&format!("{STORE} is not in the model store (brain pull {STORE})"));
        return;
    };
    let Some(f32_engine) = engine(&path, GdnState::F32) else { return };
    assert!(!f32_engine.instance().gdn_state_is_bf16());
    let prompts = prompts(&f32_engine);
    let (f32_engine, reference) = serve(f32_engine, &prompts);
    assert!(reference.iter().all(|s| s.len() == NEW_TOKENS), "every sequence ran the whole budget");
    let f32_rows = f32_engine.instance().teacher_forced_logits_batch(&prompts, &reference).expect("f32 pooled logits");
    drop(f32_engine);

    let bf16_engine = engine(&path, GdnState::Bf16).expect("the same cards");
    assert!(bf16_engine.instance().gdn_state_is_bf16());
    let bf16_rows = bf16_engine.instance().teacher_forced_logits_batch(&prompts, &reference).expect("bf16 pooled logits");
    let (bf16_engine, free) = serve(bf16_engine, &prompts);
    drop(bf16_engine);

    let (mut agree, mut kl_sum, mut worst_seq) = (0usize, 0f64, (1f64, 0usize));
    for (seq, (f, b)) in f32_rows.iter().zip(&bf16_rows).enumerate() {
        let (mut seq_agree, mut seq_kl) = (0usize, 0f64);
        for (x, y) in f.iter().zip(b) {
            assert!(y.iter().all(|v| v.is_finite()), "sequence {seq} has non-finite bf16-pool logits");
            seq_agree += usize::from(argmax(x) == argmax(y));
            seq_kl += kl(x, y);
        }
        let top1 = seq_agree as f64 / f.len() as f64;
        let prefix = free[seq].iter().zip(&reference[seq]).take_while(|(a, b)| a == b).count();
        println!("sequence {seq}: {} positions, top-1 {:.2} %, mean KL {:.3e} nats, free-running bf16 pool matches the f32 stream for {prefix} tokens", f.len(), 100.0 * top1, seq_kl / f.len() as f64);
        agree += seq_agree;
        kl_sum += seq_kl;
        if top1 < worst_seq.0 {
            worst_seq = (top1, seq);
        }
    }
    let positions = (SEQS * NEW_TOKENS) as f64;
    let (top1, mean_kl) = (agree as f64 / positions, kl_sum / positions);
    println!("bf16 pool vs f32 pool: {positions} positions, top-1 agreement {:.2} %, mean KL {mean_kl:.3e} nats, worst sequence {} at {:.2} %", 100.0 * top1, worst_seq.1, 100.0 * worst_seq.0);
    assert!(top1 >= MIN_TOP1, "top-1 agreement {top1:.4} is under {MIN_TOP1}");
    assert!(mean_kl <= MAX_MEAN_KL, "mean KL {mean_kl:.3e} nats is over {MAX_MEAN_KL:.3e}");
}
