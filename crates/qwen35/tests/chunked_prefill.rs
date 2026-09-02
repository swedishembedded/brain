// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen35::prefill_chunked` (multi-token-per-dispatch prompt replay) must
//! leave EXACTLY the decode state a token-by-token `Qwen35::step` replay
//! leaves: the same GQA KV-cache rows, the same Gated-DeltaNet recurrent
//! state, and the same causal-conv history tail. The observable proof is
//! CONTINUITY - after the prompt, a few further single-token `step` calls
//! must produce the same hidden states either way, which they cannot if any
//! of that state was seeded wrong (a wrong GDN `final_state`, a conv history
//! that lost the chunk boundary, or KV rows written at the wrong offset all
//! diverge immediately and visibly, not in the third decimal).
//!
//! The prompt is 14 tokens, replayed at chunk sizes 4 and 8: both run
//! SEVERAL rounds (4+4+4+2, 8+6) with a ragged last one, so every round after
//! the first must continue from the previous round's state rather than from
//! zero - the case a single whole-prompt dispatch would never exercise. The
//! lengths are picked so `model::gdn::gdn_chunk_size` gives a round a REAL
//! multi-token GDN chunk (4, 8 and 2 respectively): an odd round length
//! collapses it to 1, which degenerates the chunked recurrence into the
//! sequential one and would leave `gdn_chunk_fwd`'s own intra-chunk math
//! (the UT transform) untested. A third case runs the whole prompt in ONE
//! chunk, which proves the fresh-sequence (zero initial state, zero conv
//! history) path independently.
//!
//! The config is `tiny()` widened to EIGHT layers, which puts a GQA layer at
//! index 3 AND index 7 (`full_attention_interval = 4`). That second one is
//! what makes the gate sensitive to a round's INTERNAL causal masking: with a
//! single GQA layer (plain `tiny()`) the non-final rows of a round feed
//! nothing that outlives the round - each round re-embeds its own tokens, so
//! only the cache, the GDN state, and the round's last row are observable -
//! and a deliberately broken per-row causal mask still passed. With two, a
//! mis-masked row of layer 3 changes layer 7's K/V for that row, which lands
//! in the cache and is read by every later token.
//!
//! Tolerance, and why this is not a bit-exactness gate: both paths are the
//! same fp32 kernels over the same weights, but NOT the same dispatches.
//! A chunk of `n` rows selects different matmul/RMSNorm variants than the
//! `n = 1` decode tape (`Ops::matmul`'s own `m`-dependent selection,
//! `block::rms_variant`), and the Gated-DeltaNet recurrence is evaluated by
//! `gdn_chunk_fwd`'s chunked-parallel form instead of `gdn_recurrent_step`'s
//! sequential one - mathematically identical, numerically a different
//! reduction order. This is the same crossing `tests/decode_step.rs` already
//! gates (full prefill vs. per-token decode).
//!
//! The `1e-5` bound is MEASURED, not guessed, and sits two orders of
//! magnitude clear on both sides. Correct implementation: 0 on the CPU JIT,
//! 3.7e-9 on wgpu. Deliberately broken ones, each re-measured against this
//! fixture: KV rows filled at offset 0 instead of `start` -> 6.2e-3; a
//! per-query causal mask flattened to the round's first position -> 3.1e-3;
//! chunk-RELATIVE instead of absolute M-RoPE positions -> 1.4e-3 (which the
//! `2e-3` bound `decode_step.rs` uses on logits would have let through - the
//! reason this file does not simply reuse that number).
//!
//! What this gate deliberately does NOT carry is the Gated-DeltaNet
//! round-to-round state threading: on random weights a completely DROPPED
//! recurrent state moves the final hidden state by only ~5e-7 here, which no
//! honest bound separates from fp32 noise. That claim is gated where the
//! signal is undiluted instead - `crates/model/tests/gdn_mixer_stream.rs`,
//! where the same break measures 0.34 against a 2e-7 baseline.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Retune the Gated-DeltaNet decay gate into a regime where the recurrent
/// state actually SURVIVES from one token to the next - without this the
/// round-to-round state threading this file exists to gate is unobservable.
///
/// `init::init_weights` mirrors the reference's fresh-weight init: `dt_bias =
/// 1`, `A = Uniform(0,16)`. The gate is `g = -A*softplus(aproj+dt_bias)`, so a
/// fresh model's per-token state decay is `exp(g) ~ exp(-10)`: the state is
/// annihilated within about two tokens, and a chunked prefill that carried NO
/// state between rounds (verified: a deliberately zeroed `initial_state`)
/// still reproduced the per-token replay exactly. `A = 0.05`, `dt_bias = -1`
/// puts the decay at ~0.98/token, which is the regime a trained checkpoint
/// with real long-range memory is in and the only one in which "did the
/// second round continue from the first round's state?" is a question with an
/// observable answer.
fn slow_decay(cfg: &Qwen35Config, mut w: HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
    for (name, numel) in cfg.param_list() {
        if name.ends_with(".A_log") {
            w.insert(name, vec![0.05f32.ln(); numel]);
        } else if name.ends_with(".dt_bias") {
            w.insert(name, vec![-1.0f32; numel]);
        }
    }
    w
}

/// Replay `prompt` through both paths at chunk size `chunk`, then continue
/// both with the same `tail` tokens one at a time and compare.
fn run(gpu: Gpu, chunk: u32) {
    let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny() };
    let d = cfg.d_model as usize;
    let init = slow_decay(&cfg, qwen35::init::init_weights(&cfg, 7));
    let m = Qwen35::new_on(gpu, cfg.clone(), 1, cfg.block_size, &init);

    let prompt: Vec<u32> = (0..14).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    let tail: Vec<u32> = (0..3).map(|i| (i * 7 + 1) % cfg.vocab).collect();

    // Reference: the existing one-dispatch-per-token replay.
    m.reset_decode_cache();
    let mut want_last = Vec::new();
    for &tok in &prompt {
        want_last = m.step(tok);
    }
    let want_tail: Vec<Vec<f32>> = tail.iter().map(|&tok| m.step(tok)).collect();
    assert_eq!(m.decode_pos(), (prompt.len() + tail.len()) as u32);

    // Under test: the same prompt through the chunked prefill, then the SAME
    // per-token continuation.
    m.reset_decode_cache();
    let got_last = m.prefill_chunked(&prompt, chunk);
    assert_eq!(got_last.len(), d, "prefill_chunked must return one [d_model] hidden state");
    assert!(got_last.iter().all(|x| x.is_finite()), "prefill_chunked produced a non-finite hidden state");
    assert_eq!(m.decode_pos(), prompt.len() as u32, "prefill_chunked must leave the decode position at the prompt length");
    let got_tail: Vec<Vec<f32>> = tail.iter().map(|&tok| m.step(tok)).collect();
    assert_eq!(m.decode_pos(), (prompt.len() + tail.len()) as u32);

    let last_err = maxabs(&got_last, &want_last);
    assert!(last_err < 1e-5, "chunk={chunk}: prompt's last hidden state maxabs={last_err}");
    let mut worst = last_err;
    for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
        let err = maxabs(got, want);
        worst = worst.max(err);
        assert!(err < 1e-5, "chunk={chunk}: continuation token {i} hidden state maxabs={err} (chunked prefill left the decode state wrong)");
    }
    println!("chunked_prefill(chunk={chunk}): worst maxabs over prompt-last + {} continuation steps = {worst:e}", tail.len());
}

#[test]
fn chunked_prefill_matches_token_by_token_replay_cpu() {
    run(Gpu::new_cpu(pipelines()), 4);
}

#[test]
fn chunked_prefill_matches_token_by_token_replay_default_backend() {
    run(Gpu::new(pipelines()), 4);
    run(Gpu::new(pipelines()), 8);
}

/// The single-round case: chunk >= prompt length, so the whole prompt is one
/// dispatch round starting from a fresh (zero) recurrent state and empty KV
/// cache. Separated from the multi-round test above because a bug in the
/// round-to-round state threading passes this one and fails that one, and
/// vice versa for a bug in the fresh-sequence seeding.
#[test]
fn whole_prompt_single_chunk_matches_token_by_token_replay() {
    run(Gpu::new(pipelines()), 16);
}
