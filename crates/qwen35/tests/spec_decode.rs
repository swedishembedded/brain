// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Speculative decoding on the hybrid Gated-DeltaNet/GQA decoder must be
//! LOSSLESS: whatever a drafter proposes, the emitted tokens have to be the
//! ones plain greedy decoding would have emitted. That is the entire premise
//! of the technique - it buys throughput, never a different distribution - so
//! a gate that only checks "the fast path produced plausible text" checks
//! nothing.
//!
//! What makes this model harder than the textbook version, and what these
//! tests are really about: **three quarters of its layers are recurrent.**
//! `full_attention_interval = 4` puts Gated DeltaNet at every layer whose
//! index is not `3 mod 4`, and a GDN layer carries a `[bh, dk, dv]` recurrent
//! state plus a causal-conv history tail from token to token. A verify pass
//! over `k+1` speculative rows advances that state through ALL of them,
//! including the rejected tail - and unlike a KV cache, whose rejected rows
//! are simply never read again (a decode step's `seq_lens` bound stops at
//! `pos`), a recurrent state has no rows to leave behind. It is one buffer
//! that has already absorbed the wrong tokens.
//!
//! So the rollback is the correctness-critical part, and how to gate it took
//! one wrong turn worth recording, because the wrong version LOOKED right and
//! passed.
//!
//! The obvious gate is end-to-end: run an adversarial drafter, then compare
//! the recurrent state against what the plain tape leaves. Measured on this
//! fixture, with the restore deliberately deleted:
//!
//! | quantity                                | correct | no restore |
//! |-----------------------------------------|---------|------------|
//! | GDN recurrent state, worst maxabs       | 1.0e-6  | 6.0e-5     |
//! | hidden state at emitted tokens, maxabs  | 1.2e-7  | 1.8e-6     |
//!
//! Every one of those numbers is tiny, and the reason is the third one that
//! was not being looked at: **the state's own RMS on this fixture is 4.6e-7**.
//! The `tiny()` recurrent state is numerical noise, so the "correct" figure
//! is not a small error, it is a comparison of one noise sample against
//! another - and no absolute tolerance separates 1.0e-6 from 6.0e-5 in a way
//! that means anything. An early version of this file bounded that comparison
//! at 1e-4 and passed with the rollback removed. `slow_decay` (copied from
//! `tests/chunked_prefill.rs`, which needs it for the same reason) does not
//! rescue it: it fixes the DECAY, and what is missing here is magnitude in
//! the updates, not persistence of them.
//!
//! The generalizable form: before bounding an absolute error, measure the
//! scale of the thing the error is on. A gate whose "correct" reading is at
//! the noise floor is measuring the noise floor.
//!
//! So the rollback claim is carried by
//! `gdn_snapshot_restores_the_state_a_speculative_chunk_moved`, which asserts
//! EXACT equality on the snapshot/restore primitive (no tolerance to get
//! wrong, and it measures 0 - a device-side copy either copies or does not)
//! and carries its own non-vacuousness proof: it first asserts that the
//! speculative chunk really moved the state. The end-to-end tests then carry
//! the claim that actually matters to a caller - identical tokens - which is
//! a discrete property needing no tolerance either.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing differently-sized buffers");
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// See the module doc: retunes the decay gate so the recurrent state survives
/// from token to token, which is what makes a rollback bug observable.
/// Identical to `tests/chunked_prefill.rs`'s helper of the same name.
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

/// Eight layers, so GQA lands at index 3 AND 7 and the recurrent layers are
/// exercised both before and between them - the same reason
/// `tests/chunked_prefill.rs` widens `tiny()`.
fn fixture(gpu: Gpu) -> (Qwen35, Vec<u32>) {
    let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny() };
    let init = slow_decay(&cfg, qwen35::init::init_weights(&cfg, 7));
    let model = Qwen35::new_on(gpu, cfg.clone(), 1, 64, &init);
    let prompt: Vec<u32> = (0..6).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    (model, prompt)
}

/// The host-side vocab projection, exactly as `tests/sample_generate.rs` does
/// it: this crate's serving paths keep the head off-device, so a test that
/// wants token ids does the same matvec they do.
fn head_argmax(model: &Qwen35) -> impl Fn(&[f32]) -> u32 + '_ {
    let head = model.read_weight(model.cfg.head_weight());
    let (vocab, d) = (model.cfg.vocab as usize, model.cfg.d_model as usize);
    move |hidden: &[f32]| {
        let logits = model::hostmath::matvec_par(&head, hidden, vocab, d);
        let mut bi = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[bi] {
                bi = i;
            }
        }
        bi as u32
    }
}

/// Plain greedy decoding through the one-token-per-dispatch tape: the
/// reference every speculative run below must reproduce exactly. Returns the
/// tokens AND the hidden state each was emitted from, because the token
/// sequence alone is a step function over the hidden states and hides small
/// state errors (see the module doc).
fn plain_greedy(model: &Qwen35, prompt: &[u32], max_new: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let argmax = head_argmax(model);
    model.reset_decode_cache();
    let mut hidden = Vec::new();
    for &t in prompt {
        hidden = model.step(t);
    }
    let (mut toks, mut hs) = (Vec::new(), Vec::new());
    for _ in 0..max_new {
        let next = argmax(&hidden);
        toks.push(next);
        hs.push(hidden.clone());
        hidden = model.step(next);
    }
    (toks, hs)
}

/// Every GDN layer's recurrent state, concatenated - the buffer a rejected
/// speculative tail corrupts, read directly rather than through its effect on
/// a sampled token.
fn gdn_state_all(model: &Qwen35) -> Vec<f32> {
    let mut out = Vec::new();
    for (l, ty) in model.cfg.layer_types().iter().enumerate() {
        if *ty == LayerType::Linear {
            out.extend(model.debug_gdn_state(l));
        }
    }
    assert!(!out.is_empty(), "fixture has no recurrent layers - the rollback gate would be vacuous");
    out
}

/// **The losslessness gate.** An oracle drafter (proposes the true
/// continuation, so everything is accepted) and an adversarial one (proposes
/// a constant, so almost nothing is) must BOTH reproduce plain greedy's
/// tokens exactly, and the good one must cost materially fewer target
/// forwards. Ports the shape of `qwen3::serve`'s own `spec_decode_matches_
/// greedy`, which gates the same algorithm on a purely-attention model.
#[test]
fn spec_decode_matches_plain_greedy() {
    let (model, prompt) = fixture(Gpu::new(pipelines()));
    let max_new = 16usize;
    let (want, want_hidden) = plain_greedy(&model, &prompt, max_new);
    let argmax = head_argmax(&model);

    // Oracle: propose exactly what greedy will do. Nothing is ever rejected,
    // so this path never exercises the rollback - which is precisely why the
    // adversarial case below is not optional.
    let full: Vec<u32> = prompt.iter().copied().chain(want.iter().copied()).collect();
    let (got, stats) = model.spec_decode_greedy(&prompt, max_new, 4, &argmax, &mut |ctx: &[u32], want_n: u32| {
        (0..want_n as usize).map(|i| full.get(ctx.len() + i).copied().unwrap_or(0)).collect()
    });
    println!(
        "oracle draft : {} tokens in {} target forwards, {}/{} draft tokens accepted ({:.2} accepted/round)",
        got.len(),
        stats.target_forwards,
        stats.accepted,
        stats.proposed,
        stats.accepted as f64 / stats.rounds as f64
    );
    assert_eq!(got, want, "speculative decode with a perfect draft changed the output");
    assert!(stats.target_forwards < max_new, "a perfect draft must cut target forwards below one per token: {} vs {max_new}", stats.target_forwards);

    // Adversarial: a constant proposal. Rejections are the common case, so
    // every round rolls the recurrent state back.
    let (got_bad, stats_bad) = model.spec_decode_greedy(&prompt, max_new, 4, &argmax, &mut |_ctx: &[u32], want_n: u32| vec![0u32; want_n as usize]);
    println!(
        "adversarial  : {} tokens in {} target forwards, {}/{} draft tokens accepted",
        got_bad.len(),
        stats_bad.target_forwards,
        stats_bad.accepted,
        stats_bad.proposed
    );
    assert_eq!(got_bad, want, "speculative decode with a bad draft changed the output - the accept/reject or the rollback is wrong");
    assert!(stats_bad.target_forwards >= stats.target_forwards, "a bad draft cannot need fewer target forwards than a perfect one");

    // The tokens agreeing is necessary but not sufficient - argmax is a step
    // function - so pin the hidden states the speculative path actually
    // emitted from too. Measured on this fixture: 1.2e-7 correct, 1.8e-6 with
    // the recurrent restore deleted, so the 1e-5 bound sits 84x above correct
    // behaviour and 5.6x below that break. It is a real but SHALLOW gate for
    // the reason the module doc gives (this fixture's recurrent state is near
    // the noise floor); the rollback claim itself is carried by
    // `gdn_snapshot_restores_the_state_a_speculative_chunk_moved`.
    let (_, got_hidden) = model.spec_decode_greedy_hidden(&prompt, max_new, 4, &argmax, &mut |_ctx: &[u32], want_n: u32| vec![0u32; want_n as usize]);
    let worst = got_hidden.iter().zip(&want_hidden).fold(0.0f32, |m, (g, w)| m.max(maxabs(g, w)));
    println!("adversarial  : worst hidden-state maxabs vs plain greedy over {max_new} emitted tokens = {worst:e}");
    assert!(worst < 1e-5, "speculative decode's emitted hidden states drifted from plain greedy's: maxabs={worst:e}");
}

/// **The rollback gate**: the snapshot/restore primitive the speculative loop
/// undoes a rejected tail with, tested directly rather than through its
/// effect on a token.
///
/// Directly, and not end-to-end, for a reason worth writing down. The obvious
/// gate - "run an adversarial draft, compare the recurrent state against the
/// plain tape's" - is VACUOUS on a fixture like this one, and measurably so:
/// with the restore deleted the state moved by 6.0e-5 against a state whose
/// own RMS is 4.6e-7, i.e. the `tiny()` recurrent state is numerical noise
/// and any absolute tolerance either admits the break or rejects correct
/// code. That is the same dilution `tests/chunked_prefill.rs` ran into and
/// answered by deferring the recurrent claim to a fixture built for it.
///
/// So this gate asserts on the mechanism with EXACT equality, which needs no
/// tolerance and cannot be diluted, and it carries its own non-vacuousness
/// proof: the middle assertion fails if the speculative chunk did not really
/// move the state, which is the only way the round trip could pass for the
/// wrong reason.
#[test]
fn gdn_snapshot_restores_the_state_a_speculative_chunk_moved() {
    let (model, prompt) = fixture(Gpu::new(pipelines()));

    model.reset_decode_cache();
    model.prefill_chunked(&prompt, 16);
    let before = gdn_state_all(&model);

    let snap = model.gdn_snapshot();

    // A speculative verify chunk's shape: several rows at consecutive
    // positions, of which a rejection would commit only the first.
    let spec: Vec<u32> = (0..6).map(|i| (i * 11 + 2) % model.cfg.vocab).collect();
    model.prefill_chunked(&spec, 16);
    let after_chunk = gdn_state_all(&model);
    assert_ne!(
        after_chunk, before,
        "a {}-row speculative chunk left the recurrent state untouched - this gate would pass trivially, and the rollback it exists to prove would be unnecessary",
        spec.len()
    );

    model.gdn_restore(&snap);
    let after_restore = gdn_state_all(&model);
    let err = maxabs(&after_restore, &before);
    let moved_by = maxabs(&after_chunk, &before);
    println!("gdn rollback : a {}-row speculative chunk moved the recurrent state by maxabs={moved_by:e}; after restore, maxabs vs the pre-chunk state = {err:e}", spec.len());
    assert_eq!(after_restore, before, "snapshot/restore did not return the Gated-DeltaNet recurrent state exactly (worst maxabs={err:e}, against a chunk that had moved it by {moved_by:e})");
}

/// End-to-end, the losslessness claim is carried by token equality (the tests
/// above) rather than by the recurrent state, for the dilution reason
/// `gdn_snapshot_restores_the_state_a_speculative_chunk_moved` documents. What
/// this adds is that an adversarial drafter - one that forces the rollback on
/// EVERY round - still reproduces plain greedy exactly over a long enough run
/// for a leaked state to have compounded into a different token.
#[test]
fn adversarial_drafting_still_reproduces_plain_greedy() {
    let (model, prompt) = fixture(Gpu::new(pipelines()));
    let max_new = 24usize;
    let argmax = head_argmax(&model);

    let (want, _) = plain_greedy(&model, &prompt, max_new);
    let (got, stats) = model.spec_decode_greedy(&prompt, max_new, 6, &argmax, &mut |_ctx: &[u32], want_n: u32| vec![0u32; want_n as usize]);

    assert!(
        stats.proposed > stats.accepted,
        "this gate needs real rejections to be meaningful, but {}/{} draft tokens were accepted",
        stats.accepted,
        stats.proposed
    );
    println!("adversarial  : {} rounds, {}/{} accepted, {} target forwards for {max_new} tokens", stats.rounds, stats.accepted, stats.proposed, stats.target_forwards);
    assert_eq!(got, want, "adversarial-draft speculative decode changed the tokens");
}

/// A drafter that is right some of the time is the case both tests above
/// straddle without landing on: every round mixes a real accepted prefix with
/// a real rejected tail, so the rollback has to restore a state that is
/// neither "everything" nor "nothing". Runs on the CPU backend as well, which
/// is where a kernel-selection difference between the `n = 1` and `n = k+1`
/// tapes would show up differently than on wgpu.
#[test]
fn spec_decode_partial_acceptance_matches_greedy_cpu() {
    let (model, prompt) = fixture(Gpu::new_cpu(pipelines()));
    let max_new = 10usize;
    let argmax = head_argmax(&model);
    let (want, _) = plain_greedy(&model, &prompt, max_new);
    let full: Vec<u32> = prompt.iter().copied().chain(want.iter().copied()).collect();

    // Truthful for the first two proposals of each round, then wrong: forces
    // a partial accept every round rather than all-or-nothing.
    let (got, stats) = model.spec_decode_greedy(&prompt, max_new, 4, &argmax, &mut |ctx: &[u32], want_n: u32| {
        (0..want_n as usize).map(|i| if i < 2 { full.get(ctx.len() + i).copied().unwrap_or(0) } else { 0 }).collect()
    });
    println!(
        "partial      : {} rounds, {}/{} accepted ({:.2}/round), {} target forwards for {max_new} tokens",
        stats.rounds,
        stats.accepted,
        stats.proposed,
        stats.accepted as f64 / stats.rounds as f64,
        stats.target_forwards
    );
    assert_eq!(got, want, "partial-acceptance speculative decode changed the output");
    assert!(stats.accepted > 0 && stats.accepted < stats.proposed, "fixture did not produce the intended partial acceptance: {}/{}", stats.accepted, stats.proposed);
}
