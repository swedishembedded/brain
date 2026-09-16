// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint gates for SPECULATIVE decoding on the multi-GPU INT8
//! GGUF-resident serving path - `Qwen35GgufInstance::generate_speculative`.
//!
//! Two questions, and they are not the same question.
//!
//! **Is it lossless?** A speculative decoder that changes the output has not
//! accelerated anything, it has swapped models. The gate is exact token
//! equality against non-speculative decoding, at `temp = 0`, on the same
//! instance - run with a drafter chosen to be WRONG most of the time, because
//! a drafter that is always right never exercises the accept/reject logic or
//! the recurrent-state rollback the losslessness depends on.
//!
//! **Where that gate can and cannot be exact, measured.** Plain decode
//! advances one token per `run_decode_batch` dispatch set; a speculative
//! verify advances its rows per `run_prefill_chunk_stage`. Those compute the
//! same function in a different reduction order, and on the real checkpoint
//! the difference is large: `how_far_apart_are_the_decode_tape_and_the_chunk_
//! tape_on_the_real_checkpoint` measures ~1.0-1.6 absolute on logits of order
//! ten, at every position past the first.
//!
//! It reaches the ARGMAX at 1 of 24 positions - and at exactly the position
//! whose top-two margin is 0.0169, where every other position's margin is
//! 0.33 to 6.7. That is a genuine near-tie being broken differently by two
//! equally valid tapes, on a prompt (`"The capital city of France is"`) whose
//! own continuation is ambiguous; both results are fluent and correct
//! ("...France is Paris" against "...Germany is Berlin").
//!
//! So byte-identity to `generate` is not a property any speculative
//! implementation over the chunk tape can have here - confirmed by driving
//! this very loop with a drafter that proposes NOTHING, which reduces the
//! accept/reject logic to a no-op and still diverges at exactly that token -
//! and asserting it would blame this crate's speculation for a pre-existing
//! numerical property of the stack.
//!
//! What IS asserted, exactly and without tolerance, is that speculation
//! changes nothing *on a fixed tape*
//! (`speculative_decoding_is_exactly_equivalent_to_the_same_tape_without_speculation`,
//! measured with 9 of 17 proposals rejected, so the rollback runs on most
//! rounds) - which is precisely the claim the accept/reject logic and the
//! rollback are responsible for. The same claim is additionally gated against
//! `generate`-equivalent plain decoding at `tiny()` dims in
//! `tests/spec_decode.rs`, where the two tapes DO agree in argmax
//! (`tests/decode_step.rs` asserts that) and the same code runs.
//!
//! **Is it faster, and by how much?** That is decided by one number - draft
//! tokens accepted per round - and by a cost this particular model imposes
//! that a pure-attention model does not. Three quarters of its layers are
//! recurrent, and a rejected speculative tail cannot be left in a recurrent
//! state the way it can be left in a KV cache, so a round that rejects pays a
//! second target forward to re-commit its accepted prefix. Break-even is
//! therefore two accepted draft tokens per round, not one.
//!
//! So these tests measure a LADDER rather than a single figure, because a
//! single figure would confound the machinery's cost with the drafter's
//! quality:
//!
//! * an ORACLE drafter (fed the answer that tape already produced) - the
//!   ceiling this hardware allows, i.e. what a perfect draft model would buy,
//!   and the number that says whether porting one is worth it;
//! * a model-free n-gram drafter that proposes the continuation of the most
//!   recent earlier occurrence of the current suffix - a real drafter, with a
//!   real (workload-dependent) acceptance rate and no second checkpoint;
//! * the plain path, same instance, as the baseline.
//!
//! Measured on two Tesla P40s, Q8_0 weights served INT8, greedy, one load
//! per test (plain decode 5.7-6.7 tok/s across these prompts). The `before`
//! column is the same ladder when a verify round still paid a 256-token
//! prefill round's per-layer device drain, which the qwen35 ledger's M29
//! removed; nothing about the drafters or the accept/reject logic differs
//! between the two columns:
//!
//! | drafter                          | before | after | vs plain | accepted/round |
//! |----------------------------------|-------:|------:|---------:|---------------:|
//! | none (chunk tape floor)          |   3.7  |  6.18 |    0.92x |  -             |
//! | n-gram, free-form prompt         |   4.4  |  6.79 |    1.01x |  0.57          |
//! | oracle, k=3                      |  12.2  | 18.10 |    2.70x |  3.00          |
//! | oracle, k=7                      |  17.0  | 21.73 |    3.24x |  7.00          |
//! | none, repetition workload        |   3.2  |  4.75 |    0.83x |  -             |
//! | n-gram, repetition workload      |   9.1  | 10.36 |    1.81x |  5.12          |
//!
//! The floor is workload-dependent and the two `none` rows say why: it is
//! 0.92x behind a 6-token prompt and 0.83x behind a 90-token one, because what
//! remains of the gap is the chunk tape's own attention and GDN kernels doing
//! more work per row than the decode tape's at the same context depth - not a
//! fixed per-round cost, which is what this used to be.
//!
//! Read that table as the answer to "is a draft MODEL worth porting": the
//! ceiling at a full block is 3.2x, and a drafter now has to average only ~1
//! accepted token per round to break even rather than ~5, because the floor
//! it starts from is 0.92x instead of 0.55x. The remaining gap between the
//! oracle's 3.2x and the 8x a `k = 7` round would buy on a machine with no
//! per-round cost at all is the verify round's own shape, not the drafter's:
//! an 8-row round still reads every weight in the model once, the same as one
//! decode step, so 8 tokens for ~2.5 steps' wall clock is the realistic shape
//! of the win.
//!
//! Everything self-skips loudly without the real `Qwen3.8-27B*.gguf` named by
//! `BRAIN_QWEN35_GGUF`, exactly as `tests/gguf_resident_real.rs` does. Run:
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/.local/share/brain/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test --release --offline -p brain-qwen35 --test gguf_resident_spec_real \
//!   -- --nocapture --test-threads=1
//! ```

use std::time::Instant;

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35::int8_gguf_resident::{Qwen35GgufInstance, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// `prompt + max_new + speculation window`. Wider than `gguf_resident_real`'s
/// `CAP` because the repetition workload below carries a real passage.
const CAP: u32 = 1024;
const RESERVE: u64 = 2 << 30;

fn gguf_path() -> Option<String> {
    match std::env::var("BRAIN_QWEN35_GGUF") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => {
            brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
            None
        }
    }
}

fn real_devices() -> Vec<(Device, u64)> {
    gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect()
}

/// Load the real checkpoint across the real cards, as the owned concrete
/// instance (not the `dyn Instance` façade): these gates call
/// `generate`/`generate_speculative` directly so that the two paths are
/// compared with nothing else in between.
fn load() -> Option<Qwen35GgufInstance> {
    let path = gguf_path()?;
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with enough free memory for the real checkpoint");
        return None;
    }
    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    Some(r.activate_owned(&placed).expect("activate the real checkpoint across the real cards"))
}

/// **Model-free n-gram drafting.** Look for the most recent earlier
/// occurrence of the last `n` tokens of the context and propose whatever
/// followed it, longest match first.
///
/// This is a real drafter, not a stub: on any workload with input repetition
/// (quoting a source, editing a file, structured output) it accepts at a
/// useful rate for zero extra weights and roughly zero time, which makes it
/// the honest floor to measure a real draft MODEL against. On free-form
/// prose it accepts almost nothing, which is equally worth measuring - and is
/// why the two workloads below are separate tests.
fn ngram_draft(ctx: &[u32], want: u32) -> Vec<u32> {
    for n in (2usize..=4).rev() {
        if ctx.len() <= n {
            continue;
        }
        let suffix = &ctx[ctx.len() - n..];
        // Most recent earlier occurrence, scanning backwards.
        for start in (0..ctx.len() - n).rev() {
            if &ctx[start..start + n] == suffix {
                let from = start + n;
                let take = want as usize;
                if from < ctx.len() {
                    let end = (from + take).min(ctx.len());
                    let out = ctx[from..end].to_vec();
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
    }
    Vec::new()
}

struct Run {
    text: String,
    ids: Vec<u32>,
    decode_s: f64,
    tok_s: f64,
}

fn plain(inst: &Qwen35GgufInstance, prompt: &[u32], max_new: u32) -> Run {
    let t = Instant::now();
    let ids = inst.generate(prompt, max_new, 0.0, 0, 1.0, 0, &mut |_| false).expect("plain greedy decode");
    inst.poll_wait();
    let decode_s = t.elapsed().as_secs_f64();
    Run { text: inst.detokenize(&ids), tok_s: ids.len() as f64 / decode_s, ids, decode_s }
}

fn speculative(inst: &Qwen35GgufInstance, prompt: &[u32], max_new: u32, k: u32, draft: &mut dyn FnMut(&[u32], u32) -> Vec<u32>) -> (Run, qwen35::model::SpecDecodeStats) {
    let t = Instant::now();
    let (ids, stats) = inst.generate_speculative(prompt, max_new, k, draft, &mut |_| false).expect("speculative decode");
    inst.poll_wait();
    let decode_s = t.elapsed().as_secs_f64();
    (Run { text: inst.detokenize(&ids), tok_s: ids.len() as f64 / decode_s, ids, decode_s }, stats)
}

fn report(label: &str, r: &Run, stats: Option<&qwen35::model::SpecDecodeStats>, base: f64) {
    match stats {
        Some(s) => println!(
            "  {label:<22} {:>6.2} tok/s  ({:.1}x)  |  {:>2} tokens in {:>4.1}s, {} target forwards, {}/{} draft accepted, {:.2} accepted/round",
            r.tok_s,
            r.tok_s / base,
            r.ids.len(),
            r.decode_s,
            s.target_forwards,
            s.accepted,
            s.proposed,
            s.accepted_per_round()
        ),
        None => println!("  {label:<22} {:>6.2} tok/s  (1.0x)  |  {:>2} tokens in {:>4.1}s", r.tok_s, r.ids.len(), r.decode_s),
    }
}

/// **The underlying measurement**, with no speculation in it at all: how far
/// apart are the DECODE tape and the CHUNK tape on the real checkpoint, and
/// does that gap reach the argmax?
///
/// This is the question every losslessness claim here rests on, so it is
/// measured directly rather than inferred from a diverging generation. For
/// each position of a real continuation: the `[vocab]` logits a single-token
/// decode step produces, against the logits the same position gets from a
/// one-row prefill chunk. Reported per position: worst absolute difference,
/// whether the argmax moved, and the winning token's margin over the
/// runner-up - because a tape difference only becomes a different TOKEN where
/// that margin is smaller than the difference, and on a prompt whose
/// continuation is genuinely ambiguous the margin can be near zero.
///
/// `tests/decode_step.rs` asserts these two tapes agree in argmax at `tiny()`
/// dims on random weights (bound 2e-2 on logits). This measures the same
/// crossing where it actually matters.
#[test]
fn how_far_apart_are_the_decode_tape_and_the_chunk_tape_on_the_real_checkpoint() {
    let Some(inst) = load() else { return };
    let prompt = inst.tokenize("The capital city of France is");
    let steps = 24u32;

    let trace = inst.tape_comparison_trace(&prompt, steps).expect("tape comparison trace");
    let (dec_logits, chunk_logits) = (&trace.decode, &trace.chunk);

    println!("\ndecode tape vs one-row chunk tape, real checkpoint, {steps} positions:");
    println!("   pos |   maxabs |  argmax | top1-top2 margin (decode tape)");
    let mut mismatches = 0usize;
    let mut worst = 0.0f32;
    for (i, (d, c)) in dec_logits.iter().zip(chunk_logits).enumerate() {
        let err = d.iter().zip(c).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        worst = worst.max(err);
        let am = |s: &[f32]| s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(j, _)| j).expect("non-empty logits");
        let (ad, ac) = (am(d), am(c));
        let mut sorted: Vec<f32> = d.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let margin = sorted[0] - sorted[1];
        if ad != ac {
            mismatches += 1;
        }
        println!("  {i:>4} | {err:>8.2e} | {:>7} | {margin:>8.4}{}", if ad == ac { "same" } else { "MOVED" }, if ad != ac { "   <-- a different token" } else { "" });
    }
    println!("  worst maxabs over {steps} positions = {worst:e}, argmax moved at {mismatches}/{steps} positions");
    println!(
        "  -> a speculative decoder built on the chunk tape can be byte-identical to plain decode \
         only where this count is zero; where it is not, the two tapes simply pick different tokens \
         and no accept/reject logic can reconcile them."
    );
}

/// **Isolating diagnostic**: when the speculative path and the plain path
/// disagree, is the cause the accept/reject logic, or the fact that the two
/// paths run numerically different tapes for the same arithmetic?
///
/// This drives the speculative loop with a drafter that proposes NOTHING.
/// Every round is then a one-row verify chunk committing exactly one token -
/// no proposals, no rejections, no rollback, the accept/reject logic reduced
/// to a no-op - so the ONLY remaining difference from plain decode is which
/// dispatch shape computed the row. Whatever it shows is therefore a property
/// of the two tapes and not of this crate's speculation, which is why nothing
/// about the token sequence is asserted here; the magnitude of that tape
/// difference is measured by the test above, and the losslessness of the
/// speculation proper is gated exactly, on a fixed tape, by the test below.
///
/// Measured on the real checkpoint: it diverges from `generate` at token 7 of
/// `"The capital city of France is"`, both continuations fluent and correct
/// ("...France is Paris" against "...Germany is Berlin"), and runs at 3.7
/// tok/s against plain decode's 6.6 - the one-row chunk handicap that
/// `generate_speculative`'s own doc records as this path's known floor.
#[test]
fn an_empty_draft_isolates_the_tape_from_the_accept_reject_logic() {
    let Some(inst) = load() else { return };
    let prompt = inst.tokenize("The capital city of France is");
    let max_new = 24u32;

    let base = plain(&inst, &prompt, max_new);
    let (empty, stats) = speculative(&inst, &prompt, max_new, 7, &mut |_ctx: &[u32], _want: u32| Vec::new());
    println!("\nempty-draft isolation, {max_new} tokens, greedy:");
    report("plain (decode tape)", &base, None, base.tok_s);
    report("empty draft (chunk)", &empty, Some(&stats), base.tok_s);
    assert_eq!(stats.proposed, 0, "this diagnostic requires a drafter that proposes nothing");
    assert_eq!(stats.target_forwards, stats.rounds, "an empty draft must never trigger a re-commit forward");

    match base.ids.iter().zip(&empty.ids).position(|(a, b)| a != b) {
        None => println!("  -> the one-row chunk tape and the decode tape agree token for token"),
        Some(i) => println!(
            "  -> the tapes diverge at token {i}: decode tape {:?} vs chunk tape {:?}\n     plain: {:?}\n     chunk: {:?}",
            base.ids[i], empty.ids[i], base.text, empty.text
        ),
    }
    assert!(empty.text.contains("Paris"), "the chunk tape must still continue this prompt with Paris, got {:?}", empty.text);
}

/// **The losslessness gate**, stated the only way it can be true at this
/// scale: speculative decoding must produce EXACTLY what non-speculative
/// decoding produces *on the same tape*.
///
/// The baseline is this same loop driven by a drafter that proposes nothing,
/// which reduces every round to one committed token per verify chunk - no
/// proposals, no rejections, no rollback - while leaving the tape, the
/// positions and the cache handling identical. The comparison therefore holds
/// everything fixed except the thing this crate is responsible for (the
/// accept/reject decision and the recurrent-state rollback), so a mismatch
/// here is a bug in that logic and cannot be anything else.
///
/// Comparing against `generate` instead would be a weaker and misleading
/// gate: it folds in the decode-tape/chunk-tape difference that
/// `an_empty_draft_isolates_the_tape_from_the_accept_reject_logic` shows is
/// present with zero speculation involved, so it would fail for a reason
/// speculative decoding did not cause and could not fix.
///
/// The drafter is the n-gram one, which on this prompt is wrong most of the
/// time - measured 9 of 17 proposals accepted over 16 rounds, so most rounds
/// take the reject-and-roll-back path this exists to gate rather than the
/// all-accepted fast path that never exercises it.
#[test]
fn speculative_decoding_is_exactly_equivalent_to_the_same_tape_without_speculation() {
    let Some(inst) = load() else { return };
    let prompt = inst.tokenize("The capital city of France is");
    let max_new = 24u32;

    let (baseline, base_stats) = speculative(&inst, &prompt, max_new, 7, &mut |_c: &[u32], _w: u32| Vec::new());
    let (spec, stats) = speculative(&inst, &prompt, max_new, 7, &mut ngram_draft);

    println!("\nsame-tape equivalence, {max_new} tokens, greedy:");
    report("no speculation", &baseline, Some(&base_stats), baseline.tok_s);
    report("n-gram speculation", &spec, Some(&stats), baseline.tok_s);
    println!("  text: {:?}", spec.text);

    assert!(stats.proposed > 0, "this gate needs a drafter that actually proposes something");
    assert!(
        stats.accepted < stats.proposed,
        "this gate needs real REJECTIONS to exercise the rollback, but all {}/{} proposals were accepted",
        stats.accepted,
        stats.proposed
    );
    assert_eq!(spec.ids, baseline.ids, "speculation changed the output on a fixed tape - the accept/reject logic or the recurrent-state rollback is wrong");
    assert_eq!(spec.text, baseline.text, "speculation changed the decoded text on a fixed tape");
}

/// **The quality gate against the plain path.** Both paths must answer the
/// question correctly; they are NOT required to answer it with the same
/// tokens, for the tape reason the two tests above establish and measure.
///
/// This is the weaker claim that survives that finding, and it is still worth
/// gating: a speculative path that had, say, an off-by-one in its positions
/// would produce fluent text that stopped answering the prompt, and nothing
/// short of an assertion on the ANSWER catches that. "Finite and
/// non-degenerate" is not a correctness check; a fact the model cannot get
/// wrong, asked through the plainest possible request, is.
#[test]
fn speculative_decode_still_answers_the_factual_prompt() {
    let Some(inst) = load() else { return };
    let prompt = inst.tokenize("The capital city of France is");
    let max_new = 24u32;

    let base = plain(&inst, &prompt, max_new);
    println!("\nfactual continuation, {max_new} tokens, greedy:");
    report("plain decode", &base, None, base.tok_s);
    println!("  plain text : {:?}", base.text);

    let (spec, stats) = speculative(&inst, &prompt, max_new, 7, &mut ngram_draft);
    report("speculative (n-gram)", &spec, Some(&stats), base.tok_s);
    println!("  spec  text : {:?}", spec.text);

    assert!(base.text.contains("Paris"), "the plain path must continue this prompt with Paris, got {:?}", base.text);
    assert!(spec.text.contains("Paris"), "the speculative path must continue this prompt with Paris, got {:?}", spec.text);
}

/// **The ceiling, and what the drafter would have to be worth.** Same prompt,
/// three drafters on one load: plain, an n-gram drafter, and an ORACLE fed the
/// continuation the plain path already produced.
///
/// The oracle number is the useful one. It is not achievable by any real
/// drafter, but it bounds every drafter from above on THIS model and THIS
/// hardware, and it is measured rather than argued - which is the only way to
/// know whether a draft model is worth porting before porting it. All three
/// must produce identical tokens; a speedup that changed the output would be
/// meaningless.
#[test]
fn speculative_decode_speedup_ladder() {
    let Some(inst) = load() else { return };
    let prompt = inst.tokenize("The capital city of France is");
    let max_new = 32u32;

    let base = plain(&inst, &prompt, max_new);
    println!("\nspeedup ladder, {max_new} tokens, greedy, one load:");
    report("plain decode", &base, None, base.tok_s);

    // The same-tape floor: the speculative loop with nothing to speculate on.
    // Every speedup below is quoted against PLAIN decode (what a user would
    // actually have had), but correctness is checked against this, for the
    // reason `speculative_decoding_is_exactly_equivalent_to_the_same_tape_
    // without_speculation` explains.
    let (floor, floor_stats) = speculative(&inst, &prompt, max_new, 7, &mut |_c: &[u32], _w: u32| Vec::new());
    report("chunk tape, no spec", &floor, Some(&floor_stats), base.tok_s);

    // Oracle: propose exactly what that same tape goes on to produce. `ctx`
    // starts as the prompt, so the continuation index is `ctx.len() - prompt.len()`.
    let full: Vec<u32> = prompt.iter().copied().chain(floor.ids.iter().copied()).collect();
    for k in [3u32, 7] {
        let (run, stats) = speculative(&inst, &prompt, max_new, k, &mut |ctx: &[u32], want: u32| {
            (0..want as usize).filter_map(|i| full.get(ctx.len() + i).copied()).collect()
        });
        report(&format!("oracle draft (k={k})"), &run, Some(&stats), base.tok_s);
        assert_eq!(run.ids, floor.ids, "oracle-draft speculative decode changed the output at k={k}");
    }

    let (ng, stats) = speculative(&inst, &prompt, max_new, 7, &mut ngram_draft);
    report("n-gram draft (k=7)", &ng, Some(&stats), base.tok_s);
    assert_eq!(ng.ids, floor.ids, "n-gram speculative decode changed the output");
    println!(
        "  -> the ceiling above is what a PERFECT drafter buys on this hardware. \
         It is bounded by two costs speculation does not remove: a one-row verify chunk is \
         slower than a one-row decode step (the `chunk tape, no spec` row), and a rejecting \
         round pays a second forward to re-commit."
    );
}

/// **A workload where a model-free drafter actually pays.** Quoting a passage
/// back is the canonical case for n-gram drafting (as are code edits and
/// structured output): the tokens to be generated have literally occurred in
/// the context, so the drafter is right for long stretches and the round
/// accepts its whole window.
///
/// Measured separately from the factual prompt above because the two are
/// different claims. That one says the mechanism is lossless where the
/// drafter is useless; this one says it is fast where the drafter is good -
/// and quoting either number as "the" speedup, without the workload, would be
/// meaningless.
#[test]
fn speculative_decode_throughput_on_a_repetition_workload() {
    let Some(inst) = load() else { return };
    let passage = "The Gated DeltaNet recurrence maintains a matrix-valued state that is \
updated by a delta rule at every token, and decayed by a per-head gate. Because the state \
is a single matrix rather than a growing cache, its memory cost does not depend on the \
sequence length, which is what makes the hybrid layer stack cheaper to serve at long \
context than a pure attention stack would be.";
    let prompt = inst.tokenize(&format!("{passage}\n\nRepeat the paragraph above exactly, word for word:\n\n"));
    let max_new = 48u32;
    println!("\nrepetition workload: {}-token prompt, {max_new} tokens, greedy:", prompt.len());

    let base = plain(&inst, &prompt, max_new);
    report("plain decode", &base, None, base.tok_s);
    println!("  plain text : {:?}", base.text);

    let (floor, floor_stats) = speculative(&inst, &prompt, max_new, 7, &mut |_c: &[u32], _w: u32| Vec::new());
    report("chunk tape, no spec", &floor, Some(&floor_stats), base.tok_s);

    let (spec, stats) = speculative(&inst, &prompt, max_new, 7, &mut ngram_draft);
    report("speculative (n-gram)", &spec, Some(&stats), base.tok_s);

    assert_eq!(spec.ids, floor.ids, "speculative decoding changed the output on the repetition workload");
    println!(
        "  -> {:.2} accepted draft tokens per round; break-even for this model is 2.0 \
         (a rejecting round pays a second target forward to re-commit, because its recurrent layers cannot be truncated)",
        stats.accepted_per_round()
    );
}
