// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching LLM serving for clients
// whose agents resend long conversation histories. If your team needs
// expertise in LLM serving throughput then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The conformance suite every [`PagedDecoder`] registered for `brain serve`
//! must pass. The scheduler ([`Scheduler`]) owns continuous batching, fair
//! admission, chunked prefill and the order of work; the decoder only
//! implements the contract documented on [`PagedDecoder`]. A model that
//! follows the contract's rules gets these behaviours with no model-specific
//! scheduling code; this suite is what proves it did.
//!
//! The suite is generic over the decoder and runs against tiny random-weight
//! engines, so it fits CI. Each engine crate calls [`run_all`] from a test with
//! its own factory (`scripts/gates` fails a registered decoder with no call).
//!
//! The properties (all with greedy decoding, so outputs are comparable):
//!
//! * [`batched_output_equals_single_request_output`]: sequences decoded
//!   together produce what each produces alone.
//! * [`a_short_request_is_not_held_behind_a_long_decode`]: a short request
//!   admitted while a long sequence decodes finishes in about its own number
//!   of steps, and the long one is still running (no head-of-line blocking).
//! * [`a_prefix_hit_prefills_only_the_suffix`]: a prompt extending a served
//!   one reuses it (counted by `prefix_stats`) and answers exactly as a cold
//!   prefill would; a decoder that declares no prefix reuse reports none.
//! * [`adapters_never_share_kv_with_the_base`]: K/V computed under an adapter
//!   is never served to the base or to another adapter, in either direction.
//!
//! * [`embedded_prompts_batch_and_never_share_a_prefix`]: a decoder that takes
//!   embedding rows in a prompt ([`PagedDecoder::embed_width`]) decodes
//!   batched embedded requests as it does each alone, and never serves a cached
//!   prefix across differing rows (nor rows' blocks to a token-only prompt with
//!   the same token ids); a decoder that takes tokens only refuses an embedded
//!   request cleanly and keeps serving.
//! * [`speculation_equals_plain_greedy`]: a decoder that declares speculation
//!   ([`PagedDecoder::spec_max_draft`] above 0) decodes, under a drafter that
//!   is right, half right and always wrong, exactly the tokens plain greedy
//!   decode produces, with both accepted and rejected proposals exercised. A
//!   decoder that does not speculate passes it trivially (nothing to verify);
//!   once it does, the property applies with no change to its factory.
//!
//!   Bit identity: the property compares token streams exactly, so a verify
//!   kernel whose floating-point order differs from plain decode may flip an
//!   argmax that is a near tie. The contract is that verify rows are computed
//!   to the same numerics as plain decode for the same position (the
//!   decoder's `forward_batched_verify` documents any exception, and its
//!   tiny-engine factory then builds weights with a clear top-1 margin so the
//!   suite does not depend on a tie). Tolerance is not added here: a
//!   tolerant comparison would hide a verify that really diverges.
//!
//! A server that receives SIGTERM with nothing in flight exits within a bound:
//! that lives in the process layer (`brain_shutdown::bound_drain`), not the
//! decoder, and has its own test there.

use super::{AdapterId, DraftQuery, Drafter, PagedDecoder, PromptEmbeds, RejectReason, Request, SampleParams, Scheduler, SpecPolicy};

/// How a decoder is brought to the suite.
pub struct Conformance<D: PagedDecoder> {
    /// A fresh decoder with an empty KV pool and prefix cache. It must hold
    /// at least [`MIN_SEQ_TOKENS`] tokens per sequence and 4 sequences.
    pub make: fn() -> D,
    /// Whether the decoder reuses KV prefixes (`prefix_stats` then counts hits).
    pub prefix_reuse: bool,
    /// The prompt length the prefix test serves first, then extends by 8
    /// tokens: an engine that caches only prompts of some minimum length needs
    /// a longer one. The decoder must hold this plus 12 tokens per sequence.
    pub prefix_prompt_tokens: usize,
    /// Loads one distinct adapter into a decoder and returns its id; `None`
    /// for a decoder without an adapter bank.
    pub load_adapter: Option<fn(&mut D) -> AdapterId>,
}

/// The per-sequence capacity (prompt + generated) a suite decoder must have
/// (more for a `prefix_prompt_tokens` above 36).
pub const MIN_SEQ_TOKENS: usize = 48;

fn tokens(vocab: usize, seed: usize, n: usize) -> Vec<u32> {
    let span = vocab.saturating_sub(2).clamp(1, 61);
    (0..n).map(|i| 1 + ((seed * 7 + i * 13 + i * i) % span) as u32).collect()
}

fn request(prompt: Vec<u32>, max_new: usize) -> Request {
    Request { prompt, max_new, stop: Vec::new() }
}

fn complete<D: PagedDecoder>(sched: &mut Scheduler<D>, id: u64) -> Vec<u32> {
    let mut out = None;
    while sched.pending() {
        for (done, toks) in sched.step() {
            if done == id {
                out = Some(toks);
            }
        }
    }
    out.expect("the request completed")
}

/// Greedy output of one request alone on a fresh decoder.
fn alone<D: PagedDecoder>(c: &Conformance<D>, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let mut sched = Scheduler::new((c.make)(), 4);
    let id = sched.submit(request(prompt.to_vec(), max_new));
    complete(&mut sched, id)
}

pub fn run_all<D: PagedDecoder>(c: &Conformance<D>) {
    batched_output_equals_single_request_output(c);
    a_short_request_is_not_held_behind_a_long_decode(c);
    a_prefix_hit_prefills_only_the_suffix(c);
    if c.load_adapter.is_some() {
        adapters_never_share_kv_with_the_base(c);
    }
    embedded_prompts_batch_and_never_share_a_prefix(c);
    speculation_equals_plain_greedy(c);
}

pub fn batched_output_equals_single_request_output<D: PagedDecoder>(c: &Conformance<D>) {
    let mut sched = Scheduler::new((c.make)(), 4);
    let vocab = sched.decoder_mut().vocab();
    let prompts: Vec<Vec<u32>> = [5usize, 9, 17, 25].iter().enumerate().map(|(i, &n)| tokens(vocab, i + 1, n)).collect();
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(request(p.clone(), 6))).collect();
    let together = sched.run();
    for (p, id) in prompts.iter().zip(ids) {
        assert_eq!(together[&id], alone(c, p, 6), "a batched request must decode exactly as it does alone (prompt of {} tokens)", p.len());
    }
}

pub fn a_short_request_is_not_held_behind_a_long_decode<D: PagedDecoder>(c: &Conformance<D>) {
    held_behind_check(c, false);
    if c.load_adapter.is_some() {
        held_behind_check(c, true);
    }
}

/// With `adapters`, the long request runs under one adapter and the shorts
/// under the base and another adapter: a mixed batch must not wait either.
fn held_behind_check<D: PagedDecoder>(c: &Conformance<D>, adapters: bool) {
    let mut sched = Scheduler::new((c.make)(), 4);
    let vocab = sched.decoder_mut().vocab();
    let loaded: Vec<AdapterId> = match (adapters, c.load_adapter) {
        (true, Some(load)) => vec![load(sched.decoder_mut()), load(sched.decoder_mut())],
        _ => Vec::new(),
    };
    let submit = |sched: &mut Scheduler<D>, req: Request, which: usize| match loaded.get(which) {
        Some(&a) => sched.submit_for_adapter(req, a, SampleParams::greedy(), 0),
        None => sched.submit(req),
    };
    let long = submit(&mut sched, request(tokens(vocab, 1, 16), 24), 0);
    for _ in 0..3 {
        assert!(sched.step().is_empty(), "the long request must still be running");
    }
    // One short request per kind the long one is not: the base and the other adapter.
    let short_new = 3usize;
    let shorts: Vec<u64> = (0..2).map(|i| submit(&mut sched, request(tokens(vocab, 2 + i, 6), short_new), if i == 0 { usize::MAX } else { 1 })).collect();
    let mut left: Vec<u64> = shorts;
    let mut steps = 0;
    while !left.is_empty() {
        steps += 1;
        for (id, toks) in sched.step() {
            assert_ne!(id, long, "the long request finished before the short ones: it was not decoding alongside (adapters: {adapters})");
            assert_eq!(toks.len(), short_new);
            left.retain(|s| *s != id);
        }
        assert!(steps <= 2 * short_new + 8, "short requests waited {steps} steps behind the long decode (adapters: {adapters})");
    }
    assert!(sched.pending(), "the long request is still decoding when the short ones are done (adapters: {adapters})");
}

pub fn a_prefix_hit_prefills_only_the_suffix<D: PagedDecoder>(c: &Conformance<D>) {
    let mut sched = Scheduler::new((c.make)(), 4);
    let vocab = sched.decoder_mut().vocab();
    let first = tokens(vocab, 3, c.prefix_prompt_tokens);
    let mut second = first.clone();
    second.extend(tokens(vocab, 4, 8));
    let id = sched.submit(request(first.clone(), 4));
    complete(&mut sched, id);
    let (hit_before, _, _) = sched.prefix_stats();
    let id = sched.submit(request(second.clone(), 4));
    let warm = complete(&mut sched, id);
    let gained = sched.prefix_stats().0 - hit_before;
    if c.prefix_reuse {
        assert!(gained as usize >= first.len() / 2, "a prompt extending a served one reused only {gained} of {} cached tokens", first.len());
        assert!(gained as usize <= first.len(), "reuse cannot exceed the cached prompt ({gained} tokens)");
    } else {
        assert_eq!(gained, 0, "a decoder declaring no prefix reuse reported hits");
    }
    assert_eq!(warm, alone(c, &second, 4), "output after a prefix hit must equal a cold prefill's");
}

pub fn adapters_never_share_kv_with_the_base<D: PagedDecoder>(c: &Conformance<D>) {
    let load = c.load_adapter.expect("an adapter loader");
    let mut sched = Scheduler::new((c.make)(), 4);
    let adapter = load(sched.decoder_mut());
    let other = load(sched.decoder_mut());
    let vocab = sched.decoder_mut().vocab();
    let prompt = tokens(vocab, 5, c.prefix_prompt_tokens);
    let mut extended = prompt.clone();
    extended.extend(tokens(vocab, 6, 8));
    let hits = |sched: &Scheduler<D>| sched.prefix_stats().0;
    let base = sched.submit(request(prompt.clone(), 4));
    let base_out = complete(&mut sched, base);
    let before = hits(&sched);
    let id = sched.submit_for_adapter(request(extended.clone(), 4), adapter, SampleParams::greedy(), 0);
    complete(&mut sched, id);
    assert_eq!(hits(&sched), before, "an adapter request adopted K/V the base computed");
    let id = sched.submit_for_adapter(request(extended.clone(), 4), other, SampleParams::greedy(), 0);
    complete(&mut sched, id);
    assert_eq!(hits(&sched), before, "an adapter request adopted K/V another adapter computed");
    let id = sched.submit(request(extended, 4));
    complete(&mut sched, id);
    if c.prefix_reuse {
        assert!(hits(&sched) > before, "the base lost its own cached prefix to the adapters' requests");
    }
    let again = sched.submit(request(prompt, 4));
    assert_eq!(complete(&mut sched, again), base_out, "the base's answer changed after adapter requests ran");
}

/// Proposes the plain-greedy continuation of the prompt it recognises for its
/// first `right` tokens, then deliberately wrong ones (the token after the
/// true one), so a verify sees full accepts, partial accepts and rejects.
struct ScriptedDrafter {
    /// `(prompt, plain greedy output, right)` per sequence.
    scripts: Vec<(Vec<u32>, Vec<u32>, usize)>,
    vocab: u32,
}

impl Drafter for ScriptedDrafter {
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        queries
            .iter()
            .zip(ks)
            .map(|(q, &k)| {
                let Some((prompt, plain, right)) = self.scripts.iter().find(|(p, plain, _)| q.context.starts_with(p) && plain.starts_with(&q.context[p.len()..])) else {
                    return Vec::new();
                };
                let generated = q.context.len() - prompt.len();
                plain.iter().skip(generated).take(k).enumerate().map(|(i, &t)| if i < *right { t } else { (t + 1) % self.vocab }).collect()
            })
            .collect()
    }
}

/// Speculation never changes what greedy decoding emits. Repetitive prompts
/// run together under three drafters (all right, one right then wrong, all
/// wrong); every output must equal the plain greedy output of the same
/// request alone, and the run must really have accepted and rejected proposals.
pub fn speculation_equals_plain_greedy<D: PagedDecoder>(c: &Conformance<D>) {
    let mut sched = Scheduler::new((c.make)(), 4);
    if sched.decoder_mut().spec_max_draft() == 0 {
        return;
    }
    let vocab = sched.decoder_mut().vocab();
    // The scheduler speculates only up to the decoder's crossover batch size.
    let seqs = sched.decoder_mut().spec_max_batch().clamp(1, 4);
    let rights = [usize::MAX, 1, 0, usize::MAX];
    let max_new = 14;
    let scripts: Vec<(Vec<u32>, Vec<u32>, usize)> = (0..seqs)
        .map(|i| {
            let period = tokens(vocab, i + 11, 3 + i);
            let prompt: Vec<u32> = period.iter().cycle().take(10 + 4 * i).copied().collect();
            let plain = plain_greedy(c, &prompt, max_new);
            (prompt, plain, rights[i])
        })
        .collect();
    sched.set_drafter(Box::new(ScriptedDrafter { scripts: scripts.clone(), vocab: vocab as u32 }));
    let ids: Vec<u64> = scripts.iter().map(|(p, _, _)| sched.submit(request(p.clone(), max_new))).collect();
    let done = sched.run();
    for ((prompt, plain, right), id) in scripts.iter().zip(ids) {
        assert_eq!(done[&id], *plain, "speculative output differs from plain greedy (prompt of {} tokens, drafter right for {right} tokens)", prompt.len());
    }
    let stats = sched.spec_stats();
    assert!(stats.verify_steps > 0 && stats.proposed > 0, "the decoder declares speculation but no step verified a proposal: {stats:?}");
    assert!(stats.accepted > 0, "no proposal was ever accepted, so accept handling was not exercised: {stats:?}");
    if seqs > 1 {
        assert!(stats.accepted < stats.proposed, "no proposal was ever rejected, so rollback was not exercised: {stats:?}");
    }
}

/// Greedy output of one request alone with speculation off.
fn plain_greedy<D: PagedDecoder>(c: &Conformance<D>, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let mut sched = Scheduler::new((c.make)(), 4);
    sched.set_speculation(SpecPolicy { enabled: false, ..SpecPolicy::default() });
    let id = sched.submit(request(prompt.to_vec(), max_new));
    complete(&mut sched, id)
}

/// Rows for the embedded properties: `n` rows of `width`, deterministic per
/// `seed`, large enough to move any decoder's output.
fn rows(width: usize, n: usize, seed: usize) -> Vec<f32> {
    (0..n * width).map(|i| (((i * 37 + seed * 101 + (i / width) * 13) % 29) as f32 / 29.0 - 0.5) * 6.0).collect()
}

/// Greedy output of one embedded request alone on a fresh decoder.
fn alone_embedded<D: PagedDecoder>(c: &Conformance<D>, prompt: &[u32], e: &PromptEmbeds, max_new: usize) -> Vec<u32> {
    let mut sched = Scheduler::new((c.make)(), 4);
    let id = sched.submit_embedded(request(prompt.to_vec(), max_new), e.clone(), SampleParams::greedy(), 0);
    complete(&mut sched, id)
}

pub fn embedded_prompts_batch_and_never_share_a_prefix<D: PagedDecoder>(c: &Conformance<D>) {
    let mut sched = Scheduler::new((c.make)(), 4);
    let vocab = sched.decoder_mut().vocab();
    let Some(width) = sched.decoder_mut().embed_width() else {
        // Tokens only: an embedded request is refused cleanly and a plain one still runs.
        let e = PromptEmbeds::run(4, 1, rows(4, 3, 1)).expect("rows");
        let refused = sched.submit_embedded(request(tokens(vocab, 1, 12), 3), e, SampleParams::greedy(), 0);
        let plain = sched.submit(request(tokens(vocab, 2, 8), 3));
        let report = sched.step_report();
        assert!(matches!(report.rejected.as_slice(), [(id, RejectReason::InvalidEmbeds { .. })] if *id == refused), "a decoder without rows must refuse an embedded request: {:?}", report.rejected);
        let mut done: std::collections::HashMap<u64, Vec<u32>> = report.completed.into_iter().collect();
        done.extend(sched.run());
        assert_eq!(done[&plain].len(), 3, "the token-only request must be unaffected by the refusal");
        return;
    };
    let len = c.prefix_prompt_tokens.max(24);
    let prompt = tokens(vocab, 7, len);
    let run = |seed: usize| PromptEmbeds::run(width, 4, rows(width, 12, seed)).expect("rows");
    let (ea, eb) = (run(1), run(2));
    let (out_a, out_b) = (alone_embedded(c, &prompt, &ea, 6), alone_embedded(c, &prompt, &eb, 6));
    let out_tokens = alone(c, &prompt, 6);
    assert!(out_a != out_b && out_a != out_tokens, "the rows must decide the output (decoder ignores them?): {out_a:?} {out_b:?} {out_tokens:?}");

    // Batched together, each decodes as it does alone.
    let mut batch = Scheduler::new((c.make)(), 4);
    let ids = [ea.clone(), eb.clone()].map(|e| batch.submit_embedded(request(prompt.clone(), 6), e, SampleParams::greedy(), 0));
    let done = batch.run();
    assert_eq!(done[&ids[0]], out_a, "a batched embedded request must decode exactly as it does alone");
    assert_eq!(done[&ids[1]], out_b, "a batched embedded request must decode exactly as it does alone");

    // One scheduler, one after another: the cache of the first never answers the second.
    let mut seq = Scheduler::new((c.make)(), 4);
    let first = seq.submit_embedded(request(prompt.clone(), 6), ea.clone(), SampleParams::greedy(), 0);
    assert_eq!(complete(&mut seq, first), out_a);
    let second = seq.submit_embedded(request(prompt.clone(), 6), eb.clone(), SampleParams::greedy(), 0);
    assert_eq!(complete(&mut seq, second), out_b, "a prompt with the same tokens and different rows adopted another prompt's cached K/V");
    let plain = seq.submit(request(prompt.clone(), 6));
    assert_eq!(complete(&mut seq, plain), out_tokens, "a token-only prompt adopted K/V computed from embedding rows");
    let again = seq.submit_embedded(request(prompt, 6), ea, SampleParams::greedy(), 0);
    assert_eq!(complete(&mut seq, again), out_a, "the same rows must still decode the same after other prompts ran");
}
