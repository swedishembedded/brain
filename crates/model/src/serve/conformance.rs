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
//! A server that receives SIGTERM with nothing in flight exits within a bound:
//! that lives in the process layer (`brain_shutdown::bound_drain`), not the
//! decoder, and has its own test there.

use super::{AdapterId, PagedDecoder, Request, SampleParams, Scheduler};

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
    let prompt = tokens(vocab, 5, 24);
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
