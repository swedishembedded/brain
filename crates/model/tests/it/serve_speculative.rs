// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in LLM serving
// throughput then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec of the scheduler's speculative-decoding seam, against a toy RECURRENT
//! decoder with no device. Its "model" folds every consumed token into one
//! state word (like a Gated-DeltaNet layer: nothing earlier can be recovered
//! from it), so a verifier that does not roll that state back after a
//! rejection produces different tokens, and these tests see it.
//!
//! What the seam promises, and what is checked here:
//!
//! * greedy output with speculation equals greedy output without it, token for
//!   token, whatever the drafter proposes (right, wrong, half right, absent);
//! * the recurrent state and the block table of a sequence after a
//!   speculative step are exactly those of plain decode at the same length;
//! * a good drafter really saves decoder steps and a bad one is turned off
//!   per sequence (adaptive window), a big batch turns speculation off, and a
//!   decoder that does not verify is untouched;
//! * sampled speculation leaves the output distribution unchanged
//!   (rejection sampling), checked against the closed form and end to end.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::conformance::{self, Conformance};
use model::serve::{verify_proposals, AdapterAvailability, AdapterId, DraftQuery, Drafter, NgramDrafter, PagedDecoder, Request, SampleParams, Scheduler, SpecPolicy};

const VOCAB: u32 = 32;
const BLOCK: u32 = 4;
const MAX_SEQ: usize = 96;

fn mix(s: u32, t: u32) -> u32 {
    (s.wrapping_mul(2_654_435_761) ^ (t + 1).wrapping_mul(40_503)).rotate_left(5) & 0xF_FFFF
}

fn fold(tokens: &[u32]) -> u32 {
    tokens.iter().fold(7, |s, &t| mix(s, t))
}

/// The toy model's next-token logits from a state: two clearly separated
/// leaders (so top-2 sampling has something to sample), the rest below.
fn logits_of(s: u32) -> Vec<f32> {
    let a = mix(s, 1000) % VOCAB;
    let mut b = mix(s, 2000) % VOCAB;
    if b == a {
        b = (b + 1) % VOCAB;
    }
    (0..VOCAB)
        .map(|i| match i {
            i if i == a => 3.0,
            i if i == b => 2.0,
            i => (mix(s, i) % 1000) as f32 / 1000.0,
        })
        .collect()
}

fn top_of(s: u32, cands: usize) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = logits_of(s).into_iter().enumerate().map(|(i, l)| (i as u32, l)).collect();
    v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    v.truncate(cands);
    v
}

fn greedy_next(s: u32) -> u32 {
    top_of(s, 1)[0].0
}

/// Greedy continuation of `context` (the last token not yet consumed).
fn continuation(context: &[u32], n: usize) -> Vec<u32> {
    let mut s = fold(&context[..context.len() - 1]);
    s = mix(s, *context.last().unwrap());
    let mut out = Vec::new();
    for _ in 0..n {
        let t = greedy_next(s);
        out.push(t);
        s = mix(s, t);
    }
    out
}

#[derive(Default)]
struct Probe {
    plain_calls: usize,
    verify_calls: usize,
    verify_rows: Vec<Vec<usize>>,
    commits: usize,
    /// For every prompt prefilled: whether the toy drafter was prepared for it.
    prefill_feeds: Vec<bool>,
}

type Shared = Rc<RefCell<Probe>>;

/// `verifies`: whether the decoder opts into speculation at all.
struct Toy {
    alloc: BlockAllocator,
    state: HashMap<u32, u32>,
    pending: HashMap<u32, Vec<u32>>,
    probe: Shared,
    verifies: bool,
    batch_limit: usize,
    row_budget: usize,
    /// Virtual time: what a step costs, in microseconds, on the clock the scheduler is given.
    clock: Arc<AtomicU64>,
    plain_us: u64,
    row_us: u64,
    cost_aware: bool,
    /// What preparing the drafter for one prompt costs, in microseconds of the virtual clock.
    feed_us: u64,
    spec_prefill: bool,
    feed_secs: Option<f64>,
    /// Whether `adapter_availability` answers `Resident` (the toy applies no adapter itself).
    serves_adapters: bool,
    /// A deliberately wrong commit: it keeps the state after the LAST verified row whatever was accepted.
    broken_commit: bool,
}

impl Toy {
    fn new(probe: Shared, verifies: bool) -> Toy {
        Toy { alloc: BlockAllocator::new(512, BLOCK), state: HashMap::new(), pending: HashMap::new(), probe, verifies, batch_limit: usize::MAX, row_budget: usize::MAX, clock: Arc::new(AtomicU64::new(0)), plain_us: 0, row_us: 0, cost_aware: false, feed_us: 0, spec_prefill: true, feed_secs: None, serves_adapters: false, broken_commit: false }
    }
    fn slot(t: &BlockTable) -> u32 {
        t.blocks()[0]
    }
}

impl PagedDecoder for Toy {
    fn adapter_availability(&mut self, _adapter: AdapterId) -> AdapterAvailability {
        if self.serves_adapters {
            AdapterAvailability::Resident
        } else {
            AdapterAvailability::Unknown
        }
    }
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        1024
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        MAX_SEQ
    }
    fn vocab(&self) -> usize {
        VOCAB as usize
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(&b) = t.blocks().first() {
            self.state.remove(&b);
        }
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        let s = fold(prompt);
        self.state.insert(Self::slot(table), s);
        if self.feed_us > 0 {
            if self.spec_prefill {
                self.clock.fetch_add(self.feed_us, Ordering::Relaxed);
                self.feed_secs = Some(self.feed_us as f64 * 1e-6);
            }
            self.probe.borrow_mut().prefill_feeds.push(self.spec_prefill);
        }
        vec![s as f32]
    }
    fn spec_prefill_cost(&self) -> Option<f64> {
        self.feed_secs
    }
    fn set_spec_prefill(&mut self, on: bool) {
        self.spec_prefill = on;
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        logits_of(hidden[0] as u32)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.probe.borrow_mut().plain_calls += 1;
        self.clock.fetch_add(self.plain_us, Ordering::Relaxed);
        let mut out = Vec::new();
        for (t, &i) in tables.iter_mut().zip(inputs) {
            t.append(&mut self.alloc).unwrap();
            let s = mix(self.state[&Self::slot(t)], i);
            self.state.insert(Self::slot(t), s);
            out.push(greedy_next(s));
        }
        out
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        512 * BLOCK as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        self.probe.borrow_mut().plain_calls += 1;
        let mut out = Vec::new();
        for (t, &i) in tables.iter_mut().zip(inputs) {
            t.append(&mut self.alloc).unwrap();
            let s = mix(self.state[&Self::slot(t)], i);
            self.state.insert(Self::slot(t), s);
            out.push(top_of(s, k));
        }
        out
    }
    fn topk_capacity(&self) -> usize {
        4
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        top_of(hidden[0] as u32, k)
    }

    fn spec_max_draft(&self) -> usize {
        if self.verifies {
            6
        } else {
            0
        }
    }
    fn spec_max_batch(&self) -> usize {
        self.batch_limit
    }
    fn spec_row_budget(&self) -> usize {
        self.row_budget
    }
    fn spec_cost_aware(&self) -> bool {
        self.cost_aware
    }
    fn forward_batched_verify(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize) -> Vec<Vec<Vec<(u32, f32)>>> {
        assert!(self.verifies, "the scheduler verified on a decoder that declared no speculation");
        self.clock.fetch_add(self.plain_us + self.row_us * rows.iter().map(Vec::len).sum::<usize>() as u64, Ordering::Relaxed);
        let mut p = self.probe.borrow_mut();
        p.verify_calls += 1;
        p.verify_rows.push(rows.iter().map(|r| r.len()).collect());
        drop(p);
        let mut out = Vec::new();
        for (t, r) in tables.iter_mut().zip(rows) {
            let slot = Self::slot(t);
            let mut s = self.state[&slot];
            let mut states = Vec::new();
            let mut cs = Vec::new();
            for &tok in r {
                t.append(&mut self.alloc).unwrap();
                s = mix(s, tok);
                states.push(s);
                cs.push(top_of(s, cands));
            }
            self.pending.insert(slot, states);
            out.push(cs);
        }
        out
    }
    fn commit_verified(&mut self, tables: &mut [&mut BlockTable], keep: &[usize]) {
        self.probe.borrow_mut().commits += 1;
        for (t, &k) in tables.iter().zip(keep) {
            let slot = Self::slot(t);
            let states = self.pending.remove(&slot).expect("commit without a verify");
            let s = if self.broken_commit { *states.last().unwrap() } else { states[k - 1] };
            self.state.insert(slot, s);
        }
    }
}

/// Proposes the true greedy continuation for its first `right` tokens, then
/// deliberately wrong ones.
struct Scripted {
    right: usize,
}
impl Drafter for Scripted {
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        queries
            .iter()
            .zip(ks)
            .map(|(q, &k)| {
                let truth = continuation(q.context, k);
                truth.iter().enumerate().map(|(i, &t)| if i < self.right { t } else { (t + 1) % VOCAB }).collect()
            })
            .collect()
    }
}

fn req(seed: u32, len: usize, max_new: usize) -> Request {
    Request { prompt: (0..len as u32).map(|i| (seed * 7 + i * 3) % VOCAB).collect(), max_new, stop: Vec::new() }
}

fn run(mut sched: Scheduler<Toy>, reqs: Vec<Request>) -> (Vec<Vec<u32>>, Scheduler<Toy>) {
    let ids: Vec<u64> = reqs.into_iter().map(|r| sched.submit(r)).collect();
    let done = sched.run();
    (ids.iter().map(|i| done[i].clone()).collect(), sched)
}

fn plain(reqs: Vec<Request>) -> Vec<Vec<u32>> {
    run(Scheduler::new(Toy::new(Shared::default(), true), 8), reqs).0
}

fn speculative(drafter: impl Drafter + 'static, reqs: Vec<Request>) -> (Vec<Vec<u32>>, Shared) {
    let probe = Shared::default();
    let mut sched = Scheduler::new(Toy::new(probe.clone(), true), 8);
    sched.set_drafter(Box::new(drafter));
    (run(sched, reqs).0, probe)
}

fn batch() -> Vec<Request> {
    vec![req(1, 5, 40), req(2, 9, 33), req(3, 3, 1), req(4, 12, 50), req(5, 6, 17)]
}

#[test]
fn greedy_output_is_identical_to_plain_decode_for_every_kind_of_drafter() {
    let want = plain(batch());
    assert!(want.iter().all(|t| !t.is_empty()));
    for right in [0usize, 1, 2, 3, 99] {
        let (got, probe) = speculative(Scripted { right }, batch());
        assert_eq!(got, want, "a drafter right for {right} tokens changed the output");
        assert!(probe.borrow().verify_calls > 0, "speculation never ran for right={right}");
    }
    let (got, _) = speculative(NgramDrafter::new(2), batch());
    assert_eq!(got, want, "the n-gram drafter changed the output");
}

#[test]
fn a_good_drafter_saves_decoder_steps() {
    let (plain_tokens, probe_plain) = {
        let probe = Shared::default();
        let (t, _) = run(Scheduler::new(Toy::new(probe.clone(), true), 8), batch());
        (t, probe)
    };
    let (tokens, probe) = speculative(Scripted { right: 99 }, batch());
    assert_eq!(tokens, plain_tokens);
    let p = probe.borrow();
    let steps = p.plain_calls + p.verify_calls;
    let plain_steps = probe_plain.borrow().plain_calls;
    assert!(steps * 3 <= plain_steps, "an always-right drafter should cut decoder steps at least 3x: {steps} against {plain_steps}");
    assert_eq!(p.commits, p.verify_calls, "every verify is committed exactly once");
}

#[test]
fn a_stop_token_inside_an_accepted_window_ends_the_sequence_there() {
    let base = req(9, 6, 60);
    let free = plain(vec![Request { prompt: base.prompt.clone(), max_new: 60, stop: vec![] }]).remove(0);
    let stop = free[11];
    let want_len = free.iter().position(|&t| t == stop).unwrap();
    let r = |s: &Request| Request { prompt: s.prompt.clone(), max_new: 60, stop: vec![stop] };
    let want = plain(vec![r(&base)]);
    assert_eq!(want[0].len(), want_len);
    let (got, _) = speculative(Scripted { right: 99 }, vec![r(&base)]);
    assert_eq!(got, want, "tokens after the stop id leaked out of a speculative window");
}

#[test]
fn max_new_is_never_overshot_by_a_window() {
    for max_new in 1..12 {
        let (got, _) = speculative(Scripted { right: 99 }, vec![req(2, 5, max_new)]);
        assert_eq!(got[0].len(), max_new);
    }
}

#[test]
fn rejection_rolls_back_state_and_table_to_plain_decode() {
    let probe = Shared::default();
    let mut sched = Scheduler::new(Toy::new(probe.clone(), true), 8);
    sched.set_drafter(Box::new(Scripted { right: 1 }));
    let a = sched.submit(req(1, 7, 80));
    let b = sched.submit(req(2, 5, 80));
    for _ in 0..9 {
        sched.step();
        for (id, prompt_len) in [(a, 7usize), (b, 5usize)] {
            let toks = sched.tokens_of(id).unwrap().to_vec();
            let prompt = &req(if id == a { 1 } else { 2 }, prompt_len, 1).prompt;
            // Plain decode has consumed the prompt and every generated token
            // except the newest one, which is the next step's input.
            let consumed: Vec<u32> = prompt.iter().chain(&toks[..toks.len() - 1]).copied().collect();
            let (table_len, slot) = {
                let table = sched.table_of(id).expect("running");
                (table.len(), table.blocks()[0])
            };
            assert_eq!(table_len as usize, consumed.len(), "block table length after a speculative step");
            let state = sched.decoder_mut().state[&slot];
            assert_eq!(state, fold(&consumed), "recurrent state after a speculative step");
        }
    }
}

#[test]
fn a_drafter_that_is_never_right_is_turned_off_per_sequence() {
    let probe = Shared::default();
    let mut sched = Scheduler::new(Toy::new(probe.clone(), true), 8);
    sched.set_drafter(Box::new(Scripted { right: 0 }));
    sched.submit(req(1, 5, 80));
    sched.run();
    let p = probe.borrow();
    let late: usize = p.verify_rows.iter().skip(8).map(|r| r.iter().sum::<usize>()).sum();
    let late_calls = p.verify_rows.iter().skip(8).count();
    assert!(
        late_calls == 0 || late <= late_calls * 2,
        "after repeated total rejection the window should collapse (rows per verify call {late}/{late_calls})"
    );
    assert!(p.plain_calls + p.verify_calls < 80 + 8, "speculation must not cost extra steps once it has backed off");
}

#[test]
fn the_window_grows_to_the_decoders_capacity_for_a_drafter_that_is_always_right() {
    let (_, probe) = speculative(Scripted { right: 99 }, vec![req(3, 5, 90)]);
    let widest = probe.borrow().verify_rows.iter().flatten().copied().max().unwrap();
    assert_eq!(widest, 7, "1 input row + spec_max_draft (6) proposals");
}

#[test]
fn a_batch_past_the_decoders_crossover_decodes_plainly() {
    let probe = Shared::default();
    let mut dec = Toy::new(probe.clone(), true);
    dec.batch_limit = 2;
    let mut sched = Scheduler::new(dec, 8);
    sched.set_drafter(Box::new(Scripted { right: 99 }));
    for i in 0..5 {
        sched.submit(req(i, 4, 20));
    }
    let big = sched.run();
    assert_eq!(probe.borrow().verify_calls, 0, "5 running sequences are past a crossover of 2");
    let want = plain((0..5).map(|i| req(i, 4, 20)).collect());
    let mut got: Vec<_> = big.into_iter().collect();
    got.sort();
    assert_eq!(got.into_iter().map(|(_, t)| t).collect::<Vec<_>>(), want);
}

#[test]
fn a_verify_step_never_carries_more_rows_than_the_decoders_budget() {
    let probe = Shared::default();
    let mut dec = Toy::new(probe.clone(), true);
    dec.row_budget = 7;
    let mut sched = Scheduler::new(dec, 8);
    sched.set_drafter(Box::new(Scripted { right: 99 }));
    let ids: Vec<u64> = (0..3).map(|i| sched.submit(req(i, 5, 40))).collect();
    let got = sched.run();
    let p = probe.borrow();
    assert!(p.verify_calls > 0);
    for rows in &p.verify_rows {
        assert!(rows.iter().sum::<usize>() <= 7, "a verify step carried {rows:?}");
    }
    let want = plain((0..3).map(|i| req(i, 5, 40)).collect());
    assert_eq!(ids.iter().map(|i| got[i].clone()).collect::<Vec<_>>(), want);
}

/// A decoder whose steps cost `plain_us` plus `row_us` per verified row, on a virtual clock.
fn timed(plain_us: u64, row_us: u64) -> (Scheduler<Toy>, Shared) {
    let probe = Shared::default();
    let mut dec = Toy::new(probe.clone(), true);
    let clock = dec.clock.clone();
    (dec.plain_us, dec.row_us, dec.cost_aware) = (plain_us, row_us, true);
    let mut sched = Scheduler::new(dec, 8);
    sched.set_clock(Box::new(move || clock.load(Ordering::Relaxed) as f64 * 1e-6));
    sched.set_drafter(Box::new(Scripted { right: 99 }));
    (sched, probe)
}

#[test]
fn speculation_that_pays_on_the_clock_is_used_and_speculation_that_does_not_is_left_to_probes() {
    let want = plain((0..3).map(|i| req(i, 5, 80)).collect());
    for (label, plain_us, row_us, expect_verify_share) in [("cheap rows", 1000u64, 10u64, 0.8), ("rows as dear as a step", 1000, 1000, 0.15)] {
        let (mut sched, probe) = timed(plain_us, row_us);
        let ids: Vec<u64> = (0..3).map(|i| sched.submit(req(i, 5, 80))).collect();
        let got = sched.run();
        assert_eq!(ids.iter().map(|i| got[i].clone()).collect::<Vec<_>>(), want, "{label}: the output changed");
        let p = probe.borrow();
        let share = p.verify_calls as f64 / (p.verify_calls + p.plain_calls) as f64;
        if expect_verify_share > 0.5 {
            assert!(share >= expect_verify_share, "{label}: speculation pays and should carry most steps, verify share {share:.2}");
        } else {
            assert!(share <= expect_verify_share, "{label}: speculation does not pay and should only probe, verify share {share:.2}");
        }
    }
}

/// Requests one after another on a decoder whose drafter costs `feed_us` to prepare per prompt;
/// returns for each prompt whether the drafter was prepared.
fn feeds_of(feed_us: u64, max_new: usize) -> Vec<bool> {
    let (mut sched, probe) = timed(1000, 10);
    sched.decoder_mut().feed_us = feed_us;
    for i in 0..24 {
        sched.submit(req(i, 5, max_new));
        sched.run();
    }
    let feeds = probe.borrow().prefill_feeds.clone();
    feeds
}

/// The drafter's per-prompt preparation is a cost speculation has to repay with what it saves
/// decoding that prompt's reply: when replies are short and the preparation is dear, the
/// scheduler stops paying it (measured, per request); when replies are long it keeps paying.
#[test]
fn the_drafters_prefill_feed_is_skipped_when_speculation_cannot_repay_it() {
    let dear = feeds_of(500_000, 12);
    assert_eq!(dear.len(), 24);
    assert!(dear[0], "the first request learns what the feed and speculation cost: {dear:?}");
    // perf-number: the assertion message names the simulated feed cost the test sets up, not a measurement
    assert!(dear[8..].iter().filter(|&&f| f).count() <= 1, "short replies cannot repay a 0.5 s feed, only a probe may still pay it: {dear:?}");
    let cheap_and_long = feeds_of(500, 80);
    // perf-number: the assertion message names the simulated feed cost the test sets up, not a measurement
    assert!(cheap_and_long.iter().all(|&f| f), "an 80-token reply repays a 0.5 ms feed: {cheap_and_long:?}");
}

#[test]
fn a_decoder_that_does_not_verify_ignores_the_drafter() {
    let probe = Shared::default();
    let mut sched = Scheduler::new(Toy::new(probe.clone(), false), 8);
    sched.set_drafter(Box::new(Scripted { right: 99 }));
    sched.submit(req(1, 5, 20));
    let got = sched.run();
    assert_eq!(probe.borrow().verify_calls, 0);
    assert_eq!(got[&0], plain(vec![req(1, 5, 20)]).remove(0));
}

#[test]
fn speculation_can_be_switched_off() {
    let probe = Shared::default();
    let mut sched = Scheduler::new(Toy::new(probe.clone(), true), 8);
    sched.set_drafter(Box::new(Scripted { right: 99 }));
    sched.set_speculation(SpecPolicy { enabled: false, ..SpecPolicy::default() });
    sched.submit(req(1, 5, 20));
    sched.run();
    assert_eq!(probe.borrow().verify_calls, 0);
}

#[test]
fn a_step_with_an_adapter_row_decodes_plainly_and_base_only_steps_still_speculate() {
    let adapter = AdapterId::new(7).expect("a nonzero namespace is an adapter");
    let toy = |probe: &Shared| {
        let mut dec = Toy::new(probe.clone(), true);
        dec.serves_adapters = true;
        let mut sched = Scheduler::new(dec, 8);
        sched.set_drafter(Box::new(Scripted { right: 99 }));
        sched
    };

    let mixed = Shared::default();
    let mut sched = toy(&mixed);
    let want = plain(vec![req(1, 5, 30), req(2, 9, 30)]);
    let ids = [sched.submit(req(1, 5, 30)), sched.submit_for_adapter(req(2, 9, 30), adapter, SampleParams::greedy(), 0)];
    let done = sched.run();
    assert_eq!(mixed.borrow().verify_calls, 0, "no step carrying an adapter row may be a verify step");
    assert_eq!([done[&ids[0]].clone(), done[&ids[1]].clone()].to_vec(), want, "plain decode of mixed rows changed the output");

    let base_only = Shared::default();
    let mut sched = toy(&base_only);
    sched.submit(req(1, 5, 30));
    sched.run();
    assert!(base_only.borrow().verify_calls > 0, "base-only traffic must keep speculating");
}

#[test]
fn the_ngram_drafter_copies_what_the_context_repeats() {
    let mut d = NgramDrafter::new(2);
    let table = BlockTable::new();
    let ctx = [5u32, 6, 7, 8, 9, 1, 2, 5, 6];
    let q = DraftQuery { id: 0, context: &ctx, table: &table };
    assert_eq!(d.draft(&[q], &[3]), vec![vec![7, 8, 9]]);
    let ctx = [1u32, 2, 3, 4];
    let q = DraftQuery { id: 0, context: &ctx, table: &table };
    assert_eq!(d.draft(&[q], &[3]), vec![Vec::<u32>::new()], "no earlier occurrence, no proposal");
}

fn p_table() -> Vec<(u32, f32)> {
    // Logits whose softmax at temperature 1 is not flat.
    vec![(10, 1.2), (11, 0.7), (12, 0.1), (13, -0.4)]
}

fn softmax(c: &[(u32, f32)]) -> Vec<f64> {
    let z: f64 = c.iter().map(|&(_, l)| (l as f64).exp()).sum();
    c.iter().map(|&(_, l)| (l as f64).exp() / z).collect()
}

#[test]
fn greedy_verification_accepts_the_longest_matching_prefix() {
    let g = SampleParams::greedy();
    let mut rng = data::rng::Rng::new(data::rng::random_seed());
    let rows: Vec<Vec<(u32, f32)>> = vec![vec![(4, 2.0)], vec![(5, 2.0)], vec![(6, 2.0)], vec![(7, 2.0)]];
    let v = verify_proposals(&[4, 5, 9], &rows, g, &mut rng);
    assert_eq!((v.accepted, v.next), (2, 6));
    let v = verify_proposals(&[4, 5, 6], &rows, g, &mut rng);
    assert_eq!((v.accepted, v.next), (3, 7), "every proposal accepted earns the bonus token");
    let v = verify_proposals(&[], &rows[..1], g, &mut rng);
    assert_eq!((v.accepted, v.next), (0, 4));
}

#[test]
fn rejection_sampling_preserves_the_target_distribution() {
    let cands = p_table();
    let p = softmax(&cands);
    let params = SampleParams { temp: 1.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() };
    // The drafter proposes the SECOND most likely token every time.
    let rows = vec![cands.clone(), cands.clone()];
    let mut rng = data::rng::Rng::new(data::rng::random_seed());
    let n = 60_000;
    let mut first = [0usize; 4];
    for _ in 0..n {
        let v = verify_proposals(&[11], &rows, params, &mut rng);
        let tok = if v.accepted == 1 { 11 } else { v.next };
        first[(tok - 10) as usize] += 1;
    }
    for (i, &c) in first.iter().enumerate() {
        let f = c as f64 / n as f64;
        assert!((f - p[i]).abs() < 0.01, "token {} emitted with frequency {f:.4}, target {:.4}", 10 + i, p[i]);
    }
}

#[test]
fn a_proposal_outside_the_target_support_is_always_rejected() {
    let cands = p_table();
    let params = SampleParams { temp: 1.0, top_k: 2, top_p: 1.0, ..SampleParams::greedy() };
    let rows = vec![cands.clone(), cands.clone()];
    let mut rng = data::rng::Rng::new(data::rng::random_seed());
    for _ in 0..500 {
        let v = verify_proposals(&[13], &rows, params, &mut rng);
        assert_eq!(v.accepted, 0);
        assert!([10, 11].contains(&v.next), "top_k=2 support is {{10, 11}}, got {}", v.next);
    }
}

#[test]
fn sampled_speculation_matches_the_sampled_distribution_end_to_end() {
    let params = SampleParams { temp: 1.0, top_k: 2, top_p: 1.0, ..SampleParams::greedy() };
    let counts = |spec: bool| {
        let mut m: HashMap<Vec<u32>, usize> = HashMap::new();
        // seed: the 0.05 total-variation bound over 3000 draws sits near the sampling noise and was tuned on seeds 0 to 2999
        for seed in 0..3000u64 {
            let probe = Shared::default();
            let mut sched = Scheduler::new(Toy::new(probe, true), 8);
            if spec {
                sched.set_drafter(Box::new(Scripted { right: 99 }));
            }
            let id = sched.submit_sampled(req(1, 5, 4), params, seed);
            let out = sched.run().remove(&id).unwrap();
            *m.entry(out).or_default() += 1;
        }
        m
    };
    let (a, b) = (counts(false), counts(true));
    let keys: std::collections::HashSet<_> = a.keys().chain(b.keys()).cloned().collect();
    let tv: f64 = keys.iter().map(|k| ((*a.get(k).unwrap_or(&0) as f64) - (*b.get(k).unwrap_or(&0) as f64)).abs() / 3000.0).sum::<f64>() / 2.0;
    assert!(tv < 0.05, "total variation between sampled plain and speculative output is {tv:.4}");
}

fn conformance_of(make: fn() -> Toy) -> Conformance<Toy> {
    Conformance { make, prefix_reuse: false, prefix_prompt_tokens: 36, load_adapter: None }
}

fn toy_verifying() -> Toy {
    Toy::new(Shared::default(), true)
}

fn toy_plain_only() -> Toy {
    Toy::new(Shared::default(), false)
}

fn toy_broken_commit() -> Toy {
    let mut t = Toy::new(Shared::default(), true);
    t.broken_commit = true;
    t
}

#[test]
fn the_generic_suite_passes_a_correct_speculating_decoder() {
    conformance::run_all(&conformance_of(toy_verifying));
}

#[test]
fn the_speculation_property_passes_trivially_without_speculation() {
    conformance::speculation_equals_plain_greedy(&conformance_of(toy_plain_only));
}

#[test]
#[should_panic(expected = "speculative output differs from plain greedy")]
fn the_speculation_property_catches_a_verify_that_does_not_roll_back() {
    conformance::speculation_equals_plain_greedy(&conformance_of(toy_broken_commit));
}
