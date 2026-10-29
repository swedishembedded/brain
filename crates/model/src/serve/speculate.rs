// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in LLM serving
// throughput then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Speculative decoding inside the one continuous-batching [`Scheduler`].
//!
//! A decoder opts in through the speculation hooks on [`PagedDecoder`]
//! (`spec_max_draft`, `forward_batched_verify`, `commit_verified`, and
//! optionally its own `draft`); every default says "no speculation", so a
//! decoder that has not adopted them is scheduled exactly as before. The
//! scheduler owns everything model-independent:
//!
//! * **Proposals** come from the decoder's own drafter, or from an external
//!   [`Drafter`] installed with [`Scheduler::set_drafter`] (the model-free
//!   [`NgramDrafter`] is one).
//! * **One batched verify** scores `1 + k` rows per sequence in a single
//!   decoder step; the scheduler accepts the longest prefix the target agrees
//!   with ([`verify_proposals`]: greedy match, or rejection sampling so a
//!   sampled sequence keeps the target distribution), emits the accepted
//!   tokens plus the target's own next token, and rolls the block table back
//!   to the accepted length. The decoder rolls back whatever else it holds
//!   per sequence (a recurrent state) in [`PagedDecoder::commit_verified`].
//! * **The window is adaptive.** Each sequence keeps a moving average of how
//!   many proposals the target accepted and asks for one or two more than
//!   that; a drafter that keeps missing is switched off for the sequence and
//!   re-probed now and then. Past the decoder's crossover batch size
//!   ([`PagedDecoder::spec_max_batch`]) speculation is off for the whole
//!   step and decode is the plain batched step.
//! * **Adapter rows do not speculate.** A step that carries a sequence running
//!   under a runtime adapter is decoded plainly, where the adapter applies per
//!   row; base-only steps keep speculating.
//!
//! Greedy output is identical to plain decode whatever the drafter proposes;
//! the only difference a decoder can introduce is the floating-point
//! ordering of its own verify kernels, which it documents.

use super::*;

/// What a drafter may look at for one sequence.
pub struct DraftQuery<'a> {
    /// The request id: stable for the life of the sequence, so a drafter can keep per-sequence state.
    pub id: u64,
    /// The prompt followed by every token generated so far. The last element
    /// is the token the next decode step consumes.
    pub context: &'a [u32],
    /// The sequence's block table: how many positions the decoder has
    /// consumed (`table.len()`, one less than `context.len()`) and which
    /// physical slot it owns, for a drafter that keeps state per sequence.
    pub table: &'a BlockTable,
}

/// A source of proposals that needs nothing from the decoder. A wrong
/// proposal costs a verify row, never correctness.
pub trait Drafter: Send {
    /// Up to `ks[i]` continuations of `queries[i]`, possibly fewer or none.
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>>;
}

/// Prompt-lookup drafting: find the latest earlier occurrence of the
/// context's trailing `n` tokens and propose what followed it. Free to run,
/// and strong exactly where generations copy their context (code edits, tool
/// call arguments, quoted text).
///
/// Each sequence keeps an index of the n-grams it has seen (hash chains over
/// start positions, latest first), extended by only the tokens appended since
/// its previous query, so a query costs O(new tokens x n) to index plus O(n)
/// for the lookup (chains are walked past a hash collision only), not a
/// backwards scan of the whole context. A sequence that is absent from a
/// round loses its index (bounded memory) and is re-indexed, once, when it
/// returns.
pub struct NgramDrafter {
    n: usize,
    seqs: HashMap<u64, GramIndex>,
}

/// One sequence's n-gram index. `head[h]` is the latest start whose n-gram hashes to `h`; `prev[s]` the start
/// before `s` with the same hash (or [`NO_START`]). Only starts below `indexed` are in it.
#[derive(Default)]
struct GramIndex {
    head: HashMap<u64, u32>,
    prev: Vec<u32>,
    indexed: usize,
}

const NO_START: u32 = u32::MAX;

fn gram_hash(gram: &[u32]) -> u64 {
    gram.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &t| (h ^ u64::from(t)).wrapping_mul(0x0000_0100_0000_01B3))
}

impl GramIndex {
    /// Index every n-gram that starts before `upto` (`context[..upto + n - 1]` must exist).
    fn extend(&mut self, context: &[u32], n: usize, upto: usize) {
        for start in self.indexed..upto {
            let previous = self.head.insert(gram_hash(&context[start..start + n]), start as u32);
            self.prev.push(previous.unwrap_or(NO_START));
        }
        self.indexed = self.indexed.max(upto);
    }

    /// The latest indexed start whose n-gram equals `gram`.
    fn latest(&self, context: &[u32], gram: &[u32]) -> Option<usize> {
        let mut at = *self.head.get(&gram_hash(gram))?;
        while at != NO_START {
            let start = at as usize;
            if &context[start..start + gram.len()] == gram {
                return Some(start);
            }
            at = self.prev[start];
        }
        None
    }
}

impl NgramDrafter {
    /// Match on the trailing `n` tokens (`n >= 1`).
    pub fn new(n: usize) -> NgramDrafter {
        NgramDrafter { n: n.max(1), seqs: HashMap::new() }
    }
}

impl Drafter for NgramDrafter {
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        let n = self.n;
        let mut kept: HashMap<u64, GramIndex> = HashMap::with_capacity(queries.len());
        let out = queries
            .iter()
            .zip(ks)
            .map(|(q, &k)| {
                let c = q.context;
                let mut index = self.seqs.remove(&q.id).filter(|i| i.indexed < c.len()).unwrap_or_default();
                let proposal = if k == 0 || c.len() <= n {
                    Vec::new()
                } else {
                    // Earlier occurrences start at most `c.len() - n - 1`: the suffix itself is not one.
                    index.extend(c, n, c.len() - n);
                    match index.latest(c, &c[c.len() - n..]) {
                        Some(i) => c[i + n..].iter().take(k).copied().collect(),
                        None => Vec::new(),
                    }
                };
                kept.insert(q.id, index);
                proposal
            })
            .collect();
        self.seqs = kept;
        out
    }
}

/// How the scheduler uses a decoder's speculation hooks.
#[derive(Clone, Copy, Debug)]
pub struct SpecPolicy {
    /// Master switch. Off, the scheduler never drafts or verifies.
    pub enabled: bool,
    /// Upper bound on proposals per sequence per step (further bounded by the
    /// decoder's [`PagedDecoder::spec_max_draft`]).
    pub max_draft: usize,
    /// Largest number of running sequences for which a step may speculate
    /// (further bounded by the decoder's [`PagedDecoder::spec_max_batch`]).
    pub max_batch: usize,
}

impl Default for SpecPolicy {
    fn default() -> SpecPolicy {
        SpecPolicy { enabled: true, max_draft: usize::MAX, max_batch: usize::MAX }
    }
}

/// What speculation has done so far, for a server or benchmark to report.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpecStats {
    /// Sequence-rounds that verified at least one proposal.
    pub rounds: u64,
    /// Proposals submitted for verification.
    pub proposed: u64,
    /// Proposals the target accepted.
    pub accepted: u64,
    /// Decoder steps that were verify steps.
    pub verify_steps: u64,
    /// Steps where speculation was on but the drafter proposed nothing, so the
    /// step was plain decode.
    pub plain_steps: u64,
}

impl SpecStats {
    /// Mean tokens one verified sequence-round produced: the accepted
    /// proposals plus the target's own token.
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            1.0
        } else {
            (self.accepted + self.rounds) as f64 / self.rounds as f64
        }
    }
}

/// One sequence's adaptive window: a moving average of accepted proposals per
/// round, and a countdown to the next probe while the drafter is switched off.
#[derive(Clone, Copy, Debug)]
pub(super) struct SeqSpec {
    mean_accepted: f32,
    idle_steps: u32,
}

/// Weight of the newest round in the moving average.
const ACCEPT_EWMA: f32 = 0.5;
/// Below this average a sequence stops proposing and only probes.
const SWITCH_OFF_BELOW: f32 = 0.2;
/// Plain steps between probes of a switched-off drafter.
const PROBE_EVERY: u32 = 8;

impl SeqSpec {
    pub(super) fn new() -> SeqSpec {
        SeqSpec { mean_accepted: 1.0, idle_steps: 0 }
    }

    /// Proposals to ask for this step, at most `cap`.
    fn window(&mut self, cap: usize) -> usize {
        if self.mean_accepted < SWITCH_OFF_BELOW {
            self.idle_steps += 1;
            return if self.idle_steps >= PROBE_EVERY { 1.min(cap) } else { 0 };
        }
        (self.mean_accepted.floor() as usize + 2).min(cap)
    }

    fn record(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.idle_steps = 0;
        self.mean_accepted = (1.0 - ACCEPT_EWMA) * self.mean_accepted + ACCEPT_EWMA * accepted as f32;
    }
}

/// Running estimate of how many tokens per second a kind of step produces.
#[derive(Clone, Copy, Debug, Default)]
struct Rate {
    tokens: f64,
    secs: f64,
    samples: u32,
}

/// Weight of the newest step in a [`Rate`].
const RATE_EWMA: f64 = 0.25;
/// Steps after which the mode the gate is not using is tried again, so its
/// estimate follows the workload (acceptance, batch composition) rather than
/// freezing at the first measurement.
const GATE_REFRESH_STEPS: u32 = 24;
/// Speculation must beat plain decoding by this factor to be used.
const GATE_MARGIN: f64 = 1.02;
/// How many times longer a speculative mode that misses the decoder's latency target waits to be tried
/// again: each try is a step over the target, so it is rationed to well under 1 % of the steps.
const LATENCY_REFRESH_FACTOR: u32 = 16;

impl Rate {
    fn observe(&mut self, tokens: usize, secs: f64) {
        if self.samples == 0 {
            (self.tokens, self.secs) = (tokens as f64, secs);
        } else {
            self.tokens += RATE_EWMA * (tokens as f64 - self.tokens);
            self.secs += RATE_EWMA * (secs - self.secs);
        }
        self.samples += 1;
    }

    fn per_second(&self) -> f64 {
        if self.secs > 0.0 {
            self.tokens / self.secs
        } else {
            f64::INFINITY
        }
    }
}

/// What one batch-size bucket has measured of plain and speculative steps.
#[derive(Clone, Copy, Debug, Default)]
struct GateBucket {
    plain: Rate,
    spec: Rate,
    since_plain: u32,
    since_spec: u32,
}

/// **The self-calibrating crossover.** Whether a speculative step pays depends
/// on the decoder, the batch size and the workload's acceptance, and none of
/// them is known in advance. The gate times both kinds of step, per batch-size
/// bucket, and speculates only where speculation has been producing more
/// tokens per second than plain decoding; the mode it is not using is
/// re-measured every [`GATE_REFRESH_STEPS`] steps. Only decoders that say their
/// step cost is meaningful ([`PagedDecoder::spec_cost_aware`]) are gated.
#[derive(Debug, Default)]
pub(super) struct CostGate {
    buckets: [GateBucket; 9],
}

impl CostGate {
    /// The bucket of a step over `active` sequences: powers of two.
    pub(super) fn bucket(active: usize) -> usize {
        ((usize::BITS - active.max(1).leading_zeros()) as usize - 1).min(8)
    }

    /// Whether speculative steps of this bucket are too slow for the latency `target` (seconds a step may take)
    /// while plain ones meet it: the throughput they gain is paid for in the token interval of every sequence in the
    /// step, which the target says matters more.
    fn over_latency_target(&self, bucket: usize, target: Option<f64>) -> bool {
        let b = &self.buckets[bucket];
        target.is_some_and(|t| b.spec.samples > 0 && b.plain.samples > 0 && b.spec.secs > t && b.plain.secs <= t)
    }

    /// Whether the next step over this bucket should speculate, given the latency `target` of a step, if the
    /// decoder holds one.
    pub(super) fn speculate(&mut self, bucket: usize, target: Option<f64>) -> bool {
        let over = self.over_latency_target(bucket, target);
        let b = &mut self.buckets[bucket];
        if b.spec.samples == 0 {
            return true; // learn what speculation costs here
        }
        if b.plain.samples == 0 {
            return false; // and what plain decoding does
        }
        if over {
            b.since_spec += 1;
            return b.since_spec >= GATE_REFRESH_STEPS * LATENCY_REFRESH_FACTOR;
        }
        if b.spec.per_second() >= b.plain.per_second() * GATE_MARGIN {
            b.since_plain += 1;
            b.since_plain < GATE_REFRESH_STEPS
        } else {
            b.since_spec += 1;
            b.since_spec >= GATE_REFRESH_STEPS
        }
    }

    /// Seconds of decoding a generated token saves when this bucket speculates:
    /// the difference of the two measured per-token costs, `None` until both
    /// modes have been timed here. Zero where speculation is the slower one, or is not used for missing the
    /// latency `target`.
    pub(super) fn saved_secs_per_token(&self, bucket: usize, target: Option<f64>) -> Option<f64> {
        let b = &self.buckets[bucket];
        if self.over_latency_target(bucket, target) {
            return Some(0.0);
        }
        (b.plain.samples > 0 && b.spec.samples > 0).then(|| (1.0 / b.plain.per_second() - 1.0 / b.spec.per_second()).max(0.0))
    }

    pub(super) fn observe_plain(&mut self, bucket: usize, tokens: usize, secs: f64) {
        let b = &mut self.buckets[bucket];
        b.plain.observe(tokens, secs);
        b.since_plain = 0;
    }

    pub(super) fn observe_spec(&mut self, bucket: usize, tokens: usize, secs: f64) {
        let b = &mut self.buckets[bucket];
        b.spec.observe(tokens, secs);
        b.since_spec = 0;
    }
}

/// Every this-many-th prompt prepares the drafter whatever the arithmetic says, so the
/// saving it makes possible keeps being measured.
const PREFILL_PROBE_EVERY: u32 = 16;

/// **Whether a new prompt should prepare the drafter.** A drafter that needs each prompt's
/// hidden states ([`PagedDecoder::spec_prefill_cost`]) charges a cost per request; speculation
/// repays it with what it saves decoding that request's reply. Both are measured: the cost by
/// the decoder, the saving by the [`CostGate`]'s per-token rates times the running mean reply
/// length. Where speculation cannot repay the preparation (long prompts, short replies: an
/// agent turn) the prompt is not prepared, and its sequence decodes plainly; a probe every
/// [`PREFILL_PROBE_EVERY`] prompts keeps the saving measured so the answer can change back.
/// Until the cost, the reply length and both step rates are known the drafter is prepared.
#[derive(Debug, Default)]
pub(super) struct PrefillGate {
    reply_tokens: Option<f64>,
    skipped: u32,
}

impl PrefillGate {
    pub(super) fn observe_reply(&mut self, tokens: usize) {
        self.reply_tokens = Some(self.reply_tokens.map_or(tokens as f64, |m| m + RATE_EWMA * (tokens as f64 - m)));
    }

    /// Prepare the drafter for a prompt whose sequence will decode among about `active` others.
    pub(super) fn prepare(&mut self, gate: &CostGate, active: usize, cost: Option<f64>, target: Option<f64>) -> bool {
        let (Some(cost), Some(reply)) = (cost, self.reply_tokens) else { return true };
        let Some(saved) = gate.saved_secs_per_token(CostGate::bucket(active), target) else { return true };
        if reply * saved >= cost {
            self.skipped = 0;
            return true;
        }
        self.skipped += 1;
        if self.skipped >= PREFILL_PROBE_EVERY {
            self.skipped = 0;
            return true;
        }
        false
    }
}

/// The outcome of checking one sequence's proposals against the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// The leading proposals the target accepted.
    pub accepted: usize,
    /// The target's own next token: the correction at the first rejected
    /// position, or the bonus token after the last accepted one.
    pub next: u32,
}

/// Decide how many of `proposals` stand. `rows[j]` is the target's candidate
/// list (best first) for the position `proposals[j]` was proposed at, and
/// `rows[proposals.len()]` the one after the last proposal.
///
/// Greedy: accept while the proposal is the target's top candidate. Sampled:
/// the proposals come from a deterministic drafter (a point mass), so the
/// standard speculative-sampling test reduces to accepting a proposal `d`
/// with probability `p(d)` under the temperature / top-k / top-p
/// distribution the target would have sampled from, and on rejection drawing
/// from that distribution with `d` removed and renormalised. The emitted
/// tokens are then distributed exactly as plain sampling would produce them.
pub fn verify_proposals(proposals: &[u32], rows: &[Vec<(u32, f32)>], params: SampleParams, rng: &mut data::rng::Rng) -> Verdict {
    assert_eq!(rows.len(), proposals.len() + 1, "verify_proposals: {} candidate rows for {} proposals", rows.len(), proposals.len());
    if params.is_greedy() {
        let accepted = proposals.iter().zip(rows).take_while(|(d, row)| row[0].0 == **d).count();
        return Verdict { accepted, next: rows[accepted][0].0 };
    }
    for (j, &d) in proposals.iter().enumerate() {
        let (weights, kept_sum) = truncated_weights(&rows[j], params);
        let at = rows[j][..weights.len()].iter().position(|&(t, _)| t == d);
        let p_d = at.map_or(0.0, |i| weights[i]);
        if rng.next_f32() * kept_sum < p_d {
            continue;
        }
        // Rejected: draw from the target distribution without `d`.
        let rest = kept_sum - p_d;
        let r = rng.next_f32() * rest;
        let mut acc = 0.0f32;
        let mut last = None;
        for (i, &w) in weights.iter().enumerate() {
            if Some(i) == at {
                continue;
            }
            last = Some(rows[j][i].0);
            acc += w;
            if acc >= r {
                return Verdict { accepted: j, next: rows[j][i].0 };
            }
        }
        // Rounding left `r` past the last weight; the last candidate is the tail of the CDF.
        return Verdict { accepted: j, next: last.expect("a rejected proposal had probability < 1, so another candidate exists") };
    }
    let k = proposals.len();
    let (weights, kept_sum) = truncated_weights(&rows[k], params);
    let r = rng.next_f32() * kept_sum;
    let mut acc = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if acc >= r {
            return Verdict { accepted: k, next: rows[k][i].0 };
        }
    }
    Verdict { accepted: k, next: rows[k][weights.len() - 1].0 }
}

/// The block tables of the `active` sequences, in `active` order.
fn active_tables<'a>(running: &'a mut [Running], active: &[usize]) -> Vec<&'a mut BlockTable> {
    running.iter_mut().enumerate().filter(|(i, _)| active.contains(i)).map(|(_, r)| &mut r.table).collect()
}

impl<D: PagedDecoder> Scheduler<D> {
    /// Install an external drafter. It is used in place of the decoder's own
    /// [`PagedDecoder::draft`]; a decoder that cannot verify
    /// ([`PagedDecoder::spec_max_draft`] of 0) ignores it.
    pub fn set_drafter(&mut self, drafter: Box<dyn Drafter>) {
        self.drafter = Some(drafter);
    }

    /// Configure speculation (default: on whenever the decoder supports it).
    pub fn set_speculation(&mut self, policy: SpecPolicy) {
        self.spec_policy = policy;
    }

    /// What speculation has done so far.
    pub fn spec_stats(&self) -> SpecStats {
        self.spec_stats
    }

    /// A still-running sequence's block table (`None` once reaped, queued, or
    /// preempted to host RAM). Read-only: the scheduler owns its lifecycle.
    pub fn table_of(&self, id: u64) -> Option<&BlockTable> {
        self.running.iter().find(|r| r.id == id).map(|r| &r.table)
    }

    /// Run one speculative decode step over `active` (indices into
    /// `self.running`) if speculation is on, the batch is small enough, and
    /// the drafter proposed anything. Returns whether it did; `false` leaves
    /// every sequence untouched for the plain step.
    pub(super) fn speculative_step(&mut self, active: &[usize]) -> bool {
        let kmax = self.spec_policy.max_draft.min(self.dec.spec_max_draft());
        if !self.spec_policy.enabled || kmax == 0 || active.len() > self.spec_policy.max_batch.min(self.dec.spec_max_batch()) {
            return false;
        }
        let budget = self.dec.spec_row_budget();
        if active.len() > budget {
            return false;
        }
        // A verify step scores its rows under the base weights only: a decoder's
        // verify path has no per-row adapter selection, so a step that carries a
        // sequence running under an adapter is the plain batched step.
        if active.iter().any(|&i| self.running[i].table.kv_namespace() != crate::paged::BASE_KV_NAMESPACE) {
            return false;
        }
        // The request's logit processors read the tokens chosen earlier in the same
        // step, which a verify step scores in one pass: a step that carries a
        // sequence with one is the plain step.
        if active.iter().any(|&i| self.running[i].sample.has_processors() || self.running[i].think.is_some_and(|t| t.watching())) {
            return false;
        }
        let bucket = CostGate::bucket(active.len());
        let gated = self.dec.spec_cost_aware();
        if gated && !self.gate.speculate(bucket, self.dec.step_latency_target().map(|t| t.as_secs_f64())) {
            return false;
        }
        let started = (self.clock)();
        let mut ks = self.draft_windows(active, kmax);
        // Shrink the widest windows until the step fits the decoder's scratch.
        let mut rows = active.len() + ks.iter().sum::<usize>();
        while rows > budget {
            let widest = (0..ks.len()).max_by_key(|&i| ks[i]).expect("a non-empty batch");
            ks[widest] -= 1;
            rows -= 1;
        }
        let queries: Vec<DraftQuery> = active.iter().map(|&i| DraftQuery { id: self.running[i].id, context: &self.running[i].context, table: &self.running[i].table }).collect();
        let mut props = match self.drafter.as_mut() {
            Some(d) => d.draft(&queries, &ks),
            None => self.dec.draft(&queries, &ks),
        };
        drop(queries);
        assert_eq!(props.len(), active.len(), "a drafter must answer every query");
        let vocab = self.dec.vocab() as u32;
        for (p, &k) in props.iter_mut().zip(&ks) {
            p.truncate(k);
            // A drafter is untrusted input: cut at the first out-of-vocabulary
            // proposal rather than splicing the continuations around it.
            if let Some(bad) = p.iter().position(|&t| t >= vocab) {
                p.truncate(bad);
            }
        }
        if props.iter().all(|p| p.is_empty()) {
            // Nothing proposed is nothing rejected: the windows stay as they were.
            self.spec_stats.plain_steps += 1;
            return false;
        }
        for (k, p) in ks.iter_mut().zip(&props) {
            *k = p.len();
        }

        let rows: Vec<Vec<u32>> = active.iter().zip(&props).map(|(&i, p)| std::iter::once(self.running[i].next_input).chain(p.iter().copied()).collect()).collect();
        let all_greedy = active.iter().all(|&i| !self.running[i].sample.needs_candidates());
        let cands = if all_greedy { 1 } else { self.dec.topk_capacity().max(1) };
        let before: Vec<u32> = active.iter().map(|&i| self.running[i].table.len()).collect();

        let scored = self.dec.forward_batched_verify(&mut active_tables(&mut self.running, active), &rows, cands);
        assert_eq!(scored.len(), active.len(), "forward_batched_verify must answer every sequence");

        let mut keep = Vec::with_capacity(active.len());
        let mut verdicts = Vec::with_capacity(active.len());
        for (bi, &si) in active.iter().enumerate() {
            let r = &mut self.running[si];
            let v = verify_proposals(&props[bi], &scored[bi], r.sample, &mut r.rng);
            // The input row and every accepted proposal keep their state; the
            // rest of the verified block is rolled back.
            keep.push(v.accepted + 1);
            verdicts.push(v);
        }
        self.dec.commit_verified(&mut active_tables(&mut self.running, active), &keep);

        self.spec_stats.verify_steps += 1;
        let tokens_before: usize = active.iter().map(|&i| self.running[i].generated.len()).sum();
        for (bi, &si) in active.iter().enumerate() {
            let v = verdicts[bi];
            let r = &mut self.running[si];
            let new_len = before[bi] + keep[bi] as u32;
            r.table.truncate(new_len, self.dec.alloc_mut());
            r.spec.record(ks[bi], v.accepted);
            if ks[bi] > 0 {
                self.spec_stats.rounds += 1;
                self.spec_stats.proposed += ks[bi] as u64;
                self.spec_stats.accepted += v.accepted as u64;
            }
            for (j, &tok) in props[bi][..v.accepted].iter().chain(std::iter::once(&v.next)).enumerate() {
                // Row `j` scored the position `tok` was emitted at.
                accept_token(r, tok, candidate_logprob(&scored[bi][j], r.sample, tok));
                if r.done {
                    break;
                }
            }
        }
        if gated {
            let produced: usize = active.iter().map(|&i| self.running[i].generated.len()).sum::<usize>() - tokens_before;
            self.gate.observe_spec(bucket, produced, (self.clock)() - started);
        }
        true
    }

    /// Seconds on the clock the cost gate times steps with (default: wall time).
    /// A test substitutes a virtual one.
    pub fn set_clock(&mut self, clock: Box<dyn Fn() -> f64 + Send>) {
        self.clock = clock;
    }

    /// Per-sequence proposal budget for this step: the adaptive window,
    /// bounded by what the sequence can still emit and by free KV blocks.
    fn draft_windows(&mut self, active: &[usize], kmax: usize) -> Vec<usize> {
        let mut spare = self.dec.free_blocks().saturating_sub(self.committed_blocks());
        let mut ks = Vec::with_capacity(active.len());
        for &i in active {
            let r = &mut self.running[i];
            // A round emits at most accepted + 1 tokens, so more than
            // `remaining - 1` proposals could never all be used.
            let remaining = r.max_new.saturating_sub(r.generated.len());
            let mut k = r.spec.window(kmax).min(remaining.saturating_sub(1));
            let held = r.table.blocks().len() as u32;
            // The decode step's own token is covered by `ensure_decode_room`;
            // proposals each need a position beyond it.
            let extra = |k: usize| self.dec.blocks_for(r.table.len() + 1 + k as u32).saturating_sub(held.max(self.dec.blocks_for(r.table.len() + 1)));
            while k > 0 && extra(k) > spare {
                k -= 1;
            }
            spare -= extra(k);
            ks.push(k);
        }
        ks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bucket of four sequences where plain steps take 20 ms for four tokens and speculative ones 60 ms for
    /// sixteen: more tokens a second, a longer wait between them.
    fn measured() -> CostGate {
        let mut g = CostGate::default();
        for _ in 0..4 {
            g.observe_plain(2, 4, 0.020);
            g.observe_spec(2, 16, 0.060);
        }
        g
    }

    #[test]
    fn without_a_latency_target_the_faster_mode_wins() {
        let mut g = measured();
        assert!(g.speculate(2, None), "267 tokens a second beat 200");
    }

    #[test]
    fn a_mode_that_misses_the_latency_target_is_not_used_and_is_tried_rarely() {
        let mut g = measured();
        let target = Some(0.045);
        let steps = GATE_REFRESH_STEPS * LATENCY_REFRESH_FACTOR * 2;
        let tries = (0..steps)
            .filter(|_| {
                let speculated = g.speculate(2, target);
                if speculated {
                    g.observe_spec(2, 16, 0.060);
                }
                speculated
            })
            .count();
        assert_eq!(tries, 2, "once in {} steps, not every {GATE_REFRESH_STEPS}", GATE_REFRESH_STEPS * LATENCY_REFRESH_FACTOR);
        assert_eq!(g.saved_secs_per_token(2, target), Some(0.0), "so a prompt's drafter preparation does not pay");
    }

    #[test]
    fn a_target_that_plain_steps_miss_too_changes_nothing() {
        let mut g = measured();
        assert!(g.speculate(2, Some(0.010)));
    }

    /// The incremental n-gram index answers what the backwards scan it replaced answered, for
    /// contexts that grow round by round, several sequences sharing one drafter, and a sequence
    /// that drops out of a round and returns.
    #[test]
    fn the_ngram_index_equals_the_backwards_scan() {
        fn scan(c: &[u32], n: usize, k: usize) -> Vec<u32> {
            if k == 0 || c.len() <= n {
                return Vec::new();
            }
            let suffix = &c[c.len() - n..];
            match (0..c.len() - n).rev().find(|&i| &c[i..i + n] == suffix) {
                Some(i) => c[i + n..].iter().take(k).copied().collect(),
                None => Vec::new(),
            }
        }
        let table = BlockTable::new();
        let mut rng = data::rng::Lcg::new(9);
        for n in 1..=4 {
            let mut drafter = NgramDrafter::new(n);
            let mut contexts: Vec<Vec<u32>> = vec![Vec::new(); 3];
            for round in 0..120 {
                for c in contexts.iter_mut() {
                    for _ in 0..1 + (rng.unit() * 3.0) as usize {
                        c.push((rng.unit() * 4.0) as u32);
                    }
                }
                // Sequence 2 sits out some rounds.
                let present: Vec<usize> = (0..3).filter(|&i| i != 2 || round % 5 != 0).collect();
                let queries: Vec<DraftQuery> = present.iter().map(|&i| DraftQuery { id: i as u64, context: &contexts[i], table: &table }).collect();
                let ks: Vec<usize> = present.iter().map(|_| (rng.unit() * 5.0) as usize).collect();
                let got = drafter.draft(&queries, &ks);
                for (j, &i) in present.iter().enumerate() {
                    assert_eq!(got[j], scan(&contexts[i], n, ks[j]), "n {n} round {round} sequence {i}");
                }
            }
        }
    }
}
