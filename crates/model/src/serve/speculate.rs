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
use super::budget::StepCost;

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
    /// Steps the cost gate sent to plain decoding because speculation had measured slower.
    pub gated_steps: u64,
    /// Verify steps that were mixed iterations (their decode rows verified beside prompt rows).
    pub mixed_verify_steps: u64,
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

/// One sequence's acceptance record: how likely the target is to accept a proposal at each depth of a window, as the
/// maximum-likelihood per-position acceptance of a geometric run (proposals are accepted from the first until one is not), over
/// recent rounds, and a countdown to the next probe while the drafter is switched off.
#[derive(Clone, Copy, Debug)]
pub(super) struct SeqSpec {
    /// Decayed count of proposals accepted, and of rounds that ended in a rejection (a window cut short).
    accepted: f32,
    rejected: f32,
    idle_steps: u32,
}

/// Weight kept of the past at each round: the estimate follows the last handful of rounds, since repeated text comes in bursts.
const ACCEPT_DECAY: f32 = 0.75;
/// Prior of the acceptance estimate: one success and one failure.
const ACCEPT_PRIOR: f32 = 1.0;
/// Plain steps between probes of a switched-off drafter.
const PROBE_EVERY: u32 = 8;

impl SeqSpec {
    pub(super) fn new() -> SeqSpec {
        SeqSpec { accepted: ACCEPT_PRIOR, rejected: ACCEPT_PRIOR, idle_steps: 0 }
    }

    /// The probability that a proposal is accepted given the ones before it were.
    pub(super) fn acceptance(&self) -> f32 {
        self.accepted / (self.accepted + self.rejected)
    }

    /// Proposals to ask for this step, at most `cap`: those whose chance of being accepted, times what a token is worth, exceeds
    /// what a proposal costs ([`SpecPrices::window`]). A sequence that earns none proposes one now and then to learn whether
    /// its text has become predictable.
    fn window(&mut self, cap: usize, prices: SpecPrices) -> usize {
        let k = prices.window(self.acceptance()).min(cap);
        if k > 0 {
            self.idle_steps = 0;
            return k;
        }
        self.idle_steps += 1;
        if self.idle_steps >= PROBE_EVERY {
            1.min(cap)
        } else {
            0
        }
    }

    fn record(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.idle_steps = 0;
        self.accepted = ACCEPT_DECAY * self.accepted + accepted as f32;
        self.rejected = ACCEPT_DECAY * self.rejected + f32::from(accepted < proposed);
    }
}

/// **What a proposal costs and what it earns**, in seconds, for the step about to run. A proposal is one more row in the verify
/// forward and one more step of the drafter; it earns a token with the probability that it and every proposal before it are
/// accepted, and a token is worth what the batch takes to produce one (`token_secs`). The window is the proposals that earn more
/// than they cost - the marginal analysis, so a batch that is compute-bound (rows expensive, tokens cheap) asks for few and one
/// that is bound by its weights (rows nearly free) for many.
#[derive(Clone, Copy, Debug)]
pub(super) struct SpecPrices {
    /// Seconds one more verify row adds to the forward.
    pub row_secs: f64,
    /// Seconds one more proposal adds to the drafter.
    pub draft_secs: f64,
    /// Seconds the step spends per token it produces without speculation.
    pub token_secs: f64,
}

impl SpecPrices {
    /// Longest window whose last proposal still pays at per-position acceptance `alpha`.
    fn window(&self, alpha: f32) -> usize {
        let cost = self.row_secs + self.draft_secs;
        let mut reach = 1.0f64;
        let mut k = 0;
        while k < MAX_WINDOW {
            reach *= f64::from(alpha);
            if reach * self.token_secs < cost {
                break;
            }
            k += 1;
        }
        k
    }
}

/// Tokens a window of `k` proposals is expected to add at per-position acceptance `alpha`.
fn expected_gain(alpha: f32, k: usize) -> f64 {
    let (mut reach, mut gain) = (1.0f64, 0.0f64);
    for _ in 0..k {
        reach *= f64::from(alpha);
        gain += reach;
    }
    gain
}

/// A round must earn this multiple of its cost to run.
const ROUND_MARGIN: f64 = 1.05;

/// Windows are bounded by the verify kernels' rows per sequence; the economics decide below it.
const MAX_WINDOW: usize = 7;

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
const GATE_REFRESH_STEPS: u32 = 8;
/// Under this fraction of the other mode's tokens a second, a mode is far behind and is retried at the longer interval.
const GATE_FAR_BEHIND: f64 = 0.5;
const GATE_FAR_REFRESH_STEPS: u32 = 24;
/// Steps of a mode the gate measures before it may rule that mode out: the first steps of a shape pay for kernel selection
/// and recording, and say nothing about what the mode costs once it runs.
const GATE_WARMUP_STEPS: u32 = 3;
/// A step this many times slower than the mode's running estimate is a stall (a first launch, a clock change), not the cost.
const GATE_OUTLIER_FACTOR: f64 = 3.0;
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
            let secs = secs.min(self.secs * GATE_OUTLIER_FACTOR);
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
        if b.spec.samples < GATE_WARMUP_STEPS {
            return true; // learn what speculation costs here, past the first launches
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
            // A mode that loses by a wide margin is tried again less often than one that is close.
            let refresh = if b.spec.per_second() < b.plain.per_second() * GATE_FAR_BEHIND { GATE_FAR_REFRESH_STEPS } else { GATE_REFRESH_STEPS };
            b.since_spec += 1;
            b.since_spec >= refresh
        }
    }

    /// Seconds a plain step of this bucket takes, once one was timed.
    pub(super) fn plain_secs(&self, bucket: usize) -> Option<f64> {
        let b = &self.buckets[bucket];
        (b.plain.samples > 0).then_some(b.plain.secs)
    }

    /// Seconds a plain step of this bucket spends per token it produces, once one was timed.
    pub(super) fn plain_token_secs(&self, bucket: usize) -> Option<f64> {
        let b = &self.buckets[bucket];
        (b.plain.samples > 0 && b.plain.tokens > 0.0).then(|| b.plain.secs / b.plain.tokens)
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

/// What the scheduler has measured of the prices a proposal is judged by ([`SpecPrices`]).
///
/// A verify forward costs the plain step of the same sequences plus its extra rows; a draft costs a fixed part (the catch-up
/// over what the target consumed since the last draft) plus a part per proposal. Each of the four is a decayed mean of what the
/// steps that ran took beyond the others' share, so a row count the scheduler has not tried yet is priced by the rows it has.
#[derive(Debug, Default)]
pub(super) struct SpecCosts {
    /// Seconds each row of a decode-only verify forward adds to the plain step of its sequences.
    verify_row: Option<f64>,
    /// Seconds a decode-only verify forward takes beyond that plain step and its rows.
    verify_fixed: Option<f64>,
    /// Seconds the drafter took per proposal it returned, and beyond them.
    draft_per_proposal: Option<f64>,
    draft_fixed: Option<f64>,
    /// Seconds of the latest mixed iteration, for the worth of a token in one.
    mixed_iteration_secs: Option<f64>,
}

/// Prices assumed before they were measured, optimistic so that speculation is tried and measured: a token costs the 12 ms of a
/// single-stream decode step, a verify row a tenth of a token, a proposal nothing to draft. The decision to speculate uses them
/// only until the first rounds have been timed.
const PRIOR_TOKEN_SECS: f64 = 0.012;
const PRIOR_ROW_FRACTION: f64 = 0.1;

fn blend(mean: &mut Option<f64>, sample: f64) {
    if sample.is_finite() {
        *mean = Some(mean.map_or(sample, |m| m + COST_EWMA * (sample - m)));
    }
}

/// Weight of the newest step in a measured price.
const COST_EWMA: f64 = 0.2;

impl SpecCosts {
    /// A decode-only verify forward of `rows` rows over `active` sequences took `secs`; a plain step of them takes `plain_secs`.
    pub(super) fn observe_verify(&mut self, rows: usize, active: usize, secs: f64, plain_secs: Option<f64>) {
        let (Some(plain), true) = (plain_secs, rows > active) else { return };
        let extra = (rows - active) as f64;
        // Beyond the fixed part, the first observations say what a row costs; later ones refine the fixed part under it.
        let row = self.verify_row.unwrap_or(((secs - plain) / extra).max(0.0));
        blend(&mut self.verify_row, ((secs - plain) / extra).max(0.0));
        blend(&mut self.verify_fixed, (secs - plain - row * extra).max(0.0));
    }

    /// A draft that returned `proposals` took `secs`.
    pub(super) fn observe_draft(&mut self, proposals: usize, secs: f64) {
        if proposals == 0 {
            return;
        }
        let per = secs / proposals as f64;
        blend(&mut self.draft_per_proposal, per);
        blend(&mut self.draft_fixed, 0.0);
    }

    pub(super) fn observe_mixed(&mut self, secs: f64) {
        blend(&mut self.mixed_iteration_secs, secs);
    }

    /// The prices for a step over `active` sequences: of a mixed iteration (rows priced by `mixed`, its own fit) or a decode-only
    /// one, whose token costs `plain_token_secs` when plain steps of this size have been timed.
    pub(super) fn prices(&self, active: usize, mixed: Option<&StepCost>, plain_token_secs: Option<f64>) -> SpecPrices {
        let (row_secs, token_secs) = match mixed {
            Some(fit) => (fit.slope(), self.mixed_iteration_secs.map(|s| s / active.max(1) as f64)),
            None => (self.verify_row, plain_token_secs),
        };
        let token_secs = token_secs.unwrap_or(PRIOR_TOKEN_SECS);
        SpecPrices { row_secs: row_secs.unwrap_or(PRIOR_ROW_FRACTION * token_secs), draft_secs: self.draft_per_proposal.unwrap_or(0.0), token_secs }
    }

    /// Seconds a decode-only round costs beyond the plain step regardless of its proposals: the verify path's own overhead and
    /// the draft's catch-up.
    pub(super) fn fixed_secs(&self) -> f64 {
        self.verify_fixed.unwrap_or(0.0) + self.draft_fixed.unwrap_or(0.0)
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

/// What drafting settled for one step: the rows each sequence verifies, ready for the decoder's verify forward.
pub(super) struct Proposals {
    /// Per sequence, the proposed tokens (possibly none).
    props: Vec<Vec<u32>>,
    /// Per sequence, the proposals it asked for as they stand (what the window adaptation is told).
    ks: Vec<usize>,
    /// Per sequence, its next input followed by its proposals.
    pub(super) rows: Vec<Vec<u32>>,
    /// Candidates the verify forward returns per row.
    pub(super) cands: usize,
    /// Per sequence, the positions its table held before the verify appended its rows.
    before: Vec<u32>,
    started: f64,
    bucket: usize,
    gated: bool,
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
        let Some(proposals) = self.propose(active, false) else { return false };
        let (scored, verify_secs) = {
            let _verify = gpu_core::markers::range(c"spec.verify");
            let started = (self.clock)();
            let scored = self.dec.forward_batched_verify(&mut active_tables(&mut self.running, active), &proposals.rows, proposals.cands);
            (scored, (self.clock)() - started)
        };
        self.spec_costs.observe_verify(proposals.rows.iter().map(Vec::len).sum(), active.len(), verify_secs, self.gate.plain_secs(CostGate::bucket(active.len())));
        self.settle_verified(active, proposals, scored);
        true
    }

    /// **Draft for one step** over `active`: `Some` with the rows each sequence verifies (its next input then its
    /// proposals) when speculation is on, the batch fits the decoder's limits and the drafter proposed anything;
    /// `None` leaves every sequence untouched for the plain step. A decode-only step competes with a plain one on the
    /// measured tokens a second ([`CostGate`]); a `mixed` iteration, whose time is mostly its prompt rows, does not.
    pub(super) fn propose(&mut self, active: &[usize], mixed: bool) -> Option<Proposals> {
        let gated = !mixed;
        let kmax = self.spec_policy.max_draft.min(self.dec.spec_max_draft());
        if active.is_empty() || !self.spec_policy.enabled || kmax == 0 || active.len() > self.spec_policy.max_batch.min(self.dec.spec_max_batch()) {
            return None;
        }
        let budget = self.dec.spec_row_budget();
        if active.len() > budget {
            return None;
        }
        // A verify step scores its rows under the base weights only: a decoder's
        // verify path has no per-row adapter selection, so a step that carries a
        // sequence running under an adapter is the plain batched step.
        if active.iter().any(|&i| self.running[i].table.kv_namespace() != crate::paged::BASE_KV_NAMESPACE) {
            return None;
        }
        // The request's logit processors read the tokens chosen earlier in the same
        // step, which a verify step scores in one pass: a step that carries a
        // sequence with one is the plain step.
        if active.iter().any(|&i| self.running[i].sample.has_processors() || self.running[i].think.is_some_and(|t| t.watching())) {
            return None;
        }
        let bucket = CostGate::bucket(active.len());
        let gated = gated && self.dec.spec_cost_aware();
        // Speculation is ranked on tokens a second alone: at a fixed batch, a step that produces more tokens a second also gives
        // every sequence a shorter mean interval between tokens, so a step-duration target has nothing to add to the comparison.
        if gated && !self.gate.speculate(bucket, None) {
            self.spec_stats.gated_steps += 1;
            return None;
        }
        let started = (self.clock)();
        let prices = self.spec_costs.prices(active.len(), mixed.then_some(&self.step_cost), self.gate.plain_token_secs(bucket));
        let mut ks = self.draft_windows(active, kmax, prices);
        // Shrink the widest windows until the step fits the decoder's scratch.
        let mut rows = active.len() + ks.iter().sum::<usize>();
        while rows > budget {
            let widest = (0..ks.len()).max_by_key(|&i| ks[i]).expect("a non-empty batch");
            ks[widest] -= 1;
            rows -= 1;
        }
        // A decode-only round also pays the verify path's and the draft's fixed costs; it runs only if what its windows are expected to
        // earn covers them with the proposals' own prices (a mixed iteration's fixed costs are the iteration's).
        if !mixed {
            let proposals: usize = ks.iter().sum();
            let earned: f64 = active.iter().zip(&ks).map(|(&i, &k)| expected_gain(self.running[i].spec.acceptance(), k)).sum::<f64>() * prices.token_secs;
            let cost = self.spec_costs.fixed_secs() + (prices.row_secs + prices.draft_secs) * proposals as f64;
            if proposals == 0 || earned < cost * ROUND_MARGIN {
                return None;
            }
        }
        let queries: Vec<DraftQuery> = active.iter().map(|&i| DraftQuery { id: self.running[i].id, context: &self.running[i].context, table: &self.running[i].table }).collect();
        let draft_range = gpu_core::markers::range(c"spec.draft");
        let draft_started = (self.clock)();
        let mut props = match self.drafter.as_mut() {
            Some(d) => d.draft(&queries, &ks),
            None => self.dec.draft(&queries, &ks),
        };
        let draft_secs = (self.clock)() - draft_started;
        drop(draft_range);
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
            return None;
        }
        for (k, p) in ks.iter_mut().zip(&props) {
            *k = p.len();
        }
        self.spec_costs.observe_draft(ks.iter().sum(), draft_secs);

        let rows: Vec<Vec<u32>> = active.iter().zip(&props).map(|(&i, p)| std::iter::once(self.running[i].next_input).chain(p.iter().copied()).collect()).collect();
        let all_greedy = active.iter().all(|&i| !self.running[i].sample.needs_candidates());
        let cands = if all_greedy { 1 } else { self.dec.topk_capacity().max(1) };
        let before: Vec<u32> = active.iter().map(|&i| self.running[i].table.len()).collect();
        Some(Proposals { props, ks, rows, cands, before, started, bucket, gated })
    }

    /// **Settle a verify step**: `scored[i][j]` are the target's candidates for row `j` of sequence `active[i]`
    /// (`proposals.rows`, scored as one forward). Each sequence keeps the longest prefix of its proposals the target
    /// agrees with ([`verify_proposals`]) and the target's own next token; the decoder rolls back the rest of its
    /// state and the tables are truncated to what stands.
    pub(super) fn settle_verified(&mut self, active: &[usize], proposals: Proposals, scored: Vec<Vec<Vec<(u32, f32)>>>) {
        let Proposals { props, ks, before, started, bucket, gated, .. } = proposals;
        self.spec_stats.mixed_verify_steps += u64::from(!gated);
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
        {
            let _commit = gpu_core::markers::range(c"spec.commit");
            self.dec.commit_verified(&mut active_tables(&mut self.running, active), &keep);
        }

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
    }

    /// Seconds on the clock the cost gate times steps with (default: wall time).
    /// A test substitutes a virtual one.
    pub fn set_clock(&mut self, clock: Box<dyn Fn() -> f64 + Send>) {
        self.clock = clock;
    }

    /// Per-sequence proposal budget for this step: the adaptive window,
    /// bounded by what the sequence can still emit and by free KV blocks.
    fn draft_windows(&mut self, active: &[usize], kmax: usize, prices: SpecPrices) -> Vec<usize> {
        let mut spare = self.dec.free_blocks().saturating_sub(self.committed_blocks());
        let mut ks = Vec::with_capacity(active.len());
        for &i in active {
            let r = &mut self.running[i];
            // A round emits at most accepted + 1 tokens, so more than
            // `remaining - 1` proposals could never all be used.
            let remaining = r.max_new.saturating_sub(r.generated.len());
            let mut k = r.spec.window(kmax, prices).min(remaining.saturating_sub(1));
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
        // seed: a fixed stream so a failing context reproduces
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
