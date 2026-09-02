// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Architecture-agnostic continuous-batching scheduler over a paged KV cache.
//!
//! Everything here is generic over [`PagedDecoder`] — the seam a model's own
//! serving engine implements to plug into [`Scheduler`]: admission, batched
//! decode with an on-device window, cancellation, prefix reuse, and the
//! `StepReport` timeline `brain perf` measures TTFA/ITL from. None of it
//! depends on a specific architecture; `crates/qwen3/src/serve.rs::Engine` is
//! the first (and, at time of writing, only) implementation, and
//! `qwen3::serve::Scheduler` is a type alias for `Scheduler<Engine>` so no
//! caller of the qwen-specific names needs to change.
//!
//! Adopting this for another decoder (glm, gpt, moe) means writing that
//! architecture's own paged/batched forward (its `run_batched_submit`
//! equivalent) and implementing [`PagedDecoder`] over it — admission, prefix
//! cache, batching, cancellation and streaming all come from this module for
//! free.

use std::collections::{HashMap, VecDeque};

use crate::paged::{BlockAllocator, BlockTable};

/// The seam a model's paged serving engine implements. Every method mirrors
/// one qwen `Engine` method the [`Scheduler`] used to call directly (see
/// `crates/qwen3/src/serve.rs`'s `impl PagedDecoder for Engine`) — this trait
/// is the exact set [`Scheduler`] needs and nothing more.
pub trait PagedDecoder {
    /// Mutable access to the block allocator, for callers (here, [`Scheduler`])
    /// that must pass it alongside a [`BlockTable`] to a paged-cache operation
    /// (`BlockTable::truncate`/`reserve`) the decoder itself doesn't wrap.
    fn alloc_mut(&mut self) -> &mut BlockAllocator;

    /// Max prompt tokens the engine will prefill in one scheduler iteration
    /// before yielding to decode — the default per-iteration prefill budget.
    fn max_prefill_tokens(&self) -> u32;

    /// KV blocks currently free in the pool.
    fn free_blocks(&self) -> u32;

    /// The largest `prompt + max_new` a single sequence's capacity admits.
    fn max_seq_len(&self) -> usize;

    /// Vocabulary size — prompt tokens at or above this are rejected at
    /// admission rather than gathered out of bounds.
    fn vocab(&self) -> usize;

    /// KV blocks needed to hold `tokens` cached positions.
    fn blocks_for(&self, tokens: u32) -> u32;

    /// Reclaim up to `want` cache-only prefix blocks back to the pool (live
    /// sequences always outrank the prefix cache). Returns blocks actually freed.
    fn reclaim_prefix(&mut self, want: u32) -> u32;

    /// Release a finished/cancelled sequence's blocks back to the pool.
    fn release_table(&mut self, t: &mut BlockTable);

    /// Prefill a fresh prompt into `table`, returning the final hidden state
    /// (from which the caller samples the first token).
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32>;

    /// `logits = hidden @ head^T` for [`prefill`](Self::prefill)'s hidden
    /// state - a full `[vocab]` vector. [`admit_greedy`](Self::admit_greedy)/
    /// [`admit_topk`](Self::admit_topk) are what admission actually calls;
    /// this method is their host-side fallback (and whatever else a decoder
    /// needs a raw logits vector for) so it stays required rather than
    /// folded into the two admission methods.
    fn logits(&self, hidden: &[f32]) -> Vec<f32>;

    /// Admission's greedy pick for [`prefill`](Self::prefill)'s hidden state.
    /// The default is a plain host argmax over [`logits`](Self::logits)'s
    /// full vocabulary vector; a decoder with an on-device head
    /// (`qwen3::serve::Engine`) overrides this to never ship that vector to
    /// the host at all.
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        argmax(&self.logits(hidden))
    }

    /// Admission's non-greedy candidates (token id, logit) for
    /// [`prefill`](Self::prefill)'s hidden state, best first, clamped to
    /// [`Self::topk_capacity`]. The default sorts [`logits`](Self::logits)'s
    /// full vocabulary vector on the host; a decoder with on-device top-k
    /// (`qwen3::serve::Engine`) overrides this to never ship that vector to
    /// the host at all.
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut candidates: Vec<(u32, f32)> =
            self.logits(hidden).into_iter().enumerate().map(|(i, v)| (i as u32, v)).collect();
        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(k.max(1));
        candidates
    }

    /// One batched greedy decode step over every active sequence's current
    /// input, returning each sequence's next token.
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32>;

    /// The on-device decode window: `k` greedy steps per host round-trip.
    /// Returns, per sequence, the (up to `k`) tokens it produced.
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>>;

    /// `(tokens served from cache, tokens looked up, cached blocks)`.
    fn prefix_stats(&self) -> (u64, u64, usize);

    /// Device-op accounting (submits/dispatches/readbacks), where the backend counts it.
    fn device_stats(&self) -> Option<gpu_core::DeviceStats>;

    /// Device bytes the paged KV pool costs at this decoder's sizing — a
    /// REQUIRED method (unlike `device_stats`, which is genuinely `Option`
    /// because some backends don't count): this is an analytic quantity
    /// every `PagedDecoder` can derive from its own pool geometry, computed
    /// before any device allocation happens (so residency budgeting can use
    /// it as a prediction, not a postmortem) — see
    /// `qwen3::serve::kv_pool_bytes`, the one implementation today.
    fn kv_pool_bytes(&self) -> u64;

    /// The pool's total theoretical cached-token capacity (`num_blocks *
    /// block_size`), independent of dtype — see
    /// `qwen3::serve::Engine::kv_pool_capacity_tokens`.
    fn kv_pool_capacity_tokens(&self) -> u64;

    /// The largest `k` [`forward_batched_greedy_window`](Self::forward_batched_greedy_window)
    /// may be called with — the decoder's own on-device decode-window scratch
    /// capacity. The scheduler must never request more: it is not a tunable
    /// policy knob, it is the bound the decoder's device buffers were sized to.
    fn decode_window_capacity(&self) -> usize;

    /// One batched decode step returning, per row, its top-`k` (token id,
    /// logit) candidates, best first — for a caller doing real (non-greedy)
    /// sampling without reading back the whole `[bsz, vocab]` row. `k` is
    /// clamped to [`Self::topk_capacity`].
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>>;

    /// The largest `k` [`forward_batched_topk`](Self::forward_batched_topk) will
    /// honor — the decoder's on-device top-K extraction scratch capacity.
    fn topk_capacity(&self) -> usize;

    // ---- host-RAM KV offload (optional; see `crate::kv_offload`) ----
    //
    // Every method below has a default that reports "this decoder cannot
    // offload", so an existing `PagedDecoder` implementation keeps compiling
    // and behaving exactly as it did. A decoder that CAN offload implements
    // `crate::kv_offload::KvOffload` over its own pool buffers and forwards
    // these to it - see `qwen3::serve::Engine`.

    /// Host bytes available to hold demoted sequences' KV. `0` (the default)
    /// disables preemption-by-swap in [`Scheduler`] entirely.
    fn kv_offload_bytes(&self) -> u64 {
        0
    }

    /// Copy this sequence's whole KV to host RAM and free its device blocks,
    /// returning how many blocks the pool got back.
    fn demote_sequence(&mut self, key: u64, table: &mut BlockTable) -> Result<u32, crate::kv_offload::KvOffloadError> {
        let _ = (key, table);
        Err(crate::kv_offload::KvOffloadError::Unsupported)
    }

    /// Bring a demoted sequence back onto the device, exact bytes, resuming at
    /// the length it was demoted at.
    fn promote_sequence(&mut self, key: u64) -> Result<BlockTable, crate::kv_offload::KvOffloadError> {
        let _ = key;
        Err(crate::kv_offload::KvOffloadError::Unsupported)
    }

    /// Device blocks a demoted sequence needs to come back, `None` if it is not
    /// demoted - what the scheduler checks before attempting a promote.
    fn offloaded_blocks(&self, key: u64) -> Option<u32> {
        let _ = key;
        None
    }

    /// Drop a demoted sequence's host KV without restoring it (cancellation).
    fn discard_offloaded(&mut self, key: u64) -> bool {
        let _ = key;
        false
    }

    /// Swap accounting, for a server or benchmark to report.
    fn offload_stats(&self) -> crate::kv_offload::OffloadStats {
        crate::kv_offload::OffloadStats::default()
    }
}

/// Temperature / top-k / top-p sampling parameters for one sequence.
/// `SampleParams::greedy()` (the default) reproduces today's argmax-only
/// behaviour exactly, via the same fast, already-gated batched-greedy path —
/// [`Scheduler`] only takes the (slower, no on-device window) top-K sampling
/// path when at least one ACTIVE sequence in the current iteration wants real
/// sampling.
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    /// `<= 0.0` is greedy (argmax), matching `qwen3::sample::sample_logits`.
    pub temp: f32,
    /// `0` (or `>=` the decoder's candidate list) disables top-k filtering.
    pub top_k: usize,
    /// `<= 0.0` or `>= 1.0` disables nucleus filtering.
    pub top_p: f32,
}

impl Default for SampleParams {
    fn default() -> SampleParams {
        SampleParams::greedy()
    }
}

impl SampleParams {
    pub fn greedy() -> SampleParams {
        SampleParams { temp: 0.0, top_k: 0, top_p: 1.0 }
    }
    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0
    }
}

/// Sample one token from a row's sorted-descending top-K `(token id, logit)`
/// candidates — temperature scale, truncate to `top_k` (a no-op re-sort since
/// the list already arrives sorted), softmax, nucleus-truncate to `top_p`,
/// then draw via inverse-CDF. A light, self-contained duplicate of
/// `qwen3::sample::sample_logits`'s algorithm (pure math, no qwen dependency)
/// operating over a candidate LIST rather than a full vocab vector — the
/// candidate list is what makes this cheap enough to run per decode step
/// without shipping `[bsz, vocab]` back to the host. Candidates beyond
/// `candidates.len()` cannot be represented (the decoder's `topk_capacity`
/// bounds how wide the nucleus can ever be); this is a documented ceiling on
/// `top_p`'s width, not a bug.
pub fn sample_from_topk(candidates: &[(u32, f32)], params: SampleParams, rng: &mut data::rng::Rng) -> u32 {
    assert!(!candidates.is_empty(), "sample_from_topk: no candidates");
    if params.is_greedy() {
        return candidates[0].0;
    }
    let temp = params.temp.max(1e-6);
    let n = if params.top_k > 0 { params.top_k.min(candidates.len()) } else { candidates.len() };
    let mut probs: Vec<f32> = candidates[..n].iter().map(|&(_, v)| v / temp).collect();
    let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = (*p - max).exp();
        sum += *p;
    }
    // Nucleus (top-p): candidates already arrive sorted descending, so the
    // cumulative-mass prefix IS the nucleus — no re-sort needed.
    let mut cut = n;
    if params.top_p > 0.0 && params.top_p < 1.0 && sum > 0.0 {
        let mut kept = 0.0f32;
        for (rank, p) in probs.iter().enumerate() {
            kept += p;
            if kept / sum >= params.top_p {
                cut = rank + 1;
                break;
            }
        }
    }
    let kept_sum: f32 = probs[..cut].iter().sum();
    let r = rng.next_f32() * kept_sum;
    let mut acc = 0.0f32;
    for (i, &p) in probs[..cut].iter().enumerate() {
        acc += p;
        if acc >= r {
            return candidates[i].0;
        }
    }
    candidates[cut - 1].0
}

/// Greedy argmax of a logits/score vector — pure host math, no decoder
/// dependency, so it is a free function rather than a [`PagedDecoder`] method.
fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in s.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

/// A submitted generation request.
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub eos: Option<u32>,
}

/// Why a request was refused at admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Can never fit the engine's per-sequence capacity.
    ExceedsCapacity { need: u32, capacity: u32 },
    /// Refused by the installed [`AdmissionPolicy`].
    PolicyRejected { policy: &'static str },
    /// A prompt token outside the model's vocabulary. Admitting it would make
    /// the embedding gather read out of bounds — the kernels are trusted (no
    /// per-access clamps), so the failure would be silent garbage, not an
    /// error.
    InvalidToken { token: u32, vocab: u32 },
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::ExceedsCapacity { need, capacity } => {
                write!(f, "needs {need} tokens, engine capacity is {capacity}")
            }
            RejectReason::PolicyRejected { policy } => write!(f, "rejected by {policy}"),
            RejectReason::InvalidToken { token, vocab } => {
                write!(f, "token {token} is outside the vocabulary ({vocab})")
            }
        }
    }
}

/// What the queue looks like when an admission decision is made.
#[derive(Clone, Copy, Debug)]
pub struct QueueState {
    /// Requests waiting behind this one (its position in the queue).
    pub queued_ahead: usize,
    /// Sequences currently decoding.
    pub running: usize,
    /// KV blocks free in the pool.
    pub free_blocks: u32,
    /// Observed mean milliseconds to serve one request, when known.
    pub mean_service_ms: Option<f64>,
}

/// Decide what to do with work that arrives beyond capacity.
///
/// `perf overload` measured the default (queue without bound) collapsing once
/// offered load ran past capacity: goodput fell away because compute was spent on
/// answers past their deadline. An engine is rewarded for refusing work it
/// provably cannot finish in time; policies are pure functions of
/// [`QueueState`], unit-testable with no engine at all.
pub trait AdmissionPolicy: Send + Sync {
    fn name(&self) -> &'static str;
    /// May this request enter the queue / stay admissible?
    fn admit(&self, req: &Request, state: &QueueState) -> bool;
}

/// Queue without bound — the historical behaviour and the default.
pub struct UnboundedQueue;
impl AdmissionPolicy for UnboundedQueue {
    fn name(&self) -> &'static str {
        "unbounded_queue"
    }
    fn admit(&self, _req: &Request, _state: &QueueState) -> bool {
        true
    }
}

/// Refuse once more than `max` requests are already waiting.
pub struct MaxQueueDepth(pub usize);
impl AdmissionPolicy for MaxQueueDepth {
    fn name(&self) -> &'static str {
        "max_queue_depth"
    }
    fn admit(&self, _req: &Request, state: &QueueState) -> bool {
        state.queued_ahead < self.0
    }
}

/// Refuse work that provably cannot start inside its deadline: everything
/// ahead must clear first, and if that alone exceeds the budget the compute
/// would be spent on an answer nobody can use.
pub struct DeadlineAware {
    /// Per-request start deadline, ms.
    pub deadline_ms: f64,
}
impl AdmissionPolicy for DeadlineAware {
    fn name(&self) -> &'static str {
        "deadline_aware"
    }
    fn admit(&self, _req: &Request, state: &QueueState) -> bool {
        match state.mean_service_ms {
            Some(svc) => (state.queued_ahead as f64) * svc <= self.deadline_ms,
            None => true, // nothing measured yet — cannot prove lateness
        }
    }
}

/// What one [`Scheduler::step_report`] iteration did. Latency metrics
/// (time-to-first-token, inter-token latency) are computed from this: [`Scheduler::step`]
/// alone reports only *completions*, which is too coarse to see when a sequence
/// was admitted or when each token landed, so no caller can derive TTFT/ITL from it.
#[derive(Debug, Default)]
pub struct StepReport {
    /// Requests admitted (prefilled + first token sampled) this iteration.
    pub admitted: Vec<u64>,
    /// `(id, tokens produced this iteration)` — the first token counts on the
    /// iteration the request was admitted.
    pub produced: Vec<(u64, usize)>,
    /// Requests that finished this iteration.
    pub finished: Vec<u64>,
    /// Requests refused at admission — impossible sizes and policy rejections
    /// alike. Refusing beats both crashing and queueing forever.
    pub rejected: Vec<(u64, RejectReason)>,
    /// The same `(id, tokens)` pairs [`Scheduler::step`] returns.
    pub completed: Vec<(u64, Vec<u32>)>,
    /// Requests preempted this iteration: their KV went to host RAM and their
    /// device blocks back to the pool (see [`crate::kv_offload`]).
    pub demoted: Vec<u64>,
    /// Requests whose KV came back from host RAM this iteration.
    pub promoted: Vec<u64>,
}

/// A sequence the scheduler is actively decoding.
struct Running {
    id: u64,
    table: BlockTable,
    generated: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    next_input: u32,
    done: bool,
    /// This sequence's sampling params + its own RNG stream. Kept per-sequence
    /// (not per-batch) so a `seed` reproduces exactly regardless of what else
    /// is in the batch alongside it.
    sample: SampleParams,
    rng: data::rng::Rng,
}

/// Accept a freshly sampled token into `r`.
///
/// If `next` is the EOS token, `r` is marked done WITHOUT appending it to
/// `generated` — EOS is a control signal that ends the sequence, not output
/// text. Appending it (the previous behavior) meant the token's decoded text
/// (e.g. Qwen's `<|im_end|>`) leaked into every streamed response, since
/// `generated`/`tokens_of` feed the text returned to callers. Otherwise
/// `next` is appended normally, and `r` is marked done if `max_new` is now
/// reached. A free function (not `Scheduler<D>::accept_token`) because it
/// only ever touches `Running`, never `D`/`self`.
fn accept_token(r: &mut Running, next: u32) {
    if Some(next) == r.eos {
        r.done = true;
        return;
    }
    r.generated.push(next);
    r.next_input = next;
    if r.generated.len() >= r.max_new {
        r.done = true;
    }
}

/// **Continuous-batching scheduler.** Requests are submitted at any time, admitted
/// when the KV pool + batch have room (prefilled + first token sampled), then every
/// running sequence advances together in one batched decode step per iteration.
/// Finished sequences return their blocks immediately, so newly submitted requests
/// can be admitted mid-flight — the batch composition changes each iteration to keep
/// as much useful work resident as possible. Generic over the model's own
/// [`PagedDecoder`]; `qwen3::serve::Scheduler` is `Scheduler<qwen3::serve::Engine>`.
pub struct Scheduler<D: PagedDecoder> {
    dec: D,
    waiting: VecDeque<(u64, Request, SampleParams, u64)>,
    running: Vec<Running>,
    next_id: u64,
    max_running: usize,
    /// Admission policy — what to do with work arriving beyond capacity.
    admission: Box<dyn AdmissionPolicy>,
    /// EWMA of ms per completed request, feeding DeadlineAware decisions.
    mean_service_ms: Option<f64>,
    started: HashMap<u64, std::time::Instant>,
    /// Policy rejections made at submit time, surfaced in the next report.
    pending_rejects: Vec<(u64, RejectReason)>,
    /// Max prompt tokens prefilled per iteration before yielding to decode.
    ///
    /// Admission runs a FULL prefill per accepted request, so without a budget
    /// a burst of N arrivals performs N whole prompt forwards back-to-back
    /// while every running sequence stalls - measured as TTFA and inter-token
    /// tail latency each growing by an order of magnitude from concurrency 1 to
    /// 32, with the interactive SLO met at no concurrency level. Bounding the
    /// prefill work per iteration lets decode run every iteration and spreads
    /// a burst across several; the budget always admits at least one waiting
    /// request per iteration, so nothing can starve.
    prefill_budget: u32,
    /// Admitted sequences whose KV currently lives in host RAM
    /// ([`crate::kv_offload`]), waiting for device blocks to come back. FIFO:
    /// the sequence demoted longest ago resumes first.
    ///
    /// These are NOT waiting requests - they have been prefilled, they have
    /// produced tokens, and their whole cache exists; only their device blocks
    /// were taken away. That is why admission stops while any of them is
    /// outstanding: preferring new work over work already in flight would let
    /// a busy queue starve a preempted sequence for as long as it stays busy.
    swapped: VecDeque<Running>,
}

impl<D: PagedDecoder> Scheduler<D> {
    pub fn new(dec: D, max_running: usize) -> Scheduler<D> {
        // Default budget: two full prefill chunks per iteration. Enough to keep
        // admission moving under load, small enough that running sequences see
        // a decode step between arrivals.
        let prefill_budget = dec.max_prefill_tokens().saturating_mul(2).max(1);
        Scheduler {
            dec,
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            max_running,
            admission: Box::new(UnboundedQueue),
            mean_service_ms: None,
            started: HashMap::new(),
            pending_rejects: Vec::new(),
            prefill_budget,
            swapped: VecDeque::new(),
        }
    }

    /// Install an admission policy (default: [`UnboundedQueue`], the historical
    /// behaviour). Applied at submit time; a refused request is reported in the
    /// next iteration's [`StepReport::rejected`].
    pub fn set_admission(&mut self, p: Box<dyn AdmissionPolicy>) {
        self.admission = p;
    }

    /// Override the per-iteration prefill budget (tokens). `u32::MAX` restores
    /// the old admit-everything behaviour; recorded by `brain perf` in the
    /// artifact's target config so a run states the policy it used.
    pub fn set_prefill_budget(&mut self, tokens: u32) {
        self.prefill_budget = tokens.max(1);
    }

    /// Enqueue a request; returns its id (results come back keyed by it).
    /// Submit a request. The admission policy is consulted HERE — a refusal
    /// returns the id with the request never queued, and the rejection appears
    /// in the next [`Scheduler::step_report`]. Greedy — identical to
    /// `submit_sampled(req, SampleParams::greedy(), 0)`, kept as the
    /// zero-argument-change entry point every existing caller already uses.
    pub fn submit(&mut self, req: Request) -> u64 {
        self.submit_sampled(req, SampleParams::greedy(), 0)
    }

    /// [`Self::submit`] with real (non-greedy) sampling params and this
    /// sequence's own RNG seed — reproducible regardless of what else is in
    /// the batch alongside it, since each `Running` sequence carries its own
    /// RNG stream (see `Running::rng`), never a batch-shared one.
    pub fn submit_sampled(&mut self, req: Request, sample: SampleParams, seed: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let state = QueueState {
            queued_ahead: self.waiting.len(),
            running: self.running.len(),
            free_blocks: self.dec.free_blocks(),
            mean_service_ms: self.mean_service_ms,
        };
        if !self.admission.admit(&req, &state) {
            self.pending_rejects.push((id, RejectReason::PolicyRejected { policy: self.admission.name() }));
            return id;
        }
        self.started.insert(id, std::time::Instant::now());
        self.waiting.push_back((id, req, sample, seed));
        id
    }

    /// Cancel a request: drop it whether queued or mid-decode and return its KV
    /// blocks to the pool immediately. Returns the tokens produced so far, or
    /// `None` if the id is unknown (already finished, or never submitted).
    ///
    /// Without this, an abandoned request keeps decoding to `max_new` — spending
    /// device time on output nobody will read, and holding KV blocks that
    /// requests still being waited on need. Reclaiming on cancel is what stops a
    /// server under normal churn from losing its cache to dead sequences.
    pub fn cancel(&mut self, id: u64) -> Option<Vec<u32>> {
        if let Some(pos) = self.waiting.iter().position(|(qid, ..)| *qid == id) {
            self.waiting.remove(pos);
            return Some(Vec::new()); // never admitted, so nothing was produced
        }
        if let Some(pos) = self.swapped.iter().position(|r| r.id == id) {
            // Demoted: its blocks are already back in the pool, its bytes are
            // in host RAM. Dropping the host record is the whole reclaim.
            let r = self.swapped.remove(pos).expect("position just found it");
            self.dec.discard_offloaded(id);
            return Some(r.generated);
        }
        let pos = self.running.iter().position(|r| r.id == id)?;
        let mut r = self.running.remove(pos);
        self.dec.release_table(&mut r.table);
        Some(r.generated)
    }

    /// Requests currently admitted and decoding.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Requests admitted but currently preempted, their KV held in host RAM
    /// (see [`crate::kv_offload`]). They resume - exact bytes, same length -
    /// as soon as device blocks are available.
    pub fn swapped_len(&self) -> usize {
        self.swapped.len()
    }

    /// Swap accounting - see [`PagedDecoder::offload_stats`].
    pub fn offload_stats(&self) -> crate::kv_offload::OffloadStats {
        self.dec.offload_stats()
    }

    /// Consume the scheduler and hand the decoder back, releasing every
    /// sequence it still holds (running blocks to the pool, preempted KV out
    /// of host RAM) so the decoder comes back with nothing leaked.
    ///
    /// The seam a caller needs to reconfigure a live engine between workloads:
    /// `Scheduler` owns its decoder, and rebuilding one to change a policy
    /// knob would mean reloading the weights.
    pub fn into_decoder(mut self) -> D {
        for r in &mut self.running {
            self.dec.release_table(&mut r.table);
        }
        for r in &self.swapped {
            self.dec.discard_offloaded(r.id);
        }
        self.dec
    }

    /// All tokens generated so far for a still-running sequence (`None` if
    /// `id` is unknown, queued, or already reaped) — the seam a streaming
    /// caller uses to emit the NEW suffix each iteration without waiting for
    /// the sequence to finish (`step`/`step_report` only return a completed
    /// sequence's full token list, not a running one's partial progress).
    pub fn tokens_of(&self, id: u64) -> Option<&[u32]> {
        // Demoted sequences are included: preemption is invisible to a caller
        // streaming a response, so its progress must stay readable while its
        // KV sits in host RAM.
        self.running
            .iter()
            .chain(self.swapped.iter())
            .find(|r| r.id == id)
            .map(|r| r.generated.as_slice())
    }

    /// Requests admitted but not yet started.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// True while any request is waiting, running, or preempted to host RAM.
    pub fn pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }


    /// One scheduler iteration, reporting **everything that happened** — not just
    /// what finished.
    ///
    /// [`Scheduler::step`] returns only completed requests, which is all a caller
    /// collecting outputs needs but leaves per-request latency unobservable: with
    /// completions alone you cannot tell when a sequence was admitted or when
    /// each token appeared, so time-to-first-token and inter-token latency cannot
    /// be computed at all. This variant additionally reports admissions and
    /// per-sequence token counts, which is what `brain perf` measures.
    pub fn step_report(&mut self) -> StepReport {
        let mut report = StepReport::default();
        report.rejected.append(&mut self.pending_rejects);
        let produced_before: HashMap<u64, usize> =
            self.running.iter().map(|r| (r.id, r.generated.len())).collect();

        let completed = self.step_inner(&mut report);

        // Tokens produced this iteration by sequences that are still running...
        for r in &self.running {
            let prev = produced_before.get(&r.id).copied().unwrap_or(0);
            if r.generated.len() > prev {
                report.produced.push((r.id, r.generated.len() - prev));
            }
        }
        // ...and by those that finished in this iteration.
        for (id, toks) in &completed {
            let prev = produced_before.get(id).copied().unwrap_or(0);
            if toks.len() > prev {
                report.produced.push((*id, toks.len() - prev));
            }
            report.finished.push(*id);
            if let Some(t0) = self.started.remove(id) {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                self.mean_service_ms =
                    Some(self.mean_service_ms.map_or(ms, |m| 0.8 * m + 0.2 * ms));
            }
        }
        report.completed = completed;
        report
    }

    /// The number of KV blocks still free in the pool — the memory-pressure
    /// signal a benchmark records alongside its latencies.
    pub fn free_blocks(&self) -> u32 {
        self.dec.free_blocks()
    }

    /// Prefix-cache effectiveness — see [`PagedDecoder::prefix_stats`].
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        self.dec.prefix_stats()
    }

    /// Device-op accounting — see [`PagedDecoder::device_stats`].
    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.dec.device_stats()
    }

    /// KV pool byte cost — see [`PagedDecoder::kv_pool_bytes`].
    pub fn kv_pool_bytes(&self) -> u64 {
        self.dec.kv_pool_bytes()
    }

    /// KV pool token capacity — see [`PagedDecoder::kv_pool_capacity_tokens`].
    pub fn kv_pool_capacity_tokens(&self) -> u64 {
        self.dec.kv_pool_capacity_tokens()
    }

    /// One scheduler iteration: admit waiting requests that fit (prefill + sample
    /// first token), run one batched decode step over all running sequences, then
    /// reap completed ones. Returns the `(id, tokens)` of requests finished here.
    pub fn step(&mut self) -> Vec<(u64, Vec<u32>)> {
        let mut sink = StepReport::default();
        self.step_inner(&mut sink)
    }

    /// Bring preempted sequences back onto the device, oldest demotion first,
    /// while the pool can hold them and the batch has room. A promote that
    /// cannot fit yet leaves the sequence demoted and stops the loop - the
    /// queue is in demotion order, so letting a later (smaller) sequence jump
    /// ahead of a stuck one is how a long sequence gets starved.
    fn resume_swapped(&mut self, report: &mut StepReport) {
        while !self.swapped.is_empty() && self.running.len() < self.max_running {
            let id = self.swapped.front().expect("non-empty").id;
            // Its own blocks back, PLUS what the sequences already decoding
            // need to append this iteration, plus one for its own next token.
            // Promoting on the bare restore cost alone would hand the pool
            // straight back to `ensure_decode_room`, which would preempt
            // something (possibly this very sequence) again - two transfers
            // for no token.
            //
            // With NOTHING running there is no such trade to make, and the
            // slack is dropped: a sequence whose KV nearly fills the pool must
            // still come back, or a queue that emptied around it would spin
            // forever refusing to restore the one thing left to do.
            let slack = if self.running.is_empty() { 0 } else { self.blocks_for_next_token() + 1 };
            let need = self.dec.offloaded_blocks(id).unwrap_or(0) + slack;
            if self.dec.free_blocks() < need {
                // Cached prefix blocks are reclaimable capacity; a sequence
                // that already exists outranks the prefix cache exactly as
                // admission does.
                self.dec.reclaim_prefix(need - self.dec.free_blocks());
                if self.dec.free_blocks() < need {
                    break;
                }
            }
            let mut r = self.swapped.pop_front().expect("non-empty");
            match self.dec.promote_sequence(r.id) {
                Ok(table) => {
                    r.table = table;
                    report.promoted.push(r.id);
                    self.running.push(r);
                }
                Err(crate::kv_offload::KvOffloadError::DevicePoolExhausted { .. }) => {
                    // Still no room (the estimate above raced the pool): put it
                    // back at the head and try again next iteration.
                    self.swapped.push_front(r);
                    break;
                }
                Err(_) => {
                    // The host record is gone or the decoder cannot offload at
                    // all: neither is retryable, and looping on it would hang
                    // the scheduler. Finish the sequence with what it produced.
                    r.done = true;
                    self.running.push(r);
                }
            }
        }
    }

    /// Device blocks the running batch needs to append one more token each.
    fn blocks_for_next_token(&self) -> u32 {
        self.running
            .iter()
            .filter(|r| !r.done)
            .map(|r| self.dec.blocks_for(r.table.len() + 1).saturating_sub(r.table.blocks().len() as u32))
            .sum()
    }

    /// Preempt until the pool can back one more token for every active
    /// sequence, or until nothing more can be preempted.
    ///
    /// The last active sequence is never demoted: it has nothing to yield to,
    /// and swapping it out could only be undone by swapping it back in.
    /// Without host offload (`kv_offload_bytes() == 0`, every decoder that has
    /// not adopted it) this reduces to the prefix reclaim that was already
    /// here, so behaviour is unchanged.
    fn ensure_decode_room(&mut self, report: &mut StepReport) {
        let mut need = self.blocks_for_next_token();
        if self.dec.free_blocks() >= need {
            return;
        }
        self.dec.reclaim_prefix(need - self.dec.free_blocks());
        while self.dec.free_blocks() < need {
            if self.running.iter().filter(|r| !r.done).count() <= 1 {
                return;
            }
            // Never re-demote a sequence promoted in THIS iteration: it is
            // the newest entry in `running`, so the plain "newest first"
            // choice would pick it, pay both transfers and produce no token -
            // pure thrash. Anything else, newest first.
            let Some(pos) = self.running.iter().rposition(|r| !r.done && !report.promoted.contains(&r.id)) else { return };
            let mut r = self.running.remove(pos);
            match self.dec.demote_sequence(r.id, &mut r.table) {
                Ok(_) => {
                    report.demoted.push(r.id);
                    self.swapped.push_back(r);
                }
                Err(_) => {
                    // No offload, or no host budget left: put it back and stop.
                    // The decode below runs exactly as it did before this
                    // existed.
                    self.running.insert(pos, r);
                    return;
                }
            }
            need = self.blocks_for_next_token();
        }
    }

    fn step_inner(&mut self, report: &mut StepReport) -> Vec<(u64, Vec<u32>)> {
        // 0. Resume preempted sequences before anything else touches the pool:
        //    work already in flight outranks work still queued.
        self.resume_swapped(report);

        // 1. Admit while there's batch room, enough free blocks for the prompt,
        //    and prefill budget left this iteration (head-of-line guard: decode
        //    must run between bursts of admissions). Nothing new is admitted
        //    while a preempted sequence is still waiting for its blocks - see
        //    `Scheduler::swapped`.
        let mut budget_left = self.prefill_budget;
        let mut admitted_this_iter = 0u32;
        while self.swapped.is_empty() && self.running.len() < self.max_running {
            // Drop anything that can never fit, whatever the pool does — it
            // would otherwise block the queue forever (or, before the capacity
            // check, corrupt the block table).
            let cap = self.dec.max_seq_len();
            let vocab = self.dec.vocab() as u32;
            while let Some((id, req, ..)) = self.waiting.front() {
                let need = req.prompt.len() + req.max_new;
                let bad = req.prompt.iter().find(|&&t| t >= vocab).copied();
                if need <= cap && bad.is_none() {
                    break;
                }
                let (id, need) = (*id, need as u32);
                self.waiting.pop_front();
                self.started.remove(&id);
                let reason = match bad {
                    Some(token) => RejectReason::InvalidToken { token, vocab },
                    None => RejectReason::ExceedsCapacity { need, capacity: cap as u32 },
                };
                report.rejected.push((id, reason));
            }
            let fits = match self.waiting.front() {
                Some((_, req, ..)) => {
                    let need = req.prompt.len() as u32;
                    // Always admit at least one request per iteration (no
                    // starvation); after that, stop once the budget is spent.
                    if admitted_this_iter > 0 && need > budget_left {
                        break;
                    }
                    let want = self.dec.blocks_for(need + 1);
                    let free = self.dec.free_blocks();
                    if free < want {
                        // Cached prefix blocks are reclaimable capacity: live
                        // sequences always outrank the cache.
                        self.dec.reclaim_prefix(want - free);
                    }
                    self.dec.free_blocks() >= want
                }
                None => false,
            };
            if !fits {
                break;
            }
            let (id, req, sample, seed) = self.waiting.pop_front().unwrap();
            budget_left = budget_left.saturating_sub(req.prompt.len() as u32);
            admitted_this_iter += 1;
            let mut table = BlockTable::new();
            let hidden = self.dec.prefill(&mut table, &req.prompt);
            let mut rng = data::rng::Rng::new(seed);
            // `admit_greedy`/`admit_topk` are on-device for a decoder that has
            // one (`qwen3::serve::Engine`): neither ships a `[vocab]` block to
            // the host for this one-time-per-request first token, matching
            // the steady-state per-token decode loop below.
            let first = if sample.is_greedy() {
                self.dec.admit_greedy(&hidden)
            } else {
                let k = self.dec.topk_capacity().max(1);
                let candidates = self.dec.admit_topk(&hidden, k);
                sample_from_topk(&candidates, sample, &mut rng)
            };
            let mut r = Running { id, table, generated: Vec::new(), max_new: req.max_new, eos: req.eos, next_input: first, done: false, sample, rng };
            accept_token(&mut r, first);
            report.admitted.push(id);
            self.running.push(r);
        }

        // 2. Make room for this iteration's decode. Admission only checks that
        //    a PROMPT fits; a sequence that has been decoding for a while can
        //    still walk the pool dry mid-batch, and `BlockTable::append` has
        //    nowhere to go when it does. Preempt (swap out) the most recently
        //    admitted sequences until every remaining one can append its next
        //    token - the classic "newest first" victim choice, so the sequences
        //    closest to finishing (and to releasing their blocks) are the ones
        //    that keep running.
        self.ensure_decode_room(report);

        // 3. Batched decode over every running (not-done) sequence. When
        //    nothing is waiting to be admitted, decode a WINDOW of tokens per
        //    host round-trip (A4): the readback-per-token becomes a readback
        //    per window, at the cost of up to window-1 wasted decode steps for
        //    a sequence that hits EOS mid-window (its surplus K/V is rolled
        //    back below). With work waiting, the window stays 1 so admission
        //    latency is never traded away silently.
        let active: Vec<usize> = (0..self.running.len()).filter(|&i| !self.running[i].done).collect();
        if !active.is_empty() {
            let inputs: Vec<u32> = active.iter().map(|&i| self.running[i].next_input).collect();
            let remaining_min = active
                .iter()
                .map(|&i| {
                    let r = &self.running[i];
                    r.max_new.saturating_sub(r.generated.len()).max(1)
                })
                .min()
                .unwrap_or(1);
            // Real (non-greedy) sampling has no on-device window yet (W3's
            // first cut): a batch containing ANY sampling row falls back to
            // k=1 for everyone this iteration. Purely-greedy batches (today's
            // only case, and the common one) take the exact path they always
            // did, untouched.
            let all_greedy = active.iter().all(|&i| self.running[i].sample.is_greedy());
            let mut k = if self.waiting.is_empty() && all_greedy { remaining_min.min(self.dec.decode_window_capacity()) } else { 1 };
            // Every append must succeed mid-window (no host decisions there):
            // require a comfortable block reserve, else fall back to one step.
            if k > 1 && (self.dec.free_blocks() as usize) < active.len() * k {
                k = 1;
            }
            let window: Vec<Vec<u32>> = if all_greedy {
                let mut refs: Vec<&mut BlockTable> = Vec::new();
                for (idx, r) in self.running.iter_mut().enumerate() {
                    if active.contains(&idx) {
                        refs.push(&mut r.table);
                    }
                }
                if k > 1 {
                    self.dec.forward_batched_greedy_window(&mut refs, &inputs, k)
                } else {
                    self.dec
                        .forward_batched_greedy(&mut refs, &inputs)
                        .into_iter()
                        .map(|t| vec![t])
                        .collect()
                }
            } else {
                let candidates = {
                    let mut refs: Vec<&mut BlockTable> = Vec::new();
                    for (idx, r) in self.running.iter_mut().enumerate() {
                        if active.contains(&idx) {
                            refs.push(&mut r.table);
                        }
                    }
                    self.dec.forward_batched_topk(&mut refs, &inputs, self.dec.topk_capacity())
                };
                active
                    .iter()
                    .enumerate()
                    .map(|(bi, &si)| {
                        let r = &mut self.running[si];
                        vec![sample_from_topk(&candidates[bi], r.sample, &mut r.rng)]
                    })
                    .collect()
            };
            for (bi, &si) in active.iter().enumerate() {
                let r = &mut self.running[si];
                let mut used = 0usize;
                for &next in &window[bi] {
                    // `used` counts consumed window slots (a real decode step
                    // happened and wrote a KV-cache entry) regardless of
                    // whether `next` ends up in `generated` - EOS still
                    // occupies a real slot, it just isn't output text.
                    used += 1;
                    accept_token(r, next);
                    if r.done {
                        break;
                    }
                }
                // A sequence that finished mid-window consumed only `used`
                // inputs; the remaining pre-allocated slots hold garbage K/V
                // and are rolled back so the pool never leaks waste.
                let surplus = (k - used) as u32;
                if surplus > 0 {
                    let len = r.table.len();
                    r.table.truncate(len - surplus, self.dec.alloc_mut());
                }
            }
        }

        // 4. Reap completed sequences, returning their blocks to the pool.
        let mut completed = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].done {
                let mut r = self.running.remove(i);
                self.dec.release_table(&mut r.table);
                completed.push((r.id, r.generated));
            } else {
                i += 1;
            }
        }
        completed
    }

    /// Drive to completion, returning every request's tokens keyed by id.
    pub fn run(&mut self) -> HashMap<u64, Vec<u32>> {
        let mut out = HashMap::new();
        while self.pending() {
            for (id, toks) in self.step() {
                out.insert(id, toks);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_from_topk_is_greedy_at_temp_zero() {
        let candidates = [(7u32, 3.0f32), (2, 5.0), (9, 1.0)];
        // Deliberately NOT pre-sorted -- greedy must return the FIRST entry
        // (the scheduler's real candidate lists are always sorted descending
        // by construction; this asserts the greedy short-circuit trusts that
        // contract rather than re-deriving it).
        let mut rng = data::rng::Rng::new(1);
        let params = SampleParams { temp: 0.0, top_k: 0, top_p: 1.0 };
        assert_eq!(sample_from_topk(&candidates, params, &mut rng), 7);
    }

    #[test]
    fn sample_from_topk_top_k_1_is_deterministically_the_best_candidate() {
        // top_k=1 over a positive-temperature softmax must degenerate to
        // picking the single highest-value candidate, regardless of the RNG
        // draw -- there is only one candidate left to draw from.
        let candidates = [(3u32, 9.0f32), (1, 4.0), (2, -1.0)];
        let params = SampleParams { temp: 0.7, top_k: 1, top_p: 1.0 };
        for seed in 0..8u64 {
            let mut rng = data::rng::Rng::new(seed);
            assert_eq!(sample_from_topk(&candidates, params, &mut rng), 3);
        }
    }

    #[test]
    fn sample_from_topk_never_returns_a_token_outside_the_candidate_set() {
        let candidates: Vec<(u32, f32)> = (0..40).map(|i| (100 + i, 40.0 - i as f32)).collect();
        let ids: std::collections::HashSet<u32> = candidates.iter().map(|&(id, _)| id).collect();
        let params = SampleParams { temp: 1.0, top_k: 10, top_p: 0.9 };
        let mut rng = data::rng::Rng::new(42);
        for _ in 0..200 {
            let t = sample_from_topk(&candidates, params, &mut rng);
            assert!(ids.contains(&t), "sampled token {t} was never in the candidate set");
        }
    }

    #[test]
    fn sample_from_topk_is_reproducible_for_a_fixed_seed() {
        let candidates: Vec<(u32, f32)> = (0..40).map(|i| (i, 40.0 - i as f32 * 0.3)).collect();
        let params = SampleParams { temp: 0.8, top_k: 20, top_p: 0.95 };
        let draw = |seed: u64| {
            let mut rng = data::rng::Rng::new(seed);
            (0..20).map(|_| sample_from_topk(&candidates, params, &mut rng)).collect::<Vec<u32>>()
        };
        assert_eq!(draw(1234), draw(1234));
        assert_ne!(draw(1234), draw(5678), "different seeds should not collide on 20 draws");
    }

    #[test]
    fn sample_from_topk_top_p_narrows_to_the_nucleus() {
        // A sharply peaked distribution: candidate 0 alone carries ~most of
        // the mass at temp=1. A tight top_p must keep drawing it; top_p=1
        // (disabled) must occasionally draw further down the tail.
        let candidates = [(0u32, 20.0f32), (1, 1.0), (2, 0.5), (3, 0.1)];
        let narrow = SampleParams { temp: 1.0, top_k: 0, top_p: 0.5 };
        let mut rng = data::rng::Rng::new(7);
        for _ in 0..50 {
            assert_eq!(sample_from_topk(&candidates, narrow, &mut rng), 0, "tight top_p must stay on the dominant candidate");
        }
    }

    // ── accept_token ─────────────────────────────────────────────

    fn running(eos: Option<u32>, max_new: usize) -> Running {
        Running {
            id: 0,
            table: BlockTable::new(),
            generated: Vec::new(),
            max_new,
            eos,
            next_input: 0,
            done: false,
            sample: SampleParams { temp: 0.0, top_k: 0, top_p: 1.0 },
            rng: data::rng::Rng::new(0),
        }
    }

    #[test]
    fn accept_token_eos_marks_done_without_appending_it() {
        // Regression test: the EOS token id must never end up in `generated`
        // (it feeds tokens_of()/completed, which the tokenizer decodes back
        // to text - an appended EOS id decodes to its literal special-token
        // text, e.g. Qwen's "<|im_end|>", leaking into every response).
        let mut r = running(Some(99), 100);
        r.generated.push(5); // a real token already generated this turn
        accept_token(&mut r, 99);
        assert!(r.done, "EOS must mark the sequence done");
        assert_eq!(r.generated, vec![5], "EOS must not be appended to generated");
    }

    #[test]
    fn accept_token_normal_token_is_appended_and_not_done() {
        let mut r = running(Some(99), 100);
        accept_token(&mut r, 42);
        assert!(!r.done);
        assert_eq!(r.generated, vec![42]);
        assert_eq!(r.next_input, 42);
    }

    #[test]
    fn accept_token_reaching_max_new_marks_done_and_still_appends() {
        // Unlike EOS, hitting the length cap on a real token must still
        // include that token in the output - only EOS is excluded.
        let mut r = running(Some(99), 1);
        accept_token(&mut r, 42);
        assert!(r.done);
        assert_eq!(r.generated, vec![42]);
    }

    #[test]
    fn accept_token_with_no_eos_configured_never_stops_on_token_value() {
        let mut r = running(None, 100);
        accept_token(&mut r, 99); // would be EOS if configured
        assert!(!r.done);
        assert_eq!(r.generated, vec![99]);
    }
}
