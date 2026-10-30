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

use crate::paged::{BlockAllocator, BlockTable, KvNamespace};

mod budget;
mod embeds;
mod sampler;
mod speculate;
pub mod rows;
pub mod conformance;
pub use budget::{MixedLimits, StepCost};
pub use embeds::PromptEmbeds;
pub use sampler::{apply_processors, apply_processors_with, ngram_followers, DryParams, NoRepeatNgram, ProcessorState, SampleContext, ThinkingBudget, MAX_DRY_BREAKERS, MAX_NGRAM_ALLOW, MAX_THINK_CLOSE, MAX_THINK_LEAD};
pub use speculate::{verify_proposals, DraftQuery, Drafter, NgramDrafter, SpecPolicy, SpecStats, Verdict};

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

    /// The most sequences one decode step carries, or `None` (the default) when
    /// it is not bounded by the decoder's own buffers. [`Scheduler::new`] never
    /// runs more sequences than this, whatever `max_running` it is given.
    fn max_batch(&self) -> Option<usize> {
        None
    }

    /// KV blocks currently free in the pool.
    fn free_blocks(&self) -> u32;

    /// The largest `prompt + max_new` a single sequence's capacity admits.
    fn max_seq_len(&self) -> usize;

    /// Vocabulary size — prompt tokens at or above this are rejected at
    /// admission rather than gathered out of bounds.
    fn vocab(&self) -> usize;

    /// KV blocks needed to hold `tokens` cached positions.
    fn blocks_for(&self, tokens: u32) -> u32;

    // Prefix reuse comes in two shapes, both keyed by adapter namespace plus exact
    // token prefix and both counted in `prefix_stats`:
    //  * shareable KV blocks (every attention layer): `crate::paged::PrefixCache`,
    //    which the decoder consults inside `prefill` and which `reclaim_prefix`
    //    shrinks;
    //  * snapshot-at-boundary state (a recurrent layer keeps a state, not rows,
    //    so a prefix resumes only from the state exactly after it):
    //    `crate::snapshot_prefix::SnapshotPrefixCache`, the same keys and
    //    recency rule over (KV rows, state snapshot row, resume payload) entries.
    // A hybrid decoder uses both kinds' bookkeeping from the shared crate and
    // implements only the device operations (copy rows, capture and restore state).

    /// Device blocks a fresh prefill of `prompt` under `adapter` would adopt
    /// from the prefix cache instead of allocating, as the cache stands now
    /// (a peek: it neither refreshes recency nor counts as a lookup). Admission
    /// charges a request only for the rest, so the next turn of a long
    /// conversation is not refused for want of room its own cached history
    /// already occupies. A block the prefill copies before writing into it is
    /// not adopted. The default, for a decoder without a prefix cache, is 0.
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let _ = (prompt, adapter);
        0
    }

    /// Reclaim up to `want` cache-only prefix blocks back to the pool (live
    /// sequences always outrank the prefix cache). Returns blocks actually freed.
    fn reclaim_prefix(&mut self, want: u32) -> u32;

    /// [`Self::reclaim_prefix`] for the admission of a request with `prompt`
    /// under `adapter`: the cached blocks that very prompt is about to adopt
    /// are spared. A conversation's next turn finds its own earlier turn at
    /// the cold end of the cache (the other conversations ran since), so
    /// reclaiming in plain recency order would free exactly the blocks the
    /// request is admitted to reuse. The default spares nothing.
    fn reclaim_prefix_keeping(&mut self, want: u32, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let _ = (prompt, adapter);
        self.reclaim_prefix(want)
    }

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

    // ---- prefill interleaved with decode (optional) ----
    //
    // A decoder that can prefill a prompt in slices says so with
    // `prefill_interleave_tokens`; the scheduler then prefills at most that many
    // prompt tokens per iteration and decodes every running sequence in between,
    // instead of stalling all of them for a whole prompt. The defaults are the
    // whole-prompt prefill that every decoder had.

    /// Prompt tokens per scheduler iteration to prefill when decode must keep
    /// moving, or `None` (the default) if this decoder prefills a prompt in one
    /// piece.
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        None
    }

    /// Make `prompt[..at]` resumable for later prompts as soon as this
    /// sequence's prefill (of `prompt`, into `table`, possibly not started yet)
    /// has passed it, not only once the sequence ends; the decoder may round
    /// `at` down to a point it can resume from. Returns whether it will: the
    /// scheduler holds back a waiting request that shares that prefix only
    /// then, and admits it once the prefill is past `at`, so requests arriving
    /// together prefill a shared document once. The default declines.
    fn hint_shared_prefix(&mut self, table: &mut BlockTable, prompt: &[u32], at: usize) -> bool {
        let _ = (table, prompt, at);
        false
    }

    /// Prefill the next slice of `prompt` into `table`: at most `budget` (>= 1)
    /// tokens, continuing where the previous call for this sequence stopped
    /// (`table` is empty on the first). The default prefills the whole prompt.
    /// The decoder keeps whatever it needs between calls; the sequence can be
    /// released between them ([`Self::release_table`]).
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        let _ = budget;
        PrefillProgress::Done(self.prefill(table, prompt))
    }

    // ---- prompts with embedding rows (optional) ----
    //
    // A multimodal prompt carries rows from an encoder at some positions
    // ([`PromptEmbeds`]). A decoder that reads them says so with `embed_width`;
    // the scheduler then admits such requests and hands every prefill slice
    // (`prefill_chunk_embeds`, or the `embeds` of a [`PrefillJob`]) the rows. A
    // decoder that does not is never asked: the scheduler refuses the request
    // at admission. A decoder with prefix reuse must never serve a prefix across
    // differing rows: key reuse by [`PromptEmbeds::digest`] or keep embedding
    // spans out of it (the conformance suite checks this).

    /// Floats per embedding row a prompt may carry, or `None` (the default) if
    /// this decoder takes tokens only.
    fn embed_width(&self) -> Option<usize> {
        None
    }

    /// [`Self::prefill_chunk`] for a prompt some of whose positions are
    /// embedding rows: the row at position `p` is `embeds.row_at(p)`, and the
    /// prompt's token there is a placeholder the decoder does not look up. Only
    /// called when [`Self::embed_width`] is `Some`.
    fn prefill_chunk_embeds(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: &PromptEmbeds, budget: usize) -> PrefillProgress {
        let _ = (table, prompt, embeds, budget);
        unreachable!("a decoder that declares embed_width must implement prefill_chunk_embeds")
    }

    // ---- prefill of several sequences in one forward (optional) ----
    //
    // A decoder whose forward is cheaper per row the more rows it carries (every
    // weight is read once however many rows share the pass) says how many rows it
    // takes with `prefill_batch_tokens`; the scheduler then hands it the next slices
    // of SEVERAL prefilling sequences in one `prefill_chunks` call, instead of one
    // sequence's slice per call. The default is the one-slice-per-call loop.

    /// Total prompt rows the decoder prefills in one [`Self::prefill_chunks`] call
    /// across sequences, or `None` (the default) if it prefills one sequence per
    /// call. The scheduler's per-iteration prefill budget then defaults to it, and
    /// [`Self::prefill_interleave_tokens`] stays the most one sequence gets of it
    /// while others wait.
    fn prefill_batch_tokens(&self) -> Option<usize> {
        None
    }

    /// Prefill the next slice of every job in one call: job `i` gets at most
    /// `jobs[i].budget` tokens, with the semantics of [`Self::prefill_chunk`], and
    /// the answers come back in job order. The default is one `prefill_chunk` per
    /// job; a decoder that overrides [`Self::prefill_batch_tokens`] overrides this
    /// to run the jobs' rows as one forward.
    fn prefill_chunks(&mut self, jobs: &mut [PrefillJob<'_>]) -> Vec<PrefillProgress> {
        jobs.iter_mut()
            .map(|j| match j.embeds {
                Some(e) => self.prefill_chunk_embeds(j.table, j.prompt, e, j.budget),
                None => self.prefill_chunk(j.table, j.prompt, j.budget),
            })
            .collect()
    }

    // ---- prefill in the background (optional) ----
    //
    // A decoder that can read prompts in on another stream of its device while
    // decode steps run says so with `background_prefill`; the scheduler then
    // hands it every prompt still being read, every iteration, and decodes
    // without waiting for them - a prompt is never what holds a running
    // sequence's next token back. The default keeps the slices above.

    /// Whether this decoder reads prompts in while it decodes
    /// ([`Self::prefill_background`] instead of the slices above).
    fn background_prefill(&self) -> bool {
        false
    }

    /// Advance the prompts of `jobs` (every sequence still being prefilled,
    /// oldest first; `table` empty on a sequence's first call, `budget` its
    /// tokens still to read): start, continue or finish them on the decoder's
    /// background work, without waiting for it while `wait` is false - the
    /// scheduler decodes this iteration. `wait` is true when nothing decodes:
    /// the decoder then waits until a job has advanced, so the scheduler never
    /// spins. One answer per job in job order: `Partial { consumed }` with the
    /// tokens read in since the last call (0 for a job not reached yet), or
    /// `Done`. The default is [`Self::prefill_chunks`].
    fn prefill_background(&mut self, jobs: &mut [PrefillJob<'_>], wait: bool) -> Vec<PrefillProgress> {
        let _ = wait;
        self.prefill_chunks(jobs)
    }

    // ---- mixed iterations (optional) ----
    //
    // A decoder whose forward can carry the next token of every running sequence AND slices of the prompts
    // still being read in as ONE pass (each weight read once for both) says so with `mixed_step`. The
    // scheduler then runs one such forward per iteration, sizing the prompt part from the decoder's latency
    // target and the cost of the iterations it has timed ([`StepCost`]), instead of separate prefill and
    // decode passes. This supersedes the background prefill above for a decoder that offers both.

    /// The shape of mixed iteration this decoder takes, or `None` (the default) if it runs prefill and decode
    /// as separate passes.
    fn mixed_step(&self) -> Option<MixedLimits> {
        None
    }

    /// One iteration as one forward: the decode step of `tables` (row `i` is `inputs[i]` at the next position
    /// of `tables[i]`, as [`Self::forward_batched_greedy`] / [`Self::forward_batched_topk`] take them) and the
    /// next slice of every job (the semantics of [`Self::prefill_chunks`]). `cands == 0` asks for greedy
    /// picks, answered `(token, 0.0)`; `cands >= 1` for that many `(token, logit)` candidates, best first,
    /// clamped to [`Self::topk_capacity`]. Either part may be empty. Each decode row must equal what the
    /// separate decode step gives, and each job what the separate prefill gives, up to the decoder's
    /// numerics. The default runs the two passes one after the other.
    fn forward_mixed(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], cands: usize, jobs: &mut [PrefillJob<'_>]) -> MixedOutcome {
        let prefilled = if jobs.is_empty() { Vec::new() } else { self.prefill_chunks(jobs) };
        let decoded = if tables.is_empty() {
            Vec::new()
        } else if cands == 0 {
            self.forward_batched_greedy(tables, inputs).into_iter().map(|t| vec![(t, 0.0)]).collect()
        } else {
            self.forward_batched_topk(tables, inputs, cands)
        };
        MixedOutcome { decoded, prefilled, first: Vec::new() }
    }

    /// [`Self::forward_mixed`] whose decode rows are a speculative verify step: `rows[i]` are the consecutive tokens
    /// of the sequence of `tables[i]` ([`Self::forward_batched_verify`]'s rows), scored in the same forward as the
    /// next slice of every job. Answers the candidates of every verify row, the jobs' progress, and (for a job that
    /// finished) the candidates of its first token from the same head pass; the verify is settled with
    /// [`Self::commit_verified`] as usual. The default runs the two passes one after the other.
    fn forward_mixed_verify(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize, jobs: &mut [PrefillJob<'_>]) -> MixedVerifyOutcome {
        let prefilled = if jobs.is_empty() { Vec::new() } else { self.prefill_chunks(jobs) };
        let scored = self.forward_batched_verify(tables, rows, cands);
        MixedVerifyOutcome { scored, prefilled, first: Vec::new() }
    }

    /// A prefill call unwound with a device allocation failure
    /// ([`gpu_core::DeviceOutOfMemory`]): drop whatever prompt the decoder was
    /// reading in and leave the device idle, so the scheduler can release the
    /// prefilling sequences and go on. The default holds no such work.
    fn abandon_prefill(&mut self) {}

    // ---- speculative decoding (optional; see [`speculate`]) ----
    //
    // Every method below defaults to "this decoder does not speculate", so an
    // existing `PagedDecoder` keeps compiling and being scheduled exactly as
    // before. A decoder that CAN verify several positions per sequence in one
    // batched step overrides `spec_max_draft` and `forward_batched_verify`
    // (and `commit_verified`, if it holds per-sequence state a rejected
    // position must not leave behind); a decoder with a drafter of its own
    // also overrides `draft`, otherwise the scheduler's installed
    // [`Drafter`] supplies proposals.

    /// The most proposals per sequence one [`Self::forward_batched_verify`]
    /// may be asked to score. `0` (the default) means the decoder does not
    /// speculate and the scheduler never calls the other speculation methods.
    fn spec_max_draft(&self) -> usize {
        0
    }

    /// The largest number of sequences for which verifying beats plain
    /// batched decode on this decoder - the measured crossover. Past it the
    /// scheduler decodes plainly for that step.
    fn spec_max_batch(&self) -> usize {
        usize::MAX
    }

    /// The most rows - the sum over sequences of one input plus its proposals -
    /// one [`Self::forward_batched_verify`] may carry (the decoder's scratch
    /// for undoing a rejection is sized for it). The scheduler shrinks windows
    /// to fit and does not speculate when even one row per sequence will not.
    fn spec_row_budget(&self) -> usize {
        usize::MAX
    }

    /// Whether the time a step takes on this decoder is a signal worth acting
    /// on: the scheduler then times plain and speculative steps and speculates
    /// only where that has been producing more tokens per second, trying the
    /// other mode again now and then. `false` (the default) leaves speculation
    /// to the windows and [`Self::spec_max_batch`] alone.
    fn spec_cost_aware(&self) -> bool {
        false
    }

    /// Seconds the decoder's per-prompt drafter preparation (feeding the drafter
    /// the prompt's hidden states) has cost one sequence, averaged over recent
    /// sequences; `None` (the default) when the decoder has no such cost or has
    /// not measured one yet. The scheduler weighs it against what speculation
    /// saves per request ([`speculate::PrefillGate`]).
    fn spec_prefill_cost(&self) -> Option<f64> {
        None
    }

    /// The step duration the decoder is asked to hold for the sequences it carries (their token interval), when it
    /// has such a target. A cost-aware scheduler ([`Self::spec_cost_aware`]) then does not speculate where a
    /// speculative step exceeds it and a plain one does not, whatever speculation gains in tokens a second. `None`
    /// (the default) ranks the two by throughput alone.
    fn step_latency_target(&self) -> Option<std::time::Duration> {
        None
    }

    /// Whether prompts that begin from now on prepare the drafter. Off, their
    /// sequences are decoded without drafts until they end. Called by the
    /// scheduler before it starts prefilling; the default ignores it.
    fn set_spec_prefill(&mut self, on: bool) {
        let _ = on;
    }

    /// The decoder's own proposals: `ks[i]` is how many to propose after
    /// `queries[i]`'s context. Fewer (or none) is always allowed. Called only
    /// when no external [`Drafter`] is installed; the default proposes nothing.
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        let _ = ks;
        vec![Vec::new(); queries.len()]
    }

    /// Score `rows[i]` (the sequence's next input followed by its proposals)
    /// at consecutive new positions of `tables[i]`, appending `rows[i].len()`
    /// positions to each table. Returns, per sequence and row, the top
    /// `cands` `(token id, logit)` candidates of the distribution AFTER that
    /// row, best first (`cands == 1` is the greedy pick). The decoder must be
    /// able to undo the verify: until [`Self::commit_verified`] it keeps
    /// whatever it needs to restore each sequence to any row boundary.
    fn forward_batched_verify(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize) -> Vec<Vec<Vec<(u32, f32)>>> {
        let _ = (tables, rows, cands);
        unreachable!("forward_batched_verify on a decoder whose spec_max_draft is 0")
    }

    /// Settle the verify just run: sequence `i` keeps the first `keep[i]`
    /// rows (the input row plus its accepted proposals) and every later row
    /// is rolled back - recurrent state restored to the state after row
    /// `keep[i] - 1`. The tables still hold every verified position; the
    /// scheduler truncates them afterwards. A decoder with a drafter also
    /// advances it here.
    fn commit_verified(&mut self, tables: &mut [&mut BlockTable], keep: &[usize]) {
        let _ = (tables, keep);
    }

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

    // ---- runtime adapters (optional) ----
    //
    // Every method below has a default that reports "this decoder serves the
    // base model only", so an existing `PagedDecoder` implementation keeps
    // compiling and behaving exactly as it did, and refuses adapter requests
    // instead of silently answering them with the base weights.
    //
    // A decoder that CAN apply an adapter at run time resolves each row's
    // adapter from its block table: [`BlockTable::kv_namespace`] is the
    // sequence's [`AdapterId`] namespace (or the base's), so `prefill` and the
    // batched forwards need no extra argument.

    /// Can sequences using `adapter` be admitted now? A decoder with a bounded
    /// set of resident adapters starts making `adapter` resident here when it
    /// is not (loading, or evicting an unpinned one) and answers
    /// [`AdapterAvailability::Loading`] until that finishes, so the call must
    /// be cheap and idempotent: the scheduler repeats it every iteration while
    /// a request waits. An answer of `Resident` holds until the scheduler's
    /// next [`Self::pin_adapter`] for it, which follows immediately.
    fn adapter_availability(&mut self, adapter: AdapterId) -> AdapterAvailability {
        let _ = adapter;
        AdapterAvailability::Unknown
    }

    /// A sequence using `adapter` was admitted: keep it resident until the
    /// matching [`Self::unpin_adapter`]. Called once per admitted sequence,
    /// only for an adapter that just answered `Resident`.
    fn pin_adapter(&mut self, adapter: AdapterId) {
        let _ = adapter;
    }

    /// The sequence that pinned `adapter` finished or was cancelled; its
    /// slot may be reclaimed once no sequence pins it.
    fn unpin_adapter(&mut self, adapter: AdapterId) {
        let _ = adapter;
    }
}

/// One sequence's share of a [`PagedDecoder::prefill_chunks`] call.
pub struct PrefillJob<'a> {
    /// The sequence's blocks (empty on its first slice).
    pub table: &'a mut BlockTable,
    /// The whole prompt.
    pub prompt: &'a [u32],
    /// At most this many more prompt tokens (>= 1).
    pub budget: usize,
    /// The embedding rows of the prompt, when it carries any.
    pub embeds: Option<&'a PromptEmbeds>,
}

/// A job's first-token candidates from a mixed forward's head pass (empty: none).
type FirstCandidates = Vec<(u32, f32)>;

/// What one [`PagedDecoder::forward_mixed`] call produced.
pub struct MixedOutcome {
    /// Per decode row, its candidates (`cands == 0`: the greedy token with logprob 0).
    pub decoded: Vec<Vec<(u32, f32)>>,
    /// One answer per job, in job order.
    pub prefilled: Vec<PrefillProgress>,
    /// For a job that finished ([`PrefillProgress::Done`]), the candidates of its first token, taken from the same
    /// head pass as the decode rows' (so the head's weights are read once, not once more per finished prompt), in
    /// the form `cands` asked for. Empty for a job that did not finish or a decoder that has none: the scheduler
    /// then runs [`PagedDecoder::admit_greedy`] / [`PagedDecoder::admit_topk`] on the job's hidden state.
    pub first: Vec<Vec<(u32, f32)>>,
}

/// What one [`PagedDecoder::forward_mixed_verify`] call produced.
pub struct MixedVerifyOutcome {
    /// Per sequence, per verify row, its candidates ([`PagedDecoder::forward_batched_verify`]'s answer).
    pub scored: Vec<Vec<Vec<(u32, f32)>>>,
    /// One answer per job, in job order.
    pub prefilled: Vec<PrefillProgress>,
    /// As [`MixedOutcome::first`].
    pub first: Vec<Vec<(u32, f32)>>,
}

/// `f`, with a device allocation failure it unwinds with
/// ([`gpu_core::DeviceOutOfMemory`]) returned as an error. Any other panic keeps
/// unwinding: only a full card is a condition the scheduler can serve through.
fn catch_device_oom<R>(f: impl FnOnce() -> R) -> Result<R, gpu_core::DeviceOutOfMemory> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => Ok(r),
        Err(payload) => match payload.downcast::<gpu_core::DeviceOutOfMemory>() {
            Ok(oom) => Err(*oom),
            Err(other) => std::panic::resume_unwind(other),
        },
    }
}

/// What one [`PagedDecoder::prefill_chunk`] call achieved.
pub enum PrefillProgress {
    /// The prompt is in: the final hidden state, as [`PagedDecoder::prefill`] returns it.
    Done(Vec<f32>),
    /// `consumed` prompt tokens went in and more remain.
    Partial { consumed: usize },
}

/// Which adapter a request runs under. It doubles as the adapter's
/// [`KvNamespace`]: the owner of adapter identities (one per resident
/// adapter, from [`crate::paged::NamespaceIds`]) never reuses a value, and
/// "no adapter" (the base model) is the absence of an `AdapterId`, never a
/// reserved value inside one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterId(std::num::NonZeroU32);

impl AdapterId {
    /// The adapter whose KV namespace is `namespace`; `None` for the base
    /// namespace, which no adapter can have.
    pub fn new(namespace: KvNamespace) -> Option<AdapterId> {
        std::num::NonZeroU32::new(namespace).map(AdapterId)
    }

    /// The KV namespace this adapter's sequences compute and cache K/V in.
    pub fn kv_namespace(self) -> KvNamespace {
        self.0.get()
    }
}

/// A decoder's answer to [`PagedDecoder::adapter_availability`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterAvailability {
    /// Resident now: the request may be admitted.
    Resident,
    /// Not resident yet but becoming so (loading, or waiting for a slot to
    /// free up). The request is skipped, not blocked on: requests behind it
    /// are admitted past it and it is asked again next iteration.
    Loading,
    /// Not served by this decoder (never registered, unloaded, or the decoder
    /// has no adapter support): the request is rejected.
    Unknown,
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
    /// Drop every token whose probability is below `min_p` times the best
    /// token's, after temperature, top-k and top-p; `<= 0.0` disables it.
    pub min_p: f32,
    /// OpenAI `presence_penalty`: subtracted from the logit of every token that
    /// already appears in the generated text. `0.0` disables it.
    pub presence_penalty: f32,
    /// OpenAI `frequency_penalty`: subtracted once per occurrence in the
    /// generated text. `0.0` disables it.
    pub frequency_penalty: f32,
    /// vLLM `repetition_penalty`: divides a seen token's positive logit (multiplies
    /// a negative one), the prompt included. `1.0` disables it.
    pub repetition_penalty: f32,
    /// The DRY repetition penalty over the whole context (prompt and generated).
    pub dry: DryParams,
    /// Force the closing tag after this many tokens inside a think block.
    pub thinking: Option<ThinkingBudget>,
    /// Ban tokens that would repeat an n-gram of the generated text.
    pub no_repeat: NoRepeatNgram,
}

impl Default for SampleParams {
    fn default() -> SampleParams {
        SampleParams::greedy()
    }
}

impl SampleParams {
    pub fn greedy() -> SampleParams {
        SampleParams { temp: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0, dry: DryParams::off(), thinking: None, no_repeat: NoRepeatNgram::off() }
    }
    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0
    }
    /// Whether any logit processor (penalty or DRY) is asked for. Off by
    /// default; a request that asks for none samples exactly as it always did.
    pub fn has_processors(&self) -> bool {
        self.has_penalties() || self.no_repeat.active()
    }
    /// The processors that reorder the whole candidate list (penalties and DRY).
    fn has_penalties(&self) -> bool {
        self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 || (self.repetition_penalty != 1.0 && self.repetition_penalty > 0.0) || self.dry.active()
    }
    /// Whether the step must see the decoder's candidate list instead of taking
    /// its on-device argmax: real sampling, or a processor that changes which
    /// token is the best (a penalised greedy decode is an argmax of the
    /// penalised logits, so it needs the candidates too). The no-repeat filter
    /// is not here: it bans a known handful of tokens, so a step asks for just
    /// that many extra candidates only when the history bans one.
    pub fn needs_candidates(&self) -> bool {
        !self.is_greedy() || self.has_penalties()
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
    sample_from_topk_impl(candidates, params, rng).0
}

/// Like [`sample_from_topk`], but also returns the sampled token's own
/// log-probability under the exact truncated, temperature-scaled
/// distribution it was drawn from - captured for free at the point of
/// sampling instead of re-derived by a second softmax pass later (a
/// rollout's GRPO-style `pi_old`; see `crate::rollout::Completion::logprobs`).
/// Greedy decoding is deterministic, so its "probability" is `1.0`
/// (`logprob = 0.0`).
pub fn sample_from_topk_with_logprob(candidates: &[(u32, f32)], params: SampleParams, rng: &mut data::rng::Rng) -> (u32, f32) {
    sample_from_topk_impl(candidates, params, rng)
}

/// [`sample_from_topk_with_logprob`] for a sequence with a context: the request's
/// logit processors (penalties, DRY) are applied to the candidates first, so the
/// choice (greedy or drawn) and the reported log-probability are those of the
/// processed distribution. With no processor asked for it is the plain function.
pub fn sample_in_context(candidates: &[(u32, f32)], params: SampleParams, ctx: &SampleContext<'_>, rng: &mut data::rng::Rng) -> (u32, f32) {
    sample_in_context_with(candidates, params, ctx, &mut ProcessorState::default(), rng)
}

/// [`sample_in_context`] over the sequence's own [`ProcessorState`]: the scheduler's per-token form, which
/// folds only the tokens appended since the previous step into the penalties instead of rescanning the context.
pub fn sample_in_context_with(candidates: &[(u32, f32)], params: SampleParams, ctx: &SampleContext<'_>, state: &mut ProcessorState, rng: &mut data::rng::Rng) -> (u32, f32) {
    if !params.has_processors() {
        return sample_from_topk_impl(candidates, params, rng);
    }
    let mut processed = candidates.to_vec();
    apply_processors_with(&mut processed, &params, ctx, state);
    sample_from_topk_impl(&processed, params, rng)
}

/// Shared body for [`sample_from_topk`]/[`sample_from_topk_with_logprob`] -
/// one sampling implementation, two return shapes, so the two callers can
/// never drift apart on the actual math.
fn sample_from_topk_impl(candidates: &[(u32, f32)], params: SampleParams, rng: &mut data::rng::Rng) -> (u32, f32) {
    assert!(!candidates.is_empty(), "sample_from_topk: no candidates");
    if params.is_greedy() {
        return (candidates[0].0, 0.0);
    }
    let (probs, kept_sum) = truncated_weights(candidates, params);
    let r = rng.next_f32() * kept_sum;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            return (candidates[i].0, p.ln() - kept_sum.ln());
        }
    }
    let last = probs.len() - 1;
    (candidates[last].0, probs[last].ln() - kept_sum.ln())
}

/// The log-probability of `token` under the distribution [`sample_from_topk`]
/// draws from for `candidates` and `params` (`0.0` when decoding is greedy).
/// A token outside the truncated set has probability zero, which is `-inf`.
pub fn candidate_logprob(candidates: &[(u32, f32)], params: SampleParams, token: u32) -> f32 {
    if params.is_greedy() {
        return 0.0;
    }
    let (weights, kept_sum) = truncated_weights(candidates, params);
    candidates[..weights.len()].iter().position(|&(t, _)| t == token).map_or(f32::NEG_INFINITY, |i| weights[i].ln() - kept_sum.ln())
}

/// The unnormalised weights (and their sum) of the distribution a non-greedy
/// pick draws from: temperature scale, truncate to `top_k`, softmax, then
/// nucleus-truncate to `top_p`. Weight `i` belongs to `candidates[i]`, and
/// only the surviving prefix is returned. The one definition of "the target
/// distribution" shared by plain sampling and speculative verification, so
/// the two can never disagree about it.
fn truncated_weights(candidates: &[(u32, f32)], params: SampleParams) -> (Vec<f32>, f32) {
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
    probs.truncate(cut);
    // min-p: the candidates are sorted, so the survivors are a prefix, and the
    // best token (`probs[0]`) always survives.
    if params.min_p > 0.0 {
        let floor = probs[0] * params.min_p;
        let keep = probs.iter().take_while(|&&p| p >= floor).count().max(1);
        probs.truncate(keep);
    }
    let kept_sum: f32 = probs.iter().sum();
    (probs, kept_sum)
}

/// Ports vLLM's `NoRepeatNGramLogitsProcessor` verbatim
/// (`vllm/model_executor/models/deepseek_ocr.py`, the
/// `NGramPerReqLogitsProcessor` its own `deepseek-ai/DeepSeek-OCR` recipe
/// wires up) - the anti-repetition filter upstream recommends specifically
/// because a long OCR document's greedy decode can lock onto a repeated
/// span (a table row, a repeated header) with nothing to break the loop.
/// Upstream's own recipe values: `ngram_size = 30`, `window_size = 90`,
/// `whitelist_token_ids = {<td>, </td>}` - table cells legitimately repeat
/// a short tag pair far more than 30 tokens apart, so banning them would
/// corrupt real tables rather than protect against a stuck decode.
///
/// `history` is the GENERATED ids so far (never the prompt - matching
/// vLLM's own `output_ids`, which tracks only what the request itself
/// produced). Operates in place on `logits`, a full `[vocab]` row (this
/// decode step's own shape, unlike [`sample_from_topk`]'s candidate list -
/// there is no on-device top-K truncation upstream of this call to filter
/// instead). Finds every earlier position in the last `window_size`
/// generated tokens whose following `(ngram_size - 1)`-token span matches
/// the CURRENT trailing span, and bans whatever token immediately followed
/// each such match there (i.e. every token that would recreate an n-gram
/// already seen inside the window), except tokens in `whitelist`. A no-op
/// once `history.len() < ngram_size` (not enough tokens yet to form one).
pub fn apply_no_repeat_ngram(logits: &mut [f32], history: &[u32], ngram_size: usize, window_size: usize, whitelist: &std::collections::HashSet<u32>) {
    for tok in sampler::ngram_followers(history, ngram_size, window_size) {
        if whitelist.contains(&tok) {
            continue;
        }
        if let Some(l) = logits.get_mut(tok as usize) {
            *l = f32::NEG_INFINITY;
        }
    }
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
    /// The ids that end this sequence: a checkpoint may declare several
    /// (`generation_config.json`'s `eos_token_id` list), and a chat format
    /// adds its own end-of-turn marker. Empty stops only at `max_new`.
    pub stop: Vec<u32>,
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
    /// The request names an adapter the decoder does not serve (see
    /// [`AdapterAvailability::Unknown`]). Refusing beats answering with the
    /// base model's weights.
    AdapterUnavailable { adapter: AdapterId },
    /// The embedding rows do not fit the request or the decoder (see
    /// [`PromptEmbeds::refusal`]).
    InvalidEmbeds { why: String },
    /// The card had no memory left for the step that was reading this prompt in.
    /// Retrying later, or with a shorter prompt, can succeed.
    DeviceOutOfMemory { bytes: u64 },
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
            RejectReason::AdapterUnavailable { adapter } => {
                write!(f, "adapter {} is not available", adapter.kv_namespace())
            }
            RejectReason::InvalidEmbeds { why } => write!(f, "invalid embedding rows: {why}"),
            RejectReason::DeviceOutOfMemory { bytes } => write!(f, "the device ran out of memory ({bytes} bytes refused) while the prompt was read in"),
        }
    }
}

/// What [`Scheduler::admissibility`] found out about one request.
enum Admissibility {
    Admissible,
    /// Its adapter is still loading: asked about again next iteration.
    AdapterLoading,
    Refused(RejectReason),
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
    /// For each of `completed`, the log-probability of every generated token
    /// under the distribution it was decoded from: the temperature, top-k and
    /// top-p truncated softmax a sampled token was drawn from, and `0.0` for
    /// greedy decoding (deterministic). Same order and lengths as `completed`.
    pub completed_logprobs: Vec<(u64, Vec<f32>)>,
    /// For each of `completed`, how its prompt was served and why it stopped (see [`Completion`]).
    pub completed_meta: Vec<(u64, Completion)>,
    /// Requests preempted this iteration: their KV went to host RAM and their
    /// device blocks back to the pool (see [`crate::kv_offload`]).
    pub demoted: Vec<u64>,
    /// Requests whose KV came back from host RAM this iteration.
    pub promoted: Vec<u64>,
    /// Requests preempted by recompute this iteration: their KV was dropped
    /// and they were requeued to prefill their context again.
    pub recomputed: Vec<u64>,
}

/// A submitted request that has not been admitted yet.
struct Waiting {
    id: u64,
    req: Request,
    /// The adapter the request runs under; `None` is the base model.
    adapter: Option<AdapterId>,
    sample: SampleParams,
    seed: u64,
    /// `Some` when this is a running sequence that was preempted by
    /// recompute: `req.prompt` is then the context whose KV must be rebuilt and
    /// `req.max_new` is still the request's whole token budget.
    resume: Option<Resume>,
    /// Embedding rows of the prompt. They index the ORIGINAL prompt, which a
    /// preempted sequence's context starts with, so they stay valid through a
    /// recompute.
    embeds: Option<std::sync::Arc<PromptEmbeds>>,
    /// The first prompt token outside the vocabulary, found once when the request was queued (O(prompt)) so
    /// admission, which looks at every waiting request each iteration, does not scan the prompt again. `None`
    /// for a prompt that passed, which is every one a preempted sequence requeues.
    bad_token: Option<u32>,
}

/// What a sequence preempted by recompute carries through the waiting queue so
/// that, once its context is prefilled again, it continues exactly where it
/// stopped instead of starting over: the tokens already produced, the sampler's
/// RNG stream and the adaptive speculation window. Its adapter stays pinned
/// the whole time.
struct Resume {
    generated: Vec<u32>,
    /// The log-probability of each token in `generated`.
    logprobs: Vec<f32>,
    /// The token whose K/V is not in the cache yet (the last one produced).
    next_input: u32,
    spec: speculate::SeqSpec,
    rng: data::rng::Rng,
    /// Where the sequence was in its think block.
    think: Option<sampler::ThinkState>,
    /// Prompt tokens the first admission served from the prefix cache.
    cached_tokens: usize,
}

/// What a finished request reports beside its tokens, the same for every decoder whatever shape its cache has.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Completion {
    /// Prompt tokens the decoder's prefix cache served instead of computing them (0 for a cold prompt).
    pub cached_tokens: usize,
    /// The stop token the sequence ended on. It is not among the generated tokens (a stop id is a control
    /// signal, not output); `None` when the sequence ended on its token budget or was cancelled.
    pub stop_token: Option<u32>,
}

/// A request being prefilled over several iterations: it holds its blocks and a
/// slot of the batch but does not decode yet.
struct Prefilling {
    id: u64,
    prompt: Vec<u32>,
    /// Prompt tokens already in.
    done: usize,
    /// Blocks the prompt adopts from the prefix cache, asked of the decoder once at admission: they are resident
    /// already and owed nothing (see [`Scheduler::prefill_owed`]).
    adopted: u32,
    max_new: usize,
    stop: Vec<u32>,
    adapter: Option<AdapterId>,
    sample: SampleParams,
    seed: u64,
    table: BlockTable,
    resume: Option<Resume>,
    embeds: Option<std::sync::Arc<PromptEmbeds>>,
    /// What the decoder answered when asked to make a prefix of this prompt
    /// resumable for a waiting request ([`PagedDecoder::hint_shared_prefix`]):
    /// the prefix it will file, or `None` once it declined. Unasked: unset.
    shared_hint: Option<Option<usize>>,
}

/// A sequence the scheduler is actively decoding.
struct Running {
    id: u64,
    table: BlockTable,
    /// Pinned in the decoder from admission until the sequence is dropped
    /// (finished, cancelled, or the scheduler torn down), preempted or not.
    adapter: Option<AdapterId>,
    generated: Vec<u32>,
    /// `logprobs[i]` is the log-probability `generated[i]` was decoded with
    /// (see [`StepReport::completed_logprobs`]).
    logprobs: Vec<f32>,
    max_new: usize,
    stop: Vec<u32>,
    next_input: u32,
    /// The prompt followed by `generated`: what a drafter reads.
    context: Vec<u32>,
    /// This sequence's adaptive speculation window.
    spec: speculate::SeqSpec,
    done: bool,
    /// This sequence's sampling params + its own RNG stream. Kept per-sequence
    /// (not per-batch) so a `seed` reproduces exactly regardless of what else
    /// is in the batch alongside it.
    sample: SampleParams,
    /// Where the sequence is in its think block when a thinking budget was asked for.
    think: Option<sampler::ThinkState>,
    /// What the logit processors have read of `context` so far (empty until a processor is asked for).
    procs: ProcessorState,
    /// The seed `rng` started from, kept so a preempted sequence can be
    /// requeued as the request it was.
    seed: u64,
    rng: data::rng::Rng,
    /// Prompt tokens served from the decoder's prefix cache when the sequence was first admitted
    /// (see [`BlockTable::cached_rows`]); a recompute after preemption does not change it.
    cached_tokens: usize,
    /// The stop token that ended the sequence, once it has.
    stop_token: Option<u32>,
    /// The prompt's embedding rows, kept for a recompute (see [`Waiting::embeds`]).
    embeds: Option<std::sync::Arc<PromptEmbeds>>,
}

/// The block tables of the sequences `in_batch` marks, in `running` order. O(running).
fn active_tables<'a>(running: &'a mut [Running], in_batch: &[bool]) -> Vec<&'a mut BlockTable> {
    running.iter_mut().zip(in_batch).filter(|(_, &on)| on).map(|(r, _)| &mut r.table).collect()
}

/// The token each sequence of `active` samples from its candidates (`candidates[bi]` for `active[bi]`): real
/// sampling under the sequence's own parameters, penalties and random stream.
fn sample_rows(running: &mut [Running], active: &[usize], candidates: &[Vec<(u32, f32)>]) -> Vec<Vec<(u32, f32)>> {
    active
        .iter()
        .enumerate()
        .map(|(bi, &si)| {
            let r = &mut running[si];
            let ctx = SampleContext { context: &r.context, prompt_len: r.context.len() - r.generated.len() };
            vec![sample_in_context_with(&candidates[bi], r.sample, &ctx, &mut r.procs, &mut r.rng)]
        })
        .collect()
}

/// Accept a freshly sampled token into `r`.
///
/// If `next` is one of the sequence's stop ids, `r` is marked done WITHOUT
/// appending it to `generated` — a stop id is a control signal, not output
/// text. Appending it (the previous behavior) meant the token's decoded text
/// (e.g. Qwen's `<|im_end|>`) leaked into every streamed response, since
/// `generated`/`tokens_of` feed the text returned to callers. Otherwise
/// `next` is appended normally, and `r` is marked done if `max_new` is now
/// reached. A free function (not `Scheduler<D>::accept_token`) because it
/// only ever touches `Running`, never `D`/`self`.
fn accept_token(r: &mut Running, next: u32, logprob: f32) {
    if r.stop.contains(&next) {
        r.done = true;
        r.stop_token = Some(next);
        return;
    }
    if let Some(t) = r.think.as_mut() {
        t.accept(next);
    }
    r.generated.push(next);
    r.logprobs.push(logprob);
    r.context.push(next);
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
    waiting: VecDeque<Waiting>,
    /// Requests admitted and being prefilled in slices, oldest first.
    prefilling: VecDeque<Prefilling>,
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
    /// Proposal source installed with [`Scheduler::set_drafter`]; `None` uses
    /// the decoder's own [`PagedDecoder::draft`].
    drafter: Option<Box<dyn Drafter>>,
    spec_policy: SpecPolicy,
    spec_stats: SpecStats,
    /// Plain and speculative step rates per batch-size bucket - see [`speculate::CostGate`].
    gate: speculate::CostGate,
    /// Whether new prompts prepare the drafter - see [`speculate::PrefillGate`].
    prefill_gate: speculate::PrefillGate,
    /// Seconds, for the cost gate and the mixed iteration's cost model.
    clock: Box<dyn Fn() -> f64 + Send>,
    /// What a mixed iteration of a given number of rows costs - see [`StepCost`].
    step_cost: StepCost,
}

impl<D: PagedDecoder> Scheduler<D> {
    pub fn new(dec: D, max_running: usize) -> Scheduler<D> {
        let max_running = dec.max_batch().map_or(max_running, |m| max_running.min(m));
        // Default budget: two full prefill chunks per iteration. Enough to keep
        // admission moving under load, small enough that running sequences see
        // a decode step between arrivals.
        let prefill_budget = dec.prefill_batch_tokens().or(dec.prefill_interleave_tokens()).map_or_else(|| dec.max_prefill_tokens().saturating_mul(2), |n| n as u32).max(1);
        Scheduler {
            dec,
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            max_running,
            admission: Box::new(UnboundedQueue),
            mean_service_ms: None,
            started: HashMap::new(),
            pending_rejects: Vec::new(),
            prefill_budget,
            swapped: VecDeque::new(),
            drafter: None,
            spec_policy: SpecPolicy::default(),
            spec_stats: SpecStats::default(),
            gate: speculate::CostGate::default(),
            prefill_gate: speculate::PrefillGate::default(),
            clock: {
                let start = std::time::Instant::now();
                Box::new(move || start.elapsed().as_secs_f64())
            },
            step_cost: StepCost::default(),
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
        self.enqueue(req, None, sample, seed)
    }

    /// [`Self::submit_sampled`] for a request that runs under `adapter`.
    ///
    /// While the adapter is not resident ([`AdapterAvailability::Loading`])
    /// the request waits WITHOUT holding up the requests behind it, so a cold
    /// adapter cannot delay base traffic; a decoder that does not serve the
    /// adapter makes the request come back in [`StepReport::rejected`] as
    /// [`RejectReason::AdapterUnavailable`]. Once admitted, the sequence's
    /// block table is in the adapter's KV namespace and the adapter stays
    /// pinned in the decoder until the sequence is gone.
    pub fn submit_for_adapter(&mut self, req: Request, adapter: AdapterId, sample: SampleParams, seed: u64) -> u64 {
        self.enqueue(req, Some(adapter), sample, seed)
    }

    fn enqueue(&mut self, req: Request, adapter: Option<AdapterId>, sample: SampleParams, seed: u64) -> u64 {
        self.enqueue_embedded(req, None, adapter, sample, seed)
    }

    /// [`Self::submit_sampled`] for a prompt some of whose positions are
    /// embedding rows (a multimodal request). `embeds` must fit the prompt and
    /// the decoder ([`PagedDecoder::embed_width`]); a request that does not is
    /// reported in the next [`StepReport::rejected`] as
    /// [`RejectReason::InvalidEmbeds`] and never queued. The rows travel with the
    /// request through chunked prefill and preemption by recompute.
    pub fn submit_embedded(&mut self, req: Request, embeds: PromptEmbeds, sample: SampleParams, seed: u64) -> u64 {
        if let Some(why) = embeds.refusal(req.prompt.len(), self.dec.embed_width()) {
            let id = self.next_id;
            self.next_id += 1;
            self.pending_rejects.push((id, RejectReason::InvalidEmbeds { why }));
            return id;
        }
        self.enqueue_embedded(req, Some(std::sync::Arc::new(embeds)), None, sample, seed)
    }

    fn enqueue_embedded(&mut self, req: Request, embeds: Option<std::sync::Arc<PromptEmbeds>>, adapter: Option<AdapterId>, sample: SampleParams, seed: u64) -> u64 {
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
        let bad_token = self.invalid_token(&req.prompt);
        self.waiting.push_back(Waiting { id, req, adapter, sample, seed, resume: None, embeds, bad_token });
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
        self.cancel_detailed(id).map(|(tokens, _)| tokens)
    }

    /// What request `id` reports so far (see [`Completion`]); the default for an unknown id or a request not
    /// admitted yet. Ask before [`Self::cancel_detailed`], which forgets the request.
    pub fn completion_of(&self, id: u64) -> Completion {
        let cached = |cached_tokens| Completion { cached_tokens, stop_token: None };
        if let Some(r) = self.running.iter().chain(self.swapped.iter()).find(|r| r.id == id) {
            return Completion { cached_tokens: r.cached_tokens, stop_token: r.stop_token };
        }
        if let Some(p) = self.prefilling.iter().find(|p| p.id == id) {
            return cached(p.resume.as_ref().map_or(p.table.cached_rows() as usize, |r| r.cached_tokens));
        }
        cached(self.waiting.iter().find(|w| w.id == id).and_then(|w| w.resume.as_ref()).map_or(0, |r| r.cached_tokens))
    }

    /// [`Self::cancel`], also returning the log-probability of each token it
    /// produced (see [`StepReport::completed_logprobs`]).
    pub fn cancel_detailed(&mut self, id: u64) -> Option<(Vec<u32>, Vec<f32>)> {
        if let Some(pos) = self.waiting.iter().position(|w| w.id == id) {
            let w = self.waiting.remove(pos).expect("position just found it");
            // A request never admitted produced nothing; one preempted by
            // recompute keeps its adapter pinned and has output to hand back.
            return Some(w.resume.map_or_else(Default::default, |r| {
                self.unpin(w.adapter);
                (r.generated, r.logprobs)
            }));
        }
        if let Some(pos) = self.prefilling.iter().position(|p| p.id == id) {
            let mut p = self.prefilling.remove(pos).expect("position just found it");
            self.dec.release_table(&mut p.table);
            self.unpin(p.adapter);
            return Some(p.resume.map_or_else(Default::default, |r| (r.generated, r.logprobs)));
        }
        if let Some(pos) = self.swapped.iter().position(|r| r.id == id) {
            // Demoted: its blocks are already back in the pool, its bytes are
            // in host RAM. Dropping the host record is the whole reclaim.
            let r = self.swapped.remove(pos).expect("position just found it");
            self.dec.discard_offloaded(id);
            self.unpin(r.adapter);
            return Some((r.generated, r.logprobs));
        }
        let pos = self.running.iter().position(|r| r.id == id)?;
        let mut r = self.running.remove(pos);
        self.dec.release_table(&mut r.table);
        self.unpin(r.adapter);
        Some((r.generated, r.logprobs))
    }

    fn unpin(&mut self, adapter: Option<AdapterId>) {
        if let Some(adapter) = adapter {
            self.dec.unpin_adapter(adapter);
        }
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

    /// The decoder, for an operation between scheduling steps that must not
    /// rebuild it (registering or unloading an adapter on a live engine).
    /// Nothing the scheduler tracks refers into the decoder, so changing the
    /// decoder's own registries is safe; changing what a running sequence
    /// reads is not, and is the caller's to avoid.
    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.dec
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
            if let Some(adapter) = r.adapter {
                self.dec.unpin_adapter(adapter);
            }
        }
        for p in &mut self.prefilling {
            self.dec.release_table(&mut p.table);
            if let Some(adapter) = p.adapter {
                self.dec.unpin_adapter(adapter);
            }
        }
        for r in &self.swapped {
            self.dec.discard_offloaded(r.id);
            if let Some(adapter) = r.adapter {
                self.dec.unpin_adapter(adapter);
            }
        }
        for w in self.waiting.iter().filter(|w| w.resume.is_some()) {
            if let Some(adapter) = w.adapter {
                self.dec.unpin_adapter(adapter);
            }
        }
        self.dec
    }

    /// Every sequence's generated tokens so far, by request id: running, preempted to host RAM, and
    /// preempted by recompute (the same set [`Self::tokens_of`] reads). One O(sequences) pass, so a caller
    /// that streams every open request each iteration pays O(sequences) per iteration instead of a
    /// linear find per request (O(sequences^2)). The map borrows the scheduler; read what you need from
    /// it before the next `&mut` call.
    pub fn streams(&self) -> HashMap<u64, &[u32]> {
        self.streaming_sequences().collect()
    }

    /// The `(id, generated tokens)` of every sequence a streaming caller may read.
    fn streaming_sequences(&self) -> impl Iterator<Item = (u64, &[u32])> {
        // Preempted sequences are included: preemption is invisible to a caller
        // streaming a response, so its progress must stay readable while its
        // KV sits in host RAM or is being recomputed.
        let preempted = self.waiting.iter().filter_map(|w| w.resume.as_ref().map(|r| (w.id, r.generated.as_slice())));
        self.running.iter().chain(self.swapped.iter()).map(|r| (r.id, r.generated.as_slice())).chain(preempted)
    }

    /// All tokens generated so far for a still-running sequence (`None` if
    /// `id` is unknown, queued, or already reaped) — the seam a streaming
    /// caller uses to emit the NEW suffix each iteration without waiting for
    /// the sequence to finish (`step`/`step_report` only return a completed
    /// sequence's full token list, not a running one's partial progress).
    ///
    /// A linear find: right for one request. A caller reading every open
    /// request per iteration uses [`Self::streams`].
    pub fn tokens_of(&self, id: u64) -> Option<&[u32]> {
        self.streaming_sequences().find(|&(rid, _)| rid == id).map(|(_, generated)| generated)
    }

    /// Requests admitted but not yet started.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// True while any request is waiting, running, or preempted to host RAM.
    pub fn pending(&self) -> bool {
        !self.waiting.is_empty() || !self.prefilling.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
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

    /// The most context one sequence may hold, prompt and reply together:
    /// the decoder's sequence limit within the KV pool. A request that names
    /// no budget generates until it ends its turn or reaches this.
    pub fn sequence_capacity(&self) -> usize {
        let pool_tokens = usize::try_from(self.dec.kv_pool_capacity_tokens()).unwrap_or(usize::MAX);
        self.dec.max_seq_len().min(pool_tokens)
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
        while !self.swapped.is_empty() && self.running.len() + self.prefilling.len() < self.max_running {
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

    /// Device blocks the running batch needs to append one more token each. O(running).
    fn blocks_for_next_token(&self) -> u32 {
        self.running.iter().map(|r| self.next_token_blocks(r)).sum()
    }

    /// Blocks `r` needs to append one more token (none once it is done).
    fn next_token_blocks(&self, r: &Running) -> u32 {
        if r.done {
            return 0;
        }
        self.dec.blocks_for(r.table.len() + 1).saturating_sub(r.table.blocks().len() as u32)
    }

    /// Device blocks the sequences still being prefilled have been promised
    /// (their whole prompt plus the first token) but do not hold yet. Prefill
    /// takes them chunk by chunk, so until it has, the pool looks emptier than
    /// it is. A prefill that has not started yet holds no blocks; the ones it
    /// will adopt from the prefix cache are resident already and owed nothing.
    /// O(prefilling): the adopted count was taken at admission, not re-asked
    /// (a prefix-cache lookup hashes the whole prompt) on every call.
    fn prefill_debt(&self) -> u32 {
        self.prefilling.iter().map(|p| self.prefill_owed(p)).sum()
    }

    /// Blocks `p` is still owed: its prompt and first token, less what it holds
    /// (or, before it holds any, what it adopts from the cache).
    fn prefill_owed(&self, p: &Prefilling) -> u32 {
        let held = match p.table.blocks().len() as u32 {
            0 => p.adopted,
            n => n,
        };
        self.dec.blocks_for(p.prompt.len() as u32 + 1).saturating_sub(held)
    }

    /// Blocks a fresh prefill of `prompt` will adopt from the prefix cache
    /// ([`PagedDecoder::cached_prefix_blocks`]). A prompt carrying embedding
    /// rows is charged in full: its placeholder ids are not the content the
    /// cache is keyed on, so the decoder's peek would overstate the hit.
    fn adopted_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>, embeds: bool) -> u32 {
        if embeds {
            return 0;
        }
        self.dec.cached_prefix_blocks(prompt, adapter)
    }

    /// Blocks this iteration has already spoken for: one more token for every
    /// running sequence and the unfinished prefills. Admission and speculative
    /// proposals may only use what is left.
    fn committed_blocks(&self) -> u32 {
        self.blocks_for_next_token() + self.prefill_debt()
    }

    /// Make the pool able to back everything this iteration has committed to -
    /// one more token for every active sequence and the rest of every prefill -
    /// or leave it with a single sequence, which fits by construction
    /// (`next_admissible` rejects a request larger than the pool).
    ///
    /// Admission only proves that a PROMPT fits, so the sequences it lets in can
    /// outgrow the pool together while they decode. The victims are chosen newest
    /// first - the sequences furthest from finishing, so the oldest keep running
    /// and release their blocks soonest - and a sequence still prefilling is
    /// always newer than one decoding:
    ///
    /// 1. A prefilling sequence loses its partial prefill and goes back to the
    ///    head of the waiting queue; it has produced nothing.
    /// 2. A decoding sequence is demoted to host RAM when the decoder can
    ///    ([`PagedDecoder::demote_sequence`]) and has budget left: a copy of
    ///    its cache costs less than recomputing it.
    /// 3. Otherwise (no offload, or the host budget is spent) it is preempted by
    ///    **recompute**: its blocks go back to the pool and it is requeued at the
    ///    head of the waiting queue with the tokens it has produced, the sampler's
    ///    RNG stream and its speculation window, to prefill its context again
    ///    once the pool has room. Recompute was chosen over reserving
    ///    `prompt + max_new` blocks for every admitted sequence because `max_new`
    ///    is a ceiling, not a forecast: most sequences stop earlier, so a worst
    ///    case reservation would keep the batch far below what the pool can
    ///    actually hold, on every request, to avoid a cost paid only under
    ///    overload. The continuation is the same tokens: the rebuilt cache holds
    ///    the same positions, the next input is the token already chosen, and a
    ///    sampled sequence resumes its own RNG where it stopped, so its seed
    ///    still reproduces it.
    ///
    /// The last active sequence is never preempted: it has nothing to yield to.
    fn ensure_decode_room(&mut self, report: &mut StepReport) {
        // The total is computed once and each victim's own share is subtracted when it leaves, so choosing V
        // victims among B sequences costs O(B + V), not O(B x V) from recounting the batch per victim.
        let mut need = self.committed_blocks();
        if self.dec.free_blocks() >= need {
            return;
        }
        self.dec.reclaim_prefix(need - self.dec.free_blocks());
        let mut active = self.running.iter().filter(|r| !r.done).count();
        let promoted: std::collections::HashSet<u64> = report.promoted.iter().copied().collect();
        while self.dec.free_blocks() < need {
            if let Some(p) = self.prefilling.pop_back() {
                need -= self.prefill_owed(&p);
                self.requeue_prefilling(p);
                debug_assert_eq!(need, self.committed_blocks(), "the running total follows every sequence that left");
                continue;
            }
            if active <= 1 {
                return;
            }
            // Prefer not to re-demote a sequence promoted in THIS iteration: it
            // is the newest entry in `running`, so the plain "newest first"
            // choice would pick it, pay both transfers and produce no token -
            // pure thrash. Only when nothing else can yield does it go too:
            // thrash beats a pool that cannot back the step.
            let pos = self
                .running
                .iter()
                .rposition(|r| !r.done && !promoted.contains(&r.id))
                .or_else(|| self.running.iter().rposition(|r| !r.done))
                .expect("at least two active sequences");
            let mut r = self.running.remove(pos);
            need -= self.next_token_blocks(&r);
            active -= 1;
            match self.dec.demote_sequence(r.id, &mut r.table) {
                Ok(_) => {
                    report.demoted.push(r.id);
                    self.swapped.push_back(r);
                }
                Err(_) => {
                    report.recomputed.push(r.id);
                    self.requeue_for_recompute(r);
                }
            }
            debug_assert_eq!(need, self.committed_blocks(), "the running total follows every sequence that left");
        }
    }

    /// Drop a partially prefilled request's blocks and put it back at the head
    /// of the waiting queue. A sequence resumed after recompute keeps its
    /// adapter pinned; a fresh one gives the pin back, since admission pins again.
    fn requeue_prefilling(&mut self, mut p: Prefilling) {
        self.dec.release_table(&mut p.table);
        if p.resume.is_none() {
            self.unpin(p.adapter);
        }
        let req = Request { prompt: p.prompt, max_new: p.max_new, stop: p.stop };
        self.waiting.push_front(Waiting { id: p.id, req, adapter: p.adapter, sample: p.sample, seed: p.seed, resume: p.resume, embeds: p.embeds, bad_token: None });
    }

    /// Preempt a decoding sequence by recompute (see [`Self::ensure_decode_room`]):
    /// free its blocks and queue it, ahead of everything waiting, to prefill the
    /// context its cache held. The last produced token is not in the cache yet -
    /// it is the next decode input - so it travels as `next_input` rather than
    /// as part of the context to prefill.
    fn requeue_for_recompute(&mut self, mut r: Running) {
        self.dec.release_table(&mut r.table);
        let Running { id, adapter, generated, logprobs, max_new, stop, next_input, mut context, spec, sample, seed, rng, think, embeds, cached_tokens, .. } = r;
        let uncached = context.pop();
        debug_assert_eq!(uncached, Some(next_input), "the newest token is the one with no cache entry yet");
        let req = Request { prompt: context, max_new, stop };
        self.waiting.push_front(Waiting { id, req, adapter, sample, seed, resume: Some(Resume { generated, logprobs, next_input, spec, rng, think, cached_tokens }), embeds, bad_token: None });
    }

    /// Index of the first waiting request that can be admitted this
    /// iteration, in arrival order.
    ///
    /// On the way, requests that can never be served are removed and reported
    /// rejected - a prompt that exceeds the engine's capacity, a token outside
    /// the vocabulary, an adapter the decoder does not serve - since leaving
    /// them would block the queue forever (or, before the capacity check,
    /// corrupt the block table). A request whose adapter is still loading is
    /// stepped over: it stays queued in place and is asked about again next
    /// iteration. Only adapter residency is skipped; a request that merely
    /// does not fit the pool yet still holds the line (the caller stops at
    /// it), exactly as before adapters existed.
    fn next_admissible(&mut self, report: &mut StepReport) -> Option<usize> {
        let mut pos = 0;
        while let Some(w) = self.waiting.get(pos) {
            let (id, adapter) = (w.id, w.adapter);
            // Already admitted once (so it passed every check below) and now
            // waiting only for pool room to rebuild its cache.
            if w.resume.is_some() {
                return Some(pos);
            }
            let verdict = match self.queued_refusal(w.bad_token, w.req.prompt.len(), w.req.max_new) {
                Some(reason) => Admissibility::Refused(reason),
                None => self.adapter_verdict(adapter),
            };
            match verdict {
                Admissibility::Admissible if self.waits_for_shared_prefill(pos) => pos += 1,
                Admissibility::Admissible => return Some(pos),
                Admissibility::AdapterLoading => pos += 1,
                Admissibility::Refused(reject) => {
                    self.waiting.remove(pos);
                    self.started.remove(&id);
                    report.rejected.push((id, reject));
                }
            }
        }
        None
    }

    /// Whether the waiting request at `pos` should let a sequence still
    /// prefilling the start of its prompt get past it first.
    ///
    /// Requests that arrive together and share a document (one system prompt,
    /// one file, different questions) would each prefill it, since none of
    /// them is cached when they are admitted. When the shared part is at least
    /// two prefill slices, the scheduler asks the decoder to make it resumable
    /// as soon as the earlier prefill has passed it
    /// ([`PagedDecoder::hint_shared_prefix`]) and, if it agrees, steps over the
    /// request until then; it is admitted afterwards and resumes from the
    /// prefix. The wait is bounded by that one prefill's progress, and a
    /// decoder that declines is not asked again for that prefill.
    fn waits_for_shared_prefill(&mut self, pos: usize) -> bool {
        let Some(slice) = self.dec.prefill_interleave_tokens() else { return false };
        let w = &self.waiting[pos];
        if w.embeds.is_some() {
            return false;
        }
        let shared = |p: &Prefilling| p.prompt.iter().zip(&w.req.prompt).take_while(|(a, b)| a == b).count();
        let best = self
            .prefilling
            .iter()
            .enumerate()
            .filter(|(_, p)| p.adapter == w.adapter && p.embeds.is_none() && p.shared_hint != Some(None))
            .map(|(i, p)| (i, shared(p)))
            .filter(|&(i, at)| at >= 2 * slice && self.prefilling[i].done < at)
            .max_by_key(|&(_, at)| at);
        let Some((i, at)) = best else { return false };
        let p = &mut self.prefilling[i];
        match p.shared_hint {
            Some(Some(filed)) if filed <= at => true,
            _ => {
                let agreed = self.dec.hint_shared_prefix(&mut p.table, &p.prompt, at);
                p.shared_hint = Some(agreed.then_some(at));
                agreed
            }
        }
    }

    /// Ask the decoder to make the prefix the newest prefilling sequence shares
    /// with a waiting request resumable as its prefill passes it
    /// ([`PagedDecoder::hint_shared_prefix`]), when that is at least two slices:
    /// a decoder that prefills one prompt at a time admits the waiting request
    /// only after this prefill, which then leaves the shared part cached.
    fn hint_for_waiting_sharers(&mut self) {
        let Some(slice) = self.dec.prefill_interleave_tokens() else { return };
        let Some(p) = self.prefilling.back_mut() else { return };
        if p.embeds.is_some() {
            return;
        }
        let at = self
            .waiting
            .iter()
            .filter(|w| w.adapter == p.adapter && w.embeds.is_none())
            .map(|w| w.req.prompt.iter().zip(&p.prompt).take_while(|(a, b)| a == b).count())
            .max()
            .filter(|&at| at >= 2 * slice);
        if let Some(at) = at {
            let agreed = self.dec.hint_shared_prefix(&mut p.table, &p.prompt, at);
            p.shared_hint = Some(agreed.then_some(at));
        }
    }

    /// Whether `req` (for `adapter`) can ever be served by this decoder, without queueing it:
    /// `None` when it can, or when only its adapter is still loading; otherwise why not. The
    /// scheduler's admission applies the same test, so a caller that asks first gets the
    /// refusal the queue would have given it - without waiting for a slot to find out.
    pub fn refusal(&mut self, req: &Request, adapter: Option<AdapterId>) -> Option<RejectReason> {
        if let Some(reason) = self.queued_refusal(self.invalid_token(&req.prompt), req.prompt.len(), req.max_new) {
            return Some(reason);
        }
        match self.adapter_verdict(adapter) {
            Admissibility::Refused(reason) => Some(reason),
            Admissibility::Admissible | Admissibility::AdapterLoading => None,
        }
    }

    /// Why `embeds` cannot belong to a prompt of `req` for this decoder (a row
    /// outside the prompt, the wrong width, a decoder that takes tokens only),
    /// without queueing it: the refusal [`Self::submit_embedded`] would report.
    pub fn embeds_refusal(&self, req: &Request, embeds: &PromptEmbeds) -> Option<RejectReason> {
        embeds.refusal(req.prompt.len(), self.dec.embed_width()).map(|why| RejectReason::InvalidEmbeds { why })
    }

    /// The first token of `prompt` outside the vocabulary, if any. O(prompt): done once per request, at
    /// submission, not on every admission pass.
    fn invalid_token(&self, prompt: &[u32]) -> Option<u32> {
        let vocab = self.dec.vocab() as u32;
        prompt.iter().copied().find(|&t| t >= vocab)
    }

    /// Why a queued prompt (of `prompt_len` tokens, `bad_token` its [`Self::invalid_token`]) can never be
    /// served: a token outside the vocabulary, or more than the engine's capacity. O(1).
    ///
    /// A sequence can only ever run alone in the pool, so it must fit it: the KV
    /// of every token but the last one it produces.
    fn queued_refusal(&self, bad_token: Option<u32>, prompt_len: usize, max_new: usize) -> Option<RejectReason> {
        if let Some(token) = bad_token {
            return Some(RejectReason::InvalidToken { token, vocab: self.dec.vocab() as u32 });
        }
        let need = prompt_len + max_new;
        let cap = self.sequence_capacity();
        (need.saturating_sub(1) > cap).then_some(RejectReason::ExceedsCapacity { need: need as u32, capacity: cap as u32 })
    }

    fn adapter_verdict(&mut self, adapter: Option<AdapterId>) -> Admissibility {
        match adapter.map(|a| (a, self.dec.adapter_availability(a))) {
            None | Some((_, AdapterAvailability::Resident)) => Admissibility::Admissible,
            Some((_, AdapterAvailability::Loading)) => Admissibility::AdapterLoading,
            Some((adapter, AdapterAvailability::Unknown)) => Admissibility::Refused(RejectReason::AdapterUnavailable { adapter }),
        }
    }

    /// A fully prefilled request becomes a running sequence: its first token is
    /// picked from the prefill's final hidden state.
    fn start_running(&mut self, p: Prefilling, hidden: Vec<f32>, first_candidates: Vec<(u32, f32)>, report: &mut StepReport) {
        let Prefilling { id, prompt, max_new, stop, adapter, sample, seed, table, resume, embeds, .. } = p;
        if let Some(Resume { generated, logprobs, next_input, spec, rng, think, cached_tokens }) = resume {
            // The cache is whole again; the token to decode next was chosen
            // before the preemption, so the prefill's own output is not used.
            let mut context = prompt;
            context.push(next_input);
            self.running.push(Running { id, table, adapter, generated, logprobs, max_new, stop, next_input, context, spec, done: false, sample, think, procs: ProcessorState::default(), seed, rng, embeds, cached_tokens, stop_token: None });
            return;
        }
        let cached_tokens = table.cached_rows() as usize;
        let mut rng = data::rng::Rng::new(seed);
        // `admit_greedy`/`admit_topk` are on-device for a decoder that has
        // one (`qwen3::serve::Engine`): neither ships a `[vocab]` block to
        // the host for this one-time-per-request first token, matching
        // the steady-state per-token decode loop.
        let think = sample.thinking.map(sampler::ThinkState::new);
        let (first, first_logprob) = if let Some(forced) = think.and_then(|t| t.forced_next()) {
            // A prompt that ends inside a think block whose budget is zero: the closing tag comes first.
            (forced, 0.0)
        } else if !sample.needs_candidates() {
            (first_candidates.first().map_or_else(|| self.dec.admit_greedy(&hidden), |&(token, _)| token), 0.0)
        } else {
            // The iteration's head pass gave the full candidate list when it was asked for candidates (see
            // `mixed_iteration`); a shorter list is the greedy pick and the head runs again.
            let candidates = if first_candidates.len() >= self.dec.topk_capacity().max(1) { first_candidates } else { self.dec.admit_topk(&hidden, self.dec.topk_capacity().max(1)) };
            sample_in_context(&candidates, sample, &SampleContext { context: &prompt, prompt_len: prompt.len() }, &mut rng)
        };
        let mut r = Running { id, table, adapter, generated: Vec::new(), logprobs: Vec::new(), max_new, stop, next_input: first, context: prompt, spec: speculate::SeqSpec::new(), done: false, sample, think, procs: ProcessorState::default(), seed, rng, embeds, cached_tokens, stop_token: None };
        accept_token(&mut r, first, first_logprob);
        report.admitted.push(id);
        self.running.push(r);
    }

    /// Move the first admissible waiting request into `prefilling` when the batch has
    /// room and the pool has blocks for its prompt (cached prefixes are reclaimed to
    /// make them), and `allow(prompt tokens)` agrees. Nothing is admitted while a
    /// preempted sequence is still waiting for its blocks - see `Scheduler::swapped`.
    /// A request that can never be admitted (see `next_admissible`) is rejected on the
    /// way and one whose adapter is still loading is stepped over.
    fn admit_one(&mut self, report: &mut StepReport, allow: impl FnOnce(usize) -> bool) -> bool {
        if !self.swapped.is_empty() || self.running.len() + self.prefilling.len() >= self.max_running {
            return false;
        }
        let Some(pos) = self.next_admissible(report) else { return false };
        let need = match self.waiting.get(pos) {
            Some(w) => w.req.prompt.len(),
            None => return false,
        };
        if !allow(need) {
            return false;
        }
        // The prompt, its first token, and what the sequences already in flight
        // are owed this iteration: admitting into those blocks would only get the
        // newcomer preempted before it produced anything. The blocks the prompt
        // adopts from the prefix cache are resident already: charging for them
        // again would refuse a long conversation's next turn for want of the room
        // its own history occupies, forever when nothing else is running.
        let cached = {
            let w = &self.waiting[pos];
            self.adopted_blocks(&w.req.prompt, w.adapter, w.embeds.is_some())
        };
        let want = self.dec.blocks_for(need as u32 + 1).saturating_sub(cached) + self.committed_blocks();
        let free = self.dec.free_blocks();
        if free < want {
            // Cached prefix blocks are reclaimable capacity: live sequences always
            // outrank the cache, but not the cache this very request reuses.
            let waiting = &self.waiting[pos];
            self.dec.reclaim_prefix_keeping(want - free, &waiting.req.prompt, waiting.adapter);
        }
        if self.dec.free_blocks() < want {
            return false;
        }
        let Waiting { id, req, adapter, sample, seed, resume, embeds, .. } = self.waiting.remove(pos).expect("admissible position");
        let table = match adapter {
            Some(adapter) => {
                // A resumed sequence never gave its pin back.
                if resume.is_none() {
                    self.dec.pin_adapter(adapter);
                }
                BlockTable::with_namespace(adapter.kv_namespace())
            }
            None => BlockTable::new(),
        };
        self.prefilling.push_back(Prefilling { id, prompt: req.prompt, done: 0, adopted: cached, max_new: req.max_new, stop: req.stop, adapter, sample, seed, table, resume, embeds, shared_hint: None });
        self.hint_for_waiting_sharers();
        true
    }

    /// One iteration's prefill for a decoder that takes rows across sequences
    /// ([`PagedDecoder::prefill_batch_tokens`]): admit every waiting request that fits,
    /// then hand the decoder the next slices of the oldest prefilling sequences in a
    /// single call. Each gets at most [`PagedDecoder::prefill_interleave_tokens`] of the
    /// budget while later ones wait for theirs; what is left over after every
    /// sequence has had its share goes to the oldest ones, so a lone sequence uses the
    /// whole budget.
    fn prefill_batched(&mut self, report: &mut StepReport) {
        while self.admit_one(report, |_| true) {}
        if self.prefilling.is_empty() {
            return;
        }
        let takes = self.allot_prefill(self.prefill_budget as usize);
        let mut jobs: Vec<PrefillJob<'_>> =
            self.prefilling.iter_mut().zip(&takes).filter(|(_, &take)| take > 0).map(|(p, &take)| PrefillJob { table: &mut p.table, prompt: &p.prompt, budget: take, embeds: p.embeds.as_deref() }).collect();
        let progress = catch_device_oom(|| self.dec.prefill_chunks(&mut jobs));
        drop(jobs);
        match progress {
            Ok(progress) => self.settle_prefill(&takes, progress, Vec::new(), report),
            Err(oom) => self.refuse_prefilling(report, oom),
        }
    }

    /// Split `budget` prompt rows over the prefilling sequences, oldest first: each gets at most
    /// [`PagedDecoder::prefill_interleave_tokens`] while later ones wait for theirs, and what is left after every
    /// sequence has had its share goes to the oldest. One entry per prefilling sequence; O(prefilling).
    fn allot_prefill(&self, budget: usize) -> Vec<usize> {
        let share = self.dec.prefill_interleave_tokens().unwrap_or(usize::MAX).max(1);
        let mut left = budget;
        let mut takes = vec![0usize; self.prefilling.len()];
        for pass_cap in [share, usize::MAX] {
            for (p, take) in self.prefilling.iter().zip(&mut takes) {
                let room = (p.prompt.len() - p.done - *take).min(pass_cap.saturating_sub(*take)).min(left);
                *take += room;
                left -= room;
            }
        }
        takes
    }

    /// Record what a prefill call answered for the sequences `takes` gave rows to (one answer per non-zero
    /// take, in order): advance the partial ones and start decoding the ones whose prompt is in, oldest first.
    ///
    /// `first` is [`MixedOutcome::first`] (one entry per answer, or none at all when the decoder gave none).
    fn settle_prefill(&mut self, takes: &[usize], progress: Vec<PrefillProgress>, first: Vec<Vec<(u32, f32)>>, report: &mut StepReport) {
        assert_eq!(progress.len(), takes.iter().filter(|&&t| t > 0).count(), "the prefill call answers every job");
        assert!(first.is_empty() || first.len() == progress.len(), "a decoder gives the first tokens of every job or of none");
        let mut finished = Vec::new();
        let mut answers = progress.into_iter().enumerate();
        let mut first = first.into_iter();
        for (idx, p) in self.prefilling.iter_mut().enumerate() {
            if takes[idx] == 0 {
                continue;
            }
            let (_, answer) = answers.next().expect("one answer per job");
            let candidates = first.next().unwrap_or_default();
            match answer {
                PrefillProgress::Partial { consumed } => p.done += consumed,
                PrefillProgress::Done(hidden) => finished.push((idx, hidden, candidates)),
            }
        }
        // Oldest first, as the one-sequence loop starts them.
        let mut starting: Vec<(Prefilling, Vec<f32>, FirstCandidates)> = Vec::new();
        for (idx, hidden, candidates) in finished.into_iter().rev() {
            starting.push((self.prefilling.remove(idx).expect("index just visited"), hidden, candidates));
        }
        for (p, hidden, candidates) in starting.into_iter().rev() {
            self.start_running(p, hidden, candidates, report);
        }
    }

    /// One iteration's prefill for a decoder that reads prompts in while it
    /// decodes ([`PagedDecoder::background_prefill`]): admit every waiting
    /// request that fits, then hand the decoder every prompt still being read,
    /// oldest first, each with what it has left. It waits for them only when
    /// nothing will decode this iteration.
    fn prefill_in_background(&mut self, report: &mut StepReport) {
        while self.admit_one(report, |_| true) {}
        if self.prefilling.is_empty() {
            return;
        }
        let wait = self.running.iter().all(|r| r.done);
        let mut jobs: Vec<PrefillJob<'_>> =
            self.prefilling.iter_mut().map(|p| PrefillJob { table: &mut p.table, prompt: &p.prompt, budget: p.prompt.len() - p.done, embeds: p.embeds.as_deref() }).collect();
        let progress = catch_device_oom(|| self.dec.prefill_background(&mut jobs, wait));
        drop(jobs);
        let progress = match progress {
            Ok(progress) => progress,
            Err(oom) => return self.refuse_prefilling(report, oom),
        };
        assert_eq!(progress.len(), self.prefilling.len(), "prefill_background answers every job");
        let mut finished = Vec::new();
        for (idx, (p, answer)) in self.prefilling.iter_mut().zip(progress).enumerate() {
            match answer {
                PrefillProgress::Partial { consumed } => p.done += consumed,
                PrefillProgress::Done(hidden) => finished.push((idx, hidden)),
            }
        }
        // Oldest first, as the other prefill paths start them.
        let mut starting: Vec<(Prefilling, Vec<f32>)> = Vec::new();
        for (idx, hidden) in finished.into_iter().rev() {
            starting.push((self.prefilling.remove(idx).expect("index just visited"), hidden));
        }
        for (p, hidden) in starting.into_iter().rev() {
            self.start_running(p, hidden, Vec::new(), report);
        }
    }

    /// A prefill call ran the card out of memory: the prompts being read in are
    /// refused (their blocks and adapter pins returned) and the decoder drops its
    /// prefill work, while everything already decoding goes on. All of them, not
    /// only the one that hit the wall: the call's answers for the others are lost
    /// with its unwind, and a prompt half read in cannot be resumed from a state
    /// nobody reported.
    fn refuse_prefilling(&mut self, report: &mut StepReport, oom: gpu_core::DeviceOutOfMemory) {
        let bytes = oom.bytes;
        self.dec.abandon_prefill();
        for mut p in std::mem::take(&mut self.prefilling) {
            self.dec.release_table(&mut p.table);
            self.unpin(p.adapter);
            report.rejected.push((p.id, RejectReason::DeviceOutOfMemory { bytes }));
        }
    }

    fn step_inner(&mut self, report: &mut StepReport) -> Vec<(u64, Vec<u32>)> {
        // Named ranges on the driver-native profiler's timeline (NVTX under Nsight): one per iteration, one per phase.
        let _step = gpu_core::markers::range(c"serve.step");
        // 0. Resume preempted sequences before anything else touches the pool:
        //    work already in flight outranks work still queued.
        self.resume_swapped(report);

        // Prompts that begin now prepare the drafter only if speculation will repay it.
        if self.spec_policy.enabled && self.dec.spec_max_draft() > 0 && self.dec.spec_cost_aware() {
            let active = (self.running.len() + self.prefilling.len() + self.waiting.len()).clamp(1, self.max_running.max(1));
            let prepare = self.prefill_gate.prepare(&self.gate, active, self.dec.spec_prefill_cost(), self.dec.step_latency_target().map(|t| t.as_secs_f64()));
            self.dec.set_spec_prefill(prepare);
        }

        // 1. Admit while there's batch room, enough free blocks for the prompt,
        //    and prefill budget left this iteration (head-of-line guard: decode
        //    must run between bursts of admissions). Nothing new is admitted
        //    while a preempted sequence is still waiting for its blocks - see
        //    `Scheduler::swapped`.
        let prefill_range = gpu_core::markers::range(c"serve.admit_prefill");
        let interleaved = self.dec.prefill_interleave_tokens().is_some();
        let mut budget_left = self.prefill_budget as usize;
        let mut worked = false;
        let batched = self.dec.prefill_batch_tokens().is_some();
        let mixed = self.dec.mixed_step();
        if mixed.is_some() {
            // The prompts are read in by the iteration's one forward (below), beside the decode rows.
            while self.admit_one(report, |_| true) {}
        } else if self.dec.background_prefill() {
            self.prefill_in_background(report);
        } else if batched {
            self.prefill_batched(report);
        } else {
            loop {
                // The oldest request still being prefilled gets the budget first.
                if let Some(p) = self.prefilling.front_mut() {
                    if worked && budget_left == 0 {
                        break;
                    }
                    let slice = budget_left.max(1);
                    let progress = catch_device_oom(|| match p.embeds.as_deref() {
                        Some(e) => self.dec.prefill_chunk_embeds(&mut p.table, &p.prompt, e, slice),
                        None => self.dec.prefill_chunk(&mut p.table, &p.prompt, slice),
                    });
                    let progress = match progress {
                        Ok(progress) => progress,
                        Err(oom) => {
                            self.refuse_prefilling(report, oom);
                            break;
                        }
                    };
                    let consumed = match progress {
                        PrefillProgress::Partial { consumed } => {
                            p.done += consumed;
                            consumed
                        }
                        PrefillProgress::Done(hidden) => {
                            let consumed = p.prompt.len() - p.done;
                            let p = self.prefilling.pop_front().expect("the front was just prefilled");
                            self.start_running(p, hidden, Vec::new(), report);
                            consumed
                        }
                    };
                    worked = true;
                    budget_left = budget_left.saturating_sub(consumed);
                    continue;
                }
                // Admit while there's batch room and enough free blocks for the prompt.
                //
                // Always start at least one request per iteration (no starvation); after
                // that, stop once the budget is spent. A decoder that prefills in slices
                // spends it slice by slice.
                let (worked_now, left) = (worked, budget_left);
                if !self.admit_one(report, |need| !(worked_now && (left == 0 || (!interleaved && need > left)))) {
                    break;
                }
            }
        }

        // 2. Make room for this iteration's decode. Admission only checks that
        //    a PROMPT fits; a sequence that has been decoding for a while can
        //    still walk the pool dry mid-batch, and `BlockTable::append` has
        //    nowhere to go when it does. Preempt (swap out) the most recently
        //    admitted sequences until every remaining one can append its next
        //    token - the classic "newest first" victim choice, so the sequences
        //    closest to finishing (and to releasing their blocks) are the ones
        //    that keep running.
        drop(prefill_range);
        self.ensure_decode_room(report);

        // 3. Batched decode over every running (not-done) sequence. When
        //    nothing is waiting to be admitted, decode a WINDOW of tokens per
        //    host round-trip (A4): the readback-per-token becomes a readback
        //    per window, at the cost of up to window-1 wasted decode steps for
        //    a sequence that hits EOS mid-window (its surplus K/V is rolled
        //    back below). With work waiting, the window stays 1 so admission
        //    latency is never traded away silently.
        let active: Vec<usize> = (0..self.running.len()).filter(|&i| !self.running[i].done).collect();
        // Membership as a mask: the decode step picks the batch's tables out of `running` in one pass.
        let mut in_batch = vec![false; self.running.len()];
        for &i in &active {
            in_batch[i] = true;
        }
        let decode_range = gpu_core::markers::range(c"serve.decode");
        'decode: {
        if let Some(limits) = mixed.filter(|_| !self.prefilling.is_empty()) {
            self.mixed_iteration(limits, &active, &in_batch, report);
            break 'decode;
        }
        if active.is_empty() || self.speculative_step(&active) {
            break 'decode;
        }
        {
            let plain_started = (self.clock)();
            let generated_before: usize = active.iter().map(|&i| self.running[i].generated.len()).sum();
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
            //
            // The no-repeat filter bans a handful of tokens known from the history before the step: a
            // step whose history bans none is plain greedy, one that bans some asks for that many extra
            // candidates and takes the best that is not banned.
            let (all_greedy, banned_max) = self.sampling_mode(&active);
            // A thinking budget is checked on the host after every token, so such a batch takes no window either;
            // neither does a filter that must see each token before the next is chosen.
            let host_checked = active.iter().any(|&i| self.running[i].think.is_some_and(|t| t.watching()) || self.running[i].sample.no_repeat.active());
            let mut k = if self.waiting.is_empty() && self.prefilling.is_empty() && all_greedy && !host_checked { remaining_min.min(self.dec.decode_window_capacity()) } else { 1 };
            // Every append must succeed mid-window (no host decisions there):
            // require a comfortable block reserve, else fall back to one step.
            if k > 1 && (self.dec.free_blocks() as usize) < active.len() * k {
                k = 1;
            }
            let decoded = catch_device_oom(|| -> Vec<Vec<(u32, f32)>> { if all_greedy {
                let mut refs = active_tables(&mut self.running, &in_batch);
                // Greedy tokens are deterministic: probability one, logprob zero.
                let greedy = if k > 1 {
                    self.dec.forward_batched_greedy_window(&mut refs, &inputs, k)
                } else {
                    self.dec.forward_batched_greedy(&mut refs, &inputs).into_iter().map(|t| vec![t]).collect()
                };
                greedy.into_iter().map(|row| row.into_iter().map(|t| (t, 0.0)).collect()).collect()
            } else {
                let wanted = self.candidates_wanted(&active, banned_max);
                let candidates = {
                    let mut refs = active_tables(&mut self.running, &in_batch);
                    self.dec.forward_batched_topk(&mut refs, &inputs, wanted)
                };
                let _sample = gpu_core::markers::range(c"serve.sample");
                sample_rows(&mut self.running, &active, &candidates)
            }});
            let window = match decoded {
                Ok(window) => window,
                Err(oom) => {
                    // The step ran the card out of memory: the sequences it carried are refused with a classed
                    // error (their blocks returned) and the rest of the server goes on. All of them, since the
                    // step's answers for the others are lost with its unwind.
                    for &i in active.iter().rev() {
                        let mut r = self.running.remove(i);
                        self.dec.release_table(&mut r.table);
                        self.unpin(r.adapter);
                        report.rejected.push((r.id, RejectReason::DeviceOutOfMemory { bytes: oom.bytes }));
                    }
                    break 'decode;
                }
            };
            let accepted = gpu_core::markers::range(c"serve.accept");
            self.accept_window(&active, &window, k);
            drop(accepted);
            if self.dec.spec_cost_aware() {
                let produced: usize = active.iter().map(|&i| self.running[i].generated.len()).sum::<usize>() - generated_before;
                self.gate.observe_plain(speculate::CostGate::bucket(active.len()), produced, (self.clock)() - plain_started);
            }
        }
        }

        drop(decode_range);
        // 4. Reap completed sequences, returning their blocks to the pool.
        let mut completed = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].done {
                let mut r = self.running.remove(i);
                self.dec.release_table(&mut r.table);
                self.unpin(r.adapter);
                report.completed_logprobs.push((r.id, r.logprobs));
                report.completed_meta.push((r.id, Completion { cached_tokens: r.cached_tokens, stop_token: r.stop_token }));
                self.prefill_gate.observe_reply(r.generated.len());
                completed.push((r.id, r.generated));
            } else {
                i += 1;
            }
        }
        completed
    }

    /// `(every row is plain greedy, most tokens any row's no-repeat filter bans)` for the sequences `active`.
    /// The filter bans a handful of tokens known from the history before the step: a step whose history bans
    /// none is plain greedy, one that bans some asks for that many extra candidates and takes the best that is
    /// not banned.
    fn sampling_mode(&self, active: &[usize]) -> (bool, usize) {
        let banned_max = active.iter().map(|&i| self.running[i].sample.no_repeat.banned(&self.running[i].generated).len()).max().unwrap_or(0);
        let all_greedy = banned_max == 0 && active.iter().all(|&i| !self.running[i].sample.needs_candidates());
        (all_greedy, banned_max)
    }

    /// Candidates per row a non-greedy step asks for. Real sampling and the penalties read the whole candidate
    /// list; a step that only has banned tokens to step over needs those and one more.
    fn candidates_wanted(&self, active: &[usize], banned_max: usize) -> usize {
        if active.iter().any(|&i| self.running[i].sample.needs_candidates()) {
            self.dec.topk_capacity()
        } else {
            (banned_max + 1).min(self.dec.topk_capacity())
        }
    }

    /// Take the tokens a decode step produced (`window[bi]` for sequence `active[bi]`, at most `k` each) into
    /// their sequences, and roll back the slots a sequence that finished mid-window did not use.
    fn accept_window(&mut self, active: &[usize], window: &[Vec<(u32, f32)>], k: usize) {
        for (bi, &si) in active.iter().enumerate() {
            let r = &mut self.running[si];
            let mut used = 0usize;
            for &(next, logprob) in &window[bi] {
                // The budget ran out: the closing sequence replaces the model's choice.
                let (next, logprob) = match r.think.and_then(|t| t.forced_next()) {
                    Some(forced) => (forced, 0.0),
                    None => (next, logprob),
                };
                // `used` counts consumed window slots (a real decode step
                // happened and wrote a KV-cache entry) regardless of
                // whether `next` ends up in `generated` - EOS still
                // occupies a real slot, it just isn't output text.
                used += 1;
                accept_token(r, next, logprob);
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

    /// **One mixed iteration** ([`PagedDecoder::mixed_step`]): the next token of every sequence in `active` and
    /// a slice of the prompts still being read in, as one forward. The slice is sized from the decoder's
    /// latency target and the cost of the iterations timed so far ([`StepCost::prompt_rows`]), and shared out
    /// over the prefilling sequences as [`Self::allot_prefill`] does. A decoder that speculates has the decode rows verify
    /// their drafts in the same forward ([`PagedDecoder::forward_mixed_verify`]).
    fn mixed_iteration(&mut self, limits: MixedLimits, active: &[usize], in_batch: &[bool], report: &mut StepReport) {
        let target = self.dec.step_latency_target().map(|t| t.as_secs_f64());
        // Drafting comes first: the verify rows are decode rows of the forward, and the prompt slice is what the
        // target leaves after them.
        let proposals = self.propose(active, false);
        let decode_rows = proposals.as_ref().map_or(active.len(), |p| p.rows.iter().map(Vec::len).sum());
        let started = (self.clock)();
        let prompt_rows = self.step_cost.prompt_rows(limits, decode_rows, target);
        let takes = self.allot_prefill(prompt_rows);
        let inputs: Vec<u32> = active.iter().map(|&i| self.running[i].next_input).collect();
        let (mut all_greedy, banned_max) = self.sampling_mode(active);
        // A prompt that ends in this iteration has its first token taken from the same head pass as the decode rows'
        // ([`MixedOutcome::first`]), so the pass asks for what that prompt's sampling needs too.
        let finishing_needs_candidates = self.prefilling.iter().zip(&takes).any(|(p, &take)| take > 0 && p.done + take >= p.prompt.len() && p.sample.needs_candidates());
        all_greedy &= !finishing_needs_candidates;
        let cands = if all_greedy {
            0
        } else if finishing_needs_candidates {
            self.dec.topk_capacity()
        } else {
            self.candidates_wanted(active, banned_max)
        };
        let mut tables = active_tables(&mut self.running, in_batch);
        let mut jobs: Vec<PrefillJob<'_>> =
            self.prefilling.iter_mut().zip(&takes).filter(|(_, &take)| take > 0).map(|(p, &take)| PrefillJob { table: &mut p.table, prompt: &p.prompt, budget: take, embeds: p.embeds.as_deref() }).collect();
        let outcome = {
            let _forward = gpu_core::markers::range(c"serve.mixed");
            catch_device_oom(|| match &proposals {
                Some(p) => {
                    let o = self.dec.forward_mixed_verify(&mut tables, &p.rows, cands.max(p.cands), &mut jobs);
                    (o.scored, Vec::new(), o.prefilled, o.first)
                }
                None => {
                    let o = self.dec.forward_mixed(&mut tables, &inputs, cands, &mut jobs);
                    (Vec::new(), o.decoded, o.prefilled, o.first)
                }
            })
        };
        drop((tables, jobs));
        let (scored, decoded, prefilled, first) = match outcome {
            Ok(outcome) => outcome,
            Err(oom) => {
                // The forward carried every sequence and prompt in flight and its answers are lost with the unwind:
                // all are refused with a classed error (blocks returned), and the server goes on.
                for &i in active.iter().rev() {
                    let mut r = self.running.remove(i);
                    self.dec.release_table(&mut r.table);
                    self.unpin(r.adapter);
                    report.rejected.push((r.id, RejectReason::DeviceOutOfMemory { bytes: oom.bytes }));
                }
                return self.refuse_prefilling(report, oom);
            }
        };
        let rows = decode_rows + takes.iter().sum::<usize>();
        self.step_cost.observe(rows, (self.clock)() - started);
        match proposals {
            Some(proposals) => self.settle_verified(active, proposals, scored),
            None => {
                let sampled = gpu_core::markers::range(c"serve.sample");
                let window = if all_greedy { decoded } else { sample_rows(&mut self.running, active, &decoded) };
                drop(sampled);
                let accepted = gpu_core::markers::range(c"serve.accept");
                self.accept_window(active, &window, 1);
                drop(accepted);
            }
        }
        self.settle_prefill(&takes, prefilled, first, report);
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
        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        let params = SampleParams { temp: 0.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() };
        assert_eq!(sample_from_topk(&candidates, params, &mut rng), 7);
    }

    #[test]
    fn sample_from_topk_top_k_1_is_deterministically_the_best_candidate() {
        // top_k=1 over a positive-temperature softmax must degenerate to
        // picking the single highest-value candidate, regardless of the RNG
        // draw -- there is only one candidate left to draw from.
        let candidates = [(3u32, 9.0f32), (1, 4.0), (2, -1.0)];
        let params = SampleParams { temp: 0.7, top_k: 1, top_p: 1.0, ..SampleParams::greedy() };
        for _ in 0..8 {
            let mut rng = data::rng::Rng::new(data::rng::random_seed());
            assert_eq!(sample_from_topk(&candidates, params, &mut rng), 3);
        }
    }

    #[test]
    fn sample_from_topk_never_returns_a_token_outside_the_candidate_set() {
        let candidates: Vec<(u32, f32)> = (0..40).map(|i| (100 + i, 40.0 - i as f32)).collect();
        let ids: std::collections::HashSet<u32> = candidates.iter().map(|&(id, _)| id).collect();
        let params = SampleParams { temp: 1.0, top_k: 10, top_p: 0.9, ..SampleParams::greedy() };
        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        for _ in 0..200 {
            let t = sample_from_topk(&candidates, params, &mut rng);
            assert!(ids.contains(&t), "sampled token {t} was never in the candidate set");
        }
    }

    /// min-p keeps the tokens whose probability is at least `min_p` times the best token's (after
    /// temperature, top-k and top-p) and never removes the best one; 0 leaves the set alone.
    #[test]
    fn min_p_drops_tokens_far_below_the_best() {
        let candidates = [(1u32, 2.0f32), (2, 1.9), (3, -3.0)];
        let plain = SampleParams { temp: 1.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() };
        let with = SampleParams { min_p: 0.5, ..plain };
        assert!(candidate_logprob(&candidates, plain, 3).is_finite(), "without min-p the tail token can be drawn");
        assert_eq!(candidate_logprob(&candidates, with, 3), f32::NEG_INFINITY, "its probability is far below half the best's");
        assert!(candidate_logprob(&candidates, with, 2).is_finite(), "0.91 of the best survives min_p 0.5");
        let extreme = SampleParams { min_p: 1.0, ..plain };
        assert!(candidate_logprob(&candidates, extreme, 1).is_finite() && candidate_logprob(&candidates, extreme, 2) == f32::NEG_INFINITY, "min_p 1 keeps only the best");
        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        for _ in 0..200 {
            assert_ne!(sample_from_topk(&candidates, with, &mut rng), 3);
        }
    }

    #[test]
    fn sample_from_topk_is_reproducible_for_a_fixed_seed() {
        let candidates: Vec<(u32, f32)> = (0..40).map(|i| (i, 40.0 - i as f32 * 0.3)).collect();
        let params = SampleParams { temp: 0.8, top_k: 20, top_p: 0.95, ..SampleParams::greedy() };
        let draw = |seed: u64| {
            let mut rng = data::rng::Rng::new(seed);
            (0..20).map(|_| sample_from_topk(&candidates, params, &mut rng)).collect::<Vec<u32>>()
        };
        assert_eq!(draw(1234), draw(1234));
        assert_ne!(draw(1234), draw(5678), "different seeds should not collide on 20 draws");
    }

    /// The logprob returned alongside the token must be exactly `ln(chosen
    /// probability under the truncated, temperature-scaled distribution)` -
    /// the value `sample_from_topk` (which discards it) computed internally
    /// anyway, not a value re-derived by a second pass over `candidates`.
    #[test]
    fn sample_from_topk_with_logprob_matches_a_manual_softmax_over_the_kept_set() {
        let candidates = [(5u32, 2.0f32), (1, 1.0), (9, 0.5), (3, -3.0)];
        let params = SampleParams { temp: 0.7, top_k: 3, top_p: 1.0, ..SampleParams::greedy() };
        for _ in 0..16 {
            let seed = data::rng::random_seed();
            let mut rng = data::rng::Rng::new(seed);
            let (tok, logprob) = sample_from_topk_with_logprob(&candidates, params, &mut rng);
            // Independent oracle: manual temperature-scaled softmax over the
            // top-3 candidates (top_k=3 keeps everything but the last).
            let kept = &candidates[..3];
            let scaled: Vec<f32> = kept.iter().map(|&(_, v)| v / params.temp).collect();
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|&s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let rank = kept.iter().position(|&(id, _)| id == tok).expect("sampled token must be in the kept set");
            let expected = exps[rank].ln() - sum.ln();
            assert!((logprob - expected).abs() < 1e-5, "seed {seed}: logprob {logprob} != expected {expected}");
        }
    }

    /// Greedy decoding is deterministic - the sampled token's "probability"
    /// is exactly `1.0`, so its logprob must be exactly `0.0`.
    #[test]
    fn sample_from_topk_with_logprob_is_zero_at_greedy() {
        let candidates = [(7u32, 3.0f32), (2, 5.0), (9, 1.0)];
        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        let params = SampleParams { temp: 0.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() };
        let (tok, logprob) = sample_from_topk_with_logprob(&candidates, params, &mut rng);
        assert_eq!(tok, 7); // first entry, matching sample_from_topk's greedy contract
        assert_eq!(logprob, 0.0);
    }

    #[test]
    fn sample_from_topk_top_p_narrows_to_the_nucleus() {
        // A sharply peaked distribution: candidate 0 alone carries ~most of
        // the mass at temp=1. A tight top_p must keep drawing it; top_p=1
        // (disabled) must occasionally draw further down the tail.
        let candidates = [(0u32, 20.0f32), (1, 1.0), (2, 0.5), (3, 0.1)];
        let narrow = SampleParams { temp: 1.0, top_k: 0, top_p: 0.5, ..SampleParams::greedy() };
        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        for _ in 0..50 {
            assert_eq!(sample_from_topk(&candidates, narrow, &mut rng), 0, "tight top_p must stay on the dominant candidate");
        }
    }

    /// The logprob a sampled token carries is the one of the truncated,
    /// temperature-scaled distribution it was drawn from, whichever path
    /// (plain sampling or speculative verification) produced it.
    #[test]
    fn candidate_logprob_is_the_probability_the_sampler_drew_with() {
        let candidates: Vec<(u32, f32)> = vec![(7, 3.0), (2, 2.0), (9, 1.0), (4, 0.5)];
        let params = SampleParams { temp: 0.8, top_k: 3, top_p: 0.95, ..SampleParams::greedy() };
        let mut rng = data::rng::Rng::new(data::rng::random_seed());
        for _ in 0..20 {
            let (token, drawn) = sample_from_topk_with_logprob(&candidates, params, &mut rng);
            assert!((candidate_logprob(&candidates, params, token) - drawn).abs() < 1e-6);
        }
        assert_eq!(candidate_logprob(&candidates, SampleParams::greedy(), 7), 0.0);
    }

    // ── accept_token ─────────────────────────────────────────────

    fn running(stop: &[u32], max_new: usize) -> Running {
        Running {
            id: 0,
            table: BlockTable::new(),
            adapter: None,
            generated: Vec::new(),
            logprobs: Vec::new(),
            max_new,
            stop: stop.to_vec(),
            next_input: 0,
            context: Vec::new(),
            spec: speculate::SeqSpec::new(),
            done: false,
            sample: SampleParams::greedy(),
            think: None,
            procs: ProcessorState::default(),
            seed: data::rng::random_seed(),
            rng: data::rng::Rng::new(data::rng::random_seed()),
            embeds: None,
            cached_tokens: 0,
            stop_token: None,
        }
    }

    #[test]
    fn accept_token_eos_marks_done_without_appending_it() {
        // Regression test: the EOS token id must never end up in `generated`
        // (it feeds tokens_of()/completed, which the tokenizer decodes back
        // to text - an appended EOS id decodes to its literal special-token
        // text, e.g. Qwen's "<|im_end|>", leaking into every response).
        let mut r = running(&[99], 100);
        r.generated.push(5); // a real token already generated this turn
        accept_token(&mut r, 99, 0.0);
        assert!(r.done, "EOS must mark the sequence done");
        assert_eq!(r.generated, vec![5], "EOS must not be appended to generated");
    }

    /// R1-Distill-Qwen declares two eos ids; each alone ends the sequence.
    #[test]
    fn any_stop_id_finishes() {
        for id in [151643, 151645] {
            let mut r = running(&[151643, 151645], 100);
            accept_token(&mut r, id, 0.0);
            assert!(r.done && r.generated.is_empty(), "{id}");
        }
    }

    #[test]
    fn accept_token_normal_token_is_appended_and_not_done() {
        let mut r = running(&[99], 100);
        accept_token(&mut r, 42, 0.0);
        assert!(!r.done);
        assert_eq!(r.generated, vec![42]);
        assert_eq!(r.next_input, 42);
    }

    #[test]
    fn accept_token_reaching_max_new_marks_done_and_still_appends() {
        // Unlike EOS, hitting the length cap on a real token must still
        // include that token in the output - only EOS is excluded.
        let mut r = running(&[99], 1);
        accept_token(&mut r, 42, 0.0);
        assert!(r.done);
        assert_eq!(r.generated, vec![42]);
    }

    #[test]
    fn accept_token_with_no_eos_configured_never_stops_on_token_value() {
        let mut r = running(&[], 100);
        accept_token(&mut r, 99, 0.0); // would be EOS if configured
        assert!(!r.done);
        assert_eq!(r.generated, vec![99]);
    }

    /// A hand-worked case matching vLLM's own `NoRepeatNGramLogitsProcessor`
    /// semantics exactly: `ngram_size = 3` means the ban key is the trailing
    /// TWO-token prefix. History `[1,2,3, 4,5, 1,2,3, 4,5, 1,2]` ends in the
    /// prefix `(1,2)`; that pair occurred twice before (positions 0 and 5),
    /// each time followed by `3` - so `3` alone is banned, not `4` or `5`
    /// (which never immediately follow `(1,2)`).
    #[test]
    fn no_repeat_ngram_bans_exactly_the_token_that_completes_a_seen_ngram() {
        let history = [1u32, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2];
        let mut logits = vec![0.0f32; 6];
        apply_no_repeat_ngram(&mut logits, &history, 3, 90, &std::collections::HashSet::new());
        assert_eq!(logits[3], f32::NEG_INFINITY, "token 3 completes the twice-seen (1,2) ngram and must be banned");
        assert_eq!(logits[4], 0.0, "token 4 never immediately follows (1,2) and must be untouched");
        assert_eq!(logits[5], 0.0, "token 5 never immediately follows (1,2) and must be untouched");
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 0.0);
    }

    /// The whitelist exempts a token from the ban even though it would
    /// otherwise complete a seen n-gram - upstream's own reason for this
    /// (DeepSeek-OCR's `<td>`/`</td>` table tags) is exactly this shape: a
    /// short, LEGITIMATELY-repeating pair inside a real table.
    #[test]
    fn no_repeat_ngram_whitelist_exempts_an_otherwise_banned_token() {
        let history = [1u32, 2, 3, 1, 2];
        let mut logits = vec![0.0f32; 4];
        let whitelist: std::collections::HashSet<u32> = [3].into_iter().collect();
        apply_no_repeat_ngram(&mut logits, &history, 3, 90, &whitelist);
        assert_eq!(logits[3], 0.0, "3 is whitelisted, so it must survive despite completing the seen (1,2) ngram");
    }

    /// `window_size` bounds the lookback: a match older than `window_size`
    /// generated tokens must NOT be found, matching vLLM's own
    /// `search_start = max(0, len(output_ids) - window_size)`.
    #[test]
    fn no_repeat_ngram_respects_the_window() {
        // (1,2)->9 occurs at position 0, then 6 filler tokens, then the
        // current trailing prefix (1,2) again. With window_size=5 the match
        // at position 0 is outside the window (search_start = 10-5 = 5) so 9
        // must NOT be banned; with window_size=90 it must be.
        let history = [1u32, 2, 9, 0, 0, 0, 0, 0, 0, 1, 2];
        let mut narrow = vec![0.0f32; 10];
        apply_no_repeat_ngram(&mut narrow, &history, 3, 5, &std::collections::HashSet::new());
        assert_eq!(narrow[9], 0.0, "the (1,2)->9 match at position 0 is outside a 5-token window");

        let mut wide = vec![0.0f32; 10];
        apply_no_repeat_ngram(&mut wide, &history, 3, 90, &std::collections::HashSet::new());
        assert_eq!(wide[9], f32::NEG_INFINITY, "the same match must be found with a 90-token window");
    }

    /// Too few generated tokens to form even one n-gram: a no-op, not a panic
    /// (the reference's own early return, `len(output_ids) < ngram_size`).
    #[test]
    fn no_repeat_ngram_is_a_noop_below_ngram_size() {
        let history = [1u32, 2];
        let mut logits = vec![0.0f32; 4];
        apply_no_repeat_ngram(&mut logits, &history, 5, 90, &std::collections::HashSet::new());
        assert!(logits.iter().all(|&x| x == 0.0));
    }

    /// Upstream's own recipe values for `deepseek-ai/DeepSeek-OCR`
    /// (`ngram_size=30, window_size=90`) end to end: a 30-token span repeated
    /// verbatim (as a stuck decode would produce) must have its continuation
    /// banned the moment the SAME 29-token prefix recurs, and a genuine table
    /// alternating `<td>`/`</td>` every other token (a far shorter cycle than
    /// 30) must NOT be affected by the ngram gate at all - it never forms a
    /// repeated 29-token prefix in the first place, so the whitelist is
    /// belt-and-braces here, not what saves it.
    #[test]
    fn no_repeat_ngram_matches_the_deepseek_ocr_recipe_defaults() {
        const NGRAM: usize = 30;
        const WINDOW: usize = 90;
        let span: Vec<u32> = (100..129).collect(); // 29 distinct filler tokens
        let mut history = span.clone();
        history.push(7); // the span's own 30th token, first time through
        history.extend(span.clone()); // repeat the 29-token prefix verbatim
        let mut logits = vec![0.0f32; 200];
        apply_no_repeat_ngram(&mut logits, &history, NGRAM, WINDOW, &std::collections::HashSet::new());
        assert_eq!(logits[7], f32::NEG_INFINITY, "repeating the exact 29-token prefix must ban its historical continuation");

        // A real table: <td>(128821) content(50) </td>(128822) content(51) ...
        let mut table = Vec::new();
        for _ in 0..40 {
            table.extend([128821u32, 50, 128822, 51]);
        }
        let whitelist: std::collections::HashSet<u32> = [128821, 128822].into_iter().collect();
        let mut table_logits = vec![0.0f32; 128823];
        apply_no_repeat_ngram(&mut table_logits, &table, NGRAM, WINDOW, &whitelist);
        assert_eq!(table_logits[128821], 0.0, "<td> must survive inside its own real table");
        assert_eq!(table_logits[128822], 0.0, "</td> must survive inside its own real table");
    }
}
