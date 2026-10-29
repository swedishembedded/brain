// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The released-GGUF Qwen3.8-27B as a `model::serve::PagedDecoder`: the engine
//! `model::serve::Scheduler` drives, so this checkpoint is served by the same
//! continuous-batching loop as every other brain LLM rather than a decode loop
//! of its own.
//!
//! What the engine adds over [`Qwen35GgufInstance`] is only the sequence
//! bookkeeping a scheduler needs. The instance already holds `max_batch`
//! decode slots per stage (one GDN recurrent state per sequence, and a GQA
//! KV window of `cap` rows per slot or a shared block pool). A sequence's slot
//! is its `BlockTable::state_row`. In the default layout the allocator's
//! `block_size == cap`, exactly the layout `crate::serve::Engine` uses, so a
//! sequence's whole lifetime is one block and the slot is that block's id. In
//! the paged layout ([`super::Qwen35GgufResident::with_kv_block_rows`]) the
//! allocator hands out small blocks as the sequence grows, the slot comes from
//! a free list, and the engine publishes the table's blocks to the stages
//! before every forward. A decode iteration then names the
//! live sequences' slots and positions, and the instance runs them as one
//! batched step across the layer-sharded cards: weight reads are paid once
//! for the whole batch, and sequences join (prefill into a fresh slot) and
//! leave (block released) between steps without disturbing the others.
//!
//! The head is projected once per step for all rows and reduced on the device
//! (greedy pick or top-k), so a batch step reads back token ids, not a
//! `[batch, vocab]` logits block.

use std::collections::HashMap;

use gpu_core::DeviceStats;
use model::adapter::bank::{AdapterBankConfig, AdapterError, AdapterInfo, BankedEngine, UnloadStatus, VerifiedAdapter};
use model::dispatch::LORA_SLOT_NONE;
use model::paged::{BlockAllocator, BlockTable, KvNamespace, BASE_KV_NAMESPACE};
use model::serve::{AdapterAvailability, AdapterId, DraftQuery, MixedLimits, MixedOutcome, PagedDecoder, PrefillJob, PrefillProgress};

use crate::adapter_serving as adapters;

use model::snapshot_prefix::{Entry as CacheEntry, SnapshotPrefixCache};

/// Prompts shorter than this are not worth a snapshot row and a retained block.
const MIN_CACHED_TOKENS: usize = 64;

/// Prompt rows a mixed iteration always carries when a prompt waits (`MixedLimits::min_prompt_rows`): a round wide
/// enough to run as a tensor-core GEMM, so a prompt behind a busy decode batch still advances at a useful rate.
// perf-number: a 128-row round is the smallest the prefill GEMM kernels serve at tile width
const MIN_MIXED_PROMPT_ROWS: usize = 128;

/// Branch-point hints held for prompts not yet started; the oldest goes first (a hint whose prompt was cancelled
/// before it started is never taken).
const MAX_PREFIX_HINTS: usize = 16;

/// Trailing prompt tokens an entry leaves out. A chat prompt ends with the generation prompt (the assistant's role
/// header and, with thinking on, its opening tag); the next turn's prompt re-renders that position as the finished
/// assistant message, whose tokens differ (`<think>` followed by one newline against `<think>` followed by a blank
/// line). A recurrent layer resumes only from the state after exactly the cached tokens, so an entry for the whole
/// prompt never matched the next turn and every turn prefilled its full history. The entry stops short of the
/// header; the few tokens after it are recomputed.
const CHECKPOINT_TAIL: usize = 16;

/// A checkpoint sits on a whole Gated-DeltaNet chunk (64 tokens): a prompt resumed from it then runs the same chunk
/// partition as a cold prefill. A resume from a position inside a chunk rounded measurably differently.
const CHECKPOINT_ALIGN: usize = 64;

/// The prefix length a prompt of `len` tokens is cached at: short of its generation prompt (see [`CHECKPOINT_TAIL`]),
/// on a whole recurrent chunk. A later prompt sharing that prefix resumes from exactly this many tokens.
pub fn checkpoint_len(len: usize) -> usize {
    len.saturating_sub(CHECKPOINT_TAIL) / CHECKPOINT_ALIGN * CHECKPOINT_ALIGN
}

/// What the engine needs to resume a prompt from a cache entry besides its KV
/// rows and state snapshot.
struct Resume {
    /// The last prompt token's `[d_model]` hidden row, so a prompt that is
    /// exactly the cached one needs no compute.
    last_hidden: Vec<f32>,
    /// The MTP head's KV rows for positions `0..tokens.len() - 1` were filled
    /// when this prompt was prefilled. A prompt prefilled while the drafter's
    /// prefill feed was off (it did not pay) has none, and a prompt that resumes
    /// from such an entry must not copy them.
    mtp_valid: bool,
}

type PrefixCache = SnapshotPrefixCache<Resume>;
use super::background::{PrefillDone, PrefillTask, TaskStart};
use super::mtp::{FeedStart, FeedStep, PromptFeed};
use super::pace::PrefillPace;
use super::{KvLayout, KvWindow, MultiPrefillJob, Qwen35GgufInstance, MODEL};

/// Candidates a non-greedy pick can sample from. Real sampling is exact
/// within this set; it only bounds how deep into the vocabulary nucleus
/// sampling can reach (the same bound `crate::serve::Engine` uses).
const TOPK_CAPACITY: usize = 32;

/// The most prompt tokens one background task reads in before the engine chooses again which prompt goes next.
/// A prompt longer than this is read in slices, with every shorter one (a conversation's next turn, whose history
/// is cached) started between them instead of after the whole of it. A multiple of the prefill round and of the
/// recurrent chunk, so the rounds are the ones an unsliced prefill runs.
// perf-number: about 2 s of prefill at the measured 2.2k prompt tokens per second on one GH200
const BACKGROUND_SLICE_TOKENS: usize = 4096;

/// How many prompts that fit one slice start in a row while a longer prompt waits, before that prompt gets a slice:
/// a long prompt is delayed by a bounded number of short ones and never starved.
const MAX_SHORT_STREAK: u32 = 8;

/// The index of the prompt to read in next among those with `remaining` tokens left, oldest first: the oldest that
/// fits one slice, else the oldest; `streak` counts the short ones chosen in a row while a long one waited, and at
/// [`MAX_SHORT_STREAK`] the oldest long one is chosen instead.
fn choose_next(remaining: &[usize], streak: &mut u32) -> Option<usize> {
    let short = remaining.iter().position(|&r| r <= BACKGROUND_SLICE_TOKENS);
    let long = remaining.iter().position(|&r| r > BACKGROUND_SLICE_TOKENS);
    match (short, long) {
        (Some(i), Some(_)) if *streak < MAX_SHORT_STREAK => {
            *streak += 1;
            Some(i)
        }
        (_, Some(i)) => {
            *streak = 0;
            Some(i)
        }
        (short, None) => short,
    }
}

/// Weight of the newest sequence in the running mean of MTP prefill-feed time.
const FEED_EWMA: f64 = 0.3;

/// Pairs of a deferred MTP prompt feed run with each slice of prompt (and each draft call): two rounds of the
/// head, about 5 ms, so a thousand-token prompt's feed is spread over four decode steps.
// perf-number: a 128-row round of the head takes about 2.5 ms (host prep and projection 1.1, layer 1.4)
const FEED_ROWS_PER_SLICE: usize = 256;

/// Bound on the upload quanta one scoring call drives before reporting that the
/// adapter cannot become resident (see [`Qwen35GgufEngine::score_supervised`]).
const ADAPTER_UPLOAD_STEPS: u32 = 1 << 20;

/// [`Qwen35GgufInstance`] behind the paged-decoder seam. See the module doc.
pub struct Qwen35GgufEngine {
    inst: Qwen35GgufInstance,
    alloc: BlockAllocator,
    /// Recurrent-state rows not owned by a sequence (paged layout only; in the
    /// one-block layout a sequence's row is its block's id).
    free_rows: Vec<u32>,
    /// Prompts whose KV rows and recurrent-state snapshot are kept for reuse
    /// (see [`super::prefix`]).
    cache: PrefixCache,
    /// A finished prefill's cache record, held until its sequence ends: the
    /// sequence's block becomes the cache entry's KV only then (decoding
    /// appends rows past the prompt and never rewrites the prompt's own).
    pending: HashMap<u32, Vec<Pending>>,
    /// Lengths of the entries a prefill still running filed at once (paged KV), whose MTP head rows are marked
    /// valid when its prompt is in.
    filed: HashMap<u32, Vec<usize>>,
    /// Prefixes a prompt not yet started is to file as branch points ([`Self::hint_branch`]).
    hints: Vec<(KvNamespace, Vec<u32>)>,
    /// Prompts being prefilled over several scheduler iterations, by slot.
    partial: HashMap<u32, PartialPrefill>,
    /// The sequences and row counts of the verify step awaiting its commit.
    verifying: Option<Verifying>,
    /// Running sequences past which a verify step no longer beats plain
    /// batched decode (`PagedDecoder::spec_max_batch`).
    spec_batch_limit: usize,
    /// Whether a new prompt feeds the MTP head ([`PagedDecoder::set_spec_prefill`]).
    spec_prefill: bool,
    /// Seconds one sequence's MTP prefill feed took, averaged over recent ones
    /// ([`PagedDecoder::spec_prefill_cost`]); `None` until one ran.
    feed_secs: Option<f64>,
    /// Wall time spent in the speculation hooks, for the benchmark.
    times: SpecTimes,
    prefill: PrefillPolicy,
    /// Whether an iteration with prompts to read in runs them and the decode rows as one forward
    /// ([`PagedDecoder::mixed_step`]); on wherever the instance can ([`Qwen35GgufInstance::mixed_available`]), and
    /// then it supersedes the background lane and its pacing.
    mixed: bool,
    /// Whether prompts are read in on each card's normal stream while decode
    /// steps run on its urgent one (see [`super::background`]): set when every
    /// stage's device can prioritise. Decode then never waits for a prompt.
    background_on: bool,
    /// The prompt slice being read in that way, if any.
    background: Option<Background>,
    /// Prompts that fit one slice started in a row while a longer one waited ([`MAX_SHORT_STREAK`]).
    short_streak: u32,
    /// Prompt feeds to the MTP head begun and not finished, oldest first.
    feeds: Vec<DeferredFeed>,
    /// How much device time a background prompt gets between decode steps.
    pace: PrefillPace,
    /// When the last paced slice ended: the next one's start less this is the decode iteration in between.
    slice_ended: Option<std::time::Instant>,
}

/// A prompt's feed to the MTP head that is run a few rounds at a time between decode steps instead of when the
/// prompt is in ([`Qwen35GgufEngine::advance_feeds`]), and what finishing it settles.
struct DeferredFeed {
    feed: PromptFeed,
    namespace: KvNamespace,
    /// The prompt: the cache entries filed from it are marked valid for the head by its tokens.
    prompt: Vec<u32>,
    /// Seconds the rounds run so far took.
    secs: f64,
}

/// How a slice that ends a prompt hands over to what follows it.
#[derive(Clone, Copy)]
struct SliceExit {
    /// The recurrent state is in the slot's pool row and not in the working copy (a multi-sequence forward leaves
    /// it there).
    state_in_pool: bool,
    /// The MTP head's feed of the prompt is begun now and run later ([`DeferredFeed`]): a sync feed is tens of
    /// milliseconds in which no running sequence decodes.
    defer_feed: bool,
}

/// A job's slice about to run: `(job index, slot, progress, end)`.
type LiveSlice = (usize, u32, PartialPrefill, usize);

/// A prompt slice read in while sequences decode, and what finishing it needs.
struct Background {
    task: PrefillTask,
    p: PartialPrefill,
    end: usize,
}

/// Run decode-side device work on the urgent lane when prompts are read in
/// beside it, so a decode step never queues behind a prefill round.
fn decode_lane<R>(background: bool, f: impl FnOnce() -> R) -> R {
    if background {
        gpu_core::lane::urgent(f)
    } else {
        f()
    }
}

/// Where a speculating engine spends its time: the drafter, the verify
/// forward (and its head), and the commit that settles it.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecTimes {
    pub draft: std::time::Duration,
    pub verify: std::time::Duration,
    pub commit: std::time::Duration,
    /// Time inside `prefill_chunk` (every slice, including the prefix-cache lookup).
    pub prefill: std::time::Duration,
}

/// A verify step awaiting its commit: who it carried, what rows, and the
/// hidden states they produced (the MTP head's input once rows are kept).
struct Verifying {
    slots: Vec<u32>,
    rows: Vec<Vec<u32>>,
    hidden: Vec<f32>,
}

/// How the engine spends prefill between decode steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPolicy {
    /// Prompt tokens one sequence gets per scheduler iteration while others wait:
    /// the stall every running sequence sees is the time to prefill the rows of
    // perf-number: measured by the `qwen35_spec_sweep` agent-mix scheduler harness
    /// one iteration (about 0.2 s at the default 512). On the 16-request agent mix
    /// 1024 is 7 percent faster than 512 in prompt tokens a second in the scheduler
    /// harness and 3 percent through the server (within its noise), and the first
    // perf-number: measured by the `qwen35_spec_sweep` agent-mix scheduler harness
    /// requests finish 8-10 s later; 256 is 17 percent slower than 512.
    pub slice_tokens: usize,
    /// Prompt rows one iteration prefills across sequences, in one forward (the
    /// GEMMs see all of them, so the weights are read once for the lot); `0` runs
    /// one sequence's slice at a time.
    ///
    /// Off by default: on the 16-request agent mix it gains 4 to 9 percent of
    // perf-number: measured by the `qwen35_spec_sweep` agent-mix scheduler harness
    /// prompt throughput in the scheduler harness and none through the server (63.3 s
    // perf-number: measured by the `qwen35_spec_sweep` agent-mix scheduler harness
    /// against 61.8 s), while the sequences advance together, so the first requests
    // perf-number: measured by the `qwen35_spec_sweep` agent-mix scheduler harness
    /// finish later (21 s against 11 s). It pays where the weights, not the host, bound
    /// a slice.
    pub batch_tokens: usize,
    /// The scheduler iteration length, decode step included, that a prompt read in beside decoding sequences is
    /// allowed to stretch their token interval to. Each iteration the prompt gets the device time the decode
    /// iteration leaves under this (and never less than a quarter of the iteration), recorded and waited for
    /// before the decode step, so the step runs on an idle card ([`super::pace`]). Only prompts read in while
    /// something decodes are paced: with nothing decoding a prompt goes in at the card's speed.
    // perf-number: a p99 token interval of 50 ms is the interactive budget; the mean iteration is held under 45
    pub step_target: std::time::Duration,
}

impl Default for PrefillPolicy {
    fn default() -> PrefillPolicy {
        PrefillPolicy { slice_tokens: 512, batch_tokens: 0, step_target: std::time::Duration::from_millis(45) }
    }
}

/// A prompt part of the way in.
struct PartialPrefill {
    /// Where the cached prefix ended (0 for a cold prompt).
    start: usize,
    /// Prompt tokens in so far.
    done: usize,
    /// No slice has run yet, so the working GDN state is the seeded/reset one.
    fresh: bool,
    cacheable: bool,
    prefix_hidden: Option<Vec<f32>>,
    /// Hidden states of the prefilled rows, for the MTP head.
    rows_hidden: Vec<f32>,
    /// Feed the MTP head this prompt: the engine's [`Qwen35GgufEngine::set_spec_prefill`]
    /// when the prompt began, and only on a prefix whose own head rows are valid.
    mtp_feed: bool,
    /// Where this prompt leaves a cached one it shares a long prefix with
    /// (`SnapshotPrefixCache::branch_point`): the state there is filed too, so the
    /// next prompt sharing that prefix resumes from it.
    branch: Option<usize>,
}

/// What a sequence's prefill leaves for the cache once the sequence is over.
struct Pending {
    /// A branch point ([`PartialPrefill::branch`]) rather than the prompt's checkpoint.
    branch: bool,
    tokens: Vec<u32>,
    snap_row: u32,
    last_hidden: Vec<f32>,
    /// The head's KV rows of this prompt were filled (see [`Resume::mtp_valid`]).
    mtp_valid: bool,
}

impl Qwen35GgufEngine {
    /// Wrap an activated instance. The instance's `max_batch` becomes the
    /// number of sequences that may be resident at once and its `cap` the
    /// per-sequence `prompt + max_new` limit.
    ///
    /// The one-block layout's allocator holds `max_batch + prefix_slots` blocks:
    /// the scheduler never runs more than `max_batch` sequences, so a cached
    /// prompt's retained block never starves an admission. The paged layout's
    /// holds the pool's blocks, and cached prompts' blocks are reclaimed for
    /// live sequences like any cache.
    pub fn new(inst: Qwen35GgufInstance) -> Qwen35GgufEngine {
        let (alloc, free_rows) = match inst.kv_layout {
            KvLayout::OneBlock => (BlockAllocator::new(inst.max_batch + inst.prefix_slots, inst.cap), Vec::new()),
            KvLayout::Paged { block_rows, pool_rows } => (BlockAllocator::new(pool_rows / block_rows, block_rows), (0..inst.max_batch).rev().collect()),
        };
        let cache = PrefixCache::new(inst.prefix_slots);
        let mixed = inst.mixed_available();
        let background_on = !mixed && inst.background_prefill_available();
        if background_on {
            inst.share_pools_across_lanes();
        }
        Qwen35GgufEngine {
            inst,
            alloc,
            free_rows,
            cache,
            pending: HashMap::new(),
            filed: HashMap::new(),
            hints: Vec::new(),
            partial: HashMap::new(),
            verifying: None,
            spec_batch_limit: usize::MAX,
            spec_prefill: true,
            feed_secs: None,
            times: SpecTimes::default(),
            prefill: PrefillPolicy::default(),
            mixed,
            background_on,
            background: None,
            short_streak: 0,
            feeds: Vec::new(),
            pace: PrefillPace::new(PrefillPolicy::default().step_target),
            slice_ended: None,
        }
    }

    /// Free the blocks of entries the cache retired.
    fn release_retired(&mut self, retired: impl IntoIterator<Item = BlockTable>) {
        for mut table in retired {
            table.release(&mut self.alloc);
        }
    }

    /// Give the sequence a recurrent-state row if it has none: the block's id in
    /// the one-block layout (so the row and the window are the same slot), a
    /// free row in the paged one. The table must already hold its blocks.
    fn assign_state_row(&mut self, table: &mut BlockTable) -> u32 {
        if let Some(row) = table.state_row() {
            return row;
        }
        let row = match self.inst.kv_layout {
            KvLayout::OneBlock => table.blocks()[0],
            KvLayout::Paged { .. } => self.free_rows.pop().unwrap_or_else(|| panic!("{MODEL}: more sequences than the {} recurrent-state rows", self.inst.max_batch)),
        };
        table.set_state_row(row);
        row
    }

    /// The sequence's recurrent-state row, with its KV blocks made known to
    /// every stage (a no-op for the one-block layout).
    fn slot(&mut self, table: &BlockTable) -> u32 {
        let row = table.state_row().expect("a sequence being run has been given a state row");
        self.inst.set_blocks(row, table.blocks());
        row
    }

    /// The KV window of a table, in the form the stages copy rows between.
    fn window<'a>(&self, table: &'a BlockTable) -> KvWindow<'a> {
        match self.inst.kv_layout {
            KvLayout::OneBlock => KvWindow::Block(table.blocks()[0]),
            KvLayout::Paged { .. } => KvWindow::Table(table.blocks()),
        }
    }

    /// Give every stage an adapter bank, so sequences can run under LoRA
    /// adapters registered with [`Self::load_adapter`] beside base sequences,
    /// in the same batched step, on this one resident copy of the weights.
    /// Call once, before serving. Device cost is [`Self::adapter_bank_bytes`],
    /// which the placement did not charge: size `cfg` for the headroom left.
    pub fn enable_adapters(&mut self, cfg: AdapterBankConfig) -> Result<(), AdapterError> {
        let rows = self.inst.max_batch.max(self.inst.prefill_round_rows());
        adapters::enable(self.inst.stage_models(), &cfg, rows)
    }

    /// Device bytes the adapter banks occupy across all stages; 0 without one.
    pub fn adapter_bank_bytes(&self) -> u64 {
        adapters::bytes(self.inst.stage_models())
    }

    /// Register a verified adapter (host memory only) and return the id the
    /// scheduler addresses it by. It reaches a slot on every stage when the
    /// scheduler first asks about it.
    pub fn load_adapter(&mut self, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
        adapters::register(self.inst.stage_models(), &self.inst.cfg, adapter)
    }

    /// Unload an adapter and drop its cached prompts. While sequences still
    /// run under it, it only stops admitting new ones.
    pub fn unload_adapter(&mut self, id: AdapterId) -> Result<UnloadStatus, AdapterError> {
        let (status, purged) = adapters::unload(self.inst.stage_models(), id)?;
        self.purge_namespace(purged);
        Ok(status)
    }

    /// Every registered adapter, oldest first.
    pub fn adapters(&self) -> Vec<AdapterInfo> {
        adapters::infos(self.inst.stage_models())
    }

    /// Teacher-forced per-token scores of one record under the base (`None`) or a
    /// registered adapter, keeping the positions `mask` supervises.
    ///
    /// A registered adapter is uploaded and pinned for the call exactly as a
    /// scheduler admits one, so what is scored is what serving that adapter
    /// produces; the pin is released before returning. The record goes through
    /// one chunked prefill - see
    /// [`Qwen35GgufInstance::teacher_forced_scores`], whose readout this is.
    pub fn score_supervised(&mut self, ids: &[u32], mask: &[bool], adapter: Option<AdapterId>) -> Result<Vec<crate::score::TokenScore>, String> {
        if ids.len() != mask.len() {
            return Err(format!("{MODEL}: a record has {} tokens and {} mask entries", ids.len(), mask.len()));
        }
        // The scoring prefill resets the instance's KV rows and recurrent state,
        // so no cached prompt describes what the cache says it does any more.
        self.clear_prefix_cache();
        let slot = match adapter {
            None => LORA_SLOT_NONE,
            Some(id) => {
                self.admit_adapter(id)?;
                adapters::slot_of(self.inst.stage_models(), id.kv_namespace())
            }
        };
        let scored = self.inst.teacher_forced_scores(ids, slot);
        if let Some(id) = adapter {
            let purged = adapters::unpin(self.inst.stage_models(), id);
            self.purge_namespace(purged);
        }
        Ok(scored?.into_iter().filter(|s| mask[s.position as usize]).collect())
    }

    /// Drop every cached prompt and give its retained block back to the allocator.
    fn clear_prefix_cache(&mut self) {
        while let Some(table) = self.cache.evict_lru() {
            self.release_retired([table]);
        }
    }

    /// Drive a registered adapter's upload to completion and pin it for one call.
    ///
    /// Each availability call moves one upload quantum. A bank with no slot for
    /// the adapter's rank answers [`AdapterAvailability::Loading`] forever, so the
    /// loop is bounded and reports that instead of spinning.
    fn admit_adapter(&mut self, id: AdapterId) -> Result<(), String> {
        for _ in 0..ADAPTER_UPLOAD_STEPS {
            let (answer, purged) = adapters::availability(self.inst.stage_models(), id);
            self.purge_namespace(purged);
            match answer {
                AdapterAvailability::Resident => {
                    adapters::pin(self.inst.stage_models(), id);
                    return Ok(());
                }
                AdapterAvailability::Loading => {}
                AdapterAvailability::Unknown => return Err(format!("{MODEL}: adapter {} is not registered on every stage", id.kv_namespace())),
            }
        }
        Err(format!("{MODEL}: adapter {} did not become resident within {ADAPTER_UPLOAD_STEPS} upload steps (its rank may not fit any free bank slot)", id.kv_namespace()))
    }

    /// Drop the cached prompts of an adapter that is gone.
    fn purge_namespace(&mut self, ns: impl IntoIterator<Item = KvNamespace>) {
        for ns in ns {
            let tables = self.cache.evict_namespace(ns);
            self.release_retired(tables);
        }
    }

    /// How prefill is spent between decode steps (see [`PrefillPolicy`]); call before
    /// the engine is handed to a scheduler.
    pub fn set_prefill_policy(&mut self, policy: PrefillPolicy) {
        assert!(policy.slice_tokens > 0, "{MODEL}: a prefill slice of 0 tokens never finishes a prompt");
        assert!(!policy.step_target.is_zero(), "{MODEL}: a step target of 0 leaves a prompt no device time");
        self.prefill = policy;
        self.pace = PrefillPace::new(policy.step_target);
    }

    /// Read prompts in on each card's normal stream while decode runs on its
    /// urgent one (`true`, the default wherever every stage's device can
    /// prioritise), or in slices between decode steps on one stream (`false`, the
    /// A/B baseline). Asking for it where it is unavailable keeps the slices.
    /// Call before the engine is handed to a scheduler.
    pub fn set_background_prefill(&mut self, on: bool) {
        assert!(self.background.is_none(), "{MODEL}: the prefill mode changed with a prompt being read in");
        // Asking for the background lane is asking for it instead of the mixed iteration.
        self.mixed &= !on;
        self.background_on = on && self.inst.background_prefill_available();
        if self.background_on {
            self.inst.share_pools_across_lanes();
        }
    }

    /// Run an iteration's decode rows and prompt slices as one forward (`true`, the default wherever the instance
    /// can, see [`Qwen35GgufInstance::mixed_available`]) or as separate passes (`false`: the background lane where
    /// every stage prioritises, slices between decode steps otherwise). Call before the engine is handed to a
    /// scheduler.
    pub fn set_mixed_step(&mut self, on: bool) {
        assert!(self.background.is_none(), "{MODEL}: the prefill mode changed with a prompt being read in");
        self.mixed = on && self.inst.mixed_available();
        if self.mixed {
            self.background_on = false;
        } else if !on {
            self.background_on = self.inst.background_prefill_available();
            if self.background_on {
                self.inst.share_pools_across_lanes();
            }
        }
    }

    /// Set the batch size past which the scheduler decodes plainly instead of
    /// speculating (the measured crossover; unbounded by default).
    pub fn set_spec_batch_limit(&mut self, limit: usize) {
        self.spec_batch_limit = limit;
    }

    /// Time spent drafting, verifying and committing so far.
    pub fn spec_times(&self) -> SpecTimes {
        self.times
    }

    /// The wrapped instance (tokenizer, config, profiling).
    pub fn instance(&self) -> &Qwen35GgufInstance {
        &self.inst
    }

    /// Slot and absolute position of the token each table decodes next,
    /// appending that position to the table.
    fn next_positions(&mut self, tables: &mut [&mut BlockTable]) -> (Vec<u32>, Vec<u32>) {
        let mut slots = Vec::with_capacity(tables.len());
        let mut positions = Vec::with_capacity(tables.len());
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap_or_else(|e| panic!("{MODEL}: KV pool exhausted mid-decode: {e}"));
            slots.push(self.slot(t));
            // The token's absolute position (the append's offset is within its block).
            positions.push(t.len() - 1);
        }
        (slots, positions)
    }

    /// One batched decode step for `tables`, returning the `[bsz, d_model]`
    /// hidden block.
    fn decode_hidden(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<f32> {
        assert_eq!(tables.len(), inputs.len(), "{MODEL}: tables/inputs length mismatch");
        let adapter_slots: Vec<u32> = tables.iter().map(|t| adapters::slot_of(self.inst.stage_models(), t.kv_namespace())).collect();
        let (slots, positions) = self.next_positions(tables);
        self.inst.set_adapter_rows(&adapter_slots);
        let hidden = self.inst.stack_decode_hidden(inputs, &slots, &positions).unwrap_or_else(|e| panic!("{e}"));
        self.inst.set_adapter_rows(&[]);
        // A batch the scheduler will not speculate over (past the crossover or the
        // verify scratch) is not worth tracking for the drafter either.
        if slots.len() > self.spec_batch_limit.min(self.inst.speculation_rows() as usize) {
            self.inst.mtp_forget(&slots);
        } else {
            self.inst.mtp_record_step(&slots, inputs, &hidden);
        }
        hidden
    }
}

impl Qwen35GgufEngine {
    /// One decode step's candidates: `cands == 0` the greedy pick `(token, 0.0)`, else that many `(token, logit)`.
    fn decode_candidates(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], cands: usize) -> Vec<Vec<(u32, f32)>> {
        if cands == 0 {
            self.forward_batched_greedy(tables, inputs).into_iter().map(|t| vec![(t, 0.0)]).collect()
        } else {
            self.forward_batched_topk(tables, inputs, cands)
        }
    }

    /// A sequence's first slice: reserve its slot, look the prompt up in the prefix
    /// cache and seed from it (the seeded recurrent state is left in the working
    /// copy). `Some(hidden)` when the whole prompt was cached, so nothing needs
    /// computing and the sequence is already complete; otherwise its progress is
    /// recorded for [`Self::forward_slice`].
    fn begin_prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Option<Vec<f32>> {
        assert!(prompt.len() <= self.inst.cap as usize, "{MODEL}: prompt of {} tokens exceeds the per-sequence capacity of {}", prompt.len(), self.inst.cap);
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.inst.cfg.vocab) {
            panic!("{MODEL}: prompt token {bad} is outside the model vocabulary ({})", self.inst.cfg.vocab);
        }
        let cacheable = prompt.len() >= MIN_CACHED_TOKENS && self.cache.enabled();
        let hit = if cacheable { self.cache.lookup(table.kv_namespace(), prompt) } else { None };
        // The paged layout resumes by sharing the entry's blocks; the one-block layout copies the rows into
        // the sequence's own window below.
        if let (Some(i), KvLayout::Paged { block_rows, .. }) = (hit, self.inst.kv_layout) {
            self.adopt_cached_blocks(table, i, block_rows);
        }
        let rest = prompt.len() as u32 - table.len();
        table.reserve(rest, &mut self.alloc).unwrap_or_else(|e| panic!("{MODEL}: KV pool exhausted: {e}"));
        let slot = self.assign_state_row(table);
        self.inst.set_blocks(slot, table.blocks());
        let mut start = 0;
        // The cached prefix's last hidden state: the MTP head's input at the
        // position the new tokens continue from.
        let mut prefix_hidden: Option<Vec<f32>> = None;
        // A cold prompt has no seeded head rows to be wrong; a resumed one has them
        // only if the entry it resumes from had its head rows filled.
        let mut seed_valid = true;
        if let Some(i) = hit {
            let e = self.cache.entry(i);
            let (rows, snap) = (e.tokens.len(), e.snap_row);
            seed_valid = e.payload.mtp_valid;
            table.mark_cached(rows as u32);
            match self.inst.kv_layout {
                KvLayout::OneBlock => {
                    let (from, to) = (self.window(&e.table), self.window(table));
                    self.inst.seed_from_prefix(from, to, rows as u32, snap);
                    if seed_valid {
                        self.inst.mtp_seed_prefix(from, to, rows as u32);
                    }
                }
                KvLayout::Paged { .. } => self.inst.seed_state_from_snapshot(snap),
            }
            if rows == prompt.len() {
                // The whole prompt is cached: its state in place, no compute.
                let hidden = e.payload.last_hidden.clone();
                self.inst.store_prefilled(slot);
                if seed_valid {
                    self.inst.mtp_after_prefill(slot, prompt, rows, Some(&hidden), &[]);
                } else {
                    self.inst.mtp_skip_prefill(slot);
                }
                return Some(hidden);
            }
            prefix_hidden = Some(e.payload.last_hidden.clone());
            start = rows;
        }
        // A branch point needs the entry to share the sequence's blocks, which only the paged layout does.
        let paged = matches!(self.inst.kv_layout, KvLayout::Paged { .. });
        let branch = (cacheable && paged)
            .then(|| self.cache.branch_point(table.kv_namespace(), prompt, CHECKPOINT_ALIGN, MIN_CACHED_TOKENS))
            .flatten()
            .into_iter()
            .chain(self.take_hints(table.kv_namespace(), prompt))
            .filter(|&b| b > start && b < checkpoint_len(prompt.len()))
            .min();
        self.partial.insert(slot, PartialPrefill { start, done: start, fresh: true, cacheable, prefix_hidden, rows_hidden: Vec::new(), mtp_feed: self.spec_prefill && seed_valid, branch });
        None
    }

    /// Make `table` (fresh) hold the first `rows` rows of cache entry `i` by sharing its blocks instead of copying
    /// them. The entry keeps serving later prompts from the same blocks, so what this sequence may write must not
    /// be one: the block it continues in is copied first when the entry's rows end inside it, and also when they
    /// end on its boundary while the MTP head is loaded (the head's KV row `rows - 1` is written once the next
    /// token is known, and that is the last row of the last shared block).
    fn adopt_cached_blocks(&mut self, table: &mut BlockTable, i: usize, block_rows: u32) {
        let e = self.cache.entry(i);
        let rows = e.tokens.len() as u32;
        let shared = &e.table.blocks()[..rows.div_ceil(block_rows) as usize];
        table.adopt_rows(shared, rows, &mut self.alloc);
        if !rows.is_multiple_of(block_rows) || self.inst.has_mtp() {
            let live = if rows.is_multiple_of(block_rows) { block_rows } else { rows % block_rows };
            let (from, to) = table.unshare_tail(&mut self.alloc).unwrap_or_else(|| panic!("{MODEL}: a block shared with the prefix cache has a single owner"));
            self.inst.copy_block_rows(from, to, live);
        }
    }

    /// One sequence's slice on its own, from the working copy of the recurrent
    /// state (or the slot's pool row when `resume`): the rows `p.done..end` of
    /// `prompt`. Returns the last row's hidden state.
    fn forward_slice(&mut self, table: &BlockTable, prompt: &[u32], p: &mut PartialPrefill, end: usize, resume: bool) -> Vec<f32> {
        let slot = self.slot(table);
        let collect = (self.inst.has_mtp() && p.mtp_feed).then_some(&mut p.rows_hidden);
        self.inst.prefill_adapter_slot.set(adapters::slot_of(self.inst.stage_models(), table.kv_namespace()));
        let hidden = self.inst.prefill_hidden(prompt, p.done, end, slot, resume, collect).unwrap_or_else(|e| panic!("{e}"));
        self.inst.prefill_adapter_slot.set(LORA_SLOT_NONE);
        hidden
    }

    /// What follows a slice that took the sequence to `end`: record the progress, or
    /// -- when the prompt is in - hand the MTP head its hidden states and file the
    /// prompt for reuse. `state_in_pool`: the recurrent state is in the slot's pool
    /// row and not in the working copy (a multi-sequence forward leaves it there).
    #[allow(clippy::too_many_arguments)]
    fn finish_slice(&mut self, slot: u32, table: &BlockTable, prompt: &[u32], mut p: PartialPrefill, end: usize, hidden: Vec<f32>, exit: SliceExit) -> PrefillProgress {
        let SliceExit { state_in_pool, defer_feed } = exit;
        let consumed = end - p.done;
        p.done = end;
        p.fresh = false;
        if end < prompt.len() {
            // The checkpoint is cached in place of the whole prompt. Its MTP head rows are filled once the whole
            // prompt is in (below), so it is filed without them and marked once they are.
            if p.cacheable && end == checkpoint_len(prompt.len()) {
                self.file_prefix(slot, table, &prompt[..end], &hidden, state_in_pool, false);
            } else if p.branch == Some(end) {
                self.file_prefix(slot, table, &prompt[..end], &hidden, state_in_pool, true);
            }
            self.partial.insert(slot, p);
            return PrefillProgress::Partial { consumed };
        }
        let head_rows_filled = if p.mtp_feed {
            let started = std::time::Instant::now();
            match self.inst.mtp_begin_feed(slot, prompt, p.start, p.prefix_hidden.as_deref(), &p.rows_hidden) {
                FeedStart::Pending(feed) if defer_feed => {
                    self.feeds.push(DeferredFeed { feed, namespace: table.kv_namespace(), prompt: prompt.to_vec(), secs: started.elapsed().as_secs_f64() });
                    // The cache entries are marked once the feed has run.
                    return PrefillProgress::Done(hidden);
                }
                FeedStart::Pending(mut feed) => {
                    let filled = loop {
                        match self.inst.mtp_advance_feed(&mut feed, usize::MAX) {
                            FeedStep::More => {}
                            step => break step == FeedStep::Done,
                        }
                    };
                    self.observe_feed_secs(started.elapsed().as_secs_f64());
                    filled
                }
                FeedStart::Complete => {
                    self.observe_feed_secs(started.elapsed().as_secs_f64());
                    true
                }
                FeedStart::Unavailable => !self.inst.has_mtp(),
            }
        } else {
            self.inst.mtp_skip_prefill(slot);
            false
        };
        self.mark_head_rows(slot, table.kv_namespace(), prompt, head_rows_filled);
        PrefillProgress::Done(hidden)
    }

    /// Fold the seconds a prompt's feed to the head took into [`PagedDecoder::spec_prefill_cost`]'s average.
    fn observe_feed_secs(&mut self, secs: f64) {
        if self.inst.has_mtp() {
            self.feed_secs = Some(self.feed_secs.map_or(secs, |m| m + FEED_EWMA * (secs - m)));
        }
    }

    /// What is cached is the checkpoint and any branch point, filed when the prefill passed them (see
    /// `slice_end`): a prompt that resumed from a checkpoint of its own length or later leaves that entry as it
    /// is. The feed filled the head's rows of every position before the prompt's end, theirs among them, so a
    /// prompt resuming from either drafts with the head (`valid`).
    fn mark_head_rows(&mut self, slot: u32, namespace: KvNamespace, prompt: &[u32], valid: bool) {
        for pending in self.pending.get_mut(&slot).into_iter().flatten() {
            pending.mtp_valid = valid;
        }
        for len in self.filed.remove(&slot).unwrap_or_default() {
            if let Some(resume) = self.cache.payload_mut(namespace, &prompt[..len]) {
                resume.mtp_valid = valid;
            }
        }
    }

    /// Run up to `rows` pairs of the oldest unfinished prompt feed through the MTP head. A finished feed marks the
    /// cache entries filed from its prompt as valid for the head.
    fn advance_feeds(&mut self, rows: usize) {
        let Some(df) = self.feeds.first_mut() else { return };
        let started = std::time::Instant::now();
        let step = self.inst.mtp_advance_feed(&mut df.feed, rows);
        df.secs += started.elapsed().as_secs_f64();
        if step == FeedStep::More {
            return;
        }
        let DeferredFeed { feed, namespace, prompt, secs } = self.feeds.remove(0);
        if step == FeedStep::Done {
            self.observe_feed_secs(secs);
        }
        self.mark_head_rows(feed.slot(), namespace, &prompt, step == FeedStep::Done);
    }

    /// Where a slice of at most `budget` tokens ends: a slice that would run past the prompt's branch point or its
    /// checkpoint stops there, so the state at it can be filed (see [`finish_slice`](Self::finish_slice)).
    fn slice_end(&self, p: &PartialPrefill, prompt_len: usize, budget: usize) -> usize {
        let end = p.done + budget.min(prompt_len - p.done);
        let checkpoint = p.cacheable.then(|| checkpoint_len(prompt_len));
        [p.branch, checkpoint].into_iter().flatten().filter(|&stop| p.done < stop && end > stop).min().unwrap_or(end)
    }

    /// File `tokens` (a prompt's checkpoint, or its `branch` point) of the sequence prefilling into `table` for
    /// reuse: take a snapshot row and snapshot the working recurrent state into it. With paged KV the entry shares
    /// the sequence's leading blocks, so it is cached at once and a prompt arriving while this sequence still runs
    /// resumes from it (its MTP head rows are marked valid once the whole prompt is in); the one-block layout keeps
    /// the entry pending until the sequence ends and its block can be handed over. `hidden` is the last of
    /// `tokens`' rows. `state_in_pool`: the state is in the slot's pool row and not in the working copy.
    fn file_prefix(&mut self, slot: u32, table: &BlockTable, tokens: &[u32], hidden: &[f32], state_in_pool: bool, branch: bool) {
        let (row, evicted) = self.cache.take_row(table.kv_namespace(), tokens);
        self.release_retired(evicted);
        let Some(row) = row else { return };
        if state_in_pool {
            self.inst.load_work_state(slot);
        }
        self.inst.snapshot_prefill_state(row);
        let pending = Pending { branch, tokens: tokens.to_vec(), snap_row: row, last_hidden: hidden.to_vec(), mtp_valid: false };
        if matches!(self.inst.kv_layout, KvLayout::Paged { .. }) {
            let mut shared = table.fork(&mut self.alloc);
            shared.truncate(tokens.len() as u32, &mut self.alloc);
            self.file_entry(pending, shared);
            self.filed.entry(slot).or_default().push(tokens.len());
        } else {
            self.pending.entry(slot).or_default().push(pending);
        }
    }

    /// See [`PagedDecoder::hint_shared_prefix`]: the paged layout files `prompt[..at]`, rounded down to a whole
    /// recurrent chunk, as a branch point of the prefill into `table` - at once if that prefill has started and not
    /// passed it, else when it starts (the prompt is matched by its tokens, as the table has no state yet).
    fn hint_branch(&mut self, table: &BlockTable, prompt: &[u32], at: usize) -> bool {
        let at = at.min(prompt.len()) / CHECKPOINT_ALIGN * CHECKPOINT_ALIGN;
        if !matches!(self.inst.kv_layout, KvLayout::Paged { .. }) || at < MIN_CACHED_TOKENS || !self.cache.enabled() {
            return false;
        }
        let Some(slot) = table.state_row() else {
            if self.hints.len() >= MAX_PREFIX_HINTS {
                self.hints.remove(0);
            }
            self.hints.push((table.kv_namespace(), prompt[..at].to_vec()));
            return true;
        };
        match self.partial.get_mut(&slot) {
            Some(p) if p.done < at => {
                if at < checkpoint_len(prompt.len()) {
                    p.branch = Some(p.branch.map_or(at, |b| b.min(at)));
                }
                true
            }
            _ => false,
        }
    }

    /// The lengths of the hinted prefixes ([`Self::hint_branch`]) `prompt` begins with, removing them.
    fn take_hints(&mut self, namespace: KvNamespace, prompt: &[u32]) -> Vec<usize> {
        let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.hints).into_iter().partition(|(ns, tokens)| *ns == namespace && prompt.starts_with(tokens));
        self.hints = rest;
        mine.into_iter().map(|(_, tokens)| tokens.len()).collect()
    }

    /// Hand a pending record to the cache as an entry over `table`.
    fn file_entry(&mut self, p: Pending, table: BlockTable) {
        let entry = CacheEntry::new(p.tokens, table, p.snap_row, Resume { last_hidden: p.last_hidden, mtp_valid: p.mtp_valid });
        let retired = self.cache.insert(if p.branch { entry.branch() } else { entry });
        self.release_retired(retired);
    }

    fn prefill_slice(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        assert!(!prompt.is_empty(), "{MODEL}: empty prompt");
        if table.is_empty() {
            if let Some(hidden) = self.begin_prefill(table, prompt) {
                return PrefillProgress::Done(hidden);
            }
        }
        let slot = table.state_row().expect("a prefill in progress has a state row");
        let mut p = self.partial.remove(&slot).expect("a prefill in progress has its state");
        let end = self.slice_end(&p, prompt.len(), budget);
        let resume = !p.fresh && p.done > 0;
        let hidden = self.forward_slice(table, prompt, &mut p, end, resume);
        self.finish_slice(slot, table, prompt, p, end, hidden, SliceExit { state_in_pool: false, defer_feed: false })
    }

    /// The next slices of several sequences, the rows of all of them through the
    /// layers as one forward when that pays; one sequence after another otherwise
    /// (a single job, an adapter in play, a round too short to be a prefill). The
    /// recurrent state of every sequence of a multi-sequence forward lives in its
    /// pool row, so each is written there before the forward.
    fn prefill_slices(&mut self, jobs: &mut [PrefillJob<'_>]) -> Vec<PrefillProgress> {
        let adapter_in_play = jobs.iter().any(|j| adapters::slot_of(self.inst.stage_models(), j.table.kv_namespace()) != LORA_SLOT_NONE);
        if jobs.len() < 2 || adapter_in_play || !self.inst.multi_prefill_available() {
            return jobs.iter_mut().map(|j| self.prefill_slice(j.table, j.prompt, j.budget)).collect();
        }
        let (mut answers, live) = self.begin_slices(jobs);
        let rows: usize = live.iter().map(|(_, _, p, end)| end - p.done).sum();
        self.run_slices(jobs, &mut answers, live, None, rows > gpu_core::select::DECODE_REGIME_MAX_ROWS as usize);
        answers.into_iter().map(|a| a.expect("every job was answered")).collect()
    }

    /// The first half of a multi-sequence prefill: a job that starts reserves its slot, seeds from the prefix cache and
    /// has its starting state written to its pool row (the next sequence's seeding overwrites the working copy). The
    /// answers of the jobs already settled (a prompt that was wholly cached), and the jobs with rows to run as
    /// `(job index, slot, progress, end)`.
    fn begin_slices(&mut self, jobs: &mut [PrefillJob<'_>]) -> (Vec<Option<PrefillProgress>>, Vec<LiveSlice>) {
        let mut answers: Vec<Option<PrefillProgress>> = Vec::with_capacity(jobs.len());
        for j in jobs.iter_mut() {
            assert!(!j.prompt.is_empty(), "{MODEL}: empty prompt");
            if j.table.is_empty() {
                let cached = self.begin_prefill(j.table, j.prompt);
                if let Some(hidden) = cached {
                    answers.push(Some(PrefillProgress::Done(hidden)));
                    continue;
                }
                // The state this sequence starts from goes to its pool row now:
                // the next sequence's seeding overwrites the working copy.
                let slot = self.slot(j.table);
                if self.partial[&slot].start == 0 {
                    self.inst.reset_work_state();
                }
                self.inst.store_prefilled(slot);
            }
            answers.push(None);
        }
        let mut live = Vec::new();
        for (i, j) in jobs.iter().enumerate().filter(|(i, _)| answers[*i].is_none()) {
            let slot = self.slot(j.table);
            let p = self.partial.remove(&slot).expect("a prefill in progress has its state");
            let end = self.slice_end(&p, j.prompt.len(), j.budget);
            live.push((i, slot, p, end));
        }
        (answers, live)
    }

    /// The second half: run the rows of `live` and settle each job's answer. `multi` runs them as one forward (with
    /// `decode`, the next token of running sequences as `(tokens, slots, positions)`, when the forward carries any;
    /// returned as their hidden block); otherwise one sequence's slice after another.
    fn run_slices(&mut self, jobs: &mut [PrefillJob<'_>], answers: &mut [Option<PrefillProgress>], mut live: Vec<LiveSlice>, decode: Option<(&[u32], &[u32], &[u32])>, multi: bool) -> Vec<f32> {
        let mut hidden: Vec<(Vec<f32>, bool)> = Vec::with_capacity(live.len());
        let mut decoded = Vec::new();
        if live.len() >= 2 && multi || decode.is_some() && !live.is_empty() {
            let has_mtp = self.inst.has_mtp() && live.iter().any(|(_, _, p, _)| p.mtp_feed);
            let jobs_view: Vec<MultiPrefillJob<'_>> = live.iter().map(|(i, slot, p, end)| MultiPrefillJob { prompt: &*jobs[*i].prompt, start: p.done, end: *end, slot: *slot }).collect();
            let out = match decode {
                Some(decode) => self.inst.mixed_hidden(decode, &jobs_view, has_mtp).map(|(rows, out)| {
                    decoded = rows;
                    out
                }),
                None => self.inst.prefill_hidden_multi(&jobs_view, has_mtp),
            }
            .unwrap_or_else(|e| panic!("{e}"));
            for ((_, _, p, _), r) in live.iter_mut().zip(out) {
                if p.mtp_feed {
                    p.rows_hidden.extend_from_slice(&r.all);
                }
                hidden.push((r.last, true));
            }
        } else {
            for (i, _, p, end) in live.iter_mut() {
                let resume = p.done > 0;
                hidden.push((self.forward_slice(&*jobs[*i].table, jobs[*i].prompt, p, *end, resume), false));
            }
        }
        for ((i, slot, p, end), (h, in_pool)) in live.into_iter().zip(hidden) {
            answers[i] = Some(self.finish_slice(slot, jobs[i].table, jobs[i].prompt, p, end, h, SliceExit { state_in_pool: in_pool, defer_feed: false }));
        }
        decoded
    }
}

impl Qwen35GgufEngine {
    /// Make `job` the prompt read in in the background: its first slice reserves
    /// the slot and seeds from the prefix cache as [`Self::prefill_slice`] does,
    /// then its rows go to a [`PrefillTask`]. `Some` when there is nothing to
    /// read (the whole prompt was cached).
    fn start_background(&mut self, job: &mut PrefillJob<'_>) -> Option<PrefillProgress> {
        assert!(!job.prompt.is_empty(), "{MODEL}: empty prompt");
        if job.table.is_empty() {
            if let Some(hidden) = self.begin_prefill(job.table, job.prompt) {
                return Some(PrefillProgress::Done(hidden));
            }
        }
        let slot = self.slot(job.table);
        let p = self.partial.remove(&slot).expect("a prefill in progress has its state");
        let end = self.slice_end(&p, job.prompt.len(), BACKGROUND_SLICE_TOKENS);
        // The working copy holds the state after `p.done` tokens: cleared for a cold prompt, seeded from a cached
        // prefix, or - for a later slice - restored from the slot's pool row, where the previous slice left it
        // (another prompt's slice may have used the working copy since).
        if !p.fresh {
            self.inst.load_work_state(slot);
        }
        let from = if p.done == 0 { TaskStart::Fresh } else { TaskStart::Continue };
        let collect = self.inst.has_mtp() && p.mtp_feed;
        let task = self.inst.prefill_task(job.prompt, p.done, end, slot, from, collect);
        self.background = Some(Background { task, p, end });
        None
    }

    /// Prompt tokens `job` still has to read in: what is left of one begun, the prompt less its cached prefix
    /// for one not yet begun.
    fn remaining_tokens(&self, job: &PrefillJob<'_>) -> usize {
        match job.table.state_row().and_then(|slot| self.partial.get(&slot)) {
            Some(p) => job.prompt.len() - p.done,
            None => {
                let namespace = job.table.kv_namespace();
                let cached = if job.prompt.len() >= MIN_CACHED_TOKENS { self.cache.peek(namespace, job.prompt).map_or(0, |i| self.cache.entry(i).tokens.len()) } else { 0 };
                job.prompt.len() - cached
            }
        }
    }

    /// Which prompt to read in next: the oldest that fits one slice, so a conversation's next turn does not wait
    /// for another prompt's whole prefill, and otherwise the oldest; after [`MAX_SHORT_STREAK`] short ones in a
    /// row, the oldest long one gets a slice. `None` when every job has its answer.
    fn next_prompt(&mut self, jobs: &[PrefillJob<'_>], answers: &[Option<PrefillProgress>]) -> Option<usize> {
        let open: Vec<usize> = (0..jobs.len()).filter(|&i| answers[i].is_none()).collect();
        let remaining: Vec<usize> = open.iter().map(|&i| self.remaining_tokens(&jobs[i])).collect();
        choose_next(&remaining, &mut self.short_streak).map(|k| open[k])
    }

    /// One paced slice of the prompt being read in ([`PrefillPace`]): record the layers the decode iteration just
    /// measured leaves room for, and wait for the cards to finish them, so the decode step that follows runs on
    /// idle cards. `Some` once the prompt's slice is in.
    fn pump_paced(&mut self) -> Option<PrefillDone> {
        let _slice = gpu_core::markers::range(c"prefill.paced_slice");
        let started = std::time::Instant::now();
        self.pace.observe_iteration(self.slice_ended.map(|ended| (started - ended).as_secs_f64()));
        let bg = self.background.as_mut().expect("a prompt is being read in");
        let rows = bg.task.upcoming_rows();
        let layers = self.pace.layers(rows);
        let inst = &self.inst;
        let (done, spent_all) = gpu_core::lane::normal(|| {
            let done = inst.pump_prefill(&mut bg.task, Some(layers)).unwrap_or_else(|e| panic!("{e}"));
            inst.settle_prefill();
            (done, bg.task.is_recording())
        });
        let now = std::time::Instant::now();
        // A slice that finished its round recorded fewer layers than asked: its time says nothing about a layer's cost.
        if spent_all {
            self.pace.observe_slice(layers, rows, (now - started).as_secs_f64());
        }
        self.slice_ended = Some(now);
        done
    }

    /// See [`PagedDecoder::prefill_background`]: one prompt at a time, oldest
    /// first, each slice handing over to the next (or to the next prompt) as
    /// soon as it is in, without waiting for a card unless `wait`.
    fn read_prompts_in(&mut self, jobs: &mut [PrefillJob<'_>], wait: bool) -> Vec<PrefillProgress> {
        let mut answers: Vec<Option<PrefillProgress>> = (0..jobs.len()).map(|_| None).collect();
        let mut consumed = vec![0usize; jobs.len()];
        // Nothing decodes while the prompt is waited for, so the head's feeds finish; between decode steps a few
        // rounds of the oldest go in with each slice.
        self.advance_feeds(if wait { usize::MAX } else { FEED_ROWS_PER_SLICE });
        loop {
            if self.background.is_none() {
                let Some(i) = self.next_prompt(jobs, &answers) else { break };
                if let Some(done) = self.start_background(&mut jobs[i]) {
                    answers[i] = Some(done);
                    continue;
                }
            }
            let bg = self.background.as_mut().expect("a prompt is being read in");
            let slot = bg.task.slot();
            let i = jobs.iter().position(|j| j.table.state_row() == Some(slot)).unwrap_or_else(|| panic!("{MODEL}: the prompt read in for slot {slot} is not among the scheduler's prefilling sequences"));
            let finished = if wait {
                self.slice_ended = None;
                self.inst.pump_prefill(&mut bg.task, None).unwrap_or_else(|e| panic!("{e}"))
            } else {
                self.pump_paced()
            };
            let Some(done) = finished else { break };
            let Background { mut p, end, .. } = self.background.take().expect("checked above");
            p.rows_hidden.extend_from_slice(&done.all);
            match self.finish_slice(slot, jobs[i].table, jobs[i].prompt, p, end, done.last, SliceExit { state_in_pool: false, defer_feed: !wait }) {
                // Back to the scheduler after a slice: a prompt that arrived meanwhile is considered now, and
                // the decode step runs before the next slice. A finished prompt starts decoding.
                PrefillProgress::Partial { consumed: c } => {
                    consumed[i] += c;
                    break;
                }
                finished => {
                    answers[i] = Some(finished);
                    break;
                }
            }
        }
        // The next slice may follow after a pause with no prompt in: that gap is not a decode iteration.
        if self.background.is_none() {
            self.slice_ended = None;
        }
        answers.into_iter().zip(consumed).map(|(a, c)| a.unwrap_or(PrefillProgress::Partial { consumed: c })).collect()
    }
}

impl BankedEngine for Qwen35GgufEngine {
    fn load_adapter(&mut self, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
        Qwen35GgufEngine::load_adapter(self, adapter)
    }
    fn unload_adapter(&mut self, id: AdapterId) -> Result<UnloadStatus, AdapterError> {
        Qwen35GgufEngine::unload_adapter(self, id)
    }
}

impl PagedDecoder for Qwen35GgufEngine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        self.inst.prefill_round_rows()
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.inst.cap as usize
    }
    fn vocab(&self) -> usize {
        self.inst.cfg.vocab as usize
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.alloc.block_size())
    }
    /// The blocks [`Self::adopt_cached_blocks`] shares from the entry `prompt` would resume from, less the one it
    /// copies before the sequence writes into it. The one-block layout copies rows into the sequence's own block,
    /// so it adopts none.
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let KvLayout::Paged { block_rows, .. } = self.inst.kv_layout else { return 0 };
        if prompt.len() < MIN_CACHED_TOKENS || !self.cache.enabled() {
            return 0;
        }
        let namespace = adapter.map_or(BASE_KV_NAMESPACE, |a| a.kv_namespace());
        let Some(i) = self.cache.peek(namespace, prompt) else { return 0 };
        let rows = self.cache.entry(i).tokens.len() as u32;
        let copied = !rows.is_multiple_of(block_rows) || self.inst.has_mtp();
        rows.div_ceil(block_rows) - u32::from(copied)
    }
    /// Evict least recently used cached prompts until `want` blocks are freed.
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        let before = self.alloc.free_blocks();
        while self.alloc.free_blocks() - before < want {
            let Some(table) = self.cache.evict_lru() else { break };
            self.release_retired([table]);
        }
        self.alloc.free_blocks() - before
    }
    fn reclaim_prefix_keeping(&mut self, want: u32, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let namespace = adapter.map_or(BASE_KV_NAMESPACE, |a| a.kv_namespace());
        let before = self.alloc.free_blocks();
        while self.alloc.free_blocks() - before < want {
            let Some(table) = self.cache.evict_lru_keeping(namespace, prompt) else { break };
            self.release_retired([table]);
        }
        self.alloc.free_blocks() - before
    }
    fn hint_shared_prefix(&mut self, table: &mut BlockTable, prompt: &[u32], at: usize) -> bool {
        self.hint_branch(table, prompt, at)
    }
    /// A sequence that prefilled a cacheable prompt hands the blocks of its
    /// prompt to the prefix cache instead of freeing them (the rows decoding
    /// appended past the prompt are dropped); any other sequence frees all its
    /// blocks. Its recurrent-state row goes back to the pool either way.
    fn release_table(&mut self, t: &mut BlockTable) {
        let row = t.take_state_row();
        if let Some(slot) = row {
            if let Some(bg) = self.background.take_if(|b| b.task.slot() == slot) {
                gpu_core::lane::normal(|| self.inst.abandon_prefill(bg.task));
            }
            self.inst.mtp_release(slot);
            self.feeds.retain(|f| f.feed.slot() != slot);
            self.partial.remove(&slot);
            self.filed.remove(&slot);
        }
        let mut pending = row.and_then(|slot| self.pending.remove(&slot)).unwrap_or_default();
        // The longest entry takes the sequence's own table; a shorter one (a branch point) shares its leading blocks.
        pending.sort_by_key(|p| p.tokens.len());
        match pending.pop() {
            Some(last) => {
                for p in pending {
                    let mut table = t.fork(&mut self.alloc);
                    table.truncate(p.tokens.len() as u32, &mut self.alloc);
                    self.file_entry(p, table);
                }
                t.truncate(last.tokens.len() as u32, &mut self.alloc);
                let table = std::mem::replace(t, BlockTable::new());
                self.file_entry(last, table);
            }
            None => t.release(&mut self.alloc),
        }
        if let (Some(slot), KvLayout::Paged { .. }) = (row, self.inst.kv_layout) {
            self.free_rows.push(slot);
        }
    }
    /// Prefill into the sequence's own slot (other slots untouched), returning
    /// the last prompt token's `[d_model]` hidden row. A prompt that starts
    /// with a cached one resumes from it: the cached KV rows are copied in, the
    /// recurrent state is restored from the snapshot taken when that prefix was
    /// prefilled, and only the new tokens run.
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(hidden) = self.prefill_chunk(table, prompt, usize::MAX) {
                return hidden;
            }
        }
    }
    // perf-number: measured by the `qwen35_spec_sweep` agent-mix scheduler harness
    /// Slices of this many tokens go in between decode steps: about 0.2 s of
    /// prefill at the measured rate, a few decode steps' worth of stall.
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(self.prefill.slice_tokens)
    }
    /// One slice of a prompt. The first call reserves the sequence's slot, looks
    /// the prompt up in the prefix cache and seeds from it; every call prefills
    /// up to `budget` more tokens from the working GDN state the previous one
    /// left in the slot's pool row; the last one files the prompt for reuse and
    /// hands the MTP head the prompt's hidden states.
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        let started = std::time::Instant::now();
        let progress = self.prefill_slice(table, prompt, budget);
        self.times.prefill += started.elapsed();
        progress
    }
    /// Rows of one multi-sequence prefill forward ([`PrefillPolicy::batch_tokens`]).
    fn prefill_batch_tokens(&self) -> Option<usize> {
        (self.prefill.batch_tokens > 0).then_some(self.prefill.batch_tokens)
    }
    fn prefill_chunks(&mut self, jobs: &mut [PrefillJob<'_>]) -> Vec<PrefillProgress> {
        let started = std::time::Instant::now();
        let progress = self.prefill_slices(jobs);
        self.times.prefill += started.elapsed();
        progress
    }
    /// Prompts are read in on each card's normal stream while decode steps run
    /// on its urgent one, wherever every stage's device can prioritise.
    fn background_prefill(&self) -> bool {
        self.background_on && self.inst.background_prefill_available()
    }
    /// The prompt being read in is dropped and the cards drained: its slot's rows
    /// are released with the sequence, and the next prompt starts from a clean
    /// working copy of the recurrent state.
    fn abandon_prefill(&mut self) {
        if let Some(bg) = self.background.take() {
            self.inst.abandon_prefill(bg.task);
        }
    }
    fn prefill_background(&mut self, jobs: &mut [PrefillJob<'_>], wait: bool) -> Vec<PrefillProgress> {
        let started = std::time::Instant::now();
        let progress = self.read_prompts_in(jobs, wait);
        self.times.prefill += started.elapsed();
        progress
    }
    fn mixed_step(&self) -> Option<MixedLimits> {
        self.mixed.then(|| MixedLimits { max_rows: self.inst.prefill_round_rows() as usize, min_prompt_rows: MIN_MIXED_PROMPT_ROWS })
    }
    fn forward_mixed(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], cands: usize, jobs: &mut [PrefillJob<'_>]) -> MixedOutcome {
        assert_eq!(tables.len(), inputs.len(), "{MODEL}: tables/inputs length mismatch");
        // A forward of only one kind, or one an adapter row takes part in (its correction is routed per row of a
        // dispatch of exactly those rows), is the separate pass.
        let adapter_in_play = tables.iter().map(|t| t.kv_namespace()).chain(jobs.iter().map(|j| j.table.kv_namespace())).any(|ns| adapters::slot_of(self.inst.stage_models(), ns) != LORA_SLOT_NONE);
        if tables.is_empty() || jobs.is_empty() || adapter_in_play || !self.mixed {
            let prefilled = if jobs.is_empty() { Vec::new() } else { self.prefill_chunks(jobs) };
            let decoded = self.decode_candidates(tables, inputs, cands);
            return MixedOutcome { decoded, prefilled };
        }
        let started = std::time::Instant::now();
        let (mut answers, live) = self.begin_slices(jobs);
        if live.is_empty() {
            // Every prompt was served from the cache: nothing to carry.
            let prefilled = answers.into_iter().map(|a| a.expect("every job was answered")).collect();
            return MixedOutcome { decoded: self.decode_candidates(tables, inputs, cands), prefilled };
        }
        let (slots, positions) = self.next_positions(tables);
        let hidden = self.run_slices(jobs, &mut answers, live, Some((inputs, &slots, &positions)), true);
        let rows = inputs.len() as u32;
        let decoded = if cands == 0 {
            self.inst.head_argmax_rows(&hidden, rows).into_iter().map(|t| vec![(t, 0.0)]).collect()
        } else {
            self.inst.head_topk_rows(&hidden, rows, cands.clamp(1, TOPK_CAPACITY) as u32)
        };
        self.times.prefill += started.elapsed();
        MixedOutcome { decoded, prefilled: answers.into_iter().map(|a| a.expect("every job was answered")).collect() }
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        decode_lane(self.background_on, || self.inst.head_logits_rows(hidden, 1))
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        decode_lane(self.background_on, || self.inst.head_argmax_rows(hidden, 1)[0])
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        let k = k.clamp(1, TOPK_CAPACITY) as u32;
        decode_lane(self.background_on, || self.inst.head_topk_rows(hidden, 1, k).pop().expect("one row in, one row out"))
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if tables.is_empty() {
            return Vec::new();
        }
        decode_lane(self.background_on, || {
            let hidden = self.decode_hidden(tables, inputs);
            self.inst.head_argmax_rows(&hidden, inputs.len() as u32)
        })
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=self.decode_window_capacity()).contains(&k), "window {k} exceeds this engine's decode_window_capacity {}", self.decode_window_capacity());
        self.forward_batched_greedy(tables, inputs).into_iter().map(|t| vec![t]).collect()
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        if tables.is_empty() {
            return Vec::new();
        }
        let k = k.clamp(1, TOPK_CAPACITY) as u32;
        decode_lane(self.background_on, || {
            let hidden = self.decode_hidden(tables, inputs);
            self.inst.head_topk_rows(&hidden, inputs.len() as u32, k)
        })
    }
    fn step_latency_target(&self) -> Option<std::time::Duration> {
        Some(self.prefill.step_target)
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.cache.stats()
    }
    fn device_stats(&self) -> Option<DeviceStats> {
        self.inst.shards.last().and_then(|s| s.qwen35.gpu.stats())
    }
    /// Every stage's KV pool and recurrent state, as charged to the placement
    /// ([`super::layer_cost`] minus the weights).
    fn kv_pool_bytes(&self) -> u64 {
        let i = &self.inst;
        let pool_rows = match i.kv_layout {
            KvLayout::OneBlock => 0,
            KvLayout::Paged { pool_rows, .. } => pool_rows,
        };
        let spec = super::SpecCost { scratch_rows: i.spec_rows, mtp_head: false };
        let shape = super::ServingShape { prefix_slots: i.prefix_slots, kv_pool_rows: pool_rows, spec, ..super::ServingShape::new(i.cap, i.tier.clone(), i.kv_tier, i.max_batch) };
        i.cfg.layer_types().into_iter().map(|ty| super::layer_state_bytes(&i.cfg, ty, &shape)).sum()
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.alloc.num_blocks() as u64 * self.alloc.block_size() as u64
    }
    /// One batched step per window position: there is no on-device multi-step
    /// schedule across the stage boundary.
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY
    }
    fn adapter_availability(&mut self, adapter: AdapterId) -> AdapterAvailability {
        let (answer, purged) = adapters::availability(self.inst.stage_models(), adapter);
        self.purge_namespace(purged);
        answer
    }
    fn pin_adapter(&mut self, adapter: AdapterId) {
        adapters::pin(self.inst.stage_models(), adapter);
    }
    fn unpin_adapter(&mut self, adapter: AdapterId) {
        let purged = adapters::unpin(self.inst.stage_models(), adapter);
        self.purge_namespace(purged);
    }

    /// A verify step carries `1 + k` rows per sequence through the kernels'
    /// 8-row-per-sequence limit, and at most as many rows in all as the
    /// recurrent-state scratch holds.
    fn spec_max_draft(&self) -> usize {
        match self.inst.speculation_rows() {
            0 => 0,
            rows => (self.inst.max_verify_rows_per_sequence() as usize - 1).min(rows as usize - 1),
        }
    }
    fn spec_max_batch(&self) -> usize {
        self.spec_batch_limit
    }
    fn spec_row_budget(&self) -> usize {
        self.inst.speculation_rows() as usize
    }
    /// A step's cost here is real device time that depends on the row count (the
    /// int8 projections grow with rows), so the scheduler measures it.
    fn spec_cost_aware(&self) -> bool {
        self.inst.speculation_rows() > 0
    }
    fn spec_prefill_cost(&self) -> Option<f64> {
        self.feed_secs
    }

    fn set_spec_prefill(&mut self, on: bool) {
        self.spec_prefill = on;
    }
    /// Score every sequence's rows in one batched step. The rows land at
    /// consecutive positions of each sequence's own block, so the table just
    /// grows by `rows[i].len()`.
    fn forward_batched_verify(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize) -> Vec<Vec<Vec<(u32, f32)>>> {
        decode_lane(self.background_on, || self.verify_rows(tables, rows, cands))
    }
    fn commit_verified(&mut self, _tables: &mut [&mut BlockTable], keep: &[usize]) {
        decode_lane(self.background_on, || self.commit_rows(keep))
    }
    /// The checkpoint's own MTP head, when it is loaded; otherwise no proposals
    /// (the scheduler's installed drafter, if any, supplies them instead).
    fn draft(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        decode_lane(self.background_on, || self.draft_rows(queries, ks))
    }
}

impl Qwen35GgufEngine {
    /// [`PagedDecoder::forward_batched_verify`]'s work.
    fn verify_rows(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize) -> Vec<Vec<Vec<(u32, f32)>>> {
        assert_eq!(tables.len(), rows.len(), "{MODEL}: tables/rows length mismatch");
        let t0 = std::time::Instant::now();
        let mut slots = Vec::with_capacity(tables.len());
        let mut positions = Vec::with_capacity(tables.len());
        // Every packed row of a sequence runs under that sequence's adapter.
        let mut adapter_rows: Vec<u32> = Vec::new();
        for (t, r) in tables.iter().zip(rows) {
            let slot = adapters::slot_of(self.inst.stage_models(), t.kv_namespace());
            adapter_rows.extend(std::iter::repeat_n(slot, r.len()));
        }
        for (t, r) in tables.iter_mut().zip(rows) {
            assert!(!r.is_empty(), "{MODEL}: a verify row set is never empty");
            let first = t.len();
            for _ in 0..r.len() {
                t.append(&mut self.alloc).unwrap_or_else(|e| panic!("{MODEL}: KV pool exhausted mid-verify: {e}"));
            }
            slots.push(self.slot(t));
            positions.push(first);
        }
        let row_refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        self.inst.set_adapter_rows(&adapter_rows);
        let hidden = self.inst.stack_verify_hidden(&row_refs, &slots, &positions).unwrap_or_else(|e| panic!("{e}"));
        self.inst.set_adapter_rows(&[]);
        let n = rows.iter().map(|r| r.len()).sum::<usize>() as u32;
        let flat: Vec<Vec<(u32, f32)>> = if cands <= 1 {
            self.inst.head_argmax_rows(&hidden, n).into_iter().map(|t| vec![(t, 0.0)]).collect()
        } else {
            self.inst.head_topk_rows(&hidden, n, cands.min(TOPK_CAPACITY) as u32)
        };
        self.verifying = Some(Verifying { slots, rows: rows.to_vec(), hidden });
        self.times.verify += t0.elapsed();
        let mut flat = flat.into_iter();
        rows.iter().map(|r| flat.by_ref().take(r.len()).collect()).collect()
    }
    /// [`PagedDecoder::commit_verified`]'s work.
    fn commit_rows(&mut self, keep: &[usize]) {
        let t0 = std::time::Instant::now();
        let v = self.verifying.take().expect("commit_verified without a verify step");
        let counts: Vec<u32> = v.rows.iter().map(|r| r.len() as u32).collect();
        self.inst.commit_verify(&v.slots, &counts, keep);
        self.inst.mtp_record_verified(&v.slots, &v.rows, keep, &v.hidden);
        self.times.commit += t0.elapsed();
    }
    /// [`PagedDecoder::draft`]'s work.
    fn draft_rows(&mut self, queries: &[DraftQuery], ks: &[usize]) -> Vec<Vec<u32>> {
        if !self.inst.has_mtp() {
            return vec![Vec::new(); queries.len()];
        }
        // No prompt is being read in to carry the feeds on; a feed not finished by now would keep its sequence
        // from drafting.
        self.advance_feeds(FEED_ROWS_PER_SLICE);
        let slots: Vec<u32> = queries.iter().map(|q| self.slot(q.table)).collect();
        let consumed: Vec<u32> = queries.iter().map(|q| q.table.len()).collect();
        let next: Vec<u32> = queries.iter().map(|q| *q.context.last().expect("a running sequence has context")).collect();
        // Proposal `step` (from 2) writes the head's KV row at `consumed + step - 2`, and a paged table holds only the
        // blocks it has: the verify step that follows appends the rows, the draft runs before it. A sequence whose
        // rows end at a block boundary therefore drafts no further than the rows its blocks hold.
        let ks: Vec<usize> = match self.inst.kv_layout {
            KvLayout::OneBlock => ks.to_vec(),
            KvLayout::Paged { block_rows, .. } => queries
                .iter()
                .zip(ks)
                .map(|(q, &k)| k.min((q.table.blocks().len() as u32 * block_rows + 1).saturating_sub(q.table.len()) as usize))
                .collect(),
        };
        let t0 = std::time::Instant::now();
        let out = self.inst.mtp_draft(&slots, &consumed, &next, &ks);
        self.times.draft += t0.elapsed();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A conversation's next turn (a few tokens to read in) starts before a long prompt that arrived first, and the
    /// long prompt still gets a slice after a bounded number of them.
    #[test]
    fn a_short_prompt_goes_before_a_long_one_and_the_long_one_is_not_starved() {
        let mut streak = 0;
        let long = BACKGROUND_SLICE_TOKENS * 4;
        assert_eq!(choose_next(&[long, 100], &mut streak), Some(1));
        assert_eq!(choose_next(&[long, long + 1], &mut 0), Some(0), "the oldest long prompt when none is short");
        assert_eq!(choose_next(&[300, 100], &mut 0), Some(0), "the oldest short prompt");
        assert_eq!(choose_next(&[], &mut 0), None);
        let mut streak = 0;
        let picks: Vec<usize> = (0..=MAX_SHORT_STREAK).map(|_| choose_next(&[long, 100], &mut streak).unwrap()).collect();
        assert_eq!(picks.iter().filter(|&&i| i == 0).count(), 1, "one slice for the long prompt after {MAX_SHORT_STREAK} short ones: {picks:?}");
        assert_eq!(picks[MAX_SHORT_STREAK as usize], 0);
    }
}
