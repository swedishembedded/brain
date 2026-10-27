// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context language
// models at a fixed memory budget for its clients. If your team needs
// expertise in bounded-state inference engines and prefix reuse for hybrid
// recurrent/attention models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The `model::serve::PagedDecoder` for configs whose per-sequence state has a FIXED size: a sliding window
//! (a ring of `window` KV rows per full-attention layer), an anchor-memory bank, and the Gated-DeltaNet state and
//! conv history (EdgeRock). It runs under the same generic `model::serve::Scheduler` and passes the same
//! conformance suite as every other engine; the scheduler owns every policy decision (admission, prefill budget,
//! preemption, sampling), and this module implements the device operations it asks for and nothing else.
//!
//! # Why a second engine and not a mode of [`crate::serve::Engine`]
//!
//! `Engine` is built around KV that GROWS with the sequence: a pool of `max_concurrent * max_seq_len` rows per
//! layer, one block per sequence, prefix reuse that is "no cache" (plain GQA) and a per-sequence recurrent-state map
//! keyed by block id. Every one of those assumptions is false here. A sequence costs the same few hundred
//! megabytes at token 1 and at token 262144; the pool is sized by the slot count alone; and a prefix is resumable
//! only from a copy of the whole state, not from retained rows. The shared parts are already shared: the forward
//! (`Qwen35::slot_prefill` / `slot_decode` over a [`SlotPool`]), the device heads, the scheduler and the snapshot
//! prefix cache (`model::snapshot_prefix`). What is left in this file is the binding between a [`BlockTable`] and a
//! slot, which is the part that differs.
//!
//! # Slots, blocks and the position limit
//!
//! The block allocator has `max_concurrent` blocks of `max_seq_len` tokens each, so a sequence owns exactly one
//! block for its whole life and that block's id IS its slot in the pool. `max_seq_len` is the position limit of the
//! model (`max_position_embeddings`, 262144 for EdgeRock), not a memory figure: the block count is the only memory
//! knob, and the allocator's token arithmetic costs nothing. The pool holds `max_concurrent` live slots followed by
//! `prefix_slots` snapshot slots.
//!
//! # Prefix reuse
//!
//! An entry of the [`SnapshotPrefixCache`] is a snapshot slot, a copy of the whole fixed state (`slot_copy`) taken
//! the moment a prompt's prefill ends, plus the payload ([`Resume`]): the prompt's last hidden row. The entry's
//! table carries only the KV namespace; it holds NO block, because there are no KV rows to retain. A prompt that
//! extends a cached one copies the snapshot over its own slot and prefills only the rest; an exact repeat copies
//! the snapshot and returns the stored hidden row without computing. A prefix is therefore reusable at one length
//! only (the prompt that was filed), which is what a conversation resent turn after turn needs.
//!
//! [`PagedDecoder::reclaim_prefix`] frees nothing and so evicts nothing: the cache's snapshot slots are outside
//! the block pool, so dropping an entry would not let one more sequence in. Capacity pressure inside the cache is
//! handled by its own LRU rule when a prompt is filed.
//!
//! # Memory
//!
//! [`slot_state_bytes`] is the fixed cost of one slot; [`ring_slots`] sizes the live and snapshot slots to a byte
//! budget. Weights are held at the tier the caller names (bf16 by default for EdgeRock, straight from the lazy
//! HF source: no fp32 copy of the model exists on the device). The anchor memory's projections and scorer are
//! plain fp32 parameters ("never a quantized tier", see `Qwen35Config::param_list`); [`memory_param_bytes`] records
//! what they cost.

use gpu_core::{DeviceBuffer, Gpu};
use model::paged::{BlockAllocator, BlockTable, KvNamespace};
use model::serve::{PagedDecoder, PrefillProgress};
use model::snapshot_prefix::{Entry, SnapshotPrefixCache};

use crate::config::Qwen35Config;
use crate::model::{pipelines, slot_state_bytes, Qwen35, SlotPool};

/// Candidate list width of [`RingEngine::forward_batched_topk`], as in [`crate::serve::Engine`].
const TOPK_CAPACITY: usize = 32;

/// Prompt tokens per prefill round. A round's windowed attention scratch is `[chunk, n_heads, window + chunk]`
/// f32, about 100 MB at EdgeRock's shape: bounded by this, never by the context.
pub const PREFILL_ROUND_TOKENS: u32 = 256;

/// The shortest prompt worth a snapshot: copying a slot's whole state for a handful of tokens costs more than
/// prefilling them again.
pub const MIN_CACHED_TOKENS: usize = 16;

/// What resuming from a cached prompt needs beyond the snapshot slot.
pub struct Resume {
    /// The final-norm hidden row of the prompt's last token, so an exact repeat needs no compute.
    pub last_hidden: Vec<f32>,
}

/// Bytes of the anchor memories' plain fp32 parameters (projections, scorer, norms, read gate), over every layer
/// that has one. They are not part of the weight tier: a bf16 or int8 model still pays this in f32.
pub fn memory_param_bytes(cfg: &Qwen35Config) -> u64 {
    let layers = (0..cfg.n_layers as usize).filter(|&l| cfg.layer_has_memory(l)).count() as u64;
    let per_layer: u64 = cfg.memory_leaves().iter().map(|&(_, numel)| numel as u64).sum();
    layers * per_layer * 4
}

/// Most prompts kept for reuse beside the live sequences: each costs a whole slot, which a live sequence would
/// rather have, and an agent's few conversations are what the cache is for.
pub const MAX_PREFIX_SLOTS: u32 = 8;

/// Live and snapshot slots for a slot budget of `budget` bytes: `max_batch` live slots as far as the budget
/// holds them (at least one), then up to [`MAX_PREFIX_SLOTS`] (and no more than the live count) cached prompts
/// from what is left. A budget that cannot hold even one live slot still returns one: the caller's placement
/// refuses the oversized model with its own arithmetic, not this sizing.
pub fn ring_slots(cfg: &Qwen35Config, budget: u64, max_batch: u32) -> (u32, u32) {
    let per_slot = slot_state_bytes(cfg).max(1);
    let fit = (budget / per_slot).min(u64::from(u32::MAX)) as u32;
    let live = fit.min(max_batch).max(1);
    (live, fit.saturating_sub(live).min(live).min(MAX_PREFIX_SLOTS))
}

/// See the module doc.
pub struct RingEngine {
    model: Qwen35,
    pool: SlotPool,
    /// One block per live sequence: the block id is the slot.
    alloc: BlockAllocator,
    max_seq_len: u32,
    live_slots: u32,
    cache: SnapshotPrefixCache<Resume>,
}

impl RingEngine {
    /// Build over `weights` held at `tier`, on a device of its own. `max_seq_len` is capped by the model's position
    /// limit; `max_concurrent` live slots and `prefix_slots` cached prompts (0 turns prefix reuse off).
    pub fn from_source_tier(cfg: Qwen35Config, weights: &dyn checkpoint::TensorSource, max_seq_len: u32, max_concurrent: u32, prefix_slots: u32, tier: &model::ops::TierPolicy) -> RingEngine {
        Self::with_gpu(Gpu::new(pipelines()), cfg, weights, max_seq_len, max_concurrent, prefix_slots, tier)
    }

    /// [`Self::from_source_tier`] on an existing device handle (warm start).
    pub fn from_source_tier_on(parent: &Gpu, cfg: Qwen35Config, weights: &dyn checkpoint::TensorSource, max_seq_len: u32, max_concurrent: u32, prefix_slots: u32, tier: &model::ops::TierPolicy) -> RingEngine {
        Self::with_gpu(parent.new_like(pipelines()), cfg, weights, max_seq_len, max_concurrent, prefix_slots, tier)
    }

    fn with_gpu(gpu: Gpu, cfg: Qwen35Config, weights: &dyn checkpoint::TensorSource, max_seq_len: u32, max_concurrent: u32, prefix_slots: u32, tier: &model::ops::TierPolicy) -> RingEngine {
        assert!(max_seq_len > 0 && max_concurrent > 0, "qwen35::serve_ring: max_seq_len and max_concurrent must be > 0");
        assert!(cfg.swa_window.is_some(), "qwen35::serve_ring: the ring engine serves sliding-window configs; a plain config pages a growing KV cache (qwen35::serve::Engine)");
        assert!(
            max_seq_len <= cfg.max_position_embeddings,
            "qwen35::serve_ring: max_seq_len {max_seq_len} exceeds the model's position limit {}",
            cfg.max_position_embeddings
        );
        let model = Qwen35::new_on_dt(gpu, cfg, 1, 1, weights, tier);
        let pool = model.new_slot_pool(max_concurrent + prefix_slots);
        RingEngine { model, pool, alloc: BlockAllocator::new(max_concurrent, max_seq_len), max_seq_len, live_slots: max_concurrent, cache: SnapshotPrefixCache::new(prefix_slots) }
    }

    /// The model, for callers that compare against its own decoding.
    pub fn model(&self) -> &Qwen35 {
        &self.model
    }

    /// The device handle, for profiling.
    pub fn gpu(&self) -> &Gpu {
        &self.model.gpu
    }

    /// Bring every table to `position` tokens of synthetic context without computing each one: the first is
    /// prefilled with a fixed in-vocabulary prompt and the rest are copies of its state, so a measurement can sit
    /// at a decode position (ring and bank full) at the cost of one prefill. Tables must be fresh.
    pub fn seed_synthetic(&mut self, tables: &mut [BlockTable], position: u32) {
        assert!(position > 0 && position <= self.max_seq_len, "qwen35::serve_ring: synthetic position {position} outside 1..={}", self.max_seq_len);
        let vocab = u64::from(self.model.cfg.vocab);
        let prompt: Vec<u32> = (0..u64::from(position)).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let mut first = None;
        for t in tables.iter_mut() {
            assert!(t.is_empty(), "qwen35::serve_ring: seed_synthetic needs fresh tables");
            t.reserve(position, &mut self.alloc).unwrap_or_else(|e| panic!("qwen35::serve_ring: no free slot: {e}"));
            let slot = Self::slot(t);
            match first {
                None => {
                    self.model.slot_reset(&self.pool, slot);
                    self.model.slot_prefill(&self.pool, slot, &prompt, PREFILL_ROUND_TOKENS);
                    first = Some(slot);
                }
                Some(src) => self.model.slot_copy(&self.pool, src, slot),
            }
        }
    }

    /// Device bytes of the slot pool (live and snapshot slots): fixed for the engine's life.
    pub fn state_bytes(&self) -> u64 {
        self.pool.bytes()
    }

    /// The tier of the weight called `name` (a linear); `None` for a tensor held outside the weight table.
    pub fn weight_tier(&self, name: &str) -> Option<gpu_core::select::Dtype> {
        self.model.weight_tier(name)
    }

    /// Pool slot that holds snapshot row `row` of the prefix cache.
    fn snapshot_slot(&self, row: u32) -> u32 {
        self.live_slots + row
    }

    /// The slot a sequence owns: its one block's id.
    fn slot(table: &BlockTable) -> u32 {
        *table.blocks().first().expect("a prefilling or decoding sequence owns its block")
    }

    /// First slice of a sequence: take its block, clear the slot, and resume from a cached prefix when there is one.
    /// `Some(hidden)` when the whole prompt was cached.
    fn begin(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Option<Vec<f32>> {
        assert!(!prompt.is_empty(), "qwen35::serve_ring: empty prompt (no token to produce a hidden state from)");
        assert!(prompt.len() <= self.max_seq_len as usize, "qwen35::serve_ring: prompt of {} tokens exceeds the position limit of {}", prompt.len(), self.max_seq_len);
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.model.cfg.vocab) {
            panic!("qwen35::serve_ring: prompt token {bad} is outside the model vocabulary ({})", self.model.cfg.vocab);
        }
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap_or_else(|e| panic!("qwen35::serve_ring: no free slot: {e}"));
        let slot = Self::slot(table);
        self.model.slot_reset(&self.pool, slot);
        if prompt.len() < MIN_CACHED_TOKENS {
            return None;
        }
        let hit = self.cache.lookup(table.kv_namespace(), prompt)?;
        let entry = self.cache.entry(hit);
        let rows = entry.tokens.len();
        self.model.slot_copy(&self.pool, self.snapshot_slot(entry.snap_row), slot);
        table.mark_cached(rows as u32);
        (rows == prompt.len()).then(|| entry.payload.last_hidden.clone())
    }

    /// File the state `slot` holds, which is exactly `prompt`'s, as a prefix entry in `namespace`.
    fn file_prefix(&mut self, slot: u32, namespace: KvNamespace, prompt: &[u32], hidden: &[f32]) {
        if prompt.len() < MIN_CACHED_TOKENS || !self.cache.enabled() {
            return;
        }
        // The entries this displaces hold no blocks (see the module doc), so there is nothing to release.
        let (row, _evicted) = self.cache.take_row(namespace, prompt);
        let Some(row) = row else { return };
        self.model.slot_copy(&self.pool, slot, self.snapshot_slot(row));
        let _retired = self.cache.insert(Entry::new(prompt.to_vec(), BlockTable::with_namespace(namespace), row, Resume { last_hidden: hidden.to_vec() }));
    }

    /// One decode step for `(table, input)` pairs, hidden block left on the device.
    fn decode_hidden(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> DeviceBuffer {
        assert_eq!(tables.len(), inputs.len(), "qwen35::serve_ring: tables/inputs length mismatch");
        let mut items = Vec::with_capacity(tables.len());
        for (t, &input) in tables.iter_mut().zip(inputs) {
            t.append(&mut self.alloc).unwrap_or_else(|e| panic!("qwen35::serve_ring: sequence outgrew the position limit: {e}"));
            items.push((Self::slot(t), input));
        }
        self.model.slot_decode_dev(&self.pool, &items)
    }

    /// Per-slot position, for tests: tokens the slot has absorbed.
    pub fn slot_position(&self, table: &BlockTable) -> u32 {
        self.pool.position(Self::slot(table))
    }
}

impl PagedDecoder for RingEngine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        PREFILL_ROUND_TOKENS
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.live_slots as usize)
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len as usize
    }
    fn vocab(&self) -> usize {
        self.model.cfg.vocab as usize
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        // A sequence owns one block however long it is.
        tokens.div_ceil(self.max_seq_len).min(1)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(hidden) = self.prefill_chunk(table, prompt, usize::MAX) {
                return hidden;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(PREFILL_ROUND_TOKENS as usize)
    }
    /// The next slice of `prompt`, at most `budget` tokens, continuing from the slot's own position. The first
    /// call takes the slot and resumes from the prefix cache; the call that ends the prompt files it.
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        if table.is_empty() {
            if let Some(hidden) = self.begin(table, prompt) {
                return PrefillProgress::Done(hidden);
            }
        }
        let slot = Self::slot(table);
        let done = self.pool.position(slot) as usize;
        let end = done + budget.min(prompt.len() - done);
        let hidden = self.model.slot_prefill(&self.pool, slot, &prompt[done..end], PREFILL_ROUND_TOKENS);
        if end < prompt.len() {
            return PrefillProgress::Partial { consumed: end - done };
        }
        self.file_prefix(slot, table.kv_namespace(), prompt, &hidden);
        PrefillProgress::Done(hidden)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let d = self.model.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "qwen35::serve_ring::logits: hidden must be one row of {d} floats, got {}", hidden.len());
        let x = self.model.gpu.storage_init("qwen35.serve_ring.logits.hidden", hidden);
        let logits = self.model.head_logits_dev(&x);
        self.model.gpu.read(&logits, self.model.cfg.vocab as usize)
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        let d = self.model.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "qwen35::serve_ring::admit_greedy: hidden must be one row of {d} floats, got {}", hidden.len());
        let x = self.model.gpu.storage_init("qwen35.serve_ring.admit.hidden", hidden);
        self.model.head_argmax_dev(&x)
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        let d = self.model.cfg.d_model as usize;
        assert_eq!(hidden.len(), d, "qwen35::serve_ring::admit_topk: hidden must be one row of {d} floats, got {}", hidden.len());
        let x = self.model.gpu.storage_init("qwen35.serve_ring.admit.hidden", hidden);
        self.model.head_topk_dev(&x, k.clamp(1, TOPK_CAPACITY) as u32)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if tables.is_empty() {
            return Vec::new();
        }
        let hidden = self.decode_hidden(tables, inputs);
        self.model.head_argmax_rows_dev(&hidden, inputs.len() as u32)
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=self.decode_window_capacity()).contains(&k), "window {k} exceeds this engine's decode_window_capacity {}", self.decode_window_capacity());
        self.forward_batched_greedy(tables, inputs).into_iter().map(|t| vec![t]).collect()
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        if tables.is_empty() {
            return Vec::new();
        }
        let hidden = self.decode_hidden(tables, inputs);
        self.model.head_topk_rows_dev(&hidden, inputs.len() as u32, k.clamp(1, TOPK_CAPACITY) as u32)
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.cache.stats()
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.model.gpu.stats()
    }
    /// The whole pool's fixed state: it does not grow with any sequence.
    fn kv_pool_bytes(&self) -> u64 {
        self.pool.bytes()
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        u64::from(self.alloc.num_blocks()) * u64::from(self.max_seq_len)
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY
    }
}
