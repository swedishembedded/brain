// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The released-GGUF Qwen3.8-27B as a `model::serve::PagedDecoder`: the engine
//! `model::serve::Scheduler` drives, so this checkpoint is served by the same
//! continuous-batching loop as every other brain LLM rather than a decode loop
//! of its own.
//!
//! What the engine adds over [`Qwen35GgufInstance`] is only the sequence
//! bookkeeping a scheduler needs. The instance already holds `max_batch`
//! decode slots per stage (one GQA KV window of `cap` rows plus one GDN
//! recurrent state per sequence); here a slot is a physical block of a
//! [`BlockAllocator`] whose `block_size == cap`, exactly the layout
//! `crate::serve::Engine` uses, so a sequence's whole lifetime is one block
//! and `table.blocks()[0]` is its slot. A decode iteration then names the
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
use model::paged::{BlockAllocator, BlockTable, KvNamespace};
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder};

use crate::adapter_serving as adapters;

use super::prefix::{Entry, PrefixCache, MIN_CACHED_TOKENS};
use super::{Qwen35GgufInstance, MODEL};

/// Candidates a non-greedy pick can sample from. Real sampling is exact
/// within this set; it only bounds how deep into the vocabulary nucleus
/// sampling can reach (the same bound `crate::serve::Engine` uses).
const TOPK_CAPACITY: usize = 32;

/// [`Qwen35GgufInstance`] behind the paged-decoder seam. See the module doc.
pub struct Qwen35GgufEngine {
    inst: Qwen35GgufInstance,
    alloc: BlockAllocator,
    /// Prompts whose KV rows and recurrent-state snapshot are kept for reuse
    /// (see [`super::prefix`]).
    cache: PrefixCache,
    /// A finished prefill's cache record, held until its sequence ends: the
    /// sequence's block becomes the cache entry's KV only then (decoding
    /// appends rows past the prompt and never rewrites the prompt's own).
    pending: HashMap<u32, Pending>,
}

/// What a sequence's prefill leaves for the cache once the sequence is over.
struct Pending {
    tokens: Vec<u32>,
    snap_row: u32,
    last_hidden: Vec<f32>,
}

impl Qwen35GgufEngine {
    /// Wrap an activated instance. The instance's `max_batch` becomes the
    /// number of sequences that may be resident at once and its `cap` the
    /// per-sequence `prompt + max_new` limit.
    ///
    /// The allocator holds `max_batch + prefix_slots` blocks: the scheduler
    /// never runs more than `max_batch` sequences, so a cached prompt's retained
    /// block never starves an admission.
    pub fn new(inst: Qwen35GgufInstance) -> Qwen35GgufEngine {
        let alloc = BlockAllocator::new(inst.max_batch + inst.prefix_slots, inst.cap);
        let cache = PrefixCache::new(inst.prefix_slots);
        Qwen35GgufEngine { inst, alloc, cache, pending: HashMap::new() }
    }

    /// Free the blocks of entries the cache retired.
    fn release_retired(&mut self, retired: impl IntoIterator<Item = BlockTable>) {
        for mut table in retired {
            table.release(&mut self.alloc);
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

    /// Drop the cached prompts of an adapter that is gone.
    fn purge_namespace(&mut self, ns: impl IntoIterator<Item = KvNamespace>) {
        for ns in ns {
            let tables = self.cache.evict_namespace(ns);
            self.release_retired(tables);
        }
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
            let (_block, offset) = t.append(&mut self.alloc).unwrap_or_else(|e| panic!("{MODEL}: KV pool exhausted mid-decode: {e}"));
            slots.push(t.blocks()[0]);
            positions.push(offset);
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
        hidden
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
        tokens.div_ceil(self.inst.cap)
    }
    /// Evict least recently used cached prompts until `want` blocks are freed.
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        let mut freed = 0;
        while freed < want {
            let Some(table) = self.cache.evict_lru() else { break };
            self.release_retired([table]);
            freed += 1;
        }
        freed
    }
    /// A sequence that prefilled a cacheable prompt hands its block to the
    /// prefix cache instead of freeing it; any other sequence frees it.
    fn release_table(&mut self, t: &mut BlockTable) {
        let kept = t.blocks().first().and_then(|slot| self.pending.remove(slot));
        match kept {
            Some(p) => {
                let table = std::mem::replace(t, BlockTable::new());
                let retired = self.cache.insert(Entry::new(p.tokens, table, p.snap_row, p.last_hidden));
                self.release_retired(retired);
            }
            None => t.release(&mut self.alloc),
        }
    }
    /// Prefill into the sequence's own slot (other slots untouched), returning
    /// the last prompt token's `[d_model]` hidden row. A prompt that starts
    /// with a cached one resumes from it: the cached KV rows are copied in, the
    /// recurrent state is restored from the snapshot taken when that prefix was
    /// prefilled, and only the new tokens run.
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "{MODEL}: prefill expects a fresh sequence");
        assert!(!prompt.is_empty(), "{MODEL}: empty prompt");
        assert!(prompt.len() <= self.inst.cap as usize, "{MODEL}: prompt of {} tokens exceeds the per-sequence capacity of {}", prompt.len(), self.inst.cap);
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.inst.cfg.vocab) {
            panic!("{MODEL}: prompt token {bad} is outside the model vocabulary ({})", self.inst.cfg.vocab);
        }
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap_or_else(|e| panic!("{MODEL}: KV pool exhausted: {e}"));
        let slot = table.blocks()[0];
        let cacheable = prompt.len() >= MIN_CACHED_TOKENS && self.cache.enabled();
        let mut start = 0;
        if cacheable {
            if let Some(i) = self.cache.lookup(table.kv_namespace(), prompt) {
                let e = self.cache.entry(i);
                let (from, rows, snap) = (e.table.blocks()[0], e.tokens.len(), e.snap_row);
                self.inst.seed_from_prefix(from, slot, rows as u32, snap);
                if rows == prompt.len() {
                    // The whole prompt is cached: its state in place, no compute.
                    let hidden = e.last_hidden.clone();
                    self.inst.store_prefilled(slot);
                    return hidden;
                }
                start = rows;
            }
        }
        self.inst.prefill_adapter_slot.set(adapters::slot_of(self.inst.stage_models(), table.kv_namespace()));
        let hidden = self.inst.prefill_hidden(prompt, start, slot).unwrap_or_else(|e| panic!("{e}"));
        self.inst.prefill_adapter_slot.set(LORA_SLOT_NONE);
        if cacheable {
            let (row, evicted) = self.cache.take_row();
            self.release_retired(evicted);
            if let Some(row) = row {
                self.inst.snapshot_prefill_state(row);
                self.pending.insert(slot, Pending { tokens: prompt.to_vec(), snap_row: row, last_hidden: hidden.clone() });
            }
        }
        hidden
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.inst.head_logits_rows(hidden, 1)
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        self.inst.head_argmax_rows(hidden, 1)[0]
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        let k = k.clamp(1, TOPK_CAPACITY) as u32;
        self.inst.head_topk_rows(hidden, 1, k).pop().expect("one row in, one row out")
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if tables.is_empty() {
            return Vec::new();
        }
        let hidden = self.decode_hidden(tables, inputs);
        self.inst.head_argmax_rows(&hidden, inputs.len() as u32)
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
        let hidden = self.decode_hidden(tables, inputs);
        self.inst.head_topk_rows(&hidden, inputs.len() as u32, k)
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.cache.stats()
    }
    fn device_stats(&self) -> Option<DeviceStats> {
        self.inst.shards.last().and_then(|s| s.qwen35.gpu.stats())
    }
    /// Every stage's KV windows and recurrent state for all `max_batch` slots,
    /// as charged to the placement ([`super::layer_cost`] minus the weights).
    fn kv_pool_bytes(&self) -> u64 {
        let slots = (self.inst.max_batch + self.inst.prefix_slots) as u64;
        let snapshots = self.inst.prefix_slots as u64;
        self.inst
            .cfg
            .layer_types()
            .into_iter()
            .map(|ty| {
                let state = self.inst.cfg.layer_decode_state_bytes(ty, self.inst.cap, self.inst.kv_tier);
                slots * state + if ty == crate::config::LayerType::Linear { snapshots * state } else { 0 }
            })
            .sum()
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        (self.inst.max_batch + self.inst.prefix_slots) as u64 * self.inst.cap as u64
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
}
