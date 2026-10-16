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

use gpu_core::DeviceStats;
use model::paged::{BlockAllocator, BlockTable};
use model::serve::PagedDecoder;

use super::{Qwen35GgufInstance, MODEL};

/// Candidates a non-greedy pick can sample from. Real sampling is exact
/// within this set; it only bounds how deep into the vocabulary nucleus
/// sampling can reach (the same bound `crate::serve::Engine` uses).
const TOPK_CAPACITY: usize = 32;

/// [`Qwen35GgufInstance`] behind the paged-decoder seam. See the module doc.
pub struct Qwen35GgufEngine {
    inst: Qwen35GgufInstance,
    alloc: BlockAllocator,
}

impl Qwen35GgufEngine {
    /// Wrap an activated instance. The instance's `max_batch` becomes the
    /// number of sequences that may be resident at once and its `cap` the
    /// per-sequence `prompt + max_new` limit.
    pub fn new(inst: Qwen35GgufInstance) -> Qwen35GgufEngine {
        let alloc = BlockAllocator::new(inst.max_batch, inst.cap);
        Qwen35GgufEngine { inst, alloc }
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
        let (slots, positions) = self.next_positions(tables);
        self.inst.stack_decode_hidden(inputs, &slots, &positions).unwrap_or_else(|e| panic!("{e}"))
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
    /// No prefix cache: a sequence's slot holds recurrent state that is only
    /// valid for its own whole prefix.
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    /// Prefill into the sequence's own slot (cleared first, other slots
    /// untouched), returning the last prompt token's `[d_model]` hidden row.
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "{MODEL}: prefill expects a fresh sequence");
        assert!(!prompt.is_empty(), "{MODEL}: empty prompt");
        assert!(prompt.len() <= self.inst.cap as usize, "{MODEL}: prompt of {} tokens exceeds the per-sequence capacity of {}", prompt.len(), self.inst.cap);
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.inst.cfg.vocab) {
            panic!("{MODEL}: prompt token {bad} is outside the model vocabulary ({})", self.inst.cfg.vocab);
        }
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap_or_else(|e| panic!("{MODEL}: KV pool exhausted: {e}"));
        let slot = table.blocks()[0];
        self.inst.replay_prompt_hidden(prompt, slot).unwrap_or_else(|e| panic!("{e}"))
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
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<DeviceStats> {
        self.inst.shards.last().and_then(|s| s.qwen35.gpu.stats())
    }
    /// Every stage's KV windows and recurrent state for all `max_batch` slots,
    /// as charged to the placement ([`super::layer_cost`] minus the weights).
    fn kv_pool_bytes(&self) -> u64 {
        let slots = self.inst.max_batch as u64;
        self.inst.cfg.layer_types().into_iter().map(|ty| slots * self.inst.cfg.layer_decode_state_bytes(ty, self.inst.cap, self.inst.kv_tier)).sum()
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.inst.max_batch as u64 * self.inst.cap as u64
    }
    /// One batched step per window position: there is no on-device multi-step
    /// schedule across the stage boundary.
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY
    }
}
