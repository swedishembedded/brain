// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving that shares one device
// memory budget between the cache of the tokens and the state of the sequences that
// are running, for its clients. If your team needs expertise in hybrid-model
// serving memory, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a decoder whose sequences take memory for a recurrent-state row from the
//! same pool as their KV blocks (`PagedDecoder::sequence_state_blocks`) is admitted
//! against both. The row is taken when the sequence's prefill starts, so without
//! the charge a burst of admissions in one iteration promises the same blocks to
//! the prompts and to the rows, and the first prefill to find the pool short dies.

use std::collections::HashMap;

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, PrefillProgress, Request, Scheduler};

const VOCAB: usize = 64;
const BLOCK: u32 = 4;
/// Blocks the pool lends a sequence for its state row.
const STATE: u32 = 3;

struct Mock {
    alloc: BlockAllocator,
    /// Blocks lent per state row.
    lent: HashMap<u32, Vec<u32>>,
    next_row: u32,
    /// Prompt tokens in so far, per state row.
    done: HashMap<u32, usize>,
    /// The most sequences that held a state row at once.
    peak_rows: usize,
    /// Whether the decoder tells the scheduler what a state row costs.
    charges_state: bool,
}

impl Mock {
    fn new(blocks: u32, charges_state: bool) -> Mock {
        Mock { alloc: BlockAllocator::new(blocks, BLOCK), lent: HashMap::new(), next_row: 0, done: HashMap::new(), peak_rows: 0, charges_state }
    }

    fn give_state_row(&mut self, table: &mut BlockTable) {
        if table.state_row().is_some() {
            return;
        }
        let blocks = self.alloc.lend(STATE).expect("the pool had no blocks for the state row the scheduler admitted");
        self.lent.insert(self.next_row, blocks);
        table.set_state_row(self.next_row);
        self.next_row += 1;
        self.peak_rows = self.peak_rows.max(self.lent.len());
    }
}

impl PagedDecoder for Mock {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        64
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        4096
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn sequence_state_blocks(&self) -> u32 {
        if self.charges_state {
            STATE
        } else {
            0
        }
    }
    fn reclaim_prefix(&mut self, _: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(row) = t.take_state_row() {
            self.alloc.repay(self.lent.remove(&row).expect("a row that was given is repaid once"));
        }
        t.release(&mut self.alloc);
    }
    /// Prompts go in a slice a step, like the resident engine's: the first slice reserves the prompt's blocks and takes the
    /// state row, so sequences admitted in one iteration have not taken either yet when the next is admitted.
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(4)
    }
    fn prefill_batch_tokens(&self) -> Option<usize> {
        Some(64)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        if table.is_empty() {
            table.reserve(prompt.len() as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
            self.give_state_row(table);
        }
        let row = table.state_row().expect("a prefill in progress has a state row");
        let done = self.done.entry(row).or_insert(0);
        *done = (*done + budget).min(prompt.len());
        if *done < prompt.len() {
            return PrefillProgress::Partial { consumed: *done };
        }
        self.done.remove(&row);
        PrefillProgress::Done(vec![prompt.iter().sum::<u32>() as f32])
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(h) = self.prefill_chunk(table, prompt, usize::MAX) {
                return h;
            }
        }
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[hidden[0] as usize % VOCAB] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                t.append(&mut self.alloc).expect("the scheduler let a decode step run past the pool");
                (input + 1) % VOCAB as u32
            })
            .collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("these requests decode greedily")
    }
    fn topk_capacity(&self) -> usize {
        1
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
        self.alloc.num_blocks() as u64 * BLOCK as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
}

/// Six requests of 16 tokens: five blocks each with the first token, eight with the state row. A 34-block pool holds four
/// whole sequences (32 blocks); counting prompts alone it would seem to hold six (30).
fn run(charges_state: bool) -> (usize, u32, u32) {
    const POOL: u32 = 34;
    let mut sched = Scheduler::new(Mock::new(POOL, charges_state), 8);
    let ids: Vec<u64> = (0..6u32).map(|i| sched.submit(Request { prompt: (0..16).map(|t| (t + i) % VOCAB as u32).collect(), max_new: 3, stop: Vec::new() })).collect();
    let mut finished = 0;
    for _ in 0..1_000 {
        if !sched.pending() {
            break;
        }
        finished += sched.step().len();
    }
    assert_eq!(finished, ids.len(), "every request finished");
    let dec = sched.decoder_mut();
    (dec.peak_rows, dec.alloc.free_blocks(), POOL)
}

#[test]
fn a_burst_of_admissions_is_charged_for_the_state_rows_it_will_take() {
    let (peak, free, pool) = run(true);
    assert!(peak <= 4, "{peak} sequences held a state row at once in a pool that holds four");
    assert_eq!(free, pool, "every block, lent or not, is back in the pool once the work is done");
}

#[test]
#[should_panic(expected = "no blocks for the state row")]
fn without_the_charge_the_same_burst_over_promises_the_pool() {
    run(false);
}
