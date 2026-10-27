// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements LLM serving that degrades by refusing a request instead of failing the server. If
// your team needs expertise in serving large models at the memory limit of a card, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a device allocation that fails while a prompt is being read in refuses the prompts being read in with a
//! device-out-of-memory reason, returns their blocks, and leaves the scheduler serving: a sequence that was already
//! decoding finishes, and a later request is answered.

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{PagedDecoder, RejectReason, Request, Scheduler};

const VOCAB: usize = 64;
/// A prompt this long does not fit the card's workspace.
const TOO_LONG: usize = 100;

struct Toy {
    alloc: BlockAllocator,
    abandoned: usize,
    /// Decode steps left before one runs the card out of memory (`usize::MAX`: never).
    steps_until_oom: usize,
}

impl PagedDecoder for Toy {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        256
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        1024
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(8)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        if prompt.len() >= TOO_LONG {
            std::panic::resume_unwind(Box::new(gpu_core::DeviceOutOfMemory { bytes: 48 << 20, what: "storage".into(), detail: "out of memory".into() }));
        }
        vec![*prompt.last().unwrap() as f32]
    }
    fn abandon_prefill(&mut self) {
        self.abandoned += 1;
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[(hidden[0] as usize + 1) % VOCAB] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if self.steps_until_oom == 0 {
            std::panic::resume_unwind(Box::new(gpu_core::DeviceOutOfMemory { bytes: 252 << 20, what: "storage".into(), detail: "out of memory".into() }));
        }
        self.steps_until_oom = self.steps_until_oom.saturating_sub(1);
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|&i| (i + 1) % VOCAB as u32).collect()
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
        4096
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("greedy only")
    }
    fn topk_capacity(&self) -> usize {
        1
    }
}

fn prompt(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| i % VOCAB as u32).collect()
}

#[test]
fn a_full_card_refuses_the_prompt_it_was_reading_and_keeps_serving() {
    let mut sched = Scheduler::new(Toy { alloc: BlockAllocator::new(512, 8), abandoned: 0, steps_until_oom: usize::MAX }, 4);
    let free = sched.free_blocks();
    let running = sched.submit(Request { prompt: prompt(10), max_new: 6, stop: Vec::new() });
    sched.step();
    let doomed = sched.submit(Request { prompt: prompt(TOO_LONG), max_new: 4, stop: Vec::new() });

    let report = sched.step_report();
    assert!(
        matches!(report.rejected.as_slice(), [(id, RejectReason::DeviceOutOfMemory { bytes })] if *id == doomed && *bytes == 48 << 20),
        "the prompt that hit the full card is refused with its reason: {:?}",
        report.rejected
    );
    assert_eq!(sched.decoder_mut().abandoned, 1, "the decoder is told to drop its prefill work");

    // The sequence that was decoding finishes, and a request after the failure is answered.
    let later = sched.submit(Request { prompt: prompt(12), max_new: 3, stop: Vec::new() });
    let mut done = Vec::new();
    while sched.pending() {
        done.extend(sched.step());
    }
    let ids: Vec<u64> = done.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&running) && ids.contains(&later), "{ids:?}");
    assert_eq!(done.iter().find(|(id, _)| *id == running).unwrap().1.len(), 6);
    assert_eq!(sched.free_blocks(), free, "every block, the refused prompt's included, is back in the pool");
}

#[test]
fn a_full_card_in_a_decode_step_refuses_that_steps_sequences_with_a_classed_reason() {
    let mut sched = Scheduler::new(Toy { alloc: BlockAllocator::new(512, 8), abandoned: 0, steps_until_oom: 2 }, 4);
    let free = sched.free_blocks();
    let a = sched.submit(Request { prompt: prompt(10), max_new: 20, stop: Vec::new() });
    let b = sched.submit(Request { prompt: prompt(12), max_new: 20, stop: Vec::new() });
    let mut rejected = Vec::new();
    for _ in 0..8 {
        rejected.extend(sched.step_report().rejected);
    }
    let ids: Vec<u64> = rejected.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&a) && ids.contains(&b), "both sequences of the failed step are refused: {rejected:?}");
    assert!(rejected.iter().all(|(_, r)| matches!(r, RejectReason::DeviceOutOfMemory { bytes } if *bytes == 252 << 20)));
    assert!(!sched.pending(), "nothing is left running");
    assert_eq!(sched.free_blocks(), free, "their blocks are back in the pool");

    // The scheduler serves on: once the card has room again the next request is answered.
    sched.decoder_mut().steps_until_oom = usize::MAX;
    let later = sched.submit(Request { prompt: prompt(9), max_new: 3, stop: Vec::new() });
    let mut done = Vec::new();
    while sched.pending() {
        done.extend(sched.step());
    }
    assert!(done.iter().any(|(id, toks)| *id == later && toks.len() == 3));
}
