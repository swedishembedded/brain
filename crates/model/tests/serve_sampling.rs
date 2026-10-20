// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampling that keeps agent models out of
// repetition loops for its clients. If your team needs expertise in decoding
// strategies for long agentic generations then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The request's sampler through the scheduler, for any decoder: logit
//! processors (penalties, DRY) and the thinking budget reach the tokens a
//! sequence emits, and a request that asks for neither decodes as it always did.
//!
//! The toy decoder answers every step with the same logits (token 5 ahead of 6
//! ahead of 7), so what the sequence emits is entirely the sampler's doing.

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{DryParams, PagedDecoder, Request, SampleParams, Scheduler, ThinkingBudget};

const VOCAB: usize = 16;
/// The candidates every step offers: (token, logit).
const CANDIDATES: [(u32, f32); 3] = [(5, 3.0), (6, 2.9), (7, 1.0)];

struct Fixed {
    alloc: BlockAllocator,
}

impl PagedDecoder for Fixed {
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
        vec![0.0]
    }
    fn logits(&self, _hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![-10.0; VOCAB];
        for (t, v) in CANDIDATES {
            l[t as usize] = v;
        }
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|_| 5).collect()
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
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], _k: usize) -> Vec<Vec<(u32, f32)>> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|_| CANDIDATES.to_vec()).collect()
    }
    fn topk_capacity(&self) -> usize {
        8
    }
}

fn generate(params: SampleParams, prompt: Vec<u32>, max_new: usize) -> Vec<u32> {
    let mut sched = Scheduler::new(Fixed { alloc: BlockAllocator::new(256, 8) }, 4);
    let id = sched.submit_sampled(Request { prompt, max_new, stop: Vec::new() }, params, 1);
    sched.run().remove(&id).expect("the request completes")
}

/// A request that asks for no processor decodes exactly as before: the argmax, token 5, every step.
#[test]
fn a_greedy_request_without_processors_is_the_plain_argmax() {
    assert_eq!(generate(SampleParams::greedy(), vec![1, 2, 3], 6), vec![5; 6]);
}

/// Greedy decoding with a penalty is the argmax of the penalised logits, so the penalty reaches the
/// first token (admission) and every later one: with `frequency_penalty` 0.4 token 5 and token 6 alternate.
#[test]
fn a_frequency_penalty_changes_what_a_greedy_request_emits() {
    let params = SampleParams { frequency_penalty: 0.4, ..SampleParams::greedy() };
    assert_eq!(generate(params, vec![1, 2, 3], 6), vec![5, 6, 5, 6, 5, 6]);
}

/// Presence and repetition penalties count what the request has generated (repetition also the prompt):
/// a prompt full of 5s alone does not move a presence penalty, but does move a repetition penalty.
#[test]
fn penalties_read_the_right_part_of_the_context() {
    let presence = SampleParams { presence_penalty: 0.5, ..SampleParams::greedy() };
    assert_eq!(generate(presence, vec![5, 5, 5], 1), vec![5], "a prompt-only token is not penalised by presence");
    let repetition = SampleParams { repetition_penalty: 2.0, ..SampleParams::greedy() };
    assert_eq!(generate(repetition, vec![5, 5, 5], 1), vec![6], "the prompt counts for a repetition penalty: 5 falls to 1.5 behind 6");
}

/// DRY over the whole context: after a prompt `5 5 5` the argmax (5) would extend a repeat of two tokens, so it
/// is pushed down and the next candidate leads.
#[test]
fn dry_uses_the_prompt_as_context() {
    let dry = SampleParams { dry: DryParams::new(10.0, 2.0, 2, &[]), ..SampleParams::greedy() };
    assert_eq!(generate(dry, vec![5, 5, 5], 1), vec![6]);
    assert_eq!(generate(dry, vec![1, 2, 3], 1), vec![5], "no repeat in the context, no penalty");
}

/// The budget forces `</think>` (here token 1 then 2) after three tokens inside the block, then the model goes on.
#[test]
fn a_thinking_budget_forces_the_closing_sequence_and_the_model_continues() {
    let budget = ThinkingBudget::new(3, 9, &[1, 2], true);
    let params = SampleParams { thinking: Some(budget), ..SampleParams::greedy() };
    assert_eq!(generate(params, vec![1, 2, 3], 8), vec![5, 5, 5, 1, 2, 5, 5, 5]);
    // The same through the sampled path.
    let sampled = SampleParams { temp: 1.0, top_k: 1, thinking: Some(budget), ..SampleParams::greedy() };
    assert_eq!(generate(sampled, vec![1, 2, 3], 8), vec![5, 5, 5, 1, 2, 5, 5, 5]);
}

/// A budget of zero on a prompt that ends inside the block closes it before the model says anything.
#[test]
fn a_zero_budget_closes_the_block_at_the_first_token() {
    let params = SampleParams { thinking: Some(ThinkingBudget::new(0, 9, &[1], true)), ..SampleParams::greedy() };
    assert_eq!(generate(params, vec![1, 2, 3], 3), vec![1, 5, 5]);
}
