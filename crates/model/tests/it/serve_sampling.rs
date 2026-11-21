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
use model::serve::{DryParams, PagedDecoder, RejectReason, Request, SampleParams, Scheduler, ThinkingBudget};

const VOCAB: usize = 16;
/// The candidates every step offers: (token, logit).
const CANDIDATES: [(u32, f32); 3] = [(5, 3.0), (6, 2.9), (7, 1.0)];

struct Fixed {
    alloc: BlockAllocator,
    /// A decoder that can read its whole logits row back answers every step with this row; `None` is a
    /// decoder that only offers the `CANDIDATES` list.
    row: Option<Vec<f32>>,
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
        self.row.as_ref().map_or(VOCAB, Vec::len)
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
        self.row.clone().unwrap_or_else(candidate_row)
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
    fn full_row_readback(&self) -> bool {
        self.row.is_some()
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        let row = self.row.clone().expect("the scheduler asks for rows only of a decoder that offers them");
        inputs.iter().map(|_| row.clone()).collect()
    }
}

fn generate(params: SampleParams, prompt: Vec<u32>, max_new: usize) -> Vec<u32> {
    generate_with(None, params, prompt, max_new)
}

/// The whole row behind [`CANDIDATES`]: a penalty ranks the processed row, so it is sampled from the row a
/// decoder reads back.
fn candidate_row() -> Vec<f32> {
    let mut l = vec![-10.0; VOCAB];
    for (t, v) in CANDIDATES {
        l[t as usize] = v;
    }
    l
}

fn generate_penalised(params: SampleParams, prompt: Vec<u32>, max_new: usize) -> Vec<u32> {
    generate_with(Some(candidate_row()), params, prompt, max_new)
}

/// A decoder that offers only its candidate list refuses a request it cannot sample exactly from that list
/// (`top_p`, a penalty, no `top_k` within the list), rather than drawing from a truncated distribution while
/// reporting the requested one; greedy decoding and a `top_k` within the list are still served.
#[test]
fn a_decoder_without_row_readback_refuses_what_its_candidates_cannot_answer_exactly() {
    for params in [
        SampleParams { temp: 1.0, top_p: 0.9, ..SampleParams::greedy() },
        SampleParams { temp: 1.0, ..SampleParams::greedy() },
        SampleParams { temp: 1.0, top_k: 9, ..SampleParams::greedy() },
        SampleParams { frequency_penalty: 0.4, ..SampleParams::greedy() },
    ] {
        let mut sched = Scheduler::new(Fixed { alloc: BlockAllocator::new(256, 8), row: None }, 4);
        let id = sched.submit_sampled(Request { prompt: vec![1, 2, 3], max_new: 4, stop: Vec::new() }, params, 1);
        let report = sched.step_report();
        assert!(matches!(report.rejected.as_slice(), [(got, RejectReason::InexactSampling { capacity: 8 })] if *got == id), "{params:?}: {:?}", report.rejected);
    }
    assert_eq!(generate(SampleParams { temp: 1.0, top_k: 3, ..SampleParams::greedy() }, vec![1, 2, 3], 4).len(), 4);
    assert_eq!(generate(SampleParams::greedy(), vec![1, 2, 3], 4), vec![5; 4]);
}

fn generate_with(row: Option<Vec<f32>>, params: SampleParams, prompt: Vec<u32>, max_new: usize) -> Vec<u32> {
    let mut sched = Scheduler::new(Fixed { alloc: BlockAllocator::new(256, 8), row }, 4);
    let id = sched.submit_sampled(Request { prompt, max_new, stop: Vec::new() }, params, 1);
    sched.run().remove(&id).expect("the request completes")
}

/// A decoder that reads its whole row back is sampled over that row, not over its 8-token candidate list: a
/// flat 300-token row with `top_p` 0.9 has a 270-token nucleus, so over 200 draws tokens past the list width
/// appear, and none past the nucleus.
#[test]
fn a_decoder_with_row_readback_is_sampled_over_the_whole_row() {
    let params = SampleParams { temp: 1.0, top_p: 0.9, ..SampleParams::greedy() };
    let drawn = generate_with(Some(vec![1.0; 300]), params, vec![1, 2, 3], 200);
    assert!(drawn.iter().any(|&t| t >= 8), "every draw stayed inside the 8-token candidate list");
    assert!(drawn.iter().all(|&t| t < 270), "a draw left the 270-token nucleus: {drawn:?}");
}

/// A penalty is ranked against the whole row too: tokens 0..8 lead the row, a repetition penalty of 100 over a
/// context holding all of them sinks them below token 9, which the 8-token list never held.
#[test]
fn penalties_rank_the_tokens_the_candidate_list_never_held() {
    let mut row = vec![0.0f32; 20];
    for (i, l) in row.iter_mut().enumerate().take(8) {
        *l = 10.0 + i as f32 * 0.01;
    }
    row[9] = 9.0;
    let params = SampleParams { repetition_penalty: 100.0, ..SampleParams::greedy() };
    assert_eq!(generate_with(Some(row), params, (0..8).collect(), 1), vec![9]);
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
    assert_eq!(generate_penalised(params, vec![1, 2, 3], 6), vec![5, 6, 5, 6, 5, 6]);
}

/// Presence and repetition penalties count what the request has generated (repetition also the prompt):
/// a prompt full of 5s alone does not move a presence penalty, but does move a repetition penalty.
#[test]
fn penalties_read_the_right_part_of_the_context() {
    let presence = SampleParams { presence_penalty: 0.5, ..SampleParams::greedy() };
    assert_eq!(generate_penalised(presence, vec![5, 5, 5], 1), vec![5], "a prompt-only token is not penalised by presence");
    let repetition = SampleParams { repetition_penalty: 2.0, ..SampleParams::greedy() };
    assert_eq!(generate_penalised(repetition, vec![5, 5, 5], 1), vec![6], "the prompt counts for a repetition penalty: 5 falls to 1.5 behind 6");
}

/// DRY over the whole context: after a prompt `5 5 5` the argmax (5) would extend a repeat of two tokens, so it
/// is pushed down and the next candidate leads.
#[test]
fn dry_uses_the_prompt_as_context() {
    let dry = SampleParams { dry: DryParams::new(10.0, 2.0, 2, &[]), ..SampleParams::greedy() };
    assert_eq!(generate_penalised(dry, vec![5, 5, 5], 1), vec![6]);
    assert_eq!(generate_penalised(dry, vec![1, 2, 3], 1), vec![5], "no repeat in the context, no penalty");
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

/// A decoder whose greedy choice at each generated position is read from a script (position `n` of the generation
/// gets `script[n]`), except right after [`ANSWER_CUE`], which it always answers with [`ANSWER`], with an on-device
/// window of `window` tokens: what a sequence emits beyond the script is the scheduler's doing, a window rolled
/// back wrongly shifts every later token, and a window that runs on past a token the scheduler replaced shows the
/// draw that followed the model's own token instead of the replacement.
struct Scripted {
    alloc: BlockAllocator,
    prompt_len: usize,
    script: Vec<u32>,
    window: usize,
    /// The window sizes the scheduler asked for, in order.
    windows: Vec<usize>,
}

/// The token [`Scripted`] always follows with [`ANSWER`] (the closing tag of the tests below).
const ANSWER_CUE: u32 = 1;
const ANSWER: u32 = 13;

impl Scripted {
    fn at(&self, generated: usize) -> u32 {
        self.script.get(generated).copied().unwrap_or(0)
    }
}

impl PagedDecoder for Scripted {
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
        VOCAB * 8
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
        let mut row = vec![0.0; self.vocab()];
        row[self.at(0) as usize] = 1.0;
        row
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.forward_batched_greedy_window(tables, inputs, 1).into_iter().map(|w| w[0]).collect()
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        self.windows.push(k);
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                // The cache holds the prompt and every generated token but the input: the input is token `n - 1`.
                let n = t.len() as usize - self.prompt_len + 1;
                for _ in 0..k {
                    t.append(&mut self.alloc).unwrap();
                }
                let mut prev = input;
                (n..n + k)
                    .map(|i| {
                        prev = if prev == ANSWER_CUE { ANSWER } else { self.at(i) };
                        prev
                    })
                    .collect()
            })
            .collect()
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
        self.window
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("greedy only")
    }
    fn topk_capacity(&self) -> usize {
        1
    }
}

/// Runs the script greedily under `budget` with `stop` as the end-of-turn tokens; returns what the sequence
/// emitted and the window sizes the scheduler took.
fn scripted(script: &[u32], budget: ThinkingBudget, stop: Vec<u32>, max_new: usize) -> (Vec<u32>, Vec<usize>) {
    let prompt = vec![1, 2, 3];
    let dec = Scripted { alloc: BlockAllocator::new(256, 8), prompt_len: prompt.len(), script: script.to_vec(), window: 8, windows: Vec::new() };
    let mut sched = Scheduler::new(dec, 4);
    let params = SampleParams { thinking: Some(budget), ..SampleParams::greedy() };
    let id = sched.submit_sampled(Request { prompt, max_new, stop }, params, 1);
    let out = sched.run().remove(&id).expect("the request completes");
    (out, sched.decoder().windows.clone())
}

/// The budget is the turn's, and a block the model reopens outside a watched stretch is still held to it: in a
/// decode window that reopens the block and spends the rest of the budget, the closing tag replaces the next token
/// and ends the window, so what follows is the model's answer to the closing tag, not to the token it replaced.
#[test]
fn a_block_reopened_inside_a_decode_window_is_closed_when_the_turn_budget_runs_out() {
    let (open, close) = (9u32, ANSWER_CUE);
    let script = [open, 10, close, 0, open, 12, 12, 12, 12, 12];
    let (out, windows) = scripted(&script, ThinkingBudget::new(2, open, &[close], false), Vec::new(), 10);
    // Block one spends 1 token; the window [13, 9, 12, 12, ...] reopens and spends the second, so the tag replaces
    // the next 12 and the model answers it.
    assert_eq!(out, vec![open, 10, close, ANSWER, open, 12, close, ANSWER, 12, 12]);
    assert!(windows.iter().any(|&k| k > 1), "outside a block the scheduler takes decode windows: {windows:?}");
}

/// A tool call written inside the block is never cut by the budget, and reasoning after it that runs out of the
/// budget ends the turn with the request's end-of-turn token.
#[test]
fn a_call_inside_the_block_is_whole_and_running_out_after_it_ends_the_turn() {
    let (open, close, call, end_call, eot) = (9u32, 1u32, 20u32, 21u32, 99u32);
    let script = [10, call, 30, 30, 30, end_call, 10, 10, 10, 10];
    let budget = ThinkingBudget::new(2, open, &[close], true).with_call_fence(call, end_call);
    let (out, _) = scripted(&script, budget, vec![eot], 10);
    assert_eq!(out, vec![10, call, 30, 30, 30, end_call, 10], "the turn ends at the budget after the call");
    let (out, _) = scripted(&script, budget, Vec::new(), 10);
    assert_eq!(out, vec![10, call, 30, 30, 30, end_call, 10], "a request with no end-of-turn token just ends");
}
