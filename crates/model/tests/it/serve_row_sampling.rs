// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampling that stays exact at the full
// vocabulary for its clients. If your team needs expertise in decoding
// strategies for serving language models then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The sampler over a decoder's whole `[vocab]` logits row: nucleus and min-p
//! sets that are wider than any candidate list, and logit processors that move
//! a token from outside the best few.
//!
//! Every expectation is derived from the definition (a flat row's nucleus is
//! `ceil(top_p * n)` tokens, a processor's effect is arithmetic on one logit),
//! not from running the sampler.

use data::rng::Rng;
use model::serve::{sample_logits_row, SampleContext, SampleParams};

/// The width of the candidate list the old sampler was limited to.
const OLD_CAPACITY: usize = 64;

fn draws(row: &[f32], params: SampleParams, n: usize) -> Vec<(u32, f32)> {
    let mut rng = Rng::new(7);
    (0..n).map(|_| sample_logits_row(&mut row.to_vec(), params, &SampleContext::EMPTY, &mut Default::default(), &mut rng)).collect()
}

/// A flat row of 500 tokens: the nucleus of `top_p = 0.9` is the 450 tokens whose mass first reaches 0.9, far
/// wider than a 64-token list. Every draw has probability exactly 1/450, tokens past 64 are drawn, and none
/// past 449 (ties rank lowest id first, so the nucleus is ids 0..450).
#[test]
fn a_nucleus_wider_than_the_candidate_list_is_exact() {
    let row = vec![1.0f32; 500];
    let params = SampleParams { temp: 1.0, top_p: 0.9, ..SampleParams::greedy() };
    let drawn = draws(&row, params, 4000);
    let widest = drawn.iter().map(|d| d.0).max().unwrap();
    assert!(widest as usize >= OLD_CAPACITY, "no draw past the old list width: widest token {widest}");
    assert!(widest < 450, "token {widest} is outside the 450-token nucleus");
    let want = -(450f32).ln();
    for (token, logprob) in drawn {
        assert!((logprob - want).abs() < 1e-4, "token {token}: logprob {logprob}, nucleus of 450 wants {want}");
    }
}

/// min-p keeps every token with at least `min_p` times the best token's probability: 200 tokens within a factor
/// of two of the best survive `min_p = 0.5`, and the 300 far below do not.
#[test]
fn a_min_p_set_wider_than_the_candidate_list_is_exact() {
    let mut row = vec![-20.0f32; 500];
    for l in row.iter_mut().take(200) {
        *l = 1.0 - 0.5 * 0.3; // within ln 2 of the best
    }
    row[0] = 1.0;
    let params = SampleParams { temp: 1.0, min_p: 0.5, ..SampleParams::greedy() };
    let drawn = draws(&row, params, 4000);
    let widest = drawn.iter().map(|d| d.0).max().unwrap();
    assert!(widest as usize >= OLD_CAPACITY && widest < 200, "min-p set is ids 0..200, widest drawn {widest}");
}

/// With no truncation at all the draw is over the whole row: a token far down a long tail is reachable, and its
/// reported log-probability is the full-row softmax, not one renormalised over a list.
#[test]
fn an_untruncated_draw_uses_the_whole_row_normaliser() {
    let row = vec![0.0f32; 1000];
    let drawn = draws(&row, SampleParams { temp: 1.0, ..SampleParams::greedy() }, 8000);
    assert!(drawn.iter().any(|d| d.0 as usize >= 900), "a flat 1000-token row must reach its tail");
    for (_, logprob) in drawn {
        assert!((logprob + (1000f32).ln()).abs() < 1e-4, "logprob {logprob}");
    }
}

/// top_k stays what it was: exactly k tokens, lowest id first among ties, softmax over those k.
#[test]
fn top_k_keeps_exactly_k_lowest_ids_first() {
    let row = vec![1.0f32; 100];
    let drawn = draws(&row, SampleParams { temp: 1.0, top_k: 10, ..SampleParams::greedy() }, 500);
    assert!(drawn.iter().all(|d| d.0 < 10));
    assert!(drawn.iter().all(|d| (d.1 + (10f32).ln()).abs() < 1e-5));
}

fn greedy_pick(mut row: Vec<f32>, params: SampleParams, context: &[u32], prompt_len: usize) -> u32 {
    let ctx = SampleContext { context, prompt_len };
    sample_logits_row(&mut row, params, &ctx, &mut Default::default(), &mut Rng::new(1)).0
}

/// A repeated context pushes the whole top-64 below a token outside the list. A list holds 64 tokens at
/// logit ~10, all divided by 100; the token at id 900 sat at 9.9, outside the list, and is now the best.
#[test]
fn a_penalty_that_sinks_every_top_candidate_lets_an_outside_token_win() {
    let mut row = vec![0.0f32; 1000];
    for (i, l) in row.iter_mut().enumerate().take(OLD_CAPACITY) {
        *l = 10.0 + i as f32 * 1e-3;
    }
    row[900] = 9.9;
    let context: Vec<u32> = (0..OLD_CAPACITY as u32).collect();
    let params = SampleParams { repetition_penalty: 100.0, ..SampleParams::greedy() };
    assert_eq!(greedy_pick(row, params, &context, context.len()), 900);
}

/// A negative presence penalty raises the logit of a token the request has generated: 3.0 + 5.0 beats the best
/// 6.0 although the token was nowhere near the top of the row.
#[test]
fn a_negative_presence_penalty_raises_a_token_from_outside_the_list() {
    let mut row: Vec<f32> = (0..1000).map(|i| 6.0 - i as f32 * 1e-3).collect();
    row[700] = 3.0;
    let context = [1u32, 700];
    let params = SampleParams { presence_penalty: -5.0, ..SampleParams::greedy() };
    assert_eq!(greedy_pick(row, params, &context, 1), 700, "the generated part is [700]");
}

/// A repetition penalty below 1 multiplies a seen positive logit up: 4.0 / 0.5 = 8.0 beats 6.0.
#[test]
fn a_repetition_penalty_below_one_raises_a_seen_token() {
    let mut row: Vec<f32> = (0..1000).map(|i| 6.0 - i as f32 * 1e-3).collect();
    row[800] = 4.0;
    let params = SampleParams { repetition_penalty: 0.5, ..SampleParams::greedy() };
    assert_eq!(greedy_pick(row, params, &[800], 1), 800);
}

/// A token the context never touched keeps the logit the decoder produced, and an id past the row is skipped.
#[test]
fn untouched_tokens_keep_their_logits_and_context_ids_past_the_row_are_ignored() {
    let row = vec![1.0, 5.0, 2.0];
    let params = SampleParams { frequency_penalty: 0.5, ..SampleParams::greedy() };
    assert_eq!(greedy_pick(row, params, &[0, 99], 0), 1);
}
