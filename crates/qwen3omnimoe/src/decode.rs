// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampling and decode loops that behave the same
// on every serving engine for its clients. If your team needs expertise in
// token sampling for language models then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The Thinker's token-picking loop, shared by every decode path.
//!
//! The bf16 stack (`crate::generate`) and the int8 sharded stack
//! (`crate::int8_thinker_resident`) differ in how they turn a token into the
//! next logits row; everything after that row is the same: pick a token with
//! brain's one sampler (`model::serve::sample_logits_row`, over the whole row,
//! so `top_k`, `top_p`, `min_p` and the penalties are exact), stop on an end of
//! turn id, tell the caller about each new token so it can stream text, apply
//! stop strings and honour cancellation, and stop at the budget. That is
//! [`decode_tokens`]; each stack hands it a closure that runs one decode step.
//! Because the loop owns no device state it is tested without a GPU.

use data::rng::Rng;
use model::serve::{sample_logits_row, ProcessorState, SampleContext, SampleParams};

/// How one request picks its tokens: the sampler parameters, the random
/// stream a seeded request reproduces, and the processor state (penalties)
/// that follows the sequence.
pub struct TokenPicker {
    params: SampleParams,
    rng: Rng,
    state: ProcessorState,
}

impl TokenPicker {
    /// `params` drawn from the stream `seed` starts. `temp <= 0` in `params`
    /// is greedy (the argmax of the logits, after any penalty) and never
    /// reads the stream.
    pub fn new(params: SampleParams, seed: u64) -> TokenPicker {
        TokenPicker { params, rng: Rng::new(seed), state: ProcessorState::default() }
    }

    /// Greedy decoding, the behaviour of a request that names no temperature.
    pub fn greedy() -> TokenPicker {
        TokenPicker::new(SampleParams::greedy(), data::rng::random_seed())
    }

    fn pick(&mut self, row: &mut [f32], context: &[u32], prompt_len: usize) -> u32 {
        sample_logits_row(row, self.params, &SampleContext { context, prompt_len }, &mut self.state, &mut self.rng).0
    }
}

/// What a decode produced.
#[derive(Debug, PartialEq, Eq)]
pub struct Decoded {
    /// The generated tokens, an end-of-turn id excluded.
    pub tokens: Vec<u32>,
    /// The end-of-turn id the model picked when the loop ended on one.
    pub eos: Option<u32>,
}

/// Generate up to `max_new` tokens after `prompt_ids` (at least one, the caller
/// has handled a zero budget).
///
/// `first_row` is the logits row after the prompt; `next_row(token)` runs one
/// decode step for the token just picked and returns the next row, and is not
/// called after the last token is picked. `on_token` sees every generated token
/// so far after each pick and returns `true` to stop (a stop string matched, the
/// request was cancelled). An end-of-turn id ends the loop and is neither
/// reported to `on_token` nor returned.
pub fn decode_tokens(
    prompt_ids: &[u32],
    max_new: u32,
    eos_ids: &[u32],
    picker: &mut TokenPicker,
    first_row: Vec<f32>,
    mut next_row: impl FnMut(u32) -> Vec<f32>,
    on_token: &mut dyn FnMut(&[u32]) -> bool,
) -> Decoded {
    let mut context = prompt_ids.to_vec();
    let prompt_len = context.len();
    let mut row = first_row;
    loop {
        let token = picker.pick(&mut row, &context, prompt_len);
        if eos_ids.contains(&token) {
            return Decoded { tokens: context.split_off(prompt_len), eos: Some(token) };
        }
        context.push(token);
        let generated = &context[prompt_len..];
        if on_token(generated) || generated.len() as u32 >= max_new {
            return Decoded { tokens: context.split_off(prompt_len), eos: None };
        }
        row = next_row(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 16;

    /// The logits row a fixed pseudo-random "model" produces after `token`:
    /// the same row for the same token, so a decode is a pure function of the
    /// sampler and its seed.
    fn row_after(token: u32) -> Vec<f32> {
        data::rng::Lcg::new(1000 + token as u64).vec_scaled(VOCAB, 4.0)
    }

    fn run(picker: &mut TokenPicker, max_new: u32, eos: &[u32]) -> Decoded {
        decode_tokens(&[3, 5], max_new, eos, picker, row_after(5), row_after, &mut |_| false)
    }

    fn argmax_chain(max_new: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut prev = 5;
        for _ in 0..max_new {
            let row = row_after(prev);
            let best = (0..VOCAB).max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap() as u32;
            out.push(best);
            prev = best;
        }
        out
    }

    fn warm(seed: u64) -> TokenPicker {
        TokenPicker::new(SampleParams { temp: 1.5, top_k: 0, top_p: 1.0, ..SampleParams::greedy() }, seed)
    }

    /// Temperature 0 is the argmax of every row: what generation did before it sampled.
    #[test]
    fn temperature_zero_is_the_argmax_chain() {
        let got = run(&mut TokenPicker::new(SampleParams::greedy(), 7), 12, &[]);
        assert_eq!(got.tokens, argmax_chain(12));
        assert_eq!(got.eos, None);
    }

    /// A seeded request is reproducible, and the seed is what varies it.
    #[test]
    fn a_seed_reproduces_a_sampled_decode_and_another_seed_changes_it() {
        let a = run(&mut warm(11), 24, &[]).tokens;
        assert_eq!(a, run(&mut warm(11), 24, &[]).tokens);
        assert_ne!(a, argmax_chain(24), "a hot temperature must leave the argmax chain");
        assert!((100..110).any(|seed| run(&mut warm(seed), 24, &[]).tokens != a), "ten other seeds all drew the same 24 tokens");
    }

    /// `top_k = 1` leaves one candidate, so it is the argmax however hot the temperature.
    #[test]
    fn top_k_one_is_greedy_at_any_temperature() {
        let picker = &mut TokenPicker::new(SampleParams { temp: 2.0, top_k: 1, ..SampleParams::greedy() }, 3);
        assert_eq!(run(picker, 12, &[]).tokens, argmax_chain(12));
    }

    /// A penalty reaches the draw: a repetition penalty far above 1 forbids a token that is already in the context.
    #[test]
    fn the_request_penalties_apply_to_the_context() {
        let picker = &mut TokenPicker::new(SampleParams { repetition_penalty: 1000.0, ..SampleParams::greedy() }, 3);
        let tokens = run(picker, 12, &[]).tokens;
        let mut seen = vec![3, 5];
        for t in tokens {
            assert!(!seen.contains(&t), "token {t} repeated under a huge repetition penalty");
            seen.push(t);
        }
    }

    /// An end-of-turn id ends the decode and is not part of the reply.
    #[test]
    fn an_end_of_turn_id_ends_the_decode_and_is_not_returned() {
        let chain = argmax_chain(12);
        let eos = chain[4];
        let first = chain.iter().position(|&t| t == eos).unwrap();
        let got = run(&mut TokenPicker::greedy(), 12, &[eos]);
        assert_eq!(got.tokens, chain[..first]);
        assert_eq!(got.eos, Some(eos));
    }

    /// The budget bounds the reply, and no step is run for a token that will not be used.
    #[test]
    fn the_budget_bounds_the_reply_and_costs_no_extra_step() {
        let mut steps = 0;
        let got = decode_tokens(
            &[3, 5],
            4,
            &[],
            &mut TokenPicker::greedy(),
            row_after(5),
            |t| {
                steps += 1;
                row_after(t)
            },
            &mut |_| false,
        );
        assert_eq!(got.tokens.len(), 4);
        assert_eq!(steps, 3, "four tokens need one prefill row and three decode steps");
    }

    /// The caller sees every token as it is picked and can end the decode (a stop string, a cancellation).
    #[test]
    fn the_caller_sees_each_token_and_can_stop_the_decode() {
        let mut seen = Vec::new();
        let got = decode_tokens(
            &[3, 5],
            12,
            &[],
            &mut TokenPicker::greedy(),
            row_after(5),
            row_after,
            &mut |generated| {
                seen.push(generated.len());
                generated.len() == 3
            },
        );
        assert_eq!(seen, vec![1, 2, 3]);
        assert_eq!(got.tokens, argmax_chain(3));
    }
}
