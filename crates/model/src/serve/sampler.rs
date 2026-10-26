// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sampling that keeps agent models out of
// repetition loops for its clients. If your team needs expertise in decoding
// strategies for long agentic generations then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The logit processors of the shared sampler: the penalties of the OpenAI and
//! vLLM request fields (presence, frequency, repetition) and the DRY ("don't
//! repeat yourself") repetition penalty with llama.cpp's semantics. They run on
//! a decoder's candidate list (the top logits of the step) before temperature,
//! top-k, top-p and min-p, for every model the scheduler serves.
//!
//! Every processor only ever LOWERS a logit, so working on the candidate list
//! (the decoder's `topk_capacity` best tokens) is exact for the tokens that can
//! be chosen: a penalised token drops down the list and the next candidate takes
//! its place; a token outside the list was already behind all of them.
//!
//! All processors are off by default ([`SampleParams::has_processors`]); a request
//! that asks for none is sampled exactly as before.

use std::collections::HashMap;

use super::SampleParams;

/// The most sequence breakers a [`DryParams`] holds (llama.cpp's default is four).
pub const MAX_DRY_BREAKERS: usize = 16;

/// DRY: penalise a token that would extend a sequence already present in the
/// context. A token that continues a repeat of `len >= allowed_length` tokens
/// has `multiplier * base^(len - allowed_length)` subtracted from its logit.
/// `multiplier == 0` is off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DryParams {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: u32,
    breakers: [u32; MAX_DRY_BREAKERS],
    n_breakers: u8,
}

impl Default for DryParams {
    fn default() -> DryParams {
        DryParams::off()
    }
}

impl DryParams {
    /// No DRY penalty (llama.cpp's base and allowed length, multiplier 0).
    pub const fn off() -> DryParams {
        DryParams { multiplier: 0.0, base: 1.75, allowed_length: 2, breakers: [0; MAX_DRY_BREAKERS], n_breakers: 0 }
    }

    /// `breakers` are token ids that end a sequence: a repeat is not matched
    /// across one. At most [`MAX_DRY_BREAKERS`] are kept.
    pub fn new(multiplier: f32, base: f32, allowed_length: u32, breakers: &[u32]) -> DryParams {
        let mut set = [0u32; MAX_DRY_BREAKERS];
        let n = breakers.len().min(MAX_DRY_BREAKERS);
        set[..n].copy_from_slice(&breakers[..n]);
        DryParams { multiplier, base, allowed_length, breakers: set, n_breakers: n as u8 }
    }

    pub fn active(&self) -> bool {
        self.multiplier > 0.0 && self.base > 0.0
    }

    pub fn breakers(&self) -> &[u32] {
        &self.breakers[..self.n_breakers as usize]
    }
}

/// The most tokens a [`NoRepeatNgram`] exempts from its ban.
pub const MAX_NGRAM_ALLOW: usize = 4;

/// The no-repeat n-gram filter (vLLM's `NoRepeatNGramLogitsProcessor`, which
/// DeepSeek-OCR's reference recipe wires in): a token that would complete an
/// `ngram_size`-gram already present in the last `window` generated tokens is
/// banned, except the `allow`ed tokens (table-cell tags legitimately repeat).
/// It looks at the GENERATED tokens only, never the prompt. `ngram_size == 0`
/// is off.
///
/// Unlike the penalties it never needs the decoder's whole candidate list:
/// the scheduler knows the banned set from the history BEFORE the step, so it
/// asks for exactly `1 + banned` candidates (a plain greedy step when none is
/// banned, which is almost always) and takes the best one that is not banned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoRepeatNgram {
    pub ngram_size: u32,
    pub window: u32,
    allow: [u32; MAX_NGRAM_ALLOW],
    n_allow: u8,
}

impl Default for NoRepeatNgram {
    fn default() -> NoRepeatNgram {
        NoRepeatNgram::off()
    }
}

impl NoRepeatNgram {
    pub const fn off() -> NoRepeatNgram {
        NoRepeatNgram { ngram_size: 0, window: 0, allow: [0; MAX_NGRAM_ALLOW], n_allow: 0 }
    }

    /// At most [`MAX_NGRAM_ALLOW`] allowed tokens are kept.
    pub fn new(ngram_size: u32, window: u32, allow: &[u32]) -> NoRepeatNgram {
        let mut set = [0u32; MAX_NGRAM_ALLOW];
        let n = allow.len().min(MAX_NGRAM_ALLOW);
        set[..n].copy_from_slice(&allow[..n]);
        NoRepeatNgram { ngram_size, window, allow: set, n_allow: n as u8 }
    }

    pub fn active(&self) -> bool {
        self.ngram_size > 0
    }

    pub fn allowed(&self) -> &[u32] {
        &self.allow[..self.n_allow as usize]
    }

    /// The tokens that may not follow `generated` (see [`ngram_followers`]),
    /// less the allowed ones.
    pub fn banned(&self, generated: &[u32]) -> Vec<u32> {
        let mut out = ngram_followers(generated, self.ngram_size as usize, self.window as usize);
        out.retain(|t| !self.allowed().contains(t));
        out
    }
}

/// The tokens that followed an earlier occurrence, inside the last `window`
/// of `generated`, of its trailing `ngram_size - 1` tokens: the tokens that
/// would complete an n-gram already seen. Distinct, in order of appearance;
/// empty until `ngram_size` tokens exist. The one definition both the
/// scheduler's [`NoRepeatNgram`] and the logits form
/// ([`super::apply_no_repeat_ngram`]) use.
pub fn ngram_followers(generated: &[u32], ngram_size: usize, window: usize) -> Vec<u32> {
    if ngram_size == 0 || generated.len() < ngram_size {
        return Vec::new();
    }
    let prefix = &generated[generated.len() - (ngram_size - 1)..];
    let start = generated.len().saturating_sub(window);
    let end = generated.len() - ngram_size + 1;
    let mut out: Vec<u32> = Vec::new();
    for i in start..end {
        let next = generated[i + ngram_size - 1];
        if generated[i..i + ngram_size - 1] == *prefix && !out.contains(&next) {
            out.push(next);
        }
    }
    out
}

/// A thinking budget: once `max_tokens` tokens have been generated inside an
/// open think block the scheduler forces an optional lead-in (a last thought
/// that tells the model to answer from what it has, still inside the block)
/// and then the closing sequence (`</think>` and whatever separates it from the
/// answer) in place of the model's own choices, and then lets the model
/// continue with its answer. The token ids belong to the model's tokenizer, so
/// the frontend supplies them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThinkingBudget {
    /// Tokens the model may spend inside the think block (the tags excluded).
    pub max_tokens: u32,
    /// The id of the opening tag.
    pub open: u32,
    close: [u32; MAX_THINK_CLOSE],
    n_close: u8,
    lead: [u32; MAX_THINK_LEAD],
    n_lead: u8,
    /// The prompt already ended inside an open think block.
    pub open_at_start: bool,
}

/// The longest closing sequence a [`ThinkingBudget`] forces.
pub const MAX_THINK_CLOSE: usize = 4;

/// The longest lead-in a [`ThinkingBudget`] forces before its closing sequence.
pub const MAX_THINK_LEAD: usize = 48;

impl ThinkingBudget {
    /// `close` is the sequence forced when the budget runs out; its first token is
    /// the closing tag (the model emitting it ends the block by itself). At most
    /// [`MAX_THINK_CLOSE`] tokens are kept.
    pub fn new(max_tokens: u32, open: u32, close: &[u32], open_at_start: bool) -> ThinkingBudget {
        assert!(!close.is_empty(), "a thinking budget needs a closing sequence");
        let mut set = [0u32; MAX_THINK_CLOSE];
        let n = close.len().min(MAX_THINK_CLOSE);
        set[..n].copy_from_slice(&close[..n]);
        ThinkingBudget { max_tokens, open, close: set, n_close: n as u8, lead: [0; MAX_THINK_LEAD], n_lead: 0, open_at_start }
    }

    /// The same budget with `lead` forced before the closing sequence. Panics on
    /// a lead-in longer than [`MAX_THINK_LEAD`]: it is the frontend's constant,
    /// and a truncated one would be a different sentence.
    pub fn with_lead_in(mut self, lead: &[u32]) -> ThinkingBudget {
        assert!(lead.len() <= MAX_THINK_LEAD, "a thinking lead-in of {} tokens is over {MAX_THINK_LEAD}", lead.len());
        self.lead[..lead.len()].copy_from_slice(lead);
        self.n_lead = lead.len() as u8;
        self
    }

    pub fn close(&self) -> &[u32] {
        &self.close[..self.n_close as usize]
    }

    pub fn lead_in(&self) -> &[u32] {
        &self.lead[..self.n_lead as usize]
    }

    /// The `i`-th token forced when the budget runs out: the lead-in, then the closing sequence.
    fn forced_at(&self, i: usize) -> u32 {
        let lead = self.lead_in();
        if i < lead.len() { lead[i] } else { self.close()[i - lead.len()] }
    }

    fn forced_len(&self) -> usize {
        self.n_lead as usize + self.n_close as usize
    }
}

/// Where a sequence is in its think block, advanced by every accepted token.
#[derive(Clone, Copy, Debug)]
pub(super) struct ThinkState {
    budget: ThinkingBudget,
    in_think: bool,
    count: u32,
    /// The budget ran out: the lead-in and closing sequence are being forced, `forced` tokens of them so far.
    closing: bool,
    forced: usize,
    /// The block was closed (by the model or by force): nothing left to watch.
    done: bool,
}

impl ThinkState {
    pub(super) fn new(budget: ThinkingBudget) -> ThinkState {
        let mut s = ThinkState { budget, in_think: budget.open_at_start, count: 0, closing: false, forced: 0, done: false };
        s.check_budget();
        s
    }

    fn check_budget(&mut self) {
        if self.in_think && !self.closing && self.count >= self.budget.max_tokens {
            self.closing = true;
            self.forced = 0;
        }
    }

    /// The token the scheduler must emit next instead of the model's choice, while the closing sequence is being forced.
    pub(super) fn forced_next(&self) -> Option<u32> {
        self.closing.then(|| self.budget.forced_at(self.forced))
    }

    /// Whether the sequence still has to be decoded one host-checked token at a
    /// time (a multi-token on-device window would step over the budget).
    pub(super) fn watching(&self) -> bool {
        !self.done
    }

    pub(super) fn accept(&mut self, token: u32) {
        if self.closing {
            self.forced += 1;
            if self.forced == self.budget.forced_len() {
                self.closing = false;
                self.in_think = false;
                self.done = true;
            }
            return;
        }
        if token == self.budget.open {
            self.in_think = true;
            self.count = 0;
        } else if self.in_think && token == self.budget.close()[0] {
            self.in_think = false;
            self.done = true;
        } else if self.in_think {
            self.count += 1;
        }
        self.check_budget();
    }
}

/// The tokens a request has seen: the whole context (prompt then generated) and
/// where the generated part starts.
#[derive(Clone, Copy, Debug)]
pub struct SampleContext<'a> {
    pub context: &'a [u32],
    pub prompt_len: usize,
}

impl SampleContext<'_> {
    /// No context: every processor is then a no-op.
    pub const EMPTY: SampleContext<'static> = SampleContext { context: &[], prompt_len: 0 };

    fn generated(&self) -> &[u32] {
        &self.context[self.prompt_len.min(self.context.len())..]
    }
}

/// Apply the request's processors to `candidates` (token, logit), then restore
/// the descending order the rest of the sampler relies on.
pub fn apply_processors(candidates: &mut [(u32, f32)], params: &SampleParams, ctx: &SampleContext<'_>) {
    if !params.has_processors() || ctx.context.is_empty() {
        return;
    }
    if params.no_repeat.active() {
        let banned = params.no_repeat.banned(ctx.generated());
        for (t, l) in candidates.iter_mut() {
            if banned.contains(t) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    if params.repetition_penalty != 1.0 && params.repetition_penalty > 0.0 {
        let seen: std::collections::HashSet<u32> = ctx.context.iter().copied().collect();
        for (t, l) in candidates.iter_mut() {
            if seen.contains(t) {
                *l = if *l > 0.0 { *l / params.repetition_penalty } else { *l * params.repetition_penalty };
            }
        }
    }
    if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &t in ctx.generated() {
            *counts.entry(t).or_insert(0) += 1;
        }
        for (t, l) in candidates.iter_mut() {
            if let Some(&n) = counts.get(t) {
                *l -= params.frequency_penalty * n as f32 + params.presence_penalty;
            }
        }
    }
    if params.dry.active() {
        let penalties = dry_penalties(ctx.context, &params.dry);
        for (t, l) in candidates.iter_mut() {
            if let Some(p) = penalties.get(t) {
                *l -= p;
            }
        }
    }
    // Stable, so equal logits keep the decoder's order; NaN sorts last.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan())));
}

/// The DRY penalty of every token that would continue a repeat in `context`
/// (llama.cpp's algorithm: a Z-array over the reversed context gives, for each
/// earlier position, how many tokens before it equal the tokens before the end;
/// the token that followed that earlier position is the one penalised, by the
/// longest repeat it would continue).
fn dry_penalties(context: &[u32], dry: &DryParams) -> HashMap<u32, f32> {
    let mut out = HashMap::new();
    let n = context.len();
    let allowed = dry.allowed_length as usize;
    if n < 2 {
        return out;
    }
    let rev: Vec<u32> = context.iter().rev().copied().collect();
    // A repeat is not matched across a sequence breaker: only the tokens after
    // the most recent one (counting from the end) can be part of it.
    let rep_limit = rev.iter().position(|t| dry.breakers().contains(t)).unwrap_or(n);
    if rep_limit < allowed.max(1) {
        return out;
    }
    let z = z_array(&rev);
    // base^e must stay finite in f32.
    let max_exponent = if dry.base > 1.0 { (f32::MAX.ln() / dry.base.ln()).floor() as i64 } else { i64::MAX };
    for i in 1..n {
        let len = z[i].min(rep_limit);
        if len == 0 || len < allowed {
            continue;
        }
        // The earlier occurrence ended at original index n-1-i; the token after it is rev[i-1].
        let exponent = ((len - allowed) as i64).min(max_exponent);
        let penalty = dry.multiplier * dry.base.powi(exponent as i32);
        let slot = out.entry(rev[i - 1]).or_insert(0.0f32);
        *slot = slot.max(penalty);
    }
    out
}

/// `z[i]` is the length of the longest common prefix of `s` and `s[i..]` (`z[0] = s.len()`).
fn z_array(s: &[u32]) -> Vec<usize> {
    let n = s.len();
    let mut z = vec![0usize; n];
    if n == 0 {
        return z;
    }
    z[0] = n;
    let (mut l, mut r) = (0usize, 0usize);
    for i in 1..n {
        if i < r {
            z[i] = (r - i).min(z[i - l]);
        }
        while i + z[i] < n && s[z[i]] == s[i + z[i]] {
            z[i] += 1;
        }
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SampleParams {
        SampleParams::greedy()
    }

    fn ctx(context: &[u32], prompt_len: usize) -> SampleContext<'_> {
        SampleContext { context, prompt_len }
    }

    /// A request that asks for no processor is untouched: the candidate list and its
    /// order are exactly what the decoder returned.
    #[test]
    fn without_processors_the_candidates_are_untouched() {
        let mut c = vec![(5u32, 3.0f32), (6, 2.9), (7, 1.0)];
        let before = c.clone();
        apply_processors(&mut c, &params(), &ctx(&[5, 5, 6, 7], 1));
        assert_eq!(c, before);
        assert!(!params().has_processors());
    }

    /// OpenAI: presence subtracts once for a token that appeared in the generated text,
    /// frequency once per occurrence; the prompt does not count.
    #[test]
    fn presence_and_frequency_penalties_count_generated_tokens_only() {
        let p = SampleParams { presence_penalty: 0.5, frequency_penalty: 0.25, ..params() };
        let mut c = vec![(5u32, 3.0f32), (6, 2.9), (7, 2.0)];
        // context = prompt [7, 7] + generated [5, 5, 6]
        apply_processors(&mut c, &p, &ctx(&[7, 7, 5, 5, 6], 2));
        let logit = |t: u32| c.iter().find(|x| x.0 == t).unwrap().1;
        assert!((logit(5) - (3.0 - 0.5 - 0.5)).abs() < 1e-6, "twice generated: presence + 2 x frequency");
        assert!((logit(6) - (2.9 - 0.5 - 0.25)).abs() < 1e-6);
        assert_eq!(logit(7), 2.0, "a prompt-only token is not penalised");
        assert!(c.windows(2).all(|w| w[0].1 >= w[1].1), "the list stays sorted descending");
        assert_eq!(c[0].0, 6, "6 (2.15) now leads 5 and 7 (2.0)");
    }

    /// vLLM/HF: a seen token's positive logit is divided, its negative one multiplied, prompt included.
    #[test]
    fn repetition_penalty_scales_seen_tokens_in_the_whole_context() {
        let p = SampleParams { repetition_penalty: 2.0, ..params() };
        let mut c = vec![(1u32, 4.0f32), (2, 3.0), (3, -1.0)];
        apply_processors(&mut c, &p, &ctx(&[1, 3], 1));
        let logit = |t: u32| c.iter().find(|x| x.0 == t).unwrap().1;
        assert_eq!((logit(1), logit(2), logit(3)), (2.0, 3.0, -2.0));
    }

    /// llama.cpp DRY: in `1 2 3 9 1 2` the suffix `1 2` already occurred before `3`, so `3` would
    /// repeat a 2-token sequence: it is penalised by multiplier * base^(2 - allowed_length).
    #[test]
    fn dry_penalises_the_token_that_would_extend_a_repeat() {
        let context = [1u32, 2, 3, 9, 1, 2];
        let p = SampleParams { dry: DryParams::new(2.0, 2.0, 2, &[]), ..params() };
        let mut c = vec![(3u32, 5.0f32), (4, 4.0)];
        apply_processors(&mut c, &p, &ctx(&context, 0));
        assert_eq!(c, vec![(4, 4.0), (3, 3.0)], "3 lost multiplier * base^0 = 2.0 and fell behind 4");
        // A longer repeat grows geometrically: `1 2 3 9 1 2 3 9 1 2` repeats `1 2 3 9 1 2` (6 tokens) before 3.
        let long = [1u32, 2, 3, 9, 1, 2, 3, 9, 1, 2];
        let mut c = vec![(3u32, 100.0f32)];
        apply_processors(&mut c, &p, &ctx(&long, 0));
        assert!((c[0].1 - (100.0 - 2.0 * 2.0f32.powi(6 - 2))).abs() < 1e-3, "a repeat of 6 tokens: base^(6-2)");
    }

    /// Below `allowed_length` nothing is penalised.
    #[test]
    fn dry_leaves_short_repeats_alone() {
        let p = SampleParams { dry: DryParams::new(5.0, 2.0, 3, &[]), ..params() };
        let mut c = vec![(3u32, 5.0f32)];
        apply_processors(&mut c, &p, &ctx(&[1, 2, 3, 9, 1, 2], 0));
        assert_eq!(c, vec![(3, 5.0)], "a 2-token repeat is under allowed_length 3");
    }

    /// A sequence breaker ends a repeat: the match is not carried across it, and a breaker as the
    /// last token means no penalty at all.
    #[test]
    fn dry_does_not_match_across_a_sequence_breaker() {
        let p = SampleParams { dry: DryParams::new(5.0, 2.0, 2, &[9]), ..params() };
        // `1 2 3 .. 1 2`: the repeat `1 2` is intact (no breaker inside it).
        let mut c = vec![(3u32, 5.0f32)];
        apply_processors(&mut c, &p, &ctx(&[1, 2, 3, 1, 2], 0));
        assert_eq!(c, vec![(3, 0.0)]);
        // `7 9 1 2 .. 7 9 1 2`: the suffix is `9 1 2` but 9 is a breaker, so only `1 2` can match.
        let mut c = vec![(3u32, 5.0f32)];
        apply_processors(&mut c, &p, &ctx(&[7, 9, 1, 2, 3, 7, 9, 1, 2], 0));
        assert_eq!(c, vec![(3, 5.0 - 5.0)], "the 4-token repeat is cut to the 2 tokens after the breaker");
        // The last token is a breaker: nothing to repeat.
        let mut c = vec![(3u32, 5.0f32)];
        apply_processors(&mut c, &p, &ctx(&[1, 2, 3, 1, 2, 9], 0));
        assert_eq!(c, vec![(3, 5.0)]);
    }

    /// The budget counts tokens inside the block, forces the closing sequence token by token when it
    /// runs out, and then lets the model go on.
    #[test]
    fn a_thinking_budget_forces_the_closing_sequence_after_its_tokens() {
        let (open, close_tag, nl) = (100u32, 101u32, 102u32);
        let mut s = ThinkState::new(ThinkingBudget::new(3, open, &[close_tag, nl], true));
        assert_eq!(s.forced_next(), None);
        for t in [1, 2, 3] {
            assert_eq!(s.forced_next(), None);
            s.accept(t);
        }
        assert_eq!(s.forced_next(), Some(close_tag), "three tokens inside the block: the tag is next");
        s.accept(close_tag);
        assert_eq!(s.forced_next(), Some(nl));
        s.accept(nl);
        assert_eq!(s.forced_next(), None, "after the forced sequence the model answers");
        assert!(!s.watching());
    }

    /// A lead-in is forced before the closing sequence when the budget runs out (inside the block, so
    /// the model reads it as its own last thought); a model that closes in time never sees it.
    #[test]
    fn a_thinking_budget_forces_its_lead_in_then_the_closing_sequence() {
        let (open, close_tag, nl, a, b) = (100u32, 101u32, 102u32, 200u32, 201u32);
        let budget = ThinkingBudget::new(1, open, &[close_tag, nl], true).with_lead_in(&[a, b]);
        let mut s = ThinkState::new(budget);
        s.accept(7);
        let mut forced = Vec::new();
        while let Some(t) = s.forced_next() {
            forced.push(t);
            s.accept(t);
        }
        assert_eq!(forced, vec![a, b, close_tag, nl], "lead-in, then the tag and its separator");
        assert!(!s.watching());
        let mut early = ThinkState::new(budget);
        early.accept(close_tag);
        assert_eq!(early.forced_next(), None, "a block the model closes itself gets no lead-in");
    }

    /// A model that opens its own block is counted from the tag; one that closes in time is left alone.
    #[test]
    fn a_thinking_budget_starts_at_the_opening_tag_and_yields_to_a_model_that_closes_in_time() {
        let mut s = ThinkState::new(ThinkingBudget::new(2, 100, &[101], false));
        for t in [7, 8, 9, 10] {
            s.accept(t); // before the block opens nothing is counted
            assert_eq!(s.forced_next(), None);
        }
        s.accept(100);
        s.accept(5);
        s.accept(101); // closed after one token, under the budget of two
        assert!(!s.watching());
        assert_eq!(s.forced_next(), None);
        // A budget of zero closes the block as soon as it opens.
        let mut z = ThinkState::new(ThinkingBudget::new(0, 100, &[101], false));
        z.accept(100);
        assert_eq!(z.forced_next(), Some(101));
        let at_start = ThinkState::new(ThinkingBudget::new(0, 100, &[101], true));
        assert_eq!(at_start.forced_next(), Some(101), "a prompt that ends inside the block is already out of budget");
    }

    #[test]
    fn the_z_array_matches_its_definition() {
        let s = [1u32, 2, 1, 2, 1, 3];
        let z = z_array(&s);
        for i in 0..s.len() {
            let want = (0..s.len() - i).take_while(|&k| s[k] == s[i + k]).count();
            assert_eq!(z[i], want, "z[{i}]");
        }
    }
}
