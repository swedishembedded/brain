// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Greedy generation with absent-slot masking and the loop guard.
//!
//! The decoder is run one token at a time from the prompt (`<s>` and a
//! language tag) until it writes `</s>`, the token budget is spent, the loop
//! guard objects to a line, or the requested sections are complete. The
//! reference tokens of proposals that do not exist are never chosen.

use super::{CrossMemory, Decoder, Positions};
use crate::config::MirageConfig;
use crate::describe::lines;
use crate::tokenizer::Tokenizer;

/// Why generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The decoder wrote `</s>`.
    Eos,
    /// A line arrived out of section order or repeated a key.
    Guard,
    /// The next line belongs to a section that was not requested.
    Sections,
    /// The token budget was spent.
    MaxTokens,
}

/// A finished generation.
#[derive(Debug, Clone)]
pub struct Generation {
    /// Generated token ids, ending with `</s>` when the decoder wrote it.
    pub tokens: Vec<u32>,
    /// The decoded text of those tokens.
    pub text: String,
    /// Why it ended.
    pub stop: Stop,
}

/// What to generate.
pub struct GenerateParams<'a> {
    /// Prompt token ids.
    pub prompt: &'a [u32],
    /// Token ids that must never be chosen.
    pub banned: &'a [u32],
    /// Most tokens to generate.
    pub max_tokens: usize,
    /// Highest line rank to produce (`-1`: stop once the think block closes).
    pub max_rank: i32,
    /// Object reference tokens (text reference tokens follow them).
    pub object_slots: usize,
    /// Apply the loop guard (stop on out-of-order or repeated lines and on
    /// sections that were not requested). Off, only `</s>` and the budget stop.
    pub guard: bool,
}

/// Decode tokens to text, writing a reference token as its slot number.
pub fn render(tok: &Tokenizer, tokens: &[u32], object_slots: usize) -> String {
    let vocab = tok.vocab_size() as u32;
    let mut out = String::new();
    let mut run = Vec::new();
    let flush = |run: &mut Vec<u32>, out: &mut String| {
        out.push_str(&tok.decode(run).unwrap_or_default());
        run.clear();
    };
    for &t in tokens {
        if t < vocab {
            run.push(t);
        } else {
            flush(&mut run, &mut out);
            let k = (t - vocab) as usize;
            out.push_str(&(if k < object_slots { k } else { k - object_slots }).to_string());
        }
    }
    flush(&mut run, &mut out);
    out
}

/// Reference tokens of the proposal slots that are absent.
pub fn absent_references(cfg: &MirageConfig, objects: usize, text: usize) -> Vec<u32> {
    let base = (cfg.text + crate::config::LANGUAGE_TAGS) as u32;
    let objects_end = base + cfg.detections as u32;
    (base + objects as u32..objects_end).chain(objects_end + text as u32..objects_end + cfg.regions as u32).collect()
}

pub(crate) fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

/// Whether the text so far gives a reason to stop.
pub(crate) fn guard(text: &str, max_rank: i32) -> Option<Stop> {
    let (_, after) = text.split_once("</think>")?;
    if max_rank < 0 {
        return Some(Stop::Sections);
    }
    let mut parts: Vec<&str> = after.split('\n').collect();
    let partial = parts.pop().unwrap_or("");
    let (mut last, mut seen) = (i32::MIN, std::collections::BTreeSet::new());
    for line in parts.into_iter().filter(|l| !l.is_empty()) {
        let (rank, key) = (lines::rank(line), lines::key(line));
        if rank < 0 || rank < last || !seen.insert(key) {
            return Some(Stop::Guard);
        }
        last = rank;
        if rank > max_rank {
            return Some(Stop::Sections);
        }
    }
    if !partial.is_empty() {
        // An entity line is an object or a region only once its `*` slot appears (or does not).
        let entity = partial.starts_with('E');
        let rank = if entity { 1 } else { lines::rank(partial) };
        if rank < 0 || (rank < last && !(entity && last == 2)) {
            return Some(Stop::Guard);
        }
        if rank > max_rank {
            return Some(Stop::Sections);
        }
    }
    None
}

/// Greedy generation over a prepared cross-attention memory.
pub fn generate(dec: &Decoder, cross: &CrossMemory, tok: &Tokenizer, p: &GenerateParams) -> Generation {
    super::with_decode_threads(|| generate_inner(dec, cross, tok, p))
}

fn generate_inner(dec: &Decoder, cross: &CrossMemory, tok: &Tokenizer, p: &GenerateParams) -> Generation {
    let mut st = dec.new_state();
    let mut logits = dec.forward(&mut st, cross, p.prompt, Positions::Last);
    let mut tokens = Vec::new();
    let eos = tok.eos_id();
    let stop = loop {
        if tokens.len() >= p.max_tokens {
            break Stop::MaxTokens;
        }
        for &b in p.banned {
            logits[b as usize] = f32::NEG_INFINITY;
        }
        let next = argmax(&logits);
        tokens.push(next);
        if next == eos {
            break Stop::Eos;
        }
        let text = render(tok, &tokens, p.object_slots);
        if let Some(stop) = p.guard.then(|| guard(&text, p.max_rank)).flatten() {
            break stop;
        }
        logits = dec.forward(&mut st, cross, &[next], Positions::Last);
    };
    let text = render(tok, &tokens, p.object_slots);
    Generation { tokens, text, stop }
}
