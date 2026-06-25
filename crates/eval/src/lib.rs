// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Unified evaluation harness — the same metrics across models, so different
//! architectures can be compared on identical data (README §3 discipline: hold
//! the input distribution fixed, separate the metric from how you test).
//!
//! - [`gpt_val_perplexity`] — `exp(mean next-token CE)` on the val split. The
//!   architecture-agnostic comparison metric.
//! - [`gpt_exact_match`] — task accuracy for `LHS=RHS` datasets (calculator /
//!   reverser / wordcalc): greedily decode the RHS from the `LHS=` prompt on a
//!   held-out tail and check exact string equality. This is the honest
//!   "did it actually learn the rule" number, not perplexity.

pub mod detection;

use std::path::Path;

use data::binio::Meta;
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use gpt::Gpt;

/// Block size recorded in a checkpoint header.
fn block_of(weights: &str) -> u32 {
    let c = checkpoint::load(weights);
    gpt::GptConfig::from_json(&c.header["config"]).block_size
}

/// Validation perplexity = `exp(mean CE)` over `batches` random val windows.
pub fn gpt_val_perplexity(weights: &str, data_dir: &Path, batches: usize, seed: u64) -> std::io::Result<f32> {
    let block = block_of(weights);
    let model = Gpt::load(weights, 16, block);
    let val = data::binio::read_u16_bin(&data_dir.join("val.bin"))?;
    let cfg = BatchConfig { batch_size: 16, block_size: block as usize, ..Default::default() };
    let ds = TokenDataset::new(val, &cfg);
    let mut rng = Rng::new(seed);
    let mut total = 0.0f32;
    for _ in 0..batches.max(1) {
        let (x, y) = ds.get_batch(&cfg, &mut rng);
        let y_u32: Vec<u32> = y.iter().map(|&v| v as u32).collect();
        model.set_batch(&x, &y_u32);
        total += model.forward();
    }
    Ok((total / batches.max(1) as f32).exp())
}

/// Exact-match accuracy on a held-out tail of an `LHS=RHS` char dataset.
/// Returns `(accuracy, n_samples)`.
pub fn gpt_exact_match(
    weights: &str,
    data_dir: &Path,
    n_samples: usize,
    seed: u64,
) -> std::io::Result<(f32, usize)> {
    let meta = Meta::from_json(&std::fs::read_to_string(data_dir.join("meta.json"))?)
        .map_err(std::io::Error::other)?;
    let tok = CharTokenizer::from_itos(meta.itos.clone());
    let block = block_of(weights);
    let model = Gpt::load(weights, 1, block);

    // Held-out tail (last 10%) of the source text; lines with exactly one '='.
    let text = std::fs::read_to_string(data_dir.join("input.txt"))?;
    let lines: Vec<&str> = text.lines().collect();
    let tail_start = lines.len() * 9 / 10;
    let candidates: Vec<&str> = lines[tail_start..]
        .iter()
        .copied()
        .filter(|l| l.matches('=').count() == 1 && !l.is_empty())
        .collect();
    if candidates.is_empty() {
        return Ok((0.0, 0));
    }

    let mut rng = Rng::new(seed);
    let mut correct = 0usize;
    let mut n = 0usize;
    for _ in 0..n_samples {
        let line = candidates[rng.gen_range_inclusive(0, candidates.len() as i64 - 1) as usize];
        let (lhs, rhs) = line.split_once('=').unwrap();
        let prompt = format!("{lhs}=");
        let prompt_ids: Vec<u32> = tok.encode(&prompt).iter().map(|&t| t as u32).collect();
        if prompt_ids.is_empty() || prompt_ids.len() >= block as usize {
            continue;
        }
        let max_new = rhs.chars().count() + 2;
        let gen = greedy_until_newline(&model, &prompt_ids, max_new, &tok);
        if gen == rhs {
            correct += 1;
        }
        n += 1;
    }
    let acc = if n > 0 { correct as f32 / n as f32 } else { 0.0 };
    Ok((acc, n))
}

/// Greedy-decode characters until a newline (exclusive) or `max_new` reached.
fn greedy_until_newline(model: &Gpt, prompt: &[u32], max_new: usize, tok: &CharTokenizer) -> String {
    let block = model.cfg.block_size as usize;
    let mut ctx = prompt.to_vec();
    let mut out = String::new();
    for _ in 0..max_new {
        let window: Vec<u32> = if ctx.len() > block { ctx[ctx.len() - block..].to_vec() } else { ctx.clone() };
        let logits = model.logits_all(&window);
        let vocab = model.cfg.vocab as usize;
        let last = &logits[logits.len() - vocab..];
        let next = argmax(last) as u32;
        let ch = tok.decode(&[next as u16]);
        if ch == "\n" {
            break;
        }
        out.push_str(&ch);
        ctx.push(next);
    }
    out
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

#[cfg(test)]
mod tests {
    #[test]
    fn perplexity_definition_sanity() {
        // exp(ln V) = V — sanity for the perplexity definition used above.
        let v = 65f32;
        assert!(((v.ln()).exp() - v).abs() < 1e-2);
    }
}
