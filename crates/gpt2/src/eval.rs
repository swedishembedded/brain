// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Evaluating a GPT checkpoint: validation perplexity, and exact-match task
//! accuracy for `LHS=RHS` datasets (calculator, reverser, word arithmetic):
//! greedily decode the right-hand side from the `LHS=` prompt on a held-out
//! tail and check exact string equality - whether the model learned the
//! rule, not just the token statistics.
//!
//! Swedish Embedded AB implements training and honest evaluation of small
//! language models for its clients. If your team needs expertise in
//! measuring what a model actually learned then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::Path;

use data::binio::Meta;
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use crate::Gpt;

/// Block size recorded in a checkpoint header.
fn block_of(weights: &str) -> u32 {
    crate::GptConfig::from_json(&checkpoint::read_config(weights)).block_size
}

/// Validation perplexity = `exp(mean CE)` over `batches` random val windows.
pub fn val_perplexity(weights: &str, data_dir: &Path, batches: usize, seed: u64) -> std::io::Result<f32> {
    let block = block_of(weights);
    let model = Gpt::load(weights, 16, block);
    let val = data::binio::read_tokens_u32(&data_dir.join("val"))?;
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
pub fn exact_match(
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
        let prompt_ids: Vec<u32> = tok.encode(&prompt);
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
        let ch = tok.decode(&[next]);
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
    use super::*;

    /// Eight characters, so a model that knows nothing has perplexity 8.
    const ITOS: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'f', '=', '\n'];
    const BLOCK: u32 = 8;
    /// Logit margin the favoured token gets over every other one.
    const MARGIN: f32 = 10.0;

    fn config() -> crate::GptConfig {
        crate::GptConfig { vocab: 8, block_size: BLOCK, n_layers: 1, d_model: 8, n_heads: 2, d_ff: 16 }
    }

    /// A model whose output ignores its input. The final layer norm has gain
    /// zero, so it returns its bias whatever the residual stream holds, and the
    /// logits are `tok.weight * bias` (the head is tied to the embedding). With
    /// `tok.weight` the identity and a bias of `MARGIN` on token `favoured`,
    /// that token wins by exactly `MARGIN` everywhere; with no favoured token
    /// all logits are zero (uniform). Every other weight is zero.
    fn constant_model(dir: &Path, favoured: Option<usize>) -> String {
        let cfg = config();
        let d = cfg.d_model as usize;
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let mut v = vec![0.0f32; n];
                match name.as_str() {
                    "ln.bias" => {
                        if let Some(j) = favoured {
                            v[j] = MARGIN;
                        }
                    }
                    "tok.weight" => (0..d).for_each(|i| v[i * d + i] = 1.0),
                    _ => {}
                }
                (name, vec![n as u64], v)
            })
            .collect();
        let path = dir.join("model.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        path.to_str().unwrap().to_string()
    }

    /// A dataset directory: `meta.json`, a `val` split and `input.txt`.
    fn dataset(dir: &Path, val: &[u32], text: &str) {
        let meta = Meta { vocab_size: ITOS.len(), itos: ITOS.to_vec() };
        std::fs::write(dir.join("meta.json"), meta.to_json()).unwrap();
        data::binio::write_u32_bin(&dir.join("val.u32.bin"), val).unwrap();
        std::fs::write(dir.join("input.txt"), text).unwrap();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-eval-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Ten tail lines (the last tenth of a hundred) with the given text each.
    fn input_with_tail(tail_line: &str) -> String {
        let mut text = String::new();
        for _ in 0..90 {
            text.push_str("ab=c\n");
        }
        for _ in 0..10 {
            text.push_str(tail_line);
            text.push('\n');
        }
        text
    }

    #[test]
    fn argmax_takes_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[0.0, 2.0, 1.0]), 1);
        assert_eq!(argmax(&[3.0, 3.0, 1.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, -5.0]), 1);
    }

    /// Perplexity is `exp(mean cross-entropy)`: a model that knows nothing
    /// scores the vocabulary size, one that is sure and right scores near 1,
    /// one that is sure and wrong scores `exp(MARGIN)` plus the other tokens.
    #[test]
    fn perplexity_is_the_exponential_of_the_cross_entropy() {
        let dir = scratch("ppl");
        let val_all_a = vec![0u32; 200];
        dataset(&dir, &val_all_a, "");

        let uniform = constant_model(&dir, None);
        let ppl = val_perplexity(&uniform, &dir, 2, 7).unwrap();
        assert!((ppl - 8.0).abs() < 1e-3, "uniform model over 8 tokens: {ppl}");

        let right = constant_model(&dir, Some(0));
        let ppl = val_perplexity(&right, &dir, 2, 7).unwrap();
        let p = MARGIN.exp() / (MARGIN.exp() + 7.0);
        assert!((ppl - 1.0 / p).abs() < 1e-3, "sure and right: {ppl} vs {}", 1.0 / p);

        let wrong = constant_model(&dir, Some(1));
        let ppl = val_perplexity(&wrong, &dir, 2, 7).unwrap();
        let want = MARGIN.exp() + 7.0;
        assert!((ppl / want - 1.0).abs() < 1e-3, "sure and wrong: {ppl} vs {want}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Greedy decoding stops at the newline token and at `max_new`.
    #[test]
    fn greedy_decoding_stops_at_a_newline_or_the_length_limit() {
        let dir = scratch("greedy");
        let tok = CharTokenizer::from_itos(ITOS.to_vec());
        let prompt = tok.encode("ab=");

        let emits_c = Gpt::load(&constant_model(&dir, Some(2)), 1, BLOCK);
        assert_eq!(greedy_until_newline(&emits_c, &prompt, 4, &tok), "cccc");
        assert_eq!(greedy_until_newline(&emits_c, &prompt, 0, &tok), "");

        let emits_newline = Gpt::load(&constant_model(&dir, Some(7)), 1, BLOCK);
        assert_eq!(greedy_until_newline(&emits_newline, &prompt, 4, &tok), "");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A model that always answers with the end of line is right on exactly the
    /// lines whose right-hand side is empty.
    #[test]
    fn exact_match_counts_the_lines_decoded_to_the_right_hand_side() {
        let dir = scratch("exact");
        let model = constant_model(&dir, Some(7));

        dataset(&dir, &[0; 20], &input_with_tail("ab="));
        assert_eq!(exact_match(&model, &dir, 5, 3).unwrap(), (1.0, 5));

        dataset(&dir, &[0; 20], &input_with_tail("ab=c"));
        assert_eq!(exact_match(&model, &dir, 5, 3).unwrap(), (0.0, 5));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// With no `LHS=RHS` line in the held-out tail there is nothing to score.
    #[test]
    fn exact_match_without_a_candidate_line_is_zero_of_zero() {
        let dir = scratch("nocand");
        let model = constant_model(&dir, Some(7));
        dataset(&dir, &[0; 20], &input_with_tail("abc"));
        assert_eq!(exact_match(&model, &dir, 5, 3).unwrap(), (0.0, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A prompt that does not fit the block leaves no room to decode and is
    /// skipped, not counted as wrong.
    #[test]
    fn exact_match_skips_a_prompt_that_fills_the_block() {
        let dir = scratch("overlong");
        let model = constant_model(&dir, Some(7));
        // The prompt `abcdefa=` is exactly the block (8 tokens): nothing is left to decode into.
        dataset(&dir, &[0; 20], &input_with_tail("abcdefa="));
        assert_eq!(exact_match(&model, &dir, 5, 3).unwrap(), (0.0, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
