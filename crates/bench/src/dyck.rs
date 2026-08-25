// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dyck-k — **hierarchical state tracking** over a balanced-bracket language.
//!
//! Each sequence is a well-formed word of the Dyck-`k` language (`k` distinct
//! bracket pairs, all correctly nested and balanced), terminated by a newline:
//!
//! ```text
//!   ( [ ] { } ) [ ( ) ] NL          (one balanced Dyck-2 word)
//! ```
//!
//! The task is **next-token prediction over the word**, scored only at the
//! positions whose target is a **close bracket**. Choosing the right close
//! bracket is determined entirely by the current top of the bracket stack: it is
//! the canonical example of a **context-free** language that needs a *stack* (an
//! unbounded counter per bracket type), so it is a sharp test of hierarchical
//! state — a model must track arbitrary nesting depth, not just local n-grams.
//! Dyck membership/prediction is the standard formal-language probe for the
//! hierarchical structure transformers can only approximate to bounded depth.
//!
//! ## How it reuses the engine
//! - **Tokens.** `NL` (sequence end), then `k` open-bracket tokens and `k`
//!   matching close-bracket tokens. Written as a char dataset (`NL`→`'\n'`,
//!   brackets → Private-Use-Area chars so any `k` fits) so `gpt2::train` loads it
//!   unchanged.
//! - **Masking.** No `=` separator: the whole word is supervised next-token
//!   (`mask_before=None`), with windows **line-aligned** (`align_to_lines`) so
//!   each training window is exactly one balanced word — predicting opens is easy
//!   filler, predicting closes is the hierarchical signal the model must learn.
//! - **Scoring.** Next-token accuracy at **close-bracket target positions only**
//!   of each held-out word — at each close bracket, check the model's argmax (over
//!   the previous position's logits) equals the bracket the stack demands. Chance
//!   is `1 / k` (the model can tell from context that a close is due; it must pick
//!   the right *type*).
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! `k=3` bracket types (**chance ≈ 0.333**), `max_depth=4`, fixed word length 24,
//! 6000 words, 1000 steps, 2-layer / d_model-96 / 4-head GPT. **Measured
//! close-bracket accuracy ≈ 0.99 across seeds** (seeds 1337 → 0.9942, 42 →
//! 0.9992, train_ce ≈ 0.95-0.97), far above the 1/k chance and clear of the
//! **0.70** threshold, in minutes on CPU (the `tests/dyck.rs` guard drops to 600
//! steps / 4000 words / d_model-64 to run in a fraction of that, still ~0.99). Difficulty grows
//! with `k` (more bracket types to disambiguate) and `max_depth` (deeper stacks):
//! the calibrated `k=3`, `max_depth=4` is the fast, clearly-learnable sweet spot;
//! raising both is the intended hierarchical-state stress knob.

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{associative_recall, Metrics};
use crate::model::{argmax, DecoderLm, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// First bracket-token id. Opens are `[OPEN0, OPEN0+k)`; the close matching
/// open `OPEN0+i` is `OPEN0+k+i`.
const OPEN0: u16 = 1;

/// Dyck-k configuration. Defaults are calibrated to be clearly learnable by a
/// 2-layer / d_model-96 GPT within a few hundred CPU steps (see [`Dyck::default`]).
#[derive(Clone, Debug)]
pub struct Dyck {
    /// Number of distinct bracket pairs. Chance close-prediction is `1 / k`.
    pub k: usize,
    /// Maximum nesting depth (stack ceiling). A hierarchical-state difficulty knob.
    pub max_depth: usize,
    /// Fixed word length in bracket tokens (must be even; excludes the `NL`). A
    /// length/nesting difficulty knob.
    pub length: usize,
    /// Number of words in the generated corpus.
    pub n_sequences: usize,
    /// Training steps.
    pub steps: u32,
    /// GPT depth / width / heads for the scoring model.
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Words scored for the accuracy metric (drawn from the val split).
    pub eval_sequences: usize,
}

impl Default for Dyck {
    /// Calibrated config: see the module doc comment. Chance ≈ 0.333 (k=3),
    /// measured close-bracket accuracy ≈ 0.99 across seeds, threshold 0.70.
    fn default() -> Self {
        Dyck {
            k: 3,
            max_depth: 4,
            length: 24,
            n_sequences: 6000,
            steps: 1000,
            n_layers: 2,
            d_model: 96,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl Dyck {
    /// Sequence length: the word plus the trailing `NL`.
    fn seq_len(&self) -> usize {
        self.length + 1
    }

    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    /// Vocab: NL, then `k` opens and `k` closes.
    fn vocab(&self) -> usize {
        OPEN0 as usize + 2 * self.k
    }

    fn open_tok(&self, i: usize) -> u16 {
        OPEN0 + i as u16
    }
    fn close_tok(&self, i: usize) -> u16 {
        OPEN0 + self.k as u16 + i as u16
    }
    #[cfg(test)]
    fn is_close(&self, t: u16) -> bool {
        t >= OPEN0 + self.k as u16 && t < OPEN0 + 2 * self.k as u16
    }

    /// Generate one balanced Dyck-k word of exactly `length` tokens, plus the
    /// list of `(index, close_token)` answer positions (every close bracket).
    ///
    /// At each step we either push a fresh random open bracket or pop the stack
    /// top (emitting its matching close), constrained so the word stays valid and
    /// hits exactly `length`: we cannot push past `max_depth`, cannot pop an empty
    /// stack, and must reserve enough remaining slots to drain the stack by the
    /// end.
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, Vec<(usize, u16)>) {
        assert!(self.length.is_multiple_of(2), "Dyck word length must be even");
        let mut seq = Vec::with_capacity(self.seq_len());
        let mut stack: Vec<usize> = Vec::with_capacity(self.max_depth);
        let mut answers = Vec::new();

        for step in 0..self.length {
            let remaining = self.length - step; // slots left including this one
            // We must end empty: each unclosed bracket needs one slot to close.
            // So we may push only if after pushing the stack can still be drained:
            //   (depth + 1) <= remaining - 1   ⇔   depth + 2 <= remaining.
            let can_push = stack.len() < self.max_depth && stack.len() + 2 <= remaining;
            // We must pop when there is no room left to do anything but drain
            // (remaining == stack depth), and cannot pop an empty stack.
            let must_pop = remaining == stack.len();
            let can_pop = !stack.is_empty();

            let push = if must_pop {
                false
            } else if !can_pop {
                true
            } else if !can_push {
                false
            } else {
                // Free choice: bias slightly toward pushing when shallow so words
                // actually nest rather than collapsing to flat ()()() patterns.
                rng.gen_range_inclusive(0, 1) == 0
            };

            if push {
                let b = rng.gen_range_inclusive(0, self.k as i64 - 1) as usize;
                stack.push(b);
                seq.push(self.open_tok(b));
            } else {
                let b = stack.pop().expect("pop on non-empty stack");
                let tok = self.close_tok(b);
                answers.push((seq.len(), tok));
                seq.push(tok);
            }
        }
        debug_assert!(stack.is_empty(), "Dyck word must be balanced");
        seq.push(NL);
        (seq, answers)
    }

    fn build_corpus(&self, seed: u64) -> Vec<u16> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(self.n_sequences * self.seq_len());
        for _ in 0..self.n_sequences {
            let (seq, _) = self.gen_sequence(&mut rng);
            out.extend_from_slice(&seq);
        }
        out
    }

    /// Synthetic `itos`: NL→`'\n'`, brackets → Private-Use-Area chars.
    fn itos(&self) -> Vec<char> {
        let mut itos = vec!['\0'; self.vocab()];
        itos[NL as usize] = '\n';
        for i in 0..2 * self.k {
            itos[OPEN0 as usize + i] = char::from_u32(0xE000 + i as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for Dyck {
    fn name(&self) -> &str {
        "dyck"
    }

    fn description(&self) -> &str {
        "Dyck-k balanced brackets (hierarchical state / next close bracket)"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let corpus = self.build_corpus(seed);
        let split_seqs = (self.n_sequences * 9) / 10;
        let split = split_seqs * self.seq_len();
        binio::write_u16_bin(&dir.join("train.bin"), &corpus[..split])?;
        binio::write_u16_bin(&dir.join("val.bin"), &corpus[split..])?;
        let meta = Meta { vocab_size: self.vocab(), itos: self.itos() };
        std::fs::write(dir.join("meta.json"), meta.to_json())?;
        Ok(())
    }


    fn threshold(&self) -> f32 {
        // Far above 1/k chance (≈0.333) yet below the measured ~0.85+ floor across
        // seeds, with margin for run-to-run variance on the CPU backend.
        0.70
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["chance", "train_ce"]
    }
    /// Train + score with a specific architecture (any [`DecoderLm`]).
    /// [`Benchmark::evaluate`] calls this with the GPT baseline.
    fn evaluate_with(&self, lm: &dyn DecoderLm, dir: &Path, seed: u64) -> std::io::Result<Metrics> {
        // ---- TRAIN (architecture-agnostic via DecoderLm) ---------------------
        // The whole balanced word is supervised next-token (no `=` mask); windows
        // are line-aligned so each is exactly one word.
        let block = self.block_size();
        let train_cfg = TrainConfig {
            steps: self.steps,
            batch_size: 32,
            lr: 3e-3,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: true,
            seed,
        };
        let out = dir.join("dyck.safetensors");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: close-bracket accuracy on held-out (val) words -----------
        let scorer = lm.load_scorer(&out, block);
        let val = binio::read_u16_bin(&dir.join("val.bin"))?;
        let seq_len = self.seq_len();
        let n_val = val.len() / seq_len;
        let to_score = self.eval_sequences.min(n_val);

        // Replay the rng to recover the val tail's close-bracket positions.
        let mut rng = Rng::new(seed);
        let train_seqs = (self.n_sequences * 9) / 10;
        for _ in 0..train_seqs {
            self.gen_sequence(&mut rng);
        }

        let mut predicted = Vec::new();
        let mut expected = Vec::new();
        for s in 0..to_score {
            let (seq, answers) = self.gen_sequence(&mut rng);
            debug_assert_eq!(&seq[..], &val[s * seq_len..(s + 1) * seq_len]);
            let toks: Vec<u32> = seq.iter().map(|&t| t as u32).collect();
            let logits = scorer.logits_all(&toks);
            let v = scorer.vocab();
            for &(ans_pos, ans_tok) in &answers {
                // The first token can never be a close, so ans_pos >= 1 always.
                let row = &logits[(ans_pos - 1) * v..ans_pos * v];
                predicted.push(argmax(row) as u32);
                expected.push(ans_tok as u32);
            }
        }

        let acc = associative_recall(&predicted, &expected);
        let chance = 1.0 / self.k.max(1) as f32;
        Ok(Metrics::new(acc)
            .with("accuracy", acc)
            .with("chance", chance)
            .with("train_ce", final_loss)
            .with("init_ce", init_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generated word must be balanced, exactly `length` long, within
    /// `max_depth`, and its recorded answers must be its close brackets.
    #[test]
    fn word_is_balanced_and_within_depth() {
        let m = Dyck { k: 3, max_depth: 4, length: 16, ..Dyck::default() };
        let mut rng = Rng::new(1);
        for _ in 0..200 {
            let (seq, answers) = m.gen_sequence(&mut rng);
            assert_eq!(seq.len(), m.seq_len());
            assert_eq!(*seq.last().unwrap(), NL);

            let mut stack: Vec<u16> = Vec::new();
            let mut depth_max = 0usize;
            let mut closes = 0usize;
            for &t in &seq[..m.length] {
                if m.is_close(t) {
                    closes += 1;
                    let want_open = t - m.k as u16; // matching open id
                    assert_eq!(stack.pop(), Some(want_open), "close must match stack top");
                } else {
                    stack.push(t);
                    depth_max = depth_max.max(stack.len());
                }
            }
            assert!(stack.is_empty(), "word must be balanced");
            assert!(depth_max <= m.max_depth, "depth {depth_max} exceeds max");
            // Half the bracket tokens are closes.
            assert_eq!(closes, m.length / 2);
            assert_eq!(answers.len(), closes);
            for &(pos, tok) in &answers {
                assert_eq!(seq[pos], tok);
                assert!(m.is_close(tok));
            }
        }
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = Dyck { n_sequences: 100, ..Dyck::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
