// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity — **running-parity state tracking** over a random bit string.
//!
//! Each sequence is a fixed-length string of random bits followed by the
//! **running parity** (cumulative XOR) at every position:
//!
//! ```text
//!   b1 b2 b3 ... bn   SEP   p1 p2 p3 ... pn   NL
//!   └──── bits ────┘        └ running parity ┘     pi = b1 ⊕ b2 ⊕ … ⊕ bi
//! ```
//!
//! `pi` is `1` iff an odd number of the first `i` bits are set. Predicting `pi`
//! from `p(i-1)` and `bi` requires the model to *carry a single bit of state*
//! along the answer region: parity is the textbook example of a regular language
//! that a fixed-depth transformer cannot represent exactly (it is not in `AC0`),
//! so it is a sharp **state-tracking** probe — a recurrent model nails it, a
//! transformer must approximate it with a bounded-depth circuit and degrades as
//! `n_bits` grows. We keep `n_bits` small enough that a 2-layer GPT can fit the
//! circuit clearly above the 0.5 coin-flip chance within the CPU budget; pushing
//! `n_bits` up is the difficulty knob that breaks it.
//!
//! ## How it reuses the engine
//! - **Tokens.** `NL` (sequence end), `SEP` (bits→answer separator, the mask
//!   char `'='`), then bit tokens `0`/`1`. Written as a char dataset
//!   (`SEP`→`'='`, `NL`→`'\n'`, bits → `'0'`/`'1'`) so `gpt::train` loads it
//!   unchanged.
//! - **Masking.** Loss masked up to & including `SEP`, per line, line-aligned
//!   (`mask_before='='`, `mask_per_line`, `align_to_lines`) so the gradient
//!   focuses on the running-parity answer region, not the (uniformly random,
//!   unpredictable) input bits.
//! - **Scoring.** Next-token accuracy over **every answer position** of each
//!   held-out sequence — at each `pi`, check the model's argmax (over the
//!   previous position's logits) equals `pi`. Chance is **0.5** (two equiprobable
//!   bit values).
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! `n_bits=8` (**chance = 0.5**), 6000 sequences, 800 steps, 2-layer / d_model-64
//! / 4-head GPT. **Measured accuracy ≈ 1.00 across seeds** (seeds 1337 & 42 both
//! 1.0000, train_ce ≈ 0.07), far above the 0.5 coin flip and clear of the
//! **0.80** threshold, in ~3 min on CPU (the `tests/parity.rs` guard drops to 500
//! steps / 4000 sequences for ~1 min, still 1.0). Difficulty grows with `n_bits`
//! (longer state chains): the calibrated `n_bits=8` is the fast, clearly-learnable
//! sweet spot; `n_bits=20+` is where a small transformer starts to fall back to
//! chance, the intended state-tracking stress knob.

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{associative_recall, Metrics};
use crate::model::{argmax, DecoderLm, GptDecoder, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the bits→answer separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// Token id of bit `0` (maps to `'0'`).
const BIT0: u16 = 2;
/// Token id of bit `1` (maps to `'1'`).
const BIT1: u16 = 3;

/// Parity configuration. Defaults are calibrated to be clearly learnable by a
/// 2-layer / d_model-64 GPT within a few hundred CPU steps (see [`Parity::default`]).
#[derive(Clone, Debug)]
pub struct Parity {
    /// Number of random bits per sequence (also the number of answer positions).
    /// The state-tracking difficulty knob: longer chains stress the model.
    pub n_bits: usize,
    /// Number of sequences in the generated corpus.
    pub n_sequences: usize,
    /// Training steps.
    pub steps: u32,
    /// GPT depth / width / heads for the scoring model.
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Sequences scored for the accuracy metric (drawn from the val split).
    pub eval_sequences: usize,
}

impl Default for Parity {
    /// Calibrated config: see the module doc comment. Chance = 0.5, measured
    /// accuracy ≈ 1.00 across seeds, threshold 0.80.
    fn default() -> Self {
        Parity {
            n_bits: 8,
            n_sequences: 6000,
            steps: 800,
            n_layers: 2,
            d_model: 64,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl Parity {
    /// Sequence length: `n_bits` (input) + 1 (`SEP`) + `n_bits` (parity) + 1 (`NL`).
    fn seq_len(&self) -> usize {
        self.n_bits + 1 + self.n_bits + 1
    }

    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    /// Vocab: NL, SEP, BIT0, BIT1.
    fn vocab(&self) -> usize {
        4
    }

    /// Generate one sequence and its answer positions: `(index, parity_token)`
    /// for each running-parity position in the answer region.
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, Vec<(usize, u16)>) {
        let bits: Vec<u16> = (0..self.n_bits)
            .map(|_| rng.gen_range_inclusive(0, 1) as u16)
            .collect();

        let mut seq = Vec::with_capacity(self.seq_len());
        for &b in &bits {
            seq.push(if b == 1 { BIT1 } else { BIT0 });
        }
        seq.push(SEP);

        // Running parity (cumulative XOR), one answer token per input bit.
        let mut answers = Vec::with_capacity(self.n_bits);
        let mut acc = 0u16;
        for &b in &bits {
            acc ^= b;
            let pos = seq.len();
            let tok = if acc == 1 { BIT1 } else { BIT0 };
            seq.push(tok);
            answers.push((pos, tok));
        }
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

    /// Synthetic `itos`: SEP→`'='`, NL→`'\n'`, bits → `'0'`/`'1'`.
    fn itos(&self) -> Vec<char> {
        let mut itos = vec!['\0'; self.vocab()];
        itos[NL as usize] = '\n';
        itos[SEP as usize] = '=';
        itos[BIT0 as usize] = '0';
        itos[BIT1 as usize] = '1';
        itos
    }
}

impl Benchmark for Parity {
    fn name(&self) -> &str {
        "parity"
    }

    fn description(&self) -> &str {
        "running-parity state tracking over a random bit string"
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

    fn evaluate(&self, dir: &Path, seed: u64) -> std::io::Result<Metrics> {
        self.evaluate_with(&GptDecoder, dir, seed)
    }

    fn threshold(&self) -> f32 {
        // Far above the 0.5 coin-flip chance yet below the measured ~0.93+ floor
        // across seeds, with margin for run-to-run variance on the CPU backend.
        0.80
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["chance", "train_ce"]
    }
}

impl Parity {
    /// Train + score with a specific architecture (any [`DecoderLm`]).
    /// [`Benchmark::evaluate`] calls this with the GPT baseline.
    pub fn evaluate_with(&self, lm: &dyn DecoderLm, dir: &Path, seed: u64) -> std::io::Result<Metrics> {
        // ---- TRAIN (architecture-agnostic via DecoderLm) ---------------------
        let block = self.block_size();
        let train_cfg = TrainConfig {
            steps: self.steps,
            batch_size: 32,
            lr: 3e-3,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            mask_before: Some('='),
            mask_per_line: true,
            align_to_lines: true,
            seed,
        };
        let out = dir.join("parity.weights");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: next-token parity accuracy on held-out (val) sequences ---
        let scorer = lm.load_scorer(&out, block);
        let val = binio::read_u16_bin(&dir.join("val.bin"))?;
        let seq_len = self.seq_len();
        let n_val = val.len() / seq_len;
        let to_score = self.eval_sequences.min(n_val);

        // Replay the rng to recover the val tail's answer positions.
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
                let row = &logits[(ans_pos - 1) * v..ans_pos * v];
                predicted.push(argmax(row) as u32);
                expected.push(ans_tok as u32);
            }
        }

        let acc = associative_recall(&predicted, &expected);
        let chance = 0.5; // two equiprobable bit values
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

    #[test]
    fn sequence_shape_and_running_parity() {
        let m = Parity { n_bits: 6, ..Parity::default() };
        let mut rng = Rng::new(1);
        let (seq, answers) = m.gen_sequence(&mut rng);
        assert_eq!(seq.len(), m.seq_len());
        assert_eq!(seq[m.n_bits], SEP);
        assert_eq!(*seq.last().unwrap(), NL);
        assert_eq!(answers.len(), m.n_bits);

        // Recompute parity directly from the input bits and check each answer.
        let bits: Vec<u16> = (0..m.n_bits).map(|i| seq[i] - BIT0).collect();
        let mut acc = 0u16;
        for (i, &b) in bits.iter().enumerate() {
            acc ^= b;
            let (pos, tok) = answers[i];
            assert_eq!(seq[pos], tok);
            assert_eq!(tok, if acc == 1 { BIT1 } else { BIT0 });
        }
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = Parity { n_sequences: 100, ..Parity::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
