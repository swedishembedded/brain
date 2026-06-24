// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD **selective copying** — reproduce a *selected subset* of tokens, in
//! order, across noise and delays.
//!
//! Each sequence is a stream of tokens in which a few are tagged as
//! "to-be-copied" by a preceding **marker** token; the rest are distractor
//! noise. After a separator the model must emit exactly the marked tokens, in
//! their original order:
//!
//! ```text
//!   n n  M c1  n  M c2  n n  M c3  n   SEP   c1 c2 c3   NL
//!   └──────── tagged stream ────────┘        └ copy out ┘
//! ```
//!
//! `M` is the copy-marker; the token immediately after each `M` is selected.
//! Distractor tokens `n` are interspersed and at variable positions, so the model
//! cannot copy by fixed offset — it must *selectively* route the marked tokens
//! past the noise and delays. This is the MAD "selective copying" probe (the
//! task that motivated input-dependent gating in S4→Mamba).
//!
//! ## How it reuses the engine
//! - **Tokens.** `NL`, `SEP`, `MARK` (the copy marker), then a single content
//!   band used for both the copyable tokens and the noise (noise is just content
//!   tokens that are *not* preceded by `MARK`). Written as a char dataset
//!   (`SEP`→`'='`, `NL`→`'\n'`, `MARK`/content → Private-Use-Area chars) so
//!   `gpt::train` loads it unchanged.
//! - **Masking.** Loss masked up to & including `SEP`, per line, line-aligned —
//!   so only the copied-out region is a training target.
//! - **Scoring.** [`exact_match`](crate::metrics::exact_match) over the whole
//!   copied-out group, teacher-forced on the true prefix; a sequence is correct
//!   only if every selected token is reproduced in order. Per-token chance is
//!   `1 / vocab_content`, so full-sequence chance is `(1/vocab_content)^n_copy`.
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! `vocab_content=8`, `n_copy=3` selected tokens, `noise_per_slot=1` (≈4
//! distractors), 6000 sequences, 800 steps, 2-layer / d_model-96 / 4-head GPT.
//! Per-token chance 0.125; full-sequence chance `0.125^3 ≈ 0.002`. **Measured
//! exact-match ≈ 0.87** (train_ce ≈ 0.42), far clear of the **0.40** threshold,
//! in ~1.5-2 min on CPU (see `tests/mad_selective_copy.rs`).

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{exact_match, Metrics};
use crate::model::{argmax, DecoderLm, GptDecoder, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the stream→copy separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// Token id of the copy marker — the token right after it is selected.
const MARK: u16 = 2;
/// First content-token id; content tokens are `[CONTENT0, CONTENT0+vocab_content)`.
const CONTENT0: u16 = 3;

/// Selective-copying configuration.
#[derive(Clone, Debug)]
pub struct MadSelectiveCopy {
    /// Distinct content tokens (used for both copyable & noise tokens). Per-token
    /// chance is `1 / vocab_content`.
    pub vocab_content: usize,
    /// Number of tokens selected (marked) to copy out.
    pub n_copy: usize,
    /// Mean noise tokens injected per slot (between marked tokens).
    pub noise_per_slot: usize,
    /// Number of sequences in the generated corpus.
    pub n_sequences: usize,
    /// Training steps.
    pub steps: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Sequences scored for the metric (drawn from the val split).
    pub eval_sequences: usize,
}

impl Default for MadSelectiveCopy {
    /// Calibrated config: see module doc. Per-token chance 0.125, full-sequence
    /// chance ≈ 0.002, measured exact-match ≈ 0.85, threshold 0.40.
    fn default() -> Self {
        MadSelectiveCopy {
            vocab_content: 8,
            n_copy: 3,
            noise_per_slot: 1,
            n_sequences: 6000,
            steps: 800,
            n_layers: 2,
            d_model: 96,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl MadSelectiveCopy {
    fn rand_content(&self, rng: &mut Rng) -> u16 {
        CONTENT0 + rng.gen_range_inclusive(0, self.vocab_content as i64 - 1) as u16
    }

    /// Fixed token budget for the tagged stream: `n_copy` marked tokens (2 tokens
    /// each: MARK + content) plus `(n_copy+1)*noise_per_slot` noise slots.
    fn stream_budget(&self) -> usize {
        2 * self.n_copy + (self.n_copy + 1) * self.noise_per_slot
    }

    /// Fixed sequence length: stream + SEP + `n_copy` copied tokens + NL.
    fn seq_len(&self) -> usize {
        self.stream_budget() + 1 + self.n_copy + 1
    }

    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    fn vocab(&self) -> usize {
        CONTENT0 as usize + self.vocab_content
    }

    /// Generate one sequence and the copied-out group as `(start_index, tokens)`.
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, (usize, Vec<u16>)) {
        let selected: Vec<u16> = (0..self.n_copy).map(|_| self.rand_content(rng)).collect();

        // Build the tagged stream: noise burst, then MARK+selected, repeated;
        // pad to the fixed budget with trailing noise.
        let mut stream: Vec<u16> = Vec::with_capacity(self.stream_budget());
        for i in 0..self.n_copy {
            let burst = rng.gen_range_inclusive(0, self.noise_per_slot as i64) as usize;
            for _ in 0..burst {
                if stream.len() < self.stream_budget() {
                    stream.push(self.rand_content(rng));
                }
            }
            if stream.len() + 2 <= self.stream_budget() {
                stream.push(MARK);
                stream.push(selected[i]);
            }
        }
        while stream.len() < self.stream_budget() {
            stream.push(self.rand_content(rng));
        }
        stream.truncate(self.stream_budget());

        let mut seq = Vec::with_capacity(self.seq_len());
        seq.extend_from_slice(&stream);
        seq.push(SEP);
        let copy_start = seq.len();
        seq.extend_from_slice(&selected);
        seq.push(NL);
        debug_assert_eq!(seq.len(), self.seq_len());
        (seq, (copy_start, selected))
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

    fn itos(&self) -> Vec<char> {
        let mut itos = vec!['\0'; self.vocab()];
        itos[NL as usize] = '\n';
        itos[SEP as usize] = '=';
        // MARK + content map to distinct Private-Use-Area chars.
        for i in 0..(1 + self.vocab_content) {
            itos[MARK as usize + i] = char::from_u32(0xE000 + i as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for MadSelectiveCopy {
    fn name(&self) -> &str {
        "mad_selective_copy"
    }

    fn description(&self) -> &str {
        "MAD selective copying (reproduce marked tokens across noise/delays)"
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
        // Far above chance (~0.002), below the measured ~0.85, with fp32 margin.
        0.40
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["chance", "train_ce"]
    }
}

impl MadSelectiveCopy {
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
        let out = dir.join("mad_selective_copy.weights");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: whole-group exact-match on held-out (val) sequences ------
        let scorer = lm.load_scorer(&out, block);
        let val = binio::read_u16_bin(&dir.join("val.bin"))?;
        let seq_len = self.seq_len();
        let n_val = val.len() / seq_len;
        let to_score = self.eval_sequences.min(n_val);

        let mut rng = Rng::new(seed);
        let train_seqs = (self.n_sequences * 9) / 10;
        for _ in 0..train_seqs {
            self.gen_sequence(&mut rng);
        }

        let mut pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for s in 0..to_score {
            let (seq, (copy_start, selected)) = self.gen_sequence(&mut rng);
            debug_assert_eq!(&seq[..], &val[s * seq_len..(s + 1) * seq_len]);
            let toks: Vec<u32> = seq.iter().map(|&t| t as u32).collect();
            let logits = scorer.logits_all(&toks);
            let v = scorer.vocab();
            let mut pred: Vec<u32> = Vec::with_capacity(selected.len());
            for k in 0..selected.len() {
                let p = copy_start + k;
                let row = &logits[(p - 1) * v..p * v];
                pred.push(argmax(row) as u32);
            }
            let expected: Vec<u32> = selected.iter().map(|&t| t as u32).collect();
            pairs.push((pred, expected));
        }

        let acc = exact_match(&pairs);
        // Full-sequence chance: copy all n_copy tokens correctly by luck.
        let chance = (1.0 / self.vocab_content.max(1) as f32).powi(self.n_copy as i32);
        Ok(Metrics::new(acc)
            .with("exact_match", acc)
            .with("chance", chance)
            .with("train_ce", final_loss)
            .with("init_ce", init_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_out_matches_marked_tokens() {
        let m = MadSelectiveCopy::default();
        let mut rng = Rng::new(4);
        for _ in 0..50 {
            let (seq, (start, selected)) = m.gen_sequence(&mut rng);
            assert_eq!(seq.len(), m.seq_len());
            assert_eq!(*seq.last().unwrap(), NL);
            assert_eq!(seq[start - 1], SEP);
            assert_eq!(&seq[start..start + selected.len()], &selected[..]);
            // Each selected token appears immediately after a MARK in the stream.
            let stream = &seq[..m.stream_budget()];
            for &tok in &selected {
                let ok = stream.windows(2).any(|w| w[0] == MARK && w[1] == tok);
                assert!(ok, "selected token {tok} not found marked in stream");
            }
        }
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = MadSelectiveCopy { n_sequences: 100, ..MadSelectiveCopy::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
