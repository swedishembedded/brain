// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD **in-context recall** — the canonical single-query associative-recall
//! task from the *Mechanistic Architecture Design* (MAD) suite.
//!
//! Each sequence binds a fresh, random `key→value` map, then asks the model to
//! recall **one** queried value from that same sequence:
//!
//! ```text
//!   k1 v1  k2 v2 ... km vm   SEP   q1 a1   NL
//!   └────────── bindings ──────┘    └ query+answer ┘
//! ```
//!
//! `q1` is one of the earlier keys and `a1` the value it was bound to. Like
//! [`mqar`](crate::mqar) this is a data-dependent induction-head lookup that
//! n-gram statistics cannot solve. It overlaps MQAR by design — MQAR is the
//! *multi*-query generalization — but this module is the MAD-canonical
//! **single-query** variant: the map is regenerated per sequence and exactly one
//! value is queried, isolating the simplest recall mechanism.
//!
//! ## How it reuses the engine
//! - **Tokens.** `NL` (sequence end), `SEP` (bindings→query separator), then
//!   `vocab_content` content tokens, split into a lower half (keys) and a
//!   disjoint upper half (values) so a query key never collides with a value —
//!   exactly the MQAR layout. Written as a char dataset (`SEP`→`'='`, `NL`→`'\n'`,
//!   content → Private-Use-Area chars) so `gpt::train` loads it unchanged.
//! - **Masking.** Loss masked up to & including `SEP`, per line, line-aligned
//!   (`mask_before='='`, `mask_per_line`, `align_to_lines`) so the gradient
//!   focuses on the single answer rather than memorizing the random bindings.
//! - **Scoring.** [`associative_recall`](crate::metrics::associative_recall) at
//!   the lone answer position of each held-out sequence; chance is
//!   `2 / vocab_content` (the answer is one upper-half value token).
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! `vocab_content=16` (8 keys + 8 values, **chance = 0.125**), `n_pairs=2`,
//! single query, 8000 sequences, 900 steps, 2-layer / d_model-64 / 4-head GPT.
//! **Measured recall ≈ 0.51-0.58 across seeds** (train_ce ≈ 0.87), far above
//! chance and clear of the **0.40** threshold, in ~1 min on CPU (see
//! `tests/mad_recall.rs`). Unlike MQAR (which supervises *two* answer tokens per
//! sequence and reaches ~0.77), this single-query variant trains on only one
//! answer token per sequence, so the gradient is sparser and accuracy plateaus
//! lower — a faithful, stable characterization of single-query recall on this
//! small GPT.

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{associative_recall, Metrics};
use crate::model::{argmax, DecoderLm, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the bindings→query separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// First content-token id; content tokens are `[CONTENT0, CONTENT0+vocab_content)`.
const CONTENT0: u16 = 2;

/// In-context recall configuration. Defaults are calibrated to be clearly
/// learnable by a 2-layer / d_model-64 GPT in a few hundred CPU steps.
#[derive(Clone, Debug)]
pub struct MadRecall {
    /// Distinct content tokens keys & values are drawn from (split key/value
    /// halves). Chance recall is `2 / vocab_content`.
    pub vocab_content: usize,
    /// Key→value bindings per sequence (the map regenerated each sequence).
    pub n_pairs: usize,
    /// Number of sequences in the generated corpus.
    pub n_sequences: usize,
    /// Training steps.
    pub steps: u32,
    /// GPT depth / width / heads for the scoring model.
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Sequences scored for the recall metric (drawn from the val split).
    pub eval_sequences: usize,
}

impl Default for MadRecall {
    /// Calibrated config: see the module doc comment. Chance = 0.125, measured
    /// recall ≈ 0.51-0.58 across seeds, threshold 0.40.
    fn default() -> Self {
        MadRecall {
            vocab_content: 16,
            n_pairs: 2,
            n_sequences: 8000,
            steps: 900,
            n_layers: 2,
            d_model: 64,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl MadRecall {
    /// Sequence length: `2*n_pairs` (bindings) + 1 (`SEP`) + 2 (query+answer) + 1 (`NL`).
    fn seq_len(&self) -> usize {
        2 * self.n_pairs + 1 + 2 + 1
    }

    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    fn vocab(&self) -> usize {
        CONTENT0 as usize + self.vocab_content
    }

    /// Generate one sequence and its single `(answer_index, answer_token)`.
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, (usize, u16)) {
        // Keys from the lower half, values from the disjoint upper half: a query
        // key can never coincide with a value token, keeping the lookup signal
        // unambiguous. Keys are distinct so each query has exactly one binding.
        let half = self.vocab_content / 2;
        let key0 = CONTENT0;
        let val0 = CONTENT0 + half as u16;
        let keys: Vec<u16> = sample_distinct_indices(self.n_pairs, half, rng)
            .into_iter()
            .map(|i| key0 + i as u16)
            .collect();
        let values: Vec<u16> = (0..self.n_pairs)
            .map(|_| val0 + rng.gen_range_inclusive(0, half as i64 - 1) as u16)
            .collect();

        let mut seq = Vec::with_capacity(self.seq_len());
        for i in 0..self.n_pairs {
            seq.push(keys[i]);
            seq.push(values[i]);
        }
        seq.push(SEP);
        let qi = rng.gen_range_inclusive(0, self.n_pairs as i64 - 1) as usize;
        seq.push(keys[qi]); // query key
        let ans_pos = seq.len();
        seq.push(values[qi]); // answer value
        seq.push(NL);
        (seq, (ans_pos, values[qi]))
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

    /// Synthetic `itos`: SEP→`'='`, NL→`'\n'`, content → Private-Use-Area chars.
    fn itos(&self) -> Vec<char> {
        let mut itos = vec!['\0'; self.vocab()];
        itos[NL as usize] = '\n';
        itos[SEP as usize] = '=';
        for i in 0..self.vocab_content {
            itos[CONTENT0 as usize + i] = char::from_u32(0xE000 + i as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for MadRecall {
    fn name(&self) -> &str {
        "mad_recall"
    }

    fn description(&self) -> &str {
        "MAD in-context recall (single-query key->value lookup)"
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
        // Far above chance (0.125) — and above 3x chance (0.375) — yet below the
        // measured ~0.51-0.58 floor across seeds, with margin for the single
        // answer-token-per-sequence gradient's run-to-run variance.
        0.40
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["chance", "train_ce"]
    }
    /// Train + score with a specific architecture (any [`DecoderLm`]).
    /// [`Benchmark::evaluate`] calls this with the GPT baseline.
    fn evaluate_with(&self, lm: &dyn DecoderLm, dir: &Path, seed: u64) -> std::io::Result<Metrics> {
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
        let out = dir.join("mad_recall.safetensors");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: single-query recall on held-out (val) sequences ----------
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
            let (seq, (ans_pos, ans_tok)) = self.gen_sequence(&mut rng);
            debug_assert_eq!(&seq[..], &val[s * seq_len..(s + 1) * seq_len]);
            let toks: Vec<u32> = seq.iter().map(|&t| t as u32).collect();
            let logits = scorer.logits_all(&toks);
            let v = scorer.vocab();
            let row = &logits[(ans_pos - 1) * v..ans_pos * v];
            predicted.push(argmax(row) as u32);
            expected.push(ans_tok as u32);
        }

        let recall = associative_recall(&predicted, &expected);
        let chance = 1.0 / (self.vocab_content / 2).max(1) as f32;
        Ok(Metrics::new(recall)
            .with("recall", recall)
            .with("chance", chance)
            .with("train_ce", final_loss)
            .with("init_ce", init_loss))
    }
}

/// `k` distinct indices in `[0, n)` via partial Fisher–Yates (requires `k <= n`).
fn sample_distinct_indices(k: usize, n: usize, rng: &mut Rng) -> Vec<usize> {
    assert!(k <= n, "cannot draw {k} distinct of {n}");
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range_inclusive(i as i64, n as i64 - 1) as usize;
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_shape_and_answer() {
        let m = MadRecall { vocab_content: 8, n_pairs: 3, ..MadRecall::default() };
        let mut rng = Rng::new(1);
        let (seq, (pos, tok)) = m.gen_sequence(&mut rng);
        assert_eq!(seq.len(), m.seq_len());
        assert_eq!(seq[2 * m.n_pairs], SEP);
        assert_eq!(*seq.last().unwrap(), NL);
        assert_eq!(seq[pos], tok);
        // The query key (just before the answer) appeared earlier as a binding key.
        let key_positions: Vec<u16> = (0..m.n_pairs).map(|i| seq[2 * i]).collect();
        assert!(key_positions.contains(&seq[pos - 1]));
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = MadRecall { n_sequences: 100, ..MadRecall::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
