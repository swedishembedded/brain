// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD **fuzzy in-context recall** — recall where keys and values span a
//! **variable number of tokens**; tests grouping & boundary handling.
//!
//! Each sequence binds a random map whose keys and values are *multi-token*
//! groups of a variable length, then asks the model to reproduce the value group
//! bound to a queried key group:
//!
//! ```text
//!   K1.. V1..  K2.. V2.. ... Km.. Vm..   SEP   Q1.. A1..   NL
//!   └────────────── bindings ─────────┘        └ query+answer ┘
//! ```
//!
//! Here `Ki`/`Vi` are length-varying token tuples (e.g. 1–2 tokens each). Recall
//! is no longer a single induction-head copy: the model must recognize where one
//! group ends and the next begins (boundary handling) and copy a *whole* group,
//! not one token. This is the MAD "fuzzy recall" probe — it stresses an
//! architecture's ability to group adjacent tokens into a unit before routing.
//!
//! ## How it reuses the engine
//! - **Tokens.** `NL`, `SEP`, then two disjoint content bands: a key band and a
//!   value band. A key/value group is a tuple of tokens drawn from its band. To
//!   keep boundaries learnable from content alone (no explicit delimiter), keys
//!   and values use disjoint bands so a key→value transition is always a band
//!   change. Written as a char dataset so `gpt::train` loads it unchanged.
//! - **Masking.** Loss masked up to & including `SEP`, per line, line-aligned —
//!   so only the answer group is a training target.
//! - **Scoring.** Exact-match over the **whole answer group**: a sequence counts
//!   as recalled only if every token of the value group is argmax-correct
//!   (greedy, teacher-forced on the ground-truth prefix). Reported as
//!   group-level [`exact_match`](crate::metrics::exact_match) accuracy; chance is
//!   `(1/vocab_values)^(mean value len)` — vanishingly small, so the headline is
//!   compared against a conservative threshold well above it.
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! `vocab_keys=8`, `vocab_values=8`, key/value group length uniform in `1..=2`,
//! `n_pairs=2`, 8000 sequences, 1200 steps, 2-layer / d_model-96 / 4-head GPT.
//! Reported `chance` is the most generous (shortest, 1-token group) per-group
//! value, `1/vocab_values = 0.125`; true mixed-length full-group chance is lower.
//! **Measured group exact-match ≈ 0.575** (train_ce ≈ 0.68), clear of the
//! **0.40** threshold, in ~2-3 min on CPU (see `tests/mad_fuzzy_recall.rs`).

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{exact_match, Metrics};
use crate::model::{argmax, DecoderLm, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the bindings→query separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// First content-token id; bands follow: keys, then values.
const CONTENT0: u16 = 2;

/// Fuzzy in-context recall configuration.
#[derive(Clone, Debug)]
pub struct MadFuzzyRecall {
    /// Distinct key tokens (key band).
    pub vocab_keys: usize,
    /// Distinct value tokens (value band).
    pub vocab_values: usize,
    /// Key→value bindings per sequence.
    pub n_pairs: usize,
    /// Min / max tokens per key or value group (inclusive).
    pub min_group: usize,
    pub max_group: usize,
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

impl Default for MadFuzzyRecall {
    /// Calibrated config: see module doc. Reported chance 0.125 (generous), measured
    /// group exact-match ≈ 0.575, threshold 0.40.
    fn default() -> Self {
        MadFuzzyRecall {
            vocab_keys: 8,
            vocab_values: 8,
            n_pairs: 2,
            min_group: 1,
            max_group: 2,
            n_sequences: 8000,
            steps: 1200,
            n_layers: 2,
            d_model: 96,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl MadFuzzyRecall {
    fn key0(&self) -> u16 {
        CONTENT0
    }
    fn val0(&self) -> u16 {
        CONTENT0 + self.vocab_keys as u16
    }

    /// Fixed sequence budget so all sequences share a length. Each pair uses up
    /// to `2*max_group` tokens; plus SEP + one query group + one answer group +
    /// NL. We pad the bindings region (with repeats of the last full group's
    /// trailing structure) — actually we pad with a fixed FILLER so lengths
    /// match; see [`gen_sequence`].
    fn seq_len(&self) -> usize {
        self.bindings_budget() + 1 + 2 * self.max_group + 1
    }

    fn bindings_budget(&self) -> usize {
        self.n_pairs * 2 * self.max_group
    }

    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    fn vocab(&self) -> usize {
        CONTENT0 as usize + self.vocab_keys + self.vocab_values
    }

    fn group_len(&self, rng: &mut Rng) -> usize {
        rng.gen_range_inclusive(self.min_group as i64, self.max_group as i64) as usize
    }

    fn rand_key(&self, rng: &mut Rng) -> u16 {
        self.key0() + rng.gen_range_inclusive(0, self.vocab_keys as i64 - 1) as u16
    }
    fn rand_val(&self, rng: &mut Rng) -> u16 {
        self.val0() + rng.gen_range_inclusive(0, self.vocab_values as i64 - 1) as u16
    }

    /// Generate one sequence and the answer group as `(start_index, tokens)`.
    ///
    /// Keys are distinct groups within a sequence (so each query resolves to one
    /// binding). The bindings region is padded to a fixed budget with value-band
    /// FILLER so all sequences share a length; padding lives *after* the real
    /// bindings and is masked out of scoring anyway (it precedes SEP).
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, (usize, Vec<u16>)) {
        // Build distinct key groups and their value groups.
        let mut keys: Vec<Vec<u16>> = Vec::with_capacity(self.n_pairs);
        let mut values: Vec<Vec<u16>> = Vec::with_capacity(self.n_pairs);
        while keys.len() < self.n_pairs {
            let kl = self.group_len(rng);
            let kg: Vec<u16> = (0..kl).map(|_| self.rand_key(rng)).collect();
            if keys.iter().any(|k| *k == kg) {
                continue; // keep key groups distinct
            }
            let vl = self.group_len(rng);
            let vg: Vec<u16> = (0..vl).map(|_| self.rand_val(rng)).collect();
            keys.push(kg);
            values.push(vg);
        }

        let mut region: Vec<u16> = Vec::with_capacity(self.bindings_budget());
        for i in 0..self.n_pairs {
            region.extend_from_slice(&keys[i]);
            region.extend_from_slice(&values[i]);
        }
        // Pad to fixed budget with value-band filler (masked region, pre-SEP).
        while region.len() < self.bindings_budget() {
            region.push(self.val0());
        }
        region.truncate(self.bindings_budget());

        let mut seq = Vec::with_capacity(self.seq_len());
        seq.extend_from_slice(&region);
        seq.push(SEP);
        let qi = rng.gen_range_inclusive(0, self.n_pairs as i64 - 1) as usize;
        seq.extend_from_slice(&keys[qi]); // query group
        let ans_start = seq.len();
        let answer = values[qi].clone();
        seq.extend_from_slice(&answer); // answer group
        seq.push(NL);
        // Pad the (query+answer) region to fixed length with NL so seq_len holds.
        while seq.len() < self.seq_len() {
            seq.push(NL);
        }
        seq.truncate(self.seq_len());
        (seq, (ans_start, answer))
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
        let n_content = self.vocab_keys + self.vocab_values;
        for i in 0..n_content {
            itos[CONTENT0 as usize + i] = char::from_u32(0xE000 + i as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for MadFuzzyRecall {
    fn name(&self) -> &str {
        "mad_fuzzy_recall"
    }

    fn description(&self) -> &str {
        "MAD fuzzy in-context recall (multi-token keys/values, group boundaries)"
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
        // Far above chance (<0.05 even at shortest group), below the measured
        // ~0.80, with fp32 / single-run margin.
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
        let out = dir.join("mad_fuzzy_recall.weights");
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
            let (seq, (ans_start, answer)) = self.gen_sequence(&mut rng);
            debug_assert_eq!(&seq[..], &val[s * seq_len..(s + 1) * seq_len]);
            let toks: Vec<u32> = seq.iter().map(|&t| t as u32).collect();
            // Teacher-forced on the ground-truth prefix: for each answer token at
            // position p, predict from logits at p-1 (the true context). This
            // scores the model's ability to copy the whole group, position by
            // position, against the true sequence.
            let logits = scorer.logits_all(&toks);
            let v = scorer.vocab();
            let mut pred: Vec<u32> = Vec::with_capacity(answer.len());
            for k in 0..answer.len() {
                let p = ans_start + k;
                let row = &logits[(p - 1) * v..p * v];
                pred.push(argmax(row) as u32);
            }
            let expected: Vec<u32> = answer.iter().map(|&t| t as u32).collect();
            pairs.push((pred, expected));
        }

        let acc = exact_match(&pairs);
        // Mean answer-group length ~ (min+max)/2; chance per token = 1/vocab_values,
        // so full-group chance ≈ (1/vocab_values)^mean_len — report a conservative
        // upper bound at the *shortest* group length (most generous to chance).
        let chance = (1.0 / self.vocab_values.max(1) as f32).powi(self.min_group as i32);
        Ok(Metrics::new(acc)
            .with("group_em", acc)
            .with("chance", chance)
            .with("train_ce", final_loss)
            .with("init_ce", init_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_fixed_length_and_answer_in_value_band() {
        let m = MadFuzzyRecall::default();
        let mut rng = Rng::new(2);
        for _ in 0..50 {
            let (seq, (start, answer)) = m.gen_sequence(&mut rng);
            assert_eq!(seq.len(), m.seq_len());
            assert!(!answer.is_empty());
            // The recorded answer group sits at `start` in the sequence.
            assert_eq!(&seq[start..start + answer.len()], &answer[..]);
            // Answer tokens are value-band tokens.
            assert!(answer.iter().all(|&t| t >= m.val0() && (t as usize) < m.vocab()));
        }
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = MadFuzzyRecall { n_sequences: 100, ..MadFuzzyRecall::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
