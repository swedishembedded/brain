// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD **memorization** — a **fixed** key→value map shared across *all*
//! sequences; the association must be stored in the **weights**, not read from
//! context.
//!
//! Unlike the in-context recall tasks, there are no bindings in the sequence.
//! Every sequence is just a query and its answer for one entry of a single,
//! global map that is identical for every sequence in the corpus:
//!
//! ```text
//!   k   SEP   v   NL          (and the SAME k always maps to the SAME v)
//! ```
//!
//! Because the map is fixed and shared, the value cannot be recovered from the
//! current sequence — there is nothing to attend to. The only way to predict `v`
//! is to have *memorized* the `k→v` table in the parameters. This is the MAD
//! "memorization" probe: it measures an architecture's associative-memory
//! capacity (how many fixed facts it can store), as opposed to its in-context
//! routing.
//!
//! ## How it reuses the engine
//! - **Tokens.** `NL`, `SEP`, then two disjoint content bands: keys and values.
//!   The global map binds each of `n_keys` distinct keys to a fixed value.
//!   Written as a char dataset (`SEP`→`'='`, `NL`→`'\n'`, content → Private-Use-
//!   Area chars) so `gpt::train` loads it unchanged.
//! - **Masking.** Loss masked up to & including `SEP`, per line, line-aligned —
//!   so the model is trained to emit the answer given only the key.
//! - **Scoring.** [`associative_recall`](crate::metrics::associative_recall) at
//!   the answer position over held-out sequences. The *same keys* recur in the
//!   val split (the map is global), so a model that memorized the table scores
//!   high; chance is `1 / vocab_values`. (This is the intended "train/test on the
//!   same fixed facts" memorization setup — generalization is to new *instances*
//!   of the same facts, not new facts.)
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! `n_keys=32` fixed facts, `vocab_values=16` (**chance = 1/16 = 0.0625**), 6000
//! sequences, 400 steps, 2-layer / d_model-64 / 4-head GPT. **Measured recall
//! ≈ 1.00** (the 32-fact table is small and fully memorizable), far clear of the
//! **0.60** threshold, in ~20 s on CPU (see `tests/mad_memorize.rs`).

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{associative_recall, Metrics};
use crate::model::{argmax, DecoderLm, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the key→answer separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// First content-token id; bands follow: keys, then values.
const CONTENT0: u16 = 2;

/// Memorization configuration.
#[derive(Clone, Debug)]
pub struct MadMemorize {
    /// Number of distinct keys in the fixed global map (= number of facts).
    pub n_keys: usize,
    /// Distinct value tokens. Chance recall is `1 / vocab_values`.
    pub vocab_values: usize,
    /// Number of sequences in the generated corpus (each one query of the map).
    pub n_sequences: usize,
    /// Training steps.
    pub steps: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Sequences scored for the metric (drawn from the val split).
    pub eval_sequences: usize,
}

impl Default for MadMemorize {
    /// Calibrated config: see module doc. Chance = 0.0625, measured recall ≈ 1.00,
    /// threshold 0.60.
    fn default() -> Self {
        MadMemorize {
            n_keys: 32,
            vocab_values: 16,
            n_sequences: 6000,
            steps: 400,
            n_layers: 2,
            d_model: 64,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl MadMemorize {
    fn key0(&self) -> u16 {
        CONTENT0
    }
    fn val0(&self) -> u16 {
        CONTENT0 + self.n_keys as u16
    }

    /// Sequence length: key + SEP + answer + NL.
    fn seq_len(&self) -> usize {
        1 + 1 + 1 + 1
    }

    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    fn vocab(&self) -> usize {
        CONTENT0 as usize + self.n_keys + self.vocab_values
    }

    /// The fixed global map: key index `i` → a value token, derived
    /// deterministically from `seed` (so prepare & evaluate agree). Built once.
    fn build_map(&self, seed: u64) -> Vec<u16> {
        // Use a dedicated rng stream (offset seed) so the map is independent of
        // the per-sequence query sampling below.
        let mut rng = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        (0..self.n_keys)
            .map(|_| self.val0() + rng.gen_range_inclusive(0, self.vocab_values as i64 - 1) as u16)
            .collect()
    }

    /// Generate one query sequence (a random key + its fixed answer) and the
    /// `(answer_index, answer_token)`.
    fn gen_sequence(&self, rng: &mut Rng, map: &[u16]) -> (Vec<u16>, (usize, u16)) {
        let ki = rng.gen_range_inclusive(0, self.n_keys as i64 - 1) as usize;
        let key = self.key0() + ki as u16;
        let ans = map[ki];
        let mut seq = Vec::with_capacity(self.seq_len());
        seq.push(key);
        seq.push(SEP);
        let ans_pos = seq.len();
        seq.push(ans);
        seq.push(NL);
        (seq, (ans_pos, ans))
    }

    fn build_corpus(&self, seed: u64) -> Vec<u16> {
        let map = self.build_map(seed);
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(self.n_sequences * self.seq_len());
        for _ in 0..self.n_sequences {
            let (seq, _) = self.gen_sequence(&mut rng, &map);
            out.extend_from_slice(&seq);
        }
        out
    }

    fn itos(&self) -> Vec<char> {
        let mut itos = vec!['\0'; self.vocab()];
        itos[NL as usize] = '\n';
        itos[SEP as usize] = '=';
        let n_content = self.n_keys + self.vocab_values;
        for i in 0..n_content {
            itos[CONTENT0 as usize + i] = char::from_u32(0xE000 + i as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for MadMemorize {
    fn name(&self) -> &str {
        "mad_memorize"
    }

    fn description(&self) -> &str {
        "MAD memorization (fixed key->value map stored in weights)"
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
        // Far above chance (0.0625), below the measured ~0.99, with fp32 margin.
        0.60
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
        let out = dir.join("mad_memorize.weights");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: recall of the memorized map on held-out (val) sequences --
        let scorer = lm.load_scorer(&out, block);
        let val = binio::read_u16_bin(&dir.join("val.bin"))?;
        let seq_len = self.seq_len();
        let n_val = val.len() / seq_len;
        let to_score = self.eval_sequences.min(n_val);

        let map = self.build_map(seed);
        let mut rng = Rng::new(seed);
        let train_seqs = (self.n_sequences * 9) / 10;
        for _ in 0..train_seqs {
            self.gen_sequence(&mut rng, &map);
        }

        let mut predicted = Vec::new();
        let mut expected = Vec::new();
        for s in 0..to_score {
            let (seq, (ans_pos, ans_tok)) = self.gen_sequence(&mut rng, &map);
            debug_assert_eq!(&seq[..], &val[s * seq_len..(s + 1) * seq_len]);
            let toks: Vec<u32> = seq.iter().map(|&t| t as u32).collect();
            let logits = scorer.logits_all(&toks);
            let v = scorer.vocab();
            let row = &logits[(ans_pos - 1) * v..ans_pos * v];
            predicted.push(argmax(row) as u32);
            expected.push(ans_tok as u32);
        }

        let recall = associative_recall(&predicted, &expected);
        let chance = 1.0 / self.vocab_values.max(1) as f32;
        Ok(Metrics::new(recall)
            .with("recall", recall)
            .with("chance", chance)
            .with("train_ce", final_loss)
            .with("init_ce", init_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_is_fixed_across_sequences() {
        let m = MadMemorize::default();
        let map = m.build_map(42);
        // The same key always yields the same answer, regardless of rng state.
        let mut rng = Rng::new(42);
        let mut seen: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();
        for _ in 0..500 {
            let (seq, (pos, tok)) = m.gen_sequence(&mut rng, &map);
            assert_eq!(seq.len(), m.seq_len());
            assert_eq!(seq[1], SEP);
            assert_eq!(seq[pos], tok);
            let key = seq[0];
            if let Some(&prev) = seen.get(&key) {
                assert_eq!(prev, tok, "fixed map violated: key {key} mapped to two values");
            }
            seen.insert(key, tok);
        }
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = MadMemorize { n_sequences: 100, ..MadMemorize::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
