// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MQAR — **multi-query associative recall** (the reference benchmark).
//!
//! Each sequence binds several `key→value` pairs, then asks the model to recall
//! the value for several queried keys **from the same sequence**:
//!
//! ```text
//!   k1 v1  k2 v2  ... km vm   SEP   q1 a1  q2 a2 ... qn an   NL
//!   └──────── bindings ─────┘       └──── queries+answers ───┘
//! ```
//!
//! `qi` is one of the earlier keys and `ai` is the value it was bound to;
//! predicting `ai` from the adjacent `qi` is a single induction-head lookup.
//! This is
//! the classic in-context-recall probe from the "Zoology"/associative-recall
//! literature: it cannot be solved by n-gram statistics — the model must route
//! information from the binding to the matching query, so it cleanly separates
//! architectures that can do data-dependent lookup from those that cannot.
//!
//! ## How it reuses the engine
//! - **Tokens.** A small synthetic vocab: `NL` (newline / sequence end), `SEP`
//!   (the bindings→queries separator), then `vocab_content` interchangeable
//!   content tokens used for both keys and values. The dataset is written in
//!   brain's standard char-token layout (`train.bin`/`val.bin`/`meta.json`), so
//!   `gpt2::train` loads it unchanged - each token id maps to a distinct char in
//!   `meta.json` (`SEP`→`'='`, `NL`→`'\n'`, content→Private-Use-Area chars).
//! - **Masking.** Training masks loss up to & including `SEP`, per line, via the
//!   existing loader path (`mask_before = '='`, `mask_per_line = true`) — exactly
//!   the calculator/copy recipe — so the gradient focuses on the query/answer
//!   region rather than memorizing the (random) bindings.
//! - **Scoring.** [`associative_recall`](crate::metrics::associative_recall) over
//!   *answer positions only*: we forward each val sequence and check, at every
//!   `ai`, whether the model's argmax over the previous position equals `ai`.
//!   Chance is `1 / vocab_content`.

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{associative_recall, Metrics};
use crate::model::{argmax, DecoderLm, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the bindings→queries separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// First content-token id; content tokens are `[CONTENT0, CONTENT0+vocab_content)`.
const CONTENT0: u16 = 2;

/// MQAR configuration. Defaults are calibrated to be clearly learnable by a
/// 2-layer / d_model-64 GPT within a few hundred CPU steps (see [`Mqar::default`]).
#[derive(Clone, Debug)]
pub struct Mqar {
    /// Number of distinct content tokens keys & values are drawn from. Chance
    /// recall accuracy is `1 / vocab_content`.
    pub vocab_content: usize,
    /// Key→value bindings per sequence.
    pub n_pairs: usize,
    /// Queries per sequence (≤ `n_pairs`).
    pub n_queries: usize,
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

impl Default for Mqar {
    /// Calibrated config: 16-token content vocab split into 8 keys + 8 values
    /// (so chance recall = 1/8 = 0.125), 2 bindings, 2 queries, a 2-layer /
    /// d_model-64 / 4-head GPT for 600 steps. Measured recall on the CPU
    /// (Cranelift JIT) backend is ~0.77 — well above chance and the 0.55
    /// threshold — in a few minutes (see `tests/mqar.rs`). Difficulty is
    /// dominated by `n_pairs` (the number of keys to disambiguate): 3 bindings
    /// drops a same-budget run to ~0.41, so 2 is the calibrated sweet spot for a
    /// fast, clearly-above-chance reference benchmark.
    fn default() -> Self {
        Mqar {
            vocab_content: 16,
            n_pairs: 2,
            n_queries: 2,
            n_sequences: 6000,
            steps: 600,
            n_layers: 2,
            d_model: 64,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl Mqar {
    /// Sequence length in tokens: `2*n_pairs` (bindings) + 1 (`SEP`) +
    /// `2*n_queries` (interleaved query/answer pairs) + 1 (`NL`).
    fn seq_len(&self) -> usize {
        2 * self.n_pairs + 1 + 2 * self.n_queries + 1
    }

    /// Block size used for both training and scoring: one whole sequence fits.
    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    /// Total vocab size including `NL` and `SEP`.
    fn vocab(&self) -> usize {
        CONTENT0 as usize + self.vocab_content
    }

    /// Generate one sequence and the **answer positions** within it. Returns the
    /// token ids and a list of `(index_of_answer_token, answer_token)` pairs.
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, Vec<(usize, u16)>) {
        // Keys are drawn from the first half of the content vocab, values from
        // the disjoint second half. Disjoint ranges mean a query key can never
        // coincide with any *value* token, so the induction-head lookup ("find
        // the earlier occurrence of this key, copy its successor") is never
        // confused by a value that happens to equal the query — a clean,
        // unambiguous recall signal. Keys are distinct within a sequence so each
        // query has exactly one binding; values may repeat freely.
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

        // Layout: bindings, SEP, then interleaved query/answer pairs, then NL:
        //   k1 v1 .. km vm   SEP   q1 a1  q2 a2 ... qn an   NL
        // Masking up to & including SEP focuses training on the query/answer
        // region. Predicting `ai` from the immediately preceding `qi` is the
        // classic single-induction-head lookup ("attend to the earlier
        // occurrence of qi as a key, copy its successor value"), the easiest
        // recall mechanism for a small model to learn.
        let mut seq = Vec::with_capacity(self.seq_len());
        for i in 0..self.n_pairs {
            seq.push(keys[i]);
            seq.push(values[i]);
        }
        seq.push(SEP);
        let query_idx = sample_distinct_indices(self.n_queries, self.n_pairs, rng);
        let mut answers = Vec::with_capacity(self.n_queries);
        for &qi in &query_idx {
            seq.push(keys[qi]); // query key
            let ans_pos = seq.len();
            seq.push(values[qi]); // answer value
            answers.push((ans_pos, values[qi]));
        }
        seq.push(NL);
        (seq, answers)
    }

    /// Build the flat token corpus (and total byte count for bpb).
    fn build_corpus(&self, seed: u64) -> Vec<u16> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(self.n_sequences * self.seq_len());
        for _ in 0..self.n_sequences {
            let (seq, _) = self.gen_sequence(&mut rng);
            out.extend_from_slice(&seq);
        }
        out
    }

    /// Synthetic `itos`: SEP→`'='` (the mask char), NL→`'\n'`, content tokens →
    /// distinct Private-Use-Area chars, so the standard char-dataset loader +
    /// masking path works without a real text corpus.
    fn itos(&self) -> Vec<char> {
        let mut itos = vec!['\0'; self.vocab()];
        itos[NL as usize] = '\n';
        itos[SEP as usize] = '=';
        for i in 0..self.vocab_content {
            // U+E000.. is the BMP Private Use Area; vocab_content stays well within.
            itos[CONTENT0 as usize + i] = char::from_u32(0xE000 + i as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for Mqar {
    fn name(&self) -> &str {
        "mqar"
    }

    fn description(&self) -> &str {
        "multi-query associative recall (in-context key->value lookup)"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let corpus = self.build_corpus(seed);
        // 90/10 train/val split on whole sequences.
        let split_seqs = (self.n_sequences * 9) / 10;
        let split = split_seqs * self.seq_len();
        binio::write_u16_bin(&dir.join("train.bin"), &corpus[..split])?;
        binio::write_u16_bin(&dir.join("val.bin"), &corpus[split..])?;
        let meta = Meta { vocab_size: self.vocab(), itos: self.itos() };
        std::fs::write(dir.join("meta.json"), meta.to_json())?;
        Ok(())
    }

    fn threshold(&self) -> f32 {
        // Far above chance (0.125) yet below the measured ~0.77, with margin for
        // fp32 / single-run noise on the software CPU backend.
        0.55
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["chance", "train_ce"]
    }

    /// Train + score this benchmark with a specific architecture (any
    /// [`DecoderLm`]). [`Benchmark::evaluate`] calls this with the GPT baseline;
    /// scoring an alternative architecture is just passing a different
    /// `DecoderLm` — no other change. This is the architecture-agnostic core.
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
            mask_before: Some('='), // SEP — mask bindings, learn the answers
            mask_per_line: true,
            align_to_lines: true, // each window is one full sequence (NL-aligned)
            seed,
        };
        let out = dir.join("mqar.safetensors");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: associative-recall on held-out (val) sequences -----------
        let scorer = lm.load_scorer(&out, block);
        let val = binio::read_u16_bin(&dir.join("val.bin"))?;
        let seq_len = self.seq_len();
        let n_val = val.len() / seq_len;
        let to_score = self.eval_sequences.min(n_val);

        // Regenerate answer positions deterministically: gen_sequence is pure in
        // its rng, so replaying the same seed yields the same corpus + answers.
        // We only need the val tail's answer positions, so fast-forward the rng
        // through the train sequences first.
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
            let logits = scorer.logits_all(&toks); // [seq_len * vocab]
            let v = scorer.vocab();
            for &(ans_pos, ans_tok) in &answers {
                // Predict the answer token from the logits at the previous position.
                let row = &logits[(ans_pos - 1) * v..ans_pos * v];
                predicted.push(argmax(row) as u32);
                expected.push(ans_tok as u32);
            }
        }

        let recall = associative_recall(&predicted, &expected);
        // The answer is one of `vocab_content/2` value tokens (values live in the
        // disjoint upper half of the content vocab), so chance is 2/vocab_content.
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
    fn sequence_shape_and_answers() {
        let m = Mqar { vocab_content: 8, n_pairs: 3, n_queries: 2, ..Mqar::default() };
        let mut rng = Rng::new(1);
        let (seq, answers) = m.gen_sequence(&mut rng);
        assert_eq!(seq.len(), m.seq_len());
        // SEP sits right after the bindings.
        assert_eq!(seq[2 * m.n_pairs], SEP);
        assert_eq!(*seq.last().unwrap(), NL);
        assert_eq!(answers.len(), m.n_queries);
        // Each recorded answer position holds the recorded answer token, and the
        // token just before it (the query key) appeared earlier as a binding key.
        let key_positions: Vec<u16> = (0..m.n_pairs).map(|i| seq[2 * i]).collect();
        for &(pos, tok) in &answers {
            assert_eq!(seq[pos], tok);
            assert!(key_positions.contains(&seq[pos - 1]));
        }
    }

    #[test]
    fn distinct_indices_are_distinct_and_in_range() {
        let mut rng = Rng::new(5);
        let idx = sample_distinct_indices(4, 6, &mut rng);
        assert_eq!(idx.len(), 4);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
        assert!(idx.iter().all(|&i| i < 6));
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = Mqar { n_sequences: 100, ..Mqar::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }
}
