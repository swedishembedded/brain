// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring whether a fine-tune
// generalised, on the model already resident in the training process, for its
// clients. If your team needs expertise in held-out evaluation of large
// language model training then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Held-out scoring over any [`Model`]: the mean cross-entropy of a model over
//! every supervised position of a prepared chat dataset, measured with the
//! training objective's own evaluation so a score and a monitoring curve are
//! the same quantity.
//!
//! Scoring runs on the model a trainer already holds, because a second copy of
//! a large base does not fit beside it. The base score of a run is the score
//! taken before the first step (an adapter with a zero `B` leaves the base's
//! logits unchanged); the tuned score is taken on the model the run returns.

use std::path::Path;

use crate::train::{causal_lm, load_dataset_with_itos, FitOpts};
use crate::{Model, Objective};

/// A held-out record set scored in the trainer's own process: the loss of the
/// untouched base (`None` when the run continues an adapter, whose starting
/// point is not the base) and of the model the run hands back, over exactly
/// the positions training supervises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredSet {
    pub base_loss: Option<f32>,
    pub tuned_loss: Option<f32>,
    /// Supervised positions scored, and the records they come from.
    pub positions: usize,
    pub records: usize,
}

/// The mean cross-entropy of `m` over every supervised position of the
/// prepared dataset in `set` (its validation split). The scoring row is the
/// training row: `opts.block_size` must hold the longest example. `None` when
/// the split holds nothing to score.
pub fn score_dataset<M: Model>(m: &M, set: &Path, opts: &FitOpts) -> std::io::Result<Option<f32>> {
    let (train, records, cfg, _, itos) = load_dataset_with_itos(set, opts)?;
    let mut obj = causal_lm::<M>(train, records, cfg, itos);
    Ok(obj.eval(m, &mut data::rng::Rng::new(data::rng::random_seed()), 0))
}

/// `(records, supervised positions)` of the validation split of the prepared
/// dataset in `set`, counted the way the evaluation walks it.
pub fn supervised_positions(set: &Path, opts: &FitOpts) -> std::io::Result<(usize, usize)> {
    let (_, records, cfg, _, _) = load_dataset_with_itos(set, opts)?;
    let count = records.example_count().unwrap_or(0);
    let rows = cfg.batch_size.max(1);
    let mut positions = 0;
    let mut first = 0;
    while first < count {
        let Some((_, y)) = records.example_rows(&cfg, first) else { break };
        positions += y.iter().filter(|&&t| t != data::loader::IGNORE).count();
        first += rows;
    }
    Ok((count, positions))
}

/// A [`ScoredSet`] for each of `sets` before any training: the base's loss
/// unless the run continues an adapter (`continuing`).
pub fn score_base<M: Model>(m: &M, sets: &[&Path], opts: &FitOpts, continuing: bool) -> std::io::Result<Vec<ScoredSet>> {
    sets.iter()
        .map(|set| {
            let (records, positions) = supervised_positions(set, opts)?;
            let base_loss = if continuing { None } else { score_dataset(m, set, opts)? };
            Ok(ScoredSet { base_loss, tuned_loss: None, positions, records })
        })
        .collect()
}

/// One document scored for bits per byte: `tokens[0]` is a context-only
/// separator, `tokens[1..]` the document's tokens, `bytes` its UTF-8 length.
#[derive(Clone, Debug, PartialEq)]
pub struct ByteDoc {
    pub tokens: Vec<u32>,
    pub bytes: usize,
}

/// The summed next-token negative log-likelihood of a document set and what
/// it was summed over.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ByteScore {
    /// Total negative log-likelihood, nats.
    pub nats: f64,
    pub bytes: u64,
    /// Predicted tokens.
    pub tokens: u64,
    pub documents: u64,
}

impl ByteScore {
    /// Bits per UTF-8 byte: independent of the tokenizer, so models with
    /// different vocabularies are compared on the same text.
    pub fn bits_per_byte(&self) -> f64 {
        self.nats / std::f64::consts::LN_2 / self.bytes as f64
    }
}

/// The documents of a TinyStories-style text (split by
/// `data::prepare::split_documents`) whose UTF-8 length is at most
/// `max_bytes`, each tokenized by `tok` and preceded by `separator`. The
/// length rule is on bytes, so every tokenizer is scored on the same subset.
pub fn byte_docs(text: &str, tok: &dyn data::tokenizer::Tokenizer, separator: u32, max_bytes: usize) -> Vec<ByteDoc> {
    data::prepare::split_documents(text)
        .filter(|d| d.len() <= max_bytes)
        .map(|d| {
            let mut tokens = vec![separator];
            tokens.extend(tok.encode(d));
            ByteDoc { tokens, bytes: d.len() }
        })
        .collect()
}

/// Score `docs` with `m`, built for `b` rows of `t` positions: one document per
/// row, every one of its tokens predicted from the separator and the tokens
/// before it, padding unsupervised. A document longer than `t` is refused,
/// not truncated - a caller selects documents that fit by a rule that does
/// not depend on any one tokenizer.
pub fn score_bytes<M: Model>(m: &M, docs: &[ByteDoc], b: u32, t: u32) -> Result<ByteScore, String> {
    let (b, t) = (b as usize, t as usize);
    let ignore = crate::train::IGNORE;
    let mut score = ByteScore::default();
    for chunk in docs.chunks(b) {
        let mut x = vec![0u32; b * t];
        let mut y = vec![ignore; b * t];
        for (row, doc) in chunk.iter().enumerate() {
            if doc.tokens.len() < 2 || doc.tokens.len() > t {
                return Err(format!("document of {} tokens does not fit a {t}-position row with a separator", doc.tokens.len()));
            }
            x[row * t..row * t + doc.tokens.len()].copy_from_slice(&doc.tokens);
            y[row * t..row * t + doc.tokens.len() - 1].copy_from_slice(&doc.tokens[1..]);
            score.tokens += doc.tokens.len() as u64 - 1;
            score.bytes += doc.bytes as u64;
            score.documents += 1;
        }
        let supervised = y.iter().filter(|&&v| v != ignore).count();
        m.set_batch(crate::Batch::Lm { tokens: &x, targets: &y });
        score.nats += m.forward() as f64 * supervised as f64;
    }
    Ok(score)
}

/// Fill in the tuned loss of each of `scored` on the model the run returned.
pub fn score_tuned<M: Model>(m: &M, sets: &[&Path], opts: &FitOpts, scored: &mut [ScoredSet]) -> std::io::Result<()> {
    for (set, slot) in sets.iter().zip(scored.iter_mut()) {
        slot.tuned_loss = score_dataset(m, set, opts)?;
    }
    Ok(())
}
