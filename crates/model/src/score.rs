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

use crate::train::{causal_lm, load_dataset_with_itos, FitOpts, GradSync};
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
/// the split holds nothing to score. The score is the plain, unweighted
/// cross-entropy even when the set carries per-token loss weights: it measures
/// the model, not the run's training objective.
pub fn score_dataset<M: Model>(m: &M, set: &Path, opts: &FitOpts) -> std::io::Result<Option<f32>> {
    let (train, records, cfg, _, itos) = load_dataset_with_itos(set, opts)?;
    let mut obj = causal_lm::<M>(train.without_weights(), records.without_weights(), cfg, itos);
    Ok(obj.eval(m, &mut data::rng::Rng::new(data::rng::random_seed()), 0))
}

/// Every example of a prepared split as owned `(tokens, targets)` buffers,
/// `cfg.batch_size` examples a buffer (padding rows past the last example),
/// targets in the model's `u32` form ([`crate::IGNORE`] where unsupervised).
pub fn example_rows(split: &data::loader::TokenDataset, cfg: &data::loader::BatchConfig) -> Vec<(Vec<u32>, Vec<u32>)> {
    let count = split.example_count().unwrap_or(0);
    let mut out = Vec::new();
    let mut first = 0;
    while first < count {
        let Some((x, y)) = split.example_rows(cfg, first) else { break };
        out.push((x, crate::train::targets_to_u32(&y)));
        first += cfg.batch_size.max(1);
    }
    out
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
///
/// In a data-parallel run (`sync`) every rank holds the same model, so rank 0
/// scores and hands every rank its numbers ([`agreed_scores`]); a run whose
/// parameters are sharded across the ranks scores on every rank, since its
/// forward needs them all.
pub fn score_base<M: Model>(m: &M, sets: &[&Path], opts: &FitOpts, continuing: bool, sync: Option<&GradSync<'_>>) -> std::io::Result<Vec<ScoredSet>> {
    agreed_scores(sync, opts, sets.len(), || {
        sets.iter()
            .map(|set| {
                let (records, positions) = supervised_positions(set, opts)?;
                let base_loss = if continuing { None } else { score_dataset(m, set, opts)? };
                Ok(ScoredSet { base_loss, tuned_loss: None, positions, records })
            })
            .collect()
    })
}

/// `score()` on rank 0 only, its sets broadcast to every rank of `sync`'s group, when the replicas hold the same
/// model; on every rank (rank 0's result agreed) when `opts` shards the parameters, whose forward gathers them over
/// the group; plainly without a group. A rank 0 that fails to score says so to the others, which fail with it
/// instead of waiting.
pub fn agreed_scores(sync: Option<&GradSync<'_>>, opts: &FitOpts, sets: usize, score: impl FnOnce() -> std::io::Result<Vec<ScoredSet>>) -> std::io::Result<Vec<ScoredSet>> {
    let Some(sync) = sync else { return score() };
    let every_rank = opts.sharding == crate::zero::Sharding::Parameters;
    let local = (every_rank || sync.rank == 0).then(score);
    let offer = match &local {
        Some(Ok(scored)) if sync.rank == 0 => std::iter::once(1).chain(scored.iter().flat_map(set_words)).collect(),
        Some(Err(_)) if sync.rank == 0 => vec![0],
        _ => Vec::new(),
    };
    let words = crate::collective::broadcast_words(sync.collective, sync.rank, &offer, 0).map_err(std::io::Error::other)?;
    if let Some(Err(e)) = local {
        return Err(e);
    }
    match words.split_first() {
        Some((&1, sets_words)) if sets_words.len() == sets * SET_WORDS => Ok(sets_words.chunks(SET_WORDS).map(set_of_words).collect()),
        Some((&0, [])) => Err(std::io::Error::other("rank 0 could not score the held-out sets")),
        _ => Err(std::io::Error::other(format!("rank 0 sent {} words of scores for {sets} sets", words.len()))),
    }
}

/// The words one [`ScoredSet`] travels as.
const SET_WORDS: usize = 4;

/// A loss as a word: its bits, or a value no f32's bits are for `None`.
fn loss_word(loss: Option<f32>) -> u64 {
    loss.map_or(1 << 32, |l| u64::from(l.to_bits()))
}

fn word_loss(word: u64) -> Option<f32> {
    u32::try_from(word).ok().map(f32::from_bits)
}

fn set_words(set: &ScoredSet) -> [u64; SET_WORDS] {
    [loss_word(set.base_loss), loss_word(set.tuned_loss), set.positions as u64, set.records as u64]
}

fn set_of_words(words: &[u64]) -> ScoredSet {
    ScoredSet { base_loss: word_loss(words[0]), tuned_loss: word_loss(words[1]), positions: words[2] as usize, records: words[3] as usize }
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

impl std::ops::AddAssign for ByteScore {
    fn add_assign(&mut self, o: ByteScore) {
        self.nats += o.nats;
        self.bytes += o.bytes;
        self.tokens += o.tokens;
        self.documents += o.documents;
    }
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

/// [`score_bytes`] over documents grouped by length: each group of
/// documents whose token count fits `len` (one of `lengths`, ascending) is
/// scored on a model `build(len)` makes for `b` rows of `len` positions, so a
/// short document is not padded to the longest one. Every document is scored
/// alone in its row, so the result equals one [`score_bytes`] over all of
/// them up to the order of the floating-point sums. A document longer than
/// the last length is refused.
pub fn score_bytes_bucketed<M: Model>(docs: &[ByteDoc], b: u32, lengths: &[u32], mut build: impl FnMut(u32) -> M) -> Result<ByteScore, String> {
    let mut groups: Vec<Vec<ByteDoc>> = vec![Vec::new(); lengths.len()];
    for d in docs {
        let Some(i) = lengths.iter().position(|&l| d.tokens.len() as u32 <= l) else {
            return Err(format!("document of {} tokens exceeds the longest bucket {}", d.tokens.len(), lengths.last().copied().unwrap_or(0)));
        };
        groups[i].push(d.clone());
    }
    let mut score = ByteScore::default();
    for (len, group) in lengths.iter().zip(&groups) {
        if !group.is_empty() {
            score += score_bytes(&build(*len), group, b, *len)?;
        }
    }
    Ok(score)
}

/// Fill in the tuned loss of each of `scored` on the model the run returned, agreed across `sync`'s group as
/// [`score_base`] does.
pub fn score_tuned<M: Model>(m: &M, sets: &[&Path], opts: &FitOpts, scored: &mut [ScoredSet], sync: Option<&GradSync<'_>>) -> std::io::Result<()> {
    let tuned = agreed_scores(sync, opts, scored.len(), || {
        let mut tuned = scored.to_vec();
        for (set, slot) in sets.iter().zip(tuned.iter_mut()) {
            slot.tuned_loss = score_dataset(m, set, opts)?;
        }
        Ok(tuned)
    })?;
    scored.copy_from_slice(&tuned);
    Ok(())
}

#[cfg(test)]
mod agreed_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn set(loss: f32) -> ScoredSet {
        ScoredSet { base_loss: Some(loss), tuned_loss: None, positions: 70_000_001, records: 3 }
    }

    /// Two ranks of a host group each calling [`agreed_scores`] with `opts`; how many times each one scored, and what
    /// each one got.
    fn two_ranks(opts: &FitOpts, rank0: std::io::Result<Vec<ScoredSet>>) -> (Vec<usize>, Vec<std::io::Result<Vec<ScoredSet>>>) {
        let group = crate::HostCollective::new(2);
        let calls: Vec<AtomicUsize> = (0..2).map(|_| AtomicUsize::new(0)).collect();
        let rank0 = std::sync::Mutex::new(Some(rank0));
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|rank| {
                    let (group, calls, rank0) = (&group, &calls, &rank0);
                    scope.spawn(move || {
                        let sync = GradSync { collective: &**group, rank, checkpoint: None, observer: None };
                        agreed_scores(Some(&sync), opts, 1, || {
                            calls[rank].fetch_add(1, Ordering::SeqCst);
                            match rank {
                                0 => rank0.lock().unwrap().take().unwrap(),
                                _ => Ok(vec![set(9.0)]),
                            }
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        (calls.iter().map(|c| c.load(Ordering::SeqCst)).collect(), results)
    }

    #[test]
    fn replicas_score_once_on_rank_zero_and_every_rank_gets_its_exact_numbers() {
        let (calls, results) = two_ranks(&FitOpts::default(), Ok(vec![set(0.123_456_79)]));
        assert_eq!(calls, vec![1, 0], "a replica does not repeat rank 0's scoring");
        for r in results {
            assert_eq!(r.unwrap(), vec![set(0.123_456_79)], "every rank holds rank 0's numbers, bit for bit, counts past f32's exact range included");
        }
    }

    #[test]
    fn a_run_with_sharded_parameters_scores_on_every_rank_and_agrees_on_rank_zeros() {
        let opts = FitOpts { sharding: crate::zero::Sharding::Parameters, ..Default::default() };
        let (calls, results) = two_ranks(&opts, Ok(vec![set(1.5)]));
        assert_eq!(calls, vec![1, 1], "a sharded forward needs every rank");
        for r in results {
            assert_eq!(r.unwrap(), vec![set(1.5)]);
        }
    }

    #[test]
    fn a_rank_zero_that_cannot_score_fails_the_others_instead_of_leaving_them_waiting() {
        let (_, results) = two_ranks(&FitOpts::default(), Err(std::io::Error::other("the set is unreadable")));
        assert!(results[0].as_ref().unwrap_err().to_string().contains("unreadable"));
        assert!(results[1].as_ref().unwrap_err().to_string().contains("rank 0 could not score"));
    }
}
