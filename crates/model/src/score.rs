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

/// Fill in the tuned loss of each of `scored` on the model the run returned.
pub fn score_tuned<M: Model>(m: &M, sets: &[&Path], opts: &FitOpts, scored: &mut [ScoredSet]) -> std::io::Result<()> {
    for (set, slot) in sets.iter().zip(scored.iter_mut()) {
        slot.tuned_loss = score_dataset(m, set, opts)?;
    }
    Ok(())
}
