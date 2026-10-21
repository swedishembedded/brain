// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that every model in
// a training stack honours the same fine-tuning contract, for its clients. If
// your team needs expertise in memory-lean language-model heads then you can
// procure our services by sending an email to info@swedishembedded.com.

//! C8 of the conformance suite ([`crate::conformance`]): a chunked LM-head
//! cross-entropy equals the unchunked one.
//!
//! The head keeps `rows_per_chunk` rows of logits alive and recomputes them
//! in the backward ([`crate::head_ce`]); that must be invisible. The subject
//! builds the same model twice from one seed, once with a head budget that
//! holds the whole row and once with a budget of 64 rows of logits, and the
//! loss, the per-position log-probabilities and every trainable gradient
//! must agree. The weighted loss every post-training objective (DPO, GRPO,
//! distillation) trains through is held to the same check, so none of them
//! holds the logits of a whole row either.

use crate::conformance::{ChunkedHead, Subject};
use crate::conformance_grad::randomize_adapters;
use crate::training::require_token_objectives;
use crate::{Batch, Model, IGNORE};

/// Rows per chunk the check forces: the smallest legal chunk (chunk starts
/// are 256-byte binding offsets, so whole groups of 64).
const CHUNK_ROWS: u64 = 64;
/// Loss, log-probability and gradient agreement, relative to the largest value.
const LOSS_TOL: f32 = 1e-5;
const GRAD_TOL: f32 = 1e-4;

struct Observed {
    loss: f32,
    logprobs: Vec<f32>,
    grads: Vec<(String, Vec<f32>)>,
}

/// One position's loss weight of the weighted check: zero, fractional and above one all occur.
fn weight_at(i: u32) -> f32 {
    [0.0, 0.25, 1.0, 2.5, 0.75][(i % 5) as usize]
}

fn observe<M: Model>(m: &M, batch: Batch) -> Result<Observed, String> {
    m.set_batch(batch);
    m.zero_grads();
    let loss = m.forward();
    let logprobs = m.token_objectives().ok_or("the model offers no token objectives")?.batch_token_logprobs();
    m.backward();
    m.poll_wait();
    let grads = m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    Ok(Observed { loss, logprobs, grads })
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// C8 for the subject's [`ChunkedHead`] hook, for the plain and the weighted
/// loss; a subject without one is asserted to need none (it declares no
/// chunked head).
pub fn chunked_head_ce_equals_unchunked<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let Some(ChunkedHead { seq, build }) = &subject.chunked_head else { return Ok(()) };
    let rank = crate::conformance::trained_rank(subject);
    let vocab = crate::conformance_grad::subject_vocab(subject);
    // Three full chunks and a partial one; every fifth target ignored, so the
    // mask crosses chunk boundaries.
    let tokens: Vec<u32> = (0..*seq).map(|i| (i * 7 + 1) % vocab).collect();
    let targets: Vec<u32> = (0..*seq).map(|i| if i % 5 == 3 { IGNORE } else { (i * 7 + 2) % vocab }).collect();
    let weights: Vec<f32> = (0..*seq).map(weight_at).collect();

    for weighted in [false, true] {
        let loss = if weighted { "weighted " } else { "" };
        let run = |budget_words: u64| -> Result<(M, Observed), String> {
            let mut m = build(rank, budget_words, seed);
            randomize_adapters(&m, seed);
            let batch = if weighted {
                require_token_objectives(&mut m).map_err(|e| format!("{}: {e}", subject.name))?.enable_weighted_loss();
                Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights }
            } else {
                Batch::Lm { tokens: &tokens, targets: &targets }
            };
            let seen = observe(&m, batch).map_err(|e| format!("{}: {e}", subject.name))?;
            Ok((m, seen))
        };
        let (whole_model, whole) = run(u64::MAX)?;
        let (chunked_model, chunked) = run(CHUNK_ROWS * vocab as u64)?;
        let at = format!("{}: {loss}", subject.name);

        if (chunked.loss - whole.loss).abs() > LOSS_TOL * whole.loss.abs().max(1.0) {
            return Err(format!("{at}chunked head loss {} vs unchunked {}", chunked.loss, whole.loss));
        }
        // The comparison is only meaningful if one head really chunked and the other did not.
        let rows = |m: &M| m.token_objectives().and_then(|o| o.head_chunk_rows());
        match (rows(&whole_model), rows(&chunked_model)) {
            (Some(whole), Some(chunked)) if whole >= *seq && chunked < *seq => {}
            other => return Err(format!("{at}the unchunked head must hold the whole {seq}-token row and the chunked one fewer rows, got {other:?}")),
        }
        let rel = max_rel(&chunked.logprobs, &whole.logprobs);
        if rel > LOSS_TOL {
            return Err(format!("{at}chunked head log-probabilities differ by {rel:e} (relative)"));
        }
        for ((name, got), (_, want)) in chunked.grads.iter().zip(&whole.grads) {
            let rel = max_rel(got, want);
            if rel > GRAD_TOL {
                return Err(format!("{at}{name}: chunked head gradient differs by {rel:e} (relative)"));
            }
        }
    }
    Ok(())
}
