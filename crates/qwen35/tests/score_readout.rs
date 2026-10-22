// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring whether a
// fine-tuned adapter scores its own training records the way the trainer says
// it does, for its clients. If your team needs expertise in per-token
// evaluation of language-model fine-tunes then you can procure our services by
// sending an email to info@swedishembedded.com.

//! SPEC: the per-token readout of a record set is the trainer's own forward,
//! position by position.
//!
//! Two properties, both on one tiny CPU model with the chunked head the real
//! 27B trains through:
//!
//! * the mean of the per-token NLLs the readout reports is the mean loss the
//!   trainer reports for the same records, so a score and a held-out loss are
//!   the same quantity;
//! * every argmax the readout reports is the pick the model's own logits show
//!   at that position (read back independently through `Qwen35::logits_all`),
//!   so the readout is the model's choice and not an artefact of the
//!   reduction.

mod common;

use common::*;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};
use qwen35::score::{score_records, ScoredRecord};

const BLOCK: usize = 128;

fn config() -> Qwen35Config {
    Qwen35Config { block_size: BLOCK as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() }
}

/// A training build with the per-token argmax readout on, on the CPU backend.
fn model(init: &std::collections::HashMap<String, Vec<f32>>) -> Qwen35 {
    let mut m = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), config(), 1, BLOCK as u32, init, &TrainOpts::default());
    m.enable_token_argmax().expect("the CPU kernel set registers argmax_part/argmax_final");
    m
}

/// A record of `len` tokens over the tiny vocabulary with a non-contiguous
/// supervision mask (prompt-style: the first `len / 3` tokens unsupervised).
fn record(len: usize, salt: usize) -> ScoredRecord {
    let vocab = config().vocab as usize;
    let ids: Vec<u32> = (0..len).map(|i| ((i * 7 + salt * 3 + 1) % vocab) as u32).collect();
    let mask: Vec<bool> = (0..len).map(|i| i >= len / 3 && i % 5 != 0).collect();
    ScoredRecord { ids, mask }
}

/// The `(x, y)` a training step sees for one record, padded to [`BLOCK`]: the
/// same shift `data::loader::TokenDataset` applies (`y[t] = ids[t + 1]` where
/// the mask supervises that token).
fn training_row(record: &ScoredRecord) -> (Vec<u32>, Vec<u32>) {
    let mut x = vec![0u32; BLOCK];
    let mut y = vec![model::IGNORE; BLOCK];
    for t in 0..record.ids.len() - 1 {
        x[t] = record.ids[t];
        if record.mask[t + 1] {
            y[t] = record.ids[t + 1];
        }
    }
    (x, y)
}

/// SPEC: `sum(per-token NLL) / positions` is the mean loss the trainer reports
/// for the same records.
#[test]
fn the_mean_of_the_readouts_is_the_loss_the_trainer_reports() {
    let m = model(&weights(&config(), 0.3));
    let rec = record(40, 11);
    let scored = score_records(&m, std::slice::from_ref(&rec), 1, BLOCK as u32).expect("the record scores");
    let want: Vec<u32> = rec.mask.iter().enumerate().skip(1).filter(|(_, &s)| s).map(|(p, _)| p as u32).collect();
    assert_eq!(scored.records[0].iter().map(|s| s.position).collect::<Vec<_>>(), want, "one readout per supervised position past the first");
    assert_eq!(scored.positions, want.len());

    let (x, y) = training_row(&rec);
    m.set_batch(&x, &y);
    let loss = m.forward();
    let nll: f64 = scored.records[0].iter().map(|s| -(s.logprob as f64)).sum();
    let mean = nll / want.len() as f64;
    assert!((loss as f64 - mean).abs() < 1e-4 * mean.abs().max(1.0), "trainer loss {loss} vs the readout's mean NLL {mean}");
    assert!((scored.mean_loss.expect("positions were scored") as f64 - mean).abs() < 1e-4 * mean.abs().max(1.0), "the pass reports the same mean");
}

/// SPEC: every reported argmax is the model's own pick at that position.
#[test]
fn every_argmax_is_the_pick_the_models_own_logits_show() {
    let m = model(&weights(&config(), 0.3));
    let rec = record(40, 5);
    let scored = score_records(&m, std::slice::from_ref(&rec), 1, BLOCK as u32).expect("the record scores");
    let vocab = config().vocab as usize;
    // An independent read of the same positions: the full logits rows.
    let logits = m.logits_all(&rec.ids);
    assert_eq!(logits.len(), rec.ids.len() * vocab);
    for s in &scored.records[0] {
        let row = &logits[(s.position as usize - 1) * vocab..s.position as usize * vocab];
        assert_eq!(s.argmax, model::hostmath::argmax_row(row), "argmax at position {}", s.position);
        // A readout is a log-probability, never a probability or a loss.
        assert!(s.logprob <= 0.0 && s.logprob.is_finite(), "log-probability {} at position {}", s.logprob, s.position);
    }
}
