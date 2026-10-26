// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring whether a
// fine-tuned adapter scores its own training records the way the trainer
// reports them, for its clients. If your team needs expertise in per-token
// evaluation of language-model fine-tunes then you can procure our services by
// sending an email to info@swedishembedded.com.

// Scoring is the `study` surface.
#![cfg(feature = "study")]

//! SPEC: `brain::score_tokens_qwen35` - the in-process side of a train-serve
//! agreement check - on a `qwen35` base at toy width, over the CPU backend.
//!
//! * It scores exactly the positions the trainer supervises: for a record
//!   whose supervised turn closes the conversation, the run of positions that
//!   ends at its last token.
//! * Its mean log-probability is the mean loss `ChatFineTune` reports on the
//!   same records with the same base tier and no adapter: one quantity, read
//!   two ways.
//! * With the adapter a fine-tune wrote, it scores the same positions from the
//!   adapted model, whose records the adapter trained on score better.

use std::path::{Path, PathBuf};

use brain::{score_tokens_qwen35, ChatFineTune};
use model::adapter::device::AdapterTuning;

#[path = "chat_fixture/common.rs"]
mod common;
use common::{cpu, qwen35_model_dir, write_chat_dataset};

/// The records both sides score: two question/answer pairs, the answer
/// supervised.
fn records(dir: &Path) -> PathBuf {
    write_chat_dataset(dir, &[("cab", "bad"), ("fed", "deaf")])
}

/// A short fine-tune at the bf16 base tier (the tier the readout builds, so its
/// reported mean loss and the readout are the same model), scoring the records
/// it trains on before its first step.
fn fine_tune(model: &Path, records: &Path) -> ChatFineTune {
    ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35")
        .dataset(records)
        .held_out(records)
        .out_dir(model.join("run"))
        .adapter_id("local/tiny:lab:readout:v1")
        .bf16_base(true)
        .rank(2)
        .steps(4)
        .lr(1e-2)
        .eval_every(0)
        // The base score this test compares against is the zero-delta adapter's, and two runs of the comparison must see the same stream.
        // seed: the two compared runs share one stream
        .seed(7)
        .device(cpu())
}

/// SPEC: the readout's mean log-probability is the mean loss the trainer
/// reports on the same records, and the positions are the supervised turn's.
#[test]
fn the_readout_is_the_mean_loss_the_trainer_reports_on_the_same_records() {
    let model = qwen35_model_dir("q35-readout-loss");
    let records = records(&model);
    let base_loss = fine_tune(&model, &records)
        .run()
        .expect("the fine-tune runs")
        .base_score
        .expect("the base is scored on the held-out records")
        .loss
        .expect("positions were scored");

    let readout = score_tokens_qwen35(model.join("model.safetensors").to_str().unwrap(), None, &records, AdapterTuning::default(), &cpu()).expect("the records score");
    assert_eq!(readout.records.len(), 2, "one readout per record");
    for r in &readout.records {
        assert!(r.positions.len() > 1, "record {} supervises {} positions", r.record, r.positions.len());
        let first = r.positions.first().expect("non-empty").position as usize;
        let last = r.positions.last().expect("non-empty").position as usize;
        // The supervised answer closes the conversation, so its positions are a
        // run that ends on the record's last token.
        assert_eq!(last, r.tokens - 1, "record {}: the readout ends at the record's last token", r.record);
        assert_eq!(r.positions.len(), last - first + 1, "record {}: positions {}..={last} are contiguous", r.record, first);
        assert!(first >= 1, "nothing predicts a record's first token");
        for p in &r.positions {
            assert!(p.logprob <= 0.0 && p.logprob.is_finite(), "record {} position {}: log-probability {}", r.record, p.position, p.logprob);
            assert!((p.argmax as usize) < 151_936, "record {} position {}: argmax {} outside the vocabulary", r.record, p.position, p.argmax);
        }
    }
    let nll = -readout.mean_logprob.expect("positions were scored") as f64;
    let tolerance = 1e-4 * (base_loss.abs().max(1.0) as f64);
    assert!((nll - base_loss as f64).abs() < tolerance, "the trainer's mean loss {base_loss} vs the readout's mean NLL {nll}");
    assert_eq!(readout.positions, readout.records.iter().map(|r| r.positions.len()).sum::<usize>());
}

/// SPEC: the adapter a fine-tune wrote is loaded by the readout, which scores
/// the same positions from the adapted model and finds its training records
/// likelier than the base does.
#[test]
fn the_adapter_a_fine_tune_wrote_is_the_model_the_readout_scores() {
    let model = qwen35_model_dir("q35-readout-adapter");
    let records = records(&model);
    let adapter = fine_tune(&model, &records).run().expect("the fine-tune runs").adapter.expect("a completed run exports its adapter");

    let base = score_tokens_qwen35(model.join("model.safetensors").to_str().unwrap(), None, &records, AdapterTuning::default(), &cpu()).expect("the base scores");
    let tuned = score_tokens_qwen35(model.join("model.safetensors").to_str().unwrap(), Some(&adapter), &records, AdapterTuning::default(), &cpu()).expect("the adapter scores");

    for (b, t) in base.records.iter().zip(&tuned.records) {
        assert_eq!(b.tokens, t.tokens);
        assert_eq!(b.positions.iter().map(|p| p.position).collect::<Vec<_>>(), t.positions.iter().map(|p| p.position).collect::<Vec<_>>(), "record {}: the same positions under the adapter", b.record);
    }
    let (b, t) = (base.mean_logprob.expect("scored"), tuned.mean_logprob.expect("scored"));
    assert!(t > b, "records it trained on score better after training: {b} -> {t}");
}
