// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Re-recording a dispatch graph is host work, and a decision loop pays it per
//! decision.
//!
//! `set_batch` hands the device a new request. Two graphs describe what runs
//! on it, and they depend on different parts of that request: the forward
//! bakes in the row count and the span table, the reverse additionally bakes
//! in how many DISTINCT ids the embedding gradient has to scatter into. A
//! decision loop changes the ids at every step and the span layout only
//! sometimes, so conflating the two conditions re-records the forward for a
//! shape it already has - and re-records the reverse for a pass that, with a
//! frozen encoder, is never going to run.
//!
//! Measured on a DOOM policy with a frozen encoder, that was 78 ms of the
//! 78-plus-game milliseconds a decision cost inside a training process,
//! against 20 ms for the same weights in a process that had not trained.
//! Nothing about the numbers the model produces changes; what changes is
//! whether an hour of probing is an hour of arithmetic.

use decide::config::EncoderConfig;
use decide::kern::PIPELINES;
use decide::model::Encoder;

/// Spans of different lengths, because "the shape did not change" has to mean
/// the whole table and not just the row count.
const SPANS: &[(u32, u32)] = &[(0, 7), (7, 5), (12, 6)];

fn types(cfg: &EncoderConfig) -> Vec<u32> {
    let _ = cfg;
    SPANS
        .iter()
        .enumerate()
        .flat_map(|(i, &(_, l))| std::iter::repeat_n((i % 2) as u32, l as usize))
        .collect()
}

/// Ids covering `SPANS`, drawn so that successive calls differ in how many
/// DISTINCT ones there are - which is the reverse pass's own parameter, and
/// the only part of a batch that moves it without moving the forward's.
fn ids(cfg: &EncoderConfig, round: u32) -> Vec<u32> {
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let distinct = (3 + round).min(cfg.vocab - 1);
    (0..rows).map(|i| 1 + i % distinct).collect()
}

fn encoder() -> (Encoder, EncoderConfig) {
    let cfg = EncoderConfig::tiny();
    let init = decide::init::init_weights(&cfg, 7);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().expect("SPANS is not empty");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let enc = Encoder::new_train_on(gpu, cfg.clone(), rows, max_span, &init);
    (enc, cfg)
}

/// New tokens at a shape the device has already been given are not a new
/// forward graph.
#[test]
fn the_forward_graph_is_recorded_once_per_batch_shape() {
    let (mut enc, cfg) = encoder();
    let t = types(&cfg);
    enc.set_batch(&ids(&cfg, 0), &t, SPANS);
    let (first, _) = enc.graph_records();
    for round in 1..8 {
        enc.set_batch(&ids(&cfg, round), &t, SPANS);
    }
    assert_eq!(
        enc.graph_records().0,
        first,
        "seven more batches at the same shape re-recorded the forward graph"
    );

    // And a shape it has NOT been given is: the saving must come from
    // recognising the repeat, not from never recording again.
    let short: &[(u32, u32)] = &[(0, 7), (7, 5)];
    let rows: u32 = short.iter().map(|&(_, l)| l).sum();
    let short_ids: Vec<u32> = (0..rows).map(|i| i % cfg.vocab).collect();
    let short_types = vec![0u32; rows as usize];
    enc.set_batch(&short_ids, &short_types, short);
    assert_eq!(enc.graph_records().0, first + 1, "a new span table is a new forward graph");
}

/// Setting a batch does not record a reverse pass. This is the one that costs
/// a training run its wall clock: every decision of every rollout, DAgger
/// round and counterfactual probe sets a batch, and not one of them runs a
/// backward.
#[test]
fn setting_a_batch_does_not_record_a_reverse_pass() {
    let (mut enc, cfg) = encoder();
    let t = types(&cfg);
    enc.set_batch(&ids(&cfg, 0), &t, SPANS);
    let (_, before) = enc.graph_records();
    for round in 1..8 {
        enc.set_batch(&ids(&cfg, round), &t, SPANS);
    }
    assert_eq!(
        enc.graph_records().1,
        before,
        "eight batches nobody differentiated recorded a reverse pass"
    );
}

/// And the pass that runs is never one recorded for a different batch. That
/// is the failure the deferral could buy, and it would not look like a bug -
/// it would look like a gradient.
#[test]
fn preparing_records_exactly_what_the_deferral_skipped() {
    let (mut enc, cfg) = encoder();
    let t = types(&cfg);
    enc.set_batch(&ids(&cfg, 0), &t, SPANS);
    for round in 1..4 {
        enc.set_batch(&ids(&cfg, round), &t, SPANS);
    }
    let (_, deferred) = enc.graph_records();
    enc.prepare_reverse();
    assert_eq!(
        enc.graph_records().1,
        deferred + 1,
        "the reverse graph was left describing a batch that is no longer set"
    );

    // Once, though: a second caller on the same batch inherits the first
    // one's recording rather than repeating it.
    enc.prepare_reverse();
    assert_eq!(enc.graph_records().1, deferred + 1, "an unchanged batch was recorded twice");
}

/// A batch the reverse pass has not been prepared for is refused rather than
/// run. Without this the saving above is only safe as long as every caller
/// remembers, and the cost of forgetting is silently wrong gradients.
#[test]
#[should_panic(expected = "prepare_reverse")]
fn a_reverse_pass_will_not_run_against_a_batch_it_does_not_describe() {
    let (mut enc, cfg) = encoder();
    let t = types(&cfg);
    enc.set_batch(&ids(&cfg, 1), &t, SPANS);
    enc.backward_seeded();
}
