// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring the agreement between
// a 27B adapter as trained and the same adapter as served, for its clients. If
// your team needs expertise in train-serve agreement of quantised serving
// stacks then you can procure our services by sending an email to
// info@swedishembedded.com.

//! SPEC: teacher-forced scoring on the served path (`Qwen35GgufEngine::
//! score_supervised`) is the served model's own prediction at every position,
//! with and without an adapter, on a 2-layer random-weight GGUF that has the
//! real tensor names and layer kinds.
//!
//! The reference is `Qwen35GgufInstance::teacher_forced_logits`, the
//! pre-existing decode-tape read of the same stream: a score must be that
//! row's log-softmax at the recorded token, and its argmax that row's argmax.
//! The two tapes (one chunked prefill against a prefill plus per-token decode
//! steps) are the model's own numerics floor, so a position is only held to
//! its argmax when the reference row has a margin to call it - a flip at a
//! near-tie is drift, a flip at a large margin is a defect.

use gpu_core::select::Dtype;
use model::adapter::bank::{AdapterBankRequest, AdapterTargets, VerifiedAdapter};
use model::adapter::sites::LoraSites;
use model::lora::device_adapter::{AdapterSite, DeviceAdapter};
use model::ops::TierPolicy;
use model::paged::BlockTable;
use model::serve::{AdapterId, PagedDecoder};
use qwen35::int8_gguf_resident::{budgeted_gpus, Qwen35GgufEngine, Qwen35GgufResident};

const CAP: u32 = 256;

fn engine() -> Qwen35GgufEngine {
    let path = std::env::temp_dir().join(format!("brain-gguf-score-{}.gguf", std::process::id())).to_string_lossy().into_owned();
    qwen35::gguf_import::testing::write_random_servable_gguf(&path, 7);
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("this check needs a GPU").0;
    let bank = AdapterBankRequest { slots: 2, rank_cap: 4, targets: AdapterTargets::ALL, max_registered: 4 };
    let resident = Qwen35GgufResident::new(path.clone(), devices, CAP, TierPolicy::uniform(Dtype::I8)).with_prefix_slots(4).with_adapter_bank(bank);
    let (engine, ..) = resident.activate_engine(&[device]).expect("the tiny GGUF loads");
    std::fs::remove_file(path).ok();
    engine
}

/// A 48-token record whose mask supervises everything past the prompt's first
/// 16 tokens except every fifth one.
fn record() -> (Vec<u32>, Vec<bool>) {
    let ids: Vec<u32> = (0..48u32).map(|i| 3 + (i * 11 + i * i) % 26).collect();
    let mask: Vec<bool> = (0..ids.len()).map(|i| i >= 16 && i % 5 != 0).collect();
    (ids, mask)
}

fn margin(row: &[f32]) -> f32 {
    let mut sorted = row.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[0] - sorted[1]
}

/// SPEC: a score is the log-softmax of the served model's own logits at the
/// recorded token, at every supervised position, and no other position.
fn a_score_is_the_log_softmax_of_the_served_logits() {
    let mut engine = engine();
    let (ids, mask) = record();
    let scored = engine.score_supervised(&ids, &mask, None).expect("the record scores");
    let want: Vec<u32> = mask.iter().enumerate().skip(1).filter(|(_, &s)| s).map(|(p, _)| p as u32).collect();
    assert_eq!(scored.iter().map(|s| s.position).collect::<Vec<_>>(), want, "one score per supervised position past the first");

    let reference = engine.instance().teacher_forced_logits(&ids[..1], &ids[1..]).expect("the reference read of the same stream");
    let (mut worst, mut sum, mut held) = (0f32, 0f64, 0usize);
    for s in &scored {
        let row = &reference[s.position as usize - 1];
        let diff = (s.logprob - model::logprobs::row_logprob(row, ids[s.position as usize] as usize)).abs();
        worst = worst.max(diff);
        sum += diff as f64;
        // Only a position the reference itself can call is held to its pick.
        if margin(row) > 0.25 {
            held += 1;
            assert_eq!(s.argmax, model::hostmath::argmax_row(row), "argmax at position {} (reference margin {:.2})", s.position, margin(row));
        }
    }
    let mean = sum / scored.len() as f64;
    println!("qwen35 score: {} positions, mean |log-prob diff| {mean:.2e}, worst {worst:.2e}, {held} argmax positions held", scored.len());
    assert!(mean < 0.05, "mean |log-probability difference| {mean:.3e} exceeds 0.05 between the two tapes of the same model");
    assert!(worst < 0.5, "worst |log-probability difference| {worst:.3e} exceeds 0.5");
    assert!(held * 2 >= scored.len(), "only {held} of {} positions had a margin to check the argmax against", scored.len());
}

/// SPEC: an adapter registered in the bank scores the WHOLE record under its
/// own slot, and releasing it leaves the base's scores exactly as they were.
fn an_adapter_slot_scores_the_whole_record_under_the_adapter() {
    let mut engine = engine();
    let (ids, mask) = record();
    let base = engine.score_supervised(&ids, &mask, None).expect("the base scores");
    let id = load_adapter(&mut engine);
    let adapted = engine.score_supervised(&ids, &mask, Some(id)).expect("the adapter scores");
    assert_eq!(adapted.len(), base.len(), "the same positions are scored under an adapter");
    // Every position moves: a slot applied to only the record's first row (or
    // only to the head) would leave the rest of the record the base's.
    let diff: Vec<f32> = adapted.iter().zip(&base).map(|(a, b)| (a.logprob - b.logprob).abs()).collect();
    let mean = diff.iter().map(|d| *d as f64).sum::<f64>() / diff.len() as f64;
    println!("qwen35 score: adapter vs base mean |log-prob diff| {mean:.3e}, min {:.3e}", diff.iter().cloned().fold(f32::MAX, f32::min));
    assert!(mean > 0.05, "the adapter must move the scores (mean |difference| {mean:.3e})");
    assert!(diff.iter().all(|d| *d > 0.0), "every supervised position runs under the adapter, so every score moves");
    // A base score after the adapter's is the base's, bit for bit: the pin was
    // released and no row leaked into the base's dispatch.
    let again = engine.score_supervised(&ids, &mask, None).expect("the base scores again");
    assert!(again.iter().zip(&base).all(|(a, b)| a.logprob.to_bits() == b.logprob.to_bits() && a.argmax == b.argmax), "the base's scores changed after an adapter ran");
}

/// A distinct rank-2 adapter over every linear with factors large enough to
/// move the logits.
fn load_adapter(e: &mut Qwen35GgufEngine) -> AdapterId {
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |amp: f32| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
    };
    let rank = 2usize;
    let sites = e
        .instance()
        .config()
        .lora_sites()
        .iter()
        .map(|s| AdapterSite { base: s.name.clone(), a: (0..rank * s.spec.inn).map(|_| next(1.0 / (s.spec.inn as f32).sqrt())).collect(), b: (0..s.spec.out * rank).map(|_| next(0.8)).collect() })
        .collect();
    let adapter = DeviceAdapter { rank: rank as u32, alpha: 4.0, sites };
    e.load_adapter(VerifiedAdapter { label: "score-adapter".to_string(), digest: "sha256:score".to_string(), adapter }).expect("the adapter fits the tiny GGUF's sites")
}

/// SPEC: a scoring pass resets the instance's KV rows and recurrent state, so
/// the prompts the engine had cached are dropped rather than resumed from rows
/// scoring has rewritten since.
fn scoring_drops_the_prompts_the_engine_had_cached() {
    let mut engine = engine();
    // A prompt long enough to leave a checkpoint behind (a whole 64-token chunk,
    // 16 tokens short of its end).
    let prompt: Vec<u32> = (0..152u32).map(|i| 3 + (i * 7 + 1) % 26).collect();
    let mut table = BlockTable::new();
    engine.prefill(&mut table, &prompt);
    engine.release_table(&mut table);
    assert_eq!(engine.prefix_stats().2, 1, "the long prompt left a cached checkpoint");
    let (ids, mask) = record();
    engine.score_supervised(&ids, &mask, None).expect("the record scores");
    assert_eq!(engine.prefix_stats().2, 0, "scoring dropped the cached prompt");
}

gpu_core::card_tests!(
    a_score_is_the_log_softmax_of_the_served_logits,
    an_adapter_slot_scores_the_whole_record_under_the_adapter,
    scoring_drops_the_prompts_the_engine_had_cached
);
