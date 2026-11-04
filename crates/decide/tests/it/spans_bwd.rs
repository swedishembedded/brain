// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Backpropagating every packed span in ONE dispatch must compute the same
//! gradient as one dispatch per span.
//!
//! This needs a test rather than a reading of the code because the fused
//! kernels address a span from a work table where the per-span ones address it
//! from a storage-binding offset, and the two ways of saying "this span starts
//! at row 137" are only equal if the table, the score-slab base and the
//! dispatch grid all agree. A table that disagrees does not fail: it reads a
//! neighbouring option's rows, which is a plausible gradient, trains, and is
//! wrong. So the per-span path is kept and this holds the fast one against it.
//!
//! Swedish Embedded AB implements verified performance work on neural network
//! training for its clients. If your team needs a faster training step that
//! provably computes the same gradient as the one it replaced, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use decide::decide::{Decide, Limits};
use decide::primitives::{Opt, Question};

fn model() -> Option<Decide> {
    let tok_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/decide/tokenizer/tokenizer.json");
    let tok = match data::wordpiece::WordPiece::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => {
            brain_testutil::skip(&format!("{tok_path}: {e} - run scripts/data/fetch-testdata.sh"));
            return None;
        }
    };
    let cfg = decide::config::EncoderConfig::mini_lm_l6();
    let enc: HashMap<String, Vec<f32>> = decide::init::init_weights(&cfg, 1);
    let head: HashMap<String, Vec<f32>> = decide::init::init_head(&cfg, 2);
    let gpu = gpu_core::testgpu::dev(decide::kern::PIPELINES);
    // Deliberately short windows, so one state is several spans and the packing
    // is ragged in BOTH directions - long windows and short option slots.
    let limits = Limits { cap_rows: 512, cap_slots: 16, max_span: 32, overlap: 4 };
    Some(Decide::new_on(gpu, cfg, tok, limits, &enc, &head, true))
}

/// Six options of visibly different lengths, so no two slot spans are the
/// same size and an off-by-one span base cannot go unnoticed.
fn question() -> Question {
    Question::Choice {
        instructions: "which banking intent does this message express".into(),
        options: vec![
            Opt::new("card arrival"),
            Opt::new("exchange rate"),
            Opt::new("top up by bank transfer charge"),
            Opt::new("pin blocked"),
            Opt::new("declined card payment because the balance was insufficient"),
            Opt::new("cancel"),
        ],
    }
}

const STATE: &str = "I still have not received the new card I ordered three weeks ago, and the \
                     tracking page has not changed since the day it shipped, so I would like to \
                     know whether it was ever posted at all.";

/// Every parameter gradient of both halves, in one flat vector.
fn grads(m: &Decide) -> Vec<f32> {
    let mut v = Vec::new();
    for (name, _) in m.cfg.tensor_manifest() {
        v.extend(m.enc.read_grad(&name));
    }
    for (name, _) in decide::head::tensor_manifest(&m.cfg) {
        v.extend(m.head.read_grad(&name));
    }
    v
}

fn accumulate_once(m: &mut Decide) -> Vec<f32> {
    let ce = decide::loss::LossConfig::cross_entropy();
    m.zero_grads();
    m.accumulate(STATE, &question(), |s| decide::loss::decision_loss(s, 2, &ce)).expect("accumulate");
    m.enc.poll_wait();
    m.head.poll_wait();
    grads(m)
}

#[test]
fn the_fused_span_backward_is_the_per_span_one() {
    let Some(mut m) = model() else { return };

    let fused = accumulate_once(&mut m);
    assert!(
        m.enc.fused_span_bwd(),
        "this device did not select the fused reverse pass, so the comparison below would be \
         a buffer against itself"
    );

    m.enc.set_fused_span_bwd(false);
    let per_span = accumulate_once(&mut m);
    assert!(!m.enc.fused_span_bwd());

    assert_eq!(fused.len(), per_span.len());
    let scale: f32 = per_span.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let (worst, at) = fused
        .iter()
        .zip(&per_span)
        .enumerate()
        .map(|(i, (a, b))| ((a - b).abs(), i))
        .fold((0.0f32, 0usize), |acc, x| if x.0 > acc.0 { x } else { acc });
    // A tolerance, not equality: the two sum a span's keys in a different
    // ORDER, which is a rounding difference and not a different gradient.
    assert!(
        worst <= 2e-5 * scale.max(1e-6),
        "gradient {at} differs by {worst} against a largest magnitude of {scale}"
    );
    assert!(scale > 0.0, "both reverse passes produced nothing to compare");
}
