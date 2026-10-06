// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `event_nll_each`: every subject's own event NLL in one pass over the
//! batches, equal to scoring that subject alone, and averaging back to
//! `event_nll`.

use horizon::encode::{encode, Encoded};
use horizon::synthetic::{population, CODES};
use horizon::train::{event_nll, event_nll_each};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

#[test]
fn each_subjects_nll_equals_scoring_it_alone_and_averages_to_the_total() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (subjects, _) = population(40, 5);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&subjects, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::tiny(vocab.len(), codes.len() as u32);
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0, 15.0];
    // Unequal weights, so a mistake in the rescaling cannot cancel out.
    let enc: Vec<Encoded> = subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut e = encode(s, &vocab, &cfg);
            e.weight = 0.5 + (i % 4) as f32;
            e
        })
        .collect();
    // Batches of 16 over 40 subjects: a short last batch with padded slots.
    let model = Horizon::new(cfg.clone(), 16, &horizon::init_weights(&cfg, 3));
    let each = event_nll_each(&model, &enc);
    assert_eq!(each.len(), enc.len());
    assert!(each.iter().all(|x| x.is_finite() && *x >= 0.0));
    let (mut total, mut weight) = (0.0f64, 0.0f64);
    for (x, e) in each.iter().zip(&enc) {
        total += *x as f64 * e.weight as f64;
        weight += e.weight as f64;
    }
    let all = event_nll(&model, &enc);
    assert!(
        ((total / weight) as f32 - all).abs() < 1e-4 * (1.0 + all.abs()),
        "weighted mean of each {} vs event_nll {all}",
        total / weight
    );
    for i in [0usize, 7, 15, 16, 39] {
        let alone = event_nll(&model, std::slice::from_ref(&enc[i]));
        assert!(
            (each[i] - alone).abs() < 1e-4 * (1.0 + alone.abs()),
            "subject {i}: {} in a batch vs {alone} alone",
            each[i]
        );
    }
}
