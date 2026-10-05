// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient checks for `horizon`, the subject-timeline model.
//!
//! The fixture is chosen so every term of the loss is live: a subject who
//! dies of one cause, one with a non-absorbing onset followed by follow-up,
//! one censored, and an empty batch slot (all padding). Values are hidden at
//! a high rate so the value head and its detection-limit state are exercised,
//! and the token tables, the soft-bin tables and the per-subject hazard bias
//! (each SHARED across rows, so a missing share of their gradient would hide
//! from a directional check) are also checked one element at a time.

use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::{population_with_followup, CODES};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

use crate::{directional_check, elementwise_check, Report};

/// The tensors a reverse pass folds or shares across rows.
pub const SHARED: [&str; 7] = [
    "tok.gamma",
    "tok.beta",
    "value_bins.weight",
    "time_bins.weight",
    "hazard.state.bias",
    "hazard.code.bias",
    "forecast.var",
];

/// A tiny model with a batch that exercises every loss term; `additive`
/// selects the additive baseline in place of the set encoder, and the set
/// encoder carries a forecast head scored on the follow-up measurements.
pub fn fixture(seed: u64, additive: bool) -> Horizon {
    let (subjects, _) = population_with_followup(200, seed);
    let has = |code: &str| {
        subjects
            .iter()
            .position(|s| s.events.iter().any(|e| e.code == code && e.t > s.entry))
            .expect("fixture event")
    };
    let censored = subjects
        .iter()
        .position(|s| s.events.iter().all(|e| e.t < s.entry))
        .expect("a censored subject");
    let pick = [has("death:a"), has("onset"), censored, has("death:b")];
    let chosen: Vec<_> = pick.iter().map(|&i| subjects[i].clone()).collect();
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(
        &subjects,
        &codes,
        &codes[..2],
        &FitOptions {
            knots: 9,
            min_count: 1,
        },
    )
    .expect("vocab");
    let mut cfg = HorizonConfig::tiny(vocab.len(), CODES.len() as u32);
    cfg.additive = additive;
    if !additive {
        cfg.forecasts = 2;
        cfg.forecast_weight = 0.7;
    }
    let enc: Vec<Encoded> = chosen.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let refs: Vec<&Encoded> = enc.iter().take(3).collect();
    // Four slots, three subjects: the fourth slot is all padding.
    let hb = assemble(&cfg, &refs, 4, 0.6, &mut data::rng::Rng::new(seed ^ 0x51));
    assert!(hb.value_state.contains(&1), "fixture hides an exact value");
    assert!(additive || hb.forecast_state.iter().any(|&s| s != 0), "fixture scores a forecast");
    let init = horizon::init_weights(&cfg, seed);
    let model = Horizon::new(cfg, 4, &init);
    model.set_batch(&hb);
    model
}

/// Directional checks over every tensor plus element-wise checks over [`SHARED`].
pub fn check_horizon(seed: u64) -> Report {
    let model = fixture(seed, false);
    let mut report = directional_check(&model, 5e-3, 4, seed ^ 0x1234);
    for name in SHARED {
        report
            .checks
            .extend(elementwise_check(&model, name, 1e-2).checks);
    }
    report
}

/// The additive baseline: every tensor directionally, and element-wise the
/// tables its pooling sum and its per-subject fold share across rows.
pub fn check_horizon_additive(seed: u64) -> Report {
    let model = fixture(seed, true);
    let mut report = directional_check(&model, 5e-3, 4, seed ^ 0x4321);
    for name in [
        "tok.gamma",
        "tok.beta",
        "value_bins.weight",
        "time_bins.weight",
        "additive.state.weight",
        "hazard.code.bias",
    ] {
        report
            .checks
            .extend(elementwise_check(&model, name, 1e-2).checks);
    }
    report
}
