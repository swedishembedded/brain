// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The state across visits extrapolates over gaps it was never trained on.
//!
//! A risk factor drifts in continuous time and is measured at irregular
//! visits; death after entry depends on its value at entry. Both encoders are
//! trained on subjects whose last visit was at most two years before entry,
//! then asked about subjects whose last visit was six to ten years before.
//! The best possible prediction is known exactly (a Kalman filter over the
//! visits): the further back the last visit, the closer it is to the
//! population's. The set encoder only knows gaps through time-ago embeddings
//! it never saw at that length; the state across visits forgets over the
//! elapsed time at the rates it learned. Out of distribution it must be the
//! closer of the two to the best prediction, and in distribution no more
//! than a quarter worse than the set encoder.

use horizon::encode::{encode, Encoded};
use horizon::saved::Saved;
use horizon::synthetic::drifting::{self, Gaps, Posterior};
use horizon::timeline::Subject;
use horizon::train::TimelineObjective;
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

const TRAIN_GAPS: Gaps = Gaps {
    last: (0.0, 2.0),
    between: (0.5, 2.0),
    visits: (1, 4),
};
const LONG_GAPS: Gaps = Gaps {
    last: (6.0, 10.0),
    ..TRAIN_GAPS
};
const HORIZON: f64 = 5.0;

fn train(visits: u32, vocab: &Vocab, train: &[Subject], held: &[Subject]) -> Saved {
    let mut cfg = HorizonConfig::default_for(vocab.len(), 1);
    cfg.max_tokens = 8;
    cfg.d_model = 32;
    cfg.n_heads = 4;
    cfg.d_ff = 64;
    cfg.rank = 16;
    cfg.knots = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    cfg.visits = visits;
    let enc =
        |s: &[Subject]| -> Vec<Encoded> { s.iter().map(|x| encode(x, vocab, &cfg)).collect() };
    let (enc_train, enc_held) = (enc(train), enc(held));
    let b = 256;
    let model = Horizon::new(cfg.clone(), b, &horizon::init_weights(&cfg, 7));
    let opts = model::FitOpts {
        steps: 2000,
        batch_size: b,
        lr: 3e-3,
        min_lr: 3e-4,
        warmup: 50,
        decay_iters: 2000,
        weight_decay: 0.1,
        eval_interval: 100,
        patience: 6,
        checkpoint_secs: 0,
        ..Default::default()
    };
    let (_, model) = model::fit_controlled(
        model,
        TimelineObjective::new(&enc_train, Some(&enc_held), 0.3),
        &opts,
        None,
        model::FitControl::default(),
    )
    .unwrap();
    Saved {
        model,
        vocab: vocab.clone(),
    }
}

/// Mean absolute distance of the predicted risk by [`HORIZON`] from the best
/// possible one.
fn error(saved: &Saved, subjects: &[Subject], best: &[Posterior]) -> f64 {
    let curves = saved.predict(subjects).unwrap();
    curves
        .iter()
        .zip(best)
        .map(|(c, p)| (c.cif(0, HORIZON) - p.cif(HORIZON)).abs())
        .sum::<f64>()
        / best.len() as f64
}

#[test]
fn the_state_across_visits_extrapolates_over_long_gaps() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train_s, _) = drifting::population(12_000, 1, &TRAIN_GAPS, 10.0);
    let (held_s, _) = drifting::population(3_000, 2, &TRAIN_GAPS, 10.0);
    let (near_s, near_best) = drifting::population(3_000, 3, &TRAIN_GAPS, 10.0);
    let (far_s, far_best) = drifting::population(3_000, 4, &LONG_GAPS, 10.0);
    let codes = vec![drifting::CODE.to_string()];
    let vocab = Vocab::fit(&train_s, &codes, &codes, &FitOptions::default()).unwrap();
    let set = train(0, &vocab, &train_s, &held_s);
    let state = train(4, &vocab, &train_s, &held_s);
    // What a covariate-blind prediction would score: the population's risk.
    let blind = |best: &[Posterior]| {
        let mean = best.iter().map(|p| p.cif(HORIZON)).sum::<f64>() / best.len() as f64;
        best.iter()
            .map(|p| (mean - p.cif(HORIZON)).abs())
            .sum::<f64>()
            / best.len() as f64
    };
    let (set_near, state_near) = (
        error(&set, &near_s, &near_best),
        error(&state, &near_s, &near_best),
    );
    let (set_far, state_far) = (
        error(&set, &far_s, &far_best),
        error(&state, &far_s, &far_best),
    );
    println!("risk by {HORIZON} years, mean |predicted - best possible|:");
    println!("  last visit 0-2 years before:  set encoder {set_near:.4}  state across visits {state_near:.4}  population risk {:.4}", blind(&near_best));
    println!("  last visit 6-10 years before: set encoder {set_far:.4}  state across visits {state_far:.4}  population risk {:.4}", blind(&far_best));
    assert!(
        state_far < set_far,
        "out of distribution the state across visits must be closer to the best prediction"
    );
    assert!(
        state_near <= 1.25 * set_near,
        "in distribution it must not be materially worse"
    );
}
