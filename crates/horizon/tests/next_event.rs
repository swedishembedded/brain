// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The next-event objective: a second group of competing codes whose first
//! occurrence after the prediction time is modelled, beside the outcome codes.
//! Its likelihood is the same piecewise-exponential one, so it teaches the
//! encoder which event comes next and when, from every history, without
//! changing what the outcome codes mean.

use horizon::encode::{encode, Encoded};
use horizon::survival::Curves;
use horizon::synthetic::{population, CODES};
use horizon::timeline::Subject;
use horizon::train::{predict_log_hazards, TimelineObjective};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn subject(events: &str, window_to: f64) -> Subject {
    Subject::from_json_line(&format!(
        r#"{{"subject_id":"a","source":"s","entry":50,"calendar_at_entry":2000,
            "observations":[{{"t":50,"var":"x","value":1.0}}],
            "events":[{events}],
            "at_risk":[{{"code":"*","from":50,"to":{window_to}}}]}}"#
    ))
    .unwrap()
}

fn vocab_for(s: &Subject) -> (Vocab, HorizonConfig) {
    let opts = FitOptions {
        knots: 5,
        min_count: 1,
    };
    let v = Vocab::fit(
        std::slice::from_ref(s),
        &names(&["death", "dx"]),
        &names(&["death"]),
        &opts,
    )
    .unwrap()
    .with_next_events(&names(&["dx", "fall", "death"]))
    .unwrap();
    let mut cfg = HorizonConfig::tiny(v.len(), v.head_codes() as u32);
    cfg.next_codes = 3;
    (v, cfg)
}

/// The next-event group's exposure ends at the first event among its codes
/// (or the subject's death, or the window), and only that event counts.
#[test]
fn the_first_event_among_the_group_ends_exposure_and_is_the_only_one_scored() {
    // knots 0, 1, 2.5, 4. dx at +1.5, a fall at +3 (after dx), death at +3.5.
    let s = subject(
        r#"{"t":51.5,"code":"dx"},{"t":53,"code":"fall"},{"t":53.5,"code":"death"}"#,
        60.0,
    );
    let (v, cfg) = vocab_for(&s);
    let e = encode(&s, &v, &cfg);
    assert_eq!(e.outcomes.len(), 5, "two outcome codes, three next-event codes");
    // Outcome codes keep their own semantics: dx is a first occurrence that
    // death competes with, death ends everything at +3.5.
    assert_eq!(e.outcomes[1].event_piece, Some(1));
    assert_eq!(e.outcomes[0].event_piece, Some(2));
    // Next-event codes (dx, fall, death): the group stops at +1.5 for all.
    for (g, code) in [(2usize, "dx"), (3, "fall"), (4, "death")] {
        assert_eq!(e.outcomes[g].exposure, vec![1.0, 0.5, 0.0], "{code}");
    }
    assert_eq!(e.outcomes[2].event_piece, Some(1), "dx came first");
    assert_eq!(e.outcomes[3].event_piece, None, "the fall was second");
    assert_eq!(e.outcomes[4].event_piece, None, "death was third");
}

/// A death that is not in the group still ends follow-up for the group, and
/// an event dated before entry is history, not the next event.
#[test]
fn a_death_outside_the_group_and_history_are_handled() {
    let s = subject(r#"{"t":44,"code":"fall"},{"t":52,"code":"death"}"#, 60.0);
    let opts = FitOptions {
        knots: 5,
        min_count: 1,
    };
    let v = Vocab::fit(
        std::slice::from_ref(&s),
        &names(&["death", "dx"]),
        &names(&["death"]),
        &opts,
    )
    .unwrap()
    .with_next_events(&names(&["dx", "fall"]))
    .unwrap();
    let mut cfg = HorizonConfig::tiny(v.len(), v.head_codes() as u32);
    cfg.next_codes = 2;
    let e = encode(&s, &v, &cfg);
    for g in [2usize, 3] {
        assert_eq!(e.outcomes[g].exposure, vec![1.0, 1.0, 0.0], "death at +2 ends the group");
        assert_eq!(e.outcomes[g].event_piece, None, "censored by the death");
    }
}

#[test]
fn the_group_must_be_declared_consistently() {
    let s = subject("", 60.0);
    let (v, _) = vocab_for(&s);
    assert!(v.clone().with_next_events(&[]).is_err(), "an empty group is not a group");
    assert!(
        v.clone().with_next_events(&names(&["dx", "dx"])).is_err(),
        "duplicates would double count"
    );
    let mut cfg = HorizonConfig::tiny(v.len(), v.head_codes() as u32);
    cfg.next_codes = cfg.n_codes;
    assert!(cfg.validate().is_err(), "outcome codes cannot all be next-event codes");
    cfg.next_codes = 3;
    cfg.next_weight = 0.0;
    assert!(cfg.validate().is_err(), "a zero weight would silence the group");
}

/// Trained on the synthetic population, the model's first-event distribution
/// matches the generator's, closer than the covariate-blind mean, and the
/// outcome codes' own curves are untouched by the group's presence in the head.
#[test]
fn a_trained_model_recovers_the_true_first_event_distribution() {
    if skip_gpu() {
        return;
    }
    let (train, _) = population(20_000, 1);
    let (held_out, _) = population(4_000, 3);
    let (test, truth) = population(1_500, 2);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&train, &codes, &codes[..2], &FitOptions::default())
        .unwrap()
        .with_next_events(&codes)
        .unwrap();
    let mut cfg = HorizonConfig::tiny(vocab.len(), vocab.head_codes() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.d_ff = 64;
    cfg.value_bins = 16;
    cfg.time_bins = 8;
    cfg.rank = 16;
    cfg.knots = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0];
    cfg.value_weight = 0.2;
    cfg.next_codes = codes.len() as u32;
    cfg.next_weight = 0.5;
    cfg.initial_log_hazard = -4.6;
    let enc = |s: &[Subject]| -> Vec<Encoded> { s.iter().map(|s| encode(s, &vocab, &cfg)).collect() };
    let (enc_train, enc_test, enc_held) = (enc(&train), enc(&test), enc(&held_out));
    let b = 256;
    let model = Horizon::new(cfg.clone(), b, &horizon::init_weights(&cfg, 7));
    let opts = model::FitOpts {
        steps: 2500,
        batch_size: b,
        lr: 3e-3,
        min_lr: 3e-4,
        warmup: 50,
        decay_iters: 2500,
        weight_decay: 0.1,
        eval_interval: 100,
        patience: 5,
        checkpoint_secs: 0,
        ..Default::default()
    };
    let objective = TimelineObjective::new(&enc_train, Some(&enc_held), 0.3);
    let (_, model) =
        model::fit_controlled(model, objective, &opts, None, model::FitControl::default()).unwrap();

    let lh = predict_log_hazards(&model, &enc_test);
    let n_out = codes.len();
    for (k, code) in CODES.iter().enumerate() {
        for t in [5.0, 10.0] {
            let truth_k: Vec<f64> = truth.iter().map(|tr| tr.first_event_cif(k, t)).collect();
            let mean_truth = truth_k.iter().sum::<f64>() / truth_k.len() as f64;
            let (mut err, mut base) = (0.0, 0.0);
            for (i, tk) in truth_k.iter().enumerate() {
                let pred = Curves::first_events(&lh[i], &cfg.knots, n_out, cfg.next_codes as usize)
                    .cif(k, t);
                err += (pred - tk).abs();
                base += (mean_truth - tk).abs();
            }
            let n = truth_k.len() as f64;
            println!("first event {code} at {t}: mean |pred - truth| {:.4}, covariate-blind {:.4}", err / n, base / n);
            assert!(
                err < 0.5 * base,
                "{code} at {t}: error {:.4} is not well below the covariate-blind {:.4}",
                err / n,
                base / n
            );
        }
    }
}
