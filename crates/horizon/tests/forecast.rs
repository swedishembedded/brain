// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The forecast head recovers a known trajectory: `x1` drifts after entry at
//! a rate that depends on the subject's group, with known noise. Trained on
//! subjects measured again at 1, 3 and 5 years, the model's median forecast
//! for unseen subjects must track the true mean far better than carrying
//! the entry value forward, and its interval from the 5th to the 95th
//! percentile must hold about nine in ten held-out follow-up measurements.

use horizon::encode::{encode, forecast_query, Encoded};
use horizon::synthetic::{population_with_followup, CODES};
use horizon::timeline::Value;
use horizon::train::{predict_forecasts, TimelineObjective};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

#[test]
fn the_forecast_head_recovers_a_known_trajectory() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = population_with_followup(20_000, 1);
    let (held_out, _) = population_with_followup(4_000, 3);
    let (test, truth) = population_with_followup(1_500, 2);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&train, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 32;
    cfg.n_heads = 4;
    cfg.d_ff = 64;
    cfg.rank = 16;
    cfg.knots = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0];
    cfg.forecasts = 3;
    cfg.forecast_weight = 1.0;
    let enc = |s: &[horizon::timeline::Subject]| -> Vec<Encoded> {
        s.iter().map(|x| encode(x, &vocab, &cfg)).collect()
    };
    let (enc_train, enc_held, enc_test) = (enc(&train), enc(&held_out), enc(&test));
    let b = 256;
    let model = Horizon::new(cfg.clone(), b, &horizon::init_weights(&cfg, 5));
    let opts = model::FitOpts {
        steps: 3000,
        batch_size: b,
        lr: 3e-3,
        min_lr: 3e-4,
        warmup: 50,
        decay_iters: 3000,
        weight_decay: 0.1,
        eval_interval: 100,
        patience: 5,
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

    for ahead in [3.0, 5.0] {
        let queries: Vec<Vec<_>> = enc_test
            .iter()
            .map(|_| vec![forecast_query(&vocab, &cfg, "x1", ahead).unwrap()])
            .collect();
        let pred = predict_forecasts(&model, &enc_test, &queries).unwrap();
        let (mut err, mut carried, mut covered, mut measured) = (0.0, 0.0, 0usize, 0usize);
        for (i, p) in pred.iter().enumerate() {
            let (mu, sigma) = (p[0].0 as f64, p[0].1 as f64);
            let median = vocab.forecast_quantile("x1", mu, sigma, 0.5).unwrap();
            let (lo, hi) = (
                vocab.forecast_quantile("x1", mu, sigma, 0.05).unwrap(),
                vocab.forecast_quantile("x1", mu, sigma, 0.95).unwrap(),
            );
            let (true_mean, _) = truth[i].x1_at(ahead);
            let at_entry = test[i]
                .observations
                .iter()
                .find(|o| o.var == "x1" && o.t == test[i].entry)
                .map(|o| match o.value {
                    Value::Number(x) => x,
                    _ => f64::NAN,
                });
            err += (median - true_mean).abs();
            carried += (at_entry.unwrap() - true_mean).abs();
            // The follow-up measurement actually taken, when there was one.
            if let Some(Value::Number(y)) = test[i]
                .observations
                .iter()
                .find(|o| o.var == "x1" && (o.t - test[i].entry - ahead).abs() < 1e-9)
                .map(|o| o.value.clone())
            {
                measured += 1;
                covered += usize::from(lo <= y && y <= hi);
            }
        }
        let n = pred.len() as f64;
        let coverage = covered as f64 / measured as f64;
        println!("x1 at +{ahead}: median error {:.4}, carried forward {:.4}, 5th-95th coverage {coverage:.3} over {measured}", err / n, carried / n);
        assert!(
            err / n < 0.5 * carried / n,
            "the forecast must beat carrying the entry value forward"
        );
        assert!(
            (coverage - 0.90).abs() < 0.05,
            "5th-95th percentile interval coverage {coverage:.3}"
        );
    }
}
