// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model recovers known hazards: trained on a synthetic population whose
//! true cumulative incidence is known (`horizon::synthetic`), its predictions
//! on subjects it never saw must sit close to the truth - closer than the
//! best predictor that ignores the covariates (the population mean of the
//! truth, which is what a model that learned nothing but the base rate would
//! give at best), and without a systematic offset.

use horizon::encode::{encode, Encoded};
use horizon::survival::Curves;
use horizon::synthetic::{population, ABSORBING, CODES};
use horizon::train::{predict_log_hazards, TimelineObjective};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn a_trained_model_recovers_the_true_cumulative_incidence() {
    if skip_gpu() {
        return;
    }
    recovers(false);
}

/// The truth is additive on the log-hazard scale, so the additive baseline
/// must recover it too: it is the model the set encoder is compared against.
#[test]
fn the_additive_baseline_recovers_an_additive_truth() {
    if skip_gpu() {
        return;
    }
    recovers(true);
}

fn recovers(additive: bool) {
    let (train, _) = population(30_000, 1);
    let (held_out, _) = population(5_000, 3);
    let (test, truth) = population(1_500, 2);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&train, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let cfg = HorizonConfig {
        vocab: vocab.len(),
        max_tokens: 8,
        d_model: 32,
        n_layers: 2,
        n_heads: 4,
        d_ff: 64,
        value_bins: 16,
        time_bins: 8,
        rank: 16,
        n_codes: CODES.len() as u32,
        knots: vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0,
        ],
        value_weight: 0.2,
        additive,
        forecasts: 0,
        forecast_weight: 0.0,
        visits: 0,
    };
    let enc_train: Vec<Encoded> = train.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let enc_test: Vec<Encoded> = test.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let enc_held: Vec<Encoded> = held_out.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let b = 256;
    let model = Horizon::new(cfg.clone(), b, &horizon::init_weights(&cfg, 7));
    // Early stopping on the held-out event NLL, keeping the best model: the
    // procedure a real run uses, without which the model memorises training
    // subjects and its predictions on new ones drift low.
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
    let objective = TimelineObjective::new(&enc_train, Some(&enc_held), 0.3);
    let (report, model) =
        model::fit_controlled(model, objective, &opts, None, model::FitControl::default()).unwrap();
    assert!(
        report.final_loss.unwrap() < report.initial_loss,
        "training reduced the loss"
    );

    let lh = predict_log_hazards(&model, &enc_test);
    for (k, code) in CODES.iter().enumerate() {
        for t in [5.0, 10.0] {
            let truth_k: Vec<f64> = truth.iter().map(|tr| tr.cif(k, t)).collect();
            let mean_truth = truth_k.iter().sum::<f64>() / truth_k.len() as f64;
            let (mut err, mut base, mut bias) = (0.0, 0.0, 0.0);
            for (i, tk) in truth_k.iter().enumerate() {
                let pred = Curves::new(&lh[i], &cfg.knots, &ABSORBING).cif(k, t);
                err += (pred - tk).abs();
                base += (mean_truth - tk).abs();
                bias += pred - tk;
            }
            let n = truth_k.len() as f64;
            let (err, base, bias) = (err / n, base / n, bias / n);
            println!("additive={additive} {code} at {t}: mean |pred - truth| {err:.4}, covariate-blind {base:.4}, bias {bias:+.4}");
            assert!(
                err < 0.35 * base,
                "{code} at {t}: error {err:.4} is not well below the covariate-blind {base:.4}"
            );
            assert!(
                bias.abs() < 0.2 * mean_truth,
                "{code} at {t}: bias {bias:+.4} against a mean of {mean_truth:.4}"
            );
        }
    }
}
