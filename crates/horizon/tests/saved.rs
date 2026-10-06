// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model lifecycle tooling that makes a trained
// model reproducible from disk. If your team needs expertise in checkpointed
// prediction systems you can procure our services by sending an email to
// info@swedishembedded.com.

//! A saved model predicts exactly what the model it was saved from predicts,
//! for every model shape: the set encoder, the additive baseline and a model
//! with a forecast head (the visit backbones are covered by the SDK tests).

use horizon::saved::Saved;
use horizon::synthetic::{population, CODES};
use horizon::train::{predict_forecasts, predict_log_hazards};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

fn round_trip(name: &str, configure: impl Fn(&mut HorizonConfig)) {
    let (subjects, _) = population(120, 6);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&subjects, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    configure(&mut cfg);
    let model = Horizon::new(cfg.clone(), 32, &horizon::init_weights(&cfg, 11));
    let saved = Saved::new(model, vocab);
    let dir = std::env::temp_dir().join(format!("horizon-saved-{name}-{}", std::process::id()));
    saved.save(&dir).unwrap();
    let loaded = Saved::load(&dir).unwrap();
    assert_eq!(loaded.model.cfg, saved.model.cfg, "{name}: the configuration travels with the weights");

    let enc = saved.encode(&subjects[..40]).unwrap();
    assert_eq!(
        predict_log_hazards(&saved.model, &enc),
        predict_log_hazards(&loaded.model, &enc),
        "{name}: hazards after a reload"
    );
    if cfg.forecasts > 0 {
        let query = horizon::encode::forecast_query(&saved.vocab, &cfg, "x1", 2.0).unwrap();
        let queries = vec![vec![query]; enc.len()];
        let (a, b) = (
            predict_forecasts(&saved.model, &enc, &queries).unwrap(),
            predict_forecasts(&loaded.model, &enc, &queries).unwrap(),
        );
        assert_eq!(a, b, "{name}: forecasts after a reload");
        assert!(a.iter().all(|p| p[0].1 > 0.0));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_additive_baseline_survives_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    round_trip("additive", |c| c.additive = true);
}

#[test]
fn the_forecast_head_survives_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    round_trip("forecast", |c| {
        c.forecasts = 2;
        c.forecast_weight = 0.5;
    });
}

#[test]
fn the_set_encoder_survives_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    round_trip("set", |_| {});
}
