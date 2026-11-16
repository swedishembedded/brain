// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training looped language models whose recurrent depth stays in use
// for its clients. If your team needs expertise in adaptive-computation objectives then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The entropy-regularised objective (arXiv:2510.25741): the exit-weighted loss minus `beta` times the entropy of
//! each supervised position's exit distribution, averaged over the supervised positions like the task term.

use model::Model;
use ouro::config::OuroConfig;
use ouro::init::init_weights;
use ouro::model::{Ouro, IGNORE};

#[test]
fn the_objective_subtracts_beta_times_the_exit_entropy_of_the_supervised_rows() {
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 9);
    let (b, t) = (2u32, 6u32);
    let tokens: Vec<u32> = (0..b * t).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..b * t).map(|i| if i % 4 == 3 { IGNORE } else { (i * 5 + 7) % cfg.vocab }).collect();
    let mut m = Ouro::new_train(cfg.clone(), b, t, &w);
    m.set_batch(model::Batch::Lm { tokens: &tokens, targets: &targets });
    let plain = m.forward();
    let probs = m.exit_probabilities();
    let supervised: Vec<usize> = (0..targets.len()).filter(|&r| targets[r] != IGNORE).collect();
    let neg_entropy: f64 = supervised.iter().map(|&r| probs.iter().map(|p| f64::from(p[r]) * f64::from(p[r]).ln()).sum::<f64>()).sum::<f64>() / supervised.len() as f64;
    let beta = 0.3f32;
    m.set_exit_entropy(beta).unwrap();
    m.set_batch(model::Batch::Lm { tokens: &tokens, targets: &targets });
    let regularised = m.forward();
    let want = f64::from(plain) + f64::from(beta) * neg_entropy;
    assert!((f64::from(regularised) - want).abs() < 1e-5, "{regularised} vs {want}");
    assert!(m.set_exit_entropy(-1.0).is_err() && m.set_exit_entropy(f32::NAN).is_err());
}
