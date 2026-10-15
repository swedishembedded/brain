// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducible architecture
// evaluation for its clients. If your team needs expertise in benchmarking
// model architectures then you can procure our services by sending an email
// to info@swedishembedded.com.

//! A benchmark configuration that is not given a seed draws one, so two
//! unseeded runs are two independent samples; an artifact with no recorded
//! seed reports none instead of inventing `0`.

use bench::{capscale::CapScaleConfig, scaling::Sweep, TrainConfig};

#[test]
fn unseeded_configs_draw_a_different_seed_each_time() {
    assert_ne!(TrainConfig::default().seed, TrainConfig::default().seed);
    assert_ne!(Sweep::default().seed, Sweep::default().seed);
    assert_ne!(CapScaleConfig::default().seed, CapScaleConfig::default().seed);
}

#[test]
fn an_artifact_without_a_seed_loads_with_none() {
    let path = std::env::temp_dir().join(format!("brain_noseed_{}.json", std::process::id()));
    std::fs::write(&path, r#"{"arch":"gpt2","param_count":1,"commit":"c"}"#).unwrap();
    let loaded = bench::eval::load_artifact(&path).expect("load artifact");
    std::fs::remove_file(&path).ok();
    assert_eq!(loaded.seed, None);
}
