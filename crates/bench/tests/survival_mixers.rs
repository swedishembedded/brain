// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The mixer ablation of horizon's visit stack (`bench::survival_bench::
//! MixerAblation`): all-attention, all-Gated-DeltaNet and the 3:1 hybrid on
//! identical data, split, seed and optimisation steps. It prints a table and
//! checks the comparison is a fair one (parameter counts within the stated
//! tolerance, every variant ran every step, every score finite). It does NOT
//! assert which mixer wins: that is what the table measures.
//!
//! GPU tests: run with `--test-threads=1`; `MOE_SKIP_GPU_TESTS` skips them.

use bench::survival_bench::{MixerAblation, MixerData};

/// The seeds to run: `MIXER_ABLATION_SEEDS` (comma separated; each seed draws
/// its own data and initial weights), seed 1 when unset. More than one seed
/// shows how much of a difference between variants is the seed's.
fn seeds() -> Vec<u64> {
    std::env::var("MIXER_ABLATION_SEEDS")
        .map(|s| s.split(',').map(|x| x.trim().parse().expect("a seed number")).collect())
        .unwrap_or_else(|_| vec![1])
}

fn ablate(data: MixerData) {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in seeds() {
        ablate_seed(data, seed);
    }
}

fn ablate_seed(data: MixerData, seed: u64) {
    let ablation = MixerAblation::default();
    let dir = std::env::temp_dir().join(format!("survival_mixers_{data:?}_{}_{seed}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ablation.prepare(data, &dir, seed).unwrap();
    let rows = ablation.run(data, &dir, seed).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    println!("{data:?}, seed {seed}, {} steps, {} blocks, parameter tolerance {}:", ablation.steps, ablation.blocks, ablation.tolerance);
    println!("{:<16} {:>10} {:>9} {:>9} {:>8} {:>10}", "variant", "parameters", "event NLL", "int Brier", "AUC", "train (s)");
    for r in &rows {
        println!("{:<16} {:>10} {:>9.4} {:>9.4} {:>8.4} {:>10.1}", r.variant, r.parameters, r.nll, r.ibs, r.auc, r.train_secs);
        assert_eq!(r.steps, ablation.steps, "{}: every variant takes every step", r.variant);
        assert!(r.nll.is_finite() && r.ibs.is_finite() && r.auc.is_finite(), "{}: finite scores", r.variant);
    }
    assert_eq!(rows.len(), 3);
}

#[test]
fn mixer_ablation_on_informative_visit_times() {
    ablate(MixerData::Irregular);
}

#[test]
fn mixer_ablation_on_a_hidden_state_seen_through_noisy_measurements() {
    ablate(MixerData::Longitudinal);
}
