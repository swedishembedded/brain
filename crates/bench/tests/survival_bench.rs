// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The survival benchmarks' identifiability checks (see `bench::survival_bench`).
//!
//! Swedish Embedded AB implements validation of survival and longitudinal
//! models against data whose true risks are known, for its clients. If your
//! team needs expertise in checking that a risk model recovers what the data
//! contains, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Each test trains the benchmark's small horizon model on its generator's
//! data with a fixed seed and asserts what the benchmark exists to show. The
//! bounds were set from seeded runs on the CUDA backend (seeds 1 to 4) with
//! room for run-to-run variation; the measured values of the seed used here
//! are printed with `--nocapture`. The generators' own statistical checks
//! (closed-form CIF against the simulation; the competing generator's
//! analytic CIF against Aalen-Johansen) are unit tests in `horizon`.
//!
//! GPU tests: run with `--test-threads=1`; `MOE_SKIP_GPU_TESTS` skips them.

use bench::survival_bench::{SurvivalCompeting, SurvivalIrregular, SurvivalLongitudinal, SurvivalSingle};
use bench::{Benchmark, Metrics};

const SEED: u64 = 1;

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn run(b: &dyn Benchmark) -> Metrics {
    let dir = std::env::temp_dir().join(format!("{}_{}_{SEED}", b.name(), std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    b.prepare(&dir, SEED).unwrap();
    let m = b.evaluate(&dir, SEED).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let mut fields: Vec<_> = m.fields.iter().collect();
    fields.sort_by(|a, b| a.0.cmp(b.0));
    println!("{} seed {SEED}: score {:.4} {fields:?}", b.name(), m.score);
    assert!(m.score >= b.threshold(), "{}: skill {:.4} under its reference line {}", b.name(), m.score, b.threshold());
    m
}

fn f(m: &Metrics, name: &str) -> f32 {
    m.get(name).unwrap_or_else(|| panic!("metric {name} missing"))
}

/// The predicted risk ordering is the true one: rank correlation with the true
/// risk is high, the concordance is within a hair of the true risk's own on the
/// same data, and the model is calibrated.
#[test]
fn survival_single_ranks_like_the_truth() {
    if skip_gpu() {
        return;
    }
    let m = run(&SurvivalSingle::default());
    assert!(f(&m, "spearman_oracle") > 0.93, "rank correlation with the true risk");
    assert!(f(&m, "harrell") > f(&m, "harrell_oracle") - 0.025, "Harrell C vs the true risk's");
    assert!(f(&m, "uno") > f(&m, "harrell_oracle") - 0.03, "Uno C vs the true risk's");
    assert!(f(&m, "auc") > f(&m, "auc_oracle") - 0.03, "time-dependent AUC vs the true risk's");
    assert!((f(&m, "oe") - 1.0).abs() < 0.2, "observed over expected at the horizon");
}

/// The model's cause-specific cumulative incidence matches the analytic one.
#[test]
fn survival_competing_recovers_the_analytic_cumulative_incidence() {
    if skip_gpu() {
        return;
    }
    let m = run(&SurvivalCompeting::default());
    assert!(
        f(&m, "cif_mae") < 0.35 * f(&m, "cif_mae_null"),
        "error {} is not well below the covariate-blind {}",
        f(&m, "cif_mae"),
        f(&m, "cif_mae_null")
    );
    assert!(f(&m, "cif_bias").abs() < 0.2 * f(&m, "cif_mean"), "bias {} on a mean of {}", f(&m, "cif_bias"), f(&m, "cif_mean"));
    assert!(f(&m, "spearman_oracle") > 0.9);
    assert!((f(&m, "oe") - 1.0).abs() < 0.2);
}

/// The visit-blind model learns nothing from the (noise) values; the model that
/// sees when subjects were seen predicts events measurably better.
#[test]
fn survival_irregular_time_aware_beats_time_blind() {
    if skip_gpu() {
        return;
    }
    let m = run(&SurvivalIrregular::default());
    assert!(
        f(&m, "nll_gain") > 0.04,
        "held-out event NLL: aware {} vs blind {}",
        f(&m, "nll"),
        f(&m, "nll_blind")
    );
    assert!(f(&m, "ibs") < f(&m, "ibs_blind") - 0.008, "integrated Brier: aware vs blind");
    assert!(f(&m, "ibs_blind") > f(&m, "ibs_null") - 0.003, "the blind model has nothing beyond the base rate");
    assert!(f(&m, "spearman_oracle") > 0.9, "rank correlation with the best possible prediction");
}

/// The longitudinal measurements improve event prediction over baseline
/// covariates alone, and the forecast head beats the population mean.
#[test]
fn survival_longitudinal_uses_the_measurements() {
    if skip_gpu() {
        return;
    }
    let m = run(&SurvivalLongitudinal::default());
    assert!(
        f(&m, "nll_gain") > 0.04,
        "held-out event NLL: with measurements {} vs baseline covariates only {}",
        f(&m, "nll"),
        f(&m, "nll_baseline")
    );
    assert!(f(&m, "ibs") < f(&m, "ibs_baseline") - 0.008, "integrated Brier: measurements vs baseline");
    assert!(f(&m, "forecast_mae") < f(&m, "forecast_mae_mean"), "forecast vs the population mean");
    assert!(f(&m, "forecast_skill") > 0.15, "forecast skill {}", f(&m, "forecast_skill"));
    assert!(
        f(&m, "forecast_mae") < 1.15 * f(&m, "forecast_mae_truth"),
        "forecast error {} against the noise floor {}",
        f(&m, "forecast_mae"),
        f(&m, "forecast_mae_truth")
    );
    assert!(f(&m, "spearman_oracle") > 0.95, "rank correlation with the best possible prediction");
}
