// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-capability predictive-scaling + tuning-advisor integration test.
//!
//! Runs the real capscale sweep for `gpt2` at a smoke budget (3 sizes × one
//! representative benchmark per axis → train+score on `BRAIN_DEVICE` → fit each
//! axis's saturating trend), then asserts the artifact:
//!   * carries every capability axis with ≥3 size points,
//!   * a finite fitted slope (β) + a finite local slope per axis,
//!   * finite 2×/4× predictions,
//! and that the advisor, fed the capscale + a synthetic eval, emits a non-empty
//! ranked recommendation list whose top item is a real axis.
//!
//! Skipped when `MOE_SKIP_GPU_TESTS` is set (same gate as the rest). Sized for a
//! few minutes on the CPU (Cranelift) backend.

use bench::advisor;
use bench::capscale::{self, CapScaleConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn capscale_gpt_smoke_writes_valid_artifact() {
    if skip() {
        return;
    }

    // A trimmed budget keeps the 3×6 grid under a few minutes on CPU.
    let cfg = CapScaleConfig { steps: 40, n_sequences: 500, eval_sequences: 40, seed: 4242, ..Default::default() };
    let report = capscale::run("gpt2", &cfg).expect("capscale run");

    // Every canonical axis has a probe, so all should be present with ≥3 points.
    assert!(!report.axes.is_empty(), "no axes scaled");
    for ax in &report.axes {
        assert!(
            ax.params.len() >= 3,
            "axis '{}' has {} size points (<3)",
            ax.axis,
            ax.params.len()
        );
        assert_eq!(ax.params.len(), ax.scores.len(), "params/scores length mismatch");
        // Params strictly increasing (the independent axis).
        for w in ax.params.windows(2) {
            assert!(w[1] > w[0], "axis '{}' params not increasing: {} -> {}", ax.axis, w[0], w[1]);
        }
        // A finite fitted slope (β) and a finite local slope.
        assert!(ax.fit.beta.is_finite(), "axis '{}' beta not finite", ax.axis);
        assert!(ax.slope_per_doubling.is_finite(), "axis '{}' slope not finite", ax.axis);
        // Finite, in-range extrapolations.
        assert!(ax.pred_2x.is_finite() && (0.0..=1.0).contains(&ax.pred_2x), "axis '{}' bad pred@2x {}", ax.axis, ax.pred_2x);
        assert!(ax.pred_4x.is_finite() && (0.0..=1.0).contains(&ax.pred_4x), "axis '{}' bad pred@4x {}", ax.axis, ax.pred_4x);
    }

    // Write + reload the artifact.
    let out = std::env::temp_dir().join(format!("brain_capscale_test_{}.json", std::process::id()));
    capscale::write_artifact(&report, &out).expect("write capscale artifact");
    let loaded = capscale::load_artifact(&out).expect("reload capscale artifact");
    assert_eq!(loaded.arch, "gpt2");
    assert_eq!(loaded.knob, "size");
    assert!(loaded.max_params > 0);
    for ax in &report.axes {
        assert!(loaded.by_axis.contains_key(&ax.axis), "axis '{}' lost on reload", ax.axis);
    }

    // The advisor, fed a synthetic eval + the real capscale, emits a non-empty
    // ranked list whose top item is a real axis.
    let eval = serde_json::json!({
        "arch": "gpt2",
        "param_count": loaded.max_params,
        "axis_scores": {
            "recall": 0.40, "copying": 0.85, "memory": 0.99,
            "state_tracking": 0.95, "compression": 0.90, "arithmetic": 0.30,
        },
        "benchmarks": [
            { "axis": "recall", "informational": false, "metrics": { "fields": { "train_ce": 1.0, "init_ce": 2.9 } } },
            { "axis": "copying", "informational": false, "metrics": { "fields": { "train_ce": 0.4, "init_ce": 2.4 } } },
            { "axis": "memory", "informational": false, "metrics": { "fields": { "train_ce": 1.1, "init_ce": 3.9 } } },
            { "axis": "state_tracking", "informational": false, "metrics": { "fields": { "train_ce": 0.1, "init_ce": 1.4 } } },
            { "axis": "compression", "informational": false, "metrics": { "fields": { "train_ce": 0.2, "init_ce": 1.0 } } },
            { "axis": "arithmetic", "informational": true, "metrics": { "fields": { "train_ce": 1.0, "init_ce": 2.9 } } },
        ],
    });
    let advice = advisor::advise(&eval, Some(&loaded));
    advisor::print_advice(&advice);
    assert!(!advice.recommendations.is_empty(), "advisor produced no recommendations");
    let top = &advice.recommendations[0];
    assert!(
        bench::axes().contains(&top.axis.as_str()),
        "top recommendation '{}' is not a real axis",
        top.axis
    );
    // Priorities are 1..=n in order.
    for (i, r) in advice.recommendations.iter().enumerate() {
        assert_eq!(r.priority, i + 1, "priority not sequential");
    }

    std::fs::remove_file(&out).ok();
}

/// Pure-CPU (no training) guard: the advisor ranks a weak responsive axis first
/// using only an eval artifact, so this runs even under MOE_SKIP_GPU_TESTS.
#[test]
fn advisor_ranks_without_training() {
    let eval = serde_json::json!({
        "arch": "gpt2",
        "param_count": 110336u64,
        "axis_scores": { "recall": 0.35, "memory": 0.99 },
        "benchmarks": [
            { "axis": "recall", "informational": false, "metrics": { "fields": { "train_ce": 1.3, "init_ce": 2.9 } } },
            { "axis": "memory", "informational": false, "metrics": { "fields": { "train_ce": 1.1, "init_ce": 3.9 } } },
        ],
    });
    let advice = advisor::advise(&eval, None);
    assert!(!advice.recommendations.is_empty());
    assert_eq!(advice.recommendations[0].axis, "recall", "weakest axis should rank first");
}
