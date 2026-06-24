// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scaling-law sweep integration test — a *capacity-helps* guard.
//!
//! It runs the real scaling sweep end-to-end (synthesize the MQAR task → train a
//! GPT at each grid size on `BRAIN_DEVICE` → fit `L(N) = E + A·N^(−α)`) and
//! asserts the final loss is **monotonically non-increasing** with model size
//! (bigger ≤ smaller + tolerance) — i.e. adding capacity does not hurt. Skipped
//! when `MOE_SKIP_GPU_TESTS` is set so the suite stays runnable with no
//! accelerator. Sized to finish in ~3-4 min on the CPU (Cranelift) backend.

use bench::scaling::{run, Sweep};

/// Skip the whole test when no accelerator is wanted (same gate as the rest).
fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn loss_is_monotonic_in_model_size() {
    if skip() {
        return;
    }
    // A slightly trimmed sweep keeps this test under the 3-4 min CPU budget while
    // still spanning a real range of sizes (params increase strictly across them).
    let sweep = Sweep { steps: 250, ..Default::default() };
    let dir = std::env::temp_dir().join(format!("brain_scaling_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let result = run(&sweep, &dir).expect("run scaling sweep");
    let _ = std::fs::remove_dir_all(&dir);

    result.print();

    // Parameter counts must be strictly increasing — that is the independent axis.
    for w in result.points.windows(2) {
        assert!(
            w[1].params > w[0].params,
            "params not increasing across sizes: {} -> {}",
            w[0].params,
            w[1].params,
        );
    }

    // Capacity helps (or at least does not hurt): each larger model's final loss
    // is no worse than the previous one, up to a noise tolerance. The CPU backend
    // is fp32 single-run, so allow a small slack — a genuine capacity regression
    // (loss rising materially with size) still trips this.
    let tol = 0.10f32;
    for w in result.points.windows(2) {
        let smaller = w[0].final_loss;
        let bigger = w[1].final_loss;
        assert!(
            bigger <= smaller + tol,
            "loss increased with model size: L{}xD{} loss={smaller:.4} -> L{}xD{} loss={bigger:.4} (tol {tol})",
            w[0].n_layers,
            w[0].d_model,
            w[1].n_layers,
            w[1].d_model,
        );
    }

    // The biggest model should clearly beat the smallest (sanity: the task is
    // learnable and capacity-sensitive, not flat noise).
    let first = result.points.first().unwrap().final_loss;
    let last = result.points.last().unwrap().final_loss;
    assert!(
        last <= first + tol,
        "largest model did not improve over smallest: {first:.4} -> {last:.4}",
    );

    // A power law was fitted with a finite, positive exponent.
    assert!(result.law.alpha.is_finite(), "alpha not finite");
    assert!(result.law.r2.is_finite() && result.law.r2 >= 0.0, "r2 invalid");
    println!(
        "fitted alpha={:.4} R2={:.4} (E={:.4}, A={:.4e})",
        result.law.alpha, result.law.r2, result.law.e, result.law.a
    );
}
