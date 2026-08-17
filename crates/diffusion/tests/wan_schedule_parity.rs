// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity for Wan2.1's two flow-matching multistep solvers against the
//! reference implementations (`wan/utils/fm_solvers_unipc.py`,
//! `wan/utils/fm_solvers.py`), driven exactly as `wan/text2video.py` drives
//! them.
//!
//! Fixture: `$BRAIN_TESTDATA/golden/wan/schedule.safetensors` (default
//! `<repo>/testdata`). The test **skips itself** when the fixture is absent.
//!
//! Regenerate with:
//! ```text
//! python3 tools/goldens/wan_schedule_dump_reference.py
//! ```
//!
//! Two different bars, deliberately:
//!
//! * `sigmas` and `timesteps` are **bit-exact** - identical f32 bit patterns,
//!   not a tolerance. They are pure scalar math with no reduction in sight, the
//!   reference computes them in f64 and rounds once, and a schedule that is
//!   merely "close" is a silently different sampler.
//! * the `step()` trajectory carries a tolerance, because the reference
//!   evaluates the solver's scalar coefficients (`log`, `expm1`, the B(h)
//!   solve) in **f32** while this port evaluates them in f64 and rounds once
//!   per element. The deviation that leaves is printed by the test on every
//!   run. Measured: 1.2e-7 (one f32 ULP at this scale) after the first step,
//!   growing smoothly to a worst 5.2e-6 by step 50 of the worst of the 16
//!   (solver, shift, steps) combinations - pure accumulated rounding, with no
//!   jump at step 2 where the second-order path and the corrector first
//!   engage. The bar is 1e-5, roughly 2x the measured worst, which is far
//!   below what any real defect costs: a wrong order, a missed corrector or an
//!   off-by-one in the multistep history all move the trajectory by 1e-2 or
//!   more (mutation-checked).

use brain_testutil::testdata_path;
use diffusion::flowsolvers::{
    FlowDpmSolverConfig, FlowDpmSolverPlusPlusScheduler, FlowUniPcConfig, FlowUniPcScheduler,
};

/// (label, shift, steps) - the dumper's `CASES`, with the label matching its
/// `%g` formatting of the shift.
const CASES: &[(&str, f64, usize)] = &[
    ("5", 5.0, 50),
    ("3", 3.0, 40),
    ("5", 5.0, 40),
    ("3", 3.0, 50),
    ("7.5", 7.5, 25),
    ("16", 16.0, 50),
    ("1", 1.0, 10),
    ("5", 5.0, 4),
];

/// See the module header for why this is not tighter.
const TRAJ_TOL: f64 = 1e-5;

struct Golden(Vec<checkpoint::safetensors::StTensor>);

impl Golden {
    fn open() -> Option<Golden> {
        let p = testdata_path("golden/wan/schedule.safetensors");
        if !p.exists() {
            brain_testutil::skip(&format!("{} absent - run `python3 tools/goldens/wan_schedule_dump_reference.py`",
                p.display()));
            return None;
        }
        Some(Golden(
            checkpoint::safetensors::read(p.to_str().expect("utf-8 path")).expect("read golden"),
        ))
    }

    fn get(&self, name: &str) -> &[f32] {
        &self
            .0
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("golden missing {name}"))
            .data
    }
}

/// The dumper's pseudo-denoiser, reproduced so the trajectory is driven by
/// brain's OWN samples: replaying the reference's model outputs instead would
/// hide a wrong sample behind a right-looking trajectory.
fn pseudo(i: usize, x: &[f32]) -> Vec<f32> {
    let k = 0.7 + 0.1 * i as f32;
    x.iter().map(|&v| (v * k).sin() * 0.9 - 0.05 * i as f32).collect()
}

/// Number of differing f32 bit patterns, and the worst distance in ULPs.
fn bitwise(got: &[f32], want: &[f32]) -> (usize, i64) {
    assert_eq!(got.len(), want.len(), "length mismatch: {} vs {}", got.len(), want.len());
    let mut n = 0;
    let mut worst = 0i64;
    for (&a, &b) in got.iter().zip(want) {
        if a.to_bits() != b.to_bits() {
            n += 1;
            worst = worst.max((a.to_bits() as i64 - b.to_bits() as i64).abs());
        }
    }
    (n, worst)
}

fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch: {} vs {}", got.len(), want.len());
    got.iter().zip(want).map(|(&a, &b)| (a as f64 - b as f64).abs()).fold(0.0, f64::max)
}

#[test]
fn wan_flow_solvers_match_reference() {
    let Some(g) = Golden::open() else { return };
    let x0 = g.get("traj.x0").to_vec();
    let mut rows: Vec<(String, usize, i64, f64)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for solver in ["unipc", "dpmpp"] {
        for &(label, shift, steps) in CASES {
            let pfx = format!("{solver}.shift{label}.steps{steps}");

            // Both are driven the way `text2video.py` drives them: the shift
            // reaches the schedule exactly once.
            let (sigmas, timesteps, traj) = if solver == "unipc" {
                let mut s = FlowUniPcScheduler::new(FlowUniPcConfig::default());
                s.set_timesteps(steps, shift);
                let mut x = x0.clone();
                let mut traj = Vec::with_capacity(steps * x0.len());
                for i in 0..steps {
                    x = s.step(&pseudo(i, &x), &x);
                    traj.extend_from_slice(&x);
                }
                (s.sigmas().to_vec(), s.timesteps().to_vec(), traj)
            } else {
                let mut s = FlowDpmSolverPlusPlusScheduler::new(FlowDpmSolverConfig::default());
                s.set_timesteps(steps, shift);
                let mut x = x0.clone();
                let mut traj = Vec::with_capacity(steps * x0.len());
                for i in 0..steps {
                    x = s.step(&pseudo(i, &x), &x);
                    traj.extend_from_slice(&x);
                }
                (s.sigmas().to_vec(), s.timesteps().to_vec(), traj)
            };

            // -- exact: the schedule ------------------------------------------
            for (what, got) in [("sigmas", &sigmas), ("timesteps", &timesteps)] {
                let want = g.get(&format!("{pfx}.{what}"));
                let (n, ulps) = bitwise(got, want);
                if n != 0 {
                    failures.push(format!(
                        "{pfx}.{what}: {n}/{} entries differ, worst {ulps} ULP",
                        got.len()
                    ));
                }
            }

            // -- tolerance: the trajectory ------------------------------------
            let want = g.get(&format!("{pfx}.traj"));
            let d = max_abs(&traj, want);
            let (n_bits, ulps) = bitwise(&traj, want);
            rows.push((pfx.clone(), n_bits, ulps, d));
            if d > TRAJ_TOL {
                failures.push(format!("{pfx}.traj: max_abs {d:.3e} > {TRAJ_TOL:.0e}"));
            }
        }
    }

    let worst = rows.iter().map(|r| r.3).fold(0.0, f64::max);
    println!("\n{:<28} {:>10} {:>8} {:>12}", "case", "bitdiffs", "ulps", "traj max_abs");
    for (k, n, u, d) in &rows {
        println!("{k:<28} {n:>10} {u:>8} {d:>12.3e}");
    }
    println!(
        "\nschedules bit-exact; {} trajectories, worst max_abs {:.3e}\n",
        rows.len(),
        worst
    );
    assert!(failures.is_empty(), "{} failed:\n  {}", failures.len(), failures.join("\n  "));
}
