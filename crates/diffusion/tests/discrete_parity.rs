// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity for the discrete (DDPM) schedulers against diffusers' own `step()`
//! outputs on a fixed trajectory, dumped by `tools/sdxl_dump_reference.py`.
//!
//! Fixture: `$BRAIN_TESTDATA/sdxl/schedulers/steps.safetensors` (default
//! `<repo>/testdata`). The test **skips itself** when the fixture is absent.
//!
//! Regenerate with:
//! ```text
//! python3 tools/sdxl_dump_reference.py --sdxl <sdxl-base-1.0> \
//!         --out testdata/sdxl --skip-unet
//! ```
//!
//! What is gated, per `(family, prediction_type, num_steps)`:
//!   * the discrete `timesteps` vector — EXACT (integers), and the one thing
//!     that differs between the four families at the same step count
//!     (DPM-Solver++ builds `N+1` and drops the last);
//!   * the `sigmas` table, including the terminal entry;
//!   * `scale_model_input` at every step (the sigma-space rescale that is
//!     silent when omitted);
//!   * the full `step()` trajectory — every intermediate latent, not just the
//!     last, so a first divergence localises to its step.
//!
//! Tolerance: `1e-5` relative. Not tighter, and the reason is measured and
//! documented on `DiscreteConfig::betas` — brain builds the beta grid in f64
//! where torch builds it in f32, which leaves `max |Δᾱ|/ᾱ = 9.5e-7` and
//! `max |Δσ|/σ = 8.7e-6` before a single step is taken. Every number this test
//! prints is the *measured* deviation, so a real bug (which moves things by
//! orders of magnitude) stays visible under that floor.

use std::path::{Path, PathBuf};

use diffusion::discrete::{
    DdimScheduler, DiscreteConfig, DpmSolverPlusPlusScheduler, EulerAncestralScheduler,
    EulerScheduler, Prediction,
};

/// Relative tolerance; see the module header for why it is not tighter.
const TOL: f64 = 1e-5;

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    Path::new(&root).join(rel)
}

struct Golden(Vec<checkpoint::safetensors::StTensor>);

impl Golden {
    fn open() -> Option<Golden> {
        let p = testdata("sdxl/schedulers/steps.safetensors");
        if !p.exists() {
            eprintln!("SKIP: {} absent (run tools/sdxl_dump_reference.py)", p.display());
            return None;
        }
        Some(Golden(checkpoint::safetensors::read(p.to_str().expect("utf-8 path")).expect("read golden")))
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.0.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden missing {name}")).data
    }

    fn opt(&self, name: &str) -> Option<&[f32]> {
        self.0.iter().find(|t| t.name == name).map(|t| t.data.as_slice())
    }
}

/// `max |got-want| / max(|want|, 1)` — relative where the reference has scale,
/// absolute where it is ~0 (the terminal sigma, and latents that pass through
/// zero).
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch: {} vs {}", got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(&a, &b)| (a as f64 - b as f64).abs() / (b as f64).abs().max(1.0))
        .fold(0.0, f64::max)
}

/// The reference's deterministic pseudo-denoiser (`pseudo` in the dumper).
/// Reproduced here rather than dumped per step because it must be applied to
/// brain's OWN scaled sample — feeding it the reference's would hide a
/// `scale_model_input` bug behind a correct trajectory.
fn pseudo(i: usize, x: &[f32]) -> Vec<f32> {
    let k = 0.7 + 0.1 * i as f32;
    x.iter().map(|&v| (v * k).sin() * 0.9 - 0.05 * i as f32).collect()
}

struct Report {
    rows: Vec<(String, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, label: &str, got: &[f32], want: &[f32]) {
        let d = max_rel(got, want);
        self.rows.push((label.to_string(), d));
        if d > TOL {
            self.failures.push(format!("{label}: max_rel {d:.3e} > {TOL:.0e}"));
        }
    }
}

fn cfg_for(pred: &str) -> DiscreteConfig {
    let p = match pred {
        "epsilon" => Prediction::Epsilon,
        "v_prediction" => Prediction::VPrediction,
        other => panic!("unknown prediction type {other}"),
    };
    DiscreteConfig::sdxl().with_prediction(p)
}

#[test]
fn discrete_schedulers_match_diffusers() {
    let Some(g) = Golden::open() else { return };
    let mut r = Report { rows: Vec::new(), failures: Vec::new() };

    // The chain itself first: if betas/alphas_cumprod are wrong, every family
    // below is wrong for the same reason and the table should say so once.
    let cfg = DiscreteConfig::sdxl();
    r.check("chain.betas", &cfg.betas(), g.get("chain.betas"));
    r.check("chain.alphas_cumprod", &cfg.alphas_cumprod(), g.get("chain.alphas_cumprod"));

    let x0 = g.get("traj.x0").to_vec();
    let noise = g.get("traj.noise").to_vec();
    let n_elem = x0.len();

    for pred in ["epsilon", "v_prediction"] {
        let cfg = cfg_for(pred);
        for &n in &[4usize, 20] {
            // ---- DDIM (alpha-bar space; no scale_model_input) ------------
            {
                let pfx = format!("ddim.{pred}.{n}");
                let mut s = DdimScheduler::new(cfg);
                s.set_timesteps(n);
                r.check(&format!("{pfx}.timesteps"), s.timesteps(), g.get(&format!("{pfx}.timesteps")));
                let mut x = x0.clone();
                let mut traj = Vec::with_capacity(n * n_elem);
                for i in 0..n {
                    let m = pseudo(i, &x);
                    let (next, _x0pred) = s.step(&m, &x);
                    traj.extend_from_slice(&next);
                    x = next;
                }
                r.check(&format!("{pfx}.traj"), &traj, g.get(&format!("{pfx}.traj")));
            }

            // ---- Euler (sigma space) --------------------------------------
            {
                let pfx = format!("euler.{pred}.{n}");
                let mut s = EulerScheduler::new(cfg);
                s.set_timesteps(n);
                r.check(&format!("{pfx}.timesteps"), s.timesteps(), g.get(&format!("{pfx}.timesteps")));
                r.check(&format!("{pfx}.sigmas"), s.sigmas(), g.get(&format!("{pfx}.sigmas")));
                r.check(
                    &format!("{pfx}.init_noise_sigma"),
                    &[s.init_noise_sigma()],
                    g.get(&format!("{pfx}.init_noise_sigma")),
                );
                let (mut x, mut traj, mut scaled_all) = (x0.clone(), Vec::new(), Vec::new());
                for i in 0..n {
                    let scaled = s.scale_model_input(&x);
                    scaled_all.extend_from_slice(&scaled);
                    let m = pseudo(i, &scaled);
                    x = s.step(&m, &x);
                    traj.extend_from_slice(&x);
                }
                r.check(&format!("{pfx}.scaled"), &scaled_all, g.get(&format!("{pfx}.scaled")));
                r.check(&format!("{pfx}.traj"), &traj, g.get(&format!("{pfx}.traj")));
            }

            // ---- Euler-ancestral (sigma space, golden noise) --------------
            {
                let pfx = format!("euler_a.{pred}.{n}");
                let mut s = EulerAncestralScheduler::new(cfg);
                s.set_timesteps(n);
                r.check(&format!("{pfx}.timesteps"), s.timesteps(), g.get(&format!("{pfx}.timesteps")));
                r.check(&format!("{pfx}.sigmas"), s.sigmas(), g.get(&format!("{pfx}.sigmas")));
                r.check(
                    &format!("{pfx}.init_noise_sigma"),
                    &[s.init_noise_sigma()],
                    g.get(&format!("{pfx}.init_noise_sigma")),
                );
                let (mut x, mut traj, mut scaled_all) = (x0.clone(), Vec::new(), Vec::new());
                for i in 0..n {
                    let scaled = s.scale_model_input(&x);
                    scaled_all.extend_from_slice(&scaled);
                    let m = pseudo(i, &scaled);
                    let z = &noise[i * n_elem..(i + 1) * n_elem];
                    x = s.step_with_noise(&m, &x, z);
                    traj.extend_from_slice(&x);
                }
                r.check(&format!("{pfx}.scaled"), &scaled_all, g.get(&format!("{pfx}.scaled")));
                r.check(&format!("{pfx}.traj"), &traj, g.get(&format!("{pfx}.traj")));
            }

            // ---- DPM-Solver++(2M) ----------------------------------------
            {
                let pfx = format!("dpmpp.{pred}.{n}");
                let mut s = DpmSolverPlusPlusScheduler::new(cfg);
                s.set_timesteps(n);
                r.check(&format!("{pfx}.timesteps"), s.timesteps(), g.get(&format!("{pfx}.timesteps")));
                if let Some(want) = g.opt(&format!("{pfx}.sigmas")) {
                    r.check(&format!("{pfx}.sigmas"), s.sigmas(), want);
                }
                let (mut x, mut traj, mut scaled_all) = (x0.clone(), Vec::new(), Vec::new());
                for i in 0..n {
                    let scaled = s.scale_model_input(&x);
                    scaled_all.extend_from_slice(&scaled);
                    let m = pseudo(i, &scaled);
                    x = s.step(&m, &x);
                    traj.extend_from_slice(&x);
                }
                r.check(&format!("{pfx}.scaled"), &scaled_all, g.get(&format!("{pfx}.scaled")));
                r.check(&format!("{pfx}.traj"), &traj, g.get(&format!("{pfx}.traj")));
            }
        }
    }

    let worst = r.rows.iter().cloned().fold(("".to_string(), 0.0f64), |a, b| if b.1 > a.1 { b } else { a });
    println!("\n{:<34} {:>12}", "check", "max_rel");
    for (k, v) in &r.rows {
        println!("{k:<34} {v:>12.3e}");
    }
    println!("\n{} checks, worst {} at {:.3e}\n", r.rows.len(), worst.0, worst.1);
    assert!(r.failures.is_empty(), "{} failed:\n  {}", r.failures.len(), r.failures.join("\n  "));
}
