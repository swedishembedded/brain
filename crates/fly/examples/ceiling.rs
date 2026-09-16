// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! The ceiling: what can these parameters do under a stronger optimiser than a
//! local learning rule?
//!
//! Hill-climbs eleven per-cell-type gains with the wiring fixed, and reports
//! the best it finds against the local rule's own best. The comparison is the
//! point: if the search finds much more, the local rule is weak; if it finds
//! about the same, the limit is the structure, the reward or the body, and no
//! learning rule was going to fix it.
use fly::learn::{episode, Condition, GainSearch, Lcg, RewardConfig};
use fly::{Coupling, Fly, Timing};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

const BODY_LENGTH_CM: f64 = 0.25;

fn main() {
    let mj = MuJoCo::load().unwrap();
    let dir = std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()).join("manc-codex");
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz")).unwrap();
    let xml = std::env::var("BRAIN_FLYBODY_XML").unwrap();
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let cfg = RewardConfig { ticks: 200, ..RewardConfig::default() };
    let evals: usize = std::env::var("EVALS").ok().and_then(|v| v.parse().ok()).unwrap_or(80);

    let search = GainSearch::new(&c, 3e-2);
    println!("searching {} gains: {}", search.groups().len(), search.groups().join(", "));

    let model = Model::from_xml(&mj, &xml).unwrap();
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, lif, 3e-2, None, Timing::default(), Coupling::default()).unwrap();
    let mut rng = Lcg::new(0xCE111);

    // Plasticity OFF throughout: this measures what the WEIGHTS can do, not
    // what a rule can find. Mixing the two would make the result unreadable.
    let evaluate = |f: &mut Fly, gains: &[f32], rng: &mut Lcg| -> f64 {
        f.set_weights(&search.weights(gains)).unwrap();
        episode(f, cfg, Condition::Frozen, rng).unwrap().distance
    };

    let mut best = search.unit_gains();
    let mut best_score = evaluate(&mut f, &best, &mut rng);
    println!("unit gains (the connectome as imported): {best_score:+.5} cm = {:+.3} BL",
             best_score / BODY_LENGTH_CM);

    let mut sigma = 0.4f32;
    let mut since_improve = 0;
    for i in 0..evals {
        let mut trial = best.clone();
        // Perturb every gain; a coordinate-wise search would need many more
        // evaluations to find an interaction between two cell types.
        for g in trial.iter_mut() {
            *g = (*g + sigma * rng.normal()).clamp(0.0, 4.0);
        }
        let score = evaluate(&mut f, &trial, &mut rng);
        if score > best_score {
            best_score = score;
            best = trial;
            since_improve = 0;
            println!("  eval {i:3}: {best_score:+.5} cm = {:+.3} BL  sigma {sigma:.3}",
                     best_score / BODY_LENGTH_CM);
        } else {
            since_improve += 1;
            // Anneal on stagnation rather than on a schedule, so the step size
            // follows the landscape instead of the iteration count.
            if since_improve >= 12 {
                sigma *= 0.6;
                since_improve = 0;
                if sigma < 0.02 {
                    println!("  converged at eval {i}");
                    break;
                }
            }
        }
    }

    println!("\nbest gains found:");
    for (g, name) in best.iter().zip(search.groups()) {
        println!("   {g:6.3}  {name}");
    }
    println!("\nCEILING: {best_score:+.5} cm = {:+.3} body lengths in {:.2} s = {:.3} BL/s",
             best_score / BODY_LENGTH_CM, cfg.ticks as f64 * 0.002,
             best_score / BODY_LENGTH_CM / (cfg.ticks as f64 * 0.002));
    println!("A real fly walks at 1 to 3 BL/s.");
}
