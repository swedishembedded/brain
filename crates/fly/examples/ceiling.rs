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
use fly::learn::{episode_with, Condition, GainSearch, Lcg, Objective, RewardConfig, Start};
use fly::{Coupling, Fly, Reference, Timing, Wiring};
use mujoco::{Model, MuJoCo};

const BODY_LENGTH_CM: f64 = 0.25;

fn main() {
    let mj = MuJoCo::load().unwrap();
    let (neurons, edges) = connectome::find(std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()), "manc").unwrap();
    let c = connectome::load("manc", &neurons, &edges).unwrap();
    let xml = std::env::var("BRAIN_FLYBODY_XML").unwrap();
    let lif = fly::cord_lif();
    // A FIXED start, deliberately, when imitating. The search compares one
    // evaluation against another, so the objective has to be the same function
    // each time; a random start would make every evaluation a different
    // problem and the hill-climb would spend its budget chasing that noise
    // rather than the gains. Random starts are for LEARNING, where the point
    // is to see the whole trajectory.
    let reference = std::env::var_os("BRAIN_FLY_REFERENCE")
        .filter(|v| !v.is_empty())
        .map(|p| Reference::load(std::path::PathBuf::from(p)).expect("the reference loads"));
    let imitate = reference.is_some();
    let cfg = RewardConfig {
        objective: if imitate { Objective::imitate(Start::Fixed(0)) } else { Objective::Displacement },
        ticks: 200,
        ..RewardConfig::default()
    };
    println!("objective: {}", if imitate { "imitation (episode return)" } else { "displacement" });
    let evals: usize = std::env::var("EVALS").ok().and_then(|v| v.parse().ok()).unwrap_or(80);

    let search = GainSearch::new(&c, 3e-2);
    println!("searching {} gains: {}", search.groups().len(), search.groups().join(", "));

    let model = Model::from_xml(&mj, &xml).unwrap();
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, lif, Wiring { shuffle_seed: None, ..Wiring::default() }, Timing::default(), Coupling::default()).unwrap();
    let mut rng = Lcg::new(0xCE111);

    // Plasticity OFF throughout: this measures what the WEIGHTS can do, not
    // what a rule can find. Mixing the two would make the result unreadable.
    let evaluate = |f: &mut Fly, gains: &[f32], rng: &mut Lcg| -> f64 {
        f.set_weights(&search.weights(gains)).unwrap();
        let e = episode_with(f, cfg, Condition::Frozen, reference.as_ref(), rng).unwrap();
        if imitate {
            e.reward
        } else {
            e.distance
        }
    };

    let describe = |v: f64| -> String {
        if imitate {
            format!("return {v:+.3} ({:.2}% of a perfect {:.0})",
                    100.0 * v / (cfg.objective.max_per_tick() * cfg.ticks as f64),
                    cfg.objective.max_per_tick() * cfg.ticks as f64)
        } else {
            format!("{v:+.5} cm = {:+.3} BL", v / BODY_LENGTH_CM)
        }
    };

    let mut best = search.unit_gains();
    let mut best_score = evaluate(&mut f, &best, &mut rng);
    println!("unit gains (the connectome as imported): {}", describe(best_score));

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
            println!("  eval {i:3}: {}  sigma {sigma:.3}", describe(best_score));
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
    println!("\nCEILING: {} over {:.2} s simulated", describe(best_score), cfg.ticks as f64 * 0.002);
    if imitate {
        println!("The maximum is reached only by tracking the recorded gait exactly.");
        println!("What this measures is whether ANY setting of these gains can score under");
        println!("imitation. If it cannot, the limit is the coupling or the parameterisation,");
        println!("and no learning rule was going to find what is not there.");
    } else {
        println!("That is {:.3} BL/s. A real fly walks at 1 to 3 BL/s.",
                 best_score / BODY_LENGTH_CM / (cfg.ticks as f64 * 0.002));
    }
}
