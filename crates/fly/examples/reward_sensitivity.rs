// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! How much of the imitation score is about the FLY?
//!
//! An imitation reward compares a body against a recording. If the body
//! barely moves, the comparison is dominated by the recording - the reference
//! walks away, the distance grows on its own schedule, the episode ends, and
//! the return comes out about the same whatever the animal did. A metric like
//! that cannot rank two creatures, and an optimiser climbing it is climbing
//! nothing.
//!
//! This measures that directly. With the start fixed so the reference is
//! identical every time, it sweeps the descending command from silence to
//! saturation and adds a PARALYSED control with every muscle severed. The
//! spread across those conditions is the metric's entire dynamic range, and
//! the paralysed score is its floor: a reward that pays a corpse most of what
//! it pays a driven animal is measuring the recording.
use fly::learn::{episode_with, Condition, Lcg, Objective, RewardConfig, Start};
use fly::{Coupling, Fly, Reference, Timing, Wiring};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn main() {
    let mj = MuJoCo::load().unwrap();
    let dir = std::path::PathBuf::from(env("BRAIN_CONNECTOME_DIR")).join("manc-codex");
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz")).unwrap();
    let model = Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).unwrap();
    let reference = Reference::load(env("BRAIN_FLY_REFERENCE")).expect("the reference loads");
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, lif, Wiring { shuffle_seed: None, ..Wiring::default() }, Timing::default(), Coupling::default()).unwrap();

    let max = 20.0 * 300.0;
    println!("snippet 0, 300-tick budget, a perfect return would be {max:.0}");
    println!("{:>12}  {:>10}  {:>7}  {:>9}  {:>10}", "condition", "return", "ticks", "spikes", "% perfect");

    let mut scores: Vec<(String, f64)> = Vec::new();
    let mut run = |f: &mut Fly, label: &str, command: f32| {
        let cfg = RewardConfig {
            objective: Objective::imitate(Start::Fixed(0)),
            ticks: 300,
            command,
            ..RewardConfig::default()
        };
        let mut rng = Lcg::new(1);
        let e = episode_with(f, cfg, Condition::Frozen, Some(&reference), &mut rng).unwrap();
        println!(
            "{label:>12}  {:>10.3}  {:>7}  {:>9}  {:>9.3}%",
            e.reward,
            e.ticks,
            e.spikes,
            100.0 * e.reward / max
        );
        scores.push((label.to_string(), e.reward));
    };

    for command in [0.0f32, 0.5, 1.0, 2.0, 3.0, 4.0] {
        run(&mut f, &format!("drive {command:.1}"), command);
    }

    // The floor. Every muscle severed, so the nervous system is running at
    // full tilt and nothing whatsoever reaches the body.
    for i in 0..f.actuator_names().len() {
        f.set_muscle_strength(i, 0.0).unwrap();
    }
    run(&mut f, "PARALYSED", 2.0);

    let paralysed = scores.last().expect("the paralysed run ran").1;
    let (best_label, best) = scores[..scores.len() - 1]
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("at least one driven run");
    let (worst_label, worst) = scores[..scores.len() - 1]
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("at least one driven run");

    println!("\nspread across the driven conditions: {worst_label} {worst:.3} to {best_label} {best:.3}");
    println!("  that is {:.1}% of the best, and {:.4}% of a perfect return",
             100.0 * (best - worst) / best.abs().max(1e-9),
             100.0 * (best - worst) / max);
    println!("paralysed: {paralysed:.3}, which is {:.1}% of the best driven score",
             100.0 * paralysed / best.abs().max(1e-9));
    println!("\nRead it this way: the gap between PARALYSED and the best driven score is");
    println!("everything an optimiser has to work with. If it is small, the reward is");
    println!("reporting the recording's own motion and not the animal's, and no learning");
    println!("rule - local or otherwise - can extract a gait from it.");
}
