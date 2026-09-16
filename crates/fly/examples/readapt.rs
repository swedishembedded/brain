// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Break a muscle and see whether the cord compensates.
//!
//! The last entry in the definition-of-done control matrix: "body perturbed +
//! plasticity on -> re-adapts". Three phases, and the third is what makes the
//! second mean anything.
//!
//! 1. Train with an intact body.
//! 2. Sever one leg's muscles. Continue training.
//! 3. Rewind to exactly the weights phase 1 ended with, sever the same
//!    muscles, and run the same episodes with plasticity OFF.
//!
//! Phase 3 is the control, and it is not optional. A body that recovers on its
//! own - because the physics settles, because the other legs take the load -
//! looks identical to one whose nervous system re-adapted, and only a frozen
//! run over the same perturbation can tell them apart.
use fly::learn::{episode_with, Condition, Lcg, Objective, RewardConfig, Start};
use fly::{Coupling, Fly, Reference, Timing};
use mujoco::{Model, MuJoCo};
use neuro::{LifParams, PlasticityParams};
use promote::stats::sign_test;

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
    let mut f = Fly::new(gpu, &c, model, lif, 3e-2, None, Timing::default(), Coupling::default()).unwrap();
    let bound = 1.5 * f.initial_weight_scale();
    f.enable_plasticity(PlasticityParams { eta: 0.02, w_min: -bound, w_max: bound, ..Default::default() }).unwrap();

    let episodes: usize = std::env::var("EPISODES").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    // Random starts: this phase is learning, so the creature should see the
    // whole recording rather than the first fraction of one snippet.
    let cfg = RewardConfig { objective: Objective::imitate(Start::Random { min_frames: 300 }), ticks: 300, ..RewardConfig::default() };
    let lesion = std::env::var("LESION").unwrap_or_else(|_| "_T1_left".to_string());

    let run = |f: &mut Fly, n: usize, condition: Condition, seed: u64| -> Vec<f64> {
        // The SAME seed for every phase, so two phases see the same sequence
        // of reference snippets. Without that, a phase could score better
        // simply by drawing easier starts.
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| episode_with(f, cfg, condition, Some(&reference), &mut rng).unwrap().reward)
            .collect()
    };

    println!("phase 1: intact body, plasticity on, {episodes} episodes");
    let intact = run(&mut f, episodes, Condition::Learning, 0xA11);
    let trained = f.weights();
    println!("  mean return {:.3}", mean(&intact));

    let hit = f.lesion_matching(&lesion, 0.0).expect("the lesion matches something");
    println!("\nlesion: {hit} actuators matching {lesion:?} severed");

    println!("phase 2: perturbed body, plasticity ON, {episodes} episodes");
    let adapting = run(&mut f, episodes, Condition::Learning, 0xB22);
    println!("  mean return {:.3}", mean(&adapting));

    // Rewind to exactly where phase 1 ended. The lesion stays.
    f.set_weights(&trained).unwrap();
    println!("\nphase 3 (CONTROL): same weights, same lesion, plasticity OFF");
    let frozen = run(&mut f, episodes, Condition::Frozen, 0xB22);
    println!("  mean return {:.3}", mean(&frozen));

    println!("\n--- did the nervous system recover, or did the body? ---");
    let t = sign_test(&adapting, &frozen);
    println!("  plastic vs frozen under the same lesion: {}/{} episodes better, p = {:.4}", t.k, t.n, t.p_value);
    println!("  cost of the lesion, frozen:  {:+.3} ({:.1}% of intact)",
             mean(&frozen) - mean(&intact), 100.0 * mean(&frozen) / mean(&intact));
    println!("  cost of the lesion, plastic: {:+.3} ({:.1}% of intact)",
             mean(&adapting) - mean(&intact), 100.0 * mean(&adapting) / mean(&intact));
    println!("\nRe-adaptation is 'plastic recovers more of the intact score than frozen does'.");
    println!("A plastic run that merely differs from frozen is not re-adaptation; the");
    println!("direction and the sign test together are the claim.");
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
