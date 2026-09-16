// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Run the control matrix and report whether anything learned.
//!
//! This is the experiment, not a test: it takes minutes and its answer is
//! empirical. The test that gates it asserts the CONTROLS behave, which is a
//! claim that holds whether or not the learner works.
use fly::learn::{episode, Condition, Lcg, RewardConfig};
use fly::{Coupling, Fly, Timing};
use mujoco::{Model, MuJoCo};
use neuro::{LifParams, PlasticityParams};
use promote::stats::sign_test;

/// Body length of *Drosophila melanogaster*, in the model's own units.
///
/// flybody's `<option gravity="0 0 -981">` fixes those units as CENTIMETRES,
/// and an adult fruit fly is about 2.5 mm long. Every distance below is
/// reported in body lengths as well as raw units, because a raw figure like
/// "+0.008" invites being read as success when it is 1/30th of a body length
/// and a walking fly covers one to three body lengths per SECOND.
const BODY_LENGTH_CM: f64 = 0.25;

fn main() {
    let mj = MuJoCo::load().unwrap();
    let dir = std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()).join("manc-codex");
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz")).unwrap();
    let xml = std::env::var("BRAIN_FLYBODY_XML").unwrap();
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let cfg = RewardConfig::default();
    let episodes: usize = std::env::var("EPISODES").ok().and_then(|v| v.parse().ok()).unwrap_or(12);

    let mut results: Vec<(Condition, Vec<f64>, Vec<f64>)> = Vec::new();
    for condition in [
        Condition::Learning,
        Condition::Frozen,
        Condition::ShuffledReward,
        Condition::ShuffledConnectome,
    ] {
        let model = Model::from_xml(&mj, &xml).unwrap();
        let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
        // The structural control differs in its WIRING, which is fixed at
        // construction; every other condition runs the real connectome.
        let shuffle = (condition == Condition::ShuffledConnectome).then_some(0x5EEDu64);
        let mut f =
            Fly::new(gpu, &c, model, lif, 3e-2, shuffle, Timing::default(), Coupling::default()).unwrap();
        // Clamp sized from the connectome's OWN weight range, not picked. The
        // weights are scaled synapse counts running to tens, so a fixed +/-0.2
        // squashes every one of them to the bound on the first update and
        // destroys the graph - which is what the first run of this experiment
        // did, reporting a max drift of 32.2 against a 0.2 clamp.
        let bound = 1.5 * f.initial_weight_scale();
        f.enable_plasticity(PlasticityParams { eta: 0.02, w_min: -bound, w_max: bound, ..Default::default() }).unwrap();
        println!("{condition:?}: weight clamp +/-{bound:.3}");
        let mut rng = Lcg::new(0xF1A);
        let mut dist = Vec::new();
        // Distance per million spikes. The conditions do NOT end up equally
        // active - Learning finishes firing several times more than it
        // started - so a distance comparison alone cannot separate "moved
        // further because it coordinated" from "moved further because it
        // flailed more". This is the normalised view; both are reported.
        let mut per_spike = Vec::new();
        let w0 = f.weights();
        for i in 0..episodes {
            let e = episode(&mut f, cfg, condition, &mut rng).unwrap();
            dist.push(e.distance);
            per_spike.push(e.distance / (e.spikes.max(1) as f64 / 1e6));
            if i == 0 || i == episodes - 1 {
                println!(
                    "  {condition:?} ep{i:3}: distance {:+.5} cm = {:+.3} body lengths  spikes {}  proprio {}",
                    e.distance, e.distance / BODY_LENGTH_CM, e.spikes, e.proprio_spikes
                );
            }
        }
        let w1 = f.weights();
        let moved = w0.iter().zip(&w1).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let drift = w0.iter().zip(&w1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{condition:?}: {} of {} weights moved, max drift {drift:.6}", moved, w0.len());
        results.push((condition, dist, per_spike));
    }

    println!("\n--- did anything improve within its own run? (first half vs second half) ---");
    for (cond, d, _) in &results {
        let h = d.len() / 2;
        let (a, b) = (&d[..h], &d[h..]);
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let mb = b.iter().sum::<f64>() / b.len() as f64;
        println!(
            "  {cond:?}: first half {ma:+.5}, second half {mb:+.5}, change {:+.5} ({:+.3} body lengths)",
            mb - ma,
            (mb - ma) / BODY_LENGTH_CM
        );
    }

    println!("\n--- paired sign test, Learning against each control ---");
    let learning = &results[0].1;
    for (cond, d, _) in &results[1..] {
        let t = sign_test(learning, d);
        println!("  Learning vs {cond:?}: {}/{} episodes better, p = {:.4}", t.k, t.n, t.p_value);
    }

    println!("\n--- the same, normalised by spike count (cm per million spikes) ---");
    // Print the normalised means too. An identical sign-test result before and
    // after normalising is the expected outcome when the distance gap dominates
    // the activity gap, but it is also what a normalisation that silently did
    // nothing would produce, so show the numbers rather than trusting the test.
    for (cond, d, ps) in &results {
        let md = d.iter().sum::<f64>() / d.len() as f64;
        let mp = ps.iter().sum::<f64>() / ps.len() as f64;
        let sp: f64 = d.len() as f64;
        let _ = sp;
        println!("    {cond:?}: raw mean {md:+.5} cm, per-Mspike mean {mp:+.5} cm");
    }
    let learning_ps = &results[0].2;
    for (cond, _, ps) in &results[1..] {
        let t = sign_test(learning_ps, ps);
        println!("  Learning vs {cond:?}: {}/{} episodes better, p = {:.4}", t.k, t.n, t.p_value);
    }

    let best = results[0].1.iter().cloned().fold(f64::MIN, f64::max);
    println!();
    println!("Best single episode: {best:+.5} cm = {:+.3} body lengths in {:.2} s simulated.", best / BODY_LENGTH_CM, cfg.ticks as f64 * 0.002);
    println!("A walking fly covers one to three body lengths per SECOND. Read the");
    println!("sign tests as 'reward-correlated plasticity differs from reward-shuffled',");
    println!("which is what they test, and NOT as locomotion.");
}
