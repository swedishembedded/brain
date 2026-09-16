// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Run the control matrix and report whether anything learned.
//!
//! This is the experiment, not a test: it takes minutes and its answer is
//! empirical. The test that gates it asserts the CONTROLS behave, which is a
//! claim that holds whether or not the learner works.
use fly::learn::{episode_with, Condition, Lcg, Objective, RewardConfig, Start};
use fly::Reference;
use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};
use neuro::PlasticityParams;
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
    let (neurons, edges) = connectome::find(std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()), "manc").unwrap();
    let c = connectome::load("manc", &neurons, &edges).unwrap();
    let xml = std::env::var("BRAIN_FLYBODY_XML").unwrap();
    let lif = fly::cord_lif();
    // Which objective. Displacement is kept because the ceiling was measured
    // under it and because it is the honest demonstration of why it is wrong;
    // imitation is what flybody's own walking task uses.
    let reference = std::env::var_os("BRAIN_FLY_REFERENCE")
        .filter(|v| !v.is_empty())
        .map(|p| Reference::load(std::path::PathBuf::from(p)).expect("the reference loads"));
    let imitate = reference.is_some() && std::env::var("OBJECTIVE").as_deref() != Ok("displacement");
    let cfg = RewardConfig {
        objective: if imitate {
            Objective::imitate(if std::env::var("START").as_deref() == Ok("fixed") {
                Start::Fixed(0)
            } else {
                // Random by default: DeepMimic's other half. Fixed starts
                // confine every episode to the first fraction of a second of
                // one recording, and the rest of the gait is never seen.
                Start::Random { min_frames: 300 }
            })
        } else {
            Objective::Displacement
        },
        ..RewardConfig::default()
    };
    if let Some(r) = &reference {
        println!("objective: {}", if imitate { "imitation" } else { "displacement" });
        if imitate {
            println!("  reward is a PRODUCT of factors, max {:.0} per tick, episode ends on losing the reference", cfg.objective.max_per_tick());
        }
        println!("  reference: {:?}, tracking {} of {} DoF", r, r.moving_dofs().len(), r.nv());
    } else {
        println!("objective: displacement (set BRAIN_FLY_REFERENCE for imitation)");
    }
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
            Fly::new(gpu, &c, model, lif, Wiring { shuffle_seed: shuffle, ..Wiring::default() }, Timing::default(), Coupling::default()).unwrap();
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
            let e = episode_with(&mut f, cfg, condition, reference.as_ref(), &mut rng).unwrap();
            // Score on the objective actually set. Under imitation, distance
            // is a side observation and reward is the thing being optimised;
            // ranking conditions by distance while rewarding imitation would
            // be scoring a different experiment than the one being run.
            //
            // The episode RETURN, not the per-tick average. An imitation
            // episode ends when the body loses the reference, so a creature
            // that tracks for twice as long earns twice as much - and a
            // per-tick average would hide exactly that, scoring one good tick
            // followed by immediate failure as highly as sustained tracking.
            let score = if imitate { e.reward } else { e.distance };
            dist.push(score);
            per_spike.push(score / (e.spikes.max(1) as f64 / 1e6));
            if i == 0 || i == episodes - 1 {
                if imitate {
                    println!(
                        "  {condition:?} ep{i:3}: return {:.3} over {} ticks ({}), distance {:+.3} BL, spikes {}",
                        e.reward,
                        e.ticks,
                        if e.reached_end {
                            "tracked to the end"
                        } else if e.terminated {
                            "lost the reference"
                        } else {
                            "ran out of budget"
                        },
                        e.distance / BODY_LENGTH_CM,
                        e.spikes
                    );
                } else {
                    println!(
                        "  {condition:?} ep{i:3}: distance {:+.5} cm = {:+.3} BL, spikes {}",
                        e.distance,
                        e.distance / BODY_LENGTH_CM,
                        e.spikes
                    );
                }
            }
        }
        let w1 = f.weights();
        let moved = w0.iter().zip(&w1).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let drift = w0.iter().zip(&w1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{condition:?}: {} of {} weights moved, max drift {drift:.6}", moved, w0.len());
        results.push((condition, dist, per_spike));
    }

    // How a score is read back to a human. Displacement is a distance in
    // centimetres, so body lengths are the meaningful unit; imitation is a
    // per-tick reward whose meaningful unit is its own maximum. Carrying the
    // unit with the number is not decoration - a bare "+0.84" printed with the
    // wrong label is exactly how a null result gets read as a win.
    let (unit, in_units): (&str, Box<dyn Fn(f64) -> String>) = if imitate {
        // The ceiling is a perfect tick held for the whole budget. Reporting
        // against it is what stops a return of 0.8 from reading as progress.
        let max = cfg.objective.max_per_tick() * cfg.ticks as f64;
        ("return", Box::new(move |v: f64| format!("{:.2}% of a perfect {max:.0}", 100.0 * v / max)))
    } else {
        ("cm", Box::new(|v: f64| format!("{:+.3} body lengths", v / BODY_LENGTH_CM)))
    };

    println!("\n--- did anything improve within its own run? (first half vs second half) ---");
    for (cond, d, _) in &results {
        let h = d.len() / 2;
        let (a, b) = (&d[..h], &d[h..]);
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let mb = b.iter().sum::<f64>() / b.len() as f64;
        println!("  {cond:?}: first half {ma:+.5}, second half {mb:+.5}, change {:+.5} {unit}", mb - ma);
    }

    println!("\n--- paired sign test, Learning against each control ---");
    let learning = &results[0].1;
    for (cond, d, _) in &results[1..] {
        let t = sign_test(learning, d);
        println!("  Learning vs {cond:?}: {}/{} episodes better, p = {:.4}", t.k, t.n, t.p_value);
    }

    println!("\n--- normalised by spike count ({unit} per million spikes) ---");
    // Whether the sign test on this metric is independent evidence DEPENDS ON
    // THE OBJECTIVE, and getting that wrong once already produced a reported
    // "corroboration" that was arithmetic.
    //
    // Under displacement, Learning's mean is positive and every control's is
    // negative. Dividing both by a positive spike count cannot reorder a
    // positive against a negative, so the normalised test is GUARANTEED to
    // reproduce the raw one and identical k/n is not a second result.
    //
    // Under imitation every score is positive, so division by differing spike
    // counts genuinely can reorder a pair, and the test carries information.
    //
    // The MEANS carry information under either objective, because a ratio of
    // score to activity is not a restatement of the score. They do not control
    // for activity - that needs conditions matched on firing rate, which this
    // experiment does not do - but they do answer "is Learning simply firing
    // more?".
    for (cond, d, ps) in &results {
        let md = d.iter().sum::<f64>() / d.len() as f64;
        let mp = ps.iter().sum::<f64>() / ps.len() as f64;
        println!("    {cond:?}: raw mean {md:+.5} {unit} ({}), per-Mspike mean {mp:+.5}", in_units(md));
    }
    let learning_ps = &results[0].2;
    for (cond, _, ps) in &results[1..] {
        let t = sign_test(learning_ps, ps);
        println!("  Learning vs {cond:?}: {}/{} episodes better, p = {:.4}", t.k, t.n, t.p_value);
    }

    let best = results[0].1.iter().cloned().fold(f64::MIN, f64::max);
    println!();
    println!(
        "Best single episode: {best:+.5} {unit} = {} over {:.2} s simulated.",
        in_units(best),
        cfg.ticks as f64 * 0.002
    );
    if imitate {
        println!("The maximum is reached only by tracking the recorded gait exactly. Read the");
        println!("sign tests as 'reward-correlated plasticity differs from its controls',");
        println!("which is what they test, and NOT as locomotion.");
    } else {
        println!("A walking fly covers one to three body lengths per SECOND. Read the");
        println!("sign tests as 'reward-correlated plasticity differs from reward-shuffled',");
        println!("which is what they test, and NOT as locomotion.");
    }
}
