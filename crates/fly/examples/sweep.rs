// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Sweep the cord's physiology in ONE process, and score the result against a
//! real fly.
//!
//! Every parameter sweep here so far has been a shell loop running this
//! crate's examples once per value. Each iteration reloaded the connectome and
//! re-derived the graph - twenty seconds for the cord, minutes for the brain -
//! in order to change one float. Four values meant four full loads for four
//! numbers, and an afternoon disappeared into it.
//!
//! Nothing about that was necessary. The graph depends on the WIRING
//! parameters; the time constants, thresholds and adaptation are
//! `LifParams`, and `SpikingNet::set_params` changes them with no rebuild at
//! all. So the connectome loads once, the graph is derived once, and the whole
//! grid runs against the same network.
//!
//! The gait is measured from the MOTOR NEURONS rather than from the body,
//! which is the other half of the speedup and is also the better measurement.
//! A stepping rhythm is a property of the cord's output; putting it through
//! physics first adds a second system that can hide it, and costs a MuJoCo
//! step per tick. Each leg's coxa drive is its promotor population minus its
//! remotor population - agonist minus antagonist, the signal a step actually
//! swings through - and the six of them go into the same `gait::analyse` that
//! scores the real fly.
//!
//! That last point is what makes the numbers mean anything: a real fly's
//! walking scores 0.490 on this analyser at 8.25 Hz with a tripod of 0.684
//! (`reference_gait`). A row here is to be read against 0.490, not against 1.

use fly::gait::Trace;
use fly::{Cns, LifParams, Wiring};
use neuro::{DynamicalSystem, Port, SpikingNet};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// A comma-separated list of values, or a single default.
fn list(name: &str, default: &[f32]) -> Vec<f32> {
    std::env::var(name)
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<f32>>())
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

/// What a real fly scores on this analyser, measured by `reference_gait`.
const REAL_FLY: f64 = 0.490;
const REAL_FLY_HZ: f64 = 8.25;

fn main() {
    let which = match std::env::var("CNS").unwrap_or_else(|_| "cord".into()).as_str() {
        "brain" | "banc" => Cns::BrainAndCord,
        _ => Cns::Cord,
    };
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });

    // WHICH JOINT the gait is read from, and it cannot be the coxa.
    //
    // MANC annotates the front leg's muscles fully and the other two segments
    // only partially: `Tergopleural/Pleural_promotor` exists on T1 alone, so
    // four of the six legs have no coxa agonist at all and 26 middle-leg motor
    // neurons are filed under the bare name `middle_leg`. Reading a six-leg
    // rhythm therefore needs a joint every segment shares, and the femur pair
    // (`Tr_extensor` against `Tr_flexor`) is the largest that does.
    let dof = match std::env::var("DOF").unwrap_or_else(|_| "femur".into()).as_str() {
        "coxa" => flybody::LegDof::Coxa,
        "tibia" => flybody::LegDof::Tibia,
        "femur" => flybody::LegDof::Femur,
        other => {
            eprintln!("DOF={other} is not one of coxa, femur, tibia");
            std::process::exit(2)
        }
    };

    // Two motor neuron populations per leg. Agonist minus antagonist: a joint
    // whose two muscles fire together goes nowhere, and a sum of magnitudes
    // cannot tell that from a step.
    let mut promotor = vec![Vec::new(); 6];
    let mut remotor = vec![Vec::new(); 6];
    for (i, n) in c.neurons.iter().enumerate() {
        if n.super_class != "motor" || !matches!(n.class.as_str(), "fl" | "ml" | "hl") {
            continue;
        }
        let Some((seg, muscle)) = flybody::parse_sub_class(&n.sub_class) else { continue };
        let Some(side) = flybody::Side::parse(&n.soma_side) else { continue };
        let Some(leg) = flybody::LEGS.iter().position(|(s, d, _)| *s == seg && *d == side) else { continue };
        // Selected through the muscle VOCABULARY rather than by name, because
        // MANC does not annotate every segment with the same muscles: matching
        // the two T1 names directly found both populations on only two of the
        // six legs. `leg_muscle` maps a muscle to the degree of freedom it
        // moves and the direction it moves it, which is the question actually
        // being asked.
        let Some(action) = flybody::leg_muscle(&muscle) else { continue };
        if action.dof != dof {
            continue;
        }
        if action.polarity > 0.0 {
            promotor[leg].push(i as u32);
        } else {
            remotor[leg].push(i as u32);
        }
    }
    let legs_wired = (0..6).filter(|&l| !promotor[l].is_empty() && !remotor[l].is_empty()).count();
    println!(
        "{}: {} neurons, {legs_wired}/6 legs with both {dof:?} populations",
        c.dataset,
        c.neurons.len()
    );
    if legs_wired < 6 {
        eprintln!("not every leg has both an agonist and an antagonist for {dof:?}; try DOF=femur");
        std::process::exit(1);
    }

    // ONE load, ONE graph, ONE network. Everything below varies `LifParams`.
    let wiring = Wiring {
        weight_scale: num("SCALE", 0.3f32),
        inhibitory_gain: num("IE", 1.0f32),
        ..Wiring::default()
    };
    let graph = c.network_balanced(
        wiring.weight_scale,
        wiring.size_limit,
        wiring.min_synapses,
        &std::collections::HashSet::new(),
        wiring.inhibitory_gain,
    );
    let mut net = SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &graph, fly::cord_lif())
        .expect("the cord runs");
    let n = net.port_len(Port::Spike);

    let command_type = std::env::var("DN").unwrap_or_else(|_| "DNg100".into());
    let command: Vec<u32> = if command_type == "all" {
        c.population(|x| x.super_class == "descending")
    } else {
        c.population(|x| x.super_class == "descending" && x.cell_type == command_type)
    };
    if command.is_empty() {
        eprintln!("no descending neuron of type {command_type}");
        std::process::exit(1);
    }

    let ticks: usize = num("TICKS", 1500);
    let settle: usize = num("SETTLE", 250);
    let current: f32 = num("CURRENT", 8.0);

    // The grid. Milliseconds, except the adaptation increment.
    let tau_m = list("TAU_M", &[10.0]);
    let tau_inh = list("TAU_INH", &[20.0]);
    let adapt_tau = list("ADAPT_TAU", &[100.0]);
    let adapt = list("ADAPT", &[0.4]);
    let total = tau_m.len() * tau_inh.len() * adapt_tau.len() * adapt.len();
    println!(
        "{total} configurations, {ticks} ticks each, command {current} to {command_type} ({} cells)",
        command.len()
    );
    println!("a real fly scores {REAL_FLY:.3} at {REAL_FLY_HZ:.2} Hz on this analyser\n");
    println!(
        "{:>6} {:>8} {:>10} {:>6}  {:>7} {:>8} {:>8} {:>8}  {:>6}",
        "tau_m", "tau_inh", "adapt_tau", "adapt", "step Hz", "tripod", "rhythm", "SCORE", "rate"
    );

    let mut best: Option<(f64, String)> = None;
    let mut spike = vec![0.0f32; n];
    for &tm in &tau_m {
        for &ti in &tau_inh {
            for &at in &adapt_tau {
                for &ai in &adapt {
                    let lif = LifParams {
                        dt_over_tau: 2.0 / tm,
                        dt_over_tau_inh: 2.0 / ti,
                        adapt_decay: (-2.0f32 / at).exp(),
                        adapt_increment: ai,
                        ..fly::cord_lif()
                    };
                    if let Err(e) = net.set_params(lif) {
                        println!("{tm:>6} {ti:>8} {at:>10} {ai:>6}  refused: {e}");
                        continue;
                    }
                    net.reset(0);
                    let mut drive = vec![0.0f32; n];
                    for &d in &command {
                        drive[d as usize] = current;
                    }
                    net.drive(Port::Drive, &drive).expect("the drive fits");

                    let mut trace = Trace::new(fly::CONTROL_PERIOD);
                    let mut total_spikes = 0u64;
                    for t in 0..settle + ticks {
                        net.step();
                        net.read(Port::Spike, &mut spike).expect("the readback fits");
                        if t < settle {
                            continue;
                        }
                        let fired = |cells: &[u32]| cells.iter().filter(|&&i| spike[i as usize] > 0.5).count() as f32;
                        let mut per_leg = [0.0f32; 6];
                        for (leg, slot) in per_leg.iter_mut().enumerate() {
                            *slot = fired(&promotor[leg]) - fired(&remotor[leg]);
                        }
                        trace.push(per_leg);
                        total_spikes += spike.iter().filter(|&&s| s > 0.5).count() as u64;
                    }
                    let rate = total_spikes as f64 / n as f64 / (ticks as f64 * fly::CONTROL_PERIOD);
                    match fly::analyse_gait(&trace) {
                        Some(g) => {
                            let s = g.score();
                            println!(
                                "{tm:>6} {ti:>8} {at:>10} {ai:>6}  {:>7.2} {:>8.3} {:>8.3} {:>8.3}  {rate:>6.1}{}",
                                g.step_hz,
                                g.tripod,
                                g.rhythmicity,
                                s,
                                if s >= REAL_FLY { "  AT THE REAL FLY" } else { "" }
                            );
                            let label = format!("tau_m {tm}, tau_inh {ti}, adapt_tau {at}, adapt {ai}");
                            if best.as_ref().is_none_or(|(b, _)| s > *b) {
                                best = Some((s, label));
                            }
                        }
                        None => println!(
                            "{tm:>6} {ti:>8} {at:>10} {ai:>6}  {:>7} {:>8} {:>8} {:>8}  {rate:>6.1}",
                            "-", "-", "-", "no gait"
                        ),
                    }
                }
            }
        }
    }

    match best {
        Some((s, label)) => {
            println!("\nbest {s:.3} ({:.0}% of a real fly) at {label}", 100.0 * s / REAL_FLY);
        }
        None => println!("\nno configuration produced an analysable gait"),
    }
}
