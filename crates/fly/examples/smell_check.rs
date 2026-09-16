// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Does the nose carry anything, and does what it carries reach the legs?
//!
//! This crate has already had a sensory channel that was connected, silent,
//! and indistinguishable from a working one by every other reading: current
//! flowed into the proprioceptors every tick, none of them ever reached
//! threshold, lesioning the channel changed nothing, and the cord fired and
//! the body moved throughout. So a new channel gets this before it gets used
//! for anything.
//!
//! Four things are measured at each gain, and the first three are the ones
//! that separate the failure modes:
//!
//!   antennae   do the receptor neurons FIRE, or merely receive current
//!   brain      does the brain respond, or does it swallow the input
//!   descending do the cells that carry commands to the cord respond
//!   motor      does anything reach a muscle
//!
//! A row with spiking antennae and a silent cord is a real result about the
//! wiring. A row with silent antennae is a result about this file's gain and
//! nothing else, which is exactly the confusion the sweep exists to prevent.
use fly::{Cns, Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn main() {
    let mj = MuJoCo::load().unwrap();
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), Cns::BrainAndCord).unwrap();
    let (left, right) = {
        let a = fly::Antennae::of(&c);
        (a.left.len(), a.right.len())
    };
    println!("{} neurons, {left} + {right} olfactory receptor neurons", c.neurons.len());
    if left == 0 {
        println!("this nervous system has no nose; run it on a brain, not a bare cord");
        return;
    }

    let brain: Vec<u32> = c.population(|n| n.region == "central_brain" || n.region == "optic_lobe");
    let dns: Vec<u32> = c.population(|n| n.super_class == "descending");
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);

    // Four stimuli per gain, and the FIRST is the one that makes the other
    // three mean anything. A brain with a connectome in it fires whether or
    // not anything is smelled - measured, several thousand spikes a tick - so
    // "the brain responded to the odour" is only a claim about the odour if
    // the same numbers were taken with no odour at all.
    //
    // Left-only and right-only are the steering question: a nervous system
    // that responds to a smell but responds IDENTICALLY whichever antenna it
    // arrives at cannot turn towards it, however strong the response.
    let stimuli: [(&str, f32, f32); 4] = [("none", 0.0, 0.0), ("both", 1.0, 1.0), ("left", 1.0, 0.0), ("right", 0.0, 1.0)];

    println!(
        "\n{:>8}  {:>8}  {:>10}  {:>10}  {:>11}  {:>10}  {:>6} {:>6}",
        "gain", "odour", "antennae", "brain", "descending", "motor/tick", "Lleg", "Rleg"
    );
    for gain in [3.0f32, 10.0, 30.0, 100.0] {
        let mut dn_by_side = [0.0f64; 4];
        let mut leg_by_side = [[0.0f64; 2]; 4];
        for (k, (label, l, r)) in stimuli.iter().enumerate() {
            let model = Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).unwrap();
            let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
            let coupling = Coupling { odour_gain: gain, ..Coupling::default() };
            let mut f =
                Fly::new(gpu, &c, model, fly::cord_lif(), Wiring::default(), Timing::default(), coupling).unwrap();
            f.smell(*l, *r);
            let (mut ant, mut motor, mut brain_spikes, mut dn_spikes) = (0u64, 0u64, 0u64, 0u64);
            // Which SIDE's legs are being driven, which is what a turn is.
            // Summed over the coxa - the fore-aft swing that carries a step -
            // and split by the leg's own side rather than by the odour's.
            let mut legs = [0.0f64; 2];
            for _ in 0..ticks {
                let t = f.step().unwrap();
                let (a, b) = f.antenna_spikes();
                ant += (a + b) as u64;
                motor += t.motor_spikes as u64;
                let s = f.spikes();
                brain_spikes += brain.iter().filter(|&&i| s[i as usize] > 0.5).count() as u64;
                dn_spikes += dns.iter().filter(|&&i| s[i as usize] > 0.5).count() as u64;
                let opposed = f.leg_opposed();
                for (i, (_, side, _)) in flybody::LEGS.iter().enumerate() {
                    let drive = (opposed[i][0] + opposed[i][1]) as f64;
                    legs[usize::from(*side == flybody::Side::Right)] += drive;
                }
            }
            let per = |x: u64| x as f64 / ticks as f64;
            dn_by_side[k] = per(dn_spikes);
            leg_by_side[k] = [legs[0] / ticks as f64, legs[1] / ticks as f64];
            println!(
                "{gain:>8.0}  {label:>8}  {:>10.2}  {:>10.1}  {:>11.2}  {:>10.2}  {:>6.2} {:>6.2}",
                per(ant),
                per(brain_spikes),
                per(dn_spikes),
                per(motor),
                leg_by_side[k][0],
                leg_by_side[k][1]
            );
        }
        // The steering number. A smell on the left should not drive both sides
        // of the body the same way, or the animal can only go faster and
        // slower. Positive means a left-side odour drives the LEFT legs
        // harder than a right-side odour does, relative to the other side.
        let turn = (leg_by_side[2][0] - leg_by_side[2][1]) - (leg_by_side[3][0] - leg_by_side[3][1]);
        println!(
            "{:>8}  {:>8}  descending {:+.2}/tick over the control; left-vs-right descending {:+.2}, LEG BIAS {:+.3}",
            "",
            "->",
            dn_by_side[1] - dn_by_side[0],
            dn_by_side[2] - dn_by_side[3],
            turn
        );
    }
    println!("\nantennae is receptor spikes per tick out of {}; a zero there means this", left + right);
    println!("sweep measured its own gain rather than the animal. The `none` row is the");
    println!("control: without it, a brain that fires anyway reads as a brain that smelled.");
}
