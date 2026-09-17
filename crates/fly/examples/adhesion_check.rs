// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Does the fly's feet gripping change whether it walks?
//!
//! flybody's adhesion actuators exist because this animal's walking depends on
//! them, and nothing about driving the leg joints engages them. Every walking
//! measurement in this crate before now ran with all eight at zero, which is a
//! fly walking on ice: the rhythm is there, the legs move, and the feet slide
//! out from under it.
//!
//! The grip is commanded by the cord rather than by the simulator noticing
//! that a foot is down. MANC names the muscles - `Ta_depressor` presses the
//! tarsus onto the substrate, `Ta_levator` lifts it - so the claw is on the
//! same footing as every other muscle here.
use fly::learn::{episode, Condition, Lcg, Objective, RewardConfig};
use fly::{Cns, Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let mj = MuJoCo::load().unwrap();
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), Cns::Cord).unwrap();
    let model_path = std::path::PathBuf::from(env("BRAIN_FLYBODY_XML"));
    let wiring = Wiring::default();
    let ticks: u32 = num("TICKS", 5000);
    let command: f32 = num("COMMAND", 1.0);

    // The command the published screen identified, rather than a barrage of
    // every descending neuron at once.
    let dn: Option<&'static str> = match std::env::var("DN").unwrap_or_else(|_| "DNg100".into()).as_str() {
        "all" | "" => None,
        "DNg100" => Some("DNg100"),
        "DNb08" => Some("DNb08"),
        other => {
            eprintln!("DN={other} is not one this example knows; use DNg100, DNb08 or all");
            std::process::exit(2)
        }
    };
    let gains: Vec<f32> = std::env::var("GAINS")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![0.0, 0.05, 0.15, 0.4, 1.0]);

    println!(
        "holding {ticks} ticks ({:.0} s), command {command} to {}",
        ticks as f64 * fly::CONTROL_PERIOD,
        dn.unwrap_or("every descending neuron")
    );
    println!("\n{:>10}  {:>9}  {:>9}  {:>8}  {:>7}  {:>9}", "adhesion", "net cm", "BL/s", "gait", "tipped", "verdict");
    for &g in &gains {
        let model = Model::from_xml(&mj, &model_path).unwrap();
        let coupling = Coupling { adhesion_gain: g, ..Coupling::default() };
        let mut f =
            Fly::new(gpu_core::testgpu::dev(&neuro::KERNELS), &c, model, fly::cord_lif(), wiring, Timing::default(), coupling)
                .unwrap();
        if f.motor_map().adhesion.is_empty() {
            eprintln!("this body has no adhesion actuators, or the cord names no tarsus muscles");
            std::process::exit(1);
        }
        let cfg = RewardConfig {
            objective: Objective::walk(),
            ticks,
            command,
            command_type: dn,
            ..RewardConfig::default()
        };
        let e = episode(&mut f, cfg, Condition::Frozen, &mut Lcg::new(1)).expect("an episode runs");
        let seconds = e.ticks.max(1) as f64 * fly::CONTROL_PERIOD;
        println!(
            "{g:>10.2}  {:>9.4}  {:>9.3}  {:>8.3}  {:>7.2}  {}",
            e.net,
            e.net / 0.25 / seconds,
            e.gait.map_or(0.0, |x| x.score()),
            e.tipped,
            if e.terminated { "FELL" } else { "stayed up" }
        );
    }
    println!("\nadhesion 0.00 is every walking measurement this crate made before the claws");
    println!("were wired. If the rows above differ, they were all made on ice.");
}
