// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Is the cord producing a gait, and what has to change before it does?
//!
//! The body walks at 2.67 body lengths per second under a scripted alternating
//! tripod, so locomotion is reachable through these actuators. What was
//! missing is the rhythm. This measures the cord's own motor output against
//! the same gait criterion across two axes that the literature on
//! connectome-derived circuits says should matter, and one that is the control
//! for both.
//!
//! **Excitability.** A leaky membrane's input resistance goes as one over its
//! area, so a uniform threshold makes the largest cells in the cord orders of
//! magnitude more excitable than the smallest. Scaling each neuron's input by
//! its own reconstructed membrane area is the correction; running without it
//! is the control.
//!
//! **Which descending neurons are driven.** A fly has over a thousand
//! descending neurons commanding different, sometimes opposing behaviours.
//! Driving all of them at once is not "go", it is every command simultaneously,
//! and what reaches the muscles is whatever survives the collision. Driving one
//! named cell type is what a stimulation experiment actually does.
//!
//! The first row is a scripted 12 Hz tripod put through the identical
//! measurement, so nothing below it is read without a scale.
use fly::gait::{analyse, scripted_tripod, Trace};
use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};

const CONTROL_DT: f64 = 0.002;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn header() {
    println!(
        "{:>26}  {:>7}  {:>8}  {:>8}  {:>8}  {:>7}  speed",
        "condition", "step Hz", "tripod", "rhythm", "power", "score"
    );
}

fn row(label: &str, t: &Trace, distance: Option<f64>) {
    match analyse(t) {
        Some(g) => println!(
            "{label:>26}  {:>7.2}  {:>8.3}  {:>8.3}  {:>8.4}  {:>7.3}  {}",
            g.step_hz,
            g.tripod,
            g.rhythmicity,
            g.power,
            g.score(),
            match distance {
                Some(d) => format!("{:+.3} BL/s", d / 0.25 / t.duration()),
                None => String::from("-"),
            }
        ),
        None => println!("{label:>26}  trace too short to analyse"),
    }
}

fn main() {
    let mj = MuJoCo::load().unwrap();
    let (neurons, edges) = connectome::find(std::path::PathBuf::from(env("BRAIN_CONNECTOME_DIR")), "manc").unwrap();
    let c = connectome::load("manc", &neurons, &edges).unwrap();
    let lif = fly::cord_lif();
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let cell = std::env::var("CELL").unwrap_or_else(|_| "DNg100".to_string());

    header();
    row("SCRIPTED 12 Hz", &scripted_tripod(12.0, CONTROL_DT, 1000), None);

    for size_limit in [None, Some(10.0f32)] {
        let tag = if size_limit.is_some() { "sized" } else { "uniform" };
        let model = Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).unwrap();
        let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
        let mut f = Fly::new(
            gpu,
            &c,
            model,
            lif,
            Wiring { size_limit, ..Wiring::default() },
            Timing::default(),
            Coupling::default(),
        )
        .unwrap();

        let trace = |f: &mut Fly, label: &str| {
            let x0 = f.qpos()[0];
            let mut t = Trace::new(CONTROL_DT);
            for _ in 0..ticks {
                f.step().unwrap();
                t.push(f.leg_swing());
            }
            row(label, &t, Some(f.qpos()[0] - x0));
        };

        // The whole descending population at once, which is where this started.
        for command in [1.0f32, 2.0, 4.0] {
            f.reset();
            f.set_descending(&vec![command; f.descending_count()]).unwrap();
            trace(&mut f, &format!("{tag} all DNs {command:.1}"));
        }

        // One named cell type. The literature's walking command neuron is
        // DNg100; $CELL overrides it so this is a screen rather than a claim.
        for current in [50.0f32, 150.0, 250.0, 500.0] {
            f.reset();
            match f.drive_cell_type(&cell, current) {
                Ok(n) => {
                    let label = format!("{tag} {cell} x{n} @{current:.0}");
                    trace(&mut f, &label);
                }
                Err(e) => {
                    println!("  {e}");
                    break;
                }
            }
        }
    }

    println!("\nThe scripted row is what a gait scores. Read every other row against it.");
    println!("'tripod' near -1 means all six legs move together, which is a hop, not a walk;");
    println!("'rhythm' near 0 means there is no dominant frequency at all, whatever the peak.");
}
