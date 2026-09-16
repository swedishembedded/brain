// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Is the cord producing a gait, and if not, what is it producing?
//!
//! The body walks at 2.67 body lengths per second under a scripted
//! alternating tripod, so locomotion is reachable through these actuators.
//! What is missing is the rhythm. This measures the cord's own motor output
//! against the same gait criterion - stepping frequency, tripod antiphase, and
//! how much of the signal is in that oscillation at all - across a sweep of
//! descending command.
//!
//! The first row is a scripted 12 Hz tripod put through the identical
//! measurement, so every number below it is read against a known good.
use fly::gait::{analyse, Trace};
use fly::{Coupling, Fly, Timing};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

const CONTROL_DT: f64 = 0.002;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn row(label: &str, t: &Trace, distance: Option<f64>) {
    match analyse(t) {
        Some(g) => println!(
            "{label:>16}  {:>7.2}  {:>8.3}  {:>8.3}  {:>8.4}  {:>7.3}  {}",
            g.step_hz,
            g.tripod,
            g.rhythmicity,
            g.power,
            g.score(),
            match distance {
                Some(d) => format!("{:+.3} BL/s", d / 0.25 / t.duration()),
                None => "-".to_string(),
            }
        ),
        None => println!("{label:>16}  trace too short to analyse"),
    }
}

fn main() {
    println!("{:>16}  {:>7}  {:>8}  {:>8}  {:>8}  {:>7}  {}", "condition", "step Hz", "tripod", "rhythm", "power", "score", "speed");

    // The known good: a scripted 12 Hz alternating tripod, measured by the
    // same function. Not a test - the tests cover that - but a line at the top
    // of the report so nothing below it is read without a scale.
    let mut scripted = Trace::new(CONTROL_DT);
    for k in 0..500 {
        let base = 2.0 * std::f64::consts::PI * 12.0 * k as f64 * CONTROL_DT;
        let mut legs = [0.0f32; 6];
        for (i, (_, _, tripod)) in flybody::LEGS.iter().enumerate() {
            legs[i] = (base + if *tripod == 0 { 0.0 } else { std::f64::consts::PI }).sin() as f32;
        }
        scripted.push(legs);
    }
    row("SCRIPTED 12 Hz", &scripted, None);

    let mj = MuJoCo::load().unwrap();
    let dir = std::path::PathBuf::from(env("BRAIN_CONNECTOME_DIR")).join("manc-codex");
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz")).unwrap();
    let model = Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).unwrap();
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, lif, 3e-2, None, Timing::default(), Coupling::default()).unwrap();

    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    for command in [0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0] {
        f.reset();
        f.set_descending(&vec![command; f.descending_count()]).unwrap();
        let x0 = f.qpos()[0];
        let mut t = Trace::new(CONTROL_DT);
        for _ in 0..ticks {
            f.step().unwrap();
            t.push(f.leg_swing());
        }
        row(&format!("drive {command:.1}"), &t, Some(f.qpos()[0] - x0));
    }

    println!("\nThe scripted row is what a gait scores. Read every other row against it.");
    println!("'tripod' near -1 means all six legs move together, which is a hop, not a walk;");
    println!("'rhythm' near 0 means there is no dominant frequency at all, whatever the peak.");
}
