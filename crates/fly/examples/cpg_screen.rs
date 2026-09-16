// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Find the regime where the cord oscillates, if there is one.
//!
//! A network oscillator needs recurrent excitation strong enough to sustain
//! activity and inhibition strong enough to cut it off again. Too weak and a
//! command dies before it reaches the motor neurons; too strong and the whole
//! cord saturates into a single fused burst. Rhythm lives in a band between
//! those, and where that band sits depends on one number this crate has always
//! had to guess: the conversion from synapse COUNT to membrane current.
//!
//! So sweep it, with the descending drive delivered the way a stimulation
//! experiment delivers it - to one named cell type - and report the gait
//! criterion alongside the activity level, because "no rhythm" for want of any
//! activity and "no rhythm" for want of structure are different failures with
//! different fixes.
use fly::gait::{analyse, Trace};
use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

const CONTROL_DT: f64 = 0.002;

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
    let lif = LifParams { dt_over_tau: 0.1, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let wiring = Wiring::default();
    let mut f = Fly::new(gpu, &c, model, lif, wiring, Timing::default(), Coupling::default()).unwrap();

    let n = c.neurons.len() as f64;
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let cell = std::env::var("CELL").unwrap_or_else(|_| "DNg100".to_string());
    let limit = wiring.size_limit.unwrap_or(1.0);

    println!("{} neurons, {} ticks per run, drive to {cell}", c.neurons.len(), ticks);
    println!(
        "{:>10}  {:>9}  {:>7}  {:>8}  {:>8}  {:>7}  {:>9}  {:>7}",
        "scale", "active %", "motor/t", "step Hz", "tripod", "rhythm", "power", "score"
    );

    for scale in [3e-2f32, 1e-1, 3e-1, 1.0, 3.0, 10.0] {
        // Re-weighting rather than rebuilding: the graph's shape never
        // changes, so uploading new weights is the whole difference between
        // one run and the next and a rebuild would cost a connectome load each
        // time.
        f.set_weights(&c.signed_csc_sized(scale, limit).w).unwrap();
        f.reset();
        if let Err(e) = f.drive_cell_type(&cell, 250.0) {
            eprintln!("{e}");
            return;
        }
        let mut t = Trace::new(CONTROL_DT);
        let (mut spikes, mut motor) = (0u64, 0u64);
        for _ in 0..ticks {
            let beat = f.step().unwrap();
            spikes += beat.total_spikes as u64;
            motor += beat.motor_spikes as u64;
            t.push(f.leg_swing());
        }
        let active = 100.0 * spikes as f64 / (ticks as f64 * n);
        match analyse(&t) {
            Some(g) => println!(
                "{scale:>10.3}  {active:>9.3}  {:>7.1}  {:>8.2}  {:>8.3}  {:>7.3}  {:>9.4}  {:>7.3}",
                motor as f64 / ticks as f64,
                g.step_hz,
                g.tripod,
                g.rhythmicity,
                g.power,
                g.score()
            ),
            None => println!("{scale:>10.3}  {active:>9.3}  trace too short"),
        }
    }

    println!("\nA nerve cord should sit at a few percent active. Zero means the command died");
    println!("before it arrived; tens of percent means the cord fused into one burst, and");
    println!("neither can carry a rhythm.");
}
