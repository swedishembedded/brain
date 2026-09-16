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
    let whole = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz")).unwrap();
    // $REGION restricts the network to one neuropil plus the descending
    // neurons that reach it, which is the scope published circuit models of
    // this cord work at. A large recurrent graph carries feedback loops of
    // many lengths at once and a network oscillator's period is its loop
    // delay, so running the whole cord smears the rhythm across every period
    // the graph contains.
    let region = std::env::var("REGION").unwrap_or_default();
    let c = if region.is_empty() {
        whole
    } else {
        let sub = whole.subgraph(|n| n.region.starts_with(&region) || n.super_class == "descending");
        println!(
            "restricted to {region}: {} neurons, {} edges, {} synapses",
            sub.neurons.len(),
            sub.coverage.edges,
            sub.coverage.synapses
        );
        sub
    };
    let model = Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).unwrap();
    let lif = LifParams { dt_over_tau: 0.1, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let n = c.neurons.len() as f64;
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let cell = std::env::var("CELL").unwrap_or_else(|_| "DNg100".to_string());
    let scales: Vec<f32> = std::env::var("SCALES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.3, 0.6, 1.0, 2.0]);

    println!("{} neurons, {ticks} ticks per run, drive to {cell}", c.neurons.len());
    println!(
        "{:>6}  {:>5}  {:>8}  {:>9}  {:>7}  {:>8}  {:>8}  {:>7}  {:>9}  {:>7}",
        "minsyn", "tau", "scale", "active %", "motor/t", "step Hz", "tripod", "rhythm", "power", "score"
    );

    // `min_synapses` changes the graph's SHAPE and `dt_over_tau` is fixed when
    // the network is built, so each pair needs its own network; only the
    // weight scale can be swept by re-uploading weights.
    let mut model = Some(model);
    for min_syn in [1u32, 5] {
        for dt_over_tau in [0.1f32, 0.2, 0.4] {
            let lif = LifParams { dt_over_tau, ..lif };
            let wiring = Wiring { min_synapses: min_syn, ..Wiring::default() };
            let m = match model.take() {
                Some(m) => m,
                None => Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).unwrap(),
            };
            let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
            let mut f = Fly::new(gpu, &c, m, lif, wiring, Timing::default(), Coupling::default()).unwrap();

            for &scale in &scales {
                f.set_weights(&c.network(scale, wiring.size_limit, min_syn).w).unwrap();
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
                let tau = 2.0 / dt_over_tau;
                match analyse(&t) {
                    Some(g) => println!(
                        "{min_syn:>6}  {tau:>5.0}  {scale:>8.3}  {active:>9.3}  {:>7.1}  {:>8.2}  {:>8.3}  {:>7.3}  {:>9.4}  {:>7.3}",
                        motor as f64 / ticks as f64,
                        g.step_hz,
                        g.tripod,
                        g.rhythmicity,
                        g.power,
                        g.score()
                    ),
                    None => println!("{min_syn:>6}  {tau:>5.0}  {scale:>8.3}  {active:>9.3}  trace too short"),
                }
            }
        }
    }

    println!("\nA nerve cord should sit at a few percent active. Zero means the command died");
    println!("before it arrived; tens of percent means the cord fused into one burst, and");
    println!("neither can carry a rhythm.");
}
