// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Where does walking break?
//!
//! The cord, driven tonically and measured on its own, produces a 16.7 Hz
//! rhythm at the leg motor pool that a degree-matched shuffle does not. The
//! same cord in the body produces a gait score of 0.015 and goes nowhere. One
//! of the links between those two facts is broken and this finds which:
//!
//!   1  motor neurons      do the leg motor neurons fire rhythmically?
//!   2  muscle activation  does that rhythm survive the muscle's low-pass?
//!   3  joint angle        does the activation move the joint?
//!   4  the body           does the joint motion move the animal?
//!
//! Each is measured on the SAME run, so a break shows up as the first row
//! where the rhythm disappears rather than as four separate experiments that
//! have to be reconciled.
use fly::rhythm::{analyse, Rhythm};
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

const BAND: (f64, f64) = (3.0, 30.0);

fn main() {
    let mj = MuJoCo::load().unwrap();
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), Cns::Cord).unwrap();
    let path = std::path::PathBuf::from(env("BRAIN_FLYBODY_XML"));
    let ticks: usize = num("TICKS", 2500);
    let settle: usize = num("SETTLE", 250);
    let adhesion: f32 = num("ADHESION", 0.4);

    let commands: Vec<f32> = std::env::var("COMMANDS")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![2.0, 4.0, 6.0, 10.0]);
    let dn = std::env::var("DN").unwrap_or_else(|_| "DNg100".into());
    let proprio: bool = num("PROPRIO", 1u8) != 0;

    println!(
        "{} ticks ({:.0} s) after {settle} settling, command to {dn}, adhesion {adhesion}, proprioception {}",
        ticks,
        ticks as f64 * fly::CONTROL_PERIOD,
        if proprio { "on" } else { "OFF" }
    );
    println!(
        "\n{:>5}  {:>22}  {:>22}  {:>22}  {:>22}  {:>9}",
        "cmd", "motor spikes", "coxa agonist", "coxa antagonist", "net activation", "net cm"
    );
    let show = |r: &Rhythm| -> String {
        if r.rate == 0.0 && r.deviation == 0.0 {
            "          silent      ".to_string()
        } else {
            format!("{:>7.2} Hz {:>5.3} {:>6}", r.hz, r.strength, if r.is_rhythmic(0.25) { "yes" } else { "no" })
        }
    };

    for &cmd in &commands {
        let model = Model::from_xml(&mj, &path).unwrap();
        let coupling = Coupling { adhesion_gain: adhesion, ..Coupling::default() };
        let mut f = Fly::new(
            gpu_core::testgpu::dev(&neuro::KERNELS),
            &c,
            model,
            fly::cord_lif(),
            Wiring::default(),
            Timing::default(),
            coupling,
        )
        .unwrap();
        f.set_proprioception(proprio);
        f.reset();
        if f.drive_cell_type(&dn, cmd).unwrap_or(0) == 0 {
            eprintln!("no descending neuron of type {dn}");
            std::process::exit(2);
        }
        let start = f.qpos();
        let (mut motor, mut ago, mut anti, mut swing) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in 0..settle + ticks {
            let tick = f.step().expect("a tick runs");
            if t < settle {
                continue;
            }
            motor.push(tick.motor_spikes as f64);
            // One leg's fore-aft swing, split by polarity. The split is the
            // point: a joint whose two muscles both fire rhythmically IN PHASE
            // produces no movement, and the net activation that drives the
            // body reports that as no rhythm at all.
            let o = f.leg_opposed()[0];
            ago.push(o[0] as f64);
            anti.push(o[1] as f64);
            swing.push(f.leg_swing()[0] as f64);
        }
        let end = f.qpos();
        let net = {
            let dx = end.first().copied().unwrap_or(0.0) - start.first().copied().unwrap_or(0.0);
            let dy = end.get(1).copied().unwrap_or(0.0) - start.get(1).copied().unwrap_or(0.0);
            (dx * dx + dy * dy).sqrt()
        };
        println!(
            "{cmd:>5.1}  {}  {}  {}  {}  {net:>9.4}",
            show(&analyse(&motor, fly::CONTROL_PERIOD, BAND)),
            show(&analyse(&ago, fly::CONTROL_PERIOD, BAND)),
            show(&analyse(&anti, fly::CONTROL_PERIOD, BAND)),
            show(&analyse(&swing, fly::CONTROL_PERIOD, BAND)),
        );
    }
    println!("\nthe first column with no rhythm is where walking breaks.");
}
