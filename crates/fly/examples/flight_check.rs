// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Can the airframe fly at all, and where does it resonate?
//!
//! Characterises the BODY, with the wing command imposed rather than produced
//! by the cord. Measuring "does the connectome fly the fly" before knowing
//! whether the fly flies makes a negative result uninterpretable, in exactly
//! the way the walking work already learned the hard way.
//!
//! A wingbeat is a resonance: a sprung, damped, inertial hinge driven
//! periodically. Stroke amplitude therefore peaks at one frequency and falls
//! away either side of it, and lift follows amplitude. So sweep the drive
//! frequency and report stroke amplitude and vertical motion together - a
//! sweep that finds no peak is a sweep of a model that is not resonating, and
//! that is a different problem from one that resonates in the wrong place.
use fly::wing::{WingCommand, Wingbeat};
use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn main() {
    let mj = MuJoCo::load().unwrap();
    let base = std::path::PathBuf::from(env("BRAIN_FLYBODY_FRUITFLY_XML"));
    let dir = tempfile::tempdir().expect("a temp dir");
    let cfg = flybody::Flight::default();
    let scene = flybody::flight_scene(&base, dir.path(), cfg).expect("the flight scene builds");
    println!("flight model: timestep {:.0e} s, wing gain {}, fluid {:?}", cfg.timestep, cfg.wing_gain, cfg.fluidcoef);

    let cdir = std::path::PathBuf::from(env("BRAIN_CONNECTOME_DIR")).join("manc-codex");
    let c = connectome::load("manc", &cdir.join("neurons.csv.gz"), &cdir.join("connections_princeton.csv.gz")).unwrap();
    let model = Model::from_xml(&mj, &scene).expect("the flight model compiles");
    let lif = LifParams { dt_over_tau: 0.1, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);

    // The physics timestep must match the model, or the stroke advances at the
    // wrong rate while every other reading stays correct. 40 steps per control
    // tick keeps the 2 ms control period the rest of this crate uses.
    let timing = Timing { neural_per_control: 1, physics_per_control: 40, physics_dt: cfg.timestep };
    let mut f = Fly::new(gpu, &c, model, lif, Wiring::default(), timing, Coupling::default()).unwrap();
    println!("{}", f.wing_summary());
    f.enable_flight(Wingbeat::default()).expect("the body has wings");

    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let seconds = ticks as f64 * timing.physics_per_control as f64 * cfg.timestep;
    // Free fall over the MEASUREMENT window, which is the second half of the
    // run, starting from whatever vertical speed the spin-up left behind.
    let window = seconds / 2.0;
    let fall = 0.5 * 981.0 * window * window;
    println!("\n{ticks} ticks = {seconds:.3} s of flight per row, wings held at full power");
    println!("measured over the last {window:.3} s; free fall over that is {fall:.2} cm from rest");
    println!(
        "{:>7}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}",
        "feather", "hz", "stroke rad", "aoa rad", "dz (cm)", "vz (cm/s)", "supported"
    );

    // Two things set how much lift a flapping wing makes: how far it sweeps
    // and what angle of attack it holds while sweeping. The stroke is already
    // near the joint's limit at full drive, so the angle of attack is the lever
    // with room in it, and it is swept here against frequency because the two
    // interact - a wing that feathers hard at a frequency it cannot reach is
    // just a wing held edge-on.
    for feather in [0.2f32, 0.4, 0.6, 0.9, 1.2] {
        for hz in [150.0f64, 180.0, 218.0] {
            f.reset();
            f.enable_flight(Wingbeat { hz, feather, ..Wingbeat::default() }).unwrap();
            f.hold_wing_command(Some(WingCommand { power: 1.0, ..WingCommand::default() }));

            // Measured over the SECOND half only. The thorax spins up over the
            // first few beats, and a fly that is in free fall for 50 ms and
            // hovering for 450 ms shows up as one that never quite hovers if
            // the whole window is averaged. The first half is the spin-up and
            // is not what is being asked about.
            let (mut stroke, mut aoa): (f64, f64) = (0.0, 0.0);
            let mut z0 = f.qpos()[2];
            for k in 0..ticks {
                f.step().unwrap();
                if k == ticks / 2 {
                    z0 = f.qpos()[2];
                }
                let a = f.wing_angles();
                stroke = stroke.max(a[0][0].abs()).max(a[1][0].abs());
                aoa = aoa.max(a[0][2].abs()).max(a[1][2].abs());
            }
            let dz = f.qpos()[2] - z0;
            let vz = f.qvel()[2];
            // How much of its own weight the wings carried: zero is free fall,
            // one is hovering, more than one is climbing.
            let supported = 1.0 - (-dz / fall);
            println!(
                "{feather:>7.1}  {hz:>8.0}  {stroke:>10.4}  {aoa:>10.4}  {dz:>10.4}  {vz:>10.2}  {:>8.1}%",
                100.0 * supported
            );
        }
    }

}
