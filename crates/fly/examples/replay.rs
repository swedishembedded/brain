// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Re-measure a tuning, against the controls that say whether to believe it.
//!
//! A search reports its own best score, which is the one number least worth
//! trusting: it is the maximum of a noisy sample, so it is biased upward by
//! construction, and it is reported by the code that was trying to make it
//! large. This runs the tuning again, cold, and prints what it does next to
//! two things it has to beat.
//!
//!   as imported   the connectome at unit gains and the published dynamics
//!   paralysed     the same tuning with every muscle severed
//!
//! Plus the diagnostics that expose the ways a locomotion score is usually
//! cheated. `tipped` is the important one: both objectives here are built on
//! net travel, and the classic exploit is to roll the animal onto its side and
//! let it slide, which keeps the root height, keeps the legs oscillating and
//! covers ground. An animal that finishes a walk more than a radian from
//! upright did not walk.
use fly::learn::{episode, Condition, Lcg, Objective, RewardConfig};
use fly::{Cns, Coupling, Fly, Timing, Tuning, Wiring};
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
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: replay TUNING.txt   (ARENA=air for flight, CNS=brain for the joined network)");
        std::process::exit(2)
    });
    let tuning = Tuning::load(&path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });

    let mj = MuJoCo::load().unwrap();
    let air = std::env::var("ARENA").unwrap_or_default() == "air";
    let which = match std::env::var("CNS").unwrap_or_default().as_str() {
        "brain" | "banc" => Cns::BrainAndCord,
        _ => Cns::Cord,
    };
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap();
    let wiring = Wiring { shuffle_seed: None, ..Wiring::default() };
    let ticks: u32 = num("TICKS", 1000);

    let scratch = tempfile::tempdir().unwrap();
    let flight = flybody::Flight::default();
    let (model_path, timing) = if air {
        let base = env("BRAIN_FLYBODY_FRUITFLY_XML");
        let scene = flybody::flight_scene(std::path::Path::new(&base), scratch.path(), flight).unwrap();
        (scene, Timing { neural_per_control: 1, physics_per_control: fly::Timing::substeps(flight.timestep).unwrap(), physics_dt: flight.timestep })
    } else {
        (std::path::PathBuf::from(env("BRAIN_FLYBODY_XML")), Timing::default())
    };
    let model = Model::from_xml(&mj, &model_path).unwrap();
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, fly::cord_lif(), wiring, timing, Coupling::default()).unwrap();
    if air {
        f.enable_flight(fly::Wingbeat { hz: 180.0, ..fly::Wingbeat::default() }).unwrap();
    }

    let objective = if air { Objective::flight() } else { Objective::walk() };
    let run = |f: &mut Fly, command: f32| {
        let cfg = RewardConfig { objective, ticks, command, ..RewardConfig::default() };
        episode(f, cfg, Condition::Frozen, &mut Lcg::new(1)).expect("an episode runs")
    };
    let seconds = ticks as f64 * fly::CONTROL_PERIOD;
    let row = |label: &str, e: &fly::learn::Episode| {
        let quality = match e.gait {
            Some(g) => g.score(),
            None => e.airborne as f64 / e.requested.max(e.ticks).max(1) as f64,
        };
        println!(
            "{label:>12}  {:>9.5}  {:>9.4}  {:>8.3}  {:>8.2}  {:>7.2}  {:>9}  {}",
            e.score(),
            e.net,
            quality,
            e.net / 0.25 / seconds,
            e.tipped,
            e.spikes,
            if e.terminated { "ended early" } else { "ran to the end" }
        );
    };

    println!("{path}: {} neurons, {which:?}, {} arena", c.neurons.len(), if air { "air" } else { "ground" });
    println!(
        "\n{:>12}  {:>9}  {:>9}  {:>8}  {:>8}  {:>7}  {:>9}",
        "condition",
        "score",
        "net cm",
        if air { "airborne" } else { "gait" },
        "BL/s",
        "tipped",
        "spikes"
    );

    // The control first, as everywhere else here: the connectome as imported.
    let imported = run(&mut f, 1.0);
    row("as imported", &imported);

    let report = tuning.apply(&mut f, &c, wiring).expect("the tuning applies");
    if report.gains == 0 {
        eprintln!("none of this tuning's gains name a cell class this connectome has");
        std::process::exit(1);
    }
    if !report.unknown.is_empty() {
        eprintln!("ignored {} unrecognised parameter(s): {}", report.unknown.len(), report.unknown.join(", "));
    }
    let command = tuning.command().unwrap_or(1.0);
    let tuned = run(&mut f, command);
    row("tuned", &tuned);

    // And the corpse, WITH the tuning applied: whatever this animal is doing,
    // it has to be doing it with its muscles.
    for a in 0..f.actuator_count() {
        f.set_muscle_strength(a, 0.0).unwrap();
    }
    let corpse = run(&mut f, command);
    row("paralysed", &corpse);

    println!("\nthe tuned row has to beat both others, and `tipped` has to be small:");
    println!("a walk that finishes more than a radian from upright is a slide.");
}
