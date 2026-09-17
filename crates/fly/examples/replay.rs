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
        // `TIMESTEP` reproduces a viewer's integrator. The demo runs the body
        // at a coarser step than a search does, and the floor is softened to
        // match it, so a tuning validated here at the default and then watched
        // there is being watched in a different experiment.
        let timing = match std::env::var("TIMESTEP").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(dt) => Timing {
                neural_per_control: 1,
                physics_per_control: fly::Timing::substeps(dt).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2)
                }),
                physics_dt: dt,
            },
            None => Timing::default(),
        };
        (std::path::PathBuf::from(env("BRAIN_FLYBODY_XML")), timing)
    };
    let model = Model::from_xml(&mj, &model_path).unwrap();
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, fly::cord_lif(), wiring, timing, Coupling::default()).unwrap();
    if air {
        f.enable_flight(fly::Wingbeat { hz: 180.0, ..fly::Wingbeat::default() }).unwrap();
    }

    // `FOOD=x,y,z` replays a chemotaxis tuning: the animal is scored on how
    // much nearer it ended, and never told where the food is.
    let f_food: Option<[f64; 3]> = std::env::var("FOOD").ok().and_then(|v| {
        let p: Vec<f64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (p.len() == 3).then(|| [p[0], p[1], p[2]])
    });
    f.set_food(f_food);
    let objective = match (f_food.is_some(), air) {
        (true, _) => Objective::seek(),
        (_, true) => Objective::flight(),
        _ => Objective::walk(),
    };
    let run = |f: &mut Fly, command: f32| {
        let cfg = RewardConfig { objective, ticks, command, ..RewardConfig::default() };
        episode(f, cfg, Condition::Frozen, &mut Lcg::new(1)).expect("an episode runs")
    };
    let seconds = ticks as f64 * fly::CONTROL_PERIOD;
    let row = |label: &str, e: &fly::learn::Episode| {
        let quality = match e.gait {
            Some(g) => g.score(),
            None if e.airborne > 0 => e.airborne as f64 / e.requested.max(e.ticks).max(1) as f64,
            None => e.range.1,
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
        if f_food.is_some() { "range" } else if air { "airborne" } else { "gait" },
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
    drop(f);

    // THE STRUCTURAL CONTROL, and it is the one that says whether the
    // CONNECTOME did any of this. The same tuning, the same body, the same
    // command, on a degree-matched shuffle of the wiring: every neuron keeps
    // its in-degree and the weight multiset is preserved exactly, applied
    // after signing so the excitatory/inhibitory split is identical too. Only
    // which cell contacts which is destroyed.
    //
    // This is a stronger control than searching the shuffle separately and
    // comparing the two winners, which is what the earlier instrument here
    // did: two best-of-N maxima are two draws from the tail of a noisy
    // distribution, and their difference is mostly a statement about the
    // noise. Holding the parameters fixed and changing only the graph asks
    // the question directly.
    let shuffled_wiring = Wiring { shuffle_seed: Some(0x5EED), ..wiring };
    let model = Model::from_xml(&mj, &model_path).unwrap();
    let mut sf = Fly::new(
        gpu_core::testgpu::dev(&neuro::KERNELS),
        &c,
        model,
        fly::cord_lif(),
        shuffled_wiring,
        timing,
        Coupling::default(),
    )
    .unwrap();
    if air {
        sf.enable_flight(fly::Wingbeat { hz: 180.0, ..fly::Wingbeat::default() }).unwrap();
    }
    sf.set_food(f_food);
    tuning.apply(&mut sf, &c, shuffled_wiring).expect("the tuning applies to the shuffle too");
    let shuffled = run(&mut sf, command);
    row("shuffled", &shuffled);

    // THE DURATION CONTROL, and it is the one that catches an animal which
    // does not fall over and does not walk either.
    //
    // Every row above ran for exactly as long as the search's episodes did,
    // which means none of them can see the failure that matters most: a
    // tuning that is a TRANSIENT. An objective built on net displacement over
    // a two-second episode is fully satisfied by a lunge, and a lunge scores
    // well, passes the paralysed control, passes the shuffle, and finishes
    // upright. It then stops. The creature that produced the numbers above
    // travels 0.46 body lengths a second for two seconds and 0.02 for the next
    // ten, which is worse than the connectome as imported and was invisible
    // until somebody watched the screen.
    //
    // So: the same tuning, held for three times as long, against the same
    // baseline held for three times as long. A behaviour that is real does not
    // care how long it is asked to keep going.
    let long = ticks * 3;
    println!("\nheld for {long} ticks ({:.0} s), three times the episode it was tuned on:", long as f64 * fly::CONTROL_PERIOD);
    let run_long = |f: &mut Fly, command: f32| {
        let cfg = RewardConfig { objective, ticks: long, command, ..RewardConfig::default() };
        episode(f, cfg, Condition::Frozen, &mut Lcg::new(1)).expect("an episode runs")
    };
    let mut f2 = {
        let model = Model::from_xml(&mj, &model_path).unwrap();
        let mut f = Fly::new(gpu_core::testgpu::dev(&neuro::KERNELS), &c, model, fly::cord_lif(), wiring, timing, Coupling::default()).unwrap();
        if air {
            f.enable_flight(fly::Wingbeat { hz: 180.0, ..fly::Wingbeat::default() }).unwrap();
        }
        f.set_food(f_food);
        f
    };
    let seconds_long = long as f64 * fly::CONTROL_PERIOD;
    let row_long = |label: &str, e: &fly::learn::Episode| {
        println!(
            "{label:>12}  {:>9.5}  {:>9.4}  {:>8}  {:>8.2}  {:>7.2}  {:>9}  {}",
            e.score(),
            e.net,
            "",
            e.net / 0.25 / seconds_long,
            e.tipped,
            e.spikes,
            if e.terminated { "ended early" } else { "ran to the end" }
        );
    };
    let imported_long = run_long(&mut f2, 1.0);
    row_long("as imported", &imported_long);
    tuning.apply(&mut f2, &c, wiring).expect("the tuning applies");
    let tuned_long = run_long(&mut f2, command);
    row_long("tuned", &tuned_long);
    let (a, b) = (tuned_long.net / seconds_long, imported_long.net / seconds_long);
    if a <= b {
        println!(
            "\n  FAILED: over {seconds_long:.0} s the tuning travels {a:.4} cm/s against the imported"
        );
        println!("  connectome's {b:.4}. It is a transient: the objective's episode was short");
        println!("  enough that a lunge satisfies it, and nothing above could see that.");
    }

    println!("\nthe tuned row has to beat every other, and `tipped` has to be small:");
    println!("a walk that finishes more than a radian from upright is a slide, and a");
    println!("tuning that does as well on the shuffle was not using the connectome.");
}
