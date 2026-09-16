// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Search the parameters the connectome does not contain, for a WALK or a
//! FLIGHT.
//!
//! The ceiling instrument, re-aimed. `examples/ceiling` hill-climbs the
//! per-cell-type gains one coordinate at a time under a reward a corpse can
//! collect; this runs a mirrored-sampling evolution strategy over the gains
//! AND the dynamics, under `Objective::Walk`, which is net travel multiplied
//! by the gait score and which a corpse scores zero on (gated in
//! `tests/walk_reward.rs`).
//!
//! Two things are searched together because the measurements say both matter
//! and neither is in any file:
//!
//! * **Per-cell-type gains.** What a synapse from a descending neuron is
//!   WORTH relative to one from an interneuron. A connectome carries synapse
//!   COUNTS, which are not strengths.
//! * **The dynamics.** The adaptation current and the two synaptic time
//!   constants are what decide whether a network of these cells can oscillate
//!   at all - a cord with neither can coordinate its legs and cannot sustain
//!   the coordination, which is exactly what this cord was measured doing.
//!
//! Every generation prints its best and its mean, and the FIRST row is the
//! connectome as imported. Without that row a search result is a number with
//! nothing to be better than.
use fly::learn::{episode, Condition, GainSearch, Lcg, Objective, RewardConfig};
use fly::search::{Es, Knob};
use fly::{Cns, Coupling, Fly, Timing, Wiring};
use neuro::LifParams;
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
    let which = match std::env::var("CNS").unwrap_or_default().as_str() {
        "brain" | "banc" => Cns::BrainAndCord,
        _ => Cns::Cord,
    };
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap();
    // The structural control. A degree-matched shuffle preserves every
    // neuron's in-degree and the weight multiset exactly and destroys the
    // wiring; a search that does as well on it was not using the connectome.
    let shuffle: Option<u64> = std::env::var("SHUFFLE").ok().map(|v| v.parse().unwrap_or(0x5EED));
    let wiring = Wiring { shuffle_seed: shuffle, ..Wiring::default() };
    if let Some(seed) = shuffle {
        println!("SHUFFLED CONNECTOME (seed {seed:#x}): this is the control, not the animal");
    }
    let gains = GainSearch::new(&c, wiring);
    println!("{which:?}: {} neurons, {} gain groups", c.neurons.len(), gains.groups().len());

    // The searchable knobs: one gain per presynaptic cell class, then the
    // handful of dynamics constants a rhythm lives on, then the descending
    // command itself - which is a parameter of the EXPERIMENT rather than of
    // the animal, and searching it is what stops the result depending on a
    // drive somebody picked by hand.
    let mut knobs: Vec<Knob> = gains.groups().iter().map(|g| Knob::new(format!("gain:{g}"), 0.0, 4.0)).collect();
    let dyn_start = knobs.len();
    let lif = fly::cord_lif();
    knobs.extend([
        Knob::new("adapt_increment", 0.0, 1.0),
        Knob::new("adapt_decay", 0.5, 0.995),
        Knob::new("dt_over_tau_syn", 0.05, 1.0),
        Knob::new("dt_over_tau_inh", 0.02, 1.0),
        Knob::new("activation_gain", 0.005, 0.3),
        Knob::new("command", 0.0, 4.0),
    ]);
    let start: Vec<f32> = gains
        .unit_gains()
        .into_iter()
        .chain([
            lif.adapt_increment,
            lif.adapt_decay,
            lif.dt_over_tau_syn,
            lif.dt_over_tau_inh,
            Coupling::default().activation_gain,
            1.0,
        ])
        .collect();

    // `ARENA=air` searches for flight instead: the generated flight model,
    // the wings handed to the cord's own motor neurons rather than to a
    // throttle, and `Objective::Fly`. The wing gains join the knobs, because
    // what a wing motor neuron's spike is WORTH is exactly the kind of thing
    // no connectome contains.
    let air = std::env::var("ARENA").unwrap_or_default() == "air";
    if air {
        knobs.push(Knob::new("wing_power_gain", 0.001, 0.3));
        knobs.push(Knob::new("wing_steer_gain", 0.001, 0.2));
    }
    let start: Vec<f32> = if air {
        start
            .into_iter()
            .chain([Coupling::default().wing_power_gain, Coupling::default().wing_steer_gain])
            .collect()
    } else {
        start
    };

    let ticks: u32 = num("TICKS", 1000);
    let pairs: usize = num("PAIRS", 6);
    let generations: u32 = num("GENERATIONS", 20);
    // Held for the whole run: the generated flight model lives in it.
    let scratch = tempfile::tempdir().unwrap();
    let flight = flybody::Flight::default();
    let (path, timing) = if air {
        let base = env("BRAIN_FLYBODY_FRUITFLY_XML");
        let scene = flybody::flight_scene(std::path::Path::new(&base), scratch.path(), flight).unwrap();
        let steps = fly::Timing::substeps(flight.timestep).unwrap();
        (scene, Timing { neural_per_control: 1, physics_per_control: steps, physics_dt: flight.timestep })
    } else {
        (std::path::PathBuf::from(env("BRAIN_FLYBODY_XML")), Timing::default())
    };
    let model = Model::from_xml(&mj, &path).unwrap();
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, lif, wiring, timing, Coupling::default()).unwrap();
    if air {
        // 180 Hz rather than the animal's 218: this airframe's hinge resonates
        // lower than a real thorax, measured by sweep rather than assumed. The
        // wings are NOT held - the cord's own wing motor neurons drive them,
        // which is the thing being searched.
        f.enable_flight(fly::Wingbeat { hz: 180.0, ..fly::Wingbeat::default() }).unwrap();
        println!("{}", f.wing_summary());
    }

    // Plasticity OFF throughout. This measures what the PARAMETERS can do,
    // not what a rule can find, and mixing the two makes the answer
    // unreadable in both directions.
    let evaluate = |f: &mut Fly, v: &[f32], seed: u64| -> (f64, f64, f64) {
        f.set_weights(&gains.weights(&v[..dyn_start])).unwrap();
        let d = &v[dyn_start..];
        f.set_lif(LifParams {
            adapt_increment: d[0],
            adapt_decay: d[1],
            dt_over_tau_syn: d[2],
            dt_over_tau_inh: d[3],
            ..lif
        })
        .unwrap();
        let mut coupling = Coupling { activation_gain: d[4], ..Coupling::default() };
        if air {
            coupling.wing_power_gain = d[6];
            coupling.wing_steer_gain = d[7];
        }
        f.set_coupling(coupling);
        let objective = if air { Objective::flight() } else { Objective::walk() };
        let cfg = RewardConfig { objective, ticks, command: d[5], ..RewardConfig::default() };
        match episode(f, cfg, Condition::Frozen, &mut Lcg::new(seed)) {
            Ok(e) => {
                // The quality half of the score: a gait on the ground, the
                // fraction of the episode spent airborne in the air.
                let quality = match e.gait {
                    Some(g) => g.score(),
                    None => e.airborne as f64 / e.ticks.max(1) as f64,
                };
                (e.score(), e.net, quality)
            }
            // A parameter set the runtime refuses is not a crash, it is a
            // candidate worth nothing - but it must not be worth MORE than a
            // real one, so it scores below anything an episode can produce.
            Err(_) => (f64::NEG_INFINITY, 0.0, 0.0),
        }
    };

    // The control: the connectome exactly as imported, at unit gains and the
    // published dynamics. Every row below is read against this one.
    let (base, base_net, base_gait) = evaluate(&mut f, &start, 1);
    println!("\nconnectome as imported: score {base:.5} (net {base_net:.4} cm, gait {base_gait:.3})");
    println!("\n{:>4}  {:>12}  {:>12}  {:>10}  {:>8}", "gen", "best", "mean", "net cm", if air { "airborne" } else { "gait" });

    let mut es = Es::new(knobs, &start, num("SIGMA", 0.6), num("RATE", 0.5), num("SEED", 0xF1E5u64)).unwrap();
    let (mut best_ever, mut best_params) = (base, start.clone());
    for _ in 0..generations {
        let g = es.step(pairs, |v, seed| evaluate(&mut f, v, seed).0);
        if g.best > best_ever {
            best_ever = g.best;
            best_params = g.best_params.clone();
        }
        let (_, net, gait) = evaluate(&mut f, &g.centre, 1);
        println!("{:>4}  {:>12.5}  {:>12.5}  {:>10.4}  {:>8.3}", g.index, g.best, g.mean, net, gait);
    }

    let (s, net, gait) = evaluate(&mut f, &best_params, 1);
    println!("\nbest found: score {s:.5} (net {net:.4} cm, gait {gait:.3}) against {base:.5} as imported");
    println!("{:.2} body lengths per second", net / 0.25 / (ticks as f64 * fly::CONTROL_PERIOD));
    for (k, v) in es.knobs().iter().zip(&best_params) {
        println!("  {:<28} {v:.4}", k.name);
    }

    // Written out, because a result nobody can replay is a number in a log.
    // `OUT=path` and the sample's `--tuning` are the two ends of the same
    // wire: a person can watch what this found.
    if let Ok(path) = std::env::var("OUT") {
        let t = fly::tuning::from_knobs(es.knobs(), &best_params);
        match t.save(&path) {
            Ok(()) => println!("\nwrote {path}; watch it with `sample-fly-interactive --tuning {path}`"),
            Err(e) => eprintln!("could not write {path}: {e}"),
        }
    } else {
        println!("\nset OUT=path to write these where the sample's --tuning can read them");
    }
}
