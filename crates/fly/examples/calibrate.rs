// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Fit the physiology a connectome does not contain, against a PHYSIOLOGICAL
//! target.
//!
//! Every other search in this crate optimises a behaviour: walk further, stay
//! airborne longer. That conflates two problems the literature keeps apart,
//! and conflating them is why those results did not survive their shuffle. A
//! connectome is missing its time constants, its excitabilities and its
//! synaptic conductances, and inferring those is a calibration problem with a
//! right answer that has nothing to do with any task. Asking a behavioural
//! score to supply them lets the optimiser buy behaviour by leaving the
//! measured wiring behind, because nothing in the objective says it may not.
//!
//! So the target here is not a behaviour. The isolated DNg100 circuit already
//! oscillates at 13.5 Hz from the anatomy alone, with its shuffle and its
//! deleted-inhibition controls both failing (see `tests/cpg.rs`). Embedded in
//! the whole cord at the uniform physiology, that rhythm is swamped. The
//! target is simply that it should survive embedding:
//!
//!   in band       the leg motor pool oscillates between 5 and 20 Hz
//!   alive         at a firing rate a nerve cord could actually have
//!   silent        and does nothing at all without a command
//!
//! None of those is a task. A fly that satisfies all three has not been taught
//! to walk; it has been given a physiology in which what the anatomy already
//! does is visible. Whether it then walks is a separate question, asked later,
//! with these parameters frozen.
//!
//! The control is reported every generation rather than at the end: the same
//! parameters on a degree-matched shuffle. A calibration that improves the
//! shuffle as much as the real cord is fitting the objective, not the animal.

use fly::physiology::Physiology;
use fly::rhythm::analyse;
use fly::search::Es;
use fly::{Cns, Wiring};
use neuro::{DynamicalSystem, Port, SpikingNet};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

const BAND: (f64, f64) = (5.0, 20.0);
const DT: f64 = 0.002;
/// Spikes per motor neuron per tick that a cord could plausibly sustain: about
/// 20 Hz at a 2 ms tick. Not fitted - it is what the target is measured
/// against, and it comes from physiology rather than from this run.
const TARGET_RATE: f64 = 0.04;

fn main() {
    let which = match std::env::var("CNS").unwrap_or_else(|_| "cord".into()).as_str() {
        "brain" | "banc" => Cns::BrainAndCord,
        _ => Cns::Cord,
    };
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let wiring = Wiring::default();
    let ph = Physiology::new(&c, wiring);
    let command_type = std::env::var("DN").unwrap_or_else(|_| "DNg100".into());
    let command = c.population(|n| n.cell_type == command_type);
    let legs = c.population(|n| n.super_class == "motor" && matches!(n.class.as_str(), "fl" | "ml" | "hl"));
    if command.is_empty() || legs.is_empty() {
        eprintln!("{}: {} command cells, {} leg motor neurons", c.dataset, command.len(), legs.len());
        std::process::exit(1);
    }
    println!(
        "{}: {} neurons, {} classes, {} connection types, {} free parameters over {} synapses",
        c.dataset,
        c.neurons.len(),
        ph.classes().len(),
        ph.types().len(),
        ph.len(),
        ph.edges()
    );

    let ticks: usize = num("TICKS", 1200);
    let settle: usize = num("SETTLE", 200);
    let current: f32 = num("CURRENT", 4.0);

    let mut net = {
        let g = c.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
        SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &g, fly::cord_lif()).expect("the cord runs")
    };
    let mut shuffled = {
        let g = c.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses).shuffled_sources(0x5EED);
        SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &g, fly::cord_lif()).expect("the shuffle runs")
    };
    let n = net.port_len(Port::Spike);

    // One evaluation: silence without a command, rhythm with one.
    let mut spike = vec![0.0f32; n];
    let mut evaluate = |net: &mut SpikingNet, values: &[f32]| -> (f64, f64, f64, f64) {
        let p = ph.from_knobs(values);
        if net.set_weights(&ph.weights(&p)).is_err() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let (tau, exc) = ph.cell_scales(&p);
        if net.set_cell_scales(&tau, &exc).is_err() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let bias = ph.bias_drive(&p);

        let mut go = |extra: Option<f32>| -> Vec<f64> {
            let mut drive = bias.clone();
            if let Some(cur) = extra {
                for &d in &command {
                    drive[d as usize] += cur;
                }
            }
            net.reset(0);
            net.drive(Port::Drive, &drive).expect("the drive fits");
            let mut series = Vec::with_capacity(ticks);
            for t in 0..settle + ticks {
                net.step();
                net.read(Port::Spike, &mut spike).expect("the readback fits");
                if t >= settle {
                    series.push(legs.iter().map(|&i| spike[i as usize] as f64).sum());
                }
            }
            series
        };

        // Silence first, because a cord that fires without a command has
        // already failed and the rhythm measurement would be meaningless.
        let quiet: f64 = go(None).iter().sum::<f64>() / (ticks * legs.len()) as f64;
        let driven = go(Some(current));
        let r = analyse(&driven, DT, BAND);
        let rate = r.rate / legs.len() as f64;

        // A rate term that is worst at both ends: a silent cord and a
        // saturated one are both wrong, and the second is the one an
        // unconstrained search picks because it makes every other term easy.
        let alive = if rate > 0.0 {
            (-((rate / TARGET_RATE).ln().powi(2)) / (2.0 * 1.1f64.powi(2))).exp()
        } else {
            0.0
        };
        // Silence is a gate rather than a term: a cord that is 1% active with
        // no input at all is not quiet, however good its rhythm looks.
        let silent = (1.0 - quiet / 0.002).clamp(0.0, 1.0);
        // Graded in frequency rather than gated on it. A hard in-band test
        // makes the objective identically zero everywhere the cord starts
        // from, and a search cannot follow a gradient that does not exist:
        // the first run of this scored 0.0000 for every candidate in the
        // generation. Distance from the band is measured in octaves, so
        // being a factor of two out costs the same above and below.
        let nearest = r.hz.clamp(BAND.0, BAND.1);
        let octaves = if r.hz > 0.0 { (r.hz / nearest).log2().abs() } else { 6.0 };
        let in_band = (-(octaves * octaves) / (2.0 * 0.75f64.powi(2))).exp();
        let periodic = r.strength.max(0.0) * in_band;
        (periodic * alive * silent, periodic, rate, quiet)
    };

    let knobs = ph.knobs();
    // Start AT the connectome as measured, not at the bottom of every range.
    // `Knob::at` maps a value into search coordinates, and handing it zeros
    // starts the search with every one of the 707 connection gains at 0 - a
    // disconnected network, which is what the first run of this actually did.
    let unit: Vec<f32> = knobs
        .iter()
        .map(|k| if k.name.starts_with("bias:") { 0.0 } else { 1.0 })
        .collect();
    let start = unit.clone();
    let (s, per, rate, quiet) = evaluate(&mut net, &unit);
    println!("\nuniform physiology: score {s:.4} (rhythm {per:.3}, rate {rate:.4}/neuron/tick, idle {quiet:.5})");
    let (ss, sper, _, _) = evaluate(&mut shuffled, &unit);
    println!("  the same on a shuffled cord: score {ss:.4} (rhythm {sper:.3})");

    let mut es = Es::new(knobs, &start, num("SIGMA", 0.35), num("RATE", 0.25), num("SEED", 7)).expect("knobs match");
    let generations: u32 = num("GENERATIONS", 20);
    let pairs: usize = num("PAIRS", 6);
    println!("\n{:>5}  {:>10}  {:>10}  {:>9}  {:>9}  {:>10}", "gen", "best", "mean", "rhythm", "rate", "shuffled");
    let mut best_ever = (f64::NEG_INFINITY, Vec::new());
    for g in 1..=generations {
        let mut detail = (0.0, 0.0);
        let out = es.step(pairs, |values, _seed| {
            let (score, per, rate, _) = evaluate(&mut net, values);
            if score > detail.0 {
                detail = (score, per);
                let _ = rate;
            }
            score
        });
        let centre = es.centre();
        let (cs, cper, crate_, _) = evaluate(&mut net, &centre);
        let (shuf, _, _, _) = evaluate(&mut shuffled, &centre);
        println!("{g:>5}  {:>10.4}  {:>10.4}  {cper:>9.3}  {crate_:>9.4}  {shuf:>10.4}", out.best, out.mean);
        if cs > best_ever.0 {
            best_ever = (cs, centre);
        }
        let _ = out.best_params;
    }

    println!("\nbest centre: {:.4}", best_ever.0);
    let (bs, bper, brate, bquiet) = evaluate(&mut net, &best_ever.1);
    let (shuf, shufper, _, _) = evaluate(&mut shuffled, &best_ever.1);
    println!("  real cord:     score {bs:.4}, rhythm {bper:.3}, rate {brate:.4}, idle {bquiet:.5}");
    println!("  shuffled cord: score {shuf:.4}, rhythm {shufper:.3}");
    println!("\na calibration that helps the shuffle as much as the cord fitted the objective,");
    println!("not the animal.");

    if let Ok(path) = std::env::var("OUT") {
        let mut t = fly::Tuning::default();
        for (k, v) in es.knobs().iter().zip(&best_ever.1) {
            t.set(k.name.clone(), *v);
        }
        match t.save(&path) {
            Ok(()) => println!("wrote {path}"),
            Err(e) => eprintln!("{e}"),
        }
    }
}
