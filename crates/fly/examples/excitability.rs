// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Set the synaptic scale by physiology, not by a behaviour score.
//!
//! This crate's `weight_scale` default of 0.6 was chosen by sweeping it
//! against the gait criterion: the value that made a walk score best. That is
//! the same mistake as every other result here that did not survive - a
//! parameter fitted to the objective it is later used to evaluate, with
//! nothing outside the loop to say whether the number is physically sensible.
//!
//! The published whole-brain leaky integrate-and-fire model of this animal
//! fixes the scale by dimension instead. Its cells rest 7 mV below threshold
//! and each synapse contributes 0.275 mV, so roughly 25 coincident synapses
//! fire a neuron. In this runtime a cell rests at 0 with a threshold of 1, so
//! the equivalent per-synapse contribution is 1/25 = 0.04, against the 0.6
//! actually in use. That is a factor of fifteen, and it is measured here
//! rather than asserted: what a scale does to the firing rate of the
//! population is the thing that matters, and it depends on the in-degree
//! distribution, the size normalisation and the synaptic filter as well as on
//! the scale itself.
//!
//! Three targets, none of them a behaviour:
//!
//!   silent at rest    a cord with no input must not fire. Published models
//!                     give these neurons zero basal rate.
//!   a plausible rate  a driven cord should fire at tens of hertz per neuron,
//!                     not hundreds and not zero.
//!   a sparse code     where there is a mushroom body, an odour should recruit
//!                     a few percent of the Kenyon cells. This is the single
//!                     most diagnostic number in the animal, because the
//!                     sparseness is actively maintained by feedback
//!                     inhibition and is therefore a property a correct model
//!                     reproduces rather than one it can be tuned to.

use connectome::Connectome;
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

/// Spikes per neuron per second that a driven invertebrate neuron plausibly
/// sustains. Wide, because it is a sanity band and not a fit.
const RATE_BAND: (f64, f64) = (2.0, 60.0);
/// Fraction of Kenyon cells an odour should recruit.
const KC_SPARSENESS: (f64, f64) = (0.01, 0.15);

struct Measured {
    scale: f32,
    idle: f64,
    rate: f64,
    active: f64,
    kc: Option<f64>,
}

fn main() {
    let which = match std::env::var("CNS").unwrap_or_else(|_| "cord".into()).as_str() {
        "brain" | "banc" => Cns::BrainAndCord,
        _ => Cns::Cord,
    };
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let n = c.neurons.len();
    let ticks: usize = num("TICKS", 400);
    let settle: usize = num("SETTLE", 100);
    let drive_current: f32 = num("DRIVE", 4.0);

    // What gets driven, and what is watched for a sparse code.
    let command = c.population(|x| x.super_class == "descending");
    let mb = connectome::MushroomBody::find(&c, connectome::mushroom_body::Policy::default());
    let orns: Vec<u32> = {
        let want: std::collections::HashSet<String> =
            ["DM1", "DM2", "DM4", "DM5", "DM6"].iter().map(|g| format!("ORN_{g}")).collect();
        c.population(|x| x.class == "olfactory_receptor_neuron" && want.contains(&x.cell_type))
    };
    println!(
        "{}: {n} neurons, {} descending, {} Kenyon cells, {} receptor neurons",
        c.dataset,
        command.len(),
        mb.kc.len(),
        orns.len()
    );

    let scales: Vec<f32> = std::env::var("SCALES")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![0.6, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]);

    let measure = |scale: f32| -> Measured {
        let w = Wiring { weight_scale: scale, ..Wiring::default() };
        let exempt = mb.plastic_pairs(&c);
        let graph = c.network_keeping(w.weight_scale, w.size_limit, w.min_synapses, &exempt);
        let mut net =
            SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &graph, fly::cord_lif()).expect("it runs");
        let mut spike = vec![0.0f32; n];

        let mut run = |driven: &[u32], current: f32| -> (f64, f64) {
            let mut drive = vec![0.0f32; n];
            for &d in driven {
                drive[d as usize] = current;
            }
            net.reset(0);
            net.drive(Port::Drive, &drive).expect("fits");
            let mut total = 0u64;
            let mut fired = vec![false; n];
            for t in 0..settle + ticks {
                net.step();
                net.read(Port::Spike, &mut spike).expect("fits");
                if t < settle {
                    continue;
                }
                for (i, s) in spike.iter().enumerate() {
                    if *s > 0.5 {
                        total += 1;
                        fired[i] = true;
                    }
                }
            }
            // Spikes per neuron per second, and the fraction that fired at all.
            let seconds = ticks as f64 * fly::CONTROL_PERIOD;
            (total as f64 / n as f64 / seconds, fired.iter().filter(|f| **f).count() as f64 / n as f64)
        };

        let (idle, _) = run(&[], 0.0);
        let (rate, active) = run(&command, drive_current);
        let kc = (!mb.kc.is_empty() && !orns.is_empty()).then(|| {
            let mut drive = vec![0.0f32; n];
            for &i in &orns {
                drive[i as usize] = drive_current;
            }
            net.reset(0);
            net.drive(Port::Drive, &drive).expect("fits");
            let mut fired = vec![false; mb.kc.len()];
            for t in 0..settle + ticks {
                net.step();
                net.read(Port::Spike, &mut spike).expect("fits");
                if t < settle {
                    continue;
                }
                for (slot, &k) in fired.iter_mut().zip(&mb.kc) {
                    *slot |= spike[k as usize] > 0.5;
                }
            }
            fired.iter().filter(|f| **f).count() as f64 / mb.kc.len() as f64
        });
        Measured { scale, idle, rate, active, kc }
    };

    println!(
        "\n{:>8}  {:>10}  {:>10}  {:>9}  {:>9}  {}",
        "scale", "idle Hz", "driven Hz", "% active", "KC active", "verdict"
    );
    let mut ok: Vec<Measured> = Vec::new();
    for s in scales {
        let m = measure(s);
        let mut why: Vec<&str> = Vec::new();
        if m.idle > 0.1 {
            why.push("fires with no input");
        }
        if m.rate < RATE_BAND.0 {
            why.push("too quiet to drive a body");
        }
        if m.rate > RATE_BAND.1 {
            why.push("saturated");
        }
        if let Some(k) = m.kc {
            if k > KC_SPARSENESS.1 {
                why.push("no sparse odour code");
            } else if k < KC_SPARSENESS.0 {
                why.push("odour reaches no Kenyon cell");
            }
        }
        println!(
            "{:>8.3}  {:>10.2}  {:>10.2}  {:>8.1}%  {:>8}  {}",
            m.scale,
            m.idle,
            m.rate,
            100.0 * m.active,
            m.kc.map(|k| format!("{:.1}%", 100.0 * k)).unwrap_or_else(|| "-".into()),
            if why.is_empty() { "OK".to_string() } else { why.join(", ") }
        );
        if why.is_empty() {
            ok.push(m);
        }
    }

    println!("\ntargets: idle 0 Hz, driven {:.0}-{:.0} Hz, Kenyon cells {:.0}-{:.0}% of the population",
        RATE_BAND.0, RATE_BAND.1, 100.0 * KC_SPARSENESS.0, 100.0 * KC_SPARSENESS.1);
    match ok.len() {
        0 => println!("NO scale in this sweep satisfies the physiology. Widen SCALES or the model is wrong elsewhere."),
        _ => {
            // The geometric middle of the acceptable range, which is the value
            // furthest from both failure modes on a log scale.
            let lo = ok.iter().map(|m| m.scale).fold(f32::MAX, f32::min);
            let hi = ok.iter().map(|m| m.scale).fold(0.0f32, f32::max);
            println!("acceptable: {lo} to {hi}; the middle of that range is {:.3}", (lo * hi).sqrt());
        }
    }
}
