// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Is the walking rhythm already in the wiring?
//!
//! Everything else in this crate asks a search to produce a behaviour. This
//! asks the connectome a question and optimises nothing at all. One descending
//! cell type is driven with a CONSTANT current - no oscillator, no pattern
//! written by hand, nothing time-varying anywhere in the input - and the leg
//! motor neurons are watched for a rhythm. A constant input cannot contain a
//! period, so if one appears at the motor pool, the graph made it.
//!
//! The published prediction being tested is specific and quantitative: tonic
//! DNg100 activity drives walking, and the step frequency RISES with the
//! strength of the drive. A monotone frequency-current relation is a much
//! harder thing to produce by accident than a single oscillation, which is why
//! the sweep matters more than any one row.
//!
//! Controls, in the order they rule things out:
//!
//!   silence        the cord must not already be oscillating on its own
//!   other cells    a size-matched descending population that is not DNg100
//!   shuffled       degree-matched shuffle, same drive, same everything else

use fly::rhythm::{analyse, Rhythm};
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

/// A walking fly steps at something like 5 to 20 hertz. Asking inside a band
/// is what separates a measurement from a search for any structure at all.
const BAND: (f64, f64) = (3.0, 30.0);

fn main() {
    let which = match std::env::var("CNS").unwrap_or_else(|_| "cord".into()).as_str() {
        "brain" | "banc" => Cns::BrainAndCord,
        _ => Cns::Cord,
    };
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });

    let command_type = std::env::var("DN").unwrap_or_else(|_| "DNg100".into());
    let command = c.population(|n| n.cell_type == command_type);
    let legs = c.population(|n| n.super_class == "motor" && matches!(n.class.as_str(), "fl" | "ml" | "hl"));
    // The oscillator the connectome study isolated by pruning: two excitatory
    // interneurons and one inhibitory one. Watched separately, because if the
    // motor pool oscillates then these are where it should be coming from.
    let cpg: Vec<(String, Vec<u32>)> = ["IN17A001", "INXXX466", "IN16B036"]
        .iter()
        .map(|t| (t.to_string(), c.population(|n| n.cell_type == *t)))
        .collect();

    if command.is_empty() || legs.is_empty() {
        eprintln!("{}: {} command cells, {} leg motor neurons - nothing to measure", c.dataset, command.len(), legs.len());
        std::process::exit(1);
    }
    println!("{}: {} neurons", c.dataset, c.neurons.len());
    println!("command: {} {command_type} cells", command.len());
    println!("readout: {} leg motor neurons", legs.len());
    for (name, cells) in &cpg {
        println!("  {name}: {} cells", cells.len());
    }

    let wiring = Wiring { min_synapses: num("MIN_SYNAPSES", 5), ..Wiring::default() };
    let ticks: usize = num("TICKS", 1500);
    let settle: usize = num("SETTLE", 250);
    let dt = 0.002;

    let build = |seed: Option<u64>| {
        let mut graph = c.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
        if let Some(s) = seed {
            graph = graph.shuffled_sources(s);
        }
        SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &graph, fly::cord_lif()).expect("the cord runs")
    };

    // One run: constant current into `driven`, and the per-tick population
    // spike count of each watched group.
    let run = |net: &mut SpikingNet, driven: &[u32], current: f32, watch: &[&[u32]]| -> Vec<Vec<f64>> {
        net.reset(0);
        let n = net.port_len(Port::Spike);
        let mut drive = vec![0.0f32; n];
        for &d in driven {
            drive[d as usize] = current;
        }
        net.drive(Port::Drive, &drive).expect("the drive fits");
        let mut spike = vec![0.0f32; n];
        let mut out = vec![Vec::with_capacity(ticks); watch.len()];
        for t in 0..settle + ticks {
            net.step();
            net.read(Port::Spike, &mut spike).expect("the readback fits");
            if t < settle {
                continue;
            }
            for (series, group) in out.iter_mut().zip(watch) {
                series.push(group.iter().map(|&i| spike[*&i as usize] as f64).sum());
            }
        }
        out
    };

    let row = |label: &str, r: &Rhythm| {
        println!(
            "{label:>22}  {:>8.3}  {:>8.3}  {:>8.2}  {:>9.3}  {}",
            r.rate,
            r.deviation,
            r.hz,
            r.strength,
            if r.is_rhythmic(0.3) { "in band" } else if r.hz > 0.0 { "out of band" } else { "silent" }
        );
    };

    let mut net = build(None);
    let watch: Vec<&[u32]> = std::iter::once(legs.as_slice()).chain(cpg.iter().map(|(_, v)| v.as_slice())).collect();

    println!("\ndriving {command_type} tonically, {ticks} ticks at {} ms after {settle} settling", dt * 1000.0);
    println!(
        "\n{:>22}  {:>8}  {:>8}  {:>7}  {:>9}",
        "condition", "rate", "sd", "Hz", "rhythm"
    );

    // The sweep IS the experiment: one oscillation is an anecdote, a frequency
    // that tracks the drive is a mechanism.
    let currents: Vec<f32> = std::env::var("CURRENTS")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]);
    let mut sweep: Vec<(f32, Rhythm)> = Vec::new();
    for &i in &currents {
        let series = run(&mut net, &command, i, &watch);
        let r = analyse(&series[0], dt, BAND);
        row(&format!("{command_type} at {i}"), &r);
        sweep.push((i, r));
    }

    // Where the rhythm should be coming from, if it is coming from anywhere.
    let strongest = sweep.iter().skip(1).max_by(|a, b| a.1.strength.total_cmp(&b.1.strength)).map(|x| x.0).unwrap_or(8.0);
    println!("\nthe pruned oscillator at {command_type} = {strongest:.0}:");
    let series = run(&mut net, &command, strongest, &watch);
    for (i, (name, cells)) in cpg.iter().enumerate() {
        if cells.is_empty() {
            continue;
        }
        row(name, &analyse(&series[i + 1], dt, BAND));
    }

    println!("\ncontrols at {command_type} = {strongest:.0}:");
    // Size-matched descending cells that are not the ones being claimed.
    let others: Vec<u32> = c
        .population(|n| n.super_class == "descending" && n.cell_type != command_type)
        .into_iter()
        .take(command.len())
        .collect();
    if !others.is_empty() {
        let series = run(&mut net, &others, strongest, &watch);
        row("other descending", &analyse(&series[0], dt, BAND));
    }
    let mut shuffled = build(Some(0x5EED));
    let series = run(&mut shuffled, &command, strongest, &watch);
    row("shuffled cord", &analyse(&series[0], dt, BAND));

    // THE CIRCUIT ALONE.
    //
    // The connectome study did not find its oscillator by watching the whole
    // cord: it pruned the network until three interneurons were left. That is
    // the right experiment to repeat, because it separates two things this
    // measurement otherwise confounds. A circuit can be present, correctly
    // wired and correctly signed, and still be invisible inside a cord that
    // is firing at twenty spikes a tick around it - and "the oscillator is not
    // in the anatomy" and "the oscillator is drowned by everything else" call
    // for completely different work.
    let members = [command_type.as_str(), "IN17A001", "INXXX466", "IN16B036"];
    let sub = c.subgraph(|n| members.contains(&n.cell_type.as_str()));
    let s_command = sub.population(|n| n.cell_type == command_type);
    let s_cpg: Vec<(String, Vec<u32>)> = ["IN17A001", "INXXX466", "IN16B036"]
        .iter()
        .map(|t| (t.to_string(), sub.population(|n| n.cell_type == *t)))
        .collect();
    let graph = sub.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
    println!(
        "\nthe circuit alone: {} neurons, {} edges, driving {command_type}",
        sub.neurons.len(),
        graph.nnz()
    );
    let mut iso = SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &graph, fly::cord_lif()).expect("it runs");
    let s_watch: Vec<&[u32]> = s_cpg.iter().map(|(_, v)| v.as_slice()).collect();
    let mut iso_sweep: Vec<(f32, Rhythm)> = Vec::new();
    for &i in &currents {
        let series = run(&mut iso, &s_command, i, &s_watch);
        // E2 is the readout: it is one synapse from the inhibition that is
        // supposed to be shutting the loop down, so a period shows there.
        let r = analyse(&series[1], dt, BAND);
        row(&format!("INXXX466 at {i}"), &r);
        iso_sweep.push((i, r));
    }
    let best_iso = iso_sweep
        .iter()
        .filter(|(i, _)| *i > 0.0)
        .max_by(|a, b| a.1.strength.total_cmp(&b.1.strength))
        .map(|x| x.0)
        .unwrap_or(4.0);
    for (i, (name, cells)) in s_cpg.iter().enumerate() {
        if cells.is_empty() {
            continue;
        }
        let series = run(&mut iso, &s_command, best_iso, &s_watch);
        row(&format!("{name} at {best_iso:.0}"), &analyse(&series[i], dt, BAND));
    }

    // Two controls, and the second is the one that names a mechanism.
    //
    // A shuffle asks whether this particular wiring mattered or whether any
    // twenty cells with these degrees would ring. Removing the inhibitory
    // neuron asks something sharper: the proposed mechanism is that
    // excitation builds, recruits inhibition, is shut down by it, and restarts
    // as the inhibition decays. If that is what is happening, then deleting
    // IN16B036 must not slow the rhythm down or weaken it - it must abolish
    // it, and leave the excitatory cells firing steadily instead.
    println!("\ncontrols on the circuit alone, at {command_type} = {best_iso:.0}:");
    let shuffled_graph = graph.shuffled_sources(0x5EED);
    let mut iso_shuf =
        SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &shuffled_graph, fly::cord_lif()).expect("it runs");
    let series = run(&mut iso_shuf, &s_command, best_iso, &s_watch);
    row("shuffled circuit", &analyse(&series[1], dt, BAND));

    let no_inhibition = c.subgraph(|n| members.contains(&n.cell_type.as_str()) && n.cell_type != "IN16B036");
    let ni_command = no_inhibition.population(|n| n.cell_type == command_type);
    let ni_e2 = no_inhibition.population(|n| n.cell_type == "INXXX466");
    let ni_graph = no_inhibition.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
    let mut ni = SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &ni_graph, fly::cord_lif()).expect("it runs");
    let series = run(&mut ni, &ni_command, best_iso, &[ni_e2.as_slice()]);
    row("without IN16B036", &analyse(&series[0], dt, BAND));

    println!("\nthe input is constant, so any period at the motor pool was made by the graph.");
    for (what, sw) in [("whole cord", &sweep), ("circuit alone", &iso_sweep)] {
        let active: Vec<&(f32, Rhythm)> = sw.iter().filter(|(i, _)| *i > 0.0).collect();
        let rising = active.windows(2).filter(|w| w[1].1.hz > w[0].1.hz).count();
        println!(
            "{what}: frequency rose with drive on {rising} of {} steps, {} Hz at the weakest drive to {} Hz at the strongest",
            active.len().saturating_sub(1),
            active.first().map_or(0.0, |x| x.1.hz).round(),
            active.last().map_or(0.0, |x| x.1.hz).round()
        );
    }
}
