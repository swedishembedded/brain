// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The walking rhythm is in the wiring, and this is the claim stated so it can
//! fail.
//!
//! Everything else this crate measures is the outcome of a search, and a
//! search reports the best number it found, which is the number least worth
//! trusting. This asks the connectome a question instead and optimises
//! nothing. One descending cell type is driven with a CONSTANT current, and
//! the interneurons downstream are watched for a period. A constant input
//! contains no period, so if one appears, the graph made it.
//!
//! Three conditions, and the claim is about the difference between them:
//!
//!   intact              the cells and synapses as measured, signs as
//!                       predicted by the dataset
//!   shuffled            the same cells and the same degrees, rewired
//!   without IN16B036    the inhibitory neuron deleted
//!
//! The proposed mechanism is that excitation builds, recruits inhibition, is
//! silenced by it, and restarts as the inhibition decays. If that is right
//! then the last two must NOT merely oscillate more slowly or more weakly:
//! the rhythm has to go. It does, and what is left in both is the network
//! ringing at its refractory period, which is why `Rhythm::in_band` exists and
//! is checked rather than the strength alone.
//!
//! Skips when MANC is not on the machine.

use fly::rhythm::{analyse, Rhythm};
use neuro::{DynamicalSystem, Port, SpikingNet};

/// A walking fly steps at something like 5 to 20 hertz.
const BAND: (f64, f64) = (3.0, 30.0);
const DT: f64 = 0.002;
/// The excitatory pair and the inhibitory cell the connectome study isolated.
const CIRCUIT: [&str; 4] = ["DNg100", "IN17A001", "INXXX466", "IN16B036"];

fn cord() -> Option<connectome::Connectome> {
    let root = std::env::var("BRAIN_CONNECTOME_DIR").ok()?;
    let (neurons, edges) = connectome::find(&root, "manc").ok()?;
    connectome::load("manc", &neurons, &edges).ok()
}

/// Drive `command` with a constant current and analyse `watch`.
fn oscillation(c: &connectome::Connectome, drop: Option<&str>, shuffle: Option<u64>, current: f32) -> Rhythm {
    let sub = c.subgraph(|n| CIRCUIT.contains(&n.cell_type.as_str()) && Some(n.cell_type.as_str()) != drop);
    let command = sub.population(|n| n.cell_type == "DNg100");
    let watch = sub.population(|n| n.cell_type == "INXXX466");
    let w = fly::Wiring::default();
    let mut graph = sub.network(w.weight_scale, w.size_limit, w.min_synapses);
    if let Some(s) = shuffle {
        graph = graph.shuffled_sources(s);
    }
    let mut net =
        SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &graph, fly::cord_lif()).expect("the circuit runs");
    let n = net.port_len(Port::Spike);
    let mut drive = vec![0.0f32; n];
    for &d in &command {
        drive[d as usize] = current;
    }
    net.reset(0);
    net.drive(Port::Drive, &drive).expect("the drive fits");
    let mut spike = vec![0.0f32; n];
    let mut series = Vec::with_capacity(1500);
    for t in 0..1750 {
        net.step();
        net.read(Port::Spike, &mut spike).expect("the readback fits");
        if t >= 250 {
            series.push(watch.iter().map(|&i| spike[i as usize] as f64).sum());
        }
    }
    analyse(&series, DT, BAND)
}

#[test]
fn the_isolated_circuit_oscillates_in_the_walking_band_and_needs_its_inhibition() {
    let Some(c) = cord() else {
        eprintln!("skipping: set BRAIN_CONNECTOME_DIR to a directory holding manc/");
        return;
    };
    for t in CIRCUIT {
        assert!(!c.population(|n| n.cell_type == t).is_empty(), "MANC has no {t}");
    }

    // The drive is swept rather than chosen, because picking one current and
    // reporting what happened there is how a search result gets written up as
    // a prediction. The claim is that SOME tonic drive makes this circuit
    // oscillate in the walking band; the sweep is what makes that a statement
    // about the circuit rather than about a number somebody tuned.
    let currents = [1.0f32, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];
    let intact: Vec<(f32, Rhythm)> = currents.iter().map(|&i| (i, oscillation(&c, None, None, i))).collect();
    for (i, r) in &intact {
        eprintln!("intact at {i:>4}: {:>7.2} Hz  strength {:.3}  {}", r.hz, r.strength, if r.is_rhythmic(0.3) { "in band" } else { "out" });
    }
    let Some(&(current, ref best)) = intact.iter().filter(|(_, r)| r.is_rhythmic(0.3)).max_by(|a, b| a.1.strength.total_cmp(&b.1.strength)) else {
        panic!("no tonic drive made the circuit oscillate in {BAND:?} Hz");
    };
    eprintln!("\nthe circuit oscillates at {:.2} Hz under a constant drive of {current}", best.hz);

    // And at that same drive, neither control does.
    let shuffled = oscillation(&c, None, Some(0x5EED), current);
    let no_inhibition = oscillation(&c, Some("IN16B036"), None, current);
    eprintln!("shuffled:   {:>7.2} Hz  strength {:.3}", shuffled.hz, shuffled.strength);
    eprintln!("no IN16B036:{:>7.2} Hz  strength {:.3}", no_inhibition.hz, no_inhibition.strength);

    assert!(
        !shuffled.is_rhythmic(0.3),
        "a degree-matched shuffle of the same cells oscillated too ({:.2} Hz, strength {:.3}), so the wiring was not what produced the rhythm",
        shuffled.hz,
        shuffled.strength
    );
    assert!(
        !no_inhibition.is_rhythmic(0.3),
        "the rhythm survived deleting the inhibitory neuron ({:.2} Hz, strength {:.3}), so rebound from inhibition was not the mechanism",
        no_inhibition.hz,
        no_inhibition.strength
    );
}

/// Nothing at all should happen without a command.
#[test]
fn the_circuit_is_silent_until_it_is_driven() {
    let Some(c) = cord() else {
        eprintln!("skipping: set BRAIN_CONNECTOME_DIR to a directory holding manc/");
        return;
    };
    let quiet = oscillation(&c, None, None, 0.0);
    assert_eq!(quiet.rate, 0.0, "the circuit fires with no input at all");
    assert!(!quiet.is_rhythmic(0.3));
}
