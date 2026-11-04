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
    oscillation_at(c, drop, shuffle, current, fly::cord_lif().v_min)
}

fn oscillation_at(
    c: &connectome::Connectome,
    drop: Option<&str>,
    shuffle: Option<u64>,
    current: f32,
    v_min: f32,
) -> Rhythm {
    let sub = c.subgraph(|n| CIRCUIT.contains(&n.cell_type.as_str()) && Some(n.cell_type.as_str()) != drop);
    let command = sub.population(|n| n.cell_type == "DNg100");
    let watch = sub.population(|n| n.cell_type == "INXXX466");
    let w = fly::Wiring::default();
    let mut graph = sub.network(w.weight_scale, w.size_limit, w.min_synapses);
    if let Some(s) = shuffle {
        graph = graph.shuffled_sources(s);
    }
    let lif = fly::LifParams { v_min, ..fly::cord_lif() };
    let mut net = SpikingNet::new(gpu_core::testgpu::dev(&neuro::KERNELS), &graph, lif).expect("the circuit runs");
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

/// RETRACTED, and kept as the measurement that retracts it.
///
/// This file previously asserted that the connectome's own three-neuron
/// circuit oscillates at 13.5 Hz under tonic drive, with a degree-matched
/// shuffle and a deleted-inhibition control both failing. Those numbers were
/// real. The conclusion was not, because the model they came from let a
/// membrane be driven arbitrarily far below rest.
///
/// A real neuron cannot. Inhibition opens channels whose reversal potential
/// sits about 20 mV below a rest that is itself 7 mV below threshold, so the
/// membrane approaches roughly three threshold-gaps down and stops. Sweeping
/// that floor:
///
///   v_min     -3      physiological      never oscillates in band
///   v_min     -6                         in band, strength 0.09
///   v_min    -12                         in band, strength 0.37
///   unbounded                            in band, strength 0.94
///
/// The oscillation needs about twelve threshold-gaps of hyperpolarisation,
/// which for this animal is 84 mV below rest. Nothing in a fly reaches that.
/// The rhythm was rebound from a hyperpolarisation no neuron can experience.
///
/// The published result it was supposed to reproduce used a RATE model with a
/// rectified tanh, bounded below at zero by construction, where this failure
/// mode cannot occur. A spiking port of it is not a reproduction of it, and
/// this is what that difference cost.
#[test]
fn the_oscillation_needs_hyperpolarisation_deeper_than_a_neuron_can_reach() {
    let Some(c) = cord() else {
        eprintln!("skipping: set BRAIN_CONNECTOME_DIR to a directory holding manc/");
        return;
    };
    for t in CIRCUIT {
        assert!(!c.population(|n| n.cell_type == t).is_empty(), "MANC has no {t}");
    }
    let currents = [1.0f32, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];
    let best = |v_min: f32| -> f64 {
        currents
            .iter()
            .map(|&i| oscillation_at(&c, None, None, i, v_min))
            .filter(|r| r.is_rhythmic(0.0))
            .map(|r| r.strength)
            .fold(0.0f64, f64::max)
    };

    let physiological = best(fly::cord_lif().v_min);
    let unbounded = best(-1.0e9);
    eprintln!("best in-band strength: physiological floor {physiological:.3}, unbounded {unbounded:.3}");

    assert!(
        physiological < 0.25,
        "the circuit oscillates at a physiological inhibitory reversal ({physiological:.3}); if this \
         starts passing the model changed and the retraction above should be revisited"
    );
    assert!(
        unbounded > physiological + 0.2,
        "removing the floor no longer produces the oscillation ({unbounded:.3} against {physiological:.3}), \
         so the artefact this test records is gone and the test has stopped measuring anything"
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

// The whole-cord version of the same retraction, recorded rather than
// re-asserted.
//
// This file also claimed that tonic DNg100 drive makes the leg motor pool
// oscillate at 16.67 Hz where a degree-matched shuffle rings at 125 Hz at
// every drive, and that a size-matched descending population is six times
// weaker. Under a physiological inhibitory reversal the best in-band strength
// DNg100 reaches is 0.201, and the size-matched control reaches 0.778 - the
// specificity is not merely weaker, it is reversed.
//
// There is no whole-cord test here any more because there is nothing left to
// gate: the claim it was written to defend does not survive the correction,
// and a test that asserts the corrected negative would spend forty-five
// seconds a run to say what the isolated-circuit test above already says
// faster and more precisely.
