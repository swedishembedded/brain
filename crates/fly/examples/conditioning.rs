// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Differential conditioning of a connectome, with the controls that decide it.
//!
//! Prints the olfactory chain first, because a conditioning result on a fly
//! that cannot smell is a result about numerical drift. Then the four
//! conditions, of which UNPAIRED is the one that matters: same odour, same
//! dopamine, separated in time. A paired effect that unpaired reproduces is
//! not an association.

use connectome::mushroom_body::Policy;
use fly::conditioning::{index, Apparatus, Odour, Pairing, Protocol};
use fly::{Cns, Wiring};

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
    let which = match std::env::var("CNS").unwrap_or_else(|_| "brain".into()).as_str() {
        "cord" => Cns::Cord,
        _ => Cns::BrainAndCord,
    };
    let mut c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), which).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let (silent_cells, silent_synapses) = c.silenced();
    println!("cells with no predicted transmitter: {silent_cells}, carrying {silent_synapses} synapses that are multiplied by zero");
    for (cell, n) in fly::conditioning::restore_known_transmitters(&mut c) {
        println!("  restored {cell} x{n} from the literature");
    }
    let c = c;

    // Two sparse patterns over real, named glomeruli. Disjoint, so the
    // discrimination being tested is not confounded by shared input.
    let a = Odour::new(&c, "A", &["DM1", "DM2", "DM4", "DM5", "DM6"]);
    let b = Odour::new(&c, "B", &["VA1v", "VL2a", "VM4"]);
    for o in [&a, &b] {
        if o.is_empty() {
            eprintln!("odour {} names no receptor neuron in {}", o.name, c.dataset);
            std::process::exit(1);
        }
    }

    let wiring = Wiring::default();
    let plast = neuro::PlasticityParams {
        eta: num("ETA", 0.02),
        elig_decay: num("ELIG_DECAY", 0.98),
        w_min: -2.0,
        w_max: 2.0,
        ..Default::default()
    };
    // Negative: dopamine paired with Kenyon-cell activity DEPRESSES that
    // cell's output synapse. See `MushroomBody::sites`.
    let gain = num("GAIN", -1.0f32);
    let p = Protocol {
        trials: num("TRIALS", 12),
        odour_current: num("ODOUR", 4.0),
        reinforcer_current: num("DOPAMINE", 8.0),
        ..Protocol::default()
    };

    let build = |seed: Option<u64>| {
        Apparatus::new(
            gpu_core::testgpu::dev(&neuro::KERNELS),
            &c,
            Wiring { shuffle_seed: seed, ..wiring },
            fly::cord_lif(),
            plast,
            gain,
            Policy::default(),
        )
    };
    let mut app = build(None).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    println!("{}: {}", c.dataset, app.mb.summary());
    println!("plastic synapses: {} of {}", app.sites().plastic_edges(), app.weights().len());
    println!("odour A: {} receptor neurons in {:?}", a.orns.len(), a.glomeruli);
    println!("odour B: {} receptor neurons in {:?}", b.orns.len(), b.glomeruli);

    // The chain, before anything is claimed about learning. Silence first:
    // a response is only a response if it differs from no stimulus at all.
    let nkc = app.mb.kc.len();
    let quiet = app.spontaneous(&p);
    println!(
        "\n{:>10}  {:>8}  {:>10}  {:>10}  {:>10}  {:>8}",
        "stimulus", "receptor", "KC spikes", "KC active", "dopamine", "output"
    );
    println!(
        "{:>10}  {:>8}  {:>10.0}  {:>9.1}%  {:>10.0}  {:>8.0}",
        "silence",
        "-",
        quiet.kc,
        100.0 * quiet.sparseness(nkc),
        quiet.dan,
        quiet.total()
    );
    for o in [&a, &b] {
        let r = app.test(o, &p);
        println!(
            "{:>10}  {:>8.0}  {:>10.0}  {:>9.1}%  {:>10.0}  {:>8.0}",
            format!("odour {}", o.name),
            r.orn,
            r.kc,
            100.0 * r.sparseness(nkc),
            r.dan,
            r.total()
        );
    }
    println!("(a real mushroom body answers an odour with about 5% of its Kenyon cells)");

    // The aversive compartment, by the cell type the literature names.
    let dan_type = std::env::var("DAN").unwrap_or_else(|_| "PPL101".into());
    let Some(comp) = app.compartment_driven_by(&c, &dan_type) else {
        eprintln!("no compartment is innervated by {dan_type}");
        std::process::exit(1);
    };
    let name = app.mb.compartments[comp].name.clone();
    println!("\nreinforcing the {name} compartment, driven by {dan_type}\n");

    println!(
        "{:>16}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "condition", "A before", "A after", "index A", "index B", "A - B"
    );
    let mut rows: Vec<(String, f64)> = Vec::new();
    for (label, pairing, shuffled, plastic) in [
        ("paired", Pairing::Paired, false, true),
        ("unpaired", Pairing::Unpaired, false, true),
        ("odour only", Pairing::OdourOnly, false, true),
        ("dopamine only", Pairing::ReinforcerOnly, false, true),
        ("paired, frozen", Pairing::Paired, false, false),
        ("paired, shuffled", Pairing::Paired, true, true),
    ] {
        let mut app = if shuffled {
            build(Some(0x5EED)).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            })
        } else {
            build(None).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            })
        };
        let comp = app.compartment_driven_by(&c, &dan_type).unwrap_or(comp);
        let before_a = app.test(&a, &p);
        let before_b = app.test(&b, &p);
        app.set_plasticity(plastic);
        app.train(&a, &b, comp, pairing, &p);
        let after_a = app.test(&a, &p);
        let after_b = app.test(&b, &p);

        // The compartment that was taught, not the whole population: a
        // reinforcer that changed every output neuron equally changed nothing
        // an animal could act on.
        let m = app.mb.compartments[comp].mbon;
        let slot = app.mb.mbon.iter().position(|&x| x == m).expect("the compartment's output neuron");
        let (ia, ib) = (
            index(before_a.mbon[slot], after_a.mbon[slot]),
            index(before_b.mbon[slot], after_b.mbon[slot]),
        );
        println!(
            "{label:>16}  {:>9.0}  {:>9.0}  {ia:>9.4}  {ib:>9.4}  {:>9.4}",
            before_a.mbon[slot],
            after_a.mbon[slot],
            ia - ib
        );
        rows.push((label.to_string(), ia - ib));
    }

    println!("\nthe paired row has to beat every other row, and `paired, frozen` has to be");
    println!("exactly 0: an association that survives freezing the synapses was not stored");
    println!("in them.");
    let paired = rows[0].1;
    for (label, v) in &rows[1..] {
        if paired.abs() <= v.abs() {
            println!("  FAILED: {label} ({v:.4}) matches or beats paired ({paired:.4})");
        }
    }
}
