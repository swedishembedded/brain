// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Per-muscle recruitment: which leg muscles the cord actually drives, how
//! hard, and whether each one is rhythmic.
//!
//! The population of leg motor neurons oscillates at 13.9 Hz in the body, and
//! the coxa's own two muscles do not: the agonist rings at its refractory
//! period and the antagonist barely moves. A joint driven that way goes
//! nowhere however good the population rhythm looks. This asks the narrower
//! question the population measure cannot: is each muscle being recruited at
//! all, in what proportion, and in what phase relative to its antagonist.
//!
//! Phase is the number that decides it. Two muscles of one joint firing
//! rhythmically IN PHASE produce no movement at all, and every summary that
//! looks at the net drive reports that as "no rhythm" rather than as "perfect
//! rhythm, perfectly cancelled" - which need completely different fixes.
use std::collections::BTreeMap;

use fly::rhythm::analyse;
use fly::{Cns, Coupling, Fly, Timing, Wiring};
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

const BAND: (f64, f64) = (3.0, 30.0);

/// Normalised cross-correlation at lag zero: +1 in phase, -1 antiphase.
fn phase(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (ma, mb) = (a.iter().sum::<f64>() / n as f64, b.iter().sum::<f64>() / n as f64);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        0.0
    } else {
        num / (da * db).sqrt()
    }
}

fn main() {
    let mj = MuJoCo::load().unwrap();
    let c = fly::cns::load(env("BRAIN_CONNECTOME_DIR"), Cns::Cord).unwrap();
    let path = std::path::PathBuf::from(env("BRAIN_FLYBODY_XML"));
    let ticks: usize = num("TICKS", 1500);
    let settle: usize = num("SETTLE", 250);
    let command: f32 = num("COMMAND", 30.0);
    let dn = std::env::var("DN").unwrap_or_else(|_| "DNg100".into());

    // The muscles of one leg, by the connectome's own names, split into the
    // populations that drive each degree of freedom.
    let seg = flybody::Segment::T1;
    let side = flybody::Side::Left;
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for (i, n) in c.neurons.iter().enumerate() {
        if n.super_class != "motor" || !matches!(n.class.as_str(), "fl" | "ml" | "hl") {
            continue;
        }
        let Some((s, muscle)) = flybody::parse_sub_class(&n.sub_class) else { continue };
        if s != seg || flybody::Side::parse(&n.soma_side) != Some(side) {
            continue;
        }
        groups.entry(muscle).or_default().push(i as u32);
    }

    let model = Model::from_xml(&mj, &path).unwrap();
    let mut f = Fly::new(
        gpu_core::testgpu::dev(&neuro::KERNELS),
        &c,
        model,
        fly::cord_lif(),
        Wiring { weight_scale: num("SCALE", 0.6f32), ..Wiring::default() },
        Timing::default(),
        Coupling { adhesion_gain: num("ADHESION", 0.4), ..Coupling::default() },
    )
    .unwrap();
    f.reset();
    if f.drive_cell_type(&dn, command).unwrap_or(0) == 0 {
        eprintln!("no descending neuron of type {dn}");
        std::process::exit(2);
    }

    let names: Vec<String> = groups.keys().cloned().collect();
    let mut trace: Vec<Vec<f64>> = vec![Vec::with_capacity(ticks); names.len()];
    for t in 0..settle + ticks {
        f.step().expect("a tick runs");
        if t < settle {
            continue;
        }
        let spike = f.spikes();
        for (row, name) in trace.iter_mut().zip(&names) {
            row.push(groups[name].iter().filter(|&&i| spike[i as usize] > 0.5).count() as f64);
        }
    }

    let seconds = ticks as f64 * fly::CONTROL_PERIOD;
    println!("{seg:?} {side:?} leg, command {command} to {dn}, {ticks} ticks ({seconds:.0} s)\n");
    println!("{:>34}  {:>5}  {:>9}  {:>9}  {:>8}", "muscle", "cells", "Hz/cell", "rhythm", "at Hz");
    for (name, row) in names.iter().zip(&trace) {
        let cells = groups[name].len();
        let r = analyse(row, fly::CONTROL_PERIOD, BAND);
        let per_cell = row.iter().sum::<f64>() / cells as f64 / seconds;
        println!(
            "{name:>34}  {cells:>5}  {per_cell:>9.1}  {:>9.3}  {:>8.2}{}",
            r.strength,
            r.hz,
            if r.is_rhythmic(0.25) { "  IN BAND" } else { "" }
        );
    }

    // The antagonist pairs this body actually opposes, and their phase.
    println!("\n{:>24} {:>24}  {:>8}  {}", "agonist", "antagonist", "phase", "verdict");
    for (a, b) in [
        ("Tergopleural/Pleural_promotor", "Tergotr."),
        ("Ti_extensor", "Ti_flexor"),
        ("Tr_extensor", "Tr_flexor"),
        ("Ta_levator", "Ta_depressor"),
        ("Sternal_anterior_rotator", "Sternal_posterior_rotator"),
    ] {
        let (Some(i), Some(j)) = (names.iter().position(|n| n == a), names.iter().position(|n| n == b)) else {
            continue;
        };
        let p = phase(&trace[i], &trace[j]);
        println!(
            "{a:>24} {b:>24}  {p:>8.3}  {}",
            if p > 0.5 {
                "IN PHASE - the joint is being pulled both ways at once"
            } else if p < -0.3 {
                "antiphase, which is what moves a joint"
            } else {
                "uncorrelated"
            }
        );
    }
}
