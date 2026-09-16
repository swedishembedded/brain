// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Where a control tick's time actually goes, on both sides of the loop.
//!
//! Run by hand when the rate changes. The headline it was written to find:
//! stepping the network without reading it back only SUBMITS work, so the cost
//! of a neural tick is one GPU round trip rather than the kernel time, and the
//! body - not the connectome - is what caps this loop.
use neuro::{DynamicalSystem, Port, SpikingNet};
use std::time::Instant;

fn main() {
    let dir = std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()).join("manc-codex");
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz")).unwrap();
    let csc = c.signed_csc(1e-3);
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let lif = fly::cord_lif();
    let mut net = SpikingNet::new(gpu, &csc, lif).unwrap();
    let n = c.neurons.len();
    let mut spike = vec![0.0f32; n];
    let drive = vec![0.5f32; n];
    net.drive(Port::Drive, &drive).unwrap();

    for _ in 0..50 { net.step(); }

    let n_it = 500;
    let t = Instant::now();
    for _ in 0..n_it { net.step(); }
    let step_only = t.elapsed().as_secs_f64() / n_it as f64;

    let t = Instant::now();
    for _ in 0..n_it { net.step(); net.read(Port::Spike, &mut spike).unwrap(); }
    let step_read = t.elapsed().as_secs_f64() / n_it as f64;

    let t = Instant::now();
    for _ in 0..n_it { net.read(Port::Spike, &mut spike).unwrap(); }
    let read_only = t.elapsed().as_secs_f64() / n_it as f64;

    let t = Instant::now();
    for _ in 0..n_it { net.drive(Port::Drive, &drive).unwrap(); }
    let write_only = t.elapsed().as_secs_f64() / n_it as f64;

    println!("neural step (no readback) : {:8.3} ms", step_only * 1e3);
    println!("spike readback            : {:8.3} ms", read_only * 1e3);
    println!("drive write               : {:8.3} ms", write_only * 1e3);
    println!("step + readback           : {:8.3} ms", step_read * 1e3);
    println!();
    println!("one neural tick with a readback costs {:.2} ms; the 2.00 ms budget a",
             step_read * 1e3 + write_only * 1e3);
    println!("500 Hz control rate allows is therefore mostly spent on the body, not the cord.");

    // The other side of the loop, for comparison.
    if let (Ok(mjl), Ok(xml)) = (mujoco::MuJoCo::load(), std::env::var("BRAIN_FLYBODY_XML")) {
        if let Ok(model) = mujoco::Model::from_xml(&mjl, &xml) {
            let mut d = mujoco::Data::new(&model).unwrap();
            for _ in 0..200 { d.step(&model); }
            let t0 = d.time(&model);
            let t = Instant::now();
            for _ in 0..4000 { d.step(&model); }
            let wall = t.elapsed().as_secs_f64();
            let sim = d.time(&model) - t0;
            println!();
            println!("body: 4000 physics steps in {wall:.3} s wall for {sim:.4} s simulated");
            println!("      = {:.2}x slower than real time, so the body alone caps the loop", wall / sim);
            println!("        at about {:.0} Hz whatever the nervous system costs.", 1000.0 / (2.0 * wall / sim));
        }
    }
}
