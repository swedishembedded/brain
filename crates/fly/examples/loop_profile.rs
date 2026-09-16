// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Where a control tick's time actually goes, on both sides of the loop.
//!
//! Run by hand when the rate changes, or when a machine runs the fly slowly.
//!
//! Every measurement here is taken from a DRAINED device. A submission is
//! asynchronous, so a loop that only calls `step` leaves a backlog behind it,
//! and the next loop measured pays for it - which is how this example once
//! reported a per-tick cost of twice the true one. `drain` below is the fix,
//! and it is also what separates the two numbers that matter: how long a tick
//! takes when the ticks can overlap, and how long one takes when the caller
//! waits for its spikes before doing anything else. The loop in `fly::Creature`
//! is the second kind.
use neuro::{DynamicalSystem, Plastic, Port, SpikingNet};
use std::time::Instant;

fn main() {
    let (neurons, edges) =
        connectome::find(std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()), "manc").unwrap();
    let c = connectome::load("manc", &neurons, &edges).unwrap();
    // The network the creature actually runs, not the raw graph: `Wiring`'s
    // synapse floor removes most of the edge list, and profiling the unpruned
    // graph measures a network nothing in this repo steps.
    let w = fly::Wiring::default();
    let csc = c.network(w.weight_scale, w.size_limit, w.min_synapses);
    println!(
        "{} neurons, {} edges at min_synapses={} ({} unpruned)",
        csc.n,
        csc.w.len(),
        w.min_synapses,
        c.signed_csc(1e-3).w.len()
    );
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let lif = fly::cord_lif();
    let mut net = SpikingNet::new(gpu, &csc, lif).unwrap();
    let n = c.neurons.len();
    let mut spike = vec![0.0f32; n];
    let drive = vec![0.5f32; n];
    net.drive(Port::Drive, &drive).unwrap();

    // A readback is a sync: whatever was queued is finished once it returns.
    let drain = |net: &SpikingNet| {
        let mut s = vec![0.0f32; n];
        net.read(Port::Spike, &mut s).unwrap();
    };

    for _ in 0..50 {
        net.step();
    }
    drain(&net);

    let n_it = 200;

    // Pipelined: submit every tick, wait once at the end. This is the
    // device's THROUGHPUT for the two kernels, with no round trip in it.
    let t = Instant::now();
    for _ in 0..n_it {
        net.step();
    }
    drain(&net);
    let pipelined = t.elapsed().as_secs_f64() / n_it as f64;

    // Serialised: the loop the creature actually runs - step, then read this
    // tick's spikes before deciding anything.
    let t = Instant::now();
    for _ in 0..n_it {
        net.step();
        net.read(Port::Spike, &mut spike).unwrap();
    }
    let step_read = t.elapsed().as_secs_f64() / n_it as f64;

    drain(&net);
    let t = Instant::now();
    for _ in 0..n_it {
        net.read(Port::Spike, &mut spike).unwrap();
    }
    let read_only = t.elapsed().as_secs_f64() / n_it as f64;

    let t = Instant::now();
    for _ in 0..n_it {
        net.drive(Port::Drive, &drive).unwrap();
    }
    drain(&net);
    let write_only = t.elapsed().as_secs_f64() / n_it as f64;

    // The plastic tick: three more kernels per step, one of them per-EDGE.
    // `--plastic` is a supported mode of the sample, so its cost is part of
    // this picture rather than a footnote.
    net.enable_plasticity(neuro::PlasticityParams::default()).unwrap();
    net.modulate(0.1);
    for _ in 0..10 {
        net.step();
    }
    drain(&net);
    let t = Instant::now();
    for _ in 0..n_it {
        net.modulate(0.1);
        net.step();
        net.read(Port::Spike, &mut spike).unwrap();
    }
    let plastic = t.elapsed().as_secs_f64() / n_it as f64;

    println!("neural step, pipelined    : {:8.3} ms", pipelined * 1e3);
    println!("spike readback (idle dev) : {:8.3} ms", read_only * 1e3);
    println!("drive write               : {:8.3} ms", write_only * 1e3);
    println!("step + readback           : {:8.3} ms", step_read * 1e3);
    println!("  the same, plasticity on : {:8.3} ms", plastic * 1e3);
    println!();
    let tick = step_read + write_only;
    println!("one neural tick costs {:.2} ms against a 2.00 ms budget at 500 Hz", tick * 1e3);
    println!("-> {:.3}x real time from the cord alone", 0.002 / tick);

    // The other side of the loop, for comparison.
    if let (Ok(mjl), Ok(xml)) = (mujoco::MuJoCo::load(), std::env::var("BRAIN_FLYBODY_XML")) {
        if let Ok(model) = mujoco::Model::from_xml(&mjl, &xml) {
            let mut d = mujoco::Data::new(&model).unwrap();
            for _ in 0..200 {
                d.step(&model);
            }
            let t0 = d.time(&model);
            let t = Instant::now();
            for _ in 0..4000 {
                d.step(&model);
            }
            let wall = t.elapsed().as_secs_f64();
            let sim = d.time(&model) - t0;
            println!();
            println!("body: 4000 physics steps in {wall:.3} s wall for {sim:.4} s simulated");
            println!("      = {:.2}x real time, {:.3} ms per physics step", sim / wall, 1e3 * wall / 4000.0);
        }
    }
}
