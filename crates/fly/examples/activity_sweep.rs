// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Pick the weight scale by measurement: which values leave the cord silent,
//! which make it seize, and which sit in between.
use neuro::{DynamicalSystem, Port, SpikingNet};
fn main() {
    let (neurons, edges) = connectome::find(std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap()), "manc").unwrap();
    let c = connectome::load("manc", &neurons, &edges).unwrap();
    let n = c.neurons.len();
    let desc = c.population(|x| x.super_class == "descending");
    let motor = c.population(|x| x.super_class == "motor");
    println!("{:>8} {:>6} {:>12} {:>12} {:>10}", "scale", "cmd", "spikes/tick", "% of cord", "motor/tick");
    for &scale in &[1e-3f32, 3e-3, 1e-2, 3e-2, 1e-1] {
        for &cmd in &[2.0f32, 5.0] {
            let csc = c.signed_csc(scale);
            let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
            let lif = fly::cord_lif();
            let mut net = SpikingNet::new(gpu, &csc, lif).unwrap();
            let mut drive = vec![0.0f32; n];
            for &d in &desc { drive[d as usize] = cmd; }
            net.drive(Port::Drive, &drive).unwrap();
            let mut spike = vec![0.0f32; n];
            let (mut tot, mut mot) = (0u64, 0u64);
            let ticks = 100;
            for _ in 0..ticks {
                net.step();
                net.read(Port::Spike, &mut spike).unwrap();
                tot += spike.iter().filter(|&&s| s > 0.5).count() as u64;
                mot += motor.iter().filter(|&&m| spike[m as usize] > 0.5).count() as u64;
            }
            println!("{:>8.0e} {:>6.1} {:>12.1} {:>11.2}% {:>10.1}",
                scale, cmd, tot as f64/ticks as f64, 100.0*tot as f64/(ticks as f64*n as f64), mot as f64/ticks as f64);
        }
    }
}
