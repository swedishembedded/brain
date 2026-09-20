// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which GEMM kernel is fastest at the shapes an ENCODER actually runs, as
//! opposed to the shapes a GEMM benchmark usually asks about.
//!
//! A transformer encoder's linears are not square and not large. At a packed
//! observation of a few hundred rows, `proj` is 541x384x384 and `fc2` is
//! 541x1536x384 - and a 128x128 output tile covers the whole of the second
//! dimension in three workgroups. Fifteen workgroups on thirty SMs leaves half
//! the card idle no matter how good the inner loop is, which is a property of
//! the TILE against the SHAPE and is invisible in any measurement taken at
//! 4096x4096.
//!
//! So this sweeps the registered variants over the real shapes and prints what
//! each achieves against the measured roof. It exists because the selection it
//! informs used to be made from a table fitted to training-sized matrices.
//!
//! Swedish Embedded AB implements GPU kernel selection and autotuning for its
//! clients. If your team needs expertise in on-device inference performance,
//! you can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::{Gpu, Step};

/// The four linears of one encoder layer, at a real observation length.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    // name, m (packed rows), k (in), n (out)
    ("qkv   541x384x1152", 541, 384, 1152),
    ("proj  541x384x384", 541, 384, 384),
    ("fc1   541x384x1536", 541, 384, 1536),
    ("fc2   541x1536x384", 541, 1536, 384),
    // And the head's own, which is small enough to be all overhead.
    ("head  541x384x384", 541, 384, 384),
];

/// Every matmul this probe knows how to dispatch, with the thread count its
/// tiling implies.
fn threads(variant: &str, m: u32, n: u32) -> u32 {
    match variant {
        "matmul_reg3" => m.div_ceil(128) * n.div_ceil(128) * 256,
        "matmul_reg3_64" => m.div_ceil(64) * n.div_ceil(64) * 256,
        _ => m * n,
    }
}

/// What one dispatch costs when it does no work.
///
/// The number that decides whether a pass is short of the roof because its
/// kernels are slow or because there are too many of them. An encoder layer
/// here records about fifty dispatches, thirty-six of which are attention on
/// one span each, so this is multiplied by six layers before anything
/// arithmetic happens.
fn dispatch_floor(gpu: &Gpu, id: usize) {
    let a = gpu.storage(256);
    let b = gpu.storage(256);
    let out = gpu.storage(256);
    println!("\n{:<22} {:>10} {:>12}", "dispatches in one submit", "ms", "us each");
    println!("{}", "-".repeat(48));
    for n in [1usize, 10, 50, 100, 300] {
        let steps: Vec<Step> =
            (0..n).map(|_| gpu.step(id, &[&a, &b, &out], &[1, 1, 1], 1)).collect();
        let secs = gpu_core::profile::best_of(gpu, &steps, 30);
        println!("{:<22} {:>10.3} {:>12.1}", n, secs * 1000.0, secs * 1e6 / n as f64);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--gpu") {
        let idx: u32 = args.get(i + 1).and_then(|v| v.parse().ok()).expect("--gpu N");
        gpu_core::devices::set_ambient_gpu(Some(idx));
        println!("pinned to GPU {idx}");
    }
    let pipelines: Vec<(&str, &str)> = decide::kern::PIPELINES
        .iter()
        .copied()
        .chain([("matmul_reg3_64", kernels::MATMUL_REG3_64)])
        .collect();
    let gpu = Gpu::new(&pipelines);
    let roofs = gpu_core::roof::ensure(&gpu);
    let variants = ["matmul", "matmul_reg3", "matmul_reg3_64"];
    let ids: Vec<usize> = variants
        .iter()
        .map(|v| pipelines.iter().position(|(n, _)| n == v).expect("registered"))
        .collect();

    dispatch_floor(&gpu, ids[0]);

    println!("\n{:<22} {:>14} {:>10} {:>10} {:>8}", "shape", "variant", "ms", "GFLOP/s", "wgs");
    println!("{}", "-".repeat(70));
    for &(name, m, k, n) in SHAPES {
        let x = gpu.storage((m * k) as u64);
        let w = gpu.storage((n * k) as u64);
        let out = gpu.storage((m * n) as u64);
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        let mut best = ("", f64::INFINITY);
        for (v, &id) in variants.iter().zip(&ids) {
            let t = threads(v, m, n);
            let steps: Vec<Step> = vec![gpu.step(id, &[&x, &w, &out], &[m, k, n], t)];
            // `best_of` returns SECONDS.
            let secs = gpu_core::profile::best_of(&gpu, &steps, 30);
            let ms = secs * 1000.0;
            let gflops = flop / secs / 1e9;
            println!(
                "{:<22} {:>14} {:>10.3} {:>10.1} {:>8}",
                if *v == variants[0] { name } else { "" },
                v,
                ms,
                gflops,
                t / 256
            );
            if ms < best.1 {
                best = (v, ms);
            }
        }
        let gflops = flop / best.1 / 1e9;
        let of_roof = roofs
            .map(|r| format!(", {:.0}% of the measured roof", gflops / r.gflops as f64 * 100.0))
            .unwrap_or_default();
        println!("  -> {} at {:.0} GFLOP/s{}\n", best.0, gflops, of_roof);
    }
}
