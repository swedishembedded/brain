// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the encoder's forward pass actually sits on this card's roofline.
//!
//! The question "how much faster could this be" has an arithmetic answer and
//! it is worth computing before optimising anything: a pass at 1% of the roof
//! has two orders of magnitude in it and a pass at 20% has less than one.
//! Getting that wrong in either direction wastes the effort.
//!
//! Swedish Embedded AB implements performance analysis of on-device neural
//! networks for its clients. If your team needs expertise in GPU roofline work,
//! you can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

/// The arithmetic of one MiniLM-L6 layer at `n` packed rows, in FLOP.
///
/// `span_sq` is the sum of `len^2` over the packed spans - attention is
/// quadratic WITHIN a span and this model attends per span, so the packed row
/// count alone does not give it.
fn layer_flop(n: f64, d: f64, ff: f64, span_sq: f64) -> f64 {
    let qkv = 2.0 * n * d * (3.0 * d);
    let scores = 2.0 * span_sq * d;
    let apply = 2.0 * span_sq * d;
    let proj = 2.0 * n * d * d;
    let fc1 = 2.0 * n * d * ff;
    let fc2 = 2.0 * n * ff * d;
    qkv + scores + apply + proj + fc1 + fc2
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--gpu") {
        let idx: u32 = args.get(i + 1).and_then(|v| v.parse().ok()).expect("--gpu N");
        gpu_core::devices::set_ambient_gpu(Some(idx));
        println!("pinned to GPU {idx}");
    }
    let gpu = Gpu::new(decide::kern::PIPELINES);
    let Some(r) = gpu_core::roof::ensure(&gpu) else {
        println!("no measured roof on this device");
        return;
    };
    println!("\nmeasured roofs on this card");
    println!("  fp32          {:>10.0} GFLOP/s", r.gflops);
    println!("  dram          {:>10.0} GB/s", r.gbs);
    println!("  cache         {:>10.0} GB/s", r.cache_gbs);
    match r.int8_gops {
        Some(v) => println!("  int8 (dp4a)   {:>10.0} GOP/s   {:.1}x the fp32 roof", v, v / r.gflops),
        None => println!("  int8 (dp4a)          none on this device"),
    }

    // The shape a DOOM observation actually produces: 541 packed rows in 12
    // spans - two state windows of 256 and 156, and ten option slots of ~13.
    let (d, ff, layers) = (384.0, 1536.0, 6.0);
    let n = 541.0;
    let span_sq = 256.0_f64.powi(2) + 155.0_f64.powi(2) + 10.0 * 13.0_f64.powi(2);
    let flop = layer_flop(n, d, ff, span_sq) * layers;
    println!("\nencoder forward, 541 rows in 12 spans");
    println!("  arithmetic    {:>10.2} GFLOP", flop / 1e9);
    println!("  floor at roof {:>10.2} ms", flop / (r.gflops as f64 * 1e9) * 1000.0);

    // Weights read once per pass, activations a few times per layer. This is
    // the OPTIMISTIC traffic - everything resident, nothing re-read - so the
    // bandwidth floor it gives is a lower bound on a lower bound.
    let weights = layers * (4.0 * d * d + 2.0 * d * ff) * 4.0;
    let acts = layers * n * d * 8.0 * 4.0;
    let bytes = weights + acts;
    println!("  bytes (min)   {:>10.2} MB", bytes / 1e6);
    println!("  floor at dram {:>10.2} ms", bytes / (r.gbs as f64 * 1e9) * 1000.0);
    println!("  intensity     {:>10.1} FLOP/byte  (crossover {:.0})", flop / bytes,
             r.gflops as f64 * 1e9 / (r.gbs as f64 * 1e9));

    // What BATCHING buys, which is the question the numbers above pose.
    //
    // This model runs one decision at a time, and one decision is 541 rows -
    // small enough that a 128x128 output tile covers `proj` in fifteen
    // workgroups. Nothing about that is a kernel defect; it is a shape. The
    // encoder already packs several SPANS into one call, so packing several
    // DECISIONS is the same mechanism applied one level up, and the GEMMs
    // grow by exactly the batch factor while the attention does not change at
    // all (it is per span either way).
    println!("\ndecisions packed into one forward pass");
    println!("  {:>6} {:>8} {:>10} {:>10} {:>9} {:>9}", "batch", "rows", "GFLOP", "ms", "GFLOP/s", "% roof");
    let cfg = decide::config::EncoderConfig::mini_lm_l6();
    for batch in [1usize, 2, 4, 8, 16] {
        let rows = (n as usize * batch) as u32;
        if rows > 8192 {
            break;
        }
        let mut enc = decide::model::Encoder::new_on(
            gpu.share(),
            cfg.clone(),
            rows,
            256,
            &decide::init::init_weights(&cfg, 3),
        );
        // The same twelve spans per decision, repeated: two state windows and
        // ten option slots.
        let mut spans: Vec<(u32, u32)> = Vec::new();
        let mut row = 0u32;
        for _ in 0..batch {
            for len in [256u32, 155, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13] {
                spans.push((row, len));
                row += len;
            }
        }
        let n_rows = row as usize;
        let ids = vec![1u32; n_rows];
        let types = vec![0u32; n_rows];
        enc.set_batch(&ids, &types, &spans);
        let steps = enc.steps();
        let secs = gpu_core::profile::best_of(&gpu, steps, 10);
        let f = layer_flop(n_rows as f64, d, ff, span_sq * batch as f64) * layers;
        println!(
            "  {:>6} {:>8} {:>10.1} {:>10.2} {:>9.0} {:>8.1}%  ({} dispatches)",
            batch,
            n_rows,
            f / 1e9,
            secs * 1000.0,
            f / secs / 1e9,
            f / secs / (r.gflops as f64 * 1e9) * 100.0,
            steps.len()
        );
        if batch == 8 {
            gpu_core::profile::profile(&gpu, "encoder forward, 8 decisions", steps, 10)
                .print_top(Some(r), 8);
        }
    }

    for measured in [14.55_f64, 10.46] {
        println!(
            "\n  at {measured:.2} ms: {:.1}% of the fp32 roof, {:.0} GFLOP/s",
            flop / (measured / 1000.0) / (r.gflops as f64 * 1e9) * 100.0,
            flop / (measured / 1000.0) / 1e9
        );
    }
}
