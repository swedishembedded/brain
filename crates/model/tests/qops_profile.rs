// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training over quantized
// checkpoints for its clients. If your team needs expertise in profiling
// quantized GEMM paths against the dense tensor-core ceiling, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Kernel time of the quantized-execution contract against the dense path, per `Weight` tier: forward,
//! input gradient and weight gradient at one training shape. `#[ignore]`d: run on a GPU through the card
//! helper (`cargo test -p brain-model --test qops_profile -- --ignored --nocapture`); the table it prints
//! goes to `runs/qops/profile-*.md`. Each product is repeated `REPS` times in one submission, the best of twelve
//! submissions is reported, and each is timed by
//! the wall clock around submit plus a one-value read (the read is the device synchronisation).

use std::time::Instant;

use data::rng::Lcg;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::fp8::{quantize_weight_fp8, Fp8Format};
use model::int8::quantize_weight;
use model::ops::{training_kernel_list, GradDtype, Ops, Weight, FP8_BLOCK};

const REPS: usize = 8;

fn shape() -> (usize, usize, usize) {
    // tokens, output rows, reduction width: one MLP projection of a 4096-wide model.
    (4096, 12288, 4096)
}

fn weights(ops: &Ops, n: usize, k: usize) -> Vec<(&'static str, Weight)> {
    let g = ops.gpu();
    let raw = Lcg::random().vec_scaled(n * k, 0.5);
    let up = |words: &[u32]| {
        let b = g.storage(words.len() as u64);
        g.write(&b, words);
        b
    };
    let (i8p, i8s) = quantize_weight(&raw, n, k);
    let (f8p, f8s) = quantize_weight_fp8(&raw, n, k, Fp8Format::E4M3, FP8_BLOCK as usize);
    vec![
        ("f32", Weight::upload(ops, &raw, n, k, Dtype::F32)),
        ("bf16", Weight::upload(ops, &raw, n, k, Dtype::BF16)),
        ("nf4", Weight::upload(ops, &raw, n, k, Dtype::NF4)),
        ("i8", Weight::I8 { w: up(&i8p), s: g.storage_init("s", &i8s), n: n as u32, k: k as u32 }),
        (
            "fp8_e4m3",
            Weight::FP8 { w: up(&f8p), s: g.storage_init("s", &f8s), n: n as u32, k: k as u32, fmt: Fp8Format::E4M3, block: FP8_BLOCK },
        ),
    ]
}

fn time_ms(ops: &Ops, mut build: impl FnMut(&mut Vec<gpu_core::Step>), probe: &gpu_core::DeviceBuffer) -> f64 {
    let g = ops.gpu();
    let mut run = |reps: usize| {
        let mut steps = Vec::new();
        for _ in 0..reps {
            build(&mut steps);
        }
        let t = Instant::now();
        g.submit(&[], &steps);
        let _ = g.read(probe, 1);
        t.elapsed().as_secs_f64() * 1e3
    };
    run(1); // warm-up: kernel compilation, library handles
    // The card is shared with other jobs: report the best of twelve timed submissions.
    (0..12).map(|_| run(REPS) / REPS as f64).fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "GPU profile: run through the card helper"]
fn kernel_time_of_every_tier_against_the_dense_path() {
    let (m, n, k) = shape();
    let mut ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Auto);
    let g = ops.gpu();
    println!("device {} gemm path {:?}", g.kind(), ops.gemm_path());
    let mut rng = Lcg::random();
    let x = g.storage_init("x", &rng.vec_scaled(m * k, 1.0));
    let dy = g.storage_init("dy", &rng.vec_scaled(m * n, 1.0));
    let y = g.storage((m * n) as u64);
    let dx = g.storage((m * k) as u64);
    let dw32 = g.storage((n * k) as u64);
    let dw16 = g.storage((n * k / 2) as u64);
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!("shape m={m} n={n} k={k}; tile budget {} MiB; one product = {:.1} GFLOP", ops.tile_budget_bytes() >> 20, flops / 1e9);
    // The weight gradient reads no weight: one measurement serves every tier.
    let dwf = time_ms(&ops, |s| ops.weight_grad(s, &Weight::F32 { w: x.clone(), n: n as u32, k: k as u32 }, &x, &dy, m as u32, &dw32, GradDtype::F32), &dw32);
    let dwb = time_ms(&ops, |s| ops.weight_grad(s, &Weight::F32 { w: x.clone(), n: n as u32, k: k as u32 }, &x, &dy, m as u32, &dw16, GradDtype::Bf16), &dw16);
    println!("weight gradient (all tiers): fp32 {dwf:.2} ms ({:.0} TFLOP/s), bf16 {dwb:.2} ms", flops / dwf / 1e9);
    println!("| tier | forward ms | dX ms | forward TFLOP/s | dX TFLOP/s | decode f32 tile ms | decode bf16 tile ms |");
    println!("|---|---|---|---|---|---|---|");
    let tile_rows = (ops.tile_budget_bytes() / (4 * k as u64)).min(n as u64) as u32;
    let tile = g.storage((tile_rows as usize * k) as u64);
    for (label, w) in weights(&ops, n, k) {
        let act = ops.act_f32(&x, 0, m as u32, k as u32);
        let fwd = time_ms(&ops, |s| ops.matmul_rows(s, &w, (0, n as u32), &act, &y, 0), &y);
        let bwd = time_ms(&ops, |s| ops.matmul_dx(s, &w, &dy, m as u32, &dx, false), &dx);
        let (dec32, dec16) = if w.decode_format().is_some() {
            let f32_ms = time_ms(&ops, |s| ops.decode_tile(s, &w, 0, tile_rows, &tile), &tile);
            let bf16_ms = time_ms(
                &ops,
                |s| {
                    assert!(ops.decode_tile_bf16(s, &w, 0, tile_rows, &tile), "the bf16 decode was declined");
                },
                &tile,
            );
            (format!("{f32_ms:.3}"), format!("{bf16_ms:.3}"))
        } else {
            ("-".to_string(), "-".to_string())
        };
        println!("| {label} | {fwd:.2} | {bwd:.2} | {:.0} | {:.0} | {dec32} | {dec16} |", flops / fwd / 1e9, flops / bwd / 1e9);
    }
}
