// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two register-tiled GEMMs must compute the same thing.
//!
//! `Encoder::gemm` picks between a 128x128 and a 64x64 output tile by whether
//! the wider one leaves the card full, so which one runs is a property of the
//! SHAPE. That is a performance decision and it is only allowed to be one:
//! nothing about the arithmetic may change with it, including at the edges,
//! where a tile that does not divide the output is the thing most likely to be
//! wrong and least likely to be noticed.

use gpu_core::Gpu;

/// The four linears of one encoder layer at a real observation length, plus
/// two shapes chosen to land badly on both tilings.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("qkv", 541, 384, 1152),
    ("proj", 541, 384, 384),
    ("fc1", 541, 384, 1536),
    ("fc2", 541, 1536, 384),
    // One row past a 64 boundary and one past a 128 boundary: the ragged
    // edge each kernel has to guard.
    ("ragged-65", 65, 96, 65),
    ("ragged-129", 129, 96, 129),
];

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 4096.0 - 0.3
        })
        .collect()
}

#[test]
fn the_two_tilings_compute_the_same_matmul() {
    let gpu = Gpu::new(decide::kern::PIPELINES);
    let k = decide::kern::Ids::resolve(&gpu);
    for &(name, m, kk, n) in SHAPES {
        let x = gpu.storage_init("x", &rand((m * kk) as usize, 7));
        let w = gpu.storage_init("w", &rand((n * kk) as usize, 11));
        let a = gpu.storage((m * n) as u64);
        let b = gpu.storage((m * n) as u64);
        gpu.submit(
            &[],
            &[
                gpu.dispatch(k.matmul_reg3, &[&x, &w, &a], &[m, kk, n], gpu_core::Dispatch::Workgroups(m.div_ceil(128) * n.div_ceil(128))),
                gpu.dispatch(k.matmul_reg3_64, &[&x, &w, &b], &[m, kk, n], gpu_core::Dispatch::Workgroups(m.div_ceil(64) * n.div_ceil(64))),
            ],
        );
        gpu.poll_wait();
        let n_out = (m * n) as usize;
        let (ga, gb) = (gpu.read(&a, n_out), gpu.read(&b, n_out));
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for (i, (&p, &q)) in ga.iter().zip(&gb).enumerate() {
            let d = (p - q).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        // Both accumulate the same K in the same order, so this is bit-equal
        // in practice; the tolerance is here so a future retiling that
        // reorders the reduction fails loudly rather than by a hair.
        assert!(
            worst < 1e-4,
            "{name} {m}x{kk}x{n}: the tilings disagree by {worst} at element {at} \
             ({} vs {})",
            ga[at],
            gb[at]
        );
    }
}
