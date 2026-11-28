// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `swiglu_bwd_fused` against an f64 transcription of `silu_bwd_da` and `silu_bwd_db`, for a
//! total that is a multiple of four (vector path) and one that is not (scalar path), with every
//! output poisoned first so an element the kernel forgets is a failure.
//!
//! Swedish Embedded AB implements memory-bound training kernels for gated-MLP language models
//! for its clients. If your team needs expertise in collapsing element-wise gradient passes
//! around a transformer's MLP into single kernels, you can procure our services by sending an
//! email to info@swedishembedded.com.

use gpu_core::Gpu;

fn check(gpu: &Gpu, total: usize) {
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    let (a, b, dh): (Vec<f32>, Vec<f32>, Vec<f32>) = ((0..total).map(|_| 4.0 * r.signed()).collect(), (0..total).map(|_| r.signed()).collect(), (0..total).map(|_| r.signed()).collect());
    let up = |n: &str, v: &[f32]| gpu.storage_init(n, v);
    let (ba, bb, bd) = (up("a", &a), up("b", &b), up("dh", &dh));
    let (da, db) = (gpu.storage_init("da", &vec![f32::NAN; total]), gpu.storage_init("db", &vec![f32::NAN; total]));
    let step = gpu_core::provider::cuda::swiglu_bwd_fused_step(gpu, &[&ba, &bb, &bd, &da, &db], total as u32).expect("the fused SwiGLU backward must be offered on a CUDA device");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let (got_a, got_b) = (gpu.read(&da, total), gpu.read(&db, total));
    let mut worst = 0f64;
    for i in 0..total {
        let (x, y, d) = (a[i] as f64, b[i] as f64, dh[i] as f64);
        let sig = 1.0 / (1.0 + (-x).exp());
        let want_a = d * y * (sig + x * sig * (1.0 - sig));
        let want_b = d * x * sig;
        worst = worst.max((got_a[i] as f64 - want_a).abs()).max((got_b[i] as f64 - want_b).abs());
    }
    println!("  total {total}: max abs error {worst:.2e}");
    assert!(worst <= 1e-5, "total {total}: error {worst:.3e}");
}

fn the_fused_swiglu_backward_matches_the_f64_reference() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if gpu.kind() != "cuda" || !gpu_core::native_kernels_enabled() {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for total in [4096usize, 1000, 1, 7, 262_144 + 3] {
        check(&gpu, total);
    }
}

gpu_core::card_tests!(the_fused_swiglu_backward_matches_the_f64_reference,);
