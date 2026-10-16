// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native cross-entropy statistics kernel against its WGSL reference on a
//! CUDA device: the row maxima must be identical and the sums equal to the fp32
//! summation bound, over a real vocabulary width, a width that is not a multiple
//! of the block, ignored rows, and rows whose logits span many binades.
//!
//! Swedish Embedded AB implements large-vocabulary language-model training for
//! its clients. If your team needs expertise in cross-entropy over hundreds of
//! thousands of classes without materialising the softmax, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("ce_stats", kernels::CE_STATS), ("ce_stats_ref", kernels::CE_STATS)];
const NATIVE: usize = 0;
const REFERENCE: usize = 1;
const IGNORE: u32 = u32::MAX;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.signed() * 10f32.powi((r.next_u32() % 3) as i32)).collect()
}

fn the_native_ce_stats_matches_the_reference_to_the_summation_bound() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(NATIVE, &[4, 1000, IGNORE]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(NATIVE, &[4, 1000, IGNORE]), Some("ce_stats_f32"));
    for (i, &(rows, bins)) in [(1024u32, 248_320u32), (7, 1000), (33, 4097), (1, 1)].iter().enumerate() {
        let x = logits((rows * bins) as usize, 40 + i as u64);
        let targets: Vec<u32> = (0..rows).map(|r| if r % 5 == 3 { IGNORE } else { r % bins }).collect();
        let (xb, tb) = (gpu.storage_init("x", &x), gpu.storage(rows as u64));
        gpu.write(&tb, &targets);
        let outs = [gpu.storage(2 * rows as u64), gpu.storage(2 * rows as u64)];
        for (slot, out) in [NATIVE, REFERENCE].iter().zip(&outs) {
            gpu.submit(&[], &[gpu.step(*slot, &[&xb, &tb, out], &[rows, bins, IGNORE], rows)]);
        }
        gpu.poll_wait();
        let (got, want) = (gpu.read(&outs[0], 2 * rows as usize), gpu.read(&outs[1], 2 * rows as usize));
        for r in 0..rows as usize {
            assert_eq!(got[2 * r].to_bits(), want[2 * r].to_bits(), "row {r} of {rows}x{bins}: the maximum differs");
            let (g, w) = (got[2 * r + 1], want[2 * r + 1]);
            // Serial fp32 summation over `bins` terms carries a relative error of about bins * eps / 2.
            let bound = (bins as f32 * f32::EPSILON).max(1e-6) * w.abs();
            assert!((g - w).abs() <= bound, "row {r} of {rows}x{bins}: sum {g} vs {w}");
        }
    }
}

gpu_core::card_tests!(
    the_native_ce_stats_matches_the_reference_to_the_summation_bound,
);
