// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for GroupNorm's native apply pass (`gn_apply_f32` in
//! `cu/bn_elem_f32.cu`), which `gpu_core::native_upgrade` substitutes for the
//! WGSL `gn_apply` on a CUDA device.
//!
//! Swedish Embedded AB implements normalisation kernels on the memory roofline
//! for its clients. If your team needs expertise in diffusion VAEs that are
//! bound by bandwidth rather than index arithmetic, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-gpu-core --test gn_apply_native -- --device gpu1 --backend cuda
//! ```
//!
//! The pass is elementwise and the native kernel does the same arithmetic per
//! element in the same order, so it is held to the WGSL tier's RAW BITS -
//! including planes whose length is not a whole number of vectors, groups of
//! one and of several channels, batches, and bindings offset by a word (the
//! scalar path).

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("gn_apply", kernels::GN_APPLY), ("gn_apply_ref", kernels::GN_APPLY)];
const K_GN: usize = 0;
const K_REF: usize = 1;
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn values(n: usize, seed: u64, positive: bool) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n)
        .map(|_| {
            let u = (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            (if positive { 0.5 + u } else { 4.0 * u - 2.0 }).to_bits()
        })
        .collect()
}

/// `[N, C, H, W, G]` through both slots, the inputs and output bound at
/// `lead` words into their buffers.
fn check(gpu: &Gpu, p: [u32; 5], lead: u64) {
    let [n, c, h, w, g] = p;
    let total = (n * c * h * w) as usize;
    let seed = u64::from(n * 1000 + c * 10 + h + w + g);
    let window = |words: &[u32]| {
        let b = gpu.storage(words.len() as u64 + lead + 4);
        gpu.write(&b, &vec![SENTINEL; words.len() + lead as usize + 4]);
        gpu.write_at(&b, lead, words);
        b
    };
    let x = window(&values(total, seed, false));
    // stats = [mean, rstd] per (n, group): rstd positive.
    let mut stats = values((2 * n * g) as usize, seed + 1, false);
    for (i, s) in stats.iter_mut().enumerate() {
        if i % 2 == 1 {
            *s = (f32::from_bits(*s).abs() + 0.1).to_bits();
        }
    }
    let stats = window(&stats);
    let gb = window(&values((2 * c) as usize, seed + 2, false));
    let out = |_: ()| window(&vec![SENTINEL; total]);
    let (a, b) = (out(()), out(()));
    let r = |len: usize| (lead, len as u64);
    let ranges = [r(total), r((2 * n * g) as usize), r((2 * c) as usize), r(total)];
    let threads = Dispatch::Threads(total as u32);
    gpu.submit(
        &[],
        &[
            gpu.dispatch_sliced(K_GN, &[&x, &stats, &gb, &a], &ranges, &p, threads),
            gpu.dispatch_sliced(K_REF, &[&x, &stats, &gb, &b], &ranges, &p, threads),
        ],
    );
    gpu.poll_wait();
    let len = total + lead as usize + 4;
    let (ga, gb_) = (gpu.read(&a, len), gpu.read(&b, len));
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    assert!(bits(&ga) == bits(&gb_), "native gn_apply differs from the WGSL tier at {p:?} lead {lead}");
}

fn the_native_kernel_is_selected_on_cuda_only() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_GN, &[1, 32, 8, 8, 32]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(K_GN, &[1, 512, 64, 64, 32]), Some("gn_apply_f32"));
    assert_eq!(gpu.native_kernel_for(K_GN, &[1, 30, 8, 8, 32]), None, "channels not a whole number of groups");
    assert_eq!(gpu.native_kernel_for(K_GN, &[1, 32, 0, 8, 32]), None, "an empty map");
    assert_eq!(gpu.native_kernel_for(K_REF, &[1, 512, 64, 64, 32]), None);
}

fn the_native_kernel_is_bit_identical_to_the_wgsl_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for p in [[1u32, 32, 8, 8, 32], [2, 64, 5, 7, 32], [1, 128, 33, 31, 32], [3, 6, 1, 1, 3], [1, 256, 40, 32, 32], [2, 12, 17, 64, 4]] {
        for lead in [0u64, 1, 4] {
            check(&gpu, p, lead);
        }
    }
}

gpu_core::card_tests!(the_native_kernel_is_selected_on_cuda_only, the_native_kernel_is_bit_identical_to_the_wgsl_tier);
