// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native bf16 paged-KV append (`kv_append_bf16`, four values a thread through 16-byte loads and 8-byte stores) that
//! `gpu_core::native_upgrade` substitutes for the generated `paged_kv_append_batched_word#pool=bf16` on a CUDA device
//! writes exactly the words that kernel writes, to the bit: random rows, the f32 values whose rounding differs between
//! schemes (ties, infinities, NaNs, subnormals, signed zero), scattered block and offset slots, and a pool whose other
//! words keep their sentinel. The reference is the same templated source under a name the upgrade table does not know.
//!
//! Swedish Embedded AB implements bit-exact native replacements for generated kernels. If your team needs expertise
//! in swapping a serial GPU kernel for a native one without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::{Dispatch, Gpu};
use kernels::template::dtype_variant_store;
use backend_api::DType;

/// The native kernel's served name, and the reference's.
const NATIVE_NAME: &str = "paged_kv_append_batched_word#pool=bf16";
/// Written into every pool word first, so a write outside the appended slots shows.
const SENTINEL: u32 = 0x7fc0_beef;
/// Elements one thread of the generated kernel stores (`paged_kv_append_batched_word.wgsl`'s `CHUNK`).
const REFERENCE_CHUNK: u32 = 16;

fn kernels_under_test() -> Vec<(&'static str, &'static str)> {
    let native = dtype_variant_store("paged_kv_append_batched_word", kernels::PAGED_KV_APPEND_BATCHED_WORD, "pool", DType::BF16).expect("pool is a templatable store");
    let reference = dtype_variant_store("paged_kv_append_batched_word_ref", kernels::PAGED_KV_APPEND_BATCHED_WORD, "pool", DType::BF16).expect("pool is a templatable store");
    assert_eq!(native.0, NATIVE_NAME);
    vec![native, reference]
}

/// f32 values that exercise every branch of the round-to-nearest-even bit trick.
fn awkward(u: u32) -> f32 {
    match u % 16 {
        0 => f32::from_bits(0x3f80_8000), // a tie, even below: rounds down
        1 => f32::from_bits(0x3f81_8000), // a tie, odd below: rounds up
        2 => f32::INFINITY,
        3 => f32::NEG_INFINITY,
        4 => f32::NAN,
        5 => f32::from_bits(0xffff_ffff), // the NaN whose bias-add carries out of the word
        6 => f32::from_bits(1),           // the smallest subnormal
        7 => -0.0,
        8 => f32::MAX,
        _ => (u >> 8) as f32 * 1e-5 - 80.0,
    }
}

/// The pool's words after appending `batch` rows of `kv_stride` values at scattered distinct slots.
fn run(gpu: &Gpu, kernel: usize, batch: u32, kv_stride: u32, block_size: u32, seed: u64) -> Vec<u32> {
    let mut rng = data::rng::Lcg::new(seed);
    let src: Vec<f32> = (0..batch * kv_stride).map(|_| awkward(rng.next_u32())).collect();
    let num_blocks = batch + 3;
    let blocks: Vec<u32> = (0..batch).map(|b| (b * 5 + 2) % num_blocks).collect();
    let offsets: Vec<u32> = (0..batch).map(|b| (b * 3 + 1) % block_size).collect();
    let slots: std::collections::HashSet<_> = blocks.iter().zip(&offsets).collect();
    assert_eq!(slots.len(), batch as usize, "test setup: slots must be distinct");
    let pool_words = (num_blocks * block_size * kv_stride / 2) as usize;
    let words = |w: &[u32]| {
        let b = gpu.storage(w.len().max(1) as u64);
        gpu.write(&b, w);
        b
    };
    let (src_b, blocks_b, offsets_b) = (gpu.storage_init("src", &src), words(&blocks), words(&offsets));
    let pool = words(&vec![SENTINEL; pool_words]);
    let threads = batch * kv_stride.div_ceil(REFERENCE_CHUNK);
    let step = gpu.dispatch(kernel, &[&src_b, &blocks_b, &offsets_b, &pool], &[batch, kv_stride, block_size], Dispatch::Threads(threads));
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    gpu.read(&pool, pool_words).iter().map(|f| f.to_bits()).collect()
}

#[test]
fn the_native_append_writes_the_words_the_generated_kernel_writes() {
    let Ok(gpu) = Gpu::try_new_cuda(&kernels_under_test()) else {
        brain_testutil::skip_unavailable("kv_append_bf16_native: no CUDA device");
        return;
    };
    // The substitution must actually have happened, or the comparison below is the generated kernel against itself.
    assert!(gpu.set_kernel_timing(true));
    let _ = run(&gpu, 0, 4, 1024, 16, 1);
    let native_ran = gpu.kernel_times().unwrap_or_default().iter().any(|(name, _, _)| name.contains("kv_append_bf16"));
    gpu.set_kernel_timing(false);
    assert!(native_ran, "the native kv_append_bf16 kernel was not launched for the generated kernel's name");
    // The served width (4 kv heads of 256), narrow rows, a row that is not a multiple of the reference's chunk, and an odd
    // block size: every shape the substitution takes, ragged against the block size.
    for (i, &(batch, kv_stride, block_size)) in [(1u32, 1024u32, 64u32), (7, 1024, 64), (64, 1024, 64), (33, 512, 16), (5, 36, 4), (3, 4, 3), (130, 1024, 7)].iter().enumerate() {
        let want = run(&gpu, 1, batch, kv_stride, block_size, 99 + i as u64);
        let got = run(&gpu, 0, batch, kv_stride, block_size, 99 + i as u64);
        assert_eq!(got, want, "pool words differ: batch {batch} kv_stride {kv_stride} block_size {block_size}");
    }
}

/// A row stride that is not a multiple of four is not the native kernel's shape: the generated kernel keeps it, unchanged.
#[test]
fn a_row_stride_the_native_kernel_does_not_take_keeps_the_generated_kernel() {
    let Ok(gpu) = Gpu::try_new_cuda(&kernels_under_test()) else {
        brain_testutil::skip_unavailable("kv_append_bf16_native: no CUDA device");
        return;
    };
    assert!(gpu.set_kernel_timing(true));
    let got = run(&gpu, 0, 6, 18, 8, 7);
    let native_ran = gpu.kernel_times().unwrap_or_default().iter().any(|(name, _, _)| name.contains("kv_append_bf16"));
    gpu.set_kernel_timing(false);
    assert!(!native_ran, "kv_stride 18 is not a multiple of four; the native kernel must decline");
    assert_eq!(got, run(&gpu, 1, 6, 18, 8, 7));
}

/// Kernel time of the native append against the generated kernel at a decode step (64 rows) and a prefill round (4096
/// rows) of the served width; a report for kernel work, not a spec.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_native_append_rate_against_the_generated_kernel() {
    let Ok(gpu) = Gpu::try_new_cuda(&kernels_under_test()) else {
        return brain_testutil::skip_unavailable("kv_append_bf16_native: no CUDA device");
    };
    assert!(gpu.set_kernel_timing(true));
    for (batch, kv_stride) in [(64u32, 1024u32), (4096, 1024)] {
        let mut us = [0.0f64; 2];
        for (kernel, slot) in [(1usize, 0usize), (0, 1)] {
            let _ = run(&gpu, kernel, batch, kv_stride, 64, 3);
            let mut best = f64::MAX;
            for _ in 0..9 {
                gpu.reset_kernel_times();
                let _ = run(&gpu, kernel, batch, kv_stride, 64, 3);
                let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(n, _, _)| n.contains("kv_append")).map(|(_, ms, _)| ms).sum();
                best = best.min(ms);
            }
            us[slot] = best * 1e3;
        }
        let bytes = f64::from(batch) * f64::from(kv_stride) * 6.0;
        eprintln!("kv_append_bf16: {batch} rows x {kv_stride}: generated {:.1} us, native {:.1} us ({:.0} GB/s of the {:.1} MB moved)", us[0], us[1], bytes / us[1] / 1e3, bytes / 1e6);
    }
    gpu.set_kernel_timing(false);
}
