// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native int8 paged-KV append (`kv_append_i8_clipped`, one warp per token and kv head) that
//! `gpu_core::native_upgrade` substitutes for the WGSL `paged_kv_append_i8_clipped_batched` on a CUDA device writes
//! exactly the bytes and scales the WGSL tier writes, to the bit, over random rows (outliers, zero rows, a calibrated
//! clip and the "no calibration" f32::MAX sentinel), scattered block/offset slots and several head widths; the
//! reference is the same WGSL source under a name the upgrade table does not know.
//!
//! Swedish Embedded AB implements bit-exact native replacements for generated kernels. If your team needs expertise
//! in swapping a serial GPU kernel for a hand-written one without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("paged_kv_append_i8_clipped_batched", kernels::PAGED_KV_APPEND_I8_CLIPPED_BATCHED),
    ("paged_kv_append_i8_clipped_ref", kernels::PAGED_KV_APPEND_I8_CLIPPED_BATCHED),
];
const NATIVE: usize = 0;
const REFERENCE: usize = 1;
/// Written into every pool word and scale first, so a write outside the appended slots shows.
const SENTINEL: u32 = 0x7fc0_beef;

fn run(gpu: &Gpu, kernel: usize, batch: u32, n_kv: u32, hd: u32, block_size: u32, seed: u64, clip: &[f32]) -> (Vec<u32>, Vec<u32>) {
    let kv_stride = n_kv * hd;
    let mut rng = data::rng::Lcg::new(seed);
    let src: Vec<f32> = (0..batch * kv_stride)
        .map(|i| {
            let u = rng.next_u32();
            let row_zero = (i / hd) % 7 == 3;
            if row_zero {
                0.0
            } else {
                let mag = (u % 20_000) as f32 * 1e-4 * if u % 97 == 0 { 40.0 } else { 1.0 };
                if (u >> 20) % 2 == 0 { mag } else { -mag }
            }
        })
        .collect();
    // Distinct slots, scattered over a pool of more blocks than the batch needs.
    let num_blocks = batch + 3;
    let blocks: Vec<u32> = (0..batch).map(|b| (b * 5 + 2) % num_blocks).collect();
    let offsets: Vec<u32> = (0..batch).map(|b| (b * 3 + 1) % block_size).collect();
    // Two tokens may share a block but never a slot.
    let slots: std::collections::HashSet<_> = blocks.iter().zip(&offsets).collect();
    assert_eq!(slots.len(), batch as usize, "test setup: slots must be distinct");
    let pool_words = (num_blocks * block_size * kv_stride / 4) as usize;
    let scale_words = (num_blocks * block_size * n_kv) as usize;

    let words = |w: &[u32]| {
        let b = gpu.storage(w.len().max(1) as u64);
        gpu.write(&b, w);
        b
    };
    let (src_b, blocks_b, offsets_b, clip_b) = (gpu.storage_init("src", &src), words(&blocks), words(&offsets), gpu.storage_init("clip", clip));
    let pool = words(&vec![SENTINEL; pool_words]);
    let scales = words(&vec![SENTINEL; scale_words]);
    let step = gpu.dispatch(kernel, &[&src_b, &blocks_b, &offsets_b, &clip_b, &pool, &scales], &[batch, kv_stride, block_size, hd], Dispatch::Threads(batch * n_kv));
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let bits = |b: &backend_api::DeviceBuffer, n: usize| gpu.read(b, n).iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    (bits(&pool, pool_words), bits(&scales, scale_words))
}

#[test]
fn the_native_append_writes_the_bits_the_wgsl_tier_writes() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        brain_testutil::skip_unavailable("kv_append_i8_native: no CUDA device");
        return;
    };
    // The substitution must actually have happened, or the comparison below is the WGSL kernel against itself.
    assert!(gpu.set_kernel_timing(true));
    let _ = run(&gpu, NATIVE, 4, 8, 128, 16, 1, &[f32::MAX; 8]);
    let native_ran = gpu.kernel_times().unwrap_or_default().iter().any(|(name, _, _)| name.contains("kv_append_i8_clipped"));
    gpu.set_kernel_timing(false);
    assert!(native_ran, "the native kv_append_i8_clipped kernel was not launched for the WGSL kernel's name");
    for (i, &(batch, n_kv, hd, block_size)) in [(1u32, 8u32, 128u32, 16u32), (7, 8, 128, 16), (32, 8, 128, 16), (5, 4, 64, 4), (33, 2, 96, 8), (3, 1, 4, 2)].iter().enumerate() {
        for clip_kind in 0..2 {
            let clip: Vec<f32> = (0..n_kv).map(|h| if clip_kind == 0 { f32::MAX } else { 3.0 + h as f32 }).collect();
            let want = run(&gpu, REFERENCE, batch, n_kv, hd, block_size, 99 + i as u64, &clip);
            let got = run(&gpu, NATIVE, batch, n_kv, hd, block_size, 99 + i as u64, &clip);
            assert_eq!(got.1, want.1, "scales differ: batch {batch} n_kv {n_kv} hd {hd} clip {clip_kind}");
            assert_eq!(got.0, want.0, "pool bytes differ: batch {batch} n_kv {n_kv} hd {hd} clip {clip_kind}");
        }
    }
}
