// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on the exact numerics they are served with,
// for its clients. If your team needs expertise in quantization-aware reinforcement learning against an inference
// engine then you can procure our services by sending an email to info@swedishembedded.com.

//! `kv_roundtrip_i8` / `kv_roundtrip_fp8` leave a K or V buffer holding exactly what an INT8 / FP8 paged KV cache holds
//! for it: the buffer after the round trip equals the cache's codes read back through their scales, to the bit, where the
//! cache is written by the serving append kernel (`paged_kv_append_*_clipped_batched` with no calibrated ceiling). Random
//! rows with outliers and all-zero rows, several head widths.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("paged_kv_append_i8_clipped_batched", kernels::PAGED_KV_APPEND_I8_CLIPPED_BATCHED),
    ("paged_kv_append_fp8_clipped_batched", kernels::PAGED_KV_APPEND_FP8_CLIPPED_BATCHED),
    ("kv_roundtrip_i8", kernels::KV_ROUNDTRIP_I8),
    ("kv_roundtrip_fp8", kernels::KV_ROUNDTRIP_FP8),
];
const APPEND_I8: usize = 0;
const APPEND_FP8: usize = 1;
const ROUNDTRIP_I8: usize = 2;
const ROUNDTRIP_FP8: usize = 3;

/// The value of an E4M3 byte, decoded from the format's definition (independent of the kernels' own decoder).
fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((code >> 3) & 0xF), i32::from(code & 7));
    sign * if e == 0 { m as f32 * 2f32.powi(-9) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7) }
}

fn rows(batch: u32, kv_stride: u32, hd: u32, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Lcg::new(seed);
    (0..batch * kv_stride)
        .map(|i| {
            let u = rng.next_u32();
            if (i / hd) % 7 == 3 {
                return 0.0;
            }
            let mag = (u % 20_000) as f32 * 1e-4 * if u.is_multiple_of(97) { 40.0 } else { 1.0 };
            if (u >> 20).is_multiple_of(2) { mag } else { -mag }
        })
        .collect()
}

/// What the serving append writes for `src`, read back as the attention reads it: `code * scale` per element.
fn cache_values(gpu: &Gpu, append: usize, fp8: bool, src: &[f32], batch: u32, n_kv: u32, hd: u32) -> Vec<f32> {
    let kv_stride = n_kv * hd;
    let block_size = 16u32;
    let words = |w: &[u32]| {
        let b = gpu.storage(w.len().max(1) as u64);
        gpu.write(&b, w);
        b
    };
    // Token b goes to slot (block b, offset 0).
    let blocks: Vec<u32> = (0..batch).collect();
    let offsets = vec![0u32; batch as usize];
    let pool_words = (batch * block_size * kv_stride / 4) as usize;
    let scale_words = (batch * block_size * n_kv) as usize;
    let (src_b, blocks_b, offsets_b, clip_b) = (gpu.storage_init("src", src), words(&blocks), words(&offsets), gpu.storage_init("clip", &vec![f32::MAX; n_kv as usize]));
    let pool = words(&vec![0u32; pool_words]);
    let scales = words(&vec![0u32; scale_words]);
    let step = gpu.dispatch(append, &[&src_b, &blocks_b, &offsets_b, &clip_b, &pool, &scales], &[batch, kv_stride, block_size, hd], Dispatch::Threads(batch * n_kv));
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let pool_bytes: Vec<u8> = gpu.read(&pool, pool_words).iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
    let scale_f = gpu.read(&scales, scale_words);
    let mut out = Vec::with_capacity(src.len());
    for b in 0..batch as usize {
        let slot = b * block_size as usize;
        for h in 0..n_kv as usize {
            let scale = scale_f[slot * n_kv as usize + h];
            for i in 0..hd as usize {
                let byte = pool_bytes[slot * kv_stride as usize + h * hd as usize + i];
                out.push(if fp8 { e4m3(byte) * scale } else { f32::from(byte as i8) * scale });
            }
        }
    }
    out
}

fn roundtrip(gpu: &Gpu, kernel: usize, src: &[f32], batch: u32, n_kv: u32, hd: u32) -> Vec<f32> {
    let buf = gpu.storage_init("kv", src);
    let step = gpu.dispatch(kernel, &[&buf], &[batch, n_kv * hd, hd, 0], Dispatch::Threads(batch * n_kv));
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    gpu.read(&buf, src.len())
}

#[test]
fn a_round_trip_leaves_the_buffer_holding_what_the_cache_holds() {
    let gpu = Gpu::new_cpu(KERNELS);
    for (n_kv, hd) in [(2u32, 64u32), (8, 128), (4, 32)] {
        let batch = 24;
        let src = rows(batch, n_kv * hd, hd, 7 + u64::from(hd));
        for (append, kernel, fp8) in [(APPEND_I8, ROUNDTRIP_I8, false), (APPEND_FP8, ROUNDTRIP_FP8, true)] {
            let held = cache_values(&gpu, append, fp8, &src, batch, n_kv, hd);
            let rt = roundtrip(&gpu, kernel, &src, batch, n_kv, hd);
            let mismatches = held.iter().zip(&rt).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(mismatches, 0, "{} n_kv {n_kv} hd {hd}: {mismatches} of {} elements differ from the cache's", if fp8 { "fp8" } else { "int8" }, src.len());
            // The cache is lossy: the round trip is not the identity, or the test proves nothing.
            assert!(src.iter().zip(&rt).any(|(a, b)| a != b), "the round trip changed nothing");
        }
    }
}
