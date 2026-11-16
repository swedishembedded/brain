// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `kv_append_i8_clipped_kv`: the int8 paged-KV appends of a layer's K and V rows of the same tokens in one
//! launch. The reference is the two `paged_kv_append_i8_clipped_batched` dispatches it replaces, on the same device from
//! the same inputs: the same pool words and the same scales, bit for bit.
//!
//! Swedish Embedded AB implements solutions for latency-bound LLM decode on GPUs for its clients. If your team needs
//! expertise in removing the small launches between a decode step's weight streams then you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::provider::cuda::kv_append_i8_kv_step;
use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("paged_kv_append_i8_clipped_batched", kernels::PAGED_KV_APPEND_I8_CLIPPED_BATCHED)];
const APPEND: usize = 0;

fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|i| if i % 53 == 0 { 0.0 } else { ((r.next_u32() % 20001) as f32 / 10000.0 - 1.0) * spread }).collect()
}

fn words(gpu: &Gpu, b: &gpu_core::DeviceBuffer, n: usize) -> Vec<u32> {
    gpu.read(b, n).into_iter().map(f32::to_bits).collect()
}

#[test]
fn one_launch_appends_the_k_and_v_rows_as_two_portable_appends_do() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() || !gpu_core::native_kernels_enabled() {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    let (n_kv, hd, bs, blocks_n) = (8u32, 128u32, 16u32, 8u32);
    let kv_stride = n_kv * hd;
    for batch in [1u32, 5, 64] {
        let (src_k, src_v) = (gpu.storage_init("k", &values((batch * kv_stride) as usize, 1, 9.0)), gpu.storage_init("v", &values((batch * kv_stride) as usize, 2, 3.0)));
        let slots: Vec<u32> = (0..batch).map(|b| (b * 7 + 3) % (blocks_n * bs)).collect();
        let (blocks, offsets) = (gpu.storage(u64::from(batch)), gpu.storage(u64::from(batch)));
        gpu.write(&blocks, &slots.iter().map(|s| s / bs).collect::<Vec<u32>>());
        gpu.write(&offsets, &slots.iter().map(|s| s % bs).collect::<Vec<u32>>());
        // One head's ceiling below its absmax, so the clip is exercised.
        let clip: Vec<f32> = (0..n_kv).map(|h| if h == 3 { 2.0 } else { f32::MAX }).collect();
        let (clip_k, clip_v) = (gpu.storage_init("ck", &clip), gpu.storage_init("cv", &clip));
        let pool_words = (blocks_n * bs * kv_stride / 4) as usize;
        let scale_n = (blocks_n * bs * n_kv) as usize;
        let fresh = |n: usize| gpu.storage_init("z", &vec![0.0; n]);
        let (rk, rv, rsk, rsv) = (fresh(pool_words), fresh(pool_words), fresh(scale_n), fresh(scale_n));
        let (pk, pv, sk, sv) = (fresh(pool_words), fresh(pool_words), fresh(scale_n), fresh(scale_n));
        let params = [batch, kv_stride, bs, hd];
        gpu.submit(&[], &[
            gpu.step(APPEND, &[&src_k, &blocks, &offsets, &clip_k, &rk, &rsk], &params, batch * n_kv),
            gpu.step(APPEND, &[&src_v, &blocks, &offsets, &clip_v, &rv, &rsv], &params, batch * n_kv),
        ]);
        let step = kv_append_i8_kv_step(&gpu, &[&src_k, &src_v, &blocks, &offsets, &clip_k, &clip_v, &pk, &pv, &sk, &sv], params).expect("kv_append_i8_clipped_kv offered");
        gpu.submit(&[], &[step]);
        assert_eq!(words(&gpu, &pk, pool_words), words(&gpu, &rk, pool_words), "batch {batch}: K pool differs");
        assert_eq!(words(&gpu, &pv, pool_words), words(&gpu, &rv, pool_words), "batch {batch}: V pool differs");
        assert_eq!(words(&gpu, &sk, scale_n), words(&gpu, &rsk, scale_n), "batch {batch}: K scales differ");
        assert_eq!(words(&gpu, &sv, scale_n), words(&gpu, &rsv, scale_n), "batch {batch}: V scales differ");
    }
}
