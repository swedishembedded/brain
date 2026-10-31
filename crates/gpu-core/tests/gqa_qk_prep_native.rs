// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `gqa_qk_prep` kernel (`cu/gqa_qk_prep.cu`): the front half of a gated attention layer over a block of
//! prompt rows - the query half of the projection's interleaved heads, the QK RMS norm and the partial rotation - in ONE launch.
//!
//! Swedish Embedded AB implements solutions for long-context prompt processing on GPUs for its clients. If your team needs
//! expertise in cutting the memory passes between an attention layer's projections and its attention then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The reference is the chain the kernel replaces, dispatched as `model::gqa_mixer::qkv_prepare` dispatches it: `concat_split`
//! (the query halves), `rmsnorm_rows` for the query and for the key, `rope2d_partial` for both, on the same device from the same
//! inputs. The kernel keeps `rmsnorm_rows`' 64-lane accumulation order, so the outputs must be BIT-identical.

use gpu_core::provider::library::{gqa_qk_prepare, GqaQkShape};
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("concat_split", kernels::CONCAT_SPLIT), ("rmsnorm_rows", kernels::RMSNORM_ROWS), ("rope2d_partial", kernels::ROPE2D_PARTIAL)];
const CONCAT_SPLIT: usize = 0;
const RMS: usize = 1;
const ROPE: usize = 2;

const EPS: f32 = 1e-6;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|i| if i % 211 == 0 { 0.0 } else { ((r.next_u32() % 20001) as f32 / 10000.0 - 1.0) * spread }).collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn check(gpu: &Gpu, shape: GqaQkShape, spread: f32) {
    let GqaQkShape { rows, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: half, rms_eps } = shape;
    let (qd, kvd) = (nh * hd, nkv * hd);
    let q_full = gpu.storage_init("q_full", &values((rows * 2 * qd) as usize, 1, spread));
    let k = gpu.storage_init("k", &values((rows * kvd) as usize, 2, spread));
    let (qw, kw) = (gpu.storage_init("qw", &values(hd as usize, 3, 2.0)), gpu.storage_init("kw", &values(hd as usize, 4, 2.0)));
    let (cos, sin) = (gpu.storage_init("cos", &values((rows * half) as usize, 5, 1.0)), gpu.storage_init("sin", &values((rows * half) as usize, 6, 1.0)));

    // The chain, as `qkv_prepare` runs it.
    let (q_value, q_normed, k_normed) = (gpu.storage(u64::from(rows * qd)), gpu.storage(u64::from(rows * qd)), gpu.storage(u64::from(rows * kvd)));
    let steps = [
        gpu.step(CONCAT_SPLIT, &[&q_full, &q_value], &[rows * nh, 2, 1, 0, 1, hd], rows * nh * hd),
        gpu.dispatch(RMS, &[&q_value, &qw, &q_normed], &[hd, rows * nh, rms_eps.to_bits()], Dispatch::Workgroups(rows * nh)),
        gpu.dispatch(RMS, &[&k, &kw, &k_normed], &[hd, rows * nkv, rms_eps.to_bits()], Dispatch::Workgroups(rows * nkv)),
        gpu.step(ROPE, &[&q_normed, &cos, &sin], &[rows, nh, half, qd, 0, rows, 1.0f32.to_bits(), hd], rows * nh * half),
        gpu.step(ROPE, &[&k_normed, &cos, &sin], &[rows, nkv, half, kvd, 0, rows, 1.0f32.to_bits(), hd], rows * nkv * half),
    ];
    gpu.submit(&[], &steps);

    let mut native = Vec::new();
    let (q_n, k_n) = gqa_qk_prepare(gpu, &mut native, &shape, &q_full, &k, &qw, &kw, &cos, &sin).expect("gqa_qk_prep offered");
    gpu.submit(&[], &native);
    gpu.poll_wait();

    let ctx = format!("{shape:?} spread {spread}");
    assert_eq!(bits(gpu.read(&q_n, (rows * qd) as usize)), bits(gpu.read(&q_normed, (rows * qd) as usize)), "query differs: {ctx}");
    assert_eq!(bits(gpu.read(&k_n, (rows * kvd) as usize)), bits(gpu.read(&k_normed, (rows * kvd) as usize)), "key differs: {ctx}");
}

#[test]
fn the_fused_front_half_equals_the_split_norm_and_rotation_chain_bit_for_bit() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // Qwen3.8-27B's layer (24 query heads, 4 key heads of 256, a rotated block of 64), a block whose head count is no multiple
    // of the kernel's four heads a block, a single row, and a fully rotated head.
    for (rows, nh, nkv, half) in [(1u32, 24u32, 4u32, 32u32), (7, 24, 4, 32), (130, 24, 4, 32), (3, 6, 2, 128), (5, 5, 1, 0)] {
        for spread in [1.0f32, 40.0] {
            check(&gpu, GqaQkShape { rows, n_heads: nh, n_kv_heads: nkv, head_dim: 256, rotary_half: half, rms_eps: EPS }, spread);
        }
    }
}
