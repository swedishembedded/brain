// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `qknorm_rope_rows` kernel (`cu/qknorm_rope_rows.cu`): the per-head RMS norm and the half-split
//! rotary embedding of a block of query or key rows, the dense attention's front half after its projection.
//!
//! Swedish Embedded AB implements solutions for long-context prompt processing on GPUs for its clients. If your team needs
//! expertise in cutting the memory passes between an attention layer's projections and its attention then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The reference is the portable kernel it stands in for, `qknorm_rope_fused`, dispatched as `qwen3::serve` dispatches it, on
//! the same device from the same inputs. The native kernel keeps that kernel's 64-lane accumulation order, its left fold of the
//! partials and its per-element arithmetic, so the outputs must be BIT-identical.

use gpu_core::provider::library::{qk_norm_rope, qk_norm_rope_pair, QkNormRopeShape};
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("qknorm_rope_fused", kernels::QKNORM_ROPE_FUSED)];
const FUSED: usize = 0;

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

fn check(gpu: &Gpu, shape: QkNormRopeShape, spread: f32) {
    let QkNormRopeShape { tokens, heads, head_dim, rms_eps, rope_base } = shape;
    let rows = tokens * heads;
    let x = gpu.storage_init("x", &values((rows * head_dim) as usize, 1, spread));
    let w = gpu.storage_init("w", &values(head_dim as usize, 2, 2.0));
    // Positions far apart and repeated, as a decode step's rows and a prefill chunk's are.
    let positions: Vec<u32> = (0..tokens).map(|t| [t, 40_000 - t, t / 3][(t % 3) as usize]).collect();
    let pos = gpu.storage(u64::from(tokens));
    gpu.write(&pos, &positions);

    let want = gpu.storage(u64::from(rows * head_dim));
    gpu.submit(&[], &[gpu.dispatch(FUSED, &[&x, &w, &pos, &want], &[rows, heads, head_dim, rms_eps.to_bits(), rope_base.to_bits()], Dispatch::Workgroups(rows))]);

    let got = gpu.storage_init("got", &vec![f32::NAN; (rows * head_dim) as usize]);
    let mut native = Vec::new();
    assert!(qk_norm_rope(gpu, &mut native, &shape, &x, &w, &pos, &got), "qknorm_rope_rows offered for {shape:?}");
    gpu.submit(&[], &native);
    gpu.poll_wait();

    let n = (rows * head_dim) as usize;
    assert_eq!(bits(gpu.read(&got, n)), bits(gpu.read(&want, n)), "rows differ: {shape:?} spread {spread}");
}

#[test]
fn the_native_rows_equal_the_portable_fused_norm_and_rotation_bit_for_bit() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // Qwen3's query (32 heads) and key (8 heads) of 128, a head count no multiple of the kernel's four heads a block, a
    // single token, and the narrow and wide head widths the kernel serves.
    for (tokens, heads, head_dim) in [(37u32, 32u32, 128u32), (37, 8, 128), (1, 8, 128), (5, 6, 128), (3, 5, 64), (2, 3, 256), (4, 2, 16)] {
        for spread in [1.0f32, 40.0] {
            check(&gpu, QkNormRopeShape { tokens, heads, head_dim, rms_eps: 1e-6, rope_base: 1_000_000.0 }, spread);
        }
    }
}

/// A layer's query and key in one launch: each tensor's bits are the portable kernel's on that tensor.
#[test]
fn the_query_and_key_in_one_launch_equal_two_portable_dispatches_bit_for_bit() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (tokens, hq, hk, hd) in [(37u32, 32u32, 8u32, 128u32), (1, 32, 8, 128), (6, 5, 3, 64)] {
        let shape = QkNormRopeShape { tokens, heads: hq, head_dim: hd, rms_eps: 1e-6, rope_base: 1_000_000.0 };
        let pos = gpu.storage(u64::from(tokens));
        gpu.write(&pos, &(0..tokens).map(|t| 3 * t + 700).collect::<Vec<u32>>());
        let tensor = |heads: u32, seed: u64| {
            let n = (tokens * heads * hd) as usize;
            (gpu.storage_init("x", &values(n, seed, 20.0)), gpu.storage_init("w", &values(hd as usize, seed + 1, 2.0)), gpu.storage_init("out", &vec![f32::NAN; n]), gpu.storage(n as u64), n)
        };
        let (xq, wq, oq, rq, nq) = tensor(hq, 10);
        let (xk, wk, ok, rk, nk) = tensor(hk, 20);
        let fused = |x, w, out, heads: u32| gpu.dispatch(FUSED, &[x, w, &pos, out], &[tokens * heads, heads, hd, 1e-6f32.to_bits(), 1_000_000.0f32.to_bits()], Dispatch::Workgroups(tokens * heads));
        gpu.submit(&[], &[fused(&xq, &wq, &rq, hq), fused(&xk, &wk, &rk, hk)]);
        let mut native = Vec::new();
        assert!(qk_norm_rope_pair(&gpu, &mut native, &shape, hk, [&xq, &wq, &oq], [&xk, &wk, &ok], &pos), "qknorm_rope_qk offered");
        gpu.submit(&[], &native);
        assert_eq!(bits(gpu.read(&oq, nq)), bits(gpu.read(&rq, nq)), "query differs: {shape:?}");
        assert_eq!(bits(gpu.read(&ok, nk)), bits(gpu.read(&rk, nk)), "key differs: {shape:?} {hk} key heads");
    }
}
