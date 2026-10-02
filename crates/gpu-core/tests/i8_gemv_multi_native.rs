// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `matmul_i8_gemv_multi` kernel (`kernels_cuda`):
//! up to four int8 matrices that read one activation, multiplied in one launch.
//!
//! Swedish Embedded AB implements bit-exact native kernels for quantised LLM
//! decode. If your team needs expertise in merging a layer's projections into
//! one launch without changing a single output bit, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The claim is the one every substitution here makes: the raw BITS are those
//! of the WGSL tier. The reference is `matmul_i8_gemv_ref`, the same WGSL
//! source under a name the native upgrade table does not know, dispatched once
//! per matrix on the same device from the same inputs. The multi kernel runs
//! the single kernel's own block function on each block's matrix, so identity
//! is achievable; the cases below are the ways the block-to-matrix mapping
//! could still go wrong: a matrix whose rows are not a multiple of the block's
//! eight, a matrix of a single row, a matrix much smaller or larger than its
//! neighbours (the real decode groups mix 10240 rows with 48), every `m` the
//! kernel serves, and fewer than four matrices.

use backend_api::DeviceBuffer;
use gpu_core::{Dispatch, Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv_ref", kernels::MATMUL_I8_GEMV)];
const K_REF: usize = 0;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

fn packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                w |= u32::from((r.next_u32() % 256) as u8) << (8 * lane);
            }
            w
        })
        .collect()
}

fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Run `ns` (one matrix each, up to four) both ways and compare every output bit.
fn check(gpu: &Gpu, m: u32, kg: u32, ns: &[u32], seed: u64) {
    assert!(ns.len() <= 4);
    let ng = (kg / 8) as usize;
    let xq = gpu.storage_init("xq", &packed((m * kg) as usize, seed + 1).iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>());
    let sx = gpu.storage_init("sx", &scales(m as usize, seed + 2));
    let mats: Vec<(DeviceBuffer, DeviceBuffer)> = ns
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let w = packed(n as usize * kg as usize, seed + 10 + i as u64).iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>();
            (gpu.storage_init("wq", &w), gpu.storage_init("sw", &scales(n as usize * ng, seed + 20 + i as u64)))
        })
        .collect();
    let sentinel = f32::from_bits(0x7fc0_dead);
    let outs_ref: Vec<DeviceBuffer> = ns.iter().map(|&n| gpu.storage_init("o", &vec![sentinel; (m * n) as usize])).collect();
    let outs_nat: Vec<DeviceBuffer> = ns.iter().map(|&n| gpu.storage_init("o", &vec![sentinel; (m * n) as usize])).collect();

    // Reference: one WGSL GEMV per matrix.
    let steps: Vec<_> = ns
        .iter()
        .enumerate()
        .map(|(i, &n)| gpu.dispatch(K_REF, &[&xq, &mats[i].0, &sx, &mats[i].1, &outs_ref[i]], &[m, kg, n], Dispatch::Workgroups(n)))
        .collect();
    gpu.submit(&[], &steps);

    // Native: one launch. Unused sets are bound to buffers that are never
    // dereferenced (n = 0 owns no blocks), each its own one-word allocation:
    // the facade refuses a dispatch whose LAST binding appears twice.
    let mut bufs: Vec<&DeviceBuffer> = vec![&xq, &sx];
    let mut params = vec![m, kg];
    let unused: Vec<DeviceBuffer> = (ns.len()..4).map(|_| gpu.storage(1)).collect();
    for i in 0..4 {
        if i < ns.len() {
            bufs.extend([&mats[i].0, &mats[i].1, &outs_nat[i]]);
            params.push(ns[i]);
        } else {
            bufs.extend([&xq, &sx, &unused[i - ns.len()]]);
            params.push(0);
        }
    }
    let step = gpu.fused_step(Fused::I8GemvMulti, &bufs, &params).expect("the multi GEMV was declined on a CUDA device");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();

    for (i, &n) in ns.iter().enumerate() {
        let got = bits(gpu.read(&outs_nat[i], (m * n) as usize));
        let want = bits(gpu.read(&outs_ref[i], (m * n) as usize));
        assert_eq!(got, want, "matrix {i} differs: m={m} kg={kg} ns={ns:?}");
    }
}

#[test]
fn the_multi_gemv_is_offered_only_where_it_can_run() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        assert!(!gpu.has_fused(Fused::I8GemvMulti), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("needs a CUDA device with a packed int8 dot");
        return;
    }
    assert!(gpu.has_fused(Fused::I8GemvMulti));
    assert!(Fused::I8GemvMulti.serves(&[1, 1280, 10240, 48, 48, 6144]));
    assert!(Fused::I8GemvMulti.serves(&[8, 8, 1, 0, 0, 0]));
    assert!(!Fused::I8GemvMulti.serves(&[9, 1280, 64, 0, 0, 0]), "past one tile of x rows");
    assert!(!Fused::I8GemvMulti.serves(&[1, 1284, 64, 0, 0, 0]), "K not a whole number of scale groups");
    assert!(!Fused::I8GemvMulti.serves(&[1, 1280, 0, 0, 0, 0]), "no matrix");
    assert!(!Fused::I8GemvMulti.serves(&[1, 1280, 64]), "short params");
}

#[test]
fn byte_identical_to_separate_launches_over_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("needs a CUDA device with a packed int8 dot");
        return;
    }
    let groups: &[&[u32]] = &[&[64], &[1, 7], &[8, 9, 17], &[129, 1, 33, 64], &[1, 1, 1, 1], &[300, 48, 48, 9], &[16, 16]];
    for (kg, seed) in [(8u32, 1u64), (72, 2), (136, 3), (328, 4), (1280, 5)] {
        for m in [1u32, 2, 3, 8] {
            for ns in groups {
                check(&gpu, m, kg, ns, seed * 100 + u64::from(m));
            }
        }
    }
}

/// The groups a real decode layer multiplies together, at the real widths.
#[test]
fn the_real_27b_projection_groups_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("needs a CUDA device with a packed int8 dot");
        return;
    }
    // K = 5120 (kg = 1280) for the layer inputs, 17408 (kg = 4352) for `down`'s.
    check(&gpu, 1, 1280, &[10240, 48, 48, 6144], 31); // GDN: qkv, b, a, z
    check(&gpu, 1, 1280, &[12288, 1024, 1024], 32); // GQA: q, k, v
    check(&gpu, 1, 1280, &[17408, 17408], 33); // SwiGLU: gate, up
    check(&gpu, 4, 1280, &[10240, 48, 48, 6144], 34); // a batched step
}

#[test]
fn random_groups_are_byte_identical() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) || !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("needs a CUDA device with a packed int8 dot");
        return;
    }
    let mut r = data::rng::Lcg::new(0x0bad_cafe);
    for _ in 0..30 {
        let m = 1 + r.next_u32() % 8;
        let kg = 8 * (1 + r.next_u32() % 60);
        let count = 1 + (r.next_u32() % 4) as usize;
        let ns: Vec<u32> = (0..count).map(|_| 1 + r.next_u32() % 150).collect();
        check(&gpu, m, kg, &ns, u64::from(r.next_u32()));
    }
}
