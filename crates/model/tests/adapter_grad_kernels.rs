// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that the GEMM kernels
// behind a training stack's backward passes agree with their reference on the
// small, unaligned shapes adapters use, for its clients. If your team needs
// expertise in numerical kernel verification then you can procure our services
// by sending an email to info@swedishembedded.com.

//! Parity of the register-tiled weight and input gradient GEMMs with the
//! reference kernels at the shapes an adapter's backward dispatches: a
//! rank-wide contraction or output (1 to 64), a model-wide other side, row
//! counts that are not tile multiples, with and without accumulation. The
//! tiled kernels write the same `dW[n,k] += sum_m dY[m,n] X[m,k]` and
//! `dX[m,k] (+)= sum_n dY[m,n] W[n,k]`; a difference here is a kernel defect.

use gpu_core::Dispatch;

const KERNELS: &[(&str, &str)] = &[
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("grad_scale", kernels::GRAD_SCALE),
];
const DX: usize = 0;
const DX_REG: usize = 1;
const DW: usize = 2;
const DW_REG: usize = 3;
const GRAD_SCALE: usize = 4;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.scaled(1.0)).collect()
}

fn tiles(rows: u32, cols: u32) -> Dispatch {
    Dispatch::Workgroups(rows.div_ceil(128) * cols.div_ceil(128))
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn the_tiled_input_gradient_matches_the_reference_at_adapter_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for m in [1u32, 6, 12, 64, 130] {
        for k in [96u32, 128, 192, 300] {
            for n in [1u32, 2, 3, 5, 16, 64] {
                for acc in [0u32, 1] {
                    let seed = (m * 7919 + k * 31 + n) as u64;
                    let (dy, w) = (gpu.storage_init("dy", &values((m * n) as usize, seed)), gpu.storage_init("w", &values((n * k) as usize, seed + 1)));
                    let init = values((m * k) as usize, seed + 2);
                    let (a, b) = (gpu.storage_init("a", &init), gpu.storage_init("b", &init));
                    let params = [m, k, n, acc];
                    gpu.submit(&[], &[gpu.dispatch(DX_REG, &[&dy, &w, &a], &params, tiles(m, k)), gpu.dispatch(DX, &[&dy, &w, &b], &params, Dispatch::Threads(m * k))]);
                    gpu.poll_wait();
                    let (ga, gb) = (gpu.read(&a, (m * k) as usize), gpu.read(&b, (m * k) as usize));
                    let rel = max_rel(&ga, &gb);
                    assert!(rel < 1e-4, "dX tiled vs reference at m={m} k={k} n={n} accumulate={acc}: relative difference {rel:e}");
                }
            }
        }
    }
}

#[test]
fn the_tiled_weight_gradient_matches_the_reference_at_adapter_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for m in [1u32, 6, 12, 64, 130] {
        for k in [96u32, 128, 192, 300] {
            for n in [1u32, 2, 3, 5, 16, 64, 130] {
                let seed = (m * 7919 + k * 31 + n) as u64;
                let (dy, x) = (gpu.storage_init("dy", &values((m * n) as usize, seed)), gpu.storage_init("x", &values((m * k) as usize, seed + 1)));
                let init = values((n * k) as usize, seed + 2);
                let (a, b) = (gpu.storage_init("a", &init), gpu.storage_init("b", &init));
                let params = [m, k, n];
                gpu.submit(&[], &[gpu.dispatch(DW_REG, &[&dy, &x, &a], &params, tiles(n, k)), gpu.dispatch(DW, &[&dy, &x, &b], &params, Dispatch::Threads(n * k))]);
                gpu.poll_wait();
                let (ga, gb) = (gpu.read(&a, (n * k) as usize), gpu.read(&b, (n * k) as usize));
                let rel = max_rel(&ga, &gb);
                assert!(rel < 1e-4, "dW tiled vs reference at m={m} k={k} n={n}: relative difference {rel:e}");
            }
        }
    }
}

/// The adapter's backward feeds a tiled GEMM with the result of the steps just
/// before it in the same list (`dmid = scale * dY B`, then `dX += dmid A`): the
/// tiled kernel must see that result, not an earlier value of the buffer.
#[test]
fn a_tiled_gemm_reads_what_the_steps_before_it_wrote() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (m, k, n, r) = (12u32, 128u32, 40u32, 3u32);
    let run = |tiled: bool| {
        let dy = gpu.storage_init("dy", &values((m * n) as usize, 1));
        let b = gpu.storage_init("b", &values((n * r) as usize, 2));
        let a = gpu.storage_init("a", &values((r * k) as usize, 3));
        let dmid = gpu.storage(u64::from(m * r));
        let dx = gpu.storage_init("dx", &values((m * k) as usize, 4));
        // The base's input gradient, written into `dx` first.
        let (bdy, bw) = (gpu.storage_init("bdy", &values((m * n) as usize, 5)), gpu.storage_init("bw", &values((n * k) as usize, 6)));
        let last = if tiled {
            gpu.dispatch(DX_REG, &[&dmid, &a, &dx], &[m, k, r, 1], tiles(m, k))
        } else {
            gpu.dispatch(DX, &[&dmid, &a, &dx], &[m, k, r, 1], Dispatch::Threads(m * k))
        };
        gpu.submit(
            &[],
            &[gpu.step(DX, &[&bdy, &bw, &dx], &[m, k, n, 0], m * k), gpu.step(DX, &[&dy, &b, &dmid], &[m, r, n, 0], m * r), gpu.step(GRAD_SCALE, &[&dmid], &[m * r, 2.0f32.to_bits()], m * r), last],
        );
        gpu.poll_wait();
        gpu.read(&dx, (m * k) as usize)
    };
    let rel = max_rel(&run(true), &run(false));
    assert!(rel < 1e-4, "a tiled GEMM after two dependent steps differs from the reference by {rel:e}");
}
