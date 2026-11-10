// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a rotated-codebook (`CQ2`/`CQ3`/`CQ4`) weight is multiplied in place from its stored indices and norms - the
//! product equals the host product of the dequantised weight to f32 rounding for every row count, every index width and a
//! width that is not a whole number of groups; the tile decode equals the host dequantiser; each product kernel agrees
//! on its own (the cooperative GEMV where the device has workgroup barriers, the row-blocked kernel everywhere).

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};
use model::cq::{padded, Cq, GROUP};
use model::ops::{cq_kernels, training_kernel_list, Ops, Weight};

const BOOKS: [&[f32]; 3] = [&[-1.51, -0.45, 0.45, 1.51], &[-1.75, -1.05, -0.5, -0.16, 0.16, 0.5, 1.05, 1.75], &[-2.4, -1.84, -1.44, -1.1, -0.8, -0.52, -0.25, 0.0, 0.25, 0.52, 0.8, 1.1, 1.44, 1.84, 2.4, 2.9]];

fn ops() -> Ops {
    let list: Vec<(&str, &str)> = training_kernel_list().iter().chain(cq_kernels()).copied().collect();
    Ops::new(Gpu::new(Box::leak(list.into_boxed_slice()))).expect("Ops::new")
}

fn book(bits: u32) -> &'static [f32] {
    BOOKS[bits as usize - 2]
}

/// `y[i, r] = sum_c x[i, c] w[r, c]` in f64 with the root of the summed squared terms beside it.
fn host_product(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<(f64, f64)> {
    let mut y = Vec::with_capacity(m * n);
    for i in 0..m {
        for r in 0..n {
            let terms: Vec<f64> = (0..k).map(|c| f64::from(x[i * k + c]) * f64::from(w[r * k + c])).collect();
            y.push((terms.iter().sum(), terms.iter().map(|t| t * t).sum::<f64>().sqrt()));
        }
    }
    y
}

fn assert_close(label: &str, got: &[f32], want: &[(f64, f64)]) {
    let worst = got.iter().zip(want).map(|(&g, &(w, scale))| (f64::from(g) - w).abs() / (1e-6 + scale)).fold(0f64, f64::max);
    // f32 summation of up to a few hundred terms: the error is a small multiple of the unit roundoff of the term scale.
    assert!(worst < 2e-5, "{label}: worst error {worst:e} of the term scale");
}

#[test]
fn products_equal_the_dequantised_matmul_for_every_width_and_row_count() {
    let ops = ops();
    let g = ops.gpu();
    for bits in [2u32, 3, 4] {
        // 200 is not a whole number of groups: the rotation pads the activation, the weight pads with quantised zeros.
        let (n, k) = (37usize, 200usize);
        let mut rng = Lcg::new(0xC0 + u64::from(bits));
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let wd = cq.dequantize();
        let weight = Weight::cq(&ops, "cq", &cq);
        assert!(!weight.is_dense_f32() && weight.reads_f32_activations());
        assert_eq!(weight.resident_bytes() as usize, cq.bytes(), "{bits}-bit resident bytes");
        for m in [1usize, 2, 7, 8, 9, 33] {
            let x = rng.vec_scaled(m * k, 1.0);
            let xb = g.storage_init("x", &x);
            let y = g.storage((m * n) as u64);
            let mut steps = Vec::new();
            let act = ops.act_f32(&xb, 0, m as u32, k as u32);
            ops.matmul(&mut steps, &weight, &act, &y, 0);
            g.submit(&[], &steps);
            assert_close(&format!("cq{bits} m={m} Ops::matmul"), &g.read(&y, m * n), &host_product(&x, m, k, &wd, n));
        }
    }
}

#[test]
fn the_tile_decode_equals_the_host_dequantiser() {
    let ops = ops();
    let g = ops.gpu();
    for bits in [2u32, 3, 4] {
        let (n, k) = (9usize, 200usize);
        let mut rng = Lcg::new(0xD0 + u64::from(bits));
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let weight = Weight::cq(&ops, "cq", &cq);
        let out = g.storage((n * k) as u64);
        let mut steps = Vec::new();
        ops.decode_tile(&mut steps, &weight, 0, n as u32, &out);
        g.submit(&[], &steps);
        let got = g.read(&out, n * k);
        let want = cq.dequantize();
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "cq{bits}: tile decode differs from the host dequantiser by {worst:e}");
    }
}

#[test]
fn each_product_kernel_agrees_with_the_host_on_its_own() {
    let ops = ops();
    let g = ops.gpu();
    let barriers = g.caps().workgroup_reductions;
    eprintln!("cq kernels on {} (workgroup barriers: {barriers})", g.kind());
    for bits in [2u32, 3, 4] {
        let (n, k) = (21usize, 256usize);
        let mut rng = Lcg::new(0xE0 + u64::from(bits));
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let wd = cq.dequantize();
        let weight = Weight::cq(&ops, "cq", &cq);
        let Weight::Cq { w, s, book: levels, .. } = &weight else { unreachable!() };
        for m in [1usize, 3, 8] {
            let x = rng.vec_scaled(m * k, 1.0);
            let xr = g.storage_init("xr", &Cq::rotate(&x, m, k));
            let want = host_product(&x, m, k, &wd, n);
            let params = [m as u32, padded(k) as u32, n as u32, 0];
            let mut variants = vec![(kernels::template::cq_gemm_kernel_name(bits), Dispatch::Threads(n as u32 * (m as u32).div_ceil(kernels::template::CQ_GEMM_ROWS)))];
            if barriers {
                variants.push((kernels::template::cq_gemv_kernel_name(bits, m as u32), Dispatch::Workgroups(n as u32)));
            }
            for (name, grid) in variants {
                let y = g.storage((m * n) as u64);
                let index = g.kernel_index(&name).unwrap_or_else(|| panic!("{name} is registered by cq_kernels"));
                g.submit(&[], &[g.dispatch(index, &[&xr, w, s, levels, &y], &params, grid)]);
                assert_close(&format!("{name} m={m}"), &g.read(&y, m * n), &want);
            }
        }
    }
    assert_eq!(GROUP, 128, "the kernels hard-code the 128-lane group");
}

#[test]
fn rotation_kernel_matches_the_host_transform() {
    let ops = ops();
    let g = ops.gpu();
    let (m, k) = (5usize, 200usize);
    let x = Lcg::new(77).vec_scaled(m * k, 1.0);
    let xb = g.storage_init("x", &x);
    let out = g.storage((m * padded(k)) as u64);
    let index = g.kernel_index("cq_rotate").expect("registered");
    g.submit(&[], &[g.step(index, &[&xb, &out], &[m as u32, k as u32, padded(k) as u32, 0], (m * padded(k) / GROUP) as u32)]);
    let got = g.read(&out, m * padded(k));
    let want = Cq::rotate(&x, m, k);
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst < 1e-5, "rotation differs by {worst:e}");
}
