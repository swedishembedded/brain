// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fast decode for FP8 and NVFP4 checkpoints for its clients. If your team needs
// expertise in bandwidth-bound quantized GEMVs on one accelerator then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a stored FP8 or NVFP4 weight multiplied against up to `GEMV_MAX_ROWS` activation rows is read by its own
//! float-read kernel (no decoded scratch), and the product equals the host product of the exact decoded weight, for
//! every row count, a row window that does not start at 0 and a ragged row count; above the row cap the tile decode
//! answers instead and agrees. Where the vendor library has the tile product (the default) the cap is two rows, since
//! the tile product reads the weight once for more rows than that; while the library declines (a recording that is
//! replayed as a graph) the GEMVs serve all `GEMV_MAX_ROWS`.

use checkpoint::packed::{CodecRegistry, QuantEncoding, TensorDescriptor, TensorSink, Tile};
use data::rng::Lcg;
use gpu_core::Gpu;
use model::ops::{training_kernel_list, DeviceWeightSink, Ops, Weight, GEMV_MAX_ROWS};


fn decode_weight(ops: &Ops, w: &Weight) -> Vec<f32> {
    let g = ops.gpu();
    let out = g.storage((w.n() * w.k()) as u64);
    let mut steps = Vec::new();
    ops.decode_tile(&mut steps, w, 0, w.n(), &out);
    g.submit(&[], &steps);
    g.read(&out, (w.n() * w.k()) as usize)
}

/// `y[mi, c] = sum_k x[mi, k] * wd[first + c, k]` in f64, with the root of the sum of squared terms beside each
/// (the scale rounding errors of any summation order are measured against).
fn host_product(x: &[f32], m: usize, k: usize, wd: &[f32], first: usize, count: usize) -> Vec<(f64, f64)> {
    let mut y = vec![(0f64, 0f64); m * count];
    for mi in 0..m {
        for c in 0..count {
            let terms: Vec<f64> = (0..k).map(|i| x[mi * k + i] as f64 * wd[(first + c) * k + i] as f64).collect();
            y[mi * count + c] = (terms.iter().sum(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
        }
    }
    y
}

fn check(ops: &Ops, w: &Weight, label: &str) {
    let gemv_rows = if ops.gpu().declines_library() { GEMV_MAX_ROWS as usize } else { 2 };
    let (n, k) = (w.n() as usize, w.k() as usize);
    let g = ops.gpu();
    let wd = decode_weight(ops, w);
    let mut rng = Lcg::new(data::rng::random_seed());
    for (m, first, count) in [(1usize, 0usize, n), (2, 0, n), (5, 3, n - 17), (GEMV_MAX_ROWS as usize, 0, n), (GEMV_MAX_ROWS as usize + 1, 0, n)] {
        let x = rng.vec_scaled(m * k, 1.0);
        let xb = g.storage_init("x", &x);
        let y = g.storage((m * count) as u64);
        let mut steps = Vec::new();
        let act = ops.act_f32(&xb, 0, m as u32, k as u32);
        ops.matmul_rows(&mut steps, w, (first as u32, count as u32), &act, &y, 0);
        g.submit(&[], &steps);
        let got = g.read(&y, m * count);
        let want = host_product(&x, m, k, &wd, first, count);
        for (i, (a, (b, scale))) in got.iter().zip(&want).enumerate() {
            // The GEMV accumulates in f32; the tile decode above the row cap multiplies on bf16 tensor cores.
            let tol = if m > gemv_rows { 2e-2 * scale } else { 1e-5 * scale } + 1e-6;
            assert!((*a as f64 - b).abs() <= tol, "{label} m={m} first={first}: element {i} is {a}, host product {b} (scale {scale})");
        }
    }
}

/// A short weight (one 4-row round per workgroup) and a tall one (the grid walks several rounds per workgroup, as the
/// LM head does).
const SHAPES: [(usize, usize); 2] = [(37, 512), (4500, 128)];

#[test]
fn stored_fp8_and_nvfp4_weights_multiply_few_rows_by_their_own_kernels() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if !ops.gpu().caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the GEMV kernels need workgroup barriers (not the CPU JIT)");
    }
    for declines_library in [false, true] {
        ops.gpu().set_declines_library(declines_library);
        check_shapes(&ops);
    }
}

fn check_shapes(ops: &Ops) {
    for (n, k) in SHAPES {
        let raw = Lcg::new(data::rng::random_seed()).vec_scaled(n * k, 0.5);
        let fp8 = Weight::upload(ops, &raw, n, k, gpu_core::select::Dtype::F8E4M3);
        assert!(matches!(fp8, Weight::FP8 { .. }));
        check(ops, &fp8, &format!("fp8 [{n}, {k}]"));

        // NVFP4 from random valid bytes: E2M1 codes anywhere, E4M3 scales of ordinary magnitude, a tensor scale.
        let enc = QuantEncoding::nvfp4();
        let desc = TensorDescriptor::new("w", vec![n as u64, k as u64], enc.clone());
        let mut rng = Lcg::new(data::rng::random_seed());
        let payload: Vec<u8> = (0..desc.payload_bytes()).map(|_| rng.next_u32() as u8).collect();
        let scales = [1.5f32.to_le_bytes().to_vec(), (0..desc.scale_bytes(1)).map(|_| 0x30 + (rng.next_u32() % 0x20) as u8).collect::<Vec<u8>>()];
        let tile = Tile { elems: 0..desc.numel(), payload: &payload, scales: vec![&scales[0], &scales[1]] };
        let mut sink = DeviceWeightSink::new(ops);
        sink.begin(&desc).unwrap();
        sink.write_tile(&desc, &tile).unwrap();
        sink.finish(&desc).unwrap();
        let nvfp4 = sink.take_weight().unwrap();
        // The oracle of the tier is the reference codec: the device decode and the GEMV agree with it.
        let mut want = Vec::new();
        CodecRegistry::reference().decoder(&enc).unwrap().decode_tile(&desc, &tile, &mut want).unwrap();
        assert!(decode_weight(ops, &nvfp4).iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        check(ops, &nvfp4, &format!("nvfp4 [{n}, {k}]"));
    }
}
