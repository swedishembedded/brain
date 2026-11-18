// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An int8 or int4 weight already resident in its packed form multiplies on a
//! device that executes no packed-int8 kernel: the selector offers such a
//! device `Reference` alone, which no quantized tier has a kernel for, so
//! [`Ops::matmul`] reads the codes through the float-read tile decode against
//! the f32 activation rows instead of dispatching a kernel the device cannot
//! run.
//!
//! Swedish Embedded AB implements quantized inference that runs on whatever
//! device its clients have. If your team needs expertise in portable
//! low-precision linear algebra, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The CPU JIT is such a device (`numeric.int8_dot` is false); the test states
//! that precondition rather than assuming it.

use data::rng::Lcg;
use model::ops::{Ops, Weight};

fn ops() -> Ops {
    let mut list = model::ops::two_level_kernel_list().to_vec();
    model::ops::extend_kernel_list(&mut list, model::ops::frozen_base_kernels());
    let ops = Ops::new(gpu_core::Gpu::new_cpu(&list)).expect("Ops::new");
    assert!(!ops.gpu().caps().numeric.int8_dot, "this test needs a device without the packed int8 dot");
    ops
}

/// `y = x @ Wᵀ` against the weight the codes decode to, per tier.
#[test]
fn packed_weights_multiply_through_the_tile_decode_where_no_packed_kernel_runs() {
    let (m, n, k) = (5usize, 70usize, 128usize);
    let ops = ops();
    let g = ops.gpu();
    let mut rng = Lcg::new(7);
    let raw: Vec<f32> = (0..n * k).map(|_| (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();
    let x: Vec<f32> = (0..m * k).map(|_| (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();

    let (i8_codes, i8_scales) = model::int8::quantize_weight(&raw, n, k);
    let (q4_codes, q4_scales) = model::int4::quantize_weight_q4(&raw, n, k);
    let upload = |codes: &[u32], scales: &[f32]| {
        let w = g.storage(codes.len() as u64);
        g.write(&w, codes);
        (w, g.storage_init("scales", scales))
    };
    let (wi, si) = upload(&i8_codes, &i8_scales);
    let (wq, sq) = upload(&q4_codes, &q4_scales);
    let tiers = [
        (Weight::I8 { w: wi, s: si, n: n as u32, k: k as u32 }, model::int8::dequantize_weight(&i8_codes, &i8_scales, n, k)),
        (Weight::Q4 { w: wq, s: sq, n: n as u32, k: k as u32 }, model::int4::dequantize_weight_q4(&q4_codes, &q4_scales, n, k)),
    ];

    let xb = g.storage_init("x", &x);
    for (weight, decoded) in &tiers {
        let y = g.storage((m * n) as u64);
        let mut steps = Vec::new();
        // Both activation forms: a two-level one asks for a second level this path has no use for.
        let act = ops.act_two_level(&mut steps, &xb, 0, m as u32, k as u32);
        ops.matmul(&mut steps, weight, &act, &y, 0);
        g.submit(&[], &steps);
        g.poll_wait();
        let got = g.read(&y, m * n);
        for r in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|i| f64::from(x[r * k + i]) * f64::from(decoded[j * k + i])).sum();
                let err = (f64::from(got[r * n + j]) - want).abs();
                assert!(err <= 1e-4 * (1.0 + want.abs()), "{:?} y[{r},{j}] = {} want {want}", weight.dtype(), got[r * n + j]);
            }
        }
    }
}

/// A device without the packed dot widens an int8 request to f32 for a caller that dispatches the packed kernel
/// itself, and holds it at int8 for one that reads the codes against f32 activations (a frozen LoRA base): the
/// uploaded weight is the one the codes decode to either way.
#[test]
fn an_int8_request_is_held_as_int8_only_where_the_caller_reads_the_codes_by_float_arithmetic() {
    let (n, k) = (4usize, 64usize);
    let mut ops = ops();
    let raw: Vec<f32> = (0..n * k).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    assert_eq!(ops.lands(gpu_core::select::Dtype::I8), gpu_core::select::Dtype::F32, "by default this device widens int8");
    assert_eq!(Weight::upload(&ops, &raw, n, k, gpu_core::select::Dtype::I8).dtype(), gpu_core::select::Dtype::F32);

    ops.hold_int8_by_float_read();
    assert_eq!(ops.lands(gpu_core::select::Dtype::I8), gpu_core::select::Dtype::I8);
    assert_eq!(Weight::upload(&ops, &raw, n, k, gpu_core::select::Dtype::I8).dtype(), gpu_core::select::Dtype::I8);
    assert_eq!(ops.lands(gpu_core::select::Dtype::Q4), gpu_core::select::Dtype::F32, "only int8 is read by float arithmetic here");
}
