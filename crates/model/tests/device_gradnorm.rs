// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of large language models for its clients. If your
// team needs expertise in measuring a global gradient norm over gradients that never leave the device then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec of the device gradient norm (`optim::gradnorm`): the sum of squares of f32 and bf16 gradient buffers, reduced
//! on the device, equals the host's f64 sum to f32 accuracy, for buffers that span one to many workgroups and for
//! more buffers than one round of partials holds.

use data::rng::Lcg;
use gpu_core::Gpu;
use optim::gradnorm::{DeviceGradNorm, NormItem};
use model::half::pack_bf16;
use model::ops::{training_kernel_list, GradDtype};

fn host_sum_sq(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x) * f64::from(x)).sum()
}

#[test]
fn the_device_sum_of_squares_matches_the_host_for_f32_and_bf16() {
    let gpu = Gpu::new(training_kernel_list());
    let norm = DeviceGradNorm::new(&gpu).expect("kernels registered");
    let mut rng = Lcg::random();
    // One element, a few workgroups, and a buffer wide enough that its partial count is capped.
    let sizes = [2usize, 8194, 70_000, 5_000_000];
    let mut owned = Vec::new();
    for &n in &sizes {
        let v = rng.vec_scaled(n, 1.0);
        let f32_buf = gpu.storage(n as u64);
        gpu.write_f32(&f32_buf, &v);
        // The bf16 buffer holds the bf16 rounding of the same values; its reference is the rounded values.
        let words = pack_bf16(&v);
        let bf16_buf = gpu.storage(words.len() as u64);
        gpu.write(&bf16_buf, &words);
        let rounded: Vec<f32> = v.iter().map(|&x| f32::from_bits(u32::from(model::half::f32_to_bf16(x)) << 16)).collect();
        owned.push((f32_buf, bf16_buf, host_sum_sq(&v), host_sum_sq(&rounded), n));
    }
    let items: Vec<NormItem<'_>> = owned.iter().flat_map(|(a, b, _, _, n)| [NormItem { buf: a, numel: *n, dtype: GradDtype::F32 }, NormItem { buf: b, numel: *n, dtype: GradDtype::Bf16 }]).collect();
    let got = norm.sum_sq_each(&gpu, &items);
    for (i, (_, _, want32, want16, n)) in owned.iter().enumerate() {
        let (g32, g16) = (got[2 * i], got[2 * i + 1]);
        assert!((g32 - want32).abs() <= 1e-4 * want32, "f32 n={n}: {g32} vs {want32}");
        assert!((g16 - want16).abs() <= 1e-4 * want16, "bf16 n={n}: {g16} vs {want16}");
    }
    assert!((norm.sum_sq(&gpu, &items) - got.iter().sum::<f64>()).abs() < 1e-9);
}

#[test]
fn more_buffers_than_one_round_of_partials_are_reduced_in_rounds() {
    let gpu = Gpu::new(training_kernel_list());
    let norm = DeviceGradNorm::new(&gpu).expect("kernels registered");
    // 40 buffers of 1 Mi elements: 128 partials each, more than the 4096 one round holds.
    let n = 1 << 20;
    let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let bufs: Vec<_> = (0..40)
        .map(|_| {
            let b = gpu.storage(n as u64);
            gpu.write_f32(&b, &v);
            b
        })
        .collect();
    let items: Vec<NormItem<'_>> = bufs.iter().map(|b| NormItem { buf: b, numel: n, dtype: GradDtype::F32 }).collect();
    let got = norm.sum_sq_each(&gpu, &items);
    let want = host_sum_sq(&v);
    assert_eq!(got.len(), 40);
    for g in got {
        assert!((g - want).abs() <= 1e-4 * want, "{g} vs {want}");
    }
}
