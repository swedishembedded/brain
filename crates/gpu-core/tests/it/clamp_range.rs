// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `clamp_range`: elementwise `min(max(x, lo), hi)`, the clipping step of a
//! clipped SwiGLU feed-forward.
//!
//! Swedish Embedded AB implements bit-exact device kernels and the gates that
//! keep them bit-exact for its clients. If your team needs expertise in
//! numerically bounded activations for large decoders then you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! Clamping is exact (a comparison, no rounding), so the gate compares raw
//! bits: two-sided bounds, a one-sided cap written with `-f32::MAX` as the
//! inert lower bound, and values exactly on a bound.

const KERNELS: &[(&str, &str)] = &[("clamp_range", kernels::CLAMP_RANGE)];
const K_CLAMP_RANGE: usize = 0;

fn run(input: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let n = input.len();
    let x = gpu.storage(n as u64);
    gpu.write_f32(&x, input);
    let out = gpu.storage(n as u64);
    let step = gpu.step(K_CLAMP_RANGE, &[&x, &out], &[n as u32, gpu_core::f(lo), gpu_core::f(hi)], n as u32);
    gpu.submit(&[], &[step]);
    gpu.read(&out, n)
}

#[test]
fn clamp_range_bounds_both_sides_and_leaves_inner_values_untouched() {
    let input: [f32; 11] = [-30.0, -20.0, -19.999, -0.5, 0.0, 0.5, 19.999, 20.0, 20.001, 1.0e30, f32::MIN];
    let want: Vec<f32> = input.iter().map(|v| v.clamp(-20.0, 20.0)).collect();
    let got = run(&input, -20.0, 20.0);
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn clamp_range_with_an_inert_lower_bound_is_a_cap() {
    let input: [f32; 7] = [-1.0e30, -25.0, 0.0, 19.0, 20.0, 21.0, 100.0];
    let want: Vec<f32> = input.iter().map(|v| v.min(20.0)).collect();
    let got = run(&input, -f32::MAX, 20.0);
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}
