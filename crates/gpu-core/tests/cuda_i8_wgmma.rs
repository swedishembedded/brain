// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on
// quantised language models for its clients. If your team needs expertise in
// putting int8 weights onto Hopper warpgroup tensor cores without giving up
// their group-wise scales, then you can procure our services by sending an
// email to info@swedishembedded.com.

//! `gpu_core::provider::cuda::i8_wgmma_step` through the `Gpu` handle: the int8-weight GEMM on
//! warpgroup tensor cores agrees with a host reference (bf16 activations, weights decoded to
//! bf16, f64 accumulation), declines what it does not serve, and writes only its output window.
//! Skipped where there is no Hopper device.

use gpu_core::provider::cuda::{i8_wgmma_step, I8_WGMMA_MIN_ROWS};
use gpu_core::Gpu;

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}
fn bf16_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn device() -> Option<Gpu> {
    let Ok(gpu) = Gpu::try_new_cuda(&[]) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return None;
    };
    if gpu.caps().arch.compute_capability != Some((9, 0)) {
        brain_testutil::skip_unavailable("needs a Hopper device");
        return None;
    }
    Some(gpu)
}

#[test]
fn the_warpgroup_gemm_agrees_with_the_host_reference_and_writes_only_its_window() {
    let Some(gpu) = device() else { return };
    let (m, n, k) = (200usize, 256usize, 512usize); // a ragged M: rows past the end of the activations
    let mut rng = Lcg(0x5eed); // seed: any data serves, the reference reads the same values
    let a: Vec<u16> = (0..m * k).map(|_| bf16_bits((rng.next() - 0.5) * 4.0)).collect();
    let w: Vec<i8> = (0..n * k).map(|_| ((rng.next() * 255.0) as i32 - 128) as i8).collect();
    let sw: Vec<f32> = (0..n * (k / 32)).map(|_| 0.002 + rng.next() * 0.02).collect();
    let words = |bytes: &[u8]| -> Vec<f32> { bytes.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect() };
    let a_bytes: Vec<u8> = a.iter().flat_map(|h| h.to_le_bytes()).collect();
    let w_bytes: Vec<u8> = w.iter().map(|&x| x as u8).collect();
    // Buffers hold 32-bit words; the data goes in as their bit patterns.
    let act = gpu.storage_init("act", &words(&a_bytes));
    let wq = gpu.storage_init("wq", &words(&w_bytes));
    let scales = gpu.storage_init("sw", &sw);
    // The output buffer is larger than the window: a margin of canaries on both sides.
    let margin = 64usize;
    let mut init = vec![-12345.0f32; margin + m * n + margin];
    init[margin..margin + m * n].iter_mut().for_each(|x| *x = 0.0);
    let out = gpu.storage_init("out", &init);

    let step = i8_wgmma_step(&gpu, &act, &wq, &scales, &out, (margin as u64, (m * n) as u64), m as u32, n as u32, k as u32).expect("Hopper serves the shape");
    gpu.submit(&[], &[step]);
    let got = gpu.read(&out, init.len());

    assert!(got[..margin].iter().chain(&got[margin + m * n..]).all(|&x| x == -12345.0), "the kernel wrote outside its output window");
    let wd: Vec<f32> = (0..n * k).map(|i| bf16_f32(bf16_bits(w[i] as f32 * sw[(i / k) * (k / 32) + (i % k) / 32]))).collect();
    let af: Vec<f32> = a.iter().map(|&h| bf16_f32(h)).collect();
    let mut worst = 0f64;
    let mut sum_sq = 0f64;
    for r in 0..m {
        for c in 0..n {
            let want: f64 = (0..k).map(|i| af[r * k + i] as f64 * wd[c * k + i] as f64).sum();
            sum_sq += want * want;
            worst = worst.max((got[margin + r * n + c] as f64 - want).abs());
        }
    }
    let rms = (sum_sq / (m * n) as f64).sqrt();
    assert!(worst < 2e-3 * rms, "worst error {worst} against an output rms of {rms}");
}

#[test]
fn shapes_the_kernel_does_not_tile_are_declined() {
    let Some(gpu) = device() else { return };
    let buf = gpu.storage(1024);
    let step = |m: u32, n: u32, k: u32| i8_wgmma_step(&gpu, &buf, &buf, &buf, &buf, (0, 0), m, n, k);
    assert!(step(I8_WGMMA_MIN_ROWS - 1, 256, 512).is_none(), "too few rows for a 128-row tile");
    assert!(step(256, 48, 512).is_none(), "n must be a multiple of 128");
    assert!(step(256, 256, 96).is_none(), "k must be a multiple of 128");
}
