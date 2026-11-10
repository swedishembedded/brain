// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the card a test was handed can run, read from the native kernel
//! registry rather than copied into each test.

use gpu_core::Gpu;

/// Whether `gpu` is a CUDA device at the registered floor of the native kernel
/// `kernel` (`kernels_cuda`'s `min_cc`): below it the kernel is never offered,
/// and a test that expects it to run has nothing to check.
pub fn runs(gpu: &Gpu, kernel: &str) -> bool {
    let k = kernels_cuda::get(kernel).unwrap_or_else(|| panic!("no native kernel {kernel:?} is registered"));
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some_and(|cc| k.min_cc <= cc)
}

/// Whether `gpu` has the FP8 conversions every E4M3 kernel is built on
/// (`kernels_cuda::FP8_CVT_MIN_CC`).
pub fn fp8(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::FP8_CVT_MIN_CC)
}
