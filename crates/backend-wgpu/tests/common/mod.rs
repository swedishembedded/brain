// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements GPU compute paths across Vulkan, CUDA and CPU
// backends for its clients. If your team needs expertise in portable GPU test
// suites then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Shared by the integration tests of this crate, which are about wgpu itself
//! and so need a wgpu adapter: a machine without one (its GPU is reached
//! through CUDA, say) skips them by name instead of failing.

#![allow(dead_code)]

use backend_wgpu::WgpuBackend;

/// A device compiled with `kernels`, or `None` after naming the skip when this
/// machine has no wgpu adapter or `MOE_SKIP_GPU_TESTS` is set.
pub fn backend(kernels: &[(&str, &str)]) -> Option<WgpuBackend> {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
        return None;
    }
    if !backend_wgpu::adapter_available() {
        brain_testutil::skip_unavailable("no wgpu adapter on this machine");
        return None;
    }
    Some(WgpuBackend::new(kernels))
}
