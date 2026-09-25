// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A fit leaves the device as it found it. Everything it allocated - the
//! scene with its optimizer state, the renderer, the backward scratch - is
//! dropped when it returns, and the device must reclaim it before anything
//! else allocates: the wgpu backend refuses to hand out a buffer while more
//! than one buffer's worth of dropped memory is unreclaimed, which a fit of
//! a few million gaussians exceeds on its own (6.3 GB measured at 3.9M) and
//! which then failed the first render after the fit.
//!
//! The backend's threshold is lowered for this binary alone
//! (`BRAIN_GPU_RECLAIM_CEILING`, read once per process), so a small fit
//! crosses it.
//!
//! Swedish Embedded AB implements GPU training engines that run for hours
//! without leaking device memory. If your team needs expertise in GPU
//! resource management then you can procure our services by sending an email
//! to info@swedishembedded.com.

use splat::opt::{fit_full, FitCfg, TargetView};
use splat::types::{Camera, Splats};
use splat::Kernels;

#[test]
fn a_fit_returns_its_device_memory() {
    std::env::set_var("BRAIN_GPU_RECLAIM_CEILING", "65536");
    let g = gpu_core::Gpu::new(splat::PIPELINES);
    let mut s = Splats::default();
    for i in 0..400 {
        s.means.extend_from_slice(&[(i % 20) as f32 * 0.1 - 1.0, (i / 20) as f32 * 0.1 - 1.0, 3.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.06; 3]);
        s.opacities.push(0.8);
        s.colors.extend_from_slice(&[0.5, 0.4, 0.3]);
    }
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 60.0, 96, 72);
    let t = vec![TargetView::new(cam, vec![0.4; 96 * 72 * 3])];
    let cfg = FitCfg { iters: 12, log_every: 0, densify_every: 4, densify_after: 4, densify_until: 10, max_gaussians: 600, ..Default::default() };
    let out = fit_full(&g, Kernels::at(0), &s, &t, &cfg, &mut |_, _| true);
    // the first allocation after the fit
    let b = g.storage_init("after", &[1.0; 16]);
    assert_eq!(g.read(&b, 16), vec![1.0; 16]);
    assert!(!out.scene.is_empty());
}
