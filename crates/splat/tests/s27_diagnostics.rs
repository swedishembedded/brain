// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-pixel diagnostics of the ray renderer (`Renderer::diagnose`): what a
//! rendered pixel is made of, so a reconstruction can be judged by whether it
//! is a surface and not only by whether it reproduces its photographs.
//!
//! 1. The device statistics are the f64 oracle's (`reference::diagnose_ray`),
//!    through a pinhole, a Brown lens and a fisheye.
//! 2. They separate the two things an expected range cannot: an opaque
//!    surface, and a stack of translucent gaussians spread in depth whose
//!    expected range lands on the same surface.
//!
//! Swedish Embedded AB implements reconstruction pipelines whose output is
//! validated in 3D, not only against the photographs they were fitted to.
//! If your team needs expertise in radiance-field evaluation then you can
//! procure our services by sending an email to info@swedishembedded.com.

use camera::Lens;
use data::rng::Lcg;
use splat::reference::diagnose_ray;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn device(g: &gpu_core::Gpu, s: &Splats, cam: &Camera, o: &RenderOpts) -> Vec<f32> {
    let mut r = Renderer::new(g, Kernels::at(0), s.len(), cam.width, cam.height, 0).growable();
    r.render(g, &GpuSplats::upload(g, s), cam, o);
    r.diagnose(g, cam, o)
}

fn scene(n: usize, seed: u64) -> Splats {
    let mut r = Lcg::new(seed);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[r.signed() * 1.0, r.signed() * 0.8, 3.0 + r.unit() * 1.5]);
        s.quats.extend_from_slice(&[0.5 + r.unit(), r.signed() * 0.5, r.signed() * 0.5, r.signed() * 0.5]);
        s.scales.extend_from_slice(&[0.15 + 0.2 * r.unit(), 0.15 + 0.2 * r.unit(), 0.05 + 0.1 * r.unit()]);
        s.opacities.push(0.3 + 0.6 * r.unit());
        s.colors.extend_from_slice(&[r.unit(), r.unit(), r.unit()]);
    }
    s
}

#[test]
fn the_device_diagnostics_are_the_oracles_through_every_lens() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let pin = Camera::look_at([0.1, -0.05, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 70.0, 40, 30);
    let brown = Camera { lens: Lens::Brown { k: [-0.12, 0.04, -0.01, 0.0, 0.0, 0.0], p: [8e-4, -5e-4], s: [0.0; 4] }, ..pin };
    let fish = Camera { fx: pin.fx * 0.8, fy: pin.fy * 0.8, lens: Lens::Fisheye { k: [0.03, -0.01, 0.002, 0.0] }, ..pin };
    let o = RenderOpts { ray: true, ..Default::default() };
    // few enough overlaps per pixel that the renderer's window holds the
    // exact per-pixel order
    let s = scene(24, 0xd1a6);
    for (name, cam) in [("pinhole", pin), ("brown", brown), ("fisheye", fish)] {
        let (got, want) = (device(&g, &s, &cam, &o), diagnose_ray(&s, &[], &cam, &o));
        let mut checked = 0;
        for p in 0..(cam.width * cam.height) as usize {
            let (a, b) = (&got[p * 12..p * 12 + 12], &want[p * 12..p * 12 + 12]);
            if b[11] < 0.05 {
                continue;
            }
            checked += 1;
            for (k, what, tol) in [(0, "alpha", 1e-3), (1, "expected range", 2e-3), (2, "median range", 2e-3), (3, "spread", 3e-3), (4, "entropy", 3e-3), (5, "count", 0.0), (6, "share", 1e-3)] {
                assert!((a[k] - b[k]).abs() <= tol * b[k].abs().max(1.0), "{name} pixel {p}: {what} {} on the device, {} in the oracle", a[k], b[k]);
            }
            assert_eq!(a[7].to_bits(), b[7].to_bits(), "{name} pixel {p}: dominant gaussian");
            for k in 8..11 {
                assert!((a[k] - b[k]).abs() < 2e-3, "{name} pixel {p}: normal");
            }
        }
        assert!(checked * 10 > (cam.width * cam.height) as usize, "{name}: only {checked} covered pixels compared");
    }
}

#[test]
fn a_stack_is_told_from_a_surface_its_expected_range_matches() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 40.0, 32, 32);
    let o = RenderOpts { ray: true, ..Default::default() };
    let sheet = |z: f32, opacity: f32| Splats {
        means: vec![0.0, 0.0, z],
        quats: vec![1.0, 0.0, 0.0, 0.0],
        scales: vec![2.0, 2.0, 0.01],
        opacities: vec![opacity],
        colors: vec![0.5, 0.5, 0.5],
        sh_rest: None,
    };
    let surface = sheet(3.0, 0.98);
    // five translucent sheets from 2.2 to 3.8: centred on the same range
    let stack = splat::align::concat(&[sheet(2.2, 0.35), sheet(2.6, 0.35), sheet(3.0, 0.35), sheet(3.4, 0.5), sheet(3.8, 0.9)]);
    let centre = |d: &[f32]| {
        let p = (16 * 32 + 16) * 12;
        d[p..p + 12].to_vec()
    };
    let (a, b) = (centre(&device(&g, &surface, &cam, &o)), centre(&device(&g, &stack, &cam, &o)));
    assert!((a[1] - 3.0).abs() < 0.02 && (b[1] - 3.0).abs() < 0.3, "expected ranges {} and {}: both near the surface", a[1], b[1]);
    assert!(a[3] < 0.02 && b[3] > 0.4, "range spread {} for the surface, {} for the stack", a[3], b[3]);
    assert!(a[4] < 0.05 && b[4] > 1.0, "contribution entropy {} for the surface, {} for the stack", a[4], b[4]);
    assert!(a[6] > 0.99 && b[6] < 0.5, "dominant share {} for the surface, {} for the stack", a[6], b[6]);
}
