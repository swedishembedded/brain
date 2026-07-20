// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! R2 gate: the tiled pipeline must match the naive oracle path.
//!   1. tiled == pure-Rust reference on procedural scenes (several cameras,
//!      edge clipping, behind-camera, both view modes), within 1/255;
//!   2. order invariance: shuffling the input gaussian order (well-separated
//!      depths) leaves the tiled output bit-identical per backend;
//!   3. rgba8 packing matches the f32 framebuffer;
//!   4. instance-cap clamping degrades gracefully (no crash, stats.clamped).

use gpu_core::Gpu;
use splat::reference;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, Mode, RenderOpts, Splats};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next() * n as f32) as usize % n
    }
}

fn random_scene(n: usize, seed: u64) -> Splats {
    let mut r = Lcg(seed);
    let mut s = Splats::default();
    for i in 0..n {
        // well-separated depth per gaussian → unambiguous sort order even
        // after key truncation (order-invariance needs determinism).
        let z = 2.0 + i as f32 * 0.05 + r.next() * 0.01;
        s.means.extend_from_slice(&[(r.next() - 0.5) * 5.0, (r.next() - 0.5) * 4.0, z]);
        s.quats.extend_from_slice(&[
            r.next() * 2.0 - 1.0,
            r.next() * 2.0 - 1.0,
            r.next() * 2.0 - 1.0,
            r.next() * 2.0 - 1.0,
        ]);
        s.scales.extend_from_slice(&[
            0.02 + r.next() * 0.3,
            0.02 + r.next() * 0.3,
            0.02 + r.next() * 0.3,
        ]);
        s.opacities.push(0.05 + 0.95 * r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    s
}

fn cameras() -> Vec<Camera> {
    vec![
        Camera::look_at([0.0, 0.0, -2.0], [0.0, 0.0, 5.0], [0.0, -1.0, 0.0], 60.0, 200, 120),
        // off-axis, non-multiple-of-16 resolution → partial edge tiles
        Camera::look_at([3.0, -1.5, 0.0], [0.0, 0.5, 8.0], [0.0, -1.0, 0.0], 75.0, 173, 99),
        // inside the scene: some gaussians behind the camera
        Camera::look_at([0.0, 0.0, 6.0], [1.0, 0.0, 12.0], [0.0, -1.0, 0.0], 60.0, 128, 128),
    ]
}

fn check_tiled_vs_reference(g: &Gpu, tol: f32) {
    let ks = Kernels::at(0);
    let s = random_scene(400, 0xc0de);
    for (ci, cam) in cameras().iter().enumerate() {
        for mode in [Mode::Color, Mode::Depth] {
            let o = RenderOpts { mode, bg: [0.2, 0.1, 0.4], ..Default::default() };
            let expect = reference::render(&s, cam, &o);
            let mut r = Renderer::new(g, ks, s.len(), cam.width, cam.height, 0);
            let gs = GpuSplats::upload(g, &s);
            let stats = r.render(g, &gs, cam, &o);
            assert!(!stats.clamped);
            let got = r.read_rgba(g, cam.width, cam.height);
            let mut max = 0.0f32;
            for (a, b) in got.iter().zip(&expect) {
                max = max.max((a - b).abs());
            }
            assert!(
                max < tol,
                "cam {ci} depth-mode={} max abs diff {max} (isects {})",
                mode == Mode::Depth,
                stats.n_isects
            );
        }
    }
}

#[test]
fn tiled_matches_reference_cpu() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    check_tiled_vs_reference(&g, 1.0 / 255.0);
}

#[test]
fn tiled_matches_reference_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let g = Gpu::new(splat::PIPELINES);
    check_tiled_vs_reference(&g, 1.0 / 255.0);
}

#[test]
fn order_invariance_cpu() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let s = random_scene(300, 0xfeed);
    // Fisher–Yates shuffle into a second scene.
    let mut order: Vec<usize> = (0..s.len()).collect();
    let mut r = Lcg(0xd1ce);
    for i in (1..order.len()).rev() {
        order.swap(i, r.next_usize(i + 1));
    }
    let mut shuffled = Splats::default();
    for &i in &order {
        shuffled.means.extend_from_slice(&s.means[i * 3..i * 3 + 3]);
        shuffled.quats.extend_from_slice(&s.quats[i * 4..i * 4 + 4]);
        shuffled.scales.extend_from_slice(&s.scales[i * 3..i * 3 + 3]);
        shuffled.opacities.push(s.opacities[i]);
        shuffled.colors.extend_from_slice(&s.colors[i * 3..i * 3 + 3]);
    }
    let cam = cameras().remove(0);
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, s.len(), cam.width, cam.height, 0);
    let a = {
        let gs = GpuSplats::upload(&g, &s);
        ren.render(&g, &gs, &cam, &o);
        ren.read_rgba(&g, cam.width, cam.height)
    };
    let b = {
        let gs = GpuSplats::upload(&g, &shuffled);
        ren.render(&g, &gs, &cam, &o);
        ren.read_rgba(&g, cam.width, cam.height)
    };
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "tiled output depends on input order"
    );
}

#[test]
fn rgba8_pack_matches_f32() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let s = random_scene(100, 0xbeef);
    let cam = cameras().remove(0);
    let o = RenderOpts { bg: [0.5, 0.0, 1.0], ..Default::default() };
    let mut r = Renderer::new(&g, ks, s.len(), cam.width, cam.height, 0);
    let gs = GpuSplats::upload(&g, &s);
    r.render(&g, &gs, &cam, &o);
    let f32s = r.read_rgba(&g, cam.width, cam.height);
    let rgb = r.read_rgb24(&g, cam.width, cam.height);
    for i in 0..(cam.width * cam.height) as usize {
        for k in 0..3 {
            let expect = (f32s[i * 4 + k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            assert_eq!(rgb[i * 3 + k], expect, "pixel {i} chan {k}");
        }
    }
}

#[test]
fn instance_cap_clamps_gracefully() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let s = random_scene(400, 0xcafe);
    let cam = cameras().remove(0);
    let o = RenderOpts::default();
    let mut r = Renderer::new(&g, ks, s.len(), cam.width, cam.height, 512);
    let gs = GpuSplats::upload(&g, &s);
    let stats = r.render(&g, &gs, &cam, &o);
    assert!(stats.clamped);
    assert_eq!(stats.n_isects, 512);
    let img = r.read_rgba(&g, cam.width, cam.height);
    assert!(img.iter().all(|v| v.is_finite()));
}
