// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! R1 gate: projection math + naive compositing.
//!   1. analytic single-gaussian golden (closed-form isotropic case);
//!   2. device naive path == pure-Rust oracle on random scenes (CPU backend,
//!      wgpu duplicate unless MOE_SKIP_GPU_TESTS);
//!   3. PLY round-trip (write → read → identical post-activation values).

use gpu_core::Gpu;
use splat::reference;
use splat::renderer::{sorted_by_depth, GpuSplats, Renderer};
use splat::types::{Camera, Mode, RenderOpts, Splats};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32 // [0,1)
    }
}

fn one_gaussian() -> (Splats, Camera) {
    let s = Splats {
        means: vec![0.0, 0.0, 4.0],
        quats: vec![1.0, 0.0, 0.0, 0.0],
        scales: vec![0.1, 0.1, 0.1],
        opacities: vec![0.9],
        colors: vec![1.0, 0.5, 0.25],
        sh_rest: None,
    };
    let mut c2w = [0.0f32; 16];
    for i in 0..4 {
        c2w[i * 4 + i] = 1.0;
    }
    let cam = Camera { c2w, fx: 100.0, fy: 100.0, cx: 32.0, cy: 32.0, width: 64, height: 64 };
    (s, cam)
}

#[test]
fn analytic_isotropic_projection() {
    let (s, cam) = one_gaussian();
    let o = RenderOpts::default();
    let p = reference::project_one(&s, 0, &cam, &o).expect("visible");

    // Closed form: z=4, rz=.25, sigma2d = (fx*rz*s)^2 = 6.25, blurred 6.55.
    let var = 6.25f32 + 0.3;
    assert!((p.x - 32.0).abs() < 1e-5 && (p.y - 32.0).abs() < 1e-5);
    assert!((p.conic[0] - 1.0 / var).abs() < 1e-6, "conic_a {}", p.conic[0]);
    assert!(p.conic[1].abs() < 1e-7);
    assert!((p.conic[2] - 1.0 / var).abs() < 1e-6);
    assert!((p.opacity - 0.9).abs() < 1e-6); // antialiased=false
    assert!((p.depth - 4.0).abs() < 1e-6);
    let extend = (2.0f32 * (0.9f32 * 255.0).ln()).sqrt().min(3.33);
    let expect_r = (extend * var.sqrt()).ceil();
    assert_eq!(p.radius[0], expect_r);
    assert_eq!(p.radius[1], expect_r);

    // Center-adjacent pixel (31,31): dx=dy=0.5 from the (31.5,31.5) center.
    let img = reference::render(&s, &cam, &o);
    let sigma = 0.5 * (2.0 * 0.25 / var);
    let alpha = 0.9 * (-sigma).exp();
    let px = (31 * 64 + 31) * 4;
    assert!((img[px] - alpha * 1.0).abs() < 1e-5, "r {} vs {}", img[px], alpha);
    assert!((img[px + 3] - alpha).abs() < 1e-5);
}

#[test]
fn culling() {
    let (mut s, cam) = one_gaussian();
    let o = RenderOpts::default();
    // behind camera
    s.means[2] = -4.0;
    assert!(reference::project_one(&s, 0, &cam, &o).is_none());
    // transparent
    s.means[2] = 4.0;
    s.opacities[0] = 1.0 / 300.0;
    assert!(reference::project_one(&s, 0, &cam, &o).is_none());
    // far off-screen
    s.opacities[0] = 0.9;
    s.means[0] = 100.0;
    assert!(reference::project_one(&s, 0, &cam, &o).is_none());
}

fn random_scene(n: usize, seed: u64) -> (Splats, Camera) {
    let mut r = Lcg(seed);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[
            (r.next() - 0.5) * 4.0,
            (r.next() - 0.5) * 4.0,
            2.0 + r.next() * 6.0,
        ]);
        s.quats.extend_from_slice(&[
            r.next() * 2.0 - 1.0,
            r.next() * 2.0 - 1.0,
            r.next() * 2.0 - 1.0,
            r.next() * 2.0 - 1.0,
        ]);
        s.scales.extend_from_slice(&[
            0.01 + r.next() * 0.2,
            0.01 + r.next() * 0.2,
            0.01 + r.next() * 0.2,
        ]);
        s.opacities.push(r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    let cam = Camera::look_at([0.3, -0.2, -1.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 96, 80);
    (s, cam)
}

fn check_device_vs_oracle(g: &Gpu, tol: f32) {
    let ks = Kernels::at(0);
    for seed in [7u64, 8, 9] {
        let (s, cam) = random_scene(300, seed);
        for mode in [Mode::Color, Mode::Depth] {
            let o = RenderOpts { mode, bg: [0.1, 0.2, 0.3], ..Default::default() };
            let expect = reference::render(&s, &cam, &o);
            let sorted = sorted_by_depth(&s, &cam);
            let r = Renderer::new(g, ks, s.len(), cam.width, cam.height, 0);
            let gs = GpuSplats::upload(g, &sorted);
            let got = r.render_naive_gpu(g, &gs, &cam, &o);
            let mut max = 0.0f32;
            for (a, b) in got.iter().zip(&expect) {
                max = max.max((a - b).abs());
            }
            assert!(max < tol, "seed {seed} mode-depth={} max abs diff {max}", mode == Mode::Depth);
        }
    }
}

#[test]
fn device_naive_matches_oracle_cpu() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    check_device_vs_oracle(&g, 2e-4);
}

#[test]
fn device_naive_matches_oracle_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let g = Gpu::new(splat::PIPELINES);
    check_device_vs_oracle(&g, 2e-3);
}

#[test]
fn ply_roundtrip() {
    let (s, _) = random_scene(64, 42);
    let dir = std::env::temp_dir().join("splat_ply_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rt.ply");
    let path = path.to_str().unwrap();
    splat::ply::write(path, &s).unwrap();
    let r = splat::ply::read(path).unwrap();
    assert_eq!(r.len(), s.len());
    for i in 0..s.len() {
        for k in 0..3 {
            assert!((r.means[i * 3 + k] - s.means[i * 3 + k]).abs() < 1e-6);
            assert!((r.scales[i * 3 + k] - s.scales[i * 3 + k]).abs() < 1e-4);
            assert!((r.colors[i * 3 + k] - s.colors[i * 3 + k]).abs() < 1e-5);
        }
        for k in 0..4 {
            assert!((r.quats[i * 4 + k] - s.quats[i * 4 + k]).abs() < 1e-6);
        }
        assert!((r.opacities[i] - s.opacities[i]).abs() < 1e-5);
    }
}
