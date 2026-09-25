// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense geometry for a training set: multi-view stereo on the photographs,
//! its measurements attached to every target as priors at the target's own
//! resolution, and the fused cloud as the starting scene.
//!
//! The photographs are rendered from a known scene - a textured floor and a
//! ball, made of small opaque gaussians - so the truth for every target pixel
//! is the range the renderer itself composited there.
//!
//! Swedish Embedded AB implements photogrammetry pipelines, from photographs
//! to dense geometry and radiance fields, for its clients. If your team needs
//! expertise in 3D reconstruction then you can procure our services by
//! sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use imaging::Rgb8;
use recon::photogrammetry::{dense, DenseCfg};
use splat::opt::TargetView;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};

/// A floor at y = 1 and a ball resting on it, tiled with small opaque
/// gaussians of random colour: texture everywhere for stereo to match.
fn scene() -> Splats {
    let mut r = Lcg::new(0xde75e);
    let mut s = Splats::default();
    let mut add = |p: [f32; 3], q: [f32; 4], size: f32, r: &mut Lcg| {
        s.means.extend_from_slice(&p);
        s.quats.extend_from_slice(&q);
        s.scales.extend_from_slice(&[size, size, size * 0.1]);
        s.opacities.push(0.98);
        s.colors.extend_from_slice(&[r.unit(), r.unit(), r.unit()]);
    };
    // floor: normal +y, so the thin axis (z) is rotated onto y
    let floor_q = [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0];
    for iz in 0..90 {
        for ix in 0..90 {
            add([-2.2 + ix as f32 * 0.05, 1.0, 2.0 + iz as f32 * 0.05], floor_q, 0.03, &mut r);
        }
    }
    // ball: each gaussian's thin axis along the radius
    let (c, rad) = ([0.0f32, 0.45, 4.2], 0.55f32);
    for k in 0..5000 {
        let z = 1.0 - 2.0 * (k as f32 + 0.5) / 5000.0;
        let phi = k as f32 * 2.399_963;
        let n = [(1.0 - z * z).sqrt() * phi.cos(), (1.0 - z * z).sqrt() * phi.sin(), z];
        let q = splat::orient::quat_of(&frame_with_z(n)).map(|v| v as f32);
        add([c[0] + rad * n[0], c[1] + rad * n[1], c[2] + rad * n[2]], q, 0.03, &mut r);
    }
    s
}

/// A rotation (row-major) whose third column is `n`.
fn frame_with_z(n: [f32; 3]) -> [f64; 9] {
    let n = n.map(|v| v as f64);
    let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let mut x = cross(a, n);
    let l = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    x = x.map(|v| v / l);
    let y = cross(n, x);
    [x[0], y[0], n[0], x[1], y[1], n[1], x[2], y[2], n[2]]
}

fn cameras(w: u32, h: u32) -> Vec<Camera> {
    (0..6)
        .map(|i| {
            let a = -0.5 + i as f32 * 0.2;
            let eye = [2.2 * a.sin(), -0.6 - 0.1 * (i % 2) as f32, 4.2 - 2.9 * a.cos()];
            let lens = camera::Lens::Brown { k: [-0.08, 0.02, 0.0, 0.0, 0.0, 0.0], p: [0.0; 2], s: [0.0; 4] };
            Camera { lens, ..Camera::look_at(eye, [0.0, 0.6, 4.2], [0.0, -1.0, 0.0], 70.0, w, h) }
        })
        .collect()
}

fn half(c: &Camera) -> Camera {
    Camera { fx: c.fx * 0.5, fy: c.fy * 0.5, cx: c.cx * 0.5, cy: c.cy * 0.5, width: c.width / 2, height: c.height / 2, ..*c }
}

#[test]
fn stereo_priors_land_on_each_target_at_its_resolution_and_agree_with_the_surface() {
    let pipes: Vec<(&str, &str)> = splat::PIPELINES.iter().chain(mvs::PIPELINES).copied().collect();
    let pipes: &'static [(&'static str, &'static str)] = Box::leak(pipes.into_boxed_slice());
    let g = gpu_core::testgpu::dev(pipes);
    if !g.caps().workgroup_reductions {
        eprintln!("multi-view stereo needs a GPU; skipped on {:?}", g.kind());
        return;
    }
    let (sk, mk) = (splat::Kernels::at(0), mvs::Kernels::at(splat::PIPELINES.len()));
    let truth = scene();
    let o = RenderOpts { ray: true, ..Default::default() };
    let (w, h) = (320u32, 240u32);
    let full = cameras(w, h);
    let gs = GpuSplats::upload(&g, &truth);
    let mut ren = Renderer::new(&g, sk, truth.len(), w, h, 0).growable();
    let photos: Vec<Rgb8> = full
        .iter()
        .map(|c| {
            ren.render(&g, &gs, c, &o);
            Rgb8 { w, h, px: ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect() }
        })
        .collect();
    // targets one halving below the photographs; the truth rendered there
    let mut targets: Vec<TargetView> = Vec::new();
    let mut want: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for c in &full {
        let hc = half(c);
        ren.render(&g, &gs, &hc, &o);
        let rgba = ren.read_rgba(&g, hc.width, hc.height);
        let aux = ren.read_aux(&g, hc.width, hc.height);
        targets.push(TargetView::new(hc, rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()));
        want.push((rgba.chunks_exact(4).map(|p| p[3]).collect(), aux.chunks_exact(5).map(|a| a[0]).collect()));
    }
    let tracks: Vec<mvs::Track> =
        truth.means.chunks_exact(3).step_by(7).map(|m| mvs::Track { xyz: [m[0] as f64, m[1] as f64, m[2] as f64], views: Vec::new() }).collect();
    let cfg = DenseCfg { stereo: mvs::StereoCfg { halving: 0, levels: 2, ..Default::default() }, ..Default::default() };
    let refs: Vec<&Rgb8> = photos.iter().collect();
    let (init, report) = dense(&g, &mk, &full, &refs, &tracks, &mut targets, &cfg).expect("stereo runs");

    let (mut good, mut measured, mut opaque) = (0usize, 0usize, 0usize);
    let mut errs: Vec<f32> = Vec::new();
    for (t, (alpha, range)) in targets.iter().zip(&want) {
        let d = t.depth.as_ref().expect("a range prior on every target");
        assert_eq!(d.len(), (t.cam.width * t.cam.height) as usize, "the prior is at the target's own resolution");
        assert_eq!(t.normals.as_ref().expect("normals").len(), 3 * d.len());
        assert_eq!(t.depth_conf.as_ref().expect("confidence").len(), d.len());
        for i in 0..d.len() {
            if alpha[i] < 0.99 {
                continue;
            }
            opaque += 1;
            if d[i] > 0.0 {
                measured += 1;
                good += ((d[i] - range[i]).abs() < 0.01 * range[i]) as usize;
                errs.push((d[i] - range[i]) / range[i]);
            }
        }
    }
    errs.sort_by(|a, b| a.total_cmp(b));
    let median = errs[errs.len() / 2];
    println!("  coverage {:?}, {} fused points; {good}/{measured} measured within 1% of {opaque} opaque pixels, median error {median:+.4}", report.coverage, report.points);
    assert!(measured as f64 > 0.5 * opaque as f64, "stereo measured {measured} of {opaque} surface pixels");
    // A prior misplaced by a fraction of a pixel shows up as a bias across
    // the slanted floor; the stereo's own noise at 320x240 is a spread
    // around zero.
    assert!(median.abs() < 0.003, "median relative range error {median:+.4}: the priors are not aligned with their targets");
    assert!(good as f64 > 0.75 * measured as f64, "{good} of {measured} measured ranges within 1% of the rendered surface");
    assert_eq!(init.len(), report.points);

    // the dense start lies on the true surfaces, to within the stereo's
    // noise (about 1% of a 3 m range here)
    let on_surface = init
        .means
        .chunks_exact(3)
        .filter(|m| {
            let floor = (m[1] - 1.0).abs() < 0.05;
            let d = ((m[0]).powi(2) + (m[1] - 0.45).powi(2) + (m[2] - 4.2).powi(2)).sqrt();
            floor || (d - 0.55).abs() < 0.05
        })
        .count();
    assert!(on_surface as f64 > 0.9 * init.len() as f64, "{on_surface} of {} dense gaussians on the surface", init.len());
}

#[test]
fn frames_compose_as_applied_in_turn() {
    use recon::photogrammetry::Frame;
    let rot = |a: f64, axis: usize| {
        let (s, c) = a.sin_cos();
        let mut r = [0.0; 9];
        let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
        r[axis * 3 + axis] = 1.0;
        r[i * 3 + i] = c;
        r[i * 3 + j] = -s;
        r[j * 3 + i] = s;
        r[j * 3 + j] = c;
        r
    };
    let a = Frame { r: rot(0.4, 0), centre: [1.0, -2.0, 0.5] };
    let b = Frame { r: rot(-1.1, 2), centre: [0.3, 0.7, -1.2] };
    let p = [0.2, 1.5, -0.8];
    let (x, y) = (b.apply(a.apply(p)), a.then(&b).apply(p));
    for k in 0..3 {
        assert!((x[k] - y[k]).abs() < 1e-12, "{x:?} vs {y:?}");
    }
}
