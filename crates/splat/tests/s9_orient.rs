// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Re-framing a scene must move it without changing it.
//!
//! A feed-forward reconstruction lands in its FIRST camera's frame, so "up" in
//! the file is whichever way the camera was held and every viewer opens the
//! scene tipped - 63 degrees on a real capture. Straightening it is only safe
//! if the transform is genuinely rigid, and "rigid" here has a sharper meaning
//! than "orthonormal matrix": a gaussian carries an orientation and a shape,
//! so rotating one has to compose with its quaternion and leave its scales
//! alone. Getting that wrong shears every splat in a way no still frame from
//! the training views would reveal.
//!
//! So the test is not that the numbers transformed - it is that a camera moved
//! by the same transform sees an identical picture.
//!
//! Swedish Embedded AB implements 3D reconstruction pipelines whose output
//! lands in a frame the next tool can use. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::orient::{apply, frame_from_cameras, transform_c2w};
use splat::quality::psnr;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// Deliberately anisotropic and deliberately rotated: an isotropic blob would
/// hide a quaternion that was not composed at all.
fn scene(n: usize) -> Splats {
    let mut r = Lcg(0x5107);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[(r.next() - 0.5) * 2.0, (r.next() - 0.5) * 2.0, 3.0 + r.next()]);
        let q = [r.next() - 0.5, r.next() - 0.5, r.next() - 0.5, r.next() - 0.5];
        let l = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
        s.quats.extend_from_slice(&[q[0] / l, q[1] / l, q[2] / l, q[3] / l]);
        s.scales.extend_from_slice(&[0.02 + 0.10 * r.next(), 0.02, 0.02 + 0.05 * r.next()]);
        s.opacities.push(0.6 + 0.35 * r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    s
}

fn orbit(n: usize) -> Vec<[f64; 16]> {
    // cameras on a tilted ring, so the recovered "up" is nothing like an axis
    (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            let (e, tilt) = ([1.6 * a.cos(), -0.9, 3.0 + 1.6 * a.sin()], 0.5);
            let f = {
                let d = [-e[0], -e[1] - tilt, 3.0 - e[2]];
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                [d[0] / l, d[1] / l, d[2] / l]
            };
            let up = [0.0, -1.0, 0.0];
            let rr = {
                let c = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]];
                let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                [c[0] / l, c[1] / l, c[2] / l]
            };
            let d2 = [f[1] * rr[2] - f[2] * rr[1], f[2] * rr[0] - f[0] * rr[2], f[0] * rr[1] - f[1] * rr[0]];
            [rr[0], d2[0], f[0], e[0], rr[1], d2[1], f[1], e[1], rr[2], d2[2], f[2], e[2], 0.0, 0.0, 0.0, 1.0]
        })
        .collect()
}

fn cam_from(m: &[f64; 16], w: u32, h: u32) -> Camera {
    let mut c2w = [0.0f32; 16];
    for (i, v) in m.iter().enumerate() {
        c2w[i] = *v as f32;
    }
    Camera { c2w, fx: 180.0, fy: 180.0, cx: w as f32 / 2.0, cy: h as f32 / 2.0, width: w, height: h }
}

#[test]
fn re_framing_is_rigid_and_the_picture_does_not_move() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (96u32, 96u32);
    let s = scene(400);
    let cams = orbit(9);
    let (rot, centre) = frame_from_cameras(&cams, -1.0);
    let moved = apply(&s, &rot, &centre);

    assert_eq!(moved.len(), s.len());
    assert_eq!(moved.scales, s.scales, "re-framing must not touch a gaussian's shape");
    assert_eq!(moved.opacities, s.opacities, "re-framing must not touch opacity");
    assert_eq!(moved.colors, s.colors, "re-framing must not touch colour");
    let worst = (0..moved.len())
        .map(|i| (moved.quats[i * 4..i * 4 + 4].iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "re-framed quaternions are off unit length by {worst:.2e}");

    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, s.len(), w, h, s.len() * 16);
    let before = {
        let gs = GpuSplats::upload(&g, &s);
        ren.render(&g, &gs, &cam_from(&cams[3], w, h), &o);
        ren.read_rgba(&g, w, h)
    };
    let after = {
        let gs = GpuSplats::upload(&g, &moved);
        ren.render(&g, &gs, &cam_from(&transform_c2w(&cams[3], &rot, &centre), w, h), &o);
        ren.read_rgba(&g, w, h)
    };
    let db = psnr(&before, &after);
    assert!(
        db > 40.0,
        "the same view of a re-framed scene differs by {db:.1} dB - the transform is not rigid"
    );
}

/// And it has to actually straighten the scene, or it is an expensive no-op.
#[test]
fn the_scene_comes_out_with_its_orbit_axis_vertical() {
    let cams = orbit(9);
    let (rot, centre) = frame_from_cameras(&cams, -1.0);
    let eyes: Vec<[f64; 3]> = cams
        .iter()
        .map(|m| {
            let t = transform_c2w(m, &rot, &centre);
            [t[3], t[7], t[11]]
        })
        .collect();
    // every camera should now sit at nearly the same height
    let ys: Vec<f64> = eyes.iter().map(|e| e[1]).collect();
    let spread = ys.iter().cloned().fold(f64::MIN, f64::max) - ys.iter().cloned().fold(f64::MAX, f64::min);
    let radius = eyes.iter().map(|e| (e[0] * e[0] + e[2] * e[2]).sqrt()).sum::<f64>() / eyes.len() as f64;
    assert!(
        spread < 0.1 * radius,
        "after re-framing the cameras still vary {spread:.3} in height against a {radius:.3} orbit radius"
    );
    // and the subject should sit at the origin
    let cx = eyes.iter().map(|e| e[0]).sum::<f64>() / eyes.len() as f64;
    let cz = eyes.iter().map(|e| e[2]).sum::<f64>() / eyes.len() as f64;
    assert!(cx.hypot(cz) < 0.1 * radius, "the subject is not centred: ({cx:.3}, {cz:.3})");
}
