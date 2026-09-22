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
use splat::orient::{apply, frame_from_cameras, level, transform_c2w, upright};
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

/// A handheld sweep, not a lucky full ring: barely half an arc, and the
/// subject sits well off the plane the cameras travel in. Both are true of
/// real captures and both break plane fits that are not careful about which
/// point they fit around.
fn orbit(n: usize) -> Vec<[f64; 16]> {
    let target = [0.0, -0.05, 3.0];
    (0..n)
        .map(|i| {
            let a = 0.2 + std::f64::consts::PI * i as f64 / (n - 1) as f64;
            let e = [1.6 * a.cos(), -0.9, 3.0 + 1.6 * a.sin()];
            let f = {
                let d = [target[0] - e[0], target[1] - e[1], target[2] - e[2]];
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
    // and the subject should sit at the middle. Note that a partial arc's
    // centroid is nowhere near its centre, so what says the origin is right is
    // that every camera ends up the same distance out from it.
    let rs: Vec<f64> = eyes.iter().map(|e| e[0].hypot(e[2])).collect();
    let off = rs.iter().cloned().fold(f64::MIN, f64::max) - rs.iter().cloned().fold(f64::MAX, f64::min);
    assert!(off < 0.1 * radius, "the origin is not the subject: camera distances vary by {off:.3}");
}

/// The whole re-framing in one call, which is how every caller wants it.
///
/// The two halves have to move TOGETHER or the result is worse than not
/// straightening at all: a scene rotated into a new frame and handed back with
/// the cameras it had before is a scene nobody can render. Composing them by
/// hand at each call site is how that gets forgotten, so the composition is
/// the library's job and this is the gate on it.
#[test]
fn upright_moves_the_cameras_with_the_scene() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (96u32, 96u32);
    let s = scene(400);
    let mats = orbit(9);
    let cams: Vec<Camera> = mats.iter().map(|m| cam_from(m, w, h)).collect();

    let (moved, moved_cams) = upright(&s, &cams);
    assert_eq!(moved_cams.len(), cams.len(), "a camera went missing");

    // Same picture through the returned pair as through the originals.
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, s.len(), w, h, s.len() * 16);
    let before = {
        let gs = GpuSplats::upload(&g, &s);
        ren.render(&g, &gs, &cams[3], &o);
        ren.read_rgba(&g, w, h)
    };
    let after = {
        let gs = GpuSplats::upload(&g, &moved);
        ren.render(&g, &gs, &moved_cams[3], &o);
        ren.read_rgba(&g, w, h)
    };
    let db = psnr(&before, &after);
    assert!(db > 40.0, "upright() changed the picture by {db:.1} dB");

    // And it straightened it: the orbit is level in the frame it hands back.
    let ys: Vec<f64> = moved_cams.iter().map(|c| c.c2w[7] as f64).collect();
    let radius = moved_cams
        .iter()
        .map(|c| (c.c2w[3] as f64).hypot(c.c2w[11] as f64))
        .sum::<f64>()
        / moved_cams.len() as f64;
    let spread = ys.iter().cloned().fold(f64::MIN, f64::max) - ys.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread < 0.1 * radius, "upright() left {spread:.3} of height spread on a {radius:.3} orbit");
}

/// A ground plane of surface-aligned discs, tilted by `deg` about +X.
///
/// Discs, not balls: the whole signal this reads is a gaussian's SHORTEST
/// axis, which only means "surface normal" when the gaussian is flat.
fn tilted_ground(n: usize, deg: f64) -> Splats {
    let (s_, c_) = (deg.to_radians().sin(), deg.to_radians().cos());
    let mut r = Lcg(0x91d);
    let mut out = Splats::default();
    for _ in 0..n {
        let (x, z) = ((r.next() - 0.5) as f64 * 4.0, (r.next() - 0.5) as f64 * 4.0);
        // plane through the origin with normal (0, -c, s) after tilting about X
        let y = z * s_ / c_.max(1e-6) * c_;
        out.means.extend_from_slice(&[x as f32, (y * 0.0 + z * s_) as f32, (z * c_) as f32]);
        // the disc's local z on that normal: normal is (0, -cos, sin)
        // quaternion taking +z onto it
        let n3 = [0.0f64, -c_, s_];
        let (ax, ay) = (-n3[1], n3[0]);
        let sn = (ax * ax + ay * ay).sqrt();
        let ang = n3[2].clamp(-1.0, 1.0).acos();
        let (sh, ch) = ((ang * 0.5).sin(), (ang * 0.5).cos());
        let q = if sn < 1e-9 { [1.0, 0.0, 0.0, 0.0] } else { [ch, ax / sn * sh, ay / sn * sh, 0.0] };
        out.quats.extend_from_slice(&[q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]);
        out.scales.extend_from_slice(&[0.05, 0.05, 0.006]);
        out.opacities.push(0.9);
        out.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
    }
    out
}

/// Levelling reads the scene, not the path the photographer walked.
///
/// `frame_from_cameras` infers up from the camera ORBIT, which is only as
/// good as the orbit: a handheld arc that rises, or covers a third of a
/// circle, leaves the scene visibly tipped however rigid the transform was.
/// The surface the capture is OF is a much better witness - a reconstruction
/// aligns its gaussians to measured normals, so the dominant surface's normal
/// IS the up the viewer expects - and reading it costs no forward pass.
#[test]
fn levelling_straightens_a_scene_its_cameras_could_not() {
    for deg in [7.0f64, 18.0, 31.0] {
        let s = tilted_ground(6000, deg);
        // deliberately useless cameras: a short, rising arc
        let cams: Vec<Camera> = (0..4)
            .map(|i| {
                let a = i as f64 * 0.12;
                let m = [
                    1.0, 0.0, 0.0, 3.0 * a.sin(),
                    0.0, 1.0, 0.0, -1.0 - 0.4 * i as f64,
                    0.0, 0.0, 1.0, 3.0 * a.cos(),
                    0.0, 0.0, 0.0, 1.0,
                ];
                cam_from(&m, 64, 64)
            })
            .collect();

        let (levelled, _) = level(&s, &cams);
        assert_eq!(levelled.len(), s.len());
        // the dominant plane's normal must now be world up (-Y)
        let n = splat::orient::dominant_normal(&levelled, [0.0, -1.0, 0.0]).expect("a plane");
        let off = (n[1].abs()).clamp(0.0, 1.0).acos().to_degrees();
        assert!(off < 1.5, "a {deg} degree tilt was left {off:.2} degrees off level");
        assert_eq!(levelled.scales, s.scales, "levelling must not reshape a gaussian");
    }
}
