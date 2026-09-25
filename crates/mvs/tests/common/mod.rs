// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic captures with known geometry: analytic textured shapes, ray
//! cast through any `camera::Intrinsics`, so the stereo is checked against
//! exact ranges and normals through the same lens models it runs on.

#![allow(dead_code)] // each test binary uses its own part of the harness

use camera::{Intrinsics, Lens};
use data::rng::Lcg;
use sfm::linalg::{add, dot, norm, normalize, scale, sub, V3};
use splat::types::Camera;

/// An analytic surface.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// The square `|x - centre| <= half` on the plane through `centre` with
    /// normal `n`, spanned by `u` and `n x u`.
    Square { centre: V3, n: V3, u: V3, half: f64 },
    Sphere { centre: V3, r: f64 },
    /// The inside walls of an axis-aligned box.
    Room { lo: V3, hi: V3 },
}

/// A solid texture: a sum of sinusoids in 3D per channel, so every surface
/// is textured and the texture belongs to the point, not the view.
pub struct Texture {
    waves: Vec<[f64; 5]>, // direction (3), frequency (cycles per unit), phase
}

impl Texture {
    pub fn new(seed: u64, lo_freq: f64, hi_freq: f64) -> Texture {
        let mut g = Lcg::new(seed);
        let waves = (0..3 * 16)
            .map(|_| {
                let d = normalize([g.signed() as f64, g.signed() as f64, g.signed() as f64]);
                let f = lo_freq * (hi_freq / lo_freq).powf(g.unit() as f64);
                [d[0], d[1], d[2], f, std::f64::consts::TAU * g.unit() as f64]
            })
            .collect();
        Texture { waves }
    }

    pub fn rgb(&self, x: V3) -> [f64; 3] {
        std::array::from_fn(|c| {
            let s: f64 = self.waves[16 * c..16 * (c + 1)]
                .iter()
                .map(|w| (std::f64::consts::TAU * w[3] * dot([w[0], w[1], w[2]], x) + w[4]).sin())
                .sum();
            (0.5 + 0.12 * s).clamp(0.0, 1.0)
        })
    }
}

pub struct Scene {
    pub shapes: Vec<Shape>,
    pub tex: Texture,
}

/// Nearest hit of the ray `o + t d` (unit `d`): distance and unit normal
/// facing the ray.
pub fn hit(shapes: &[Shape], o: V3, d: V3) -> Option<(f64, V3)> {
    let mut best: Option<(f64, V3)> = None;
    let mut consider = |t: f64, n: V3| {
        if t > 1e-6 && best.is_none_or(|b| t < b.0) {
            let n = normalize(n);
            best = Some((t, if dot(n, d) > 0.0 { scale(n, -1.0) } else { n }));
        }
    };
    for s in shapes {
        match *s {
            Shape::Square { centre, n, u, half } => {
                let den = dot(n, d);
                if den.abs() < 1e-12 {
                    continue;
                }
                let t = dot(n, sub(centre, o)) / den;
                let x = sub(add(o, scale(d, t)), centre);
                let v = sfm::linalg::cross(n, u);
                if dot(x, u).abs() <= half && dot(x, v).abs() <= half {
                    consider(t, n);
                }
            }
            Shape::Sphere { centre, r } => {
                let oc = sub(o, centre);
                let b = dot(oc, d);
                let disc = b * b - (dot(oc, oc) - r * r);
                if disc < 0.0 {
                    continue;
                }
                for t in [-b - disc.sqrt(), -b + disc.sqrt()] {
                    if t > 1e-6 {
                        consider(t, sub(add(o, scale(d, t)), centre));
                        break;
                    }
                }
            }
            Shape::Room { lo, hi } => {
                for a in 0..3 {
                    if d[a].abs() < 1e-12 {
                        continue;
                    }
                    for wall in [lo[a], hi[a]] {
                        let t = (wall - o[a]) / d[a];
                        let x = add(o, scale(d, t));
                        let inside = (0..3).all(|b| b == a || (x[b] >= lo[b] && x[b] <= hi[b]));
                        if inside {
                            let mut n = [0.0; 3];
                            n[a] = 1.0;
                            consider(t, n);
                        }
                    }
                }
            }
        }
    }
    best
}

/// Distance from `x` to the nearest surface of `shapes`.
pub fn surface_distance(shapes: &[Shape], x: V3) -> f64 {
    shapes
        .iter()
        .map(|s| match *s {
            Shape::Square { centre, n, .. } => dot(n, sub(x, centre)).abs(),
            Shape::Sphere { centre, r } => (norm(sub(x, centre)) - r).abs(),
            Shape::Room { lo, hi } => (0..3).map(|a| (x[a] - lo[a]).abs().min((hi[a] - x[a]).abs())).fold(f64::INFINITY, f64::min),
        })
        .fold(f64::INFINITY, f64::min)
}

/// Unit normal of the surface of `shapes` nearest to `x` (either sign).
pub fn surface_normal(shapes: &[Shape], x: V3) -> V3 {
    let d = surface_distance(shapes, x);
    for s in shapes {
        match *s {
            Shape::Square { centre, n, .. } if (dot(n, sub(x, centre)).abs() - d).abs() < 1e-12 => return n,
            Shape::Sphere { centre, r } if ((norm(sub(x, centre)) - r).abs() - d).abs() < 1e-12 => return normalize(sub(x, centre)),
            Shape::Room { lo, hi } => {
                for a in 0..3 {
                    if ((x[a] - lo[a]).abs() - d).abs() < 1e-12 || ((hi[a] - x[a]).abs() - d).abs() < 1e-12 {
                        let mut n = [0.0; 3];
                        n[a] = 1.0;
                        return n;
                    }
                }
            }
            _ => {}
        }
    }
    unreachable!("the nearest surface is one of the shapes")
}

/// Camera-to-world rows of a camera at `eye` looking at `target`, +Y down in
/// the image with `up` pointing up.
pub fn look_at(eye: V3, target: V3, up: V3) -> [f32; 16] {
    let f = normalize(sub(target, eye));
    let r = normalize(sfm::linalg::cross(f, up));
    let dn = sfm::linalg::cross(f, r);
    [
        r[0] as f32, dn[0] as f32, f[0] as f32, eye[0] as f32,
        r[1] as f32, dn[1] as f32, f[1] as f32, eye[1] as f32,
        r[2] as f32, dn[2] as f32, f[2] as f32, eye[2] as f32,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// World ray of the camera through continuous pixel `px`.
pub fn world_ray(cam: &Camera, px: [f64; 2]) -> Option<(V3, V3)> {
    let d = cam.intrinsics().unproject(px)?;
    let m = cam.c2w.map(f64::from);
    let w = [m[0] * d[0] + m[1] * d[1] + m[2] * d[2], m[4] * d[0] + m[5] * d[1] + m[6] * d[2], m[8] * d[0] + m[9] * d[1] + m[10] * d[2]];
    Some(([m[3], m[7], m[11]], w))
}

/// The photograph `cam` takes: `ss x ss` supersampled, 8-bit RGB.
pub fn render(scene: &Scene, cam: &Camera, ss: u32) -> Vec<u8> {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 3];
            for sy in 0..ss {
                for sx in 0..ss {
                    let px = [x as f64 + (sx as f64 + 0.5) / ss as f64, y as f64 + (sy as f64 + 0.5) / ss as f64];
                    let Some((o, d)) = world_ray(cam, px) else { continue };
                    if let Some((t, _)) = hit(&scene.shapes, o, d) {
                        let c = scene.tex.rgb(add(o, scale(d, t)));
                        for k in 0..3 {
                            acc[k] += c[k];
                        }
                    }
                }
            }
            for k in 0..3 {
                out[(y * w + x) * 3 + k] = (acc[k] / (ss * ss) as f64 * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Exact range and camera-frame normal at every pixel centre of `cam`
/// (range 0 where the ray hits nothing).
pub fn truth(scene: &Scene, cam: &Camera) -> (Vec<f32>, Vec<[f64; 3]>) {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let m = cam.c2w.map(f64::from);
    let mut range = vec![0.0f32; w * h];
    let mut normal = vec![[0.0; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let Some((o, d)) = world_ray(cam, [x as f64 + 0.5, y as f64 + 0.5]) else { continue };
            if let Some((t, n)) = hit(&scene.shapes, o, d) {
                range[y * w + x] = t as f32;
                // world -> camera: R^T n
                normal[y * w + x] = [
                    m[0] * n[0] + m[4] * n[1] + m[8] * n[2],
                    m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
                    m[2] * n[0] + m[6] * n[1] + m[10] * n[2],
                ];
            }
        }
    }
    (range, normal)
}

/// Sparse tracks as structure from motion would give them: surface points
/// under random pixels of every view, with every view that sees each one
/// unoccluded.
pub fn tracks(scene: &Scene, cams: &[Camera], per_view: usize, seed: u64) -> Vec<mvs::Track> {
    let mut g = Lcg::new(seed);
    let mut out = Vec::new();
    for cam in cams {
        for _ in 0..per_view {
            let px = [g.unit() as f64 * cam.width as f64, g.unit() as f64 * cam.height as f64];
            let Some((o, d)) = world_ray(cam, px) else { continue };
            let Some((t, _)) = hit(&scene.shapes, o, d) else { continue };
            let x = add(o, scale(d, t));
            let views = cams
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    let e = [c.c2w[3] as f64, c.c2w[7] as f64, c.c2w[11] as f64];
                    let dir = sub(x, e);
                    let dist = norm(dir);
                    let m = c.c2w.map(f64::from);
                    let cam_dir = [
                        m[0] * dir[0] + m[4] * dir[1] + m[8] * dir[2],
                        m[1] * dir[0] + m[5] * dir[1] + m[9] * dir[2],
                        m[2] * dir[0] + m[6] * dir[1] + m[10] * dir[2],
                    ];
                    let inside = c.intrinsics().project(cam_dir).is_some_and(|p| p[0] >= 0.0 && p[1] >= 0.0 && p[0] < c.width as f64 && p[1] < c.height as f64);
                    inside && hit(&scene.shapes, e, scale(dir, 1.0 / dist)).is_some_and(|(t, _)| (t - dist).abs() < 1e-6 * dist.max(1.0))
                })
                .map(|(i, _)| i)
                .collect();
            out.push(mvs::Track { xyz: x, views });
        }
    }
    out
}

/// Accuracy of one map against the truth.
#[derive(Debug)]
pub struct Score {
    /// Measured pixels over pixels with a surface.
    pub coverage: f64,
    /// Of the measured pixels: within 0.5 % in range.
    pub within_half_percent: f64,
    /// Of the measured pixels: off by more than 2 %, or where there is no
    /// surface at all.
    pub outliers: f64,
    /// Median angle between measured and true normals, degrees.
    pub median_normal_deg: f64,
    /// Of the measured pixels: normal within 5 degrees.
    pub normal_within_5deg: f64,
}

pub fn score(map: &mvs::DepthMap, truth: &(Vec<f32>, Vec<[f64; 3]>)) -> Score {
    let (tr, tn) = truth;
    let surface = tr.iter().filter(|r| **r > 0.0).count().max(1);
    let (mut measured, mut good, mut bad, mut n5) = (0usize, 0usize, 0usize, 0usize);
    let mut angles = Vec::new();
    for i in 0..tr.len() {
        let r = map.range[i];
        if r <= 0.0 {
            continue;
        }
        measured += 1;
        if tr[i] <= 0.0 {
            bad += 1;
            continue;
        }
        let rel = ((r - tr[i]) / tr[i]).abs();
        good += (rel < 0.005) as usize;
        bad += (rel > 0.02) as usize;
        let n = [map.normal[3 * i] as f64, map.normal[3 * i + 1] as f64, map.normal[3 * i + 2] as f64];
        let a = dot(normalize(n), tn[i]).clamp(-1.0, 1.0).acos().to_degrees();
        n5 += (a < 5.0) as usize;
        angles.push(a);
    }
    angles.sort_by(f64::total_cmp);
    let m = measured.max(1) as f64;
    Score {
        coverage: measured as f64 / surface as f64,
        within_half_percent: good as f64 / m,
        outliers: bad as f64 / m,
        median_normal_deg: angles.get(angles.len() / 2).copied().unwrap_or(180.0),
        normal_within_5deg: n5 as f64 / m,
    }
}

/// A textured square with a sphere resting on it, seen from six cameras on
/// an arc - 50 degrees of azimuth, looking down at 40 degrees.
pub fn pinhole_capture() -> (Scene, Vec<Camera>) {
    let scene = Scene {
        shapes: vec![
            Shape::Square { centre: [0.0, 0.0, 0.0], n: [0.0, 0.0, 1.0], u: [1.0, 0.0, 0.0], half: 3.0 },
            Shape::Sphere { centre: [0.3, -0.2, 0.7], r: 0.7 },
        ],
        tex: Texture::new(7, 1.5, 8.0),
    };
    let k = Intrinsics::pinhole(300.0, 320, 240);
    let cams = (0..6)
        .map(|i| {
            let az = (-25.0f64 + 10.0 * i as f64).to_radians();
            let el = 40f64.to_radians();
            let eye = [4.5 * el.cos() * az.sin(), -4.5 * el.cos() * az.cos(), 4.5 * el.sin()];
            Camera::with_intrinsics(look_at(eye, [0.0, 0.0, 0.3], [0.0, 0.0, 1.0]), &k)
        })
        .collect();
    (scene, cams)
}

/// Inside a textured room with a sphere in it, through a fisheye seeing
/// ~160 degrees across, from six cameras on a small circle.
pub fn fisheye_capture() -> (Scene, Vec<Camera>) {
    let scene = Scene {
        shapes: vec![Shape::Room { lo: [-4.0, -4.0, 0.0], hi: [4.0, 4.0, 4.0] }, Shape::Sphere { centre: [0.0, 1.0, 1.5], r: 0.8 }],
        tex: Texture::new(11, 0.6, 3.0),
    };
    let k = Intrinsics { fx: 110.0, fy: 110.0, cx: 161.0, cy: 119.0, lens: Lens::Fisheye { k: [0.03, -0.01, 0.002, 0.0] }, width: 320, height: 240 };
    let cams = (0..6)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / 6.0;
            let eye = [0.8 * a.cos(), -2.0, 1.6 + 0.5 * a.sin()];
            Camera::with_intrinsics(look_at(eye, [0.2 * a.sin(), 1.0, 1.5], [0.0, 0.0, 1.0]), &k)
        })
        .collect();
    (scene, cams)
}

pub fn cfg() -> mvs::StereoCfg {
    mvs::StereoCfg {
        halving: 0,
        levels: 2,
        select: mvs::SelectCfg { max_sources: 5, min_shared: 8, ..Default::default() },
        ..Default::default()
    }
}

pub fn run(scene: &Scene, cams: &[Camera]) -> mvs::Stereo {
    let images: Vec<Vec<u8>> = cams.iter().map(|c| render(scene, c, 3)).collect();
    let views: Vec<mvs::View> = cams.iter().zip(&images).map(|(c, rgb)| mvs::View { cam: *c, rgb }).collect();
    let tr = tracks(scene, cams, 150, 3);
    let gpu = gpu_core::testgpu::dev(mvs::PIPELINES);
    mvs::depth_maps(&gpu, &mvs::Kernels::at(0), &views, &tr, &cfg()).expect("stereo")
}
