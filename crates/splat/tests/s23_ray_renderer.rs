// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ray-evaluated renderer (`splat_ray_*.wgsl`) against its definition.
//!
//! 1. The forward reproduces `splat::reference::render_ray`, an f64 oracle
//!    that evaluates every gaussian at every pixel along the lens's own rays -
//!    through a pinhole, an OpenCV Brown lens, a Kannala-Brandt fisheye and a
//!    rolling shutter, with and without the 2D and 3D Mip filters. The oracle
//!    has no tiles, so an image-space bound that clips a visible gaussian is a
//!    difference here rather than a reproduced mistake.
//! 2. For a pinhole camera and small, distant gaussians - where the EWA
//!    linearization is exact in the limit - the ray renderer and the EWA
//!    renderer draw the same image.
//! 3. Every gradient the backward returns - gaussian parameters through
//!    colour, alpha, expected range, normal and distortion, and the camera's
//!    pose, rolling shutter and calibration - agrees with a central
//!    difference of the device forward.
//!
//! Swedish Embedded AB implements differentiable renderers for 3D
//! reconstruction that image through real lenses. If your team needs
//! expertise in gaussian splatting then you can procure our services by
//! sending an email to info@swedishembedded.com.

use camera::Lens;
use data::rng::Lcg;
use gpu_core::Gpu;
use splat::renderer::{BwdScratch, CameraGrad, GpuSplats, Renderer, SplatGrads};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn scene(n: usize, seed: u64, depth: f32, size: f32) -> Splats {
    let mut r = Lcg::new(seed);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[r.signed() * 0.9 * depth / 3.0, r.signed() * 0.7 * depth / 3.0, depth + r.unit() * 1.5]);
        s.quats.extend_from_slice(&[0.5 + r.unit(), r.signed() * 0.5, r.signed() * 0.5, r.signed() * 0.5]);
        s.scales.extend_from_slice(&[size * (0.4 + r.unit()), size * (0.4 + r.unit()), size * (0.1 + 0.5 * r.unit())]);
        s.opacities.push(0.3 + 0.4 * r.unit());
        s.colors.extend_from_slice(&[r.unit(), r.unit(), r.unit()]);
    }
    s
}

fn base_cam(w: u32, h: u32) -> Camera {
    let mut c = Camera::look_at([0.1, -0.05, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 70.0, w, h);
    c.cx += 0.37;
    c.cy -= 0.21;
    c
}

fn cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let pin = base_cam(w, h);
    let brown = Camera { lens: Lens::Brown { k: [-0.12, 0.04, -0.01, 0.0, 0.0, 0.0], p: [8e-4, -5e-4], s: [0.0; 4] }, ..pin };
    let fish = Camera { fx: pin.fx * 0.8, fy: pin.fy * 0.8, lens: Lens::Fisheye { k: [0.03, -0.01, 0.002, 0.0] }, ..pin };
    let rs = Camera { shutter: [0.04, -0.03, 0.02, 0.05, 0.02, -0.03], ..brown };
    vec![("pinhole", pin), ("brown", brown), ("fisheye", fish), ("rolling shutter", rs)]
}

fn device_render(g: &Gpu, s: &Splats, filt: &[f32], cam: &Camera, o: &RenderOpts) -> (Vec<f32>, Vec<f32>) {
    let mut ren = Renderer::new(g, Kernels::at(0), s.len(), cam.width, cam.height, 0).growable();
    let gs = GpuSplats::upload(g, s).with_filter3d(g, filt);
    ren.render(g, &gs, cam, &RenderOpts { ray: true, ..*o });
    (ren.read_rgba(g, cam.width, cam.height), ren.read_aux(g, cam.width, cam.height))
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    a.iter().zip(b).enumerate().map(|(i, (x, y))| ((x - y).abs(), i)).fold((0.0, 0), |m, v| if v.0 > m.0 { v } else { m })
}

#[test]
fn the_forward_is_the_oracle_through_every_lens() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let s = scene(40, 0x1a2b, 3.0, 0.25);
    let mut r = Lcg::new(7);
    let filt: Vec<f32> = (0..s.len()).map(|_| 1e-4 * r.unit()).collect();
    for (name, cam) in cameras(48, 36) {
        for aa in [false, true] {
            for filt in [vec![0.0; s.len()], filt.clone()] {
                let o = RenderOpts { antialiased: aa, eps2d: 0.3, bg: [0.1, 0.2, 0.3], ..Default::default() };
                let (img, aux) = device_render(&g, &s, &filt, &cam, &o);
                let (ri, ra) = splat::reference::render_ray(&s, &filt, &cam, &o);
                let (di, at) = max_diff(&img, &ri);
                assert!(di < 2e-4, "{name} aa {aa}: rgba differs by {di} at pixel {} channel {}", at / 4, at % 4);
                // range scales with the scene; the normal sum and distortion
                // are O(1)
                for c in 0..5 {
                    let (dv, at) = max_diff(&aux.iter().skip(c).step_by(5).copied().collect::<Vec<_>>(), &ra.iter().skip(c).step_by(5).copied().collect::<Vec<_>>());
                    let tol = if c == 0 { 2e-3 } else { 5e-4 };
                    assert!(dv < tol, "{name} aa {aa}: aux channel {c} differs by {dv} at pixel {at}");
                }
                assert!(img.chunks_exact(4).filter(|p| p[3] > 0.05).count() > 120, "{name}: the scene barely covers the frame");
            }
        }
    }
}

#[test]
fn small_distant_gaussians_render_as_their_ewa_splats() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let s = scene(60, 0x55, 8.0, 0.05);
    let cam = base_cam(64, 48);
    let o = RenderOpts { antialiased: false, eps2d: 0.3, ..Default::default() };
    let mut ren = Renderer::new(&g, Kernels::at(0), s.len(), cam.width, cam.height, 0);
    let gs = GpuSplats::upload(&g, &s);
    ren.render(&g, &gs, &cam, &o);
    let ewa = ren.read_rgba(&g, cam.width, cam.height);
    let (ray, _) = device_render(&g, &s, &vec![0.0; s.len()], &cam, &o);
    let rgb = |v: &[f32]| v.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect::<Vec<f32>>();
    let psnr = splat::quality::psnr(&rgb(&ewa), &rgb(&ray));
    assert!(psnr > 38.0, "EWA and ray evaluation of small distant gaussians differ: {psnr:.1} dB");
}

/// A fixed random linear functional of everything the forward outputs.
struct Probe {
    wimg: Vec<f32>,
    waux: Vec<f32>,
}

impl Probe {
    /// Random weights on every output; the expected range only where the
    /// frame is opaque enough for it to be supervised at all
    /// (`splat::renderer::MIN_DEPTH_ALPHA`, with a margin a finite difference
    /// cannot step across), which is where its backward is defined.
    fn new(alpha: &[f32], seed: u64) -> Probe {
        let px = alpha.len();
        let mut r = Lcg::new(seed);
        let mask: Vec<f32> = std::env::var("PROBE_MASK").map(|m| m.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or(vec![1.0; 9]);
        let wimg = (0..px * 4).map(|i| r.signed() * mask[i % 4]).collect();
        let waux = (0..px * 5)
            .map(|i| {
                let w = r.signed() * mask[4 + i % 5];
                match i % 5 {
                    0 if alpha[i / 5] < 4.0 * splat::renderer::MIN_DEPTH_ALPHA => 0.0,
                    0 => 0.05 * w,
                    _ => 0.5 * w,
                }
            })
            .collect();
        Probe { wimg, waux }
    }

    fn loss(&self, img: &[f32], aux: &[f32]) -> f64 {
        img.iter().zip(&self.wimg).map(|(a, b)| (a * b) as f64).sum::<f64>() + aux.iter().zip(&self.waux).map(|(a, b)| (a * b) as f64).sum::<f64>()
    }

    /// dL/d(expected range) enters the backward through the accumulated range
    /// and the alpha, which is what `add_expected_depth_vjp` does.
    fn upstream(&self, img: &[f32], aux: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let px = img.len() / 4;
        let mut dimg = self.wimg.clone();
        let mut daux = self.waux.clone();
        let vdn: Vec<f32> = (0..px).map(|i| self.waux[i * 5]).collect();
        let depth: Vec<f32> = (0..px).map(|i| aux[i * 5]).collect();
        let mut dacc = vec![0.0f32; px];
        splat::renderer::add_expected_depth_vjp(&vdn, &depth, img, &mut dimg, &mut dacc);
        for i in 0..px {
            daux[i * 5] = dacc[i];
        }
        (dimg, daux)
    }
}

/// A probe for what `s` renders from `cam`.
fn probe_for(g: &Gpu, s: &Splats, filt: &[f32], cam: &Camera, o: &RenderOpts, seed: u64) -> Probe {
    let (img, _) = device_render(g, s, filt, cam, o);
    Probe::new(&img.chunks_exact(4).map(|p| p[3]).collect::<Vec<f32>>(), seed)
}

struct Analytic {
    d_gauss: Vec<f32>,
    d_opac: Vec<f32>,
    d_colors: Vec<f32>,
    camera: CameraGrad,
}

fn analytic(g: &Gpu, s: &Splats, filt: &[f32], cam: &Camera, o: &RenderOpts, probe: &Probe) -> Analytic {
    let o = RenderOpts { ray: true, ..*o };
    let mut ren = Renderer::new(g, Kernels::at(0), s.len(), cam.width, cam.height, 0).growable();
    let gs = GpuSplats::upload(g, s).with_filter3d(g, filt);
    ren.render(g, &gs, cam, &o);
    let img = ren.read_rgba(g, cam.width, cam.height);
    let aux = ren.read_aux(g, cam.width, cam.height);
    let (dimg, daux) = probe.upstream(&img, &aux);
    let grads = SplatGrads::new(g, s.len());
    g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors, &grads.d_absgrad, &grads.d_sumgrad], &[]);
    let (dimg, daux) = (g.storage_init("dimg", &dimg), g.storage_init("daux", &daux));
    let px = (cam.width * cam.height) as usize;
    let mut scr = BwdScratch::new(g, s.len(), px, 0);
    let camera = ren.render_bwd_ray(g, &gs, cam, &o, &dimg, Some(&daux), &mut scr, &grads, true).expect("fits").expect("asked for");
    Analytic {
        d_gauss: g.read(&grads.d_gauss, 10 * s.len()),
        d_opac: g.read(&grads.d_opac, s.len()),
        d_colors: g.read(&grads.d_colors, 3 * s.len()),
        camera,
    }
}

/// Compares analytic gradients with central differences of the device
/// forward - only where the difference has CONVERGED.
///
/// The forward is not smooth everywhere, by definition: a pair whose alpha
/// crosses 1/255 appears or vanishes, and at a small image a single crossing
/// moves the probe by more than the whole gradient over a finite step. So
/// every entry is differenced at two steps, `h` and `h/3`, and only entries
/// whose two differences agree are a measurement of the derivative; the rest
/// straddle a crossing and are counted as skipped, which must stay a minority.
///
/// `floor` is the gradient magnitude below which agreement is judged in
/// absolute terms: an f32 forward summed over hundreds of pixels differences
/// to about `1e-2 * floor` at these steps, so a relative test of a smaller
/// gradient measures the noise rather than the derivative.
struct Checker {
    worst: (f64, String),
    checked: usize,
    skipped: usize,
    rows: Vec<(f64, String)>,
}

impl Checker {
    fn new() -> Checker {
        Checker { worst: (0.0, String::new()), checked: 0, skipped: 0, rows: Vec::new() }
    }

    /// `loss(x)` is the probe with the parameter moved by `x`.
    fn check(&mut self, what: String, an: f64, loss: &dyn Fn(f64) -> f64, h: f64, floor: f64) {
        let fd = |h: f64| (loss(h) - loss(-h)) / (2.0 * h);
        let (coarse, fine) = (fd(h), fd(h / 3.0));
        if coarse.abs() < floor && fine.abs() < floor && an.abs() < floor {
            return;
        }
        if (coarse - fine).abs() > 0.03 * coarse.abs().max(fine.abs()).max(floor) {
            self.skipped += 1;
            return;
        }
        let rel = (an - fine).abs() / an.abs().max(fine.abs()).max(floor);
        self.checked += 1;
        let row = format!("{what}: analytic {an:.6} vs finite difference {fine:.6}");
        if rel > self.worst.0 {
            self.worst = (rel, row.clone());
        }
        self.rows.push((rel, row));
    }

    fn assert(mut self, min: usize, tol: f64) {
        assert!(self.checked >= min, "only {} gradients measured ({} skipped at a discontinuity)", self.checked, self.skipped);
        assert!(self.skipped <= self.checked, "{} of {} entries straddle a discontinuity: the scene is not a gradient test", self.skipped, self.skipped + self.checked);
        if self.worst.0 >= tol {
            self.rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (r, m) in self.rows.iter().take(15) {
                println!("  {:6.2}%  {m}", 100.0 * r);
            }
        }
        assert!(self.worst.0 < tol, "worst disagreement {:.2}%: {}", 100.0 * self.worst.0, self.worst.1);
    }
}

#[test]
fn every_gaussian_gradient_matches_a_central_difference() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    for (name, cam) in cameras(28, 22) {
        let s = scene(7, 0x9e3 ^ name.len() as u64, 3.0, 0.22);
        let mut r = Lcg::new(3);
        let filt: Vec<f32> = (0..s.len()).map(|_| 2e-4 * r.unit()).collect();
        let o = RenderOpts { antialiased: true, eps2d: 0.3, bg: [0.2, 0.1, 0.4], ..Default::default() };
        let probe = probe_for(&g, &s, &filt, &cam, &o, 0xabc);
        let an = analytic(&g, &s, &filt, &cam, &o, &probe);
        let lossof = |s: &Splats| {
            let (img, aux) = device_render(&g, s, &filt, &cam, &o);
            probe.loss(&img, &aux)
        };
        let mut ck = Checker::new();
        for i in 0..s.len() {
            let at = |edit: &dyn Fn(&mut Splats, f32), x: f64| {
                let mut t = s.clone();
                edit(&mut t, x as f32);
                lossof(&t)
            };
            for k in 0..3 {
                ck.check(format!("{name} mean[{i}][{k}]"), an.d_gauss[i * 10 + k] as f64, &|x| at(&|t, v| t.means[i * 3 + k] += v, x), 1e-3, 5e-2);
                ck.check(format!("{name} scale[{i}][{k}]"), an.d_gauss[i * 10 + 3 + k] as f64, &|x| at(&|t, v| t.scales[i * 3 + k] += v, x), 1e-3, 5e-2);
                ck.check(format!("{name} colour[{i}][{k}]"), an.d_colors[i * 3 + k] as f64, &|x| at(&|t, v| t.colors[i * 3 + k] += v, x), 1e-2, 1e-3);
            }
            for k in 0..4 {
                ck.check(format!("{name} quat[{i}][{k}]"), an.d_gauss[i * 10 + 6 + k] as f64, &|x| at(&|t, v| t.quats[i * 4 + k] += v, x), 2e-3, 5e-2);
            }
            ck.check(format!("{name} opacity[{i}]"), an.d_opac[i] as f64, &|x| at(&|t, v| t.opacities[i] += v, x), 1e-3, 5e-2);
        }
        ck.assert(60, 0.01);
    }
}

/// Move a camera by `(omega, tau)` in its own frame: `c2w · [Exp(omega) | tau]`.
fn moved(c: &Camera, omega: [f64; 3], tau: [f64; 3]) -> Camera {
    let th = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let k = if th > 0.0 { omega.map(|v| v / th) } else { [0.0; 3] };
    let (c_, s_) = (th.cos(), th.sin());
    let r: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
            let id = if i == j { 1.0 } else { 0.0 };
            id * c_ + s_ * kx[i][j] + (1.0 - c_) * k[i] * k[j]
        })
    });
    let mut out = *c;
    for (row, dst) in c.c2w.chunks_exact(4).zip(out.c2w.chunks_exact_mut(4)).take(3) {
        let rot = |col: &dyn Fn(usize) -> f64| (0..3).map(|t| row[t] as f64 * col(t)).sum::<f64>();
        for (j, d) in dst.iter_mut().take(3).enumerate() {
            *d = rot(&|t| r[t][j]) as f32;
        }
        dst[3] = (row[3] as f64 + rot(&|t| tau[t])) as f32;
    }
    out
}

#[test]
fn camera_gradients_match_central_differences() {
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    let mut ck = Checker::new();
    for (name, cam) in cameras(28, 22) {
        let s = scene(9, 0x77 ^ name.len() as u64, 3.0, 0.2);
        let filt = vec![if std::env::var("NOFILT").is_ok() { 0.0 } else { 5e-5f32 }; s.len()];
        let o = RenderOpts { antialiased: std::env::var("NOAA").is_err(), eps2d: 0.3, bg: [0.3, 0.1, 0.2], ..Default::default() };
        let probe = probe_for(&g, &s, &filt, &cam, &o, 0x5151);
        let an = analytic(&g, &s, &filt, &cam, &o, &probe);
        let lossof = |c: &Camera| {
            let (img, aux) = device_render(&g, &s, &filt, c, &o);
            probe.loss(&img, &aux)
        };
        for k in 0..3 {
            let axis = |x: f64| {
                let mut e = [0.0; 3];
                e[k] = x;
                e
            };
            ck.check(format!("{name} rotation[{k}]"), an.camera.rotation[k], &|x| lossof(&moved(&cam, axis(x), [0.0; 3])), 3e-4, 3e-1);
            ck.check(format!("{name} translation[{k}]"), an.camera.translation[k], &|x| lossof(&moved(&cam, [0.0; 3], axis(x))), 3e-4, 3e-1);
        }
        if cam.shutter.iter().any(|v| *v != 0.0) {
            for k in 0..6 {
                let at = |x: f64| {
                    let mut c = cam;
                    c.shutter[k] += x as f32;
                    lossof(&c)
                };
                ck.check(format!("{name} shutter[{k}]"), an.camera.shutter[k], &at, 3e-4, 3e-1);
            }
        }
        let k0 = cam.intrinsics();
        let p0 = k0.params();
        for (i, pname) in k0.param_names().iter().enumerate() {
            let at = |x: f64| {
                let mut p = p0.clone();
                p[i] += x;
                lossof(&Camera { shutter: cam.shutter, ..Camera::with_intrinsics(cam.c2w, &k0.with_params(&p)) })
            };
            ck.check(format!("{name} {pname}"), an.camera.lens[i], &at, if i < 4 { 1e-2 } else { 1e-3 }, 3e-1);
        }
    }
    ck.assert(30, 0.02);
}

