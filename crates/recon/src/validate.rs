// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Whether a reconstruction is right in 3D, and not only at its photographs.
//!
//! A splat scene can reproduce every training photograph and still be a
//! stack of translucent gaussians cooperating along each training ray: the
//! objective `sum_v L(R(G, C_v), I_v)` has many minimizers `G`, and they
//! part ways as soon as a camera leaves the rays that constrained them. So a
//! capture is judged here by what the training views cannot show:
//!
//! - **Held-out designs by angle** ([`Split`], [`Orbit`]): views held out
//!   singly between their neighbours (interpolation), whole azimuth wedges
//!   (wide interpolation and extrapolation across the gap) and whole
//!   elevation bands (extrapolation in elevation), each held-out camera
//!   with its angular [`deviation`] from everything trained on.
//! - **What each pixel is made of** ([`Viewer::diagnose`]): the ray
//!   renderer's per-pixel statistics (`splat::renderer::Renderer::diagnose`:
//!   median range, range spread, contribution entropy, dominant share) and,
//!   per pixel, the observational support of the surface it renders -
//!   which training views measured the same surface there, from how far
//!   away in angle, and which saw empty space where it renders one
//!   (`splat_view_support.wgsl`, through multi-view stereo's own
//!   consistency test).
//! - **Error against support** ([`Diagnosis::error_by_support`]): a
//!   held-out photograph's error binned by how far each pixel's surface is
//!   from anything observed. Quality as a function of view deviation is the
//!   curve that says whether novel views fail because they are novel or
//!   because the geometry is wrong.
//!
//! Swedish Embedded AB implements 3D reconstruction whose output is
//! validated in 3D - held-out trajectories, observational support and
//! surface statistics - not only against its own photographs. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::DeviceBuffer;
use splat::renderer::{ray_view_params, Renderer};
use splat::types::{Camera, RenderOpts};

use crate::eval::Viewer;

/// Floats per pixel of [`Diagnosis::pixels`].
pub const DIAG: usize = Renderer::DIAG_WORDS;

/// Floats per pixel of [`Diagnosis::support`] (`splat_view_support.wgsl`).
pub const SUPPORT: usize = 4;

/// Upper edges, in degrees, of the support bands [`SurfaceStats::support`]
/// and [`Diagnosis::error_by_support`] use; the last band is everything
/// beyond, and pixels no training view supports are counted apart.
pub const SUPPORT_BANDS: [f32; 4] = [5.0, 15.0, 30.0, 60.0];

/// Every training view's rendered surface, on the device: the scene's median
/// range along each pixel's ray where the pixel is at least half covered, 0
/// elsewhere - what [`Viewer::diagnose`] tests a view's surface against.
pub struct Observed {
    cams: Vec<Camera>,
    ranges: DeviceBuffer,
    plane: usize,
}

impl Observed {
    pub fn cameras(&self) -> &[Camera] {
        &self.cams
    }
}

/// One view's render, what its pixels are made of and how well observed
/// their surfaces are.
#[derive(Clone, Debug)]
pub struct Diagnosis {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB, as [`Viewer::render`] gives it.
    pub rgb: Vec<f32>,
    /// `[W*H*DIAG]`, as `Renderer::diagnose` documents it.
    pub pixels: Vec<f32>,
    /// `[W*H*SUPPORT]`: supporting views, the smallest angle to one (deg,
    /// -1 = none), free-space violations, the supporting directions'
    /// angular radius (deg). Zero where the view was diagnosed without
    /// training views.
    pub support: Vec<f32>,
}

/// A view's surface statistics over the pixels it covers (alpha >= 1/2).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceStats {
    /// Fraction of the frame covered.
    pub covered: f64,
    /// Mean range spread as a fraction of the median range: 0 for a
    /// surface, large for a stack.
    pub spread: f64,
    /// Mean contribution entropy (nats).
    pub entropy: f64,
    /// Mean dominant share.
    pub share: f64,
    /// Fraction of covered pixels no training view supports, then in each
    /// band of [`SUPPORT_BANDS`] and beyond the last.
    pub support: [f64; 6],
    /// Fraction of covered pixels some training view saw empty space at.
    pub free_space: f64,
    /// Mean number of supporting training views.
    pub views: f64,
}

/// A held-out photograph's error in one support band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandError {
    /// The band's upper edge in degrees (`f32::INFINITY` for the last),
    /// `-1` for pixels no training view supports, `-2` for pixels the scene
    /// does not cover.
    pub upper: f32,
    pub pixels: usize,
    pub mean_abs: f64,
    pub psnr: f64,
}

fn band_of(angle: f32) -> usize {
    SUPPORT_BANDS.iter().position(|&u| angle < u).unwrap_or(SUPPORT_BANDS.len()) + 1
}

impl Diagnosis {
    fn covered(&self, p: usize) -> bool {
        self.pixels[p * DIAG] >= 0.5
    }

    /// [`SurfaceStats`] over the covered pixels (and those `mask` keeps,
    /// when given).
    pub fn stats(&self, mask: Option<&[f32]>) -> SurfaceStats {
        let n = (self.width * self.height) as usize;
        let mut s = SurfaceStats::default();
        let mut cov = 0usize;
        for p in 0..n {
            if !self.covered(p) || mask.is_some_and(|m| m[p] <= 0.0) {
                continue;
            }
            cov += 1;
            let d = &self.pixels[p * DIAG..p * DIAG + DIAG];
            let range = if d[2] > 0.0 { d[2] } else { d[1] };
            s.spread += (d[3] / range.max(1e-12)) as f64;
            s.entropy += d[4] as f64;
            s.share += d[6] as f64;
            let v = &self.support[p * SUPPORT..p * SUPPORT + SUPPORT];
            s.views += v[0] as f64;
            s.support[if v[0] > 0.0 { band_of(v[1]) } else { 0 }] += 1.0;
            s.free_space += (v[2] > 0.0) as u32 as f64;
        }
        let masked = mask.map_or(n, |m| m.iter().filter(|&&v| v > 0.0).count());
        s.covered = cov as f64 / masked.max(1) as f64;
        let c = cov.max(1) as f64;
        s.spread /= c;
        s.entropy /= c;
        s.share /= c;
        s.views /= c;
        s.free_space /= c;
        s.support.iter_mut().for_each(|v| *v /= c);
        s
    }

    /// The error of this render against `photo` (interleaved RGB of the
    /// same view) binned by each pixel's support, over the pixels `mask`
    /// keeps.
    pub fn error_by_support(&self, photo: &[f32], mask: Option<&[f32]>) -> Vec<BandError> {
        let n = (self.width * self.height) as usize;
        // uncovered, unsupported, then the bands
        let mut acc = vec![(0usize, 0.0f64, 0.0f64); SUPPORT_BANDS.len() + 3];
        for p in 0..n {
            if mask.is_some_and(|m| m[p] <= 0.0) {
                continue;
            }
            let v = &self.support[p * SUPPORT..p * SUPPORT + SUPPORT];
            let slot = if !self.covered(p) {
                0
            } else if v[0] <= 0.0 {
                1
            } else {
                1 + band_of(v[1])
            };
            let (abs, sq) = (0..3).fold((0.0f64, 0.0f64), |(a, q), c| {
                let e = (self.rgb[p * 3 + c] - photo[p * 3 + c]) as f64;
                (a + e.abs(), q + e * e)
            });
            let e = &mut acc[slot];
            e.0 += 1;
            e.1 += abs / 3.0;
            e.2 += sq / 3.0;
        }
        acc.iter()
            .enumerate()
            .map(|(i, &(px, abs, sq))| BandError {
                upper: match i {
                    0 => -2.0,
                    1 => -1.0,
                    i if i - 2 < SUPPORT_BANDS.len() => SUPPORT_BANDS[i - 2],
                    _ => f32::INFINITY,
                },
                pixels: px,
                mean_abs: if px > 0 { abs / px as f64 } else { 0.0 },
                psnr: if px > 0 && sq > 0.0 { 10.0 * (px as f64 / sq).log10() } else { f64::INFINITY },
            })
            .collect()
    }

    /// The diagnosis as images: `(name, image)` for the render, the median
    /// range (inverse, turbo), range spread relative to range, contribution
    /// entropy, the unit normal, the smallest support angle (turbo, black =
    /// unsupported) and free-space violations.
    pub fn images(&self) -> Vec<(&'static str, imaging::Rgb8)> {
        use imaging::viz::{colorize, Bounds, Colormap};
        let (w, h) = (self.width, self.height);
        let n = (w * h) as usize;
        let chan = |f: &dyn Fn(usize) -> f32| -> Vec<f32> { (0..n).map(f).collect() };
        let img = |px: Vec<u8>| imaging::Rgb8 { w, h, px };
        let masked = |v: Vec<f32>, bounds: Bounds, map: Colormap| -> Vec<u8> {
            let mut px = colorize(&v, bounds, map);
            for p in 0..n {
                if !v[p].is_finite() {
                    px[p * 3..p * 3 + 3].fill(0);
                }
            }
            px
        };
        let cov = |p: usize| self.covered(p);
        let inv = chan(&|p| {
            let d = &self.pixels[p * DIAG..];
            let r = if d[2] > 0.0 { d[2] } else { d[1] };
            if cov(p) && r > 0.0 { 1.0 / r } else { f32::NAN }
        });
        let finite = |v: &[f32]| v.iter().copied().filter(|x| x.is_finite()).collect::<Vec<f32>>();
        let spread = chan(&|p| {
            let d = &self.pixels[p * DIAG..];
            let r = if d[2] > 0.0 { d[2] } else { d[1] };
            if cov(p) { d[3] / r.max(1e-12) } else { f32::NAN }
        });
        let entropy = chan(&|p| if cov(p) { self.pixels[p * DIAG + 4] } else { f32::NAN });
        let angle = chan(&|p| {
            let v = &self.support[p * SUPPORT..];
            if cov(p) && v[0] > 0.0 { v[1] } else { f32::NAN }
        });
        let violations = chan(&|p| if cov(p) { self.support[p * SUPPORT + 2] } else { f32::NAN });
        let normal: Vec<u8> = (0..n)
            .flat_map(|p| {
                let d = &self.pixels[p * DIAG + 8..p * DIAG + 11];
                if cov(p) { [0, 1, 2].map(|k| ((d[k] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8) } else { [0; 3] }
            })
            .collect();
        vec![
            ("rgb", img(self.rgb.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect())),
            ("depth", img(masked(inv.clone(), Bounds::from_percentiles(&finite(&inv), 0.02, 0.98), Colormap::Turbo))),
            ("spread", img(masked(spread, Bounds { lo: 0.0, hi: 0.1 }, Colormap::Turbo))),
            ("entropy", img(masked(entropy, Bounds { lo: 0.0, hi: 3.0 }, Colormap::Turbo))),
            ("normal", img(normal)),
            ("support", img(masked(angle, Bounds { lo: 0.0, hi: 60.0 }, Colormap::Turbo))),
            ("violations", img(masked(violations, Bounds { lo: 0.0, hi: 4.0 }, Colormap::Turbo))),
        ]
    }
}

impl Viewer {
    fn support_kernel(&self) -> usize {
        self.gpu
            .kernel_index("splat_view_support")
            .expect("the device was not built with recon::photogrammetry::pipelines(): splat_view_support is missing")
    }

    /// What each of `cams` observes of the scene: its rendered surface range
    /// per pixel ([`Observed`]). The cameras must share one size.
    pub fn observe(&mut self, cams: &[Camera]) -> Observed {
        assert!(!cams.is_empty(), "observe: no cameras");
        let plane = (cams[0].width * cams[0].height) as usize;
        let mut ranges = Vec::with_capacity(plane * cams.len());
        for c in cams {
            assert_eq!((c.width * c.height) as usize, plane, "observe: the cameras must share one size");
            self.render(c);
            let d = self.renderer.diagnose(&self.gpu, c, &self.opts);
            ranges.extend((0..plane).map(|p| {
                let x = &d[p * DIAG..p * DIAG + DIAG];
                let r = if x[2] > 0.0 { x[2] } else { x[1] };
                if x[0] >= 0.5 { r } else { 0.0 }
            }));
        }
        Observed { cams: cams.to_vec(), ranges: self.gpu.storage_init("validate.ranges", &ranges), plane }
    }

    /// Render `cam` and diagnose it against `observed` (the training views;
    /// `skip` leaves one of them out - the view itself, when a training view
    /// is judged against the others).
    pub fn diagnose(&mut self, cam: &Camera, observed: Option<&Observed>, skip: Option<usize>) -> Diagnosis {
        let rgb = self.render(cam);
        let pixels = self.renderer.diagnose(&self.gpu, cam, &self.opts);
        let n = (cam.width * cam.height) as usize;
        let support = match observed {
            None => vec![0.0; n * SUPPORT],
            Some(obs) => gpu_core::reclaiming(&self.gpu, || self.support(cam, &pixels, obs, skip)),
        };
        Diagnosis { width: cam.width, height: cam.height, rgb, pixels, support }
    }

    fn support(&self, cam: &Camera, pixels: &[f32], obs: &Observed, skip: Option<usize>) -> Vec<f32> {
        let g = &self.gpu;
        let n = (cam.width * cam.height) as usize;
        // every source relative to `cam`; the kernel leaves out `skip`
        let words: Vec<u32> = obs.cams.iter().flat_map(|c| mvs::cam_record(cam, c)).collect();
        let cams = g.storage(words.len() as u64);
        g.write(&cams, &words);
        let diag = g.storage_init("validate.diag", pixels);
        let out = g.storage((n * SUPPORT) as u64);
        let mut params = ray_view_params(0, cam, &RenderOpts { ray: true, ..self.opts }).to_vec();
        let skip = skip.map_or(u32::MAX, |k| k as u32);
        params.extend_from_slice(&[obs.cams.len() as u32, obs.plane as u32, skip, 0, gpu_core::f(1.0), gpu_core::f(0.01), 0, 0]);
        let step = g.step(self.support_kernel(), &[&diag, &cams, &obs.ranges, &out], &params, n as u32);
        g.submit(&[], &[step]);
        g.read(&out, n * SUPPORT)
    }
}

/// The capture's orbit: the point every view looks towards and a frame whose
/// Y axis is the orbit's axis (`splat::orient::frame_from_cameras`).
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub centre: [f64; 3],
    /// Row-major rotation into the orbit frame.
    pub r: [f64; 9],
}

impl Orbit {
    pub fn of(cams: &[Camera]) -> Orbit {
        let mats: Vec<[f64; 16]> = cams.iter().map(|c| std::array::from_fn(|i| c.c2w[i] as f64)).collect();
        let (r, centre) = splat::orient::frame_from_cameras(&mats, -1.0);
        Orbit { centre, r }
    }

    /// `cam`'s azimuth about the orbit axis and elevation above the orbit's
    /// plane, degrees.
    pub fn angles(&self, cam: &Camera) -> (f64, f64) {
        let e = cam.eye();
        let v: [f64; 3] = std::array::from_fn(|k| e[k] as f64 - self.centre[k]);
        let o: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| self.r[i * 3 + k] * v[k]).sum());
        // up is -Y in the orbit frame (`up_sign` -1)
        let az = o[2].atan2(o[0]).to_degrees().rem_euclid(360.0);
        let el = (-o[1]).atan2((o[0] * o[0] + o[2] * o[2]).sqrt()).to_degrees();
        (az, el)
    }
}

/// Which views a validation holds out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Split {
    /// Every `k`-th view in capture order.
    Every(usize),
    /// Every view whose azimuth lies within `width / 2` of `centre`
    /// (degrees): a gap in the orbit, whatever the elevation.
    Wedge { centre: f64, width: f64 },
    /// Every view whose elevation lies in `[low, high)` degrees.
    Band { low: f64, high: f64 },
}

/// Which of `cams` `split` holds out.
pub fn split(cams: &[Camera], split: Split) -> Vec<bool> {
    let orbit = Orbit::of(cams);
    cams.iter()
        .enumerate()
        .map(|(i, c)| match split {
            Split::Every(k) => crate::eval::holdout(cams.len(), k)[i],
            Split::Wedge { centre, width } => {
                let d = (orbit.angles(c).0 - centre).rem_euclid(360.0);
                d.min(360.0 - d) <= width / 2.0
            }
            Split::Band { low, high } => {
                let el = orbit.angles(c).1;
                el >= low && el < high
            }
        })
        .collect()
}

/// How far `cam` is from everything in `train`, degrees: the smallest angle,
/// seen from `centre`, between its direction and a training camera's.
pub fn deviation(cam: &Camera, train: &[Camera], centre: [f64; 3]) -> f64 {
    let dir = |c: &Camera| -> [f64; 3] {
        let e = c.eye();
        let v: [f64; 3] = std::array::from_fn(|k| e[k] as f64 - centre[k]);
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        v.map(|x| x / l)
    };
    let d = dir(cam);
    train
        .iter()
        .map(|t| {
            let e = dir(t);
            (d[0] * e[0] + d[1] * e[1] + d[2] * e[2]).clamp(-1.0, 1.0).acos().to_degrees()
        })
        .fold(f64::INFINITY, f64::min)
}
