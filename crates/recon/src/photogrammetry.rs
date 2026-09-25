// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Photographs to a splat training set, with no learned model anywhere in
//! the loop: structure from motion ([`sfm`]) recovers every camera's pose,
//! each sensor's calibration through its REAL lens and a sparse coloured
//! point cloud, and every photograph becomes a target exactly as it was
//! recorded - its own pixels, imaged through that lens.
//!
//! Nothing is resampled into an idealized camera. The renderer the fit uses
//! evaluates every gaussian along each pixel's own ray through the lens
//! model, so undistorting a photograph could only lose what it recorded:
//! interpolation spreads each pixel over its neighbours and blurs exactly the
//! detail a fit is trying to recover, and the corners a pinhole frame cannot
//! reach are thrown away. A lower training resolution is reached by exact
//! halvings (a 2x2 box, which is the area average at that scale and moves
//! every pixel coordinate by exactly one half), never by a resample at an
//! arbitrary factor.
//!
//! A photograph loaded with [`imaging::photo::load_photo`] brings more than
//! its pixels, and [`reconstruct_photos`] / [`training_set_photos`] carry it
//! through: its GPS fix to structure from motion, its EXIF exposure
//! (relative to the capture, [`Photometry`]) and its encoding into the
//! target, so the camera model starts from the exposure it knows and a
//! linear, deep or bracketed capture is fitted in linear light.
//!
//! [`dense`] adds what the sparse points cannot give a fit: multi-view stereo
//! ([`mvs`]) measures every photograph's range, surface normal and their
//! confidence per pixel, through the same lens, which become each target's
//! geometry priors, and fuses them into a dense starting scene of thin
//! gaussians lying in the measured surfaces.
//!
//! Swedish Embedded AB implements photogrammetry pipelines - from a folder of
//! photographs to calibrated cameras and a radiance field - for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use gpu_core::Gpu;
use imaging::photo::{Gps, Transfer};
use imaging::Rgb8;
use sfm::georef::Wgs84;
use sfm::incremental::{Photo, Reconstruction, SfmCfg, SfmError};
use splat::isp::{ColorSpace, Encoding};
use splat::opt::{FitCfg, TargetView};
use splat::types::{Camera, Splats};

/// A rigid change of frame `p' = r (p - centre)`, `r` row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub r: [f64; 9],
    pub centre: [f64; 3],
}

impl Frame {
    pub const IDENTITY: Frame = Frame { r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], centre: [0.0; 3] };

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [p[0] - self.centre[0], p[1] - self.centre[1], p[2] - self.centre[2]];
        std::array::from_fn(|i| self.r[i * 3] * d[0] + self.r[i * 3 + 1] * d[1] + self.r[i * 3 + 2] * d[2])
    }

    /// `next` after `self`: `next.apply(self.apply(p))`.
    pub fn then(&self, next: &Frame) -> Frame {
        let r = std::array::from_fn(|k| {
            let (i, j) = (k / 3, k % 3);
            (0..3).map(|m| next.r[i * 3 + m] * self.r[m * 3 + j]).sum()
        });
        // r1 (p - c1) - c2 = r1 (p - c1 - r1^T c2)
        let back: [f64; 3] = std::array::from_fn(|j| (0..3).map(|i| self.r[i * 3 + j] * next.centre[i]).sum());
        Frame { r, centre: std::array::from_fn(|k| self.centre[k] + back[k]) }
    }
}

/// What a fit needs, recovered from photographs alone.
pub struct TrainingSet {
    /// One per REGISTERED photograph, in input order.
    pub targets: Vec<TargetView>,
    /// Input index of each target.
    pub source: Vec<usize>,
    /// The starting scene, from the structure-from-motion points.
    pub init: Splats,
    /// The full reconstruction, in its own frame.
    pub sfm: Reconstruction,
    /// From the reconstruction's frame to the targets' and `init`'s.
    pub frame: Frame,
}

impl TrainingSet {
    /// The same set landed upright: structure from motion leaves the scene in
    /// its gauge's frame (see `sfm::incremental::Gauge`) - the seed
    /// photograph's, however that camera was held, or east-north-up - and a
    /// viewer opening the result expects the ground to be down. Where the
    /// reconstruction knows which way gravity points ([`Reconstruction::up`])
    /// that is what is landed on world up; otherwise the capture's orbit is
    /// taken to be level.
    pub fn upright(self) -> TrainingSet {
        let cams: Vec<Camera> = self.targets.iter().map(|t| t.cam).collect();
        let mats: Vec<[f64; 16]> = cams.iter().map(|c| std::array::from_fn(|i| c.c2w[i] as f64)).collect();
        let (r, centre) = match self.sfm.up {
            // up in the reconstruction's frame, turned into the targets'
            Some(up) => splat::orient::frame_along(&mats, std::array::from_fn(|i| (0..3).map(|j| self.frame.r[i * 3 + j] * up[j]).sum())),
            None => splat::orient::frame_from_cameras(&mats, -1.0),
        };
        let init = splat::orient::apply(&self.init, &r, &centre);
        let targets = self
            .targets
            .into_iter()
            .zip(&mats)
            .map(|(t, m)| {
                let c2w = splat::orient::transform_c2w(m, &r, &centre);
                TargetView { cam: Camera { c2w: std::array::from_fn(|i| c2w[i] as f32), ..t.cam }, ..t }
            })
            .collect();
        TrainingSet { targets, init, frame: self.frame.then(&Frame { r, centre }), ..self }
    }

    /// The same set with multi-view stereo's geometry: every target carries
    /// its measured range, normal and confidence as priors, and `init` is
    /// the dense scene fused from them. `photos` are the photographs the set
    /// was built from, in input order; `gpu` must hold [`mvs::PIPELINES`] at
    /// `ks`.
    pub fn densify(mut self, gpu: &Gpu, ks: &mvs::Kernels, photos: &[Rgb8], cfg: &DenseCfg) -> Result<(TrainingSet, DenseReport), mvs::MvsError> {
        let full: Vec<Camera> = self
            .targets
            .iter()
            .zip(&self.source)
            .map(|(t, &i)| Camera { c2w: t.cam.c2w, shutter: t.cam.shutter, ..Camera::with_intrinsics(t.cam.c2w, &self.sfm.intrinsics[self.sfm.sensor[i]]) })
            .collect();
        let mut view_of = vec![None; self.sfm.poses.len()];
        for (v, &i) in self.source.iter().enumerate() {
            view_of[i] = Some(v);
        }
        let tracks: Vec<mvs::Track> = mvs::select::tracks_from_sfm(&self.sfm, &view_of)
            .into_iter()
            .map(|t| mvs::Track { xyz: self.frame.apply(t.xyz), ..t })
            .collect();
        let rgb: Vec<&Rgb8> = self.source.iter().map(|&i| &photos[i]).collect();
        let (init, report) = dense(gpu, ks, &full, &rgb, &tracks, &mut self.targets, cfg)?;
        Ok((TrainingSet { init, ..self }, report))
    }
}

/// Multi-view stereo, fusion and the dense starting scene, as [`dense`]
/// runs them.
#[derive(Clone, Debug, Default)]
pub struct DenseCfg {
    pub stereo: mvs::StereoCfg,
    pub fuse: mvs::FuseCfg,
    pub init: mvs::SplatInit,
}

/// What [`dense`] measured.
#[derive(Clone, Debug)]
pub struct DenseReport {
    /// Fraction of each target's pixels with a range prior.
    pub coverage: Vec<f64>,
    /// Points in the fused cloud, one gaussian each.
    pub points: usize,
    pub timings: mvs::Timings,
}

/// Measure every view by multi-view stereo, attach the measurements to
/// `targets` as priors at each target's own resolution, and fuse them into a
/// dense scene.
///
/// `cams` are the views' cameras at the photographs' full resolution, in the
/// targets' frame; `tracks` are the structure-from-motion points in that
/// frame, which choose each view's sources and bound its search. A target
/// imaged at a lower resolution than the stereo's finest level gets the map
/// halved exactly (`mvs::DepthMap::halved`, which keeps edges); one imaged
/// finer than it is an error, since a map cannot be upsampled without
/// inventing geometry.
pub fn dense(
    gpu: &Gpu,
    ks: &mvs::Kernels,
    cams: &[Camera],
    photos: &[&Rgb8],
    tracks: &[mvs::Track],
    targets: &mut [TargetView],
    cfg: &DenseCfg,
) -> Result<(Splats, DenseReport), mvs::MvsError> {
    assert_eq!(cams.len(), targets.len(), "dense: one camera per target");
    assert_eq!(photos.len(), targets.len(), "dense: one photograph per target");
    let views: Vec<mvs::View> = cams.iter().zip(photos).map(|(c, p)| mvs::View { cam: *c, rgb: &p.px }).collect();
    let st = mvs::depth_maps(gpu, ks, &views, tracks, &cfg.stereo)?;
    let fused = mvs::fuse(gpu, ks, &st.cams, &st.depth, &st.rgb, &cfg.fuse)?;
    let mut coverage = Vec::with_capacity(targets.len());
    for ((t, map), cam) in targets.iter_mut().zip(&st.depth).zip(&st.cams) {
        let (mut map, mut cam) = (map.clone(), *cam);
        while map.width > t.cam.width {
            map = map.halved(&cam.intrinsics());
            cam = Camera { fx: cam.fx * 0.5, fy: cam.fy * 0.5, cx: cam.cx * 0.5, cy: cam.cy * 0.5, width: cam.width / 2, height: cam.height / 2, ..cam };
        }
        if (map.width, map.height) != (t.cam.width, t.cam.height) {
            return Err(mvs::MvsError::Mismatch(format!(
                "a {}x{} target cannot take a {}x{} depth map: stereo must run at the target's resolution or finer",
                t.cam.width, t.cam.height, map.width, map.height
            )));
        }
        coverage.push(map.coverage());
        t.depth = Some(map.range);
        t.depth_conf = Some(map.conf);
        t.normals = Some(map.normal);
    }
    let init = mvs::to_splats(&fused, &cfg.init);
    Ok((init, DenseReport { coverage, points: fused.len(), timings: st.timings }))
}

/// A photograph as a target: its own pixels in [0,1], through its camera,
/// with the pixels its lens has no ray for (outside a fisheye's image
/// circle) masked out rather than supervised against nothing.
pub fn target(photo: &Rgb8, cam: Camera, sensor: usize) -> TargetView {
    let rgb: Vec<f32> = photo.px.iter().map(|&v| v as f32 / 255.0).collect();
    lens_masked(TargetView::new(cam, rgb).with_sensor(sensor))
}

/// `t` with the pixels its camera's lens has no ray for masked out.
fn lens_masked(t: TargetView) -> TargetView {
    let k = t.cam.intrinsics();
    let (w, h) = (t.cam.width as usize, t.cam.height as usize);
    let mask: Vec<f32> = backend_cpu::par::map_f32(w * h, |i| {
        let px = [(i % w) as f64 + 0.5, (i / w) as f64 + 0.5];
        if k.unproject(px).is_some() { 1.0 } else { 0.0 }
    });
    if mask.iter().all(|&m| m == 1.0) { t } else { t.with_mask(mask) }
}

/// How a capture's photographs enter a fit: the colour space the scene is
/// fitted in, and the exposure every photograph's EXIF is measured against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Photometry {
    pub color_space: ColorSpace,
    /// Mean [`imaging::Exif::exposure_log2`] of the photographs that record
    /// one: each target's exposure hint is relative to it, so the capture's
    /// average camera sits at 0.
    pub reference_ev: Option<f64>,
}

impl Photometry {
    /// Chosen from the photographs themselves: scene-linear when any of them
    /// is linear or deeper than 8 bits (the precision is there to be used),
    /// or when their exposures differ by more than half a stop (a bracket is
    /// several measurements of ONE radiance, which only a linear scene can
    /// be); display-referred otherwise. A scene-linear fit needs the camera
    /// model ([`splat::opt::FitCfg::isp`]) to put the transfer back.
    pub fn of(photos: &[&imaging::Photo]) -> Photometry {
        let evs: Vec<f64> = photos.iter().filter_map(|p| p.exif.exposure_log2()).collect();
        let reference_ev = (!evs.is_empty()).then(|| evs.iter().sum::<f64>() / evs.len() as f64);
        let spread = evs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - evs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let deep = photos.iter().any(|p| p.transfer == Transfer::Linear || p.bits > 8);
        let color_space = if deep || spread > 0.5 { ColorSpace::SceneLinear } else { ColorSpace::Display };
        Photometry { color_space, reference_ev }
    }

    /// `photo`'s exposure relative to the capture, in log2 stops; 0 when its
    /// EXIF does not say.
    pub fn exposure(&self, photo: &imaging::Photo) -> f32 {
        match (photo.exif.exposure_log2(), self.reference_ev) {
            (Some(ev), Some(r)) => (ev - r) as f32,
            _ => 0.0,
        }
    }
}

/// A photograph as a target, with what its file says about how it was
/// recorded: its exposure relative to the capture, and its values in the
/// form `ph`'s colour space fits against - display-referred sRGB, or for a
/// scene-linear fit the values as recorded when they are sRGB-encoded
/// (the camera model applies the transfer) and linear Rec.709 light when
/// they are not (a linear or profiled photograph, wide-gamut primaries).
/// `photo` must be at `cam`'s resolution. Halve it with
/// [`TargetView::half_in`] in `ph.color_space`.
pub fn photo_target(photo: &imaging::Photo, cam: Camera, sensor: usize, ph: &Photometry) -> TargetView {
    assert_eq!((photo.width, photo.height), (cam.width, cam.height), "a photograph at its camera's resolution");
    let (rgb, encoding) = match ph.color_space {
        ColorSpace::Display => (photo.display(), Encoding::Srgb),
        ColorSpace::SceneLinear if photo.transfer == Transfer::Srgb && photo.to_rec709.is_none() => (photo.encoded.clone(), Encoding::Srgb),
        ColorSpace::SceneLinear => (photo.linear(), Encoding::Linear),
    };
    lens_masked(TargetView::new(cam, rgb).with_sensor(sensor).with_exposure(ph.exposure(photo)).with_encoding(encoding))
}

/// Recover cameras from `photos` and build the training set `halvings`
/// exact halvings below the photographs' own resolution.
/// `opacity` is what every starting gaussian gets.
pub fn training_set(photos: &[Rgb8], halvings: u32, opacity: f32, cfg: &SfmCfg) -> Result<TrainingSet, SfmError> {
    training_set_located(photos, &[], halvings, opacity, cfg)
}

/// [`training_set`] for photographs with satellite fixes (EXIF GPS, in
/// input order; missing entries and a short slice mean none): when the fixes
/// span the capture the scene comes back in metres east-north-up, otherwise
/// they are reported unused in `sfm.report`.
pub fn training_set_located(photos: &[Rgb8], gps: &[Option<Gps>], halvings: u32, opacity: f32, cfg: &SfmCfg) -> Result<TrainingSet, SfmError> {
    let make = |i: usize, cam: Camera, sensor: usize| target(&photos[i], cam, sensor);
    solve(photos, gps, halvings, ColorSpace::Display, opacity, cfg, make)
}

/// [`training_set_located`] from photographs loaded with
/// [`imaging::photo::load_photo`]: their own GPS fixes, and every target
/// with its exposure and encoding as [`photo_target`] makes them under `ph`,
/// halved in `ph`'s colour space.
pub fn training_set_photos(photos: &[imaging::Photo], halvings: u32, opacity: f32, cfg: &SfmCfg, ph: &Photometry) -> Result<TrainingSet, SfmError> {
    let rgb8: Vec<Rgb8> = photos.iter().map(imaging::Photo::rgb8).collect();
    let gps: Vec<Option<Gps>> = photos.iter().map(|p| p.exif.gps).collect();
    let make = |i: usize, cam: Camera, sensor: usize| photo_target(&photos[i], cam, sensor, ph);
    solve(&rgb8, &gps, halvings, ph.color_space, opacity, cfg, make)
}

/// Structure from motion on `photos`, each registered one made a target by
/// `make(input index, camera, sensor)` at full resolution and halved in
/// `space`.
fn solve(
    photos: &[Rgb8],
    gps: &[Option<Gps>],
    halvings: u32,
    space: ColorSpace,
    opacity: f32,
    cfg: &SfmCfg,
    make: impl Fn(usize, Camera, usize) -> TargetView,
) -> Result<TrainingSet, SfmError> {
    let fix = |i: usize| gps.get(i).copied().flatten().map(|g| Wgs84 { latitude_deg: g.latitude_deg, longitude_deg: g.longitude_deg, altitude_m: g.altitude_m });
    let views: Vec<Photo> = photos.iter().enumerate().map(|(i, p)| Photo { width: p.w, height: p.h, rgb: &p.px, sensor: 0, focal_px: None, gps: fix(i) }).collect();
    let rec = sfm::incremental::reconstruct(&views, cfg)?;
    let mut targets = Vec::new();
    let mut source = Vec::new();
    for (i, pose) in rec.poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        let s = rec.sensor[i];
        let m = pose.c2w();
        let cam = Camera::with_intrinsics(std::array::from_fn(|k| m[k] as f32), &rec.intrinsics[s]);
        let mut t = make(i, cam, s);
        for _ in 0..halvings {
            t = t.half_in(space);
        }
        targets.push(t);
        source.push(i);
    }
    let xyz: Vec<f32> = rec.points.iter().flat_map(|p| p.xyz.map(|v| v as f32)).collect();
    let rgb: Vec<f32> = rec.points.iter().flat_map(|p| p.rgb).collect();
    let init = splat::init::from_points(&xyz, &rgb, opacity);
    Ok(TrainingSet { targets, source, init, sfm: rec, frame: Frame::IDENTITY })
}

/// Every WGSL kernel [`reconstruct`] dispatches: `splat`'s at 0, then
/// `mvs`'s at `splat::PIPELINES.len()`. Build the device with this list.
pub fn pipelines() -> Vec<(&'static str, &'static str)> {
    splat::PIPELINES.iter().chain(mvs::PIPELINES).copied().collect()
}

/// How [`reconstruct`] turns photographs into a scene. Every `None` is
/// decided from the capture itself.
#[derive(Clone, Debug)]
pub struct PhotoCfg {
    pub sfm: SfmCfg,
    /// Widest training image: the photographs are halved exactly until they
    /// fit. The fit itself climbs a pyramid up to it.
    pub max_width: u32,
    /// Multi-view stereo priors and dense start; `None` starts from the
    /// structure-from-motion points alone.
    pub dense: Option<DenseCfg>,
    /// Optimizer steps; `None` = [`auto_iterations`].
    pub iterations: Option<usize>,
    /// Gaussian budget; `None` = [`auto_budget`].
    pub max_gaussians: Option<usize>,
    /// Fit a photometric camera model alongside the scene (the default: the
    /// global model - exposure, white balance, vignetting, colour matrix and
    /// response - measured neutral to slightly better held out on a capture
    /// at one exposure, and what a capture with changing exposure needs).
    pub camera_model: Option<splat::isp::IspCfg>,
    /// Fit the environment behind the scene (sky, distant scenery) at this
    /// spherical-harmonic degree (`splat::env`).
    pub environment: Option<u32>,
    /// The photographs may hold transient content (people, traffic), which
    /// the fit should stop supervising (`FitCfg::transients`).
    pub transients: bool,
}

impl Default for PhotoCfg {
    fn default() -> Self {
        PhotoCfg {
            sfm: SfmCfg::default(),
            max_width: 2048,
            dense: Some(DenseCfg::default()),
            iterations: None,
            max_gaussians: None,
            camera_model: Some(splat::isp::IspCfg::default()),
            environment: None,
            transients: false,
        }
    }
}

/// Optimizer steps for `views` training views: every view revisited about
/// five hundred times at two views a step, never fewer than 3000 steps (the
/// schedule's density rounds need room) nor more than 30 000 (3DGS's own
/// length, for captures of hundreds of views).
pub fn auto_iterations(views: usize) -> usize {
    (250 * views).clamp(3000, 30_000)
}

/// The gaussian budget for a start of `start` gaussians against views of
/// `pixels` pixels in all: a dense start already has about the population
/// its surfaces need and gets a quarter more for density control to spend;
/// a sparse one grows to one gaussian per eight training pixels, within
/// 200 000 and 3 000 000.
pub fn auto_budget(start: usize, pixels: usize, dense: bool) -> usize {
    if dense {
        start + start / 4
    } else {
        (pixels / 8).clamp(200_000, 3_000_000).max(start)
    }
}

/// What [`reconstruct`] made.
pub struct Reconstructed {
    /// The fitted scene with its 3D filter baked in: what any viewer renders.
    pub scene: Splats,
    /// Every registered photograph's camera at the training resolution,
    /// refined by the fit.
    pub cameras: Vec<Camera>,
    /// Input index of each camera's photograph.
    pub source: Vec<usize>,
    /// Structure from motion's reprojection error, pixels.
    pub reprojection_rms_px: f64,
    /// The fitted photometric camera model, when one was asked for.
    pub isp: Option<splat::isp::Isp>,
    /// What `scene`'s colours mean: scene-linear when the photographs asked
    /// for it and the camera model was on ([`Photometry::of`]).
    pub color_space: ColorSpace,
    /// The fitted environment, when one was asked for.
    pub env: Option<splat::env::EnvMap>,
    /// How the render options the scene was fitted under.
    pub render: splat::types::RenderOpts,
    /// Stereo coverage and point count, when the dense path ran.
    pub dense: Option<DenseReport>,
    pub iterations: usize,
    pub max_gaussians: usize,
    /// The objective at the last step.
    pub loss: f32,
}

impl Reconstructed {
    /// The scene as a standard splat viewer should get it: the gaussians,
    /// and the environment (if any) baked into a shell of distant gaussians
    /// fifty times as far out as the cameras spread, since such a viewer
    /// has no environment of its own; display-referred
    /// ([`splat::isp::bake_display`]) when it was fitted scene-linear, since
    /// such a viewer shows stored colour.
    pub fn export(&self) -> Splats {
        let whole = self.with_environment();
        match self.color_space {
            ColorSpace::Display => whole,
            ColorSpace::SceneLinear => splat::isp::bake_display(&whole),
        }
    }

    fn with_environment(&self) -> Splats {
        let Some(env) = &self.env else { return self.scene.clone() };
        let n = self.cameras.len().max(1) as f32;
        let centre: [f32; 3] = std::array::from_fn(|k| self.cameras.iter().map(|c| c.eye()[k]).sum::<f32>() / n);
        let spread = self
            .cameras
            .iter()
            .map(|c| (0..3).map(|k| (c.eye()[k] - centre[k]).powi(2)).sum::<f32>().sqrt())
            .fold(0.0f32, f32::max)
            .max(1e-3);
        let shell = env.to_splats(centre, 50.0 * spread, 20_000);
        splat::align::concat(&[self.scene.clone(), shell])
    }
}

/// Why [`reconstruct`] stopped.
#[derive(Debug)]
pub enum ReconstructError {
    Sfm(SfmError),
    Stereo(mvs::MvsError),
}

impl std::fmt::Display for ReconstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReconstructError::Sfm(e) => write!(f, "structure from motion: {e}"),
            ReconstructError::Stereo(e) => write!(f, "multi-view stereo: {e}"),
        }
    }
}

impl std::error::Error for ReconstructError {}

/// Photographs to a finished scene: structure from motion through each
/// photograph's real lens, the set landed upright, multi-view stereo priors
/// and a dense start (unless `cfg.dense` is `None`), and the fit. `gpu` must
/// have been built with [`pipelines`]. `log` receives one line per stage;
/// `on_step(iteration, loss)` is polled every step and stops the fit early
/// by returning `false`.
///
/// Pictures with nothing known about how they were taken; a capture loaded
/// with [`imaging::photo::load_photo`] goes to [`reconstruct_photos`].
pub fn reconstruct(
    gpu: &Gpu,
    photos: &[Rgb8],
    cfg: &PhotoCfg,
    log: &mut dyn FnMut(&str),
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> Result<Reconstructed, ReconstructError> {
    let photos: Vec<imaging::Photo> = photos.iter().map(imaging::Photo::from_rgb8).collect();
    reconstruct_photos(gpu, &photos, cfg, log, on_step)
}

/// [`reconstruct`] of photographs as recorded: their GPS fixes place the
/// scene, their EXIF exposures start the camera model's, and a linear, deep
/// or bracketed capture is fitted scene-linear when `cfg.camera_model` is
/// on ([`Photometry::of`] decides; the camera model's own colour space is
/// replaced by that decision).
pub fn reconstruct_photos(
    gpu: &Gpu,
    photos: &[imaging::Photo],
    cfg: &PhotoCfg,
    log: &mut dyn FnMut(&str),
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> Result<Reconstructed, ReconstructError> {
    let widest = photos.iter().map(|p| p.width).max().unwrap_or(0);
    let halvings = (0..16).find(|&h| widest >> h <= cfg.max_width.max(1)).unwrap_or(16);
    let mut ph = Photometry::of(&photos.iter().collect::<Vec<_>>());
    if cfg.camera_model.is_none() {
        ph.color_space = ColorSpace::Display;
    }
    log(&format!("photometry: {:?}, exposures from EXIF {}", ph.color_space, if ph.reference_ev.is_some() { "recorded" } else { "unknown" }));
    let t = std::time::Instant::now();
    let set = training_set_photos(photos, halvings, 0.1, &cfg.sfm, &ph).map_err(ReconstructError::Sfm)?.upright();
    log(&format!(
        "structure from motion: {}/{} photographs placed, {} points, {:.2} px rms, lens {:?}, {:.0} s",
        set.targets.len(),
        photos.len(),
        set.sfm.points.len(),
        set.sfm.rms_px,
        set.sfm.lens,
        t.elapsed().as_secs_f64()
    ));
    let (set, dense) = match &cfg.dense {
        None => (set, None),
        Some(dcfg) => {
            let t = std::time::Instant::now();
            let dcfg = DenseCfg { stereo: mvs::StereoCfg { halving: halvings, ..dcfg.stereo.clone() }, ..dcfg.clone() };
            let ks = mvs::Kernels::at(splat::PIPELINES.len());
            let rgb8: Vec<Rgb8> = photos.iter().map(imaging::Photo::rgb8).collect();
            let (set, report) = set.densify(gpu, &ks, &rgb8, &dcfg).map_err(ReconstructError::Stereo)?;
            let cover = report.coverage.iter().sum::<f64>() / report.coverage.len().max(1) as f64;
            log(&format!("multi-view stereo: {} points, {:.0}% of pixels measured, {:.0} s", report.points, 100.0 * cover, t.elapsed().as_secs_f64()));
            (set, Some(report))
        }
    };
    let views = set.targets.len();
    let pixels: usize = set.targets.iter().map(|t| (t.cam.width * t.cam.height) as usize).sum();
    let iterations = cfg.iterations.unwrap_or_else(|| auto_iterations(views));
    let budget = cfg.max_gaussians.unwrap_or_else(|| auto_budget(set.init.len(), pixels, dense.is_some()));
    let preset = if dense.is_some() {
        FitCfg::from_dense_stereo(iterations, budget, views)
    } else {
        FitCfg::from_sparse_points(iterations, budget, views)
    };
    let isp = cfg.camera_model.map(|i| splat::isp::IspCfg { color_space: ph.color_space, ..i }).or(preset.isp);
    let fit_cfg = FitCfg { log_every: 0, isp, environment: cfg.environment, transients: cfg.transients, ..preset };
    let (w, h) = (set.targets[0].cam.width, set.targets[0].cam.height);
    log(&format!("fit: {views} views at {w}x{h}, {} gaussians to start, budget {budget}, {iterations} steps", set.init.len()));
    let t = std::time::Instant::now();
    let fitted = splat::opt::fit_full(gpu, splat::Kernels::at(0), &set.init, &set.targets, &fit_cfg, on_step);
    log(&format!("fit: {} gaussians, loss {:.5}, {:.0} s", fitted.scene.len(), fitted.loss, t.elapsed().as_secs_f64()));
    Ok(Reconstructed {
        scene: fitted.baked(),
        cameras: fitted.cams,
        source: set.source,
        reprojection_rms_px: set.sfm.rms_px,
        isp: fitted.isp,
        color_space: ph.color_space,
        env: fitted.env,
        render: splat::types::RenderOpts { ray: true, antialiased: fit_cfg.antialiased, eps2d: fit_cfg.eps2d, ..Default::default() },
        dense,
        iterations,
        max_gaussians: budget,
        loss: fitted.loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A photograph is supervised as recorded: its pixels unchanged, every
    /// one of them whose ray the lens can trace - and outside a fisheye's
    /// image circle, none.
    #[test]
    fn a_target_is_the_photograph_itself_masked_where_the_lens_sees_nothing() {
        let (w, h) = (40u32, 30u32);
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let img = Rgb8 { w, h, px: px.clone() };
        let k = camera::Intrinsics { lens: camera::Lens::radial(-0.05, 0.01), ..camera::Intrinsics::pinhole(30.0, w, h) };
        let t = target(&img, Camera::with_intrinsics(std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 }), &k), 0);
        for (i, (got, &want)) in t.rgb.iter().zip(&px).enumerate() {
            assert!((got - want as f32 / 255.0).abs() < 1e-6, "pixel value {i}");
        }
        assert!(t.mask.is_none(), "every pixel of a mild barrel lens has a ray");
        // an equidistant fisheye whose image circle is smaller than the
        // frame: 180 degrees off axis is pi f = 18.8 px from the centre, and
        // the corners are 25 px out
        let fish = camera::Intrinsics { lens: camera::Lens::Fisheye { k: [0.0; 4] }, ..camera::Intrinsics::pinhole(6.0, w, h) };
        let t = target(&img, Camera::with_intrinsics(std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 }), &fish), 0);
        let mask = t.mask.expect("pixels beyond the fisheye's reach are masked");
        assert_eq!(mask[(h / 2 * w + w / 2) as usize], 1.0);
        assert!(mask.contains(&0.0), "a corner past 180 degrees has no ray");
    }

    fn photograph(bits: u8, transfer: imaging::photo::Transfer, exposure_s: f64) -> imaging::Photo {
        let (w, h) = (6u32, 4u32);
        imaging::Photo {
            width: w,
            height: h,
            encoded: (0..w * h * 3).map(|i| (i % 11) as f32 / 10.0).collect(),
            bits,
            transfer,
            to_rec709: None,
            exif: imaging::Exif { exposure_s: Some(exposure_s), f_number: Some(2.0), iso: Some(100.0), ..Default::default() },
        }
    }

    /// What a photograph's file says about how it was recorded reaches the
    /// fit: its exposure relative to the capture (EXIF), and in a
    /// scene-linear fit its values either as encoded (the camera model then
    /// applies the transfer) or, when its own transfer is not sRGB, decoded
    /// to linear light.
    #[test]
    fn a_photograph_target_carries_its_exposure_and_encoding() {
        use imaging::photo::Transfer;
        use splat::isp::{ColorSpace, Encoding};
        let cam = Camera::with_intrinsics(std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 }), &camera::Intrinsics::pinhole(5.0, 6, 4));
        // an 8-bit sRGB pair at one exposure: display-referred
        let (a, b) = (photograph(8, Transfer::Srgb, 1.0 / 100.0), photograph(8, Transfer::Srgb, 1.0 / 100.0));
        let ph = Photometry::of(&[&a, &b]);
        assert_eq!(ph.color_space, ColorSpace::Display);
        let t = photo_target(&a, cam, 0, &ph);
        assert_eq!((t.exposure, t.encoding), (0.0, Encoding::Srgb));
        assert_eq!(t.rgb, a.display());
        // a bracket (one stop apart): scene-linear, each view's exposure
        // relative to the capture's mean
        let (a, b) = (photograph(8, Transfer::Srgb, 1.0 / 200.0), photograph(8, Transfer::Srgb, 1.0 / 100.0));
        let ph = Photometry::of(&[&a, &b]);
        assert_eq!(ph.color_space, ColorSpace::SceneLinear);
        let (ta, tb) = (photo_target(&a, cam, 0, &ph), photo_target(&b, cam, 0, &ph));
        assert!((ta.exposure + 0.5).abs() < 1e-6 && (tb.exposure - 0.5).abs() < 1e-6, "{} {}", ta.exposure, tb.exposure);
        assert_eq!((ta.encoding, &ta.rgb), (Encoding::Srgb, &a.encoded), "an sRGB photograph is supervised as recorded");
        // a 16-bit linear photograph: scene-linear, supervised in linear light
        let lin = photograph(16, Transfer::Linear, 1.0 / 100.0);
        let ph = Photometry::of(&[&lin]);
        assert_eq!(ph.color_space, ColorSpace::SceneLinear);
        let t = photo_target(&lin, cam, 0, &ph);
        assert_eq!((t.encoding, &t.rgb), (Encoding::Linear, &lin.linear()));
    }
}
