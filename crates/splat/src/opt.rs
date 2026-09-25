// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3DGS scene optimization ("fit"): a splat scene, and optionally the cameras
//! and the photometric camera model, fitted to posed photographs.
//!
//! The forward model is the ray renderer ([`crate::renderer::Renderer`] with
//! [`RenderOpts::ray`]): every gaussian is evaluated exactly along each
//! pixel's own ray through the camera's REAL lens, so a fit compares the
//! scene with the photograph's own pixels - never with a resampled,
//! undistorted copy of them - and the same pass supplies the expected range,
//! the composited normal and the depth distortion the geometry terms need.
//!
//! The optimization runs on the device: a scene is raw parameters
//! (`crate::train`), log scales and opacity logits stepped by
//! `splat_adam.wgsl` with per-gaussian position rates and shape bounds, so
//! nothing is clamped on the host and nothing is read back per step but the
//! loss. Adam's state survives density control (see `crate::train`), and the
//! global timestep runs across it.
//!
//! What a fit may refine besides the scene, each after its own warm-up
//! ([`CameraRefine`]): every camera's pose (in its own frame, with the gauge
//! fixed - the first camera and the scale do not move), each sensor's
//! calibration (focal length, principal point, lens distortion), and a
//! rolling shutter's readout motion - the gradients all come from the ray
//! renderer's backward exactly.
//!
//! Swedish Embedded AB implements 3D reconstruction optimizers for its
//! clients. If your team needs expertise in differentiable rendering and
//! photogrammetry then you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};

use crate::density::{self, Evidence};
use crate::env::{EnvDevice, EnvMap};
use crate::isp::{ColorSpace, DeviceIsp, Encoding, Isp, IspCfg, Shot};
use crate::loss::{DeviceLoss, PixelLoss};
use crate::renderer::{dispatched_groups, BwdScratch, CameraGrad, Renderer};
use crate::train::{Bounds, DeviceScene, Rates};
use crate::types::{Camera, Mode, RenderOpts, Splats};
use crate::Kernels;

/// Which density-control strategy `fit` runs when `densify_every > 0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Densify {
    /// Upstream 3DGS: threshold a positional-gradient statistic, then SPLIT
    /// the large gaussians above it and CLONE the small ones, and drop
    /// whatever has gone transparent.
    #[default]
    Heuristic,
    /// 3DGS-MCMC (Kheradmand et al., NeurIPS 2024, arXiv:2404.09591):
    /// no thresholds and no deletion. Transparent gaussians are RELOCATED onto
    /// opaque ones with the opacity and scale correction that keeps the render
    /// unchanged at the moment of the move, and the budget is spent up to
    /// `max_gaussians`. See [`crate::mcmc`].
    Mcmc,
    /// Credit-assigned, budgeted density control ([`crate::density`]): every
    /// gaussian is scored by the ranks of the image residual it is
    /// responsible for, the part of it on edges, and its AbsGS gradient; the
    /// top of that ranking is refined within a population schedule that
    /// reaches `max_gaussians` two thirds of the way through density control;
    /// gaussians that contribute nothing twice in a row pay for it.
    Hybrid,
}

/// What of the cameras a fit refines, and from when. Fractions are of the
/// whole fit; 1.0 (the default) never starts.
///
/// Late on purpose. A pose error blurs the scene because gaussians have to
/// compromise between rays that disagree, but a pose refined against a scene
/// that is still a cloud of blobs follows the blobs. The calibration comes
/// after the poses, and distortion is the weakest-constrained of all.
#[derive(Clone, Copy, Debug)]
pub struct CameraRefine {
    pub pose_after: f32,
    /// Step, in radians and scene-normalized units (the fit works in a frame
    /// about one unit across).
    pub pose_lr: f32,
    pub intrinsics_after: f32,
    /// Step of the focal length, as a fraction of itself.
    pub focal_lr: f32,
    /// Step of the principal point, in pixels.
    pub principal_lr: f32,
    /// Step of the lens's distortion coefficients.
    pub distortion_lr: f32,
    /// Refine the principal point at all.
    pub principal: bool,
    /// Refine each view's rolling-shutter readout motion from `pose_after`.
    pub shutter: bool,
}

impl Default for CameraRefine {
    fn default() -> Self {
        CameraRefine {
            pose_after: 1.0,
            pose_lr: 1e-4,
            intrinsics_after: 1.0,
            focal_lr: 1e-5,
            principal_lr: 1e-2,
            distortion_lr: 1e-5,
            principal: true,
            shutter: false,
        }
    }
}

pub struct FitCfg {
    pub iters: usize,
    /// Position step at the start, in scene-normalized units (the fit works
    /// in a frame about one unit across) - or, with
    /// [`FitCfg::relative_position`], in units of each gaussian's own largest
    /// axis - decaying to a hundredth of itself over the fit (3DGS's
    /// schedule). The whole fit then moves a gaussian about
    /// `0.215 * iters * lr_position` of that unit at most (Adam's normalized
    /// step can exceed its rate while a gradient's scale settles, measured at
    /// about 10% over), which is how a caller whose
    /// geometry is already metric says "settle onto the surface, do not leave
    /// it": `lr_position = 4.65 / iters` is one radius over the fit.
    pub lr_position: f32,
    /// Log-scale step: a relative change of size per step.
    pub lr_scale: f32,
    /// Quaternion step.
    pub lr_rotation: f32,
    /// Opacity-logit step.
    pub lr_opacity: f32,
    /// Base-colour step; the SH bands take a twentieth of it.
    pub lr_color: f32,
    /// Scale each gaussian's position step by its own size. A single rate is
    /// simultaneously a rounding error for a blob that has to travel across
    /// an unseeded object and a jump of several radii for the detail density
    /// control just subdivided - the second turns fine geometry into fog.
    pub relative_position: bool,
    /// No axis below this, in scene-normalized units. The 3D Mip filter, not
    /// this, is what keeps a gaussian from shrinking below what its cameras
    /// sampled; this only keeps the logarithm finite.
    pub min_scale: f32,
    /// Largest ratio a gaussian's longest axis may have to its MIDDLE one.
    /// <= 1 = unconstrained.
    ///
    /// Sort a gaussian's axes a >= b >= c and the two ways it can be
    /// anisotropic are not equally wrong. A DISC - a close to b, both far
    /// above c - is a surface element, which is what a reconstruction of a
    /// surface is supposed to contain. A NEEDLE - a far above b and c - is
    /// always wrong: lined up with a training view's ray it lowers that view's
    /// loss while being invisible in it, and is a streak from every other
    /// direction. Measured on a real capture, fitting under an a/c bound took
    /// the 90th percentile of a/b from 1.94 to 9.29.
    pub max_needle: f32,
    /// Bound on b/c, the flatness of a gaussian: looser than `max_needle`
    /// because a surface element is legitimately a disc, but not absent - a
    /// 14 x 7 x 0.26 pixel blade is a needle seen edge on.
    pub max_flat: f32,
    /// Largest a gaussian may be, in PIXELS as its finest camera sees it.
    /// 0 = unbounded.
    pub max_scale_pixels: f32,
    /// How many times its STARTING size a gaussian may grow; the children of
    /// a split inherit their parent's start. 0 = unbounded.
    ///
    /// A ceiling in pixels alone cannot be right for every scene: a sparse
    /// scene legitimately has gaussians many pixels across, while a
    /// feed-forward reconstruction emits SUB-pixel ones (median 0.6 px), and
    /// what was measured going wrong there is GROWTH - over 400 iterations a
    /// fit inflated a real reconstruction's longest axis 7.5x, which is what
    /// turns a scene into fog from anywhere it was not fitted. A bound
    /// relative to where each gaussian started says "refine what the
    /// reconstruction proposed, do not replace it with something far larger".
    pub max_growth: f32,
    pub log_every: usize,
    /// Run density control every N iterations, 0 = never.
    pub densify_every: usize,
    /// Skip density control until this iteration.
    pub densify_after: usize,
    /// Stop density control at this iteration, 0 = run it to the end. A fit
    /// still rearranging its scene on the last iteration never converges on
    /// the arrangement it chose.
    pub densify_until: usize,
    /// Fraction of gaussians refined per round when there is no budget.
    pub densify_frac: f32,
    /// The opacity below which a gaussian is not carrying anything:
    /// [`Densify::Heuristic`] drops it, [`Densify::Mcmc`] relocates it.
    pub prune_opacity: f32,
    /// Refuse to grow past this many gaussians, 0 = unbounded.
    pub max_gaussians: usize,
    /// A pixel's footprint variance, pixels². The ray renderer applies it at
    /// each gaussian's range as the 2D Mip filter's 3D analogue.
    pub eps2d: f32,
    /// Compensate the footprint filter's energy (Mip-Splatting's 2D Mip
    /// filter). Clear it only to fit for a viewer that renders Inria's
    /// uncompensated dilation.
    pub antialiased: bool,
    /// Mip-Splatting's 3D smoothing filter: every gaussian is low-passed to
    /// `mip_scale` samples of the finest camera that SAW it. 0 disables.
    pub mip_scale: f32,
    /// Spherical-harmonic degree for view-dependent colour: 0, 1, 2 or 3.
    pub sh_degree: u32,
    /// How density control decides where gaussians go.
    pub strategy: Densify,
    /// [`Densify::Mcmc`] without a budget: growth per round.
    pub mcmc_grow_frac: f32,
    /// SGLD amplitude (3DGS-MCMC's lambda_noise) in units of each gaussian's
    /// own size per position step, weighted to the near-transparent so only
    /// what is explaining nothing explores. 0 = plain descent.
    pub noise: f32,
    /// L1 regularization of every gaussian's opacity and of its scale
    /// (3DGS-MCMC's lambda_o = 0.01, lambda_S = 0.01 under its loss). They
    /// are what manufactures the dead gaussians relocation recycles; a
    /// capture's floaters are exactly what the opacity term fades.
    pub opacity_reg: f32,
    pub scale_reg: f32,
    /// The photometric objective. [`PixelLoss::Mse`] (the default) is what
    /// `fit` has always minimised; [`PixelLoss::gaussian_splatting`] is the
    /// L1 + D-SSIM objective 3DGS is defined with.
    pub loss: PixelLoss,
    /// Fit a photometric camera model alongside the scene - see
    /// [`crate::isp`]. `None` compares renders with photographs directly.
    pub isp: Option<IspCfg>,
    /// Weight of the depth term (robust log-range against
    /// [`TargetView::depth`]) relative to RGB; 0 = off.
    pub depth_weight: f32,
    /// Knee of the depth term's pseudo-Huber, in log range: past it a prior
    /// pulls with bounded force.
    pub depth_delta: f32,
    /// Weight of the 2DGS depth distortion, 0 = off.
    pub distortion_weight: f32,
    /// Weight of normal consistency against the rendered range map's own
    /// normals, where there is no prior; 0 = off.
    pub normal_consistency_weight: f32,
    /// Weight of supervision by [`TargetView::normals`], 0 = off.
    pub normal_prior_weight: f32,
    /// Fraction of the fit after which the regularizers (distortion,
    /// self-consistency) switch on; surface regularizers applied before the
    /// scene has any shape pull gaussians onto whatever surface happens to be
    /// rendered first. External priors (depth, normals) are on throughout.
    pub geometry_after: f32,
    /// Fraction of the fit by which every SH band up to `sh_degree` is on.
    pub sh_ramp: f32,
    /// Coarse-to-fine: the fit runs `pyramid` halvings below full resolution,
    /// finest last, spending the first `coarse` of the fit on them. 0 levels
    /// = full resolution throughout.
    pub pyramid: u32,
    pub coarse: f32,
    /// Views per iteration, 0 = all of them.
    pub batch: usize,
    /// What of the cameras to refine.
    pub camera: CameraRefine,
    /// Fit the environment the scene is seen against - radiance by
    /// direction, composited behind the gaussians ([`crate::env`]) - as a
    /// spherical-harmonic expansion of this degree (at most
    /// [`crate::env::MAX_DEGREE`]); `None` = the background is black.
    pub environment: Option<u32>,
    /// The photographs may hold transient content (people, traffic): at
    /// every density round, stop supervising each view's large coherent
    /// regions the scene cannot explain
    /// ([`crate::loss::transient_mask`]). Off for a static capture, where it
    /// can only withhold supervision from regions that are merely hard.
    pub transients: bool,
}

impl Default for FitCfg {
    fn default() -> Self {
        FitCfg {
            iters: 200,
            lr_position: 1.6e-4,
            lr_scale: 5e-3,
            lr_rotation: 1e-3,
            lr_opacity: 5e-2,
            lr_color: 2.5e-3,
            relative_position: false,
            min_scale: 1e-7,
            max_needle: 2.0,
            max_flat: 4.0,
            max_scale_pixels: 16.0,
            max_growth: 2.0,
            log_every: 20,
            densify_every: 0,
            densify_after: 30,
            densify_until: 0,
            densify_frac: 0.05,
            prune_opacity: 0.02,
            max_gaussians: 0,
            eps2d: RenderOpts::default().eps2d,
            antialiased: RenderOpts::default().antialiased,
            mip_scale: crate::mip::DEFAULT_SCALE,
            sh_degree: 0,
            strategy: Densify::Heuristic,
            mcmc_grow_frac: 0.05,
            noise: 0.0,
            opacity_reg: 0.0,
            scale_reg: 0.0,
            loss: PixelLoss::Mse,
            isp: None,
            depth_weight: 0.0,
            depth_delta: 0.05,
            distortion_weight: 0.0,
            normal_consistency_weight: 0.0,
            normal_prior_weight: 0.0,
            geometry_after: 0.0,
            sh_ramp: 0.0,
            pyramid: 0,
            coarse: 0.0,
            batch: 0,
            camera: CameraRefine::default(),
            environment: None,
            transients: false,
        }
    }
}

/// The spherical-harmonic degree a capture of `views` photographs can
/// support: the largest `d` with `8 (d+1)² <= views`. Flat colour below
/// 32 views, full degree 3 from 128.
pub fn sh_degree_for_views(views: usize) -> u32 {
    (0..=3u32).rev().find(|d| 8 * (d + 1) * (d + 1) <= views as u32).unwrap_or(0)
}

impl FitCfg {
    /// Everything a scene started from a SPARSE point cloud (structure from
    /// motion) needs to become a finished reconstruction of `views`
    /// photographs in `iters` iterations and at most `budget` gaussians: the
    /// L1 + D-SSIM objective, credit-assigned density control from 5% to 60%
    /// of the fit, the 3D Mip filter under the 0.3 px dilation standard viewers
    /// render, as much view-dependent colour as the
    /// capture has views to support ([`sh_degree_for_views`]), the surface
    /// regularizers once the scene has a shape (40%), two views per step, and
    /// a pyramid of two halvings over the first 30%.
    pub fn from_sparse_points(iters: usize, budget: usize, views: usize) -> FitCfg {
        let every = (iters / 20).max(1);
        FitCfg {
            iters,
            strategy: Densify::Hybrid,
            densify_every: every,
            densify_after: every,
            densify_until: iters * 6 / 10,
            max_gaussians: budget,
            sh_degree: sh_degree_for_views(views),
            // thirty of its own radii over the fit: a sparse cloud has to
            // travel to cover what its points only sample
            relative_position: true,
            lr_position: 30.0 * 4.65 / iters.max(1) as f32,
            loss: PixelLoss::gaussian_splatting(),
            // the dilation viewers render: see `RenderOpts::antialiased`
            antialiased: false,
            prune_opacity: 0.005,
            max_scale_pixels: 32.0,
            // a sparse cloud has to grow a long way to cover what its points
            // only sample
            max_growth: 0.0,
            max_needle: 2.0,
            max_flat: 10.0,
            distortion_weight: 0.1,
            normal_consistency_weight: 0.05,
            geometry_after: 0.4,
            sh_ramp: 0.3,
            pyramid: 2,
            coarse: 0.3,
            batch: 2,
            log_every: 50,
            ..Default::default()
        }
    }

    /// The fit for a scene started from multi-view stereo: thin gaussians
    /// already lying in the measured surfaces, and every view carrying the
    /// stereo's range and normal as priors. The gaussians start where they
    /// belong, so they move a few of their own radii rather than thirty, and
    /// the priors hold the geometry the photographs alone leave ambiguous -
    /// untextured and glossy surfaces, which a photometric loss would
    /// otherwise explain with floaters.
    pub fn from_dense_stereo(iters: usize, budget: usize, views: usize) -> FitCfg {
        FitCfg {
            lr_position: 5.0 * 4.65 / iters.max(1) as f32,
            depth_weight: 0.1,
            normal_prior_weight: 0.05,
            ..FitCfg::from_sparse_points(iters, budget, views)
        }
    }
}

/// One posed target view: its camera and the photograph's RGB `[W*H*3]` in
/// [0,1] as the camera recorded it - NATIVE pixels through the camera's own
/// lens - plus the optional per-pixel supervision RGB alone cannot carry.
#[derive(Clone)]
pub struct TargetView {
    pub cam: Camera,
    pub rgb: Vec<f32>,
    /// Per-pixel depth prior `[W*H]`: the RANGE to the surface along the
    /// pixel's own ray (not a z-depth), in the scene's world units, 0 = no
    /// data at that pixel.
    pub depth: Option<Vec<f32>>,
    /// How far to trust each depth pixel, `[W*H]`, `None` = fully.
    pub depth_conf: Option<Vec<f32>>,
    /// Per-pixel loss weight `[W*H]` in [0,1], `None` = supervise everything.
    /// It divides out of the normalizer too, so the reported loss stays the
    /// loss of the pixels that were actually supervised.
    pub mask: Option<Vec<f32>>,
    /// Which physical camera took this view: views of one sensor share the
    /// calibration a fit refines, and the lens/sensor parts of the camera
    /// model.
    pub sensor: usize,
    /// Exposure of this view in log2 stops relative to the others, when it is
    /// KNOWN (EXIF). The camera model fits a residual around it.
    pub exposure: f32,
    /// How `rgb` is encoded; consulted by a scene-linear fit only.
    pub encoding: Encoding,
    /// Per-pixel surface normal prior `[W*H*3]` in THIS camera's frame
    /// (+X right, +Y down, +Z forward), facing the camera; a zero vector =
    /// no prior at that pixel.
    pub normals: Option<Vec<f32>>,
}

impl TargetView {
    /// A view supervised by its colours alone.
    pub fn new(cam: Camera, rgb: Vec<f32>) -> TargetView {
        TargetView {
            cam,
            rgb,
            depth: None,
            depth_conf: None,
            mask: None,
            sensor: 0,
            exposure: 0.0,
            encoding: Encoding::Srgb,
            normals: None,
        }
    }

    /// Add a camera-frame surface normal prior `[W*H*3]`.
    pub fn with_normals(mut self, normals: Vec<f32>) -> TargetView {
        self.normals = Some(normals);
        self
    }

    /// Declare this view's exposure, in log2 stops relative to the capture.
    pub fn with_exposure(mut self, ev: f32) -> TargetView {
        self.exposure = ev;
        self
    }

    /// Declare which physical camera took this view.
    pub fn with_sensor(mut self, sensor: usize) -> TargetView {
        self.sensor = sensor;
        self
    }

    /// Declare how this view's pixels are encoded.
    pub fn with_encoding(mut self, encoding: Encoding) -> TargetView {
        self.encoding = encoding;
        self
    }

    /// Add a range prior `[W*H]` (0 = no data) and, optionally, how much to
    /// trust each of its pixels.
    pub fn with_depth(mut self, depth: Vec<f32>, conf: Option<Vec<f32>>) -> TargetView {
        self.depth = Some(depth);
        self.depth_conf = conf;
        self
    }

    /// Restrict supervision to the pixels `mask` `[W*H]` weights.
    pub fn with_mask(mut self, mask: Vec<f32>) -> TargetView {
        self.mask = Some(mask);
        self
    }

    /// This view at half the resolution: every per-pixel quantity averaged
    /// over 2x2 blocks (normals renormalized, range taken from the nearest
    /// sample of the block rather than averaged across an edge), the camera's
    /// intrinsics halved. An odd last row or column is dropped, which leaves
    /// pixel coordinates - and so the principal point - exactly halved; the
    /// lens's coefficients act on normalized coordinates and do not change.
    pub fn half(&self) -> TargetView {
        let (w, h) = (self.cam.width as usize, self.cam.height as usize);
        let (hw, hh) = ((w / 2).max(1), (h / 2).max(1));
        let down = |src: &[f32], ch: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; hw * hh * ch];
            for y in 0..hh {
                for x in 0..hw {
                    for c in 0..ch {
                        let at = |xx: usize, yy: usize| src[((2 * y + yy).min(h - 1) * w + (2 * x + xx).min(w - 1)) * ch + c];
                        out[(y * hw + x) * ch + c] = 0.25 * (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1));
                    }
                }
            }
            out
        };
        let normals = self.normals.as_ref().map(|n| {
            let mut m = down(n, 3);
            for v in m.chunks_exact_mut(3) {
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if l > 1e-6 {
                    v.iter_mut().for_each(|c| *c /= l);
                }
            }
            m
        });
        let cam = Camera {
            fx: self.cam.fx * 0.5,
            fy: self.cam.fy * 0.5,
            cx: self.cam.cx * 0.5,
            cy: self.cam.cy * 0.5,
            width: hw as u32,
            height: hh as u32,
            ..self.cam
        };
        TargetView {
            cam,
            rgb: down(&self.rgb, 3),
            depth: self.depth.as_ref().map(|d| {
                let mut out = vec![0.0f32; hw * hh];
                for y in 0..hh {
                    for x in 0..hw {
                        let mut best = 0.0f32;
                        for (xx, yy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let v = d[(2 * y + yy).min(h - 1) * w + (2 * x + xx).min(w - 1)];
                            if v > 0.0 && (best == 0.0 || v < best) {
                                best = v;
                            }
                        }
                        out[y * hw + x] = best;
                    }
                }
                out
            }),
            depth_conf: self.depth_conf.as_ref().map(|c| down(c, 1)),
            mask: self.mask.as_ref().map(|m| down(m, 1)),
            normals,
            sensor: self.sensor,
            exposure: self.exposure,
            encoding: self.encoding,
        }
    }

    /// [`TargetView::half`] for a fit in colour space `space`: an sRGB
    /// target of a scene-linear fit is averaged in linear light and encoded
    /// again, since the camera integrates light before it encodes; any other
    /// is averaged as stored.
    pub fn half_in(&self, space: ColorSpace) -> TargetView {
        if space != ColorSpace::SceneLinear || self.encoding != Encoding::Srgb {
            return self.half();
        }
        let mut h = TargetView { rgb: self.rgb.iter().map(|&v| crate::isp::srgb_decode(v)).collect(), ..self.clone() }.half();
        h.rgb.iter_mut().for_each(|v| *v = crate::isp::srgb_encode(*v));
        h
    }

    /// The per-pixel supervision weight a fit uses for this view: the mask,
    /// times the camera model's clipped-pixel weight when there is one.
    /// `None` = every pixel counts fully.
    fn weights(&self, isp: Option<&Isp>) -> Option<Vec<f32>> {
        let clip = isp.filter(|i| i.cfg().clip > 0.0).map(|i| i.clip_weight(&self.rgb));
        match (&self.mask, clip) {
            (None, c) => c,
            (Some(m), None) => Some(m.clone()),
            (Some(m), Some(c)) => Some(m.iter().zip(&c).map(|(a, b)| a * b).collect()),
        }
    }

    /// `{range, confidence, normal (3)}` per pixel, as `splat_geom_loss`
    /// reads it; `None` when the view carries neither prior.
    fn geometry_target(&self) -> Option<Vec<f32>> {
        if self.depth.is_none() && self.normals.is_none() {
            return None;
        }
        let px = (self.cam.width * self.cam.height) as usize;
        let mut out = vec![0.0f32; px * 5];
        for i in 0..px {
            if let Some(d) = &self.depth {
                out[i * 5] = d[i];
                out[i * 5 + 1] = self.depth_conf.as_ref().map_or(1.0, |c| c[i]);
            }
            if let Some(n) = &self.normals {
                out[i * 5 + 2..i * 5 + 5].copy_from_slice(&n[i * 3..i * 3 + 3]);
            }
        }
        Some(out)
    }
}

/// Everything a fit produces.
pub struct FitResult {
    pub scene: Splats,
    /// The cameras, refined as [`FitCfg::camera`] asked.
    pub cams: Vec<Camera>,
    /// The objective at the last iteration.
    pub loss: f32,
    /// The fitted camera model, when [`FitCfg::isp`] asked for one.
    pub isp: Option<Isp>,
    /// Mip-Splatting's 3D filter variance per gaussian of `scene`, in world
    /// units² - part of the scene: it is what the scene was fitted under.
    pub filter3d: Vec<f32>,
    /// The fitted environment, when [`FitCfg::environment`] asked for one.
    pub env: Option<EnvMap>,
}

impl FitResult {
    /// The scene with its 3D filter baked into the gaussians
    /// ([`crate::mip::bake_filter3d`]): what to hand to anything that renders
    /// without the filter. Keep [`FitResult::scene`] and
    /// [`FitResult::filter3d`] separate to go on fitting.
    pub fn baked(&self) -> Splats {
        crate::mip::bake_filter3d(&self.scene, &self.filter3d)
    }
}

/// Fit `init` against the targets; returns the optimized scene - with the 3D
/// filter it was fitted under baked in, so it renders as it was fitted with
/// no filter at all - and the objective it ends at. `on_step(iter, loss)` is
/// polled once per completed iteration; returning `false` aborts at the end
/// of that iteration.
pub fn fit(gpu: &Gpu, ks: Kernels, init: &Splats, targets: &[TargetView], cfg: &FitCfg, on_step: &mut dyn FnMut(usize, f32) -> bool) -> (Splats, f32) {
    let r = fit_full(gpu, ks, init, targets, cfg, on_step);
    (r.baked(), r.loss)
}

/// Fit the scene AND the cameras, returning the refined poses (the scene
/// baked as [`fit`] returns it).
pub fn fit_bundle(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    targets: &[TargetView],
    cfg: &FitCfg,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> (Splats, Vec<Camera>, f32) {
    let r = fit_full(gpu, ks, init, targets, cfg, on_step);
    (r.baked(), r.cams, r.loss)
}

/// Fit the scene, and whatever else `cfg` asks to be fitted with it - the
/// cameras, the photometric camera model - returning all of it.
pub fn fit_full(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    targets: &[TargetView],
    cfg: &FitCfg,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> FitResult {
    assert!(!targets.is_empty(), "a fit needs at least one target view");
    let mut isp = cfg.isp.map(|c| {
        let views: Vec<(usize, f32, Encoding)> = targets.iter().map(|t| (t.sensor, t.exposure, t.encoding)).collect();
        Isp::new(c, &views)
    });
    // Fit in a frame where the scene is about one unit across.
    //
    // A reconstruction has no intrinsic units, but the optimizer is full of
    // quantities that do: every learning rate is a fixed DISTANCE, and a
    // scene a hundred times too big barely moves while one a hundred times
    // too small overshoots every step - measured across that range the same
    // fit varied by 8 dB. Normalising once at the door is exactly
    // equivalent: intrinsics are in pixels, so scaling the world and the
    // camera positions together changes no rendered image.
    let k = scene_unit(init, targets);
    let scaled_init = rescale_scene(init, k);
    let scaled: Vec<TargetView> = targets
        .iter()
        .map(|t| {
            let mut v = t.clone();
            v.cam = rescale_cam(&t.cam, k);
            // a range prior is in world units and rescales with the world
            v.depth = t.depth.as_ref().map(|d| d.iter().map(|v| v * k).collect());
            v
        })
        .collect();
    let (scene, cams, loss, filter3d, env) = Fit::new(gpu, ks, cfg, &scaled_init, &scaled, isp.as_mut()).run(on_step);
    if let (Some(i), true) = (&isp, cfg.log_every > 0) {
        print!("fit: camera model\n{}", i.summary());
    }
    FitResult {
        scene: rescale_scene(&scene, 1.0 / k),
        cams: cams.iter().map(|c| rescale_cam(c, 1.0 / k)).collect(),
        loss,
        isp,
        filter3d: filter3d.iter().map(|v| v / (k * k)).collect(),
        env,
    }
}

/// The environment being fitted: its coefficients, on the host and the
/// device, and their Adam state.
struct EnvFit {
    map: EnvMap,
    dev: EnvDevice,
    grad: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl EnvFit {
    /// Starts uniform at the photographs' mean colour: at zero the clamp
    /// that keeps radiance non-negative would pass no gradient at all.
    fn new(gpu: &Gpu, degree: u32, targets: &[TargetView]) -> EnvFit {
        let (mut sum, mut n) = ([0.0f64; 3], 0usize);
        for t in targets {
            for p in t.rgb.chunks_exact(3) {
                for c in 0..3 {
                    sum[c] += p[c] as f64;
                }
                n += 1;
            }
        }
        let map = EnvMap::uniform(degree, sum.map(|v| (v / n.max(1) as f64) as f32));
        let n = map.coeffs.len();
        EnvFit { dev: EnvDevice::new(gpu, &map), map, grad: vec![0.0; n], m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    /// One Adam step on the gradient accumulated since the last one.
    fn step(&mut self, gpu: &Gpu, lr: f32) {
        self.t += 1;
        let (b1, b2) = (0.9f64, 0.999f64);
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for i in 0..self.map.coeffs.len() {
            let g = self.grad[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            self.map.coeffs[i] -= (lr as f64 * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-15)) as f32;
            self.grad[i] = 0.0;
        }
        self.dev.upload(gpu, &self.map);
    }
}

/// Where an iteration's wall clock goes: every phase below ends at a device
/// sync, so host timing is a faithful account. `BRAIN_SPLAT_PROFILE=1`
/// prints it.
#[derive(Default)]
struct Prof {
    on: bool,
    t: Vec<(&'static str, f64)>,
}

impl Prof {
    fn new() -> Prof {
        Prof { on: std::env::var_os("BRAIN_SPLAT_PROFILE").is_some(), t: Vec::new() }
    }
    fn add(&mut self, k: &'static str, d: std::time::Duration) {
        if !self.on {
            return;
        }
        match self.t.iter_mut().find(|(n, _)| *n == k) {
            Some(e) => e.1 += d.as_secs_f64(),
            None => self.t.push((k, d.as_secs_f64())),
        }
    }
    fn report(&self, iters: usize, n: usize) {
        if !self.on || iters == 0 {
            return;
        }
        let total: f64 = self.t.iter().map(|(_, v)| v).sum();
        println!("\nfit profile: {n} gaussians, {iters} iters, {total:.1}s total");
        let mut rows = self.t.clone();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (k, v) in rows {
            println!("  {k:22} {:8.1} ms/iter  {:5.1}%", 1e3 * v / iters as f64, 100.0 * v / total);
        }
    }
}

/// The per-view device state of one resolution level.
struct Level {
    targets: Vec<TargetView>,
    tgt: Vec<DeviceBuffer>,
    wt: Vec<Option<DeviceBuffer>>,
    wsum: Vec<f64>,
    geo: Vec<Option<DeviceBuffer>>,
    edges: Vec<Vec<f32>>,
    /// How many halvings below full resolution.
    halvings: u32,
}

/// Adam state of one camera's pose (and rolling shutter).
#[derive(Clone, Copy, Default)]
struct PoseState {
    m: [f64; 12],
    v: [f64; 12],
    t: i32,
}

/// Adam state of one sensor's calibration.
#[derive(Clone, Default)]
struct LensState {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

/// The device scratch one fit's iterations share.
struct Scratch {
    /// Gaussians the renderer and the backward scratch are sized for.
    cap: usize,
    renderer: Renderer,
    bscr: BwdScratch,
    dloss: DeviceLoss,
    dimg: DeviceBuffer,
    daux: DeviceBuffer,
    pred: DeviceBuffer,
    ones: DeviceBuffer,
    no_geo: DeviceBuffer,
    geom_partial: DeviceBuffer,
    /// The camera model's device side, when the fit has one.
    isp: Option<DeviceIsp>,
}

/// One fit, in the normalized frame.
struct Fit<'a> {
    gpu: &'a Gpu,
    ks: Kernels,
    cfg: &'a FitCfg,
    init: &'a Splats,
    full: &'a [TargetView],
    isp: Option<&'a mut Isp>,
    /// The cameras at FULL resolution, refined in place.
    cams: Vec<Camera>,
    /// Their starting point, for the gauge and for handing back the input.
    cams0: Vec<Camera>,
    pose: Vec<PoseState>,
    lens: Vec<LensState>,
    /// Each gaussian's largest axis when it entered the fit, carried along
    /// density control's origin map, for [`FitCfg::max_growth`].
    start: Vec<f32>,
    env: Option<EnvFit>,
    prof: Prof,
}

impl<'a> Fit<'a> {
    fn new(gpu: &'a Gpu, ks: Kernels, cfg: &'a FitCfg, init: &'a Splats, full: &'a [TargetView], isp: Option<&'a mut Isp>) -> Fit<'a> {
        let cams: Vec<Camera> = full.iter().map(|t| t.cam).collect();
        let sensors = full.iter().map(|t| t.sensor).max().unwrap_or(0) + 1;
        Fit {
            gpu,
            ks,
            cfg,
            init,
            full,
            isp,
            cams0: cams.clone(),
            cams,
            pose: vec![PoseState::default(); full.len()],
            lens: vec![LensState::default(); sensors],
            start: largest_axes(init),
            env: cfg.environment.map(|d| EnvFit::new(gpu, d, full)),
            prof: Prof::new(),
        }
    }

    fn opts(&self) -> RenderOpts {
        RenderOpts { mode: Mode::Color, eps2d: self.cfg.eps2d, antialiased: self.cfg.antialiased, ray: true, ..Default::default() }
    }

    /// The level whose views are `halvings` below full resolution, with the
    /// cameras as refined so far.
    fn level(&self, halvings: u32) -> Level {
        let mut targets: Vec<TargetView> = self.full.to_vec();
        for (t, c) in targets.iter_mut().zip(&self.cams) {
            t.cam = *c;
        }
        let space = self.cfg.isp.map_or(ColorSpace::Display, |i| i.color_space);
        for _ in 0..halvings {
            targets = targets.iter().map(|t| t.half_in(space)).collect();
        }
        let isp = self.isp.as_deref();
        let weights: Vec<Option<Vec<f32>>> = targets.iter().map(|t| t.weights(isp)).collect();
        let wsum = targets
            .iter()
            .zip(&weights)
            .map(|(t, w)| w.as_ref().map_or((t.cam.width * t.cam.height) as f64, |m| m.iter().map(|&v| v as f64).sum()))
            .collect();
        let gpu = self.gpu;
        Level {
            tgt: targets.iter().map(|t| gpu.storage_init("fit.target", &t.rgb)).collect(),
            wt: weights.iter().map(|w| w.as_ref().map(|m| gpu.storage_init("fit.weight", m))).collect(),
            geo: targets.iter().map(|t| t.geometry_target().map(|g| gpu.storage_init("fit.geometry", &g))).collect(),
            edges: if self.cfg.strategy == Densify::Hybrid && self.cfg.densify_every > 0 {
                targets.iter().map(|t| density::edge_map(&t.rgb, t.cam.width as usize, t.cam.height as usize)).collect()
            } else {
                Vec::new()
            },
            wsum,
            targets,
            halvings,
        }
    }

    /// How many halvings iteration `it` runs at.
    fn halvings_at(&self, it: usize) -> u32 {
        let levels = self.cfg.pyramid;
        let coarse_end = (self.cfg.coarse.clamp(0.0, 1.0) * self.cfg.iters as f32) as usize;
        if levels == 0 || it >= coarse_end {
            return 0;
        }
        let per = coarse_end.div_ceil(levels as usize).max(1);
        levels - (it / per) as u32
    }

    fn run(mut self, on_step: &mut dyn FnMut(usize, f32) -> bool) -> (Splats, Vec<Camera>, f32, Vec<f32>, Option<EnvMap>) {
        let cfg = self.cfg;
        let gpu = self.gpu;
        let n_views = self.full.len();
        let views_per = if cfg.batch == 0 || cfg.batch >= n_views { n_views } else { cfg.batch };
        let epoch = n_views.div_ceil(views_per);
        let (maxw, maxh) = self.full.iter().fold((0u32, 0u32), |(w, h), t| (w.max(t.cam.width), h.max(t.cam.height)));
        let max_px = (maxw * maxh) as usize;
        let cap = self.init.len().max(cfg.max_gaussians).max(1);
        let ones = gpu.storage(max_px as u64);
        gpu.write_f32(&ones, &vec![1.0; max_px]);
        let no_geo = gpu.storage(5 * max_px as u64);
        gpu.submit(&[&no_geo], &[]);
        let mut scr = Scratch {
            cap,
            renderer: Renderer::new(gpu, self.ks, cap, maxw, maxh, 0).growable(),
            bscr: BwdScratch::new(gpu, cap, max_px, 0),
            dloss: DeviceLoss::new(gpu, max_px),
            dimg: gpu.storage(4 * max_px as u64),
            daux: gpu.storage(5 * max_px as u64),
            pred: gpu.storage(4 * max_px as u64),
            ones,
            no_geo,
            geom_partial: gpu.storage((4 * dispatched_groups(max_px)) as u64),
            isp: self.isp.as_deref().map(|m| DeviceIsp::new(gpu, m, max_px)),
        };

        let mut scene = DeviceScene::new(gpu, self.init, cfg.sh_degree);
        self.refresh_bounds(&scene, None);
        // What the scene as handed in scores over every view at full
        // resolution: the floor the fit has to beat to have been worth
        // running.
        let full = self.level(0);
        let first_loss = self.dataset_loss(&scene, &mut scr, &full);
        drop(full);
        let mut level = self.level(self.halvings_at(0));
        let mut starved: Vec<f32> = Vec::new();
        let mut absgrad = vec![0.0f32; scene.n];
        let mut smooth = f64::NAN;
        let mut last_loss = f32::NAN;
        let (mut lr_scale, mut rises, mut prev) = (1.0f32, 0u32, f32::INFINITY);
        let (mut window_sum, mut window_n, mut grace) = (0.0f64, 0usize, true);
        let until = if cfg.densify_until > 0 { cfg.densify_until.min(cfg.iters) } else { cfg.iters };
        let first_round = cfg.densify_after.max(cfg.densify_every);
        let rounds = if cfg.densify_every > 0 { until.saturating_sub(first_round).div_ceil(cfg.densify_every) } else { 0 };
        let o = self.opts();
        let mut done = 0usize;
        while done < cfg.iters {
            let it = done;
            let halvings = self.halvings_at(it);
            if halvings != level.halvings {
                level = self.level(halvings);
                // a finer level measures a different loss: its trend starts
                // over
                (smooth, rises, prev) = (f64::NAN, 0, f32::INFINITY);
                (window_sum, window_n, grace) = (0.0, 0, true);
            }
            let progress = it as f32 / cfg.iters as f32;
            let tm = std::time::Instant::now();
            scene.activate(gpu, &self.ks);
            scene.zero_grads(gpu);
            self.prof.add("activate + zero", tm.elapsed());
            let refine_pose = progress >= cfg.camera.pose_after;
            let refine_lens = progress >= cfg.camera.intrinsics_after;
            let geometry_now = progress >= cfg.geometry_after;
            let skip = sh_skip(cfg, scene.ksh, it);
            let mut loss_sum = 0.0f64;
            let batch = batch_views(n_views, views_per, it);
            let mut cam_grads: Vec<(usize, CameraGrad)> = Vec::new();
            for &vi in &batch {
                if level.wsum[vi] <= 0.0 {
                    continue;
                }
                let cam = level.targets[vi].cam;
                let tm = std::time::Instant::now();
                let eye = cam.eye();
                scene.shade(gpu, &self.ks, eye, skip);
                let gs = scene.splats();
                scr.renderer.render(gpu, &gs, &cam, &o);
                if let Some(e) = &self.env {
                    e.dev.composite(gpu, &self.ks, &scr.renderer.img, &cam, &o);
                }
                self.prof.add("render forward", tm.elapsed());
                // the render is radiance; the photograph is what this view's
                // camera made of it
                let tm = std::time::Instant::now();
                if let (Some(model), Some(dev)) = (self.isp.as_deref(), &scr.isp) {
                    dev.forward(gpu, self.ks, model, Shot::Training(vi), &cam, &scr.renderer.img, &scr.pred);
                }
                let pred = if self.isp.is_some() { &scr.pred } else { &scr.renderer.img };
                loss_sum += scr.dloss.eval(gpu, self.ks, cfg.loss, pred, &level.tgt[vi], level.wt[vi].as_ref(), level.wsum[vi], cam.width, cam.height, &scr.dimg);
                if let (Some(model), Some(dev)) = (self.isp.as_deref_mut(), &scr.isp) {
                    dev.backward(gpu, self.ks, model, vi, &cam, &scr.renderer.img, &scr.dimg);
                }
                if let Some(e) = self.env.as_mut() {
                    let g = e.dev.backward(gpu, &self.ks, &scr.renderer.img, &scr.dimg, &cam, &o);
                    for (a, b) in e.grad.iter_mut().zip(&g) {
                        *a += b;
                    }
                }
                self.prof.add("pixel loss", tm.elapsed());
                // geometry terms: external priors throughout, regularizers
                // once the scene has a shape
                let tm = std::time::Instant::now();
                let (wd, wn) = (cfg.depth_weight, cfg.normal_prior_weight);
                let (wself, wdist) = if geometry_now { (cfg.normal_consistency_weight, cfg.distortion_weight) } else { (0.0, 0.0) };
                let has_prior = level.geo[vi].is_some();
                let geometry = (has_prior && (wd > 0.0 || wn > 0.0)) || wself > 0.0 || wdist > 0.0;
                if geometry {
                    loss_sum += self.geometry(&scr, &level, vi, [wd, wn, wself, wdist]);
                }
                self.prof.add("geometry terms", tm.elapsed());
                let tm = std::time::Instant::now();
                let camera = (refine_pose && (vi > 0 || cfg.lr_position <= 0.0)) || refine_lens;
                let cg = scr
                    .renderer
                    .render_bwd_ray(gpu, &gs, &cam, &o, &scr.dimg, geometry.then_some(&scr.daux), &mut scr.bscr, &scene.grads, camera)
                    .unwrap_or_else(|e| panic!("{e}"));
                if let Some(cg) = cg {
                    cam_grads.push((vi, cg));
                }
                scene.shade_vjp(gpu, &self.ks, eye, skip);
                self.prof.add("render backward", tm.elapsed());
            }
            let tm = std::time::Instant::now();
            let decay = 0.01f32.powf(progress);
            let rates = Rates {
                position: cfg.lr_position * decay * lr_scale,
                scale: cfg.lr_scale * lr_scale,
                rotation: cfg.lr_rotation * lr_scale,
                opacity: cfg.lr_opacity * lr_scale,
                color: cfg.lr_color * lr_scale,
                sh: cfg.lr_color / 20.0 * lr_scale,
                relative: cfg.relative_position,
                noise: cfg.noise,
                opacity_reg: cfg.opacity_reg,
                scale_reg: cfg.scale_reg,
            };
            let bounds = Bounds { min_scale: cfg.min_scale, max_needle: cfg.max_needle, max_flat: cfg.max_flat };
            scene.step(gpu, &self.ks, it + 1, &rates, &bounds);
            self.prof.add("optimizer", tm.elapsed());
            if let Some(model) = self.isp.as_deref_mut() {
                model.step(progress);
            }
            if let Some(e) = self.env.as_mut() {
                e.step(gpu, cfg.lr_color * lr_scale);
            }
            self.refine_cameras(&cam_grads, refine_pose, refine_lens, &mut level);
            if cfg.densify_every > 0 && (it + 1) % cfg.densify_every + 4 * epoch >= cfg.densify_every {
                // the positional-gradient statistic over the epochs before a
                // round
                let ag = gpu.read(&scene.grads.d_absgrad, scene.n);
                for (a, b) in absgrad.iter_mut().zip(&ag) {
                    *a += b;
                }
            }

            // an epoch-length average of the batch losses
            let batch_loss = loss_sum / batch.len().max(1) as f64;
            let alpha = batch.len() as f64 / n_views as f64;
            smooth = if smooth.is_nan() { batch_loss } else { (1.0 - alpha) * smooth + alpha * batch_loss };
            last_loss = smooth as f32;
            // Back off a step size this scene will not take: the mean loss of
            // three windows in a row rising is a trend, not an unlucky draw. A
            // window is at least ten iterations and at least an epoch, so it
            // averages over the views; a change of resolution or of the
            // population changes what the loss measures, so the comparison
            // restarts at both - and skips the window right after, while new
            // gaussians settle.
            window_sum += batch_loss;
            window_n += 1;
            if window_n >= epoch.max(10) {
                let mean = (window_sum / window_n as f64) as f32;
                if prev.is_finite() {
                    if mean > prev {
                        rises += 1;
                    } else {
                        rises = 0;
                    }
                }
                prev = if grace { f32::INFINITY } else { mean };
                grace = false;
                (window_sum, window_n) = (0.0, 0);
            }
            if rises >= 3 && lr_scale > 1e-3 {
                lr_scale *= 0.5;
                rises = 0;
                if cfg.log_every > 0 {
                    println!("fit iter {it:5}: loss rising, step sizes x{lr_scale:.3}");
                }
            }
            if cfg.log_every > 0 && (it.is_multiple_of(cfg.log_every) || it + 1 == cfg.iters) {
                let c = level.targets[0].cam;
                println!("fit iter {it:5}: loss {last_loss:.6}, {} gaussians at {}x{}", scene.n, c.width, c.height);
            }
            done += 1;
            if !on_step(it, last_loss) {
                break;
            }
            // ---- density control ----
            if cfg.densify_every > 0 && done >= first_round && done < until && done.is_multiple_of(cfg.densify_every) {
                let round = (done - first_round) / cfg.densify_every;
                scene = self.densify(scene, &mut scr, &mut level, &absgrad, &mut starved, round, rounds, it);
                absgrad = vec![0.0; scene.n];
                if scene.n > scr.cap {
                    // unbudgeted growth: the scratch sized for the start has
                    // to follow the scene
                    scr.cap = scene.n + scene.n / 4;
                    scr.renderer = Renderer::new(gpu, self.ks, scr.cap, maxw, maxh, 0).growable();
                    scr.bscr = BwdScratch::new(gpu, scr.cap, max_px, 0);
                }
                (smooth, rises, prev) = (f64::NAN, 0, f32::INFINITY);
                (window_sum, window_n, grace) = (0.0, 0, true);
            }
        }
        self.prof.report(done, scene.n);
        // Never hand back something worse than what came in, judged the same
        // way both times: every view, full resolution.
        let full = self.level(0);
        let final_loss = self.dataset_loss(&scene, &mut scr, &full);
        let filter3d = gpu.read(&scene.filter3d, scene.n);
        let out = scene.download(gpu);
        if cfg.log_every > 0 {
            println!("fit: loss over every view {first_loss:.6} -> {final_loss:.6} ({last_loss:.6} at the end of the schedule)");
        }
        if first_loss.is_finite() && (!final_loss.is_finite() || final_loss > first_loss) {
            if cfg.log_every > 0 {
                println!("fit: ended worse than it started; keeping the input scene");
            }
            return (self.init.clone(), self.cams0.clone(), first_loss, vec![0.0; self.init.len()], self.cfg.environment.map(EnvMap::new));
        }
        (out, self.cams, final_loss, filter3d, self.env.map(|e| e.map))
    }

    /// The whole objective over every view of `level`, forward only - the
    /// photometric loss and every geometry term with the weights it ends the
    /// fit with. The guard compares this, the same function at both ends:
    /// with the geometry left out, a fit that correctly moves a scene to the
    /// depth its prior says renders a little worse in RGB on the way, and was
    /// handed back unmoved.
    fn dataset_loss(&self, scene: &DeviceScene, scr: &mut Scratch, level: &Level) -> f32 {
        let gpu = self.gpu;
        let cfg = self.cfg;
        scene.activate(gpu, &self.ks);
        let o = self.opts();
        let skip = sh_skip(cfg, scene.ksh, cfg.iters);
        let (wself, wdist) = if cfg.geometry_after < 1.0 { (cfg.normal_consistency_weight, cfg.distortion_weight) } else { (0.0, 0.0) };
        let mut sum = 0.0f64;
        for (vi, t) in level.targets.iter().enumerate() {
            let cam = t.cam;
            scene.shade(gpu, &self.ks, cam.eye(), skip);
            scr.renderer.render(gpu, &scene.splats(), &cam, &o);
            if let Some(e) = &self.env {
                e.dev.composite(gpu, &self.ks, &scr.renderer.img, &cam, &o);
            }
            let pred = match (self.isp.as_deref(), &scr.isp) {
                (Some(model), Some(dev)) => {
                    dev.forward(gpu, self.ks, model, Shot::Training(vi), &cam, &scr.renderer.img, &scr.pred);
                    &scr.pred
                }
                _ => &scr.renderer.img,
            };
            sum += scr.dloss.eval(gpu, self.ks, cfg.loss, pred, &level.tgt[vi], level.wt[vi].as_ref(), level.wsum[vi], cam.width, cam.height, &scr.dimg);
            let (wd, wn) = (cfg.depth_weight, cfg.normal_prior_weight);
            if (level.geo[vi].is_some() && (wd > 0.0 || wn > 0.0)) || wself > 0.0 || wdist > 0.0 {
                sum += self.geometry(scr, level, vi, [wd, wn, wself, wdist]);
            }
        }
        (sum / level.targets.len().max(1) as f64) as f32
    }

    /// The geometry objective of view `vi` on the device: writes `daux`,
    /// adds the expected-range share to `dimg`'s alpha, returns the terms'
    /// sum.
    fn geometry(&self, scr: &Scratch, level: &Level, vi: usize, w: [f32; 4]) -> f64 {
        let gpu = self.gpu;
        let cam = level.targets[vi].cam;
        let px = (cam.width * cam.height) as usize;
        let mut params = crate::renderer::ray_view_params(0, &cam, &self.opts()).to_vec();
        params.extend_from_slice(&[
            gpu_core::f(w[0]),
            gpu_core::f(w[1]),
            gpu_core::f(w[2]),
            gpu_core::f(w[3]),
            gpu_core::f(self.cfg.depth_delta),
            gpu_core::f((1.0 / level.wsum[vi]) as f32),
            gpu_core::f(crate::renderer::MIN_DEPTH_ALPHA),
            level.wt[vi].is_some() as u32,
        ]);
        let step = gpu.dispatch(
            self.ks.splat_geom_loss,
            &[
                &scr.renderer.aux,
                &scr.renderer.img,
                level.geo[vi].as_ref().unwrap_or(&scr.no_geo),
                level.wt[vi].as_ref().unwrap_or(&scr.ones),
                &scr.dimg,
                &scr.daux,
                &scr.geom_partial,
            ],
            &params,
            gpu_core::Dispatch::Workgroups(px.div_ceil(64) as u32),
        );
        gpu.submit(&[], &[step]);
        gpu.read(&scr.geom_partial, 4 * dispatched_groups(px)).iter().map(|v| *v as f64).sum()
    }

    /// One Adam step of every camera the batch measured, with the gauge held.
    fn refine_cameras(&mut self, grads: &[(usize, CameraGrad)], pose: bool, lens: bool, level: &mut Level) {
        if grads.is_empty() {
            return;
        }
        let cr = self.cfg.camera;
        let (b1, b2) = (0.9f64, 0.999f64);
        let adam = |m: &mut f64, v: &mut f64, g: f64, t: i32, lr: f64| -> f64 {
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            -lr * (*m / (1.0 - b1.powi(t))) / ((*v / (1.0 - b2.powi(t))).sqrt() + 1e-12)
        };
        if pose {
            // Gauge: while the scene moves too, the first camera does not
            // move, and the second may not move along the line between them -
            // together those fix the similarity the images cannot see. A scene
            // held still is its own gauge, and every camera is free.
            let gauge = self.cfg.lr_position > 0.0;
            let baseline = (gauge && self.cams0.len() > 1).then(|| {
                let (a, b) = (self.cams0[0].eye(), self.cams0[1].eye());
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
                d.map(|v| (v / l) as f64)
            });
            for &(vi, g) in grads {
                if gauge && vi == 0 {
                    continue;
                }
                let st = &mut self.pose[vi];
                st.t += 1;
                let mut step = [0.0f64; 12];
                for k in 0..3 {
                    step[k] = adam(&mut st.m[k], &mut st.v[k], g.rotation[k], st.t, cr.pose_lr as f64);
                    step[3 + k] = adam(&mut st.m[3 + k], &mut st.v[3 + k], g.translation[k], st.t, cr.pose_lr as f64);
                }
                if cr.shutter {
                    for k in 0..6 {
                        step[6 + k] = adam(&mut st.m[6 + k], &mut st.v[6 + k], g.shutter[k], st.t, cr.pose_lr as f64);
                    }
                }
                let c = &mut self.cams[vi];
                if let (1, Some(b)) = (vi, baseline) {
                    // the translation step in the world, less its component
                    // along the baseline
                    let r = rot3(&c.c2w);
                    let tw: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| r[i * 3 + k] * step[3 + k]).sum());
                    let along = tw[0] * b[0] + tw[1] * b[1] + tw[2] * b[2];
                    let tw = [tw[0] - along * b[0], tw[1] - along * b[1], tw[2] - along * b[2]];
                    for k in 0..3 {
                        step[3 + k] = (0..3).map(|i| r[i * 3 + k] * tw[i]).sum();
                    }
                }
                c.c2w = compose_local(&c.c2w, &[step[0], step[1], step[2]], &[step[3], step[4], step[5]]);
                for k in 0..6 {
                    c.shutter[k] += step[6 + k] as f32;
                }
            }
        }
        if lens {
            for s in 0..self.lens.len() {
                let views: Vec<&CameraGrad> = grads.iter().filter(|(vi, _)| self.full[*vi].sensor == s).map(|(_, g)| g).collect();
                let Some(v0) = grads.iter().find(|(vi, _)| self.full[*vi].sensor == s).map(|(vi, _)| *vi) else { continue };
                let k = self.cams[v0].intrinsics();
                let np = k.param_count();
                let mut g = vec![0.0f64; np];
                for cg in &views {
                    for (gi, cgi) in g.iter_mut().zip(&cg.lens) {
                        *gi += cgi;
                    }
                }
                let st = &mut self.lens[s];
                if st.m.len() != np {
                    st.m = vec![0.0; np];
                    st.v = vec![0.0; np];
                }
                st.t += 1;
                let mut p = k.params();
                // square pixels: one focal length, stepped in its logarithm
                let df = adam(&mut st.m[0], &mut st.v[0], g[0] * k.fx + g[1] * k.fy, st.t, cr.focal_lr as f64);
                p[0] *= df.exp();
                p[1] *= df.exp();
                if cr.principal {
                    for i in 2..4 {
                        p[i] += adam(&mut st.m[i], &mut st.v[i], g[i], st.t, cr.principal_lr as f64);
                    }
                }
                for i in 4..np {
                    p[i] += adam(&mut st.m[i], &mut st.v[i], g[i], st.t, cr.distortion_lr as f64);
                }
                let k2 = k.with_params(&p);
                for (vi, t) in self.full.iter().enumerate() {
                    if t.sensor == s {
                        let c = self.cams[vi];
                        self.cams[vi] = Camera { shutter: c.shutter, ..Camera::with_intrinsics(c.c2w, &k2) };
                    }
                }
            }
        }
        // the level's cameras follow, at its resolution
        let s = 0.5f32.powi(level.halvings as i32);
        for (t, c) in level.targets.iter_mut().zip(&self.cams) {
            let (w, h) = (t.cam.width, t.cam.height);
            t.cam = Camera { fx: c.fx * s, fy: c.fy * s, cx: c.cx * s, cy: c.cy * s, width: w, height: h, ..*c };
        }
    }

    /// Recompute the scene-dependent bounds: the 3D filter and the scale
    /// ceiling. `seen[v][i]` says whether view `v` saw gaussian `i` at all;
    /// without it a camera counts wherever the centre lands in its frame.
    fn refresh_bounds(&self, scene: &DeviceScene, seen: Option<&[Vec<bool>]>) {
        let host = scene.download(self.gpu);
        let visible = |i: usize, v: usize| seen.is_none_or(|s| s[v][i]);
        if self.cfg.mip_scale > 0.0 {
            let sig = crate::mip::smoothing_sigma(&host, &self.cams, self.cfg.mip_scale, &visible);
            scene.set_filter3d(self.gpu, &sig.iter().map(|s| s * s).collect::<Vec<f32>>());
        }
        if self.cfg.max_scale_pixels > 0.0 || self.cfg.max_growth > 0.0 {
            let mut ceil = if self.cfg.max_scale_pixels > 0.0 {
                crate::mip::smoothing_sigma(&host, &self.cams, self.cfg.max_scale_pixels, &visible)
                    .into_iter()
                    .map(|v| if v > 0.0 { v } else { f32::MAX })
                    .collect()
            } else {
                vec![f32::MAX; host.len()]
            };
            if self.cfg.max_growth > 0.0 {
                for (c, s) in ceil.iter_mut().zip(&self.start) {
                    *c = c.min(s * self.cfg.max_growth);
                }
            }
            scene.set_max_scale(self.gpu, &ceil);
        }
    }

    /// One density-control round: evidence over EVERY view at the current
    /// parameters, the strategy's move, and the scene rebuilt with its
    /// optimizer state carried along the move.
    #[allow(clippy::too_many_arguments)]
    fn densify(
        &mut self,
        scene: DeviceScene,
        scr: &mut Scratch,
        level: &mut Level,
        absgrad: &[f32],
        starved: &mut Vec<f32>,
        round: usize,
        rounds: usize,
        it: usize,
    ) -> DeviceScene {
        let gpu = self.gpu;
        let cfg = self.cfg;
        let tm = std::time::Instant::now();
        scene.activate(gpu, &self.ks);
        let n = scene.n;
        // Credit assignment: a backward whose upstream is (1, r, e r) per
        // pixel returns per gaussian its contribution, residual and edge
        // residual summed by its own compositing weights - and says which
        // views saw it at all, which is what the 3D filter is bounded by.
        let mut ev = Evidence::new(n);
        let mut seen: Vec<Vec<bool>> = Vec::with_capacity(level.targets.len());
        let o = self.opts();
        let skip = sh_skip(cfg, scene.ksh, it);
        let (mut unexplained, mut total) = (0.0f64, 0.0f64);
        for (vi, t) in level.targets.iter().enumerate() {
            let cam = t.cam;
            scene.shade(gpu, &self.ks, cam.eye(), skip);
            let gs = scene.splats();
            scr.renderer.render(gpu, &gs, &cam, &o);
            if let Some(e) = &self.env {
                e.dev.composite(gpu, &self.ks, &scr.renderer.img, &cam, &o);
            }
            let rgba = scr.renderer.read_rgba(gpu, cam.width, cam.height);
            let rgb = crate::renderer::rgba_to_rgb(&rgba);
            let pred = match (self.isp.as_deref(), &scr.isp) {
                (Some(model), Some(dev)) => {
                    dev.forward(gpu, self.ks, model, Shot::Training(vi), &cam, &scr.renderer.img, &scr.pred);
                    crate::renderer::rgba_to_rgb(&gpu.read(&scr.pred, 4 * (cam.width * cam.height) as usize))
                }
                _ => rgb,
            };
            let mut weights = t.weights(self.isp.as_deref());
            if cfg.transients {
                let (w, h) = (cam.width as usize, cam.height as usize);
                let r: Vec<f32> = (0..w * h).map(|p| (0..3).map(|c| (pred[p * 3 + c] - t.rgb[p * 3 + c]).abs()).sum::<f32>() / 3.0).collect();
                let keep = crate::loss::transient_mask(&r, w, h);
                let wt: Vec<f32> = match &weights {
                    Some(b) => b.iter().zip(&keep).map(|(a, k)| a * k).collect(),
                    None => keep,
                };
                level.wsum[vi] = wt.iter().map(|&v| v as f64).sum();
                level.wt[vi] = Some(gpu.storage_init("fit.weight", &wt));
                weights = Some(wt);
            }
            let aux = scr.renderer.read_aux(gpu, cam.width, cam.height);
            let (sites, u, a) = residual_sites(t, &pred, &rgba, &aux, weights.as_deref(), round as u64 ^ (vi as u64) << 32);
            ev.sites.extend(sites);
            unexplained += u;
            total += a;
            let edges = level.edges.get(vi).cloned().unwrap_or_else(|| vec![0.0; (cam.width * cam.height) as usize]);
            let alpha: Vec<f32> = rgba.chunks_exact(4).map(|p| p[3]).collect();
            let rendered: Vec<f32> = aux.chunks_exact(5).map(|a| a[0]).collect();
            let geom = t.depth.as_ref().map(|prior| density::range_residual(&rendered, &alpha, prior));
            let up = density::credit_upstream(&pred, &t.rgb, &edges, weights.as_deref(), geom.as_deref());
            gpu.write_f32(&scr.dimg, &up);
            gpu.submit(&[&scene.grads.d_colors], &[]);
            scr.renderer.render_bwd_ray(gpu, &gs, &cam, &o, &scr.dimg, None, &mut scr.bscr, &scene.grads, false).unwrap_or_else(|e| panic!("{e}"));
            let credit = gpu.read(&scene.grads.d_colors, 3 * n);
            // a gaussian a view composited half a pixel's worth of has been
            // seen by it
            seen.push(credit.chunks_exact(3).map(|c| c[0] > 0.5).collect());
            ev.add_view(&credit);
        }
        ev.absgrad = absgrad.to_vec();
        ev.site_share = if total > 0.0 { (unexplained / total) as f32 } else { 0.0 };
        self.prof.add("credit assignment", tm.elapsed());

        let snap = scene.snapshot(gpu);
        let mut next = snap.scene.clone();
        let before = next.len();
        let origin: Vec<Option<usize>> = match cfg.strategy {
            Densify::Heuristic => densify(&mut next, absgrad, cfg),
            Densify::Hybrid => {
                let target = density::target_population(self.init.len(), cfg.max_gaussians, round, rounds, next.len(), cfg.densify_frac);
                let px1 = crate::mip::smoothing_sigma(&next, &self.cams, 1.0, &|i, v| seen[v][i]);
                let size_px: Vec<f32> = (0..n)
                    .map(|i| {
                        let s = next.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max);
                        if px1[i] > 0.0 { s / px1[i] } else { 0.0 }
                    })
                    .collect();
                let policy = density::Policy {
                    refine_frac: if cfg.max_gaussians > 0 { 1.0 } else { cfg.densify_frac },
                    dead_opacity: cfg.prune_opacity,
                    ..Default::default()
                };
                let seed = 0x6879_6272_6964_u64 ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                let r = density::round(&mut next, &ev, &size_px, starved, target, &policy, seed);
                if cfg.log_every > 0 {
                    println!(
                        "fit: density round {round}: split {}, cloned {}, grown {}, spawned {} of {} sites ({:.0}% of the residual), \
                         suppressed {}, reclaimed {} (target {target})",
                        r.split, r.cloned, r.grown, r.spawned, ev.sites.len(), 100.0 * ev.site_share, r.suppressed, r.reclaimed
                    );
                }
                r.origin
            }
            Densify::Mcmc => {
                let (moved, added, origin) = crate::mcmc::step(&mut next, cfg, round, rounds);
                if cfg.log_every > 0 {
                    println!("fit: density round {round}: relocated {moved}, added {added}");
                }
                origin
            }
        };
        if cfg.log_every > 0 && next.len() != before {
            println!("fit: density control {before} -> {} gaussians", next.len());
        }
        let mut out = DeviceScene::remap(gpu, &snap, &next, &origin, cfg.sh_degree);
        if out.ksh > 0 {
            // each gaussian's view dependence, as far as its own views support
            // it; a new sample has none until the next round has seen it
            let eyes: Vec<[f32; 3]> = level.targets.iter().map(|t| t.cam.eye()).collect();
            let old = &snap.scene;
            let limit: Vec<u32> = backend_cpu::par::map(old.len(), |i| {
                let mine: Vec<[f32; 3]> = (0..eyes.len()).filter(|&v| seen[v][i]).map(|v| eyes[v]).collect();
                crate::sh::supported_coefficients([old.means[i * 3], old.means[i * 3 + 1], old.means[i * 3 + 2]], &mine, cfg.sh_degree)
            });
            out.set_sh_limit(gpu, &origin.iter().map(|o| o.map_or(0, |i| limit[i])).collect::<Vec<u32>>());
        }
        let fresh = largest_axes(&next);
        self.start = origin.iter().zip(&fresh).map(|(o, f)| o.map_or(*f, |i| self.start[i])).collect();
        // what saw each gaussian carries over along the same map; a new
        // sample has not been seen by anything yet, so its centre decides
        let seen_next: Vec<Vec<bool>> = seen.iter().map(|s| origin.iter().map(|o| o.is_none_or(|i| s[i])).collect()).collect();
        self.refresh_bounds(&out, Some(&seen_next));
        out
    }
}

/// A view's residual pixels the scene cannot fix by refining what is there,
/// as spawn sites, with the residual they carry and the view's total
/// residual.
///
/// A pixel qualifies when its residual is in the view's top tenth and the
/// scene there is empty (alpha below one half) or, where the view has a range
/// prior, at the wrong depth (off by more than 5% in range). Its site is
/// where the prior - or, without one, the scene's own rendered range - puts
/// the surface along the pixel's ray; a pixel with neither says nothing
/// about where its surface is and is skipped.
fn residual_sites(t: &TargetView, pred: &[f32], rgba: &[f32], aux: &[f32], weights: Option<&[f32]>, seed: u64) -> (Vec<density::Site>, f64, f64) {
    let (w, h) = (t.cam.width as usize, t.cam.height as usize);
    let r: Vec<f32> = (0..w * h)
        .map(|p| weights.map_or(1.0, |m| m[p]) * (0..3).map(|c| (pred[p * 3 + c] - t.rgb[p * 3 + c]).abs()).sum::<f32>() / 3.0)
        .collect();
    let total: f64 = r.iter().map(|v| *v as f64).sum();
    let mut sorted: Vec<f32> = r.iter().copied().filter(|v| *v > 0.0).collect();
    if sorted.is_empty() {
        return (Vec::new(), 0.0, total);
    }
    let k = (sorted.len() * 9 / 10).min(sorted.len() - 1);
    let thr = *sorted.select_nth_unstable_by(k, f32::total_cmp).1;
    let k_int = t.cam.intrinsics();
    let rot = |v: [f64; 3]| -> [f32; 3] {
        let m = &t.cam.c2w;
        std::array::from_fn(|i| (m[i * 4] as f64 * v[0] + m[i * 4 + 1] as f64 * v[1] + m[i * 4 + 2] as f64 * v[2]) as f32)
    };
    let eye = t.cam.eye();
    // at most this many sites per view: a round spawns a few percent of the
    // population, and the sites are drawn from by weight anyway
    let keep = 1.0f32.min(20_000.0 / (w * h) as f32 * 10.0);
    let mut sites = Vec::new();
    let mut unexplained = 0.0f64;
    for p in 0..w * h {
        if r[p] < thr || r[p] <= 0.0 {
            continue;
        }
        let alpha = rgba[p * 4 + 3];
        let rendered = (alpha > 0.5).then_some(aux[p * 5]).filter(|v| *v > 0.0);
        let prior = t.depth.as_ref().map(|d| d[p]).filter(|v| *v > 0.0);
        let wrong_depth = matches!((prior, rendered), (Some(a), Some(b)) if (a / b).ln().abs() > 0.05);
        if alpha >= 0.5 && !wrong_depth {
            continue;
        }
        let Some(range) = prior.or(rendered) else { continue };
        unexplained += r[p] as f64;
        if jitter(p, seed) >= keep {
            continue;
        }
        let Some(d) = k_int.unproject([(p % w) as f64 + 0.5, (p / w) as f64 + 0.5]) else { continue };
        let dw = rot(d);
        let normal = t.normals.as_ref().map(|n| [n[p * 3] as f64, n[p * 3 + 1] as f64, n[p * 3 + 2] as f64]).filter(|n| n.iter().any(|v| *v != 0.0)).map(rot);
        sites.push(density::Site {
            pos: std::array::from_fn(|i| eye[i] + dw[i] * range),
            radius: range / t.cam.fx.max(t.cam.fy),
            normal,
            rgb: [t.rgb[p * 3], t.rgb[p * 3 + 1], t.rgb[p * 3 + 2]],
            weight: r[p],
        });
    }
    (sites, unexplained, total)
}

/// The views iteration `it` optimizes: `per` of them, walking a fresh
/// deterministic permutation of all `n` each epoch; all of them when `per`
/// covers `n`.
fn batch_views(n: usize, per: usize, it: usize) -> Vec<usize> {
    if per >= n {
        return (0..n).collect();
    }
    (0..per)
        .map(|j| {
            let pos = it * per + j;
            let (epoch, at) = (pos / n, pos % n);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| jitter(a, epoch as u64 ^ 0xba7c).total_cmp(&jitter(b, epoch as u64 ^ 0xba7c)));
            order[at]
        })
        .collect()
}

/// How many of the `ksh` highest SH coefficients are still held out at
/// iteration `it` under [`FitCfg::sh_ramp`]: bands switch on one degree at a
/// time, as 3DGS does, so view-dependent colour cannot explain away error
/// geometry has not had the chance to.
fn sh_skip(cfg: &FitCfg, ksh: usize, it: usize) -> u32 {
    if cfg.sh_ramp <= 0.0 || ksh == 0 {
        return 0;
    }
    let degree = cfg.sh_degree.min(3) as f32;
    let progress = it as f32 / (cfg.sh_ramp * cfg.iters.max(1) as f32);
    let on = ((progress * (degree + 1.0)).floor() as usize).min(degree as usize);
    (ksh - ((on + 1) * (on + 1) - 1)) as u32
}

/// Every gaussian's largest axis.
fn largest_axes(s: &Splats) -> Vec<f32> {
    (0..s.len()).map(|i| s.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max)).collect()
}

/// Deterministic per-gaussian jitter in [0,1).
pub(crate) fn jitter(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
}

/// Density control as a step, for tests that need to look at what it did.
pub fn densify_for_test(scene: &mut Splats, grad: &[f32], cfg: &FitCfg) -> Vec<Option<usize>> {
    densify(scene, grad, cfg)
}

/// The 3DGS heuristic: grow the scene where the loss is still pulling
/// hardest, and drop what has gone transparent. Returns each new gaussian's
/// origin (see [`crate::density::Round::origin`]).
///
/// `grad` is the accumulated AbsGS magnitude of each gaussian's positional
/// gradient, thresholded at a FRACTION of the population rather than 3DGS's
/// absolute 2e-4: this loss is normalized per pixel and per view, so no
/// absolute threshold transfers between scenes, and a fraction bounds growth
/// by construction. Large gaussians SPLIT (two children at 1/1.6 the scale,
/// offset along the parent's dominant axis), small ones CLONE.
fn densify(scene: &mut Splats, grad: &[f32], cfg: &FitCfg) -> Vec<Option<usize>> {
    let n = scene.len();
    if n == 0 || grad.len() != n {
        return (0..n).map(Some).collect();
    }
    let cap = if cfg.max_gaussians > 0 { cfg.max_gaussians } else { usize::MAX };
    let alive = (0..n).filter(|&i| scene.opacities[i] >= cfg.prune_opacity).count();
    let mut extras = cap.saturating_sub(alive);
    let want = ((n as f32 * cfg.densify_frac) as usize).min(extras);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| grad[b].total_cmp(&grad[a]));
    let chosen: std::collections::HashSet<usize> = order.into_iter().take(want).collect();
    let mut sizes: Vec<f32> = (0..n).map(|i| scene.scales[i * 3..i * 3 + 3].iter().fold(0.0f32, |m, &v| m.max(v))).collect();
    sizes.sort_by(f32::total_cmp);
    let big = sizes[(n as f32 * 0.8) as usize % n];
    let mut out = Splats::default();
    let shk = crate::mcmc::sh_stride(scene);
    out.sh_rest = scene.sh_rest.as_ref().map(|(d, _)| (*d, Vec::new()));
    let mut origin = Vec::with_capacity(n + want);
    let mut push = |o: &mut Splats, i: usize, dm: [f32; 3], shrink: f32| {
        for (k, d) in dm.iter().enumerate() {
            o.means.push(scene.means[i * 3 + k] + d);
        }
        o.quats.extend_from_slice(&scene.quats[i * 4..i * 4 + 4]);
        for k in 0..3 {
            o.scales.push((scene.scales[i * 3 + k] * shrink).max(cfg.min_scale));
        }
        o.opacities.push(scene.opacities[i]);
        o.colors.extend_from_slice(&scene.colors[i * 3..i * 3 + 3]);
        if let (Some((_, src)), Some((_, dst))) = (&scene.sh_rest, &mut o.sh_rest) {
            dst.extend_from_slice(&src[i * shk..i * shk + shk]);
        }
        origin.push(Some(i));
    };
    for i in 0..n {
        if scene.opacities[i] < cfg.prune_opacity {
            continue;
        }
        let s = &scene.scales[i * 3..i * 3 + 3];
        let axis = (0..3).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
        let room = extras > 0;
        if room && chosen.contains(&i) && s[axis] >= big {
            let q = &scene.quats[i * 4..i * 4 + 4];
            let nq = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
            let col = crate::geometry::axis([q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq], axis);
            let d = s[axis] * 0.5;
            extras -= 1;
            push(&mut out, i, [col[0] * d, col[1] * d, col[2] * d], 1.0 / 1.6);
            push(&mut out, i, [-col[0] * d, -col[1] * d, -col[2] * d], 1.0 / 1.6);
        } else if room && chosen.contains(&i) {
            extras -= 1;
            push(&mut out, i, [0.0; 3], 1.0);
            push(&mut out, i, [0.0; 3], 1.0);
        } else {
            push(&mut out, i, [0.0; 3], 1.0);
        }
    }
    if out.is_empty() {
        return (0..n).map(Some).collect();
    }
    *scene = out;
    origin
}

/// Row-major 3x3 rotation block of a row-major 4x4.
fn rot3(m: &[f32; 16]) -> [f64; 9] {
    std::array::from_fn(|i| m[(i / 3) * 4 + i % 3] as f64)
}

/// `c2w * [exp(omega^) | tau]`: move the camera in its OWN frame.
fn compose_local(c2w: &[f32; 16], omega: &[f64; 3], tau: &[f64; 3]) -> [f32; 16] {
    let th = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let (a, b) = if th < 1e-8 { (1.0, 0.5) } else { (th.sin() / th, (1.0 - th.cos()) / (th * th)) };
    let k = [[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]];
    let mut d = [[0.0f64; 3]; 3];
    for (i, row) in d.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let kk: f64 = (0..3).map(|t| k[i][t] * k[t][j]).sum();
            *v = if i == j { 1.0 } else { 0.0 } + a * k[i][j] + b * kk;
        }
    }
    let r = rot3(c2w);
    let mut out = *c2w;
    for i in 0..3 {
        for j in 0..3 {
            out[i * 4 + j] = (0..3).map(|t| r[i * 3 + t] * d[t][j]).sum::<f64>() as f32;
        }
        out[i * 4 + 3] = (c2w[i * 4 + 3] as f64 + (0..3).map(|t| r[i * 3 + t] * tau[t]).sum::<f64>()) as f32;
    }
    out
}

/// The factor that puts a scene roughly one unit across, from the spread of
/// the cameras and the scene together - a scene with one stray gaussian at
/// infinity should not be judged by it.
fn scene_unit(s: &Splats, targets: &[TargetView]) -> f32 {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut note = |p: [f32; 3]| {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    };
    for t in targets {
        note(t.cam.eye());
    }
    let step = (s.len() / 4096).max(1);
    for i in (0..s.len()).step_by(step) {
        note([s.means[i * 3], s.means[i * 3 + 1], s.means[i * 3 + 2]]);
    }
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
    if ext.is_finite() && ext > 1e-20 {
        1.0 / ext
    } else {
        1.0
    }
}

fn rescale_scene(s: &Splats, k: f32) -> Splats {
    let mut out = s.clone();
    for v in out.means.iter_mut() {
        *v *= k;
    }
    for v in out.scales.iter_mut() {
        *v *= k;
    }
    out
}

/// A camera in a world scaled by `k`: its position moves, and so does the
/// distance a rolling shutter's readout travels.
fn rescale_cam(c: &Camera, k: f32) -> Camera {
    let mut out = *c;
    for i in 0..3 {
        out.c2w[i * 4 + 3] *= k;
        out.shutter[3 + i] *= k;
    }
    out
}
