// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3DGS scene optimization ("fit"): AdamW on gaussian parameters against
//! posed target images, driven by the atomic-free rasterizer backward. The
//! demonstrable use of `render_bwd` — and the training path for scenes.
//!
//! Parameterization: packed geometry `[N*10] = {means, scales(linear),
//! quats(raw)}` (matches `d_gauss`, so one AdamW dispatch covers it), plus
//! separate opacity/color buffers. Linear scales/opacities are clamped
//! host-side after each step (projected gradient) — simpler than log/logit
//! reparameterization and adequate for fitting; antialiased mode is not
//! supported by the backward (compensation chain unmodeled).

use gpu_core::{f, DeviceBuffer, Gpu};

use crate::density::{self, Evidence};
use crate::geometry;
use crate::isp::{Encoding, Isp, IspCfg};
use crate::loss::{DeviceLoss, PixelLoss};
use crate::renderer::{BwdScratch, GpuSplats, Renderer, SplatGrads};
use crate::types::{Camera, Mode, RenderOpts, Splats};
use crate::Kernels;

/// Which density-control strategy `fit` runs when `densify_every > 0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Densify {
    /// Upstream 3DGS: threshold a positional-gradient statistic, then SPLIT
    /// the large gaussians above it and CLONE the small ones, and drop
    /// whatever has gone transparent. The default, so that turning density
    /// control on changes nothing about what it then does.
    #[default]
    Heuristic,
    /// 3DGS-MCMC (Kheradmand et al., NeurIPS 2024, arXiv:2404.09591):
    /// no thresholds and no deletion. Transparent gaussians are RELOCATED onto
    /// opaque ones with the opacity and scale correction that keeps the render
    /// unchanged at the moment of the move, the budget is spent up to
    /// `max_gaussians`, and positions carry noise proportional to the learning
    /// rate so the fit samples rather than descends. See [`crate::mcmc`].
    Mcmc,
    /// Credit-assigned, budgeted density control ([`crate::density`]): every
    /// gaussian is scored by the ranks of the image residual it is
    /// responsible for, the part of it on edges, and its AbsGS gradient; the
    /// top of that ranking is refined within a population schedule that
    /// reaches `max_gaussians` two thirds of the way through density control;
    /// gaussians that contribute nothing twice in a row pay for it.
    Hybrid,
}

pub struct FitCfg {
    pub iters: usize,
    pub lr: f32,
    /// Clamp every step: scales into [min_scale, 0.3], opacity into [ε, 1-ε].
    pub min_scale: f32,
    /// Largest ratio a gaussian's longest axis may have to its MIDDLE one.
    /// 0 = unconstrained.
    ///
    /// Sort a gaussian's axes a >= b >= c and the two ways it can be
    /// anisotropic are not equally wrong. A DISC - a close to b, both far
    /// above c - is a surface element, which is what a reconstruction of a
    /// surface is supposed to contain. A NEEDLE - a far above b and c - is
    /// always wrong: lined up with a training view's ray it lowers that view's
    /// loss while being invisible in it, and is a streak from every other
    /// direction.
    ///
    /// So the bound is on a/b, not a/c. Bounding a/c is wrong twice over: it
    /// permits needles right up to the limit while forbidding the thin discs a
    /// surface actually wants. Measured on a real capture, the model emits
    /// gaussians with a/b at the 90th percentile of 1.94, and fitting under an
    /// a/c bound took that to 9.29 with 27.7% of the scene past 3:1.
    pub max_needle: f32,
    /// Bound on b/c, the flatness of a gaussian - see [`clamp_axes`]. A
    /// surface element is legitimately a disc, so this is looser than
    /// `max_needle` rather than absent, which is what it used to be.
    pub max_flat: f32,
    pub log_every: usize,
    /// Run density control every N iterations, 0 = never (a fixed set of
    /// gaussians, which is all this optimizer could ever do before).
    ///
    /// OFF by default, which is the conservative choice rather than the
    /// obvious one. Density control is the fix for a scene too SPARSE to hold
    /// its detail, and a feed-forward reconstruction is the opposite problem:
    /// it starts at one gaussian per source pixel per view, so a 48-frame
    /// video arrives at ~9.7M gaussians before anything is added. Growing that
    /// pushes the backward past the per-binding gradient-record ceiling, which
    /// fails the run outright. Turn it on for sparse scenes; reach for
    /// `splat::prune` on dense ones.
    pub densify_every: usize,
    /// Skip density control until this iteration, so the gradients it reads
    /// describe the scene rather than the first few steps of chaos.
    pub densify_after: usize,
    /// Stop density control at this iteration, 0 = run it to the end.
    ///
    /// A fit that is still rearranging its scene on the last iteration never
    /// gets to converge on the arrangement it chose: whatever was just added
    /// or moved is left wherever it landed, with a fresh Adam state and no
    /// steps to spend. Both the reference pipelines stop well before the end
    /// for this reason (upstream 3DGS at half of training, 3DGS-MCMC at 83%
    /// of it).
    pub densify_until: usize,
    /// Fraction of gaussians, by positional-gradient magnitude, considered
    /// under-reconstructed at each density-control step.
    pub densify_frac: f32,
    /// The opacity below which a gaussian is not carrying anything, and what
    /// the density control then does about it: [`Densify::Heuristic`] drops
    /// it, [`Densify::Mcmc`] relocates it. One threshold, because the
    /// judgement it encodes is the same one.
    pub prune_opacity: f32,
    /// Refuse to grow past this many gaussians, 0 = the device decides. A
    /// runaway subdivision is a much worse failure than a soft scene.
    pub max_gaussians: usize,
    /// The anti-alias dilation the fit optimizes UNDER. It is part of the
    /// forward model being inverted, so the optimizer folds compensation for
    /// it into the gaussians: a scene fitted at one value and rendered at
    /// another comes out wrong in the direction of the difference. Render with
    /// the same value.
    pub eps2d: f32,
    /// Put the energy back when the low-pass spreads it (Mip-Splatting's 2D
    /// Mip filter). Clear it only to fit a scene for a viewer that renders
    /// Inria's uncompensated dilation.
    pub antialiased: bool,
    /// Floor every gaussian's axes at the finest detail its own cameras
    /// sampled, scaled by this (Mip-Splatting's 3D smoothing filter as a
    /// constraint rather than a post-hoc convolution). 0 disables it.
    ///
    /// Without a floor the optimizer is free to shrink a gaussian below the
    /// pixel that supervises it, where it can lower the loss on the views it
    /// hides in and alias in every other. The fixed `min_scale` cannot express
    /// that limit, because the limit is a property of where the cameras were.
    pub mip_scale: f32,
    /// Largest a gaussian may be, in PIXELS as its own cameras see it.
    ///
    /// The bound used to be 0.3 in world units, a constant with no relation to
    /// the scene: on a capture whose camera orbit has radius 0.5 that is 141
    /// pixels, so the optimizer was free to cover a gap with a splat spanning
    /// a fifth of the frame. Those are the streaks that make a fitted scene
    /// look like hair from any angle it was not fitted at. A bound in pixels
    /// is a bound on the detail the cameras could actually have resolved.
    ///
    /// It was then set to 16 pixels, which never bound: measured on a real
    /// capture that ceiling has a median of 0.0358 world units against a
    /// median gaussian of 0.00134, twenty seven times larger, so a fit simply
    /// grew everything until it reached it. The reconstruction this fits emits
    /// SUB-pixel gaussians (median 0.6 px), and a couple of pixels is already
    /// generous headroom over that.
    pub max_scale_pixels: f32,
    /// How many times its STARTING size a gaussian may grow. 0 = unbounded.
    ///
    /// A ceiling in pixels alone cannot be right for every scene: a sparse
    /// scene legitimately has gaussians many pixels across, while the
    /// pixel-aligned reconstruction this usually fits emits SUB-pixel ones
    /// (median 0.6 px). Tightening the pixel ceiling far enough to constrain
    /// the second stops the first from representing itself at all.
    ///
    /// What was actually measured going wrong is GROWTH: over 400 iterations
    /// the fit inflated the longest axis of a real reconstruction 7.5x, which
    /// is what turns a scene into fog when viewed from anywhere it was not
    /// fitted. A bound relative to where each gaussian started says "the fit
    /// may refine what the reconstruction proposed, not replace it with
    /// something far larger", and that is scene-adaptive by construction.
    pub max_growth: f32,
    /// Fraction of the LARGE gaussians to split each round regardless of what
    /// the gradient says, and to reseed at a probed depth. 0 disables it.
    ///
    /// Two gates keep a blurry region blurry no matter how many views see it.
    /// A splat's image-plane gradient is orthogonal to the viewing ray, so
    /// nothing ever pushes a gaussian along depth - a wrongly-placed one can
    /// only slide sideways. And alpha blending attenuates the gradient
    /// reaching anything behind something else, so an occluded region never
    /// reaches the split threshold. Neither is a thresholding problem, so no
    /// choice of threshold escapes them; the way out is to try anyway,
    /// occasionally, without asking the gradient for permission.
    pub explore_frac: f32,
    /// Spherical-harmonic degree for view-dependent colour: 0, 1, 2 or 3.
    ///
    /// At 0 a gaussian looks the same from everywhere, so the only way a fit
    /// can explain a surface that is brighter from one side is to move
    /// geometry - which is why glossy objects come back as smears of
    /// duplicated surfaces at slightly different depths.
    pub sh_degree: u32,
    /// Weight of the DEPTH term relative to RGB, 0 = off (and then a fit is
    /// byte-for-byte the RGB-only fit it always was).
    ///
    /// A splat's image-plane gradient is orthogonal to its own viewing ray, so
    /// a scene placed at the wrong DISTANCE and scaled to subtend the same
    /// angle renders the identical image and has an exactly zero RGB gradient.
    /// Measured against an analytic ground truth, a reconstruction put a thin
    /// object 5.7% too far away, entirely systematically: no amount of RGB
    /// optimization touches that, because the loss cannot see it.
    ///
    /// The term is an ANCHOR, not a source of truth. The depth a caller has is
    /// the same model's, and that model is what was 5.7% wrong; supervising
    /// against a single view's prediction re-imposes exactly the error being
    /// removed. Feed it a MULTI-VIEW FUSED depth with per-pixel confidence
    /// ([`TargetView::with_depth`]) - cross-view disagreement measured 1.1%
    /// per-view against 0.41% fused - and what it buys is geometry that stops
    /// drifting: an RGB-only fit measured over 400 iterations grew the longest
    /// gaussian axis nearly ninefold and took the flatness ratio from 4.14 to
    /// 26.34, which is blades that look right from the training cameras and
    /// render as fur from anywhere else.
    pub depth_weight: f32,
    /// How density control decides where gaussians go.
    pub strategy: Densify,
    /// [`Densify::Mcmc`] only, and only when `max_gaussians` is 0: how much
    /// bigger the scene may get at each density-control round, as a fraction
    /// of its current size. The paper's 5%.
    ///
    /// With a budget this is not consulted at all - the budget IS the
    /// schedule, see [`crate::mcmc::step`]. It is the fallback for a fit that
    /// was given no budget to aim at, where 5% a round is the only answer
    /// available and a short fit will not get far on it.
    pub mcmc_grow_frac: f32,
    /// [`Densify::Mcmc`] only: how fast an unsupported gaussian fades, as
    /// AdamW's decoupled decay on opacity alone (`o -= lr * decay * o`).
    ///
    /// The paper regularizes with `λ_o·Σ|o_i|` added to the loss, and the
    /// point of that term is to MANUFACTURE dead gaussians: relocation is the
    /// move that distinguishes MCMC from cloning, and it has nothing to move
    /// until something has gone transparent. Measured on a scene with no dead
    /// gaussians in it, a 200-iteration fit relocated ONE.
    ///
    /// It is a decay rather than a term in the loss for the same reason
    /// `densify_frac` is a fraction rather than an absolute gradient
    /// threshold: this loss is normalized per pixel and per view, so an
    /// absolute `λ_o` means something different at every image size and view
    /// count, while a fraction of the opacity per step does not.
    ///
    /// OFF by default, because what it buys is not free and the bill arrives
    /// first. Every gaussian fades and only the loss puts it back, so the fit
    /// pays for the recycling immediately and collects when the recycled
    /// samples have had time to settle somewhere useful. Measured at 0.0 /
    /// 0.05 / 0.15 / 0.5 on one scene: 200 iterations 0.008978, 0.009675,
    /// 0.010887, 0.011409 - monotonically worse; 600 iterations 0.003055,
    /// 0.002984 - the sign has flipped. Turn it on for long fits.
    pub mcmc_opacity_decay: f32,
    /// [`Densify::Mcmc`] only: the paper's λ_noise, the coefficient of the
    /// position noise, in units of the gaussian's OWN size per learning-rate
    /// step. The displacement is `mcmc_noise * lr * N(0, Σ_i)`, weighted per
    /// gaussian so only the near-transparent ones move - see
    /// [`crate::mcmc::add_noise`]. 0 turns the chain back into plain descent.
    ///
    /// 1.0 is the natural unit of this parameterization rather than a fitted
    /// constant: one learning-rate step is worth one standard deviation of
    /// the gaussian's own shape. It is also what the measurements prefer,
    /// though not by much - on one scene 0.007965 without the noise against
    /// 0.007128 with it, and 0.3 and 3.0 on either side of it are worse.
    pub mcmc_noise: f32,
    /// Step size for refining the CAMERAS, in radians and world units per
    /// iteration. 0 freezes them, which is what `fit` does.
    ///
    /// A pipeline that predicts its own poses hands the optimizer cameras that
    /// are wrong, and gaussian positions are depth unprojected through a
    /// camera - so a pose error is a position error for every pixel of that
    /// frame, and no amount of moving gaussians reconciles two frames that
    /// disagree about where they were taken from.
    pub pose_lr: f32,
    /// How far a gaussian may travel over the WHOLE fit, in multiples of its
    /// own radius. <= 0 (the default) spends `lr` on positions directly.
    ///
    /// Off by default because only the caller knows whether the geometry it
    /// passed in is worth preserving. A fit that starts from a sparse point
    /// cloud has to let gaussians travel many radii to find a surface at all,
    /// and budgeting it would only stop it converging; a fit that starts from
    /// a feed-forward model's metric depth is in the opposite situation and
    /// should set this.
    ///
    /// `p_geo` interleaves a position in world units, a LINEAR scale, and a
    /// unit quaternion, and Adam's step is ~lr whatever the gradient is. So a
    /// single `lr` is simultaneously a rounding error on the scene diagonal
    /// and a large fraction of a small gaussian's own size, and which of those
    /// it is depends on nothing but how finely the scene happens to be
    /// tessellated. Budgeting the travel against the gaussian's own radius is
    /// what makes one setting mean the same thing to a sparse scene of large
    /// blobs and a dense one of small ones.
    ///
    /// A scene from a feed-forward model is already metric - every gaussian
    /// sits on the surface it was unprojected from - so the honest budget is
    /// about one radius: enough to settle onto the surface, not enough to
    /// leave it. Measured on a real capture, an unbudgeted fit moved 94% of
    /// gaussians more than three radii, which is the scene coming apart.
    pub position_budget: f32,
    /// Fractional change in a gaussian's size over the whole fit, as a
    /// multiple of its own radius. <= 0 spends `lr` on scales directly.
    ///
    /// Unbudgeted, the same mismatch polarizes the scene rather than blurring
    /// it: a third of a real capture collapsed past 2:1 while a quarter sat
    /// pinned against `max_growth`. Holes and stray blobs are the two halves
    /// of one defect.
    pub scale_budget: f32,
    /// Change in quaternion components over the whole fit. <= 0 spends `lr`
    /// on rotations directly.
    ///
    /// A quaternion is already O(1), so this is the one geometry group a raw
    /// `lr` UNDER-trains rather than over-trains.
    pub rotation_budget: f32,
    /// The photometric objective. [`PixelLoss::Mse`] (the default) is what
    /// `fit` has always minimised; [`PixelLoss::gaussian_splatting`] is the
    /// L1 + D-SSIM objective 3DGS is defined with, which charges a blurred
    /// edge far more than the same energy spread as an offset. What the fit
    /// REPORTS is this objective, so a run's numbers are only comparable with
    /// another run's under the same loss.
    pub loss: PixelLoss,
    /// Fit a photometric camera model (exposure, white balance, vignetting,
    /// response) alongside the scene - see [`crate::isp`]. `None` (the
    /// default) compares renders with photographs directly, as `fit` always
    /// did.
    pub isp: Option<IspCfg>,
    /// Weight of the 2DGS depth-distortion term, 0 = off: penalizes
    /// compositing weight spread ALONG each ray, which is what a stack of
    /// semi-transparent layers standing in for one surface looks like. See
    /// [`crate::geometry`].
    pub distortion_weight: f32,
    /// Weight of normal consistency against the rendered depth map's own
    /// normals, 0 = off: turns each gaussian's flat axis onto the surface it
    /// sits on, so a disc seen edge-on from a new view is not a streak.
    pub normal_consistency_weight: f32,
    /// Weight of supervision by [`TargetView::normals`], 0 = off.
    pub normal_prior_weight: f32,
    /// Fraction of the fit after which the geometry terms above switch on.
    /// Surface regularizers applied before the scene has any shape pull
    /// gaussians onto whatever surface happens to be rendered first.
    pub geometry_after: f32,
    /// Run the geometry terms on every Nth iteration only, with their weights
    /// multiplied by N so the gradient they contribute is the same on
    /// average. Each term is a full extra forward AND backward of every view
    /// it touches, so at 1 the two of them triple the cost of an iteration.
    pub geometry_every: usize,
    /// Fraction of the fit by which every SH band up to `sh_degree` is on,
    /// 0 = all of them from the start. Bands switch on one degree at a time
    /// at even steps before it, as 3DGS does: with every band free from the
    /// first iteration, view-dependent colour explains away error that
    /// geometry has not yet had the chance to.
    pub sh_ramp: f32,
    /// Fraction of the fit run at HALF resolution before switching to full,
    /// 0 = full resolution throughout.
    ///
    /// A scene started from a sparse cloud is, for its first stretch, a few
    /// thousand large gaussians that cannot use full-resolution detail - and
    /// the backward's cost is proportional to pixels times compositing depth,
    /// so those are the most expensive iterations of the whole fit. A quarter
    /// of the pixels costs about a quarter of the backward.
    pub coarse: f32,
    /// Views per iteration, 0 = all of them (full batch, what `fit` always
    /// did). Each epoch visits every view once in a fresh deterministic
    /// order.
    ///
    /// Full batch spends a render and a backward of EVERY view on each
    /// optimizer step. 3DGS takes one view per step, and for the same compute
    /// a minibatch fit takes many more, noisier steps - which is what an
    /// optimizer this far from its solution wants. The reported loss, and the
    /// step-size backoff that watches it, are then an exponential average
    /// over about one epoch, since one batch's loss says little.
    pub batch: usize,
}

impl Default for FitCfg {
    fn default() -> Self {
        FitCfg {
            iters: 200,
            lr: 5e-3,
            min_scale: 1e-4,
            max_needle: 2.0,
            max_flat: 4.0,
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
            max_scale_pixels: 16.0,
            max_growth: 2.0,
            explore_frac: 0.05,
            sh_degree: 0,
            depth_weight: 0.0,
            strategy: Densify::Heuristic,
            mcmc_grow_frac: 0.05,
            mcmc_opacity_decay: 0.0,
            mcmc_noise: 1.0,
            pose_lr: 0.0,
            position_budget: 0.0,
            scale_budget: 0.0,
            rotation_budget: 0.0,
            loss: PixelLoss::Mse,
            isp: None,
            distortion_weight: 0.0,
            normal_consistency_weight: 0.0,
            normal_prior_weight: 0.0,
            geometry_after: 0.0,
            geometry_every: 1,
            sh_ramp: 0.0,
            coarse: 0.0,
            batch: 0,
        }
    }
}

impl FitCfg {
    /// Everything a scene started from a SPARSE point cloud (structure from
    /// motion) needs to become a finished reconstruction in `iters`
    /// iterations and at most `budget` gaussians: the L1 + D-SSIM objective,
    /// the photometric camera model, credit-assigned density control from 5%
    /// to 60% of the fit, the Mip filter, full view-dependent colour, and
    /// the surface regularizers once the scene has a shape (40%) on every
    /// fourth step, and two views per step rather than all of them.
    ///
    /// Geometry moves at rates relative to each gaussian's own size - over the
    /// whole fit, 30 of its radii of travel, 10 of resize, 2 of quaternion -
    /// and no gaussian may exceed 32 pixels of its finest camera. With the
    /// parameters linear and Adam stepping about `lr` whatever the gradient,
    /// a raw `lr` of 5e-3 in a scene one unit across moves positions ~30x
    /// faster than 3DGS does and lets a 0.002-unit gaussian grow several-fold
    /// in ONE step: measured on a 16-photo capture, 60 iterations turned the
    /// scene into uniform grey fog. The growth bound relative to a gaussian's
    /// START stays off: a sparse cloud has to grow a long way to cover what
    /// its points only sample.
    pub fn from_sparse_points(iters: usize, budget: usize) -> FitCfg {
        let every = (iters / 20).max(1);
        FitCfg {
            iters,
            loss: PixelLoss::gaussian_splatting(),
            isp: Some(IspCfg::default()),
            strategy: Densify::Hybrid,
            densify_every: every,
            densify_after: every,
            densify_until: iters * 6 / 10,
            max_gaussians: budget,
            antialiased: true,
            sh_degree: 3,
            max_growth: 0.0,
            max_scale_pixels: 32.0,
            position_budget: 30.0,
            scale_budget: 10.0,
            rotation_budget: 2.0,
            max_flat: 10.0,
            distortion_weight: 0.1,
            normal_consistency_weight: 0.05,
            geometry_after: 0.4,
            geometry_every: 4,
            sh_ramp: 0.3,
            coarse: 0.3,
            batch: 2,
            log_every: 50,
            ..Default::default()
        }
    }
}

/// One posed target view: camera + RGB f32 `[W*H*3]` in [0,1], plus the
/// optional per-pixel supervision RGB alone cannot carry.
#[derive(Clone)]
pub struct TargetView {
    pub cam: Camera,
    pub rgb: Vec<f32>,
    /// Per-pixel depth prior `[W*H]`, in the SCENE's own world units, 0 = no
    /// data at that pixel. Compared against the render's expected depth; see
    /// [`FitCfg::depth_weight`] for what it is and is not good for.
    pub depth: Option<Vec<f32>>,
    /// How far to trust each depth pixel, `[W*H]`, `None` = fully.
    ///
    /// Anchoring hard where the prior is wrong is worse than not anchoring at
    /// all, and a depth prior is wrong in a way that varies across the frame -
    /// a multi-view fusion knows at every pixel how many views agreed, and
    /// that number, normalized, is exactly what belongs here.
    pub depth_conf: Option<Vec<f32>>,
    /// Per-pixel loss weight `[W*H]` in [0,1], `None` = supervise everything.
    ///
    /// The loss otherwise covers every pixel including background the
    /// reconstruction has no geometry for, and against a black render
    /// background that is a large permanent error - which the optimizer
    /// answers by dragging gaussians outward to cover it. A mask lets the
    /// caller say those pixels are not evidence. It divides out of the
    /// normalizer too, so the reported MSE stays the MSE of the pixels that
    /// were actually supervised.
    pub mask: Option<Vec<f32>>,
    /// Which physical camera took this view, for the parts of the camera
    /// model that belong to the lens and sensor rather than the shot
    /// (vignetting, response curve). 0 unless the capture mixes cameras.
    pub sensor: usize,
    /// Exposure of this view in log2 stops relative to the others, when it is
    /// KNOWN (EXIF shutter/ISO/aperture, a bracket's EV offset). The camera
    /// model then fits a residual around it instead of discovering it.
    pub exposure: f32,
    /// How `rgb` is encoded; consulted by a scene-linear fit only.
    pub encoding: Encoding,
    /// Per-pixel surface normal prior `[W*H*3]` in THIS camera's frame
    /// (+X right, +Y down, +Z forward), facing the camera; a zero vector =
    /// no prior at that pixel. Used by [`FitCfg::normal_prior_weight`].
    pub normals: Option<Vec<f32>>,
}

impl TargetView {
    /// A view supervised by its colours alone - what `fit` has always done.
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

    /// Add a depth prior `[W*H]` (0 = no data) and, optionally, how much to
    /// trust each of its pixels.
    pub fn with_depth(mut self, depth: Vec<f32>, conf: Option<Vec<f32>>) -> TargetView {
        self.depth = Some(depth);
        self.depth_conf = conf;
        self
    }

    /// This view at half the resolution: every per-pixel quantity averaged
    /// over 2x2 blocks (normals renormalized), the camera's intrinsics halved.
    /// An odd last row or column is dropped, which leaves pixel coordinates -
    /// and so the principal point - exactly halved.
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
            // a depth average across an edge is a depth nowhere; take the
            // near one of the block, where it has data
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

    /// Restrict supervision to the pixels `mask` `[W*H]` weights.
    pub fn with_mask(mut self, mask: Vec<f32>) -> TargetView {
        self.mask = Some(mask);
        self
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
}

/// Everything a fit produces.
pub struct FitResult {
    pub scene: Splats,
    /// The cameras, refined when `pose_lr > 0`.
    pub cams: Vec<Camera>,
    /// The objective at the last iteration.
    pub loss: f32,
    /// The fitted camera model, when [`FitCfg::isp`] asked for one.
    pub isp: Option<Isp>,
}

/// Fit `init` against the targets; returns the optimized scene and the final
/// mean MSE across views.
///
/// `on_step(iter, mse)` is polled once per completed iteration, inside this
/// loop (not bolted on from outside) - returning `false` aborts early, at the
/// end of whichever iteration just ran. `crates/cli/src/splat_cli.rs::fit_cmd`
/// passes a closure that only prints and always returns `true`;
/// `splat::caps::fit` polls the invocation's cancel token from its own.
/// Fit the scene AND the cameras, returning the refined poses.
pub fn fit_bundle(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    targets: &[TargetView],
    cfg: &FitCfg,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> (Splats, Vec<Camera>, f32) {
    let r = fit_full(gpu, ks, init, targets, cfg, on_step);
    (r.scene, r.cams, r.loss)
}

pub fn fit(gpu: &Gpu, ks: Kernels, init: &Splats, targets: &[TargetView], cfg: &FitCfg, on_step: &mut dyn FnMut(usize, f32) -> bool) -> (Splats, f32) {
    let r = fit_full(gpu, ks, init, targets, cfg, on_step);
    (r.scene, r.loss)
}

/// Fit the scene, and whatever else `cfg` asks to be fitted with it - the
/// cameras' poses, the photometric camera model - returning all of it.
pub fn fit_full(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    targets: &[TargetView],
    cfg: &FitCfg,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> FitResult {
    let mut isp = cfg.isp.map(|c| {
        let views: Vec<(usize, f32, Encoding)> = targets.iter().map(|t| (t.sensor, t.exposure, t.encoding)).collect();
        Isp::new(c, &views)
    });
    let (scene, cams, loss) = fit_inner(gpu, ks, init, targets, cfg, &mut isp, on_step);
    if let (Some(i), true) = (&isp, cfg.log_every > 0) {
        print!("fit: camera model\n{}", i.summary());
    }
    FitResult { scene, cams, loss, isp }
}

fn fit_inner(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    targets: &[TargetView],
    cfg: &FitCfg,
    isp: &mut Option<Isp>,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> (Splats, Vec<Camera>, f32) {
    assert!(!targets.is_empty());
    // Fit in a frame where the scene is about one unit across.
    //
    // A reconstruction has no intrinsic units - photograph a model railway or
    // a mountain and the pipeline sees the same rays - but the optimizer is
    // full of quantities that do: a minimum scale in world units, and above
    // all a learning rate, because Adam's step is a fixed DISTANCE rather than
    // a fixed fraction. On a scene a hundred times too big the fit can barely
    // move geometry at all; a hundred times too small and every step
    // overshoots. Measured across that range the same fit varied by 8 dB and
    // went from blur at one end to speckle at the other.
    //
    // Normalising once at the door fixes the whole class at a stroke, and it
    // is exactly equivalent: intrinsics are in pixels, so scaling the world
    // and the camera positions together changes no rendered image.
    let k = scene_unit(init, targets);
    if (k - 1.0).abs() > 1e-6 {
        let scaled_init = rescale_scene(init, k);
        let scaled: Vec<TargetView> = targets
            .iter()
            .map(|t| {
                let mut v = t.clone();
                v.cam = rescale_cam(&t.cam, k);
                // A depth prior is in world units, so it rescales with the
                // world. Carrying it through unscaled would have the depth
                // term anchor the scene to a distance the normalization just
                // moved, which is a silent, total corruption of the fit.
                v.depth = t.depth.as_ref().map(|d| d.iter().map(|v| v * k).collect());
                v
            })
            .collect();
        let (s, c, l) = fit_inner(gpu, ks, &scaled_init, &scaled, cfg, isp, on_step);
        return (
            rescale_scene(&s, 1.0 / k),
            c.iter().map(|c| rescale_cam(c, 1.0 / k)).collect(),
            l,
        );
    }
    // With density control off there is exactly ONE stage, and this is the
    // function it always was - same buffers, same Adam state, start to finish.
    let stage_len = if cfg.densify_every == 0 { cfg.iters } else { cfg.densify_every };
    // The unit of every geometry budget: the scene as handed in, fixed for the
    // whole fit. How far a capture's gaussians have to travel is a property
    // of the capture - an object structure from motion left without points
    // is as far from its neighbours after density control subdivides them as
    // before - so re-measuring the median on each stage's population shrank
    // every step with every round and left such objects half grown.
    let radius = median_radius(init);
    let mut scene = init.clone();
    let mut cams: Vec<Camera> = targets.iter().map(|t| t.cam).collect();
    let mut loss = 0.0f32;
    let mut done = 0usize;
    let mut stop = false;
    // The loss of the scene as handed in, reported by the first iteration
    // (which measures before it updates anything). It is the floor the fit has
    // to beat to have been worth running.
    let mut first_loss = f32::NAN;
    // Recovery marks of the hybrid controller, carried across rounds.
    let mut suspect: Vec<bool> = Vec::new();
    let mut seen = |it: usize, mse: f32, k: &mut dyn FnMut(usize, f32) -> bool| {
        if first_loss.is_nan() {
            first_loss = mse;
        }
        k(it, mse)
    };
    // Coarse-to-fine: the first `coarse` of the fit sees every target and
    // camera at half resolution. A stage never straddles the switch.
    let coarse_end = (cfg.coarse.clamp(0.0, 1.0) * cfg.iters as f32) as usize;
    let half: Vec<TargetView> = if coarse_end > 0 { targets.iter().map(TargetView::half).collect() } else { Vec::new() };
    while done < cfg.iters && !stop {
        let coarse = done < coarse_end;
        let mut iters = stage_len.min(cfg.iters - done);
        if coarse {
            iters = iters.min(coarse_end - done);
        }
        let out = {
            let mut tap = |it: usize, mse: f32| seen(it, mse, on_step);
            if coarse {
                // the half-size cameras carry the SAME poses, so pose
                // refinement made at half resolution carries back
                let mut small: Vec<Camera> = cams.iter().zip(&half).map(|(c, t)| Camera { c2w: c.c2w, ..t.cam }).collect();
                let out = fit_stage(gpu, ks, &scene, radius, &half, &mut small, cfg, isp, iters, done, &mut tap);
                for (c, s) in cams.iter_mut().zip(&small) {
                    c.c2w = s.c2w;
                }
                out
            } else {
                fit_stage(gpu, ks, &scene, radius, targets, &mut cams, cfg, isp, iters, done, &mut tap)
            }
        };
        scene = out.scene;
        loss = out.loss;
        let grad = out.absgrad;
        done += iters;
        stop = out.aborted;
        let until = if cfg.densify_until > 0 { cfg.densify_until.min(cfg.iters) } else { cfg.iters };
        if cfg.densify_every > 0 && done >= cfg.densify_after && done < until && !stop {
            let before = scene.len();
            // Where the cameras are, so exploration can probe ALONG the
            // viewing ray - the one direction the gradient cannot supply.
            let eye = {
                let mut c = [0.0f64; 3];
                for t in targets {
                    let m = t.cam.c2w;
                    for k in 0..3 {
                        c[k] += m[k * 4 + 3] as f64 / targets.len() as f64;
                    }
                }
                [c[0] as f32, c[1] as f32, c[2] as f32]
            };
            match cfg.strategy {
                Densify::Heuristic => densify(&mut scene, &grad, cfg, eye),
                Densify::Hybrid => {
                    let mut ev = out.evidence.expect("a hybrid stage collects evidence");
                    ev.absgrad = grad;
                    let first = cfg.densify_after.max(cfg.densify_every);
                    let round = done.saturating_sub(first) / cfg.densify_every;
                    let rounds = (until.saturating_sub(first)).div_ceil(cfg.densify_every);
                    let target = density::target_population(init.len(), cfg.max_gaussians, round, rounds, scene.len(), cfg.densify_frac);
                    // A gaussian's size in pixels as the camera that sees it
                    // most finely does: whether a refinement is a split or a
                    // clone is a question about the image, not the world.
                    let px1 = crate::mip::smoothing_sigma(&scene, &cams, 1.0);
                    let size_px: Vec<f32> = (0..scene.len())
                        .map(|i| {
                            let s = scene.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max);
                            if px1[i] > 0.0 { s / px1[i] } else { 0.0 }
                        })
                        .collect();
                    // With a budget the schedule IS the growth rate; the
                    // fraction only paces a fit that was given none.
                    let policy = density::Policy {
                        refine_frac: if cfg.max_gaussians > 0 { 1.0 } else { cfg.densify_frac },
                        dead_opacity: cfg.prune_opacity,
                        ..Default::default()
                    };
                    let seed = 0x6879_6272_6964_u64 ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let r = density::round(&mut scene, &ev, &size_px, &mut suspect, target, &policy, seed);
                    if cfg.log_every > 0 {
                        println!(
                            "fit iter {done:4}: split {}, cloned {}, grown {}, suppressed {}, reclaimed {} (target {target})",
                            r.split, r.cloned, r.grown, r.suppressed, r.reclaimed
                        );
                    }
                }
                // The chain has no use for the gradient statistic: where a
                // sample goes is decided by opacity, which is the model's own
                // statement about whether that sample is explaining anything.
                Densify::Mcmc => {
                    let first = cfg.densify_after.max(cfg.densify_every);
                    let round = done.saturating_sub(first) / cfg.densify_every;
                    let rounds = (until.saturating_sub(first)).div_ceil(cfg.densify_every);
                    let (moved, added) = crate::mcmc::step(&mut scene, cfg, round, rounds);
                    if cfg.log_every > 0 && moved > 0 {
                        println!("fit iter {done:4}: relocated {moved}, added {added}");
                    }
                }
            }
            if cfg.log_every > 0 && scene.len() != before {
                println!("fit iter {done:4}: density control {before} -> {} gaussians", scene.len());
            }
        }
    }
    // Never hand back something worse than what came in. Backoff makes that
    // rare, but "rare" is not a guarantee, and a fit that destroys a scene
    // while reporting a rising number every twenty iterations is the worst of
    // both: it looks like it ran.
    if first_loss.is_finite() && (!loss.is_finite() || loss > first_loss) {
        if cfg.log_every > 0 {
            println!(
                "fit: ended at mse {loss:.6} against {first_loss:.6} at the start; keeping the \
                 input scene. The step size is too large for this scene even after backoff."
            );
        }
        return (init.clone(), targets.iter().map(|t| t.cam).collect(), first_loss);
    }
    (scene, cams, loss)
}

/// The median gaussian's largest axis. Median rather than mean because a
/// reconstruction's size distribution has a long tail that a mean would let
/// set the rate for everything else.
fn median_radius(s: &Splats) -> f32 {
    let mut r: Vec<f32> = (0..s.len()).map(|i| s.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max)).collect();
    r.sort_by(f32::total_cmp);
    r.get(r.len() / 2).copied().unwrap_or(0.0)
}

/// Deterministic per-gaussian jitter in [0,1). A fit has to give the same
/// answer twice, so exploration is pseudo-random in the scene's own indices
/// rather than in wall-clock entropy.
pub(crate) fn jitter(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
}

/// How many of the `ksh` highest SH coefficients are still held out at
/// iteration `global` under [`FitCfg::sh_ramp`].
fn sh_skip(cfg: &FitCfg, ksh: usize, global: usize) -> u32 {
    if cfg.sh_ramp <= 0.0 || ksh == 0 {
        return 0;
    }
    let degree = cfg.sh_degree.min(3) as f32;
    let progress = global as f32 / (cfg.sh_ramp * cfg.iters.max(1) as f32);
    let on = ((progress * (degree + 1.0)).floor() as usize).min(degree as usize);
    (ksh - ((on + 1) * (on + 1) - 1)) as u32
}

/// The views iteration `global` optimizes: `per` of them, walking a fresh
/// deterministic permutation of all `n` each epoch; all of them when `per`
/// covers `n`.
fn batch_views(n: usize, per: usize, global: usize) -> Vec<usize> {
    if per >= n {
        return (0..n).collect();
    }
    (0..per)
        .map(|j| {
            let pos = global * per + j;
            let (epoch, at) = (pos / n, pos % n);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| jitter(a, epoch as u64 ^ 0xba7c).total_cmp(&jitter(b, epoch as u64 ^ 0xba7c)));
            order[at]
        })
        .collect()
}

/// Density control as a step, for tests that need to look at what it did
/// rather than at how a whole fit came out.
pub fn densify_for_test(scene: &mut Splats, grad: &[f32], cfg: &FitCfg, eye: [f32; 3]) {
    densify(scene, grad, cfg, eye)
}

/// Grow the scene where the loss is still pulling hardest, and drop what has
/// gone transparent.
///
/// `grad` is the accumulated magnitude of each gaussian's positional gradient.
/// Upstream 3DGS thresholds that at an absolute 2e-4; a FRACTION is used here
/// instead because this loss is normalized per pixel and per view, so the
/// absolute scale of a gradient depends on image size and view count and no
/// constant transfers between scenes. A fraction also bounds growth by
/// construction, which an absolute threshold does not.
///
/// Large gaussians SPLIT (two children at 1/1.6 the scale, offset along the
/// parent's dominant axis) and small ones CLONE, following the reference: a
/// big gaussian covering detail it cannot represent needs subdividing, while a
/// small one in an under-populated region needs a neighbour.
fn densify(scene: &mut Splats, grad: &[f32], cfg: &FitCfg, eye: [f32; 3]) {
    let n = scene.len();
    if n == 0 || grad.len() != n {
        return;
    }
    let cap = if cfg.max_gaussians > 0 { cfg.max_gaussians } else { usize::MAX };

    // How many SECOND children the whole round may emit. The cap bounded the
    // gradient-chosen set only, so exploration spent budget nobody had
    // counted and a fit handed a hard limit of 256 came back with 258. What
    // the limit is about is the final count, so it is tracked against the
    // survivors: everything that is not pruned is emitted whatever happens.
    //
    // Being AT the cap used to return here, which also skipped the prune -
    // so a scene that started at its budget kept every transparent gaussian
    // it had forever, and density control on it was a no-op rather than a
    // redistribution. Running out of room to grow is not a reason to stop
    // reclaiming.
    let alive = (0..n).filter(|&i| scene.opacities[i] >= cfg.prune_opacity).count();
    let mut extras = cap.saturating_sub(alive);

    // the gradient threshold, as a fraction of the population
    let want = ((n as f32 * cfg.densify_frac) as usize).min(extras);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| grad[b].total_cmp(&grad[a]));
    let chosen: std::collections::HashSet<usize> = order.into_iter().take(want).collect();

    // "large" relative to THIS scene, so the rule does not depend on units
    let mut sizes: Vec<f32> = (0..n).map(|i| scene.scales[i * 3..i * 3 + 3].iter().fold(0.0f32, |m, &v| m.max(v))).collect();
    sizes.sort_by(f32::total_cmp);
    let big = sizes[(n as f32 * 0.8) as usize % n];

    let mut out = Splats::default();
    // Higher-order colour travels with the gaussian it belongs to. It used to
    // be left behind here, so a fit with `sh_degree` above 0 and density
    // control on silently lost every harmonic at the first round and started
    // again from flat colour.
    let shk = crate::mcmc::sh_stride(scene);
    out.sh_rest = scene.sh_rest.as_ref().map(|(d, _)| (*d, Vec::new()));
    let push = |o: &mut Splats, i: usize, dm: [f32; 3], shrink: f32| {
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
    };
    for i in 0..n {
        if scene.opacities[i] < cfg.prune_opacity {
            continue; // pruned: too transparent to be carrying anything
        }
        let s = &scene.scales[i * 3..i * 3 + 3];
        let axis = (0..3).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
        // Anything with no budget for a second child is carried over
        // unchanged, which is what the loop already does for a gaussian
        // nothing selected.
        let room = extras > 0;
        // Exploration: a share of the large gaussians split whether or not the
        // gradient asked, and one in four of those has a child pushed along
        // the viewing ray instead of along the parent's axis.
        let explore = cfg.explore_frac > 0.0 && s[axis] >= big && jitter(i, 0x5eed) < cfg.explore_frac;
        if room && explore && !chosen.contains(&i) && jitter(i, 0xd39d) < 0.25 {
            let d = [
                scene.means[i * 3] - eye[0],
                scene.means[i * 3 + 1] - eye[1],
                scene.means[i * 3 + 2] - eye[2],
            ];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
            // a step scaled to the gaussian, not to the scene, so a near
            // object is not flung past a far one
            let step = s[axis] * (0.5 + 2.0 * jitter(i, 0xa17e));
            let u = [d[0] / l * step, d[1] / l * step, d[2] / l * step];
            extras -= 1;
            push(&mut out, i, [0.0; 3], 1.0);
            push(&mut out, i, u, 1.0);
            continue;
        }
        if room && (chosen.contains(&i) || explore) && s[axis] >= big {
            // split: two smaller children straddling the parent's long axis,
            // rotated into world space by the parent's own orientation
            let q = &scene.quats[i * 4..i * 4 + 4];
            let nq = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
            let (w, x, y, z) = (q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq);
            let col = match axis {
                0 => [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
                1 => [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
                _ => [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
            };
            let d = s[axis] * 0.5;
            extras -= 1;
            push(&mut out, i, [col[0] * d, col[1] * d, col[2] * d], 1.0 / 1.6);
            push(&mut out, i, [-col[0] * d, -col[1] * d, -col[2] * d], 1.0 / 1.6);
        } else if room && chosen.contains(&i) {
            // clone: a second gaussian for the optimizer to walk off the first
            extras -= 1;
            push(&mut out, i, [0.0; 3], 1.0);
            push(&mut out, i, [0.0; 3], 1.0);
        } else {
            push(&mut out, i, [0.0; 3], 1.0);
        }
    }
    if !out.is_empty() {
        *scene = out;
    }
}

/// One run of the optimizer over a FIXED set of gaussians. Returns the scene,
/// the last loss, each gaussian's accumulated positional-gradient magnitude,
/// and whether `on_step` asked to stop.
#[allow(clippy::too_many_arguments)]
/// Where an iteration's wall clock goes.
///
/// Every phase below ends at a device sync - a readback, or a submit the next
/// readback waits on - so host timing is a faithful account rather than an
/// approximation, and it is the only account that includes the host-side work
/// and the transfers, which is where a splat fit tends to actually spend its
/// day. Set `BRAIN_SPLAT_PROFILE=1` to print it.
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
    fn report(&self, iters: usize, n: usize, views: usize) {
        if !self.on || iters == 0 {
            return;
        }
        let total: f64 = self.t.iter().map(|(_, v)| v).sum();
        println!("\nfit profile: {n} gaussians x {views} view(s), {iters} iters, {total:.1}s total");
        let mut rows = self.t.clone();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (k, v) in rows {
            println!("  {k:22} {:8.1} ms/iter  {:5.1}%", 1e3 * v / iters as f64, 100.0 * v / total);
        }
    }
}

fn fit_stage(
    gpu: &Gpu,
    ks: Kernels,
    init: &Splats,
    radius: f32,
    targets: &[TargetView],
    cams: &mut [Camera],
    cfg: &FitCfg,
    isp: &mut Option<Isp>,
    iters: usize,
    it0: usize,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> StageOut {
    let n = init.len();
    let (maxw, maxh) = targets
        .iter()
        .fold((0u32, 0u32), |(mw, mh), t| (mw.max(t.cam.width), mh.max(t.cam.height)));
    // Pose refinement state: six numbers per camera with their own Adam
    // moments, so `pose_lr` means the same thing whatever the scene's scale
    // and however many gaussians happen to be pulling on it.
    const POSE_SLOTS: usize = 4096;
    let pose_buf = gpu.storage(6 * POSE_SLOTS as u64);
    let mut pose_m = vec![[0.0f64; 6]; targets.len()];
    let mut pose_v = vec![[0.0f64; 6]; targets.len()];
    let max_px = (maxw * maxh) as usize;

    // ---- parameter buffers ----
    let mut packed = Vec::with_capacity(n * 10);
    for i in 0..n {
        packed.extend_from_slice(&init.means[i * 3..i * 3 + 3]);
        packed.extend_from_slice(&init.scales[i * 3..i * 3 + 3]);
        packed.extend_from_slice(&init.quats[i * 4..i * 4 + 4]);
    }
    let p_geo = gpu.storage_init("fit.geo", &packed);
    // The host's copy of what `p_geo` holds, refreshed by the clamp step
    // that already reads it back: the auxiliary features are functions of it.
    let mut host_geo = packed;
    let p_op = gpu.storage_init("fit.op", &init.opacities);
    let p_col = gpu.storage_init("fit.col", &init.colors);
    // View-dependent colour. `p_col` stays the base (what a degree-0 scene
    // shows from everywhere) and `p_sh` holds what varies with direction, so
    // a fit with sh_degree 0 is byte-for-byte the fit that was here before.
    let ksh = match cfg.sh_degree {
        0 => 0usize,
        1 => 3,
        2 => 8,
        _ => 15,
    };
    let sh_init: Vec<f32> = match (&init.sh_rest, ksh) {
        (_, 0) => Vec::new(),
        (Some((d, r)), _) if (((*d + 1) * (*d + 1) - 1) as usize) == ksh && r.len() == n * 3 * ksh => r.clone(),
        _ => vec![0.0; n * 3 * ksh],
    };
    let p_sh = gpu.storage_init("fit.sh", if sh_init.is_empty() { &[0.0f32] } else { &sh_init });
    let col_view = gpu.storage(3 * n as u64);
    let d_base = gpu.storage(3 * n as u64);
    let d_sh = gpu.storage((3 * n * ksh).max(1) as u64);
    let shn = (3 * n * ksh).max(1) as u64;
    let (m_sh, v_sh) = (gpu.storage(shn), gpu.storage(shn));
    let means = gpu.storage(3 * n as u64);
    let scales = gpu.storage(3 * n as u64);
    let quats = gpu.storage(4 * n as u64);
    let adam = |numel: usize| (gpu.storage(numel as u64), gpu.storage(numel as u64));
    let (m_geo, v_geo) = adam(10 * n);
    let (m_op, v_op) = adam(n);
    let (m_col, v_col) = adam(3 * n);
    let grads = SplatGrads::new(gpu, n);
    let dimg = gpu.storage(4 * max_px as u64);
    let ddepth = gpu.storage(max_px as u64);
    // Auxiliary passes composite a per-gaussian FEATURE through the same
    // rasterizer (see `crate::geometry` and `crate::density`). The geometry
    // passes share the training gradients for geometry and opacity - the
    // alpha chain is the same parameters - and keep their own feature slot;
    // the credit pass is a measurement and owns every buffer it writes.
    let geometry_terms =
        cfg.distortion_weight > 0.0 || cfg.normal_consistency_weight > 0.0 || cfg.normal_prior_weight > 0.0;
    let hybrid = cfg.densify_every > 0 && cfg.strategy == Densify::Hybrid;
    let aux = geometry_terms.then(|| Aux {
        feat: gpu.storage(3 * n as u64),
        grads: SplatGrads {
            d_gauss: grads.d_gauss.clone(),
            d_opac: grads.d_opac.clone(),
            d_colors: gpu.storage(3 * n as u64),
            d_absgrad: gpu.storage(n as u64),
            d_sumgrad: gpu.storage(2 * n as u64),
        },
    });
    let credit = hybrid.then(|| SplatGrads::new(gpu, n));
    let mut evidence = hybrid.then(|| Evidence::new(n));
    let edges: Vec<Vec<f32>> = if hybrid {
        targets.iter().map(|t| density::edge_map(&t.rgb, t.cam.width as usize, t.cam.height as usize)).collect()
    } else {
        Vec::new()
    };
    let mut renderer = Renderer::new(gpu, ks, n, maxw, maxh, 0).growable();
    let mut bscr = BwdScratch::new(gpu, n, max_px, 0);
    let opts = RenderOpts {
        mode: Mode::Color,
        eps2d: cfg.eps2d,
        antialiased: cfg.antialiased,
        ..Default::default()
    };

    // The per-gaussian scale floor, from where the cameras actually were.
    // Computed once from the starting geometry: a fit moves means by far less
    // than it would take to change which view sampled a point most densely.
    let floor: Vec<f32> = if cfg.mip_scale > 0.0 {
        crate::mip::smoothing_sigma(init, cams, cfg.mip_scale)
            .into_iter()
            .map(|v| v.max(cfg.min_scale))
            .collect()
    } else {
        vec![cfg.min_scale; n]
    };
    // The same conversion gives the ceiling: `smoothing_sigma` is "this many
    // pixels, in world units, at the rate this gaussian was best sampled".
    let mut ceil: Vec<f32> = if cfg.max_scale_pixels > 0.0 {
        crate::mip::smoothing_sigma(init, cams, cfg.max_scale_pixels)
            .into_iter()
            .map(|v| if v > 0.0 { v } else { 0.3 })
            .collect()
    } else {
        vec![0.3; n]
    };
    // and no gaussian may grow far past what it started as, whichever bound
    // is tighter for it
    if cfg.max_growth > 0.0 {
        for i in 0..n.min(ceil.len()) {
            let start = init.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max);
            if start > 0.0 {
                ceil[i] = ceil[i].min(start * cfg.max_growth);
            }
        }
    }

    // `adamw.wgsl` (M6.4) binds param/grad/m/v PLUS a per-tensor `numel`
    // descriptor and a device-resident grad-scale coefficient, and reads its
    // hyperparameters from an 8-field uniform (lr, beta1, beta2, eps, wd,
    // bc1, bc2, scale); see `crates/optim::Optim::build`/`::step`, the
    // canonical caller this mirrors. `desc`/`coef` are write-once (numel is
    // fixed for the run; no grad clipping here so coef is always 1.0);
    // `hparams` is rewritten once per iteration (only bc1/bc2 move).
    let unit_coef = gpu.storage(1);
    gpu.write(&unit_coef, &[f(1.0)]);
    let mk_desc = |numel: usize| {
        // Shape AND contents from `adamw.wgsl`'s own declaration of them - a
        // descriptor one word short of what the kernel reads does not fail,
        // it zeroes the update. No LoRA+ groups here, so every tensor is 1.0.
        let words = kernels::adamw_desc(numel, 1.0);
        let d = gpu.storage(words.len() as u64);
        gpu.write(&d, &words);
        d
    };
    // Per-component learning rates for the packed gaussian, derived from the
    // scene rather than configured in absolute units: a budget is a multiple
    // of `radius` (see [`median_radius`]), and Adam's step is ~lr, so
    // `budget * radius / iters` is the per-step rate that spends exactly that
    // budget over the fit.
    let (desc_geo, lr_pos) = {
        let rad = radius.max(1e-12);
        let steps = cfg.iters.max(1) as f32;
        let mult = |budget: f32, unit: f32| {
            if budget > 0.0 && cfg.lr > 0.0 { budget * unit / steps / cfg.lr } else { 1.0 }
        };
        let (mp, ms, mq) = (
            mult(cfg.position_budget, rad),
            mult(cfg.scale_budget, rad),
            mult(cfg.rotation_budget, 1.0),
        );
        let group = [mp, mp, mp, ms, ms, ms, mq, mq, mq, mq];
        let words = kernels::adamw_desc_grouped(10 * n, 1.0, &group);
        let d = gpu.storage(words.len() as u64);
        gpu.write(&d, &words);
        (d, cfg.lr * mp)
    };
    let desc_op = mk_desc(n);
    let desc_col = mk_desc(3 * n);
    let desc_sh = mk_desc((3 * n * ksh).max(1));
    let hparams = gpu.uniform_dynamic(8);

    let adamw_step = |bufs: [&DeviceBuffer; 4], desc: &DeviceBuffer, numel: usize| {
        let s = gpu.step_buf(
            ks.adamw,
            &hparams,
            &[bufs[0], bufs[1], bufs[2], bufs[3], desc, &unit_coef],
            numel as u32,
        );
        gpu.submit(&[], &[s]);
    };

    // Per-view supervision weights (mask x clipped pixels), fixed for the
    // stage: both are properties of the photographs, not of the scene.
    let weights: Vec<Option<Vec<f32>>> = targets.iter().map(|t| t.weights(isp.as_ref())).collect();
    // The photographs, their weights and the loss scratch live on the device
    // for the stage: the pixel loss runs there (`crate::loss`).
    let wsums: Vec<f64> = targets
        .iter()
        .zip(&weights)
        .map(|(t, w)| w.as_ref().map_or((t.cam.width * t.cam.height) as f64, |m| m.iter().map(|&v| v as f64).sum()))
        .collect();
    let tgt_bufs: Vec<DeviceBuffer> = targets.iter().map(|t| gpu.storage_init("fit.target", &t.rgb)).collect();
    let wt_bufs: Vec<Option<DeviceBuffer>> =
        weights.iter().map(|w| w.as_ref().map(|m| gpu.storage_init("fit.weight", m))).collect();
    let dloss = DeviceLoss::new(gpu, max_px);
    let pred_buf = gpu.storage(4 * max_px as u64);

    let mut last_loss = 0.0f32;
    // Step-size backoff state: see where `rises` is updated below.
    let mut lr_scale = 1.0f32;
    let mut rises = 0u32;
    let mut prev_loss = f32::INFINITY;
    let mut aborted = false;
    // Positional-gradient magnitude per gaussian, averaged over the tail of
    // the stage. Read back over a WINDOW rather than every iteration: one
    // iteration is noisy and every iteration would be 40 bytes per gaussian
    // per step off the device for a signal that is used once.
    // With a minibatch the window has to span an epoch, or some views never
    // contribute to what density control reads.
    let views_per_iter = if cfg.batch == 0 || cfg.batch >= targets.len() { targets.len() } else { cfg.batch };
    let window = targets.len().div_ceil(views_per_iter).max(4).clamp(1, iters.max(1));
    let mut smooth = f64::NAN;
    let mut gsum = vec![0.0f32; n];
    let mut prof = Prof::new();
    for it in 0..iters {
        // zero grads
        let t0 = std::time::Instant::now();
        gpu.submit(
            &[
                &grads.d_gauss,
                &grads.d_opac,
                &grads.d_colors,
                &grads.d_absgrad,
                &grads.d_sumgrad,
                &d_base,
                &d_sh,
            ],
            &[],
        );
        prof.add("zero grads", t0.elapsed());
        let mut loss_sum = 0.0f64;
        // Geometry gradient the host chain rules produce from auxiliary
        // feature gradients, added to the device gradient before the step.
        let mut extra_geo = if aux.is_some() { vec![0.0f32; 10 * n] } else { Vec::new() };
        // Running totals of the pose reduction. It is LINEAR in the gaussian
        // gradients, and those accumulate across views, so differencing the
        // running total gives each view's own contribution exactly - no extra
        // buffer and no per-view zeroing.
        let mut pose_running = [0.0f64; 6];
        let mut pose_grads: Vec<[f64; 6]> = vec![[0.0; 6]; targets.len()];
        let batch = batch_views(targets.len(), views_per_iter, it0 + it);
        for &vi in &batch {
            let t = &targets[vi];
            let cam = cams[vi];
            let px = (cam.width * cam.height) as usize;
            // unpack params for the forward
            let tm = std::time::Instant::now();
            let unpack = gpu.step(
                ks.splat_unpack,
                &[&p_geo, &means, &scales, &quats],
                &[n as u32],
                n as u32,
            );
            gpu.submit(&[], &[unpack]);
            prof.add("unpack params", tm.elapsed());
            let eye = [cam.c2w[3], cam.c2w[7], cam.c2w[11]];
            let sh_params =
                [n as u32, ksh as u32, 0, sh_skip(cfg, ksh, it0 + it), f(eye[0]), f(eye[1]), f(eye[2]), 0];
            if ksh > 0 {
                let e = gpu.step(
                    ks.splat_sh,
                    &[&means, &p_col, &p_sh, &col_view, &d_base, &d_sh],
                    &sh_params,
                    n as u32,
                );
                gpu.submit(&[], &[e]);
            }
            let gs = GpuSplats {
                n,
                means: means.clone(),
                quats: quats.clone(),
                scales: scales.clone(),
                opacities: p_op.clone(),
                colors: if ksh > 0 { col_view.clone() } else { p_col.clone() },
            };
            let tm = std::time::Instant::now();
            renderer.render(gpu, &gs, &cam, &opts);
            prof.add("render forward", tm.elapsed());
            // The mask divides out of the normalizer as well as multiplying
            // into the loss, so the number reported is the loss of the pixels
            // that were actually supervised - comparable with an unmasked run
            // rather than diluted by however much of the frame was excluded.
            let wts = weights[vi].as_deref();
            let wsum = wsums[vi];
            if wsum <= 0.0 {
                continue; // this view's mask keeps nothing
            }
            let want_depth = cfg.depth_weight > 0.0 && t.depth.is_some();
            // The render is radiance; the photograph is what this view's
            // camera made of it. With no camera model the render IS the
            // prediction and never leaves the device.
            let tm = std::time::Instant::now();
            let img = if isp.is_some() || want_depth { renderer.read_rgba(gpu, cam.width, cam.height) } else { Vec::new() };
            prof.add("read image back", tm.elapsed());
            let tm = std::time::Instant::now();
            let rgb = if isp.is_some() { crate::renderer::rgba_to_rgb(&img) } else { Vec::new() };
            let pred_dev = match isp.as_ref() {
                None => &renderer.img,
                Some(model) => {
                    let pred = model.forward(vi, &cam, &rgb);
                    let rgba: Vec<f32> = pred.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 0.0]).collect();
                    gpu.write_f32(&pred_buf, &rgba);
                    &pred_buf
                }
            };
            prof.add("camera model", tm.elapsed());
            let tm = std::time::Instant::now();
            loss_sum += dloss.eval(gpu, ks, cfg.loss, pred_dev, &tgt_bufs[vi], wt_bufs[vi].as_ref(), wsum, cam.width, cam.height, &dimg);
            prof.add("pixel loss", tm.elapsed());
            // What still runs on the host - the camera model's backward and
            // the depth term - edits dL/dimg there.
            if isp.is_some() || want_depth {
                let tm = std::time::Instant::now();
                let mut d = gpu.read(&dimg, px * 4);
                if let Some(model) = isp.as_mut() {
                    let g3: Vec<f32> = d.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
                    let g3 = model.backward(vi, &cam, &rgb, &g3);
                    for i in 0..px {
                        d[i * 4..i * 4 + 3].copy_from_slice(&g3[i * 3..i * 3 + 3]);
                    }
                }
                // The depth term, on the SAME normalizer, so `depth_weight`
                // reads as a ratio against RGB rather than as a number whose
                // meaning depends on how much of the frame carries a prior.
                if want_depth {
                    let rendered_depth = renderer.read_depth(gpu, cam.width, cam.height);
                    let mut ddepth_h = vec![0.0f32; px];
                    let tgt = t.depth.as_ref().unwrap();
                    let dscale = (2.0 * cfg.depth_weight as f64 / wsum) as f32;
                    let mut vdn = vec![0.0f32; px];
                    let mut dsum = 0.0f64;
                    for i in 0..px {
                        let w = wts.map_or(1.0, |m| m[i]) * t.depth_conf.as_ref().map_or(1.0, |c| c[i]);
                        if w <= 0.0 || tgt[i] <= 0.0 {
                            continue; // masked out, distrusted, or no prior here
                        }
                        let diff = rendered_depth[i] - tgt[i];
                        dsum += (w * diff * diff) as f64;
                        vdn[i] = dscale * w * diff;
                    }
                    // The depth term belongs in the REPORTED loss too, not
                    // only in the gradient. Everything that watches the loss -
                    // the step size backoff, the guarantee that a fit never
                    // returns something worse than it was given, the number
                    // printed every few iterations - has to be watching the
                    // objective actually being minimised. Leaving depth out of
                    // it made the guard compare a different function: on a
                    // scene that already renders correctly but sits at the
                    // wrong depth, RGB error RISES as the depth term does its
                    // job, so the fit was correctly moving the geometry and
                    // then handing back the untouched input.
                    loss_sum += cfg.depth_weight as f64 * dsum / wsum;
                    crate::renderer::add_expected_depth_vjp(&vdn, &rendered_depth, &img, &mut d, &mut ddepth_h);
                    gpu.write(&ddepth, cast(&ddepth_h));
                }
                gpu.write(&dimg, cast(&d));
                prof.add("camera model + depth dL/dimg", tm.elapsed());
            }
            let tm = std::time::Instant::now();
            renderer
                .render_bwd(
                    gpu, &gs, &cam, &opts, &dimg,
                    if want_depth { Some(&ddepth) } else { None },
                    &mut bscr, &grads,
                )
                .unwrap_or_else(|e| panic!("{e}"));
            prof.add("render backward", tm.elapsed());
            let geometry_now = (it0 + it) as f32 >= cfg.geometry_after * cfg.iters as f32
                && (it0 + it).is_multiple_of(cfg.geometry_every.max(1));
            if let Some(a) = aux.as_ref().filter(|_| geometry_now) {
                let tm = std::time::Instant::now();
                let ctx = AuxCtx { gpu, gs: &gs, dimg: &dimg, ddepth: &ddepth, opts: &opts, cfg };
                loss_sum += geometry_passes(&ctx, &mut renderer, &mut bscr, a, &cam, t, wts, wsum, &host_geo, &mut extra_geo);
                prof.add("geometry passes", tm.elapsed());
            }
            if cfg.pose_lr > 0.0 {
                let tm = std::time::Instant::now();
                let step = gpu.step(
                    ks.splat_pose_grad,
                    &[&means, &quats, &grads.d_gauss, &pose_buf],
                    &[n as u32, POSE_SLOTS as u32, 0, 0, f(eye[0]), f(eye[1]), f(eye[2]), 0],
                    POSE_SLOTS as u32,
                );
                gpu.submit(&[], &[step]);
                let part = gpu.read(&pose_buf, 6 * POSE_SLOTS);
                let mut total = [0.0f64; 6];
                for c in part.chunks_exact(6) {
                    for k in 0..6 {
                        total[k] += c[k] as f64;
                    }
                }
                for k in 0..6 {
                    pose_grads[vi][k] = total[k] - pose_running[k];
                    pose_running[k] = total[k];
                }
                prof.add("pose gradient", tm.elapsed());
            }
            if ksh > 0 {
                // The rasterizer wrote d/d(colour shown from HERE); split it
                // between the base and the direction-dependent part while the
                // direction that produced it is still the current one.
                let mut vjp = sh_params;
                vjp[2] = 1;
                let e = gpu.step(
                    ks.splat_sh,
                    &[&means, &p_col, &p_sh, &grads.d_colors, &d_base, &d_sh],
                    &vjp,
                    n as u32,
                );
                gpu.submit(&[], &[e]);
                gpu.submit(&[&grads.d_colors], &[]);
            }
        }
        if extra_geo.iter().any(|v| *v != 0.0) {
            let mut dg = gpu.read(&grads.d_gauss, 10 * n);
            for (a, b) in dg.iter_mut().zip(&extra_geo) {
                *a += b;
            }
            gpu.write(&grads.d_gauss, cast(&dg));
        }
        if let Some(model) = isp.as_mut() {
            model.step((it0 + it) as f32 / cfg.iters.max(1) as f32);
        }
        // Adam's bias correction counts from the start of THIS stage, because
        // its moments do too: m and v are fresh buffers per stage, and pairing
        // zeroed moments with a bias correction for a much later timestep is
        // not a well-formed Adam step. Staging still costs something - measured
        // at ~3.5% worse final loss than a single unbroken run, from losing the
        // momentum - which density control has to earn back before it pays.
        let ts = it as i32 + 1;
        let bc1 = 1.0 - 0.9f32.powi(ts);
        let bc2 = 1.0 - 0.999f32.powi(ts);
        let tm = std::time::Instant::now();
        gpu.write(&hparams, &[f(cfg.lr * lr_scale), f(0.9), f(0.999), f(1e-8), f(0.0), f(bc1), f(bc2), f(1.0)]);
        adamw_step([&p_geo, &grads.d_gauss, &m_geo, &v_geo], &desc_geo, 10 * n);
        adamw_step([&p_op, &grads.d_opac, &m_op, &v_op], &desc_op, n);
        if ksh > 0 {
            adamw_step([&p_col, &d_base, &m_col, &v_col], &desc_col, 3 * n);
            adamw_step([&p_sh, &d_sh, &m_sh, &v_sh], &desc_sh, 3 * n * ksh);
        } else {
            adamw_step([&p_col, &grads.d_colors, &m_col, &v_col], &desc_col, 3 * n);
        }
        prof.add("adamw", tm.elapsed());

        if cfg.pose_lr > 0.0 {
            let ts = it as i32 + 1;
            let (b1, b2) = (0.9f64, 0.999f64);
            let (bc1, bc2) = (1.0 - b1.powi(ts), 1.0 - b2.powi(ts));
            // only the views this step measured have a gradient to follow
            for &vi in &batch {
                // The reduction measured a SCENE motion about the camera
                // centre; the camera moves the opposite way, and in its own
                // frame, which is what keeps rotation and translation from
                // trading against each other.
                let r = rot3(&cams[vi].c2w);
                let du = rt_mul(&r, &pose_grads[vi][3..6]);
                let dv = rt_mul(&r, &pose_grads[vi][0..3]);
                let g = [-du[0], -du[1], -du[2], -dv[0], -dv[1], -dv[2]];
                let mut om = [0.0f64; 3];
                let mut ta = [0.0f64; 3];
                for k in 0..6 {
                    pose_m[vi][k] = b1 * pose_m[vi][k] + (1.0 - b1) * g[k];
                    pose_v[vi][k] = b2 * pose_v[vi][k] + (1.0 - b2) * g[k] * g[k];
                    let step = -(cfg.pose_lr as f64) * (pose_m[vi][k] / bc1)
                        / ((pose_v[vi][k] / bc2).sqrt() + 1e-12);
                    if k < 3 {
                        om[k] = step;
                    } else {
                        ta[k - 3] = step;
                    }
                }
                cams[vi].c2w = compose_local(&cams[vi].c2w, &om, &ta);
            }
        }
        // projected-gradient clamps (host; N is fit-sized)
        let tm = std::time::Instant::now();
        let mut geo = gpu.read(&p_geo, 10 * n);
        let mut op = gpu.read(&p_op, n);
        prof.add("clamp: readback", tm.elapsed());
        let tm = std::time::Instant::now();
        // The budgets are in each gaussian's OWN radii: the device stepped
        // every position and scale at the rate of the reference radius, and
        // each gaussian's step is rescaled here by its size relative to it.
        // One shared rate cannot serve both ends of a scene: the fine
        // gaussians density control makes jitter into fog at the rate a coarse
        // one needs to grow across an object structure from motion left
        // empty, and at the fine ones' rate the coarse ones never arrive.
        if cfg.position_budget > 0.0 || cfg.scale_budget > 0.0 {
            let rad = radius.max(1e-12);
            for i in 0..n {
                let prev = &host_geo[i * 10..i * 10 + 10];
                let own = prev[3..6].iter().copied().fold(0.0f32, f32::max) / rad;
                let (from, to) = (if cfg.position_budget > 0.0 { 0 } else { 3 }, if cfg.scale_budget > 0.0 { 6 } else { 3 });
                for k in from..to {
                    geo[i * 10 + k] = prev[k] + (geo[i * 10 + k] - prev[k]) * own;
                }
            }
        }
        // The chain's exploration noise, after the rescale (Eq. 8 already
        // scales it by each gaussian's own covariance) and before the clamps,
        // so a gaussian cannot be pushed out of bounds and left there. It
        // rides along on the readback the clamps already needed, so sampling
        // costs no extra transfer.
        if cfg.strategy == Densify::Mcmc {
            // Eq. 8's noise is proportional to the learning rate of the
            // POSITIONS it perturbs, which is `lr_pos` and not `cfg.lr` - the
            // two stopped being the same number when the geometry groups were
            // budgeted separately. Scaled by the wrong one, the chain is
            // shaken harder than it descends and relocation loses to the
            // heuristic it is supposed to beat.
            crate::mcmc::add_noise(&mut geo, &op, cfg.mcmc_noise * lr_pos, (it0 + it) as u64 + 1);
        }
        for i in 0..n {
            let lo = floor[i.min(floor.len() - 1)];
            let hi = ceil[i.min(ceil.len() - 1)].max(lo);
            for k in 3..6 {
                geo[i * 10 + k] = geo[i * 10 + k].clamp(lo, hi);
            }
            clamp_axes(&mut geo[i * 10 + 3..i * 10 + 6], cfg.max_needle, cfg.max_flat);
        }
        prof.add("clamp: host loop", tm.elapsed());
        let tm = std::time::Instant::now();
        gpu.write(&p_geo, cast(&geo));
        host_geo = geo;
        // The chain's opacity regularizer, as decoupled decay: every gaussian
        // fades a little every step and only the loss puts it back, so the
        // ones explaining nothing end up below `prune_opacity` where
        // relocation can recycle them.
        let fade = if cfg.strategy == Densify::Mcmc { 1.0 - cfg.lr * cfg.mcmc_opacity_decay } else { 1.0 };
        for v in op.iter_mut() {
            *v = (*v * fade).clamp(1e-4, 1.0 - 1e-4);
        }
        gpu.write(&p_op, cast(&op));
        prof.add("clamp: write back", tm.elapsed());

        if cfg.densify_every > 0 && it + window >= iters {
            let tm = std::time::Instant::now();
            // The homodirectional criterion, summed per pixel in the reduction
            // rather than taken as the norm of the summed gradient here. The
            // two differ most for exactly the gaussians density control exists
            // to find: one big enough to straddle an edge is pulled both ways,
            // and the norm of the sum reports it as converged.
            let ag = gpu.read(&grads.d_absgrad, n);
            for i in 0..n {
                gsum[i] += ag[i];
            }
            prof.add("densify grad readback", tm.elapsed());
        }

        // An epoch-length average of the batch losses; with the full batch the
        // weight is 1 and this is the batch loss itself.
        let batch_loss = loss_sum / batch.len() as f64;
        let alpha = batch.len() as f64 / targets.len() as f64;
        smooth = if smooth.is_nan() { batch_loss } else { (1.0 - alpha) * smooth + alpha * batch_loss };
        last_loss = smooth as f32;
        // Back off a step size this scene will not take. The rate is already
        // normalised against the scene's extent, which makes one value work
        // across scales but does not make it work everywhere: a rate that
        // converges on one capture can diverge on the next, and Adam has no
        // opinion about that. Two rises in a row is the signal - one can be a
        // clamp or an unlucky view ordering, two is a trend. With a minibatch
        // "in a row" is counted in EPOCHS: consecutive batches see different
        // views, and their losses rise and fall with which views they drew.
        let epoch = targets.len().div_ceil(views_per_iter);
        if (it + 1).is_multiple_of(epoch) {
            if last_loss > prev_loss {
                rises += 1;
            } else {
                rises = 0;
            }
            prev_loss = last_loss;
        }
        if rises >= 2 && lr_scale > 1e-3 {
            lr_scale *= 0.5;
            rises = 0;
            if cfg.log_every > 0 {
                println!("fit iter {:4}: loss rising, learning rate -> {:.3e}", it0 + it, cfg.lr * lr_scale);
            }
        }
        let global = it0 + it;
        if cfg.log_every > 0 && (global.is_multiple_of(cfg.log_every) || global + 1 == cfg.iters) {
            println!("fit iter {global:4}: loss {last_loss:.6}");
        }
        if !on_step(global, last_loss) {
            aborted = true;
            break;
        }
    }

    // Credit assignment for density control: EVERY view once, against the
    // parameters the stage ends with. Collected inside the loop over the
    // stage's last iterations instead, a minibatch window straddles two
    // epochs with different view orders - some views twice, some never - and
    // a gaussian seen only in the missed views reads as dead: measured as 41%
    // of a sparse start's first round.
    if let (Some(cg), Some(ev)) = (&credit, evidence.as_mut()) {
        if !aborted {
            let tm = std::time::Instant::now();
            let unpack = gpu.step(ks.splat_unpack, &[&p_geo, &means, &scales, &quats], &[n as u32], n as u32);
            gpu.submit(&[], &[unpack]);
            let last = it0 + iters.saturating_sub(1);
            for (vi, t) in targets.iter().enumerate() {
                let cam = cams[vi];
                if ksh > 0 {
                    let eye = cam.eye();
                    let params = [n as u32, ksh as u32, 0, sh_skip(cfg, ksh, last), f(eye[0]), f(eye[1]), f(eye[2]), 0];
                    let e = gpu.step(ks.splat_sh, &[&means, &p_col, &p_sh, &col_view, &d_base, &d_sh], &params, n as u32);
                    gpu.submit(&[], &[e]);
                }
                let gs = GpuSplats {
                    n,
                    means: means.clone(),
                    quats: quats.clone(),
                    scales: scales.clone(),
                    opacities: p_op.clone(),
                    colors: if ksh > 0 { col_view.clone() } else { p_col.clone() },
                };
                renderer.render(gpu, &gs, &cam, &opts);
                let rgb = crate::renderer::rgba_to_rgb(&renderer.read_rgba(gpu, cam.width, cam.height));
                let pred = match isp.as_ref() {
                    None => rgb,
                    Some(model) => model.forward(vi, &cam, &rgb),
                };
                let up = density::credit_upstream(&pred, &t.rgb, &edges[vi], weights[vi].as_deref());
                gpu.write(&dimg, cast(&up));
                gpu.submit(&[&cg.d_colors], &[]);
                renderer.render_bwd(gpu, &gs, &cam, &opts, &dimg, None, &mut bscr, cg).unwrap_or_else(|e| panic!("{e}"));
                ev.add_view(&gpu.read(&cg.d_colors, 3 * n));
            }
            prof.add("credit assignment", tm.elapsed());
        }
    }

    prof.report(iters, n, targets.len());

    // read back the optimized scene
    let geo = gpu.read(&p_geo, 10 * n);
    let op = gpu.read(&p_op, n);
    let col = gpu.read(&p_col, 3 * n);
    let mut out = Splats::default();
    if ksh > 0 {
        out.sh_rest = Some((cfg.sh_degree, gpu.read(&p_sh, 3 * n * ksh)));
    }
    for i in 0..n {
        out.means.extend_from_slice(&geo[i * 10..i * 10 + 3]);
        out.scales.extend_from_slice(&geo[i * 10 + 3..i * 10 + 6]);
        out.quats.extend_from_slice(&geo[i * 10 + 6..i * 10 + 10]);
        out.opacities.push(op[i]);
        out.colors.extend_from_slice(&col[i * 3..i * 3 + 3]);
    }
    StageOut { scene: out, loss: last_loss, absgrad: gsum, evidence, aborted }
}

/// What one run of the optimizer over a fixed set of gaussians produced.
struct StageOut {
    scene: Splats,
    /// The objective at the last iteration.
    loss: f32,
    /// Each gaussian's AbsGS positional-gradient magnitude over the tail of
    /// the stage.
    absgrad: Vec<f32>,
    /// Credit-assignment evidence over the same tail, when the hybrid
    /// density controller asked for it.
    evidence: Option<Evidence>,
    /// Whether `on_step` asked to stop.
    aborted: bool,
}

/// Buffers of the geometry passes.
struct Aux {
    /// Per-gaussian feature composited in place of colour, `[N*3]`.
    feat: DeviceBuffer,
    /// Shares `d_gauss`/`d_opac` with the training gradients; own feature
    /// and density-statistic slots.
    grads: SplatGrads,
}

/// What every auxiliary pass of one view shares.
struct AuxCtx<'a> {
    gpu: &'a Gpu,
    /// The view's colour scene; its geometry and opacity are reused.
    gs: &'a GpuSplats,
    dimg: &'a DeviceBuffer,
    ddepth: &'a DeviceBuffer,
    opts: &'a RenderOpts,
    cfg: &'a FitCfg,
}

/// Run the geometry terms that are switched on for one view, accumulating
/// their device gradient into the training gradients and their feature
/// gradient, carried back to means and rotations, into `extra` (`[N*10]`).
/// Returns their contribution to the objective.
#[allow(clippy::too_many_arguments)]
fn geometry_passes(
    ctx: &AuxCtx,
    renderer: &mut Renderer,
    bscr: &mut BwdScratch,
    aux: &Aux,
    cam: &Camera,
    t: &TargetView,
    wts: Option<&[f32]>,
    wsum: f64,
    host_geo: &[f32],
    extra: &mut [f32],
) -> f64 {
    let (gpu, cfg) = (ctx.gpu, ctx.cfg);
    let n = ctx.gs.n;
    // run every `geometry_every` iterations, so each run carries that many
    let every = cfg.geometry_every.max(1) as f32;
    let px = (cam.width * cam.height) as usize;
    // Features composite against nothing: a background would add itself to
    // every sum these terms are defined over.
    let opts = RenderOpts { bg: [0.0; 3], ..*ctx.opts };
    let gs = GpuSplats {
        n,
        means: ctx.gs.means.clone(),
        quats: ctx.gs.quats.clone(),
        scales: ctx.gs.scales.clone(),
        opacities: ctx.gs.opacities.clone(),
        colors: aux.feat.clone(),
    };
    let mut loss = 0.0f64;
    let mut pass = |feat: &[f32], build: &mut dyn FnMut(&[f32], &[f32]) -> geometry::AuxGrad| -> Vec<f32> {
        gpu.write(&aux.feat, cast(feat));
        renderer.render(gpu, &gs, cam, &opts);
        let rgba = renderer.read_rgba(gpu, cam.width, cam.height);
        let depth = renderer.read_depth(gpu, cam.width, cam.height);
        let geometry::AuxGrad { loss: l, dimg, ddepth } = build(&rgba, &depth);
        loss += l;
        gpu.write(ctx.dimg, cast(&dimg));
        if let Some(dd) = &ddepth {
            gpu.write(ctx.ddepth, cast(dd));
        }
        gpu.submit(&[&aux.grads.d_colors], &[]);
        renderer
            .render_bwd(gpu, &gs, cam, &opts, ctx.dimg, ddepth.as_ref().map(|_| ctx.ddepth), bscr, &aux.grads)
            .unwrap_or_else(|e| panic!("{e}"));
        gpu.read(&aux.grads.d_colors, 3 * n)
    };
    if cfg.distortion_weight > 0.0 {
        let feat = geometry::distortion_features(&geometry::depths(host_geo, cam));
        let dfeat = pass(&feat, &mut |rgba, depth| geometry::distortion_loss(rgba, depth, wts, wsum, every * cfg.distortion_weight));
        geometry::distortion_backward(host_geo, cam, &dfeat, extra);
    }
    let prior = t.normals.as_deref().filter(|_| cfg.normal_prior_weight > 0.0);
    if cfg.normal_consistency_weight > 0.0 || prior.is_some() {
        let feat = geometry::normals(host_geo, cam);
        let dfeat = pass(&feat, &mut |rgba, depth| {
            let mut dimg = vec![0.0f32; px * 4];
            let mut l = 0.0;
            if cfg.normal_consistency_weight > 0.0 {
                let alpha: Vec<f32> = rgba.chunks_exact(4).map(|p| p[3]).collect();
                let target = geometry::depth_normals(depth, &alpha, cam);
                l += geometry::normal_loss(rgba, &target, wts, wsum, every * cfg.normal_consistency_weight, &mut dimg);
            }
            if let Some(target) = prior {
                l += geometry::normal_loss(rgba, target, wts, wsum, every * cfg.normal_prior_weight, &mut dimg);
            }
            geometry::AuxGrad { loss: l, dimg, ddepth: None }
        });
        geometry::normals_backward(host_geo, cam, &dfeat, extra);
    }
    loss
}

/// Row-major 3x3 rotation block of a row-major 4x4.
fn rot3(m: &[f32; 16]) -> [f64; 9] {
    std::array::from_fn(|i| m[(i / 3) * 4 + i % 3] as f64)
}

/// `R^T v`, which takes a world-frame vector into the camera's own frame.
fn rt_mul(r: &[f64; 9], v: &[f64]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| r[k * 3 + i] * v[k]).sum())
}

/// `c2w * [exp(omega^) | tau]`: move the camera in its OWN frame.
fn compose_local(c2w: &[f32; 16], omega: &[f64; 3], tau: &[f64; 3]) -> [f32; 16] {
    let th = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    // Rodrigues, with the small-angle limit taken where the series is better
    // conditioned than the closed form.
    let (a, b) = if th < 1e-8 {
        (1.0, 0.5)
    } else {
        (th.sin() / th, (1.0 - th.cos()) / (th * th))
    };
    let k = [
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ];
    let mut d = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let kk: f64 = (0..3).map(|t| k[i][t] * k[t][j]).sum();
            d[i][j] = if i == j { 1.0 } else { 0.0 } + a * k[i][j] + b * kk;
        }
    }
    let r = rot3(c2w);
    let mut out = *c2w;
    for i in 0..3 {
        for j in 0..3 {
            out[i * 4 + j] = (0..3).map(|t| r[i * 3 + t] * d[t][j]).sum::<f64>() as f32;
        }
        out[i * 4 + 3] =
            (c2w[i * 4 + 3] as f64 + (0..3).map(|t| r[i * 3 + t] * tau[t]).sum::<f64>()) as f32;
    }
    out
}

/// The factor that puts a scene roughly one unit across, from the spread of
/// the cameras and the scene together - the cameras are what define what
/// "close" means for a capture, and a scene with one stray gaussian at
/// infinity should not be judged by it.
/// Bound a gaussian's SHAPE, in place, by raising its two smaller axes.
///
/// Sorted `a >= b >= c`, there are two independent ratios and both have to be
/// bounded, because a fit will escape through whichever one is left free.
///
/// `a/b` is prolateness - a stick. Bounding it is what refuses a needle, and
/// it has to be bounded on a/b rather than a/c: an a/c bound permits needles
/// right up to the limit while forbidding the thin discs a surface actually
/// wants. Measured on a real capture, fitting under an a/c bound took the 90th
/// percentile of a/b from 1.94 to 9.29.
///
/// `b/c` is flatness - a disc. A surface element IS a disc, so this one gets a
/// looser bound rather than none, which is what it had. Left free, a 400
/// iteration fit took the 99th percentile of b/c from 4.14 to 26.34 while a/b
/// sat pinned at its own bound of 2.00, and grew the longest axis nearly
/// ninefold. A 14 x 7 x 0.26 pixel blade is not a surface element, and seen
/// edge on it is exactly the needle the other bound exists to refuse - which
/// is how a scene that scores 28.8 dB against its own training views renders
/// as fur from a grazing angle.
///
/// Raising rather than shrinking keeps the constraint from fighting the
/// gradient that grew the long axis: it declines the degenerate shape without
/// discarding what the fit learned.
fn clamp_axes(s: &mut [f32], max_needle: f32, max_flat: f32) {
    // sort the three axis indices by scale, descending
    let (mut i0, mut i1, mut i2) = (0usize, 1usize, 2usize);
    if s[i1] > s[i0] {
        std::mem::swap(&mut i0, &mut i1);
    }
    if s[i2] > s[i0] {
        std::mem::swap(&mut i0, &mut i2);
    }
    if s[i2] > s[i1] {
        std::mem::swap(&mut i1, &mut i2);
    }
    if max_needle > 1.0 {
        s[i1] = s[i1].max(s[i0] / max_needle);
    }
    if max_flat > 1.0 {
        s[i2] = s[i2].max(s[i1] / max_flat);
    }
}

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
        note([t.cam.c2w[3], t.cam.c2w[7], t.cam.c2w[11]]);
    }
    // a coarse subsample is plenty for an order of magnitude
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

fn rescale_cam(c: &Camera, k: f32) -> Camera {
    let mut out = *c;
    for i in 0..3 {
        out.c2w[i * 4 + 3] *= k;
    }
    out
}

fn cast(v: &[f32]) -> &[u32] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    fn sorted(s: [f32; 3]) -> (f32, f32, f32) {
        let mut v = s;
        v.sort_by(f32::total_cmp);
        (v[2], v[1], v[0])
    }

    /// Both degenerate shapes are refused, not just the one that looks like a
    /// stick. A blade is a needle seen edge on, and a fit will find whichever
    /// ratio is left unbounded.
    #[test]
    fn a_blade_is_refused_as_firmly_as_a_needle() {
        let (needle_max, flat_max) = (2.0f32, 4.0f32);
        for raw in [
            [1.0, 0.5, 0.25],        // already fine
            [1.0, 0.02, 0.01],       // a needle
            [1.0, 0.9, 0.004],       // a blade: a/b fine, b/c 225:1
            [0.03, 0.015, 0.00026],  // what a real 400-iteration fit produced
            [1e-6, 1e-9, 1e-12],     // degenerate but positive
        ] {
            let mut s = raw;
            clamp_axes(&mut s, needle_max, flat_max);
            let (a, b, c) = sorted(s);
            assert!(
                a / b <= needle_max * 1.001,
                "{raw:?} -> {s:?}: a/b is {:.2}, past the {needle_max} bound", a / b
            );
            assert!(
                b / c <= flat_max * 1.001,
                "{raw:?} -> {s:?}: b/c is {:.2}, past the {flat_max} bound", b / c
            );
            // and the overall anisotropy is bounded by the product, which is
            // what makes a gaussian look like itself from any direction
            assert!(a / c <= needle_max * flat_max * 1.001, "{raw:?} -> {s:?}: a/c is {:.2}", a / c);
        }
    }

    /// The bound RAISES the small axes and never shrinks the large one, so it
    /// declines a degenerate shape without discarding what the fit learned
    /// about the direction that actually carries detail.
    #[test]
    fn clamping_a_shape_never_shrinks_it() {
        let raw = [0.03, 0.015, 0.00026];
        let mut s = raw;
        clamp_axes(&mut s, 2.0, 4.0);
        for k in 0..3 {
            assert!(s[k] >= raw[k], "axis {k} shrank: {raw:?} -> {s:?}");
        }
    }
}
