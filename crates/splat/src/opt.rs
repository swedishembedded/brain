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
}

impl TargetView {
    /// A view supervised by its colours alone - what `fit` has always done.
    pub fn new(cam: Camera, rgb: Vec<f32>) -> TargetView {
        TargetView { cam, rgb, depth: None, depth_conf: None, mask: None }
    }

    /// Add a depth prior `[W*H]` (0 = no data) and, optionally, how much to
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

    /// Per-pixel loss weight, 1 where no mask was given.
    fn weight(&self, i: usize) -> f32 {
        self.mask.as_ref().map_or(1.0, |m| m[i])
    }

    /// Sum of the per-pixel weights: the loss normalizer, so a masked fit's
    /// MSE is comparable with an unmasked one's.
    fn weight_sum(&self, px: usize) -> f64 {
        match &self.mask {
            None => px as f64,
            Some(m) => m.iter().take(px).map(|v| *v as f64).sum(),
        }
    }
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
    fit_inner(gpu, ks, init, targets, cfg, on_step)
}

pub fn fit(gpu: &Gpu, ks: Kernels, init: &Splats, targets: &[TargetView], cfg: &FitCfg, on_step: &mut dyn FnMut(usize, f32) -> bool) -> (Splats, f32) {
    let (s, _, l) = fit_inner(gpu, ks, init, targets, cfg, on_step);
    (s, l)
}

fn fit_inner(gpu: &Gpu, ks: Kernels, init: &Splats, targets: &[TargetView], cfg: &FitCfg, on_step: &mut dyn FnMut(usize, f32) -> bool) -> (Splats, Vec<Camera>, f32) {
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
            .map(|t| TargetView {
                cam: rescale_cam(&t.cam, k),
                rgb: t.rgb.clone(),
                // A depth prior is in world units, so it rescales with the
                // world. Carrying it through unscaled would have the depth
                // term anchor the scene to a distance the normalization just
                // moved, which is a silent, total corruption of the fit.
                depth: t.depth.as_ref().map(|d| d.iter().map(|v| v * k).collect()),
                depth_conf: t.depth_conf.clone(),
                mask: t.mask.clone(),
            })
            .collect();
        let (s, c, l) = fit_inner(gpu, ks, &scaled_init, &scaled, cfg, on_step);
        return (
            rescale_scene(&s, 1.0 / k),
            c.iter().map(|c| rescale_cam(c, 1.0 / k)).collect(),
            l,
        );
    }
    // With density control off there is exactly ONE stage, and this is the
    // function it always was - same buffers, same Adam state, start to finish.
    let stage_len = if cfg.densify_every == 0 { cfg.iters } else { cfg.densify_every };
    let mut scene = init.clone();
    let mut cams: Vec<Camera> = targets.iter().map(|t| t.cam).collect();
    let mut loss = 0.0f32;
    let mut done = 0usize;
    let mut stop = false;
    // The loss of the scene as handed in, reported by the first iteration
    // (which measures before it updates anything). It is the floor the fit has
    // to beat to have been worth running.
    let mut first_loss = f32::NAN;
    let mut seen = |it: usize, mse: f32, k: &mut dyn FnMut(usize, f32) -> bool| {
        if first_loss.is_nan() {
            first_loss = mse;
        }
        k(it, mse)
    };
    while done < cfg.iters && !stop {
        let iters = stage_len.min(cfg.iters - done);
        let (next, l, grad, aborted) = {
            let mut tap = |it: usize, mse: f32| seen(it, mse, on_step);
            fit_stage(gpu, ks, &scene, targets, &mut cams, cfg, iters, done, &mut tap)
        };
        scene = next;
        loss = l;
        done += iters;
        stop = aborted;
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
/// Deterministic per-gaussian jitter in [0,1). A fit has to give the same
/// answer twice, so exploration is pseudo-random in the scene's own indices
/// rather than in wall-clock entropy.
pub(crate) fn jitter(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
}

/// Density control as a step, for tests that need to look at what it did
/// rather than at how a whole fit came out.
pub fn densify_for_test(scene: &mut Splats, grad: &[f32], cfg: &FitCfg, eye: [f32; 3]) {
    densify(scene, grad, cfg, eye)
}

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
    targets: &[TargetView],
    cams: &mut [Camera],
    cfg: &FitCfg,
    iters: usize,
    it0: usize,
    on_step: &mut dyn FnMut(usize, f32) -> bool,
) -> (Splats, f32, Vec<f32>, bool) {
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
    let mut renderer = Renderer::new(gpu, ks, n, maxw, maxh, 0);
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
    let desc_geo = mk_desc(10 * n);
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
    let window = iters.clamp(1, 4);
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
        // Running totals of the pose reduction. It is LINEAR in the gaussian
        // gradients, and those accumulate across views, so differencing the
        // running total gives each view's own contribution exactly - no extra
        // buffer and no per-view zeroing.
        let mut pose_running = [0.0f64; 6];
        let mut pose_grads: Vec<[f64; 6]> = vec![[0.0; 6]; targets.len()];
        for (vi, t) in targets.iter().enumerate() {
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
                [n as u32, ksh as u32, 0, 0, f(eye[0]), f(eye[1]), f(eye[2]), 0];
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
            // host loss: MSE over rgb; alpha unsupervised
            let tm = std::time::Instant::now();
            let img = renderer.read_rgba(gpu, cam.width, cam.height);
            let want_depth = cfg.depth_weight > 0.0 && t.depth.is_some();
            let rendered_depth =
                if want_depth { renderer.read_depth(gpu, cam.width, cam.height) } else { Vec::new() };
            prof.add("read image back", tm.elapsed());
            let tm = std::time::Instant::now();
            let mut d = vec![0.0f32; px * 4];
            // The mask divides out of the normalizer as well as multiplying
            // into the loss, so the number reported is the MSE of the pixels
            // that were actually supervised - comparable with an unmasked run
            // rather than diluted by however much of the frame was excluded.
            let wsum = t.weight_sum(px);
            if wsum <= 0.0 {
                continue; // this view's mask keeps nothing
            }
            let scale = (2.0 / (wsum * 3.0)) as f32;
            let mut lsum = 0.0f64;
            for i in 0..px {
                let m = t.weight(i);
                if m == 0.0 {
                    continue;
                }
                for c in 0..3 {
                    let diff = img[i * 4 + c] - t.rgb[i * 3 + c];
                    lsum += (m * diff * diff) as f64;
                    d[i * 4 + c] = scale * m * diff;
                }
            }
            loss_sum += lsum / (wsum * 3.0);
            // The depth term, on the SAME normalizer, so `depth_weight` reads
            // as a ratio against RGB rather than as a number whose meaning
            // depends on how much of the frame carries a depth prior.
            let mut ddepth_h = vec![0.0f32; px];
            if want_depth {
                let tgt = t.depth.as_ref().unwrap();
                let dscale = (2.0 * cfg.depth_weight as f64 / wsum) as f32;
                let mut vdn = vec![0.0f32; px];
                let mut dsum = 0.0f64;
                for i in 0..px {
                    let w = t.weight(i) * t.depth_conf.as_ref().map_or(1.0, |c| c[i]);
                    if w <= 0.0 || tgt[i] <= 0.0 {
                        continue; // masked out, distrusted, or no prior here
                    }
                    let diff = rendered_depth[i] - tgt[i];
                    dsum += (w * diff * diff) as f64;
                    vdn[i] = dscale * w * diff;
                }
                // The depth term belongs in the REPORTED loss too, not only in
                // the gradient. Everything that watches the loss - the step
                // size backoff, the guarantee that a fit never returns
                // something worse than it was given, the number printed every
                // few iterations - has to be watching the objective actually
                // being minimised. Leaving depth out of it made the guard
                // compare a different function: on a scene that already
                // renders correctly but sits at the wrong depth, RGB error
                // RISES as the depth term does its job, so the fit was
                // correctly moving the geometry and then handing back the
                // untouched input.
                loss_sum += cfg.depth_weight as f64 * dsum / wsum as f64;
                crate::renderer::add_expected_depth_vjp(&vdn, &rendered_depth, &img, &mut d, &mut ddepth_h);
            }
            prof.add("host loss + dL/dimg", tm.elapsed());
            let tm = std::time::Instant::now();
            gpu.write(&dimg, cast(&d));
            if want_depth {
                gpu.write(&ddepth, cast(&ddepth_h));
            }
            prof.add("upload dL/dimg", tm.elapsed());
            let tm = std::time::Instant::now();
            renderer
                .render_bwd(
                    gpu, &gs, &cam, &opts, &dimg,
                    if want_depth { Some(&ddepth) } else { None },
                    &mut bscr, &grads,
                )
                .unwrap_or_else(|e| panic!("{e}"));
            prof.add("render backward", tm.elapsed());
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
            for vi in 0..cams.len() {
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
        // The chain's exploration noise, before the clamps rather than after,
        // so a gaussian cannot be pushed out of bounds and left there. It
        // rides along on the readback the clamps already needed, so sampling
        // costs no extra transfer.
        if cfg.strategy == Densify::Mcmc {
            crate::mcmc::add_noise(&mut geo, &op, cfg.mcmc_noise * cfg.lr, (it0 + it) as u64 + 1);
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

        last_loss = (loss_sum / targets.len() as f64) as f32;
        // Back off a step size this scene will not take. The rate is already
        // normalised against the scene's extent, which makes one value work
        // across scales but does not make it work everywhere: a rate that
        // converges on one capture can diverge on the next, and Adam has no
        // opinion about that. Two rises in a row is the signal - one can be a
        // clamp or an unlucky view ordering, two is a trend.
        if last_loss > prev_loss {
            rises += 1;
        } else {
            rises = 0;
        }
        if rises >= 2 && lr_scale > 1e-3 {
            lr_scale *= 0.5;
            rises = 0;
            if cfg.log_every > 0 {
                println!("fit iter {:4}: loss rising, learning rate -> {:.3e}", it0 + it, cfg.lr * lr_scale);
            }
        }
        prev_loss = last_loss;
        let global = it0 + it;
        if cfg.log_every > 0 && (global.is_multiple_of(cfg.log_every) || global + 1 == cfg.iters) {
            println!("fit iter {global:4}: mse {last_loss:.6}");
        }
        if !on_step(global, last_loss) {
            aborted = true;
            break;
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
    (out, last_loss, gsum, aborted)
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
