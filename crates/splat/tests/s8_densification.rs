// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adaptive density control: the fit may ADD gaussians where the scene needs
//! detail, instead of only moving and growing the ones it started with.
//!
//! Without it, a fit has exactly one way to cover an under-represented
//! region - make the gaussians it already has bigger. Measured on a real
//! six-photograph reconstruction, that is visible in the size distribution:
//! fitting drove the median projected splat DOWN (0.61 to 0.37 px) while
//! dragging the 99th percentile UP (3.34 to 8.72 px). The scene grows a tail
//! of blobs precisely where it should have grown detail.
//!
//! The reference pipeline has this and brain did not. Upstream 3DGS densifies
//! on the positional gradient, splitting gaussians that are large and cloning
//! those that are small, and prunes the near-transparent ones.
//!
//! Swedish Embedded AB implements 3D Gaussian Splatting optimizers, including
//! the density control that decides where detail can appear at all. If your
//! team needs reconstruction quality that is not capped by its initialization,
//! you can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// A checkerboard slab of `cells`x`cells` gaussians spanning the same area
/// whatever `cells` is - so a coarse one is the SAME scene under-sampled, not
/// a smaller scene.
fn board(cells: usize, flat: bool) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[
                -1.2 + step * (ix as f32 + 0.5),
                -1.2 + step * (iy as f32 + 0.5),
                3.0,
            ]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[step * 0.45, step * 0.45, step * 0.45]);
            s.opacities.push(0.99);
            let v = if flat { 0.5 } else if (ix + iy) % 2 == 0 { 0.88 } else { 0.12 };
            s.colors.extend_from_slice(&[v, v, v * 0.92 + 0.04]);
        }
    }
    s
}

fn targets(g: &Gpu, ks: Kernels, truth: &Splats, w: u32, h: u32) -> Vec<TargetView> {
    let cams = [
        Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([0.7, -0.25, 0.3], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
        Camera::look_at([-0.7, 0.25, 0.3], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 55.0, w, h),
    ];
    let o = RenderOpts::default();
    let mut ren = Renderer::new(g, ks, truth.len(), w, h, 0);
    let gs = GpuSplats::upload(g, truth);
    cams.iter()
        .map(|c| {
            ren.render(g, &gs, c, &o);
            let img = ren.read_rgba(g, c.width, c.height);
            TargetView::new(*c, img.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect())
        })
        .collect()
}

/// The scene to recover has detail at a spatial frequency the starting set of
/// gaussians cannot represent - eight times fewer of them, spread over the
/// same area. No amount of moving, recolouring or resizing 64 gaussians makes
/// a 16x16 checkerboard, so a fit that cannot add any is capped by its
/// initialization and a fit that can is not.
#[test]
fn a_fit_that_can_add_gaussians_beats_one_that_cannot() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let t = targets(&g, ks, &truth, w, h);
    // Sixteen gaussians against a 16x16 checkerboard: no amount of moving,
    // recolouring or resizing them makes the target, so density control is the
    // only thing that can close the gap.
    let coarse = board(4, true);

    // a short fit: every group steps fast enough to settle in 160 iterations
    let fixed = FitCfg { iters: 160, lr_position: 1e-2, lr_color: 1e-2, lr_scale: 1e-2, log_every: 0, densify_every: 0, ..Default::default() };
    let (a, mse_fixed) = fit(&g, ks, &coarse, &t, &fixed, &mut |_, _| true);

    let grown = FitCfg { densify_every: 20, densify_after: 10, densify_frac: 0.30, ..fixed };
    let (b, mse_grown) = fit(&g, ks, &coarse, &t, &grown, &mut |_, _| true);

    // A margin is demanded rather than any improvement being accepted: a
    // round disturbs the fit (3DGS's clones double the opacity at their
    // site until they separate), and density control has to earn that back
    // before it pays at all. The margin is 5%, measured at 6.0% under
    // per-group log-space step sizes; it was 8.7% when one learning rate was
    // a fixed distance for every group, because the fit that CANNOT add
    // gaussians was much weaker then. A stronger baseline leaves density
    // control less to recover, which is the right outcome and not a
    // regression in it.
    assert_eq!(a.len(), coarse.len(), "the fixed-set fit must not change the gaussian count");
    assert!(
        b.len() > coarse.len(),
        "density control added nothing: still {} gaussians from {}",
        b.len(), coarse.len()
    );
    assert!(
        mse_grown < mse_fixed * 0.95,
        "density control did not pay for itself: {mse_grown:.6} against {mse_fixed:.6} for a fit \
         that cannot add gaussians, on a target the initial set cannot represent"
    );

    // and the improvement is visible in the rendered frame, not only in the loss
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, b.len().max(a.len()), w, h, 0);
    let mut render = |s: &Splats| -> Vec<f32> {
        let gs = GpuSplats::upload(&g, s);
        ren.render(&g, &gs, &t[0].cam, &o);
        ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
    };
    let (ra, rb) = (render(&a), render(&b));
    let (da, db) = (psnr(&ra, &t[0].rgb), psnr(&rb, &t[0].rgb));
    // Smaller than the loss margin on purpose: this view is one of three the
    // fit optimised, and the gain is spread across all of them.
    assert!(
        db > da + 0.3,
        "densified fit renders at {db:.1} dB against {da:.1} dB for the fixed one"
    );
}

/// Density control must not run away: a scene that ALREADY represents its
/// targets has no under-reconstructed region to subdivide, so the count must
/// stay in the same order of magnitude rather than doubling every interval.
#[test]
fn density_control_leaves_an_adequate_scene_alone() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(12, false);
    let t = targets(&g, ks, &truth, w, h);

    let cfg = FitCfg { iters: 120, lr_position: 5e-3, log_every: 0, densify_every: 30, densify_after: 10, ..Default::default() };
    let (out, _) = fit(&g, ks, &truth, &t, &cfg, &mut |_, _| true);
    assert!(
        out.len() < truth.len() * 3,
        "density control grew an already-adequate scene from {} to {} gaussians",
        truth.len(), out.len()
    );
}

/// A fitted gaussian has to stay a blob rather than becoming a hair.
///
/// `fit` clamped each axis to `[min_scale, 0.3]` INDEPENDENTLY, which bounds
/// how big a gaussian gets and says nothing about its shape. The optimizer
/// exploited that: a needle aligned with a training view's ray lowers that
/// view's loss and is invisible in it, while from any other angle it is a
/// streak across the frame. Measured on a real 23-view reconstruction, the
/// model's own output was 0.1% needles (worst axis ratio 54:1) and the same
/// scene after 400 iterations of fitting was 40.7% needles at up to 3000:1 -
/// entirely manufactured by the fit, and entirely invisible to a metric that
/// only renders the views it was fitted to.
#[test]
fn fitting_does_not_stretch_gaussians_into_needles() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let t = targets(&g, ks, &truth, w, h);

    // start from a scene that must move a long way, which is when the
    // optimizer reaches for degenerate shapes
    let mut init = board(8, true);
    for v in init.means.iter_mut() {
        *v *= 1.04;
    }

    // Sort the axes a >= b >= c. a/b says how NEEDLE-like a gaussian is and
    // b/c how DISC-like, and only the first is a defect: a disc is a surface
    // element, a needle is a splat that hides along one view's ray and streaks
    // across every other.
    let shape = |s: &Splats| -> (f32, f32) {
        let mut needle = 0.0f32;
        let mut disc = 0.0f32;
        for i in 0..s.len() {
            let mut a = [s.scales[i * 3], s.scales[i * 3 + 1], s.scales[i * 3 + 2]];
            a.sort_by(|x, y| y.total_cmp(x));
            needle = needle.max(a[0] / a[1].max(1e-12));
            disc = disc.max(a[1] / a[2].max(1e-12));
        }
        (needle, disc)
    };

    let cfg = FitCfg { iters: 200, lr_position: 2e-2, log_every: 0, max_needle: 2.0, ..Default::default() };
    let (fitted, _) = fit(&g, ks, &init, &t, &cfg, &mut |_, _| true);
    let (worst, discs) = shape(&fitted);
    assert!(
        worst <= cfg.max_needle + 1e-3,
        "a fitted gaussian reached {worst:.1}:1 long-to-middle against a {}:1 cap; the clamp is \
         not binding",
        cfg.max_needle
    );
    // The bound must not have achieved that by forbidding flat gaussians: a
    // reconstruction of a surface is supposed to be able to make discs.
    assert!(
        discs > cfg.max_needle,
        "the fitted scene's flattest gaussian is only {discs:.1}:1 middle-to-short, so the \
         constraint is squashing surface elements into balls rather than refusing needles"
    );

    // and the cap has to be the thing doing it - without one, the same fit
    // reaches for far more extreme shapes
    let loose = FitCfg { iters: 200, lr_position: 2e-2, log_every: 0, max_needle: 0.0, ..Default::default() };
    let (unclamped, _) = fit(&g, ks, &init, &t, &loose, &mut |_, _| true);
    assert!(
        shape(&unclamped).0 > worst * 1.5,
        "the unclamped fit only reached {:.1}:1, so this test is not exercising the clamp",
        shape(&unclamped).0
    );
}

/// Density control must not be blind to the gaussians it exists to find.
///
/// A gaussian big enough to straddle a detail is pushed one way by the pixels
/// on one side and the other way by the pixels on the other. Those pushes
/// cancel in the summed gradient, so the standard criterion - the magnitude of
/// the summed 2D position gradient - reports the blurriest gaussian in the
/// scene as perfectly converged and never splits it. That is the whole
/// mechanism behind reconstructions that stay soft no matter how long they
/// train, and it is why AbsGS sums magnitudes instead.
///
/// Constructed so the cancellation is exact rather than approximate: one
/// gaussian centred in frame with a uniform image-space gradient. The pull
/// from every pixel left of centre is the exact negative of its mirror on the
/// right, so the summed criterion is zero by symmetry while every pixel is
/// telling the gaussian something.
#[test]
fn the_split_criterion_sees_a_gaussian_that_straddles_a_detail() {
    use splat::renderer::{BwdScratch, SplatGrads};

    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h);

    let s = Splats {
        means: vec![0.0, 0.0, 4.0],
        quats: vec![1.0, 0.0, 0.0, 0.0],
        scales: vec![0.5, 0.5, 0.5],
        opacities: vec![0.8],
        colors: vec![0.9, 0.9, 0.9],
        sh_rest: None,
    };
    let px = (w * h) as usize;
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, 1, w, h, 0);
    let gs = GpuSplats::upload(&g, &s);
    ren.render(&g, &gs, &cam, &o);

    // uniform dL/dimg: symmetric about the gaussian, so the SUM cancels
    let wimg: Vec<f32> = (0..px * 4).map(|i| if i % 4 == 3 { 0.0 } else { 1.0 }).collect();
    let dimg = g.storage_init("dimg", &wimg);
    let grads = SplatGrads::new(&g, 1);
    g.submit(
        &[&grads.d_gauss, &grads.d_opac, &grads.d_colors, &grads.d_absgrad, &grads.d_sumgrad],
        &[],
    );
    let mut scr = BwdScratch::new(&g, 1, px, 0);
    ren.render_bwd(&g, &gs, &cam, &o, &dimg, None, &mut scr, &grads).expect("fits");

    let sg = g.read(&grads.d_sumgrad, 2);
    let summed = (sg[0] * sg[0] + sg[1] * sg[1]).sqrt();
    let homodirectional = g.read(&grads.d_absgrad, 1)[0];

    assert!(
        homodirectional > 1e-3,
        "the homodirectional criterion is {homodirectional:.3e}: every pixel is pulling on this \
         gaussian, so it must not read as zero"
    );
    assert!(
        summed < 0.02 * homodirectional,
        "the summed criterion reads {summed:.3e} against a homodirectional {homodirectional:.3e}. \
         These are supposed to disagree by a lot here - if they do not, the cancellation this \
         test is built on is not happening and it is checking nothing."
    );
}

/// A surface that looks different from different sides cannot be fitted by a
/// scene whose colour does not depend on where you look from.
///
/// Give the fit the SAME geometry photographed from three angles, with the
/// appearance genuinely changing between them - a glossy object, in miniature.
/// With degree-0 colour the optimizer has exactly two moves: average the views
/// and be wrong everywhere, or move geometry until the views disagree less,
/// which is how a shiny object turns into a smear of duplicated surfaces at
/// slightly different depths. Give it somewhere honest to put the variation
/// and it stops doing that.
#[test]
fn a_view_dependent_surface_needs_view_dependent_colour() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (48u32, 48u32);
    let cams = [
        Camera::look_at([-1.6, 0.0, 0.4], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h),
        Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h),
        Camera::look_at([1.6, 0.0, 0.4], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h),
    ];
    // one flat slab, three appearances
    let tints: [[f32; 3]; 3] = [[0.85, 0.20, 0.20], [0.30, 0.75, 0.30], [0.20, 0.25, 0.90]];
    let slab = |tint: [f32; 3]| {
        let mut s = Splats::default();
        for iy in 0..14 {
            for ix in 0..14 {
                let (fx, fy) = (ix as f32 * 0.16 - 1.04, iy as f32 * 0.16 - 1.04);
                s.means.extend_from_slice(&[fx, fy, 4.0]);
                s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
                s.scales.extend_from_slice(&[0.10, 0.10, 0.10]);
                s.opacities.push(0.95);
                let shade = 0.6 + 0.4 * ((ix + iy) % 2) as f32;
                s.colors.extend_from_slice(&[tint[0] * shade, tint[1] * shade, tint[2] * shade]);
            }
        }
        s
    };

    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, slab(tints[0]).len(), w, h, 0);
    let targets: Vec<TargetView> = cams
        .iter()
        .zip(&tints)
        .map(|(c, t)| {
            let truth = slab(*t);
            let gs = GpuSplats::upload(&g, &truth);
            ren.render(&g, &gs, c, &o);
            let rgba = ren.read_rgba(&g, w, h);
            TargetView::new(*c, rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect())
        })
        .collect();

    // start from the middle appearance, so neither fit is handed the answer
    let init = slab([0.45, 0.40, 0.45]);
    let score = |deg: u32| -> f64 {
        // a short fit: colour - and with it the harmonics, at a twentieth of
        // its rate - has to move far in 160 iterations
        let cfg = FitCfg { iters: 160, lr_position: 1e-2, lr_color: 4e-2, log_every: 0, sh_degree: deg, ..Default::default() };
        let (fitted, _) = fit(&g, ks, &init, &targets, &cfg, &mut |_, _| true);
        let mut r = Renderer::new(&g, ks, fitted.len(), w, h, 0);
        let gs = GpuSplats::upload(&g, &fitted);
        // rendered through the same colour model the fit used
        let mut acc = 0.0;
        for t in &targets {
            let rgb = splat::sh::render_rgb(&g, ks, &mut r, &gs, &fitted, &t.cam, &o);
            acc += psnr(&rgb, &t.rgb);
        }
        acc / targets.len() as f64
    };

    let flat = score(0);
    let view_dep = score(2);
    assert!(
        view_dep > flat + 4.0,
        "view-dependent colour scored {view_dep:.1} dB against flat colour's {flat:.1} dB. It is \
         supposed to be able to represent this and flat colour is not, so a small gap means the \
         harmonics are not reaching the loss."
    );
}

/// A scene the optimizer WANTS to inflate: the truth rendered from three
/// nearby views, started from a tenth of its gaussians, so covering what is
/// missing by growing what is left is the cheapest move available.
///
/// Deliberately SMALL in world units. The bound this exercises used to be 0.3
/// world units, which is about 6 px on a scene three units away and roughly
/// 140 px on a real capture whose camera orbit has radius 0.5 - invisible at
/// one scale and ruinous at the other, which is what a bound in world units
/// gets you.
fn inflatable_scene(g: &Gpu, ks: Kernels) -> (Splats, Vec<TargetView>) {
    let (g, ks) = (g, ks);
    let (w, h) = (64u32, 64u32);
    // A SMALL scene, which is the whole point. The old bound of 0.3 world
    // units is only about 6 px on a scene three units away, and roughly 140 on
    // a real capture whose camera orbit has radius 0.5 - the bug is invisible
    // at one scale and ruinous at the other, which is what a bound in world
    // units gets you.
    const K: f32 = 0.1;
    let mut truth = board(16, false);
    for v in truth.means.iter_mut() {
        *v *= K;
    }
    for v in truth.scales.iter_mut() {
        *v *= K;
    }
    let cam_at = |e: [f32; 3]| {
        Camera::look_at(e, [0.0, 0.0, 3.0 * K], [0.0, -1.0, 0.0], 55.0, w, h)
    };
    let o = RenderOpts::default();
    let mut ren = Renderer::new(g, ks, truth.len(), w, h, 0);
    let gs = GpuSplats::upload(g, &truth);
    let shots: Vec<TargetView> = [[0.0, 0.0, 0.0], [0.7 * K, -0.25 * K, 0.3 * K], [-0.7 * K, 0.25 * K, 0.3 * K]]
        .iter()
        .map(|e| {
            let c = cam_at(*e);
            ren.render(g, &gs, &c, &o);
            let img = ren.read_rgba(g, w, h);
            TargetView::new(c, img.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect())
        })
        .collect();

    // Start from a tenth of the gaussians, so covering the scene is the
    // cheapest thing the optimizer can do and growing them is how it does it.
    let mut init = Splats::default();
    for i in (0..truth.len()).step_by(10) {
        init.means.extend_from_slice(&truth.means[i * 3..i * 3 + 3]);
        init.quats.extend_from_slice(&truth.quats[i * 4..i * 4 + 4]);
        init.scales.extend_from_slice(&truth.scales[i * 3..i * 3 + 3]);
        init.opacities.push(truth.opacities[i]);
        init.colors.extend_from_slice(&truth.colors[i * 3..i * 3 + 3]);
    }

    (init, shots)
}

/// A fit may refine what the reconstruction proposed; it may not replace it
/// with something several times larger.
///
/// The pixel ceiling above cannot carry this on its own. It has to be right
/// for every scene, and scenes differ: a sparse one legitimately has gaussians
/// many pixels across, while the pixel-aligned reconstruction this usually
/// fits emits SUB-pixel ones (median 0.6 px on a real capture). Tightening the
/// pixel ceiling far enough to constrain the second stops the first from
/// representing itself - measured, as a convergence test that stopped
/// converging.
///
/// What was actually observed going wrong is GROWTH: a 400 iteration fit
/// inflated the longest axis of a real reconstruction 7.5x, which is what
/// turns a scene into fog seen from anywhere it was not fitted. A bound
/// relative to each gaussian's own starting size is scene-adaptive by
/// construction.
#[test]
fn a_fit_may_not_inflate_a_splat_far_past_where_it_started() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (init, shots) = inflatable_scene(&g, ks);

    let grew = |s: &Splats| -> f32 {
        let mut worst = 0.0f32;
        for i in 0..s.len() {
            let a = s.scales[i * 3].max(s.scales[i * 3 + 1]).max(s.scales[i * 3 + 2]);
            let b = init.scales[i * 3].max(init.scales[i * 3 + 1]).max(init.scales[i * 3 + 2]);
            if b > 0.0 {
                worst = worst.max(a / b);
            }
        }
        worst
    };

    // sizes step fast enough to take up the temptation in 150 iterations;
    // measured on the fitted scale itself, before the 3D filter is baked in
    let base = FitCfg {
        iters: 150, lr_position: 8e-3, lr_scale: 2e-2, log_every: 0, max_scale_pixels: 0.0, max_growth: 0.0,
        ..Default::default()
    };
    let loose = splat::opt::fit_full(&g, ks, &init, &shots, &base, &mut |_, _| true).scene;
    let cap = 2.0f32;
    let held = splat::opt::fit_full(&g, ks, &init, &shots, &FitCfg { max_growth: cap, ..base }, &mut |_, _| true).scene;

    let (l, h) = (grew(&loose), grew(&held));
    assert!(
        l > cap * 1.5,
        "the unbounded fit grew a splat only {l:.2}x, so this scene never tempted it to inflate \
         anything and the test proves nothing"
    );
    assert!(h <= cap * 1.05, "the bounded fit grew a splat {h:.2}x against a cap of {cap}");
}

/// A fit may not cover a gap with a splat the size of the room.
///
/// Bounding a gaussian's axes by a constant in WORLD units is a bound on
/// nothing: what 0.3 means depends entirely on how big the scene is, and on a
/// capture whose camera orbit has radius 0.5 it permits a splat spanning a
/// fifth of the frame. The optimizer takes that offer, because inflating one
/// gaussian is the cheapest way to cover a region it cannot otherwise explain,
/// and the result is a scene that looks correct from the views it was fitted
/// to and like hair from every other.
///
/// The bound that means something is in pixels, through the cameras.
#[test]
fn a_fit_may_not_grow_a_splat_past_what_its_cameras_resolve() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (init, shots) = inflatable_scene(&g, ks);
    let cams: Vec<Camera> = shots.iter().map(|t| t.cam).collect();
    let biggest = |s: &Splats, limit_px: f32| -> f32 {
        // express every gaussian's longest axis in pixels, at the rate its own
        // cameras sampled it
        let unit = splat::mip::smoothing_sigma(s, &cams, 1.0, &|_, _| true);
        let mut worst = 0.0f32;
        for (i, &u) in unit.iter().enumerate() {
            if u <= 0.0 {
                continue;
            }
            let lng = s.scales[i * 3].max(s.scales[i * 3 + 1]).max(s.scales[i * 3 + 2]);
            worst = worst.max(lng / u);
        }
        let _ = limit_px;
        worst
    };

    // A deliberately tight cap, so the mechanism is exercised rather than
    // merely present. The shipped default is looser; what is under test is
    // that the bound is expressed through the cameras and actually binds.
    // `max_growth` is off in BOTH runs here. It is a second, independent
    // ceiling - relative to where each gaussian started rather than to what
    // the cameras resolve - and leaving it on would bound the control run too,
    // making this compare two bounded fits and prove nothing. Its own bound is
    // tested separately.
    // sizes step fast enough to take up the temptation in 150 iterations;
    // measured on the fitted scale itself, before the 3D filter is baked in
    let base = FitCfg {
        iters: 150, lr_position: 8e-3, lr_scale: 3e-2, log_every: 0, max_scale_pixels: 4.0, max_growth: 0.0,
        ..Default::default()
    };
    let bounded = splat::opt::fit_full(&g, ks, &init, &shots, &base, &mut |_, _| true).scene;
    let loose = splat::opt::fit_full(&g, ks, &init, &shots, &FitCfg { max_scale_pixels: 0.0, ..base }, &mut |_, _| true).scene;

    let b = biggest(&bounded, base.max_scale_pixels);
    let l = biggest(&loose, 0.0);
    // Measured against where the gaussians ENDED UP, while the bound was
    // computed from where they started, so a gaussian that drifted toward a
    // camera legitimately measures larger than its own limit. The headroom is
    // for that drift, not for the bound failing to bind.
    assert!(
        b <= base.max_scale_pixels * 1.75,
        "the bounded fit produced a splat {b:.1} px across against a limit of {}",
        base.max_scale_pixels
    );
    assert!(
        l > b * 2.0,
        "the unbounded fit reached {l:.1} px and the bounded one {b:.1} px. They are supposed to \
         differ a lot here - if they do not, this scene never tempted the optimizer to inflate \
         anything and the test proves nothing."
    );
}

