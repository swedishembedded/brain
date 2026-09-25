// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Density control as MCMC: "3D Gaussian Splatting as Markov Chain Monte
//! Carlo" (Kheradmand et al., NeurIPS 2024, arXiv:2404.09591).
//!
//! The classic heuristic decides where detail may appear by thresholding a
//! positional-gradient statistic and then splitting or cloning. Every part of
//! that is a knob: which statistic, which threshold, which shrink factor, and
//! none of them transfers between scenes - our own margin for it moved from
//! 10% to 8.7% when unrelated code changed, and the thresholds had to be
//! recalibrated rather than being derived from anything.
//!
//! MCMC replaces the decision with a MOVE. Gaussians are samples from a
//! distribution, dead ones are teleported to where live ones are, and the
//! teleport is made IMAGE-PRESERVING by correcting opacity and scale for the
//! fact that N gaussians now sit where one did. That correction is the whole
//! method: get it wrong and every relocation is a visible edit to the render
//! that the optimizer then has to undo, which is worse than not relocating.
//! So it is what these tests check, rather than checking that the count went
//! up.
//!
//! Swedish Embedded AB implements sampling-based 3D reconstruction optimizers,
//! including the image-preserving relocation that lets a fit redistribute its
//! gaussian budget instead of only adding to it. If your team needs
//! reconstruction quality that is not hostage to a hand-tuned threshold, you
//! can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::mcmc;
use splat::opt::{fit, Densify, FitCfg, TargetView};
use splat::quality::psnr;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

fn cam(w: u32, h: u32) -> Camera {
    Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, w, h)
}

fn render(g: &Gpu, ks: Kernels, s: &Splats, c: &Camera) -> Vec<f32> {
    // Big splats and no dilation: the correction is derived for the gaussian
    // itself, and `eps2d` adds a constant to the PROJECTED covariance that no
    // change of 3D scale can account for. Rendering a sub-pixel splat through
    // the dilation would measure the dilation, not the formula.
    let o = RenderOpts { bg: [0.15, 0.15, 0.2], eps2d: 0.0, ..Default::default() };
    let mut r = Renderer::new(g, ks, s.len().max(1), c.width, c.height, 0);
    let gs = GpuSplats::upload(g, s);
    r.render(g, &gs, c, &o);
    r.read_rgba(g, c.width, c.height).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
}

/// One gaussian, repeated `n` times at the same place, with the paper's
/// opacity and scale correction applied to all `n` of them.
fn copies(one: &Splats, n: usize, corrected: bool) -> Splats {
    let (op, coeff) = if corrected { mcmc::relocation(one.opacities[0], n) } else { (one.opacities[0], 1.0) };
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&one.means);
        s.quats.extend_from_slice(&one.quats);
        s.scales.extend(one.scales.iter().map(|v| v * coeff));
        s.opacities.push(op);
        s.colors.extend_from_slice(&one.colors);
    }
    s
}

fn one_gaussian(opacity: f32) -> Splats {
    Splats {
        means: vec![0.0, 0.0, 4.0],
        quats: vec![1.0, 0.0, 0.0, 0.0],
        scales: vec![0.30, 0.22, 0.26],
        opacities: vec![opacity],
        colors: vec![0.85, 0.55, 0.35],
        sh_rest: None,
    }
}

/// The load-bearing property: N gaussians placed where one was must RENDER as
/// the one did.
///
/// Relocation only buys anything if it is free at the moment it happens.
/// Without the correction the copies composite to `1-(1-o)^N` instead of `o`
/// and stack their tails, so the site goes darker and fatter - a step change
/// in the loss that the optimizer spends the next hundred iterations undoing.
/// Eq. 9 of the paper solves for the opacity and scale that keep the alpha
/// integral along a ray equal to what one gaussian gave.
#[test]
fn copies_of_a_gaussian_render_as_the_gaussian_did() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let c = cam(96, 96);

    for &o in &[0.35f32, 0.7, 0.95] {
        let one = one_gaussian(o);
        let base = render(&g, ks, &one, &c);
        for n in 2..=5 {
            let good = psnr(&render(&g, ks, &copies(&one, n, true), &c), &base);
            let naive = psnr(&render(&g, ks, &copies(&one, n, false), &c), &base);
            // Measured 43.7 dB (o=0.95, n=5) to 67.2 dB (o=0.35, n=2) with
            // both halves of Eq. 9 applied, and 38.8 dB for o=0.95 n=2 with
            // the opacity correction alone - so the bar is above what the
            // easy half of the formula can reach on its own.
            assert!(
                good > 42.0,
                "{n} corrected copies of an opacity-{o} gaussian render at {good:.1} dB against \
                 the single gaussian they replaced; relocation is supposed to be invisible at \
                 the moment it happens"
            );
            // and the correction has to be what achieved it - an uncorrected
            // clone of the same gaussian is a visible edit
            assert!(
                naive < good - 10.0,
                "{n} UNcorrected copies render at {naive:.1} dB and the corrected ones at \
                 {good:.1} dB. They are supposed to differ a lot here; if they do not, this \
                 scene does not exercise the correction and the test proves nothing"
            );
        }
    }
}

/// The same property through the actual move: a scene whose dead gaussians
/// have been teleported onto its live ones must render as it did before.
///
/// This is what `copies_of_a_gaussian_render_as_the_gaussian_did` cannot
/// catch: the correction has to be applied to the SOURCE as well as to the
/// gaussians that landed on it, and the multiplicity has to be the number
/// that ended up at that site rather than two.
#[test]
fn relocating_dead_gaussians_preserves_the_image() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let c = cam(96, 96);

    let mut s = Splats::default();
    for i in 0..48 {
        let a = i as f32 * 0.7;
        let live = i % 3 != 0;
        s.means.extend_from_slice(&[a.sin() * 0.9, a.cos() * 0.9, 4.0 + 0.2 * (i % 5) as f32]);
        s.quats.extend_from_slice(&[1.0, 0.1 * a.cos(), 0.1 * a.sin(), 0.0]);
        s.scales.extend_from_slice(&[0.16, 0.13, 0.15]);
        // a third of the scene is dead: below any sane threshold, contributing
        // nothing to the render, and pure waste until it is recycled
        s.opacities.push(if live { 0.45 + 0.4 * (i % 4) as f32 / 4.0 } else { 1e-4 });
        s.colors.extend_from_slice(&[0.2 + 0.7 * (i % 3) as f32 / 3.0, 0.6, 0.35]);
    }
    let before = render(&g, ks, &s, &c);
    let n = s.len();
    let orig = s.clone();
    let site = |s: &Splats, i: usize| [s.means[i * 3], s.means[i * 3 + 1], s.means[i * 3 + 2]];
    let dead: Vec<usize> = (0..n).filter(|&i| orig.opacities[i] < 0.02).collect();
    let live: Vec<usize> = (0..n).filter(|&i| orig.opacities[i] >= 0.02).collect();

    let moved = mcmc::relocate(&mut s, 0.02, 0x5eed);
    assert_eq!(moved.len(), dead.len(), "every dead gaussian must be recycled");
    assert_eq!(s.len(), n, "relocation MOVES gaussians, it does not add or drop any");

    let after = render(&g, ks, &s, &c);
    let d = psnr(&after, &before);
    assert!(
        d > 45.0,
        "relocating {} dead gaussians changed the render: {d:.1} dB. The opacity and scale \
         correction exists precisely so this is a no-op at the moment it happens",
        moved.len()
    );

    // The control, built from where the move actually put things: the same
    // teleport with the parameters copied verbatim, which is what a clone is.
    let mut naive = orig.clone();
    for &i in &dead {
        let j = live
            .iter()
            .copied()
            .find(|&j| site(&s, i) == site(&orig, j))
            .expect("relocated onto a live site");
        for k in 0..3 {
            naive.means[i * 3 + k] = orig.means[j * 3 + k];
            naive.colors[i * 3 + k] = orig.colors[j * 3 + k];
            naive.scales[i * 3 + k] = orig.scales[j * 3 + k];
        }
        for k in 0..4 {
            naive.quats[i * 4 + k] = orig.quats[j * 4 + k];
        }
        naive.opacities[i] = orig.opacities[j];
    }
    let n_db = psnr(&render(&g, ks, &naive, &c), &before);
    assert!(
        n_db < d - 10.0,
        "the same move without the correction renders at {n_db:.1} dB against the corrected \
         {d:.1} dB. They are supposed to differ a lot here; if they do not, this scene does not \
         exercise the correction and the test proves nothing"
    );

    // and the dead ones really went somewhere: each now sits exactly on a
    // gaussian that was alive, rather than merely having been made opaque
    for &i in &dead {
        assert!(
            live.iter().any(|&j| site(&s, i) == site(&orig, j)),
            "dead gaussian {i} is at {:?}, which is not any live gaussian's site",
            site(&s, i)
        );
        assert!(s.opacities[i] > 0.02, "a relocated gaussian must come back alive");
    }
}

/// Relocation must prefer the gaussians that are carrying the image.
///
/// The target is drawn with probability proportional to opacity, which is the
/// paper's proposal distribution: a site that is already explaining something
/// is where another sample is most likely to be useful, and a site that is
/// nearly transparent is about to be recycled itself.
#[test]
fn relocation_targets_are_drawn_by_opacity() {
    let mut s = Splats::default();
    // two live gaussians, one nine times as opaque as the other, and a crowd
    // of dead ones to be dealt among them
    for (i, o) in [0.9f32, 0.1].into_iter().enumerate() {
        s.means.extend_from_slice(&[i as f32, 0.0, 4.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.1, 0.1, 0.1]);
        s.opacities.push(o);
        s.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
    }
    for _ in 0..400 {
        s.means.extend_from_slice(&[9.0, 9.0, 9.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.1, 0.1, 0.1]);
        s.opacities.push(1e-4);
        s.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
    }
    let reference = s.clone();
    mcmc::relocate(&mut s, 0.02, 0x1234);
    let at_first = (0..s.len()).filter(|&i| s.means[i * 3] < 0.5).count();
    let at_second = s.len() - at_first;
    assert!(
        at_first > at_second * 3,
        "the opacity-0.9 site took {at_first} of the relocated gaussians and the opacity-0.1 \
         site {at_second}; the draw is supposed to be weighted by opacity"
    );

    // and it must be reproducible: a fit has to give the same answer twice
    let mut again = reference;
    mcmc::relocate(&mut again, 0.02, 0x1234);
    assert_eq!(again.means, s.means, "relocation is not deterministic");
}

// ---------------------------------------------------------------------------
// The fit-level claim: same budget, lower error.
// ---------------------------------------------------------------------------

/// A checkerboard slab of `cells`x`cells` gaussians over a fixed area - the
/// same scene sampled coarsely or finely (as in `s8_densification`).
fn board(cells: usize, flat: bool) -> Splats {
    let mut s = Splats::default();
    let step = 2.4 / cells as f32;
    for iy in 0..cells {
        for ix in 0..cells {
            s.means.extend_from_slice(&[-1.2 + step * (ix as f32 + 0.5), -1.2 + step * (iy as f32 + 0.5), 3.0]);
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

/// A scene that already holds its gaussians and cannot use most of them: a
/// coarse live surface plus a crowd sitting BEHIND the cameras, where a bad
/// depth prediction routinely puts them.
///
/// Being behind the near plane is what makes them wasted rather than merely
/// faint. A transparent gaussian that is still in frame is not wasted at all -
/// opacity is a free parameter and the optimizer simply turns it back up, as
/// it does here for every one of them that is in view. One that is not
/// rendered gets no gradient of any kind, so nothing short of moving it
/// bodily will ever bring it back, and moving it bodily is the one thing the
/// heuristic cannot do.
fn wasteful(live_cells: usize, dead: usize) -> Splats {
    let mut s = board(live_cells, true);
    for i in 0..dead {
        let a = i as f32 * 0.9;
        s.means.extend_from_slice(&[a.sin() * 0.5, a.cos() * 0.5, -0.4 - 0.1 * (i % 5) as f32]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.06, 0.06, 0.06]);
        s.opacities.push(0.01);
        s.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
    }
    s
}

/// Given the SAME gaussian budget, MCMC must reconstruct better.
///
/// The heuristic has exactly two things it can do with a gaussian it cannot
/// use: leave it, or delete it. Deleting it hands the budget back to a split
/// rule that can only subdivide what is already there, so the scene has to
/// climb back to its budget two children at a time from whatever survived.
/// Relocation is the move it does not have: the sample is put where the
/// density already is, at no cost to the render, and the whole wasted third
/// of the scene is back in service in one round.
///
/// Measured at 64x64 over three views, 150 iterations, a 400-gaussian budget,
/// starting from 36 live gaussians and 220 behind the cameras:
///
/// ```text
///   no density control at all                mse 0.010884   256 gaussians
///   density control that only stages the run  mse 0.011737   256 gaussians
///   heuristic, densify_frac 0.05 (default)   mse 0.010820    55 gaussians
///   heuristic, densify_frac 0.30             mse 0.008462   158 gaussians
///   heuristic, densify_frac 0.60             mse 0.007954   297 gaussians
///   heuristic, densify_frac 1.00 and above   mse 0.007801   400 gaussians
///   MCMC                                     mse 0.004440   400 gaussians
/// ```
///
/// The comparison below is against the fourth row - the heuristic wound up
/// until it saturates the budget and stops improving - rather than against
/// its default, because beating the default would only be beating a growth
/// rate. The rate is the heuristic's own problem here (at its default it
/// reaches 55 of the 400 it was allowed) but it is not the interesting one.
#[test]
fn mcmc_reconstructs_better_than_the_heuristic_at_the_same_budget() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (64u32, 64u32);
    let truth = board(16, false);
    let t = targets(&g, ks, &truth, w, h);
    let init = wasteful(6, 220);
    const BUDGET: usize = 400;

    let base = FitCfg {
        iters: 150,
        lr_position: 1e-2,
        log_every: 0,
        densify_every: 25,
        densify_after: 25,
        densify_until: 105,
        // wound up until the heuristic saturates its budget: 0.6 gets it to
        // 297 gaussians and anything from 1.0 up gives the same 400 and the
        // same loss, so this is the best the split rule has to offer here
        densify_frac: 1.0,
        max_gaussians: BUDGET,
        // `max_growth` off in BOTH arms. It bounds a gaussian against the size
        // it had at the start of the stage, and MCMC relocation deliberately
        // SHRINKS what it moves - the paper's opacity and scale correction is
        // what keeps a relocation from changing the rendered image. Leaving it
        // on caps how fast a relocated gaussian can take up its new place and
        // penalises one strategy for doing its job. What is under test here is
        // where gaussians go, not how large they may be.
        max_growth: 0.0,
        ..Default::default()
    };
    let (a, mse_heur) = fit(&g, ks, &init, &t, &base, &mut |_, _| true);
    let mcmc_cfg = FitCfg { strategy: Densify::Mcmc, noise: 1.0, ..base };
    let (b, mse_mcmc) = fit(&g, ks, &init, &t, &mcmc_cfg, &mut |_, _| true);

    assert!(a.len() <= BUDGET && b.len() <= BUDGET, "budget broken: {} and {}", a.len(), b.len());
    assert!(
        b.len() * 10 >= a.len() * 9,
        "MCMC spent only {} of the {BUDGET} gaussians against the heuristic's {}, so this is not \
         an equal-budget comparison",
        b.len(), a.len()
    );
    // Measured at 0.006626 against 0.006916, about 4% at an equal budget.
    //
    // It was 43% when this was written, against a fit with no bound on how
    // FLAT a gaussian could become and none on how far it could be INFLATED.
    // Both landed since and they independently remove part of what relocation
    // was fixing: a heuristic that can no longer answer a badly placed
    // gaussian by stretching it into a blade is a much stronger baseline. The
    // remaining margin is real but it is not a difference in kind, and the
    // bound demanded says so rather than preserving a number measured against
    // a weaker control.
    assert!(
        mse_mcmc < mse_heur * 0.98,
        "MCMC finished at mse {mse_mcmc:.6} with {} gaussians against the heuristic's \
         {mse_heur:.6} with {}",
        b.len(), a.len()
    );

    // and it is visible in the frame, not only in the loss
    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, a.len().max(b.len()), w, h, 0);
    let mut shot = |s: &Splats| -> Vec<f32> {
        let gs = GpuSplats::upload(&g, s);
        ren.render(&g, &gs, &t[0].cam, &o);
        ren.read_rgba(&g, w, h).chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
    };
    let (da, db) = (psnr(&shot(&a), &t[0].rgb), psnr(&shot(&b), &t[0].rgb));
    // 22.4 against 22.1 dB, small for the same reason the loss margin is.
    assert!(db > da + 0.2, "MCMC renders at {db:.1} dB against the heuristic's {da:.1} dB");
}

/// Whatever moves or copies a gaussian has to bring its higher-order colour
/// with it.
///
/// `sh_rest` lives outside the per-gaussian arrays, so the density-control
/// pass rebuilt the scene without it: a fit with `sh_degree` above 0 and
/// `densify_every` set silently dropped every harmonic at the first round and
/// carried on from flat colour, having thrown away the only thing that can
/// represent a surface that looks different from different sides. Both
/// strategies move gaussians, so both have to carry it.
#[test]
fn density_control_carries_view_dependent_colour() {
    const K: usize = 9; // degree 1: 3 coefficients x 3 channels
    let mut base = Splats::default();
    let mut sh = Vec::new();
    for i in 0..40 {
        let a = i as f32 * 0.6;
        base.means.extend_from_slice(&[a.sin(), a.cos(), 4.0]);
        base.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        base.scales.extend_from_slice(&[0.1, 0.1, 0.1]);
        base.opacities.push(if i % 4 == 0 { 1e-4 } else { 0.8 });
        base.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
        // a value unique to this gaussian, so a block can be traced back to
        // the gaussian it came from
        sh.extend(std::iter::repeat_n(1.0 + i as f32, K));
    }
    base.sh_rest = Some((1, sh));

    let check = |s: &Splats, what: &str| {
        let (deg, r) = s.sh_rest.as_ref().unwrap_or_else(|| panic!("{what} dropped sh_rest"));
        assert_eq!(*deg, 1);
        assert_eq!(r.len(), s.len() * K, "{what} left sh_rest the wrong length");
        for i in 0..s.len() {
            let b = &r[i * K..i * K + K];
            assert!(b.iter().all(|v| *v == b[0]), "{what} spliced two gaussians' harmonics");
            assert!(
                (1.0..=40.0).contains(&b[0]),
                "{what} gave gaussian {i} harmonics {} that belong to no gaussian in the scene",
                b[0]
            );
        }
    };

    let mut heuristic = base.clone();
    let grad: Vec<f32> = (0..base.len()).map(|i| i as f32).collect();
    let cfg = FitCfg { densify_every: 1, densify_frac: 0.3, ..Default::default() };
    splat::opt::densify_for_test(&mut heuristic, &grad, &cfg);
    // measured against what it KEPT: the heuristic prunes the transparent
    // quarter of this scene, so the count it grows from is the live one
    let alive = base.opacities.iter().filter(|o| **o >= cfg.prune_opacity).count();
    assert!(heuristic.len() > alive, "the heuristic added nothing, so this proves nothing");
    check(&heuristic, "the heuristic");

    let mut chain = base.clone();
    mcmc::relocate(&mut chain, 0.02, 0x5eed);
    mcmc::grow(&mut chain, base.len() + 20, 0x5eed);
    assert!(chain.len() > base.len());
    check(&chain, "MCMC");
}

/// The default must not change: MCMC is opt-in, and a `FitCfg` that does not
/// ask for it gets the density control it always got.
#[test]
fn the_heuristic_is_still_the_default() {
    assert!(matches!(FitCfg::default().strategy, Densify::Heuristic));
}
