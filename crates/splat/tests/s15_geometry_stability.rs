// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T15 gate: a fit must refine appearance without relocating geometry.
//!
//! A feed-forward reconstruction hands the optimizer a scene whose geometry is
//! already metric - every gaussian sits on the surface it was unprojected
//! from. What remains wrong is appearance. So the fit's job here is the
//! opposite of the one the original 3DGS optimizer was designed for, which
//! starts from a sparse point cloud that HAS to migrate a long way.
//!
//! The failure this gate exists to catch is a single learning rate shared by
//! parameter groups whose natural magnitudes differ by orders of magnitude.
//! `p_geo` packs a position in world units, a LINEAR scale, and a unit
//! quaternion into one buffer; Adam's step is ~lr regardless of gradient, so
//! one step is a rounding error on the scene diagonal, a sixth of a typical
//! gaussian's own radius, and nothing at all on a quaternion. Measured on a
//! real capture that drove 94% of gaussians more than three radii off their
//! surface and pinned a quarter of the scene against the growth clamp, which
//! reads as stray gaussians and lost sharpness however good the initial
//! geometry was.
//!
//! Swedish Embedded AB implements differentiable renderers and the numerical
//! gates that keep them stable. If your team needs expertise in 3D
//! reconstruction or GPU optimization then you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::{fit, FitCfg, TargetView};
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

/// A shell of small gaussians on a sphere, the shape a real capture produces.
///
/// Small RELATIVE to the scene is the whole point: the defect only shows up
/// when a gaussian's own radius is orders of magnitude below the scene extent,
/// because that is the ratio a single learning rate cannot serve at both ends.
fn shell(n: usize, seed: u64) -> Splats {
    let mut r = Lcg(seed);
    let mut s = Splats::default();
    for _ in 0..n {
        let (u, v) = (r.next() * std::f32::consts::TAU, r.next() * 2.0 - 1.0);
        let rad = (1.0f32 - v * v).sqrt();
        let p = [rad * u.cos(), rad * u.sin(), v];
        s.means.extend_from_slice(&[p[0], p[1], p[2] + 4.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.02, 0.02, 0.02]);
        s.opacities.push(0.6 + 0.3 * r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    s
}

fn views(w: u32, h: u32) -> Vec<Camera> {
    (0..6)
        .map(|i| {
            let a = i as f32 / 6.0 * std::f32::consts::TAU;
            Camera::look_at(
                [3.0 * a.sin(), 1.2 * (a * 0.5).cos(), 4.0 - 3.0 * a.cos()],
                [0.0, 0.0, 4.0],
                [0.0, -1.0, 0.0],
                60.0,
                w,
                h,
            )
        })
        .collect()
}

fn targets(g: &Gpu, ks: Kernels, s: &Splats, cams: &[Camera]) -> Vec<TargetView> {
    let o = RenderOpts::default();
    cams.iter()
        .map(|c| {
            let mut ren = Renderer::new(g, ks, s.len(), c.width, c.height, 0);
            let gs = GpuSplats::upload(g, s);
            ren.render(g, &gs, c, &o);
            let rgba = ren.read_rgba(g, c.width, c.height);
            TargetView::new(*c, rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect())
        })
        .collect()
}

fn radii(s: &Splats) -> Vec<f32> {
    (0..s.len()).map(|i| s.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max)).collect()
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn pct(v: &mut [f32], q: f32) -> f32 {
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * q / 100.0) as usize]
}

/// The gate. Geometry is already correct; only colour is wrong.
///
/// The fit must fix the colour and leave the geometry where it found it. Both
/// halves are asserted, because a fit that simply refuses to move satisfies
/// the stability half trivially and is no use.
#[test]
fn fit_refines_colour_without_relocating_geometry() {
    let g = Gpu::new(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = shell(4000, 7);
    let cams = views(160, 120);
    let tgts = targets(&g, ks, &truth, &cams);

    // Same geometry, wrong colours: exactly the feed-forward situation.
    let mut init = truth.clone();
    let mut r = Lcg(99);
    for c in init.colors.iter_mut() {
        *c = (*c + (r.next() - 0.5) * 0.6).clamp(0.0, 1.0);
    }

    // `lr` is the APPEARANCE rate now that the geometry groups are budgeted
    // against their own radius, and the budget divides by `lr` - so raising
    // this cannot move a gaussian any further than it already could. The rate
    // that used to be forced on every group by the slowest-moving one is
    // exactly what this decoupling buys back.
    // The budgets are what a caller with metric geometry declares, and they
    // are off by default because a fit from a sparse cloud needs the opposite.
    // `lr` is the APPEARANCE rate once they are set: the budget divides by it,
    // so raising this cannot move a gaussian any further than it already
    // could. The single rate that every group used to share was held down to
    // whatever the most fragile of them could survive, and that is exactly
    // what this decoupling buys back.
    let cfg = FitCfg {
        iters: 300,
        lr: 5e-3,
        log_every: 0,
        position_budget: 1.0,
        scale_budget: 0.5,
        rotation_budget: 0.5,
        ..Default::default()
    };
    let (out, _) = fit(&g, ks, &init, &tgts, &cfg, &mut |_, _| true);
    assert_eq!(out.len(), init.len(), "density control must be off for a paired comparison");

    let err = |s: &Splats| {
        s.colors.iter().zip(&truth.colors).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / s.colors.len() as f32
    };
    let (before, after) = (err(&init), err(&out));

    let r0 = radii(&init);
    let mut drift: Vec<f32> = (0..out.len())
        .map(|i| {
            let d: f32 = (0..3)
                .map(|k| (out.means[i * 3 + k] - init.means[i * 3 + k]).powi(2))
                .sum::<f32>()
                .sqrt();
            d / r0[i]
        })
        .collect();
    let med = median(&mut drift.clone());
    let p99 = pct(&mut drift, 99.0);

    let r1 = radii(&out);
    let mut ratio: Vec<f32> = r1.iter().zip(&r0).map(|(a, b)| a / b).collect();
    let p95 = pct(&mut ratio, 95.0);
    let mut inv: Vec<f32> = ratio.iter().map(|v| 1.0 / v).collect();
    let shrink = pct(&mut inv, 95.0);
    println!(
        "colour {before:.4} -> {after:.4} | drift med {med:.2}r p99 {p99:.2}r | grow p95 {p95:.2}x shrink p95 {shrink:.2}x"
    );
    assert!(after < before * 0.75, "fit must actually fix colour: {before:.4} -> {after:.4}");
    assert!(med < 1.0, "median gaussian drifted {med:.2} of its own radius (must stay on surface)");
    assert!(p99 < 4.0, "p99 gaussian drifted {p99:.2} of its own radius");
    assert!(p95 < 1.5, "p95 gaussian grew {p95:.2}x on a scene that was already correct");
    assert!(shrink < 1.5, "p95 gaussian shrank {shrink:.2}x on a scene that was already correct");
}

/// The same budget on the CPU backend, and proof that it is the descriptor
/// doing the work.
///
/// `adamw.wgsl` reads the per-component multiplier at `desc[3 + idx % period]`,
/// a DYNAMIC subscript, which is the kind of indexing a JIT backend can get
/// wrong on its own. Running the budgeted and unbudgeted fits side by side
/// asserts the two are different functions here as well, so a backend that
/// quietly ignored the tail of the descriptor could not pass.
#[test]
fn the_position_budget_binds_on_the_cpu_backend() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let truth = shell(300, 3);
    let cams = views(48, 48)[..2].to_vec();
    let tgts = targets(&g, ks, &truth, &cams);

    let mut init = truth.clone();
    let mut r = Lcg(5);
    for c in init.colors.iter_mut() {
        *c = (*c + (r.next() - 0.5) * 0.6).clamp(0.0, 1.0);
    }

    let free = FitCfg { iters: 60, lr: 5e-3, log_every: 0, ..Default::default() };
    let budgeted = FitCfg { position_budget: 1.0, scale_budget: 0.5, rotation_budget: 0.5, ..free };
    let drift = |out: &Splats| {
        let mut d: Vec<f32> = (0..out.len())
            .map(|i| {
                (0..3)
                    .map(|k| (out.means[i * 3 + k] - init.means[i * 3 + k]).powi(2))
                    .sum::<f32>()
                    .sqrt()
                    / init.scales[i * 3..i * 3 + 3].iter().copied().fold(0.0f32, f32::max)
            })
            .collect();
        median(&mut d)
    };
    let (a, _) = fit(&g, ks, &init, &tgts, &free, &mut |_, _| true);
    let (b, _) = fit(&g, ks, &init, &tgts, &budgeted, &mut |_, _| true);
    let (da, db) = (drift(&a), drift(&b));
    println!("cpu drift: unbudgeted {da:.3}r, budgeted {db:.3}r");
    assert!(
        db < da * 0.5,
        "the per-component descriptor did not bind on this backend: {db:.3}r budgeted against \
         {da:.3}r free"
    );
    assert!(db < 1.0, "budgeted drift {db:.3}r exceeds the one radius asked for");
}
