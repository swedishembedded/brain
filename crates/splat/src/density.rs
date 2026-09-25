// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Credit-assigned, budgeted density control ([`crate::opt::Densify::Hybrid`]).
//!
//! The heuristic asks one question of each gaussian - did its centre receive a
//! large gradient? - and that question has two known blind spots. A gaussian
//! big enough to straddle an edge is pulled both ways and its summed gradient
//! cancels (the "gradient collision" AbsGS, Ye et al. 2024, fixes by summing
//! magnitudes per pixel - which [`crate::renderer::SplatGrads::d_absgrad`]
//! already does). And the gradient says nothing about how much of the
//! remaining IMAGE ERROR a gaussian is responsible for, which is the question
//! density control is actually trying to answer.
//!
//! This controller answers it directly, by credit assignment. The rasterizer's
//! colour gradient is `Σ_p T_p,i α_p,i · dL/dC_p`, so a backward pass whose
//! upstream "gradient" is a per-pixel quantity returns, per gaussian, that
//! quantity summed by the gaussian's own compositing weights. One extra
//! backward per view with upstream `(1, r_p, e_p·r_p)` - `r` the pixel's
//! photometric residual, `e` the target's edge strength - yields for every
//! gaussian its contribution `U = Σ w`, its residual responsibility `Σ w r`
//! and its edge-weighted responsibility `Σ w e r`, with no new kernel.
//!
//! From that evidence, each round:
//!
//! 1. **Score** every gaussian that contributes meaningfully by the RANK of
//!    its mean residual, its mean edge residual and its per-view AbsGS
//!    gradient. Ranks, not thresholds: an absolute gradient threshold means
//!    something different at every image size, view count and scene, a
//!    percentile does not.
//! 2. **Budget** the round from a growth schedule `N_target(round)` that
//!    reaches `max_gaussians` about two thirds of the way through density
//!    control and then holds (Taming 3DGS, Mallick et al. 2024: population
//!    is a resource to schedule, not a side effect of thresholds).
//! 3. **Reclaim** what contributes nothing - but only on the SECOND round in a
//!    row it does so. The first time, it is suppressed (opacity halved) and
//!    marked; a gaussian that was only temporarily starved recovers and is
//!    unmarked, one that was genuinely redundant does not (recovery-aware
//!    pruning, after ImprovedGS, Deng et al. 2026). Reclaimed slots pay for
//!    new samples, the way 3DGS-MCMC relocation recycles dead ones.
//! 4. **Refine** the top-scored gaussians within the budget, by SHAPE: one
//!    larger than a pixel and elongated is split along its long axis into
//!    two children fitted to reproduce its composited alpha profile (the
//!    long-axis split of ImprovedGS); one larger than a pixel and roughly
//!    isotropic is too coarse in every direction and is subdivided in every
//!    direction (3DGS's split); a small one is cloned into two samples displaced in
//!    opposite directions by its own covariance, with the 3DGS-MCMC opacity
//!    correction so the pair renders as the one did.
//!
//! Swedish Embedded AB implements 3D reconstruction optimizers whose density
//! control spends a primitive budget where the image error is, not where a
//! threshold happens to fire. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::geometry::axis;
use crate::mcmc::{relocation, sh_stride};
use crate::opt::jitter;
use crate::types::Splats;

/// Per-gaussian evidence accumulated over a density-control window.
#[derive(Clone, Debug, Default)]
pub struct Evidence {
    /// `Σ_views Σ_pixels T α` - how much image each gaussian produced.
    pub contribution: Vec<f32>,
    /// `Σ T α r` - the photometric residual attributed to it.
    pub residual: Vec<f32>,
    /// `Σ T α e r` - the part of that residual that sits on an edge.
    pub edge: Vec<f32>,
    /// In how many views it contributed at all.
    pub views: Vec<u32>,
    /// AbsGS: summed per-pixel magnitudes of its screen-space position
    /// gradient.
    pub absgrad: Vec<f32>,
    /// Where the scene has no gaussian to refine for an error: pixels whose
    /// residual is large and whose scene is empty or at the wrong depth.
    pub sites: Vec<Site>,
    /// The share of the residual those pixels carry: of the samples a round
    /// adds beyond refinement, this share is spawned at [`Self::sites`].
    pub site_share: f32,
}

/// A place to spawn a gaussian: the surface point a residual pixel's range
/// puts under it, its footprint there, the photograph's colour and the
/// surface normal where a prior gives one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    pub pos: [f32; 3],
    /// The pixel's footprint at `pos`, world units.
    pub radius: f32,
    /// Unit surface normal, world frame.
    pub normal: Option<[f32; 3]>,
    pub rgb: [f32; 3],
    /// The pixel's residual: sites are drawn in proportion to it.
    pub weight: f32,
}

impl Evidence {
    pub fn new(n: usize) -> Evidence {
        Evidence {
            contribution: vec![0.0; n],
            residual: vec![0.0; n],
            edge: vec![0.0; n],
            views: vec![0; n],
            absgrad: vec![0.0; n],
            sites: Vec::new(),
            site_share: 0.0,
        }
    }

    /// Fold in one view's credit-assignment pass: the colour gradient of a
    /// backward run with upstream `(1, r, e·r)`, `[N*3]`.
    pub fn add_view(&mut self, credit: &[f32]) {
        for (i, c) in credit.chunks_exact(3).enumerate() {
            self.contribution[i] += c[0];
            self.residual[i] += c[1];
            self.edge[i] += c[2];
            if c[0] > 1e-6 {
                self.views[i] += 1;
            }
        }
    }
}

/// The per-pixel upstream of a credit-assignment pass, `[W*H*4]`:
/// `(m, m·r, m·e·r, 0)` with `r` the mean absolute residual over channels,
/// `e` the edge strength of the target and `m` the supervision weight.
pub fn credit_upstream(pred: &[f32], target: &[f32], edges: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    let px = edges.len();
    let mut out = vec![0.0f32; px * 4];
    for p in 0..px {
        let m = weights.map_or(1.0, |w| w[p]);
        let r = (0..3).map(|c| (pred[p * 3 + c] - target[p * 3 + c]).abs()).sum::<f32>() / 3.0;
        out[p * 4] = m;
        out[p * 4 + 1] = m * r;
        out[p * 4 + 2] = m * edges[p] * r;
    }
    out
}

/// Edge strength of an interleaved RGB image in [0,1]: the Sobel magnitude of
/// its luminance, normalized by its own 99th percentile so the measure is
/// independent of the photograph's contrast.
pub fn edge_map(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    let lum: Vec<f32> = rgb.chunks_exact(3).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
    let at = |x: isize, y: isize| lum[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let mut mag = vec![0.0f32; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let gx = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1);
            let gy = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1);
            mag[y as usize * w + x as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    let mut sorted = mag.clone();
    sorted.sort_by(f32::total_cmp);
    let top = sorted[((sorted.len() as f32 * 0.99) as usize).min(sorted.len() - 1)].max(1e-6);
    mag.iter().map(|v| (v / top).min(1.0)).collect()
}

/// Rank of each value among `vals[idx]`, as a fraction in [0,1].
fn percentile(vals: &[f32], idx: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; vals.len()];
    let mut order = idx.to_vec();
    order.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let d = (order.len().max(2) - 1) as f32;
    for (r, &i) in order.iter().enumerate() {
        out[i] = r as f32 / d;
    }
    out
}

/// How big the scene should be at density round `round` of `rounds`, starting
/// from `start`: a smooth ramp that reaches `cap` two thirds of the way in and
/// holds, so the last third of density control and everything after it
/// refine a population that has stopped changing size. With no cap, `grow`
/// (a fraction per round) is all there is to go on.
pub fn target_population(start: usize, cap: usize, round: usize, rounds: usize, n: usize, grow: f32) -> usize {
    if cap == 0 {
        return n + ((n as f32 * grow) as usize).max(1);
    }
    let full = ((rounds as f32 * 2.0 / 3.0).ceil() as usize).max(1);
    let t = ((round + 1) as f32 / full as f32).min(1.0);
    let s = t * t * (3.0 - 2.0 * t);
    (start as f32 + (cap as f32 - start.min(cap) as f32) * s) as usize
}

/// What one round did, for the log, and where every gaussian of the new
/// scene came from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Round {
    pub split: usize,
    pub cloned: usize,
    pub suppressed: usize,
    pub reclaimed: usize,
    /// Samples added at top-scored sites to fill the schedule.
    pub grown: usize,
    /// Gaussians spawned at residual pixels ([`Evidence::sites`]).
    pub spawned: usize,
    /// Per gaussian of the new scene, the gaussian of the old one whose
    /// optimizer state it continues - itself, or the parent of a split or a
    /// clone - and `None` for a new sample placed at a site, which starts
    /// its own. See `crate::opt`: discarding Adam's moments at every round
    /// throws away the momentum exactly when refinement should accelerate.
    pub origin: Vec<Option<usize>>,
}

/// Knobs of the hybrid controller - all fractions, none of them absolute
/// gradient or error thresholds.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// Most of the population refined in one round.
    pub refine_frac: f32,
    /// Opacity below which a gaussian is a reclaim candidate outright.
    pub dead_opacity: f32,
    /// Compositing weight, in pixels per view that sees it, below which a
    /// gaussian is a reclaim candidate: one that renders essentially nothing
    /// anywhere. An ABSOLUTE measure on purpose - relative to the median it
    /// condemns gaussians for being small (11k of 26k flagged in a round of
    /// a blob-dominated scene) - and a small one: once a scene has about as
    /// many gaussians as pixels, a quarter pixel per view is an ordinary
    /// contribution, and a 0.25 px threshold was measured suppressing 150k of
    /// 344k in one round and doubling the loss.
    pub starve_px: f32,
    /// Weights of the three ranked signals: residual, edge residual, AbsGS.
    pub weights: [f32; 3],
    /// Longest-to-middle axis ratio above which a gaussian is split along
    /// its long axis only; below it, every axis is subdivided.
    pub elongated: f32,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { refine_frac: 0.05, dead_opacity: 0.02, starve_px: 0.01, weights: [0.4, 0.3, 0.3], elongated: 1.5 }
    }
}

/// One density round. `size_px[i]` is gaussian `i`'s largest axis in pixels
/// as its best-sampling camera sees it; `target` is the population to aim
/// for; `suspect` carries the recovery marks across rounds (resized to the
/// new scene on return). Deterministic in `seed`.
pub fn round(
    scene: &mut Splats,
    ev: &Evidence,
    size_px: &[f32],
    suspect: &mut Vec<bool>,
    target: usize,
    policy: &Policy,
    seed: u64,
) -> Round {
    let n = scene.len();
    assert_eq!(ev.contribution.len(), n);
    assert_eq!(size_px.len(), n);
    suspect.resize(n, false);
    let mut stats = Round::default();

    let starving = |i: usize| ev.contribution[i] / (ev.views[i].max(1) as f32) < policy.starve_px;

    // 3. reclaim: second strike removes, first strike suppresses
    let mut remove = vec![false; n];
    for i in 0..n {
        let starved = starving(i) || scene.opacities[i] < policy.dead_opacity;
        if starved && suspect[i] {
            remove[i] = true;
            stats.reclaimed += 1;
        } else if starved {
            suspect[i] = true;
            scene.opacities[i] *= 0.5;
            stats.suppressed += 1;
        } else {
            suspect[i] = false;
        }
    }

    // 1. score by ranks among the gaussians with evidence worth ranking
    let eligible: Vec<usize> = (0..n).filter(|&i| !remove[i] && !suspect[i] && !starving(i)).collect();
    let mean = |num: &[f32]| -> Vec<f32> { (0..n).map(|i| num[i] / ev.contribution[i].max(1e-12)).collect() };
    let grad: Vec<f32> = (0..n).map(|i| ev.absgrad[i] / ev.views[i].max(1) as f32).collect();
    let (pr, pe, pg) = (percentile(&mean(&ev.residual), &eligible), percentile(&mean(&ev.edge), &eligible), percentile(&grad, &eligible));
    let [wr, we, wg] = policy.weights;
    let mut ranked = eligible.clone();
    let score = |i: usize| wr * pr[i] + we * pe[i] + wg * pg[i];
    ranked.sort_by(|&a, &b| score(b).total_cmp(&score(a)));

    // 2. budget. Refinement and growth both work near gaussians that exist,
    // which cannot reach an object the scene has nothing near; the share of
    // the residual that sits where the scene is empty or at the wrong depth
    // is reserved for spawning there instead.
    let alive = n - stats.reclaimed;
    let room = target.saturating_sub(alive);
    let spawn_n = ((room as f32 * ev.site_share.clamp(0.0, 1.0)).round() as usize).min(ev.sites.len());
    let room = room - spawn_n;
    let want = ((n as f32 * policy.refine_frac).ceil() as usize).min(room).min(ranked.len());
    let chosen: Vec<usize> = ranked[..want].to_vec();
    let mut refine = vec![false; n];
    for &i in &chosen {
        refine[i] = true;
    }

    // 4. rebuild
    let shk = sh_stride(scene);
    let mut out = Splats { sh_rest: scene.sh_rest.as_ref().map(|(d, _)| (*d, Vec::new())), ..Default::default() };
    let mut marks = Vec::with_capacity(n + want);
    let mut origin: Vec<Option<usize>> = Vec::with_capacity(n + want);
    // each new gaussian's claim on the samples that fill the schedule: its
    // (parent's) score, 0 for anything not ranked
    let mut weight: Vec<f32> = Vec::with_capacity(n + want);
    let mut ranked_flag = vec![false; n];
    for &i in &ranked {
        ranked_flag[i] = true;
    }
    let push = |o: &mut Splats, i: usize, mean: [f32; 3], scale: [f32; 3], opacity: f32| {
        o.means.extend_from_slice(&mean);
        o.quats.extend_from_slice(&scene.quats[i * 4..i * 4 + 4]);
        o.scales.extend_from_slice(&scale);
        o.opacities.push(opacity);
        o.colors.extend_from_slice(&scene.colors[i * 3..i * 3 + 3]);
        if let (Some((_, src)), Some((_, dst))) = (&scene.sh_rest, &mut o.sh_rest) {
            dst.extend_from_slice(&src[i * shk..i * shk + shk]);
        }
    };
    for i in 0..n {
        if remove[i] {
            continue;
        }
        let mu = [scene.means[i * 3], scene.means[i * 3 + 1], scene.means[i * 3 + 2]];
        let s = [scene.scales[i * 3], scene.scales[i * 3 + 1], scene.scales[i * 3 + 2]];
        let o = scene.opacities[i];
        let w = if ranked_flag[i] { score(i).max(1e-6) } else { 0.0 };
        if !refine[i] {
            push(&mut out, i, mu, s, o);
            marks.push(suspect[i]);
            weight.push(w);
            origin.push(Some(i));
            continue;
        }
        origin.push(Some(i));
        origin.push(Some(i));
        weight.push(w);
        weight.push(w);
        let q = unit(&scene.quats[i * 4..i * 4 + 4]);
        let mut sorted = s;
        sorted.sort_by(|a, b| b.total_cmp(a));
        let elongated = sorted[0] > policy.elongated * sorted[1];
        if size_px[i] > 1.0 && !elongated {
            // A blob, not a stick: it is too coarse in EVERY direction, so
            // halving one axis would leave its children as coarse as it was
            // in the other two. Subdivide it the 3DGS way instead - two
            // children drawn from its own density, every axis shrunk by 1.6.
            let e = [normal(i * 3, seed), normal(i * 3 + 1, seed), normal(i * 3 + 2, seed)];
            let mut d = [0.0f32; 3];
            for (k, ek) in e.iter().enumerate() {
                let a = axis(q, k);
                for c in 0..3 {
                    d[c] += a[c] * ek * s[k];
                }
            }
            let cs = s.map(|v| v / 1.6);
            push(&mut out, i, [mu[0] + d[0], mu[1] + d[1], mu[2] + d[2]], cs, o);
            push(&mut out, i, [mu[0] - d[0], mu[1] - d[1], mu[2] - d[2]], cs, o);
            stats.split += 1;
        } else if size_px[i] > 1.0 {
            // Long-axis split: two children at ±0.5σ along the longest axis,
            // that axis shrunk to 0.78σ and the opacity lowered to
            // 1-(1-o)^0.6. Compositing is not additive, so matching moments
            // does not preserve the image; these constants are the least-
            // squares fit of two composited children to the parent's alpha
            // profile along the axis, which they reproduce to 0.07-0.7%
            // relative error for opacities 0.3-0.95 (offsets of 0.75σ cannot
            // do better than 3-5%). Further apart is not better: the fit
            // separates children the image asks to be separated.
            let k = (0..3).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
            let a = axis(q, k);
            let off = 0.5 * s[k];
            let mut cs = s;
            cs[k] = 0.78 * s[k];
            let co = 1.0 - (1.0 - o).powf(0.6);
            push(&mut out, i, [mu[0] + a[0] * off, mu[1] + a[1] * off, mu[2] + a[2] * off], cs, co);
            push(&mut out, i, [mu[0] - a[0] * off, mu[1] - a[1] * off, mu[2] - a[2] * off], cs, co);
            stats.split += 1;
        } else {
            // Clone: two samples displaced in opposite directions by a draw
            // from the gaussian's own covariance, so the pair keeps the
            // parent's mean and the optimizer is not handed two identical
            // gaussians it can never tell apart.
            let (co, coeff) = relocation(o, 2);
            let e = [normal(i * 3, seed), normal(i * 3 + 1, seed), normal(i * 3 + 2, seed)];
            let mut d = [0.0f32; 3];
            for (k, ek) in e.iter().enumerate() {
                let a = axis(q, k);
                for c in 0..3 {
                    d[c] += 0.5 * a[c] * ek * s[k];
                }
            }
            let cs = [s[0] * coeff, s[1] * coeff, s[2] * coeff];
            push(&mut out, i, [mu[0] + d[0], mu[1] + d[1], mu[2] + d[2]], cs, co);
            push(&mut out, i, [mu[0] - d[0], mu[1] - d[1], mu[2] - d[2]], cs, co);
            stats.cloned += 1;
        }
        marks.push(false);
        marks.push(false);
    }
    // 5. fill whatever of the schedule refinement could not. Refinement
    // doubles at most the gaussians that were ranked, and in a scene still
    // made of large overlapping blobs most of the children of a 3D split land
    // behind the surface, are occluded in every view and are reclaimed the
    // next round - measured as a population that shrank for three rounds
    // while the schedule asked it to grow tenfold. So the rest of the room is
    // spent the way 3DGS-MCMC grows, but aimed by the credit score rather
    // than by opacity: extra samples at the top-scored sites, each displaced
    // by a draw from its site's own covariance, with the opacity and scale
    // correction that keeps a site rendering as it did.
    stats.spawned = spawn(&mut out, &ev.sites, spawn_n, seed ^ 0x7370_6177);
    let left = target.saturating_sub(out.len());
    if left > 0 {
        stats.grown = grow_at(&mut out, &weight, left, seed ^ 0x6f77_6e67);
    }
    marks.resize(out.len(), false);
    origin.resize(out.len(), None);
    *scene = out;
    *suspect = marks;
    stats.origin = origin;
    stats
}

/// Spawn `count` gaussians at `sites` drawn without replacement in
/// proportion to their weight. Each lies in the surface its site's normal
/// describes (thin along it, a pixel footprint across) or is a round pixel
/// footprint without one, carries the photograph's colour and starts at a
/// modest opacity the fit can raise or prune. Returns how many were added.
fn spawn(scene: &mut Splats, sites: &[Site], count: usize, seed: u64) -> usize {
    if count == 0 {
        return 0;
    }
    // weighted sampling without replacement: the largest u^(1/w)
    let mut keyed: Vec<(f32, usize)> = sites
        .iter()
        .enumerate()
        .filter(|(_, s)| s.weight > 0.0)
        .map(|(i, s)| (jitter(i, seed).max(1e-12).ln() / s.weight, i))
        .collect();
    let count = count.min(keyed.len());
    if count == 0 {
        return 0;
    }
    keyed.select_nth_unstable_by(count - 1, |a, b| b.0.total_cmp(&a.0));
    let shk = sh_stride(scene);
    for &(_, i) in &keyed[..count] {
        let s = &sites[i];
        let (q, scale) = match s.normal {
            Some(n) => (quat_with_z(n), [s.radius, s.radius, 0.1 * s.radius]),
            None => ([1.0, 0.0, 0.0, 0.0], [s.radius; 3]),
        };
        scene.means.extend_from_slice(&s.pos);
        scene.quats.extend_from_slice(&q);
        scene.scales.extend_from_slice(&scale);
        scene.opacities.push(0.3);
        scene.colors.extend_from_slice(&s.rgb);
        if let Some((_, r)) = &mut scene.sh_rest {
            r.extend(std::iter::repeat_n(0.0, shk));
        }
    }
    count
}

/// A unit quaternion `(w, x, y, z)` whose rotation takes +z to `n`.
fn quat_with_z(n: [f32; 3]) -> [f32; 4] {
    // half-way rotation from z to n
    let (x, y, z) = (n[0], n[1], n[2]);
    if z < -0.999_999 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let w = 1.0 + z;
    let q = [w, -y, x, 0.0];
    let l = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
    q.map(|v| v / l)
}

/// Add `count` samples to `scene` at sites drawn in proportion to `weight`,
/// correcting every touched site for its new multiplicity. Returns how many
/// were added.
fn grow_at(scene: &mut Splats, weight: &[f32], count: usize, seed: u64) -> usize {
    let n = scene.len();
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0f64;
    for &w in weight.iter().take(n) {
        acc += w.max(0.0) as f64;
        cdf.push(acc);
    }
    if acc <= 0.0 {
        return 0;
    }
    let mut copies = vec![0usize; n];
    for j in 0..count {
        let u = jitter(j, seed) as f64 * acc;
        copies[cdf.partition_point(|&c| c <= u).min(n - 1)] += 1;
    }
    let shk = sh_stride(scene);
    let mut added = 0;
    for (i, &m) in copies.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let (op, coeff) = relocation(scene.opacities[i], m + 1);
        let sc: [f32; 3] = std::array::from_fn(|k| (scene.scales[i * 3 + k] * coeff).max(1e-8));
        scene.opacities[i] = op;
        scene.scales[i * 3..i * 3 + 3].copy_from_slice(&sc);
        let q = unit(&scene.quats[i * 4..i * 4 + 4]);
        for c in 0..m {
            let e = [normal(i * 7 + c * 3, seed), normal(i * 7 + c * 3 + 1, seed), normal(i * 7 + c * 3 + 2, seed)];
            let mut d = [0.0f32; 3];
            for (k, ek) in e.iter().enumerate() {
                let a = axis(q, k);
                for x in 0..3 {
                    d[x] += 0.5 * a[x] * ek * sc[k];
                }
            }
            for (k, dk) in d.iter().enumerate() {
                let v = scene.means[i * 3 + k] + dk;
                scene.means.push(v);
            }
            scene.quats.extend_from_within(i * 4..i * 4 + 4);
            scene.scales.extend_from_slice(&sc);
            scene.opacities.push(op);
            scene.colors.extend_from_within(i * 3..i * 3 + 3);
            if let Some((_, r)) = &mut scene.sh_rest {
                r.extend_from_within(i * shk..i * shk + shk);
            }
            added += 1;
        }
    }
    added
}

fn unit(q: &[f32]) -> [f32; 4] {
    let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Standard normal from the scene's own indices (Box-Muller).
fn normal(i: usize, seed: u64) -> f32 {
    let u = jitter(i, seed).max(1e-7);
    let v = jitter(i, seed ^ 0x1234_5678_9abc_def0);
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}
