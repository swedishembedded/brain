// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Density control as Markov Chain Monte Carlo: "3D Gaussian Splatting as
//! Markov Chain Monte Carlo", Kheradmand et al., NeurIPS 2024
//! (arXiv:2404.09591).
//!
//! The classic heuristic ([`crate::opt::densify`]) decides where detail may
//! appear by thresholding a positional-gradient statistic and then splitting
//! or cloning. Every part of that is a knob - which statistic, which
//! threshold, which shrink factor - and none of them is derived from the model
//! being fitted, so they have to be recalibrated per scene and drift whenever
//! anything else changes.
//!
//! MCMC drops the decision. The scene is a set of SAMPLES from a distribution
//! over gaussians, and the only moves are: relocate a sample that has gone
//! transparent to where an opaque one already is, and perturb positions with
//! noise proportional to the learning rate, which turns the optimizer into
//! SGLD-style sampling rather than descent. Nothing is ever split on a
//! threshold and nothing is deleted.
//!
//! What makes the relocation legitimate is that it is IMAGE-PRESERVING. Moving
//! a gaussian is only free if the render does not change at the moment it
//! happens, which is not true of a naive teleport: N gaussians stacked at one
//! site composite to `1-(1-o)^N` and pile up their tails. [`relocation`] is the
//! paper's correction (its Eq. 9) for exactly that, and it is the load-bearing
//! piece of the method - see `tests/s13_mcmc.rs`.
//!
//! Swedish Embedded AB implements sampling-based 3D reconstruction, including
//! the image-preserving relocation that lets a fit redistribute a fixed
//! gaussian budget instead of only growing it. If your team needs
//! reconstruction quality that is not hostage to a hand-tuned threshold, you
//! can procure our services by sending an email to info@swedishembedded.com.

use crate::opt::{jitter, FitCfg};
use crate::types::Splats;

/// Largest multiplicity the correction is evaluated at, as in the reference
/// implementation. The correction is an alternating binomial sum; past a few
/// dozen samples at one site it is both numerically thin and pointless, since
/// a site that busy should be spreading rather than stacking.
const N_MAX: usize = 51;

/// The paper's Eq. 9: what opacity and what scale `n` gaussians need in order
/// to render as the ONE gaussian of opacity `opacity` they are replacing.
/// Returns `(opacity, scale_multiplier)`, both to be applied to all `n`.
///
/// Along a ray through the shared centre the old gaussian contributes
/// `o·exp(-t²/2σ²)`, whose integral is `o·σ·sqrt(2π)`. The `n` new ones
/// composite to `1 - (1 - o'·exp(-t²/2σ'²))^n`, whose integral is
/// `σ'·sqrt(2π)·Σ_k C(n,k)(-1)^{k+1} o'^k / sqrt(k)`. Equating the two, with
/// `o'` fixed by requiring the peak alpha to match (`1-(1-o')^n = o`), gives
/// the multiplier below. The paper states it as a factor on the COVARIANCE,
/// `Σ' = (o/denom)² Σ`, which is the same thing: these scales are linear, and
/// scaling them by `c` scales the covariance by `c²`.
///
/// `n = 1` is the identity, which is what makes this safe to apply to every
/// site rather than only to the ones that gained a sample.
pub fn relocation(opacity: f32, n: usize) -> (f32, f32) {
    let n = n.clamp(1, N_MAX);
    if n == 1 {
        return (opacity, 1.0);
    }
    let o = f64::from(opacity.clamp(1e-6, 1.0 - 1e-6));
    let o_new = 1.0 - (1.0 - o).powf(1.0 / n as f64);
    // Σ_k C(n,k) (-1)^{k+1} o'^k / sqrt(k), accumulated with the binomial
    // carried forward rather than tabulated. The terms behave like
    // (n·o')^k / k! and n·o' is bounded by -ln(1-o), so the alternating sum
    // does not lose its leading digits for any opacity a fit can hold.
    let mut denom = 0.0f64;
    let mut binom = 1.0f64;
    let mut pow = 1.0f64;
    for k in 1..=n {
        binom = binom * (n - k + 1) as f64 / k as f64;
        pow *= o_new;
        let term = binom * pow / (k as f64).sqrt();
        denom += if k % 2 == 1 { term } else { -term };
    }
    let coeff = if denom > 1e-12 { o / denom } else { 1.0 };
    (o_new as f32, coeff as f32)
}

/// Draw `k` indices from `pool` with probability proportional to opacity.
///
/// Deterministic in `(seed, draw index)`: a fit has to give the same answer
/// twice, so the chain's randomness is a hash of where it is rather than
/// wall-clock entropy.
fn sample_by_opacity(pool: &[usize], opacities: &[f32], k: usize, seed: u64) -> Vec<usize> {
    let mut cdf = Vec::with_capacity(pool.len());
    let mut acc = 0.0f64;
    for &i in pool {
        acc += f64::from(opacities[i].max(0.0));
        cdf.push(acc);
    }
    if acc <= 0.0 {
        // every candidate is transparent: fall back to a uniform draw rather
        // than to index 0, which would stack the whole budget on one site
        return (0..k).map(|j| pool[(jitter(j, seed) * pool.len() as f32) as usize % pool.len()]).collect();
    }
    (0..k)
        .map(|j| {
            let u = f64::from(jitter(j, seed)) * acc;
            let at = cdf.partition_point(|&c| c <= u).min(pool.len() - 1);
            pool[at]
        })
        .collect()
}

/// How many higher-order SH coefficients each gaussian carries, 0 if the
/// scene has none. Every move that copies a gaussian has to bring these with
/// it or the scene quietly reverts to flat colour.
pub(crate) fn sh_stride(scene: &Splats) -> usize {
    match &scene.sh_rest {
        Some((_, r)) if !scene.is_empty() && r.len().is_multiple_of(scene.len()) => r.len() / scene.len(),
        _ => 0,
    }
}

/// Put `dst[j]` at the site of `src[j]`, correcting every site for how many
/// gaussians ended up on it.
///
/// The correction has to cover the SOURCE too, not only what landed on it -
/// there are now `1 + count` gaussians there, and the one that was already
/// there is one of them.
fn place(scene: &mut Splats, src: &[usize], dst: &[usize]) {
    debug_assert_eq!(src.len(), dst.len());
    let mut mult = std::collections::HashMap::<usize, usize>::new();
    for &s in src {
        *mult.entry(s).or_insert(1) += 1;
    }
    // Corrected parameters per source, computed from the ORIGINAL values
    // before anything is written back, so a source that is also read by its
    // own copies is not corrected twice.
    let fixed: std::collections::HashMap<usize, (f32, [f32; 3])> = mult
        .iter()
        .map(|(&s, &n)| {
            let (op, coeff) = relocation(scene.opacities[s], n);
            let sc = std::array::from_fn(|k| (scene.scales[s * 3 + k] * coeff).max(1e-8));
            (s, (op, sc))
        })
        .collect();
    let shk = sh_stride(scene);
    for (j, &d) in dst.iter().enumerate() {
        let s = src[j];
        for k in 0..3 {
            scene.means[d * 3 + k] = scene.means[s * 3 + k];
            scene.colors[d * 3 + k] = scene.colors[s * 3 + k];
        }
        for k in 0..4 {
            scene.quats[d * 4 + k] = scene.quats[s * 4 + k];
        }
        if let Some((_, r)) = &mut scene.sh_rest {
            r.copy_within(s * shk..s * shk + shk, d * shk);
        }
        let (op, sc) = fixed[&s];
        scene.opacities[d] = op;
        scene.scales[d * 3..d * 3 + 3].copy_from_slice(&sc);
    }
    for (&s, &(op, sc)) in &fixed {
        scene.opacities[s] = op;
        scene.scales[s * 3..s * 3 + 3].copy_from_slice(&sc);
    }
}

/// Teleport every gaussian whose opacity is below `dead_below` onto a live one
/// drawn by opacity, image-preservingly. Returns the slots that moved: each
/// is now a new sample and starts its own optimizer state.
///
/// This is the move that the heuristic has no equivalent of. Pruning a dead
/// gaussian gives its budget back to nobody; relocating it puts the sample
/// where the density is, which is the only reason a FIXED budget can keep
/// improving.
pub fn relocate(scene: &mut Splats, dead_below: f32, seed: u64) -> Vec<usize> {
    let n = scene.len();
    let dead: Vec<usize> = (0..n).filter(|&i| scene.opacities[i] < dead_below).collect();
    let live: Vec<usize> = (0..n).filter(|&i| scene.opacities[i] >= dead_below).collect();
    if dead.is_empty() || live.is_empty() {
        return Vec::new();
    }
    let src = sample_by_opacity(&live, &scene.opacities, dead.len(), seed);
    place(scene, &src, &dead);
    dead
}

/// Grow the scene to `target` gaussians by duplicating sites drawn by opacity,
/// image-preservingly. Returns how many were added.
pub fn grow(scene: &mut Splats, target: usize, seed: u64) -> usize {
    let n = scene.len();
    if n == 0 || target <= n {
        return 0;
    }
    let add = target - n;
    let all: Vec<usize> = (0..n).collect();
    let src = sample_by_opacity(&all, &scene.opacities, add, seed);
    let shk = sh_stride(scene);
    for &s in &src {
        scene.means.extend_from_within(s * 3..s * 3 + 3);
        scene.quats.extend_from_within(s * 4..s * 4 + 4);
        scene.scales.extend_from_within(s * 3..s * 3 + 3);
        scene.colors.extend_from_within(s * 3..s * 3 + 3);
        scene.opacities.push(scene.opacities[s]);
        if let Some((_, r)) = &mut scene.sh_rest {
            r.extend_from_within(s * shk..s * shk + shk);
        }
    }
    let dst: Vec<usize> = (n..n + add).collect();
    place(scene, &src, &dst);
    add
}

/// One MCMC density-control step, `round` of `rounds`: recycle the dead, then
/// spend whatever budget is left. Returns `(relocated, added, origin)`, with
/// `origin` per gaussian of the new scene as in
/// [`crate::density::Round::origin`]: a relocated or added sample is new.
///
/// Relocation runs FIRST so the gaussians it frees are counted against the
/// cap before anything new is asked for - a scene at its budget still has
/// this move available, which is the point of the method.
///
/// With a budget there is no growth RATE to choose. `max_gaussians` says how
/// many samples the scene is allowed and the schedule's only job is to get
/// there early enough that they all get optimized, so the scene grows
/// geometrically into its budget over the first FIFTH of the rounds and the
/// rest of the fit refines a set of gaussians that is no longer changing
/// size. That fifth is the paper's own constant read off its schedule: 5%
/// every 100 iterations of 30k, from a COLMAP initialization to a cap an
/// order of magnitude above it, reaches the cap about a sixth of the way in.
/// Without a budget there is nothing to aim at, and the literal 5% per round
/// is all that is left.
pub fn step(scene: &mut Splats, cfg: &FitCfg, round: usize, rounds: usize) -> (usize, usize, Vec<Option<usize>>) {
    let seed = 0x3d47_535f_4d43_4d43_u64 ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let n = scene.len();
    let mut origin: Vec<Option<usize>> = (0..n).map(Some).collect();
    let moved = relocate(scene, cfg.prune_opacity, seed);
    for &d in &moved {
        origin[d] = None;
    }
    let want = if cfg.max_gaussians == 0 {
        n + ((n as f32 * cfg.mcmc_grow_frac) as usize).max(1)
    } else {
        let cap = cfg.max_gaussians;
        let full_by = (rounds / 5).max(1);
        if round >= full_by || n >= cap {
            cap
        } else {
            let rate = (cap as f32 / n as f32).powf(1.0 / (full_by - round) as f32);
            ((n as f32 * rate) as usize).clamp(n + 1, cap)
        }
    };
    let added = grow(scene, want, seed ^ 0xa5a5_a5a5);
    origin.resize(scene.len(), None);
    (moved.len(), added, origin)
}
