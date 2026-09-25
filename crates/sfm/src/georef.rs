// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putting a reconstruction into the world. Photographs alone fix a scene
//! only up to a similarity - where it is, which way it faces, how big it
//! is. Two things a camera records pin parts of that down:
//!
//! * SATELLITE FIXES (EXIF GPS). The camera centres are matched to their
//!   fixes, expressed as local east-north-up metres (WGS84, the ellipsoid
//!   GPS reports on), by a similarity - Umeyama, "Least-squares estimation
//!   of transformation parameters between two point patterns" (TPAMI 1991) -
//!   inside RANSAC, because a phone's fix can be tens of metres out. The
//!   result is refused when the fixes' own scatter is comparable to the
//!   ground the capture covers: a fix from the cell network repeats the same
//!   arc-second for a whole walk round a table and says nothing about scale.
//! * HOW THE PHOTOGRAPHS WERE HELD. A photograph is taken with the camera's
//!   horizontal axis close to level - tilted up or down freely, but seldom
//!   rolled - and its pixels are stored upright (EXIF orientation applied),
//!   so every camera's +X axis lies close to the horizontal plane and gravity
//!   is the direction most nearly perpendicular to all of them: the smallest
//!   eigenvector of their scatter, found robustly so a photograph turned on
//!   its side is outvoted. This is the rule OpenSfM's "horizontal"
//!   alignment uses, and unlike an orbit-plane fit it assumes nothing about
//!   the path the photographer walked.
//!
//! When both are available the gravity direction from the cameras is held
//! exactly in the similarity and the fixes supply position, heading and
//! scale: a phone's altitude is its least reliable coordinate, and over a
//! capture a few tens of metres across it would tilt the scene by degrees.
//!
//! Swedish Embedded AB implements georeferenced photogrammetry for its
//! clients. If your team needs reconstructions in metres and on the map, you
//! can procure our services by sending an email to info@swedishembedded.com.

use crate::camera::Pose;
use crate::linalg::{add, det, dot, eigh, mm, mv, norm, normalize, scale, sub, svd3, transpose, M3, V3};
use data::rng::Lcg;

/// A satellite fix, WGS84.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wgs84 {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    /// Metres above the ellipsoid or sea level as recorded; a fix without
    /// one cannot place a camera in three dimensions and is not used.
    pub altitude_m: Option<f64>,
}

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;

fn ecef(p: &Wgs84) -> V3 {
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let (lat, lon) = (p.latitude_deg.to_radians(), p.longitude_deg.to_radians());
    let h = p.altitude_m.unwrap_or(0.0);
    let n = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    [(n + h) * lat.cos() * lon.cos(), (n + h) * lat.cos() * lon.sin(), (n * (1.0 - e2) + h) * lat.sin()]
}

/// `p` in metres east, north and up of `origin` (the local tangent frame at
/// `origin`). A missing altitude counts as 0.
pub fn enu(origin: &Wgs84, p: &Wgs84) -> V3 {
    let d = sub(ecef(p), ecef(origin));
    let (lat, lon) = (origin.latitude_deg.to_radians(), origin.longitude_deg.to_radians());
    let (sl, cl, so, co) = (lat.sin(), lat.cos(), lon.sin(), lon.cos());
    [-so * d[0] + co * d[1], -sl * co * d[0] - sl * so * d[1] + cl * d[2], cl * co * d[0] + cl * so * d[1] + sl * d[2]]
}

/// A similarity `x -> s R x + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sim3 {
    pub s: f64,
    pub r: M3,
    pub t: V3,
}

impl Sim3 {
    pub fn apply(&self, x: V3) -> V3 {
        add(scale(mv(&self.r, x), self.s), self.t)
    }

    /// A world-to-camera pose in the transformed world: the same camera,
    /// its centre moved by the similarity, its frame turned with it.
    pub fn apply_pose(&self, p: &Pose) -> Pose {
        let r = mm(&p.r, &transpose(&self.r));
        let t = sub(scale(p.t, self.s), mv(&r, self.t));
        Pose { r, t }
    }

    /// A direction in the transformed world.
    pub fn apply_direction(&self, d: V3) -> V3 {
        mv(&self.r, d)
    }
}

/// The least-squares similarity taking `src` onto `dst` (Umeyama 1991),
/// optionally also turning the direction `dir.0` onto `dir.1` with weight
/// `dir.2` relative to the points' own (a large weight holds it exactly and
/// leaves the points only the rotation about it). `None` for fewer than two
/// points or a degenerate configuration.
pub fn umeyama(src: &[V3], dst: &[V3], dir: Option<(V3, V3, f64)>) -> Option<Sim3> {
    let n = src.len();
    if n < 2 || dst.len() != n {
        return None;
    }
    let mean = |p: &[V3]| scale(p.iter().fold([0.0; 3], |a, b| add(a, *b)), 1.0 / n as f64);
    let (ms, md) = (mean(src), mean(dst));
    let mut cov = [0.0f64; 9];
    let mut var = 0.0;
    let mut lever = 0.0;
    for (a, b) in src.iter().zip(dst) {
        let (a, b) = (sub(*a, ms), sub(*b, md));
        var += dot(a, a);
        lever += norm(a) * norm(b);
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += b[i] * a[j];
            }
        }
    }
    if var <= 1e-300 {
        return None;
    }
    if let Some((from, to, w)) = dir {
        let (f, t) = (normalize(from), normalize(to));
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += w * lever * t[i] * f[j];
            }
        }
    }
    let (u, _, v) = svd3(&cov);
    let mut d = [1.0, 1.0, 1.0];
    if det(&u) * det(&v) < 0.0 {
        d[2] = -1.0;
    }
    let ud: M3 = std::array::from_fn(|k| u[k] * d[k % 3]);
    let r = mm(&ud, &transpose(&v));
    // with the rotation fixed, the scale that best fits the points
    let s = src.iter().zip(dst).map(|(a, b)| dot(sub(*b, md), mv(&r, sub(*a, ms)))).sum::<f64>() / var;
    if !(s.is_finite() && s > 0.0) {
        return None;
    }
    Some(Sim3 { s, r, t: sub(md, scale(mv(&r, ms), s)) })
}

#[derive(Clone, Copy, Debug)]
pub struct GeorefCfg {
    /// Largest distance, metres, at which a camera may sit from its fix and
    /// still count as agreeing with it.
    pub inlier_m: f64,
    /// Fewest agreeing fixes.
    pub min_fixes: usize,
    /// Largest RMS misfit of the agreeing fixes as a fraction of their
    /// spread: above it the fixes cannot tell the capture's scale.
    pub max_relative_error: f64,
    pub ransac_iters: usize,
}

impl Default for GeorefCfg {
    fn default() -> Self {
        GeorefCfg { inlier_m: 6.0, min_fixes: 3, max_relative_error: 0.1, ransac_iters: 2000 }
    }
}

/// Why fixes were not used.
#[derive(Clone, Debug, PartialEq)]
pub enum GeorefError {
    /// Fewer than [`GeorefCfg::min_fixes`] registered photographs carry a
    /// fix with altitude, or fewer than that agree with any similarity.
    TooFewFixes { usable: usize },
    /// The agreeing fixes miss by `rms_m` over a spread of `spread_m`: too
    /// coarse for the capture's size.
    Uninformative { rms_m: f64, spread_m: f64 },
    /// The fixes lie on a line and no gravity direction is known, so the
    /// rotation about that line is free.
    Collinear,
}

impl std::fmt::Display for GeorefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeorefError::TooFewFixes { usable } => write!(f, "{usable} usable satellite fixes agree, too few to place the capture"),
            GeorefError::Uninformative { rms_m, spread_m } => {
                write!(f, "the satellite fixes miss by {rms_m:.1} m over {spread_m:.1} m of ground: too coarse to scale the capture")
            }
            GeorefError::Collinear => write!(f, "the satellite fixes lie on a line and gravity is unknown"),
        }
    }
}

/// A fitted georeference.
#[derive(Clone, Debug)]
pub struct EnuFit {
    /// Reconstruction frame to metres east-north-up of `origin`.
    pub sim3: Sim3,
    pub origin: Wgs84,
    /// Per photograph, whether its fix agreed (false with no fix).
    pub inliers: Vec<bool>,
    /// RMS distance of the agreeing cameras from their fixes, metres.
    pub rms_m: f64,
}

/// The similarity from reconstruction-frame camera `centres` (`None` for
/// unregistered photographs) onto their `fixes`, in metres east-north-up of
/// the fixes' median. With `up` - gravity's opposite in the reconstruction
/// frame - it is held onto +Z exactly.
pub fn solve_enu(centres: &[Option<V3>], fixes: &[Option<Wgs84>], up: Option<V3>, cfg: &GeorefCfg) -> Result<EnuFit, GeorefError> {
    let idx: Vec<usize> = (0..centres.len().min(fixes.len())).filter(|&i| centres[i].is_some() && fixes[i].is_some_and(|f| f.altitude_m.is_some())).collect();
    if idx.len() < cfg.min_fixes.max(2) {
        return Err(GeorefError::TooFewFixes { usable: idx.len() });
    }
    let median = |mut v: Vec<f64>| {
        let m = v.len() / 2;
        *v.select_nth_unstable_by(m, f64::total_cmp).1
    };
    let fx: Vec<Wgs84> = idx.iter().map(|&i| fixes[i].unwrap()).collect();
    let origin = Wgs84 {
        latitude_deg: median(fx.iter().map(|f| f.latitude_deg).collect()),
        longitude_deg: median(fx.iter().map(|f| f.longitude_deg).collect()),
        altitude_m: Some(median(fx.iter().map(|f| f.altitude_m.unwrap()).collect())),
    };
    let src: Vec<V3> = idx.iter().map(|&i| centres[i].unwrap()).collect();
    let dst: Vec<V3> = fx.iter().map(|f| enu(&origin, f)).collect();
    // gravity dominates the tilt when known
    let dir = up.map(|u| (u, [0.0, 0.0, 1.0], 1e3));
    if dir.is_none() && collinear(&dst) {
        return Err(GeorefError::Collinear);
    }
    let fit = |sel: &[usize]| -> Option<Sim3> {
        let (s, d): (Vec<V3>, Vec<V3>) = sel.iter().map(|&k| (src[k], dst[k])).unzip();
        umeyama(&s, &d, dir)
    };
    let agree = |sim: &Sim3| -> Vec<usize> { (0..src.len()).filter(|&k| norm(sub(sim.apply(src[k]), dst[k])) <= cfg.inlier_m).collect() };
    let mut rng = Lcg::new(0x5eed);
    let m = src.len();
    let mut best: Vec<usize> = Vec::new();
    let mut need = cfg.ransac_iters;
    let mut done = 0;
    while done < need {
        done += 1;
        let mut sample = [0usize; 3];
        for k in 0..3 {
            loop {
                let c = rng.next_u32() as usize % m;
                if !sample[..k].contains(&c) {
                    sample[k] = c;
                    break;
                }
            }
        }
        let Some(sim) = fit(&sample) else { continue };
        let mut inl = agree(&sim);
        // local optimization: refit on the consensus until it stops growing
        for _ in 0..4 {
            let Some(refit) = fit(&inl) else { break };
            let grown = agree(&refit);
            if grown.len() <= inl.len() {
                break;
            }
            inl = grown;
        }
        if inl.len() > best.len() {
            best = inl;
            need = need.min(crate::ransac_iterations(best.len() as f64 / m as f64, 3, cfg.ransac_iters));
        }
    }
    if best.len() < cfg.min_fixes.max(2) {
        return Err(GeorefError::TooFewFixes { usable: best.len() });
    }
    let sim = fit(&best).ok_or(GeorefError::TooFewFixes { usable: best.len() })?;
    let best = agree(&sim);
    if best.len() < cfg.min_fixes.max(2) {
        return Err(GeorefError::TooFewFixes { usable: best.len() });
    }
    let rms_m = (best.iter().map(|&k| norm(sub(sim.apply(src[k]), dst[k])).powi(2)).sum::<f64>() / best.len() as f64).sqrt();
    let centroid = scale(best.iter().fold([0.0; 3], |a, &k| add(a, dst[k])), 1.0 / best.len() as f64);
    let spread_m = (best.iter().map(|&k| norm(sub(dst[k], centroid)).powi(2)).sum::<f64>() / best.len() as f64).sqrt();
    if rms_m > cfg.max_relative_error * spread_m {
        return Err(GeorefError::Uninformative { rms_m, spread_m });
    }
    let mut inliers = vec![false; centres.len()];
    for &k in &best {
        inliers[idx[k]] = true;
    }
    Ok(EnuFit { sim3: sim, origin, inliers, rms_m })
}

/// Whether points lie on a line: the second principal spread is under 2% of
/// the first.
fn collinear(p: &[V3]) -> bool {
    let n = p.len() as f64;
    let c = scale(p.iter().fold([0.0; 3], |a, b| add(a, *b)), 1.0 / n);
    let mut m = [0.0f64; 9];
    for q in p {
        let d = sub(*q, c);
        for i in 0..3 {
            for j in 0..3 {
                m[i * 3 + j] += d[i] * d[j];
            }
        }
    }
    let (vals, _) = eigh(&m, 3);
    vals[1] <= 4e-4 * vals[2].max(1e-300)
}

/// Gravity from the cameras.
#[derive(Clone, Debug)]
pub struct Gravity {
    /// Unit vector pointing UP (against gravity) in the reconstruction frame.
    pub up: V3,
    /// Per camera, whether it was held level enough to count.
    pub inliers: Vec<bool>,
    /// RMS angle of the counted cameras' horizontal axes out of the
    /// horizontal plane, degrees - how level they were held.
    pub spread_deg: f64,
}

/// Robust Cauchy scale on a camera's roll, degrees.
const ROLL_SCALE_DEG: f64 = 5.0;
/// A camera rolled further than this is not counted.
const ROLL_OUTLIER_DEG: f64 = 20.0;

/// The direction of gravity from world-to-camera `poses` whose images are
/// stored upright (+Y down in the image): the direction most nearly
/// perpendicular to every camera's +X axis, by iteratively reweighted
/// principal axes; when the +X axes all point one way (everyone facing down
/// one corridor) that leaves a plane of candidates, and the cameras' own
/// mean vertical picks the one in it. `None` for fewer than two cameras or
/// when fewer than half of them agree.
pub fn gravity_from_cameras(poses: &[Pose]) -> Option<Gravity> {
    let n = poses.len();
    if n < 2 {
        return None;
    }
    // rows of a world-to-camera rotation are the camera's axes in the world
    let axis = |p: &Pose, k: usize| [p.r[k * 3], p.r[k * 3 + 1], p.r[k * 3 + 2]];
    let xs: Vec<V3> = poses.iter().map(|p| axis(p, 0)).collect();
    let mut w = vec![1.0f64; n];
    let mut up = [0.0f64; 3];
    let sigma = ROLL_SCALE_DEG.to_radians();
    for _ in 0..20 {
        let mut m = [0.0f64; 9];
        let mut down = [0.0f64; 3];
        for (i, x) in xs.iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    m[a * 3 + b] += w[i] * x[a] * x[b];
                }
            }
            down = add(down, scale(axis(&poses[i], 1), w[i]));
        }
        let (vals, vecs) = eigh(&m, 3);
        let mut u: V3 = [vecs[0][0], vecs[0][1], vecs[0][2]];
        if vals[1] < 0.05 * vals[2] {
            // one horizontal direction only: the plane perpendicular to it
            // holds up, and the cameras' mean vertical chooses within it
            let h = [vecs[2][0], vecs[2][1], vecs[2][2]];
            let minus_down = scale(down, -1.0);
            let proj = sub(minus_down, scale(h, dot(minus_down, h)));
            if norm(proj) < 1e-9 {
                return None;
            }
            u = normalize(proj);
        }
        // the sign: photographs look at the horizon or down more than up,
        // so up is where the cameras' -Y points, helped by where their +Z
        // does not
        let vote: f64 = poses.iter().zip(&w).map(|(p, wi)| wi * dot(scale(add(axis(p, 1), scale(axis(p, 2), 0.5)), -1.0), u)).sum();
        if vote < 0.0 {
            u = scale(u, -1.0);
        }
        let moved = norm(sub(u, up));
        up = u;
        for (i, x) in xs.iter().enumerate() {
            let roll = dot(*x, up).clamp(-1.0, 1.0).asin();
            w[i] = 1.0 / (1.0 + (roll / sigma).powi(2));
        }
        if moved < 1e-12 {
            break;
        }
    }
    let roll: Vec<f64> = xs.iter().map(|x| dot(*x, up).clamp(-1.0, 1.0).asin().abs().to_degrees()).collect();
    let inliers: Vec<bool> = roll.iter().map(|&r| r <= ROLL_OUTLIER_DEG).collect();
    let count = inliers.iter().filter(|&&b| b).count();
    if count * 2 < n {
        return None;
    }
    let spread_deg = (roll.iter().zip(&inliers).filter(|(_, &b)| b).map(|(r, _)| r * r).sum::<f64>() / count as f64).sqrt();
    Some(Gravity { up, inliers, spread_deg })
}
