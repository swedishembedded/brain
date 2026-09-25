// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Incremental structure from motion, in the shape Schönberger & Frahm
//! describe ("Structure-from-Motion Revisited", CVPR 2016), implemented from
//! the paper:
//!
//! 1. features in every image, putative matches between every pair, and
//!    geometric verification of each pair against an essential matrix;
//! 2. TRACKS - the connected components of verified matches, one per scene
//!    point, dropping any component that claims two features in one image;
//! 3. a seed pair chosen for many inliers AND a wide enough baseline that its
//!    points triangulate well;
//! 4. then repeatedly: register the image that sees the most triangulated
//!    points (PnP in RANSAC), triangulate the tracks it newly completes,
//!    bundle-adjust, and drop observations the adjustment cannot explain.
//!
//! The camera is self-calibrated: one focal length and two radial distortion
//! coefficients shared by every image, refined by bundle adjustment once
//! three views constrain them. A capture with no metadata starts from a
//! focal-length guess.

use crate::ba::{bundle_adjust, BaCfg, Observation};
use crate::camera::{Intrinsics, Pose};
use crate::linalg::V3;
use crate::matching::match_descriptors;
use crate::sift::{detect, Keypoint, SiftCfg};
use crate::twoview::{median_angle, ransac_essential, ray_angle, relative_pose, triangulate};

/// One input photograph, interleaved 8-bit RGB.
pub struct Photo<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct SfmCfg {
    pub sift: SiftCfg,
    /// Lowe's ratio for putative matches.
    pub ratio: f32,
    /// Verified inliers a pair needs to count as overlapping.
    pub min_inliers: usize,
    /// Inlier threshold for verification and registration, pixels.
    pub thresh_px: f64,
    pub ransac_iters: usize,
    /// Focal length guess as a multiple of the longer side. 0.8 is about a
    /// 26 mm-equivalent lens, what phone and compact main cameras ship;
    /// bundle adjustment refines it.
    pub focal_guess: f64,
    /// Smallest ray angle a new point may be triangulated with, degrees.
    pub min_angle_deg: f64,
    /// Choose the focal length by solving the capture under a range of
    /// candidates, instead of trusting `focal_guess`. Off only when the focal
    /// length is known.
    pub estimate_focal: bool,
    /// Largest reprojection error an observation may keep, pixels.
    pub max_error_px: f64,
    /// Fewest 2D-3D inliers an image needs to be registered.
    pub min_register: usize,
    /// Most rounds of re-triangulating every track through the refined
    /// calibration and re-adjusting, stopped early once the focal length
    /// holds still.
    pub retriangulate: usize,
    /// Print progress.
    pub verbose: bool,
}

impl Default for SfmCfg {
    fn default() -> Self {
        SfmCfg {
            sift: SiftCfg::default(),
            ratio: 0.8,
            min_inliers: 40,
            thresh_px: 4.0,
            ransac_iters: 50_000,
            focal_guess: 0.8,
            estimate_focal: true,
            min_angle_deg: 1.5,
            max_error_px: 4.0,
            min_register: 20,
            retriangulate: 3,
            verbose: false,
        }
    }
}

/// A triangulated scene point.
#[derive(Clone, Debug)]
pub struct Point {
    pub xyz: V3,
    /// Mean colour of its observations, 0..1.
    pub rgb: [f32; 3],
    /// `(image, keypoint)` of every observation that supports it.
    pub obs: Vec<(usize, usize)>,
}

/// The recovered capture.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    pub intrinsics: Intrinsics,
    /// Per input image, `None` where it could not be registered.
    pub poses: Vec<Option<Pose>>,
    pub points: Vec<Point>,
    pub keypoints: Vec<Vec<Keypoint>>,
    /// RMS reprojection error of the final adjustment, pixels.
    pub rms_px: f64,
}

#[derive(Debug)]
pub enum SfmError {
    TooFewImages,
    MixedSizes,
    NoSeedPair,
}

impl std::fmt::Display for SfmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SfmError::TooFewImages => write!(f, "structure from motion needs at least two photographs"),
            SfmError::MixedSizes => write!(f, "every photograph must have the same size (one camera is assumed)"),
            SfmError::NoSeedPair => write!(
                f,
                "no pair of photographs overlaps enough, with enough baseline, to start a reconstruction; \
                 take more photographs with more overlap between neighbours"
            ),
        }
    }
}

fn uf_find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Every image's keypoints and descriptors.
type Features = Vec<(Vec<Keypoint>, Vec<f32>)>;
/// A verified image pair and its inlier matches.
type Pair = (usize, usize, Vec<(usize, usize)>);

/// Features, verified matches and tracks: everything that does not depend
/// on the calibration being right.
struct Prepared {
    n: usize,
    feats: Features,
    pairs: Vec<Pair>,
    tracks: Vec<Vec<(usize, usize)>>,
    track_of: Vec<Vec<Option<usize>>>,
}

/// What one incremental solve under a given starting calibration produced.
struct Solved {
    k: Intrinsics,
    poses: Vec<Option<Pose>>,
    points: Vec<Point>,
    rms: f64,
}

impl Solved {
    fn registered(&self) -> usize {
        self.poses.iter().flatten().count()
    }
}

/// Run structure from motion on `photos`, all taken with one camera.
pub fn reconstruct(photos: &[Photo], cfg: &SfmCfg) -> Result<Reconstruction, SfmError> {
    let n = photos.len();
    if n < 2 {
        return Err(SfmError::TooFewImages);
    }
    let (w, h) = (photos[0].width, photos[0].height);
    if photos.iter().any(|p| p.width != w || p.height != h) {
        return Err(SfmError::MixedSizes);
    }
    let log = |s: String| {
        if cfg.verbose {
            println!("sfm: {s}");
        }
    };

    // ---- 1. features ----
    let feats: Features = photos
        .iter()
        .map(|p| {
            let gray: Vec<f32> = p
                .rgb
                .chunks_exact(3)
                .map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0)
                .collect();
            detect(&gray, w as usize, h as usize, &cfg.sift)
        })
        .collect();
    log(format!("features: {:?}", feats.iter().map(|f| f.0.len()).collect::<Vec<_>>()));

    // ---- 2. matching + verification, through the guessed calibration ----
    let guess = Intrinsics::guess(w, h, cfg.focal_guess);
    let nrm = normalized(&feats, &guess);
    let mut pairs: Vec<Pair> = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let m = match_descriptors(&feats[a].1, &feats[b].1, cfg.ratio);
            if m.len() < cfg.min_inliers {
                continue;
            }
            let xa: Vec<[f64; 2]> = m.iter().map(|&(i, _)| nrm[a][i]).collect();
            let xb: Vec<[f64; 2]> = m.iter().map(|&(_, j)| nrm[b][j]).collect();
            let seed = (a * n + b) as u64 + 1;
            let Some((_, mask)) = ransac_essential(&xa, &xb, cfg.thresh_px / guess.f, cfg.ransac_iters, seed) else { continue };
            let inl: Vec<(usize, usize)> = m.iter().zip(&mask).filter(|(_, &ok)| ok).map(|(p, _)| *p).collect();
            if inl.len() >= cfg.min_inliers {
                pairs.push((a, b, inl));
            }
        }
    }
    log(format!("{} verified pairs", pairs.len()));

    // ---- 3. tracks ----
    let offs: Vec<usize> = feats
        .iter()
        .scan(0usize, |s, f| {
            let o = *s;
            *s += f.0.len();
            Some(o)
        })
        .collect();
    let total = offs.last().unwrap() + feats.last().unwrap().0.len();
    let mut parent: Vec<usize> = (0..total).collect();
    for (a, b, m) in &pairs {
        for &(i, j) in m {
            let (x, y) = (uf_find(&mut parent, offs[*a] + i), uf_find(&mut parent, offs[*b] + j));
            if x != y {
                parent[x] = y;
            }
        }
    }
    let mut comp: std::collections::HashMap<usize, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for img in 0..n {
        for kp in 0..feats[img].0.len() {
            let r = uf_find(&mut parent, offs[img] + kp);
            comp.entry(r).or_default().push((img, kp));
        }
    }
    let mut tracks: Vec<Vec<(usize, usize)>> = comp
        .into_values()
        .filter(|t| t.len() >= 2)
        .filter(|t| {
            let mut seen = std::collections::HashSet::new();
            t.iter().all(|(i, _)| seen.insert(*i))
        })
        .collect();
    tracks.sort();
    let mut track_of: Vec<Vec<Option<usize>>> = feats.iter().map(|f| vec![None; f.0.len()]).collect();
    for (ti, t) in tracks.iter().enumerate() {
        for &(i, kp) in t {
            track_of[i][kp] = Some(ti);
        }
    }
    log(format!("{} tracks", tracks.len()));
    let prep = Prepared { n, feats, pairs, tracks, track_of };

    // ---- 4. focal length ----
    // With no metadata the focal length is a guess, and a scene seeded
    // through a wrong one is projectively distorted: 10% of focal error is
    // ~100 px at the edge of a 2048 px frame, so later views stop
    // registering, and bundle adjustment creeps out of the distorted basin
    // by about a percent per round rather than leaving it. So the focal
    // length is chosen by what it is FOR: solve the whole capture with each
    // candidate held fixed and keep the one that registers the most views at
    // the lowest reprojection error. (Counting epipolar inliers over the
    // best pairs instead was measured choosing 0.45x the long side on a
    // capture whose solves bottom out at 0.68x.)
    let f0 = if cfg.estimate_focal {
        let side = w.max(h) as f64;
        let steps = 12;
        let candidates: Vec<f64> = (0..=steps).map(|i| side * 0.5 * (1.3f64 / 0.5).powf(i as f64 / steps as f64)).collect();
        let quiet = |_: String| {};
        let scored: Vec<Option<(usize, f64)>> = backend_cpu::par::map(candidates.len(), |i| {
            solve(&prep, Intrinsics { f: candidates[i], ..guess }, false, 0, cfg, &quiet).ok().map(|s| (s.registered(), s.rms))
        });
        let mut best: Option<(f64, usize, f64)> = None;
        for (f, sc) in candidates.iter().zip(&scored) {
            if let Some((r, e)) = sc {
                log(format!("  focal {f:7.1} px: {r} registered, rms {e:.3} px"));
                if best.is_none_or(|(_, br, be)| *r > br || (*r == br && *e < be)) {
                    best = Some((*f, *r, *e));
                }
            }
        }
        let f = best.map_or(guess.f, |b| b.0);
        log(format!("focal length {f:.1} px ({:.2} x the long side)", f / side));
        f
    } else {
        guess.f
    };

    // ---- 5. the reconstruction, calibration free ----
    let solved = solve(&prep, Intrinsics { f: f0, ..guess }, true, cfg.retriangulate, cfg, &log)?;
    let Solved { k, poses, mut points, rms } = solved;
    for p in &mut points {
        let mut c = [0.0f32; 3];
        for &(i, kp) in &p.obs {
            let f = prep.feats[i].0[kp];
            let (x, y) = ((f.x as u32).min(w - 1), (f.y as u32).min(h - 1));
            let o = ((y * w + x) * 3) as usize;
            for (ch, v) in c.iter_mut().enumerate() {
                *v += photos[i].rgb[o + ch] as f32 / 255.0;
            }
        }
        p.rgb = c.map(|v| v / p.obs.len() as f32);
    }
    Ok(Reconstruction { intrinsics: k, poses, points, keypoints: prep.feats.into_iter().map(|f| f.0).collect(), rms_px: rms })
}

/// Normalized, undistorted coordinates of every keypoint through `k`.
fn normalized(feats: &[(Vec<Keypoint>, Vec<f32>)], k: &Intrinsics) -> Vec<Vec<[f64; 2]>> {
    feats.iter().map(|(kp, _)| kp.iter().map(|p| k.to_normalized([p.x as f64, p.y as f64])).collect()).collect()
}

/// Seed, register, triangulate and adjust from starting calibration `k`.
/// `free` lets bundle adjustment refine the calibration; `retriangulate`
/// rounds of re-triangulation through it follow the incremental pass.
fn solve(prep: &Prepared, mut k: Intrinsics, free: bool, retriangulate: usize, cfg: &SfmCfg, log: &dyn Fn(String)) -> Result<Solved, SfmError> {
    let Prepared { n, feats, pairs, tracks, track_of, .. } = prep;
    let n = *n;
    let mut nrm = normalized(feats, &k);

    // Many points AND a wide baseline: a narrow pair triangulates poorly and
    // everything registered onto it inherits the error. Score by points
    // times baseline, saturating at 10 degrees.
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pairs[i].2.len()));
    let mut seed: Option<(usize, usize, Pose, f64)> = None;
    for &pi in order.iter().take(30) {
        let (a, b, m) = &pairs[pi];
        let xa: Vec<[f64; 2]> = m.iter().map(|&(i, _)| nrm[*a][i]).collect();
        let xb: Vec<[f64; 2]> = m.iter().map(|&(_, j)| nrm[*b][j]).collect();
        let Some((e, mask)) = ransac_essential(&xa, &xb, cfg.thresh_px / k.f, cfg.ransac_iters, 7) else { continue };
        let (pose, pts) = relative_pose(&e, &xa, &xb, &mask);
        let ang = median_angle(&pose, &pts).to_degrees();
        let good = pts.iter().flatten().count();
        if ang < 3.0 || good < cfg.min_inliers {
            continue;
        }
        let score = good as f64 * (ang / 10.0).min(1.0);
        if seed.as_ref().is_none_or(|s| score > s.3) {
            seed = Some((*a, *b, pose, score));
        }
    }
    let (sa, sb, sp, _) = seed.ok_or(SfmError::NoSeedPair)?;
    log(format!("seed pair {sa}-{sb}"));
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    poses[sa] = Some(Pose::identity());
    poses[sb] = Some(sp);
    let mut point_of: Vec<Option<usize>> = vec![None; tracks.len()];
    let mut points: Vec<Point> = Vec::new();

    let min_angle = cfg.min_angle_deg.to_radians();
    let max_err = cfg.max_error_px;
    let triangulate_new = |poses: &[Option<Pose>], k: &Intrinsics, nrm: &[Vec<[f64; 2]>], point_of: &mut Vec<Option<usize>>, points: &mut Vec<Point>| {
        let mut added = 0;
        for (ti, t) in tracks.iter().enumerate() {
            if point_of[ti].is_some() {
                continue;
            }
            let seen: Vec<(usize, usize)> = t.iter().copied().filter(|(i, _)| poses[*i].is_some()).collect();
            if seen.len() < 2 {
                continue;
            }
            let ps: Vec<&Pose> = seen.iter().map(|(i, _)| poses[*i].as_ref().unwrap()).collect();
            let us: Vec<[f64; 2]> = seen.iter().map(|(i, kp)| nrm[*i][*kp]).collect();
            let Some(x) = triangulate(&ps, &us) else { continue };
            // every supporting view must see it in front, close to where it
            // was detected, and some pair of them from far enough apart
            let mut ok = true;
            let mut best_angle = 0.0f64;
            for (j, p) in ps.iter().enumerate() {
                let Some(uv) = crate::camera::project(k, p, x) else {
                    ok = false;
                    break;
                };
                let kp = feats[seen[j].0].0[seen[j].1];
                if (uv[0] - kp.x as f64).hypot(uv[1] - kp.y as f64) > max_err {
                    ok = false;
                    break;
                }
                for q in ps.iter().skip(j + 1) {
                    best_angle = best_angle.max(ray_angle(p.centre(), q.centre(), x));
                }
            }
            if !ok || best_angle < min_angle {
                continue;
            }
            point_of[ti] = Some(points.len());
            points.push(Point { xyz: x, rgb: [0.0; 3], obs: seen });
            added += 1;
        }
        added
    };
    triangulate_new(&poses, &k, &nrm, &mut point_of, &mut points);

    let adjust = |k: &mut Intrinsics, poses: &mut [Option<Pose>], points: &mut Vec<Point>, point_of: &mut Vec<Option<usize>>, iters: usize| -> f64 {
        let reg: Vec<usize> = (0..n).filter(|&i| poses[i].is_some()).collect();
        let mut cam_ix = vec![usize::MAX; n];
        for (c, &i) in reg.iter().enumerate() {
            cam_ix[i] = c;
        }
        let mut cp: Vec<Pose> = reg.iter().map(|&i| poses[i].unwrap()).collect();
        let mut xs: Vec<V3> = points.iter().map(|p| p.xyz).collect();
        let obs: Vec<Observation> = points
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| p.obs.iter().map(move |&(i, kp)| (pi, i, kp)))
            .map(|(pi, i, kp)| {
                let f = feats[i].0[kp];
                Observation { cam: cam_ix[i], point: pi, px: [f.x as f64, f.y as f64] }
            })
            .collect();
        let bcfg = BaCfg { iters, intrinsics: free && reg.len() >= 4, fixed: vec![cam_ix[sa]], ..BaCfg::default() };
        let rep = bundle_adjust(k, &mut cp, &mut xs, &obs, &bcfg);
        for (c, &i) in reg.iter().enumerate() {
            poses[i] = Some(cp[c]);
        }
        // drop observations the adjustment cannot explain, and points left
        // with fewer than two
        let mut keep: Vec<Point> = Vec::new();
        for (pi, mut p) in points.drain(..).enumerate() {
            p.xyz = xs[pi];
            p.obs.retain(|&(i, kp)| {
                let f = feats[i].0[kp];
                crate::camera::project(k, poses[i].as_ref().unwrap(), p.xyz)
                    .is_some_and(|uv| (uv[0] - f.x as f64).hypot(uv[1] - f.y as f64) <= max_err)
            });
            if p.obs.len() >= 2 {
                keep.push(p);
            }
        }
        *points = keep;
        point_of.iter_mut().for_each(|v| *v = None);
        for (pi, p) in points.iter().enumerate() {
            let (i, kp) = p.obs[0];
            if let Some(t) = track_of[i][kp] {
                point_of[t] = Some(pi);
            }
        }
        rep.rms_after
    };
    let r = adjust(&mut k, &mut poses, &mut points, &mut point_of, 30);
    log(format!("seed: {} points, rms {r:.2} px", points.len()));

    // register the rest, most-connected first
    let mut failed = vec![false; n];
    let mut rms = r;
    loop {
        nrm = normalized(feats, &k);
        let mut best: Option<(usize, usize)> = None;
        for i in 0..n {
            if poses[i].is_some() || failed[i] {
                continue;
            }
            let c = (0..feats[i].0.len()).filter(|&kp| track_of[i][kp].and_then(|t| point_of[t]).is_some()).count();
            if best.is_none_or(|(_, bc)| c > bc) {
                best = Some((i, c));
            }
        }
        let Some((img, count)) = best else { break };
        if count < cfg.min_register {
            failed[img] = true;
            continue;
        }
        let corr: Vec<(V3, [f64; 2])> = (0..feats[img].0.len())
            .filter_map(|kp| track_of[img][kp].and_then(|t| point_of[t]).map(|pi| (points[pi].xyz, nrm[img][kp])))
            .collect();
        let (xs, us): (Vec<V3>, Vec<[f64; 2]>) = corr.into_iter().unzip();
        let reg = crate::pnp::ransac_pnp(&xs, &us, cfg.thresh_px / k.f, cfg.ransac_iters, img as u64 + 99);
        let Some((pose, mask)) = reg.filter(|(_, m)| m.iter().filter(|&&v| v).count() >= cfg.min_register) else {
            failed[img] = true;
            log(format!("image {img}: registration failed ({count} candidate correspondences)"));
            continue;
        };
        poses[img] = Some(pose);
        // attach the image's inlier observations to existing points
        let mut j = 0;
        for (kp, track) in track_of[img].iter().enumerate() {
            if let Some(pi) = track.and_then(|t| point_of[t]) {
                if mask[j] {
                    points[pi].obs.push((img, kp));
                }
                j += 1;
            }
        }
        let added = triangulate_new(&poses, &k, &nrm, &mut point_of, &mut points);
        rms = adjust(&mut k, &mut poses, &mut points, &mut point_of, 20);
        log(format!(
            "image {img}: {} inliers, +{added} points, {} total, rms {rms:.2} px, f {:.1} k1 {:+.4} k2 {:+.4}",
            mask.iter().filter(|&&v| v).count(),
            points.len(),
            k.f,
            k.k1,
            k.k2
        ));
        // a failure may succeed once more of the scene exists
        failed.iter_mut().for_each(|f| *f = false);
    }
    // Points triangulated early were triangulated through the calibration
    // of the moment. Re-triangulate every track through the refined camera
    // and adjust again, until the focal length holds still.
    for round in 0..retriangulate {
        let before = k.f;
        nrm = normalized(feats, &k);
        points.clear();
        point_of.iter_mut().for_each(|v| *v = None);
        triangulate_new(&poses, &k, &nrm, &mut point_of, &mut points);
        rms = adjust(&mut k, &mut poses, &mut points, &mut point_of, 100);
        log(format!("re-triangulation {round}: {} points, rms {rms:.3} px, f {:.1} k1 {:+.4} k2 {:+.4}", points.len(), k.f, k.k1, k.k2));
        if (k.f - before).abs() < 1e-3 * before {
            break;
        }
    }
    Ok(Solved { k, poses, points, rms })
}
