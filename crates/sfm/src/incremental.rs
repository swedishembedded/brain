// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structure from motion, from photographs to calibrated cameras and a
//! sparse point cloud, in the shape Schönberger & Frahm describe
//! ("Structure-from-Motion Revisited", CVPR 2016) with a global initializer
//! in the shape of GLOMAP (Pan et al., "Global Structure-from-Motion
//! Revisited", ECCV 2024), both implemented from the papers:
//!
//! 1. features in every image; the pairs worth matching chosen by a global
//!    image descriptor ([`crate::retrieval`]; every pair of a small
//!    capture), putative matches between them, and geometric verification
//!    of each pair against an essential matrix;
//! 2. TRACKS - the connected components of verified matches, one per scene
//!    point, dropping any component that claims two features in one image;
//! 3. the cameras and points, by one of two initializers:
//!    * GLOBAL - every camera's rotation at once by robust rotation
//!      averaging over the view graph ([`crate::rotation`]), then every
//!      camera centre and point at once from the bearings
//!      ([`crate::positioning`]), then triangulation and bundle adjustment;
//!    * INCREMENTAL - a seed pair chosen for many inliers AND a wide enough
//!      baseline, then repeatedly: register the image that sees the most
//!      triangulated points (PnP in RANSAC), triangulate the tracks it newly
//!      completes, bundle-adjust, and drop observations the adjustment
//!      cannot explain.
//!
//!    By default the global one runs first and is kept when it registers
//!    every photograph the view graph connects; otherwise the incremental
//!    one runs too and the result that registers more views (then has the
//!    lower reprojection error) is kept ([`SfmReport::initializer`]).
//!    Photographs either leaves out are registered by PnP afterwards;
//! 4. the lens model chosen by fitting every candidate to the result
//!    ([`crate::lens`]), and the tracks re-triangulated through it;
//! 5. the frame put in the world where the photographs say how
//!    ([`crate::georef`]): metres east-north-up from satellite fixes, and
//!    the direction of gravity from how the cameras were held.
//!
//! Every image measurement is a BEARING through the current calibration
//! (`camera::Intrinsics::unproject`), so the geometry works for any lens the
//! camera model describes, and thresholds are angles: a pixel threshold
//! divided by the focal length.
//!
//! The cameras are self-calibrated PER SENSOR: photographs from one physical
//! camera ([`Photo::sensor`]) share one calibration. A sensor with a focal
//! length from metadata starts from it and bundle adjustment refines it
//! under a soft prior; one without is found by solving the capture under a
//! sweep of candidates. Bundle adjustment frees the calibration in stages -
//! poses and points first, then focal length and the two lowest lens
//! coefficients once a sensor has three views, then the principal point and
//! the full lens once it has [`SfmCfg::full_model_views`].
//!
//! THE GAUGE is [`Reconstruction::gauge`]: with usable satellite fixes,
//! metres east-north-up (+Z up) about a named origin; otherwise the seed
//! pair's - the first photograph of the pair is the world frame and the
//! distance between the pair's camera centres the unit of length, held
//! exactly by bundle adjustment ([`crate::ba`]) - with
//! [`Reconstruction::up`] naming the direction of gravity in it when the
//! cameras say.

use crate::ba::{bundle_adjust, BaCfg, Observation, Param};
use crate::camera::Pose;
use crate::georef::{gravity_from_cameras, solve_enu, GeorefCfg, GeorefError, Wgs84};
use crate::lens::{describe, fit_lenses, stage_for, LensCfg, LensChoice, LensFit, Stage};
use crate::linalg::{dot, mv, scale, V3};
use crate::matching::match_descriptors;
use crate::positioning::{global_positions, PositioningCfg, RayObservation};
use crate::retrieval::{select_pairs, PairSelection};
use crate::rotation::{average_rotations, RelativeRotation, RotationCfg};
use crate::sift::{detect, Keypoint, SiftCfg};
use crate::twoview::{median_angle, ransac_essential, ray_angle, relative_pose, triangulate};
use ::camera::{Intrinsics, Lens};
use std::time::Instant;

/// One input photograph, interleaved 8-bit RGB.
pub struct Photo<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb: &'a [u8],
    /// Which physical camera (and zoom setting) took it: photographs of one
    /// sensor share one calibration. Sensors are numbered from 0 with no
    /// gaps.
    pub sensor: usize,
    /// Focal length in pixels from metadata (EXIF focal length over the
    /// pixel pitch), when there is one.
    pub focal_px: Option<f64>,
    /// Where it was taken (EXIF GPS), when recorded.
    pub gps: Option<Wgs84>,
}

/// How the cameras were first put together; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Initializer {
    Global,
    Incremental,
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
    /// Focal length guess as a multiple of the longer side, for a sensor
    /// with no metadata when `estimate_focal` is off. 0.8 is about a 26
    /// mm-equivalent lens, what phone and compact main cameras ship.
    pub focal_guess: f64,
    /// Smallest ray angle a new point may be triangulated with, degrees.
    pub min_angle_deg: f64,
    /// Choose each metadata-less sensor's focal length by solving the
    /// capture under a range of candidates, instead of trusting
    /// `focal_guess`.
    pub estimate_focal: bool,
    /// Largest reprojection error an observation may keep, pixels.
    pub max_error_px: f64,
    /// Fewest 2D-3D inliers an image needs to be registered.
    pub min_register: usize,
    /// Most rounds of re-triangulating every track through the refined
    /// calibration and re-adjusting, stopped early once every focal length
    /// holds still.
    pub retriangulate: usize,
    /// The lens model, or `Auto` to choose it by information criterion.
    pub lens: LensChoice,
    /// Tie `fx = fy`.
    pub square_pixels: bool,
    /// Views a sensor needs before its principal point and full lens model
    /// are refined.
    pub full_model_views: usize,
    /// Sigma of the prior on a metadata focal length, as a fraction of it.
    pub focal_prior: f64,
    /// Sigma of the prior holding the principal point near the image
    /// centre, as a fraction of the longer side.
    pub pp_prior: f64,
    /// Which image pairs are matched.
    pub pairs: PairSelection,
    /// The initializer, or `None` to choose by result (see the module
    /// documentation).
    pub initializer: Option<Initializer>,
    pub rotation: RotationCfg,
    pub positioning: PositioningCfg,
    /// When satellite fixes are trusted.
    pub georef: GeorefCfg,
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
            lens: LensChoice::Auto,
            square_pixels: true,
            full_model_views: 6,
            focal_prior: 0.05,
            pp_prior: 0.02,
            pairs: PairSelection::default(),
            initializer: None,
            rotation: RotationCfg::default(),
            positioning: PositioningCfg::default(),
            georef: GeorefCfg::default(),
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
    /// One calibration per sensor.
    pub intrinsics: Vec<Intrinsics>,
    /// Per input image, its sensor: `intrinsics[sensor[i]]` is image `i`'s
    /// calibration.
    pub sensor: Vec<usize>,
    /// Per input image, `None` where it could not be registered.
    pub poses: Vec<Option<Pose>>,
    pub points: Vec<Point>,
    pub keypoints: Vec<Vec<Keypoint>>,
    /// RMS reprojection error of the final adjustment, pixels.
    pub rms_px: f64,
    /// The lens model the calibration is written in.
    pub lens: LensChoice,
    /// Every candidate model's fit, the chosen one included.
    pub lens_fits: Vec<LensFit>,
    /// The frame and unit of length everything above is expressed in.
    pub gauge: Gauge,
    /// The direction opposite gravity in that frame, unit length: +Z in
    /// [`Gauge::Enu`], estimated from how the cameras were held in
    /// [`Gauge::Seed`] (`None` when they do not agree on one).
    pub up: Option<V3>,
    pub report: SfmReport,
}

/// The similarity photographs cannot fix, fixed.
#[derive(Clone, Debug, PartialEq)]
pub enum Gauge {
    /// From the photographs alone: image `anchor` is the world frame
    /// (identity pose) and the distance from its centre to image `scale`'s
    /// is the unit of length.
    Seed { anchor: usize, scale: usize },
    /// From satellite fixes: metres east (+X), north (+Y) and up (+Z) of
    /// `origin`, on the WGS84 ellipsoid's local tangent plane. `fixes`
    /// photographs agreed with the fit, to `rms_m`.
    Enu { origin: Wgs84, rms_m: f64, fixes: usize },
}

/// What the run did and what it cost.
#[derive(Clone, Debug)]
pub struct SfmReport {
    /// Image pairs matched, and of those verified.
    pub pairs_matched: usize,
    pub pairs_verified: usize,
    /// The initializer whose result was kept.
    pub initializer: Initializer,
    /// Wall-clock seconds: features, matching with verification, the focal
    /// length sweep, and everything after.
    pub features_s: f64,
    pub matching_s: f64,
    pub focal_s: f64,
    pub solve_s: f64,
    /// Why the photographs' satellite fixes were not applied, when they
    /// carried some.
    pub georef_refused: Option<GeorefError>,
}

#[derive(Debug)]
pub enum SfmError {
    TooFewImages,
    /// Photographs of one sensor differ in size.
    MixedSizes { sensor: usize },
    /// No photograph names this sensor, though a higher-numbered one is used.
    UnusedSensor { sensor: usize },
    NoSeedPair,
    /// Rotation averaging connected too few photographs to position
    /// (only when the global initializer is asked for by name).
    WeakViewGraph { cameras: usize },
}

impl std::fmt::Display for SfmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SfmError::TooFewImages => write!(f, "structure from motion needs at least two photographs"),
            SfmError::MixedSizes { sensor } => write!(f, "the photographs of sensor {sensor} differ in size; one sensor is one camera, so they must match"),
            SfmError::UnusedSensor { sensor } => write!(f, "no photograph is from sensor {sensor}; number sensors from 0 without gaps"),
            SfmError::NoSeedPair => write!(
                f,
                "no pair of photographs overlaps enough, with enough baseline, to start a reconstruction; \
                 take more photographs with more overlap between neighbours"
            ),
            SfmError::WeakViewGraph { cameras } => write!(
                f,
                "only {cameras} photographs have consistent relative rotations; the global initializer needs at least three"
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
/// Per image, per keypoint, its bearing through the current calibration.
type Rays = Vec<Vec<Option<V3>>>;

/// Features, verified matches and tracks: everything that does not depend
/// on the calibration being right.
struct Prepared {
    n: usize,
    sensor: Vec<usize>,
    feats: Features,
    pairs: Vec<Pair>,
    tracks: Vec<Vec<(usize, usize)>>,
    track_of: Vec<Vec<Option<usize>>>,
}

/// Bearings of every keypoint through its image's sensor's calibration.
fn bearings(prep_feats: &[(Vec<Keypoint>, Vec<f32>)], sensor: &[usize], ks: &[Intrinsics]) -> Rays {
    prep_feats.iter().zip(sensor).map(|((kp, _), &s)| kp.iter().map(|p| ks[s].unproject([p.x as f64, p.y as f64])).collect()).collect()
}

/// The lens a solve starts in: the model asked for, or the two-coefficient
/// radial camera from which every other perspective model is reached.
fn starting_lens(choice: LensChoice) -> Lens {
    match choice {
        LensChoice::Pinhole => Lens::Pinhole,
        LensChoice::Fisheye => Lens::Fisheye { k: [0.0; 4] },
        LensChoice::Auto | LensChoice::Radial | LensChoice::Brown => Lens::radial(0.0, 0.0),
    }
}

/// Run structure from motion on `photos`.
pub fn reconstruct(photos: &[Photo], cfg: &SfmCfg) -> Result<Reconstruction, SfmError> {
    let n = photos.len();
    if n < 2 {
        return Err(SfmError::TooFewImages);
    }
    let nsensors = photos.iter().map(|p| p.sensor).max().unwrap() + 1;
    let mut size: Vec<Option<(u32, u32)>> = vec![None; nsensors];
    let mut prior: Vec<Option<f64>> = vec![None; nsensors];
    for p in photos {
        match size[p.sensor] {
            None => size[p.sensor] = Some((p.width, p.height)),
            Some(s) if s != (p.width, p.height) => return Err(SfmError::MixedSizes { sensor: p.sensor }),
            _ => {}
        }
        if prior[p.sensor].is_none() {
            prior[p.sensor] = p.focal_px.filter(|f| f.is_finite() && *f > 0.0);
        }
    }
    if let Some(s) = size.iter().position(|s| s.is_none()) {
        return Err(SfmError::UnusedSensor { sensor: s });
    }
    let size: Vec<(u32, u32)> = size.into_iter().flatten().collect();
    let sensor: Vec<usize> = photos.iter().map(|p| p.sensor).collect();
    let log = |s: String| {
        if cfg.verbose {
            println!("sfm: {s}");
        }
    };

    // ---- 1. features ----
    let clock = Instant::now();
    let feats: Features = photos
        .iter()
        .map(|p| {
            let gray: Vec<f32> = p
                .rgb
                .chunks_exact(3)
                .map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0)
                .collect();
            detect(&gray, p.width as usize, p.height as usize, &cfg.sift)
        })
        .collect();
    let features_s = clock.elapsed().as_secs_f64();
    log(format!("features: {:?} in {features_s:.1} s", feats.iter().map(|f| f.0.len()).collect::<Vec<_>>()));

    // ---- 2. matching + verification, through the guessed calibration ----
    let lens0 = starting_lens(cfg.lens);
    let guess: Vec<Intrinsics> = size
        .iter()
        .zip(&prior)
        .map(|(&(w, h), p)| Intrinsics { lens: lens0, ..Intrinsics::pinhole(p.unwrap_or(cfg.focal_guess * w.max(h) as f64), w, h) })
        .collect();
    for (s, k) in guess.iter().enumerate() {
        let from = if prior[s].is_some() { "metadata" } else { "guess" };
        log(format!("sensor {s}: {}x{}, {} photographs, focal {:.1} px ({from})", k.width, k.height, sensor.iter().filter(|&&x| x == s).count(), k.fx));
    }
    let clock = Instant::now();
    let rays = bearings(&feats, &sensor, &guess);
    let descs: Vec<&[f32]> = feats.iter().map(|f| f.1.as_slice()).collect();
    let candidates = select_pairs(&descs, &cfg.pairs, 0x7e7a);
    log(format!("{} of {} pairs selected for matching", candidates.len(), n * (n - 1) / 2));
    let mut pairs: Vec<Pair> = Vec::new();
    for &(a, b) in &candidates {
        let m = match_descriptors(&feats[a].1, &feats[b].1, cfg.ratio);
        let m: Vec<(usize, usize)> = m.into_iter().filter(|&(i, j)| rays[a][i].is_some() && rays[b][j].is_some()).collect();
        if m.len() < cfg.min_inliers {
            continue;
        }
        let xa: Vec<V3> = m.iter().map(|&(i, _)| rays[a][i].unwrap()).collect();
        let xb: Vec<V3> = m.iter().map(|&(_, j)| rays[b][j].unwrap()).collect();
        let seed = (a * n + b) as u64 + 1;
        let thresh = pair_threshold(cfg.thresh_px, &guess[sensor[a]], &guess[sensor[b]]);
        let Some((_, mask)) = ransac_essential(&xa, &xb, thresh, cfg.ransac_iters, seed) else { continue };
        let inl: Vec<(usize, usize)> = m.iter().zip(&mask).filter(|(_, &ok)| ok).map(|(p, _)| *p).collect();
        let verified = inl.len() >= cfg.min_inliers;
        log(format!("pair {a}-{b}: {} matches, {} inliers ({:.0}%){}", m.len(), inl.len(), 100.0 * inl.len() as f64 / m.len() as f64, if verified { "" } else { ", rejected" }));
        if verified {
            pairs.push((a, b, inl));
        }
    }
    let matching_s = clock.elapsed().as_secs_f64();
    log(format!("{} verified pairs of {} matched in {matching_s:.1} s", pairs.len(), candidates.len()));

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
    let reachable = largest_component(n, &pairs);
    let prep = Prepared { n, sensor, feats, pairs, tracks, track_of };

    // ---- 4. focal length, per sensor without metadata ----
    // With no metadata the focal length is a guess, and a scene seeded
    // through a wrong one is projectively distorted: 10% of focal error is
    // ~100 px at the edge of a 2048 px frame, so later views stop
    // registering, and bundle adjustment creeps out of the distorted basin
    // by about a percent per round rather than leaving it. So the focal
    // length is chosen by what it is FOR: solve the whole capture with each
    // candidate held fixed and keep the one that registers the most views at
    // the lowest reprojection error (see knowledge #159 for what scoring by
    // epipolar inliers picks instead). The sweep reaches down to 0.3x the
    // long side, where phone ultra-wides and fisheyes sit.
    let bcfg = BaCfg { focal_prior: prior.iter().map(|p| p.map(|f| (f, cfg.focal_prior * f))).collect(), pp_prior: cfg.pp_prior, ..BaCfg::default() };
    let clock = Instant::now();
    let mut ks = guess.clone();
    if cfg.estimate_focal {
        for s in 0..nsensors {
            if prior[s].is_some() {
                continue;
            }
            let side = ks[s].width.max(ks[s].height) as f64;
            let steps = 18;
            let (lo, hi) = (0.3, 1.3);
            let candidates: Vec<f64> = (0..=steps).map(|i| side * lo * (hi / lo).powf(i as f64 / steps as f64)).collect();
            let quiet = |_: String| {};
            let scored: Vec<Option<(usize, f64)>> = backend_cpu::par::map(candidates.len(), |i| {
                let mut trial = ks.clone();
                trial[s] = Intrinsics { fx: candidates[i], fy: candidates[i], ..trial[s] };
                Solver::seed(&prep, trial, false, bcfg.clone(), cfg, &quiet).ok().map(|mut sv| {
                    sv.register_all(&quiet);
                    (sv.registered(), sv.rms)
                })
            });
            let mut best: Option<(f64, usize, f64)> = None;
            for (f, sc) in candidates.iter().zip(&scored) {
                if let Some((r, e)) = sc {
                    log(format!("  sensor {s} focal {f:7.1} px: {r} registered, rms {e:.3} px"));
                    if best.is_none_or(|(_, br, be)| *r > br || (*r == br && *e < be)) {
                        best = Some((*f, *r, *e));
                    }
                }
            }
            let f = best.map_or(ks[s].fx, |b| b.0);
            // a winner on the edge of the sweep is a bound, not an optimum
            // (knowledge #163)
            let edge = if f == candidates[0] || f == candidates[steps] { " - AT THE EDGE of the sweep, the true focal length may lie beyond it" } else { "" };
            log(format!("sensor {s}: focal length {f:.1} px ({:.2} x the long side){edge}", f / side));
            ks[s] = Intrinsics { fx: f, fy: f, ..ks[s] };
        }
    }

    let focal_s = clock.elapsed().as_secs_f64();

    // ---- 5. the reconstruction, calibration free ----
    let clock = Instant::now();
    let incremental = |log: &dyn Fn(String)| -> Result<Solver, SfmError> {
        let mut sv = Solver::seed(&prep, ks.clone(), true, bcfg.clone(), cfg, log)?;
        sv.register_all(log);
        Ok(sv)
    };
    let global = |log: &dyn Fn(String)| -> Result<Solver, SfmError> {
        let mut sv = Solver::global(&prep, ks.clone(), bcfg.clone(), cfg, log)?;
        sv.register_all(log);
        Ok(sv)
    };
    let (mut sv, initializer) = match cfg.initializer {
        Some(Initializer::Incremental) => (incremental(&log)?, Initializer::Incremental),
        Some(Initializer::Global) => (global(&log)?, Initializer::Global),
        None => match global(&log) {
            Ok(g) if g.registered() >= reachable => (g, Initializer::Global),
            g => {
                if let Ok(g) = &g {
                    log(format!("global initializer registered {} of the {reachable} connected photographs, rms {:.3} px; trying the incremental one", g.registered(), g.rms));
                }
                let inc = incremental(&log);
                match (g, inc) {
                    (Ok(g), Ok(i)) => {
                        let global_wins = g.registered() > i.registered() || (g.registered() == i.registered() && g.rms <= i.rms);
                        log(format!("incremental: {} registered, rms {:.3} px; keeping the {}", i.registered(), i.rms, if global_wins { "global" } else { "incremental" }));
                        if global_wins { (g, Initializer::Global) } else { (i, Initializer::Incremental) }
                    }
                    (Ok(g), Err(_)) => (g, Initializer::Global),
                    (Err(_), inc) => (inc?, Initializer::Incremental),
                }
            }
        },
    };
    log(format!("initializer: {initializer:?}, {} registered, rms {:.3} px", sv.registered(), sv.rms));
    sv.retriangulate(cfg.retriangulate, &log);

    // ---- 6. the lens model ----
    let (cams, cam_sensor, obs) = sv.problem();
    let cam_poses: Vec<Pose> = cams.iter().map(|&i| sv.poses[i].unwrap()).collect();
    let xs: Vec<V3> = sv.points.iter().map(|p| p.xyz).collect();
    let anchor = cams.iter().position(|&i| i == sv.seed.0);
    let scale = cams.iter().position(|&i| i == sv.seed.1);
    let lcfg = LensCfg {
        choice: cfg.lens,
        square_pixels: cfg.square_pixels,
        full_model_views: cfg.full_model_views,
        ba: BaCfg { iters: 100, anchor, scale, ..bcfg.clone() },
    };
    let fits = fit_lenses(&sv.ks, &cam_sensor, &cam_poses, &xs, &obs, &lcfg);
    let best = fits.iter().enumerate().min_by(|a, b| a.1.fit.bic.total_cmp(&b.1.fit.bic)).map(|(i, _)| i).unwrap();
    for (i, c) in fits.iter().enumerate() {
        let f = &c.fit;
        log(format!(
            "lens {:<8} rms {:.4} px, {} parameters, BIC {:.1}{}",
            format!("{:?}", f.choice),
            f.rms_px,
            f.params,
            f.bic,
            if i == best { "  <- chosen" } else { "" }
        ));
        for (s, k) in f.intrinsics.iter().enumerate() {
            log(format!("    sensor {s}: {}", describe(k)));
        }
    }
    let chosen = &fits[best];
    sv.ks = chosen.fit.intrinsics.clone();
    sv.lens = chosen.fit.choice;
    for (c, &i) in cams.iter().enumerate() {
        sv.poses[i] = Some(chosen.poses[c]);
    }
    for (p, x) in sv.points.iter_mut().zip(&chosen.points) {
        p.xyz = *x;
    }
    // re-triangulate every track through the chosen lens: observations the
    // radial camera could not explain may now be inliers
    sv.retriangulate(cfg.retriangulate.max(1), &log);
    let lens = sv.lens;
    let seed = sv.seed;
    let Solver { ks, mut poses, mut points, rms, .. } = sv;
    for p in &mut points {
        let mut c = [0.0f32; 3];
        for &(i, kp) in &p.obs {
            let f = prep.feats[i].0[kp];
            let (w, h) = (photos[i].width, photos[i].height);
            let (x, y) = ((f.x as u32).min(w - 1), (f.y as u32).min(h - 1));
            let o = ((y * w + x) * 3) as usize;
            for (ch, v) in c.iter_mut().enumerate() {
                *v += photos[i].rgb[o + ch] as f32 / 255.0;
            }
        }
        p.rgb = c.map(|v| v / p.obs.len() as f32);
    }

    // ---- 7. the world: gravity, and metres where the fixes allow ----
    let held: Vec<Pose> = poses.iter().flatten().copied().collect();
    let gravity = gravity_from_cameras(&held);
    if let Some(g) = &gravity {
        log(format!("gravity from {} of {} cameras, held level to {:.1} deg rms", g.inliers.iter().filter(|&&b| b).count(), held.len(), g.spread_deg));
    }
    let mut up = gravity.map(|g| g.up);
    let mut gauge = Gauge::Seed { anchor: seed.0, scale: seed.1 };
    let mut georef_refused = None;
    let fixes: Vec<Option<Wgs84>> = photos.iter().map(|p| p.gps).collect();
    if fixes.iter().any(|f| f.is_some()) {
        let centres: Vec<Option<V3>> = poses.iter().map(|p| p.map(|p| p.centre())).collect();
        match solve_enu(&centres, &fixes, up, &cfg.georef) {
            Ok(fit) => {
                let agreed = fit.inliers.iter().filter(|&&b| b).count();
                log(format!("georeferenced: {agreed} fixes agree to {:.2} m rms, {:.4} m per unit", fit.rms_m, fit.sim3.s));
                for p in poses.iter_mut().flatten() {
                    *p = fit.sim3.apply_pose(p);
                }
                for p in &mut points {
                    p.xyz = fit.sim3.apply(p.xyz);
                }
                gauge = Gauge::Enu { origin: fit.origin, rms_m: fit.rms_m, fixes: agreed };
                up = Some([0.0, 0.0, 1.0]);
            }
            Err(e) => {
                log(format!("satellite fixes not used: {e}"));
                georef_refused = Some(e);
            }
        }
    }
    let solve_s = clock.elapsed().as_secs_f64();
    let report = SfmReport { pairs_matched: candidates.len(), pairs_verified: prep.pairs.len(), initializer, features_s, matching_s, focal_s, solve_s, georef_refused };
    Ok(Reconstruction {
        intrinsics: ks,
        sensor: prep.sensor,
        poses,
        points,
        keypoints: prep.feats.into_iter().map(|f| f.0).collect(),
        rms_px: rms,
        lens,
        lens_fits: fits.into_iter().map(|c| c.fit).collect(),
        gauge,
        up,
        report,
    })
}

/// Photographs in the largest connected part of the verified view graph:
/// what any initializer can hope to register.
fn largest_component(n: usize, pairs: &[Pair]) -> usize {
    let mut parent: Vec<usize> = (0..n).collect();
    for (a, b, _) in pairs {
        let (x, y) = (uf_find(&mut parent, *a), uf_find(&mut parent, *b));
        if x != y {
            parent[x] = y;
        }
    }
    let mut size = vec![0usize; n];
    for v in 0..n {
        size[uf_find(&mut parent, v)] += 1;
    }
    size.into_iter().max().unwrap_or(0)
}

/// Most observations the global positioning problem is given.
const POSITIONING_OBSERVATIONS: usize = 400_000;

/// The angular inlier threshold for a pair of images: the pixel threshold
/// through each image's focal length, averaged - the Sampson error spreads
/// a mismatch over both rays.
fn pair_threshold(px: f64, a: &Intrinsics, b: &Intrinsics) -> f64 {
    0.5 * px * (1.0 / a.fx + 1.0 / b.fx)
}

/// One incremental solve: the registered cameras, the points and the
/// calibration they are expressed through.
struct Solver<'a> {
    prep: &'a Prepared,
    cfg: &'a SfmCfg,
    ks: Vec<Intrinsics>,
    /// Whether bundle adjustment may move the calibration.
    free: bool,
    /// The model the calibration is written in, which decides what its
    /// stages free.
    lens: LensChoice,
    bcfg: BaCfg,
    poses: Vec<Option<Pose>>,
    points: Vec<Point>,
    point_of: Vec<Option<usize>>,
    rays: Rays,
    seed: (usize, usize),
    rms: f64,
    /// Largest reprojection error an observation may keep, pixels: the
    /// configured one, or looser while a global start settles.
    max_err: f64,
}

impl<'a> Solver<'a> {
    /// Choose the seed pair under calibration `ks`, triangulate it and
    /// adjust it. `free` lets bundle adjustment refine the calibration from
    /// here on.
    fn seed(prep: &'a Prepared, ks: Vec<Intrinsics>, free: bool, bcfg: BaCfg, cfg: &'a SfmCfg, log: &dyn Fn(String)) -> Result<Solver<'a>, SfmError> {
        let rays = bearings(&prep.feats, &prep.sensor, &ks);
        // Many points AND a wide baseline: a narrow pair triangulates poorly
        // and everything registered onto it inherits the error. Score by
        // points times baseline, saturating at 10 degrees.
        let mut order: Vec<usize> = (0..prep.pairs.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(prep.pairs[i].2.len()));
        let mut seed: Option<(usize, usize, Pose, f64)> = None;
        for &pi in order.iter().take(30) {
            let (a, b, m) = &prep.pairs[pi];
            let m: Vec<(usize, usize)> = m.iter().copied().filter(|&(i, j)| rays[*a][i].is_some() && rays[*b][j].is_some()).collect();
            let xa: Vec<V3> = m.iter().map(|&(i, _)| rays[*a][i].unwrap()).collect();
            let xb: Vec<V3> = m.iter().map(|&(_, j)| rays[*b][j].unwrap()).collect();
            let thresh = pair_threshold(cfg.thresh_px, &ks[prep.sensor[*a]], &ks[prep.sensor[*b]]);
            let Some((e, mask)) = ransac_essential(&xa, &xb, thresh, cfg.ransac_iters, 7) else { continue };
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
        let mut poses = vec![None; prep.n];
        poses[sa] = Some(Pose::identity());
        poses[sb] = Some(sp);
        let lens = LensChoice::of(&ks[0].lens);
        let mut sv = Solver {
            prep,
            cfg,
            ks,
            free,
            lens,
            bcfg,
            poses,
            points: Vec::new(),
            point_of: vec![None; prep.tracks.len()],
            rays,
            seed: (sa, sb),
            rms: 0.0,
            max_err: cfg.max_error_px,
        };
        let added = sv.triangulate_new();
        sv.rms = sv.adjust(30, Stage::Full);
        log(format!("seed pair {sa}-{sb}: {added} points triangulated, {} kept, rms {:.3} px", sv.points.len(), sv.rms));
        Ok(sv)
    }

    /// The global initializer under calibration `ks` (freed from here on):
    /// every pair's relative rotation through `ks`, the rotations averaged,
    /// the camera centres and points positioned from the bearings, then the
    /// tracks triangulated and bundle-adjusted under an error bound that
    /// tightens to the configured one. Views the start placed badly lose
    /// their observations to that bound and are handed back unregistered.
    fn global(prep: &'a Prepared, ks: Vec<Intrinsics>, bcfg: BaCfg, cfg: &'a SfmCfg, log: &dyn Fn(String)) -> Result<Solver<'a>, SfmError> {
        let rays = bearings(&prep.feats, &prep.sensor, &ks);
        let rel: Vec<Option<RelativeRotation>> = backend_cpu::par::map(prep.pairs.len(), |pi| {
            let (a, b, m) = &prep.pairs[pi];
            let m: Vec<(usize, usize)> = m.iter().copied().filter(|&(i, j)| rays[*a][i].is_some() && rays[*b][j].is_some()).collect();
            let xa: Vec<V3> = m.iter().map(|&(i, _)| rays[*a][i].unwrap()).collect();
            let xb: Vec<V3> = m.iter().map(|&(_, j)| rays[*b][j].unwrap()).collect();
            let thresh = pair_threshold(cfg.thresh_px, &ks[prep.sensor[*a]], &ks[prep.sensor[*b]]);
            let (e, mask) = ransac_essential(&xa, &xb, thresh, cfg.ransac_iters, (a * prep.n + b) as u64 + 3)?;
            let (pose, pts) = relative_pose(&e, &xa, &xb, &mask);
            let good = pts.iter().flatten().count();
            (good >= cfg.min_inliers).then_some(RelativeRotation { a: *a, b: *b, r: pose.r, weight: good as f64 })
        });
        let edges: Vec<RelativeRotation> = rel.into_iter().flatten().collect();
        let averaged = average_rotations(prep.n, &edges, &cfg.rotation);
        let outliers = averaged.inlier.iter().filter(|&&ok| !ok).count();
        let cams: Vec<usize> = (0..prep.n).filter(|&i| averaged.rotations[i].is_some()).collect();
        log(format!("rotation averaging: {} of {} pairs agree, {} photographs oriented", edges.len() - outliers, edges.len(), cams.len()));
        if cams.len() < 3 {
            return Err(SfmError::WeakViewGraph { cameras: cams.len() });
        }
        let mut local = vec![usize::MAX; prep.n];
        for (c, &i) in cams.iter().enumerate() {
            local[i] = c;
        }
        let rots: Vec<crate::linalg::M3> = cams.iter().map(|&i| averaged.rotations[i].unwrap()).collect();
        // the longest tracks carry the most constraint per unknown; cap the
        // problem so a large capture positions in bounded time
        let mut order: Vec<usize> = (0..prep.tracks.len()).collect();
        order.sort_by_key(|&t| std::cmp::Reverse(prep.tracks[t].len()));
        let mut obs = Vec::new();
        let mut npoints = 0;
        for t in order {
            let seen: Vec<RayObservation> = prep.tracks[t]
                .iter()
                .filter_map(|&(i, kp)| (local[i] != usize::MAX).then_some(()).and(rays[i][kp]).map(|ray| RayObservation { cam: local[i], point: npoints, ray }))
                .collect();
            if seen.len() < 2 {
                continue;
            }
            obs.extend(seen);
            npoints += 1;
            if obs.len() >= POSITIONING_OBSERVATIONS {
                break;
            }
        }
        let placed = global_positions(&rots, npoints, &obs, &cfg.positioning);
        let agree = placed.inlier.iter().filter(|&&ok| ok).count();
        log(format!("global positioning: {npoints} points, {agree} of {} observations along their rays", obs.len()));
        let mut poses = vec![None; prep.n];
        for (c, &i) in cams.iter().enumerate() {
            if let Some(centre) = placed.centres[c] {
                poses[i] = Some(Pose { r: rots[c], t: scale(mv(&rots[c], centre), -1.0) });
            }
        }
        let lens = LensChoice::of(&ks[0].lens);
        let mut sv = Solver {
            prep,
            cfg,
            ks,
            free: true,
            lens,
            bcfg,
            poses,
            points: Vec::new(),
            point_of: vec![None; prep.tracks.len()],
            rays,
            seed: (0, 0),
            rms: 0.0,
            max_err: cfg.max_error_px,
        };
        sv.pick_seed()?;
        for loosen in [8.0, 2.0, 1.0] {
            sv.max_err = cfg.max_error_px * loosen;
            // bearings through the calibration the last round refined
            sv.rays = bearings(&prep.feats, &prep.sensor, &sv.ks);
            sv.points.clear();
            sv.point_of.iter_mut().for_each(|v| *v = None);
            let added = sv.triangulate_new();
            sv.rms = sv.adjust(30, Stage::Full);
            log(format!("  global round x{loosen}: {added} triangulated, {} kept, {} views, rms {:.3} px, {}", sv.points.len(), sv.registered(), sv.rms, describe(&sv.ks[0])));
            sv.drop_weak_views();
            sv.pick_seed()?;
        }
        log(format!("global start: {} photographs, {} points, rms {:.3} px", sv.registered(), sv.points.len(), sv.rms));
        Ok(sv)
    }

    /// The gauge pair of a start that did not grow from one: the registered
    /// pair with the most verified inliers.
    fn pick_seed(&mut self) -> Result<(), SfmError> {
        let best = self.prep.pairs.iter().filter(|(a, b, _)| self.poses[*a].is_some() && self.poses[*b].is_some()).max_by_key(|(_, _, m)| m.len());
        let (a, b, _) = best.ok_or(SfmError::NoSeedPair)?;
        self.seed = (*a, *b);
        Ok(())
    }

    /// Unregister the views left with fewer observations than registration
    /// needs, and the points they leave with fewer than two.
    fn drop_weak_views(&mut self) {
        let mut count = vec![0usize; self.prep.n];
        for p in &self.points {
            for &(i, _) in &p.obs {
                count[i] += 1;
            }
        }
        let weak: Vec<bool> = (0..self.prep.n).map(|i| self.poses[i].is_some() && count[i] < self.cfg.min_register).collect();
        if !weak.iter().any(|&w| w) {
            return;
        }
        for (i, w) in weak.iter().enumerate() {
            if *w {
                self.poses[i] = None;
            }
        }
        for p in &mut self.points {
            p.obs.retain(|&(i, _)| !weak[i]);
        }
        self.points.retain(|p| p.obs.len() >= 2);
        self.point_of.iter_mut().for_each(|v| *v = None);
        for (pi, p) in self.points.iter().enumerate() {
            let (i, kp) = p.obs[0];
            if let Some(t) = self.prep.track_of[i][kp] {
                self.point_of[t] = Some(pi);
            }
        }
    }

    fn registered(&self) -> usize {
        self.poses.iter().flatten().count()
    }

    /// Triangulate every track seen by two registered views that is not a
    /// point yet: every supporting view must see it ahead along its ray,
    /// close to where it was detected, and some pair of them from far
    /// enough apart.
    fn triangulate_new(&mut self) -> usize {
        let min_angle = self.cfg.min_angle_deg.to_radians();
        let mut added = 0;
        for (ti, t) in self.prep.tracks.iter().enumerate() {
            if self.point_of[ti].is_some() {
                continue;
            }
            let seen: Vec<(usize, usize)> = t.iter().copied().filter(|(i, kp)| self.poses[*i].is_some() && self.rays[*i][*kp].is_some()).collect();
            if seen.len() < 2 {
                continue;
            }
            let ps: Vec<&Pose> = seen.iter().map(|(i, _)| self.poses[*i].as_ref().unwrap()).collect();
            let us: Vec<V3> = seen.iter().map(|(i, kp)| self.rays[*i][*kp].unwrap()).collect();
            let Some(x) = triangulate(&ps, &us) else { continue };
            let mut ok = true;
            let mut best_angle = 0.0f64;
            for (j, p) in ps.iter().enumerate() {
                let (img, kp) = seen[j];
                let c = p.to_cam(x);
                let near = dot(us[j], c) > 0.0
                    && self.ks[self.prep.sensor[img]].project(c).is_some_and(|uv| {
                        let f = self.prep.feats[img].0[kp];
                        (uv[0] - f.x as f64).hypot(uv[1] - f.y as f64) <= self.max_err
                    });
                if !near {
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
            self.point_of[ti] = Some(self.points.len());
            self.points.push(Point { xyz: x, rgb: [0.0; 3], obs: seen });
            added += 1;
        }
        added
    }

    /// The bundle-adjustment problem of the current state: the registered
    /// images (in input order), each one's sensor, and every observation
    /// with cameras numbered in that order.
    fn problem(&self) -> (Vec<usize>, Vec<usize>, Vec<Observation>) {
        let cams: Vec<usize> = (0..self.prep.n).filter(|&i| self.poses[i].is_some()).collect();
        let mut cam_ix = vec![usize::MAX; self.prep.n];
        for (c, &i) in cams.iter().enumerate() {
            cam_ix[i] = c;
        }
        let cam_sensor = cams.iter().map(|&i| self.prep.sensor[i]).collect();
        let obs = self
            .points
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| p.obs.iter().map(move |&(i, kp)| (pi, i, kp)))
            .map(|(pi, i, kp)| {
                let f = self.prep.feats[i].0[kp];
                Observation { cam: cam_ix[i], point: pi, px: [f.x as f64, f.y as f64] }
            })
            .collect();
        (cams, cam_sensor, obs)
    }

    /// Bundle-adjust everything - poses and points alone first, then with
    /// each sensor's calibration freed as far as its views support (capped
    /// at `cap`) - then drop the observations the adjustment cannot explain
    /// and the points left with fewer than two. Returns the RMS error.
    fn adjust(&mut self, iters: usize, cap: Stage) -> f64 {
        let (cams, cam_sensor, obs) = self.problem();
        let mut cp: Vec<Pose> = cams.iter().map(|&i| self.poses[i].unwrap()).collect();
        let mut xs: Vec<V3> = self.points.iter().map(|p| p.xyz).collect();
        let anchor = cams.iter().position(|&i| i == self.seed.0);
        let scale = cams.iter().position(|&i| i == self.seed.1);
        let base = BaCfg { anchor, scale, ..self.bcfg.clone() };
        let mut rep = bundle_adjust(&mut self.ks, &cam_sensor, &mut cp, &mut xs, &obs, &BaCfg { iters: (iters / 3).max(5), free: Vec::new(), ..base.clone() });
        if self.free {
            let views: Vec<usize> = (0..self.ks.len()).map(|s| cam_sensor.iter().filter(|&&x| x == s).count()).collect();
            let free: Vec<Vec<Param>> = views.iter().map(|&v| self.lens.params(stage_for(v, self.cfg.full_model_views).min(cap), self.cfg.square_pixels)).collect();
            if free.iter().any(|f| !f.is_empty()) {
                rep = bundle_adjust(&mut self.ks, &cam_sensor, &mut cp, &mut xs, &obs, &BaCfg { iters, free, ..base });
            }
        }
        for (c, &i) in cams.iter().enumerate() {
            self.poses[i] = Some(cp[c]);
        }
        let max_err = self.max_err;
        let mut keep: Vec<Point> = Vec::new();
        for (pi, mut p) in self.points.drain(..).enumerate() {
            p.xyz = xs[pi];
            p.obs.retain(|&(i, kp)| {
                let f = self.prep.feats[i].0[kp];
                crate::camera::project(&self.ks[self.prep.sensor[i]], self.poses[i].as_ref().unwrap(), p.xyz)
                    .is_some_and(|uv| (uv[0] - f.x as f64).hypot(uv[1] - f.y as f64) <= max_err)
            });
            if p.obs.len() >= 2 {
                keep.push(p);
            }
        }
        self.points = keep;
        self.point_of.iter_mut().for_each(|v| *v = None);
        for (pi, p) in self.points.iter().enumerate() {
            let (i, kp) = p.obs[0];
            if let Some(t) = self.prep.track_of[i][kp] {
                self.point_of[t] = Some(pi);
            }
        }
        rep.rms_after
    }

    /// Register the rest, the image that sees the most points first.
    fn register_all(&mut self, log: &dyn Fn(String)) {
        let (prep, cfg) = (self.prep, self.cfg);
        let n = prep.n;
        let mut failed = vec![false; n];
        loop {
            self.rays = bearings(&prep.feats, &prep.sensor, &self.ks);
            let mut best: Option<(usize, usize)> = None;
            for (i, _) in failed.iter().enumerate().filter(|(i, f)| !**f && self.poses[*i].is_none()) {
                let c = (0..prep.feats[i].0.len()).filter(|&kp| self.rays[i][kp].is_some() && prep.track_of[i][kp].and_then(|t| self.point_of[t]).is_some()).count();
                if best.is_none_or(|(_, bc)| c > bc) {
                    best = Some((i, c));
                }
            }
            let Some((img, count)) = best else { break };
            if count < cfg.min_register {
                failed[img] = true;
                continue;
            }
            let corr: Vec<(usize, usize, V3)> = (0..prep.feats[img].0.len())
                .filter_map(|kp| {
                    let ray = self.rays[img][kp]?;
                    prep.track_of[img][kp].and_then(|t| self.point_of[t]).map(|pi| (kp, pi, ray))
                })
                .collect();
            let xs: Vec<V3> = corr.iter().map(|c| self.points[c.1].xyz).collect();
            let us: Vec<V3> = corr.iter().map(|c| c.2).collect();
            let k = &self.ks[prep.sensor[img]];
            let reg = crate::pnp::ransac_pnp(&xs, &us, cfg.thresh_px / k.fx, cfg.ransac_iters, img as u64 + 99);
            let Some((pose, mask)) = reg.filter(|(_, m)| m.iter().filter(|&&v| v).count() >= cfg.min_register) else {
                failed[img] = true;
                log(format!("image {img}: registration failed ({count} candidate correspondences)"));
                continue;
            };
            self.poses[img] = Some(pose);
            let inliers = mask.iter().filter(|&&v| v).count();
            for (c, ok) in corr.iter().zip(&mask) {
                if *ok {
                    self.points[c.1].obs.push((img, c.0));
                }
            }
            let added = self.triangulate_new();
            self.rms = self.adjust(20, Stage::Full);
            let s = prep.sensor[img];
            log(format!(
                "image {img} (sensor {s}): {inliers}/{} inliers, +{added} points, {} total, rms {:.3} px, {}",
                corr.len(),
                self.points.len(),
                self.rms,
                describe(&self.ks[s])
            ));
            // a failure may succeed once more of the scene exists
            failed.iter_mut().for_each(|f| *f = false);
        }
        for (i, p) in self.poses.iter().enumerate() {
            if p.is_none() {
                log(format!("image {i}: not registered"));
            }
        }
    }

    /// Points triangulated early were triangulated through the calibration
    /// of the moment. Re-triangulate every track through the current camera
    /// and adjust again, until every focal length holds still.
    fn retriangulate(&mut self, rounds: usize, log: &dyn Fn(String)) {
        for round in 0..rounds {
            let before: Vec<f64> = self.ks.iter().map(|k| k.fx).collect();
            self.rays = bearings(&self.prep.feats, &self.prep.sensor, &self.ks);
            self.points.clear();
            self.point_of.iter_mut().for_each(|v| *v = None);
            self.triangulate_new();
            self.rms = self.adjust(100, Stage::Full);
            log(format!("re-triangulation {round}: {} points, rms {:.4} px", self.points.len(), self.rms));
            for (s, k) in self.ks.iter().enumerate() {
                log(format!("    sensor {s}: {}", describe(k)));
            }
            if self.ks.iter().zip(&before).all(|(k, b)| (k.fx - b).abs() < 1e-3 * b) {
                break;
            }
        }
    }
}
