// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which views a reference view is matched against, and what range of
//! distances its scene lies at - both read off the sparse structure-from-
//! motion tracks, before any pixel is looked at.
//!
//! Source views. Every candidate view `j` of reference `i` is scored by the
//! tracks the two share (Goesele, Snavely, Curless, Hoppe, Seitz, "Multi-View
//! Stereo for Community Photo Collections", ICCV 2007, §4.1: shared features
//! weighted by the triangulation angle they are seen under, and by how
//! similar the two views' sampling of them is), times the fraction of the
//! reference image the shared tracks spread over (a view that shares many
//! tracks in one corner helps one corner). The angle weight is 1 inside
//! `[min_angle, max_angle]` (5 to 60 degrees by default: below it depth is
//! ill-conditioned, above it the two views see different appearance and
//! different occlusions) and falls off quadratically outside. The best `K`
//! with enough shared tracks are kept.
//!
//! Range bounds. The ranges (distance from the camera centre - the quantity
//! the stereo searches, along each pixel's own unit ray) of the tracks a
//! view observes, between two quantiles and widened by a margin, bound the
//! random initialisation of its hypotheses.
//!
//! A [`Track`] with no observation list (a sparse cloud read back from a
//! `.ply`) is taken to be seen by every view it projects into - an
//! over-estimate that ignores occlusion, which only costs selection
//! precision, never correctness.
//!
//! Swedish Embedded AB implements multi-view reconstruction pipelines for its
//! clients. If your team needs photographs turned into dense, accurate
//! geometry, you can procure our services by sending an email to
//! info@swedishembedded.com.

use sfm::camera::Pose;
use sfm::linalg::{dot, norm, sub, V3};
use splat::types::Camera;

use crate::pose_of;

/// A sparse scene point and the views that observe it.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub xyz: [f64; 3],
    /// Indices of the views that observe it; empty = unknown, then every
    /// view it projects into counts.
    pub views: Vec<usize>,
}

/// Tracks of a structure-from-motion reconstruction, re-indexed to the
/// views MVS runs on: `view_of[image]` is the view index of input image
/// `image`, `None` for an image left out (unregistered).
pub fn tracks_from_sfm(rec: &sfm::incremental::Reconstruction, view_of: &[Option<usize>]) -> Vec<Track> {
    rec.points
        .iter()
        .map(|p| {
            let mut views: Vec<usize> = p.obs.iter().filter_map(|&(img, _)| view_of.get(img).copied().flatten()).collect();
            views.sort_unstable();
            views.dedup();
            Track { xyz: p.xyz, views }
        })
        .collect()
}

/// How source views are chosen.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectCfg {
    /// Most source views per reference.
    pub max_sources: usize,
    /// Fewest tracks a source must share with the reference.
    pub min_shared: usize,
    /// Triangulation angles, degrees, that get full weight.
    pub min_angle_deg: f64,
    pub max_angle_deg: f64,
    /// Quantiles of the observed track ranges that bound a view's depth.
    pub range_quantiles: (f64, f64),
    /// Widening of those bounds: `[lo / (1 + m), hi * (1 + m)]`.
    pub range_margin: f64,
}

impl Default for SelectCfg {
    fn default() -> Self {
        SelectCfg {
            max_sources: 8,
            min_shared: 12,
            min_angle_deg: 5.0,
            max_angle_deg: 60.0,
            range_quantiles: (0.02, 0.98),
            range_margin: 0.3,
        }
    }
}

/// A view's pose and calibration, in f64.
struct Frame {
    pose: Pose,
    c: V3,
    k: camera::Intrinsics,
}

impl Frame {
    fn of(cam: &Camera) -> Frame {
        let pose = pose_of(cam);
        Frame { c: pose.centre(), pose, k: cam.intrinsics() }
    }

    /// Pixel of world point `x`, when the lens images it inside the frame.
    fn pixel(&self, x: V3) -> Option<[f64; 2]> {
        let p = self.k.project(self.pose.to_cam(x))?;
        (p[0] >= 0.0 && p[1] >= 0.0 && p[0] < self.k.width as f64 && p[1] < self.k.height as f64).then_some(p)
    }
}

/// Per view, the indices of the tracks it sees.
fn visibility(cams: &[Camera], tracks: &[Track]) -> Vec<Vec<usize>> {
    let frames: Vec<Frame> = cams.iter().map(Frame::of).collect();
    let mut seen = vec![Vec::new(); cams.len()];
    for (t, tr) in tracks.iter().enumerate() {
        if tr.views.is_empty() {
            for (v, f) in frames.iter().enumerate() {
                if f.pixel(tr.xyz).is_some() {
                    seen[v].push(t);
                }
            }
        } else {
            for &v in &tr.views {
                if v < cams.len() {
                    seen[v].push(t);
                }
            }
        }
    }
    seen
}

/// Weight of a triangulation angle (degrees): 1 inside `[lo, hi]`, falling
/// off quadratically below and above.
fn angle_weight(deg: f64, lo: f64, hi: f64) -> f64 {
    if deg < lo {
        (deg / lo).powi(2)
    } else if deg > hi {
        (hi / deg).powi(2)
    } else {
        1.0
    }
}

/// Grid over the reference image the overlap is measured on.
const OVERLAP_GRID: usize = 8;

/// The source views of every view, best first. A view that shares fewer
/// than `min_shared` tracks with every other gets no sources (and so no
/// depth map).
pub fn select_sources(cams: &[Camera], tracks: &[Track], cfg: &SelectCfg) -> Vec<Vec<usize>> {
    let frames: Vec<Frame> = cams.iter().map(Frame::of).collect();
    let seen = visibility(cams, tracks);
    let mut in_view = vec![vec![false; tracks.len()]; cams.len()];
    for (v, ts) in seen.iter().enumerate() {
        for &t in ts {
            in_view[v][t] = true;
        }
    }
    (0..cams.len())
        .map(|i| {
            let fi = &frames[i];
            // grid cells of the reference each of its tracks falls in
            let cell = |x: [f64; 3]| {
                fi.pixel(x).map(|p| {
                    let gx = ((p[0] / fi.k.width as f64) * OVERLAP_GRID as f64) as usize;
                    let gy = ((p[1] / fi.k.height as f64) * OVERLAP_GRID as f64) as usize;
                    gy.min(OVERLAP_GRID - 1) * OVERLAP_GRID + gx.min(OVERLAP_GRID - 1)
                })
            };
            let mut ref_cells = [false; OVERLAP_GRID * OVERLAP_GRID];
            for &t in &seen[i] {
                if let Some(c) = cell(tracks[t].xyz) {
                    ref_cells[c] = true;
                }
            }
            let ref_cell_count = ref_cells.iter().filter(|c| **c).count().max(1);
            let mut scored: Vec<(f64, usize)> = Vec::new();
            for (j, fj) in frames.iter().enumerate() {
                if j == i {
                    continue;
                }
                let mut shared = 0usize;
                let mut score = 0.0;
                let mut cells = [false; OVERLAP_GRID * OVERLAP_GRID];
                for &t in &seen[i] {
                    if !in_view[j][t] {
                        continue;
                    }
                    let x = tracks[t].xyz;
                    let (a, b) = (sub(x, fi.c), sub(x, fj.c));
                    let (na, nb) = (norm(a), norm(b));
                    if na <= 0.0 || nb <= 0.0 {
                        continue;
                    }
                    shared += 1;
                    let deg = (dot(a, b) / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees();
                    // sampling similarity: the ratio of the two footprints
                    let (fa, fb) = (na / fi.k.fx.abs().max(1e-9), nb / fj.k.fx.abs().max(1e-9));
                    let ratio = (fa / fb).min(fb / fa);
                    let scale = (2.0 * ratio).min(1.0);
                    score += angle_weight(deg, cfg.min_angle_deg, cfg.max_angle_deg) * scale;
                    if let Some(c) = cell(x) {
                        cells[c] = true;
                    }
                }
                if shared < cfg.min_shared {
                    continue;
                }
                let overlap = cells.iter().filter(|c| **c).count() as f64 / ref_cell_count as f64;
                scored.push((score * overlap, j));
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            scored.into_iter().filter(|s| s.0 > 0.0).take(cfg.max_sources).map(|s| s.1).collect()
        })
        .collect()
}

/// Per view, the range interval `[near, far]` its hypotheses are drawn
/// from: quantiles of the ranges of the tracks it sees, widened by the
/// margin. `None` for a view that sees fewer than 3 tracks in front of it.
pub fn range_bounds(cams: &[Camera], tracks: &[Track], cfg: &SelectCfg) -> Vec<Option<(f32, f32)>> {
    let frames: Vec<Frame> = cams.iter().map(Frame::of).collect();
    let seen = visibility(cams, tracks);
    frames
        .iter()
        .zip(&seen)
        .map(|(f, ts)| {
            let mut r: Vec<f64> = ts
                .iter()
                .filter_map(|&t| {
                    let x = f.pose.to_cam(tracks[t].xyz);
                    // a perspective lens sees only the half-space ahead
                    (!f.k.lens.is_perspective() || x[2] > 0.0).then(|| norm(x)).filter(|r| *r > 0.0)
                })
                .collect();
            if r.len() < 3 {
                return None;
            }
            r.sort_by(f64::total_cmp);
            let q = |p: f64| r[((r.len() - 1) as f64 * p).round() as usize];
            let (lo, hi) = (q(cfg.range_quantiles.0), q(cfg.range_quantiles.1));
            Some(((lo / (1.0 + cfg.range_margin)) as f32, (hi * (1.0 + cfg.range_margin)) as f32))
        })
        .collect()
}
