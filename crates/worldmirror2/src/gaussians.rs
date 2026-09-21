// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side output assembly: camera decode (`camera_utils.py` /
//! `rotation.py` parity) and per-pixel Gaussian construction
//! (`rasterization.py::prepare_splats` + `act_gs.py` parity).
//!
//! Per source pixel: means = gs_depth back-projected through the predicted
//! camera (pixel-index grid, no +0.5 — reference convention); quat wxyz
//! normalized; scales exp-clamped at 0.3; opacity sigmoid; color =
//! SH-decode(residual + RGB2SH(rgb)) = residual·C0 + rgb.

use gpu_core::Gpu;
use splat::ply::SH_C0;
use splat::types::{Camera, Splats};

use crate::model::{Head, Mirror};

/// Decode raw camera 9-vectors `[s,9]` (fov relu applied here) into cameras.
pub fn decode_cameras(raw: &[f32], s: usize, width: u32, height: u32) -> Vec<Camera> {
    let (w, h) = (width as f32, height as f32);
    (0..s)
        .map(|i| {
            let v = &raw[i * 9..i * 9 + 9];
            let (t, q) = (&v[0..3], &v[3..7]);
            let fov_v = v[7].max(0.0);
            let fov_u = v[8].max(0.0);
            // scalar-last xyzw quaternion -> w2c rotation
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-12);
            let (x, y, z, r) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
            let rm = [
                1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * r), 2.0 * (x * z + y * r),
                2.0 * (x * y + z * r), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * r),
                2.0 * (x * z - y * r), 2.0 * (y * z + x * r), 1.0 - 2.0 * (x * x + y * y),
            ];
            // c2w = rigid inverse of [R|t]
            let rt = [rm[0], rm[3], rm[6], rm[1], rm[4], rm[7], rm[2], rm[5], rm[8]];
            let ct = [
                -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
                -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
                -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
            ];
            let c2w = [
                rt[0], rt[1], rt[2], ct[0],
                rt[3], rt[4], rt[5], ct[1],
                rt[6], rt[7], rt[8], ct[2],
                0.0, 0.0, 0.0, 1.0,
            ];
            Camera {
                c2w,
                fx: 0.5 * w / (0.5 * fov_u).tan().max(1e-6),
                fy: 0.5 * h / (0.5 * fov_v).tan().max(1e-6),
                cx: 0.5 * w,
                cy: 0.5 * h,
                width,
                height,
            }
        })
        .collect()
}

/// Reconcile per-view depth maps against each other, in place.
///
/// Each frame's depth is individually good and they disagree with each other.
/// The disagreement is along the viewing ray, which is the one direction it is
/// invisible from the frame that made it, and the one direction a fit cannot
/// correct - a splat's image-plane gradient is orthogonal to its own ray. So
/// it has to be settled here, before the depths become geometry.
///
/// For every pixel of every frame, the point it implies is projected into
/// every other frame. Where that frame's own depth agrees to within `rtol`,
/// the two are looking at the same surface, and its opinion - expressed back
/// along THIS pixel's ray, so the pixel keeps its own direction and only its
/// distance changes - joins a confidence-weighted average. Where the depths
/// disagree by more than `rtol` they are looking at different surfaces, one
/// occluding the other, and averaging them would invent a surface that is in
/// neither; the same relative test silhouette rejection uses, for the same
/// reason.
///
/// `conf` is the depth head's second channel, which the model emits to say how
/// much each of its own depths is worth.
///
/// Returns, per frame and pixel, how many OTHER frames agreed that a surface
/// is there - the count the fused depth was averaged over.
///
/// That count is the honest measure of whether a piece of geometry is real.
/// A pixel a dozen views agree about is a surface; one that stands alone is a
/// guess, and the places where a depth map guesses are exactly the places it
/// cannot see properly - thin structures, silhouettes, anything dark or
/// specular. Those guesses are individually consistent with the view that made
/// them, so they project correctly onto every training image and the optimizer
/// never learns they are wrong. They are only visible as smears from a
/// direction nothing was fitted against, which is precisely why they have to
/// be rejected on agreement rather than on appearance.
pub fn fuse_depths(
    depth: &mut [Vec<f32>],
    conf: &[Vec<f32>],
    cams: &[Camera],
    width: u32,
    height: u32,
    rtol: f32,
) -> Vec<Vec<u16>> {
    let s = cams.len();
    if s < 2 || depth.len() != s || conf.len() != s {
        return vec![vec![0u16; (width * height) as usize]; depth.len()];
    }
    let (w, h) = (width as usize, height as usize);
    let hw = w * h;
    let views: Vec<[f32; 12]> = cams.iter().map(|c| c.viewmat()).collect();
    let out: Vec<(Vec<f32>, Vec<u16>)> = (0..s)
        .map(|i| {
            let ci = &cams[i];
            let mi = &ci.c2w;
            let mut fused = vec![0.0f32; hw];
            let mut support = vec![0u16; hw];
            for py in 0..h {
                for px in 0..w {
                    let k = py * w + px;
                    let zi = depth[i][k];
                    let mut acc = (conf[i][k] * zi) as f64;
                    let mut wsum = conf[i][k] as f64;
                    let mut agree = 0u16;
                    if zi <= 0.0 {
                        fused[k] = zi;
                        continue;
                    }
                    // the ray this pixel looks along, in world space
                    let (xc, yc) = ((px as f32 + 0.5 - ci.cx) / ci.fx, (py as f32 + 0.5 - ci.cy) / ci.fy);
                    let dir = [
                        mi[0] * xc + mi[1] * yc + mi[2],
                        mi[4] * xc + mi[5] * yc + mi[6],
                        mi[8] * xc + mi[9] * yc + mi[10],
                    ];
                    let eye = [mi[3], mi[7], mi[11]];
                    let p = [eye[0] + zi * dir[0], eye[1] + zi * dir[1], eye[2] + zi * dir[2]];
                    for j in 0..s {
                        if j == i {
                            continue;
                        }
                        let vj = &views[j];
                        let cj = &cams[j];
                        let zj = vj[8] * p[0] + vj[9] * p[1] + vj[10] * p[2] + vj[11];
                        if zj <= 1e-6 {
                            continue;
                        }
                        let xj = vj[0] * p[0] + vj[1] * p[1] + vj[2] * p[2] + vj[3];
                        let yj = vj[4] * p[0] + vj[5] * p[1] + vj[6] * p[2] + vj[7];
                        let (ux, uy) = (cj.fx * xj / zj + cj.cx, cj.fy * yj / zj + cj.cy);
                        if ux < 0.0 || uy < 0.0 || ux >= w as f32 || uy >= h as f32 {
                            continue;
                        }
                        let kj = (uy as usize).min(h - 1) * w + (ux as usize).min(w - 1);
                        let dj = depth[j][kj];
                        if dj <= 0.0 || ((dj - zj) / zj).abs() > rtol {
                            continue; // different surfaces, not a disagreement
                        }
                        // j's surface point, expressed as a distance along
                        // THIS pixel's ray: the pixel keeps its direction.
                        let mj = &cj.c2w;
                        let (xk, yk) = ((ux - cj.cx) / cj.fx, (uy - cj.cy) / cj.fy);
                        let dj3 = [
                            mj[0] * xk + mj[1] * yk + mj[2],
                            mj[4] * xk + mj[5] * yk + mj[6],
                            mj[8] * xk + mj[9] * yk + mj[10],
                        ];
                        let ej = [mj[3], mj[7], mj[11]];
                        let q = [ej[0] + dj * dj3[0], ej[1] + dj * dj3[1], ej[2] + dj * dj3[2]];
                        let along = (q[0] - eye[0]) * dir[0] + (q[1] - eye[1]) * dir[1] + (q[2] - eye[2]) * dir[2];
                        let dd = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
                        let cand = along / dd.max(1e-12);
                        if cand <= 0.0 {
                            continue;
                        }
                        acc += (conf[j][kj] * cand) as f64;
                        wsum += conf[j][kj] as f64;
                        agree += 1;
                    }
                    fused[k] = if wsum > 0.0 { (acc / wsum) as f32 } else { zi };
                    support[k] = agree;
                }
            }
            (fused, support)
        })
        .collect();
    let mut sup = Vec::with_capacity(s);
    for (d, (f, c)) in depth.iter_mut().zip(out) {
        *d = f;
        sup.push(c);
    }
    sup
}

/// The pointmap head's world-space prediction, re-expressed as a DEPTH along
/// each pixel's own viewing ray, plus its confidence.
///
/// Turning a point back into a depth looks like throwing information away, and
/// it is - deliberately. The component of a pointmap across the ray is a
/// disagreement with the camera the scene is being built on, and honouring it
/// would put a gaussian somewhere the pixel it came from does not look. What
/// is worth having from the second head is its opinion about DISTANCE, which
/// is the axis the gaussian branch is least certain about.
fn points_as_depth(pts: &[f32], cam: &Camera, width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
    let hw = width * height;
    let m = &cam.c2w;
    // world -> camera is the inverse of the rigid c2w: R^T (p - t)
    let t = [m[3], m[7], m[11]];
    let mut depth = vec![0.0f32; hw];
    let mut conf = vec![0.0f32; hw];
    for i in 0..hw {
        let p = [inv_log(pts[i]) - t[0], inv_log(pts[hw + i]) - t[1], inv_log(pts[2 * hw + i]) - t[2]];
        // third row of R^T is the camera's forward axis
        depth[i] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2];
        conf[i] = 1.0 + pts[3 * hw + i].exp();
    }
    (depth, conf)
}

/// The normals head, activated: unit vector plus confidence. Predicted in the
/// camera's frame, like the depth it accompanies, so it is rotated into world
/// here to be comparable with geometry-derived normals.
fn head_normals(norm: &[f32], cam: &Camera, width: usize, height: usize) -> Vec<[f32; 3]> {
    let hw = width * height;
    let m = &cam.c2w;
    (0..hw)
        .map(|i| {
            let v = [norm[i], norm[hw + i], norm[2 * hw + i]];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let v = if l > 1e-20 { [v[0] / l, v[1] / l, v[2] / l] } else { [0.0, 0.0, -1.0] };
            [
                m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
                m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
                m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
            ]
        })
        .collect()
}

/// Per-pixel surface normal in WORLD space, from the geometry the depth map
/// itself implies: the cross product of the two tangents between neighbouring
/// back-projected points.
///
/// Derived from the depth rather than read from the normals head, so it agrees
/// with where the gaussians are actually being put. A head-predicted normal
/// that disagreed with the depth would tilt a disc off the surface it is
/// supposed to lie in.
fn depth_normals(depth: &[f32], cam: &Camera, width: usize, height: usize) -> Vec<[f32; 3]> {
    let m = &cam.c2w;
    let cam_pt = |px: usize, py: usize| -> [f32; 3] {
        let z = depth[py * width + px];
        [(px as f32 + 0.5 - cam.cx) * z / cam.fx, (py as f32 + 0.5 - cam.cy) * z / cam.fy, z]
    };
    let mut out = vec![[0.0f32; 3]; width * height];
    for py in 0..height {
        for px in 0..width {
            let (x0, x1) = (px.saturating_sub(1), (px + 1).min(width - 1));
            let (y0, y1) = (py.saturating_sub(1), (py + 1).min(height - 1));
            let (a, b) = (cam_pt(x1, py), cam_pt(x0, py));
            let (c, d) = (cam_pt(px, y1), cam_pt(px, y0));
            let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let v = [c[0] - d[0], c[1] - d[1], c[2] - d[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let n = if len > 1e-20 { [n[0] / len, n[1] / len, n[2] / len] } else { [0.0, 0.0, -1.0] };
            // camera -> world (rotation only)
            out[py * width + px] = [
                m[0] * n[0] + m[1] * n[1] + m[2] * n[2],
                m[4] * n[0] + m[5] * n[1] + m[6] * n[2],
                m[8] * n[0] + m[9] * n[1] + m[10] * n[2],
            ];
        }
    }
    out
}

/// A unit quaternion (wxyz) whose THIRD local axis is `n`, with the other two
/// spanning the surface. Which two is arbitrary - the disc is symmetric in its
/// own plane - so this picks whichever reference axis `n` is least parallel to.
fn quat_with_z(n: [f32; 3]) -> [f32; 4] {
    let up = if n[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let mut t = [
        up[1] * n[2] - up[2] * n[1],
        up[2] * n[0] - up[0] * n[2],
        up[0] * n[1] - up[1] * n[0],
    ];
    let l = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt().max(1e-20);
    for v in t.iter_mut() {
        *v /= l;
    }
    let b = [
        n[1] * t[2] - n[2] * t[1],
        n[2] * t[0] - n[0] * t[2],
        n[0] * t[1] - n[1] * t[0],
    ];
    // columns (t, b, n) as a rotation matrix -> quaternion (Shepperd)
    let (m00, m01, m02) = (t[0], b[0], n[0]);
    let (m10, m11, m12) = (t[1], b[1], n[1]);
    let (m20, m21, m22) = (t[2], b[2], n[2]);
    let tr = m00 + m11 + m22;
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    };
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-20);
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Assembly filters. `min_opacity` drops near-transparent gaussians early;
/// `max_depth` clips runaway sky depths (0 = off).
pub struct AssembleOpts {
    pub min_opacity: f32,
    pub max_depth: f32,
    /// Keep a pixel only where the GS head's own validity mask, after a
    /// sigmoid, is above this. The reference thresholds at 0.5; `0` disables.
    pub gs_mask_threshold: f32,
    /// Relative depth tolerance for silhouette rejection. A pixel whose depth
    /// differs from a 4-neighbour by more than this fraction sits ON a
    /// discontinuity, where the predicted depth is a blend of two surfaces
    /// and the gaussian lands between them. Reference `0.03`; `0` disables.
    pub edge_depth_rtol: f32,
    /// Reconcile the per-view depth maps against each other before they become
    /// geometry, rejecting pairs that differ by more than this fraction as
    /// occlusions rather than disagreements. 0 disables it.
    pub fuse_depth_rtol: f32,
    /// Keep a pixel only if at least this many OTHER frames agreed there is a
    /// surface there. 0 keeps everything.
    pub min_support: u16,
    /// Where a gaussian's position comes from. The model predicts the scene's
    /// geometry twice, through two heads that fail differently.
    pub position_from: PositionSource,
    /// Where the surface normal comes from when `surface_align` is on.
    pub normals_from: NormalSource,
    /// Lay each gaussian flat against the surface it sits on, at this
    /// thickness ratio. 0 keeps the orientation the model predicted.
    ///
    /// 4:1 by default, chosen by looking at 4 against 8 and against the
    /// model's own orientation on a real capture at a grazing angle, which is
    /// the view that shows the difference.
    ///
    /// The model does not orient its gaussians to anything: measured on a real
    /// capture, the angle between a gaussian's shortest axis and the surface
    /// normal implied by its own depth map is indistinguishable from random
    /// (median |cos| 0.416 against 0.5 for random, 5.0% within 20 degrees
    /// against 6% by chance), and they are near-isotropic anyway (b/c median
    /// 1.38). So each one is a little ball sitting at the right depth rather
    /// than a piece of surface.
    ///
    /// A disc lying IN the surface covers more of it per gaussian and is
    /// thinner THROUGH it, which is the shape a surface element should have -
    /// the observation behind 2D Gaussian Splatting and Gaussian Surfels. The
    /// normal comes from the depth map's own geometry rather than the normals
    /// head, so it is consistent with the points actually being placed.
    pub surface_align: f32,
    /// Drop this percentage of pixels, the least confident first, by the depth
    /// head's own confidence channel. 0 keeps everything.
    ///
    /// The model knows where it is guessing. Measured against multi-view
    /// agreement on a real capture, its least-confident quartile disagrees
    /// with the other views twice as much as its most-confident one (0.99% vs
    /// 0.53%, rank correlation -0.39). The merge weight carries no such signal
    /// (-0.03), so confidence is the only per-pixel quality estimate available.
    /// The threshold is global across frames, not per frame: a whole view can
    /// be harder than another and should lose more pixels, not the same share.
    pub conf_percentile: f32,
}

impl Default for AssembleOpts {
    fn default() -> Self {
        AssembleOpts { min_opacity: 0.01, max_depth: 0.0, gs_mask_threshold: 0.5, edge_depth_rtol: 0.03, fuse_depth_rtol: 0.05, min_support: 0, conf_percentile: 0.0, surface_align: 4.0, position_from: PositionSource::GsDepth, normals_from: NormalSource::Geometry }
    }
}

/// Flag every pixel that sits on a depth discontinuity.
///
/// `rtol` is relative, as upstream's `depth_edge(..., rtol=...)` is: what
/// matters at a silhouette is the RATIO of the two depths, not their
/// difference, so the same test works at any distance. Both sides of a step
/// are flagged, because the blend that makes the pixel unreliable straddles
/// it.
pub fn depth_edges(z: &[f32], w: usize, h: usize, rtol: f32) -> Vec<bool> {
    let mut out = vec![false; w * h];
    if rtol <= 0.0 {
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = z[i];
            if !(a.is_finite() && a > 0.0) {
                out[i] = true;
                continue;
            }
            let mut nb = [None; 4];
            if x > 0 { nb[0] = Some(z[i - 1]); }
            if x + 1 < w { nb[1] = Some(z[i + 1]); }
            if y > 0 { nb[2] = Some(z[i - w]); }
            if y + 1 < h { nb[3] = Some(z[i + w]); }
            for b in nb.into_iter().flatten() {
                if !(b.is_finite() && b > 0.0) || (a - b).abs() / a.min(b) > rtol {
                    out[i] = true;
                    break;
                }
            }
        }
    }
    out
}

/// The raw per-frame head outputs assembly reads, before any activation.
///
/// Separating these from the model is what makes assembly answerable offline.
/// Every option below - the mask threshold, the edge and fusion tolerances,
/// the support floor - decides which pixels become geometry, and picking one
/// used to cost a full forward pass, so they were picked by argument rather
/// than by measurement.
#[derive(Clone, Default)]
pub struct HeadOutputs {
    /// per frame, `3 * w * h`: log depth, log confidence, mask logit
    pub gsd: Vec<Vec<f32>>,
    /// per frame, `12 * w * h`: quat, log scale, opacity logit, colour, merge
    pub gsp: Vec<Vec<f32>>,
    /// per frame, `3 * w * h`: the [0,1] input frame, the colour source
    pub rgb: Vec<Vec<f32>>,
    /// per frame, `4 * w * h`: the POINTMAP head, pre-activation. A second,
    /// independent estimate of where every pixel is - already in world
    /// coordinates rather than a depth to unproject - with its own confidence
    /// in the fourth channel.
    pub pts: Vec<Vec<f32>>,
    /// per frame, `4 * w * h`: the NORMALS head, pre-activation, plus its
    /// confidence in the fourth channel.
    pub norm: Vec<Vec<f32>>,
    /// per frame, `3 * w * h`: the plain depth head, pre-activation. Predicted
    /// separately from the gaussian branch's own depth, so the two disagreeing
    /// is information rather than noise.
    pub depth: Vec<Vec<f32>>,
    pub width: u32,
    pub height: u32,
}

/// `sign(x) * (exp(|x|) - 1)`, the reference's `inv_log` attribute activation.
/// The pointmap head is trained through it, so its raw output means nothing
/// without it.
pub fn inv_log(x: f32) -> f32 {
    x.signum() * x.abs().exp_m1()
}

/// Where a gaussian's position comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PositionSource {
    /// Unproject the gaussian branch's own depth through the camera. What the
    /// reference does for the splat path, and the default.
    #[default]
    GsDepth,
    /// The pointmap head's world-space prediction, used directly. A different
    /// estimate through a different head, so it fails differently.
    Points,
    /// Confidence-weighted blend of the two, each expressed as a depth along
    /// the pixel's own ray so the blend cannot move a point sideways.
    Blend,
}

/// Where the surface normal for [`AssembleOpts::surface_align`] comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum NormalSource {
    /// Cross product of the tangents between neighbouring back-projected
    /// points: guaranteed consistent with where the gaussians actually are.
    #[default]
    Geometry,
    /// The normals head's own prediction.
    Head,
}

impl HeadOutputs {
    /// Read every frame's heads off the device - all of them, not only the
    /// gaussian branch. The pointmap, the normals and the plain depth head are
    /// separate predictions of the same scene, and a second opinion is only
    /// useful if it is available.
    pub fn read(gpu: &Gpu, model: &Mirror, s: usize, width: u32, height: u32) -> HeadOutputs {
        let hw = (width * height) as usize;
        HeadOutputs {
            gsd: (0..s).map(|fi| gpu.read(model.head_out(Head::GsDepth, fi), 3 * hw)).collect(),
            gsp: (0..s).map(|fi| gpu.read(model.head_out(Head::GsParams, fi), 12 * hw)).collect(),
            pts: (0..s).map(|fi| gpu.read(model.head_out(Head::Points, fi), 4 * hw)).collect(),
            norm: (0..s).map(|fi| gpu.read(model.head_out(Head::Normals, fi), 4 * hw)).collect(),
            depth: (0..s).map(|fi| gpu.read(model.head_out(Head::Depth, fi), 3 * hw)).collect(),
            rgb: Vec::new(),
            width,
            height,
        }
    }

    pub fn len(&self) -> usize {
        self.gsd.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gsd.is_empty()
    }

    /// Write every frame's heads to `dir` as raw little-endian f32, next to a
    /// one-line manifest naming the shape.
    ///
    /// Raw rather than an image format on purpose: depth is the model's
    /// primary output and quantising it to 8 bits is exactly the information a
    /// question about depth needs.
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join("heads.json"),
            format!(
                "{{\"frames\":{},\"width\":{},\"height\":{}}}\n",
                self.len(),
                self.width,
                self.height
            ),
        )?;
        for fi in 0..self.len() {
            write_f32(&dir.join(format!("gsd_{fi:03}.f32")), &self.gsd[fi])?;
            write_f32(&dir.join(format!("gsp_{fi:03}.f32")), &self.gsp[fi])?;
            for (tag, v) in [("pts", &self.pts), ("norm", &self.norm), ("dep", &self.depth)] {
                if let Some(p) = v.get(fi) {
                    write_f32(&dir.join(format!("{tag}_{fi:03}.f32")), p)?;
                }
            }
            if let Some(rgb) = self.rgb.get(fi) {
                write_f32(&dir.join(format!("rgb_{fi:03}.f32")), rgb)?;
            }
        }
        Ok(())
    }

    /// Read back what [`save`][Self::save] wrote.
    pub fn load(dir: impl AsRef<std::path::Path>) -> std::io::Result<HeadOutputs> {
        let dir = dir.as_ref();
        let man = std::fs::read_to_string(dir.join("heads.json"))?;
        let field = |k: &str| -> std::io::Result<usize> {
            man.split(&format!("\"{k}\":"))
                .nth(1)
                .and_then(|t| t.trim_start().split(|c: char| !c.is_ascii_digit()).next())
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("{}/heads.json has no {k}", dir.display()),
                    )
                })
        };
        let (s, width, height) = (field("frames")?, field("width")? as u32, field("height")? as u32);
        let hw = (width * height) as usize;
        let mut out = HeadOutputs { width, height, ..Default::default() };
        for fi in 0..s {
            out.gsd.push(read_f32(&dir.join(format!("gsd_{fi:03}.f32")), 3 * hw)?);
            out.gsp.push(read_f32(&dir.join(format!("gsp_{fi:03}.f32")), 12 * hw)?);
            out.rgb.push(read_f32(&dir.join(format!("rgb_{fi:03}.f32")), 3 * hw)?);
            // Optional: a dump written before these heads were carried still
            // loads, and assembly falls back to the gaussian branch alone.
            for (tag, v, n) in [
                ("pts", &mut out.pts, 4 * hw),
                ("norm", &mut out.norm, 4 * hw),
                ("dep", &mut out.depth, 3 * hw),
            ] {
                let p = dir.join(format!("{tag}_{fi:03}.f32"));
                if p.exists() {
                    v.push(read_f32(&p, n)?);
                }
            }
        }
        Ok(out)
    }
}

fn write_f32(path: &std::path::Path, v: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn read_f32(path: &std::path::Path, n: usize) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != n * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} holds {} bytes, expected {}", path.display(), bytes.len(), n * 4),
        ));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Read the GS head outputs for all frames and build the host scene.
/// `frames_chw` = the raw [0,1] input frames (color source). Also returns the
/// per-gaussian sigmoid merge weights for `splat::prune` - the same values
/// the scene's opacities carry, since the merge weight IS the opacity of the
/// assembled scene.
pub fn assemble(
    gpu: &Gpu,
    model: &Mirror,
    frames_chw: &[f32],
    s: usize,
    width: u32,
    height: u32,
    opts: &AssembleOpts,
    known: Option<&[Camera]>,
) -> (Splats, Vec<Camera>, Vec<f32>) {
    let mut heads = HeadOutputs::read(gpu, model, s, width, height);
    let hw = (width * height) as usize;
    heads.rgb = (0..s).map(|fi| frames_chw[fi * 3 * hw..(fi + 1) * 3 * hw].to_vec()).collect();
    let cams = match known {
        Some(k) if k.len() == s => k.to_vec(),
        _ => decode_cameras(&model.cam_pred_raw(), s, width, height),
    };
    assemble_from(&heads, &cams, opts)
}

/// Build the scene from head outputs that are already in host memory.
///
/// The same function `assemble` runs, with the device and the model taken out
/// of it, so a dumped forward pass can be re-assembled under different options
/// in seconds instead of a quarter of an hour.
pub fn assemble_from(
    heads: &HeadOutputs,
    cams: &[Camera],
    opts: &AssembleOpts,
) -> (Splats, Vec<Camera>, Vec<f32>) {
    let (s, width, height) = (heads.len(), heads.width, heads.height);
    // Back-project through the cameras the caller KNOWS when it has them, and
    // through the predicted ones otherwise. The reference does exactly this
    // and says why: gaussian positions are depth unprojected through a camera,
    // so a camera error becomes a position error for every pixel of that
    // frame. Conditioning the trunk on a pose only informs the features; it
    // does not stop the camera head from being the thing the geometry is
    // built on.
    let hw = (width * height) as usize;

    // Settle the frames' disagreement about depth before any of it becomes
    // geometry. They cannot settle it afterwards: the error is along each
    // pixel's own viewing ray, invisible from the frame that made it and
    // untouchable by a fit whose gradients are orthogonal to that ray.
    let mut depths: Vec<Vec<f32>> = heads
        .gsd
        .iter()
        .map(|g| g[..hw].iter().map(|v| v.exp()).collect())
        .collect();
    // The pointmap head is a second, independent estimate of the same
    // geometry. Taken alone it replaces the gaussian branch's depth; blended,
    // each pixel is a confidence-weighted average of the two, which is the
    // same reconciliation `fuse_depths` performs across views applied across
    // HEADS - and for the same reason, since neither head can see its own
    // error along the ray.
    if opts.position_from != PositionSource::GsDepth && heads.pts.len() == s {
        for (fi, cam) in cams.iter().enumerate() {
            let (pd, pc) = points_as_depth(&heads.pts[fi], cam, width as usize, height as usize);
            let gc: Vec<f32> = heads.gsd[fi][hw..2 * hw].iter().map(|v| 1.0 + v.exp()).collect();
            for i in 0..hw {
                if !(pd[i] > 0.0) {
                    continue;
                }
                depths[fi][i] = match opts.position_from {
                    PositionSource::Points => pd[i],
                    _ => {
                        let (a, b) = (gc[i], pc[i]);
                        (a * depths[fi][i] + b * pd[i]) / (a + b).max(1e-12)
                    }
                };
            }
        }
    }
    let support = if opts.fuse_depth_rtol > 0.0 && s > 1 {
        // the head's second channel is its own confidence in the first
        let confs: Vec<Vec<f32>> = heads
            .gsd
            .iter()
            .map(|g| g[hw..2 * hw].iter().map(|v| 1.0 + v.exp()).collect())
            .collect();
        fuse_depths(&mut depths, &confs, cams, width, height, opts.fuse_depth_rtol)
    } else {
        Vec::new()
    };

    // One global confidence floor, found before anything is built.
    let conf_floor = if opts.conf_percentile > 0.0 && opts.conf_percentile < 100.0 {
        let mut all: Vec<f32> =
            heads.gsd.iter().flat_map(|g| g[hw..2 * hw].iter().copied()).collect();
        let k = ((all.len() as f64) * (opts.conf_percentile as f64) / 100.0) as usize;
        let k = k.min(all.len().saturating_sub(1));
        all.select_nth_unstable_by(k, f32::total_cmp);
        all[k]
    } else {
        f32::NEG_INFINITY
    };

    let mut out = Splats::default();
    let mut weights = Vec::new();
    for (fi, cam) in cams.iter().enumerate() {
        // The GS depth head emits THREE channels - depth, confidence, and a
        // validity mask - laid out like `Head::Depth`'s. Only the first was
        // ever read, so pixels the model itself reports as not-geometry became
        // gaussians anyway.
        let gsd = &heads.gsd[fi];
        let gsp = &heads.gsp[fi];
        let rgb = &heads.rgb[fi];
        let m = &cam.c2w;
        let depth = std::mem::take(&mut depths[fi]);
        let edge = depth_edges(&depth, width as usize, height as usize, opts.edge_depth_rtol);
        let normals = (opts.surface_align > 1.0).then(|| {
            match opts.normals_from {
                NormalSource::Head if heads.norm.len() == s => {
                    head_normals(&heads.norm[fi], cam, width as usize, height as usize)
                }
                _ => depth_normals(&depth, cam, width as usize, height as usize),
            }
        });
        for py in 0..height as usize {
            for px in 0..width as usize {
                let i = py * width as usize + px;
                // Channel 11, the learned merge weight, IS the opacity of the
                // assembled scene - not channel 7, the raw opacity head. The
                // reference's assembly substitutes it (see
                // `splat::prune::voxel_merge`, which computes the same thing
                // as a weighted mean when it merges), and the difference is
                // not cosmetic: measured on a six-photograph reconstruction
                // rendered from the camera the model recovered for it, channel
                // 7 gives 12.5 dB carrying 7% of the photograph's detail, and
                // channel 11 gives 22.5 dB carrying 47%. Channel 7 renders as
                // a milky smear.
                let op = sigmoid(gsp[11 * hw + i]);
                if op < opts.min_opacity {
                    continue;
                }
                if opts.gs_mask_threshold > 0.0 && sigmoid(gsd[2 * hw + i]) < opts.gs_mask_threshold {
                    continue;
                }
                if edge[i] {
                    continue;
                }
                if opts.min_support > 0
                    && support.get(fi).is_some_and(|s| s[i] < opts.min_support)
                {
                    continue; // no other view agrees a surface is here
                }
                if gsd[hw + i] < conf_floor {
                    continue; // the model says it is guessing here
                }
                let z = depth[i];
                if opts.max_depth > 0.0 && z > opts.max_depth {
                    continue;
                }
                // A pixel is an AREA and the rasterizer samples it at its
                // centre, so unprojecting pixel n uses n + 0.5.
                //
                // Worth knowing that the reference does NOT: it unprojects
                // integer pixel indices against `cx = w/2`, which is half a
                // pixel off from its own grid centre. Measured on a real
                // capture, the choice is a wash - median cross-view depth
                // disagreement 0.6823% with the half pixel and 0.6854%
                // without - so this keeps the convention that is right for
                // the rasterizer that consumes it rather than chasing a
                // difference below the noise.
                let xc = (px as f32 + 0.5 - cam.cx) * z / cam.fx;
                let yc = (py as f32 + 0.5 - cam.cy) * z / cam.fy;
                out.means.extend_from_slice(&[
                    m[0] * xc + m[1] * yc + m[2] * z + m[3],
                    m[4] * xc + m[5] * yc + m[6] * z + m[7],
                    m[8] * xc + m[9] * yc + m[10] * z + m[11],
                ]);
                // Normalized here, not left to the renderer. The module doc
                // has always said "quat wxyz normalized" and the code did not
                // do it; upstream normalizes before encoding, and a
                // non-unit quaternion scales the covariance it rotates.
                let sc = [
                    gsp[4 * hw + i].exp().min(0.3),
                    gsp[5 * hw + i].exp().min(0.3),
                    gsp[6 * hw + i].exp().min(0.3),
                ];
                match &normals {
                    Some(nm) => {
                        // Keep the size the model chose - it is calibrated to
                        // the sampling rate - and only change the SHAPE: the
                        // two in-plane axes take the largest of the three, the
                        // one along the normal is thinned by the ratio.
                        let r = sc[0].max(sc[1]).max(sc[2]);
                        out.quats.extend_from_slice(&quat_with_z(nm[i]));
                        out.scales.extend_from_slice(&[r, r, r / opts.surface_align]);
                    }
                    None => {
                        let q = [gsp[i], gsp[hw + i], gsp[2 * hw + i], gsp[3 * hw + i]];
                        let qn = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
                        out.quats.extend_from_slice(&[q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn]);
                        out.scales.extend_from_slice(&sc);
                    }
                }
                out.opacities.push(op);
                out.colors.extend_from_slice(&[
                    gsp[8 * hw + i] * SH_C0 + rgb[i],
                    gsp[9 * hw + i] * SH_C0 + rgb[hw + i],
                    gsp[10 * hw + i] * SH_C0 + rgb[2 * hw + i],
                ]);
                weights.push(op);
            }
        }
    }
    (out, cams.to_vec(), weights)
}

/// Depth/normal/confidence maps for one frame, activations applied.
pub struct FrameMaps {
    pub depth: Vec<f32>,   // exp
    pub conf: Vec<f32>,    // 1+exp
    pub mask: Vec<f32>,    // sigmoid
    pub normals: Vec<f32>, // unit, [3,H,W]
}

pub fn frame_maps(gpu: &Gpu, model: &Mirror, fi: usize, width: u32, height: u32) -> FrameMaps {
    let hw = (width * height) as usize;
    let d = gpu.read(model.head_out(Head::Depth, fi), 3 * hw);
    let n = gpu.read(model.head_out(Head::Normals, fi), 4 * hw);
    let mut normals = vec![0.0f32; 3 * hw];
    for i in 0..hw {
        let v = [n[i], n[hw + i], n[2 * hw + i]];
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
        normals[i] = v[0] / len;
        normals[hw + i] = v[1] / len;
        normals[2 * hw + i] = v[2] / len;
    }
    FrameMaps {
        depth: d[..hw].iter().map(|&v| v.exp()).collect(),
        conf: d[hw..2 * hw].iter().map(|&v| 1.0 + v.exp()).collect(),
        mask: d[2 * hw..3 * hw].iter().map(|&v| sigmoid(v)).collect(),
        normals,
    }
}
