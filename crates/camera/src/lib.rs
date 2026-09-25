// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Camera models: how a direction in a camera's own frame becomes a pixel,
//! and back.
//!
//! One implementation for every stage that reasons about a lens - structure
//! from motion calibrates it, multi-view stereo walks rays through it, and the
//! splat renderer projects gaussians through it and composites along its
//! pixels' rays.
//!
//! Conventions, shared with `splat::types::Camera`:
//!
//! * camera frame +X right, +Y down, +Z forward;
//! * CONTINUOUS pixel coordinates with the origin at the image's top-left
//!   CORNER: pixel `(i, j)` covers `[i, i+1) x [j, j+1)` and its centre is
//!   `(i + 0.5, j + 0.5)`. Resampling an image by `s` therefore scales every
//!   pixel coordinate - and `fx, fy, cx, cy` - by exactly `s`;
//! * [`Intrinsics::unproject`] returns UNIT ray directions, so a distance
//!   along a ray is a range, the same quantity for a perspective lens and a
//!   fisheye looking past 90 degrees.
//!
//! The models, implemented from their defining equations:
//!
//! * [`Lens::Pinhole`];
//! * [`Lens::Brown`] - Brown-Conrady as OpenCV parameterizes it: rational
//!   radial `(1 + k1 r² + k2 r⁴ + k3 r⁶) / (1 + k4 r² + k5 r⁴ + k6 r⁶)`,
//!   decentering (tangential) `p1, p2` and thin prism `s1..s4` (Brown,
//!   "Decentering Distortion of Lenses", 1966; Conrady 1919);
//! * [`Lens::Fisheye`] - Kannala-Brandt (TPAMI 2006), the equidistant model
//!   with the odd polynomial `θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)` that
//!   OpenCV's fisheye module uses;
//! * [`Lens::Equirect`] - longitude/latitude, for 360-degree captures.
//!
//! A polynomial distortion is only a lens inside the radius where it is
//! monotonic: past it, points further from the axis land CLOSER to the image
//! centre, and a projection that trusted the polynomial there would fold
//! geometry from behind the field of view back into the frame. So every model
//! carries its own validity bound ([`Intrinsics::valid_radius`]) and
//! projection refuses anything past it.
//!
//! Swedish Embedded AB implements camera calibration and lens modelling for
//! photogrammetry, robotics and embedded vision. If your team needs expertise
//! in camera models and calibration, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::f64::consts::PI;

/// How a lens bends rays, beyond the linear map every model shares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lens {
    Pinhole,
    /// OpenCV's full perspective model: rational radial `k1..k6`,
    /// tangential `p1, p2`, thin prism `s1..s4`.
    Brown { k: [f64; 6], p: [f64; 2], s: [f64; 4] },
    /// Kannala-Brandt equidistant fisheye, `k1..k4`.
    Fisheye { k: [f64; 4] },
    /// Equirectangular: `u = cx + fx·lon`, `v = cy + fy·lat`, +lat down.
    Equirect,
}

impl Lens {
    /// A Brown lens with only the two radial terms most calibrations need.
    pub fn radial(k1: f64, k2: f64) -> Lens {
        Lens::Brown { k: [k1, k2, 0.0, 0.0, 0.0, 0.0], p: [0.0; 2], s: [0.0; 4] }
    }

    /// Distortion coefficients, in [`Intrinsics::params`] order after the
    /// four linear ones.
    pub fn coeffs(&self) -> Vec<f64> {
        match self {
            Lens::Pinhole | Lens::Equirect => Vec::new(),
            Lens::Brown { k, p, s } => k.iter().chain(p).chain(s).copied().collect(),
            Lens::Fisheye { k } => k.to_vec(),
        }
    }

    fn with_coeffs(&self, c: &[f64]) -> Lens {
        match self {
            Lens::Pinhole => Lens::Pinhole,
            Lens::Equirect => Lens::Equirect,
            Lens::Brown { .. } => Lens::Brown {
                k: std::array::from_fn(|i| c[i]),
                p: [c[6], c[7]],
                s: std::array::from_fn(|i| c[8 + i]),
            },
            Lens::Fisheye { .. } => Lens::Fisheye { k: std::array::from_fn(|i| c[i]) },
        }
    }

    /// The model's name in `cameras.json`.
    pub fn name(&self) -> &'static str {
        match self {
            Lens::Pinhole => "pinhole",
            Lens::Brown { .. } => "opencv",
            Lens::Fisheye { .. } => "fisheye",
            Lens::Equirect => "equirect",
        }
    }

    /// The code the device kernels switch on.
    pub fn code(&self) -> u32 {
        match self {
            Lens::Pinhole => 0,
            Lens::Brown { .. } => 1,
            Lens::Fisheye { .. } => 2,
            Lens::Equirect => 3,
        }
    }

    /// Whether the model sees only the half-space in front of the camera.
    pub fn is_perspective(&self) -> bool {
        matches!(self, Lens::Pinhole | Lens::Brown { .. })
    }
}

/// A camera's intrinsic calibration: the linear map `fx, fy, cx, cy` in
/// pixels, the lens, and the image size it describes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub lens: Lens,
    pub width: u32,
    pub height: u32,
}

/// A projected pixel, its Jacobian in the direction (2x3) and, when asked
/// for, its Jacobian in the calibration parameters (one column per
/// parameter).
pub type Projection = ([f64; 2], [[f64; 3]; 2], Vec<[f64; 2]>);

/// Radius (normalized, `r = |(x, y)|` of `X/Z, Y/Z`) the Brown validity scan
/// looks out to: 4 is 76 degrees off axis, past any lens this model can
/// describe - a wider one is a fisheye.
const BROWN_SCAN_R: f64 = 4.0;

impl Intrinsics {
    /// A pinhole camera of focal length `f` pixels, principal point at the
    /// image centre.
    pub fn pinhole(f: f64, width: u32, height: u32) -> Intrinsics {
        Intrinsics { fx: f, fy: f, cx: width as f64 / 2.0, cy: height as f64 / 2.0, lens: Lens::Pinhole, width, height }
    }

    /// The equirectangular camera covering the full sphere at `width x
    /// height` (2:1 by convention).
    pub fn equirect(width: u32, height: u32) -> Intrinsics {
        Intrinsics {
            fx: width as f64 / (2.0 * PI),
            fy: height as f64 / PI,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
            lens: Lens::Equirect,
            width,
            height,
        }
    }

    /// The same camera after the image is resampled to `width x height`.
    /// Continuous pixel coordinates scale exactly, so this is exact.
    pub fn resized(&self, width: u32, height: u32) -> Intrinsics {
        let (sx, sy) = (width as f64 / self.width as f64, height as f64 / self.height as f64);
        Intrinsics { fx: self.fx * sx, fy: self.fy * sy, cx: self.cx * sx, cy: self.cy * sy, width, height, ..*self }
    }

    /// Every calibration parameter: `fx, fy, cx, cy`, then the lens's
    /// coefficients ([`Lens::coeffs`]).
    pub fn params(&self) -> Vec<f64> {
        let mut v = vec![self.fx, self.fy, self.cx, self.cy];
        v.extend(self.lens.coeffs());
        v
    }

    pub fn param_count(&self) -> usize {
        4 + self.lens.coeffs().len()
    }

    /// The same model with its parameters replaced, in [`Self::params`]
    /// order.
    pub fn with_params(&self, p: &[f64]) -> Intrinsics {
        assert_eq!(p.len(), self.param_count(), "{} parameters for a {} camera", p.len(), self.lens.name());
        Intrinsics { fx: p[0], fy: p[1], cx: p[2], cy: p[3], lens: self.lens.with_coeffs(&p[4..]), ..*self }
    }

    /// Names of [`Self::params`], for reports.
    pub fn param_names(&self) -> Vec<&'static str> {
        let mut v = vec!["fx", "fy", "cx", "cy"];
        match self.lens {
            Lens::Pinhole | Lens::Equirect => {}
            Lens::Brown { .. } => v.extend(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"]),
            Lens::Fisheye { .. } => v.extend(["k1", "k2", "k3", "k4"]),
        }
        v
    }

    /// The largest off-axis extent the lens model is a lens over: the
    /// normalized radius `r` for [`Lens::Brown`], the angle `θ` in radians
    /// for [`Lens::Fisheye`], infinity where nothing folds.
    ///
    /// Found by walking out from the axis until the radial map stops
    /// increasing - the first place a larger input lands no further out.
    pub fn valid_radius(&self) -> f64 {
        match self.lens {
            Lens::Pinhole | Lens::Equirect => f64::INFINITY,
            Lens::Brown { k, .. } => {
                // d(r·rad(r²))/dr, which must stay positive
                let slope = |r: f64| {
                    let r2 = r * r;
                    let num = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[2]));
                    let den = 1.0 + r2 * (k[3] + r2 * (k[4] + r2 * k[5]));
                    let dnum = k[0] + r2 * (2.0 * k[1] + 3.0 * k[2] * r2);
                    let dden = k[3] + r2 * (2.0 * k[4] + 3.0 * k[5] * r2);
                    if den <= 1e-9 {
                        return -1.0;
                    }
                    num / den + 2.0 * r2 * (dnum * den - num * dden) / (den * den)
                };
                memo_fold(&k, BROWN_SCAN_R, slope).unwrap_or(f64::INFINITY)
            }
            Lens::Fisheye { k } => {
                let slope = |t: f64| {
                    let t2 = t * t;
                    1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])))
                };
                memo_fold(&k, PI, slope).unwrap_or(PI)
            }
        }
    }

    /// The pixel a camera-frame point or direction `d` lands on, `None` when
    /// the lens does not image it (behind a perspective camera, or past the
    /// lens's [`valid_radius`](Self::valid_radius)).
    pub fn project(&self, d: [f64; 3]) -> Option<[f64; 2]> {
        self.project_jac(d).map(|(p, _)| p)
    }

    /// [`Self::project`] and its Jacobian `d pixel / d d` (2x3).
    pub fn project_jac(&self, d: [f64; 3]) -> Option<([f64; 2], [[f64; 3]; 2])> {
        self.project_full(d, false).map(|(p, j, _)| (p, j))
    }

    /// [`Self::project`], its Jacobian in the direction, and its Jacobian in
    /// the calibration [`params`](Self::params) (2 x param_count).
    pub fn project_param_jac(&self, d: [f64; 3]) -> Option<Projection> {
        self.project_full(d, true)
    }

    fn project_full(&self, d: [f64; 3], want_params: bool) -> Option<Projection> {
        let [x3, y3, z3] = d;
        match self.lens {
            Lens::Pinhole | Lens::Brown { .. } => {
                if z3 <= 1e-12 {
                    return None;
                }
                let iz = 1.0 / z3;
                let (x, y) = (x3 * iz, y3 * iz);
                // d(x, y) / d(X, Y, Z)
                let dn = [[iz, 0.0, -x * iz], [0.0, iz, -y * iz]];
                let (xd, yd, j2, jk) = match self.lens {
                    Lens::Brown { k, p, s } => {
                        let r2 = x * x + y * y;
                        let rmax = self.valid_radius();
                        if r2 > rmax * rmax {
                            return None;
                        }
                        brown(x, y, &k, &p, &s, want_params)
                    }
                    _ => (x, y, [[1.0, 0.0], [0.0, 1.0]], Vec::new()),
                };
                let u = [self.fx * xd + self.cx, self.fy * yd + self.cy];
                let mut jd = [[0.0; 3]; 2];
                for c in 0..3 {
                    jd[0][c] = self.fx * (j2[0][0] * dn[0][c] + j2[0][1] * dn[1][c]);
                    jd[1][c] = self.fy * (j2[1][0] * dn[0][c] + j2[1][1] * dn[1][c]);
                }
                let jp = if want_params {
                    let mut v = vec![[xd, 0.0], [0.0, yd], [1.0, 0.0], [0.0, 1.0]];
                    v.extend(jk.iter().map(|g| [self.fx * g[0], self.fy * g[1]]));
                    v
                } else {
                    Vec::new()
                };
                Some((u, jd, jp))
            }
            Lens::Fisheye { k } => {
                let a2 = x3 * x3 + y3 * y3;
                let a = a2.sqrt();
                let rho2 = a2 + z3 * z3;
                if rho2 <= 1e-24 {
                    return None;
                }
                let theta = a.atan2(z3);
                if theta > self.valid_radius() {
                    return None;
                }
                let t2 = theta * theta;
                let poly = 1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])));
                let td = theta * poly;
                let dtd = 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
                // xd = s X, yd = s Y with s = θd / a; near the axis the
                // quotient is taken from its series, where it is exact to
                // rounding and the closed form is 0/0.
                let (s, c1, dsdz) = if a > 1e-9 * z3.abs().max(1e-300) {
                    (td / a, (dtd * z3 / rho2 - td / a) / a2, -dtd / rho2)
                } else if z3 < 0.0 {
                    return None; // straight behind: no azimuth to project along
                } else {
                    let iz = 1.0 / z3;
                    (iz * (1.0 + (k[0] - 1.0 / 3.0) * a2 * iz * iz), 2.0 * (k[0] - 1.0 / 3.0) * iz * iz * iz, -iz * iz)
                };
                let (xd, yd) = (s * x3, s * y3);
                let u = [self.fx * xd + self.cx, self.fy * yd + self.cy];
                let jx = [s + x3 * x3 * c1, x3 * y3 * c1, x3 * dsdz];
                let jy = [x3 * y3 * c1, s + y3 * y3 * c1, y3 * dsdz];
                let jd = [jx.map(|v| self.fx * v), jy.map(|v| self.fy * v)];
                let jp = if want_params {
                    // d θd / d k_i = θ^(2i+3); xd = θd X / a
                    let (ux, uy) = if a > 1e-12 { (x3 / a, y3 / a) } else { (0.0, 0.0) };
                    let mut v = vec![[xd, 0.0], [0.0, yd], [1.0, 0.0], [0.0, 1.0]];
                    let mut tp = theta * t2;
                    for _ in 0..4 {
                        v.push([self.fx * tp * ux, self.fy * tp * uy]);
                        tp *= t2;
                    }
                    v
                } else {
                    Vec::new()
                };
                Some((u, jd, jp))
            }
            Lens::Equirect => {
                let h2 = x3 * x3 + z3 * z3;
                let rho2 = h2 + y3 * y3;
                if h2 <= 1e-24 {
                    return None; // the poles: longitude is undefined
                }
                let h = h2.sqrt();
                let lon = x3.atan2(z3);
                let lat = y3.atan2(h);
                let u = [self.cx + self.fx * lon, self.cy + self.fy * lat];
                let jd = [
                    [self.fx * z3 / h2, 0.0, -self.fx * x3 / h2],
                    [-self.fy * y3 * x3 / (h * rho2), self.fy * h / rho2, -self.fy * y3 * z3 / (h * rho2)],
                ];
                let jp = if want_params { vec![[lon, 0.0], [0.0, lat], [1.0, 0.0], [0.0, 1.0]] } else { Vec::new() };
                Some((u, jd, jp))
            }
        }
    }

    /// The unit ray direction through continuous pixel coordinate `px`,
    /// `None` when no direction the lens images lands there (outside a
    /// fisheye's image circle, or where the distortion cannot be inverted).
    pub fn unproject(&self, px: [f64; 2]) -> Option<[f64; 3]> {
        let xd = (px[0] - self.cx) / self.fx;
        let yd = (px[1] - self.cy) / self.fy;
        match self.lens {
            Lens::Pinhole => Some(unit([xd, yd, 1.0])),
            Lens::Brown { k, p, s } => {
                let rmax = self.valid_radius();
                // Newton on distort(x) = xd from x = xd, damped so a step
                // never leaves the valid disc.
                let (mut x, mut y) = (xd, yd);
                for _ in 0..40 {
                    let (fx, fy, j, _) = brown(x, y, &k, &p, &s, false);
                    let (ex, ey) = (fx - xd, fy - yd);
                    if ex.abs() < 1e-14 && ey.abs() < 1e-14 {
                        break;
                    }
                    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                    if det.abs() < 1e-18 {
                        return None;
                    }
                    let dx = (j[1][1] * ex - j[0][1] * ey) / det;
                    let dy = (-j[1][0] * ex + j[0][0] * ey) / det;
                    let mut step = 1.0;
                    while step > 1e-6 && (x - step * dx).hypot(y - step * dy) > rmax {
                        step *= 0.5;
                    }
                    x -= step * dx;
                    y -= step * dy;
                }
                let (fx, fy, _, _) = brown(x, y, &k, &p, &s, false);
                if (fx - xd).abs() > 1e-9 || (fy - yd).abs() > 1e-9 || x.hypot(y) > rmax {
                    return None;
                }
                Some(unit([x, y, 1.0]))
            }
            Lens::Fisheye { k } => {
                let rd = xd.hypot(yd);
                let tmax = self.valid_radius();
                // Newton on θ (1 + k1 θ² + ...) = rd, which is monotonic
                // below tmax.
                let mut t = rd.min(tmax);
                for _ in 0..40 {
                    let t2 = t * t;
                    let f = t * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - rd;
                    let df = 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
                    if df <= 0.0 {
                        return None;
                    }
                    let nt = (t - f / df).clamp(0.0, tmax);
                    if (nt - t).abs() < 1e-15 {
                        t = nt;
                        break;
                    }
                    t = nt;
                }
                let t2 = t * t;
                let back = t * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
                if (back - rd).abs() > 1e-9 * rd.max(1.0) {
                    return None;
                }
                let (st, ct) = t.sin_cos();
                if rd < 1e-15 {
                    return Some([0.0, 0.0, 1.0]);
                }
                Some([st * xd / rd, st * yd / rd, ct])
            }
            Lens::Equirect => {
                let lon = xd;
                let lat = yd;
                if lat.abs() > PI / 2.0 || lon.abs() > PI {
                    return None;
                }
                Some([lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()])
            }
        }
    }

    /// How the ray through `px` moves with the calibration: `d dir / d
    /// params` (3 x param_count), holding the pixel fixed. Differentiates
    /// `project(dir(θ); θ) = px` implicitly, with the unit-norm constraint
    /// `dirᵀ d dir = 0` closing the system.
    pub fn unproject_param_jac(&self, px: [f64; 2]) -> Option<([f64; 3], Vec<[f64; 3]>)> {
        let d = self.unproject(px)?;
        let (_, jd, jp) = self.project_param_jac(d)?;
        // [jd; dᵀ] (3x3) * ddir = [-jp; 0]
        let m = [jd[0], jd[1], d];
        let inv = inv3(&m)?;
        let cols = jp
            .iter()
            .map(|g| {
                let rhs = [-g[0], -g[1], 0.0];
                std::array::from_fn(|r| inv[r][0] * rhs[0] + inv[r][1] * rhs[1] + inv[r][2] * rhs[2])
            })
            .collect();
        Some((d, cols))
    }

    /// Pixels per radian at the image point `d` projects to - the local
    /// sampling rate, which is what a footprint in pixels has to be divided
    /// by to become an angle. The geometric mean of the two singular values
    /// of the projection's Jacobian restricted to the plane across the ray.
    pub fn pixels_per_radian(&self, d: [f64; 3]) -> Option<f64> {
        let (_, j) = self.project_jac(d)?;
        let n = unit(d);
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        // an orthonormal pair across the ray
        let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let e1 = unit(cross(n, a));
        let e2 = cross(n, e1);
        let col = |e: [f64; 3]| [(j[0][0] * e[0] + j[0][1] * e[1] + j[0][2] * e[2]) * norm, (j[1][0] * e[0] + j[1][1] * e[1] + j[1][2] * e[2]) * norm];
        let (c1, c2) = (col(e1), col(e2));
        let det = (c1[0] * c2[1] - c1[1] * c2[0]).abs();
        Some(det.sqrt())
    }

    /// `cameras.json` fields of this calibration: `fx fy cx cy width height`
    /// and, for anything but a pinhole, `model` and its coefficients.
    pub fn to_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
        });
        if self.lens != Lens::Pinhole {
            v["model"] = self.lens.name().into();
            match self.lens {
                Lens::Brown { k, p, s } => {
                    v["k"] = k.to_vec().into();
                    v["p"] = p.to_vec().into();
                    v["s"] = s.to_vec().into();
                }
                Lens::Fisheye { k } => v["k"] = k.to_vec().into(),
                _ => {}
            }
        }
        v
    }

    /// The calibration of a `cameras.json` entry; a missing `model` is a
    /// pinhole, which is what every older file describes.
    pub fn from_json(c: &serde_json::Value) -> Result<Intrinsics, String> {
        let f = |k: &str| c[k].as_f64().ok_or_else(|| format!("camera has no '{k}'"));
        let u = |k: &str| c[k].as_u64().ok_or_else(|| format!("camera has no '{k}'"));
        let list = |k: &str, n: usize| -> Result<Vec<f64>, String> {
            match c.get(k) {
                None => Ok(vec![0.0; n]),
                Some(a) => {
                    let v: Vec<f64> = a.as_array().ok_or(format!("'{k}' must be an array"))?.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                    if v.len() > n {
                        return Err(format!("'{k}' has {} entries, at most {n}", v.len()));
                    }
                    let mut out = vec![0.0; n];
                    out[..v.len()].copy_from_slice(&v);
                    Ok(out)
                }
            }
        };
        let lens = match c.get("model").and_then(|m| m.as_str()).unwrap_or("pinhole") {
            "pinhole" => Lens::Pinhole,
            "opencv" | "brown" => {
                let (k, p, s) = (list("k", 6)?, list("p", 2)?, list("s", 4)?);
                Lens::Brown { k: std::array::from_fn(|i| k[i]), p: [p[0], p[1]], s: std::array::from_fn(|i| s[i]) }
            }
            "fisheye" | "kannala-brandt" => {
                let k = list("k", 4)?;
                Lens::Fisheye { k: std::array::from_fn(|i| k[i]) }
            }
            "equirect" | "equirectangular" => Lens::Equirect,
            other => return Err(format!("unknown camera model '{other}'")),
        };
        Ok(Intrinsics { fx: f("fx")?, fy: f("fy")?, cx: f("cx")?, cy: f("cy")?, lens, width: u("width")? as u32, height: u("height")? as u32 })
    }
}

/// A distorted normalized point, its 2x2 Jacobian, and its coefficient
/// Jacobian.
type Distorted = (f64, f64, [[f64; 2]; 2], Vec<[f64; 2]>);

/// Brown distortion of normalized `(x, y)`: the distorted point, its 2x2
/// Jacobian in `(x, y)`, and (when asked) its Jacobian in the twelve
/// coefficients `k1..k6, p1, p2, s1..s4`, UNSCALED by the focal lengths.
fn brown(x: f64, y: f64, k: &[f64; 6], p: &[f64; 2], s: &[f64; 4], want: bool) -> Distorted {
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let num = 1.0 + k[0] * r2 + k[1] * r4 + k[2] * r6;
    let den = 1.0 + k[3] * r2 + k[4] * r4 + k[5] * r6;
    let rad = num / den;
    let dnum = k[0] + 2.0 * k[1] * r2 + 3.0 * k[2] * r4;
    let dden = k[3] + 2.0 * k[4] * r2 + 3.0 * k[5] * r4;
    let drad = (dnum * den - num * dden) / (den * den); // d rad / d r2
    let xd = x * rad + 2.0 * p[0] * x * y + p[1] * (r2 + 2.0 * x * x) + s[0] * r2 + s[1] * r4;
    let yd = y * rad + p[0] * (r2 + 2.0 * y * y) + 2.0 * p[1] * x * y + s[2] * r2 + s[3] * r4;
    let j = [
        [
            rad + 2.0 * x * x * drad + 2.0 * p[0] * y + 6.0 * p[1] * x + 2.0 * s[0] * x + 4.0 * s[1] * r2 * x,
            2.0 * x * y * drad + 2.0 * p[0] * x + 2.0 * p[1] * y + 2.0 * s[0] * y + 4.0 * s[1] * r2 * y,
        ],
        [
            2.0 * x * y * drad + 2.0 * p[0] * x + 2.0 * p[1] * y + 2.0 * s[2] * x + 4.0 * s[3] * r2 * x,
            rad + 2.0 * y * y * drad + 6.0 * p[0] * y + 2.0 * p[1] * x + 2.0 * s[2] * y + 4.0 * s[3] * r2 * y,
        ],
    ];
    let jk = if want {
        let dd = num / (den * den);
        vec![
            [x * r2 / den, y * r2 / den],
            [x * r4 / den, y * r4 / den],
            [x * r6 / den, y * r6 / den],
            [-x * r2 * dd, -y * r2 * dd],
            [-x * r4 * dd, -y * r4 * dd],
            [-x * r6 * dd, -y * r6 * dd],
            [2.0 * x * y, r2 + 2.0 * y * y],
            [r2 + 2.0 * x * x, 2.0 * x * y],
            [r2, 0.0],
            [r4, 0.0],
            [0.0, r2],
            [0.0, r4],
        ]
    } else {
        Vec::new()
    };
    (xd, yd, j, jk)
}

/// The first place in `(0, limit]` where `slope` stops being positive, `None`
/// if it never does - the edge of a radial polynomial's valid disc.
///
/// A coarse walk brackets the first sign change and bisection pins it down.
/// Projection asks this for every point it projects, so the answer is kept
/// per thread for the last coefficient set seen: a capture has one lens, and
/// bundle adjustment changes it once per iteration, not once per point.
fn memo_fold<const N: usize>(coeffs: &[f64; N], limit: f64, slope: impl Fn(f64) -> f64) -> Option<f64> {
    /// `(coefficient bits, limit, fold)` of the last few lenses asked about.
    type Memo = Vec<(Vec<u64>, f64, Option<f64>)>;
    thread_local! {
        static LAST: std::cell::RefCell<Memo> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let key: Vec<u64> = coeffs.iter().map(|v| v.to_bits()).collect();
    if let Some(hit) = LAST.with(|c| c.borrow().iter().find(|(k, l, _)| *k == key && *l == limit).map(|e| e.2)) {
        return hit;
    }
    let steps = 512;
    let mut fold = None;
    let mut lo = 0.0;
    for i in 1..=steps {
        let r = limit * i as f64 / steps as f64;
        if slope(r) <= 0.0 {
            let mut hi = r;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if slope(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            fold = Some(lo);
            break;
        }
        lo = r;
    }
    LAST.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= 8 {
            c.remove(0);
        }
        c.push((key, limit, fold));
    });
    fold
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-300 {
        return None;
    }
    let i = 1.0 / det;
    Some([
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * i, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * i, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * i],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * i, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * i, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * i],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * i, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * i, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * i],
    ])
}
