// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The photometric camera model: radiance in, recorded pixel values out.
//!
//! A fit compares renders against photographs, and photographs are not
//! radiance measurements taken under one fixed camera. Between two frames the
//! exposure and white balance change; within one, the lens darkens the corners
//! (differently per colour channel) and the sensor pipeline bends the
//! response. With no model of that, the only thing a fit can change to explain
//! a brighter view is the scene, and it does - brighter gaussians on one side,
//! floaters in front of a vignetted corner.
//!
//! The decomposition follows PPISP (Deutsch et al., 2026), implemented here
//! from its description rather than its code, into factors that are each
//! physically meaningful and each owned by the thing that varies:
//!
//! ```text
//!   radiance L_c
//!     x 2^(hint_v + e_v)            per VIEW   exposure (log2, EXIF hint + residual)
//!     x exp(b_vc)                   per VIEW   white balance (log gains)
//!     x (1 + a1 r² + a2 r⁴ + a3 r⁶) per SENSOR chromatic vignetting, r = radius / half-diagonal
//!     -> (x + ε)^γ_c - ε^γ_c        per SENSOR response curve, γ = exp(ρ)
//!     -> sRGB OETF                  only for a scene-linear fit against encoded photos
//! ```
//!
//! ## Gauges
//!
//! Several of these trade exactly against the scene: brighten every gaussian
//! and darken every exposure and nothing renders differently. Such a direction
//! is not something data can decide, so it is FIXED rather than regularized:
//! after every step the exposure residuals are centred across views, and the
//! white-balance gains are double-centred (zero mean over channels within a
//! view, since exposure owns brightness; zero mean over views per channel,
//! since the scene owns its colour). What a fitted scene shows from the
//! identity camera is then the capture's AVERAGE camera, which is what a
//! novel view should use.
//!
//! The directions data can decide but weakly (vignetting, the response curve)
//! carry a small identity prior and are switched on late, after the scene has
//! taken the shape the images agree on: a response curve that is free from
//! the first iteration can explain part of the scene's own contrast.
//!
//! ## Scene-linear fitting
//!
//! With [`ColorSpace::SceneLinear`] the scene stores linear radiance and the
//! sRGB transfer sits INSIDE the camera, so photographs are compared in the
//! encoding they were recorded in and exposure brackets are simply several
//! measurements of one radiance. Pixels that clipped at the sensor carry no
//! information about how bright the scene was and are down-weighted to zero
//! ([`IspCfg::clip`]). Standard splat viewers expect display-referred colour,
//! so [`bake_display`] converts a linear scene for export.
//!
//! Swedish Embedded AB implements photometrically calibrated 3D reconstruction
//! for its clients. If your team needs captures with changing exposure, white
//! balance or HDR brackets turned into consistent scenes, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::types::{Camera, Splats};

/// What the scene's colours mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// Display-referred, like every Inria-convention PLY: the colour a viewer
    /// shows is the colour stored. The camera model is then a set of
    /// corrections around identity.
    #[default]
    Display,
    /// Scene-linear radiance; the camera applies the sRGB transfer to its
    /// prediction before it is compared with an sRGB-encoded photograph.
    SceneLinear,
}

/// How a target photograph's values are encoded. Only consulted by a
/// [`ColorSpace::SceneLinear`] fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Encoding {
    /// An ordinary 8-bit photograph.
    #[default]
    Srgb,
    /// Already linear (EXR/HDR input): no transfer function to model.
    Linear,
}

/// Camera-model configuration. Stage starts are FRACTIONS of the fit.
#[derive(Clone, Copy, Debug)]
pub struct IspCfg {
    pub color_space: ColorSpace,
    /// Adam step for every camera parameter (log2 stops, log gains, vignetting
    /// coefficients and log gamma are all O(1) quantities).
    pub lr: f32,
    /// When exposure and white balance start moving.
    pub exposure_after: f32,
    /// When the vignetting polynomial starts moving.
    pub vignetting_after: f32,
    /// When the response curve starts moving. It is the parameter most able
    /// to impersonate the scene, so it comes last.
    pub response_after: f32,
    /// Weight of the pull toward the identity camera on the parameters that
    /// are not gauge-fixed (vignetting, response), relative to the per-pixel
    /// mean loss.
    pub prior: f32,
    /// A target pixel whose brightest channel reaches this is treated as
    /// clipped and not supervised; the weight ramps down over the last 5%
    /// below it. 0 disables clip handling.
    pub clip: f32,
}

impl Default for IspCfg {
    fn default() -> Self {
        IspCfg {
            color_space: ColorSpace::Display,
            lr: 1e-2,
            exposure_after: 0.0,
            vignetting_after: 0.3,
            response_after: 0.5,
            prior: 1e-5,
            clip: 0.99,
        }
    }
}

/// Parameters per view and per sensor in the flat vector.
const PER_VIEW: usize = 4; // exposure residual, 3 white-balance log gains
const PER_SENSOR: usize = 12; // 3x3 vignetting coefficients, 3 log gammas
const EPS: f32 = 1e-3;
const VIG_FLOOR: f32 = 0.05;

/// sRGB opto-electronic transfer, extended linearly past both ends so a
/// prediction outside [0,1] still has a gradient.
pub fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else if x <= 1.0 {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    } else {
        1.0 + (1.055 / 2.4) * (x - 1.0)
    }
}

fn srgb_encode_grad(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        12.92
    } else if x <= 1.0 {
        (1.055 / 2.4) * x.powf(1.0 / 2.4 - 1.0)
    } else {
        1.055 / 2.4
    }
}

/// Inverse of [`srgb_encode`] on [0,1].
pub fn srgb_decode(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// The fitted camera model of one capture.
#[derive(Clone, Debug)]
pub struct Isp {
    cfg: IspCfg,
    view_sensor: Vec<usize>,
    view_linear_target: Vec<bool>,
    hints: Vec<f32>,
    sensors: usize,
    theta: Vec<f32>,
    grad: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl Isp {
    /// One set of per-view parameters per entry of `views`, which gives each
    /// view's `(sensor, exposure hint in log2 stops, target encoding)`.
    pub fn new(cfg: IspCfg, views: &[(usize, f32, Encoding)]) -> Isp {
        let sensors = views.iter().map(|v| v.0 + 1).max().unwrap_or(1);
        let n = views.len() * PER_VIEW + sensors * PER_SENSOR;
        Isp {
            view_sensor: views.iter().map(|v| v.0).collect(),
            view_linear_target: views.iter().map(|v| v.2 == Encoding::Linear).collect(),
            hints: views.iter().map(|v| v.1).collect(),
            sensors,
            cfg,
            theta: vec![0.0; n],
            grad: vec![0.0; n],
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    pub fn cfg(&self) -> &IspCfg {
        &self.cfg
    }

    pub fn views(&self) -> usize {
        self.view_sensor.len()
    }

    fn view_at(&self, v: usize) -> usize {
        v * PER_VIEW
    }

    fn sensor_at(&self, s: usize) -> usize {
        self.view_sensor.len() * PER_VIEW + s * PER_SENSOR
    }

    /// The exposure of view `v` in log2 stops relative to the capture's mean
    /// camera, hint included.
    pub fn exposure(&self, v: usize) -> f32 {
        self.hints[v] + self.theta[self.view_at(v)]
    }

    /// White-balance log gains of view `v`.
    pub fn white_balance(&self, v: usize) -> [f32; 3] {
        let o = self.view_at(v) + 1;
        [self.theta[o], self.theta[o + 1], self.theta[o + 2]]
    }

    /// Lens transmission of sensor `s` at normalized radius `r` (1 = corner).
    pub fn vignetting(&self, s: usize, r: f32) -> [f32; 3] {
        let o = self.sensor_at(s);
        let r2 = r * r;
        std::array::from_fn(|c| {
            let a = &self.theta[o + c * 3..o + c * 3 + 3];
            (1.0 + a[0] * r2 + a[1] * r2 * r2 + a[2] * r2 * r2 * r2).max(VIG_FLOOR)
        })
    }

    /// Response-curve exponents of sensor `s`.
    pub fn gamma(&self, s: usize) -> [f32; 3] {
        let o = self.sensor_at(s) + 9;
        std::array::from_fn(|c| self.theta[o + c].exp())
    }

    fn encodes(&self, v: usize) -> bool {
        self.cfg.color_space == ColorSpace::SceneLinear && !self.view_linear_target[v]
    }

    /// Per-pixel supervision weight from the target alone: 0 where the
    /// photograph clipped, ramping to 1 over the 5% below the clip level.
    pub fn clip_weight(&self, target: &[f32]) -> Vec<f32> {
        let hi = self.cfg.clip;
        target
            .chunks_exact(3)
            .map(|p| {
                if hi <= 0.0 {
                    return 1.0;
                }
                let m = p[0].max(p[1]).max(p[2]);
                ((hi - m) / (0.05 * hi)).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// What view `v`'s camera records given `radiance` (interleaved RGB for
    /// the whole frame of `cam`).
    pub fn forward(&self, v: usize, cam: &Camera, radiance: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; radiance.len()];
        let row = 3 * cam.width as usize;
        backend_cpu::par::rows_mut(&mut out, row, |y, dst| {
            self.walk_row(v, cam, y, &radiance[y * row..y * row + row], |x, c, px| dst[x * 3 + c] = px.pred);
        });
        out
    }

    /// Given dLoss/d(prediction) for view `v`, return dLoss/d(radiance) and
    /// accumulate the camera parameters' own gradient for the next
    /// [`Isp::step`].
    pub fn backward(&mut self, v: usize, cam: &Camera, radiance: &[f32], dpred: &[f32]) -> Vec<f32> {
        let row = 3 * cam.width as usize;
        // rows in parallel, each with its own partial parameter gradient
        let parts: Vec<(Vec<f32>, [f64; PER_VIEW + PER_SENSOR])> = backend_cpu::par::map(cam.height as usize, |y| {
            let mut drad = vec![0.0f32; row];
            let mut g = [0.0f64; PER_VIEW + PER_SENSOR];
            let up = &dpred[y * row..y * row + row];
            self.walk_row(v, cam, y, &radiance[y * row..y * row + row], |x, c, px| {
                let dy = up[x * 3 + c] * px.dpred_dy;
                let dx = dy * px.dy_dx;
                drad[x * 3 + c] = dx * px.gain * px.vig;
                let dxs = (dx * px.x) as f64;
                g[0] += dxs * std::f64::consts::LN_2;
                g[1 + c] += dxs;
                if px.vig > VIG_FLOOR {
                    let base = (dx * px.gain * px.radiance) as f64;
                    let r2 = px.r2 as f64;
                    g[PER_VIEW + c * 3] += base * r2;
                    g[PER_VIEW + c * 3 + 1] += base * r2 * r2;
                    g[PER_VIEW + c * 3 + 2] += base * r2 * r2 * r2;
                }
                g[PER_VIEW + 9 + c] += (dy * px.dy_drho) as f64;
            });
            (drad, g)
        });
        let (vo, so) = (self.view_at(v), self.sensor_at(self.view_sensor[v]));
        let mut drad = Vec::with_capacity(radiance.len());
        for (d, g) in parts {
            drad.extend_from_slice(&d);
            for (k, gk) in g[..PER_VIEW].iter().enumerate() {
                self.grad[vo + k] += gk;
            }
            for (k, gk) in g[PER_VIEW..].iter().enumerate() {
                self.grad[so + k] += gk;
            }
        }
        drad
    }

    /// The camera applied to row `y` of the frame, `radiance` being that row.
    fn walk_row(&self, v: usize, cam: &Camera, y: usize, radiance: &[f32], mut f: impl FnMut(usize, usize, Px)) {
        let w = cam.width as usize;
        let ev = self.exposure(v);
        let wb = self.white_balance(v);
        let so = self.sensor_at(self.view_sensor[v]);
        let gamma = self.gamma(self.view_sensor[v]);
        let gain: [f32; 3] = std::array::from_fn(|c| 2f32.powf(ev) * wb[c].exp());
        let norm = 0.25 * (cam.width as f32).powi(2) + 0.25 * (cam.height as f32).powi(2);
        let encode = self.encodes(v);
        let dy2 = (y as f32 + 0.5 - cam.cy).powi(2);
        for x in 0..w {
            let r2 = ((x as f32 + 0.5 - cam.cx).powi(2) + dy2) / norm;
            for c in 0..3 {
                let a = &self.theta[so + c * 3..so + c * 3 + 3];
                let vig = (1.0 + a[0] * r2 + a[1] * r2 * r2 + a[2] * r2 * r2 * r2).max(VIG_FLOOR);
                let l = radiance[x * 3 + c];
                let xv = gain[c] * vig * l;
                // response: (x + ε)^γ - ε^γ, flat below zero
                let g = gamma[c];
                let (y_out, dy_dx, dy_drho) = if xv <= 0.0 {
                    (0.0, 0.0, 0.0)
                } else if g == 1.0 {
                    // the identity curve, which is what the response is until
                    // its stage starts: no transcendental per pixel for it
                    (xv, 1.0, xv * (xv + EPS).ln() + EPS * ((xv + EPS).ln() - EPS.ln()))
                } else {
                    let b = xv + EPS;
                    let yb = b.powf(g);
                    let ye = EPS.powf(g);
                    (yb - ye, g * b.powf(g - 1.0), g * (yb * b.ln() - ye * EPS.ln()))
                };
                let (pred, dpred_dy) = if encode { (srgb_encode(y_out), srgb_encode_grad(y_out)) } else { (y_out, 1.0) };
                f(x, c, Px { pred, dpred_dy, dy_dx, dy_drho, x: xv, gain: gain[c], vig, radiance: l, r2 });
            }
        }
    }

    /// One Adam step on the stages that are active at `progress` (fraction of
    /// the fit done), then the gauge fixes. Clears the accumulated gradient.
    pub fn step(&mut self, progress: f32) {
        let nv = self.views();
        let active = |k: usize| -> bool {
            if k < nv * PER_VIEW {
                progress >= self.cfg.exposure_after
            } else {
                let j = (k - nv * PER_VIEW) % PER_SENSOR;
                if j < 9 { progress >= self.cfg.vignetting_after } else { progress >= self.cfg.response_after }
            }
        };
        // identity prior on the parameters no gauge fixes
        let prior = self.cfg.prior as f64;
        for k in nv * PER_VIEW..self.theta.len() {
            self.grad[k] += 2.0 * prior * self.theta[k] as f64;
        }
        self.t += 1;
        let (b1, b2) = (0.9f64, 0.999f64);
        let (bc1, bc2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for k in 0..self.theta.len() {
            if !active(k) {
                continue;
            }
            let g = self.grad[k];
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g;
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g;
            let step = self.cfg.lr as f64 * (self.m[k] / bc1) / ((self.v[k] / bc2).sqrt() + 1e-12);
            self.theta[k] -= step as f32;
        }
        self.grad.fill(0.0);
        self.fix_gauges();
    }

    fn fix_gauges(&mut self) {
        let nv = self.views();
        if nv == 0 {
            return;
        }
        // exposure residuals: zero mean across views
        let mean = (0..nv).map(|v| self.theta[v * PER_VIEW]).sum::<f32>() / nv as f32;
        for v in 0..nv {
            self.theta[v * PER_VIEW] -= mean;
        }
        // white balance: double-centred
        for v in 0..nv {
            let o = v * PER_VIEW + 1;
            let m = (self.theta[o] + self.theta[o + 1] + self.theta[o + 2]) / 3.0;
            for c in 0..3 {
                self.theta[o + c] -= m;
            }
        }
        for c in 0..3 {
            let m = (0..nv).map(|v| self.theta[v * PER_VIEW + 1 + c]).sum::<f32>() / nv as f32;
            for v in 0..nv {
                self.theta[v * PER_VIEW + 1 + c] -= m;
            }
        }
    }

    /// One line per view and sensor, for a fit's log.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for v in 0..self.views() {
            let wb = self.white_balance(v);
            s += &format!("  view {v:3}: exposure {:+.3} EV, wb [{:+.3} {:+.3} {:+.3}]\n", self.exposure(v), wb[0], wb[1], wb[2]);
        }
        for k in 0..self.sensors {
            let c = self.vignetting(k, 1.0);
            let g = self.gamma(k);
            s += &format!(
                "  sensor {k}: corner transmission [{:.3} {:.3} {:.3}], gamma [{:.3} {:.3} {:.3}]\n",
                c[0], c[1], c[2], g[0], g[1], g[2]
            );
        }
        s
    }
}

/// Per-pixel, per-channel quantities of one [`Isp::walk`].
struct Px {
    pred: f32,
    dpred_dy: f32,
    dy_dx: f32,
    dy_drho: f32,
    x: f32,
    gain: f32,
    vig: f32,
    radiance: f32,
    r2: f32,
}

/// Convert a scene-linear scene to display-referred colour for viewers that
/// show stored colour directly (every Inria-convention one).
///
/// Exact for an opaque surface seen through the mean camera; blends are
/// encoded after compositing in the real camera and before it here, which is
/// the approximation every "bake the tone curve into the splats" export makes.
/// Higher-order SH is scaled by the transfer's slope at the base colour, its
/// first-order expansion.
pub fn bake_display(scene: &Splats) -> Splats {
    let mut out = scene.clone();
    let shk = match &scene.sh_rest {
        Some((_, r)) if !scene.is_empty() => r.len() / scene.len() / 3,
        _ => 0,
    };
    for i in 0..scene.len() {
        for c in 0..3 {
            let l = scene.colors[i * 3 + c].max(0.0);
            out.colors[i * 3 + c] = srgb_encode(l);
            if shk > 0 {
                let slope = srgb_encode_grad(l.max(1e-4));
                if let Some((_, r)) = &mut out.sh_rest {
                    let o = i * 3 * shk + c * shk;
                    for v in &mut r[o..o + shk] {
                        *v *= slope;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera::look_at([0.0; 3], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 60.0, 7, 5)
    }

    /// The backward is the derivative of the forward, for every parameter and
    /// for the radiance, in both colour spaces.
    #[test]
    fn backward_is_the_derivative_of_forward() {
        for space in [ColorSpace::Display, ColorSpace::SceneLinear] {
            let cfg = IspCfg { color_space: space, ..IspCfg::default() };
            let mut isp = Isp::new(cfg, &[(0, 0.3, Encoding::Srgb), (0, -0.2, Encoding::Srgb)]);
            for (k, t) in isp.theta.iter_mut().enumerate() {
                *t = 0.05 * ((k * 7 % 11) as f32 - 5.0) / 5.0;
            }
            let c = cam();
            let rad: Vec<f32> = (0..7 * 5 * 3).map(|i| 0.05 + 0.9 * ((i * 13 % 17) as f32 / 17.0)).collect();
            let up: Vec<f32> = (0..rad.len()).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
            let loss = |isp: &Isp, rad: &[f32]| -> f64 {
                isp.forward(1, &c, rad).iter().zip(&up).map(|(a, b)| (a * b) as f64).sum()
            };
            let drad = isp.backward(1, &c, &rad, &up);
            let pg = isp.grad.clone();
            let eps = 1e-3f32;
            for i in [0usize, 17, 40, 104] {
                let (mut a, mut b) = (rad.clone(), rad.clone());
                a[i] += eps;
                b[i] -= eps;
                let fd = (loss(&isp, &a) - loss(&isp, &b)) / (2.0 * eps as f64);
                assert!((fd - drad[i] as f64).abs() < 2e-2 * fd.abs().max(1.0), "{space:?} radiance {i}: {fd} vs {}", drad[i]);
            }
            // view 0 is not the view differentiated
            for (k, &want) in pg.iter().enumerate().skip(PER_VIEW) {
                let mut a = isp.clone();
                a.theta[k] += eps;
                let mut b = isp.clone();
                b.theta[k] -= eps;
                let fd = (loss(&a, &rad) - loss(&b, &rad)) / (2.0 * eps as f64);
                assert!((fd - want).abs() < 2e-2 * fd.abs().max(1.0), "{space:?} parameter {k}: {fd} vs {want}");
            }
        }
    }

    /// The identity camera records radiance unchanged (display) or exactly
    /// sRGB-encoded (scene-linear).
    #[test]
    fn a_fresh_model_is_the_identity_camera() {
        let c = cam();
        let rad: Vec<f32> = (0..7 * 5 * 3).map(|i| i as f32 / 105.0).collect();
        let d = Isp::new(IspCfg::default(), &[(0, 0.0, Encoding::Srgb)]).forward(0, &c, &rad);
        let l = Isp::new(IspCfg { color_space: ColorSpace::SceneLinear, ..IspCfg::default() }, &[(0, 0.0, Encoding::Srgb)])
            .forward(0, &c, &rad);
        for i in 0..rad.len() {
            assert!((d[i] - rad[i]).abs() < 2e-3, "display {i}: {} vs {}", d[i], rad[i]);
            assert!((l[i] - srgb_encode(rad[i])).abs() < 3e-3, "linear {i}: {} vs {}", l[i], srgb_encode(rad[i]));
        }
    }
}
