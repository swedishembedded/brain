// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The photometric camera model: radiance in, recorded pixel values out.
//!
//! A fit compares renders against photographs, and photographs are not
//! radiance measurements taken under one fixed camera. Between two frames the
//! exposure and white balance change; within one, the lens darkens the corners
//! (differently per colour channel), the sensor's colour matrix mixes the
//! channels, the pipeline bends the response, and a phone's local tone mapping
//! brightens one part of the frame and not another. With no model of that, the
//! only thing a fit can change to explain a brighter view is the scene, and it
//! does - brighter gaussians on one side, floaters in front of a vignetted
//! corner.
//!
//! The global decomposition follows PPISP (Deutsch et al., 2026), implemented
//! here from its description rather than its code, into factors that are each
//! physically meaningful and each owned by the thing that varies; the local
//! residual is the bilateral grid of Wang et al., "Bilateral Guided Radiance
//! Field Processing" (SIGGRAPH Asia 2024), from the paper:
//!
//! ```text
//!   radiance L_c
//!     x 2^(hint_v + e_v)            per VIEW   exposure (log2, EXIF hint + residual)
//!     x exp(b_vc)                   per VIEW   white balance (log gains)
//!     x (1 + a1 r² + a2 r⁴ + a3 r⁶) per SENSOR chromatic vignetting, r = radius / half-diagonal
//!     -> M x, rows of M summing to 1 per SENSOR colour correction matrix
//!     -> f_c, monotone              per SENSOR response curve, per channel
//!     -> sRGB OETF                  only for a scene-linear fit against encoded photos
//!     -> A(x, y, luma) [e; 1]       per VIEW   bilateral grid of 3x4 affine transforms (optional)
//! ```
//!
//! The response curve is piecewise linear with knots one stop apart
//! (0, 2^-7 .. 2^-1, and on past 1/2 with the last slope), each segment's slope
//! `exp(δ)`: monotone by construction, the identity at δ = 0, and smooth in the
//! log-slope domain through a second-difference prior. The grid has
//! `cells[0] x cells[1]` cells across the frame and `cells[2]` along the luma
//! of what the global chain produced; it is sliced trilinearly, starts at the
//! identity, and carries a total-variation prior, so it can only express what
//! varies smoothly over the frame and the tone scale.
//!
//! ## Gauges
//!
//! Several of these trade exactly against the scene: brighten every gaussian
//! and darken every exposure and nothing renders differently. Such a direction
//! is not something data can decide, so it is FIXED rather than regularized,
//! after every step:
//!
//! * exposure residuals are centred across views;
//! * white-balance gains are double-centred - zero mean over channels within a
//!   view (exposure owns brightness), zero mean over views per channel (the
//!   scene owns its colour);
//! * a colour matrix's rows sum to 1, so it maps grey to grey and cannot act
//!   as a white balance; and the matrices are centred across sensors, since a
//!   matrix every sensor shares is the scene's colour. A capture from ONE
//!   sensor therefore keeps the identity: its matrix is not observable;
//! * the bilateral grids are centred across views, cell by cell.
//!
//! What a fitted scene shows from the identity camera is then the capture's
//! AVERAGE camera, which is what a novel view should use ([`NovelShot`]).
//!
//! The response curve has gauges no centring can fix, and they bound what a
//! fit can claim to recover: each channel's curve only up to the scale of its
//! input (the scene's colour per channel is free), and with free exposures
//! not even its shape along a power law - `(2^h L)^γ` is exactly the identity
//! curve seen at exposure `γ h` of the scene `L^γ`. Known exposures (EXIF,
//! brackets) are what pin it down, as they do in Debevec and Malik's
//! recovery; otherwise the identity and smoothness priors choose.
//!
//! The directions data can decide but weakly (vignetting, the colour matrix,
//! the response curve, the grid) carry a small identity prior and are switched
//! on late, after the scene has taken the shape the images agree on: a
//! response curve that is free from the first iteration can explain part of
//! the scene's own contrast.
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
//! ## Where it runs
//!
//! A fit runs the model on the device ([`DeviceIsp`], kernels `isp_pixel` and
//! `isp_grid_grad`): the renderer's output never leaves it. The host
//! implementation here ([`Isp::render`], [`Isp::backward`]) is the same math
//! and the kernels' test oracle, and serves render-time use and held-out
//! evaluation ([`Isp::fit_novel`]).
//!
//! Swedish Embedded AB implements photometrically calibrated 3D reconstruction
//! for its clients. If your team needs captures with changing exposure, white
//! balance, tone mapping or HDR brackets turned into consistent scenes, you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};

use crate::types::{Camera, Splats};
use crate::Kernels;

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
    /// Already linear (EXR/HDR input, or a photograph decoded through its own
    /// transfer curve): no transfer function to model.
    Linear,
}

/// The per-view bilateral grid (Wang et al. 2024).
#[derive(Clone, Copy, Debug)]
pub struct GridCfg {
    /// Cells across the frame's width, its height, and the luma axis; each at
    /// least 2. The paper's 16 x 16 x 8 is the default.
    pub cells: [usize; 3],
    /// When the grids start moving, as a fraction of the fit: after the
    /// global camera has had its say.
    pub after: f32,
    /// Adam step of the grid coefficients.
    pub lr: f32,
    /// Weight of the total-variation prior (mean squared difference between
    /// neighbouring cells, over the cells), relative to the per-pixel mean
    /// loss.
    pub tv: f32,
}

impl Default for GridCfg {
    fn default() -> Self {
        GridCfg { cells: [16, 16, 8], after: 0.5, lr: 2e-3, tv: 1e-2 }
    }
}

/// Camera-model configuration. Stage starts are FRACTIONS of the fit.
#[derive(Clone, Copy, Debug)]
pub struct IspCfg {
    pub color_space: ColorSpace,
    /// Adam step for every global camera parameter (log2 stops, log gains,
    /// vignetting coefficients, colour matrix entries and log slopes are all
    /// O(1) quantities).
    pub lr: f32,
    /// When exposure and white balance start moving.
    pub exposure_after: f32,
    /// When the vignetting polynomial starts moving.
    pub vignetting_after: f32,
    /// When the colour correction matrices start moving.
    pub ccm_after: f32,
    /// When the response curve starts moving. It is the global parameter
    /// most able to impersonate the scene, so it comes last.
    pub response_after: f32,
    /// Weight of the pull toward the identity camera on the parameters that
    /// are not gauge-fixed (vignetting, colour matrix, response, grid),
    /// relative to the per-pixel mean loss.
    pub prior: f32,
    /// Weight of the response curve's smoothness prior: the squared second
    /// difference of its log slopes, summed over segments and channels.
    pub response_smooth: f32,
    /// A target pixel whose brightest channel reaches this is treated as
    /// clipped and not supervised; the weight ramps down over the last 5%
    /// below it. 0 disables clip handling.
    pub clip: f32,
    /// The per-view bilateral grid; `None` = the global camera only.
    pub grid: Option<GridCfg>,
}

impl Default for IspCfg {
    fn default() -> Self {
        IspCfg {
            color_space: ColorSpace::Display,
            lr: 1e-2,
            exposure_after: 0.0,
            vignetting_after: 0.3,
            ccm_after: 0.3,
            response_after: 0.5,
            prior: 1e-5,
            response_smooth: 1e-3,
            clip: 0.99,
            grid: None,
        }
    }
}

/// Segments of the response curve. The kernels unroll over exactly this
/// many (`lib/isp.wgsl`).
pub const RESPONSE_SEGMENTS: usize = 8;
/// Parameters per view in the flat vector: exposure residual, 3 white-balance
/// log gains.
const PER_VIEW: usize = 4;
/// Within a sensor's block: 3x3 vignetting coefficients (channel-major), the
/// colour matrix's 6 off-diagonal entries, 3 x 8 response log slopes
/// (channel-major).
const VIG_AT: usize = 0;
const CCM_AT: usize = 9;
const CRF_AT: usize = 15;
const PER_SENSOR: usize = CRF_AT + 3 * RESPONSE_SEGMENTS;
/// The colour matrix's free entries; the diagonal follows from rows summing
/// to 1.
const OFFDIAG: [(usize, usize); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
/// d(camera) channels the device reduces: gain 3, vignetting 9, colour
/// matrix 9, response slopes 24 (see `isp_pixel.wgsl`).
const CAM_GRADS: usize = 21 + 3 * RESPONSE_SEGMENTS;
/// Coefficients of one grid cell: a row-major 3x4 affine transform.
const CELL: usize = 12;
const VIG_FLOOR: f32 = 0.05;
/// The grid's guide: Rec.709 luma.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// Words of the kernels' uniform (`IspView` + `IspCam`).
const PARAM_WORDS: usize = 12 + 4 * (7 + RESPONSE_SEGMENTS);

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

/// Which camera a frame is seen through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shot {
    /// Training view `v`, with every parameter the fit gave it.
    Training(usize),
    /// A view the fit never saw: its sensor's lens, colour matrix and
    /// response, and the per-view settings given.
    Novel(NovelShot),
}

/// The per-view settings of a view the fit never saw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NovelShot {
    pub sensor: usize,
    /// Log2 stops relative to the capture, like [`crate::opt::TargetView::exposure`].
    pub exposure: f32,
    /// Log gains per channel.
    pub white_balance: [f32; 3],
    pub encoding: Encoding,
}

impl NovelShot {
    /// The capture's average camera at the exposure a view is known to have
    /// had (EXIF; 0 when unknown): what a novel view renders through.
    pub fn neutral(sensor: usize, exposure: f32, encoding: Encoding) -> NovelShot {
        NovelShot { sensor, exposure, white_balance: [0.0; 3], encoding }
    }
}

/// One frame's camera resolved from the parameters: exactly what the kernels
/// read in `IspCam`.
#[derive(Clone, Copy, Debug)]
struct Resolved {
    gain: [f32; 3],
    /// `vig[k][c]`: coefficient of r^(2k+2), channel c.
    vig: [[f32; 3]; 3],
    ccm: [[f32; 3]; 3],
    /// `slope[j][c]`: response slope of segment j, channel c.
    slope: [[f32; 3]; RESPONSE_SEGMENTS],
    encode: bool,
}

/// The frame geometry the per-pixel chain needs.
struct Frame<'a> {
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    inv_norm: f32,
    /// The grid's deviation from identity and its cell counts.
    grid: Option<(&'a [f32], [usize; 3])>,
}

impl Frame<'_> {
    fn new<'a>(cam: &Camera, grid: Option<(&'a [f32], [usize; 3])>) -> Frame<'a> {
        let (w, h) = (cam.width as f32, cam.height as f32);
        Frame { w: cam.width as usize, h: cam.height as usize, cx: cam.cx, cy: cam.cy, inv_norm: 1.0 / (0.25 * w * w + 0.25 * h * h), grid }
    }

    fn r2(&self, x: usize, y: usize) -> f32 {
        let dx = x as f32 + 0.5 - self.cx;
        let dy = y as f32 + 0.5 - self.cy;
        (dx * dx + dy * dy) * self.inv_norm
    }

    /// Continuous grid coordinates of pixel (x, y) with guide `luma`.
    fn grid_coord(&self, cells: [usize; 3], x: usize, y: usize, luma: f32) -> [f32; 3] {
        [
            (x as f32 + 0.5) / self.w as f32 * (cells[0] - 1) as f32,
            (y as f32 + 0.5) / self.h as f32 * (cells[1] - 1) as f32,
            luma.clamp(0.0, 1.0) * (cells[2] - 1) as f32,
        ]
    }
}

fn grid_base(u: f32, g: usize) -> usize {
    (u.floor().max(0.0) as usize).min(g - 2)
}

/// How much of response segment `j` lies below `y` (`lib/isp.wgsl`'s
/// `isp_portion`).
fn portion(j: usize, y: f32) -> f32 {
    let t1 = 0.007_812_5f32;
    if j == 0 {
        return y.min(t1);
    }
    let mut t = t1;
    for _ in 1..j {
        t *= 2.0;
    }
    if j == RESPONSE_SEGMENTS - 1 {
        (y - t).max(0.0)
    } else {
        (y - t).clamp(0.0, t)
    }
}

fn crf(slope: &[[f32; 3]; RESPONSE_SEGMENTS], c: usize, y: f32) -> f32 {
    (0..RESPONSE_SEGMENTS).fold(0.0, |f, j| f + slope[j][c] * portion(j, y))
}

fn crf_slope(slope: &[[f32; 3]; RESPONSE_SEGMENTS], c: usize, y: f32) -> f32 {
    let mut d = slope[0][c];
    let mut t = 0.007_812_5f32;
    for s in &slope[1..] {
        if y >= t {
            d = s[c];
        }
        t *= 2.0;
    }
    d
}

/// Every stage of the global chain at one pixel.
struct Trace {
    vig: [f32; 3],
    x: [f32; 3],
    m: [f32; 3],
    y: [f32; 3],
    e: [f32; 3],
}

fn global(cam: &Resolved, l: [f32; 3], r2: f32) -> Trace {
    let (r4, r6) = (r2 * r2, r2 * r2 * r2);
    let vig: [f32; 3] = std::array::from_fn(|c| (1.0 + cam.vig[0][c] * r2 + cam.vig[1][c] * r4 + cam.vig[2][c] * r6).max(VIG_FLOOR));
    let x: [f32; 3] = std::array::from_fn(|c| cam.gain[c] * vig[c] * l[c]);
    let m: [f32; 3] = std::array::from_fn(|r| cam.ccm[r][0] * x[0] + cam.ccm[r][1] * x[1] + cam.ccm[r][2] * x[2]);
    let y: [f32; 3] = std::array::from_fn(|c| crf(&cam.slope, c, m[c]));
    let e = if cam.encode { y.map(srgb_encode) } else { y };
    Trace { vig, x, m, y, e }
}

fn luma(e: &[f32; 3]) -> f32 {
    LUMA[0] * e[0] + LUMA[1] * e[1] + LUMA[2] * e[2]
}

/// The grid's slice at a pixel: the interpolated 3x4 deviation `d`, its
/// derivative along the luma coordinate `z`, and the eight corners' cells and
/// weights.
struct Slice {
    d: [[f32; 4]; 3],
    z: [[f32; 4]; 3],
    corners: [(usize, f32); 8],
}

fn slice(grid: &[f32], cells: [usize; 3], u: [f32; 3]) -> Slice {
    let base = [grid_base(u[0], cells[0]), grid_base(u[1], cells[1]), grid_base(u[2], cells[2])];
    let f = [u[0] - base[0] as f32, u[1] - base[1] as f32, u[2] - base[2] as f32];
    let mut s = Slice { d: [[0.0; 4]; 3], z: [[0.0; 4]; 3], corners: [(0, 0.0); 8] };
    for q in 0..8 {
        let (bx, by, bz) = (q & 1, (q >> 1) & 1, q >> 2);
        let wx = if bx == 1 { f[0] } else { 1.0 - f[0] };
        let wy = if by == 1 { f[1] } else { 1.0 - f[1] };
        let (wz, sz) = if bz == 1 { (f[2], 1.0) } else { (1.0 - f[2], -1.0) };
        let cell = ((base[2] + bz) * cells[1] + base[1] + by) * cells[0] + base[0] + bx;
        let o = cell * CELL;
        let wxy = wx * wy;
        for r in 0..3 {
            for k in 0..4 {
                s.d[r][k] += wxy * wz * grid[o + r * 4 + k];
                s.z[r][k] += wxy * sz * grid[o + r * 4 + k];
            }
        }
        s.corners[q] = (cell, wxy * wz);
    }
    s
}

fn affine(a: &[f32; 4], e: &[f32; 3]) -> f32 {
    a[0] * e[0] + a[1] * e[1] + a[2] * e[2] + a[3]
}

/// One pixel forward.
fn pixel(cam: &Resolved, fr: &Frame, x: usize, y: usize, l: [f32; 3]) -> [f32; 3] {
    let tr = global(cam, l, fr.r2(x, y));
    match fr.grid {
        None => tr.e,
        Some((grid, cells)) => {
            let s = slice(grid, cells, fr.grid_coord(cells, x, y, luma(&tr.e)));
            std::array::from_fn(|r| tr.e[r] + affine(&s.d[r], &tr.e))
        }
    }
}

/// One pixel backward: dLoss/d(radiance), and the pixel's share of
/// dLoss/d(camera) added to `dcam` and of dLoss/d(grid) to `dgrid`.
#[allow(clippy::too_many_arguments)]
fn pixel_backward(cam: &Resolved, fr: &Frame, x: usize, y: usize, l: [f32; 3], dp: [f32; 3], dcam: &mut [f64; CAM_GRADS], dgrid: &mut [f64]) -> [f32; 3] {
    let r2 = fr.r2(x, y);
    let tr = global(cam, l, r2);
    let mut de = dp;
    if let Some((grid, cells)) = fr.grid {
        let lu = luma(&tr.e);
        let s = slice(grid, cells, fr.grid_coord(cells, x, y, lu));
        de = std::array::from_fn(|c| dp[c] + s.d[0][c] * dp[0] + s.d[1][c] * dp[1] + s.d[2][c] * dp[2]);
        if lu > 0.0 && lu < 1.0 {
            let g = (dp[0] * affine(&s.z[0], &tr.e) + dp[1] * affine(&s.z[1], &tr.e) + dp[2] * affine(&s.z[2], &tr.e)) * (cells[2] - 1) as f32;
            for c in 0..3 {
                de[c] += g * LUMA[c];
            }
        }
        let e4 = [tr.e[0], tr.e[1], tr.e[2], 1.0];
        for (cell, w) in s.corners {
            for r in 0..3 {
                for k in 0..4 {
                    dgrid[cell * CELL + r * 4 + k] += (w * dp[r] * e4[k]) as f64;
                }
            }
        }
    }
    let dy: [f32; 3] = if cam.encode { std::array::from_fn(|c| de[c] * srgb_encode_grad(tr.y[c])) } else { de };
    for j in 0..RESPONSE_SEGMENTS {
        for c in 0..3 {
            dcam[21 + 3 * j + c] += (dy[c] * portion(j, tr.m[c])) as f64;
        }
    }
    let dm: [f32; 3] = std::array::from_fn(|c| dy[c] * crf_slope(&cam.slope, c, tr.m[c]));
    let dx: [f32; 3] = std::array::from_fn(|c| cam.ccm[0][c] * dm[0] + cam.ccm[1][c] * dm[1] + cam.ccm[2][c] * dm[2]);
    for r in 0..3 {
        for c in 0..3 {
            dcam[12 + 3 * r + c] += (dm[r] * tr.x[c]) as f64;
        }
    }
    let (r4, r6) = (r2 * r2, r2 * r2 * r2);
    for c in 0..3 {
        dcam[c] += (dx[c] * tr.vig[c] * l[c]) as f64;
        if tr.vig[c] > VIG_FLOOR {
            let dv = dx[c] * cam.gain[c] * l[c];
            dcam[3 + c] += (dv * r2) as f64;
            dcam[6 + c] += (dv * r4) as f64;
            dcam[9 + c] += (dv * r6) as f64;
        }
    }
    std::array::from_fn(|c| dx[c] * cam.gain[c] * tr.vig[c])
}

/// The fitted camera model of one capture.
#[derive(Clone, Debug)]
pub struct Isp {
    cfg: IspCfg,
    view_sensor: Vec<usize>,
    view_linear_target: Vec<bool>,
    hints: Vec<f32>,
    sensors: usize,
    /// Coefficients per grid (0 without one).
    grid_len: usize,
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
        let grid_len = cfg.grid.map_or(0, |g| {
            assert!(g.cells.iter().all(|&c| c >= 2), "a bilateral grid needs at least 2 cells per axis, got {:?}", g.cells);
            g.cells.iter().product::<usize>() * CELL
        });
        let n = views.len() * (PER_VIEW + grid_len) + sensors * PER_SENSOR;
        Isp {
            view_sensor: views.iter().map(|v| v.0).collect(),
            view_linear_target: views.iter().map(|v| v.2 == Encoding::Linear).collect(),
            hints: views.iter().map(|v| v.1).collect(),
            sensors,
            grid_len,
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

    pub fn sensors(&self) -> usize {
        self.sensors
    }

    /// Every parameter, flat: per view (exposure residual, 3 white-balance
    /// log gains), per sensor (9 vignetting coefficients, 6 colour-matrix
    /// entries, 24 response log slopes), per view (grid deviation from
    /// identity, `cells` x 12). What a fitted camera IS; enough to save and
    /// restore it with [`Isp::set_params`].
    pub fn params(&self) -> &[f32] {
        &self.theta
    }

    /// Replace every parameter (the layout of [`Isp::params`]). No gauge is
    /// applied: the caller hands in a camera, not a step.
    pub fn set_params(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.theta.len(), "an Isp of this shape has {} parameters", self.theta.len());
        self.theta.copy_from_slice(params);
    }

    /// dLoss/d[`Isp::params`] accumulated by [`Isp::backward`] or
    /// [`DeviceIsp::backward`] since the last [`Isp::step`].
    pub fn gradient(&self) -> &[f64] {
        &self.grad
    }

    pub fn clear_gradient(&mut self) {
        self.grad.fill(0.0);
    }

    fn view_at(&self, v: usize) -> usize {
        v * PER_VIEW
    }

    fn sensor_at(&self, s: usize) -> usize {
        self.views() * PER_VIEW + s * PER_SENSOR
    }

    fn grid_at(&self, v: usize) -> usize {
        self.views() * PER_VIEW + self.sensors * PER_SENSOR + v * self.grid_len
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
        let o = self.sensor_at(s) + VIG_AT;
        let r2 = r * r;
        std::array::from_fn(|c| {
            let a = &self.theta[o + c * 3..o + c * 3 + 3];
            (1.0 + a[0] * r2 + a[1] * r2 * r2 + a[2] * r2 * r2 * r2).max(VIG_FLOOR)
        })
    }

    /// Colour correction matrix of sensor `s`, row-major; rows sum to 1.
    pub fn ccm(&self, s: usize) -> [[f32; 3]; 3] {
        let o = self.sensor_at(s) + CCM_AT;
        let mut m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (k, &(r, c)) in OFFDIAG.iter().enumerate() {
            m[r][c] = self.theta[o + k];
            m[r][r] -= self.theta[o + k];
        }
        m
    }

    fn slopes(&self, s: usize) -> [[f32; 3]; RESPONSE_SEGMENTS] {
        let o = self.sensor_at(s) + CRF_AT;
        std::array::from_fn(|j| std::array::from_fn(|c| self.theta[o + c * RESPONSE_SEGMENTS + j].exp()))
    }

    /// Response curve of sensor `s` at `y`, per channel.
    pub fn response(&self, s: usize, y: f32) -> [f32; 3] {
        let slope = self.slopes(s);
        std::array::from_fn(|c| crf(&slope, c, y))
    }

    /// View `v`'s bilateral grid, as deviations from the identity transform:
    /// cell (i, j, k) at `((k * cells[1] + j) * cells[0] + i) * 12`, a
    /// row-major 3x4 matrix. `None` without a grid.
    pub fn grid(&self, v: usize) -> Option<&[f32]> {
        (self.grid_len > 0).then(|| &self.theta[self.grid_at(v)..self.grid_at(v) + self.grid_len])
    }

    fn encodes(&self, linear_target: bool) -> bool {
        self.cfg.color_space == ColorSpace::SceneLinear && !linear_target
    }

    fn resolve(&self, shot: Shot) -> Resolved {
        let (sensor, exposure, wb, linear_target) = match shot {
            Shot::Training(v) => (self.view_sensor[v], self.exposure(v), self.white_balance(v), self.view_linear_target[v]),
            Shot::Novel(n) => (n.sensor, n.exposure, n.white_balance, n.encoding == Encoding::Linear),
        };
        assert!(sensor < self.sensors, "sensor {sensor} of a camera model fitted with {}", self.sensors);
        let o = self.sensor_at(sensor) + VIG_AT;
        Resolved {
            gain: std::array::from_fn(|c| 2f32.powf(exposure) * wb[c].exp()),
            vig: std::array::from_fn(|k| std::array::from_fn(|c| self.theta[o + c * 3 + k])),
            ccm: self.ccm(sensor),
            slope: self.slopes(sensor),
            encode: self.encodes(linear_target),
        }
    }

    fn frame<'a>(&'a self, shot: Shot, cam: &Camera) -> Frame<'a> {
        let grid = match (shot, self.cfg.grid) {
            (Shot::Training(v), Some(g)) => Some((self.grid(v).expect("grid"), g.cells)),
            _ => None,
        };
        Frame::new(cam, grid)
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

    /// What the camera of `shot` records given `radiance` (interleaved RGB
    /// for the whole frame of `cam`): a training view through everything the
    /// fit gave it, a novel one through its sensor and the settings given.
    pub fn render(&self, shot: Shot, cam: &Camera, radiance: &[f32]) -> Vec<f32> {
        let res = self.resolve(shot);
        let fr = self.frame(shot, cam);
        let mut out = vec![0.0f32; radiance.len()];
        let row = 3 * fr.w;
        backend_cpu::par::rows_mut(&mut out, row, |y, dst| {
            for x in 0..fr.w {
                let p = y * row + x * 3;
                dst[x * 3..x * 3 + 3].copy_from_slice(&pixel(&res, &fr, x, y, [radiance[p], radiance[p + 1], radiance[p + 2]]));
            }
        });
        out
    }

    /// What training view `v`'s camera records: [`Isp::render`] of
    /// [`Shot::Training`].
    pub fn forward(&self, v: usize, cam: &Camera, radiance: &[f32]) -> Vec<f32> {
        self.render(Shot::Training(v), cam, radiance)
    }

    /// Given dLoss/d(prediction) for training view `v`, return
    /// dLoss/d(radiance) and accumulate the camera parameters' own gradient
    /// for the next [`Isp::step`]. The host twin of [`DeviceIsp::backward`].
    pub fn backward(&mut self, v: usize, cam: &Camera, radiance: &[f32], dpred: &[f32]) -> Vec<f32> {
        let res = self.resolve(Shot::Training(v));
        let (grid_len, row) = (self.grid_len, 3 * cam.width as usize);
        let h = cam.height as usize;
        // bands of rows in parallel, each with its own partial gradient
        let bands = h.clamp(1, 16);
        let parts: Vec<(Vec<f32>, [f64; CAM_GRADS], Vec<f64>)> = {
            let fr = self.frame(Shot::Training(v), cam);
            backend_cpu::par::map(bands, |b| {
                let (y0, y1) = (b * h / bands, (b + 1) * h / bands);
                let mut drad = vec![0.0f32; (y1 - y0) * row];
                let mut dcam = [0.0f64; CAM_GRADS];
                let mut dgrid = vec![0.0f64; grid_len];
                for y in y0..y1 {
                    for x in 0..fr.w {
                        let p = y * row + x * 3;
                        let l = [radiance[p], radiance[p + 1], radiance[p + 2]];
                        let d = pixel_backward(&res, &fr, x, y, l, [dpred[p], dpred[p + 1], dpred[p + 2]], &mut dcam, &mut dgrid);
                        drad[(y - y0) * row + x * 3..(y - y0) * row + x * 3 + 3].copy_from_slice(&d);
                    }
                }
                (drad, dcam, dgrid)
            })
        };
        let mut drad = Vec::with_capacity(radiance.len());
        let mut dcam = [0.0f64; CAM_GRADS];
        let mut dgrid = vec![0.0f64; grid_len];
        for (d, c, g) in parts {
            drad.extend_from_slice(&d);
            for (a, b) in dcam.iter_mut().zip(&c) {
                *a += b;
            }
            for (a, b) in dgrid.iter_mut().zip(&g) {
                *a += b;
            }
        }
        self.accumulate(v, &res, &dcam, &dgrid);
        drad
    }

    /// Chain a frame's dLoss/d(camera) and dLoss/d(grid) to the parameters.
    fn accumulate(&mut self, v: usize, res: &Resolved, dcam: &[f64; CAM_GRADS], dgrid: &[f64]) {
        let vo = self.view_at(v);
        let so = self.sensor_at(self.view_sensor[v]);
        for c in 0..3 {
            let dg = dcam[c] * res.gain[c] as f64;
            self.grad[vo] += dg * std::f64::consts::LN_2;
            self.grad[vo + 1 + c] += dg;
            for k in 0..3 {
                self.grad[so + VIG_AT + c * 3 + k] += dcam[3 + 3 * k + c];
            }
        }
        for (k, &(r, c)) in OFFDIAG.iter().enumerate() {
            self.grad[so + CCM_AT + k] += dcam[12 + 3 * r + c] - dcam[12 + 3 * r + r];
        }
        for j in 0..RESPONSE_SEGMENTS {
            for c in 0..3 {
                self.grad[so + CRF_AT + c * RESPONSE_SEGMENTS + j] += dcam[21 + 3 * j + c] * res.slope[j][c] as f64;
            }
        }
        if self.grid_len > 0 {
            let go = self.grid_at(v);
            for (a, b) in self.grad[go..go + self.grid_len].iter_mut().zip(dgrid) {
                *a += b;
            }
        }
    }

    /// One Adam step on the stages that are active at `progress` (fraction of
    /// the fit done), then the gauge fixes. Clears the accumulated gradient.
    pub fn step(&mut self, progress: f32) {
        let (nv, ns) = (self.views(), self.sensors);
        let sensors_at = nv * PER_VIEW;
        let grids_at = sensors_at + ns * PER_SENSOR;
        let cfg = self.cfg;
        let grid_after = cfg.grid.map_or(f32::INFINITY, |g| g.after);
        let starts = |k: usize| -> f32 {
            if k < sensors_at {
                cfg.exposure_after
            } else if k < grids_at {
                match (k - sensors_at) % PER_SENSOR {
                    j if j < CCM_AT => cfg.vignetting_after,
                    j if j < CRF_AT => cfg.ccm_after,
                    _ => cfg.response_after,
                }
            } else {
                grid_after
            }
        };
        self.add_priors();
        self.t += 1;
        let (b1, b2) = (0.9f64, 0.999f64);
        let (bc1, bc2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        let grid_lr = cfg.grid.map_or(0.0, |g| g.lr) as f64;
        for k in 0..self.theta.len() {
            if progress < starts(k) {
                continue;
            }
            let g = self.grad[k];
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g;
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g;
            let lr = if k >= grids_at { grid_lr } else { cfg.lr as f64 };
            let step = lr * (self.m[k] / bc1) / ((self.v[k] / bc2).sqrt() + 1e-12);
            self.theta[k] -= step as f32;
        }
        self.grad.fill(0.0);
        self.fix_gauges();
    }

    /// The priors' gradients: identity on everything no gauge fixes,
    /// smoothness of the response's log slopes, total variation of the grids.
    fn add_priors(&mut self) {
        let (nv, ns) = (self.views(), self.sensors);
        let prior = self.cfg.prior as f64;
        for k in nv * PER_VIEW..self.theta.len() {
            self.grad[k] += 2.0 * prior * self.theta[k] as f64;
        }
        let smooth = self.cfg.response_smooth as f64;
        for s in 0..ns {
            for c in 0..3 {
                let o = self.sensor_at(s) + CRF_AT + c * RESPONSE_SEGMENTS;
                for j in 1..RESPONSE_SEGMENTS - 1 {
                    let d = (self.theta[o + j - 1] - 2.0 * self.theta[o + j] + self.theta[o + j + 1]) as f64;
                    self.grad[o + j - 1] += 2.0 * smooth * d;
                    self.grad[o + j] -= 4.0 * smooth * d;
                    self.grad[o + j + 1] += 2.0 * smooth * d;
                }
            }
        }
        if let Some(g) = self.cfg.grid {
            let [gx, gy, gz] = g.cells;
            let w = 2.0 * g.tv as f64 / (gx * gy * gz) as f64;
            for v in 0..nv {
                let o = self.grid_at(v);
                for k in 0..gz {
                    for j in 0..gy {
                        for i in 0..gx {
                            let a = o + ((k * gy + j) * gx + i) * CELL;
                            let next = [(i + 1 < gx, 1), (j + 1 < gy, gx), (k + 1 < gz, gx * gy)];
                            for (inside, stride) in next {
                                if !inside {
                                    continue;
                                }
                                let b = a + stride * CELL;
                                for q in 0..CELL {
                                    let d = (self.theta[a + q] - self.theta[b + q]) as f64;
                                    self.grad[a + q] += w * d;
                                    self.grad[b + q] -= w * d;
                                }
                            }
                        }
                    }
                }
            }
        }
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
        // colour matrices: centred across sensors
        let ccm_at: Vec<usize> = (0..self.sensors).map(|s| self.sensor_at(s) + CCM_AT).collect();
        for k in 0..OFFDIAG.len() {
            let m = ccm_at.iter().map(|&o| self.theta[o + k]).sum::<f32>() / self.sensors as f32;
            for &o in &ccm_at {
                self.theta[o + k] -= m;
            }
        }
        // grids: centred across views, cell by cell
        if self.grid_len > 0 {
            let base = self.grid_at(0);
            for q in 0..self.grid_len {
                let m = (0..nv).map(|v| self.theta[base + v * self.grid_len + q]).sum::<f32>() / nv as f32;
                for v in 0..nv {
                    self.theta[base + v * self.grid_len + q] -= m;
                }
            }
        }
    }

    /// The per-view settings of a view the fit never saw, fitted to what its
    /// photograph recorded where `weights` (`[W*H]`) is non-zero: exposure
    /// and white balance, starting from `start`, everything the sensor owns
    /// held. The standard appearance-fitted protocol for judging held-out
    /// views fits on one part of an image and scores on another.
    ///
    /// Levenberg-Marquardt on the three log gains, minimizing the weighted
    /// squared error; the result's white balance is zero-mean, its exposure
    /// carrying the common gain.
    pub fn fit_novel(&self, start: NovelShot, cam: &Camera, radiance: &[f32], target: &[f32], weights: &[f32]) -> NovelShot {
        let ln2 = std::f64::consts::LN_2;
        let shot = |k: [f64; 3]| -> NovelShot {
            let mean = (k[0] + k[1] + k[2]) / 3.0;
            NovelShot { exposure: (mean / ln2) as f32, white_balance: k.map(|v| (v - mean) as f32), ..start }
        };
        let residual = |k: [f64; 3]| -> Vec<f64> {
            let p = self.render(Shot::Novel(shot(k)), cam, radiance);
            p.iter().zip(target).enumerate().map(|(i, (a, b))| weights[i / 3].sqrt() as f64 * (a - b) as f64).collect()
        };
        let cost = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
        let mut k: [f64; 3] = std::array::from_fn(|c| start.exposure as f64 * ln2 + start.white_balance[c] as f64);
        let mut r = residual(k);
        let mut c0 = cost(&r);
        let mut lambda = 1e-3f64;
        let h = 1e-3f64;
        for _ in 0..40 {
            let cols: Vec<Vec<f64>> = (0..3)
                .map(|j| {
                    let (mut a, mut b) = (k, k);
                    a[j] += h;
                    b[j] -= h;
                    residual(a).iter().zip(residual(b)).map(|(p, q)| (p - q) / (2.0 * h)).collect()
                })
                .collect();
            let jtj: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum()));
            let jtr: [f64; 3] = std::array::from_fn(|i| cols[i].iter().zip(&r).map(|(a, b)| a * b).sum());
            let mut accepted = false;
            for _ in 0..10 {
                let a: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| jtj[i][j] + if i == j { lambda * jtj[i][i].max(1e-12) } else { 0.0 }));
                let Some(d) = solve3(a, jtr.map(|v| -v)) else { break };
                let next: [f64; 3] = std::array::from_fn(|i| k[i] + d[i]);
                let rn = residual(next);
                let cn = cost(&rn);
                if cn < c0 {
                    let small = d.iter().all(|v| v.abs() < 1e-6);
                    (k, r, c0) = (next, rn, cn);
                    lambda = (lambda / 3.0).max(1e-9);
                    accepted = !small;
                    break;
                }
                lambda *= 4.0;
            }
            if !accepted {
                break;
            }
        }
        shot(k)
    }

    /// One line per view and sensor, for a fit's log.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for v in 0..self.views() {
            let wb = self.white_balance(v);
            s += &format!("  view {v:3}: exposure {:+.3} EV, wb [{:+.3} {:+.3} {:+.3}]", self.exposure(v), wb[0], wb[1], wb[2]);
            if let Some(g) = self.grid(v) {
                s += &format!(", grid rms {:.4}", (g.iter().map(|x| x * x).sum::<f32>() / g.len() as f32).sqrt());
            }
            s += "\n";
        }
        for k in 0..self.sensors {
            let c = self.vignetting(k, 1.0);
            let m = self.ccm(k);
            // the response's local exponent d log f / d log y, mid-tones
            let exponent = |y: f32| {
                let (hi, lo) = (self.response(k, 2.0 * y), self.response(k, y));
                [0, 1, 2].map(|c| (hi[c] / lo[c]).log2())
            };
            s += &format!(
                "  sensor {k}: corner transmission [{:.3} {:.3} {:.3}], colour matrix [{:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3}], \
                 response exponent at 0.05 {:.2?}, at 0.25 {:.2?}\n",
                c[0], c[1], c[2], m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
                exponent(0.05), exponent(0.25)
            );
        }
        s
    }
}

/// Solve a 3x3 system by Cramer's rule; `None` when it is singular.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-300 || !d.is_finite() {
        return None;
    }
    Some(std::array::from_fn(|j| {
        let mut m = a;
        for i in 0..3 {
            m[i][j] = b[i];
        }
        det(&m) / d
    }))
}

/// The camera model's device state: a fit's per-pixel camera runs here, on
/// the renderer's output buffer, with nothing read back but the camera's own
/// few gradients.
pub struct DeviceIsp {
    grid: DeviceBuffer,
    dgrid: DeviceBuffer,
    partial: DeviceBuffer,
    dcam: DeviceBuffer,
    grid_len: usize,
    max_px: usize,
}

impl DeviceIsp {
    /// Scratch for frames of up to `max_px` pixels through `isp`'s shape.
    pub fn new(gpu: &Gpu, isp: &Isp, max_px: usize) -> DeviceIsp {
        let grid_words = isp.grid_len.max(CELL) as u64;
        DeviceIsp {
            grid: gpu.storage(grid_words),
            dgrid: gpu.storage(grid_words),
            partial: gpu.storage((CAM_GRADS * crate::renderer::dispatched_groups(max_px)) as u64),
            dcam: gpu.storage(CAM_GRADS as u64),
            grid_len: isp.grid_len,
            max_px,
        }
    }

    /// The kernels' uniform for `shot` through `cam`, and whether the grid is
    /// sliced; uploads the grid when it is.
    fn bind(&self, gpu: &Gpu, isp: &Isp, shot: Shot, cam: &Camera, mode: u32) -> Vec<u32> {
        assert_eq!(isp.grid_len, self.grid_len, "a DeviceIsp is sized for the camera model it was made for");
        let px = (cam.width * cam.height) as usize;
        assert!(px <= self.max_px, "a {px}-pixel frame past the camera model's {}-pixel scratch", self.max_px);
        let res = isp.resolve(shot);
        let fr = isp.frame(shot, cam);
        let cells = fr.grid.map_or([2, 2, 2], |(g, cells)| {
            gpu.write_f32(&self.grid, g);
            cells
        });
        let f = gpu_core::f;
        let mut w = Vec::with_capacity(PARAM_WORDS);
        w.extend_from_slice(&[cam.width, cam.height, mode, res.encode as u32]);
        w.extend_from_slice(&[f(fr.cx), f(fr.cy), f(fr.inv_norm), fr.grid.is_some() as u32]);
        w.extend_from_slice(&[cells[0] as u32, cells[1] as u32, cells[2] as u32, 0]);
        let vec4 = |w: &mut Vec<u32>, v: [f32; 3]| w.extend_from_slice(&[f(v[0]), f(v[1]), f(v[2]), 0]);
        vec4(&mut w, res.gain);
        for k in 0..3 {
            vec4(&mut w, res.vig[k]);
        }
        for r in 0..3 {
            vec4(&mut w, res.ccm[r]);
        }
        for j in 0..RESPONSE_SEGMENTS {
            vec4(&mut w, res.slope[j]);
        }
        debug_assert_eq!(w.len(), PARAM_WORDS);
        w
    }

    /// What `shot`'s camera records given the renderer's `radiance`
    /// (RGBA `[W*H*4]`), into `pred` (RGBA, alpha 0).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gpu: &Gpu, ks: Kernels, isp: &Isp, shot: Shot, cam: &Camera, radiance: &DeviceBuffer, pred: &DeviceBuffer) {
        let params = self.bind(gpu, isp, shot, cam, 0);
        let px = (cam.width * cam.height) as usize;
        let step = gpu.dispatch(ks.isp_pixel, &[radiance, &self.grid, pred, &self.partial], &params, gpu_core::Dispatch::Workgroups(px.div_ceil(64) as u32));
        gpu.submit(&[], &[step]);
    }

    /// Training view `v`'s backward: `dimg` holds dLoss/d(prediction) (RGBA)
    /// and is replaced by dLoss/d(radiance) in RGB, alpha untouched; the
    /// camera parameters' gradient is accumulated into `isp` for the next
    /// [`Isp::step`].
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, gpu: &Gpu, ks: Kernels, isp: &mut Isp, v: usize, cam: &Camera, radiance: &DeviceBuffer, dimg: &DeviceBuffer) {
        let shot = Shot::Training(v);
        let params = self.bind(gpu, isp, shot, cam, 1);
        let px = (cam.width * cam.height) as usize;
        let mut steps = Vec::new();
        let cells = isp.cfg.grid.map(|g| g.cells.iter().product::<usize>());
        if let Some(n) = cells {
            // the grid reads the upstream the per-pixel backward replaces
            steps.push(gpu.dispatch(ks.isp_grid_grad, &[radiance, dimg, &self.dgrid], &params, gpu_core::Dispatch::Workgroups(n as u32)));
        }
        steps.push(gpu.dispatch(ks.isp_pixel, &[radiance, &self.grid, dimg, &self.partial], &params, gpu_core::Dispatch::Workgroups(px.div_ceil(64) as u32)));
        let slices = crate::renderer::dispatched_groups(px) as u32;
        steps.push(gpu.dispatch(ks.dw_splitk_reduce, &[&self.partial, &self.dcam], &[CAM_GRADS as u32, slices, 0], gpu_core::Dispatch::Threads(CAM_GRADS as u32)));
        gpu.submit(&[], &steps);
        let d = gpu.read(&self.dcam, CAM_GRADS);
        let dcam: [f64; CAM_GRADS] = std::array::from_fn(|k| d[k] as f64);
        let dgrid: Vec<f64> = if cells.is_some() { gpu.read(&self.dgrid, self.grid_len).iter().map(|&v| v as f64).collect() } else { Vec::new() };
        let res = isp.resolve(shot);
        isp.accumulate(v, &res, &dcam, &dgrid);
    }
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

    /// A camera model with every stage away from identity, the grid included.
    fn perturbed(space: ColorSpace) -> Isp {
        let cfg = IspCfg { color_space: space, grid: Some(GridCfg { cells: [3, 2, 3], ..GridCfg::default() }), ..IspCfg::default() };
        let mut isp = Isp::new(cfg, &[(0, 0.3, Encoding::Srgb), (1, -0.2, Encoding::Srgb)]);
        for (k, t) in isp.theta.iter_mut().enumerate() {
            *t = 0.06 * ((k * 7 % 11) as f32 - 5.0) / 5.0;
        }
        isp
    }

    /// Whether an analytic derivative agrees with finite differences of `f`
    /// at 0. The model is piecewise smooth (response knots, grid cell faces,
    /// the vignetting floor): where a step straddles a kink the central
    /// difference is the mean of two slopes and the derivative is one of
    /// them, so a one-sided difference agreeing is agreement too.
    fn agrees(f: impl Fn(f32) -> f64, analytic: f64) -> Result<(), String> {
        let eps = 1e-3f32;
        let (p, z, m) = (f(eps), f(0.0), f(-eps));
        let central = (p - m) / (2.0 * eps as f64);
        let tol = 2e-2 * central.abs().max(1.0);
        let one_sided = [(p - z) / eps as f64, (z - m) / eps as f64];
        if (central - analytic).abs() < tol || one_sided.iter().any(|d| (d - analytic).abs() < 2.0 * tol) {
            Ok(())
        } else {
            Err(format!("central {central}, one-sided {one_sided:?} vs analytic {analytic}"))
        }
    }

    /// The backward is the derivative of the forward - finite differences -
    /// for every parameter (exposure, white balance, vignetting, colour
    /// matrix, response slopes, grid cells) and for the radiance, in both
    /// colour spaces.
    #[test]
    fn backward_is_the_derivative_of_forward() {
        for space in [ColorSpace::Display, ColorSpace::SceneLinear] {
            let mut isp = perturbed(space);
            let c = cam();
            // radiance spread over every response segment
            let rad: Vec<f32> = (0..7 * 5 * 3).map(|i| 0.004 + 0.9 * ((i * 13 % 17) as f32 / 17.0).powi(3)).collect();
            let up: Vec<f32> = (0..rad.len()).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
            let loss = |isp: &Isp, rad: &[f32]| -> f64 { isp.forward(1, &c, rad).iter().zip(&up).map(|(a, b)| (a * b) as f64).sum() };
            let drad = isp.backward(1, &c, &rad, &up);
            let pg = isp.grad.clone();
            for i in [0usize, 17, 40, 104] {
                let f = |d: f32| {
                    let mut r = rad.clone();
                    r[i] += d;
                    loss(&isp, &r)
                };
                agrees(f, drad[i] as f64).unwrap_or_else(|e| panic!("{space:?} radiance {i}: {e}"));
            }
            let mut touched = 0;
            // view 0 is not the view differentiated, and sensor 0 not its sensor
            for (k, &want) in pg.iter().enumerate() {
                let f = |d: f32| {
                    let mut a = isp.clone();
                    a.theta[k] += d;
                    loss(&a, &rad)
                };
                agrees(f, want).unwrap_or_else(|e| panic!("{space:?} parameter {k}: {e}"));
                touched += (want != 0.0) as usize;
            }
            // view 1: 4, sensor 1: 39, grid 1: 3*2*3*12 = 216, minus cells no
            // pixel's stencil reaches in luma
            assert!(touched > 4 + 39 + 100, "{space:?}: only {touched} parameters had a gradient");
        }
    }

    /// The identity camera records radiance unchanged (display) or exactly
    /// sRGB-encoded (scene-linear), grid and all.
    #[test]
    fn a_fresh_model_is_the_identity_camera() {
        let c = cam();
        let rad: Vec<f32> = (0..7 * 5 * 3).map(|i| i as f32 / 105.0).collect();
        let grid = Some(GridCfg::default());
        let d = Isp::new(IspCfg { grid, ..IspCfg::default() }, &[(0, 0.0, Encoding::Srgb)]).forward(0, &c, &rad);
        let l = Isp::new(IspCfg { color_space: ColorSpace::SceneLinear, grid, ..IspCfg::default() }, &[(0, 0.0, Encoding::Srgb)]).forward(0, &c, &rad);
        for i in 0..rad.len() {
            assert!((d[i] - rad[i]).abs() < 1e-6, "display {i}: {} vs {}", d[i], rad[i]);
            assert!((l[i] - srgb_encode(rad[i])).abs() < 1e-6, "linear {i}: {} vs {}", l[i], srgb_encode(rad[i]));
        }
    }

    /// The response curve is monotone whatever its parameters, and the
    /// colour matrix maps grey to grey.
    #[test]
    fn the_response_is_monotone_and_the_colour_matrix_keeps_grey() {
        let mut isp = perturbed(ColorSpace::Display);
        let o = isp.sensor_at(1) + CRF_AT;
        for (j, t) in isp.theta[o..o + 3 * RESPONSE_SEGMENTS].iter_mut().enumerate() {
            *t = if j % 2 == 0 { 2.0 } else { -3.0 };
        }
        let mut last = [f32::NEG_INFINITY; 3];
        for i in 0..400 {
            let y = -0.1 + 1.5 * i as f32 / 400.0;
            let f = isp.response(1, y);
            for c in 0..3 {
                assert!(f[c] > last[c], "channel {c} at {y}: {} after {}", f[c], last[c]);
            }
            last = f;
        }
        for s in 0..2 {
            for row in isp.ccm(s) {
                assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
        }
    }

    /// Fitting a novel view's exposure and white balance on the left half of
    /// its frame recovers the settings it was shot with, through a sensor
    /// with a colour matrix, a response curve and a lens.
    #[test]
    fn a_novel_views_appearance_is_fitted_from_part_of_its_frame() {
        let isp = perturbed(ColorSpace::SceneLinear);
        let c = Camera::look_at([0.0; 3], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 60.0, 24, 16);
        let rad: Vec<f32> = (0..24 * 16 * 3).map(|i| 0.02 + 0.5 * ((i * 29 % 37) as f32 / 37.0)).collect();
        let truth = NovelShot { sensor: 1, exposure: 0.7, white_balance: [0.08, -0.03, -0.05], encoding: Encoding::Srgb };
        let photo = isp.render(Shot::Novel(truth), &c, &rad);
        let left: Vec<f32> = (0..24 * 16).map(|i| if i % 24 < 12 { 1.0 } else { 0.0 }).collect();
        let got = isp.fit_novel(NovelShot::neutral(1, 0.0, Encoding::Srgb), &c, &rad, &photo, &left);
        assert!((got.exposure - truth.exposure).abs() < 1e-3, "exposure {} against {}", got.exposure, truth.exposure);
        for ch in 0..3 {
            assert!((got.white_balance[ch] - truth.white_balance[ch]).abs() < 1e-3, "white balance {:?} against {:?}", got.white_balance, truth.white_balance);
        }
    }
}
