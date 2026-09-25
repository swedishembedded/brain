// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side scene types: Gaussians (post-activation SoA), the camera, and
//! render options. Conventions match gsplat/WorldMirror: +X right, +Y down,
//! +Z forward; `c2w` is camera-to-world (SE(3), row-major 4×4); the
//! rasterizer consumes the world-to-camera rows from [`Camera::viewmat`].

pub use camera::Lens;

/// A camera: pose, the linear intrinsics `fx fy cx cy` in pixels (continuous
/// coordinates, pixel `i`'s centre at `i + 0.5`), the lens, and the image
/// size. The EWA renderer understands the pinhole part only; the ray renderer
/// ([`RenderOpts::ray`]) images through [`Camera::lens`] and
/// [`Camera::shutter`] exactly.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Camera-to-world, row-major 4×4 (last row 0 0 0 1). Must be rigid. For
    /// a rolling-shutter camera, the pose at the MIDDLE of the readout.
    pub c2w: [f32; 16],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    /// Distortion beyond the pinhole map.
    pub lens: Lens,
    /// Rolling shutter: the camera's motion over the readout, as a twist in
    /// its own frame `(angular [3], linear [3])` per frame height. The row at
    /// continuous `v` is exposed at `tau = v / height - 1/2`, from the pose
    /// `c2w · [Exp(tau·angular) | tau·linear]`. Zero is a global shutter.
    pub shutter: [f32; 6],
}

impl Camera {
    /// A global-shutter pinhole camera.
    pub fn pinhole(c2w: [f32; 16], fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Camera {
        Camera { c2w, fx, fy, cx, cy, width, height, lens: Lens::Pinhole, shutter: [0.0; 6] }
    }

    /// A global-shutter camera with calibration `k`.
    pub fn with_intrinsics(c2w: [f32; 16], k: &camera::Intrinsics) -> Camera {
        Camera {
            c2w,
            fx: k.fx as f32,
            fy: k.fy as f32,
            cx: k.cx as f32,
            cy: k.cy as f32,
            width: k.width,
            height: k.height,
            lens: k.lens,
            shutter: [0.0; 6],
        }
    }

    /// This camera's calibration.
    pub fn intrinsics(&self) -> camera::Intrinsics {
        camera::Intrinsics {
            fx: self.fx as f64,
            fy: self.fy as f64,
            cx: self.cx as f64,
            cy: self.cy as f64,
            lens: self.lens,
            width: self.width,
            height: self.height,
        }
    }

    /// Whether the EWA renderer's linear pinhole model is this camera
    /// exactly: no lens distortion and no rolling shutter.
    pub fn is_pinhole(&self) -> bool {
        self.lens == Lens::Pinhole && self.shutter.iter().all(|v| *v == 0.0)
    }

    /// The same camera imaging `width x height`: every continuous pixel
    /// coordinate scales by the size ratio, so this is exact for any lens.
    pub fn resized(&self, width: u32, height: u32) -> Camera {
        let (sx, sy) = (width as f32 / self.width as f32, height as f32 / self.height as f32);
        Camera { fx: self.fx * sx, fy: self.fy * sy, cx: self.cx * sx, cy: self.cy * sy, width, height, ..*self }
    }

    /// World-to-camera `[R|t]` rows (rigid inverse of `c2w`):
    /// `[r00 r01 r02 tx, r10 r11 r12 ty, r20 r21 r22 tz]`.
    pub fn viewmat(&self) -> [f32; 12] {
        let m = &self.c2w;
        // R_w2c = R_c2w^T, t_w2c = -R^T * t
        let r = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
        let t = [m[3], m[7], m[11]];
        [
            r[0], r[1], r[2], -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
            r[3], r[4], r[5], -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
            r[6], r[7], r[8], -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]),
        ]
    }

    /// Camera position in world space.
    pub fn eye(&self) -> [f32; 3] {
        [self.c2w[3], self.c2w[7], self.c2w[11]]
    }

    /// Build a camera at `eye` looking at `target` with world-`up` hint, y-down
    /// image convention: right = forward×up, down = forward×right.
    pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3], fov_y_deg: f32, width: u32, height: u32) -> Camera {
        let f = norm3(sub3(target, eye));
        let r = norm3(cross3(f, up));
        let d = norm3(cross3(f, r));
        let c2w = [
            r[0], d[0], f[0], eye[0],
            r[1], d[1], f[1], eye[1],
            r[2], d[2], f[2], eye[2],
            0.0, 0.0, 0.0, 1.0,
        ];
        let fy = 0.5 * height as f32 / (0.5 * fov_y_deg.to_radians()).tan();
        Camera::pinhole(c2w, fy, fy, width as f32 / 2.0, height as f32 / 2.0, width, height)
    }
}

/// Parse a `cameras.json` document: a JSON array of
/// `{c2w: [16], fx, fy, cx, cy, width, height}`, plus a lens (`model` and its
/// coefficients, see `camera::Intrinsics::from_json`) and a rolling-shutter
/// twist (`shutter: [6]`) where the camera has them. The one reader - `fit`
/// over the wire, the CLI and the tools all take cameras in this shape.
pub fn cameras_from_json(raw: &str) -> Result<Vec<Camera>, String> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("cameras must be a JSON array: {e}"))?;
    let arr = v.as_array().ok_or("cameras must be a JSON array")?;
    arr.iter()
        .enumerate()
        .map(|(i, c)| {
            let c2w: Vec<f32> = c["c2w"]
                .as_array()
                .ok_or_else(|| format!("camera {i} has no 'c2w'"))?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect();
            let c2w: [f32; 16] = c2w.try_into().map_err(|v: Vec<f32>| format!("camera {i}: 'c2w' has {} entries, expected 16", v.len()))?;
            let k = camera::Intrinsics::from_json(c).map_err(|e| format!("camera {i}: {e}"))?;
            let mut cam = Camera::with_intrinsics(c2w, &k);
            if let Some(sh) = c.get("shutter") {
                let v: Vec<f32> = sh.as_array().ok_or(format!("camera {i}: 'shutter' must be an array"))?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                cam.shutter = v.try_into().map_err(|v: Vec<f32>| format!("camera {i}: 'shutter' has {} entries, expected 6", v.len()))?;
            }
            Ok(cam)
        })
        .collect()
}

/// One camera as a `cameras.json` entry.
pub fn camera_to_json(c: &Camera) -> serde_json::Value {
    let mut v = c.intrinsics().to_json();
    v["c2w"] = c.c2w.iter().map(|v| *v as f64).collect::<Vec<f64>>().into();
    if c.shutter.iter().any(|v| *v != 0.0) {
        v["shutter"] = c.shutter.iter().map(|v| *v as f64).collect::<Vec<f64>>().into();
    }
    v
}

/// The `cameras.json` document [`cameras_from_json`] reads.
pub fn cameras_to_json(cams: &[Camera]) -> String {
    let arr: Vec<serde_json::Value> = cams.iter().map(camera_to_json).collect();
    serde_json::to_string_pretty(&arr).expect("plain numbers serialize")
}

/// Frame a scene of known axis-aligned bounds: eye backed off along -Z from
/// the bounds center, looking back at it. Shared by the CLI (`splat_cli::render`/
/// `::view`) and `caps::render` - hoisted here so neither holds its own copy.
pub fn auto_camera_from_bounds(bounds: ([f32; 3], [f32; 3]), width: u32, height: u32, fov: f32) -> Camera {
    let (lo, hi) = bounds;
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let r = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt() / 2.0;
    let eye = [c[0], c[1], c[2] - 2.2 * r.max(1e-3)];
    Camera::look_at(eye, c, [0.0, -1.0, 0.0], fov, width, height)
}

/// [`auto_camera_from_bounds`] over `s`'s own bounds - the convenience form
/// most callers want.
pub fn auto_camera(s: &Splats, width: u32, height: u32, fov: f32) -> Camera {
    auto_camera_from_bounds(s.bounds(), width, height, fov)
}

pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Host Gaussian scene, SoA, activations ALREADY applied: `scales` linear,
/// `opacities` in [0,1], `quats` wxyz (normalized on use), `colors` linear RGB
/// (SH degree-0 decoded: `0.282095*dc + 0.5`).
#[derive(Clone, Default)]
pub struct Splats {
    pub means: Vec<f32>,      // N*3
    pub quats: Vec<f32>,      // N*4 wxyz
    pub scales: Vec<f32>,     // N*3
    pub opacities: Vec<f32>,  // N
    pub colors: Vec<f32>,     // N*3
    /// Higher-order SH (degree, coeffs) parsed from PLY, channel-planar per
    /// gaussian (Inria layout). Kept for round-trips; not rendered yet.
    pub sh_rest: Option<(u32, Vec<f32>)>,
}

impl Splats {
    pub fn len(&self) -> usize {
        self.opacities.len()
    }
    pub fn is_empty(&self) -> bool {
        self.opacities.is_empty()
    }

    /// Axis-aligned bounds of the means: (min, max).
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for m in self.means.chunks_exact(3) {
            for k in 0..3 {
                lo[k] = lo[k].min(m[k]);
                hi[k] = hi[k].max(m[k]);
            }
        }
        (lo, hi)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Color,
    /// Alpha-weighted expected depth replicated to RGB.
    Depth,
}

#[derive(Clone, Copy)]
pub struct RenderOpts {
    pub bg: [f32; 3],
    pub mode: Mode,
    /// Put back the energy the low-pass spread out, by scaling opacity by
    /// `sqrt(|S| / |S + eps I|)` - Mip-Splatting's 2D Mip filter, which
    /// approximates a box filter over the pixel rather than simply inflating
    /// the splat. Without it, dilation makes a splat blurrier AND brighter,
    /// and the default kernel size costs most of a scene's fine detail.
    ///
    /// Whether it WINS depends on how big the splats are, and a pixel-aligned
    /// reconstruction is not the regime the technique was designed for.
    /// Measured against a box-downsampled reference:
    ///
    /// ```text
    ///   splat std   inria 0.3   mip 0.1
    ///     0.30 px    20.7 dB     12.1 dB
    ///     0.50 px    18.0 dB     20.1 dB
    /// ```
    ///
    /// The crossover sits near half a pixel because that is where the
    /// compensation stops being a correction and starts being most of the
    /// opacity. A feed-forward pass emits one gaussian per source pixel and
    /// lands at a median projected size of 0.46 px with 60% below half a
    /// pixel, so it sits on the losing side: on a real capture, fitted and
    /// scored against held-out photographs, the Mip filter gives 21.98 dB
    /// against the dilation's 22.70 dB.
    ///
    /// So the default is the dilation, which is also what Inria-trained PLYs
    /// and the viewers that render them expect. Turn this on for scenes whose
    /// splats are comfortably larger than a pixel, and use
    /// [`crate::mip::recalibrate_opacity`] when switching a scene over.
    pub antialiased: bool,
    pub eps2d: f32,
    pub near: f32,
    pub far: f32,
    /// Evaluate every gaussian exactly along each pixel's own ray through
    /// the camera's lens (`splat_ray_*.wgsl`) instead of splatting its EWA
    /// linearization. Required for any camera that is not a plain pinhole
    /// (see [`Camera::is_pinhole`]), which renders this way whatever this
    /// says; for a pinhole it is the exact image EWA approximates. The
    /// filters then act in 3D: `eps2d` is a pixel's footprint variance at the
    /// gaussian's range, and `antialiased` compensates it on the ray
    /// marginal - Mip-Splatting's 2D filter, evaluated where the ray is.
    pub ray: bool,
}

impl Default for RenderOpts {
    fn default() -> Self {
        RenderOpts {
            bg: [0.0; 3],
            mode: Mode::Color,
            antialiased: false,
            eps2d: 0.3,
            near: 0.01,
            far: 1e10,
            ray: false,
        }
    }
}
