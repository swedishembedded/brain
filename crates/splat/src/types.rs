// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side scene types: Gaussians (post-activation SoA), the OpenCV pinhole
//! camera, and render options. Conventions match gsplat/WorldMirror: +X right,
//! +Y down, +Z forward; `c2w` is camera-to-world (SE(3), row-major 4×4); the
//! rasterizer consumes the world-to-camera rows from [`Camera::viewmat`].

/// Pinhole camera, OpenCV convention.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Camera-to-world, row-major 4×4 (last row 0 0 0 1). Must be rigid.
    pub c2w: [f32; 16],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl Camera {
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
        Camera { c2w, fx: fy, fy, cx: width as f32 / 2.0, cy: height as f32 / 2.0, width, height }
    }
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
    /// Multiply the anti-alias blur compensation into opacity (gsplat
    /// `antialiased` mode). Inria-trained PLYs expect `false`.
    pub antialiased: bool,
    pub eps2d: f32,
    pub near: f32,
    pub far: f32,
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
        }
    }
}
