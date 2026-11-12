// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The off-axis ("generalized perspective") projection through a physical
//! display from a tracked eye.
//!
//! The method is the one in Kooima's *Generalized Perspective Projection*
//! (2008): express the display's corners in a basis aligned with the glass,
//! find the frustum extents at the near plane by similar triangles from the
//! eye, and view along the display normal from the eye. The scene is never
//! rotated to fake parallax; only the eye moves, so geometry placed behind
//! (or in front of) the glass stays where it is in the room.
//!
//! The output targets wgpu/WebGPU/Vulkan/D3D clip space: right-handed view
//! space looking down -Z, clip depth in `[0, 1]`, matrices column-major
//! `[f32; 16]` (the layout WGSL's `mat4x4<f32>` reads). The arithmetic is f64;
//! only the final matrices are narrowed.

use sfm::linalg::{dot, sub};

use crate::display::Display;
use crate::{GeometryError, V3};

/// A frustum from one eye through one display.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffAxis {
    /// Near-plane extents in the display-aligned eye frame.
    pub left: f64,
    pub right: f64,
    pub bottom: f64,
    pub top: f64,
    pub near: f64,
    pub far: f64,
    /// The eye, in the screen frame.
    pub eye: V3,
    /// Display-aligned basis: right, up, normal (out of the glass).
    pub basis: [V3; 3],
}

/// The frustum from `eye` through `display`, clipped at `near`/`far` metres
/// from the eye along the display normal.
pub fn off_axis(display: &Display, eye: V3, near: f64, far: f64) -> Result<OffAxis, GeometryError> {
    if !(near > 0.0 && far > near && far.is_finite()) {
        return Err(GeometryError::BadClipPlanes { near, far });
    }
    if !eye.iter().all(|c| c.is_finite()) {
        return Err(GeometryError::OutOfRange(format!("eye position {eye:?} is not finite")));
    }
    let (vr, vu, vn) = (display.right(), display.up(), display.normal());
    let va = sub(display.lower_left, eye);
    let vb = sub(display.lower_right, eye);
    let vc = sub(display.upper_left, eye);
    // distance from the eye to the display plane, positive in front of it
    let d = -dot(va, vn);
    if d <= 1e-6 {
        return Err(GeometryError::EyeBehindDisplay { distance_m: d });
    }
    let k = near / d;
    Ok(OffAxis {
        left: dot(vr, va) * k,
        right: dot(vr, vb) * k,
        bottom: dot(vu, va) * k,
        top: dot(vu, vc) * k,
        near,
        far,
        eye,
        basis: [vr, vu, vn],
    })
}

impl OffAxis {
    /// The view matrix (screen frame to the display-aligned eye frame), as
    /// rows of a 4x4 in f64: rotate into `(right, up, normal)` then move the
    /// eye to the origin.
    fn view_rows(&self) -> [[f64; 4]; 4] {
        let [vr, vu, vn] = self.basis;
        let e = self.eye;
        [
            [vr[0], vr[1], vr[2], -dot(vr, e)],
            [vu[0], vu[1], vu[2], -dot(vu, e)],
            [vn[0], vn[1], vn[2], -dot(vn, e)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// The asymmetric perspective, as rows, clip depth `[0, 1]`.
    fn projection_rows(&self) -> [[f64; 4]; 4] {
        let (l, r, b, t, n, f) = (self.left, self.right, self.bottom, self.top, self.near, self.far);
        [
            [2.0 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2.0 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, f / (n - f), n * f / (n - f)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    /// Column-major view matrix.
    pub fn view(&self) -> [f32; 16] {
        column_major(&self.view_rows())
    }

    /// Column-major projection matrix.
    pub fn projection(&self) -> [f32; 16] {
        column_major(&self.projection_rows())
    }

    /// Column-major `projection * view`, computed in f64 before narrowing.
    pub fn view_projection(&self) -> [f32; 16] {
        column_major(&mul(&self.projection_rows(), &self.view_rows()))
    }

    /// Where a screen-frame point lands in normalized device coordinates
    /// (`x, y` in `[-1, 1]` across the display, `z` in `[0, 1]` between the
    /// clip planes), or `None` when it is behind the eye.
    pub fn ndc(&self, p: V3) -> Option<V3> {
        let m = mul(&self.projection_rows(), &self.view_rows());
        let h = [p[0], p[1], p[2], 1.0];
        let c: [f64; 4] = std::array::from_fn(|i| (0..4).map(|j| m[i][j] * h[j]).sum());
        if c[3] <= 1e-12 {
            return None;
        }
        Some([c[0] / c[3], c[1] / c[3], c[2] / c[3]])
    }
}

fn mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

fn column_major(rows: &[[f64; 4]; 4]) -> [f32; 16] {
    std::array::from_fn(|i| rows[i % 4][i / 4] as f32)
}
