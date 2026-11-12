// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The physical glass a view is rendered through, as three measured corners.
//!
//! Three corners rather than a width and a height, because the general case
//! (a monitor that is not parallel to the webcam, a window that is only part
//! of the screen) is the same arithmetic as the simple one, and a projection
//! written against corners never has to special-case either.

use sfm::linalg::{add, cross, dot, norm, scale, sub};

use crate::{GeometryError, V3};

/// A rectangle in the screen frame: `lower_left`, `lower_right`, `upper_left`
/// as the viewer sees them. The fourth corner is implied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    pub lower_left: V3,
    pub lower_right: V3,
    pub upper_left: V3,
}

/// How far from perpendicular the two edges may be before the corners are
/// refused as not a rectangle (cosine of the angle between them).
const MAX_EDGE_COSINE: f64 = 1e-3;

impl Display {
    /// Validate three measured corners.
    pub fn from_corners(lower_left: V3, lower_right: V3, upper_left: V3) -> Result<Display, GeometryError> {
        let d = Display { lower_left, lower_right, upper_left };
        let (w, h) = (d.width(), d.height());
        if !(w.is_finite() && h.is_finite()) || w < 1e-4 || h < 1e-4 {
            return Err(GeometryError::DegenerateDisplay(format!("edges of {w} m and {h} m")));
        }
        let c = dot(sub(lower_right, lower_left), sub(upper_left, lower_left)) / (w * h);
        if c.abs() > MAX_EDGE_COSINE {
            return Err(GeometryError::DegenerateDisplay(format!("edges are {:.2} degrees from perpendicular", 90.0 - c.acos().to_degrees())));
        }
        Ok(d)
    }

    /// A flat display of `width_m` x `height_m` centred on the screen-frame
    /// origin, in the XY plane, facing +Z.
    pub fn flat(width_m: f64, height_m: f64) -> Result<Display, GeometryError> {
        let (x, y) = (width_m / 2.0, height_m / 2.0);
        Display::from_corners([-x, -y, 0.0], [x, -y, 0.0], [-x, y, 0.0])
    }

    pub fn width(&self) -> f64 {
        norm(sub(self.lower_right, self.lower_left))
    }

    pub fn height(&self) -> f64 {
        norm(sub(self.upper_left, self.lower_left))
    }

    /// Unit vector along the bottom edge, left to right.
    pub fn right(&self) -> V3 {
        scale(sub(self.lower_right, self.lower_left), 1.0 / self.width())
    }

    /// Unit vector along the left edge, bottom to top.
    pub fn up(&self) -> V3 {
        scale(sub(self.upper_left, self.lower_left), 1.0 / self.height())
    }

    /// Unit normal pointing out of the glass toward the viewer.
    pub fn normal(&self) -> V3 {
        let n = cross(self.right(), self.up());
        scale(n, 1.0 / norm(n))
    }

    /// The point at fractions `(u, v)` of the way across (left to right) and
    /// up (bottom to top) the glass.
    pub fn point(&self, u: f64, v: f64) -> V3 {
        add(self.lower_left, add(scale(sub(self.lower_right, self.lower_left), u), scale(sub(self.upper_left, self.lower_left), v)))
    }

    pub fn centre(&self) -> V3 {
        self.point(0.5, 0.5)
    }

    /// Signed distance of `p` from the display plane, positive on the
    /// viewer's side.
    pub fn distance(&self, p: V3) -> f64 {
        dot(sub(p, self.lower_left), self.normal())
    }

    /// The part of this display a window covers, from pixel geometry: the
    /// display is `display_px` pixels across, the window's client area
    /// starts at `(x, y)` (top-left origin, y down, the window system's
    /// convention) and is `w` x `h` pixels. A window that hangs off the
    /// screen is still a valid rectangle; it is not clamped, because the
    /// glass it describes is exactly as large as the window.
    pub fn window(&self, display_px: (u32, u32), x: i32, y: i32, w: u32, h: u32) -> Result<Display, GeometryError> {
        let (dw, dh) = (display_px.0 as f64, display_px.1 as f64);
        if dw < 1.0 || dh < 1.0 || w == 0 || h == 0 {
            return Err(GeometryError::OutOfRange(format!("window {w}x{h} on a {dw}x{dh} display")));
        }
        let u0 = x as f64 / dw;
        let u1 = (x as f64 + w as f64) / dw;
        let v_top = 1.0 - y as f64 / dh;
        let v_bottom = 1.0 - (y as f64 + h as f64) / dh;
        Display::from_corners(self.point(u0, v_bottom), self.point(u1, v_bottom), self.point(u0, v_top))
    }
}
