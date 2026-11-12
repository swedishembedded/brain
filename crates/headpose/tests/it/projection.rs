// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The off-axis projection: the display's corners are the viewport's corners
//! from every eye, a point lands where the ray through it crosses the glass,
//! and parallax has the physical sign.

use headpose::{off_axis, Display, GeometryError};

const EYES: &[[f64; 3]] = &[
    [0.0, 0.0, 0.6],
    [0.18, -0.05, 0.45],
    [-0.25, 0.12, 0.9],
    [0.02, 0.3, 0.25],
    [0.4, -0.2, 1.5],
];

fn laptop() -> Display {
    Display::flat(0.344, 0.194).unwrap()
}

/// A display that is neither centred nor axis-aligned: yawed 25 degrees and
/// moved off the origin, as a monitor beside a laptop would be.
fn yawed() -> Display {
    let (s, c) = 25f64.to_radians().sin_cos();
    let r = [c, 0.0, -s];
    let u = [0.0, 1.0, 0.0];
    let o = [0.1, 0.05, -0.05];
    let at = |x: f64, y: f64| [o[0] + r[0] * x + u[0] * y, o[1] + r[1] * x + u[1] * y, o[2] + r[2] * x + u[2] * y];
    Display::from_corners(at(-0.26, -0.15), at(0.26, -0.15), at(-0.26, 0.15)).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn display_corners_are_the_viewport_corners_from_every_eye() {
    for d in [laptop(), yawed()] {
        for &e in EYES {
            if d.distance(e) <= 0.05 {
                continue;
            }
            let p = off_axis(&d, e, 0.01, 10.0).unwrap();
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)] {
                let n = p.ndc(d.point(u, v)).unwrap();
                assert!(close(n[0], 2.0 * u - 1.0, 1e-9) && close(n[1], 2.0 * v - 1.0, 1e-9), "eye {e:?} corner ({u},{v}) -> {n:?}");
            }
        }
    }
}

#[test]
fn a_point_lands_where_its_ray_from_the_eye_crosses_the_glass() {
    let d = laptop();
    let points = [[0.05, 0.02, -0.3], [-0.12, 0.07, -0.05], [0.0, -0.06, 0.15], [0.3, 0.2, -2.0]];
    for &e in EYES {
        let p = off_axis(&d, e, 0.01, 10.0).unwrap();
        for q in points {
            // the ray e -> q meets z = 0 at s = e.z / (e.z - q.z)
            let s = e[2] / (e[2] - q[2]);
            let hit = [e[0] + (q[0] - e[0]) * s, e[1] + (q[1] - e[1]) * s];
            let n = p.ndc(q).unwrap();
            assert!(close(n[0], hit[0] / (0.344 / 2.0), 1e-9) && close(n[1], hit[1] / (0.194 / 2.0), 1e-9), "eye {e:?} point {q:?}: {n:?} vs {hit:?}");
        }
    }
}

#[test]
fn moving_the_eye_shifts_far_points_with_it_and_near_points_against_it() {
    let d = laptop();
    let behind = [0.0, 0.0, -0.4];
    let in_front = [0.0, 0.0, 0.2];
    let left = off_axis(&d, [-0.1, 0.0, 0.6], 0.01, 10.0).unwrap();
    let right = off_axis(&d, [0.1, 0.0, 0.6], 0.01, 10.0).unwrap();
    assert!(right.ndc(behind).unwrap()[0] > left.ndc(behind).unwrap()[0], "a point behind the glass follows the eye");
    assert!(right.ndc(in_front).unwrap()[0] < left.ndc(in_front).unwrap()[0], "a point in front of the glass moves against the eye");
    // and a point ON the glass does not move at all
    let on = [0.03, -0.02, 0.0];
    assert!(close(left.ndc(on).unwrap()[0], right.ndc(on).unwrap()[0], 1e-12));
}

#[test]
fn a_centred_eye_is_an_ordinary_symmetric_perspective() {
    let d = laptop();
    let p = off_axis(&d, [0.0, 0.0, 0.5], 0.1, 10.0).unwrap();
    assert!(close(p.left, -p.right, 1e-12) && close(p.bottom, -p.top, 1e-12));
    // vertical field of view is the angle the glass subtends
    let fov = 2.0 * (p.top / p.near).atan();
    assert!(close(fov, 2.0 * (0.097f64 / 0.5).atan(), 1e-12));
}

#[test]
fn clip_depth_runs_zero_at_near_to_one_at_far_along_the_normal() {
    let d = laptop();
    let e = [0.05, 0.02, 0.6];
    let p = off_axis(&d, e, 0.1, 5.0).unwrap();
    assert!(close(p.ndc([e[0], e[1], e[2] - 0.1]).unwrap()[2], 0.0, 1e-9));
    assert!(close(p.ndc([e[0], e[1], e[2] - 5.0]).unwrap()[2], 1.0, 1e-9));
}

#[test]
fn the_matrices_agree_with_ndc() {
    let d = yawed();
    let p = off_axis(&d, [0.1, 0.1, 0.7], 0.05, 8.0).unwrap();
    let m = p.view_projection();
    let q = [0.07, -0.03, -0.25];
    // column-major: clip = sum_j m[j*4 + i] * q[j]
    let h = [q[0], q[1], q[2], 1.0];
    let c: Vec<f64> = (0..4).map(|i| (0..4).map(|j| m[j * 4 + i] as f64 * h[j]).sum()).collect();
    let n = p.ndc(q).unwrap();
    for k in 0..3 {
        assert!(close(c[k] / c[3], n[k], 1e-5), "component {k}: {} vs {}", c[k] / c[3], n[k]);
    }
}

#[test]
fn an_eye_at_or_behind_the_glass_is_refused() {
    let d = laptop();
    assert!(matches!(off_axis(&d, [0.0, 0.0, 0.0], 0.01, 10.0), Err(GeometryError::EyeBehindDisplay { .. })));
    assert!(matches!(off_axis(&d, [0.0, 0.0, -0.3], 0.01, 10.0), Err(GeometryError::EyeBehindDisplay { .. })));
    assert!(matches!(off_axis(&d, [0.0, 0.0, 0.5], 0.5, 0.1), Err(GeometryError::BadClipPlanes { .. })));
}

#[test]
fn a_window_is_the_part_of_the_glass_its_pixels_cover() {
    let d = laptop();
    assert_eq!(d.window((1920, 1080), 0, 0, 1920, 1080).unwrap(), d);
    // the top-left quarter
    let w = d.window((1920, 1080), 0, 0, 960, 540).unwrap();
    let c = w.centre();
    assert!(close(c[0], -0.344 / 4.0, 1e-12) && close(c[1], 0.194 / 4.0, 1e-12));
    assert!(Display::from_corners([0.0; 3], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]).is_err(), "skewed corners are not a rectangle");
}
