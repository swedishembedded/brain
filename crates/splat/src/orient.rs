// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Re-framing a reconstructed scene so it opens the right way up.
//!
//! A feed-forward reconstruction's world frame is its FIRST camera's frame -
//! `c2w[0]` comes back as the identity - so "up" in the file is whichever way
//! the camera happened to be held when the capture started, and every viewer
//! shows the scene tipped by that angle. On a real capture it was 63 degrees.
//!
//! The cameras themselves say which way is up, with no assumption about the
//! subject: they orbit it, so the normal of the plane they lie in is the
//! scene's vertical, and their view rays intersect at its centre.
//!
//! Every transform here is RIGID. Rotating a gaussian rotates its mean and
//! composes with its orientation; its scales and opacity are properties of the
//! gaussian, not of the frame it is expressed in. So re-framing cannot change
//! what a render looks like from a correspondingly moved camera - which is the
//! property the tests hold it to.
//!
//! Swedish Embedded AB implements 3D reconstruction pipelines whose output
//! lands in a frame the next tool can use. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use crate::types::Splats;

/// Rotation (row-major 3x3) and centre that map the scene into a frame whose
/// vertical is the camera orbit's axis, pointing along `up_sign * Y`.
pub fn frame_from_cameras(c2w: &[[f64; 16]], up_sign: f64) -> ([f64; 9], [f64; 3]) {
    let n = c2w.len() as f64;
    let eyes: Vec<[f64; 3]> = c2w.iter().map(|m| [m[3], m[7], m[11]]).collect();
    let fwd: Vec<[f64; 3]> = c2w.iter().map(|m| [m[2], m[6], m[10]]).collect();

    // centre: the point closest to every view ray
    let (mut a, mut b) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
    for (e, f) in eyes.iter().zip(&fwd) {
        let l = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt().max(1e-12);
        let d = [f[0] / l, f[1] / l, f[2] / l];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += if i == j { 1.0 } else { 0.0 } - d[i] * d[j];
            }
            b[i] += (0..3).map(|j| (if i == j { 1.0 } else { 0.0 } - d[i] * d[j]) * e[j]).sum::<f64>();
        }
    }
    let centre = solve3(&a, &b).unwrap_or_else(|| {
        let mut m = [0.0; 3];
        for e in &eyes {
            for k in 0..3 {
                m[k] += e[k] / n;
            }
        }
        m
    });

    // up: the normal of the plane the cameras lie in, via the smallest
    // principal axis of their offsets.
    //
    // Fit that plane about the cameras' own CENTROID, not about the subject.
    // The subject is generally off the plane the cameras travel in - by 1.5x
    // the sweep radius on a real handheld capture - and every camera then
    // carries that same offset along the normal, inflating the one eigenvalue
    // that is supposed to be smallest until the fit returns an axis with no
    // relation to the trajectory.
    let mut mid = [0.0f64; 3];
    for e in &eyes {
        for k in 0..3 {
            mid[k] += e[k] / n;
        }
    }
    let mut cov = [[0.0f64; 3]; 3];
    for e in &eyes {
        let v = [e[0] - mid[0], e[1] - mid[1], e[2] - mid[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += v[i] * v[j];
            }
        }
    }
    let mut up = smallest_axis(&cov);
    // An axis has no sign, and for a closed orbit the cameras' mean offset from
    // the centre is zero, so their positions cannot say which end is up. Their
    // ORIENTATION can: column 1 of a c2w is the camera's own down axis, and no
    // one films a scene upside down, so the end of the axis that disagrees with
    // where the cameras think down is, is up.
    let down: f64 = c2w.iter().map(|m| up[0] * m[1] + up[1] * m[5] + up[2] * m[9]).sum();
    if down > 0.0 {
        up = [-up[0], -up[1], -up[2]];
    }
    (rotation_taking(up, [0.0, up_sign, 0.0]), centre)
}

/// Apply a rigid re-framing to every gaussian.
pub fn apply(s: &Splats, r: &[f64; 9], centre: &[f64; 3]) -> Splats {
    let rq = quat_of(r);
    let mut out = Splats::default();
    for i in 0..s.len() {
        let m = [
            s.means[i * 3] as f64 - centre[0],
            s.means[i * 3 + 1] as f64 - centre[1],
            s.means[i * 3 + 2] as f64 - centre[2],
        ];
        for k in 0..3 {
            out.means.push((r[k * 3] * m[0] + r[k * 3 + 1] * m[1] + r[k * 3 + 2] * m[2]) as f32);
        }
        let q = &s.quats[i * 4..i * 4 + 4];
        // stored wxyz, as `splat::ply` writes them
        let c = qmul(&rq, &[q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]);
        let nrm = (c.iter().map(|v| v * v).sum::<f64>()).sqrt().max(1e-12);
        for v in c {
            out.quats.push((v / nrm) as f32);
        }
        out.scales.extend_from_slice(&s.scales[i * 3..i * 3 + 3]);
        out.opacities.push(s.opacities[i]);
        out.colors.extend_from_slice(&s.colors[i * 3..i * 3 + 3]);
    }
    out
}

/// The same re-framing applied to a camera-to-world matrix.
pub fn transform_c2w(m: &[f64; 16], r: &[f64; 9], centre: &[f64; 3]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    out[15] = 1.0;
    for i in 0..3 {
        for j in 0..3 {
            out[i * 4 + j] = (0..3).map(|k| r[i * 3 + k] * m[k * 4 + j]).sum();
        }
        let t = [m[3] - centre[0], m[7] - centre[1], m[11] - centre[2]];
        out[i * 4 + 3] = r[i * 3] * t[0] + r[i * 3 + 1] * t[1] + r[i * 3 + 2] * t[2];
    }
    out
}

fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let d = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut x = [0.0; 3];
    for k in 0..3 {
        let mut m = *a;
        for i in 0..3 {
            m[i][k] = b[i];
        }
        let dk = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        x[k] = dk / d;
    }
    Some(x)
}

/// Eigenvector of the smallest eigenvalue, by inverse power iteration on a
/// shifted matrix - three cameras is enough data and this needs no LAPACK.
fn smallest_axis(cov: &[[f64; 3]; 3]) -> [f64; 3] {
    // Inverse iteration converges to the eigenvalue NEAREST the shift, and a
    // covariance's eigenvalues are all non-negative, so the shift belongs just
    // below zero - close enough to single out the smallest, far enough that a
    // perfectly planar orbit (whose smallest eigenvalue is exactly zero) still
    // leaves an invertible matrix.
    let tr = (cov[0][0] + cov[1][1] + cov[2][2]).max(1e-12);
    let mut m = *cov;
    for i in 0..3 {
        m[i][i] += tr * 1e-6;
    }
    let mut v = [0.577, 0.577, 0.577];
    for _ in 0..64 {
        let w = solve3(&m, &v).unwrap_or(v);
        let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().max(1e-12);
        v = [w[0] / n, w[1] / n, w[2] / n];
    }
    v
}

/// Shortest rotation taking `from` to `to`, both unit-ish.
fn rotation_taking(from: [f64; 3], to: [f64; 3]) -> [f64; 9] {
    let n = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let (a, b) = (n(from), n(to));
    let v = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if s < 1e-9 {
        // parallel or antiparallel: identity, or a half turn about any
        // perpendicular axis
        return if c > 0.0 {
            [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        } else {
            let p = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let ax = n([a[1] * p[2] - a[2] * p[1], a[2] * p[0] - a[0] * p[2], a[0] * p[1] - a[1] * p[0]]);
            [
                2.0 * ax[0] * ax[0] - 1.0, 2.0 * ax[0] * ax[1], 2.0 * ax[0] * ax[2],
                2.0 * ax[1] * ax[0], 2.0 * ax[1] * ax[1] - 1.0, 2.0 * ax[1] * ax[2],
                2.0 * ax[2] * ax[0], 2.0 * ax[2] * ax[1], 2.0 * ax[2] * ax[2] - 1.0,
            ]
        };
    }
    let k = (1.0 - c) / (s * s);
    [
        1.0 + k * (-v[2] * v[2] - v[1] * v[1]), -v[2] + k * v[0] * v[1], v[1] + k * v[0] * v[2],
        v[2] + k * v[0] * v[1], 1.0 + k * (-v[2] * v[2] - v[0] * v[0]), -v[0] + k * v[1] * v[2],
        -v[1] + k * v[0] * v[2], v[0] + k * v[1] * v[2], 1.0 + k * (-v[1] * v[1] - v[0] * v[0]),
    ]
}

/// wxyz quaternion of a row-major rotation.
fn quat_of(r: &[f64; 9]) -> [f64; 4] {
    let t = r[0] + r[4] + r[8];
    if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [0.25 * s, (r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s]
    } else if r[0] > r[4] && r[0] > r[8] {
        let s = (1.0 + r[0] - r[4] - r[8]).sqrt() * 2.0;
        [(r[7] - r[5]) / s, 0.25 * s, (r[1] + r[3]) / s, (r[2] + r[6]) / s]
    } else if r[4] > r[8] {
        let s = (1.0 + r[4] - r[0] - r[8]).sqrt() * 2.0;
        [(r[2] - r[6]) / s, (r[1] + r[3]) / s, 0.25 * s, (r[5] + r[7]) / s]
    } else {
        let s = (1.0 + r[8] - r[0] - r[4]).sqrt() * 2.0;
        [(r[3] - r[1]) / s, (r[2] + r[6]) / s, (r[5] + r[7]) / s, 0.25 * s]
    }
}

fn qmul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
