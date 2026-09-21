// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putting two reconstructions of the same place into one world.
//!
//! A feed-forward pass holds a bounded number of frames, so a long capture is
//! reconstructed in chunks - and each chunk comes back in its own world,
//! anchored to whichever frame led it and sized by whatever that chunk's
//! content normalised to. Two chunks of one walk are therefore related by a
//! SIMILARITY: rotation, translation, and a scale.
//!
//! The frames two chunks share supply the correspondence, since each has a
//! pose in both worlds. Those frames are consecutive frames of one smooth
//! sweep, though, so their camera centres are close to collinear and a fit
//! that uses only centres is free to spin the scene about the line through
//! them. The cameras' HEADINGS are what remove that freedom: each shared frame
//! contributes three axis correspondences, which pin the rotation whatever the
//! centres do. Scale and translation then follow from the centres.
//!
//! Swedish Embedded AB implements 3D reconstruction that scales past one
//! model's context window. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::orient::{qmul, quat_of};
use crate::types::Splats;

/// Maps a point of the second world into the first: `p_a = s * R * p_b + t`.
#[derive(Clone, Copy, Debug)]
pub struct Sim3 {
    pub s: f64,
    /// Row-major 3x3.
    pub r: [f64; 9],
    pub t: [f64; 3],
}

impl Default for Sim3 {
    fn default() -> Self {
        Sim3 { s: 1.0, r: [1., 0., 0., 0., 1., 0., 0., 0., 1.], t: [0.0; 3] }
    }
}

/// Solve for the similarity taking world `b` onto world `a`, given the SAME
/// frames posed in each. Needs at least two shared frames whose centres differ.
pub fn sim3_from_cameras(a: &[[f64; 16]], b: &[[f64; 16]]) -> Option<Sim3> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }
    // Rotation from the cameras' axes. Each shared frame gives three unit
    // correspondences, so this stays well posed no matter how the centres sit.
    let mut s3 = [[0.0f64; 3]; 3];
    for (x, y) in a.iter().zip(b) {
        for k in 0..3 {
            let (u, v) = ([x[k], x[4 + k], x[8 + k]], [y[k], y[4 + k], y[8 + k]]);
            for i in 0..3 {
                for j in 0..3 {
                    s3[i][j] += v[i] * u[j];
                }
            }
        }
    }
    let r = nearest_rotation(&s3);

    // Scale from how much further apart the same two cameras are in each
    // world. The median shrugs off a single badly posed shared frame.
    let (ea, eb): (Vec<[f64; 3]>, Vec<[f64; 3]>) =
        (a.iter().map(|m| [m[3], m[7], m[11]]).collect(), b.iter().map(|m| [m[3], m[7], m[11]]).collect());
    let mut ratios: Vec<f64> = Vec::new();
    for i in 0..ea.len() {
        for j in i + 1..ea.len() {
            let da = dist(&ea[i], &ea[j]);
            let db = dist(&eb[i], &eb[j]);
            if db > 1e-9 && da > 1e-9 {
                ratios.push(da / db);
            }
        }
    }
    if ratios.is_empty() {
        return None;
    }
    ratios.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let s = ratios[ratios.len() / 2];

    let n = a.len() as f64;
    let ca = centroid(&ea, n);
    let cb = centroid(&eb, n);
    let rb = mul3(&r, &cb);
    Some(Sim3 { s, r, t: [ca[0] - s * rb[0], ca[1] - s * rb[1], ca[2] - s * rb[2]] })
}

impl Sim3 {
    /// `self` applied AFTER `inner`. Chaining chunks means composing these,
    /// and composition is why a long capture drifts: every link's error is
    /// carried by every chunk that follows it.
    pub fn after(&self, inner: &Sim3) -> Sim3 {
        let mut r = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = (0..3).map(|k| self.r[i * 3 + k] * inner.r[k * 3 + j]).sum();
            }
        }
        let ri = mul3(&self.r, &inner.t);
        Sim3 {
            s: self.s * inner.s,
            r,
            t: [self.s * ri[0] + self.t[0], self.s * ri[1] + self.t[1], self.s * ri[2] + self.t[2]],
        }
    }
}

/// How far the shared cameras land from where they should, worst case. This is
/// the honest read on whether two chunks really do overlap.
pub fn camera_residual(a: &[[f64; 16]], b: &[[f64; 16]], m: &Sim3) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let t = transform_c2w_sim3(y, m);
            dist(&[x[3], x[7], x[11]], &[t[3], t[7], t[11]])
        })
        .fold(0.0, f64::max)
}

/// Re-express a scene in the other world. Sizes scale with the world; opacity
/// and colour are properties of the gaussian and do not.
pub fn apply_sim3(s: &Splats, m: &Sim3) -> Splats {
    let rq = quat_of(&m.r);
    let sf = m.s as f32;
    let mut out = Splats::default();
    for i in 0..s.len() {
        let p = mul3(&m.r, &[s.means[i * 3] as f64, s.means[i * 3 + 1] as f64, s.means[i * 3 + 2] as f64]);
        for k in 0..3 {
            out.means.push((m.s * p[k] + m.t[k]) as f32);
        }
        let q = &s.quats[i * 4..i * 4 + 4];
        let c = qmul(&rq, &[q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]);
        let nrm = (c.iter().map(|v| v * v).sum::<f64>()).sqrt().max(1e-12);
        for v in c {
            out.quats.push((v / nrm) as f32);
        }
        for k in 0..3 {
            out.scales.push(s.scales[i * 3 + k] * sf);
        }
        out.opacities.push(s.opacities[i]);
        out.colors.extend_from_slice(&s.colors[i * 3..i * 3 + 3]);
    }
    out
}

/// The same re-expression applied to a camera-to-world matrix. A camera's axes
/// stay unit vectors, so only its position takes the scale.
pub fn transform_c2w_sim3(m: &[f64; 16], w: &Sim3) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    out[15] = 1.0;
    for i in 0..3 {
        for j in 0..3 {
            out[i * 4 + j] = (0..3).map(|k| w.r[i * 3 + k] * m[k * 4 + j]).sum();
        }
    }
    let e = mul3(&w.r, &[m[3], m[7], m[11]]);
    for i in 0..3 {
        out[i * 4 + 3] = w.s * e[i] + w.t[i];
    }
    out
}

/// Concatenate, keeping every gaussian. Overlapping surface is left to voxel
/// merging, which already knows how to fuse duplicates.
pub fn concat(parts: &[Splats]) -> Splats {
    let mut out = Splats::default();
    for p in parts {
        out.means.extend_from_slice(&p.means);
        out.quats.extend_from_slice(&p.quats);
        out.scales.extend_from_slice(&p.scales);
        out.opacities.extend_from_slice(&p.opacities);
        out.colors.extend_from_slice(&p.colors);
    }
    out
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn centroid(v: &[[f64; 3]], n: f64) -> [f64; 3] {
    let mut c = [0.0; 3];
    for p in v {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    c
}

fn mul3(r: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [
        r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
        r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
        r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
    ]
}

/// Given `S = sum over correspondences of b * a^T`, the rotation with
/// `a ~= R b`, via Horn's quaternion form: the largest eigenvector of a
/// symmetric 4x4 built from S. No SVD needed.
fn nearest_rotation(s: &[[f64; 3]; 3]) -> [f64; 9] {
    let (xx, xy, xz) = (s[0][0], s[0][1], s[0][2]);
    let (yx, yy, yz) = (s[1][0], s[1][1], s[1][2]);
    let (zx, zy, zz) = (s[2][0], s[2][1], s[2][2]);
    let n = [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ];
    let q = largest_eigenvector4(&n);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Cyclic Jacobi on a symmetric 4x4. Small, exact enough, and dependency-free.
fn largest_eigenvector4(m: &[[f64; 4]; 4]) -> [f64; 4] {
    let mut a = *m;
    let mut v = [[0.0f64; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in p + 1..4 {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-18 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut best = 0;
    for i in 1..4 {
        if a[i][i] > a[best][best] {
            best = i;
        }
    }
    let q = [v[0][best], v[1][best], v[2][best], v[3][best]];
    let l = (q.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-18);
    let sgn = if q[0] < 0.0 { -1.0 } else { 1.0 };
    [sgn * q[0] / l, sgn * q[1] / l, sgn * q[2] / l, sgn * q[3] / l]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composing_two_hops_equals_doing_them_one_after_the_other() {
        let a = Sim3 { s: 1.7, r: [0., -1., 0., 1., 0., 0., 0., 0., 1.], t: [0.5, -2.0, 3.0] };
        let b = Sim3 { s: 0.4, r: [1., 0., 0., 0., 0., -1., 0., 1., 0.], t: [-1.0, 0.25, 0.5] };
        let p = [0.3, -1.4, 2.2];
        let step = |m: &Sim3, v: [f64; 3]| {
            let r = mul3(&m.r, &v);
            [m.s * r[0] + m.t[0], m.s * r[1] + m.t[1], m.s * r[2] + m.t[2]]
        };
        let want = step(&a, step(&b, p));
        let got = step(&a.after(&b), p);
        let err = (0..3).map(|k| (want[k] - got[k]).abs()).fold(0.0, f64::max);
        assert!(err < 1e-12, "composed hop lands {err:.2e} from the two separate hops");
    }

    #[test]
    fn nearest_rotation_recovers_a_known_rotation() {
        let (c, sn) = (0.8f64, 0.6f64);
        let ax = [0.5773502691896258, -0.5773502691896258, 0.5773502691896258];
        let mut r = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = if i == j { c } else { 0.0 } + (1.0 - c) * ax[i] * ax[j];
            }
        }
        let k = [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] += sn * k[i][j];
            }
        }
        // b = e_k and a = R e_k, so S = sum b a^T = R^T
        let s3 = [[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]];
        let got = nearest_rotation(&s3);
        let err = (0..9).map(|i| (got[i] - r[i]).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "recovered\n{got:?}\nwanted\n{r:?}\nerr {err:.2e}");
    }
}
