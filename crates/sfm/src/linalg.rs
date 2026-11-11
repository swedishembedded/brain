// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The small dense f64 linear algebra structure-from-motion needs: 3-vectors
//! and 3x3 matrices, the symmetric eigendecomposition (cyclic Jacobi) every
//! null-space and SVD question here reduces to, a triangular factorization solve for the
//! bundle adjustment's reduced camera system, and the SO(3) exponential and
//! logarithm.
//!
//! Everything is `f64`: these systems are tiny (at most a few hundred
//! unknowns after the Schur complement) and badly scaled - pixel residuals
//! against unit rotations - so the cost of double precision is nothing and
//! the conditioning it buys is real.

use backend_cpu::par;

pub type V3 = [f64; 3];
/// Row-major 3x3.
pub type M3 = [f64; 9];

pub const I3: M3 = [1., 0., 0., 0., 1., 0., 0., 0., 1.];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / norm(a).max(1e-300))
}

pub fn mv(m: &M3, v: V3) -> V3 {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}
/// `mᵀ v`.
pub fn mtv(m: &M3, v: V3) -> V3 {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}
pub fn mm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| {
        let (r, c) = (i / 3, i % 3);
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
    })
}
pub fn transpose(a: &M3) -> M3 {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}
pub fn det(a: &M3) -> f64 {
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
}
pub fn skew(v: V3) -> M3 {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}

/// Eigendecomposition of a symmetric `n x n` matrix (row-major) by cyclic
/// Jacobi rotations. Returns eigenvalues ASCENDING and the matching
/// eigenvectors as rows.
pub fn eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        let scale: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum::<f64>().max(1e-300);
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]));
    let vals = order.iter().map(|&i| a[i * n + i]).collect();
    let vecs = order.iter().map(|&i| (0..n).map(|k| v[k * n + i]).collect()).collect();
    (vals, vecs)
}

/// The unit vector minimising `|A x|` for `A` with `cols` columns given as
/// rows: the eigenvector of `AᵀA` with the smallest eigenvalue.
pub fn null_vector(rows: &[Vec<f64>], cols: usize) -> Vec<f64> {
    let mut ata = vec![0.0f64; cols * cols];
    for r in rows {
        for i in 0..cols {
            for j in i..cols {
                ata[i * cols + j] += r[i] * r[j];
            }
        }
    }
    for i in 0..cols {
        for j in 0..i {
            ata[i * cols + j] = ata[j * cols + i];
        }
    }
    eigh(&ata, cols).1.swap_remove(0)
}

/// Singular value decomposition of a 3x3: `a = U diag(s) Vᵀ`, singular values
/// DESCENDING, `U` and `V` orthogonal (not necessarily proper).
pub fn svd3(a: &M3) -> (M3, V3, M3) {
    let ata = mm(&transpose(a), a);
    let (vals, vecs) = eigh(&ata, 3);
    // descending
    let idx = [2usize, 1, 0];
    let mut v = [0.0f64; 9];
    let mut s = [0.0f64; 3];
    for (c, &k) in idx.iter().enumerate() {
        s[c] = vals[k].max(0.0).sqrt();
        for r in 0..3 {
            v[r * 3 + c] = vecs[k][r];
        }
    }
    let mut u = [0.0f64; 9];
    let col = |m: &M3, c: usize| -> V3 { [m[c], m[3 + c], m[6 + c]] };
    let mut uc: [V3; 3] = [[0.0; 3]; 3];
    for c in 0..3 {
        let av = mv(a, col(&v, c));
        uc[c] = if s[c] > 1e-12 * s[0].max(1e-300) { scale(av, 1.0 / s[c]) } else { [0.0; 3] };
    }
    // complete a basis where singular values vanished
    if norm(uc[2]) < 0.5 {
        uc[2] = if norm(uc[1]) > 0.5 { normalize(cross(uc[0], uc[1])) } else { [0.0; 3] };
    }
    if norm(uc[1]) < 0.5 {
        let t = if uc[0][0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        uc[1] = normalize(cross(uc[0], t));
        uc[2] = normalize(cross(uc[0], uc[1]));
    }
    for c in 0..3 {
        for r in 0..3 {
            u[r * 3 + c] = uc[c][r];
        }
    }
    (u, s, v)
}

/// The rotation nearest `a` in matrix L2 norm.
pub fn nearest_rotation(a: &M3) -> M3 {
    let (u, _, v) = svd3(a);
    let mut r = mm(&u, &transpose(&v));
    if det(&r) < 0.0 {
        let mut u2 = u;
        for row in 0..3 {
            u2[row * 3 + 2] = -u2[row * 3 + 2];
        }
        r = mm(&u2, &transpose(&v));
    }
    r
}

/// Rodrigues: the rotation by angle `|w|` about `w`.
pub fn exp_so3(w: V3) -> M3 {
    let th = norm(w);
    let k = skew(w);
    let kk = mm(&k, &k);
    let (a, b) = if th < 1e-8 { (1.0 - th * th / 6.0, 0.5 - th * th / 24.0) } else { (th.sin() / th, (1.0 - th.cos()) / (th * th)) };
    std::array::from_fn(|i| I3[i] + a * k[i] + b * kk[i])
}

/// Inverse of [`exp_so3`].
pub fn log_so3(r: &M3) -> V3 {
    // The angle from both of its witnesses, cos from the trace and sin from
    // the skew part: acos of the trace alone is flat near a half turn and
    // loses half its digits there.
    let c = (r[0] + r[4] + r[8] - 1.0) / 2.0;
    let w = [r[7] - r[5], r[2] - r[6], r[3] - r[1]];
    let th = (norm(w) / 2.0).atan2(c);
    if th < 1e-7 {
        return scale(w, 0.5);
    }
    if std::f64::consts::PI - th < 1e-6 {
        // near pi: the axis from the symmetric part, (R + R^T)/2 = c I +
        // (1 - c) a a^T, taking the largest diagonal entry as the pivot
        let d = [r[0], r[4], r[8]];
        let i = (0..3).max_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap();
        let mut ax = [0.0; 3];
        ax[i] = ((d[i] - c) / (1.0 - c)).max(0.0).sqrt();
        for j in 0..3 {
            if j != i {
                ax[j] = (r[i * 3 + j] + r[j * 3 + i]) / (2.0 * (1.0 - c) * ax[i]);
            }
        }
        // The diagonal fixes the axis only up to sign; the skew part, still
        // nonzero short of an exact half turn, says which.
        if dot(ax, w) < 0.0 {
            ax = scale(ax, -1.0);
        }
        return scale(normalize(ax), th);
    }
    scale(w, th / (2.0 * th.sin()))
}

/// Smallest system the blocked parallel factorization is used for: below
/// it the pool's fork and join cost more than the arithmetic they share out.
const PARALLEL_CHOLESKY_MIN: usize = 192;
/// Columns per panel of the blocked factorization: wide enough that the
/// trailing update is a dot product long enough to vectorize, narrow enough
/// that the serial diagonal factor stays a small share of the work.
const CHOLESKY_PANEL: usize = 64;

/// Solve `A x = b` for symmetric positive definite `A` (`n x n`, row-major;
/// only the lower triangle is read). `None` if `A` is not numerically
/// positive definite.
///
/// A system of [`PARALLEL_CHOLESKY_MIN`] unknowns or more is factored by
/// panels on the CPU pool: the reduced camera system of a bundle adjustment
/// is dense and a thousand unknowns wide, and its factor was the one serial
/// cubic step in each iteration.
pub fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = if n >= PARALLEL_CHOLESKY_MIN { cholesky_factor_blocked(a, n)? } else { cholesky_factor(a, n)? };
    Some(cholesky_substitute(&l, b, n))
}

/// The lower factor `L` of `A = L Lᵀ`, one row at a time.
fn cholesky_factor(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// [`cholesky_factor`] by panels of [`CHOLESKY_PANEL`] columns: factor the
/// diagonal block, solve the rows below it against that block, then subtract
/// the panel's outer product from every row of the trailing matrix. The last
/// two steps are independent per row, so they run on the pool.
fn cholesky_factor_blocked(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let nb = CHOLESKY_PANEL;
    // the lower triangle of A, updated in place into L
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        l[i * n..i * n + i + 1].copy_from_slice(&a[i * n..i * n + i + 1]);
    }
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + nb).min(n);
        let w = k1 - k0;
        // the diagonal block, which every earlier panel has already updated
        for i in k0..k1 {
            for j in k0..=i {
                let mut s = l[i * n + j];
                for k in k0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if s <= 0.0 || !s.is_finite() {
                        return None;
                    }
                    l[i * n + i] = s.sqrt();
                } else {
                    l[i * n + j] = s / l[j * n + j];
                }
            }
        }
        if k1 == n {
            break;
        }
        let (head, tail) = l.split_at_mut(k1 * n);
        let diag: &[f64] = &head[k0 * n..];
        // the rows below solved against the diagonal block
        par::chunks_mut(tail, n, |_, row| {
            for j in 0..w {
                let mut s = row[k0 + j];
                for k in 0..j {
                    s -= row[k0 + k] * diag[j * n + k0 + k];
                }
                row[k0 + j] = s / diag[j * n + k0 + j];
            }
        });
        // the panel, copied out so the trailing rows can be written while
        // every other row's panel is read
        let panel: Vec<f64> = tail.chunks(n).flat_map(|row| row[k0..k1].iter().copied()).collect();
        par::chunks_mut(tail, n, |r, row| {
            let mine = &panel[r * w..(r + 1) * w];
            for (j, x) in row[k1..k1 + r + 1].iter_mut().enumerate() {
                let theirs = &panel[j * w..(j + 1) * w];
                *x -= mine.iter().zip(theirs).map(|(p, q)| p * q).sum::<f64>();
            }
        });
        k0 = k1;
    }
    Some(l)
}

/// Solve `L Lᵀ x = b`.
fn cholesky_substitute(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Inverse of a symmetric positive definite 3x3, `None` if singular.
pub fn inv3_spd(a: &M3) -> Option<M3> {
    let d = det(a);
    if d.abs() < 1e-300 || !d.is_finite() {
        return None;
    }
    let c = [
        a[4] * a[8] - a[5] * a[7],
        a[2] * a[7] - a[1] * a[8],
        a[1] * a[5] - a[2] * a[4],
        a[5] * a[6] - a[3] * a[8],
        a[0] * a[8] - a[2] * a[6],
        a[2] * a[3] - a[0] * a[5],
        a[3] * a[7] - a[4] * a[6],
        a[1] * a[6] - a[0] * a[7],
        a[0] * a[4] - a[1] * a[3],
    ];
    Some(c.map(|v| v / d))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// acos is flat near -1, so an angle taken from the trace loses digits
    /// close to a half turn; the rotation vector must survive the round trip
    /// there too.
    #[test]
    fn log_so3_is_accurate_near_a_half_turn() {
        let axis = normalize([0.48, -0.6, 0.64]);
        for eps in [1e-7, 1.5e-6, 3e-6, 1e-5, 1e-3] {
            let w = scale(axis, std::f64::consts::PI - eps);
            let l = log_so3(&exp_so3(w));
            for k in 0..3 {
                assert!((l[k] - w[k]).abs() < 1e-8, "pi - {eps}: component {k} is {} not {}", l[k], w[k]);
            }
        }
    }

    #[test]
    fn svd_reconstructs_and_rotations_round_trip() {
        let a: M3 = [2.0, -1.0, 0.3, 0.5, 1.5, -0.7, 0.2, 0.4, 3.0];
        let (u, s, v) = svd3(&a);
        let us: M3 = std::array::from_fn(|i| u[i] * s[i % 3]);
        let back = mm(&us, &transpose(&v));
        for i in 0..9 {
            assert!((back[i] - a[i]).abs() < 1e-9, "{i}: {} vs {}", back[i], a[i]);
        }
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        let w = [0.3, -1.1, 0.7];
        let l = log_so3(&exp_so3(w));
        for k in 0..3 {
            assert!((l[k] - w[k]).abs() < 1e-10);
        }
        let r = nearest_rotation(&a);
        assert!((det(&r) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cholesky_solves_spd_systems() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let b = [1.0, 2.0, 3.0];
        let x = cholesky_solve(&a, &b, 3).unwrap();
        for i in 0..3 {
            let r: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!((r - b[i]).abs() < 1e-12);
        }
    }

    /// The panelled parallel factorization solves what the row-at-a-time one
    /// does, for a size that is not a multiple of the panel, and refuses a
    /// matrix that is not positive definite.
    #[test]
    fn the_blocked_cholesky_solves_like_the_unblocked_one() {
        let n = PARALLEL_CHOLESKY_MIN + CHOLESKY_PANEL + 7;
        let mut state = 12345u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        let m: Vec<f64> = (0..n * n).map(|_| rnd()).collect();
        // A = M Mᵀ + n I is symmetric positive definite
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|k| m[i * n + k] * m[j * n + k]).sum::<f64>() + if i == j { n as f64 } else { 0.0 };
            }
        }
        let b: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let blocked = cholesky_solve(&a, &b, n).expect("positive definite");
        let reference = cholesky_substitute(&cholesky_factor(&a, n).unwrap(), &b, n);
        let worst = blocked.iter().zip(&reference).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-9, "blocked and unblocked solutions differ by {worst}");
        for i in 0..n {
            let r: f64 = (0..n).map(|j| a[i * n + j] * blocked[j]).sum::<f64>() - b[i];
            assert!(r.abs() < 1e-8, "row {i} residual {r}");
        }
        a[(n - 1) * n + n - 1] = -1.0;
        assert!(cholesky_solve(&a, &b, n).is_none(), "an indefinite matrix has no factor");
    }
}
