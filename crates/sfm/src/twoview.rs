// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibrated two-view geometry on BEARING vectors - unit rays in each
//! camera's frame, from `camera::Intrinsics::unproject` - so a fisheye ray
//! more than 90 degrees off the optical axis is as usable as a central one.
//!
//! * The minimal five-point essential-matrix solver (Nistér, "An Efficient
//!   Solution to the Five-Point Relative Pose Problem", TPAMI 2004): the
//!   four-dimensional null space of the five epipolar constraints
//!   `f₂ᵀ E f₁ = 0`, the ten cubic constraints `det E = 0` and
//!   `2 E Eᵀ E − tr(E Eᵀ) E = 0` (Nistér eq. 3-4) in the three unknown
//!   null-space weights, Gauss-Jordan elimination of their 10x20 coefficient
//!   matrix in Nistér's monomial order, and the hidden-variable 3x3
//!   polynomial matrix whose determinant is the tenth-degree polynomial in
//!   `z` (Nistér §3.2, the `⟨k⟩ ⟨l⟩ ⟨m⟩` rows) - its real roots are the
//!   solutions.
//! * LO-RANSAC (Chum, Matas & Kittler, "Locally Optimized RANSAC", DAGM
//!   2003) with MSAC truncated-quadratic scoring (Torr & Zisserman,
//!   "MLESAC", CVIU 2000): every new best model is locally optimized by
//!   iterated least squares over a shrinking threshold and by an inner
//!   RANSAC of non-minimal eight-point fits on its inliers; the iteration
//!   bound adapts to the best inlier ratio. The eight-point algorithm
//!   (Hartley & Zisserman, "Multiple View Geometry", §11.2) is the
//!   least-squares fit on many inliers.
//! * A final Levenberg-Marquardt refinement of `E = [t]ₓ R` on its inliers,
//!   on the five-dimensional essential manifold (rotation, unit
//!   translation).
//! * The four-fold decomposition of `E` and the cheirality test that picks
//!   one (H&Z §9.6.2), and linear triangulation (H&Z §12.2) from bearings.
//!
//! Errors are ANGLES: [`sampson`] is the first-order squared angular
//! distance of a correspondence to the epipolar geometry, so a pixel
//! threshold becomes a threshold here by dividing by the focal length.

use crate::camera::Pose;
use crate::linalg::{cholesky_solve, cross, dot, exp_so3, mm, mtv, mv, norm, normalize, null_vector, scale, sub, svd3, transpose, M3, V3};
use data::rng::Lcg;

/// First-order squared angular distance of the correspondence `a ↔ b`
/// (unit bearings) to the epipolar geometry of `e`: the Sampson
/// approximation (H&Z §11.4.3) with the gradient of `bᵀ E a` taken in the
/// tangent planes of the two unit spheres instead of two image planes. For
/// rays near the optical axis it is the classical normalized-coordinate
/// Sampson error.
pub fn sampson(e: &M3, a: V3, b: V3) -> f64 {
    let ea = mv(e, a);
    let etb = mtv(e, b);
    let num = dot(b, ea);
    let g1 = sub(etb, scale(a, dot(a, etb)));
    let g2 = sub(ea, scale(b, num));
    num * num / (dot(g1, g1) + dot(g2, g2)).max(1e-300)
}

/// The row of the linear epipolar system `bᵀ E a = 0` in the row-major
/// entries of `E`.
fn epipolar_row(a: V3, b: V3) -> [f64; 9] {
    std::array::from_fn(|i| b[i / 3] * a[i % 3])
}

/// The essential matrix nearest `e`: two equal singular values, one zero.
fn to_essential(e: &M3) -> Option<M3> {
    let (u, s, v) = svd3(e);
    if s[0] <= 0.0 {
        return None;
    }
    let d = [1.0, 1.0, 0.0];
    let ud: M3 = std::array::from_fn(|i| u[i] * d[i % 3]);
    Some(mm(&ud, &transpose(&v)))
}

/// The essential matrix of at least eight correspondences, least squares on
/// the linear constraints and projected onto the essential manifold.
pub fn essential_8pt(a: &[V3], b: &[V3]) -> Option<M3> {
    if a.len() < 8 {
        return None;
    }
    let rows: Vec<Vec<f64>> = a.iter().zip(b).map(|(p, q)| epipolar_row(*p, *q).to_vec()).collect();
    let e = null_vector(&rows, 9);
    to_essential(&std::array::from_fn(|i| e[i]))
}

/// Nistér's order of the twenty monomials of degree at most three in
/// `(x, y, z)`, as exponents: the ten eliminated first, then the ten the
/// hidden-variable step reads - `x`, `y` and `1` times powers of `z`.
const MONO: [[u8; 3]; 20] = [
    [3, 0, 0], // x³
    [0, 3, 0], // y³
    [2, 1, 0], // x²y
    [1, 2, 0], // xy²
    [2, 0, 1], // x²z
    [2, 0, 0], // x²
    [0, 2, 1], // y²z
    [0, 2, 0], // y²
    [1, 1, 1], // xyz
    [1, 1, 0], // xy
    [1, 0, 2], // xz²
    [1, 0, 1], // xz
    [1, 0, 0], // x
    [0, 1, 2], // yz²
    [0, 1, 1], // yz
    [0, 1, 0], // y
    [0, 0, 3], // z³
    [0, 0, 2], // z²
    [0, 0, 1], // z
    [0, 0, 0], // 1
];

/// A polynomial of degree at most three in `(x, y, z)`, over [`MONO`].
type Poly = [f64; 20];

/// `PRODUCT[i][j]`: the monomial `MONO[i]·MONO[j]`, or `u8::MAX` past degree
/// three.
fn product_table() -> &'static [[u8; 20]; 20] {
    static T: std::sync::OnceLock<[[u8; 20]; 20]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let e = [MONO[i][0] + MONO[j][0], MONO[i][1] + MONO[j][1], MONO[i][2] + MONO[j][2]];
                MONO.iter().position(|m| *m == e).map_or(u8::MAX, |p| p as u8)
            })
        })
    })
}

fn pmul(a: &Poly, b: &Poly) -> Poly {
    let t = product_table();
    let mut out = [0.0; 20];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj != 0.0 {
                let k = t[i][j];
                debug_assert!(k != u8::MAX, "product past degree three");
                out[k as usize] += ai * bj;
            }
        }
    }
    out
}

fn padd(a: &Poly, b: &Poly, s: f64) -> Poly {
    std::array::from_fn(|i| a[i] + s * b[i])
}

/// Univariate polynomial helpers, coefficients in ascending order.
mod upoly {
    pub fn mul(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] += x * y;
            }
        }
        out
    }

    fn combine(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
        (0..a.len().max(b.len())).map(|i| a.get(i).copied().unwrap_or(0.0) + s * b.get(i).copied().unwrap_or(0.0)).collect()
    }

    pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        combine(a, b, 1.0)
    }

    pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
        combine(a, b, -1.0)
    }

    pub fn eval(p: &[f64], x: f64) -> f64 {
        p.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    /// Every real root of `p`. Between two consecutive real roots of `p'` a
    /// polynomial is monotonic and holds at most one root, so the roots of
    /// the derivative (found the same way, recursively) bracket the roots of
    /// `p` inside the Cauchy bound, and bisection pins each down to the last
    /// bit - slower than a companion-matrix eigensolve and unable to lose a
    /// real root to a complex pair's rounding.
    pub fn real_roots(p: &[f64]) -> Vec<f64> {
        let big = p.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        if big == 0.0 || !big.is_finite() {
            return Vec::new();
        }
        let mut n = p.len();
        while n > 1 && p[n - 1].abs() <= 1e-14 * big {
            n -= 1;
        }
        let p = &p[..n];
        match n {
            0 | 1 => return Vec::new(),
            2 => return vec![-p[0] / p[1]],
            _ => {}
        }
        let lead = p[n - 1];
        let bound = 1.0 + p[..n - 1].iter().fold(0.0f64, |m, c| m.max((c / lead).abs()));
        let dp: Vec<f64> = (1..n).map(|i| i as f64 * p[i]).collect();
        let mut knots = vec![-bound];
        knots.extend(real_roots(&dp).into_iter().filter(|r| r.abs() < bound));
        knots.push(bound);
        knots.sort_by(f64::total_cmp);
        let mut roots: Vec<f64> = Vec::new();
        for w in knots.windows(2) {
            let (mut lo, mut hi) = (w[0], w[1]);
            let (flo, fhi) = (eval(p, lo), eval(p, hi));
            if flo == 0.0 || fhi == 0.0 {
                let r = if flo == 0.0 { lo } else { hi };
                if roots.last().is_none_or(|&l| (l - r).abs() > 1e-12 * r.abs().max(1.0)) {
                    roots.push(r);
                }
                continue;
            }
            if flo.signum() == fhi.signum() {
                continue;
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if mid <= lo || mid >= hi {
                    break;
                }
                let fm = eval(p, mid);
                if fm == 0.0 {
                    (lo, hi) = (mid, mid);
                    break;
                }
                if fm.signum() == flo.signum() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        roots
    }
}

/// An orthonormal basis of the null space of the five epipolar rows: the
/// rows orthonormalized (Gram-Schmidt, twice for stability), then completed
/// with the standard basis vectors that stay furthest from their span.
fn null_basis(rows: &[[f64; 9]; 5]) -> Option<[[f64; 9]; 4]> {
    fn orth(v: &mut [f64; 9], q: &[[f64; 9]]) {
        for _ in 0..2 {
            for b in q {
                let d: f64 = (0..9).map(|i| v[i] * b[i]).sum();
                for i in 0..9 {
                    v[i] -= d * b[i];
                }
            }
        }
    }
    let n2 = |v: &[f64; 9]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut q: Vec<[f64; 9]> = Vec::with_capacity(9);
    for r in rows {
        let mut v = *r;
        let before = n2(&v);
        orth(&mut v, &q);
        let n = n2(&v);
        if n <= 1e-10 * before.max(1e-300) {
            return None; // dependent constraints: a degenerate sample
        }
        q.push(v.map(|x| x / n));
    }
    let mut out = [[0.0; 9]; 4];
    for slot in out.iter_mut() {
        let best = (0..9)
            .map(|i| {
                let mut v = [0.0; 9];
                v[i] = 1.0;
                orth(&mut v, &q);
                v
            })
            .max_by(|a, b| n2(a).total_cmp(&n2(b)))
            .unwrap();
        let n = n2(&best);
        let v = best.map(|x| x / n);
        q.push(v);
        *slot = v;
    }
    Some(out)
}

/// Every essential matrix consistent with five correspondences of unit
/// bearings `a ↔ b` (`bᵀ E a = 0`): at most ten, each of unit Frobenius
/// norm. Empty for a degenerate sample.
pub fn essential_5pt(a: &[V3], b: &[V3]) -> Vec<M3> {
    assert!(a.len() == 5 && b.len() == 5, "the five-point solver takes exactly five correspondences");
    let rows: [[f64; 9]; 5] = std::array::from_fn(|i| epipolar_row(a[i], b[i]));
    let Some([bx, by, bz, bw]) = null_basis(&rows) else { return Vec::new() };
    // E = x X + y Y + z Z + W, entrywise a linear polynomial
    let lin = |i: usize| -> Poly {
        let mut p = [0.0; 20];
        p[12] = bx[i];
        p[15] = by[i];
        p[18] = bz[i];
        p[19] = bw[i];
        p
    };
    let e: [Poly; 9] = std::array::from_fn(lin);
    let at = |r: usize, c: usize| &e[r * 3 + c];
    // det E
    let minor = |r1: usize, c1: usize, r2: usize, c2: usize| padd(&pmul(at(r1, c1), at(r2, c2)), &pmul(at(r1, c2), at(r2, c1)), -1.0);
    let det = padd(
        &padd(&pmul(at(0, 0), &minor(1, 1, 2, 2)), &pmul(at(0, 1), &minor(1, 0, 2, 2)), -1.0),
        &pmul(at(0, 2), &minor(1, 0, 2, 1)),
        1.0,
    );
    // E Eᵀ (degree two) and the trace constraint 2 E Eᵀ E − tr(E Eᵀ) E
    let eet: [Poly; 9] = std::array::from_fn(|i| {
        let (r, c) = (i / 3, i % 3);
        (0..3).fold([0.0; 20], |acc, k| padd(&acc, &pmul(at(r, k), at(c, k)), 1.0))
    });
    let tr = padd(&padd(&eet[0], &eet[4], 1.0), &eet[8], 1.0);
    let mut m = [[0.0f64; 20]; 10];
    m[0] = det;
    for i in 0..9 {
        let (r, c) = (i / 3, i % 3);
        let two_eete = (0..3).fold([0.0; 20], |acc, k| padd(&acc, &pmul(&eet[r * 3 + k], at(k, c)), 2.0));
        m[1 + i] = padd(&two_eete, &pmul(&tr, at(r, c)), -1.0);
    }
    let constraints = m;
    // Gauss-Jordan on the first ten columns, partial pivoting
    for col in 0..10 {
        let piv = (col..10).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs())).unwrap();
        if m[piv][col].abs() < 1e-12 {
            return Vec::new();
        }
        m.swap(col, piv);
        let d = m[col][col];
        for v in m[col].iter_mut() {
            *v /= d;
        }
        for r in 0..10 {
            if r != col && m[r][col] != 0.0 {
                let f = m[r][col];
                let pivot_row = m[col];
                for (v, p) in m[r].iter_mut().zip(pivot_row) {
                    *v -= f * p;
                }
            }
        }
    }
    // Row r now reads MONO[r] + Σ_{c≥10} m[r][c] MONO[c] = 0. Rows 4/5
    // (x²z, x²), 6/7 (y²z, y²) and 8/9 (xyz, xy) differ by a factor z in the
    // leading monomial, so row(2i) − z·row(2i+1) is free of it and leaves
    // x·(cubic in z) + y·(cubic in z) + (quartic in z) = 0.
    let px = |r: usize| [m[r][12], m[r][11], m[r][10]];
    let py = |r: usize| [m[r][15], m[r][14], m[r][13]];
    let p1 = |r: usize| [m[r][19], m[r][18], m[r][17], m[r][16]];
    let zmul = |p: &[f64]| upoly::mul(p, &[0.0, 1.0]);
    let bmat: [[Vec<f64>; 3]; 3] = std::array::from_fn(|i| {
        let (r0, r1) = (4 + 2 * i, 5 + 2 * i);
        [upoly::sub(&px(r0), &zmul(&px(r1))), upoly::sub(&py(r0), &zmul(&py(r1))), upoly::sub(&p1(r0), &zmul(&p1(r1)))]
    });
    let cof = |r1: usize, c1: usize, r2: usize, c2: usize| upoly::sub(&upoly::mul(&bmat[r1][c1], &bmat[r2][c2]), &upoly::mul(&bmat[r1][c2], &bmat[r2][c1]));
    let det_b = {
        let t0 = upoly::mul(&bmat[0][0], &cof(1, 1, 2, 2));
        let t1 = upoly::mul(&bmat[0][1], &cof(1, 0, 2, 2));
        let t2 = upoly::mul(&bmat[0][2], &cof(1, 0, 2, 1));
        upoly::add(&upoly::sub(&t0, &t1), &t2)
    };
    let mut out = Vec::new();
    for z in upoly::real_roots(&det_b) {
        let b: [V3; 3] = std::array::from_fn(|r| std::array::from_fn(|c| upoly::eval(&bmat[r][c], z)));
        // (x, y, 1) spans the null space of B(z): the largest cross product
        // of two of its rows
        let v = [cross(b[0], b[1]), cross(b[0], b[2]), cross(b[1], b[2])].into_iter().max_by(|p, q| norm(*p).total_cmp(&norm(*q))).unwrap();
        if v[2].abs() < 1e-12 * norm(v).max(1e-300) {
            continue;
        }
        let [x, y, z] = polish(&constraints, [v[0] / v[2], v[1] / v[2], z]);
        let em: M3 = std::array::from_fn(|i| x * bx[i] + y * by[i] + z * bz[i] + bw[i]);
        let n = em.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n.is_finite() && n > 0.0 {
            out.push(em.map(|v| v / n));
        }
    }
    out
}

/// Value and gradient of `p` at `(x, y, z)`.
fn eval_grad(p: &Poly, v: V3) -> (f64, V3) {
    let pw = |b: f64, e: u8| if e == 0 { 1.0 } else { b.powi(e as i32) };
    let dpw = |b: f64, e: u8| if e == 0 { 0.0 } else { e as f64 * pw(b, e - 1) };
    let (mut val, mut g) = (0.0, [0.0; 3]);
    for (c, m) in p.iter().zip(&MONO) {
        if *c == 0.0 {
            continue;
        }
        let (px, py, pz) = (pw(v[0], m[0]), pw(v[1], m[1]), pw(v[2], m[2]));
        val += c * px * py * pz;
        g[0] += c * dpw(v[0], m[0]) * py * pz;
        g[1] += c * px * dpw(v[1], m[1]) * pz;
        g[2] += c * px * py * dpw(v[2], m[2]);
    }
    (val, g)
}

/// Gauss-Newton on the ten ORIGINAL cubic constraints from a root of the
/// eliminated system. The elimination and the hidden-variable determinant
/// lose digits on ill-conditioned samples (a root came back 2.5e-7 from
/// the exact essential matrix on one exact five-point sample in twenty);
/// two or three steps on the unreduced equations restore them.
fn polish(constraints: &[Poly; 10], mut v: V3) -> V3 {
    let resid = |v: V3| constraints.iter().map(|p| eval_grad(p, v).0.powi(2)).sum::<f64>();
    let mut cur = resid(v);
    for _ in 0..4 {
        let mut jtj = [0.0f64; 9];
        let mut jtr = [0.0f64; 3];
        for p in constraints {
            let (r, g) = eval_grad(p, v);
            for i in 0..3 {
                jtr[i] -= g[i] * r;
                for j in 0..3 {
                    jtj[i * 3 + j] += g[i] * g[j];
                }
            }
        }
        let Some(d) = cholesky_solve(&jtj, &jtr, 3) else { break };
        let nv = [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
        let next = resid(nv);
        // a NaN step is refused along with a worse one
        if next.partial_cmp(&cur) != Some(std::cmp::Ordering::Less) {
            break;
        }
        (v, cur) = (nv, next);
    }
    v
}

/// Draw `k` distinct indices below `n`.
fn sample(rng: &mut Lcg, n: usize, k: usize, out: &mut Vec<usize>) {
    out.clear();
    while out.len() < k {
        let c = (rng.next_u32() as usize) % n;
        if !out.contains(&c) {
            out.push(c);
        }
    }
}

/// MSAC score (lower is better) and inlier count of `e` at squared
/// threshold `t2`.
fn msac(e: &M3, a: &[V3], b: &[V3], t2: f64) -> (f64, usize) {
    let mut s = 0.0;
    let mut c = 0;
    for (p, q) in a.iter().zip(b) {
        let d = sampson(e, *p, *q);
        if d < t2 {
            s += d;
            c += 1;
        } else {
            s += t2;
        }
    }
    (s, c)
}

fn inliers(e: &M3, a: &[V3], b: &[V3], t2: f64) -> Vec<usize> {
    (0..a.len()).filter(|&i| sampson(e, a[i], b[i]) < t2).collect()
}

/// Least-squares refits over a threshold shrinking from `LO_WIDEN` times the
/// inlier threshold down to it (Lebeda, Matas & Chum, "Fixing the Locally
/// Optimized RANSAC", BMVC 2012, §3).
const LO_WIDEN: f64 = 3.0;
const LO_STEPS: usize = 4;
/// Inner RANSAC of the local optimization: samples, and their size - a
/// non-minimal sample of inliers is far likelier to be all-good than a
/// minimal one of the data.
const LO_INNER: usize = 10;
const LO_SAMPLE: usize = 14;

/// Chum's local optimization of a new best model.
fn local_opt(e0: M3, a: &[V3], b: &[V3], t2: f64, rng: &mut Lcg) -> (M3, f64, usize) {
    let iterate = |mut e: M3| -> M3 {
        for step in 0..LO_STEPS {
            let widen = LO_WIDEN - (LO_WIDEN - 1.0) * step as f64 / (LO_STEPS - 1) as f64;
            let inl = inliers(&e, a, b, t2 * widen * widen);
            let (ia, ib): (Vec<V3>, Vec<V3>) = inl.iter().map(|&i| (a[i], b[i])).unzip();
            match essential_8pt(&ia, &ib) {
                Some(n) => e = n,
                None => break,
            }
        }
        e
    };
    let (s0, c0) = msac(&e0, a, b, t2);
    let mut best = (e0, s0, c0);
    let consider = |e: M3, best: &mut (M3, f64, usize)| {
        let (s, c) = msac(&e, a, b, t2);
        if s < best.1 {
            *best = (e, s, c);
        }
    };
    consider(iterate(e0), &mut best);
    let mut idx = Vec::new();
    for _ in 0..LO_INNER {
        let inl = inliers(&best.0, a, b, t2);
        if inl.len() <= LO_SAMPLE {
            break;
        }
        sample(rng, inl.len(), LO_SAMPLE, &mut idx);
        let (sa, sb): (Vec<V3>, Vec<V3>) = idx.iter().map(|&i| (a[inl[i]], b[inl[i]])).unzip();
        if let Some(e) = essential_8pt(&sa, &sb) {
            consider(iterate(e), &mut best);
        }
    }
    best
}

/// LO-RANSAC over [`essential_5pt`] with MSAC scoring. `thresh` is the
/// angular inlier threshold in radians (a pixel threshold divided by the
/// focal length); `iters` caps the adaptive iteration bound. Returns the
/// essential matrix, nonlinearly refined on its inliers, and the inlier
/// mask.
pub fn ransac_essential(a: &[V3], b: &[V3], thresh: f64, iters: usize, seed: u64) -> Option<(M3, Vec<bool>)> {
    let n = a.len();
    if n < 8 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Lcg::new(seed);
    let mut best: Option<(M3, f64, usize)> = None;
    let mut need = iters;
    let mut done = 0;
    let mut idx = Vec::with_capacity(5);
    while done < need {
        done += 1;
        sample(&mut rng, n, 5, &mut idx);
        let sa: Vec<V3> = idx.iter().map(|&i| a[i]).collect();
        let sb: Vec<V3> = idx.iter().map(|&i| b[i]).collect();
        for e in essential_5pt(&sa, &sb) {
            let (s, _) = msac(&e, a, b, t2);
            if best.as_ref().is_none_or(|bst| s < bst.1) {
                let lo = local_opt(e, a, b, t2, &mut rng);
                need = need.min(crate::ransac_iterations(lo.2 as f64 / n as f64, 5, iters));
                best = Some(lo);
            }
        }
    }
    let (mut e, _, _) = best?;
    for _ in 0..2 {
        let inl = inliers(&e, a, b, t2);
        if inl.len() < 8 {
            break;
        }
        let (ia, ib): (Vec<V3>, Vec<V3>) = inl.iter().map(|&i| (a[i], b[i])).unzip();
        e = refine_essential(&e, &ia, &ib, 20);
    }
    let mask = (0..n).map(|i| sampson(&e, a[i], b[i]) < t2).collect();
    Some((e, mask))
}

/// Levenberg-Marquardt on the signed Sampson distances of `a ↔ b` over the
/// essential manifold: `E = [t]ₓ R` with a left rotation increment and a
/// unit translation moved in its tangent plane - five parameters, so the
/// result is an exact essential matrix at every step. Returns `e` unchanged
/// if no step lowers the cost.
pub fn refine_essential(e: &M3, a: &[V3], b: &[V3], iters: usize) -> M3 {
    let (mut u, _, mut v) = svd3(e);
    if crate::linalg::det(&u) < 0.0 {
        u = u.map(|x| -x);
    }
    if crate::linalg::det(&v) < 0.0 {
        v = v.map(|x| -x);
    }
    let w: M3 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut r = mm(&mm(&u, &w), &transpose(&v));
    let mut t = [u[2], u[5], u[8]];
    let residuals = |r: &M3, t: V3| -> Vec<f64> {
        let e = mm(&crate::linalg::skew(t), r);
        a.iter()
            .zip(b)
            .map(|(p, q)| {
                let ea = mv(&e, *p);
                let etb = mtv(&e, *q);
                let num = dot(*q, ea);
                let g1 = sub(etb, scale(*p, dot(*p, etb)));
                let g2 = sub(ea, scale(*q, num));
                num / (dot(g1, g1) + dot(g2, g2)).max(1e-300).sqrt()
            })
            .collect()
    };
    let step = |r: &M3, t: V3, d: &[f64]| -> (M3, V3) {
        let (e1, e2) = tangent_basis(t);
        (mm(&exp_so3([d[0], d[1], d[2]]), r), normalize([t[0] + d[3] * e1[0] + d[4] * e2[0], t[1] + d[3] * e1[1] + d[4] * e2[1], t[2] + d[3] * e1[2] + d[4] * e2[2]]))
    };
    let mut res = residuals(&r, t);
    let mut cost: f64 = res.iter().map(|x| x * x).sum();
    let mut lambda = 1e-4;
    for _ in 0..iters {
        // central-difference Jacobian: five columns, cheap next to the
        // residuals themselves
        let h = 1e-7;
        let cols: Vec<Vec<f64>> = (0..5)
            .map(|k| {
                let mut dp = [0.0; 5];
                dp[k] = h;
                let (rp, tp) = step(&r, t, &dp);
                dp[k] = -h;
                let (rm, tm) = step(&r, t, &dp);
                residuals(&rp, tp).iter().zip(residuals(&rm, tm)).map(|(p, m)| (p - m) / (2.0 * h)).collect()
            })
            .collect();
        let mut jtj = [0.0f64; 25];
        let mut jtr = [0.0f64; 5];
        for i in 0..res.len() {
            for p in 0..5 {
                jtr[p] -= cols[p][i] * res[i];
                for q in 0..5 {
                    jtj[p * 5 + q] += cols[p][i] * cols[q][i];
                }
            }
        }
        let mut improved = false;
        for _ in 0..6 {
            let mut damped = jtj;
            for p in 0..5 {
                damped[p * 5 + p] += lambda * jtj[p * 5 + p].max(1e-12);
            }
            let Some(d) = cholesky_solve(&damped, &jtr, 5) else {
                lambda *= 10.0;
                continue;
            };
            let (nr, nt) = step(&r, t, &d);
            let nres = residuals(&nr, nt);
            let ncost: f64 = nres.iter().map(|x| x * x).sum();
            if ncost < cost {
                let gain = cost - ncost;
                (r, t, res, cost) = (nr, nt, nres, ncost);
                lambda = (lambda * 0.3).max(1e-12);
                improved = gain > 1e-12 * cost;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    mm(&crate::linalg::skew(t), &r)
}

/// Two unit vectors completing `v` (unit) to an orthonormal basis.
pub fn tangent_basis(v: V3) -> (V3, V3) {
    let a = if v[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1 = normalize(cross(v, a));
    (e1, cross(v, e1))
}

/// Linear triangulation of one point seen along unit bearings `rays` from
/// `poses`: the DLT over `f × (R X + t) = 0` (all three rows of each cross
/// product), solved once, then again with each view's rows divided by the
/// point's range from it so that every row measures the sine of an angular
/// error rather than a range-weighted one. Cheirality is the caller's.
pub fn triangulate(poses: &[&Pose], rays: &[V3]) -> Option<V3> {
    let mut weight = vec![1.0f64; poses.len()];
    let mut x = None;
    for _pass in 0..2 {
        let mut rows = Vec::with_capacity(poses.len() * 3);
        for ((p, f), w) in poses.iter().zip(rays).zip(&weight) {
            let r = &p.r;
            let row = |k: usize| [r[k * 3], r[k * 3 + 1], r[k * 3 + 2], p.t[k]];
            let pr = [row(0), row(1), row(2)];
            for (i, j) in [(1usize, 2usize), (2, 0), (0, 1)] {
                // (f × P X)_k = f_i (P_j X) − f_j (P_i X) for (k, i, j) cyclic
                rows.push((0..4).map(|c| w * (f[i] * pr[j][c] - f[j] * pr[i][c])).collect::<Vec<f64>>());
            }
        }
        let h = null_vector(&rows, 4);
        if h[3].abs() < 1e-12 {
            return None;
        }
        let xs = [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
        for (w, p) in weight.iter_mut().zip(poses) {
            *w = 1.0 / norm(p.to_cam(xs)).max(1e-12);
        }
        x = Some(xs);
    }
    x
}

/// The angle in radians between the rays from two camera centres to `x`.
pub fn ray_angle(c1: V3, c2: V3, x: V3) -> f64 {
    let (a, b) = (normalize(sub(x, c1)), normalize(sub(x, c2)));
    dot(a, b).clamp(-1.0, 1.0).acos()
}

/// The relative pose `(R, t)` (second camera, first at identity, `|t| = 1`)
/// of the four decompositions of `e` that puts the most correspondences
/// ahead along both of their bearings. Returns the pose and the
/// triangulated points of the correspondences in `mask` (`None` for ones
/// that did not triangulate ahead of both cameras).
pub fn relative_pose(e: &M3, a: &[V3], b: &[V3], mask: &[bool]) -> (Pose, Vec<Option<V3>>) {
    let (mut u, _, mut v) = svd3(e);
    if crate::linalg::det(&u) < 0.0 {
        u = u.map(|x| -x);
    }
    if crate::linalg::det(&v) < 0.0 {
        v = v.map(|x| -x);
    }
    let w: M3 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let r1 = mm(&mm(&u, &w), &transpose(&v));
    let r2 = mm(&mm(&u, &transpose(&w)), &transpose(&v));
    let t = [u[2], u[5], u[8]];
    let first = Pose::identity();
    let mut best: Option<(usize, Pose, Vec<Option<V3>>)> = None;
    for (r, t) in [(r1, t), (r1, t.map(|x| -x)), (r2, t), (r2, t.map(|x| -x))] {
        let second = Pose { r, t };
        let mut pts = Vec::with_capacity(a.len());
        let mut good = 0usize;
        for i in 0..a.len() {
            if !mask[i] {
                pts.push(None);
                continue;
            }
            let x = triangulate(&[&first, &second], &[a[i], b[i]]);
            let ok = x.filter(|x| dot(a[i], *x) > 0.0 && dot(b[i], second.to_cam(*x)) > 0.0);
            if ok.is_some() {
                good += 1;
            }
            pts.push(ok);
        }
        if best.as_ref().is_none_or(|(g, _, _)| good > *g) {
            best = Some((good, second, pts));
        }
    }
    let (_, p, pts) = best.unwrap();
    (p, pts)
}

/// Median angle between the two rays of the triangulated points - the
/// baseline measure that decides whether a pair can seed a reconstruction.
pub fn median_angle(second: &Pose, pts: &[Option<V3>]) -> f64 {
    let (c1, c2) = ([0.0; 3], second.centre());
    let mut a: Vec<f64> = pts.iter().flatten().map(|x| ray_angle(c1, c2, *x)).collect();
    if a.is_empty() {
        return 0.0;
    }
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}

/// `[t]ₓ R`, the essential matrix of a relative pose.
pub fn essential_of(p: &Pose) -> M3 {
    let t = if norm(p.t) > 0.0 { normalize(p.t) } else { p.t };
    mm(&crate::linalg::skew(t), &p.r)
}

#[cfg(test)]
mod tests {
    use super::upoly;

    /// The root finder returns every real root of a polynomial with close,
    /// widely spread and negative roots, and nothing for one with none.
    #[test]
    fn real_roots_finds_every_real_root() {
        let want = [-40.0, -1.5, 0.001, 0.002, 3.0, 250.0];
        let mut p = vec![1.0];
        for r in want {
            p = upoly::mul(&p, &[-r, 1.0]);
        }
        // times (z² + 1): two complex roots that must not appear
        p = upoly::mul(&p, &[1.0, 0.0, 1.0]);
        let got = upoly::real_roots(&p);
        assert_eq!(got.len(), want.len(), "{got:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9 * w.abs().max(1.0), "{g} against {w}");
        }
        assert!(upoly::real_roots(&[1.0, 0.0, 1.0]).is_empty());
    }
}
