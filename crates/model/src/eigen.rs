// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model compression and the numerical tooling
// it rests on for its clients. If your team needs expertise in decomposing and
// compressing neural-network weights and activations, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The symmetric eigendecomposition of a dense matrix of model width: the
//! principal directions of a covariance or second-moment matrix of hidden
//! states, whose dimension is a model's residual width (thousands).
//!
//! Householder reduction to tridiagonal form followed by the implicit QL
//! algorithm (the EISPACK `tred2`/`tql2` pair), `O(n³)` with one pass instead
//! of the many `O(n³)` sweeps cyclic Jacobi needs, so a 4096-wide matrix takes
//! minutes rather than hours. The transform is accumulated transposed (row `j`
//! of the working matrix is column `j` of the classical algorithm's `V`), which
//! makes every inner loop walk a contiguous row.

/// Eigenvalues ASCENDING and the matching unit eigenvectors as rows, of the
/// symmetric `n x n` matrix `a` (row-major; only its symmetric part is
/// meaningful, the lower triangle is read).
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    assert_eq!(a.len(), n * n, "symmetric_eigen: a holds {} values, an {n} x {n} matrix needs {}", a.len(), n * n);
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // w[j][k] = V[k][j]: the classical algorithm's V, transposed.
    let mut w: Vec<f64> = (0..n * n).map(|i| a[(i % n) * n + i / n]).collect();
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    tridiagonalize(&mut w, &mut d, &mut e, n);
    diagonalize(&mut w, &mut d, &mut e, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let values = order.iter().map(|&i| d[i]).collect();
    let vectors = order.iter().map(|&i| w[i * n..(i + 1) * n].to_vec()).collect();
    (values, vectors)
}

/// `tred2`: Householder reduction of the matrix held in `w` to a tridiagonal
/// one (diagonal `d`, subdiagonal `e[1..]`), leaving the accumulated
/// orthogonal transform in `w`.
fn tridiagonalize(w: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |j: usize, k: usize| j * n + k; // V[k][j]
    for j in 0..n {
        d[j] = w[at(j, n - 1)];
    }
    for i in (1..n).rev() {
        let scale: f64 = d[..i].iter().map(|v| v.abs()).sum();
        let mut h = 0.0;
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = w[at(j, i - 1)];
                w[at(j, i)] = 0.0;
                w[at(i, j)] = 0.0;
            }
        } else {
            for v in &mut d[..i] {
                *v /= scale;
                h += *v * *v;
            }
            let f = d[i - 1];
            let g = if f > 0.0 { -h.sqrt() } else { h.sqrt() };
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            e[..i].fill(0.0);
            for j in 0..i {
                let f = d[j];
                w[at(i, j)] = f;
                let row = &w[at(j, 0)..at(j, 0) + i];
                let mut g = e[j] + row[j] * f;
                for k in j + 1..i {
                    g += row[k] * d[k];
                    e[k] += row[k] * f;
                }
                e[j] = g;
            }
            let mut f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                let (f, g) = (d[j], e[j]);
                let row = &mut w[at(j, 0)..at(j, 0) + i];
                for k in j..i {
                    row[k] -= f * e[k] + g * d[k];
                }
                d[j] = w[at(j, i - 1)];
                w[at(j, i)] = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate the transformations.
    for i in 0..n - 1 {
        w[at(i, n - 1)] = w[at(i, i)];
        w[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = w[at(i + 1, k)] / h;
            }
            let (head, tail) = w.split_at_mut(at(i + 1, 0));
            let next = &tail[..=i];
            for j in 0..=i {
                let row = &mut head[at(j, 0)..at(j, 0) + i + 1];
                let g: f64 = next.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                row.iter_mut().zip(&d[..=i]).for_each(|(v, dk)| *v -= g * dk);
            }
        }
        w[at(i + 1, 0)..at(i + 1, 0) + i + 1].fill(0.0);
    }
    for j in 0..n {
        d[j] = w[at(j, n - 1)];
        w[at(j, n - 1)] = 0.0;
    }
    w[at(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

/// `tql2`: the implicit QL algorithm on the tridiagonal matrix `(d, e)`,
/// applying every rotation to the transform in `w` so that its rows end as
/// the eigenvectors and `d` as the eigenvalues (unordered).
fn diagonalize(w: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let (mut f, mut tst1) = (0.0f64, 0.0f64);
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > eps * tst1 {
            m += 1;
        }
        if m > l {
            loop {
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let h = g - d[l];
                for v in &mut d[l + 2..n] {
                    *v -= h;
                }
                f += h;
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0f64, 1.0f64, 1.0f64);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0f64, 0.0f64);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    let h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    let (lo, hi) = w.split_at_mut((i + 1) * n);
                    let (vi, vi1) = (&mut lo[i * n..], &mut hi[..n]);
                    for (a, b) in vi.iter_mut().zip(vi1.iter_mut()) {
                        let h = *b;
                        *b = s * *a + c * h;
                        *a = c * *a - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pair is an eigenpair, the vectors are orthonormal and the values
    /// ascend: for a random dense matrix, and for one with a repeated
    /// eigenvalue and a zero block (the degenerate cases the reduction special-cases).
    #[test]
    fn decomposes_symmetric_matrices() {
        let n = 37;
        let g = crate::hostmath::gaussian(n * n, 7);
        let random: Vec<f64> = (0..n * n).map(|i| (g[i] + g[(i % n) * n + i / n]) as f64).collect();
        let mut degenerate = vec![0.0f64; n * n];
        for i in 0..n / 2 {
            degenerate[i * n + i] = if i % 3 == 0 { 2.0 } else { -1.5 };
        }
        degenerate[n + 2] = 0.5;
        degenerate[2 * n + 1] = 0.5;
        for a in [random, degenerate] {
            let (values, vectors) = symmetric_eigen(&a, n);
            let scale = values.iter().map(|v| v.abs()).fold(1.0, f64::max);
            assert!(values.windows(2).all(|p| p[0] <= p[1]), "eigenvalues are not ascending: {values:?}");
            for (lambda, v) in values.iter().zip(&vectors) {
                for r in 0..n {
                    let av: f64 = (0..n).map(|c| a[r * n + c] * v[c]).sum();
                    assert!((av - lambda * v[r]).abs() < 1e-10 * scale, "A v != lambda v at row {r}: {av} vs {}", lambda * v[r]);
                }
            }
            for (i, u) in vectors.iter().enumerate() {
                for (j, v) in vectors.iter().enumerate() {
                    let dot: f64 = u.iter().zip(v).map(|(a, b)| a * b).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-10, "vectors {i} and {j} have dot {dot}");
                }
            }
        }
    }
}
