// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sequence primitives: rotary embeddings, masked grouped-query attention, causal taps, Sinkhorn normalisation,
//! lane mixing and the Kronecker rotations of the Hadamard MLP, each with its backward.

use super::ops::{rows, softmax_in_place};
use super::real::{r, Real};
use super::tape::{Id, Tape};

/// Shape of an attention call: `tq` queries over `tk` keys, `hq` query heads sharing `hk` key/value heads.
#[derive(Debug, Clone, Copy)]
pub struct AttnShape {
    /// Query rows.
    pub tq: usize,
    /// Key rows.
    pub tk: usize,
    /// Query heads.
    pub hq: usize,
    /// Key/value heads (divides `hq`).
    pub hk: usize,
    /// Query/key head width.
    pub qk: usize,
    /// Value head width.
    pub vh: usize,
    /// Whether query `i` sees only keys `0..=i`.
    pub causal: bool,
}

impl AttnShape {
    fn visible(&self, qi: usize) -> usize {
        if self.causal {
            qi + 1
        } else {
            self.tk
        }
    }
}

impl<R: Real> Tape<R> {
    /// Rotate pairs `(j, j + hd/2)` of every head of every row by `angles[row * hd/2 + j]` (radians).
    pub fn rope(&mut self, x: Id, heads: usize, hd: usize, angles: &[f64]) -> Id {
        let pairs = hd / 2;
        let width = heads * hd;
        assert_eq!(self.len(x) % width, 0, "rope: input does not tile {heads} heads of {hd}");
        assert_eq!(angles.len(), self.len(x) / width * pairs, "rope: one angle per row and pair");
        let cs: Vec<(R, R)> = angles.iter().map(|a| (r::<R>(a.cos()), r::<R>(a.sin()))).collect();
        let rotate = move |buf: &mut [R], sign: R| {
            rows(buf, width, |ri, row| {
                for head in row.chunks_exact_mut(hd) {
                    for j in 0..pairs {
                        let (c, s) = cs[ri * pairs + j];
                        let s = s * sign;
                        let (a, b) = (head[j], head[j + pairs]);
                        head[j] = a * c - b * s;
                        head[j + pairs] = b * c + a * s;
                    }
                }
            });
        };
        let mut y = self.val(x).to_vec();
        rotate(&mut y, R::ONE);
        self.op(y, move |_, gy, g| {
            let mut dx = gy.to_vec();
            rotate(&mut dx, -R::ONE);
            g.add(x, &dx);
        })
    }

    /// Scaled dot-product attention with softmax over the visible keys. `q: [tq, hq*qk]`, `k: [tk, hk*qk]`,
    /// `v: [tk, hk*vh]`; output `[tq, hq*vh]`.
    pub fn attention(&mut self, q: Id, k: Id, v: Id, s: AttnShape) -> Id {
        let (kw, vw) = (s.hk * s.qk, s.hk * s.vh);
        assert!(s.hk > 0 && s.hq.is_multiple_of(s.hk), "attention: {} query heads do not group over {} key heads", s.hq, s.hk);
        assert_eq!(self.len(q), s.tq * s.hq * s.qk, "attention: q shape");
        assert_eq!(self.len(k), s.tk * kw, "attention: k shape");
        assert_eq!(self.len(v), s.tk * vw, "attention: v shape");
        let group = s.hq / s.hk;
        let scale = r::<R>(1.0 / (s.qk as f64).sqrt());
        let probs = move |qv: &[R], kv: &[R], qi: usize, h: usize| -> Vec<R> {
            let g = h / group;
            let qh = &qv[(qi * s.hq + h) * s.qk..][..s.qk];
            let mut sc: Vec<R> = (0..s.visible(qi)).map(|t| qh.iter().zip(&kv[t * kw + g * s.qk..][..s.qk]).map(|(a, b)| *a * *b).sum::<R>() * scale).collect();
            softmax_in_place(&mut sc);
            sc
        };
        let mut out = vec![R::ZERO; s.tq * s.hq * s.vh];
        {
            let (qv, kv, vv) = (self.val(q), self.val(k), self.val(v));
            rows(&mut out, s.hq * s.vh, |qi, row| {
                for h in 0..s.hq {
                    let g = h / group;
                    let p = probs(qv, kv, qi, h);
                    let o = &mut row[h * s.vh..(h + 1) * s.vh];
                    for (t, pt) in p.iter().enumerate() {
                        o.iter_mut().zip(&vv[t * vw + g * s.vh..][..s.vh]).for_each(|(a, b)| *a += *pt * *b);
                    }
                }
            });
        }
        self.op(out, move |vals, gy, grads| {
            let (qv, kv, vv) = (&vals[q.0], &vals[k.0], &vals[v.0]);
            // Pass 1, per query row: probabilities and the score gradient of every (head, key), plus dq.
            let per_row = s.hq * s.tk;
            let mut ds = vec![R::ZERO; s.tq * per_row];
            let mut pr = vec![R::ZERO; s.tq * per_row];
            let mut dq = vec![R::ZERO; qv.len()];
            super::ops::rows_pair(&mut ds, per_row, &mut pr, per_row, |qi, dsr, prr| {
                for h in 0..s.hq {
                    let g = h / group;
                    let p = probs(qv, kv, qi, h);
                    let go = &gy[(qi * s.hq + h) * s.vh..][..s.vh];
                    let dp: Vec<R> = (0..p.len()).map(|t| go.iter().zip(&vv[t * vw + g * s.vh..][..s.vh]).map(|(a, b)| *a * *b).sum()).collect();
                    let dot: R = p.iter().zip(&dp).map(|(p, d)| *p * *d).sum();
                    for t in 0..p.len() {
                        prr[h * s.tk + t] = p[t];
                        dsr[h * s.tk + t] = p[t] * (dp[t] - dot) * scale;
                    }
                }
            });
            rows(&mut dq, s.hq * s.qk, |qi, row| {
                for h in 0..s.hq {
                    let g = h / group;
                    for t in 0..s.visible(qi) {
                        let w = ds[qi * per_row + h * s.tk + t];
                        row[h * s.qk..(h + 1) * s.qk].iter_mut().zip(&kv[t * kw + g * s.qk..][..s.qk]).for_each(|(a, b)| *a += w * *b);
                    }
                }
            });
            grads.add(q, &dq);
            // Pass 2, per key row: dk and dv gather over the queries that see it.
            let mut dk = vec![R::ZERO; kv.len()];
            let mut dv = vec![R::ZERO; vv.len()];
            super::ops::rows_pair(&mut dk, kw, &mut dv, vw, |t, dkr, dvr| {
                let first = if s.causal { t } else { 0 };
                for qi in first..s.tq {
                    for h in 0..s.hq {
                        let g = h / group;
                        let w = ds[qi * per_row + h * s.tk + t];
                        let p = pr[qi * per_row + h * s.tk + t];
                        dkr[g * s.qk..(g + 1) * s.qk].iter_mut().zip(&qv[(qi * s.hq + h) * s.qk..][..s.qk]).for_each(|(a, b)| *a += w * *b);
                        dvr[g * s.vh..(g + 1) * s.vh].iter_mut().zip(&gy[(qi * s.hq + h) * s.vh..][..s.vh]).for_each(|(a, b)| *a += p * *b);
                    }
                }
            });
            grads.add(k, &dk);
            grads.add(v, &dv);
        })
    }

    /// Causal dilated taps: `y[r] = sum_j taps[j] * x[r - j * dilation]` (rows before the start contribute
    /// nothing), `x: [t, w]`, `taps: [kt, w]`.
    pub fn causal_taps(&mut self, x: Id, taps: Id, w: usize, dilation: usize) -> Id {
        let kt = self.len(taps) / w;
        assert_eq!(self.len(taps), kt * w, "causal_taps: taps do not tile {w}");
        let t = self.len(x) / w;
        let mut y = vec![R::ZERO; t * w];
        {
            let (xv, tv) = (self.val(x), self.val(taps));
            rows(&mut y, w, |ri, row| {
                for j in 0..kt {
                    if ri >= j * dilation {
                        let src = &xv[(ri - j * dilation) * w..][..w];
                        row.iter_mut().zip(src.iter().zip(&tv[j * w..(j + 1) * w])).for_each(|(a, (s, k))| *a += *s * *k);
                    }
                }
            });
        }
        self.op(y, move |vals, gy, g| {
            let (xv, tv) = (&vals[x.0], &vals[taps.0]);
            let mut dx = vec![R::ZERO; xv.len()];
            let mut dt = vec![R::ZERO; tv.len()];
            for ri in 0..t {
                for j in 0..kt {
                    if ri < j * dilation {
                        continue;
                    }
                    let si = ri - j * dilation;
                    for c in 0..w {
                        dx[si * w + c] += gy[ri * w + c] * tv[j * w + c];
                        dt[j * w + c] += gy[ri * w + c] * xv[si * w + c];
                    }
                }
            }
            g.add(x, &dx);
            g.add(taps, &dt);
        })
    }

    /// Sinkhorn normalisation in log space of each `n x n` block of logits (`[t, n*n]`): alternately normalise
    /// rows and columns of `exp(logits)` for `iters` rounds; returns the doubly stochastic matrices.
    pub fn sinkhorn(&mut self, logits: Id, n: usize, iters: usize) -> Id {
        let nn = n * n;
        assert_eq!(self.len(logits) % nn, 0, "sinkhorn: logits do not tile {n}x{n}");
        let mut y = self.val(logits).to_vec();
        rows(&mut y, nn, |_, blk| {
            sinkhorn_trajectory(blk, n, iters);
        });
        y.iter_mut().for_each(|v| *v = v.exp());
        self.op(y, move |vals, gy, g| {
            let mut dl = vec![R::ZERO; gy.len()];
            let x = &vals[logits.0];
            rows(&mut dl, nn, |ri, gl| {
                // Replay the forward, keeping the state after every half step.
                let mut steps: Vec<Vec<R>> = Vec::with_capacity(2 * iters);
                let mut cur = x[ri * nn..(ri + 1) * nn].to_vec();
                for _ in 0..iters {
                    normalise_rows(&mut cur, n);
                    steps.push(cur.clone());
                    normalise_cols(&mut cur, n);
                    steps.push(cur.clone());
                }
                let last = steps.last().cloned().unwrap_or_else(|| cur.clone());
                let mut grad: Vec<R> = (0..nn).map(|i| gy[ri * nn + i] * last[i].exp()).collect();
                for it in (0..iters).rev() {
                    let col_out = &steps[2 * it + 1];
                    for j in 0..n {
                        let s: R = (0..n).map(|i| grad[i * n + j]).sum();
                        for i in 0..n {
                            grad[i * n + j] -= col_out[i * n + j].exp() * s;
                        }
                    }
                    let row_out = &steps[2 * it];
                    for i in 0..n {
                        let s: R = grad[i * n..(i + 1) * n].iter().copied().sum();
                        for j in 0..n {
                            grad[i * n + j] -= row_out[i * n + j].exp() * s;
                        }
                    }
                }
                gl.copy_from_slice(&grad);
            });
            g.add(logits, &dl);
        })
    }

    /// Per-row mixing of `b` source vectors into `a` outputs: `out[r, i, :] = sum_j h[r, i*b + j] * x[r, j, :]`
    /// with `h: [t, a*b]`, `x: [t, b*d]`, output `[t, a*d]`.
    pub fn row_mix(&mut self, h: Id, x: Id, a: usize, b: usize, d: usize) -> Id {
        let t = self.len(h) / (a * b);
        assert_eq!(self.len(h), t * a * b, "row_mix: weights do not tile {a}x{b}");
        assert_eq!(self.len(x), t * b * d, "row_mix: sources do not tile {b}x{d}");
        let mut y = vec![R::ZERO; t * a * d];
        {
            let (hv, xv) = (self.val(h), self.val(x));
            rows(&mut y, a * d, |ri, row| {
                for i in 0..a {
                    for j in 0..b {
                        let w = hv[ri * a * b + i * b + j];
                        row[i * d..(i + 1) * d].iter_mut().zip(&xv[(ri * b + j) * d..][..d]).for_each(|(o, v)| *o += w * *v);
                    }
                }
            });
        }
        self.op(y, move |vals, gy, g| {
            let (hv, xv) = (&vals[h.0], &vals[x.0]);
            let mut dh = vec![R::ZERO; hv.len()];
            rows(&mut dh, a * b, |ri, row| {
                for i in 0..a {
                    for j in 0..b {
                        row[i * b + j] = gy[(ri * a + i) * d..][..d].iter().zip(&xv[(ri * b + j) * d..][..d]).map(|(g, x)| *g * *x).sum();
                    }
                }
            });
            g.add(h, &dh);
            let mut dx = vec![R::ZERO; xv.len()];
            rows(&mut dx, b * d, |ri, row| {
                for j in 0..b {
                    for i in 0..a {
                        let w = hv[ri * a * b + i * b + j];
                        row[j * d..(j + 1) * d].iter_mut().zip(&gy[(ri * a + i) * d..][..d]).for_each(|(o, g)| *o += w * *g);
                    }
                }
            });
            g.add(x, &dx);
        })
    }

    /// Kronecker rotation of each row viewed as `[fa, fb]`: `out = A^T Z B` with `a: [fa, fa]`, `b: [fb, fb]`.
    pub fn kron(&mut self, z: Id, a: Id, b: Id, fa: usize, fb: usize) -> Id {
        let n = fa * fb;
        assert_eq!(self.len(a), fa * fa, "kron: left factor");
        assert_eq!(self.len(b), fb * fb, "kron: right factor");
        assert_eq!(self.len(z) % n, 0, "kron: input does not tile {fa}x{fb}");
        let mut y = self.val(z).to_vec();
        {
            let (av, bv) = (self.val(a), self.val(b));
            rows(&mut y, n, |_, row| kron_row(row, av, bv, fa, fb));
        }
        self.op(y, move |vals, gy, g| {
            let (zv, av, bv) = (&vals[z.0], &vals[a.0], &vals[b.0]);
            // dZ = A dOut B^T, via dtmp = A dOut (dtmp[i, l] = sum_k a[i, k] gout[k, l]).
            let dtmp_of = |gr: &[R]| -> Vec<R> {
                let mut dtmp = vec![R::ZERO; n];
                for i in 0..fa {
                    for k in 0..fa {
                        let w = av[i * fa + k];
                        dtmp[i * fb..(i + 1) * fb].iter_mut().zip(&gr[k * fb..(k + 1) * fb]).for_each(|(o, g)| *o += w * *g);
                    }
                }
                dtmp
            };
            let mut dz = vec![R::ZERO; zv.len()];
            rows(&mut dz, n, |ri, row| {
                let dtmp = dtmp_of(&gy[ri * n..(ri + 1) * n]);
                for i in 0..fa {
                    for j in 0..fb {
                        row[i * fb + j] = (0..fb).map(|l| dtmp[i * fb + l] * bv[j * fb + l]).sum();
                    }
                }
            });
            g.add(z, &dz);
            let mut da = vec![R::ZERO; av.len()];
            let mut db = vec![R::ZERO; bv.len()];
            for ri in 0..zv.len() / n {
                let zr = &zv[ri * n..(ri + 1) * n];
                let gr = &gy[ri * n..(ri + 1) * n];
                // tmp = Z B
                let mut tmp = vec![R::ZERO; n];
                for i in 0..fa {
                    for j in 0..fb {
                        let w = zr[i * fb + j];
                        tmp[i * fb..(i + 1) * fb].iter_mut().zip(&bv[j * fb..(j + 1) * fb]).for_each(|(o, b)| *o += w * *b);
                    }
                }
                let dtmp = dtmp_of(gr);
                for i in 0..fa {
                    for k in 0..fa {
                        da[i * fa + k] += (0..fb).map(|l| tmp[i * fb + l] * gr[k * fb + l]).sum::<R>();
                    }
                    for j in 0..fb {
                        for l in 0..fb {
                            db[j * fb + l] += zr[i * fb + j] * dtmp[i * fb + l];
                        }
                    }
                }
            }
            g.add(a, &da);
            g.add(b, &db);
        })
    }

    /// `out[r, i] = z[r, perm[i]]` for rows of width `n`.
    pub fn permute_cols(&mut self, z: Id, n: usize, perm: &[u32]) -> Id {
        assert_eq!(perm.len(), n, "permute_cols: one source per column");
        let mut y = vec![R::ZERO; self.len(z)];
        {
            let zv = self.val(z);
            rows(&mut y, n, |ri, row| row.iter_mut().zip(perm).for_each(|(o, &p)| *o = zv[ri * n + p as usize]));
        }
        let perm = perm.to_vec();
        self.op(y, move |_, gy, g| {
            let mut dz = vec![R::ZERO; gy.len()];
            rows(&mut dz, n, |ri, row| perm.iter().enumerate().for_each(|(i, &p)| row[p as usize] += gy[ri * n + i]));
            g.add(z, &dz);
        })
    }
}

/// `row = A^T row B` for one row viewed as `[fa, fb]`.
fn kron_row<R: Real>(row: &mut [R], a: &[R], b: &[R], fa: usize, fb: usize) {
    let n = fa * fb;
    let mut tmp = vec![R::ZERO; n];
    for i in 0..fa {
        for l in 0..fb {
            tmp[i * fb + l] = (0..fb).map(|j| row[i * fb + j] * b[j * fb + l]).sum();
        }
    }
    for k in 0..fa {
        for l in 0..fb {
            row[k * fb + l] = (0..fa).map(|i| a[i * fa + k] * tmp[i * fb + l]).sum();
        }
    }
}

fn log_sum_exp<R: Real>(values: impl Iterator<Item = R> + Clone) -> R {
    let first = values.clone().next().unwrap_or(R::ZERO);
    let max = values.clone().fold(first, |a, b| a.max(b));
    max + values.map(|v| (v - max).exp()).sum::<R>().ln()
}

fn normalise_rows<R: Real>(l: &mut [R], n: usize) {
    for i in 0..n {
        let lse = log_sum_exp(l[i * n..(i + 1) * n].iter().copied());
        l[i * n..(i + 1) * n].iter_mut().for_each(|v| *v -= lse);
    }
}

fn normalise_cols<R: Real>(l: &mut [R], n: usize) {
    for j in 0..n {
        let lse = log_sum_exp((0..n).map(|i| l[i * n + j]));
        (0..n).for_each(|i| l[i * n + j] -= lse);
    }
}

fn sinkhorn_trajectory<R: Real>(l: &mut [R], n: usize, iters: usize) {
    for _ in 0..iters {
        normalise_rows(l, n);
        normalise_cols(l, n);
    }
}
