// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Elementwise, dense and normalisation primitives of the training graph, each with its backward.
//!
//! Layouts follow the checkpoint: dense kernels are Flax `[in, out]`, feature maps are channel-last rows. A
//! "row" is one token or pixel; operations that take a width `d` treat the buffer as `len / d` rows.

use backend_cpu::par;

use super::real::{gelu_tanh, gelu_tanh_grad, r, sigmoid, silu, silu_grad, Real};
use super::tape::{Id, Tape};

/// Epsilon inside the square root of the zero-centred RMS norm.
pub const NORM_EPS: f64 = 1e-6;

/// Run `f(row_index, row)` over the rows of `buf` in parallel.
pub(crate) fn rows<R: Real>(buf: &mut [R], width: usize, f: impl Fn(usize, &mut [R]) + Sync) {
    if !buf.is_empty() {
        par::chunks_mut(buf, width, f);
    }
}

/// Like [`rows`] over two buffers at once: row `i` of `a` (width `wa`) with row `i` of `b` (width `wb`).
pub(crate) fn rows_pair<R: Real>(a: &mut [R], wa: usize, b: &mut [R], wb: usize, f: impl Fn(usize, &mut [R], &mut [R]) + Sync) {
    if !a.is_empty() {
        par::chunks2_mut(a, wa, b, wb, f);
    }
}

/// One weighted copy of a plan: `out[dst..dst+len] += w * src[src..src+len]`.
#[derive(Debug, Clone, Copy)]
pub struct Seg<R: Real> {
    /// Offset in the output.
    pub dst: usize,
    /// Offset in the source.
    pub src: usize,
    /// Elements copied.
    pub len: usize,
    /// Weight.
    pub w: R,
}

impl<R: Real> Seg<R> {
    /// A unit-weight copy.
    pub fn copy(dst: usize, src: usize, len: usize) -> Seg<R> {
        Seg { dst, src, len, w: R::ONE }
    }
}

impl<R: Real> Tape<R> {
    /// `a + b` (equal lengths).
    pub fn add(&mut self, a: Id, b: Id) -> Id {
        assert_eq!(self.len(a), self.len(b), "add: operand lengths differ");
        let v = self.val(a).iter().zip(self.val(b)).map(|(x, y)| *x + *y).collect();
        self.op(v, move |_, gy, g| {
            g.add(a, gy);
            g.add(b, gy);
        })
    }

    /// `a * b` elementwise (equal lengths).
    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        assert_eq!(self.len(a), self.len(b), "mul: operand lengths differ");
        let v = self.val(a).iter().zip(self.val(b)).map(|(x, y)| *x * *y).collect();
        self.op(v, move |vals, gy, g| {
            let da: Vec<R> = gy.iter().zip(&vals[b.0]).map(|(g, y)| *g * *y).collect();
            let db: Vec<R> = gy.iter().zip(&vals[a.0]).map(|(g, x)| *g * *x).collect();
            g.add(a, &da);
            g.add(b, &db);
        })
    }

    /// `c * a` for a constant.
    pub fn scale(&mut self, a: Id, c: f64) -> Id {
        let c = r::<R>(c);
        let v = self.val(a).iter().map(|x| *x * c).collect();
        self.op(v, move |_, gy, g| g.add(a, &gy.iter().map(|d| *d * c).collect::<Vec<_>>()))
    }

    /// `x[i] + b[i % len(b)]`: a bias per channel (period = `len(b)`).
    pub fn bias_add(&mut self, x: Id, b: Id) -> Id {
        let (n, bl) = (self.len(x), self.len(b));
        assert!(bl > 0 && n % bl == 0, "bias_add: bias of {bl} does not tile {n} elements");
        let v = self.val(x).iter().enumerate().map(|(i, v)| *v + self.val(b)[i % bl]).collect();
        self.op(v, move |_, gy, g| {
            g.add(x, gy);
            let db = g.slot(b, bl);
            gy.iter().enumerate().for_each(|(i, d)| db[i % bl] += *d);
        })
    }

    /// `x[i] * s[i % len(s)]`: a scale per channel.
    pub fn chan_mul(&mut self, x: Id, s: Id) -> Id {
        let (n, sl) = (self.len(x), self.len(s));
        assert!(sl > 0 && n % sl == 0, "chan_mul: scale of {sl} does not tile {n} elements");
        let v = self.val(x).iter().enumerate().map(|(i, v)| *v * self.val(s)[i % sl]).collect();
        self.op(v, move |vals, gy, g| {
            let dx: Vec<R> = gy.iter().enumerate().map(|(i, d)| *d * vals[s.0][i % sl]).collect();
            g.add(x, &dx);
            let ds = g.slot(s, sl);
            gy.iter().zip(&vals[x.0]).enumerate().for_each(|(i, (d, x))| ds[i % sl] += *d * *x);
        })
    }

    /// `x * s` for a one-element `s`.
    pub fn scalar_mul(&mut self, x: Id, s: Id) -> Id {
        assert_eq!(self.len(s), 1, "scalar_mul: the scale is one element");
        let sv = self.val(s)[0];
        let v = self.val(x).iter().map(|v| *v * sv).collect();
        self.op(v, move |vals, gy, g| {
            let sv = vals[s.0][0];
            g.add(x, &gy.iter().map(|d| *d * sv).collect::<Vec<_>>());
            let ds: R = gy.iter().zip(&vals[x.0]).map(|(d, x)| *d * *x).sum();
            g.add(s, &[ds]);
        })
    }

    /// `sigmoid(x)`.
    pub fn sigmoid(&mut self, x: Id) -> Id {
        let v = self.val(x).iter().map(|v| sigmoid(*v)).collect();
        let out_id = Id(self.value_count());
        self.op(v, move |vals, gy, g| {
            let y = &vals[out_id.0];
            g.add(x, &gy.iter().zip(y).map(|(d, y)| *d * *y * (R::ONE - *y)).collect::<Vec<_>>());
        })
    }

    /// `x * sigmoid(x)`.
    pub fn silu(&mut self, x: Id) -> Id {
        let v = self.val(x).iter().map(|v| silu(*v)).collect();
        self.op(v, move |vals, gy, g| g.add(x, &gy.iter().zip(&vals[x.0]).map(|(d, x)| *d * silu_grad(*x)).collect::<Vec<_>>()))
    }

    /// GELU with the tanh approximation.
    pub fn gelu_tanh(&mut self, x: Id) -> Id {
        let v = self.val(x).iter().map(|v| gelu_tanh(*v)).collect();
        self.op(v, move |vals, gy, g| g.add(x, &gy.iter().zip(&vals[x.0]).map(|(d, x)| *d * gelu_tanh_grad(*x)).collect::<Vec<_>>()))
    }

    /// Identity forward; the gradient flowing back is multiplied by `factor`. This is the explicit gradient scale
    /// the heads send into the shared encoder (`head_grad`): a deliberate departure from the true derivative.
    pub fn grad_scale(&mut self, x: Id, factor: f64) -> Id {
        let f = r::<R>(factor);
        let v = self.val(x).to_vec();
        self.op(v, move |_, gy, g| g.add(x, &gy.iter().map(|d| *d * f).collect::<Vec<_>>()))
    }

    /// `y = x W` for `x: [rows, inn]`, `w: [inn, out]` (a Flax kernel).
    pub fn linear(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id {
        assert_eq!(self.len(x), rows_n * inn, "linear: input is not {rows_n}x{inn}");
        assert_eq!(self.len(w), inn * out, "linear: kernel is not {inn}x{out}");
        let mut y = vec![R::ZERO; rows_n * out];
        {
            let (xv, wv) = (self.val(x), self.val(w));
            rows(&mut y, out, |ri, row| {
                for (i, xi) in xv[ri * inn..(ri + 1) * inn].iter().enumerate() {
                    let wr = &wv[i * out..(i + 1) * out];
                    row.iter_mut().zip(wr).for_each(|(a, w)| *a += *xi * *w);
                }
            });
        }
        self.op(y, move |vals, gy, g| {
            let (xv, wv) = (&vals[x.0], &vals[w.0]);
            let mut dx = vec![R::ZERO; rows_n * inn];
            rows(&mut dx, inn, |ri, row| {
                let gr = &gy[ri * out..(ri + 1) * out];
                for (i, a) in row.iter_mut().enumerate() {
                    *a = gr.iter().zip(&wv[i * out..(i + 1) * out]).map(|(g, w)| *g * *w).sum();
                }
            });
            g.add(x, &dx);
            let mut dw = vec![R::ZERO; inn * out];
            rows(&mut dw, out, |i, row| {
                for ri in 0..rows_n {
                    let xi = xv[ri * inn + i];
                    row.iter_mut().zip(&gy[ri * out..(ri + 1) * out]).for_each(|(a, g)| *a += xi * *g);
                }
            });
            g.add(w, &dw);
        })
    }

    /// `y = x W^T` for `x: [rows, inn]`, `w: [out, inn]` (a table used as a projection, e.g. tied logits).
    pub fn linear_nt(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id {
        assert_eq!(self.len(x), rows_n * inn, "linear_nt: input is not {rows_n}x{inn}");
        assert_eq!(self.len(w), out * inn, "linear_nt: table is not {out}x{inn}");
        let mut y = vec![R::ZERO; rows_n * out];
        {
            let (xv, wv) = (self.val(x), self.val(w));
            rows(&mut y, out, |ri, row| {
                let xr = &xv[ri * inn..(ri + 1) * inn];
                for (o, a) in row.iter_mut().enumerate() {
                    *a = xr.iter().zip(&wv[o * inn..(o + 1) * inn]).map(|(x, w)| *x * *w).sum();
                }
            });
        }
        self.op(y, move |vals, gy, g| {
            let (xv, wv) = (&vals[x.0], &vals[w.0]);
            let mut dx = vec![R::ZERO; rows_n * inn];
            rows(&mut dx, inn, |ri, row| {
                for (o, gv) in gy[ri * out..(ri + 1) * out].iter().enumerate() {
                    row.iter_mut().zip(&wv[o * inn..(o + 1) * inn]).for_each(|(a, w)| *a += *gv * *w);
                }
            });
            g.add(x, &dx);
            let mut dw = vec![R::ZERO; out * inn];
            rows(&mut dw, inn, |o, row| {
                for ri in 0..rows_n {
                    let gv = gy[ri * out + o];
                    row.iter_mut().zip(&xv[ri * inn..(ri + 1) * inn]).for_each(|(a, x)| *a += gv * *x);
                }
            });
            g.add(w, &dw);
        })
    }

    /// `(1 + scale) * x / sqrt(mean(x^2) + eps)` over rows of width `d`.
    pub fn zc_norm(&mut self, x: Id, d: usize, scale: Id) -> Id {
        assert_eq!(self.len(scale), d, "zc_norm: scale is not {d} wide");
        assert!(d > 0 && self.len(x).is_multiple_of(d), "zc_norm: input does not tile {d}");
        let mut y = self.val(x).to_vec();
        {
            let s = self.val(scale);
            rows(&mut y, d, |_, row| {
                let inv = inv_rms(row, d);
                row.iter_mut().zip(s).for_each(|(v, s)| *v = *v * (R::ONE + *s) * inv);
            });
        }
        self.op(y, move |vals, gy, g| {
            let (xv, s) = (&vals[x.0], &vals[scale.0]);
            let mut dx = vec![R::ZERO; xv.len()];
            rows(&mut dx, d, |ri, row| {
                let xr = &xv[ri * d..(ri + 1) * d];
                let gr = &gy[ri * d..(ri + 1) * d];
                let inv = inv_rms(xr, d);
                let dot: R = (0..d).map(|j| (R::ONE + s[j]) * gr[j] * xr[j]).sum();
                let k = inv * inv * inv * dot / r::<R>(d as f64);
                for j in 0..d {
                    row[j] = inv * (R::ONE + s[j]) * gr[j] - xr[j] * k;
                }
            });
            g.add(x, &dx);
            let ds = g.slot(scale, d);
            for ri in 0..xv.len() / d {
                let xr = &xv[ri * d..(ri + 1) * d];
                let inv = inv_rms(xr, d);
                for j in 0..d {
                    ds[j] += gy[ri * d + j] * xr[j] * inv;
                }
            }
        })
    }

    /// `x / sqrt(mean(x^2) + eps)` over rows of width `d` (no scale).
    pub fn rms_unit(&mut self, x: Id, d: usize) -> Id {
        assert!(d > 0 && self.len(x).is_multiple_of(d), "rms_unit: input does not tile {d}");
        let mut y = self.val(x).to_vec();
        rows(&mut y, d, |_, row| {
            let inv = inv_rms(row, d);
            row.iter_mut().for_each(|v| *v *= inv);
        });
        self.op(y, move |vals, gy, g| {
            let xv = &vals[x.0];
            let mut dx = vec![R::ZERO; xv.len()];
            rows(&mut dx, d, |ri, row| {
                let xr = &xv[ri * d..(ri + 1) * d];
                let gr = &gy[ri * d..(ri + 1) * d];
                let inv = inv_rms(xr, d);
                let dot: R = xr.iter().zip(gr).map(|(x, g)| *x * *g).sum();
                let k = inv * inv * inv * dot / r::<R>(d as f64);
                for j in 0..d {
                    row[j] = inv * gr[j] - xr[j] * k;
                }
            });
            g.add(x, &dx);
        })
    }

    /// Zero the rows of width `d` whose `valid` entry is false.
    pub fn mask_rows(&mut self, x: Id, d: usize, valid: &[bool]) -> Id {
        assert_eq!(self.len(x), valid.len() * d, "mask_rows: mask does not match the rows");
        let mut y = self.val(x).to_vec();
        for (row, ok) in y.chunks_exact_mut(d).zip(valid) {
            if !ok {
                row.fill(R::ZERO);
            }
        }
        let valid = valid.to_vec();
        self.op(y, move |_, gy, g| {
            let mut dx = gy.to_vec();
            for (row, ok) in dx.chunks_exact_mut(d).zip(&valid) {
                if !ok {
                    row.fill(R::ZERO);
                }
            }
            g.add(x, &dx);
        })
    }

    /// A linear data-movement op: `out = sum of weighted segment copies of src` (`out_len` elements).
    /// Gathers, scatters, concatenation, slicing, upsampling, pooling and broadcasts are all plans.
    pub fn plan(&mut self, src: Id, out_len: usize, segs: Vec<Seg<R>>) -> Id {
        let n = self.len(src);
        let mut y = vec![R::ZERO; out_len];
        {
            let sv = self.val(src);
            for s in &segs {
                assert!(s.dst + s.len <= out_len && s.src + s.len <= n, "plan: a segment leaves its buffer");
                y[s.dst..s.dst + s.len].iter_mut().zip(&sv[s.src..s.src + s.len]).for_each(|(a, v)| *a += s.w * *v);
            }
        }
        self.op(y, move |_, gy, g| {
            let dx = g.slot(src, n);
            for s in &segs {
                dx[s.src..s.src + s.len].iter_mut().zip(&gy[s.dst..s.dst + s.len]).for_each(|(a, d)| *a += s.w * *d);
            }
        })
    }

    /// Rows `idx` of `x` (width `d`), in order.
    pub fn gather_rows(&mut self, x: Id, d: usize, idx: &[usize]) -> Id {
        let segs = idx.iter().enumerate().map(|(o, &i)| Seg::copy(o * d, i * d, d)).collect();
        self.plan(x, idx.len() * d, segs)
    }

    /// `a` then `b` along the row axis.
    pub fn concat(&mut self, a: Id, b: Id) -> Id {
        let (na, nb) = (self.len(a), self.len(b));
        let mut v = self.val(a).to_vec();
        v.extend_from_slice(self.val(b));
        self.op(v, move |_, gy, g| {
            g.add(a, &gy[..na]);
            g.add(b, &gy[na..na + nb]);
        })
    }

    /// Elements `[start, start + len)` of `x`.
    pub fn slice(&mut self, x: Id, start: usize, len: usize) -> Id {
        self.plan(x, len, vec![Seg::copy(0, start, len)])
    }

    /// Row-wise softmax over width `d`.
    pub fn softmax_rows(&mut self, x: Id, d: usize) -> Id {
        let mut y = self.val(x).to_vec();
        rows(&mut y, d, |_, row| softmax_in_place(row));
        let out_id = Id(self.value_count());
        self.op(y, move |vals, gy, g| {
            let p = &vals[out_id.0];
            let mut dx = vec![R::ZERO; p.len()];
            rows(&mut dx, d, |ri, row| {
                let (pr, gr) = (&p[ri * d..(ri + 1) * d], &gy[ri * d..(ri + 1) * d]);
                let dot: R = pr.iter().zip(gr).map(|(p, g)| *p * *g).sum();
                for j in 0..d {
                    row[j] = pr[j] * (gr[j] - dot);
                }
            });
            g.add(x, &dx);
        })
    }

    /// Per-row dot product of `a` and `b` (width `d`): `[rows]`.
    pub fn row_dot(&mut self, a: Id, b: Id, d: usize) -> Id {
        assert_eq!(self.len(a), self.len(b), "row_dot: operand lengths differ");
        let v: Vec<R> = self.val(a).chunks_exact(d).zip(self.val(b).chunks_exact(d)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| *p * *q).sum()).collect();
        self.op(v, move |vals, gy, g| {
            let da: Vec<R> = vals[b.0].chunks_exact(d).zip(gy).flat_map(|(y, g)| y.iter().map(move |y| *y * *g)).collect();
            let db: Vec<R> = vals[a.0].chunks_exact(d).zip(gy).flat_map(|(x, g)| x.iter().map(move |x| *x * *g)).collect();
            g.add(a, &da);
            g.add(b, &db);
        })
    }

    /// `x[r, :] * s[r]` for one scale per row of width `d`.
    pub fn row_mul(&mut self, s: Id, x: Id, d: usize) -> Id {
        assert_eq!(self.len(x), self.len(s) * d, "row_mul: one scale per row");
        let v: Vec<R> = self.val(x).chunks_exact(d).zip(self.val(s)).flat_map(|(row, s)| row.iter().map(move |v| *v * *s)).collect();
        self.op(v, move |vals, gy, g| {
            let dx: Vec<R> = gy.chunks_exact(d).zip(&vals[s.0]).flat_map(|(gr, s)| gr.iter().map(move |g| *g * *s)).collect();
            g.add(x, &dx);
            let ds: Vec<R> = gy.chunks_exact(d).zip(vals[x.0].chunks_exact(d)).map(|(gr, xr)| gr.iter().zip(xr).map(|(g, x)| *g * *x).sum()).collect();
            g.add(s, &ds);
        })
    }

    /// Sum of all elements of `x` times `weight`: a scalar.
    pub fn sum_weighted(&mut self, x: Id, weight: f64) -> Id {
        let w = r::<R>(weight);
        let s: R = self.val(x).iter().copied().sum::<R>() * w;
        let n = self.len(x);
        self.op(vec![s], move |_, gy, g| g.add(x, &vec![gy[0] * w; n]))
    }
}

/// `1 / sqrt(mean(row^2) + eps)`.
#[inline]
pub(crate) fn inv_rms<R: Real>(row: &[R], d: usize) -> R {
    let ms = row.iter().map(|v| *v * *v).sum::<R>() / r::<R>(d as f64);
    R::ONE / (ms + r::<R>(NORM_EPS)).sqrt()
}

/// Numerically stable softmax of one row, in place.
pub(crate) fn softmax_in_place<R: Real>(row: &mut [R]) {
    let max = row.iter().copied().fold(row[0], |a, b| a.max(b));
    let mut sum = R::ZERO;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    row.iter_mut().for_each(|v| *v = *v / sum);
}
