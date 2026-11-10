// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spatial and channel-mixing primitives: depthwise and dense convolutions, the sequency Walsh transform, the
//! banded channel convolution and global response normalisation, each with its backward.

use super::ops::{rows, NORM_EPS};
use super::real::{r, Real};
use super::tape::{Id, Tape};
use crate::vision::ops::Map;

fn out_len(n: usize, k: usize, s: usize) -> usize {
    (n + 2 * (k / 2) - k) / s + 1
}

/// Geometry of a convolution with zero padding `k / 2`.
#[derive(Clone, Copy)]
struct Geo {
    m: Map,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
    ho: usize,
    wo: usize,
}

impl Geo {
    fn new(m: Map, kh: usize, kw: usize, sy: usize, sx: usize) -> Geo {
        Geo { m, kh, kw, sy, sx, ho: out_len(m.h, kh, sy), wo: out_len(m.w, kw, sx) }
    }

    /// Input row of output row `oy` and kernel row `ky`, when it lies in the image.
    fn in_y(&self, oy: usize, ky: usize) -> Option<usize> {
        let v = (oy * self.sy + ky) as isize - (self.kh / 2) as isize;
        (0..self.m.h as isize).contains(&v).then_some(v as usize)
    }

    fn in_x(&self, ox: usize, kx: usize) -> Option<usize> {
        let v = (ox * self.sx + kx) as isize - (self.kw / 2) as isize;
        (0..self.m.w as isize).contains(&v).then_some(v as usize)
    }

    /// Output row feeding input row `iy` through kernel row `ky`.
    fn out_y(&self, iy: usize, ky: usize) -> Option<usize> {
        let num = (iy + self.kh / 2) as isize - ky as isize;
        (num >= 0 && num % self.sy as isize == 0 && (num / self.sy as isize) < self.ho as isize).then_some((num / self.sy as isize) as usize)
    }
}

impl<R: Real> Tape<R> {
    /// Depthwise convolution, kernel `[kh * kw, c]`, zero padding `k / 2`.
    pub fn depthwise(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, sy: usize, sx: usize) -> (Id, Map) {
        let geo = Geo::new(m, kh, kw, sy, sx);
        assert_eq!(self.len(x), m.h * m.w * m.c, "depthwise: input does not match its map");
        assert_eq!(self.len(kernel), kh * kw * m.c, "depthwise: kernel does not match");
        let c = m.c;
        let mut y = vec![R::ZERO; geo.ho * geo.wo * c];
        {
            let (xv, kv) = (self.val(x), self.val(kernel));
            rows(&mut y, geo.wo * c, |oy, row| {
                for ox in 0..geo.wo {
                    let acc = &mut row[ox * c..(ox + 1) * c];
                    for ky in 0..kh {
                        let Some(iy) = geo.in_y(oy, ky) else { continue };
                        for kx in 0..kw {
                            let Some(ix) = geo.in_x(ox, kx) else { continue };
                            let src = &xv[(iy * m.w + ix) * c..][..c];
                            let k = &kv[(ky * kw + kx) * c..][..c];
                            acc.iter_mut().zip(src.iter().zip(k)).for_each(|(a, (s, k))| *a += *s * *k);
                        }
                    }
                }
            });
        }
        let id = self.op(y, move |vals, gy, g| {
            let (xv, kv) = (&vals[x.0], &vals[kernel.0]);
            let mut dx = vec![R::ZERO; xv.len()];
            rows(&mut dx, m.w * c, |iy, row| {
                for ky in 0..kh {
                    let Some(oy) = geo.out_y(iy, ky) else { continue };
                    for ox in 0..geo.wo {
                        let gr = &gy[(oy * geo.wo + ox) * c..][..c];
                        for kx in 0..kw {
                            let Some(ix) = geo.in_x(ox, kx) else { continue };
                            let k = &kv[(ky * kw + kx) * c..][..c];
                            row[ix * c..(ix + 1) * c].iter_mut().zip(gr.iter().zip(k)).for_each(|(a, (g, k))| *a += *g * *k);
                        }
                    }
                }
            });
            g.add(x, &dx);
            let mut dk = vec![R::ZERO; kv.len()];
            rows(&mut dk, c, |t, row| {
                let (ky, kx) = (t / kw, t % kw);
                for oy in 0..geo.ho {
                    let Some(iy) = geo.in_y(oy, ky) else { continue };
                    for ox in 0..geo.wo {
                        let Some(ix) = geo.in_x(ox, kx) else { continue };
                        let src = &xv[(iy * m.w + ix) * c..][..c];
                        let gr = &gy[(oy * geo.wo + ox) * c..][..c];
                        row.iter_mut().zip(src.iter().zip(gr)).for_each(|(a, (s, g))| *a += *s * *g);
                    }
                }
            });
            g.add(kernel, &dk);
        });
        (id, Map { h: geo.ho, w: geo.wo, c })
    }

    /// Dense convolution, kernel HWIO `[kh, kw, cin, cout]`, zero padding `k / 2`.
    pub fn conv_dense(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, cout: usize, sy: usize, sx: usize) -> (Id, Map) {
        let geo = Geo::new(m, kh, kw, sy, sx);
        let cin = m.c;
        assert_eq!(self.len(x), m.h * m.w * cin, "conv_dense: input does not match its map");
        assert_eq!(self.len(kernel), kh * kw * cin * cout, "conv_dense: kernel does not match");
        let mut y = vec![R::ZERO; geo.ho * geo.wo * cout];
        {
            let (xv, kv) = (self.val(x), self.val(kernel));
            rows(&mut y, geo.wo * cout, |oy, row| {
                for ox in 0..geo.wo {
                    let acc = &mut row[ox * cout..(ox + 1) * cout];
                    for ky in 0..kh {
                        let Some(iy) = geo.in_y(oy, ky) else { continue };
                        for kx in 0..kw {
                            let Some(ix) = geo.in_x(ox, kx) else { continue };
                            let src = &xv[(iy * m.w + ix) * cin..][..cin];
                            for (ci, s) in src.iter().enumerate() {
                                let k = &kv[((ky * kw + kx) * cin + ci) * cout..][..cout];
                                acc.iter_mut().zip(k).for_each(|(a, k)| *a += *s * *k);
                            }
                        }
                    }
                }
            });
        }
        let id = self.op(y, move |vals, gy, g| {
            let (xv, kv) = (&vals[x.0], &vals[kernel.0]);
            let mut dx = vec![R::ZERO; xv.len()];
            rows(&mut dx, m.w * cin, |iy, row| {
                for ky in 0..kh {
                    let Some(oy) = geo.out_y(iy, ky) else { continue };
                    for ox in 0..geo.wo {
                        let gr = &gy[(oy * geo.wo + ox) * cout..][..cout];
                        for kx in 0..kw {
                            let Some(ix) = geo.in_x(ox, kx) else { continue };
                            for ci in 0..cin {
                                let k = &kv[((ky * kw + kx) * cin + ci) * cout..][..cout];
                                row[ix * cin + ci] += gr.iter().zip(k).map(|(g, k)| *g * *k).sum::<R>();
                            }
                        }
                    }
                }
            });
            g.add(x, &dx);
            let mut dk = vec![R::ZERO; kv.len()];
            rows(&mut dk, cout, |t, row| {
                let (ky, kx, ci) = (t / (kw * cin), (t / cin) % kw, t % cin);
                for oy in 0..geo.ho {
                    let Some(iy) = geo.in_y(oy, ky) else { continue };
                    for ox in 0..geo.wo {
                        let Some(ix) = geo.in_x(ox, kx) else { continue };
                        let s = xv[(iy * m.w + ix) * cin + ci];
                        row.iter_mut().zip(&gy[(oy * geo.wo + ox) * cout..][..cout]).for_each(|(a, g)| *a += s * *g);
                    }
                }
            });
            g.add(kernel, &dk);
        });
        (id, Map { h: geo.ho, w: geo.wo, c: cout })
    }

    /// Nearest-neighbour 2x upsample of a map.
    pub fn upsample2(&mut self, x: Id, m: Map) -> (Id, Map) {
        let o = Map { h: m.h * 2, w: m.w * 2, c: m.c };
        let mut segs = Vec::with_capacity(o.h * o.w);
        for oy in 0..o.h {
            for ox in 0..o.w {
                segs.push(super::ops::Seg::copy((oy * o.w + ox) * m.c, ((oy / 2) * m.w + ox / 2) * m.c, m.c));
            }
        }
        (self.plan(x, o.h * o.w * o.c, segs), o)
    }

    /// Normalised sequency-ordered Walsh transform of every row of width `n`. It is orthonormal and symmetric, so
    /// the backward is the same transform of the output gradient.
    pub fn walsh(&mut self, x: Id, n: usize) -> Id {
        assert!(n.is_power_of_two() && self.len(x).is_multiple_of(n), "walsh: width {n} is not a power of two tiling the input");
        let mut y = self.val(x).to_vec();
        walsh_rows(&mut y, n);
        self.op(y, move |_, gy, g| {
            let mut dx = gy.to_vec();
            walsh_rows(&mut dx, n);
            g.add(x, &dx);
        })
    }

    /// Per-channel banded convolution along the channel axis: `out[j] = sum_t w[j, t] * u[j + t - r]` with zero
    /// padding, `w: [n, k]`, `k = 2r + 1`.
    pub fn band(&mut self, u: Id, n: usize, w: Id, k: usize) -> Id {
        assert_eq!(self.len(w), n * k, "band: weights are not {n}x{k}");
        assert!(n > 0 && self.len(u).is_multiple_of(n), "band: input does not tile {n}");
        let rad = k / 2;
        let mut y = vec![R::ZERO; self.len(u)];
        {
            let (uv, wv) = (self.val(u), self.val(w));
            rows(&mut y, n, |ri, out| {
                let src = &uv[ri * n..(ri + 1) * n];
                for (j, o) in out.iter_mut().enumerate() {
                    let lo = rad.saturating_sub(j);
                    let hi = k.min(n + rad - j);
                    *o = wv[j * k + lo..j * k + hi].iter().zip(&src[j + lo - rad..j + hi - rad]).map(|(a, b)| *a * *b).sum();
                }
            });
        }
        self.op(y, move |vals, gy, g| {
            let (uv, wv) = (&vals[u.0], &vals[w.0]);
            let mut du = vec![R::ZERO; uv.len()];
            rows(&mut du, n, |ri, out| {
                let gr = &gy[ri * n..(ri + 1) * n];
                for (m, o) in out.iter_mut().enumerate() {
                    // `u[m]` feeds `y[j]` for j in m - rad ..= m + rad through tap `m - j + rad`.
                    let (lo, hi) = (m.saturating_sub(rad), (m + rad + 1).min(n));
                    *o = (lo..hi).map(|j| gr[j] * wv[j * k + m + rad - j]).sum();
                }
            });
            g.add(u, &du);
            let mut dw = vec![R::ZERO; wv.len()];
            rows(&mut dw, k, |j, taps| {
                let lo = rad.saturating_sub(j);
                let hi = k.min(n + rad - j);
                for ri in 0..uv.len() / n {
                    let gv = gy[ri * n + j];
                    let src = &uv[ri * n..(ri + 1) * n];
                    for t in lo..hi {
                        taps[t] += gv * src[j + t - rad];
                    }
                }
            });
            g.add(w, &dw);
        })
    }

    /// The per-channel scale of global response normalisation over the valid rows of `y` (width `d`):
    /// `gamma * E / (mean(E) + eps) + 1` with `E = sqrt(sum_valid y^2 + eps)`.
    pub fn grn_scale(&mut self, y: Id, d: usize, valid: &[bool], gamma: Id) -> Id {
        assert_eq!(self.len(y), valid.len() * d, "grn_scale: mask does not match the rows");
        assert_eq!(self.len(gamma), d, "grn_scale: gamma is not {d} wide");
        let eps = r::<R>(NORM_EPS);
        let energy = grn_energy(self.val(y), d, valid);
        let mean = energy.iter().copied().sum::<R>() / r::<R>(d as f64);
        let scale: Vec<R> = (0..d).map(|c| self.val(gamma)[c] * energy[c] / (mean + eps) + R::ONE).collect();
        let valid = valid.to_vec();
        self.op(scale, move |vals, gs, g| {
            let (yv, gamma_v) = (&vals[y.0], &vals[gamma.0]);
            let energy = grn_energy(yv, d, &valid);
            let mean = energy.iter().copied().sum::<R>() / r::<R>(d as f64);
            let den = mean + eps;
            let dgamma: Vec<R> = (0..d).map(|c| gs[c] * energy[c] / den).collect();
            g.add(gamma, &dgamma);
            let cross: R = (0..d).map(|c| gs[c] * gamma_v[c] * energy[c]).sum();
            let de: Vec<R> = (0..d).map(|c| gs[c] * gamma_v[c] / den - cross / (den * den * r::<R>(d as f64))).collect();
            let mut dy = vec![R::ZERO; yv.len()];
            rows(&mut dy, d, |ri, row| {
                if valid[ri] {
                    for c in 0..d {
                        row[c] = de[c] * yv[ri * d + c] / energy[c];
                    }
                }
            });
            g.add(y, &dy);
        })
    }
}

fn grn_energy<R: Real>(y: &[R], d: usize, valid: &[bool]) -> Vec<R> {
    let mut e = vec![R::ZERO; d];
    for (row, ok) in y.chunks_exact(d).zip(valid) {
        if *ok {
            e.iter_mut().zip(row).for_each(|(a, v)| *a += *v * *v);
        }
    }
    e.iter().map(|v| (*v + r::<R>(NORM_EPS)).sqrt()).collect()
}

/// Normalised sequency Walsh transform of each row of width `n`, in place (same arithmetic as the inference
/// operator).
pub(crate) fn walsh_rows<R: Real>(x: &mut [R], n: usize) {
    let scale = R::ONE / r::<R>((n as f64).sqrt());
    rows(x, n, |_, row| {
        let mut scratch = vec![R::ZERO; n];
        let mut width = 1;
        while width < n {
            for (block, out) in row.chunks_exact(2 * width).zip(scratch.chunks_exact_mut(2 * width)) {
                let (left, right) = block.split_at(width);
                for p in 0..width {
                    let rv = if p % 2 == 0 { right[p] } else { -right[p] };
                    out[2 * p] = left[p] + rv;
                    out[2 * p + 1] = left[p] - rv;
                }
            }
            row.copy_from_slice(&scratch);
            width *= 2;
        }
        row.iter_mut().for_each(|v| *v *= scale);
    });
}
