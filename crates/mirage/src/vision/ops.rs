// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host operators of the Mirage vision path on row-major `f32` buffers.
//!
//! Every feature map is channel-last (`[height, width, channels]` flattened),
//! so a "row" below is one pixel's channel vector. Work fans out over rows
//! through `backend_cpu::par`; each output row depends only on its own inputs,
//! so results do not depend on the thread count.

use backend_cpu::par;
use model::hostmath;

/// Epsilon inside the square root of the zero-centred RMS norm.
pub const NORM_EPS: f32 = 1e-6;

/// A dense matrix applied as `y = x W`, stored transposed (`[out, in]`) so the
/// shared batched matmul streams it.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weights, `[out, in]` row-major.
    pub wt: Vec<f32>,
    /// Input width.
    pub inn: usize,
    /// Output width.
    pub out: usize,
}

impl Linear {
    /// From a Flax kernel `[in, out]`.
    pub fn from_flax(kernel: &[f32], inn: usize, out: usize) -> Linear {
        assert_eq!(kernel.len(), inn * out, "kernel is {}, expected {inn}x{out}", kernel.len());
        let mut wt = vec![0f32; inn * out];
        for i in 0..inn {
            for o in 0..out {
                wt[o * inn + i] = kernel[i * out + o];
            }
        }
        Linear { wt, inn, out }
    }

    /// `[rows, in] -> [rows, out]`.
    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        if rows == 0 {
            return Vec::new();
        }
        hostmath::linear_rows(x, &self.wt, rows, self.inn, self.out)
    }

    /// [`Linear::apply`] plus a per-output bias.
    pub fn apply_bias(&self, x: &[f32], rows: usize, bias: &[f32]) -> Vec<f32> {
        let mut y = self.apply(x, rows);
        par::rows_mut(&mut y, self.out, |_, row| row.iter_mut().zip(bias).for_each(|(v, b)| *v += b));
        y
    }
}

/// `(1 + scale) * x / sqrt(mean(x^2) + eps)` over the last axis of width `d`.
pub fn zc_norm(x: &[f32], d: usize, scale: &[f32]) -> Vec<f32> {
    assert_eq!(scale.len(), d);
    let mut y = x.to_vec();
    par::rows_mut(&mut y, d, |_, row| {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + NORM_EPS).sqrt();
        row.iter_mut().zip(scale).for_each(|(v, s)| *v *= (1.0 + s) * inv);
    });
    y
}

/// Unit-RMS normalisation without a scale (`x / sqrt(mean(x^2) + eps)`).
pub fn rms_unit(x: &[f32], d: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    par::rows_mut(&mut y, d, |_, row| {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + NORM_EPS).sqrt();
        row.iter_mut().for_each(|v| *v *= inv);
    });
    y
}

/// `x * sigmoid(x)` in place.
pub fn silu_inplace(x: &mut [f32]) {
    par::chunks_mut(x, 4096, |_, c| c.iter_mut().for_each(|v| *v = hostmath::silu(*v)));
}

/// Logistic function.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Zero the rows of `x` (width `d`) whose mask entry is false.
pub fn mask_rows(x: &mut [f32], d: usize, valid: &[bool]) {
    par::rows_mut(x, d, |r, row| {
        if !valid[r] {
            row.fill(0.0);
        }
    });
}

/// Which cells of a `size x size` grid lie inside a resized image `h x w` on
/// a 512 canvas: `ceil(h / stride)` rows and `ceil(w / stride)` columns.
pub fn valid_cells(h: usize, w: usize, size: usize) -> Vec<bool> {
    let stride = 512 / size;
    let (rows, cols) = (h.div_ceil(stride), w.div_ceil(stride));
    (0..size * size).map(|i| i / size < rows && i % size < cols).collect()
}

/// A convolution's output geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Map {
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Channels.
    pub c: usize,
}

fn out_len(n: usize, k: usize, s: usize) -> usize {
    (n + 2 * (k / 2) - k) / s + 1
}

/// Depthwise convolution, zero padding `k / 2`, kernel `[kh * kw, c]` (an HWIO
/// `[kh, kw, 1, c]` tensor flattened).
pub fn depthwise(x: &[f32], m: Map, kernel: &[f32], kh: usize, kw: usize, sy: usize, sx: usize) -> (Vec<f32>, Map) {
    assert_eq!(x.len(), m.h * m.w * m.c);
    assert_eq!(kernel.len(), kh * kw * m.c);
    let (ho, wo) = (out_len(m.h, kh, sy), out_len(m.w, kw, sx));
    let mut y = vec![0f32; ho * wo * m.c];
    let (ph, pw) = ((kh / 2) as isize, (kw / 2) as isize);
    par::rows_mut(&mut y, wo * m.c, |oy, row| {
        for ox in 0..wo {
            let acc = &mut row[ox * m.c..(ox + 1) * m.c];
            for ky in 0..kh {
                let iy = (oy * sy) as isize + ky as isize - ph;
                if iy < 0 || iy >= m.h as isize {
                    continue;
                }
                for kx in 0..kw {
                    let ix = (ox * sx) as isize + kx as isize - pw;
                    if ix < 0 || ix >= m.w as isize {
                        continue;
                    }
                    let src = &x[(iy as usize * m.w + ix as usize) * m.c..][..m.c];
                    let k = &kernel[(ky * kw + kx) * m.c..][..m.c];
                    acc.iter_mut().zip(src.iter().zip(k)).for_each(|(a, (s, k))| *a += s * k);
                }
            }
        }
    });
    (y, Map { h: ho, w: wo, c: m.c })
}

/// Dense convolution, zero padding `k / 2`, kernel HWIO `[kh, kw, cin, cout]`.
pub fn conv_dense(x: &[f32], m: Map, kernel: &[f32], kh: usize, kw: usize, cout: usize, sy: usize, sx: usize) -> (Vec<f32>, Map) {
    assert_eq!(x.len(), m.h * m.w * m.c);
    assert_eq!(kernel.len(), kh * kw * m.c * cout);
    let (ho, wo) = (out_len(m.h, kh, sy), out_len(m.w, kw, sx));
    let mut y = vec![0f32; ho * wo * cout];
    let (ph, pw) = ((kh / 2) as isize, (kw / 2) as isize);
    par::rows_mut(&mut y, wo * cout, |oy, row| {
        for ox in 0..wo {
            let acc = &mut row[ox * cout..(ox + 1) * cout];
            for ky in 0..kh {
                let iy = (oy * sy) as isize + ky as isize - ph;
                if iy < 0 || iy >= m.h as isize {
                    continue;
                }
                for kx in 0..kw {
                    let ix = (ox * sx) as isize + kx as isize - pw;
                    if ix < 0 || ix >= m.w as isize {
                        continue;
                    }
                    let src = &x[(iy as usize * m.w + ix as usize) * m.c..][..m.c];
                    for (ci, &s) in src.iter().enumerate() {
                        let k = &kernel[((ky * kw + kx) * m.c + ci) * cout..][..cout];
                        acc.iter_mut().zip(k).for_each(|(a, k)| *a += s * k);
                    }
                }
            }
        }
    });
    (y, Map { h: ho, w: wo, c: cout })
}

/// Nearest-neighbour 2x upsample.
pub fn upsample2(x: &[f32], m: Map) -> (Vec<f32>, Map) {
    let o = Map { h: m.h * 2, w: m.w * 2, c: m.c };
    let mut y = vec![0f32; o.h * o.w * o.c];
    par::rows_mut(&mut y, o.w * o.c, |oy, row| {
        for ox in 0..o.w {
            row[ox * o.c..(ox + 1) * o.c].copy_from_slice(&x[((oy / 2) * m.w + ox / 2) * m.c..][..m.c]);
        }
    });
    (y, o)
}

/// Elementwise `a += b`.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    par::chunks_mut(a, 4096, |i, c| c.iter_mut().zip(&b[i * 4096..]).for_each(|(a, b)| *a += b));
}

/// Normalised sequency-ordered Walsh transform of every row of width `n`
/// (a power of two); it is its own inverse.
pub fn walsh_sequency_rows(x: &mut [f32], n: usize) {
    assert!(n.is_power_of_two());
    let scale = 1.0 / (n as f32).sqrt();
    par::rows_mut(x, n, |_, row| {
        let mut scratch = vec![0f32; n];
        let mut width = 1;
        while width < n {
            // Blocks of `2 * width`: left half L, right half R (alternating sign by position),
            // written back as interleaved (L + R, L - R) pairs per position.
            for (block, out) in row.chunks_exact(2 * width).zip(scratch.chunks_exact_mut(2 * width)) {
                let (left, right) = block.split_at(width);
                for p in 0..width {
                    let r = if p % 2 == 0 { right[p] } else { -right[p] };
                    out[2 * p] = left[p] + r;
                    out[2 * p + 1] = left[p] - r;
                }
            }
            row.copy_from_slice(&scratch);
            width *= 2;
        }
        row.iter_mut().for_each(|v| *v *= scale);
    });
}

/// Per-channel banded convolution along the channel axis: `out[j] = sum_t
/// w[j, t] * u[j + t - r]` with zero padding, `w: [n, k]`, `k = 2r + 1`.
pub fn band_rows(u: &[f32], n: usize, w: &[f32], k: usize) -> Vec<f32> {
    assert_eq!(w.len(), n * k);
    let r = k / 2;
    let mut y = vec![0f32; u.len()];
    par::rows_mut(&mut y, n, |row, out| {
        let src = &u[row * n..(row + 1) * n];
        for (j, o) in out.iter_mut().enumerate() {
            let lo = r.saturating_sub(j);
            let hi = k.min(n + r - j);
            let taps = &w[j * k + lo..j * k + hi];
            let vals = &src[j + lo - r..j + hi - r];
            *o = taps.iter().zip(vals).map(|(a, b)| a * b).sum();
        }
    });
    y
}

/// Row-wise softmax in place over width `d`.
pub fn softmax_rows(x: &mut [f32], d: usize) {
    par::rows_mut(x, d, |_, row| hostmath::softmax(row));
}
