// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CPU fast paths for the memory-bound NCHW kernels (concat / batchnorm
//! eval / SiLU / upsample). Like [`crate::fast_conv`], these are execution-only
//! optimizations of the corresponding WGSL kernels — same math, validated
//! against a scalar reference — that replace the one-invocation-per-element JIT
//! loop (whose per-element index decode and per-element libm `expf` dominate)
//! with structured loops, bulk `memcpy`, and AVX2 vectorization.

use rayon::prelude::*;

/// `silu`: `out[i] = x[i] / (1 + exp(-x[i]))`, elementwise.
pub fn silu(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    // Parallel chunks; each chunk vectorised (AVX2) or scalar.
    let chunk = (n / (rayon::current_num_threads() * 4)).max(4096);
    out[..n]
        .par_chunks_mut(chunk)
        .zip(x[..n].par_chunks(chunk))
        .for_each(|(o, xi)| {
            #[cfg(target_arch = "x86_64")]
            if crate::fast_conv::avx2_available() {
                unsafe { silu_avx2(xi, o) };
                return;
            }
            for (oo, &v) in o.iter_mut().zip(xi) {
                *oo = v / (1.0 + (-v).exp());
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let one = _mm256_set1_ps(1.0);
    let neg = _mm256_set1_ps(-1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        // sigmoid(v) = 1/(1+exp(-v)); silu = v*sigmoid(v).
        let e = exp256_ps(_mm256_mul_ps(v, neg));
        let den = _mm256_add_ps(one, e);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_div_ps(v, den));
        i += 8;
    }
    for j in i..n {
        let v = *x.get_unchecked(j);
        *out.get_unchecked_mut(j) = v / (1.0 + (-v).exp());
    }
}

/// Apply `out = silu(out*s + b)` in place over a slice (the fused conv epilogue:
/// BatchNorm-eval affine collapsed to `(s,b)` per channel, then SiLU). Scalar
/// fallback; AVX2 variant below.
pub(crate) fn affine_silu_inplace(buf: &mut [f32], s: f32, b: f32) {
    #[cfg(target_arch = "x86_64")]
    if crate::fast_conv::avx2_available() {
        unsafe { affine_silu_avx2(buf, s, b) };
        return;
    }
    for v in buf.iter_mut() {
        let z = *v * s + b;
        *v = z / (1.0 + (-z).exp());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_silu_avx2(buf: &mut [f32], s: f32, b: f32) {
    use std::arch::x86_64::*;
    let n = buf.len();
    let sv = _mm256_set1_ps(s);
    let bv = _mm256_set1_ps(b);
    let one = _mm256_set1_ps(1.0);
    let neg = _mm256_set1_ps(-1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(buf.as_ptr().add(i));
        let z = _mm256_fmadd_ps(v, sv, bv);
        let e = exp256_ps(_mm256_mul_ps(z, neg));
        _mm256_storeu_ps(buf.as_mut_ptr().add(i), _mm256_div_ps(z, _mm256_add_ps(one, e)));
        i += 8;
    }
    for j in i..n {
        let z = *buf.get_unchecked(j) * s + b;
        *buf.get_unchecked_mut(j) = z / (1.0 + (-z).exp());
    }
}

/// Vectorised single-precision `exp` (Cephes minimax, ~1 ULP). x86_64/AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp256_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let hi = _mm256_set1_ps(88.3762626647949);
    let lo = _mm256_set1_ps(-88.3762626647949);
    let log2ef = _mm256_set1_ps(1.44269504088896341);
    let half = _mm256_set1_ps(0.5);
    let ln2hi = _mm256_set1_ps(0.693359375);
    let ln2lo = _mm256_set1_ps(-2.12194440e-4);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    // fx = floor(x*log2ef + 0.5)
    let mut fx = _mm256_fmadd_ps(x, log2ef, half);
    fx = _mm256_floor_ps(fx);
    // r = x - fx*ln2
    let x = _mm256_fnmadd_ps(fx, ln2hi, x);
    let x = _mm256_fnmadd_ps(fx, ln2lo, x);
    let z = _mm256_mul_ps(x, x);
    let p0 = _mm256_set1_ps(1.9875691500e-4);
    let p1 = _mm256_set1_ps(1.3981999507e-3);
    let p2 = _mm256_set1_ps(8.3334519073e-3);
    let p3 = _mm256_set1_ps(4.1665795894e-2);
    let p4 = _mm256_set1_ps(1.6666665459e-1);
    let p5 = _mm256_set1_ps(5.0000001201e-1);
    let mut y = p0;
    y = _mm256_fmadd_ps(y, x, p1);
    y = _mm256_fmadd_ps(y, x, p2);
    y = _mm256_fmadd_ps(y, x, p3);
    y = _mm256_fmadd_ps(y, x, p4);
    y = _mm256_fmadd_ps(y, x, p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    // 2^fx: build float from integer exponent.
    let imm = _mm256_cvtps_epi32(fx);
    let imm = _mm256_add_epi32(imm, _mm256_set1_epi32(0x7f));
    let imm = _mm256_slli_epi32(imm, 23);
    let pow2 = _mm256_castsi256_ps(imm);
    _mm256_mul_ps(y, pow2)
}

/// `bn_eval`: `out = (x-mean[c])/sqrt(var[c]+eps)*gamma[c]+beta[c]` over NCHW.
/// `mv[2c]=mean, mv[2c+1]=var`; `gb[2c]=gamma, gb[2c+1]=beta`; eps=1e-5.
pub fn bn_eval(params: &[u32], x: &[f32], mv: &[f32], gb: &[f32], out: &mut [f32]) {
    let (n, c, h, w) = (params[0] as usize, params[1] as usize, params[2] as usize, params[3] as usize);
    let hw = h * w;
    // Per-channel collapse to an affine: out = x*scale + bias.
    let scale: Vec<f32> = (0..c).map(|ci| gb[2 * ci] / (mv[2 * ci + 1] + 1e-5).sqrt()).collect();
    let bias: Vec<f32> = (0..c).map(|ci| gb[2 * ci + 1] - mv[2 * ci] * scale[ci]).collect();
    // Coarse parallelism: ~threads*4 tasks, each handling many (n,c) planes, so
    // rayon scheduling cost stays negligible vs the per-plane affine.
    let planes = n * c;
    let group = planes.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    out.par_chunks_mut(hw * group).enumerate().for_each(|(gi, chunk)| {
        for (k, o) in chunk.chunks_mut(hw).enumerate() {
            let plane = gi * group + k;
            let ci = plane % c;
            let (s, b) = (scale[ci], bias[ci]);
            let xi = &x[plane * hw..plane * hw + o.len()];
            #[cfg(target_arch = "x86_64")]
            if crate::fast_conv::avx2_available() {
                unsafe { affine_avx2(xi, s, b, o) };
                continue;
            }
            for (oo, &v) in o.iter_mut().zip(xi) {
                *oo = v * s + b;
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_avx2(x: &[f32], s: f32, b: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let sv = _mm256_set1_ps(s);
    let bv = _mm256_set1_ps(b);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(v, sv, bv));
        i += 8;
    }
    for j in i..n {
        *out.get_unchecked_mut(j) = *x.get_unchecked(j) * s + b;
    }
}

/// `concat2`: channel-concat `y[N,Ca+Cb,H,W]` from `a[N,Ca,H,W]`,`b[N,Cb,H,W]`.
/// Each (n) is two contiguous block copies — no per-element index math.
pub fn concat2(params: &[u32], a: &[f32], b: &[f32], y: &mut [f32]) {
    let (n, ca, cb, h, w) =
        (params[0] as usize, params[1] as usize, params[2] as usize, params[3] as usize, params[4] as usize);
    let hw = h * w;
    let ctot = ca + cb;
    // Coarse parallelism: group many planes per task (per-plane tasks would be
    // dominated by rayon scheduling for a plain memcpy).
    let planes = n * ctot;
    let group = planes.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    y.par_chunks_mut(hw * group).enumerate().for_each(|(gi, chunk)| {
        for (k, o) in chunk.chunks_mut(hw).enumerate() {
            let plane = gi * group + k;
            let (nn, cc) = (plane / ctot, plane % ctot);
            let src = if cc < ca { (nn * ca + cc) * hw } else { (nn * cb + (cc - ca)) * hw };
            o.copy_from_slice(&(if cc < ca { a } else { b })[src..src + o.len()]);
        }
    });
}

/// `concat_split`: copy channel range `da[N,Csrc,H,W] = dy[N,Ctot,H,W][c_off..]`.
pub fn concat_split(params: &[u32], dy: &[f32], da: &mut [f32]) {
    let (n, ctot, csrc, c_off, h, w) = (
        params[0] as usize, params[1] as usize, params[2] as usize,
        params[3] as usize, params[4] as usize, params[5] as usize,
    );
    let hw = h * w;
    let planes = n * csrc;
    let group = planes.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    da.par_chunks_mut(hw * group).enumerate().for_each(|(gi, chunk)| {
        for (k, o) in chunk.chunks_mut(hw).enumerate() {
            let plane = gi * group + k;
            let (nn, cc) = (plane / csrc, plane % csrc);
            let src = ((nn * ctot) + (cc + c_off)) * hw;
            o.copy_from_slice(&dy[src..src + o.len()]);
        }
    });
}

/// `upsample2`: nearest-neighbour x2, `y[n,c,ho,wo] = x[n,c,ho/2,wo/2]`.
pub fn upsample2(params: &[u32], x: &[f32], y: &mut [f32]) {
    let (n, c, h, w) = (params[0] as usize, params[1] as usize, params[2] as usize, params[3] as usize);
    let (oh, ow) = (h * 2, w * 2);
    let planes = n * c;
    let group = planes.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    y.par_chunks_mut(oh * ow * group).enumerate().for_each(|(gi, chunk)| {
        for (k, o) in chunk.chunks_mut(oh * ow).enumerate() {
            let nc = gi * group + k; // (n*C + c) flattened
            let xc = &x[nc * h * w..nc * h * w + h * w];
            for ho in 0..oh {
                let hi = ho / 2;
                let orow = &mut o[ho * ow..ho * ow + ow];
                let xrow = &xc[hi * w..hi * w + w];
                for wo in 0..ow {
                    orow[wo] = xrow[wo / 2];
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u32) -> f32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn silu_matches_scalar() {
        let mut s = 1u32;
        let x: Vec<f32> = (0..1000).map(|_| lcg(&mut s) * 8.0).collect();
        let mut o = vec![0.0f32; x.len()];
        silu(&x, &mut o);
        for (i, &v) in x.iter().enumerate() {
            let r = v / (1.0 + (-v).exp());
            assert!((o[i] - r).abs() < 1e-4, "silu {v} -> {} vs {r}", o[i]);
        }
    }

    #[test]
    fn bn_eval_matches_scalar() {
        let (n, c, h, w) = (1, 7, 5, 9);
        let mut s = 2u32;
        let x: Vec<f32> = (0..n * c * h * w).map(|_| lcg(&mut s)).collect();
        let mv: Vec<f32> = (0..2 * c).map(|i| if i % 2 == 1 { lcg(&mut s).abs() + 0.1 } else { lcg(&mut s) }).collect();
        let gb: Vec<f32> = (0..2 * c).map(|_| lcg(&mut s)).collect();
        let mut o = vec![0.0f32; x.len()];
        bn_eval(&[n as u32, c as u32, h as u32, w as u32], &x, &mv, &gb, &mut o);
        let hw = h * w;
        for idx in 0..x.len() {
            let ci = (idx / hw) % c;
            let inv = 1.0 / (mv[2 * ci + 1] + 1e-5).sqrt();
            let r = (x[idx] - mv[2 * ci]) * inv * gb[2 * ci] + gb[2 * ci + 1];
            assert!((o[idx] - r).abs() < 1e-4, "bn {idx}");
        }
    }

    #[test]
    fn concat2_matches_scalar() {
        let (n, ca, cb, h, w) = (2, 3, 5, 4, 6);
        let mut s = 3u32;
        let a: Vec<f32> = (0..n * ca * h * w).map(|_| lcg(&mut s)).collect();
        let b: Vec<f32> = (0..n * cb * h * w).map(|_| lcg(&mut s)).collect();
        let ctot = ca + cb;
        let mut y = vec![0.0f32; n * ctot * h * w];
        concat2(&[n as u32, ca as u32, cb as u32, h as u32, w as u32], &a, &b, &mut y);
        let hw = h * w;
        for idx in 0..y.len() {
            let ww = idx % w;
            let t1 = idx / w;
            let hh = t1 % h;
            let t2 = t1 / h;
            let cc = t2 % ctot;
            let nn = t2 / ctot;
            let exp = if cc < ca {
                a[((nn * ca + cc) * h + hh) * w + ww]
            } else {
                b[((nn * cb + (cc - ca)) * h + hh) * w + ww]
            };
            assert_eq!(y[idx], exp);
            let _ = hw;
        }
    }

    #[test]
    fn upsample2_matches_scalar() {
        let (n, c, h, w) = (1, 3, 4, 5);
        let mut s = 4u32;
        let x: Vec<f32> = (0..n * c * h * w).map(|_| lcg(&mut s)).collect();
        let (oh, ow) = (h * 2, w * 2);
        let mut y = vec![0.0f32; n * c * oh * ow];
        upsample2(&[n as u32, c as u32, h as u32, w as u32], &x, &mut y);
        for idx in 0..y.len() {
            let wo = idx % ow;
            let t1 = idx / ow;
            let ho = t1 % oh;
            let t2 = t1 / oh;
            let cc = t2 % c;
            let nn = t2 / c;
            let exp = x[((nn * c + cc) * h + ho / 2) * w + wo / 2];
            assert_eq!(y[idx], exp);
        }
    }
}
