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

/// `matmul` (`matmul.wgsl`): `C[M,N] = sum_k A[M,K]·B[N,K]` — i.e. `A @ Bᵀ` with
/// K contiguous in both operands. This is the transformer hot path (every q/k/v/o
/// projection, FFN, and head), which otherwise runs as the scalar per-element JIT
/// loop. Threaded over output rows; each row uses AVX2 FMA with 4-column register
/// blocking so each A-row load feeds four B-row dot products.
pub fn matmul_abt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let row = |arow: &[f32], crow: &mut [f32]| {
        #[cfg(target_arch = "x86_64")]
        if crate::fast_conv::avx2_available() {
            unsafe { row_abt_avx2(arow, b, crow, k, n) };
            return;
        }
        row_abt_scalar(arow, b, crow, k, n);
    };
    // Small problems: rayon fan-out costs more than it saves — run inline (still
    // AVX2). Threshold ~ a few hundred K MACs, below which the tiny transformer
    // matmuls (patch/head) were slower threaded than the scalar JIT loop.
    if m * n * k < 262_144 {
        for r in 0..m {
            row(&a[r * k..r * k + k], &mut c[r * n..r * n + n]);
        }
        return;
    }
    let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
    c.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cchunk)| {
        let row0 = ci * rows_per;
        let nrows = cchunk.len() / n;
        for r in 0..nrows {
            row(&a[(row0 + r) * k..(row0 + r) * k + k], &mut cchunk[r * n..r * n + n]);
        }
    });
}

#[allow(dead_code)]
fn row_abt_scalar(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    for (j, cj) in c.iter_mut().enumerate() {
        let brow = &b[j * k..j * k + k];
        *cj = a.iter().zip(brow).map(|(x, y)| x * y).sum();
    }
    let _ = n;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_abt_avx2(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    use std::arch::x86_64::*;
    #[inline]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut j = 0usize;
    while j + 4 <= n {
        let (p0, p1, p2, p3) =
            (bp.add(j * k), bp.add((j + 1) * k), bp.add((j + 2) * k), bp.add((j + 3) * k));
        let (mut a0, mut a1, mut a2, mut a3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut kk = 0usize;
        while kk + 8 <= k {
            let av = _mm256_loadu_ps(ap.add(kk));
            a0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(p0.add(kk)), a0);
            a1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(p1.add(kk)), a1);
            a2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(p2.add(kk)), a2);
            a3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(p3.add(kk)), a3);
            kk += 8;
        }
        let (mut s0, mut s1, mut s2, mut s3) = (hsum(a0), hsum(a1), hsum(a2), hsum(a3));
        while kk < k {
            let av = *ap.add(kk);
            s0 += av * *p0.add(kk);
            s1 += av * *p1.add(kk);
            s2 += av * *p2.add(kk);
            s3 += av * *p3.add(kk);
            kk += 1;
        }
        c[j] = s0;
        c[j + 1] = s1;
        c[j + 2] = s2;
        c[j + 3] = s3;
        j += 4;
    }
    while j < n {
        let p0 = bp.add(j * k);
        let mut acc = _mm256_setzero_ps();
        let mut kk = 0usize;
        while kk + 8 <= k {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(kk)), _mm256_loadu_ps(p0.add(kk)), acc);
            kk += 8;
        }
        let mut s = hsum(acc);
        while kk < k {
            s += *ap.add(kk) * *p0.add(kk);
            kk += 1;
        }
        c[j] = s;
        j += 1;
    }
}

/// `out[i] = x[i] >= 0 ? x[i] : slope*x[i]` (`leaky_relu.wgsl`; slope 0 is ReLU,
/// slope 1 is the aliasing copy some blocks use). Branch-free select
/// auto-vectorizes; ~40 dispatches per ZipDepth frame ran as scalar JIT before.
pub(crate) fn leaky_relu(x: &[f32], out: &mut [f32], slope: f32) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = if v >= 0.0 { v } else { slope * v };
    }
}

/// Apply `out = out*s + b` in place (fused conv epilogue, `act = 0`/identity —
/// e.g. ZipDepth's QARep branches, whose activation comes after the branch sum).
/// The plain FMA auto-vectorizes; no hand-rolled AVX2 needed.
pub(crate) fn affine_inplace(buf: &mut [f32], s: f32, b: f32) {
    for v in buf.iter_mut() {
        *v = *v * s + b;
    }
}

/// Apply `out = max(out*s + b, 0)` in place (fused conv epilogue, `act = 1` —
/// the ReLU nets: ZipDepth). FMA + max auto-vectorize.
pub(crate) fn affine_relu_inplace(buf: &mut [f32], s: f32, b: f32) {
    for v in buf.iter_mut() {
        *v = (*v * s + b).max(0.0);
    }
}

/// Apply `out = sigmoid(out*s + b)` in place (fused conv epilogue, `act = 3` —
/// gate-producing convs). AVX2 via the shared `exp256_ps` when available.
pub(crate) fn affine_sigmoid_inplace(buf: &mut [f32], s: f32, b: f32) {
    #[cfg(target_arch = "x86_64")]
    if crate::fast_conv::avx2_available() {
        unsafe { affine_sigmoid_avx2(buf, s, b) };
        return;
    }
    for v in buf.iter_mut() {
        let z = *v * s + b;
        *v = 1.0 / (1.0 + (-z).exp());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_sigmoid_avx2(buf: &mut [f32], s: f32, b: f32) {
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
        _mm256_storeu_ps(buf.as_mut_ptr().add(i), _mm256_div_ps(one, _mm256_add_ps(one, e)));
        i += 8;
    }
    for j in i..n {
        let z = *buf.get_unchecked(j) * s + b;
        *buf.get_unchecked_mut(j) = 1.0 / (1.0 + (-z).exp());
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
    // 5th word = fused activation selector (0 identity, 1 relu, 2 silu,
    // 3 sigmoid), mirroring bn_eval.wgsl. The dispatch layer pads uniforms to
    // 16 bytes so the word always exists there; a DIRECT caller with the
    // legacy 4-word slice gets the same treatment (absent = 0 = identity).
    let act = params.get(4).copied().unwrap_or(0);
    let hw = h * w;
    // Per-channel collapse to an affine: out = x*scale + bias, then act.
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
            if act == 0 {
                #[cfg(target_arch = "x86_64")]
                if crate::fast_conv::avx2_available() {
                    unsafe { affine_avx2(xi, s, b, o) };
                    continue;
                }
                for (oo, &v) in o.iter_mut().zip(xi) {
                    *oo = v * s + b;
                }
            } else {
                // Reuse the fused-conv epilogues: copy the affine input into
                // place, then the in-place affine+act (AVX2 where it matters).
                o.copy_from_slice(xi);
                match act {
                    1 => affine_relu_inplace(o, s, b),
                    2 => affine_silu_inplace(o, s, b),
                    3 => affine_sigmoid_inplace(o, s, b),
                    _ => affine_inplace(o, s, b),
                }
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
    let (a_n, b_n, per_n) = (ca * hw, cb * hw, ctot * hw);
    // Segmented parallel copy: each (n, source) run is contiguous in both src and
    // dst, so it copies as one bulk memcpy. Coarse flat chunks across threads.
    let total = n * per_n;
    let chunk = total.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    y.par_chunks_mut(chunk).enumerate().for_each(|(ci, out)| {
        let base = ci * chunk;
        let mut o = 0usize;
        while o < out.len() {
            let g = base + o;
            let (nn, local) = (g / per_n, g % per_n);
            if local < a_n {
                let cnt = (a_n - local).min(out.len() - o);
                out[o..o + cnt].copy_from_slice(&a[nn * a_n + local..nn * a_n + local + cnt]);
                o += cnt;
            } else {
                let bl = local - a_n;
                let cnt = (b_n - bl).min(out.len() - o);
                out[o..o + cnt].copy_from_slice(&b[nn * b_n + bl..nn * b_n + bl + cnt]);
                o += cnt;
            }
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
    // Per n, da[n] = dy[n][c_off*hw .. (c_off+csrc)*hw] is one contiguous run.
    let (src_n, dst_n) = (ctot * hw, csrc * hw);
    let total = n * dst_n;
    let chunk = total.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    da.par_chunks_mut(chunk).enumerate().for_each(|(ci, out)| {
        let base = ci * chunk;
        let mut o = 0usize;
        while o < out.len() {
            let g = base + o;
            let (nn, local) = (g / dst_n, g % dst_n);
            let cnt = (dst_n - local).min(out.len() - o);
            let src = nn * src_n + c_off * hw + local;
            out[o..o + cnt].copy_from_slice(&dy[src..src + cnt]);
            o += cnt;
        }
    });
}

/// `chan_place`: write `src[N,Csrc,H,W]` into channels `[c_off, c_off+Csrc)` of
/// `dst[N,Ctot,H,W]`. Per n, one contiguous bulk memcpy (inverse of concat_split).
pub fn chan_place(params: &[u32], src: &[f32], dst: &mut [f32]) {
    let (n, ctot, csrc, c_off, h, w) = (
        params[0] as usize, params[1] as usize, params[2] as usize,
        params[3] as usize, params[4] as usize, params[5] as usize,
    );
    let hw = h * w;
    let (src_n, dst_n) = (csrc * hw, ctot * hw);
    // Each n's source block is a contiguous run; place it at the channel offset.
    // Parallelise over n (and split large copies via coarse flat chunks of src).
    let total = n * src_n;
    let chunk = total.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    // SAFETY: the destination slices written by distinct flat chunks are disjoint
    // (each maps to a distinct (n, channel-range, position) of dst).
    let dptr = SendMutPtr(dst.as_mut_ptr());
    src.par_chunks(chunk).enumerate().for_each(|(ci, sin)| {
        let dptr = dptr;
        let base = ci * chunk;
        let mut o = 0usize;
        while o < sin.len() {
            let g = base + o;
            let (nn, local) = (g / src_n, g % src_n);
            let cnt = (src_n - local).min(sin.len() - o);
            let dst_off = nn * dst_n + c_off * hw + local;
            unsafe {
                std::ptr::copy_nonoverlapping(sin.as_ptr().add(o), dptr.0.add(dst_off), cnt);
            }
            o += cnt;
        }
    });
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

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
    fn matmul_abt_matches_scalar() {
        // sweep shapes incl. non-multiples of 8 (K tail) and 4 (N tail).
        let mut s = 7u32;
        for &(m, k, n) in &[(1, 16, 32), (5, 63, 17), (33, 128, 40), (8, 7, 3), (2, 512, 1024)] {
            let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
            let mut c = vec![0.0f32; m * n];
            matmul_abt(&a, &b, &mut c, m, k, n);
            let mut maxerr = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let r: f32 = (0..k).map(|kk| a[i * k + kk] * b[j * k + kk]).sum();
                    maxerr = maxerr.max((c[i * n + j] - r).abs() / (r.abs() + 1e-3));
                }
            }
            assert!(maxerr < 2e-3, "matmul_abt rel err {maxerr} for ({m},{k},{n})");
        }
    }

    // Perf microbench (run: cargo test -p brain-backend-cpu --release matmul_bench -- --ignored --nocapture)
    #[test]
    #[ignore]
    fn matmul_bench() {
        let (m, k, n) = (512, 512, 1024); // Kronos-scale linear
        let mut s = 3u32;
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let mut c = vec![0.0f32; m * n];
        let iters = 20;
        // AVX2 + threaded
        matmul_abt(&a, &b, &mut c, m, k, n); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            matmul_abt(&a, &b, &mut c, m, k, n);
        }
        let avx = t.elapsed().as_secs_f64() / iters as f64;
        // scalar single-thread reference
        let mut c2 = vec![0.0f32; m * n];
        let t = std::time::Instant::now();
        for r in 0..m {
            row_abt_scalar(&a[r * k..r * k + k], &b, &mut c2[r * n..r * n + n], k, n);
        }
        let scal = t.elapsed().as_secs_f64();
        // scalar + rayon (the true JIT baseline is threaded-scalar)
        let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
        let t = std::time::Instant::now();
        for _ in 0..iters {
            c2.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cc)| {
                let row0 = ci * rows_per;
                for r in 0..cc.len() / n {
                    row_abt_scalar(&a[(row0 + r) * k..(row0 + r) * k + k], &b, &mut cc[r * n..r * n + n], k, n);
                }
            });
        }
        let scalt = t.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        eprintln!(
            "matmul {m}x{k}x{n} ({} threads): AVX2+threads {:.2} ms ({:.1} GFLOP/s) | scalar+threads {:.2} ms ({:.1} GFLOP/s) | scalar-1t {:.2} ms | AVX2-vs-scalar-threaded {:.1}x",
            rayon::current_num_threads(), avx * 1e3, gflops / avx, scalt * 1e3, gflops / scalt, scal * 1e3, scalt / avx
        );
    }

    #[test]
    fn bn_eval_matches_scalar() {
        let (n, c, h, w) = (1, 7, 5, 9);
        let mut s = 2u32;
        let x: Vec<f32> = (0..n * c * h * w).map(|_| lcg(&mut s)).collect();
        let mv: Vec<f32> = (0..2 * c).map(|i| if i % 2 == 1 { lcg(&mut s).abs() + 0.1 } else { lcg(&mut s) }).collect();
        let gb: Vec<f32> = (0..2 * c).map(|_| lcg(&mut s)).collect();
        let hw = h * w;
        let affine = |idx: usize| {
            let ci = (idx / hw) % c;
            let inv = 1.0 / (mv[2 * ci + 1] + 1e-5).sqrt();
            (x[idx] - mv[2 * ci]) * inv * gb[2 * ci] + gb[2 * ci + 1]
        };
        // Legacy 4-word params: the absent act word means identity — the same
        // contract the padded dispatch uniform provides.
        let mut o = vec![0.0f32; x.len()];
        bn_eval(&[n as u32, c as u32, h as u32, w as u32], &x, &mv, &gb, &mut o);
        for idx in 0..x.len() {
            let r = affine(idx);
            assert!((o[idx] - r).abs() < 1e-4, "bn {idx}");
        }
        // All four act codes against the scalar reference.
        for act in 0..4u32 {
            bn_eval(&[n as u32, c as u32, h as u32, w as u32, act], &x, &mv, &gb, &mut o);
            for idx in 0..x.len() {
                let z = affine(idx);
                let r = match act {
                    1 => z.max(0.0),
                    2 => z / (1.0 + (-z).exp()),
                    3 => 1.0 / (1.0 + (-z).exp()),
                    _ => z,
                };
                assert!((o[idx] - r).abs() < 1e-4, "bn act={act} {idx}");
            }
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
    fn chan_place_matches_scalar() {
        let (n, ctot, csrc, c_off, h, w) = (2, 11, 3, 5, 4, 6);
        let mut s = 9u32;
        let src: Vec<f32> = (0..n * csrc * h * w).map(|_| lcg(&mut s)).collect();
        let mut dst = vec![-1.0f32; n * ctot * h * w];
        chan_place(&[n as u32, ctot as u32, csrc as u32, c_off as u32, h as u32, w as u32], &src, &mut dst);
        let hw = h * w;
        for idx in 0..src.len() {
            let cc = (idx / hw) % csrc;
            let nn = idx / (csrc * hw);
            let pos = idx % hw;
            let di = (nn * ctot + (c_off + cc)) * hw + pos;
            assert_eq!(dst[di], src[idx], "chan_place idx {idx}");
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

/// `gn_stats`: two-pass GroupNorm statistics over NCHW (matches
/// `gn_stats.wgsl`: population variance, eps inside the rsqrt). One entry per
/// (n, g): `stats[2k] = mean`, `stats[2k+1] = 1/sqrt(var + eps)`. The group's
/// channels are contiguous, so each reduction is one contiguous slice —
/// parallelized over chunks with rayon and combined (fp32 accumulation in
/// chunk-partials; validated against the scalar JIT within fp32 tolerance
/// like the conv fast paths).
pub fn gn_stats(params: &[u32], x: &[f32], stats: &mut [f32]) {
    use rayon::prelude::*;
    let (n, c, h, w, g) =
        (params[0] as usize, params[1] as usize, params[2] as usize, params[3] as usize, params[4] as usize);
    let eps = f32::from_bits(params[5]);
    let cpg = c / g;
    let m = cpg * h * w;
    // n*g is tiny (2..8): keep groups sequential, parallelize each reduction
    // over a few LARGE chunks (no nested rayon — nesting inside the backend's
    // pool oversubscribes and measured slower than the scalar JIT).
    const CH: usize = 32 * 1024;
    for k in 0..n * g {
        let (ni, gi) = (k / g, k % g);
        let base = (ni * c + gi * cpg) * h * w;
        let sl = &x[base..base + m];
        let mean = if m >= 2 * CH {
            sl.par_chunks(CH).map(|ch| ch.iter().sum::<f32>()).sum::<f32>() / m as f32
        } else {
            sl.iter().sum::<f32>() / m as f32
        };
        let var = if m >= 2 * CH {
            sl.par_chunks(CH)
                .map(|ch| ch.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>())
                .sum::<f32>()
                / m as f32
        } else {
            sl.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / m as f32
        };
        stats[2 * k] = mean;
        stats[2 * k + 1] = 1.0 / (var + eps).sqrt();
    }
}

/// `gn_apply`: `y = gb[c] * (x - mean_k) * rstd_k + gb[C + c]` (matches
/// `gn_apply.wgsl`). Folded to one affine per contiguous channel slice:
/// `y = a_c * x + b_c` with `a_c = gamma_c * rstd`, `b_c = beta_c - a_c*mean`.
pub fn gn_apply(params: &[u32], x: &[f32], stats: &[f32], gb: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;
    let (n, c, h, w, g) =
        (params[0] as usize, params[1] as usize, params[2] as usize, params[3] as usize, params[4] as usize);
    let hw = h * w;
    let cpg = c / g;
    // Coarse row batching: each rayon task handles >= ~32k elements so task
    // overhead never dominates the (memory-bound) affine.
    let rows_per_task = (32 * 1024 / hw).max(1);
    y.par_chunks_mut(hw * rows_per_task).enumerate().for_each(|(t, yo)| {
        let row0 = t * rows_per_task;
        for (r, yrow) in yo.chunks_mut(hw).enumerate() {
            let row = row0 + r;
            let (ni, ci) = (row / c, row % c);
            let k = ni * g + ci / cpg;
            let (mean, rstd) = (stats[2 * k], stats[2 * k + 1]);
            let a = gb[ci] * rstd;
            let b = gb[c + ci] - a * mean;
            let base = row * hw;
            let xs = &x[base..base + hw];
            for (o, &v) in yrow.iter_mut().zip(xs) {
                *o = a * v + b;
            }
        }
    });
    let _ = n;
}

/// `gn_part` (stage 1): per-(group, t) partial (sum, sumsq) over contiguous
/// chunks — matches `gn_part.wgsl`. Parallel over the partial index.
pub fn gn_part(params: &[u32], x: &[f32], part: &mut [f32]) {
    use rayon::prelude::*;
    let (n, c, h, w, g, pp) = (
        params[0] as usize,
        params[1] as usize,
        params[2] as usize,
        params[3] as usize,
        params[4] as usize,
        params[5] as usize,
    );
    let cpg = c / g;
    let m = cpg * h * w;
    let chunk = m.div_ceil(pp);
    part.par_chunks_mut(2).enumerate().take(n * g * pp).for_each(|(idx, o)| {
        let (k, t) = (idx / pp, idx % pp);
        let (ni, gi) = (k / g, k % g);
        let base = (ni * c + gi * cpg) * h * w;
        let lo = (t * chunk).min(m);
        let hi = (lo + chunk).min(m);
        let (mut s, mut s2) = (0.0f32, 0.0f32);
        for &v in &x[base + lo..base + hi] {
            s += v;
            s2 += v * v;
        }
        o[0] = s;
        o[1] = s2;
    });
}

/// `gn_stats2` (stage 2): combine partials into (mean, rstd) — matches
/// `gn_stats2.wgsl` (population variance via E[x^2] - mean^2, clamped at 0).
pub fn gn_stats2(params: &[u32], part: &[f32], stats: &mut [f32]) {
    let (n, c, h, w, g, pp) = (
        params[0] as usize,
        params[1] as usize,
        params[2] as usize,
        params[3] as usize,
        params[4] as usize,
        params[5] as usize,
    );
    let eps = f32::from_bits(params[6]);
    let m = (c / g) * h * w;
    for k in 0..n * g {
        let (mut s, mut s2) = (0.0f32, 0.0f32);
        for t in 0..pp {
            s += part[(k * pp + t) * 2];
            s2 += part[(k * pp + t) * 2 + 1];
        }
        let mean = s / m as f32;
        let va = (s2 / m as f32 - mean * mean).max(0.0);
        stats[2 * k] = mean;
        stats[2 * k + 1] = 1.0 / (va + eps).sqrt();
    }
}

/// dX[m,k] = sum_n dY[m,n] * W[n,k]   (+ accumulate into dx if `acc`).
/// Backward of `out = x·Wᵀ` w.r.t. x — `matmul_dx`/`matmul_dx_reg` on CPU.
/// dY is [M,N] row-major, W is [N,K] row-major, dX is [M,K].
pub fn matmul_dx(dy: &[f32], w: &[f32], dx: &mut [f32], m: usize, k: usize, n: usize, acc: bool) {
    if m == 0 || k == 0 {
        return;
    }
    let row = |dyr: &[f32], dxr: &mut [f32]| {
        if !acc {
            dxr.iter_mut().for_each(|v| *v = 0.0);
        }
        // dxr[kk] += dy[nn] * w[nn*k + kk]; stream W row-major, one dy scalar per n.
        for nn in 0..n {
            let dyv = dyr[nn];
            if dyv == 0.0 {
                continue;
            }
            let wr = &w[nn * k..nn * k + k];
            for (dstv, &wv) in dxr.iter_mut().zip(wr) {
                *dstv += dyv * wv;
            }
        }
    };
    if m * n * k < 262_144 {
        for r in 0..m {
            row(&dy[r * n..r * n + n], &mut dx[r * k..r * k + k]);
        }
        return;
    }
    let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
    dx.par_chunks_mut(rows_per * k).enumerate().for_each(|(ci, chunk)| {
        let row0 = ci * rows_per;
        let nrows = chunk.len() / k;
        for r in 0..nrows {
            row(&dy[(row0 + r) * n..(row0 + r) * n + n], &mut chunk[r * k..r * k + k]);
        }
    });
}

/// dW[n,k] += sum_m dY[m,n] * X[m,k]   (always accumulates).
/// Backward of `out = x·Wᵀ` w.r.t. W — `matmul_dw`/`matmul_dw_reg` on CPU.
/// dY is [M,N] row-major, X is [M,K] row-major, dW is [N,K].
pub fn matmul_dw(dy: &[f32], x: &[f32], dw: &mut [f32], m: usize, k: usize, n: usize) {
    if n == 0 || k == 0 {
        return;
    }
    // Parallelise over N (output rows). Each n reads column n of dY (strided) and
    // all of X; accumulate a [K] row.
    let row = |nn: usize, dwr: &mut [f32]| {
        for mm in 0..m {
            let dyv = dy[mm * n + nn];
            if dyv == 0.0 {
                continue;
            }
            let xr = &x[mm * k..mm * k + k];
            for (dstv, &xv) in dwr.iter_mut().zip(xr) {
                *dstv += dyv * xv;
            }
        }
    };
    if m * n * k < 262_144 {
        for nn in 0..n {
            row(nn, &mut dw[nn * k..nn * k + k]);
        }
        return;
    }
    let rows_per = (n / (rayon::current_num_threads() * 4)).max(1);
    dw.par_chunks_mut(rows_per * k).enumerate().for_each(|(ci, chunk)| {
        let n0 = ci * rows_per;
        let nrows = chunk.len() / k;
        for r in 0..nrows {
            row(n0 + r, &mut chunk[r * k..r * k + k]);
        }
    });
}

// ---------------------------------------------------------------------------
// Cross-attention family (attn_{scores,softmax,apply}_cross.wgsl) — the
// substrate of query-chunked bidirectional attention (`model::block::
// chunked_bidir_fwd`). Per (batch, head) these are small GEMMs over strided
// head slices of fused buffers; the JIT's one-invocation-per-element loops ran
// them at ~1-2 GFLOPS (75% of an encoder forward). Packing each head's slice
// contiguous and reusing [`matmul_abt`] (AVX2+FMA, rayon over rows) is the
// one-implementation route to the tuned GEMM.
// ---------------------------------------------------------------------------

/// `attn_scores_cross`: scores[b,h,i,j] = (q[b,i,h,:]·kv_k[b,j,h,:]) / √hd.
/// params = [bsz, heads, t_dec, t_enc, head_dim, q_stride, kv_stride, q_off, k_off].
#[allow(clippy::too_many_arguments)]
pub fn attn_scores_cross(
    q: &[f32],
    kv: &[f32],
    scores: &mut [f32],
    bsz: usize,
    heads: usize,
    tq: usize,
    tk: usize,
    hd: usize,
    q_stride: usize,
    kv_stride: usize,
    q_off: usize,
    k_off: usize,
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut qh = vec![0f32; tq * hd];
    let mut kh = vec![0f32; tk * hd];
    for b in 0..bsz {
        for h in 0..heads {
            // Pack this head's q (scale folded in) and k slices contiguous.
            for i in 0..tq {
                let src = (b * tq + i) * q_stride + q_off + h * hd;
                for d in 0..hd {
                    qh[i * hd + d] = q[src + d] * scale;
                }
            }
            for j in 0..tk {
                let src = (b * tk + j) * kv_stride + k_off + h * hd;
                kh[j * hd..j * hd + hd].copy_from_slice(&kv[src..src + hd]);
            }
            let out = &mut scores[((b * heads + h) * tq) * tk..((b * heads + h) * tq + tq) * tk];
            matmul_abt(&qh, &kh, out, tq, hd, tk);
        }
    }
}

/// `attn_softmax_cross`: row softmax over the key axis, scores → probs.
/// params = [bsz, heads, t_dec, t_enc].
pub fn attn_softmax_cross(scores: &[f32], probs: &mut [f32], rows: usize, tk: usize) {
    use rayon::prelude::*;
    probs.par_chunks_mut(tk).enumerate().take(rows).for_each(|(r, p)| {
        let s = &scores[r * tk..r * tk + tk];
        let mx = s.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let mut sum = 0f32;
        for (pv, &sv) in p.iter_mut().zip(s) {
            let e = (sv - mx).exp();
            *pv = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
        for pv in p.iter_mut() {
            *pv *= inv;
        }
    });
}

/// `attn_apply_cross`: out[b,i,h,:] = Σ_j probs[b,h,i,j]·kv_v[b,j,h,:], written
/// into the contiguous `[rows, d_model]` context at column h·hd.
/// params = [bsz, heads, t_dec, t_enc, head_dim, kv_stride, v_off, d_model].
#[allow(clippy::too_many_arguments)]
pub fn attn_apply_cross(
    probs: &[f32],
    kv: &[f32],
    out: &mut [f32],
    bsz: usize,
    heads: usize,
    tq: usize,
    tk: usize,
    hd: usize,
    kv_stride: usize,
    v_off: usize,
    d_model: usize,
) {
    let mut vt = vec![0f32; hd * tk]; // v transposed: vt[d, j]
    let mut ctxh = vec![0f32; tq * hd];
    for b in 0..bsz {
        for h in 0..heads {
            for j in 0..tk {
                let src = (b * tk + j) * kv_stride + v_off + h * hd;
                for d in 0..hd {
                    vt[d * tk + j] = kv[src + d];
                }
            }
            let p = &probs[((b * heads + h) * tq) * tk..((b * heads + h) * tq + tq) * tk];
            // ctx[i,d] = Σ_j P[i,j]·V[j,d] = abt(P[tq,tk], Vᵀ[hd,tk]).
            matmul_abt(p, &vt, &mut ctxh, tq, tk, hd);
            for i in 0..tq {
                let dst = (b * tq + i) * d_model + h * hd;
                out[dst..dst + hd].copy_from_slice(&ctxh[i * hd..i * hd + hd]);
            }
        }
    }
}
