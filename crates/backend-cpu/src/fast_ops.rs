// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CPU fast paths for the memory-bound NCHW kernels (concat / batchnorm
//! eval / SiLU / upsample). Like [`crate::fast_conv`], these are execution-only
//! optimizations of the corresponding WGSL kernels - same math, validated
//! against a scalar reference - that replace the one-invocation-per-element JIT
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

/// `matmul` (`matmul.wgsl`): `C[M,N] = sum_k A[M,K]·B[N,K]` - i.e. `A @ Bᵀ` with
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
        if crate::fast_conv::avx512_available() {
            unsafe { row_abt_avx512(arow, b, crow, k, n) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if crate::fast_conv::avx2_available() {
            unsafe { row_abt_avx2(arow, b, crow, k, n) };
            return;
        }
        row_abt_scalar(arow, b, crow, k, n);
    };
    // Small problems: rayon fan-out costs more than it saves - run inline (still
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

/// AVX-512 twin of [`row_abt_avx2`] - same 4-column register-blocked
/// accumulation, same tail handling, `__m512` (16-wide) lanes instead of
/// `__m256` (8-wide) and `_mm512_reduce_add_ps` instead of the hand-rolled
/// `hsum`. Gated behind [`crate::fast_conv::avx512_available`] - see that
/// function's doc comment for why this microkernel is compiled and
/// shape-tested but NOT execution-verified on this development machine (no
/// AVX-512 host available). Deliberately kept structurally identical to
/// `row_abt_avx2` (not "improved" independently) so a future host that CAN
/// exercise it is comparing the same algorithm at a wider vector width, not a
/// second implementation that could silently diverge.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn row_abt_avx512(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    use std::arch::x86_64::*;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut j = 0usize;
    while j + 4 <= n {
        let (p0, p1, p2, p3) =
            (bp.add(j * k), bp.add((j + 1) * k), bp.add((j + 2) * k), bp.add((j + 3) * k));
        let (mut a0, mut a1, mut a2, mut a3) =
            (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
        let mut kk = 0usize;
        while kk + 16 <= k {
            let av = _mm512_loadu_ps(ap.add(kk));
            a0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(p0.add(kk)), a0);
            a1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(p1.add(kk)), a1);
            a2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(p2.add(kk)), a2);
            a3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(p3.add(kk)), a3);
            kk += 16;
        }
        let (mut s0, mut s1, mut s2, mut s3) =
            (_mm512_reduce_add_ps(a0), _mm512_reduce_add_ps(a1), _mm512_reduce_add_ps(a2), _mm512_reduce_add_ps(a3));
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
        let mut acc = _mm512_setzero_ps();
        let mut kk = 0usize;
        while kk + 16 <= k {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(kk)), _mm512_loadu_ps(p0.add(kk)), acc);
            kk += 16;
        }
        let mut s = _mm512_reduce_add_ps(acc);
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

/// Apply `out = out*s + b` in place (fused conv epilogue, `act = 0`/identity -
/// e.g. ZipDepth's QARep branches, whose activation comes after the branch sum).
/// The plain FMA auto-vectorizes; no hand-rolled AVX2 needed.
pub(crate) fn affine_inplace(buf: &mut [f32], s: f32, b: f32) {
    for v in buf.iter_mut() {
        *v = *v * s + b;
    }
}

/// Apply `out = max(out*s + b, 0)` in place (fused conv epilogue, `act = 1` -
/// the ReLU nets: ZipDepth). FMA + max auto-vectorize.
pub(crate) fn affine_relu_inplace(buf: &mut [f32], s: f32, b: f32) {
    for v in buf.iter_mut() {
        *v = (*v * s + b).max(0.0);
    }
}

/// Apply `out = sigmoid(out*s + b)` in place (fused conv epilogue, `act = 3` -
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
    let hi = _mm256_set1_ps(88.376_26);
    let lo = _mm256_set1_ps(-88.376_26);
    // log2(e). The std constant rounds to the SAME f32 as the Cephes literal
    // `1.44269504088896341`, so this is bit-identical and not a retune.
    let log2ef = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let half = _mm256_set1_ps(0.5);
    let ln2hi = _mm256_set1_ps(0.693_359_4);
    let ln2lo = _mm256_set1_ps(-2.121_944_4e-4);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    // fx = floor(x*log2ef + 0.5)
    let mut fx = _mm256_fmadd_ps(x, log2ef, half);
    fx = _mm256_floor_ps(fx);
    // r = x - fx*ln2
    let x = _mm256_fnmadd_ps(fx, ln2hi, x);
    let x = _mm256_fnmadd_ps(fx, ln2lo, x);
    let z = _mm256_mul_ps(x, x);
    let p0 = _mm256_set1_ps(1.987_569_1e-4);
    let p1 = _mm256_set1_ps(1.398_199_9e-3);
    let p2 = _mm256_set1_ps(8.333_452e-3);
    let p3 = _mm256_set1_ps(4.166_579_6e-2);
    let p4 = _mm256_set1_ps(1.666_666_6e-1);
    let p5 = _mm256_set1_ps(5e-1);
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
/// Each (n) is two contiguous block copies - no per-element index math.
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
        // Rebinding the whole `Send` newtype is REQUIRED, not redundant: under
        // Rust 2021 disjoint capture a closure that only touches `dptr.0`
        // captures that raw pointer directly, which is not `Send`. Verified by
        // deletion - it fails with E0277 `*mut f32` cannot be shared between
        // threads safely.
        #[allow(clippy::redundant_locals)]
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

/// `gn_stats`: two-pass GroupNorm statistics over NCHW (matches
/// `gn_stats.wgsl`: population variance, eps inside the rsqrt). One entry per
/// (n, g): `stats[2k] = mean`, `stats[2k+1] = 1/sqrt(var + eps)`. The group's
/// channels are contiguous, so each reduction is one contiguous slice -
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
    // over a few LARGE chunks (no nested rayon - nesting inside the backend's
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
/// chunks - matches `gn_part.wgsl`. Parallel over the partial index.
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

/// `gn_stats2` (stage 2): combine partials into (mean, rstd) - matches
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
/// Backward of `out = x·Wᵀ` w.r.t. x - `matmul_dx`/`matmul_dx_reg` on CPU.
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
/// Backward of `out = x·Wᵀ` w.r.t. W - `matmul_dw`/`matmul_dw_reg` on CPU.
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

/// `dst[i] += scale * src[i]` over `min(dst.len(), src.len())`. The shared
/// SAXPY-style primitive behind every gated/backward accumulation below
/// (`moe_linear_gated_dx`, the GQA backward quartet): each of those kernels'
/// inner loop is "for one index of the reduced axis, scale-and-accumulate a
/// contiguous row" - this is that operation, vectorised once and reused
/// rather than re-derived per call site.
#[inline]
fn axpy(dst: &mut [f32], scale: f32, src: &[f32]) {
    if scale == 0.0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::fast_conv::avx2_available() {
        unsafe { axpy_avx2(dst, scale, src) };
        return;
    }
    let n = dst.len().min(src.len());
    for (d, &s) in dst[..n].iter_mut().zip(&src[..n]) {
        *d += scale * s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(dst: &mut [f32], scale: f32, src: &[f32]) {
    use std::arch::x86_64::*;
    let n = dst.len().min(src.len());
    let sv = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(sv, s, d));
        i += 8;
    }
    for j in i..n {
        *dst.get_unchecked_mut(j) += scale * *src.get_unchecked(j);
    }
}

// ---------------------------------------------------------------------------
// Sparse-MoE gated linear family (moe_linear_gated{,_dx,_dw}.wgsl) - the
// decode loop's dominant cost (measured at 66.9% of DeepSeek-OCR's profiled
// decode time), previously running as the scalar one-invocation-per-element
// JIT loop with zero vectorisation.
// Same contract as `matmul_abt`/`matmul_dx`/`matmul_dw` PLUS a per-row gate
// early-exit: a row whose `gate[row*n_experts+e_idx] <= 0` is genuinely never
// reduced (not computed-then-discarded), exactly mirroring each kernel's own
// WGSL doc comment. Forward reuses the proven `row_abt_{avx512,avx2}`
// microkernels directly (same shape as `matmul_abt`, just row-gated); the two
// backward siblings reuse the shared `axpy` primitive above.
// ---------------------------------------------------------------------------

/// `moe_linear_gated`: `out[row,:] = 0` if `gate[row*n_experts+e_idx] <= 0`,
/// else `out[row,:] = x[row,:] @ Wᵀ` - `moe_linear_gated.wgsl`.
#[allow(clippy::too_many_arguments)]
pub fn moe_linear_gated_fwd(
    x: &[f32],
    w: &[f32],
    gate: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    e_idx: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let row = |r: usize, xrow: &[f32], orow: &mut [f32]| {
        if gate[r * n_experts + e_idx] <= 0.0 {
            orow.iter_mut().for_each(|v| *v = 0.0);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if crate::fast_conv::avx512_available() {
            unsafe { row_abt_avx512(xrow, w, orow, k, n) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if crate::fast_conv::avx2_available() {
            unsafe { row_abt_avx2(xrow, w, orow, k, n) };
            return;
        }
        row_abt_scalar(xrow, w, orow, k, n);
    };
    if m * n * k < 262_144 {
        for r in 0..m {
            row(r, &x[r * k..r * k + k], &mut out[r * n..r * n + n]);
        }
        return;
    }
    let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
    out.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cchunk)| {
        let row0 = ci * rows_per;
        let nrows = cchunk.len() / n;
        for r in 0..nrows {
            row(row0 + r, &x[(row0 + r) * k..(row0 + r) * k + k], &mut cchunk[r * n..r * n + n]);
        }
    });
}

/// `moe_linear_gated_dx`: `dX[row,:] = sum_n dY[row,n]*W[n,:]` when
/// `gate[row*n_experts+e_idx] > 0`, else left untouched if `acc` else zeroed -
/// `moe_linear_gated_dx.wgsl`. A non-routed row's `dY` is already exactly
/// zero end to end (see the WGSL kernel's own doc), so skipping its
/// reduction changes nothing about the sum; it only removes FLOPs.
#[allow(clippy::too_many_arguments)]
pub fn moe_linear_gated_dx(
    dy: &[f32],
    w: &[f32],
    gate: &[f32],
    dx: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    e_idx: usize,
    acc: bool,
) {
    if m == 0 || k == 0 {
        return;
    }
    let row = |r: usize, dyr: &[f32], dxr: &mut [f32]| {
        if gate[r * n_experts + e_idx] <= 0.0 {
            if !acc {
                dxr.iter_mut().for_each(|v| *v = 0.0);
            }
            return;
        }
        if !acc {
            dxr.iter_mut().for_each(|v| *v = 0.0);
        }
        for nn in 0..n {
            let dyv = dyr[nn];
            if dyv == 0.0 {
                continue;
            }
            axpy(dxr, dyv, &w[nn * k..nn * k + k]);
        }
    };
    if m * n * k < 262_144 {
        for r in 0..m {
            row(r, &dy[r * n..r * n + n], &mut dx[r * k..r * k + k]);
        }
        return;
    }
    let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
    dx.par_chunks_mut(rows_per * k).enumerate().for_each(|(ci, chunk)| {
        let row0 = ci * rows_per;
        let nrows = chunk.len() / k;
        for r in 0..nrows {
            row(row0 + r, &dy[(row0 + r) * n..(row0 + r) * n + n], &mut chunk[r * k..r * k + k]);
        }
    });
}

/// `moe_linear_gated_dw`: `dW[n,:] += sum_{row routed} dY[row,n]*X[row,:]` -
/// `moe_linear_gated_dw.wgsl`. UNLIKE `moe_linear_gated_dx`, the gated axis
/// here is the summed one (every output element still visits every OTHER
/// routed row), so a non-routed row is a loop `continue`, not a whole-row exit.
#[allow(clippy::too_many_arguments)]
pub fn moe_linear_gated_dw(
    dy: &[f32],
    x: &[f32],
    gate: &[f32],
    dw: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    e_idx: usize,
) {
    if n == 0 || k == 0 {
        return;
    }
    let row = |nn: usize, dwr: &mut [f32]| {
        for mm in 0..m {
            if gate[mm * n_experts + e_idx] <= 0.0 {
                continue;
            }
            let dyv = dy[mm * n + nn];
            if dyv == 0.0 {
                continue;
            }
            axpy(dwr, dyv, &x[mm * k..mm * k + k]);
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
// Self-attention family (gqa_scores.wgsl / attn_softmax.wgsl / gqa_apply.wgsl
// + the gqa_bwd_{dscores,dv,dq,dk}.wgsl backward quartet) - plain causal
// grouped-query (MHA is the `n_kv_heads == n_heads` special case) self-
// attention, used by every decoder's own attention (`gpt`, `qwen3`, `glm`,
// `deepseekv2`, ...) and SAM/CLIP's windowed/global attention. Only the
// cross-attention twin (`attn_{scores,softmax,apply}_cross`) had a native
// path before this; this is the same GEMM-packing idea applied to the
// causal-masked, grouped-head shape. `q`/`ctx` are `[B*T, n_heads*head_dim]`;
// `k`/`v` are `[B*T, n_kv_heads*head_dim]`; `scores`/`probs`/`d_scores` are
// `[B*n_heads*T*T]` - all contiguous, no stride/offset params (unlike the
// cross family, which serves a chunked/fused-buffer caller).
// ---------------------------------------------------------------------------

/// `gqa_scores`: `scores[b,h,i,j] = (q[b,i,h,:]·k[b,j,hkv,:])/√hd` for `j<=i`,
/// else `-inf`. Sequential over `(b,h)` (matching `attn_scores_cross`'s own
/// shape) - each head's GEMM already saturates all cores via `matmul_abt`'s
/// internal threading once `T` is large enough to matter (prefill-scale `T`).
#[allow(clippy::too_many_arguments)]
pub fn gqa_scores(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    bsz: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    hd: usize,
    group: usize,
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let q_row = n_heads * hd;
    let k_row = n_kv_heads * hd;
    let mut qh = vec![0f32; t * hd];
    let mut kh = vec![0f32; t * hd];
    for b in 0..bsz {
        for h in 0..n_heads {
            let hkv = h / group;
            for i in 0..t {
                let src = (b * t + i) * q_row + h * hd;
                for d in 0..hd {
                    qh[i * hd + d] = q[src + d] * scale;
                }
            }
            for j in 0..t {
                let src = (b * t + j) * k_row + hkv * hd;
                kh[j * hd..j * hd + hd].copy_from_slice(&k[src..src + hd]);
            }
            let base = (b * n_heads + h) * t * t;
            let out = &mut scores[base..base + t * t];
            matmul_abt(&qh, &kh, out, t, hd, t);
            for i in 0..t {
                for oj in out[i * t + i + 1..i * t + t].iter_mut() {
                    *oj = -3.4e38;
                }
            }
        }
    }
}

/// `attn_softmax`: row-wise causal softmax over the key axis (also serves
/// dense MHA - the kernel doesn't distinguish, see `attn_softmax.wgsl`'s
/// header). Parallel over rows, matching `attn_softmax_cross`'s own shape.
pub fn attn_softmax_causal(scores: &[f32], probs: &mut [f32], bsz: usize, n_heads: usize, t: usize) {
    let rows = bsz * n_heads * t;
    probs.par_chunks_mut(t).zip(scores.par_chunks(t)).take(rows).enumerate().for_each(|(r, (p, s))| {
        let i = r % t;
        let valid = &s[..=i];
        let mx = valid.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let mut sum = 0f32;
        for (pv, &sv) in p[..=i].iter_mut().zip(valid) {
            let e = (sv - mx).exp();
            *pv = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
        for pv in p[..=i].iter_mut() {
            *pv *= inv;
        }
        for pv in p[i + 1..].iter_mut() {
            *pv = 0.0;
        }
    });
}

/// `gqa_apply`: `ctx[b,i,h,:] = Σ_{j<=i} probs[b,h,i,j]·v[b,j,hkv,:]`.
/// Sequential over `(b,h)`, packing `v`'s head slice transposed and reusing
/// `matmul_abt` - the causal zeros `attn_softmax_causal` already wrote into
/// `probs` for `j>i` make the "sum over `j<=i`" and "sum over all `j`" the
/// same computation, so a plain dense GEMM is exact (matching
/// `attn_apply_cross`'s own reasoning for its unmasked case).
#[allow(clippy::too_many_arguments)]
pub fn gqa_apply(
    probs: &[f32],
    v: &[f32],
    ctx: &mut [f32],
    bsz: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    hd: usize,
    group: usize,
) {
    let q_row = n_heads * hd;
    let k_row = n_kv_heads * hd;
    let mut vt = vec![0f32; hd * t]; // vt[d, j]
    let mut ctxh = vec![0f32; t * hd];
    for b in 0..bsz {
        for h in 0..n_heads {
            let hkv = h / group;
            for j in 0..t {
                let src = (b * t + j) * k_row + hkv * hd;
                for d in 0..hd {
                    vt[d * t + j] = v[src + d];
                }
            }
            let base = (b * n_heads + h) * t * t;
            let p = &probs[base..base + t * t];
            matmul_abt(p, &vt, &mut ctxh, t, t, hd);
            for i in 0..t {
                let dst = (b * t + i) * q_row + h * hd;
                ctx[dst..dst + hd].copy_from_slice(&ctxh[i * hd..i * hd + hd]);
            }
        }
    }
}

/// `gqa_bwd_dscores`: gradient through `probs@v` and the softmax jacobian.
/// `DProb[i,j] = Σ_d d_ctx[i,d]·v[j,d]` is exactly the same shape as
/// `gqa_scores`'s own q·k GEMM (with `d_ctx` standing in for `q`, `v` for
/// `k`, no scale, no mask), so it reuses `matmul_abt` the same way; the
/// causal masking falls out for free because `probs[i,j]==0` for `j>i`
/// already zeroes `d_scores[i,j] = probs[i,j]*(DProb[i,j]-dot[i])` there -
/// `dot[i] = Σ_j probs[i,j]*DProb[i,j]` over ALL `j` equals the causal-only
/// sum for the same reason.
#[allow(clippy::too_many_arguments)]
pub fn gqa_bwd_dscores(
    d_ctx: &[f32],
    v: &[f32],
    probs: &[f32],
    d_scores: &mut [f32],
    bsz: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    hd: usize,
    group: usize,
) {
    let q_row = n_heads * hd;
    let k_row = n_kv_heads * hd;
    let mut ctxh = vec![0f32; t * hd];
    let mut vh = vec![0f32; t * hd];
    let mut dprob = vec![0f32; t * t];
    for b in 0..bsz {
        for h in 0..n_heads {
            let hkv = h / group;
            for i in 0..t {
                let src = (b * t + i) * q_row + h * hd;
                ctxh[i * hd..i * hd + hd].copy_from_slice(&d_ctx[src..src + hd]);
            }
            for j in 0..t {
                let src = (b * t + j) * k_row + hkv * hd;
                vh[j * hd..j * hd + hd].copy_from_slice(&v[src..src + hd]);
            }
            matmul_abt(&ctxh, &vh, &mut dprob, t, hd, t);
            let base = (b * n_heads + h) * t * t;
            let p = &probs[base..base + t * t];
            let out = &mut d_scores[base..base + t * t];
            for i in 0..t {
                let prow = &p[i * t..i * t + t];
                let dprow = &dprob[i * t..i * t + t];
                let dot: f32 = prow.iter().zip(dprow).map(|(a, b)| a * b).sum();
                let orow = &mut out[i * t..i * t + t];
                for jj in 0..t {
                    orow[jj] = prow[jj] * (dprow[jj] - dot);
                }
            }
        }
    }
}

/// `gqa_bwd_dv`: `d_v[b,hkv,j,:] = Σ_{h∈group(hkv)} Σ_{i>=j} probs[b,h,i,j]·d_ctx[b,i,h,:]`.
/// Threaded over `d_v`'s own `[B*T, n_kv_heads*head_dim]` rows (the same
/// `par_chunks_mut`-over-output-rows shape `matmul_dx`/`matmul_dw` already
/// use), each row accumulated via the shared `axpy` primitive.
#[allow(clippy::too_many_arguments)]
pub fn gqa_bwd_dv(
    probs: &[f32],
    d_ctx: &[f32],
    d_v: &mut [f32],
    bsz: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    hd: usize,
    group: usize,
) {
    let q_row = n_heads * hd;
    let k_row = n_kv_heads * hd;
    d_v.par_chunks_mut(k_row).take(bsz * t).enumerate().for_each(|(bj, row)| {
        let b = bj / t;
        let j = bj % t;
        for hkv in 0..n_kv_heads {
            let out = &mut row[hkv * hd..hkv * hd + hd];
            out.iter_mut().for_each(|v| *v = 0.0);
            for gi in 0..group {
                let h = hkv * group + gi;
                let p_base = (b * n_heads + h) * t * t;
                for i in j..t {
                    let scale = probs[p_base + i * t + j];
                    if scale == 0.0 {
                        continue;
                    }
                    let src = (b * t + i) * q_row + h * hd;
                    axpy(out, scale, &d_ctx[src..src + hd]);
                }
            }
        }
    });
}

/// `gqa_bwd_dq`: `d_q[b,i,h,:] = scale·Σ_{j<=i} d_scores[b,h,i,j]·k[b,j,hkv,:]`.
/// Threaded over `d_q`'s `[B*T, n_heads*head_dim]` rows.
#[allow(clippy::too_many_arguments)]
pub fn gqa_bwd_dq(
    d_scores: &[f32],
    k: &[f32],
    d_q: &mut [f32],
    bsz: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    hd: usize,
    group: usize,
) {
    let scale0 = 1.0 / (hd as f32).sqrt();
    let q_row = n_heads * hd;
    let k_row = n_kv_heads * hd;
    d_q.par_chunks_mut(q_row).take(bsz * t).enumerate().for_each(|(bi, row)| {
        let b = bi / t;
        let i = bi % t;
        for h in 0..n_heads {
            let hkv = h / group;
            let out = &mut row[h * hd..h * hd + hd];
            out.iter_mut().for_each(|v| *v = 0.0);
            let s_base = (b * n_heads + h) * t * t + i * t;
            for j in 0..=i {
                let ds = d_scores[s_base + j];
                if ds == 0.0 {
                    continue;
                }
                let src = (b * t + j) * k_row + hkv * hd;
                axpy(out, ds * scale0, &k[src..src + hd]);
            }
        }
    });
}

/// `gqa_bwd_dk`: `d_k[b,j,hkv,:] = scale·Σ_{h∈group(hkv)} Σ_{i>=j} d_scores[b,h,i,j]·q[b,i,h,:]`.
/// Threaded over `d_k`'s `[B*T, n_kv_heads*head_dim]` rows, mirroring
/// `gqa_bwd_dv` exactly with `q` in place of `d_ctx` and the `1/√hd` scale.
#[allow(clippy::too_many_arguments)]
pub fn gqa_bwd_dk(
    d_scores: &[f32],
    q: &[f32],
    d_k: &mut [f32],
    bsz: usize,
    n_heads: usize,
    n_kv_heads: usize,
    t: usize,
    hd: usize,
    group: usize,
) {
    let scale0 = 1.0 / (hd as f32).sqrt();
    let q_row = n_heads * hd;
    let k_row = n_kv_heads * hd;
    d_k.par_chunks_mut(k_row).take(bsz * t).enumerate().for_each(|(bj, row)| {
        let b = bj / t;
        let j = bj % t;
        for hkv in 0..n_kv_heads {
            let out = &mut row[hkv * hd..hkv * hd + hd];
            out.iter_mut().for_each(|v| *v = 0.0);
            for gi in 0..group {
                let h = hkv * group + gi;
                let s_base = (b * n_heads + h) * t * t;
                for i in j..t {
                    let ds = d_scores[s_base + i * t + j];
                    if ds == 0.0 {
                        continue;
                    }
                    let src = (b * t + i) * q_row + h * hd;
                    axpy(out, ds * scale0, &q[src..src + hd]);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Cross-attention family (attn_{scores,softmax,apply}_cross.wgsl) - the
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

    // Scalar-threaded moe_linear_gated reference, matching moe_linear_gated.wgsl's
    // OWN row-gated early exit (not `matmul_abt` post-masked) - the honest
    // apples-to-apples baseline for what the Cranelift-JIT scalar path costs at
    // this shape, mirroring `matmul_bench`'s "scalar+threads" comparator above.
    fn moe_fwd_scalar_threaded(x: &[f32], w: &[f32], gate: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, ne: usize, e: usize) {
        let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
        out.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cchunk)| {
            let row0 = ci * rows_per;
            for r in 0..cchunk.len() / n {
                let row = row0 + r;
                let orow = &mut cchunk[r * n..r * n + n];
                if gate[row * ne + e] <= 0.0 {
                    orow.iter_mut().for_each(|v| *v = 0.0);
                    continue;
                }
                row_abt_scalar(&x[row * k..row * k + k], w, orow, k, n);
            }
        });
    }

    // Perf microbench at DeepSeek-OCR's real decoder shape (12-layer MoE,
    // d_model=1280, moe_ff=896, 64 experts top_k=6 - see
    // `deepseekv2::config::DeepseekV2Config::real`) - the kernel this repo's own
    // `BRAIN_PROFILE` run measured at 66.9% of the whole decode loop. `m=283`
    // is the real prompt-prefill row count that run used.
    //
    // A single-decode-row (`m=1`) variant of this bench was tried and DROPPED:
    // at that shape the whole call is a handful of KFLOPs, small enough that
    // repeated measurements read an impossible ~10-30 TFLOP/s (this
    // workspace's LTO build proving the repeated, near-identical-input call
    // loop-invariant and eliding/hoisting the real work despite `black_box`
    // on both the arguments and a per-iteration input perturbation - neither
    // defeated it). Rather than ship a benchmark number that cannot be
    // trusted, this only measures the shape it CAN measure honestly.
    // (run: cargo test -p brain-backend-cpu --release moe_linear_gated_bench -- --ignored --nocapture)
    #[test]
    #[ignore]
    fn moe_linear_gated_bench() {
        let (m, k, n, ne, e) = (283usize, 1280usize, 896usize, 64usize, 3usize);
        let mut s = 5u32;
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        // top_k=6 of 64 experts routed per row -> ~9.4% of rows live for a
        // given expert, matching the real router's own selection rate.
        let gate: Vec<f32> = (0..m * ne).map(|_| if lcg(&mut s).abs() < 6.0 / 64.0 { 0.3 } else { 0.0 }).collect();
        let mut out = vec![0f32; m * n];
        let iters = 50;
        moe_linear_gated_fwd(&x, &w, &gate, &mut out, m, k, n, ne, e); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            moe_linear_gated_fwd(&x, &w, &gate, &mut out, m, k, n, ne, e);
        }
        let avx = t.elapsed().as_secs_f64() / iters as f64;
        let mut out2 = vec![0f32; m * n];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            moe_fwd_scalar_threaded(&x, &w, &gate, &mut out2, m, k, n, ne, e);
        }
        let scalt = t.elapsed().as_secs_f64() / iters as f64;
        let live_rows = gate.iter().step_by(ne).filter(|&&g| g > 0.0).count().max(1);
        let gflops = 2.0 * live_rows as f64 * k as f64 * n as f64 / 1e9;
        eprintln!(
            "moe_linear_gated m={m} k={k} n={n} ({live_rows} live/{m} rows, {} threads): AVX2 {:.2} ms ({:.1} GFLOP/s) | scalar+threads {:.2} ms ({:.1} GFLOP/s) | speedup {:.2}x",
            rayon::current_num_threads(), avx * 1e3, gflops / avx, scalt * 1e3, gflops / scalt, scalt / avx
        );
    }

    // Perf microbench for the self-attention family at DeepSeek-OCR's real
    // decoder shape (n_heads=n_kv_heads=10, head_dim=128 - plain MHA, group=1)
    // and its real prompt-prefill length (T=283). Scalar-threaded references
    // mirror the WGSL kernels' own one-thread-per-output-element,
    // serial-inner-reduction shape, threaded the same way `matmul_bench`'s
    // "scalar+threads" comparator is (the honest JIT-execution-cost proxy).
    // (run: cargo test -p brain-backend-cpu --release gqa_family_bench -- --ignored --nocapture)
    #[test]
    #[ignore]
    fn gqa_family_bench() {
        let (bsz, n_heads, n_kv_heads, t, hd) = (1usize, 10usize, 10usize, 283usize, 128usize);
        let group = n_heads / n_kv_heads;
        let f = GqaFixture {
            bsz,
            n_heads,
            n_kv_heads,
            t,
            hd,
            group,
            q: (0..bsz * t * n_heads * hd).map(|i| ((i as f32) * 0.0001).sin()).collect(),
            k: (0..bsz * t * n_kv_heads * hd).map(|i| ((i as f32) * 0.0002).sin()).collect(),
            v: (0..bsz * t * n_kv_heads * hd).map(|i| ((i as f32) * 0.0003).sin()).collect(),
        };
        let iters = 20;
        let flops_scores = 2.0 * (bsz * n_heads * t * t * hd) as f64 / 1e9;
        let flops_apply = flops_scores;

        let mut scores = vec![0f32; bsz * n_heads * t * t];
        gqa_scores(&f.q, &f.k, &mut scores, bsz, n_heads, n_kv_heads, t, hd, group); // warm
        let start = std::time::Instant::now();
        for _ in 0..iters {
            gqa_scores(&f.q, &f.k, &mut scores, bsz, n_heads, n_kv_heads, t, hd, group);
        }
        let avx_scores = start.elapsed().as_secs_f64() / iters as f64;
        let mut scores2 = vec![0f32; bsz * n_heads * t * t];
        let start = std::time::Instant::now();
        for _ in 0..iters {
            scores2.copy_from_slice(&scores_scalar(&f));
        }
        let scal_scores = start.elapsed().as_secs_f64() / iters as f64;

        let probs = softmax_scalar(&scores, bsz, n_heads, t);
        let mut ctx = vec![0f32; bsz * t * n_heads * hd];
        gqa_apply(&probs, &f.v, &mut ctx, bsz, n_heads, n_kv_heads, t, hd, group); // warm
        let start = std::time::Instant::now();
        for _ in 0..iters {
            gqa_apply(&probs, &f.v, &mut ctx, bsz, n_heads, n_kv_heads, t, hd, group);
        }
        let avx_apply = start.elapsed().as_secs_f64() / iters as f64;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = apply_scalar(&probs, &f);
        }
        let scal_apply = start.elapsed().as_secs_f64() / iters as f64;

        eprintln!(
            "gqa_scores  bsz={bsz} heads={n_heads} T={t} hd={hd} ({} threads): AVX2 {:.2} ms ({:.1} GFLOP/s) | scalar-1t {:.2} ms | speedup {:.1}x",
            rayon::current_num_threads(), avx_scores * 1e3, flops_scores / avx_scores, scal_scores * 1e3, scal_scores / avx_scores
        );
        eprintln!(
            "gqa_apply   bsz={bsz} heads={n_heads} T={t} hd={hd} ({} threads): AVX2 {:.2} ms ({:.1} GFLOP/s) | scalar-1t {:.2} ms | speedup {:.1}x",
            rayon::current_num_threads(), avx_apply * 1e3, flops_apply / avx_apply, scal_apply * 1e3, scal_apply / avx_apply
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
        // Legacy 4-word params: the absent act word means identity - the same
        // contract the padded dispatch uniform provides.
        let mut o = vec![0.0f32; x.len()];
        bn_eval(&[n as u32, c as u32, h as u32, w as u32], &x, &mv, &gb, &mut o);
        for (idx, &oi) in o.iter().enumerate() {
            let r = affine(idx);
            assert!((oi - r).abs() < 1e-4, "bn {idx}");
        }
        // All four act codes against the scalar reference.
        for act in 0..4u32 {
            bn_eval(&[n as u32, c as u32, h as u32, w as u32, act], &x, &mv, &gb, &mut o);
            for (idx, &oi) in o.iter().enumerate() {
                let z = affine(idx);
                let r = match act {
                    1 => z.max(0.0),
                    2 => z / (1.0 + (-z).exp()),
                    3 => 1.0 / (1.0 + (-z).exp()),
                    _ => z,
                };
                assert!((oi - r).abs() < 1e-4, "bn act={act} {idx}");
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
        for (idx, &yi) in y.iter().enumerate() {
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
            assert_eq!(yi, exp);
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
        for (idx, &si) in src.iter().enumerate() {
            let cc = (idx / hw) % csrc;
            let nn = idx / (csrc * hw);
            let pos = idx % hw;
            let di = (nn * ctot + (c_off + cc)) * hw + pos;
            assert_eq!(dst[di], si, "chan_place idx {idx}");
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
        for (idx, &yi) in y.iter().enumerate() {
            let wo = idx % ow;
            let t1 = idx / ow;
            let ho = t1 % oh;
            let t2 = t1 / oh;
            let cc = t2 % c;
            let nn = t2 / c;
            let exp = x[((nn * c + cc) * h + ho / 2) * w + wo / 2];
            assert_eq!(yi, exp);
        }
    }

    // -----------------------------------------------------------------
    // moe_linear_gated family - scalar references mirror the WGSL kernels'
    // own contract exactly (row-gated early exit / continue), not matmul_abt
    // with a post-hoc mask, so a bug in the gating logic itself would show up.
    // -----------------------------------------------------------------

    fn moe_fwd_scalar(x: &[f32], w: &[f32], gate: &[f32], m: usize, k: usize, n: usize, ne: usize, e: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            if gate[r * ne + e] <= 0.0 {
                continue;
            }
            for c in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += x[r * k + kk] * w[c * k + kk];
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    #[test]
    fn moe_linear_gated_fwd_matches_scalar() {
        let (m, k, n, ne, e) = (13usize, 37usize, 21usize, 4usize, 2usize);
        let mut s = 11u32;
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        // Deterministic mixed gate: every third row routed out.
        let gate: Vec<f32> = (0..m * ne).map(|i| if i % 3 == 0 { 0.0 } else { lcg(&mut s).abs() + 0.01 }).collect();
        let want = moe_fwd_scalar(&x, &w, &gate, m, k, n, ne, e);
        let mut got = vec![-1.0f32; m * n]; // -1 sentinel: a missed gate write would show up as -1, not 0
        moe_linear_gated_fwd(&x, &w, &gate, &mut got, m, k, n, ne, e);
        for i in 0..m * n {
            let r = i / n;
            if gate[r * ne + e] <= 0.0 {
                assert_eq!(got[i], 0.0, "non-routed row {r} elem {i} must be exactly zero");
            } else {
                let rel = (got[i] - want[i]).abs() / (want[i].abs() + 1e-3);
                assert!(rel < 2e-3, "moe_linear_gated_fwd row {r} elem {i}: got {} want {} rel {rel}", got[i], want[i]);
            }
        }
    }

    #[test]
    fn moe_linear_gated_dx_matches_scalar() {
        let (m, k, n, ne, e) = (9usize, 23usize, 15usize, 3usize, 1usize);
        let mut s = 13u32;
        let dy: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let gate: Vec<f32> = (0..m * ne).map(|i| if i % 4 == 0 { 0.0 } else { lcg(&mut s).abs() + 0.01 }).collect();
        for &acc in &[false, true] {
            let seed_dx: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001 - 0.5).collect();
            let mut want = seed_dx.clone();
            for r in 0..m {
                if gate[r * ne + e] <= 0.0 {
                    if !acc {
                        for c in 0..k {
                            want[r * k + c] = 0.0;
                        }
                    }
                    continue;
                }
                if !acc {
                    for c in 0..k {
                        want[r * k + c] = 0.0;
                    }
                }
                for nn in 0..n {
                    let dyv = dy[r * n + nn];
                    for c in 0..k {
                        want[r * k + c] += dyv * w[nn * k + c];
                    }
                }
            }
            let mut got = seed_dx.clone();
            moe_linear_gated_dx(&dy, &w, &gate, &mut got, m, k, n, ne, e, acc);
            for i in 0..m * k {
                let rel = (got[i] - want[i]).abs() / (want[i].abs() + 1e-3);
                assert!(rel < 2e-3, "moe_linear_gated_dx acc={acc} elem {i}: got {} want {} rel {rel}", got[i], want[i]);
            }
        }
    }

    #[test]
    fn moe_linear_gated_dw_matches_scalar() {
        let (m, k, n, ne, e) = (17usize, 11usize, 8usize, 5usize, 3usize);
        let mut s = 17u32;
        let dy: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let gate: Vec<f32> = (0..m * ne).map(|i| if i % 5 == 0 { 0.0 } else { lcg(&mut s).abs() + 0.01 }).collect();
        let seed_dw: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.002 - 0.3).collect();
        let mut want = seed_dw.clone();
        for nn in 0..n {
            for mm in 0..m {
                if gate[mm * ne + e] <= 0.0 {
                    continue;
                }
                let dyv = dy[mm * n + nn];
                for c in 0..k {
                    want[nn * k + c] += dyv * x[mm * k + c];
                }
            }
        }
        let mut got = seed_dw.clone();
        moe_linear_gated_dw(&dy, &x, &gate, &mut got, m, k, n, ne, e);
        for i in 0..n * k {
            let rel = (got[i] - want[i]).abs() / (want[i].abs() + 1e-3);
            assert!(rel < 2e-3, "moe_linear_gated_dw elem {i}: got {} want {} rel {rel}", got[i], want[i]);
        }
    }

    // -----------------------------------------------------------------
    // Self-attention (gqa_scores / attn_softmax / gqa_apply + backward)
    // family - scalar references mirror the WGSL kernels' own formulas
    // exactly (see the .wgsl files' own doc comments).
    // -----------------------------------------------------------------

    struct GqaFixture {
        bsz: usize,
        n_heads: usize,
        n_kv_heads: usize,
        t: usize,
        hd: usize,
        group: usize,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
    }

    fn gqa_fixture(seed: u32) -> GqaFixture {
        let (bsz, n_heads, n_kv_heads, t, hd) = (2usize, 4usize, 2usize, 5usize, 6usize);
        let group = n_heads / n_kv_heads;
        let mut s = seed;
        let q: Vec<f32> = (0..bsz * t * n_heads * hd).map(|_| lcg(&mut s)).collect();
        let k: Vec<f32> = (0..bsz * t * n_kv_heads * hd).map(|_| lcg(&mut s)).collect();
        let v: Vec<f32> = (0..bsz * t * n_kv_heads * hd).map(|_| lcg(&mut s)).collect();
        GqaFixture { bsz, n_heads, n_kv_heads, t, hd, group, q, k, v }
    }

    fn scores_scalar(f: &GqaFixture) -> Vec<f32> {
        let (bsz, nh, nkv, t, hd, group) = (f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        let scale = 1.0 / (hd as f32).sqrt();
        let q_row = nh * hd;
        let k_row = nkv * hd;
        let mut out = vec![0f32; bsz * nh * t * t];
        for b in 0..bsz {
            for h in 0..nh {
                let hkv = h / group;
                for i in 0..t {
                    for j in 0..t {
                        let idx = ((b * nh + h) * t + i) * t + j;
                        if j > i {
                            out[idx] = -3.4e38;
                            continue;
                        }
                        let mut acc = 0f32;
                        for d in 0..hd {
                            acc += f.q[(b * t + i) * q_row + h * hd + d] * f.k[(b * t + j) * k_row + hkv * hd + d];
                        }
                        out[idx] = acc * scale;
                    }
                }
            }
        }
        out
    }

    fn softmax_scalar(scores: &[f32], bsz: usize, nh: usize, t: usize) -> Vec<f32> {
        let mut out = vec![0f32; bsz * nh * t * t];
        for r in 0..bsz * nh * t {
            let i = r % t;
            let base = r * t;
            let mx = scores[base..=base + i].iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let mut sum = 0f32;
            for j in 0..=i {
                let e = (scores[base + j] - mx).exp();
                out[base + j] = e;
                sum += e;
            }
            for j in 0..=i {
                out[base + j] /= sum;
            }
        }
        out
    }

    fn apply_scalar(probs: &[f32], f: &GqaFixture) -> Vec<f32> {
        let (bsz, nh, nkv, t, hd, group) = (f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        let q_row = nh * hd;
        let k_row = nkv * hd;
        let mut ctx = vec![0f32; bsz * t * nh * hd];
        for b in 0..bsz {
            for h in 0..nh {
                let hkv = h / group;
                for i in 0..t {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for j in 0..=i {
                            acc += probs[((b * nh + h) * t + i) * t + j] * f.v[(b * t + j) * k_row + hkv * hd + d];
                        }
                        ctx[(b * t + i) * q_row + h * hd + d] = acc;
                    }
                }
            }
        }
        ctx
    }

    #[test]
    fn gqa_scores_matches_scalar() {
        let f = gqa_fixture(21);
        let want = scores_scalar(&f);
        let mut got = vec![0f32; f.bsz * f.n_heads * f.t * f.t];
        gqa_scores(&f.q, &f.k, &mut got, f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        for i in 0..got.len() {
            if want[i] < -1e30 {
                assert!(got[i] < -1e30, "gqa_scores elem {i} should be masked, got {}", got[i]);
            } else {
                assert!((got[i] - want[i]).abs() < 1e-4, "gqa_scores elem {i}: got {} want {}", got[i], want[i]);
            }
        }
    }

    #[test]
    fn attn_softmax_causal_matches_scalar() {
        let f = gqa_fixture(23);
        let scores = scores_scalar(&f);
        let want = softmax_scalar(&scores, f.bsz, f.n_heads, f.t);
        let mut got = vec![-1f32; scores.len()];
        attn_softmax_causal(&scores, &mut got, f.bsz, f.n_heads, f.t);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).abs() < 1e-5, "attn_softmax_causal elem {i}: got {} want {}", got[i], want[i]);
        }
    }

    #[test]
    fn gqa_apply_matches_scalar() {
        let f = gqa_fixture(29);
        let scores = scores_scalar(&f);
        let probs = softmax_scalar(&scores, f.bsz, f.n_heads, f.t);
        let want = apply_scalar(&probs, &f);
        let mut got = vec![0f32; f.bsz * f.t * f.n_heads * f.hd];
        gqa_apply(&probs, &f.v, &mut got, f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).abs() < 1e-4, "gqa_apply elem {i}: got {} want {}", got[i], want[i]);
        }
    }

    #[test]
    fn gqa_bwd_dscores_matches_scalar() {
        let f = gqa_fixture(31);
        let scores = scores_scalar(&f);
        let probs = softmax_scalar(&scores, f.bsz, f.n_heads, f.t);
        let mut s = 41u32;
        let d_ctx: Vec<f32> = (0..f.bsz * f.t * f.n_heads * f.hd).map(|_| lcg(&mut s)).collect();

        // Scalar reference: matches gqa_bwd_dscores.wgsl's own two-loop formula exactly.
        let (bsz, nh, nkv, t, hd, group) = (f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        let q_row = nh * hd;
        let k_row = nkv * hd;
        let mut want = vec![0f32; bsz * nh * t * t];
        for b in 0..bsz {
            for h in 0..nh {
                let hkv = h / group;
                for i in 0..t {
                    let mut dot = 0f32;
                    for j in 0..=i {
                        let mut dprob = 0f32;
                        for d in 0..hd {
                            dprob += d_ctx[(b * t + i) * q_row + h * hd + d] * f.v[(b * t + j) * k_row + hkv * hd + d];
                        }
                        dot += probs[((b * nh + h) * t + i) * t + j] * dprob;
                    }
                    for j in 0..t {
                        let idx = ((b * nh + h) * t + i) * t + j;
                        if j > i {
                            want[idx] = 0.0;
                            continue;
                        }
                        let mut dprob = 0f32;
                        for d in 0..hd {
                            dprob += d_ctx[(b * t + i) * q_row + h * hd + d] * f.v[(b * t + j) * k_row + hkv * hd + d];
                        }
                        want[idx] = probs[idx] * (dprob - dot);
                    }
                }
            }
        }
        let mut got = vec![0f32; want.len()];
        gqa_bwd_dscores(&d_ctx, &f.v, &probs, &mut got, bsz, nh, nkv, t, hd, group);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).abs() < 1e-4, "gqa_bwd_dscores elem {i}: got {} want {}", got[i], want[i]);
        }
    }

    #[test]
    fn gqa_bwd_dv_matches_scalar() {
        let f = gqa_fixture(37);
        let scores = scores_scalar(&f);
        let probs = softmax_scalar(&scores, f.bsz, f.n_heads, f.t);
        let mut s = 43u32;
        let d_ctx: Vec<f32> = (0..f.bsz * f.t * f.n_heads * f.hd).map(|_| lcg(&mut s)).collect();
        let (bsz, nh, nkv, t, hd, group) = (f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        let q_row = nh * hd;
        let k_row = nkv * hd;
        let mut want = vec![0f32; bsz * t * nkv * hd];
        for b in 0..bsz {
            for hkv in 0..nkv {
                for j in 0..t {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for gi in 0..group {
                            let h = hkv * group + gi;
                            for i in j..t {
                                acc += probs[((b * nh + h) * t + i) * t + j] * d_ctx[(b * t + i) * q_row + h * hd + d];
                            }
                        }
                        want[(b * t + j) * k_row + hkv * hd + d] = acc;
                    }
                }
            }
        }
        let mut got = vec![-1f32; want.len()];
        gqa_bwd_dv(&probs, &d_ctx, &mut got, bsz, nh, nkv, t, hd, group);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).abs() < 1e-4, "gqa_bwd_dv elem {i}: got {} want {}", got[i], want[i]);
        }
    }

    #[test]
    fn gqa_bwd_dq_matches_scalar() {
        let f = gqa_fixture(47);
        let mut s = 53u32;
        let d_scores: Vec<f32> = (0..f.bsz * f.n_heads * f.t * f.t).map(|_| lcg(&mut s)).collect();
        let (bsz, nh, nkv, t, hd, group) = (f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        let scale = 1.0 / (hd as f32).sqrt();
        let q_row = nh * hd;
        let k_row = nkv * hd;
        let mut want = vec![0f32; bsz * t * nh * hd];
        for b in 0..bsz {
            for h in 0..nh {
                let hkv = h / group;
                for i in 0..t {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for j in 0..=i {
                            acc += d_scores[((b * nh + h) * t + i) * t + j] * f.k[(b * t + j) * k_row + hkv * hd + d];
                        }
                        want[(b * t + i) * q_row + h * hd + d] = acc * scale;
                    }
                }
            }
        }
        let mut got = vec![-1f32; want.len()];
        gqa_bwd_dq(&d_scores, &f.k, &mut got, bsz, nh, nkv, t, hd, group);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).abs() < 1e-4, "gqa_bwd_dq elem {i}: got {} want {}", got[i], want[i]);
        }
    }

    #[test]
    fn gqa_bwd_dk_matches_scalar() {
        let f = gqa_fixture(59);
        let mut s = 61u32;
        let d_scores: Vec<f32> = (0..f.bsz * f.n_heads * f.t * f.t).map(|_| lcg(&mut s)).collect();
        let (bsz, nh, nkv, t, hd, group) = (f.bsz, f.n_heads, f.n_kv_heads, f.t, f.hd, f.group);
        let scale = 1.0 / (hd as f32).sqrt();
        let q_row = nh * hd;
        let k_row = nkv * hd;
        let mut want = vec![0f32; bsz * t * nkv * hd];
        for b in 0..bsz {
            for hkv in 0..nkv {
                for j in 0..t {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for gi in 0..group {
                            let h = hkv * group + gi;
                            for i in j..t {
                                acc += d_scores[((b * nh + h) * t + i) * t + j] * f.q[(b * t + i) * q_row + h * hd + d];
                            }
                        }
                        want[(b * t + j) * k_row + hkv * hd + d] = acc * scale;
                    }
                }
            }
        }
        let mut got = vec![-1f32; want.len()];
        gqa_bwd_dk(&d_scores, &f.q, &mut got, bsz, nh, nkv, t, hd, group);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).abs() < 1e-4, "gqa_bwd_dk elem {i}: got {} want {}", got[i], want[i]);
        }
    }

    // AVX-512 tier: gated on `avx512_available()`, exactly like every other
    // fast-path microkernel's own test in this module - EXCEPT that on this
    // development machine (no AVX-512 host, see `fast_conv::avx512_available`'s
    // doc) the gate is always false, so this test can only prove the kernel
    // compiles and is shape-correct when skipped; it explicitly reports that
    // rather than silently passing as if verified.
    #[test]
    fn row_abt_avx512_matches_scalar_when_available() {
        if !crate::fast_conv::avx512_available() {
            eprintln!("row_abt_avx512: AVX-512 not available on this host, skipping execution-verification (compiled-only)");
            return;
        }
        let mut s = 71u32;
        for &(k, n) in &[(16usize, 8usize), (23, 5), (64, 32)] {
            let a: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();
            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
            let mut c = vec![0f32; n];
            unsafe { row_abt_avx512(&a, &b, &mut c, k, n) };
            for j in 0..n {
                let want: f32 = (0..k).map(|kk| a[kk] * b[j * k + kk]).sum();
                assert!((c[j] - want).abs() / (want.abs() + 1e-3) < 2e-3, "row_abt_avx512 ({k},{n}) elem {j}");
            }
        }
    }
}
