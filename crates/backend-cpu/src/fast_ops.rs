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
    // Resolved once, before the chunk fan-out, not re-checked per chunk.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar;
    out[..n]
        .par_chunks_mut(chunk)
        .zip(x[..n].par_chunks(chunk))
        .for_each(|(o, xi)| {
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
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

/// `silu_mul` (`silu_mul.wgsl`): `out[i] = SiLU(a[i]) * b[i]` - the SwiGLU
/// activation core every SwiGLU MLP calls (`gpt`'s GELU sibling aside, this is
/// the one every MoE/dense decoder in this tree uses: `qwen3`, `glm`,
/// `qwen35moe`, `deepseekv2`, `lfm`, ...). Unlike [`silu`] above this reads
/// TWO operands, so it needed its own microkernel rather than reuse. A real
/// DeepSeek-OCR resident-server run (`BRAIN_PROFILE`, quiet machine) found
/// this among the largest single entries in the decoder's own profiled CPU
/// time, behind only the matmul family - tens of thousands
/// of decode-step calls, each a single row (`moe_ff` or `d_model` wide, a few
/// hundred to ~1300 elements), previously falling through to the generic
/// dispatch's rayon-chunked JIT path with no native fast path at all.
pub fn silu_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let chunk = (n / (rayon::current_num_threads() * 4)).max(4096);
    // Resolved once, before the chunk fan-out, not re-checked per chunk.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar;
    out[..n]
        .par_chunks_mut(chunk)
        .zip(a[..n].par_chunks(chunk))
        .zip(b[..n].par_chunks(chunk))
        .for_each(|((o, ai), bi)| {
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                unsafe { silu_mul_avx2(ai, bi, o) };
                return;
            }
            for ((oo, &av), &bv) in o.iter_mut().zip(ai).zip(bi) {
                let s = av / (1.0 + (-av).exp());
                *oo = s * bv;
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len()).min(out.len());
    let one = _mm256_set1_ps(1.0);
    let neg = _mm256_set1_ps(-1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let e = exp256_ps(_mm256_mul_ps(av, neg));
        let den = _mm256_add_ps(one, e);
        let silu = _mm256_div_ps(av, den);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(silu, bv));
        i += 8;
    }
    for j in i..n {
        let av = *a.get_unchecked(j);
        let bv = *b.get_unchecked(j);
        let silu = av / (1.0 + (-av).exp());
        *out.get_unchecked_mut(j) = silu * bv;
    }
}

/// `matmul` (`matmul.wgsl`): `C[M,N] = sum_k A[M,K]·B[N,K]` - i.e. `A @ Bᵀ` with
/// K contiguous in both operands. This is the transformer hot path (every q/k/v/o
/// projection, FFN, and head), which otherwise runs as the scalar per-element JIT
/// loop. Threaded over output rows.
///
/// ## Why the loop nest is column-outer, row-inner
///
/// The obvious nest - "for each output row, dot it against every B row" - is
/// what this function used to do, and it is **DRAM-bandwidth-bound, not
/// compute-bound**: it walks the whole of `B` once per row of `A`, so a
/// thread holding `R` rows moves `R · n · k · 4` bytes. At MiniMax-H3's FFN
/// shape (`k=5376, n=14336`, a 308 MB weight) that measured 32 GFLOP/s
/// against a ~100 GB/s memory system - i.e. it was running at essentially
/// 100% of achievable bandwidth and ~2% of achievable FLOPs, so no amount of
/// extra vector width or extra threads could have helped.
///
/// Inverting the nest - hold a 4-column tile of `B` and sweep ALL of this
/// thread's `A` rows through it - drops that to `n · k · 4` bytes per thread
/// (the tile is reused across every row instead of the row being reused
/// across every tile), and the tile itself (`4·k·4` = 86 KB at the shape
/// above) stays resident in L2 for the whole sweep. `A`'s rows are then the
/// re-read operand, but a thread's row block is small enough to stay in
/// L2/L3, which DRAM is not.
///
/// ## The microkernel
///
/// [`block3x4_abt`] holds a 3-row x 4-column tile: 12 AVX2 accumulators, 3
/// A-vector registers and one B temporary - exactly the 16 `ymm` registers
/// Haswell has, chosen so nothing spills. The 12 independent FMA chains also
/// cover the 5-cycle FMA latency (2 FMAs/cycle needs >=10 in flight), which
/// the old 4-accumulator row kernel did not: it was additionally capped at
/// ~4/5 of issue rate even when its data was already in cache.
///
/// Edges (the `m % 3` rows and `n % 4` columns a whole tile cannot cover) fall
/// back to the row kernel and a narrow column loop respectively, so the
/// contract is unchanged for every shape - this function still WRITES `c`
/// (never accumulates into it) and is bit-identical in structure to the old
/// one for `m < 3` or `n < 4`.
pub fn matmul_abt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    // Resolved once, before `row` is defined, not re-walked per row.
    #[cfg(target_arch = "x86_64")]
    let tier = crate::fast_conv::isa_tier();
    let row = |arow: &[f32], crow: &mut [f32]| {
        #[cfg(target_arch = "x86_64")]
        if tier == crate::fast_conv::IsaTier::Avx512 {
            unsafe { row_abt_avx512(arow, b, crow, k, n) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if tier == crate::fast_conv::IsaTier::Avx2 {
            unsafe { row_abt_avx2(arow, b, crow, k, n) };
            return;
        }
        row_abt_scalar(arow, b, crow, k, n);
    };
    // One thread's share of the output rows, blocked as above. `rows` is the
    // number of A rows starting at `row0`; `cchunk` is their `[rows, n]` slice
    // of C.
    let threads = rayon::current_num_threads();
    // The ORIGINAL chunking, and still the one every row-outer shape gets, so
    // that nothing which did not opt into the new nest changes behaviour at
    // all: same kernel, same chunk size, same schedule.
    let rows_row_outer = (m / (threads * 4)).max(1);
    // WHICH operand should stay resident decides the nest, and it is a
    // property of the shape, not a constant. Column-outer keeps a B tile
    // resident and re-reads this thread's A rows; row-outer (the original)
    // keeps one A row resident and re-reads B.
    //
    // Column-outer only pays when B is BOTH the bigger operand (`n*k` vs one
    // chunk's `rows*k`, so just `n` vs the chunk height) AND too big to
    // survive in cache across a row sweep - because its whole benefit is
    // turning B's per-row re-read into a single stream, and a B that was
    // already being re-read out of L2/L3 had nothing to gain. Below the
    // threshold the extra per-tile bookkeeping (12 horizontal reductions and a
    // strided 3x4 store per 4-column tile) is pure loss, and at a short `k`
    // there are too few FMAs per tile to hide it.
    //
    // Both directions are measured, not assumed - `matmul_shape_bench` runs
    // the two nests interleaved at these exact shapes:
    //   * `adaln_proj` (`m=5376, k=2688, n=2`): B is 21 KB against a 57 MB A,
    //     so column-outer would stream all of A once PER COLUMN - 212 ms
    //     against row-outer's 30 ms.
    //   * per-head attention scores (`m=960, k=128, n=960`): B is 492 KB,
    //     cache-resident, and column-outer measured 18 GFLOP/s against
    //     row-outer's 120.
    //   * the FFN (`m=960, k=5376, n=14336`): B is 308 MB, nothing holds it,
    //     and column-outer measured 225 GFLOP/s against 59.
    const B_RESIDENT_FLOATS: usize = 512 * 1024; // 2 MB, past any per-core cache share
    #[cfg(target_arch = "x86_64")]
    let col_outer = (tier == crate::fast_conv::IsaTier::Avx2 || tier == crate::fast_conv::IsaTier::Avx512) && n >= 4 && n * k > B_RESIDENT_FLOATS && n >= (m / threads).max(1);
    #[cfg(not(target_arch = "x86_64"))]
    let col_outer = false;
    // Column-outer wants the FEWEST, TALLEST chunks it can get - its whole
    // point is amortizing each B tile over as many of a thread's rows as
    // possible, and four chunks per thread would re-stream every tile four
    // times. Floor division, not `div_ceil`, so `m >= threads` still yields at
    // least one chunk per thread (`div_ceil` would leave threads idle: `m=49`
    // over 48 threads is 25 chunks of 2, not 49 of 1).
    let rows_per = if col_outer { (m / threads).max(1) } else { rows_row_outer };
    let block = |row0: usize, rows: usize, cchunk: &mut [f32]| {
        #[cfg(target_arch = "x86_64")]
        if col_outer {
            // AVX-512 hosts run the AVX2 microkernel here rather than a 512-bit
            // twin: this nest is bandwidth-shaped, and widening the vector
            // without a host to verify it on is exactly the untested-second-
            // implementation trap `row_abt_avx512`'s own doc records. The
            // 512-bit row kernel still owns the row tail below.
            unsafe { block_abt_avx2(a, b, cchunk, row0, rows, k, n) };
            return;
        }
        for r in 0..rows {
            row(&a[(row0 + r) * k..(row0 + r) * k + k], &mut cchunk[r * n..r * n + n]);
        }
    };
    // Small problems: rayon fan-out costs more than it saves - run inline (still
    // AVX2). Threshold ~ a few hundred K MACs, below which the tiny transformer
    // matmuls (patch/head) were slower threaded than the scalar JIT loop.
    if m * n * k < 262_144 {
        block(0, m, c);
        return;
    }
    // `m == 1` is a GEMV, and a row split has exactly one non-empty task for
    // it: every projection of every autoregressive decode step used to run on
    // ONE core no matter how many the pool had - on a 2x12-core Xeon E5-2690
    // v3, 11.3 GB/s against a measured 39.0 GB/s parallel DRAM roof, i.e. one
    // core's share of it. [`gemv_cols`] splits the output columns instead,
    // bit-identically; see its doc.
    if m == 1 && n >= 8 {
        gemv_cols(a, b, c, k, n);
        return;
    }
    c.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cchunk)| {
        block(ci * rows_per, cchunk.len() / n, cchunk);
    });
}

/// [`matmul_abt`]'s blocked nest for one thread's row range - see that
/// function's doc for the blocking argument. `cchunk` is `[rows, n]`,
/// corresponding to `a`'s rows `row0 .. row0+rows`.
///
/// # Safety
/// `a` must hold `(row0+rows)*k` floats, `b` `n*k`, `cchunk` `rows*n`, and the
/// host must have AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_abt_avx2(a: &[f32], b: &[f32], cchunk: &mut [f32], row0: usize, rows: usize, k: usize, n: usize) {
    let m3 = rows - rows % 3;
    let n4 = n - n % 4;
    let ap = a.as_ptr().add(row0 * k);
    let bp = b.as_ptr();
    let cp = cchunk.as_mut_ptr();
    // Column-outer so each B tile is loaded once per thread, row-inner so it is
    // reused across every one of this thread's rows while still in L2.
    let mut j = 0usize;
    while j < n4 {
        let mut i = 0usize;
        while i < m3 {
            block3x4_abt(ap.add(i * k), bp.add(j * k), cp.add(i * n + j), k, n);
            i += 3;
        }
        j += 4;
    }
    // Column tail (< 4 columns) for the rows the tile nest covered.
    while j < n {
        let bj = bp.add(j * k);
        for i in 0..m3 {
            *cp.add(i * n + j) = dot_avx2(ap.add(i * k), bj, k);
        }
        j += 1;
    }
    // Row tail (< 3 rows), full width - the original row kernel, unchanged.
    for i in m3..rows {
        let arow = std::slice::from_raw_parts(ap.add(i * k), k);
        let crow = std::slice::from_raw_parts_mut(cp.add(i * n), n);
        row_abt_avx2(arow, b, crow, k, n);
    }
}

/// One 3-row x 4-column output tile of `A·Bᵀ`, accumulated over the full `k`.
///
/// 12 accumulators + 3 A vectors + 1 B temporary = the 16 `ymm` registers
/// AVX2 has, so the inner loop spills nothing; the 12 independent FMA chains
/// keep both FMA ports busy through their 5-cycle latency.
///
/// # Safety
/// `ap` must hold `3*k` floats (rows `k` apart), `bp` `4*k` (rows `k` apart),
/// `cp` must be writable at offsets `{0,n,2n} + {0,1,2,3}`, AVX2+FMA present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn block3x4_abt(ap: *const f32, bp: *const f32, cp: *mut f32, k: usize, n: usize) {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 12];
    let (a0p, a1p, a2p) = (ap, ap.add(k), ap.add(2 * k));
    let (b0p, b1p, b2p, b3p) = (bp, bp.add(k), bp.add(2 * k), bp.add(3 * k));
    let mut kk = 0usize;
    while kk + 8 <= k {
        let av0 = _mm256_loadu_ps(a0p.add(kk));
        let av1 = _mm256_loadu_ps(a1p.add(kk));
        let av2 = _mm256_loadu_ps(a2p.add(kk));
        let bv = _mm256_loadu_ps(b0p.add(kk));
        acc[0] = _mm256_fmadd_ps(av0, bv, acc[0]);
        acc[1] = _mm256_fmadd_ps(av1, bv, acc[1]);
        acc[2] = _mm256_fmadd_ps(av2, bv, acc[2]);
        let bv = _mm256_loadu_ps(b1p.add(kk));
        acc[3] = _mm256_fmadd_ps(av0, bv, acc[3]);
        acc[4] = _mm256_fmadd_ps(av1, bv, acc[4]);
        acc[5] = _mm256_fmadd_ps(av2, bv, acc[5]);
        let bv = _mm256_loadu_ps(b2p.add(kk));
        acc[6] = _mm256_fmadd_ps(av0, bv, acc[6]);
        acc[7] = _mm256_fmadd_ps(av1, bv, acc[7]);
        acc[8] = _mm256_fmadd_ps(av2, bv, acc[8]);
        let bv = _mm256_loadu_ps(b3p.add(kk));
        acc[9] = _mm256_fmadd_ps(av0, bv, acc[9]);
        acc[10] = _mm256_fmadd_ps(av1, bv, acc[10]);
        acc[11] = _mm256_fmadd_ps(av2, bv, acc[11]);
        kk += 8;
    }
    let mut out = [0f32; 12];
    for (o, v) in out.iter_mut().zip(acc) {
        *o = hsum256(v);
    }
    // Scalar k-tail, in the same (column-major within the tile) order.
    while kk < k {
        let (x0, x1, x2) = (*a0p.add(kk), *a1p.add(kk), *a2p.add(kk));
        for (t, bpt) in [b0p, b1p, b2p, b3p].iter().enumerate() {
            let bw = *bpt.add(kk);
            out[t * 3] += x0 * bw;
            out[t * 3 + 1] += x1 * bw;
            out[t * 3 + 2] += x2 * bw;
        }
        kk += 1;
    }
    for t in 0..4 {
        *cp.add(t) = out[t * 3];
        *cp.add(n + t) = out[t * 3 + 1];
        *cp.add(2 * n + t) = out[t * 3 + 2];
    }
}

/// Horizontal sum of an AVX2 vector - shared by [`block3x4_abt`] and
/// [`dot_avx2`], and identical to the `hsum` [`row_abt_avx2`] defines inline.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// `Σ_i x[i]·y[i]` over `k` floats - [`block_abt_avx2`]'s narrow column tail.
///
/// # Safety
/// `x` and `y` must each hold `k` floats and the host must have AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_avx2(x: *const f32, y: *const f32, k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= k {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i)), acc);
        i += 8;
    }
    let mut s = hsum256(acc);
    while i < k {
        s += *x.add(i) * *y.add(i);
        i += 1;
    }
    s
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

/// AVX2 packed-int8 GEMM/GEMV - `kernel-performance.md` M8.11, the first real
/// int8 SIMD path this backend has ever had (`ArchDesc.tier(I8)` was `Absent`
/// through M8.1: "no VNNI fast path yet"; this is that fast path, on plain
/// AVX2 - `_mm256_maddubs_epi16`, not VNNI/AMX, since that is what this box's
/// real Haswell-class core has).
///
/// Reproduces `matmul_i8_gemv.wgsl`/`matmul_i8_dyn.wgsl`'s shared math EXACTLY,
/// not merely within tolerance:
///
///   out[m,n] = sx[m] * Σ_g  dot_g(m,n) * sw[n,g]
///   dot_g(m,n) = Σ_{i=0}^{31} i8(xq[m, 32g+i]) * i8(wq[n, 32g+i])     (INTEGER)
///
/// `xq`/`wq` are `[rows, kg]` `u32` with `kg = K/4` (4 signed int8 lanes per
/// word, `dot4I8Packed`'s own packing); `sw` is `[n, kg/8]` (one scale per
/// 8-word = 32-int8 GROUP, `model::int8::GROUP`); `sx` is `[m]`.
///
/// **Why this is bit-identical to the WGSL reference, not "reassociation
/// tolerance" like `fast_conv`'s conv2d/matmul_abt**: the per-group inner sum
/// is INTEGER (associative/commutative exactly, no rounding whichever order
/// the 32 lanes are summed in - scalar one-word-at-a-time here, one
/// `_mm256_maddubs_epi16`+`_mm256_madd_epi16` reduction under AVX2), and the
/// OUTER fold across groups runs in the identical ascending order both WGSL
/// kernels use (`for g in 0..ng`, matching `matmul_i8_gemv.wgsl`'s `WPG=8`
/// grouping and `matmul_i8_dyn.wgsl`'s `QPG=2` fold point - both fold every 8
/// words, the identical `GROUP=32` int8 the model's own quantizer scales by).
/// So the only thing that can legally differ between the scalar and AVX2
/// paths here is HOW one group's 32-lane integer dot product is computed,
/// never in what order groups are folded into the f32 total - which is why
/// [`tests::avx2_int8_gemm_matches_scalar_reference`] asserts exact equality,
/// not an epsilon.
///
/// **The sign trick** (why `_mm256_maddubs_epi16`, which wants one UNSIGNED
/// and one SIGNED `i8` operand, can compute a SIGNED-times-SIGNED dot at all):
/// `model::int8::quantize`'s own `.clamp(-127.0, 127.0)` (never emits `-128`)
/// means every lane's absolute value fits in `u8`'s `0..=127`, so
/// `dot(a,b) == dot(|a|, sign(a)*b)` never overflows either operand -
/// `_mm256_abs_epi8`/`_mm256_sign_epi8` compute exactly that, the same trick
/// ggml's own AVX2 int8 kernels use for the identical reason.
pub fn matmul_i8_dyn(xq: &[u32], wq: &[u32], sx: &[f32], sw: &[f32], out: &mut [f32], m: usize, kg: usize, n: usize) {
    // Packed u32 words per weight-scale group: GROUP(32 int8) / 4 lanes/word -
    // the same `WPG`/implicit-`QPG=2`-fold-point both WGSL kernels use.
    const WPG: usize = 8;
    if m == 0 || n == 0 || kg == 0 {
        return;
    }
    let ng = kg / WPG;
    // Resolved once, before the per-(m,n) inner loop, not re-probed per
    // group: `Avx512Vnni > Avx2 > Scalar`, mirroring `matmul_abt`'s own
    // `IsaTier` priority one dtype family over. `Int8IsaTier::current()` is
    // ALWAYS `Scalar` or `Avx2` on this repo's real hardware today (no box
    // this campaign has run on has AVX-512 VNNI) - see that function's own
    // doc for the M8.12 capability this wires but cannot execution-verify
    // here.
    let tier = Int8IsaTier::current();
    let row = |mi: usize, orow: &mut [f32]| {
        let xrow = &xq[mi * kg..mi * kg + kg];
        for ni in 0..n {
            let wrow = &wq[ni * kg..ni * kg + kg];
            let swrow = &sw[ni * ng..ni * ng + ng];
            let mut acc_f = 0.0f32;
            for (g, &sw_g) in swrow.iter().enumerate().take(ng) {
                let off = g * WPG;
                let xg = &xrow[off..off + WPG];
                let wg_ = &wrow[off..off + WPG];
                let acc_i = match tier {
                    #[cfg(target_arch = "x86_64")]
                    Int8IsaTier::Avx512Vnni => unsafe { dot32_i8_avx512vnni(xg, wg_) },
                    #[cfg(target_arch = "x86_64")]
                    Int8IsaTier::Avx2 => unsafe { dot32_i8_avx2(xg, wg_) },
                    Int8IsaTier::Scalar => dot_group_scalar(xg, wg_),
                };
                acc_f += acc_i as f32 * sw_g;
            }
            orow[ni] = acc_f * sx[mi];
        }
    };
    if m * n * kg < 65_536 {
        for mi in 0..m {
            row(mi, &mut out[mi * n..mi * n + n]);
        }
        return;
    }
    let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
    out.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, ochunk)| {
        let row0 = ci * rows_per;
        let nrows = ochunk.len() / n;
        for r in 0..nrows {
            row(row0 + r, &mut ochunk[r * n..r * n + n]);
        }
    });
}

/// The ISA tier [`matmul_i8_dyn`] dispatches its per-group dot product at -
/// `Avx512Vnni` (`kernel-performance.md` M8.12) beats `Avx2` (M8.11) beats
/// `Scalar`, the identical priority shape `fast_conv::IsaTier`/`isa_tier()`
/// already established for the f32 GEMM family, kept as its own enum rather
/// than widening that one: this is a DIFFERENT function tree (the int8
/// kernels), gated on a DIFFERENT CPUID bit (VNNI, orthogonal to plain
/// AVX-512F/VL/DQ), and conflating the two would make a device with AVX-512F
/// but no VNNI (or vice versa - both exist in the wild) impossible to
/// express correctly through one shared tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Int8IsaTier {
    Scalar,
    #[allow(dead_code)] // constructed only under target_arch = "x86_64"
    Avx2,
    #[allow(dead_code)] // constructed only under target_arch = "x86_64"
    Avx512Vnni,
}

impl Int8IsaTier {
    /// Resolved once per process via a `OnceLock`, the same convention
    /// `fast_conv::isa_tier()` uses - call ONCE per hot-loop function
    /// (`matmul_i8_dyn` does), never per group.
    fn current() -> Int8IsaTier {
        static TIER: std::sync::OnceLock<Int8IsaTier> = std::sync::OnceLock::new();
        *TIER.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            {
                if crate::fast_conv::avx512_vnni_available() {
                    Int8IsaTier::Avx512Vnni
                } else if crate::fast_conv::avx2_available() {
                    Int8IsaTier::Avx2
                } else {
                    Int8IsaTier::Scalar
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Int8IsaTier::Scalar
            }
        })
    }
}

/// One `dot4I8Packed`-per-word group sum, portable scalar - both the
/// non-x86_64 fallback and [`tests::avx2_int8_gemm_matches_scalar_reference`]'s
/// oracle.
#[inline]
fn dot_group_scalar(a: &[u32], b: &[u32]) -> i32 {
    let mut acc = 0i32;
    for (&aw, &bw) in a.iter().zip(b) {
        for lane in 0..4 {
            let ai = ((aw >> (lane * 8)) & 0xFF) as u8 as i8 as i32;
            let bi = ((bw >> (lane * 8)) & 0xFF) as u8 as i8 as i32;
            acc += ai * bi;
        }
    }
    acc
}

/// AVX2 32-lane (8-word) signed-int8 dot product via the sign trick - see
/// [`matmul_i8_dyn`]'s own doc comment for why this is exact, not
/// approximate. `a`/`b` are exactly 8 `u32` words (32 packed `i8` lanes each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot32_i8_avx2(a: &[u32], b: &[u32]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), 8);
    debug_assert_eq!(b.len(), 8);
    let av = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let bv = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let abs_a = _mm256_abs_epi8(av);
    let signed_b = _mm256_sign_epi8(bv, av);
    let prod16 = _mm256_maddubs_epi16(abs_a, signed_b);
    let prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
    let lo = _mm256_castsi256_si128(prod32);
    let hi = _mm256_extracti128_si256(prod32, 1);
    let s128 = _mm_add_epi32(lo, hi);
    let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
    _mm_cvtsi128_si32(s32)
}

/// AVX-512-VNNI twin of [`dot32_i8_avx2`] - `kernel-performance.md` M8.12.
/// Same 32-lane group, same sign trick (see [`matmul_i8_dyn`]'s own doc for
/// why it is legal), `_mm512_dpbusd_epi32` (`VPDPBUSD`) in place of the AVX2
/// `maddubs`+`madd` two-step: VNNI's whole point is that this dot-product-
/// accumulate is ONE instruction, not two, at the wider 512-bit width.
///
/// **Deliberately loads only the lower 256 bits of each 512-bit register**
/// (`_mm256_zextsi256_si512` zero-extends; the upper 256 bits are zero and
/// contribute nothing to the dot product, so this is CORRECT, just not
/// exploiting the full width) - kept to the identical 32-lane/8-word group
/// [`matmul_i8_dyn`]'s scale-fold boundary already fixes, matching
/// [`dot32_i8_avx2`]'s own signature exactly rather than inventing a
/// 64-lane/two-group variant this milestone has no hardware to validate
/// either shape of. A genuine width-doubling version (batching two
/// weight-scale groups' int8 lanes into one 512-bit dot, folding both
/// separately afterward) is a real follow-up once real AVX-512-VNNI
/// hardware exists to measure it against - see this function's own module
/// doc / the `kernel-performance.md` M8.12 ledger entry for why that is not
/// attempted blind.
///
/// UNVALIDATED ON THIS BOX - see `fast_conv::avx512_vnni_available`'s own
/// honesty note and [`tests::avx512vnni_int8_dot_matches_scalar_reference`]'s
/// `skip_unvalidated_capability` call. Compiled and shape-tested only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot32_i8_avx512vnni(a: &[u32], b: &[u32]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), 8);
    debug_assert_eq!(b.len(), 8);
    let av = _mm512_zextsi256_si512(_mm256_loadu_si256(a.as_ptr() as *const __m256i));
    let bv = _mm512_zextsi256_si512(_mm256_loadu_si256(b.as_ptr() as *const __m256i));
    let abs_a = _mm512_abs_epi8(av);
    // AVX-512 dropped `VPSIGNB` (no `_mm512_sign_epi8`) - reconstruct
    // `sign(a) * b` with a compare-mask + blend instead: `neg_mask` bit `i`
    // set iff `a[i] < 0`, `neg_b = 0 - b` (byte-wise negation), then select
    // per lane. `a[i] == 0` needs no special case: `abs_a[i]` is already 0
    // there, so whatever `signed_b[i]` holds contributes `0 * x = 0` either
    // way - the identical reasoning `dot32_i8_avx2`'s `_mm256_sign_epi8` doc
    // note makes for the AVX2 path.
    let neg_mask = _mm512_movepi8_mask(av);
    let neg_b = _mm512_sub_epi8(_mm512_setzero_si512(), bv);
    let signed_b = _mm512_mask_blend_epi8(neg_mask, bv, neg_b);
    let acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_a, signed_b);
    _mm512_reduce_add_epi32(acc)
}

/// NEON (`SDOT`) twin of [`dot32_i8_avx2`]/[`dot32_i8_avx512vnni`] -
/// `kernel-performance.md` M8.13. Same 32-lane group. NEON's `vdotq_s32`
/// (ARMv8.2-A dot-product) is a NATIVE signed-times-signed 4-lane
/// dot-accumulate - unlike AVX2/AVX-512 (which need the abs/sign trick
/// because `maddubs`/`dpbusd` want one unsigned operand), NEON needs no sign
/// workaround at all here: `vdotq_s32(acc, a, b)` computes
/// `acc[k] += sum_{i=0..3}(a[4k+i] * b[4k+i])` directly over SIGNED `i8`
/// lanes, four output lanes (128 bits / 32 bits) per call, folded over 8
/// groups-of-4 to cover all 32 lanes.
///
/// **UNVALIDATED ANYWHERE IN THIS CAMPAIGN, not merely on this box**: unlike
/// M8.12's AVX-512-VNNI kernel (compiled and shape-checked on this real
/// x86_64 host, just never execution-verified), this function has never
/// been compiled AT ALL in this campaign - there is no aarch64 target
/// installed in this sandbox and none could be added (see
/// `fast_conv::neon_dotprod_available`'s own doc). `#[cfg(target_arch =
/// "aarch64")]` means this entire function does not exist in the binary
/// this campaign's own tests build and run.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot32_i8_neon(a: &[u32], b: &[u32]) -> i32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a.len(), 8);
    debug_assert_eq!(b.len(), 8);
    // Each `u32` word IS 4 packed `i8` lanes (`dot4I8Packed`'s own packing) -
    // reinterpret the 8-word slices as 32 `i8` lanes directly, no bit-twiddling
    // needed (unlike the AVX2/AVX-512 sign trick).
    let ap = a.as_ptr() as *const i8;
    let bp = b.as_ptr() as *const i8;
    let mut acc = vdupq_n_s32(0);
    // 32 lanes = two `int8x16_t` (16-lane) chunks, each folded via two
    // `vdotq_s32` groups-of-4 internally (the intrinsic's own 4-lane stride).
    let a0 = vld1q_s8(ap);
    let b0 = vld1q_s8(bp);
    acc = vdotq_s32(acc, a0, b0);
    let a1 = vld1q_s8(ap.add(16));
    let b1 = vld1q_s8(bp.add(16));
    acc = vdotq_s32(acc, a1, b1);
    vaddvq_s32(acc)
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
    if crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar {
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
    if crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar {
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
    // Resolved once, before the per-plane fan-out, not re-checked per plane.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar;
    out.par_chunks_mut(hw * group).enumerate().for_each(|(gi, chunk)| {
        for (k, o) in chunk.chunks_mut(hw).enumerate() {
            let plane = gi * group + k;
            let ci = plane % c;
            let (s, b) = (scale[ci], bias[ci]);
            let xi = &x[plane * hw..plane * hw + o.len()];
            if act == 0 {
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
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
    if crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_set_avx2(dst: &mut [f32], scale: f32, src: &[f32]) {
    use std::arch::x86_64::*;
    let n = dst.len().min(src.len());
    let sv = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= n {
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(sv, s));
        i += 8;
    }
    for j in i..n {
        *dst.get_unchecked_mut(j) = scale * *src.get_unchecked(j);
    }
}

/// `scale_add` (`scale_add.wgsl`): the MoE combine step for one expert -
///   `accumulate == false`: `acc[t,:] = gate[t,e_idx] * src[t,:]`
///   `accumulate == true`:  `acc[t,:] += gate[t,e_idx] * src[t,:]`
/// (a per-row scalar broadcast-multiply-add over `d_model` columns; `gate` is
/// `[seq_len, n_experts]`). Every routed expert in every MoE layer calls this
/// once per forward (this repo evaluates experts densely - see
/// `moe_linear_gated.wgsl`'s own doc), so a decode step alone dispatches
/// `n_routed_experts * n_moe_layers` calls, each covering ONE row. Same
/// bug/fix shape as [`silu_mul`] above: no native path existed, so all of
/// this ran through the generic rayon-chunked JIT dispatch every call.
#[allow(clippy::too_many_arguments)]
pub fn scale_add(
    gate: &[f32],
    src: &[f32],
    acc: &mut [f32],
    seq_len: usize,
    d_model: usize,
    n_experts: usize,
    e_idx: usize,
    accumulate: bool,
) {
    if seq_len == 0 || d_model == 0 {
        return;
    }
    // Resolved once, before `row` is defined, not re-checked per row.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::fast_conv::isa_tier() != crate::fast_conv::IsaTier::Scalar;
    let row = |t: usize, srow: &[f32], arow: &mut [f32]| {
        let g = gate[t * n_experts + e_idx];
        if accumulate {
            // `axpy`'s own `scale == 0.0` early exit is exactly the WGSL
            // no-op this represents: acc += 0*src leaves acc untouched.
            axpy(arow, g, srow);
        } else {
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                unsafe { scale_set_avx2(arow, g, srow) };
                return;
            }
            for (a, &s) in arow.iter_mut().zip(srow) {
                *a = g * s;
            }
        }
    };
    let total = seq_len * d_model;
    if total < 4096 {
        for t in 0..seq_len {
            row(t, &src[t * d_model..t * d_model + d_model], &mut acc[t * d_model..t * d_model + d_model]);
        }
        return;
    }
    let rows_per = (seq_len / (rayon::current_num_threads() * 4)).max(1);
    acc.par_chunks_mut(rows_per * d_model).enumerate().for_each(|(ci, chunk)| {
        let row0 = ci * rows_per;
        let nrows = chunk.len() / d_model;
        for r in 0..nrows {
            let t = row0 + r;
            row(t, &src[t * d_model..t * d_model + d_model], &mut chunk[r * d_model..r * d_model + d_model]);
        }
    });
}

// ---------------------------------------------------------------------------
// Sparse-MoE gated linear family (moe_linear_gated{,_dx,_dw}.wgsl) - the
// decode loop's dominant cost (measured as the majority of DeepSeek-OCR's
// profiled decode time), previously running as the scalar
// one-invocation-per-element JIT loop with zero vectorisation.
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
    // Resolved once, before `row` is defined, not re-walked per row.
    #[cfg(target_arch = "x86_64")]
    let tier = crate::fast_conv::isa_tier();
    let row = |r: usize, xrow: &[f32], orow: &mut [f32]| {
        if gate[r * n_experts + e_idx] <= 0.0 {
            orow.iter_mut().for_each(|v| *v = 0.0);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if tier == crate::fast_conv::IsaTier::Avx512 {
            unsafe { row_abt_avx512(xrow, w, orow, k, n) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if tier == crate::fast_conv::IsaTier::Avx2 {
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
    // How many rows this expert was actually routed - the work this call has,
    // as opposed to the `m` rows it is shaped over. One strided pass over the
    // gate's own column, cheaper than any of the arithmetic it decides.
    let live = (0..m).filter(|r| gate[r * n_experts + e_idx] > 0.0).count();
    if live == 0 {
        // The common case at inference: with `top_k` of `n_experts` routed,
        // `1 - top_k/n_experts` of every layer's expert dispatches have no
        // work at all and are a `memset` wearing a GEMM's clothes. Splitting
        // that over `m` rayon tasks is how a 58-of-64 no-op came to cost more
        // than the 6 that had work.
        zero_par(out);
        return;
    }
    if live == 1 && n >= 8 {
        // ONE routed row - every decode step's shape, since a single new token
        // picks `top_k` experts and each of those sees exactly one row. A row
        // split has one non-empty task by construction, so the whole GEMV ran
        // on one core; splitting COLUMNS is what gives it the pool. See
        // `matmul_abt`'s `gemv_cols` for the bit-identity argument, which is
        // the same one here - this calls the same row microkernel.
        let r = (0..m).find(|r| gate[r * n_experts + e_idx] > 0.0).expect("live == 1");
        let (head, rest) = out.split_at_mut(r * n);
        let (orow, tail) = rest.split_at_mut(n);
        zero_par(head);
        zero_par(tail);
        gemv_cols(&x[r * k..r * k + k], w, orow, k, n);
        return;
    }
    // Row-parallel, but with the task count set by the WORK (`live`) rather
    // than by `m`: at `m = 283` and 26 routed rows the old
    // `rows_per = m/(threads*4)` floored to 1, so a prefill round fanned 283
    // rayon tasks out over the pool to run 26 of them - and paid the wake-up
    // for all 283. Each task still walks its own contiguous row span in
    // order, zeroing the rows that are not routed, so every output element is
    // computed by the same microkernel over the same `k` in the same order as
    // before.
    let rows_per = m.div_ceil(live.min(rayon::current_num_threads()).max(1));
    out.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cchunk)| {
        let row0 = ci * rows_per;
        let nrows = cchunk.len() / n;
        for r in 0..nrows {
            row(row0 + r, &x[(row0 + r) * k..(row0 + r) * k + k], &mut cchunk[r * n..r * n + n]);
        }
    });
}

/// Bytes of streamed operand one rayon task must be given before fanning out
/// pays for itself on this scheduler.
///
/// Waking a sleeping worker is a futex round trip, and a dispatch that hands
/// 48 of them a few kilobytes each spends more time in the kernel than in the
/// arithmetic - measured on a 2x12-core host as 586 s of system time against
/// 131 s of user time across one real DeepSeek-OCR page, almost all of it in
/// per-row fan-out on kernels like this one. At a single core's ~11 GB/s
/// streaming rate, 256 KiB is ~22 us of work - past that cost, and measured
/// as the point where extra parallelism stops paying for extra wake-ups. The
/// sweep that picked it (128 KiB / 256 KiB / 512 KiB / 1 MiB) ran on the REAL
/// served path rather than on a microbenchmark, because the microbenchmark
/// that models a whole prefill round has to hold ~10 GB of distinct weights
/// and its run-to-run spread at that footprint is wider than the effect being
/// tuned. On one real page's prefill: 17.1 s of kernel time at 256 KiB
/// against 18.7 / 19.3 / 20.2 at 128 KiB / 512 KiB / 1 MiB.
const MIN_TASK_BYTES: usize = 256 * 1024;

/// Task count for `bytes` of streamed work: enough to use the pool on
/// something big, exactly one (i.e. no fan-out at all) on something small.
fn task_count(bytes: usize) -> usize {
    (bytes / MIN_TASK_BYTES).clamp(1, rayon::current_num_threads())
}

/// `buf.fill(0.0)`, fanned out only when there is enough of it to be worth a
/// wake-up.
fn zero_par(buf: &mut [f32]) {
    let tasks = task_count(buf.len() * 4);
    if tasks <= 1 {
        buf.fill(0.0);
        return;
    }
    let chunk = buf.len().div_ceil(tasks);
    buf.par_chunks_mut(chunk).for_each(|c| c.fill(0.0));
}

/// `out[j] = Σ_i a[i]·b[j,i]` for ONE row, fanned out over output COLUMNS.
///
/// The skinny-`m` case every resident decoder's steady state is made of: one
/// new token against a weight read once and never revisited, at
/// `2k / 4k` = 0.5 FLOP/byte - bounded by weight traffic, so the only thing
/// that matters is how many cores are pulling it. Splitting the output ROWS
/// cannot do that when there is one row; splitting the output COLUMNS gives
/// each task a contiguous `b[j0*k .. j1*k]` slab, which is the coalesced half
/// of `b` anyway.
///
/// **Bit-identical to the unsplit call.** Every output column is a full
/// `k`-length reduction that touches no other column, and each task runs the
/// same [`row_abt_avx2`]/[`row_abt_avx512`]/[`row_abt_scalar`] microkernel
/// over its own columns; the split changes which core computes a column, never
/// the order in which that column's `k` products are summed. Boundaries are
/// held at multiples of 4 so the microkernel's 4-column register block covers
/// the same column quadruples it would have unsplit, leaving only the original
/// `n % 4` tail to the narrow path.
fn gemv_cols(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(c.len(), n);
    #[cfg(target_arch = "x86_64")]
    let tier = crate::fast_conv::isa_tier();
    let run = |b: &[f32], c: &mut [f32], cols: usize| {
        #[cfg(target_arch = "x86_64")]
        if tier == crate::fast_conv::IsaTier::Avx512 {
            unsafe { row_abt_avx512(a, b, c, k, cols) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if tier == crate::fast_conv::IsaTier::Avx2 {
            unsafe { row_abt_avx2(a, b, c, k, cols) };
            return;
        }
        row_abt_scalar(a, b, c, k, cols);
    };
    let tasks = task_count(n * k * 4);
    if tasks <= 1 {
        run(b, c, n);
        return;
    }
    let cols = n.div_ceil(tasks).div_ceil(4) * 4;
    c.par_chunks_mut(cols).enumerate().for_each(|(t, cc)| {
        let j0 = t * cols;
        run(&b[j0 * k..(j0 + cc.len()) * k], cc, cc.len());
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
// them at a small fraction of what this host's own GEMM reaches, which made
// them the bulk of an encoder forward. Packing each head's slice
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
    scores_packed(q, kv, scores, bsz, heads, tq, tk, hd, q_stride, kv_stride, q_off, k_off, 1.0 / (hd as f32).sqrt());
}

/// `attn_scores_qk`: the SELF-attention twin of [`attn_scores_cross`] - one
/// `[seq, seq]` score block per head from two separate q/k buffers that share
/// one channel stride, with the scale supplied by the caller rather than
/// assumed to be `1/√head_dim`.
///
/// Deliberately not folded into `attn_scores_cross` by passing `q` twice: the
/// kernel's own `scale` uniform is a genuine degree of freedom (its two
/// documented callers use `1/√hd`, but nothing in the contract fixes that),
/// and a hardcoded scale would not crash - it would silently scale every
/// score wrong and still produce plausible-looking output. The CAUSAL variant
/// is not handled here either: the dispatcher leaves `causal != 0` on the JIT
/// path rather than masking after the fact, for the same reason.
#[allow(clippy::too_many_arguments)]
pub fn attn_scores_qk(q: &[f32], k: &[f32], scores: &mut [f32], bsz: usize, heads: usize, seq: usize, hd: usize, qk_stride: usize, scale: f32) {
    scores_packed(q, k, scores, bsz, heads, seq, seq, hd, qk_stride, qk_stride, 0, 0, scale);
}

/// The packing + per-head GEMM both score kernels above share: pack this
/// head's strided `[t, hd]` q and k slices contiguous (folding `scale` into q,
/// so the GEMM itself stays a plain unscaled `A·Bᵀ`) and hand the pair to
/// [`matmul_abt`].
#[allow(clippy::too_many_arguments)]
fn scores_packed(
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
    scale: f32,
) {
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

/// `kv_k_headt`: transpose the K region of a fused KV slab to key-minor,
/// `kt[c*t_enc + j] = kv[j*kv_stride + k_off + c]`.
/// params = [t_enc, d_model, kv_stride, k_off].
///
/// The same shuffle [`attn_apply_cross`] already does to V, so it shares
/// [`transpose_rows_tiled`] and its cache-locality argument verbatim.
pub fn kv_k_headt(kv: &[f32], kt: &mut [f32], t_enc: usize, d_model: usize, kv_stride: usize, k_off: usize) {
    if t_enc == 0 || d_model == 0 {
        return;
    }
    transpose_rows_tiled(kv, kv_stride, k_off, t_enc, d_model, kt);
}

/// `attn_scores_cross_kt`: the same scores as [`attn_scores_cross`], with K
/// already key-minor - `kt` is `[d_model, t_enc]`, head `h` occupying rows
/// `h*hd .. (h+1)*hd`.
/// params = [bsz, heads, t_dec, t_enc, head_dim, q_stride, q_off].
///
/// Head `h`'s slice of `kt` IS `Kᵀ` `[hd, t_enc]` row-major, so where
/// [`attn_scores_cross`] packs K and uses the `A·Bᵀ` GEMM, this one packs only
/// `q` and multiplies straight through with the row-major `A·B` GEMM
/// [`matmul_dx`] (named for its backward role; the arithmetic is a plain
/// product). Undoing the transpose per head to reach [`matmul_abt`] instead
/// measured SLOWER on SAM-1's 4096-key global attention - a second full shuffle
/// costs more than the two GEMMs differ by.
#[allow(clippy::too_many_arguments)]
pub fn attn_scores_cross_kt(
    q: &[f32],
    kt: &[f32],
    scores: &mut [f32],
    bsz: usize,
    heads: usize,
    tq: usize,
    tk: usize,
    hd: usize,
    q_stride: usize,
    q_off: usize,
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut qh = vec![0f32; tq * hd];
    for b in 0..bsz {
        for h in 0..heads {
            for i in 0..tq {
                let src = (b * tq + i) * q_stride + q_off + h * hd;
                for d in 0..hd {
                    qh[i * hd + d] = q[src + d] * scale;
                }
            }
            let mut kh = vec![0f32; tk * hd];
            transpose_rows_tiled(&kt[h * hd * tk..], tk, 0, hd, tk, &mut kh);
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

/// Transpose `[tk, hd]` (row `j` = `src[j*stride+off .. +hd]`) into `vt[d,j]`
/// (`[hd, tk]`, row-major), in `JT`-row tiles.
///
/// The naive one-pass loop (`for j { for d { vt[d*tk+j] = src[..] } }`) reads
/// each source row contiguously but WRITES `hd` elements `tk` floats apart --
/// `hd` separate cache lines touched per `j`, `tk` times, so the write side
/// streams through the whole `hd*tk` destination in a scatter pattern with no
/// reuse. Measured the dominant cost of [`attn_apply_cross`] at SAM-1
/// DeepSeek-OCR's real T=4096 global-attention shape: the majority of the
/// whole tower's CPU forward, at an order of magnitude the per-call cost of
/// `attn_scores_cross` doing the SAME total FLOPs -- the only structural
/// difference is this transpose.
///
/// Tiling by `JT` source rows, buffered THEN written `d`-major, bounds each
/// tile's destination footprint to `hd*JT` floats (4 KiB at `hd=64, JT=16`,
/// one 64-byte cache line per destination row) and - the part that actually
/// matters - writes each of those lines exactly ONCE, fully, instead of
/// revisiting it `JT` times interleaved with `hd-1` OTHER lines in between
/// (the naive `for j { for d { .. } }` order: row `d`'s line for tile-column
/// `j` is written, then `hd-1` unrelated lines are touched before that same
/// line is written again for `j+1`, likely evicted from L1 by then at
/// `hd*JT*4 = 4 KiB` per row times 64 rows). Buffering the tile's `JT` source
/// rows first (a plain contiguous read per row, `hd` floats each) makes the
/// `d`-outer / `j`-inner write order possible without re-reading the source
/// out of order.
///
/// The tile is sized in BYTES and blocked on BOTH axes, because callers do not
/// agree on the row width: `attn_apply_cross` passes one head (64-128 floats),
/// and a caller transposing a fused KV slab passes a whole `d_model` row
/// (768-1536, and 128 is already past the old fixed tile). A tile fixed at some
/// row count would overrun its staging buffer on the wide callers - and a
/// `debug_assert` on the width is no guard, because every path that reaches
/// this runs in release - while a tile derived only from the width degenerates
/// to a single source row there, which is the untiled loop this exists to
/// avoid. Deriving BOTH extents from the byte budget keeps the staged tile in
/// L1 at every caller's shape.
fn transpose_rows_tiled(src: &[f32], stride: usize, off: usize, tk: usize, hd: usize, vt: &mut [f32]) {
    const TILE: usize = 16 * 64; // 4 KiB of f32, the footprint the tiling is chosen for.
    if tk == 0 || hd == 0 {
        return;
    }
    let mut buf = [0f32; TILE];
    let dt = hd.min(64); // columns per tile
    let jt = (TILE / dt).max(1); // source rows per tile
    let mut d0 = 0usize;
    while d0 < hd {
        let dn = (d0 + dt).min(hd);
        let w = dn - d0;
        let mut j0 = 0usize;
        while j0 < tk {
            let jn = (j0 + jt).min(tk);
            for (jj, j) in (j0..jn).enumerate() {
                let at = j * stride + off + d0;
                buf[jj * w..jj * w + w].copy_from_slice(&src[at..at + w]);
            }
            for (dd, d) in (d0..dn).enumerate() {
                let row = &mut vt[d * tk + j0..d * tk + jn];
                for (jj, dst) in row.iter_mut().enumerate() {
                    *dst = buf[jj * w + dd];
                }
            }
            j0 = jn;
        }
        d0 = dn;
    }
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
            let base = b * tk * kv_stride + v_off + h * hd;
            transpose_rows_tiled(&kv[base..], kv_stride, 0, tk, hd, &mut vt);
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

    /// `transpose_rows_tiled` against the naive `vt[d*tk+j] = src[j*stride+off+d]`
    /// definition, over shapes that are and are not multiples of the tile's row
    /// count and that exercise a non-zero `off`/`stride` (a real caller slices
    /// one head out of a fused `[tk, heads*hd]` buffer).
    ///
    /// `hd = 128` is a diffusion transformer's head width and is the case that
    /// indexed a fixed 64-wide staging buffer out of bounds; `hd = 1500` is
    /// wider than the whole tile, the case that falls back to one row at a
    /// time. Both are here because the head width is the caller's, not this
    /// function's, and the previous guard on it was a `debug_assert` in code
    /// that only ever runs in release.
    #[test]
    fn transpose_rows_tiled_matches_the_naive_definition() {
        let mut seed = 3u32;
        for &(tk, hd, stride, off) in &[
            (4096usize, 64usize, 768usize, 128usize),
            (17, 5, 5, 0),
            (1, 3, 3, 0),
            (33, 64, 64, 0),
            (512, 128, 1536, 0),
            (7, 128, 384, 256),
            (3, 1500, 1500, 0),
        ] {
            let src: Vec<f32> = (0..tk * stride).map(|_| lcg(&mut seed)).collect();
            let mut vt = vec![0f32; hd * tk];
            transpose_rows_tiled(&src, stride, off, tk, hd, &mut vt);
            for j in 0..tk {
                for d in 0..hd {
                    let want = src[j * stride + off + d];
                    assert_eq!(vt[d * tk + j], want, "tk={tk} hd={hd} stride={stride} off={off} j={j} d={d}");
                }
            }
        }
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
        // Sweep shapes incl. non-multiples of 8 (K tail), 4 (N tail) and 3 (M
        // tail, the 3x4 register tile's row edge).
        //
        // The last three shapes are all past the 262_144-MAC threading
        // threshold, which is a distinct code path and not merely a bigger
        // one: it splits the rows into per-thread chunks, so a tile-edge bug
        // can hide at every chunk boundary rather than only at the end of the
        // matrix. `(1,16,32)`/`(2,512,1024)` keep the `m < 3` case (row kernel
        // only) covered on both sides of that threshold, and `(8,7,3)` the
        // `n < 4` case (column tail only).
        let mut s = 7u32;
        for &(m, k, n) in &[(1, 16, 32), (5, 63, 17), (33, 128, 40), (8, 7, 3), (2, 512, 1024), (100, 130, 70), (97, 129, 67), (12, 4096, 12)] {
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

    /// The one-row microkernel run over the WHOLE weight, i.e. what
    /// [`gemv_cols`] does with no split at all - the reference a split result
    /// must equal bit for bit, not merely within a tolerance.
    fn gemv_unsplit_ref(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            let tier = crate::fast_conv::isa_tier();
            if tier == crate::fast_conv::IsaTier::Avx512 {
                unsafe { row_abt_avx512(a, b, c, k, n) };
                return;
            }
            if tier == crate::fast_conv::IsaTier::Avx2 {
                unsafe { row_abt_avx2(a, b, c, k, n) };
                return;
            }
        }
        row_abt_scalar(a, b, c, k, n);
    }

    /// **Spec: splitting a decode step's GEMV across the pool changes where a
    /// column is computed, never what it computes.**
    ///
    /// `m = 1` is every autoregressive step's shape and the one the row split
    /// could not parallelise at all. Fanning it out over columns is only a
    /// legitimate scheduling change if it is bit-identical - a tolerance would
    /// let a real reassociation through, and this model's own gates
    /// (`deepseek2/tests/generate.rs` matching llama.cpp token for token,
    /// `deepseek2ocr/tests/real_weight_long_context.rs` matching itself across
    /// two prefill mechanisms) are argmax comparisons that a last-ulp drift can
    /// flip.
    ///
    /// Shapes are the real decoder's four GEMVs plus deliberately awkward ones:
    /// `n` not a multiple of 4 (so the narrow column tail is exercised at the
    /// LAST task rather than the last column of a whole matrix) and `k` not a
    /// multiple of 8 (so the scalar k-tail runs inside every task).
    #[test]
    fn gemv_column_split_is_bit_identical() {
        let mut s = 23u32;
        for &(k, n) in &[(1280usize, 896usize), (1280, 1280), (1280, 1792), (1280, 4099), (129, 3001), (1280, 9)] {
            let a: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();
            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
            let mut got = vec![0.0f32; n];
            let mut want = vec![0.0f32; n];
            matmul_abt(&a, &b, &mut got, 1, k, n);
            gemv_unsplit_ref(&a, &b, &mut want, k, n);
            assert_eq!(got, want, "matmul_abt m=1 k={k} n={n} is not bit-identical to the unsplit microkernel");
        }
    }

    /// **Spec: an expert's routed rows decide its result, not how the rows are
    /// scheduled** - across all three of `moe_linear_gated_fwd`'s regimes
    /// (nothing routed, exactly one row routed, many rows routed), each of
    /// which now takes a different path through the pool.
    ///
    /// Bit-identity again, against the row-at-a-time reference the WGSL kernel
    /// describes: a non-routed row is exactly zero and a routed one is exactly
    /// the same reduction the unsplit microkernel performs.
    #[test]
    fn moe_linear_gated_row_regimes_are_bit_identical() {
        let (k, n, ne) = (1280usize, 896usize, 64usize);
        let mut s = 29u32;
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        // (m, rows routed to expert 3) - none, one, and a real prefill round's
        // share of 283 rows at top_k=6 of 64.
        // A run of routed rows long enough that a task holds several of them
        // exercises the compacted nest's 3-row register tile AND its `% 3` row
        // tail; the sparse cases above leave a task with one row, which only
        // ever reaches the tail.
        let dense: Vec<usize> = (0..131).map(|i| i * 2 + 1).collect();
        for &(m, live_rows) in &[
            (283usize, &[] as &[usize]),
            (283, &[0]),
            (283, &[137]),
            (283, &[282]),
            (1, &[0]),
            (283, &[3, 40, 41, 99, 200, 281]),
            (283, &dense[..]),
            (283, &dense[..7]),
            (283, &dense[..2]),
        ] {
            let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
            let mut gate = vec![0.0f32; m * ne];
            for &r in live_rows {
                gate[r * ne + 3] = 0.5;
            }
            let mut got = vec![7.0f32; m * n];
            moe_linear_gated_fwd(&x, &b, &gate, &mut got, m, k, n, ne, 3);
            let mut want = vec![0.0f32; m * n];
            for &r in live_rows {
                gemv_unsplit_ref(&x[r * k..r * k + k], &b, &mut want[r * n..r * n + n], k, n);
            }
            assert_eq!(got, want, "moe_linear_gated_fwd m={m} with {} routed rows is not bit-identical", live_rows.len());
        }
    }

    /// The **decode-step** shape: `m = 1`, one new token against a weight that
    /// is read once and thrown away. This is what every autoregressive step of
    /// a resident decoder actually dispatches, and it is a pure GEMV -
    /// arithmetic intensity `2k·n / (n·k·4)` = 0.5 FLOP/byte, i.e. bounded by
    /// weight traffic, never by the FMA rate.
    ///
    /// An earlier attempt at an `m = 1` microbench was dropped as untrustworthy
    /// (it read a physically impossible ~10-30 TFLOP/s). Two things made it
    /// lie, and this one fixes both rather than dropping the shape:
    ///
    /// * **One weight, re-read.** A single `n·k` matrix at this shape is 4.6 MB
    ///   and lives in a server L3 for the whole loop, so the second
    ///   iteration onward measured cache bandwidth, not the DRAM traffic a real
    ///   decode step pays. Here a **bank** of `EXPERTS` distinct weights is
    ///   swept in order, sized past L3 on purpose - which is also exactly the
    ///   real access pattern (a decoder walks every layer's weights once per
    ///   token and comes back to none of them).
    /// * **A loop-invariant call.** Rotating the weight pointer per iteration
    ///   makes the call genuinely different each time, and the accumulated
    ///   output checksum is returned through `black_box`, so nothing can be
    ///   hoisted.
    ///
    /// Reported as **GB/s of weight traffic against the measured roof**, not as
    /// GFLOP/s: at 0.5 FLOP/byte the FLOP rate is a derived quantity and citing
    /// it would name the wrong bound (see this repo's rule on reporting the
    /// bound a kernel is actually under).
    ///
    /// Run: `cargo test -p brain-backend-cpu --release decode_gemv_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn decode_gemv_bench() {
        // (label, k, n) at DeepSeek-OCR's real decoder dims (d_model 1280,
        // moe_ff 896, shared 2x896, vocab 129280) - the four GEMV shapes one
        // decode step issues, in descending call count.
        let shapes = [("moe expert ", 1280usize, 896usize), ("attn q/k/v/o", 1280, 1280), ("shared expert", 1280, 1792), ("lm_head    ", 1280, 129280)];
        let mut s = 11u32;
        let a: Vec<f32> = (0..1280).map(|_| lcg(&mut s)).collect();
        for (label, k, n) in shapes {
            // Enough distinct weights to blow past any current server L3 -
            // the bank, not any one matrix, is what must exceed the cache.
            let bytes = (n * k * 4) as f64;
            let experts = ((256.0 * 1024.0 * 1024.0 / bytes).ceil() as usize).max(2);
            let bank: Vec<Vec<f32>> = (0..experts).map(|_| (0..n * k).map(|_| lcg(&mut s)).collect()).collect();
            let mut c = vec![0.0f32; n];
            let mut sink = 0.0f64;
            for b in &bank {
                matmul_abt(&a, b, &mut c, 1, k, n); // warm every page in
            }
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                for b in &bank {
                    matmul_abt(&a, b, &mut c, 1, k, n);
                    sink += c[0] as f64;
                }
                best = best.min(t.elapsed().as_secs_f64() / experts as f64);
            }
            std::hint::black_box(sink);
            eprintln!(
                "decode gemv {label} m=1 k={k:<5} n={n:<6} ({experts:>3} x {:.2} MB bank, {} threads): {:>8.3} ms/call, {:>7.1} GB/s",
                bytes / 1e6,
                rayon::current_num_threads(),
                best * 1e3,
                bytes / best / 1e9,
            );
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

    /// [`matmul_abt`]'s two nests head to head at the shapes that actually
    /// decide a large diffusion-transformer forward, reporting the MINIMUM of
    /// several runs (a mean on a shared host measures the other tenant).
    ///
    /// Both nests are run in ONE process, back to back, on the same buffers -
    /// the only way to attribute a difference to the code rather than to
    /// whatever else the machine was doing between two separate builds.
    ///
    /// Shapes are MiniMax-H3's real per-layer GEMMs at a 960-row packed
    /// sequence (`hidden=5376`, `inner=7168`, `ffn=14336`), plus its per-head
    /// attention score block (`k=128`) and its `adaln_proj` projection, which
    /// is the deliberate counter-example: `n=2` is the shape where the
    /// column-outer nest is the WRONG choice and the selector must reject it.
    ///
    /// Run: `cargo test -p brain-backend-cpu --release matmul_shape_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn matmul_shape_bench() {
        let mut s = 5u32;
        // (label, m, k, n)
        let shapes = [
            ("h3 q/k/v   ", 960usize, 5376usize, 7168usize),
            ("h3 to_out  ", 960, 7168, 5376),
            ("h3 fc1     ", 960, 5376, 14336),
            ("h3 fc2     ", 960, 14336, 5376),
            ("h3 attn qk ", 960, 128, 960),
            ("h3 adaln   ", 5376, 2688, 2),
            // Either side of the B_RESIDENT_FLOATS threshold, to show the
            // selector is not balanced on a knife edge: same k, B just under
            // and well over 2 MB.
            ("edge k128 B1MB ", 960, 128, 2048),
            ("edge k128 B8MB ", 960, 128, 16384),
            ("edge k1024 B4MB", 960, 1024, 1024),
        ];
        for (label, m, k, n) in shapes {
            let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
            let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
            let mut c = vec![0.0f32; m * n];
            let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
            let iters = 5;

            // The pre-existing row-outer nest, verbatim, as the baseline.
            let rows_per = (m / (rayon::current_num_threads() * 4)).max(1);
            let mut c2 = vec![0.0f32; m * n];
            let row_outer = |c2: &mut Vec<f32>| {
                c2.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, cc)| {
                    let row0 = ci * rows_per;
                    for r in 0..cc.len() / n {
                        unsafe { row_abt_avx2(&a[(row0 + r) * k..(row0 + r) * k + k], &b, &mut cc[r * n..r * n + n], k, n) };
                    }
                });
            };
            matmul_abt(&a, &b, &mut c, m, k, n); // warm
            row_outer(&mut c2);

            // INTERLEAVED, one iteration of each arm at a time, rather than
            // all of one then all of the other: on a host with other tenants
            // the load drifts over the minutes a full sweep takes, and running
            // the arms in sequence silently charges that drift to whichever
            // arm ran second. Min-of-iters on top of that reports the least
            // contended observation each arm actually got.
            let (mut best, mut base) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..iters {
                let t = std::time::Instant::now();
                matmul_abt(&a, &b, &mut c, m, k, n);
                best = best.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                row_outer(&mut c2);
                base = base.min(t.elapsed().as_secs_f64());
            }
            let mut maxrel = 0f32;
            for i in 0..c.len() {
                maxrel = maxrel.max((c[i] - c2[i]).abs() / (c2[i].abs() + 1e-3));
            }
            eprintln!(
                "{label} m={m:<5} k={k:<6} n={n:<6} | now {:>8.2} ms ({:>6.1} GFLOP/s) | row-outer {:>8.2} ms ({:>6.1} GFLOP/s) | {:.2}x | max rel err {maxrel:.2e}",
                best * 1e3,
                gflop / best,
                base * 1e3,
                gflop / base,
                base / best
            );
        }
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
    // `deepseek2::config::DeepseekV2Config::real`) - the kernel this repo's own
    // `BRAIN_PROFILE` run measured as the majority of the whole decode loop.
    // `m=283` is the real prompt-prefill row count that run used.
    //
    // A single-decode-row (`m=1`) variant of this bench was tried and DROPPED:
    // at that shape the whole call is a handful of KFLOPs, small enough that
    // repeated measurements read a physically impossible rate (this
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
        // top_k=6 of 64 experts routed per row, so only a small fraction of
        // rows is live for a given expert, matching the real router's own
        // selection rate.
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

        // The SAME shape against a COLD weight - the arm the reused-`w` loop
        // above cannot measure. A real prefill round walks 2112 distinct
        // expert matrices (11 MoE layers x 64 experts x 3 projections, 9.7 GB
        // at this shape) and revisits none of them inside the round, so every
        // call's weight arrives from DRAM. Reusing one 4.59 MB `w` leaves it in
        // L3 from the second iteration on, which is why the number
        // above and the one below can differ by an order of magnitude - and the
        // served path only ever sees the cold one.
        let bank_bytes = 512usize * 1024 * 1024;
        let experts = (bank_bytes / (n * k * 4)).max(2);
        let bank: Vec<Vec<f32>> = (0..experts).map(|_| (0..n * k).map(|_| lcg(&mut s)).collect()).collect();
        let mut sink = 0.0f64;
        for b in &bank {
            moe_linear_gated_fwd(&x, b, &gate, &mut out, m, k, n, ne, e); // warm the pages in
        }
        let mut cold = f64::INFINITY;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for b in &bank {
                moe_linear_gated_fwd(&x, b, &gate, &mut out, m, k, n, ne, e);
                sink += out[0] as f64;
            }
            cold = cold.min(t.elapsed().as_secs_f64() / experts as f64);
        }
        std::hint::black_box(sink);
        eprintln!(
            "moe_linear_gated m={m} k={k} n={n} COLD ({experts} x {:.2} MB bank): {:.2} ms ({:.1} GFLOP/s, {:.1} GB/s of weight)",
            (n * k * 4) as f64 / 1e6,
            cold * 1e3,
            gflops / cold,
            (n * k * 4) as f64 / cold / 1e9,
        );
    }

    /// A whole **prefill round** of DeepSeek-OCR's MoE stack, replayed at the
    /// real call pattern rather than one call in a loop: 64 experts x 3
    /// projections of distinct weights (881 MB, so nothing survives in cache
    /// between calls) swept 11 times for the 11 MoE layers, against a gate
    /// where every one of the `m` rows selects exactly `top_k` of the 64
    /// experts - the invariant `router_gate.wgsl` actually produces, which an
    /// independent-Bernoulli gate does not (it gets the mean right and the
    /// per-expert distribution wrong).
    ///
    /// This is the shape that decides the served path's prefill, and the one
    /// that single-call benchmarks cannot see: they measure a warm weight and
    /// a warm rayon pool, and report a time an order of magnitude away from
    /// what the same kernel costs inside a real round.
    ///
    /// Run: `cargo test -p brain-backend-cpu --release moe_prefill_round_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn moe_prefill_round_bench() {
        let (k, n, ne, top_k, layers) = (1280usize, 896usize, 64usize, 6usize, 11usize);
        for m in [283usize, 1] {
            let mut s = 17u32;
            let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
            // Exactly `top_k` of `ne` per row, as the real router emits.
            let mut gate = vec![0f32; m * ne];
            for r in 0..m {
                for t in 0..top_k {
                    gate[r * ne + (r * top_k + t) % ne] = 0.25;
                }
            }
            // Every weight the round touches, distinct - 11 layers x 64
            // experts x 3 projections is 9.7 GB at this shape, which is the
            // whole point: a bank small enough to be revisited measures a
            // cache the served path never gets.
            let bank: Vec<Vec<f32>> = (0..layers * ne * 3).map(|_| (0..n * k).map(|_| lcg(&mut s)).collect()).collect();
            // One output slab per expert, as `model::moe::MoeActs` holds -
            // not one buffer reused 2112 times, which would sit in L3 and
            // leave the weight more of it than it really has.
            let mut outs: Vec<Vec<f32>> = (0..ne).map(|_| vec![0f32; m * n]).collect();
            let mut sink = 0.0f64;
            let mut best = f64::INFINITY;
            for _ in 0..2 {
                let t = std::time::Instant::now();
                for (i, b) in bank.iter().enumerate() {
                    let e = (i / 3) % ne;
                    let out = &mut outs[e];
                    moe_linear_gated_fwd(&x, b, &gate, out, m, k, n, ne, e);
                    sink += out[0] as f64;
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box(sink);
            let calls = layers * ne * 3;
            let live_flop = 2.0 * (m * top_k / ne).max(1) as f64 * k as f64 * n as f64 * calls as f64;
            eprintln!(
                "moe prefill round m={m:<4} ({calls} calls over a {:.0} MB bank, {} threads): {:>8.1} ms total, {:>7.3} ms/call, {:>6.1} GFLOP/s",
                (bank.len() * n * k * 4) as f64 / 1e6,
                rayon::current_num_threads(),
                best * 1e3,
                best * 1e3 / calls as f64,
                live_flop / best / 1e9,
            );
        }
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

    // Perf microbench (run: cargo test -p brain-backend-cpu --release attn_apply_cross_bench -- --ignored --nocapture)
    //
    // A/B, in the SAME process, back to back -- the two calls see the same
    // contention from whatever else is on the box, so the RATIO stays
    // meaningful even when the absolute numbers do not.
    #[test]
    #[ignore]
    fn attn_apply_cross_bench() {
        // SAM-1 ViT-B's global-attention block at DeepSeek-OCR's real shape
        // (`SamViTConfig::deepseek_ocr()`): bsz=1, heads=12, hd=64, one query
        // chunk (attn_chunk=256) against the full T=4096 key/value extent.
        let (bsz, heads, tq, tk, hd) = (1usize, 12usize, 256usize, 4096usize, 64usize);
        let kv_stride = 3 * heads * hd; // fused qkv, v at the top third
        let v_off = 2 * heads * hd;
        let d_model = heads * hd;
        let mut s = 11u32;
        let probs: Vec<f32> = (0..bsz * heads * tq * tk).map(|_| lcg(&mut s).abs()).collect();
        let kv: Vec<f32> = (0..bsz * tk * kv_stride).map(|_| lcg(&mut s)).collect();
        let mut out = vec![0f32; bsz * tq * d_model];

        // Naive reference: the pre-tiling transpose this file used to run
        // (`vt[d*tk+j] = kv[..]`, one `j` at a time, `hd` scattered writes each).
        fn attn_apply_cross_naive(
            probs: &[f32], kv: &[f32], out: &mut [f32], bsz: usize, heads: usize, tq: usize, tk: usize, hd: usize,
            kv_stride: usize, v_off: usize, d_model: usize,
        ) {
            let mut vt = vec![0f32; hd * tk];
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
                    matmul_abt(p, &vt, &mut ctxh, tq, tk, hd);
                    for i in 0..tq {
                        let dst = (b * tq + i) * d_model + h * hd;
                        out[dst..dst + hd].copy_from_slice(&ctxh[i * hd..i * hd + hd]);
                    }
                }
            }
        }

        let iters = 8;
        attn_apply_cross(&probs, &kv, &mut out, bsz, heads, tq, tk, hd, kv_stride, v_off, d_model); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            attn_apply_cross(&probs, &kv, &mut out, bsz, heads, tq, tk, hd, kv_stride, v_off, d_model);
        }
        let tiled = t.elapsed().as_secs_f64() / iters as f64;

        attn_apply_cross_naive(&probs, &kv, &mut out, bsz, heads, tq, tk, hd, kv_stride, v_off, d_model); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            attn_apply_cross_naive(&probs, &kv, &mut out, bsz, heads, tq, tk, hd, kv_stride, v_off, d_model);
        }
        let naive = t.elapsed().as_secs_f64() / iters as f64;

        eprintln!(
            "attn_apply_cross bsz={bsz} heads={heads} tq={tq} tk={tk} hd={hd}: \
             naive-transpose {:.2} ms | tiled-transpose {:.2} ms | {:.2}x",
            naive * 1e3,
            tiled * 1e3,
            naive / tiled
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
            brain_testutil::skip_unavailable("row_abt_avx512: AVX-512 not available");
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

    // ---------------------------------------------------------------------
    // silu_mul / scale_add: a real DeepSeek-OCR resident-server run
    // (BRAIN_PROFILE, quiet machine) found these two elementwise kernels among
    // the largest single entries EACH in the decoder's own profiled CPU time
    // (bigger than everything but matmul/moe_linear_gated) -- neither had a
    // FastIdx entry, so every decode-step call (single row, d_model=1280 or
    // moe_ff=896 elements) paid the generic dispatch's rayon-chunked-JIT path
    // in full: `total.div_ceil(threads*8)` rounded up to a whole workgroup
    // (64) spawns ~14 separate `par_iter` tasks for a ~900-element call. Same
    // bug class as `moe_linear_gated` earlier this session (F.3/F.4: a hot
    // kernel silently missing the native path its siblings already have).
    // ---------------------------------------------------------------------

    #[test]
    fn silu_mul_matches_scalar() {
        let mut s = 11u32;
        // Decode-shape (moe_ff=896, one row), a non-8-multiple width, and a
        // multi-row (prefill-shape) case that crosses the chunk threshold.
        for &n in &[896usize, 1280, 1, 5, 37, 4096 * 3 + 17] {
            let a: Vec<f32> = (0..n).map(|_| lcg(&mut s) * 6.0).collect();
            let b: Vec<f32> = (0..n).map(|_| lcg(&mut s) * 6.0).collect();
            let mut o = vec![0.0f32; n];
            silu_mul(&a, &b, &mut o);
            for i in 0..n {
                let want = (a[i] / (1.0 + (-a[i]).exp())) * b[i];
                assert!((o[i] - want).abs() < 1e-4, "silu_mul n={n} i={i}: got {} want {want}", o[i]);
            }
        }
    }

    // Scalar reference mirroring `scale_add.wgsl` exactly (not `axpy` with a
    // post-hoc mask - the set-vs-accumulate branch itself is under test).
    fn scale_add_scalar_ref(
        gate: &[f32],
        src: &[f32],
        acc: &mut [f32],
        seq_len: usize,
        d_model: usize,
        n_experts: usize,
        e_idx: usize,
        accumulate: bool,
    ) {
        for t in 0..seq_len {
            let g = gate[t * n_experts + e_idx];
            for c in 0..d_model {
                let idx = t * d_model + c;
                let contrib = g * src[idx];
                acc[idx] = if accumulate { acc[idx] + contrib } else { contrib };
            }
        }
    }

    #[test]
    fn scale_add_matches_scalar() {
        let mut s = 13u32;
        // (seq_len, d_model, n_experts): a single decode row (the real hot
        // shape), a non-8-multiple d_model, and a prefill-scale multi-row case
        // that crosses the rayon chunk threshold. Both accumulate branches,
        // including a zero-gate row (the accumulate-skip early exit).
        for &(seq_len, d_model, n_experts) in &[(1usize, 896usize, 64usize), (1, 5, 3), (7, 37, 4), (600, 1280, 64)] {
            let gate: Vec<f32> = (0..seq_len * n_experts).map(|_| lcg(&mut s).abs()).collect();
            let mut gate = gate;
            // Force at least one zero-gated row so the accumulate-skip path runs.
            if seq_len > 1 {
                for g in gate.iter_mut().take(n_experts) {
                    *g = 0.0;
                }
            }
            let src: Vec<f32> = (0..seq_len * d_model).map(|_| lcg(&mut s)).collect();
            let e_idx = n_experts / 2;
            for accumulate in [false, true] {
                let init: Vec<f32> = (0..seq_len * d_model).map(|_| lcg(&mut s)).collect();
                let mut got = init.clone();
                let mut want = init.clone();
                scale_add(&gate, &src, &mut got, seq_len, d_model, n_experts, e_idx, accumulate);
                scale_add_scalar_ref(&gate, &src, &mut want, seq_len, d_model, n_experts, e_idx, accumulate);
                for i in 0..got.len() {
                    assert!(
                        (got[i] - want[i]).abs() < 1e-5,
                        "scale_add seq_len={seq_len} d_model={d_model} accumulate={accumulate} i={i}: got {} want {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    /// The exact `matmul_i8_gemv.wgsl`/`matmul_i8_dyn.wgsl` math, one (m,n) at a
    /// time, scalar - the correctness oracle [`matmul_i8_dyn`] (M8.11) is gated
    /// against. Deliberately reimplemented from the WGSL doc's formula rather
    /// than calling `dot_group_scalar`/`matmul_i8_dyn` itself, so a shared bug in
    /// this module cannot cancel out in both places.
    fn matmul_i8_ref(xq: &[u32], wq: &[u32], sx: &[f32], sw: &[f32], m: usize, kg: usize, n: usize) -> Vec<f32> {
        const WPG: usize = 8;
        let ng = kg / WPG;
        let mut out = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc_f = 0.0f32;
                for g in 0..ng {
                    let mut acc_i = 0i32;
                    for w in 0..WPG {
                        let k = g * WPG + w;
                        let aw = xq[mi * kg + k];
                        let bw = wq[ni * kg + k];
                        for lane in 0..4 {
                            let ai = ((aw >> (lane * 8)) & 0xFF) as u8 as i8 as i32;
                            let bi = ((bw >> (lane * 8)) & 0xFF) as u8 as i8 as i32;
                            acc_i += ai * bi;
                        }
                    }
                    acc_f += acc_i as f32 * sw[ni * ng + g];
                }
                out[mi * n + ni] = acc_f * sx[mi];
            }
        }
        out
    }

    /// Random signed int8 lanes packed 4/word, clamped to `[-127, 127]` -
    /// `model::int8::quantize`'s own contract (never `-128`), which is what
    /// makes the AVX2 sign-trick in `dot32_i8_avx2` legal - see
    /// [`matmul_i8_dyn`]'s doc comment.
    fn random_packed_i8(words: usize, s: &mut u32) -> Vec<u32> {
        (0..words)
            .map(|_| {
                let mut w = 0u32;
                for lane in 0..4 {
                    let v = ((lcg(s) * 127.0).round().clamp(-127.0, 127.0)) as i8;
                    w |= (v as u8 as u32) << (lane * 8);
                }
                w
            })
            .collect()
    }

    /// [`matmul_i8_dyn`] (whichever ISA path this host takes - AVX2 on every
    /// box this test has run on) matches [`matmul_i8_ref`] EXACTLY (not an
    /// epsilon) - see `matmul_i8_dyn`'s own doc comment for why exact equality
    /// is the right bar here, not fp-reassociation tolerance.
    #[test]
    fn avx2_int8_gemm_matches_scalar_reference() {
        let mut s = 4242u32;
        // (m, n, kg): a decode-shaped GEMV (m=1), a small prefill GEMM, a
        // multi-group-per-row shape (kg=24 => 3 groups of 8 words), and a
        // shape that crosses the rayon-parallel threshold (m*n*kg >= 65536).
        for &(m, n, kg) in &[(1usize, 5usize, 8usize), (7, 11, 24), (3, 4, 8), (64, 128, 64)] {
            let xq = random_packed_i8(m * kg, &mut s);
            let wq = random_packed_i8(n * kg, &mut s);
            let ng = kg / 8;
            let sx: Vec<f32> = (0..m).map(|_| lcg(&mut s).abs() + 0.01).collect();
            let sw: Vec<f32> = (0..n * ng).map(|_| lcg(&mut s).abs() + 0.01).collect();

            let want = matmul_i8_ref(&xq, &wq, &sx, &sw, m, kg, n);
            let mut got = vec![0.0f32; m * n];
            matmul_i8_dyn(&xq, &wq, &sx, &sw, &mut got, m, kg, n);

            assert_eq!(got, want, "m={m} n={n} kg={kg}: AVX2 int8 GEMM diverged from the WGSL-matching scalar oracle");
        }
    }

    /// [`dot32_i8_avx2`] alone, against [`dot_group_scalar`], across every
    /// sign combination the clamp-to-127 contract allows (not just random
    /// data) - `-127 * -127`/`127 * -127`/etc are the corners the sign trick's
    /// `_mm256_abs_epi8`/`_mm256_sign_epi8` pairing must get exactly right.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot32_i8_avx2_matches_scalar_on_sign_corners() {
        if !crate::fast_conv::avx2_available() {
            eprintln!("skip: dot32_i8_avx2_matches_scalar_on_sign_corners needs AVX2, absent on this host");
            return;
        }
        let (a, b) = sign_corner_lanes();
        let want = dot_group_scalar(&a, &b);
        let got = unsafe { dot32_i8_avx2(&a, &b) };
        assert_eq!(got, want, "AVX2 sign-trick dot diverged from the scalar reference on the clamp corners");
    }

    /// The `[-127, -1, 0, 1, 127]` sign-corner pattern
    /// [`dot32_i8_avx2_matches_scalar_on_sign_corners`]/
    /// [`avx512vnni_int8_dot_matches_scalar_on_sign_corners`] both check -
    /// factored out so the AVX-512-VNNI test pins the EXACT same corner
    /// cases the already-validated AVX2 one does, not a second hand-picked
    /// set that could accidentally miss the one combination that matters.
    fn sign_corner_lanes() -> ([u32; 8], [u32; 8]) {
        let corners = [-127i8, -1, 0, 1, 127];
        let mut a = [0u32; 8];
        let mut b = [0u32; 8];
        for i in 0..32 {
            let av = corners[i % corners.len()];
            let bv = corners[(i / corners.len() + 1) % corners.len()];
            a[i / 4] |= (av as u8 as u32) << ((i % 4) * 8);
            b[i / 4] |= (bv as u8 as u32) << ((i % 4) * 8);
        }
        (a, b)
    }

    /// M8.12: the AVX-512-VNNI 32-lane dot, against the SAME scalar oracle
    /// and the SAME sign-corner pattern the already-hardware-validated AVX2
    /// test above pins. UNVALIDATED ON THIS BOX (confirmed: this Meteor Lake
    /// core has no AVX-512 of any kind, `avx512_vnni_available() ==
    /// false`) - `brain_testutil::skip_unvalidated_capability` says so
    /// loudly and records it to the capability ledger, rather than a silent
    /// `return` a future reader could mistake for "not applicable here".
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512vnni_int8_dot_matches_scalar_on_sign_corners() {
        if !crate::fast_conv::avx512_vnni_available() {
            brain_testutil::skip_unvalidated_capability(
                "avx512-vnni",
                "fast_ops::dot32_i8_avx512vnni (kernel-performance.md M8.12) needs AVX-512-VNNI \
                 (VPDPBUSD); this box (Core Ultra 7 155H / Meteor Lake) has no AVX-512 of any kind, \
                 confirmed against /proc/cpuinfo - compiled and shape-tested only, never run on real \
                 VNNI hardware. MAY diverge from the scalar reference on hardware that actually has it.",
            );
            return;
        }
        let (a, b) = sign_corner_lanes();
        let want = dot_group_scalar(&a, &b);
        let got = unsafe { dot32_i8_avx512vnni(&a, &b) };
        assert_eq!(got, want, "AVX-512-VNNI sign-trick dot diverged from the scalar reference on the clamp corners");
    }

    /// M8.13: the NEON `SDOT` 32-lane dot, against the SAME scalar oracle and
    /// SAME sign-corner pattern the AVX2/AVX-512-VNNI tests above pin.
    /// `#[cfg(target_arch = "aarch64")]`-gated on the TEST ITSELF, not just
    /// the kernel it calls - this test does not exist at all in the binary
    /// this campaign's own `cargo test` builds (x86_64), which is the honest
    /// reflection of "never compiled here", stronger than a runtime skip.
    /// `skip_unvalidated_capability`'s own gate is kept anyway for the day a
    /// real aarch64 CI leg compiles this file: even THERE, `neon_dotprod`
    /// may be genuinely absent (armv8.0 cores predate the dotprod extension),
    /// and the "unvalidated, may fail on hardware that has it" caveat still
    /// applies until this exact function has run once on real dotprod
    /// silicon - neither of which this campaign can confirm today.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_int8_dot_matches_scalar_on_sign_corners() {
        if !crate::fast_conv::neon_dotprod_available() {
            brain_testutil::skip_unvalidated_capability(
                "neon-dotprod",
                "fast_ops::dot32_i8_neon (kernel-performance.md M8.13) needs ARMv8.2-A NEON dot-product \
                 (SDOT/vdotq_s32); this function has never been compiled OR run anywhere in this \
                 campaign (no aarch64 toolchain target available in the sandbox that wrote it) - even \
                 on real aarch64 hardware, dotprod may be genuinely absent (pre-ARMv8.2 cores). MAY FAIL \
                 in ways a purely x86_64-tested campaign cannot catch.",
            );
            return;
        }
        let (a, b) = sign_corner_lanes();
        let want = dot_group_scalar(&a, &b);
        let got = unsafe { dot32_i8_neon(&a, &b) };
        assert_eq!(got, want, "NEON SDOT dot diverged from the scalar reference on the clamp corners");
    }

    /// Measured throughput, scalar vs AVX2, on this box's real core (M8.11's
    /// own "fully measurable here" claim) - printed, not asserted, since CI
    /// hardware speed is not a correctness property; run with
    /// `--test fast_ops -- --nocapture --ignored` to see the numbers.
    #[test]
    #[ignore = "throughput measurement, not a correctness gate - run explicitly with --ignored --nocapture"]
    fn avx2_int8_gemm_throughput_vs_scalar() {
        let (m, n, kg) = (32usize, 4096usize, 1024usize); // a realistic prefill-tile shape
        let mut s = 7u32;
        let xq = random_packed_i8(m * kg, &mut s);
        let wq = random_packed_i8(n * kg, &mut s);
        let ng = kg / 8;
        let sx: Vec<f32> = (0..m).map(|_| lcg(&mut s).abs() + 0.01).collect();
        let sw: Vec<f32> = (0..n * ng).map(|_| lcg(&mut s).abs() + 0.01).collect();
        let mut out = vec![0.0f32; m * n];

        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for mi in 0..m {
                for ni in 0..n {
                    let xrow = &xq[mi * kg..mi * kg + kg];
                    let wrow = &wq[ni * kg..ni * kg + kg];
                    let swrow = &sw[ni * ng..ni * ng + ng];
                    let mut acc_f = 0.0f32;
                    for (g, &sw_g) in swrow.iter().enumerate().take(ng) {
                        let off = g * 8;
                        acc_f += dot_group_scalar(&xrow[off..off + 8], &wrow[off..off + 8]) as f32 * sw_g;
                    }
                    out[mi * n + ni] = acc_f * sx[mi];
                }
            }
        }
        let scalar_secs = t0.elapsed().as_secs_f64() / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            matmul_i8_dyn(&xq, &wq, &sx, &sw, &mut out, m, kg, n);
        }
        let avx2_secs = t1.elapsed().as_secs_f64() / iters as f64;

        let macs = (m * n * kg * 4) as f64; // 4 int8 MACs per packed word
        eprintln!(
            "int8 GEMM {m}x{n}x{k}: scalar single-thread {:.2} ms ({:.2} GMAC/s), matmul_i8_dyn (AVX2 + rayon, {} threads) {:.2} ms ({:.2} GMAC/s), speedup {:.2}x",
            scalar_secs * 1e3,
            macs / scalar_secs / 1e9,
            rayon::current_num_threads(),
            avx2_secs * 1e3,
            macs / avx2_secs / 1e9,
            scalar_secs / avx2_secs,
            k = kg,
        );
    }

    /// The bidirectional self-attention trio's native CPU paths against the
    /// three `.wgsl` kernels' OWN definitions, transcribed index-for-index
    /// from `attn_scores_qk.wgsl`, `attn_softmax_bidir.wgsl` and
    /// `attn_apply_full.wgsl` rather than from the cross-family fast ops they
    /// route into - the whole point is to catch a mis-mapped uniform or a
    /// swapped q/k, and a reference derived from the same code being tested
    /// could not see either.
    ///
    /// `head_dim=128` with a `qk_stride > heads*hd` (a fused-buffer slice) and
    /// a `seq` that is not a multiple of the GEMM's 4-column block are all
    /// deliberate: they are the real MiniMax-H3 head width and the shapes
    /// where an off-by-one in the packing loop would otherwise hide.
    #[test]
    fn bidirectional_self_attention_fast_paths_match_the_kernel_definitions() {
        let mut s = 7u32;
        for &(bsz, heads, seq, hd) in &[(1usize, 3usize, 13usize, 128usize), (2, 2, 8, 16), (1, 4, 5, 32)] {
            let stride = heads * hd;
            let scale = 1.0 / (hd as f32).sqrt() * 1.7; // deliberately NOT 1/√hd
            let q: Vec<f32> = (0..bsz * seq * stride).map(|_| lcg(&mut s)).collect();
            let k: Vec<f32> = (0..bsz * seq * stride).map(|_| lcg(&mut s)).collect();
            let v: Vec<f32> = (0..bsz * seq * stride).map(|_| lcg(&mut s)).collect();

            // --- attn_scores_qk.wgsl ---
            let mut want_sc = vec![0f32; bsz * heads * seq * seq];
            for b in 0..bsz {
                for h in 0..heads {
                    for i in 0..seq {
                        for j in 0..seq {
                            let qb = (b * seq + i) * stride + h * hd;
                            let kb = (b * seq + j) * stride + h * hd;
                            let dot: f32 = (0..hd).map(|d| q[qb + d] * k[kb + d]).sum();
                            want_sc[((b * heads + h) * seq + i) * seq + j] = dot * scale;
                        }
                    }
                }
            }
            let mut got_sc = vec![0f32; bsz * heads * seq * seq];
            attn_scores_qk(&q, &k, &mut got_sc, bsz, heads, seq, hd, stride, scale);
            for i in 0..got_sc.len() {
                assert!((got_sc[i] - want_sc[i]).abs() < 1e-4, "scores_qk {bsz}/{heads}/{seq}/{hd} i={i}: got {} want {}", got_sc[i], want_sc[i]);
            }

            // --- attn_softmax_bidir.wgsl ---
            let rows = bsz * heads * seq;
            let mut want_p = vec![0f32; rows * seq];
            for r in 0..rows {
                let row = &want_sc[r * seq..r * seq + seq];
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
                let sum: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                for (j, w) in want_p[r * seq..r * seq + seq].iter_mut().enumerate() {
                    *w = (row[j] - mx).exp() / sum;
                }
            }
            let mut got_p = vec![0f32; rows * seq];
            attn_softmax_cross(&want_sc, &mut got_p, rows, seq);
            for i in 0..got_p.len() {
                assert!((got_p[i] - want_p[i]).abs() < 1e-6, "softmax_bidir {bsz}/{heads}/{seq} i={i}: got {} want {}", got_p[i], want_p[i]);
            }

            // --- attn_apply_full.wgsl ---
            let mut want_o = vec![0f32; bsz * seq * stride];
            for b in 0..bsz {
                for h in 0..heads {
                    for i in 0..seq {
                        for d in 0..hd {
                            let pb = ((b * heads + h) * seq + i) * seq;
                            let acc: f32 = (0..seq).map(|j| want_p[pb + j] * v[(b * seq + j) * stride + h * hd + d]).sum();
                            want_o[(b * seq + i) * stride + h * hd + d] = acc;
                        }
                    }
                }
            }
            let mut got_o = vec![0f32; bsz * seq * stride];
            attn_apply_cross(&want_p, &v, &mut got_o, bsz, heads, seq, seq, hd, stride, 0, stride);
            for i in 0..got_o.len() {
                assert!((got_o[i] - want_o[i]).abs() < 1e-5, "apply_full {bsz}/{heads}/{seq}/{hd} i={i}: got {} want {}", got_o[i], want_o[i]);
            }
        }
    }
}
