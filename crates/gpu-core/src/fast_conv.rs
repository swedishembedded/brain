// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fast native CPU conv2d, an execution-only optimization of the scalar
//! `conv2d.wgsl` kernel.
//!
//! WGSL stays the source of truth: this routine computes *exactly* the same
//! bias-free NCHW convolution (`y[n,co,ho,wo] = Σ x·w` with implicit zero-pad),
//! but routes it through a register-blocked, cache-friendly GEMM with an
//! AVX2+FMA microkernel and rayon tile parallelism instead of one scalar
//! invocation per output element. It is validated bit-approximately (fp
//! reassociation tolerance) against the scalar reference in the unit tests, and
//! gated behind runtime AVX2 detection with a portable scalar fallback so the
//! result is identical-in-spirit on any host.
//!
//! Mapping (per image `n`): the convolution is the matrix product
//!   `C[Cout, P] = A[Cout, Kg] · B[Kg, P]`,  Kg = Cin·K·K,  P = Ho·Wo
//! where `A` is the weight tensor viewed as rows of length `Kg` (already
//! contiguous in `[Cout,Cin,K,K]` layout) and `B` is the im2col of `x`. The
//! microkernel vectorises over `P` (contiguous in both `B` and the output),
//! accumulating a `4×16` tile in eight `__m256` registers.

use rayon::prelude::*;

/// Decoded `conv2d.wgsl` uniform params (`[N,Cin,H,W,Cout,K,stride,pad,Ho,Wo]`).
#[derive(Clone, Copy, Debug)]
pub struct ConvParams {
    pub n: usize,
    pub cin: usize,
    pub h: usize,
    pub w: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub ho: usize,
    pub wo: usize,
}

impl ConvParams {
    pub fn from_u32(p: &[u32]) -> ConvParams {
        ConvParams {
            n: p[0] as usize,
            cin: p[1] as usize,
            h: p[2] as usize,
            w: p[3] as usize,
            cout: p[4] as usize,
            k: p[5] as usize,
            stride: p[6] as usize,
            pad: p[7] as usize,
            ho: p[8] as usize,
            wo: p[9] as usize,
        }
    }
    pub fn x_len(&self) -> usize { self.n * self.cin * self.h * self.w }
    pub fn w_len(&self) -> usize { self.cout * self.cin * self.k * self.k }
    pub fn y_len(&self) -> usize { self.n * self.cout * self.ho * self.wo }
}

/// True iff the host can run the AVX2 fast path.
#[inline]
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Compute the bias-free NCHW convolution, matching `conv2d.wgsl` exactly (up to
/// fp reassociation). Uses the AVX2 GEMM path when available, else a portable
/// scalar GEMM with the same tiling.
pub fn conv2d(p: &ConvParams, x: &[f32], w: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), p.x_len());
    debug_assert_eq!(w.len(), p.w_len());
    debug_assert_eq!(y.len(), p.y_len());

    let kg = p.cin * p.k * p.k;
    let psz = p.ho * p.wo; // spatial positions per (n, co)
    if kg == 0 || psz == 0 || p.cout == 0 {
        y.iter_mut().for_each(|v| *v = 0.0);
        return;
    }

    // A 1×1, stride-1, no-pad conv needs no im2col: B aliases x directly.
    let one_by_one = p.k == 1 && p.stride == 1 && p.pad == 0;

    for n in 0..p.n {
        let y_img = &mut y[n * p.cout * psz..(n + 1) * p.cout * psz];
        if one_by_one {
            // B = x[n] viewed as [Cin, H*W] == [Kg, P].
            let b = &x[n * p.cin * p.h * p.w..(n + 1) * p.cin * p.h * p.w];
            gemm(p.cout, kg, psz, w, b, y_img);
        } else {
            let mut b = vec![0.0f32; kg * psz];
            im2col(p, &x[n * p.cin * p.h * p.w..(n + 1) * p.cin * p.h * p.w], &mut b);
            gemm(p.cout, kg, psz, w, &b, y_img);
        }
    }
}

/// Build the im2col matrix `B[Kg, P]` for image `x` ([Cin,H,W]).
/// `B[(ci*K+kh)*K+kw, ho*Wo+wo] = x[ci, ho*stride-pad+kh, wo*stride-pad+kw]` or 0.
fn im2col(p: &ConvParams, x: &[f32], b: &mut [f32]) {
    let psz = p.ho * p.wo;
    // Parallelise over Kg rows; each row is written disjointly.
    b.par_chunks_mut(psz).enumerate().for_each(|(kg_idx, row)| {
        let kw = kg_idx % p.k;
        let t = kg_idx / p.k;
        let kh = t % p.k;
        let ci = t / p.k;
        let xc = &x[ci * p.h * p.w..(ci + 1) * p.h * p.w];
        for ho in 0..p.ho {
            let hi = ho * p.stride + kh;
            // hi - pad in bounds?
            if hi < p.pad || hi - p.pad >= p.h {
                continue;
            }
            let hi = hi - p.pad;
            let row_off = ho * p.wo;
            let x_row = &xc[hi * p.w..(hi + 1) * p.w];
            for wo in 0..p.wo {
                let wi = wo * p.stride + kw;
                if wi < p.pad || wi - p.pad >= p.w {
                    continue;
                }
                row[row_off + wo] = x_row[wi - p.pad];
            }
        }
    });
}

/// `C[M,P] = A[M,Kg] · B[Kg,P]`, all row-major. Parallel over P column ranges so
/// both channel-heavy (large M) and spatial-heavy (large P) layers scale; within
/// each thread's band, `B` is walked in L2-sized column panels so a panel stays
/// resident across the whole output-channel loop (instead of re-streaming the
/// full `[Kg,P]` im2col once per channel tile).
fn gemm(m: usize, kg: usize, psz: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // Column-range chunks aligned to NR=16, ~4 per thread for load balance.
    let threads = rayon::current_num_threads().max(1);
    let nr = 16usize;
    let ntiles = psz.div_ceil(nr);
    let tiles_per_chunk = ntiles.div_ceil(threads * 4).max(1);
    let chunk_cols = (tiles_per_chunk * nr).max(nr);

    // B-panel width: keep the resident `[Kg, nc]` panel near ~192 KiB (L2-ish),
    // rounded to NR and at least one tile.
    let nc = {
        let budget_floats = (192 * 1024) / 4; // ~192 KiB of f32
        let cols = (budget_floats / kg.max(1)) / nr * nr;
        cols.clamp(nr, chunk_cols.max(nr))
    };

    let use_avx2 = avx2_available();
    // SAFETY: each chunk owns a disjoint [pa,pb) column band of every C row, so
    // the raw-pointer writes from concurrent chunks never alias.
    let cptr = SendMutPtr(c.as_mut_ptr());
    let starts: Vec<usize> = (0..psz).step_by(chunk_cols).collect();
    starts.par_iter().for_each(|&pa| {
        let cptr = cptr;
        let pb = (pa + chunk_cols).min(psz);
        unsafe {
            let c = std::slice::from_raw_parts_mut(cptr.0, m * psz);
            let mut pp = pa;
            while pp < pb {
                let pe = (pp + nc).min(pb);
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    gemm_band_avx2(m, kg, psz, a, b, c, pp, pe);
                    #[cfg(not(target_arch = "x86_64"))]
                    gemm_band_scalar(m, kg, psz, a, b, c, pp, pe);
                } else {
                    gemm_band_scalar(m, kg, psz, a, b, c, pp, pe);
                }
                pp = pe;
            }
        }
    });
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

/// Scalar GEMM over column band `[pa,pb)` — the portable reference path.
/// # Safety: `c` must be the full `[M*P]` output; the caller guarantees disjoint
/// bands across threads.
unsafe fn gemm_band_scalar(
    m: usize, kg: usize, psz: usize, a: &[f32], b: &[f32], c: &mut [f32], pa: usize, pb: usize,
) {
    for co in 0..m {
        let arow = &a[co * kg..co * kg + kg];
        let crow = c.as_mut_ptr().add(co * psz);
        for p in pa..pb {
            let mut acc = 0.0f32;
            for (kgi, &av) in arow.iter().enumerate() {
                acc += av * *b.get_unchecked(kgi * psz + p);
            }
            *crow.add(p) = acc;
        }
    }
}

/// AVX2+FMA GEMM over column band `[pa,pb)`. Computes a `4×16` register tile.
/// # Safety: requires avx2+fma; `c` is the full `[M*P]` output with disjoint
/// bands across threads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_band_avx2(
    m: usize, kg: usize, psz: usize, a: &[f32], b: &[f32], c: &mut [f32], pa: usize, pb: usize,
) {
    use std::arch::x86_64::*;
    let aptr = a.as_ptr();
    let bptr = b.as_ptr();
    let cptr = c.as_mut_ptr();

    let mut co = 0usize;
    while co + 4 <= m {
        let a0 = aptr.add(co * kg);
        let a1 = aptr.add((co + 1) * kg);
        let a2 = aptr.add((co + 2) * kg);
        let a3 = aptr.add((co + 3) * kg);
        let c0 = cptr.add(co * psz);
        let c1 = cptr.add((co + 1) * psz);
        let c2 = cptr.add((co + 2) * psz);
        let c3 = cptr.add((co + 3) * psz);

        let mut p = pa;
        while p + 16 <= pb {
            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();
            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();
            let mut acc20 = _mm256_setzero_ps();
            let mut acc21 = _mm256_setzero_ps();
            let mut acc30 = _mm256_setzero_ps();
            let mut acc31 = _mm256_setzero_ps();
            let mut bp = bptr.add(p);
            for kgi in 0..kg {
                let b0 = _mm256_loadu_ps(bp);
                let b1 = _mm256_loadu_ps(bp.add(8));
                let v0 = _mm256_set1_ps(*a0.add(kgi));
                acc00 = _mm256_fmadd_ps(v0, b0, acc00);
                acc01 = _mm256_fmadd_ps(v0, b1, acc01);
                let v1 = _mm256_set1_ps(*a1.add(kgi));
                acc10 = _mm256_fmadd_ps(v1, b0, acc10);
                acc11 = _mm256_fmadd_ps(v1, b1, acc11);
                let v2 = _mm256_set1_ps(*a2.add(kgi));
                acc20 = _mm256_fmadd_ps(v2, b0, acc20);
                acc21 = _mm256_fmadd_ps(v2, b1, acc21);
                let v3 = _mm256_set1_ps(*a3.add(kgi));
                acc30 = _mm256_fmadd_ps(v3, b0, acc30);
                acc31 = _mm256_fmadd_ps(v3, b1, acc31);
                bp = bp.add(psz);
            }
            _mm256_storeu_ps(c0.add(p), acc00);
            _mm256_storeu_ps(c0.add(p + 8), acc01);
            _mm256_storeu_ps(c1.add(p), acc10);
            _mm256_storeu_ps(c1.add(p + 8), acc11);
            _mm256_storeu_ps(c2.add(p), acc20);
            _mm256_storeu_ps(c2.add(p + 8), acc21);
            _mm256_storeu_ps(c3.add(p), acc30);
            _mm256_storeu_ps(c3.add(p + 8), acc31);
            p += 16;
        }
        // Column remainder (<16): scalar, still correct.
        for pp in p..pb {
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for kgi in 0..kg {
                let bv = *bptr.add(kgi * psz + pp);
                s0 += *a0.add(kgi) * bv;
                s1 += *a1.add(kgi) * bv;
                s2 += *a2.add(kgi) * bv;
                s3 += *a3.add(kgi) * bv;
            }
            *c0.add(pp) = s0;
            *c1.add(pp) = s1;
            *c2.add(pp) = s2;
            *c3.add(pp) = s3;
        }
        co += 4;
    }
    // Channel remainder (<4 rows): one row at a time, vectorised over P.
    while co < m {
        let a0 = aptr.add(co * kg);
        let c0 = cptr.add(co * psz);
        let mut p = pa;
        while p + 16 <= pb {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut bp = bptr.add(p);
            for kgi in 0..kg {
                let v0 = _mm256_set1_ps(*a0.add(kgi));
                acc0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(bp), acc0);
                acc1 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(bp.add(8)), acc1);
                bp = bp.add(psz);
            }
            _mm256_storeu_ps(c0.add(p), acc0);
            _mm256_storeu_ps(c0.add(p + 8), acc1);
            p += 16;
        }
        for pp in p..pb {
            let mut s0 = 0.0f32;
            for kgi in 0..kg {
                s0 += *a0.add(kgi) * *bptr.add(kgi * psz + pp);
            }
            *c0.add(pp) = s0;
        }
        co += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the exact `conv2d.wgsl` math, one output element at a time.
    fn conv_ref(p: &ConvParams, x: &[f32], w: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; p.y_len()];
        for n in 0..p.n {
            for co in 0..p.cout {
                for ho in 0..p.ho {
                    for wo in 0..p.wo {
                        let mut acc = 0.0f32;
                        for ci in 0..p.cin {
                            for kh in 0..p.k {
                                let hib = ho * p.stride + kh;
                                if hib < p.pad { continue; }
                                let hi = hib - p.pad;
                                if hi >= p.h { continue; }
                                for kw in 0..p.k {
                                    let wib = wo * p.stride + kw;
                                    if wib < p.pad { continue; }
                                    let wi = wib - p.pad;
                                    if wi >= p.w { continue; }
                                    let xi = ((n * p.cin + ci) * p.h + hi) * p.w + wi;
                                    let wi2 = ((co * p.cin + ci) * p.k + kh) * p.k + kw;
                                    acc += x[xi] * w[wi2];
                                }
                            }
                        }
                        let yi = ((n * p.cout + co) * p.ho + ho) * p.wo + wo;
                        y[yi] = acc;
                    }
                }
            }
        }
        y
    }

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn check(p: ConvParams) {
        let mut s = 12345u32 ^ (p.cout as u32 * 131 + p.k as u32 * 17 + p.h as u32);
        let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..p.w_len()).map(|_| lcg(&mut s)).collect();
        let mut y = vec![0.0f32; p.y_len()];
        conv2d(&p, &x, &w, &mut y);
        let yref = conv_ref(&p, &x, &w);
        let mut maxerr = 0.0f32;
        for (a, b) in y.iter().zip(yref.iter()) {
            maxerr = maxerr.max((a - b).abs() / (b.abs() + 1e-3));
        }
        assert!(maxerr < 2e-3, "rel err {maxerr} too large for {p:?}");
    }

    fn cp(n: usize, cin: usize, h: usize, w: usize, cout: usize, k: usize, stride: usize, pad: usize) -> ConvParams {
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        ConvParams { n, cin, h, w, cout, k, stride, pad, ho, wo }
    }

    #[test]
    fn conv_1x1() {
        check(cp(1, 8, 10, 10, 16, 1, 1, 0));
        check(cp(1, 13, 7, 5, 6, 1, 1, 0)); // odd dims -> partial tiles
    }

    #[test]
    fn conv_3x3_s1_p1() {
        check(cp(1, 4, 12, 12, 8, 3, 1, 1));
        check(cp(1, 3, 9, 11, 7, 3, 1, 1)); // partial channel + spatial tiles
    }

    #[test]
    fn conv_3x3_s2_p1() {
        check(cp(1, 6, 16, 16, 10, 3, 2, 1));
        check(cp(2, 5, 13, 13, 9, 3, 2, 1)); // batch 2
    }

    #[test]
    fn conv_stem_shape() {
        // yolov8 stem: 3->16, 3x3 s2 p1, on a small spatial proxy.
        check(cp(1, 3, 32, 32, 16, 3, 2, 1));
    }

    /// Contention-robust conv throughput probe: times each representative
    /// yolov8n@640 layer `N` times and reports the *minimum* wall time (the
    /// least-contended sample) + GFLOP/s. Ignored by default; run with:
    ///   cargo test --release -p brain-gpu-core -- --ignored --nocapture bench_conv_gflops
    #[test]
    #[ignore]
    fn bench_conv_gflops() {
        use std::time::Instant;
        // (label, cin,h,w,cout,k,stride,pad) — the heavy yolov8n@640 conv layers.
        let shapes = [
            ("stem 3->16 3x3 s2 @640", 3, 640, 640, 16, 3, 2, 1),
            ("16->32 3x3 s2 @320", 16, 320, 320, 32, 3, 2, 1),
            ("32->32 3x3 s1 @160", 32, 160, 160, 32, 3, 1, 1),
            ("64->64 3x3 s1 @80", 64, 80, 80, 64, 3, 1, 1),
            ("128->128 3x3 s1 @40", 128, 40, 40, 128, 3, 1, 1),
            ("1x1 128->128 @80", 128, 80, 80, 128, 1, 1, 0),
            ("1x1 256->256 @20", 256, 20, 20, 256, 1, 1, 0),
        ];
        let n = 30;
        let mut total_flop = 0.0f64;
        let mut total_min = 0.0f64;
        println!("\n=== conv microbench (min of {n}, threads={}) ===", rayon::current_num_threads());
        for (label, cin, h, w, cout, k, stride, pad) in shapes {
            let p = cp(1, cin, h, w, cout, k, stride, pad);
            let mut s = 999u32;
            let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
            let wt: Vec<f32> = (0..p.w_len()).map(|_| lcg(&mut s)).collect();
            let mut y = vec![0.0f32; p.y_len()];
            // warm
            conv2d(&p, &x, &wt, &mut y);
            let mut best = f64::INFINITY;
            for _ in 0..n {
                let t = Instant::now();
                conv2d(&p, &x, &wt, &mut y);
                best = best.min(t.elapsed().as_secs_f64());
            }
            let flop = 2.0 * (p.cout * p.cin * p.k * p.k * p.ho * p.wo) as f64;
            let gflops = flop / best / 1e9;
            total_flop += flop;
            total_min += best;
            println!("  {label:<26} {:7.2} ms  {gflops:7.1} GFLOP/s", best * 1e3);
        }
        println!("  {:<26} {:7.2} ms  {:7.1} GFLOP/s (aggregate)", "TOTAL",
                 total_min * 1e3, total_flop / total_min / 1e9);
    }
}
