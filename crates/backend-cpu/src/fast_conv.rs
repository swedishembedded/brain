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
/// `dilation` is 1 for the dense ABI; the grouped ABI (`from_u32_gd`) carries it.
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
    pub dilation: usize,
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
            dilation: 1,
            ho: p[8] as usize,
            wo: p[9] as usize,
        }
    }
    /// The grouped/dilated 12-u32 ABI of `conv2d_gd.wgsl` / `conv2d_gd_reg.wgsl`
    /// (`[N,Cin,H,W,Cout,K,stride,pad,dilation,groups,Ho,Wo]`). Returns the
    /// params (dilation folded in) and `groups` separately — the GEMM machinery
    /// is per-group dense, so groups never enters `ConvParams` itself.
    pub fn from_u32_gd(p: &[u32]) -> (ConvParams, usize) {
        (
            ConvParams {
                n: p[0] as usize,
                cin: p[1] as usize,
                h: p[2] as usize,
                w: p[3] as usize,
                cout: p[4] as usize,
                k: p[5] as usize,
                stride: p[6] as usize,
                pad: p[7] as usize,
                dilation: p[8] as usize,
                ho: p[10] as usize,
                wo: p[11] as usize,
            },
            p[9] as usize,
        )
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
    conv2d_impl(p, x, w, y, None);
}

/// Fused conv -> per-channel affine -> activation (matches `conv_act.wgsl`).
/// `sb` is `[2*Cout]` with `sb[2c]=scale[c]`, `sb[2c+1]=bias[c]`; `act` selects
/// the epilogue like the WGSL `p.act` (0 identity, 1 ReLU, 2 SiLU, 3 sigmoid),
/// applied panel-hot right after each GEMM panel, so no separate bn_eval/act
/// memory passes are needed.
pub fn conv2d_act(p: &ConvParams, x: &[f32], w: &[f32], sb: &[f32], y: &mut [f32], act: u32) {
    debug_assert_eq!(sb.len(), 2 * p.cout);
    conv2d_impl(p, x, w, y, Some((sb, act)));
}

/// True iff Winograd F(2×2,3×3) applies: a 3×3 stride-1 pad-1 conv (the dominant
/// yolov8 backbone/neck shape). Other shapes use the GEMM path.
#[inline]
fn winograd_applicable(p: &ConvParams) -> bool {
    p.k == 3 && p.stride == 1 && p.pad == 1 && p.dilation == 1 && p.n >= 1 && p.ho >= 1 && p.wo >= 1
}

/// Grouped/dilated conv (matches `conv2d_gd.wgsl` / `conv2d_gd_reg.wgsl`
/// exactly, up to fp reassociation). Two routes:
///
///   * depthwise (`groups == cin == cout`): a dedicated channel-parallel loop —
///     per-group GEMM would be a [1 x K²]·[K² x P] degenerate product whose
///     im2col/panel overhead dwarfs the K² multiplies;
///   * general grouped: each `(image, group)` is a DENSE conv on contiguous
///     channel slices (x, w and y are all channel-major), so it reuses the
///     whole GEMM machinery — im2col is dilation-aware.
///
/// This is what un-JITs ZipDepth's hottest remaining kernel (the grouped 1x1
/// fusion projections + the dilated depthwise branches were 56% of a CPU frame
/// as scalar JIT loops).
pub fn conv2d_gd(p: &ConvParams, groups: usize, x: &[f32], w: &[f32], y: &mut [f32]) {
    let g = groups.max(1);
    let (cin_g, cout_g) = (p.cin / g, p.cout / g);
    if cin_g == 1 && cout_g == 1 {
        return conv2d_depthwise(p, x, w, y);
    }
    let (hw, psz, kg) = (p.h * p.w, p.ho * p.wo, cin_g * p.k * p.k);
    let sub = ConvParams { n: 1, cin: cin_g, cout: cout_g, ..*p };
    for n in 0..p.n {
        for gi in 0..g {
            let xo = (n * p.cin + gi * cin_g) * hw;
            let yo = (n * p.cout + gi * cout_g) * psz;
            let wo = gi * cout_g * kg;
            conv2d_impl(&sub, &x[xo..xo + cin_g * hw], &w[wo..wo + cout_g * kg], &mut y[yo..yo + cout_g * psz], None);
        }
    }
}

/// Depthwise conv (`groups == cin == cout`), channel-parallel, dilation-aware.
/// Interior rows skip the bounds checks (the hot path for K=3/5 same-pad).
fn conv2d_depthwise(p: &ConvParams, x: &[f32], w: &[f32], y: &mut [f32]) {
    let (hw, psz, kk) = (p.h * p.w, p.ho * p.wo, p.k * p.k);
    y.par_chunks_mut(psz).enumerate().for_each(|(plane, yc)| {
        let c = plane % p.cout;
        let n = plane / p.cout;
        let xc = &x[(n * p.cin + c) * hw..(n * p.cin + c) * hw + hw];
        let wc = &w[c * kk..(c + 1) * kk];
        for ho in 0..p.ho {
            let hbase = ho * p.stride;
            for wo_ in 0..p.wo {
                let wbase = wo_ * p.stride;
                let mut acc = 0.0f32;
                for kh in 0..p.k {
                    let hib = hbase + kh * p.dilation;
                    if hib < p.pad || hib - p.pad >= p.h {
                        continue;
                    }
                    let row = (hib - p.pad) * p.w;
                    for kw in 0..p.k {
                        let wib = wbase + kw * p.dilation;
                        if wib < p.pad || wib - p.pad >= p.w {
                            continue;
                        }
                        acc += xc[row + (wib - p.pad)] * wc[kh * p.k + kw];
                    }
                }
                yc[ho * p.wo + wo_] = acc;
            }
        }
    });
}

/// Fused conv + per-output-channel bias (matches `conv_bias.wgsl`): the
/// bias-free conv, then `y[co,..] += bias[co]`. `bias` is `[Cout]`.
pub fn conv2d_bias(p: &ConvParams, x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32]) {
    conv2d_impl(p, x, w, y, None);
    let psz = p.ho * p.wo;
    if psz == 0 {
        return;
    }
    // plane index = n*Cout + co, so the channel is plane % Cout.
    y.par_chunks_mut(psz).enumerate().for_each(|(plane, row)| {
        let b = bias[plane % p.cout];
        for v in row.iter_mut() {
            *v += b;
        }
    });
}

fn conv2d_impl(p: &ConvParams, x: &[f32], w: &[f32], y: &mut [f32], sb: Option<(&[f32], u32)>) {
    debug_assert_eq!(x.len(), p.x_len());
    debug_assert_eq!(w.len(), p.w_len());
    debug_assert_eq!(y.len(), p.y_len());

    // Winograd F(2,3) for 3×3 s1 p1 exists and is validated, but as implemented
    // (scalar transforms, 16 coord-parallel GEMMs) it is SLOWER than the tuned
    // AVX2 GEMM below — the transform + materialization overhead and weaker
    // parallelization eat the 2.25× multiply saving. So it is OPT-IN
    // (BRAIN_WINOGRAD=1) scaffolding for the Phase-7 work (vectorized fused
    // transforms, column-parallel transform-domain GEMM) rather than the default.
    if winograd_applicable(p)
        && avx2_available()
        && std::env::var("BRAIN_WINOGRAD").map(|v| v != "0").unwrap_or(false)
    {
        winograd::conv2d_f23(p, x, w, sb, y);
        return;
    }

    let kg = p.cin * p.k * p.k;
    let psz = p.ho * p.wo; // spatial positions per (n, co)
    if kg == 0 || psz == 0 || p.cout == 0 {
        y.iter_mut().for_each(|v| *v = 0.0);
        return;
    }

    // A 1×1, stride-1, no-pad conv needs no im2col: B aliases x directly.
    let one_by_one = p.k == 1 && p.stride == 1 && p.pad == 0;

    let use_avx2 = avx2_available();
    // P-panel width: keep the resident im2col panel `[Kg, nc]` near ~192 KiB
    // (L2-ish), rounded to the NR=16 microkernel tile, ≥1 tile.
    let nc = {
        let budget_floats = (192 * 1024) / 4;
        ((budget_floats / kg.max(1)) / 16 * 16).clamp(16, psz.max(16))
    };
    // Thread-parallel column bands (~4 per thread), each subdivided into nc
    // panels so the panel stays cache-resident across the channel loop.
    let threads = rayon::current_num_threads().max(1);
    let band = (psz.div_ceil(threads * 4)).max(1).div_ceil(16) * 16;

    for n in 0..p.n {
        let x_img = &x[n * p.cin * p.h * p.w..(n + 1) * p.cin * p.h * p.w];
        let y_img = &mut y[n * p.cout * psz..(n + 1) * p.cout * psz];
        let cptr = SendMutPtr(y_img.as_mut_ptr());
        let starts: Vec<usize> = (0..psz).step_by(band).collect();
        starts.par_iter().for_each(|&pa| {
            let cptr = cptr;
            let pb = (pa + band).min(psz);
            // Reusable per-band im2col scratch (general conv only).
            let mut bpanel: Vec<f32> = if one_by_one { Vec::new() } else { vec![0.0f32; kg * nc] };
            let mut coords: Vec<(usize, usize)> = if one_by_one { Vec::new() } else { vec![(0, 0); nc] };
            let cbase = cptr.0;
            let mut pp = pa;
            while pp < pb {
                let pw = (pp + nc).min(pb) - pp;
                // SAFETY: this band owns the disjoint C columns [pa,pb) of every
                // row; bpanel is band-local; pointers stay in bounds.
                unsafe {
                    let (b_base, bstride) = if one_by_one {
                        (x_img.as_ptr().add(pp), psz)
                    } else {
                        build_im2col_panel(p, x_img, pp, pw, &mut bpanel, &mut coords);
                        (bpanel.as_ptr(), pw)
                    };
                    let c_base = cbase.add(pp);
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        gemm_cols_avx2(p.cout, kg, w.as_ptr(), b_base, bstride, c_base, psz, pw);
                        #[cfg(not(target_arch = "x86_64"))]
                        gemm_cols_scalar(p.cout, kg, w.as_ptr(), b_base, bstride, c_base, psz, pw);
                    } else {
                        gemm_cols_scalar(p.cout, kg, w.as_ptr(), b_base, bstride, c_base, psz, pw);
                    }
                    // Fused epilogue: apply per-channel affine + activation to
                    // this panel's freshly-written output columns (still
                    // cache-hot), so bn_eval+act need no separate memory passes.
                    if let Some((sb, act)) = sb {
                        for co in 0..p.cout {
                            let row = std::slice::from_raw_parts_mut(c_base.add(co * psz), pw);
                            let (s, b) = (sb[2 * co], sb[2 * co + 1]);
                            match act {
                                1 => crate::fast_ops::affine_relu_inplace(row, s, b),
                                2 => crate::fast_ops::affine_silu_inplace(row, s, b),
                                3 => crate::fast_ops::affine_sigmoid_inplace(row, s, b),
                                _ => crate::fast_ops::affine_inplace(row, s, b),
                            }
                        }
                    }
                }
                pp += pw;
            }
        });
    }
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

/// Build the im2col panel for output columns `[pp0, pp0+pw)` into
/// `bpanel[kg*pw + j]` (`j` = local column). Out-of-bounds taps are zeroed.
/// `coords` is scratch for the `(ho,wo)` of each panel column (computed once,
/// reused across the Kg rows — avoids a div per cell).
unsafe fn build_im2col_panel(
    p: &ConvParams, x: &[f32], pp0: usize, pw: usize, bpanel: &mut [f32], coords: &mut [(usize, usize)],
) {
    // Roll (ho,wo) across the panel without a per-column division.
    let mut ho = pp0 / p.wo;
    let mut wo = pp0 - ho * p.wo;
    for c in coords.iter_mut().take(pw) {
        *c = (ho, wo);
        wo += 1;
        if wo == p.wo {
            wo = 0;
            ho += 1;
        }
    }
    let hw = p.h * p.w;
    let xptr = x.as_ptr();
    for kgi in 0..p.k * p.k * p.cin {
        let kw = kgi % p.k;
        let t = kgi / p.k;
        let kh = t % p.k;
        let ci = t / p.k;
        let xc = xptr.add(ci * hw);
        let row = bpanel.as_mut_ptr().add(kgi * pw);
        for (j, &(ho, wo)) in coords.iter().enumerate().take(pw) {
            let hib = ho * p.stride + kh * p.dilation;
            let wib = wo * p.stride + kw * p.dilation;
            let v = if hib >= p.pad && hib - p.pad < p.h && wib >= p.pad && wib - p.pad < p.w {
                *xc.add((hib - p.pad) * p.w + (wib - p.pad))
            } else {
                0.0
            };
            *row.add(j) = v;
        }
    }
}

/// `C[co, j] = Σ_kg A[co,kg]·B[kg,j]` for `co∈0..m`, `j∈0..ncols`. `B` rows have
/// stride `bstride`, `C` rows stride `cstride`; `a`/`b_base`/`c_base` are bases.
/// Scalar reference path. # Safety: caller guarantees disjoint C columns.
unsafe fn gemm_cols_scalar(
    m: usize, kg: usize, a: *const f32, b_base: *const f32, bstride: usize,
    c_base: *mut f32, cstride: usize, ncols: usize,
) {
    for co in 0..m {
        let arow = a.add(co * kg);
        let crow = c_base.add(co * cstride);
        for j in 0..ncols {
            let mut acc = 0.0f32;
            for kgi in 0..kg {
                acc += *arow.add(kgi) * *b_base.add(kgi * bstride + j);
            }
            *crow.add(j) = acc;
        }
    }
}

/// AVX2+FMA variant of [`gemm_cols_scalar`]: a `4×16` register tile over
/// (channel, column). # Safety: requires avx2+fma; disjoint C columns per call.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_cols_avx2(
    m: usize, kg: usize, a: *const f32, b_base: *const f32, bstride: usize,
    c_base: *mut f32, cstride: usize, ncols: usize,
) {
    use std::arch::x86_64::*;
    let mut co = 0usize;
    while co + 4 <= m {
        let a0 = a.add(co * kg);
        let a1 = a.add((co + 1) * kg);
        let a2 = a.add((co + 2) * kg);
        let a3 = a.add((co + 3) * kg);
        let c0 = c_base.add(co * cstride);
        let c1 = c_base.add((co + 1) * cstride);
        let c2 = c_base.add((co + 2) * cstride);
        let c3 = c_base.add((co + 3) * cstride);
        let mut j = 0usize;
        while j + 16 <= ncols {
            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();
            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();
            let mut acc20 = _mm256_setzero_ps();
            let mut acc21 = _mm256_setzero_ps();
            let mut acc30 = _mm256_setzero_ps();
            let mut acc31 = _mm256_setzero_ps();
            let mut bp = b_base.add(j);
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
                bp = bp.add(bstride);
            }
            _mm256_storeu_ps(c0.add(j), acc00);
            _mm256_storeu_ps(c0.add(j + 8), acc01);
            _mm256_storeu_ps(c1.add(j), acc10);
            _mm256_storeu_ps(c1.add(j + 8), acc11);
            _mm256_storeu_ps(c2.add(j), acc20);
            _mm256_storeu_ps(c2.add(j + 8), acc21);
            _mm256_storeu_ps(c3.add(j), acc30);
            _mm256_storeu_ps(c3.add(j + 8), acc31);
            j += 16;
        }
        for jj in j..ncols {
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for kgi in 0..kg {
                let bv = *b_base.add(kgi * bstride + jj);
                s0 += *a0.add(kgi) * bv;
                s1 += *a1.add(kgi) * bv;
                s2 += *a2.add(kgi) * bv;
                s3 += *a3.add(kgi) * bv;
            }
            *c0.add(jj) = s0;
            *c1.add(jj) = s1;
            *c2.add(jj) = s2;
            *c3.add(jj) = s3;
        }
        co += 4;
    }
    while co < m {
        let a0 = a.add(co * kg);
        let c0 = c_base.add(co * cstride);
        let mut j = 0usize;
        while j + 16 <= ncols {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut bp = b_base.add(j);
            for kgi in 0..kg {
                let v0 = _mm256_set1_ps(*a0.add(kgi));
                acc0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(bp), acc0);
                acc1 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(bp.add(8)), acc1);
                bp = bp.add(bstride);
            }
            _mm256_storeu_ps(c0.add(j), acc0);
            _mm256_storeu_ps(c0.add(j + 8), acc1);
            j += 16;
        }
        for jj in j..ncols {
            let mut s0 = 0.0f32;
            for kgi in 0..kg {
                s0 += *a0.add(kgi) * *b_base.add(kgi * bstride + jj);
            }
            *c0.add(jj) = s0;
        }
        co += 1;
    }
}

/// Winograd F(2×2, 3×3) convolution. For a 3×3 stride-1 pad-1 conv it produces a
/// 2×2 output tile from a 4×4 input tile with 16 element-wise products instead of
/// 36 direct multiplies (2.25× fewer), doing the per-channel reduction as 16
/// GEMMs in the transform domain (reusing the AVX2 microkernel). Matches the
/// direct conv up to Winograd's (looser) fp conditioning.
mod winograd {
    use super::*;

    // F(2,3) transform matrices.
    // Input:  V = Bᵀ d B          (B = Btᵀ)
    // Weight: U = G g Gᵀ
    // Output: Y = Aᵀ M A          (A = Atᵀ)
    const BT: [[f32; 4]; 4] = [
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
    ];
    const G: [[f32; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.0, 0.0, 1.0],
    ];
    const AT: [[f32; 4]; 2] = [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]];

    /// U[4×4] = G g Gᵀ for a 3×3 filter `g` (row-major 9).
    fn weight_transform(g: &[f32], u: &mut [f32; 16]) {
        // t = G g  (4×3)
        let mut t = [[0.0f32; 3]; 4];
        for (i, ti) in t.iter_mut().enumerate() {
            for (k, tik) in ti.iter_mut().enumerate() {
                *tik = G[i][0] * g[k] + G[i][1] * g[3 + k] + G[i][2] * g[6 + k];
            }
        }
        // U = t Gᵀ  (4×4): U[i][j] = Σ_k t[i][k] G[j][k]
        for i in 0..4 {
            for j in 0..4 {
                u[i * 4 + j] = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
            }
        }
    }

    /// V[4×4] = Bᵀ d B for a 4×4 input tile `d`.
    fn input_transform(d: &[f32; 16], v: &mut [f32; 16]) {
        // t = Bᵀ d (4×4): t[i][j] = Σ_k BT[i][k] d[k][j]
        let mut t = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += BT[i][k] * d[k * 4 + j];
                }
                t[i][j] = s;
            }
        }
        // V = t B = t Btᵀ: V[i][j] = Σ_k t[i][k] BT[j][k]
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += t[i][k] * BT[j][k];
                }
                v[i * 4 + j] = s;
            }
        }
    }

    /// Y[2×2] = Aᵀ M A for a 4×4 transform-domain tile `m`.
    fn output_transform(m: &[f32; 16], y: &mut [f32; 4]) {
        // t = Aᵀ M (2×4): t[i][j] = Σ_k AT[i][k] m[k][j]
        let mut t = [[0.0f32; 4]; 2];
        for i in 0..2 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += AT[i][k] * m[k * 4 + j];
                }
                t[i][j] = s;
            }
        }
        // Y = t A = t Atᵀ (2×2): Y[i][j] = Σ_k t[i][k] AT[j][k]
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += t[i][k] * AT[j][k];
                }
                y[i * 2 + j] = s;
            }
        }
    }

    pub fn conv2d_f23(p: &ConvParams, x: &[f32], w: &[f32], sb: Option<(&[f32], u32)>, y: &mut [f32]) {
        let (cin, cout) = (p.cin, p.cout);
        let (h, wd, ho, wo) = (p.h, p.w, p.ho, p.wo);
        let (tth, ttw) = (ho.div_ceil(2), wo.div_ceil(2)); // output tile grid
        let nt = tth * ttw; // tiles per image
        if nt == 0 {
            y.iter_mut().for_each(|v| *v = 0.0);
            return;
        }

        // 1. Weight transform U[16][Cout][Cin]  (parallel over Cout).
        let mut u = vec![0.0f32; 16 * cout * cin];
        {
            let uptr = SendMutPtr(u.as_mut_ptr());
            (0..cout).into_par_iter().for_each(|co| {
                let uptr = uptr;
                let mut uf = [0.0f32; 16];
                for ci in 0..cin {
                    let g = &w[(co * cin + ci) * 9..(co * cin + ci) * 9 + 9];
                    weight_transform(g, &mut uf);
                    for (t, &uv) in uf.iter().enumerate() {
                        // U[t][co][ci]
                        unsafe { *uptr.0.add((t * cout + co) * cin + ci) = uv; }
                    }
                }
            });
        }

        for n in 0..p.n {
            let x_img = &x[n * cin * h * wd..(n + 1) * cin * h * wd];
            // 2. Input transform V[16][Cin][nt]  (parallel over Cin).
            let mut v = vec![0.0f32; 16 * cin * nt];
            {
                let vptr = SendMutPtr(v.as_mut_ptr());
                (0..cin).into_par_iter().for_each(|ci| {
                    let vptr = vptr;
                    let xc = &x_img[ci * h * wd..(ci + 1) * h * wd];
                    let mut d = [0.0f32; 16];
                    let mut vf = [0.0f32; 16];
                    for tyi in 0..tth {
                        for txi in 0..ttw {
                            // 4×4 input patch at (2*ty-1, 2*tx-1) (pad=1).
                            let h0 = (2 * tyi) as isize - 1;
                            let w0 = (2 * txi) as isize - 1;
                            for r in 0..4 {
                                let hr = h0 + r as isize;
                                for c in 0..4 {
                                    let wc = w0 + c as isize;
                                    d[r * 4 + c] = if hr >= 0 && (hr as usize) < h && wc >= 0 && (wc as usize) < wd {
                                        xc[hr as usize * wd + wc as usize]
                                    } else {
                                        0.0
                                    };
                                }
                            }
                            input_transform(&d, &mut vf);
                            let tile = tyi * ttw + txi;
                            for (t, &vv) in vf.iter().enumerate() {
                                unsafe { *vptr.0.add((t * cin + ci) * nt + tile) = vv; }
                            }
                        }
                    }
                });
            }

            // 3. 16 transform-domain GEMMs: M[t] = U[t]·V[t] -> [Cout][nt]
            //    (parallel over the 16 coords; each GEMM uses the AVX2 microkernel).
            let mut m = vec![0.0f32; 16 * cout * nt];
            {
                let mptr = SendMutPtr(m.as_mut_ptr());
                (0..16usize).into_par_iter().for_each(|t| {
                    let mptr = mptr;
                    let a = &u[t * cout * cin..(t + 1) * cout * cin];
                    let b = &v[t * cin * nt..(t + 1) * cin * nt];
                    unsafe {
                        let c = mptr.0.add(t * cout * nt);
                        #[cfg(target_arch = "x86_64")]
                        gemm_cols_avx2(cout, cin, a.as_ptr(), b.as_ptr(), nt, c, nt, nt);
                        #[cfg(not(target_arch = "x86_64"))]
                        gemm_cols_scalar(cout, cin, a.as_ptr(), b.as_ptr(), nt, c, nt, nt);
                    }
                });
            }

            // 4. Output transform + scatter (+ optional fused affine/act),
            //    parallel over Cout.
            let y_img = &mut y[n * cout * ho * wo..(n + 1) * cout * ho * wo];
            let yptr = SendMutPtr(y_img.as_mut_ptr());
            (0..cout).into_par_iter().for_each(|co| {
                let yptr = yptr;
                let (scale, bias) = match sb {
                    Some((sb, _)) => (sb[2 * co], sb[2 * co + 1]),
                    None => (1.0, 0.0),
                };
                let act = sb.map(|(_, a)| a);
                let mut mf = [0.0f32; 16];
                let mut yf = [0.0f32; 4];
                for tyi in 0..tth {
                    for txi in 0..ttw {
                        let tile = tyi * ttw + txi;
                        for t in 0..16 {
                            mf[t] = m[(t * cout + co) * nt + tile];
                        }
                        output_transform(&mf, &mut yf);
                        // scatter 2×2, clipping to (ho,wo); apply epilogue.
                        for dr in 0..2 {
                            let oy = 2 * tyi + dr;
                            if oy >= ho { continue; }
                            for dc in 0..2 {
                                let ox = 2 * txi + dc;
                                if ox >= wo { continue; }
                                let mut z = yf[dr * 2 + dc];
                                if let Some(act) = act {
                                    z = z * scale + bias;
                                    match act {
                                        1 => z = z.max(0.0),
                                        2 => z /= 1.0 + (-z).exp(),
                                        3 => z = 1.0 / (1.0 + (-z).exp()),
                                        _ => {}
                                    }
                                }
                                unsafe { *yptr.0.add((co * ho + oy) * wo + ox) = z; }
                            }
                        }
                    }
                }
            });
        }
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

    /// Reference for the GROUPED/DILATED conv: the exact `conv2d_gd.wgsl` math,
    /// one output element at a time (`w` is `[Cout, Cin/G, K, K]`).
    fn conv_gd_ref(p: &ConvParams, groups: usize, x: &[f32], w: &[f32]) -> Vec<f32> {
        let (cin_g, cout_g) = (p.cin / groups, p.cout / groups);
        let mut y = vec![0.0f32; p.y_len()];
        for n in 0..p.n {
            for co in 0..p.cout {
                let g = co / cout_g;
                for ho in 0..p.ho {
                    for wo in 0..p.wo {
                        let mut acc = 0.0f32;
                        for cl in 0..cin_g {
                            let ci = g * cin_g + cl;
                            for kh in 0..p.k {
                                let hib = ho * p.stride + kh * p.dilation;
                                if hib < p.pad { continue; }
                                let hi = hib - p.pad;
                                if hi >= p.h { continue; }
                                for kw in 0..p.k {
                                    let wib = wo * p.stride + kw * p.dilation;
                                    if wib < p.pad { continue; }
                                    let wi = wib - p.pad;
                                    if wi >= p.w { continue; }
                                    let xi = ((n * p.cin + ci) * p.h + hi) * p.w + wi;
                                    let wi2 = ((co * cin_g + cl) * p.k + kh) * p.k + kw;
                                    acc += x[xi] * w[wi2];
                                }
                            }
                        }
                        y[((n * p.cout + co) * p.ho + ho) * p.wo + wo] = acc;
                    }
                }
            }
        }
        y
    }

    #[test]
    fn conv2d_gd_matches_grouped_reference() {
        // (cin, cout, groups, k, stride, pad, dilation): grouped 1x1 with a
        // non-multiple-of-8 cout_g (the fusion-projection shape), grouped 3x3,
        // depthwise 3x3, and depthwise DILATED 3x3 (MinimalMultiScale).
        let cases = [
            (24, 24, 2, 1, 1, 0, 1),
            (12, 20, 4, 3, 1, 1, 1),
            (16, 16, 16, 3, 1, 1, 1),
            (16, 16, 16, 3, 1, 2, 2),
            (8, 12, 2, 3, 2, 1, 1),
        ];
        for (cin, cout, groups, k, stride, pad, dil) in cases {
            let eff = dil * (k - 1) + 1;
            let (h, w) = (13usize, 11usize);
            let p = ConvParams {
                n: 2,
                cin,
                h,
                w,
                cout,
                k,
                stride,
                pad,
                dilation: dil,
                ho: (h + 2 * pad - eff) / stride + 1,
                wo: (w + 2 * pad - eff) / stride + 1,
            };
            let mut s = 999u32 ^ (cout as u32 * 31 + groups as u32);
            let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
            let wlen = cout * (cin / groups) * k * k;
            let wt: Vec<f32> = (0..wlen).map(|_| lcg(&mut s)).collect();
            let mut y = vec![0.0f32; p.y_len()];
            conv2d_gd(&p, groups, &x, &wt, &mut y);
            let yref = conv_gd_ref(&p, groups, &x, &wt);
            let mut maxerr = 0.0f32;
            for (a, b) in y.iter().zip(yref.iter()) {
                maxerr = maxerr.max((a - b).abs() / (b.abs() + 1e-3));
            }
            assert!(
                maxerr < 2e-3,
                "grouped conv rel err {maxerr} for cin={cin} cout={cout} g={groups} k={k} s={stride} p={pad} d={dil}"
            );
        }
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
        ConvParams { n, cin, h, w, cout, k, stride, pad, dilation: 1, ho, wo }
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

    #[test]
    fn winograd_matches_scalar() {
        // Winograd F(2,3) applies to 3x3 s1 p1; validate vs the direct conv
        // (looser tol — Winograd has weaker fp conditioning than direct).
        for p in [cp(1, 8, 12, 12, 16, 3, 1, 1), cp(1, 3, 9, 11, 7, 3, 1, 1), cp(2, 5, 16, 14, 9, 3, 1, 1)] {
            let mut s = 21u32 ^ p.cout as u32;
            let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
            let w: Vec<f32> = (0..p.w_len()).map(|_| lcg(&mut s)).collect();
            let mut y = vec![0.0f32; p.y_len()];
            winograd::conv2d_f23(&p, &x, &w, None, &mut y);
            let yref = conv_ref(&p, &x, &w);
            let mut maxerr = 0.0f32;
            for (a, b) in y.iter().zip(yref.iter()) {
                maxerr = maxerr.max((a - b).abs() / (b.abs() + 1e-2));
            }
            assert!(maxerr < 5e-3, "winograd rel err {maxerr} for {p:?}");
        }
    }

    #[test]
    fn conv_act_matches_conv_then_affine_act() {
        // All four epilogues of the act selector (0 identity, 1 relu, 2 silu,
        // 3 sigmoid) against the unfused reference: conv, then per-channel
        // affine, then the activation applied on the host.
        let act_ref = |act: u32, z: f32| -> f32 {
            match act {
                1 => z.max(0.0),
                2 => z / (1.0 + (-z).exp()),
                3 => 1.0 / (1.0 + (-z).exp()),
                _ => z,
            }
        };
        for p in [cp(1, 8, 12, 12, 16, 3, 1, 1), cp(1, 13, 7, 5, 6, 1, 1, 0), cp(1, 6, 16, 16, 10, 3, 2, 1)] {
            for act in 0..4u32 {
                let mut s = 7u32 ^ p.cout as u32 ^ (act << 8);
                let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
                let w: Vec<f32> = (0..p.w_len()).map(|_| lcg(&mut s)).collect();
                let sb: Vec<f32> = (0..2 * p.cout).map(|_| lcg(&mut s)).collect();
                let mut y = vec![0.0f32; p.y_len()];
                conv2d_act(&p, &x, &w, &sb, &mut y, act);
                let yc = conv_ref(&p, &x, &w);
                let psz = p.ho * p.wo;
                let mut maxerr = 0.0f32;
                for co in 0..p.cout {
                    for j in 0..psz {
                        let z = yc[co * psz + j] * sb[2 * co] + sb[2 * co + 1];
                        let r = act_ref(act, z);
                        maxerr = maxerr.max((y[co * psz + j] - r).abs() / (r.abs() + 1e-3));
                    }
                }
                assert!(maxerr < 2e-3, "fused conv_act rel err {maxerr} for {p:?} act={act}");
            }
        }
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
