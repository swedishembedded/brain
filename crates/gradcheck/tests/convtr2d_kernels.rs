// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the `convtr2d` kernel family (ConvTranspose2d forward,
//! `_dx`, `_dw`), driven directly through `gpu_core` like `depth_kernels.rs` —
//! no model is built. Run with `BRAIN_DEVICE=cpu` for a GPU-free run.
//!
//! Three independent techniques, because each catches a different class of bug:
//!
//! 1. **Forward parity against a CPU oracle written in the SCATTER form.** The
//!    kernel is a *gather*: it inverts `ho = hi*stride - pad + kh*dilation` and
//!    walks the taps that land on each output. The oracle instead iterates the
//!    inputs and accumulates forward, which is the definition of
//!    ConvTranspose2d and shares no index algebra with the thing it checks. An
//!    oracle that re-uses the kernel's inverted map would prove nothing — it
//!    would agree with an off-by-one in the inversion.
//!
//! 2. **Adjointness**, which is exact rather than tolerant. The forward is
//!    *bilinear*: linear in `x` for fixed `w`, and linear in `w` for fixed `x`.
//!    So for every `x, w, dy`
//!
//!    ```text
//!    <A_w(x), dy> == <x, dx>      and      <B_x(w), dy> == <w, dw>
//!    ```
//!
//!    to fp32 round-off. A dropped edge tap, a transposed group index or an
//!    off-by-one window breaks this immediately, and unlike FD it needs no
//!    step size. `dw` ACCUMULATES, so its buffer is zeroed via `submit`'s
//!    clear list — the identity is only meaningful against a pre-zeroed dw.
//!
//! 3. **Central finite differences** of `L(x,w) = <convtr2d(x,w), dy>`, sampled
//!    rather than exhaustive. Adjointness is sharper but it is an identity
//!    between two *kernels*; FD is the independent check that the pair also
//!    matches the oracle's derivative.
//!
//! The oracle accumulates in **f64** deliberately, for both reasons: it is an
//! independent-precision reference for the f32 kernel, and the FD loss is a sum
//! over the whole output map whose sequential f32 error (~7e-4 here) is 20x the
//! `2*eps*dL` signal it has to resolve. An f32 oracle reports a broken `dx` for
//! a correct kernel — that is not a tolerance to loosen, it is a conditioning
//! bug in the check.
//!
//! Plus two structural probes that numbers alone do not catch: group isolation
//! (zeroing one input channel may only move the output channels of its own
//! group — a wrong group index still produces plausible numbers), and the
//! `output_padding` semantics — it widens Ho/Wo *without moving any interior
//! value*, and its extra band is NOT zero-fill (verified against PyTorch: with
//! stride > 1 the far-side `pad` crop hides output positions that genuinely
//! receive input, and out_pad un-crops them).
//!
//! WEIGHT LAYOUT under test is PyTorch's ConvTranspose2d convention
//! `w : [Cin, Cout/G, K, K]` — input channel OUTER. The grouped cases below are
//! what pin that down: with `Cin == Cout` and `G == 1` the transposed layout is
//! merely a permutation and a wrong reading still "works".

use data::rng::Lcg;
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("convtr2d", kernels::CONVTR2D),       // 0
    ("convtr2d_dx", kernels::CONVTR2D_DX), // 1
    ("convtr2d_dw", kernels::CONVTR2D_DW), // 2
];
const K_FWD: usize = 0;
const K_DX: usize = 1;
const K_DW: usize = 2;

/// Presence of the variable is the signal, never its value.
fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

// ---- shape ------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Shape {
    n: u32,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    groups: u32,
    /// PyTorch's `output_padding`. It only widens Ho/Wo; the kernels never see it.
    out_pad: u32,
}

impl Shape {
    /// `Lo = (L-1)*stride - 2*pad + dilation*(K-1) + out_pad + 1`, in i64 because
    /// the middle of that expression is legitimately negative for small inputs.
    fn out_dim(&self, l: u32) -> u32 {
        let v = (l as i64 - 1) * self.stride as i64 - 2 * self.pad as i64
            + self.dilation as i64 * (self.k as i64 - 1)
            + self.out_pad as i64
            + 1;
        assert!(v > 0, "degenerate output extent {v}");
        v as u32
    }
    fn ho(&self) -> u32 {
        self.out_dim(self.h)
    }
    fn wo(&self) -> u32 {
        self.out_dim(self.w)
    }
    fn xn(&self) -> usize {
        (self.n * self.cin * self.h * self.w) as usize
    }
    fn yn(&self) -> usize {
        (self.n * self.cout * self.ho() * self.wo()) as usize
    }
    fn wn(&self) -> usize {
        (self.cin * (self.cout / self.groups) * self.k * self.k) as usize
    }
    /// The 12-word ABI shared by all three kernels of the family.
    fn params(&self) -> [u32; 12] {
        [
            self.n,
            self.cin,
            self.h,
            self.w,
            self.cout,
            self.k,
            self.stride,
            self.pad,
            self.dilation,
            self.groups,
            self.ho(),
            self.wo(),
        ]
    }
}

// ---- CPU reference oracle (scatter form; matches wgsl/convtr2d.wgsl) ---------

/// ConvTranspose2d by definition: iterate the INPUT and accumulate forward.
/// Deliberately not the kernel's gather formulation, and deliberately f64 — see
/// the module doc.
fn convtr2d_ref(s: &Shape, x: &[f32], w: &[f32]) -> Vec<f64> {
    let (ho, wo) = (s.ho() as i64, s.wo() as i64);
    let cin_g = s.cin / s.groups;
    let cout_g = s.cout / s.groups;
    let mut y = vec![0f64; s.yn()];
    for n in 0..s.n {
        for ci in 0..s.cin {
            let g = ci / cin_g;
            for hi in 0..s.h {
                for wi in 0..s.w {
                    let xv = x[(((n * s.cin + ci) * s.h + hi) * s.w + wi) as usize];
                    for cl in 0..cout_g {
                        let co = g * cout_g + cl;
                        for kh in 0..s.k {
                            let oh = hi as i64 * s.stride as i64 + kh as i64 * s.dilation as i64
                                - s.pad as i64;
                            if oh < 0 || oh >= ho {
                                continue;
                            }
                            for kw in 0..s.k {
                                let ow = wi as i64 * s.stride as i64
                                    + kw as i64 * s.dilation as i64
                                    - s.pad as i64;
                                if ow < 0 || ow >= wo {
                                    continue;
                                }
                                let wv = w[(((ci * cout_g + cl) * s.k + kh) * s.k + kw) as usize];
                                let yi = ((n as i64 * s.cout as i64 + co as i64) * ho + oh) * wo + ow;
                                y[yi as usize] += xv as f64 * wv as f64;
                            }
                        }
                    }
                }
            }
        }
    }
    y
}

// ---- helpers ----------------------------------------------------------------
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// f32 kernel output vs the f64 oracle.
fn max_abs64(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs()).fold(0.0, f64::max)
}

fn amax64(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn fwd_gpu(gpu: &Gpu, s: &Shape, x: &[f32], w: &[f32]) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let yb = gpu.storage(s.yn() as u64);
    let st = gpu.step(K_FWD, &[&xb, &wb, &yb], &s.params(), (s.yn()) as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&yb, s.yn())
}

fn dx_gpu(gpu: &Gpu, s: &Shape, dy: &[f32], w: &[f32]) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let wb = gpu.storage_init("w", w);
    let dxb = gpu.storage(s.xn() as u64);
    let st = gpu.step(K_DX, &[&dyb, &wb, &dxb], &s.params(), (s.xn()) as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&dxb, s.xn())
}

/// `convtr2d_dw` ACCUMULATES, so the dw buffer goes in `submit`'s clear list.
fn dw_gpu(gpu: &Gpu, s: &Shape, dy: &[f32], x: &[f32]) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let xb = gpu.storage_init("x", x);
    let dwb = gpu.storage(s.wn() as u64);
    let st = gpu.step(K_DW, &[&dyb, &xb, &dwb], &s.params(), (s.wn()) as u32);
    gpu.submit(&[&dwb], &[st]);
    gpu.poll_wait();
    gpu.read(&dwb, s.wn())
}

/// Forward parity + both adjoint identities + sampled central differences.
fn check(tag: &str, s: Shape, seed: u64) {
    assert_eq!(s.cin % s.groups, 0, "{tag}: Cin must divide by groups");
    assert_eq!(s.cout % s.groups, 0, "{tag}: Cout must divide by groups");
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(seed);
    let x = r.vec(s.xn());
    let w = r.vec(s.wn());
    let dy = r.vec(s.yn()); // random cotangent

    // 1. forward parity vs the scatter oracle.
    let y_gpu = fwd_gpu(&gpu, &s, &x, &w);
    let y_ref = convtr2d_ref(&s, &x, &w);
    let tol = 1e-4 + 1e-4 * amax64(&y_ref);
    assert!(
        max_abs64(&y_gpu, &y_ref) < tol,
        "{tag}: forward mismatch {} (tol {tol})",
        max_abs64(&y_gpu, &y_ref)
    );

    let dx_a = dx_gpu(&gpu, &s, &dy, &w);
    let dw_a = dw_gpu(&gpu, &s, &dy, &x);

    // 2. adjointness (exact identities; the forward is bilinear).
    let lhs = dot(&y_gpu, &dy);
    let rx = dot(&x, &dx_a);
    let rw = dot(&w, &dw_a);
    let atol = 1e-4 * lhs.abs().max(rx.abs()).max(rw.abs()).max(1.0);
    assert!(
        (lhs - rx).abs() < atol,
        "{tag}: dx adjointness broken — <A(x),dy> = {lhs}, <x,dx> = {rx} (diff {:.3e})",
        (lhs - rx).abs()
    );
    assert!(
        (lhs - rw).abs() < atol,
        "{tag}: dw adjointness broken — <B(w),dy> = {lhs}, <w,dw> = {rw} (diff {:.3e})",
        (lhs - rw).abs()
    );

    // 3. central finite differences of L(x,w) = <ref(x,w), dy>, sampled.
    // eps is a power of two so `base[i] +/- eps` is exact in f32 wherever the
    // exponents allow, and the loss accumulates in f64 (see the module doc).
    let loss = |x: &[f32], w: &[f32]| -> f64 {
        convtr2d_ref(&s, x, w).iter().zip(&dy).map(|(a, b)| a * *b as f64).sum()
    };
    let eps = 1.0f32 / 1024.0;
    let fd = |base: &[f32], i: usize, f: &dyn Fn(&[f32]) -> f64| {
        let mut p = base.to_vec();
        p[i] = base[i] + eps;
        let lp = f(&p);
        p[i] = base[i] - eps;
        let lm = f(&p);
        ((lp - lm) / (2.0 * eps as f64)) as f32
    };
    let xn = s.xn();
    for i in (0..xn).step_by((xn / 17).max(1)) {
        let num = fd(&x, i, &|xx| loss(xx, &w));
        assert!(
            (num - dx_a[i]).abs() < 1e-3 + 1e-3 * num.abs().max(dx_a[i].abs()),
            "{tag}: dx[{i}] num={num} ana={}",
            dx_a[i]
        );
    }
    let wn = s.wn();
    for i in (0..wn).step_by((wn / 13).max(1)) {
        let num = fd(&w, i, &|ww| loss(&x, ww));
        assert!(
            (num - dw_a[i]).abs() < 1e-3 + 1e-3 * num.abs().max(dw_a[i].abs()),
            "{tag}: dw[{i}] num={num} ana={}",
            dw_a[i]
        );
    }
}

// ---- cases ------------------------------------------------------------------

/// Baseline: stride 1, no padding, no grouping. If this fails nothing else is
/// worth reading.
#[test]
fn convtr2d_plain() {
    if skip() {
        return;
    }
    check(
        "plain",
        Shape { n: 2, cin: 3, h: 5, w: 4, cout: 4, k: 3, stride: 1, pad: 0, dilation: 1, groups: 1, out_pad: 0 },
        1,
    );
}

/// SAM 2's mask decoder shape: K=2, stride=2 — an exact 2x upsample with
/// non-overlapping taps. Every output is hit by exactly one (kh,kw), so a
/// divisibility bug in the inverted map shows up as whole rows of zeros.
#[test]
fn convtr2d_upsample2x() {
    if skip() {
        return;
    }
    let s = Shape { n: 1, cin: 4, h: 4, w: 5, cout: 3, k: 2, stride: 2, pad: 0, dilation: 1, groups: 1, out_pad: 0 };
    assert_eq!((s.ho(), s.wo()), (8, 10));
    check("upsample2x", s, 2);
}

/// The classic "double the resolution" decoder config: K=3, stride=2, pad=1,
/// output_padding=1 gives Ho == 2H exactly, and exercises overlapping taps, a
/// cropped border and a padded tail in one shape.
#[test]
fn convtr2d_stride2_pad1_outpad1() {
    if skip() {
        return;
    }
    let s = Shape { n: 2, cin: 3, h: 4, w: 5, cout: 2, k: 3, stride: 2, pad: 1, dilation: 1, groups: 1, out_pad: 1 };
    assert_eq!((s.ho(), s.wo()), (8, 10));
    check("stride2_pad1_outpad1", s, 3);
}

/// Grouping with a non-trivial dilation, and deliberately Cin != Cout so the
/// `[Cin, Cout/G, K, K]` weight layout cannot be confused with `[Cout, Cin/G, K, K]`.
#[test]
fn convtr2d_grouped_dilated() {
    if skip() {
        return;
    }
    check(
        "grouped_dilated",
        Shape { n: 1, cin: 4, h: 5, w: 4, cout: 6, k: 3, stride: 2, pad: 1, dilation: 2, groups: 2, out_pad: 0 },
        4,
    );
}

/// Depthwise transposed conv: G == Cin == Cout, so w is [C, 1, K, K].
#[test]
fn convtr2d_depthwise() {
    if skip() {
        return;
    }
    check(
        "depthwise",
        Shape { n: 2, cin: 3, h: 4, w: 4, cout: 3, k: 3, stride: 2, pad: 1, dilation: 1, groups: 3, out_pad: 0 },
        5,
    );
}

/// A wrong group index still produces plausible numbers, so check it
/// structurally: zeroing input channel `ci` may change only the output channels
/// of `ci`'s own group, and must change all of them.
#[test]
fn convtr2d_group_isolation() {
    if skip() {
        return;
    }
    let s = Shape { n: 1, cin: 4, h: 4, w: 4, cout: 6, k: 3, stride: 2, pad: 1, dilation: 1, groups: 2, out_pad: 0 };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(7);
    let x = r.vec(s.xn());
    let w = r.vec(s.wn());
    let base = fwd_gpu(&gpu, &s, &x, &w);

    let cin_g = s.cin / s.groups;
    let cout_g = s.cout / s.groups;
    let (ho, wo) = (s.ho(), s.wo());
    for ci in 0..s.cin {
        let mut xz = x.clone();
        let off = (ci * s.h * s.w) as usize; // n == 1
        for v in &mut xz[off..off + (s.h * s.w) as usize] {
            *v = 0.0;
        }
        let y = fwd_gpu(&gpu, &s, &xz, &w);
        let g = ci / cin_g;
        for co in 0..s.cout {
            let lo = ((co * ho) * wo) as usize;
            let hi = lo + (ho * wo) as usize;
            let d = max_abs(&base[lo..hi], &y[lo..hi]);
            if co / cout_g == g {
                assert!(d > 1e-6, "ci={ci} co={co}: same group but output unchanged (d={d})");
            } else {
                assert!(d == 0.0, "ci={ci} co={co}: cross-group leak (d={d})");
            }
        }
    }
}

/// `convtr2d_dw` ACCUMULATES (`dw[i] = dw[i] + acc`) — a documented contract that
/// nothing else here exercises, because every other call site zeroes dw through
/// `submit`'s clear list. Dispatch it twice into the same buffer with a single
/// clear: the result must be exactly 2x the single-dispatch value. A `dw[i] = acc`
/// terminal write passes every other test in this file and fails only this one.
#[test]
fn convtr2d_dw_accumulates() {
    if skip() {
        return;
    }
    let s = Shape { n: 1, cin: 4, h: 4, w: 5, cout: 6, k: 3, stride: 2, pad: 1, dilation: 1, groups: 2, out_pad: 0 };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(13);
    let x = r.vec(s.xn());
    let dy = r.vec(s.yn());
    let once = dw_gpu(&gpu, &s, &dy, &x);

    let dyb = gpu.storage_init("dy", &dy);
    let xb = gpu.storage_init("x", &x);
    let dwb = gpu.storage(s.wn() as u64);
    let a = gpu.step(K_DW, &[&dyb, &xb, &dwb], &s.params(), s.wn() as u32);
    let b = gpu.step(K_DW, &[&dyb, &xb, &dwb], &s.params(), s.wn() as u32);
    gpu.submit(&[&dwb], &[a, b]);
    gpu.poll_wait();
    let twice = gpu.read(&dwb, s.wn());

    for i in 0..s.wn() {
        assert!(
            (twice[i] - 2.0 * once[i]).abs() <= 1e-5 * (2.0 * once[i]).abs().max(1e-3),
            "dw[{i}] does not accumulate: one pass {} two passes {}",
            once[i],
            twice[i]
        );
    }
}

/// `output_padding` semantics, which are the easiest thing in this family to get
/// plausibly wrong. It is NOT zero-padding: it widens Ho/Wo and the gather then
/// covers whatever taps land in the widened range. Verified against PyTorch —
/// `conv_transpose2d(stride=2, padding=1, output_padding=1)` on a 4x4 input has
/// a bottom row and right column that are *not* zero, because `padding` had
/// cropped away the real position fed by `hi = H-1, kh = K-1`.
///
/// The two properties that must hold:
///   a) the un-padded result is an exact prefix of the padded one (out_pad may
///      not move an interior value — a gather that folded out_pad into the
///      index math would shift everything);
///   b) the extra band is genuinely populated, so this test cannot be satisfied
///      by a kernel that zero-fills it.
#[test]
fn convtr2d_out_pad_extends_without_moving() {
    if skip() {
        return;
    }
    let base = Shape { n: 1, cin: 2, h: 4, w: 4, cout: 2, k: 3, stride: 2, pad: 1, dilation: 1, groups: 1, out_pad: 0 };
    let padded = Shape { out_pad: 1, ..base };
    let (bh, bw) = (base.ho(), base.wo());
    let (ph, pw) = (padded.ho(), padded.wo());
    assert_eq!((bh, bw), (7, 7));
    assert_eq!((ph, pw), (bh + 1, bw + 1));

    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(11);
    let x = r.vec(base.xn());
    let w = r.vec(base.wn());
    let yb = fwd_gpu(&gpu, &base, &x, &w);
    let yp = fwd_gpu(&gpu, &padded, &x, &w);
    let ref_p = convtr2d_ref(&padded, &x, &w);

    let mut tail_energy = 0.0f32;
    for co in 0..base.cout {
        for ho in 0..ph {
            for wo in 0..pw {
                let v = yp[(((co * ph) + ho) * pw + wo) as usize];
                if ho < bh && wo < bw {
                    // (a) exact prefix — same gather, so bit-identical.
                    let b = yb[(((co * bh) + ho) * bw + wo) as usize];
                    assert!(v == b, "co={co} ({ho},{wo}): out_pad moved an interior value {v} vs {b}");
                } else {
                    tail_energy += v.abs();
                }
            }
        }
    }
    // (b) the extra band carries real contributions, and matches the oracle.
    assert!(tail_energy > 1e-3, "out_pad band is all zeros — the gather is dropping its far-side taps");
    let tol = 1e-4 + 1e-4 * amax64(&ref_p);
    assert!(max_abs64(&yp, &ref_p) < tol, "out_pad forward mismatch {}", max_abs64(&yp, &ref_p));
}
