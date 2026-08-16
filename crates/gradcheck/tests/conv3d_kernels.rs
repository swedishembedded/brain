// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the `conv3d` kernel family (forward, `_dx`, `_dw`),
//! driven directly through `gpu_core` like `convtr2d_kernels.rs` - no model is
//! built. Run with `BRAIN_DEVICE=cpu` for a GPU-free run.
//!
//! The family exists for causal video convolutions: symmetric padding in space,
//! and a one-sided pad on the LOW (past) side of time so that an output frame
//! never reads a future input frame. **A wrong time pad is not a visible
//! failure** - it produces smooth, plausible output that has simply read the
//! frames it was supposed to predict, and no numeric check on a single tensor
//! can see it. So causality is tested structurally and separately from the
//! numbers, twice: once through the forward (perturbing input frame `tf` may
//! not move any output frame before `tf`) and once through `_dx` (the gradient
//! of output frame `tf` may not reach any input frame after it).
//!
//! Everything else follows the `convtr2d` gate, for the same reasons:
//!
//! 1. **Forward parity against a CPU oracle in a DIFFERENT formulation.** The
//!    kernel pads implicitly - it skips taps whose input coordinate falls
//!    outside the volume. The oracle instead materialises the padded volume and
//!    runs a dense unconditional convolution over it, which is literally what
//!    `F.pad(x, padding)` + `Conv3d(padding=0)` does upstream. An oracle that
//!    re-used the skip conditions would agree with an off-by-one in them.
//!
//! 2. **Adjointness**, exact rather than tolerant. The bias-free forward is
//!    bilinear, so for every `x, w, dy`
//!
//!    ```text
//!    <A_w(x), dy> == <x, dx>      and      <B_x(w), dy> == <w, dw>
//!    ```
//!
//!    to fp32 round-off. A dropped edge tap or a transposed group index breaks
//!    this immediately and it needs no step size. `dw` ACCUMULATES, so its
//!    buffer is zeroed through `submit`'s clear list.
//!
//! 3. **Central finite differences** of `L(x,w) = <conv3d(x,w), dy>`, sampled.
//!    Adjointness is sharper but it is an identity between two *kernels*; FD is
//!    the independent check that the pair also matches the oracle's derivative.
//!
//! The oracle accumulates in **f64** deliberately: it is an independent-precision
//! reference for the f32 kernel, and the FD loss is a sum over the whole output
//! volume whose sequential f32 error dwarfs the `2*eps*dL` signal it has to
//! resolve. An f32 oracle reports a broken `dx` for a correct kernel.
//!
//! `pt` throughout is the ALREADY-DOUBLED low pad, matching the kernels and
//! `dwconv3d`: `CausalConv3d(..., padding=1)` is `pt = 2`, not `pt = 1`.

use data::rng::Lcg;
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("conv3d", kernels::CONV3D),       // 0
    ("conv3d_dx", kernels::CONV3D_DX), // 1
    ("conv3d_dw", kernels::CONV3D_DW), // 2
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
    t: u32,
    h: u32,
    w: u32,
    cout: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    st: u32,
    sh: u32,
    sw: u32,
    /// Low-side (past) temporal pad, already doubled. The high side gets none.
    pt: u32,
    ph: u32,
    pw: u32,
    groups: u32,
}

impl Shape {
    fn to(&self) -> u32 {
        let v = (self.t as i64 + self.pt as i64 - self.kt as i64) / self.st as i64 + 1;
        assert!(v > 0, "degenerate temporal extent {v}");
        v as u32
    }
    fn ho(&self) -> u32 {
        let v = (self.h as i64 + 2 * self.ph as i64 - self.kh as i64) / self.sh as i64 + 1;
        assert!(v > 0, "degenerate height extent {v}");
        v as u32
    }
    fn wo(&self) -> u32 {
        let v = (self.w as i64 + 2 * self.pw as i64 - self.kw as i64) / self.sw as i64 + 1;
        assert!(v > 0, "degenerate width extent {v}");
        v as u32
    }
    fn xn(&self) -> usize {
        (self.n * self.cin * self.t * self.h * self.w) as usize
    }
    fn yn(&self) -> usize {
        (self.n * self.cout * self.to() * self.ho() * self.wo()) as usize
    }
    fn wn(&self) -> usize {
        (self.cout * (self.cin / self.groups) * self.kt * self.kh * self.kw) as usize
    }
    /// The 19-word ABI shared by all three kernels of the family.
    fn params(&self) -> [u32; 19] {
        [
            self.n,
            self.cin,
            self.t,
            self.h,
            self.w,
            self.cout,
            self.kt,
            self.kh,
            self.kw,
            self.st,
            self.sh,
            self.sw,
            self.pt,
            self.ph,
            self.pw,
            self.groups,
            self.to(),
            self.ho(),
            self.wo(),
        ]
    }
}

// ---- CPU reference oracle (explicit pad, dense taps) -------------------------

/// `F.pad(x, (pw,pw,ph,ph,pt,0))` then a padding-free dense Conv3d - the
/// upstream `CausalConv3d` formulation, deliberately NOT the kernel's
/// skip-out-of-range one. f64 (see the module doc).
fn conv3d_ref(s: &Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f64> {
    let (tp, hp, wp) = (s.t + s.pt, s.h + 2 * s.ph, s.w + 2 * s.pw);
    // The padded input volume, materialised.
    let mut xp = vec![0f64; (s.n * s.cin * tp * hp * wp) as usize];
    for n in 0..s.n {
        for ci in 0..s.cin {
            for ti in 0..s.t {
                for hi in 0..s.h {
                    for wi in 0..s.w {
                        let src = (((n * s.cin + ci) * s.t + ti) * s.h + hi) * s.w + wi;
                        let dst = (((n * s.cin + ci) * tp + ti + s.pt) * hp + hi + s.ph) * wp
                            + wi
                            + s.pw;
                        xp[dst as usize] = x[src as usize] as f64;
                    }
                }
            }
        }
    }
    let (to, ho, wo) = (s.to(), s.ho(), s.wo());
    let cin_g = s.cin / s.groups;
    let cout_g = s.cout / s.groups;
    let mut y = vec![0f64; s.yn()];
    for n in 0..s.n {
        for co in 0..s.cout {
            let ci0 = (co / cout_g) * cin_g;
            for ot in 0..to {
                for oh in 0..ho {
                    for ow in 0..wo {
                        let mut acc = bias[co as usize] as f64;
                        for cl in 0..cin_g {
                            let ci = ci0 + cl;
                            for kt in 0..s.kt {
                                for kh in 0..s.kh {
                                    for kw in 0..s.kw {
                                        let xi = (((n * s.cin + ci) * tp + ot * s.st + kt) * hp
                                            + oh * s.sh
                                            + kh)
                                            * wp
                                            + ow * s.sw
                                            + kw;
                                        let wi = (((co * cin_g + cl) * s.kt + kt) * s.kh + kh)
                                            * s.kw
                                            + kw;
                                        acc += xp[xi as usize] * w[wi as usize] as f64;
                                    }
                                }
                            }
                        }
                        let yi = (((n * s.cout + co) * to + ot) * ho + oh) * wo + ow;
                        y[yi as usize] = acc;
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

fn max_abs64(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs()).fold(0.0, f64::max)
}

fn amax64(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn fwd_gpu(gpu: &Gpu, s: &Shape, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let bb = gpu.storage_init("bias", bias);
    let yb = gpu.storage(s.yn() as u64);
    let st = gpu.step(K_FWD, &[&xb, &wb, &bb, &yb], &s.params(), s.yn() as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&yb, s.yn())
}

fn dx_gpu(gpu: &Gpu, s: &Shape, dy: &[f32], w: &[f32]) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let wb = gpu.storage_init("w", w);
    let dxb = gpu.storage(s.xn() as u64);
    let st = gpu.step(K_DX, &[&dyb, &wb, &dxb], &s.params(), s.xn() as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&dxb, s.xn())
}

/// `conv3d_dw` ACCUMULATES, so the dw buffer goes in `submit`'s clear list.
fn dw_gpu(gpu: &Gpu, s: &Shape, dy: &[f32], x: &[f32]) -> Vec<f32> {
    let dyb = gpu.storage_init("dy", dy);
    let xb = gpu.storage_init("x", x);
    let dwb = gpu.storage(s.wn() as u64);
    let st = gpu.step(K_DW, &[&dyb, &xb, &dwb], &s.params(), s.wn() as u32);
    gpu.submit(&[&dwb], &[st]);
    gpu.poll_wait();
    gpu.read(&dwb, s.wn())
}

/// Forward parity (with bias) + both adjoint identities + sampled central
/// differences.
fn check(tag: &str, s: Shape, seed: u64) {
    assert_eq!(s.cin % s.groups, 0, "{tag}: Cin must divide by groups");
    assert_eq!(s.cout % s.groups, 0, "{tag}: Cout must divide by groups");
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(seed);
    let x = r.vec(s.xn());
    let w = r.vec(s.wn());
    let bias = r.vec(s.cout as usize);
    let dy = r.vec(s.yn()); // random cotangent
    let zero_bias = vec![0f32; s.cout as usize];

    // 1. forward parity vs the explicit-pad oracle, bias included.
    let y_gpu = fwd_gpu(&gpu, &s, &x, &w, &bias);
    let y_ref = conv3d_ref(&s, &x, &w, &bias);
    let tol = 1e-4 + 1e-4 * amax64(&y_ref);
    assert!(
        max_abs64(&y_gpu, &y_ref) < tol,
        "{tag}: forward mismatch {} (tol {tol})",
        max_abs64(&y_gpu, &y_ref)
    );

    // 2. adjointness. Only the bias-free map is bilinear, so the identity is
    //    stated against a zero-bias forward.
    let y0 = fwd_gpu(&gpu, &s, &x, &w, &zero_bias);
    let dx_a = dx_gpu(&gpu, &s, &dy, &w);
    let dw_a = dw_gpu(&gpu, &s, &dy, &x);
    let lhs = dot(&y0, &dy);
    let rx = dot(&x, &dx_a);
    let rw = dot(&w, &dw_a);
    let atol = 1e-4 * lhs.abs().max(rx.abs()).max(rw.abs()).max(1.0);
    assert!(
        (lhs - rx).abs() < atol,
        "{tag}: dx adjointness broken - <A(x),dy> = {lhs}, <x,dx> = {rx} (diff {:.3e})",
        (lhs - rx).abs()
    );
    assert!(
        (lhs - rw).abs() < atol,
        "{tag}: dw adjointness broken - <B(w),dy> = {lhs}, <w,dw> = {rw} (diff {:.3e})",
        (lhs - rw).abs()
    );

    // 3. central finite differences of L(x,w) = <ref(x,w), dy>, sampled. eps is
    //    a power of two so `base[i] +/- eps` is exact in f32 wherever the
    //    exponents allow, and the loss accumulates in f64 (see the module doc).
    let loss = |x: &[f32], w: &[f32]| -> f64 {
        conv3d_ref(&s, x, w, &zero_bias).iter().zip(&dy).map(|(a, b)| a * *b as f64).sum()
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

/// The smallest shape whose answer can be written down by hand: one channel,
/// one pixel, K=(3,1,1), pt=2, so the padded time axis is `[0, 0, x0, x1, x2]`
/// and
///     y0 = x0*w2 + b,  y1 = x0*w1 + x1*w2 + b,  y2 = x0*w0 + x1*w1 + x2*w2 + b.
/// `y0` is where the causal convention is visible as arithmetic: it sees only
/// the past, and the two taps that would reach forward land on the pad.
#[test]
fn conv3d_hand_computed_causal_line() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 1,
        cin: 1,
        t: 3,
        h: 1,
        w: 1,
        cout: 1,
        kt: 3,
        kh: 1,
        kw: 1,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 0,
        pw: 0,
        groups: 1,
    };
    assert_eq!((s.to(), s.ho(), s.wo()), (3, 1, 1));
    let x = [2.0f32, -3.0, 5.0];
    let w = [0.5f32, -1.5, 0.25];
    let b = [0.75f32];
    let want = [
        x[0] * w[2] + b[0],
        x[0] * w[1] + x[1] * w[2] + b[0],
        x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + b[0],
    ];
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let y = fwd_gpu(&gpu, &s, &x, &w, &b);
    for i in 0..3 {
        assert!((y[i] - want[i]).abs() < 1e-5, "y[{i}] = {} want {}", y[i], want[i]);
    }
}

/// Wan's `ResidualBlock` shape: `CausalConv3d(c_in, c_out, 3, padding=1)`, i.e.
/// a genuine (3,3,3) kernel with pt = 2*1 = 2 and symmetric spatial pad 1. The
/// output volume must match the input volume exactly.
#[test]
fn conv3d_causal_3x3x3() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 2,
        cin: 3,
        t: 5,
        h: 4,
        w: 4,
        cout: 4,
        kt: 3,
        kh: 3,
        kw: 3,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 1,
        pw: 1,
        groups: 1,
    };
    assert_eq!((s.to(), s.ho(), s.wo()), (s.t, s.h, s.w));
    check("causal_3x3x3", s, 1);
}

/// `CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))` - the temporal-only conv
/// the VAE's `upsample3d` uses. Per-axis kernel extents, so a kernel that
/// assumed a cubic K would index `wt` wrongly here.
#[test]
fn conv3d_temporal_only_3x1x1() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 2,
        cin: 3,
        t: 6,
        h: 3,
        w: 4,
        cout: 6,
        kt: 3,
        kh: 1,
        kw: 1,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 0,
        pw: 0,
        groups: 1,
    };
    assert_eq!((s.to(), s.ho(), s.wo()), (s.t, s.h, s.w));
    check("temporal_only_3x1x1", s, 2);
}

/// `CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1))` - the VAE's temporal
/// downsample. Per-axis stride with the spatial axes left at 1, and pt = 0
/// (upstream leaves `padding` at its default here), so To shrinks.
#[test]
fn conv3d_temporal_stride2() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 1,
        cin: 4,
        t: 7,
        h: 3,
        w: 3,
        cout: 4,
        kt: 3,
        kh: 1,
        kw: 1,
        st: 2,
        sh: 1,
        sw: 1,
        pt: 0,
        ph: 0,
        pw: 0,
        groups: 1,
    };
    assert_eq!((s.to(), s.ho(), s.wo()), (3, 3, 3));
    check("temporal_stride2", s, 3);
}

/// Stride 2 in time *with* a causal low pad, which is where `_dx`'s
/// divisibility test earns its keep: only every second input frame is reachable
/// from any output frame, and the offset of that lattice is set by `pt`.
#[test]
fn conv3d_temporal_stride2_padded() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 2,
        cin: 2,
        t: 7,
        h: 3,
        w: 3,
        cout: 3,
        kt: 3,
        kh: 3,
        kw: 3,
        st: 2,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 1,
        pw: 1,
        groups: 1,
    };
    assert_eq!((s.to(), s.ho(), s.wo()), (4, 3, 3));
    check("temporal_stride2_padded", s, 4);
}

/// Grouping, with Cin != Cout so that a `[Cout, Cin/G, KT, KH, KW]` weight
/// cannot be confused with any other reading, plus spatial stride 2 to exercise
/// all three strides at once.
#[test]
fn conv3d_grouped_strided() {
    if skip() {
        return;
    }
    check(
        "grouped_strided",
        Shape {
            n: 1,
            cin: 4,
            t: 5,
            h: 5,
            w: 4,
            cout: 6,
            kt: 3,
            kh: 3,
            kw: 3,
            st: 1,
            sh: 2,
            sw: 2,
            pt: 2,
            ph: 1,
            pw: 1,
            groups: 2,
        },
        5,
    );
}

/// Depthwise: G == Cin == Cout, so the weight is `[C, 1, KT, KH, KW]`. Overlaps
/// `dwconv3d`'s job on purpose - the two must agree on what a per-channel 3D
/// conv means, and this is the cheap half of that.
#[test]
fn conv3d_depthwise() {
    if skip() {
        return;
    }
    check(
        "depthwise",
        Shape {
            n: 2,
            cin: 3,
            t: 4,
            h: 4,
            w: 4,
            cout: 3,
            kt: 3,
            kh: 3,
            kw: 3,
            st: 1,
            sh: 1,
            sw: 1,
            pt: 2,
            ph: 1,
            pw: 1,
            groups: 3,
        },
        6,
    );
}

// ---- causality --------------------------------------------------------------

/// THE test this family exists for. With KT=3, pt=2, st=1 output frame `ot`
/// reads input frames `[ot-2, ot]`, so perturbing input frame `tf` must move
/// exactly the output frames `tf ..= tf+2` and leave every other one
/// BIT-identical. The `ot < tf` half is causality (a symmetric temporal pad
/// passes every numeric test above - it is a perfectly self-consistent
/// convolution - and fails only here); the `ot > tf+2` half pins the receptive
/// field, so the first half cannot be satisfied by a kernel that simply reads
/// too little.
///
/// Bit-identical, not merely close: the forward is a deterministic sum in a
/// fixed order, so an output frame that genuinely does not read frame `tf`
/// re-computes to the same bits regardless of what `tf` holds.
#[test]
fn conv3d_forward_never_reads_the_future() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 1,
        cin: 2,
        t: 6,
        h: 3,
        w: 3,
        cout: 2,
        kt: 3,
        kh: 3,
        kw: 3,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 1,
        pw: 1,
        groups: 1,
    };
    assert_eq!(s.to(), s.t);
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(21);
    let x = r.vec(s.xn());
    let w = r.vec(s.wn());
    let bias = r.vec(s.cout as usize);
    let base = fwd_gpu(&gpu, &s, &x, &w, &bias);
    let frame = (s.h * s.w) as usize;

    for tf in 0..s.t {
        let mut xp = x.clone();
        for ci in 0..s.cin {
            let off = ((ci * s.t + tf) * s.h * s.w) as usize; // n == 1
            for (k, v) in xp[off..off + frame].iter_mut().enumerate() {
                *v += 1.0 + k as f32;
            }
        }
        let y = fwd_gpu(&gpu, &s, &xp, &w, &bias);
        for co in 0..s.cout {
            for ot in 0..s.to() {
                let lo = ((co * s.to() + ot) * s.h * s.w) as usize;
                let same = base[lo..lo + frame] == y[lo..lo + frame];
                let reads = ot >= tf && ot < tf + s.kt;
                if ot < tf {
                    assert!(
                        same,
                        "output frame {ot} changed when input frame {tf} did: the conv is \
                         reading the FUTURE (co={co})"
                    );
                } else if reads {
                    assert!(
                        !same,
                        "output frame {ot} did not react to input frame {tf} (co={co}) - the \
                         receptive field is truncated, so this test proves nothing"
                    );
                } else {
                    assert!(
                        same,
                        "output frame {ot} reacted to input frame {tf} (co={co}), which is \
                         outside the {}-frame receptive field",
                        s.kt
                    );
                }
            }
        }
    }
}

/// The same property through the backward: seed a cotangent on exactly one
/// output frame `tf` and require the input gradient to vanish EXACTLY on every
/// input frame outside `[tf-2, tf]` - zero after `tf` is causality, zero before
/// `tf-2` is the receptive field. `_dx` inverts the index map independently of
/// the forward, so a symmetric pad there would leak the future into training
/// while the forward stayed correct - a bug no forward parity test can see.
#[test]
fn conv3d_dx_never_reaches_the_future() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 1,
        cin: 2,
        t: 6,
        h: 3,
        w: 3,
        cout: 2,
        kt: 3,
        kh: 3,
        kw: 3,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 1,
        pw: 1,
        groups: 1,
    };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(22);
    let w = r.vec(s.wn());
    let frame = (s.h * s.w) as usize;

    for tf in 0..s.to() {
        let mut dy = vec![0f32; s.yn()];
        for co in 0..s.cout {
            let off = ((co * s.to() + tf) * s.h * s.w) as usize;
            for (k, v) in dy[off..off + frame].iter_mut().enumerate() {
                *v = 1.0 + 0.5 * k as f32;
            }
        }
        let dx = dx_gpu(&gpu, &s, &dy, &w);
        for ci in 0..s.cin {
            for ti in 0..s.t {
                let lo = ((ci * s.t + ti) * s.h * s.w) as usize;
                let energy: f32 = dx[lo..lo + frame].iter().map(|v| v.abs()).sum();
                let reached = ti <= tf && ti + s.kt > tf;
                if ti > tf {
                    assert!(
                        energy == 0.0,
                        "dx leaked to input frame {ti} from output frame {tf} (ci={ci}, \
                         energy {energy}) - the adjoint is acausal"
                    );
                } else if reached {
                    assert!(
                        energy > 0.0,
                        "no gradient reached input frame {ti} from output frame {tf} \
                         (ci={ci}) - the adjoint is dropping taps"
                    );
                } else {
                    assert!(
                        energy == 0.0,
                        "dx reached input frame {ti} from output frame {tf} (ci={ci}), \
                         outside the {}-frame receptive field",
                        s.kt
                    );
                }
            }
        }
    }
}

/// A wrong group index still produces plausible numbers, so check it
/// structurally: zeroing input channel `ci` may change only the output channels
/// of `ci`'s own group, and must change all of them.
#[test]
fn conv3d_group_isolation() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 1,
        cin: 4,
        t: 4,
        h: 3,
        w: 3,
        cout: 6,
        kt: 3,
        kh: 3,
        kw: 3,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 1,
        pw: 1,
        groups: 2,
    };
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut r = Lcg::new(7);
    let x = r.vec(s.xn());
    let w = r.vec(s.wn());
    let bias = vec![0f32; s.cout as usize];
    let base = fwd_gpu(&gpu, &s, &x, &w, &bias);

    let cin_g = s.cin / s.groups;
    let cout_g = s.cout / s.groups;
    let vol = (s.to() * s.ho() * s.wo()) as usize;
    let xvol = (s.t * s.h * s.w) as usize;
    for ci in 0..s.cin {
        let mut xz = x.clone();
        let off = (ci as usize) * xvol; // n == 1
        for v in &mut xz[off..off + xvol] {
            *v = 0.0;
        }
        let y = fwd_gpu(&gpu, &s, &xz, &w, &bias);
        let g = ci / cin_g;
        for co in 0..s.cout {
            let lo = (co as usize) * vol;
            let d = base[lo..lo + vol]
                .iter()
                .zip(&y[lo..lo + vol])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if co / cout_g == g {
                assert!(d > 1e-6, "ci={ci} co={co}: same group but output unchanged (d={d})");
            } else {
                assert!(d == 0.0, "ci={ci} co={co}: cross-group leak (d={d})");
            }
        }
    }
}

/// `conv3d_dw` ACCUMULATES (`dw[i] = dw[i] + acc`) - a documented contract that
/// nothing else here exercises, because every other call site zeroes dw through
/// `submit`'s clear list. Dispatch it twice into the same buffer with a single
/// clear: the result must be exactly 2x the single-dispatch value. A `dw[i] = acc`
/// terminal write passes every other test in this file and fails only this one.
#[test]
fn conv3d_dw_accumulates() {
    if skip() {
        return;
    }
    let s = Shape {
        n: 1,
        cin: 4,
        t: 5,
        h: 3,
        w: 4,
        cout: 6,
        kt: 3,
        kh: 3,
        kw: 3,
        st: 1,
        sh: 1,
        sw: 1,
        pt: 2,
        ph: 1,
        pw: 1,
        groups: 2,
    };
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
