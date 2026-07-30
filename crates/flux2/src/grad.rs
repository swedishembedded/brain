// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host reference forward + analytic backward for ONE FLUX.2 block (double and
//! single stream), the units the whole DiT training step is built from. This is
//! the correctness anchor: a finite-difference gradcheck (`tests/block_grad.rs`)
//! gates the analytic gradients, exactly as brain gates every hand-written
//! backward.
//!
//! The forward mirrors [`crate::model::Flux2Model::forward`] op-for-op, but in
//! the **unfolded** modulation form: where the device path folds the global
//! modulation into LayerNorm affine params (`gamma = 1+scale`, `beta = shift`),
//! this reference computes `y = (1+scale)·LN_noaffine(x) + shift` explicitly and
//! differentiates that — so the backward yields per-site `d_shift/d_scale/
//! d_gate` gradients that [`crate::modelgrad`] routes through the three global
//! modulation linears back into the conditioning vector.
//!
//! Exactness contract (kept in lockstep with `model.rs` / the WGSL kernels):
//! affine-free LayerNorm with eps 1e-6 (population variance); per-head
//! QK-RMSNorm over `head_dim` with eps 1e-6 and a learnable scale; interleaved
//! RoPE on adjacent channel pairs from per-token host tables; joint
//! bidirectional attention (txt rows first) scaled `1/√head_dim`; SwiGLU with
//! the silu-gated half FIRST; per-channel gated residuals `y = x + gate⊙h`.
//!
//! Generic over [`Fp`] (`f64`/`f32`): the `f64` instantiation is the
//! finite-difference gradcheck oracle (AGENTS.md exception 1 — the math here is
//! re-derived independently of the device kernels it anchors, in f64, and gated
//! by FD, which shares no code with it); the **same code** instantiated at
//! `f32` is the host training path ([`crate::finetune`]), so there is exactly
//! one implementation of the training math. The only per-type divergence is the
//! [`Fp::matvec`] hot-loop hook: `f32` routes through the sanctioned
//! `model::hostmath::matvec_par`, `f64` (oracle-only sizes) stays serial.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Scalar the reference math is generic over. See the module doc for why both
/// instantiations exist and why this is not a duplicate implementation.
pub trait Fp:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn fr(v: f64) -> Self;
    fn f64(self) -> f64;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    /// `y[o] = Σ_i w[o·inn+i]·x[i]` — the hot inner product of every linear.
    fn matvec(w: &[Self], x: &[Self], out: usize, inn: usize) -> Vec<Self>;
}

impl Fp for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    fn fr(v: f64) -> f64 {
        v
    }
    fn f64(self) -> f64 {
        self
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn matvec(w: &[f64], x: &[f64], out: usize, inn: usize) -> Vec<f64> {
        (0..out).map(|o| w[o * inn..o * inn + inn].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }
}

impl Fp for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    fn fr(v: f64) -> f32 {
        v as f32
    }
    fn f64(self) -> f64 {
        self as f64
    }
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    fn matvec(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
        model::hostmath::matvec_par(w, x, out, inn)
    }
}

/// LayerNorm / QK-RMSNorm epsilon — 1e-6 in every FLUX.2 variant (`model::EPS`).
const EPS: f64 = 1e-6;

/// Shape of the block being differentiated. `nt` text rows first in the joint
/// slab, then `ni` image rows; `d = nh·hd`; `mlp` is the SwiGLU inner width.
#[derive(Clone, Copy)]
pub struct Dims {
    pub nt: usize,
    pub ni: usize,
    pub d: usize,
    pub nh: usize,
    pub mlp: usize,
}

impl Dims {
    pub fn n(&self) -> usize {
        self.nt + self.ni
    }
    pub fn hd(&self) -> usize {
        self.d / self.nh
    }
}

/// One modulation site: per-channel `[d]` shift/scale/gate from the global
/// modulation linears (chunk order in the checkpoint: shift, scale, gate).
/// The final-layer site has an empty `gate`.
#[derive(Clone)]
pub struct Mod<T> {
    pub shift: Vec<T>,
    pub scale: Vec<T>,
    pub gate: Vec<T>,
}

/// Gradients w.r.t. one modulation site (same layout as [`Mod`]).
#[derive(Clone)]
pub struct ModGrad<T> {
    pub shift: Vec<T>,
    pub scale: Vec<T>,
    pub gate: Vec<T>,
}

impl<T: Fp> ModGrad<T> {
    pub fn zeros(d: usize) -> ModGrad<T> {
        ModGrad { shift: vec![T::ZERO; d], scale: vec![T::ZERO; d], gate: vec![T::ZERO; d] }
    }
    pub fn add(&mut self, o: &ModGrad<T>) {
        for (a, b) in self.shift.iter_mut().zip(&o.shift) {
            *a += *b;
        }
        for (a, b) in self.scale.iter_mut().zip(&o.scale) {
            *a += *b;
        }
        for (a, b) in self.gate.iter_mut().zip(&o.gate) {
            *a += *b;
        }
    }
}

/// One attention/MLP weight set (a double block holds two: img and txt). All
/// linears `[out, in]` row-major, bias-free; `nq`/`nk` are the QK-RMSNorm
/// scales `[hd]`. These are the SPLIT projections (the checkpoint fuses
/// qkv/mlp.0 — [`crate::modelgrad::ModelWeights::from_tensors`] splits them the
/// same way `model.rs` does at build time).
#[derive(Clone)]
pub struct StreamW<T> {
    pub wq: Vec<T>,
    pub wk: Vec<T>,
    pub wv: Vec<T>,
    pub nq: Vec<T>,
    pub nk: Vec<T>,
    pub wo: Vec<T>,
    pub w1: Vec<T>,
    pub w3: Vec<T>,
    pub w2: Vec<T>,
}

/// Gradients mirroring [`StreamW`].
#[derive(Clone)]
pub struct StreamG<T> {
    pub wq: Vec<T>,
    pub wk: Vec<T>,
    pub wv: Vec<T>,
    pub nq: Vec<T>,
    pub nk: Vec<T>,
    pub wo: Vec<T>,
    pub w1: Vec<T>,
    pub w3: Vec<T>,
    pub w2: Vec<T>,
}

impl<T: Fp> StreamG<T> {
    fn zeros(dm: &Dims) -> StreamG<T> {
        let (d, hd, mlp) = (dm.d, dm.hd(), dm.mlp);
        StreamG {
            wq: vec![T::ZERO; d * d],
            wk: vec![T::ZERO; d * d],
            wv: vec![T::ZERO; d * d],
            nq: vec![T::ZERO; hd],
            nk: vec![T::ZERO; hd],
            wo: vec![T::ZERO; d * d],
            w1: vec![T::ZERO; mlp * d],
            w3: vec![T::ZERO; mlp * d],
            w2: vec![T::ZERO; d * mlp],
        }
    }
}

/// One double block's weights (separate img/txt streams, joint attention).
#[derive(Clone)]
pub struct DoubleW<T> {
    pub img: StreamW<T>,
    pub txt: StreamW<T>,
}

/// The four modulation sites a double block reads (shared across ALL double
/// blocks — the modulation is global, so `modelgrad` accumulates the site
/// grads over the block stack).
#[derive(Clone)]
pub struct DoubleMods<T> {
    pub img1: Mod<T>,
    pub img2: Mod<T>,
    pub txt1: Mod<T>,
    pub txt2: Mod<T>,
}

/// One single block's weights. `wo_a`/`wo_b` are the linear2 column split:
/// `out = ctx @ wo_aᵀ + swiglu @ wo_bᵀ`.
#[derive(Clone)]
pub struct SingleW<T> {
    pub wq: Vec<T>,
    pub wk: Vec<T>,
    pub wv: Vec<T>,
    pub nq: Vec<T>,
    pub nk: Vec<T>,
    pub w1: Vec<T>,
    pub w3: Vec<T>,
    pub wo_a: Vec<T>,
    pub wo_b: Vec<T>,
}

/// Gradients mirroring [`SingleW`], plus `dx` and the block's modulation-site
/// contribution.
#[derive(Clone)]
pub struct SingleGrads<T> {
    pub wq: Vec<T>,
    pub wk: Vec<T>,
    pub wv: Vec<T>,
    pub nq: Vec<T>,
    pub nk: Vec<T>,
    pub w1: Vec<T>,
    pub w3: Vec<T>,
    pub wo_a: Vec<T>,
    pub wo_b: Vec<T>,
    pub m: ModGrad<T>,
    pub dx: Vec<T>,
}

/// Gradients for a double block: both streams, the four site contributions,
/// and `dx` to the previous block.
#[derive(Clone)]
pub struct DoubleGrads<T> {
    pub img: StreamG<T>,
    pub txt: StreamG<T>,
    pub img1: ModGrad<T>,
    pub img2: ModGrad<T>,
    pub txt1: ModGrad<T>,
    pub txt2: ModGrad<T>,
    pub dx: Vec<T>,
}

// ---- primitive fwd/bwd (host) ----

pub(crate) fn sigmoid<T: Fp>(x: T) -> T {
    T::ONE / (T::ONE + (-x).exp())
}

pub(crate) fn silu<T: Fp>(x: T) -> T {
    x * sigmoid(x)
}

/// d silu(x) / dx = σ(x) + x·σ(x)·(1−σ(x)).
pub(crate) fn dsilu<T: Fp>(x: T) -> T {
    let s = sigmoid(x);
    s + x * s * (T::ONE - s)
}

/// `y = x @ wᵀ`, `x:[rows,inn]`, `w:[out,inn]` → `y:[rows,out]` (nn.Linear,
/// bias-free — matches `matmul.wgsl`).
pub fn linear<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize) -> Vec<T> {
    let mut y = Vec::with_capacity(rows * out);
    for r in 0..rows {
        y.extend(T::matvec(w, &x[r * inn..(r + 1) * inn], out, inn));
    }
    y
}

fn transpose<T: Fp>(w: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

/// Linear backward: `dx = dy @ w`, `dw = dyᵀ @ x`. Both products are routed
/// through [`Fp::matvec`] on pre-transposed operands so the f32 trainer keeps
/// its rows parallel; the transposes are cheap next to the O(rows·out·in) work.
pub fn linear_bwd<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize, dy: &[T]) -> (Vec<T>, Vec<T>) {
    let wt = transpose(w, out, inn); // [inn, out]
    let mut dx = Vec::with_capacity(rows * inn);
    for r in 0..rows {
        dx.extend(T::matvec(&wt, &dy[r * out..(r + 1) * out], inn, out));
    }
    let xt = transpose(x, rows, inn); // [inn, rows]
    let mut dw = vec![T::ZERO; out * inn];
    let mut dyc = vec![T::ZERO; rows];
    for o in 0..out {
        for r in 0..rows {
            dyc[r] = dy[r * out + o];
        }
        let row = T::matvec(&xt, &dyc, inn, rows);
        dw[o * inn..(o + 1) * inn].copy_from_slice(&row);
    }
    (dx, dw)
}

/// Affine-free LayerNorm over the last `d` of `[rows,d]` (population variance,
/// eps 1e-6 — matches `layernorm.wgsl` with gamma=1, beta=0). Returns
/// `(xhat, inv[rows])`.
pub fn layernorm<T: Fp>(x: &[T], rows: usize, d: usize) -> (Vec<T>, Vec<T>) {
    let mut y = vec![T::ZERO; rows * d];
    let mut inv = vec![T::ZERO; rows];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let mut mean = T::ZERO;
        for &v in xr {
            mean += v;
        }
        mean = mean / dn;
        let mut var = T::ZERO;
        for &v in xr {
            var += (v - mean) * (v - mean);
        }
        var = var / dn;
        let iv = T::ONE / (var + T::fr(EPS)).sqrt();
        inv[r] = iv;
        for c in 0..d {
            y[r * d + c] = (xr[c] - mean) * iv;
        }
    }
    (y, inv)
}

/// Affine-free LayerNorm backward from the cached `xhat`:
/// `dx = inv·(dxhat − mean(dxhat) − xhat·mean(dxhat⊙xhat))`.
pub fn layernorm_bwd<T: Fp>(xhat: &[T], inv: &[T], rows: usize, d: usize, dxhat: &[T]) -> Vec<T> {
    let mut dx = vec![T::ZERO; rows * d];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let (mut mdy, mut mdyx) = (T::ZERO, T::ZERO);
        for c in 0..d {
            mdy += dxhat[r * d + c];
            mdyx += dxhat[r * d + c] * xhat[r * d + c];
        }
        mdy = mdy / dn;
        mdyx = mdyx / dn;
        for c in 0..d {
            dx[r * d + c] = inv[r] * (dxhat[r * d + c] - mdy - xhat[r * d + c] * mdyx);
        }
    }
    dx
}

/// RMSNorm over the last `d` of `[rows,d]` with eps 1e-6 (matches
/// `rmsnorm_eps.wgsl`): `y = w ⊙ x·inv`, `inv = 1/√(mean(x²)+eps)`. Returns
/// `(y, inv[rows])`. Used per-head (`d = head_dim`) for QK-norm.
pub fn rmsnorm<T: Fp>(x: &[T], rows: usize, d: usize, w: &[T]) -> (Vec<T>, Vec<T>) {
    let mut y = vec![T::ZERO; rows * d];
    let mut inv = vec![T::ZERO; rows];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let mut ss = T::ZERO;
        for &v in xr {
            ss += v * v;
        }
        let iv = T::ONE / (ss / dn + T::fr(EPS)).sqrt();
        inv[r] = iv;
        for c in 0..d {
            y[r * d + c] = w[c] * xr[c] * iv;
        }
    }
    (y, inv)
}

/// RMSNorm backward. Accumulates the scale grad into `dw` (len `d`), returns
/// `dx`: `dx[c] = inv·g[c] − x[c]·inv³·(1/d)·Σ_k g[k]·x[k]`, `g[c]=w[c]·dy[c]`.
pub fn rmsnorm_bwd<T: Fp>(x: &[T], rows: usize, d: usize, w: &[T], inv: &[T], dy: &[T], dw: &mut [T]) -> Vec<T> {
    let mut dx = vec![T::ZERO; rows * d];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let iv = inv[r];
        let mut dot = T::ZERO;
        for c in 0..d {
            let g = w[c] * dy[r * d + c];
            dot += g * xr[c];
            dw[c] += dy[r * d + c] * xr[c] * iv;
        }
        let coef = iv * iv * iv / dn * dot;
        for c in 0..d {
            let g = w[c] * dy[r * d + c];
            dx[r * d + c] = iv * g - xr[c] * coef;
        }
    }
    dx
}

/// Interleaved RoPE on `[n, nh·hd]`: adjacent pair `(2j, 2j+1)` rotated by
/// table row `t` (`cos/sin:[n, hd/2]`), same table for every head — matches
/// `rope_interleave_table.wgsl`.
pub fn rope<T: Fp>(x: &[T], n: usize, nh: usize, hd: usize, cos: &[T], sin: &[T]) -> Vec<T> {
    let half = hd / 2;
    let mut y = x.to_vec();
    for t in 0..n {
        for h in 0..nh {
            for j in 0..half {
                let base = (t * nh + h) * hd + 2 * j;
                let (c, s) = (cos[t * half + j], sin[t * half + j]);
                let (e, o) = (x[base], x[base + 1]);
                y[base] = e * c - o * s;
                y[base + 1] = e * s + o * c;
            }
        }
    }
    y
}

/// RoPE backward (rotate the grad by −angle).
pub fn rope_bwd<T: Fp>(dy: &[T], n: usize, nh: usize, hd: usize, cos: &[T], sin: &[T]) -> Vec<T> {
    let half = hd / 2;
    let mut dx = dy.to_vec();
    for t in 0..n {
        for h in 0..nh {
            for j in 0..half {
                let base = (t * nh + h) * hd + 2 * j;
                let (c, s) = (cos[t * half + j], sin[t * half + j]);
                let (de, dob) = (dy[base], dy[base + 1]);
                dx[base] = de * c + dob * s;
                dx[base + 1] = -de * s + dob * c;
            }
        }
    }
    dx
}

/// Joint bidirectional multi-head attention over all `n` rows, scale
/// `1/√hd`. Returns `(probs[nh·n·n], ctx[n·d])`.
fn attn_fwd<T: Fp>(qr: &[T], kr: &[T], v: &[T], n: usize, nh: usize, hd: usize) -> (Vec<T>, Vec<T>) {
    let scale = T::fr(1.0 / (hd as f64).sqrt());
    let mut probs = vec![T::ZERO; nh * n * n];
    let mut ctx = vec![T::ZERO; n * nh * hd];
    for h in 0..nh {
        for i in 0..n {
            let mut row = vec![T::ZERO; n];
            let mut mx = T::fr(f64::NEG_INFINITY);
            for j in 0..n {
                let mut s = T::ZERO;
                for dd in 0..hd {
                    s += qr[(i * nh + h) * hd + dd] * kr[(j * nh + h) * hd + dd];
                }
                row[j] = s * scale;
                if row[j] > mx {
                    mx = row[j];
                }
            }
            let mut den = T::ZERO;
            for j in 0..n {
                row[j] = (row[j] - mx).exp();
                den += row[j];
            }
            for j in 0..n {
                let p = row[j] / den;
                probs[(h * n + i) * n + j] = p;
                for dd in 0..hd {
                    ctx[(i * nh + h) * hd + dd] += p * v[(j * nh + h) * hd + dd];
                }
            }
        }
    }
    (probs, ctx)
}

/// Attention backward: `dctx` → `(dqr, dkr, dv)`.
#[allow(clippy::too_many_arguments)]
fn attn_bwd<T: Fp>(probs: &[T], qr: &[T], kr: &[T], v: &[T], n: usize, nh: usize, hd: usize, dctx: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let scale = T::fr(1.0 / (hd as f64).sqrt());
    let d = nh * hd;
    let mut dqr = vec![T::ZERO; n * d];
    let mut dkr = vec![T::ZERO; n * d];
    let mut dv = vec![T::ZERO; n * d];
    for h in 0..nh {
        for i in 0..n {
            let mut dprobs = vec![T::ZERO; n];
            for j in 0..n {
                let p = probs[(h * n + i) * n + j];
                let mut dp = T::ZERO;
                for dd in 0..hd {
                    let gc = dctx[(i * nh + h) * hd + dd];
                    dp += gc * v[(j * nh + h) * hd + dd];
                    dv[(j * nh + h) * hd + dd] += p * gc;
                }
                dprobs[j] = dp;
            }
            // softmax jacobian: dscore[j] = p[j]·(dprobs[j] − Σ p·dprobs)
            let mut sdot = T::ZERO;
            for j in 0..n {
                sdot += probs[(h * n + i) * n + j] * dprobs[j];
            }
            for j in 0..n {
                let p = probs[(h * n + i) * n + j];
                let dscore = p * (dprobs[j] - sdot) * scale;
                for dd in 0..hd {
                    dqr[(i * nh + h) * hd + dd] += dscore * kr[(j * nh + h) * hd + dd];
                    dkr[(j * nh + h) * hd + dd] += dscore * qr[(i * nh + h) * hd + dd];
                }
            }
        }
    }
    (dqr, dkr, dv)
}

/// Apply a modulation site over rows `r0..r1`: `y = (1+scale)⊙xhat + shift`.
fn mod_ln<T: Fp>(xhat: &[T], m: &Mod<T>, r0: usize, r1: usize, d: usize, y: &mut [T]) {
    for r in r0..r1 {
        for c in 0..d {
            y[r * d + c] = (T::ONE + m.scale[c]) * xhat[r * d + c] + m.shift[c];
        }
    }
}

/// Modulated-LN backward over rows `r0..r1`: accumulates `d_scale`/`d_shift`
/// into `mg` and writes `dxhat` (for [`layernorm_bwd`]).
#[allow(clippy::too_many_arguments)]
fn mod_ln_bwd<T: Fp>(xhat: &[T], m: &Mod<T>, mg: &mut ModGrad<T>, r0: usize, r1: usize, d: usize, dy: &[T], dxhat: &mut [T]) {
    for r in r0..r1 {
        for c in 0..d {
            let g = dy[r * d + c];
            mg.scale[c] += g * xhat[r * d + c];
            mg.shift[c] += g;
            dxhat[r * d + c] = (T::ONE + m.scale[c]) * g;
        }
    }
}

/// Gated residual over rows `r0..r1`: `y = x + gate ⊙ h` (per-channel gate).
fn gate_rows<T: Fp>(x: &[T], gate: &[T], h: &[T], r0: usize, r1: usize, d: usize, y: &mut [T]) {
    for r in r0..r1 {
        for c in 0..d {
            y[r * d + c] = x[r * d + c] + gate[c] * h[r * d + c];
        }
    }
}

// ---- double block ----

/// Everything the double-block backward needs from the forward pass.
pub struct DoubleCache<T> {
    xhat1: Vec<T>,
    inv1: Vec<T>,
    n1: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    inv_q: Vec<T>,
    inv_k: Vec<T>,
    qr: Vec<T>,
    kr: Vec<T>,
    probs: Vec<T>,
    ctx: Vec<T>,
    proj: Vec<T>,
    xhat2: Vec<T>,
    inv2: Vec<T>,
    n2: Vec<T>,
    h1: Vec<T>,
    h2: Vec<T>,
    hs: Vec<T>,
    mlpo: Vec<T>,
    cos: Vec<T>,
    sin: Vec<T>,
}

/// Double-block forward on the joint slab `x:[n,d]` (txt rows `0..nt`, img
/// rows `nt..n`). `cos/sin:[n, hd/2]`. Returns `(out[n·d], cache)`.
pub fn double_forward<T: Fp>(dm: Dims, w: &DoubleW<T>, x: &[T], m: &DoubleMods<T>, cos: &[T], sin: &[T]) -> (Vec<T>, DoubleCache<T>) {
    let (nt, n, d, nh, hd, mlp) = (dm.nt, dm.n(), dm.d, dm.nh, dm.hd(), dm.mlp);
    // attention halves of both streams into the joint q/k/v
    let (xhat1, inv1) = layernorm(x, n, d);
    let mut n1 = vec![T::ZERO; n * d];
    mod_ln(&xhat1, &m.txt1, 0, nt, d, &mut n1);
    mod_ln(&xhat1, &m.img1, nt, n, d, &mut n1);
    let mut q = linear(&n1[..nt * d], nt, d, &w.txt.wq, d);
    q.extend(linear(&n1[nt * d..], n - nt, d, &w.img.wq, d));
    let mut k = linear(&n1[..nt * d], nt, d, &w.txt.wk, d);
    k.extend(linear(&n1[nt * d..], n - nt, d, &w.img.wk, d));
    let mut v = linear(&n1[..nt * d], nt, d, &w.txt.wv, d);
    v.extend(linear(&n1[nt * d..], n - nt, d, &w.img.wv, d));
    // per-head QK-RMSNorm with per-stream learnable scale
    let (qn_t, inv_q_t) = rmsnorm(&q[..nt * d], nt * nh, hd, &w.txt.nq);
    let (qn_i, inv_q_i) = rmsnorm(&q[nt * d..], (n - nt) * nh, hd, &w.img.nq);
    let (kn_t, inv_k_t) = rmsnorm(&k[..nt * d], nt * nh, hd, &w.txt.nk);
    let (kn_i, inv_k_i) = rmsnorm(&k[nt * d..], (n - nt) * nh, hd, &w.img.nk);
    let qn: Vec<T> = qn_t.into_iter().chain(qn_i).collect();
    let kn: Vec<T> = kn_t.into_iter().chain(kn_i).collect();
    let inv_q: Vec<T> = inv_q_t.into_iter().chain(inv_q_i).collect();
    let inv_k: Vec<T> = inv_k_t.into_iter().chain(inv_k_i).collect();
    let qr = rope(&qn, n, nh, hd, cos, sin);
    let kr = rope(&kn, n, nh, hd, cos, sin);
    let (probs, ctx) = attn_fwd(&qr, &kr, &v, n, nh, hd);
    // per-stream out-proj + gated residual
    let mut proj = linear(&ctx[..nt * d], nt, d, &w.txt.wo, d);
    proj.extend(linear(&ctx[nt * d..], n - nt, d, &w.img.wo, d));
    let mut x1 = vec![T::ZERO; n * d];
    gate_rows(x, &m.txt1.gate, &proj, 0, nt, d, &mut x1);
    gate_rows(x, &m.img1.gate, &proj, nt, n, d, &mut x1);
    // MLP halves
    let (xhat2, inv2) = layernorm(&x1, n, d);
    let mut n2 = vec![T::ZERO; n * d];
    mod_ln(&xhat2, &m.txt2, 0, nt, d, &mut n2);
    mod_ln(&xhat2, &m.img2, nt, n, d, &mut n2);
    let mut h1 = linear(&n2[..nt * d], nt, d, &w.txt.w1, mlp);
    h1.extend(linear(&n2[nt * d..], n - nt, d, &w.img.w1, mlp));
    let mut h2 = linear(&n2[..nt * d], nt, d, &w.txt.w3, mlp);
    h2.extend(linear(&n2[nt * d..], n - nt, d, &w.img.w3, mlp));
    let hs: Vec<T> = h1.iter().zip(&h2).map(|(&a, &b)| silu(a) * b).collect();
    let mut mlpo = linear(&hs[..nt * mlp], nt, mlp, &w.txt.w2, d);
    mlpo.extend(linear(&hs[nt * mlp..], n - nt, mlp, &w.img.w2, d));
    let mut out = vec![T::ZERO; n * d];
    gate_rows(&x1, &m.txt2.gate, &mlpo, 0, nt, d, &mut out);
    gate_rows(&x1, &m.img2.gate, &mlpo, nt, n, d, &mut out);

    let cache = DoubleCache {
        xhat1, inv1, n1, q, k, v, inv_q, inv_k, qr, kr, probs, ctx, proj,
        xhat2, inv2, n2, h1, h2, hs, mlpo,
        cos: cos.to_vec(), sin: sin.to_vec(),
    };
    (out, cache)
}

/// `dh = gate ⊙ dy` over rows `r0..r1`, accumulating `dgate += Σ dy⊙h`.
/// (The gated residual's `dx` is the identity — callers reuse `dy` directly,
/// matching the gate_row backward contract.)
#[allow(clippy::too_many_arguments)]
fn gate_bwd<T: Fp>(gate: &[T], h: &[T], dy: &[T], r0: usize, r1: usize, d: usize, dh: &mut [T], dgate: &mut [T]) {
    for r in r0..r1 {
        for c in 0..d {
            let g = dy[r * d + c];
            dh[r * d + c] = gate[c] * g;
            dgate[c] += g * h[r * d + c];
        }
    }
}

/// Double-block backward: `dout[n·d]` → both streams' weight grads, the four
/// modulation-site grads, and `dx`. `m` must be the sites the forward ran with.
pub fn double_backward<T: Fp>(dm: Dims, w: &DoubleW<T>, m: &DoubleMods<T>, c: &DoubleCache<T>, dout: &[T]) -> DoubleGrads<T> {
    let (nt, n, d, nh, hd, mlp) = (dm.nt, dm.n(), dm.d, dm.nh, dm.hd(), dm.mlp);
    let ni = n - nt;
    let mut g = DoubleGrads {
        img: StreamG::zeros(&dm),
        txt: StreamG::zeros(&dm),
        img1: ModGrad::zeros(d),
        img2: ModGrad::zeros(d),
        txt1: ModGrad::zeros(d),
        txt2: ModGrad::zeros(d),
        dx: vec![T::ZERO; n * d],
    };

    // out = x1 + gate_s2 ⊙ mlpo
    let mut dx1 = dout.to_vec();
    let mut dmlpo = vec![T::ZERO; n * d];
    gate_bwd(&m.txt2.gate, &c.mlpo, dout, 0, nt, d, &mut dmlpo, &mut g.txt2.gate);
    gate_bwd(&m.img2.gate, &c.mlpo, dout, nt, n, d, &mut dmlpo, &mut g.img2.gate);
    // mlpo = hs @ w2ᵀ (per stream)
    let (dhs_t, dw2_t) = linear_bwd(&c.hs[..nt * mlp], nt, mlp, &w.txt.w2, d, &dmlpo[..nt * d]);
    let (dhs_i, dw2_i) = linear_bwd(&c.hs[nt * mlp..], ni, mlp, &w.img.w2, d, &dmlpo[nt * d..]);
    g.txt.w2 = dw2_t;
    g.img.w2 = dw2_i;
    let dhs: Vec<T> = dhs_t.into_iter().chain(dhs_i).collect();
    // hs = silu(h1) ⊙ h2  (silu-gated half FIRST)
    let mut dh1 = vec![T::ZERO; n * mlp];
    let mut dh2 = vec![T::ZERO; n * mlp];
    for i in 0..n * mlp {
        dh1[i] = dhs[i] * c.h2[i] * dsilu(c.h1[i]);
        dh2[i] = dhs[i] * silu(c.h1[i]);
    }
    // h1 = n2 @ w1ᵀ, h2 = n2 @ w3ᵀ (per stream)
    let (dn2a_t, dw1_t) = linear_bwd(&c.n2[..nt * d], nt, d, &w.txt.w1, mlp, &dh1[..nt * mlp]);
    let (dn2a_i, dw1_i) = linear_bwd(&c.n2[nt * d..], ni, d, &w.img.w1, mlp, &dh1[nt * mlp..]);
    let (dn2b_t, dw3_t) = linear_bwd(&c.n2[..nt * d], nt, d, &w.txt.w3, mlp, &dh2[..nt * mlp]);
    let (dn2b_i, dw3_i) = linear_bwd(&c.n2[nt * d..], ni, d, &w.img.w3, mlp, &dh2[nt * mlp..]);
    g.txt.w1 = dw1_t;
    g.img.w1 = dw1_i;
    g.txt.w3 = dw3_t;
    g.img.w3 = dw3_i;
    let mut dn2: Vec<T> = dn2a_t.into_iter().chain(dn2a_i).collect();
    for (a, b) in dn2.iter_mut().zip(dn2b_t.into_iter().chain(dn2b_i)) {
        *a += b;
    }
    // n2 = (1+scale_s2)⊙xhat2 + shift_s2 ; xhat2 = LN(x1)
    let mut dxhat2 = vec![T::ZERO; n * d];
    mod_ln_bwd(&c.xhat2, &m.txt2, &mut g.txt2, 0, nt, d, &dn2, &mut dxhat2);
    mod_ln_bwd(&c.xhat2, &m.img2, &mut g.img2, nt, n, d, &dn2, &mut dxhat2);
    for (a, b) in dx1.iter_mut().zip(layernorm_bwd(&c.xhat2, &c.inv2, n, d, &dxhat2)) {
        *a += b;
    }

    // x1 = x + gate_s1 ⊙ proj
    let mut dx = dx1.clone();
    let mut dproj = vec![T::ZERO; n * d];
    gate_bwd(&m.txt1.gate, &c.proj, &dx1, 0, nt, d, &mut dproj, &mut g.txt1.gate);
    gate_bwd(&m.img1.gate, &c.proj, &dx1, nt, n, d, &mut dproj, &mut g.img1.gate);
    // proj = ctx @ woᵀ (per stream)
    let (dctx_t, dwo_t) = linear_bwd(&c.ctx[..nt * d], nt, d, &w.txt.wo, d, &dproj[..nt * d]);
    let (dctx_i, dwo_i) = linear_bwd(&c.ctx[nt * d..], ni, d, &w.img.wo, d, &dproj[nt * d..]);
    g.txt.wo = dwo_t;
    g.img.wo = dwo_i;
    let dctx: Vec<T> = dctx_t.into_iter().chain(dctx_i).collect();
    // joint attention + rope + qk-norm
    let (dqr, dkr, dv) = attn_bwd(&c.probs, &c.qr, &c.kr, &c.v, n, nh, hd, &dctx);
    let dqn = rope_bwd(&dqr, n, nh, hd, &c.cos, &c.sin);
    let dkn = rope_bwd(&dkr, n, nh, hd, &c.cos, &c.sin);
    let dq_t = rmsnorm_bwd(&c.q[..nt * d], nt * nh, hd, &w.txt.nq, &c.inv_q[..nt * nh], &dqn[..nt * d], &mut g.txt.nq);
    let dq_i = rmsnorm_bwd(&c.q[nt * d..], ni * nh, hd, &w.img.nq, &c.inv_q[nt * nh..], &dqn[nt * d..], &mut g.img.nq);
    let dk_t = rmsnorm_bwd(&c.k[..nt * d], nt * nh, hd, &w.txt.nk, &c.inv_k[..nt * nh], &dkn[..nt * d], &mut g.txt.nk);
    let dk_i = rmsnorm_bwd(&c.k[nt * d..], ni * nh, hd, &w.img.nk, &c.inv_k[nt * nh..], &dkn[nt * d..], &mut g.img.nk);
    // q,k,v = n1 @ {wq,wk,wv}ᵀ (per stream)
    let (dn1q_t, dwq_t) = linear_bwd(&c.n1[..nt * d], nt, d, &w.txt.wq, d, &dq_t);
    let (dn1q_i, dwq_i) = linear_bwd(&c.n1[nt * d..], ni, d, &w.img.wq, d, &dq_i);
    let (dn1k_t, dwk_t) = linear_bwd(&c.n1[..nt * d], nt, d, &w.txt.wk, d, &dk_t);
    let (dn1k_i, dwk_i) = linear_bwd(&c.n1[nt * d..], ni, d, &w.img.wk, d, &dk_i);
    let (dn1v_t, dwv_t) = linear_bwd(&c.n1[..nt * d], nt, d, &w.txt.wv, d, &dv[..nt * d]);
    let (dn1v_i, dwv_i) = linear_bwd(&c.n1[nt * d..], ni, d, &w.img.wv, d, &dv[nt * d..]);
    g.txt.wq = dwq_t;
    g.img.wq = dwq_i;
    g.txt.wk = dwk_t;
    g.img.wk = dwk_i;
    g.txt.wv = dwv_t;
    g.img.wv = dwv_i;
    let mut dn1: Vec<T> = dn1q_t.into_iter().chain(dn1q_i).collect();
    for (a, b) in dn1.iter_mut().zip(dn1k_t.into_iter().chain(dn1k_i)) {
        *a += b;
    }
    for (a, b) in dn1.iter_mut().zip(dn1v_t.into_iter().chain(dn1v_i)) {
        *a += b;
    }
    // n1 = (1+scale_s1)⊙xhat1 + shift_s1 ; xhat1 = LN(x)
    let mut dxhat1 = vec![T::ZERO; n * d];
    mod_ln_bwd(&c.xhat1, &m.txt1, &mut g.txt1, 0, nt, d, &dn1, &mut dxhat1);
    mod_ln_bwd(&c.xhat1, &m.img1, &mut g.img1, nt, n, d, &dn1, &mut dxhat1);
    for (a, b) in dx.iter_mut().zip(layernorm_bwd(&c.xhat1, &c.inv1, n, d, &dxhat1)) {
        *a += b;
    }
    g.dx = dx;
    g
}

// ---- single block ----

/// Everything the single-block backward needs from the forward pass.
pub struct SingleCache<T> {
    xhat: Vec<T>,
    inv: Vec<T>,
    n1: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    inv_q: Vec<T>,
    inv_k: Vec<T>,
    qr: Vec<T>,
    kr: Vec<T>,
    probs: Vec<T>,
    ctx: Vec<T>,
    proj: Vec<T>,
    h1: Vec<T>,
    h2: Vec<T>,
    hs: Vec<T>,
    mlpo: Vec<T>,
    cos: Vec<T>,
    sin: Vec<T>,
}

/// Single-block forward: ONE shared modulated LN feeds attention ‖ SwiGLU in
/// parallel; linear2 is column-split into `wo_a` (attn) + `wo_b` (mlp); one
/// gated residual applied distributively: `out = x + gate⊙proj + gate⊙mlpo`.
pub fn single_forward<T: Fp>(dm: Dims, w: &SingleW<T>, x: &[T], m: &Mod<T>, cos: &[T], sin: &[T]) -> (Vec<T>, SingleCache<T>) {
    let (n, d, nh, hd, mlp) = (dm.n(), dm.d, dm.nh, dm.hd(), dm.mlp);
    let (xhat, inv) = layernorm(x, n, d);
    let mut n1 = vec![T::ZERO; n * d];
    mod_ln(&xhat, m, 0, n, d, &mut n1);
    let q = linear(&n1, n, d, &w.wq, d);
    let k = linear(&n1, n, d, &w.wk, d);
    let v = linear(&n1, n, d, &w.wv, d);
    let (qn, inv_q) = rmsnorm(&q, n * nh, hd, &w.nq);
    let (kn, inv_k) = rmsnorm(&k, n * nh, hd, &w.nk);
    let qr = rope(&qn, n, nh, hd, cos, sin);
    let kr = rope(&kn, n, nh, hd, cos, sin);
    let (probs, ctx) = attn_fwd(&qr, &kr, &v, n, nh, hd);
    let h1 = linear(&n1, n, d, &w.w1, mlp);
    let h2 = linear(&n1, n, d, &w.w3, mlp);
    let hs: Vec<T> = h1.iter().zip(&h2).map(|(&a, &b)| silu(a) * b).collect();
    let proj = linear(&ctx, n, d, &w.wo_a, d);
    let mlpo = linear(&hs, n, mlp, &w.wo_b, d);
    let mut out = vec![T::ZERO; n * d];
    for r in 0..n {
        for c in 0..d {
            out[r * d + c] = x[r * d + c] + m.gate[c] * (proj[r * d + c] + mlpo[r * d + c]);
        }
    }
    let cache = SingleCache {
        xhat, inv, n1, q, k, v, inv_q, inv_k, qr, kr, probs, ctx, proj, h1, h2, hs, mlpo,
        cos: cos.to_vec(), sin: sin.to_vec(),
    };
    (out, cache)
}

/// Single-block backward: `dout[n·d]` → weight grads, the block's
/// modulation-site grad contribution, and `dx`.
pub fn single_backward<T: Fp>(dm: Dims, w: &SingleW<T>, m: &Mod<T>, c: &SingleCache<T>, dout: &[T]) -> SingleGrads<T> {
    let (n, d, nh, hd, mlp) = (dm.n(), dm.d, dm.nh, dm.hd(), dm.mlp);
    let mut g = SingleGrads {
        wq: vec![T::ZERO; d * d],
        wk: vec![T::ZERO; d * d],
        wv: vec![T::ZERO; d * d],
        nq: vec![T::ZERO; hd],
        nk: vec![T::ZERO; hd],
        w1: vec![T::ZERO; mlp * d],
        w3: vec![T::ZERO; mlp * d],
        wo_a: vec![T::ZERO; d * d],
        wo_b: vec![T::ZERO; d * mlp],
        m: ModGrad::zeros(d),
        dx: vec![T::ZERO; n * d],
    };
    // out = x + gate ⊙ (proj + mlpo)
    let mut dx = dout.to_vec();
    let mut dproj = vec![T::ZERO; n * d];
    let mut dmlpo = vec![T::ZERO; n * d];
    for r in 0..n {
        for cc in 0..d {
            let go = dout[r * d + cc];
            dproj[r * d + cc] = m.gate[cc] * go;
            dmlpo[r * d + cc] = m.gate[cc] * go;
            g.m.gate[cc] += go * (c.proj[r * d + cc] + c.mlpo[r * d + cc]);
        }
    }
    // proj = ctx @ wo_aᵀ ; mlpo = hs @ wo_bᵀ
    let (dctx, dwo_a) = linear_bwd(&c.ctx, n, d, &w.wo_a, d, &dproj);
    let (dhs, dwo_b) = linear_bwd(&c.hs, n, mlp, &w.wo_b, d, &dmlpo);
    g.wo_a = dwo_a;
    g.wo_b = dwo_b;
    // hs = silu(h1) ⊙ h2
    let mut dh1 = vec![T::ZERO; n * mlp];
    let mut dh2 = vec![T::ZERO; n * mlp];
    for i in 0..n * mlp {
        dh1[i] = dhs[i] * c.h2[i] * dsilu(c.h1[i]);
        dh2[i] = dhs[i] * silu(c.h1[i]);
    }
    let (dn1a, dw1) = linear_bwd(&c.n1, n, d, &w.w1, mlp, &dh1);
    let (dn1b, dw3) = linear_bwd(&c.n1, n, d, &w.w3, mlp, &dh2);
    g.w1 = dw1;
    g.w3 = dw3;
    // attention chain
    let (dqr, dkr, dv) = attn_bwd(&c.probs, &c.qr, &c.kr, &c.v, n, nh, hd, &dctx);
    let dqn = rope_bwd(&dqr, n, nh, hd, &c.cos, &c.sin);
    let dkn = rope_bwd(&dkr, n, nh, hd, &c.cos, &c.sin);
    let dq = rmsnorm_bwd(&c.q, n * nh, hd, &w.nq, &c.inv_q, &dqn, &mut g.nq);
    let dk = rmsnorm_bwd(&c.k, n * nh, hd, &w.nk, &c.inv_k, &dkn, &mut g.nk);
    let (dn1q, dwq) = linear_bwd(&c.n1, n, d, &w.wq, d, &dq);
    let (dn1k, dwk) = linear_bwd(&c.n1, n, d, &w.wk, d, &dk);
    let (dn1v, dwv) = linear_bwd(&c.n1, n, d, &w.wv, d, &dv);
    g.wq = dwq;
    g.wk = dwk;
    g.wv = dwv;
    // n1 feeds five linears: q,k,v,w1,w3
    let mut dn1 = dn1a;
    for (a, b) in dn1.iter_mut().zip(dn1b) {
        *a += b;
    }
    for (a, b) in dn1.iter_mut().zip(dn1q) {
        *a += b;
    }
    for (a, b) in dn1.iter_mut().zip(dn1k) {
        *a += b;
    }
    for (a, b) in dn1.iter_mut().zip(dn1v) {
        *a += b;
    }
    // n1 = (1+scale)⊙xhat + shift ; xhat = LN(x)
    let mut dxhat = vec![T::ZERO; n * d];
    mod_ln_bwd(&c.xhat, m, &mut g.m, 0, n, d, &dn1, &mut dxhat);
    for (a, b) in dx.iter_mut().zip(layernorm_bwd(&c.xhat, &c.inv, n, d, &dxhat)) {
        *a += b;
    }
    g.dx = dx;
    g
}
