// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host reference forward + analytic backward for ONE `WanAttentionBlock` - the
//! unit the whole DiT training step is built from.
//!
//! The forward mirrors [`crate::block::build_block_steps`] op for op: an
//! affine-free LayerNorm carrying the timestep modulation, QK-normalised
//! self-attention with three-axis RoPE and a gated residual, an **affine**
//! LayerNorm (`norm3`, the only learned-affine norm in the block) into a
//! separate text cross-attention with an UNGATED residual, and a GELU(tanh) FFN
//! behind a second modulated LayerNorm and a second gate.
//!
//! ## Differentiating the unfolded modulation
//!
//! The device path exploits
//! `LN_noaffine(x)·(1 + scale) + shift == LayerNorm(x, gamma = 1+scale, beta = shift)`
//! and folds `modulation + e0` into two `(gamma, beta)` pairs once per forward.
//! A structural shortcut like that is only safe if the backward oracle
//! differentiates the **unfolded** form, and this one does:
//! this module computes `y = (1 + scale)·LN_noaffine(x) + shift` explicitly, so
//! the backward yields per-site `d_shift/d_scale/d_gate` for all six vectors.
//! Because the fold's operand is the SUM `modulation + e0`, those six grads are
//! simultaneously
//!
//! * `d(blocks.{l}.modulation)` - this block's own parameter, and
//! * this block's contribution to `d e0`, which [`crate::modelgrad`] accumulates
//!   over the whole stack and routes through `time_projection` -> `silu` ->
//!   `time_embedding` into the timestep path.
//!
//! Dropping either half leaves a gradient that is locally plausible and globally
//! wrong, which is why [`BlockGrads::modulation`] is one vector used twice
//! rather than two separately-derived ones.
//!
//! ## One implementation, two instantiations
//!
//! Generic over [`Fp`]: the `f64` instantiation is the finite-difference
//! gradcheck oracle (`gradcheck::check_wan`, `tests/block_grad.rs`); the `f32`
//! instantiation is the host trainer [`crate::finetune`] drives. Same code, so
//! oracle and trainer cannot drift.
//!
//! AGENTS.md exception 1 applies to the primitives below: they are a
//! *deliberate* second derivation of the math the WGSL kernels implement (an
//! oracle sharing code with the thing it checks proves nothing), written against
//! each kernel's own contract - population-variance LayerNorm with a runtime
//! eps, `rmsnorm_eps`'s `1/sqrt(mean(x²)+eps)` over the FULL model width (not
//! per head - see [`crate::block`]), `rope_interleave_table`'s interleaved
//! `(2j, 2j+1)` pairs with one table row per token shared by every head,
//! `1/sqrt(head_dim)` attention scaling, `gelu`'s tanh approximation and
//! `gate_row`'s per-channel gated residual.

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
    fn tanh(self) -> Self;
    /// `y[o] = Σ_i w[o·inn+i]·x[i]` - the hot inner product of every linear.
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
    fn tanh(self) -> f64 {
        f64::tanh(self)
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
    fn tanh(self) -> f32 {
        f32::tanh(self)
    }
    fn matvec(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
        model::hostmath::matvec_par(w, x, out, inn)
    }
}

/// Shape of the block being differentiated. `t` latent tokens attend over
/// themselves and cross-attend into `te` text rows; `dim = nh·hd`.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub t: usize,
    pub te: usize,
    pub dim: usize,
    pub nh: usize,
    pub ffn: usize,
    /// Shared by every LayerNorm and RMSNorm in the block (`WanConfig::eps`).
    pub eps: f64,
}

impl Dims {
    pub fn hd(&self) -> usize {
        self.dim / self.nh
    }
}

/// A biased linear, `[out, in]` row-major plus `[out]` bias - every projection
/// in a Wan block has a bias (unlike FLUX.2's, which are bias-free). Doubles as
/// the gradient container.
#[derive(Clone, Debug, PartialEq)]
pub struct Lin<T> {
    pub w: Vec<T>,
    pub b: Vec<T>,
}

impl<T: Fp> Lin<T> {
    pub fn zeros(out: usize, inn: usize) -> Lin<T> {
        Lin { w: vec![T::ZERO; out * inn], b: vec![T::ZERO; out] }
    }
}

/// One block's trainable tensors, named as upstream names them.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockW<T> {
    /// `[6·dim]`: `(shift1, scale1, gate1, shift2, scale2, gate2)`, added to
    /// `e0` before the fold.
    pub modulation: Vec<T>,
    pub sq: Lin<T>,
    pub sk: Lin<T>,
    pub sv: Lin<T>,
    pub so: Lin<T>,
    pub snq: Vec<T>,
    pub snk: Vec<T>,
    pub cq: Lin<T>,
    pub ck: Lin<T>,
    pub cv: Lin<T>,
    pub co: Lin<T>,
    pub cnq: Vec<T>,
    pub cnk: Vec<T>,
    pub norm3_w: Vec<T>,
    pub norm3_b: Vec<T>,
    pub ff1: Lin<T>,
    pub ff2: Lin<T>,
}

/// Gradients mirroring [`BlockW`], plus the two upstream adjoints: `dx` to the
/// previous block and `dctx` to the shared text encoding.
#[derive(Clone, Debug)]
pub struct BlockGrads<T> {
    /// `d(modulation)`, which is ALSO this block's contribution to `d e0` - the
    /// two are the same vector because the fold's operand is their sum.
    pub modulation: Vec<T>,
    pub sq: Lin<T>,
    pub sk: Lin<T>,
    pub sv: Lin<T>,
    pub so: Lin<T>,
    pub snq: Vec<T>,
    pub snk: Vec<T>,
    pub cq: Lin<T>,
    pub ck: Lin<T>,
    pub cv: Lin<T>,
    pub co: Lin<T>,
    pub cnq: Vec<T>,
    pub cnk: Vec<T>,
    pub norm3_w: Vec<T>,
    pub norm3_b: Vec<T>,
    pub ff1: Lin<T>,
    pub ff2: Lin<T>,
    pub dx: Vec<T>,
    /// Grad w.r.t. the embedded text context. Every block reads the SAME `ctx`,
    /// so the model backward sums these before entering `text_embedding`.
    pub dctx: Vec<T>,
}

impl<T: Fp> BlockGrads<T> {
    fn zeros(d: Dims) -> BlockGrads<T> {
        let (dim, ffn, t, te) = (d.dim, d.ffn, d.t, d.te);
        BlockGrads {
            modulation: vec![T::ZERO; 6 * dim],
            sq: Lin::zeros(dim, dim),
            sk: Lin::zeros(dim, dim),
            sv: Lin::zeros(dim, dim),
            so: Lin::zeros(dim, dim),
            snq: vec![T::ZERO; dim],
            snk: vec![T::ZERO; dim],
            cq: Lin::zeros(dim, dim),
            ck: Lin::zeros(dim, dim),
            cv: Lin::zeros(dim, dim),
            co: Lin::zeros(dim, dim),
            cnq: vec![T::ZERO; dim],
            cnk: vec![T::ZERO; dim],
            norm3_w: vec![T::ZERO; dim],
            norm3_b: vec![T::ZERO; dim],
            ff1: Lin::zeros(ffn, dim),
            ff2: Lin::zeros(dim, ffn),
            dx: vec![T::ZERO; t * dim],
            dctx: vec![T::ZERO; te * dim],
        }
    }
}

// ---- primitives (host, generic) ----

pub(crate) fn sigmoid<T: Fp>(x: T) -> T {
    T::ONE / (T::ONE + (-x).exp())
}

/// SiLU, the timestep MLP's activation (`hostmath::silu_slice` on the f32 path).
pub(crate) fn silu<T: Fp>(x: T) -> T {
    x * sigmoid(x)
}

/// `d silu(x) / dx = σ(x) + x·σ(x)·(1−σ(x))`.
pub(crate) fn dsilu<T: Fp>(x: T) -> T {
    let s = sigmoid(x);
    s + x * s * (T::ONE - s)
}

const GELU_K: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
const GELU_C: f64 = 0.044_715;

/// GELU, tanh approximation - `nn.GELU(approximate='tanh')`, the function
/// `gelu.wgsl` implements and [`crate::model::gelu_tanh`] mirrors on f32.
pub(crate) fn gelu<T: Fp>(x: T) -> T {
    let u = T::fr(GELU_K) * (x + T::fr(GELU_C) * x * x * x);
    T::fr(0.5) * x * (T::ONE + u.tanh())
}

/// `d gelu(x) / dx` for the same tanh approximation.
pub(crate) fn dgelu<T: Fp>(x: T) -> T {
    let inner = x + T::fr(GELU_C) * x * x * x;
    let th = (T::fr(GELU_K) * inner).tanh();
    let dinner = T::ONE + T::fr(3.0 * GELU_C) * x * x;
    T::fr(0.5) * (T::ONE + th) + T::fr(0.5) * x * (T::ONE - th * th) * T::fr(GELU_K) * dinner
}

/// `y = x @ wᵀ + b`, `x:[rows,inn]`, `w:[out,inn]` -> `y:[rows,out]` - the
/// `matmul` + `bias_add` pair every Wan linear dispatches.
pub fn linear<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], b: &[T], out: usize) -> Vec<T> {
    let mut y = Vec::with_capacity(rows * out);
    for r in 0..rows {
        let mut row = T::matvec(w, &x[r * inn..(r + 1) * inn], out, inn);
        for (v, &bo) in row.iter_mut().zip(b) {
            *v += bo;
        }
        y.append(&mut row);
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

/// Biased-linear backward: `dx = dy @ w`, `dw = dyᵀ @ x`, `db = Σ_rows dy`.
/// Both products route through [`Fp::matvec`] on pre-transposed operands so the
/// f32 trainer keeps its rows parallel.
pub fn linear_bwd<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize, dy: &[T]) -> (Vec<T>, Lin<T>) {
    let wt = transpose(w, out, inn); // [inn, out]
    let mut dx = Vec::with_capacity(rows * inn);
    for r in 0..rows {
        dx.append(&mut T::matvec(&wt, &dy[r * out..(r + 1) * out], inn, out));
    }
    let xt = transpose(x, rows, inn); // [inn, rows]
    let mut g = Lin::<T>::zeros(out, inn);
    let mut dyc = vec![T::ZERO; rows];
    for o in 0..out {
        let mut bacc = T::ZERO;
        for r in 0..rows {
            dyc[r] = dy[r * out + o];
            bacc += dyc[r];
        }
        g.b[o] = bacc;
        let row = T::matvec(&xt, &dyc, inn, rows);
        g.w[o * inn..(o + 1) * inn].copy_from_slice(&row);
    }
    (dx, g)
}

/// Affine-free LayerNorm over the last `d` of `[rows,d]` (population variance -
/// `layernorm.wgsl` with gamma=1, beta=0). Returns `(xhat, inv[rows])`.
pub fn layernorm<T: Fp>(x: &[T], rows: usize, d: usize, eps: f64) -> (Vec<T>, Vec<T>) {
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
        let iv = T::ONE / (var + T::fr(eps)).sqrt();
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

/// The affine half of a LayerNorm, applied per channel: `y = g⊙xhat + b`.
///
/// Both of the block's forms go through this: the learned `norm3` (`g = weight`,
/// `b = bias`) and a modulation site (`g = 1 + scale`, `b = shift`). The second
/// is the UNFOLDED modulation - the device path's folded
/// `LayerNorm(gamma, beta)` is the same function, and `dg == d(scale)` because
/// `d(1+scale)/d(scale) == 1`.
pub(crate) fn affine<T: Fp>(xhat: &[T], g: &[T], b: &[T], rows: usize, d: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; rows * d];
    for r in 0..rows {
        for c in 0..d {
            y[r * d + c] = g[c] * xhat[r * d + c] + b[c];
        }
    }
    y
}

/// [`affine`] backward: accumulates `dg`/`db` and returns `dxhat`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn affine_bwd<T: Fp>(xhat: &[T], g: &[T], rows: usize, d: usize, dy: &[T], dg: &mut [T], db: &mut [T]) -> Vec<T> {
    let mut dxhat = vec![T::ZERO; rows * d];
    for r in 0..rows {
        for c in 0..d {
            let gr = dy[r * d + c];
            dg[c] += gr * xhat[r * d + c];
            db[c] += gr;
            dxhat[r * d + c] = g[c] * gr;
        }
    }
    dxhat
}

/// RMSNorm with a runtime eps over the last `d` of `[rows,d]`
/// (`rmsnorm_eps.wgsl`): `y = w ⊙ x·inv`, `inv = 1/√(mean(x²)+eps)`.
///
/// **`d` is the FULL model width here, not `head_dim`.** `WanRMSNorm(dim)` runs
/// before the `view(b, s, n, d)` that splits the heads; per-head normalisation
/// would divide by a different scalar per head and still produce plausible
/// video. See [`crate::block::qk_norm`].
pub fn rmsnorm<T: Fp>(x: &[T], rows: usize, d: usize, w: &[T], eps: f64) -> (Vec<T>, Vec<T>) {
    let mut y = vec![T::ZERO; rows * d];
    let mut inv = vec![T::ZERO; rows];
    let dn = T::fr(d as f64);
    for r in 0..rows {
        let xr = &x[r * d..r * d + d];
        let mut ss = T::ZERO;
        for &v in xr {
            ss += v * v;
        }
        let iv = T::ONE / (ss / dn + T::fr(eps)).sqrt();
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

/// Interleaved RoPE on `[n, nh·hd]`: adjacent pair `(2j, 2j+1)` rotated by table
/// row `t` (`cos/sin:[n, hd/2]`), the same row for every head - exactly
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

/// Bidirectional multi-head attention from `nq` query rows into `nk` key rows,
/// scale `1/√hd`. ONE function for both of the block's attentions: self is
/// `nq == nk` over the latent tokens, cross is `nq = t` queries into `nk = te`
/// text rows. Returns `(probs[nh·nq·nk], out[nq·nh·hd])`.
#[allow(clippy::too_many_arguments)]
fn attn_fwd<T: Fp>(q: &[T], nq: usize, k: &[T], v: &[T], nk: usize, nh: usize, hd: usize) -> (Vec<T>, Vec<T>) {
    let scale = T::fr(1.0 / (hd as f64).sqrt());
    let mut probs = vec![T::ZERO; nh * nq * nk];
    let mut out = vec![T::ZERO; nq * nh * hd];
    let mut row = vec![T::ZERO; nk];
    for h in 0..nh {
        for i in 0..nq {
            let mut mx = T::fr(f64::NEG_INFINITY);
            for (j, slot) in row.iter_mut().enumerate() {
                let mut s = T::ZERO;
                for dd in 0..hd {
                    s += q[(i * nh + h) * hd + dd] * k[(j * nh + h) * hd + dd];
                }
                *slot = s * scale;
                if *slot > mx {
                    mx = *slot;
                }
            }
            let mut den = T::ZERO;
            for e in row.iter_mut() {
                *e = (*e - mx).exp();
                den += *e;
            }
            for j in 0..nk {
                let p = row[j] / den;
                probs[(h * nq + i) * nk + j] = p;
                for dd in 0..hd {
                    out[(i * nh + h) * hd + dd] += p * v[(j * nh + h) * hd + dd];
                }
            }
        }
    }
    (probs, out)
}

/// [`attn_fwd`] backward: `dout` -> `(dq, dk, dv)`.
#[allow(clippy::too_many_arguments)]
fn attn_bwd<T: Fp>(probs: &[T], q: &[T], k: &[T], v: &[T], nq: usize, nk: usize, nh: usize, hd: usize, dout: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let scale = T::fr(1.0 / (hd as f64).sqrt());
    let d = nh * hd;
    let mut dq = vec![T::ZERO; nq * d];
    let mut dk = vec![T::ZERO; nk * d];
    let mut dv = vec![T::ZERO; nk * d];
    let mut dprobs = vec![T::ZERO; nk];
    for h in 0..nh {
        for i in 0..nq {
            for (j, slot) in dprobs.iter_mut().enumerate() {
                let p = probs[(h * nq + i) * nk + j];
                let mut dp = T::ZERO;
                for dd in 0..hd {
                    let gc = dout[(i * nh + h) * hd + dd];
                    dp += gc * v[(j * nh + h) * hd + dd];
                    dv[(j * nh + h) * hd + dd] += p * gc;
                }
                *slot = dp;
            }
            // softmax jacobian: dscore[j] = p[j]·(dprobs[j] − Σ p·dprobs)
            let mut sdot = T::ZERO;
            for j in 0..nk {
                sdot += probs[(h * nq + i) * nk + j] * dprobs[j];
            }
            for j in 0..nk {
                let p = probs[(h * nq + i) * nk + j];
                let dscore = p * (dprobs[j] - sdot) * scale;
                for dd in 0..hd {
                    dq[(i * nh + h) * hd + dd] += dscore * k[(j * nh + h) * hd + dd];
                    dk[(j * nh + h) * hd + dd] += dscore * q[(i * nh + h) * hd + dd];
                }
            }
        }
    }
    (dq, dk, dv)
}

/// Per-channel gated residual `y = x + gate ⊙ h` (`gate_row.wgsl` at
/// `rows_per_cond = rows`, i.e. one `[dim]` gate for every row).
fn gate_rows<T: Fp>(x: &[T], gate: &[T], h: &[T], rows: usize, d: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; rows * d];
    for r in 0..rows {
        for c in 0..d {
            y[r * d + c] = x[r * d + c] + gate[c] * h[r * d + c];
        }
    }
    y
}

/// Gated-residual backward: `dh = gate ⊙ dy`, `dgate += Σ_rows dy⊙h`. `dx` is
/// the identity (the kernel's own documented backward contract), so callers
/// reuse `dy`.
fn gate_bwd<T: Fp>(gate: &[T], h: &[T], dy: &[T], rows: usize, d: usize, dgate: &mut [T]) -> Vec<T> {
    let mut dh = vec![T::ZERO; rows * d];
    for r in 0..rows {
        for c in 0..d {
            let g = dy[r * d + c];
            dh[r * d + c] = gate[c] * g;
            dgate[c] += g * h[r * d + c];
        }
    }
    dh
}

// ---- the block ----

/// Everything the block backward needs from the forward pass.
pub struct BlockCache<T> {
    /// `modulation + e0`, `[6·dim]` - the folded operand, kept so the backward
    /// reads the same six vectors the forward used.
    p: Vec<T>,
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
    actx: Vec<T>,
    ao: Vec<T>,
    xhat3: Vec<T>,
    inv3: Vec<T>,
    n3: Vec<T>,
    ctx: Vec<T>,
    xq: Vec<T>,
    xk: Vec<T>,
    xv: Vec<T>,
    inv_xq: Vec<T>,
    inv_xk: Vec<T>,
    xqn: Vec<T>,
    xkn: Vec<T>,
    xprobs: Vec<T>,
    xctx: Vec<T>,
    xhat2: Vec<T>,
    inv2: Vec<T>,
    n2: Vec<T>,
    h1: Vec<T>,
    hg: Vec<T>,
    ff: Vec<T>,
    cos: Vec<T>,
    sin: Vec<T>,
}

/// `1 + scale` for a modulation site: the gamma the device path folds into the
/// LayerNorm. Chunk `c` of `p` is one of the six `[dim]` vectors.
fn one_plus<T: Fp>(p: &[T], c: usize, d: usize) -> Vec<T> {
    p[c * d..(c + 1) * d].iter().map(|&v| T::ONE + v).collect()
}

fn chunk<T: Fp>(p: &[T], c: usize, d: usize) -> &[T] {
    &p[c * d..(c + 1) * d]
}

/// One block's forward. `x:[t·dim]`, `e0:[6·dim]` (the timestep projection, NOT
/// yet summed with `modulation`), `ctx:[te·dim]` (the embedded text),
/// `cos`/`sin:[t·hd/2]`.
pub fn block_forward<T: Fp>(d: Dims, w: &BlockW<T>, x: &[T], e0: &[T], ctx: &[T], cos: &[T], sin: &[T]) -> (Vec<T>, BlockCache<T>) {
    let (t, te, dim, nh, hd, ffn) = (d.t, d.te, d.dim, d.nh, d.hd(), d.ffn);
    assert_eq!(x.len(), t * dim, "block x size");
    assert_eq!(e0.len(), 6 * dim, "e0 must be [6, dim]");
    assert_eq!(w.modulation.len(), 6 * dim, "modulation must be [6, dim]");
    assert_eq!(ctx.len(), te * dim, "ctx size");
    assert_eq!(cos.len(), t * hd / 2, "rope table size");

    // The fold's operand. `p` chunks: shift1 scale1 gate1 shift2 scale2 gate2.
    let p: Vec<T> = w.modulation.iter().zip(e0).map(|(&a, &b)| a + b).collect();
    let (g1, gate1) = (one_plus(&p, 1, dim), chunk(&p, 2, dim));
    let (g2, gate2) = (one_plus(&p, 4, dim), chunk(&p, 5, dim));

    // --- self-attention ---
    let (xhat1, inv1) = layernorm(x, t, dim, d.eps);
    let n1 = affine(&xhat1, &g1, chunk(&p, 0, dim), t, dim);
    let q = linear(&n1, t, dim, &w.sq.w, &w.sq.b, dim);
    let k = linear(&n1, t, dim, &w.sk.w, &w.sk.b, dim);
    let v = linear(&n1, t, dim, &w.sv.w, &w.sv.b, dim);
    let (qn, inv_q) = rmsnorm(&q, t, dim, &w.snq, d.eps);
    let (kn, inv_k) = rmsnorm(&k, t, dim, &w.snk, d.eps);
    let qr = rope(&qn, t, nh, hd, cos, sin);
    let kr = rope(&kn, t, nh, hd, cos, sin);
    let (probs, actx) = attn_fwd(&qr, t, &kr, &v, t, nh, hd);
    let ao = linear(&actx, t, dim, &w.so.w, &w.so.b, dim);
    let x1 = gate_rows(x, gate1, &ao, t, dim);

    // --- text cross-attention (the residual here is UNGATED) ---
    let (xhat3, inv3) = layernorm(&x1, t, dim, d.eps);
    let n3 = affine(&xhat3, &w.norm3_w, &w.norm3_b, t, dim);
    let xq = linear(&n3, t, dim, &w.cq.w, &w.cq.b, dim);
    let xk = linear(ctx, te, dim, &w.ck.w, &w.ck.b, dim);
    let xv = linear(ctx, te, dim, &w.cv.w, &w.cv.b, dim);
    let (xqn, inv_xq) = rmsnorm(&xq, t, dim, &w.cnq, d.eps);
    let (xkn, inv_xk) = rmsnorm(&xk, te, dim, &w.cnk, d.eps);
    let (xprobs, xctx) = attn_fwd(&xqn, t, &xkn, &xv, te, nh, hd);
    let xo = linear(&xctx, t, dim, &w.co.w, &w.co.b, dim);
    let x2: Vec<T> = x1.iter().zip(&xo).map(|(&a, &b)| a + b).collect();

    // --- FFN ---
    let (xhat2, inv2) = layernorm(&x2, t, dim, d.eps);
    let n2 = affine(&xhat2, &g2, chunk(&p, 3, dim), t, dim);
    let h1 = linear(&n2, t, dim, &w.ff1.w, &w.ff1.b, ffn);
    let hg: Vec<T> = h1.iter().map(|&v| gelu(v)).collect();
    let ff = linear(&hg, t, ffn, &w.ff2.w, &w.ff2.b, dim);
    let out = gate_rows(&x2, gate2, &ff, t, dim);

    let cache = BlockCache {
        p, xhat1, inv1, n1, q, k, v, inv_q, inv_k, qr, kr, probs, actx, ao,
        xhat3, inv3, n3, ctx: ctx.to_vec(), xq, xk, xv, inv_xq, inv_xk, xqn, xkn, xprobs, xctx,
        xhat2, inv2, n2, h1, hg, ff,
        cos: cos.to_vec(), sin: sin.to_vec(),
    };
    (out, cache)
}

/// One block's backward: `dout[t·dim]` -> every weight grad, the six-vector
/// modulation grad (= `d(modulation)` = this block's `d e0` contribution),
/// `dx` and `dctx`.
pub fn block_backward<T: Fp>(d: Dims, w: &BlockW<T>, c: &BlockCache<T>, dout: &[T]) -> BlockGrads<T> {
    let (t, te, dim, nh, hd, ffn) = (d.t, d.te, d.dim, d.nh, d.hd(), d.ffn);
    let mut g = BlockGrads::<T>::zeros(d);
    let (g1, g2) = (one_plus(&c.p, 1, dim), one_plus(&c.p, 4, dim));
    let mut dp = vec![T::ZERO; 6 * dim];

    // out = x2 + gate2 ⊙ ff
    let mut dshift2 = vec![T::ZERO; dim];
    let mut dscale2 = vec![T::ZERO; dim];
    let mut dgate2 = vec![T::ZERO; dim];
    let dff = gate_bwd(chunk(&c.p, 5, dim), &c.ff, dout, t, dim, &mut dgate2);
    let mut dx2 = dout.to_vec();
    // ff = gelu(n2 @ ff1ᵀ + b) @ ff2ᵀ + b
    let (dhg, gff2) = linear_bwd(&c.hg, t, ffn, &w.ff2.w, dim, &dff);
    g.ff2 = gff2;
    let dh1: Vec<T> = dhg.iter().zip(&c.h1).map(|(&gr, &v)| gr * dgelu(v)).collect();
    let (dn2, gff1) = linear_bwd(&c.n2, t, dim, &w.ff1.w, ffn, &dh1);
    g.ff1 = gff1;
    // n2 = (1+scale2)⊙xhat2 + shift2 ; xhat2 = LN(x2)
    let dxhat2 = affine_bwd(&c.xhat2, &g2, t, dim, &dn2, &mut dscale2, &mut dshift2);
    for (a, b) in dx2.iter_mut().zip(layernorm_bwd(&c.xhat2, &c.inv2, t, dim, &dxhat2)) {
        *a += b;
    }

    // x2 = x1 + xo (ungated)
    let mut dx1 = dx2.clone();
    let (dxctx, gco) = linear_bwd(&c.xctx, t, dim, &w.co.w, dim, &dx2);
    g.co = gco;
    let (dxqn, dxkn, dxv) = attn_bwd(&c.xprobs, &c.xqn, &c.xkn, &c.xv, t, te, nh, hd, &dxctx);
    let dxq = rmsnorm_bwd(&c.xq, t, dim, &w.cnq, &c.inv_xq, &dxqn, &mut g.cnq);
    let dxk = rmsnorm_bwd(&c.xk, te, dim, &w.cnk, &c.inv_xk, &dxkn, &mut g.cnk);
    let (dn3, gcq) = linear_bwd(&c.n3, t, dim, &w.cq.w, dim, &dxq);
    g.cq = gcq;
    // k and v both read the SHARED text context: their dctx contributions add.
    let (dctx_k, gck) = linear_bwd(&c.ctx, te, dim, &w.ck.w, dim, &dxk);
    let (dctx_v, gcv) = linear_bwd(&c.ctx, te, dim, &w.cv.w, dim, &dxv);
    g.ck = gck;
    g.cv = gcv;
    for (a, (b, cc)) in g.dctx.iter_mut().zip(dctx_k.iter().zip(&dctx_v)) {
        *a = *b + *cc;
    }
    // n3 = norm3_w ⊙ xhat3 + norm3_b ; xhat3 = LN(x1)
    let dxhat3 = affine_bwd(&c.xhat3, &w.norm3_w, t, dim, &dn3, &mut g.norm3_w, &mut g.norm3_b);
    for (a, b) in dx1.iter_mut().zip(layernorm_bwd(&c.xhat3, &c.inv3, t, dim, &dxhat3)) {
        *a += b;
    }

    // x1 = x + gate1 ⊙ ao
    let mut dshift1 = vec![T::ZERO; dim];
    let mut dscale1 = vec![T::ZERO; dim];
    let mut dgate1 = vec![T::ZERO; dim];
    let dao = gate_bwd(chunk(&c.p, 2, dim), &c.ao, &dx1, t, dim, &mut dgate1);
    let mut dx = dx1.clone();
    let (dactx, gso) = linear_bwd(&c.actx, t, dim, &w.so.w, dim, &dao);
    g.so = gso;
    let (dqr, dkr, dv) = attn_bwd(&c.probs, &c.qr, &c.kr, &c.v, t, t, nh, hd, &dactx);
    let dqn = rope_bwd(&dqr, t, nh, hd, &c.cos, &c.sin);
    let dkn = rope_bwd(&dkr, t, nh, hd, &c.cos, &c.sin);
    let dq = rmsnorm_bwd(&c.q, t, dim, &w.snq, &c.inv_q, &dqn, &mut g.snq);
    let dk = rmsnorm_bwd(&c.k, t, dim, &w.snk, &c.inv_k, &dkn, &mut g.snk);
    let (dn1q, gsq) = linear_bwd(&c.n1, t, dim, &w.sq.w, dim, &dq);
    let (dn1k, gsk) = linear_bwd(&c.n1, t, dim, &w.sk.w, dim, &dk);
    let (dn1v, gsv) = linear_bwd(&c.n1, t, dim, &w.sv.w, dim, &dv);
    g.sq = gsq;
    g.sk = gsk;
    g.sv = gsv;
    let mut dn1 = dn1q;
    for (a, b) in dn1.iter_mut().zip(dn1k) {
        *a += b;
    }
    for (a, b) in dn1.iter_mut().zip(dn1v) {
        *a += b;
    }
    // n1 = (1+scale1)⊙xhat1 + shift1 ; xhat1 = LN(x)
    let dxhat1 = affine_bwd(&c.xhat1, &g1, t, dim, &dn1, &mut dscale1, &mut dshift1);
    for (a, b) in dx.iter_mut().zip(layernorm_bwd(&c.xhat1, &c.inv1, t, dim, &dxhat1)) {
        *a += b;
    }

    // The six sites, in the checkpoint's chunk order. This vector is BOTH
    // `d(modulation)` and this block's contribution to `d e0` - see the header.
    for (i, part) in [dshift1, dscale1, dgate1, dshift2, dscale2, dgate2].iter().enumerate() {
        dp[i * dim..(i + 1) * dim].copy_from_slice(part);
    }
    g.modulation = dp;
    g.dx = dx;
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gelu`/`dgelu` must be the same function `gelu.wgsl` and
    /// `model::gelu_tanh` implement - the FFN's only nonlinearity, and the one
    /// place a wrong constant is invisible in a forward-only test.
    #[test]
    fn gelu_matches_the_kernel_and_its_own_finite_differences() {
        let want = [-0.0454, -0.1543, 0.0, 0.3457, 1.9546];
        for (x, w) in [-2.0f64, -0.5, 0.0, 0.5, 2.0].iter().zip(want) {
            assert!((gelu(*x) - w).abs() < 1e-4, "gelu({x}) = {} vs {w}", gelu::<f64>(*x));
            let h = 1e-6;
            let fd = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            assert!((dgelu(*x) - fd).abs() < 1e-6, "dgelu({x}) = {} vs {fd}", dgelu::<f64>(*x));
        }
    }

    /// The forward must run identically at both instantiations up to f32
    /// rounding: this is what makes the f64 oracle a proof about the f32
    /// trainer rather than about a different function.
    #[test]
    fn silu_and_its_derivative_agree_with_finite_differences() {
        for x in [-3.0f64, -0.25, 0.0, 1.5] {
            let h = 1e-6;
            let fd = (silu(x + h) - silu(x - h)) / (2.0 * h);
            assert!((dsilu(x) - fd).abs() < 1e-6, "dsilu({x})");
        }
    }
}
