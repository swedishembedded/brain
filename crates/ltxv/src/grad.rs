// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host reference forward + analytic backward for ONE video-only `LtxBlock`
//! (`crate::block::LtxBlock` / `self_attn_and_text_ca` + `mlp_sublayer`'s
//! `run_vx` sequence) - the unit the whole video-only DiT training step is
//! built from. The audio stream and the audio<->video cross-attention
//! (`crate::block::LtxAvBlock`) are NOT covered here - see this crate's
//! roadmap ledger for that as a separate, later milestone.
//!
//! The forward mirrors `crate::block`'s op sequence exactly:
//!
//! 1. adaLN-single self-attn modulation: an UNWEIGHTED RMSNorm (`rms_norm(x,
//!    eps)`, no learnable gain) folded with a PER-TOKEN `(1+scale, shift)`
//!    pair.
//! 2. Self-attention: biased QKV, learnable QK-RMSNorm over the full
//!    `inner_dim` (not per head), split/rotate-half RoPE (GPT-NeoX style,
//!    per-head sub-tables - see `crate::rope`), scaled-dot-product attention,
//!    biased output projection.
//! 3. A per-token GATED residual, then the fused re-norm (again an
//!    UNWEIGHTED RMSNorm - the only norm between self-attention and text
//!    cross-attention; there is no separate `norm2`).
//! 4. Text cross-attention with AdaLN modulation: the query side gets a
//!    PER-TOKEN `(1+scale, shift)` pair (same table as step 1, rows 6-7); the
//!    key/value side gets this BLOCK's own STATIC `(1+scale, shift)` pair
//!    (`prompt_scale_shift_table`, broadcast over every context row, NOT
//!    per-token) applied directly to the RAW context - no norm on that side.
//!    No RoPE on either side of this attention. The residual add is gated
//!    (row 8).
//! 5. MLP: another per-token `(1+scale, shift)` pair (rows 3-4) over another
//!    UNWEIGHTED RMSNorm, a bias-free GELU(tanh) FFN at width `4*dim`, a
//!    gated residual (row 5).
//!
//! ## The per-token modulation fold - and what makes it genuinely different
//! from Wan's
//!
//! Wan's block-level backward oracle (`wan::grad`) folds `modulation + e0`
//! into a **per-channel** `(gamma, beta)` pair shared by every token in the
//! sequence - the model-shared conditioning vector `e0` is the SAME for
//! every row. LTX's own fold combines the model-shared per-token table
//! (`adaln_shared`, `[T, 9*dim]`, one row per TOKEN) with this block's own
//! per-block `[9*dim]` table (`w.scale_shift_table`, broadcast identically to
//! every token) - so the SITE gradient (`dcombined`, `[T, 9*dim]`) splits
//! into `d(scale_shift_table) = Σ_rows dcombined` (this block's own trained
//! parameter) and `d(adaln_shared) = dcombined` UNREDUCED (this block's
//! contribution to the shared per-token table every block reads, summed over
//! the whole stack by `crate::modelgrad::backward`) - the token-count-many
//! duality Wan's own doc warns a reused helper would silently get wrong (see
//! `dit::adaln::add_table`'s doc: "going from token-independent to
//! token-dependent is a one-line change to the `rows` argument", but the
//! GRADIENT of that broadcast is genuinely different at `rows=1` vs
//! `rows=T`).
//!
//! `gate_msa`/`gate_mlp`/`gate_q` are themselves per-token `[T,dim]` vectors
//! too (unlike Wan's per-block `[dim]` gate), so the gated residual here is a
//! plain elementwise `y = x + gate⊙h` with NO reduction on the `dgate` side
//! either - see [`gate_elemwise_bwd`].
//!
//! ## One implementation, two instantiations
//!
//! Generic over [`Fp`], same discipline as `wan::grad`: the `f64`
//! instantiation is the finite-difference gradcheck oracle
//! (`gradcheck::check_ltxv`, `crates/ltxv/tests/block_grad.rs`), the `f32`
//! instantiation is the host trainer `crate::finetune` drives. One
//! implementation, so the oracle and the trainer cannot drift apart.
//!
//! This is a deliberate SECOND derivation of the math `crate::block`'s WGSL
//! kernel graph implements (an oracle sharing code with the thing it checks
//! proves nothing) - plain host math, no GPU dispatch at all, written
//! against each kernel's own documented contract: population-variance-free
//! RMSNorm with a runtime eps (`rmsnorm_eps.wgsl`, weight-less at the three
//! modulation sites via an implicit all-ones gain, learnable at QK-norm),
//! `1/sqrt(head_dim)` attention scaling, `gelu.wgsl`'s tanh approximation,
//! and the split/rotate-half rotation `rope2d.wgsl` applies per head from a
//! host-precomputed table (`crate::rope::ltx_rope_tables`).

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Scalar the reference math is generic over. See the module doc for why
/// both instantiations exist and why this is not a duplicate implementation.
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
/// themselves and cross-attend into `te` text rows; `dim = nh·hd`. The FFN
/// width is always `4*dim` (`crate::block::mlp_sublayer`'s hardcoded
/// `ff_dim = dim * 4`, not a separate config field).
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub t: usize,
    pub te: usize,
    pub dim: usize,
    pub nh: usize,
    /// Shared by every RMSNorm/LayerNorm (`LtxDitConfig::norm_eps`).
    pub eps: f64,
}

impl Dims {
    pub fn hd(&self) -> usize {
        self.dim / self.nh
    }
    pub fn ffn(&self) -> usize {
        self.dim * 4
    }
}

/// A biased linear, `[out, in]` row-major plus `[out]` bias - every
/// `to_q`/`to_k`/`to_v`/`to_out.0` projection and `patchify_proj`/
/// `proj_out`/the timestep MLP's two linears carry a bias. Doubles as the
/// gradient container.
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

/// A bias-free linear, `[out, in]` row-major - the FFN's two linears
/// (`ff.net.0.proj`/`ff.net.2`), always bias-free in this milestone's config
/// (`crate::block::FfWeights`/`mlp_sublayer` never dispatch a `bias_add` for
/// either).
#[derive(Clone, Debug, PartialEq)]
pub struct LinNB<T> {
    pub w: Vec<T>,
}

impl<T: Fp> LinNB<T> {
    pub fn zeros(out: usize, inn: usize) -> LinNB<T> {
        LinNB { w: vec![T::ZERO; out * inn] }
    }
}

/// One `Attention` module's trainable tensors (`attn1` or `attn2`) - QKV +
/// output projection, plus the learnable full-`inner_dim` QK-RMSNorm gains.
#[derive(Clone, Debug, PartialEq)]
pub struct AttnW<T> {
    pub q: Lin<T>,
    pub k: Lin<T>,
    pub v: Lin<T>,
    pub o: Lin<T>,
    pub qn: Vec<T>,
    pub kn: Vec<T>,
}

impl<T: Fp> AttnW<T> {
    fn zeros(dim: usize) -> AttnW<T> {
        AttnW { q: Lin::zeros(dim, dim), k: Lin::zeros(dim, dim), v: Lin::zeros(dim, dim), o: Lin::zeros(dim, dim), qn: vec![T::ZERO; dim], kn: vec![T::ZERO; dim] }
    }
}

/// One block's trainable tensors, named as `crate::dit::dit_tensor_manifest`
/// names them (minus the `transformer_blocks.{l}.` prefix, which the model
/// level owns).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockW<T> {
    /// `[9*dim]`: `(shift_msa,scale_msa,gate_msa, shift_mlp,scale_mlp,gate_mlp,
    /// shift_q,scale_q,gate_q)`, added to the model-shared per-token
    /// `adaln_shared` table before the fold - see this module's doc.
    pub scale_shift_table: Vec<T>,
    /// `[2*dim]`: `(shift_kv, scale_kv)`, this block's own STATIC (not
    /// per-token) text-context modulation.
    pub prompt_scale_shift_table: Vec<T>,
    pub attn1: AttnW<T>,
    pub attn2: AttnW<T>,
    pub ff1: LinNB<T>,
    pub ff2: LinNB<T>,
}

impl<T: Fp> BlockW<T> {
    pub fn zeros(dim: usize) -> BlockW<T> {
        BlockW {
            scale_shift_table: vec![T::ZERO; 9 * dim],
            prompt_scale_shift_table: vec![T::ZERO; 2 * dim],
            attn1: AttnW::zeros(dim),
            attn2: AttnW::zeros(dim),
            ff1: LinNB::zeros(4 * dim, dim),
            ff2: LinNB::zeros(dim, 4 * dim),
        }
    }
}

/// Gradients mirroring [`AttnW`].
#[derive(Clone, Debug)]
pub struct AttnGrads<T> {
    pub q: Lin<T>,
    pub k: Lin<T>,
    pub v: Lin<T>,
    pub o: Lin<T>,
    pub qn: Vec<T>,
    pub kn: Vec<T>,
}

impl<T: Fp> AttnGrads<T> {
    fn zeros(dim: usize) -> AttnGrads<T> {
        AttnGrads { q: Lin::zeros(dim, dim), k: Lin::zeros(dim, dim), v: Lin::zeros(dim, dim), o: Lin::zeros(dim, dim), qn: vec![T::ZERO; dim], kn: vec![T::ZERO; dim] }
    }
}

/// Gradients mirroring [`BlockW`], plus the three upstream adjoints: `dx` to
/// the previous block, `dadaln_shared` (this block's contribution to the
/// model-shared per-token table) and `dctx` to the shared text encoding.
#[derive(Clone, Debug)]
pub struct BlockGrads<T> {
    pub scale_shift_table: Vec<T>,
    pub prompt_scale_shift_table: Vec<T>,
    pub attn1: AttnGrads<T>,
    pub attn2: AttnGrads<T>,
    pub ff1: LinNB<T>,
    pub ff2: LinNB<T>,
    pub dx: Vec<T>,
    /// `[t*9*dim]` - see this module's doc on why this is UNREDUCED (unlike
    /// `scale_shift_table`'s own `[9*dim]` gradient).
    pub dadaln_shared: Vec<T>,
    pub dctx: Vec<T>,
}

// ---- primitives (host, generic) ----

pub(crate) fn sigmoid<T: Fp>(x: T) -> T {
    T::ONE / (T::ONE + (-x).exp())
}

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

/// GELU, tanh approximation - the function `gelu.wgsl` implements.
pub(crate) fn gelu<T: Fp>(x: T) -> T {
    let u = T::fr(GELU_K) * (x + T::fr(GELU_C) * x * x * x);
    T::fr(0.5) * x * (T::ONE + u.tanh())
}

pub(crate) fn dgelu<T: Fp>(x: T) -> T {
    let inner = x + T::fr(GELU_C) * x * x * x;
    let th = (T::fr(GELU_K) * inner).tanh();
    let dinner = T::ONE + T::fr(3.0 * GELU_C) * x * x;
    T::fr(0.5) * (T::ONE + th) + T::fr(0.5) * x * (T::ONE - th * th) * T::fr(GELU_K) * dinner
}

/// `y = x @ wᵀ + b`, `x:[rows,inn]`, `w:[out,inn]` -> `y:[rows,out]`.
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

/// Bias-free linear forward - the FFN's own shape.
pub fn linear_nb<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize) -> Vec<T> {
    let mut y = Vec::with_capacity(rows * out);
    for r in 0..rows {
        y.append(&mut T::matvec(w, &x[r * inn..(r + 1) * inn], out, inn));
    }
    y
}

/// [`linear_nb`] backward - the same recipe as [`linear_bwd`] minus the bias.
pub fn linear_nb_bwd<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize, dy: &[T]) -> (Vec<T>, LinNB<T>) {
    let wt = transpose(w, out, inn);
    let mut dx = Vec::with_capacity(rows * inn);
    for r in 0..rows {
        dx.append(&mut T::matvec(&wt, &dy[r * out..(r + 1) * out], inn, out));
    }
    let xt = transpose(x, rows, inn);
    let mut g = LinNB::<T>::zeros(out, inn);
    let mut dyc = vec![T::ZERO; rows];
    for o in 0..out {
        for r in 0..rows {
            dyc[r] = dy[r * out + o];
        }
        let row = T::matvec(&xt, &dyc, inn, rows);
        g.w[o * inn..(o + 1) * inn].copy_from_slice(&row);
    }
    (dx, g)
}

/// RMSNorm with a runtime eps over the last `d` of `[rows,d]`
/// (`rmsnorm_eps.wgsl`): `y = w ⊙ x·inv`, `inv = 1/√(mean(x²)+eps)`. Passing
/// an all-ones `w` (and discarding the returned `dw`) is exactly how
/// `crate::block`'s device path implements the three UNWEIGHTED norm sites
/// (`ada_zero_function`/`post_sa_function`'s "no learnable gain" RMSNorm) -
/// there is no separate no-affine kernel, and there is no separate no-affine
/// host function here either, by the same reasoning.
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

/// `out[r,i] = table[i] + v[r,i]` - the generic-`T` twin of
/// `dit::adaln::add_table` at `rows=T` (per-token, the ONLY mode this
/// milestone uses - LTX's modulation is per-token, never per-forward).
pub fn add_table<T: Fp>(v: &[T], table: &[T], rows: usize, width: usize) -> Vec<T> {
    assert_eq!(v.len(), rows * width, "add_table: v is {}, need {}", v.len(), rows * width);
    assert_eq!(table.len(), width, "add_table: table is {}, need {width}", table.len());
    let mut out = vec![T::ZERO; rows * width];
    for r in 0..rows {
        for i in 0..width {
            out[r * width + i] = table[i] + v[r * width + i];
        }
    }
    out
}

/// Extract sub-plane `i` of a `[rows, k*width]` row-major combined table,
/// viewed as `k` stacked `[rows,width]` planes (`crate::block::slice_row`'s
/// generic-`T` twin).
pub(crate) fn plane<T: Fp>(combined: &[T], rows: usize, width: usize, k: usize, i: usize) -> Vec<T> {
    let mut v = vec![T::ZERO; rows * width];
    for r in 0..rows {
        v[r * width..(r + 1) * width].copy_from_slice(&combined[(r * k + i) * width..(r * k + i) * width + width]);
    }
    v
}

/// [`plane`] plus the modulation site's `1+scale` fold.
pub(crate) fn one_plus_plane<T: Fp>(combined: &[T], rows: usize, width: usize, k: usize, i: usize) -> Vec<T> {
    plane(combined, rows, width, k, i).into_iter().map(|v| T::ONE + v).collect()
}

/// Write `dplane` (`[rows,width]`) into sub-plane `i` of a `[rows,k*width]`
/// combined-table gradient - [`plane`]'s adjoint.
pub(crate) fn write_plane<T: Fp>(dcombined: &mut [T], rows: usize, width: usize, k: usize, i: usize, dplane: &[T]) {
    for r in 0..rows {
        dcombined[(r * k + i) * width..(r * k + i) * width + width].copy_from_slice(&dplane[r * width..(r + 1) * width]);
    }
}

/// PER-TOKEN modulation: `y[r,d] = g[r,d]·xhat[r,d] + b[r,d]`, `g`/`b` the
/// SAME shape as `xhat` (unlike Wan's per-channel fold, where `g`/`b` are
/// `[dim]` broadcast over every row - see this module's doc). `g` is already
/// `1+scale` (the modulation site's own fold); `d(1+scale)/d(scale) == 1` so
/// [`mod_affine_bwd`]'s `dg` output IS `d(scale)` directly.
pub(crate) fn mod_affine<T: Fp>(xhat: &[T], g: &[T], b: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; n];
    for i in 0..n {
        y[i] = g[i] * xhat[i] + b[i];
    }
    y
}

/// [`mod_affine`] backward - no reduction anywhere, since `g`/`b` already
/// match `xhat`'s own shape.
pub(crate) fn mod_affine_bwd<T: Fp>(xhat: &[T], g: &[T], dy: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let n = xhat.len();
    let mut dxhat = vec![T::ZERO; n];
    let mut dg = vec![T::ZERO; n];
    let db = dy.to_vec();
    for i in 0..n {
        dxhat[i] = g[i] * dy[i];
        dg[i] = xhat[i] * dy[i];
    }
    (dxhat, dg, db)
}

/// The block's own STATIC (per-block, not per-token) modulation of the raw
/// text context: `y[r,d] = g[d]·x[r,d] + b[d]`, `g`/`b` shared across every
/// one of the `rows` context rows - the one modulation site in this block
/// that IS a per-channel broadcast (`use_prompt_adaln_single=false`: no
/// timestep MLP drives this table, see `crate::config`'s doc).
pub(crate) fn affine_shared<T: Fp>(x: &[T], g: &[T], b: &[T], rows: usize, d: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; rows * d];
    for r in 0..rows {
        for c in 0..d {
            y[r * d + c] = g[c] * x[r * d + c] + b[c];
        }
    }
    y
}

/// [`affine_shared`] backward: accumulates `dg`/`db` over every row.
pub(crate) fn affine_shared_bwd<T: Fp>(x: &[T], g: &[T], rows: usize, d: usize, dy: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut dx = vec![T::ZERO; rows * d];
    let mut dg = vec![T::ZERO; d];
    let mut db = vec![T::ZERO; d];
    for r in 0..rows {
        for c in 0..d {
            let gr = dy[r * d + c];
            dg[c] += gr * x[r * d + c];
            db[c] += gr;
            dx[r * d + c] = g[c] * gr;
        }
    }
    (dx, dg, db)
}

/// Split/rotate-half RoPE (GPT-NeoX style: pair `(j, j+hd/2)`), per-head
/// SUB-TABLES (`crate::rope::LtxRopeTables`'s `[heads, T, half]` layout,
/// front-padded/band-major/axis-minor construction - see `crate::rope`'s
/// module doc for why this is not `wan::grad::rope`'s single shared table).
/// `x`: `[t, nh*hd]`, heads contiguous per token. `cos`/`sin`: `[nh, t, half]`.
pub fn rope_ltx<T: Fp>(x: &[T], t: usize, nh: usize, hd: usize, cos: &[T], sin: &[T]) -> Vec<T> {
    let half = hd / 2;
    let mut y = x.to_vec();
    for h in 0..nh {
        for ti in 0..t {
            let base = (ti * nh + h) * hd;
            let tab = (h * t + ti) * half;
            for j in 0..half {
                let (c, s) = (cos[tab + j], sin[tab + j]);
                let (x1, x2) = (x[base + j], x[base + half + j]);
                y[base + j] = x1 * c - x2 * s;
                y[base + half + j] = x2 * c + x1 * s;
            }
        }
    }
    y
}

/// [`rope_ltx`] backward - a rotation matrix's inverse is its transpose.
pub fn rope_ltx_bwd<T: Fp>(dy: &[T], t: usize, nh: usize, hd: usize, cos: &[T], sin: &[T]) -> Vec<T> {
    let half = hd / 2;
    let mut dx = dy.to_vec();
    for h in 0..nh {
        for ti in 0..t {
            let base = (ti * nh + h) * hd;
            let tab = (h * t + ti) * half;
            for j in 0..half {
                let (c, s) = (cos[tab + j], sin[tab + j]);
                let (dy1, dy2) = (dy[base + j], dy[base + half + j]);
                dx[base + j] = dy1 * c + dy2 * s;
                dx[base + half + j] = -dy1 * s + dy2 * c;
            }
        }
    }
    dx
}

/// Bidirectional multi-head attention, `nq` query rows into `nk` key rows,
/// scale `1/√hd` - `attn_scores_cross`/`attn_softmax_cross`/`attn_apply_cross`'s
/// exact math (`crate::block::attention`'s device dispatch), non-causal, no
/// mask. Returns `(probs[nh·nq·nk], out[nq·nh·hd])`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_fwd<T: Fp>(q: &[T], nq: usize, k: &[T], v: &[T], nk: usize, nh: usize, hd: usize) -> (Vec<T>, Vec<T>) {
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
pub(crate) fn attn_bwd<T: Fp>(probs: &[T], q: &[T], k: &[T], v: &[T], nq: usize, nk: usize, nh: usize, hd: usize, dout: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
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

/// PER-TOKEN gated residual `y = x + gate⊙h`, `gate` the SAME shape as
/// `x`/`h` (unlike Wan's per-channel `[dim]` gate, broadcast over every
/// row - see `wan::grad::gate_rows`). `dx` is the identity (callers add
/// `dy` directly), so only `(dh, dgate)` are returned.
pub(crate) fn gate_elemwise<T: Fp>(x: &[T], gate: &[T], h: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; n];
    for i in 0..n {
        y[i] = x[i] + gate[i] * h[i];
    }
    y
}

/// [`gate_elemwise`] backward - no reduction on `dgate` either.
pub(crate) fn gate_elemwise_bwd<T: Fp>(gate: &[T], h: &[T], dy: &[T]) -> (Vec<T>, Vec<T>) {
    let n = dy.len();
    let mut dh = vec![T::ZERO; n];
    let mut dgate = vec![T::ZERO; n];
    for i in 0..n {
        dh[i] = gate[i] * dy[i];
        dgate[i] = h[i] * dy[i];
    }
    (dh, dgate)
}

/// ONE-ROW gated residual `y[r,d] = x[r,d] + gate[d]*h[r,d]`, `gate` a SINGLE
/// `[dim]` row broadcast over every one of the `rows` tokens - a THIRD point
/// on the per-forward/per-token spectrum [`gate_elemwise`]'s doc names,
/// exactly `crate::block::gate_row`'s `rows_per_cond = rows` case: the
/// audio<->video cross-attention residual's gate is driven by the CROSS
/// modality's scalar sigma, not a per-token value, so one gate row serves
/// every token of this stream (`crate::block::LtxAvBlock`'s doc, step 3).
pub(crate) fn gate_bcast<T: Fp>(x: &[T], gate: &[T], h: &[T], rows: usize, dim: usize) -> Vec<T> {
    assert_eq!(gate.len(), dim, "gate_bcast: gate must be one [dim] row");
    let mut y = vec![T::ZERO; rows * dim];
    for r in 0..rows {
        for d in 0..dim {
            y[r * dim + d] = x[r * dim + d] + gate[d] * h[r * dim + d];
        }
    }
    y
}

/// [`gate_bcast`] backward: `dh` is per-token (unreduced), `dgate` is the
/// row-SUM over every token - the broadcast's own adjoint, unlike
/// [`gate_elemwise_bwd`]'s per-token `dgate`.
pub(crate) fn gate_bcast_bwd<T: Fp>(gate: &[T], h: &[T], rows: usize, dim: usize, dy: &[T]) -> (Vec<T>, Vec<T>) {
    let mut dh = vec![T::ZERO; rows * dim];
    let mut dgate = vec![T::ZERO; dim];
    for r in 0..rows {
        for d in 0..dim {
            let g = dy[r * dim + d];
            dh[r * dim + d] = gate[d] * g;
            dgate[d] += h[r * dim + d] * g;
        }
    }
    (dh, dgate)
}

// ---- the block ----
//
// Split into two composable phases - self-attention + text-CA (module doc
// steps 1-4) and the MLP sublayer (step 5) - the SAME seam `crate::block`'s
// own device path already draws (`self_attn_and_text_ca`/`mlp_sublayer`, two
// functions shared verbatim by `LtxBlock` and `LtxAvBlock`). `crate::av_grad`
// reuses these two pieces directly for BOTH streams of the AV block, with the
// audio<->video cross-attention step inserted between them - the AV cross
// residual sits at exactly this boundary (`x2`, the state after text-CA and
// before the MLP), so this is the natural interface, not an arbitrary cut.
// [`block_forward`]/[`block_backward`] below are unchanged in behaviour -
// thin wrappers composing the two phases - verified by `crates/ltxv/tests/
// block_grad.rs` and every model-level gate this module's doc already lists.

/// Everything [`self_attn_and_text_ca_bwd`] needs from
/// [`self_attn_and_text_ca_fwd`] - steps 1-4 of the block (self-attention,
/// gated residual, fused re-norm, text cross-attention with AdaLN
/// modulation, gated residual).
pub(crate) struct SattCaCache<T> {
    x0: Vec<T>,
    scale_msa: Vec<T>,
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
    attn1_out: Vec<T>,
    gate_msa: Vec<T>,
    x_fma: Vec<T>,
    xhat_fused: Vec<T>,
    inv_fused: Vec<T>,
    scale_q: Vec<T>,
    attn_input: Vec<T>,
    ctx: Vec<T>,
    enc_hidden: Vec<T>,
    xq2: Vec<T>,
    xk2: Vec<T>,
    xv2: Vec<T>,
    inv_xq: Vec<T>,
    inv_xk: Vec<T>,
    xqn: Vec<T>,
    xkn: Vec<T>,
    xprobs: Vec<T>,
    xctx: Vec<T>,
    ca_raw: Vec<T>,
    gate_q: Vec<T>,
    cos: Vec<T>,
    sin: Vec<T>,
}

/// Grads mirroring [`SattCaCache`]'s owner: both attention modules' weight
/// grads, the six per-token planes this phase writes (`shift_msa`/
/// `scale_msa`/`gate_msa`, `shift_q`/`scale_q`/`gate_q` - planes 0,1,2,6,7,8
/// of the combined 9-row table), this block's own `prompt_scale_shift_table`
/// grad, the upstream `dx` (into the block's own input) and `dctx`.
pub(crate) struct SattCaGrads<T> {
    pub attn1: AttnGrads<T>,
    pub attn2: AttnGrads<T>,
    pub dshift_msa: Vec<T>,
    pub dscale_msa: Vec<T>,
    pub dgate_msa: Vec<T>,
    pub dshift_q: Vec<T>,
    pub dscale_q: Vec<T>,
    pub dgate_q: Vec<T>,
    pub dprompt_scale_shift_table: Vec<T>,
    pub dx: Vec<T>,
    pub dctx: Vec<T>,
}

/// One stream's self-attention + gated residual + fused re-norm + text
/// cross-attention with AdaLN modulation - `crate::block::self_attn_and_
/// text_ca`'s generic-`T` twin, reused for both the video-only path
/// ([`block_forward`]) and both streams of the AV path (`crate::av_grad`).
///
/// `x`: `[t*dim]` this stream's current hidden state. `combined`: `[t,9*dim]`
/// the per-token adaLN-single table ALREADY combined with this block's own
/// `scale_shift_table` (`add_table(adaln_shared, scale_shift_table, t,
/// 9*dim)` - the caller's job, since a caller with more than one such table
/// per block, like [`crate::av_grad`], only wants to build it once). `ctx`:
/// `[te*dim]` this stream's RAW text context. `attn1`/`attn2`: this stream's
/// two attention modules. `prompt_sst`: `[2*dim]` this block's own STATIC
/// text-context modulation. `cos`/`sin`: `[nh, t, hd/2]` self-attention RoPE
/// tables. Returns `(x2, cache)` - `x2` is ALSO this function's own return
/// value (not merely cached), since the AV cross-attention step reads it
/// directly before the MLP ever runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn self_attn_and_text_ca_fwd<T: Fp>(d: Dims, attn1: &AttnW<T>, attn2: &AttnW<T>, prompt_sst: &[T], x: &[T], combined: &[T], ctx: &[T], cos: &[T], sin: &[T]) -> (Vec<T>, SattCaCache<T>) {
    let (t, te, dim, nh, hd) = (d.t, d.te, d.dim, d.nh, d.hd());
    let td = t * dim;
    assert_eq!(x.len(), td, "self_attn_and_text_ca_fwd: x size");
    assert_eq!(combined.len(), t * 9 * dim, "self_attn_and_text_ca_fwd: combined must be [t, 9*dim]");
    assert_eq!(prompt_sst.len(), 2 * dim, "self_attn_and_text_ca_fwd: prompt_sst must be [2*dim]");
    assert_eq!(ctx.len(), te * dim, "self_attn_and_text_ca_fwd: ctx size");
    assert_eq!(cos.len(), nh * t * hd / 2, "self_attn_and_text_ca_fwd: rope table size");

    let shift_msa = plane(combined, t, dim, 9, 0);
    let scale_msa = one_plus_plane(combined, t, dim, 9, 1);
    let gate_msa = plane(combined, t, dim, 9, 2);
    let shift_q = plane(combined, t, dim, 9, 6);
    let scale_q = one_plus_plane(combined, t, dim, 9, 7);
    let gate_q = plane(combined, t, dim, 9, 8);

    let ones = vec![T::ONE; dim];

    // --- self-attention ---
    let (xhat1, inv1) = rmsnorm(x, t, dim, &ones, d.eps);
    let n1 = mod_affine(&xhat1, &scale_msa, &shift_msa, td);
    let q = linear(&n1, t, dim, &attn1.q.w, &attn1.q.b, dim);
    let k = linear(&n1, t, dim, &attn1.k.w, &attn1.k.b, dim);
    let v = linear(&n1, t, dim, &attn1.v.w, &attn1.v.b, dim);
    let (qn, inv_q) = rmsnorm(&q, t, dim, &attn1.qn, d.eps);
    let (kn, inv_k) = rmsnorm(&k, t, dim, &attn1.kn, d.eps);
    let qr = rope_ltx(&qn, t, nh, hd, cos, sin);
    let kr = rope_ltx(&kn, t, nh, hd, cos, sin);
    let (probs, actx) = attn_fwd(&qr, t, &kr, &v, t, nh, hd);
    let attn1_out = linear(&actx, t, dim, &attn1.o.w, &attn1.o.b, dim);
    let x_fma = gate_elemwise(x, &gate_msa, &attn1_out, td);

    // --- fused re-norm (no separate norm2) ---
    let (xhat_fused, inv_fused) = rmsnorm(&x_fma, t, dim, &ones, d.eps);

    // --- text cross-attention with adaLN modulation ---
    let attn_input = mod_affine(&xhat_fused, &scale_q, &shift_q, td);
    let shift_kv = &prompt_sst[0..dim];
    let scale_kv: Vec<T> = prompt_sst[dim..2 * dim].iter().map(|&v| T::ONE + v).collect();
    let enc_hidden = affine_shared(ctx, &scale_kv, shift_kv, te, dim);

    let xq2 = linear(&attn_input, t, dim, &attn2.q.w, &attn2.q.b, dim);
    let xk2 = linear(&enc_hidden, te, dim, &attn2.k.w, &attn2.k.b, dim);
    let xv2 = linear(&enc_hidden, te, dim, &attn2.v.w, &attn2.v.b, dim);
    let (xqn, inv_xq) = rmsnorm(&xq2, t, dim, &attn2.qn, d.eps);
    let (xkn, inv_xk) = rmsnorm(&xk2, te, dim, &attn2.kn, d.eps);
    let (xprobs, xctx) = attn_fwd(&xqn, t, &xkn, &xv2, te, nh, hd);
    let ca_raw = linear(&xctx, t, dim, &attn2.o.w, &attn2.o.b, dim);
    let x2 = gate_elemwise(&x_fma, &gate_q, &ca_raw, td);

    let cache = SattCaCache {
        x0: x.to_vec(), scale_msa, xhat1, inv1, n1, q, k, v, inv_q, inv_k, qr, kr, probs, actx, attn1_out, gate_msa,
        x_fma, xhat_fused, inv_fused, scale_q, attn_input, ctx: ctx.to_vec(), enc_hidden, xq2, xk2, xv2, inv_xq, inv_xk,
        xqn, xkn, xprobs, xctx, ca_raw, gate_q, cos: cos.to_vec(), sin: sin.to_vec(),
    };
    (x2, cache)
}

/// [`self_attn_and_text_ca_fwd`] backward. `dx2`: the COMPLETE gradient
/// flowing into `x2` (both the MLP sublayer's residual passthrough AND its
/// norm branch - [`mlp_bwd`]'s own `dx2` output, or an external loss adjoint
/// for a caller with no MLP downstream, e.g. a block-level gradcheck fixture
/// that only exercises this phase).
pub(crate) fn self_attn_and_text_ca_bwd<T: Fp>(d: Dims, attn1: &AttnW<T>, attn2: &AttnW<T>, prompt_sst: &[T], c: &SattCaCache<T>, dx2: &[T]) -> SattCaGrads<T> {
    let (t, te, dim, nh, hd) = (d.t, d.te, d.dim, d.nh, d.hd());
    let td = t * dim;
    let ones = vec![T::ONE; dim];
    let mut attn1g = AttnGrads::<T>::zeros(dim);
    let mut attn2g = AttnGrads::<T>::zeros(dim);

    // x2 = x_fma + gate_q ⊙ ca_raw
    let (dca_raw, dgate_q) = gate_elemwise_bwd(&c.gate_q, &c.ca_raw, dx2);
    let mut dx_fma = dx2.to_vec();

    let (dxctx, gco) = linear_bwd(&c.xctx, t, dim, &attn2.o.w, dim, &dca_raw);
    attn2g.o = gco;
    let (dxqn, dxkn, dxv2) = attn_bwd(&c.xprobs, &c.xqn, &c.xkn, &c.xv2, t, te, nh, hd, &dxctx);
    let dxq2 = rmsnorm_bwd(&c.xq2, t, dim, &attn2.qn, &c.inv_xq, &dxqn, &mut attn2g.qn);
    let dxk2 = rmsnorm_bwd(&c.xk2, te, dim, &attn2.kn, &c.inv_xk, &dxkn, &mut attn2g.kn);
    let (dattn_input, gcq) = linear_bwd(&c.attn_input, t, dim, &attn2.q.w, dim, &dxq2);
    attn2g.q = gcq;
    let (denc_hidden_k, gck) = linear_bwd(&c.enc_hidden, te, dim, &attn2.k.w, dim, &dxk2);
    let (denc_hidden_v, gcv) = linear_bwd(&c.enc_hidden, te, dim, &attn2.v.w, dim, &dxv2);
    attn2g.k = gck;
    attn2g.v = gcv;
    let mut denc_hidden = vec![T::ZERO; te * dim];
    for i in 0..te * dim {
        denc_hidden[i] = denc_hidden_k[i] + denc_hidden_v[i];
    }

    let scale_kv: Vec<T> = prompt_sst[dim..2 * dim].iter().map(|&v| T::ONE + v).collect();
    let (dctx, dscale_kv, dshift_kv) = affine_shared_bwd(&c.ctx, &scale_kv, te, dim, &denc_hidden);
    let mut dprompt_scale_shift_table = vec![T::ZERO; 2 * dim];
    dprompt_scale_shift_table[0..dim].copy_from_slice(&dshift_kv);
    dprompt_scale_shift_table[dim..2 * dim].copy_from_slice(&dscale_kv);

    let (dxhat_fused_from_ca, dscale_q, dshift_q) = mod_affine_bwd(&c.xhat_fused, &c.scale_q, &dattn_input);
    let mut dw_scratch2 = vec![T::ZERO; dim];
    let dxhat_fused_full = rmsnorm_bwd(&c.x_fma, t, dim, &ones, &c.inv_fused, &dxhat_fused_from_ca, &mut dw_scratch2);
    for i in 0..td {
        dx_fma[i] += dxhat_fused_full[i];
    }

    // x_fma = x + gate_msa ⊙ attn1_out
    let (dattn1_out, dgate_msa) = gate_elemwise_bwd(&c.gate_msa, &c.attn1_out, &dx_fma);
    let mut dx = dx_fma.clone();

    let (dactx, gso) = linear_bwd(&c.actx, t, dim, &attn1.o.w, dim, &dattn1_out);
    attn1g.o = gso;
    let (dqr, dkr, dv) = attn_bwd(&c.probs, &c.qr, &c.kr, &c.v, t, t, nh, hd, &dactx);
    let dqn = rope_ltx_bwd(&dqr, t, nh, hd, &c.cos, &c.sin);
    let dkn = rope_ltx_bwd(&dkr, t, nh, hd, &c.cos, &c.sin);
    let dq = rmsnorm_bwd(&c.q, t, dim, &attn1.qn, &c.inv_q, &dqn, &mut attn1g.qn);
    let dk = rmsnorm_bwd(&c.k, t, dim, &attn1.kn, &c.inv_k, &dkn, &mut attn1g.kn);
    let (dn1q, gsq) = linear_bwd(&c.n1, t, dim, &attn1.q.w, dim, &dq);
    let (dn1k, gsk) = linear_bwd(&c.n1, t, dim, &attn1.k.w, dim, &dk);
    let (dn1v, gsv) = linear_bwd(&c.n1, t, dim, &attn1.v.w, dim, &dv);
    attn1g.q = gsq;
    attn1g.k = gsk;
    attn1g.v = gsv;
    let mut dn1 = dn1q;
    for i in 0..dn1.len() {
        dn1[i] += dn1k[i] + dn1v[i];
    }
    let (dxhat1, dscale_msa, dshift_msa) = mod_affine_bwd(&c.xhat1, &c.scale_msa, &dn1);
    let mut dw_scratch3 = vec![T::ZERO; dim];
    let dxhat1_full = rmsnorm_bwd(&c.x0, t, dim, &ones, &c.inv1, &dxhat1, &mut dw_scratch3);
    for i in 0..td {
        dx[i] += dxhat1_full[i];
    }

    SattCaGrads { attn1: attn1g, attn2: attn2g, dshift_msa, dscale_msa, dgate_msa, dshift_q, dscale_q, dgate_q, dprompt_scale_shift_table, dx, dctx }
}

/// Everything [`mlp_bwd`] needs from [`mlp_fwd`] - the MLP sublayer (module
/// doc step 5).
pub(crate) struct MlpCache<T> {
    x2: Vec<T>,
    scale_mlp: Vec<T>,
    xhat2: Vec<T>,
    inv2: Vec<T>,
    n2: Vec<T>,
    h1: Vec<T>,
    hg: Vec<T>,
    ff_out: Vec<T>,
    gate_mlp: Vec<T>,
}

/// Grads mirroring [`MlpCache`]'s owner: the FFN weight grads, the three
/// per-token planes this phase writes (`shift_mlp`/`scale_mlp`/`gate_mlp` -
/// planes 3,4,5), and `dx2` - the COMPLETE gradient into `x2` (residual
/// passthrough `dout` PLUS the norm branch), [`self_attn_and_text_ca_bwd`]'s
/// own `dx2` input.
pub(crate) struct MlpGrads<T> {
    pub ff1: LinNB<T>,
    pub ff2: LinNB<T>,
    pub dshift_mlp: Vec<T>,
    pub dscale_mlp: Vec<T>,
    pub dgate_mlp: Vec<T>,
    pub dx2: Vec<T>,
}

/// The MLP sublayer, bias-free FFN (the video-only stream's own
/// `ff.net.{0.proj,2}` convention - `crate::block::mlp_sublayer`'s
/// generic-`T` twin at `ff_bias=false`; `crate::av_grad`'s audio stream uses
/// a BIASED sibling instead, since `audio_ff` carries bias regardless of the
/// video-only `ff_bias` flag - see `dit::push_ff`'s doc). `x2`: `[t*dim]`
/// the state this phase modulates (post text-CA). `combined`: the SAME
/// `[t,9*dim]` table [`self_attn_and_text_ca_fwd`] read.
pub(crate) fn mlp_fwd<T: Fp>(d: Dims, ff1: &LinNB<T>, ff2: &LinNB<T>, x2: &[T], combined: &[T]) -> (Vec<T>, MlpCache<T>) {
    let (t, dim) = (d.t, d.dim);
    let td = t * dim;
    let shift_mlp = plane(combined, t, dim, 9, 3);
    let scale_mlp = one_plus_plane(combined, t, dim, 9, 4);
    let gate_mlp = plane(combined, t, dim, 9, 5);
    let ones = vec![T::ONE; dim];

    let (xhat2, inv2) = rmsnorm(x2, t, dim, &ones, d.eps);
    let n2 = mod_affine(&xhat2, &scale_mlp, &shift_mlp, td);
    let h1 = linear_nb(&n2, t, dim, &ff1.w, 4 * dim);
    let hg: Vec<T> = h1.iter().map(|&v| gelu(v)).collect();
    let ff_out = linear_nb(&hg, t, 4 * dim, &ff2.w, dim);
    let out = gate_elemwise(x2, &gate_mlp, &ff_out, td);

    (out, MlpCache { x2: x2.to_vec(), scale_mlp, xhat2, inv2, n2, h1, hg, ff_out, gate_mlp })
}

/// [`mlp_fwd`] backward.
pub(crate) fn mlp_bwd<T: Fp>(d: Dims, ff1: &LinNB<T>, ff2: &LinNB<T>, c: &MlpCache<T>, dout: &[T]) -> MlpGrads<T> {
    let (t, dim) = (d.t, d.dim);
    let td = t * dim;
    let ones = vec![T::ONE; dim];

    // out = x2 + gate_mlp ⊙ ff_out
    let (dff_out, dgate_mlp) = gate_elemwise_bwd(&c.gate_mlp, &c.ff_out, dout);
    let mut dx2 = dout.to_vec();

    let (dhg, ff2g) = linear_nb_bwd(&c.hg, t, 4 * dim, &ff2.w, dim, &dff_out);
    let dh1: Vec<T> = dhg.iter().zip(&c.h1).map(|(&gr, &v)| gr * dgelu(v)).collect();
    let (dn2, ff1g) = linear_nb_bwd(&c.n2, t, dim, &ff1.w, 4 * dim, &dh1);
    let (dxhat2, dscale_mlp, dshift_mlp) = mod_affine_bwd(&c.xhat2, &c.scale_mlp, &dn2);
    let mut dw_scratch = vec![T::ZERO; dim];
    let dxhat2_full = rmsnorm_bwd(&c.x2, t, dim, &ones, &c.inv2, &dxhat2, &mut dw_scratch);
    for i in 0..td {
        dx2[i] += dxhat2_full[i];
    }

    MlpGrads { ff1: ff1g, ff2: ff2g, dshift_mlp, dscale_mlp, dgate_mlp, dx2 }
}

/// Everything the block backward needs from the forward pass.
pub struct BlockCache<T> {
    satt: SattCaCache<T>,
    mlp: MlpCache<T>,
}

/// One block's forward. `x`: `[t*dim]`. `adaln_shared`: `[t*9*dim]`, the
/// model-shared per-token adaLN-single raw table (BEFORE this block's own
/// `scale_shift_table` is added - the fold happens inside, mirroring
/// `crate::block::LtxBlock::forward`'s own `dit::adaln::add_table` call).
/// `ctx`: `[te*dim]`, the RAW text context (this block's own
/// `prompt_scale_shift_table` modulates it fresh, every block).
/// `cos`/`sin`: `[nh, t, hd/2]` (`crate::rope::LtxRopeTables`'s layout).
pub fn block_forward<T: Fp>(d: Dims, w: &BlockW<T>, x: &[T], adaln_shared: &[T], ctx: &[T], cos: &[T], sin: &[T]) -> (Vec<T>, BlockCache<T>) {
    assert_eq!(w.scale_shift_table.len(), 9 * d.dim, "scale_shift_table must be [9*dim]");
    let combined = add_table(adaln_shared, &w.scale_shift_table, d.t, 9 * d.dim);
    let (x2, satt) = self_attn_and_text_ca_fwd(d, &w.attn1, &w.attn2, &w.prompt_scale_shift_table, x, &combined, ctx, cos, sin);
    let (out, mlp) = mlp_fwd(d, &w.ff1, &w.ff2, &x2, &combined);
    (out, BlockCache { satt, mlp })
}

/// One block's backward: `dout[t*dim]` -> every weight grad, `dx`,
/// `dadaln_shared` (this block's contribution to the shared per-token
/// table) and `dctx`.
pub fn block_backward<T: Fp>(d: Dims, w: &BlockW<T>, c: &BlockCache<T>, dout: &[T]) -> BlockGrads<T> {
    let (t, dim) = (d.t, d.dim);
    let mg = mlp_bwd(d, &w.ff1, &w.ff2, &c.mlp, dout);
    let sg = self_attn_and_text_ca_bwd(d, &w.attn1, &w.attn2, &w.prompt_scale_shift_table, &c.satt, &mg.dx2);

    let mut dcombined = vec![T::ZERO; t * 9 * dim];
    write_plane(&mut dcombined, t, dim, 9, 0, &sg.dshift_msa);
    write_plane(&mut dcombined, t, dim, 9, 1, &sg.dscale_msa);
    write_plane(&mut dcombined, t, dim, 9, 2, &sg.dgate_msa);
    write_plane(&mut dcombined, t, dim, 9, 3, &mg.dshift_mlp);
    write_plane(&mut dcombined, t, dim, 9, 4, &mg.dscale_mlp);
    write_plane(&mut dcombined, t, dim, 9, 5, &mg.dgate_mlp);
    write_plane(&mut dcombined, t, dim, 9, 6, &sg.dshift_q);
    write_plane(&mut dcombined, t, dim, 9, 7, &sg.dscale_q);
    write_plane(&mut dcombined, t, dim, 9, 8, &sg.dgate_q);

    // Split the site gradient: the block's own STATIC table is the row-sum,
    // the model-shared per-token table's contribution is the UNREDUCED
    // tensor (see this module's doc).
    let mut scale_shift_table = vec![T::ZERO; 9 * dim];
    for r in 0..t {
        for i in 0..9 * dim {
            scale_shift_table[i] += dcombined[r * 9 * dim + i];
        }
    }

    BlockGrads {
        scale_shift_table,
        prompt_scale_shift_table: sg.dprompt_scale_shift_table,
        attn1: sg.attn1,
        attn2: sg.attn2,
        ff1: mg.ff1,
        ff2: mg.ff2,
        dx: sg.dx,
        dadaln_shared: dcombined,
        dctx: sg.dctx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gelu`/`dgelu` must be the same function `gelu.wgsl` implements - the
    /// FFN's only nonlinearity, and the one place a wrong constant is
    /// invisible in a forward-only test.
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

    #[test]
    fn silu_and_its_derivative_agree_with_finite_differences() {
        for x in [-3.0f64, -0.25, 0.0, 1.5] {
            let h = 1e-6;
            let fd = (silu(x + h) - silu(x - h)) / (2.0 * h);
            assert!((dsilu(x) - fd).abs() < 1e-6, "dsilu({x})");
        }
    }

    /// `rope_ltx` must be a unit rotation (orthogonal), the structural
    /// invariant that makes [`rope_ltx_bwd`]'s "transpose is the inverse"
    /// derivation valid in the first place.
    #[test]
    fn rope_ltx_preserves_squared_norm() {
        let (t, nh, hd) = (3usize, 2usize, 4usize);
        let half = hd / 2;
        let x: Vec<f64> = (0..t * nh * hd).map(|i| (i as f64 * 0.37).sin()).collect();
        let cos: Vec<f64> = (0..nh * t * half).map(|i| (i as f64 * 0.19).cos()).collect();
        let sin: Vec<f64> = (0..nh * t * half).map(|i| (i as f64 * 0.19).sin()).collect();
        // Normalize (cos,sin) pairs onto the unit circle first - arbitrary
        // cos/sin from unrelated trig calls are not itself a rotation.
        let (mut cosn, mut sinn) = (cos.clone(), sin.clone());
        for i in 0..cos.len() {
            let n = (cos[i] * cos[i] + sin[i] * sin[i]).sqrt();
            cosn[i] = cos[i] / n;
            sinn[i] = sin[i] / n;
        }
        let y = rope_ltx(&x, t, nh, hd, &cosn, &sinn);
        let nx: f64 = x.iter().map(|v| v * v).sum();
        let ny: f64 = y.iter().map(|v| v * v).sum();
        assert!((nx - ny).abs() < 1e-9, "rope must preserve squared norm: {nx} vs {ny}");
    }
}
