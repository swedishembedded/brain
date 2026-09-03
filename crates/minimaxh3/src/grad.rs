// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3 DiT core host **training** reference: forward + hand-derived
//! analytic backward, generic over a float type `T` - `f64` is the
//! finite-difference gradcheck oracle (`gradcheck::check_minimaxh3`/
//! `check_minimaxh3_conditioning`), `f32` is the eventual host trainer. One
//! implementation, two instantiations, the same discipline
//! `crates/ltxv/src/{grad,av_modelgrad}.rs`/`crates/wan/src/modelgrad.rs`
//! already establish in this workspace.
//!
//! This is a deliberate SECOND derivation of the block math
//! [`crate::block`]/[`crate::model`]'s device kernel graph implements - an FD
//! oracle sharing code with the thing it checks proves nothing (porting.md
//! §8). Nothing in this module calls [`crate::block`] or [`crate::model`].
//! Two small, genuinely non-differentiable pieces ARE reused directly,
//! exactly as `ltxv::av_modelgrad` reuses `crate::rope::ltx_rope_tables` and
//! `crate::modelgrad::timestep_embedding` reuses no device code either: the
//! RoPE cos/sin table builder ([`crate::rope::build_tables`] - a pure
//! function of the config and the non-trainable `position_ids`, zero
//! trainable parameters) and the per-row AdaLN table address
//! ([`crate::model::adaln_indices`] - integer arithmetic on non-trainable
//! `token_tags`/`timestep_indices`, this port's OWN row-order convention,
//! see [`crate::block`]'s doc). Sharing these does not let a wrong gradient
//! hide, because neither one is differentiated.
//!
//! ## Scope - what this covers and what it does not
//!
//! The DiT core only: `proj_in`/`audio_proj_in`/`context_embedder`, the
//! token refiner, the `num_layers` main block stack (RoPE'd, per-row
//! AdaLN-Zero modulated self-attention + SwiGLU FFN, exactly
//! [`crate::block::block_forward`]'s op sequence), `norm_out`, and the two
//! output heads - see [`crate::model`]'s own module doc for why
//! packed-sequence LAYOUT is a later phase's job, not this one's: this
//! module's own [`Batch`]/[`make_flow_batch`] builds a FIXED synthetic
//! packing (`[text | audio | video]`) sufficient to gradient-check the DiT
//! core, not the real `t2va`/`fl2va` layout `crate::model::PackedInputs`
//! expects at real-checkpoint time.
//!
//! ## Genuine per-row diffusion forcing, not a per-stream scalar
//!
//! Unlike `ltxv::av_modelgrad`'s two-STREAM model (one scalar sigma per
//! stream), H3's own AdaLN indexing is genuinely per-ROW (the roadmap's
//! "true diffusion-forcing": `timestep_indices` is `[seq_len]`, not a
//! per-modality scalar). [`make_flow_batch`] exercises this for real: video
//! and audio rows alternate between BOTH of the batch's distinct timesteps
//! by row parity (mirroring `crate::model`'s own tiny smoke test's
//! convention), each row noised at ITS OWN timestep's sigma - not one
//! shared per-modality value. This is exactly the shape that would break if
//! [`crate::model::adaln_indices`]'s row-order convention were wired wrong
//! anywhere in this module's own re-derivation.
//!
//! ## The AdaLN fold - three model-shared sites, not one
//!
//! `temb_silu` (`silu(time_embedder(timestep))`, `[num_timesteps,
//! time_embed_dim]`) is read by THREE kinds of consumer: every block's own
//! `adaln_proj` (private per-block weight, but the SAME `temb_silu` row),
//! and `norm_out`'s own shift/scale projection. Each block's
//! [`block_backward`] therefore returns its own UNREDUCED contribution to
//! `d(temb_silu)` (this module's twin of `ltxv::grad`'s `dadaln_shared`
//! duality), and [`backward`] sums every block's contribution plus
//! `norm_out`'s own before routing the total back through
//! [`build_temb_bwd`] into `time_embedder.linear_{1,2}`'s weight grads -
//! exactly the T5 `rel_bias`-fold failure shape porting.md §8/the T5
//! precedent warns about, and exactly what
//! `gradcheck::check_minimaxh3_conditioning`'s per-entry elementwise check
//! (rather than [`gradcheck::check_minimaxh3`]'s per-tensor directional
//! contraction) exists to catch.
//!
//! Within one block, the per-row modulation gather (`shift_msa` etc., one
//! row per packed-sequence token, addressed by `adaln_indices` into a
//! `[MODALITY_NUM*num_timesteps, hidden]` table) has the SAME "many tokens
//! read one shared row" duality LTX's own `adaln_shared` doc explains: the
//! table's own gradient is a scatter-ADD over every token that read a given
//! row ([`gather_rows_bwd`]), not a plain copy.
//!
//! Swedish Embedded AB implements this diffusion-transformer training
//! scaffold for its clients. If your team needs expertise in gradient-
//! checking ported transformer backbones or building host training
//! references for new architectures, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::config::{H3TransformerConfig, MODALITY_NUM, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
use crate::model::adaln_indices;
use crate::rope::build_tables;

const TIME_MAX_PERIOD: f64 = 10000.0;

// ---- scalar abstraction ----

/// Scalar the reference math is generic over. See the module doc for why
/// both instantiations exist.
pub trait Fp: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + AddAssign + 'static {
    const ZERO: Self;
    const ONE: Self;
    fn fr(v: f64) -> Self;
    fn f64(self) -> f64;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
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

fn sigmoid<T: Fp>(x: T) -> T {
    T::ONE / (T::ONE + (-x).exp())
}

fn silu<T: Fp>(x: T) -> T {
    x * sigmoid(x)
}

fn dsilu<T: Fp>(x: T) -> T {
    let s = sigmoid(x);
    s + x * s * (T::ONE - s)
}

// ---- linear algebra primitives ----

fn transpose<T: Fp>(w: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

/// `y = x @ wᵀ + b`, `x:[rows,inn]`, `w:[out,inn]` -> `y:[rows,out]`.
fn linear<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], b: &[T], out: usize) -> Vec<T> {
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

/// A biased linear's weight+bias pair, `[out,in]` row-major - doubles as the
/// gradient container.
#[derive(Clone, Debug, PartialEq)]
pub struct Lin<T> {
    pub w: Vec<T>,
    pub b: Vec<T>,
}

/// Biased-linear backward: `dx = dy @ w`, `dw = dyᵀ @ x`, `db = Σ_rows dy`.
fn linear_bwd<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize, dy: &[T]) -> (Vec<T>, Lin<T>) {
    let wt = transpose(w, out, inn);
    let mut dx = Vec::with_capacity(rows * inn);
    for r in 0..rows {
        dx.append(&mut T::matvec(&wt, &dy[r * out..(r + 1) * out], inn, out));
    }
    let xt = transpose(x, rows, inn);
    let mut g = Lin { w: vec![T::ZERO; out * inn], b: vec![T::ZERO; out] };
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

/// Bias-free linear's weight, `[out,in]` row-major - every attention/FFN
/// projection (`bias=False` throughout the real checkpoint).
#[derive(Clone, Debug, PartialEq)]
pub struct LinNB<T> {
    pub w: Vec<T>,
}

/// Bias-free linear forward.
fn linear_nb<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize) -> Vec<T> {
    let mut y = Vec::with_capacity(rows * out);
    for r in 0..rows {
        y.append(&mut T::matvec(w, &x[r * inn..(r + 1) * inn], out, inn));
    }
    y
}

/// [`linear_nb`] backward.
fn linear_nb_bwd<T: Fp>(x: &[T], rows: usize, inn: usize, w: &[T], out: usize, dy: &[T]) -> (Vec<T>, LinNB<T>) {
    let wt = transpose(w, out, inn);
    let mut dx = Vec::with_capacity(rows * inn);
    for r in 0..rows {
        dx.append(&mut T::matvec(&wt, &dy[r * out..(r + 1) * out], inn, out));
    }
    let xt = transpose(x, rows, inn);
    let mut g = LinNB { w: vec![T::ZERO; out * inn] };
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
/// (`rmsnorm_eps.wgsl`'s contract): `y = w ⊙ x·inv`, `inv = 1/√(mean(x²)+eps)`.
/// Every RMSNorm site in H3 carries a real learnable gain (unlike ltxv's
/// three UNweighted adaLN-zero norms) - there is no separate no-affine path.
fn rmsnorm<T: Fp>(x: &[T], rows: usize, d: usize, w: &[T], eps: f64) -> (Vec<T>, Vec<T>) {
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

/// [`rmsnorm`] backward. Accumulates the scale grad into `dw` (len `d`).
fn rmsnorm_bwd<T: Fp>(x: &[T], rows: usize, d: usize, w: &[T], inv: &[T], dy: &[T], dw: &mut [T]) -> Vec<T> {
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

/// PER-TOKEN modulation `y[i] = g[i]·xhat[i] + b[i]`, `g`/`b` the same shape
/// as `xhat` - `g` is already `1+scale`, so [`mod_affine_bwd`]'s `dg` output
/// IS `d(scale)` directly (`d(1+scale)/d(scale) == 1`).
fn mod_affine<T: Fp>(xhat: &[T], g: &[T], b: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; n];
    for i in 0..n {
        y[i] = g[i] * xhat[i] + b[i];
    }
    y
}

fn mod_affine_bwd<T: Fp>(xhat: &[T], g: &[T], dy: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let n = xhat.len();
    let mut dxhat = vec![T::ZERO; n];
    let mut dg = vec![T::ZERO; n];
    for i in 0..n {
        dxhat[i] = g[i] * dy[i];
        dg[i] = xhat[i] * dy[i];
    }
    (dxhat, dg, dy.to_vec())
}

/// PER-TOKEN gated residual `y = x + gate⊙h`.
fn gate_elemwise<T: Fp>(x: &[T], gate: &[T], h: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::ZERO; n];
    for i in 0..n {
        y[i] = x[i] + gate[i] * h[i];
    }
    y
}

fn gate_elemwise_bwd<T: Fp>(gate: &[T], h: &[T], dy: &[T]) -> (Vec<T>, Vec<T>) {
    let n = dy.len();
    let mut dh = vec![T::ZERO; n];
    let mut dgate = vec![T::ZERO; n];
    for i in 0..n {
        dh[i] = gate[i] * dy[i];
        dgate[i] = h[i] * dy[i];
    }
    (dh, dgate)
}

// ---- row gather/scatter (the AdaLN table addressing + sequence packing) ----

/// `out[i,:] = table[idx[i],:]`.
fn gather_rows<T: Fp>(table: &[T], idx: &[usize], d: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; idx.len() * d];
    for (i, &r) in idx.iter().enumerate() {
        out[i * d..(i + 1) * d].copy_from_slice(&table[r * d..(r + 1) * d]);
    }
    out
}

/// [`gather_rows`] backward - a SCATTER-ADD (many rows can share one `idx`
/// value, e.g. every video row at a given timestep reads the same AdaLN
/// table row; see this module's doc).
fn gather_rows_bwd<T: Fp>(dout: &[T], idx: &[usize], d: usize, table_rows: usize) -> Vec<T> {
    let mut dtable = vec![T::ZERO; table_rows * d];
    for (i, &r) in idx.iter().enumerate() {
        for c in 0..d {
            dtable[r * d + c] += dout[i * d + c];
        }
    }
    dtable
}

/// `out[idx[i],:] = narrow[i,:]`, zero elsewhere - `idx` assumed disjoint
/// (packing each modality's own rows into the shared sequence buffer, the
/// reference's `index_copy`). Its own backward is exactly [`gather_rows`]
/// applied to the wide gradient at the same `idx` - no separate function
/// needed, one-to-one indices need no accumulation.
fn scatter_rows<T: Fp>(narrow: &[T], idx: &[usize], d: usize, wide_rows: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; wide_rows * d];
    for (i, &r) in idx.iter().enumerate() {
        out[r * d..(r + 1) * d].copy_from_slice(&narrow[i * d..(i + 1) * d]);
    }
    out
}

// ---- RoPE: table-driven partial rotation, shared across every head ----

/// `crates/kernels/wgsl/rope2d_partial.wgsl`'s exact contract at `tmod ==
/// rows` (every packed-sequence row owns its own table row, no wraparound):
/// rotate the leading `2*half` channels of every `head_dim`-wide head, one
/// cos/sin table SHARED across every head of a row (`[seq_len,half]`,
/// [`crate::rope::build_tables`]'s layout) - unlike ltxv's per-head
/// sub-tables. Channels `[2*half, head_dim)` pass through unrotated.
fn rope_h3<T: Fp>(x: &[T], seq_len: usize, heads: usize, head_dim: usize, half: usize, cos: &[T], sin: &[T]) -> Vec<T> {
    let mut y = x.to_vec();
    for r in 0..seq_len {
        for h in 0..heads {
            let base = r * heads * head_dim + h * head_dim;
            let tab = r * half;
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

/// [`rope_h3`] backward - a rotation matrix's inverse is its transpose.
fn rope_h3_bwd<T: Fp>(dy: &[T], seq_len: usize, heads: usize, head_dim: usize, half: usize, cos: &[T], sin: &[T]) -> Vec<T> {
    let mut dx = dy.to_vec();
    for r in 0..seq_len {
        for h in 0..heads {
            let base = r * heads * head_dim + h * head_dim;
            let tab = r * half;
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

// ---- full bidirectional self-attention (no mask, no cross-attention anywhere in H3) ----

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

// ---- attention module weights (bias-free q/k/v/o + qk-norm gains) ----

#[derive(Clone, Debug, PartialEq)]
pub struct AttnW<T> {
    pub q: LinNB<T>,
    pub k: LinNB<T>,
    pub v: LinNB<T>,
    pub o: LinNB<T>,
    /// Per-head RMSNorm gain, `[head_dim]`.
    pub qn: Vec<T>,
    pub kn: Vec<T>,
}

/// Grads mirroring [`AttnW`].
pub struct AttnGrads<T> {
    pub q: LinNB<T>,
    pub k: LinNB<T>,
    pub v: LinNB<T>,
    pub o: LinNB<T>,
    pub qn: Vec<T>,
    pub kn: Vec<T>,
}

// ---- token refiner block (plain pre-norm, no AdaLN, no RoPE) ----

/// One `MiniMaxH3TokenRefinerBlock`'s trainable tensors, named as
/// `crate::model::H3Transformer::load` names them (minus the
/// `token_refiner.refiner_blocks.{i}.` prefix). `fc1` is the FUSED
/// `[2*ffn,hidden]` SwiGLU projection (chunked `(value,gate)` at forward
/// time, `crate::block`'s own doc on why this port never offset-views it at
/// the wrong stage) - never split into two host tensors, so no name is
/// invented that the real checkpoint's `ff.net.0.proj.weight` does not have.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinerBlockW<T> {
    pub attn: AttnW<T>,
    pub norm1: Vec<T>,
    pub norm2: Vec<T>,
    pub fc1: LinNB<T>,
    pub fc2: LinNB<T>,
}

/// Grads mirroring [`RefinerBlockW`].
pub struct RefinerBlockGrads<T> {
    pub attn: AttnGrads<T>,
    pub norm1: Vec<T>,
    pub norm2: Vec<T>,
    pub fc1: LinNB<T>,
    pub fc2: LinNB<T>,
}

struct RefinerBlockCache<T> {
    x: Vec<T>,
    inv1: Vec<T>,
    xhat1: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    inv_q: Vec<T>,
    inv_k: Vec<T>,
    qn: Vec<T>,
    kn: Vec<T>,
    probs: Vec<T>,
    actx: Vec<T>,
    x1: Vec<T>,
    inv2: Vec<T>,
    xhat2: Vec<T>,
    value: Vec<T>,
    gate: Vec<T>,
    act: Vec<T>,
}

fn refiner_block_forward<T: Fp>(cfg: &Cfg, w: &RefinerBlockW<T>, x: &[T]) -> (Vec<T>, RefinerBlockCache<T>) {
    let (hidden, heads, head_dim, inner, ffn) = (cfg.hidden(), cfg.heads(), cfg.head_dim(), cfg.inner(), cfg.ffn());
    let seq_len = x.len() / hidden;

    let (xhat1, inv1) = rmsnorm(x, seq_len, hidden, &w.norm1, cfg.norm_eps());
    let q = linear_nb(&xhat1, seq_len, hidden, &w.attn.q.w, inner);
    let k = linear_nb(&xhat1, seq_len, hidden, &w.attn.k.w, inner);
    let v = linear_nb(&xhat1, seq_len, hidden, &w.attn.v.w, inner);
    let (qn, inv_q) = rmsnorm(&q, seq_len * heads, head_dim, &w.attn.qn, cfg.qk_norm_eps());
    let (kn, inv_k) = rmsnorm(&k, seq_len * heads, head_dim, &w.attn.kn, cfg.qk_norm_eps());
    let (probs, actx) = attn_fwd(&qn, seq_len, &kn, &v, seq_len, heads, head_dim);
    let attn_out = linear_nb(&actx, seq_len, inner, &w.attn.o.w, hidden);
    let x1: Vec<T> = x.iter().zip(&attn_out).map(|(&a, &b)| a + b).collect();

    let (xhat2, inv2) = rmsnorm(&x1, seq_len, hidden, &w.norm2, cfg.norm_eps());
    let (fc1_value_w, fc1_gate_w) = w.fc1.w.split_at(ffn * hidden);
    let value = linear_nb(&xhat2, seq_len, hidden, fc1_value_w, ffn);
    let gate = linear_nb(&xhat2, seq_len, hidden, fc1_gate_w, ffn);
    let act: Vec<T> = value.iter().zip(&gate).map(|(&val, &g)| val * silu(g)).collect();
    let ff_out = linear_nb(&act, seq_len, ffn, &w.fc2.w, hidden);
    let out: Vec<T> = x1.iter().zip(&ff_out).map(|(&a, &b)| a + b).collect();

    (out, RefinerBlockCache { x: x.to_vec(), inv1, xhat1, q, k, v, inv_q, inv_k, qn, kn, probs, actx, x1, inv2, xhat2, value, gate, act })
}

fn refiner_block_backward<T: Fp>(cfg: &Cfg, w: &RefinerBlockW<T>, c: &RefinerBlockCache<T>, dout: &[T]) -> (Vec<T>, RefinerBlockGrads<T>) {
    let (hidden, heads, head_dim, inner, ffn) = (cfg.hidden(), cfg.heads(), cfg.head_dim(), cfg.inner(), cfg.ffn());
    let seq_len = dout.len() / hidden;
    let td = seq_len * hidden;

    let mut dx1 = dout.to_vec();
    let (fc1_value_w, fc1_gate_w) = w.fc1.w.split_at(ffn * hidden);
    let (dact, g_fc2) = linear_nb_bwd(&c.act, seq_len, ffn, &w.fc2.w, hidden, dout);
    let dgate: Vec<T> = dact.iter().zip(&c.value).zip(&c.gate).map(|((&da, &val), &g)| da * val * dsilu(g)).collect();
    let dvalue: Vec<T> = dact.iter().zip(&c.gate).map(|(&da, &g)| da * silu(g)).collect();
    let (dxhat2_g, g_fc1_gate) = linear_nb_bwd(&c.xhat2, seq_len, hidden, fc1_gate_w, ffn, &dgate);
    let (dxhat2_v, g_fc1_value) = linear_nb_bwd(&c.xhat2, seq_len, hidden, fc1_value_w, ffn, &dvalue);
    let mut dxhat2 = vec![T::ZERO; td];
    for i in 0..td {
        dxhat2[i] = dxhat2_g[i] + dxhat2_v[i];
    }
    let mut g_norm2 = vec![T::ZERO; hidden];
    let dxhat2_full = rmsnorm_bwd(&c.x1, seq_len, hidden, &w.norm2, &c.inv2, &dxhat2, &mut g_norm2);
    for i in 0..td {
        dx1[i] += dxhat2_full[i];
    }
    let mut g_fc1_w = vec![T::ZERO; 2 * ffn * hidden];
    g_fc1_w[..ffn * hidden].copy_from_slice(&g_fc1_value.w);
    g_fc1_w[ffn * hidden..].copy_from_slice(&g_fc1_gate.w);

    let dattn_out = dx1.clone();
    let mut dx = dx1;

    let (dactx, g_o) = linear_nb_bwd(&c.actx, seq_len, inner, &w.attn.o.w, hidden, &dattn_out);
    let (dqn, dkn, dv) = attn_bwd(&c.probs, &c.qn, &c.kn, &c.v, seq_len, seq_len, heads, head_dim, &dactx);
    let mut g_qn = vec![T::ZERO; head_dim];
    let dq = rmsnorm_bwd(&c.q, seq_len * heads, head_dim, &w.attn.qn, &c.inv_q, &dqn, &mut g_qn);
    let mut g_kn = vec![T::ZERO; head_dim];
    let dk = rmsnorm_bwd(&c.k, seq_len * heads, head_dim, &w.attn.kn, &c.inv_k, &dkn, &mut g_kn);

    let (dxhat1_q, g_q) = linear_nb_bwd(&c.xhat1, seq_len, hidden, &w.attn.q.w, inner, &dq);
    let (dxhat1_k, g_k) = linear_nb_bwd(&c.xhat1, seq_len, hidden, &w.attn.k.w, inner, &dk);
    let (dxhat1_v, g_v) = linear_nb_bwd(&c.xhat1, seq_len, hidden, &w.attn.v.w, inner, &dv);
    let mut dxhat1 = vec![T::ZERO; td];
    for i in 0..td {
        dxhat1[i] = dxhat1_q[i] + dxhat1_k[i] + dxhat1_v[i];
    }
    let mut g_norm1 = vec![T::ZERO; hidden];
    let dxhat1_full = rmsnorm_bwd(&c.x, seq_len, hidden, &w.norm1, &c.inv1, &dxhat1, &mut g_norm1);
    for i in 0..td {
        dx[i] += dxhat1_full[i];
    }

    (dx, RefinerBlockGrads { attn: AttnGrads { q: g_q, k: g_k, v: g_v, o: g_o, qn: g_qn, kn: g_kn }, norm1: g_norm1, norm2: g_norm2, fc1: LinNB { w: g_fc1_w }, fc2: g_fc2 })
}

// ---- AdaLN table build (per-block adaln_proj -> six modality-major tables) ----

/// This block's six per-(modality,timestep) modulation tables, each
/// row-major `[MODALITY_NUM*num_timesteps, hidden]`, [`crate::block::
/// adaln_tables`]'s generic-`T`, independently re-derived twin (same
/// modality-major row order, see that module's own doc for why).
fn adaln_tables<T: Fp>(w: &Lin<T>, temb_silu: &[T], num_ts: usize, hidden: usize, te: usize) -> [Vec<T>; 6] {
    let mut tables: [Vec<T>; 6] = Default::default();
    for t in tables.iter_mut() {
        *t = vec![T::ZERO; MODALITY_NUM as usize * num_ts * hidden];
    }
    for modality in 0..MODALITY_NUM as usize {
        for (param, table) in tables.iter_mut().enumerate() {
            let feat_row0 = (modality * 6 + param) * hidden;
            let w_slice = &w.w[feat_row0 * te..(feat_row0 + hidden) * te];
            let b_slice = &w.b[feat_row0..feat_row0 + hidden];
            let out = linear_nb(temb_silu, num_ts, te, w_slice, hidden);
            for t_idx in 0..num_ts {
                for c in 0..hidden {
                    table[(modality * num_ts + t_idx) * hidden + c] = out[t_idx * hidden + c] + b_slice[c];
                }
            }
        }
    }
    tables
}

/// [`adaln_tables`] backward. `d_tables`: each table's gradient, ALREADY
/// reduced across every packed-sequence row that read it (the caller's
/// [`gather_rows_bwd`] scatter-add - see this module's doc). Returns
/// `(d(adaln_proj), d(temb_silu))`.
fn adaln_tables_bwd<T: Fp>(w: &Lin<T>, temb_silu: &[T], num_ts: usize, hidden: usize, te: usize, d_tables: &[Vec<T>; 6]) -> (Lin<T>, Vec<T>) {
    let mut gw = vec![T::ZERO; w.w.len()];
    let mut gb = vec![T::ZERO; w.b.len()];
    let mut d_temb_silu = vec![T::ZERO; num_ts * te];
    for modality in 0..MODALITY_NUM as usize {
        for (param, dtable) in d_tables.iter().enumerate() {
            let feat_row0 = (modality * 6 + param) * hidden;
            let w_slice = &w.w[feat_row0 * te..(feat_row0 + hidden) * te];
            let mut d_out = vec![T::ZERO; num_ts * hidden];
            for t_idx in 0..num_ts {
                for c in 0..hidden {
                    d_out[t_idx * hidden + c] = dtable[(modality * num_ts + t_idx) * hidden + c];
                }
            }
            for c in 0..hidden {
                let mut acc = T::ZERO;
                for row in d_out.iter().skip(c).step_by(hidden) {
                    acc += *row;
                }
                gb[feat_row0 + c] = acc;
            }
            let (d_temb_silu_slice, g_w_slice) = linear_nb_bwd(temb_silu, num_ts, te, w_slice, hidden, &d_out);
            gw[feat_row0 * te..(feat_row0 + hidden) * te].copy_from_slice(&g_w_slice.w);
            for i in 0..num_ts * te {
                d_temb_silu[i] += d_temb_silu_slice[i];
            }
        }
    }
    (Lin { w: gw, b: gb }, d_temb_silu)
}

// ---- the main block (RoPE'd, per-row AdaLN-Zero modulated) ----

/// One `MiniMaxH3TransformerBlock`'s trainable tensors, named as
/// `crate::model::H3Transformer::load` names them (minus the
/// `transformer_blocks.{i}.` prefix).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockW<T> {
    pub attn: AttnW<T>,
    pub norm1: Vec<T>,
    pub norm2: Vec<T>,
    pub fc1: LinNB<T>,
    pub fc2: LinNB<T>,
    /// `adaln_proj.linear`, `[6*hidden*MODALITY_NUM, time_embed_dim]` +
    /// `[6*hidden*MODALITY_NUM]` bias - this block's OWN private weight
    /// (never folded across blocks; only `temb_silu`, its shared INPUT, is).
    pub adaln_proj: Lin<T>,
}

/// Grads mirroring [`BlockW`].
pub struct BlockGrads<T> {
    pub attn: AttnGrads<T>,
    pub norm1: Vec<T>,
    pub norm2: Vec<T>,
    pub fc1: LinNB<T>,
    pub fc2: LinNB<T>,
    pub adaln_proj: Lin<T>,
}

struct BlockCache<T> {
    x0: Vec<T>,
    inv1: Vec<T>,
    xhat1: Vec<T>,
    scale_msa: Vec<T>,
    mod1: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    inv_q: Vec<T>,
    inv_k: Vec<T>,
    qr: Vec<T>,
    kr: Vec<T>,
    probs: Vec<T>,
    actx: Vec<T>,
    attn_out: Vec<T>,
    gate_msa: Vec<T>,
    h1: Vec<T>,
    inv2: Vec<T>,
    xhat2: Vec<T>,
    scale_mlp: Vec<T>,
    mod2: Vec<T>,
    value: Vec<T>,
    gate: Vec<T>,
    act: Vec<T>,
    ff_out: Vec<T>,
    gate_mlp: Vec<T>,
}

/// `MiniMaxH3TransformerBlock.forward` - `crate::block::block_forward`'s
/// independently re-derived, generic-`T` twin (see this module's doc for
/// why the reuse is limited to non-differentiable helpers). `x`:
/// `[seq_len,hidden]`. `adaln_idx`: `[seq_len]`, THIS port's row convention
/// (`crate::model::adaln_indices`, reused directly - pure index arithmetic,
/// not differentiated). `cos`/`sin`: `[seq_len,half]`
/// (`crate::rope::build_tables`, reused directly for the same reason).
/// `temb_silu`: `[num_timesteps,time_embed_dim]`, this forward's shared
/// timestep conditioning (every block reads the identical tensor).
#[allow(clippy::too_many_arguments)]
fn block_forward<T: Fp>(cfg: &Cfg, w: &BlockW<T>, x: &[T], adaln_idx: &[usize], cos: &[T], sin: &[T], temb_silu: &[T]) -> (Vec<T>, BlockCache<T>) {
    let (hidden, heads, head_dim, inner, ffn, half, num_ts) = (cfg.hidden(), cfg.heads(), cfg.head_dim(), cfg.inner(), cfg.ffn(), cfg.half(), cfg.num_timesteps);
    let seq_len = adaln_idx.len();
    let td = seq_len * hidden;

    let tables = adaln_tables(&w.adaln_proj, temb_silu, num_ts, hidden, cfg.time_embed_dim());
    let shift_msa = gather_rows(&tables[0], adaln_idx, hidden);
    let scale_msa: Vec<T> = gather_rows(&tables[1], adaln_idx, hidden).iter().map(|&v| T::ONE + v).collect();
    let gate_msa = gather_rows(&tables[2], adaln_idx, hidden);
    let shift_mlp = gather_rows(&tables[3], adaln_idx, hidden);
    let scale_mlp: Vec<T> = gather_rows(&tables[4], adaln_idx, hidden).iter().map(|&v| T::ONE + v).collect();
    let gate_mlp = gather_rows(&tables[5], adaln_idx, hidden);

    let (xhat1, inv1) = rmsnorm(x, seq_len, hidden, &w.norm1, cfg.norm_eps());
    let mod1 = mod_affine(&xhat1, &scale_msa, &shift_msa, td);
    let q = linear_nb(&mod1, seq_len, hidden, &w.attn.q.w, inner);
    let k = linear_nb(&mod1, seq_len, hidden, &w.attn.k.w, inner);
    let v = linear_nb(&mod1, seq_len, hidden, &w.attn.v.w, inner);
    let (qn, inv_q) = rmsnorm(&q, seq_len * heads, head_dim, &w.attn.qn, cfg.qk_norm_eps());
    let (kn, inv_k) = rmsnorm(&k, seq_len * heads, head_dim, &w.attn.kn, cfg.qk_norm_eps());
    let qr = rope_h3(&qn, seq_len, heads, head_dim, half, cos, sin);
    let kr = rope_h3(&kn, seq_len, heads, head_dim, half, cos, sin);
    let (probs, actx) = attn_fwd(&qr, seq_len, &kr, &v, seq_len, heads, head_dim);
    let attn_out = linear_nb(&actx, seq_len, inner, &w.attn.o.w, hidden);
    let h1 = gate_elemwise(x, &gate_msa, &attn_out, td);

    let (xhat2, inv2) = rmsnorm(&h1, seq_len, hidden, &w.norm2, cfg.norm_eps());
    let mod2 = mod_affine(&xhat2, &scale_mlp, &shift_mlp, td);
    let (fc1_value_w, fc1_gate_w) = w.fc1.w.split_at(ffn * hidden);
    let value = linear_nb(&mod2, seq_len, hidden, fc1_value_w, ffn);
    let gate = linear_nb(&mod2, seq_len, hidden, fc1_gate_w, ffn);
    let act: Vec<T> = value.iter().zip(&gate).map(|(&val, &g)| val * silu(g)).collect();
    let ff_out = linear_nb(&act, seq_len, ffn, &w.fc2.w, hidden);
    let out = gate_elemwise(&h1, &gate_mlp, &ff_out, td);

    (
        out,
        BlockCache { x0: x.to_vec(), inv1, xhat1, scale_msa, mod1, q, k, v, inv_q, inv_k, qr, kr, probs, actx, attn_out, gate_msa, h1, inv2, xhat2, scale_mlp, mod2, value, gate, act, ff_out, gate_mlp },
    )
}

/// [`block_forward`] backward. Returns `(dx, grads, d(temb_silu))` - the
/// third element is this block's own UNREDUCED contribution to the
/// model-shared `temb_silu` tensor (see this module's doc on the three-site
/// AdaLN fold); the caller sums it across the whole block stack.
#[allow(clippy::too_many_arguments)]
fn block_backward<T: Fp>(cfg: &Cfg, w: &BlockW<T>, c: &BlockCache<T>, cos: &[T], sin: &[T], adaln_idx: &[usize], temb_silu: &[T], dout: &[T]) -> (Vec<T>, BlockGrads<T>, Vec<T>) {
    let (hidden, heads, head_dim, inner, ffn, half, num_ts) = (cfg.hidden(), cfg.heads(), cfg.head_dim(), cfg.inner(), cfg.ffn(), cfg.half(), cfg.num_timesteps);
    let seq_len = adaln_idx.len();
    let td = seq_len * hidden;

    let (dff_out, dgate_mlp) = gate_elemwise_bwd(&c.gate_mlp, &c.ff_out, dout);
    let mut dh1 = dout.to_vec();

    let (fc1_value_w, fc1_gate_w) = w.fc1.w.split_at(ffn * hidden);
    let (dact, g_fc2) = linear_nb_bwd(&c.act, seq_len, ffn, &w.fc2.w, hidden, &dff_out);
    let dgate: Vec<T> = dact.iter().zip(&c.value).zip(&c.gate).map(|((&da, &val), &g)| da * val * dsilu(g)).collect();
    let dvalue: Vec<T> = dact.iter().zip(&c.gate).map(|(&da, &g)| da * silu(g)).collect();
    let (dmod2_g, g_fc1_gate) = linear_nb_bwd(&c.mod2, seq_len, hidden, fc1_gate_w, ffn, &dgate);
    let (dmod2_v, g_fc1_value) = linear_nb_bwd(&c.mod2, seq_len, hidden, fc1_value_w, ffn, &dvalue);
    let mut dmod2 = vec![T::ZERO; td];
    for i in 0..td {
        dmod2[i] = dmod2_g[i] + dmod2_v[i];
    }

    let (dxhat2, dscale_mlp, dshift_mlp) = mod_affine_bwd(&c.xhat2, &c.scale_mlp, &dmod2);
    let mut g_norm2 = vec![T::ZERO; hidden];
    let dxhat2_full = rmsnorm_bwd(&c.h1, seq_len, hidden, &w.norm2, &c.inv2, &dxhat2, &mut g_norm2);
    for i in 0..td {
        dh1[i] += dxhat2_full[i];
    }
    let mut g_fc1_w = vec![T::ZERO; 2 * ffn * hidden];
    g_fc1_w[..ffn * hidden].copy_from_slice(&g_fc1_value.w);
    g_fc1_w[ffn * hidden..].copy_from_slice(&g_fc1_gate.w);

    let (dattn_out, dgate_msa) = gate_elemwise_bwd(&c.gate_msa, &c.attn_out, &dh1);
    let mut dx = dh1;

    let (dactx, g_o) = linear_nb_bwd(&c.actx, seq_len, inner, &w.attn.o.w, hidden, &dattn_out);
    let (dqr, dkr, dv) = attn_bwd(&c.probs, &c.qr, &c.kr, &c.v, seq_len, seq_len, heads, head_dim, &dactx);
    let dqn = rope_h3_bwd(&dqr, seq_len, heads, head_dim, half, cos, sin);
    let dkn = rope_h3_bwd(&dkr, seq_len, heads, head_dim, half, cos, sin);
    let mut g_qn = vec![T::ZERO; head_dim];
    let dq = rmsnorm_bwd(&c.q, seq_len * heads, head_dim, &w.attn.qn, &c.inv_q, &dqn, &mut g_qn);
    let mut g_kn = vec![T::ZERO; head_dim];
    let dk = rmsnorm_bwd(&c.k, seq_len * heads, head_dim, &w.attn.kn, &c.inv_k, &dkn, &mut g_kn);

    let (dmod1_q, g_q) = linear_nb_bwd(&c.mod1, seq_len, hidden, &w.attn.q.w, inner, &dq);
    let (dmod1_k, g_k) = linear_nb_bwd(&c.mod1, seq_len, hidden, &w.attn.k.w, inner, &dk);
    let (dmod1_v, g_v) = linear_nb_bwd(&c.mod1, seq_len, hidden, &w.attn.v.w, inner, &dv);
    let mut dmod1 = vec![T::ZERO; td];
    for i in 0..td {
        dmod1[i] = dmod1_q[i] + dmod1_k[i] + dmod1_v[i];
    }

    let (dxhat1, dscale_msa, dshift_msa) = mod_affine_bwd(&c.xhat1, &c.scale_msa, &dmod1);
    let mut g_norm1 = vec![T::ZERO; hidden];
    let dxhat1_full = rmsnorm_bwd(&c.x0, seq_len, hidden, &w.norm1, &c.inv1, &dxhat1, &mut g_norm1);
    for i in 0..td {
        dx[i] += dxhat1_full[i];
    }

    let mtn = MODALITY_NUM as usize * num_ts;
    let d_tables: [Vec<T>; 6] = [
        gather_rows_bwd(&dshift_msa, adaln_idx, hidden, mtn),
        gather_rows_bwd(&dscale_msa, adaln_idx, hidden, mtn),
        gather_rows_bwd(&dgate_msa, adaln_idx, hidden, mtn),
        gather_rows_bwd(&dshift_mlp, adaln_idx, hidden, mtn),
        gather_rows_bwd(&dscale_mlp, adaln_idx, hidden, mtn),
        gather_rows_bwd(&dgate_mlp, adaln_idx, hidden, mtn),
    ];
    let (g_adaln_proj, d_temb_silu) = adaln_tables_bwd(&w.adaln_proj, temb_silu, num_ts, hidden, cfg.time_embed_dim(), &d_tables);

    let grads =
        BlockGrads { attn: AttnGrads { q: g_q, k: g_k, v: g_v, o: g_o, qn: g_qn, kn: g_kn }, norm1: g_norm1, norm2: g_norm2, fc1: LinNB { w: g_fc1_w }, fc2: g_fc2, adaln_proj: g_adaln_proj };
    (dx, grads, d_temb_silu)
}

// ---- the shared sinusoidal timestep MLP (time_embedder) ----

/// `sinusoid(t,freq_dim) -> linear_1 -> SiLU -> linear_2` - `flip_sin_to_cos
/// = true`, `downscale_freq_shift = 0`, `max_period = 10000.0`
/// (`crate::model`'s own `TIME_MAX_PERIOD`; nothing in the reference
/// overrides it - see the roadmap's own citation). One row per DISTINCT
/// timestep, matching the reference's own `Timesteps`/`TimestepEmbedding`
/// pair.
fn timestep_embedding<T: Fp>(t: f64, dim: usize) -> Vec<T> {
    assert!(dim.is_multiple_of(2), "timestep_embedding: dim {dim} must be even");
    let half = dim / 2;
    let mut e = vec![T::ZERO; dim];
    for k in 0..half {
        let freq = (-(TIME_MAX_PERIOD.ln()) * k as f64 / half as f64).exp();
        let arg = t * freq;
        e[k] = T::fr(arg.cos());
        e[half + k] = T::fr(arg.sin());
    }
    e
}

struct TimeCache<T> {
    sinusoid: Vec<T>,
    h0pre: Vec<T>,
    h0: Vec<T>,
    temb: Vec<T>,
}

/// Returns `(temb_silu, cache)` - `temb_silu` is what every block's
/// `adaln_proj` and `norm_out` consume; `cache.temb` (pre-`silu`) is kept
/// only for [`build_temb_bwd`]'s own `dsilu`.
fn build_temb<T: Fp>(w_l1: &Lin<T>, w_l2: &Lin<T>, timestep: &[f64], freq_dim: usize, teh: usize, te: usize) -> (Vec<T>, TimeCache<T>) {
    let num_ts = timestep.len();
    let mut sinusoid = vec![T::ZERO; num_ts * freq_dim];
    for (t_idx, &t) in timestep.iter().enumerate() {
        sinusoid[t_idx * freq_dim..(t_idx + 1) * freq_dim].copy_from_slice(&timestep_embedding::<T>(t, freq_dim));
    }
    let h0pre = linear(&sinusoid, num_ts, freq_dim, &w_l1.w, &w_l1.b, teh);
    let h0: Vec<T> = h0pre.iter().map(|&v| silu(v)).collect();
    let temb = linear(&h0, num_ts, teh, &w_l2.w, &w_l2.b, te);
    let temb_silu: Vec<T> = temb.iter().map(|&v| silu(v)).collect();
    (temb_silu, TimeCache { sinusoid, h0pre, h0, temb })
}

/// [`build_temb`] backward, from `d_temb_silu` ALREADY summed across every
/// consumer (every block's `adaln_proj` plus `norm_out` - see this module's
/// doc on the three-site fold).
fn build_temb_bwd<T: Fp>(w_l1: &Lin<T>, w_l2: &Lin<T>, c: &TimeCache<T>, num_ts: usize, freq_dim: usize, teh: usize, te: usize, d_temb_silu: &[T]) -> (Lin<T>, Lin<T>) {
    let d_temb: Vec<T> = d_temb_silu.iter().zip(&c.temb).map(|(&g, &tv)| g * dsilu(tv)).collect();
    let (dh0, g_l2) = linear_bwd(&c.h0, num_ts, teh, &w_l2.w, te, &d_temb);
    let dh0pre: Vec<T> = dh0.iter().zip(&c.h0pre).map(|(&g, &v)| g * dsilu(v)).collect();
    let (_dsinusoid, g_l1) = linear_bwd(&c.sinusoid, num_ts, freq_dim, &w_l1.w, teh, &dh0pre);
    (g_l1, g_l2)
}

// ---- shape ----

/// Shape of the H3 training problem: the [`H3TransformerConfig`] fields the
/// host path needs, plus the fixed token/timestep counts of this module's
/// own synthetic packing (see this module's doc - real `t2va`/`fl2va`
/// layout is a later phase's job).
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub tcfg: H3TransformerConfig,
    pub num_video: usize,
    pub num_audio: usize,
    pub num_text: usize,
    pub num_timesteps: usize,
}

impl Cfg {
    /// Every count differs from every other (lesson #4): 5 video tokens, 3
    /// audio tokens, 4 text rows, 2 distinct timesteps - none coincide with
    /// each other or with [`H3TransformerConfig::tiny`]'s own dims.
    pub fn tiny() -> Cfg {
        Cfg { tcfg: H3TransformerConfig::tiny(), num_video: 5, num_audio: 3, num_text: 4, num_timesteps: 2 }
    }
    pub fn seq_len(&self) -> usize {
        self.num_video + self.num_audio + self.num_text
    }
    pub fn hidden(&self) -> usize {
        self.tcfg.hidden_size as usize
    }
    pub fn heads(&self) -> usize {
        self.tcfg.num_attention_heads as usize
    }
    pub fn head_dim(&self) -> usize {
        self.tcfg.attention_head_dim as usize
    }
    pub fn inner(&self) -> usize {
        self.tcfg.inner_dim() as usize
    }
    pub fn ffn(&self) -> usize {
        self.tcfg.ffn_dim as usize
    }
    pub fn half(&self) -> usize {
        3 * self.tcfg.rope_freq_dim as usize
    }
    pub fn video_patch_dim(&self) -> usize {
        self.tcfg.video_patch_dim() as usize
    }
    pub fn audio_in_channels(&self) -> usize {
        self.tcfg.audio_in_channels as usize
    }
    pub fn text_dim(&self) -> usize {
        self.tcfg.text_dim as usize
    }
    pub fn freq_dim(&self) -> usize {
        self.tcfg.freq_dim as usize
    }
    pub fn time_embed_hidden_dim(&self) -> usize {
        self.tcfg.time_embed_hidden_dim as usize
    }
    pub fn time_embed_dim(&self) -> usize {
        self.tcfg.time_embed_dim as usize
    }
    pub fn norm_eps(&self) -> f64 {
        self.tcfg.norm_eps as f64
    }
    pub fn qk_norm_eps(&self) -> f64 {
        self.tcfg.qk_norm_eps as f64
    }
    pub fn final_norm_eps(&self) -> f64 {
        self.tcfg.final_norm_eps as f64
    }
}

// ---- weights ----

/// Every trainable tensor of the H3 DiT core, in the host training layout,
/// named as `crate::model::H3Transformer::load` names them.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelW<T> {
    pub proj_in: Lin<T>,
    pub audio_proj_in: Lin<T>,
    pub context_embedder: Lin<T>,
    pub time_embedder_l1: Lin<T>,
    pub time_embedder_l2: Lin<T>,
    pub refiner_blocks: Vec<RefinerBlockW<T>>,
    pub refiner_final_norm: Vec<T>,
    pub blocks: Vec<BlockW<T>>,
    pub norm_out_norm: Vec<T>,
    /// `norm_out.linear`, `[2*hidden,time_embed_dim]` - rows `[0,hidden)`
    /// are the shift projection, `[hidden,2*hidden)` the scale projection
    /// (`crate::model`'s own split convention).
    pub norm_out_linear: Lin<T>,
    pub proj_out: Lin<T>,
    pub audio_proj_out: Lin<T>,
}

/// Grads mirroring [`ModelW`].
pub struct ModelGrads<T> {
    pub proj_in: Lin<T>,
    pub audio_proj_in: Lin<T>,
    pub context_embedder: Lin<T>,
    pub time_embedder_l1: Lin<T>,
    pub time_embedder_l2: Lin<T>,
    pub refiner_blocks: Vec<RefinerBlockGrads<T>>,
    pub refiner_final_norm: Vec<T>,
    pub blocks: Vec<BlockGrads<T>>,
    pub norm_out_norm: Vec<T>,
    pub norm_out_linear: Lin<T>,
    pub proj_out: Lin<T>,
    pub audio_proj_out: Lin<T>,
}

/// Saved forward state for the backward pass.
pub struct ModelCache<T> {
    video_patches: Vec<T>,
    audio_latents: Vec<T>,
    text_ctx: Vec<T>,
    refiner_caches: Vec<RefinerBlockCache<T>>,
    text_after_refiner: Vec<T>,
    text_inv_final: Vec<T>,
    time_cache: TimeCache<T>,
    temb_silu: Vec<T>,
    cos: Vec<T>,
    sin: Vec<T>,
    adaln_idx: Vec<usize>,
    timestep_indices: Vec<usize>,
    video_indices: Vec<usize>,
    audio_indices: Vec<usize>,
    text_indices: Vec<usize>,
    block_caches: Vec<BlockCache<T>>,
    h_final: Vec<T>,
    xhat_out: Vec<T>,
    inv_out: Vec<T>,
    scale_g: Vec<T>,
    modulated: Vec<T>,
}

/// One packed sequence, ready for [`forward`] - this module's own fixed
/// synthetic packing (see this module's doc), NOT `crate::model::
/// PackedInputs`'s real `t2va`/`fl2va` layout.
#[derive(Clone)]
pub struct Batch<T> {
    pub video_patches: Vec<T>,
    pub audio_latents: Vec<T>,
    pub text_ctx: Vec<T>,
    /// `[num_timesteps]`, DISTINCT raw sigma values in `(0,1)`.
    pub timestep: Vec<f64>,
    /// `[seq_len]`, index into `timestep` per row.
    pub timestep_indices: Vec<u32>,
    /// `[seq_len]`, `config::TAG_{VIDEO,TEXT,AUDIO}` per row.
    pub token_tags: Vec<u32>,
    /// `[seq_len,3]` row-major `(t,h,w)`.
    pub position_ids: Vec<f32>,
    pub video_indices: Vec<usize>,
    pub audio_indices: Vec<usize>,
    pub text_indices: Vec<usize>,
    pub video_target: Vec<T>,
    pub audio_target: Vec<T>,
}

/// Full forward: input projections + token refiner, pack into one sequence,
/// the `num_layers` block stack, `norm_out`, the two output heads.
pub fn forward<T: Fp>(cfg: &Cfg, w: &ModelW<T>, b: &Batch<T>) -> (Vec<T>, Vec<T>, ModelCache<T>) {
    let seq_len = cfg.seq_len();
    let hidden = cfg.hidden();

    let video_embeds = linear(&b.video_patches, cfg.num_video, cfg.video_patch_dim(), &w.proj_in.w, &w.proj_in.b, hidden);
    let audio_embeds = linear(&b.audio_latents, cfg.num_audio, cfg.audio_in_channels(), &w.audio_proj_in.w, &w.audio_proj_in.b, hidden);
    let text_embeds0 = linear(&b.text_ctx, cfg.num_text, cfg.text_dim(), &w.context_embedder.w, &w.context_embedder.b, hidden);

    let mut text_x = text_embeds0;
    let mut refiner_caches = Vec::with_capacity(w.refiner_blocks.len());
    for rb in &w.refiner_blocks {
        let (out, rc) = refiner_block_forward(cfg, rb, &text_x);
        text_x = out;
        refiner_caches.push(rc);
    }
    let text_after_refiner = text_x;
    let (text_final, text_inv_final) = rmsnorm(&text_after_refiner, cfg.num_text, hidden, &w.refiner_final_norm, cfg.final_norm_eps());

    let h0 = {
        let mut out = scatter_rows(&video_embeds, &b.video_indices, hidden, seq_len);
        let a = scatter_rows(&audio_embeds, &b.audio_indices, hidden, seq_len);
        let t = scatter_rows(&text_final, &b.text_indices, hidden, seq_len);
        for i in 0..out.len() {
            out[i] = out[i] + a[i] + t[i];
        }
        out
    };

    let rt = build_tables(&cfg.tcfg, &b.position_ids);
    let cos: Vec<T> = rt.cos.iter().map(|&v| T::fr(v as f64)).collect();
    let sin: Vec<T> = rt.sin.iter().map(|&v| T::fr(v as f64)).collect();

    let (temb_silu, time_cache) = build_temb(&w.time_embedder_l1, &w.time_embedder_l2, &b.timestep, cfg.freq_dim(), cfg.time_embed_hidden_dim(), cfg.time_embed_dim());

    let adaln_idx_u32 = adaln_indices(&b.token_tags, &b.timestep_indices, cfg.num_timesteps);
    let adaln_idx: Vec<usize> = adaln_idx_u32.iter().map(|&i| i as usize).collect();
    let timestep_indices: Vec<usize> = b.timestep_indices.iter().map(|&i| i as usize).collect();

    let mut h = h0;
    let mut block_caches = Vec::with_capacity(w.blocks.len());
    for bw in &w.blocks {
        let (out, bc) = block_forward(cfg, bw, &h, &adaln_idx, &cos, &sin, &temb_silu);
        h = out;
        block_caches.push(bc);
    }
    let h_final = h;

    let te = cfg.time_embed_dim();
    let (shift_w, scale_w) = w.norm_out_linear.w.split_at(hidden * te);
    let (shift_b, scale_b) = w.norm_out_linear.b.split_at(hidden);
    let shift_tbl = linear(&temb_silu, cfg.num_timesteps, te, shift_w, shift_b, hidden);
    let scale_tbl = linear(&temb_silu, cfg.num_timesteps, te, scale_w, scale_b, hidden);
    let shift_g = gather_rows(&shift_tbl, &timestep_indices, hidden);
    let scale_g: Vec<T> = gather_rows(&scale_tbl, &timestep_indices, hidden).iter().map(|&v| T::ONE + v).collect();
    let (xhat_out, inv_out) = rmsnorm(&h_final, seq_len, hidden, &w.norm_out_norm, cfg.final_norm_eps());
    let modulated = mod_affine(&xhat_out, &scale_g, &shift_g, seq_len * hidden);

    let video_full = linear(&modulated, seq_len, hidden, &w.proj_out.w, &w.proj_out.b, cfg.video_patch_dim());
    let audio_full = linear(&modulated, seq_len, hidden, &w.audio_proj_out.w, &w.audio_proj_out.b, cfg.audio_in_channels());
    let video_pred = gather_rows(&video_full, &b.video_indices, cfg.video_patch_dim());
    let audio_pred = gather_rows(&audio_full, &b.audio_indices, cfg.audio_in_channels());

    let cache = ModelCache {
        video_patches: b.video_patches.clone(),
        audio_latents: b.audio_latents.clone(),
        text_ctx: b.text_ctx.clone(),
        refiner_caches,
        text_after_refiner,
        text_inv_final,
        time_cache,
        temb_silu,
        cos,
        sin,
        adaln_idx,
        timestep_indices,
        video_indices: b.video_indices.clone(),
        audio_indices: b.audio_indices.clone(),
        text_indices: b.text_indices.clone(),
        block_caches,
        h_final,
        xhat_out,
        inv_out,
        scale_g,
        modulated,
    };
    (video_pred, audio_pred, cache)
}

/// Full backward from `(dvideo_pred, daudio_pred)`.
pub fn backward<T: Fp>(cfg: &Cfg, w: &ModelW<T>, cache: &ModelCache<T>, dvideo_pred: &[T], daudio_pred: &[T]) -> ModelGrads<T> {
    let seq_len = cfg.seq_len();
    let hidden = cfg.hidden();
    let te = cfg.time_embed_dim();

    let dvideo_full = gather_rows_bwd(dvideo_pred, &cache.video_indices, cfg.video_patch_dim(), seq_len);
    let daudio_full = gather_rows_bwd(daudio_pred, &cache.audio_indices, cfg.audio_in_channels(), seq_len);
    let (dmodulated_v, g_proj_out) = linear_bwd(&cache.modulated, seq_len, hidden, &w.proj_out.w, cfg.video_patch_dim(), &dvideo_full);
    let (dmodulated_a, g_audio_proj_out) = linear_bwd(&cache.modulated, seq_len, hidden, &w.audio_proj_out.w, cfg.audio_in_channels(), &daudio_full);
    let mut dmodulated = vec![T::ZERO; seq_len * hidden];
    for i in 0..dmodulated.len() {
        dmodulated[i] = dmodulated_v[i] + dmodulated_a[i];
    }

    let (dxhat_out, dscale_g, dshift_g) = mod_affine_bwd(&cache.xhat_out, &cache.scale_g, &dmodulated);
    let mut g_norm_out_norm = vec![T::ZERO; hidden];
    let dh_final = rmsnorm_bwd(&cache.h_final, seq_len, hidden, &w.norm_out_norm, &cache.inv_out, &dxhat_out, &mut g_norm_out_norm);

    let d_shift_tbl = gather_rows_bwd(&dshift_g, &cache.timestep_indices, hidden, cfg.num_timesteps);
    let d_scale_tbl = gather_rows_bwd(&dscale_g, &cache.timestep_indices, hidden, cfg.num_timesteps);
    let (shift_w, scale_w) = w.norm_out_linear.w.split_at(hidden * te);
    let (d_temb_silu_shift, g_shift) = linear_bwd(&cache.temb_silu, cfg.num_timesteps, te, shift_w, hidden, &d_shift_tbl);
    let (d_temb_silu_scale, g_scale) = linear_bwd(&cache.temb_silu, cfg.num_timesteps, te, scale_w, hidden, &d_scale_tbl);
    let mut d_temb_silu = vec![T::ZERO; cfg.num_timesteps * te];
    for i in 0..d_temb_silu.len() {
        d_temb_silu[i] = d_temb_silu_shift[i] + d_temb_silu_scale[i];
    }
    let mut g_norm_out_linear_w = vec![T::ZERO; 2 * hidden * te];
    g_norm_out_linear_w[..hidden * te].copy_from_slice(&g_shift.w);
    g_norm_out_linear_w[hidden * te..].copy_from_slice(&g_scale.w);
    let mut g_norm_out_linear_b = vec![T::ZERO; 2 * hidden];
    g_norm_out_linear_b[..hidden].copy_from_slice(&g_shift.b);
    g_norm_out_linear_b[hidden..].copy_from_slice(&g_scale.b);

    let mut dh = dh_final;
    let mut block_grads = Vec::with_capacity(w.blocks.len());
    for (bw, bc) in w.blocks.iter().zip(&cache.block_caches).rev() {
        let (dx, g, d_ts) = block_backward(cfg, bw, bc, &cache.cos, &cache.sin, &cache.adaln_idx, &cache.temb_silu, &dh);
        dh = dx;
        for i in 0..d_temb_silu.len() {
            d_temb_silu[i] += d_ts[i];
        }
        block_grads.push(g);
    }
    block_grads.reverse();

    let dvideo_embeds = gather_rows(&dh, &cache.video_indices, hidden);
    let daudio_embeds = gather_rows(&dh, &cache.audio_indices, hidden);
    let dtext_final = gather_rows(&dh, &cache.text_indices, hidden);

    let mut g_refiner_final_norm = vec![T::ZERO; hidden];
    let dtext_after_refiner = rmsnorm_bwd(&cache.text_after_refiner, cfg.num_text, hidden, &w.refiner_final_norm, &cache.text_inv_final, &dtext_final, &mut g_refiner_final_norm);

    let mut dtext = dtext_after_refiner;
    let mut refiner_grads = Vec::with_capacity(w.refiner_blocks.len());
    for (rb, rc) in w.refiner_blocks.iter().zip(&cache.refiner_caches).rev() {
        let (dx, g) = refiner_block_backward(cfg, rb, rc, &dtext);
        dtext = dx;
        refiner_grads.push(g);
    }
    refiner_grads.reverse();

    let (_d_video_patches, g_proj_in) = linear_bwd(&cache.video_patches, cfg.num_video, cfg.video_patch_dim(), &w.proj_in.w, hidden, &dvideo_embeds);
    let (_d_audio_latents, g_audio_proj_in) = linear_bwd(&cache.audio_latents, cfg.num_audio, cfg.audio_in_channels(), &w.audio_proj_in.w, hidden, &daudio_embeds);
    let (_d_text_ctx, g_context_embedder) = linear_bwd(&cache.text_ctx, cfg.num_text, cfg.text_dim(), &w.context_embedder.w, hidden, &dtext);

    let (g_time_l1, g_time_l2) = build_temb_bwd(&w.time_embedder_l1, &w.time_embedder_l2, &cache.time_cache, cfg.num_timesteps, cfg.freq_dim(), cfg.time_embed_hidden_dim(), te, &d_temb_silu);

    ModelGrads {
        proj_in: g_proj_in,
        audio_proj_in: g_audio_proj_in,
        context_embedder: g_context_embedder,
        time_embedder_l1: g_time_l1,
        time_embedder_l2: g_time_l2,
        refiner_blocks: refiner_grads,
        refiner_final_norm: g_refiner_final_norm,
        blocks: block_grads,
        norm_out_norm: g_norm_out_norm,
        norm_out_linear: Lin { w: g_norm_out_linear_w, b: g_norm_out_linear_b },
        proj_out: g_proj_out,
        audio_proj_out: g_audio_proj_out,
    }
}

/// Flow-matching velocity-MSE loss, combined across BOTH the video and audio
/// predictions (mean over the concatenated element count -
/// `ltxv::av_modelgrad::loss`'s own convention).
pub fn loss<T: Fp>(video_pred: &[T], video_target: &[T], audio_pred: &[T], audio_target: &[T]) -> (f64, Vec<T>, Vec<T>) {
    assert_eq!(video_pred.len(), video_target.len(), "minimaxh3 loss: video prediction/target size");
    assert_eq!(audio_pred.len(), audio_target.len(), "minimaxh3 loss: audio prediction/target size");
    let n = T::fr((video_pred.len() + audio_pred.len()) as f64);
    let two = T::fr(2.0);
    let mut l = 0.0;
    let mut dv = vec![T::ZERO; video_pred.len()];
    for i in 0..video_pred.len() {
        let err = video_pred[i] - video_target[i];
        l += (err * err / n).f64();
        dv[i] = two * err / n;
    }
    let mut da = vec![T::ZERO; audio_pred.len()];
    for i in 0..audio_pred.len() {
        let err = audio_pred[i] - audio_target[i];
        l += (err * err / n).f64();
        da[i] = two * err / n;
    }
    (l, dv, da)
}

/// One training evaluation: forward + loss + backward.
pub fn grads<T: Fp>(cfg: &Cfg, w: &ModelW<T>, b: &Batch<T>) -> (f64, ModelGrads<T>) {
    let (video_pred, audio_pred, cache) = forward(cfg, w, b);
    let (l, dv, da) = loss(&video_pred, &b.video_target, &audio_pred, &b.audio_target);
    (l, backward(cfg, w, &cache, &dv, &da))
}

/// Build one training example under this module's own FIXED synthetic
/// packing `[text | audio | video]` (see this module's doc). Video and
/// audio rows alternate between the batch's `num_timesteps` distinct
/// sigmas by row parity (genuine per-row diffusion forcing, not a
/// per-stream scalar - see this module's doc); text rows always address
/// timestep index 0 (text carries no noise level of its own, but every row
/// still needs an AdaLN address). Same `x_σ = (1-σ)·x0 + σ·ε`, `v = ε - x0`
/// convention every other flow-matching batch builder in this workspace
/// uses, applied per ROW rather than per stream.
pub fn make_flow_batch<T: Fp>(cfg: &Cfg, video_x0: &[T], audio_x0: &[T], text_ctx: &[T], video_noise: &[T], audio_noise: &[T]) -> Batch<T> {
    assert_eq!(video_x0.len(), cfg.num_video * cfg.video_patch_dim(), "minimaxh3 flow batch: video_x0 size");
    assert_eq!(audio_x0.len(), cfg.num_audio * cfg.audio_in_channels(), "minimaxh3 flow batch: audio_x0 size");
    assert_eq!(video_noise.len(), video_x0.len(), "minimaxh3 flow batch: video_noise size");
    assert_eq!(audio_noise.len(), audio_x0.len(), "minimaxh3 flow batch: audio_noise size");

    let seq_len = cfg.seq_len();
    let text_indices: Vec<usize> = (0..cfg.num_text).collect();
    let audio_indices: Vec<usize> = (cfg.num_text..cfg.num_text + cfg.num_audio).collect();
    let video_indices: Vec<usize> = (cfg.num_text + cfg.num_audio..seq_len).collect();

    let mut token_tags = vec![0u32; seq_len];
    let mut timestep_indices = vec![0u32; seq_len];
    for &i in &text_indices {
        token_tags[i] = TAG_TEXT;
    }
    for (n, &i) in audio_indices.iter().enumerate() {
        token_tags[i] = TAG_AUDIO;
        timestep_indices[i] = (n % cfg.num_timesteps) as u32;
    }
    for (n, &i) in video_indices.iter().enumerate() {
        token_tags[i] = TAG_VIDEO;
        timestep_indices[i] = (n % cfg.num_timesteps) as u32;
    }

    let timestep: Vec<f64> = (0..cfg.num_timesteps).map(|k| (k as f64 + 1.0) / (cfg.num_timesteps as f64 + 1.0)).collect();

    let mut position_ids = vec![0f32; seq_len * 3];
    for r in 0..seq_len {
        position_ids[r * 3] = r as f32;
        position_ids[r * 3 + 1] = (r % 3) as f32;
        position_ids[r * 3 + 2] = (r % 2) as f32;
    }

    let vpd = cfg.video_patch_dim();
    let mut video_latent = vec![T::ZERO; video_x0.len()];
    let mut video_target = vec![T::ZERO; video_x0.len()];
    for (n, _) in video_indices.iter().enumerate() {
        let s = T::fr(timestep[n % cfg.num_timesteps]);
        for c in 0..vpd {
            let idx = n * vpd + c;
            video_latent[idx] = (T::ONE - s) * video_x0[idx] + s * video_noise[idx];
            video_target[idx] = video_noise[idx] - video_x0[idx];
        }
    }
    let apd = cfg.audio_in_channels();
    let mut audio_latent = vec![T::ZERO; audio_x0.len()];
    let mut audio_target = vec![T::ZERO; audio_x0.len()];
    for (n, _) in audio_indices.iter().enumerate() {
        let s = T::fr(timestep[n % cfg.num_timesteps]);
        for c in 0..apd {
            let idx = n * apd + c;
            audio_latent[idx] = (T::ONE - s) * audio_x0[idx] + s * audio_noise[idx];
            audio_target[idx] = audio_noise[idx] - audio_x0[idx];
        }
    }

    Batch { video_patches: video_latent, audio_latents: audio_latent, text_ctx: text_ctx.to_vec(), timestep, timestep_indices, token_tags, position_ids, video_indices, audio_indices, text_indices, video_target, audio_target }
}

// ---- weight init (gradchecks + synthetic training; real weights come from a checkpoint) ----

/// Deterministic random init at any scalar type.
pub fn init_model<T: Fp>(cfg: &Cfg, seed: u64) -> ModelW<T> {
    let mut rng = data::rng::Rng::new(seed);
    let mut v = |n: usize, s: f64| -> Vec<T> { (0..n).map(|_| T::fr((rng.next_f64() - 0.5) * 2.0 * s)).collect() };
    let gain = |n: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Vec<T> { r(n, 0.1).iter().map(|&x| T::ONE + x).collect() };
    let lin = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> Lin<T> { Lin { w: r(out * inn, s), b: r(out, 0.05) } };
    let lin_nb = |out: usize, inn: usize, s: f64, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> LinNB<T> { LinNB { w: r(out * inn, s) } };
    let attn_w = |inner: usize, hidden: usize, hd: usize, r: &mut dyn FnMut(usize, f64) -> Vec<T>| -> AttnW<T> {
        AttnW { q: lin_nb(inner, hidden, 0.2, r), k: lin_nb(inner, hidden, 0.2, r), v: lin_nb(inner, hidden, 0.2, r), o: lin_nb(hidden, inner, 0.2, r), qn: gain(hd, r), kn: gain(hd, r) }
    };

    let (hidden, inner, hd, ffn, te, freq, teh) = (cfg.hidden(), cfg.inner(), cfg.head_dim(), cfg.ffn(), cfg.time_embed_dim(), cfg.freq_dim(), cfg.time_embed_hidden_dim());

    let refiner_blocks = (0..cfg.tcfg.num_refiner_layers)
        .map(|_| RefinerBlockW { attn: attn_w(inner, hidden, hd, &mut v), norm1: gain(hidden, &mut v), norm2: gain(hidden, &mut v), fc1: lin_nb(2 * ffn, hidden, 0.2, &mut v), fc2: lin_nb(hidden, ffn, 0.2, &mut v) })
        .collect();

    let blocks = (0..cfg.tcfg.num_layers)
        .map(|_| BlockW {
            attn: attn_w(inner, hidden, hd, &mut v),
            norm1: gain(hidden, &mut v),
            norm2: gain(hidden, &mut v),
            fc1: lin_nb(2 * ffn, hidden, 0.2, &mut v),
            fc2: lin_nb(hidden, ffn, 0.2, &mut v),
            adaln_proj: lin(6 * hidden * MODALITY_NUM as usize, te, 0.1, &mut v),
        })
        .collect();

    ModelW {
        proj_in: lin(hidden, cfg.video_patch_dim(), 0.2, &mut v),
        audio_proj_in: lin(hidden, cfg.audio_in_channels(), 0.2, &mut v),
        context_embedder: lin(hidden, cfg.text_dim(), 0.2, &mut v),
        time_embedder_l1: lin(teh, freq, 0.1, &mut v),
        time_embedder_l2: lin(te, teh, 0.1, &mut v),
        refiner_blocks,
        refiner_final_norm: gain(hidden, &mut v),
        blocks,
        norm_out_norm: gain(hidden, &mut v),
        norm_out_linear: lin(2 * hidden, te, 0.1, &mut v),
        proj_out: lin(cfg.video_patch_dim(), hidden, 0.2, &mut v),
        audio_proj_out: lin(cfg.audio_in_channels(), hidden, 0.2, &mut v),
    }
}

// ---- parameter enumeration (FD tests) ----

fn push_attn_params<'a, T>(v: &mut Vec<(String, &'a mut Vec<T>)>, p: &str, aw: &'a mut AttnW<T>) {
    v.push((format!("{p}.norm_q.weight"), &mut aw.qn));
    v.push((format!("{p}.norm_k.weight"), &mut aw.kn));
    v.push((format!("{p}.to_q.weight"), &mut aw.q.w));
    v.push((format!("{p}.to_k.weight"), &mut aw.k.w));
    v.push((format!("{p}.to_v.weight"), &mut aw.v.w));
    v.push((format!("{p}.to_out.0.weight"), &mut aw.o.w));
}

fn push_attn_grads<'a, T>(v: &mut Vec<(String, &'a Vec<T>)>, p: &str, g: &'a AttnGrads<T>) {
    v.push((format!("{p}.norm_q.weight"), &g.qn));
    v.push((format!("{p}.norm_k.weight"), &g.kn));
    v.push((format!("{p}.to_q.weight"), &g.q.w));
    v.push((format!("{p}.to_k.weight"), &g.k.w));
    v.push((format!("{p}.to_v.weight"), &g.v.w));
    v.push((format!("{p}.to_out.0.weight"), &g.o.w));
}

/// Every trainable tensor, named as `crate::model::H3Transformer::load`
/// names them, in that loader's own relative order (mutable views).
pub fn params_mut<T>(w: &mut ModelW<T>) -> Vec<(String, &mut Vec<T>)> {
    let mut v: Vec<(String, &mut Vec<T>)> = vec![
        ("proj_in.weight".into(), &mut w.proj_in.w),
        ("proj_in.bias".into(), &mut w.proj_in.b),
        ("audio_proj_in.weight".into(), &mut w.audio_proj_in.w),
        ("audio_proj_in.bias".into(), &mut w.audio_proj_in.b),
        ("context_embedder.weight".into(), &mut w.context_embedder.w),
        ("context_embedder.bias".into(), &mut w.context_embedder.b),
        ("time_embedder.linear_1.weight".into(), &mut w.time_embedder_l1.w),
        ("time_embedder.linear_1.bias".into(), &mut w.time_embedder_l1.b),
        ("time_embedder.linear_2.weight".into(), &mut w.time_embedder_l2.w),
        ("time_embedder.linear_2.bias".into(), &mut w.time_embedder_l2.b),
        ("token_refiner.final_norm.weight".into(), &mut w.refiner_final_norm),
        ("norm_out.norm.weight".into(), &mut w.norm_out_norm),
        ("norm_out.linear.weight".into(), &mut w.norm_out_linear.w),
        ("norm_out.linear.bias".into(), &mut w.norm_out_linear.b),
        ("proj_out.weight".into(), &mut w.proj_out.w),
        ("proj_out.bias".into(), &mut w.proj_out.b),
        ("audio_proj_out.weight".into(), &mut w.audio_proj_out.w),
        ("audio_proj_out.bias".into(), &mut w.audio_proj_out.b),
    ];
    for (i, rb) in w.refiner_blocks.iter_mut().enumerate() {
        let p = format!("token_refiner.refiner_blocks.{i}");
        push_attn_params(&mut v, &format!("{p}.attn"), &mut rb.attn);
        v.push((format!("{p}.norm1.weight"), &mut rb.norm1));
        v.push((format!("{p}.norm2.weight"), &mut rb.norm2));
        v.push((format!("{p}.ff.net.0.proj.weight"), &mut rb.fc1.w));
        v.push((format!("{p}.ff.net.2.weight"), &mut rb.fc2.w));
    }
    for (i, b) in w.blocks.iter_mut().enumerate() {
        let p = format!("transformer_blocks.{i}");
        push_attn_params(&mut v, &format!("{p}.attn"), &mut b.attn);
        v.push((format!("{p}.norm1.weight"), &mut b.norm1));
        v.push((format!("{p}.norm2.weight"), &mut b.norm2));
        v.push((format!("{p}.ff.net.0.proj.weight"), &mut b.fc1.w));
        v.push((format!("{p}.ff.net.2.weight"), &mut b.fc2.w));
        v.push((format!("{p}.adaln_proj.linear.weight"), &mut b.adaln_proj.w));
        v.push((format!("{p}.adaln_proj.linear.bias"), &mut b.adaln_proj.b));
    }
    v
}

/// Gradient views in the SAME order as [`params_mut`].
pub fn grad_views<T>(g: &ModelGrads<T>) -> Vec<(String, &Vec<T>)> {
    let mut v: Vec<(String, &Vec<T>)> = vec![
        ("proj_in.weight".into(), &g.proj_in.w),
        ("proj_in.bias".into(), &g.proj_in.b),
        ("audio_proj_in.weight".into(), &g.audio_proj_in.w),
        ("audio_proj_in.bias".into(), &g.audio_proj_in.b),
        ("context_embedder.weight".into(), &g.context_embedder.w),
        ("context_embedder.bias".into(), &g.context_embedder.b),
        ("time_embedder.linear_1.weight".into(), &g.time_embedder_l1.w),
        ("time_embedder.linear_1.bias".into(), &g.time_embedder_l1.b),
        ("time_embedder.linear_2.weight".into(), &g.time_embedder_l2.w),
        ("time_embedder.linear_2.bias".into(), &g.time_embedder_l2.b),
        ("token_refiner.final_norm.weight".into(), &g.refiner_final_norm),
        ("norm_out.norm.weight".into(), &g.norm_out_norm),
        ("norm_out.linear.weight".into(), &g.norm_out_linear.w),
        ("norm_out.linear.bias".into(), &g.norm_out_linear.b),
        ("proj_out.weight".into(), &g.proj_out.w),
        ("proj_out.bias".into(), &g.proj_out.b),
        ("audio_proj_out.weight".into(), &g.audio_proj_out.w),
        ("audio_proj_out.bias".into(), &g.audio_proj_out.b),
    ];
    for (i, rb) in g.refiner_blocks.iter().enumerate() {
        let p = format!("token_refiner.refiner_blocks.{i}");
        push_attn_grads(&mut v, &format!("{p}.attn"), &rb.attn);
        v.push((format!("{p}.norm1.weight"), &rb.norm1));
        v.push((format!("{p}.norm2.weight"), &rb.norm2));
        v.push((format!("{p}.ff.net.0.proj.weight"), &rb.fc1.w));
        v.push((format!("{p}.ff.net.2.weight"), &rb.fc2.w));
    }
    for (i, b) in g.blocks.iter().enumerate() {
        let p = format!("transformer_blocks.{i}");
        push_attn_grads(&mut v, &format!("{p}.attn"), &b.attn);
        v.push((format!("{p}.norm1.weight"), &b.norm1));
        v.push((format!("{p}.norm2.weight"), &b.norm2));
        v.push((format!("{p}.ff.net.0.proj.weight"), &b.fc1.w));
        v.push((format!("{p}.ff.net.2.weight"), &b.fc2.w));
        v.push((format!("{p}.adaln_proj.linear.weight"), &b.adaln_proj.w));
        v.push((format!("{p}.adaln_proj.linear.bias"), &b.adaln_proj.b));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny config must not accidentally make two different quantities
    /// equal - lesson #4.
    #[test]
    fn the_tiny_config_has_no_coincidental_dimensions() {
        let c = Cfg::tiny();
        assert_ne!(c.num_video, c.num_audio);
        assert_ne!(c.num_video, c.num_text);
        assert_ne!(c.num_audio, c.num_text);
        assert_ne!(c.num_timesteps, c.num_video);
        assert!(c.hidden() > 0 && c.inner() != c.hidden());
    }

    /// The batch convention must be the one `crate::schedule`'s own inverse
    /// mapping expects, checked independently per row/timestep since sigma
    /// genuinely varies by row here (not just by stream).
    #[test]
    fn flow_batch_matches_the_edm_convention_per_row() {
        let cfg = Cfg::tiny();
        let video_x0: Vec<f64> = (0..cfg.num_video * cfg.video_patch_dim()).map(|i| i as f64 * 0.01).collect();
        let audio_x0: Vec<f64> = (0..cfg.num_audio * cfg.audio_in_channels()).map(|i| i as f64 * 0.02).collect();
        let video_noise: Vec<f64> = (0..video_x0.len()).map(|_| 0.5).collect();
        let audio_noise: Vec<f64> = (0..audio_x0.len()).map(|_| 0.4).collect();
        let text_ctx = vec![0.25f64; cfg.num_text * cfg.text_dim()];

        let b = make_flow_batch(&cfg, &video_x0, &audio_x0, &text_ctx, &video_noise, &audio_noise);
        assert_eq!(b.video_target, video_noise.iter().zip(&video_x0).map(|(&e, &x)| e - x).collect::<Vec<_>>());
        assert_eq!(b.audio_target, audio_noise.iter().zip(&audio_x0).map(|(&e, &x)| e - x).collect::<Vec<_>>());
        // At least two distinct sigmas are genuinely exercised (per-row
        // diffusion forcing, not one shared scalar).
        assert_eq!(b.timestep.len(), cfg.num_timesteps);
        assert!(b.timestep.windows(2).all(|w| w[0] < w[1]));
    }

    /// Every tensor [`params_mut`] enumerates must appear exactly once, and
    /// [`grad_views`] must line up name-for-name and length-for-length -
    /// `ltxv::av_modelgrad`'s own coverage test, adapted (no external
    /// tensor-manifest function exists yet in this crate to cross-check
    /// against - this test is self-consistency only).
    #[test]
    fn params_and_grads_line_up_name_for_name() {
        let cfg = Cfg::tiny();
        let mut w = init_model::<f64>(&cfg, 3);
        let names: Vec<String> = params_mut(&mut w).into_iter().map(|(n, _)| n).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate parameter name");

        let b = make_flow_batch(
            &cfg,
            &vec![0.1; cfg.num_video * cfg.video_patch_dim()],
            &vec![0.1; cfg.num_audio * cfg.audio_in_channels()],
            &vec![0.2; cfg.num_text * cfg.text_dim()],
            &vec![0.3; cfg.num_video * cfg.video_patch_dim()],
            &vec![0.3; cfg.num_audio * cfg.audio_in_channels()],
        );
        let (_l, g) = grads(&cfg, &w, &b);
        let gv = grad_views(&g);
        let pm: Vec<(String, usize)> = params_mut(&mut w).into_iter().map(|(n, val)| (n, val.len())).collect();
        assert_eq!(gv.len(), pm.len());
        for ((gn, gvv), (pn, pl)) in gv.iter().zip(&pm) {
            assert_eq!(gn, pn, "grad_views order");
            assert_eq!(gvv.len(), *pl, "{gn}: grad length");
        }
    }
}
