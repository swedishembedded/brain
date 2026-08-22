// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host reference forward + analytic backward for CosyVoice's speech-token LM
//! (`crate::llm::CosyVoiceLm`'s trainable graph): a Qwen2-style causal decoder
//! stack - RMSNorm -> biased QKV -> half-split RoPE -> causal GQA attention ->
//! output projection -> residual -> RMSNorm -> SwiGLU MLP -> residual, matching
//! `qwen3::model::Qwen::forward_steps`'s op order exactly (verified by reading
//! that function, not re-derived from a paper: RMSNorm before Q/K/V, a biased
//! QKV for Qwen2 - `cfg.attn_bias` - half-split rotate-half RoPE with
//! `inv_freq = theta^(-2i/head_dim)`, `1/sqrt(head_dim)` attention scaling, an
//! unbiased output projection, then the identical shape again around a SwiGLU
//! MLP) - plus CosyVoice's own bolted-on embedding tables and untied
//! `llm_decoder` head.
//!
//! ## Why this is a fresh host reference and not a call into `qwen3::Qwen`
//!
//! `crate::llm::CosyVoiceLm` never runs `qwen3::Qwen`'s batched training graph
//! (`set_batch`/`forward`/`backward`): it drives a **decode-only** build
//! (`Qwen::from_tensors_decode`) one row at a time through `step_embed`, which
//! allocates no backward buffers at all (`Qwen::run_backward` asserts
//! `!self.decode_only`). Worse, its three bolted-on tables
//! (`llm_embedding`/`speech_embedding`/`llm_decoder`) live entirely outside
//! `qwen3::Qwen`'s own parameter set and disagree in row count with the
//! backbone's own tied `tok.weight`/`lm_head` (151936 real BPE ids vs a
//! separate ~6564/6761-row speech vocabulary) - `qwen3::Qwen`'s own training
//! path assumes ONE table shared by embedding and (tied or untied) head. The
//! nearest existing seam, `Qwen::enable_mm_splice`, replaces a single
//! CONTIGUOUS row range with externally-supplied embeddings while every row
//! outside it still comes from the backbone's OWN tied table - CosyVoice's
//! layout needs three genuinely disjoint row sources feeding one sequence
//! (frozen backbone text embedding, a small special-token table, and the
//! speech-token table), which does not fit that seam without changing its
//! contract. Retrofitting `qwen3::Qwen` to expose "drive the transformer body
//! from an externally-assembled per-row embedding and read back a raw hidden
//! state" would be new, invasive surface area on a heavily-depended-on shared
//! crate serving many other architectures.
//!
//! Instead this module follows the pattern this workspace already uses for
//! exactly this situation - a model whose trainable graph does not fit an
//! existing device-model seam gets a **fresh, self-contained, `Fp`-generic
//! host reference** (`wan::grad`/`wan::modelgrad`, `flux2::grad`,
//! `ltxv::grad`), checked against the served path only by *architecture*
//! (matching per-layer op order and conventions), never by shared code - the
//! documented gradcheck-oracle exception: an oracle sharing code with the
//! thing it checks proves nothing. One implementation, two instantiations:
//! `f64` is the finite-difference gradcheck oracle
//! (`gradcheck::check_cosyvoice_lm_block`/`check_cosyvoice_lm`), `f32` is the
//! host trainer (`crate::lmlora`, `tests/lm_overfit.rs`) - so oracle and
//! trainer cannot drift apart.
//!
//! ## Training objective
//!
//! Teacher-forced next-speech-token prediction, matching the reference
//! recipe's intent: the input sequence is `sos ++ text ++ task_id ++
//! speech[..-1]` (the same prompt shape `crate::llm`'s module doc documents,
//! minus the final speech token, which has nothing left to predict) and the
//! loss is the mean cross-entropy of `llm_decoder(hidden)` against the FULL
//! `speech` sequence at every position from `task_id` onward - exactly the
//! positions whose hidden state predicts a speech token in
//! `crate::llm::CosyVoiceLm::generate`. Text/sos positions never contribute a
//! loss term (there is nothing for the LM to predict there), matching the
//! reference's own masked cross-entropy.
//!
//! ## Two backends, no kernels to differ between them
//!
//! This module dispatches no `gpu_core` step, no WGSL, no `Backend` at all -
//! every op is a plain scalar Rust loop, generic only over [`Fp`] (confirm
//! with `grep -n "gpu_core\|Backend\|Gpu::" crates/cosyvoice/src/lmgrad.rs`:
//! no hits). The documented workgroup-reduction-on-`backend-cpu` bug class is
//! about WGSL kernel dispatch diverging between `backend-wgpu` and
//! `backend-cpu`; there is no kernel here for that class to hide in, which is
//! why `gradcheck`'s `check_cosyvoice_lm*` tests are run - and pass
//! identically, by construction - under both `BRAIN_DEVICE` settings rather
//! than skipped for one of them.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Scalar type the reference math is generic over - see the module doc.
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
}

fn acc<T: Fp>(dst: &mut [T], src: &[T]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// `y[o] = Σ_i w[o·inn+i]·x[i] (+ b[o])`.
fn linear_fwd<T: Fp>(w: &[T], b: Option<&[T]>, x: &[T], out: usize, inn: usize) -> Vec<T> {
    let mut y = Vec::with_capacity(out);
    for o in 0..out {
        let row = &w[o * inn..(o + 1) * inn];
        let mut a = T::ZERO;
        for i in 0..inn {
            a += row[i] * x[i];
        }
        if let Some(b) = b {
            a += b[o];
        }
        y.push(a);
    }
    y
}

/// Returns `(dw [out*inn], db [out], dx [inn])` - caller accumulates into its
/// own running gradient buffers (`db` is harmless to compute even when the
/// projection is bias-free; the caller simply does not read it).
fn linear_bwd<T: Fp>(w: &[T], x: &[T], dy: &[T], out: usize, inn: usize) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut dw = vec![T::ZERO; out * inn];
    let mut db = vec![T::ZERO; out];
    let mut dx = vec![T::ZERO; inn];
    for o in 0..out {
        let dyo = dy[o];
        db[o] = dyo;
        let wrow = &w[o * inn..(o + 1) * inn];
        let dwrow = &mut dw[o * inn..(o + 1) * inn];
        for i in 0..inn {
            dwrow[i] = dyo * x[i];
            dx[i] += dyo * wrow[i];
        }
    }
    (dw, db, dx)
}

/// `y = x·inv·gamma`, `inv = 1/sqrt(mean(x^2)+eps)`. Returns `(y, inv)`.
fn rmsnorm_fwd<T: Fp>(x: &[T], gamma: &[T], eps: f64, d: usize) -> (Vec<T>, T) {
    let mut ss = 0.0f64;
    for &xi in x {
        ss += xi.f64() * xi.f64();
    }
    let mean = ss / d as f64;
    let inv = T::fr(1.0 / (mean + eps).sqrt());
    let y: Vec<T> = (0..d).map(|i| x[i] * inv * gamma[i]).collect();
    (y, inv)
}

/// Standard RMSNorm backward: `dgamma_i = dy_i·x_i·inv`,
/// `dx_j = inv·(dxhat_j − inv²/d·x_j·Σ_i dxhat_i·x_i)` with `dxhat_i = dy_i·gamma_i`.
fn rmsnorm_bwd<T: Fp>(x: &[T], gamma: &[T], inv: T, dy: &[T], d: usize) -> (Vec<T>, Vec<T>) {
    let mut dgamma = vec![T::ZERO; d];
    let mut dxhat = vec![T::ZERO; d];
    let mut sum_dx = T::ZERO;
    for i in 0..d {
        dgamma[i] = dy[i] * x[i] * inv;
        dxhat[i] = dy[i] * gamma[i];
        sum_dx += dxhat[i] * x[i];
    }
    let inv3_over_d = inv * inv * inv * T::fr(1.0 / d as f64);
    let dx: Vec<T> = (0..d).map(|j| inv * dxhat[j] - inv3_over_d * x[j] * sum_dx).collect();
    (dx, dgamma)
}

/// Half-split ("rotate-half") RoPE, in place, one row of `n_heads·head_dim`.
/// `inv_freq_i = theta^(-2i/head_dim)`, matching `block::rope_fwd`'s
/// convention (verified against `crates/qwen3/src/model.rs`'s own per-layer
/// dispatch, not assumed).
fn rope_apply<T: Fp>(buf: &mut [T], n_heads: usize, hd: usize, pos: usize, theta: f64) {
    let half = hd / 2;
    for h in 0..n_heads {
        let base = h * hd;
        for i in 0..half {
            let inv_freq = theta.powf(-2.0 * (i as f64) / (hd as f64));
            let ang = pos as f64 * inv_freq;
            let (s, c) = ang.sin_cos();
            let (cs, sn) = (T::fr(c), T::fr(s));
            let x1 = buf[base + i];
            let x2 = buf[base + half + i];
            buf[base + i] = x1 * cs - x2 * sn;
            buf[base + half + i] = x2 * cs + x1 * sn;
        }
    }
}

/// Backward of [`rope_apply`]: the per-pair rotation is orthogonal, so its
/// adjoint is the same rotation with the off-diagonal sign flipped.
fn rope_bwd<T: Fp>(dbuf: &mut [T], n_heads: usize, hd: usize, pos: usize, theta: f64) {
    let half = hd / 2;
    for h in 0..n_heads {
        let base = h * hd;
        for i in 0..half {
            let inv_freq = theta.powf(-2.0 * (i as f64) / (hd as f64));
            let ang = pos as f64 * inv_freq;
            let (s, c) = ang.sin_cos();
            let (cs, sn) = (T::fr(c), T::fr(s));
            let dy1 = dbuf[base + i];
            let dy2 = dbuf[base + half + i];
            dbuf[base + i] = cs * dy1 + sn * dy2;
            dbuf[base + half + i] = cs * dy2 - sn * dy1;
        }
    }
}

/// Which bolted-on table an input row's embedding comes from - mirrors
/// `crate::config::SpecialTokenSource`: `Special` reads `special_embed` when
/// `LmDims::special_vocab > 0` (CosyVoice 2's dedicated `llm_embedding`
/// table), or `speech_embed` when it is `0` (CosyVoice 3: `sos`/`task_id` are
/// just rows of the same table speech tokens use).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokKind {
    Text,
    Special,
    Speech,
}

/// Shape of the LM being differentiated. Deliberately non-degenerate
/// (`n_heads ≠ head_dim`, `d_model ≠ n_heads·head_dim`) so a head-count/
/// head-width or hidden-width swap could not pass unnoticed.
#[derive(Clone, Copy, Debug)]
pub struct LmDims {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub rope_theta: f64,
    pub rms_eps: f64,
    pub text_vocab: usize,
    /// `0` selects `SpecialTokenSource::SpeechEmbedding` (CosyVoice 3); `> 0`
    /// selects `SpecialTokenSource::LlmEmbedding` (CosyVoice 2) with this many
    /// dedicated rows.
    pub special_vocab: usize,
    /// Shared width of the speech input-embedding table AND the
    /// `llm_decoder` output head (`CosyVoiceLmConfig::speech_vocab`).
    pub speech_vocab: usize,
    pub decoder_bias: bool,
}

impl LmDims {
    /// A tiny, deliberately non-degenerate config for gradcheck/overfit tests.
    pub fn tiny() -> LmDims {
        LmDims {
            d_model: 20,
            n_layers: 2,
            n_heads: 6,
            n_kv_heads: 2,
            head_dim: 4,
            d_ff: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            text_vocab: 11,
            special_vocab: 2,
            speech_vocab: 9,
            decoder_bias: true,
        }
    }

    /// The CosyVoice 3 branch of [`Self::tiny`]: no dedicated special table.
    pub fn tiny_cv3() -> LmDims {
        LmDims { special_vocab: 0, decoder_bias: false, ..LmDims::tiny() }
    }

    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    fn group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerW<T> {
    pub ln1: Vec<T>,
    pub wq: Vec<T>,
    pub bq: Vec<T>,
    pub wk: Vec<T>,
    pub bk: Vec<T>,
    pub wv: Vec<T>,
    pub bv: Vec<T>,
    pub wo: Vec<T>,
    pub ln2: Vec<T>,
    pub gate: Vec<T>,
    pub up: Vec<T>,
    pub down: Vec<T>,
}

impl<T: Fp> LayerW<T> {
    /// A zeroed layer sized for `d` - the gradient-accumulation target
    /// `layer_backward` writes into, and a starting point for tests that
    /// build one layer in isolation (`gradcheck::check_cosyvoice_lm_block`).
    pub fn zeros(d: &LmDims) -> LayerW<T> {
        LayerW {
            ln1: vec![T::ZERO; d.d_model],
            wq: vec![T::ZERO; d.q_dim() * d.d_model],
            bq: vec![T::ZERO; d.q_dim()],
            wk: vec![T::ZERO; d.kv_dim() * d.d_model],
            bk: vec![T::ZERO; d.kv_dim()],
            wv: vec![T::ZERO; d.kv_dim() * d.d_model],
            bv: vec![T::ZERO; d.kv_dim()],
            wo: vec![T::ZERO; d.d_model * d.q_dim()],
            ln2: vec![T::ZERO; d.d_model],
            gate: vec![T::ZERO; d.d_ff * d.d_model],
            up: vec![T::ZERO; d.d_ff * d.d_model],
            down: vec![T::ZERO; d.d_model * d.d_ff],
        }
    }
}

/// Every trainable tensor of the LM. All fields are trainable (full
/// fine-tune), so the gradient container is the identical shape - see
/// [`LmGrads`].
#[derive(Clone, Debug, PartialEq)]
pub struct LmWeights<T> {
    pub text_embed: Vec<T>,
    pub special_embed: Vec<T>,
    pub speech_embed: Vec<T>,
    pub layers: Vec<LayerW<T>>,
    pub norm_f: Vec<T>,
    pub decoder_w: Vec<T>,
    pub decoder_b: Vec<T>,
}

/// Same shape as [`LmWeights`] - a type alias rather than a hand-mirrored
/// struct because every field here IS trainable (unlike e.g. Wan's
/// `ModelWeights`/`ModelGrads` split, which also carries non-differentiable
/// config alongside the tensors).
pub type LmGrads<T> = LmWeights<T>;

impl<T: Fp> LmWeights<T> {
    fn zeros(d: &LmDims) -> LmWeights<T> {
        LmWeights {
            text_embed: vec![T::ZERO; d.text_vocab * d.d_model],
            special_embed: vec![T::ZERO; d.special_vocab * d.d_model],
            speech_embed: vec![T::ZERO; d.speech_vocab * d.d_model],
            layers: (0..d.n_layers).map(|_| LayerW::zeros(d)).collect(),
            norm_f: vec![T::ZERO; d.d_model],
            decoder_w: vec![T::ZERO; d.speech_vocab * d.d_model],
            decoder_b: vec![T::ZERO; if d.decoder_bias { d.speech_vocab } else { 0 }],
        }
    }
}

fn fill<T: Fp>(v: &mut [T], scale: f64, rng: &mut data::rng::Lcg) {
    for x in v.iter_mut() {
        *x = T::fr(rng.signed() as f64 * scale);
    }
}

/// Deterministic init (`data::rng::Lcg`, this repo's sanctioned generator for
/// both test fixtures and production deterministic init): norms at gain 1,
/// everything else small random.
pub fn init_weights<T: Fp>(d: &LmDims, seed: u64) -> LmWeights<T> {
    let mut rng = data::rng::Lcg::new(seed);
    let mut w = LmWeights::<T>::zeros(d);
    fill(&mut w.text_embed, 0.05, &mut rng);
    fill(&mut w.special_embed, 0.05, &mut rng);
    fill(&mut w.speech_embed, 0.05, &mut rng);
    for layer in w.layers.iter_mut() {
        for x in layer.ln1.iter_mut() {
            *x = T::ONE;
        }
        fill(&mut layer.wq, 0.1, &mut rng);
        fill(&mut layer.bq, 0.02, &mut rng);
        fill(&mut layer.wk, 0.1, &mut rng);
        fill(&mut layer.bk, 0.02, &mut rng);
        fill(&mut layer.wv, 0.1, &mut rng);
        fill(&mut layer.bv, 0.02, &mut rng);
        fill(&mut layer.wo, 0.1, &mut rng);
        for x in layer.ln2.iter_mut() {
            *x = T::ONE;
        }
        fill(&mut layer.gate, 0.1, &mut rng);
        fill(&mut layer.up, 0.1, &mut rng);
        fill(&mut layer.down, 0.1, &mut rng);
    }
    for x in w.norm_f.iter_mut() {
        *x = T::ONE;
    }
    fill(&mut w.decoder_w, 0.1, &mut rng);
    fill(&mut w.decoder_b, 0.02, &mut rng);
    w
}

fn embed_row<'a, T: Fp>(d: &LmDims, w: &'a LmWeights<T>, kind: TokKind, id: usize) -> &'a [T] {
    let dm = d.d_model;
    match kind {
        TokKind::Text => &w.text_embed[id * dm..(id + 1) * dm],
        TokKind::Special if d.special_vocab > 0 => &w.special_embed[id * dm..(id + 1) * dm],
        TokKind::Special => &w.speech_embed[id * dm..(id + 1) * dm],
        TokKind::Speech => &w.speech_embed[id * dm..(id + 1) * dm],
    }
}

fn embed_row_grad_mut<'a, T: Fp>(d: &LmDims, g: &'a mut LmGrads<T>, kind: TokKind, id: usize) -> &'a mut [T] {
    let dm = d.d_model;
    match kind {
        TokKind::Text => &mut g.text_embed[id * dm..(id + 1) * dm],
        TokKind::Special if d.special_vocab > 0 => &mut g.special_embed[id * dm..(id + 1) * dm],
        TokKind::Special => &mut g.speech_embed[id * dm..(id + 1) * dm],
        TokKind::Speech => &mut g.speech_embed[id * dm..(id + 1) * dm],
    }
}

/// One training example: `sos ++ text_ids ++ task_id ++ speech_tokens[..-1]`
/// in, `speech_tokens` as the teacher-forced target (see the module doc's
/// "training objective").
#[derive(Clone, Debug)]
pub struct Example {
    pub text_ids: Vec<usize>,
    pub special_sos: usize,
    pub special_task: usize,
    /// At least one entry; the last is the loss target for the final
    /// position and is never fed back in as an input embedding.
    pub speech_tokens: Vec<usize>,
}

pub struct LayerCache<T> {
    x_in: Vec<T>,
    xn1: Vec<T>,
    inv1: Vec<T>,
    v: Vec<T>,
    qr: Vec<T>,
    kr: Vec<T>,
    probs: Vec<T>,
    ctx: Vec<T>,
    xmid: Vec<T>,
    xn2: Vec<T>,
    inv2: Vec<T>,
    gate_pre: Vec<T>,
    up: Vec<T>,
    h_act: Vec<T>,
}

pub fn layer_forward<T: Fp>(d: &LmDims, w: &LayerW<T>, x_in: &[T], n: usize) -> (Vec<T>, LayerCache<T>) {
    let dm = d.d_model;
    let (hq, hkv, hd) = (d.q_dim(), d.kv_dim(), d.head_dim);
    let (nh, nkv, grp) = (d.n_heads, d.n_kv_heads, d.group());
    let ff = d.d_ff;

    let mut xn1 = vec![T::ZERO; n * dm];
    let mut inv1 = vec![T::ZERO; n];
    for t in 0..n {
        let (y, inv) = rmsnorm_fwd(&x_in[t * dm..(t + 1) * dm], &w.ln1, d.rms_eps, dm);
        xn1[t * dm..(t + 1) * dm].copy_from_slice(&y);
        inv1[t] = inv;
    }

    let mut qr = vec![T::ZERO; n * hq];
    let mut kr = vec![T::ZERO; n * hkv];
    let mut v = vec![T::ZERO; n * hkv];
    for t in 0..n {
        let xr = &xn1[t * dm..(t + 1) * dm];
        let q = linear_fwd(&w.wq, Some(&w.bq), xr, hq, dm);
        let k = linear_fwd(&w.wk, Some(&w.bk), xr, hkv, dm);
        let vv = linear_fwd(&w.wv, Some(&w.bv), xr, hkv, dm);
        qr[t * hq..(t + 1) * hq].copy_from_slice(&q);
        kr[t * hkv..(t + 1) * hkv].copy_from_slice(&k);
        v[t * hkv..(t + 1) * hkv].copy_from_slice(&vv);
    }
    for t in 0..n {
        rope_apply(&mut qr[t * hq..(t + 1) * hq], nh, hd, t, d.rope_theta);
        rope_apply(&mut kr[t * hkv..(t + 1) * hkv], nkv, hd, t, d.rope_theta);
    }

    let mut probs = vec![T::ZERO; nh * n * n];
    let mut ctx = vec![T::ZERO; n * hq];
    let scale = T::fr(1.0 / (hd as f64).sqrt());
    for h in 0..nh {
        let kvh = h / grp;
        for t in 0..n {
            let qrow = &qr[t * hq + h * hd..t * hq + h * hd + hd];
            let mut scores = vec![T::ZERO; t + 1];
            for s in 0..=t {
                let krow = &kr[s * hkv + kvh * hd..s * hkv + kvh * hd + hd];
                let mut a = T::ZERO;
                for i in 0..hd {
                    a += qrow[i] * krow[i];
                }
                scores[s] = a * scale;
            }
            let maxv = scores.iter().fold(scores[0], |a, &b| if b.f64() > a.f64() { b } else { a });
            let mut exps = vec![T::ZERO; t + 1];
            let mut sum = T::ZERO;
            for s in 0..=t {
                let e = T::fr((scores[s] - maxv).f64().exp());
                exps[s] = e;
                sum += e;
            }
            for s in 0..=t {
                probs[h * n * n + t * n + s] = exps[s] / sum;
            }
            let mut c = vec![T::ZERO; hd];
            for s in 0..=t {
                let p = probs[h * n * n + t * n + s];
                let vrow = &v[s * hkv + kvh * hd..s * hkv + kvh * hd + hd];
                for i in 0..hd {
                    c[i] += p * vrow[i];
                }
            }
            ctx[t * hq + h * hd..t * hq + h * hd + hd].copy_from_slice(&c);
        }
    }

    let mut proj = vec![T::ZERO; n * dm];
    for t in 0..n {
        let po = linear_fwd(&w.wo, None, &ctx[t * hq..(t + 1) * hq], dm, hq);
        proj[t * dm..(t + 1) * dm].copy_from_slice(&po);
    }
    let mut xmid = vec![T::ZERO; n * dm];
    for i in 0..n * dm {
        xmid[i] = x_in[i] + proj[i];
    }

    let mut xn2 = vec![T::ZERO; n * dm];
    let mut inv2 = vec![T::ZERO; n];
    for t in 0..n {
        let (y, inv) = rmsnorm_fwd(&xmid[t * dm..(t + 1) * dm], &w.ln2, d.rms_eps, dm);
        xn2[t * dm..(t + 1) * dm].copy_from_slice(&y);
        inv2[t] = inv;
    }
    let mut gate_pre = vec![T::ZERO; n * ff];
    let mut up = vec![T::ZERO; n * ff];
    let mut h_act = vec![T::ZERO; n * ff];
    for t in 0..n {
        let xr = &xn2[t * dm..(t + 1) * dm];
        let g = linear_fwd(&w.gate, None, xr, ff, dm);
        let u = linear_fwd(&w.up, None, xr, ff, dm);
        for i in 0..ff {
            let gi = g[i];
            let sig = T::fr(1.0 / (1.0 + (-gi.f64()).exp()));
            h_act[t * ff + i] = gi * sig * u[i];
        }
        gate_pre[t * ff..(t + 1) * ff].copy_from_slice(&g);
        up[t * ff..(t + 1) * ff].copy_from_slice(&u);
    }
    let mut mlp_out = vec![T::ZERO; n * dm];
    for t in 0..n {
        let o = linear_fwd(&w.down, None, &h_act[t * ff..(t + 1) * ff], dm, ff);
        mlp_out[t * dm..(t + 1) * dm].copy_from_slice(&o);
    }
    let mut out = vec![T::ZERO; n * dm];
    for i in 0..n * dm {
        out[i] = xmid[i] + mlp_out[i];
    }

    (out, LayerCache { x_in: x_in.to_vec(), xn1, inv1, v, qr, kr, probs, ctx, xmid, xn2, inv2, gate_pre, up, h_act })
}

#[allow(clippy::too_many_arguments)]
pub fn layer_backward<T: Fp>(d: &LmDims, w: &LayerW<T>, c: &LayerCache<T>, dout: &[T], n: usize, g: &mut LayerW<T>) -> Vec<T> {
    let dm = d.d_model;
    let (hq, hkv, hd) = (d.q_dim(), d.kv_dim(), d.head_dim);
    let (nh, grp) = (d.n_heads, d.group());
    let ff = d.d_ff;

    // Second residual: `out = xmid + down(h_act)`.
    let mut d_h_act = vec![T::ZERO; n * ff];
    for t in 0..n {
        let (dw, _db, dx) = linear_bwd(&w.down, &c.h_act[t * ff..(t + 1) * ff], &dout[t * dm..(t + 1) * dm], dm, ff);
        acc(&mut g.down, &dw);
        d_h_act[t * ff..(t + 1) * ff].copy_from_slice(&dx);
    }
    let mut d_gate_pre = vec![T::ZERO; n * ff];
    let mut d_up = vec![T::ZERO; n * ff];
    for i in 0..n * ff {
        let gi = c.gate_pre[i];
        let sig = T::fr(1.0 / (1.0 + (-gi.f64()).exp()));
        let silu = gi * sig;
        // d(silu)/dgate = sigmoid(x)*(1 + x*(1-sigmoid(x)))
        let dsilu = sig * (T::ONE + gi * (T::ONE - sig));
        d_gate_pre[i] = d_h_act[i] * c.up[i] * dsilu;
        d_up[i] = d_h_act[i] * silu;
    }
    let mut d_xn2 = vec![T::ZERO; n * dm];
    for t in 0..n {
        let (dw_g, _, dx_g) = linear_bwd(&w.gate, &c.xn2[t * dm..(t + 1) * dm], &d_gate_pre[t * ff..(t + 1) * ff], ff, dm);
        let (dw_u, _, dx_u) = linear_bwd(&w.up, &c.xn2[t * dm..(t + 1) * dm], &d_up[t * ff..(t + 1) * ff], ff, dm);
        acc(&mut g.gate, &dw_g);
        acc(&mut g.up, &dw_u);
        for i in 0..dm {
            d_xn2[t * dm + i] = dx_g[i] + dx_u[i];
        }
    }
    let mut d_xmid_from_ln = vec![T::ZERO; n * dm];
    for t in 0..n {
        let (dx, dgamma) = rmsnorm_bwd(&c.xmid[t * dm..(t + 1) * dm], &w.ln2, c.inv2[t], &d_xn2[t * dm..(t + 1) * dm], dm);
        acc(&mut g.ln2, &dgamma);
        d_xmid_from_ln[t * dm..(t + 1) * dm].copy_from_slice(&dx);
    }
    let mut d_xmid = vec![T::ZERO; n * dm];
    for i in 0..n * dm {
        d_xmid[i] = dout[i] + d_xmid_from_ln[i];
    }

    // First residual: `xmid = x_in + wo(ctx)`.
    let mut d_ctx = vec![T::ZERO; n * hq];
    for t in 0..n {
        let (dw, _, dx) = linear_bwd(&w.wo, &c.ctx[t * hq..(t + 1) * hq], &d_xmid[t * dm..(t + 1) * dm], dm, hq);
        acc(&mut g.wo, &dw);
        d_ctx[t * hq..(t + 1) * hq].copy_from_slice(&dx);
    }

    let mut d_qr = vec![T::ZERO; n * hq];
    let mut d_kr = vec![T::ZERO; n * hkv];
    let mut d_v = vec![T::ZERO; n * hkv];
    let scale = T::fr(1.0 / (hd as f64).sqrt());
    for h in 0..nh {
        let kvh = h / grp;
        for t in 0..n {
            let dc = &d_ctx[t * hq + h * hd..t * hq + h * hd + hd];
            let mut d_probs = vec![T::ZERO; t + 1];
            for s in 0..=t {
                let vrow = &c.v[s * hkv + kvh * hd..s * hkv + kvh * hd + hd];
                let mut a = T::ZERO;
                for i in 0..hd {
                    a += dc[i] * vrow[i];
                }
                d_probs[s] = a;
                let p = c.probs[h * n * n + t * n + s];
                for i in 0..hd {
                    d_v[s * hkv + kvh * hd + i] += p * dc[i];
                }
            }
            let mut dot = T::ZERO;
            for (s, &dp) in d_probs.iter().enumerate() {
                dot += c.probs[h * n * n + t * n + s] * dp;
            }
            let mut d_scores = vec![T::ZERO; t + 1];
            for s in 0..=t {
                let p = c.probs[h * n * n + t * n + s];
                d_scores[s] = p * (d_probs[s] - dot);
            }
            let qrow = &c.qr[t * hq + h * hd..t * hq + h * hd + hd];
            for s in 0..=t {
                let ds = d_scores[s] * scale;
                let krow = &c.kr[s * hkv + kvh * hd..s * hkv + kvh * hd + hd];
                for i in 0..hd {
                    d_qr[t * hq + h * hd + i] += ds * krow[i];
                    d_kr[s * hkv + kvh * hd + i] += ds * qrow[i];
                }
            }
        }
    }
    for t in 0..n {
        rope_bwd(&mut d_qr[t * hq..(t + 1) * hq], nh, hd, t, d.rope_theta);
        rope_bwd(&mut d_kr[t * hkv..(t + 1) * hkv], d.n_kv_heads, hd, t, d.rope_theta);
    }

    let mut d_xn1 = vec![T::ZERO; n * dm];
    for t in 0..n {
        let (dw_q, db_q, dx_q) = linear_bwd(&w.wq, &c.xn1[t * dm..(t + 1) * dm], &d_qr[t * hq..(t + 1) * hq], hq, dm);
        let (dw_k, db_k, dx_k) = linear_bwd(&w.wk, &c.xn1[t * dm..(t + 1) * dm], &d_kr[t * hkv..(t + 1) * hkv], hkv, dm);
        let (dw_v, db_v, dx_v) = linear_bwd(&w.wv, &c.xn1[t * dm..(t + 1) * dm], &d_v[t * hkv..(t + 1) * hkv], hkv, dm);
        acc(&mut g.wq, &dw_q);
        acc(&mut g.bq, &db_q);
        acc(&mut g.wk, &dw_k);
        acc(&mut g.bk, &db_k);
        acc(&mut g.wv, &dw_v);
        acc(&mut g.bv, &db_v);
        for i in 0..dm {
            d_xn1[t * dm + i] = dx_q[i] + dx_k[i] + dx_v[i];
        }
    }
    let mut d_x_in = vec![T::ZERO; n * dm];
    for t in 0..n {
        let (dx, dgamma) = rmsnorm_bwd(&c.x_in[t * dm..(t + 1) * dm], &w.ln1, c.inv1[t], &d_xn1[t * dm..(t + 1) * dm], dm);
        acc(&mut g.ln1, &dgamma);
        for i in 0..dm {
            d_x_in[t * dm + i] = d_xmid[t * dm + i] + dx[i];
        }
    }
    d_x_in
}

/// The forward pass's cached activations - opaque to callers outside this
/// module, threaded from [`forward`] to [`loss`]/[`backward`].
pub struct LmCache<T> {
    kinds: Vec<TokKind>,
    ids: Vec<usize>,
    layer_caches: Vec<LayerCache<T>>,
    pre_final: Vec<T>,
    hidden_final: Vec<T>,
    inv_f: Vec<T>,
    logits: Vec<T>,
    pred_start: usize,
}

pub fn forward<T: Fp>(d: &LmDims, w: &LmWeights<T>, ex: &Example) -> LmCache<T> {
    assert!(!ex.speech_tokens.is_empty(), "an example needs at least one speech token to predict");
    let n_text = ex.text_ids.len();
    let n_speech_in = ex.speech_tokens.len() - 1;
    let n = 1 + n_text + 1 + n_speech_in;
    let mut kinds = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);
    kinds.push(TokKind::Special);
    ids.push(ex.special_sos);
    for &t in &ex.text_ids {
        kinds.push(TokKind::Text);
        ids.push(t);
    }
    kinds.push(TokKind::Special);
    ids.push(ex.special_task);
    for &s in &ex.speech_tokens[..n_speech_in] {
        kinds.push(TokKind::Speech);
        ids.push(s);
    }

    let dm = d.d_model;
    let mut x = vec![T::ZERO; n * dm];
    for t in 0..n {
        let row = embed_row(d, w, kinds[t], ids[t]);
        x[t * dm..(t + 1) * dm].copy_from_slice(row);
    }
    let mut layer_caches = Vec::with_capacity(d.n_layers);
    for l in 0..d.n_layers {
        let (out, c) = layer_forward(d, &w.layers[l], &x, n);
        layer_caches.push(c);
        x = out;
    }
    let pre_final = x;
    let mut hidden_final = vec![T::ZERO; n * dm];
    let mut inv_f = vec![T::ZERO; n];
    for t in 0..n {
        let (y, inv) = rmsnorm_fwd(&pre_final[t * dm..(t + 1) * dm], &w.norm_f, d.rms_eps, dm);
        hidden_final[t * dm..(t + 1) * dm].copy_from_slice(&y);
        inv_f[t] = inv;
    }
    let pred_start = 1 + n_text;
    let n_pred = n - pred_start;
    let v = d.speech_vocab;
    let mut logits = vec![T::ZERO; n_pred * v];
    for i in 0..n_pred {
        let t = pred_start + i;
        let hrow = &hidden_final[t * dm..(t + 1) * dm];
        let bias = if d.decoder_bias { Some(&w.decoder_b[..]) } else { None };
        let lo = linear_fwd(&w.decoder_w, bias, hrow, v, dm);
        logits[i * v..(i + 1) * v].copy_from_slice(&lo);
    }
    LmCache { kinds, ids, layer_caches, pre_final, hidden_final, inv_f, logits, pred_start }
}

/// Mean cross-entropy over the predicted positions. Returns `(loss, dlogits)`.
pub fn loss<T: Fp>(d: &LmDims, cache: &LmCache<T>, targets: &[usize]) -> (f64, Vec<T>) {
    let v = d.speech_vocab;
    let n_pred = targets.len();
    assert_eq!(cache.logits.len(), n_pred * v, "targets must match the cache's predicted-position count");
    let mut total = 0.0f64;
    let mut dlogits = vec![T::ZERO; n_pred * v];
    for i in 0..n_pred {
        let row = &cache.logits[i * v..(i + 1) * v];
        let maxv = row.iter().fold(row[0], |a, &b| if b.f64() > a.f64() { b } else { a });
        let mut sumexp = 0.0f64;
        let mut exps = vec![0.0f64; v];
        for j in 0..v {
            let e = (row[j] - maxv).f64().exp();
            exps[j] = e;
            sumexp += e;
        }
        let logsumexp = sumexp.ln();
        let tgt = targets[i];
        let logp_t = (row[tgt] - maxv).f64() - logsumexp;
        total += -logp_t;
        for j in 0..v {
            let p = exps[j] / sumexp;
            let ind = if j == tgt { 1.0 } else { 0.0 };
            dlogits[i * v + j] = T::fr((p - ind) / n_pred as f64);
        }
    }
    (total / n_pred as f64, dlogits)
}

pub fn backward<T: Fp>(d: &LmDims, w: &LmWeights<T>, cache: &LmCache<T>, dlogits: &[T]) -> LmGrads<T> {
    let mut g = LmWeights::<T>::zeros(d);
    let dm = d.d_model;
    let v = d.speech_vocab;
    let n = cache.pre_final.len() / dm;
    let n_pred = n - cache.pred_start;

    let mut d_hidden_final = vec![T::ZERO; n * dm];
    for i in 0..n_pred {
        let t = cache.pred_start + i;
        let hrow = &cache.hidden_final[t * dm..(t + 1) * dm];
        let (dw, db, dx) = linear_bwd(&w.decoder_w, hrow, &dlogits[i * v..(i + 1) * v], v, dm);
        acc(&mut g.decoder_w, &dw);
        if d.decoder_bias {
            acc(&mut g.decoder_b, &db);
        }
        d_hidden_final[t * dm..(t + 1) * dm].copy_from_slice(&dx);
    }
    let mut d_x = vec![T::ZERO; n * dm];
    for t in 0..n {
        let (dx, dgamma) = rmsnorm_bwd(&cache.pre_final[t * dm..(t + 1) * dm], &w.norm_f, cache.inv_f[t], &d_hidden_final[t * dm..(t + 1) * dm], dm);
        acc(&mut g.norm_f, &dgamma);
        d_x[t * dm..(t + 1) * dm].copy_from_slice(&dx);
    }
    for l in (0..d.n_layers).rev() {
        d_x = layer_backward(d, &w.layers[l], &cache.layer_caches[l], &d_x, n, &mut g.layers[l]);
    }
    for t in 0..n {
        let dst = embed_row_grad_mut(d, &mut g, cache.kinds[t], cache.ids[t]);
        acc(dst, &d_x[t * dm..(t + 1) * dm]);
    }
    g
}

/// Forward + loss + backward on one example - the unit `gradcheck` and the
/// host trainer both drive.
pub fn grads<T: Fp>(d: &LmDims, w: &LmWeights<T>, ex: &Example) -> (f64, LmGrads<T>) {
    let cache = forward(d, w, ex);
    let (l, dlogits) = loss(d, &cache, &ex.speech_tokens);
    let g = backward(d, w, &cache, &dlogits);
    (l, g)
}

/// Every trainable tensor, named, in a fixed order - the FD loop's and the
/// host trainer's shared enumeration.
/// One layer's tensors, named - the single list both [`params_mut`]'s
/// per-layer loop and [`layer_grad_views`] read, so the whole-model
/// enumeration and the block-level gradcheck's own enumeration cannot drift
/// apart into two different orderings of the same fields (the workspace's
/// own documented failure shape for a name list kept in two places).
pub fn layer_params_mut<T>(w: &mut LayerW<T>) -> Vec<(&'static str, &mut Vec<T>)> {
    vec![
        ("ln1", &mut w.ln1),
        ("wq", &mut w.wq),
        ("bq", &mut w.bq),
        ("wk", &mut w.wk),
        ("bk", &mut w.bk),
        ("wv", &mut w.wv),
        ("bv", &mut w.bv),
        ("wo", &mut w.wo),
        ("ln2", &mut w.ln2),
        ("gate", &mut w.gate),
        ("up", &mut w.up),
        ("down", &mut w.down),
    ]
}

/// Same enumeration as [`layer_params_mut`], immutable.
pub fn layer_grad_views<T>(g: &LayerW<T>) -> Vec<(&'static str, &Vec<T>)> {
    vec![
        ("ln1", &g.ln1),
        ("wq", &g.wq),
        ("bq", &g.bq),
        ("wk", &g.wk),
        ("bk", &g.bk),
        ("wv", &g.wv),
        ("bv", &g.bv),
        ("wo", &g.wo),
        ("ln2", &g.ln2),
        ("gate", &g.gate),
        ("up", &g.up),
        ("down", &g.down),
    ]
}

pub fn params_mut<T>(w: &mut LmWeights<T>) -> Vec<(String, &mut Vec<T>)> {
    let mut v: Vec<(String, &mut Vec<T>)> =
        vec![("text_embed".into(), &mut w.text_embed), ("special_embed".into(), &mut w.special_embed), ("speech_embed".into(), &mut w.speech_embed)];
    for (l, layer) in w.layers.iter_mut().enumerate() {
        for (name, t) in layer_params_mut(layer) {
            v.push((format!("layers.{l}.{name}"), t));
        }
    }
    v.push(("norm_f".into(), &mut w.norm_f));
    v.push(("decoder_w".into(), &mut w.decoder_w));
    v.push(("decoder_b".into(), &mut w.decoder_b));
    v
}

/// Same enumeration as [`params_mut`], immutable - the analytic side of the
/// FD loop.
pub fn grad_views<T>(g: &LmGrads<T>) -> Vec<(String, &Vec<T>)> {
    let mut v: Vec<(String, &Vec<T>)> = vec![("text_embed".into(), &g.text_embed), ("special_embed".into(), &g.special_embed), ("speech_embed".into(), &g.speech_embed)];
    for (l, layer) in g.layers.iter().enumerate() {
        for (name, t) in layer_grad_views(layer) {
            v.push((format!("layers.{l}.{name}"), t));
        }
    }
    v.push(("norm_f".into(), &g.norm_f));
    v.push(("decoder_w".into(), &g.decoder_w));
    v.push(("decoder_b".into(), &g.decoder_b));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(d: &LmDims) -> Example {
        Example { text_ids: vec![3, 5, 1, 7], special_sos: 0, special_task: if d.special_vocab > 0 { 1 } else { d.speech_vocab - 2 }, speech_tokens: vec![2, 4, 6, 1, 3] }
    }

    #[test]
    fn forward_is_finite_for_both_special_token_sources() {
        for d in [LmDims::tiny(), LmDims::tiny_cv3()] {
            let w = init_weights::<f32>(&d, 42);
            let ex = example(&d);
            let cache = forward(&d, &w, &ex);
            assert!(cache.logits.iter().all(|x| x.is_finite()));
            let (l, dlogits) = loss(&d, &cache, &ex.speech_tokens);
            assert!(l.is_finite() && l > 0.0);
            assert!(dlogits.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn params_mut_and_grad_views_agree_on_names_and_lengths() {
        let d = LmDims::tiny();
        let mut w = init_weights::<f32>(&d, 1);
        let ex = example(&d);
        let (_l, g) = grads(&d, &w, &ex);
        let pm = params_mut(&mut w);
        let gv = grad_views(&g);
        assert_eq!(pm.len(), gv.len());
        for ((pn, pv), (gn, gvv)) in pm.iter().zip(gv.iter()) {
            assert_eq!(pn, gn);
            assert_eq!(pv.len(), gvv.len(), "{pn} length mismatch between params and grads");
        }
    }
}
