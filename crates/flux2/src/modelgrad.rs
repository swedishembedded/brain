// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full FLUX.2 Klein **training** reference (host): forward + analytic backward
//! for the whole DiT under the rectified-flow velocity-MSE loss. This chains the
//! block reference ([`crate::grad`]) across the double- and single-stream
//! stacks and wraps it with the pieces a block doesn't have: the img/txt
//! embedders, the final modulated-LN + linear head, and the whole conditioning
//! path (timestep sinusoid → `time_in` MLP → the three global modulation
//! linears + the final-layer adaLN).
//!
//! The conditioning grad is the structural point: every modulated-LN site and
//! gate contributes `d_shift/d_scale/d_gate`, accumulated **across the whole
//! block stack** (the modulation is global — all double blocks share the same
//! four sites, all single blocks the same one), then routed through the
//! modulation linears into `d(silu(vec))`, through `silu'`, and back through
//! the `time_in` MLP — the coupling that trains the network as one.
//!
//! Generic over [`Fp`] like `grad`: the f64 instantiation is the FD-gradcheck
//! oracle (`tests/model_grad.rs`, `gradcheck::check_flux2`); the f32
//! instantiation is the host training path `finetune` drives. Same code, one
//! implementation.

use crate::config::Flux2Config;
use crate::grad::{
    double_backward, double_forward, layernorm, layernorm_bwd, linear, linear_bwd, single_backward,
    single_forward, dsilu, silu, Dims, DoubleCache, DoubleGrads, DoubleMods, DoubleW, Fp, Mod,
    ModGrad, SingleCache, SingleGrads, SingleW, StreamW,
};

/// Minimal config for the training reference — the [`Flux2Config`] fields the
/// host path needs plus the training latent grid (`lh×lw` tokens, no
/// reference images).
#[derive(Clone, Debug)]
pub struct Cfg {
    pub in_channels: usize,
    pub context_in_dim: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub depth_double: usize,
    pub depth_single: usize,
    pub mlp: usize,
    pub txt_len: usize,
    pub lh: usize,
    pub lw: usize,
    pub axes_dim: [usize; 4],
    pub rope_theta: f64,
}

impl Cfg {
    /// Derive from a [`Flux2Config`] at latent grid `lh×lw`.
    pub fn from_flux2(c: &Flux2Config, lh: usize, lw: usize) -> Cfg {
        Cfg {
            in_channels: c.in_channels,
            context_in_dim: c.context_in_dim,
            hidden: c.hidden,
            n_heads: c.n_heads,
            depth_double: c.depth_double,
            depth_single: c.depth_single,
            mlp: c.mlp_hidden(),
            txt_len: c.txt_len,
            lh,
            lw,
            axes_dim: c.axes_dim,
            rope_theta: c.rope_theta,
        }
    }

    /// Tiny klein-topology config for gradchecks and unit tests (head_dim 8 =
    /// Σ axes_dim — per-axis dims must be even for the interleaved RoPE pairs
    /// to exist; 2×2 latent grid).
    pub fn tiny() -> Cfg {
        Cfg {
            in_channels: 4,
            context_in_dim: 6,
            hidden: 16,
            n_heads: 2,
            depth_double: 2,
            depth_single: 2,
            mlp: 12,
            txt_len: 3,
            lh: 2,
            lw: 2,
            axes_dim: [2, 2, 2, 2],
            rope_theta: 2000.0,
        }
    }

    pub fn n_img(&self) -> usize {
        self.lh * self.lw
    }
    pub fn n(&self) -> usize {
        self.txt_len + self.n_img()
    }
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
    pub fn dims(&self) -> Dims {
        Dims { nt: self.txt_len, ni: self.n_img(), d: self.hidden, nh: self.n_heads, mlp: self.mlp }
    }
}

/// All trainable weights of the model (host, generic scalar). Blocks hold the
/// SPLIT projections; [`ModelWeights::from_tensors`] performs the same fused →
/// split slicing `model.rs` does at device-build time.
#[derive(Clone)]
pub struct ModelWeights<T> {
    pub img_in: Vec<T>,      // [D, in_channels]
    pub txt_in: Vec<T>,      // [D, context_in_dim]
    pub time_a: Vec<T>,      // [D, 256] time_in.in_layer
    pub time_b: Vec<T>,      // [D, D]   time_in.out_layer
    pub mod_img: Vec<T>,     // [6D, D]
    pub mod_txt: Vec<T>,     // [6D, D]
    pub mod_single: Vec<T>,  // [3D, D]
    pub final_adaln: Vec<T>, // [2D, D] (BFL chunk order: shift, scale)
    pub final_w: Vec<T>,     // [in_channels, D]
    pub dbl: Vec<DoubleW<T>>,
    pub sgl: Vec<SingleW<T>>,
}

/// Grads mirroring [`ModelWeights`]. The per-block grads keep their `dx` and
/// modulation-site contributions (harmless extras, useful in tests); the
/// accumulated site grads are already folded into the `mod_*`/`final_adaln`/
/// `time_*` fields here.
#[derive(Clone)]
pub struct ModelGrads<T> {
    pub img_in: Vec<T>,
    pub txt_in: Vec<T>,
    pub time_a: Vec<T>,
    pub time_b: Vec<T>,
    pub mod_img: Vec<T>,
    pub mod_txt: Vec<T>,
    pub mod_single: Vec<T>,
    pub final_adaln: Vec<T>,
    pub final_w: Vec<T>,
    pub dbl: Vec<DoubleGrads<T>>,
    pub sgl: Vec<SingleGrads<T>>,
}

/// Timestep sinusoid width: `timestep_embedding` emits `[TDIM]` and
/// `time_in.in_layer` is `[hidden, TDIM]`.
pub const TDIM: usize = 256;

/// `timestep_embedding(t·1000, 256)`: 128 freqs, **cos first** — the generic-`T`
/// twin of `model::hostmath::timestep_embedding` (angles in f64, like the device
/// path). It is a deliberate second implementation: AGENTS.md exception 1 — a
/// gradcheck oracle that shared code with the thing it checks would prove
/// nothing, and this one instantiates at `f64` for the FD check.
pub fn timestep_embedding<T: Fp>(t: f64) -> Vec<T> {
    let half = TDIM / 2;
    let x = t * 1000.0;
    let mut emb = vec![T::ZERO; TDIM];
    for i in 0..half {
        let freq = (-(10000.0f64.ln()) * i as f64 / half as f64).exp();
        let arg = x * freq;
        emb[i] = T::fr(arg.cos());
        emb[half + i] = T::fr(arg.sin());
    }
    emb
}

fn chunk<T: Fp>(m: &[T], c: usize, d: usize) -> Vec<T> {
    m[c * d..(c + 1) * d].to_vec()
}

/// Build one (shift, scale, gate) site from modulation output `m` at triple `c`.
fn site<T: Fp>(m: &[T], c: usize, d: usize) -> Mod<T> {
    Mod { shift: chunk(m, 3 * c, d), scale: chunk(m, 3 * c + 1, d), gate: chunk(m, 3 * c + 2, d) }
}

/// Saved forward state for the backward pass.
pub struct ModelCache<T> {
    te: Vec<T>,
    hpre: Vec<T>,
    h: Vec<T>,
    vec_: Vec<T>,
    sv: Vec<T>,
    dmods: DoubleMods<T>,
    smod: Mod<T>,
    fin: Mod<T>, // gate empty
    img_tokens: Vec<T>,
    ctx: Vec<T>,
    dbl_c: Vec<DoubleCache<T>>,
    sgl_c: Vec<SingleCache<T>>,
    xhat_f: Vec<T>, // final-LN xhat over the image rows
    inv_f: Vec<T>,
    n_f: Vec<T>, // modulated final-LN output (the head's input)
}

/// Full forward. `img_tokens:[n_img·in_channels]` (packed latent tokens),
/// `ctx:[txt_len·context_in_dim]`, `cos/sin:[n·head_dim/2]` (joint tables,
/// text rows first). Returns the velocity prediction for ALL image tokens
/// `[n_img·in_channels]` plus the cache.
pub fn forward<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, img_tokens: &[T], ctx: &[T], t: f64, cos: &[T], sin: &[T]) -> (Vec<T>, ModelCache<T>) {
    let (d, cin) = (cfg.hidden, cfg.in_channels);
    let (nt, ni, n) = (cfg.txt_len, cfg.n_img(), cfg.n());
    assert_eq!(img_tokens.len(), ni * cin, "img tokens size");
    assert_eq!(ctx.len(), nt * cfg.context_in_dim, "ctx size");
    assert_eq!(cos.len(), n * cfg.head_dim() / 2, "rope table size");

    // conditioning: timestep MLP (bias-free, silu between) → vec → silu → mods
    let te = timestep_embedding::<T>(t);
    let hpre = linear(&te, 1, TDIM, &w.time_a, d);
    let h: Vec<T> = hpre.iter().map(|&v| silu(v)).collect();
    let vec_ = linear(&h, 1, d, &w.time_b, d);
    let sv: Vec<T> = vec_.iter().map(|&v| silu(v)).collect();
    let m_img = linear(&sv, 1, d, &w.mod_img, 6 * d);
    let m_txt = linear(&sv, 1, d, &w.mod_txt, 6 * d);
    let m_sgl = linear(&sv, 1, d, &w.mod_single, 3 * d);
    let m_fin = linear(&sv, 1, d, &w.final_adaln, 2 * d);
    let dmods = DoubleMods {
        img1: site(&m_img, 0, d),
        img2: site(&m_img, 1, d),
        txt1: site(&m_txt, 0, d),
        txt2: site(&m_txt, 1, d),
    };
    let smod = site(&m_sgl, 0, d);
    // final layer chunk order (BFL): shift first, then scale; no gate
    let fin = Mod { shift: chunk(&m_fin, 0, d), scale: chunk(&m_fin, 1, d), gate: Vec::new() };

    // embed both streams into the joint slab x = [txt | img]
    let mut x = linear(ctx, nt, cfg.context_in_dim, &w.txt_in, d);
    x.extend(linear(img_tokens, ni, cin, &w.img_in, d));

    let dims = cfg.dims();
    let mut dbl_c = Vec::with_capacity(w.dbl.len());
    for bw in &w.dbl {
        let (o, c) = double_forward(dims, bw, &x, &dmods, cos, sin);
        x = o;
        dbl_c.push(c);
    }
    let mut sgl_c = Vec::with_capacity(w.sgl.len());
    for bw in &w.sgl {
        let (o, c) = single_forward(dims, bw, &x, &smod, cos, sin);
        x = o;
        sgl_c.push(c);
    }

    // final layer on the image rows: modulated LN → linear to in_channels
    let (xhat_f, inv_f) = layernorm(&x[nt * d..], ni, d);
    let mut n_f = vec![T::ZERO; ni * d];
    for r in 0..ni {
        for c in 0..d {
            n_f[r * d + c] = (T::ONE + fin.scale[c]) * xhat_f[r * d + c] + fin.shift[c];
        }
    }
    let pred = linear(&n_f, ni, d, &w.final_w, cin);

    let cache = ModelCache {
        te, hpre, h, vec_, sv, dmods, smod, fin,
        img_tokens: img_tokens.to_vec(), ctx: ctx.to_vec(),
        dbl_c, sgl_c, xhat_f, inv_f, n_f,
    };
    (pred, cache)
}

/// Velocity-MSE rectified-flow loss + its `dpred`: `L = mean((pred − v)²)`.
pub fn loss<T: Fp>(pred: &[T], v_target: &[T]) -> (f64, Vec<T>) {
    let n = T::fr(pred.len() as f64);
    let two = T::fr(2.0);
    let mut l = 0.0;
    let mut dpred = vec![T::ZERO; pred.len()];
    for i in 0..pred.len() {
        let e = pred[i] - v_target[i];
        l += (e * e / n).f64();
        dpred[i] = two * e / n;
    }
    (l, dpred)
}

/// Full backward from `dpred` (grad of the loss w.r.t. the predicted image
/// tokens). Accumulates the conditioning grad from EVERY modulation site plus
/// the final-layer adaLN and routes it back through the `time_in` MLP.
pub fn backward<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, cache: &ModelCache<T>, dpred: &[T]) -> ModelGrads<T> {
    let (d, cin) = (cfg.hidden, cfg.in_channels);
    let (nt, ni, n) = (cfg.txt_len, cfg.n_img(), cfg.n());
    let dims = cfg.dims();

    // ---- final layer ----
    let (dn_f, g_final_w) = linear_bwd(&cache.n_f, ni, d, &w.final_w, cin, dpred);
    let mut fin_g = ModGrad::<T>::zeros(d); // gate slot unused for the final site
    let mut dxhat_f = vec![T::ZERO; ni * d];
    for r in 0..ni {
        for c in 0..d {
            let g = dn_f[r * d + c];
            fin_g.scale[c] += g * cache.xhat_f[r * d + c];
            fin_g.shift[c] += g;
            dxhat_f[r * d + c] = (T::ONE + cache.fin.scale[c]) * g;
        }
    }
    let dx_img = layernorm_bwd(&cache.xhat_f, &cache.inv_f, ni, d, &dxhat_f);
    let mut dx = vec![T::ZERO; n * d];
    dx[nt * d..].copy_from_slice(&dx_img);

    // ---- single blocks (reverse), site grads accumulated across the stack ----
    let mut sgl_g: Vec<SingleGrads<T>> = Vec::with_capacity(w.sgl.len());
    let mut sgl_site = ModGrad::<T>::zeros(d);
    for (bw, ca) in w.sgl.iter().zip(&cache.sgl_c).rev() {
        let g = single_backward(dims, bw, &cache.smod, ca, &dx);
        dx = g.dx.clone();
        sgl_site.add(&g.m);
        sgl_g.push(g);
    }
    sgl_g.reverse();

    // ---- double blocks (reverse) ----
    let mut dbl_g: Vec<DoubleGrads<T>> = Vec::with_capacity(w.dbl.len());
    let (mut s_img1, mut s_img2) = (ModGrad::<T>::zeros(d), ModGrad::<T>::zeros(d));
    let (mut s_txt1, mut s_txt2) = (ModGrad::<T>::zeros(d), ModGrad::<T>::zeros(d));
    for (bw, ca) in w.dbl.iter().zip(&cache.dbl_c).rev() {
        let g = double_backward(dims, bw, &cache.dmods, ca, &dx);
        dx = g.dx.clone();
        s_img1.add(&g.img1);
        s_img2.add(&g.img2);
        s_txt1.add(&g.txt1);
        s_txt2.add(&g.txt2);
        dbl_g.push(g);
    }
    dbl_g.reverse();

    // ---- embedders (inputs are data → weight grads only) ----
    let (_dc, g_txt_in) = linear_bwd(&cache.ctx, nt, cfg.context_in_dim, &w.txt_in, d, &dx[..nt * d]);
    let (_di, g_img_in) = linear_bwd(&cache.img_tokens, ni, cin, &w.img_in, d, &dx[nt * d..]);

    // ---- modulation linears: site grads → d_m vectors → weight grads + d_sv ----
    let mut d_m_img = vec![T::ZERO; 6 * d];
    let mut d_m_txt = vec![T::ZERO; 6 * d];
    let mut d_m_sgl = vec![T::ZERO; 3 * d];
    let mut d_m_fin = vec![T::ZERO; 2 * d];
    let put = |dst: &mut [T], c: usize, mg: &ModGrad<T>| {
        dst[3 * c * d..(3 * c + 1) * d].copy_from_slice(&mg.shift);
        dst[(3 * c + 1) * d..(3 * c + 2) * d].copy_from_slice(&mg.scale);
        dst[(3 * c + 2) * d..(3 * c + 3) * d].copy_from_slice(&mg.gate);
    };
    put(&mut d_m_img, 0, &s_img1);
    put(&mut d_m_img, 1, &s_img2);
    put(&mut d_m_txt, 0, &s_txt1);
    put(&mut d_m_txt, 1, &s_txt2);
    put(&mut d_m_sgl, 0, &sgl_site);
    d_m_fin[..d].copy_from_slice(&fin_g.shift); // BFL: shift rows first
    d_m_fin[d..].copy_from_slice(&fin_g.scale);

    let mut d_sv = vec![T::ZERO; d];
    let mut mod_bwd = |mat: &[T], dm: &[T], rows: usize| -> Vec<T> {
        let (dsv, gw) = linear_bwd(&cache.sv, 1, d, mat, rows, dm);
        for (a, b) in d_sv.iter_mut().zip(dsv) {
            *a += b;
        }
        gw
    };
    let g_mod_img = mod_bwd(&w.mod_img, &d_m_img, 6 * d);
    let g_mod_txt = mod_bwd(&w.mod_txt, &d_m_txt, 6 * d);
    let g_mod_single = mod_bwd(&w.mod_single, &d_m_sgl, 3 * d);
    let g_final_adaln = mod_bwd(&w.final_adaln, &d_m_fin, 2 * d);

    // ---- time_in MLP: sv = silu(vec), vec = time_b @ silu(time_a @ te) ----
    let d_vec: Vec<T> = d_sv.iter().zip(&cache.vec_).map(|(&g, &v)| g * dsilu(v)).collect();
    let (d_h, g_time_b) = linear_bwd(&cache.h, 1, d, &w.time_b, d, &d_vec);
    let d_hpre: Vec<T> = d_h.iter().zip(&cache.hpre).map(|(&g, &v)| g * dsilu(v)).collect();
    let (_dte, g_time_a) = linear_bwd(&cache.te, 1, TDIM, &w.time_a, d, &d_hpre);

    ModelGrads {
        img_in: g_img_in,
        txt_in: g_txt_in,
        time_a: g_time_a,
        time_b: g_time_b,
        mod_img: g_mod_img,
        mod_txt: g_mod_txt,
        mod_single: g_mod_single,
        final_adaln: g_final_adaln,
        final_w: g_final_w,
        dbl: dbl_g,
        sgl: sgl_g,
    }
}

// ---- flow-matching batch ----

/// One training example, ready for [`forward`].
#[derive(Clone)]
pub struct Batch<T> {
    pub img: Vec<T>, // x_σ tokens [n_img·in_channels]
    pub ctx: Vec<T>,
    pub t: f64, // model time input (= σ; klein integrates σ 1→0)
    pub cos: Vec<T>,
    pub sin: Vec<T>,
    pub target: Vec<T>, // velocity v = ε − x₀
}

/// Build one rectified-flow batch from a clean latent-token set `x0`
/// (`[n_img·in_channels]`), caption features `ctx`, noise level `σ ∈ (0,1]`
/// and standard-normal `noise` (same length as `x0`):
/// `x_σ = (1−σ)·x₀ + σ·ε`, target `v = ε − x₀`, model time `t = σ` — exactly
/// the convention [`crate::pipeline`]'s Euler integrator inverts
/// (`x += dt·v` with σ stepping 1→0). RoPE tables come from
/// [`crate::position_ids`] (no reference images) through `dit::rope`.
pub fn make_flow_batch<T: Fp>(cfg: &Cfg, x0: &[T], ctx: &[T], sigma: f64, noise: &[T]) -> Batch<T> {
    assert_eq!(x0.len(), cfg.n_img() * cfg.in_channels, "latent size");
    assert_eq!(noise.len(), x0.len(), "noise size");
    assert_eq!(ctx.len(), cfg.txt_len * cfg.context_in_dim, "caption size");
    let s = T::fr(sigma);
    let img: Vec<T> = x0.iter().zip(noise).map(|(&x, &e)| (T::ONE - s) * x + s * e).collect();
    let target: Vec<T> = x0.iter().zip(noise).map(|(&x, &e)| e - x).collect();

    let ids = crate::model::position_ids(cfg.txt_len, cfg.lh, cfg.lw, &[]);
    let rc = dit::rope::RopeConfig {
        axes_dims: cfg.axes_dim.iter().map(|&a| a as u32).collect(),
        axes_lens: vec![4096, 4096, 4096, 4096],
        theta: cfg.rope_theta,
    };
    let tables = dit::rope::tables_for_ids(&rc, &ids, 4);
    let cast = |v: &[f32]| -> Vec<T> { v.iter().map(|&x| T::fr(x as f64)).collect() };
    Batch { img, ctx: ctx.to_vec(), t: sigma, cos: cast(&tables.cos), sin: cast(&tables.sin), target }
}

/// One training evaluation: forward + loss + backward. The f32 instantiation
/// is the finetune trainer's step core.
pub fn grads<T: Fp>(cfg: &Cfg, w: &ModelWeights<T>, b: &Batch<T>) -> (f64, ModelGrads<T>) {
    let (pred, cache) = forward(cfg, w, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    let (l, dpred) = loss(&pred, &b.target);
    (l, backward(cfg, w, &cache, &dpred))
}

// ---- weight construction ----

/// Take one tensor OUT of the map, by name. The whole-model weight set is the
/// same bytes in a different arrangement, so a `from_tensors` that cloned
/// would hold the model twice at its peak - 72 GB for klein-9B, which is what
/// stands between that model and a box with 184 GB of RAM shared by several
/// jobs. Removing as it converts keeps the peak at one copy plus the tensor in
/// flight.
fn take(ts: &mut crate::import::Tensors, name: &str) -> Result<Vec<f32>, String> {
    ts.remove(name).map(|(_, v)| v).ok_or_else(|| format!("from_tensors: missing {name}"))
}

fn rows(v: &[f32], cols: usize, r0: usize, r1: usize) -> Vec<f32> {
    v[r0 * cols..r1 * cols].to_vec()
}

/// One double-block stream, split out of the fused checkpoint tensors.
fn take_stream(ts: &mut crate::import::Tensors, p: &str, d: usize, mlp: usize) -> Result<StreamW<f32>, String> {
    let qkv = take(ts, &format!("{p}_attn.qkv.weight"))?;
    let (wq, wk, wv) = (rows(&qkv, d, 0, d), rows(&qkv, d, d, 2 * d), rows(&qkv, d, 2 * d, 3 * d));
    drop(qkv);
    let m0 = take(ts, &format!("{p}_mlp.0.weight"))?;
    // SwiGLU chunk order: x1 (silu-gated) is the FIRST half
    let (w1, w3) = (rows(&m0, d, 0, mlp), rows(&m0, d, mlp, 2 * mlp));
    drop(m0);
    Ok(StreamW {
        wq,
        wk,
        wv,
        nq: take(ts, &format!("{p}_attn.norm.query_norm.scale"))?,
        nk: take(ts, &format!("{p}_attn.norm.key_norm.scale"))?,
        wo: take(ts, &format!("{p}_attn.proj.weight"))?,
        w1,
        w3,
        w2: take(ts, &format!("{p}_mlp.2.weight"))?,
    })
}

impl ModelWeights<f32> {
    /// Build host training weights from imported (BFL-named) tensors, splitting
    /// the fused `qkv`/`mlp.0`/`linear1`/`linear2` exactly as
    /// [`crate::model::Flux2Model::new`] does (row slices; linear2 column
    /// split) — validated against the device forward by
    /// `tests/host_forward_parity.rs`.
    ///
    /// **Consumes `ts`**: every tensor it reads is REMOVED from the map (see
    /// [`take`]). A caller that still needs the fused layout afterwards has to
    /// clone the map first, and at klein scale it should think hard about
    /// whether it does.
    pub fn from_tensors(cfg: &Cfg, ts: &mut crate::import::Tensors) -> Result<ModelWeights<f32>, String> {
        let (d, mlp) = (cfg.hidden, cfg.mlp);
        let mut dbl = Vec::with_capacity(cfg.depth_double);
        for b in 0..cfg.depth_double {
            dbl.push(DoubleW {
                img: take_stream(ts, &format!("double_blocks.{b}.img"), d, mlp)?,
                txt: take_stream(ts, &format!("double_blocks.{b}.txt"), d, mlp)?,
            });
        }
        let mut sgl = Vec::with_capacity(cfg.depth_single);
        for b in 0..cfg.depth_single {
            let p = format!("single_blocks.{b}");
            let l1 = take(ts, &format!("{p}.linear1.weight"))?;
            let (wq, wk, wv) = (rows(&l1, d, 0, d), rows(&l1, d, d, 2 * d), rows(&l1, d, 2 * d, 3 * d));
            let (w1, w3) = (rows(&l1, d, 3 * d, 3 * d + mlp), rows(&l1, d, 3 * d + mlp, 3 * d + 2 * mlp));
            drop(l1);
            let l2 = take(ts, &format!("{p}.linear2.weight"))?;
            // linear2 is [D, D+mlp]; split its input (column) dim
            let mut wo_a = Vec::with_capacity(d * d);
            let mut wo_b = Vec::with_capacity(d * mlp);
            for r in 0..d {
                wo_a.extend_from_slice(&l2[r * (d + mlp)..r * (d + mlp) + d]);
                wo_b.extend_from_slice(&l2[r * (d + mlp) + d..(r + 1) * (d + mlp)]);
            }
            drop(l2);
            sgl.push(SingleW {
                wq,
                wk,
                wv,
                nq: take(ts, &format!("{p}.norm.query_norm.scale"))?,
                nk: take(ts, &format!("{p}.norm.key_norm.scale"))?,
                w1,
                w3,
                wo_a,
                wo_b,
            })
        }
        Ok(ModelWeights {
            img_in: take(ts, "img_in.weight")?,
            txt_in: take(ts, "txt_in.weight")?,
            time_a: take(ts, "time_in.in_layer.weight")?,
            time_b: take(ts, "time_in.out_layer.weight")?,
            mod_img: take(ts, "double_stream_modulation_img.lin.weight")?,
            mod_txt: take(ts, "double_stream_modulation_txt.lin.weight")?,
            mod_single: take(ts, "single_stream_modulation.lin.weight")?,
            final_adaln: take(ts, "final_layer.adaLN_modulation.1.weight")?,
            final_w: take(ts, "final_layer.linear.weight")?,
            dbl,
            sgl,
        })
    }
}

/// Deterministic random init at any scalar type — for gradchecks and synthetic
/// training tests (real runs import a checkpoint).
pub fn init_model<T: Fp>(cfg: &Cfg, seed: u64) -> ModelWeights<T> {
    let mut rng = data::rng::Rng::new(seed);
    let mut v = |n: usize, s: f64| -> Vec<T> {
        (0..n).map(|_| T::fr((rng.next_f64() - 0.5) * 2.0 * s)).collect()
    };
    let (d, hd, mlp) = (cfg.hidden, cfg.head_dim(), cfg.mlp);
    let stream = |v: &mut dyn FnMut(usize, f64) -> Vec<T>| StreamW {
        wq: v(d * d, 0.2),
        wk: v(d * d, 0.2),
        wv: v(d * d, 0.2),
        nq: v(hd, 0.1).iter().map(|&x| T::ONE + x).collect(),
        nk: v(hd, 0.1).iter().map(|&x| T::ONE + x).collect(),
        wo: v(d * d, 0.2),
        w1: v(mlp * d, 0.2),
        w3: v(mlp * d, 0.2),
        w2: v(d * mlp, 0.2),
    };
    let dbl = (0..cfg.depth_double).map(|_| DoubleW { img: stream(&mut v), txt: stream(&mut v) }).collect();
    let sgl = (0..cfg.depth_single)
        .map(|_| SingleW {
            wq: v(d * d, 0.2),
            wk: v(d * d, 0.2),
            wv: v(d * d, 0.2),
            nq: v(hd, 0.1).iter().map(|&x| T::ONE + x).collect(),
            nk: v(hd, 0.1).iter().map(|&x| T::ONE + x).collect(),
            w1: v(mlp * d, 0.2),
            w3: v(mlp * d, 0.2),
            wo_a: v(d * d, 0.2),
            wo_b: v(d * mlp, 0.2),
        })
        .collect();
    ModelWeights {
        img_in: v(d * cfg.in_channels, 0.2),
        txt_in: v(d * cfg.context_in_dim, 0.2),
        time_a: v(d * TDIM, 0.05),
        time_b: v(d * d, 0.05),
        mod_img: v(6 * d * d, 0.05),
        mod_txt: v(6 * d * d, 0.05),
        mod_single: v(3 * d * d, 0.05),
        final_adaln: v(2 * d * d, 0.05),
        final_w: v(cfg.in_channels * d, 0.2),
        dbl,
        sgl,
    }
}

// ---- parameter enumeration (FD tests + gradcheck::check_flux2) ----

fn stream_params<'a, T>(p: &str, s: &'a mut StreamW<T>, out: &mut Vec<(String, &'a mut Vec<T>)>) {
    out.push((format!("{p}.wq"), &mut s.wq));
    out.push((format!("{p}.wk"), &mut s.wk));
    out.push((format!("{p}.wv"), &mut s.wv));
    out.push((format!("{p}.nq"), &mut s.nq));
    out.push((format!("{p}.nk"), &mut s.nk));
    out.push((format!("{p}.wo"), &mut s.wo));
    out.push((format!("{p}.w1"), &mut s.w1));
    out.push((format!("{p}.w3"), &mut s.w3));
    out.push((format!("{p}.w2"), &mut s.w2));
}

/// Every trainable tensor, named, in a fixed order (mutable views).
pub fn params_mut<T>(w: &mut ModelWeights<T>) -> Vec<(String, &mut Vec<T>)> {
    let mut v: Vec<(String, &mut Vec<T>)> = vec![
        ("img_in".into(), &mut w.img_in),
        ("txt_in".into(), &mut w.txt_in),
        ("time_a".into(), &mut w.time_a),
        ("time_b".into(), &mut w.time_b),
        ("mod_img".into(), &mut w.mod_img),
        ("mod_txt".into(), &mut w.mod_txt),
        ("mod_single".into(), &mut w.mod_single),
        ("final_adaln".into(), &mut w.final_adaln),
        ("final_w".into(), &mut w.final_w),
    ];
    for (i, b) in w.dbl.iter_mut().enumerate() {
        stream_params(&format!("dbl{i}.img"), &mut b.img, &mut v);
        stream_params(&format!("dbl{i}.txt"), &mut b.txt, &mut v);
    }
    for (i, b) in w.sgl.iter_mut().enumerate() {
        let p = format!("sgl{i}");
        v.push((format!("{p}.wq"), &mut b.wq));
        v.push((format!("{p}.wk"), &mut b.wk));
        v.push((format!("{p}.wv"), &mut b.wv));
        v.push((format!("{p}.nq"), &mut b.nq));
        v.push((format!("{p}.nk"), &mut b.nk));
        v.push((format!("{p}.w1"), &mut b.w1));
        v.push((format!("{p}.w3"), &mut b.w3));
        v.push((format!("{p}.wo_a"), &mut b.wo_a));
        v.push((format!("{p}.wo_b"), &mut b.wo_b));
    }
    v
}

/// Gradient views in the SAME order as [`params_mut`].
pub fn grad_views<T>(g: &ModelGrads<T>) -> Vec<(String, &Vec<T>)> {
    let mut v: Vec<(String, &Vec<T>)> = vec![
        ("img_in".into(), &g.img_in),
        ("txt_in".into(), &g.txt_in),
        ("time_a".into(), &g.time_a),
        ("time_b".into(), &g.time_b),
        ("mod_img".into(), &g.mod_img),
        ("mod_txt".into(), &g.mod_txt),
        ("mod_single".into(), &g.mod_single),
        ("final_adaln".into(), &g.final_adaln),
        ("final_w".into(), &g.final_w),
    ];
    for (i, b) in g.dbl.iter().enumerate() {
        for (name, s) in [("img", &b.img), ("txt", &b.txt)] {
            let p = format!("dbl{i}.{name}");
            v.push((format!("{p}.wq"), &s.wq));
            v.push((format!("{p}.wk"), &s.wk));
            v.push((format!("{p}.wv"), &s.wv));
            v.push((format!("{p}.nq"), &s.nq));
            v.push((format!("{p}.nk"), &s.nk));
            v.push((format!("{p}.wo"), &s.wo));
            v.push((format!("{p}.w1"), &s.w1));
            v.push((format!("{p}.w3"), &s.w3));
            v.push((format!("{p}.w2"), &s.w2));
        }
    }
    for (i, b) in g.sgl.iter().enumerate() {
        let p = format!("sgl{i}");
        v.push((format!("{p}.wq"), &b.wq));
        v.push((format!("{p}.wk"), &b.wk));
        v.push((format!("{p}.wv"), &b.wv));
        v.push((format!("{p}.nq"), &b.nq));
        v.push((format!("{p}.nk"), &b.nk));
        v.push((format!("{p}.w1"), &b.w1));
        v.push((format!("{p}.w3"), &b.w3));
        v.push((format!("{p}.wo_a"), &b.wo_a));
        v.push((format!("{p}.wo_b"), &b.wo_b));
    }
    v
}
