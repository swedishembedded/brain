// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! InstantID's `image_proj` Resampler: one ArcFace embedding -> 16 ID tokens.
//!
//! Built entirely from [`model::rowemit::RowEmit`], which is the same emitter
//! `crates/pulid`'s `IDFormer` records with — the two are the same
//! `PerceiverAttention` from the same IP-Adapter lineage, and this crate adds no
//! second copy of the row arithmetic, the fused-kv strides or the attention
//! dispatch. It also adds **no kernel**.
//!
//! # The concatenation is a row offset, not a copy
//!
//! The reference computes `kv_input = cat(x, latents)` where `x` is the SINGLE
//! projected ArcFace token. Here `norm1(x)` is written into row 0 of one buffer
//! and `norm2(latents)` into rows `1..1+num_queries` of the same buffer, so:
//!
//! * `to_kv` runs over all `1 + num_queries` rows in one dispatch;
//! * `to_q` runs over the SAME buffer sliced from row 1 — the reference applies
//!   it to `norm2(latents)`, which is exactly those rows.
//!
//! Sizing k/v at `num_queries` instead of `1 + num_queries` is the trap
//! [`crate::config::ResamplerConfig::kv_rows`] exists to name: it is shape-legal
//! at one query and silently drops the ArcFace context everywhere else.
//!
//! # Buffers are reused across layers
//!
//! Every scratch buffer is shared by all `depth` layers, so a stage is only
//! readable immediately after the step that wrote it. [`Resampler::read_tap`]
//! therefore REPLAYS the graph up to that step rather than reading after one
//! full forward — the same approach `pulid::IdFormer` takes, and the reason the
//! parity ladder can gate every layer rather than only the output.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::rowemit::{RowEmit, RowKernels};

use crate::config::ResamplerConfig;

/// The kernels this crate dispatches: the shared row-emitter set plus `add2`
/// (the two residual adds) and `gelu_erf` (the feed-forward activation —
/// `nn.GELU()` with PyTorch's default `approximate='none'`, i.e. the erf form,
/// NOT the tanh approximation).
pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("region_copy", kernels::REGION_COPY),
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
];

/// LayerNorm epsilon. `nn.LayerNorm`'s default, which the reference does not
/// override.
pub const EPS: f32 = 1e-5;

fn idx(names: &[(&str, &str)], k: &str) -> usize {
    names
        .iter()
        .position(|(n, _)| *n == k)
        .unwrap_or_else(|| panic!("instantid: the Gpu was built without the `{k}` kernel"))
}

/// A named stage snapshot: the buffer, its float offset and length, and the step
/// index just after the step that wrote it.
struct Tap {
    name: String,
    buf: DeviceBuffer,
    off: usize,
    len: usize,
    step: usize,
}

/// `image_proj`: `[1, embedding_dim] -> [num_queries, output_dim]`.
pub struct Resampler {
    gpu: Gpu,
    cfg: ResamplerConfig,
    x_in: DeviceBuffer,
    out: DeviceBuffer,
    steps: Vec<Step>,
    taps: Vec<Tap>,
    /// The uploaded weights. Held so their device buffers outlive the recorded
    /// steps that bind them — the steps keep clones, but owning the map here is
    /// what makes that ownership visible rather than incidental.
    _weights: HashMap<String, DeviceBuffer>,
}

impl Resampler {
    /// Build on an existing device. `w` maps the released `image_proj` tensor
    /// names to their data (see [`crate::import`]).
    pub fn new_on(gpu: Gpu, cfg: ResamplerConfig, names: &[(&str, &str)], w: &HashMap<String, Vec<f32>>) -> Resampler {
        let k = RowKernels::resolve("instantid", names);
        let (k_add2, k_gelu) = (idx(names, "add2"), idx(names, "gelu_erf"));
        let e = RowEmit::new(&gpu, k, EPS);

        let (d, nq, kvr) = (cfg.dim, cfg.num_queries, cfg.kv_rows());
        let (inner, ff) = (cfg.inner_dim(), cfg.ff_inner());
        let od = cfg.output_dim;

        // Weights, uploaded once.
        let up: HashMap<String, DeviceBuffer> = w
            .iter()
            .map(|(n, v)| {
                let b = gpu.storage(v.len() as u64);
                gpu.write_f32(&b, v);
                (n.clone(), b)
            })
            .collect();
        let wb = |n: &str| up.get(n).unwrap_or_else(|| panic!("instantid: missing weight `{n}`"));

        // Buffers, reused across layers.
        let x_in = gpu.storage(cfg.embedding_dim as u64);
        let x1 = gpu.storage(d as u64);
        let lat_a = gpu.storage((nq * d) as u64);
        let lat_b = gpu.storage((nq * d) as u64);
        let nkv = gpu.storage((kvr * d) as u64);
        let q = gpu.storage((nq * inner) as u64);
        let kv = gpu.storage((kvr * 2 * inner) as u64);
        let scores = gpu.storage((cfg.heads * nq * kvr) as u64);
        let probs = gpu.storage((cfg.heads * nq * kvr) as u64);
        let ctx = gpu.storage((nq * inner) as u64);
        let aout = gpu.storage((nq * d) as u64);
        let ffn = gpu.storage((nq * d) as u64);
        let fh = gpu.storage((nq * ff) as u64);
        let fg = gpu.storage((nq * ff) as u64);
        let fo = gpu.storage((nq * d) as u64);
        let pout = gpu.storage((nq * od) as u64);
        let out = gpu.storage((nq * od) as u64);

        let mut s: Vec<Step> = Vec::new();
        let mut taps: Vec<Tap> = Vec::new();
        let tap = |s: &Vec<Step>, taps: &mut Vec<Tap>, name: &str, buf: &DeviceBuffer, off: usize, len: usize| {
            taps.push(Tap { name: name.into(), buf: buf.clone(), off, len, step: s.len() });
        };

        // The learned latents are a weight, copied into the working buffer once.
        e.copy_rows(&mut s, wb("latents"), 0, &lat_a, 0, nq, d);
        tap(&s, &mut taps, "latents_init", &lat_a, 0, nq * d);

        // proj_in: the single ArcFace token.
        e.linear(&mut s, &x_in, 0, wb("proj_in.weight"), Some(wb("proj_in.bias")), &x1, 0, 1, cfg.embedding_dim, d);
        tap(&s, &mut taps, "proj_in", &x1, 0, d);

        for l in 0..cfg.depth {
            let p = |n: &str| format!("layers.{l}.{n}");
            // kv input = cat(norm1(x), norm2(latents)) in ONE buffer: x at row 0,
            // the latents from row 1. `to_q` then slices the SAME buffer from
            // row 1, which is the reference's `to_q(norm2(latents))`.
            e.ln(&mut s, &x1, 0, wb(&p("0.norm1.weight")), wb(&p("0.norm1.bias")), &nkv, 0, 1, d);
            e.ln(&mut s, &lat_a, 0, wb(&p("0.norm2.weight")), wb(&p("0.norm2.bias")), &nkv, 1, nq, d);
            e.linear(&mut s, &nkv, 1, wb(&p("0.to_q.weight")), None, &q, 0, nq, d, inner);
            e.linear(&mut s, &nkv, 0, wb(&p("0.to_kv.weight")), None, &kv, 0, kvr, d, 2 * inner);
            e.cross_attn(&mut s, &q, 0, &kv, &scores, &probs, &ctx, cfg.heads, cfg.dim_head, nq, kvr);
            e.linear(&mut s, &ctx, 0, wb(&p("0.to_out.weight")), None, &aout, 0, nq, inner, d);
            // latents = attn(...) + latents
            s.push(gpu.step(k_add2, &[&lat_a, &aout, &lat_b], &[(nq * d) as u32], (nq * d) as u32));
            tap(&s, &mut taps, &format!("layer{l}_attn"), &lat_b, 0, nq * d);

            // Feed-forward: LayerNorm -> Linear -> GELU -> Linear, all bias-free
            // after the norm (the reference's nn.Sequential, hence the positional
            // 1.0 / 1.1 / 1.3 names).
            e.ln(&mut s, &lat_b, 0, wb(&p("1.0.weight")), wb(&p("1.0.bias")), &ffn, 0, nq, d);
            e.linear(&mut s, &ffn, 0, wb(&p("1.1.weight")), None, &fh, 0, nq, d, ff);
            s.push(gpu.step(k_gelu, &[&fh, &fg], &[(nq * ff) as u32], (nq * ff) as u32));
            e.linear(&mut s, &fg, 0, wb(&p("1.3.weight")), None, &fo, 0, nq, ff, d);
            // latents = ff(...) + latents
            s.push(gpu.step(k_add2, &[&lat_b, &fo, &lat_a], &[(nq * d) as u32], (nq * d) as u32));
            tap(&s, &mut taps, &format!("layer{l}_ff"), &lat_a, 0, nq * d);
        }

        e.linear(&mut s, &lat_a, 0, wb("proj_out.weight"), Some(wb("proj_out.bias")), &pout, 0, nq, d, od);
        tap(&s, &mut taps, "proj_out", &pout, 0, nq * od);
        e.ln(&mut s, &pout, 0, wb("norm_out.weight"), wb("norm_out.bias"), &out, 0, nq, od);
        tap(&s, &mut taps, "id_tokens", &out, 0, nq * od);

        Resampler { gpu, cfg, x_in, out, steps: s, taps, _weights: up }
    }

    /// Build on its own device handle.
    pub fn new(cfg: ResamplerConfig, w: &HashMap<String, Vec<f32>>) -> Resampler {
        let gpu = Gpu::new(KERNELS);
        Resampler::new_on(gpu, cfg, KERNELS, w)
    }

    /// Upload the ArcFace embedding. **Raw, not L2-normalised** — see
    /// `pulid::idcond` for why that distinction silently matters.
    pub fn set_embedding(&self, e: &[f32]) {
        assert_eq!(e.len(), self.cfg.embedding_dim, "instantid: ArcFace embedding width");
        self.gpu.write_f32(&self.x_in, e);
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    /// The projected ID tokens `[num_queries, output_dim]`.
    pub fn read_id_tokens(&self) -> Vec<f32> {
        self.gpu.read(&self.out, self.cfg.num_queries * self.cfg.output_dim)
    }

    pub fn tap_names(&self) -> Vec<String> {
        self.taps.iter().map(|t| t.name.clone()).collect()
    }

    /// Re-run the forward up to the step that produced `name`, then read it.
    /// Scratch buffers are reused across layers, so a stage is only readable
    /// immediately after its own step.
    pub fn read_tap(&self, name: &str) -> Vec<f32> {
        let t = self.taps.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("instantid: no tap `{name}`"));
        self.gpu.submit(&[], &self.steps[..t.step]);
        self.gpu.read(&t.buf, t.off + t.len)[t.off..].to_vec()
    }
}

// ---------------------------------------------------------------------------
// Decoupled cross-attention
// ---------------------------------------------------------------------------

/// One SDXL cross-attention site's **decoupled** ID branch.
///
/// Decoupled is the load-bearing word, and both wrong readings still run and
/// still produce a face:
///
/// * it does **not replace** the text cross-attention;
/// * the ID tokens are **not concatenated** onto the text tokens.
///
/// It is a SECOND attention over the same queries, with its own `k`/`v`
/// projections, whose result the caller adds with its own scale —
/// `hidden = text_attn + scale * ip_attn` (upstream
/// `attention_processor.py::IPAttnProcessor`). There is no `to_out` on this
/// branch: the shared one is applied to the sum afterwards, so [`SiteAttn::run`]
/// returns the pre-`to_out` context exactly as the reference's
/// `ip_hidden_states` does.
///
/// The scale is deliberately NOT applied here. It is per-call (and often
/// scheduled over sampling steps), while this object is per-site and built once.
pub struct SiteAttn {
    gpu: Gpu,
    k: RowKernels,
    cfg: crate::config::SiteConfig,
    num_queries: usize,
    kv_w: DeviceBuffer,
    id: DeviceBuffer,
    kv: DeviceBuffer,
    q: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    max_img: usize,
}

impl SiteAttn {
    /// `kv_w` is the FUSED `[2*hidden, token_dim]` weight from
    /// [`crate::import::validate_sites`] — `vstack(to_k_ip, to_v_ip)`, which is
    /// what makes one linear produce the `k | v` row layout
    /// `attn_apply_cross` reads (`kv_stride = 2*hidden`, `v_off = hidden`).
    pub fn new_on(
        gpu: Gpu,
        names: &[(&str, &str)],
        cfg: crate::config::SiteConfig,
        num_queries: usize,
        max_img: usize,
        kv_w: &[f32],
    ) -> SiteAttn {
        let k = RowKernels::resolve("instantid", names);
        let hidden = cfg.hidden;
        assert_eq!(kv_w.len(), 2 * hidden * cfg.token_dim, "instantid: fused kv weight size");
        assert_eq!(hidden % 64, 0, "instantid: site hidden {hidden} is not a multiple of head_dim 64");
        let w = gpu.storage(kv_w.len() as u64);
        gpu.write_f32(&w, kv_w);
        let heads = hidden / 64;
        SiteAttn {
            id: gpu.storage((num_queries * cfg.token_dim) as u64),
            kv: gpu.storage((num_queries * 2 * hidden) as u64),
            q: gpu.storage((max_img * hidden) as u64),
            scores: gpu.storage((heads * max_img * num_queries) as u64),
            probs: gpu.storage((heads * max_img * num_queries) as u64),
            ctx: gpu.storage((max_img * hidden) as u64),
            gpu,
            k,
            cfg,
            num_queries,
            kv_w: w,
            max_img,
        }
    }

    /// Upload the ID tokens. Once per identity, not once per step — the `k`/`v`
    /// projection depends only on them, so it is recorded here and reused.
    pub fn set_id(&self, id: &[f32]) {
        assert_eq!(id.len(), self.num_queries * self.cfg.token_dim, "instantid: id token slab");
        self.gpu.write_f32(&self.id, id);
    }

    /// `ip_out[n_img, hidden]` for the given image queries — the term the caller
    /// scales and adds to the text cross-attention's output.
    pub fn run(&self, q: &[f32], n_img: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden;
        assert!(n_img <= self.max_img, "instantid: {n_img} image rows > max_img {}", self.max_img);
        assert_eq!(q.len(), n_img * hidden, "instantid: query slab");
        self.gpu.write_f32(&self.q, q);
        let e = RowEmit::new(&self.gpu, self.k, EPS);
        let mut s = Vec::new();
        // One linear over the FUSED weight produces `[nq, 2*hidden]` laid out as
        // k | v per row — the layout the cross-attention kernels expect.
        e.linear(&mut s, &self.id, 0, &self.kv_w, None, &self.kv, 0, self.num_queries, self.cfg.token_dim, 2 * hidden);
        e.cross_attn(&mut s, &self.q, 0, &self.kv, &self.scores, &self.probs, &self.ctx, hidden / 64, 64, n_img, self.num_queries);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.ctx, n_img * hidden)
    }

    /// The projected `k` and `v` for the uploaded ID tokens, de-interleaved —
    /// the parity ladder's view into the fused buffer.
    pub fn read_kv(&self) -> (Vec<f32>, Vec<f32>) {
        let hidden = self.cfg.hidden;
        let fused = self.gpu.read(&self.kv, self.num_queries * 2 * hidden);
        let mut k = Vec::with_capacity(self.num_queries * hidden);
        let mut v = Vec::with_capacity(self.num_queries * hidden);
        for r in fused.chunks(2 * hidden) {
            k.extend_from_slice(&r[..hidden]);
            v.extend_from_slice(&r[hidden..]);
        }
        (k, v)
    }
}

// ---------------------------------------------------------------------------
// The adapter: all 70 sites, as one CrossAttnInject
// ---------------------------------------------------------------------------

/// Every decoupled site, ready to be injected into a backbone's cross-attention.
///
/// Implements [`model::attninject::CrossAttnInject`], so `crates/unet` consumes
/// it without knowing what an identity is — the same way `crates/controlnet`'s
/// residuals are consumed without the backbone knowing what a control image is.
///
/// # The scale is folded into `v`, and that is exact
///
/// The reference applies a per-call `scale` to the ID branch's output. A
/// recorded graph cannot take a per-call scalar in a kernel's `Params` (those
/// are baked when the step is recorded), and re-recording 70 sites per sampling
/// step to change one number would be absurd.
///
/// It does not need to: `ip_out = softmax(q·kᵀ/√d)·V` is **linear in `V`**, so
/// scaling `V` by `s` scales the output by exactly `s`. [`SiteAttnSet::set_scale`]
/// therefore re-uploads the `v` half of each fused weight scaled — one upload per
/// site, only when the scale changes, and bit-exact rather than approximate.
/// Scaling `k` instead would go through the softmax and be a different function
/// entirely.
pub struct SiteAttnSet {
    gpu: Gpu,
    k: RowKernels,
    k_axpy: usize,
    sites: Vec<crate::config::SiteConfig>,
    /// Unscaled `[2*hidden, token_dim]` per site, kept so `set_scale` can be
    /// re-applied from the original rather than compounding.
    base: Vec<Vec<f32>>,
    kv_w: Vec<DeviceBuffer>,
    id: DeviceBuffer,
    /// Per-site scratch: the projected kv, the score/prob slabs and the context.
    scratch: Vec<SiteScratch>,
    num_queries: usize,
    /// Atomic because the trait is `Send + Sync` (a backbone may be shared) and
    /// `set_scale` takes `&self` — it writes device buffers, which is already a
    /// shared-reference operation.
    scale: std::sync::atomic::AtomicU32,
}

struct SiteScratch {
    kv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
}

impl SiteAttnSet {
    /// `w` is the fused per-site weight map from [`crate::import::validate_sites`].
    /// `max_t` bounds the query rows any site will see (the largest latent
    /// resolution in the backbone).
    pub fn new_on(
        gpu: Gpu,
        names: &[(&str, &str)],
        w: &crate::import::SiteWeights,
        num_queries: usize,
        max_t: usize,
    ) -> SiteAttnSet {
        let k = RowKernels::resolve("instantid", names);
        let k_axpy = idx(names, "axpy");
        let mut base = Vec::with_capacity(w.cfg.len());
        let mut kv_w = Vec::with_capacity(w.cfg.len());
        let mut scratch = Vec::with_capacity(w.cfg.len());
        for s in &w.cfg {
            assert_eq!(s.hidden % 64, 0, "instantid: site hidden {} is not a multiple of head_dim 64", s.hidden);
            let fused = w.kv.get(&s.index).unwrap_or_else(|| panic!("instantid: no fused kv for site {}", s.index));
            let b = gpu.storage(fused.len() as u64);
            gpu.write_f32(&b, fused);
            let heads = s.hidden / 64;
            scratch.push(SiteScratch {
                kv: gpu.storage((num_queries * 2 * s.hidden) as u64),
                scores: gpu.storage((heads * max_t * num_queries) as u64),
                probs: gpu.storage((heads * max_t * num_queries) as u64),
                ctx: gpu.storage((max_t * s.hidden) as u64),
            });
            base.push(fused.clone());
            kv_w.push(b);
        }
        let token_dim = w.cfg[0].token_dim;
        SiteAttnSet {
            id: gpu.storage((num_queries * token_dim) as u64),
            gpu,
            k,
            k_axpy,
            sites: w.cfg.clone(),
            base,
            kv_w,
            scratch,
            num_queries,
            scale: std::sync::atomic::AtomicU32::new(1.0f32.to_bits()),
        }
    }

    /// Upload the ID tokens produced by [`Resampler`]. Once per identity.
    pub fn set_id(&self, id: &[f32]) {
        assert_eq!(id.len(), self.num_queries * self.sites[0].token_dim, "instantid: id token slab");
        self.gpu.write_f32(&self.id, id);
    }

    /// Set the ID strength. Folded into each site's `v` rows — exact, see the
    /// type docs. Re-applied from the unscaled weights, so calling it twice does
    /// not compound.
    pub fn set_scale(&self, s: f32) {
        for (i, site) in self.sites.iter().enumerate() {
            let half = site.hidden * site.token_dim;
            let mut w = self.base[i].clone();
            for x in &mut w[half..] {
                *x *= s;
            }
            self.gpu.write_f32(&self.kv_w[i], &w);
        }
        self.scale.store(s.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// The kernels [`SiteAttnSet`] dispatches on the BACKBONE's device, beyond what
/// a UNet already registers: the row-emitter set plus `axpy` for the add.
pub const INJECT_EXTRA: &[(&str, &str)] = &[("axpy", kernels::AXPY)];

impl model::attninject::CrossAttnInject for SiteAttnSet {
    fn kernels(&self) -> &'static [(&'static str, &'static str)] {
        INJECT_EXTRA
    }

    fn sites(&self) -> usize {
        self.sites.len()
    }

    fn inject(&self, steps: &mut Vec<Step>, gpu: &Gpu, k: usize, q: &DeviceBuffer, ctx: &DeviceBuffer, t: u32, c: u32) {
        let site = &self.sites[k];
        assert_eq!(
            site.hidden as u32, c,
            "instantid: site {k} is {} wide but the backbone's cross-attention is {c}",
            site.hidden
        );
        let sc = &self.scratch[k];
        let e = RowEmit::new(gpu, self.k, EPS);
        let (nq, hidden) = (self.num_queries, site.hidden);
        // One linear over the FUSED weight yields `[nq, 2*hidden]` = k | v per
        // row, which is the layout the cross-attention kernels read.
        e.linear(steps, &self.id, 0, &self.kv_w[k], None, &sc.kv, 0, nq, site.token_dim, 2 * hidden);
        e.cross_attn(steps, q, 0, &sc.kv, &sc.scores, &sc.probs, &sc.ctx, hidden / 64, 64, t as usize, nq);
        // ctx += ip_ctx. The scale is already inside `v`, so this is a plain add
        // (`axpy` with s = 1) rather than a second scaling.
        steps.push(gpu.step(self.k_axpy, &[ctx, &sc.ctx], &[t * c, 1.0f32.to_bits()], t * c));
    }
}
