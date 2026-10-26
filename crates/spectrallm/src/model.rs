// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements efficient neural-network architectures and
// the evidence that they work for its clients. If your team needs expertise
// in structured and spectral model operators, you can procure our services
// by sending an email to info@swedishembedded.com.

//! SpectralLM's training and full-sequence execution graph.
//!
//! Per pre-norm block (no biases), exactly Qwen3's except the mixer:
//!
//! ```text
//! h = RMSNorm(x)·ln1
//! q, k, v = h Wqᵀ, h Wkᵀ, h Wvᵀ                 (GQA)
//! q = RoPE(RMSNorm_hd(q)·q_norm),  k = RoPE(RMSNorm_hd(k)·k_norm)
//! x += Wo · attention(q, k, v)
//! x += Mixer(RMSNorm(x)·ln2)                    (crate::mixer)
//! logits = E · RMSNorm(x)·norm                  (E tied to the embedding by default)
//! ```
//!
//! Boundary residuals are kept per layer; one set of per-layer scratch is
//! shared by every layer, and the backward pass rebuilds a layer from its
//! input boundary before differentiating it. Both mixers pay that recompute,
//! so it never favours one of them in a comparison.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, FlashGqaIds, Gqa, KernelIds};
use model::saved::{Residuals, SavedFormat, SavedPolicy};
use model::streamed::StreamedStore;
use model::tape::LayerTape;
use paramstore::ParamStore;

use crate::config::SpectralConfig;
use crate::decode::Decode;
use crate::kernels::*;
use crate::mixer::{Basis, MixerBufs, Transform};

/// Cross-entropy ignore index (unsupervised positions).
pub const IGNORE: u32 = 0xFFFF_FFFF;

/// Per-layer forward scratch, reused by every layer.
struct Scratch {
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    k_pre: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    /// Flash attention's per-(row, head) log-sum-exp; one element otherwise.
    lse: DeviceBuffer,
    proj: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    mlp_out: DeviceBuffer,
    mixer: MixerBufs,
}

/// Backward scratch (training builds only).
struct Grad {
    dres: Vec<DeviceBuffer>,
    dlogits: DeviceBuffer,
    d_xf: DeviceBuffer,
    rms_inv: DeviceBuffer,
    d_xn2: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_xmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    /// Flash attention backward's per-(row, head) `rowsum(dO * O)`.
    dsum: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    d_q_pre: DeviceBuffer,
    d_k_pre: DeviceBuffer,
    d_xn1: DeviceBuffer,
}

pub struct SpectralLm {
    pub gpu: Gpu,
    pub cfg: SpectralConfig,
    pub ps: ParamStore,
    pub(crate) ids: KernelIds,
    pub(crate) basis: Option<Basis>,
    flash: Option<FlashGqaIds>,
    /// Per-row softmax `(max, sum)` of the logits, written with the loss.
    ce_stats: DeviceBuffer,
    b: u32,
    t: u32,
    train: bool,
    count: Cell<f32>,
    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    res: Residuals,
    xf: DeviceBuffer,
    logits: DeviceBuffer,
    ce_rows: DeviceBuffer,
    scratch: Scratch,
    grad: Option<Grad>,
    /// Embedding, the layers (one segment each) and the final norm.
    trunk: LayerTape,
    head: Vec<Step>,
    loss_steps: Vec<Step>,
    /// The head, the layers from the last down (one segment each), the embedding.
    bwd_steps: LayerTape,
    /// The per-layer tensors held in a device window of layers, when the run streams them
    /// ([`model::Model::set_resident`]); `None` keeps every layer in the store.
    streamed: Option<StreamedStore>,
    /// A streamed build runs its backward inside the update: set by `backward`, taken by `adamw_step`.
    backward_pending: Cell<bool>,
    ce_grad_uniform: DeviceBuffer,
    opt: Option<optim::Optim>,
    pub(crate) dec: RefCell<Option<Decode>>,
}

impl SpectralLm {
    /// Frozen weights from any tensor source (a checkpoint or an init map).
    pub fn new_inference(cfg: SpectralConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> SpectralLm {
        Self::build(Gpu::new(&optim::with_moment_kernels(PIPELINES)), cfg, b, t, src, false)
    }

    /// Trainable: gradients, AdamW moments and the backward graph.
    pub fn new_train(cfg: SpectralConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> SpectralLm {
        Self::build(Gpu::new(&optim::with_moment_kernels(PIPELINES)), cfg, b, t, src, true)
    }

    /// [`Self::new_train`]/[`Self::new_inference`] on a caller's device
    /// (registered with [`PIPELINES`]), so a test or a process holds one.
    pub fn new_on(gpu: Gpu, cfg: SpectralConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool) -> SpectralLm {
        Self::build(gpu, cfg, b, t, src, train)
    }

    /// Load a checkpoint written by [`model::Model::save`], sized for `b x t`.
    pub fn load(path: &str, b: u32, t: u32) -> Result<SpectralLm, String> {
        let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let cfg = SpectralConfig::from_json(&reader.config())?;
        Ok(Self::new_inference(cfg, b, t, &reader))
    }

    fn build(gpu: Gpu, cfg: SpectralConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool) -> SpectralLm {
        cfg.validate().expect("spectrallm config");
        assert!(b > 0 && t > 0 && t <= cfg.block_size, "spectrallm: sequence {t} must be in 1..={}", cfg.block_size);
        let role = if train { paramstore::Role::Trainable } else { paramstore::Role::Frozen };
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, role)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let opt = train.then(|| optim::Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF));
        let basis = Basis::new(&gpu, &cfg);

        let n = (b * t) as u64;
        let (d, v) = (cfg.d_model as u64, cfg.vocab as u64);
        let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        let flash = flash_ids(&gpu, cfg.head_dim, t);
        // The score matrix exists only on the materialized path.
        let bht2 = if flash.is_some() { 1 } else { b as u64 * cfg.n_heads as u64 * t as u64 * t as u64 };
        assert!(bht2 <= u32::MAX as u64, "materialized attention exceeds u32 index space");
        let stat = if flash.is_some() { b as u64 * cfg.n_heads as u64 * t as u64 } else { 1 };
        let st = |x: u64| gpu.storage(x);
        let scratch = Scratch {
            xn1: st(n * d),
            q_pre: st(n * hq),
            k_pre: st(n * hkv),
            q: st(n * hq),
            k: st(n * hkv),
            v: st(n * hkv),
            scores: st(bht2),
            probs: st(bht2),
            ctx: st(n * hq),
            lse: st(stat),
            proj: st(n * d),
            xmid: st(n * d),
            xn2: st(n * d),
            mlp_out: st(n * d),
            mixer: MixerBufs::new(&gpu, &cfg, b * t, train),
        };
        let grad = train.then(|| Grad {
            dres: (0..=cfg.n_layers).map(|_| st(n * d)).collect(),
            dlogits: st(n * v),
            d_xf: st(n * d),
            rms_inv: st(n * (cfg.n_heads as u64).max(1)),
            d_xn2: st(n * d),
            d_tmp: st(n * d),
            d_xmid: st(n * d),
            d_ctx: st(n * hq),
            d_scores: st(bht2),
            dsum: st(stat),
            d_q: st(n * hq),
            d_k: st(n * hkv),
            d_v: st(n * hkv),
            d_q_pre: st(n * hq),
            d_k_pre: st(n * hkv),
            d_xn1: st(n * d),
        });
        let mut m = SpectralLm {
            tokens: gpu.buffer("tokens", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST),
            targets: gpu.buffer("targets", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST),
            res: Residuals::new((0..=cfg.n_layers).map(|_| st(n * d)).collect(), 0..cfg.n_layers as usize + 1),
            xf: st(n * d),
            logits: st(n * v),
            ce_rows: st(n),
            ce_grad_uniform: gpu.uniform_dynamic(4),
            ids: block_ids(),
            basis,
            flash,
            ce_stats: st(2 * n),
            scratch,
            grad,
            opt,
            ps,
            cfg,
            b,
            t,
            train,
            count: Cell::new(1.0),
            trunk: LayerTape::default(),
            head: Vec::new(),
            loss_steps: Vec::new(),
            bwd_steps: LayerTape::default(),
            streamed: None,
            backward_pending: Cell::new(false),
            dec: RefCell::new(None),
            gpu,
        };
        m.trunk = m.trunk_steps();
        m.head = m.head_steps();
        m.loss_steps = m.loss_steps();
        if train {
            m.bwd_steps = m.backward_steps();
        }
        m
    }

    pub fn batch(&self) -> u32 {
        self.b
    }
    pub fn seq(&self) -> u32 {
        self.t
    }
    fn rows(&self) -> u32 {
        self.b * self.t
    }
    /// The transform path this instance resolved to (`None` for SwiGLU).
    pub fn transform(&self) -> Option<Transform> {
        self.basis.as_ref().map(|b| b.path)
    }

    pub(crate) fn head_weight(&self) -> &DeviceBuffer {
        self.ps.w(if self.cfg.tie_embeddings { "tok.weight" } else { "lm_head.weight" })
    }
    fn head_grad(&self) -> &DeviceBuffer {
        self.ps.g(if self.cfg.tie_embeddings { "tok.weight" } else { "lm_head.weight" })
    }

    /// `y[rows, out] = x[rows, k] · w[out, k]ᵀ`, tiled where the shape fills tiles.
    pub(crate) fn linear(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer, rows: u32, k: u32, out: u32) {
        let (kind, grid) = block::pick_gemm(rows as usize, out as usize, MATMUL, MATMUL_REG3, false);
        s.push(self.gpu.dispatch(kind, &[x, w, y], &[rows, k, out], grid));
    }

    /// `dx[rows, k] (+)= dy[rows, out] · w[out, k]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_dx(&self, s: &mut Vec<Step>, dy: &DeviceBuffer, w: &DeviceBuffer, dx: &DeviceBuffer, rows: u32, k: u32, out: u32, acc: bool) {
        let (kind, grid) = block::pick_gemm(rows as usize, k as usize, MATMUL_DX, MATMUL_DX_REG, false);
        s.push(self.gpu.dispatch(kind, &[dy, w, dx], &[rows, k, out, acc as u32], grid));
    }

    /// `dw[out, k] += dy[rows, out]ᵀ · x[rows, k]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_dw(&self, s: &mut Vec<Step>, dy: &DeviceBuffer, x: &DeviceBuffer, dw: &DeviceBuffer, rows: u32, k: u32, out: u32) {
        let (kind, grid) = block::pick_gemm(out as usize, k as usize, MATMUL_DW, MATMUL_DW_REG, false);
        s.push(self.gpu.dispatch(kind, &[dy, x, dw], &[rows, k, out], grid));
    }

    /// One block's forward from `h_in` to `h_out`, filling the shared scratch.
    fn layer_fwd(&self, s: &mut Vec<Step>, l: usize, h_in: &DeviceBuffer, h_out: &DeviceBuffer) {
        let cfg = &self.cfg;
        let (ids, n) = (&self.ids, self.rows());
        let (d, eps) = (cfg.d_model, cfg.rms_eps);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let sc = &self.scratch;
        let w = |name: &str| self.ps.w(&format!("blocks.{l}.{name}"));
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        s.push(block::rmsnorm_fwd(&self.gpu, ids, h_in, w("ln1.weight"), &sc.xn1, d, n, eps));
        self.linear(s, &sc.xn1, w("attn.wq.weight"), &sc.q_pre, n, d, hq);
        self.linear(s, &sc.xn1, w("attn.wk.weight"), &sc.k_pre, n, d, hkv);
        self.linear(s, &sc.xn1, w("attn.wv.weight"), &sc.v, n, d, hkv);
        s.push(block::rmsnorm_fwd(&self.gpu, ids, &sc.q_pre, w("attn.q_norm.weight"), &sc.q, hd, n * nh, eps));
        s.push(block::rmsnorm_fwd(&self.gpu, ids, &sc.k_pre, w("attn.k_norm.weight"), &sc.k, hd, n * nkv, eps));
        s.push(block::rope_fwd(&self.gpu, ids, &sc.q, n, nh, hd, hq, self.t, cfg.rope_theta));
        s.push(block::rope_fwd(&self.gpu, ids, &sc.k, n, nkv, hd, hkv, self.t, cfg.rope_theta));
        match &self.flash {
            Some(fl) => s.push(block::flash_gqa_causal_fwd(&self.gpu, fl, &ga, &sc.q, &sc.k, &sc.v, &sc.ctx, &sc.lse)),
            None => s.extend(block::gqa_fwd(&self.gpu, ids, &ga, &sc.q, &sc.k, &sc.v, &sc.scores, &sc.probs, &sc.ctx)),
        }
        self.linear(s, &sc.ctx, w("attn.wo.weight"), &sc.proj, n, hq, d);
        s.push(self.gpu.step(ADD2, &[h_in, &sc.proj, &sc.xmid], &[n * d], n * d));
        s.push(block::rmsnorm_fwd(&self.gpu, ids, &sc.xmid, w("ln2.weight"), &sc.xn2, d, n, eps));
        self.mixer_fwd(s, l, &sc.xn2, &sc.mlp_out, &sc.mixer, n);
        s.push(self.gpu.step(ADD2, &[&sc.xmid, &sc.mlp_out, h_out], &[n * d], n * d));
    }

    /// Backward of [`Self::layer_fwd`]; the scratch must hold that layer's
    /// forward. Writes the input-boundary gradient `d_in`.
    fn layer_bwd(&self, s: &mut Vec<Step>, l: usize, h_in: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let cfg = &self.cfg;
        let (ids, n) = (&self.ids, self.rows());
        let (d, eps) = (cfg.d_model, cfg.rms_eps);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let sc = &self.scratch;
        let g = self.grad.as_ref().expect("backward on an inference build");
        let w = |name: &str| self.ps.w(&format!("blocks.{l}.{name}"));
        let gr = |name: &str| self.ps.g(&format!("blocks.{l}.{name}"));
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        // h_out = xmid + Mixer(RMSNorm(xmid))
        self.mixer_bwd(s, l, &sc.xn2, d_out, &g.d_xn2, &sc.mixer, n);
        s.extend(block::rmsnorm_bwd(&self.gpu, ids, &sc.xmid, w("ln2.weight"), &g.d_xn2, &g.d_tmp, &g.rms_inv, Some(gr("ln2.weight")), d, n, eps));
        s.push(self.gpu.step(ADD2, &[d_out, &g.d_tmp, &g.d_xmid], &[n * d], n * d));
        // xmid = h_in + Wo ctx
        self.linear_dw(s, &g.d_xmid, &sc.ctx, gr("attn.wo.weight"), n, hq, d);
        self.linear_dx(s, &g.d_xmid, w("attn.wo.weight"), &g.d_ctx, n, hq, d, false);
        match &self.flash {
            Some(fl) => s.extend(block::flash_gqa_causal_bwd(&self.gpu, fl, &ga, &sc.q, &sc.k, &sc.v, &sc.ctx, &sc.lse, &g.d_ctx, &g.dsum, &g.d_q, &g.d_k, &g.d_v)),
            None => s.extend(block::gqa_bwd(&self.gpu, ids, &ga, &sc.q, &sc.k, &sc.v, &sc.probs, &g.d_ctx, &g.d_scores, &g.d_q, &g.d_k, &g.d_v)),
        }
        s.push(block::rope_bwd(&self.gpu, ids, &g.d_q, n, nh, hd, hq, self.t, cfg.rope_theta));
        s.push(block::rope_bwd(&self.gpu, ids, &g.d_k, n, nkv, hd, hkv, self.t, cfg.rope_theta));
        s.extend(block::rmsnorm_bwd(&self.gpu, ids, &sc.q_pre, w("attn.q_norm.weight"), &g.d_q, &g.d_q_pre, &g.rms_inv, Some(gr("attn.q_norm.weight")), hd, n * nh, eps));
        s.extend(block::rmsnorm_bwd(&self.gpu, ids, &sc.k_pre, w("attn.k_norm.weight"), &g.d_k, &g.d_k_pre, &g.rms_inv, Some(gr("attn.k_norm.weight")), hd, n * nkv, eps));
        self.linear_dw(s, &g.d_q_pre, &sc.xn1, gr("attn.wq.weight"), n, d, hq);
        self.linear_dx(s, &g.d_q_pre, w("attn.wq.weight"), &g.d_xn1, n, d, hq, false);
        self.linear_dw(s, &g.d_k_pre, &sc.xn1, gr("attn.wk.weight"), n, d, hkv);
        self.linear_dx(s, &g.d_k_pre, w("attn.wk.weight"), &g.d_xn1, n, d, hkv, true);
        self.linear_dw(s, &g.d_v, &sc.xn1, gr("attn.wv.weight"), n, d, hkv);
        self.linear_dx(s, &g.d_v, w("attn.wv.weight"), &g.d_xn1, n, d, hkv, true);
        s.extend(block::rmsnorm_bwd(&self.gpu, ids, h_in, w("ln1.weight"), &g.d_xn1, &g.d_tmp, &g.rms_inv, Some(gr("ln1.weight")), d, n, eps));
        s.push(self.gpu.step(ADD2, &[&g.d_xmid, &g.d_tmp, d_in], &[n * d], n * d));
    }

    fn trunk_steps(&self) -> LayerTape {
        let (n, d) = (self.rows(), self.cfg.d_model);
        let mut tape = LayerTape::default();
        let mut s = vec![self.gpu.step(EMBED, &[&self.tokens, self.ps.w("tok.weight"), &self.res[0]], &[d, n], n * d)];
        self.res.round_trip(&self.gpu, &mut s, 0);
        tape.segment(&mut s, None);
        for l in 0..self.cfg.n_layers as usize {
            self.layer_fwd(&mut s, l, &self.res[l], &self.res[l + 1]);
            self.res.round_trip(&self.gpu, &mut s, l + 1);
            tape.segment(&mut s, Some(l));
        }
        let last = &self.res[self.cfg.n_layers as usize];
        s.push(block::rmsnorm_fwd(&self.gpu, &self.ids, last, self.ps.w("norm.weight"), &self.xf, d, n, self.cfg.rms_eps));
        tape.segment(&mut s, None);
        tape
    }

    /// Per-position cross-entropy and the per-row softmax statistics the
    /// backward reads: one cooperative workgroup per row where the device runs
    /// workgroup reductions, the per-row reference kernels otherwise.
    fn loss_steps(&self) -> Vec<Step> {
        let (n, v) = (self.rows(), self.cfg.vocab);
        let caps = self.gpu.caps();
        if caps.workgroup_reductions && caps.max_workgroup_size >= 256 {
            vec![self.gpu.dispatch(CE_VALUE_STATS_ROWS, &[&self.logits, &self.targets, &self.ce_rows, &self.ce_stats], &[n, v, IGNORE], gpu_core::Dispatch::Workgroups(n))]
        } else {
            vec![
                self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_rows], &[n, v, IGNORE], n),
                self.gpu.step(CE_STATS, &[&self.logits, &self.targets, &self.ce_stats], &[n, v, IGNORE], n),
            ]
        }
    }

    fn head_steps(&self) -> Vec<Step> {
        let mut s = Vec::new();
        self.linear(&mut s, &self.xf, self.head_weight(), &self.logits, self.rows(), self.cfg.d_model, self.cfg.vocab);
        s
    }

    /// The whole backward: the head, then every layer rebuilt and
    /// differentiated in reverse, then the embedding.
    fn backward_steps(&self) -> LayerTape {
        let (n, d, v) = (self.rows(), self.cfg.d_model, self.cfg.vocab);
        let g = self.grad.as_ref().expect("backward on an inference build");
        let layers = self.cfg.n_layers as usize;
        let mut tape = LayerTape::default();
        // The loss steps wrote the per-row softmax statistics this reads.
        let mut s = vec![self.gpu.step_buf(CE_GRAD_STATS, &self.ce_grad_uniform, &[&self.logits, &self.targets, &self.ce_stats, &g.dlogits], n * v)];
        self.linear_dw(&mut s, &g.dlogits, &self.xf, self.head_grad(), n, d, v);
        self.linear_dx(&mut s, &g.dlogits, self.head_weight(), &g.d_xf, n, d, v, false);
        self.res.restore(&self.gpu, &mut s, layers);
        s.extend(block::rmsnorm_bwd(&self.gpu, &self.ids, &self.res[layers], self.ps.w("norm.weight"), &g.d_xf, &g.dres[layers], &g.rms_inv, Some(self.ps.g("norm.weight")), d, n, self.cfg.rms_eps));
        tape.segment(&mut s, None);
        for l in (0..layers).rev() {
            self.res.restore(&self.gpu, &mut s, l);
            self.layer_fwd(&mut s, l, &self.res[l], &self.res[l + 1]);
            self.layer_bwd(&mut s, l, &self.res[l], &g.dres[l + 1], &g.dres[l]);
            tape.segment(&mut s, Some(l));
        }
        s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &g.dres[0], self.ps.g("tok.weight")], &[n, d, v], v * d));
        tape.segment(&mut s, None);
        tape
    }

    /// Submit the trunk; a layer-streamed build loads each layer into its window slot as it is reached.
    fn submit_trunk(&self) {
        match &self.streamed {
            Some(stream) => stream.forward(&self.gpu, &self.trunk).unwrap_or_else(|e| panic!("spectrallm layer-streamed forward: {e}")),
            None => self.trunk.submit(&self.gpu),
        }
    }

    pub(crate) fn is_streamed(&self) -> bool {
        self.streamed.is_some()
    }

    fn holds_streamed(&self, name: &str) -> bool {
        self.streamed.as_ref().is_some_and(|s| s.holds(name))
    }

    /// Stream every per-layer tensor through a device window under `cfg` (see [`model::streamed`]): the layers' f32
    /// values and optimizer state move to the host, the device keeps the window's slots. The embedding (the tied
    /// head) and the final norm stay in the store under its own optimizer.
    pub fn set_resident(&mut self, cfg: &model::resident_ft::ResidentFtConfig) -> Result<(), String> {
        if !self.train {
            return Err("spectrallm: an inference build has nothing to stream".into());
        }
        if self.streamed.is_some() {
            return Err("spectrallm: the layers already stream".into());
        }
        let stream = StreamedStore::convert(&self.gpu, &mut self.ps, cfg.clone(), &model::streamed::blocks_layer).map_err(|e| format!("spectrallm: {e}"))?;
        self.opt.as_ref().expect("a training build has an optimizer").invalidate();
        self.streamed = Some(stream);
        // The recorded passes bound the store's buffers as they were: record them again over the window's slots.
        self.trunk = self.trunk_steps();
        self.head = self.head_steps();
        self.bwd_steps = self.backward_steps();
        Ok(())
    }

    pub fn set_tokens(&self, tokens: &[u32]) {
        assert_eq!(tokens.len(), self.rows() as usize, "spectrallm: tokens must be b*t");
        self.gpu.write(&self.tokens, tokens);
    }

    /// Upload a causal-LM batch (`targets` hold the next token, [`IGNORE`]
    /// masks a position).
    pub fn set_tokens_targets(&self, tokens: &[u32], targets: &[u32]) {
        assert_eq!(targets.len(), self.rows() as usize, "spectrallm: targets must be b*t");
        self.set_tokens(tokens);
        self.gpu.write(&self.targets, targets);
        self.count.set(targets.iter().filter(|t| **t != IGNORE).count().max(1) as f32);
    }

    /// Full-sequence logits `[b*t, vocab]` for the uploaded tokens.
    pub fn forward_logits(&self) -> Vec<f32> {
        self.submit_trunk();
        self.gpu.submit(&[], &self.head);
        self.gpu.read(&self.logits, self.rows() as usize * self.cfg.vocab as usize)
    }

    /// Per-position next-token negative log-likelihood (nats) of the
    /// uploaded batch, 0 at [`IGNORE`]d positions.
    pub fn token_nll(&self) -> Vec<f32> {
        self.submit_trunk();
        self.gpu.submit(&[], &self.head);
        self.gpu.submit(&[], &self.loss_steps);
        self.gpu.read(&self.ce_rows, self.rows() as usize)
    }

    /// `[b*t, d_model]` residual stream entering layer `layer` (`n_layers`
    /// is the pre-norm output) after the last forward.
    pub fn read_res(&self, layer: usize) -> Vec<f32> {
        assert!(self.res.format() == SavedFormat::F32, "spectrallm: narrow boundary residuals keep no per-layer stream to read");
        self.gpu.read(&self.res[layer], self.rows() as usize * self.cfg.d_model as usize)
    }

    /// Per-kernel-kind device profiles of the uploaded batch's forward (trunk,
    /// head and loss) and, on a training build, backward pass.
    pub fn profile(&self, reps: usize) -> (gpu_core::profile::PassProfile, Option<gpu_core::profile::PassProfile>) {
        let fwd: Vec<Step> = self.trunk.steps().iter().chain(&self.head).chain(&self.loss_steps).cloned().collect();
        let f = gpu_core::profile::profile(&self.gpu, "spectrallm forward", &fwd, reps);
        let b = self.train.then(|| {
            let (n, v) = (self.rows(), self.cfg.vocab);
            self.gpu.write(&self.ce_grad_uniform, &[n, v, IGNORE, gpu_core::f(self.count.get())]);
            gpu_core::profile::profile(&self.gpu, "spectrallm backward", self.bwd_steps.steps(), reps)
        });
        (f, b)
    }

    /// Forward and backward cost of one step, from the kernels' formulas.
    pub fn cost(&self) -> gpu_core::cost::CostReport {
        let mut report = gpu_core::cost::CostReport::default();
        let names: Vec<String> = PIPELINES.iter().map(|(n, _)| n.to_string()).collect();
        for steps in [self.trunk.steps(), &self.head, &self.loss_steps, self.bwd_steps.steps()] {
            gpu_core::cost::tally(&mut report, &names, steps);
        }
        report
    }
}

impl model::Model for SpectralLm {
    fn optimizer_state(&mut self) -> Option<model::OptimizerState<'_>> {
        Some(model::OptimizerState { gpu: &self.gpu, store: &mut self.ps, optim: self.opt.as_ref()? })
    }

    fn optimizer_parts(&self) -> Option<model::OptimizerParts<'_>> {
        Some(model::OptimizerParts { gpu: &self.gpu, store: &self.ps, optim: self.opt.as_ref()? })
    }

    /// SpectralLM keeps only the residual between layers and rebuilds each layer before its backward (one scratch set
    /// serves every layer): it runs recompute, with the residuals in f32 or, once narrowed, in bf16 or FP8
    /// ([`Residuals`]). Keeping every layer is refused by name.
    fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        let current = SavedPolicy::recompute_with(self.res.format());
        if !current.narrows_to(policy) {
            return Err(format!("spectrallm recomputes each layer from its {:?} residual in the backward; {policy:?} is not available", self.res.format()));
        }
        if let SavedPolicy::Quantized(format) = policy {
            if policy != current {
                self.res.quantize(&self.gpu, format, self.rows(), self.cfg.d_model).map_err(|e| format!("spectrallm: {e}"))?;
                self.trunk = self.trunk_steps();
                if self.train {
                    self.bwd_steps = self.backward_steps();
                }
            }
        }
        Ok(())
    }

    type Config = SpectralConfig;

    fn new(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        SpectralLm::new_train(cfg, b, t, init)
    }

    fn new_inference(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        SpectralLm::new_inference(cfg, b, t, init)
    }

    fn init_weights(cfg: &Self::Config, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => self.set_tokens_targets(tokens, targets),
            other => panic!("spectrallm: unsupported batch shape {:?}", std::mem::discriminant(&other)),
        }
    }

    /// Mean next-token cross-entropy over the supervised positions.
    fn forward(&self) -> f32 {
        let nll = self.token_nll();
        (nll.iter().map(|x| *x as f64).sum::<f64>() / self.count.get() as f64) as f32
    }

    fn backward(&self) {
        assert!(self.train, "backward on an inference build");
        let (n, v) = (self.rows(), self.cfg.vocab);
        self.gpu.write(&self.ce_grad_uniform, &[n, v, IGNORE, gpu_core::f(self.count.get())]);
        if self.streamed.as_ref().is_some_and(StreamedStore::is_layer_streamed) {
            // The weights of a layer-streamed build are not all on the device: the backward runs layer by layer in
            // `adamw_step`, where the layers it loads are the ones it updates.
            assert!(!self.backward_pending.replace(true), "spectrallm layer-streamed full fine-tune: a second backward before the update (gradient accumulation) is not supported; every micro-batch is its own step");
            return;
        }
        self.bwd_steps.submit(&self.gpu);
    }

    fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        let opt = self.opt.as_ref().expect("adamw_step on an inference build");
        match &self.streamed {
            Some(stream) => {
                if stream.is_layer_streamed() {
                    assert!(self.backward_pending.replace(false), "spectrallm layer-streamed full fine-tune: adamw_step without a backward");
                }
                let args = model::resident_ft::StepArgs { step: t, lr, wd, adam, clip, extra_scale };
                stream.adamw_step(&self.gpu, &self.ps, opt, &self.bwd_steps, &args).unwrap_or_else(|e| panic!("spectrallm layer-streamed update: {e}"));
            }
            None => opt.step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale),
        }
    }

    fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    fn param_names(&self) -> Vec<String> {
        self.cfg.param_list().into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        match &self.streamed {
            Some(stream) if stream.holds(name) => stream.read_weight(&self.gpu, name).unwrap_or_else(|e| panic!("spectrallm: reading {name}: {e}")),
            _ => self.ps.read_weight(&self.gpu, name),
        }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        match &self.streamed {
            Some(stream) if stream.holds(name) => stream.write_weight(&self.gpu, name, data).unwrap_or_else(|e| panic!("spectrallm: writing {name}: {e}")),
            _ => self.gpu.write_f32(self.ps.w(name), data),
        }
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        assert!(!self.holds_streamed(name), "spectrallm: {name} streams through a layer window; no gradient outlives its layer");
        self.ps.read_grad(&self.gpu, name)
    }

    fn set_resident(&mut self, cfg: &model::resident_ft::ResidentFtConfig) -> Result<(), String> {
        SpectralLm::set_resident(self, cfg)
    }

    fn off_device_state(&self) -> Option<&dyn model::OffDeviceState> {
        self.streamed.as_ref().map(|_| self as &dyn model::OffDeviceState)
    }

    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.set_tokens(tokens);
        Some(self.forward_logits())
    }

    fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            self.cfg.param_list().into_iter().map(|(name, numel)| (name.clone(), vec![numel as u64], self.read_weight(&name))).collect();
        let card = checkpoint::st::ModelCard::new("brain/spectrallm", "spectrallm");
        checkpoint::save_carded(path, self.cfg.to_json(), &tensors, &card);
    }
}

impl model::OffDeviceState for SpectralLm {
    fn commit(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let stream = self.streamed.as_ref().ok_or_else(|| std::io::Error::other("spectrallm: no streamed state to commit"))?;
        self.gpu.flush();
        stream.commit(&self.gpu, dir, step)
    }

    fn restore(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let stream = self.streamed.as_ref().ok_or_else(|| std::io::Error::other("spectrallm: no streamed state to restore"))?;
        self.gpu.flush();
        stream.restore(&self.gpu, dir, step)
    }
}

impl model::ModelConfig for SpectralConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        SpectralConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        SpectralConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        SpectralConfig::from_json(v).expect("spectrallm config")
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.vocab = vocab;
        self.block_size = block_size;
        self
    }
}
