// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recurrent-depth execution - a prelude, a
// shared core re-run with the input re-injected, and a coda - for its clients.
// If your team needs expertise in running a prelude/core/coda decoder or in
// proving it matches its reference then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Huginn's execution graph (forward).
//!
//! Every block is a sandwich with four norms:
//!
//! ```text
//! x = norm_2(attn(norm_1(x)) + x)
//! x = norm_4(mlp(norm_3(x)) + x)
//! ```
//!
//! The attention is a fused QKV projection with a per-head Q/K bias and
//! interleaved-pair RoPE (the `rope_train` kernel, not the half-split
//! `rope_base`); the MLP is a fused gate/up FC into SiLU(gate) * up.
//!
//! The same block is composed three ways, in this order:
//!
//! 1. prelude blocks over `emb * sqrt(width)`; their output replaces the
//!    embeddings for the rest of the run;
//! 2. `num_steps` repetitions of `x = adapter(concat(x, prelude_out))`
//!    followed by the recurrent core blocks (only these weights repeat);
//! 3. the final norm, the coda blocks, the final norm again, and the tied
//!    head.
//!
//! The starting latent state is the caller's (`set_state`): the reference's
//! own `like-init` state is a random draw, so a parity run hands it the same
//! state the golden used, and generation draws it once per request.

use std::cell::Cell;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use paramstore::ParamStore;

use crate::config::HuginnConfig;

const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const ROPE_TRAIN: usize = 3;
const GQA_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const GQA_APPLY: usize = 6;
const SILU_MUL: usize = 7;
const ADD2: usize = 8;
const BIAS_ADD: usize = 9;
const COPY_COLS: usize = 10;
const SCALE_ROW: usize = 11;
const ROPE_AT: usize = 12;
const KV_APPEND: usize = 13;
const ATTN_DECODE_SCORES: usize = 14;
const DECODE_SOFTMAX: usize = 15;
const ATTN_DECODE_APPLY: usize = 16;

// Slots the decode engine composes its own graph from.
pub(crate) const EMBED_ID: usize = EMBED;
pub(crate) const MATMUL_ID: usize = MATMUL;
pub(crate) const ADD2_ID: usize = ADD2;
pub(crate) const BIAS_ADD_ID: usize = BIAS_ADD;
pub(crate) const COPY_COLS_ID: usize = COPY_COLS;
pub(crate) const ROPE_AT_ID: usize = ROPE_AT;
pub(crate) const SILU_MUL_ID: usize = SILU_MUL;
pub(crate) const SCALE_ROW_ID: usize = SCALE_ROW;
pub(crate) const KV_APPEND_ID: usize = KV_APPEND;
pub(crate) const ATTN_DECODE_SCORES_ID: usize = ATTN_DECODE_SCORES;
pub(crate) const DECODE_SOFTMAX_ID: usize = DECODE_SOFTMAX;
pub(crate) const ATTN_DECODE_APPLY_ID: usize = ATTN_DECODE_APPLY;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_train", kernels::ROPE_TRAIN),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
    ("copy_cols", kernels::COPY_COLS),
    ("scale_row", kernels::SCALE_ROW),
    ("rope_at", kernels::ROPE_AT),
    ("kv_append", kernels::KV_APPEND),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
];

pub(crate) fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        rope: ROPE_TRAIN,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: SILU_MUL,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
    }
}

struct Scratch {
    xn1: DeviceBuffer,
    xn2: DeviceBuffer,
    xn3: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    tmp: DeviceBuffer,
    fc: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    mh: DeviceBuffer,
    mout: DeviceBuffer,
    concat: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

pub struct Huginn {
    pub gpu: Gpu,
    pub cfg: HuginnConfig,
    pub ps: ParamStore,
    b: u32,
    t: u32,
    tokens: DeviceBuffer,
    /// Embedding buffer pair: the prelude rewrites the embeddings in place.
    embeds: [DeviceBuffer; 2],
    /// The caller's starting state, kept apart from the residual buffers: the
    /// recurrence consumes `h[0]` on its first block, so a second `forward`
    /// has to restore this rather than start from a block output.
    init_state: DeviceBuffer,
    /// Residual-stream pair the recurrence ping-pongs through.
    h: [DeviceBuffer; 2],
    /// Residual pair for the coda.
    coda: [DeviceBuffer; 2],
    /// The normed state after the last recurrent step, before the coda.
    latent: DeviceBuffer,
    logits: DeviceBuffer,
    scale: DeviceBuffer,
    /// Buffer index holding the last recurrence's state, for `read_latent`.
    state_buf: Cell<usize>,
    scratch: Scratch,
}

impl Huginn {
    pub fn new(cfg: HuginnConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Huginn {
        cfg.validate().expect("huginn config");
        assert!(t <= cfg.block_size, "sequence {t} exceeds positional capacity {}", cfg.block_size);
        assert!(b > 0 && t > 0);
        let gpu = Gpu::new(PIPELINES);
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let nh = cfg.n_heads as u64;
        let st = |x: u64| gpu.storage(x);
        let tokens = gpu.buffer("huginn_tokens", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let scale = gpu.buffer("huginn_emb_scale", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        gpu.write_f32(&scale, &vec![cfg.emb_scale; n as usize]);
        let logits = st(n * cfg.vocab as u64);
        let embeds = [st(n * d), st(n * d)];
        let init_state = st(n * d);
        let h = [st(n * d), st(n * d)];
        let coda = [st(n * d), st(n * d)];
        let latent = st(n * d);
        let scratch = Scratch {
            xn1: st(n * d),
            xn2: st(n * d),
            xn3: st(n * d),
            qkv: st(n * 3 * d),
            q: st(n * d),
            k: st(n * d),
            v: st(n * d),
            ctx: st(n * d),
            proj: st(n * d),
            tmp: st(n * d),
            fc: st(n * 2 * ff),
            gate: st(n * ff),
            up: st(n * ff),
            mh: st(n * ff),
            mout: st(n * d),
            concat: st(n * 2 * d),
            scores: st(nh * n * n),
            probs: st(nh * n * n),
        };
        Huginn {
            gpu,
            cfg,
            ps,
            b,
            t,
            tokens,
            embeds,
            init_state,
            h,
            coda,
            latent,
            logits,
            scale,
            state_buf: Cell::new(0),
            scratch,
        }
    }

    pub fn set_tokens(&self, tokens: &[u32]) {
        assert_eq!(tokens.len(), (self.b * self.t) as usize);
        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));
    }

    /// The starting latent state of the recurrence, `[b*t, width]` row-major.
    pub fn set_state(&self, state: &[f32]) {
        assert_eq!(state.len(), (self.b * self.t) as usize * self.cfg.d_model as usize);
        self.gpu.write_f32(&self.init_state, state);
    }

    fn rows(&self) -> u32 {
        self.b * self.t
    }

    /// `out = x @ wᵀ` for `x:[m, k]`, `w:[n, k]`, `out:[m, n]`.
    fn linear(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, w: &str, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
        steps.push(self.gpu.step(MATMUL, &[x, self.ps.w(w), out], &[m, k, n], m * n));
    }

    /// Copy `cnt` columns starting at `src_off` of each of `rows` rows from a
    /// `src_stride`-wide matrix into a `dst_stride`-wide one.
    fn copy_cols(&self, steps: &mut Vec<Step>, src: &DeviceBuffer, src_stride: u32, src_off: u32, dst: &DeviceBuffer, dst_stride: u32, dst_off: u32, cnt: u32) {
        let rows = self.rows();
        steps.push(self.gpu.step(COPY_COLS, &[src, dst], &[rows, cnt, src_stride, src_off, dst_stride, dst_off], rows * cnt));
    }

    fn rope(&self, steps: &mut Vec<Step>, buf: &DeviceBuffer, n_heads: u32, head_dim: u32) {
        let rows = self.rows();
        steps.push(self.gpu.step(
            ROPE_TRAIN,
            &[buf],
            &[rows, n_heads, head_dim, n_heads * head_dim, 0, self.t, gpu_core::f(self.cfg.rope_theta)],
            rows * n_heads * (head_dim / 2),
        ));
    }

    /// One sandwich block: `name` is `prelude.{i}` / `core.{i}` / `coda.{i}`.
    fn block_steps(&self, steps: &mut Vec<Step>, name: &str, x_in: &DeviceBuffer, x_out: &DeviceBuffer) {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (nh, hd) = (cfg.n_heads, cfg.head_dim);
        let s = &self.scratch;
        let w = |leaf: &str| self.ps.w(&format!("{name}.{leaf}"));
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nh, head_dim: hd };

        steps.push(block::rmsnorm_fwd(&self.gpu, &k, x_in, w("ln1.weight"), &s.xn1, d, n, cfg.norm_eps));
        self.linear(steps, &s.xn1, &format!("{name}.attn.wqkv.weight"), &s.qkv, n, d, 3 * d);
        self.copy_cols(steps, &s.qkv, 3 * d, 0, &s.q, d, 0, d);
        self.copy_cols(steps, &s.qkv, 3 * d, d, &s.k, d, 0, d);
        self.copy_cols(steps, &s.qkv, 3 * d, 2 * d, &s.v, d, 0, d);
        // Per-head Q/K bias (the checkpoint's `[2, 1, heads, dim]` tensor,
        // imported as its two halves).
        for (buf, bias) in [(&s.q, "attn.q_bias"), (&s.k, "attn.k_bias")] {
            steps.push(self.gpu.step(BIAS_ADD, &[buf, w(bias)], &[n, d], n * d));
        }
        self.rope(steps, &s.q, nh, hd);
        self.rope(steps, &s.k, nh, hd);
        steps.extend(block::gqa_fwd(&self.gpu, &k, &ga, &s.q, &s.k, &s.v, &s.scores, &s.probs, &s.ctx));
        self.linear(steps, &s.ctx, &format!("{name}.attn.wo.weight"), &s.proj, n, d, d);
        steps.push(self.gpu.step(ADD2, &[&s.proj, x_in, &s.tmp], &[n * d], n * d));
        steps.push(block::rmsnorm_fwd(&self.gpu, &k, &s.tmp, w("ln2.weight"), &s.xn2, d, n, cfg.norm_eps));
        // Fully connected: norm_3, the fused gate/up FC split into its halves,
        // SiLU(gate) * up, projection, residual, norm_4.
        steps.push(block::rmsnorm_fwd(&self.gpu, &k, &s.xn2, w("ln3.weight"), &s.xn3, d, n, cfg.norm_eps));
        self.linear(steps, &s.xn3, &format!("{name}.mlp.fc.weight"), &s.fc, n, d, 2 * ff);
        self.copy_cols(steps, &s.fc, 2 * ff, 0, &s.gate, ff, 0, ff);
        self.copy_cols(steps, &s.fc, 2 * ff, ff, &s.up, ff, 0, ff);
        steps.push(block::swiglu_fwd(&self.gpu, &k, &s.gate, &s.up, &s.mh, n * ff));
        self.linear(steps, &s.mh, &format!("{name}.mlp.proj.weight"), &s.mout, n, ff, d);
        steps.push(self.gpu.step(ADD2, &[&s.mout, &s.xn2, &s.tmp], &[n * d], n * d));
        steps.push(block::rmsnorm_fwd(&self.gpu, &k, &s.tmp, w("ln4.weight"), x_out, d, n, cfg.norm_eps));
    }

    /// Record embed → prelude → recurrence → norm → coda → norm → head.
    fn record(&self, steps: u32) -> Vec<Step> {
        let cfg = &self.cfg;
        let k = kernel_ids();
        let n = self.rows();
        let d = cfg.d_model;
        let mut g: Vec<Step> = Vec::new();

        g.push(self.gpu.step(EMBED, &[&self.tokens, self.ps.w("tok.weight"), &self.embeds[0]], &[d, n], n * d));
        g.push(self.gpu.step(SCALE_ROW, &[&self.embeds[0], &self.scale, &self.embeds[1]], &[n * d, d], n * d));
        let mut cur = 1usize;
        for i in 0..cfg.n_prelude {
            self.block_steps(&mut g, &format!("prelude.{i}"), &self.embeds[cur], &self.embeds[1 - cur]);
            cur = 1 - cur;
        }
        // The prelude's output, not the raw embedding, is what every recurrent
        // step re-injects.
        let inject = &self.embeds[cur];

        // Recurrence. `state` holds the running latent; the injection lands in
        // the other residual buffer, which block 0 consumes, so no state copy
        // is ever needed after the initial one.
        let mut state = 0usize;
        self.copy_cols(&mut g, &self.init_state, d, 0, &self.h[0], d, 0, d);
        for _ in 0..steps.max(1) {
            let mid = 1 - state;
            self.copy_cols(&mut g, &self.h[state], d, 0, &self.scratch.concat, 2 * d, 0, d);
            self.copy_cols(&mut g, inject, d, 0, &self.scratch.concat, 2 * d, d, d);
            self.linear(&mut g, &self.scratch.concat, "adapter.weight", &self.h[mid], n, 2 * d, d);
            self.block_steps(&mut g, "core.0", &self.h[mid], &self.h[state]);
            for i in 1..cfg.n_recurrent {
                self.block_steps(&mut g, &format!("core.{i}"), &self.h[state], &self.h[1 - state]);
                state = 1 - state;
            }
        }
        self.state_buf.set(state);

        // Final norm, then the coda from the normed state, then the norm again.
        g.push(block::rmsnorm_fwd(&self.gpu, &k, &self.h[state], self.ps.w("ln_f.weight"), &self.latent, d, n, cfg.norm_eps));
        let mut cur = 0usize;
        for i in 0..cfg.n_coda {
            let x_in = if i == 0 { &self.latent } else { &self.coda[cur] };
            self.block_steps(&mut g, &format!("coda.{i}"), x_in, &self.coda[1 - cur]);
            cur = 1 - cur;
        }
        g.push(block::rmsnorm_fwd(&self.gpu, &k, &self.coda[cur], self.ps.w("ln_f.weight"), &self.coda[1 - cur], d, n, cfg.norm_eps));
        cur = 1 - cur;
        self.linear(&mut g, &self.coda[cur], "tok.weight", &self.logits, n, d, cfg.vocab);
        g
    }

    /// Logits after `steps` recurrent iterations, `[vocab]` of the last token.
    pub fn forward(&self, steps: u32) -> Vec<f32> {
        self.forward_rows(steps).split_off((self.rows() as usize - 1) * self.cfg.vocab as usize)
    }

    /// Every row's logits, `[b*t, vocab]`.
    pub fn forward_rows(&self, steps: u32) -> Vec<f32> {
        let g = self.record(steps);
        self.gpu.submit(&[], &g);
        self.gpu.read(&self.logits, self.rows() as usize * self.cfg.vocab as usize)
    }

    /// The normed latent state after the last recurrent step (`ln_f` applied,
    /// coda not), `[b*t, width]`.
    pub fn read_latent(&self) -> Vec<f32> {
        self.gpu.read(&self.latent, self.rows() as usize * self.cfg.d_model as usize)
    }

    /// The raw (norm-free) state after the last recurrent step.
    pub fn read_state(&self) -> Vec<f32> {
        self.gpu.read(&self.h[self.state_buf.get()], self.rows() as usize * self.cfg.d_model as usize)
    }
}
